"""CPU tests for the combined-study plan compiler (arms A–E; never launches a process)."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from runtime import combined_plan as planner
from runtime import study_config as sc

ROOT = Path(__file__).parent
EXAMPLE = ROOT / "fixtures" / "combined_plan_spec.example.json"
RANDOM = [{"name": f"random_s{seed}", "selection": "random", "seed": seed, "offload_layers": layers}
          for seed, layers in ((1, [0, 2]), (2, [1, 3]), (3, [0, 3]))]
SMALL = {
    "schema": planner.SPEC_SCHEMA, "seed": 5,
    "split": {"training_sha256": "a" * 64, "heldout_sha256": "b" * 64},
    "geometry": {"tp_size": 4, "first_layer": 0, "layers": 4, "dense_layers": 0, "hot_rows": 32, "page": 4,
                 "top_k_blocks": 4, "block_size": 4, "head_dim": 4, "dtype_bytes": 2, "kv_heads_per_rank": 1,
                 "slots": 8, "plane_rows": 2308, "host_to_device_ratio": 2, "index_bytes_per_token": 16},
    "offload_lists": RANDOM + [{"name": "calibrated", "selection": "calibrated", "offload_layers": [2, 3],
                                "calibration": {"training_sha256": "c" * 64, "list_sha256": "d" * 64}}],
    "static_quota_rows": {"calibrated": {"2": 16, "3": 48}},
    "pressure_points": [{"name": "small", "input_tokens": [64, 128], "real_batch": 4},
                        {"name": "too_big", "input_tokens": [64, 128], "real_batch": 60}],
    "repeats": 3,
    "upstream_args": ["--model-path", "$MODEL_DIR", "--tp-size", "4", "--enable-hisparse", "--hisparse-config",
                      '{"top_k":16,"device_buffer_size":32}', "--disable-overlap-schedule", "--disable-radix-cache"],
    "sm80_bf16_projections": False,
}


def run(spec, out):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(spec, handle)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
        code = planner.main(["--spec", handle.name, "--output-dir", str(out)])
    Path(handle.name).unlink()
    plan = json.loads((Path(out) / "plan.json").read_text()) if code == 0 else None
    return code, err.getvalue(), plan


class CombinedPlan(unittest.TestCase):
    def test_example_spec_compiles_the_factorial_and_keeps_one_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, err, plan = run(json.loads(EXAMPLE.read_text()), tmp)
            self.assertEqual(code, 0, err)
            for arm in plan["arms"]:
                self.assertEqual(sc.read(Path(tmp) / arm["config"])[0].digest, arm["config_sha256"])
        arms = {a["arm"]: a for a in plan["arms"]}
        self.assertEqual(len(arms), 3 + 3 * 3)
        self.assertEqual(set(plan["skipped"]), {"calibrated_selection"} | {f"random_t40_s{i}_static_quota"
                                                                           for i in (1, 2, 3)})
        self.assertEqual(arms["stock"]["capacity"]["requests_upper_bound"], 93)
        self.assertEqual(arms["all_shared_arena"]["capacity"]["requests_upper_bound"], 91)
        fixed = arms["random_t40_s1_shared_arena"]
        self.assertEqual(len(fixed["offload_layers"]), 40)
        self.assertEqual(fixed["ledger"]["per_request"]["history_payload_bytes"], 40 * 4096 * 512)
        self.assertLess(fixed["capacity"]["device_tokens"], arms["stock"]["capacity"]["device_tokens"])
        self.assertLessEqual(fixed["ledger"]["main_kv_bytes"], arms["stock"]["ledger"]["main_kv_bytes"])
        kinds = {p["kind"] for p in plan["pairs"]}
        self.assertEqual(kinds, {"addressing", "sharing", "placement", "placement_borrowing_off",
                                 "placement_borrowing_on"})
        self.assertEqual(plan["interaction"]["random_t40_s2"]["borrowing_on"],
                         ["all_shared_arena", "random_t40_s2_shared_arena"])
        self.assertTrue(plan["split_placeholder"])
        self.assertTrue(all("--max-total-tokens" not in cmd for cmd in plan["launch"].values()))

    def test_static_quota_calibrated_selection_and_infeasible_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, err, plan = run(SMALL, tmp)
            self.assertEqual(code, 0, err)
        self.assertEqual(set(plan["skipped"]), {f"{item['name']}_static_quota" for item in RANDOM})
        self.assertEqual(len(plan["arms"]), 3 + 4 * 3 + 1)
        changed = {(p["control"], p["candidate"]): (p["kind"], p["changed_fields"]) for p in plan["pairs"]}
        self.assertEqual(len(changed), 30)
        self.assertEqual(changed[("calibrated_isolated_arena", "calibrated_static_quota")],
                         ("static_quota", ["hot_pool.quota", "hot_pool.quota_rows"]))
        self.assertEqual(changed[("calibrated_static_quota", "calibrated_shared_arena")][1],
                         ["hot_pool.mode", "hot_pool.quota", "hot_pool.quota_rows"])
        self.assertEqual(changed[("random_s2_shared_arena", "calibrated_shared_arena")],
                         ("selection_shared_arena", ["placement.calibration", "placement.offload_layers",
                                                     "placement.seed", "placement.selection"]))
        self.assertEqual(changed[("stock", "calibrated_stock_layout")][1],
                         ["placement.calibration", "placement.mode", "placement.offload_layers",
                          "placement.selection"])
        self.assertEqual({run_["point"] for run_ in plan["schedule"]}, {"small"})
        self.assertEqual(len(plan["schedule"]), 30 * 3)
        self.assertEqual({entry["point"] for entry in plan["infeasible"]}, {"too_big"})
        arms = {a["arm"]: a for a in plan["arms"]}
        self.assertTrue(arms["stock"]["estimated_feasible"]["too_big"])
        self.assertFalse(arms["calibrated_shared_arena"]["estimated_feasible"]["too_big"])

    def test_invalid_specs_fail_without_a_plan(self):
        bad_quota = copy.deepcopy(SMALL)
        bad_quota["static_quota_rows"]["calibrated"]["3"] = 40       # no longer partitions 2 x 32 rows
        unknown_layer = copy.deepcopy(SMALL)
        unknown_layer["offload_lists"][0]["offload_layers"] = [0, 9]
        two_randoms = dict(SMALL, offload_lists=SMALL["offload_lists"][1:])
        duplicate = dict(SMALL, offload_lists=SMALL["offload_lists"] + [dict(RANDOM[0])])
        typo = copy.deepcopy(SMALL)
        typo["offload_lists"][0]["sede"] = 1
        sized = dict(SMALL, upstream_args=SMALL["upstream_args"] + ["--max-total-tokens", "4096"])
        empty = dict(SMALL, offload_lists=[dict(RANDOM[0], offload_layers=[])])
        for name, spec in (("quota", bad_quota), ("layer", unknown_layer), ("randoms", two_randoms),
                           ("duplicate", duplicate), ("typo", typo), ("sized", sized), ("empty", empty)):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(case=name):
                self.assertEqual(run(spec, tmp)[0], 1)
                self.assertFalse((Path(tmp) / "plan.json").exists())
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(run(SMALL, tmp)[0], 0)
            self.assertEqual(run(SMALL, tmp)[0], 1)                    # never overwrites a plan


if __name__ == "__main__":
    unittest.main()