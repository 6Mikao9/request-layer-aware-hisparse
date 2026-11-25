"""CPU tests for the combined-study plan compiler (arms A–E; never launches a process)."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from runtime import combined_calibration as calibration
from runtime import combined_plan as planner
from runtime import study_config as sc

ROOT = Path(__file__).parent
EXAMPLE = ROOT / "fixtures" / "combined_plan_spec.example.json"
RANDOM = [{"name": f"random_s{seed}", "selection": "random", "seed": seed, "offload_layers": layers}
          for seed, layers in ((1, [0, 2]), (2, [1, 3]), (3, [0, 3]))]
MEASUREMENT = {"max_new_tokens": 96, "warmup_requests": 1, "warmup_steps": 8, "min_window_steps": 32,
               "min_decode_log_samples": 4, "functional_requests": 2, "functional_max_new_tokens": 8,
               "ttft_slo_s": 5, "tpot_slo_s": 0.5, "server_start_timeout_s": 30, "request_timeout_s": 30}
ACCEPTANCE = {"candidate": "calibrated_shared_arena", "baselines": ["stock", "calibrated_static_quota"],
              "min_gain": 0.05, "tpot_p95_regression_max": 0.02, "ci_level": 0.95, "min_pressure_points": 2,
              "min_pairs": 3, "quality_mismatch_over_stock_max": 0.0}
SMALL = {
    "schema": planner.SPEC_SCHEMA, "seed": 5,
    "split": {"training_sha256": "a" * 64, "heldout_sha256": "b" * 64},
    "geometry": {"tp_size": 4, "first_layer": 0, "layers": 4, "dense_layers": 0, "hot_rows": 32, "page": 4,
                 "top_k_blocks": 4, "block_size": 4, "head_dim": 4, "dtype_bytes": 2, "kv_heads_per_rank": 1,
                 "slots": 8, "plane_rows": 2308, "host_to_device_ratio": 2, "index_bytes_per_token": 16},
    "offload_lists": RANDOM + [{"name": "calibrated", "selection": "calibrated", "offload_layers": [2, 3],
                                "calibration": {"training_sha256": "a" * 64, "list_sha256": "d" * 64}}],
    "static_quota_rows": {"calibrated": {"2": 16, "3": 48}},
    "resident_direct_lists": ["calibrated"],
    "pressure_points": [{"name": "small", "input_tokens": [64, 128], "real_batch": 4},
                        {"name": "too_big", "input_tokens": [64, 128], "real_batch": 60}],
    "repeats": 3,
    "measurement": MEASUREMENT,
    "acceptance": ACCEPTANCE,
    "upstream_args": ["--model-path", "$MODEL_DIR", "--tp-size", "4", "--enable-hisparse", "--hisparse-config",
                      '{"top_k":16,"device_buffer_size":32}', "--disable-overlap-schedule", "--disable-radix-cache",
                      "--decode-log-interval", "4"],
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
    def test_example_spec_holds_the_training_calibration_and_compiles_the_factorial(self):
        spec = json.loads(EXAMPLE.read_text())
        source = ROOT.parent / "data" / "a100-minimax-20261122.json"
        frozen = calibration.calibrate(source.read_bytes(), source_name="data/a100-minimax-20261122.json",
                                       training=("8-16", "64-128"), heldout=("16-32",), sizes=(40,),
                                       hot_rows=4096, selected=2048)
        self.assertEqual(json.loads((ROOT / "fixtures" / "combined_calibration.example.json").read_text()), frozen)
        self.assertEqual((spec["offload_lists"], spec["static_quota_rows"], spec["split"]["training_sha256"]),
                         (frozen["offload_lists"], frozen["static_quota_rows"], frozen["source"]["sha256"]))
        with tempfile.TemporaryDirectory() as tmp:
            code, err, plan = run(spec, tmp)
            self.assertEqual(code, 0, err)
            configs = {arm["arm"]: sc.read(Path(tmp) / arm["config"])[0] for arm in plan["arms"]}
            for arm in plan["arms"]:
                self.assertEqual(configs[arm["arm"]].digest, arm["config_sha256"])
        arms = {a["arm"]: a for a in plan["arms"]}
        self.assertEqual((len(arms), plan["skipped"]), (3 + 5 * 4 + 1, {}))
        self.assertEqual(arms["stock"]["capacity"]["requests_upper_bound"], 93)
        self.assertEqual(arms["all_shared_arena"]["capacity"]["requests_upper_bound"], 91)
        fixed = arms["calibrated_t40_shared_arena"]
        self.assertEqual(len(fixed["offload_layers"]), 40)
        self.assertEqual(fixed["ledger"]["per_request"]["history_payload_bytes"], 40 * 4096 * 512)
        self.assertLess(fixed["capacity"]["device_tokens"], arms["stock"]["capacity"]["device_tokens"])
        self.assertEqual({name for name, c in configs.items() if c.uses_fixed_layers and not c.uses_resident_direct},
                         {"calibrated_t40_shared_resident_resolver"})
        kinds = {p["kind"] for p in plan["pairs"]}
        self.assertEqual(kinds, set(planner.PRIORITY))
        accepted = {(p["control"], p["candidate"]) for p in plan["pairs"] if p["acceptance"]}
        self.assertEqual(accepted, {(base, "calibrated_t40_shared_arena") for base in spec["acceptance"]["baselines"]})
        self.assertEqual(plan["schedule"][0]["kind"], "aa_noise")
        self.assertEqual({e["priority"] for e in plan["schedule"][:9 + 45]}, {0, 1})
        self.assertTrue(plan["split_placeholder"])
        for cmd in plan["launch"].values():
            self.assertNotIn("--max-total-tokens", cmd)
            self.assertEqual(cmd[cmd.index("--decode-log-interval") + 1], "8")

    def test_pairs_priorities_rounds_and_infeasible_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, err, plan = run(SMALL, tmp)
            self.assertEqual(code, 0, err)
        self.assertEqual((len(plan["arms"]), len(plan["pairs"]), plan["skipped"]), (17, 33, {
            f"{item['name']}_static_quota": "no frozen static quota for this list (freeze it from training first)"
            for item in RANDOM}))
        changed = {(p["control"], p["candidate"]): (p["kind"], p["changed_fields"], p["acceptance"])
                   for p in plan["pairs"]}
        self.assertEqual(changed[("stock", "stock")], ("aa_noise", [], False))
        self.assertEqual(changed[("calibrated_isolated_arena", "calibrated_static_quota")],
                         ("static_quota", ["hot_pool.quota", "hot_pool.quota_rows"], False))
        self.assertEqual(changed[("calibrated_static_quota", "calibrated_shared_arena")],
                         ("sharing_vs_static", ["hot_pool.mode", "hot_pool.quota", "hot_pool.quota_rows"], True))
        self.assertEqual(changed[("calibrated_shared_resident_resolver", "calibrated_shared_arena")],
                         ("resident_direct", ["features.resident_direct"], False))
        self.assertEqual(changed[("random_s2_shared_arena", "calibrated_shared_arena")][:2],
                         ("selection_shared_arena", ["placement.calibration", "placement.offload_layers",
                                                     "placement.seed", "placement.selection"]))
        self.assertEqual(changed[("stock", "calibrated_stock_layout")][1],
                         ["features.resident_direct", "placement.calibration", "placement.mode",
                          "placement.offload_layers", "placement.selection"])
        self.assertEqual(changed[("stock", "calibrated_shared_arena")][::2], ("acceptance", True))
        arms = {a["arm"]: a for a in plan["arms"]}
        schedule = plan["schedule"]
        self.assertEqual([e["priority"] for e in schedule], sorted(e["priority"] for e in schedule))
        for entry in schedule:
            self.assertTrue(all(arms[name]["estimated_feasible"][entry["point"]] for name in entry["order"]))
            self.assertEqual(entry["input_batch"], entry["repeat"])
            self.assertEqual(entry["order"], [entry["control"], entry["candidate"]][::1 if entry["repeat"] % 2 == 0
                                                                                   else -1])
        self.assertEqual(len(schedule), 33 * 3 + 3, "every pair at the small point, only stock A/A fits B=60")
        self.assertEqual({e["point"] for e in plan["infeasible"]}, {"too_big"})
        self.assertEqual(len(plan["infeasible"]), 32)

    def test_invalid_specs_fail_without_a_plan(self):
        bad_quota = copy.deepcopy(SMALL)
        bad_quota["static_quota_rows"]["calibrated"]["3"] = 40       # no longer partitions 2 x 32 rows
        unknown_layer = copy.deepcopy(SMALL)
        unknown_layer["offload_lists"][0]["offload_layers"] = [0, 9]
        foreign = copy.deepcopy(SMALL)
        foreign["offload_lists"][-1]["calibration"]["training_sha256"] = "c" * 64
        typo = copy.deepcopy(SMALL)
        typo["offload_lists"][0]["sede"] = 1
        no_log = dict(SMALL, upstream_args=SMALL["upstream_args"][:-2])
        cases = dict(
            quota=bad_quota, layer=unknown_layer, foreign_training=foreign, typo=typo, no_decode_log=no_log,
            randoms=dict(SMALL, offload_lists=SMALL["offload_lists"][1:]),
            duplicate=dict(SMALL, offload_lists=SMALL["offload_lists"] + [dict(RANDOM[0])]),
            sized=dict(SMALL, upstream_args=SMALL["upstream_args"] + ["--max-total-tokens", "4096"]),
            empty=dict(SMALL, offload_lists=[dict(RANDOM[0], offload_layers=[])]),
            one_point=dict(SMALL, pressure_points=SMALL["pressure_points"][:1]),
            unknown_arm=dict(SMALL, acceptance=dict(ACCEPTANCE, baselines=["stock", "nonexistent"])),
            pairs=dict(SMALL, acceptance=dict(ACCEPTANCE, min_pairs=4)),
            window=dict(SMALL, measurement=dict(MEASUREMENT, max_new_tokens=40)),
            resolver_list=dict(SMALL, resident_direct_lists=["missing"]))
        for name, spec in cases.items():
            with tempfile.TemporaryDirectory() as tmp, self.subTest(case=name):
                self.assertEqual(run(spec, tmp)[0], 1)
                self.assertFalse((Path(tmp) / "plan.json").exists())
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(run(SMALL, tmp)[0], 0)
            self.assertEqual(run(SMALL, tmp)[0], 1)                    # never overwrites a plan


if __name__ == "__main__":
    unittest.main()
