"""CPU tests for the paired-plan compiler (never launches a process)."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from runtime import shared_lru_plan as planner
from runtime import study_config as sc

ROOT = Path(__file__).parent
EXAMPLE = ROOT / "fixtures" / "shared_lru_plan_spec.example.json"
SMALL = {
    "schema": planner.SPEC_SCHEMA, "seed": 5,
    "split": {"training_sha256": "a" * 64, "heldout_sha256": "b" * 64},
    "geometry": {"tp_size": 4, "first_layer": 0, "layers": 4, "hot_rows": 32, "page": 4, "top_k_blocks": 4,
                 "block_size": 4, "head_dim": 4, "dtype_bytes": 2, "kv_heads_per_rank": 1, "slots": 8,
                 "plane_rows": 292},
    "static_quota_rows": {"0": 16, "1": 16, "2": 48, "3": 48},
    "floor_rows": {"2": 24, "3": 24},
    "bypass_layers": [0],
    "pressure_points": [{"name": "small", "input_tokens": [64, 128], "real_batch": 4}],
    "repeats": 3,
    "upstream_args": ["--model-path", "$MODEL_DIR", "--tp-size", "4", "--enable-hisparse", "--hisparse-config",
                      '{"top_k":16,"device_buffer_size":32}', "--disable-overlap-schedule", "--disable-radix-cache"],
    "sm80_bf16_projections": False,
}


def run(spec, out, *extra):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(spec, handle)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
        code = planner.main(["--spec", handle.name, "--output-dir", str(out), *extra])
    Path(handle.name).unlink()
    return code, err.getvalue()


class PairedPlan(unittest.TestCase):
    def test_example_spec_skips_unfrozen_arms_and_alternates_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, err = run(json.loads(EXAMPLE.read_text()), tmp)
            self.assertEqual(code, 0, err)
            plan = json.loads((Path(tmp) / "plan.json").read_text())
            self.assertEqual([a["arm"] for a in plan["arms"]],
                             ["stock", "isolated_arena", "shared_lru", "shared_fifo", "shared_lru_owner_summary",
                              "shared_lru_touched_bitmap", "shared_lru_optimized", "isolated_arena_optimized"])
            self.assertEqual(set(plan["skipped"]), {"isolated_static", "shared_soft_floor", "shared_bypass"})
            self.assertTrue(plan["split_placeholder"])
            self.assertEqual(len(plan["pairs"]), 9)
            self.assertEqual(len(plan["schedule"]), 9 * 3 * 3)
            orders = [run_["order"] for run_ in plan["schedule"][:3]]
            self.assertEqual(orders, [["stock", "isolated_arena"], ["isolated_arena", "stock"],
                                      ["stock", "isolated_arena"]])
            shared = next(a for a in plan["arms"] if a["arm"] == "shared_lru")
            self.assertEqual(shared["ledger"]["per_request"]["history_payload_bytes"], 120 * 2**20)
            self.assertGreater(shared["capacity"]["requests_charged_for_metadata"], 0)
            self.assertEqual(plan["launch"]["stock"][:4], ["python", "-m", "runtime.launch", "--hisparse-study-config"])
            for arm in plan["arms"]:
                self.assertEqual(sc.read(Path(tmp) / arm["config"])[0].digest, arm["config_sha256"])

    def test_every_pair_changes_exactly_the_declared_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, err = run(SMALL, tmp)
            self.assertEqual(code, 0, err)
            plan = json.loads((Path(tmp) / "plan.json").read_text())
        self.assertEqual(plan["skipped"], {})
        changed = {(p["control"], p["candidate"]): p["changed_fields"] for p in plan["pairs"]}
        self.assertEqual(changed[("isolated_arena", "shared_lru")], ["hot_pool.mode"])
        self.assertEqual(changed[("stock", "isolated_arena")], ["hot_pool.layout"])
        self.assertEqual(changed[("shared_lru", "shared_bypass")], ["hot_pool.admission", "hot_pool.bypass_layers"])
        self.assertEqual(len(plan["arms"]), 11)
        both = ["features.owner_summary", "features.touched_bitmap"]
        self.assertEqual(changed[("shared_lru", "shared_lru_owner_summary")], ["features.owner_summary"])
        self.assertEqual(changed[("shared_lru", "shared_lru_touched_bitmap")], ["features.touched_bitmap"])
        self.assertEqual(changed[("shared_lru", "shared_lru_optimized")], both)
        self.assertEqual(changed[("isolated_arena", "isolated_arena_optimized")], both)
        self.assertEqual(changed[("isolated_arena_optimized", "shared_lru_optimized")], ["hot_pool.mode"])
        self.assertEqual(changed[("stock", "shared_lru_optimized")], both + ["hot_pool.layout", "hot_pool.mode"])
        arms = {a["arm"]: a for a in plan["arms"]}
        summary_bytes = (8 + 1) * 4 * 8              # (slots + template) x ceil(4 x 32 / 32) words x 8 B
        self.assertEqual(arms["shared_lru_optimized"]["capacity"]["arena_metadata_bytes"]
                         - arms["shared_lru"]["capacity"]["arena_metadata_bytes"], summary_bytes)
        self.assertEqual(arms["shared_lru_touched_bitmap"]["capacity"]["arena_metadata_bytes"],
                         arms["shared_lru"]["capacity"]["arena_metadata_bytes"])       # shared memory only
        self.assertEqual(arms["shared_lru_optimized"]["capabilities"]["features"],
                         {"owner_summary": True, "touched_bitmap": True})

    def test_invalid_specs_ledgers_and_selection_mismatch_fail(self):
        bad_quota = copy.deepcopy(SMALL)
        bad_quota["static_quota_rows"]["3"] = 40          # no longer partitions 4 x 32 rows
        two_repeats = dict(SMALL, repeats=2)
        same_split = dict(SMALL, split={"training_sha256": "a" * 64, "heldout_sha256": "a" * 64})
        multi_stream = dict(SMALL, upstream_args=SMALL["upstream_args"][:-2])
        for spec in (bad_quota, two_repeats, same_split, multi_stream):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(spec=spec is bad_quota):
                self.assertEqual(run(spec, tmp)[0], 1)
        with tempfile.TemporaryDirectory() as tmp:
            results = []
            for name, digest in (("a.json", "1" * 64), ("b.json", "2" * 64)):
                path = Path(tmp) / name
                path.write_text(json.dumps({"selection_sha256": digest}))
                results.append(str(path))
            self.assertEqual(run(SMALL, Path(tmp) / "plan", "--replay-results", *results)[0], 1)
            self.assertEqual(run(SMALL, Path(tmp) / "ok", "--replay-results", results[0], results[0])[0], 0)
            self.assertEqual(run(SMALL, Path(tmp) / "ok")[0], 1)       # never overwrites a plan

    def test_changed_fields_helper(self):
        stock = sc.StudyConfig()
        shared = sc.parse({"schema": sc.SCHEMA, "hot_pool": {"mode": "shared", "layout": "arena"}})
        self.assertEqual(planner.changed_fields(stock, shared), ("hot_pool.layout", "hot_pool.mode"))
        self.assertEqual(planner.changed_fields(shared, shared), ())


if __name__ == "__main__":
    unittest.main()
