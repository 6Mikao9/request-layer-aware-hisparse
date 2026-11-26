"""CPU tests of the training-only calibration that freezes offload lists and static quotas."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

from runtime import combined_calibration as cal

# Six layers, two ranks. Rank 1 is the slow rank of layers 1 and 4; layer 5 never misses.
MISSES = {0: (10, 20), 1: (50, 90), 2: (30, 30), 3: (70, 10), 4: (5, 80), 5: (0, 0)}


def source(heldout_counts="not even a list"):
    bucket = lambda label, scale: {"input_ki_tokens": label, "actual_batch": 4, "counts_on_layers": [
        {"layer": layer, "selected": [1000 * scale, 1000 * scale], "misses": [a * scale, b * scale]}
        for layer, (a, b) in MISSES.items()]}
    held = {"input_ki_tokens": "h", "counts_on_layers": heldout_counts}
    return json.dumps({"model": "synthetic", "tensor_parallel": 2,
                       "calibration": [bucket("a", 1), held, bucket("b", 3)]}).encode()


def calibrate(raw=None, **kwargs):
    args = dict(source_name="synthetic.json", training=("a", "b"), heldout=("h",), sizes=(2,), hot_rows=10,
                selected=4)
    return cal.calibrate(raw or source(), **(args | kwargs))


class TrainingOnlyLists(unittest.TestCase):
    def test_costs_lists_and_digests_come_from_training_buckets_only(self):
        result = calibrate()
        cost = {int(k): v["max_rank_misses"] / v["selected"] for k, v in result["layer_cost"].items()}
        self.assertEqual(cost, {0: 0.02, 1: 0.09, 2: 0.03, 3: 0.07, 4: 0.08, 5: 0.0})
        lists = {item["name"]: item for item in result["offload_lists"]}
        self.assertEqual(lists["calibrated_t2"]["offload_layers"], [0, 5])
        self.assertEqual(lists["reverse_t2"]["offload_layers"], [1, 4])
        for seed in cal.RANDOM_SEEDS:
            self.assertEqual(lists[f"random_t2_s{seed}"]["offload_layers"],
                             sorted(random.Random(seed).sample(range(6), 2)))
            self.assertEqual((lists[f"random_t2_s{seed}"]["selection"], lists[f"random_t2_s{seed}"]["seed"]),
                             ("random", seed))
        digest = hashlib.sha256(source()).hexdigest()
        self.assertEqual(lists["calibrated_t2"]["calibration"],
                         dict(training_sha256=digest, list_sha256=hashlib.sha256(b"[0,5]").hexdigest()))
        self.assertEqual(result["split"], dict(training_buckets=["a", "b"], heldout_buckets=["h"],
                                               heldout_counts_read=False))
        other = calibrate(source(heldout_counts=[]))
        self.assertEqual([item["offload_layers"] for item in other["offload_lists"]],
                         [item["offload_layers"] for item in result["offload_lists"]])
        self.assertEqual(other["static_quota_rows"], result["static_quota_rows"], "held-out contents change nothing")
        self.assertNotEqual(other["source"]["sha256"], digest, "the digest still pins the exact source bytes")

    def test_static_quota_follows_estimated_working_sets_at_equal_bytes(self):
        quota = calibrate()["static_quota_rows"]
        for name, rows in quota.items():
            self.assertEqual(sum(rows.values()), 2 * 10, name)
            self.assertGreaterEqual(min(rows.values()), 4)
        self.assertEqual(quota["reverse_t2"], {"1": 10, "4": 10})        # 10 + 4*0.09 vs 10 + 4*0.08 rounds even
        cost = {1: 0, 2: 10}
        self.assertEqual(cal.static_quota((1, 2), cost, hot_rows=10, selected=8), {1: 8, 2: 12}, "pinned at K")
        self.assertEqual(cal.static_quota((3, 7), {3: 1, 7: 1}, hot_rows=5, selected=1), {3: 5, 7: 5})
        self.assertEqual(cal.static_quota((3, 7, 9), {3: 0, 7: 0, 9: 0}, hot_rows=1, selected=1), {3: 1, 7: 1, 9: 1})

    def test_overlapping_missing_or_malformed_inputs_fail(self):
        bad = json.loads(source())
        bad["calibration"][0]["counts_on_layers"][0]["misses"] = [2000, 0]       # more misses than selections
        cases = [dict(heldout=("a",)), dict(training=("a", "z")), dict(sizes=(6,)), dict(sizes=(0,)),
                 dict(seeds=(1, 2)), dict(hot_rows=3, selected=4), dict(raw=json.dumps(bad).encode()),
                 dict(raw=source().replace(b'"layer": 5', b'"layer": 4'))]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                calibrate(**case)

    def test_cli_writes_new_files_and_updates_only_the_frozen_spec_fields(self):
        spec = {"schema": "combined-plan-spec-v1", "seed": 1, "split": {"training_sha256": "0" * 64,
                                                                         "heldout_sha256": "b" * 64},
                "offload_lists": [], "static_quota_rows": None, "repeats": 3}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "source.json").write_bytes(source())
            (tmp / "spec.json").write_text(json.dumps(spec))
            argv = ["--source", str(tmp / "source.json"), "--training-bucket", "a", "--training-bucket", "b",
                    "--heldout-bucket", "h", "--offload-layers", "2", "--hot-rows", "10", "--selected", "4",
                    "--output", str(tmp / "out.json"), "--spec", str(tmp / "spec.json"),
                    "--spec-output", str(tmp / "new.json")]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cal.main(argv), 0)
            result, updated = (json.loads((tmp / name).read_text()) for name in ("out.json", "new.json"))
            self.assertEqual(updated["offload_lists"], result["offload_lists"])
            self.assertEqual(updated["split"], {"training_sha256": result["source"]["sha256"],
                                                "heldout_sha256": "b" * 64})
            self.assertEqual((updated["seed"], updated["repeats"]), (1, 3))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cal.main(argv)                                           # never overwrites
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cal.main(argv[:-4] + ["--output", str(tmp / "other.json"), "--spec", str(tmp / "spec.json")])


if __name__ == "__main__":
    unittest.main()
