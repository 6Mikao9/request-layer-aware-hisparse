"""CPU tests for the replay CLI on the tracked fixture (stdlib only)."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from runtime import shared_lru_replay as replay
from runtime import study_config as sc

ROOT = Path(__file__).parent
FIXTURE = ROOT / "fixtures" / "shared_lru_fixture.json"
ARMS = ROOT / "fixtures" / "shared_lru_arms"
EXTRA_ARMS = {
    "shared_fifo": {"mode": "shared", "layout": "arena", "eviction": "fifo"},
    "shared_soft_floor": {"mode": "shared", "layout": "arena", "quota": "soft_floor",
                          "floor_rows": {"0": 24, "1": 24, "2": 32, "3": 32}},
    "isolated_static": {"mode": "isolated", "layout": "arena", "quota": "static",
                        "quota_rows": {"0": 16, "1": 16, "2": 48, "3": 48}},
    "shared_bypass": {"mode": "shared", "layout": "arena", "admission": "bypass_list", "bypass_layers": [0]},
}


def run(config, output):
    with contextlib.redirect_stdout(io.StringIO()):
        code = replay.main(["--fixture", str(FIXTURE), "--config", str(config), "--output", str(output)])
    return code, json.loads(Path(output).read_text()) if code == 0 else None


class ReplayCli(unittest.TestCase):
    def test_tracked_arms_share_selections_and_repeat_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = {}
            for arm in ("stock", "isolated_arena", "shared_arena"):
                code, results[arm] = run(ARMS / f"{arm}.json", Path(tmp) / f"{arm}.json")
                self.assertEqual(code, 0)
                self.assertEqual(results[arm]["status"], "pass")
                self.assertGreater(results[arm]["checks"]["direct_reader_positions"], 10_000)
            digests = {r["selection_sha256"] for r in results.values()}
            self.assertEqual(len(digests), 1, "arms must replay identical selections")
            self.assertEqual(results["stock"]["engine"], "NativeLayerLRU")
            self.assertEqual(results["shared_arena"]["checks"]["kernel_mirror"], "lockstep")
            # Fixture property (synthetic): layers 2-3 cycle more blocks than H holds.
            self.assertGreater(results["shared_arena"]["history_hit_rate"],
                               results["isolated_arena"]["history_hit_rate"] + 0.1)
            code, again = run(ARMS / "shared_arena.json", Path(tmp) / "again.json")
            self.assertEqual(code, 0)
            self.assertEqual((Path(tmp) / "again.json").read_bytes(), (Path(tmp) / "shared_arena.json").read_bytes())

    def test_every_policy_arm_replays_with_mirror_lockstep(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, pool in EXTRA_ARMS.items():
                path = Path(tmp) / f"{name}.json"
                path.write_text(json.dumps({"schema": sc.SCHEMA, "hot_pool": pool}))
                with self.subTest(arm=name):
                    code, result = run(path, Path(tmp) / f"{name}.out.json")
                    self.assertEqual(code, 0)
                    self.assertEqual(result["resolved_config"]["hot_pool"]["mode"], pool["mode"])
            counters = run(Path(tmp) / "shared_bypass.json", Path(tmp) / "bypass2.json")[1]["counters"]["total"]
            self.assertGreater(counters["bypassed"], 0)

    def test_resolver_switches_replay_identical_rows_and_bytes(self):
        """Resolver switches change how rows are found, stored and ordered, never which rows."""
        same = ("selection_sha256", "output_sha256", "device_bytes_sha256", "counters", "history_hit_rate")
        pr8 = {"owner_summary": True, "touched_bitmap": True, "hit_index": False, "incremental_lru": False,
               "compact_owner": False}
        issue6 = {"owner_summary": False, "touched_bitmap": False, "hit_index": True, "incremental_lru": True,
                  "compact_owner": True}
        with tempfile.TemporaryDirectory() as tmp:
            for plain, optimized, features in (("shared_arena", "shared_arena_optimized", pr8),
                                               ("isolated_arena", "isolated_arena_optimized", pr8),
                                               ("shared_arena", "shared_arena_indexed", issue6),
                                               ("isolated_arena", "isolated_arena_indexed", issue6)):
                with self.subTest(arm=optimized):
                    _, a = run(ARMS / f"{plain}.json", Path(tmp) / f"{optimized}.plain.json")
                    code, b = run(ARMS / f"{optimized}.json", Path(tmp) / f"{optimized}.json")
                    self.assertEqual(code, 0)
                    self.assertEqual({k: a[k] for k in same}, {k: b[k] for k in same})
                    self.assertNotEqual(a["config_sha256"], b["config_sha256"])
                    self.assertEqual(b["capabilities"]["features"], features)
                    fixture = json.loads(FIXTURE.read_text())
                    rows, slots = len(fixture["layers"]) * fixture["hot_rows"], len(fixture["requests"]) + 1
                    rank = next(iter(b["ledger"]))
                    extra = b["ledger"][rank]["arena_metadata_bytes"] - a["ledger"][rank]["arena_metadata_bytes"]
                    if features is pr8:
                        self.assertEqual(extra, (slots + 1) * -(-rows // 32) * 8)   # one 64-bit word per 32 rows
                    else:
                        self.assertEqual(b["ledger"][rank]["arena_metadata_breakdown"]["row_owner"], slots * rows)
                        self.assertIn("ring", b["ledger"][rank]["arena_metadata_breakdown"])
            for name, pool in EXTRA_ARMS.items():                                 # every policy with both sets
                plain = Path(tmp) / f"{name}.json"
                plain.write_text(json.dumps({"schema": sc.SCHEMA, "hot_pool": pool}))
                _, a = run(plain, Path(tmp) / f"{name}.plain.out.json")
                for label, features in (("pr8", pr8), ("issue6", issue6)):
                    path = Path(tmp) / f"{name}_{label}.json"
                    path.write_text(json.dumps({"schema": sc.SCHEMA, "hot_pool": pool,
                                                "features": {k: v for k, v in features.items() if v}}))
                    with self.subTest(arm=name, features=label):
                        code, b = run(path, Path(tmp) / f"{name}.{label}.out.json")
                        self.assertEqual(code, 0)
                        self.assertEqual({k: a[k] for k in same}, {k: b[k] for k in same})

    def test_invalid_unsupported_and_existing_outputs_exit_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = {"typo": {"schema": sc.SCHEMA, "hot_pool": {"mode": "shared"}},
                   "placement": {"schema": sc.SCHEMA, "placement": {"mode": "fixed_layers", "offload_layers": [1]}},
                   "unknown_layer": {"schema": sc.SCHEMA, "hot_pool": {"mode": "shared", "layout": "arena",
                                                                      "admission": "bypass_list",
                                                                      "bypass_layers": [9]}}}
            for name, value in bad.items():
                path = Path(tmp) / f"{name}.json"
                path.write_text(json.dumps(value))
                with self.subTest(arm=name), contextlib.redirect_stderr(io.StringIO()):
                    try:
                        code = replay.main(["--fixture", str(FIXTURE), "--config", str(path),
                                            "--output", str(Path(tmp) / f"{name}.out")])
                    except SystemExit as exit_:
                        code = exit_.code
                    self.assertNotEqual(code, 0)
            existing = Path(tmp) / "exists.json"
            existing.write_text("{}")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                replay.main(["--fixture", str(FIXTURE), "--config", str(ARMS / "stock.json"),
                             "--output", str(existing)])
            self.assertEqual(existing.read_text(), "{}")


if __name__ == "__main__":
    unittest.main()
