"""CPU tests for the common hisparse-study-config-v1 parser (stdlib only)."""
import json
from pathlib import Path
import tempfile
import unittest

from runtime import study_config as sc

CONTRACT_EXAMPLE = {
    "schema": "hisparse-study-config-v1",
    "placement": {"mode": "native", "selection": "fixed", "offload_layers": []},
    "hot_pool": {"mode": "isolated", "scope": "request_rank"},
    "features": {},
    "diagnostics": {"mode": "none"},
}
SHARED = {"schema": sc.SCHEMA, "hot_pool": {"mode": "shared", "layout": "arena"}}


def parse(value):
    raw = value if isinstance(value, (bytes, str)) else json.dumps(value)
    return sc.parse(sc.loads(raw))


def with_pool(**pool):
    return {"schema": sc.SCHEMA, "hot_pool": pool}


class StudyConfigParser(unittest.TestCase):
    def test_contract_example_and_empty_file_select_stock(self):
        for value in (CONTRACT_EXAMPLE, {"schema": sc.SCHEMA}):
            config = parse(value)
            self.assertTrue(config.is_native)
            self.assertFalse(config.uses_arena)
            self.assertEqual(sc.capabilities(config)["runtime"], "stock_hisparse")
            self.assertEqual(sc.unsupported_reasons(config), [])
        self.assertEqual(parse(CONTRACT_EXAMPLE).digest, parse({"schema": sc.SCHEMA}).digest)

    def test_strict_json(self):
        bad = [b'{"schema":"hisparse-study-config-v1","schema":"hisparse-study-config-v1"}',
               b'{"schema":"hisparse-study-config-v1","hot_pool":{"mode":"shared","mode":"shared"}}',
               b'{"schema":"hisparse-study-config-v1","placement":{"seed":NaN}}',
               b'{"schema":"hisparse-study-config-v1","placement":{"seed":Infinity}}',
               b'\xef\xbb\xbf{"schema":"hisparse-study-config-v1"}',
               b'{"schema":"hisparse-study-config-v1"} trailing',
               b'{"schema":"hisparse-study-config-\xff"}',
               b'[]']
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse(raw)

    def test_unknown_fields_and_types(self):
        for value in ({"schema": "hisparse-study-config-v2"}, {},
                      {"schema": sc.SCHEMA, "extra": 1},
                      {"schema": sc.SCHEMA, "hot_pool": {"layoutt": "arena"}},
                      {"schema": sc.SCHEMA, "hot_pool": {"mode": "Shared"}},
                      {"schema": sc.SCHEMA, "hot_pool": {"scope": "request"}},
                      {"schema": sc.SCHEMA, "hot_pool": []},
                      {"schema": sc.SCHEMA, "diagnostics": {"mode": "trace"}},
                      {"schema": sc.SCHEMA, "features": {"zero_miss_fast_path": False}},
                      {"schema": sc.SCHEMA, "features": []}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse(value)

    def test_hot_pool_consistency(self):
        rejected = [
            with_pool(mode="shared"),                                  # shared needs arena
            with_pool(layout="native", eviction="fifo"),               # native supports native policy only
            with_pool(mode="isolated", layout="arena", admission="bypass_list", bypass_layers=[1]),
            with_pool(mode="shared", layout="arena", admission="bypass_list"),
            with_pool(mode="shared", layout="arena", bypass_layers=[1]),
            with_pool(mode="shared", layout="arena", quota="static", quota_rows={"0": 8}),
            with_pool(mode="isolated", layout="arena", quota="static"),
            with_pool(mode="isolated", layout="arena", quota="soft_floor", floor_rows={"0": 1}),
            with_pool(mode="shared", layout="arena", quota="soft_floor"),
            with_pool(mode="shared", layout="arena", floor_rows={"0": 1}),
            with_pool(mode="isolated", layout="arena", quota="static", quota_rows={"00": 8}),
            with_pool(mode="isolated", layout="arena", quota="static", quota_rows={"-1": 8}),
            with_pool(mode="isolated", layout="arena", quota="static", quota_rows={"0": True}),
            with_pool(mode="isolated", layout="arena", quota="static", quota_rows={}),
            with_pool(mode="shared", layout="arena", admission="bypass_list", bypass_layers=[2, 2]),
        ]
        for value in rejected:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse(value)
        accepted = [
            with_pool(mode="isolated", layout="arena"),
            with_pool(mode="isolated", layout="arena", eviction="fifo"),
            with_pool(mode="isolated", layout="arena", quota="static", quota_rows={"1": 6, "0": 2}),
            with_pool(mode="shared", layout="arena", quota="soft_floor", floor_rows={"3": 1}),
            with_pool(mode="shared", layout="arena", admission="bypass_list", bypass_layers=[5, 1]),
        ]
        for value in accepted:
            with self.subTest(value=value):
                config = parse(value)
                self.assertTrue(config.uses_arena)
                self.assertEqual(sc.capabilities(config)["runtime"], "shared_lru_arena")
                self.assertEqual(sc.unsupported_reasons(config), [])
        static = parse(accepted[2]).hot_pool
        self.assertEqual(static.quota_rows, ((0, 2), (1, 6)))
        self.assertEqual(parse(accepted[4]).hot_pool.bypass_layers, (1, 5))

    def test_placement_consistency_and_capability_gate(self):
        rejected = [
            {"schema": sc.SCHEMA, "placement": {"mode": "fixed_layers"}},
            {"schema": sc.SCHEMA, "placement": {"mode": "native", "offload_layers": [1]}},
            {"schema": sc.SCHEMA, "placement": {"mode": "full_resident", "offload_layers": [2]}},
            {"schema": sc.SCHEMA, "placement": {"mode": "fixed_layers", "offload_layers": [-1]}},
            {"schema": sc.SCHEMA, "placement": {"mode": "native", "selection": "random", "seed": 1}},
            {"schema": sc.SCHEMA, "placement": {"mode": "fixed_layers", "offload_layers": [1], "selection": "random"}},
            {"schema": sc.SCHEMA, "placement": {"mode": "fixed_layers", "offload_layers": [1], "seed": 3}},
            {"schema": sc.SCHEMA, "placement": {"mode": "fixed_layers", "offload_layers": [1],
                                                "selection": "calibrated", "calibration": {"list_sha256": "0" * 64}}},
            {"schema": sc.SCHEMA, "placement": {"mode": "fixed_layers", "offload_layers": [1, 1]}},
        ]
        for value in rejected:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse(value)
        fixed = parse({"schema": sc.SCHEMA, "placement": {"mode": "fixed_layers", "offload_layers": [7, 3],
                                                          "selection": "calibrated",
                                                          "calibration": {"training_sha256": "a" * 64,
                                                                          "list_sha256": "b" * 64}}})
        self.assertEqual(fixed.placement.offload_layers, (3, 7))
        self.assertTrue(fixed.uses_fixed_layers)
        self.assertEqual(sc.unsupported_reasons(fixed), [])
        caps = sc.capabilities(fixed)
        self.assertEqual((caps["runtime"], caps["offload_layers"], caps["host_backing"]),
                         ("fixed_layers+stock_hisparse", [3, 7], "offloaded layers only"))
        self.assertTrue(caps["experimental_gpu_state"])
        shared = parse(dict(SHARED, placement={"mode": "fixed_layers", "offload_layers": [5, 15]}))
        self.assertEqual(sc.capabilities(shared)["runtime"], "fixed_layers+shared_lru_arena")
        self.assertIn("offloaded layers", sc.capabilities(shared)["borrowing_domain"])
        empty = parse({"schema": sc.SCHEMA, "placement": {"mode": "fixed_layers", "offload_layers": []}})
        self.assertIn("at least one host-backed layer", sc.unsupported_reasons(empty)[0])
        resident = parse({"schema": sc.SCHEMA, "placement": {"mode": "full_resident"}})
        self.assertIn("without --enable-hisparse", sc.unsupported_reasons(resident)[0])
        counters = parse({"schema": sc.SCHEMA, "diagnostics": {"mode": "counters"}})
        self.assertTrue(sc.unsupported_reasons(counters))
        self.assertEqual(sc.unsupported_reasons(parse(dict(SHARED, diagnostics={"mode": "counters"}))), [])

    def test_canonical_digest_and_file_pin(self):
        a = parse(json.dumps({"hot_pool": {"layout": "arena", "mode": "shared"}, "schema": sc.SCHEMA}, indent=3))
        b = parse(SHARED)
        self.assertEqual(a.digest, b.digest)
        self.assertNotEqual(a.digest, parse(with_pool(mode="isolated", layout="arena")).digest)
        self.assertEqual(parse(a.canonical()), a)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arm.json"
            path.write_text(json.dumps(SHARED))
            config, digest = sc.read(path)
            self.assertEqual(config, b)
            self.assertEqual(sc.read(path, digest)[0], b)
            path.write_text(json.dumps(SHARED) + " ")
            with self.assertRaisesRegex(ValueError, "changed"):
                sc.read(path, digest)


if __name__ == "__main__":
    unittest.main()
