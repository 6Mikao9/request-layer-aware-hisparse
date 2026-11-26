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
        self.assertEqual(sc.unsupported_reasons(fixed), [])
        self.assertEqual(sc.capabilities(fixed)["runtime"], "fixed_layers_stock_paths")
        fixed_arena = parse({"schema": sc.SCHEMA, "placement": {"mode": "fixed_layers", "offload_layers": [3]},
                             "hot_pool": {"mode": "shared", "layout": "arena"}})
        self.assertIn("combined effort", sc.unsupported_reasons(fixed_arena)[0])
        self.assertIn("without --enable-hisparse",
                      sc.unsupported_reasons(parse({"schema": sc.SCHEMA, "placement": {"mode": "full_resident"}}))[0])
        counters = parse({"schema": sc.SCHEMA, "diagnostics": {"mode": "counters"}})
        self.assertTrue(sc.unsupported_reasons(counters))
        self.assertEqual(sc.unsupported_reasons(parse(dict(SHARED, diagnostics={"mode": "counters"}))), [])

    def test_dynamic_offload_switches_and_registered_lists(self):
        def fixed(features=None, alternates=None, **pool):
            placement = {"mode": "fixed_layers", "offload_layers": [4, 2]}
            if alternates is not None:
                placement["alternate_offload_layers"] = alternates
            return {"schema": sc.SCHEMA, "placement": placement, "features": features or {},
                    **({"hot_pool": pool} if pool else {})}
        rejected = [
            (fixed({"dynamic_offload": True}, [[2, 6]]), "needs features.route_indirect"),
            (fixed({"route_indirect": True, "dynamic_offload": True}), "alternate_offload_layers to switch"),
            (fixed({"route_indirect": True, "host_shadow": True}, [[2, 6]]), "needs features.dynamic_offload"),
            (fixed(alternates=[[2, 6]]), "needs features.route_indirect"),
            (fixed({"route_indirect": True}, [[2, 6, 8]]), "as many layers"),
            (fixed({"route_indirect": True}, [[6, 2], [2, 6]]), "repeats"),
            (fixed({"route_indirect": True}, [[2, 4]]), "repeats"),
            (fixed({"route_indirect": True}, []), "non-empty"),
            (fixed({"route_indirect": True}, [[2, 2]]), "repeats a layer"),
            (fixed({"route_indirect": True}, mode="shared", layout="arena"), "layout=native"),
            ({"schema": sc.SCHEMA, "features": {"route_indirect": True}}, "fixed_layers"),
            ({"schema": sc.SCHEMA, "placement": {"mode": "native", "alternate_offload_layers": [[1]]}}, "fixed_layers"),
        ]
        for value, pattern in rejected:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, pattern):
                parse(value)
        dynamic = parse(fixed({"route_indirect": True, "dynamic_offload": True, "host_shadow": True},
                              [[2, 6], [8, 4]]))
        self.assertEqual(dynamic.offload_lists, ((2, 4), (2, 6), (4, 8)))
        self.assertEqual(dynamic.as_dict()["placement"]["alternate_offload_layers"], [[2, 6], [4, 8]])
        self.assertEqual(parse(dynamic.canonical()), dynamic)
        caps = sc.capabilities(dynamic)
        self.assertEqual((caps["runtime"], caps["placement"]["switchable_layers"]), ("dynamic_layer_offload",
                                                                                     [2, 4, 6, 8]))
        self.assertEqual(caps["features"], dict(route_indirect=True, dynamic_offload=True, host_shadow=True))
        indirect = parse(fixed({"route_indirect": True}, [[2, 6]]))
        self.assertEqual(sc.capabilities(indirect)["runtime"], "fixed_layers_route_indirect")
        self.assertIsNone(sc.capabilities(indirect)["switching"])
        self.assertNotEqual(indirect.digest, parse(fixed({"route_indirect": True})).digest)
        self.assertEqual(sc.unsupported_reasons(dynamic), [])

    def test_resolver_switches_are_arena_only_and_canonical_when_enabled(self):
        self.assertEqual(sc.ARENA_FEATURES, ("owner_summary", "touched_bitmap", "hit_index", "incremental_lru",
                                             "compact_owner"))
        phase = dict(owner_summary="hit_search", hit_index="hit_search", touched_bitmap="recency_update",
                     incremental_lru="recency_update", compact_owner="owner_tags")
        for name in sc.ARENA_FEATURES:
            on = parse(dict(SHARED, features={name: True}))
            with self.subTest(feature=name):
                self.assertTrue(on.feature(name))
                self.assertEqual(sc.unsupported_reasons(on), [])
                self.assertEqual(on.as_dict()["features"], {name: True})
                self.assertNotEqual(on.digest, parse(SHARED).digest)
                caps = sc.capabilities(on)
                self.assertEqual(caps["features"], {n: n == name for n in sc.ARENA_FEATURES})
                self.assertTrue(caps[phase[name]].startswith(name))
                for value in (with_pool(), with_pool(mode="isolated", layout="native")):
                    with self.assertRaisesRegex(ValueError, "layout=native"):
                        parse(dict(value, features={name: True}))
                with self.assertRaises(ValueError):
                    parse(dict(SHARED, features={name: 1}))
        # false and absent are one configuration, including the all-default stock arm
        self.assertEqual(parse(dict(SHARED, features={"owner_summary": False})).digest, parse(SHARED).digest)
        stock = parse({"schema": sc.SCHEMA, "features": {"touched_bitmap": False}})
        self.assertTrue(stock.is_native)
        self.assertEqual(sc.capabilities(stock)["features"], {})
        both = parse(dict(SHARED, features={"touched_bitmap": True, "owner_summary": True}))
        self.assertEqual(both.features, (("owner_summary", True), ("touched_bitmap", True)))
        self.assertEqual(parse(both.canonical()), both)
        isolated = parse(dict(with_pool(mode="isolated", layout="arena"), features={"owner_summary": True}))
        self.assertTrue(isolated.feature("owner_summary") and not isolated.feature("touched_bitmap"))
        with self.assertRaises(KeyError):
            both.feature("zero_miss_fast_path")

    def test_alternatives_for_one_phase_are_exclusive(self):
        """One ablation measures one technique: the two hit searches or the two recency updates never stack."""
        for pair in sc.EXCLUSIVE_FEATURES:
            with self.subTest(pair=pair), self.assertRaisesRegex(ValueError, "same resolver phase"):
                parse(dict(SHARED, features=dict.fromkeys(pair, True)))
        indexed = parse(dict(SHARED, features={"hit_index": True, "incremental_lru": True, "compact_owner": True}))
        self.assertEqual([name for name, _ in indexed.features], ["compact_owner", "hit_index", "incremental_lru"])
        caps = sc.capabilities(indexed)
        self.assertTrue(caps["hit_search"].startswith("hit_index") and caps["recency_update"].startswith("incremental"))
        self.assertTrue(caps["owner_tags"].startswith("compact_owner"))
        cross = parse(dict(SHARED, features={"hit_index": True, "touched_bitmap": True}))
        self.assertTrue(cross.feature("hit_index") and cross.feature("touched_bitmap"))

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
