from pathlib import Path
import json
import unittest

from runtime import elastic_config as ec
from runtime import elastic_policy as ep

FIXTURES = Path(__file__).with_name("fixtures") / "elastic"


def base(**layers):
    return {"schema": ec.SCHEMA, "model": "minimax_m3", "layers": {"eviction_order": [[0, 1], [2]], **layers}}


class ParseTests(unittest.TestCase):
    def test_every_fixture_parses_and_explains_its_levels(self):
        for path in sorted(FIXTURES.glob("*.json")):
            with self.subTest(path.name):
                config, digest = ec.read(path)
                self.assertEqual(len(digest), 64)
                if config.resolved:
                    self.assertEqual(config.offloaded(0), frozenset(config.always_offload))
                    self.assertEqual(len(config.offloaded(config.max_level)), len(config.managed))

    def test_groups_flat_lists_and_granularity(self):
        flat = ec.parse(base(eviction_order=[5, 4, 3, 2, 1], granularity=2))
        self.assertEqual(flat.groups, ((5, 4), (3, 2), (1,)))
        self.assertEqual(flat.offloaded(2), frozenset({2, 3, 4, 5}))
        self.assertEqual(flat.max_level, 3)
        self.assertEqual(ec.parse(base()).precapture_levels(), [0, 1, 2])

    def test_rejections(self):
        bad = [
            dict(base(), schema="hisparse-study-config-v1"),
            dict(base(), model="qwen"),
            dict(base(), extra=1),
            base(eviction_order=[[0], 1]),
            base(eviction_order=[[0, 1], [1]]),
            base(eviction_order=[[0]], always_offload=[0]),
            base(eviction_order=[[0]], granularity=2),
            base(eviction_order=[0, 1], granularity=0),
            base(always_offload=ec.SHARED_INDEX_FOLLOWERS),
            base(eviction_order=[[0]], pinned_resident=[0]),
            dict(base(), controller={"low_watermark": 0.95}),
            dict(base(), controller={"initial_level": 3}),
            dict(base(), memory={"chunk_mib": 3}),
            dict(base(), features={"fused": True}),
            dict(base(), slo={"tail_factor": 1.2}),
        ]
        for data in bad:
            with self.subTest(data=json.dumps(data)[:120]), self.assertRaises(ValueError):
                ec.parse(data)

    def test_steady_mode_has_no_stale_hits_and_slo_enables_admission(self):
        steady = ec.parse(dict(base(), features={"stale_hits": True}))
        self.assertFalse(steady.feature("stale_hits"))
        smooth = ec.parse(dict(base(), mode="smooth", slo={"tpot_ms": 50}))
        self.assertTrue(smooth.feature("stale_hits") and smooth.feature("slo_admission"))
        self.assertEqual(smooth.slo, {"tpot_ms": 50.0, "tail_factor": 1.1, "margin_ms": 1.0})


class ResolveTests(unittest.TestCase):
    def test_symbolic_lists_resolve_against_the_bound_model(self):
        config = ec.parse({"schema": ec.SCHEMA, "model": "dsa", "mode": "smooth",
                           "layers": {"eviction_order": ec.BY_MISS_RATE, "granularity": 2,
                                      "always_offload": ec.SHARED_INDEX_FOLLOWERS, "pinned_resident": [0]}})
        self.assertFalse(config.resolved)
        with self.assertRaises(ValueError):
            config.offloaded(0)
        rates = {1: 0.4, 4: 0.1, 5: 0.2, 7: 0.05}
        bound = config.resolve(range(8), followers=(2, 3, 6), miss_rate=lambda layer: rates.get(layer, 0.3))
        self.assertEqual(bound.groups, ((7, 4), (5, 1)))
        self.assertEqual(bound.always_offload, (2, 3, 6))
        self.assertEqual(bound.max_level, 2)
        self.assertNotIn(0, bound.offloaded(2))

    def test_unknown_layers_are_rejected_at_bind(self):
        config = ec.parse(base())
        with self.assertRaises(ValueError):
            config.resolve(range(2))
        self.assertEqual(config.resolve(range(3)).groups, config.groups)

    def test_measured_profile_fills_only_defaults(self):
        config, _ = ec.read(FIXTURES / "minimax_m3_steady.json")
        profile = ep.load_profile(config, Path(__file__).parents[1])
        self.assertEqual(len(profile["miss_rate"]["65536"]), 60)
        explicit = ec.parse(dict(base(), profile={"measurements": config.profile["measurements"], "fixed_us": 7.0}))
        self.assertEqual(ep.load_profile(explicit, Path(__file__).parents[1])["fixed_us"], 7.0)


if __name__ == "__main__":
    unittest.main()
