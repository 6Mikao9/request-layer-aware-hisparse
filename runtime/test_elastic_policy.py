import unittest

from runtime import elastic_config as ec
from runtime import elastic_memory as em
from runtime import elastic_policy as ep

PAGE = 4


def config(**overrides):
    data = {"schema": ec.SCHEMA, "model": "minimax_m3", "mode": "steady",
            "layers": {"eviction_order": [[0, 1], [2, 3]]},
            "profile": {"miss_rate": {"1024": {"0": 0.01, "1": 0.01, "2": 0.05, "3": 0.05}}, "fixed_us": 20.0,
                        "per_request_us": 0.0, "selected_rows": 64, "h2d_gbps": 10.0, "bytes_per_miss": 512},
            "controller": {"interval_steps": 1, "dwell_steps": 4, "horizon_steps": 2}}
    for key, value in overrides.items():
        data[key] = value if not isinstance(value, dict) else {**data.get(key, {}), **value}
    return ec.parse(data)


def geometry(chunks=None, layers=4):
    specs = [em.BufferSpec("idx", 64, em.ALWAYS, pad_rows=PAGE)]
    specs += [em.BufferSpec(f"{kind}{layer}", 256, em.SWITCHABLE, layer=layer, pad_rows=PAGE)
              for layer in range(layers) for kind in "kv"]
    layout = em.Layout(specs, chunk_bytes=4096, max_tokens=1024, hot_rows=64, page_tokens=PAGE)
    if chunks is None:
        chunks = layout.chunks_needed(tokens=96, hot_rows=32, offloaded=frozenset())
    return ep.Geometry(layout, chunks, hot_per_request=8, max_running=8)


def obs(**kw):
    base = dict(step=0, level=0, tokens=96, tokens_used=0, tokens_peak=0, hot_rows=32, hot_used=0, hot_peak=0,
                running=0, mean_context=64.0, waiting=0, waiting_tokens=0)
    return ep.Observation(**{**base, **kw})


class CostModelTests(unittest.TestCase):
    def test_layer_cost_follows_misses_and_bandwidth(self):
        model = ep.CostModel(config().profile)
        # 4 requests x 64 rows x 5% x 512 B at 10 GB/s = 0.655 us, plus 20 us fixed.
        self.assertAlmostEqual(model.layer_ms(2, 4, 64.0), (20.0 + 4 * 64 * 0.05 * 512 / 1e4) / 1e3)
        self.assertLess(model.layer_ms(0, 4, 64.0), model.layer_ms(2, 4, 64.0))
        self.assertEqual(model.layer_ms(0, 0, 64.0), 0.0)

    def test_measured_profile_fits_the_repository_aggregate(self):
        profile = ep.profile_from_measurements("data/a100-hisparse-graph-and-c4-20261124.json")
        mm = profile["minimax"]
        self.assertEqual(sorted(mm["miss_rate"]), ["16384", "32768", "65536"])
        self.assertEqual(len(mm["miss_rate"]["32768"]), 60)
        self.assertTrue(8.0 < mm["h2d_gbps"] < 16.0)
        self.assertEqual(len(profile["deepseek_v4"]["miss_rate"]), 2)

    def test_base_estimator_recovers_a_line(self):
        base = ep.BaseEstimator()
        for batch in (2, 4, 8, 16, 2, 4, 8, 16):
            base.observe(batch, 10.0 + 0.5 * batch)
        a, b = base.coefficients()
        self.assertAlmostEqual(a, 10.0, places=6)
        self.assertAlmostEqual(b, 0.5, places=6)


class ControllerTests(unittest.TestCase):
    def test_grows_into_free_chunks_before_offloading(self):
        g = geometry(chunks=geometry().chunks + 4)
        ctl = ep.Controller(config(), g)
        decision = ctl.tick(obs(tokens_used=90, tokens_peak=92, running=2))
        self.assertEqual(decision.level, 0)
        self.assertGreater(decision.tokens, 96)
        self.assertEqual(decision.reason, "grow")

    def test_offloads_to_the_smallest_level_whose_batch_admits_the_queue(self):
        ctl = ep.Controller(config(), geometry())
        pressure = obs(tokens_used=94, tokens_peak=96, running=4, waiting=2, waiting_tokens=40)
        per_new = 40 / 2 + 2
        points = {level: ctl.operating_point(level, pressure, per_new) for level in range(3)}
        smallest = min(level for level, point in points.items() if point[0] >= 6)
        decision = ctl.tick(pressure)
        self.assertEqual(decision.level, smallest)
        self.assertEqual(decision.reason, f"offload to level {smallest}")
        batch, rows, cap = points[smallest]
        demand = 96 + 4 * 2 + (batch - 4) * per_new             # live rows + horizon, plus each admitted one
        lazy = min(cap, max(pressure.tokens, -(-int(demand / 0.9) // PAGE) * PAGE))
        self.assertEqual((decision.tokens, decision.hot_rows), (lazy, rows))    # only what the batch needs
        self.assertGreaterEqual(decision.tokens, 96 + 4 * 2 + 2 * per_new)
        self.assertGreaterEqual(rows, 6 * 8)

    def test_slo_stops_offloading_and_requests_queue(self):
        pressure = obs(tokens_used=94, tokens_peak=96, running=4, waiting=2, waiting_tokens=40)
        base = {"base_ms": [10.0, 0.0]}
        tight = ep.Controller(config(slo={"tpot_ms": 10.03, "tail_factor": 1.0, "margin_ms": 0.0},
                                     profile=base), geometry())
        self.assertFalse(tight.slo_ok(1, 6, 64.0))
        full = tight.capacity(0, 32)
        refused = tight.tick(ep.Observation(**{**pressure.__dict__, "tokens": full}))
        self.assertEqual((refused.level, refused.tokens), (0, full))     # no offload: requests queue
        self.assertEqual(refused.admission_cap, 4)     # the running batch: admitting more forces an offload
        loose = ep.Controller(config(slo={"tpot_ms": 10.06, "tail_factor": 1.0, "margin_ms": 0.0},
                                     profile=base), geometry())
        self.assertTrue(loose.slo_ok(1, 6, 64.0) and not loose.slo_ok(2, 6, 64.0))
        partial = loose.tick(pressure)
        self.assertEqual(partial.level, 1)          # the highest level the SLO allows, short of the demand

    def test_throughput_guard_prefers_the_lower_level_when_offloading_does_not_pay(self):
        # Base 10 ms + 2 ms per request: one more request no longer pays for 20 us x 2 layers plus their misses.
        heavy = {"base_ms": [10.0, 2.0], "fixed_us": 4000.0}
        ctl = ep.Controller(config(profile=heavy), geometry())
        pressure = obs(tokens_used=94, tokens_peak=96, running=4, waiting=2, waiting_tokens=40)
        decision = ctl.tick(ep.Observation(**{**pressure.__dict__, "tokens": ctl.capacity(0, 32)}))
        self.assertEqual(decision.level, 0)
        unguarded = ep.Controller(config(profile=heavy, features={"throughput_guard": False}), geometry())
        self.assertGreater(unguarded.tick(ep.Observation(**{**pressure.__dict__,
                                                          "tokens": ctl.capacity(0, 32)})).level, 0)

    def test_backfills_only_after_dwelling_at_low_demand(self):
        ctl = ep.Controller(config(), geometry())
        cap1 = ctl.capacity(1, 32)
        low = dict(level=1, tokens=cap1, tokens_used=8, tokens_peak=12, running=1)
        reasons = [ctl.tick(obs(step=s, **low)).reason for s in range(10, 16)]
        self.assertEqual(reasons[:4], ["low demand, dwelling before backfill"] * 4)
        self.assertEqual(reasons[4], "backfill to level 0")

    def test_initial_state_uses_every_chunk_of_its_level(self):
        g = geometry()
        decision = ep.Controller(config(), g).initial()
        self.assertEqual(decision.tokens, g.layout.max_tokens_for(g.chunks, hot_rows=32, offloaded=frozenset()))
        table = ep.describe_levels(config(), g, 32)
        self.assertEqual([row["level"] for row in table], [0, 1, 2])
        self.assertLess(table[0]["tokens"], table[2]["tokens"])


if __name__ == "__main__":
    unittest.main()
