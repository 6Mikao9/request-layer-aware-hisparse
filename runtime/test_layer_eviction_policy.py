import unittest

from runtime.layer_eviction_policy import Candidate, LayerEvictionPolicy, LayerKey


def candidate(layer, reclaim, *, cost=1, transfer=0, since=0, generation=1):
    return Candidate(LayerKey(1, generation, layer), reclaim, cost, transfer, since)


class TestLayerEvictionPolicy(unittest.TestCase):
    def policy(self, **kwargs):
        return LayerEvictionPolicy(
            trigger_free_bytes=(10, 10), target_free_bytes=(30, 30), **kwargs
        )

    def test_surplus_on_one_rank_cannot_cover_another(self):
        policy = self.policy(max_migrations=3)
        first = candidate(13, (1000, 0))
        second = candidate(1, (0, 40))
        useless = candidate(4, (1000, 0), cost=0)
        plan = policy.plan((5, 0), [first, second, useless], now=10)
        self.assertEqual(plan.keys, (useless.key, second.key))
        self.assertEqual(plan.projected_free_bytes, (1005, 40))
        self.assertTrue(plan.target_reached)

    def test_hysteresis_and_partial_completion_use_observed_bytes(self):
        policy = self.policy(max_migrations=2, cooldown=5)
        first = candidate(3, (12, 0))
        second = candidate(8, (0, 20))
        pending = policy.plan((10, 10), [first, second], now=10)
        self.assertEqual(pending.projected_free_bytes, (22, 30))
        self.assertFalse(pending.target_reached)
        policy.commit(pending, succeeded=[second.key], free_bytes=(15, 30), now=11)
        self.assertIsNone(policy.plan((15, 30), [first], now=15))
        pending = policy.plan((15, 30), [first], now=16)
        self.assertEqual(pending.keys, (first.key,))
        policy.commit(pending, succeeded=[], free_bytes=(15, 30), now=17)
        self.assertIsNone(policy.plan((15, 30), [first], now=21))
        self.assertIsNotNone(policy.plan((15, 30), [first], now=22))

    def test_min_residency_weight_limit_and_cancel(self):
        policy = self.policy(max_migrations=1, min_residency=4, transfer_weight=3)
        slow = candidate(9, (20, 20), transfer=10)
        cheap = candidate(13, (20, 20), cost=5)
        young = candidate(13, (20, 20), cost=0, since=8, generation=2)
        pending = policy.plan((5, 5), [slow, cheap, young], now=10)
        self.assertEqual(pending.keys, (cheap.key,))
        with self.assertRaises(RuntimeError):
            policy.plan((5, 5), [], now=10)
        with self.assertRaises(ValueError):
            policy.commit(pending, succeeded=[young.key], free_bytes=(5, 5), now=10)
        policy.cancel(pending)
        pending = policy.plan((5, 5), [slow, cheap, young], now=12)
        self.assertEqual(pending.keys, (young.key,))
        policy.commit(pending, succeeded=[young.key], free_bytes=(31, 31), now=12)
        self.assertIsNone(policy.plan((15, 30), [cheap], now=13))

    def test_no_pressure_and_no_useful_reclaim(self):
        policy = self.policy(max_migrations=1)
        useful = candidate(1, (100, 100))
        self.assertIsNone(policy.plan((11, 30), [useful], now=10))
        self.assertIsNone(policy.plan((0, 30), [candidate(1, (0, 100))], now=11))
        self.assertIsNone(policy.plan((0, 30), [candidate(1, (0, 0))], now=12))

    def test_invalid_inputs_leave_pending_plan_available(self):
        with self.assertRaises(ValueError):
            self.policy(max_migrations=True)
        with self.assertRaises(ValueError):
            LayerEvictionPolicy(
                trigger_free_bytes=(False,), target_free_bytes=(30,), max_migrations=1
            )
        policy = self.policy(max_migrations=1)
        for item in (
            Candidate(LayerKey(0, 1, 0), (10, 0), 1, 0, 0),
            Candidate(LayerKey(1, 0, 0), (10, 0), 1, 0, 0),
            Candidate(LayerKey(1, 1, -1), (10, 0), 1, 0, 0),
            Candidate(LayerKey(1, 1, True), (10, 0), 1, 0, 0),
            candidate(1, (False, 0)),
            candidate(1, (10,), since=0),
            candidate(1, (10, 0), since=11),
        ):
            with self.subTest(item=item), self.assertRaises(ValueError):
                policy.plan((0, 0), [item], now=10)
        with self.assertRaises(ValueError):
            policy.plan((0, 0), [], now=float("nan"))
        with self.assertRaises(ValueError):
            policy.plan((False, 0), [], now=10)
        valid = candidate(0, (40, 40))
        pending = policy.plan((0, 0), [valid], now=10)
        with self.assertRaises(ValueError):
            policy.commit(pending, succeeded=[], free_bytes=(0, 0), now=9)
        policy.commit(pending, succeeded=[valid.key], free_bytes=(40, 40), now=11)
        with self.assertRaises(ValueError):
            policy.plan((0, 0), [valid], now=10)
        retry = policy.plan((0, 0), [valid], now=11)
        self.assertEqual(retry, pending)
        with self.assertRaises(ValueError):
            policy.commit(pending, succeeded=[valid.key], free_bytes=(40, 40), now=11)
        policy.cancel(retry)
