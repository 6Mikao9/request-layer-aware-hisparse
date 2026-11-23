from dataclasses import FrozenInstanceError, replace
import unittest

from runtime.layer_hybrid_pool import (
    LayerGroup,
    LayerLayout,
    LayerPagePool,
    PageBudgetExceeded,
    RequestLease,
    require_same_budget,
)


def layout(*, hybrid=True):
    groups = (
        (LayerGroup("early", (0,), "resident", 8), LayerGroup("late", (1, 2, 3), "hot", 4, 8))
        if hybrid
        else (LayerGroup("early", (0,), "hot", 5, 8), LayerGroup("late", (1, 2, 3), "hot", 5, 8))
    )
    return LayerLayout(4, 4, 4, 16, 2, 4, groups, 1536)


class LayerLayoutTests(unittest.TestCase):
    def test_exact_layer_weighted_backing_includes_each_padding_sink(self):
        baseline, hybrid = layout(hybrid=False), layout()
        require_same_budget(baseline, hybrid)
        self.assertEqual(hybrid.backing_bytes, 1536)
        self.assertEqual(hybrid.digest, layout().digest)
        self.assertNotEqual(baseline.digest, hybrid.digest)
        self.assertEqual(hybrid.hot_page_limit(hybrid.groups[1]), 3)
        with self.assertRaises(FrozenInstanceError):
            hybrid.groups[0].mode = "hot"
        with self.assertRaisesRegex(ValueError, "budget"):
            replace(hybrid, budget_bytes=hybrid.budget_bytes - 1)
        with self.assertRaisesRegex(ValueError, "block_tokens"):
            require_same_budget(baseline, replace(hybrid, block_tokens=2))

    def test_rejects_incomplete_and_overlapping_layers_and_invalid_hot_rows(self):
        spec = layout()
        early, late = spec.groups
        for bad_group, error in (
            (replace(late, layers=(0, 1, 2, 3)), "partition"),
            (replace(late, layers=(1, 3)), "partition"),
            (replace(late, hot_capacity=4), "selected tokens"),
            (replace(late, hot_capacity=9), "page aligned"),
            (replace(late, hot_capacity=16), "latest-token"),
        ):
            with self.subTest(bad_group=bad_group), self.assertRaisesRegex(ValueError, error):
                replace(spec, groups=(early, bad_group))


class LayerPoolTests(unittest.TestCase):
    def setUp(self):
        self.pool = LayerPagePool(layout())

    def test_group_release_and_slot_reuse_do_not_touch_equal_page_ids_in_other_group(self):
        pool = self.pool
        first, second = pool.open_request(0), pool.open_request(1)
        pool.grow(first, {"early": 1, "late": 1})
        self.assertEqual(pool.page_ids(first, "early"), (1,))
        self.assertEqual(pool.page_ids(first, "late"), (1,))
        self.assertEqual(pool.route(0).mode, "resident")
        self.assertEqual(pool.route(3).mode, "hot")
        late = pool.location(first, 3, 2, write=True)
        pool.release_tail(first, "early", 0)
        pool.grow(second, {"early": 1})
        self.assertEqual(pool.page_ids(second, "early"), (1,))
        self.assertEqual(pool.location(first, 3, 2, write=True), late)
        self.assertEqual(pool.free_pages("late"), 3)
        with self.assertRaisesRegex(ValueError, "outside"):
            pool.location(first, 0, 0)

    def test_atomic_admission_and_hot_limit_including_reserved_page(self):
        pool = self.pool
        first, second = pool.open_request(0), pool.open_request(1)
        pool.grow(first, {"late": 3})
        with self.assertRaisesRegex(PageBudgetExceeded, "late needs"):
            pool.grow(second, {"early": 2, "late": 2})
        self.assertEqual(pool.page_ids(second, "early"), ())
        self.assertEqual(pool.free_pages("early"), 8)
        self.assertEqual(pool.free_pages("late"), 1)
        self.assertEqual(pool.location(first, 1, 8, write=True), 12)
        with self.assertRaisesRegex(PageBudgetExceeded, "latest-token"):
            pool.grow(first, {"late": 4})
        pool.finish(first)
        pool.grow(second, {"early": 2, "late": 2})
        self.assertEqual(pool.page_ids(second, "late"), (1, 2))

    def test_resident_prefix_remains_readable_until_last_reference_then_reuses_writable(self):
        pool = self.pool
        first, second, third = [pool.open_request(row) for row in range(3)]
        pool.grow(first, {"early": 2, "late": 1})
        with self.assertRaisesRegex(ValueError, "sealed"):
            pool.share_resident_prefix(first, second, "early", 1)
        pool.seal_resident_prefix(first, "early", 1)
        pool.share_resident_prefix(first, second, "early", 1)
        pool.share_resident_prefix(second, third, "early", 1)
        pool.grow(second, {"early": 2})
        self.assertEqual(pool.page_ids(second, "early"), (1, 3))
        for lease in (first, second, third):
            self.assertEqual(pool.location(lease, 0, 0), 4)
            with self.assertRaisesRegex(ValueError, "sealed"):
                pool.location(lease, 0, 0, write=True)
        pool.finish(first)
        self.assertEqual(pool.free_pages("late"), 4)
        self.assertEqual(pool.free_pages("early"), 6)
        pool.release_tail(second, "early", 0)
        self.assertEqual(pool.location(third, 0, 0), 4)
        self.assertEqual(pool.free_pages("early"), 7)
        pool.finish(third)
        self.assertEqual(pool.free_pages("early"), 8)
        pool.grow(second, {"early": 1})
        self.assertEqual(pool.location(second, 0, 0, write=True), 4)

    def test_row_generations_reject_stale_foreign_and_forged_leases(self):
        pool = self.pool
        first = pool.open_request(0)
        with self.assertRaisesRegex(ValueError, "already active"):
            pool.open_request(0)
        pool.finish(first)
        current = pool.open_request(0)
        self.assertEqual(current.generation, first.generation + 1)
        pool.grow(current, {"early": 1, "late": 1})
        other = LayerPagePool(layout()).open_request(0)
        for lease in (first, other, RequestLease(current.row, current.generation)):
            with self.assertRaisesRegex(ValueError, "Stale or foreign"):
                pool.finish(lease)
        self.assertEqual(pool.page_ids(current, "early"), (1,))
        self.assertEqual(pool.page_ids(current, "late"), (1,))

    def test_live_locations_are_stable_across_growth_and_other_rows_reuse(self):
        pool = self.pool
        first, second = pool.open_request(0), pool.open_request(1)
        pool.grow(first, {"early": 1, "late": 1})
        locs = [pool.location(first, layer, 3) for layer in range(4)]
        pool.grow(second, {"early": 3, "late": 1})
        pool.finish(second)
        pool.grow(first, {"early": 4, "late": 3})
        self.assertEqual([pool.location(first, layer, 3) for layer in range(4)], locs)
        self.assertTrue(all(loc >= pool.layout.page_tokens for loc in locs))

    def test_hot_prefix_is_not_shareable(self):
        pool = self.pool
        first, second = pool.open_request(0), pool.open_request(1)
        pool.grow(first, {"late": 1})
        with self.assertRaisesRegex(ValueError, "resident"):
            pool.seal_resident_prefix(first, "late", 1)
        with self.assertRaisesRegex(ValueError, "resident"):
            pool.share_resident_prefix(first, second, "late", 1)
        self.assertEqual(pool.page_ids(second, "late"), ())


if __name__ == "__main__":
    unittest.main()
