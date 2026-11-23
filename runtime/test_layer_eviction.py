import unittest
from unittest.mock import patch

import torch

from runtime.layer_eviction import RequestLayerEvictionKV
from runtime.layer_hybrid_kv import LayerHybridPoisoned
from runtime.layer_hybrid_pool import LayerGroup, LayerLayout, PageBudgetExceeded
from runtime.test_layer_hybrid_kv import CPUMHAPool, values


class RequestLayerEvictionTests(unittest.TestCase):
    def setUp(self):
        self.history = {}
        self.page_bytes = 64
        self.host_bytes = 1024

    def adapter(self, host_budget=3072):
        groups = (LayerGroup("first", (0,), "resident", 12),
                  LayerGroup("second", (1,), "resident", 12))
        layout = LayerLayout(2, 5, 4, 16, 2, 4, groups, 26 * self.page_bytes)
        return RequestLayerEvictionKV(
            layout, 256, 4, "cpu", max_context=64,
            host_budget_bytes=host_budget, pool_factory=CPUMHAPool,
        )

    def append(self, adapter, lease, layer, count):
        ids, keys, vals = self.history.setdefault((lease.row, layer), ([], [], []))
        start = len(ids)
        logical = [lease.row * 48 + pos for pos in range(start, start + count)]
        k = values(count, lease.row * 37 + start * 4 + layer)
        v = -k - 2
        adapter.append(lease, layer, logical, k, v)
        ids.extend(logical)
        keys.extend(k)
        vals.extend(v)

    def decode(self, adapter, layer, rows, blocks):
        req_to_token = torch.full((5, 64), -1, dtype=torch.int32)
        lengths = []
        for row in rows:
            ids, _, _ = self.history[row, layer]
            lengths.append(len(ids))
            req_to_token[row, :len(ids)] = torch.tensor(ids)
        slots = adapter.resolve_decode(
            layer, torch.tensor(blocks, dtype=torch.int32),
            torch.tensor(rows, dtype=torch.int64),
            torch.tensor(lengths, dtype=torch.int32), req_to_token,
        )
        k, v = adapter.get_kv_buffer(layer)
        self.assertEqual(slots.shape, (len(rows), 8))
        for i, row in enumerate(rows):
            _, keys, vals = self.history[row, layer]
            for col, slot in enumerate(slots[i].tolist()):
                block = blocks[i][col // 4]
                token = block * 4 + col % 4
                if block < 0 or token >= lengths[i]:
                    self.assertEqual(slot, -1)
                else:
                    self.assertGreaterEqual(slot, 4)
                    torch.testing.assert_close(k[slot], keys[token], rtol=0, atol=0)
                    torch.testing.assert_close(v[slot], vals[token], rtol=0, atol=0)
        return slots

    def test_preparation_metadata_oom_does_not_leak_host_budget(self):
        adapter = self.adapter()
        lease = adapter.open_request(1)
        self.append(adapter, lease, 0, 21)
        before = adapter.accounting()
        with patch("runtime.layer_eviction.DemotionTicket", side_effect=MemoryError("ticket")):
            with self.assertRaises(MemoryError):
                adapter.prepare_demote(lease, 0, 8)
        self.assertEqual(adapter.accounting(), before)

        class FailedPublication(dict):
            def __setitem__(self, key, value):
                raise MemoryError("pending metadata")

        with patch.object(adapter, "_pending", FailedPublication()):
            with self.assertRaises(MemoryError):
                adapter.prepare_demote(lease, 0, 8)
            self.assertEqual(adapter.accounting(), before)
        self.assertEqual(adapter.mode(lease, 0), "resident")
        self.decode(adapter, 0, [1], [[4, 5]])
        ticket = adapter.prepare_demote(lease, 0, 8)
        adapter.cancel_demote(ticket)
        self.assertEqual(adapter.accounting(), before)

    def test_mixed_order_real_page_reuse_and_latest_history(self):
        adapter = self.adapter()
        first, second, third = (adapter.open_request(row) for row in (1, 2, 3))
        for lease in (first, second):
            for layer in (0, 1):
                self.append(adapter, lease, layer, 21)
        old_tail = self.decode(adapter, 0, [1], [[4, 5]])[0].tolist()
        with self.assertRaises(PageBudgetExceeded):
            self.append(adapter, third, 0, 9)
        plan = adapter.prepare_demote(first, 0, 8)
        self.assertEqual(adapter.mode(first, 0), "resident")
        with self.assertRaisesRegex(RuntimeError, "prepared"):
            self.append(adapter, first, 0, 1)
        with self.assertRaisesRegex(RuntimeError, "prepared"):
            self.decode(adapter, 0, [1], [[0, 5]])
        with self.assertRaisesRegex(RuntimeError, "prepared"):
            adapter.finish(first)
        self.decode(adapter, 1, [1, 2], [[5, -1], [-1, 4]])
        adapter.cancel_demote(plan)
        self.assertEqual(adapter.accounting()["host_reserved_bytes"], 0)
        self.decode(adapter, 0, [1], [[4, 5]])
        with self.assertRaisesRegex(ValueError, "ticket"):
            adapter.commit_demote(plan)

        before = adapter.accounting()
        ticket = adapter.prepare_demote(first, 0, 8)
        self.assertEqual(ticket.reclaimable_bytes, 3 * self.page_bytes)
        self.assertEqual(adapter.commit_demote(ticket), ticket.reclaimable_bytes)
        after = adapter.accounting()
        self.assertEqual(after["main_backing_bytes"], before["main_backing_bytes"])
        self.assertEqual(after["free_bytes_by_layer"], {0: 192, 1: 0})
        self.assertEqual(adapter.mode(first, 0), "hot")
        self.assertEqual(adapter.mode(first, 1), "resident")
        self.assertEqual(adapter.mode(second, 0), "resident")
        self.append(adapter, third, 0, 9)
        new_tail = self.decode(adapter, 0, [3], [[2, 1]])[0].tolist()
        self.assertTrue((set(old_tail) & set(new_tail)) - {-1})
        self.assertEqual(adapter.accounting()["free_bytes_by_layer"][0], 0)

        second_ticket = adapter.prepare_demote(second, 0, 12)
        self.assertEqual(adapter.commit_demote(second_ticket), 2 * self.page_bytes)
        for _ in range(3):
            self.append(adapter, first, 0, 1)
        self.append(adapter, second, 0, 1)
        self.decode(adapter, 0, [3, 2, 1], [[2, -1], [5, 0], [5, 4]])
        self.decode(adapter, 0, [1, 3, 2], [[0, 1], [-5, 0], [-2, 3]])
        self.decode(adapter, 0, [2, 1], [[5, -1], [5, -1]])
        self.decode(adapter, 1, [2, 1], [[5, -1], [4, 0]])
        adapter.finish(first)
        self.assertEqual(adapter.accounting()["host_reserved_bytes"], self.host_bytes)
        self.assertEqual(adapter.accounting()["free_bytes_by_layer"], {0: 320, 1: 384})
        replacement = adapter.open_request(1)
        self.assertNotEqual(first.generation, replacement.generation)
        with self.assertRaises(ValueError):
            adapter.mode(first, 0)
        with self.assertRaisesRegex(ValueError, "ticket"):
            adapter.commit_demote(ticket)
        del self.history[1, 0]
        self.append(adapter, replacement, 0, 1)
        self.decode(adapter, 0, [1], [[0, -1]])

    def test_host_budget_and_partial_preparation_failure_preserve_resident_kv(self):
        adapter = self.adapter(host_budget=self.host_bytes)
        lease = adapter.open_request(1)
        for layer in (0, 1):
            self.append(adapter, lease, layer, 21)
        before = adapter.accounting()

        def partial_copy(lease, layer, hot, length):
            self.assertGreater(length, 4)
            original(lease, layer, hot, 4)
            raise RuntimeError("partial backup")

        original = adapter._backup_history
        with patch.object(adapter, "_backup_history", side_effect=partial_copy):
            with self.assertRaisesRegex(RuntimeError, "partial backup"):
                adapter.prepare_demote(lease, 0, 8)
        self.assertEqual(adapter.accounting(), before)
        self.decode(adapter, 0, [1], [[0, 5]])

        ticket = adapter.prepare_demote(lease, 0, 8)
        pending = adapter.accounting()
        with self.assertRaises(MemoryError):
            adapter.prepare_demote(lease, 1, 8)
        self.assertEqual(adapter.accounting(), pending)
        self.decode(adapter, 1, [1], [[5, 0]])
        adapter.cancel_demote(ticket)
        self.assertEqual(adapter.accounting(), before)
        self.append(adapter, lease, 0, 1)
        self.decode(adapter, 0, [1], [[5, 4]])

    def test_commit_failure_poison_keeps_pages_reserved(self):
        adapter = self.adapter()
        lease = adapter.open_request(1)
        self.append(adapter, lease, 0, 21)
        ticket = adapter.prepare_demote(lease, 0, 8)
        before = adapter._kv._pages.free_pages("first")
        with patch.object(adapter._kv, "_unbind", side_effect=RuntimeError("unbind failed")):
            with self.assertRaisesRegex(RuntimeError, "unbind failed"):
                adapter.commit_demote(ticket)
        self.assertEqual(adapter._kv._pages.free_pages("first"), before)
        with self.assertRaises(LayerHybridPoisoned):
            adapter.finish(lease)
        with self.assertRaises(LayerHybridPoisoned):
            adapter.cancel_demote(ticket)

    def test_rejects_zero_reclaim_and_sealed_resident_page(self):
        adapter = self.adapter()
        lease = adapter.open_request(1)
        self.append(adapter, lease, 0, 12)
        with self.assertRaisesRegex(ValueError, "no complete"):
            adapter.prepare_demote(lease, 0, 8)
        self.append(adapter, lease, 0, 9)
        adapter._kv._pages.seal_resident_prefix(lease, "first", 1)
        before = adapter.accounting()
        with self.assertRaisesRegex(ValueError, "unsealed"):
            adapter.prepare_demote(lease, 0, 8)
        self.assertEqual(adapter.accounting(), before)
        self.decode(adapter, 0, [1], [[5, 0]])

    def test_allocation_oom_rolls_back_reservation_and_can_retry(self):
        adapter = self.adapter()
        lease = adapter.open_request(1)
        self.append(adapter, lease, 0, 21)
        before = adapter.accounting()
        with patch.object(adapter, "_allocate_hot", side_effect=MemoryError("allocation OOM")):
            with self.assertRaises(MemoryError):
                adapter.prepare_demote(lease, 0, 8)
        self.assertEqual(adapter.accounting(), before)
        self.decode(adapter, 0, [1], [[5, 0]])
        ticket = adapter.prepare_demote(lease, 0, 8)
        adapter.commit_demote(ticket)
        self.decode(adapter, 0, [1], [[4, 5]])

    def test_post_binding_store_failure_poisons_even_within_existing_page(self):
        adapter = self.adapter()
        lease = adapter.open_request(1)
        self.append(adapter, lease, 0, 1)
        pages = adapter._kv._pages.page_ids(lease, "first")
        with patch.object(adapter._kv, "set_kv_buffer", side_effect=MemoryError("pre-store allocation")):
            with self.assertRaises(MemoryError):
                self.append(adapter, lease, 0, 1)
        self.assertEqual(adapter._kv._pages.page_ids(lease, "first"), pages)
        with self.assertRaises(LayerHybridPoisoned):
            self.append(adapter, lease, 0, 1)
        with self.assertRaises(LayerHybridPoisoned):
            adapter.finish(lease)

    def test_duplicate_history_rejected_before_mutation_and_latest_allowed(self):
        adapter = self.adapter()
        first, second = adapter.open_request(1), adapter.open_request(2)
        for lease in (first, second):
            self.append(adapter, lease, 0, 21)
        adapter.commit_demote(adapter.prepare_demote(first, 0, 8))
        hot = adapter._history[first, 0].hot
        tags, lru = hot.tokens.clone(), hot.lru.clone()
        with self.assertRaisesRegex(ValueError, "Duplicate historical"):
            self.decode(adapter, 0, [2, 1], [[3, 4], [3, 3]])
        torch.testing.assert_close(hot.tokens, tags)
        torch.testing.assert_close(hot.lru, lru)
        # Latest token 20 is the only valid element of block 5, and is handled
        # directly by the native kernel, outside the history hit hash.
        self.decode(adapter, 0, [1, 2], [[5, 5], [0, 0]])

    def test_post_clear_fence_failure_does_not_release_pages(self):
        adapter = self.adapter()
        lease = adapter.open_request(1)
        self.append(adapter, lease, 0, 21)
        ticket = adapter.prepare_demote(lease, 0, 8)
        before = adapter._kv._pages.free_pages("first")
        with patch.object(adapter._kv, "_fence_before_reuse",
                          side_effect=[None, RuntimeError("fence failed")]):
            with self.assertRaisesRegex(RuntimeError, "fence failed"):
                adapter.commit_demote(ticket)
        self.assertEqual(adapter._kv._pages.free_pages("first"), before)
        with self.assertRaises(LayerHybridPoisoned):
            adapter.open_request(2)

    def test_hot_context_limit_and_logical_ownership_do_not_change_state(self):
        adapter = self.adapter()
        lease, other = adapter.open_request(1), adapter.open_request(2)
        self.append(adapter, lease, 0, 21)
        adapter.commit_demote(adapter.prepare_demote(lease, 0, 8))
        with self.assertRaisesRegex(ValueError, "already owned"):
            adapter.append(other, 0, [48], values(1), values(1))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.append(adapter, lease, 0, 2)
        while len(self.history[1, 0][0]) < 64:
            self.append(adapter, lease, 0, 1)
        before = adapter.accounting()
        with self.assertRaisesRegex(ValueError, "context capacity"):
            self.append(adapter, lease, 0, 1)
        self.assertEqual(adapter.accounting(), before)
        self.decode(adapter, 0, [1], [[15, 14]])
