from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from runtime.layer_eviction import RequestLayerEvictionKV
from runtime.layer_eviction_scheduler import LayerEvictionSchedulerBridge
from runtime.layer_eviction_tp import LayerEvictionTP, DemotionProtocolError
from runtime.layer_hybrid_kv import LayerHybridPoisoned
from runtime.layer_hybrid_pool import LayerGroup, LayerLayout, PageBudgetExceeded
from runtime.test_layer_hybrid_kv import CPUMHAPool, values


class LocalControl:
    rank, world_size = 0, 1

    def all_gather(self, value):
        return [value]


class SchedulerTests(unittest.TestCase):
    def build(self, pages=16):
        groups = tuple(LayerGroup(f"l{i}", (i,), "resident", pages) for i in range(2))
        layout = LayerLayout(2, 4, 4, 16, 2, 4, groups, 2*(pages+1)*64)
        kv = RequestLayerEvictionKV(layout, 256, 4, "cpu", max_context=64,
                                    host_budget_bytes=8192, pool_factory=CPUMHAPool)
        table = torch.full((4, 64), -1, dtype=torch.int32)
        tp = LayerEvictionTP(kv, LocalControl(), config_digest="test-config")
        return LayerEvictionSchedulerBridge(kv, table, tp_controller=tp)

    def batch(self, bridge, leases, chunks, mode="prefill"):
        for lease, chunk in zip(leases, chunks):
            old = bridge.adapter._state(lease, 0).logical
            bridge.req_to_token[lease.row, :len(old)+len(chunk)] = torch.tensor(old+list(chunk))
        return bridge.begin_forward(leases, chunks, mode=mode)

    def write(self, bridge, ticket, layer):
        if not bridge._submitted:
            bridge.begin_model(ticket)
        ids = [i for chunk in ticket.logical_chunks for i in chunk]
        k = torch.tensor(ids).reshape(-1, 1, 1).repeat(1, 1, 4).bfloat16() + layer
        bridge.set_kv_buffer(ticket, SimpleNamespace(layer_id=layer), torch.tensor(ids), k, -k)

    def prefill(self, bridge, lease, start=4, count=21):
        ticket = self.batch(bridge, [lease], [tuple(range(start, start+count))])
        for layer in range(2):
            self.write(bridge, ticket, layer)
            mapping = bridge.get_prefill_mapping(ticket, layer)
            self.assertTrue(torch.all(mapping[start:start+count] > 0))
        bridge.end_forward(ticket)

    def test_prefill_demotion_mixed_decode_and_row_reuse(self):
        bridge = self.build()
        a, b = bridge.open_request(1), bridge.open_request(2)
        self.prefill(bridge, a)
        self.prefill(bridge, b, 64)
        result = bridge.demote(0, a, 0, 8)
        self.assertEqual(result.reclaimed_bytes_by_rank, (192,))
        ticket = self.batch(bridge, [b, a], [(85,), (25,)], mode="decode")
        for layer in range(2):
            self.write(bridge, ticket, layer)
            blocks = torch.tensor([[5, 0], [4, 5]], dtype=torch.int32)
            locations = bridge.swap_in_selected_blocks(ticket, torch.tensor([2, 1]),
                torch.tensor([22, 22]), blocks, layer, 4)
            k, v = bridge.get_kv_buffer(layer)
            for row, start in enumerate((64, 4)):
                for c, loc in enumerate(locations[row].tolist()):
                    pos = blocks[row, c//4].item()*4+c%4
                    if pos < 22:
                        self.assertEqual(k[loc, 0, 0].item(), start+pos+layer)
                        self.assertEqual(v[loc, 0, 0].item(), -(start+pos+layer))
                    else:
                        self.assertEqual(loc, -1)
        bridge.end_forward(ticket)
        bridge.finish(a)
        new = bridge.open_request(1)
        self.assertNotEqual(a.generation, new.generation)
        with self.assertRaises(ValueError):
            bridge.finish(a)

    def test_aggregate_batch_oom_is_atomic(self):
        bridge = self.build(pages=6)
        a, b = bridge.open_request(1), bridge.open_request(2)
        before = bridge.adapter.accounting()
        with self.assertRaises(PageBudgetExceeded):
            self.batch(bridge, [a, b], [tuple(range(4, 20)), tuple(range(64, 80))])
        self.assertEqual(before, bridge.adapter.accounting())
        self.assertEqual(bridge.adapter._state(a, 0).logical, [])

    def test_cancel_before_writes_reclaims_all_layers_and_stale_ticket_fails(self):
        bridge = self.build()
        a = bridge.open_request(1)
        before = bridge.adapter.accounting()
        ticket = self.batch(bridge, [a], [tuple(range(4, 25))])
        with self.assertRaisesRegex(RuntimeError, "Forward is active"):
            bridge.finish(a)
        bridge.abort_forward(ticket)
        self.assertEqual(before, bridge.adapter.accounting())
        with self.assertRaises(ValueError):
            self.write(bridge, ticket, 0)

    def test_partial_forward_cannot_be_retired_or_rolled_back(self):
        bridge = self.build()
        a = bridge.open_request(1)
        ticket = self.batch(bridge, [a], [(4, 5)])
        self.write(bridge, ticket, 0)
        with self.assertRaisesRegex(RuntimeError, "missing layers"):
            bridge.end_forward(ticket)
        with self.assertRaisesRegex(RuntimeError, "submitted model"):
            bridge.abort_forward(ticket)
        with self.assertRaises(LayerHybridPoisoned):
            bridge.finish(a)

    def test_wrong_writer_order_and_layer_duplication(self):
        bridge = self.build()
        a = bridge.open_request(1)
        ticket = self.batch(bridge, [a], [(4, 5)])
        bridge.begin_model(ticket)
        with self.assertRaisesRegex(ValueError, "Writer order"):
            bridge.set_kv_buffer(ticket, SimpleNamespace(layer_id=0), torch.tensor([5, 4]),
                                 values(2), values(2))
        self.write(bridge, ticket, 0)
        with self.assertRaisesRegex(ValueError, "already submitted"):
            self.write(bridge, ticket, 0)
        self.write(bridge, ticket, 1)
        bridge.end_forward(ticket)

    def test_failed_writer_and_completion_fence_fail_closed(self):
        for failure in ("writer", "fence"):
            with self.subTest(failure=failure):
                bridge = self.build()
                a = bridge.open_request(1)
                ticket = self.batch(bridge, [a], [(4, 5)])
                if failure == "writer":
                    with patch.object(bridge.adapter, "append", side_effect=RuntimeError("write failed")):
                        with self.assertRaises(RuntimeError):
                            self.write(bridge, ticket, 0)
                else:
                    for layer in range(2):
                        self.write(bridge, ticket, layer)
                    with patch.object(bridge.adapter._kv, "_fence_before_reuse", side_effect=RuntimeError("fence")):
                        with self.assertRaises(RuntimeError):
                            bridge.end_forward(ticket)
                with self.assertRaises(LayerHybridPoisoned):
                    bridge.open_request(2)

    def test_native_table_mismatch_and_hot_extend_rejected_before_reservation(self):
        bridge = self.build()
        a = bridge.open_request(1)
        before = bridge.adapter.accounting()
        with self.assertRaisesRegex(ValueError, "Native request table"):
            bridge.begin_forward([a], [(4,)], mode="prefill")
        self.assertEqual(before, bridge.adapter.accounting())
        self.prefill(bridge, a)
        bridge.demote(0, a, 0, 8)
        before = bridge.adapter.accounting()
        with self.assertRaisesRegex(ValueError, "cannot enter prefill"):
            self.batch(bridge, [a], [(25, 26)], mode="prefill")
        self.assertEqual(before, bridge.adapter.accounting())

    def test_active_forward_votes_rejection_instead_of_skipping_tp_collective(self):
        bridge = self.build()
        a = bridge.open_request(1)
        self.batch(bridge, [a], [(4,)])
        transport = bridge.tp._transport
        with patch.object(transport, "all_gather", wraps=transport.all_gather) as gather:
            with self.assertRaises(DemotionProtocolError):
                bridge.demote(0, a, 0, 8)
            self.assertEqual(gather.call_count, 1)
        with self.assertRaises(DemotionProtocolError):
            bridge.open_request(2)

    def test_model_submission_before_first_main_kv_write_cannot_roll_back(self):
        bridge = self.build()
        a = bridge.open_request(1)
        ticket = self.batch(bridge, [a], [(4,)])
        bridge.begin_model(ticket)
        with self.assertRaisesRegex(RuntimeError, "submitted model"):
            bridge.abort_forward(ticket)
        with self.assertRaises(LayerHybridPoisoned):
            bridge.open_request(2)
