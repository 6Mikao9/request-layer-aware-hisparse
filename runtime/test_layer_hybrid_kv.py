from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from runtime.layer_hybrid_kv import HostPrefillTarget, HotLayerState, LayerHybridKV, LayerHybridPoisoned
from runtime.layer_hybrid_pool import PageBudgetExceeded
from runtime.test_layer_hybrid_pool import layout


class CPUMHAPool:
    """Real CPU tensor backing with the pinned MHA constructor/write protocol."""

    def __init__(self, **config):
        self.config = config
        shape = (config["size"] + config["page_size"], config["head_num"], config["head_dim"])
        self.layers = [
            tuple(torch.zeros(shape, dtype=config["dtype"]) for _ in range(2))
            for _ in range(config["layer_num"])
        ]

    def get_kv_buffer(self, local_layer):
        return self.layers[local_layer]

    def set_kv_buffer(self, layer, locations, k, v, *, layer_id_override):
        dst_k, dst_v = self.layers[layer_id_override]
        dst_k[locations], dst_v[locations] = k, v


class CPUHostPool:
    def __init__(self, layers):
        self.k_buffer = [torch.zeros((32, 1, 4), dtype=torch.bfloat16) for _ in range(layers)]
        self.v_buffer = [torch.zeros_like(tensor) for tensor in self.k_buffer]
        self.fail_after_copy = False
        self.backups = 0

    def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, *, io_backend):
        assert io_backend == "kernel"
        self.backups += 1
        for layer, (host_k, host_v) in enumerate(zip(self.k_buffer, self.v_buffer)):
            k, v = device_pool.get_kv_buffer(layer)
            host_k[host_indices] = k[device_indices]
            host_v[host_indices] = v[device_indices]
            if self.fail_after_copy:
                raise RuntimeError("simulated partial D2H failure")


def values(count, start=1):
    return torch.arange(start, start + count * 4).reshape(count, 1, 4).bfloat16()


class LayerHybridKVTests(unittest.TestCase):
    def make_adapter(self, hybrid=True):
        return LayerHybridKV(layout(hybrid=hybrid), 64, 4, "cpu", pool_factory=CPUMHAPool)

    def test_layer_backing_uses_exact_weighted_bytes_and_group_local_writes(self):
        adapter = self.make_adapter()
        with self.assertRaisesRegex(ValueError, "padding"):
            adapter.open_request(0)
        lease = adapter.open_request(1)
        adapter.grow(lease, {"early": 2, "late": 1})
        adapter.bind_logical_slots(lease, 0, {12: 5})
        adapter.bind_logical_slots(lease, 3, {12: 2})
        for layer in range(4):
            adapter.set_kv_buffer(SimpleNamespace(layer_id=layer), torch.tensor([12]),
                                  values(1, layer + 1), values(1, layer + 50))
        key, value = adapter.get_kv_buffer(0)
        self.assertTrue(torch.equal(key[9:10], values(1, 1)))
        self.assertTrue(torch.equal(value[9:10], values(1, 50)))
        for layer in (1, 2, 3):
            key, value = adapter.get_kv_buffer(layer)
            self.assertTrue(torch.equal(key[6:7], values(1, layer + 1)))
            self.assertTrue(torch.equal(value[6:7], values(1, layer + 50)))
        buffers = [tensor for layer in range(4) for tensor in adapter.get_kv_buffer(layer)]
        self.assertEqual(len(set(tensor.data_ptr() for tensor in buffers)), 8)
        self.assertEqual(sum(tensor.numel() * tensor.element_size() for tensor in buffers),
                         adapter.layout.budget_bytes)

    def test_resident_topk_order_partial_block_and_sentinels(self):
        adapter = self.make_adapter()
        lease = adapter.open_request(2)
        adapter.grow(lease, {"early": 2})
        logical_ids = [13, 9, 12, 8, 22, 16]
        adapter.bind_logical_slots(lease, 0, dict(zip(logical_ids, range(len(logical_ids)))))
        adapter.set_kv_buffer(SimpleNamespace(layer_id=0), torch.tensor(logical_ids), values(6), values(6, 30))
        req_to_token = torch.full((4, 8), -1, dtype=torch.int32)
        req_to_token[2, :6] = torch.tensor(logical_ids)
        request = torch.tensor([2], dtype=torch.int64)
        lengths = torch.tensor([6], dtype=torch.int32)
        result = adapter.resolve_decode(0, torch.tensor([[1, 0]], dtype=torch.int32),
                                        request, lengths, req_to_token)
        self.assertEqual(result.dtype, torch.int32)
        self.assertEqual(result.tolist(), [[8, 9, -1, -1, 4, 5, 6, 7]])
        result = adapter.resolve_decode(0, torch.tensor([[-1, 0]], dtype=torch.int32),
                                        request, lengths, req_to_token)
        self.assertEqual(result.tolist(), [[-1, -1, -1, -1, 4, 5, 6, 7]])
        key, value = adapter.get_kv_buffer(0)
        self.assertTrue(torch.equal(key[result[0, 4:]], values(4)))
        self.assertTrue(torch.equal(value[result[0, 4:]], values(4, 30)))

    def test_retire_binding_and_row_reuse_keep_backing_and_drop_old_mapping(self):
        adapter = self.make_adapter()
        layer = SimpleNamespace(layer_id=0)
        lease = adapter.open_request(1)
        adapter.grow(lease, {"early": 1})
        adapter.bind_logical_slots(lease, 0, {8: 0})
        adapter.set_kv_buffer(layer, torch.tensor([8]), values(1), values(1))
        addresses = tuple(tensor.data_ptr() for tensor in adapter.get_kv_buffer(0))
        with self.assertRaisesRegex(ValueError, "same physical"):
            adapter.bind_logical_slots(lease, 0, {9: 0})
        adapter.unbind_logical_slots(lease, 0, [8])
        adapter.bind_logical_slots(lease, 0, {9: 0})
        adapter.finish(lease)
        next_lease = adapter.open_request(1)
        adapter.grow(next_lease, {"early": 1})
        self.assertEqual(tuple(tensor.data_ptr() for tensor in adapter.get_kv_buffer(0)), addresses)
        for logical in (8, 9):
            with self.assertRaisesRegex(ValueError, "Missing"):
                adapter.set_kv_buffer(layer, torch.tensor([logical]), values(1), values(1))
        with self.assertRaisesRegex(ValueError, "Stale"):
            adapter.bind_logical_slots(lease, 0, {8: 0})
        adapter.bind_logical_slots(next_lease, 0, {9: 0})
        self.assertEqual(adapter.get_prefill_mapping(0)[9].item(), 4)

    def test_resident_rejects_cross_request_selection(self):
        adapter = self.make_adapter()
        for row in (1, 2):
            lease = adapter.open_request(row)
            adapter.grow(lease, {"early": 1})
            adapter.bind_logical_slots(lease, 0, {12 + row: 0})
        req_to_token = torch.tensor([[-1], [14], [14], [-1]], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "this request"):
            adapter.resolve_decode(0, torch.tensor([[0, -1]], dtype=torch.int32),
                                   torch.tensor([1]), torch.tensor([1]), req_to_token)

    def test_tail_release_clears_only_that_groups_write_binding(self):
        adapter = self.make_adapter()
        first, second = adapter.open_request(1), adapter.open_request(2)
        adapter.grow(first, {"early": 2, "late": 1})
        adapter.bind_logical_slots(first, 0, {8: 0, 12: 4})
        adapter.bind_logical_slots(first, 1, {12: 0})
        adapter.release_tail(first, 0, 1)
        adapter.grow(second, {"early": 1})
        adapter.bind_logical_slots(second, 0, {14: 0})
        mapping = adapter.get_prefill_mapping(0)
        self.assertEqual(mapping[[8, 12, 14]].tolist(), [4, -1, 8])
        self.assertEqual(adapter.get_prefill_mapping(1)[12].item(), 4)

    def test_prefill_commit_backs_up_every_hot_layer_before_reusing_tail(self):
        adapter = self.make_adapter()
        lease = adapter.open_prefill(1)
        logical = list(range(10, 23))
        host = CPUHostPool(3)
        host_indices = torch.arange(20, 7, -1)
        target = {"late": HostPrefillTarget(host, host_indices)}
        for chunk in (logical[:6], logical[6:]):
            adapter.append_prefill(lease, chunk)
            for layer in range(4):
                offset = logical.index(chunk[0]) * 4 + layer * 50
                adapter.set_kv_buffer(SimpleNamespace(layer_id=layer), torch.tensor(chunk),
                                      values(len(chunk), offset), values(len(chunk), offset + 20))
        with self.assertRaisesRegex(ValueError, "transition"):
            adapter.release_tail(lease, 1, 3)
        hot_before = adapter.get_prefill_mapping(1)[logical].tolist()
        resident_before = adapter.get_prefill_mapping(0)[logical].tolist()
        addresses = tuple(t.data_ptr() for t in adapter.get_kv_buffer(1))
        kept = adapter.commit_prefill(lease, target)
        self.assertEqual(kept, {"late": tuple(range(4, 16))})
        self.assertEqual(host.backups, 1)
        for local, layer in enumerate((1, 2, 3)):
            self.assertTrue(torch.equal(host.k_buffer[local][host_indices], values(13, layer * 50)))
            self.assertTrue(torch.equal(host.v_buffer[local][host_indices], values(13, layer * 50 + 20)))
        self.assertEqual(adapter.get_prefill_mapping(1)[logical].tolist(), [-1] * len(logical))
        self.assertEqual(adapter.get_prefill_mapping(0)[logical].tolist(), resident_before)
        next_lease = adapter.open_request(2)
        adapter.grow(next_lease, {"late": 1})
        adapter.bind_logical_slots(next_lease, 1, {25: 0})
        self.assertEqual(adapter.get_prefill_mapping(1)[25].item(), hot_before[-1])
        self.assertEqual(tuple(t.data_ptr() for t in adapter.get_kv_buffer(1)), addresses)

    def test_partial_backup_rolls_back_extra_page_and_prefill_is_retryable(self):
        adapter = self.make_adapter()
        lease = adapter.open_prefill(1)
        other = adapter.open_request(2)
        logical = list(range(10, 15))
        adapter.append_prefill(lease, logical)
        adapter.grow(other, {"late": 1})
        host = CPUHostPool(3)
        target = {"late": HostPrefillTarget(host, torch.arange(len(logical)))}
        with self.assertRaisesRegex(ValueError, "entire prefill"):
            adapter.commit_prefill(lease, target)
        for layer in range(4):
            adapter.set_kv_buffer(SimpleNamespace(layer_id=layer), torch.tensor(logical),
                                  values(len(logical), layer), values(len(logical), layer))
        hot_before = adapter.get_prefill_mapping(1)[logical].tolist()
        host.fail_after_copy = True
        with self.assertRaisesRegex(RuntimeError, "partial D2H"):
            adapter.commit_prefill(lease, target)
        self.assertEqual(adapter.get_prefill_mapping(1)[logical].tolist(), hot_before)
        adapter.grow(other, {"late": 2})
        with self.assertRaises(PageBudgetExceeded):
            adapter.commit_prefill(lease, target)
        self.assertEqual(host.backups, 1)
        adapter.finish(other)
        host.fail_after_copy = False
        self.assertEqual(len(adapter.commit_prefill(lease, target)["late"]), 12)

    def test_prefill_group_oom_leaves_no_bindings_and_abort_reuses_entire_group(self):
        adapter = self.make_adapter()
        first = adapter.open_prefill(1)
        other = adapter.open_prefill(2)
        adapter.append_prefill(first, range(10, 23))
        with self.assertRaises(PageBudgetExceeded):
            adapter.append_prefill(other, [25])
        self.assertEqual(adapter.get_prefill_mapping(0)[25].item(), -1)
        self.assertEqual(adapter.get_prefill_mapping(1)[25].item(), -1)
        adapter.finish(first)
        adapter.append_prefill(other, [25])
        for layer in range(4):
            self.assertTrue(torch.all(adapter.get_prefill_mapping(layer)[10:23] == -1))
            self.assertEqual(adapter.get_prefill_mapping(layer)[25].item(), 4)

    def test_partial_group_binding_poisons_adapter_and_keeps_pages_reserved(self):
        adapter = self.make_adapter()
        lease = adapter.open_prefill(1)
        bind = adapter._bind_logical_slots

        def bind_then_fail(lease, layer, positions):
            if layer == 1:
                raise RuntimeError("second-group binding failed")
            bind(lease, layer, positions)

        with patch.object(adapter, "_bind_logical_slots", side_effect=bind_then_fail):
            with self.assertRaisesRegex(RuntimeError, "second-group binding"):
                adapter.append_prefill(lease, [10])
        self.assertEqual(adapter._maps["early"][10].item(), 4)
        self.assertEqual(adapter._maps["late"][10].item(), -1)
        self.assertEqual(adapter._pages.page_ids(lease, "early"), (1,))
        self.assertEqual(adapter._pages.page_ids(lease, "late"), (1,))
        for operation in (lambda: adapter.finish(lease), lambda: adapter.open_request(2),
                          lambda: adapter.get_prefill_mapping(0)):
            with self.assertRaisesRegex(LayerHybridPoisoned, "prefill binding"):
                operation()

    def test_prefill_fence_failure_does_not_retry_or_release_pages(self):
        for failure_fence in (1, 2):
            with self.subTest(failure_fence=failure_fence):
                adapter = self.make_adapter()
                lease = adapter.open_prefill(1)
                logical = list(range(10, 23))
                adapter.append_prefill(lease, logical)
                for layer in range(4):
                    adapter.set_kv_buffer(SimpleNamespace(layer_id=layer), torch.tensor(logical),
                                          values(len(logical), layer), values(len(logical), layer))
                original_pages = adapter._pages.page_ids(lease, "late")
                target = {"late": HostPrefillTarget(CPUHostPool(3), torch.arange(len(logical)))}
                failure = RuntimeError("simulated CUDA fence error")
                calls = [None] * (failure_fence - 1) + [failure, None]
                with patch.object(adapter, "_fence_before_reuse", side_effect=calls) as fence:
                    with self.assertRaisesRegex(RuntimeError, "simulated CUDA fence"):
                        adapter.commit_prefill(lease, target)
                    self.assertEqual(fence.call_count, failure_fence)
                self.assertEqual(adapter._pages.page_ids(lease, "late"), original_pages)
                with self.assertRaisesRegex(LayerHybridPoisoned, "prefill commit"):
                    adapter.finish(lease)

    def test_partial_native_write_poisons_before_progress_can_be_reused(self):
        adapter = self.make_adapter()
        lease = adapter.open_prefill(1)
        adapter.append_prefill(lease, [10])
        pool = adapter._pools["early"]

        def write_k_then_fail(layer, locations, k, v, *, layer_id_override):
            pool.get_kv_buffer(layer_id_override)[0][locations] = k
            raise RuntimeError("native V store failed")

        with patch.object(pool, "set_kv_buffer", side_effect=write_k_then_fail):
            with self.assertRaisesRegex(RuntimeError, "native V store"):
                adapter.set_kv_buffer(SimpleNamespace(layer_id=0), torch.tensor([10]), values(1), values(1))
        self.assertTrue(torch.equal(pool.get_kv_buffer(0)[0][4:5], values(1)))
        self.assertEqual(adapter._prefill[lease].written, {})
        with self.assertRaisesRegex(LayerHybridPoisoned, "native K/V store"):
            adapter.finish(lease)

    def test_all_hot_and_hybrid_hot_use_same_dispatch_and_fail_closed_without_state(self):
        topk = torch.tensor([[0, -1]], dtype=torch.int32)
        req = torch.tensor([1])
        lengths = torch.tensor([1])
        table = torch.full((4, 1), -1, dtype=torch.int32)
        for hybrid, layer in ((True, 3), (False, 0), (False, 3)):
            adapter = self.make_adapter(hybrid)
            lease = adapter.open_request(1)
            with self.assertRaisesRegex(RuntimeError, "requires group-local"):
                adapter.resolve_decode(layer, topk, req, lengths, table)
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                adapter.resolve_decode(layer, topk, req, lengths, table, hot_state=object())
            with patch.object(adapter, "_swap_hot", return_value=topk) as dispatch:
                state = object()
                self.assertIs(adapter.resolve_decode(layer, topk, req, lengths, table, hot_state=state), topk)
                dispatch.assert_called_once_with(
                    layer, adapter.layout.groups[0 if layer == 0 else 1], topk, req, lengths, [lease], state
                )

    def test_fused_index_preflight_checks_short_latest_lru_and_host_bounds(self):
        adapter = self.make_adapter()
        short, long = adapter.open_request(1), adapter.open_request(2)
        adapter.grow(short, {"late": 1})
        adapter.grow(long, {"late": 3})
        locations = torch.zeros((4, 12), dtype=torch.int32)
        locations[1, :4] = torch.arange(4, 8)
        locations[2] = torch.arange(8, 20)
        lru = torch.arange(8, dtype=torch.int16).repeat(4, 1)
        tags = torch.full_like(locations, -1)
        host_locations = torch.full((4, 9), -1, dtype=torch.int64)
        host_locations[2, :8] = torch.arange(8)
        host = values(8)
        state = HotLayerState(tags, locations, lru, host_locations, host, host, lambda: None)
        topk = torch.tensor([[0, 1], [0, 2]], dtype=torch.int32)
        rows = torch.tensor([1, 2])
        lengths = torch.tensor([3, 9])
        group = adapter.layout.groups[1]
        adapter._validate_hot_indices(3, group, topk, rows, lengths, [short, long], state)
        locations[2, 8] = 4
        with self.assertRaisesRegex(ValueError, "page owner"):
            adapter._validate_hot_indices(3, group, topk, rows, lengths, [short, long], state)
        locations[2, 8] = 16
        lru[2, 0] = 1
        with self.assertRaisesRegex(ValueError, "permute"):
            adapter._validate_hot_indices(3, group, topk, rows, lengths, [short, long], state)
        lru[2, 0] = 0
        host_locations[2, 0] = -1
        with self.assertRaisesRegex(ValueError, "historical host"):
            adapter._validate_hot_indices(3, group, topk, rows, lengths, [short, long], state)


if __name__ == "__main__":
    unittest.main()
