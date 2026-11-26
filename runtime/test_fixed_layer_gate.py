"""CPU self-test of the fixed-layer CUDA gate with emulated swap, split-copy, resolver and consumer kernels.

``runtime.cuda_fixed_layer_correctness`` runs the pinned swap kernels, the repository's
resident resolver and the pinned Triton consumer. Here each is an emulator that reads and
writes the tensors the kernel receives. The swap follows ``selection_oracle.reference_lru``
(the pinned batch-LRU transition) and the pinned output conventions: short requests read
in place, the newest token from the tail row, -1 for invalid positions and padded rows.
Injected faults show that the gate's checks catch them, so a failure on the A100 points at a
kernel rather than the harness.
"""
import sys
import types
import unittest
from unittest.mock import patch

from runtime.selection_oracle import reference_lru
from runtime.test_combined_gate import emulated_consumer, emulated_resident_resolver

try:
    import torch
except ImportError:
    torch = None


def native_swap(top_k, tokens, host_locs, locs, host_k, host_v, dev_k, dev_v, out, rpi, seq, lru, num_real, page,
                item, src=None, dst=None, count=None, *, block, skip_io=False, newest_from_host=False):
    """``load_cache_to_device_buffer`` with the pinned positional arguments; records the miss plan if given."""
    hot = lru.shape[1]
    for b in range(top_k.shape[0]):
        if b >= int(num_real[0]):
            out[b] = -1
            continue
        r, s = int(rpi[b]), int(seq[b])
        positions = [x * block + n if x >= 0 else -1 for x in top_k[b].tolist() for n in range(block)]
        if s <= hot:
            out[b] = torch.tensor([int(locs[r, t]) if 0 <= t < s else -1 for t in positions])
            if count is not None:
                count[b] = 0
            continue
        newest = -1 if newest_from_host else s - 1
        wanted = [t for t in positions if 0 <= t < s and t != newest]
        tags = tokens[r, :hot].tolist()
        after_tags, after_order, slots, victims = reference_lru(tags, lru[r].tolist(), wanted)
        found, present = dict(zip(wanted, slots)), set(tags)
        out[b] = torch.tensor([-1 if not 0 <= t < s else int(locs[r, hot]) if t == newest else int(locs[r, found[t]])
                               for t in positions])
        tokens[r, :hot], lru[r] = torch.tensor(after_tags), torch.tensor(after_order, dtype=torch.int16)
        for m, (token, victim) in enumerate(zip([t for t in wanted if t not in present], victims)):
            source, target = int(host_locs[r, token]), int(locs[r, victim])
            if src is not None:
                src[b, m], dst[b, m] = source, target
            if not skip_io:
                dev_k[target], dev_v[target] = host_k[source], host_v[source]
        if count is not None:
            count[b] = len(victims)
def emulated_swap(**faults):
    """The pinned ``HiSparseCoordinator.swap_in_selected_blocks`` around the emulated fused kernel."""

    def swap(c, rpi, seq, blocks, layer_id, block):
        plane = layer_id - c.mem_pool_device.start_layer
        out = c.top_k_device_locs_buffer[:rpi.shape[0], :blocks.shape[1] * block]
        native_swap(blocks, c.req_device_buffer_tokens[plane], c.req_to_host_pool,
                    c.req_device_buffer_token_locs[plane],
                    c.mem_pool_host.k_buffer[plane], c.mem_pool_host.v_buffer[plane],
                    c.mem_pool_device.get_key_buffer(layer_id), c.mem_pool_device.get_value_buffer(layer_id), out, rpi,
                    seq, c.lru_slots[plane], c.num_real_reqs, 1, c.item_size_bytes, block=block, **faults)
        return out

    return swap


def split_modules(*, drop_last=False):
    """``sglang.kernels.ops.kvcache.hisparse`` with the planner and copier ``PlacementBinding.split_copy`` loads."""
    module = types.ModuleType("sglang.kernels.ops.kvcache.hisparse")

    def planner(item, block_size, num_top_k, hot, *, top_k_block_size, record_miss_plan, skip_io, **flags):
        assert record_miss_plan and skip_io, "the split planner records the plan and copies nothing"
        return types.SimpleNamespace(load_cache_to_device_buffer=lambda *args: native_swap(
            *args, block=top_k_block_size, skip_io=True))

    def copy(src, dst, count, num_real, host_k, host_v, dev_k, dev_v, ctas, item):
        for b in range(int(num_real[0])):
            for m in range(int(count[b]) - drop_last):
                dev_k[int(dst[b, m])], dev_v[int(dst[b, m])] = host_k[int(src[b, m])], host_v[int(src[b, m])]

    module._jit_sparse_module = planner
    module._jit_copy_planned_module = lambda *template: types.SimpleNamespace(copy_cache_planned=copy)
    parents = ("sglang", "sglang.kernels", "sglang.kernels.ops", "sglang.kernels.ops.kvcache")
    return {**{name: types.ModuleType(name) for name in parents}, module.__name__: module}


@unittest.skipUnless(torch is not None, "CPU torch required")
class FixedLayerGateHarness(unittest.TestCase):
    def run_gate(self, *, steps=6, swap=None, split=None, consumer=None, resolver=None):
        from runtime import cuda_fixed_layer_correctness as gate
        from runtime import layer_placement_device as adapter
        kernels = dict(swap=swap or emulated_swap(), consumer=consumer or emulated_consumer())
        module = resolver or emulated_resident_resolver()
        with patch.dict(sys.modules, split or split_modules()), \
                patch.object(adapter, "_resident_module", lambda blocks, block: module):
            return gate.run_case(torch, geometry=gate.SMALL, offload=(1, 3), graph=False, seq_dtype=torch.int32,
                                 rpi_dtype=torch.int64, seed=5, steps=steps, kernels=kernels, device="cpu")

    def test_every_arm_equals_stock_on_emulated_kernels(self):
        from runtime import cuda_fixed_layer_correctness as gate
        self.assertEqual(self.run_gate(steps=8), 8 * gate.SMALL.layers * (1 + len(gate.ARMS)))

    def test_a_newest_token_read_from_the_host_is_caught(self):
        with self.assertRaisesRegex(AssertionError, "stock layer \\d+ row \\d+: max error"):
            self.run_gate(swap=emulated_swap(newest_from_host=True))

    def test_a_copy_that_drops_a_miss_is_caught(self):
        with self.assertRaisesRegex(AssertionError, "split layer [13] row \\d+: output differs from stock"):
            self.run_gate(split=split_modules(drop_last=True))

    def test_a_wrong_s_plane_is_caught(self):
        from runtime import layer_placement_device as adapter
        with patch.object(adapter.PlacementBinding, "plane", lambda self, layer: 1 - self.plane_of[layer]), \
                self.assertRaisesRegex(AssertionError, "direct layer 1 row 0: output differs from stock"):
            self.run_gate(steps=1)

    def test_a_misrouted_resident_read_is_caught(self):
        with self.assertRaisesRegex(AssertionError, "direct layer 0 row 0: output differs from stock"):
            self.run_gate(steps=1, consumer=emulated_consumer(shift_resident=1))

    def test_a_resident_resolver_that_reads_past_seq_len_is_caught(self):
        with self.assertRaisesRegex(AssertionError, "resident resolver slots differ"):
            self.run_gate(steps=4, resolver=emulated_resident_resolver(ignore_seq_len=True))

    def test_production_list_is_the_calibrated_t40(self):
        from runtime import cuda_fixed_layer_correctness as gate
        offload = gate.calibrated_t40()
        self.assertEqual((len(offload), offload[:3], offload[-3:]), (40, (0, 1, 2), (57, 58, 59)))
        self.assertLessEqual(gate.PRODUCTION.blocks * gate.PRODUCTION.block, gate.PRODUCTION.hot)


if __name__ == "__main__":
    unittest.main()
