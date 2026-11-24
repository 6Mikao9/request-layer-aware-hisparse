"""CPU self-test of the combined CUDA gate with emulated resolver and consumer kernels.

``runtime.cuda_combined_correctness`` runs the SM80 resolver and the pinned Triton
consumer. Here both are replaced by emulators that read and write the very tensors
the kernels receive, so the routing, twins, references and bookkeeping of the gate
are exercised on CPU. A failure on the A100 then points at a kernel, not the harness.
"""
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


def emulated_consumer(*, shift_resident=0):
    """``flash_decode_with_gqa_share_sparse`` addressing: HiSparse slots, paged-contiguous or gathered rows."""

    def consumer(*, q, sink, k_cache, v_cache, req_to_token, seq_lens, slot_ids, block_size, topk_idx,
                 page_size=0, hisparse_slots=None):
        out, max_slots = torch.zeros(q.shape, dtype=q.dtype), k_cache.shape[0]
        for b in range(q.shape[0]):
            seq, sid, rows = int(seq_lens[b]), int(slot_ids[b]), []
            blocks = topk_idx[0, b].tolist()
            for j, block in enumerate(blocks[:sum(x >= 0 for x in blocks)]):
                for n in range(block_size):
                    position = block * block_size + n
                    if position >= seq:
                        continue
                    if hisparse_slots is not None:
                        slot = int(hisparse_slots[0, b, j * block_size + n])
                    elif page_size >= block_size:
                        slot = int(req_to_token[sid, block * block_size]) + n + shift_resident
                    else:
                        slot = int(req_to_token[sid, position])
                    rows.append(slot % max_slots)
            if rows:
                keys, values = k_cache[rows, 0].float(), v_cache[rows, 0].float()
                weights = torch.softmax((q[b].float() @ keys.T) * q.shape[-1] ** -0.5, dim=-1)
                out[b] = (weights @ values).to(q.dtype)
        return out

    return consumer


@unittest.skipUnless(torch is not None, "CPU torch required")
class CombinedGateHarness(unittest.TestCase):
    def run_arms(self, consumer, arms=None, steps=8):
        from runtime import cuda_combined_correctness as gate
        from runtime import shared_lru_device as device
        from runtime.test_shared_lru_gate import Emulator
        plans, launches = {}, 0

        def jit(block_size, num_top_k, sparse_block_size, fifo, soft_floor, bypass, counters):
            return Emulator(plans["plan"], num_top_k, sparse_block_size, bypass)

        original = device.ArenaRuntime.__init__

        def capture_plan(self, *args, **kwargs):
            original(self, *args, **kwargs)
            plans["plan"] = self.plan

        for index, (name, pool) in enumerate((arms or gate.ARMS).items()):
            with patch.object(device, "_jit_module", jit), patch.object(device.ArenaRuntime, "__init__", capture_plan):
                launches += gate.run_case(torch, pool, seq_dtype=torch.int32, rpi_dtype=torch.int64, graph=False,
                                          seed=11 + index, steps=steps, consumer=consumer, device="cpu")
        return launches

    def test_eager_cases_pass_against_emulated_kernels(self):
        from runtime import cuda_combined_correctness as gate
        self.assertEqual(self.run_arms(emulated_consumer()), len(gate.ARMS) * 8 * len(gate.LAYERS))

    def test_a_misrouted_resident_read_is_caught(self):
        from runtime import cuda_combined_correctness as gate
        with self.assertRaisesRegex(AssertionError, "layer 0 row 0: dispatched read differs"):
            self.run_arms(emulated_consumer(shift_resident=1), arms={"shared": gate.ARMS["shared"]}, steps=1)


if __name__ == "__main__":
    unittest.main()
