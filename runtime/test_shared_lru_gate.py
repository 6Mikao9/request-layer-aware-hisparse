"""CPU self-test of the CUDA gate harness with a table-level emulator of the kernel ABI.

``runtime.cuda_shared_lru_correctness`` compares the CUDA kernel with
``KernelMirror``. Here the JIT module is replaced by an emulator that reads and
writes the very tensors the kernel receives, so the adapter's argument
plumbing, the slot tables and the gate's own bookkeeping are exercised on CPU.
A failure on the A100 then points at the kernel, not at the harness.
"""
import unittest
from unittest.mock import patch

from runtime import shared_lru as reference

try:
    import torch
except ImportError:
    torch = None


class Emulator:
    """Implements ``module.resolve(*args)`` by running KernelMirror on the passed tensors."""

    def __init__(self, plan, num_top_k, sparse_block_size, bypass, features=()):
        self.plan, self.num_top_k, self.block, self.bypass = plan, num_top_k, sparse_block_size, bypass
        self.features = features

    def resolve(self, top_k, rpi, seq, num_real, row_token, row_owner, order, occupancy, floors, generation,
                device_locs, host_locs, host_k, host_v, arena_k, arena_v, out, status, counters, owner_summary,
                plane, num_planes, hot_rows, plane_rows, order_offset, order_size, scratch_offset,
                scratch_size, item_size_bytes):
        plan, width = self.plan, self.num_top_k * self.block
        assert (order_offset, order_size) == plan.domain(plane) and (scratch_offset, scratch_size) == plan.scratch
        assert self.bypass == (plane in plan.bypass) and floors.tolist() == list(plan.floors)
        assert ("owner_summary" in self.features) == bool(owner_summary.numel())
        mirror = reference.KernelMirror(plan, plane_rows=plane_rows, features=self.features)
        flat_k, flat_v = arena_k.view(-1, *arena_k.shape[2:]), arena_v.view(-1, *arena_v.shape[2:])
        for b in range(top_k.shape[0]):
            if b >= int(num_real[0]):
                out[b, :width] = -1
                continue
            rid = int(rpi[b])
            mirror.slots[rid] = dict(
                gen=int(generation[rid]), locs=device_locs[rid].tolist(), owner=row_owner[rid].tolist(),
                token=row_token[:, rid, :hot_rows].reshape(-1).tolist(), order=order[rid].tolist(),
                occ=occupancy[rid].tolist())
            if owner_summary.numel():
                mirror.slots[rid]["summary"] = [w & (2**64 - 1) for w in owner_summary[rid].tolist()]
            flat, copies, _ = mirror.resolve_row(rid, plane, top_k[b].tolist(), self.block, int(seq[b]))
            for row, token in copies:
                index = mirror.flat(rid, row)
                flat_k[index].copy_(host_k[int(host_locs[rid, token])])
                flat_v[index].copy_(host_v[int(host_locs[rid, token])])
            state = mirror.slots[rid]
            out[b, :width] = torch.tensor(flat, dtype=torch.int32)
            row_owner[rid] = torch.tensor(state["owner"], dtype=torch.int32)
            order[rid] = torch.tensor(state["order"], dtype=torch.int32)
            occupancy[rid] = torch.tensor(state["occ"], dtype=torch.int32)
            row_token[:, rid, :hot_rows] = torch.tensor(state["token"], dtype=torch.int32).reshape(num_planes, -1)
            if owner_summary.numel():
                owner_summary[rid] = torch.tensor([w - 2**64 if w >> 63 else w for w in state["summary"]],
                                                  dtype=torch.int64)
        status.fill_(mirror.status)


@unittest.skipUnless(torch is not None, "CPU torch required")
class GateHarness(unittest.TestCase):
    def run_arms(self, features, dtypes, steps):
        from runtime import cuda_shared_lru_correctness as gate
        from runtime import shared_lru_device as device
        launches = 0
        for index, (name, pool) in enumerate(gate.ARMS.items()):
            plans = {}

            def jit(block_size, num_top_k, sparse_block_size, fifo, soft_floor, bypass, counters, owner_summary,
                    touched_bitmap):
                self.assertEqual((owner_summary, touched_bitmap),
                                 ("owner_summary" in features, "touched_bitmap" in features))
                return Emulator(plans["plan"], num_top_k, sparse_block_size, bypass, features)

            original = device.ArenaRuntime.__init__

            def capture_plan(self, *args, **kwargs):
                original(self, *args, **kwargs)
                plans["plan"] = self.plan

            for seq_dtype, rpi_dtype in dtypes:
                with self.subTest(arm=name, seq=seq_dtype, features=features), \
                        patch.object(device, "_jit_module", jit), \
                        patch.object(device.ArenaRuntime, "__init__", capture_plan):
                    launches += gate.run_case(torch, pool, counters=False, seq_dtype=seq_dtype, rpi_dtype=rpi_dtype,
                                              graph=False, seed=7 + index, steps=steps, device="cpu",
                                              features=features)
        return launches

    def test_eager_cases_pass_against_the_emulated_abi(self):
        launches = self.run_arms((), ((torch.int32, torch.int64), (torch.int64, torch.int32)), 12)
        self.assertGreater(launches, 100)

    def test_every_switch_set_passes_against_the_emulated_abi(self):
        from runtime import cuda_shared_lru_correctness as gate
        for name, features in gate.FEATURE_SETS.items():
            if features:
                with self.subTest(features=name):
                    self.assertGreater(self.run_arms(features, ((torch.int32, torch.int64),), 10), 50)


if __name__ == "__main__":
    unittest.main()
