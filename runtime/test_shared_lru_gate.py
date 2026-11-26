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

    def resolve(self, *args):
        from runtime.shared_lru_device import RESOLVE_INTS, RESOLVE_TENSORS, ring_launch_args
        assert len(args) == len(RESOLVE_TENSORS) + len(RESOLVE_INTS)
        a = dict(zip(RESOLVE_TENSORS + RESOLVE_INTS, args))
        plan, width, plane, hot_rows = self.plan, self.num_top_k * self.block, a["plane"], a["hot_rows"]
        assert (a["order_offset"], a["order_size"]) == plan.domain(plane)
        assert (a["scratch_offset"], a["scratch_size"]) == plan.scratch
        assert self.bypass == (plane in plan.bypass) and a["floors"].tolist() == list(plan.floors)
        assert ("owner_summary" in self.features) == bool(a["owner_summary"].numel())
        assert ("hit_index" in self.features) == bool(a["index"].numel())
        incremental = "incremental_lru" in self.features
        assert incremental == bool(a["ring"].numel()) != bool(a["order"].numel())
        assert tuple(a[name] for name in RESOLVE_INTS[-5:]) == ring_launch_args(plan, plane, incremental)
        mirror = reference.KernelMirror(plan, plane_rows=a["plane_rows"], features=self.features)
        if "hit_index" in self.features:
            assert (a["index_half"], a["row_bits"]) == (mirror.index_half, mirror.row_bits)
        rings = mirror.rings if incremental else ()
        out, row_token = a["out"], a["row_token"]
        flat_k, flat_v = a["arena_k"].view(-1, *a["arena_k"].shape[2:]), a["arena_v"].view(-1, *a["arena_v"].shape[2:])
        for b in range(a["top_k"].shape[0]):
            if b >= int(a["num_real_reqs"][0]):
                out[b, :width] = -1
                continue
            rid = int(a["req_pool_indices"][b])
            state = mirror.slots[rid] = dict(
                gen=int(a["generation"][rid]), locs=a["device_locs"][rid].tolist(), owner=a["row_owner"][rid].tolist(),
                token=row_token[:, rid, :hot_rows].reshape(-1).tolist(), occ=a["occupancy"][rid].tolist())
            if incremental:
                meta = a["ring_meta"][rid].tolist()
                state.update(ring=a["ring"][rid].tolist(), pos=a["ring_pos"][rid].tolist(),
                             dead=reference.ring_dead_bits(a["ring_dead"][rid].tolist(), rings),
                             meta=[meta[4 * d:4 * d + 3] for d in range(len(rings))])
            else:
                state["order"] = a["order"][rid].tolist()
            if a["owner_summary"].numel():
                state["summary"] = [w & (2**64 - 1) for w in a["owner_summary"][rid].tolist()]
            if a["index"].numel():
                flat = a["index"][rid].tolist()
                state.update(index=[flat[i:i + reference.INDEX_WAYS] for i in range(0, len(flat), reference.INDEX_WAYS)],
                             index_flag=int(a["index_flags"][rid]))
            flat, copies, _ = mirror.resolve_row(rid, plane, a["top_k"][b].tolist(), self.block, int(a["seq_lens"][b]))
            for row, token in copies:
                index = mirror.flat(rid, row)
                flat_k[index].copy_(a["host_k"][int(a["host_locs"][rid, token])])
                flat_v[index].copy_(a["host_v"][int(a["host_locs"][rid, token])])
            out[b, :width] = torch.tensor(flat, dtype=torch.int32)
            a["row_owner"][rid] = torch.tensor(state["owner"], dtype=a["row_owner"].dtype)
            a["occupancy"][rid] = torch.tensor(state["occ"], dtype=torch.int32)
            row_token[:, rid, :hot_rows] = torch.tensor(state["token"], dtype=torch.int32).reshape(a["num_planes"], -1)
            if incremental:
                a["ring"][rid] = torch.tensor(state["ring"], dtype=torch.int32)
                a["ring_pos"][rid] = torch.tensor(state["pos"], dtype=torch.int32)
                a["ring_dead"][rid] = torch.tensor(reference.ring_dead_words(state["dead"], rings), dtype=torch.int32)
                a["ring_meta"][rid] = torch.tensor([v for m in state["meta"] for v in (*m, 0)], dtype=torch.int32)
            else:
                a["order"][rid] = torch.tensor(state["order"], dtype=torch.int32)
            if a["owner_summary"].numel():
                a["owner_summary"][rid] = torch.tensor([w - 2**64 if w >> 63 else w for w in state["summary"]],
                                                       dtype=torch.int64)
            if a["index"].numel():
                a["index"][rid] = torch.tensor(mirror.index_entries(rid), dtype=torch.int32)
                a["index_flags"][rid] = state["index_flag"]
        a["status"].fill_(mirror.status)


@unittest.skipUnless(torch is not None, "CPU torch required")
class GateHarness(unittest.TestCase):
    def run_arms(self, features, dtypes, steps):
        from runtime import cuda_shared_lru_correctness as gate
        from runtime import shared_lru_device as device
        launches = 0
        for index, (name, pool) in enumerate(gate.ARMS.items()):
            plans = {}

            def jit(block_size, num_top_k, sparse_block_size, fifo, soft_floor, bypass, counters, *switches):
                self.assertEqual(switches, tuple(name in features for name in device.RESOLVER_SWITCHES))
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
