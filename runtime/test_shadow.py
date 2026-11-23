"""CPU tensor tests for diagnostic isolation, not GPU/model equivalence evidence."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from runtime.policy import StaticPolicy
from runtime.shadow import ShadowEngine, byte_equal, require_identity, span_bytes


class ShadowTests(unittest.TestCase):
    def test_only_identity(self):
        require_identity(StaticPolicy(64, (64,) * 60))
        with self.assertRaises(ValueError):
            require_identity(StaticPolicy(64, (32,) * 60))

    def test_preserve_strides_and_nan_payload_bytes(self):
        with TemporaryDirectory() as folder:
            engine = ShadowEngine(torch, folder, 0, 4, synchronize=lambda: None, allocated_bytes=lambda: 0)
            source = torch.arange(45, dtype=torch.float32).reshape(3, 15)[:, :7]
            source[1, 2] = float("nan")
            cloned = engine.copy("fixture", source)
            self.assertEqual(cloned.stride(), source.stride())
            self.assertEqual(span_bytes(cloned), 37 * 4)
            self.assertTrue(byte_equal(torch, source, cloned))
            self.assertFalse(torch.equal(source, cloned))
            self.assertNotEqual(source.data_ptr(), cloned.data_ptr())
            pointer = cloned.data_ptr()
            self.assertEqual(engine.copy("fixture", source).data_ptr(), pointer)

    def fixtures(self):
        table = lambda: torch.zeros((2, 6), dtype=torch.int32)[:, :4]
        return dict(device_buffer_tokens=table(), lru_slots=table(), top_k_device_locs=table(),
                    device_buffer_k=torch.zeros((20, 4)), device_buffer_v=torch.zeros((20, 4)),
                    top_k_blocks=table(), host_cache_locs=table(), device_buffer_locs=table(),
                    req_pool_indices=torch.tensor([0]), seq_lens=torch.tensor([80]),
                    num_real_reqs=torch.tensor([1]), host_cache_k=torch.ones((20, 4)),
                    host_cache_v=torch.ones((20, 4)), active_hot_buffer_size=64,
                    hot_buffer_size=64, skip_io=False)

    @staticmethod
    def op(**kwargs):
        kwargs["device_buffer_tokens"].add_(1)
        kwargs["lru_slots"].add_(2)
        kwargs["top_k_device_locs"].fill_(3)
        kwargs["device_buffer_k"].copy_(kwargs["host_cache_k"])
        kwargs["device_buffer_v"].copy_(kwargs["host_cache_v"])

    def test_live_commits_once_shadow_has_same_prestate_and_no_aliases(self):
        with TemporaryDirectory() as folder:
            engine = ShadowEngine(torch, folder, 1, 4, synchronize=lambda: None, allocated_bytes=lambda: 123)
            inputs = self.fixtures()
            def stock(**kwargs):
                self.assertNotIn("active_hot_buffer_size", kwargs)
                for name in ("device_buffer_k", "device_buffer_tokens", "req_pool_indices"):
                    self.assertNotEqual(inputs[name].data_ptr(), kwargs[name].data_ptr())
                self.assertTrue(torch.all(kwargs["device_buffer_tokens"] == 0))
                self.op(**kwargs)
            engine.check(59, inputs, self.op, stock, final_layer=59)
            self.assertTrue(torch.all(inputs["device_buffer_tokens"] == 1))
            report = json.loads((Path(folder) / "rank-1.json").read_text())
            self.assertEqual(report["checked_calls"], 1)
            self.assertEqual(report["mismatches"], 0)
            self.assertEqual(report["layers"][0]["seq_min"], 80)
            self.assertNotIn("input_ids", report)
            with self.assertRaises(FileExistsError):
                ShadowEngine(torch, folder, 1, 4, synchronize=lambda: None, allocated_bytes=lambda: 0)

    def test_failure_is_recorded_without_altering_live_or_hanging_other_ranks(self):
        with TemporaryDirectory() as folder:
            engine = ShadowEngine(torch, folder, 0, 4, synchronize=lambda: None, allocated_bytes=lambda: 0)
            inputs = self.fixtures()
            def bad_stock(**kwargs):
                self.op(**kwargs)
                kwargs["device_buffer_k"][0, 0] = -3
                kwargs["top_k_blocks"][0, 0] = 99
            engine.check(0, inputs, self.op, bad_stock, final_layer=59)
            self.assertEqual(engine.report["mismatches"], 2)
            self.assertEqual(inputs["device_buffer_k"][0, 0].item(), 1)
            self.assertEqual(inputs["top_k_blocks"][0, 0].item(), 0)
            self.assertEqual(json.loads(engine.path.read_text())["mismatches"], 2)

    def test_actual_lifecycle_reset_and_reuse_accounting(self):
        with TemporaryDirectory() as folder:
            engine = ShadowEngine(torch, folder, 0, 4, synchronize=lambda: None, allocated_bytes=lambda: 0)
            coordinator = SimpleNamespace(req_device_buffer_tokens=torch.full((2, 1, 4), -1),
                req_device_buffer_token_locs=torch.full((2, 1, 4), -1),
                lru_slots=torch.arange(4).repeat(2, 1, 1), _lru_init=torch.arange(4))
            engine.allocated(0)
            engine.finished(coordinator, 0)
            engine.allocated(0)
            coordinator.req_device_buffer_tokens[0, 0, 0] = 0
            engine.finished(coordinator, 0)
            self.assertEqual(engine.report["lifecycle"], dict(allocations=2, finished=2,
                reused=1, reset_checks=2, reset_failures=1, unallocated_finish=0))
            self.assertEqual(engine.report["mismatches"], 1)
            engine.finished(coordinator, 0)
            self.assertEqual(engine.report["lifecycle"]["unallocated_finish"], 1)
            self.assertEqual(engine.report["lifecycle"]["reset_checks"], 2)
            engine.allocated(0)
            self.assertEqual(engine.report["lifecycle"]["reused"], 1)

    def test_memory_guard_precedes_scratch_allocation(self):
        with TemporaryDirectory() as folder:
            engine = ShadowEngine(torch, folder, 0, 4, synchronize=lambda: None,
                                  allocated_bytes=lambda: 0, free_bytes=lambda: 128)
            with self.assertRaisesRegex(RuntimeError, "1 GiB"):
                engine.copy("fixture", torch.zeros((1, 1)))
            self.assertEqual(engine.scratch, {})


if __name__ == "__main__":
    unittest.main()
