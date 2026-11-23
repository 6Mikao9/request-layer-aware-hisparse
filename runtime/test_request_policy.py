"""Ownership and fixed-budget request routing tests; no CUDA dependency."""
import sys
import types
import unittest
from dataclasses import replace
from unittest.mock import patch

from runtime.request_policy import RequestTypePolicy, RequestPolicyRegistry
from runtime.request_dispatch import (
    load_grouped_blocks_to_device_buffer_mha, validate_groups, verify_grouped_sources,
)


def policy():
    return RequestTypePolicy(8192, (
        ('chatbot', (2048,) * 60),
        ('agent', (4096,) * 60),
        ('long_document', (6144,) * 30 + (2048,) * 30),
    ), default_type='agent', expected_tp_size=8)


class RequestPolicies(unittest.TestCase):
    def test_joint_tables_and_unknown_label_are_explicit(self):
        p = policy()
        self.assertEqual(p.groups(0), (2048, 4096, 6144))
        self.assertEqual(p.groups(59), (2048, 4096))
        self.assertEqual(p.resolve('unseen'), p.resolve('agent'))
        self.assertEqual(p.digest, replace(p, tables=tuple(reversed(p.tables))).digest)
        self.assertNotEqual(p.digest, replace(p, default_type='chatbot').digest)

    def test_reserved_requests_charge_same_budget_and_failure_is_atomic(self):
        r = RequestPolicyRegistry(policy(), 3, 60 * 4096)
        a = r.reserve(2, 'a', 10, 'chatbot')
        b = r.reserve(0, 'b', 11, 'chatbot')
        with self.assertRaises(MemoryError):
            r.reserve(1, 'c', 12, 'agent')
        self.assertEqual(r.charged_slots, 60 * 4096)
        self.assertTrue(all(v == 0 for row in r.capacity_rows() for v in row))
        r.cancel(b)
        r.commit(a)
        self.assertEqual(r.capacity_rows()[0], (0, 0, 2048))
        with self.assertRaises(RuntimeError):
            r.cancel(a)
        r.release(a)
        self.assertEqual(r.charged_slots, 0)

    def test_old_finish_cannot_clear_reused_slot_or_stale_batch(self):
        r = RequestPolicyRegistry(policy(), 1, 60 * 8192)
        old = r.reserve(0, 'r', 10, 'agent')
        r.commit(old)
        r.release(old)
        new = r.reserve(0, 'r', 10, 'chatbot')
        r.commit(new)
        self.assertGreater(new.generation, old.generation)
        for operation in (r.release, r.cancel):
            with self.assertRaises(RuntimeError):
                operation(old)
        with self.assertRaises(RuntimeError):
            r.validate_batch([old])
        self.assertEqual(r.validate_batch([new]), (0,))
        self.assertEqual(r.charged_slots, 60 * 2048)

    def test_no_decode_before_publication_or_duplicate_row(self):
        r = RequestPolicyRegistry(policy(), 3, 60 * 16384)
        a = r.reserve(2, 'a', 10, 'long_document')
        with self.assertRaises(RuntimeError):
            r.validate_batch([a])
        with self.assertRaises(RuntimeError):
            r.reserve(1, 'a', 11, 'agent')
        r.commit(a)
        b = r.reserve(0, 'b', 12, 'chatbot')
        r.commit(b)
        self.assertEqual(r.validate_batch([a, b]), (2, 0))
        with self.assertRaises(RuntimeError):
            r.validate_batch([a, a])
        self.assertEqual(r.capacity_rows()[0], (2048, 0, 6144))
        self.assertEqual(r.capacity_rows()[59], (2048, 0, 2048))

    def test_reject_illegal_table_and_topk_width(self):
        p = policy()
        for change in (dict(tables=p.tables + (p.tables[0],)),
                       dict(default_type='missing'), dict(hmax=4096),
                       dict(selected_tokens=8192), dict(expected_tp_size=16)):
            with self.assertRaises(ValueError):
                replace(p, **change)
        for groups in ((4096, 2048), (2048, 2048), (), (1024,), (True,)):
            with self.assertRaises(ValueError):
                validate_groups(groups, 8192, 16, 128)

    def test_generated_variant_matches_verified_inputs(self):
        verify_grouped_sources()


class Tensor:
    def __init__(self, values, shape=None, dtype='i32'):
        self.values = values
        self.shape = shape or (len(values),)
        self.dtype = dtype
        self.device = 'cuda'
        self.fills = 0

    def size(self, dim):
        return self.shape[dim]

    def is_contiguous(self):
        return True

    def fill_(self, value):
        self.fills += 1
        self.values[:] = [value] * len(self.values)


class GroupedDispatch(unittest.TestCase):
    def test_groups_preserve_other_rows_and_batch_order(self):
        # The fake op models only the capacity filter and output ownership.
        # Actual K/V/LRU correctness is independently checked by the CUDA fixture.
        torch = types.ModuleType('torch')
        torch.int32 = 'i32'
        ops = types.ModuleType('runtime.generated_grouped.ops')
        calls = []
        def kernel(assigned_capacities, active_hot_buffer_size, **kwargs):
            calls.append(active_hot_buffer_size)
            for row, slot in enumerate((2, 0, 1)):
                if assigned_capacities.values[slot] == active_hot_buffer_size:
                    kwargs['top_k_device_locs'].values[row] = slot * 100 + active_hot_buffer_size
        ops.load_blocks_to_device_buffer_mha = kernel
        output = Tensor([999] * 4)
        args = dict(assigned_capacities=Tensor([4096, 2048, 6144]),
                    capacities=(2048, 3072, 4096, 6144), hot_buffer_size=8192,
                    top_k_blocks=Tensor([], (4, 16)), sparse_block_size=128,
                    device_buffer_tokens=Tensor([], (3, 8320)), top_k_device_locs=output)
        with patch.dict(sys.modules, {'torch': torch, 'runtime.generated_grouped.ops': ops}):
            load_grouped_blocks_to_device_buffer_mha(**args)
            self.assertEqual(output.values, [6344, 4096, 2148, -1])
            self.assertEqual(output.fills, 1)
            self.assertEqual(calls, [2048, 3072, 4096, 6144])
            # Next layer changes capacities, so stale previous-layer rows must go.
            args['assigned_capacities'] = Tensor([2048, 2048, 4096])
            args['capacities'] = (2048, 4096)
            load_grouped_blocks_to_device_buffer_mha(**args)
            self.assertEqual(output.values, [4296, 2048, 2148, -1])

    def test_reject_split_or_duplicate_group_before_output_mutation(self):
        output = Tensor([999])
        common = dict(assigned_capacities=Tensor([4096]), hot_buffer_size=8192,
                      top_k_blocks=Tensor([], (1, 16)), sparse_block_size=128,
                      device_buffer_tokens=Tensor([], (1, 8320)), top_k_device_locs=output)
        for extra in (dict(capacities=(4096,), skip_io=True),
                      dict(capacities=(4096, 4096)),
                      dict(capacities=(4096,), active_hot_buffer_size=4096)):
            with self.assertRaises(ValueError):
                load_grouped_blocks_to_device_buffer_mha(**common, **extra)
        self.assertEqual(output.values, [999])


if __name__ == '__main__':
    unittest.main()
