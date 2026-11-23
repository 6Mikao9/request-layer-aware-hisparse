"""CPU contract tests; no torch import, service, allocator or GPU required."""
import ast
import gc
import sys
import types
import weakref
from unittest.mock import patch
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from runtime.policy import StaticPolicy, validate_capacity, validate_metadata_shapes, validate_metadata_strides
from runtime.build_overlay import generate, digest
from runtime.hook import install, uninstall

ROOT = Path(__file__).parent

class Contracts(unittest.TestCase):
    def test_actual_topk_width_boundary(self):
        validate_capacity(32, 64, 4, 8)
        validate_capacity(64, 64, 4, 8)
        for args in [(31,64,4,8),(65,64,4,8),(0,64,4,8),(64,64,0,8),(64,64,4,0),(64,32768,4,8),(True,64,1,1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_capacity(*args)

    def test_frozen_absolute_layer_table(self):
        p = StaticPolicy(64, (32,)*60)
        self.assertEqual(p.digest, StaticPolicy(64,(32,)*60).digest)
        self.assertNotEqual(p.digest, StaticPolicy(64,(64,)*60).digest)
        with self.assertRaises(FrozenInstanceError):
            p.hmax=32
        for caps in [(32,)*59, [32]*60, (65,)*60]:
            with self.assertRaises(ValueError):
                StaticPolicy(64,caps)

    def test_tp_contract_is_hashed(self):
        self.assertNotEqual(StaticPolicy(64,(64,)*60,expected_tp_size=4).digest,
                            StaticPolicy(64,(64,)*60,expected_tp_size=8).digest)
        for tp in (1,2,3,16,True):
            with self.assertRaises(ValueError):
                StaticPolicy(64,(64,)*60,expected_tp_size=tp)

    def test_token_addressing_is_not_physical_page_padding(self):
        for width in (65,192):
            validate_metadata_shapes((3,64),(3,width),(3,width),64)
        for shapes in [((3,64),(3,64),(3,64)), ((3,64),(3,192),(3,65)), ((3,32),(3,192),(3,192))]:
            with self.assertRaises(ValueError):
                validate_metadata_shapes(*shapes,64)

    def test_shared_token_location_row_stride(self):
        validate_metadata_strides((69,1),(195,1),(195,1))
        validate_metadata_strides((64,1),(65,1),(65,1))
        for strides in [((69,1),(195,1),(196,1)), ((69,1),(195,2),(195,1)), ((69,2),(195,1),(195,1))]:
            with self.assertRaises(ValueError):
                validate_metadata_strides(*strides)

    def test_disabled_is_dependency_free_noop(self):
        self.assertFalse(install(StaticPolicy(64,(64,)*60)))
        self.assertFalse(uninstall())

    def test_hook_lifetime_requires_new_process_after_binding(self):
        # Fake modules exercise the actual hook lifecycle without importing torch
        # or pretending that coordinator collection proves CUDA graph teardown.
        from runtime import hook
        class TensorShape:
            def __init__(self, shape):
                self.shape = shape
            def size(self, axis):
                return self.shape[axis]
        class Coordinator:
            def __init__(self, mha=True):
                self.is_m3_hisparse = mha
                self.tp_world_size, self.tp_group = 4, object()
                self.device_buffer_size, self.padded_buffer_size = 64, 192
                self.mem_pool_device = types.SimpleNamespace(start_layer=0,end_layer=60,page_size=128)
                self.req_device_buffer_tokens = TensorShape((60,3,192))
                self.req_device_buffer_token_locs = TensorShape((60,3,192))
                self.lru_slots = TensorShape((60,3,64))
            def swap_in_selected_blocks(self):
                return 'stock'
        original_init, original_swap = Coordinator.__init__, Coordinator.swap_in_selected_blocks
        torch = types.ModuleType('torch')
        dist = types.ModuleType('torch.distributed')
        dist.is_initialized = lambda: True
        dist.get_world_size = lambda: 4
        def gather(output, value, group):
            output[:] = [value] * 4
        dist.all_gather_object = gather
        torch.distributed = dist
        torch.cuda = types.SimpleNamespace(get_device_name=lambda: 'NVIDIA A100')
        sglang = types.ModuleType('sglang')
        sglang.__file__ = str(ROOT/'fake_sglang.py')
        context = types.ModuleType('sglang.srt.runtime_context')
        context.get_parallel = lambda: types.SimpleNamespace(pp_size=1)
        coordinator = types.ModuleType('sglang.srt.managers.hisparse_coordinator')
        coordinator.HiSparseCoordinator = Coordinator
        generated = types.ModuleType('runtime.generated.coordinator_method')
        generated.swap_in_selected_blocks = lambda self: 'patched'
        fake_modules = {'torch':torch, 'torch.distributed':dist, 'sglang':sglang,
                        'sglang.srt.runtime_context':context,
                        'sglang.srt.managers.hisparse_coordinator':coordinator,
                        'runtime.generated.coordinator_method':generated}
        with patch.dict(sys.modules,fake_modules), patch.object(hook,'verify_sources'), \
             patch.object(hook,'_state',None), patch.object(hook,'_ever_bound',False), \
             patch.object(hook,'_bound',weakref.WeakSet()):
            policy=StaticPolicy(64,(32,)*60)
            self.assertTrue(hook.install(policy,enabled=True))
            self.assertTrue(hook.install(policy,enabled=True))
            non_mha=Coordinator(mha=False)
            self.assertEqual(non_mha.swap_in_selected_blocks(),'stock')
            self.assertFalse(hook._ever_bound)
            self.assertTrue(hook.uninstall())
            self.assertIs(Coordinator.__init__,original_init)
            self.assertIs(Coordinator.swap_in_selected_blocks,original_swap)
            self.assertTrue(hook.install(policy,enabled=True))
            mha=Coordinator()
            self.assertEqual(mha.swap_in_selected_blocks(),'patched')
            self.assertTrue(hook._ever_bound)
            del mha
            gc.collect()
            self.assertFalse(hook._bound)
            with self.assertRaisesRegex(RuntimeError,'start a new process'):
                hook.uninstall()
            self.assertIsNot(Coordinator.swap_in_selected_blocks,original_swap)

    def test_source_guard_rejects_before_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'source').write_text('changed')
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                generate(p,p/'out',{'source':'0'*64})
            self.assertFalse((p/'out').exists())

    def test_generated_python_syntax_and_integrity(self):
        manifest=json.loads((ROOT/'PROVENANCE.json').read_text())
        for name, expected in manifest['generated_sha256'].items():
            self.assertEqual(digest(ROOT/'generated'/name),expected)
            if name.endswith('.py'):
                ast.parse((ROOT/'generated'/name).read_text())

if __name__ == '__main__':
    unittest.main()
