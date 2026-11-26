"""CPU tests for the fixed-offload-set SGLang adapter: fake pinned modules plus CPU torch.

Fakes keep only the call contracts the hooks rely on. With ``SGLANG_SRC`` set, the
pinned ``MiniMaxSparseKVPool``, ``HiSparseMHAMainPool``, swap, backup and backend
sources are executed against the same hooks instead of the fakes.
"""
import __future__
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import types
import unittest
from unittest.mock import patch

from runtime import layer_placement as lp
from runtime import layer_placement_device as device
from runtime import shared_lru_device as arena
from runtime import study_config as sc

try:
    import torch
except ImportError:  # pure-Python environments still run the other suites
    torch = None

MEMORY = "sglang.srt.mem_cache.memory_pool"
MAIN = device.MAIN_POOL_MODULE
COORDINATOR = "sglang.srt.managers.hisparse_coordinator"
DENSE, SPARSE = (0,), (1, 2, 3, 4, 5)


def fixed(offload, direct=True, **pool):
    """Fixed offload set; resident layers read directly unless ``direct=False`` selects the resolver."""
    data = {"schema": sc.SCHEMA, "placement": {"mode": "fixed_layers", "offload_layers": list(offload)},
            "features": {"resident_direct": direct}}
    if pool:
        data["hot_pool"] = pool
    return sc.parse(data)


def owned_by(module, *objects):
    """Give fakes the defining module of the pinned symbols, as the stacked-hook check expects."""
    for obj in objects:
        obj.__module__ = module
        for value in vars(obj).values() if isinstance(obj, type) else ():
            if callable(value):
                value.__module__ = module
    return objects


def fake_pool_classes():
    """Stock pool contracts: planes indexed by ``layer_id - start_layer``, writes honour ``layer_id_override``."""

    class MHATokenToKVPool:
        def __init__(self, size, page_size, dtype, head_num, head_dim, layer_num, device, enable_memory_saver,
                     start_layer=None, end_layer=None):
            self.size, self.page_size, self.dtype, self.store_dtype = size, page_size, dtype, dtype
            self.head_num, self.head_dim, self.layer_num, self.device = head_num, head_dim, layer_num, device
            self.v_head_dim = head_dim
            self.start_layer, self.end_layer = start_layer or 0, end_layer or layer_num - 1
            self.kv_cache_layout, self.use_hnd, self.post_capture_active = "nhd", False, False
            self.is_quantized_kv_cache, self.layer_transfer_counter = False, None
            self.memory_saver_adapter = types.SimpleNamespace(region=lambda _: contextlib.nullcontext())
            self.enable_custom_mem_pool, self.custom_mem_pool = False, None
            self._create_buffers_normal()
            self.mem_usage = sum(t.numel() * t.element_size() for t in self.k_buffer + self.v_buffer) / 2**30
            self.model_layer_ids = tuple(range(self.start_layer, self.start_layer + layer_num))

        def _kv_buffer_shapes(self):
            shape = (self.size + self.page_size, self.head_num, self.head_dim)
            return shape, shape

        def _create_buffers_normal(self):
            k, v = self._kv_buffer_shapes()
            self.k_buffer = [torch.zeros(k, dtype=self.dtype) for _ in range(self.layer_num)]
            self.v_buffer = [torch.zeros(v, dtype=self.dtype) for _ in range(self.layer_num)]

        def get_key_buffer(self, layer_id):
            return self.k_buffer[layer_id - self.start_layer]

        def get_value_buffer(self, layer_id):
            return self.v_buffer[layer_id - self.start_layer]

        def get_kv_buffer(self, layer_id):
            return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

        def set_kv_buffer(self, layer, loc_info, cache_k, cache_v, k_scale=None, v_scale=None,
                          layer_id_override=None, dcp_kv_mask=None):
            plane = (layer_id_override if layer_id_override is not None else layer.layer_id) - self.start_layer
            self.k_buffer[plane][loc_info] = cache_k
            self.v_buffer[plane][loc_info] = cache_v

        def get_kv_size_bytes(self):
            return (sum(t.numel() * t.element_size() for t in self.k_buffer),
                    sum(t.numel() * t.element_size() for t in self.v_buffer))

    class HiSparseMHAMainPool(MHATokenToKVPool):
        def register_mapping(self, mapping):
            self.full_to_hisparse_device_index_mapping = mapping

        def set_kv_buffer(self, layer, loc, cache_k, cache_v, *args, **kwargs):
            super().set_kv_buffer(layer, self.full_to_hisparse_device_index_mapping[loc], cache_k, cache_v,
                                  *args, **kwargs)

    owned_by(MEMORY, MHATokenToKVPool)
    owned_by(MAIN, HiSparseMHAMainPool)
    return MHATokenToKVPool, HiSparseMHAMainPool
FAKE_SPARSE_POOL = '''
class MiniMaxSparseKVPool:
    """The pinned HiSparse branch only: dense pool, call-time HiSparse main pool import, stock routing."""

    def __init__(self, size, page_size, dtype, head_num, head_dim, idx_head_dim, dense_layer_ids, sparse_layer_ids,
                 device, disable_value_sparse_layer_ids=None, enable_memory_saver=False, index_dtype=None,
                 start_layer=None, end_layer=None, main_pool_cls=MHATokenToKVPool, index_kv_pool_cls=MHATokenToKVPool,
                 index_k_pool_cls=None, enable_hisparse=False, host_to_device_ratio=2):
        local_dense = [lid for lid in dense_layer_ids if start_layer <= lid < end_layer]
        local_sparse = [lid for lid in sparse_layer_ids if start_layer <= lid < end_layer]
        self._dense_layer_ids, self.page_size, self.start_layer = set(local_dense), page_size, start_layer
        self.layer_transfer_counter, self._enable_hisparse = None, enable_hisparse
        from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseMHAMainPool
        self.dense_pool = main_pool_cls(size=size * host_to_device_ratio, page_size=page_size, dtype=dtype,
                                        head_num=head_num, head_dim=head_dim, layer_num=len(local_dense),
                                        device=device, enable_memory_saver=enable_memory_saver,
                                        start_layer=start_layer, end_layer=start_layer + len(local_dense))
        self.main_pool = HiSparseMHAMainPool(size=size, page_size=page_size, dtype=dtype, head_num=head_num,
                                             head_dim=head_dim, layer_num=len(local_sparse), device=device,
                                             enable_memory_saver=enable_memory_saver,
                                             start_layer=local_sparse[0], end_layer=end_layer)
        self.main_pool.model_layer_ids = tuple(local_sparse)
        self.mem_usage = self.main_pool.mem_usage + self.dense_pool.mem_usage

    def register_mapping(self, mapping):
        self.main_pool.register_mapping(mapping)

    def _wait_for_layer(self, layer_id):
        self.waited = layer_id

    def _pool_for(self, layer_id):
        return self.dense_pool if layer_id in self._dense_layer_ids else self.main_pool

    def get_key_buffer(self, layer_id):
        self._wait_for_layer(layer_id)
        return self._pool_for(layer_id).get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id):
        self._wait_for_layer(layer_id)
        return self._pool_for(layer_id).get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id):
        self._wait_for_layer(layer_id)
        return self._pool_for(layer_id).get_kv_buffer(layer_id)

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None):
        self._pool_for(layer.layer_id).set_kv_buffer(layer, loc, cache_k, cache_v, k_scale, v_scale)

    def set_fused_kv_index_buffer(self, layer, loc, cache_k, cache_v, cache_idx_k, cache_idx_v, k_scale=None,
                                  v_scale=None, idx_k_scale=None, idx_v_scale=None):
        self.set_kv_buffer(layer, loc, cache_k, cache_v, k_scale, v_scale)

    def get_kv_size_bytes(self):
        sizes = [p.get_kv_size_bytes() for p in (self.main_pool, self.dense_pool)]
        return sum(k for k, _ in sizes), sum(v for _, v in sizes)
'''


def context(tp=8):
    from runtime.test_shared_lru_device import fake_context
    ctx = fake_context()
    ctx.get_parallel = lambda: types.SimpleNamespace(pp_size=1, attn_tp_size=tp)
    return ctx


def fake_sglang(tmp, sparse_pool_cls=None):
    """Every pinned module the two installers import, mapped to small fakes; returns (modules, handles)."""
    MHA, Main = fake_pool_classes()
    names = ("sglang", "sglang.srt", "sglang.srt.mem_cache", MEMORY, MAIN, "sglang.srt.constants",
             "sglang.srt.model_executor", "sglang.srt.model_executor.pool_configurator", "sglang.srt.layers",
             "sglang.srt.layers.attention", "sglang.srt.layers.attention.minimax_sparse_backend",
             "sglang.srt.layers.attention.minimax_sparse_ops", device.OPS_MODULE, "sglang.srt.managers", COORDINATOR,
             "sglang.srt.runtime_context", "sglang.srt.configs", "sglang.srt.configs.model_config",
             "sglang.srt.mem_cache.sparsity")
    modules = {name: types.ModuleType(name) for name in names}
    modules["sglang"].__file__ = str(Path(tmp) / "__init__.py")
    modules["sglang.srt.constants"].GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
    namespace = {"__name__": MEMORY, "MHATokenToKVPool": MHA}
    exec(FAKE_SPARSE_POOL, namespace)
    Sparse = sparse_pool_cls or namespace["MiniMaxSparseKVPool"]
    vars(modules[MEMORY]).update(MiniMaxSparseKVPool=Sparse, MHATokenToKVPool=MHA)
    modules[MAIN].HiSparseMHAMainPool = Main
    log = dict(prefill=[], swap=[], consumer=[], cell=[])

    class DefaultPoolConfigurator:
        def _compute_cell_size(self, kvc, num_layers):
            log["cell"].append(num_layers)
            return 1000

    class MiniMaxSparseAttnBackend:
        def __init__(self, kv_pool):
            self.kv_pool = kv_pool

        def _hisparse_swap_in_blocks(self, forward_batch, topk_idx, layer_id):
            return ("stock swap-in", layer_id)

    def minimax_sparse_prefill(q, k_cache, v_cache, *rest, loc_mapping=None, **kwargs):
        log["prefill"].append(loc_mapping)
        return "prefill"

    def flash_decode_with_gqa_share_sparse(**kwargs):
        log["consumer"].append((kwargs["k_cache"], kwargs["v_cache"]))
        return "attention"

    owned_by("sglang.srt.model_executor.pool_configurator", DefaultPoolConfigurator)
    owned_by("sglang.srt.layers.attention.minimax_sparse_backend", MiniMaxSparseAttnBackend)
    owned_by(device.OPS_MODULE, minimax_sparse_prefill)
    owned_by("sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse", flash_decode_with_gqa_share_sparse)
    modules["sglang.srt.model_executor.pool_configurator"].DefaultPoolConfigurator = DefaultPoolConfigurator
    modules["sglang.srt.layers.attention.minimax_sparse_backend"].MiniMaxSparseAttnBackend = MiniMaxSparseAttnBackend
    vars(modules[device.OPS_MODULE]).update(minimax_sparse_prefill=minimax_sparse_prefill,
                                            flash_decode_with_gqa_share_sparse=flash_decode_with_gqa_share_sparse)
    modules[COORDINATOR].HiSparseCoordinator = fake_coordinator_class(log)
    vars(modules["sglang.srt.runtime_context"]).update(vars(context()))
    vars(modules["sglang.srt.configs.model_config"]).update(
        is_minimax_sparse=lambda hf: True, uses_minimax_m3_nvfp4_native_stack=lambda q: q == "nvfp4",
        get_minimax_sparse_attention_config=lambda hf: hf,
        get_minimax_sparse_layer_ids=lambda cfg: (list(DENSE), list(SPARSE)))
    modules["sglang.srt.mem_cache.sparsity"].parse_hisparse_config = lambda: types.SimpleNamespace(
        host_to_device_ratio=2)
    handles = types.SimpleNamespace(MHA=MHA, Main=Main, Sparse=Sparse, Configurator=DefaultPoolConfigurator,
                                    Backend=MiniMaxSparseAttnBackend, ops=modules[device.OPS_MODULE],
                                    Coordinator=modules[COORDINATOR].HiSparseCoordinator, log=log)
    return modules, handles


def fake_coordinator_class(log):
    class HiSparseCoordinator:
        def __init__(self, req_to_token_pool, token_to_kv_pool_allocator, top_k, device_buffer_size, *a, **kw):
            pool = token_to_kv_pool_allocator.get_kvcache().main_pool
            H, page, L, slots, i32 = device_buffer_size, pool.page_size, pool.layer_num, 4, torch.int32
            self.token_to_kv_pool_allocator, self.mem_pool_device = token_to_kv_pool_allocator, pool
            self.is_m3_hisparse, self.is_dsv4_hisparse, self.skip_io, self.enable_prefetch = True, False, False, False
            self.spec_swap = types.SimpleNamespace(enabled=False)
            self.top_k, self.device_buffer_size, self.padded_buffer_size = top_k, H, H + page
            self.item_size_bytes, self.swap_in_block_size = pool.head_num * pool.head_dim * 2, 960
            self.req_device_buffer_tokens = torch.full((L, slots, H + page), -1, dtype=i32)
            self.req_device_buffer_token_locs = torch.full((L, slots, H + page), -1, dtype=i32)
            self.lru_slots = torch.zeros((L, slots, H), dtype=torch.int16)
            self.top_k_device_locs_buffer = torch.full((slots, H), -1, dtype=i32)
            self.num_real_reqs = torch.zeros(1, dtype=i32)
            self.req_to_host_pool = torch.full((slots, 64), -1, dtype=torch.int64)
            self.mem_pool_host = types.SimpleNamespace(
                layer_num=L, k_buffer=[torch.zeros((64, 1, pool.head_dim)) for _ in range(L)],
                v_buffer=[torch.zeros((64, 1, pool.head_dim)) for _ in range(L)])
            self.tp_world_size, self.tp_group = 1, None

        def alloc_device_buffer(self, req):
            self.req_device_buffer_tokens[:, req.kv.req_pool_idx, :self.device_buffer_size] = torch.arange(
                self.device_buffer_size)

        def request_finished(self, req):
            log["swap"].append(("finished", req.kv.req_pool_idx))

        def admit_request_direct(self, req):
            return "stock direct"

        def swap_in_selected_blocks(self, req_pool_indices, seq_lens, top_k_blocks, layer_id, sparse_block_size):
            log["swap"].append(layer_id)
            return "stock swap", layer_id

    owned_by(COORDINATOR, HiSparseCoordinator)
    return HiSparseCoordinator


def build_pool(handles, *, size=16, page=2, ratio=2, **overrides):
    kwargs = dict(size=size, page_size=page, dtype=torch.bfloat16, head_num=1, head_dim=4, idx_head_dim=4,
                  dense_layer_ids=list(DENSE), sparse_layer_ids=list(SPARSE), device="cpu",
                  disable_value_sparse_layer_ids=list(SPARSE), start_layer=0, end_layer=6, enable_hisparse=True,
                  host_to_device_ratio=ratio)
    kwargs.update(overrides)
    pool = handles.Sparse(**kwargs)
    mapping = torch.zeros(size * ratio + page + 1, dtype=torch.int64)
    mapping[page:page + 8] = torch.arange(size + page - 8, size + page)    # logical 2..9 -> device rows 10..17
    pool.register_mapping(mapping)
    allocator = types.SimpleNamespace(get_kvcache=lambda: pool, size_full=size * ratio)
    return pool, allocator


@unittest.skipUnless(torch is not None, "CPU torch required")
class Harness(unittest.TestCase):
    """Fake pinned modules, clean installer state and single-rank collectives."""
    CONFIG = fixed((2, 4))

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.modules, self.h = fake_sglang(tmp.name, getattr(self, "SPARSE_POOL", None))
        for patcher in (patch.dict(sys.modules, self.modules), patch.object(device, "_STATE", None),
                        patch.object(arena, "_STATE", None), patch.object(arena, "_CREATED_POOLS", 0),
                        patch.object(device, "verify_digests", lambda root: {}),
                        patch.object(arena, "verify_digests", lambda root, digests=None: {}),
                        patch.object(torch.cuda, "get_device_capability", lambda *a: (8, 0)),
                        patch.object(torch.distributed, "get_rank", lambda *a: 0, create=True),
                        patch.object(torch.distributed, "all_gather_object",
                                     lambda out, value, group=None: out.__setitem__(slice(None), [value] * len(out)),
                                     create=True)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def coordinator(self, allocator, *, top_k=4, H=8):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            coordinator = self.h.Coordinator(None, allocator, top_k, H)
        return coordinator, [json.loads(line) for line in out.getvalue().splitlines()]


class Installer(Harness):
    def test_native_placements_install_nothing_and_unsupported_fail_closed(self):
        pool_methods = dict(vars(self.h.Sparse))
        for config in (sc.StudyConfig(), sc.parse({"schema": sc.SCHEMA, "hot_pool": {"mode": "shared",
                                                                                    "layout": "arena"}})):
            self.assertFalse(device.install(config, "0" * 64))
        self.assertEqual(dict(vars(self.h.Sparse)), pool_methods)
        for config in (fixed(()), sc.parse({"schema": sc.SCHEMA, "placement": {"mode": "full_resident"}})):
            with self.assertRaises(RuntimeError):
                device.install(config, "0" * 64)
        self.assertFalse(device.uninstall())

    def test_frozen_config_stacking_and_uninstall(self):
        self.assertTrue(device.install(self.CONFIG, "a" * 64))
        self.assertTrue(device.install(self.CONFIG, "a" * 64))
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            device.install(fixed((2,)), "a" * 64)
        self.assertTrue(device.uninstall())
        self.assertIs(vars(self.h.Sparse)["__init__"].__module__, MEMORY)
        self.h.Backend._hisparse_swap_in_blocks = lambda *a: "other hook"
        with self.assertRaisesRegex(RuntimeError, "refusing"):
            device.install(self.CONFIG, "a" * 64)

    def test_uninstall_after_a_split_pool_needs_a_new_process(self):
        device.install(self.CONFIG, "a" * 64)
        build_pool(self.h)
        with self.assertRaisesRegex(RuntimeError, "new process"):
            device.uninstall()

    def test_drifted_constructor_is_refused(self):
        again = ("HiSparseMHAMainPool(size=size, page_size=page_size, dtype=dtype, head_num=head_num, "
                 "head_dim=head_dim, layer_num=len(local_sparse), device=device, "
                 "enable_memory_saver=enable_memory_saver, start_layer=local_sparse[0], end_layer=end_layer)\n"
                 "        self.main_pool.model_layer_ids = tuple(local_sparse)")
        namespace = {"__name__": MEMORY, "MHATokenToKVPool": self.h.MHA}
        exec(FAKE_SPARSE_POOL.replace("self.main_pool.model_layer_ids = tuple(local_sparse)", again), namespace)
        self.modules[MEMORY].MiniMaxSparseKVPool = namespace["MiniMaxSparseKVPool"]
        device.install(self.CONFIG, "a" * 64)
        with self.assertRaisesRegex(RuntimeError, "differently"):
            build_pool(types.SimpleNamespace(Sparse=namespace["MiniMaxSparseKVPool"]))
        self.assertIs(self.modules[MAIN].HiSparseMHAMainPool, self.h.Main, "the scoped constructor swap is undone")


class SplitPools(Harness):
    def setUp(self):
        super().setUp()
        self.assertTrue(device.install(self.CONFIG, "a" * 64))
        self.pool, self.allocator = build_pool(self.h)

    def test_offloaded_planes_resident_planes_and_routes(self):
        main, resident = self.pool.main_pool, self.pool._rlh_resident_pool
        self.assertIsInstance(main, self.h.Main)
        self.assertEqual((main.layer_num, main.start_layer, main.end_layer, main.model_layer_ids), (2, 0, 2, (2, 4)))
        self.assertEqual(tuple(main.k_buffer[0].shape), (18, 1, 4))           # size + page device rows
        self.assertEqual((resident.layer_num, resident.model_layer_ids), (3, (1, 3, 5)))
        self.assertEqual(tuple(resident.k_buffer[0].shape), (34, 1, 4))       # size * ratio + page logical rows
        self.assertEqual(self.pool.dense_pool.model_layer_ids, (0,))
        self.assertIs(self.pool.get_key_buffer(4), main.k_buffer[1])
        self.assertIs(self.pool.get_value_buffer(3), resident.v_buffer[1])
        self.assertEqual(self.pool.get_kv_buffer(5), (resident.k_buffer[2], resident.v_buffer[2]))
        self.assertIs(self.pool.get_key_buffer(0), self.pool.dense_pool.k_buffer[0])
        self.assertEqual(self.pool.waited, 0)
        k, v = self.pool.get_kv_size_bytes()
        planes = main.k_buffer + resident.k_buffer + self.pool.dense_pool.k_buffer
        self.assertEqual(k, sum(t.numel() * 2 for t in planes))
        self.assertAlmostEqual(self.pool.mem_usage, main.mem_usage + resident.mem_usage
                               + self.pool.dense_pool.mem_usage)

    def test_writes_land_in_the_routed_plane_and_row(self):
        loc = torch.tensor([3, 7])
        for layer_id in (0, 1, 2, 3, 4, 5):
            k = torch.full((2, 1, 4), float(layer_id + 1), dtype=torch.bfloat16)
            self.pool.set_fused_kv_index_buffer(types.SimpleNamespace(layer_id=layer_id), loc, k, -k, None, None)
        main, resident, dense = self.pool.main_pool, self.pool._rlh_resident_pool, self.pool.dense_pool
        rows = self.pool.main_pool.full_to_hisparse_device_index_mapping[loc]
        self.assertEqual(rows.tolist(), [11, 15])
        for plane, layer_id in enumerate((2, 4)):
            self.assertTrue(torch.all(main.k_buffer[plane][rows] == layer_id + 1))     # translated device rows
            self.assertTrue(torch.all(main.k_buffer[plane][loc] == 0), "offloaded planes never take logical rows")
        for plane, layer_id in enumerate((1, 3, 5)):
            self.assertTrue(torch.all(resident.k_buffer[plane][loc] == layer_id + 1))  # logical rows
            self.assertTrue(torch.all(resident.v_buffer[plane][loc] == -(layer_id + 1)))
        self.assertTrue(torch.all(dense.k_buffer[0][loc] == 1))

    def test_constructor_rejects_layouts_it_cannot_split(self):
        for overrides, error in ((dict(enable_hisparse=False), RuntimeError),
                                 (dict(main_pool_cls=type("Other", (self.h.MHA,), {})), RuntimeError),
                                 (dict(sparse_layer_ids=[1, 3, 5]), ValueError)):     # S names layers 2 and 4
            with self.subTest(overrides=overrides), self.assertRaises(error):
                build_pool(self.h, **overrides)
        subclass = type("Subclass", (self.h.Sparse,), {})
        with self.assertRaisesRegex(RuntimeError, "stock MiniMaxSparseKVPool"):
            build_pool(types.SimpleNamespace(Sparse=subclass))

    def test_cell_size_charges_resident_layers_at_the_logical_ratio(self):
        kvc = types.SimpleNamespace(
            model_config=types.SimpleNamespace(hf_config={}, quantization=None, head_dim=4,
                                               get_num_kv_heads=lambda tp: 1),
            server_args=types.SimpleNamespace(enable_hisparse=True), kv_cache_dtype=torch.bfloat16,
            layer_info=types.SimpleNamespace(start_layer=0, end_layer=6))
        configurator = self.h.Configurator()
        self.assertEqual(configurator._compute_cell_size(kvc, 6), 1000 + 3 * (2 - 1) * 2 * 1 * 4 * 2)
        charge = device._STATE["charge"]
        self.assertEqual((charge["stock_cell_bytes"], charge["resident_layers"]), (1000, [1, 3, 5]))
        kvc.model_config.quantization = "nvfp4"
        with self.assertRaises(RuntimeError):
            configurator._compute_cell_size(kvc, 6)

    def test_resident_layers_skip_swap_in_and_location_mapping(self):
        backend = self.h.Backend(self.pool)
        self.assertIsNone(backend._hisparse_swap_in_blocks(None, None, 3))
        self.assertEqual(backend._hisparse_swap_in_blocks(None, None, 4), ("stock swap-in", 4))
        with self.assertRaises(RuntimeError):
            backend._hisparse_swap_in_blocks(None, None, 0)            # dense layers never reach sparse decode
        mapping = self.pool.main_pool.full_to_hisparse_device_index_mapping
        prefill = getattr(self.h.ops, device.PREFILL)
        for layer_id, expected in ((3, None), (4, mapping)):
            k, v = self.pool.get_kv_buffer(layer_id)
            self.assertEqual(prefill("q", k, v, None, loc_mapping=mapping), "prefill")
            self.assertIs(self.h.log["prefill"][-1], expected)
        main_k, main_v = self.pool.get_kv_buffer(2)
        for args, kwargs in (((torch.zeros(2), main_k, main_v), {}),
                             (("q", main_k, main_v), dict(loc_mapping=None)),
                             (("q", main_k.clone(), main_v), dict(loc_mapping=mapping))):
            with self.assertRaises(RuntimeError):
                prefill(*args, **kwargs)

    def test_stock_layout_coordinator_gates_maps_planes_and_prints_receipt(self):
        coordinator, receipts = self.coordinator(self.allocator)
        receipt = receipts[0]
        self.assertEqual((receipt["event"], receipt["rank"], receipt["config_file_sha256"]),
                         ("hisparse_layer_placement_bound", 0, "a" * 64))
        self.assertEqual(receipt["signature"]["offload_layers"], [2, 4])
        self.assertEqual(receipt["signature"]["resident_layers"], [1, 3, 5])
        self.assertEqual(receipt["bytes"]["resident_main_kv"], 2 * 3 * 34 * 4 * 2)
        self.assertEqual(receipt["bytes"]["hot_main_kv"], 2 * 2 * 18 * 4 * 2)
        self.assertEqual(coordinator.swap_in_selected_blocks("rpi", "seq", "blocks", 4, 128), ("stock swap", 1))
        self.assertEqual(coordinator.swap_in_selected_blocks("rpi", "seq", "blocks", 2, 128), ("stock swap", 0))
        with self.assertRaisesRegex(RuntimeError, "complete GPU main KV"):
            coordinator.swap_in_selected_blocks("rpi", "seq", "blocks", 3, 128)
        with self.assertRaises(RuntimeError):
            coordinator.admit_request_direct(types.SimpleNamespace())
        with self.assertRaisesRegex(RuntimeError, "One HiSparse coordinator"):
            self.coordinator(self.allocator)

    def test_runner_receipt_directory_keeps_one_file_per_rank(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {arena.RECEIPT_DIR_ENV: tmp}):
            _, printed = self.coordinator(self.allocator)
            files = {path.name: json.loads(path.read_text()) for path in Path(tmp).iterdir()}
        self.assertEqual(files, {"hisparse_layer_placement_bound.rank0.json": printed[0]})

    def test_gate_rejects_mismatched_tables_hosts_and_modes(self):
        mutations = [lambda c: setattr(c.mem_pool_host, "layer_num", 5),
                     lambda c: setattr(c, "lru_slots", torch.zeros((5, 4, 8), dtype=torch.int16)),
                     lambda c: setattr(c, "enable_prefetch", True),
                     lambda c: setattr(self.pool._rlh_resident_pool, "size", 16),
                     lambda c: self.pool._rlh_routes.pop(5)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                device._STATE["binding"] = None
                coordinator = self.h.Coordinator.__new__(self.h.Coordinator)
                device._STATE["originals"]["coordinator"]["__init__"](coordinator, None, self.allocator, 4, 8)
                mutate(coordinator)
                with self.assertRaises(RuntimeError):
                    device.bind(coordinator, torch=torch, context=context())
                self.pool, self.allocator = build_pool(self.h)
        with patch.object(torch.cuda, "get_device_capability", lambda *a: (9, 0)), self.assertRaises(RuntimeError):
            self.coordinator(self.allocator)

    def test_tp_ranks_must_agree_on_the_offload_set(self):
        stock_init = device._STATE["originals"]["coordinator"]["__init__"]

        def two_ranks(coordinator, *args, **kwargs):
            stock_init(coordinator, *args, **kwargs)
            coordinator.tp_world_size = 2

        with patch.dict(device._STATE["originals"]["coordinator"], {"__init__": two_ranks}), \
                patch.object(torch.distributed, "all_gather_object",
                             lambda out, value, group=None: out.__setitem__(slice(None), [value, "other"])), \
                self.assertRaisesRegex(RuntimeError, "disagree"):
            self.coordinator(self.allocator)


class ResidentResolver(Harness):
    """features.resident_direct=false: resident layers resolve their req_to_token rows into the native buffer."""
    CONFIG = fixed((2, 4), direct=False)

    def setUp(self):
        super().setUp()
        self.assertTrue(device.install(self.CONFIG, "c" * 64))
        self.pool, self.allocator = build_pool(self.h)
        self.coordinator_, self.receipts = self.coordinator(self.allocator)
        self.calls = []

        def resolve_resident(top_k, rpi, seq, num_real, req_to_token, out):       # the CPU mirror stands in
            self.calls.append((top_k, rpi, seq, num_real, req_to_token, out))
            table = lp.resident_slot_table(top_k.tolist(), rpi.tolist(), seq.tolist(), req_to_token.tolist(),
                                           num_real=int(num_real[0]), block_size=2)
            out.copy_(torch.tensor(table, dtype=torch.int32))

        module = types.SimpleNamespace(resolve_resident=resolve_resident)
        patcher = patch.object(device, "_resident_module", lambda blocks, block: module)
        patcher.start()
        self.addCleanup(patcher.stop)

    def backend(self):
        backend = self.h.Backend(self.pool)
        backend.hisparse_coordinator, backend.block_size_k = self.coordinator_, 2
        backend.req_to_token = (torch.arange(4 * 16, dtype=torch.int32).reshape(4, 16) * 7) % 31 + 2
        return backend

    def test_resident_rows_become_slots_in_the_native_output_buffer(self):
        backend, coordinator = self.backend(), self.coordinator_
        batch = types.SimpleNamespace(req_pool_indices=torch.tensor([2, 0, 1], dtype=torch.int64),
                                      seq_lens=torch.tensor([9, 5, 7], dtype=torch.int32))
        topk = torch.tensor([[[4, -1], [2, 2], [3, 1]]], dtype=torch.int32)   # invalid/out-of-range, duplicate
        coordinator.num_real_reqs.fill_(2)                                     # the third row is graph padding
        slots = backend._hisparse_swap_in_blocks(batch, topk, 3)
        self.assertEqual(tuple(slots.shape), (1, 3, 4))
        self.assertEqual(slots.data_ptr(), coordinator.top_k_device_locs_buffer.data_ptr(), "native output buffer")
        r2t = backend.req_to_token
        self.assertEqual(slots[0].tolist(), [[r2t[2, 8].item(), -1, -1, -1],
                                             [r2t[0, 4].item(), -1, r2t[0, 4].item(), -1],
                                             [-1] * 4])
        top_k, rpi, seq, num_real, req_to_token, _ = self.calls[-1]
        self.assertIs(num_real, coordinator.num_real_reqs)
        self.assertIs(req_to_token, r2t)
        self.assertEqual((top_k.tolist(), rpi.tolist(), seq.tolist()), (topk[0].tolist(), [2, 0, 1], [9, 5, 7]))
        self.assertEqual(backend._hisparse_swap_in_blocks(batch, topk, 4), ("stock swap-in", 4))
        self.assertEqual(len(self.calls), 1, "offloaded layers never launch the resident resolver")
        with self.assertRaisesRegex(ValueError, "resident resolver contract"):
            backend._hisparse_swap_in_blocks(batch, topk.long(), 5)

    def test_receipt_and_tp_signature_name_the_resolver_and_its_kernel(self):
        receipt = self.receipts[0]
        self.assertEqual(receipt["signature"]["resident_path"], "resolver")
        self.assertEqual(receipt["signature"]["resident_kernel_sha256"], device.kernel_sha256())
        self.assertEqual(receipt["resident_resolver"]["output"], "coordinator.top_k_device_locs_buffer")
        self.assertTrue(receipt["capabilities"]["resident_path"].startswith("resolver"))
        self.assertEqual(receipt["capabilities"]["features"], {"resident_direct": False})


class ArenaOverTheOffloadSet(Harness):
    CONFIG = fixed((2, 4), mode="shared", layout="arena")

    def setUp(self):
        super().setUp()
        self.assertTrue(device.install(self.CONFIG, "b" * 64))
        self.assertTrue(arena.install(self.CONFIG, "b" * 64))
        self.assertNotIn("__init__", device._STATE["hooks"]["coordinator"], "the arena owns the coordinator")
        self.pool, self.allocator = build_pool(self.h)
        self.coordinator_, self.receipts = self.coordinator(self.allocator)

    def test_arena_spans_only_the_offloaded_planes(self):
        runtime = self.coordinator_._request_layer_hisparse_arena
        k_all, _ = arena.verify_planes(self.pool.main_pool)
        self.assertEqual(tuple(k_all.shape), (2, 18, 1, 4))
        self.assertEqual((runtime.layers, runtime.plane_of, runtime.plan.rows), ((2, 4), {2: 0, 4: 1}, 16))
        receipt = self.receipts[0]
        self.assertEqual(receipt["event"], "hisparse_shared_lru_bound")
        self.assertEqual(receipt["placement"]["signature"]["resident_layers"], [1, 3, 5])
        self.assertEqual(receipt["signature"]["layers"], [2, 4])
        self.assertIs(self.coordinator_._request_layer_hisparse_placement, device._STATE["binding"])

    def test_resolver_addresses_s_planes_and_resident_layers_never_reach_it(self):
        calls = []
        module = types.SimpleNamespace(resolve=lambda *args: calls.append(args))
        rpi, seq = torch.tensor([1], dtype=torch.int64), torch.tensor([30], dtype=torch.int32)
        blocks = torch.tensor([[5, 1]], dtype=torch.int32)
        with patch.object(arena, "_jit_module", lambda *variant: module):
            self.coordinator_.swap_in_selected_blocks(rpi, seq, blocks, 4, 2)
            self.assertEqual(calls[-1][19:21], (1, 2))                           # plane 1 of 2 planes
            self.assertIs(calls[-1][12], self.coordinator_.mem_pool_host.k_buffer[1])
            with self.assertRaisesRegex(ValueError, "not a host-backed plane"):
                self.coordinator_.swap_in_selected_blocks(rpi, seq, blocks, 3, 2)
        consumer = getattr(self.h.ops, arena.CONSUMER)
        runtime = self.coordinator_._request_layer_hisparse_arena
        consumer(k_cache=self.pool.get_key_buffer(4), v_cache=self.pool.get_value_buffer(4), hisparse_slots="s")
        self.assertIs(self.h.log["consumer"][-1][0], runtime.flat_k)
        resident = self.pool.get_kv_buffer(3)
        consumer(k_cache=resident[0], v_cache=resident[1], hisparse_slots=None)
        self.assertIs(self.h.log["consumer"][-1][0], resident[0], "resident planes reach the stock consumer")
        self.assertIsNone(self.h.Backend(self.pool)._hisparse_swap_in_blocks(None, None, 3))


def pinned_source(root, rel, qualified):
    path = Path(root) / rel
    text = path.read_text()
    node = arena._find(ast.parse(text), qualified)
    source = textwrap.dedent(" " * node.col_offset + ast.get_source_segment(text, node))
    return node.name, compile(source, str(path), "exec", flags=__future__.annotations.compiler_flag,
                              dont_inherit=True)


def load_pinned(root, rel, qualified, namespace):
    """Execute one pinned class or function with stand-in module globals; nothing from SGLang is imported."""
    name, code = pinned_source(root, rel, qualified)
    exec(code, namespace)
    return namespace[name]


class KOnlyPool:
    def __init__(self, size, page_size, dtype, head_num, head_dim, layer_num, device, enable_memory_saver):
        self.dtype = self.store_dtype = dtype
        self.head_dim, self.mem_usage = head_dim, 0.0
        self.k_buffer = [torch.zeros((size + page_size, head_num, head_dim), dtype=dtype) for _ in range(layer_num)]

    def set_k_buffer(self, layer_id, loc, cache_k):
        self.k_buffer[layer_id][loc] = cache_k

    def get_kv_size_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.k_buffer), 0


@unittest.skipUnless(torch is not None and "SGLANG_SRC" in os.environ, "set SGLANG_SRC to a pinned checkout")
class PinnedSources(Harness):
    """The pinned constructor, routing, fused store, swap, backup and backend sources under the hooks."""

    def setUp(self):
        super().setUp()
        root = Path(os.environ["SGLANG_SRC"])
        self.root = root / "python" / "sglang" if (root / "python" / "sglang").is_dir() else root
        self.calls = dict(swap=[], transfer=[])
        envs = types.SimpleNamespace(SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE=types.SimpleNamespace(
            get=lambda: False))
        self.h.Main = load_pinned(self.root, "srt/mem_cache/hisparse_memory_pool.py", "HiSparseMHAMainPool", dict(
            __name__=MAIN, MHATokenToKVPool=self.h.MHA, torch=torch, unwrap_write_loc=lambda loc: (loc, None, None)))
        self.h.Sparse = load_pinned(self.root, device.MEMORY_POOL, "MiniMaxSparseKVPool", dict(
            __name__=MEMORY, KVCache=object, MHATokenToKVPool=self.h.MHA, MHATokenToKOnlyPool=KOnlyPool,
            envs=envs, torch=torch, _is_cuda=False))
        self.modules[MAIN].HiSparseMHAMainPool = self.h.Main
        self.modules[MEMORY].MiniMaxSparseKVPool = self.h.Sparse
        self.h.Coordinator.swap_in_selected_blocks = load_pinned(
            self.root, arena.COORDINATOR, "HiSparseCoordinator.swap_in_selected_blocks",
            dict(__name__=COORDINATOR, torch=torch,
                 load_blocks_to_device_buffer_mha=lambda **kw: self.calls["swap"].append(kw)))
        self.h.Backend._hisparse_swap_in_blocks = load_pinned(
            self.root, device.BACKEND, "MiniMaxSparseAttnBackend._hisparse_swap_in_blocks",
            dict(__name__="sglang.srt.layers.attention.minimax_sparse_backend"))
        self.assertTrue(device.install(self.CONFIG, "c" * 64))
        self.pool, self.allocator = build_pool(self.h)

    def test_pinned_constructor_routing_and_fused_store(self):
        main, resident = self.pool.main_pool, self.pool._rlh_resident_pool
        self.assertEqual((main.layer_num, main.model_layer_ids, resident.model_layer_ids), (2, (2, 4), (1, 3, 5)))
        self.assertEqual(self.pool.index_k_pool.k_buffer[0].shape[0], 34, "index K stays logical for all layers")
        loc = torch.tensor([3, 7])
        for layer_id in SPARSE:
            k = torch.full((2, 1, 4), float(layer_id), dtype=torch.bfloat16)
            self.pool.set_fused_kv_index_buffer(types.SimpleNamespace(layer_id=layer_id), loc, k, -k, k, None)
        rows = main.full_to_hisparse_device_index_mapping[loc]
        for plane, layer_id in enumerate((2, 4)):
            self.assertTrue(torch.all(self.pool.get_key_buffer(layer_id)[rows] == layer_id))
        for layer_id in (1, 3, 5):
            self.assertTrue(torch.all(self.pool.get_value_buffer(layer_id)[loc] == -layer_id))
        for layer_id in SPARSE:
            index = self.pool.index_k_pool.k_buffer[self.pool.index_k_layer_id_mapping[layer_id]]
            self.assertTrue(torch.all(index[loc] == layer_id))
        self.assertIs(self.pool.get_key_buffer(0), self.pool.dense_pool.k_buffer[0])
        k, _ = self.pool.get_kv_size_bytes()
        self.assertGreater(k, sum(t.numel() * 2 for t in main.k_buffer + self.pool.dense_pool.k_buffer))

    def test_pinned_decode_chain_and_backup_walk_only_the_offloaded_planes(self):
        coordinator, _ = self.coordinator(self.allocator)
        backend = self.h.Backend(self.pool)
        backend.hisparse_coordinator, backend.block_size_k = coordinator, 2
        batch = types.SimpleNamespace(req_pool_indices=torch.tensor([1]), seq_lens=torch.tensor([30]))
        topk = torch.tensor([[[5, 1]]], dtype=torch.int32)
        out = backend._hisparse_swap_in_blocks(batch, topk, 4)
        kw = self.calls["swap"][-1]
        main, host = self.pool.main_pool, coordinator.mem_pool_host
        self.assertIs(kw["device_buffer_k"], main.k_buffer[1])
        self.assertIs(kw["host_cache_k"], host.k_buffer[1])
        self.assertEqual(kw["device_buffer_tokens"].data_ptr(), coordinator.req_device_buffer_tokens[1].data_ptr())
        self.assertEqual(kw["lru_slots"].data_ptr(), coordinator.lru_slots[1].data_ptr())
        self.assertEqual(tuple(out.shape), (1, 1, 4))
        self.assertIsNone(backend._hisparse_swap_in_blocks(batch, topk, 3))
        self.assertEqual(len(self.calls["swap"]), 1, "resident layers never launch the swap kernel")
        backup = load_pinned(self.root, "srt/mem_cache/pool_host/mha.py",
                             "HiSparseMHATokenToKVPoolHost.backup_from_device_all_layer",
                             dict(transfer_kv_per_layer=lambda **kw: self.calls["transfer"].append(kw)))
        backup(types.SimpleNamespace(k_buffer=host.k_buffer, v_buffer=host.v_buffer, token_stride_size=8), main,
               torch.tensor([0]), torch.tensor([10]), "kernel")
        self.assertEqual([(c["src_k"].data_ptr(), c["dst_k"].data_ptr()) for c in self.calls["transfer"]],
                         [(main.k_buffer[p].data_ptr(), host.k_buffer[p].data_ptr()) for p in range(2)])


class DryRunContracts(unittest.TestCase):
    def tree(self, tmp, **overrides):
        files = {}
        for rel, symbol, params in device.SYMBOLS + arena.SYMBOLS:
            files.setdefault(rel, [])
            files[rel].append((symbol, params or ("self",)))
        for rel, symbols in files.items():
            lines, classes = [], {}
            for symbol, params in symbols:
                cls, _, name = symbol.rpartition(".")
                classes.setdefault(cls, []).append((name, params))
            for cls, members in classes.items():
                indent = "    " if cls else ""
                if cls:
                    lines.append(f"class {cls}:")
                for name, params in members:
                    lines.append(f"{indent}def {name}({', '.join(params)}, loc_mapping=None, main_pool_cls=None, "
                                 "enable_hisparse=False, host_to_device_ratio=2, start_layer=None, end_layer=None):")
                    lines.append(f"{indent}    {BODIES.get(name, 'pass')}")
            path = Path(tmp) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(overrides.get(rel, "\n".join(lines) + "\n"))

    def test_contracts_hold_on_the_synthetic_tree_and_flag_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.tree(tmp)
            failed = [r for r in device.resolve_symbols(tmp) if r["status"] != "ok"]
            self.assertEqual(failed, [])
            text = (Path(tmp) / device.BACKEND).read_text()
            self.tree(tmp, **{device.BACKEND: text.replace("loc_mapping=lm", "lm")})
            failed = {r["symbol"] for r in device.resolve_symbols(tmp) if r["status"] != "ok"}
            self.assertEqual(failed, {f"forward_extend imports {device.PREFILL} at call time and passes "
                                      "loc_mapping by keyword"})
            status = device.verify_digests(tmp)
            self.assertEqual(status["srt/model_executor/pool_configurator.py"], "mismatch")

    @unittest.skipUnless("SGLANG_SRC" in os.environ, "set SGLANG_SRC to a pinned checkout")
    def test_pinned_checkout_resolves(self):
        args = ["dry-run", "--sglang-src", os.environ["SGLANG_SRC"], "--config",
                str(Path(__file__).parent / "fixtures" / "combined_arms" / "combined_example.json")]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(device.main(args), 0)
        report = json.loads(out.getvalue())
        self.assertEqual((report["status"], report["placement"]["offload"]), ("pass", [5, 15]))
        self.assertEqual(report["budget_units"]["per_request"]["history_payload_bytes"], 4 * 2**20)
        self.assertEqual(report["budget_units"]["per_request"]["tail_payload_bytes"], 128 * 2**10)


BODIES = {
    "__init__": "from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseMHAMainPool",
    "forward_extend": ("from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import "
                       "minimax_sparse_prefill; minimax_sparse_prefill(1, 2, 3, loc_mapping=lm)"),
    "forward_decode": "return self._hisparse_swap_in_blocks",
}


if __name__ == "__main__":
    unittest.main()