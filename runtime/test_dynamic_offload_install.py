"""CPU smoke tests of ``dynamic_offload_device.install`` on fake pinned SGLang modules, for MiniMax-M3 and DeepSeek-V4
C4: sizing, the pool split, store and read routing, coordinator hooks, the TP switch through /set_internal_state
and slot release."""
import collections
from contextlib import nullcontext
import contextlib
import io
import json
import sys
import types
import unittest
from unittest.mock import patch

from . import dynamic_offload_device as dev
from . import study_config as sc

try:
    import torch
except ImportError:
    torch = None

ns = types.SimpleNamespace
H, PAGE, SLOTS, SIZE, LAYERS = 8, 4, 5, 32, (1, 2, 3, 4, 5)       # layer 0 is dense, 1..5 sparse
FIXED = {"schema": sc.SCHEMA, "placement": {"mode": "fixed_layers", "offload_layers": [2, 3]}}
DYNAMIC = {"schema": sc.SCHEMA, "placement": {"mode": "fixed_layers", "offload_layers": [2, 3],
                                              "alternate_offload_layers": [[1, 4]]},
           "features": {"route_indirect": True, "dynamic_offload": True}}
# DeepSeek-V4: C4 layers 2, 4, 6, 7; 64-record pages, 256 device records, 2048 logical tokens, top-4 records.
DS_RATIOS, DS_FULL, DS_DEVICE, DS_TOP = (0, 0, 4, 128, 4, 128, 4, 4), 2048, 256, 4
DS_FIXED = {"schema": sc.SCHEMA, "placement": {"mode": "fixed_layers", "offload_layers": [2, 4]}}
DS_DYNAMIC = {"schema": sc.SCHEMA, "placement": {"mode": "fixed_layers", "offload_layers": [2, 4],
                                                 "alternate_offload_layers": [[4, 6]]},
              "features": {"route_indirect": True, "dynamic_offload": True}}
KVC, CONF = "sglang.srt.mem_cache.kv_cache_configurator", "sglang.srt.model_executor.pool_configurator"
COORD, STATE = "sglang.srt.managers.hisparse_coordinator", "sglang.srt.managers.scheduler_components.internal_state"
BACKEND = "sglang.srt.layers.attention.minimax_sparse_backend"
OPS = "sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse"
DSV4, DSV4_MODEL = "sglang.srt.mem_cache.deepseek_v4_memory_pool", "sglang.srt.models.deepseek_v4"


def kv_rows(uid, layer, tokens):
    """K rows [n, 1, 4] naming (request, layer, token), exact in BF16; V rows are their negation."""
    return torch.tensor([[[uid, layer, token, 1]] for token in tokens], dtype=torch.bfloat16)


def adopt(module, *objects):
    """Give fake classes and functions the pinned module name, as the hooks' stacking check requires."""
    for obj in objects:
        obj.__module__ = module.__name__
        for value in vars(obj).values() if isinstance(obj, type) else ():
            if callable(value):
                value.__module__ = module.__name__
        setattr(module, obj.__name__, obj)


def fake_sglang():
    m = {name: types.ModuleType(name) for name in (
        "sglang", "sglang.srt", "sglang.srt.constants", "sglang.srt.mem_cache", "sglang.srt.mem_cache.memory_pool",
        "sglang.srt.mem_cache.hisparse_memory_pool", "sglang.srt.mem_cache.sparsity", "sglang.srt.configs",
        "sglang.srt.configs.model_config", KVC, "sglang.srt.model_executor", CONF, "sglang.srt.managers", COORD,
        "sglang.srt.managers.scheduler_components", STATE, "sglang.srt.layers", "sglang.srt.layers.attention",
        BACKEND, "sglang.srt.layers.attention.minimax_sparse_ops", OPS, "sglang.srt.runtime_context", DSV4,
        "sglang.srt.models", DSV4_MODEL)}
    m["sglang"].__file__ = "/nonexistent/sglang/__init__.py"
    m["sglang.srt.constants"].GPU_MEMORY_TYPE_KV_CACHE = "kv"
    pool_mod, main_mod = m["sglang.srt.mem_cache.memory_pool"], m["sglang.srt.mem_cache.hisparse_memory_pool"]

    class MHATokenToKVPool:
        def __init__(self, size, page_size, dtype, head_num, head_dim, layer_num, device, enable_memory_saver,
                     start_layer=None, end_layer=None):
            self.size, self.page_size, self.dtype, self.store_dtype, self.device = size, page_size, dtype, dtype, device
            self.head_num, self.head_dim, self.v_head_dim, self.layer_num = head_num, head_dim, head_dim, layer_num
            self.start_layer, self.end_layer = start_layer or 0, end_layer
            self.kv_cache_layout, self.use_hnd, self.memory_saver_adapter = "nhd", False, ns(region=nullcontext)
            self.custom_mem_pool, self.enable_custom_mem_pool, self.alt_stream = None, False, None
            self.device_module, self.row_dim = None, head_num * head_dim
            self.v_row_dim, self.is_quantized_kv_cache, self.post_capture_active = self.row_dim, False, False
            self._create_buffers_normal()
            self.mem_usage = sum(self.get_kv_size_bytes()) / 2 ** 30

        def _kv_buffer_shapes(self):
            rows = self.size + self.page_size
            return (rows, self.head_num, self.head_dim), (rows, self.head_num, self.v_head_dim)

        def _create_buffers_normal(self):
            k, v = self._kv_buffer_shapes()
            self.k_buffer = [torch.zeros(k, dtype=self.store_dtype) for _ in range(self.layer_num)]
            self.v_buffer = [torch.zeros(v, dtype=self.store_dtype) for _ in range(self.layer_num)]

        def get_kv_size_bytes(self):
            return tuple(sum(t.numel() * t.element_size() for t in ts) for ts in (self.k_buffer, self.v_buffer))

        def get_key_buffer(self, layer_id):
            return self.k_buffer[layer_id - self.start_layer]

        def get_value_buffer(self, layer_id):
            return self.v_buffer[layer_id - self.start_layer]

        def get_kv_buffer(self, layer_id):
            return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

        def set_kv_buffer(self, layer, loc_info, cache_k, cache_v, k_scale=None, v_scale=None, layer_id_override=None):
            index = (layer.layer_id if layer_id_override is None else layer_id_override) - self.start_layer
            self.k_buffer[index][loc_info], self.v_buffer[index][loc_info] = cache_k, cache_v

    class HiSparseMHAMainPool(MHATokenToKVPool):
        def __init__(self, size, page_size, dtype, head_num, head_dim, layer_num, device, enable_memory_saver,
                     start_layer=None, end_layer=None):
            super().__init__(size, page_size, dtype, head_num, head_dim, layer_num, device, enable_memory_saver,
                             start_layer, end_layer)
            self.full_to_hisparse_device_index_mapping = None

        def translate_loc_from_full_to_compressed(self, full):
            return full

        def set_kv_buffer(self, layer, loc, cache_k, cache_v, *args, **kwargs):
            super().set_kv_buffer(layer, self.full_to_hisparse_device_index_mapping[loc], cache_k, cache_v, *args,
                                  **kwargs)

    class MiniMaxSparseKVPool:
        """The pinned wrapper restated: a dense pool and the HiSparse main pool (call-time import), no index pools."""

        def __init__(self, size, page_size, dtype, head_num, head_dim, idx_head_dim, dense_layer_ids,
                     sparse_layer_ids, device, disable_value_sparse_layer_ids=None, enable_memory_saver=False,
                     index_dtype=None, start_layer=None, end_layer=None, main_pool_cls=MHATokenToKVPool,
                     index_kv_pool_cls=None, index_k_pool_cls=None, enable_hisparse=False, host_to_device_ratio=2):
            from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseMHAMainPool as main_cls
            dense = [x for x in dense_layer_ids if start_layer <= x < end_layer]
            local = [x for x in sparse_layer_ids if start_layer <= x < end_layer]
            common = dict(page_size=page_size, dtype=dtype, head_num=head_num, head_dim=head_dim, device=device,
                          enable_memory_saver=enable_memory_saver)
            self.dense_pool = main_pool_cls(size=size * host_to_device_ratio, layer_num=len(dense),
                                            start_layer=start_layer, end_layer=start_layer + len(dense), **common)
            self.main_pool = main_cls(size=size, layer_num=len(local), start_layer=local[0], end_layer=end_layer,
                                      **common)
            self.main_pool.model_layer_ids, self._dense = tuple(local), set(dense)
            self.mem_usage = self.main_pool.mem_usage + self.dense_pool.mem_usage

        def _wait_for_layer(self, layer_id):
            pass

        def _pool_for(self, layer_id):
            return self.dense_pool if layer_id in self._dense else self.main_pool

        def get_key_buffer(self, layer_id):
            return self._pool_for(layer_id).get_key_buffer(layer_id)

        def get_value_buffer(self, layer_id):
            return self._pool_for(layer_id).get_value_buffer(layer_id)

        def get_kv_buffer(self, layer_id):
            return self._pool_for(layer_id).get_kv_buffer(layer_id)

        def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None):
            self._pool_for(layer.layer_id).set_kv_buffer(layer, loc, cache_k, cache_v, k_scale, v_scale)

        def get_kv_size_bytes(self):
            sizes = [pool.get_kv_size_bytes() for pool in (self.main_pool, self.dense_pool)]
            return sum(k for k, _ in sizes), sum(v for _, v in sizes)

    def _set_kv_buffer_impl(k, v, k_cache, v_cache, indices, row_dim, store_dtype, device_module, size_limit,
                            alt_stream=None, v_row_dim=None):
        k_cache[indices], v_cache[indices] = k, v

    adopt(pool_mod, MHATokenToKVPool, MiniMaxSparseKVPool, _set_kv_buffer_impl)
    pool_mod.unwrap_write_loc = lambda loc: (loc, None, None)
    adopt(main_mod, HiSparseMHAMainPool)
    m["sglang.srt.mem_cache.sparsity"].parse_hisparse_config = lambda: ns(device_buffer_size=H, host_to_device_ratio=2)
    models = m["sglang.srt.configs.model_config"]
    models.is_minimax_sparse = lambda hf_config: True
    models.uses_minimax_m3_nvfp4_native_stack = lambda quantization: False
    models.get_minimax_sparse_attention_config = lambda hf_config: hf_config
    models.get_minimax_sparse_layer_ids = lambda config: ([0], list(LAYERS))

    class DefaultPoolConfigurator:
        def _compute_cell_size(self, kvc, num_layers):
            return 1000

    class DSV4PoolConfigurator:
        c4_shrink_factor = 2

        def _get_bytes_per_full_token(self):
            return 100.0

        def _get_paged_kv_bytes_per_token(self, compress_ratio):
            return 37440 / 64 / compress_ratio

        def num_layers(self, ratio):
            return sum(r == ratio for r in DS_RATIOS)

    class KVCacheConfigurator:
        model_config, is_draft_worker = ns(hf_config=None), False

        def _build_token_to_kv_pool(self, sizes, is_dsa_model, is_dsv4_model, req_to_token_pool):
            if is_dsv4_model:
                return m[DSV4].DeepSeekV4TokenToKVPool(full_size=DS_FULL, c4_size=sizes, ratios=DS_RATIOS,
                                                       page_size=256)
            return pool_mod.MiniMaxSparseKVPool(
                size=sizes, page_size=PAGE, dtype=torch.bfloat16, head_num=1, head_dim=4, idx_head_dim=4,
                dense_layer_ids=[0], sparse_layer_ids=list(LAYERS), device="cpu", start_layer=0, end_layer=6,
                enable_hisparse=True, host_to_device_ratio=2)

    adopt(m[CONF], DefaultPoolConfigurator, DSV4PoolConfigurator)
    adopt(m[KVC], KVCacheConfigurator)

    class Event:
        def record(self, stream=None):
            pass

        def wait(self, stream=None):
            pass

    coord = m[COORD]
    coord.device_module = ns(Event=Event, stream=lambda stream: nullcontext())
    coord.HiSparseAct = collections.namedtuple("HiSparseAct", "start_event finish_event req")

    class HiSparseCoordinator:
        """The pinned coordinator restated without speculation, for MiniMax-M3 (``ratio`` 1) and DeepSeek-V4 C4
        (``ratio`` 4): buffer allocation and growth, the location remap, request release, and the stock swap of
        short requests (direct control only)."""

        def __init__(self, req_to_token_pool, allocator, host, *, ratio=1, page=PAGE, hot=H, width=H):
            t, slots, layers = torch, req_to_token_pool.req_to_token.shape[0], host.layer_num
            kv = allocator.get_kvcache()
            self.req_to_token_pool, self.mem_pool_host, self.token_to_kv_pool_allocator = req_to_token_pool, host, \
                allocator
            self.mem_pool_device = kv.main_pool if ratio == 1 else kv.c4_kv_pool
            self.mem_pool_device.full_to_hisparse_device_index_mapping = allocator.full_to_hisparse_device_index_mapping
            self.is_m3_hisparse, self.is_dsv4_hisparse, self.compress_ratio, self.page = ratio == 1, ratio == 4, \
                ratio, page
            self.skip_io, self.enable_prefetch, self.spec_swap = False, False, ns(enabled=False)
            self.tp_world_size, self.tp_group, self.device_buffer_size, self.padded_buffer_size = 1, None, hot, \
                hot + page
            self.req_to_device_buffer = t.zeros((slots, hot + page), dtype=t.int64)
            self.req_device_buffer_size = t.zeros(slots, dtype=t.int64)
            self.req_to_host_pool = t.full((slots, 128), -1, dtype=t.int64)
            self.req_to_host_pool_allocated_len = t.zeros(slots, dtype=t.int64)
            self.req_device_buffer_tokens = t.full((layers, slots, hot + page), -1, dtype=t.int32)
            self.req_device_buffer_token_locs = t.full((layers, slots, hot + page), -1, dtype=t.int32)
            self.lru_slots = t.arange(hot, dtype=t.int16).repeat(layers, slots, 1)
            self.top_k_device_locs_buffer = t.full((slots, width), -1, dtype=t.int32)
            self.num_real_reqs = t.zeros(1, dtype=t.int32)
            self.decode_backup_stream = self.decode_producer_stream = self.write_staging_stream = None
            self._skip_first_backup, self.ack_staging_queue, self.native = [False] * slots, [], []

        def admit_request_direct(self, req):
            raise AssertionError("disaggregated admission is not exercised")

        def admit_request_into_staging(self, req):
            self.native.append("staging")
            self.ack_staging_queue.append(coord.HiSparseAct(None, None, req))

        def collect_ready_reqs(self):
            ready = [act.req for act in self.ack_staging_queue]
            self.ack_staging_queue.clear()
            for req in ready:
                self.alloc_device_buffer(req)
                self._skip_first_backup[req.kv.req_pool_idx] = True
            return ready

        def alloc_device_buffer(self, req):
            rid, hot, page = req.kv.req_pool_idx, self.device_buffer_size, self.page
            length = req.extend_end if self.is_dsv4_hisparse else req.kv.kv_allocated_len
            size = self.padded_buffer_size if self.is_dsv4_hisparse else min(-(-length // page) * page, hot)
            size = self.padded_buffer_size if size == hot else size
            compressed = self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[rid, :length].long())
            rows = self.token_to_kv_pool_allocator.alloc_device_buffer(compressed, size)
            self.req_to_device_buffer[rid, :size], self.req_device_buffer_size[rid] = rows, size
            self.req_device_buffer_tokens[:, rid, :hot] = torch.arange(hot, dtype=torch.int32)
            self.req_device_buffer_token_locs[:, rid, :size] = rows.to(torch.int32)

        def _grow_device_buffers(self, seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu):
            hot, page = self.device_buffer_size, self.page
            for rid, seq in zip(req_pool_indices_cpu.tolist(), seq_lens_cpu.tolist()):
                cap, new = int(self.req_device_buffer_size[rid]), min(-(-seq // page) * page, hot)
                new = self.padded_buffer_size if new == hot else new
                if seq <= hot and new > cap:
                    rows = self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(new - cap)
                    self.req_to_device_buffer[rid, cap:new], self.req_device_buffer_size[rid] = rows, new
                    self.req_device_buffer_token_locs[:, rid, cap:new] = rows.to(torch.int32)
            return self.req_to_device_buffer[req_pool_indices, (seq_lens - 1).clamp(max=hot)]

        def map_last_loc_to_buffer(self, seq_lens, out_cache_loc, req_pool_indices, seq_lens_cpu,
                                   req_pool_indices_cpu):
            self._eager_backup_previous_token(seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu)
            mapping, hot = self.mem_pool_device.full_to_hisparse_device_index_mapping, self.device_buffer_size
            if not self.is_dsv4_hisparse:
                reserved = self._grow_device_buffers(seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu)
                self.req_device_buffer_token_locs[:, req_pool_indices, hot] = reserved.to(torch.int32)
                mapping[out_cache_loc] = reserved
                return
            active = seq_lens % self.compress_ratio == 0
            if active.any():                                    # the step's token closes a C4 record
                rids = req_pool_indices[active]
                reserved = self.req_to_device_buffer[rids, (seq_lens[active] // 4 - 1).clamp(max=hot)]
                self.req_device_buffer_token_locs[:, rids, hot] = reserved.to(torch.int32)
                mapping[self.token_to_kv_pool_allocator.get_last_loc_compressed(out_cache_loc[active])] = reserved

        def _eager_backup_previous_token(self, seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu):
            self.native.append("eager_backup")

        def wait_for_pending_backup(self):
            pass

        def request_finished(self, req):
            rid, length = req.kv.req_pool_idx, req.kv.kv_allocated_len
            owned = self.req_device_buffer_token_locs[:, rid, :int(self.req_device_buffer_size[rid])].reshape(-1)
            self.token_to_kv_pool_allocator.free_hisparse_indices(torch.unique(owned[owned > 0]).long())
            mapping = self.mem_pool_device.full_to_hisparse_device_index_mapping
            mapping[self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[rid, :length].long())] = 0
            for table in (self.req_device_buffer_tokens[:, rid], self.req_device_buffer_token_locs[:, rid],
                          self.req_to_host_pool[rid]):
                table.fill_(-1)
            self.req_to_device_buffer[rid], self.req_device_buffer_size[rid] = 0, 0
            self.lru_slots[:, rid] = torch.arange(self.device_buffer_size, dtype=torch.int16)

        def swap_in_selected_blocks(self, req_pool_indices, seq_lens, top_k_blocks, layer_id, sparse_block_size):
            self.native.append(("swap", layer_id))
            return self.req_device_buffer_token_locs[layer_id][req_pool_indices[:, None], top_k_blocks.long()]

        def swap_in_selected_pages(self, req_pool_indices, compressed_seq_lens, top_k_result, layer_id):
            self.native.append(("swap_pages", layer_id))         # short requests: every record has its hot slot
            top_k = top_k_result.long().clamp(min=0)
            locs = self.req_device_buffer_token_locs[layer_id][req_pool_indices[:, None], top_k]
            return torch.where(top_k_result >= 0, locs, -1)

    adopt(coord, HiSparseCoordinator)

    class UpdateRejected(Exception):
        pass

    class _UpdateHandler:
        def __init__(self, validate, apply):
            self.validate, self.apply = validate, apply

    class SchedulerInternalStateController:
        def __init__(self, scheduler):
            self.scheduler = scheduler

        def _update_handlers(self):
            return {}

        def set_internal_state(self, recv_req):
            """The pinned control flow: validate every key, then apply; False means updated=False."""
            handlers = self._update_handlers()
            try:
                for key, value in recv_req.server_args.items():
                    if key not in handlers:
                        raise UpdateRejected(f"Updating {key} is not supported.")
                    handlers[key].validate(key, value)
            except UpdateRejected:
                return False
            for key, value in recv_req.server_args.items():
                handlers[key].apply(key, value)
            return True

    adopt(m[STATE], UpdateRejected, _UpdateHandler, SchedulerInternalStateController)

    class MiniMaxSparseAttnBackend:
        """Decode restated: store the new token, swap in (or read a resident layer directly), gather main KV."""

        def __init__(self, kv_pool, coordinator):
            self.kv_pool, self.coordinator = kv_pool, coordinator

        def _hisparse_swap_in_blocks(self, forward_batch, topk_idx, layer_id):
            return self.coordinator.swap_in_selected_blocks(forward_batch.req_pool_indices, forward_batch.seq_lens,
                                                            topk_idx, layer_id, 1)

        def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
            self.kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
            rows = self._hisparse_swap_in_blocks(forward_batch, q, layer.layer_id)
            if rows is None:
                req_to_token = self.coordinator.req_to_token_pool.req_to_token
                rows = req_to_token[forward_batch.req_pool_indices[:, None], q.long()]
            keys, values = self.kv_pool.get_kv_buffer(layer.layer_id)
            return keys[rows.long()], values[rows.long()]

    def minimax_sparse_prefill(q, k_cache, v_cache, *, locs, loc_mapping):
        rows = locs if loc_mapping is None else loc_mapping[locs]
        return k_cache[rows], v_cache[rows]

    adopt(m[BACKEND], MiniMaxSparseAttnBackend)
    adopt(m[OPS], minimax_sparse_prefill)
    context = m["sglang.srt.runtime_context"]
    context.get_schedule = lambda: ns(disable_overlap_schedule=True)
    context.get_exec = lambda: ns(overlap=ns(enable_two_batch_overlap=False, enable_single_batch_overlap=False),
                                  graph=ns(cuda_graph_config=ns(decode=ns(backend="full"),
                                                                prefill=ns(backend="disabled"))))
    context.get_memory = lambda: ns(disable_radix_cache=True)
    context.get_spec = lambda: ns(speculative_algorithm=None)
    context.get_disagg = lambda: ns(disaggregation_mode="null", enable_pdmux=False)
    context.get_parallel = lambda: ns(pp_size=1, attn_tp_size=1)
    fake_dsv4(m, coord)
    return m


def fake_dsv4(m, coord):
    """DeepSeek-V4 restated: the C4 pools, layer mapping and extra key buffer, and one C4 layer's compressor
    store, C4 indexer swap-in and attention read (``MQALayer.forward``)."""
    ds = m[DSV4]
    v4 = ns(value="v4", page_bytes=lambda page: -(-page * 584 // 576) * 576)

    def _num_dsv4_physical_kv_pages(size, physical_page_size, logical_page_size):
        return -(-(size + logical_page_size) // physical_page_size)

    class DeepSeekV4SingleKVPool:
        def __init__(self, size, page_size, dtype, qk_nope_head_dim, qk_rope_head_dim, layer_num, device,
                     enable_memory_saver, start_layer=None, end_layer=None, kv_layout=v4, global_page_size=None):
            self.size, self.page_size, self.dtype, self.store_dtype = size, page_size, dtype, dtype
            self.layer_num, self.device, self.kv_layout, self.global_page_size = layer_num, device, kv_layout, \
                global_page_size
            self.kv_buffer = [self.create_buffer(num_pages=ds._num_dsv4_physical_kv_pages(
                size, page_size, global_page_size)) for _ in range(layer_num)]

        def create_buffer(self, *, num_pages):
            return torch.zeros((num_pages, self.kv_layout.page_bytes(self.page_size)), dtype=torch.uint8)

        def get_key_buffer(self, layer_id):
            return self.kv_buffer[layer_id]

    class HiSparseC4DevicePool(DeepSeekV4SingleKVPool):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.data_ptrs = torch.tensor([x.data_ptr() for x in self.kv_buffer], dtype=torch.uint64)
            self.compress_ratio, self.full_to_hisparse_device_index_mapping = 4, None

        def translate_loc_from_full_to_compressed(self, full_indices):
            return full_indices[(full_indices + 1) % 4 == 0] // 4

        def translate_loc_to_hisparse_device(self, compressed_indices):
            return self.full_to_hisparse_device_index_mapping[compressed_indices].to(torch.int32)

        def _translate_loc_to_hisparse_device(self, compressed_indices):
            return self.full_to_hisparse_device_index_mapping[compressed_indices]

    class DeepSeekV4TokenToKVPool:
        def __init__(self, full_size, c4_size, ratios, page_size):
            self.full_size, self.compression_ratios, self.uniform_fp8 = full_size, list(ratios), False
            self._stage_start, self._stage_end = 0, len(ratios)
            self.kv_pools = {4: self._make_kv_pool(
                size=c4_size, page_size=page_size // 4, dtype=torch.uint8, layer_num=sum(r == 4 for r in ratios),
                device="cpu", enable_memory_saver=False, global_page_size=page_size, cls=ds.HiSparseC4DevicePool,
                kv_layout=v4)}
            self.c4_kv_pool = self.kv_pools[4]
            self._init_compressed_layer_mapping()

        def _make_kv_pool(self, *, size, page_size, dtype, layer_num, device, enable_memory_saver, global_page_size,
                          cls, kv_layout):
            return cls(size, page_size, dtype, 448, 64, layer_num, device, enable_memory_saver, kv_layout=kv_layout,
                       global_page_size=global_page_size)

        def _init_compressed_layer_mapping(self):
            count, self.layer_mapping = 0, [None] * len(self.compression_ratios)
            for idx in range(self._stage_start, self._stage_end):
                ratio = self.compression_ratios[idx]
                self.layer_mapping[idx] = ds.DeepSeekV4LayerItem(compress_ratio=ratio, compress_layer_id=count if
                                                                 ratio == 4 else 0,
                                                                 compress_kv_pool=self.kv_pools.get(ratio))
                count += ratio == 4

        def get_extra_key_buffer(self, layer_id):
            _, index, pool = self.layer_mapping[layer_id]
            return pool.get_key_buffer(index)

    adopt(ds, DeepSeekV4SingleKVPool, HiSparseC4DevicePool, DeepSeekV4TokenToKVPool, _num_dsv4_physical_kv_pages)
    ds.DeepSeekV4LayerItem = collections.namedtuple("DeepSeekV4LayerItem",
                                                    "compress_ratio compress_layer_id compress_kv_pool")

    class MQALayer:
        """One C4 layer of the pinned model restated: the compressor stores the records the batch's tokens close
        (location 0 for the others), the C4 indexer swaps the selection in (decode) or translates it (prefill),
        and attention reads the selected records."""

        def __init__(self, layer_id, kv, coordinator):
            self.layer_id, self.kv, self.coordinator = layer_id, kv, coordinator

        def forward(self, x, positions, forward_batch):
            kv, layer = self.kv, self.layer_id
            _, index, pool = kv.layer_mapping[layer]
            loc = x["closing"]
            if hasattr(pool, "translate_loc_to_hisparse_device"):
                loc = pool._translate_loc_to_hisparse_device(loc)
            write_records(kv.get_extra_key_buffer(layer), loc, x["records"])
            if forward_batch.forward_mode.is_decode():
                rows = self.coordinator.swap_in_selected_pages(
                    req_pool_indices=forward_batch.req_pool_indices, compressed_seq_lens=forward_batch.seq_lens // 4,
                    top_k_result=x["top_k"], layer_id=index)
            else:
                rows = kv.c4_kv_pool.translate_loc_to_hisparse_device(x["pages"]).to(torch.int32)
            return read_records(kv.get_extra_key_buffer(layer), rows)

    adopt(m[DSV4_MODEL], MQALayer)


def c4_parts(buffer):
    """(value, scale) views [pages, 64, 576] and [pages, 64, 8] of a C4 page buffer."""
    pages = buffer.view(-1, buffer.shape[-1])
    return pages[:, :64 * 576].unflatten(1, (64, 576)), pages[:, 64 * 576:64 * 584].unflatten(1, (64, 8))


def write_records(buffer, rows, idents):
    """C4 records naming (request, layer, record) in their first value bytes, with a check byte in the scale."""
    (value, scale), rows = c4_parts(buffer), rows.long()
    value[rows // 64, rows % 64, :3] = idents.to(torch.uint8)
    scale[rows // 64, rows % 64, 0] = (idents.sum(-1) % 251).to(torch.uint8)


def read_records(buffer, rows):
    """Per graph row, the (request, layer, record) each selected row holds; None for -1, 'scale' if torn."""
    value, scale = c4_parts(buffer)
    out = []
    for row in rows.tolist():
        out.append([None if r < 0 else tuple(value[r // 64, r % 64, :3].tolist())
                    if int(scale[r // 64, r % 64, 0]) == int(value[r // 64, r % 64, :3].long().sum()) % 251
                    else "scale" for r in row])
    return out


class Allocator:
    """The pinned HiSparse token allocator restated: logical rows, hot rows (C4 records for ``ratio`` 4), the
    logical -> hot mapping and ``alloc_device_buffer``'s reuse of the prefill rows. Page 0 of both is padding."""

    def __init__(self, kv, *, page=PAGE, logical=2 * SIZE, hot=SIZE, ratio=1):
        self.kv, self.size_full, self._size_hisparse, self.page, self.ratio = kv, logical, hot, page, ratio
        self.full_to_hisparse_device_index_mapping = torch.zeros(logical // ratio + page, dtype=torch.int64)
        self.logical, self.hot = list(range(page * ratio, logical)), list(range(page, hot + page))
        self.hisparse_attn_allocator, self.freed = ns(alloc=self.alloc_hot), []

    def get_kvcache(self):
        return self.kv

    def alloc_logical(self, count):
        rows, self.logical = self.logical[:count], self.logical[count:]
        return torch.tensor(rows, dtype=torch.int64)

    def alloc_hot(self, count):
        rows, self.hot = self.hot[:count], self.hot[count:]
        return torch.tensor(rows, dtype=torch.int64)

    def get_last_loc_compressed(self, locs):
        return locs // self.ratio

    def alloc_device_buffer(self, allocated_indices, need_size):
        mapping, page = self.full_to_hisparse_device_index_mapping, self.page
        hot = mapping[allocated_indices]
        mapping[allocated_indices] = 0
        hot = hot[hot > 0]
        if len(hot) >= need_size:
            self.free_hisparse_indices(hot[need_size:])
            return hot[:need_size]
        if len(hot) % page:
            hot = torch.cat([hot, torch.arange(int(hot[-1]) + 1, int(hot[-1]) + page - len(hot) % page + 1)])
        return torch.cat([hot, self.alloc_hot(need_size - len(hot))])

    def free_hisparse_indices(self, rows):
        """Page-granular, as the pinned paged allocator releases ``unique(rows // page_size)``."""
        page = self.page
        pages = sorted(set((rows[rows >= page] // page).tolist()))
        self.freed += pages
        self.hot += [index * page + i for index in pages for i in range(page)]


class Host:
    """HiSparse host pool of the slab layers, rows handed out in order: one K and one V buffer per layer, or one
    page buffer per C4 layer (``paged``)."""

    def __init__(self, layers, *, paged=False):
        self.layer_num, self.next = layers, 0
        if paged:
            self.kv_buffer = [torch.zeros((2, 37440), dtype=torch.uint8) for _ in range(layers)]
            self.data_ptrs = torch.tensor([b.data_ptr() for b in self.kv_buffer], dtype=torch.uint64)
        else:
            self.k_buffer = [torch.zeros((128, 1, 4), dtype=torch.bfloat16) for _ in range(layers)]
            self.v_buffer = [torch.zeros((128, 1, 4), dtype=torch.bfloat16) for _ in range(layers)]

    def alloc_paged_token_slots(self, req_to_host_pool, allocated_len, req_pool_idx, start_pos, num_tokens):
        rows = torch.arange(self.next, self.next + num_tokens)
        self.next += num_tokens
        req_to_host_pool[req_pool_idx, start_pos:start_pos + num_tokens] = rows
        allocated_len[req_pool_idx] = start_pos + num_tokens
        return rows


class Worker(unittest.TestCase):
    """One worker process on fake pinned SGLang: install, size and build the pools, bind the coordinator, then drive
    the scheduler's prefill, staging, decode, switch and release calls; every read checks what it gets."""

    def setUp(self):
        from . import layer_eviction_tp, selection_oracle      # noqa: F401  imported before sys.modules is patched
        self.m, self.out = fake_sglang(), io.StringIO()

        def gather(records, value, group=None):
            records[:] = [value]

        def route(state, c, req_pool_indices, seq_lens, top_k, layer_id, top_k_block_size=1):
            out = c.top_k_device_locs_buffer[:req_pool_indices.size(0), :top_k.size(1)]
            return dev.emulate_route(state, c, req_pool_indices, seq_lens, top_k.tolist(), layer_id, out)

        for context in (patch.dict(sys.modules, self.m), patch.object(dev, "verify_digests", lambda root: {}),
                        patch.object(dev, "route_swap", route),
                        patch.object(torch.cuda, "get_device_capability", lambda *args: (8, 0)),
                        patch.multiple(torch.distributed, all_gather_object=gather, new_group=lambda **kwargs: "gloo",
                                       get_backend=lambda group: "gloo", get_rank=lambda group=None: 0,
                                       get_world_size=lambda group=None: 1),
                        contextlib.redirect_stdout(self.out)):
            self.enterContext(context)
        self.addCleanup(self.reset)

    @staticmethod
    def reset():
        if dev._STATE is not None:
            dev._restore(dev._STATE)
            dev._STATE = None

    def emitted(self):
        return json.loads(self.out.getvalue().splitlines()[-1])

    def finish(self, slot):
        """request_finished frees exactly the pages of the request's plane-relative buffer rows."""
        req, c, freed, page = self.reqs.pop(slot), self.c, len(self.alloc.freed), self.alloc.page
        pages = {row // page for row in c.req_to_device_buffer[slot, :int(c.req_device_buffer_size[slot])].tolist()}
        c.request_finished(req)
        self.assertEqual(self.alloc.freed[freed:], sorted(pages))

    def set_list(self, value, chunked_req=None):
        controller = self.m[STATE].SchedulerInternalStateController(ns(chunked_req=chunked_req))
        return controller.set_internal_state(ns(server_args={dev.LIST_KEY: value}))


@unittest.skipIf(torch is None, "CPU torch unavailable")
class InstallTests(Worker):
    """MiniMax-M3: every read checks the (request, layer, token) of its K and V rows."""

    def boot(self, config):
        m = self.m
        self.assertTrue(dev.install(sc.parse(config), "f" * 64))
        kvc = ns(model_config=ns(hf_config=None, quantization=None, head_dim=4, get_num_kv_heads=lambda tp: 1),
                 server_args=ns(enable_hisparse=True), layer_info=ns(start_layer=0, end_layer=6),
                 kv_cache_dtype=torch.bfloat16)
        self.cell = m[CONF].DefaultPoolConfigurator()._compute_cell_size(kvc, 6)
        rtp = ns(req_to_token=torch.zeros((SLOTS, 64), dtype=torch.int32))
        self.kv = m[KVC].KVCacheConfigurator()._build_token_to_kv_pool(
            sizes=SIZE, is_dsa_model=False, is_dsv4_model=False, req_to_token_pool=rtp)
        self.p, self.alloc, self.reqs = self.kv._rlh_placement, Allocator(self.kv), {}
        self.c = m[COORD].HiSparseCoordinator(rtp, self.alloc, Host(len(self.p.slab)))
        self.backend = m[BACKEND].MiniMaxSparseAttnBackend(self.kv, self.c)
        return self.emitted()

    def admit(self, slot, uid, prompt):
        """Prefill stores and reads of every layer, then staging and the device buffer (collect_ready_reqs)."""
        a, kv, c = self.alloc, self.kv, self.c
        logical = a.alloc_logical(prompt)
        a.full_to_hisparse_device_index_mapping[logical] = a.alloc_hot(-(-prompt // PAGE) * PAGE)[:prompt]
        c.req_to_token_pool.req_to_token[slot, :prompt] = logical.to(torch.int32)
        for layer in (0, *LAYERS):
            k = kv_rows(uid, layer, range(prompt))
            kv.set_kv_buffer(ns(layer_id=layer), logical, k, -k)
            if layer:
                keys, values = self.m[OPS].minimax_sparse_prefill(
                    None, *kv.get_kv_buffer(layer), locs=logical, loc_mapping=a.full_to_hisparse_device_index_mapping)
                self.assertTrue(torch.equal(keys, k) and torch.equal(values, -k), f"prefill layer {layer}")
        self.assertTrue(torch.equal(kv.get_key_buffer(0)[logical], kv_rows(uid, 0, range(prompt))))
        req = ns(rid=uid, extend_end=prompt, kv=ns(req_pool_idx=slot, kv_allocated_len=prompt))
        c.admit_request_into_staging(req)
        self.assertEqual(c.collect_ready_reqs(), [req])
        self.reqs[slot] = req

    def decode(self, picks):
        """One decode step of the slots in ``picks`` (slot -> selected positions, the new token included): the
        scheduler's location remap with its eager backup, then every sparse layer's store, swap-in and read."""
        a, c, slots = self.alloc, self.c, sorted(picks)
        new = a.alloc_logical(len(slots))
        for slot, loc in zip(slots, new.tolist()):
            req = self.reqs[slot]
            c.req_to_token_pool.req_to_token[slot, req.kv.kv_allocated_len] = loc
            req.kv.kv_allocated_len += 1
        rpi, seq = torch.tensor(slots), torch.tensor([self.reqs[slot].kv.kv_allocated_len for slot in slots])
        c.map_last_loc_to_buffer(seq, new, rpi, seq, rpi)
        c.num_real_reqs[0] = len(slots)
        batch = ns(req_pool_indices=rpi, seq_lens=seq, out_cache_loc=new)
        top_k = torch.tensor([picks[slot] for slot in slots], dtype=torch.int32)
        for layer in LAYERS:
            k = torch.cat([kv_rows(self.reqs[slot].rid, layer, [int(n) - 1]) for slot, n in zip(slots, seq)])
            keys, values = self.backend.forward_decode(top_k, k, -k, ns(layer_id=layer), batch)
            expect = torch.stack([kv_rows(self.reqs[slot].rid, layer, picks[slot]) for slot in slots])
            self.assertTrue(torch.equal(keys, expect) and torch.equal(values, -expect), f"decode layer {layer}")

    def test_dynamic_worker_switches_lists_under_load(self):
        bound, p, kv = self.boot(DYNAMIC), self.p, self.kv
        self.assertEqual((p.slab, p.resident, type(kv.main_pool).__name__, self.cell), ((1, 2, 3, 4), (5,),
                                                                                       "SlabMainPool", 1000 + 3 * 16))
        self.assertEqual((bound["event"], bound["cell_size"]["cell_bytes"], bound["ledger"]["slab_rows"],
                          bound["logical_kv"]), ("hisparse_dynamic_offload_bound", self.cell, p.layout.rows,
                                                 dict(tokens=2 * SIZE, hot_rows=SIZE)))
        slab, size = kv.main_pool._rlh_slab, lambda *ts: sum(t.numel() * t.element_size() for t in ts)
        self.assertEqual(kv.get_kv_size_bytes(), (size(slab[0], *kv.dense_pool.k_buffer, *p.resident_pool.k_buffer),
                                                  size(slab[1], *kv.dense_pool.v_buffer, *p.resident_pool.v_buffer)))
        self.admit(1, 1, 6)
        self.admit(2, 2, 10)                                     # longer than the 8 hot slots
        self.decode({1: [0, 3, 5, 6], 2: [0, 8, 9, 10]})          # 8 and 9 load from host on offloaded layers
        self.decode({1: [2, 6, 7, 1], 2: [10, 11, 2, 9]})         # records 6 and 10 flush from their tails
        self.assertFalse(self.set_list(2))
        self.assertFalse(self.set_list(1, chunked_req=object()))
        self.assertEqual((self.emitted()["status"], p.state.routes.offload), ("refused", (2, 3)))
        self.assertTrue(self.set_list(0))
        self.assertEqual(self.emitted()["status"], "unchanged")
        self.assertTrue(self.set_list(1))
        receipt = self.emitted()
        self.assertEqual((receipt["status"], receipt["target"], p.state.routes.offload), ("committed", "1", (1, 4)))
        self.decode({1: [0, 4, 7, 8], 2: [0, 6, 11, 12]})
        self.finish(1)
        self.admit(1, 3, 3)                                      # prefill through the switched routes
        self.decode({1: [0, 1, 2, 3], 2: [3, 10, 12, 13]})
        self.assertTrue(self.set_list(0))
        self.assertEqual((self.emitted()["status"], p.state.routes.offload), ("committed", (2, 3)))
        self.decode({1: [0, 2, 3, 4], 2: [1, 9, 13, 14]})       # slot 1 grows its device buffer
        self.decode({1: [4, 5, 1, 0], 2: [14, 15, 0, 5]})
        for slot in (1, 2):
            self.finish(slot)
        self.assertEqual((p.state.home, self.c.native), ({}, []))
        with self.assertRaisesRegex(RuntimeError, "start a new process"):
            dev.uninstall()

    def test_frozen_worker_keeps_the_stock_paths(self):
        bound, p, kv = self.boot(FIXED), self.p, self.kv
        self.assertEqual((p.slab, p.resident, p.state, bound["ledger"], type(kv.main_pool).__name__),
                         ((2, 3), (1, 4, 5), None, None, "HiSparseMHAMainPool"))
        self.admit(1, 1, 6)
        self.decode({1: [0, 2, 5, 6]})
        self.decode({1: [1, 3, 6, 7]})
        self.assertEqual(self.c.native, ["staging", *["eager_backup", ("swap", 0), ("swap", 1)] * 2])
        self.assertFalse(self.set_list(0))                       # no trigger without features.dynamic_offload
        self.finish(1)


@unittest.skipIf(torch is None, "CPU torch unavailable")
class DSv4InstallTests(Worker):
    """DeepSeek-V4 C4 layers 2, 4, 6, 7: every read checks the (request, layer, record) and scale byte it gets."""
    C4 = tuple(i for i, ratio in enumerate(DS_RATIOS) if ratio == 4)

    def boot(self, config):
        m = self.m
        self.assertTrue(dev.install(sc.parse(config), "f" * 64))
        self.charge = m[CONF].DSV4PoolConfigurator()._get_bytes_per_full_token()
        rtp = ns(req_to_token=torch.zeros((SLOTS, 128), dtype=torch.int32))
        self.kv = m[KVC].KVCacheConfigurator()._build_token_to_kv_pool(
            sizes=DS_DEVICE, is_dsa_model=False, is_dsv4_model=True, req_to_token_pool=rtp)
        self.p, self.reqs = self.kv.c4_kv_pool._rlh_placement, {}
        self.alloc = Allocator(self.kv, page=64, logical=DS_FULL, hot=DS_DEVICE, ratio=4)
        self.c = m[COORD].HiSparseCoordinator(rtp, self.alloc, Host(len(self.p.slab), paged=True), ratio=4, page=64,
                                              width=DS_TOP)
        self.layers = {layer: m[DSV4_MODEL].MQALayer(layer, self.kv, self.c) for layer in self.C4}
        return self.emitted()

    def admit(self, slot, uid, prompt):
        """Prefill: each layer's compressor stores the prompt's records and its indexer translates all of them;
        then staging and the device buffer (collect_ready_reqs)."""
        a, c = self.alloc, self.c
        tokens, records = a.alloc_logical(prompt + 32), prompt // 4      # the request's rows: prompt, then decode
        compressed = tokens[3:4 * records:4] // 4                         # record r closes at position 4r + 3
        a.full_to_hisparse_device_index_mapping[compressed] = a.alloc_hot(-(-records // 64) * 64)[:records]
        c.req_to_token_pool.req_to_token[slot, :prompt] = tokens[:prompt].to(torch.int32)
        closing = torch.where((torch.arange(prompt) + 1) % 4 == 0, tokens[:prompt] // 4, 0)
        for layer, mqa in self.layers.items():
            ids = torch.tensor([[uid, layer, p // 4] if (p + 1) % 4 == 0 else [255] * 3 for p in range(prompt)])
            got = mqa.forward(dict(closing=closing, records=ids, pages=compressed[None]), None,
                              ns(forward_mode=ns(is_decode=lambda: False)))
            self.assertEqual(got, [[(uid, layer, r) for r in range(records)]], f"prefill layer {layer}")
        req = ns(rid=uid, extend_end=prompt, tokens=tokens, kv=ns(req_pool_idx=slot, kv_allocated_len=prompt))
        c.admit_request_into_staging(req)
        self.assertEqual(c.collect_ready_reqs(), [req])
        self.reqs[slot] = req

    def decode(self, picks):
        """One decode step; ``picks`` maps slot -> older record positions, read with the newest record: the
        scheduler's location remap with its eager backup, then every C4 layer's store, swap-in and read."""
        a, c, slots = self.alloc, self.c, sorted(picks)
        for slot in slots:
            req = self.reqs[slot]
            c.req_to_token_pool.req_to_token[slot, req.kv.kv_allocated_len] = int(req.tokens[req.kv.kv_allocated_len])
            req.kv.kv_allocated_len += 1
        rpi, seq = torch.tensor(slots), torch.tensor([self.reqs[slot].kv.kv_allocated_len for slot in slots])
        new = torch.stack([self.reqs[slot].tokens[n - 1] for slot, n in zip(slots, seq.tolist())])
        c.map_last_loc_to_buffer(seq, new, rpi, seq, rpi)
        c.num_real_reqs[0] = len(slots)
        batch = ns(forward_mode=ns(is_decode=lambda: True), req_pool_indices=rpi, seq_lens=seq)
        closing = torch.where(seq % 4 == 0, new // 4, 0)            # the step's token closes record seq // 4 - 1
        chosen = [[n // 4 - 1, *picks[slot]] for slot, n in zip(slots, seq.tolist())]
        top_k = torch.tensor([x + [-1] * (DS_TOP - len(x)) for x in chosen], dtype=torch.int32)
        for layer, mqa in self.layers.items():
            ids = torch.tensor([[self.reqs[slot].rid, layer, n // 4 - 1] if n % 4 == 0 else [255] * 3
                                for slot, n in zip(slots, seq.tolist())])
            got = mqa.forward(dict(closing=closing, records=ids, top_k=top_k), None, batch)
            want = [[None if r < 0 else (self.reqs[slot].rid, layer, r) for r in row]
                    for slot, row in zip(slots, top_k.tolist())]
            self.assertEqual(got, want, f"decode layer {layer}")

    def test_dynamic_c4_worker_switches_lists_under_load(self):
        bound, p, kv = self.boot(DS_DYNAMIC), self.p, self.kv
        self.assertEqual((p.slab, p.resident, type(kv.c4_kv_pool).__name__), ((2, 4, 6), (7,), "SlabC4Pool"))
        self.assertEqual([type(kv.layer_mapping[layer].compress_kv_pool).__name__ for layer in self.C4],
                         ["C4Layer"] * 3 + ["DeepSeekV4SingleKVPool"])
        self.assertEqual((bound["logical_kv"], bound["ledger"]["slab_rows"], bound["cell_size"]["resident_c4_layers"],
                          self.charge), (dict(tokens=DS_FULL, hot_rows=DS_DEVICE), p.layout.rows, 2,
                                         100.0 + 2 * (1 / 4 - 1 / 8) * 37440 / 64 / 4))
        self.admit(1, 1, 20)                                     # 5 records
        self.admit(2, 2, 44)                                     # 11 records, more than the 8 hot slots
        self.decode({1: [0, 2], 2: [3, 9, 0]})                   # record 9 loads from host on offloaded layers
        self.decode({1: [1, 3], 2: [9, 8, 1]})
        self.decode({1: [0], 2: [2, 7, 9]})
        self.decode({1: [2, 3], 2: [9, 10, 4]})                  # closes records 5 and 11 into the tails
        self.decode({1: [4, 0], 2: [10, 6, 0]})                  # flushes them home and to host
        self.assertFalse(self.set_list(2))
        self.assertFalse(self.set_list(1, chunked_req=object()))
        self.assertTrue(self.set_list(1))
        self.assertEqual((self.emitted()["status"], p.state.routes.offload), ("committed", (4, 6)))
        self.decode({1: [4, 1], 2: [10, 3, 7]})
        self.finish(1)
        self.admit(1, 3, 12)                                     # prefill through the switched routes
        self.decode({1: [0, 1], 2: [2, 8, 10]})
        self.decode({1: [1], 2: [0, 5, 9]})                      # closes record 12 of slot 2
        self.assertTrue(self.set_list(0))
        self.assertEqual((self.emitted()["status"], p.state.routes.offload), ("committed", (2, 4)))
        self.decode({1: [0, 1], 2: [11, 3, 6]})                  # flushes record 12 under the source routes
        self.decode({1: [2, 0], 2: [10, 1, 5]})                  # closes record 3 of slot 1
        for slot in (1, 2):
            self.finish(slot)
        self.assertEqual((p.state.home, self.c.native), ({}, []))
        with self.assertRaisesRegex(RuntimeError, "start a new process"):
            dev.uninstall()

    def test_frozen_c4_worker_keeps_the_stock_paths(self):
        bound, p, kv = self.boot(DS_FIXED), self.p, self.kv
        self.assertEqual((p.slab, p.resident, p.state, bound["ledger"], type(kv.c4_kv_pool).__name__),
                         ((2, 4), (6, 7), None, None, "HiSparseC4DevicePool"))
        self.admit(1, 1, 20)
        self.decode({1: [0, 2]})
        self.decode({1: [1, 3]})
        self.assertEqual(self.c.native, ["staging", *["eager_backup", ("swap_pages", 0), ("swap_pages", 1)] * 2])
        self.assertFalse(self.set_list(0))
        self.finish(1)


if __name__ == "__main__":
    unittest.main()
