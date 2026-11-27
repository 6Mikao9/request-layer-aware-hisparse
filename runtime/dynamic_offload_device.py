"""SGLang integration of runtime per-layer offload (issue #11): explicit, process-local, opt-in.

Importing this module imports neither torch nor SGLang. ``install`` runs in a fresh worker before the
KV pools and the HiSparse coordinator exist and patches only the pinned symbols in ``SYMBOLS``. The
stock and all-default configurations install nothing. One adapter interface serves MiniMax-M3 (sparse
MHA layers, K and V rows) and DeepSeek-V4 (C4 records, K-only page-padded rows); ``placement.offload_layers``
and its alternates name absolute layer IDs of that model's eligible layers.

* ``features.route_indirect`` off (frozen S, the direct control): the HiSparse device pool holds the
  planes of S only, a stock pool keeps the complete main KV of every other eligible layer at its
  logical rows, the stock swap serves S and the stock consumer reads resident layers directly.
* ``features.route_indirect`` on: the switchable layers U (union of the registered lists) live in one
  slab per K/V buffer (``dynamic_offload.SlabLayout``) with a device route table; eligible layers outside
  U are the direct resident pool above. Decode stores write the newest record to its tail row, the
  coordinator's eager backup becomes the tail flush, staging backs up through the routes, and one
  ``rlh_dynamic_offload_route`` launch per layer (``generated_route/hisparse_route.cuh``) serves either
  mode. A CUDA Graph captured once stays valid across switches: a switch rewrites the route table,
  native tables and rows, never a pointer.
* ``features.dynamic_offload``: ``POST /set_internal_state {"server_args": {"hisparse_offload_list": i}}``
  switches every TP rank to registered list i through ``dynamic_offload.SwitchProtocol``.
"""
import argparse
import ast
from contextlib import contextmanager, nullcontext
import functools
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

from . import dynamic_offload as do
from . import study_config as sc

PROVENANCE = Path(__file__).with_name("PROVENANCE.json")
ROUTE_HEADER = Path(__file__).with_name("generated_route") / "hisparse_route.cuh"
JIT_NAME = "rlh_dynamic_offload_route"
LIST_KEY = "hisparse_offload_list"
MODELS = ("minimax_m3", "deepseek_v4")
COORDINATOR = "srt/managers/hisparse_coordinator.py"
MEMORY_POOL = "srt/mem_cache/memory_pool.py"
DSV4_POOL = "srt/mem_cache/deepseek_v4_memory_pool.py"
DSV4_INDEXER = "srt/layers/attention/dsv4/indexer.py"
DSV4_COMPRESSOR = "srt/layers/attention/dsv4/compressor_v2.py"
MINIMAX_BACKEND = "srt/layers/attention/minimax_sparse_backend.py"
MINIMAX_OPS = "srt/layers/attention/minimax_sparse_ops/minimax_sparse.py"
INTERNAL_STATE = "srt/managers/scheduler_components/internal_state.py"
PREFILL = "minimax_sparse_prefill"
_COORD = ("__init__", "alloc_device_buffer", "admit_request_direct", "admit_request_into_staging",
          "request_finished", "_eager_backup_previous_token", "_grow_device_buffers")
# (pinned file, qualified symbol, required leading parameters or None); "*" entries are shared by both models.
SYMBOLS = {
    "*": (
        *((COORDINATOR, f"HiSparseCoordinator.{name}", None) for name in _COORD),
        (COORDINATOR, "HiSparseCoordinator.map_last_loc_to_buffer", None),
        (COORDINATOR, "HiSparseCoordinator.wait_for_pending_backup", ("self",)),
        (INTERNAL_STATE, "SchedulerInternalStateController._update_handlers", ("self",)),
        (INTERNAL_STATE, "_UpdateHandler.__init__", ("self", "validate", "apply")),
        (INTERNAL_STATE, "UpdateRejected", None),
        ("srt/mem_cache/pool_host/hisparse.py", "HiSparseHostPoolMixin.alloc_paged_token_slots",
         ("self", "req_to_host_pool", "req_to_host_pool_allocated_len", "req_pool_idx", "start_pos", "num_tokens")),
        ("srt/model_executor/pool_configurator.py", "DefaultPoolConfigurator._compute_cell_size",
         ("self", "kvc", "num_layers")),
        ("srt/mem_cache/kv_cache_configurator.py", "KVCacheConfigurator._build_token_to_kv_pool",
         ("self", "sizes", "is_dsa_model", "is_dsv4_model", "req_to_token_pool")),
    ),
    "minimax_m3": (
        (MEMORY_POOL, "MiniMaxSparseKVPool.__init__", ("self", "size", "page_size", "dtype", "head_num", "head_dim",
                                                       "idx_head_dim", "dense_layer_ids", "sparse_layer_ids",
                                                       "device")),
        *((MEMORY_POOL, f"MiniMaxSparseKVPool.{name}", ("self", "layer_id"))
          for name in ("get_key_buffer", "get_value_buffer", "get_kv_buffer")),
        (MEMORY_POOL, "MiniMaxSparseKVPool.set_kv_buffer", ("self", "layer", "loc", "cache_k", "cache_v", "k_scale",
                                                            "v_scale")),
        (MEMORY_POOL, "MiniMaxSparseKVPool.set_fused_kv_index_buffer", ("self", "layer", "loc", "cache_k", "cache_v")),
        (MEMORY_POOL, "MiniMaxSparseKVPool.get_kv_size_bytes", ("self",)),
        (MEMORY_POOL, "MHATokenToKVPool._create_buffers_normal", ("self",)),
        (MEMORY_POOL, "MHATokenToKVPool.set_kv_buffer", ("self", "layer", "loc_info", "cache_k", "cache_v", "k_scale",
                                                         "v_scale", "layer_id_override")),
        (MEMORY_POOL, "_set_kv_buffer_impl", ("k", "v", "k_cache", "v_cache", "indices", "row_dim", "store_dtype",
                                               "device_module", "size_limit", "alt_stream", "v_row_dim")),
        (MEMORY_POOL, "unwrap_write_loc", ("loc_info",)),
        ("srt/mem_cache/hisparse_memory_pool.py", "HiSparseMHAMainPool.__init__", None),
        ("srt/mem_cache/pool_host/mha.py", "HiSparseMHATokenToKVPoolHost.backup_from_device_all_layer",
         ("self", "device_pool", "host_indices", "device_indices", "io_backend")),
        (MINIMAX_BACKEND, "MiniMaxSparseAttnBackend._hisparse_swap_in_blocks",
         ("self", "forward_batch", "topk_idx", "layer_id")),
        (MINIMAX_BACKEND, "MiniMaxSparseAttnBackend.forward_decode", None),
        (MINIMAX_BACKEND, "MiniMaxSparseAttnBackend.forward_extend", None),
        (MINIMAX_OPS, PREFILL, ("q", "k_cache", "v_cache")),
        (COORDINATOR, "HiSparseCoordinator.swap_in_selected_blocks",
         ("self", "req_pool_indices", "seq_lens", "top_k_blocks", "layer_id", "sparse_block_size")),
    ),
    "deepseek_v4": (
        (DSV4_POOL, "DeepSeekV4TokenToKVPool._make_kv_pool", ("self", "size", "page_size", "dtype", "layer_num",
                                                              "device", "enable_memory_saver", "global_page_size",
                                                              "cls", "kv_layout")),
        (DSV4_POOL, "DeepSeekV4TokenToKVPool._init_compressed_layer_mapping", ("self",)),
        (DSV4_POOL, "DeepSeekV4TokenToKVPool.get_extra_key_buffer", ("self", "layer_id")),
        (DSV4_POOL, "DeepSeekV4SingleKVPool.create_buffer", ("self", "num_pages")),
        (DSV4_POOL, "HiSparseC4DevicePool.translate_loc_to_hisparse_device", ("self", "compressed_indices")),
        (DSV4_POOL, "HiSparseC4DevicePool._translate_loc_to_hisparse_device", ("self", "compressed_indices")),
        (DSV4_POOL, "HiSparseC4DevicePool.translate_loc_from_full_to_compressed", ("self", "full_indices")),
        (DSV4_POOL, "_num_dsv4_physical_kv_pages", ("size", "physical_page_size", "logical_page_size")),
        (DSV4_POOL, "DeepSeekV4LayerItem", None),
        ("srt/mem_cache/memory_pool_host.py", "DeepSeekV4PagedHostPool.__init__",
         ("self", "pool_name", "device_buffers", "item_bytes", "num_host_pages", "slot_page_size")),
        ("srt/model_executor/pool_configurator.py", "DSV4PoolConfigurator._get_bytes_per_full_token", ("self",)),
        ("srt/model_executor/pool_configurator.py", "DSV4PoolConfigurator._get_paged_kv_bytes_per_token",
         ("self", "compress_ratio")),
        ("srt/model_executor/pool_configurator.py", "DSV4PoolConfigurator.num_layers", ("self", "ratio")),
        ("srt/models/deepseek_v4.py", "MQALayer.forward", ("self", "x", "positions", "forward_batch")),
        (DSV4_INDEXER, "C4IndexerBackendMixin.forward_c4_indexer",
         ("self", "x", "q_lora", "c4_indexer", "forward_batch")),
        (DSV4_COMPRESSOR, "CompressorBackendMixin.forward_unified", None),
        (COORDINATOR, "HiSparseCoordinator.swap_in_selected_pages",
         ("self", "req_pool_indices", "compressed_seq_lens", "top_k_result", "layer_id")),
    ),
}


def _calls(node, attribute):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == attribute]


# (model, file, symbol, contract, predicate over the symbol's AST node): call sites the hooks rely on.
CONTRACTS = (
    ("minimax_m3", MINIMAX_BACKEND, "MiniMaxSparseAttnBackend.forward_extend",
     f"imports {PREFILL} at call time and passes loc_mapping by keyword",
     lambda n: any(isinstance(i, ast.ImportFrom) and any(a.name == PREFILL for a in i.names) for i in ast.walk(n))
     and all("loc_mapping" in {k.arg for k in c.keywords} for c in ast.walk(n)
             if isinstance(c, ast.Call) and getattr(c.func, "id", None) == PREFILL)),
    ("minimax_m3", MINIMAX_BACKEND, "MiniMaxSparseAttnBackend.forward_decode",
     "reads main KV through kv_pool.get_kv_buffer and swaps in through self._hisparse_swap_in_blocks",
     lambda n: bool(_calls(n, "get_kv_buffer")) and any(
         isinstance(a, ast.Attribute) and a.attr == "_hisparse_swap_in_blocks" for a in ast.walk(n))),
    ("minimax_m3", MEMORY_POOL, "MiniMaxSparseKVPool.set_fused_kv_index_buffer",
     "falls back to self.set_kv_buffer under HiSparse", lambda n: bool(_calls(n, "set_kv_buffer"))),
    ("deepseek_v4", DSV4_COMPRESSOR, "CompressorBackendMixin.forward_unified",
     "stores C4 records into get_extra_key_buffer through the layer_mapping pool's location translation",
     lambda n: bool(_calls(n, "get_extra_key_buffer")) and bool(_calls(n, "_translate_loc_to_hisparse_device"))),
    ("deepseek_v4", DSV4_INDEXER, "C4IndexerBackendMixin.forward_c4_indexer",
     "decode swaps in with layer_id=compress_layer_id; prefill translates through c4_kv_pool",
     lambda n: any({k.arg for k in c.keywords} >= {"layer_id", "compressed_seq_lens"}
                   for c in _calls(n, "swap_in_selected_pages"))
     and bool(_calls(n, "translate_loc_to_hisparse_device"))),
)
PENDING_HARDWARE = ("sm80_jit_build", "route_kernel_bit_equality", "cuda_graph_switch_without_recapture",
                    "tp8_switch_receipts", "physical_memory_receipt", "serving_tpot_throughput",
                    "nsys_all_rank_timelines", "ncu_route_kernel")
def pinned_digests():
    manifest = json.loads(PROVENANCE.read_text())
    return {**manifest["source_sha256"], **manifest["runtime_dependency_sha256"],
            **manifest["shared_lru_dependency_sha256"], **manifest["dynamic_offload_dependency_sha256"]}


def verify_digests(root):
    """Map every pinned SGLang input to ok, mismatch or missing."""
    status = {}
    for rel, expected in sorted(pinned_digests().items()):
        path = Path(root) / rel
        status[rel] = ("missing" if not path.is_file() else
                       "ok" if hashlib.sha256(path.read_bytes()).hexdigest() == expected else "mismatch")
    return status


def _find(tree, qualified):
    node = tree
    for part in qualified.split("."):
        matches = [n for n in getattr(node, "body", ())
                   if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == part]
        if not matches:
            return None
        node = matches[-1]
    return node


def resolve_symbols(root, model):
    """AST check of every patched or relied-upon symbol and call contract of one model; never imports SGLang."""
    trees, results = {}, []

    def node_of(rel, symbol):
        path = Path(root) / rel
        if rel not in trees:
            trees[rel] = ast.parse(path.read_bytes(), filename=str(path)) if path.is_file() else None
        return None if trees[rel] is None else _find(trees[rel], symbol)

    for rel, symbol, params in (*SYMBOLS["*"], *SYMBOLS[model]):
        node = node_of(rel, symbol)
        status = "missing" if node is None else "ok"
        if params is not None and isinstance(node, ast.FunctionDef):
            names = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs][:len(params)]
            status = "ok" if tuple(names) == params else f"signature {names}"
        results.append(dict(file=rel, symbol=symbol, status=status))
    for owner, rel, symbol, contract, holds in CONTRACTS:
        if owner == model:
            node = node_of(rel, symbol)
            results.append(dict(file=rel, symbol=f"{symbol}: {contract}",
                                status="ok" if node is not None and holds(node) else "contract changed"))
    return results


class MHARows:
    """MiniMax rows: K and V slabs [rows, heads, dim]; host planes are the HiSparse MHA host pool's layers."""
    buffers = 2

    def __init__(self, torch, k, v, host):
        self.torch, self.k, self.v, self.host = torch, k, v, host
        self.item = k.stride(0) * k.element_size()

    def d2d(self, src, dst):
        self.k[dst] = self.k[src]
        self.v[dst] = self.v[src]

    def _host(self, index, src, dst, to_host):
        hk, hv = self.host.k_buffer[index], self.host.v_buffer[index]
        pairs = ((self.k, hk), (self.v, hv)) if to_host else ((hk, self.k), (hv, self.v))
        if self.k.is_cuda:
            from sgl_kernel.kvcacheio import transfer_kv_per_layer
            (sk, dk), (sv, dv) = pairs
            transfer_kv_per_layer(src_k=sk, dst_k=dk, src_v=sv, dst_v=dv, src_indices=src, dst_indices=dst,
                                  item_size=self.item)
            return
        for source, target in pairs:
            target[dst] = source[src]

    def d2h(self, index, src, host_rows):
        self._host(index, src, host_rows, True)

    def h2d(self, index, host_rows, dst):
        self._host(index, host_rows, dst, False)

    def plan(self, indices, bases):
        """Host copies of the listed layers from rows ``base + src`` of their planes, issued per layer as the
        pinned HiSparse MHA backup does."""
        return list(indices), self.torch.tensor(bases, dtype=self.torch.int64, device=self.k.device)[:, None]

    def d2h_plan(self, plan, src, host_rows):
        indices, bases = plan
        if indices:
            rows = bases + src
            for j, index in enumerate(indices):
                self.d2h(index, rows[j], host_rows)


class PagedRows:
    """DeepSeek-V4 C4 rows: one K-only slab of 64-row pages [pages, 37440] uint8, value 576 B at slot * 576 and
    scale 8 B at 36864 + slot * 8 of each page; host planes are the C4 host pool's per-layer page buffers."""
    buffers, PAGE, VALUE, SCALE = 1, 64, 576, 8

    def __init__(self, torch, k, host):
        self.torch, self.k, self.host = torch, k, host
        self.slab_ptr = torch.tensor([k.data_ptr()], dtype=torch.uint64, device=k.device)

    def _parts(self, buffer):
        pages = buffer.view(-1, buffer.shape[-1])
        scale = self.PAGE * self.VALUE
        return (pages[:, :scale].unflatten(1, (self.PAGE, self.VALUE)),
                pages[:, scale:scale + self.PAGE * self.SCALE].unflatten(1, (self.PAGE, self.SCALE)))

    def _copy(self, source, src, target, dst, index=None, to_host=None):
        if self.k.is_cuda:
            from sglang.kernels.ops.kvcache.hisparse import transfer_cache_dsv4_mla
            host = None if index is None else self.host.data_ptrs[index:index + 1]
            ptrs = (self.slab_ptr, self.slab_ptr) if index is None else \
                (self.slab_ptr, host) if to_host else (host, self.slab_ptr)
            transfer_cache_dsv4_mla(*ptrs, src, dst)
            return
        for part_s, part_t in zip(self._parts(source), self._parts(target)):
            part_t[dst // self.PAGE, dst % self.PAGE] = part_s[src // self.PAGE, src % self.PAGE]

    def d2d(self, src, dst):
        self._copy(self.k, src, self.k, dst)

    def d2h(self, index, src, host_rows):
        self._copy(self.k, src, self.host.kv_buffer[index], host_rows, index, True)

    def h2d(self, index, host_rows, dst):
        self._copy(self.host.kv_buffer[index], host_rows, self.k, dst, index, False)

    def plan(self, indices, bases):
        """One copy for all listed layers, as the pinned C4 backup issues it: each layer's device pointer starts
        at its (page-aligned) plane row ``base`` and every layer shares the row and host indices."""
        t, page_bytes = self.torch, self.k.stride(0) * self.k.element_size()
        if any(base % self.PAGE for base in bases):
            raise ValueError("C4 plane rows must start on a page")
        src = t.tensor([self.k.data_ptr() + base // self.PAGE * page_bytes for base in bases], dtype=t.uint64,
                       device=self.k.device)
        dst = None
        if self.k.is_cuda:
            layers = t.tensor(indices, dtype=t.int64, device=self.host.data_ptrs.device)
            dst = self.host.data_ptrs.view(t.int64)[layers].view(t.uint64)
        return list(indices), list(bases), src, dst

    def d2h_plan(self, plan, src, host_rows):
        indices, bases, src_ptrs, dst_ptrs = plan
        if not indices:
            return
        if self.k.is_cuda:
            from sglang.kernels.ops.kvcache.hisparse import transfer_cache_dsv4_mla
            transfer_cache_dsv4_mla(src_ptrs, dst_ptrs, src.long(), host_rows.long())
            return
        for index, base in zip(indices, bases):
            self.d2h(index, src + base, host_rows)
class SlabState:
    """Routes, per-slot record counts and the switch executor of one rank's slab.

    ``home[slot]`` counts the records (tokens; C4 records for DeepSeek-V4) at their home rows: resident rows,
    the short-request hot slots of offloaded layers and host. A record produced by decode sits only in its
    tail row until the next flush. Device work runs on the coordinator's streams as the native backup does.
    """

    def __init__(self, layout, lists, rows, *, active, host_shadow, ratio, torch, device):
        self.layout, self.geometry, self.lists, self.rows = layout, layout.geometry, lists, rows
        self.host_shadow, self.ratio, self.torch, self.device = host_shadow, ratio, torch, device
        self.routes = do.initial_routes(layout, lists[active])
        self.route = torch.tensor(self.routes.descriptor(layout), dtype=torch.int32, device=device)
        self.home, self.coordinator, self.busy, self._plans = {}, None, lambda: None, None
        self.copy_plans()

    def layer_index(self, layer):
        return self.geometry.index(layer)

    def offloaded(self, layer):
        return self.routes.route(layer)[0] == do.OFFLOADED

    def plane(self, layer):
        """(first absolute row, rows) of the layer's current plane."""
        mode, _ = self.routes.route(layer)
        g = self.geometry
        return self.routes.base(self.layout, layer), g.hot_rows if mode == do.OFFLOADED else g.resident_rows

    def _long(self, values):
        return self.torch.as_tensor(values, dtype=self.torch.int64, device=self.device)

    def records(self, slot, count):
        """Logical rows of records 0..count-1 (a C4 record's row is its closing token's location / 4)."""
        tokens = self.coordinator.req_to_token_pool.req_to_token[slot, :count * self.ratio]
        return tokens[self.ratio - 1::self.ratio].long() // self.ratio

    def fix_locs(self, slot, start, stop):
        """Absolute slab rows for the native locations of buffer slots start..stop of ``slot``."""
        c = self.coordinator
        rows = c.req_to_device_buffer[slot, start:stop].to(self.torch.int32)
        offload = (self.route[:, 0] == do.OFFLOADED)[:, None]
        c.req_device_buffer_token_locs[:, slot, start:stop] = self.torch.where(
            offload, rows[None] + self.route[:, 1:2], rows[None])

    def restore_locs(self, slot):
        """Plane-relative native locations again, so the pinned request_finished frees allocator rows."""
        c = self.coordinator
        cap = int(c.req_device_buffer_size[slot])
        c.req_device_buffer_token_locs[:, slot, :cap] = c.req_to_device_buffer[slot, :cap].to(self.torch.int32)

    def _stream(self, stream):
        return self.torch.cuda.stream(stream) if self.device.type == "cuda" else nullcontext()

    def copy_plans(self):
        """Host-copy plans of the current routes, built at bind and publication (off the decode path): the layers
        whose decode records go to host, with their tail views, and the prefill staging of each mode."""
        key = (self.routes, self.host_shadow)
        if self._plans is None or self._plans[0] != key:
            g, rows = self.geometry, self.rows
            offloaded = [i for i, layer in enumerate(g.layers) if self.offloaded(layer)]
            kept = [i for i in range(len(g.layers)) if i not in offloaded] if self.host_shadow else []
            base = [self.plane(layer)[0] for layer in g.layers]
            self._plans = key, types.SimpleNamespace(
                flush=rows.plan(offloaded + kept, [self.layout.tail_view(g.layers[i]) for i in offloaded + kept]),
                stage_offloaded=rows.plan(offloaded, [base[i] for i in offloaded]),
                stage_resident=rows.plan(kept, [base[i] for i in kept]))
        return self._plans[1]

    def flush(self, slots, records, host_rows):
        """Copy each slot's record ``records[i]`` from its tail row to its home rows and host, on the decode
        backup stream like the pinned eager backup; the next forward waits for it. Fixed work per step and no host
        synchronization: one pinned index upload, one tail-to-home copy for all layers, and the host copies of the
        pinned backup (per layer for MiniMax, one launch for C4)."""
        t, c, g, cuda = self.torch, self.coordinator, self.geometry, self.device.type == "cuda"
        plans = self.copy_plans()
        index = t.tensor([slots, records], dtype=t.int64, pin_memory=cuda).to(self.device, non_blocking=True)
        slots_t, records_t, host_t = index[0], index[1], self._long(host_rows)
        req_to_token = c.req_to_token_pool.req_to_token
        logical = req_to_token[slots_t, (records_t + 1) * self.ratio - 1].long() // self.ratio
        buffer = c.req_to_device_buffer[slots_t, records_t.clamp(max=g.hot_slots)].long()
        if cuda:
            c.wait_for_pending_backup()
            schedule = t.cuda.current_stream()
        with self._stream(c.decode_backup_stream):
            if cuda:
                c.decode_backup_stream.wait_stream(schedule)
                if c.decode_producer_stream is not None:
                    c.decode_backup_stream.wait_stream(c.decode_producer_stream)
            route = self.route.long()
            tails = route[:, 2:3] + slots_t                     # [|U|, n]
            # Home rows: a resident layer's logical row, an offloaded record's hot slot; an offloaded record past
            # the hot slots has none, so its tail row is copied onto itself.
            hot = t.where(records_t < g.hot_slots, buffer + route[:, 1:2], tails)
            home = t.where(route[:, 0:1] == do.OFFLOADED, hot, logical + route[:, 1:2])
            self.rows.d2d(tails.flatten(), home.flatten())
            self.rows.d2h_plan(plans.flush, slots_t, host_t)
            if cuda:
                c._backup_done_event.record()
                for tensor in (index, host_t, logical, buffer, route, tails, hot, home):
                    tensor.record_stream(c.decode_backup_stream)
        if cuda:
            c._has_pending_backup = True
        for slot, record in zip(slots, records):
            self.home[slot] = record + 1

    def stage(self, logical, device_rows, host_rows):
        """Prefill backup through the routes: offloaded layers from their hot rows, resident ones (host_shadow)
        from their logical rows."""
        plans = self.copy_plans()
        self.rows.d2h_plan(plans.stage_offloaded, device_rows, host_rows)
        self.rows.d2h_plan(plans.stage_resident, logical, host_rows)

    # SwitchProtocol executor
    def quiesce(self):
        reason = self.busy()
        if reason:
            raise do.SwitchError(reason)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
        c = self.coordinator
        state = [(slot, self.home[slot], int(c.req_device_buffer_size[slot])) for slot in sorted(self.home)]
        return hashlib.sha256(json.dumps(state).encode()).hexdigest()

    def _requests(self):
        c = self.coordinator
        for slot in sorted(self.home):
            home, cap = self.home[slot], int(c.req_device_buffer_size[slot])
            yield slot, home, self.records(slot, home), c.req_to_host_pool[slot, :home].long(), \
                c.req_to_device_buffer[slot, :cap].long()

    def prepare(self, plan):
        """Phase 1: the host history of the layers that become offloaded (skipped with host_shadow)."""
        if self.host_shadow:
            return 0
        moved = 0
        for layer, old_plane, _ in plan.entering:
            base = self.layout.resident_base(old_plane)
            parts = [(logical + base, host) for _, _, logical, host, _ in self._requests()]
            if parts:
                src, host = (self.torch.cat(p) for p in zip(*parts))
                self.rows.d2h(self.layer_index(layer), src, host)
                moved += src.numel()
        self._sync()
        return moved

    def apply(self, plan):
        """Phase 2 from host only, native tables of the layers that become offloaded, then the route table."""
        t, c, g = self.torch, self.coordinator, self.geometry
        requests, moved = list(self._requests()), 0
        for layer, _, new_plane in plan.entering:
            index, base = self.layer_index(layer), self.layout.hot_base(new_plane)
            hot = [(host[:min(home, g.hot_slots, len(buffer))], buffer[:min(home, g.hot_slots, len(buffer))] + base)
                   for _, home, _, host, buffer in requests]
            moved += self._h2d(index, hot)
            for slot, _, _, _, buffer in requests:
                width = min(len(buffer), g.hot_slots + g.page)
                c.req_device_buffer_tokens[index, slot].fill_(-1)
                c.req_device_buffer_tokens[index, slot, :g.hot_slots] = t.arange(g.hot_slots, device=self.device)
                c.req_device_buffer_token_locs[index, slot].fill_(-1)
                c.req_device_buffer_token_locs[index, slot, :width] = (buffer[:width] + base).to(t.int32)
                c.lru_slots[index, slot] = t.arange(g.hot_slots, dtype=c.lru_slots.dtype, device=self.device)
        for layer, _, new_plane in plan.leaving:
            base = self.layout.resident_base(new_plane)
            moved += self._h2d(self.layer_index(layer), [(host, logical + base)
                                                        for _, _, logical, host, _ in requests])
        self.route.copy_(t.tensor(plan.target.descriptor(self.layout), dtype=t.int32))
        self._sync()
        return moved

    def _h2d(self, index, parts):
        parts = [p for p in parts if p[0].numel()]
        if not parts:
            return 0
        host, dst = (self.torch.cat(p) for p in zip(*parts))
        self.rows.h2d(index, host, dst)
        return dst.numel()

    def _sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()

    def publish(self, routes):
        self.routes = routes
        self.copy_plans()

    def signature(self):
        return dict(layout=self.layout.digest, lists={k: list(v) for k, v in self.lists.items()},
                    host_shadow=self.host_shadow, ratio=self.ratio, route_kernel_sha256=route_sha256())
def route_sha256():
    return hashlib.sha256(ROUTE_HEADER.read_bytes()).hexdigest()


def route_template(block_size, num_top_k, hot_slots, mla, top_k_block_size, ratio):
    """Template arguments of ``load_cache_to_device_buffer`` in the route header (no miss plan, real IO)."""
    return [block_size, num_top_k, hot_slots, mla, mla, top_k_block_size, not mla, False, False, ratio]


@functools.cache
def _route_module(item_size_bytes, *template):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args
    args = make_cpp_args(*template)
    return load_jit(JIT_NAME, *make_cpp_args(item_size_bytes, *template), cuda_files=[str(ROUTE_HEADER)],
                    cuda_wrappers=[("load_cache_to_device_buffer", f"load_cache_to_device_buffer<{args}>")])


def route_swap(state, c, req_pool_indices, seq_lens, top_k, layer_id, top_k_block_size=1):
    """One launch for one switchable layer in either mode; graph-safe (views, constants, no sync)."""
    torch, rows = state.torch, state.rows
    index, mla = state.layer_index(layer_id), rows.buffers == 1
    width = top_k.size(1) * top_k_block_size
    if top_k.dtype != torch.int32 or width > c.top_k_device_locs_buffer.shape[1] or width > c.device_buffer_size:
        raise ValueError("Selection width or dtype differs from the bound route kernel contract")
    out = c.top_k_device_locs_buffer[:req_pool_indices.size(0), :width]
    empty = torch.empty(0, device=top_k.device)
    host = rows.host.kv_buffer[index] if mla else rows.host.k_buffer[index]
    module = _route_module(c.item_size_bytes, *route_template(c.swap_in_block_size, top_k.size(1),
                                                              c.device_buffer_size, mla, top_k_block_size,
                                                              state.ratio))
    module.load_cache_to_device_buffer(
        top_k, c.req_device_buffer_tokens[index], c.req_to_host_pool, c.req_device_buffer_token_locs[index], host,
        empty if mla else rows.host.v_buffer[index], rows.k, empty if mla else rows.v, out, req_pool_indices,
        seq_lens, c.lru_slots[index], c.num_real_reqs, 1, c.item_size_bytes, empty, empty, empty, state.route,
        index, c.req_to_token_pool.req_to_token)
    return out


def emulate_route(state, c, req_pool_indices, seq_lens, selections, layer_id, out):
    """CPU statement of ``route_swap`` (tests and the A100 gate's expected values). ``selections[b]`` lists the
    selected positions of graph row b, None for an invalid entry; misses are copied host -> slab."""
    from .selection_oracle import reference_lru
    index, g = state.layer_index(layer_id), state.geometry
    mode, base, tail, _ = state.route[index].tolist()
    blocks = state.rows.buffers == 2         # MiniMax selects blocks; DeepSeek-V4 selects record positions
    out.fill_(-1)
    for b in range(int(c.num_real_reqs[0])):
        rid, seq = int(req_pool_indices[b]), int(seq_lens[b])
        short = mode == do.RESIDENT or seq <= g.hot_slots
        count = len(selections[b]) if blocks or not short else min(seq, len(selections[b]))
        valid = [(i, p) for i, p in enumerate(selections[b][:count]) if p is not None and 0 <= p < seq]
        older = [p for _, p in valid if p != seq - 1]
        locs = c.req_device_buffer_token_locs[index, rid].tolist()
        if mode == do.RESIDENT:
            tokens = c.req_to_token_pool.req_to_token[rid].tolist()
            rows = {p: base + tokens[(p + 1) * state.ratio - 1] // state.ratio for p in older}
        elif seq <= g.hot_slots:
            rows = {p: locs[p] for p in older}
        else:
            tags = c.req_device_buffer_tokens[index, rid, :g.hot_slots].tolist()
            order = c.lru_slots[index, rid].tolist()
            present = set(tags)
            tags, order, picked, victims = reference_lru(tags, order, older)
            misses = [p for p in older if p not in present]
            if misses:
                host = c.req_to_host_pool[rid].tolist()
                state.rows.h2d(index, state._long([host[p] for p in misses]),
                               state._long([locs[v] for v in victims]))
            c.req_device_buffer_tokens[index, rid, :g.hot_slots] = state._long(tags)
            c.lru_slots[index, rid] = state._long(order)
            rows = {p: locs[s] for p, s in zip(older, picked)}
        rows[seq - 1] = tail + rid
        for i, p in valid:
            out[b, i] = rows[p]
    return out
def split_layers(config, eligible):
    """(registered lists by name, slab layers, direct resident layers) of this stage's eligible layers.

    The slab layers are U with ``route_indirect`` and S otherwise; every other eligible layer keeps its
    complete main KV in a stock pool and is read directly."""
    lists = {str(i): tuple(layers) for i, layers in enumerate(config.offload_lists)}
    named = set().union(*lists.values())
    if not named <= set(eligible) or not lists["0"]:
        raise RuntimeError(f"Offload lists name layers {sorted(named - set(eligible))} that are not eligible "
                           f"host-backed layers of this stage {list(eligible)}")
    slab = tuple(sorted(named)) if config.feature("route_indirect") else lists["0"]
    return lists, slab, tuple(layer for layer in eligible if layer not in slab)


def resident_extra_cell_bytes(eligible, k, *, ratio, row_bytes):
    """Extra bytes per device token: each resident eligible layer holds ``ratio`` logical rows, not one hot row."""
    if type(ratio) is not int or ratio < 1:
        raise RuntimeError("host_to_device_ratio must be a positive integer for logical-row sizing")
    return (len(eligible) - k) * (ratio - 1) * row_bytes


def _same(a, b):
    return (a.data_ptr() == b.data_ptr() and tuple(a.shape) == tuple(b.shape) and a.stride() == b.stride()
            and a.dtype == b.dtype)


class Placement:
    """Layer routes of one model's KV pool: dense layers stock, slab layers (S, or U with route_indirect),
    direct resident layers in a stock pool. ``state`` is the SlabState with route_indirect, else None."""

    def __init__(self, config, lists, slab, resident, *, main, resident_pool, layout=None, store=None):
        self.config, self.lists, self.slab, self.resident = config, lists, slab, resident
        self.main, self.resident_pool, self.layout, self.store_impl = main, resident_pool, layout, store
        self.hot_index = {layer: i for i, layer in enumerate(slab)}
        self.resident_index = {layer: i for i, layer in enumerate(resident)}
        self.state, self.decode, self.layer = None, None, None
        self.protocol, self.binding, self.facades = None, None, {}

    @contextmanager
    def decoding(self, forward_batch):
        """Stores and getters of a decode forward address the tails and the whole slab."""
        self.decode = forward_batch
        try:
            yield
        finally:
            self.decode = None

    @contextmanager
    def in_layer(self, layer_id, forward_batch):
        """DeepSeek-V4: the absolute layer whose indexer, compressor and attention run now."""
        self.layer = layer_id
        try:
            with self.decoding(forward_batch if forward_batch.forward_mode.is_decode() else None):
                yield
        finally:
            self.layer = None

    def current_layer(self):
        if self.layer is None:
            raise RuntimeError("C4 swap-in outside an MQALayer forward; the layer offload hooks are incomplete")
        return self.layer

    def views(self, layer_id, kind):
        """``kind`` 'k', 'v' or 'kv' of a slab layer: the whole slab in decode, else the layer's current plane."""
        k, v = self.state.rows.k, getattr(self.state.rows, "v", None)
        if self.decode is None:
            base, rows = self.state.plane(layer_id)
            k, v = k[base:base + rows], None if v is None else v[base:base + rows]
        return dict(k=k, v=v, kv=(k, v))[kind]

    def store_rows(self, layer_id, raw_loc):
        """Absolute slab rows of a store: the request's tail row in decode, else the layer's home rows."""
        if self.decode is not None:
            return self.decode.req_pool_indices + self.layout.tail_view(layer_id)
        base, _ = self.state.plane(layer_id)
        rows = self.main.full_to_hisparse_device_index_mapping[raw_loc] if self.state.offloaded(layer_id) else raw_loc
        return rows + base

    def resident_plane(self, cache):
        """True for a resident plane, False for a hot plane, None for anything else (prefill routing)."""
        if self.resident_pool is not None and any(_same(cache, t) for t in self.resident_pool.k_buffer):
            return True
        if self.state is None:
            return False if any(_same(cache, t) for t in self.main.k_buffer) else None
        whole = self.state.rows.k
        offset = cache.data_ptr() - whole.data_ptr()
        row_bytes = whole.stride(0) * whole.element_size()
        if offset < 0 or offset % row_bytes or cache.stride() != whole.stride() or offset // row_bytes >= len(whole):
            return None
        return offset // row_bytes >= self.layout.k * self.layout.geometry.hot_rows

    def receipt_bytes(self, nbytes):
        def pool(p):          # MHA pools hold k_buffer and v_buffer, C4 pools kv_buffer
            names = ("k_buffer", "v_buffer", "kv_buffer")
            return 0 if p is None else sum(nbytes(getattr(p, name, [])) for name in names)

        out = dict(slab_hot_or_s=pool(self.main), direct_resident=pool(self.resident_pool))
        if self.state is not None:
            rows = self.state.rows
            out["slab_hot_or_s"] = nbytes(rows.k) + (nbytes(rows.v) if rows.buffers == 2 else 0)
            out["route_table"] = nbytes(self.state.route)
        return out
_STATE = None
_POOL_HOOKS = ("__init__", "get_key_buffer", "get_value_buffer", "get_kv_buffer", "set_kv_buffer", "get_kv_size_bytes")


def _state():
    if _STATE is None:
        raise RuntimeError("runtime.dynamic_offload_device was not installed in this worker")
    return _STATE


def _placement(kv):
    placement = getattr(kv, "_rlh_placement", None)
    if placement is None:
        raise RuntimeError("The KV pool was built before install(); start a fresh worker")
    return placement


def _patch(state, owner, name, hook):
    """Replace one pinned symbol after checking nobody else did; records it for uninstall."""
    current = owner.__dict__.get(name) if isinstance(owner, type) else getattr(owner, name)
    module = owner.__module__ if isinstance(owner, type) else owner.__name__
    if current is None or getattr(current, "__module__", None) != module or hasattr(current, "__wrapped__"):
        raise RuntimeError(f"Another hook already replaced {module}.{name}; refusing to stack hooks")
    state["patched"].append((owner, name, current, hook))
    setattr(owner, name, functools.update_wrapper(hook, current) if callable(hook) and callable(current) else hook)
    return current


def _minimax_pool(state, torch, memory, main_module):
    """Pool hooks of MiniMax-M3: S or U planes replace the all-layer HiSparse main pool."""
    import inspect
    from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
    pool_cls, mha_cls, real_main, config = memory.MiniMaxSparseKVPool, memory.MHATokenToKVPool, \
        main_module.HiSparseMHAMainPool, state["config"]
    indirect, originals = config.feature("route_indirect"), {}

    class SlabMainPool(real_main):
        """HiSparse main pool over U whose per-layer K/V are the tail views of one slab (route_indirect)."""

        def _create_buffers_normal(self):
            layout = state["pending_layout"]
            if self.kv_cache_layout != "nhd" or getattr(self, "use_hnd", False):
                raise RuntimeError("The dynamic offload slab requires plain NHD main-KV rows")
            (_, *k_row), (_, *v_row) = self._kv_buffer_shapes()
            with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE), \
                    (torch.cuda.use_mem_pool(self.custom_mem_pool) if self.enable_custom_mem_pool else nullcontext()):
                k = torch.zeros((layout.rows, *k_row), dtype=self.store_dtype, device=self.device)
                v = torch.zeros((layout.rows, *v_row), dtype=self.store_dtype, device=self.device)
            views = [(layout.tail_view(layer), layout.tail_view(layer) + layout.tail_stride)
                     for layer in layout.geometry.layers]
            self.k_buffer, self.v_buffer = [k[a:b] for a, b in views], [v[a:b] for a, b in views]
            self._rlh_slab = (k, v)

        def get_kv_size_bytes(self):
            """The whole slab, so the allocation log and mem_usage count every plane and tail."""
            return tuple(t.numel() * t.element_size() for t in self._rlh_slab)

    def pool_init(self, *args, **kwargs):
        a = signature.bind(self, *args, **kwargs)
        a.apply_defaults()
        a = a.arguments
        if type(self) is not pool_cls or a["main_pool_cls"] is not mha_cls or not a["enable_hisparse"]:
            raise RuntimeError("Layer offload supports the stock MiniMaxSparseKVPool with --enable-hisparse")
        eligible = tuple(layer for layer in a["sparse_layer_ids"] if a["start_layer"] <= layer < a["end_layer"])
        lists, slab, resident = split_layers(config, eligible)
        ratio, page, size, layout = a["host_to_device_ratio"], a["page_size"], a["size"], None
        if indirect:
            from sglang.srt.mem_cache.sparsity import parse_hisparse_config
            row = a["head_num"] * a["head_dim"] * torch._utils._element_size(a["dtype"])
            geometry = do.Geometry(layers=slab, hot_rows=size + page, resident_rows=size * ratio + page,
                                   hot_slots=parse_hisparse_config().device_buffer_size, slots=state["slots"],
                                   page=page, page_bytes=page * row)
            layout = state["pending_layout"] = do.SlabLayout(geometry, len(lists["0"]))
        calls = []

        def main_pool(*positional, **pool_kwargs):
            if positional or calls or pool_kwargs.get("layer_num") != len(eligible):
                raise RuntimeError("The pinned constructor built its HiSparse main pool differently than expected")
            calls.append(pool_kwargs)
            cls = SlabMainPool if indirect else real_main
            return cls(**dict(pool_kwargs, layer_num=len(slab), start_layer=0, end_layer=len(slab)))

        main_module.HiSparseMHAMainPool = main_pool
        try:
            originals["__init__"](self, *args, **kwargs)
        finally:
            main_module.HiSparseMHAMainPool = real_main
            state.pop("pending_layout", None)
        if len(calls) != 1:
            raise RuntimeError("The HiSparse main pool constructor was not intercepted")
        self.main_pool.model_layer_ids = slab
        resident_pool = None
        if resident:
            resident_pool = mha_cls(size=size * ratio, page_size=page, dtype=a["dtype"], head_num=a["head_num"],
                                    head_dim=a["head_dim"], layer_num=len(resident), device=a["device"],
                                    enable_memory_saver=a["enable_memory_saver"], start_layer=0,
                                    end_layer=len(resident))
            resident_pool.model_layer_ids = resident
            self.mem_usage += resident_pool.mem_usage
        self._rlh_placement = Placement(config, lists, slab, resident, main=self.main_pool,
                                        resident_pool=resident_pool, layout=layout)
        state["pools"].append(self)

    def getter(name, kind):
        def hook(self, layer_id):
            p = _placement(self)
            if layer_id in p.resident_index:
                self._wait_for_layer(layer_id)
                return getattr(p.resident_pool, name)(p.resident_index[layer_id])
            if layer_id in p.hot_index:
                self._wait_for_layer(layer_id)
                if p.layout is None:
                    return getattr(self.main_pool, name)(p.hot_index[layer_id])
                if p.state is None:
                    raise RuntimeError("Slab layers are read only after the coordinator binds")
                return p.views(layer_id, kind)
            return originals[name](self, layer_id)
        return hook

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None):
        p, layer_id = _placement(self), layer.layer_id
        if layer_id in p.resident_index:
            return p.resident_pool.set_kv_buffer(layer, loc, cache_k, cache_v, k_scale, v_scale,
                                                 layer_id_override=p.resident_index[layer_id])
        if layer_id not in p.hot_index:
            return originals["set_kv_buffer"](self, layer, loc, cache_k, cache_v, k_scale, v_scale)
        if p.layout is None:
            return self.main_pool.set_kv_buffer(layer, loc, cache_k, cache_v, k_scale, v_scale,
                                                layer_id_override=p.hot_index[layer_id])
        main, rows = p.main, p.state.rows
        if {k_scale, v_scale} - {None, 1.0} or cache_k.dtype != main.dtype or main.store_dtype != main.dtype:
            raise RuntimeError("The slab stores unscaled unquantized K/V in the pool dtype only")
        raw, _, _ = memory.unwrap_write_loc(loc)
        memory._set_kv_buffer_impl(cache_k, cache_v, rows.k, rows.v, p.store_rows(layer_id, raw), row_dim=main.row_dim,
                                   store_dtype=main.store_dtype, device_module=main.device_module,
                                   size_limit=rows.k.shape[0], alt_stream=main.alt_stream, v_row_dim=main.v_row_dim)

    def get_kv_size_bytes(self):
        k, v = originals["get_kv_size_bytes"](self)
        p = _placement(self)
        if p.resident_pool is not None:
            rk, rv = p.resident_pool.get_kv_size_bytes()
            k, v = k + rk, v + rv
        return k, v

    hooks = dict(__init__=pool_init, get_key_buffer=getter("get_key_buffer", "k"),
                 get_value_buffer=getter("get_value_buffer", "v"), get_kv_buffer=getter("get_kv_buffer", "kv"),
                 set_kv_buffer=set_kv_buffer, get_kv_size_bytes=get_kv_size_bytes)
    signature = inspect.signature(pool_cls.__init__)
    for name in _POOL_HOOKS:
        originals[name] = _patch(state, pool_cls, name, hooks[name])


def _minimax_backend(state):
    """Backend hooks of MiniMax-M3, installed when the coordinator binds (before the backend exists)."""
    import inspect
    backend_cls = importlib.import_module("sglang.srt.layers.attention.minimax_sparse_backend").MiniMaxSparseAttnBackend
    ops = importlib.import_module("sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse")
    decode = inspect.signature(backend_cls.forward_decode)

    def swap_blocks(self, forward_batch, topk_idx, layer_id):
        if layer_id in _placement(self.kv_pool).resident_index:
            return None             # direct: the stock consumer reads req_to_token rows of the resident plane
        return original_swap(self, forward_batch, topk_idx, layer_id)

    def forward_decode(self, *args, **kwargs):
        p = _placement(self.kv_pool)
        if p.layout is None:
            return original_decode(self, *args, **kwargs)
        with p.decoding(decode.bind(self, *args, **kwargs).arguments["forward_batch"]):
            return original_decode(self, *args, **kwargs)

    def prefill(*args, **kwargs):
        if len(args) < 3 or "loc_mapping" not in kwargs:
            raise RuntimeError(f"{PREFILL} expects the pinned positional K/V caches and loc_mapping keyword")
        kinds = {_placement(pool).resident_plane(args[1]) for pool in state["pools"]} - {None}
        if len(kinds) != 1:
            raise RuntimeError("Sparse prefill main KV is neither a resident nor an offloaded plane")
        if kinds == {True}:
            kwargs["loc_mapping"] = None    # resident rows live at their logical locations
        elif kwargs["loc_mapping"] is None:
            raise RuntimeError("Offloaded prefill must read through the HiSparse location mapping")
        return original_prefill(*args, **kwargs)

    original_swap = _patch(state, backend_cls, "_hisparse_swap_in_blocks", swap_blocks)
    original_decode = _patch(state, backend_cls, "forward_decode", forward_decode)
    original_prefill = _patch(state, ops, PREFILL, prefill)
def context_errors(c, context, *, indirect, torch):
    """Execution the slab path is validated for: single-stream decode, eager prefill, sm_80."""
    errors = []
    if c.skip_io or c.spec_swap.enabled or c.enable_prefetch:
        errors.append("skip-IO probes, speculative decoding and shared-index prefetch are unsupported")
    schedule, execution, memory = context.get_schedule(), context.get_exec(), context.get_memory()
    if (not schedule.disable_overlap_schedule or execution.overlap.enable_two_batch_overlap
            or execution.overlap.enable_single_batch_overlap or not memory.disable_radix_cache
            or context.get_spec().speculative_algorithm not in (None, "NONE")
            or context.get_disagg().disaggregation_mode != "null" or context.get_disagg().enable_pdmux
            or context.get_parallel().pp_size != 1):
        errors.append("requires single-stream decode: no overlap, Radix, speculation, disaggregation or PP")
    graph = execution.graph.cuda_graph_config
    if graph.decode.backend not in ("disabled", "full"):
        errors.append("decode must be eager or full CUDA Graph")
    if indirect and graph.prefill.backend != "disabled":
        errors.append("route_indirect routes prefill stores in Python; run prefill eagerly")
    if indirect and torch.cuda.get_device_capability() != (8, 0):
        errors.append("the route kernel is built and validated for sm_80 (A100) only")
    return errors


def check_minimax(c, kv, p, *, torch):
    errors, main, resident = [], kv.main_pool, p.resident_pool
    if not c.is_m3_hisparse or getattr(c, "is_dsv4_hisparse", False) or c.mem_pool_device is not main:
        errors.append("the MiniMax binding needs the M3 MHA HiSparse coordinator over the placement's main pool")
    for name, pool in (("slab", main), ("resident", resident)):
        if pool is not None and (pool.head_num != 1 or pool.is_quantized_kv_cache or pool.dtype != torch.bfloat16
                                 or pool.store_dtype != pool.dtype or pool.post_capture_active):
            errors.append(f"{name} main KV must be one BF16 head per rank, unquantized, without post-capture backing")
    if (tuple(main.model_layer_ids) != p.slab or main.layer_num != len(p.slab)
            or c.req_device_buffer_tokens.shape[0] != len(p.slab) or c.mem_pool_host.layer_num != len(p.slab)):
        errors.append("HiSparse planes, swap tables and host history must cover exactly the slab layers")
    if resident is not None and resident.size != c.token_to_kv_pool_allocator.size_full:
        errors.append("resident planes must span the logical allocator")
    return errors


def _bind(c, *, torch, dist, context):
    """Gate, slab state, TP agreement and switch protocol of one coordinator; runs before any request."""
    st = _state()
    if st["binding"] is not None:
        raise RuntimeError("One HiSparse coordinator per worker process")
    config, kv = st["config"], c.token_to_kv_pool_allocator.get_kvcache()
    p = _placement(kv if st["model"] == "minimax_m3" else kv.c4_kv_pool)
    indirect = p.layout is not None
    errors = context_errors(c, context, indirect=indirect, torch=torch)
    errors += check_minimax(c, kv, p, torch=torch) if st["model"] == "minimax_m3" else check_dsv4(c, kv, p, torch=torch)
    g = p.layout.geometry if indirect else None
    if indirect and (g.slots != c.req_device_buffer_tokens.shape[1] or g.hot_slots != c.device_buffer_size
                     or c.padded_buffer_size != g.hot_slots + g.page):
        errors.append("slab tails or hot slots differ from the coordinator's request slots and device buffer")
    if errors:
        raise RuntimeError("Layer offload incompatible: " + "; ".join(errors))
    if indirect:
        slab = c.mem_pool_device._rlh_slab
        rows = MHARows(torch, *slab, c.mem_pool_host) if st["model"] == "minimax_m3" else \
            PagedRows(torch, slab[0], c.mem_pool_host)
        p.state = SlabState(p.layout, p.lists, rows, active="0", host_shadow=config.feature("host_shadow"),
                            ratio=c.compress_ratio, torch=torch, device=rows.k.device)
        p.state.coordinator = c
    signature = dict(model=st["model"], slab=list(p.slab), resident=list(p.resident),
                     state=None if p.state is None else p.state.signature())
    identity = hashlib.sha256(json.dumps(signature, sort_keys=True).encode() + config.digest.encode()).hexdigest()
    hashes = [None] * c.tp_world_size
    dist.all_gather_object(hashes, identity, group=c.tp_group)
    if hashes != [identity] * c.tp_world_size:
        raise RuntimeError("TP ranks disagree on the offload lists or the slab geometry")
    if config.feature("dynamic_offload"):
        from .layer_eviction_tp import GlooDemotionTransport
        group = dist.new_group(backend="gloo")
        p.protocol = do.SwitchProtocol(p.layout, p.lists, p.state, GlooDemotionTransport(group), active="0",
                                       config_digest=config.digest)
    p.binding, st["binding"] = c, p
    (_minimax_backend if st["model"] == "minimax_m3" else _dsv4_backend)(st)
    emit(receipt(c, p, signature, torch=torch), dist.get_rank())


def emit(record, rank):
    print(json.dumps(dict(record, rank=rank), sort_keys=True), flush=True)


def receipt(c, p, signature, *, torch):
    def nbytes(value):
        if isinstance(value, (list, tuple)):
            return sum(nbytes(item) for item in value)
        return value.numel() * value.element_size()

    st, memory = _state(), None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        memory = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                      max_allocated=torch.cuda.max_memory_allocated(), device_used=total - free, device_total=total)
    host = c.mem_pool_host
    host_bytes = nbytes([*getattr(host, "k_buffer", []), *getattr(host, "v_buffer", []),
                         *getattr(host, "kv_buffer", [])])
    tables = dict(tokens=nbytes(c.req_device_buffer_tokens), locs=nbytes(c.req_device_buffer_token_locs),
                  lru=nbytes(c.lru_slots))
    allocator = c.token_to_kv_pool_allocator
    return dict(event="hisparse_dynamic_offload_bound", tp_size=c.tp_world_size, config_file_sha256=st["digest"],
                config_sha256=st["config"].digest, resolved_config=st["config"].as_dict(),
                capabilities=sc.capabilities(st["config"]), signature=signature, cell_size=st.get("charge"),
                logical_kv=dict(tokens=allocator.size_full, hot_rows=getattr(allocator, "_size_hisparse", None)),
                ledger=None if p.layout is None else do.ledger(p.layout),
                bytes=dict(p.receipt_bytes(nbytes), host_pinned=host_bytes, native_tables=tables),
                device_memory_at_bind=memory,
                unavailable=dict(weights="not attributed at bind", graph_workspace="after capture",
                                 index_kv="identical in every arm; see the pool allocation log"),
                pending_hardware=list(PENDING_HARDWARE))
def _slab(c):
    """The bound SlabState (route_indirect) of a coordinator, or None for the direct control."""
    p = _state()["binding"]
    return None if p is None or p.binding is not c else p.state


def _coordinator(state, torch, dist, context):
    module = importlib.import_module("sglang.srt.managers.hisparse_coordinator")
    cls, device_module = module.HiSparseCoordinator, module.device_module
    originals = {}

    def init(self, *args, **kwargs):
        originals["__init__"](self, *args, **kwargs)
        _bind(self, torch=torch, dist=dist, context=context)

    def admit_request_direct(self, req):
        raise RuntimeError("Direct (disaggregated) admission is not supported with layer offload")

    def allocated_records(self, req):
        rid = req.kv.req_pool_idx
        length = req.extend_end if self.is_dsv4_hisparse else req.kv.kv_allocated_len
        full = self.req_to_token_pool.req_to_token[rid, :length].to(torch.int64)
        return self.mem_pool_device.translate_loc_from_full_to_compressed(full)

    def alloc_device_buffer(self, req):
        s, rid = _slab(self), req.kv.req_pool_idx
        records = None if s is None else allocated_records(self, req)
        if s is not None and self.is_dsv4_hisparse and len(records):
            # The newest C4 record of the prompt goes to the tails before the pinned call frees prefill rows.
            hot = self.mem_pool_device.full_to_hisparse_device_index_mapping[records[-1:]]
            for layer in s.geometry.layers:
                base, _ = s.plane(layer)
                s.rows.d2d((hot if s.offloaded(layer) else records[-1:]) + base,
                           s._long([s.layout.tail_row(layer, rid)]))
        originals["alloc_device_buffer"](self, req)
        if s is not None:
            s.fix_locs(rid, 0, int(self.req_device_buffer_size[rid]))
            s.home[rid] = len(records)

    def grow(self, seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu):
        s = _slab(self)
        before = None if s is None else self.req_device_buffer_size[req_pool_indices_cpu].clone()
        out = originals["_grow_device_buffers"](self, seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu)
        if s is not None:
            after = self.req_device_buffer_size[req_pool_indices_cpu]
            for slot, old, new in zip(req_pool_indices_cpu.tolist(), before.tolist(), after.tolist()):
                if new != old:
                    s.fix_locs(slot, old, new)
        return out

    def eager_backup(self, seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu):
        s = _slab(self)
        if s is None:
            return originals["_eager_backup_previous_token"](self, seq_lens, req_pool_indices, seq_lens_cpu,
                                                             req_pool_indices_cpu)
        slots, records = [], []
        for seq, rid in zip(seq_lens_cpu.tolist(), req_pool_indices_cpu.tolist()):
            if self._skip_first_backup[rid]:
                self._skip_first_backup[rid] = False
            elif (seq - 1) % self.compress_ratio == 0:      # the previous step produced record (seq - 1) // r - 1
                slots.append(rid)
                records.append((seq - 1) // self.compress_ratio - 1)
        if slots:
            host = torch.cat([self.mem_pool_host.alloc_paged_token_slots(
                self.req_to_host_pool, self.req_to_host_pool_allocated_len, rid, record, 1)
                for rid, record in zip(slots, records)])
            s.flush(slots, records, host)

    def admit_request_into_staging(self, req):
        s = _slab(self)
        if s is None:
            return originals["admit_request_into_staging"](self, req)
        req.hisparse_staging = True
        rid = req.kv.req_pool_idx
        full = self.req_to_token_pool.req_to_token[rid, :req.extend_end].to(dtype=torch.int64, copy=True)
        records = self.mem_pool_device.translate_loc_from_full_to_compressed(full)
        device_rows = self.mem_pool_device.full_to_hisparse_device_index_mapping[records].long()
        host = self.mem_pool_host.alloc_paged_token_slots(self.req_to_host_pool, self.req_to_host_pool_allocated_len,
                                                          rid, 0, len(device_rows))
        start, finish = device_module.Event(), device_module.Event()
        start.record()
        with device_module.stream(self.write_staging_stream):
            start.wait(self.write_staging_stream)
            s.stage(records, device_rows, host)
            finish.record()
            for tensor in (host, device_rows, records):
                if tensor.is_cuda:
                    tensor.record_stream(self.write_staging_stream)
        self.ack_staging_queue.append(module.HiSparseAct(start, finish, req))

    def request_finished(self, req):
        s, rid = _slab(self), req.kv.req_pool_idx
        if s is not None and int(self.req_device_buffer_size[rid]) > 0:
            s.restore_locs(rid)
        originals["request_finished"](self, req)
        if s is not None:
            s.home.pop(rid, None)

    def swap_blocks(self, req_pool_indices, seq_lens, top_k_blocks, layer_id, sparse_block_size):
        p, s = _state()["binding"], _slab(self)
        if s is not None:
            return route_swap(s, self, req_pool_indices, seq_lens, top_k_blocks, layer_id, sparse_block_size)
        return originals["swap_in_selected_blocks"](self, req_pool_indices, seq_lens, top_k_blocks,
                                                    p.hot_index[layer_id], sparse_block_size)

    def swap_pages(self, req_pool_indices, compressed_seq_lens, top_k_result, layer_id):
        """DeepSeek-V4 C4: ``layer_id`` is a pool-local index, so the layer comes from the MQALayer context."""
        p, s = _state()["binding"], _slab(self)
        layer = p.current_layer()
        if layer in p.resident_index:
            return resident_slots(self, req_pool_indices, compressed_seq_lens, top_k_result, torch)
        if s is not None:
            return route_swap(s, self, req_pool_indices, compressed_seq_lens, top_k_result, layer)
        return originals["swap_in_selected_pages"](self, req_pool_indices, compressed_seq_lens, top_k_result,
                                                   p.hot_index[layer])

    hooks = dict(__init__=init, admit_request_direct=admit_request_direct, alloc_device_buffer=alloc_device_buffer,
                 _grow_device_buffers=grow, _eager_backup_previous_token=eager_backup,
                 admit_request_into_staging=admit_request_into_staging, request_finished=request_finished,
                 swap_in_selected_blocks=swap_blocks, swap_in_selected_pages=swap_pages)
    for name, hook in hooks.items():
        originals[name] = _patch(state, cls, name, hook)


def resident_slots(c, req_pool_indices, compressed_seq_lens, top_k, torch):
    """DeepSeek-V4 C4 layers resident outside the slab: record p lives at row req_to_token[rid, 4p + 3] / 4 of
    the resident pool. Graph-safe gathers into the native output buffer; -1 for invalid and padded rows."""
    out = c.top_k_device_locs_buffer[:req_pool_indices.size(0), :top_k.size(1)]
    rows = c.req_to_token_pool.req_to_token[req_pool_indices.long()[:, None], ((top_k.long() + 1) * 4 - 1).clamp(min=0)]
    real = torch.arange(top_k.size(0), device=top_k.device)[:, None] < c.num_real_reqs
    valid = (top_k >= 0) & (top_k < compressed_seq_lens[:, None]) & real
    out.copy_(torch.where(valid, rows // 4, -1))
    return out
class C4Layer:
    """``layer_mapping`` pool of one slab C4 layer (DeepSeek-V4): its getter returns the whole slab in decode and
    its current plane otherwise, and locations translate through the layer's route (decode: the tail row of
    the request, or slot 0's tail for tokens that close no group)."""
    PAGE = 64

    def __init__(self, placement, layer, main, torch):
        self.placement, self.layer, self.main, self.torch = placement, layer, main, torch

    def __getattr__(self, name):        # page_size, kv_layout, dtype, store_dtype, bytes_per_page_padded, ...
        return getattr(self.main, name)

    def get_key_buffer(self, _index):
        p, s = self.placement, self.placement.state
        if s is None:
            raise RuntimeError("Slab layers are read only after the coordinator binds")
        pages = s.rows.k
        if p.decode is None:
            base, rows = s.plane(self.layer)
            pages = pages[base // self.PAGE:(base + rows) // self.PAGE]
        return pages.view(self.main.dtype) if self.main.store_dtype != self.main.dtype else pages

    def _translate_loc_to_hisparse_device(self, loc):
        p, s = self.placement, self.placement.state
        if p.decode is not None:
            tail = s.layout.tail_view(self.layer)
            return self.torch.where(loc > 0, p.decode.req_pool_indices + tail, tail)
        return self.main.full_to_hisparse_device_index_mapping[loc] if s.offloaded(self.layer) else loc

    def translate_loc_to_hisparse_device(self, loc):
        return self._translate_loc_to_hisparse_device(loc).to(self.torch.int32)

    def set_key_buffer(self, *args, **kwargs):
        raise NotImplementedError("The C4 slab is written by the CUDA compressor only (no HIP pack store)")

    set_key_buffer_fused = set_key_buffer


def _dsv4_pool(state, torch, module):
    """Pool hooks of DeepSeek-V4: S or U C4 planes replace the all-C4-layer HiSparse pool."""
    pool_cls, hisparse_cls, config = module.DeepSeekV4TokenToKVPool, module.HiSparseC4DevicePool, state["config"]
    single_cls, indirect, originals = hisparse_cls.__mro__[1], config.feature("route_indirect"), {}

    class SlabC4Pool(hisparse_cls):
        """HiSparse C4 pool over U whose per-layer buffers are the tail views of one page slab."""
        _rlh_slab = None

        def create_buffer(self, *, num_pages):
            layout = state["pending_layout"]
            if self._rlh_slab is None:
                self._rlh_slab, self._rlh_next = (super().create_buffer(num_pages=layout.rows // C4Layer.PAGE),), 0
            first = layout.tail_view(layout.geometry.layers[self._rlh_next]) // C4Layer.PAGE
            self._rlh_next += 1
            return self._rlh_slab[0][first:first + layout.tail_stride // C4Layer.PAGE]

    def make_kv_pool(self, **kwargs):
        if kwargs.get("cls") is not hisparse_cls:
            return originals["_make_kv_pool"](self, **kwargs)
        size, page_size, layer_num = kwargs["size"], kwargs["page_size"], kwargs["layer_num"]
        global_page_size, kv_layout = kwargs["global_page_size"], kwargs["kv_layout"]

        def build(**changes):
            return originals["_make_kv_pool"](self, **dict(kwargs, **changes))

        c4 = tuple(i for i in range(self._stage_start, self._stage_end) if self.compression_ratios[i] == 4)
        if len(c4) != layer_num:
            raise RuntimeError("The pinned pool built its C4 HiSparse pool differently than expected")
        lists, slab, resident = split_layers(config, c4)
        layout = None
        if indirect:
            from sglang.srt.mem_cache.sparsity import parse_hisparse_config
            pages = module._num_dsv4_physical_kv_pages
            geometry = do.Geometry(layers=slab, hot_rows=pages(size, page_size, global_page_size) * page_size,
                                   resident_rows=pages(self.full_size // 4, page_size, global_page_size) * page_size,
                                   hot_slots=parse_hisparse_config().device_buffer_size, slots=state["slots"],
                                   page=page_size, page_bytes=kv_layout.page_bytes(page_size), buffers=1)
            layout = state["pending_layout"] = do.SlabLayout(geometry, len(lists["0"]))
        try:
            main = build(size=size, layer_num=len(slab), cls=SlabC4Pool if indirect else hisparse_cls)
        finally:
            state.pop("pending_layout", None)
        resident_pool = build(size=self.full_size // 4, layer_num=len(resident), cls=single_cls) if resident else None
        self._rlh_c4 = dict(lists=lists, slab=slab, resident=resident, layout=layout, main=main,
                            resident_pool=resident_pool)
        return main

    def init_mapping(self):
        originals["_init_compressed_layer_mapping"](self)
        c4 = getattr(self, "_rlh_c4", None)
        if c4 is None:
            raise RuntimeError("The DeepSeek-V4 pool did not build its C4 pool through the layer offload hook")
        p = Placement(config, c4["lists"], c4["slab"], c4["resident"], main=c4["main"],
                      resident_pool=c4["resident_pool"], layout=c4["layout"])
        if indirect:
            p.facades = {layer: C4Layer(p, layer, c4["main"], torch) for layer in p.slab}
        for layer in (*p.slab, *p.resident):
            item = self.layer_mapping[layer]
            index, pool = ((p.resident_index[layer], p.resident_pool) if layer in p.resident_index else
                           (p.hot_index[layer], p.facades.get(layer, c4["main"])))
            self.layer_mapping[layer] = type(item)(compress_ratio=4, compress_layer_id=index, compress_kv_pool=pool)
        c4["main"]._rlh_placement = p
        state["pools"].append(c4["main"])

    def translate(name):
        def hook(self, loc):
            p = getattr(self, "_rlh_placement", None)
            if p is None or p.layer is None or (p.layer not in p.resident_index and not p.facades):
                return originals[name](self, loc)
            if p.layer in p.resident_index:
                return loc.to(torch.int32) if name == "translate_loc_to_hisparse_device" else loc
            return getattr(p.facades[p.layer], name)(loc)
        return hook

    for owner, name, hook in ((pool_cls, "_make_kv_pool", make_kv_pool),
                              (pool_cls, "_init_compressed_layer_mapping", init_mapping),
                              (hisparse_cls, "translate_loc_to_hisparse_device",
                               translate("translate_loc_to_hisparse_device")),
                              (hisparse_cls, "_translate_loc_to_hisparse_device",
                               translate("_translate_loc_to_hisparse_device"))):
        originals[name] = _patch(state, owner, name, hook)


def _dsv4_backend(state):
    """Layer context for the C4 indexer, compressor and attention of DeepSeek-V4, installed at bind."""
    layer_cls = importlib.import_module("sglang.srt.models.deepseek_v4").MQALayer

    def forward(self, x, positions, forward_batch, *args, **kwargs):
        with _state()["binding"].in_layer(self.layer_id, forward_batch):
            return original(self, x, positions, forward_batch, *args, **kwargs)

    original = _patch(state, layer_cls, "forward", forward)


def check_dsv4(c, kv, p, *, torch):
    errors, main = [], p.main
    if not c.is_dsv4_hisparse or c.mem_pool_device is not main or kv.c4_kv_pool is not main or c.compress_ratio != 4:
        errors.append("the DeepSeek-V4 binding needs the C4 HiSparse coordinator over the placement's C4 pool")
    if getattr(kv, "uniform_fp8", False) or main.kv_layout.value != "v4":
        errors.append("the C4 slab supports the V4 page layout only (no trtllm uniform FP8)")
    if (main.layer_num != len(p.slab) or c.req_device_buffer_tokens.shape[0] != len(p.slab)
            or c.mem_pool_host.layer_num != len(p.slab)):
        errors.append("C4 planes, swap tables and host history must cover exactly the slab layers")
    return errors
def _sizing(state, torch, context):
    """One HBM budget: resident eligible layers are charged their logical rows when pools are sized, and the
    pool builder records the request slots (tails) and the model family before any pool exists."""
    kvc_cls = importlib.import_module("sglang.srt.mem_cache.kv_cache_configurator").KVCacheConfigurator
    configurators = importlib.import_module("sglang.srt.model_executor.pool_configurator")
    models = importlib.import_module("sglang.srt.configs.model_config")
    config, k, originals = state["config"], len(state["config"].offload_lists[0]), {}

    def build(self, *, sizes, is_dsa_model, is_dsv4_model, req_to_token_pool):
        model = "deepseek_v4" if is_dsv4_model else \
            "minimax_m3" if models.is_minimax_sparse(self.model_config.hf_config) else None
        if model is None or self.is_draft_worker or state["model"] not in (None, model):
            raise RuntimeError("Layer offload supports the MiniMax-M3 and DeepSeek-V4 target pools only")
        if state["model"] is None:
            state["model"], state["slots"] = model, req_to_token_pool.req_to_token.shape[0]
            if model == "minimax_m3":
                _minimax_pool(state, torch, importlib.import_module("sglang.srt.mem_cache.memory_pool"),
                              importlib.import_module("sglang.srt.mem_cache.hisparse_memory_pool"))
            else:
                _dsv4_pool(state, torch, importlib.import_module("sglang.srt.mem_cache.deepseek_v4_memory_pool"))
        return originals["_build_token_to_kv_pool"](self, sizes=sizes, is_dsa_model=is_dsa_model,
                                                    is_dsv4_model=is_dsv4_model, req_to_token_pool=req_to_token_pool)

    def minimax_cell(self, kvc, num_layers):
        cell = originals["_compute_cell_size"](self, kvc, num_layers)
        model = kvc.model_config
        if not models.is_minimax_sparse(model.hf_config) or not kvc.server_args.enable_hisparse \
                or models.uses_minimax_m3_nvfp4_native_stack(model.quantization):
            raise RuntimeError("Layer offload sizes MiniMax sparse BF16/FP8 pools with --enable-hisparse only")
        from sglang.srt.mem_cache.sparsity import parse_hisparse_config
        _, sparse = models.get_minimax_sparse_layer_ids(models.get_minimax_sparse_attention_config(model.hf_config))
        eligible = [layer for layer in sparse if kvc.layer_info.start_layer <= layer < kvc.layer_info.end_layer]
        ratio = parse_hisparse_config().host_to_device_ratio
        row = 2 * model.get_num_kv_heads(context.get_parallel().attn_tp_size) * model.head_dim \
            * torch._utils._element_size(kvc.kv_cache_dtype)
        extra = resident_extra_cell_bytes(eligible, k, ratio=ratio, row_bytes=row)
        state["charge"] = dict(stock_cell_bytes=cell, resident_extra_bytes=extra, cell_bytes=cell + extra,
                               resident_layers=len(eligible) - k, host_to_device_ratio=ratio)
        return cell + extra

    def dsv4_bytes(self):
        stock = originals["_get_bytes_per_full_token"](self)
        resident = self.num_layers(4) - k
        if resident < 0:
            raise RuntimeError("The offload list names more C4 layers than this stage owns")
        extra = resident * (1 / 4 - 1 / (4 * self.c4_shrink_factor)) * self._get_paged_kv_bytes_per_token(4)
        state["charge"] = dict(stock_bytes_per_full_token=stock, resident_extra_bytes=extra,
                               resident_c4_layers=resident, host_to_device_ratio=self.c4_shrink_factor)
        return stock + extra

    for owner, name, hook in ((kvc_cls, "_build_token_to_kv_pool", build),
                              (configurators.DefaultPoolConfigurator, "_compute_cell_size", minimax_cell),
                              (configurators.DSV4PoolConfigurator, "_get_bytes_per_full_token", dsv4_bytes)):
        originals[name] = _patch(state, owner, name, hook)


def _trigger(state, torch):
    """``hisparse_offload_list`` in POST /set_internal_state switches every TP rank to registered list i."""
    module = importlib.import_module("sglang.srt.managers.scheduler_components.internal_state")
    controller_cls = module.SchedulerInternalStateController

    def switch(controller, key, value):
        p = _state()["binding"]
        if p is None or p.protocol is None:
            raise module.UpdateRejected(f"{key} needs a bound worker with features.dynamic_offload")
        if type(value) is not int or str(value) not in p.lists:
            raise module.UpdateRejected(f"{key} must be the index of a registered offload list 0..{len(p.lists) - 1}")
        scheduler, c = controller.scheduler, p.binding
        p.state.busy = lambda: ("a chunked prefill is in flight" if scheduler.chunked_req is not None else
                                "prefill staging copies are in flight" if c.ack_staging_queue else None)
        rank = torch.distributed.get_rank()
        try:
            emit(p.protocol.switch(str(value)), rank)
        except do.SwitchError as error:
            emit(dict(p.protocol.receipts[-1], error=str(error)) if p.protocol.receipts else
                 dict(event="hisparse_dynamic_offload_switch", status="refused", target=str(value), error=str(error)),
                 rank)
            raise module.UpdateRejected(str(error)) from None

    def update_handlers(self):
        handlers = original(self)
        handlers[LIST_KEY] = module._UpdateHandler(functools.partial(switch, self), lambda key, value: None)
        return handlers

    original = _patch(state, controller_cls, "_update_handlers", update_handlers)


def install(config, config_sha256):
    """Patch pinned symbols in this worker before the KV pools, coordinator and backend exist.

    Returns False and patches nothing unless ``placement.mode=fixed_layers`` over the stock hot pool.
    A process keeps one frozen configuration; a different one requires a new process.
    """
    global _STATE
    reasons = sc.unsupported_reasons(config)
    if reasons:
        raise RuntimeError("; ".join(reasons))
    if not config.uses_fixed_layers or config.uses_arena:
        return False
    if _STATE is not None:
        if _STATE["config"].digest != config.digest:
            raise RuntimeError("The offload configuration is frozen for this worker")
        return True
    import torch
    import torch.distributed as dist
    import sglang
    bad = {k: v for k, v in verify_digests(Path(sglang.__file__).resolve().parent).items() if v != "ok"}
    expected = json.loads(PROVENANCE.read_text())["dynamic_offload_generated_sha256"][ROUTE_HEADER.name]
    if bad or route_sha256() != expected:
        raise RuntimeError(f"Pinned SGLang source or route header mismatch: {bad or ROUTE_HEADER.name}")
    context = importlib.import_module("sglang.srt.runtime_context")
    state = dict(config=config, digest=config_sha256, pools=[], binding=None, model=None, slots=None, patched=[],
                 charge=None)
    _STATE = state
    try:
        _sizing(state, torch, context)
        _coordinator(state, torch, dist, context)
        if config.feature("dynamic_offload"):
            _trigger(state, torch)
    except BaseException:
        _restore(state)
        _STATE = None
        raise
    return True


def _restore(state):
    for owner, name, original, _ in reversed(state["patched"]):
        setattr(owner, name, original)
    state["patched"].clear()


def uninstall():
    """Restore pinned symbols only if no pool or coordinator was built under the hooks."""
    global _STATE
    if _STATE is None:
        return False
    if _STATE["pools"] or _STATE["binding"] is not None:
        raise RuntimeError("Layer offload state exists in this process; start a new process to restore stock behavior")
    for owner, name, _, hook in _STATE["patched"]:
        current = owner.__dict__.get(name) if isinstance(owner, type) else getattr(owner, name)
        if getattr(current, "__wrapped__", None) is None or current.__code__ is not hook.__code__:
            raise RuntimeError("Another hook replaced these symbols; refusing to overwrite it")
    _restore(_STATE)
    _STATE = None
    return True
# Planning geometry per model (one rank); a worker re-derives every value from its real pools.
PRESETS = {
    "minimax_m3": dict(eligible="0-59", device_tokens=396_544, ratio=2, hot_slots=4096, slots=129, page=128,
                       row_bytes=256, top_k=16, top_k_block_size=128),
    "deepseek_v4": dict(eligible="2-42:2", device_tokens=131_072, ratio=2, hot_slots=1024, slots=129, page=64,
                        row_bytes=584, top_k=512, top_k_block_size=1),
}


def parse_layers(spec):
    """'0-59', '2-42:2' or '3,7,11' -> ascending absolute layer IDs."""
    layers = []
    for part in spec.split(","):
        span, _, step = part.partition(":")
        first, _, last = span.partition("-")
        layers += range(int(first), int(last or first) + 1, int(step or 1))
    return tuple(sorted(set(layers)))


def plan_geometry(config, model, *, eligible, device_tokens, ratio, hot_slots, slots, page, row_bytes, **_):
    """(lists, slab layers, direct resident layers, slab layout or None) of one rank's pools."""
    lists, slab, resident = split_layers(config, eligible)
    if not config.feature("route_indirect"):
        return lists, slab, resident, None
    if model == "deepseek_v4":       # page-padded C4 pages; rows as _num_dsv4_physical_kv_pages counts them
        pages = lambda size: -(-(size + 4 * page) // page) * page
        geometry = do.Geometry(layers=slab, hot_rows=pages(device_tokens), resident_rows=pages(device_tokens * ratio),
                               hot_slots=hot_slots, slots=slots, page=page,
                               page_bytes=-(-page * row_bytes // 576) * 576, buffers=1)
    else:
        geometry = do.Geometry(layers=slab, hot_rows=device_tokens + page, resident_rows=device_tokens * ratio + page,
                               hot_slots=hot_slots, slots=slots, page=page, page_bytes=page * row_bytes)
    return lists, slab, resident, do.SlabLayout(geometry, len(lists["0"]))


def dry_run(root, config, model, geo):
    """CPU-only adapter resolution against a pinned source tree: no torch, model or GPU."""
    lists, slab, resident, layout = plan_geometry(config, model, **geo)
    digests, symbols = verify_digests(root), resolve_symbols(root, model)
    manifest = json.loads(PROVENANCE.read_text())
    header_ok = route_sha256() == manifest["dynamic_offload_generated_sha256"][ROUTE_HEADER.name]
    ok = header_ok and all(v == "ok" for v in digests.values()) and all(s["status"] == "ok" for s in symbols)
    k, frozen_hot = len(lists["0"]), len(lists["0"]) * (geo["device_tokens"] + geo["page"])
    route = None if layout is None else dict(
        jit_module=JIT_NAME, header=ROUTE_HEADER.name, header_sha256=route_sha256(),
        header_matches_provenance=header_ok,
        template_args=route_template(960, geo["top_k"], geo["hot_slots"], model == "deepseek_v4",
                                     geo["top_k_block_size"], 4 if model == "deepseek_v4" else 1),
        launches="one per switchable layer and decode step; route table [|U|, 4] int32 (mode, base, tail view, 0)")
    return dict(status="pass" if ok else "fail", model=model, config_sha256=config.digest,
                resolved_config=config.as_dict(), capabilities=sc.capabilities(config), digests=digests,
                symbols=symbols, placement=dict(lists={n: list(v) for n, v in lists.items()}, slab=list(slab),
                                                direct_resident=list(resident), eligible=list(geo["eligible"])),
                geometry={key: list(v) if isinstance(v, tuple) else v for key, v in geo.items()},
                layout=None if layout is None else dict(rows=layout.rows, k=layout.k, tail_base=layout.tail_base,
                                                        tail_stride=layout.tail_stride, digest=layout.digest),
                ledger=do.ledger(layout) if layout is not None else
                dict(frozen_hot_rows=frozen_hot, resident_layers=len(resident),
                     resident_rows_each=geo["device_tokens"] * geo["ratio"] + geo["page"]),
                route_kernel=route, trigger=None if not config.feature("dynamic_offload") else
                f'POST /set_internal_state {{"server_args": {{"{LIST_KEY}": <list index>}}}}',
                pending_hardware=list(PENDING_HARDWARE))


def build_receipt(config, model, geo, output):
    """GPU-only: JIT-build the route kernel variant of this arm and record toolchain and artifact identity."""
    import platform
    import shutil
    import subprocess
    import torch
    import sglang
    from sglang.kernels.jit.utils.compile.cache import cache_root
    bad = {k: v for k, v in verify_digests(Path(sglang.__file__).resolve().parent).items() if v != "ok"}
    if bad or torch.cuda.get_device_capability() != (8, 0) or not config.feature("route_indirect"):
        raise RuntimeError(f"Build receipt needs route_indirect, the pinned source and sm_80; mismatches: {bad}")
    template = route_template(960, geo["top_k"], geo["hot_slots"], model == "deepseek_v4", geo["top_k_block_size"],
                              4 if model == "deepseek_v4" else 1)
    _route_module(geo["row_bytes"], *template)

    def run(*command):
        if shutil.which(command[0]) is None:
            return None
        return subprocess.run(command, capture_output=True, text=True, check=False).stdout.strip()

    artifacts = [dict(name=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      cuobjdump_list_elf=run("cuobjdump", "--list-elf", str(path)))
                 for path in sorted(cache_root().rglob(f"*{JIT_NAME}*.so"))]
    record = dict(event="hisparse_dynamic_offload_build", model=model, config_sha256=config.digest,
                  template_args=template, header_sha256=route_sha256(), python=platform.python_version(),
                  torch=torch.__version__, torch_cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                  capability=list(torch.cuda.get_device_capability()), nvcc=run("nvcc", "--version"),
                  artifacts=artifacts)
    with Path(output).open("x") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Layer offload adapter: CPU dry-run against pinned SGLang, or the "
                                                 "GPU build receipt of the route kernel.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("dry-run", "build-receipt"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True, help="placement.mode=fixed_layers study config")
        sub.add_argument("--model", choices=MODELS, required=True)
        for key, value in PRESETS["minimax_m3"].items():
            sub.add_argument("--" + key.replace("_", "-"), type=str if key == "eligible" else int, default=None,
                             help=f"default per model, e.g. {value}")
        if name == "dry-run":
            sub.add_argument("--sglang-src", type=Path, required=True, help="pinned checkout or its python/sglang")
        else:
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--allow-cuda-build", action="store_true")
    args = parser.parse_args(argv)
    try:
        config, _ = sc.read(args.config)
        if not config.uses_fixed_layers or config.uses_arena:
            raise ValueError("Only placement.mode=fixed_layers over hot_pool.layout=native uses this adapter")
        reasons = sc.unsupported_reasons(config)
        if reasons:
            raise ValueError("; ".join(reasons))
        geo = {key: getattr(args, key) if getattr(args, key) is not None else value
               for key, value in PRESETS[args.model].items()}
        geo["eligible"] = parse_layers(geo["eligible"])
        if args.command == "dry-run":
            root = args.sglang_src / "python" / "sglang" if (args.sglang_src / "python" / "sglang").is_dir() \
                else args.sglang_src
            report = dry_run(root, config, args.model, geo)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    if args.command == "dry-run":
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "pass" else 1
    if not args.allow_cuda_build:
        parser.error("Run only in an approved idle GPU process with --allow-cuda-build")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    print(json.dumps(build_receipt(config, args.model, geo, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
