"""SGLang integration of a frozen offload set S (explicit, process-local, opt-in).

Importing this module imports neither torch nor SGLang. ``install`` runs in a fresh
worker before the KV pool exists and patches only the pinned symbols below; the stock
and all-default configurations install nothing.

* ``MiniMaxSparseKVPool.__init__``: the pinned constructor runs unchanged except that
  its HiSparse main pool is built for S only (planes in S order, ``start_layer=0``), so
  the host pool, staging, decode backup and swap tables cover S only. One stock
  ``MHATokenToKVPool`` then holds the complete main KV of every other sparse layer at
  the logical location, sized like the index pools (``size * host_to_device_ratio``).
* ``get_key_buffer``/``get_value_buffer``/``get_kv_buffer``/``set_kv_buffer``: sparse
  layers route to (pool, plane) and writes pass ``layer_id_override``; dense layers keep
  the stock route. ``get_kv_size_bytes`` includes the resident planes.
* ``DefaultPoolConfigurator._compute_cell_size``: each resident sparse layer costs
  ``host_to_device_ratio`` rows per device token, so automatic sizing keeps one HBM
  budget and the hot budget follows from what remains.
* ``MiniMaxSparseAttnBackend._hisparse_swap_in_blocks``: resident layers return no
  HiSparse slots with ``features.resident_direct``, so the stock consumer reads their
  ``req_to_token`` rows directly; without it ``rlh_layer_placement::resolve_resident``
  (``layer_placement_kernel.cuh``) writes the same rows as slots into the native
  output buffer, the correct resident resolver the direct path is measured against.
* ``minimax_sparse_ops.minimax_sparse.minimax_sparse_prefill``: resident planes are
  read without the HiSparse location mapping.
* Stock hot-pool layout only: ``HiSparseCoordinator.__init__`` (gate, TP agreement,
  receipt), ``swap_in_selected_blocks`` (absolute layer to S plane for the native
  kernel) and ``admit_request_direct`` (rejected). With ``features.split_copy`` the
  swap is two pinned launches on the same stream: the fused kernel instantiated with
  ``RecordMissPlan`` and ``SkipIO`` updates hits, LRU order and slots and records the
  miss plan, then ``copy_cache_planned`` copies every miss of the rank over
  ``SPLIT_COPY_CTAS`` blocks instead of one block per request. Arena layouts bind
  through ``shared_lru_device``, whose resolver addresses the planes of S.
"""
import argparse
import ast
import functools
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys

from . import layer_placement as reference
from . import shared_lru as hot_reference
from . import shared_lru_device as arena
from . import study_config as sc

MEMORY_POOL = "srt/mem_cache/memory_pool.py"
BACKEND = "srt/layers/attention/minimax_sparse_backend.py"
OPS = "srt/layers/attention/minimax_sparse_ops/minimax_sparse.py"
OPS_MODULE = arena.CONSUMER_MODULE
SWAP_OPS = "kernels/ops/kvcache/hisparse.py"
PREFILL = "minimax_sparse_prefill"
MAIN_POOL_MODULE = "sglang.srt.mem_cache.hisparse_memory_pool"
# (pinned file, qualified symbol, required leading parameters or None)
SYMBOLS = (
    (MEMORY_POOL, "MiniMaxSparseKVPool.__init__", ("self", "size", "page_size", "dtype", "head_num", "head_dim",
                                                   "idx_head_dim", "dense_layer_ids", "sparse_layer_ids", "device")),
    (MEMORY_POOL, "MiniMaxSparseKVPool._pool_for", ("self", "layer_id")),
    (MEMORY_POOL, "MiniMaxSparseKVPool._wait_for_layer", ("self", "layer_id")),
    (MEMORY_POOL, "MiniMaxSparseKVPool.get_key_buffer", ("self", "layer_id")),
    (MEMORY_POOL, "MiniMaxSparseKVPool.get_value_buffer", ("self", "layer_id")),
    (MEMORY_POOL, "MiniMaxSparseKVPool.get_kv_buffer", ("self", "layer_id")),
    (MEMORY_POOL, "MiniMaxSparseKVPool.set_kv_buffer", ("self", "layer", "loc", "cache_k", "cache_v", "k_scale",
                                                        "v_scale")),
    (MEMORY_POOL, "MiniMaxSparseKVPool.set_fused_kv_index_buffer", ("self", "layer", "loc", "cache_k", "cache_v")),
    (MEMORY_POOL, "MiniMaxSparseKVPool.get_kv_size_bytes", ("self",)),
    (MEMORY_POOL, "MHATokenToKVPool.set_kv_buffer", ("self", "layer", "loc_info", "cache_k", "cache_v", "k_scale",
                                                     "v_scale", "layer_id_override")),
    ("srt/mem_cache/hisparse_memory_pool.py", "HiSparseMHAMainPool.set_kv_buffer",
     ("self", "layer", "loc", "cache_k", "cache_v")),
    ("srt/mem_cache/pool_host/mha.py", "HiSparseMHATokenToKVPoolHost.backup_from_device_all_layer",
     ("self", "device_pool", "host_indices", "device_indices", "io_backend")),
    ("srt/model_executor/pool_configurator.py", "DefaultPoolConfigurator._compute_cell_size",
     ("self", "kvc", "num_layers")),
    (BACKEND, "MiniMaxSparseAttnBackend._hisparse_swap_in_blocks", ("self", "forward_batch", "topk_idx", "layer_id")),
    (BACKEND, "MiniMaxSparseAttnBackend.forward_extend", None),
    (BACKEND, "MiniMaxSparseAttnBackend.forward_decode", None),
    (OPS, PREFILL, ("q", "k_cache", "v_cache")),
    (arena.COORDINATOR, "HiSparseCoordinator.swap_in_selected_blocks",
     ("self", "req_pool_indices", "seq_lens", "top_k_blocks", "layer_id", "sparse_block_size")),
    (SWAP_OPS, "_jit_sparse_module", ("item_size_bytes", "block_size", "num_top_k", "hot_buffer_size", "is_mla",
                                      "is_dsv4_layout", "top_k_block_size", "top_k_is_blocks", "record_miss_plan",
                                      "skip_io")),
    (SWAP_OPS, "_jit_copy_planned_module", ("block_size", "is_mla", "is_dsv4_layout", "skip_io")),
)
PENDING_HARDWARE = ("sm80_build", "cuda_graph_capture_replay", "tp8_split_pool_receipt", "physical_memory_receipt",
                    "serving_tpot_throughput")
KERNEL_SOURCE = Path(__file__).with_name("layer_placement_kernel.cuh")
JIT_NAME = "rlh_layer_placement_resident"
RESIDENT_BLOCK_SIZE = 256      # threads per graph row; one CTA writes K * block slots
SPLIT_COPY_CTAS, SPLIT_COPY_THREADS = 128, 256    # features.split_copy: one fixed copy grid of 1024 warps


@functools.cache
def _resident_module(num_top_k, sparse_block_size):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH
    args = make_cpp_args(RESIDENT_BLOCK_SIZE, num_top_k, sparse_block_size)
    return load_jit(JIT_NAME, *args, cuda_files=[str(KERNEL_SOURCE)],
                    cuda_wrappers=[("resolve_resident", f"rlh_layer_placement::resolve_resident<{args}>")],
                    extra_include_paths=[str(KERNEL_PATH / "csrc")])


def kernel_sha256():
    return hashlib.sha256(KERNEL_SOURCE.read_bytes()).hexdigest()


def pinned_digests():
    manifest = json.loads(arena.PROVENANCE.read_text())
    return {**arena.pinned_digests(), **manifest["layer_placement_dependency_sha256"]}


def verify_digests(root):
    return arena.verify_digests(root, pinned_digests())


def _params(node):
    return [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]


def _imports(node, module, name):
    return any(isinstance(n, ast.ImportFrom) and n.module == module and any(a.name == name for a in n.names)
               for n in ast.walk(node))


def resolve_symbols(root):
    """AST check of every patched or relied-upon symbol and call contract; never imports SGLang."""
    trees, results, nodes = {}, [], {}
    for rel, symbol, params in SYMBOLS:
        path = Path(root) / rel
        if rel not in trees:
            trees[rel] = ast.parse(path.read_bytes(), filename=str(path)) if path.is_file() else None
        node = nodes[symbol] = None if trees[rel] is None else arena._find(trees[rel], symbol)
        status = "missing" if node is None else "ok"
        if node is not None and params is not None and tuple(_params(node))[:len(params)] != params:
            status = f"signature {_params(node)[:len(params)]}"
        results.append(dict(file=rel, symbol=symbol, status=status))
    init, extend = nodes["MiniMaxSparseKVPool.__init__"], nodes["MiniMaxSparseAttnBackend.forward_extend"]
    decode, prefill = nodes["MiniMaxSparseAttnBackend.forward_decode"], nodes[PREFILL]
    calls = [] if extend is None else [n for n in ast.walk(extend) if isinstance(n, ast.Call)
                                       and getattr(n.func, "id", None) == PREFILL]
    contracts = {
        (MEMORY_POOL, "MiniMaxSparseKVPool.__init__ builds HiSparseMHAMainPool through a call-time import"):
            init is not None and _imports(init, MAIN_POOL_MODULE, "HiSparseMHAMainPool")
            and {"main_pool_cls", "enable_hisparse", "host_to_device_ratio", "start_layer", "end_layer"}
            <= set(_params(init)),
        (BACKEND, f"forward_extend imports {PREFILL} at call time and passes loc_mapping by keyword"):
            extend is not None and _imports(extend, OPS_MODULE, PREFILL) and bool(calls)
            and all(len(c.args) >= 3 and "loc_mapping" in {k.arg for k in c.keywords} for c in calls),
        (BACKEND, "forward_decode swaps in through self._hisparse_swap_in_blocks"):
            decode is not None and any(isinstance(n, ast.Attribute) and n.attr == "_hisparse_swap_in_blocks"
                                       for n in ast.walk(decode)),
        (OPS, f"{PREFILL} accepts loc_mapping"): prefill is not None and "loc_mapping" in _params(prefill),
    }
    results += [dict(file=rel, symbol=name, status="ok" if ok else "contract changed")
                for (rel, name), ok in contracts.items()]
    return results


def split_pool_layers(config, *, sparse_layer_ids, start_layer, end_layer, enable_hisparse):
    """Placement of this stage's sparse layers, from the pool constructor's own arguments."""
    if not enable_hisparse:
        raise RuntimeError("placement.mode=fixed_layers needs --enable-hisparse for the offloaded layers")
    placement = reference.plan_placement(
        config, tuple(layer for layer in sparse_layer_ids if start_layer <= layer < end_layer))
    if not placement.offload:
        raise RuntimeError("The offload set is empty on this rank; S must name host-backed sparse layers")
    return placement


def resident_cell_bytes(placement, *, ratio, kv_heads, head_dim, kv_size):
    """Extra bytes per device token: a resident sparse layer holds ``ratio`` logical rows, not one hot row."""
    if type(ratio) is not int or ratio < 1:
        raise RuntimeError("host_to_device_ratio must be a positive integer for logical-row sizing")
    return len(placement.resident) * (ratio - 1) * 2 * kv_heads * head_dim * kv_size


def _same(a, b):
    return (a.data_ptr() == b.data_ptr() and tuple(a.shape) == tuple(b.shape) and a.stride() == b.stride()
            and a.dtype == b.dtype)


def classify_planes(pools, k_cache, v_cache):
    """``resident`` or ``hot`` when (k, v) are exactly one plane of a split pool, else None."""
    for kv in pools:
        for label, pool in ((reference.RESIDENT, kv._rlh_resident_pool), (reference.HOT, kv.main_pool)):
            if pool is not None and any(_same(k, k_cache) and _same(v, v_cache)
                                        for k, v in zip(pool.k_buffer, pool.v_buffer)):
                return label
    return None


def _nbytes(value):
    if isinstance(value, (list, tuple)):
        return sum(_nbytes(item) for item in value)
    return value.numel() * value.element_size()


def check_split(coordinator, *, torch, context):
    """Bind-time gate for the split pools; runs before the coordinator serves any request."""
    kv = coordinator.token_to_kv_pool_allocator.get_kvcache()
    routes = getattr(kv, "_rlh_routes", None)
    if routes is None:
        raise RuntimeError("The KV pool was built before install(); start a fresh worker")
    errors, main, resident = [], kv.main_pool, kv._rlh_resident_pool
    offload = tuple(getattr(main, "_rlh_offload_layers", ()))
    resident_layers = tuple(getattr(resident, "model_layer_ids", ())) if resident is not None else ()
    if not coordinator.is_m3_hisparse or getattr(coordinator, "is_dsv4_hisparse", False) \
            or coordinator.mem_pool_device is not main:
        errors.append("the fixed offload set supports the MiniMax M3 MHA HiSparse coordinator only")
    if coordinator.skip_io or coordinator.spec_swap.enabled or coordinator.enable_prefetch:
        errors.append("skip-IO probes, speculative decoding and shared-index prefetch are unsupported")
    if (not offload or tuple(main.model_layer_ids) != offload or main.start_layer != 0
            or main.layer_num != len(offload) or main.end_layer != len(offload)):
        errors.append("the HiSparse main pool must hold exactly the offloaded planes in S order")
    if set(routes) != set(offload) | set(resident_layers) or set(offload) & set(resident_layers) \
            or any(routes[layer] != (main, plane) for plane, layer in enumerate(offload)) \
            or any(routes[layer] != (resident, plane) for plane, layer in enumerate(resident_layers)):
        errors.append("layer routes must cover every local sparse layer exactly once")
    if getattr(coordinator.mem_pool_host, "layer_num", None) != len(offload):
        errors.append("pinned host history must cover the offloaded planes only")
    if (coordinator.req_device_buffer_tokens.shape[0] != len(offload)
            or coordinator.lru_slots.shape[0] != len(offload)):
        errors.append("coordinator swap tables must cover the offloaded planes only")
    for name, pool in (("offloaded", main), ("resident", resident)):
        if pool is not None and (pool.head_num != 1 or pool.is_quantized_kv_cache or pool.dtype != torch.bfloat16
                                 or pool.store_dtype != pool.dtype or pool.post_capture_active
                                 or getattr(pool, "kv_cache_layout", "nhd") != "nhd"):
            errors.append(f"{name} main KV must be one BF16 NHD head per rank, unquantized, without post-capture "
                          "backing")
    if resident is not None and (
            resident.size != coordinator.token_to_kv_pool_allocator.size_full or resident.start_layer != 0
            or resident.layer_num != len(resident_layers)
            or any(tuple(t.shape[1:]) != tuple(main.k_buffer[0].shape[1:]) for t in (*resident.k_buffer,
                                                                                     *resident.v_buffer))):
        errors.append("resident planes must span the logical allocator with the offloaded row shape")
    errors += arena.context_errors(context, torch=torch)
    if errors:
        raise RuntimeError("Fixed offload set incompatible: " + "; ".join(errors))
    return kv


class PlacementBinding:
    """Split pools of one bound coordinator (one TP rank) and the S plane map the swap uses."""

    def __init__(self, coordinator, kv, config, config_sha256, charge, *, torch):
        self.torch, self.config, self.config_sha256, self.charge = torch, config, config_sha256, charge
        self.main, self.resident_pool = kv.main_pool, kv._rlh_resident_pool
        self.offload = tuple(self.main._rlh_offload_layers)
        self.resident = () if self.resident_pool is None else tuple(self.resident_pool.model_layer_ids)
        self.plane_of = {layer: plane for plane, layer in enumerate(self.offload)}
        self.hot_rows, self.page = coordinator.device_buffer_size, self.main.page_size
        self.logical_tokens = coordinator.token_to_kv_pool_allocator.size_full
        self.resident_direct = config.uses_resident_direct
        self.split = None
        if config.uses_split_copy:      # miss plan of the layer in flight; allocated before any graph capture
            out = coordinator.top_k_device_locs_buffer
            self.split = tuple(torch.zeros(shape, dtype=dtype, device=out.device) for shape, dtype in (
                (out.shape, torch.int64), (out.shape, torch.int32), (out.shape[:1], torch.int32)))

    def plane(self, layer_id):
        plane = self.plane_of.get(layer_id)
        if plane is None:
            raise RuntimeError(f"Layer {layer_id} keeps complete GPU main KV; it never reaches the HiSparse swap")
        return plane

    def split_copy(self, coordinator, req_pool_indices, seq_lens, top_k_blocks, layer_id, sparse_block_size):
        """``features.split_copy``: the native swap of one S plane as plan, then copy. Same arguments and output
        buffer as the pinned method; stream order puts the copy before attention and the next layer's plan."""
        from sglang.kernels.ops.kvcache.hisparse import _jit_copy_planned_module, _jit_sparse_module
        c, plane, batch, (src, dst, count) = coordinator, self.plane(layer_id), req_pool_indices.size(0), self.split
        width = top_k_blocks.size(1) * sparse_block_size
        host_k, host_v = c.mem_pool_host.k_buffer[plane], c.mem_pool_host.v_buffer[plane]
        device_k, device_v = c.mem_pool_device.get_key_buffer(plane), c.mem_pool_device.get_value_buffer(plane)
        rows = (host_k, host_v, device_k, device_v)
        if (width > c.device_buffer_size or batch > src.shape[0]
                or any(t.stride(0) * t.element_size() != c.item_size_bytes for t in rows)):
            raise ValueError("Selection, batch or K/V row stride differs from the bound split-copy contract")
        out = c.top_k_device_locs_buffer[:batch, :width]
        planner = _jit_sparse_module(c.item_size_bytes, c.swap_in_block_size, top_k_blocks.size(1),
                                     c.device_buffer_size, is_mla=False, is_dsv4_layout=False,
                                     top_k_block_size=sparse_block_size, top_k_is_blocks=True, record_miss_plan=True,
                                     skip_io=True)
        planner.load_cache_to_device_buffer(top_k_blocks, c.req_device_buffer_tokens[plane], c.req_to_host_pool,
                                            c.req_device_buffer_token_locs[plane], host_k, host_v, device_k, device_v,
                                            out, req_pool_indices, seq_lens, c.lru_slots[plane], c.num_real_reqs, 1,
                                            c.item_size_bytes, src[:batch], dst[:batch], count[:batch])
        _jit_copy_planned_module(SPLIT_COPY_THREADS, False, False, False).copy_cache_planned(
            src[:batch], dst[:batch], count[:batch], c.num_real_reqs, host_k, host_v, device_k, device_v,
            SPLIT_COPY_CTAS, c.item_size_bytes)
        return out

    def resolve_resident(self, coordinator, req_pool_indices, seq_lens, top_k_blocks, req_to_token, block_size):
        """Resident resolver (``resident_direct=false``): one launch into the native output buffer, no sync."""
        torch, width = self.torch, top_k_blocks.size(1) * block_size
        buffer = coordinator.top_k_device_locs_buffer
        if (self.resident_direct or top_k_blocks.dtype != torch.int32 or req_to_token.dtype != torch.int32
                or width > buffer.shape[1] or req_pool_indices.size(0) > buffer.shape[0]):
            raise ValueError("Selection, req_to_token or batch differs from the bound resident resolver contract")
        out = buffer[:req_pool_indices.size(0), :width]
        _resident_module(top_k_blocks.size(1), block_size).resolve_resident(
            top_k_blocks, req_pool_indices, seq_lens, coordinator.num_real_reqs, req_to_token, out)
        return out

    def signature(self):
        resident = self.resident_pool
        return dict(offload_layers=list(self.offload), resident_layers=list(self.resident), hot_rows=self.hot_rows,
                    page=self.page, hot_plane_rows=int(self.main.k_buffer[0].shape[0]),
                    resident_plane_rows=None if resident is None else int(resident.k_buffer[0].shape[0]),
                    row_shape=[int(x) for x in self.main.k_buffer[0].shape[1:]], dtype=str(self.main.dtype),
                    logical_tokens=int(self.logical_tokens),
                    resident_path="direct" if self.resident_direct else "resolver",
                    resident_kernel_sha256=None if self.resident_direct else kernel_sha256(),
                    native_swap=None if self.config.uses_arena else "split" if self.split else "fused")

    def receipt(self, coordinator):
        torch, resident, host = self.torch, self.resident_pool, coordinator.mem_pool_host
        memory = None
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            memory = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                          max_allocated=torch.cuda.max_memory_allocated(), device_used=total - free, device_total=total)
        tables = dict(req_device_buffer_tokens=_nbytes(coordinator.req_device_buffer_tokens),
                      req_device_buffer_token_locs=_nbytes(coordinator.req_device_buffer_token_locs),
                      lru_slots=_nbytes(coordinator.lru_slots))
        return dict(event="hisparse_layer_placement_bound", config_file_sha256=self.config_sha256,
                    config_sha256=self.config.digest, resolved_config=self.config.as_dict(),
                    capabilities=sc.capabilities(self.config), signature=self.signature(), cell_size=self.charge,
                    resident_resolver=None if self.resident_direct else dict(
                        kernel="rlh_layer_placement::resolve_resident", kernel_sha256=kernel_sha256(),
                        block_size=RESIDENT_BLOCK_SIZE, output="coordinator.top_k_device_locs_buffer"),
                    split_copy=None if self.split is None else dict(
                        planner="load_cache_to_device_buffer<RecordMissPlan=true, SkipIO=true>",
                        copier="copy_cache_planned", ctas=SPLIT_COPY_CTAS, threads=SPLIT_COPY_THREADS),
                    bytes=dict(hot_main_kv=_nbytes(self.main.k_buffer) + _nbytes(self.main.v_buffer),
                               resident_main_kv=0 if resident is None else _nbytes(resident.k_buffer)
                               + _nbytes(resident.v_buffer),
                               host_pinned=_nbytes(host.k_buffer) + _nbytes(host.v_buffer), native_tables=tables,
                               split_copy_plan=0 if self.split is None else _nbytes(self.split)),
                    device_memory_at_bind=memory,
                    unavailable=dict(weights="not attributed at bind", graph_workspace="after capture",
                                     index_kv="identical in every arm; see the pool allocation log"),
                    pending_hardware=list(PENDING_HARDWARE))


_STATE = None
_POOL_HOOKS = ("__init__", "get_key_buffer", "get_value_buffer", "get_kv_buffer", "set_kv_buffer", "get_kv_size_bytes")
_COORDINATOR_HOOKS = ("__init__", "swap_in_selected_blocks", "admit_request_direct")


def bind(coordinator, *, torch, context):
    """Gate the split pools when the coordinator binds; returns this rank's placement binding."""
    state = _STATE
    if state is None:
        raise RuntimeError("The fixed offload set was not installed before the coordinator was built")
    if state["binding"] is not None:
        raise RuntimeError("One HiSparse coordinator per worker process")
    kv = check_split(coordinator, torch=torch, context=context)
    state["binding"] = PlacementBinding(coordinator, kv, state["config"], state["digest"], state["charge"],
                                        torch=torch)
    return state["binding"]


def install(config, config_sha256):
    """Patch pinned symbols in this worker before the KV pool, coordinator and backend exist.

    Returns False and patches nothing unless ``placement.mode=fixed_layers``. A process
    keeps one frozen configuration; a different one requires a new process.
    """
    global _STATE
    reasons = sc.unsupported_reasons(config)
    if reasons:
        raise RuntimeError("; ".join(reasons))
    if not config.uses_fixed_layers:
        return False
    if _STATE is not None:
        if _STATE["config"].digest != config.digest:
            raise RuntimeError("The offload set is frozen for this worker")
        return True
    import torch
    import torch.distributed as dist
    import sglang
    bad = {k: v for k, v in verify_digests(Path(sglang.__file__).resolve().parent).items() if v != "ok"}
    if bad:
        raise RuntimeError(f"Pinned SGLang source mismatch: {bad}")
    memory = importlib.import_module("sglang.srt.mem_cache.memory_pool")
    main_module = importlib.import_module(MAIN_POOL_MODULE)
    configurator_cls = importlib.import_module("sglang.srt.model_executor.pool_configurator").DefaultPoolConfigurator
    backend_cls = importlib.import_module("sglang.srt.layers.attention.minimax_sparse_backend").MiniMaxSparseAttnBackend
    ops = importlib.import_module(OPS_MODULE)
    coordinator_cls = importlib.import_module("sglang.srt.managers.hisparse_coordinator").HiSparseCoordinator
    context = importlib.import_module("sglang.srt.runtime_context")
    pool_cls, mha_cls = memory.MiniMaxSparseKVPool, memory.MHATokenToKVPool
    stock_layout = not config.uses_arena
    originals = dict(pool={name: pool_cls.__dict__.get(name) for name in _POOL_HOOKS},
                     cell=configurator_cls.__dict__.get("_compute_cell_size"),
                     swap_blocks=backend_cls.__dict__.get("_hisparse_swap_in_blocks"), prefill=getattr(ops, PREFILL),
                     coordinator={name: coordinator_cls.__dict__.get(name) for name in _COORDINATOR_HOOKS}
                     if stock_layout else {})
    owned = ([(fn, memory.__name__) for fn in originals["pool"].values()]
             + [(originals["cell"], configurator_cls.__module__), (originals["swap_blocks"], backend_cls.__module__),
                (originals["prefill"], ops.__name__)]
             + [(fn, coordinator_cls.__module__) for fn in originals["coordinator"].values()])
    if any(fn is None or getattr(fn, "__module__", None) != module or hasattr(fn, "__wrapped__")
           for fn, module in owned) or main_module.HiSparseMHAMainPool.__module__ != MAIN_POOL_MODULE:
        raise RuntimeError("Another hook already replaced a pinned symbol; refusing to stack hooks")
    state = dict(config=config, digest=config_sha256, pools=[], binding=None, charge=None, originals=originals,
                 classes=dict(pool=pool_cls, configurator=configurator_cls, backend=backend_cls, ops=ops,
                              coordinator=coordinator_cls))
    signature = inspect.signature(originals["pool"]["__init__"])
    real_main = main_module.HiSparseMHAMainPool

    def pool_init(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        if type(self) is not pool_cls or a["main_pool_cls"] is not mha_cls:
            raise RuntimeError("The fixed offload set supports the stock MiniMaxSparseKVPool with MHA main planes")
        placement = split_pool_layers(config, sparse_layer_ids=a["sparse_layer_ids"], start_layer=a["start_layer"],
                                      end_layer=a["end_layer"], enable_hisparse=a["enable_hisparse"])
        offload, resident, calls = placement.offload, placement.resident, []

        def offloaded_planes(*positional, **pool_kwargs):
            if positional or calls or pool_kwargs.get("layer_num") != len(placement.layers):
                raise RuntimeError("The pinned constructor built its HiSparse main pool differently than expected")
            calls.append(pool_kwargs)
            return real_main(**dict(pool_kwargs, layer_num=len(offload), start_layer=0, end_layer=len(offload)))

        main_module.HiSparseMHAMainPool = offloaded_planes
        try:
            originals["pool"]["__init__"](self, *args, **kwargs)
        finally:
            main_module.HiSparseMHAMainPool = real_main
        if len(calls) != 1:
            raise RuntimeError("The HiSparse main pool constructor was not intercepted")
        self.main_pool.model_layer_ids = offload
        self.main_pool._rlh_offload_layers = offload
        self._rlh_resident_pool = None
        if resident:
            pool = mha_cls(size=a["size"] * a["host_to_device_ratio"], page_size=a["page_size"], dtype=a["dtype"],
                           head_num=a["head_num"], head_dim=a["head_dim"], layer_num=len(resident),
                           device=a["device"], enable_memory_saver=a["enable_memory_saver"], start_layer=0,
                           end_layer=len(resident))
            pool.model_layer_ids = resident
            self.mem_usage += pool.mem_usage
            self._rlh_resident_pool = pool
        self._rlh_routes = {layer: (self.main_pool, plane) for plane, layer in enumerate(offload)}
        self._rlh_routes.update({layer: (self._rlh_resident_pool, plane) for plane, layer in enumerate(resident)})
        state["pools"].append(self)

    def route(self, layer_id):
        routes = getattr(self, "_rlh_routes", None)
        if routes is None:
            raise RuntimeError("MiniMaxSparseKVPool was built before install(); start a fresh worker")
        return routes.get(layer_id)

    def getter(name):
        def hook(self, layer_id):
            hit = route(self, layer_id)
            if hit is None:                                 # dense layers keep the stock route
                return originals["pool"][name](self, layer_id)
            self._wait_for_layer(layer_id)
            return getattr(hit[0], name)(hit[1])
        return functools.update_wrapper(hook, originals["pool"][name])

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None):
        hit = route(self, layer.layer_id)
        if hit is None:
            return originals["pool"]["set_kv_buffer"](self, layer, loc, cache_k, cache_v, k_scale, v_scale)
        hit[0].set_kv_buffer(layer, loc, cache_k, cache_v, k_scale, v_scale, layer_id_override=hit[1])

    def get_kv_size_bytes(self):
        k, v = originals["pool"]["get_kv_size_bytes"](self)
        if getattr(self, "_rlh_resident_pool", None) is None:
            return k, v
        rk, rv = self._rlh_resident_pool.get_kv_size_bytes()
        return k + rk, v + rv

    def compute_cell_size(self, kvc, num_layers):
        cell = originals["cell"](self, kvc, num_layers)
        from sglang.srt.configs.model_config import (get_minimax_sparse_attention_config, get_minimax_sparse_layer_ids,
                                                     is_minimax_sparse, uses_minimax_m3_nvfp4_native_stack)
        from sglang.srt.mem_cache.sparsity import parse_hisparse_config
        model = kvc.model_config
        if (not is_minimax_sparse(model.hf_config) or not kvc.server_args.enable_hisparse
                or uses_minimax_m3_nvfp4_native_stack(model.quantization)):
            raise RuntimeError("The fixed offload set sizes MiniMax sparse BF16/FP8 pools with --enable-hisparse only")
        _, sparse = get_minimax_sparse_layer_ids(get_minimax_sparse_attention_config(model.hf_config))
        placement = split_pool_layers(config, sparse_layer_ids=sparse, start_layer=kvc.layer_info.start_layer,
                                      end_layer=kvc.layer_info.end_layer, enable_hisparse=True)
        ratio = parse_hisparse_config().host_to_device_ratio
        kv_heads = model.get_num_kv_heads(context.get_parallel().attn_tp_size)
        kv_size = torch._utils._element_size(kvc.kv_cache_dtype)
        extra = resident_cell_bytes(placement, ratio=ratio, kv_heads=kv_heads, head_dim=model.head_dim,
                                    kv_size=kv_size)
        state["charge"] = dict(stock_cell_bytes=cell, resident_extra_bytes=extra, cell_bytes=cell + extra,
                               resident_layers=list(placement.resident), host_to_device_ratio=ratio)
        return cell + extra

    def hisparse_swap_in_blocks(self, forward_batch, topk_idx, layer_id):
        hit = route(self.kv_pool, layer_id)
        if hit is None:
            raise RuntimeError(f"Layer {layer_id} is not a sparse layer of the split pool")
        if hit[0] is self.kv_pool.main_pool:
            return originals["swap_blocks"](self, forward_batch, topk_idx, layer_id)
        if config.uses_resident_direct:
            return None             # direct: the stock consumer reads req_to_token rows of the resident plane
        binding = state["binding"]  # resolver: the same rows as HiSparse slots, consumed through hisparse_slots
        if binding is None or topk_idx.size(0) != 1:
            raise RuntimeError("Resident resolver needs a bound coordinator and one KV head per rank")
        slots = binding.resolve_resident(self.hisparse_coordinator, forward_batch.req_pool_indices,
                                         forward_batch.seq_lens, topk_idx[0], self.req_to_token, self.block_size_k)
        return slots.unsqueeze(0)

    @functools.wraps(originals["prefill"])
    def prefill_hook(*args, **kwargs):
        if len(args) < 3 or "loc_mapping" not in kwargs:
            raise RuntimeError(f"{PREFILL} expects the pinned positional K/V caches and loc_mapping keyword")
        planes = classify_planes(state["pools"], args[1], args[2])
        if planes is None:
            raise RuntimeError("Sparse prefill main KV is neither a resident nor an offloaded plane")
        if planes == reference.RESIDENT:
            kwargs["loc_mapping"] = None    # resident rows live at their logical locations
        elif kwargs["loc_mapping"] is None:
            raise RuntimeError("Offloaded prefill must read through the HiSparse location mapping")
        return originals["prefill"](*args, **kwargs)

    def coordinator_init(self, *args, **kwargs):
        originals["coordinator"]["__init__"](self, *args, **kwargs)
        binding = bind(self, torch=torch, context=context)
        identity = hashlib.sha256(json.dumps(binding.signature(), sort_keys=True).encode()
                                  + config.digest.encode()).hexdigest()
        hashes = [None] * self.tp_world_size
        dist.all_gather_object(hashes, identity, group=self.tp_group)
        if hashes != [identity] * self.tp_world_size:
            raise RuntimeError("TP ranks disagree on the offload set or the split pool geometry")
        self._request_layer_hisparse_placement = binding
        arena.emit(dict(binding.receipt(self), tp_size=self.tp_world_size), dist.get_rank())

    def swap_in_selected_blocks(self, req_pool_indices, seq_lens, top_k_blocks, layer_id, sparse_block_size):
        binding = getattr(self, "_request_layer_hisparse_placement", None)
        if binding is None:
            raise RuntimeError("Coordinator was not bound to the fixed offload set; install before construction")
        if binding.split is not None:
            return binding.split_copy(self, req_pool_indices, seq_lens, top_k_blocks, layer_id, sparse_block_size)
        return originals["coordinator"]["swap_in_selected_blocks"](
            self, req_pool_indices, seq_lens, top_k_blocks, binding.plane(layer_id), sparse_block_size)

    def admit_request_direct(self, req):
        raise RuntimeError("Direct (disaggregated) admission is not supported with a fixed offload set")

    hooks = dict(pool=dict(__init__=functools.update_wrapper(pool_init, originals["pool"]["__init__"]),
                           get_key_buffer=getter("get_key_buffer"), get_value_buffer=getter("get_value_buffer"),
                           get_kv_buffer=getter("get_kv_buffer"),
                           set_kv_buffer=functools.update_wrapper(set_kv_buffer, originals["pool"]["set_kv_buffer"]),
                           get_kv_size_bytes=functools.update_wrapper(get_kv_size_bytes,
                                                                      originals["pool"]["get_kv_size_bytes"])),
                 cell=compute_cell_size, swap_blocks=hisparse_swap_in_blocks, prefill=prefill_hook,
                 coordinator=dict(__init__=coordinator_init, swap_in_selected_blocks=swap_in_selected_blocks,
                                  admit_request_direct=admit_request_direct) if stock_layout else {})
    for name, hook in hooks["pool"].items():
        setattr(pool_cls, name, hook)
    configurator_cls._compute_cell_size = hooks["cell"]
    backend_cls._hisparse_swap_in_blocks = hooks["swap_blocks"]
    setattr(ops, PREFILL, hooks["prefill"])
    for name, hook in hooks["coordinator"].items():
        setattr(coordinator_cls, name, hook)
    state["hooks"] = hooks
    _STATE = state
    return True


def uninstall():
    """Restore pinned symbols only if no pool or coordinator was built under the hooks."""
    global _STATE
    if _STATE is None:
        return False
    if _STATE["pools"] or _STATE["binding"] is not None:
        raise RuntimeError("Split pools exist in this process; start a new process to restore stock behavior")
    classes, hooks, originals = _STATE["classes"], _STATE["hooks"], _STATE["originals"]
    current = ([(classes["pool"].__dict__.get(n), hooks["pool"][n]) for n in _POOL_HOOKS]
               + [(classes["configurator"].__dict__.get("_compute_cell_size"), hooks["cell"]),
                  (classes["backend"].__dict__.get("_hisparse_swap_in_blocks"), hooks["swap_blocks"]),
                  (getattr(classes["ops"], PREFILL), hooks["prefill"])]
               + [(classes["coordinator"].__dict__.get(n), h) for n, h in hooks["coordinator"].items()])
    if any(now is not hook for now, hook in current):
        raise RuntimeError("Another hook replaced these symbols; refusing to overwrite it")
    for name, fn in originals["pool"].items():
        setattr(classes["pool"], name, fn)
    classes["configurator"]._compute_cell_size = originals["cell"]
    classes["backend"]._hisparse_swap_in_blocks = originals["swap_blocks"]
    setattr(classes["ops"], PREFILL, originals["prefill"])
    for name, fn in originals["coordinator"].items():
        setattr(classes["coordinator"], name, fn)
    _STATE = None
    return True


def capacity(config, geo, *, dense_layers, ratio, index_bytes):
    """Device tokens left for the hot pools when resident planes are charged against the stock HBM budget."""
    placement = reference.plan_placement(config, tuple(geo["layers"]))
    page, row, N = geo["page"], geo["kv_row_bytes"], len(placement.layers)
    stock_size = geo["plane_rows"] - page
    common = dict(sparse_layers=N, dense_layers=dense_layers, row_bytes=row, ratio=ratio, index_bytes=index_bytes)
    stock_cell = reference.cell_bytes(offload_layers=N, **common)
    cell = reference.cell_bytes(offload_layers=len(placement.offload), **common)
    size = stock_size * stock_cell // cell // page * page
    return placement, dict(
        estimated=True, stock_device_tokens=stock_size, stock_cell_bytes=stock_cell, cell_bytes=cell,
        device_tokens=size, device_rows=size + page, logical_tokens=size * ratio, logical_rows=size * ratio + page,
        long_requests_upper_bound=size // (geo["hot_rows"] + page), host_to_device_ratio=ratio,
        index_bytes_per_token=index_bytes,
        note="same KV budget as the stock run; workers report the real auto-sized pools (omit --max-total-tokens)"
             + ("" if index_bytes else "; index KV excluded because --index-bytes-per-token was 0"))


def tensor_contracts(geo, placement, cap):
    S, R, H, page = len(placement.offload), len(placement.resident), geo["hot_rows"], geo["page"]
    row = f"{geo['kv_heads_per_rank']}, {geo['head_dim']}"
    return {
        "offloaded main KV": f"HiSparseMHAMainPool k/v_buffer x{S} of [{cap['device_rows']}, {row}]; plane p is S[p]",
        "resident main KV": f"MHATokenToKVPool k/v_buffer x{R} of [{cap['logical_rows']}, {row}] at req_to_token rows",
        "coordinator tables": f"req_device_buffer_tokens/locs [{S}, slots, {H + page}] int32; "
                              f"lru_slots [{S}, slots, {H}]",
        "host history": f"HiSparseMHATokenToKVPoolHost x{S} planes of the logical capacity; resident layers none",
        "decode": "offloaded: selector blocks -> swap on the S plane -> hisparse_slots; resident with "
                  "resident_direct: hisparse_slots None, the stock consumer reads req_to_token rows; resident "
                  "without it: resolve_resident writes those rows to top_k_device_locs_buffer [B, K] int32 (-1 "
                  "invalid) as hisparse_slots",
        "prefill": "offloaded: loc_mapping = full_to_hisparse_device_index_mapping; resident: loc_mapping None",
        "writes": "set_kv_buffer(..., layer_id_override=plane): offloaded rows translated, resident rows logical",
    }


def dry_run(root, config, geo, *, dense_layers=0, ratio=2, index_bytes=0):
    """CPU-only adapter resolution against a pinned source tree: no torch, model or GPU."""
    placement, cap = capacity(config, geo, dense_layers=dense_layers, ratio=ratio, index_bytes=index_bytes)
    if not placement.offload:
        raise ValueError("The offload set is empty; S must name host-backed sparse layers")
    plan, arena_report = None, None
    if config.uses_arena:
        plan = hot_reference.plan_arena(config, placement.offload, geo["hot_rows"], geo["selected"])
        arena_report = arena.dry_run(root, config, dict(geo, layers=list(placement.offload),
                                                        plane_rows=cap["device_rows"]))
    digests, symbols = verify_digests(root), resolve_symbols(root)
    ledger = reference.placement_ledger(placement=placement, hot_rows=geo["hot_rows"], page=geo["page"],
                                        item_bytes=geo["item_size_bytes"], slots=geo["slots"],
                                        device_rows=cap["device_rows"], logical_rows=cap["logical_rows"], plan=plan,
                                        counters=config.diagnostics == "counters")
    ok = (all(v == "ok" for v in digests.values()) and all(s["status"] == "ok" for s in symbols)
          and (arena_report is None or arena_report["status"] == "pass"))
    hot_pool = "native load_blocks_to_device_buffer_mha on S planes" if plan is None else arena_report["implementation"]
    if config.uses_split_copy:
        hot_pool = ("native swap on S planes in two pinned launches: load_cache_to_device_buffer<RecordMissPlan, "
                    f"SkipIO>, then copy_cache_planned over {SPLIT_COPY_CTAS} x {SPLIT_COPY_THREADS} threads")
    resident = ("stock flash_decode_with_gqa_share_sparse / flash_prefill_with_gqa_share_sparse over req_to_token rows"
                + ("" if config.uses_resident_direct else
                   f"; decode slots from rlh_layer_placement::resolve_resident (kernel sha256 {kernel_sha256()})"))
    return dict(status="pass" if ok else "fail", config_sha256=config.digest, resolved_config=config.as_dict(),
                capabilities=sc.capabilities(config), digests=digests, symbols=symbols,
                placement=dict(eligible=list(placement.layers), offload=list(placement.offload),
                               resident=list(placement.resident)),
                implementation=dict(hot_pool=hot_pool, resident=resident),
                arena_plan=None if arena_report is None else arena_report["plan"],
                geometry=geo, capacity=cap, tensor_contracts=tensor_contracts(geo, placement, cap),
                budget_units=ledger, pending_hardware=list(PENDING_HARDWARE))


def build_receipt(config, geo, output):
    """GPU-only: JIT-build what this arm adds (resident resolver, split-copy specializations of the pinned swap)
    and record toolchain and artifact identity."""
    import platform
    import shutil
    import subprocess
    import torch
    import sglang
    from sglang.kernels.jit.utils.compile.cache import cache_root
    bad = {k: v for k, v in verify_digests(Path(sglang.__file__).resolve().parent).items() if v != "ok"}
    if bad or torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError(f"Build receipt requires the pinned source on sm_80; mismatches: {bad}")
    if config.uses_resident_direct and not config.uses_split_copy:
        raise RuntimeError("features.resident_direct with the fused swap builds nothing; pass the resolver or "
                           "split_copy arm config")
    variant, names, split = (geo["top_k_blocks"], geo["block_size"]), [], None
    if not config.uses_resident_direct:
        _resident_module(*variant)
        names.append(JIT_NAME)
    if config.uses_split_copy:      # template arguments the binding passes: pinned swap_in_block_size default 960
        from sglang.kernels.ops.kvcache.hisparse import _jit_copy_planned_module, _jit_sparse_module
        planner = [geo["item_size_bytes"], 960, geo["top_k_blocks"], geo["hot_rows"]]
        _jit_sparse_module(*planner, is_mla=False, is_dsv4_layout=False, top_k_block_size=geo["block_size"],
                           top_k_is_blocks=True, record_miss_plan=True, skip_io=True)
        _jit_copy_planned_module(SPLIT_COPY_THREADS, False, False, False)
        names += ["sparse_cache", "sparse_copy_planned"]
        split = dict(planner_args=planner, copier_threads=SPLIT_COPY_THREADS, ctas=SPLIT_COPY_CTAS)

    def run(*command):
        if shutil.which(command[0]) is None:
            return None
        return subprocess.run(command, capture_output=True, text=True, check=False).stdout.strip()

    artifacts = [dict(name=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      cuobjdump_list_elf=run("cuobjdump", "--list-elf", str(path)))
                 for name in names for path in sorted(cache_root().rglob(f"*{name}*.so"))]
    receipt = dict(event="hisparse_layer_placement_build", config_sha256=config.digest,
                   template_args=None if config.uses_resident_direct else [RESIDENT_BLOCK_SIZE, *variant],
                   kernel_source_sha256=kernel_sha256(), split_copy=split,
                   python=platform.python_version(), torch=torch.__version__, torch_cuda=torch.version.cuda,
                   cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI), nccl=list(torch.cuda.nccl.version()),
                   device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                   nvcc=run("nvcc", "--version"), host_cxx=run("c++", "--version"), artifacts=artifacts)
    with Path(output).open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fixed offload set adapter: CPU dry-run against pinned SGLang, "
                                                 "or the GPU build receipt of an arm's kernels.")
    commands = parser.add_subparsers(dest="command", required=True)
    dry = commands.add_parser("dry-run", help="resolve pinned symbols, split pools and contracts; CPU only")
    dry.add_argument("--sglang-src", type=Path, required=True, help="pinned checkout or its python/sglang package")
    dry.add_argument("--config", type=Path, required=True, help="placement.mode=fixed_layers study config")
    arena._geometry_args(dry)
    sizing = dry.add_argument_group("pool sizing (--layers counts eligible sparse layers; --plane-rows is the "
                                    "stock HiSparse plane including its padding page)")
    sizing.add_argument("--dense-layers", type=int, default=0)
    sizing.add_argument("--host-to-device-ratio", type=int, default=2)
    sizing.add_argument("--index-bytes-per-token", type=int, default=0,
                        help="index K(+V) bytes per logical token over all sparse layers; 0 leaves it out")
    build = commands.add_parser("build-receipt", help="GPU only: JIT build of the resident resolver or the split-copy "
                                                      "specializations, and toolchain")
    build.add_argument("--config", type=Path, required=True,
                       help="fixed_layers config without resident_direct or with split_copy")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--allow-cuda-build", action="store_true")
    arena._geometry_args(build)
    args = parser.parse_args(argv)
    try:
        config, _ = sc.read(args.config)
        if not config.uses_fixed_layers:
            raise ValueError("Only placement.mode=fixed_layers splits the pools; native placement uses "
                             "runtime.shared_lru_device")
        reasons = sc.unsupported_reasons(config)
        if reasons:
            raise ValueError("; ".join(reasons))
        geo = arena.describe_geometry(
            tp_size=args.tp_size, layers=range(args.first_layer, args.first_layer + args.layers),
            hot_rows=args.hot_rows, page=args.page, top_k_blocks=args.top_k_blocks, block_size=args.block_size,
            head_dim=args.head_dim, dtype_bytes=args.dtype_bytes, kv_heads_per_rank=args.kv_heads_per_rank,
            slots=args.slots, plane_rows=args.plane_rows)
        if args.command == "dry-run":
            root = args.sglang_src / "python" / "sglang" if (args.sglang_src / "python" / "sglang").is_dir() \
                else args.sglang_src
            report = dry_run(root, config, geo, dense_layers=args.dense_layers, ratio=args.host_to_device_ratio,
                             index_bytes=args.index_bytes_per_token)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.command == "dry-run":
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "pass" else 1
    if not args.allow_cuda_build:
        parser.error("Run only in an approved idle GPU process with --allow-cuda-build")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    print(json.dumps(build_receipt(config, geo, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())