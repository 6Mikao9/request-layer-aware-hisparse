"""SGLang integration of the shared cross-layer hot pool (explicit, process-local, opt-in).

Importing this module imports neither torch nor SGLang. ``install`` runs in a
fresh worker before the KV pool and the coordinator exist and patches only the
pinned symbols below. The stock and all-default configurations install nothing.

* ``HiSparseMHAMainPool._create_buffers_normal``: one contiguous K and one V
  allocation for all planes; per-layer buffers become views, so every stock
  reader and writer keeps its addresses and bytes.
* ``HiSparseCoordinator.__init__``: compatibility gate, TP identity agreement,
  arena tables allocated before graph capture, startup receipt.
* ``alloc_device_buffer`` / ``request_finished``: per-slot generation, owner
  tags, recency lists and occupancy; native tables stay authoritative.
* ``admit_request_direct``: rejected; disaggregated direct admission is unsupported.
* ``swap_in_selected_blocks``: ``rlh_shared_lru::resolve`` replaces the native
  layer-local kernel; output shape, dtype and the -1 sentinel are unchanged.
* ``minimax_sparse.flash_decode_with_gqa_share_sparse``: the consumer reads the
  contiguous view addressed by the resolver's flat indices.
"""
import argparse
import ast
import functools
import hashlib
import importlib
import json
from pathlib import Path
import sys

from . import shared_lru as reference
from . import study_config as sc

PROVENANCE = Path(__file__).with_name("PROVENANCE.json")
KERNEL_SOURCE = Path(__file__).with_name("shared_lru_kernel.cuh")
JIT_NAME = "request_layer_hisparse_shared_lru_v1"
BLOCK_SIZE = 1024
MAX_DEVICE_PLANES = 64
MAX_DEVICE_SELECTION = 2048
COORDINATOR = "srt/managers/hisparse_coordinator.py"
# (pinned file, qualified symbol, required leading parameters or None)
SYMBOLS = (
    (COORDINATOR, "HiSparseCoordinator.__init__", None),
    (COORDINATOR, "HiSparseCoordinator.alloc_device_buffer", ("self", "req")),
    (COORDINATOR, "HiSparseCoordinator.admit_request_direct", ("self", "req")),
    (COORDINATOR, "HiSparseCoordinator.admit_request_into_staging", ("self", "req")),
    (COORDINATOR, "HiSparseCoordinator.request_finished", ("self", "req")),
    (COORDINATOR, "HiSparseCoordinator.map_last_loc_to_buffer", None),
    (COORDINATOR, "HiSparseCoordinator._eager_backup_previous_token", None),
    (COORDINATOR, "HiSparseCoordinator.wait_for_pending_backup", ("self",)),
    (COORDINATOR, "HiSparseCoordinator.swap_in_selected_blocks",
     ("self", "req_pool_indices", "seq_lens", "top_k_blocks", "layer_id", "sparse_block_size")),
    ("srt/mem_cache/hisparse_memory_pool.py", "HiSparseMHAMainPool.__init__", None),
    ("srt/mem_cache/memory_pool.py", "MHATokenToKVPool._create_buffers_normal", ("self",)),
    ("srt/mem_cache/memory_pool.py", "MHATokenToKVPool._kv_buffer_shapes", ("self",)),
    ("srt/layers/attention/minimax_sparse_backend.py", "MiniMaxSparseAttnBackend._hisparse_swap_in_blocks",
     ("self", "forward_batch", "topk_idx", "layer_id")),
    ("srt/layers/attention/minimax_sparse_ops/minimax_sparse.py", "minimax_sparse_decode", None),
    ("kernels/ops/attention/minimax_sparse/decode/topk_sparse.py", "flash_decode_with_gqa_share_sparse", None),
    ("kernels/ops/kvcache/hisparse.py", "load_blocks_to_device_buffer_mha", None),
)
CONSUMER_MODULE = "sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse"
CONSUMER = "flash_decode_with_gqa_share_sparse"
_CREATED_POOLS = 0  # contiguous pools built in this process (blocks uninstall)
def pinned_digests():
    manifest = json.loads(PROVENANCE.read_text())
    return {**manifest["source_sha256"], **manifest["runtime_dependency_sha256"],
            **manifest["shared_lru_dependency_sha256"]}


def verify_digests(root):
    """Map every pinned SGLang input to ok, mismatch or missing."""
    status = {}
    for rel, expected in sorted(pinned_digests().items()):
        path = Path(root) / rel
        if not path.is_file():
            status[rel] = "missing"
        else:
            status[rel] = "ok" if hashlib.sha256(path.read_bytes()).hexdigest() == expected else "mismatch"
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


def resolve_symbols(root):
    """AST check of every patched or relied-upon symbol; never imports SGLang."""
    trees, results = {}, []
    for rel, symbol, params in SYMBOLS:
        path = Path(root) / rel
        if rel not in trees:
            trees[rel] = ast.parse(path.read_bytes(), filename=str(path)) if path.is_file() else None
        node = None if trees[rel] is None else _find(trees[rel], symbol)
        status = "missing" if node is None else "ok"
        if node is not None and params is not None:
            names = tuple(a.arg for a in node.args.args)[:len(params)]
            if names != params:
                status = f"signature {list(names)}"
        results.append(dict(file=rel, symbol=symbol, status=status))
    consumer = trees.get("srt/layers/attention/minimax_sparse_ops/minimax_sparse.py")
    kernel = trees.get("kernels/ops/attention/minimax_sparse/decode/topk_sparse.py")
    calls = [] if consumer is None else [n for n in ast.walk(consumer) if isinstance(n, ast.Call)
                                         and getattr(n.func, "id", None) == CONSUMER]
    imported = consumer is not None and any(
        isinstance(n, ast.ImportFrom) and any(a.name == CONSUMER for a in n.names) for n in consumer.body)
    keyword = bool(calls) and all(not c.args and {"k_cache", "v_cache", "hisparse_slots"} <= {k.arg for k in c.keywords}
                                  for c in calls)
    target = None if kernel is None else _find(kernel, CONSUMER)
    accepts = target is not None and "hisparse_slots" in [a.arg for a in target.args.args + target.args.kwonlyargs]
    results.append(dict(file="srt/layers/attention/minimax_sparse_ops/minimax_sparse.py",
                        symbol=f"{CONSUMER} call sites",
                        status="ok" if imported and keyword and accepts else
                        f"module_import={imported} keyword_calls={keyword} accepts_hisparse_slots={accepts}"))
    return results
def describe_geometry(*, tp_size, layers, hot_rows, page, top_k_blocks, block_size, head_dim, dtype_bytes,
                      kv_heads_per_rank, slots, plane_rows):
    """Rank-local geometry for planning; a worker re-derives every value from its real pool."""
    if type(tp_size) is not int or tp_size not in (4, 8):
        raise ValueError("Plan TP4 or TP8 explicitly")
    if kv_heads_per_rank != 1:
        raise ValueError("The pinned MiniMax HiSparse coordinator requires one KV head per TP rank")
    item = kv_heads_per_rank * head_dim * dtype_bytes
    return dict(tp_size=tp_size, layers=list(layers), hot_rows=hot_rows, page=page, top_k_blocks=top_k_blocks,
                block_size=block_size, selected=top_k_blocks * block_size, kv_heads_per_rank=kv_heads_per_rank,
                head_dim=head_dim, dtype_bytes=dtype_bytes, item_size_bytes=item, kv_row_bytes=2 * item,
                slots=slots, plane_rows=plane_rows)


def validate_device_geometry(planes, selected):
    if not 0 < planes <= MAX_DEVICE_PLANES:
        raise ValueError(f"The SM80 arena supports 1–{MAX_DEVICE_PLANES} planes")
    if not 0 < selected <= MAX_DEVICE_SELECTION or selected & (selected - 1):
        raise ValueError(f"The SM80 selection width must be a power of two up to {MAX_DEVICE_SELECTION}")


def kernel_variants(config, plan, top_k_blocks, block_size):
    """JIT template arguments a worker will build for this arm (one or two variants)."""
    pool = config.hot_pool
    common = (BLOCK_SIZE, top_k_blocks, block_size, pool.eviction == "fifo", pool.quota == "soft_floor")
    bypass = sorted({plane in plan.bypass for plane in range(len(plan.layers))})
    return [common + (b, config.diagnostics == "counters") for b in bypass]


def tensor_contracts(geo, plan):
    L, H, page, rows = len(geo["layers"]), geo["hot_rows"], geo["page"], geo["plane_rows"]
    return {
        "top_k_blocks": f"[B, {geo['top_k_blocks']}] int32 selector blocks; -1 is invalid",
        "req_pool_indices, seq_lens": "[B] int32 or int64 graph inputs; num_real_reqs [1] int32",
        "row_token": f"native req_device_buffer_tokens [{L}, slots, {H + page}] int32; "
                     f"row r -> [r // {H}, slot, r % {H}]",
        "row_owner": f"[slots, {plan.rows}] int32: generation << 16 | plane, -1 free",
        "order": f"[slots, {plan.rows}] int32 recency regions {list(plan.domains)}",
        "occupancy, floors, generation": f"[slots, {L}] int32, [{L}] int32, [slots] int32",
        "device_locs": f"req_device_buffer_token_locs[plane] [slots, {H + page}] int32; column {H} is the tail",
        "host_k, host_v": f"mem_pool_host.k_buffer[plane] / v_buffer[plane], {geo['item_size_bytes']} B per row",
        "arena_k, arena_v": f"contiguous [{L}, {rows}, {geo['kv_heads_per_rank']}, {geo['head_dim']}]; "
                            f"consumer view [{L * rows}, heads, dim]",
        "output": f"top_k_device_locs_buffer[:B, :{geo['selected']}] int32 = plane * {rows} + loc; -1 invalid",
    }


def dry_run(root, config, geo):
    """CPU-only adapter resolution against a pinned source tree: no torch, model or GPU."""
    plan = reference.plan_arena(config, tuple(geo["layers"]), geo["hot_rows"], geo["selected"])
    if len(plan.layers) * geo["plane_rows"] > 2**31 - 1:
        raise ValueError("Flat consumer index would exceed int32")
    validate_device_geometry(len(plan.layers), geo["selected"])
    digests, symbols = verify_digests(root), resolve_symbols(root)
    ledger = reference.byte_ledger(layers=geo["layers"], hot_rows=geo["hot_rows"], page=geo["page"],
                                   item_bytes=geo["item_size_bytes"], slots=geo["slots"], pool_rows=geo["plane_rows"],
                                   plan=plan, counters=config.diagnostics == "counters")
    ok = all(v == "ok" for v in digests.values()) and all(s["status"] == "ok" for s in symbols)
    return dict(status="pass" if ok else "fail", config_sha256=config.digest, resolved_config=config.as_dict(),
                capabilities=sc.capabilities(config), digests=digests, symbols=symbols,
                implementation=dict(jit_module=JIT_NAME, kernel_source=KERNEL_SOURCE.name,
                                    kernel_source_sha256=hashlib.sha256(KERNEL_SOURCE.read_bytes()).hexdigest(),
                                    template_args=[list(v) for v in kernel_variants(
                                        config, plan, geo["top_k_blocks"], geo["block_size"])],
                                    consumer=f"{CONSUMER_MODULE}.{CONSUMER}"),
                plan=dict(rows=plan.rows, domains=list(plan.domains), scratch=list(plan.scratch),
                          bypass_planes=list(plan.bypass), floors=list(plan.floors)),
                geometry=geo, tensor_contracts=tensor_contracts(geo, plan), budget_units=ledger,
                pending_hardware=["sm80_jit_build", "cuda_graph_capture_replay", "physical_memory_receipt",
                                  "serving_tpot_throughput"])
def contiguous_create_buffers(self):
    """Replacement for ``HiSparseMHAMainPool._create_buffers_normal``: same shapes, one allocation."""
    global _CREATED_POOLS
    import torch
    from contextlib import nullcontext
    from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
    if self.kv_cache_layout != "nhd" or getattr(self, "use_hnd", False):
        raise RuntimeError("The shared arena requires plain NHD main-KV planes")
    _CREATED_POOLS += 1
    k_shape, v_shape = self._kv_buffer_shapes()
    with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
        with (torch.cuda.use_mem_pool(self.custom_mem_pool) if self.enable_custom_mem_pool else nullcontext()):
            k_all = torch.zeros((self.layer_num, *k_shape), dtype=self.store_dtype, device=self.device)
            v_all = torch.zeros((self.layer_num, *v_shape), dtype=self.store_dtype, device=self.device)
    self.k_buffer = list(k_all.unbind(0))
    self.v_buffer = list(v_all.unbind(0))
    self._rlh_contiguous_kv = (k_all, v_all)


def verify_planes(pool):
    """Prove every per-layer buffer is the expected view of one allocation before using one base."""
    pair = getattr(pool, "_rlh_contiguous_kv", None)
    if pair is None:
        raise RuntimeError("Main-KV planes were allocated before install(); start a fresh worker")
    for name, whole, planes in (("K", pair[0], pool.k_buffer), ("V", pair[1], pool.v_buffer)):
        stride = whole.stride(0) * whole.element_size()
        if not whole.is_contiguous() or len(planes) != whole.shape[0]:
            raise RuntimeError(f"{name} arena is not one contiguous allocation of every plane")
        for index, plane in enumerate(planes):
            if (plane.data_ptr() != whole.data_ptr() + index * stride or tuple(plane.shape) != tuple(whole.shape[1:])
                    or plane.stride() != whole.stride()[1:] or plane.dtype != whole.dtype
                    or plane.untyped_storage().data_ptr() != whole.untyped_storage().data_ptr()):
                raise RuntimeError(f"{name} plane {index} is not a view of the contiguous arena")
    return pair


def check_coordinator(coordinator, *, torch, context):
    """Bind-time compatibility gate; runs before any experimental allocation."""
    pool, errors = coordinator.mem_pool_device, []
    if not coordinator.is_m3_hisparse or getattr(coordinator, "is_dsv4_hisparse", False):
        errors.append("the shared arena supports the MiniMax M3 MHA HiSparse path only")
    if coordinator.skip_io or coordinator.spec_swap.enabled or coordinator.enable_prefetch:
        errors.append("skip-IO probes, speculative decoding and shared-index prefetch are unsupported")
    if (pool.head_num != 1 or pool.is_quantized_kv_cache or pool.dtype != torch.bfloat16
            or pool.store_dtype != pool.dtype or pool.post_capture_active):
        errors.append("main KV must be one BF16 head per rank, unquantized, without post-capture backing")
    layers = tuple(range(pool.start_layer, pool.start_layer + pool.layer_num))
    if tuple(pool.model_layer_ids) != layers or pool.end_layer - pool.start_layer != pool.layer_num:
        errors.append("host-backed layers must be one contiguous physical range")
    H, page = coordinator.device_buffer_size, pool.page_size
    tokens, locs = coordinator.req_device_buffer_tokens, coordinator.req_device_buffer_token_locs
    if (coordinator.padded_buffer_size != H + page or tuple(tokens.shape) != tuple(locs.shape)
            or tuple(tokens.shape[::2]) != (pool.layer_num, H + page) or tokens.dtype != torch.int32
            or locs.dtype != torch.int32 or coordinator.top_k > H
            or coordinator.top_k_device_locs_buffer.dtype != torch.int32):
        errors.append("coordinator tables differ from the pinned MiniMax layout")
    try:
        validate_device_geometry(len(layers), coordinator.top_k)
    except ValueError as error:
        errors.append(str(error))
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
    if torch.cuda.get_device_capability() != (8, 0):
        errors.append("the resolver is built and validated for sm_80 (A100) only")
    if errors:
        raise RuntimeError("Shared arena incompatible: " + "; ".join(errors))
    return layers
@functools.cache
def _jit_module(block_size, num_top_k, sparse_block_size, fifo, soft_floor, bypass, counters):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH
    args = make_cpp_args(block_size, num_top_k, sparse_block_size, fifo, soft_floor, bypass, counters)
    return load_jit(JIT_NAME, *args, cuda_files=[str(KERNEL_SOURCE)],
                    cuda_wrappers=[("resolve", f"rlh_shared_lru::resolve<{args}>")],
                    extra_include_paths=[str(KERNEL_PATH / "csrc")])


class ArenaRuntime:
    """Device tables and dispatch of one bound coordinator (one TP rank)."""

    def __init__(self, coordinator, config, config_sha256, *, torch):
        pool = coordinator.mem_pool_device
        self.torch, self.config, self.config_sha256 = torch, config, config_sha256
        self.k_all, self.v_all = verify_planes(pool)
        self.layers = tuple(range(pool.start_layer, pool.start_layer + pool.layer_num))
        self.start_layer, self.hot_rows, self.page = pool.start_layer, coordinator.device_buffer_size, pool.page_size
        self.plane_rows, self.item_size_bytes = self.k_all.shape[1], coordinator.item_size_bytes
        self.plan = reference.plan_arena(config, self.layers, self.hot_rows, coordinator.top_k)
        planes, rows = len(self.layers), self.plan.rows
        if planes * self.plane_rows > 2**31 - 1:
            raise RuntimeError("Flat consumer index would exceed int32")
        validate_device_geometry(planes, self.plan.selected)
        if self.k_all.stride(1) * self.k_all.element_size() != self.item_size_bytes:
            raise RuntimeError("Main-KV row stride differs from the coordinator item size")
        tokens = coordinator.req_device_buffer_tokens
        slots, device, i32 = tokens.shape[1], tokens.device, torch.int32
        owners = self.plan.owner_template()
        self._owner_base = torch.tensor([-1 if o is None else o for o in owners], dtype=i32, device=device)
        self._owned = torch.tensor([0 if o is None else 1 for o in owners], dtype=i32, device=device)
        self._order_template = torch.tensor(self.plan.order_template(), dtype=i32, device=device)
        self._occupancy_template = torch.tensor([owners.count(p) for p in range(planes)], dtype=i32, device=device)
        self.row_owner = torch.full((slots, rows), reference.FREE_TAG, dtype=i32, device=device)
        self.order = self._order_template.repeat(slots, 1)
        self.occupancy = torch.zeros((slots, planes), dtype=i32, device=device)
        self.floors = torch.tensor(self.plan.floors, dtype=i32, device=device)
        self.generation = torch.zeros(slots, dtype=i32, device=device)
        self.status = torch.zeros(1, dtype=i32, device=device)
        width = len(reference.COUNTER_FIELDS) if config.diagnostics == "counters" else 0
        self.counters = torch.zeros((planes if width else 0, width), dtype=torch.int64, device=device)
        self.flat_k = self.k_all.view(planes * self.plane_rows, *self.k_all.shape[2:])
        self.flat_v = self.v_all.view(planes * self.plane_rows, *self.v_all.shape[2:])
        self._allocations = [0] * slots

    def tables(self):
        return dict(row_owner=self.row_owner, order=self.order, occupancy=self.occupancy, floors=self.floors,
                    generation=self.generation, status=self.status, counters=self.counters,
                    owner_template=self._owner_base, owned_template=self._owned,
                    order_template=self._order_template, occupancy_template=self._occupancy_template)

    def signature(self):
        kernel = hashlib.sha256(KERNEL_SOURCE.read_bytes()).hexdigest()
        return dict(layers=list(self.layers), hot_rows=self.hot_rows, page=self.page, plane_rows=self.plane_rows,
                    item_size_bytes=self.item_size_bytes, plan_rows=self.plan.rows, domains=list(self.plan.domains),
                    scratch=list(self.plan.scratch), kernel_sha256=kernel)

    def init_slot(self, rid):
        """New request generation in slot ``rid``; runs eagerly after the native buffer allocation."""
        generation = reference.device_generation(self._allocations[rid])
        self._allocations[rid] += 1
        owner = self.row_owner[rid]
        owner.copy_(self._owner_base)
        owner.add_(self._owned, alpha=generation << 16)
        self.order[rid].copy_(self._order_template)
        self.occupancy[rid].copy_(self._occupancy_template)
        self.generation[rid] = generation
        return generation

    def reset_slot(self, rid):
        self.row_owner[rid].fill_(reference.FREE_TAG)
    def resolve(self, coordinator, req_pool_indices, seq_lens, top_k_blocks, layer_id, sparse_block_size):
        """Graph-safe swap-in: views, constants and one kernel launch; no sync, no allocation."""
        torch, plan = self.torch, self.plan
        num_blocks = top_k_blocks.size(1)
        if num_blocks * sparse_block_size != plan.selected or top_k_blocks.dtype != torch.int32:
            raise ValueError("Selection width/dtype differs from the bound arena plan")
        plane = layer_id - self.start_layer
        if not 0 <= plane < len(self.layers):
            raise ValueError(f"Layer {layer_id} is not a host-backed plane of this rank")
        out = coordinator.top_k_device_locs_buffer[:req_pool_indices.size(0), :plan.selected]
        offset, size = plan.domain(plane)
        pool = self.config.hot_pool
        module = _jit_module(BLOCK_SIZE, num_blocks, sparse_block_size, pool.eviction == "fifo",
                             pool.quota == "soft_floor", plane in plan.bypass, self.config.diagnostics == "counters")
        host = coordinator.mem_pool_host
        args = (top_k_blocks, req_pool_indices, seq_lens, coordinator.num_real_reqs,
                coordinator.req_device_buffer_tokens, self.row_owner, self.order, self.occupancy, self.floors,
                self.generation, coordinator.req_device_buffer_token_locs[plane], coordinator.req_to_host_pool,
                host.k_buffer[plane], host.v_buffer[plane], self.k_all, self.v_all, out, self.status, self.counters,
                plane, len(self.layers), self.hot_rows, self.plane_rows, offset, size,
                plan.scratch[0], plan.scratch[1], self.item_size_bytes)
        if self.config.diagnostics == "nsys":
            with torch.cuda.nvtx.range(f"rlh_shared_lru.resolve L{layer_id}"):
                module.resolve(*args)
        else:
            module.resolve(*args)
        return out

    def consumer_views(self, k_cache, v_cache):
        """Swap a layer plane for the contiguous view after checking it is that exact plane."""
        for name, whole, cache in (("K", self.k_all, k_cache), ("V", self.v_all, v_cache)):
            stride = whole.stride(0) * whole.element_size()
            offset = cache.data_ptr() - whole.data_ptr()
            if (offset < 0 or offset % stride or offset // stride >= whole.shape[0]
                    or tuple(cache.shape) != tuple(whole.shape[1:]) or cache.stride() != whole.stride()[1:]
                    or cache.dtype != whole.dtype):
                raise RuntimeError(f"Consumer {name} cache is not a plane of the bound arena")
        return self.flat_k, self.flat_v

    def counter_snapshot(self):
        """Diagnostics only (synchronizes): per-layer counters and sticky status bits."""
        values = self.counters.cpu().tolist()
        return dict(event="hisparse_shared_lru_counters", status=int(self.status.item()),
                    layers={str(layer): dict(zip(reference.COUNTER_FIELDS, row))
                            for layer, row in zip(self.layers, values)})

    def receipt(self, coordinator, rank, tp_size):
        torch = self.torch

        def size(tensor):
            return tensor.numel() * tensor.element_size()

        host = coordinator.mem_pool_host
        native = dict(req_device_buffer_tokens=size(coordinator.req_device_buffer_tokens),
                      req_device_buffer_token_locs=size(coordinator.req_device_buffer_token_locs),
                      lru_slots_unused_in_arena=size(coordinator.lru_slots),
                      top_k_device_locs_buffer=size(coordinator.top_k_device_locs_buffer))
        memory = None
        if torch.cuda.is_available():
            memory = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                          max_allocated=torch.cuda.max_memory_allocated())
        return dict(event="hisparse_shared_lru_bound", rank=rank, tp_size=tp_size,
                    config_file_sha256=self.config_sha256, config_sha256=self.config.digest,
                    resolved_config=self.config.as_dict(), capabilities=sc.capabilities(self.config),
                    signature=self.signature(), plan=dict(bypass_planes=list(self.plan.bypass),
                                                          floors=list(self.plan.floors)),
                    bytes=dict(main_kv_k=size(self.k_all), main_kv_v=size(self.v_all),
                               arena_tables={k: size(v) for k, v in self.tables().items()}, native_tables=native,
                               host_pinned=sum(size(t) for t in (*host.k_buffer, *host.v_buffer))),
                    device_memory_at_bind=memory,
                    unavailable=dict(weights="not attributed at bind", graph_workspace="after capture",
                                     prefill_overlap="measure on the GPU run", fragmentation="allocator stats"),
                    pending_hardware=["cuda_graph_capture_replay", "serving_tpot_throughput"])
_STATE = None
_PATCHED = ("__init__", "alloc_device_buffer", "request_finished", "admit_request_direct", "swap_in_selected_blocks")


def _runtime(coordinator):
    runtime = getattr(coordinator, "_request_layer_hisparse_arena", None)
    if runtime is None:
        raise RuntimeError("Coordinator was not bound to the shared arena; install before construction")
    return runtime


def install(config, config_sha256):
    """Patch pinned symbols in this worker before pool/coordinator construction.

    Returns False and patches nothing for stock layouts. A process keeps one
    frozen configuration; a different one requires a new process.
    """
    global _STATE
    reasons = sc.unsupported_reasons(config)
    if reasons:
        raise RuntimeError("; ".join(reasons))
    if not config.uses_arena:
        return False
    if _STATE is not None:
        if _STATE["config"].digest != config.digest:
            raise RuntimeError("The arena configuration is frozen for this worker")
        return True
    import torch
    import torch.distributed as dist
    import sglang
    bad = {k: v for k, v in verify_digests(Path(sglang.__file__).resolve().parent).items() if v != "ok"}
    if bad:
        raise RuntimeError(f"Pinned SGLang source mismatch: {bad}")
    coordinator_cls = importlib.import_module("sglang.srt.managers.hisparse_coordinator").HiSparseCoordinator
    pool_cls = importlib.import_module("sglang.srt.mem_cache.hisparse_memory_pool").HiSparseMHAMainPool
    consumer_module = importlib.import_module(CONSUMER_MODULE)
    context = importlib.import_module("sglang.srt.runtime_context")
    originals = {name: getattr(coordinator_cls, name) for name in _PATCHED}
    consumer = getattr(consumer_module, CONSUMER)
    if (any(getattr(fn, "__module__", None) != coordinator_cls.__module__ for fn in originals.values())
            or "_create_buffers_normal" in pool_cls.__dict__
            or consumer.__module__ != "sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse"):
        raise RuntimeError("Another hook already replaced a pinned symbol; refusing to stack hooks")
    state = dict(config=config, runtime=None, originals=originals, consumer=consumer, bound=False,
                 classes=(coordinator_cls, pool_cls, consumer_module))

    def init(self, *args, **kwargs):
        originals["__init__"](self, *args, **kwargs)
        if state["runtime"] is not None:
            raise RuntimeError("One HiSparse coordinator per worker process")
        check_coordinator(self, torch=torch, context=context)
        state["bound"] = True
        runtime = ArenaRuntime(self, config, config_sha256, torch=torch)
        identity = hashlib.sha256(json.dumps(runtime.signature(), sort_keys=True).encode()
                                  + config.digest.encode()).hexdigest()
        hashes = [None] * self.tp_world_size
        dist.all_gather_object(hashes, identity, group=self.tp_group)
        if hashes != [identity] * self.tp_world_size:
            raise RuntimeError("TP ranks disagree on the arena configuration or geometry")
        self._request_layer_hisparse_arena = state["runtime"] = runtime
        print(json.dumps(runtime.receipt(self, dist.get_rank(), self.tp_world_size), sort_keys=True), flush=True)
        if config.diagnostics == "counters":
            import atexit
            atexit.register(lambda: print(json.dumps(runtime.counter_snapshot(), sort_keys=True), flush=True))

    def alloc_device_buffer(self, req):
        originals["alloc_device_buffer"](self, req)
        _runtime(self).init_slot(req.kv.req_pool_idx)

    def request_finished(self, req):
        originals["request_finished"](self, req)
        _runtime(self).reset_slot(req.kv.req_pool_idx)

    def admit_request_direct(self, req):
        raise RuntimeError("Direct (disaggregated) admission is not supported by the shared arena")

    def swap_in_selected_blocks(self, req_pool_indices, seq_lens, top_k_blocks, layer_id, sparse_block_size):
        return _runtime(self).resolve(self, req_pool_indices, seq_lens, top_k_blocks, layer_id, sparse_block_size)

    @functools.wraps(consumer)
    def consumer_hook(*args, **kwargs):
        runtime = state["runtime"]
        if runtime is not None and kwargs.get("hisparse_slots") is not None:
            if args:
                raise RuntimeError("The arena consumer expects the pinned keyword-only call")
            kwargs["k_cache"], kwargs["v_cache"] = runtime.consumer_views(kwargs["k_cache"], kwargs["v_cache"])
        return consumer(*args, **kwargs)

    hooks = dict(__init__=init, alloc_device_buffer=alloc_device_buffer, request_finished=request_finished,
                 admit_request_direct=admit_request_direct, swap_in_selected_blocks=swap_in_selected_blocks)
    for name, hook in hooks.items():
        setattr(coordinator_cls, name, hook)
    pool_cls._create_buffers_normal = contiguous_create_buffers
    setattr(consumer_module, CONSUMER, consumer_hook)
    state["hooks"] = dict(hooks, consumer=consumer_hook)
    _STATE = state
    return True
def uninstall():
    """Restore pinned symbols only if no pool or coordinator was built under the hooks."""
    global _STATE
    if _STATE is None:
        return False
    if _STATE["bound"] or _CREATED_POOLS:
        raise RuntimeError("Arena state exists in this process; start a new process to restore stock behavior")
    coordinator_cls, pool_cls, consumer_module = _STATE["classes"]
    hooks = _STATE["hooks"]
    if (any(getattr(coordinator_cls, name) is not hooks[name] for name in _PATCHED)
            or getattr(consumer_module, CONSUMER) is not hooks["consumer"]
            or pool_cls.__dict__.get("_create_buffers_normal") is not contiguous_create_buffers):
        raise RuntimeError("Another hook replaced these symbols; refusing to overwrite it")
    for name, fn in _STATE["originals"].items():
        setattr(coordinator_cls, name, fn)
    del pool_cls._create_buffers_normal
    setattr(consumer_module, CONSUMER, _STATE["consumer"])
    _STATE = None
    return True


def build_receipt(config, geo, output):
    """GPU-only: JIT-build this arm's variants and record toolchain and artifact identity."""
    import platform
    import shutil
    import subprocess
    import torch
    import sglang
    from sglang.kernels.jit.utils.compile.cache import cache_root
    bad = {k: v for k, v in verify_digests(Path(sglang.__file__).resolve().parent).items() if v != "ok"}
    if bad or torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError(f"Build receipt requires the pinned source on sm_80; mismatches: {bad}")
    plan = reference.plan_arena(config, tuple(geo["layers"]), geo["hot_rows"], geo["selected"])
    variants = kernel_variants(config, plan, geo["top_k_blocks"], geo["block_size"])
    for variant in variants:
        _jit_module(*variant)

    def run(*command):
        if shutil.which(command[0]) is None:
            return None
        return subprocess.run(command, capture_output=True, text=True, check=False).stdout.strip()

    artifacts = []
    for path in sorted(cache_root().rglob(f"*{JIT_NAME}*.so")):
        artifacts.append(dict(name=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              cuobjdump_list_elf=run("cuobjdump", "--list-elf", str(path))))
    receipt = dict(event="hisparse_shared_lru_build", config_sha256=config.digest, template_args=variants,
                   kernel_source_sha256=hashlib.sha256(KERNEL_SOURCE.read_bytes()).hexdigest(),
                   python=platform.python_version(), torch=torch.__version__, torch_cuda=torch.version.cuda,
                   cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI), nccl=list(torch.cuda.nccl.version()),
                   device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                   nvcc=run("nvcc", "--version"), host_cxx=run("c++", "--version"), artifacts=artifacts)
    with Path(output).open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    return receipt
def _geometry_args(parser):
    group = parser.add_argument_group("planning geometry (defaults: MiniMax-M3.1-Flash TP8 example; a worker "
                                      "re-derives every value from its pool)")
    group.add_argument("--tp-size", type=int, default=8)
    group.add_argument("--first-layer", type=int, default=0)
    group.add_argument("--layers", type=int, default=60, help="host-backed physical KV layers on the rank")
    group.add_argument("--hot-rows", type=int, default=4096)
    group.add_argument("--page", type=int, default=128)
    group.add_argument("--top-k-blocks", type=int, default=16)
    group.add_argument("--block-size", type=int, default=128)
    group.add_argument("--head-dim", type=int, default=128)
    group.add_argument("--dtype-bytes", type=int, default=2)
    group.add_argument("--kv-heads-per-rank", type=int, default=1)
    group.add_argument("--slots", type=int, default=128, help="request slots (req_to_token rows)")
    group.add_argument("--plane-rows", type=int, default=396_672, help="rows per main-KV plane incl. padding page")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Shared-arena adapter: CPU dry-run or GPU build receipt.")
    commands = parser.add_subparsers(dest="command", required=True)
    dry = commands.add_parser("dry-run", help="resolve pinned symbols, config and contracts; CPU only")
    dry.add_argument("--sglang-src", type=Path, required=True, help="pinned checkout or its python/sglang package")
    dry.add_argument("--config", type=Path, required=True)
    _geometry_args(dry)
    build = commands.add_parser("build-receipt", help="GPU only: JIT build and toolchain receipt")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--allow-cuda-build", action="store_true")
    _geometry_args(build)
    args = parser.parse_args(argv)
    try:
        config, _ = sc.read(args.config)
        if not config.uses_arena:
            raise ValueError("The stock layout runs the native kernel; there is no arena adapter to resolve")
        reasons = sc.unsupported_reasons(config)
        if reasons:
            raise ValueError("; ".join(reasons))
        geo = describe_geometry(
            tp_size=args.tp_size, layers=range(args.first_layer, args.first_layer + args.layers),
            hot_rows=args.hot_rows, page=args.page, top_k_blocks=args.top_k_blocks, block_size=args.block_size,
            head_dim=args.head_dim, dtype_bytes=args.dtype_bytes, kv_heads_per_rank=args.kv_heads_per_rank,
            slots=args.slots, plane_rows=args.plane_rows)
        if args.command == "dry-run":
            root = args.sglang_src / "python" / "sglang" if (args.sglang_src / "python" / "sglang").is_dir() \
                else args.sglang_src
            report = dry_run(root, config, geo)
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
