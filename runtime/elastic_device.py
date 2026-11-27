"""SGLang integration of elastic per-layer HiSparse (issue #13): explicit, process-local, opt-in.

``install(config, digest)`` runs in a fresh worker before the KV pools exist and patches the pinned SGLang
e7399a46 symbols in ``SYMBOLS``. The HiSparse device pool's per-layer buffers become CUDA-VMM reservations with a
hot region (the stock device rows) and a resident region (one row per logical location); a physical chunk pool of
the bytes stock would have allocated backs whichever regions the current offload level needs. Logical capacity is
an allocator ceiling that grows as layers leave the GPU and shrinks (after compaction) when they return.

* resident layer: stores and reads go to its logical rows, like the stock non-HiSparse path; no swap-in.
* offloaded layer: the stock HiSparse path over its hot rows.
* steady mode: one decode CUDA Graph set per level (captured at start-up or on first use); a transition runs
  between batches, stops decode, copies, and selects another set.
* smooth mode: one graph set; every managed layer selects through its row of a device route table (the fused
  Triton route of ``elastic_route``, or its torch statement) around the stock swap-in, so a transition only
  rewrites the table; backfill copies are paced across steps and offloaded layers keep hitting stale rows.

Every managed layer writes its decode tokens through to host whichever mode wrote them, so offloading moves no
bytes to host. The controller (``elastic_policy``) runs on TP rank 0 every ``interval_steps`` decode batches and
its decision is broadcast on the coordinator's Gloo group, so all ranks change level at the same batch boundary.
Models: MiniMax-M3 (sparse MHA layers) and the DSA family (GLM-5 / 5.1 / 5.2, DeepSeek-V3.2, with shared-index
groups); DeepSeek-V4 C4 layers are refused at install for now.
"""
from contextlib import contextmanager, nullcontext
import hashlib
import importlib
import json
from pathlib import Path
import time

from . import elastic_config as ec
from . import elastic_kv as ekv
from . import elastic_memory as em
from . import elastic_policy as ep
from .dynamic_offload_device import PROVENANCE, _find, _patch, _restore, emit, pinned_digests

REPO_ROOT = Path(__file__).resolve().parents[1]

COORDINATOR = "srt/managers/hisparse_coordinator.py"
MEMORY_POOL = "srt/mem_cache/memory_pool.py"
HISPARSE_POOL = "srt/mem_cache/hisparse_memory_pool.py"
ALLOCATOR = "srt/mem_cache/allocator/hisparse.py"
SCHEDULER = "srt/managers/scheduler.py"
DISAGG = "srt/disaggregation/decode.py"
DECODE_GRAPH = "srt/model_executor/runner/decode_cuda_graph_runner.py"
FULL_GRAPH = "srt/model_executor/runner_backend/full_cuda_graph_backend.py"
MINIMAX_BACKEND = "srt/layers/attention/minimax_sparse_backend.py"
MINIMAX_OPS = "srt/layers/attention/minimax_sparse_ops/minimax_sparse.py"
DSA_BACKEND = "srt/layers/attention/dsa_backend.py"
INTERNAL_STATE = "srt/managers/scheduler_components/internal_state.py"
LEVEL_KEY = "hisparse_elastic_level"
PREFILL = "minimax_sparse_prefill"
_COORD = ("__init__", "_eager_backup_previous_token", "admit_request_into_staging", "admit_request_direct",
          "_preload_to_device_buffer", "alloc_device_buffer", "request_finished", "map_last_loc_to_buffer",
          "swap_in_selected_pages")
SYMBOLS = {
    "*": (
        *((COORDINATOR, f"HiSparseCoordinator.{name}", None) for name in _COORD),
        (COORDINATOR, "HiSparseCoordinator._run_swap_in_kernel", ("self", "req_pool_indices",
                                                                   "compressed_seq_lens", "top_k_result",
                                                                   "layer_id", "record_plan")),
        (ALLOCATOR, "HiSparseTokenToKVPoolAllocator.__init__", ("self", "size", "page_size", "dtype", "device",
                                                                "kvcache", "need_sort", "host_to_device_ratio")),
        ("srt/mem_cache/allocator/paged.py", "PagedTokenToKVPoolAllocator.clear", ("self",)),
        (SCHEDULER, "Scheduler.get_next_batch_to_run", ("self", "running_batch", "last_batch")),
        (DISAGG, "SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run",
         ("self", "running_batch")),
        (DECODE_GRAPH, "DecodeCudaGraphRunner.capture", ("self",)),
        (FULL_GRAPH, "FullCudaGraphBackend.capture_one", ("self", "shape_key", "forward_fn")),
        (FULL_GRAPH, "FullCudaGraphBackend.can_run", ("self", "forward_batch", "shape_key")),
        (FULL_GRAPH, "FullCudaGraphBackend.replay", ("self", "shape_key", "static_forward_batch")),
        (INTERNAL_STATE, "SchedulerInternalStateController._update_handlers", ("self",)),
        (INTERNAL_STATE, "_UpdateHandler.__init__", ("self", "validate", "apply")),
    ),
    "minimax_m3": (
        (MEMORY_POOL, "MiniMaxSparseKVPool.__init__", None),
        (MEMORY_POOL, "MHATokenToKVPool._create_buffers_normal", ("self",)),
        *((MEMORY_POOL, f"MiniMaxSparseKVPool.{name}", ("self", "layer_id"))
          for name in ("get_key_buffer", "get_value_buffer", "get_kv_buffer")),
        (MEMORY_POOL, "MiniMaxSparseKVPool.set_kv_buffer", ("self", "layer", "loc", "cache_k", "cache_v")),
        (MEMORY_POOL, "_set_kv_buffer_impl", ("k", "v", "k_cache", "v_cache", "indices")),
        (MEMORY_POOL, "unwrap_write_loc", ("loc_info",)),
        (HISPARSE_POOL, "HiSparseMHAMainPool.__init__", None),
        (MINIMAX_BACKEND, "MiniMaxSparseAttnBackend._hisparse_swap_in_blocks",
         ("self", "forward_batch", "topk_idx", "layer_id")),
        (MINIMAX_BACKEND, "MiniMaxSparseAttnBackend.forward_decode", None),
        (MINIMAX_OPS, PREFILL, ("q", "k_cache", "v_cache")),
        (COORDINATOR, "HiSparseCoordinator.swap_in_selected_blocks",
         ("self", "req_pool_indices", "seq_lens", "top_k_blocks", "layer_id", "sparse_block_size")),
    ),
    "dsa": (
        (MEMORY_POOL, "MLATokenToKVPool._create_buffers", ("self",)),
        (MEMORY_POOL, "MLATokenToKVPool._write_mla_kv_buffer",
         ("self", "dst_buffer", "loc", "cache_k_nope", "cache_k_rope")),
        (HISPARSE_POOL, "HiSparseDSATokenToKVPool.__init__", None),
        (HISPARSE_POOL, "HiSparseDSATokenToKVPool.set_mla_kv_buffer",
         ("self", "layer", "loc_info", "cache_k_nope", "cache_k_rope")),
        (HISPARSE_POOL, "HiSparseDSATokenToKVPool.translate_loc_to_hisparse_device", ("self", "compressed_indices")),
        (MEMORY_POOL, "MLATokenToKVPool.get_key_buffer", ("self", "layer_id")),
        (MEMORY_POOL, "MLATokenToKVPool.get_value_buffer", ("self", "layer_id")),
        *((DSA_BACKEND, f"DeepseekSparseAttnBackend.{name}", ("self", "q", "k", "v", "layer", "forward_batch"))
          for name in ("forward_extend", "forward_decode")),
        (COORDINATOR, "HiSparseCoordinator._run_copy_only_kernel", ("self", "num_reqs", "skip_layer")),
        ("kernels/ops/kvcache/hisparse.py", "copy_cache_planned_mla",
         ("miss_src", "miss_dst", "miss_count", "num_real_reqs", "host_cache", "device_buffer", "item_size_bytes",
          "num_blocks")),
        ("srt/mem_cache/index_key_cache.py", "IndexKeyCache.move", ("self", "tgt_loc", "src_loc")),
    ),
    "deepseek_v4": (),
}


def _calls(node, attribute):
    import ast
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == attribute]


# (model, file, symbol, contract, predicate over the symbol's AST): stock call sites the hooks rely on.
CONTRACTS = (
    ("minimax_m3", MINIMAX_BACKEND, "MiniMaxSparseAttnBackend.forward_decode",
     "reads main KV through kv_pool.get_kv_buffer and swaps in through self._hisparse_swap_in_blocks",
     lambda n: bool(_calls(n, "get_kv_buffer")) and bool(_calls(n, "_hisparse_swap_in_blocks"))),
    ("dsa", DSA_BACKEND, "DeepseekSparseAttnBackend.forward_decode",
     "takes kv_cache from get_key_buffer before swap_in_selected_pages translates the top-k",
     lambda n: bool(_calls(n, "get_key_buffer")) and bool(_calls(n, "swap_in_selected_pages"))),
    ("*", COORDINATOR, "HiSparseCoordinator.map_last_loc_to_buffer",
     "backs the previous token up through _eager_backup_previous_token before the forward",
     lambda n: bool(_calls(n, "_eager_backup_previous_token"))),
    ("*", COORDINATOR, "HiSparseCoordinator.admit_request_direct",
     "allocates the device buffer, then preloads short requests through _preload_to_device_buffer",
     lambda n: bool(_calls(n, "alloc_device_buffer")) and bool(_calls(n, "_preload_to_device_buffer"))),
)
PENDING_HARDWARE = ("sm80_vmm_map_unmap", "fused_route_self_check", "steady_graph_sets_capture",
                    "tp8_level_broadcast", "pd_decode_admission_fill", "serving_tpot_throughput_goodput",
                    "vllm_hybrid_hisparse_comparison", "nsys_transition_timeline")


def resolve_symbols(root, model):
    """AST check of every patched or relied-upon symbol and stock call contract; never imports SGLang."""
    import ast
    from pathlib import Path
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
        if owner in ("*", model):
            node = node_of(rel, symbol)
            results.append(dict(file=rel, symbol=f"{symbol}: {contract}",
                                status="ok" if node is not None and holds(node) else "contract changed"))
    return results


# ---- worker state ------------------------------------------------------------------------------------------
_STATE = None


def _state():
    if _STATE is None:
        raise RuntimeError("runtime.elastic_device was not installed in this worker")
    return _STATE


class Deferred(Exception):
    """A transition that does not fit yet, raised before it changed any state; the worker keeps serving."""


class Ceiling:
    """Capacity of one stock paged allocator: only pages 1..ceiling are handed out. Pages above it are parked
    (they exist in the virtual reservation but are not backed); ``clear`` keeps the ceiling."""

    def __init__(self, allocator, torch):
        self.a, self.torch = allocator, torch
        self.ceiling = allocator.num_pages
        stock_clear = type(allocator).clear
        ceiling = self

        def clear():
            stock_clear(allocator)
            ceiling._clip()
        allocator.clear = clear

    def _free(self):
        return self.torch.cat((self.a.free_pages, *self.a.staged_pages))

    def _clip(self):
        free = self._free()
        self.a.free_pages = self.torch.sort(free[free <= self.ceiling])[0]
        self.a.staged_pages, self.a.num_staged_pages = [], 0

    def tokens(self):
        return self.ceiling * self.a.page_size

    def used_tokens(self):
        return self.tokens() - self.a.available_size()

    def peak_tokens(self):
        """One past the highest live location (a device reduction and one host sync)."""
        t = self.torch
        free = self._free()
        live = t.ones(self.ceiling + 1, dtype=t.bool, device=free.device)
        live[0] = False
        live[free[free <= self.ceiling]] = False
        top = t.nonzero(live).max().item() if bool(live.any()) else 0
        return (top + 1) * self.a.page_size if top else 0

    def live_pages_above(self, pages):
        t = self.torch
        free = self._free()
        have = t.zeros(self.ceiling + 1, dtype=t.bool, device=free.device)
        have[free[free <= self.ceiling]] = True
        above = t.arange(pages + 1, self.ceiling + 1, device=free.device)
        return above[~have[above]].tolist()

    def set(self, tokens, *, moves=None):
        """Move the ceiling to ``tokens`` (a page multiple). Growing adds the parked pages; shrinking needs the
        pages above the new ceiling to be free, after ``moves`` relocated the live ones to free pages below."""
        t, page = self.torch, self.a.page_size
        pages = tokens // page
        if pages > self.a.num_pages:
            raise ValueError(f"ceiling {tokens} exceeds the allocator's {self.a.num_pages * page} tokens")
        old = self.ceiling
        if pages >= old:
            extra = t.arange(old + 1, pages + 1, dtype=self.a.free_pages.dtype, device=self.a.free_pages.device)
            self.ceiling = pages
            self.a.free_pages = t.sort(t.cat((self._free(), extra)))[0]
            self.a.staged_pages, self.a.num_staged_pages = [], 0
            return
        free = self._free()
        if moves:
            taken = t.tensor([n for _, n in moves], dtype=free.dtype, device=free.device)
            freed = t.tensor([o for o, _ in moves], dtype=free.dtype, device=free.device)
            free = t.cat((free[~t.isin(free, taken)], freed))
        live = old - pages - int(((free > pages) & (free <= old)).sum())
        if live:
            raise RuntimeError(f"{live} live pages sit above the new ceiling {pages}")
        self.ceiling = pages
        self.a.free_pages = t.sort(free[free <= pages])[0]
        self.a.staged_pages, self.a.num_staged_pages = [], 0

    def plan_moves(self, pages):
        """[(live page above ``pages``, free page at or below it)] for compaction, or None if too few free."""
        live = self.live_pages_above(pages)
        free = self._free()
        low = self.torch.sort(free[free <= pages])[0].tolist()
        if len(low) < len(live):
            return None
        return list(zip(live, low[:len(live)]))


class Runtime:
    """Everything one worker needs after the coordinator binds: memory, device state, controller, graphs."""

    def __init__(self, state, coordinator, *, torch, dist):
        self.st, self.c, self.torch, self.dist = state, coordinator, torch, dist
        cfg = state["config"]
        self.cfg, self.mode = cfg, cfg.mode
        self.memory = state["memory"]
        a = coordinator.token_to_kv_pool_allocator
        self.logical = Ceiling(a.logical_attn_allocator, torch)
        self.hot = Ceiling(a.hisparse_attn_allocator, torch)
        self.group = coordinator.tp_group                  # the coordinator's CPU (Gloo) TP group
        self.rank = dist.get_rank(group=self.group)
        self.steps, self.last_tick, self.step_ms = 0, None, []
        self.pinned_level, self.receipts = None, []
        self.captured = set()
        self.capture_level = None
        self.decoding, self.layer = False, None
        layout = self.memory.layout
        geometry = ep.Geometry(layout, len(self.memory.handles), coordinator.padded_buffer_size,
                               state["max_running"], hot_min=coordinator.padded_buffer_size)
        profile = ep.load_profile(cfg, state["repo_root"])
        self.controller = ep.Controller(cfg, geometry, ep.CostModel(profile))
        # Hot rows cost chunks only for offloaded layers; with none offloaded start at the stock hot pool.
        start_hot = layout.hot_rows if not cfg.offloaded(cfg.controller["initial_level"]) else \
            min(layout.hot_rows, geometry.hot_per_request * min(state["max_running"], 16))
        first = self.controller.initial(hot_rows=start_hot)
        self.level, self.tokens, self.hot_rows = first.level, first.tokens, first.hot_rows
        self.memory.resize(tokens=self.tokens, hot_rows=self.hot_rows, offloaded=cfg.offloaded(self.level))
        self.logical.set(self._align(self.tokens, self.logical))
        self.hot.set(self._align(self.hot_rows, self.hot))
        self.kv = state["make_kv"](self)
        self.kv.publish_stale(self._stale())
        self.admission, self.stock_running = None, None
        self.fill_job = None            # smooth backfill in flight: (target level, pool indices, layers)
        self.held = []                  # PD requests whose resident-layer fill is still copying

    @staticmethod
    def _align(tokens, ceiling):
        page = ceiling.a.page_size
        return min(tokens // page * page, ceiling.a.num_pages * page)

    @contextmanager
    def decode_context(self):
        """A decode forward: smooth-mode getters return the whole combined buffer and stores use route rows."""
        self.decoding = True
        try:
            yield
        finally:
            self.decoding = False

    @contextmanager
    def layer_context(self, layer):
        """The layer whose attention runs now (DSA location translation has no layer argument)."""
        self.layer = layer
        try:
            yield
        finally:
            self.layer = None

    def layer_mode(self, layer):
        """Mode a forward should trace for ``layer``: the capture level during a steady graph capture."""
        level = self.level if self.capture_level is None else self.capture_level
        return ekv.OFFLOADED if layer in self.cfg.offloaded(level) else ekv.RESIDENT

    def _stale(self):
        if not self.cfg.feature("stale_hits"):
            return {}
        return {self.kv.index[layer]: self.memory.stale_rows(layer) for layer in self.cfg.managed
                if layer in self.kv.index}

    # ---- observation and decision ------------------------------------------------------------------------
    def observe(self, scheduler):
        batch = scheduler.running_batch
        reqs = [] if batch is None or batch.is_empty() else batch.reqs
        waiting = list(scheduler.waiting_queue)
        prealloc = getattr(getattr(scheduler, "disagg_decode_prealloc_queue", None), "queue", [])
        queued = waiting + self.held + [getattr(d, "req", d) for d in prealloc]
        busy = bool(self.c.has_ongoing_staging() or getattr(scheduler, "chunked_req", None) is not None
                    or self.fill_job is not None and self.mode == "steady")
        step_ms = sum(self.step_ms) / len(self.step_ms) if self.step_ms else None
        self.step_ms = []
        lengths = [len(r.origin_input_ids) + len(r.output_ids) for r in reqs]
        chunk = getattr(scheduler, "chunked_prefill_size", None) or 0
        decode_node = getattr(scheduler, "disagg_decode_prealloc_queue", None) is not None
        prefill = 0 if decode_node or not waiting else len(waiting[0].origin_input_ids)
        return ep.Observation(
            step=self.steps, level=self.level, tokens=self.logical.tokens(),
            tokens_used=self.logical.used_tokens(), tokens_peak=self.logical.peak_tokens(),
            hot_rows=self.hot.tokens(), hot_used=self.hot.used_tokens(), hot_peak=self.hot.peak_tokens(),
            running=len(reqs), mean_context=sum(lengths) / len(lengths) if lengths else 1.0,
            waiting=len(queued), waiting_tokens=sum(len(r.origin_input_ids) for r in queued[:256]),
            step_ms=step_ms, busy=busy, prefill_rows=min(prefill, chunk) if chunk > 0 else prefill)

    def _broadcast(self, payload):
        if self.dist.get_world_size(group=self.group) == 1:
            return payload
        box = [payload if self.rank == 0 else None]
        self.dist.broadcast_object_list(box, src=self.dist.get_global_rank(self.group, 0), group=self.group)
        return box[0]

    def _agree_min(self, value):
        if self.dist.get_world_size(group=self.group) == 1:
            return value
        box = self.torch.tensor([value], dtype=self.torch.int64)
        self.dist.all_reduce(box, op=self.dist.ReduceOp.MIN, group=self.group)
        return int(box.item())

    def _release_filled(self, scheduler):
        """PD decode: requests stay out of the waiting queue until their resident-layer fill finished on every
        rank (FIFO, one agreed count; every rank admitted the same requests), so no decode step waits for a
        long host-to-device copy."""
        kv = self.kv
        self.held = [req for _, req in kv.fills]
        if not kv.fills:
            return
        pending = {id(req) for req in self.held}
        scheduler.waiting_queue[:] = [r for r in scheduler.waiting_queue if id(r) not in pending]
        done = self._agree_min(kv.ready())
        scheduler.waiting_queue.extend(req for _, req in kv.fills[:done])
        del kv.fills[:done]
        self.held = [req for _, req in kv.fills]

    def tick(self, scheduler, last_batch=None):
        """Between batches: count decode steps, and every ``interval_steps`` let rank 0 decide and every rank
        apply the decision (a paced smooth backfill advances on every step)."""
        if self.stock_running is None:
            self.stock_running = scheduler.max_running_requests
        now = time.monotonic()
        last = last_batch if last_batch is not None else getattr(scheduler, "last_batch", None)
        if last is not None and last.forward_mode.is_decode():
            self.steps += 1
            if self.last_tick is not None:
                self.step_ms.append(1e3 * (now - self.last_tick))
        self.last_tick = now
        self._release_filled(scheduler)
        if self.fill_job is not None:
            self._advance_fill(scheduler)
        if self.steps == 0 or self.steps % self.cfg.controller["interval_steps"]:
            return
        decision = None
        if self.rank == 0:
            obs = self.observe(scheduler)
            if self.pinned_level is None:
                d = self.controller.tick(obs)
            else:
                d = self.pinned(obs)
            decision = dict(level=d.level, tokens=d.tokens, hot_rows=d.hot_rows, admission=d.admission_cap,
                            reason=d.reason, step=obs.step)
        decision = self._broadcast(decision)
        if decision is not None:
            self.apply(decision, scheduler)

    def pinned(self, obs):
        """The operating point of the level an operator pinned (``POST /set_internal_state``); the current state
        while the live KV does not fit that level yet."""
        level = self.pinned_level
        if level == obs.level:
            return ep.Decision(level, obs.tokens, obs.hot_rows, None, "pinned")
        hot = max(obs.hot_peak, self.controller._hot_rows_for(obs, max(1, obs.running)))
        tokens = self.controller.capacity(level, hot)
        floor = obs.tokens_used if self.cfg.feature("compaction") else obs.tokens_peak
        if tokens < floor + obs.running * self.cfg.controller["horizon_steps"]:
            return ep.Decision(obs.level, obs.tokens, obs.hot_rows, None, f"pinned level {level} does not fit yet")
        return ep.Decision(level, tokens, hot, None, f"pinned to level {level}")

    def apply(self, decision, scheduler):
        stock = self.stock_running or self.st["max_running"]
        self.admission = decision["admission"]
        scheduler.max_running_requests = stock if self.admission is None else max(1, min(stock, self.admission))
        level, tokens, hot = decision["level"], decision["tokens"], decision["hot_rows"]
        if (level, tokens, hot) == (self.level, self.logical.tokens(), self.hot.tokens()) or \
                self.fill_job and level != self.level:      # a paced backfill finishes before the next level
            return
        start = time.monotonic()
        try:
            if level > self.level:
                record = self.offload(level, tokens, hot)
            elif level < self.level:
                record = self.backfill(level, tokens, hot, scheduler)
            else:
                aligned = self._align(tokens, self.logical)
                self._require(self._can_shrink(aligned), "live pages above the new ceiling")
                self._require(self._room(self._align(self._hot_floor(hot), self.hot), self.cfg.offloaded(level),
                                         self._filling()) >= aligned, "the chunk pool cannot back that many tokens")
                record = self.resize(tokens, hot)
            status = "committed"
        except Deferred as why:          # checked before any state changed: the next tick decides again
            record, status = dict(reason=str(why)), "deferred"
        except Exception as error:      # every rank fails the same transition: stop the worker group loudly
            record, status = dict(error=f"{type(error).__name__}: {error}"), "failed"
            raise
        finally:
            self._receipt(decision, status, time.monotonic() - start, locals().get("record"))

    def _receipt(self, decision, status, seconds, record):
        entry = dict(event="hisparse_elastic_transition", status=status, mode=self.mode, level=self.level,
                     tokens=self.logical.tokens(), hot_rows=self.hot.tokens(), decision=decision,
                     seconds=round(seconds, 6), detail=record, free_chunks=self.memory.free_chunks)
        self.receipts.append(entry)
        emit(entry, self.rank)

    # ---- transitions -------------------------------------------------------------------------------------
    def _sync(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    @staticmethod
    def _require(ok, reason):
        if not ok:
            raise Deferred(reason)

    def _room(self, hot, offloaded, filling=()):
        """Logical tokens the chunk pool backs in a state (-1 when not even the hot rows fit)."""
        return self.memory.layout.max_tokens_for(len(self.memory.handles), hot_rows=hot, offloaded=offloaded,
                                                 filling=filling)

    def _can_shrink(self, tokens):
        """The logical ceiling can drop to ``tokens``: no live page above it, or compaction can move them down."""
        if tokens >= self.logical.tokens():
            return True
        moves = self.logical.plan_moves(tokens // self.logical.a.page_size)
        return moves is not None and (not moves or self.cfg.feature("compaction"))

    def _hot_floor(self, hot):
        """Hot rows never drop below the highest live device-buffer row."""
        page = self.hot.a.page_size
        return max(hot, -(-self.hot.peak_tokens() // page) * page)

    def resize(self, tokens, hot):
        """Same level: lower the ceilings that shrink (compaction moves live pages down first), move the backing
        in one step (freed chunks back the grown regions), then raise the ceilings that grow."""
        tokens, hot = self._align(tokens, self.logical), self._align(self._hot_floor(hot), self.hot)
        moved = self._shrink_ceilings(tokens, hot)
        changed = self.memory.resize(tokens=tokens, hot_rows=hot, offloaded=self.cfg.offloaded(self.level),
                                     filling=self._filling())
        self.logical.set(tokens)
        self.hot.set(hot)
        self.kv.publish_stale(self._stale())
        return dict(compacted_pages=moved, memory=changed)

    def _filling(self):
        return () if self.fill_job is None else self.fill_job[2]

    def _shrink_ceilings(self, tokens, hot):
        moved = 0
        if tokens < self.logical.tokens():
            pages = tokens // self.logical.a.page_size
            moves = self.logical.plan_moves(pages)
            if moves is None:
                raise RuntimeError("compaction needs more free logical pages than exist below the new ceiling")
            if moves and not self.cfg.feature("compaction"):
                raise RuntimeError("live pages above the new ceiling and features.compaction is off")
            if moves:
                self._sync()
                buffers, movers = self.st["compaction_buffers"](self)
                moved = self.kv.compact(moves, self.logical.a.page_size, buffers,
                                        self.c.mem_pool_device.full_to_hisparse_device_index_mapping, movers)
                self.kv.stale_limit.zero_()       # pages moved under the stale regions: no more stale reads
            self.logical.set(self._align(tokens, self.logical), moves=moves)
        if hot < self.hot.tokens():
            if self.hot.live_pages_above(hot // self.hot.a.page_size):
                raise RuntimeError("live hot rows above the new hot ceiling")
            self.hot.set(self._align(hot, self.hot))
        return moved

    def offload(self, level, tokens, hot):
        """Resident -> offloaded for the next groups: flush their newest records to host, back their hot rows,
        warm them from the resident rows, publish the modes, then let the freed resident chunks back more
        logical tokens."""
        cfg, kv = self.cfg, self.kv
        leaving = sorted(cfg.offloaded(level) - cfg.offloaded(self.level))
        indices = [kv.index[layer] for layer in leaving]
        tokens, hot = self._align(tokens, self.logical), self._align(self._hot_floor(hot), self.hot)
        warm_tokens = min(tokens, self.logical.tokens())
        self._require(self._room(hot, cfg.offloaded(level)) >= tokens and self._can_shrink(warm_tokens),
                      f"level {level} does not fit {tokens} tokens and {hot} hot rows yet")
        self._sync()
        self.c.wait_for_pending_backup()
        kv.flush_newest(indices)
        moved = self._shrink_ceilings(warm_tokens, min(hot, self.hot.tokens()))
        self._sync()
        state = dict(tokens=self.logical.tokens(), hot_rows=hot, offloaded=cfg.offloaded(level))
        # Back the new hot regions without reclaiming the leaving layers' resident rows the warm copy reads;
        # when only those chunks are left, warm from host (complete after the flush) instead.
        source = "device" if self.memory.plan(**state, protect=leaving) is not None else "host"
        self.memory.resize(**state, protect=leaving if source == "device" else ())
        self.hot.set(hot)
        warmed = kv.warm(indices, source=source, long_hits=cfg.feature("warm_start"))
        self._sync()
        if self.mode == "steady":
            self.ensure_graphs(level)
        kv.set_modes(cfg.offloaded(level))
        self.level = level
        detail = self.resize(tokens, hot)
        return dict(leaving=leaving, warmed_rows=warmed, warm_source=source, compacted_first=moved, **detail)

    def _stale_bounds(self, layers):
        """Backed rows of each layer's inactive resident region: a backfill skips records still valid there."""
        return {self.kv.index[layer]: self.memory.stale_rows(layer) for layer in layers}

    def backfill(self, level, tokens, hot, scheduler):
        """Offloaded -> resident for the last groups, refilled from host, skipping records still valid in their
        stale rows. Steady stops decode: flush the newest records, swap the regions (hot released, resident
        backed) and copy everything before the next batch. Smooth keeps decoding: both regions stay backed
        while the copy is paced across steps (``_advance_fill``); the modes flip once it completes. The
        logical ceiling first drops to what the copy state can back."""
        cfg, kv = self.cfg, self.kv
        returning = sorted(cfg.offloaded(self.level) - cfg.offloaded(level))
        indices = [kv.index[layer] for layer in returning]
        state = dict(offloaded=cfg.offloaded(level)) if self.mode == "steady" else \
            dict(offloaded=cfg.offloaded(self.level), filling=returning)
        target = min(self._align(tokens, self.logical), self.logical.tokens(),
                     self._align(max(0, self._room(self.hot.tokens(), **state)), self.logical))
        self._require(target > 0 and self._can_shrink(target), f"level {level} does not fit the live KV yet")
        self._sync()
        self.c.wait_for_pending_backup()
        moved = self._shrink_ceilings(target, self.hot.tokens())
        if self.mode == "steady":
            kv.flush_newest(indices)
            self._sync()
        bounds = self._stale_bounds(returning)
        self.memory.resize(tokens=self.logical.tokens(), hot_rows=self.hot.tokens(), **state)
        kv.begin_fill(indices)
        self.fill_job = (level, indices, returning, bounds)
        if self.mode == "smooth":
            return dict(returning=returning, compacted_pages=moved, paced=True)
        left = kv.fill(indices, stale_rows=bounds)
        if left:
            raise RuntimeError(f"steady backfill left {left} records")
        return dict(returning=returning, compacted_first=moved, **self._finish_fill(tokens, hot))

    def _advance_fill(self, scheduler):
        """Smooth backfill: one paced step of host-to-device copies; on the last one, the newest records come
        from the hot rows and the modes flip."""
        level, indices, returning, bounds = self.fill_job
        step = self.step_ms[-1] if self.step_ms else 20.0
        budget = ep.backfill_rows_per_step(self.cfg, step, self.kv.record_bytes())
        if self.kv.fill(indices, budget=budget, stale_rows=bounds):
            return
        self._sync()
        self.c.wait_for_pending_backup()
        self.kv.fill(indices, stale_rows=bounds)            # records backed up while the copy ran
        self.kv.catch_up(indices)
        decision = dict(level=level, tokens=self.logical.tokens(), hot_rows=self.hot.tokens(),
                        admission=self.admission, reason="backfill complete", step=self.steps)
        start = time.monotonic()
        detail = self._finish_fill(self.logical.tokens(), self.hot.tokens())
        self._receipt(decision, "committed", time.monotonic() - start, detail)

    def _finish_fill(self, tokens, hot):
        level, indices, returning, _ = self.fill_job
        self._sync()
        if self.mode == "steady":
            self.ensure_graphs(level)
        self.kv.set_modes(self.cfg.offloaded(level))
        self.level, self.fill_job = level, None
        tokens = min(tokens, self.controller.capacity(level, self.hot.tokens()))
        detail = self.resize(tokens, hot)
        return dict(filled_layers=returning, **detail)

    # ---- steady-mode graph sets --------------------------------------------------------------------------
    def precapture(self, runner):
        """Steady mode at start-up, before any request: capture the graph set of every
        ``memory.precapture_levels`` level and the start level, each over a minimal backing of that level's
        regions (capture forwards touch only the padding rows), then restore the start-up state."""
        cfg, page = self.cfg, self.logical.a.page_size
        try:
            for level in sorted(set(cfg.precapture_levels()) | {self.level}):
                self.memory.resize(tokens=page, hot_rows=page, offloaded=cfg.offloaded(level), keep_stale=False)
                self.capture(runner, [level])
        finally:
            self.memory.resize(tokens=self.logical.tokens(), hot_rows=self.hot.tokens(),
                               offloaded=cfg.offloaded(self.level), keep_stale=False)
            self.kv.publish_stale(self._stale())

    def ensure_graphs(self, level):
        runner = self.st.get("graph_runner")
        if runner is None or level in self.captured:
            return
        self.capture(runner, [level])

    def capture(self, runner, levels):
        stock = self.st["stock_capture"]
        for level in levels:
            self.capture_level = level
            try:
                stock(runner)
            finally:
                self.capture_level = None
            self.captured.add(level)

    def graph_key(self, shape_key):
        if self.mode != "steady":
            return shape_key
        level = self.level if self.capture_level is None else self.capture_level
        return ("hisparse-elastic", level, shape_key)


# ---- VMM buffers of the HiSparse device pool ---------------------------------------------------------------
class Factory:
    """Builds the HiSparse device pool's per-layer buffers as VMM reservations while the stock constructor runs.
    Stock code keeps seeing its usual ``[size + page, ...]`` hot views; ``seal`` creates a chunk pool of exactly
    the bytes those stock buffers would have allocated."""

    def __init__(self, state, torch, *, size, ratio, page, layers, chunk_bytes):
        self.state, self.torch, self.layers = state, torch, tuple(layers)
        self.size, self.page, self.ratio = size, page, ratio
        layout = em.Layout([], chunk_bytes=chunk_bytes, max_tokens=size * ratio, hot_rows=size, page_tokens=page)
        backend = state.get("backend") or (em.CudaVmmBackend(torch) if torch.cuda.is_available()
                                           else em.CpuBackend(torch))
        self.memory = em.ElasticMemory(layout, backend, torch=torch)
        self.stock_bytes, self.views = 0, {}

    def buffer(self, part, layer, row_shape, dtype):
        """Hot view (rows 0..size+page-1) of a new combined buffer; the resident and full views are recorded."""
        t = self.torch
        row_bytes = int(t.Size(row_shape).numel()) * t._utils._element_size(dtype)
        spec = em.BufferSpec(f"{part}{layer}", row_bytes, em.SWITCHABLE, layer=layer, pad_rows=self.page,
                             shape=tuple(row_shape), dtype=str(dtype).replace("torch.", ""))
        buf = self.memory.add(spec)
        full = self.memory.view(spec.name)
        self.views[(layer, part)] = dict(full=full, hot=full[:self.size + self.page],
                                         resident=full[buf.hot_offset:], hot_offset=buf.hot_offset)
        self.stock_bytes += (self.size + self.page) * row_bytes
        return self.views[(layer, part)]["hot"]

    def seal(self):
        chunks = self.stock_bytes // self.memory.layout.chunk
        self.memory.seal(chunks)
        return chunks


def _runtime():
    return _state().get("runtime")


def _view(layer, part):
    """The tensor a stock caller should see for ``layer``: the whole combined buffer inside a smooth decode
    forward (route rows), else the view of the layer's current (or capture) mode."""
    st, rt = _state(), _runtime()
    views = st["views"][(layer, part)]
    if rt is None:
        return views["hot"]
    if rt.mode == "smooth" and rt.decoding and layer in rt.cfg.managed:
        return views["full"]
    return views["resident"] if rt.layer_mode(layer) == ekv.RESIDENT else views["hot"]


def _store_target(layer, raw, mapping):
    """(rows, view name) of a store into ``layer``, or None for the stock offloaded path."""
    rt = _runtime()
    if rt is None:
        return None
    if rt.mode == "smooth" and rt.decoding and layer in rt.cfg.managed:
        return rt.kv.store_rows(rt.kv.index[layer], raw, mapping[raw]), "full"
    if rt.layer_mode(layer) == ekv.RESIDENT:
        return raw, "resident"
    return None


# ---- MiniMax-M3 ---------------------------------------------------------------------------------------------
def _minimax_pool(state, torch):
    """The HiSparse main pool's sparse layers get combined buffers; dense and index pools stay stock."""
    import inspect
    memory_mod = importlib.import_module("sglang.srt.mem_cache.memory_pool")
    main_cls = importlib.import_module("sglang.srt.mem_cache.hisparse_memory_pool").HiSparseMHAMainPool
    pool_cls, mha_cls, originals = memory_mod.MiniMaxSparseKVPool, memory_mod.MHATokenToKVPool, {}
    signature = inspect.signature(pool_cls.__init__)

    def create_buffers(self):
        factory = state.get("factory")
        if factory is None or not isinstance(self, main_cls):
            return originals["_create_buffers_normal"](self)
        if self.kv_cache_layout != "nhd" or getattr(self, "use_hnd", False) or self.post_capture_active:
            raise RuntimeError("elastic layers need plain NHD main-KV rows without post-capture VA backing")
        if self.layer_num != len(factory.layers):
            raise RuntimeError("the HiSparse main pool does not cover exactly the sparse layers")
        k_shape, v_shape = self._kv_buffer_shapes()
        self.k_buffer = [factory.buffer("k", layer, k_shape[1:], self.store_dtype) for layer in factory.layers]
        self.v_buffer = [factory.buffer("v", layer, v_shape[1:], self.store_dtype) for layer in factory.layers]

    def pool_init(self, *args, **kwargs):
        a = signature.bind(self, *args, **kwargs)
        a.apply_defaults()
        a = a.arguments
        if type(self) is not pool_cls or not a["enable_hisparse"] or a["main_pool_cls"] is not mha_cls:
            raise RuntimeError("elastic layers support the stock MiniMaxSparseKVPool with --enable-hisparse")
        eligible = tuple(layer for layer in a["sparse_layer_ids"] if a["start_layer"] <= layer < a["end_layer"])
        if eligible != tuple(range(eligible[0], eligible[0] + len(eligible))):
            raise RuntimeError("the coordinator indexes sparse layers contiguously; this stage's are not")
        state["factory"] = Factory(state, torch, size=a["size"], ratio=a["host_to_device_ratio"],
                                   page=a["page_size"], layers=eligible, chunk_bytes=state["chunk_bytes"])
        try:
            originals["__init__"](self, *args, **kwargs)
        finally:
            factory = state.pop("factory")
        state.update(memory=factory.memory, views=factory.views, eligible=eligible, pool=self,
                     physical_chunks=factory.seal(), stock_bytes=factory.stock_bytes)

    def getter(name, part):
        def hook(self, layer_id):
            if layer_id not in state.get("eligible", ()):
                return originals[name](self, layer_id)
            self._wait_for_layer(layer_id)
            k, v = _view(layer_id, "k"), _view(layer_id, "v")
            return dict(k=k, v=v, kv=(k, v))[part]
        return hook

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None):
        layer_id, main = layer.layer_id, self.main_pool
        if layer_id not in state.get("eligible", ()):
            return originals["set_kv_buffer"](self, layer, loc, cache_k, cache_v, k_scale, v_scale)
        raw, _, _ = memory_mod.unwrap_write_loc(loc)
        target = _store_target(layer_id, raw, main.full_to_hisparse_device_index_mapping)
        if target is None:
            return originals["set_kv_buffer"](self, layer, loc, cache_k, cache_v, k_scale, v_scale)
        rows, view = target
        if {k_scale, v_scale} - {None, 1.0} or cache_k.dtype != main.dtype or main.store_dtype != main.dtype:
            raise RuntimeError("elastic layers store unscaled, unquantized K/V in the pool dtype only")
        k, v = state["views"][(layer_id, "k")][view], state["views"][(layer_id, "v")][view]
        memory_mod._set_kv_buffer_impl(cache_k, cache_v, k, v, rows, row_dim=main.row_dim,
                                       store_dtype=main.store_dtype, device_module=main.device_module,
                                       size_limit=k.shape[0], alt_stream=main.alt_stream, v_row_dim=main.v_row_dim)

    hooks = dict(__init__=pool_init, get_key_buffer=getter("get_key_buffer", "k"),
                 get_value_buffer=getter("get_value_buffer", "v"), get_kv_buffer=getter("get_kv_buffer", "kv"),
                 set_kv_buffer=set_kv_buffer)
    for name, hook in hooks.items():
        originals[name] = _patch(state, pool_cls, name, hook)
    originals["_create_buffers_normal"] = _patch(state, mha_cls, "_create_buffers_normal", create_buffers)


def _minimax_backend(state, torch):
    """Decode swap-in and sparse prefill of MiniMax-M3, installed when the coordinator binds."""
    backend_cls = importlib.import_module("sglang.srt.layers.attention.minimax_sparse_backend") \
        .MiniMaxSparseAttnBackend
    ops = importlib.import_module("sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse")

    def swap_blocks(self, forward_batch, topk_idx, layer_id):
        rt = _runtime()
        if layer_id not in state["eligible"] or rt is None:
            return original_swap(self, forward_batch, topk_idx, layer_id)
        if rt.mode == "smooth" and rt.decoding and layer_id in rt.cfg.managed:
            kv = rt.kv
            masked, rows, reads = kv.smooth_select(kv.index[layer_id], forward_batch.req_pool_indices,
                                                   forward_batch.seq_lens, topk_idx[0], block=self.block_size_k)
            swapped = original_swap(self, forward_batch, masked[None], layer_id)
            return kv.merge(swapped[0], rows, reads)[None]
        if rt.layer_mode(layer_id) == ekv.RESIDENT:
            return None             # the stock consumer reads req_to_token rows of the resident view
        return original_swap(self, forward_batch, topk_idx, layer_id)

    def forward_decode(self, *args, **kwargs):
        rt = _runtime()
        if rt is None or rt.mode != "smooth":
            return original_decode(self, *args, **kwargs)
        with rt.decode_context():
            return original_decode(self, *args, **kwargs)

    def prefill(*args, **kwargs):
        if len(args) < 3 or "loc_mapping" not in kwargs:
            raise RuntimeError(f"{PREFILL} expects the pinned positional K/V caches and loc_mapping keyword")
        if args[1].data_ptr() in state["resident_ptrs"]:
            kwargs["loc_mapping"] = None    # resident rows live at their logical locations
        elif kwargs["loc_mapping"] is None:
            raise RuntimeError("an offloaded prefill must read through the HiSparse location mapping")
        return original_prefill(*args, **kwargs)

    state["resident_ptrs"] = {v["resident"].data_ptr() for (_, part), v in state["views"].items() if part == "k"}
    original_swap = _patch(state, backend_cls, "_hisparse_swap_in_blocks", swap_blocks)
    original_decode = _patch(state, backend_cls, "forward_decode", forward_decode)
    original_prefill = _patch(state, ops, PREFILL, prefill)


# ---- DSA: GLM-5 / GLM-5.1 / GLM-5.2, DeepSeek-V3.2 ---------------------------------------------------------
def _dsa_pool(state, torch):
    """The HiSparse DSA pool's per-layer latent buffers get combined buffers; the indexer caches stay stock."""
    import inspect
    memory_mod = importlib.import_module("sglang.srt.mem_cache.memory_pool")
    pool_cls = importlib.import_module("sglang.srt.mem_cache.hisparse_memory_pool").HiSparseDSATokenToKVPool
    mla_cls, originals = memory_mod.MLATokenToKVPool, {}
    signature = inspect.signature(pool_cls.__init__)

    def create_buffers(self):
        factory = state.get("factory")
        if factory is None or not isinstance(self, pool_cls):
            return originals["_create_buffers"](self)
        if self.layer_num != len(factory.layers):
            raise RuntimeError("the HiSparse DSA pool does not cover exactly its stage's layers")
        self.kv_buffer = [factory.buffer("kv", layer, (1, self.kv_cache_dim), self.store_dtype)
                          for layer in factory.layers]

    def pool_init(self, *args, **kwargs):
        a = signature.bind(self, *args, **kwargs)
        a.apply_defaults()
        a = a.arguments
        if type(self) is not pool_cls or (a["start_layer"] or 0) != 0:
            raise RuntimeError("elastic layers support the stock HiSparseDSATokenToKVPool of one pipeline stage")
        if a["dtype"] != torch.bfloat16 or a["kv_cache_dim"] != a["kv_lora_rank"] + a["qk_rope_head_dim"]:
            raise RuntimeError("elastic DSA layers need an unquantized BF16 latent cache (--kv-cache-dtype bfloat16)")
        layers = tuple(range(a["layer_num"]))
        state["factory"] = Factory(state, torch, size=a["size"], ratio=a["host_to_device_ratio"],
                                   page=a["page_size"], layers=layers, chunk_bytes=state["chunk_bytes"])
        try:
            originals["__init__"](self, *args, **kwargs)
        finally:
            factory = state.pop("factory")
        state.update(memory=factory.memory, views=factory.views, eligible=layers, pool=self,
                     physical_chunks=factory.seal(), stock_bytes=factory.stock_bytes)

    def getter(name):
        def hook(self, layer_id):
            if self is not state.get("pool"):
                return originals[name](self, layer_id)
            if self.layer_transfer_counter is not None:
                self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
            kv = _view(layer_id, "kv")
            return kv if name == "get_key_buffer" else kv[..., :self.kv_lora_rank]
        return hook

    def set_mla_kv_buffer(self, layer, loc_info, cache_k_nope, cache_k_rope):
        raw, _, _ = memory_mod.unwrap_write_loc(loc_info)
        target = _store_target(layer.layer_id, raw, self.full_to_hisparse_device_index_mapping)
        if target is None:
            return originals["set_mla_kv_buffer"](self, layer, loc_info, cache_k_nope, cache_k_rope)
        rows, view = target
        mla_cls._write_mla_kv_buffer(self, state["views"][(layer.layer_id, "kv")][view], rows, cache_k_nope,
                                     cache_k_rope)

    def translate(self, compressed_indices):
        """Prefill page tables: a resident layer's rows are its logical locations."""
        rt = _runtime()
        if rt is not None and rt.layer is not None and rt.layer_mode(rt.layer) == ekv.RESIDENT:
            return compressed_indices
        return originals["translate_loc_to_hisparse_device"](self, compressed_indices)

    for owner, name, hook in ((pool_cls, "__init__", pool_init), (pool_cls, "set_mla_kv_buffer", set_mla_kv_buffer),
                              (pool_cls, "translate_loc_to_hisparse_device", translate),
                              (mla_cls, "_create_buffers", create_buffers),
                              (mla_cls, "get_key_buffer", getter("get_key_buffer")),
                              (mla_cls, "get_value_buffer", getter("get_value_buffer"))):
        originals[name] = _patch(state, owner, name, hook)


def _dsa_backend(state, torch):
    """Layer context of every DSA attention call (location translation has no layer argument), and the smooth
    decode context; installed when the coordinator binds."""
    backend_cls = importlib.import_module("sglang.srt.layers.attention.dsa_backend").DeepseekSparseAttnBackend
    originals = {}

    def wrap(name, decode):
        def hook(self, *args, **kwargs):
            rt = _runtime()
            if rt is None:
                return originals[name](self, *args, **kwargs)
            layer = args[3] if len(args) > 3 else kwargs["layer"]
            with rt.layer_context(layer.layer_id), \
                    (rt.decode_context() if decode and rt.mode == "smooth" else nullcontext()):
                return originals[name](self, *args, **kwargs)
        return hook

    for name, decode in (("forward_extend", False), ("forward_decode", True)):
        originals[name] = _patch(state, backend_cls, name, wrap(name, decode))


ANCHOR_COPY_BLOCKS = 32     # grid of a smooth anchor's own planned copy (on the critical path, unlike followers')


def _planned_copy(module, c, kv, i, num_reqs, blocks):
    """Replay the anchor's miss plan into pool layer i only while i is offloaded: the request count the kernel
    reads is ``num_real_reqs`` gated by the route table on the device, so one captured graph serves both modes."""
    gate = c.num_real_reqs * (kv.route[i, 0:1] == ekv.OFFLOADED).to(c.num_real_reqs.dtype)
    module.copy_cache_planned_mla(miss_src=c._miss_src[:num_reqs], miss_dst=c._miss_dst[:num_reqs],
                                  miss_count=c._miss_count[:num_reqs], num_real_reqs=gate,
                                  host_cache=c.mem_pool_host.kv_buffer[i], device_buffer=c.mem_pool_device.kv_buffer[i],
                                  item_size_bytes=c.item_size_bytes, num_blocks=blocks,
                                  is_dsv4_layout=c.is_dsv4_hisparse, skip_io=c.skip_io)


def _dsa_coordinator(state, torch, module):
    """DSA decode swap-in by layer mode. Steady: a resident layer reads its logical rows; a resident anchor whose
    shared-index followers are offloaded still runs its kernel with ``skip_io`` (slot table and miss plan, no
    bytes of its own). Smooth: masked selection plus merge, except anchors and followers of shared-index groups,
    which keep the full plan and gate their bytes on the route table (``_planned_copy``)."""
    cls, originals = module.HiSparseCoordinator, {}

    def smooth(rt, layer):
        return rt.mode == "smooth" and rt.decoding and layer in rt.cfg.managed

    def resident(rt, layer):
        return rt.layer_mode(layer) == ekv.RESIDENT

    def group_of(c, layer):
        return c._prefetch_groups.get(layer) if c.enable_prefetch else None

    def follower(c, layer):
        return c.enable_prefetch and c._is_shared_index_layer[layer]

    def swap_pages(self, req_pool_indices, seq_lens, top_k, layer_id):
        rt = _runtime()
        stock = originals["swap_in_selected_pages"]
        if rt is None or top_k.ndim != 2:
            return stock(self, req_pool_indices, seq_lens, top_k, layer_id)
        kv, i = rt.kv, rt.kv.index[layer_id]
        group = group_of(self, layer_id)
        if smooth(rt, layer_id):
            if group or follower(self, layer_id):
                hot = stock(self, req_pool_indices, seq_lens, top_k, layer_id)
                if group:
                    _planned_copy(module, self, kv, i, req_pool_indices.size(0), ANCHOR_COPY_BLOCKS)
                _, rows, reads = kv.smooth_select(i, req_pool_indices, seq_lens, top_k)
            else:
                masked, rows, reads = kv.smooth_select(i, req_pool_indices, seq_lens, top_k)
                hot = stock(self, req_pool_indices, seq_lens, masked, layer_id)
            return kv.merge(hot, rows, reads)
        if not resident(rt, layer_id):
            return stock(self, req_pool_indices, seq_lens, top_k, layer_id)
        # A smooth graph is captured once, so a pinned anchor of switchable followers always plans for them.
        feeds = group and any(f in rt.cfg.managed if rt.mode == "smooth" and rt.decoding else not resident(rt, f)
                              for f in group)
        if feeds:
            stock(self, req_pool_indices, seq_lens, top_k, layer_id)
        return kv.resident_page_table(req_pool_indices, top_k)

    def run_swap(self, req_pool_indices, compressed_seq_lens, top_k_result, layer_id, record_plan=False):
        rt = _runtime()
        stock = originals["_run_swap_in_kernel"]
        if rt is None:
            return stock(self, req_pool_indices, compressed_seq_lens, top_k_result, layer_id, record_plan)
        quiet = bool(group_of(self, layer_id)) if smooth(rt, layer_id) else resident(rt, layer_id)
        saved, self.skip_io = self.skip_io, self.skip_io or quiet
        try:
            return stock(self, req_pool_indices, compressed_seq_lens, top_k_result, layer_id, record_plan)
        finally:
            self.skip_io = saved

    def copy_only(self, num_reqs, skip_layer):
        rt = _runtime()
        if rt is not None and smooth(rt, skip_layer):
            return _planned_copy(module, self, rt.kv, rt.kv.index[skip_layer], num_reqs, self._prefetch_copy_blocks)
        if rt is not None and resident(rt, skip_layer):
            return None             # a resident follower reads its own logical rows
        return originals["_run_copy_only_kernel"](self, num_reqs, skip_layer)

    for name, hook in (("swap_in_selected_pages", swap_pages), ("_run_swap_in_kernel", run_swap),
                       ("_run_copy_only_kernel", copy_only)):
        originals[name] = _patch(state, cls, name, hook)


# ---- HiSparse coordinator: write-through, staging, admission ------------------------------------------------
def _coordinator(state, torch, dist):
    module = importlib.import_module("sglang.srt.managers.hisparse_coordinator")
    cls, device_module, originals = module.HiSparseCoordinator, module.device_module, {}

    def init(self, *args, **kwargs):
        originals["__init__"](self, *args, **kwargs)
        _bind(state, self, torch=torch, dist=dist, module=module)

    def eager_backup(self, seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu):
        """The stock previous-token backup, from the rows each layer's last forward wrote (write-through)."""
        rt = _runtime()
        if rt is None:
            return originals["_eager_backup_previous_token"](self, seq_lens, req_pool_indices, seq_lens_cpu,
                                                             req_pool_indices_cpu)
        r, slots, records = self.compress_ratio, [], []
        for seq, rid in zip(seq_lens_cpu.tolist(), req_pool_indices_cpu.tolist()):
            if self._skip_first_backup[rid]:
                self._skip_first_backup[rid] = False
            elif (seq - 1) % r == 0:      # the previous step produced record (seq - 1) // r - 1
                slots.append(rid)
                records.append((seq - 1) // r - 1)
        kv = rt.kv
        if slots:
            host = torch.cat([self.mem_pool_host.alloc_paged_token_slots(
                self.req_to_host_pool, self.req_to_host_pool_allocated_len, rid, record, 1)
                for rid, record in zip(slots, records)])
            kv.write_through(slots, records, host)
        kv.forward_begins()
        kv.decode_begins(req_pool_indices_cpu.tolist(), [seq // r for seq in seq_lens_cpu.tolist()])

    def admit_request_into_staging(self, req):
        rt = _runtime()
        if rt is None:
            return originals["admit_request_into_staging"](self, req)
        req.hisparse_staging = True
        slot = req.kv.req_pool_idx
        full = self.req_to_token_pool.req_to_token[slot, :req.extend_end].to(dtype=torch.int64, copy=True)
        hot = self.mem_pool_device.translate_loc_from_full_to_hisparse_device(full)
        host = self.mem_pool_host.alloc_paged_token_slots(self.req_to_host_pool, self.req_to_host_pool_allocated_len,
                                                          slot, 0, len(hot))
        start, finish = device_module.Event(), device_module.Event()
        start.record()
        with device_module.stream(self.write_staging_stream):
            start.wait(self.write_staging_stream)
            rt.kv.stage(slot, host, hot)
            finish.record()
            for tensor in (host, hot):
                if tensor.is_cuda:
                    tensor.record_stream(self.write_staging_stream)
        self.ack_staging_queue.append(module.HiSparseAct(start, finish, req))

    def alloc_device_buffer(self, req):
        originals["alloc_device_buffer"](self, req)
        rt = _runtime()
        if rt is not None:          # staged prompts are complete in host; direct ones arrived there
            staged = getattr(req, "hisparse_staging", False)
            rt.kv.admitted(req.kv.req_pool_idx,
                           req.extend_end if staged else self.host_token_len(req.kv.kv_allocated_len))

    def admit_request_direct(self, req):
        """PD decode: stock admission (offloaded layers start all-miss, or preloaded when short), then the
        resident layers' logical rows are filled from host on the staging stream."""
        originals["admit_request_direct"](self, req)
        rt = _runtime()
        if rt is not None:
            slot, n = req.kv.req_pool_idx, self.host_token_len(req.kv.kv_allocated_len)
            rt.kv.fill_request(slot, n, self.req_to_host_pool[slot, :n],
                               stream=self.write_staging_stream if torch.cuda.is_available() else None, req=req)

    def preload(self, req):
        rt = _runtime()
        if rt is None:
            return originals["_preload_to_device_buffer"](self, req)
        slot, n = req.kv.req_pool_idx, self.host_token_len(req.kv.kv_allocated_len)
        rt.kv.preload(self.req_to_host_pool[slot, :n], self.req_to_device_buffer[slot, :n])

    def request_finished(self, req):
        rt = _runtime()
        if rt is not None:
            for event, held in [f for f in rt.kv.fills if f[1] is req]:
                event.synchronize()     # the fill writes rows this release hands back
                rt.kv.fills.remove((event, held))
        originals["request_finished"](self, req)
        if rt is not None:
            rt.kv.finished(req.kv.req_pool_idx)

    hooks = dict(__init__=init, _eager_backup_previous_token=eager_backup,
                 admit_request_into_staging=admit_request_into_staging, alloc_device_buffer=alloc_device_buffer,
                 admit_request_direct=admit_request_direct, _preload_to_device_buffer=preload,
                 request_finished=request_finished)
    for name, hook in hooks.items():
        originals[name] = _patch(state, cls, name, hook)
    return module


def _context_errors():
    """Execution the elastic path is built for: single-stream decode, eager prefill, no speculation."""
    context = importlib.import_module("sglang.srt.runtime_context")
    errors = []
    schedule, execution, memory = context.get_schedule(), context.get_exec(), context.get_memory()
    disagg = context.get_disagg()
    if (not schedule.disable_overlap_schedule or execution.overlap.enable_two_batch_overlap
            or execution.overlap.enable_single_batch_overlap or not memory.disable_radix_cache
            or context.get_spec().speculative_algorithm not in (None, "NONE")
            or disagg.disaggregation_mode not in ("null", "decode") or disagg.enable_pdmux
            or context.get_parallel().pp_size != 1):
        errors.append("requires single-stream decode: no overlap, Radix, speculation, prefill-side "
                      "disaggregation, PD multiplexing or PP")
    graph = execution.graph.cuda_graph_config
    if graph.decode.backend not in ("disabled", "full"):
        errors.append("decode must be eager or full CUDA Graph")
    if graph.prefill.backend != "disabled":
        errors.append("prefill must run eagerly: its stores follow the layer modes of the moment")
    return errors


def _model_errors(state, c, cfg):
    pool, errors = state.get("pool"), []
    if pool is None:
        return ["the HiSparse pool was built before install(); start a fresh worker"]
    if cfg.model == "minimax_m3" and (not c.is_m3_hisparse or c.mem_pool_device is not pool.main_pool):
        errors.append("the MiniMax binding needs the M3 HiSparse coordinator over the elastic main pool")
    if cfg.model == "dsa" and (c.is_m3_hisparse or c.is_dsv4_hisparse or c.mem_pool_device is not pool):
        errors.append("the DSA binding needs the HiSparse DSA coordinator over the elastic pool")
    if c.spec_swap.enabled or c.skip_io:
        errors.append("speculative decoding and the skip-IO probe are unsupported")
    if c.req_device_buffer_tokens.shape[0] != len(state["eligible"]):
        errors.append("swap tables must cover exactly the elastic layers")
    if cfg.feature("compaction") and cfg.model == "minimax_m3":
        subs = [p for p in (pool.dense_pool, pool.index_kv_pool, pool.index_k_pool) if p is not None]
        if any(getattr(p, "kv_cache_layout", "nhd") != "nhd" for p in subs):
            errors.append("compaction moves token rows of the dense and index pools; they must be NHD")
    if cfg.feature("compaction") and cfg.model == "dsa" and getattr(pool, "index_kpool", 1) != 1:
        errors.append("compaction relocates indexer keys per token, which needs index_kpool == 1")
    return errors


def _movers(state, cfg):
    """Stock relocations of the logical-location pools outside the elastic buffers (compaction)."""
    pool = state["pool"]
    if cfg.model == "dsa":
        return [pool.index_key_cache.move]
    tensors = [t for p in (pool.dense_pool, pool.index_kv_pool, pool.index_k_pool) if p is not None
               for t in (*p.k_buffer, *getattr(p, "v_buffer", []))]

    def move(dst, src):
        for tensor in tensors:
            tensor[dst] = tensor[src]
    return [move]

def _bind(state, c, *, torch, dist, module):
    """Gate, config resolution, TP agreement and the Runtime of one coordinator; runs before any request."""
    if state.get("runtime") is not None:
        raise RuntimeError("one HiSparse coordinator per worker process")
    cfg = state["config"]
    errors = _context_errors() + _model_errors(state, c, cfg)
    if errors:
        raise RuntimeError("elastic HiSparse incompatible: " + "; ".join(errors))
    eligible = state["eligible"]
    first = eligible[0]
    followers = [layer for layer in eligible if c.enable_prefetch and c._is_shared_index_layer[layer - first]]
    cost = ep.CostModel(ep.load_profile(cfg, REPO_ROOT))
    bins = [int(k) for k in cost.p["miss_rate"]] or [32768]
    bound = cfg.resolve(eligible, followers=followers,
                        miss_rate=lambda layer: sum(cost.miss_rate(layer, b) for b in bins) / len(bins))
    owner = {skip: anchor for anchor, skips in c._prefetch_groups.items() for skip in skips} \
        if c.enable_prefetch else {}
    views, parts = state["views"], ("k", "v") if cfg.model == "minimax_m3" else ("kv",)
    offsets = [views[(layer, parts[0])]["hot_offset"] for layer in eligible]
    if any(views[(layer, part)]["hot_offset"] != offsets[i] for i, layer in enumerate(eligible) for part in parts):
        raise RuntimeError("K and V rows of a layer differ in size; one hot offset per layer is required")
    signature = dict(model=cfg.model, eligible=list(eligible), managed=list(bound.managed), offsets=offsets,
                     chunks=state["physical_chunks"], config=bound.digest)
    identity = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    hashes = [None] * c.tp_world_size
    dist.all_gather_object(hashes, identity, group=c.tp_group)
    if hashes != [identity] * c.tp_world_size:
        raise RuntimeError("TP ranks disagree on the elastic layers or the chunk pool")

    def make_kv(rt):
        if cfg.model == "minimax_m3":
            rows = ekv.MHALayers(torch, [views[(layer, "k")]["full"] for layer in eligible],
                                 [views[(layer, "v")]["full"] for layer in eligible], c.mem_pool_host)
        else:
            rows = ekv.MLALayers(torch, [views[(layer, "kv")]["full"] for layer in eligible], c.mem_pool_host)
        return ekv.ElasticKV(rows, c, layers=eligible, managed=bound.managed, hot_offset=offsets,
                             offloaded=bound.offloaded(rt.level), torch=torch, device=torch.device(c.device),
                             ratio=c.compress_ratio, owner=owner, through=bound.feature("write_through"))

    def compaction_buffers(rt):
        inactive = set(rt.cfg.offloaded(rt.level)) - set(rt._filling())
        return ([(v["full"], v["hot_offset"]) for (layer, _), v in views.items() if layer not in inactive],
                _movers(state, cfg))

    state.update(config=bound, max_running=c.req_to_token_pool.req_to_token.shape[0] - 1, make_kv=make_kv,
                 compaction_buffers=compaction_buffers, repo_root=REPO_ROOT, signature=signature)
    state["runtime"] = rt = Runtime(state, c, torch=torch, dist=dist)
    if bound.mode == "smooth" and bound.feature("fused_route") and torch.cuda.is_available():
        from . import elastic_route
        checks = [elastic_route.self_check(torch, rt.kv.device, ratio=c.compress_ratio)]
        if cfg.model == "minimax_m3":
            checks.append(elastic_route.self_check(torch, rt.kv.device, block=128))
        if not all(checks):
            raise RuntimeError("the fused smooth route disagrees with its torch statement on this device; "
                               "set features.fused_route to false")
        rt.kv.fused = elastic_route
    if cfg.model == "minimax_m3":
        _minimax_backend(state, torch)
    else:
        _dsa_backend(state, torch)
    emit(_bound_receipt(state, c, rt), rt.rank)


def _bound_receipt(state, c, rt):
    cfg, layout = rt.cfg, rt.memory.layout
    return dict(event="hisparse_elastic_bound", tp_size=c.tp_world_size, model=cfg.model, mode=cfg.mode,
                config_file_sha256=state["digest"], config_sha256=cfg.digest, resolved_config=cfg.as_dict(),
                chunk_bytes=layout.chunk, physical_chunks=state["physical_chunks"], stock_bytes=state["stock_bytes"],
                levels=ep.describe_levels(cfg, rt.controller.geometry, rt.hot.tokens()),
                start=dict(level=rt.level, tokens=rt.logical.tokens(), hot_rows=rt.hot.tokens()),
                signature=state["signature"], pending_hardware=list(PENDING_HARDWARE))


# ---- scheduler, graphs and the operator pin ----------------------------------------------------------------
def _scheduler(state):
    """The controller ticks between batches, on every TP rank, in both the normal and the PD decode loop."""
    scheduler_cls = importlib.import_module("sglang.srt.managers.scheduler").Scheduler
    mixin = importlib.import_module("sglang.srt.disaggregation.decode").SchedulerDisaggregationDecodeMixin

    def next_batch(self, running_batch, last_batch):
        rt = _runtime()
        if rt is not None:
            rt.tick(self, last_batch)
        return original_next(self, running_batch, last_batch)

    def next_disagg_batch(self, running_batch):
        rt = _runtime()
        if rt is not None:
            rt.tick(self)
        return original_disagg(self, running_batch)

    original_next = _patch(state, scheduler_cls, "get_next_batch_to_run", next_batch)
    original_disagg = _patch(state, mixin, "get_next_disagg_decode_batch_to_run", next_disagg_batch)


def _graphs(state):
    """Steady mode keys the decode runner's graphs by level and captures every precapture level at start-up."""
    runner_cls = importlib.import_module("sglang.srt.model_executor.runner.decode_cuda_graph_runner") \
        .DecodeCudaGraphRunner
    backend_cls = importlib.import_module("sglang.srt.model_executor.runner_backend.full_cuda_graph_backend") \
        .FullCudaGraphBackend
    originals = {}

    def capture(self):
        rt = _runtime()
        if rt is None or getattr(self.model_runner, "is_draft_worker", False):
            return originals["capture"](self)
        state["graph_runner"], state["stock_capture"] = self, originals["capture"]
        if rt.mode != "steady":
            return originals["capture"](self)
        return rt.precapture(self)

    def key(backend, shape_key):
        rt = _runtime()
        mine = rt is not None and backend._cuda_graph_runner is state.get("graph_runner")
        return rt.graph_key(shape_key) if mine else shape_key

    def capture_one(self, shape_key, forward_fn, *args, **kwargs):
        return originals["capture_one"](self, key(self, shape_key), forward_fn, *args, **kwargs)

    def can_run(self, forward_batch, shape_key):
        return originals["can_run"](self, forward_batch, key(self, shape_key))

    def replay(self, shape_key, static_forward_batch, **kwargs):
        return originals["replay"](self, key(self, shape_key), static_forward_batch, **kwargs)

    originals["capture"] = _patch(state, runner_cls, "capture", capture)
    for name, hook in (("capture_one", capture_one), ("can_run", can_run), ("replay", replay)):
        originals[name] = _patch(state, backend_cls, name, hook)

def _trigger(state):
    """``hisparse_elastic_level`` in POST /set_internal_state pins every TP rank to a level (null: controller)."""
    module = importlib.import_module("sglang.srt.managers.scheduler_components.internal_state")
    controller_cls = module.SchedulerInternalStateController

    def validate(key, value):
        rt = _runtime()
        if rt is None:
            raise module.UpdateRejected(f"{key} needs a worker bound to elastic HiSparse")
        if value is not None and (type(value) is not int or not 0 <= value <= rt.cfg.max_level):
            raise module.UpdateRejected(f"{key} must be null (controller) or a level 0..{rt.cfg.max_level}")

    def apply(key, value):
        rt = _runtime()
        rt.pinned_level = value
        emit(dict(event="hisparse_elastic_pin", level=value, step=rt.steps), rt.rank)

    def update_handlers(self):
        handlers = original(self)
        handlers[LEVEL_KEY] = module._UpdateHandler(validate, apply)
        return handlers

    original = _patch(state, controller_cls, "_update_handlers", update_handlers)


# ---- install ------------------------------------------------------------------------------------------------
def elastic_digests():
    """Pinned SGLang inputs of this adapter: the issue #11 manifest plus ``elastic_dependency_sha256``."""
    return {**pinned_digests(), **json.loads(PROVENANCE.read_text())["elastic_dependency_sha256"]}


def verify_digests(root):
    return {rel: ("missing" if not (Path(root) / rel).is_file() else
                  "ok" if hashlib.sha256((Path(root) / rel).read_bytes()).hexdigest() == expected else "mismatch")
            for rel, expected in sorted(elastic_digests().items())}


def install(config, config_sha256):
    """Patch the pinned symbols in this worker before the KV pools, coordinator and backends exist. A process
    keeps one frozen configuration; a different one needs a new process."""
    global _STATE
    if _STATE is not None:
        if _STATE["digest"] != config_sha256:
            raise RuntimeError("the elastic configuration is frozen for this worker")
        return True
    if config.model == "deepseek_v4":
        raise NotImplementedError("elastic DeepSeek-V4 C4 layers are not wired yet; use the issue #11 "
                                  "dynamic_offload adapter for DeepSeek-V4 (see docs/implementation)")
    import torch
    import torch.distributed as dist
    import sglang
    root = Path(sglang.__file__).resolve().parent
    bad = {rel: status for rel, status in verify_digests(root).items() if status != "ok"}
    broken = [r for r in resolve_symbols(root, config.model) if r["status"] != "ok"]
    if bad or broken:
        raise RuntimeError(f"pinned SGLang mismatch: {bad or broken}")
    state = dict(config=config, digest=config_sha256, patched=[], runtime=None,
                 chunk_bytes=config.memory["chunk_mib"] << 20)
    _STATE = state
    try:
        (_minimax_pool if config.model == "minimax_m3" else _dsa_pool)(state, torch)
        module = _coordinator(state, torch, dist)
        if config.model == "dsa":
            _dsa_coordinator(state, torch, module)
        _scheduler(state)
        _graphs(state)
        _trigger(state)
    except BaseException:
        _restore(state)
        _STATE = None
        raise
    return True


def uninstall():
    """Restore the pinned symbols, only while no pool or coordinator was built under the hooks."""
    global _STATE
    if _STATE is None:
        return False
    if _STATE.get("pool") is not None or _STATE.get("runtime") is not None:
        raise RuntimeError("elastic state exists in this process; start a new process to restore stock behavior")
    _restore(_STATE)
    _STATE = None
    return True

# ---- CPU dry run --------------------------------------------------------------------------------------------
# Planning geometry per model (one TP rank); a worker re-derives every value from its real pools.
PRESETS = {
    "minimax_m3": dict(eligible="0-59", device_tokens=396_544, ratio=2, hot_slots=4096, slots=129, page=128,
                       row_bytes=256, buffers=2, followers=""),
    "dsa": dict(eligible="0-77", device_tokens=196_608, ratio=2, hot_slots=2048, slots=129, page=64,
                row_bytes=1152, buffers=1, followers=""),
}


def plan_layout(config, *, eligible, device_tokens, ratio, hot_slots, slots, page, row_bytes, buffers, **_):
    """The Layout and chunk pool a worker with this geometry would build (``Factory`` without a device)."""
    layout = em.Layout([], chunk_bytes=config.memory["chunk_mib"] << 20, max_tokens=device_tokens * ratio,
                       hot_rows=device_tokens, page_tokens=page)
    specs = [em.BufferSpec(f"{part}{layer}", row_bytes, em.SWITCHABLE, layer=layer, pad_rows=page)
             for layer in eligible for part in ("k", "v")[:buffers]]
    layout.specs = tuple(specs)
    chunks = len(specs) * (device_tokens + page) * row_bytes // layout.chunk
    return layout, ep.Geometry(layout, chunks, hot_slots + page, slots - 1, hot_min=hot_slots + page)


def dry_run(root, config, model, geo):
    """Pinned-source check and per-level capacity of one rank: no torch, model or GPU."""
    from .dynamic_offload_device import parse_layers
    eligible = parse_layers(geo["eligible"])
    followers = parse_layers(geo["followers"]) if geo["followers"] else ()
    cost = ep.CostModel(ep.load_profile(config, REPO_ROOT))
    bins = [int(k) for k in cost.p["miss_rate"]] or [32768]
    bound = config.resolve(eligible, followers=followers,
                           miss_rate=lambda layer: sum(cost.miss_rate(layer, b) for b in bins) / len(bins))
    layout, geometry = plan_layout(bound, **dict(geo, eligible=eligible))
    digests, symbols = verify_digests(root), resolve_symbols(root, model)
    ok = all(v == "ok" for v in digests.values()) and all(s["status"] == "ok" for s in symbols)
    hot = geometry.hot_per_request * min(geometry.max_running, 64)
    return dict(status="pass" if ok else "fail", model=model, config_sha256=bound.digest,
                resolved_config=bound.as_dict(), digests=digests, symbols=symbols,
                geometry={k: v for k, v in geo.items()}, chunks=geometry.chunks, hot_rows_at_64_requests=hot,
                levels=ep.describe_levels(bound, geometry, hot),
                trigger=f'POST /set_internal_state {{"server_args": {{"{LEVEL_KEY}": <level or null>}}}}',
                pending_hardware=list(PENDING_HARDWARE))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Elastic per-layer HiSparse: CPU dry run against pinned SGLang.")
    parser.add_argument("--config", type=Path, required=True, help="hisparse-elastic-v1 config")
    parser.add_argument("--sglang-src", type=Path, required=True, help="pinned checkout or its python/sglang")
    for key, value in PRESETS["minimax_m3"].items():
        parser.add_argument("--" + key.replace("_", "-"), type=str if key in ("eligible", "followers") else int,
                            default=None, help=f"default per model, e.g. {value!r}")
    args = parser.parse_args(argv)
    try:
        config, _ = ec.read(args.config)
        if config.model not in PRESETS:
            raise ValueError(f"no dry-run preset for {config.model}")
        geo = {key: getattr(args, key) if getattr(args, key) is not None else value
               for key, value in PRESETS[config.model].items()}
        root = args.sglang_src / "python" / "sglang" if (args.sglang_src / "python" / "sglang").is_dir() \
            else args.sglang_src
        report = dry_run(root, config, config.model, geo)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
