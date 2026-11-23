"""Explicit, eager-only same-state identity diagnostic; never a performance mode.

The real overlay commits once. Stock executes on private, stride-preserving
copies of the same device pre-state. Read-only pinned host K/V remain shared.
No tensor contents, model token IDs, or request identifiers are written.
"""

from functools import wraps
import json
import os
from pathlib import Path

from .policy import StaticPolicy

SCHEMA = "hisparse-same-state-identity-v1"
MUTABLE = ("device_buffer_tokens", "lru_slots", "top_k_device_locs",
           "device_buffer_k", "device_buffer_v")
READ_ONLY = ("top_k_blocks", "host_cache_locs", "device_buffer_locs",
             "req_pool_indices", "seq_lens", "num_real_reqs")
MAX_SCRATCH_BYTES = 512 * 1024**2
MIN_FREE_AFTER_COPY = 1024**3
_installed = None


def require_identity(policy):
    if not isinstance(policy, StaticPolicy) or any(h != policy.hmax for h in policy.capacities):
        raise ValueError("Same-state shadow requires an immutable all-Hmax identity policy")


def span_bytes(tensor):
    if tensor.numel() == 0:
        return 0
    if any(s < 0 for s in tensor.stride()):
        raise ValueError("Negative tensor strides are not supported")
    return (1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))) * tensor.element_size()


def byte_equal(torch, a, b):
    # Floating point equality treats equal NaN payloads as unequal. Unused KV
    # slots can contain arbitrary bits, so compare their exact bytes instead.
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


class ShadowEngine:
    def __init__(self, torch, directory, rank, tp_size, *, synchronize, allocated_bytes, free_bytes=None):
        self.torch = torch
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise ValueError("Shadow output directory must already exist")
        self.path = self.directory / f"rank-{rank}.json"
        # Claim this rank once, before any op. Never overwrite an older run.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        self.synchronize = synchronize
        self.allocated_bytes = allocated_bytes
        self.free_bytes = free_bytes
        self.scratch = {}
        self.layer_stats = {}
        self.live_rows = {}
        self.finished_rows = set()
        self.report = dict(schema=SCHEMA, rank=rank, tp_size=tp_size,
                           checked_calls=0, mismatches=0, layers=[], checks={},
                           scratch_span_bytes_max=0, observed_allocated_bytes_max=0,
                           scratch_limit_bytes=MAX_SCRATCH_BYTES, reserve_free_bytes=MIN_FREE_AFTER_COPY,
                           lifecycle=dict(allocations=0, finished=0, reused=0,
                                          reset_checks=0, reset_failures=0, unallocated_finish=0),
                           first_failures=[],
                           scope_note="Every actual identity-overlay swap call; stock on same-state scratch. "
                                      "Exact logical tensor bytes, preserved strides; pinned host KV shared read-only. "
                                      "Diagnostic synchronization and copying invalidate performance measurements.")
        self.save()

    def save(self):
        self.report["layers"] = [self.layer_stats[k] for k in sorted(self.layer_stats)]
        temporary = self.path.with_suffix(f".tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)

    def copy(self, key, source):
        target = self.scratch.get(key)
        if (target is None or target.shape != source.shape or target.stride() != source.stride()
                or target.dtype != source.dtype or target.device != source.device):
            new_span = span_bytes(source)
            expected_total = sum(span_bytes(v) for k, v in self.scratch.items() if k != key) + new_span
            if expected_total > MAX_SCRATCH_BYTES:
                raise RuntimeError("Same-state diagnostic exceeds its 512 MiB scratch limit")
            if self.free_bytes is not None and int(self.free_bytes()) < new_span + MIN_FREE_AFTER_COPY:
                raise RuntimeError("Same-state diagnostic would leave less than 1 GiB of free GPU memory")
            target = self.torch.empty_strided(source.shape, source.stride(), dtype=source.dtype,
                                             device=source.device)
            self.scratch[key] = target
        target.copy_(source)
        return target

    def record_check(self, name, passed, layer_id=None):
        entry = self.report["checks"].setdefault(name, {"checked": 0, "failed": 0})
        entry["checked"] += 1
        if not passed:
            entry["failed"] += 1
            self.report["mismatches"] += 1
            if len(self.report["first_failures"]) < 16:
                self.report["first_failures"].append(dict(check=name, layer_id=layer_id,
                                                          call=self.report["checked_calls"]))

    def allocated(self, row):
        # The actual coordinator allocation hook, not a guessed request boundary.
        self.report["lifecycle"]["allocations"] += 1
        self.record_check("allocation_without_live_row", row not in self.live_rows)
        if row in self.finished_rows:
            self.report["lifecycle"]["reused"] += 1
            self.finished_rows.remove(row)
        self.live_rows[row] = True

    def finished(self, coordinator, row):
        self.synchronize()
        self.report["lifecycle"]["finished"] += 1
        if self.live_rows.pop(row, False):
            t = self.torch
            passed = (bool(t.all(coordinator.req_device_buffer_tokens[:, row] == -1))
                      and bool(t.all(coordinator.req_device_buffer_token_locs[:, row] == -1))
                      and bool(t.all(coordinator.lru_slots[:, row] == coordinator._lru_init)))
            self.report["lifecycle"]["reset_checks"] += 1
            self.report["lifecycle"]["reset_failures"] += int(not passed)
            self.record_check("real_coordinator_reset", passed)
            if passed:
                self.finished_rows.add(row)
        else:
            self.report["lifecycle"]["unallocated_finish"] += 1
        self.save()

    def check(self, layer_id, kwargs, live, stock, *, final_layer):
        if kwargs.get("skip_io", False):
            raise RuntimeError("Shadow cannot validate a skip-IO path")
        if kwargs.get("active_hot_buffer_size") != kwargs["hot_buffer_size"]:
            raise RuntimeError("Shadow requires active capacity equal to physical Hmax")
        if kwargs.get("num_real_reqs") is None:
            raise RuntimeError("Expected the real coordinator's explicit active-row count")
        if not 0 <= layer_id < 60:
            raise RuntimeError("Unexpected absolute layer")
        self.synchronize()  # Finish host backup and live device writes before copying.
        shadow = dict(kwargs)
        shadow.pop("active_hot_buffer_size")
        for key in MUTABLE:
            shadow[key] = self.copy(key, kwargs[key])
        token_prestate = self.copy("before_device_buffer_tokens", kwargs["device_buffer_tokens"])
        snapshots = {}
        for key in READ_ONLY:
            snapshots[key] = self.copy("before_" + key, kwargs[key])
            shadow[key] = self.copy("shadow_" + key, kwargs[key])
        self.report["scratch_span_bytes_max"] = max(self.report["scratch_span_bytes_max"],
                                                    sum(span_bytes(v) for v in self.scratch.values()))
        # Both operations see identical pre-state. Only live commits to the model.
        live(**kwargs)
        stock(**shadow)
        self.synchronize()
        self.report["observed_allocated_bytes_max"] = max(self.report["observed_allocated_bytes_max"],
                                                          int(self.allocated_bytes()))
        self.report["checked_calls"] += 1
        stats = self.layer_stats.setdefault(layer_id, dict(layer_id=layer_id, checked=0,
                                                            seq_min=None, seq_max=None,
                                                            long_rows=0, short_rows=0,
                                                            token_replacement_calls=0))
        stats["checked"] += 1
        stats["token_replacement_calls"] += int(not byte_equal(self.torch,
                                                              kwargs["device_buffer_tokens"], token_prestate))
        num_real = int(kwargs["num_real_reqs"].item())
        if not 0 <= num_real <= kwargs["seq_lens"].numel():
            raise RuntimeError("Invalid real row count in actual model call")
        if num_real:
            seq = kwargs["seq_lens"][:num_real]
            lo, hi = int(seq.min().item()), int(seq.max().item())
            long_rows = int((seq > kwargs["hot_buffer_size"]).sum().item())
            stats["long_rows"] += long_rows
            stats["short_rows"] += num_real - long_rows
            stats["seq_min"] = lo if stats["seq_min"] is None else min(stats["seq_min"], lo)
            stats["seq_max"] = hi if stats["seq_max"] is None else max(stats["seq_max"], hi)
        before_failures = self.report["mismatches"]
        for key in MUTABLE:
            self.record_check(key, byte_equal(self.torch, kwargs[key], shadow[key]), layer_id)
        for key in READ_ONLY:
            self.record_check("live_readonly_" + key,
                              byte_equal(self.torch, kwargs[key], snapshots[key]), layer_id)
            self.record_check("stock_readonly_" + key,
                              byte_equal(self.torch, shadow[key], snapshots[key]), layer_id)
        # Exact full K/V and returned physical slots imply exact selected K/V.
        if layer_id == final_layer or self.report["mismatches"] != before_failures:
            self.save()


def install(policy, output_directory):
    """Register the real identity hook plus an isolated diagnostic in every worker."""
    require_identity(policy)
    return _install(policy, output_directory, "identity")


def install_selected(policy, output_directory):
    """Check selected KV semantics for a frozen, possibly nonuniform capacity table."""
    if not isinstance(policy, StaticPolicy):
        raise ValueError("Selection oracle requires a frozen StaticPolicy")
    return _install(policy, output_directory, "selected")


def _install(policy, output_directory, mode):
    global _installed
    directory = str(Path(output_directory).resolve(strict=True))
    identity = (policy.digest, directory, mode)
    if _installed is not None:
        if _installed != identity:
            raise RuntimeError("Shadow configuration is frozen in this process")
        return
    from .hook import install as install_hook
    install_hook(policy, enabled=True)
    import torch
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.kernels.ops.kvcache.hisparse import load_blocks_to_device_buffer_mha as stock
    from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.runtime_context import get_device, get_disagg, get_memory, get_schedule, get_spec
    from .generated import coordinator_method
    live = coordinator_method.load_blocks_to_device_buffer_mha
    original_swap = HiSparseCoordinator.swap_in_selected_blocks
    original_alloc = HiSparseCoordinator.alloc_device_buffer
    original_finish = HiSparseCoordinator.request_finished
    original_load = ModelRunner.load_model
    engine = None
    context = None
    validated_layouts = set()

    def guard():
        if not (cuda_graph_fully_disabled() and get_schedule().disable_overlap_schedule
                and get_memory().disable_radix_cache and get_spec().speculative_algorithm in (None, "NONE")
                and get_disagg().disaggregation_mode == "null"
                and torch.device(get_device().device).type == "cuda"):
            raise RuntimeError("Shadow requires resolved eager CUDA without overlap, radix, spec or disaggregation")

    @wraps(original_load)
    def load_model(self, *args, **kwargs):
        guard()
        return original_load(self, *args, **kwargs)

    def get_engine():
        nonlocal engine
        if engine is None:
            guard()
            rank = torch.distributed.get_rank()
            engine_class = ShadowEngine
            if mode == "selected":
                from .selection_oracle import SelectionOracle
                engine_class = SelectionOracle
            engine = engine_class(torch, directory, rank, policy.expected_tp_size,
                                  synchronize=torch.cuda.synchronize, allocated_bytes=torch.cuda.memory_allocated,
                                  free_bytes=lambda: torch.cuda.mem_get_info()[0])
        return engine

    @wraps(original_swap)
    def swap(self, req_pool_indices, seq_lens, top_k_blocks, layer_id, sparse_block_size):
        nonlocal context
        if context is not None or not self.is_m3_hisparse:
            raise RuntimeError("Shadow requires non-reentrant MiniMax MHA swap")
        context = (layer_id, self.mem_pool_device.end_layer - 1)
        try:
            return original_swap(self, req_pool_indices, seq_lens, top_k_blocks, layer_id, sparse_block_size)
        finally:
            context = None

    def checked_op(**kwargs):
        if context is None:
            raise RuntimeError("Identity op called outside the guarded coordinator method")
        tensors = {key: kwargs[key] for key in MUTABLE + READ_ONLY}
        layout = tuple((key, t.data_ptr(), tuple(t.shape), t.stride(), t.dtype) for key, t in tensors.items())
        if layout not in validated_layouts:
            if any(not t.is_cuda for t in tensors.values()):
                raise RuntimeError("Expected CUDA device state for the diagnostic")
            for key in ("device_buffer_k", "device_buffer_v", "host_cache_k", "host_cache_v"):
                tensor = kwargs[key]
                if (tensor.dtype != torch.bfloat16 or not tensor.is_contiguous()
                        or tensor.stride(0) * tensor.element_size() != kwargs["item_size_bytes"]):
                    raise RuntimeError("Diagnostic requires the inspected contiguous BF16 MHA KV layout")
                if key.startswith("host_") and (tensor.is_cuda or not tensor.is_pinned()):
                    raise RuntimeError("Diagnostic host KV must be pinned CPU memory")
            ranges = {key: (t.data_ptr(), t.data_ptr() + span_bytes(t)) for key, t in tensors.items()}
            for key in MUTABLE:
                a, b = ranges[key]
                for other, (c, d) in ranges.items():
                    if key != other and max(a, c) < min(b, d):
                        raise RuntimeError(f"Diagnostic cannot clone aliased state independently: {key}/{other}")
            validated_layouts.add(layout)
        get_engine().check(context[0], kwargs, live, stock, final_layer=context[1])

    @wraps(original_alloc)
    def allocated(self, req):
        result = original_alloc(self, req)
        get_engine().allocated(int(req.kv.req_pool_idx))
        return result

    @wraps(original_finish)
    def finished(self, req):
        row = int(req.kv.req_pool_idx)
        result = original_finish(self, req)
        get_engine().finished(self, row)
        return result

    ModelRunner.load_model = load_model
    HiSparseCoordinator.swap_in_selected_blocks = swap
    HiSparseCoordinator.alloc_device_buffer = allocated
    HiSparseCoordinator.request_finished = finished
    coordinator_method.load_blocks_to_device_buffer_mha = checked_op
    _installed = identity
