"""Opt-in same-stream MHA planning and copy using pinned SGLang templates.

The source of both kernels is recorded in PROVENANCE.json/NOTICE. This module
does not modify either template, create a side stream, or change physical Hmax.
The extra copy launch is useful only for some small-batch miss distributions;
the immutable routing config must be selected on training measurements.
"""
from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class SplitCopyConfig:
    layers: tuple[int, ...]
    copy_grid: int = 64
    plan_rows: int = 2

    def __post_init__(self):
        if (type(self.layers) is not tuple or not self.layers
                or any(type(x) is not int or not 0 <= x < 60 for x in self.layers)
                or self.layers != tuple(sorted(set(self.layers)))):
            raise ValueError('Split layers must be a sorted unique tuple of absolute layers in [0,60)')
        if type(self.copy_grid) is not int or self.copy_grid != 64:
            raise ValueError('The validated split-copy grid is 64')
        if type(self.plan_rows) is not int or self.plan_rows != 2:
            raise ValueError('The initial split-copy prototype supports capture batches 1 and 2')

    @classmethod
    def from_json(cls, value):
        if isinstance(value, str):
            def unique(pairs):
                result = dict(pairs)
                if len(result) != len(pairs):
                    raise ValueError('Duplicate split configuration key')
                return result
            value = json.loads(value, object_pairs_hook=unique)
        if (type(value) is not dict or set(value) != {'layers', 'copy_grid', 'plan_rows'}
                or type(value['layers']) is not list):
            raise ValueError('Split config requires layers array, copy_grid and plan_rows')
        return cls(tuple(value['layers']), value['copy_grid'], value['plan_rows'])

    def as_dict(self):
        return dict(layers=list(self.layers), copy_grid=self.copy_grid, plan_rows=self.plan_rows)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()

    def uses_split(self, layer, captured_batch):
        # This is a CPU shape decision, frozen when a CUDA graph is captured.
        # Never read the device num_real_reqs tensor to choose a Python branch.
        return layer in self.layers and 1 <= captured_batch <= self.plan_rows


def execution_identity(policy, config):
    """One collective identity for both fused and split ranks (no extra barrier)."""
    return hashlib.sha256(json.dumps(dict(policy=policy.digest,
        split=None if config is None else config.as_dict()), sort_keys=True).encode()).hexdigest()


class SplitCopyState:
    """Fixed plan storage, allocated once before capture; sequential layer reuse."""
    def __init__(self, config, device):
        import torch
        self.config = config
        self.src = torch.empty((config.plan_rows, 2048), dtype=torch.int64, device=device)
        self.dst = torch.empty((config.plan_rows, 2048), dtype=torch.int32, device=device)
        self.counts = torch.empty(config.plan_rows, dtype=torch.int32, device=device)

    def load(self, *, top_k_blocks, device_buffer_tokens, host_cache_locs,
             device_buffer_locs, host_cache_k, host_cache_v, device_buffer_k,
             device_buffer_v, top_k_device_locs, req_pool_indices, seq_lens,
             lru_slots, item_size_bytes, hot_buffer_size, active_hot_buffer_size,
             sparse_block_size, page_size=1, block_size=960, num_real_reqs=None,
             skip_io=False):
        import torch
        from .policy import validate_capacity, validate_metadata_shapes, validate_metadata_strides
        from .generated.ops import _jit_sparse_module
        from sglang.kernels.ops.kvcache.hisparse import _jit_copy_planned_module
        batch, width = tuple(top_k_blocks.shape)
        if (not 1 <= batch <= self.config.plan_rows or width != 16
                or sparse_block_size != 128 or page_size != 1 or block_size != 960
                or item_size_bytes != 256 or skip_io or num_real_reqs is None):
            raise ValueError('Split copy requires the pinned BF16 TP4 MHA specialization and explicit real count')
        validate_capacity(active_hot_buffer_size, hot_buffer_size, width, sparse_block_size)
        validate_metadata_shapes(tuple(lru_slots.shape), tuple(device_buffer_tokens.shape),
                                 tuple(device_buffer_locs.shape), hot_buffer_size)
        validate_metadata_strides(tuple(lru_slots.stride()), tuple(device_buffer_tokens.stride()),
                                  tuple(device_buffer_locs.stride()))
        if (top_k_device_locs.size(0) != batch or top_k_device_locs.size(1) < 2048
                or req_pool_indices.size(0) != batch or seq_lens.size(0) != batch
                or any(x.dtype != torch.bfloat16 for x in (host_cache_k, host_cache_v, device_buffer_k, device_buffer_v))
                or any(x.stride(0) * x.element_size() != item_size_bytes
                       for x in (host_cache_k, host_cache_v, device_buffer_k, device_buffer_v))
                or self.src.device != top_k_blocks.device
                or num_real_reqs.dtype != torch.int32 or num_real_reqs.numel() != 1
                or num_real_reqs.device != self.src.device):
            raise ValueError('Split copy tensor layout/device differs from the validated specialization')
        planner = _jit_sparse_module(item_size_bytes, block_size, width, active_hot_buffer_size,
            is_mla=False, is_dsv4_layout=False, top_k_block_size=sparse_block_size,
            top_k_is_blocks=True, record_miss_plan=True, skip_io=True)
        copier = _jit_copy_planned_module(256, False, False, False)
        # Both templates use TVMFFIEnvGetStream. Completion of the copy precedes
        # attention and the next layer's overwrite of these plan buffers.
        planner.load_cache_to_device_buffer(top_k_blocks, device_buffer_tokens,
            host_cache_locs, device_buffer_locs, host_cache_k, host_cache_v,
            device_buffer_k, device_buffer_v, top_k_device_locs, req_pool_indices,
            seq_lens, lru_slots, num_real_reqs, page_size, item_size_bytes,
            self.src, self.dst, self.counts)
        copier.copy_cache_planned(self.src, self.dst, self.counts, num_real_reqs,
            host_cache_k, host_cache_v, device_buffer_k, device_buffer_v,
            self.config.copy_grid, item_size_bytes)


def bind(coordinator, policy, config):
    import torch
    from sglang.srt.runtime_context import get_device, get_disagg, get_exec, get_memory, get_schedule, get_spec
    if (policy.expected_tp_size != 4 or coordinator.mem_pool_device.page_size != 128
            or coordinator.item_size_bytes != 256 or coordinator.swap_in_block_size != 960
            or coordinator.skip_io or not get_schedule().disable_overlap_schedule
            or get_exec().overlap.enable_two_batch_overlap
            or get_exec().overlap.enable_single_batch_overlap
            or not get_memory().disable_radix_cache
            or get_spec().speculative_algorithm not in (None, 'NONE')
            or get_disagg().disaggregation_mode != 'null' or get_disagg().enable_pdmux
            or torch.device(get_device().device).type != 'cuda'):
        raise RuntimeError('Split copy requires TP4 pinned MHA with no overlap/radix/spec/disaggregation')
    # Graph backend configuration is additionally checked by the explicit
    # launcher. Only eager prefill and eager/full decode are supported initially.
    graph = get_exec().graph.cuda_graph_config
    if graph.prefill.backend != 'disabled' or graph.decode.backend not in ('disabled', 'full'):
        raise RuntimeError('Split copy supports eager prefill and eager/full decode only')
    if graph.decode.backend == 'full' and list(graph.decode.bs) != [1, 2]:
        raise RuntimeError('Split copy initially requires exact decode capture buckets [1,2]')
    state = SplitCopyState(config, coordinator.req_device_buffer_tokens.device)
    coordinator._request_layer_hisparse_split = state
    print(json.dumps(dict(event='hisparse_split_copy_bound', rank=torch.distributed.get_rank(),
        config_sha256=config.digest, policy_sha256=policy.digest, **config.as_dict(),
        plan_src_shape=list(state.src.shape), plan_src_dtype=str(state.src.dtype),
        plan_dst_dtype=str(state.dst.dtype), count_dtype=str(state.counts.dtype))), flush=True)


def swap_in_selected_blocks(self, req_pool_indices, seq_lens, top_k_blocks, layer_id, sparse_block_size):
    from .generated.coordinator_method import swap_in_selected_blocks as fused
    state = self._request_layer_hisparse_split
    if not state.config.uses_split(layer_id, req_pool_indices.size(0)):
        return fused(self, req_pool_indices, seq_lens, top_k_blocks, layer_id, sparse_block_size)
    policy = self._request_layer_hisparse_policy
    if self.device_buffer_size != policy.hmax:
        raise ValueError('Physical Hmax changed after binding')
    count = top_k_blocks.size(1) * sparse_block_size
    output = self.top_k_device_locs_buffer[:req_pool_indices.size(0), :count]
    host_layer = layer_id - self.mem_pool_device.start_layer
    state.load(top_k_blocks=top_k_blocks, device_buffer_tokens=self.req_device_buffer_tokens[host_layer],
        host_cache_locs=self.req_to_host_pool, device_buffer_locs=self.req_device_buffer_token_locs[host_layer],
        host_cache_k=self.mem_pool_host.k_buffer[host_layer], host_cache_v=self.mem_pool_host.v_buffer[host_layer],
        device_buffer_k=self.mem_pool_device.get_key_buffer(layer_id), device_buffer_v=self.mem_pool_device.get_value_buffer(layer_id),
        top_k_device_locs=output, req_pool_indices=req_pool_indices, seq_lens=seq_lens,
        lru_slots=self.lru_slots[host_layer], item_size_bytes=self.item_size_bytes,
        hot_buffer_size=self.device_buffer_size, active_hot_buffer_size=policy.capacities[layer_id],
        sparse_block_size=sparse_block_size, page_size=1, block_size=self.swap_in_block_size,
        num_real_reqs=self.num_real_reqs, skip_io=self.skip_io)
    return output
