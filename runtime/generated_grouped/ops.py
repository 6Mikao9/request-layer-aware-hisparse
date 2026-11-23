# Modified from pinned SGLang (Apache-2.0); see ../NOTICE and ../LICENSE.
from __future__ import annotations
import functools
from pathlib import Path
import torch
from sglang.kernels.jit.utils import make_cpp_args, load_jit
@functools.cache
def _jit_sparse_module(
    item_size_bytes: int,
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    is_mla: bool = False,
    is_dsv4_layout: bool = False,
    top_k_block_size: int = 1,
    top_k_is_blocks: bool = False,
    record_miss_plan: bool = False,
    skip_io: bool = False,
) -> Module:
    # record_miss_plan / skip_io are compile-time kernel flags; the
    # (False, False) production instantiation stays byte-identical.
    template_args = make_cpp_args(
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla,
        is_dsv4_layout,
        top_k_block_size,
        top_k_is_blocks,
        record_miss_plan,
        skip_io,
    )
    cache_args = make_cpp_args(
        item_size_bytes,
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla,
        is_dsv4_layout,
        top_k_block_size,
        top_k_is_blocks,
        record_miss_plan,
        skip_io,
    )
    return load_jit(
        "request_type_layer_hisparse_v1",
        *cache_args,
        cuda_files=[str(Path(__file__).with_name("hisparse_active.cuh"))],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer",
                f"load_cache_to_device_buffer<{template_args}>",
            )
        ],
    )

def load_blocks_to_device_buffer_mha(
    top_k_blocks: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache_k: torch.Tensor,
    host_cache_v: torch.Tensor,
    device_buffer_k: torch.Tensor,
    device_buffer_v: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    hot_buffer_size: int,
    sparse_block_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
    skip_io: bool = False,
    active_hot_buffer_size: int | None = None,
    assigned_capacities: torch.Tensor | None = None,
) -> None:
    """Swap block-selected MHA K/V into the HiSparse device pool."""
    from runtime.policy import validate_capacity, validate_metadata_shapes, validate_metadata_strides
    active_hot_buffer_size = hot_buffer_size if active_hot_buffer_size is None else active_hot_buffer_size
    validate_capacity(active_hot_buffer_size, hot_buffer_size, top_k_blocks.size(1), sparse_block_size)
    validate_metadata_shapes(tuple(lru_slots.shape), tuple(device_buffer_tokens.shape), tuple(device_buffer_locs.shape), hot_buffer_size)
    validate_metadata_strides(tuple(lru_slots.stride()), tuple(device_buffer_tokens.stride()), tuple(device_buffer_locs.stride()))
    if (assigned_capacities is None or assigned_capacities.dtype != torch.int32
            or tuple(assigned_capacities.shape) != (device_buffer_tokens.shape[0],)
            or not assigned_capacities.is_contiguous()
            or assigned_capacities.device != top_k_blocks.device):
        raise ValueError("Expected contiguous device int32 capacity per request slot")
    if skip_io:
        raise ValueError("Request routing requires the fused copy path")
    num_top_k_blocks = top_k_blocks.size(1)
    num_top_k_tokens = num_top_k_blocks * sparse_block_size
    assert hot_buffer_size >= num_top_k_tokens, (
        f"hot_buffer_size ({hot_buffer_size}) must be >= selected tokens "
        f"({num_top_k_tokens})"
    )
    assert top_k_device_locs.size(1) >= num_top_k_tokens
    k_stride = host_cache_k.stride(0) * host_cache_k.element_size()
    v_stride = host_cache_v.stride(0) * host_cache_v.element_size()
    assert k_stride == v_stride == item_size_bytes, (
        "K/V token strides must equal item_size_bytes: "
        f"k_stride={k_stride}, v_stride={v_stride}, "
        f"item_size_bytes={item_size_bytes}"
    )

    module = _jit_sparse_module(
        item_size_bytes,
        block_size,
        num_top_k_blocks,
        active_hot_buffer_size,
        is_mla=False,
        is_dsv4_layout=False,
        top_k_block_size=sparse_block_size,
        top_k_is_blocks=True,
        record_miss_plan=False,
        skip_io=skip_io,
    )
    empty = torch.empty(0, device=top_k_blocks.device)
    if num_real_reqs is None:
        num_real_reqs = torch.tensor(
            [top_k_blocks.size(0)], dtype=torch.int32, device=top_k_blocks.device
        )

    module.load_cache_to_device_buffer(
        top_k_blocks,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache_k,
        host_cache_v,
        device_buffer_k,
        device_buffer_v,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
        page_size,
        item_size_bytes,
        empty,
        empty,
        empty,
        assigned_capacities,
    )
