# Modified from pinned SGLang (Apache-2.0); see ../NOTICE and ../LICENSE.
from __future__ import annotations
import torch
from .ops import load_blocks_to_device_buffer_mha
def swap_in_selected_blocks(
    self,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k_blocks: torch.Tensor,
    layer_id: int,
    sparse_block_size: int,
) -> torch.Tensor:
    assert self.is_m3_hisparse
    policy = self._request_layer_hisparse_policy
    if self.device_buffer_size != policy.hmax:
        raise ValueError("Physical Hmax changed after binding")
    num_reqs = req_pool_indices.size(0)
    num_selected_tokens = top_k_blocks.size(1) * sparse_block_size
    assert num_selected_tokens <= self.device_buffer_size, (
        f"MiniMax M3 selected {num_selected_tokens} tokens, but the "
        f"HiSparse device buffer holds only {self.device_buffer_size}."
    )
    top_k_indices = self.top_k_device_locs_buffer[:num_reqs, :num_selected_tokens]
    host_layer = layer_id - self.mem_pool_device.start_layer
    load_blocks_to_device_buffer_mha(
        top_k_blocks=top_k_blocks,
        device_buffer_tokens=self.req_device_buffer_tokens[host_layer],
        host_cache_locs=self.req_to_host_pool,
        device_buffer_locs=self.req_device_buffer_token_locs[host_layer],
        host_cache_k=self.mem_pool_host.k_buffer[host_layer],
        host_cache_v=self.mem_pool_host.v_buffer[host_layer],
        device_buffer_k=self.mem_pool_device.get_key_buffer(layer_id),
        device_buffer_v=self.mem_pool_device.get_value_buffer(layer_id),
        top_k_device_locs=top_k_indices,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=self.lru_slots[host_layer],
        item_size_bytes=self.item_size_bytes,
        hot_buffer_size=self.device_buffer_size,
        active_hot_buffer_size=policy.capacities[layer_id],
        sparse_block_size=sparse_block_size,
        page_size=1,
        block_size=self.swap_in_block_size,
        num_real_reqs=self.num_real_reqs,
        skip_io=self.skip_io,
    )
    return top_k_indices
