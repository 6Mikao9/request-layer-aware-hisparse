"""Explicit opt-in Hybrid BF16 physical-row attention for SM80.

Requires the separately audited SM80 indexer/model kernels. Does not replace
Hybrid writes, ownership, resolve, gather, stream fences or retirement.
"""
import torch
from vllm.config import get_current_vllm_config
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend, FlashMLASparseImpl, FlashMLASparseMetadata,
)
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonMetadataBuilder, SparseMLAPrefillMetadata,
)
from vllm.v1.hisparse.runtime import build_hisparse_prefill_staging_plan
from vllm.v1.attention.ops.hybrid_sm80_abi import make_bf16_method
from vllm.v1.attention.ops.triton_mla_sparse_kernel import triton_mla_sparse_attention


def validate_scope(config):
    if not config.model_config.enforce_eager:
        raise ValueError("HYBRID_TRITON_SM80 first gate requires enforce_eager")
    parallel = config.parallel_config
    if (parallel.pipeline_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1):
        raise ValueError("Only PP1/DCP1/PCP1 supported")
    if config.speculative_config is not None or config.kv_transfer_config is not None:
        raise ValueError("No speculative or disaggregated KV mode")
    if config.cache_config.cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("BF16 main KV only")
    if config.model_config.dtype != torch.bfloat16:
        raise ValueError("BF16 activations required")
    hf = config.model_config.hf_text_config
    if (hf.kv_lora_rank, hf.qk_rope_head_dim, hf.index_topk, hf.index_head_dim) != (512, 64, 2048, 128):
        raise ValueError("Requires MLA512+64/K2048/index128 geometry")
    if config.cache_config.enable_prefix_caching:
        raise ValueError("First SM80 gate requires prefix caching disabled")


class HybridTritonSM80Backend(FlashMLASparseBackend):
    supported_kv_cache_dtypes = ["auto", "bfloat16"]

    @staticmethod
    def get_name(): return "HYBRID_TRITON_SM80"

    @staticmethod
    def get_impl_cls(): return HybridTritonSM80Impl

    @staticmethod
    def get_builder_cls(): return HybridTritonSM80MetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability):
        return (capability.major, capability.minor) == (8, 0)


class HybridTritonSM80MetadataBuilder(SparseMLACommonMetadataBuilder[FlashMLASparseMetadata]):
    metadata_cls = FlashMLASparseMetadata
    _cudagraph_support = AttentionCGSupport.NEVER
    require_uniform_decodes = True
    hisparse_supports_multi_token_decode = False

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        validate_scope(vllm_config)
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        attach_mqa_prefill(self, metadata, common_attn_metadata)
        return metadata


def attach_mqa_prefill(builder, metadata, common):
    """Create real staging metadata even with no dense-attention backend.

    Pinned common builder creates this only inside its dense-backend branch.
    CPU lengths are canonical scheduler metadata, not GPU readback.
    """
    if metadata.num_prefills == 0 or metadata.prefill is not None:
        return
    if builder.vllm_config.attention_config.hisparse_config is None:
        return
    nd, np = metadata.num_decodes, metadata.num_prefills
    lengths = common.seq_lens_cpu_upper_bound
    if lengths is None:
        raise ValueError("Canonical CPU sequence lengths required for staging")
    lengths = lengths[nd:nd + np]
    bs = builder.kv_cache_spec.block_size
    capacity = int(((lengths + bs - 1) // bs).sum())
    table = common.block_table_tensor[nd:nd + np]
    plan = build_hisparse_prefill_staging_plan(
        table, common.seq_lens[nd:nd + np], bs, capacity)
    qsl, max_qlen, qlens = builder._build_prefill_fields(common, nd, np)
    metadata.prefill_max_seq_len = int(lengths.max())
    metadata.prefill = SparseMLAPrefillMetadata(
        block_table=table, query_start_loc=qsl, max_query_len=max_qlen,
        query_lens_cpu=qlens, chunked_context=None,
        q_data_type=builder.model_config.dtype, output_dtype=builder.model_config.dtype,
        prefill_backend=None, use_dense_mha=False, topk_mask_workspace=None,
        host_staging_plan=plan)


class HybridTritonSM80Impl(FlashMLASparseImpl):
    supports_dense_mha_prefill = False
    can_return_lse_for_decode = False

    def __init__(self, *args, **kwargs):
        cfg = get_current_vllm_config()
        validate_scope(cfg)
        cap = current_platform.get_device_capability()
        if cap is None or not HybridTritonSM80Backend.supports_compute_capability(cap):
            raise ValueError("HYBRID_TRITON_SM80 requires actual SM80")
        super().__init__(*args, **kwargs)

    _bf16_flash_mla_kernel = make_bf16_method(triton_mla_sparse_attention)
