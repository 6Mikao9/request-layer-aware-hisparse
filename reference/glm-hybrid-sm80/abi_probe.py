"""Uninstalled ABI seam prototype. No registry edits, no claim of serving support.

Keep Hybrid's index group, lifecycle and prefill staging; replace only its BF16
physical-row consumer. CPU tests inject a reference consumer. A future CUDA
installer must separately validate SM80, DCP=PP=1, BF16, no speculation/graphs,
and port the indexer/metadata builder. This file deliberately is not an installer.
"""
import torch


def make_bf16_method(kernel, *, cpu_fixture=False):
    def consume(self, q, kv, topk_indices, topk_length=None, actual_num_heads=None):
        if getattr(self, "need_to_return_lse_for_decode", False):
            raise ValueError("The Triton consumer does not expose LSE; DCP is unsupported")
        if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
            raise ValueError("BF16 main KV and Q only")
        if q.ndim != 3 or q.shape[-1] != 576 or kv.shape[-1] != 576:
            raise ValueError("Requires MLA 512 NoPE + 64 RoPE")
        if topk_indices.dtype != torch.int32 or topk_indices.ndim != 2:
            raise ValueError("Resolved physical int32 [tokens, K] indices required")
        n, k = topk_indices.shape
        if n != q.shape[0] or k != 2048:
            raise ValueError("This prototype restricts K to 2048")
        heads = q.shape[1] if actual_num_heads is None else actual_num_heads
        if type(heads) is not int or not 0 < heads <= q.shape[1]:
            raise ValueError("Invalid actual head count")
        if q.stride(-1) != 1 or kv.stride(-1) != 1:
            raise ValueError("Contiguous latent dimension required")
        if q.device != kv.device or q.device != topk_indices.device:
            raise ValueError("All tensors must share a device")
        if not cpu_fixture and q.device.type != "cuda":
            raise ValueError("CUDA required outside explicit CPU fixture")
        if topk_length is not None:
            if (topk_length.shape != (n,) or topk_length.dtype != torch.int32
                    or topk_length.device != q.device):
                raise ValueError("Invalid valid-count metadata")
            # Do NOT mask columns >= count: the upstream converter may leave
            # -1 holes, so a count is not a valid-prefix length. Triton masks -1
            # across all K entries. Validate values only in the CPU fixture;
            # no .item()/CPU readback is added to a prospective GPU hot path.
            if cpu_fixture and not torch.equal(
                    (topk_indices >= 0).sum(-1).to(torch.int32), topk_length):
                raise ValueError("Upstream count/physical-index mismatch")
        out = kernel(q[:, :heads], kv.view(-1, 1, 576),
                     topk_indices.view(n, 1, k), self.softmax_scale)
        return out, None
    return consume


def make_indexer_adapters(prefill_kernel, decode_kernel):
    """Explicitly narrow new tuple-Q ABI to third-party FP8-indexer ABI.

    These functions do not patch the metadata builder's DeepGEMM requirement.
    Q scales describe FP4 in the frozen Hybrid version and cannot be discarded.
    """
    def unpack(q):
        if not isinstance(q, tuple) or len(q) != 2 or q[1] is not None:
            raise ValueError("Only unscaled E4M3 Q; FP4 unsupported")
        if q[0].dtype != torch.float8_e4m3fn:
            raise ValueError("E4M3 index Q required")
        return q[0]

    def prefill(q, kv, weights, ks, ke, clean_logits=False):
        return prefill_kernel(unpack(q), kv, weights, ks, ke,
                              clean_logits=clean_logits)

    def decode(q, kv, weights, context_lens, block_tables, schedule_metadata,
               *, max_model_len, clean_logits=False, indices=None):
        q_tensor = unpack(q)
        if indices is not None or q_tensor.ndim != 4 or q_tensor.shape[1] != 1:
            raise ValueError("No flattened, speculative or multi-token decode")
        if context_lens.shape != (q_tensor.shape[0], 1):
            raise ValueError("Expected native [B, 1] sequence lengths")
        if kv.dtype != torch.uint8 or kv.ndim != 4 or kv.shape[2] != 1:
            raise ValueError("Packed uint8 FP8 index cache required")
        if kv.shape[-1] != q_tensor.shape[-1] + 4:
            raise ValueError("Packed per-block index/float32 scale ABI mismatch")
        return decode_kernel(q_tensor, kv, weights, context_lens[:, 0].contiguous(),
                             block_tables, max_model_len=max_model_len,
                             clean_logits=clean_logits)
    return prefill, decode
