// Resident-layer resolver of the fixed offload set S (MiniMax M3, BF16, sm_80).
//
// features.resident_direct=false routes every resident layer through this kernel:
// one CTA per graph row writes the req_to_token location of each selected position
// as a HiSparse slot, in the output layout of the native swap kernel (-1 for invalid
// blocks, positions at or past seq_len and padded graph rows). The stock consumer
// then reads the resident plane through hisparse_slots. With resident_direct=true
// the consumer reads req_to_token itself and this kernel never runs. Statement-level
// mirror: runtime/layer_placement.py::resident_slot_table; the A100 gate
// runtime/cuda_combined_correctness.py checks this kernel against it.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <stdexcept>
#include <stdint.h>
#include <string>

// Pinned SGLang source (Apache-2.0), digest-checked through runtime/PROVENANCE.json:
// resolve_selected_token, the native block-to-token mapping of the swap kernel.
#include "kvcacheio/hisparse.cuh"

namespace sglang {
namespace rlh_layer_placement {

template <int BLOCK_SIZE, int NUM_TOP_K, int SPARSE_BLOCK_SIZE, typename SeqT, typename RpiT>
__global__ __launch_bounds__(BLOCK_SIZE) void resolve_resident_kernel(
    const int32_t* __restrict__ top_k,
    int64_t top_k_stride,
    const RpiT* __restrict__ req_pool_indices,
    const SeqT* __restrict__ seq_lens,
    const int32_t* __restrict__ num_real_reqs,
    const int32_t* __restrict__ req_to_token,
    int64_t req_to_token_stride,
    int64_t context,
    int32_t* __restrict__ out,
    int64_t out_stride) {
  constexpr int N = NUM_TOP_K * SPARSE_BLOCK_SIZE;
  const int64_t bid = blockIdx.x;
  int32_t* row = out + bid * out_stride;
  if (bid >= num_real_reqs[0]) {  // padded graph row: invalid, like the native swap output
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) row[i] = -1;
    return;
  }
  const int64_t rid = static_cast<int64_t>(req_pool_indices[bid]);
  const int64_t seq_len = static_cast<int64_t>(seq_lens[bid]);
  const int32_t* blocks = top_k + bid * top_k_stride;
  const int32_t* locs = req_to_token + rid * req_to_token_stride;
  for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
    const int32_t token = resolve_selected_token<SPARSE_BLOCK_SIZE, true>(blocks, i);
    row[i] = (token >= 0 && token < seq_len && token < context) ? locs[token] : -1;
  }
}

[[noreturn]] inline void reject(const char* message) {
  throw std::runtime_error(std::string("rlh_layer_placement::resolve_resident: ") + message);
}

// Host entry point. Python validates dtypes and the bound geometry before capture; these
// checks guard the address arithmetic the kernel relies on. No allocation, no sync.
template <int BLOCK_SIZE, int NUM_TOP_K, int SPARSE_BLOCK_SIZE>
void resolve_resident(
    tvm::ffi::TensorView top_k,
    tvm::ffi::TensorView req_pool_indices,
    tvm::ffi::TensorView seq_lens,
    tvm::ffi::TensorView num_real_reqs,
    tvm::ffi::TensorView req_to_token,
    tvm::ffi::TensorView out) {
  using namespace host;
  constexpr int N = NUM_TOP_K * SPARSE_BLOCK_SIZE;
  const int64_t batch = top_k.shape()[0];
  if (top_k.ndim() != 2 || top_k.shape()[1] != NUM_TOP_K || top_k.strides()[1] != 1) {
    reject("top_k must be [B, NUM_TOP_K] with unit column stride");
  }
  if (out.ndim() != 2 || out.shape()[0] != batch || out.shape()[1] < N || out.strides()[1] != 1) {
    reject("output must be [B, >= selected tokens] with unit column stride");
  }
  if (req_to_token.ndim() != 2 || req_to_token.strides()[1] != 1) {
    reject("req_to_token must be [slots, context] with unit column stride");
  }
  if (req_pool_indices.ndim() != 1 || seq_lens.ndim() != 1 || req_pool_indices.shape()[0] < batch ||
      seq_lens.shape()[0] < batch || num_real_reqs.ndim() != 1 || num_real_reqs.shape()[0] != 1) {
    reject("req_pool_indices/seq_lens must cover the batch and num_real_reqs must be one element");
  }
  if (batch == 0) return;
  const auto device = top_k.device();
  const auto seq_dtype = seq_lens.dtype();
  const auto rpi_dtype = req_pool_indices.dtype();
  const bool seq_is_i64 = seq_dtype.code == kDLInt && seq_dtype.bits == 64;
  const bool rpi_is_i64 = rpi_dtype.code == kDLInt && rpi_dtype.bits == 64;
  const auto* blocks = static_cast<const int32_t*>(top_k.data_ptr());
  const auto* real = static_cast<const int32_t*>(num_real_reqs.data_ptr());
  const auto* locs = static_cast<const int32_t*>(req_to_token.data_ptr());
  auto* slots = static_cast<int32_t*>(out.data_ptr());
#define RLH_LAYER_PLACEMENT_LAUNCH(SEQ_T, RPI_T)                                                             \
  LaunchKernel(static_cast<unsigned>(batch), BLOCK_SIZE, device)(                                           \
      resolve_resident_kernel<BLOCK_SIZE, NUM_TOP_K, SPARSE_BLOCK_SIZE, SEQ_T, RPI_T>, blocks,              \
      top_k.strides()[0], static_cast<const RPI_T*>(req_pool_indices.data_ptr()),                           \
      static_cast<const SEQ_T*>(seq_lens.data_ptr()), real, locs, req_to_token.strides()[0],               \
      req_to_token.shape()[1], slots, out.strides()[0])
  if (seq_is_i64 && rpi_is_i64) {
    RLH_LAYER_PLACEMENT_LAUNCH(int64_t, int64_t);
  } else if (seq_is_i64) {
    RLH_LAYER_PLACEMENT_LAUNCH(int64_t, int32_t);
  } else if (rpi_is_i64) {
    RLH_LAYER_PLACEMENT_LAUNCH(int32_t, int64_t);
  } else {
    RLH_LAYER_PLACEMENT_LAUNCH(int32_t, int32_t);
  }
#undef RLH_LAYER_PLACEMENT_LAUNCH
}

}  // namespace rlh_layer_placement
}  // namespace sglang
