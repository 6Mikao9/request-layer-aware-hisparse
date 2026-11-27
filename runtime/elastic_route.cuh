// Decode-time route kernels of elastic per-layer HiSparse (issue #13). Each replaces a handful of small torch
// ops of one managed layer and decode step with one launch, reading the layer's route row (mode, resident base,
// backed locations) and stale limits from the device, so one captured CUDA Graph serves every offload level.
//
//   resident    ElasticKV.resident_page_table: decode page table of a resident layer (DSA, DeepSeek-V4 C4)
//   select      ElasticKV.reference_select: smooth mode, which selected records read the combined buffer
//   merge       ElasticKV.reference_merge: their combined-buffer rows, else the stock swap-in result
//   store_rows  ElasticKV.store_rows: smooth mode decode store rows
//   gate        ElasticKV.gate: request count of a planned copy, num_real_reqs while the layer is offloaded
//
// The torch statements in runtime/elastic_kv.py are the reference; runtime/elastic_route.py compares both on the
// device before a worker binds these kernels.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <stdexcept>
#include <stdint.h>
#include <string>

namespace sglang {
namespace rlh_elastic_route {

constexpr int32_t kOffloaded = 0;   // ElasticKV route modes
constexpr int32_t kResident = 1;
constexpr int kThreads = 256;
constexpr int64_t kMaxBlock = 1024;

// An int32 or int64 tensor of one or two dimensions, addressed by element strides.
struct Ints {
  void* data;
  int64_t row_stride;
  int64_t col_stride;
  bool wide;

  __device__ __forceinline__ int64_t at(int64_t row, int64_t col = 0) const {
    const int64_t i = row * row_stride + col * col_stride;
    return wide ? static_cast<const int64_t*>(data)[i] : static_cast<const int32_t*>(data)[i];
  }

  __device__ __forceinline__ void put(int64_t row, int64_t col, int64_t value) const {
    const int64_t i = row * row_stride + col * col_stride;
    if (wide) {
      static_cast<int64_t*>(data)[i] = value;
    } else {
      static_cast<int32_t*>(data)[i] = static_cast<int32_t>(value);
    }
  }
};

struct Route {
  Ints top_k;                  // [B, K] record positions, or MiniMax block ids when block > 0; -1 padded
  Ints req_pool_indices;       // [B]
  Ints seq_lens;               // [B] records of each request
  Ints req_to_token;           // [slots, context] token locations
  const int32_t* stale_limit;  // this layer's ElasticKV.stale_limit row, one entry per request slot
  const int32_t* route;        // this layer's ElasticKV.route row: mode, resident base, backed locations
  int64_t ratio;               // tokens per record (4 for C4)
  int64_t max_pos;             // last record position req_to_token addresses
  int64_t block;               // MiniMax sparse block, 0 for record selection
  int64_t num_top_k;
};

// Logical location of record ``pos`` of a request slot: its closing token's location / ratio.
__device__ __forceinline__ int64_t logical_row(const Route& r, int64_t slot, int64_t pos) {
  return r.req_to_token.at(slot, ((pos < r.max_pos ? pos : r.max_pos) + 1) * r.ratio - 1) / r.ratio;
}

// A selected in-range record reads the combined buffer: every record of a resident layer, the still-valid stale
// ones of an offloaded layer.
__device__ __forceinline__ bool reads_combined(const Route& r, int64_t slot, int64_t pos, int64_t row) {
  return r.route[0] == kResident || (pos < r.stale_limit[slot] && row < r.route[2]);
}

__global__ void resident_kernel(const Route r, const Ints table, const int64_t offset) {
  const int64_t b = blockIdx.x;
  const int64_t j = static_cast<int64_t>(blockIdx.y) * kThreads + threadIdx.x;
  if (j >= r.num_top_k) return;
  const int64_t pos = r.top_k.at(b, j);
  int64_t row = -1;
  if (pos >= 0 && pos < r.seq_lens.at(b)) {
    row = logical_row(r, r.req_pool_indices.at(b), pos) + offset;
  }
  table.put(b, j, row);
}

__global__ void select_tokens_kernel(const Route r, const Ints masked, int64_t* rows, int8_t* reads) {
  const int64_t b = blockIdx.x;
  const int64_t j = static_cast<int64_t>(blockIdx.y) * kThreads + threadIdx.x;
  if (j >= r.num_top_k) return;
  const int64_t pos = r.top_k.at(b, j);
  const int64_t slot = r.req_pool_indices.at(b);
  const bool in_range = pos >= 0 && pos < r.seq_lens.at(b);
  const int64_t row = in_range ? logical_row(r, slot, pos) : 0;
  const bool hit = in_range && reads_combined(r, slot, pos, row);
  masked.put(b, j, hit ? -1 : pos);
  rows[b * r.num_top_k + j] = row + r.route[1];
  reads[b * r.num_top_k + j] = hit ? 1 : 0;
}

// MiniMax: one thread block per selected block of ``block`` records; it skips the stock swap-in only when every
// in-range record of it reads the combined buffer.
__global__ void select_blocks_kernel(const Route r, const Ints masked, int64_t* rows, int8_t* reads) {
  const int64_t b = blockIdx.x;
  const int64_t j = blockIdx.y;
  const int64_t t = threadIdx.x;
  const int64_t selected = r.top_k.at(b, j);
  const int64_t pos = selected >= 0 ? selected * r.block + t : -1;
  const int64_t slot = r.req_pool_indices.at(b);
  const bool in_range = pos >= 0 && pos < r.seq_lens.at(b);
  const int64_t row = in_range ? logical_row(r, slot, pos) : 0;
  const bool hit = in_range && reads_combined(r, slot, pos, row);
  const bool whole = !__syncthreads_or(in_range && !hit);
  if (t == 0) masked.put(b, j, whole ? -1 : selected);
  const int64_t i = (b * r.num_top_k + j) * r.block + t;
  rows[i] = row + r.route[1];
  reads[i] = whole && in_range ? 1 : 0;
}

__global__ void merge_kernel(const Ints swapped, const int64_t* rows, const int8_t* reads, const Ints out,
                             const int64_t width) {
  const int64_t b = blockIdx.x;
  const int64_t j = static_cast<int64_t>(blockIdx.y) * kThreads + threadIdx.x;
  if (j >= width) return;
  const int64_t i = b * width + j;
  out.put(b, j, reads[i] ? rows[i] : swapped.at(b, j));
}

// The hot row while the layer is offloaded, else resident base + location; a negative location (a C4 token
// that closes no record) stays negative.
__global__ void store_rows_kernel(const Ints loc, const Ints mapping, const int32_t* route, const Ints out,
                                  const int64_t count) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (i >= count) return;
  const int64_t l = loc.at(i);
  out.put(i, 0, l < 0 ? l : (route[0] == kOffloaded ? mapping.at(l) : l + route[1]));
}

__global__ void gate_kernel(const int32_t* num_real_reqs, const int32_t* route, int32_t* out) {
  out[0] = route[0] == kOffloaded ? num_real_reqs[0] : 0;
}

[[noreturn]] inline void reject(const char* message) {
  throw std::runtime_error(std::string("rlh_elastic_route: ") + message);
}

inline bool is_int(const tvm::ffi::TensorView& t, int bits) {
  const DLDataType dtype = t.dtype();
  return dtype.code == kDLInt && dtype.bits == bits && dtype.lanes == 1;
}

inline Ints ints(const tvm::ffi::TensorView& t, int ndim, const char* message) {
  if (t.ndim() != ndim || !(is_int(t, 32) || is_int(t, 64))) reject(message);
  return Ints{t.data_ptr(), t.strides()[0], ndim == 2 ? t.strides()[1] : 0, is_int(t, 64)};
}

inline const int32_t* int32_row(const tvm::ffi::TensorView& t, int64_t min_size, const char* message) {
  if (t.ndim() != 1 || !is_int(t, 32) || t.strides()[0] != 1 || t.shape()[0] < min_size) reject(message);
  return static_cast<const int32_t*>(t.data_ptr());
}

// A contiguous [rows, cols] output of ``bits``-bit integers (0: int32 or int64).
inline void check_out(const tvm::ffi::TensorView& t, int64_t rows, int64_t cols, int bits, const char* message) {
  const bool dtype = bits ? is_int(t, bits) : (is_int(t, 32) || is_int(t, 64));
  if (t.ndim() != 2 || !dtype || t.shape()[0] != rows || t.shape()[1] != cols || t.strides()[1] != 1 ||
      t.strides()[0] != cols) {
    reject(message);
  }
}

inline Route route_of(const tvm::ffi::TensorView& top_k, const tvm::ffi::TensorView& req_pool_indices,
                      const tvm::ffi::TensorView& seq_lens, const tvm::ffi::TensorView& req_to_token,
                      int64_t ratio) {
  Route r{};
  r.top_k = ints(top_k, 2, "top_k must be [B, K] int32/int64");
  r.req_pool_indices = ints(req_pool_indices, 1, "req_pool_indices must be [B] int32/int64");
  r.seq_lens = ints(seq_lens, 1, "seq_lens must be [B] int32/int64");
  r.req_to_token = ints(req_to_token, 2, "req_to_token must be [slots, context] int32/int64");
  if (req_pool_indices.shape()[0] != top_k.shape()[0] || seq_lens.shape()[0] != top_k.shape()[0]) {
    reject("req_pool_indices and seq_lens need one entry per top_k row");
  }
  if (ratio < 1 || req_to_token.shape()[1] < ratio) reject("ratio must be at least 1 and fit req_to_token");
  r.ratio = ratio;
  r.max_pos = req_to_token.shape()[1] / ratio - 1;
  r.num_top_k = top_k.shape()[1];
  return r;
}

inline dim3 grid_of(int64_t rows, int64_t cols) {
  return dim3(static_cast<unsigned>(rows), static_cast<unsigned>((cols + kThreads - 1) / kThreads));
}

// Host entry points (exported through the pinned SGLang JIT); shapes and dtypes are checked here because the
// kernels index with them.
inline void resident(
    tvm::ffi::TensorView top_k, tvm::ffi::TensorView req_pool_indices, tvm::ffi::TensorView seq_lens,
    tvm::ffi::TensorView req_to_token, tvm::ffi::TensorView table, int64_t ratio, int64_t offset) {
  const Route r = route_of(top_k, req_pool_indices, seq_lens, req_to_token, ratio);
  const int64_t batch = top_k.shape()[0];
  check_out(table, batch, r.num_top_k, 32, "table must be a contiguous [B, K] int32 tensor");
  if (batch == 0 || r.num_top_k == 0) return;
  const Ints out{table.data_ptr(), r.num_top_k, 1, false};
  host::LaunchKernel(grid_of(batch, r.num_top_k), kThreads, host::LaunchKernel::resolve_device(top_k.device()))(
      resident_kernel, r, out, offset);
}

inline void select(
    tvm::ffi::TensorView top_k, tvm::ffi::TensorView req_pool_indices, tvm::ffi::TensorView seq_lens,
    tvm::ffi::TensorView req_to_token, tvm::ffi::TensorView stale_limit, tvm::ffi::TensorView route,
    tvm::ffi::TensorView masked, tvm::ffi::TensorView rows, tvm::ffi::TensorView reads, int64_t ratio,
    int64_t block) {
  Route r = route_of(top_k, req_pool_indices, seq_lens, req_to_token, ratio);
  const int64_t batch = top_k.shape()[0];
  if (block < 0 || block > kMaxBlock) reject("block must be 0 (records) or a MiniMax block of at most 1024");
  const int64_t width = block ? r.num_top_k * block : r.num_top_k;
  r.block = block;
  r.stale_limit = int32_row(stale_limit, req_to_token.shape()[0], "stale_limit must be one int32 per slot");
  r.route = int32_row(route, 3, "route must be the layer's int32 route row");
  check_out(masked, batch, r.num_top_k, top_k.dtype().bits, "masked must be a contiguous copy of top_k's shape");
  check_out(rows, batch, width, 64, "rows must be a contiguous [B, width] int64 tensor");
  check_out(reads, batch, width, 8, "reads must be a contiguous [B, width] int8 tensor");
  if (batch == 0 || r.num_top_k == 0) return;
  const Ints out{masked.data_ptr(), r.num_top_k, 1, is_int(masked, 64)};
  auto* row_out = static_cast<int64_t*>(rows.data_ptr());
  auto* read_out = static_cast<int8_t*>(reads.data_ptr());
  const cudaStream_t stream = host::LaunchKernel::resolve_device(top_k.device());
  if (block) {
    host::LaunchKernel(dim3(static_cast<unsigned>(batch), static_cast<unsigned>(r.num_top_k)),
                       static_cast<unsigned>(block), stream)(select_blocks_kernel, r, out, row_out, read_out);
  } else {
    host::LaunchKernel(grid_of(batch, r.num_top_k), kThreads, stream)(select_tokens_kernel, r, out, row_out,
                                                                      read_out);
  }
}

inline void merge(
    tvm::ffi::TensorView swapped, tvm::ffi::TensorView rows, tvm::ffi::TensorView reads, tvm::ffi::TensorView out) {
  const Ints from = ints(swapped, 2, "swapped must be [B, >= width] int32/int64");
  if (rows.ndim() != 2) reject("rows must be a contiguous [B, width] int64 tensor");
  const int64_t batch = rows.shape()[0], width = rows.shape()[1];
  check_out(rows, batch, width, 64, "rows must be a contiguous [B, width] int64 tensor");
  check_out(reads, batch, width, 8, "reads must be a contiguous [B, width] int8 tensor");
  check_out(out, batch, width, swapped.dtype().bits, "out must be a contiguous [B, width] tensor of swapped's dtype");
  if (swapped.shape()[0] != batch || swapped.shape()[1] < width) reject("swapped needs [B, >= width]");
  if (batch == 0 || width == 0) return;
  const Ints to{out.data_ptr(), width, 1, is_int(out, 64)};
  host::LaunchKernel(grid_of(batch, width), kThreads, host::LaunchKernel::resolve_device(rows.device()))(
      merge_kernel, from, static_cast<const int64_t*>(rows.data_ptr()),
      static_cast<const int8_t*>(reads.data_ptr()), to, width);
}

inline void store_rows(
    tvm::ffi::TensorView loc, tvm::ffi::TensorView mapping, tvm::ffi::TensorView route, tvm::ffi::TensorView out) {
  const Ints from = ints(loc, 1, "loc must be 1-D int32/int64");
  const Ints hot = ints(mapping, 1, "mapping must be 1-D int32/int64");
  const int64_t count = loc.shape()[0];
  if (out.ndim() != 1 || out.shape()[0] != count || out.strides()[0] != 1 || !is_int(out, 64)) {
    reject("out must be a contiguous int64 tensor of loc's length");
  }
  const int32_t* row = int32_row(route, 2, "route must be the layer's int32 route row");
  if (count == 0) return;
  const Ints to{out.data_ptr(), 1, 0, true};
  host::LaunchKernel(static_cast<unsigned>((count + kThreads - 1) / kThreads), kThreads,
                     host::LaunchKernel::resolve_device(loc.device()))(store_rows_kernel, from, hot, row, to, count);
}

inline void gate(tvm::ffi::TensorView num_real_reqs, tvm::ffi::TensorView route, tvm::ffi::TensorView out) {
  const int32_t* count = int32_row(num_real_reqs, 1, "num_real_reqs must be a one-element int32 tensor");
  const int32_t* row = int32_row(route, 1, "route must be the layer's int32 route row");
  if (out.ndim() != 1 || !is_int(out, 32) || out.shape()[0] != 1) reject("out must be a one-element int32 tensor");
  host::LaunchKernel(1, 1, host::LaunchKernel::resolve_device(out.device()))(
      gate_kernel, count, row, static_cast<int32_t*>(out.data_ptr()));
}

}  // namespace rlh_elastic_route
}  // namespace sglang
