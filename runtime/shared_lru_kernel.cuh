// Shared cross-layer HiSparse hot pool for MHA main KV (MiniMax M3, BF16, sm_80).
//
// One CTA resolves one graph row of one layer invocation. Every phase has a
// sequential statement in runtime/shared_lru.py::KernelMirror; the CPU tests
// check that mirror against the reference and runtime/cuda_shared_lru_correctness.py
// checks this kernel against the mirror on an A100.
//
// Arena row r of a request is native hot slot r % H of main-KV plane r / H. The
// arena therefore reuses the request's native lockstep device locations in
// every plane and allocates no second hot cache. Planes are one contiguous
// allocation, so the consumer reads index plane * plane_rows + loc.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <climits>
#include <stdexcept>
#include <stdint.h>
#include <string>

// Pinned SGLang source (Apache-2.0), digest-checked through runtime/PROVENANCE.json:
// WARP_SIZE, FULL_WARP_MASK, hash_slot and transfer_item_warp.
#include "kvcacheio/hisparse.cuh"

namespace sglang {
namespace rlh_shared_lru {

static_assert(WARP_SIZE == 32, "The shared resolver targets CUDA warps (sm_80)");

constexpr int32_t kFreeTag = -1;   // row holds no published key of this generation
constexpr int32_t kTakenTag = -2;  // row chosen as a victim during this invocation
constexpr int32_t kHashEmpty = -1;
constexpr int32_t kPosInvalid = -1;
constexpr int32_t kPosLatest = -2;
constexpr int kMaxPlanes = 64;
constexpr int kFreeOwner = kMaxPlanes;  // owner index of free rows
constexpr int kOwners = kMaxPlanes + 1;
constexpr int32_t kStatusCapacity = 1;  // fewer eligible victims than misses (capacity preflight forbids)
constexpr int32_t kStatusOrder = 2;     // recency list would lose or duplicate a row
constexpr int kNumCounters = 8;         // unique, hits, misses, bypassed, bytes, breaches, latest, invalid
struct Params {
  const int32_t* top_k;             // [B, NUM_TOP_K] selector blocks, -1 invalid
  int64_t top_k_stride;
  const void* req_pool_indices;     // [B] int32 or int64
  const void* seq_lens;             // [B] int32 or int64
  const int32_t* num_real_reqs;     // [1] real rows of the captured batch
  int32_t* row_token;               // native req_device_buffer_tokens [planes, slots, cols]
  int64_t token_plane_stride;
  int64_t token_slot_stride;
  int32_t* row_owner;               // [slots, planes * H] owner tag = generation << 16 | plane
  int64_t owner_slot_stride;
  int32_t* order;                   // [slots, planes * H] recency lists, one region per domain
  int64_t order_slot_stride;
  int32_t* occupancy;               // [slots, planes] rows owned per plane
  int64_t occupancy_slot_stride;
  const int32_t* floors;            // [planes] soft-floor rows
  const int32_t* generation;        // [slots] device generation, 1..0x7FFF
  const int32_t* device_locs;       // this layer's req_device_buffer_token_locs [slots, cols]
  int64_t locs_slot_stride;
  const int64_t* host_locs;         // req_to_host_pool [slots, context]
  int64_t host_slot_stride;
  const void* host_k;               // requesting layer's pinned host planes
  const void* host_v;
  void* arena_k;                    // contiguous all-plane device K/V
  void* arena_v;
  int32_t* out;                     // [B, >= NUM_TOP_K * SPARSE_BLOCK_SIZE] flat consumer indices
  int64_t out_stride;
  int32_t* status;                  // [1] sticky error bits
  unsigned long long* counters;     // [planes, kNumCounters] when diagnostics are on
  int64_t plane_rows;
  int64_t item_size_bytes;
  int32_t plane;
  int32_t num_planes;
  int32_t hot_rows;
  int32_t order_offset;
  int32_t order_size;
  int32_t scratch_offset;
};

template <int N, int BLOCK_SIZE>
struct SmemLayout {
  static constexpr int WARPS = BLOCK_SIZE / WARP_SIZE;
  static constexpr int HASH = 2 * N;
  // int32: pos_token, uid_token, uid_row, victim_row, miss_uid, sort, hash keys/values,
  // scan, scalars, surplus, owner table; int16: pos_hslot, victim_old; uint8: uid_miss.
  static constexpr int I32 = 6 * N + 2 * HASH + (WARPS + 1) + 8 + kOwners + WARPS * kOwners;
  static constexpr int I16 = 2 * N;
  static constexpr size_t BYTES = I32 * sizeof(int32_t) + I16 * sizeof(int16_t) + N * sizeof(uint8_t);
};
// Exclusive prefix sum over the CTA in thread order. Every thread must call it.
template <int BLOCK_SIZE>
__device__ __forceinline__ int block_exclusive_scan(int value, int* s_scan, int& total) {
  constexpr int WARPS = BLOCK_SIZE / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;
  int inclusive = value;
#pragma unroll
  for (int d = 1; d < WARP_SIZE; d <<= 1) {
    const int n = __shfl_up_sync(FULL_WARP_MASK, inclusive, d);
    if (lane >= d) inclusive += n;
  }
  if (lane == WARP_SIZE - 1) s_scan[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    const int mine = lane < WARPS ? s_scan[lane] : 0;
    int prefix = mine;
#pragma unroll
    for (int d = 1; d < WARP_SIZE; d <<= 1) {
      const int n = __shfl_up_sync(FULL_WARP_MASK, prefix, d);
      if (lane >= d) prefix += n;
    }
    if (lane < WARPS) s_scan[lane] = prefix - mine;
    if (lane == WARP_SIZE - 1) s_scan[WARPS] = prefix;
  }
  __syncthreads();
  total = s_scan[WARPS];
  const int result = s_scan[warp] + inclusive - value;
  __syncthreads();  // s_scan is reused by the next call
  return result;
}

template <int HASH>
__device__ __forceinline__ int hash_insert(int32_t* keys, int32_t token) {
  int slot = hash_slot(token, HASH);
  while (true) {
    const int32_t old = atomicCAS(&keys[slot], kHashEmpty, token);
    if (old == kHashEmpty || old == token) return slot;
    slot = (slot + 1) % HASH;
  }
}

template <int HASH>
__device__ __forceinline__ int hash_find(const int32_t* keys, int32_t token) {
  int slot = hash_slot(token, HASH);
  while (true) {
    const int32_t key = keys[slot];
    if (key == token) return slot;
    if (key == kHashEmpty) return -1;
    slot = (slot + 1) % HASH;
  }
}
// Flattened selection position -> logical token, or kPosInvalid (same rule as the pinned kernel).
template <int SPARSE_BLOCK_SIZE>
__device__ __forceinline__ int32_t selected_token(const int32_t* top_k, int position, int64_t seq_len) {
  const int32_t block = top_k[position / SPARSE_BLOCK_SIZE];
  if (block < 0) return kPosInvalid;
  const int64_t token = static_cast<int64_t>(block) * SPARSE_BLOCK_SIZE + position % SPARSE_BLOCK_SIZE;
  return token < seq_len ? static_cast<int32_t>(token) : kPosInvalid;
}

// 64-bit row offset into the contiguous all-plane view: plane * plane_rows + loc.
__device__ __forceinline__ int64_t flat_row(const Params& p, const int32_t* locs, int32_t row) {
  return static_cast<int64_t>(row / p.hot_rows) * p.plane_rows + locs[row % p.hot_rows];
}

// Reverse token of an arena row lives in the native token table of the row's plane and slot.
__device__ __forceinline__ int32_t* token_of(const Params& p, int64_t rid, int32_t row) {
  return p.row_token + static_cast<int64_t>(row / p.hot_rows) * p.token_plane_stride +
         rid * p.token_slot_stride + row % p.hot_rows;
}

template <int N, int BLOCK_SIZE>
__device__ __forceinline__ void bitonic_sort(int32_t* data) {
  for (int k = 2; k <= N; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        const int partner = i ^ j;
        if (partner > i) {
          const int32_t a = data[i];
          const int32_t b = data[partner];
          if ((a > b) == ((i & k) == 0)) {
            data[i] = b;
            data[partner] = a;
          }
        }
      }
      __syncthreads();
    }
  }
}
template <
    int BLOCK_SIZE,
    int NUM_TOP_K,
    int SPARSE_BLOCK_SIZE,
    bool kFifo,
    bool kSoftFloor,
    bool kBypass,
    bool kCounters,
    typename SeqLensT,
    typename ReqPoolIndicesT>
__global__ __launch_bounds__(BLOCK_SIZE, 1) void resolve_kernel(const Params p) {
  constexpr int N = NUM_TOP_K * SPARSE_BLOCK_SIZE;
  constexpr int HASH = SmemLayout<N, BLOCK_SIZE>::HASH;
  constexpr int WARPS = SmemLayout<N, BLOCK_SIZE>::WARPS;
  constexpr int PER = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;  // contiguous positions per thread
  static_assert(BLOCK_SIZE % WARP_SIZE == 0 && WARPS <= WARP_SIZE, "CTA must be whole warps, <= 1024 threads");
  static_assert(N > 0 && N <= 2048 && (N & (N - 1)) == 0,
                "SM80 selection width must be a power of two up to 2048");
  static_assert(HASH <= 32767, "hash slots are stored as int16");

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid % WARP_SIZE;
  const int warp = tid / WARP_SIZE;
  int32_t* out = p.out + bid * p.out_stride;
  if (bid >= p.num_real_reqs[0]) {  // padded graph row: invalid output, no state change
    for (int i = tid; i < N; i += BLOCK_SIZE) out[i] = -1;
    return;
  }
  const int64_t rid = static_cast<const ReqPoolIndicesT*>(p.req_pool_indices)[bid];
  const int64_t seq_len = static_cast<const SeqLensT*>(p.seq_lens)[bid];
  const int32_t* top_k = p.top_k + bid * p.top_k_stride;
  const int32_t* locs = p.device_locs + rid * p.locs_slot_stride;
  const int64_t base = static_cast<int64_t>(p.plane) * p.plane_rows;

  if (seq_len <= p.hot_rows) {  // native short path, shifted into this layer's plane
    for (int i = tid; i < N; i += BLOCK_SIZE) {
      const int32_t token = selected_token<SPARSE_BLOCK_SIZE>(top_k, i, seq_len);
      const int32_t loc = token >= 0 ? locs[token] : -1;
      out[i] = loc >= 0 ? static_cast<int32_t>(base + loc) : -1;
    }
    return;
  }

  extern __shared__ __align__(16) char smem_raw[];
  int32_t* s_pos_token = reinterpret_cast<int32_t*>(smem_raw);
  int32_t* s_uid_token = s_pos_token + N;
  int32_t* s_uid_row = s_uid_token + N;
  int32_t* s_victim_row = s_uid_row + N;
  int32_t* s_miss_uid = s_victim_row + N;
  int32_t* s_sort = s_miss_uid + N;
  int32_t* s_hash_key = s_sort + N;
  int32_t* s_hash_val = s_hash_key + HASH;  // first position, then unique id
  int32_t* s_scan = s_hash_val + HASH;
  int32_t* s_vars = s_scan + WARPS + 1;     // 0 hits, 1 victims, 2 write, 3 breaches, 4 touched, 5 latest, 6 invalid
  int32_t* s_surplus = s_vars + 8;
  int32_t* s_owner_tab = s_surplus + kOwners;
  int16_t* s_pos_hslot = reinterpret_cast<int16_t*>(s_owner_tab + WARPS * kOwners);
  int16_t* s_victim_old = s_pos_hslot + N;
  uint8_t* s_uid_miss = reinterpret_cast<uint8_t*>(s_victim_old + N);
  for (int i = tid; i < HASH; i += BLOCK_SIZE) {
    s_hash_key[i] = kHashEmpty;
    s_hash_val[i] = INT_MAX;
  }
  if (tid < 8) s_vars[tid] = 0;
  __syncthreads();

  // Phase 1: classify positions; history tokens enter the hash with their first position.
  const int32_t newest = static_cast<int32_t>(seq_len - 1);
  const int64_t latest_flat = base + locs[p.hot_rows];
  for (int i = tid; i < N; i += BLOCK_SIZE) {
    const int32_t token = selected_token<SPARSE_BLOCK_SIZE>(top_k, i, seq_len);
    s_pos_hslot[i] = -1;
    if (token < 0) {
      s_pos_token[i] = kPosInvalid;
      out[i] = -1;
      if constexpr (kCounters) atomicAdd(&s_vars[6], 1);
    } else if (token == newest) {
      s_pos_token[i] = kPosLatest;
      out[i] = static_cast<int32_t>(latest_flat);
      if constexpr (kCounters) atomicAdd(&s_vars[5], 1);
    } else {
      s_pos_token[i] = token;
      const int slot = hash_insert<HASH>(s_hash_key, token);
      s_pos_hslot[i] = static_cast<int16_t>(slot);
      atomicMin(&s_hash_val[slot], i);
    }
  }
  __syncthreads();

  // Phase 2: unique ids in first-occurrence order; duplicates share one id, one row, one copy.
  bool canonical[PER];
  int local = 0;
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    const int i = tid * PER + k;
    canonical[k] = i < N && s_pos_hslot[i] >= 0 && s_hash_val[s_pos_hslot[i]] == i;
    local += canonical[k] ? 1 : 0;
  }
  int unique = 0;
  int uid = block_exclusive_scan<BLOCK_SIZE>(local, s_scan, unique);
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    const int i = tid * PER + k;
    if (canonical[k]) {
      s_hash_val[s_pos_hslot[i]] = uid;
      s_uid_token[uid] = s_pos_token[i];
      s_uid_row[uid] = -1;
      s_uid_miss[uid] = 0;
      ++uid;
    }
  }
  __syncthreads();
  const int32_t gen = p.generation[rid];
  const int32_t my_tag = (gen << 16) | p.plane;
  int32_t* region = p.order + rid * p.order_slot_stride + p.order_offset;
  int32_t* owner = p.row_owner + rid * p.owner_slot_stride;
  // Unique id of a row that holds a selected token of this layer and generation, else -1.
  auto lookup = [&](int32_t row) -> int {
    if (owner[row] != my_tag) return -1;
    const int slot = hash_find<HASH>(s_hash_key, *token_of(p, rid, row));
    return slot < 0 ? -1 : s_hash_val[slot];
  };
  auto owner_index = [&](int32_t tag) -> int {
    return (tag < 0 || (tag >> 16) != gen) ? kFreeOwner : (tag & 0xFFFF);
  };

  // Phase 3: snapshot every hit of the complete selection before choosing any victim.
  for (int j = tid; j < p.order_size; j += BLOCK_SIZE) {
    const int32_t row = region[j];
    const int u = lookup(row);
    if (u >= 0) {
      s_uid_row[u] = row;
      atomicAdd(&s_vars[0], 1);
    }
  }
  __syncthreads();
  const int need = unique - s_vars[0];

  // Phase 4: ordered victim scan of the domain list (free rows lead it). A floor-aware pass
  // takes a row only while its owner keeps more rows than its soft floor.
  auto scan_victims = [&](bool floor_aware) {
    for (int start = 0; start < p.order_size; start += BLOCK_SIZE) {
      if (s_vars[1] >= need) break;
      const int j = start + tid;
      int32_t row = -1;
      int owner_idx = kFreeOwner;
      bool candidate = false;
      if (j < p.order_size) {
        row = region[j];
        const int32_t tag = owner[row];
        candidate = tag != kTakenTag && lookup(row) < 0;
        owner_idx = owner_index(tag);
      }
      bool accept = candidate;
      if (floor_aware) {  // rank among same-owner candidates of this chunk, in list order
        const unsigned peers = __match_any_sync(FULL_WARP_MASK, candidate ? owner_idx : -1 - lane);
        const int in_warp = __popc(peers & ((1u << lane) - 1u));
        for (int x = tid; x < WARPS * kOwners; x += BLOCK_SIZE) s_owner_tab[x] = 0;
        __syncthreads();
        if (candidate && in_warp == 0) s_owner_tab[warp * kOwners + owner_idx] = __popc(peers);
        __syncthreads();
        if (tid < kOwners) {
          int running = 0;
          for (int w = 0; w < WARPS; ++w) {
            const int count = s_owner_tab[w * kOwners + tid];
            s_owner_tab[w * kOwners + tid] = running;
            running += count;
          }
        }
        __syncthreads();
        const int rank = candidate ? s_owner_tab[warp * kOwners + owner_idx] + in_warp : 0;
        accept = candidate && (owner_idx == kFreeOwner || s_surplus[owner_idx] - rank > 0);
      }
      int accepted = 0;
      const int position = block_exclusive_scan<BLOCK_SIZE>(accept ? 1 : 0, s_scan, accepted);
      const int taken = s_vars[1];
      if (accept && taken + position < need) {
        s_victim_row[taken + position] = row;
        s_victim_old[taken + position] = static_cast<int16_t>(owner_idx == kFreeOwner ? -1 : owner_idx);
        owner[row] = kTakenTag;
        if (floor_aware && owner_idx != kFreeOwner) atomicSub(&s_surplus[owner_idx], 1);
      }
      __syncthreads();
      if (tid == 0) s_vars[1] = min(need, taken + accepted);
      __syncthreads();
    }
  };
  if (need > 0) {
    if constexpr (kBypass) {  // weak-reuse layer: charged scratch rows, nothing is admitted
      for (int k = tid; k < need; k += BLOCK_SIZE) {
        s_victim_row[k] = p.scratch_offset + k;
        s_victim_old[k] = -1;
      }
      if (tid == 0) s_vars[1] = need;
    } else {
      if constexpr (kSoftFloor) {
        const int32_t* occupancy = p.occupancy + rid * p.occupancy_slot_stride;
        for (int o = tid; o < kOwners; o += BLOCK_SIZE) {
          s_surplus[o] = o < p.num_planes ? occupancy[o] - p.floors[o] : 0;
        }
        __syncthreads();
        scan_victims(true);
      }
      const int before = s_vars[1];
      scan_victims(false);
      if (kSoftFloor && tid == 0) s_vars[3] = s_vars[1] - before;  // published floor breaches
    }
    __syncthreads();
  }
  const int victims = s_vars[1];
  if (victims < need) {  // all candidates unsafe: restore tags, publish the error, read nothing
    for (int k = tid; k < victims; k += BLOCK_SIZE) {
      const int old = s_victim_old[k];
      owner[s_victim_row[k]] = old < 0 ? kFreeTag : ((gen << 16) | old);
    }
    for (int i = tid; i < N; i += BLOCK_SIZE) {
      if (s_pos_hslot[i] >= 0) out[i] = -1;
    }
    if (tid == 0) atomicOr(p.status, kStatusCapacity);
    return;
  }

  // Phase 5: misses take victims in unique-id order.
  bool missed[PER];
  local = 0;
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    const int u = tid * PER + k;
    missed[k] = u < unique && s_uid_row[u] < 0;
    local += missed[k] ? 1 : 0;
  }
  int total_missed = 0;
  int rank = block_exclusive_scan<BLOCK_SIZE>(local, s_scan, total_missed);
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    if (missed[k]) {
      const int u = tid * PER + k;
      s_uid_row[u] = s_victim_row[rank];
      s_uid_miss[u] = 1;
      s_miss_uid[rank] = u;
      ++rank;
    }
  }
  __syncthreads();

  // Phase 6: one warp per miss copies K and V from the request's host history.
  const int64_t* host_locs = p.host_locs + rid * p.host_slot_stride;
  for (int k = warp; k < need; k += WARPS) {
    const int u = s_miss_uid[k];
    const int64_t src = host_locs[s_uid_token[u]] * p.item_size_bytes;
    const int64_t dst = flat_row(p, locs, s_uid_row[u]) * p.item_size_bytes;
    transfer_item_warp(lane, static_cast<const char*>(p.host_k) + src, static_cast<char*>(p.arena_k) + dst,
                       p.item_size_bytes);
    transfer_item_warp(lane, static_cast<const char*>(p.host_v) + src, static_cast<char*>(p.arena_v) + dst,
                       p.item_size_bytes);
  }
  __syncthreads();
  // Phase 7: publish after both payloads arrived: reverse token, then owner tag and occupancy.
  if constexpr (!kBypass) {
    int32_t* occupancy = p.occupancy + rid * p.occupancy_slot_stride;
    for (int k = tid; k < need; k += BLOCK_SIZE) {
      const int u = s_miss_uid[k];
      const int32_t row = s_uid_row[u];
      *token_of(p, rid, row) = s_uid_token[u];
      owner[row] = my_tag;
      if (s_victim_old[k] >= 0) atomicSub(&occupancy[s_victim_old[k]], 1);
    }
    if (tid == 0 && need > 0) atomicAdd(&occupancy[p.plane], need);
  }
  __syncthreads();

  // Phase 8: fill the consumer table in the original selection order.
  for (int i = tid; i < N; i += BLOCK_SIZE) {
    const int16_t slot = s_pos_hslot[i];
    if (slot >= 0) out[i] = static_cast<int32_t>(flat_row(p, locs, s_uid_row[s_hash_val[slot]]));
  }

  // Phase 9: recency. Untouched rows keep their order; touched rows move to the end by row ID
  // (LRU: hits and admitted misses, FIFO: admitted misses only).
  for (int i = tid; i < N; i += BLOCK_SIZE) {
    int32_t value = INT_MAX;
    if (i < unique) {
      const bool miss = s_uid_miss[i] != 0;
      const bool touched = kFifo ? (miss && !kBypass) : !(miss && kBypass);
      if (touched) {
        value = s_uid_row[i];
        atomicAdd(&s_vars[4], 1);
      }
    }
    s_sort[i] = value;
  }
  __syncthreads();
  bitonic_sort<N, BLOCK_SIZE>(s_sort);
  for (int start = 0; start < p.order_size; start += BLOCK_SIZE) {
    const int j = start + tid;
    int32_t row = -1;
    bool keep = false;
    if (j < p.order_size) {
      row = region[j];
      const int u = lookup(row);
      keep = !(u >= 0 && (!kFifo || s_uid_miss[u] != 0));
    }
    int kept = 0;
    const int position = block_exclusive_scan<BLOCK_SIZE>(keep ? 1 : 0, s_scan, kept);
    const int write = s_vars[2];
    if (keep) region[write + position] = row;  // write index <= j: in-place stable compaction
    __syncthreads();
    if (tid == 0) s_vars[2] = write + kept;
    __syncthreads();
  }
  const int write = s_vars[2];
  const int touched = s_vars[4];
  if (write + touched == p.order_size) {
    for (int k = tid; k < touched; k += BLOCK_SIZE) region[write + k] = s_sort[k];
  } else if (tid == 0) {
    atomicOr(p.status, kStatusOrder);
  }

  if constexpr (kCounters) {
    if (tid == 0) {
      unsigned long long* c = p.counters + static_cast<int64_t>(p.plane) * kNumCounters;
      atomicAdd(&c[0], static_cast<unsigned long long>(unique));
      atomicAdd(&c[1], static_cast<unsigned long long>(unique - need));
      atomicAdd(&c[2], static_cast<unsigned long long>(need));
      atomicAdd(&c[3], static_cast<unsigned long long>(kBypass ? need : 0));
      atomicAdd(&c[4], static_cast<unsigned long long>(2 * need * p.item_size_bytes));
      atomicAdd(&c[5], static_cast<unsigned long long>(s_vars[3]));
      atomicAdd(&c[6], static_cast<unsigned long long>(s_vars[5]));
      atomicAdd(&c[7], static_cast<unsigned long long>(s_vars[6]));
    }
  }
}
template <auto kKernel, int BLOCK_SIZE, size_t kSmem>
inline void launch(DLDevice device, int64_t batch, const Params& params) {
  if constexpr (kSmem > 48u * 1024u) {
    host::runtime::set_max_dynamic_smem_per_device<kKernel>(device, kSmem);
  }
  host::LaunchKernel(batch, BLOCK_SIZE, host::LaunchKernel::resolve_device(device), kSmem)(kKernel, params);
}

[[noreturn]] inline void reject(const char* message) {
  throw std::runtime_error(std::string("rlh_shared_lru::resolve: ") + message);
}

// Host entry point. Shapes and dtypes are validated in Python before capture; the checks
// here guard the address arithmetic the kernel relies on.
template <
    int BLOCK_SIZE,
    int NUM_TOP_K,
    int SPARSE_BLOCK_SIZE,
    bool kFifo,
    bool kSoftFloor,
    bool kBypass,
    bool kCounters>
void resolve(
    tvm::ffi::TensorView top_k,
    tvm::ffi::TensorView req_pool_indices,
    tvm::ffi::TensorView seq_lens,
    tvm::ffi::TensorView num_real_reqs,
    tvm::ffi::TensorView row_token,
    tvm::ffi::TensorView row_owner,
    tvm::ffi::TensorView order,
    tvm::ffi::TensorView occupancy,
    tvm::ffi::TensorView floors,
    tvm::ffi::TensorView generation,
    tvm::ffi::TensorView device_locs,
    tvm::ffi::TensorView host_locs,
    tvm::ffi::TensorView host_k,
    tvm::ffi::TensorView host_v,
    tvm::ffi::TensorView arena_k,
    tvm::ffi::TensorView arena_v,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView status,
    tvm::ffi::TensorView counters,
    int64_t plane,
    int64_t num_planes,
    int64_t hot_rows,
    int64_t plane_rows,
    int64_t order_offset,
    int64_t order_size,
    int64_t scratch_offset,
    int64_t scratch_size,
    int64_t item_size_bytes) {
  using namespace host;
  constexpr int N = NUM_TOP_K * SPARSE_BLOCK_SIZE;
  constexpr size_t kSmem = SmemLayout<N, BLOCK_SIZE>::BYTES;
  const int64_t batch = top_k.shape()[0];
  if (num_planes < 1 || num_planes > kMaxPlanes) reject("the SM80 arena supports 1 to 64 planes");
  const int64_t arena_rows = num_planes * hot_rows;
  if (top_k.ndim() != 2 || top_k.shape()[1] != NUM_TOP_K || top_k.strides()[1] != 1) {
    reject("top_k must be [B, NUM_TOP_K] with unit column stride");
  }
  if (out.ndim() != 2 || out.shape()[0] != batch || out.shape()[1] < N || out.strides()[1] != 1) {
    reject("output must be [B, >= selected tokens] with unit column stride");
  }
  if (row_token.ndim() != 3 || row_token.shape()[0] != num_planes || row_token.shape()[2] < hot_rows ||
      row_token.strides()[2] != 1) {
    reject("row_token must be the native [planes, slots, H + page] token table");
  }
  if (row_owner.ndim() != 2 || row_owner.shape()[1] != arena_rows || order.ndim() != 2 ||
      order.shape()[1] != arena_rows || occupancy.ndim() != 2 || occupancy.shape()[1] != num_planes) {
    reject("arena tables must be [slots, planes * H] and [slots, planes]");
  }
  if (device_locs.ndim() != 2 || device_locs.shape()[1] <= hot_rows || host_locs.ndim() != 2) {
    reject("location tables must be [slots, H + page] and [slots, context]");
  }
  if (plane < 0 || plane >= num_planes || hot_rows <= 0 || hot_rows < N || order_offset < 0 || order_size < N ||
      order_offset + order_size > arena_rows) {
    reject("plane or domain outside the arena, or domain smaller than one selection");
  }
  if (num_planes * plane_rows > INT_MAX) reject("flat consumer index exceeds int32");
  if (kBypass && (scratch_size < N || scratch_offset < 0 || scratch_offset + scratch_size > arena_rows)) {
    reject("bypass scratch must hold one complete selection inside the arena");
  }
  if (kCounters && (counters.ndim() != 2 || counters.shape()[0] != num_planes || counters.shape()[1] != kNumCounters)) {
    reject("counters must be [planes, 8] int64");
  }
  if (batch == 0) return;

  Params params{};
  params.top_k = static_cast<const int32_t*>(top_k.data_ptr());
  params.top_k_stride = top_k.strides()[0];
  params.req_pool_indices = req_pool_indices.data_ptr();
  params.seq_lens = seq_lens.data_ptr();
  params.num_real_reqs = static_cast<const int32_t*>(num_real_reqs.data_ptr());
  params.row_token = static_cast<int32_t*>(row_token.data_ptr());
  params.token_plane_stride = row_token.strides()[0];
  params.token_slot_stride = row_token.strides()[1];
  params.row_owner = static_cast<int32_t*>(row_owner.data_ptr());
  params.owner_slot_stride = row_owner.strides()[0];
  params.order = static_cast<int32_t*>(order.data_ptr());
  params.order_slot_stride = order.strides()[0];
  params.occupancy = static_cast<int32_t*>(occupancy.data_ptr());
  params.occupancy_slot_stride = occupancy.strides()[0];
  params.floors = static_cast<const int32_t*>(floors.data_ptr());
  params.generation = static_cast<const int32_t*>(generation.data_ptr());
  params.device_locs = static_cast<const int32_t*>(device_locs.data_ptr());
  params.locs_slot_stride = device_locs.strides()[0];
  params.host_locs = static_cast<const int64_t*>(host_locs.data_ptr());
  params.host_slot_stride = host_locs.strides()[0];
  params.host_k = runtime::get_device_accessible_ptr(host_k);
  params.host_v = runtime::get_device_accessible_ptr(host_v);
  params.arena_k = arena_k.data_ptr();
  params.arena_v = arena_v.data_ptr();
  params.out = static_cast<int32_t*>(out.data_ptr());
  params.out_stride = out.strides()[0];
  params.status = static_cast<int32_t*>(status.data_ptr());
  params.counters = kCounters ? static_cast<unsigned long long*>(counters.data_ptr()) : nullptr;
  params.plane_rows = plane_rows;
  params.item_size_bytes = item_size_bytes;
  params.plane = static_cast<int32_t>(plane);
  params.num_planes = static_cast<int32_t>(num_planes);
  params.hot_rows = static_cast<int32_t>(hot_rows);
  params.order_offset = static_cast<int32_t>(order_offset);
  params.order_size = static_cast<int32_t>(order_size);
  params.scratch_offset = static_cast<int32_t>(scratch_offset);
  const auto device = top_k.device();
  const auto seq_dtype = seq_lens.dtype();
  const auto rpi_dtype = req_pool_indices.dtype();
  const bool seq_is_i64 = seq_dtype.code == kDLInt && seq_dtype.bits == 64;
  const bool rpi_is_i64 = rpi_dtype.code == kDLInt && rpi_dtype.bits == 64;
#define RLH_SHARED_LRU_LAUNCH(SEQ_T, RPI_T)                                                               \
  launch<                                                                                                 \
      resolve_kernel<BLOCK_SIZE, NUM_TOP_K, SPARSE_BLOCK_SIZE, kFifo, kSoftFloor, kBypass, kCounters, SEQ_T, \
                     RPI_T>,                                                                              \
      BLOCK_SIZE,                                                                                         \
      kSmem>(device, batch, params)
  if (seq_is_i64 && rpi_is_i64) {
    RLH_SHARED_LRU_LAUNCH(int64_t, int64_t);
  } else if (seq_is_i64) {
    RLH_SHARED_LRU_LAUNCH(int64_t, int32_t);
  } else if (rpi_is_i64) {
    RLH_SHARED_LRU_LAUNCH(int32_t, int64_t);
  } else {
    RLH_SHARED_LRU_LAUNCH(int32_t, int32_t);
  }
#undef RLH_SHARED_LRU_LAUNCH
}

}  // namespace rlh_shared_lru
}  // namespace sglang
