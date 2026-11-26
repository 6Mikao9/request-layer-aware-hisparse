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
//
// Two engineering switches change how the resolver finds rows, never which rows
// (features.owner_summary, features.touched_bitmap; each is its own template
// argument, and off compiles the original statements):
// * owner_summary keeps, per 32-row block of the arena, the bit set of planes that
//   own a live row in it. Phase 3 walks physical blocks and reads owner tags only
//   where the requesting plane's bit is set; publication recomputes the bits of
//   every block that received a victim.
// * touched_bitmap marks the rows a call hits and admits in shared memory. The
//   victim scan and the recency compaction test that bit instead of repeating the
//   owner/token/hash lookup, compaction runs on multi-entry chunks with uniform
//   fast paths, and touched rows are appended in bit order instead of being sorted.
// Three more switches remove the whole-domain work instead of speeding it up:
// * hit_index keeps, per slot, two-choice buckets of 16 entries (fingerprint | row)
//   keyed by (plane, token). Phase 3 probes two buckets per selected key and confirms
//   a fingerprint with the owner tag and reverse token; publication removes the
//   victims' old keys and inserts the new ones. A full bucket pair sets the slot's
//   sticky flag and the slot walks its domain rows until the next generation.
// * incremental_lru replaces the recency list by a ring per domain: live entries in
//   [head, tail) in list order, a dead bit per entry and each row's ring position.
//   A call kills and appends only its touched rows (sorted, as before) and scans for
//   victims from the head; the head moves past the entries the scan consumed. A
//   stable rebuild compacts the live entries when the next append could overflow and
//   at the domain's first plane once the tail passes D + K, i.e. once per decode step.
// * compact_owner stores one byte per row (plane, 254 taken, 255 free) instead of
//   generation << 16 | plane; init_slot rewrites every tag before a new generation.
// Off compiles the original statements; owner_summary/hit_index and
// touched_bitmap/incremental_lru are alternatives for one phase.
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
constexpr int32_t kStatusIndex = 4;     // a victim's key was missing from the hit index
constexpr int kNumCounters = 8;         // unique, hits, misses, bypassed, bytes, breaches, latest, invalid
constexpr int kSummaryRows = 32;        // arena rows per owner-summary block: one warp lane per row
constexpr int kSummaryUnroll = 4;       // summary blocks whose owner tags one warp loads together
constexpr int kCompactUnroll = 8;       // recency entries each thread loads per compaction chunk
constexpr int kIndexWays = 16;          // hit-index entries per bucket: four 16-byte loads
constexpr int32_t kIndexEmpty = -1;

// Owner tag of a row: generation << 16 | plane (int32), or the plane alone (compact_owner).
template <bool kCompact>
struct OwnerTags {
  using T = int32_t;
  static constexpr T kFree = kFreeTag;
  static constexpr T kTaken = kTakenTag;
  __device__ static T tag(int32_t gen, int32_t plane) { return (gen << 16) | plane; }
  __device__ static int index(T tag, int32_t gen) { return (tag < 0 || (tag >> 16) != gen) ? kFreeOwner : (tag & 0xFFFF); }
};
template <>
struct OwnerTags<true> {
  using T = uint8_t;
  static constexpr T kFree = 255;
  static constexpr T kTaken = 254;
  __device__ static T tag(int32_t, int32_t plane) { return static_cast<T>(plane); }
  __device__ static int index(T tag, int32_t) { return tag < kMaxPlanes ? tag : kFreeOwner; }
};

struct Params {
  const int32_t* top_k;             // [B, NUM_TOP_K] selector blocks, -1 invalid
  int64_t top_k_stride;
  const void* req_pool_indices;     // [B] int32 or int64
  const void* seq_lens;             // [B] int32 or int64
  const int32_t* num_real_reqs;     // [1] real rows of the captured batch
  int32_t* row_token;               // native req_device_buffer_tokens [planes, slots, cols]
  int64_t token_plane_stride;
  int64_t token_slot_stride;
  void* row_owner;                  // [slots, planes * H] owner tag (int32, or uint8 with compact_owner)
  int64_t owner_slot_stride;
  int32_t* order;                   // [slots, planes * H] recency lists, one region per domain (not incremental_lru)
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
  unsigned long long* owner_summary;  // [slots, ceil(planes * H / 32)] owner-plane bits per block (owner_summary)
  int64_t summary_slot_stride;
  int32_t* index;                   // [slots, 2 * index_half * 16] fingerprint << row_bits | row, -1 empty (hit_index)
  int64_t index_slot_stride;
  int32_t* index_flags;             // [slots] sticky: a full bucket pair parked the index (hit_index)
  int32_t* ring;                    // [slots, sum of capacities] recency rings (incremental_lru)
  int64_t ring_slot_stride;
  uint32_t* ring_dead;              // [slots, ring words] dead bit per ring entry
  int64_t dead_slot_stride;
  int32_t* ring_pos;                // [slots, planes * H] ring index of each domain row
  int64_t pos_slot_stride;
  int32_t* ring_meta;               // [slots, domains * 4] head, tail, rebuilds, 0
  int64_t meta_slot_stride;
  int64_t plane_rows;
  int64_t item_size_bytes;
  int32_t plane;
  int32_t num_planes;
  int32_t hot_rows;
  int32_t arena_rows;
  int32_t order_offset;
  int32_t order_size;
  int32_t scratch_offset;
  int32_t index_half;               // buckets per half: the left bucket lies below it, the right one above
  int32_t row_bits;                 // row bits of an index entry; the fingerprint keeps 31 - row_bits
  int32_t ring_offset;              // this domain's ring region, dead words and meta entry
  int32_t ring_capacity;
  int32_t dead_offset;
  int32_t meta_offset;
  int32_t rebuild_tail;             // first plane of the domain: rebuild once the tail passes D + K
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
  // touched_bitmap: one bit per domain row after the fixed layout (size depends on the domain).
  static constexpr size_t BITMAP_OFFSET = (BYTES + 15) / 16 * 16;
  static constexpr size_t bytes_with_bitmap(int64_t domain_rows) {
    return BITMAP_OFFSET + static_cast<size_t>((domain_rows + 31) / 32) * sizeof(uint32_t);
  }
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

// Shared-memory bit set over domain-relative row indices (touched_bitmap).
__device__ __forceinline__ bool bit_test(const uint32_t* bits, int index) {
  return (bits[index >> 5] >> (index & 31)) & 1u;
}
__device__ __forceinline__ void bit_set(uint32_t* bits, int index) {
  atomicOr(&bits[index >> 5], 1u << (index & 31));
}
__device__ __forceinline__ void bit_clear(uint32_t* bits, int index) {
  atomicAnd(&bits[index >> 5], ~(1u << (index & 31)));
}

// hit_index key of (plane, token): murmur3 fmix64, as shared_lru.py::index_hash.
__host__ __device__ __forceinline__ uint64_t index_mix(int32_t plane, int32_t token) {
  uint64_t x = (static_cast<uint64_t>(static_cast<uint32_t>(token)) << 6) | static_cast<uint64_t>(plane);
  x ^= x >> 33;
  x *= 0xFF51AFD7ED558CCDull;
  x ^= x >> 33;
  x *= 0xC4CEB9FE1A85EC53ull;
  return x ^ (x >> 33);
}
struct IndexKey {
  int left;         // bucket in [0, half)
  int right;        // bucket in [half, 2 * half)
  int32_t fp;       // fingerprint << row_bits
};
__host__ __device__ __forceinline__ IndexKey index_key(const Params& p, int32_t plane, int32_t token) {
  const uint64_t x = index_mix(plane, token);
  const uint64_t half = static_cast<uint64_t>(p.index_half);
  const uint32_t mask = (1u << (31 - p.row_bits)) - 1u;
  return IndexKey{static_cast<int>(((x & 0xFFFFFFFFull) * half) >> 32),
                  p.index_half + static_cast<int>(((x >> 32) * half) >> 32),
                  static_cast<int32_t>((static_cast<uint32_t>(x >> 32) & mask) << p.row_bits)};
}

// incremental_lru: stable compaction of the live ring entries [head, tail) to [0, live), new
// positions for every moved row, all dead bits cleared. Chunks of kCompactUnroll entries per thread
// as in the touched_bitmap compaction: every write goes to an index <= the entry's own and every
// read of a chunk precedes its first barrier. Every thread must call it; returns the live count.
template <int BLOCK_SIZE>
__device__ int ring_rebuild(int32_t* ring, uint32_t* dead, int32_t* pos, int head, int tail, int capacity,
                            int32_t* s_count) {
  constexpr int WARPS = BLOCK_SIZE / WARP_SIZE;
  constexpr int CHUNK = BLOCK_SIZE * kCompactUnroll;
  constexpr int COUNTS = kCompactUnroll * WARPS;
  constexpr int SEGMENT = (COUNTS + WARP_SIZE - 1) / WARP_SIZE;
  const int tid = threadIdx.x;
  const int lane = tid % WARP_SIZE;
  const int warp = tid / WARP_SIZE;
  int write = 0;
  for (int start = head; start < tail; start += CHUNK) {
    int32_t row[kCompactUnroll];
    bool keep[kCompactUnroll];
    int removed = 0;
#pragma unroll
    for (int k = 0; k < kCompactUnroll; ++k) {
      const int j = start + k * BLOCK_SIZE + tid;
      keep[k] = j < tail && !((dead[j >> 5] >> (j & 31)) & 1u);
      row[k] = keep[k] ? ring[j] : -1;
      removed += j < tail && !keep[k] ? 1 : 0;
    }
    const int entries = min(CHUNK, tail - start);
    if (__syncthreads_or(removed) == 0) {
      if (write != start) {
#pragma unroll
        for (int k = 0; k < kCompactUnroll; ++k) {
          const int at = write + k * BLOCK_SIZE + tid;
          if (keep[k]) {
            ring[at] = row[k];
            pos[row[k]] = at;
          }
        }
      }
      write += entries;
      continue;
    }
    int rank[kCompactUnroll];
#pragma unroll
    for (int k = 0; k < kCompactUnroll; ++k) {  // list order inside a chunk is (k, warp, lane)
      const unsigned kept = __ballot_sync(FULL_WARP_MASK, keep[k]);
      rank[k] = __popc(kept & ((1u << lane) - 1u));
      if (lane == 0) s_count[k * WARPS + warp] = __popc(kept);
    }
    __syncthreads();
    if (warp == 0) {  // exclusive prefix of the counts; s_count[COUNTS] is the chunk total
      const int first = min(COUNTS, lane * SEGMENT);
      const int end = min(COUNTS, first + SEGMENT);
      int sum = 0;
      for (int i = first; i < end; ++i) sum += s_count[i];
      int prefix = sum;
#pragma unroll
      for (int d = 1; d < WARP_SIZE; d <<= 1) {
        const int n = __shfl_up_sync(FULL_WARP_MASK, prefix, d);
        if (lane >= d) prefix += n;
      }
      if (lane == WARP_SIZE - 1) s_count[COUNTS] = prefix;
      int running = prefix - sum;
      for (int i = first; i < end; ++i) {
        const int count = s_count[i];
        s_count[i] = running;
        running += count;
      }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kCompactUnroll; ++k) {
      if (keep[k]) {
        const int at = write + s_count[k * WARPS + warp] + rank[k];
        ring[at] = row[k];
        pos[row[k]] = at;
      }
    }
    write += s_count[COUNTS];
  }
  __syncthreads();  // every dead bit was read
  for (int w = tid; w < (capacity + 31) / 32; w += BLOCK_SIZE) dead[w] = 0u;
  __syncthreads();
  return write;
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
    bool kOwnerSummary,
    bool kTouchedBitmap,
    bool kHitIndex,
    bool kIncremental,
    bool kCompactOwner,
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
  static_assert(!(kOwnerSummary && kHitIndex) && !(kTouchedBitmap && kIncremental),
                "owner_summary/hit_index and touched_bitmap/incremental_lru implement the same phase");
  using Tags = OwnerTags<kCompactOwner>;
  using OwnerT = typename Tags::T;

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
  uint32_t* s_touched = reinterpret_cast<uint32_t*>(smem_raw + SmemLayout<N, BLOCK_SIZE>::BITMAP_OFFSET);
  const int bitmap_words = (p.order_size + 31) / 32;  // allocated only by touched_bitmap launches
  for (int i = tid; i < HASH; i += BLOCK_SIZE) {
    s_hash_key[i] = kHashEmpty;
    s_hash_val[i] = INT_MAX;
  }
  if constexpr (kTouchedBitmap) {
    for (int w = tid; w < bitmap_words; w += BLOCK_SIZE) s_touched[w] = 0u;
  }
  if (tid < 8) s_vars[tid] = kIncremental && tid == 2 ? -1 : 0;  // incremental_lru: 2 = last ring entry taken
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
  const OwnerT my_tag = Tags::tag(gen, p.plane);
  int32_t* region = kIncremental ? nullptr : p.order + rid * p.order_slot_stride + p.order_offset;
  OwnerT* owner = static_cast<OwnerT*>(p.row_owner) + rid * p.owner_slot_stride;
  // Unique id of a row that holds a selected token of this layer and generation, else -1.
  auto lookup = [&](int32_t row) -> int {
    if (owner[row] != my_tag) return -1;
    const int slot = hash_find<HASH>(s_hash_key, *token_of(p, rid, row));
    return slot < 0 ? -1 : s_hash_val[slot];
  };
  auto owner_index = [&](OwnerT tag) -> int { return Tags::index(tag, gen); };
  auto record_hit = [&](int u, int32_t row) {
    s_uid_row[u] = row;
    atomicAdd(&s_vars[0], 1);
    if constexpr (kTouchedBitmap) bit_set(s_touched, row - p.order_offset);
  };
  // incremental_lru: this domain's ring. head/tail are uniform registers; a call that could overflow the
  // ring, or the domain's first plane once the tail passed D + K, compacts the live entries first.
  int32_t* ring = nullptr;
  uint32_t* dead = nullptr;
  int32_t* ring_pos = nullptr;
  int32_t* meta = nullptr;
  int ring_head = 0;
  int ring_tail = 0;
  if constexpr (kIncremental) {
    ring = p.ring + rid * p.ring_slot_stride + p.ring_offset;
    dead = p.ring_dead + rid * p.dead_slot_stride + p.dead_offset;
    ring_pos = p.ring_pos + rid * p.pos_slot_stride;
    meta = p.ring_meta + rid * p.meta_slot_stride + p.meta_offset;
    ring_head = meta[0];
    ring_tail = meta[1];
    if (ring_tail + N > p.ring_capacity || ring_tail > p.rebuild_tail) {
      const int rebuilds = meta[2];
      ring_tail = ring_rebuild<BLOCK_SIZE>(ring, dead, ring_pos, ring_head, ring_tail, p.ring_capacity, s_owner_tab);
      ring_head = 0;
      if (tid == 0) {
        meta[0] = 0;
        meta[1] = ring_tail;
        meta[2] = rebuilds + 1;
        if (ring_tail != p.order_size) atomicOr(p.status, kStatusOrder);
      }
    }
  }
  // hit_index: the slot's index serves lookups until a full bucket pair parks it for this generation.
  const bool index_live = kHitIndex && p.index_flags[rid] == 0;
  // Every domain row once: the recency list, or the contiguous row range when the list is a ring.
  auto walk_domain_hits = [&]() {
    if constexpr (kIncremental) {
      for (int row = p.order_offset + tid; row < p.order_offset + p.order_size; row += BLOCK_SIZE) {
        const int u = lookup(row);
        if (u >= 0) record_hit(u, row);
      }
    } else {
      for (int j = tid; j < p.order_size; j += BLOCK_SIZE) {
        const int32_t row = region[j];
        const int u = lookup(row);
        if (u >= 0) record_hit(u, row);
      }
    }
  };

  // Phase 3: snapshot every hit of the complete selection before choosing any victim.
  if (kHitIndex && index_live) {
    // hit_index: probe the left then the right bucket of each selected key. A fingerprint names a
    // candidate row; its owner tag and reverse token confirm the key, so a match is never wrong.
    const int32_t* index = p.index + rid * p.index_slot_stride;
    const int32_t row_mask = (1 << p.row_bits) - 1;
    for (int u = tid; u < unique; u += BLOCK_SIZE) {
      const int32_t token = s_uid_token[u];
      const IndexKey key = index_key(p, p.plane, token);
      int32_t found = -1;
      for (int side = 0; side < 2 && found < 0; ++side) {
        const int4* bucket = reinterpret_cast<const int4*>(index + static_cast<int64_t>(side ? key.right : key.left) *
                                                                       kIndexWays);
#pragma unroll
        for (int q = 0; q < kIndexWays / 4; ++q) {
          const int4 four = bucket[q];
          const int32_t entry[4] = {four.x, four.y, four.z, four.w};
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            if (found < 0 && entry[w] >= 0 && (entry[w] & ~row_mask) == key.fp) {
              const int32_t row = entry[w] & row_mask;
              if (owner[row] == my_tag && *token_of(p, rid, row) == token) found = row;
            }
          }
        }
      }
      if (found >= 0) record_hit(u, found);
    }
  } else if constexpr (kOwnerSummary) {
    // owner_summary: the domain's rows are one contiguous range. Walk it in 32-row blocks and read
    // owner tags only where the block's summary has this plane's bit; summaries cover every live
    // row, so this finds exactly the rows the recency-list walk would find.
    const unsigned long long* summary = p.owner_summary + rid * p.summary_slot_stride;
    const unsigned long long plane_bit = 1ull << p.plane;
    const int lo = p.order_offset;
    const int hi = p.order_offset + p.order_size;
    const int last = (hi - 1) / kSummaryRows;
    for (int b0 = lo / kSummaryRows + warp * WARP_SIZE; b0 <= last; b0 += WARPS * WARP_SIZE) {
      const int b = b0 + lane;
      unsigned pending = __ballot_sync(FULL_WARP_MASK, b <= last && (summary[b] & plane_bit) != 0ull);
      while (pending != 0u) {  // warp-uniform; owner tags and reverse tokens of up to kSummaryUnroll
                               // blocks load together (for H % 32 == 0 a block lies in one plane)
        int32_t row[kSummaryUnroll];
        OwnerT tag[kSummaryUnroll];
        int32_t token[kSummaryUnroll];
#pragma unroll
        for (int q = 0; q < kSummaryUnroll; ++q) {
          row[q] = -1;
          if (pending != 0u) {
            const int r = (b0 + __ffs(pending) - 1) * kSummaryRows + lane;
            pending &= pending - 1u;
            if (r >= lo && r < hi) row[q] = r;
          }
          tag[q] = row[q] >= 0 ? owner[row[q]] : Tags::kFree;
          token[q] = row[q] >= 0 ? *token_of(p, rid, row[q]) : kHashEmpty;
        }
#pragma unroll
        for (int q = 0; q < kSummaryUnroll; ++q) {
          if (tag[q] == my_tag) {
            const int slot = hash_find<HASH>(s_hash_key, token[q]);
            if (slot >= 0) record_hit(s_hash_val[slot], row[q]);
          }
        }
      }
    }
  } else {
    walk_domain_hits();
  }
  __syncthreads();
  const int need = unique - s_vars[0];
  // A domain row holds a selected token of this layer: the phase-3 hits.
  auto is_hit = [&](int32_t row) -> bool {
    if constexpr (kTouchedBitmap) {
      return bit_test(s_touched, row - p.order_offset);
    } else {
      return lookup(row) >= 0;
    }
  };

  // Phase 4: ordered victim scan of the domain list (free rows lead it). A floor-aware pass
  // takes a row only while its owner keeps more rows than its soft floor. incremental_lru scans the
  // ring from its head, skips dead entries and records the last ring entry it took.
  auto scan_victims = [&](bool floor_aware) {
    const int first = kIncremental ? ring_head : 0;
    const int end = kIncremental ? ring_tail : p.order_size;
    for (int start = first; start < end; start += BLOCK_SIZE) {
      if (s_vars[1] >= need) break;
      const int j = start + tid;
      int32_t row = -1;
      int owner_idx = kFreeOwner;
      bool candidate = false;
      if (j < end) {
        if constexpr (kIncremental) {
          row = (dead[j >> 5] >> (j & 31)) & 1u ? -1 : ring[j];
        } else {
          row = region[j];
        }
      }
      if (row >= 0) {
        const OwnerT tag = owner[row];
        candidate = tag != Tags::kTaken && !is_hit(row);
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
        owner[row] = Tags::kTaken;
        if (floor_aware && owner_idx != kFreeOwner) atomicSub(&s_surplus[owner_idx], 1);
        if constexpr (kIncremental) atomicMax(&s_vars[2], j);
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
      owner[s_victim_row[k]] = old < 0 ? Tags::kFree : Tags::tag(gen, old);
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

  // hit_index: drop the victims' old keys while their reverse tokens still name them, then insert the
  // new keys into the less loaded of their two buckets (ties left). Every entry names one row, so
  // removals never collide; a CAS claims each empty way. A full pair parks the slot's index.
  if constexpr (kHitIndex && !kBypass) {
    if (index_live) {
      int32_t* index = p.index + rid * p.index_slot_stride;
      for (int k = tid; k < need; k += BLOCK_SIZE) {
        const int old = s_victim_old[k];
        if (old < 0) continue;  // a free row carries no key
        const int32_t row = s_victim_row[k];
        const IndexKey key = index_key(p, old, *token_of(p, rid, row));
        bool removed = false;
        for (int side = 0; side < 2 && !removed; ++side) {
          int32_t* bucket = index + static_cast<int64_t>(side ? key.right : key.left) * kIndexWays;
          for (int w = 0; w < kIndexWays && !removed; ++w) {
            if (bucket[w] == (key.fp | row)) {
              bucket[w] = kIndexEmpty;
              removed = true;
            }
          }
        }
        if (!removed) atomicOr(p.status, kStatusIndex);
      }
      __syncthreads();
      for (int k = tid; k < need; k += BLOCK_SIZE) {
        const int u = s_miss_uid[k];
        const IndexKey key = index_key(p, p.plane, s_uid_token[u]);
        int32_t* left = index + static_cast<int64_t>(key.left) * kIndexWays;
        int32_t* right = index + static_cast<int64_t>(key.right) * kIndexWays;
        int load_left = 0;
        int load_right = 0;
        for (int w = 0; w < kIndexWays; ++w) {
          load_left += left[w] != kIndexEmpty ? 1 : 0;
          load_right += right[w] != kIndexEmpty ? 1 : 0;
        }
        bool placed = false;
        for (int side = 0; side < 2 && !placed; ++side) {
          int32_t* bucket = (side == 0) == (load_left <= load_right) ? left : right;
          for (int w = 0; w < kIndexWays && !placed; ++w) {
            placed = bucket[w] == kIndexEmpty && atomicCAS(&bucket[w], kIndexEmpty, key.fp | s_uid_row[u]) == kIndexEmpty;
          }
        }
        if (!placed) atomicOr(&p.index_flags[rid], 1);
      }
    }
  }

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
  if constexpr (kOwnerSummary && !kBypass) {
    // owner_summary: recompute every block that received a victim from the published tags, so a
    // block's bits stay exactly the planes owning a live row in it (the new owner included).
    unsigned long long* summary = p.owner_summary + rid * p.summary_slot_stride;
    for (int k = warp; k < need; k += WARPS) {
      const int b = s_victim_row[k] / kSummaryRows;
      const int r = b * kSummaryRows + lane;
      const int o = r < p.arena_rows ? owner_index(owner[r]) : kFreeOwner;
      const unsigned low = __reduce_or_sync(FULL_WARP_MASK, o < 32 ? 1u << o : 0u);
      const unsigned high = __reduce_or_sync(FULL_WARP_MASK, o >= 32 && o < kMaxPlanes ? 1u << (o - 32) : 0u);
      if (lane == 0) summary[b] = (static_cast<unsigned long long>(high) << 32) | low;
    }
  }

  // Phase 8: fill the consumer table in the original selection order.
  for (int i = tid; i < N; i += BLOCK_SIZE) {
    const int16_t slot = s_pos_hslot[i];
    if (slot >= 0) out[i] = static_cast<int32_t>(flat_row(p, locs, s_uid_row[s_hash_val[slot]]));
  }

  // Phase 9: recency. Untouched rows keep their order; touched rows move to the end by row ID
  // (LRU: hits and admitted misses, FIFO: admitted misses only).
  if constexpr (kTouchedBitmap) {
    // touched_bitmap: the bitmap holds the hits; turn it into the touched set.
    for (int u = tid; u < unique; u += BLOCK_SIZE) {
      const bool miss = s_uid_miss[u] != 0;
      if (miss && !kBypass) bit_set(s_touched, s_uid_row[u] - p.order_offset);
      if (!miss && kFifo) bit_clear(s_touched, s_uid_row[u] - p.order_offset);
      if (kFifo ? (miss && !kBypass) : !(miss && kBypass)) atomicAdd(&s_vars[4], 1);
    }
    __syncthreads();
    // Stable in-place compaction in chunks of kCompactUnroll entries per thread. A chunk without a
    // touched row is left in place (or shifted as a whole) without a scan. Every write goes to an
    // index <= the entry's own, and every read of a chunk precedes its first barrier.
    constexpr int CHUNK = BLOCK_SIZE * kCompactUnroll;
    constexpr int COUNTS = kCompactUnroll * WARPS;  // kept entries per (sub-chunk, warp), list order
    constexpr int SEGMENT = (COUNTS + WARP_SIZE - 1) / WARP_SIZE;
    static_assert(COUNTS + 1 <= WARPS * kOwners, "chunk counts reuse the floor-aware owner table");
    int32_t* s_count = s_owner_tab;
    int write = 0;
    for (int start = 0; start < p.order_size; start += CHUNK) {
      int32_t row[kCompactUnroll];
      bool keep[kCompactUnroll];
      int removed = 0;
#pragma unroll
      for (int k = 0; k < kCompactUnroll; ++k) {
        const int j = start + k * BLOCK_SIZE + tid;
        row[k] = j < p.order_size ? region[j] : -1;
      }
#pragma unroll
      for (int k = 0; k < kCompactUnroll; ++k) {
        keep[k] = row[k] >= 0 && !bit_test(s_touched, row[k] - p.order_offset);
        removed += row[k] >= 0 && !keep[k] ? 1 : 0;
      }
      const int entries = min(CHUNK, p.order_size - start);
      if (__syncthreads_or(removed) == 0) {
        if (write != start) {
#pragma unroll
          for (int k = 0; k < kCompactUnroll; ++k) {
            if (row[k] >= 0) region[write + k * BLOCK_SIZE + tid] = row[k];
          }
        }
        write += entries;
        continue;
      }
      int rank[kCompactUnroll];
#pragma unroll
      for (int k = 0; k < kCompactUnroll; ++k) {  // list order inside a chunk is (k, warp, lane)
        const unsigned kept = __ballot_sync(FULL_WARP_MASK, keep[k]);
        rank[k] = __popc(kept & ((1u << lane) - 1u));
        if (lane == 0) s_count[k * WARPS + warp] = __popc(kept);
      }
      __syncthreads();
      if (warp == 0) {  // exclusive prefix of the counts; s_count[COUNTS] is the chunk total
        const int first = min(COUNTS, lane * SEGMENT);
        const int end = min(COUNTS, first + SEGMENT);
        int sum = 0;
        for (int i = first; i < end; ++i) sum += s_count[i];
        int prefix = sum;
#pragma unroll
        for (int d = 1; d < WARP_SIZE; d <<= 1) {
          const int n = __shfl_up_sync(FULL_WARP_MASK, prefix, d);
          if (lane >= d) prefix += n;
        }
        if (lane == WARP_SIZE - 1) s_count[COUNTS] = prefix;
        int running = prefix - sum;
        for (int i = first; i < end; ++i) {
          const int count = s_count[i];
          s_count[i] = running;
          running += count;
        }
      }
      __syncthreads();
#pragma unroll
      for (int k = 0; k < kCompactUnroll; ++k) {
        if (keep[k]) region[write + s_count[k * WARPS + warp] + rank[k]] = row[k];
      }
      write += s_count[COUNTS];
    }
    // Touched rows in ascending row ID are the set bits in index order: no sort.
    const int per = (bitmap_words + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int w_begin = min(bitmap_words, tid * per);
    const int w_end = min(bitmap_words, w_begin + per);
    int mine = 0;
    for (int w = w_begin; w < w_end; ++w) mine += __popc(s_touched[w]);
    int set_bits = 0;
    int position = write + block_exclusive_scan<BLOCK_SIZE>(mine, s_scan, set_bits);
    const int touched = s_vars[4];
    if (write + touched == p.order_size && set_bits == touched) {
      for (int w = w_begin; w < w_end; ++w) {
        for (unsigned bits = s_touched[w]; bits != 0u; bits &= bits - 1u) {
          region[position++] = p.order_offset + w * 32 + __ffs(bits) - 1;
        }
      }
    } else if (tid == 0) {
      atomicOr(p.status, kStatusOrder);
    }
  } else if constexpr (kIncremental) {
    // incremental_lru: sort the touched rows as the plain update does, kill their ring entries, append
    // them in row order and move the head past the entries the victim scan consumed.
    for (int i = tid; i < N; i += BLOCK_SIZE) {
      int32_t value = INT_MAX;
      if (i < unique) {
        const bool miss = s_uid_miss[i] != 0;
        if (kFifo ? (miss && !kBypass) : !(miss && kBypass)) {
          value = s_uid_row[i];
          atomicAdd(&s_vars[4], 1);
        }
      }
      s_sort[i] = value;
    }
    __syncthreads();
    bitonic_sort<N, BLOCK_SIZE>(s_sort);
    const int touched = s_vars[4];
    for (int k = tid; k < touched; k += BLOCK_SIZE) {  // each live entry dies once
      const int32_t at = ring_pos[s_sort[k]];
      bool fresh = false;
      if (at >= ring_head && at < ring_tail) {
        fresh = !((atomicOr(&dead[at >> 5], 1u << (at & 31)) >> (at & 31)) & 1u);
      }
      if (!fresh) atomicOr(p.status, kStatusOrder);
    }
    __syncthreads();
    for (int k = tid; k < touched; k += BLOCK_SIZE) {
      ring[ring_tail + k] = s_sort[k];
      ring_pos[s_sort[k]] = ring_tail + k;
    }
    const int consumed = max(ring_head, s_vars[2] + 1);  // one past the last ring entry the scan took
    if (tid == 0) s_vars[7] = consumed;
    __syncthreads();
    for (int j = ring_head + tid; j < consumed; j += BLOCK_SIZE) {
      if (!((dead[j >> 5] >> (j & 31)) & 1u)) atomicMin(&s_vars[7], j);  // a skipped row stays live
    }
    __syncthreads();
    if (tid == 0) {
      meta[0] = s_vars[7];
      meta[1] = ring_tail + touched;
    }
  } else {
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
[[noreturn]] inline void reject(const char* message) {
  throw std::runtime_error(std::string("rlh_shared_lru::resolve: ") + message);
}

// kRuntimeSmem (touched_bitmap): the bitmap grows with the domain, so the kernel opts in once to the
// device's shared-memory limit and every launch checks its own size against it.
template <auto kKernel, int BLOCK_SIZE, size_t kSmem, bool kRuntimeSmem>
inline void launch(DLDevice device, int64_t batch, const Params& params, size_t smem) {
  if constexpr (kRuntimeSmem) {
    const size_t limit = host::runtime::get_max_smem_per_block(device.device_id);
    if (smem > limit) reject("the touched-row bitmap of this domain exceeds the device shared memory");
    host::runtime::set_max_dynamic_smem_per_device<kKernel>(device, limit);
  } else if constexpr (kSmem > 48u * 1024u) {
    host::runtime::set_max_dynamic_smem_per_device<kKernel>(device, kSmem);
  }
  host::LaunchKernel(batch, BLOCK_SIZE, host::LaunchKernel::resolve_device(device), smem)(kKernel, params);
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
    bool kCounters,
    bool kOwnerSummary,
    bool kTouchedBitmap,
    bool kHitIndex,
    bool kIncremental,
    bool kCompactOwner>
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
    tvm::ffi::TensorView owner_summary,
    tvm::ffi::TensorView index,
    tvm::ffi::TensorView index_flags,
    tvm::ffi::TensorView ring,
    tvm::ffi::TensorView ring_dead,
    tvm::ffi::TensorView ring_pos,
    tvm::ffi::TensorView ring_meta,
    int64_t plane,
    int64_t num_planes,
    int64_t hot_rows,
    int64_t plane_rows,
    int64_t order_offset,
    int64_t order_size,
    int64_t scratch_offset,
    int64_t scratch_size,
    int64_t item_size_bytes,
    int64_t index_half,
    int64_t row_bits,
    int64_t ring_offset,
    int64_t ring_capacity,
    int64_t dead_offset,
    int64_t meta_offset,
    int64_t rebuild_tail) {
  using namespace host;
  constexpr int N = NUM_TOP_K * SPARSE_BLOCK_SIZE;
  constexpr size_t kSmem = SmemLayout<N, BLOCK_SIZE>::BYTES;
  static_assert(!(kOwnerSummary && kHitIndex) && !(kTouchedBitmap && kIncremental),
                "owner_summary/hit_index and touched_bitmap/incremental_lru implement the same phase");
  const int64_t batch = top_k.shape()[0];
  if (num_planes < 1 || num_planes > kMaxPlanes) reject("the SM80 arena supports 1 to 64 planes");
  const int64_t arena_rows = num_planes * hot_rows;
  const int64_t slots = row_owner.shape()[0];
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
  if (row_owner.ndim() != 2 || row_owner.shape()[1] != arena_rows || row_owner.strides()[1] != 1 ||
      row_owner.dtype().bits != (kCompactOwner ? 8 : 32) || occupancy.ndim() != 2 ||
      occupancy.shape()[1] != num_planes) {
    reject("owner tags must be [slots, planes * H] int32 (uint8 with compact_owner); occupancy [slots, planes]");
  }
  if (!kIncremental && (order.ndim() != 2 || order.shape()[1] < order_offset + order_size)) {
    reject("order must be [slots, >= domain end]");
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
  const int64_t summary_blocks = (arena_rows + kSummaryRows - 1) / kSummaryRows;
  if (kOwnerSummary && (owner_summary.ndim() != 2 || owner_summary.shape()[0] != row_owner.shape()[0] ||
                        owner_summary.shape()[1] != summary_blocks || owner_summary.strides()[1] != 1 ||
                        owner_summary.dtype().bits != 64)) {
    reject("owner_summary must be [slots, ceil(planes * H / 32)] 64-bit words");
  }
  if (kHitIndex && (row_bits < 1 || row_bits > 23 || (int64_t{1} << row_bits) < arena_rows || index_half < 1 ||
                    index.ndim() != 2 || index.shape()[0] != slots || index.strides()[1] != 1 ||
                    index.shape()[1] != 2 * index_half * kIndexWays || index.strides()[0] % 4 != 0 ||
                    reinterpret_cast<uintptr_t>(index.data_ptr()) % 16 != 0 || index_flags.ndim() != 1 ||
                    index_flags.shape()[0] != slots)) {
    reject("hit_index must be [slots, 2 * half * 16] 16-byte aligned int32 with one flag per slot");
  }
  if (kIncremental && (ring_offset < 0 || ring_capacity < order_size + N || dead_offset < 0 || meta_offset < 0 ||
                       ring.ndim() != 2 || ring.shape()[0] != slots || ring.shape()[1] < ring_offset + ring_capacity ||
                       ring_dead.ndim() != 2 || ring_dead.shape()[0] != slots ||
                       ring_dead.shape()[1] < dead_offset + (ring_capacity + 31) / 32 || ring_pos.ndim() != 2 ||
                       ring_pos.shape()[0] != slots || ring_pos.shape()[1] != arena_rows || ring_meta.ndim() != 2 ||
                       ring_meta.shape()[0] != slots || ring_meta.shape()[1] < meta_offset + 4 ||
                       ring.strides()[1] != 1 || ring_dead.strides()[1] != 1 || ring_pos.strides()[1] != 1 ||
                       ring_meta.strides()[1] != 1 || ring_offset + ring_capacity > INT_MAX)) {
    reject("incremental_lru ring tables must cover this domain: ring, dead words, positions and meta");
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
  params.row_owner = row_owner.data_ptr();
  params.owner_slot_stride = row_owner.strides()[0];
  params.order = kIncremental ? nullptr : static_cast<int32_t*>(order.data_ptr());
  params.order_slot_stride = kIncremental ? 0 : order.strides()[0];
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
  params.owner_summary = kOwnerSummary ? static_cast<unsigned long long*>(owner_summary.data_ptr()) : nullptr;
  params.summary_slot_stride = kOwnerSummary ? owner_summary.strides()[0] : 0;
  params.index = kHitIndex ? static_cast<int32_t*>(index.data_ptr()) : nullptr;
  params.index_slot_stride = kHitIndex ? index.strides()[0] : 0;
  params.index_flags = kHitIndex ? static_cast<int32_t*>(index_flags.data_ptr()) : nullptr;
  params.ring = kIncremental ? static_cast<int32_t*>(ring.data_ptr()) : nullptr;
  params.ring_slot_stride = kIncremental ? ring.strides()[0] : 0;
  params.ring_dead = kIncremental ? static_cast<uint32_t*>(ring_dead.data_ptr()) : nullptr;
  params.dead_slot_stride = kIncremental ? ring_dead.strides()[0] : 0;
  params.ring_pos = kIncremental ? static_cast<int32_t*>(ring_pos.data_ptr()) : nullptr;
  params.pos_slot_stride = kIncremental ? ring_pos.strides()[0] : 0;
  params.ring_meta = kIncremental ? static_cast<int32_t*>(ring_meta.data_ptr()) : nullptr;
  params.meta_slot_stride = kIncremental ? ring_meta.strides()[0] : 0;
  params.plane_rows = plane_rows;
  params.item_size_bytes = item_size_bytes;
  params.plane = static_cast<int32_t>(plane);
  params.num_planes = static_cast<int32_t>(num_planes);
  params.hot_rows = static_cast<int32_t>(hot_rows);
  params.arena_rows = static_cast<int32_t>(arena_rows);
  params.order_offset = static_cast<int32_t>(order_offset);
  params.order_size = static_cast<int32_t>(order_size);
  params.scratch_offset = static_cast<int32_t>(scratch_offset);
  params.index_half = static_cast<int32_t>(index_half);
  params.row_bits = static_cast<int32_t>(row_bits);
  params.ring_offset = static_cast<int32_t>(ring_offset);
  params.ring_capacity = static_cast<int32_t>(ring_capacity);
  params.dead_offset = static_cast<int32_t>(dead_offset);
  params.meta_offset = static_cast<int32_t>(meta_offset);
  params.rebuild_tail = static_cast<int32_t>(rebuild_tail < INT_MAX ? rebuild_tail : INT_MAX);
  const size_t smem = kTouchedBitmap ? SmemLayout<N, BLOCK_SIZE>::bytes_with_bitmap(order_size) : kSmem;
  const auto device = top_k.device();
  const auto seq_dtype = seq_lens.dtype();
  const auto rpi_dtype = req_pool_indices.dtype();
  const bool seq_is_i64 = seq_dtype.code == kDLInt && seq_dtype.bits == 64;
  const bool rpi_is_i64 = rpi_dtype.code == kDLInt && rpi_dtype.bits == 64;
#define RLH_SHARED_LRU_LAUNCH(SEQ_T, RPI_T)                                                                     \
  launch<                                                                                                       \
      resolve_kernel<BLOCK_SIZE, NUM_TOP_K, SPARSE_BLOCK_SIZE, kFifo, kSoftFloor, kBypass, kCounters,          \
                     kOwnerSummary, kTouchedBitmap, kHitIndex, kIncremental, kCompactOwner, SEQ_T, RPI_T>,     \
      BLOCK_SIZE,                                                                                               \
      kSmem,                                                                                                    \
      kTouchedBitmap>(device, batch, params, smem)
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
