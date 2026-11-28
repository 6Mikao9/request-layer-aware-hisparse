"""CPU checks of the CUDA route kernels in elastic_route.cuh.

* Syntax/type check with clang's CUDA front end (sm_80) on the stub headers of test_shared_lru_kernel (the pinned
  SGLang/tvm-ffi signatures the kernels touch). Not an nvcc build.
* The kernels themselves compiled as host C++ (CUDA keywords removed, one OS thread per CUDA thread where a block
  synchronizes) and run against the ElasticKV torch statements on random tables, ratio 1 and 4.

Neither says anything about the GPU; ``elastic_route.self_check`` and ``runtime.cuda_elastic_gate`` check that."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import unittest

try:
    import torch
except ImportError:
    torch = None

from .test_shared_lru_kernel import HEADERS, clang_cuda

KERNEL = Path(__file__).with_name("elastic_route.cuh")
HOST_STUB = r"""
#pragma once
#include <cstddef>
#include <cstdint>
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
struct dim3 { unsigned x, y, z; dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
struct uint3 { unsigned x, y, z; };
extern thread_local uint3 threadIdx;
extern thread_local uint3 blockIdx;
typedef struct CUstream_st* cudaStream_t;
int __syncthreads_or(int predicate);
"""
# Reads one case from stdin, runs every kernel over its grid, prints one output tensor per line.
HOST_MAIN = r"""
#include "cuda_stub.h"
#include "elastic_route.cuh"
#include <atomic>
#include <barrier>
#include <cstdio>
#include <thread>
#include <vector>
using namespace sglang::rlh_elastic_route;
thread_local uint3 threadIdx;
thread_local uint3 blockIdx;
static std::barrier<>* block_barrier = nullptr;
static std::atomic<int> block_any{0};
int __syncthreads_or(int predicate) {
  if (predicate) block_any.store(1);
  block_barrier->arrive_and_wait();
  return block_any.load();
}
template <class T> std::vector<T> read(long long n) {
  std::vector<T> v(n);
  for (auto& x : v) { long long y; if (std::scanf("%lld", &y) != 1) throw 1; x = static_cast<T>(y); }
  return v;
}
template <class T> void show(const std::vector<T>& v) {
  for (auto x : v) std::printf("%lld ", static_cast<long long>(x));
  std::printf("\n");
}
template <class F> void grid(unsigned gx, unsigned gy, unsigned threads, F&& body) {
  for (unsigned y = 0; y < gy; ++y)
    for (unsigned x = 0; x < gx; ++x)
      for (unsigned t = 0; t < threads; ++t) { blockIdx = {x, y, 0}; threadIdx = {t, 0, 0}; body(); }
}
template <class F> void grid_sync(unsigned gx, unsigned gy, unsigned threads, F&& body) {
  for (unsigned y = 0; y < gy; ++y)
    for (unsigned x = 0; x < gx; ++x) {
      std::barrier<> barrier(threads);
      block_barrier = &barrier;
      block_any = 0;
      std::vector<std::thread> pool;
      for (unsigned t = 0; t < threads; ++t)
        pool.emplace_back([&, t] { blockIdx = {x, y, 0}; threadIdx = {t, 0, 0}; body(); });
      for (auto& thread : pool) thread.join();
    }
}
"""
HOST_MAIN += r"""
int main() {
  long long ratio, n, k, stride, kb, block, slots, ctx, m, l;
  std::scanf("%lld %lld %lld %lld %lld %lld %lld %lld %lld %lld", &ratio, &n, &k, &stride, &kb, &block, &slots, &ctx,
             &m, &l);
  auto top_k = read<int32_t>(n * stride);
  auto blocks = read<int32_t>(n * kb);
  auto rid = read<int64_t>(n), seq = read<int64_t>(n);
  auto r2t = read<int32_t>(slots * ctx);
  auto stale = read<int32_t>(3 * slots), route = read<int32_t>(3 * 4);
  auto swapped = read<int32_t>(n * k);
  auto mapping = read<int64_t>(m), loc = read<int64_t>(l);
  auto num_real = read<int32_t>(1);
  Route r{};
  r.req_pool_indices = Ints{rid.data(), 1, 0, true};
  r.seq_lens = Ints{seq.data(), 1, 0, true};
  r.req_to_token = Ints{r2t.data(), ctx, 1, false};
  r.ratio = ratio;
  r.max_pos = ctx / ratio - 1;
  r.top_k = Ints{top_k.data(), stride, 1, false};
  r.num_top_k = k;
  const unsigned gy = (k + kThreads - 1) / kThreads, gl = (l + kThreads - 1) / kThreads;
  std::vector<int32_t> table(n * k);
  grid(n, gy, kThreads, [&] { resident_kernel(r, Ints{table.data(), k, 1, false}, 1000); });
  show(table);
  for (int i = 0; i < 3; ++i) {
    r.stale_limit = stale.data() + i * slots;
    r.route = route.data() + i * 4;
    r.block = 0, r.num_top_k = k, r.top_k = Ints{top_k.data(), stride, 1, false};
    std::vector<int32_t> masked(n * k), merged(n * k), gate(1);
    std::vector<int64_t> rows(n * k), stores(l);
    std::vector<int8_t> reads(n * k);
    grid(n, gy, kThreads, [&] {
      select_tokens_kernel(r, Ints{masked.data(), k, 1, false}, rows.data(), reads.data());
    });
    grid(n, gy, kThreads, [&] {
      merge_kernel(Ints{swapped.data(), k, 1, false}, rows.data(), reads.data(), Ints{merged.data(), k, 1, false}, k);
    });
    grid(gl, 1, kThreads, [&] {
      store_rows_kernel(Ints{loc.data(), 1, 0, true}, Ints{mapping.data(), 1, 0, true}, r.route,
                        Ints{stores.data(), 1, 0, true}, l);
    });
    grid(1, 1, 1, [&] { gate_kernel(num_real.data(), r.route, gate.data()); });
    show(masked), show(rows), show(reads), show(merged), show(stores), show(gate);
    r.block = block, r.num_top_k = kb, r.top_k = Ints{blocks.data(), kb, 1, false};
    std::vector<int32_t> block_masked(n * kb);
    std::vector<int64_t> block_rows(n * kb * block);
    std::vector<int8_t> block_reads(n * kb * block);
    grid_sync(n, kb, block, [&] {
      select_blocks_kernel(r, Ints{block_masked.data(), kb, 1, false}, block_rows.data(), block_reads.data());
    });
    show(block_masked), show(block_rows), show(block_reads);
  }
}
"""


@unittest.skipUnless(clang_cuda(), "clang++ with a CUDA front end is required")
class KernelSyntax(unittest.TestCase):
    def test_route_kernels_compile_for_host_and_sm80_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel, text in HEADERS.items():
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text(text)
            unit = root / "unit.cu"
            unit.write_text(f'#include "cuda_stub.h"\n#include "{KERNEL}"\n')
            for side in ("--cuda-host-only", "--cuda-device-only"):
                result = subprocess.run(
                    [clang_cuda(), "-x", "cuda", "--cuda-gpu-arch=sm_80", "-nocudainc", "-nocudalib", side,
                     "-std=c++20", "-fsyntax-only", "-Wall", "-Wextra", "-Wno-unknown-cuda-version", "-I", str(root),
                     str(unit)], capture_output=True, text=True)
                with self.subTest(side=side):
                    self.assertEqual(result.returncode, 0, result.stderr[-4000:])
                    self.assertNotIn("warning", result.stderr, result.stderr[-4000:])


@unittest.skipUnless(shutil.which("clang++") and torch is not None, "clang++ and torch are required")
class KernelsOnHost(unittest.TestCase):
    def test_kernels_match_the_torch_statements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel, text in dict(HEADERS, **{"cuda_stub.h": HOST_STUB, "elastic_route.cuh": KERNEL.read_text(),
                                             "main.cpp": HOST_MAIN}).items():
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text(text)
            binary = root / "main"
            build = subprocess.run([shutil.which("clang++"), "-x", "c++", "-std=c++20", "-O1", "-pthread", "-I",
                                    str(root), str(root / "main.cpp"), "-o", str(binary)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr[-4000:])
            for ratio in (1, 4):
                with self.subTest(ratio=ratio):
                    self.check(binary, ratio)

    def check(self, binary, ratio):
        from .elastic_kv import ElasticKV as K
        t = torch
        g = t.Generator().manual_seed(ratio)
        slots, ctx, n, k, stride, kb, block = 6, 256, 5, 9, 12, 3, 16
        r2t = t.randperm(slots * ctx, generator=g).view(slots, ctx).to(t.int32)
        stale = t.randint(1, ctx // ratio, (slots,), generator=g).to(t.int32)
        # Route rows: offloaded with half the rows backed, resident, offloaded with every row backed.
        probe = types.SimpleNamespace(torch=t, ratio=ratio, fused=None, c=types.SimpleNamespace(
            req_to_token_pool=types.SimpleNamespace(req_to_token=r2t)), stale_limit=t.stack((stale,) * 3),
            route=t.tensor([[0, 1000, slots * ctx // ratio // 2, 0], [1, 1000, 0, 0], [0, 1000, slots * ctx, 0]],
                           dtype=t.int32),
            mapping=t.randint(1, 1 << 20, (slots * ctx + 1,), generator=g))
        top_full = t.randint(-1, ctx // ratio, (n, stride), generator=g).to(t.int32)   # top_k: a strided view
        top_k = top_full[:, :k]
        blocks = t.randint(-1, ctx // ratio // block, (n, kb), generator=g).to(t.int32)
        rid, seq = t.randint(1, slots, (n,), generator=g), t.randint(1, ctx // ratio, (n,), generator=g)
        limit = stale[rid].long()
        seq = t.maximum(seq, limit + 1)
        seq[0] = ctx // ratio
        top_k[:, 0], top_k[:, 1], top_k[:, 2], top_k[:, 3] = limit, limit - 1, seq - 1, seq      # edges
        top_k[1, 4] = 0
        r2t[rid[1], ratio - 1] = probe.route[0, 2] * ratio      # record 0 of request 1 sits on the backed bound
        swapped = t.randint(0, 1 << 20, (n, k), generator=g).to(t.int32)
        loc, num_real = t.randint(-1, slots * ctx, (3 * n,), generator=g), t.tensor([n], dtype=t.int32)
        loc[:2] = t.tensor([-1, 0])
        head = [ratio, n, k, stride, kb, block, slots, ctx, len(probe.mapping), len(loc)]
        tensors = [top_full, blocks, rid, seq, r2t, probe.stale_limit, probe.route, swapped, probe.mapping, loc,
                   num_real]
        stdin = "\n".join(" ".join(map(str, x)) for x in [head] + [x.reshape(-1).tolist() for x in tensors]) + "\n"
        run = subprocess.run([str(binary)], input=stdin, capture_output=True, text=True, check=True)
        out = [[int(v) for v in line.split()] for line in run.stdout.splitlines()]
        flat = lambda x: x.reshape(-1).tolist()      # noqa: E731
        self.assertEqual(out[0], flat(K.resident_page_table(probe, rid, top_k, seq, offset=1000)))
        for i in (0, 1, 2):
            masked, rows, reads, merged, stores, gate, b_masked, b_rows, b_reads = out[1 + 9 * i:10 + 9 * i]
            for (want_masked, want_rows, hit), got in (
                    (K.reference_select(probe, i, rid, seq, top_k), (masked, rows, reads)),
                    (K.reference_select(probe, i, rid, seq, blocks, block=block), (b_masked, b_rows, b_reads))):
                hit = flat(hit)
                self.assertEqual(got[0], flat(want_masked), i)
                self.assertEqual([bool(v) for v in got[2]], hit, i)
                self.assertEqual([v for v, h in zip(got[1], hit) if h], [v for v, h in zip(flat(want_rows), hit) if h])
            want = K.reference_select(probe, i, rid, seq, top_k)
            self.assertEqual(merged, flat(K.reference_merge(probe, swapped, want[1], want[2])))
            self.assertEqual(stores, flat(K.store_rows(probe, i, loc)))
            self.assertEqual(gate, flat(K.gate(probe, i, num_real)))
            if i != 1:                                  # offloaded rows: stale hits and misses
                self.assertTrue(0 < sum(reads) < len(reads))


if __name__ == "__main__":
    unittest.main()
