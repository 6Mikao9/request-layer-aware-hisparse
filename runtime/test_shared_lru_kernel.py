"""Syntax/type check of shared_lru_kernel.cuh with clang's CUDA front end (sm_80).

No CUDA toolkit is needed: the SGLang/tvm-ffi/CUDA APIs the kernel touches are
declared by minimal stubs with the pinned signatures, and every template
variant is instantiated for host and device. This catches C++ and CUDA front-end
errors on a CPU machine; it is not an nvcc build and proves nothing about
runtime behavior (``runtime.cuda_shared_lru_correctness`` does that on an A100).
"""
import itertools
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

KERNEL = Path(__file__).with_name("shared_lru_kernel.cuh")
CUDA_STUB = r"""
#pragma once
#include <cstddef>
#include <cstdint>
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __forceinline__ inline __attribute__((always_inline))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#define __align__(n) __attribute__((aligned(n)))
struct dim3 { unsigned x, y, z; __host__ __device__ dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
struct uint3 { unsigned x, y, z; };
struct __attribute__((aligned(16))) int4 { int x, y, z, w; };
extern const __device__ uint3 threadIdx;
extern const __device__ uint3 blockIdx;
typedef struct CUstream_st* cudaStream_t;
__device__ void __syncthreads();
__device__ int __syncthreads_or(int predicate);
__device__ int __shfl_up_sync(unsigned mask, int value, unsigned delta, int width = 32);
__device__ unsigned __match_any_sync(unsigned mask, int value);
__device__ unsigned __ballot_sync(unsigned mask, int predicate);
__device__ unsigned __reduce_or_sync(unsigned mask, unsigned value);
__device__ int __popc(unsigned value);
__device__ int __ffs(int value);
__device__ int atomicCAS(int* address, int compare, int value);
__device__ unsigned long long atomicExch(unsigned long long* address, unsigned long long value);
__device__ int atomicMin(int* address, int value);
__device__ int atomicMax(int* address, int value);
__device__ int atomicAdd(int* address, int value);
__device__ unsigned long long atomicAdd(unsigned long long* address, unsigned long long value);
__device__ int atomicSub(int* address, int value);
__device__ int atomicOr(int* address, int value);
__device__ unsigned atomicOr(unsigned* address, unsigned value);
__device__ unsigned atomicAnd(unsigned* address, unsigned value);
__device__ int min(int a, int b);
__device__ int max(int a, int b);
"""
TENSOR_STUB = r"""
#pragma once
#include "cuda_stub.h"
#include <cstdint>
enum { kDLCPU = 1, kDLCUDA = 2, kDLInt = 0 };
struct DLDevice { int device_type; int device_id; };
struct DLDataType { uint8_t code; uint8_t bits; uint16_t lanes; };
namespace tvm { namespace ffi {
struct Shape { const int64_t* data; int64_t operator[](int i) const { return data[i]; } };
struct TensorView {
  void* data_ptr() const; int ndim() const; Shape shape() const; Shape strides() const;
  DLDataType dtype() const; DLDevice device() const;
};
}}  // namespace tvm::ffi
"""
SGL_STUB = r"""
#pragma once
#include "tensor_stub.h"
namespace sglang {
namespace host {
struct LaunchKernel {
  LaunchKernel(dim3 grid, dim3 block, cudaStream_t stream, std::size_t smem = 0);
  static cudaStream_t resolve_device(DLDevice device);
  template <typename T, typename... Args> void operator()(T&& kernel, Args&&... args) const;
};
namespace runtime {
namespace details { struct MaybeDevice { MaybeDevice(DLDevice d) : device_id(d.device_id) {} unsigned device_id; }; }
void* get_device_accessible_ptr(const tvm::ffi::TensorView& tensor);
std::size_t get_max_smem_per_block(int device_id, bool opt_in = true, bool use_cache = true);
template <auto kernel> void set_max_dynamic_smem_per_device(details::MaybeDevice device, std::size_t bytes);
}  // namespace runtime
}  // namespace host
}  // namespace sglang
"""
HISPARSE_STUB = r"""
#pragma once
#include "sgl_stub.h"
namespace sglang {
constexpr int WARP_SIZE = 32;
using BallotMask = unsigned int;
constexpr BallotMask FULL_WARP_MASK = 0xFFFFFFFFu;
__device__ __forceinline__ int hash_slot(int32_t key, int hash_size) { return ((uint32_t)key * 2654435761u) % (uint32_t)hash_size; }
__device__ void transfer_item_warp(int32_t lane_id, const void* src_addr, void* dst_addr, int64_t item_size_bytes);
}
"""
HEADERS = {"cuda_stub.h": CUDA_STUB, "tensor_stub.h": TENSOR_STUB, "sgl_stub.h": SGL_STUB,
           "sgl_kernel/tensor.h": '#include "../sgl_stub.h"\n', "sgl_kernel/utils.h": '#include "../sgl_stub.h"\n',
           "sgl_kernel/runtime.cuh": '#include "../sgl_stub.h"\n', "sgl_kernel/utils.cuh": '#include "../sgl_stub.h"\n',
           "dlpack/dlpack.h": '#include "../tensor_stub.h"\n', "tvm/ffi/container/tensor.h": '#include "../../../tensor_stub.h"\n',
           "kvcacheio/hisparse.cuh": HISPARSE_STUB}
FLAGS = (False, True)
# (owner_summary, touched_bitmap, hit_index, incremental_lru, compact_owner): the plain kernel, each switch
# alone, the PR #8 pair, the issue #6 set and the cross pairs of the two alternatives per phase.
SWITCHES = [(False, False, False, False, False), (True, False, False, False, False), (False, True, False, False, False),
            (True, True, False, False, False), (False, False, True, False, False), (False, False, False, True, False),
            (False, False, False, False, True), (False, False, True, True, True), (False, True, True, False, True),
            (True, False, False, True, False)]
# (fifo, soft_floor, bypass, counters) x switch set: every policy with every switch set.
VARIANTS = [(1024, 16, 128, *policy, *switches) for policy in itertools.product(FLAGS, repeat=4) for switches in SWITCHES]
VARIANTS += [(1024, 2, 4, False, True, True, True, True, True, False, False, False),
             (256, 4, 2, True, False, False, False, True, True, False, False, False),
             (256, 4, 2, False, True, False, False, False, True, False, False, False),
             (256, 4, 2, True, True, False, True, False, False, True, True, True),
             (64, 8, 4, False, True, True, False, False, False, True, False, True)]


def clang_cuda():
    clang = shutil.which("clang++")
    if clang is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.cu"
        probe.write_text('extern "C" __attribute__((global)) void k(int* p) { p[0] = 1; }\n')
        result = subprocess.run([clang, "-x", "cuda", "--cuda-gpu-arch=sm_80", "-nocudainc", "-nocudalib",
                                 "--cuda-device-only", "-fsyntax-only", str(probe)], capture_output=True)
    return clang if result.returncode == 0 else None


@unittest.skipUnless(clang_cuda(), "clang++ with a CUDA front end is required")
class KernelSyntax(unittest.TestCase):
    def test_every_variant_compiles_for_host_and_sm80_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel, text in HEADERS.items():
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text(text)
            uses = "\n".join(f"  (void)&sglang::rlh_shared_lru::resolve<{', '.join(str(x).lower() for x in v)}>;"
                             for v in VARIANTS)
            unit = root / "unit.cu"
            unit.write_text(f'#include "cuda_stub.h"\n#include "{KERNEL}"\nvoid instantiate() {{\n{uses}\n}}\n')
            for side in ("--cuda-host-only", "--cuda-device-only"):
                result = subprocess.run(
                    [clang_cuda(), "-x", "cuda", "--cuda-gpu-arch=sm_80", "-nocudainc", "-nocudalib", side,
                     "-std=c++20", "-fsyntax-only", "-Wall", "-Wno-unknown-cuda-version", "-I", str(root),
                     str(unit)], capture_output=True, text=True)
                with self.subTest(side=side):
                    self.assertEqual(result.returncode, 0, result.stderr[-4000:])
                    self.assertNotIn("warning", result.stderr, result.stderr[-4000:])


# Plain host C++: the CUDA keywords vanish, so the header's __host__ __device__ helpers run on the CPU.
HOST_STUB = r"""
#pragma once
#include <cstddef>
#include <cstdint>
#define __global__
#define __device__
#define __host__
#define __shared__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __align__(n)
struct dim3 { unsigned x, y, z; dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
struct uint3 { unsigned x, y, z; };
struct alignas(16) int4 { int x, y, z, w; };
extern const uint3 threadIdx;
extern const uint3 blockIdx;
typedef struct CUstream_st* cudaStream_t;
""" + "\n".join(line.replace("__device__ ", "") for line in CUDA_STUB.splitlines() if line.startswith("__device__"))


@unittest.skipUnless(shutil.which("clang++"), "clang++ is required")
class HostHelpers(unittest.TestCase):
    def test_index_key_matches_the_python_statement(self):
        """hit_index buckets and fingerprints from the CUDA header itself, compiled for the host."""
        from runtime import shared_lru as s
        import random
        rng = random.Random(6)
        cases = [(plane, token, half, bits) for plane, token in ((0, 0), (63, 2**31 - 1), (1, 1), (62, 4095))
                 for half, bits in ((1, 1), (15_360, 18), (2**19, 23))]
        cases += [(rng.randrange(64), rng.randrange(2**31), rng.randrange(1, 2**20), rng.randrange(1, 24))
                  for _ in range(400)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel, text in dict(HEADERS, **{"cuda_stub.h": HOST_STUB}).items():
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text(text)
            main = root / "main.cpp"
            main.write_text(f'#include "cuda_stub.h"\n#include "{KERNEL}"\n#include <cstdio>\n'
                            "int main() {\n  long long plane, token, half, bits;\n"
                            '  while (std::scanf("%lld %lld %lld %lld", &plane, &token, &half, &bits) == 4) {\n'
                            "    sglang::rlh_shared_lru::Params p{};\n    p.index_half = (int)half;\n"
                            "    p.row_bits = (int)bits;\n"
                            "    const auto key = sglang::rlh_shared_lru::index_key(p, (int)plane, (int)token);\n"
                            '    std::printf("%llu %d %d %d\\n", (unsigned long long)sglang::rlh_shared_lru::'
                            "index_mix((int)plane, (int)token), key.left, key.right, key.fp);\n  }\n}\n")
            binary = root / "main"
            build = subprocess.run([shutil.which("clang++"), "-x", "c++", "-std=c++20", "-O1", "-I", str(root),
                                    str(main), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr[-4000:])
            run = subprocess.run([str(binary)], input="".join(f"{p} {t} {h} {b}\n" for p, t, h, b in cases),
                                 capture_output=True, text=True, check=True)
        for (plane, token, half, bits), line in zip(cases, run.stdout.splitlines(), strict=True):
            mix, left, right, fp = map(int, line.split())
            want_left, want_right, want_fp = s.index_slots(plane, token, half, bits)
            self.assertEqual((mix, left, right, fp), (s.index_hash(plane, token), want_left, want_right,
                                                      want_fp << bits), (plane, token, half, bits))
            self.assertTrue(0 <= left < half <= right < 2 * half and 0 <= fp + (1 << bits) - 1 < 2**31)


if __name__ == "__main__":
    unittest.main()
