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
__device__ int atomicMin(int* address, int value);
__device__ int atomicAdd(int* address, int value);
__device__ unsigned long long atomicAdd(unsigned long long* address, unsigned long long value);
__device__ int atomicSub(int* address, int value);
__device__ int atomicOr(int* address, int value);
__device__ unsigned atomicOr(unsigned* address, unsigned value);
__device__ unsigned atomicAnd(unsigned* address, unsigned value);
__device__ int min(int a, int b);
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
# (fifo, soft_floor, bypass, counters, owner_summary, touched_bitmap): every policy with every switch pair.
VARIANTS = [(1024, 16, 128, *flags) for flags in itertools.product(FLAGS, repeat=6)]
VARIANTS += [(1024, 2, 4, False, True, True, True, True, True), (256, 4, 2, True, False, False, False, True, True),
             (256, 4, 2, False, True, False, False, False, True)]


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


if __name__ == "__main__":
    unittest.main()
