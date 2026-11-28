"""Derive the route-aware swap kernel of the dynamic offload slab (issue #11) from the pinned hisparse.cuh.

One launch per switchable layer reads that layer's row (mode, plane base, tail view, 0) of a device route
table. Resident mode writes base + logical row for every selected token (a C4 record's row is its closing
token's logical row / RESIDENT_RATIO) and returns; offloaded mode is the pinned kernel over absolute slab
rows. Either way the newest token resolves to the layer's tail row. The table is the only state a switch
changes, so a captured CUDA Graph keeps its pointers and launch arguments.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .build_overlay import replace_one

SOURCE = "kernels/jit/csrc/kvcacheio/hisparse.cuh"
OUTPUT = Path(__file__).with_name("generated_route") / "hisparse_route.cuh"
RESIDENT_PATH = """  int16_t* req_lru_slots = lru_slots + rid * lru_slot_stride_0;

  // Dynamic offload route (issue #11): (mode, plane base, tail view, 0). The newest token always resolves to
  // this layer's tail row; a resident layer reads its complete main KV at base + logical row.
  const int32_t* layer_route = route + route_index * 4;
  const int32_t newest_row = layer_route[2] + static_cast<int32_t>(rid);
  if (layer_route[0] == 1) {
    const int32_t resident_base = layer_route[1];
    const int32_t* req_rows = req_to_token + rid * req_to_token_stride;
    const int count = (seq_len < NUM_TOP_K_TOKENS) ? static_cast<int>(seq_len) : NUM_TOP_K_TOKENS;
    for (int i = tid; i < NUM_TOP_K_TOKENS; i += BLOCK_SIZE) {
      const int32_t token_pos = resolve_selected_token<SPARSE_BLOCK_SIZE, TopKIsBlocks>(req_top_k, i);
      int32_t device_loc = -1;
      if (token_pos >= 0 && token_pos < seq_len && (TopKIsBlocks || i < count)) {
        device_loc = token_pos == seq_len - 1
                         ? newest_row
                         : resident_base + req_rows[(token_pos + 1) * RESIDENT_RATIO - 1] / RESIDENT_RATIO;
      }
      req_top_k_device_locs[i] = device_loc;
    }
    if constexpr (RecordMissPlan) {
      if (tid == 0) {
        miss_count_out[bid] = 0;
      }
    }
    return;
  }
"""
CHECKS = """  constexpr int NUM_TOP_K_TOKENS = NUM_TOP_K * (TopKIsBlocks ? SPARSE_BLOCK_SIZE : 1);
  const int64_t bs = top_k.shape()[0];
  if (route.dtype().code != kDLInt || route.dtype().bits != 32 || route.ndim() != 2 || route.shape()[1] != 4 ||
      route.strides()[1] != 1 || route.strides()[0] != 4 || route_index < 0 || route_index >= route.shape()[0] ||
      req_to_token.dtype().code != kDLInt || req_to_token.dtype().bits != 32 || req_to_token.ndim() != 2 ||
      req_to_token.strides()[1] != 1) {
    throw std::runtime_error("load_cache_to_device_buffer: expected a contiguous [layers, 4] int32 route table, "
                             "an in-range route index and a row-major int32 req_to_token");
  }
"""


def replace_all(source, old, new, count):
    if source.count(old) != count:
        raise ValueError(f"Expected {count} occurrences: {old!r}")
    return source.replace(old, new)


def derive(header):
    header = replace_one(header, "    bool SkipIO,\n    typename SeqLensT,",
                         "    bool SkipIO,\n    int RESIDENT_RATIO,\n    typename SeqLensT,")
    header = replace_one(header, "    int64_t plan_stride) {",
                         "    int64_t plan_stride,\n    const int32_t* __restrict__ route,\n    int64_t route_index,\n"
                         "    const int32_t* __restrict__ req_to_token,\n    int64_t req_to_token_stride) {")
    header = replace_all(header, "device_loc = req_device_buffer_locs[token_pos];",
                         "device_loc = token_pos == seq_len - 1 ? newest_row : req_device_buffer_locs[token_pos];", 2)
    header = replace_one(header, "      req_top_k_device_locs[i] = req_device_buffer_locs[newest_slot];",
                         "      req_top_k_device_locs[i] = newest_row;")
    header = replace_one(header, "  const int newest_slot = HOT_BUFFER_SIZE;\n", "")
    header = replace_one(header, "  int16_t* req_lru_slots = lru_slots + rid * lru_slot_stride_0;\n", RESIDENT_PATH)
    header = replace_one(header, "    bool SkipIO>\nvoid load_cache_to_device_buffer(",
                         "    bool SkipIO,\n    int RESIDENT_RATIO>\nvoid load_cache_to_device_buffer(")
    header = replace_one(header, "    tvm::ffi::TensorView miss_count_out) {",
                         "    tvm::ffi::TensorView miss_count_out,\n    tvm::ffi::TensorView route,\n"
                         "    int64_t route_index,\n    tvm::ffi::TensorView req_to_token) {")
    first = "  constexpr int NUM_TOP_K_TOKENS = NUM_TOP_K * (TopKIsBlocks ? SPARSE_BLOCK_SIZE : 1);\n"
    header = replace_one(header, first + "  const int64_t bs = top_k.shape()[0];\n", CHECKS)
    header = replace_one(header, "        plan_stride);",
                         "        plan_stride,\n        static_cast<const int32_t*>(route.data_ptr()),\n"
                         "        route_index,\n        static_cast<const int32_t*>(req_to_token.data_ptr()),\n"
                         "        req_to_token.strides()[0]);")
    header = replace_all(header, "            SkipIO,\n", "            SkipIO,\n            RESIDENT_RATIO,\n", 4)
    return "// Derived from pinned SGLang for issue #11; see ../PROVENANCE.json and ../NOTICE.\n" + header


def generate(source_root, output=OUTPUT):
    """Write the derived header once; an existing file must match byte for byte."""
    manifest = json.loads(Path(__file__).with_name("PROVENANCE.json").read_text())
    raw = (Path(source_root) / SOURCE).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["source_sha256"][SOURCE]:
        raise ValueError(f"Source SHA256 mismatch: {SOURCE}")
    data = derive(raw.decode("utf-8")).encode("utf-8")
    output = Path(output)
    if output.exists() and output.read_bytes() != data:
        raise ValueError(f"Refusing to overwrite divergent artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, required=True, help="pinned python/sglang package directory")
    print(json.dumps({OUTPUT.name: generate(parser.parse_args().source_root)}, indent=2))
