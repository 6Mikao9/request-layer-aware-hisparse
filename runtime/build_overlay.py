"""Build an isolated HiSparse MHA overlay; never modifies the supplied source tree."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import textwrap

IMAGE = 'sha256:614b1509bcb729c653528d448bff9850f8977fcad9d12ba56bd6dbe3d7985e87'
FILES = ('kernels/jit/csrc/kvcacheio/hisparse.cuh', 'kernels/ops/kvcache/hisparse.py', 'srt/managers/hisparse_coordinator.py')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def replace_one(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected exactly one occurrence: {old!r}')
    return source.replace(old, new)

def extract(source, name):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            return textwrap.dedent('\n'.join(source.splitlines()[start-1:node.end_lineno])) + '\n'
    raise ValueError(name)

def generate(source_root, output, expected):
    source_root, output = Path(source_root), Path(output)
    for rel, sha in expected.items():
        if digest(source_root / rel) != sha:
            raise ValueError(f'Source SHA256 mismatch: {rel}')
    header = (source_root / FILES[0]).read_text(encoding='utf-8')
    header = replace_one(header, '    int64_t lru_slot_stride_0,', '    int64_t lru_slot_stride_0,\n    int64_t physical_hot_buffer_size,')
    header = replace_one(header, '  if (seq_len <= HOT_BUFFER_SIZE) {', '  if (seq_len <= physical_hot_buffer_size) {')
    header = replace_one(header, '  const int newest_slot = HOT_BUFFER_SIZE;', '  const int newest_slot = static_cast<int>(physical_hot_buffer_size);')
    header = replace_one(header, '        lru_slot_stride_0,', '        lru_slot_stride_0,\n        lru_slots.shape()[1],')
    header = replace_one(header, '  constexpr int NUM_TOP_K_TOKENS = NUM_TOP_K * (TopKIsBlocks ? SPARSE_BLOCK_SIZE : 1);\n  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;', '  constexpr int NUM_TOP_K_TOKENS = NUM_TOP_K * (TopKIsBlocks ? SPARSE_BLOCK_SIZE : 1);\n  static_assert(HOT_BUFFER_SIZE >= NUM_TOP_K_TOKENS, "active H must cover selected token width");\n  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;')
    # HOT_BUFFER_SIZE remains the active template capacity everywhere else.
    header = '// Derived from pinned SGLang; see ../PROVENANCE.json and ../NOTICE.\n' + header
    ops = (source_root / FILES[1]).read_text(encoding='utf-8')
    jit = extract(ops, '_jit_sparse_module')
    jit = replace_one(jit, '"sparse_cache",', '"request_layer_hisparse_v1",')
    jit = replace_one(jit, 'cuda_files=["kvcacheio/hisparse.cuh"],', 'cuda_files=[str(Path(__file__).with_name("hisparse_active.cuh"))],')
    mha = extract(ops, 'load_blocks_to_device_buffer_mha')
    mha = replace_one(mha, '    skip_io: bool = False,', '    skip_io: bool = False,\n    active_hot_buffer_size: int | None = None,')
    mha = replace_one(mha, '    num_top_k_blocks = top_k_blocks.size(1)', '''    from runtime.policy import validate_capacity, validate_metadata_shapes, validate_metadata_strides
    active_hot_buffer_size = hot_buffer_size if active_hot_buffer_size is None else active_hot_buffer_size
    validate_capacity(active_hot_buffer_size, hot_buffer_size, top_k_blocks.size(1), sparse_block_size)
    validate_metadata_shapes(tuple(lru_slots.shape), tuple(device_buffer_tokens.shape), tuple(device_buffer_locs.shape), hot_buffer_size)
    validate_metadata_strides(tuple(lru_slots.stride()), tuple(device_buffer_tokens.stride()), tuple(device_buffer_locs.stride()))
    num_top_k_blocks = top_k_blocks.size(1)''')
    mha = replace_one(mha, '        hot_buffer_size,', '        active_hot_buffer_size,')
    prefix = '# Modified from pinned SGLang (Apache-2.0); see ../NOTICE and ../LICENSE.\nfrom __future__ import annotations\nimport functools\nfrom pathlib import Path\nimport torch\nfrom sglang.kernels.jit.utils import make_cpp_args, load_jit\n'
    coordinator = (source_root / FILES[2]).read_text(encoding='utf-8')
    method = extract(coordinator, 'swap_in_selected_blocks')
    method = replace_one(method, '    assert self.is_m3_hisparse', '    assert self.is_m3_hisparse\n    policy = self._request_layer_hisparse_policy\n    if self.device_buffer_size != policy.hmax:\n        raise ValueError("Physical Hmax changed after binding")')
    method = replace_one(method, '        hot_buffer_size=self.device_buffer_size,', '        hot_buffer_size=self.device_buffer_size,\n        active_hot_buffer_size=policy.capacities[layer_id],')
    products = {'hisparse_active.cuh': header, 'ops.py': prefix+jit+'\n'+mha, 'coordinator_method.py': '# Modified from pinned SGLang (Apache-2.0); see ../NOTICE and ../LICENSE.\nfrom __future__ import annotations\nimport torch\nfrom .ops import load_blocks_to_device_buffer_mha\n'+method, '__init__.py': ''}
    output.mkdir(parents=True, exist_ok=True)
    for name, data in products.items():
        path = output / name
        raw = data.encode('utf-8')
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f'Refusing to overwrite divergent artifact: {path}')
        if not path.exists():
            path.write_bytes(raw)
    return {name: digest(output/name) for name in products}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    provenance = json.loads((root/'PROVENANCE.json').read_text())
    result = generate(args.source_root, root/'generated', provenance['source_sha256'])
    print(json.dumps(result, indent=2))
