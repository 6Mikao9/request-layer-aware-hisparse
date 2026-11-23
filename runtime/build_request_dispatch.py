"""Derive an opt-in capacity-filtered fused kernel from the pinned active overlay.

The stock/static experiment artifacts are immutable. This is a distinct JIT
variant, and both arms of a request-type comparison must use this variant.
"""
import hashlib
import json
from pathlib import Path

from .build_overlay import replace_one


ROOT = Path(__file__).parent


def products(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / 'PROVENANCE.json').read_text())
    src = {}
    for name in ('hisparse_active.cuh', 'ops.py'):
        raw = (root / 'generated' / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest['generated_sha256'][name]:
            raise RuntimeError('Unverified base overlay: ' + name)
        src[name] = raw.decode('utf-8')
    header = src['hisparse_active.cuh']
    header = replace_one(header, '    int64_t physical_hot_buffer_size,',
                         '    int64_t physical_hot_buffer_size,\n'
                         '    const int32_t* __restrict__ assigned_capacities,')
    header = replace_one(header, '  const int64_t rid = req_pool_indices[bid];',
                         '  const int64_t rid = req_pool_indices[bid];\n'
                         '  // Every real row is owned by exactly one capacity launch.\n'
                         '  // Padding has already returned; mismatches never touch output.\n'
                         '  if (assigned_capacities[rid] != HOT_BUFFER_SIZE) {\n'
                         '    return;\n  }')
    header = replace_one(header, '    tvm::ffi::TensorView miss_count_out) {',
                         '    tvm::ffi::TensorView miss_count_out,\n'
                         '    tvm::ffi::TensorView assigned_capacities) {')
    header = replace_one(header, '        lru_slots.shape()[1],',
                         '        lru_slots.shape()[1],\n'
                         '        static_cast<const int32_t*>(assigned_capacities.data_ptr()),')
    ops = src['ops.py']
    ops = replace_one(ops, '"request_layer_hisparse_v1",', '"request_type_layer_hisparse_v1",')
    ops = replace_one(ops, '    active_hot_buffer_size: int | None = None,',
                     '    active_hot_buffer_size: int | None = None,\n'
                     '    assigned_capacities: torch.Tensor | None = None,')
    ops = replace_one(ops, '    num_top_k_blocks = top_k_blocks.size(1)',
                     '    if (assigned_capacities is None or assigned_capacities.dtype != torch.int32\n'
                     '            or tuple(assigned_capacities.shape) != (device_buffer_tokens.shape[0],)\n'
                     '            or not assigned_capacities.is_contiguous()\n'
                     '            or assigned_capacities.device != top_k_blocks.device):\n'
                     '        raise ValueError("Expected contiguous device int32 capacity per request slot")\n'
                     '    if skip_io:\n'
                     '        raise ValueError("Request routing requires the fused copy path")\n'
                     '    num_top_k_blocks = top_k_blocks.size(1)')
    ops = replace_one(ops, '        empty,\n        empty,\n        empty,\n    )',
                     '        empty,\n        empty,\n        empty,\n'
                     '        assigned_capacities,\n    )')
    return {'hisparse_active.cuh': header, 'ops.py': ops, '__init__.py': ''}


def generate(output=None):
    output = ROOT / 'generated_grouped' if output is None else Path(output)
    data = {name: text.encode('utf-8') for name, text in products().items()}
    # Check all files before writing any, avoiding a partially updated variant.
    for name, raw in data.items():
        path = output / name
        if path.exists() and path.read_bytes() != raw:
            raise RuntimeError('Refusing to replace divergent generated source: ' + str(path))
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in data.items():
        if not (output / name).exists():
            (output / name).write_bytes(raw)
    return {name: hashlib.sha256(raw).hexdigest() for name, raw in data.items()}


if __name__ == '__main__':
    print(json.dumps(generate(), indent=2))
