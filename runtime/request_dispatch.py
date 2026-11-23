"""Experimental device-side request routing; no automatic service installation.

Requires an independently validated CPU admission owner for every real input row
and an immutable per-request capacity until fenced cleanup. This module does not
provide that scheduler integration. See request_type_dispatch.md for the gate.
"""
import hashlib
from pathlib import Path

from .policy import validate_capacity


def validate_groups(capacities, hmax, topk_blocks, block_size):
    if type(capacities) is not tuple or not capacities:
        raise ValueError('Supply a frozen nonempty tuple of layer capacity groups')
    for cap in capacities:
        validate_capacity(cap, hmax, topk_blocks, block_size)
    if tuple(sorted(set(capacities))) != capacities:
        raise ValueError('Each capacity must be launched exactly once in sorted order')


def verify_grouped_sources():
    from .build_request_dispatch import products
    root = Path(__file__).with_name('generated_grouped')
    for name, text in products().items():
        if hashlib.sha256((root / name).read_bytes()).digest() != hashlib.sha256(text.encode()).digest():
            raise RuntimeError('Unverified request dispatch source: ' + name)


def load_grouped_blocks_to_device_buffer_mha(*, assigned_capacities, capacities, **kwargs):
    """Each launch sees original batch order; mismatching CTAs do no work.

    No GPU -> CPU read, per-row sort, or split copy path. Output initialization is
    once per layer, never between groups. Hmax allocation and latest slot stay
    unchanged. Call verify_grouped_sources once before any GPU compilation/use.
    """
    if 'active_hot_buffer_size' in kwargs:
        raise ValueError('Capacity is chosen by request group, not one batch scalar')
    if kwargs.get('skip_io', False):
        raise ValueError('Request routing requires the fused copy path')
    validate_groups(capacities, kwargs['hot_buffer_size'],
                    kwargs['top_k_blocks'].size(1), kwargs['sparse_block_size'])
    import torch
    # Validate before touching output. Per-row values are admission-owned, not
    # copied to CPU here. A fixture must verify every real row has exactly one H.
    if (assigned_capacities.dtype != torch.int32 or not assigned_capacities.is_contiguous()
            or tuple(assigned_capacities.shape) != (kwargs['device_buffer_tokens'].shape[0],)
            or assigned_capacities.device != kwargs['top_k_blocks'].device):
        raise ValueError('Invalid device capacity table')
    from .generated_grouped.ops import load_blocks_to_device_buffer_mha
    kwargs['top_k_device_locs'].fill_(-1)
    for cap in capacities:
        load_blocks_to_device_buffer_mha(assigned_capacities=assigned_capacities,
                                        active_hot_buffer_size=cap, **kwargs)
