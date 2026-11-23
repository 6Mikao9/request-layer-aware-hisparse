"""Immutable static policy. No torch dependency and no allocation controller."""
from dataclasses import dataclass
import hashlib
import json

def validate_capacity(active, hmax, topk_blocks, block_size):
    if any(type(x) is not int for x in (active, hmax, topk_blocks, block_size)):
        raise ValueError('Capacities and widths must be integers')
    if not (0 < topk_blocks and 0 < block_size and topk_blocks * block_size <= active <= hmax <= 32767):
        raise ValueError('Require 0 < selected tokens <= active H <= Hmax <= 32767 (int16 LRU)')

@dataclass(frozen=True)
class StaticPolicy:
    hmax: int
    capacities: tuple[int, ...]
    expected_tp_size: int = 4
    model: str = 'MiniMax-M3.1-Flash'
    hardware: str = 'A100'

    def __post_init__(self):
        if type(self.capacities) is not tuple or len(self.capacities) != 60:
            raise ValueError('Supply an immutable table of exactly 60 absolute layer capacities')
        if type(self.expected_tp_size) is not int or self.expected_tp_size not in (4, 8):
            raise ValueError('Only explicit TP4 or TP8 is supported')
        if self.model != 'MiniMax-M3.1-Flash' or self.hardware != 'A100':
            raise ValueError('Prototype is restricted to MiniMax-M3.1-Flash / A100')
        for active in self.capacities:
            validate_capacity(active, self.hmax, 1, 1)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps({'hmax': self.hmax, 'capacities': self.capacities,
            'model': self.model, 'hardware': self.hardware, 'expected_tp_size': self.expected_tp_size}, sort_keys=True).encode()).hexdigest()


def validate_metadata_shapes(lru_shape, token_shape, location_shape, hmax):
    # Kernel page_size=1 is token addressing, NOT the physical pool page padding.
    if (len(lru_shape) != 2 or len(token_shape) != 2 or len(location_shape) != 2
            or lru_shape[1] != hmax or token_shape != location_shape
            or lru_shape[0] != token_shape[0] or token_shape[1] < hmax + 1):
        raise ValueError('Require Hmax LRU columns and matching metadata rows with at least Hmax+1 columns')


def validate_metadata_strides(lru_stride, token_stride, location_stride):
    # The CUDA kernel intentionally shares token_stride[0] for tokens and locs.
    if (any(len(stride) != 2 or stride[1] != 1 for stride in (lru_stride, token_stride, location_stride))
            or token_stride[0] != location_stride[0]):
        raise ValueError('Metadata columns must be contiguous and token/location row strides must match')
