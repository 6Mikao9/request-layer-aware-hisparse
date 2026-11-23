"""Frozen request-type x layer policies and admission ownership, without CUDA.

This accounts for LOGICAL active slots. Native Hmax backing remains allocated.
The caller must publish an admitted row only after native allocation succeeds,
and release it only after native request cleanup has completed its GPU fences.
There is intentionally no live resize or implicit reuse of a request slot.
"""
from dataclasses import dataclass
import hashlib
import json

from .policy import StaticPolicy, validate_capacity


@dataclass(frozen=True)
class RequestTypePolicy:
    hmax: int
    tables: tuple[tuple[str, tuple[int, ...]], ...]
    default_type: str
    expected_tp_size: int = 4
    selected_tokens: int = 2048

    def __post_init__(self):
        if type(self.tables) is not tuple or not self.tables:
            raise ValueError('Supply immutable nonempty type tables')
        names = []
        for entry in self.tables:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError('Each entry must be (type label, layer table)')
            label, caps = entry
            if not isinstance(label, str) or not label or label.strip() != label:
                raise ValueError('Type labels must be nonempty, canonical strings')
            names.append(label)
            StaticPolicy(self.hmax, caps, expected_tp_size=self.expected_tp_size)
            for cap in caps:
                validate_capacity(cap, self.hmax, self.selected_tokens, 1)
                if cap % 128 or self.hmax % 128:
                    raise ValueError('Pinned MiniMax capacities must align to 128 tokens')
        if len(set(names)) != len(names) or self.default_type not in names:
            raise ValueError('Type labels must be unique and include the default')

    def resolve(self, request_type):
        tables = dict(self.tables)
        label = request_type if request_type in tables else self.default_type
        return label, tables[label]

    def groups(self, layer):
        if type(layer) is not int or not 0 <= layer < 60:
            raise ValueError('Expected absolute layer index 0..59')
        return tuple(sorted({caps[layer] for _, caps in self.tables}))

    @property
    def digest(self):
        payload = dict(hmax=self.hmax, tables=sorted(self.tables),
                       default_type=self.default_type, tp=self.expected_tp_size,
                       selected_tokens=self.selected_tokens)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class RequestLease:
    slot: int
    generation: int
    rid: str
    owner: int
    request_type: str
    capacities: tuple[int, ...]

    @property
    def logical_slots(self):
        return sum(self.capacities)


class RequestPolicyRegistry:
    """Transactional CPU ownership; not a GPU allocator or a service hook.

    reserve -> native allocation/device publication -> commit -> decode ->
    native fenced cleanup -> release. On pre-commit allocation failure cancel.
    Reservations count against the same budget as committed requests. Admission
    and cleanup must be serialized by the owning scheduler; no thread safety is
    claimed. A stale callback cannot free a new owner of the same numeric slot.
    """
    def __init__(self, policy, slot_count, logical_slot_budget):
        if not isinstance(policy, RequestTypePolicy):
            raise TypeError('policy must be RequestTypePolicy')
        if any(type(n) is not int or n <= 0 for n in (slot_count, logical_slot_budget)):
            raise ValueError('Positive integer slot count and logical budget required')
        self.policy = policy
        self.slot_count = slot_count
        self.logical_slot_budget = logical_slot_budget
        self._generation = [0] * slot_count
        self._entries = {}
        self.charged_slots = 0

    def reserve(self, slot, rid, owner, request_type=None):
        if type(slot) is not int or not 0 <= slot < self.slot_count:
            raise ValueError('Invalid request slot')
        if not isinstance(rid, str) or not rid or type(owner) is not int:
            raise ValueError('Explicit request ID and owner object identity required')
        if slot in self._entries or any(e[0].rid == rid for e in self._entries.values()):
            raise RuntimeError('Request slot or RID already has a live reservation')
        label, caps = self.policy.resolve(request_type)
        needed = sum(caps)
        if self.charged_slots + needed > self.logical_slot_budget:
            raise MemoryError('Request-type tables exceed the common logical slot budget')
        self._generation[slot] += 1
        lease = RequestLease(slot, self._generation[slot], rid, owner, label, caps)
        self._entries[slot] = (lease, 'reserved')
        self.charged_slots += needed
        return lease

    def _require(self, lease, state):
        if not isinstance(lease, RequestLease) or self._entries.get(lease.slot) != (lease, state):
            raise RuntimeError('Stale owner, generation or invalid reservation state')

    def commit(self, lease):
        self._require(lease, 'reserved')
        self._entries[lease.slot] = (lease, 'active')

    def _remove(self, lease, state):
        self._require(lease, state)
        del self._entries[lease.slot]
        self.charged_slots -= lease.logical_slots

    def cancel(self, lease):
        self._remove(lease, 'reserved')

    def release(self, lease):
        self._remove(lease, 'active')

    def validate_batch(self, leases):
        leases = tuple(leases)
        if len({lease.slot for lease in leases}) != len(leases):
            raise RuntimeError('A decode batch cannot consume one request slot twice')
        for lease in leases:
            self._require(lease, 'active')
        return tuple(lease.slot for lease in leases)

    def capacity_rows(self):
        """CPU staging snapshot [layer, slot]; zero means not decode-ready.

        A real runtime should copy rows at admission/cleanup, not take this full
        snapshot or synchronize device metadata in every layer.
        """
        return tuple(tuple(self._entries[slot][0].capacities[layer]
                           if slot in self._entries and self._entries[slot][1] == 'active'
                           else 0 for slot in range(self.slot_count)) for layer in range(60))
