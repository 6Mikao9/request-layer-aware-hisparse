"""Allocator-derived main-KV reclaim planning, before collective reservation.

Snapshots are serializable control-plane values, not allocation permissions.
The engine gathers them in its ordered TP control phase, validates freshness
before the first migration, executes demotions through LayerEvictionTP, then
calls LayerForwardTP.begin again. Only that final reservation permits a forward.
This module performs no transport, model execution, or native logical allocation.
"""

from dataclasses import dataclass
import hashlib
import json
from math import isfinite

from .layer_eviction_policy import Candidate, LayerKey
from .policy import validate_capacity


def _digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class DemotionProposal:
    key: LayerKey
    hot_capacity: int
    switch_cost: float
    exposed_transfer_cost: float
    resident_since: float
    decode_eligible: bool


@dataclass(frozen=True)
class LocalReclaim:
    proposal: DemotionProposal
    reclaim_bytes: tuple
    host_reserve_bytes: int
    demand_relief_bytes: tuple


@dataclass(frozen=True)
class ExcludedTarget:
    key: LayerKey
    hot_capacity: int
    reason: str


def exclusion_targets(excluded, proposals):
    """Resolve typed exclusions and unambiguous legacy (key, reason) records.

    Legacy values remain untouched so historical capture hashes stay valid.
    A legacy key cannot identify an exclusion when its round declares many H.
    """
    declared, capacities = set(), {}
    for proposal in proposals:
        if type(proposal) is not DemotionProposal or type(proposal.key) is not LayerKey:
            raise ValueError("Expected a typed proposal directory")
        identity = proposal.key, proposal.hot_capacity
        if identity in declared:
            raise ValueError("Duplicate proposal target")
        declared.add(identity)
        capacities.setdefault(proposal.key, []).append(proposal.hot_capacity)
    result = {}
    for item in excluded:
        if type(item) is ExcludedTarget:
            key, hot_capacity, reason = item.key, item.hot_capacity, item.reason
        elif type(item) is tuple and len(item) == 2:
            key, reason = item
            if type(key) is not LayerKey or len(capacities.get(key, ())) != 1:
                raise ValueError("Legacy exclusion must identify exactly one declared H")
            hot_capacity = capacities[key][0]
        else:
            raise ValueError("Expected a typed target exclusion or legacy key/reason pair")
        identity = key, hot_capacity
        if (type(key) is not LayerKey or type(hot_capacity) is not int or hot_capacity < 1
                or type(reason) is not str or not reason or identity not in declared
                or identity in result):
            raise ValueError("Invalid or duplicate target exclusion")
        result[identity] = reason
    return result


@dataclass(frozen=True)
class RankAdmissionSnapshot:
    rank: int
    world_size: int
    configuration: str
    forward_identity: str
    logical_identity: str
    proposal_identity: str
    local_state: str
    groups: tuple
    needed_bytes: tuple
    free_bytes: tuple
    host_free_bytes: int
    candidates: tuple
    excluded: tuple


@dataclass(frozen=True)
class AdmissionDecision:
    # main_pages_ready is still not a reservation, let alone global admission.
    status: str
    domains: tuple
    needed_bytes: tuple
    free_bytes: tuple
    plan: object = None
    actions: tuple = ()  # (LayerKey, H); no rank-local pointer crosses the wire.


def capture_admission(bridge, rank, leases, logical_chunks, *, mode, proposals):
    """Capture the exact next-forward page needs and legal resident demotions.

    decode_eligible must come from the native scheduler: the adapter cannot know
    whether a request has unfinished prefill chunks. Times in all proposals use
    the policy coordinator's common monotonic clock, not different rank clocks.
    Native logical slots/table are already provisionally assigned by the caller.
    """
    if type(rank) is not int or rank < 0:
        raise ValueError("Expected a nonnegative TP rank")
    if bridge.tp is None:
        raise ValueError("Use an explicit TP control domain, including single-rank runs")
    transport = bridge.tp._transport
    if (type(transport.world_size) is not int or transport.world_size < 1 or
            type(transport.rank) is not int or not 0 <= rank < transport.world_size or
            rank != transport.rank):
        raise ValueError("Snapshot rank must match the actual TP control-group member")
    demand = bridge.inspect_forward(leases, logical_chunks, mode=mode)
    adapter = bridge.adapter
    if adapter._pending:
        raise RuntimeError("A prepared demotion must finish before admission planning")
    logical, physical = [], []
    for row, lease in sorted(bridge._leases.items()):
        rid = bridge._request_ids[row]
        if type(rid) is not str or not rid:
            raise ValueError("Admission snapshots require stable request IDs")
        history, pages = [], []
        for layer in range(adapter.layout.layer_count):
            state = adapter._state(lease, layer)
            history.append((len(state.logical), state.logical_digest.hexdigest(),
                            None if state.hot is None else state.hot.capacity))
            group = adapter._kv._pages.route(layer)
            owner = adapter._kv._pages._groups[group.name]
            ids = adapter._kv._pages.page_ids(lease, group.name)
            pages.append((ids, tuple((owner.refcounts[p], p in owner.sealed) for p in ids)))
        logical.append((rid, row, lease.generation, history))
        physical.append((row, lease.generation, pages))
    forward = (mode, tuple((bridge._request_ids[l.row], l.row, l.generation, chunk)
        for l, chunk in zip(demand.leases, demand.logical_chunks)))
    config = _digest((adapter.layout.digest, adapter._max_context, adapter._kv._logical_slots,
                      None if bridge.tp is None else bridge.tp._config))
    needed = tuple(n*demand.page_bytes for n in demand.needed_pages)
    free = tuple(n*demand.page_bytes for n in demand.free_pages)
    host_free = adapter._host_budget - adapter._host_reserved
    if host_free < 0:
        raise RuntimeError("Invalid host budget accounting")
    stamp = _digest((logical, physical, free, adapter._host_budget, adapter._host_reserved,
                     bridge._sequence))
    local, excluded, intents, seen, clocks = [], [], [], set(), {}
    for proposal in proposals:
        if type(proposal) is not DemotionProposal or type(proposal.key) is not LayerKey:
            raise TypeError("Expected typed demotion proposals")
        key = proposal.key
        identity = key, proposal.hot_capacity
        if identity in seen:
            raise ValueError("Duplicate target capacity for one request-layer")
        seen.add(identity)
        if any(type(v) is not int or v < lower for v, lower in
               ((key.row, 1), (key.generation, 1), (key.layer, 0))):
            raise ValueError("Invalid proposal request generation or layer")
        if type(proposal.decode_eligible) is not bool or any(
            not isfinite(v) or v < 0 for v in
            (proposal.switch_cost, proposal.exposed_transfer_cost)
        ) or not isfinite(proposal.resident_since):
            raise ValueError("Invalid proposal eligibility, cost or time")
        if clocks.setdefault(key, proposal.resident_since) != proposal.resident_since:
            raise ValueError("H alternatives must share the resident history clock")
        validate_capacity(proposal.hot_capacity, proposal.hot_capacity,
                          adapter.layout.topk_blocks, adapter.layout.block_tokens)
        if proposal.hot_capacity % adapter.layout.page_tokens:
            raise ValueError("Hot capacity must be page aligned")
        adapter._kv._pages.route(key.layer)
        lease = bridge._leases.get(key.row)
        if lease is None or lease.generation != key.generation:
            raise ValueError("Stale proposal request generation")
        intents.append((key.row, key.generation, key.layer, proposal.hot_capacity,
                        proposal.resident_since, proposal.decode_eligible))
        if not proposal.decode_eligible:
            excluded.append(ExcludedTarget(key, proposal.hot_capacity, "scheduler_not_decode_eligible"))
            continue
        if mode == "prefill" and any(l is lease for l in demand.leases):
            excluded.append(ExcludedTarget(key, proposal.hot_capacity, "participates_in_pending_prefill"))
            continue
        try:
            reclaim = adapter.reclaimable_bytes(lease, key.layer, proposal.hot_capacity)
        except ValueError as exc:
            excluded.append(ExcludedTarget(key, proposal.hot_capacity, str(exc)))
            continue
        group = adapter._kv._pages.route(key.layer).name
        vector = tuple(reclaim if name == group else 0 for name in demand.groups)
        # The implementation reserves full max_context host K/V, not just the
        # current history or released rows. GPU metadata OOM is still possible.
        charge = adapter._max_context * adapter.layout.kv_bytes_per_token_per_layer
        # A decode participant that becomes hot no longer needs its next
        # resident growth page. Keep this separate from actual reclaimed pages.
        reduction = sum((target-old)*demand.page_bytes
            for l, name, old, target in demand.reservations
            if mode == "decode" and l is lease and name == group)
        relief = tuple(reduction if name == group else 0 for name in demand.groups)
        local.append(LocalReclaim(proposal, vector, charge, relief))
    local.sort(key=lambda c: (c.proposal.key.row, c.proposal.key.generation,
                             c.proposal.key.layer, c.proposal.hot_capacity))
    excluded.sort(key=lambda c: (c.key.row, c.key.generation, c.key.layer, c.hot_capacity))
    return RankAdmissionSnapshot(rank, transport.world_size, config, _digest(forward), _digest(logical),
        _digest(sorted(intents)), stamp, demand.groups, needed, free, host_free,
        tuple(local), tuple(excluded))


def require_current_snapshot(bridge, snapshot, leases, logical_chunks, *, mode, proposals):
    """Revalidate immediately before migration, under serialized scheduler ownership.

    Equality also checks native request-table preflight, histories, generations,
    mode, physical ownership and host reservations, not just free-byte counts.
    It creates no lock/reservation: the scheduler must exclude concurrent work.
    After our own migration, deliberately recapture instead of checking this old
    snapshot. Native logical rollback remains the caller's responsibility.
    """
    current = capture_admission(bridge, snapshot.rank, leases, logical_chunks,
                               mode=mode, proposals=proposals)
    if current != snapshot:
        raise ValueError("Admission snapshot became stale; replan before demoting")


def plan_admission(policy, snapshots, *, now):
    """Plan on gathered current snapshots, with (rank, group) resource domains.

    Timings use max over ranks as a conservative local surrogate, not a measured
    global critical path. Shared-link interference and future batch costs need
    measured estimates from the caller; the search does not promise optimality.
    """
    snapshots = tuple(sorted(snapshots, key=lambda s: s.rank))
    if not snapshots or tuple(s.rank for s in snapshots) != tuple(range(snapshots[0].world_size)):
        raise ValueError("Need exactly one snapshot for every TP rank starting at zero")
    first = snapshots[0]
    fields = ("world_size", "configuration", "forward_identity", "logical_identity", "proposal_identity", "groups")
    if any(getattr(s, f) != getattr(first, f) for s in snapshots for f in fields):
        raise ValueError("TP request, history, configuration or proposal disagreement")
    domains = tuple((s.rank, group) for s in snapshots for group in s.groups)
    needed = tuple(n for s in snapshots for n in s.needed_bytes)
    free = tuple(n for s in snapshots for n in s.free_bytes)
    by_rank = [{(c.proposal.key, c.proposal.hot_capacity): c for c in s.candidates} for s in snapshots]
    if any(len(rank) != len(s.candidates) for rank, s in zip(by_rank, snapshots)):
        raise ValueError("Duplicate rank-local candidate target")
    keys = set(by_rank[0]).intersection(*(set(rank) for rank in by_rank[1:]))
    candidates = []
    for identity in sorted(keys, key=lambda target: (target[0].row, target[0].generation,
                                                    target[0].layer, target[1])):
        key, hot_capacity = identity
        local = [rank[identity] for rank in by_rank]
        if any(c.proposal.resident_since != local[0].proposal.resident_since for c in local):
            raise ValueError("TP candidate residency clocks disagree")
        candidates.append(Candidate(key,
            tuple(b for c in local for b in c.reclaim_bytes),
            max(c.proposal.switch_cost for c in local),
            max(c.proposal.exposed_transfer_cost for c in local),
            local[0].proposal.resident_since,
            tuple(c.host_reserve_bytes for c in local),
            tuple(b for c in local for b in c.demand_relief_bytes), hot_capacity=hot_capacity))
    plan = policy.plan_for_admission(free, candidates, required_free_bytes=needed,
        host_free_bytes=tuple(s.host_free_bytes for s in snapshots), now=now)
    if all(n <= f for n, f in zip(needed, free)):
        status = "main_pages_ready"
    else:
        status = "planned" if plan is not None else "no_complete_plan"
    actions = () if plan is None else plan.actions
    return AdmissionDecision(status, domains, needed, free, plan, actions)
