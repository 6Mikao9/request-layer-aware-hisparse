"""CPU planning for request-layer eviction under independent budget pressures."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class LayerKey:
    row: int
    generation: int
    layer: int


@dataclass(frozen=True)
class Candidate:
    key: LayerKey
    reclaim_bytes: tuple[int, ...]
    switch_cost: float
    exposed_transfer_cost: float
    resident_since: float
    host_reserve_bytes: tuple[int, ...] = ()
    demand_relief_bytes: tuple[int, ...] = ()
    hot_capacity: int | None = None  # None is the legacy single-target planner API.


@dataclass(frozen=True)
class EvictionPlan:
    keys: tuple[LayerKey, ...]
    projected_free_bytes: tuple[int, ...]
    target_reached: bool
    purpose: str = "watermark"
    projected_host_free_bytes: tuple[int, ...] = ()
    required_free_bytes_after: tuple[int, ...] = ()
    actions: tuple = ()  # Explicit (LayerKey, H) choices; never two H for one key.


class LayerEvictionPolicy:
    """Watermarks in a fixed resource-domain order; use a monotonic clock.

    A domain is a rank for a genuinely fungible rank-local allocator. For
    separate layer pools it must be (rank, layer): free pages in layer A cannot
    satisfy an allocation in layer B. Never sum noninterchangeable budgets.
    """

    def __init__(
        self, *, trigger_free_bytes: tuple[int, ...],
        target_free_bytes: tuple[int, ...], max_migrations: int,
        cooldown: float = 0.0, min_residency: float = 0.0,
        transfer_weight: float = 1.0,
    ):
        if not trigger_free_bytes or len(trigger_free_bytes) != len(target_free_bytes):
            raise ValueError("watermarks must cover the same nonempty rank set")
        self._trigger = tuple(trigger_free_bytes)
        self._validate_bytes(trigger_free_bytes)
        self._validate_bytes(target_free_bytes)
        if any(not 0 <= trigger < target for trigger, target in
               zip(trigger_free_bytes, target_free_bytes)):
            raise ValueError("require 0 <= trigger < target on every rank")
        if type(max_migrations) is not int or max_migrations < 1 or any(
            not isfinite(value) or value < 0
            for value in (cooldown, min_residency, transfer_weight)
        ):
            raise ValueError("invalid eviction policy settings")
        self._target = tuple(target_free_bytes)
        self._max_migrations = max_migrations
        self._cooldown = cooldown
        self._min_residency = min_residency
        self._transfer_weight = transfer_weight
        self._pressure = False
        self._pending = None
        self._next_plan_at = float("-inf")
        self._last_now = float("-inf")

    def _validate_bytes(self, values):
        if len(values) != len(self._trigger) or any(
            type(value) is not int or value < 0 for value in values
        ):
            raise ValueError("expected nonnegative integer bytes for each rank")

    def _validate_now(self, now):
        if not isfinite(now) or now < self._last_now:
            raise ValueError("now must be finite and monotonic")

    def _observe(self, free_bytes):
        if all(free >= target for free, target in zip(free_bytes, self._target)):
            self._pressure = False
        elif any(free <= trigger for free, trigger in zip(free_bytes, self._trigger)):
            self._pressure = True

    def _eligible_candidates(self, candidates, now):
        remaining = {}
        seen, by_key = set(), {}
        for candidate in candidates:
            key = candidate.key
            if not isinstance(key, LayerKey) or any(
                type(value) is not int or value < minimum
                for value, minimum in ((key.row, 1), (key.generation, 1), (key.layer, 0))
            ):
                raise ValueError("invalid row, generation, or layer")
            capacity = candidate.hot_capacity
            if capacity is not None and (type(capacity) is not int or capacity < 1):
                raise ValueError("Expected an explicit positive target capacity or legacy None")
            action = (key, capacity)
            if action in seen:
                raise ValueError("duplicate eviction candidate")
            seen.add(action)
            previous = by_key.get(key)
            if previous is not None and (capacity is None or previous.hot_capacity is None
                                         or previous.resident_since != candidate.resident_since):
                raise ValueError("Alternative capacities require explicit H and one residency clock")
            by_key[key] = candidate
            self._validate_bytes(candidate.reclaim_bytes)
            if not isfinite(candidate.resident_since) or candidate.resident_since > now or any(
                not isfinite(cost) or cost < 0 for cost in
                (candidate.switch_cost, candidate.exposed_transfer_cost)
            ):
                raise ValueError("invalid candidate timing or cost")
            if now - candidate.resident_since >= self._min_residency:
                remaining[action] = candidate
        return remaining

    @staticmethod
    def _order(action):
        key, capacity = action
        return key.row, key.generation, key.layer, -1 if capacity is None else capacity

    @staticmethod
    def _remove_alternatives(remaining, key):
        for action in tuple(remaining):
            if action[0] == key:
                del remaining[action]

    @staticmethod
    def _actions(selected):
        return tuple((c.key, c.hot_capacity) for c in selected if c.hot_capacity is not None)

    def plan(self, free_bytes, candidates, *, now: float) -> EvictionPlan | None:
        """Choose a bounded batch. A new plan requires commit or cancel first."""
        if self._pending is not None:
            raise RuntimeError("an eviction plan is already pending")
        self._validate_bytes(free_bytes)
        self._validate_now(now)
        remaining = self._eligible_candidates(candidates, now)
        self._last_now = now
        self._observe(free_bytes)
        if not self._pressure or now < self._next_plan_at:
            return None
        projected = list(free_bytes)
        selected = []
        while remaining and len(selected) < self._max_migrations:
            deficits = [max(0, target - free) for target, free in
                        zip(self._target, projected)]
            scores = {}
            for key, candidate in remaining.items():
                # Cap each rank independently. Surplus on one rank cannot
                # meet a different rank's deficit.
                useful = sum(min(deficit, reclaim) for deficit, reclaim in
                             zip(deficits, candidate.reclaim_bytes))
                if useful:
                    cost = candidate.switch_cost + (
                        self._transfer_weight * candidate.exposed_transfer_cost
                    )
                    scores[key] = cost / useful
            if not scores:
                break
            action = min(scores, key=lambda a: (scores[a], self._order(a)))
            candidate = remaining[action]
            self._remove_alternatives(remaining, candidate.key)
            selected.append(candidate)
            projected = [free + reclaim for free, reclaim in
                         zip(projected, candidate.reclaim_bytes)]
        if not selected:
            return None
        reached = all(free >= target for free, target in zip(projected, self._target))
        self._pending = EvictionPlan(tuple(c.key for c in selected), tuple(projected), reached,
                                     actions=self._actions(selected))
        return self._pending

    def plan_for_admission(self, free_bytes, candidates, *, required_free_bytes,
                           host_free_bytes, now: float) -> EvictionPlan | None:
        """Plan only a complete main-page deficit, independently of watermarks.

        Host domains are a separate per-rank vector. Candidates must reserve
        their full preparation charge there. None means no action is proposed,
        not proof of infeasibility: the bounded greedy search may miss a solution.
        A projected fit is never permission to submit model work.
        """
        if self._pending is not None:
            raise RuntimeError("an eviction plan is already pending")
        self._validate_bytes(free_bytes)
        self._validate_bytes(required_free_bytes)
        self._validate_now(now)
        candidates = tuple(candidates)
        host = tuple(host_free_bytes)
        if not host or any(type(v) is not int or v < 0 for v in host):
            raise ValueError("Expected per-rank nonnegative host byte budgets")
        for c in candidates:
            if len(c.host_reserve_bytes) != len(host) or any(
                type(v) is not int or v < 0 for v in c.host_reserve_bytes
            ):
                raise ValueError("Each admission candidate needs every host-domain charge")
            if c.demand_relief_bytes:
                self._validate_bytes(c.demand_relief_bytes)
        remaining = self._eligible_candidates(candidates, now)
        zero = (0,)*len(free_bytes)
        relief = {(c.key, c.hot_capacity): c.demand_relief_bytes or zero
                  for c in candidates}
        # Options for one request-layer are mutually exclusive. The physical
        # next-forward requirement can include that layer's relief only once.
        relief_by_key = {}
        for action, vector in relief.items():
            previous = relief_by_key.setdefault(action[0], {})
            for d, value in enumerate(vector):
                if value:
                    previous[d] = max(previous.get(d, 0), value)
        total_relief = [0]*len(free_bytes)
        for vector in relief_by_key.values():
            for d, value in vector.items():
                total_relief[d] += value
        if any(total > need for total, need in zip(total_relief, required_free_bytes)):
            raise ValueError("Candidate demand relief cannot exceed the actual forward requirement")
        self._last_now = now
        projected = list(free_bytes)
        required = list(required_free_bytes)
        if now < self._next_plan_at or all(f >= n for f, n in zip(projected, required_free_bytes)):
            return None
        # Per-domain maxima over alternatives give an optimistic bound, not a
        # realizable sum of all H. Domain maxima may come from different H; this
        # is only a necessary condition, ignoring host and migration limits.
        upper_by_key, contributions = {}, {}
        for action, candidate in remaining.items():
            sparse = {d: r+q for d,(r,q) in enumerate(zip(candidate.reclaim_bytes,relief[action])) if r or q}
            contributions[action] = sparse
            previous = upper_by_key.setdefault(candidate.key, {})
            for d, value in sparse.items():
                previous[d] = max(previous.get(d, 0), value)
        upper = [0]*len(free_bytes)
        for vector in upper_by_key.values():
            for d,value in vector.items():
                upper[d] += value
        if any(f + extra < need
               for f, extra, need in zip(projected, upper, required_free_bytes)):
            return None
        selected = []
        host = list(host)
        while remaining and len(selected) < self._max_migrations:
            deficits = [max(0, need-free) for need, free in zip(required, projected)]
            if not any(deficits):
                break
            scores = {}
            for action, c in remaining.items():
                if any(charge > free for charge, free in zip(c.host_reserve_bytes, host)):
                    continue
                # Avoid selecting a cheap large H when excluding its smaller-H
                # alternatives would make a complete fit demonstrably impossible.
                # This is still bounded greedy, not backtracking or an oracle.
                contribution = contributions[action]
                # Other domains cannot change when this key's alternatives
                # disappear. Their optimistic fit held initially and after each
                # prior selection, so only this key's nonzero domains need a
                # repeated check. Dense input validation remains above.
                if any(projected[d] + contribution.get(d, 0) + upper[d] - group_max < required[d]
                       for d,group_max in upper_by_key[c.key].items()):
                    continue
                useful = sum(min(deficits[d], value) for d,value in contribution.items())
                if useful:
                    scores[action] = (c.switch_cost + self._transfer_weight*c.exposed_transfer_cost) / useful
            if not scores:
                break
            action = min(scores, key=lambda a: (scores[a], self._order(a)))
            c = remaining[action]
            self._remove_alternatives(remaining, c.key)
            selected.append(c)
            projected = [f+r for f, r in zip(projected, c.reclaim_bytes)]
            required = [n-r for n, r in zip(required, relief[action])]
            host = [f-r for f, r in zip(host, c.host_reserve_bytes)]
            for d,used in upper_by_key[c.key].items():
                upper[d] -= used
        if not all(f >= n for f, n in zip(projected, required)):
            return None  # Do not execute partial demotions solely for this admission.
        self._pending = EvictionPlan(tuple(c.key for c in selected), tuple(projected), True,
                                     "main_pool_admission", tuple(host), tuple(required),
                                     self._actions(selected))
        return self._pending

    def _require_pending(self, plan):
        if plan is None or plan is not self._pending:
            raise ValueError("expected the current pending plan")

    def commit(self, plan: EvictionPlan, *, succeeded, free_bytes, now: float):
        """Confirm successful keys and observe measured post-migration free bytes."""
        self._require_pending(plan)
        self._validate_bytes(free_bytes)
        self._validate_now(now)
        succeeded = set(succeeded)
        if not succeeded.issubset(plan.keys):
            raise ValueError("successful keys must belong to the pending plan")
        self._observe(free_bytes)
        self._last_now = now
        self._next_plan_at = now + self._cooldown
        self._pending = None

    def cancel(self, plan: EvictionPlan):
        """Discard a pending plan with no completed migrations."""
        self._require_pending(plan)
        self._pending = None
