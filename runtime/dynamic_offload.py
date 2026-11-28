"""Runtime per-layer dynamic offload (issue #11): slab layout, routes, switch plans and the TP switch protocol.

Standard library only. The switchable layers of one row type (MiniMax-M3 sparse layers,
DeepSeek-V4 C4 layers) are the union U of the pre-registered offload lists; layers outside U
never switch and keep their stock pools. One rank keeps, for U, K hot planes (the native
per-layer hot buffer of each offloaded layer), |U|-K resident planes (the complete main KV of
each resident layer at its logical rows) and one tail row per (layer, request slot) for the
token each decode step writes. Every plane sits in one slab per K and V buffer and every row
index is absolute, so the swap kernel, the attention consumer and each captured CUDA Graph
keep one base pointer while routes change. A switch rewrites the route descriptor, the
location tables of the layers that become offloaded and the rows of the planes that change
hands; it never moves a pointer. A plane belongs to one layer at a time: there is no
cross-layer slot borrowing and no shared LRU.

Switches move whole layers between two pre-registered offload lists of the same size K, so the
slab never grows or shrinks. Host history is the source of truth of every rebuilt plane: phase 1
makes it complete for the layers that become offloaded (a device-to-host copy unless
``host_shadow`` kept it current), phase 2 rebuilds the reassigned planes from host only, so a
failure on any rank rebuilds the old routes the same way. The newest token stays in its tail
row throughout and moves to its home row (resident row, short-request hot row, host) in the
next step's flush.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from threading import Lock
import time

OFFLOADED, RESIDENT = 0, 1
MODES = {OFFLOADED: "offloaded", RESIDENT: "resident"}


class SwitchError(RuntimeError):
    """The switch was refused or rolled back; the old routes still hold on every rank."""


class SwitchProtocolError(RuntimeError):
    """A rank could not restore the old routes; the TP worker group must stop."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Geometry:
    """Per-rank sizes of one row type. Rows of a plane are pages of ``page`` rows."""
    layers: tuple            # switchable absolute layer IDs (U), ascending
    hot_rows: int            # one hot plane: device rows of the native pool, padding page included
    resident_rows: int       # one resident plane: logical rows plus the padding page
    hot_slots: int           # device_buffer_size: hot rows per request and offloaded layer
    slots: int               # request slots (req_to_token rows)
    page: int
    page_bytes: int          # bytes of one page in one buffer (K or V), padding included
    buffers: int = 2         # 2 for MHA (K and V), 1 for K-only MLA records

    def __post_init__(self):
        if (type(self.layers) is not tuple or not self.layers or list(self.layers) != sorted(set(self.layers))
                or any(type(x) is not int or x < 0 for x in self.layers)):
            raise ValueError("layers must be ascending distinct non-negative absolute layer IDs")
        for name in ("hot_rows", "resident_rows", "hot_slots", "slots", "page", "page_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hot_rows % self.page or self.resident_rows % self.page:
            raise ValueError("plane rows must be whole pages so absolute rows map to absolute pages")
        if self.buffers not in (1, 2):
            raise ValueError("buffers is 2 (K and V) or 1 (K-only records)")

    def row_bytes(self, rows):
        return -(-rows // self.page) * self.page_bytes * self.buffers

    def index(self, layer):
        if layer not in self.layers:
            raise ValueError(f"layer {layer} is not a switchable layer of this row type")
        return self.layers.index(layer)


@dataclass(frozen=True)
class SlabLayout:
    """Absolute rows of one slab: hot planes, then resident planes, then one tail view per layer.

    A tail view holds one row per request slot, padded to whole pages; slot 0 is the request
    pool's padding row, so padded graph rows (slot 0, location 0) write and read its row harmlessly.
    """
    geometry: Geometry
    k: int

    def __post_init__(self):
        if type(self.k) is not int or not 0 < self.k <= len(self.geometry.layers):
            raise ValueError("K counts 1..|U| offloaded layers")

    @property
    def resident_planes(self):
        return len(self.geometry.layers) - self.k

    def hot_base(self, plane):
        return plane * self.geometry.hot_rows

    def resident_base(self, plane):
        return self.k * self.geometry.hot_rows + plane * self.geometry.resident_rows

    @property
    def tail_base(self):
        return self.resident_base(self.resident_planes)

    @property
    def tail_stride(self):
        g = self.geometry
        return -(-g.slots // g.page) * g.page

    def tail_view(self, layer):
        """First row of ``layer``'s tail view; the decode store writes view row = request slot."""
        return self.tail_base + self.geometry.index(layer) * self.tail_stride

    def tail_row(self, layer, slot):
        if not 0 <= slot < self.geometry.slots:
            raise ValueError(f"request slot {slot} is outside 0..{self.geometry.slots - 1}")
        return self.tail_view(layer) + slot

    @property
    def rows(self):
        return self.tail_base + len(self.geometry.layers) * self.tail_stride

    @property
    def digest(self):
        return _digest(dict(asdict(self.geometry), k=self.k))


@dataclass(frozen=True)
class Routes:
    """layer -> (mode, plane) of every switchable layer; ``epoch`` counts committed switches."""
    entries: tuple           # ((layer, mode, plane), ...) by layer
    epoch: int = 0

    @property
    def offload(self):
        return tuple(layer for layer, mode, _ in self.entries if mode == OFFLOADED)

    def route(self, layer):
        for entry in self.entries:
            if entry[0] == layer:
                return entry[1:]
        raise ValueError(f"layer {layer} has no route")

    def base(self, layout, layer):
        mode, plane = self.route(layer)
        return layout.hot_base(plane) if mode == OFFLOADED else layout.resident_base(plane)

    def descriptor(self, layout):
        """GPU route table rows (mode, plane base, tail view, 0) in switchable-layer order."""
        return [(mode, self.base(layout, layer), layout.tail_view(layer), 0) for layer, mode, _ in self.entries]

    @property
    def digest(self):
        return _digest([list(e) for e in self.entries])


def _check_list(layout, offload, error=ValueError):
    offload = tuple(sorted(offload))
    if len(offload) != layout.k or len(set(offload)) != len(offload) or not set(offload) <= set(layout.geometry.layers):
        raise error(f"an offload list needs {layout.k} distinct switchable layers (same K as the slab)")
    return offload


def initial_routes(layout, offload):
    """Offloaded layers take hot planes in layer order, the others resident planes in layer order."""
    offload = _check_list(layout, offload)
    resident = [layer for layer in layout.geometry.layers if layer not in offload]
    return Routes(tuple((layer, OFFLOADED, offload.index(layer)) if layer in offload
                        else (layer, RESIDENT, resident.index(layer)) for layer in layout.geometry.layers))


@dataclass(frozen=True)
class SwitchPlan:
    """Layers entering S (resident -> offloaded) take the hot planes the leaving layers free, and the leaving
    layers take the resident planes the entering layers free, pairwise in layer order."""
    source: Routes
    target: Routes
    entering: tuple          # ((layer, old resident plane, new hot plane), ...)
    leaving: tuple           # ((layer, old hot plane, new resident plane), ...)


def plan_switch(layout, routes, target):
    target = _check_list(layout, target, SwitchError)
    current = routes.offload
    entering = [layer for layer in target if layer not in current]
    leaving = [layer for layer in current if layer not in target]
    pairs_in = [(layer, routes.route(layer)[1], routes.route(out)[1]) for layer, out in zip(entering, leaving)]
    pairs_out = [(layer, routes.route(layer)[1], routes.route(into)[1]) for layer, into in zip(leaving, entering)]
    moved = {layer: (OFFLOADED, hot) for layer, _, hot in pairs_in}
    moved.update({layer: (RESIDENT, plane) for layer, _, plane in pairs_out})
    entries = tuple((layer, *moved.get(layer, (mode, plane))) for layer, mode, plane in routes.entries)
    return SwitchPlan(routes, Routes(entries, routes.epoch + 1), tuple(pairs_in), tuple(pairs_out))


def rollback_plan(plan):
    """The exact inverse: every moved layer returns to the plane it held before the switch."""
    return SwitchPlan(plan.target, plan.source,
                      tuple((layer, new, old) for layer, old, new in plan.leaving),
                      tuple((layer, new, old) for layer, old, new in plan.entering))


@dataclass(frozen=True)
class Request:
    """One active request on one rank. Its first ``home`` tokens (C4 records for DeepSeek-V4) are at their
    home rows; a newest token produced by decode and not yet flushed sits only in its tail row."""
    slot: int
    home: int
    token_rows: tuple        # logical rows of tokens 0..home-1 (req_to_token, or compressed rows)
    host_rows: tuple         # host rows of tokens 0..home-1 (req_to_host_pool)
    buffer_rows: tuple       # plane-relative rows the native allocator gave this request's hot buffer


def copies(layout, plan, requests, *, host_shadow):
    """Row copies of one switch for the active requests, as ((space, row), (space, row)) pairs.

    ``("slab", row)`` is an absolute slab row and ``("host", layer, row)`` a host row of one layer. Phase 1
    completes the host history of the entering layers (none with ``host_shadow``); phase 2 rebuilds the
    reassigned planes from host only. ``tables`` are the native tables (tokens, locations, LRU order) the
    entering layers start from: the state the pinned allocator leaves after admission (slot j holds token
    j, LRU in slot order) with absolute locations. Tail rows never move.
    """
    g, phase1, phase2, tables = layout.geometry, [], [], []
    for request in requests:
        home = request.home
        for layer, old_plane, new_plane in plan.entering:
            old_base, new_base = layout.resident_base(old_plane), layout.hot_base(new_plane)
            if not host_shadow:
                phase1 += [(("slab", old_base + request.token_rows[t]), ("host", layer, request.host_rows[t]))
                           for t in range(home)]
            phase2 += [(("host", layer, request.host_rows[t]), ("slab", new_base + request.buffer_rows[t]))
                       for t in range(min(home, g.hot_slots, len(request.buffer_rows)))]
            locs = [new_base + row for row in request.buffer_rows[:g.hot_slots + g.page]]
            tables.append((layer, request.slot, list(range(g.hot_slots)) + [-1] * g.page,
                           locs + [-1] * (g.hot_slots + g.page - len(locs)), list(range(g.hot_slots))))
        for layer, _, new_plane in plan.leaving:
            base = layout.resident_base(new_plane)
            phase2 += [(("host", layer, request.host_rows[t]), ("slab", base + request.token_rows[t]))
                       for t in range(home)]
    return dict(phase1=phase1, phase2=phase2, tables=tables)


def table_bytes(geometry):
    """Native tables of one layer that can be offloaded: tokens and locations int32, LRU order int16."""
    g = geometry
    return g.slots * ((g.hot_slots + g.page) * 4 * 2 + g.hot_slots * 2)


def ledger(layout, *, host_rows=None):
    """Device and pinned host bytes of the switchable layers in the dynamic arm and in a frozen arm with the
    same K: identical planes; the dynamic arm adds the tails, native tables and host history for every
    layer of U instead of the K offloaded ones, and the route descriptor. ``host_rows`` defaults to one
    resident plane of rows. Layers outside U cost the same in both arms and are not counted."""
    g = layout.geometry
    host_rows = g.resident_rows if host_rows is None else host_rows
    planes = dict(hot=layout.k * g.row_bytes(g.hot_rows),
                  resident=layout.resident_planes * g.row_bytes(g.resident_rows))
    tails = g.row_bytes(layout.rows - layout.tail_base)
    dynamic = dict(planes, tails=tails, tables=len(g.layers) * table_bytes(g), descriptor=len(g.layers) * 2 * 4)
    frozen = dict(planes, tails=0, tables=layout.k * table_bytes(g), descriptor=0)
    return dict(slab_rows=layout.rows, switchable_layers=list(g.layers), k=layout.k,
                device=dict(dynamic=dict(dynamic, total=sum(dynamic.values())),
                            frozen=dict(frozen, total=sum(frozen.values()))),
                host=dict(dynamic=len(g.layers) * g.row_bytes(host_rows), frozen=layout.k * g.row_bytes(host_rows)))


class SwitchProtocol:
    """TP switch of one rank between pre-registered offload lists.

    The scheduler calls ``switch`` on every TP rank between batches with the same list name; no
    forward runs until it returns. The executor owns the device:

    * ``quiesce()`` waits for the previous step on every stream and returns a digest of the active
      requests; it raises ``SwitchError`` while a prefill chunk or staging copy is in flight;
    * ``prepare(plan)`` runs phase 1 and returns its row count; it never changes what the current
      routes read;
    * ``apply(plan)`` runs phase 2, writes the native tables of the layers that become offloaded and
      the route descriptor, synchronizes and returns its row count. ``apply(rollback_plan(plan))``
      rebuilds the source routes the same way;
    * ``publish(routes)`` switches host-side state (staging, flush and prefill routing) to the routes.

    Any refusal or failure before publication leaves or puts every rank back on the source routes and
    raises ``SwitchError``; a rank that cannot restore them poisons the group (``SwitchProtocolError``).
    """

    def __init__(self, layout, lists, executor, transport, *, active, config_digest):
        if type(config_digest) is not str or not config_digest:
            raise ValueError("Expected a shared nonempty configuration digest")
        if type(lists) is not dict or active not in lists:
            raise ValueError("lists maps names to offload lists and must contain the active list")
        self.lists = {name: _check_list(layout, offload) for name, offload in lists.items()}
        if len(set(self.lists.values())) != len(self.lists):
            raise ValueError("two registered names hold the same offload list")
        if set().union(*self.lists.values()) != set(layout.geometry.layers):
            raise ValueError("the switchable layers must be exactly the union of the registered lists")
        self.layout, self.executor, self._transport = layout, executor, transport
        self.active, self.routes = active, initial_routes(layout, self.lists[active])
        self._config = (config_digest, layout.digest, sorted(self.lists.items()))
        self._failure = None
        self._lock = Lock()
        self.receipts = []

    @property
    def poisoned(self):
        return self._failure is not None

    def _gather(self, phase, value):
        records = self._transport.all_gather((phase, self._transport.rank, value))
        if len(records) != self._transport.world_size or any(
            type(record) is not tuple or len(record) != 3 or record[:2] != (phase, rank)
            for rank, record in enumerate(records)
        ):
            raise SwitchProtocolError(f"Invalid rank or phase in {phase} vote")
        return tuple(record[2] for record in records)

    def switch(self, name):
        """Move to the list ``name``; returns the committed transition receipt."""
        if not self._lock.acquire(blocking=False):
            self._failure = "concurrent switch"
            raise SwitchProtocolError(self._failure)
        try:
            return self._switch(name)
        except SwitchError:
            raise
        except BaseException as exc:
            self._failure = self._failure or f"{type(exc).__name__}: {exc}"
            if isinstance(exc, SwitchProtocolError):
                raise
            raise SwitchProtocolError("TP switch failed; stop the worker group") from exc
        finally:
            self._lock.release()

    def _attempt(self, call, *args):
        """Run one rank-local step; failures (interruptions included) become a vote, not a crash."""
        try:
            return True, call(*args)
        except BaseException as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def _switch(self, name):
        if self.poisoned:
            raise SwitchProtocolError("TP switch group is poisoned")
        identity = (self.routes.epoch, self.routes.digest, self._config)
        target = self.lists.get(name)
        proposals = self._gather("propose", (identity, name, target is not None))
        if any(proposal[0] != identity for proposal in proposals):
            raise SwitchProtocolError("TP ranks disagree on epoch, routes or configuration")
        if any(proposal[1:] != (name, True) for proposal in proposals):
            raise SwitchError(f"TP ranks proposed different or unregistered lists: {[p[1] for p in proposals]}")
        plan = plan_switch(self.layout, self.routes, target)
        receipt = dict(event="hisparse_dynamic_offload_switch", epoch=plan.target.epoch, source=self.active,
                       target=name, source_routes=plan.source.digest, target_routes=plan.target.digest,
                       entering=[list(e) for e in plan.entering], leaving=[list(e) for e in plan.leaving])
        if not plan.entering:
            return dict(receipt, epoch=self.routes.epoch, status="unchanged")
        started = time.perf_counter()
        ok, value = self._attempt(lambda: (self.executor.quiesce(), self.executor.prepare(plan)))
        votes = self._gather("prepare", (ok, value))
        prepared = time.perf_counter()
        if not all(ok for ok, _ in votes):
            self._record(receipt, "refused", votes)
            raise SwitchError(f"switch to {name} refused: {[value for ok, value in votes if not ok]}")
        if len({value for _, value in votes}) != 1:
            self._record(receipt, "refused", votes)
            raise SwitchError(f"switch to {name} refused: TP ranks hold different requests (rank skew)")
        ok, value = self._attempt(self.executor.apply, plan)
        applied = self._gather("apply", (ok, value))
        if not all(ok for ok, _ in applied) or len({value for _, value in applied}) != 1:
            restored, _ = self._attempt(self.executor.apply, rollback_plan(plan))
            if not all(self._gather("rollback", restored)):
                raise SwitchProtocolError("A rank could not restore the source routes")
            self._record(receipt, "rolled_back", applied)
            raise SwitchError(f"switch to {name} rolled back on every rank: {[v for ok, v in applied if not ok]}")
        receipt.update(status="committed", rows=dict(prepare=votes[0][1][1], apply=applied[0][1]),
                       requests_sha256=votes[0][1][0])
        digest = _digest(receipt)
        ok, _ = self._attempt(self.executor.publish, plan.target)
        if any(value != digest for value in self._gather("commit", digest if ok else None)):
            raise SwitchProtocolError("TP switch publication failed after every rank applied the target routes")
        self.routes, self.active = plan.target, name
        return self._record(receipt, "committed", None, receipt_sha256=digest,
                            seconds=dict(prepare=prepared - started, apply=time.perf_counter() - prepared))

    def _record(self, receipt, status, votes, **extra):
        entry = dict(receipt, status=status, **extra)
        if votes is not None:
            entry["votes"] = [list(vote) for vote in votes]
        self.receipts.append(entry)
        return entry
