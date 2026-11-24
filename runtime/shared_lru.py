"""CPU reference for the per-request, per-TP-rank cross-layer HiSparse hot pool.

Standard library only; importing this module never imports torch or a CUDA
extension. Three independent views of the same contract live here:

* ``SharedLRUReference``: the specified semantics with explicit epochs, row
  states, leases, generations and policies (isolated/shared, LRU/FIFO,
  static/soft-floor quotas, bypass admission).
* ``NativeLayerLRU``: the pinned SGLang layer-local LRU list (stock control).
* ``KernelMirror``: the exact sequential statement of ``shared_lru_kernel.cuh``
  on flat tables, used to cross-check the reference and as the GPU oracle.

A physical row is ``(plane, loc)``: the native lockstep device location ``loc``
in the main-KV plane of an eligible layer. Arena row ``r`` of a request is the
native hot slot ``r % H`` of plane ``r // H``. ``DirectReader`` validates
resolved addresses against synthesized bytes without reading any policy state.
"""
from dataclasses import dataclass

HIT, MISS, LATEST, DIRECT, PAD, BYPASS = "hit", "miss", "latest", "direct", "pad", "bypass"
FREE, LOADING, VALID, SCRATCH = "free", "loading", "valid", "scratch"
EPOCH_MASK = (1 << 32) - 1
FREE_TAG, TAKEN_TAG = -1, -2  # device owner-tag sentinels (KernelMirror / CUDA)
GENERATION_MASK = 0x7FFF      # device tag = generation << 16 | eligible plane index


class Deferred(RuntimeError):
    """Resolution would be unsafe now (pinned victims, pending backup); no state changed."""


class StaleCompletion(RuntimeError):
    """A completion refers to an unknown, finished or reused lease; no state changed."""


class RequestFailed(RuntimeError):
    """The request saw a failed copy before publication and must be finished."""


def _mix64(x):
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def payload(generation, rank, layer, token, plane, item_bytes):
    """Synthetic host bytes for one K or V item; every byte depends on all inputs."""
    if plane not in ("k", "v"):
        raise ValueError("plane must be 'k' or 'v'")
    seed = _mix64(_mix64(_mix64(_mix64(generation) ^ rank) ^ layer) ^ token) ^ (1 if plane == "v" else 2)
    return bytes(_mix64(seed ^ (element * 0xD1B54A32D192ED03)) & 0xFF for element in range(item_bytes))
@dataclass(frozen=True)
class ArenaPlan:
    """Arena geometry for one request on one rank, from the config and the real pool mapping."""
    layers: tuple    # eligible absolute physical KV layers, in plane order
    hot_rows: int    # H: native saturated historical rows per eligible layer
    selected: int    # K: flattened selection width (blocks x sparse block size)
    mode: str
    eviction: str
    quota: str
    domains: tuple   # isolated: ((offset, size), ...) per plane; shared: one entry
    scratch: tuple   # (offset, size): bypass scratch rows charged to the arena
    bypass: tuple    # plane indices whose misses go through scratch
    floors: tuple    # soft-floor rows per plane (zeros unless quota=soft_floor)

    @property
    def rows(self):
        return len(self.layers) * self.hot_rows

    def plane(self, layer):
        try:
            return self.layers.index(layer)
        except ValueError:
            raise ValueError(f"Layer {layer} is not an eligible host-backed layer") from None

    def domain(self, plane):
        return self.domains[plane] if self.mode == "isolated" else self.domains[0]

    def domain_index(self, row):
        """Isolated domain (plane index) containing ``row``; None in the shared mode."""
        if self.mode == "shared":
            return None
        for index, (offset, size) in enumerate(self.domains):
            if offset <= row < offset + size:
                return index
        raise ValueError(f"Row {row} is outside every isolated domain")

    def owner_template(self):
        """Plane owning each row at admission; None marks free or scratch rows."""
        owners = []
        for row in range(self.rows):
            plane = row // self.hot_rows
            if self.scratch[1] and row >= self.scratch[0]:
                owners.append(None)
            elif self.mode == "shared":
                owners.append(plane)
            else:
                owners.append(plane if self.domain_index(row) == plane else None)
        return owners

    def order_template(self):
        """Device recency lists at admission: per domain, free rows then valid rows, by row ID."""
        owners, order = self.owner_template(), []
        for offset, size in self.domains:
            rows = range(offset, offset + size)
            order += [r for r in rows if owners[r] is None] + [r for r in rows if owners[r] is not None]
        return order
def plan_arena(config, layers, hot_rows, selected):
    """Validate geometry-dependent rules the JSON parser cannot know."""
    pool = config.hot_pool
    if pool.layout != "arena":
        raise ValueError("An arena plan requires hot_pool.layout=arena")
    layers = tuple(layers)
    if (not layers or any(type(x) is not int or x < 0 for x in layers)
            or list(layers) != sorted(set(layers))):
        raise ValueError("Eligible layers must be sorted unique absolute physical layer IDs")
    if any(type(x) is not int for x in (hot_rows, selected)) or not 0 < selected <= hot_rows:
        raise ValueError("Require 0 < selected width <= H")
    index = {layer: i for i, layer in enumerate(layers)}
    rows = len(layers) * hot_rows
    for name, ids in (("bypass_layers", pool.bypass_layers), ("quota_rows", [x for x, _ in pool.quota_rows]),
                      ("floor_rows", [x for x, _ in pool.floor_rows])):
        if set(ids) - set(layers):
            raise ValueError(f"hot_pool.{name} names non-eligible layer(s) {sorted(set(ids) - set(layers))}")
    floors = [0] * len(layers)
    if pool.mode == "isolated":
        sizes = [hot_rows] * len(layers)
        if pool.quota == "static":
            quota = dict(pool.quota_rows)
            if set(quota) != set(layers):
                raise ValueError("Static quota must list every eligible layer")
            sizes = [quota[layer] for layer in layers]
            if sum(sizes) != rows:
                raise ValueError(f"Static quota must partition exactly {rows} arena rows, got {sum(sizes)}")
            if min(sizes) < selected:
                raise ValueError("Every static quota must hold one complete selection")
        offsets = [sum(sizes[:i]) for i in range(len(sizes))]
        return ArenaPlan(layers, hot_rows, selected, pool.mode, pool.eviction, pool.quota,
                         tuple(zip(offsets, sizes)), (rows, 0), (), tuple(floors))
    bypass = tuple(index[x] for x in pool.bypass_layers)
    scratch = selected if bypass else 0
    shared = rows - scratch
    if shared < selected:
        raise ValueError("Shared domain cannot hold one complete selection after scratch")
    for layer, count in pool.floor_rows:
        floors[index[layer]] = count
    if sum(floors) > shared:
        raise ValueError("Soft floors exceed the shared domain")
    return ArenaPlan(layers, hot_rows, selected, pool.mode, pool.eviction, pool.quota,
                     ((0, shared),), (shared, scratch), bypass, tuple(floors))


def _age(epoch, stamp):
    return (epoch - stamp) & EPOCH_MASK


def _newer(epoch, last):
    """Wrap-safe strict order on 32-bit epochs (valid while spans stay below 2**31)."""
    return last is None or 0 < ((epoch - last) & EPOCH_MASK) < (1 << 31)


def selected_tokens(blocks, block_size, seq_len):
    """Token per flattened position; -1 for invalid blocks, tails and out-of-range positions."""
    out = []
    for block in blocks:
        for offset in range(block_size):
            token = block * block_size + offset if block >= 0 else -1
            out.append(token if 0 <= token < seq_len else -1)
    return out
class FakeSpanAllocator:
    """Page-granular lockstep device locations, like the stock HiSparse allocator.

    A span reserves the same ``loc`` in every plane. ``allocated_rows`` counts
    physical rows (locs x planes); the arena reports logical occupancy
    separately. Page 0 is the padding sink and is never handed out.
    """

    def __init__(self, total_locs, page, planes):
        if any(type(x) is not int or x <= 0 for x in (total_locs, page, planes)) or total_locs % page:
            raise ValueError("Require positive integers with total_locs divisible by page")
        self.page, self.planes, self.total_locs = page, planes, total_locs
        self._free = list(range(page, total_locs, page))
        self._owned = {}

    def alloc(self, slot, locs):
        if slot in self._owned:
            raise ValueError("Slot already owns a device span")
        pages = -(-locs // self.page)
        if pages > len(self._free):
            return None
        starts = [self._free.pop() for _ in range(pages)]  # highest pages first: non-identity locs
        self._owned[slot] = starts
        return [start + i for start in starts for i in range(self.page)][:locs]

    def free(self, slot):
        starts = self._owned.pop(slot, None)
        if starts is None:
            return False
        self._free = sorted(self._free + starts)
        return True

    @property
    def allocated_rows(self):
        return sum(len(s) for s in self._owned.values()) * self.page * self.planes

    @property
    def free_pages(self):
        return len(self._free)


@dataclass(frozen=True)
class Selection:
    """One graph row of a layer invocation: ordered selector blocks of one request."""
    slot: int
    generation: int
    blocks: tuple
    seq_len: int
    real: bool = True


@dataclass(frozen=True)
class Batch:
    layer: int
    block_size: int
    rows: tuple
@dataclass
class Resolved:
    """Per graph row: addresses and classes in original selection order."""
    addresses: list            # (plane, loc) per position, None for padding/invalid
    classes: list
    unique: int = 0            # unique history selections (duplicates counted once)
    hits: int = 0
    misses: int = 0
    bypassed: int = 0
    bytes_requested: int = 0   # K+V payload bytes copied from host
    floor_breaches: int = 0
    lease: int | None = None
    victims: tuple = ()        # arena rows reused for misses, in miss order


class _Row:
    __slots__ = ("key", "state", "stamp", "pins")

    def __init__(self, state, stamp):
        self.key, self.state, self.stamp, self.pins = None, state, stamp, 0


class _Request:
    def __init__(self, slot, generation, locs, seq_len, rows, initial_stamp):
        self.slot, self.generation, self.locs, self.seq_len = slot, generation, locs, seq_len
        self.rows, self.initial_stamp = rows, initial_stamp
        self.index = {}                       # (generation, layer, token) -> arena row
        self.last_epoch = None
        self.failed = False
        self.host_visible = set(range(seq_len))  # staging backed up every prefill token
        self.tail_token = None


@dataclass
class _Lease:
    id: int
    slot: int
    generation: int
    layer: int | None
    rows: tuple
    pending: set
    loads: tuple = ()          # (row, token) copies not yet published
    undo: tuple = ()           # (row, key, state, stamp) restored by cancel
    backup_token: int | None = None


@dataclass
class _RowPlan:
    req: object
    tokens: list
    kind: str                  # pad, short or long
    hits: dict = None          # token -> row
    misses: tuple = ()
    victims: tuple = ()
    bypass: bool = False
    breaches: int = 0
class SharedLRUReference:
    """Specified arena semantics: one CPU model per TP rank, requests never share rows."""

    def __init__(self, plan, *, page, item_bytes, rank=0, allocator=None, total_locs=None):
        self.plan, self.page, self.item_bytes, self.rank = plan, page, item_bytes, rank
        self.allocator = allocator or FakeSpanAllocator(
            total_locs or (plan.hot_rows + page) * 64 + page, page, len(plan.layers))
        self.device = {}            # (plane, loc) -> (k bytes, v bytes)
        self._requests, self._leases, self._next_lease = {}, {}, 1
        self._owners = plan.owner_template()

    # -- lifecycle --------------------------------------------------------
    def admit(self, slot, generation, rank, eligible_layers, initial_rows, capacity, *, seq_len, epoch=0):
        plan, H = self.plan, self.plan.hot_rows
        if tuple(eligible_layers) != plan.layers:
            raise ValueError("Eligible layers differ from the bound plan")
        if rank != self.rank:
            raise ValueError("TP rank state is local; use one reference per rank")
        if capacity != plan.rows:
            raise ValueError(f"Physical allowance must be the {plan.rows} arena rows")
        if slot in self._requests:
            raise ValueError("Request slot is still live")
        if type(seq_len) is not int or seq_len < 1 or not 0 <= initial_rows <= min(seq_len, H):
            raise ValueError("Initial rows must be loaded prefill positions below H")
        locs = self.allocator.alloc(slot, H + self.page)
        if locs is None:
            raise MemoryError("Device span allocation failed; request not admitted")
        stamp = (epoch - 1) & EPOCH_MASK
        lo, size = plan.scratch
        rows = [_Row(SCRATCH if size and row >= lo else FREE, stamp) for row in range(plan.rows)]
        req = _Request(slot, generation, locs, seq_len, rows, stamp)
        self._requests[slot] = req
        for token in range(initial_rows):
            self._write_position(req, token)
        if seq_len > H:
            req.tail_token = min(seq_len - 1, H)  # prefill KV of position H sits in the tail slot
            for plane, layer in enumerate(plan.layers):
                self.device[(plane, locs[H])] = self._payload(req, layer, req.tail_token)
        return locs

    def _payload(self, req, layer, token):
        return (payload(req.generation, self.rank, layer, token, "k", self.item_bytes),
                payload(req.generation, self.rank, layer, token, "v", self.item_bytes))

    def _write_position(self, req, token):
        """Normal KV write of a short-phase position into every plane's own slot."""
        H = self.plan.hot_rows
        for plane, layer in enumerate(self.plan.layers):
            self.device[(plane, req.locs[token])] = self._payload(req, layer, token)
            row = plane * H + token
            if self._owners[row] == plane:
                r = req.rows[row]
                if r.state != FREE or r.key is not None:
                    raise AssertionError("Short-phase write found an occupied arena row")
                r.key, r.state, r.stamp = (req.generation, layer, token), VALID, req.initial_stamp
                req.index[r.key] = row

    def advance(self, slot, generation):
        """Append one decoded token. A tail is overwritten only after its host backup."""
        req, H = self._live(slot, generation), self.plan.hot_rows
        if req.tail_token is not None and req.tail_token not in req.host_visible:
            raise Deferred("Tail overwrite before previous-token backup completed")
        token = req.seq_len
        req.seq_len += 1
        if token < H:
            self._write_position(req, token)
        else:
            for plane, layer in enumerate(self.plan.layers):
                self.device[(plane, req.locs[H])] = self._payload(req, layer, token)
            req.tail_token = token
        return token

    def backup(self, slot, generation):
        """Start the previous-token D2H backup; returns a lease or None if already visible."""
        req = self._live(slot, generation)
        token = req.seq_len - 1
        if token in req.host_visible:
            return None
        return self._lease(req, None, (), {"backup"}, backup_token=token).id
    def _live(self, slot, generation):
        req = self._requests.get(slot)
        if req is None or req.generation != generation:
            raise StaleCompletion(f"No live request generation {generation} in slot {slot}")
        return req

    def _lease(self, req, layer, rows, pending, **extra):
        lease = _Lease(self._next_lease, req.slot, req.generation, layer, tuple(rows), set(pending), **extra)
        self._next_lease += 1
        for row in lease.rows:
            req.rows[row].pins += 1
        self._leases[lease.id] = lease
        return lease

    def _phys(self, req, row):
        H = self.plan.hot_rows
        return (row // H, req.locs[row % H])

    # -- one layer invocation -------------------------------------------
    def resolve(self, batch, epoch, *, defer_gather=False):
        """Resolve every graph row atomically: plan all rows, then apply, else change nothing."""
        plane = self.plan.plane(batch.layer)
        real = [sel.slot for sel in batch.rows if sel.real]
        if len(set(real)) != len(real):
            raise ValueError("A request appears twice in one layer invocation")
        plans = [self._plan_row(batch, plane, sel, epoch) for sel in batch.rows]
        return [self._apply_row(batch, plane, sel, p, epoch, defer_gather) for sel, p in zip(batch.rows, plans)]

    def _plan_row(self, batch, plane, sel, epoch):
        width = len(sel.blocks) * batch.block_size
        if not sel.real:
            return _RowPlan(None, [-1] * width, "pad")
        req = self._live(sel.slot, sel.generation)
        if req.failed:
            raise RequestFailed(f"Request in slot {sel.slot} failed before publication; finish it")
        if sel.seq_len != req.seq_len:
            raise ValueError("Selection sequence length differs from the request state")
        if not _newer(epoch, req.last_epoch):
            raise ValueError("Epochs must increase in per-request layer execution order")
        if width > self.plan.selected:
            raise ValueError("Selection wider than the planned width K")
        tokens = selected_tokens(sel.blocks, batch.block_size, sel.seq_len)
        if sel.seq_len <= self.plan.hot_rows:
            return _RowPlan(req, tokens, "short")
        if req.tail_token != sel.seq_len - 1:
            raise ValueError("The newest token must be written to the tail (advance) before decode")
        newest, layer = sel.seq_len - 1, batch.layer
        unique = list(dict.fromkeys(t for t in tokens if t >= 0 and t != newest))
        hits = {t: req.index[(req.generation, layer, t)] for t in unique
                if (req.generation, layer, t) in req.index}
        misses = tuple(t for t in unique if t not in hits)
        hidden = [t for t in misses if t not in req.host_visible]
        if hidden:
            raise Deferred(f"Token(s) {hidden[:4]} have no complete host copy yet")
        if plane in self.plan.bypass:
            lo = self.plan.scratch[0]
            victims = tuple(range(lo, lo + len(misses)))
            if any(req.rows[r].pins for r in victims):
                raise Deferred("Bypass scratch is still pinned by an earlier reader")
            return _RowPlan(req, tokens, "long", hits, misses, victims, True)
        victims, breaches = self._victims(req, plane, set(hits.values()), len(misses), epoch)
        return _RowPlan(req, tokens, "long", hits, misses, victims, False, breaches)
    def _victims(self, req, plane, hit_rows, need, epoch):
        """Free rows first, then minimum (policy stamp, row ID); soft floors prefer surplus owners."""
        if need == 0:
            return (), 0
        offset, size = self.plan.domain(plane)
        rows = req.rows
        candidates = [r for r in range(offset, offset + size)
                      if r not in hit_rows and rows[r].pins == 0 and rows[r].state in (FREE, VALID)]
        candidates.sort(key=lambda r: (0, 0, r) if rows[r].state == FREE else (1, -_age(epoch, rows[r].stamp), r))
        chosen, breaches = [], 0
        if self.plan.quota == "soft_floor":
            occupancy = self.occupancy(req.slot, req.generation)
            surplus = [occupancy[layer] - floor for layer, floor in zip(self.plan.layers, self.plan.floors)]
            for r in candidates:
                if len(chosen) == need:
                    break
                if rows[r].state == FREE:
                    chosen.append(r)
                    continue
                owner = self.plan.plane(rows[r].key[1])
                if surplus[owner] > 0:
                    surplus[owner] -= 1
                    chosen.append(r)
            taken = set(chosen)
            for r in candidates:
                if len(chosen) == need:
                    break
                if r not in taken:
                    chosen.append(r)
                    breaches += 1
        else:
            chosen = candidates[:need]
        if len(chosen) < need:
            raise Deferred(f"Only {len(chosen)} of {need} eligible victims are unpinned")
        return tuple(chosen), breaches

    def _apply_row(self, batch, plane, sel, p, epoch, defer_gather):
        H, layer = self.plan.hot_rows, batch.layer
        if p.kind == "pad":
            return Resolved([None] * len(p.tokens), [PAD] * len(p.tokens))
        req = p.req
        req.last_epoch = epoch
        if p.kind == "short":
            return Resolved([(plane, req.locs[t]) if t >= 0 else None for t in p.tokens],
                            [DIRECT if t >= 0 else PAD for t in p.tokens])
        if self.plan.eviction == "lru":
            for row in p.hits.values():
                req.rows[row].stamp = epoch
        undo, loads, placed = [], [], dict(p.hits)
        for token, row in zip(p.misses, p.victims):
            r = req.rows[row]
            undo.append((row, r.key, r.state, r.stamp))
            loads.append((row, token))
            placed[token] = row
            if p.bypass:
                continue
            if r.key is not None:
                del req.index[r.key]
            r.key, r.state, r.stamp = (req.generation, layer, token), LOADING, epoch
            req.index[r.key] = row
        lease = self._lease(req, layer, tuple(p.hits.values()) + p.victims,
                            {"attention"} | ({"gather"} if loads else set()), loads=tuple(loads), undo=tuple(undo))
        if loads and not defer_gather:
            self._gather(lease)
            lease.pending.discard("gather")
        newest, addresses, classes = sel.seq_len - 1, [], []
        for t in p.tokens:
            if t < 0:
                addresses.append(None)
                classes.append(PAD)
            elif t == newest:
                addresses.append((plane, req.locs[H]))
                classes.append(LATEST)
            else:
                addresses.append(self._phys(req, placed[t]))
                classes.append(HIT if t in p.hits else (BYPASS if p.bypass else MISS))
        return Resolved(addresses, classes, len(placed), len(p.hits), len(p.misses),
                        len(p.misses) if p.bypass else 0, 2 * self.item_bytes * len(p.misses),
                        p.breaches, lease.id, p.victims)

    def _gather(self, lease):
        req = self._requests[lease.slot]
        for row, token in lease.loads:
            self.device[self._phys(req, row)] = self._payload(req, lease.layer, token)
            if req.rows[row].state == LOADING:
                req.rows[row].state = VALID  # publish only after both K and V arrived
    # -- completions and failures ----------------------------------------
    def _lease_of(self, lease_id):
        lease = self._leases.get(lease_id)
        req = None if lease is None else self._requests.get(lease.slot)
        if lease is None or req is None or req.generation != lease.generation:
            raise StaleCompletion(f"Lease {lease_id} is unknown, finished or from a reused slot")
        return lease, req

    def _release(self, lease, req):
        for row in lease.rows:
            req.rows[row].pins -= 1
        del self._leases[lease.id]

    def complete(self, lease_id, kind):
        """Record one completion (gather, attention, backup or prefetch); unpin at the last one."""
        lease, req = self._lease_of(lease_id)
        if kind not in lease.pending:
            raise ValueError(f"Lease {lease_id} has no pending {kind!r}")
        if kind == "gather":
            self._gather(lease)
        elif kind == "backup":
            req.host_visible.add(lease.backup_token)
        lease.pending.discard(kind)
        if not lease.pending:
            self._release(lease, req)

    def cancel(self, lease_id):
        """Abort before publication: restore old mappings; payload bytes were never written."""
        lease, req = self._lease_of(lease_id)
        if "gather" in lease.pending:
            for row, key, state, stamp in reversed(lease.undo):
                r = req.rows[row]
                if r.key is not None and r.state != SCRATCH:
                    del req.index[r.key]
                r.key, r.state, r.stamp = key, state, stamp
                if key is not None:
                    req.index[key] = row
        self._release(lease, req)

    def fail(self, lease_id):
        """A copy failed mid-flight: its rows hold undefined bytes, the request must finish."""
        lease, req = self._lease_of(lease_id)
        if "gather" in lease.pending:
            for row, _ in lease.loads:
                r = req.rows[row]
                if r.state != SCRATCH:
                    if r.key is not None:
                        del req.index[r.key]
                    r.key, r.state, r.stamp = None, FREE, req.initial_stamp
            req.failed = True
        self._release(lease, req)

    def pin(self, slot, generation, layer, tokens, kind="prefetch"):
        """Hold rows of published keys for an extra reader (e.g. prefetch) until completion."""
        req = self._live(slot, generation)
        rows = []
        for token in tokens:
            row = req.index.get((generation, layer, token))
            if row is None or req.rows[row].state != VALID:
                raise Deferred(f"Token {token} of layer {layer} is not a published row")
            rows.append(row)
        return self._lease(req, layer, rows, {kind}).id

    def finish(self, slot, generation):
        """Retire a generation after its completion fences. Repeated finish is a no-op."""
        req = self._requests.get(slot)
        if req is None or req.generation != generation:
            return False
        if any(lease.slot == slot for lease in self._leases.values()):
            raise Deferred("Outstanding completions; finish after every fence")
        self.allocator.free(slot)
        del self._requests[slot]
        return True
    # -- inspection -------------------------------------------------------
    def occupancy(self, slot, generation):
        """Rows owned (valid or loading) per eligible layer: logical occupancy, not span size."""
        req = self._live(slot, generation)
        counts = {layer: 0 for layer in self.plan.layers}
        for key in req.index:
            counts[key[1]] += 1
        return counts

    def order(self, slot, generation, plane):
        """Domain rows in victim-preference order; equals the device recency list."""
        req = self._live(slot, generation)
        offset, size = self.plan.domain(plane)
        epoch = req.last_epoch if req.last_epoch is not None else (req.initial_stamp + 1) & EPOCH_MASK
        rows = req.rows
        return sorted(range(offset, offset + size),
                      key=lambda r: (0, 0, r) if rows[r].state == FREE else (1, -_age(epoch, rows[r].stamp), r))

    def locs(self, slot, generation):
        return list(self._live(slot, generation).locs)

    @property
    def outstanding_leases(self):
        return len(self._leases)


class NativeLayerLRU:
    """Pinned SGLang layer-local hot buffer (stock control) on the same physical model.

    Mirrors ``load_cache_to_device_buffer_kernel``: per layer H slots in lockstep
    locations, newest token in slot H, short requests read their own positions,
    and the LRU write-back is [stale evictables][misses][hits]. The pinned kernel
    resolves duplicate tokens through a racy shared-memory hash; this model
    coalesces them, which keeps attention bytes identical and transfers minimal.
    """

    def __init__(self, layers, hot_rows, *, page, item_bytes, rank=0, total_locs=None):
        self.layers, self.hot_rows, self.page = tuple(layers), hot_rows, page
        self.item_bytes, self.rank = item_bytes, rank
        self.allocator = FakeSpanAllocator(total_locs or (hot_rows + page) * 64 + page, page, len(self.layers))
        self.device, self._requests = {}, {}

    def admit(self, slot, generation, *, seq_len):
        H = self.hot_rows
        locs = self.allocator.alloc(slot, H + self.page)
        if locs is None:
            raise MemoryError("Device span allocation failed; request not admitted")
        state = dict(generation=generation, locs=locs, seq_len=seq_len, host_visible=set(range(seq_len)),
                     tokens=[list(range(H)) for _ in self.layers], lru=[list(range(H)) for _ in self.layers],
                     tail=H if seq_len > H else None)
        self._requests[slot] = state
        for token in range(min(seq_len, H)):
            self._write(state, token, token)
        if seq_len > H:
            self._write(state, H, H)
        return locs

    def _write(self, state, slot_index, token):
        for plane, layer in enumerate(self.layers):
            self.device[(plane, state["locs"][slot_index])] = (
                payload(state["generation"], self.rank, layer, token, "k", self.item_bytes),
                payload(state["generation"], self.rank, layer, token, "v", self.item_bytes))

    def advance(self, slot, generation):
        state = self._requests[slot]
        token = state["seq_len"]
        if state["tail"] is not None and state["tail"] not in state["host_visible"]:
            raise Deferred("Tail overwrite before previous-token backup completed")
        state["seq_len"] += 1
        self._write(state, min(token, self.hot_rows), token)
        if token >= self.hot_rows:
            state["tail"] = token
        return token

    def backup(self, slot, generation):
        state = self._requests[slot]
        state["host_visible"].add(state["seq_len"] - 1)

    def resolve(self, batch, epoch=None):
        plane = self.layers.index(batch.layer)
        return [self._resolve_row(batch, plane, sel) for sel in batch.rows]

    def _resolve_row(self, batch, plane, sel):
        H = self.hot_rows
        tokens = selected_tokens(sel.blocks, batch.block_size, sel.seq_len)
        if not sel.real:
            return Resolved([None] * len(tokens), [PAD] * len(tokens))
        state = self._requests[sel.slot]
        locs = state["locs"]
        if sel.seq_len <= H:
            return Resolved([(plane, locs[t]) if t >= 0 else None for t in tokens],
                            [DIRECT if t >= 0 else PAD for t in tokens])
        if state["tail"] != sel.seq_len - 1:
            raise ValueError("The newest token must be written to the tail (advance) before decode")
        newest = sel.seq_len - 1
        slots, lru = state["tokens"][plane], state["lru"][plane]
        unique = list(dict.fromkeys(t for t in tokens if t >= 0 and t != newest))
        where = {token: s for s, token in enumerate(slots) if token >= 0}
        hit_slots = {where[t] for t in unique if t in where}
        misses = [t for t in unique if t not in where]
        if any(t not in state["host_visible"] for t in misses):
            raise Deferred("Miss on a token without a complete host copy")
        evictable = [s for s in lru if s not in hit_slots]
        victims = evictable[:len(misses)]
        for token, s in zip(misses, victims):
            slots[s] = token
            where[token] = s
            self.device[(plane, locs[s])] = (
                payload(state["generation"], self.rank, batch.layer, token, "k", self.item_bytes),
                payload(state["generation"], self.rank, batch.layer, token, "v", self.item_bytes))
        state["lru"][plane] = evictable[len(misses):] + victims + [s for s in lru if s in hit_slots]
        addresses, classes = [], []
        for t in tokens:
            if t < 0:
                addresses.append(None)
                classes.append(PAD)
            elif t == newest:
                addresses.append((plane, locs[H]))
                classes.append(LATEST)
            else:
                addresses.append((plane, locs[where[t]]))
                classes.append(MISS if t in misses else HIT)
        return Resolved(addresses, classes, len(unique), len(unique) - len(misses), len(misses), 0,
                        2 * self.item_bytes * len(misses), 0, None, tuple(victims))

    def locs(self, slot, generation):
        return list(self._requests[slot]["locs"])

    def finish(self, slot, generation):
        state = self._requests.get(slot)
        if state is None or state["generation"] != generation:
            return False
        self.allocator.free(slot)
        del self._requests[slot]
        return True
def device_generation(generation):
    """Non-zero 15-bit generation carried in device owner tags."""
    return generation % GENERATION_MASK + 1


def owner_tag(generation, plane):
    return (device_generation(generation) << 16) | plane


class KernelMirror:
    """Sequential statement of ``rlh_shared_lru::resolve`` on device-shaped tables.

    Per slot: ``owner`` (tag, FREE or TAKEN), ``token`` (native
    ``req_device_buffer_tokens`` entry of the row's plane/slot), ``order``
    (recency list per domain region) and ``occ`` (rows owned per plane).
    Decode is single-stream, so the device keeps no lease table.
    """

    def __init__(self, plan, *, plane_rows):
        self.plan, self.plane_rows, self.status = plan, plane_rows, 0
        self._owners, self._order = plan.owner_template(), plan.order_template()
        self.slots = {}

    def init_slot(self, slot, generation, locs):
        H, planes = self.plan.hot_rows, len(self.plan.layers)
        self.slots[slot] = dict(
            gen=device_generation(generation), locs=list(locs),
            owner=[FREE_TAG if o is None else owner_tag(generation, o) for o in self._owners],
            token=[row % H for row in range(self.plan.rows)], order=list(self._order),
            occ=[sum(1 for o in self._owners if o == p) for p in range(planes)])

    def reset_slot(self, slot):
        if slot in self.slots:
            self.slots[slot]["owner"] = [FREE_TAG] * self.plan.rows

    def flat(self, slot, row):
        H = self.plan.hot_rows
        return (row // H) * self.plane_rows + self.slots[slot]["locs"][row % H]

    def address(self, flat):
        return None if flat < 0 else (flat // self.plane_rows, flat % self.plane_rows)
    def resolve_row(self, slot, plane, blocks, block_size, seq_len):
        """One CTA: returns (flat index per position, copies [(row, token)], counters)."""
        plan, H, st = self.plan, self.plan.hot_rows, self.slots[slot]
        locs, base = st["locs"], plane * self.plane_rows
        tokens = selected_tokens(blocks, block_size, seq_len)
        counters = dict(unique=0, hits=0, misses=0, bypassed=0, breaches=0)
        if seq_len <= H:  # native fast path, shifted into this layer's plane
            return [base + locs[t] if t >= 0 and locs[t] >= 0 else -1 for t in tokens], [], counters
        newest, out, first = seq_len - 1, [-1] * len(tokens), {}
        for i, t in enumerate(tokens):
            if t == newest:
                out[i] = base + locs[H]
            elif t >= 0:
                first.setdefault(t, i)
        uid_token = sorted(first, key=first.get)  # uids follow first occurrence
        uid = {t: u for u, t in enumerate(uid_token)}
        uid_row = [-1] * len(uid_token)
        tag, owner, token, order = (st["gen"] << 16) | plane, st["owner"], st["token"], st["order"]
        bypass = plane in plan.bypass
        offset, size = plan.domain(plane)
        region = order[offset:offset + size]

        def is_hit(row):
            return owner[row] == tag and token[row] in uid

        for row in region:
            if is_hit(row):
                uid_row[uid[token[row]]] = row
        hits = sum(r >= 0 for r in uid_row)
        need = len(uid_token) - hits
        victims, old, breaches = [], [], 0

        def owner_index(value):
            return None if value == FREE_TAG or (value >> 16) != st["gen"] else value & 0xFFFF

        def take(row):
            victims.append(row)
            old.append(owner_index(owner[row]))
            owner[row] = TAKEN_TAG

        if need and bypass:
            victims, old = list(range(plan.scratch[0], plan.scratch[0] + need)), [None] * need
        elif need:
            if plan.quota == "soft_floor":
                surplus = [o - f for o, f in zip(st["occ"], plan.floors)]
                for row in region:
                    if len(victims) == need:
                        break
                    if owner[row] == TAKEN_TAG or is_hit(row):
                        continue
                    o = owner_index(owner[row])
                    if o is None or surplus[o] > 0:
                        if o is not None:
                            surplus[o] -= 1
                        take(row)
            before = len(victims)
            for row in region:
                if len(victims) == need:
                    break
                if owner[row] != TAKEN_TAG and not is_hit(row):
                    take(row)
            breaches = len(victims) - before if plan.quota == "soft_floor" else 0
            if len(victims) < need:  # capacity preflight makes this unreachable; roll back tags
                for row, o in zip(victims, old):
                    owner[row] = FREE_TAG if o is None else (st["gen"] << 16) | o
                self.status |= 1
                return out, [], counters
        miss_uid = [u for u, row in enumerate(uid_row) if row < 0]
        for k, u in enumerate(miss_uid):
            uid_row[u] = victims[k]
        copies = [(uid_row[u], uid_token[u]) for u in miss_uid]
        if not bypass:  # publish after the copies: reverse token, then owner tag
            for k, u in enumerate(miss_uid):
                row = uid_row[u]
                token[row], owner[row] = uid_token[u], tag
                if old[k] is not None:
                    st["occ"][old[k]] -= 1
            st["occ"][plane] += len(miss_uid)
        for i, t in enumerate(tokens):
            if t >= 0 and t != newest:
                out[i] = self.flat(slot, uid_row[uid[t]])
        missed = set(miss_uid)
        if plan.eviction == "lru":
            touched = [uid_row[u] for u in range(len(uid_row)) if not (bypass and u in missed)]
        else:
            touched = [] if bypass else [uid_row[u] for u in miss_uid]
        touched_set = set(touched)
        kept = [row for row in region if row not in touched_set]
        if len(kept) + len(touched) != size:
            self.status |= 2
        order[offset:offset + size] = kept + sorted(touched)
        counters.update(unique=len(uid_row), hits=hits, misses=need,
                        bypassed=need if bypass else 0, breaches=breaches)
        return out, copies, counters


class DirectReader:
    """Second reader: checks each resolved address, offset, validity bit and byte content."""

    def __init__(self, device, *, rank, item_bytes, hot_rows, planes):
        self.device, self.rank, self.item_bytes, self.hot_rows, self.planes = device, rank, item_bytes, hot_rows, planes

    def check(self, resolved, *, generation, layer, plane, tokens, seq_len, locs):
        H, span = self.hot_rows, set(locs[:self.hot_rows])
        if len(resolved.addresses) != len(tokens) or len(resolved.classes) != len(tokens):
            raise AssertionError("Resolved row length differs from the selection width")
        for position, (token, address, cls) in enumerate(zip(tokens, resolved.addresses, resolved.classes)):
            where = f"layer {layer} position {position} token {token}"
            if token < 0:
                if address is not None or cls != PAD:
                    raise AssertionError(f"{where}: invalid position must stay padding")
                continue
            if address is None:
                raise AssertionError(f"{where}: valid position has no address")
            target_plane, loc = address
            if cls == LATEST:
                if token != seq_len - 1 or address != (plane, locs[H]):
                    raise AssertionError(f"{where}: latest row must be this layer's tail slot")
            elif cls == DIRECT:
                if seq_len > H or address != (plane, locs[token]):
                    raise AssertionError(f"{where}: short row must read its own position")
            elif not (0 <= target_plane < self.planes and loc in span):
                raise AssertionError(f"{where}: history row {address} is outside the request span")
            expected = (payload(generation, self.rank, layer, token, "k", self.item_bytes),
                        payload(generation, self.rank, layer, token, "v", self.item_bytes))
            if self.device.get(address) != expected:
                raise AssertionError(f"{where}: bytes at {address} are not this layer/token's K/V")
        return sum(t >= 0 for t in tokens)
COUNTER_FIELDS = ("unique", "hits", "misses", "bypassed", "bytes", "breaches", "latest", "invalid")


def byte_ledger(*, layers, hot_rows, page, item_bytes, slots, pool_rows, plan=None, counters=False):
    """Estimated per-rank byte categories for one arm; GPU receipts replace them with allocations.

    ``pool_rows`` is the row count of one main-KV plane including its padding
    page. Categories are mutually exclusive. Main KV and per-request
    history/tails overlap by construction (the arena reuses native hot slots)
    and are therefore listed under ``overlaps`` instead of being summed.
    """
    L, H = len(layers), hot_rows
    padded = H + page
    native_tables = 2 * L * slots * padded * 4 + L * slots * H * 2   # tokens/locs int32, LRU int16
    arena_tables = 0
    if plan is not None:
        live_tables = 2 * slots * plan.rows + slots * L + slots + L + 1
        init_templates = 3 * plan.rows + L
        arena_tables = (live_tables + init_templates) * 4
        if counters:
            arena_tables += L * len(COUNTER_FIELDS) * 8
    scratch = 0 if plan is None else plan.scratch[1]
    return dict(
        estimated=True,
        rank_local=True,
        main_kv_bytes=L * pool_rows * 2 * item_bytes,
        main_kv_layout="contiguous all-layer planes" if plan is not None else "one allocation per layer",
        native_metadata_bytes=native_tables,
        arena_metadata_bytes=arena_tables,
        per_request=dict(history_rows=L * H, history_payload_bytes=L * H * 2 * item_bytes,
                         tail_rows=L * page, tail_payload_bytes=L * page * 2 * item_bytes,
                         bypass_scratch_rows=scratch, cacheable_history_rows=L * H - scratch),
        overlaps=["per_request history and tails are rows of main_kv_bytes"],
        unavailable={name: "measured only by the GPU receipt" for name in (
            "weights", "graph_workspace", "allocator_fragmentation", "prefill_overlap", "device_peak",
            "pinned_host")},
    )
