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
SUMMARY_ROWS = 32             # arena rows per owner-summary block (kSummaryRows in the CUDA kernel)
# study_config.FEATURES the resolver implements, and pairs that implement the same phase two ways.
RESOLVER_FEATURES = ("owner_summary", "touched_bitmap", "hit_index", "incremental_lru", "compact_owner")
EXCLUSIVE_FEATURES = (("owner_summary", "hit_index"), ("touched_bitmap", "incremental_lru"))
INDEX_WAYS = 16               # entries per hit-index bucket (kIndexWays): one 64-byte load per probe
INDEX_EMPTY = -1
COMPACT_FREE, COMPACT_TAKEN = 255, 254   # compact_owner codes (kCompactFree, kCompactTaken)
MASK32, MASK64 = (1 << 32) - 1, (1 << 64) - 1


def summary_blocks(rows):
    return -(-rows // SUMMARY_ROWS)


def owner_summary(planes):
    """Bit set of owning planes per 32-row block; ``planes`` holds a plane index or None per arena row."""
    words = [0] * summary_blocks(len(planes))
    for row, plane in enumerate(planes):
        if plane is not None:
            words[row // SUMMARY_ROWS] |= 1 << plane
    return words


# -- hit_index: per-slot two-choice bucket index (plane, token) -> arena row --------------------
def index_geometry(plan):
    """(buckets per half, row bits). Two halves of 16-way buckets hold 2x the rows that can carry a key."""
    keyed = plan.rows - plan.scratch[1]
    bits = max(1, (plan.rows - 1).bit_length())
    if 31 - bits < 8:
        raise ValueError("hit_index keeps a fingerprint of at least 8 bits: the arena must stay below 2**23 rows")
    return -(-keyed // INDEX_WAYS), bits


def index_hash(plane, token):
    """murmur3 fmix64 of (token << 6 | plane), as ``index_mix`` in shared_lru_kernel.cuh."""
    x = ((token << 6) | plane) & MASK64
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & MASK64
    x ^= x >> 33
    x = (x * 0xC4CEB9FE1A85EC53) & MASK64
    return x ^ (x >> 33)


def index_slots(plane, token, half, row_bits):
    """(left bucket, right bucket, fingerprint) of one key."""
    x = index_hash(plane, token)
    return (((x & MASK32) * half) >> 32, half + (((x >> 32) * half) >> 32),
            (x >> 32) & ((1 << (31 - row_bits)) - 1))


def index_insert(buckets, plane, token, row, half, row_bits):
    """Less loaded of the two buckets (ties left), first empty way; False when both are full."""
    left, right, fp = index_slots(plane, token, half, row_bits)
    loads = [INDEX_WAYS - buckets[b].count(INDEX_EMPTY) for b in (left, right)]
    for b in ((left, right) if loads[0] <= loads[1] else (right, left)):
        if INDEX_EMPTY in buckets[b]:
            buckets[b][buckets[b].index(INDEX_EMPTY)] = fp << row_bits | row
            return True
    return False


def index_remove(buckets, plane, token, row, half, row_bits):
    left, right, fp = index_slots(plane, token, half, row_bits)
    for b in (left, right):
        if fp << row_bits | row in buckets[b]:
            buckets[b][buckets[b].index(fp << row_bits | row)] = INDEX_EMPTY
            return True
    return False


def index_find(buckets, plane, token, half, row_bits, verify):
    """Row whose entry carries the key's fingerprint and passes ``verify`` (owner tag and reverse token)."""
    left, right, fp = index_slots(plane, token, half, row_bits)
    mask = (1 << row_bits) - 1
    for b in (left, right):
        for entry in buckets[b]:
            if entry != INDEX_EMPTY and entry >> row_bits == fp and verify(entry & mask):
                return entry & mask
    return -1


def build_index(plan):
    """Template buckets: every admission owner's key inserted in row order (the device copies this table)."""
    flat = index_from_keys(plan, {row: (plane, row % plan.hot_rows)
                                  for row, plane in enumerate(plan.owner_template()) if plane is not None})
    return [flat[i:i + INDEX_WAYS] for i in range(0, len(flat), INDEX_WAYS)]


def index_from_keys(plan, keys):
    """Flat index holding ``{row: (plane, token)}``, inserted in row order."""
    half, bits = index_geometry(plan)
    buckets = [[INDEX_EMPTY] * INDEX_WAYS for _ in range(2 * half)]
    for row in sorted(keys):
        if not index_insert(buckets, *keys[row], row, half, bits):
            raise ValueError("hit_index template overflows its buckets")
    return [entry for bucket in buckets for entry in bucket]


def index_problems(entries, live, half, row_bits):
    """How a flat index (device or mirror) differs from the live keys ``{row: (plane, token)}``.

    Placement inside the two buckets is not compared: concurrent CUDA inserts may pick
    another way or bucket. Every live key must appear once, in one of its buckets."""
    problems, seen, mask = [], set(), (1 << row_bits) - 1
    for position, entry in enumerate(entries):
        if entry == INDEX_EMPTY:
            continue
        row, bucket = entry & mask, position // INDEX_WAYS
        if entry < 0 or row not in live:
            problems.append(f"entry {position} ({entry}) names no live row")
            continue
        left, right, fp = index_slots(*live[row], half, row_bits)
        if bucket not in (left, right) or entry >> row_bits != fp or row in seen:
            problems.append(f"entry {position}: row {row} misplaced, wrong fingerprint or duplicated")
        seen.add(row)
    if set(live) - seen:
        problems.append(f"{len(set(live) - seen)} live keys are missing from the index")
    return problems


# -- incremental_lru: recency ring per domain --------------------------------------------------
def ring_domains(plan):
    """Per recency domain: (ring offset, capacity, dead-word offset, first plane).

    Capacity is D + (planes using the domain + 1) x K. The domain's first plane rebuilds
    once the tail passes D + K, so the remaining planes of a decode step append without
    a rebuild; any call that could overflow rebuilds first."""
    users = 1 if plan.mode == "isolated" else len(plan.layers)
    out, ring, words = [], 0, 0
    for index, (_, size) in enumerate(plan.domains):
        capacity = size + (users + 1) * plan.selected
        out.append((ring, capacity, words, index if plan.mode == "isolated" else 0))
        ring, words = ring + capacity, words + -(-capacity // 32)
    return tuple(out)


def ring_domain(plan, plane):
    return plane if plan.mode == "isolated" else 0


def ring_templates(plan):
    """(ring, positions): each domain's admission recency list at its ring start; row -> ring index."""
    domains = ring_domains(plan)
    ring, positions = [-1] * sum(c for _, c, _, _ in domains), [-1] * plan.rows
    order, start = plan.order_template(), 0
    for (_, size), (offset, _, _, _) in zip(plan.domains, domains):
        for index, row in enumerate(order[start:start + size]):
            ring[offset + index], positions[row] = row, index
        start += size
    return ring, positions


def ring_dead_bits(words, rings):
    """Device dead-bit words (int32, one region per domain) -> one 0/1 per ring entry (``dead``)."""
    bits = []
    for _, capacity, offset, _ in rings:
        bits += [(words[offset + i // 32] & MASK32) >> (i % 32) & 1 for i in range(capacity)]
    return bits


def ring_dead_words(bits, rings):
    """Inverse of ``ring_dead_bits``, as int32 words."""
    words = [0] * sum(-(-capacity // 32) for _, capacity, _, _ in rings)
    for ring_offset, capacity, offset, _ in rings:
        for i in range(capacity):
            words[offset + i // 32] |= bits[ring_offset + i] << (i % 32)
    return [w - (1 << 32) if w >> 31 else w for w in words]


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

    ``features`` selects the statements of the resolver switches.
    ``owner_summary`` adds ``summary`` (owner-plane bits per 32-row block) and
    finds hits by walking the blocks that carry the plane's bit;
    ``touched_bitmap`` tests the hit/touched row set instead of repeating the
    owner/token lookup and appends touched rows in row order.
    ``hit_index`` adds ``index`` (two-choice 16-way buckets of fingerprint|row)
    and ``index_flag`` (sticky overflow: the slot falls back to the row scan).
    ``incremental_lru`` replaces ``order`` by ``ring``/``dead``/``pos``/``meta``
    (head, tail, rebuilds per domain): live ring entries in [head, tail) are the
    recency list. ``compact_owner`` stores one byte per row (plane, 254 taken,
    255 free) instead of the generation-tagged int32. Every switch set must leave
    outputs and the logical tables (``owner_view``, ``order_view``) equal to
    the plain statement.
    """

    def __init__(self, plan, *, plane_rows, features=()):
        features = frozenset(features)
        unknown = features - set(RESOLVER_FEATURES)
        if unknown:
            raise ValueError(f"Unknown resolver feature(s) {sorted(unknown)}")
        for pair in EXCLUSIVE_FEATURES:
            if set(pair) <= features:
                raise ValueError(f"Resolver features {list(pair)} implement the same phase; enable one")
        self.plan, self.plane_rows, self.status = plan, plane_rows, 0
        self.features = features
        self.compact = "compact_owner" in features
        self.free_tag, self.taken_tag = (COMPACT_FREE, COMPACT_TAKEN) if self.compact else (FREE_TAG, TAKEN_TAG)
        self._owners, self._order = plan.owner_template(), plan.order_template()
        if "hit_index" in features:
            self.index_half, self.row_bits = index_geometry(plan)
            self._index = build_index(plan)
        if "incremental_lru" in features:
            self.rings = ring_domains(plan)
            self._ring, self._pos = ring_templates(plan)
        self.slots = {}

    def tag(self, generation, plane):
        """Owner tag of ``plane`` for a device generation (compact_owner: the plane itself)."""
        return plane if self.compact else (generation << 16) | plane

    def owner_index(self, slot, value):
        """Plane owning a row of this slot's generation, or None (free, taken or another generation)."""
        if self.compact:
            return value if 0 <= value < 64 else None
        return None if value < 0 or (value >> 16) != self.slots[slot]["gen"] else value & 0xFFFF

    def init_slot(self, slot, generation, locs):
        H, planes, gen = self.plan.hot_rows, len(self.plan.layers), device_generation(generation)
        state = self.slots[slot] = dict(
            gen=gen, locs=list(locs),
            owner=[self.free_tag if o is None else self.tag(gen, o) for o in self._owners],
            token=[row % H for row in range(self.plan.rows)],
            occ=[sum(1 for o in self._owners if o == p) for p in range(planes)])
        if "incremental_lru" in self.features:
            state.update(ring=list(self._ring), dead=[0] * len(self._ring), pos=list(self._pos),
                         meta=[[0, size, 0] for _, size in self.plan.domains])
        else:
            state["order"] = list(self._order)
        if "owner_summary" in self.features:
            state["summary"] = owner_summary(self._owners)
        if "hit_index" in self.features:
            state.update(index=[list(bucket) for bucket in self._index], index_flag=0)

    def reset_slot(self, slot):
        if slot in self.slots:
            state = self.slots[slot]
            state["owner"] = [self.free_tag] * self.plan.rows
            if "summary" in state:
                state["summary"] = [0] * summary_blocks(self.plan.rows)
            if "index" in state:
                state.update(index=[[INDEX_EMPTY] * INDEX_WAYS for _ in state["index"]], index_flag=0)

    def owner_view(self, slot):
        return [self.owner_index(slot, value) for value in self.slots[slot]["owner"]]

    def live_summary(self, slot):
        """Owner bits recomputed from the tags: what the maintained ``summary`` must equal."""
        return owner_summary(self.owner_view(slot))

    def order_view(self, slot):
        """Recency list per domain region, as the plain statement stores ``order``."""
        state = self.slots[slot]
        if "incremental_lru" not in self.features:
            return list(state["order"])
        out = []
        for (offset, _, _, _), (head, tail, _) in zip(self.rings, state["meta"]):
            out += [state["ring"][offset + i] for i in range(head, tail) if not state["dead"][offset + i]]
        return out

    def live_keys(self, slot):
        """{row: (plane, token)} of every row owned by the slot's generation: what the index must hold."""
        token = self.slots[slot]["token"]
        return {row: (plane, token[row]) for row, plane in enumerate(self.owner_view(slot)) if plane is not None}

    def index_entries(self, slot):
        return [entry for bucket in self.slots[slot]["index"] for entry in bucket]

    def _ring_rebuild(self, slot, domain):
        """Stable compaction of the live entries of [head, tail) to the ring start (the CUDA rebuild)."""
        state = self.slots[slot]
        offset, capacity, _, _ = self.rings[domain]
        head, tail, rebuilds = state["meta"][domain]
        ring, dead, pos = state["ring"], state["dead"], state["pos"]
        live = [ring[offset + i] for i in range(head, tail) if not dead[offset + i]]
        for index, row in enumerate(live):
            ring[offset + index], pos[row] = row, index
        dead[offset:offset + capacity] = [0] * capacity
        state["meta"][domain] = [0, len(live), rebuilds + 1]
        if len(live) != self.plan.domains[domain][1]:
            self.status |= 2

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
        tag, owner, token = self.tag(st["gen"], plane), st["owner"], st["token"]
        bypass = plane in plan.bypass
        offset, size = plan.domain(plane)
        incremental = "incremental_lru" in self.features
        if incremental:  # rebuild first if this call could overflow the ring, or at the domain's first plane
            domain = ring_domain(plan, plane)
            ring_offset, capacity, _, first_plane = self.rings[domain]
            tail = st["meta"][domain][1]
            if tail + plan.selected > capacity or (plane == first_plane and tail > size + plan.selected):
                self._ring_rebuild(slot, domain)
            head, tail = st["meta"][domain][:2]
            ring, dead, pos = st["ring"], st["dead"], st["pos"]
            region = [(i, ring[ring_offset + i]) for i in range(head, tail) if not dead[ring_offset + i]]
        else:
            region = list(enumerate(st["order"][offset:offset + size]))   # (list position, row)

        def is_hit(row):
            return owner[row] == tag and token[row] in uid

        indexed = "hit_index" in self.features and not st["index_flag"]
        if indexed:  # two buckets per selected key; the owner tag and reverse token confirm a fingerprint
            for u, t in enumerate(uid_token):
                uid_row[u] = index_find(st["index"], plane, t, self.index_half, self.row_bits,
                                        lambda row: owner[row] == tag and token[row] == t)
        elif "owner_summary" in self.features:  # physical 32-row blocks whose summary carries the plane's bit
            summary, end = st["summary"], offset + size
            for block in range(offset // SUMMARY_ROWS, (end - 1) // SUMMARY_ROWS + 1):
                if summary[block] >> plane & 1:
                    for row in range(max(offset, block * SUMMARY_ROWS), min(end, (block + 1) * SUMMARY_ROWS)):
                        if is_hit(row):
                            uid_row[uid[token[row]]] = row
        else:  # recency list walk; incremental_lru (and an overflowed index) walks the domain rows instead
            for row in range(offset, offset + size) if incremental else (row for _, row in region):
                if is_hit(row):
                    uid_row[uid[token[row]]] = row
        if "touched_bitmap" in self.features:  # later phases test the phase-3 hit set, not the tables
            hit_rows = {row for row in uid_row if row >= 0}
            is_hit = hit_rows.__contains__
        hits = sum(r >= 0 for r in uid_row)
        need = len(uid_token) - hits
        victims, old, breaches, last_taken = [], [], 0, -1

        def take(index, row):
            nonlocal last_taken
            victims.append(row)
            old.append(self.owner_index(slot, owner[row]))
            owner[row] = self.taken_tag
            last_taken = max(last_taken, index)

        if need and bypass:
            victims, old = list(range(plan.scratch[0], plan.scratch[0] + need)), [None] * need
        elif need:
            if plan.quota == "soft_floor":
                surplus = [o - f for o, f in zip(st["occ"], plan.floors)]
                for index, row in region:
                    if len(victims) == need:
                        break
                    if owner[row] == self.taken_tag or is_hit(row):
                        continue
                    o = self.owner_index(slot, owner[row])
                    if o is None or surplus[o] > 0:
                        if o is not None:
                            surplus[o] -= 1
                        take(index, row)
            before = len(victims)
            for index, row in region:
                if len(victims) == need:
                    break
                if owner[row] != self.taken_tag and not is_hit(row):
                    take(index, row)
            breaches = len(victims) - before if plan.quota == "soft_floor" else 0
            if len(victims) < need:  # capacity preflight makes this unreachable; roll back tags
                for row, o in zip(victims, old):
                    owner[row] = self.free_tag if o is None else self.tag(st["gen"], o)
                self.status |= 1
                return out, [], counters
        miss_uid = [u for u, row in enumerate(uid_row) if row < 0]
        for k, u in enumerate(miss_uid):
            uid_row[u] = victims[k]
        copies = [(uid_row[u], uid_token[u]) for u in miss_uid]
        if not bypass:
            if indexed:  # drop the victims' old keys while their reverse tokens still name them
                for row, o in zip(victims, old):
                    if o is not None and not index_remove(st["index"], o, token[row], row, self.index_half,
                                                          self.row_bits):
                        self.status |= 4
            for k, u in enumerate(miss_uid):  # publish after the copies: reverse token, then owner tag
                row = uid_row[u]
                token[row], owner[row] = uid_token[u], tag
                if old[k] is not None:
                    st["occ"][old[k]] -= 1
            st["occ"][plane] += len(miss_uid)
            if indexed:  # then the new keys; a full bucket pair parks the index for this generation
                for u in miss_uid:
                    if not st["index_flag"] and not index_insert(st["index"], plane, uid_token[u], uid_row[u],
                                                                 self.index_half, self.row_bits):
                        st["index_flag"] = 1
            if "owner_summary" in self.features:  # recompute every block that received a victim
                for row in victims:
                    first = row // SUMMARY_ROWS * SUMMARY_ROWS
                    block = [self.owner_index(slot, owner[r]) for r in range(first, min(plan.rows, first + SUMMARY_ROWS))]
                    st["summary"][row // SUMMARY_ROWS] = owner_summary(block)[0]
        for i, t in enumerate(tokens):
            if t >= 0 and t != newest:
                out[i] = self.flat(slot, uid_row[uid[t]])
        missed = set(miss_uid)
        if plan.eviction == "lru":
            touched = [uid_row[u] for u in range(len(uid_row)) if not (bypass and u in missed)]
        else:
            touched = [] if bypass else [uid_row[u] for u in miss_uid]
        if incremental:  # kill each touched row's entry, append the rows in row order, move the head
            for row in touched:
                i = pos[row]
                if not head <= i < tail or dead[ring_offset + i]:
                    self.status |= 2
                dead[ring_offset + i] = 1
            for k, row in enumerate(sorted(touched)):
                ring[ring_offset + tail + k], pos[row] = row, tail + k
            end = last_taken + 1 if last_taken >= 0 else head   # entries the victim scan consumed
            new_head = next((i for i in range(head, end) if not dead[ring_offset + i]), end)
            st["meta"][domain][:2] = [new_head, tail + len(touched)]
        elif "touched_bitmap" in self.features:  # bit set of the touched rows, appended in bit order
            order = st["order"]
            bits = set(hit_rows) if plan.eviction == "lru" else set()
            if not bypass:
                bits.update(uid_row[u] for u in miss_uid)
            kept = [row for _, row in region if row not in bits]
            appended = [row for row in range(offset, offset + size) if row in bits]
            if len(kept) + len(touched) != size or len(appended) != len(touched):
                self.status |= 2
            order[offset:offset + size] = kept + appended
        else:
            order, touched_set = st["order"], set(touched)
            kept = [row for _, row in region if row not in touched_set]
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


def arena_tables(plan, *, slots, counters=False, features=()):
    """Bytes of every arena tensor ``ArenaRuntime.tables()`` allocates, by its name (zero-size ones omitted).

    With S slots, R arena rows, L planes, D_d domain sizes, K the selection width:
    owner S*R*b (b = 1 with compact_owner, else 4) plus templates; recency either
    ``order`` S*sum(D_d)*4 or, with incremental_lru, ``ring`` S*sum(C_d)*4 with
    C_d = D_d + (planes using d + 1)*K, ``ring_dead`` S*sum(ceil(C_d/32))*4,
    ``ring_pos`` S*R*4 and ``ring_meta`` S*domains*16; hit_index adds
    ``index`` S*2*ceil((R - scratch)/16)*16*4 and one flag per slot;
    owner_summary adds (S+1)*ceil(R/32)*8.
    """
    features = set(features)
    L, R = len(plan.layers), plan.rows
    owner = 1 if "compact_owner" in features else 4
    out = dict(row_owner=slots * R * owner, owner_template=R * owner, occupancy=slots * L * 4,
               occupancy_template=L * 4, generation=slots * 4, floors=L * 4, status=4)
    if "compact_owner" not in features:
        out["owned_template"] = R * 4               # adds the generation to the int32 template
    if "incremental_lru" in features:
        rings = ring_domains(plan)
        ring_rows, words = sum(c for _, c, _, _ in rings), sum(-(-c // 32) for _, c, _, _ in rings)
        out.update(ring=slots * ring_rows * 4, ring_dead=slots * words * 4, ring_pos=slots * R * 4,
                   ring_meta=slots * len(rings) * 16, ring_template=ring_rows * 4, ring_pos_template=R * 4,
                   ring_meta_template=len(rings) * 16)
    else:
        order_rows = sum(size for _, size in plan.domains)   # bypass scratch rows have no recency entry
        out.update(order=slots * order_rows * 4, order_template=order_rows * 4)
    if counters:
        out["counters"] = L * len(COUNTER_FIELDS) * 8
    if "owner_summary" in features:
        out.update(owner_summary=slots * summary_blocks(R) * 8, owner_summary_template=summary_blocks(R) * 8)
    if "hit_index" in features:
        entries = 2 * index_geometry(plan)[0] * INDEX_WAYS
        out.update(index=slots * entries * 4, index_flags=slots * 4, index_template=entries * 4)
    return out


def byte_ledger(*, layers, hot_rows, page, item_bytes, slots, pool_rows, plan=None, counters=False,
                features=()):
    """Estimated per-rank byte categories for one arm; GPU receipts replace them with allocations.

    ``pool_rows`` is the row count of one main-KV plane including its padding
    page. Categories are mutually exclusive. Main KV and per-request
    history/tails overlap by construction (the arena reuses native hot slots)
    and are therefore listed under ``overlaps`` instead of being summed.
    ``arena_tables`` states the formula of every arena tensor. touched_bitmap
    lives in shared memory; no switch allocates HBM workspace per launch.
    """
    L, H = len(layers), hot_rows
    padded = H + page
    native_tables = 2 * L * slots * padded * 4 + L * slots * H * 2   # tokens/locs int32, LRU int16
    breakdown = {} if plan is None else arena_tables(plan, slots=slots, counters=counters, features=features)
    one_slot = {} if plan is None else arena_tables(plan, slots=1, counters=counters, features=features)
    no_slot = {} if plan is None else arena_tables(plan, slots=0, counters=counters, features=features)
    arena_tables_bytes = sum(breakdown.values())
    scratch = 0 if plan is None else plan.scratch[1]
    return dict(
        estimated=True,
        rank_local=True,
        main_kv_bytes=L * pool_rows * 2 * item_bytes,
        main_kv_layout="contiguous all-layer planes" if plan is not None else "one allocation per layer",
        native_metadata_bytes=native_tables,
        arena_metadata_bytes=arena_tables_bytes,
        arena_metadata_breakdown=breakdown,
        arena_metadata_per_slot_bytes=sum(one_slot.values()) - sum(no_slot.values()),
        arena_metadata_fixed_bytes=sum(no_slot.values()),
        resolver_hbm_workspace_bytes=0,
        per_request=dict(history_rows=L * H, history_payload_bytes=L * H * 2 * item_bytes,
                         tail_rows=L * page, tail_payload_bytes=L * page * 2 * item_bytes,
                         bypass_scratch_rows=scratch, cacheable_history_rows=L * H - scratch),
        overlaps=["per_request history and tails are rows of main_kv_bytes"],
        unavailable={name: "measured only by the GPU receipt" for name in (
            "weights", "graph_workspace", "allocator_fragmentation", "prefill_overlap", "device_peak",
            "pinned_host")},
    )
