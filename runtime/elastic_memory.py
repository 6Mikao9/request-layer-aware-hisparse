"""Elastic KV memory over CUDA virtual memory: physical chunks move between layers instead of being freed.

Every KV buffer reserves virtual addresses for its largest shape once; only the rows in use are backed. Two index
spaces size the backing: the ``logical`` space (token locations shared by every resident layer, the dense layers
and the indexer caches) and the ``hot`` space (HiSparse device rows shared by every offloaded layer). A buffer of
a switchable layer holds a hot region ``[0, hot_offset)`` and a resident region after it, so one pointer and one
row index address both modes. Offloading a layer deactivates its resident region and activates its hot region;
the freed chunks back new logical rows of the remaining resident layers, i.e. more tokens and requests. Chunks are
the only unit of physical memory, so nothing fragments: any free chunk can back any region.

An inactive resident region keeps its chunks until another region needs them ("stale" rows, reclaimed from the
top down), so an offloaded layer can still read rows written before it was offloaded and a backfill only copies
what changed. Virtual addresses never move, so CUDA Graphs captured over these buffers stay valid.

Backends: ``CpuBackend`` (torch CPU tensors; unmapped bytes are poisoned so tests catch stale reads) and
``CudaVmmBackend`` (driver API through ctypes: cuMemCreate, cuMemMap, cuMemSetAccess, cuMemUnmap).
"""
import ctypes
from dataclasses import dataclass, field
import math

LOGICAL, HOT = "logical", "hot"
ALWAYS, SWITCHABLE, HOT_ONLY = "always", "switchable", "hot_only"
POISON = 0xA5


@dataclass(frozen=True)
class BufferSpec:
    """One KV tensor. ``layer`` is the model layer of a switchable or hot-only buffer, None for an always-resident
    one. ``tokens_per_row`` converts logical tokens to rows (4 for a C4 record). ``pad_rows`` precede the first
    real row (SGLang's padding page)."""
    name: str
    row_bytes: int
    kind: str
    layer: int | None = None
    tokens_per_row: int = 1
    pad_rows: int = 0
    shape: tuple = ()          # trailing shape of one row for the tensor view (product * itemsize == row_bytes)
    dtype: str = "uint8"


@dataclass
class _Buffer:
    spec: BufferSpec
    va: int
    nbytes: int
    hot_offset: int                      # first row of the resident region (0 for always-resident buffers)
    chunks: list = field(default_factory=list)   # physical chunk index per virtual chunk, or None


class Layout:
    """Pure chunk accounting; no device. ``max_tokens`` and ``hot_rows`` size the virtual reservations."""

    def __init__(self, specs, *, chunk_bytes, max_tokens, hot_rows, page_tokens):
        if max_tokens % page_tokens or hot_rows % page_tokens:
            raise ValueError("max_tokens and hot_rows must be page multiples")
        self.specs, self.chunk, self.page = tuple(specs), chunk_bytes, page_tokens
        self.max_tokens, self.hot_rows = max_tokens, hot_rows
        names = [s.name for s in self.specs]
        if len(set(names)) != len(names):
            raise ValueError("duplicate buffer names")

    def hot_offset(self, spec):
        """Rows of the hot region, rounded so the resident region starts on a chunk when the row size allows."""
        if spec.kind == ALWAYS:
            return 0
        rows = spec.pad_rows + self.hot_rows
        step = self.chunk // math.gcd(self.chunk, spec.row_bytes)
        return -(-rows // step) * step if step * spec.row_bytes <= 64 * self.chunk else rows

    def region_rows(self, spec, space, size):
        """Rows a region needs to hold ``size`` units of its space (tokens or hot rows), padding included."""
        if space == HOT:
            return spec.pad_rows + size
        return spec.pad_rows + -(-size // spec.tokens_per_row)

    def total_rows(self, spec):
        resident = self.region_rows(spec, LOGICAL, self.max_tokens) if spec.kind != HOT_ONLY else 0
        return self.hot_offset(spec) + resident if spec.kind != ALWAYS else resident

    def ranges(self, spec, *, tokens, hot_rows, offloaded, filling=()):
        """Byte ranges of ``spec`` that must be backed in a state. A ``filling`` layer is offloaded while its
        resident region is being refilled, so both regions are backed."""
        rb, out = spec.row_bytes, []
        if spec.kind == ALWAYS:
            out.append((0, self.region_rows(spec, LOGICAL, tokens) * rb))
            return out
        if spec.layer in offloaded:
            out.append((0, self.region_rows(spec, HOT, hot_rows) * rb))
        if spec.kind == SWITCHABLE and (spec.layer not in offloaded or spec.layer in filling):
            start = self.hot_offset(spec) * rb
            out.append((start, start + self.region_rows(spec, LOGICAL, tokens) * rb))
        return out

    def chunk_set(self, spec, ranges):
        chunks = set()
        for lo, hi in ranges:
            if hi > lo:
                chunks.update(range(lo // self.chunk, -(-hi // self.chunk)))
        return chunks

    def chunks_needed(self, *, tokens, hot_rows, offloaded, filling=()):
        return sum(len(self.chunk_set(s, self.ranges(s, tokens=tokens, hot_rows=hot_rows, offloaded=offloaded,
                                                     filling=filling)))
                   for s in self.specs)

    def max_tokens_for(self, chunks, *, hot_rows, offloaded, filling=()):
        """Largest page multiple of logical tokens (<= max_tokens) that fits ``chunks``; -1 if even 0 does not."""
        need = dict(hot_rows=hot_rows, offloaded=offloaded, filling=filling)
        if self.chunks_needed(tokens=0, **need) > chunks:
            return -1
        lo, hi = 0, self.max_tokens // self.page
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.chunks_needed(tokens=mid * self.page, **need) <= chunks:
                lo = mid
            else:
                hi = mid - 1
        return lo * self.page

    def bytes_per_token(self, offloaded):
        """Marginal device bytes per logical token at an offload set (for reports and the cost model)."""
        return sum(s.row_bytes / s.tokens_per_row for s in self.specs
                   if s.kind == ALWAYS or (s.kind == SWITCHABLE and s.layer not in offloaded))


class CpuBackend:
    """Reference backend: every reservation is backed by a CPU tensor; unmapped bytes hold ``POISON``."""

    def __init__(self, torch):
        self.torch, self._store, self._next = torch, {}, 1 << 20

    def reserve(self, nbytes, chunk):
        va, self._next = self._next, self._next + -(-nbytes // chunk) * chunk + chunk
        self._store[va] = self.torch.full((nbytes,), POISON, dtype=self.torch.uint8)
        return va

    def create(self, count, chunk):
        return list(range(count))

    def _slice(self, va, offset, nbytes):
        base = max(k for k in self._store if k <= va)
        data = self._store[base]
        lo = va - base + offset
        return data[lo:min(lo + nbytes, data.numel())]

    def map(self, va, first, chunk, handles):
        self._slice(va, first * chunk, len(handles) * chunk).zero_()

    def unmap(self, va, offset, chunk):
        self._slice(va, offset, chunk).fill_(POISON)

    def tensor(self, va, nbytes):
        return self._slice(va, 0, nbytes)

    def synchronize(self):
        pass


class _Prop(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int), ("loc_type", ctypes.c_int),
                ("loc_id", ctypes.c_int), ("win32", ctypes.c_void_p), ("compression", ctypes.c_ubyte),
                ("rdma", ctypes.c_ubyte), ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]


class _Access(ctypes.Structure):
    _fields_ = [("loc_type", ctypes.c_int), ("loc_id", ctypes.c_int), ("flags", ctypes.c_int)]


class _CudaArray:
    def __init__(self, va, nbytes):
        self.__cuda_array_interface__ = dict(shape=(nbytes,), typestr="|u1", data=(va, False), version=3)


class CudaVmmBackend:
    """CUDA driver VMM: one physical allocation per chunk, mapped into stable virtual reservations."""
    PINNED, DEVICE, READWRITE = 1, 1, 3

    def __init__(self, torch, device=None):
        self.torch = torch
        torch.cuda.init()
        self.device = torch.cuda.current_device() if device is None else device
        torch.empty(1, device=f"cuda:{self.device}")    # make the primary context current
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.prop = _Prop(type=self.PINNED, loc_type=self.DEVICE, loc_id=self.device)
        self.access = _Access(loc_type=self.DEVICE, loc_id=self.device, flags=self.READWRITE)
        granularity = ctypes.c_size_t()
        self._check(self.lib.cuMemGetAllocationGranularity(ctypes.byref(granularity), ctypes.byref(self.prop), 0),
                    "cuMemGetAllocationGranularity")
        self.granularity = granularity.value

    def _check(self, status, what):
        if status != 0:
            raise RuntimeError(f"{what} failed with CUresult {status}")

    def reserve(self, nbytes, chunk):
        if chunk % self.granularity:
            raise ValueError(f"chunk {chunk} is not a multiple of the VMM granularity {self.granularity}")
        va = ctypes.c_uint64()
        size = -(-nbytes // chunk) * chunk
        self._check(self.lib.cuMemAddressReserve(ctypes.byref(va), ctypes.c_size_t(size), ctypes.c_size_t(chunk),
                                                 ctypes.c_uint64(0), ctypes.c_uint64(0)), "cuMemAddressReserve")
        return va.value

    def create(self, count, chunk):
        handles = []
        for _ in range(count):
            handle = ctypes.c_uint64()
            self._check(self.lib.cuMemCreate(ctypes.byref(handle), ctypes.c_size_t(chunk), ctypes.byref(self.prop),
                                             ctypes.c_uint64(0)), "cuMemCreate")
            handles.append(handle.value)
        return handles

    def map(self, va, first, chunk, handles):
        """Map consecutive chunks starting at chunk ``first``; one access call and one zero fill per run."""
        base = va + first * chunk
        for i, handle in enumerate(handles):
            self._check(self.lib.cuMemMap(ctypes.c_uint64(base + i * chunk), ctypes.c_size_t(chunk),
                                          ctypes.c_size_t(0), ctypes.c_uint64(handle), ctypes.c_uint64(0)),
                        "cuMemMap")
        size = len(handles) * chunk
        self._check(self.lib.cuMemSetAccess(ctypes.c_uint64(base), ctypes.c_size_t(size), ctypes.byref(self.access),
                                            ctypes.c_size_t(1)), "cuMemSetAccess")
        self.tensor(base, size).zero_()

    def unmap(self, va, offset, chunk):
        self._check(self.lib.cuMemUnmap(ctypes.c_uint64(va + offset), ctypes.c_size_t(chunk)), "cuMemUnmap")

    def tensor(self, va, nbytes):
        return self.torch.as_tensor(_CudaArray(va, nbytes), device=f"cuda:{self.device}")

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)


class ElasticMemory:
    """Mapping state of one rank. ``resize`` moves the backing to a new (tokens, hot rows, offload set) state;
    the caller synchronizes the device first when chunks leave a region that kernels may still read."""

    def __init__(self, layout, backend, physical_chunks, *, torch):
        self.layout, self.backend, self.torch = layout, backend, torch
        self.handles = backend.create(physical_chunks, layout.chunk)
        self.free = list(range(physical_chunks - 1, -1, -1))     # pop() hands out low chunk IDs first
        self.buffers = {}
        for spec in layout.specs:
            nbytes = layout.total_rows(spec) * spec.row_bytes
            buf = _Buffer(spec, backend.reserve(nbytes, layout.chunk), nbytes, layout.hot_offset(spec))
            buf.chunks = [None] * (-(-nbytes // layout.chunk))
            self.buffers[spec.name] = buf
        self.tokens, self.hot_rows, self.offloaded, self.filling = 0, 0, frozenset(), frozenset()
        self.stale = {}                       # (layer, buffer) -> bytes of its inactive resident region backed

    @property
    def free_chunks(self):
        return len(self.free)

    @property
    def used_chunks(self):
        return len(self.handles) - len(self.free)

    def view(self, name, region=None):
        """Tensor over a buffer (``region`` "hot" or "resident" for switchable buffers), typed by its spec."""
        buf = self.buffers[name]
        spec, rb = buf.spec, buf.spec.row_bytes
        lo = buf.hot_offset * rb if region == "resident" else 0
        hi = buf.hot_offset * rb if region == "hot" else buf.nbytes
        flat = self.backend.tensor(buf.va, buf.nbytes)[lo:hi]
        dtype = getattr(self.torch, spec.dtype)
        rows = (hi - lo) // rb
        return flat[:rows * rb].view(dtype).view(rows, *spec.shape) if spec.shape else flat[:rows * rb]

    def stale_rows(self, layer):
        """Rows of ``layer``'s inactive resident region that are still backed; row r holds location r (row 0..pad
        is SGLang's padding page), so a location below this bound may still hold its pre-offload KV."""
        out = None
        for buf in self.buffers.values():
            if buf.spec.layer == layer and buf.spec.kind == SWITCHABLE:
                rows = self.stale.get((layer, buf.spec.name), 0) // buf.spec.row_bytes
                out = rows if out is None else min(out, rows)
        return out or 0

    def _wanted(self, tokens, hot_rows, offloaded, filling=()):
        return {name: self.layout.chunk_set(buf.spec, self.layout.ranges(
            buf.spec, tokens=tokens, hot_rows=hot_rows, offloaded=offloaded, filling=filling))
            for name, buf in self.buffers.items()}

    def plan(self, *, tokens, hot_rows, offloaded, filling=(), keep_stale=True):
        """(maps, unmaps, reclaims) of (buffer name, chunk index) to reach a state; None if it does not fit.
        Unneeded chunks of newly inactive resident regions stay mapped (stale) when ``keep_stale``; reclaims take
        stale chunks from the top of their regions only when the free pool runs short."""
        wanted = self._wanted(tokens, hot_rows, offloaded, filling)
        maps, unmaps, stale = [], [], []
        for name, buf in self.buffers.items():
            have = {c for c, h in enumerate(buf.chunks) if h is not None}
            maps += [(name, c) for c in sorted(wanted[name] - have)]
            spec = buf.spec
            keep = keep_stale and spec.kind == SWITCHABLE and spec.layer in offloaded
            resident_lo = buf.hot_offset * spec.row_bytes // self.layout.chunk
            for c in sorted(have - wanted[name]):
                (stale if keep and c >= resident_lo else unmaps).append((name, c))
        shortfall = len(maps) - len(self.free) - len(unmaps)
        reclaims = []
        if shortfall > 0:
            stale.sort(key=lambda item: -item[1])          # top of each region first
            if shortfall > len(stale):
                return None
            reclaims = stale[:shortfall]
        return maps, unmaps, reclaims

    def resize(self, *, tokens, hot_rows, offloaded, filling=(), keep_stale=True):
        planned = self.plan(tokens=tokens, hot_rows=hot_rows, offloaded=offloaded, filling=filling,
                            keep_stale=keep_stale)
        if planned is None:
            raise MemoryError(f"state tokens={tokens} hot_rows={hot_rows} offloaded={sorted(offloaded)} does not "
                              f"fit {len(self.handles)} chunks")
        maps, unmaps, reclaims = planned
        chunk = self.layout.chunk
        if unmaps or reclaims:
            self.backend.synchronize()
        for name, c in unmaps + reclaims:
            buf = self.buffers[name]
            self.backend.unmap(buf.va, c * chunk, chunk)
            self.free.append(buf.chunks[c])
            buf.chunks[c] = None
        runs = {}
        for name, c in maps:
            buf = self.buffers[name]
            buf.chunks[c] = self.free.pop()
            group = runs.setdefault(name, [])
            if group and group[-1][0] + len(group[-1][1]) == c:
                group[-1][1].append(self.handles[buf.chunks[c]])
            else:
                group.append((c, [self.handles[buf.chunks[c]]]))
        for name, group in runs.items():
            for first, handles in group:
                self.backend.map(self.buffers[name].va, first, chunk, handles)
        newly = frozenset(offloaded) - self.offloaded
        self.tokens, self.hot_rows, self.offloaded = tokens, hot_rows, frozenset(offloaded)
        self.filling = frozenset(filling)
        self._update_stale(newly)
        return dict(mapped=len(maps), unmapped=len(unmaps), reclaimed=len(reclaims))

    def _update_stale(self, newly):
        """Backed bytes of each inactive resident region, as a prefix of the region: a hole ends the prefix."""
        for name, buf in self.buffers.items():
            spec = buf.spec
            key = (spec.layer, name)
            if spec.kind != SWITCHABLE or spec.layer not in self.offloaded:
                self.stale.pop(key, None)
                continue
            start = buf.hot_offset * spec.row_bytes
            first = start // self.layout.chunk
            backed = 0
            for c in range(first, len(buf.chunks)):
                if buf.chunks[c] is None:
                    break
                backed = (c + 1) * self.layout.chunk - start
            limit = self.stale.get(key) if spec.layer not in newly else None
            self.stale[key] = backed if limit is None else min(limit, backed)

    def check(self):
        """Invariants: each physical chunk backs at most one virtual chunk and free plus used is the pool."""
        owners = [h for buf in self.buffers.values() for h in buf.chunks if h is not None]
        if len(owners) != len(set(owners)) or set(owners) & set(self.free) or \
                len(owners) + len(self.free) != len(self.handles):
            raise AssertionError("physical chunk accounting broken")
        wanted = self._wanted(self.tokens, self.hot_rows, self.offloaded, self.filling)
        for name, buf in self.buffers.items():
            have = {c for c, h in enumerate(buf.chunks) if h is not None}
            if not wanted[name] <= have:
                raise AssertionError(f"{name} misses backing for its active rows")
