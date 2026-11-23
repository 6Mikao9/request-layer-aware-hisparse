"""CPU page ownership for static layer groups, independent of CUDA backing."""

from dataclasses import asdict, dataclass
import hashlib
import json

from .policy import validate_capacity


@dataclass(frozen=True)
class LayerGroup:
    name: str
    layers: tuple[int, ...]
    mode: str
    allocatable_pages: int
    hot_capacity: int | None = None

    def __post_init__(self):
        if type(self.name) is not str or not self.name:
            raise ValueError("Group name must be nonempty")
        if (
            type(self.layers) is not tuple
            or not self.layers
            or any(type(layer) is not int or layer < 0 for layer in self.layers)
            or self.layers != tuple(sorted(set(self.layers)))
        ):
            raise ValueError("Group layers must be a strictly increasing tuple")
        if type(self.allocatable_pages) is not int or self.allocatable_pages < 1:
            raise ValueError("Group must have allocatable pages in addition to page zero")
        if self.mode not in ("resident", "hot"):
            raise ValueError("Expected a static resident or hot group")
        if self.mode == "resident" and self.hot_capacity is not None:
            raise ValueError("Resident groups have no hot capacity")
        if self.mode == "hot" and type(self.hot_capacity) is not int:
            raise ValueError("Hot groups require an integer capacity")

    @property
    def backing_pages(self):
        """Include the fixed page-zero padding sink."""
        return self.allocatable_pages + 1


@dataclass(frozen=True)
class LayerLayout:
    layer_count: int
    max_requests: int
    page_tokens: int
    kv_bytes_per_token_per_layer: int
    topk_blocks: int
    block_tokens: int
    groups: tuple[LayerGroup, ...]
    budget_bytes: int

    def __post_init__(self):
        dimensions = (
            self.layer_count, self.max_requests, self.page_tokens,
            self.kv_bytes_per_token_per_layer, self.topk_blocks,
            self.block_tokens, self.budget_bytes,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("Layout dimensions and byte budget must be positive integers")
        if type(self.groups) is not tuple or any(type(group) is not LayerGroup for group in self.groups):
            raise ValueError("Layout must contain an immutable tuple of groups")
        if len({group.name for group in self.groups}) != len(self.groups):
            raise ValueError("Group names must be unique")
        layers = sorted(layer for group in self.groups for layer in group.layers)
        if layers != list(range(self.layer_count)):
            raise ValueError("Groups must partition all absolute layers exactly once")
        for group in self.groups:
            if group.mode == "hot":
                validate_capacity(group.hot_capacity, group.hot_capacity, self.topk_blocks, self.block_tokens)
                if group.hot_capacity % self.page_tokens:
                    raise ValueError("Hot capacity must be page aligned")
                if self.hot_page_limit(group) > group.allocatable_pages:
                    raise ValueError("A hot group must fit one full row and its latest-token page")
        if self.backing_bytes != self.budget_bytes:
            raise ValueError("Declared budget must equal total main KV backing bytes")

    def hot_page_limit(self, group):
        """Full hot row including the latest-token reserve page."""
        if group.mode != "hot":
            raise ValueError("Only hot groups have a hot page limit")
        return group.hot_capacity // self.page_tokens + 1

    @property
    def backing_bytes(self):
        pages_across_layers = sum(len(group.layers) * group.backing_pages for group in self.groups)
        return pages_across_layers * self.page_tokens * self.kv_bytes_per_token_per_layer

    @property
    def digest(self):
        """Stable rank handshake input. Physical addresses are rank-local."""
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def require_same_budget(baseline, candidate):
    """Require matched per-rank main KV bytes, request geometry, and Top-K."""
    fields = (
        "budget_bytes", "layer_count", "max_requests", "page_tokens",
        "kv_bytes_per_token_per_layer", "topk_blocks", "block_tokens",
    )
    for field in fields:
        if getattr(baseline, field) != getattr(candidate, field):
            raise ValueError(f"Layouts differ in {field}")


@dataclass(frozen=True, eq=False)
class RequestLease:
    row: int
    generation: int


class PageBudgetExceeded(RuntimeError):
    pass


class _PageGroup:
    def __init__(self, spec):
        self.spec = spec
        self.free = list(range(spec.allocatable_pages, 0, -1))
        self.refcounts = {}
        self.sealed = set()

    def allocate(self):
        page = self.free.pop()
        self.refcounts[page] = 1
        return page

    def release(self, page):
        self.refcounts[page] -= 1
        if self.refcounts[page] == 0:
            del self.refcounts[page]
            self.sealed.discard(page)
            self.free.append(page)


class LayerPagePool:
    """Single-scheduler ownership. The caller fences all readers before release."""

    def __init__(self, layout):
        if type(layout) is not LayerLayout:
            raise TypeError("Expected a validated LayerLayout")
        self._layout = layout
        self._groups = {group.name: _PageGroup(group) for group in layout.groups}
        self._layers = {
            layer: group for group in layout.groups for layer in group.layers
        }
        self._generations = [0] * layout.max_requests
        self._requests = {}
        self._pages = {}

    @property
    def layout(self):
        return self._layout

    def route(self, layer):
        """Return the frozen group/mode for an absolute layer."""
        if type(layer) is not int or layer not in self._layers:
            raise ValueError("Unknown layer")
        return self._layers[layer]

    def open_request(self, row):
        if type(row) is not int or not 0 <= row < self.layout.max_requests:
            raise ValueError("Request row is out of range")
        if row in self._requests:
            raise ValueError("Request row is already active")
        self._generations[row] += 1
        lease = RequestLease(row, self._generations[row])
        self._requests[row] = lease
        self._pages[row] = {name: [] for name in self._groups}
        return lease

    def active_lease(self, row):
        """Return the currently active lease for a scheduler row."""
        if type(row) is not int or row not in self._requests:
            raise ValueError("Request row is not active")
        return self._requests[row]

    def _request_pages(self, lease):
        if type(lease) is not RequestLease or self._requests.get(lease.row) is not lease:
            raise ValueError("Stale or foreign request lease")
        return self._pages[lease.row]

    def grow(self, lease, targets):
        """Grow named groups to requested page totals as one admission decision."""
        self._grow(lease, targets, prefill=False)

    def grow_prefill(self, lease, targets):
        """Reserve full prefill pages from the same charged group budgets."""
        self._grow(lease, targets, prefill=True)

    def _grow(self, lease, targets, *, prefill):
        pages = self._request_pages(lease)
        additions = {}
        for name, total in targets.items():
            group = self._groups[name]
            current = len(pages[name])
            if type(total) is not int or total < current:
                raise ValueError("Growth target must be at least the current page count")
            if not prefill and group.spec.mode == "hot" and total > self.layout.hot_page_limit(group.spec):
                raise PageBudgetExceeded(f"{name} exceeds its hot row including the latest-token page")
            needed = total - current
            if needed > len(group.free):
                raise PageBudgetExceeded(f"{name} needs {needed} additional pages and has {len(group.free)}")
            additions[name] = needed
        for name, needed in additions.items():
            group = self._groups[name]
            pages[name].extend(group.allocate() for _ in range(needed))

    def page_ids(self, lease, name):
        """Snapshot numeric page IDs. Keep the lease live while using these IDs."""
        return tuple(self._request_pages(lease)[name])

    def free_pages(self, name):
        return len(self._groups[name].free)

    def location(self, lease, layer, slot, *, write=False):
        """Translate a group slot into a token offset inside that layer's backing.

        The resident slot is the request's logical token offset. A hot slot is
        the cache row offset, including any position in the latest-token page.
        The hot resolver still owns the logical-token to hot-slot mapping.
        """
        group = self.route(layer)
        pages = self._request_pages(lease)[group.name]
        if type(slot) is not int or not 0 <= slot < len(pages) * self.layout.page_tokens:
            raise ValueError("Slot is outside the allocated request pages")
        page_index, offset = divmod(slot, self.layout.page_tokens)
        page = pages[page_index]
        if write and page in self._groups[group.name].sealed:
            raise ValueError("Cannot write a sealed resident page")
        return page * self.layout.page_tokens + offset

    def seal_resident_prefix(self, lease, name, count):
        """Mark a completed full-page prefix immutable after its producer completes."""
        group = self._groups[name]
        pages = self._request_pages(lease)[name]
        if group.spec.mode != "resident":
            raise ValueError("Only resident groups support sharing")
        if type(count) is not int or not 0 <= count <= len(pages):
            raise ValueError("Prefix must be within the allocated pages")
        group.sealed.update(pages[:count])

    def share_resident_prefix(self, source, target, name, count):
        """Reference a sealed prefix in an empty target group."""
        source_pages = self._request_pages(source)[name]
        target_pages = self._request_pages(target)[name]
        group = self._groups[name]
        if group.spec.mode != "resident":
            raise ValueError("Only resident groups support sharing")
        if source is target or target_pages:
            raise ValueError("Prefix target must be a distinct request with an empty group")
        if type(count) is not int or not 1 <= count <= len(source_pages):
            raise ValueError("Prefix must contain full allocated pages")
        prefix = source_pages[:count]
        if any(page not in group.sealed for page in prefix):
            raise ValueError("Every shared page must be sealed before sharing")
        for page in prefix:
            group.refcounts[page] += 1
        target_pages.extend(prefix)

    def release_tail(self, lease, name, keep_pages):
        """Drop a group's tail after callers retire its metadata and fence readers."""
        pages = self._request_pages(lease)[name]
        if type(keep_pages) is not int or not 0 <= keep_pages <= len(pages):
            raise ValueError("Retained page count is outside the allocated range")
        group = self._groups[name]
        for page in reversed(pages[keep_pages:]):
            group.release(page)
        del pages[keep_pages:]

    def finish(self, lease):
        """Drop every group reference and make the request row reusable."""
        for name in self._request_pages(lease):
            self.release_tail(lease, name, 0)
        del self._pages[lease.row]
        del self._requests[lease.row]
