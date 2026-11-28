"""Strict ``hisparse-study-config-v1`` parser shared by the three study efforts.

Standard library only. Parsing checks syntax and internal consistency; checks
that need the real model (eligible physical layers, quota totals, selected
width) run when a worker binds the coordinator. ``unsupported_reasons`` states
which arms this build cannot execute so the launcher fails before any worker
starts instead of falling back to another path.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

SCHEMA = "hisparse-study-config-v1"
PLACEMENT_MODES = ("native", "full_resident", "fixed_layers")
SELECTIONS = ("fixed", "random", "calibrated")
POOL_MODES = ("isolated", "shared")
SCOPES = ("request_rank",)
LAYOUTS = ("native", "arena")
EVICTIONS = ("lru", "fifo")
ADMISSIONS = ("all", "bypass_list")
QUOTAS = ("none", "static", "soft_floor")
DIAGNOSTICS = ("none", "counters", "nsys")
# Named engineering switches. An effort registers a name together with its
# implementation and off/on parity test; unregistered names are rejected. Only
# enabled switches enter the canonical form, so false and absent are one config.
# Both switches below change how the arena resolver finds rows, never which rows
# (shared-lru-scan-microbenchmark.md, "优先消融"):
# owner_summary: per 32-row block owner-plane bits; the hit search reads owner tags
#   only in blocks that can hold a row of the requesting layer.
# touched_bitmap: a shared-memory bitmap of the rows one invocation hits or admits
#   replaces the repeated owner/token/hash lookups of the victim scan and the
#   stable recency compaction, and orders the touched rows without a sort.
# The next three replace whole-domain work instead of speeding it up (issue #6):
# hit_index: per-slot GPU hash index (plane, token) -> arena row; the hit search
#   probes only the selected keys (alternative to owner_summary).
# incremental_lru: per-domain recency ring with row positions; a call kills and
#   appends only its touched rows and scans only from the oldest live entry
#   (alternative to touched_bitmap). Dead entries are reclaimed by a stable
#   rebuild at the domain's first plane, at most once per decode step.
# compact_owner: owner tags shrink from int32 (generation << 16 | plane) to one
#   byte per row; slot reuse relies on init_slot rewriting every tag in stream order.
ARENA_FEATURES = ("owner_summary", "touched_bitmap", "hit_index", "incremental_lru", "compact_owner")
# The next three belong to a fixed offload set over the stock hot pool (issue #11, dynamic-offload.md):
# route_indirect: the switchable layers (the union of the registered offload lists) live in one slab per K
#   and V with absolute rows; a GPU route table gives each layer its plane, the decode store writes the
#   newest token to a fixed tail row, and one route-aware swap kernel per layer serves either mode, so a
#   captured CUDA Graph stays valid across switches. Off: every layer keeps its stock pool and path.
# dynamic_offload: the scheduler may move between the registered lists at runtime, agreed on every TP rank
#   and rolled back as a group on failure.
# host_shadow: resident switchable layers keep their host history current, so a switch skips the
#   device-to-host copy of the layers that become offloaded.
PLACEMENT_FEATURES = ("route_indirect", "dynamic_offload", "host_shadow")
FEATURES = ARENA_FEATURES + PLACEMENT_FEATURES
# Pairs that implement the same phase two ways; one ablation measures one of them.
EXCLUSIVE_FEATURES = (("owner_summary", "hit_index"), ("touched_bitmap", "incremental_lru"))

_TOP = {"schema", "placement", "hot_pool", "features", "diagnostics"}
_PLACEMENT = {"mode", "selection", "offload_layers", "alternate_offload_layers", "seed", "calibration"}
_HOT_POOL = {"mode", "scope", "layout", "eviction", "admission", "bypass_layers",
             "quota", "quota_rows", "floor_rows"}


def _strict_object(pairs):
    data = dict(pairs)
    if len(data) != len(pairs):
        raise ValueError("Duplicate JSON object key")
    return data


def _reject_constant(name):
    raise ValueError(f"Non-finite JSON number {name}")


def loads(raw):
    if type(raw) is str:
        raw = raw.encode("utf-8")
    text = bytes(raw).decode("utf-8")  # strict: invalid UTF-8 raises
    if text.startswith("\ufeff"):
        raise ValueError("Study config must not start with a byte-order mark")
    data = json.loads(text, object_pairs_hook=_strict_object, parse_constant=_reject_constant)
    if type(data) is not dict:
        raise ValueError("Study config must be a JSON object")
    return data


def _section(data, key, allowed):
    value = data.get(key, {})
    if type(value) is not dict:
        raise ValueError(f"{key} must be a JSON object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {key} field(s): {sorted(unknown)}")
    return value


def _choice(section, key, choices, where):
    value = section.get(key, choices[0])
    if type(value) is not str or value not in choices:
        raise ValueError(f"{where}.{key} must be one of {list(choices)}")
    return value


def _count(value, where):
    if type(value) is not int or value < 0:
        raise ValueError(f"{where} must be a non-negative integer")
    return value


def _layers(value, where):
    if type(value) is not list:
        raise ValueError(f"{where} must be an array of absolute physical layer IDs")
    layers = tuple(_count(x, where) for x in value)
    if len(set(layers)) != len(layers):
        raise ValueError(f"{where} repeats a layer")
    return tuple(sorted(layers))


def _layer_rows(value, where):
    if type(value) is not dict or not value:
        raise ValueError(f"{where} must be a non-empty object keyed by absolute layer ID")
    rows = []
    for key, count in value.items():
        if not key.isdigit() or not key.isascii() or str(int(key)) != key:
            raise ValueError(f"{where} key {key!r} is not a canonical decimal layer ID")
        rows.append((int(key), _count(count, f"{where}[{key}]")))
    return tuple(sorted(rows))


def _digest(value, where):
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{where} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class Placement:
    mode: str = "native"
    selection: str = "fixed"
    offload_layers: tuple = ()
    seed: int | None = None
    calibration: tuple | None = None  # (training_sha256, list_sha256)
    alternates: tuple = ()            # further offload lists of the same size; list i + 1 is alternates[i]

    def as_dict(self):
        out = dict(mode=self.mode, selection=self.selection, offload_layers=list(self.offload_layers))
        if self.alternates:
            out["alternate_offload_layers"] = [list(layers) for layers in self.alternates]
        if self.seed is not None:
            out["seed"] = self.seed
        if self.calibration is not None:
            out["calibration"] = dict(training_sha256=self.calibration[0], list_sha256=self.calibration[1])
        return out


@dataclass(frozen=True)
class HotPool:
    mode: str = "isolated"
    scope: str = "request_rank"
    layout: str = "native"
    eviction: str = "lru"
    admission: str = "all"
    bypass_layers: tuple = ()
    quota: str = "none"
    quota_rows: tuple = ()  # ((absolute layer, rows), ...)
    floor_rows: tuple = ()

    def as_dict(self):
        out = dict(mode=self.mode, scope=self.scope, layout=self.layout, eviction=self.eviction,
                   admission=self.admission, quota=self.quota)
        if self.bypass_layers:
            out["bypass_layers"] = list(self.bypass_layers)
        if self.quota_rows:
            out["quota_rows"] = {str(layer): rows for layer, rows in self.quota_rows}
        if self.floor_rows:
            out["floor_rows"] = {str(layer): rows for layer, rows in self.floor_rows}
        return out


@dataclass(frozen=True)
class StudyConfig:
    placement: Placement = Placement()
    hot_pool: HotPool = HotPool()
    features: tuple = ()  # ((name, enabled), ...)
    diagnostics: str = "none"

    def as_dict(self):
        return dict(schema=SCHEMA, placement=self.placement.as_dict(), hot_pool=self.hot_pool.as_dict(),
                    features={name: enabled for name, enabled in self.features},
                    diagnostics=dict(mode=self.diagnostics))

    def canonical(self):
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self):
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    @property
    def is_native(self):
        """All-default configuration: stock path, no hooks, no experimental state."""
        return self == StudyConfig()

    @property
    def uses_arena(self):
        return self.hot_pool.layout == "arena"

    @property
    def uses_fixed_layers(self):
        """Listed layers are host-backed with the stock hot pool; the other eligible layers keep full GPU KV."""
        return self.placement.mode == "fixed_layers"

    @property
    def offload_lists(self):
        """Registered offload lists by index; list 0 (``offload_layers``) is active at startup."""
        return (self.placement.offload_layers, *self.placement.alternates)

    def feature(self, name):
        """Resolved value of a registered engineering switch (absent means off)."""
        if name not in FEATURES:
            raise KeyError(f"Unregistered feature switch {name!r}")
        return dict(self.features).get(name, False)


def parse(data):
    if type(data) is not dict:
        raise ValueError("Study config must be a JSON object")
    unknown = set(data) - _TOP
    if unknown:
        raise ValueError(f"Unknown study config field(s): {sorted(unknown)}")
    if data.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA!r}")

    raw = _section(data, "placement", _PLACEMENT)
    mode = _choice(raw, "mode", PLACEMENT_MODES, "placement")
    selection = _choice(raw, "selection", SELECTIONS, "placement")
    offload = _layers(raw.get("offload_layers", []), "placement.offload_layers")
    if (mode == "fixed_layers") != bool(offload):
        raise ValueError("placement.offload_layers is required for fixed_layers and forbidden otherwise")
    if mode != "fixed_layers" and selection != "fixed":
        raise ValueError("placement.selection applies only to placement.mode=fixed_layers")
    seed = raw.get("seed")
    if (selection == "random") != (seed is not None):
        raise ValueError("placement.seed is required for random selection and forbidden otherwise")
    if seed is not None:
        _count(seed, "placement.seed")
    calibration = raw.get("calibration")
    if (selection == "calibrated") != (calibration is not None):
        raise ValueError("placement.calibration is required for calibrated selection and forbidden otherwise")
    if calibration is not None:
        if type(calibration) is not dict or set(calibration) != {"training_sha256", "list_sha256"}:
            raise ValueError("placement.calibration needs exactly training_sha256 and list_sha256")
        calibration = (_digest(calibration["training_sha256"], "placement.calibration.training_sha256"),
                       _digest(calibration["list_sha256"], "placement.calibration.list_sha256"))
    alternates = raw.get("alternate_offload_layers")
    if alternates is not None:
        if mode != "fixed_layers" or type(alternates) is not list or not alternates:
            raise ValueError("placement.alternate_offload_layers is a non-empty array of offload lists for "
                             "placement.mode=fixed_layers")
        alternates = tuple(_layers(item, f"placement.alternate_offload_layers[{index}]")
                           for index, item in enumerate(alternates))
        if any(len(item) != len(offload) for item in alternates):
            raise ValueError("every alternate offload list has as many layers as placement.offload_layers")
        if len(set((offload, *alternates))) != len(alternates) + 1:
            raise ValueError("placement.alternate_offload_layers repeats a registered offload list")
    placement = Placement(mode, selection, offload, seed, calibration, alternates or ())

    raw = _section(data, "hot_pool", _HOT_POOL)
    pool_mode = _choice(raw, "mode", POOL_MODES, "hot_pool")
    scope = _choice(raw, "scope", SCOPES, "hot_pool")
    layout = _choice(raw, "layout", LAYOUTS, "hot_pool")
    eviction = _choice(raw, "eviction", EVICTIONS, "hot_pool")
    admission = _choice(raw, "admission", ADMISSIONS, "hot_pool")
    quota = _choice(raw, "quota", QUOTAS, "hot_pool")
    bypass = _layers(raw["bypass_layers"], "hot_pool.bypass_layers") if "bypass_layers" in raw else ()
    quota_rows = _layer_rows(raw["quota_rows"], "hot_pool.quota_rows") if "quota_rows" in raw else ()
    floor_rows = _layer_rows(raw["floor_rows"], "hot_pool.floor_rows") if "floor_rows" in raw else ()
    if pool_mode == "shared" and layout != "arena":
        raise ValueError("hot_pool.mode=shared requires hot_pool.layout=arena")
    if layout == "native" and (eviction, admission, quota) != ("lru", "all", "none"):
        raise ValueError("hot_pool.layout=native supports only the native policy (lru, all, none)")
    if (admission == "bypass_list") != bool(bypass):
        raise ValueError("hot_pool.bypass_layers is required for bypass_list and forbidden otherwise")
    if admission == "bypass_list" and pool_mode != "shared":
        raise ValueError("hot_pool.admission=bypass_list charges scratch to the shared domain; use mode=shared")
    if (quota == "static") != bool(quota_rows):
        raise ValueError("hot_pool.quota_rows is required for quota=static and forbidden otherwise")
    if (quota == "soft_floor") != bool(floor_rows):
        raise ValueError("hot_pool.floor_rows is required for quota=soft_floor and forbidden otherwise")
    if quota == "static" and pool_mode != "isolated":
        raise ValueError("hot_pool.quota=static partitions isolated domains; use mode=isolated")
    if quota == "soft_floor" and pool_mode != "shared":
        raise ValueError("hot_pool.quota=soft_floor applies to the shared domain; use mode=shared")
    hot_pool = HotPool(pool_mode, scope, layout, eviction, admission, bypass, quota, quota_rows, floor_rows)

    raw = data.get("features", {})
    if type(raw) is not dict:
        raise ValueError("features must be a JSON object")
    for name, enabled in raw.items():
        if name not in FEATURES:
            raise ValueError(f"Unknown feature switch {name!r}; this build registers {list(FEATURES)}")
        if type(enabled) is not bool:
            raise ValueError(f"features.{name} must be a boolean")
        if enabled and name in ARENA_FEATURES and layout != "arena":
            raise ValueError(f"features.{name} changes the arena resolver; hot_pool.layout=native runs the "
                             "stock kernel unmodified")
    features = tuple(sorted((name, True) for name, enabled in raw.items() if enabled))
    for pair in EXCLUSIVE_FEATURES:
        if set(pair) <= {name for name, _ in features}:
            raise ValueError(f"features.{pair[0]} and features.{pair[1]} implement the same resolver phase; "
                             "enable at most one")
    enabled = {name for name, _ in features}
    if enabled & set(PLACEMENT_FEATURES) and (mode != "fixed_layers" or layout != "native"):
        raise ValueError("features.route_indirect, dynamic_offload and host_shadow apply to placement.mode="
                         "fixed_layers with hot_pool.layout=native")
    if "dynamic_offload" in enabled and "route_indirect" not in enabled:
        raise ValueError("features.dynamic_offload needs features.route_indirect: a captured CUDA Graph can only "
                         "switch through the route table")
    if "dynamic_offload" in enabled and not placement.alternates:
        raise ValueError("features.dynamic_offload needs placement.alternate_offload_layers to switch to")
    if "host_shadow" in enabled and "dynamic_offload" not in enabled:
        raise ValueError("features.host_shadow only shortens switches; it needs features.dynamic_offload")
    if placement.alternates and "route_indirect" not in enabled:
        raise ValueError("placement.alternate_offload_layers needs features.route_indirect, whose slab covers "
                         "every registered list")

    raw = _section(data, "diagnostics", {"mode"})
    diagnostics = _choice(raw, "mode", DIAGNOSTICS, "diagnostics")
    return StudyConfig(placement, hot_pool, features, diagnostics)


def read(path, expected_sha256=None):
    """Return (config, file SHA-256). The digest pins the bytes workers re-read."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Study config file SHA-256 changed since launch validation")
    return parse(loads(raw)), digest


def unsupported_reasons(config):
    """Arms this build cannot execute. The launcher refuses them before workers start."""
    reasons = []
    if config.placement.mode == "full_resident":
        reasons.append("placement.mode=full_resident is the all-resident control; serve it with the pinned SGLang "
                       "without --enable-hisparse")
    if config.uses_fixed_layers and config.uses_arena:
        reasons.append("a fixed offload set over the shared arena is owned by the combined effort and is not "
                       "implemented in this build")
    if config.hot_pool.layout == "native" and config.diagnostics != "none":
        reasons.append("diagnostics counters/nsys instrument the arena resolver; stock layout runs unmodified")
    return reasons


def _placement_capabilities(config):
    indirect, dynamic = config.feature("route_indirect"), config.feature("dynamic_offload")
    lists = config.offload_lists
    switchable = sorted(set().union(*lists)) if indirect else list(lists[0])
    return dict(
        runtime="dynamic_layer_offload" if dynamic else "fixed_layers_route_indirect" if indirect else
        "fixed_layers_stock_paths",
        experimental_gpu_state=True,
        placement=dict(offload_lists=[list(layers) for layers in lists], startup_list=0,
                       switchable_layers=switchable),
        offloaded_path=("route-aware swap rlh_dynamic_offload::route (CUDA C++, sm_80) over the slab" if indirect
                        else "stock swap on the offload set's own HiSparse planes"),
        resident_path=("switchable layers: the same kernel writes base + req_to_token rows as HiSparse slots; "
                       "others: the stock consumer reads req_to_token rows" if indirect
                       else "the stock consumer reads req_to_token rows of the full GPU main KV"),
        host_backing="every switchable layer (host_shadow keeps resident ones current)" if indirect
        else "offloaded layers only",
        switching=("TP-agreed runtime switch between registered lists, group rollback" if dynamic else None),
        features={name: config.feature(name) for name in PLACEMENT_FEATURES},
    )


def capabilities(config):
    """Effective implementation selected for a parsed configuration (startup receipt)."""
    pool, arena = config.hot_pool, config.uses_arena
    hit_search = recency_update = owner_tags = None
    if arena:
        if config.feature("hit_index"):
            hit_search = "hit_index: two-choice 16-way buckets keyed by (plane, token), probed per selected key"
        elif config.feature("owner_summary"):
            hit_search = "owner_summary: physical 32-row blocks whose owner-plane bits include the layer"
        else:
            hit_search = "recency list walk with an owner lookup per row"
        if config.feature("incremental_lru"):
            recency_update = ("incremental_lru: recency ring, touched rows killed and appended, victim scan "
                              "from the oldest live entry, stable rebuild at the domain's first plane")
        elif config.feature("touched_bitmap"):
            recency_update = "touched_bitmap: shared-memory bitmap, chunked stable compaction, bit-order append"
        else:
            recency_update = "per-row owner/token/hash lookup, per-1024-row compaction, bitonic sort"
        owner_tags = ("compact_owner: uint8 plane per row, 254 taken, 255 free" if config.feature("compact_owner")
                      else "int32 generation << 16 | plane, -1 free, -2 taken")
    caps = dict(
        runtime="stock_hisparse" if not arena else "shared_lru_arena",
        experimental_gpu_state=arena,
        resolver=None if not arena else "rlh_shared_lru::resolve (CUDA C++, sm_80)",
        consumer_base="per-layer main KV plane" if not arena else "contiguous all-layer main KV view",
        borrowing=pool.mode == "shared",
        eviction=pool.eviction,
        quota=pool.quota,
        admission=pool.admission,
        diagnostics=config.diagnostics,
        registered_features=list(FEATURES),
        features={name: config.feature(name) for name in ARENA_FEATURES} if arena else {},
        hit_search=hit_search,
        recency_update=recency_update,
        owner_tags=owner_tags,
        unsupported=unsupported_reasons(config),
    )
    if config.uses_fixed_layers and not arena:
        caps.update(_placement_capabilities(config))
        caps["consumer_base"] = ("one slab of the switchable layers' planes and tail rows" if
                                 config.feature("route_indirect") else "per-layer main KV plane")
    return caps
