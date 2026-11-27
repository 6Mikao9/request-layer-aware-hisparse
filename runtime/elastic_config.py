"""Strict ``hisparse-elastic-v1`` parser for pressure-driven per-layer HiSparse (issue #13).

Standard library only. The config names the model family, the version (``steady`` or ``smooth``), the order in
which groups of layers leave the GPU as memory pressure rises, the layers that are always offloaded, an optional
TPOT SLO with the cost inputs that predict whether one more group would break it, and the controller and memory
knobs. Layers named nowhere stay resident for the whole run. Checks that need the real model (eligible layer IDs,
buffer geometry) run when a worker binds; ``read`` returns the parsed config and the SHA-256 of the file bytes.

    {"schema": "hisparse-elastic-v1", "model": "minimax_m3", "mode": "steady",
     "layers": {"eviction_order": [[0, 1, 2, 3], [6, 15, 17, 18]], "always_offload": []},
     "slo": {"tpot_ms": 60.0}}
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

from .study_config import _choice, _count, _section, loads

SCHEMA = "hisparse-elastic-v1"
# minimax_m3: sparse MHA layers (K and V rows); dsa: MLA + indexer layers of GLM-5 / GLM-5.1 / GLM-5.2 and
# DeepSeek-V3.2; deepseek_v4: the C4 layers of DeepSeek-V4-Flash (K-only page-padded records).
MODELS = ("minimax_m3", "dsa", "deepseek_v4")
# steady: one CUDA Graph set per offload level with the stock kernels (resident layers read their logical rows
#   directly, offloaded layers run the native swap-in); a switch stops decode, copies, and replays another set.
# smooth: one graph set and a device route table; a switch rewrites the table, backfill copies are paced across
#   steps, and offloaded layers can still hit their old resident rows until those chunks are reused.
MODES = ("steady", "smooth")
# Engineering switches; every one defaults to its listed value and enters the canonical form.
FEATURES = {
    "write_through": True,   # every switchable layer keeps a current host copy, so offload needs no D2H
    "warm_start": True,      # an offloaded layer's hot rows start from its resident rows (D2D), not all-miss
    "stale_hits": True,      # smooth: offloaded layers read resident rows that are still mapped and valid
    "backfill": True,        # layers return to the GPU when pressure falls
    "compaction": True,      # live pages above a shrink ceiling move down instead of blocking the shrink
    "slo_admission": True,   # with an SLO, cap the running batch at the largest size the model predicts fits
    "throughput_guard": True,  # offload another group only if the model expects more decode tokens per second
}
_TOP = {"schema", "model", "mode", "layers", "slo", "profile", "controller", "memory", "features"}
_LAYERS = {"eviction_order", "always_offload", "granularity", "pinned_resident"}
# Symbolic layer lists resolved when a worker binds the real model (``ElasticConfig.resolve``).
BY_MISS_RATE = "by_miss_rate"                     # eligible layers, lowest profiled miss rate first
SHARED_INDEX_FOLLOWERS = "shared_index_followers"  # DSA layers that reuse the previous layer's top-k
_SLO = {"tpot_ms", "tail_factor", "margin_ms"}
_PROFILE = {"bytes_per_miss", "selected_rows", "h2d_gbps", "fixed_us", "per_request_us", "miss_rate",
            "default_miss_rate", "exposed", "base_ms", "measurements"}
_CONTROLLER = {"interval_steps", "high_watermark", "low_watermark", "dwell_steps", "horizon_steps",
               "backfill_gbps", "max_level", "min_level", "initial_level", "min_gain"}
_MEMORY = {"chunk_mib", "hot_rows", "max_logical_tokens", "host_kv_gib", "precapture_levels"}


def _number(value, where, *, low=0.0, high=None, strict=False):
    if type(value) not in (int, float) or value != value:
        raise ValueError(f"{where} must be a finite number")
    if value < low or (strict and value == low) or (high is not None and value > high):
        raise ValueError(f"{where} is out of range")
    return float(value)


def _layer_list(value, where):
    if type(value) is not list:
        raise ValueError(f"{where} must be an array of absolute layer IDs")
    return tuple(_count(x, where) for x in value)


def _groups(layers):
    order = layers.get("eviction_order")
    if order is None:
        raise ValueError("layers.eviction_order is required (an empty array keeps every layer resident)")
    granularity = layers.get("granularity")
    step = 1 if granularity is None else _count(granularity, "layers.granularity")
    if step < 1:
        raise ValueError("layers.granularity must be at least 1")
    if order == BY_MISS_RATE:
        return BY_MISS_RATE, step
    if type(order) is not list:
        raise ValueError(f"layers.eviction_order must be an array or {BY_MISS_RATE!r}")
    nested = [type(x) is list for x in order]
    if any(nested) and not all(nested):
        raise ValueError("layers.eviction_order mixes layer IDs and groups")
    if all(nested) and order:
        if granularity is not None:
            raise ValueError("layers.granularity only applies to a flat or derived eviction_order")
        groups = tuple(_layer_list(g, "layers.eviction_order[]") for g in order)
    else:
        flat = _layer_list(order, "layers.eviction_order")
        groups = tuple(flat[i:i + step] for i in range(0, len(flat), step))
    if any(not g for g in groups):
        raise ValueError("layers.eviction_order has an empty group")
    return groups, step


def _chunk(flat, step):
    return tuple(tuple(flat[i:i + step]) for i in range(0, len(flat), step))


@dataclass(frozen=True)
class ElasticConfig:
    model: str
    mode: str
    groups: tuple                       # eviction order: groups[0] leaves the GPU first (or BY_MISS_RATE)
    always_offload: tuple               # (or SHARED_INDEX_FOLLOWERS)
    slo: dict | None
    profile: dict
    controller: dict
    memory: dict
    features: dict = field(default_factory=dict)
    pinned_resident: tuple = ()
    granularity: int = 1

    @property
    def resolved(self):
        return type(self.groups) is tuple and type(self.always_offload) is tuple

    def _need_resolved(self):
        if not self.resolved:
            raise ValueError("symbolic layer lists must be resolved against the bound model first")

    @property
    def switchable(self):
        """Layers whose residency can change at runtime, in eviction order."""
        self._need_resolved()
        return tuple(layer for group in self.groups for layer in group)

    @property
    def managed(self):
        """Every layer this config moves or keeps offloaded."""
        return tuple(sorted(self.always_offload + self.switchable))

    @property
    def max_level(self):
        self._need_resolved()
        return min(len(self.groups), self.controller["max_level"])

    def offloaded(self, level):
        """Offloaded layers at ``level``: the always-offloaded ones plus the first ``level`` groups."""
        self._need_resolved()
        if not 0 <= level <= len(self.groups):
            raise ValueError(f"level {level} outside 0..{len(self.groups)}")
        return frozenset(self.always_offload).union(*self.groups[:level])

    def feature(self, name):
        return self.features[name]

    def precapture_levels(self):
        """Steady mode: levels whose graph sets are captured at startup (others are captured on first use)."""
        levels = self.memory["precapture_levels"]
        top = self.max_level
        return list(range(top + 1)) if levels == "all" else [level for level in levels if level <= top]

    def resolve(self, eligible, *, followers=(), miss_rate=None):
        """Concrete layer lists for the bound model. ``eligible`` are the offloadable layers, ``followers`` the
        DSA shared-index layers, ``miss_rate(layer)`` the profiled rate used by BY_MISS_RATE."""
        eligible = tuple(sorted(eligible))
        always = tuple(sorted(followers)) if self.always_offload == SHARED_INDEX_FOLLOWERS else self.always_offload
        groups = self.groups
        if groups == BY_MISS_RATE:
            rest = [layer for layer in eligible if layer not in always and layer not in self.pinned_resident]
            rest.sort(key=lambda layer: (miss_rate(layer) if miss_rate else 0.0, layer))
            groups = _chunk(rest, self.granularity)
        bound = ElasticConfig(self.model, self.mode, groups, always, self.slo, self.profile,
                              dict(self.controller, max_level=min(self.controller["max_level"], len(groups))),
                              self.memory, self.features, self.pinned_resident, self.granularity)
        bound.check_layers(eligible)
        flat = bound.managed
        if len(set(flat)) != len(flat) or set(flat) & set(self.pinned_resident):
            raise ValueError("resolved layer lists overlap each other or pinned_resident")
        if bound.controller["initial_level"] > bound.max_level:
            raise ValueError("controller.initial_level exceeds the resolved eviction order")
        return bound

    def as_dict(self):
        groups = self.groups if type(self.groups) is str else [list(g) for g in self.groups]
        always = self.always_offload if type(self.always_offload) is str else list(self.always_offload)
        return dict(schema=SCHEMA, model=self.model, mode=self.mode,
                    layers=dict(eviction_order=groups, always_offload=always,
                                pinned_resident=list(self.pinned_resident), granularity=self.granularity),
                    slo=self.slo, profile=self.profile, controller=self.controller, memory=self.memory,
                    features=self.features)

    @property
    def digest(self):
        canonical = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()

    def check_layers(self, eligible):
        """Reject layers the bound model cannot offload (``eligible`` comes from the real pools)."""
        bad = sorted(set(self.managed) - set(eligible))
        if bad:
            raise ValueError(f"layers {bad} are not offloadable in this model (eligible: {sorted(eligible)})")


def parse(data):
    if type(data) is not dict:
        raise ValueError("Elastic config must be a JSON object")
    unknown = set(data) - _TOP
    if unknown:
        raise ValueError(f"Unknown top-level field(s): {sorted(unknown)}")
    if data.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA!r}")
    model = data.get("model")
    if model not in MODELS:
        raise ValueError(f"model must be one of {list(MODELS)}")
    mode = _choice(data, "mode", MODES, "config")
    layers = _section(data, "layers", _LAYERS)
    groups, granularity = _groups(layers)
    raw_always = layers.get("always_offload", [])
    if raw_always == SHARED_INDEX_FOLLOWERS:
        if model != "dsa":
            raise ValueError(f"{SHARED_INDEX_FOLLOWERS!r} only exists for DSA models")
        always = SHARED_INDEX_FOLLOWERS
    else:
        always = tuple(sorted(_layer_list(raw_always, "layers.always_offload")))
    pinned = tuple(sorted(set(_layer_list(layers.get("pinned_resident", []), "layers.pinned_resident"))))
    flat = [layer for group in (groups if type(groups) is tuple else ()) for layer in group]
    flat += list(always) if type(always) is tuple else []
    if len(set(flat)) != len(flat):
        raise ValueError("a layer appears twice across eviction_order and always_offload")
    if set(flat) & set(pinned):
        raise ValueError("layers.pinned_resident overlaps the offloaded layers")
    levels = len(groups) if type(groups) is tuple else None

    slo = None
    if data.get("slo") is not None:
        raw = _section(data, "slo", _SLO)
        if "tpot_ms" not in raw:
            raise ValueError("slo.tpot_ms is required when slo is given")
        slo = dict(tpot_ms=_number(raw["tpot_ms"], "slo.tpot_ms", strict=True),
                   tail_factor=_number(raw.get("tail_factor", 1.1), "slo.tail_factor", low=1.0),
                   margin_ms=_number(raw.get("margin_ms", 1.0), "slo.margin_ms"))

    raw = _section(data, "profile", _PROFILE)
    profile = dict(bytes_per_miss=_count(raw.get("bytes_per_miss", 512), "profile.bytes_per_miss"),
                   selected_rows=_count(raw.get("selected_rows", 2048), "profile.selected_rows"),
                   h2d_gbps=_number(raw.get("h2d_gbps", 14.3), "profile.h2d_gbps", strict=True),
                   fixed_us=_number(raw.get("fixed_us", 18.0), "profile.fixed_us"),
                   per_request_us=_number(raw.get("per_request_us", 0.4), "profile.per_request_us"),
                   default_miss_rate=_number(raw.get("default_miss_rate", 0.15), "profile.default_miss_rate",
                                             high=1.0),
                   miss_rate=_miss_rates(raw.get("miss_rate", {})),
                   exposed={str(_count(int(k), "profile.exposed key")): _number(v, "profile.exposed", high=1.0)
                            for k, v in _object(raw.get("exposed", {}), "profile.exposed").items()},
                   base_ms=_base(raw.get("base_ms")),
                   measurements=raw.get("measurements"),
                   explicit=sorted(raw))
    if profile["measurements"] is not None and type(profile["measurements"]) is not str:
        raise ValueError("profile.measurements must be a repo-relative path")

    raw = _section(data, "controller", _CONTROLLER)
    controller = dict(interval_steps=max(1, _count(raw.get("interval_steps", 8), "controller.interval_steps")),
                      high_watermark=_number(raw.get("high_watermark", 0.9), "controller.high_watermark",
                                             strict=True, high=1.0),
                      low_watermark=_number(raw.get("low_watermark", 0.6), "controller.low_watermark", high=1.0),
                      dwell_steps=_count(raw.get("dwell_steps", 512), "controller.dwell_steps"),
                      horizon_steps=_count(raw.get("horizon_steps", 64), "controller.horizon_steps"),
                      backfill_gbps=_number(raw.get("backfill_gbps", 6.0), "controller.backfill_gbps", strict=True),
                      max_level=_count(raw.get("max_level", 10 ** 6 if levels is None else levels),
                                       "controller.max_level"),
                      min_level=_count(raw.get("min_level", 0), "controller.min_level"),
                      initial_level=_count(raw.get("initial_level", 0), "controller.initial_level"),
                      min_gain=_number(raw.get("min_gain", 0.02), "controller.min_gain", high=1.0))
    if controller["low_watermark"] >= controller["high_watermark"]:
        raise ValueError("controller.low_watermark must be below controller.high_watermark")
    top = controller["max_level"] if levels is None else min(controller["max_level"], levels)
    if not controller["min_level"] <= controller["initial_level"] <= top:
        raise ValueError("controller levels must satisfy min_level <= initial_level <= max_level <= groups")

    raw = _section(data, "memory", _MEMORY)
    memory = dict(chunk_mib=_count(raw.get("chunk_mib", 2), "memory.chunk_mib"),
                  hot_rows=_optional_count(raw.get("hot_rows"), "memory.hot_rows"),
                  max_logical_tokens=_optional_count(raw.get("max_logical_tokens"), "memory.max_logical_tokens"),
                  host_kv_gib=(None if raw.get("host_kv_gib") is None else
                               _number(raw["host_kv_gib"], "memory.host_kv_gib", strict=True)),
                  precapture_levels=_precapture(raw.get("precapture_levels", "all"), levels))
    if memory["chunk_mib"] < 2 or memory["chunk_mib"] & (memory["chunk_mib"] - 1):
        raise ValueError("memory.chunk_mib must be a power of two of at least 2 (the CUDA VMM granularity)")

    raw = _object(data.get("features", {}), "features")
    unknown = set(raw) - set(FEATURES)
    if unknown:
        raise ValueError(f"Unknown feature(s): {sorted(unknown)}")
    if any(type(v) is not bool for v in raw.values()):
        raise ValueError("features values must be booleans")
    features = {**FEATURES, **raw}
    if mode == "steady":
        features["stale_hits"] = False   # the stock kernels of the steady sets have no stale-row path
    if slo is None:
        features["slo_admission"] = False
    return ElasticConfig(model, mode, groups, always, slo, profile, controller, memory, features, pinned,
                         granularity)


def _object(value, where):
    if type(value) is not dict:
        raise ValueError(f"{where} must be a JSON object")
    return value


def _optional_count(value, where):
    return None if value is None else _count(value, where)


def _miss_rates(value):
    """{"<context upper bound>": {"<layer>": rate}}; bounds ascend, the last bin covers longer contexts."""
    rates = {}
    for bound, layers in _object(value, "profile.miss_rate").items():
        upper = _count(int(bound), "profile.miss_rate key")
        rates[str(upper)] = {str(_count(int(k), "profile.miss_rate layer")): _number(v, "profile.miss_rate",
                                                                                  high=1.0)
                             for k, v in _object(layers, "profile.miss_rate bin").items()}
    return dict(sorted(rates.items(), key=lambda item: int(item[0])))


def _base(value):
    if value is None:
        return None
    if type(value) is not list or len(value) != 2:
        raise ValueError("profile.base_ms must be [a_ms, b_ms_per_request]")
    return [_number(value[0], "profile.base_ms[0]"), _number(value[1], "profile.base_ms[1]")]


def _precapture(value, levels):
    if value == "all":
        return "all"
    picked = sorted(set(_layer_list(value, "memory.precapture_levels")))
    if levels is not None and any(level > levels for level in picked):
        raise ValueError("memory.precapture_levels names a level beyond the eviction order")
    return picked


def read(path, expected_sha256=None):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Elastic config changed since launch validation")
    return parse(loads(raw)), digest
