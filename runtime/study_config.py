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
# implementation and off/on parity test; unregistered names are rejected.
FEATURES = ()

_TOP = {"schema", "placement", "hot_pool", "features", "diagnostics"}
_PLACEMENT = {"mode", "selection", "offload_layers", "seed", "calibration"}
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

    def as_dict(self):
        out = dict(mode=self.mode, selection=self.selection, offload_layers=list(self.offload_layers))
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
        """Frozen S: listed layers are host-backed with a hot pool, the rest keep full GPU main KV."""
        return self.placement.mode == "fixed_layers"


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
    if mode == "fixed_layers" and "offload_layers" not in raw:
        raise ValueError("placement.offload_layers is required for fixed_layers "
                         "(an empty list keeps every layer resident)")
    if mode != "fixed_layers" and offload:
        raise ValueError("placement.offload_layers applies only to placement.mode=fixed_layers")
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
    placement = Placement(mode, selection, offload, seed, calibration)

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
    features = tuple(sorted(raw.items()))

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
    placement = config.placement
    if placement.mode == "full_resident":
        reasons.append("placement.mode=full_resident is the all-resident control; the pinned SGLang serves it "
                       "without --enable-hisparse and this build does not route it")
    if placement.mode == "fixed_layers" and not placement.offload_layers:
        reasons.append("placement.offload_layers=[] keeps every layer resident; the SGLang HiSparse runtime needs "
                       "at least one host-backed layer (run the all-resident control without --enable-hisparse)")
    enabled = [name for name, on in config.features if on]
    if enabled:
        reasons.append(f"feature switch(es) {enabled} have no implementation in this build")
    if config.hot_pool.layout == "native" and config.diagnostics != "none":
        reasons.append("diagnostics counters/nsys instrument the arena resolver; stock layout runs unmodified")
    return reasons


def capabilities(config):
    """Effective implementation selected for a parsed configuration (startup receipt)."""
    pool, fixed = config.hot_pool, config.uses_fixed_layers
    runtime = "stock_hisparse" if not config.uses_arena else "shared_lru_arena"
    return dict(
        runtime=f"fixed_layers+{runtime}" if fixed else runtime,
        experimental_gpu_state=config.uses_arena or fixed,
        placement=config.placement.mode,
        offload_layers=list(config.placement.offload_layers) if fixed else None,
        host_backing="offloaded layers only" if fixed else "every host-backed sparse layer",
        resident_path=("native sparse attention over full GPU main KV through req_to_token "
                       "(no HiSparse slots or location mapping)") if fixed else None,
        resolver=None if not config.uses_arena else "rlh_shared_lru::resolve (CUDA C++, sm_80)",
        consumer_base="per-layer main KV plane" if not config.uses_arena else (
            "contiguous main KV view of the offloaded layers" if fixed else "contiguous all-layer main KV view"),
        borrowing=pool.mode == "shared",
        borrowing_domain=None if pool.mode != "shared" else (
            "offloaded layers of one request on one TP rank" if fixed
            else "host-backed layers of one request on one TP rank"),
        eviction=pool.eviction,
        quota=pool.quota,
        admission=pool.admission,
        diagnostics=config.diagnostics,
        registered_features=list(FEATURES),
        unsupported=unsupported_reasons(config),
    )
