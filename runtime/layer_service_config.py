"""Frozen, CPU-readable configuration for the independent layer service.

This describes main-KV pages, not a whole-device memory guarantee. Costs are
explicit hypotheses/calibrations; absent context coverage creates no victim.
Optional capacity_profiles offer executable, mutually exclusive H alternatives
with their own bound cost tables; omitted profiles preserve the single-H path.
No model, torch or SGLang imports are needed to inspect/validate a JSON file.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re

from .layer_hybrid_pool import LayerGroup, LayerLayout
from .request_policy import RequestTypePolicy


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("Duplicate JSON key")
        out[key] = value
    return out


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected.split()):
        raise ValueError(f"Unexpected/missing fields in {name}")


def _int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, positive=False):
    if type(value) not in (int, float) or not isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"Invalid {name}")
    return value


def _string(value, name):
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"Expected nonempty canonical {name}")
    return value


def _hash(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected lowercase SHA-256")
    return value


@dataclass(frozen=True)
class CostBucket:
    switch_us: tuple
    exposed_us: tuple  # cumulative over horizon_steps, NOT per decode step


@dataclass(frozen=True)
class CostRange:
    request_type: str
    min_tokens: int
    max_tokens: int  # inclusive; completed history, not pending decode +1
    ranks: tuple


@dataclass(frozen=True)
class CostModel:
    kind: str  # surrogate or measured; no implied runtime calibration
    unit: str
    horizon_steps: int
    protocol: str
    source_sha256: str
    policy_sha256: str  # costs are valid only for these exact frozen H tables
    ranges: tuple

    def lookup(self, label, rank, completed_tokens):
        for bucket in self.ranges:
            if bucket.request_type == label and bucket.min_tokens <= completed_tokens <= bucket.max_tokens:
                if type(rank) is not int or not 0 <= rank < len(bucket.ranks):
                    raise ValueError("Cost lookup TP rank is outside coverage")
                return bucket.ranks[rank]
        return None


@dataclass(frozen=True)
class LayerServiceConfig:
    model_identity: str
    expected_tp_size: int
    max_context: int
    max_running_requests: int
    logical_token_capacity: int
    layout: LayerLayout
    host_budget_bytes: int
    control_timeout_seconds: float
    mode: str
    type_policy: RequestTypePolicy
    request_types: tuple
    unknown_request_action: str
    cost_model: CostModel
    max_migrations: int
    cooldown_seconds: float
    min_residency_seconds: float
    transfer_weight: float
    prepared_attempts: int
    catalog_diagnostics: object = None
    # Executable alternative targets; each nested config is single-H and has
    # its own policy-bound costs. The outer/native type policy is unchanged.
    capacity_profiles: tuple = ()

    @property
    def layer_count(self):
        return self.layout.layer_count

    def make_spec(self):
        from .layer_service_factory import LayerServiceSpec
        return LayerServiceSpec(self.layout, self.logical_token_capacity, self.max_context,
                                self.host_budget_bytes, self.model_identity)

    def make_memory_pool_config(self):
        from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
        return MemoryPoolConfig(max_total_num_tokens=self.logical_token_capacity,
                                max_running_requests=self.max_running_requests)

    def make_policy(self):
        from .layer_eviction_policy import LayerEvictionPolicy
        domains = self.expected_tp_size * self.layer_count
        # Demand-based admission uses actual next-forward deficits; these
        # structural watermarks are not an extra background eviction policy.
        return LayerEvictionPolicy(trigger_free_bytes=(0,) * domains,
            target_free_bytes=(1,) * domains, max_migrations=self.max_migrations,
            cooldown=self.cooldown_seconds, min_residency=self.min_residency_seconds,
            transfer_weight=self.transfer_weight)

    def make_proposals(self, runtime, scheduler):
        from .layer_service_proposals import build_proposal_provider
        return build_proposal_provider(self, runtime, scheduler)

    def make_catalog_callbacks(self, runtime, scheduler, config_sha256):
        if self.catalog_diagnostics is None:
            return None, None
        from .layer_catalog_probe import build_catalog_callbacks
        return build_catalog_callbacks(self.catalog_diagnostics, runtime, scheduler, config_sha256)

    def summary(self):
        """Public run metadata; no raw request IDs or private request text."""
        return dict(schema="layer-hisparse-service-v1", model_identity=self.model_identity,
            tp=self.expected_tp_size, mode=self.mode, main_budget_bytes_per_rank=self.layout.budget_bytes,
            host_budget_bytes_per_rank=self.host_budget_bytes, logical_tokens=self.logical_token_capacity,
            max_context=self.max_context, max_running_requests=self.max_running_requests,
            layout_sha256=self.layout.digest, type_policy_sha256=self.type_policy.digest,
            request_manifest_count=len(self.request_types), cost_kind=self.cost_model.kind,
            cost_unit=self.cost_model.unit, cost_horizon_steps=self.cost_model.horizon_steps,
            cost_source_sha256=self.cost_model.source_sha256, cost_ranges=len(self.cost_model.ranges),
            capacity_profile_count=len(self.capacity_profiles),
            multi_h_candidates_enabled=bool(self.capacity_profiles),
            capacity_mode="explicit_profile_options" if self.capacity_profiles else "single_frozen_target",
            online_selector="none" if self.mode == "resident_only" else "time_cost_per_useful_byte_greedy",
            time_aggregation="rank_max_additive_surrogate",
            capacity_profile_policies=tuple(c.type_policy.digest for _,c in self.capacity_profiles),
            capacity_profile_cost_sources=tuple(c.cost_model.source_sha256 for _,c in self.capacity_profiles),
            catalog_diagnostics_enabled=self.catalog_diagnostics is not None,
            measurement_role="diagnostic_only" if self.catalog_diagnostics else "uninstrumented")


def parse_service_config(value):
    diagnostic_value = None
    capacity_value = None
    if type(value) is dict and "capacity_profiles" in value:
        value = dict(value)
        capacity_value = value.pop("capacity_profiles")
        if type(capacity_value) is not list or not 1 <= len(capacity_value) <= 16:
            raise ValueError("Omit capacity_profiles for single H; otherwise supply 1..16 executable profiles")
    if type(value) is dict and "catalog_diagnostics" in value:
        value = dict(value)
        diagnostic_value = value.pop("catalog_diagnostics")
        if type(diagnostic_value) is not dict:
            raise ValueError("Omit catalog_diagnostics to disable it; enabled configuration must be an object")
    _keys(value, "schema model_identity expected_tp_size max_context max_running_requests logical_token_capacity "
        "main host_budget_bytes control_timeout_seconds mode type_policy request_types unknown_request_action "
        "cost_model eviction", "service")
    if value["schema"] != "layer-hisparse-service-v1":
        raise ValueError("Unknown layer service schema")
    identity = _string(value["model_identity"], "model/revision identity")
    tp = _int(value["expected_tp_size"], "TP")
    if tp not in (4, 8):
        raise ValueError("Only explicit TP4/TP8 MiniMax service supported")
    context = _int(value["max_context"], "context")
    running = _int(value["max_running_requests"], "request count")
    logical = _int(value["logical_token_capacity"], "logical capacity")
    if context >= 2**31 or logical % 128 or logical + 128 >= 2**31 or logical < context:
        raise ValueError("Require aligned int32 logical capacity covering one full context")
    main = value["main"]
    _keys(main, "layer_count page_tokens kv_bytes_per_token_per_layer topk_blocks block_tokens allocatable_pages budget_bytes", "main")
    if main["layer_count"] != 60 or type(main["layer_count"]) is not int or main["page_tokens"] != 128:
        raise ValueError("Pinned MiniMax requires 60 layers and page size 128")
    pages = main["allocatable_pages"]
    if type(pages) is not list or len(pages) != 60:
        raise ValueError("Explicit allocatable pages for every layer required")
    layout = LayerLayout(60, running + 1, main["page_tokens"], main["kv_bytes_per_token_per_layer"],
        main["topk_blocks"], main["block_tokens"],
        tuple(LayerGroup(f"layer_{i}", (i,), "resident", n) for i, n in enumerate(pages)), main["budget_bytes"])
    host = _int(value["host_budget_bytes"], "host budget", 0)
    timeout = _number(value["control_timeout_seconds"], "control timeout", positive=True)
    if value["mode"] not in ("resident_only", "pressure"):
        raise ValueError("Expected resident_only or pressure mode")
    p = value["type_policy"]
    _keys(p, "hmax default_type tables", "type_policy")
    if type(p["tables"]) is not dict:
        raise ValueError("Type tables must be a JSON object")
    if any(type(caps) is not list for caps in p["tables"].values()):
        raise ValueError("Each type table must be an array")
    policy = RequestTypePolicy(p["hmax"], tuple(sorted((k, tuple(v)) for k,v in p["tables"].items())),
        p["default_type"], tp, layout.topk_blocks * layout.block_tokens)
    for _, caps in policy.tables:
        if any(cap // 128 + 1 > pages[layer] for layer, cap in enumerate(caps)):
            raise ValueError("Every target hot row plus latest-token page must fit its own layer pool")
    manifest = value["request_types"]
    if type(manifest) is not dict:
        raise ValueError("request_types must be a frozen RID -> type object")
    labels = dict(policy.tables)
    for rid, label in manifest.items():
        _string(rid, "request ID")
        if type(label) is not str or label not in labels:
            raise ValueError("Request manifest refers to an unknown type")
    if value["unknown_request_action"] not in ("reject", "default"):
        raise ValueError("Declare unknown request handling")
    if value["mode"] == "pressure" and value["unknown_request_action"] != "default":
        raise ValueError("Serving pressure mode requires default handling: native warmup has unlisted IDs; "
                         "request-level ingress rejection is not implemented")
    c = value["cost_model"]
    _keys(c, "kind unit horizon_steps protocol source_sha256 policy_sha256 ranges", "cost_model")
    if c["kind"] not in ("surrogate", "measured") or c["unit"] != "us":
        raise ValueError("Costs must explicitly be surrogate/measured in microseconds")
    horizon = _int(c["horizon_steps"], "cost horizon")
    protocol = _string(c["protocol"], "cost protocol")
    source_hash = _hash(c["source_sha256"])
    policy_hash = _hash(c["policy_sha256"])
    if policy_hash != policy.digest:
        raise ValueError("Cost policy_sha256 does not match the frozen type policy")
    if type(c["ranges"]) is not list:
        raise ValueError("Expected explicit cost coverage ranges")
    ranges = []
    for b in c["ranges"]:
        _keys(b, "request_type min_tokens max_tokens ranks", "cost range")
        if type(b["request_type"]) is not str or b["request_type"] not in labels:
            raise ValueError("Cost range type is unknown")
        low = _int(b["min_tokens"], "minimum completed tokens", 0)
        high = _int(b["max_tokens"], "maximum completed tokens", 0)
        if not low <= high <= context:
            raise ValueError("Cost range must be inside the context limit")
        if type(b["ranks"]) is not list or len(b["ranks"]) != tp:
            raise ValueError("Every cost range must cover all TP ranks")
        costs = []
        for r in b["ranks"]:
            _keys(r, "switch_us exposed_us", "rank cost")
            for v in r.values():
                if type(v) is not list or len(v) != 60:
                    raise ValueError("Cost vectors must cover all 60 layers")
                for n in v:
                    _number(n, "layer cost")
            costs.append(CostBucket(tuple(r["switch_us"]), tuple(r["exposed_us"])))
        ranges.append(CostRange(b["request_type"], low, high, tuple(costs)))
    ranges.sort(key=lambda b:(b.request_type,b.min_tokens))
    if any(a.request_type == b.request_type and a.max_tokens >= b.min_tokens for a,b in zip(ranges,ranges[1:])):
        raise ValueError("Cost ranges must not overlap")
    costs = CostModel(c["kind"], c["unit"], horizon, protocol, source_hash, policy_hash, tuple(ranges))
    if value["mode"] == "pressure" and (not ranges or host < context * layout.kv_bytes_per_token_per_layer):
        raise ValueError("Pressure mode needs explicit cost coverage and host space for one whole layer history")
    e = value["eviction"]
    _keys(e, "max_migrations cooldown_seconds min_residency_seconds transfer_weight prepared_attempts", "eviction")
    diagnostics = None
    if diagnostic_value is not None:
        from .layer_catalog_probe import parse_catalog_diagnostics
        # value has neither optional field: diagnostic rounds stay single-H,
        # even when the serving policy offers executable multi-H alternatives.
        diagnostics = parse_catalog_diagnostics(diagnostic_value, value)
    capacity_profiles = () if capacity_value is None else _parse_capacity_profiles(
        capacity_value, value, policy, costs)
    return LayerServiceConfig(identity, tp, context, running, logical, layout, host, timeout, value["mode"],
        policy, tuple(sorted(manifest.items())), value["unknown_request_action"], costs,
        _int(e["max_migrations"], "migration bound"), _number(e["cooldown_seconds"], "cooldown"),
        _number(e["min_residency_seconds"], "minimum residency"), _number(e["transfer_weight"], "transfer weight"),
        _int(e["prepared_attempts"], "prepared attempt bound"), diagnostics, capacity_profiles)


def _parse_capacity_profiles(profiles, base_service, base_policy, base_costs):
    """Reuse all service constraints; do not overwrite the native/base H table.

    The explicit profiles are the whole candidate set (base is not implicitly
    added). Comparable ranges/protocols avoid missing-H evidence advantages.
    Identical (dispatch type, layer, H, context) costs may be deduplicated by the
    provider; contradictory aliases are rejected for every rank at startup.
    """
    parsed, names, duplicate_costs = [], set(), {}
    scope = lambda costs: (costs.kind, costs.unit, costs.horizon_steps, costs.protocol,
        tuple((b.request_type,b.min_tokens,b.max_tokens) for b in costs.ranges))
    labels = {label for label,_ in base_policy.tables}
    for profile in profiles:
        _keys(profile, "name type_policy cost_model", "capacity profile")
        name = profile["name"]
        if type(name) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", name):
            raise ValueError("Capacity profile names must be short opaque labels")
        if name in names:
            raise ValueError("Duplicate capacity profile name")
        names.add(name)
        candidate = parse_service_config(dict(base_service, mode="pressure",
            type_policy=profile["type_policy"], cost_model=profile["cost_model"]))
        if candidate.type_policy.hmax != base_policy.hmax:
            raise ValueError("Capacity profiles must preserve the base policy Hmax")
        if ({label for label,_ in candidate.type_policy.tables} != labels
                or candidate.type_policy.default_type != base_policy.default_type):
            raise ValueError("Capacity profiles must preserve request labels and default dispatch")
        if scope(candidate.cost_model) != scope(base_costs):
            raise ValueError("Capacity profile costs must share kind/unit/horizon/protocol/coverage ranges")
        for bucket in candidate.cost_model.ranges:
            _, capacities = candidate.type_policy.resolve(bucket.request_type)
            for layer, hot in enumerate(capacities):
                key = bucket.request_type, bucket.min_tokens, bucket.max_tokens, layer, hot
                costs = tuple((rank.switch_us[layer],rank.exposed_us[layer]) for rank in bucket.ranks)
                if key in duplicate_costs and duplicate_costs[key] != costs:
                    raise ValueError("Conflicting costs for duplicate capacity profile type/layer/H")
                duplicate_costs[key] = costs
        parsed.append((name,candidate))
    return tuple(parsed)


def read_service_config(path, expected_sha256=None):
    data = Path(path).read_bytes()
    digest = sha256(data).hexdigest()
    if expected_sha256 is not None and digest != _hash(expected_sha256):
        raise ValueError("Layer service configuration changed since launch")
    def invalid_constant(_):
        raise ValueError("Non-finite JSON numbers are forbidden")
    value = json.loads(data.decode("utf-8"), object_pairs_hook=_object, parse_constant=invalid_constant)
    return parse_service_config(value), digest
