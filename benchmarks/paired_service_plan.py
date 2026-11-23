"""Compile pinned paired-run metadata and argv; never launch a process/GPU.

Closed-loop smoke and frozen open-loop arrivals with the existing cost/useful-
byte selector have command templates. Fixed-batch interventions and alternative
online selectors remain explicit pending items. A compiled command is NOT a resource
grant, a correctness certificate, actual fixed batching or a performance result.
Input bundles and command plans stay in the private experiment workspace.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from analysis.layer_service_costs import _json, _read_pin
from runtime.layer_service_config import _hash, _int, _keys, _number, parse_service_config
from runtime.layer_service_launch import check_upstream
from runtime.launch import flag_value
from .artifact_paths import _named_path, is_private_path
from .sglang_arrival_client import read_schedule, read_workload


SCHEMA = "paired-layer-service-plan-v1"
STAGES = ("closed_loop_smoke", "fixed_batch_intervention", "open_loop_admission")
SUPPORTED = ("resident_reference", "uniform_target", "layer_target", "type_layer_target", "multih_time_surrogate")
PENDING = ("miss_marginal", "whole_plan_time", "whole_request")


def _label(value):
    if (type(value) is not str or not value or len(value) > 96
            or not value[0].isalnum() or any(not (c.isascii() and (c.isalnum() or c in "_.-")) for c in value)):
        raise ValueError("Expected a short opaque plan label")
    return value


def _directory(value):
    if type(value) is not str or not value.startswith("/") or "\\" in value or "\x00" in value:
        raise ValueError("Use a canonical absolute container directory")
    path = PurePosixPath(value)
    if ".." in path.parts or is_private_path(path) or str(path) != value:
        raise ValueError("Use a canonical absolute container directory")
    return path




def _ref_path(root, ref):
    _keys(ref, "path sha256", "artifact reference")
    name = ref["path"]
    if (type(name) is not str or not name or "\\" in name or ":" in name
            or "\x00" in name):
        raise ValueError("Artifact reference must stay inside the named input bundle")
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name or ".." in path.parts or is_private_path(path):
        raise ValueError("Artifact reference must stay inside the named input bundle")
    raw = _read_pin(_named_path(root / name), ref["sha256"])
    return raw, ref["path"]


def native_generation_budget(config, input_tokens):
    """Pinned eager TP/no-DCP scheduler plus custom allocator's static bounds.

    This is a preflight bound, not a guarantee the runtime can admit or complete
    the request. Effective native values/environment must still be checked.
    """
    _int(input_tokens, "prompt token count")
    page = config.layout.page_tokens
    max_req_len = min(config.max_context - 1, config.logical_token_capacity - 1)
    if input_tokens >= max_req_len - 5:
        raise ValueError("Prompt must be below the pinned native input ceiling")
    paged = (input_tokens + page - 1) // page * page
    if paged > min(g.allocatable_pages for g in config.layout.groups) * page:
        raise ValueError("Prompt exceeds an empty resident layer pool")
    budget = min(max_req_len - input_tokens - 1,
                 config.logical_token_capacity - paged - page - 1,
                 config.max_context - input_tokens - 1)
    if budget < 1:
        raise ValueError("Native/static logical budget leaves no generation capacity")
    return budget


def _workload(root, reference, config):
    raw, name = _ref_path(root, reference)
    requests, aliases = [], set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = _json(line)
        _keys(row, "request_alias workload input_ids max_new_tokens", "closed-loop workload row")
        alias, label = _label(row["request_alias"]), _label(row["workload"])
        tokens = row["input_ids"]
        if alias in aliases or type(tokens) is not list or not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("Workload has duplicate identity or invalid tokens")
        aliases.add(alias)
        budget = _int(row["max_new_tokens"], "generation capacity")
        if budget != native_generation_budget(config, len(tokens)):
            raise ValueError("Natural-EOS study must allow the full native generation budget; length stops stay censored")
        requested_type = dict(config.request_types).get(alias)
        assigned, capacities = config.type_policy.resolve(requested_type)
        if assigned != label:
            raise ValueError("Workload type must match the frozen request/default dispatch")
        if config.mode == "pressure":
            # Do not plan a mechanism comparison where an entire type silently
            # has no candidates because its cost table is absent. Check all
            # potentially reclaimable completed lengths, not only prompt length.
            page = config.layout.page_tokens
            tables = ([profile.type_policy.resolve(requested_type)[1] for _, profile in config.capacity_profiles]
                      if config.capacity_profiles else [capacities])
            cursor = max(len(tokens), min((h // page + 1) * page + 1 for table in tables for h in table))
            upper = len(tokens) + budget - 1
            for bucket in config.cost_model.ranges:
                if bucket.request_type != label or bucket.max_tokens < cursor:
                    continue
                if bucket.min_tokens > cursor:
                    break
                cursor = bucket.max_tokens + 1
            if cursor <= upper:
                raise ValueError("Workload has potentially reclaimable history outside frozen cost coverage")
        requests.append(row)
    if not requests:
        raise ValueError("Workload must be nonempty")
    return dict(path=name, sha256=reference["sha256"], requests=len(requests),
                input_tokens=sum(len(r["input_ids"]) for r in requests))


def _cost_scope(config):
    c = config.cost_model
    return (c.kind, c.unit, c.horizon_steps, c.protocol,
            tuple((r.request_type, r.min_tokens, r.max_tokens) for r in c.ranges))


def _common_config(value):
    # Mode differs only for the explicitly labelled resident reference. Costs
    # and target H are the declared intervention. Every other field is fixed.
    return {k:v for k,v in value.items() if k not in ("mode", "type_policy", "cost_model", "capacity_profiles")}


def _check_arm(kind, config):
    if config.catalog_diagnostics is not None:
        raise ValueError("Paired service runs must disable catalog diagnostics")
    if config.mode != ("resident_only" if kind == "resident_reference" else "pressure"):
        raise ValueError("Arm label disagrees with service mode")
    if kind == "multih_time_surrogate":
        if not config.capacity_profiles:
            raise ValueError("Multi-H arm requires explicit executable capacity profiles")
    elif config.capacity_profiles:
        raise ValueError("Fixed-H/reference/pending arms cannot silently enable executable multi-H profiles")
    caps = tuple(table for _, table in config.type_policy.tables)
    if kind == "uniform_target" and len({h for table in caps for h in table}) != 1:
        raise ValueError("Uniform target requires one H across request types and layers")
    if kind == "layer_target" and len(set(caps)) != 1:
        raise ValueError("Layer-only target must use the same layer table for every type")


def compile_plan(manifest_path, expected_sha256):
    """Validate all pinned inputs; return anonymous metadata and argv templates.

    Execution paths refer to the same input bundle mounted at execution_root.
    The caller must preserve/verify those files before launch. Repetitions use
    balanced cyclic order: each arm occupies each position once per N blocks.
    Seed is the native model seed, not a promise of cross-hardware determinism.
    """
    path = _named_path(manifest_path)
    root = path.parent
    spec = _json(_read_pin(path, expected_sha256))
    _keys(spec, "schema stage runtime_sha256 execution_root output_root upstream endpoint "
          "sm80_bf16_projections repetitions seed client workload warmup arrival_schedule arms", "paired plan")
    if spec["schema"] != SCHEMA or spec["stage"] not in STAGES:
        raise ValueError("Unknown plan schema/stage")
    _hash(spec["runtime_sha256"])
    execution, output = _directory(spec["execution_root"]), _directory(spec["output_root"])
    if output == execution or execution.is_relative_to(output) or output.is_relative_to(execution):
        raise ValueError("Input bundle and new run output roots must not overlap")
    if type(spec["sm80_bf16_projections"]) is not bool:
        raise ValueError("Explicit SM80 adapter flag required")
    repeats, seed = _int(spec["repetitions"], "repetitions"), _int(spec["seed"], "model seed", 0)
    if repeats > 100 or seed >= 2**32:
        raise ValueError("Bound repeats to 100 and model seed to uint32")
    argv = spec["upstream"]
    if type(argv) is not list or not argv or any(type(a) is not str or not a or "\x00" in a for a in argv):
        raise ValueError("Native arguments must be an explicit argv array")
    if flag_value(argv, "--random-seed") != str(seed):
        raise ValueError("Declared model seed must equal explicit --random-seed")
    endpoint = urlsplit(spec["endpoint"])
    port = flag_value(argv, "--port")
    if (endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1" or endpoint.path != "/generate"
            or endpoint.query or endpoint.fragment or endpoint.username or endpoint.password
            or str(endpoint.port) != port or flag_value(argv, "--host") != "127.0.0.1"):
        raise ValueError("Client endpoint must match the explicit loopback native host/port")
    client = spec["client"]
    _keys(client, "concurrency timeout_s ttft_slo_s tpot_slo_s", "client")
    _int(client["concurrency"], "client concurrency")
    for name in ("timeout_s", "ttft_slo_s", "tpot_slo_s"):
        _number(client[name], name, positive=True)
    if spec["stage"] != "open_loop_admission" and spec["arrival_schedule"] is not None:
        raise ValueError("This stage cannot silently consume an ignored arrival schedule")
    arrivals = None
    if spec["stage"] == "open_loop_admission":
        raw, name = _ref_path(root, spec["arrival_schedule"])
        work_raw, _ = _ref_path(root, spec["workload"])
        schedule = read_schedule(raw, read_workload(work_raw), spec["workload"]["sha256"])
        arrivals = dict(path=name, sha256=sha256(raw).hexdigest(), validation="complete_alias_hash_phase_binding",
                        phases=schedule["phases"], requests=len(schedule["arrivals"]))
    arms = spec["arms"]
    if type(arms) is not list or not 2 <= len(arms) <= 16:
        raise ValueError("A paired plan needs 2..16 arms")
    names, prepared, common, dispatch, scope = set(), [], None, None, None
    work = warm = None
    for arm in arms:
        _keys(arm, "name kind service_config", "arm")
        name, kind = _label(arm["name"]), arm["kind"]
        if name in names or kind not in SUPPORTED + PENDING:
            raise ValueError("Duplicate arm name or unknown selector label")
        names.add(name)
        raw, config_path = _ref_path(root, arm["service_config"])
        value = _json(raw)
        config = parse_service_config(value)
        _check_arm(kind, config)
        check_upstream(argv, config)
        current = _common_config(value)
        table_shape = (config.type_policy.hmax, config.type_policy.default_type,
                       tuple(label for label, _ in config.type_policy.tables))
        if common is None:
            common, dispatch = current, table_shape
        elif current != common or table_shape != dispatch:
            raise ValueError("Arms change undeclared geometry/budget/dispatch/eviction or Hmax")
        if kind != "resident_reference":
            if scope is None:
                scope = _cost_scope(config)
            elif _cost_scope(config) != scope:
                raise ValueError("Cost units/protocol/horizon/coverage must match across pressure arms")
        w, u = _workload(root, spec["workload"], config), _workload(root, spec["warmup"], config)
        if work is None:
            work, warm = w, u
        elif (work, warm) != (w, u):
            raise ValueError("Workload or warmup differs across paired arms")
        reasons = []
        if spec["stage"] == "fixed_batch_intervention":
            reasons.append("stage_driver_not_implemented:" + spec["stage"])
        if kind in PENDING:
            reasons.append("online_selector_not_implemented:" + kind)
        prepared.append(dict(name=name, kind=kind, service_config_path=config_path,
            service_config_sha256=arm["service_config"]["sha256"], policy_sha256=config.type_policy.digest,
            cost_kind=config.cost_model.kind, selector=("none" if kind == "resident_reference" else
                "multih_time_useful_bytes_greedy" if kind == "multih_time_surrogate" else
                "frozen_cost_useful_bytes" if kind in SUPPORTED else "pending"),
            capacity_profiles=[dict(name=n, policy_sha256=c.type_policy.digest,
                cost_source_sha256=c.cost_model.source_sha256) for n,c in config.capacity_profiles],
            pending_reasons=reasons))
    jobs = []
    for repetition in range(repeats):
        shift = repetition % len(prepared)
        for position, arm in enumerate(prepared[shift:] + prepared[:shift]):
            directory = output / f"repeat-{repetition:03d}" / arm["name"]
            commands = None
            if not arm["pending_reasons"]:
                server = ["python3", "-B", "-m", "runtime.layer_service_launch", "--service-config",
                          str(execution / arm["service_config_path"])]
                if spec["sm80_bf16_projections"]:
                    server.append("--sm80-bf16-projections")
                def client_args(source, filename):
                    return ["python3", "-B", "-m", "benchmarks.sglang_stream_client", "--url", spec["endpoint"],
                        "--workload", str(execution/source["path"]), "--output", str(directory/filename),
                        "--concurrency", str(client["concurrency"]), "--timeout", str(client["timeout_s"]),
                        "--ttft-slo-s", str(client["ttft_slo_s"]), "--tpot-slo-s", str(client["tpot_slo_s"])]
                commands = dict(server=server+["--"]+argv, warmup=client_args(warm,"warmup.json"),
                                measurement=client_args(work,"measurement.json"))
                if spec["stage"] == "open_loop_admission":
                    command = commands["measurement"]
                    command[command.index("benchmarks.sglang_stream_client")] = "benchmarks.sglang_arrival_client"
                    command[command.index("--concurrency")] = "--max-workers"
                    command.extend(["--schedule", str(execution/arrivals["path"])])
            jobs.append(dict(repetition=repetition, position=position, arm=arm["name"],
                             output_directory=str(directory), commands=commands))
    return dict(schema=SCHEMA, plan_sha256=expected_sha256, stage=spec["stage"], runtime_sha256=spec["runtime_sha256"],
        runtime_identity_scope="caller-declared frozen runtime; live service must attest separately",
        model_identity=config.model_identity, workload=work, warmup=warm, arrival_schedule=arrivals,
        client=client, seed=seed, arms=prepared, jobs=jobs,
        command_templates_ready=all(not a["pending_reasons"] for a in prepared),
        input_bundle_root=str(execution), physical_main_bytes_per_rank=config.layout.budget_bytes,
        logical_token_capacity=config.logical_token_capacity, host_budget_bytes_per_rank=config.host_budget_bytes,
        actual_gpu_batch="unverified; client concurrency is not GPU batch size",
        gpu_gate="No GPU launched. Eight A100 idle continuously >=1200s with <=60s sample gaps and fresh contention check before GPU use.",
        independent_correctness_required=True, natural_eos=True, length_stops="censored; not complete natural-EOS evidence",
        native_length_contract="pinned max_req_len=min(context-1,logical-1); reserve final slot plus allocator page headroom; prompt conservatively below max_req_len-5",
        required_execution_checks=["Effective model context and logical/index capacities equal frozen configuration",
            "SGLANG_MAX_NEW_TOKENS_LIMIT unset/nonpositive and auto-truncation disabled",
            "Actual request budget equals submitted native_generation_budget; preserve all length stops as censored",
            "No external profiler/diagnostic/allocator overlay; model mounts read-only; actual GPU batch recorded"],
        order="prespecified cyclic rotation; repetitions not divisible by arm count have incomplete position balance",
        lifecycle="one self-owned service at a time; explicit readiness/correctness/warmup/drain/teardown required; not automated here")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-command-templates", action="store_true")
    args = parser.parse_args(argv)
    report = compile_plan(args.manifest, args.sha256)
    if args.require_command_templates and not report["command_templates_ready"]:
        parser.error("Requested stage/online selector is pending; no runnable command templates")
    output = _named_path(args.output, writing=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
