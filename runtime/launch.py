"""Explicit, spawn-safe launcher for the pinned HiSparse runtime experiment."""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import sys

from .policy import StaticPolicy, validate_capacity


SPAWN_ENV = "_REQUEST_LAYER_HISPARSE_LAUNCH"
SM80_ENV = "EKV_MINIMAX_SM80_BF16"  # Required by the unchanged vendored adapter.
SM80_SHA256 = "207df72e8294982f5153a236043871f38a26b37338c543d9c8322fd56d0c2bf0"
SPLIT_COPY_MODE = "static_layer_split_copy"
SPLIT_DECODE_GRAPH = {"decode": {"backend": "full", "bs": [1, 2]}, "prefill": {"backend": "disabled"}}


@dataclass(frozen=True)
class LaunchPlan:
    tp_size: int
    hmax: int
    top_k: int
    sm80_bf16_projections: bool
    policy_path: str | None
    policy_sha256: str | None
    shadow_identity_directory: str | None = None
    selection_oracle_directory: str | None = None
    split_copy_config: str | None = None
    six_layer_timing: bool = False
    six_layer_state: bool = False
    request_config_path: str | None = None
    request_config_sha256: str | None = None
    study_config_path: str | None = None
    study_config_sha256: str | None = None
    elastic_config_path: str | None = None
    elastic_config_sha256: str | None = None


def strict_object(pairs):
    data = dict(pairs)
    if len(data) != len(pairs):
        raise ValueError("Duplicate JSON object key")
    return data


def parse_object(raw):
    data = json.loads(raw, object_pairs_hook=strict_object)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data


def read_policy(path, expected_sha256=None):
    raw = Path(path).read_bytes()
    digest = sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Policy file SHA-256 changed since launch validation")
    values = parse_object(raw)
    if (not {"hmax", "capacities"} <= values.keys()
            or values.keys() - {"hmax", "capacities", "expected_tp_size", "model", "hardware"}
            or not isinstance(values["capacities"], list)):
        raise ValueError("Policy JSON must use the StaticPolicy fields and a capacities array")
    values["capacities"] = tuple(values["capacities"])
    return StaticPolicy(**values), digest


def flag_value(argv, *names):
    values = []
    for index, token in enumerate(argv):
        flag, delimiter, value = token.partition("=")
        if flag not in names:
            continue
        if not delimiter:
            value = argv[index + 1] if index + 1 < len(argv) else ""
        if not value or value.startswith("--"):
            raise ValueError(f"{flag} requires a value")
        values.append(value)
    if len(values) > 1:
        raise ValueError(f"Repeated upstream option: {names[0]}")
    return values[0] if values else None


def canonical_split_config(config):
    return json.dumps(config.as_dict(), sort_keys=True, separators=(",", ":"))


def read_split_copy_config(path):
    from .split_copy import SplitCopyConfig

    return SplitCopyConfig.from_json(parse_object(Path(path).read_bytes()))


def check_split_execution(argv):
    flags = {arg.partition("=")[0] for arg in argv}
    unsafe = {"--config", "--enable-two-batch-overlap", "--enable-single-batch-overlap", "--enable-pdmux"}
    if "--disable-overlap-schedule" not in argv or flags & unsafe:
        raise ValueError("Split copy requires explicit no-overlap execution and no config overrides")
    if "--disable-radix-cache" not in argv or flag_value(argv, "--disaggregation-mode") not in (None, "null"):
        raise ValueError("Split copy requires Radix and disaggregation disabled")
    if flag_value(argv, "--page-size") != "128" or flag_value(argv, "--attention-backend") != "triton":
        raise ValueError("Split copy requires the pinned Triton attention backend and page size 128")
    if flag_value(argv, "--speculative-algorithm") not in (None, "NONE"):
        raise ValueError("Split copy requires non-speculative decode")
    eager = {"--disable-cuda-graph", "--disable-prefill-cuda-graph"}
    graph_flags = {flag for flag in flags if flag.startswith("--cuda-graph-")}
    if eager.issubset(argv) and not graph_flags:
        return
    raw_graph = flag_value(argv, "--cuda-graph-config")
    if (flags & eager or graph_flags != {"--cuda-graph-config"} or raw_graph is None
            or json.dumps(parse_object(raw_graph), sort_keys=True) != json.dumps(SPLIT_DECODE_GRAPH, sort_keys=True)):
        raise ValueError("Split copy requires explicit eager or decode graph buckets [1, 2] with eager prefill")
    max_running = flag_value(argv, "--max-running-requests")
    if max_running is None or int(max_running) < 2:
        raise ValueError("Split copy decode graph requires a running request limit of at least 2")


def check_request_execution(argv):
    """The mixed-type hook is an explicitly eager, single-scheduler experiment."""
    flags = {arg.partition('=')[0] for arg in argv}
    required = {'--disable-cuda-graph', '--disable-prefill-cuda-graph',
                '--disable-overlap-schedule', '--disable-radix-cache'}
    forbidden = {'--config', '--enable-two-batch-overlap', '--enable-single-batch-overlap', '--enable-pdmux'}
    if (not required.issubset(argv) or flags & forbidden
            or any(flag.startswith('--cuda-graph-') for flag in flags)):
        raise ValueError('Request-type routing requires explicit eager/no-overlap/no-Radix execution')
    if (flag_value(argv, '--page-size') != '128'
            or flag_value(argv, '--attention-backend') != 'triton'
            or flag_value(argv, '--disaggregation-mode') not in (None, 'null')
            or flag_value(argv, '--speculative-algorithm') not in (None, 'NONE')):
        raise ValueError('Request-type routing requires pinned Triton/page128 without disaggregation/spec')


def check_study_execution(argv):
    """Arena arms decode on one stream: no overlap, Radix, speculation or disaggregation.

    Eager or full decode CUDA Graph are both allowed; the worker re-checks the
    resolved runtime context before allocating anything.
    """
    flags = {arg.partition("=")[0] for arg in argv}
    forbidden = {"--config", "--enable-two-batch-overlap", "--enable-single-batch-overlap", "--enable-pdmux"}
    if not {"--disable-overlap-schedule", "--disable-radix-cache"}.issubset(argv) or flags & forbidden:
        raise ValueError("Shared-arena arms require --disable-overlap-schedule, --disable-radix-cache "
                         "and no config/overlap overrides")
    if (flag_value(argv, "--disaggregation-mode") not in (None, "null")
            or flag_value(argv, "--speculative-algorithm") not in (None, "NONE")):
        raise ValueError("Shared-arena arms require non-speculative, non-disaggregated decode")
    raw_graph = flag_value(argv, "--cuda-graph-config")
    if raw_graph is not None:
        decode = parse_object(raw_graph).get("decode", {})
        if type(decode) is not dict or decode.get("backend", "full") not in ("full", "disabled"):
            raise ValueError("Shared-arena decode must be eager or full CUDA Graph")


def check_offload_execution(argv, study_config):
    """Fixed offload sets share the arena's single-stream rules; route_indirect also needs eager prefill, whose
    stores and reads are routed in Python at call time."""
    check_study_execution(argv)
    if not study_config.feature("route_indirect"):
        return
    if "--enable-mixed-chunk" in argv:
        raise ValueError("features.route_indirect routes decode stores to tail rows; mixed chunks would send decode "
                         "tokens through the prefill path")
    raw_graph = flag_value(argv, "--cuda-graph-config")
    prefill = parse_object(raw_graph).get("prefill", {}) if raw_graph is not None else None
    if "--disable-prefill-cuda-graph" not in argv and (type(prefill) is not dict
                                                       or prefill.get("backend") != "disabled"):
        raise ValueError("features.route_indirect requires eager prefill: pass --disable-prefill-cuda-graph or a "
                         "--cuda-graph-config prefill backend 'disabled'")


def check_elastic_execution(argv, model):
    """Elastic per-layer HiSparse (issue #13): single-stream decode, eager prefill (its stores follow the layer
    modes of the moment), BF16 KV for MiniMax and DSA (DeepSeek-V4 keeps its V4 C4 pages); PD decode nodes are
    allowed, prefill nodes run stock SGLang."""
    flags = {arg.partition("=")[0] for arg in argv}
    forbidden = {"--config", "--enable-two-batch-overlap", "--enable-single-batch-overlap", "--enable-pdmux",
                 "--enable-mixed-chunk"}
    if not {"--disable-overlap-schedule", "--disable-radix-cache"}.issubset(argv) or flags & forbidden:
        raise ValueError("Elastic layers require --disable-overlap-schedule, --disable-radix-cache and no "
                         "config, overlap, PD multiplexing or mixed-chunk overrides")
    if (flag_value(argv, "--disaggregation-mode") not in (None, "null", "decode")
            or flag_value(argv, "--speculative-algorithm") not in (None, "NONE")):
        raise ValueError("Elastic layers run on a normal or PD decode node without speculation")
    if model != "deepseek_v4" and flag_value(argv, "--kv-cache-dtype") not in (None, "auto", "bf16", "bfloat16"):
        raise ValueError("Elastic MiniMax and DSA layers store unquantized BF16 KV")
    raw_graph = flag_value(argv, "--cuda-graph-config")
    graph = parse_object(raw_graph) if raw_graph is not None else {}
    decode, prefill = graph.get("decode", {}), graph.get("prefill", {})
    if type(decode) is not dict or decode.get("backend", "full") not in ("full", "disabled"):
        raise ValueError("Elastic decode must be eager or full CUDA Graph")
    if "--disable-prefill-cuda-graph" not in argv and (type(prefill) is not dict
                                                       or prefill.get("backend") != "disabled"):
        raise ValueError("Elastic layers require eager prefill: pass --disable-prefill-cuda-graph or a "
                         "--cuda-graph-config prefill backend 'disabled'")


def check_plan(plan, policy, request_config=None, study_config=None, elastic_config=None):
    if type(plan.tp_size) is not int or plan.tp_size not in (4, 8):
        raise ValueError("The launcher requires explicit TP4 or TP8")
    if type(plan.sm80_bf16_projections) is not bool:
        raise ValueError("Invalid SM80 projection selection")
    validate_capacity(plan.hmax, plan.hmax, plan.top_k, 1)
    if (plan.elastic_config_path is None) != (plan.elastic_config_sha256 is None):
        raise ValueError("Incomplete frozen elastic configuration identity")
    if plan.elastic_config_path is not None:
        from .elastic_config import ElasticConfig
        if (not isinstance(elastic_config, ElasticConfig) or policy is not None or request_config is not None
                or study_config is not None
                or any(value is not None for value in (plan.policy_path, plan.request_config_path,
                    plan.study_config_path, plan.shadow_identity_directory, plan.selection_oracle_directory,
                    plan.split_copy_config))
                or plan.six_layer_timing or plan.six_layer_state):
            raise ValueError("--hisparse-elastic-config is separate from every other overlay and diagnostic")
        return None
    if elastic_config is not None:
        raise ValueError("Unexpected elastic configuration without a frozen file")
    if (plan.study_config_path is None) != (plan.study_config_sha256 is None):
        raise ValueError("Incomplete frozen study configuration identity")
    if plan.study_config_path is not None:
        from .study_config import StudyConfig, unsupported_reasons
        if (not isinstance(study_config, StudyConfig) or policy is not None or request_config is not None
                or any(value is not None for value in (plan.policy_path, plan.request_config_path,
                    plan.shadow_identity_directory, plan.selection_oracle_directory, plan.split_copy_config))
                or plan.six_layer_timing or plan.six_layer_state):
            raise ValueError("--hisparse-study-config is separate from static policies, request routing "
                             "and the existing diagnostics")
        reasons = unsupported_reasons(study_config)
        if reasons:
            raise ValueError("; ".join(reasons))
        return None
    if study_config is not None:
        raise ValueError("Unexpected study configuration without a frozen file")
    if (plan.request_config_path is None) != (plan.request_config_sha256 is None):
        raise ValueError('Incomplete frozen request configuration identity')
    if plan.request_config_path is not None:
        from .request_config import RequestConfig
        if (not isinstance(request_config, RequestConfig) or policy is not None
                or any(value is not None for value in (plan.policy_path, plan.policy_sha256,
                    plan.shadow_identity_directory, plan.selection_oracle_directory, plan.split_copy_config))
                or plan.six_layer_timing or plan.six_layer_state):
            raise ValueError('Request-type routing is separate from static policies and existing diagnostics')
        typed = request_config.policy
        if (typed.hmax != plan.hmax or typed.expected_tp_size != plan.tp_size
                or typed.selected_tokens != plan.top_k or plan.top_k != 2048):
            raise ValueError('Request-type policy TP/Hmax/Top-K must match the MiniMax service')
    elif request_config is not None:
        raise ValueError('Unexpected request configuration without a frozen file')
    if sum(value is not None for value in (plan.shadow_identity_directory, plan.selection_oracle_directory,
                                          plan.split_copy_config)) > 1:
        raise ValueError("Split copy and cache diagnostics are mutually exclusive")
    split_config = None
    if plan.six_layer_state and not plan.six_layer_timing:
        raise ValueError("Six-layer device state requires marked profiling")
    if plan.six_layer_timing and (policy is None or any(value is not None for value in (
        plan.shadow_identity_directory, plan.selection_oracle_directory, plan.split_copy_config,
    ))):
        raise ValueError("Six-layer timing requires the same fused policy overlay in both arms")
    if plan.split_copy_config is not None:
        from .split_copy import SplitCopyConfig

        if type(plan.split_copy_config) is not str:
            raise ValueError("Expected an immutable split copy config string")
        split_config = SplitCopyConfig.from_json(parse_object(plan.split_copy_config))
        if canonical_split_config(split_config) != plan.split_copy_config:
            raise ValueError("Expected canonical split copy JSON")
        if policy is None or plan.tp_size != 4 or plan.top_k != 2048 or split_config.plan_rows != 2:
            raise ValueError("Split copy requires a static policy, TP4, top_k 2048 and two plan rows")
    if plan.shadow_identity_directory is not None:
        if policy is None or any(active != plan.hmax for active in policy.capacities):
            raise ValueError("Shadow identity requires a policy with all 60 capacities equal to physical Hmax")
        if not Path(plan.shadow_identity_directory).is_dir():
            raise ValueError("Shadow identity requires an existing output directory")
    if plan.selection_oracle_directory is not None:
        if policy is None:
            raise ValueError("Selection oracle requires a static policy and explicit TP4/TP8")
        if not Path(plan.selection_oracle_directory).is_dir():
            raise ValueError("Selection oracle requires an existing output directory")
    if policy is None:
        return None
    if policy.expected_tp_size != plan.tp_size or policy.hmax != plan.hmax:
        raise ValueError("Policy TP size and Hmax must match --tp-size and --hisparse-config")
    for active in policy.capacities:
        validate_capacity(active, plan.hmax, plan.top_k, 1)
    return split_config


def resolve(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    policy_options = parser.add_mutually_exclusive_group()
    policy_options.add_argument("--policy", type=Path)
    policy_options.add_argument("--request-type-config", type=Path)
    policy_options.add_argument("--hisparse-study-config", type=Path, metavar="PATH",
                                help="hisparse-study-config-v1 arm (all-default selects stock HiSparse)")
    policy_options.add_argument("--hisparse-elastic-config", type=Path, metavar="PATH",
                                help="hisparse-elastic-v1: pressure-driven per-layer offload (issue #13)")
    parser.add_argument("--sm80-bf16-projections", action="store_true")
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument("--shadow-identity", type=Path, metavar="DIRECTORY")
    diagnostics.add_argument("--selection-oracle", type=Path, metavar="DIRECTORY")
    diagnostics.add_argument("--split-copy-config", type=Path, metavar="JSON")
    diagnostics.add_argument("--six-layer-timing", action="store_true")
    parser.add_argument("--six-layer-state", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    index = argv.index("--") if "--" in argv else len(argv)
    options = parser.parse_args(argv[:index])
    sglang_argv = list(argv[index + 1:])
    try:
        if not sglang_argv or sglang_argv.count("--enable-hisparse") != 1:
            raise ValueError("Pass explicit SGLang HiSparse arguments after --")
        tp = flag_value(sglang_argv, "--tp-size", "--tensor-parallel-size")
        if tp not in ("4", "8"):
            raise ValueError("Pass explicit --tp-size 4 or --tp-size 8")
        for names in (("--pp-size", "--pipeline-parallel-size"), ("--dp-size", "--data-parallel-size"),
                      ("--attn-cp-size",)):
            if flag_value(sglang_argv, *names) not in (None, "1"):
                raise ValueError(f"Only {names[0]} 1 is supported")
        config = flag_value(sglang_argv, "--hisparse-config")
        if config is None:
            raise ValueError("Pass --hisparse-config with top_k and device_buffer_size")
        hisparse = parse_object(config)
        if not {"top_k", "device_buffer_size"} <= hisparse.keys():
            raise ValueError("HiSparse config requires top_k and device_buffer_size")
        path = str(options.policy.expanduser().resolve(strict=True)) if options.policy else None
        policy, digest = read_policy(path) if path else (None, None)
        request_path = (str(options.request_type_config.expanduser().resolve(strict=True))
                        if options.request_type_config else None)
        request_config, request_digest = None, None
        if request_path is not None:
            from .request_config import read_request_config
            request_config, request_digest = read_request_config(request_path)
            check_request_execution(sglang_argv)
        study_path = (str(options.hisparse_study_config.expanduser().resolve(strict=True))
                      if options.hisparse_study_config else None)
        study_config, study_digest = None, None
        if study_path is not None:
            from .study_config import read as read_study_config
            study_config, study_digest = read_study_config(study_path)
            if study_config.uses_arena:
                check_study_execution(sglang_argv)
            elif study_config.uses_fixed_layers:
                check_offload_execution(sglang_argv, study_config)
        elastic_path = (str(options.hisparse_elastic_config.expanduser().resolve(strict=True))
                        if options.hisparse_elastic_config else None)
        elastic_config, elastic_digest = None, None
        if elastic_path is not None:
            from .elastic_config import read as read_elastic_config
            elastic_config, elastic_digest = read_elastic_config(elastic_path)
            check_elastic_execution(sglang_argv, elastic_config.model)
        if options.six_layer_timing:
            required = {"--disable-cuda-graph", "--disable-prefill-cuda-graph",
                        "--disable-overlap-schedule", "--disable-radix-cache"}
            flags = {token.partition("=")[0] for token in sglang_argv}
            if (not required.issubset(sglang_argv) or "--config" in flags
                    or any(flag.startswith("--cuda-graph-") for flag in flags)):
                raise ValueError("Six-layer timing requires explicit eager execution without overlap or Radix")
        diagnostic = options.shadow_identity or options.selection_oracle
        if diagnostic is not None:
            diagnostic = diagnostic.expanduser().resolve(strict=True)
            required = {"--disable-cuda-graph", "--disable-prefill-cuda-graph", "--disable-overlap-schedule",
                        "--disable-radix-cache"}
            flags = {token.partition("=")[0] for token in sglang_argv}
            overrides = {"--config", "--enable-two-batch-overlap", "--enable-single-batch-overlap"}
            if (not required.issubset(sglang_argv) or flags & overrides
                    or any(flag.startswith("--cuda-graph-") for flag in flags)):
                raise ValueError("Cache diagnostics require explicit eager execution with overlap and Radix disabled")
            if not diagnostic.is_dir() or any(diagnostic.iterdir()):
                raise ValueError("Cache diagnostics require an empty output directory")
        split_config = None
        if options.split_copy_config is not None:
            check_split_execution(sglang_argv)
            split_config = canonical_split_config(read_split_copy_config(options.split_copy_config))
        plan = LaunchPlan(int(tp), hisparse["device_buffer_size"], hisparse["top_k"],
                          options.sm80_bf16_projections, path, digest,
                          str(diagnostic) if options.shadow_identity is not None else None,
                          str(diagnostic) if options.selection_oracle is not None else None,
                          split_config, options.six_layer_timing, options.six_layer_state,
                          request_path, request_digest, study_path, study_digest, elastic_path, elastic_digest)
        check_plan(plan, policy, request_config, study_config, elastic_config)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return plan, sglang_argv, options.dry_run


def install_for_process(plan):
    if (plan.policy_path is None) != (plan.policy_sha256 is None):
        raise ValueError("Incomplete frozen policy identity")
    policy = read_policy(plan.policy_path, plan.policy_sha256)[0] if plan.policy_path is not None else None
    request_config = None
    if plan.request_config_path is not None:
        from .request_config import read_request_config
        request_config = read_request_config(plan.request_config_path, plan.request_config_sha256)[0]
    study_config = None
    if plan.study_config_path is not None:
        from .study_config import read as read_study_config
        study_config = read_study_config(plan.study_config_path, plan.study_config_sha256)[0]
    elastic_config = None
    if plan.elastic_config_path is not None:
        from .elastic_config import read as read_elastic_config
        elastic_config = read_elastic_config(plan.elastic_config_path, plan.elastic_config_sha256)[0]
    split_config = check_plan(plan, policy, request_config, study_config, elastic_config)
    if plan.sm80_bf16_projections:
        source = Path(__file__).with_name("compat") / "minimax_sm80.py"
        if sha256(source.read_bytes()).hexdigest() != SM80_SHA256:
            raise RuntimeError("Vendored SM80 adapter SHA-256 differs")

    if elastic_config is not None:
        from .elastic_device import install
        install(elastic_config, plan.elastic_config_sha256)
    elif request_config is not None:
        from .request_hook import install
        install(**request_config.install_kwargs())
    elif study_config is not None and study_config.uses_arena:
        from .shared_lru_device import install
        install(study_config, plan.study_config_sha256)
    elif study_config is not None and study_config.uses_fixed_layers:
        from .dynamic_offload_device import install
        install(study_config, plan.study_config_sha256)
    elif plan.shadow_identity_directory is not None:
        from .shadow import install
        install(policy, Path(plan.shadow_identity_directory))
    elif plan.selection_oracle_directory is not None:
        from .shadow import install_selected
        install_selected(policy, Path(plan.selection_oracle_directory))
    elif split_config is not None:
        from .hook import install
        install(policy, enabled=True, split_config=split_config)
    elif policy is not None:
        from .hook import install
        install(policy, enabled=True)
    else:
        import sglang
        from .hook import verify_sources
        verify_sources(Path(sglang.__file__).resolve().parent)
    if plan.sm80_bf16_projections:
        os.environ[SM80_ENV] = "1"
        from .compat.minimax_sm80 import install as install_sm80
        install_sm80()
    if plan.six_layer_timing:
        from .six_layer_timing_markers import install as install_timing_markers
        install_timing_markers(snapshot_state=plan.six_layer_state)


def install_spawned_worker():
    payload = os.environ.get(SPAWN_ENV)
    if payload is not None:
        install_for_process(LaunchPlan(**parse_object(payload)))


def main(argv=None):
    plan, upstream, dry_run = resolve(list(sys.argv[1:] if argv is None else argv))
    mode = "static_layer_overlay" if plan.policy_path else "stock_hisparse"
    if plan.request_config_path is not None:
        mode = 'request_type_layer_overlay'
    elif plan.study_config_path is not None:
        mode = "hisparse_study_config"
    elif plan.elastic_config_path is not None:
        mode = "hisparse_elastic"
    elif plan.shadow_identity_directory is not None:
        mode = "shadow_identity"
    elif plan.selection_oracle_directory is not None:
        mode = "selection_oracle"
    elif plan.split_copy_config is not None:
        mode = SPLIT_COPY_MODE
    summary = asdict(plan) | {"mode": mode}
    if plan.study_config_path is not None:
        from .study_config import capabilities, read as read_study_config
        study = read_study_config(plan.study_config_path, plan.study_config_sha256)[0]
        summary |= {"study_config_canonical_sha256": study.digest, "study_resolved_config": study.as_dict(),
                    "study_capabilities": capabilities(study)}
    if plan.elastic_config_path is not None:
        from .elastic_config import read as read_elastic_config
        elastic = read_elastic_config(plan.elastic_config_path, plan.elastic_config_sha256)[0]
        summary |= {"elastic_config_canonical_sha256": elastic.digest, "elastic_config": elastic.as_dict()}
    print(json.dumps(summary, sort_keys=True), flush=True)
    if dry_run:
        return 0

    method = multiprocessing.get_start_method(allow_none=True)
    if method is None:
        multiprocessing.set_start_method("spawn")
    elif method != "spawn":
        raise RuntimeError("This launch entry point requires multiprocessing spawn")
    old_environment = {name: os.environ.get(name) for name in (SPAWN_ENV, SM80_ENV)}
    os.environ[SPAWN_ENV] = json.dumps(asdict(plan), sort_keys=True)
    try:
        install_for_process(plan)
        from sglang.srt.entrypoints.http_server import launch_server
        from sglang.srt.server_args import prepare_server_args
        launch_server(prepare_server_args(upstream))
    finally:
        for name, value in old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return 0


if __name__ == "__mp_main__":
    install_spawned_worker()
elif __name__ == "__main__":
    raise SystemExit(main())
