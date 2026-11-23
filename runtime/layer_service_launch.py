"""Opt-in, spawn-safe launcher for the pinned eager layer-eviction service.

Dry-run only validates CPU-readable configuration and flags. It does not claim
model compatibility, memory fit, or CUDA execution. The native worker factory
validates canonical SGLang configuration again before allocating custom pools.
"""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import sys

from .layer_service_config import read_service_config
from .launch import flag_value, parse_object, SM80_ENV, SM80_SHA256, SPAWN_ENV as OLD_SPAWN_ENV


SPAWN_ENV = "_REQUEST_LAYER_EVICTION_SERVICE"


@dataclass(frozen=True)
class ServiceLaunchPlan:
    config_path: str
    config_sha256: str
    sm80_bf16_projections: bool


def check_upstream(argv, config):
    flags = {token.partition("=")[0] for token in argv if token.startswith("--")}
    required = {"--disable-cuda-graph", "--disable-prefill-cuda-graph",
                "--disable-overlap-schedule", "--disable-radix-cache"}
    if not required.issubset(argv):
        raise ValueError("Require explicit eager, no overlap and no Radix execution")
    forbidden = {"--config", "--enable-hisparse", "--hisparse-config", "--enable-hierarchical-cache",
        "--enable-unified-memory", "--enable-memory-saver", "--enable-mixed-chunk", "--enable-pdmux",
        "--enable-two-batch-overlap", "--enable-single-batch-overlap", "--enable-dp-attention",
        "--enable-dp-lm-head", "--enable-priority-scheduling", "--attn-dcp-size",
        "--tp", "--pp", "--dp", "--enable-hicache",
        "--enable-prefill-delayer", "--enable-pd-role-switch"}
    if flags & forbidden or any(f.startswith(("--cuda-graph-", "--speculative-", "--disaggregation-",
        "--dllm-", "--mamba-")) for f in flags):
        raise ValueError("Unsupported legacy HiSparse, alternate scheduling or memory mode")
    # Require matching explicit capacities: native autosizing cannot replace
    # the independently budgeted main, logical/index and request-row domains.
    for names, expected in (
        (("--tp-size", "--tensor-parallel-size"), config.expected_tp_size),
        (("--context-length",), config.max_context),
        (("--max-running-requests",), config.max_running_requests),
        (("--max-total-tokens",), config.logical_token_capacity),
        (("--page-size",), 128),
    ):
        actual = flag_value(argv, *names)
        if actual is None or actual != str(expected):
            raise ValueError(f"{names[0]} must match frozen service configuration: {expected}")
    for names in (("--pp-size", "--pipeline-parallel-size"),
                  ("--dp-size", "--data-parallel-size"), ("--nnodes",),
                  ("--attn-cp-size",)):
        actual = flag_value(argv, *names)
        if actual not in (None, "1"):
            raise ValueError("Only single-node TP with no PP/DP/CP is supported")
    if flag_value(argv, "--attention-backend") != "triton":
        raise ValueError("Use the pinned triton attention backend")
    if flags & {"--decode-attention-backend", "--prefill-attention-backend"}:
        raise ValueError("Split attention backend overrides are not supported")
    if flag_value(argv, "--dtype") not in ("bfloat16", "bf16"):
        raise ValueError("Declare BF16 model execution")
    if flag_value(argv, "--kv-cache-dtype") not in ("auto", "bfloat16", "bf16"):
        raise ValueError("Declare BF16/auto KV dtype; runtime will verify actual BF16")
    if not flag_value(argv, "--model-path", "--model"):
        raise ValueError("Explicit read-only model path required")


def resolve(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-config", required=True, type=Path)
    parser.add_argument("--sm80-bf16-projections", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    if "--" not in argv:
        parser.error("Separate SGLang arguments with --")
    split = argv.index("--")
    options = parser.parse_args(argv[:split])
    upstream = argv[split+1:]
    try:
        path = str(options.service_config.expanduser().resolve(strict=True))
        config, digest = read_service_config(path)
        check_upstream(upstream, config)
        if os.environ.get(OLD_SPAWN_ENV):
            raise ValueError("Cannot combine the old overlay launcher with the layer service")
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return ServiceLaunchPlan(path, digest, options.sm80_bf16_projections), config, upstream, options.dry_run


def install_for_process(plan):
    if (type(plan) is not ServiceLaunchPlan or type(plan.sm80_bf16_projections) is not bool
            or type(plan.config_path) is not str or not Path(plan.config_path).is_absolute()):
        raise ValueError("Invalid frozen spawn plan")
    if os.environ.get(OLD_SPAWN_ENV):
        raise ValueError("Old overlay and independent layer service are mutually exclusive")
    config, digest = read_service_config(plan.config_path, plan.config_sha256)
    if plan.sm80_bf16_projections:
        source = Path(__file__).with_name("compat") / "minimax_sm80.py"
        if sha256(source.read_bytes()).hexdigest() != SM80_SHA256:
            raise RuntimeError("Vendored SM80 adapter SHA-256 differs")
        os.environ[SM80_ENV] = "1"
        from .compat.minimax_sm80 import install as install_sm80
        install_sm80()
    elif os.environ.get(SM80_ENV) not in (None, "", "0"):
        raise ValueError("Ambient SM80 adapter setting differs from frozen launch plan")
    from .layer_worker_bootstrap import install
    return install(config, digest, enabled=True)


def install_spawned_worker():
    payload = os.environ.get(SPAWN_ENV)
    if payload is not None:
        values = parse_object(payload)
        if set(values) != {"config_path", "config_sha256", "sm80_bf16_projections"}:
            raise ValueError("Unknown spawn service fields")
        return install_for_process(ServiceLaunchPlan(**values))


def main(argv=None):
    plan, config, upstream, dry = resolve(list(sys.argv[1:] if argv is None else argv))
    print(json.dumps(config.summary() | {"config_sha256": plan.config_sha256,
        "sm80_bf16_projections": plan.sm80_bf16_projections, "dry_run": dry}, sort_keys=True), flush=True)
    if dry:
        return 0
    method = multiprocessing.get_start_method(allow_none=True)
    if method is None:
        multiprocessing.set_start_method("spawn")
    elif method != "spawn":
        raise RuntimeError("Layer service launcher requires multiprocessing spawn")
    old_environment = {name:os.environ.get(name) for name in (SPAWN_ENV, SM80_ENV)}
    if old_environment[SPAWN_ENV] is not None:
        raise ValueError("Nested layer service launch is not supported")
    os.environ[SPAWN_ENV] = json.dumps(asdict(plan), sort_keys=True)
    try:
        install_for_process(plan)
        from .layer_worker_bootstrap import run_scheduler_process
        from sglang.srt.entrypoints.http_server import launch_server
        from sglang.srt.server_args import prepare_server_args
        launch_server(prepare_server_args(upstream), run_scheduler_process_func=run_scheduler_process)
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
