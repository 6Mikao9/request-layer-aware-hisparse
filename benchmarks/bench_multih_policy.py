"""Bounded synthetic CPU planner microbenchmark; no torch, GPU or serving.

Run with python -B -m benchmarks.bench_multih_policy --output NEW_FILE.json.
Fixture construction, policy construction and verification are not timed.
Only plan_for_admission is timed, with three fresh policies, no warmup.
All 60 layers have resident candidates; only migration-cap layers have deficits.
Every legal H can cover its layer's deficit, so a complete plan is known to
exist within the cap and host budget. This is not a measured workload model.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import itertools
import json
import os
from pathlib import Path
import platform
import random
from statistics import median
import sys
from time import perf_counter_ns

from benchmarks.artifact_paths import is_private_path
from runtime.layer_eviction_policy import Candidate, LayerEvictionPolicy, LayerKey
import runtime.layer_eviction_policy as implementation


def machine():
    brand = platform.processor()
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            brand = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return dict(cpu=brand, logical_cpus=os.cpu_count(), platform=platform.platform(),
                python=platform.python_version(), implementation=platform.python_implementation(),
                affinity="not pinned")


def fixture(tp, requests, h_options, cap, seed):
    rng = random.Random(seed)
    layers, page_tokens, page_bytes = 60, 128, 128*512
    domains = tp*layers
    stressed = tuple(sorted(rng.sample(range(layers),cap)))
    capacities = (8192,) if h_options == 1 else (2048,4096,8192)
    # An incoming 512-token chunk needs four pages per domain. Victims are
    # different, completed resident requests, hence zero pending-demand relief.
    free = tuple((0 if layer in stressed else 8)*page_bytes for _ in range(tp) for layer in range(layers))
    required = (4*page_bytes,)*domains
    relief = (0,)*domains
    host_charge = 32768*512  # Full max-context backing per selected layer/rank.
    host_free = (cap*host_charge,)*tp
    candidates = []
    for row in range(1,requests+1):
        for layer in range(layers):
            for hot in capacities:
                reclaimed_pages = 80-(hot//page_tokens+1)  # H plus latest page.
                reclaim = tuple(reclaimed_pages*page_bytes if col == layer else 0
                                for _ in range(tp) for col in range(layers))
                candidates.append(Candidate(LayerKey(row,1,layer),reclaim,
                    rng.randint(1,20),rng.randint(0,200),0.,(host_charge,)*tp,relief,hot))
    return free,required,host_free,tuple(candidates),stressed


def run(*, seed=20261117, max_timed_seconds=25., largest_only=False):
    code_sha = sha256(Path(implementation.__file__).read_bytes()).hexdigest()
    cases = list(itertools.product((4,8),(4,16),(1,3),(4,8)))
    cases.sort(key=lambda c:c[0]*c[1]*c[2]*c[3],reverse=True)
    if largest_only:
        cases = cases[:1]
    results, total_ns, stopped = [], 0, False
    for tp,requests,h_options,cap in cases:
        # Keep scope below the requested 30s budget; the initial largest case
        # supplies a conservative next-call estimate for the remaining cases.
        if total_ns and total_ns + 2*max(ns for r in results for ns in r["samples_ns"]) > max_timed_seconds*1e9:
            stopped = True
            break
        free,required,host,candidates,stressed = fixture(tp,requests,h_options,cap,seed)
        samples, complete, actions = [], [], []
        for _ in range(3):
            if total_ns and total_ns + 2*max(samples+[ns for r in results for ns in r["samples_ns"]]) > max_timed_seconds*1e9:
                stopped = True
                break
            policy = LayerEvictionPolicy(trigger_free_bytes=(0,)*len(free),
                target_free_bytes=(1,)*len(free),max_migrations=cap,min_residency=1.)
            start = perf_counter_ns()
            plan = policy.plan_for_admission(free,candidates,required_free_bytes=required,
                                            host_free_bytes=host,now=10.)
            elapsed = perf_counter_ns()-start
            samples.append(elapsed)
            total_ns += elapsed
            valid = (plan is not None and plan.target_reached
                     and len(plan.keys) == cap and len(set(plan.keys)) == cap
                     and {key.layer for key in plan.keys} == set(stressed)
                     and all(f >= n for f,n in zip(plan.projected_free_bytes,plan.required_free_bytes_after))
                     and all(n >= 0 for n in plan.projected_host_free_bytes))
            complete.append(valid)
            actions.append(0 if plan is None else len(plan.actions))
            if not valid:
                raise AssertionError("Known-feasible synthetic fixture did not yield a complete legal plan")
        if samples:
            results.append(dict(tp=tp,layers=60,active_requests=requests,H_options=h_options,
                migration_cap=cap,candidates=len(candidates),resource_domains=len(free),
                deficit_layers=list(stressed),samples_ns=samples,median_ms=median(samples)/1e6,
                complete=complete,selected_actions=actions))
        if stopped:
            break
    if sha256(Path(implementation.__file__).read_bytes()).hexdigest() != code_sha:
        raise RuntimeError("Planner source changed during the microbenchmark")
    return dict(schema="synthetic-multih-planner-cpu-benchmark-v1",timestamp_utc=datetime.now(timezone.utc).isoformat(),
        machine=machine(),seed=seed,planner_sha256=code_sha,total_timed_seconds=total_ns/1e9,
        requested_cases=len(cases),completed_cases=len(results),stopped_for_budget=stopped,
        results=results,limitations=["Anonymous synthetic integer resource vectors and nonnegative costs; no real request traces.",
            "CPU plan_for_admission only; excludes fixture/policy construction, capture, TP transport, migration and model execution.",
            "Three samples, no warmup, process affinity not pinned; descriptive timing, not a statistical performance claim.",
            "No GPU access, no inference speedup, and no end-to-end control-latency or goodput conclusion."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--largest-only",action="store_true")
    args = parser.parse_args()
    if args.output is not None and is_private_path(args.output.absolute()):
        parser.error("Private area is not a benchmark destination")
    result = run(largest_only=args.largest_only)
    raw = json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    if args.output is not None:
        with args.output.open("x",encoding="utf-8") as stream:
            stream.write(raw)
    print(raw)


if __name__ == "__main__":
    main()
