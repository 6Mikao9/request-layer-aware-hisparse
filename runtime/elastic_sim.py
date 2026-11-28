"""Discrete-step serving simulator for elastic per-layer HiSparse (issue #13). CPU only; a model, not a result.

It replays one arrival trace against a decode instance (PD decode: requests arrive with their KV) under several
policies, with step time from ``elastic_policy.CostModel`` plus a resident base ``a + b * B``:

* ``full_resident``: level 0 forever (no HiSparse work, smallest capacity).
* ``stock``: every eligible layer offloaded (stock HiSparse capacity and cost).
* ``fixed:<level>``: one frozen level, e.g. the calibrated T40 of issue #3.
* ``hybrid``: a page-spill model of vLLM Hybrid HiSparse: every layer runs the resolver once a request spills
  (``hybrid_fixed_us`` per layer per step), spilled pages miss at each layer's rate, capacity as ``stock``.
* ``steady`` / ``smooth``: the controller; steady stalls for a backfill copy, smooth paces it at
  ``controller.backfill_gbps`` and pays ``route_us`` per switchable resident layer per step.

Use it to see how watermarks, dwell and SLO interact on a workload and to choose what to measure. The numbers
come from the cost model's assumptions (measured A100 miss rates and swap-kernel fit, guessed base and hybrid
overheads); only paired runs on the A100 measure the system.

    python -m runtime.elastic_sim --config runtime/fixtures/elastic/minimax_m3_steady.json --rate 1.5 --requests 300
"""
import argparse
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import random

from . import elastic_config as ec
from . import elastic_memory as em
from . import elastic_policy as ep

ROOT = Path(__file__).resolve().parents[1]
NEW_TOKEN_RATIO = 0.3      # share of the remaining output SGLang's admission reserves per request (decayed value)


@dataclass(frozen=True)
class Preset:
    """Per-rank geometry: ``layers`` eligible layers with K and V rows of ``row_bytes`` each, an indexer cache of
    ``index_bytes`` per token, ``budget_gib`` of KV chunks."""
    layers: int
    row_bytes: int
    index_bytes: int
    page: int
    hot_per_request: int
    max_running: int
    budget_gib: float
    max_tokens: int
    base_ms: tuple
    stock_hot_rows: int = 396_544      # stock MiniMax TP8 device rows measured on the A100 (logical = 2x)
    pcie_gbps: float = 20.0


PRESETS = {
    # MiniMax-M3.1-Flash TP8: one KV head of 128 BF16 per rank; index bytes are a placeholder until a bind receipt.
    "minimax_m3_tp8": Preset(60, 256, 2048, 128, 4224, 128, 16.0, 4 * 1024 * 1024, (18.0, 0.35)),
}


@dataclass
class Request:
    rid: int
    arrival: float
    prompt: int
    output: int
    admitted: float = -1.0
    first: float = -1.0
    done: float = -1.0
    generated: int = 0

    @property
    def context(self):
        return self.prompt + self.generated


def workload(rate, count, prompt, output, seed):
    rng, t, out = random.Random(seed), 0.0, []
    for rid in range(count):
        t += rng.expovariate(rate)
        out.append(Request(rid, t, rng.randint(*prompt), rng.randint(*output)))
    return out


def layout_for(preset, chunk_mib=2):
    specs = [em.BufferSpec("index", preset.index_bytes, em.ALWAYS, pad_rows=preset.page)]
    for layer in range(preset.layers):
        specs += [em.BufferSpec(f"{kind}{layer}", preset.row_bytes, em.SWITCHABLE, layer=layer, pad_rows=preset.page)
                  for kind in "kv"]
    hot = preset.hot_per_request * preset.max_running
    return em.Layout(specs, chunk_bytes=chunk_mib << 20, max_tokens=preset.max_tokens, hot_rows=hot,
                     page_tokens=preset.page)


@dataclass
class Result:
    policy: str
    finished: int
    makespan_s: float
    output_tokens: int
    tpot_ms: list = field(default_factory=list)
    ttft_ms: list = field(default_factory=list)
    levels: dict = field(default_factory=dict)
    transitions: int = 0
    stall_ms: float = 0.0
    retracted: int = 0

    def summary(self, slo):
        def q(values, p):
            return round(sorted(values)[min(len(values) - 1, int(p * len(values)))], 2) if values else None
        ok = sum(1 for a, b in zip(self.tpot_ms, self.ttft_ms)
                 if (slo.get("tpot_ms") is None or a <= slo["tpot_ms"]) and
                 (slo.get("ttft_ms") is None or b <= slo["ttft_ms"]))
        return dict(policy=self.policy, finished=self.finished, makespan_s=round(self.makespan_s, 2),
                    throughput_tok_s=round(self.output_tokens / max(self.makespan_s, 1e-9), 1),
                    tpot_p50_ms=q(self.tpot_ms, 0.5), tpot_p95_ms=q(self.tpot_ms, 0.95),
                    ttft_p50_ms=q(self.ttft_ms, 0.5), ttft_p95_ms=q(self.ttft_ms, 0.95),
                    goodput_req_s=round(ok / max(self.makespan_s, 1e-9), 3), transitions=self.transitions,
                    stall_ms=round(self.stall_ms, 1), retracted=self.retracted,
                    steps_by_level={str(k): v for k, v in sorted(self.levels.items())})


def simulate(config, preset, requests, policy, *, hybrid_fixed_us=15.0, route_us=4.0, noise=0.03, seed=0,
             max_steps=2_000_000):
    """Run ``requests`` (copied) under ``policy`` ("full_resident", "stock", "fixed:<level>", "hybrid",
    "elastic"); ``config.mode`` picks steady or smooth for "elastic"."""
    rng = random.Random(seed)
    layout = layout_for(preset, config.memory["chunk_mib"])
    chunks = int(preset.budget_gib * (1 << 30)) // layout.chunk
    geometry = ep.Geometry(layout, chunks, preset.hot_per_request, preset.max_running,
                           hot_min=preset.hot_per_request)
    cost = ep.CostModel(ep.load_profile(config, ROOT))
    ctl = ep.Controller(config, geometry, cost)
    every = frozenset(range(preset.layers))
    page, per_req = preset.page, preset.hot_per_request
    align = lambda rows: -(-rows // page) * page

    if policy in ("stock", "hybrid"):          # stock sizing: a static hot pool and logical = 2 x device rows
        offloaded_at = lambda level: every
        hot = align(min(preset.stock_hot_rows, layout.hot_rows))
        tokens = min(2 * hot, layout.max_tokens_for(chunks, hot_rows=hot, offloaded=every))
        level = 0
    else:
        offloaded_at = config.offloaded
        level = int(policy.split(":")[1]) if policy.startswith("fixed:") else (
            0 if policy == "full_resident" else config.controller["initial_level"])
        hot = align(per_req * (preset.max_running if policy != "elastic" else 8))
        tokens = layout.max_tokens_for(chunks, hot_rows=hot, offloaded=offloaded_at(level))
        if tokens < 0:
            raise ValueError(f"{policy}: level {level} does not fit the budget with {hot} hot rows")
    resident_cap = layout.max_tokens_for(chunks, hot_rows=hot, offloaded=frozenset())
    pending, waiting, running = sorted((replace(r) for r in requests), key=lambda r: r.arrival), [], []
    now, step, result = 0.0, 0, Result(policy, 0, 0.0, 0)
    fill = None                                # smooth backfill in flight: [target, bytes left]
    cap = None                                 # admission cap of the last controller decision
    used = lambda: sum(align(r.context) for r in running)
    measured = []
    while (pending or waiting or running) and step < max_steps:
        while pending and pending[0].arrival <= now:
            waiting.append(pending.pop(0))
        off = offloaded_at(level)
        limit = min(preset.max_running, cap or preset.max_running)
        if off:
            limit = min(limit, hot // per_req)
        reserve = sum(int(NEW_TOKEN_RATIO * (r.output - r.generated)) for r in running)
        while waiting and len(running) < limit:
            r = waiting[0]                     # SGLang-style: reserve a share of every request's remaining output
            ask = align(r.prompt) + int(NEW_TOKEN_RATIO * r.output)
            if used() + reserve + ask > tokens:
                break
            reserve += int(NEW_TOKEN_RATIO * r.output)
            waiting.pop(0)
            fill_ms = r.prompt * (preset.layers - len(off)) * 2 * preset.row_bytes / (preset.pcie_gbps * 1e6)
            r.admitted = now + fill_ms / 1e3   # PD decode: resident layers load the prompt from host
            running.append(r)
        if not running:
            if not pending:
                break
            now = max(now, pending[0].arrival)
            continue
        batch = len(running)
        ctx = sum(r.context for r in running) / batch
        base_ms = preset.base_ms[0] + preset.base_ms[1] * batch
        if policy == "hybrid":
            spill = max(0.0, 1.0 - resident_cap / max(used(), 1))
            ms = base_ms + (preset.layers * hybrid_fixed_us / 1e3 + spill * cost.offload_ms(every, batch, ctx)
                            if spill > 0 else 0.0)
        else:
            ms = base_ms + cost.offload_ms(off, batch, ctx)
            if policy == "elastic" and config.mode == "smooth":
                ms += route_us * sum(1 for layer in config.switchable if layer not in off) / 1e3
        ms *= 1.0 + noise * (2 * rng.random() - 1)
        if fill is not None:                   # paced backfill shares the H2D link with the swap-ins
            fill[1] -= config.controller["backfill_gbps"] * 1e6 * ms / 1e3
            if fill[1] <= 0:
                level, fill = fill[0], None
                result.transitions += 1
        now += ms / 1e3
        step += 1
        result.levels[level] = result.levels.get(level, 0) + 1
        measured.append(ms)
        for r in list(running):
            if r.first < 0:
                r.first = max(now, r.admitted)
            r.generated += 1
            if r.generated >= r.output:
                r.done = now
                running.remove(r)
                result.finished += 1
                result.output_tokens += r.output
                result.tpot_ms.append(1e3 * (r.done - r.first) / max(1, r.output - 1))
                result.ttft_ms.append(1e3 * (r.first - r.arrival))
        while used() > tokens and running:     # decode growth overflowed: retract the newest request
            victim = max(running, key=lambda r: r.admitted)
            running.remove(victim)
            victim.generated, victim.first = 0, -1.0
            waiting.insert(0, victim)
            result.retracted += 1
        if policy != "elastic" or step % config.controller["interval_steps"] or fill is not None:
            continue
        live = used()
        obs = ep.Observation(step=step, level=level, tokens=tokens, tokens_used=live, tokens_peak=live,
                             hot_rows=hot, hot_used=len(running) * per_req, hot_peak=len(running) * per_req,
                             running=len(running), mean_context=ctx, waiting=len(waiting),
                             waiting_tokens=sum(r.prompt for r in waiting[:preset.max_running]),
                             step_ms=sum(measured) / len(measured))
        measured = []
        decision = ctl.tick(obs)
        cap = decision.admission_cap
        if not decision.changes(obs):
            continue
        if decision.level < level:             # backfill: copy the returning layers' history from host
            returning = config.offloaded(level) - config.offloaded(decision.level)
            nbytes = live * len(returning) * 2 * preset.row_bytes
            if config.mode == "smooth":
                fill = [decision.level, nbytes]
                tokens, hot = min(tokens, decision.tokens), decision.hot_rows
                continue
            stall = nbytes / (preset.pcie_gbps * 1e6)
            now += stall / 1e3
            result.stall_ms += stall
        elif decision.level > level and config.mode == "steady":
            now += 5e-3                        # graph-set swap and hot-row warm start
            result.stall_ms += 5.0
        if decision.level != level:
            result.transitions += 1
        level, tokens, hot = decision.level, decision.tokens, decision.hot_rows
    result.makespan_s = now
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preset", default="minimax_m3_tp8", choices=sorted(PRESETS))
    parser.add_argument("--rate", type=float, default=1.0, help="arrivals per second")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--prompt", type=int, nargs=2, default=(16384, 32768))
    parser.add_argument("--output", type=int, nargs=2, default=(256, 1024))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policies", nargs="+",
                        default=["full_resident", "stock", "fixed:5", "hybrid", "elastic"])
    parser.add_argument("--tpot-slo-ms", type=float)
    parser.add_argument("--ttft-slo-ms", type=float)
    options = parser.parse_args(argv)
    config, _ = ec.read(options.config)
    preset = PRESETS[options.preset]
    requests = workload(options.rate, options.requests, tuple(options.prompt), tuple(options.output), options.seed)
    slo = dict(tpot_ms=options.tpot_slo_ms or (config.slo or {}).get("tpot_ms"), ttft_ms=options.ttft_slo_ms)
    rows = [simulate(config, preset, requests, policy, seed=options.seed).summary(slo)
            for policy in options.policies]
    print(json.dumps(dict(config=str(options.config), mode=config.mode, preset=options.preset,
                          rate=options.rate, slo=slo, results=rows), indent=1))


if __name__ == "__main__":
    main()
