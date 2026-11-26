"""Steady decode timelines per step and TP rank from an Nsight Systems SQLite export; standard library only.

The diagnostics stage of ``runtime.combined_run`` profiles one diagnostic run per arm with
``nsys profile --trace=cuda,nvtx --cuda-graph-trace=node`` and exports it with
``nsys export --type sqlite``. CUPTI gives every kernel of one CUDA Graph launch the
launch's correlation ID, so the graph kernels of one (process, device, correlation ID)
are one replay. On each rank the steady decode graph is the replay signature (kernel
names and grids) seen most often; its k-th replay is decode step k on every rank, since
TP ranks replay in lockstep. Ranks are ordered by CUDA device ID (SGLang puts TP rank r
on device base_gpu_id + r). Per step and rank, as in NATIVE_GRAPH_20261123.md:

* whole step: first kernel start to last kernel end of the replay, gaps included. The
  bottleneck rank of a step has the longest whole step; its own per-layer timeline
  explains that step (per-layer maxima of different ranks are never added up).
* sparse layer j ends at the j-th ``_merge_topk_attn_out_kernel``. Since the previous
  merge: read-path kernel time (native swap, arena resolver, resident resolver), its
  exposed part (no other kernel of the replay running), read or attention start to
  merge end, and the merge-to-merge period (one full decoder layer). The period is
  also split exclusively into swap, resolver, attention (with merge), collective,
  other kernels and idle; an instant with several kernels counts once, for the first
  of these categories, so the six parts add up to the period.
* collective: union of NCCL and custom all-reduce kernels. The k-th collective of a
  step is one synchronization of all ranks: they leave it together, so a rank's TP
  wait is its kernel time above the shortest one, that of the last rank to arrive.
  Durations need no common clock across devices. The critical rank of a step waited
  least in total; that of a layer waited least at the collectives of its period. A
  period opens with the previous layer's output and MoE reductions, so the critical
  rank of layer j+1 is the rank that finished layer j's read path and attention last.

Kernel names come from the pinned sources. A schema or kernel-count mismatch gives
reasons and null fields, never a guess. Medians cover every replay of the steady graph,
including the few full-graph steps of batch ramp-up and drain.
"""
import itertools
import sqlite3
from collections import Counter, defaultdict
from statistics import median

from benchmarks.compare_runs import quantile

REQUIRED = {"CUPTI_ACTIVITY_KIND_KERNEL": ("start", "end", "deviceId", "correlationId", "globalPid", "shortName",
                                          "graphNodeId", "gridX", "gridY", "gridZ"),
            "StringIds": ("id", "value")}
KINDS = {   # short kernel names: pinned SGLang e7399a46 and this repository's resolvers
    "swap": ("load_cache_to_device_buffer_kernel",     # kernels/jit/csrc/kvcacheio/hisparse.cuh: native swap-in
             "copy_cache_planned_kernel"),             # same file: the copy launch of features.split_copy
    "resolver": ("resolve_kernel",                     # runtime/shared_lru_kernel.cuh: arena resolver
                 "resolve_resident_kernel"),           # runtime/layer_placement_kernel.cuh: resident resolver
    "attention": ("_gqa_share_sparse_decode_kernel_impl",),   # minimax_sparse/decode/topk_sparse.py (Triton)
    "merge": ("_merge_topk_attn_out_kernel",),
}
READ = ("swap", "resolver")
COLLECTIVE = ("nccl", "allreduce", "all_reduce", "cross_device_reduce")
SWEEP = ("swap", "resolver", "attention", "collective", "other")   # exclusive split, in overlap priority
LAYER_FIELDS = ("read_path_us", "exposed_wait_us", "read_to_merge_us", "layer_us", *(f"{c}_us" for c in SWEEP),
                "idle_us", "tp_wait_us")


class TimelineError(ValueError):
    pass


def kind_of(name):
    """swap, resolver, attention, merge, collective or None for one short kernel name."""
    for kind, bases in KINDS.items():
        if any(name == base or name.startswith(base + "_") for base in bases):
            return kind
    lowered = name.lower()
    return "collective" if any(token in lowered for token in COLLECTIVE) else None


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _length(merged):
    return sum(end - start for start, end in merged)


def _overlap(a, b):
    """Shared length of two merged interval lists."""
    total, i, j = 0, 0, 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def sweep(intervals, lo, hi):
    """Exclusive ns of each SWEEP category inside [lo, hi] and the idle rest; earlier categories win overlaps."""
    covered, out = [], {}
    for category in SWEEP:
        mine = _union([(max(s, lo), min(e, hi)) for s, e, c in intervals if c == category and min(e, hi) > max(s, lo)])
        out[category] = _length(mine) - _overlap(mine, covered)
        covered = _union(covered + mine)
    return out, hi - lo - _length(covered)


def replay_summary(kernels, kinds):
    """One graph replay [(start ns, end ns, name id)] by start: whole step, layers, read kernels, collectives.

    ``collectives`` lists (layer period, ns) in launch order; the period after the last merge is ``len(layers)``.
    """
    begin, previous, window, carry, layers, collectives = kernels[0][0], kernels[0][0], [], [], [], []
    for start, end, name in kernels:
        kind = kinds.get(name)
        window.append((start, end, kind))
        if kind == "collective":
            collectives.append((len(layers), end - start))
        if kind != "merge":
            continue
        read = _union([(s, e) for s, e, k in window if k in READ])
        rest = _union([(s, e) for s, e, k in window if k not in READ])
        first = min([s for s, _, k in window if k in (*READ, "attention")] or [start])
        # kernels of a side stream may outlast a merge; their remainder belongs to the next period
        parts, idle = sweep([(s, e, "attention" if k == "merge" else k or "other") for s, e, k in carry + window],
                            previous, end)
        layers.append(dict(read_path_us=sum(e - s for s, e, k in window if k in READ) / 1e3,
                           exposed_wait_us=(_length(read) - _overlap(read, rest)) / 1e3,
                           read_to_merge_us=(end - first) / 1e3, layer_us=(end - previous) / 1e3,
                           **{f"{c}_us": ns / 1e3 for c, ns in parts.items()}, idle_us=idle / 1e3, tp_wait_us=None))
        carry = [k for k in carry + window if k[1] > end]
        window, previous = [], end
    reads = Counter(name for _, _, name in kernels if kinds.get(name) in READ)
    collective = _union([(s, e) for s, e, name in kernels if kinds.get(name) == "collective"])
    return dict(start=begin, whole=(max(end for _, end, _ in kernels) - begin) / 1e3, layers=layers, reads=reads,
                collective=_length(collective) / 1e3, collectives=collectives, tp_wait=None)


def _columns(db, table):
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def _percentiles(values):
    ordered = sorted(values)
    return dict(p50=quantile(ordered, 0.50), p95=quantile(ordered, 0.95)) if ordered else None


def _replays(database):
    """{(process, device): {graph signature: [replay summary]}}, the string table and the export schema."""
    db = None
    try:
        db = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        missing = {table: sorted(set(columns) - _columns(db, table)) for table, columns in REQUIRED.items()}
        missing = {table: columns for table, columns in missing.items() if columns}
        if missing:
            raise TimelineError(f"the Nsight export lacks {missing}; export it with the nsys that captured it")
        schema = None
        if {"name", "value"} <= _columns(db, "META_DATA_EXPORT"):
            schema = {name: value for name, value in db.execute("SELECT name, value FROM META_DATA_EXPORT")
                      if "VERSION" in str(name).upper()}
        names = dict(db.execute("SELECT id, value FROM StringIds"))
        kinds = {key: kind_of(str(value)) for key, value in names.items()}
        rows = db.execute("SELECT globalPid, deviceId, correlationId, start, end, shortName, gridX, gridY, gridZ "
                          "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId IS NOT NULL AND graphNodeId != 0 "
                          "ORDER BY globalPid, deviceId, correlationId, start")
        replays = defaultdict(lambda: defaultdict(list))
        for (pid, device, _), group in itertools.groupby(rows, key=lambda row: row[:3]):
            group = list(group)
            replays[(pid, device)][hash(tuple(row[5:] for row in group))].append(
                replay_summary([row[3:6] for row in group], kinds))
    except sqlite3.Error as error:
        raise TimelineError(f"unreadable Nsight export: {error}") from error
    finally:
        if db is not None:
            db.close()
    if not replays:
        raise TimelineError("the capture holds no CUDA Graph kernel nodes; profile with --cuda-graph-trace=node")
    return replays, names, schema


def analyze(database, *, layers, tp_size):
    """Steady decode timeline of one capture: per-rank summaries, every step's bottleneck and critical ranks with
    all-rank whole steps and TP waits, and the per-layer medians of those two ranks.

    ``layers`` are the sparse layer IDs in model order; each has one attention merge per step.
    """
    replays, names, schema = _replays(database)
    reasons, ranks = [], sorted(replays, key=lambda key: (key[1], key[0]))
    if len(ranks) != tp_size:
        reasons.append(f"{len(ranks)} (process, device) pairs replayed graphs; the plan runs TP{tp_size}")
    steady = {key: sorted(max(replays[key].values(), key=len), key=lambda replay: replay["start"]) for key in ranks}
    counts = sorted({len(runs) for runs in steady.values()})
    steps = counts[0]
    if len(counts) > 1:
        reasons.append(f"ranks replayed the steady graph {counts} times; steps are aligned on the first {steps}")
    merges = sorted({len(replay["layers"]) for runs in steady.values() for replay in runs[:steps]})
    layered = merges == [len(layers)]
    if not layered:
        reasons.append(f"{merges} attention merges per steady replay; the model has {len(layers)} sparse layers")
    paired = all(len({tuple(period for period, _ in steady[key][step]["collectives"]) for key in ranks}) == 1
                 for step in range(steps))
    if not paired:
        reasons.append("ranks launched different collectives in one step; TP waits and critical ranks are null")
    timeline, bottleneck, critical, layer_critical = [], Counter(), Counter(), [Counter() for _ in layers]
    for step in range(steps):
        runs = [steady[key][step] for key in ranks]
        whole = [run["whole"] for run in runs]
        slowest = max(range(len(ranks)), key=whole.__getitem__)
        bottleneck[slowest] += 1
        entry = dict(step=step, bottleneck_rank=slowest, whole_step_us=whole, critical_rank=None, tp_wait_us=None)
        if paired:
            waits = [Counter() for _ in runs]
            for column in zip(*(run["collectives"] for run in runs)):     # one synchronization of all ranks
                shortest = min(ns for _, ns in column)
                for wait, (period, ns) in zip(waits, column):
                    wait[period] += (ns - shortest) / 1e3
            for run, wait in zip(runs, waits):
                run["tp_wait"] = sum(wait.values())
                for j, row in enumerate(run["layers"]):
                    row["tp_wait_us"] = wait[j]
            totals = [run["tp_wait"] for run in runs]
            entry.update(critical_rank=min(range(len(runs)), key=totals.__getitem__), tp_wait_us=totals)
            critical[entry["critical_rank"]] += 1
            for j in sorted({period for period, _ in runs[0]["collectives"]} & set(range(len(layers)))):
                layer_critical[j][min(range(len(runs)), key=lambda r: waits[r][j])] += 1
        timeline.append(entry)

    def per_layer(runs):
        if not layered:
            return None
        values = [{field: [run["layers"][j][field] for run in runs] for field in LAYER_FIELDS}
                  for j in range(len(layers))]
        return [dict(layer=layer, **{field: None if None in v else median(v) for field, v in row.items()})
                for layer, row in zip(layers, values)]

    summaries = []
    for rank, key in enumerate(ranks):
        runs, reads = steady[key][:steps], Counter()
        for run in runs:
            reads.update(run["reads"])
        summaries.append(dict(
            rank=rank, device=key[1], graph_replays=sum(map(len, replays[key].values())),
            steady_replays=len(steady[key]), whole_step_us=_percentiles(run["whole"] for run in runs),
            collective_us=_percentiles(run["collective"] for run in runs), bottleneck_steps=bottleneck[rank],
            tp_wait_us=_percentiles(run["tp_wait"] for run in runs) if paired else None,
            critical_steps=critical[rank] if paired else None,
            read_kernels_per_step=dict(sorted((names[name], count / steps) for name, count in reads.items())),
            layers=per_layer(runs)))
    slowest = [steady[ranks[t["bottleneck_rank"]]][t["step"]] for t in timeline]
    chosen = [steady[ranks[t["critical_rank"]]][t["step"]] for t in timeline] if paired else None
    critical_layers = per_layer(chosen) if paired else None
    for row, counts in zip(critical_layers or [], layer_critical):
        row["critical_ranks"] = dict(sorted(counts.items()))
    return dict(status="fail" if reasons else "pass", reasons=reasons, export_schema=schema, tp_size=tp_size,
                steps=steps, layer_ids=list(layers), ranks=summaries,
                bottleneck=dict(whole_step_us=_percentiles(run["whole"] for run in slowest), layers=per_layer(slowest)),
                critical=None if not paired else dict(whole_step_us=_percentiles(run["whole"] for run in chosen),
                                                      tp_wait_us=_percentiles(run["tp_wait"] for run in chosen),
                                                      layers=critical_layers),
                timeline=timeline, kernel_kinds=dict(KINDS, collective=COLLECTIVE))
