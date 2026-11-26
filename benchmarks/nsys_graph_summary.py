"""Summarize CUDA Graph launches in an Nsight Systems SQLite export (``nsys export --type sqlite``).

    python -m benchmarks.nsys_graph_summary trace.sqlite --output graph-summary.json

Each ``cudaGraphLaunch`` runtime record is matched to the GPU work in the same
process with its correlation ID (``nsys profile --cuda-graph-trace=node`` keeps
one record per node).
Per launch the summary lists the nodes in start order (kind, name, graph node ID,
stream, start offset, duration), the stream order between consecutive nodes, the
largest number of overlapping nodes, the queue delay from the end of the launch
call to the first node, and the GPU span. Standard library only; the trace is
opened read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import sys

ACTIVITY = (("kernel", "CUPTI_ACTIVITY_KIND_KERNEL"), ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY"),
            ("memset", "CUPTI_ACTIVITY_KIND_MEMSET"))
RUNTIME = "CUPTI_ACTIVITY_KIND_RUNTIME"
# In the Nsight export, the low 24 bits hold the thread ID in a serialized GlobalId.
TID_BITS = 24


def _columns(db, table):
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def _stats(values):
    return None if not values else dict(count=len(values), median=statistics.median(values), min=min(values),
                                        max=max(values))


def _column(columns, name):
    return name if name in columns else "NULL"


def summarize(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"StringIds", RUNTIME} <= tables:
            raise ValueError("not an nsys SQLite export with CUDA runtime records")
        strings = dict(db.execute("SELECT id, value FROM StringIds"))
        runtime_pid = f"r.globalTid >> {TID_BITS}" if "globalTid" in _columns(db, RUNTIME) else "NULL"
        launches = db.execute(
            f"SELECT r.start, r.end, r.correlationId, {runtime_pid} FROM {RUNTIME} r JOIN StringIds n "
            "ON r.nameId = n.id WHERE n.value LIKE 'cudaGraphLaunch%' ORDER BY r.start").fetchall()
        launch_keys = {(pid, correlation) for _, _, correlation, pid in launches}
        if len(launch_keys) != len(launches):
            raise ValueError("multiple CUDA Graph launches with the same process and correlation ID")
        launch_by_corr = {correlation: pid for _, _, correlation, pid in launches}
        unknown_launch_pids = {correlation for pid, correlation in launch_keys if pid is None}
        activity_pids = set()
        unattributed = []
        work = {}
        for kind, table in ACTIVITY:
            if table not in tables:
                continue
            columns = _columns(db, table)
            name = next((c for c in ("demangledName", "shortName") if c in columns), "NULL")
            activity_pid = f"globalPid >> {TID_BITS}" if "globalPid" in columns else "NULL"
            identifiers = ", ".join(_column(columns, c) for c in ("deviceId", "contextId", "streamId", "graphNodeId"))
            query = f"SELECT start, end, correlationId, {activity_pid}, {identifiers}, {name} FROM {table}"
            for start, end, correlation, pid, device, context, stream, node, name_id in db.execute(query):
                if pid is not None:
                    activity_pids.add(pid)
                if correlation not in launch_by_corr:
                    continue
                activity = dict(kind=kind, name=strings.get(name_id, kind), graph_node=node, device=device,
                                context=context, stream=stream, start=start, end=end)
                key = (pid, correlation)
                if pid is None or correlation in unknown_launch_pids:
                    unattributed.append((correlation, activity))
                elif key in launch_keys:
                    work.setdefault(key, []).append(activity)
        if unattributed:
            if runtime_pid != "NULL":
                activity_pids.update(pid for (pid,) in db.execute(f"SELECT DISTINCT {runtime_pid} FROM {RUNTIME} r")
                                     if pid is not None)
            repeated_correlations = {
                correlation for (correlation,) in db.execute(f"SELECT correlationId FROM {RUNTIME} "
                                                             "GROUP BY correlationId HAVING COUNT(*) > 1")
            }
            if len(activity_pids) > 1 or repeated_correlations.intersection(c for c, _ in unattributed):
                raise ValueError("missing process identifiers prevent an unambiguous CUDA Graph match")
            for correlation, activity in unattributed:
                work.setdefault((launch_by_corr[correlation], correlation), []).append(activity)
    finally:
        db.close()
    summary = []
    for api_start, api_end, correlation, pid in launches:
        nodes = sorted(work.get((pid, correlation), ()), key=lambda n: (n["start"], n["end"]))
        if not nodes:
            continue
        first, last = nodes[0]["start"], max(n["end"] for n in nodes)
        running = overlap = 0
        for _, step in sorted([(n["start"], 1) for n in nodes] + [(n["end"], -1) for n in nodes]):
            running += step
            overlap = max(overlap, running)
        stream_order, previous_nodes = [], {}
        for n in nodes:
            if n["stream"] is None or n["graph_node"] is None:
                continue
            key = (n["device"], n["context"], n["stream"])
            if key in previous_nodes:
                stream_order.append([previous_nodes[key], n["graph_node"]])
            previous_nodes[key] = n["graph_node"]
        summary.append(dict(
            global_process_id=pid, correlation_id=correlation,
            api_ns=api_end - api_start, launch_queue_ns=max(0, first - api_end), gpu_span_ns=last - first,
            max_concurrent_nodes=overlap,
            stream_order=stream_order,
            nodes=[dict(kind=n["kind"], name=n["name"], graph_node=n["graph_node"], device=n["device"],
                        context=n["context"], stream=n["stream"],
                        start_ns=n["start"] - first, duration_ns=n["end"] - n["start"]) for n in nodes]))
    if not summary:
        raise ValueError("no CUDA Graph launches matched GPU nodes; use --cuda-graph-trace=node")
    kernels = {}
    for launch in summary:
        for node in launch["nodes"]:
            if node["kind"] == "kernel":
                kernels.setdefault(node["name"], []).append(node["duration_ns"])
    return dict(event="nsys_graph_summary", source_sha256=None, graph_launches=len(summary),
                unmatched_graph_launches=len(launches) - len(summary),
                nodes_per_launch=sorted({len(launch["nodes"]) for launch in summary}),
                kernel_ns={name: _stats(values) for name, values in sorted(kernels.items())},
                launch_queue_ns=_stats([launch["launch_queue_ns"] for launch in summary]),
                gpu_span_ns=_stats([launch["gpu_span_ns"] for launch in summary]), launches=summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace", type=Path, help="nsys export --type sqlite output")
    parser.add_argument("--output", type=Path, required=True, help="summary JSON; never overwritten")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    try:
        summary = summarize(args.trace)
    except (OSError, sqlite3.Error, ValueError) as error:
        print(json.dumps(dict(status="fail", error=str(error))), file=sys.stderr)
        return 1
    with args.trace.open("rb") as source:
        summary["source_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    with args.output.open("x") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(dict(status="pass", graph_launches=summary["graph_launches"],
                          nodes_per_launch=summary["nodes_per_launch"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
