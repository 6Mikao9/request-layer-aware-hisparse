"""Summarize CUDA Graph launches in an Nsight Systems SQLite export (``nsys export --type sqlite``).

    python -m benchmarks.nsys_graph_summary trace.sqlite --output graph-summary.json

Each ``cudaGraphLaunch`` runtime record is matched to the GPU work that carries its
correlation ID (``nsys profile --cuda-graph-trace=node`` keeps one record per node).
Per launch the summary lists the nodes in start order (kind, name, graph node ID,
stream, start offset, duration), the stream order between consecutive nodes, the
largest number of overlapping nodes, the exposed wait from the end of the launch
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


def _columns(db, table):
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def _stats(values):
    return None if not values else dict(count=len(values), median=statistics.median(values), min=min(values),
                                        max=max(values))


def summarize(path):
    db = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    try:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"StringIds", "CUPTI_ACTIVITY_KIND_RUNTIME"} <= tables:
            raise ValueError("not an nsys SQLite export with CUDA runtime records")
        strings = dict(db.execute("SELECT id, value FROM StringIds"))
        launches = db.execute(
            "SELECT r.start, r.end, r.correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds n "
            "ON r.nameId = n.id WHERE n.value LIKE 'cudaGraphLaunch%' ORDER BY r.start").fetchall()
        work = {}
        for kind, table in ACTIVITY:
            if table not in tables:
                continue
            columns = _columns(db, table)
            name = next((c for c in ("demangledName", "shortName") if c in columns), None)
            node = "graphNodeId" if "graphNodeId" in columns else None
            stream = "streamId" if "streamId" in columns else None
            query = (f"SELECT start, end, correlationId, {stream or 'NULL'}, {node or 'NULL'}, {name or 'NULL'} "
                     f"FROM {table}")
            for start, end, correlation, stream_id, node_id, name_id in db.execute(query):
                work.setdefault(correlation, []).append(dict(
                    kind=kind, name=strings.get(name_id, kind), graph_node=node_id, stream=stream_id,
                    start=start, end=end))
    finally:
        db.close()
    summary = []
    for api_start, api_end, correlation in launches:
        nodes = sorted(work.get(correlation, ()), key=lambda n: (n["start"], n["end"]))
        if not nodes:
            continue
        first, last = nodes[0]["start"], max(n["end"] for n in nodes)
        running = overlap = 0
        for _, step in sorted([(n["start"], 1) for n in nodes] + [(n["end"], -1) for n in nodes]):
            running += step
            overlap = max(overlap, running)
        summary.append(dict(
            api_ns=api_end - api_start, exposed_wait_ns=max(0, first - api_end), gpu_span_ns=last - first,
            max_concurrent_nodes=overlap,
            stream_order=[[a["graph_node"], b["graph_node"]] for a, b in zip(nodes, nodes[1:])
                          if a["stream"] == b["stream"]],
            nodes=[dict(kind=n["kind"], name=n["name"], graph_node=n["graph_node"], stream=n["stream"],
                        start_ns=n["start"] - first, duration_ns=n["end"] - n["start"]) for n in nodes]))
    kernels = {}
    for launch in summary:
        for node in launch["nodes"]:
            if node["kind"] == "kernel":
                kernels.setdefault(node["name"], []).append(node["duration_ns"])
    return dict(event="nsys_graph_summary", source_sha256=None, graph_launches=len(summary),
                nodes_per_launch=sorted({len(launch["nodes"]) for launch in summary}),
                kernel_ns={name: _stats(values) for name, values in sorted(kernels.items())},
                exposed_wait_ns=_stats([launch["exposed_wait_ns"] for launch in summary]),
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
    summary["source_sha256"] = hashlib.sha256(args.trace.read_bytes()).hexdigest()
    with args.output.open("x") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(dict(status="pass", graph_launches=summary["graph_launches"],
                          nodes_per_launch=summary["nodes_per_launch"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
