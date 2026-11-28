"""Issue #11 report of one ``benchmarks/run_dynamic_offload_a100.sh`` output directory; standard library only.

    python -m benchmarks.dynamic_offload_report all --dir OUT --model minimax_m3 --output OUT/report.json

* correctness: every pass of the correctness workload against stock, request by request (output token hashes).
  B=1 must match exactly; B=3 (a padded Graph bucket) is reported next to the stock repeat's run-to-run noise. Per
  dynamic correctness server, the switch receipts and the Graph captures logged after its first switch (none).
* serving: per run and pressure point, TTFT and TPOT p50/p95, output tokens/s, natural EOS and SLO goodput, and the
  real batch: decode-log lines at #running-req equal to the client concurrency on CUDA Graph. A pair is qualified
  when both runs completed without failures and both logs show at least ``MIN_GRAPH_LINES`` such lines. Per
  comparison and pressure point, the relative change of every qualified pair (one per repeat, AB/BA interleaved),
  their mean and a two-sided 95% Student t interval over pairs.
* memory: per run, each card's peak memory.used in the nvidia-smi samples, /server_info capacity and the bind
  receipts (logical KV, ledger, allocated bytes, allocator state) of every rank.
* switch: per dynamic serving run, the client wall time of the switch and every rank's receipt.
* nsys: per profiled arm, ``benchmarks.nsys_layer_timeline`` over its export, and the paired layer plots when
  matplotlib is installed.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev

from benchmarks import nsys_layer_timeline as layer_timeline
from benchmarks.compare_runs import quantile

ARMS = ("full_resident", "stock", "frozen_a", "frozen_b", "indirect", "dynamic", "dynamic_shadow")
COMPARISONS = (("frozen_a", "stock"), ("frozen_b", "stock"), ("frozen_a", "full_resident"), ("indirect", "frozen_a"),
               ("dynamic", "frozen_a"), ("dynamic", "indirect"), ("dynamic_shadow", "dynamic"))
PLOTS = (("stock", "frozen_a"), ("frozen_a", "indirect"), ("frozen_a", "dynamic"))
MIN_GRAPH_LINES = 8           # decode log lines (one per 8 steps): 64 or more Graph steps at the full batch
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
DECODE = re.compile(r"Decode batch.*?#running-req: (\d+),.*?graph: (True|False)")
EVENT = re.compile(r'\{.*"event": "hisparse_dynamic_offload_(bound|switch)".*\}')
NSYS_LAYERS = {"minimax_m3": "0-59"}


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _hashes(report):
    """request alias -> output hash (None when the request failed) of a correctness or stream client report."""
    return {r["request_alias"]: r.get("output_ids_sha256") if r.get("successful") else None
            for r in (report or {}).get("requests", [])}


def events(log):
    """Bind and switch receipts printed by every rank into one server log."""
    found = dict(bound=[], switch=[])
    try:
        lines = Path(log).read_text(errors="replace").splitlines()
    except OSError:
        return found
    for line in lines:
        match = EVENT.search(line)
        if match:
            try:
                found[match.group(1)].append(json.loads(match.group(0)))
            except ValueError:
                pass
    return found


def correctness(root):
    folder, out = Path(root) / "correctness", dict(passes={}, servers={})
    stock = {batch: _hashes(_json(folder / f"stock.{batch}.json")) for batch in ("b1", "b3")}
    for path in sorted(folder.glob("*.b[13].json")):
        label, batch = path.name[:-len(".bX.json")], path.name[-7:-5]
        if label == "stock":
            continue
        got, want = _hashes(_json(path)), stock[batch]
        aliases = sorted(want)
        out["passes"].setdefault(label, {})[batch] = dict(
            compared=len(aliases), identical=sum(got.get(a) is not None and got.get(a) == want[a] for a in aliases),
            mismatched=[a for a in aliases if got.get(a) is not None and want[a] is not None and got[a] != want[a]],
            failed=[a for a in aliases if got.get(a) is None or want[a] is None])
    for log in sorted(Path(root).glob("correctness-dynamic*.server.log")):
        run = log.name[:-len(".server.log")]
        captures = (Path(root) / f"{run}.captures-after-switch.txt")
        out["servers"][run] = dict(
            captures_after_first_switch=int(captures.read_text().split()[0]) if captures.exists() else None,
            switches=[dict(rank=e.get("rank"), target=e.get("target"), status=e.get("status"), rows=e.get("rows"),
                           seconds=e.get("seconds")) for e in events(log)["switch"]])
    b1 = [p["b1"] for p in out["passes"].values() if "b1" in p]
    out["b1_identical"] = bool(b1) and all(p["compared"] > 0 and p["identical"] == p["compared"] for p in b1)
    out["b3_noise_mismatches"] = len(out["passes"].get("stock_repeat", {}).get("b3", {}).get("mismatched", []))
    out["no_capture_after_switch"] = all(s["captures_after_first_switch"] == 0 for s in out["servers"].values()
                                         if s["captures_after_first_switch"] is not None)
    return out


def decode_lines(log):
    """{(running requests, on CUDA Graph): decode log lines} of one server log."""
    counts = {}
    try:
        text = Path(log).read_text(errors="replace")
    except OSError:
        return counts
    for running, graph in DECODE.findall(text):
        key = (int(running), graph == "True")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _percentiles(values):
    ordered = sorted(v for v in values if v is not None)
    return dict(p50=quantile(ordered, 0.5), p95=quantile(ordered, 0.95)) if ordered else None


def run_metrics(report, lines, concurrency):
    """Serving metrics of one stream client report and the real-batch evidence of its server log."""
    records, summary = report.get("requests", []), report.get("summary", {})
    ok = [r for r in records if r.get("successful")]
    graph = lines.get((concurrency, True), 0)
    return dict(status=report.get("status"), requests=summary.get("requests"), failures=summary.get("failures"),
                ttft_s=_percentiles(r.get("ttft_s") for r in ok), tpot_s=_percentiles(r.get("tpot_s") for r in ok),
                output_tokens_per_s=summary.get("completed_output_tokens_per_s"),
                natural_eos=summary.get("natural_eos_requests"), length_limited=summary.get("length_limited_requests"),
                natural_eos_slo_goodput_per_s=summary.get("slo", {}).get("natural_eos_slo_goodput_per_s"),
                response_slo_goodput_per_s=summary.get("slo", {}).get("response_slo_goodput_per_s"),
                graph_lines_at_batch=graph,
                qualified=report.get("status") == "complete" and summary.get("failures") == 0
                and graph >= MIN_GRAPH_LINES)


def interval(values):
    """Mean and two-sided 95% Student t interval of independent paired changes."""
    if len(values) < 2:
        return dict(pairs=len(values), mean=values[0] if values else None, ci95=None)
    half = T975.get(len(values) - 1, 1.96) * stdev(values) / math.sqrt(len(values))
    return dict(pairs=len(values), mean=mean(values), ci95=[mean(values) - half, mean(values) + half])


def serving(root):
    root, runs = Path(root), {}
    for path in sorted(root.glob("*-r*-c*.json")):
        match = re.fullmatch(r"(.+)-r(\d+)-c(\d+)\.json", path.name)
        report = _json(path)
        if match is None or report is None or match.group(1) not in ARMS:
            continue
        arm, repeat, concurrency = match.group(1), int(match.group(2)), int(match.group(3))
        lines = decode_lines(root / f"{arm}-r{repeat}.server.log")
        runs.setdefault(arm, {}).setdefault(concurrency, {})[repeat] = run_metrics(report, lines, concurrency)
    comparisons = []
    for candidate, baseline in COMPARISONS:
        for concurrency in sorted(set(runs.get(candidate, {})) & set(runs.get(baseline, {}))):
            a, b = runs[candidate][concurrency], runs[baseline][concurrency]
            pairs = [r for r in sorted(set(a) & set(b)) if a[r]["qualified"] and b[r]["qualified"]]

            def change(metric, key=None):
                values = []
                for r in pairs:
                    x, y = (a[r][metric], b[r][metric]) if key is None else (
                        (a[r][metric] or {}).get(key), (b[r][metric] or {}).get(key))
                    if x is not None and y:
                        values.append(x / y - 1)
                return interval(values)

            comparisons.append(dict(candidate=candidate, baseline=baseline, concurrency=concurrency,
                                    qualified_repeats=pairs, unqualified_repeats=sorted((set(a) | set(b)) - set(pairs)),
                                    output_tokens_per_s=change("output_tokens_per_s"),
                                    tpot_p50=change("tpot_s", "p50"), tpot_p95=change("tpot_s", "p95"),
                                    ttft_p50=change("ttft_s", "p50")))
    return dict(runs=runs, comparisons=comparisons, min_graph_lines=MIN_GRAPH_LINES)


def card_peaks(path):
    """{GPU index: peak memory.used MiB} of one nvidia-smi CSV sample file."""
    peaks = {}
    try:
        rows = list(csv.reader(Path(path).read_text(errors="replace").splitlines()))
    except OSError:
        return peaks
    for row in rows[1:]:
        try:
            index, used = int(row[1]), float(row[2].strip().split()[0])
        except (IndexError, ValueError):
            continue
        peaks[index] = max(peaks.get(index, 0.0), used)
    return peaks


def _find(value, key):
    """First value of ``key`` anywhere in nested dicts and lists."""
    if isinstance(value, dict) and key in value:
        return value[key]
    for item in value.values() if isinstance(value, dict) else value if isinstance(value, list) else ():
        found = _find(item, key)
        if found is not None:
            return found
    return None


def memory(root):
    """Per run: card peaks, /server_info capacity and every rank's bind receipt."""
    root, out = Path(root), {}
    for log in sorted(root.glob("*.server.log")):
        run = log.name[:-len(".server.log")]
        info, bound = _json(root / f"{run}.server-info.json"), events(log)["bound"]
        out[run] = dict(
            card_peak_mib=card_peaks(root / f"{run}.gpu-memory.csv"),
            max_total_num_tokens=_find(info, "max_total_num_tokens"),
            max_running_requests=_find(info, "max_running_requests"),
            ranks=[dict(rank=e.get("rank"), logical_kv=e.get("logical_kv"), cell_size=e.get("cell_size"),
                        bytes=e.get("bytes"), device_total=_find(e.get("ledger"), "dynamic"),
                        device_memory_at_bind=e.get("device_memory_at_bind")) for e in bound])
    return out


def switches(root):
    """Per dynamic serving run: the client wall time of each switch and every rank's receipts."""
    root, out = Path(root), {}
    for log in sorted(root.glob("dynamic*-r*.server.log")):
        run = log.name[:-len(".server.log")]
        out[run] = dict(http=[_json(p) for p in sorted(root.glob(f"{run}.switch*-time.json"))],
                        receipts=[dict(rank=e.get("rank"), status=e.get("status"), target=e.get("target"),
                                       rows=e.get("rows"), seconds=e.get("seconds"), error=e.get("error"))
                                  for e in events(log)["switch"]])
    return out


def nsys(root, model, profile=None, layers=None):
    """Layer timelines of the profiled arms and their paired plots."""
    root, out = Path(root), dict(timelines={}, plots={})
    profile = profile or layer_timeline.PROFILES.get(model)
    layers = layers or NSYS_LAYERS.get(model)
    if profile is None or layers is None:
        return dict(out, skipped=f"{model}: pass --attention-kernel, --layer-end-kernel and --nsys-layers")
    tp_size = 8 if model == "minimax_m3" else 4
    for trace in sorted(root.glob("*-nsys-*.sqlite")):
        arm = trace.name.split("-nsys-")[0]
        try:
            out["timelines"][arm] = layer_timeline.analyze(trace, layers=layer_timeline.parse_layers(layers),
                                                           tp_size=tp_size, profile=profile)
        except layer_timeline.TimelineError as error:
            out["timelines"][arm] = dict(status="fail", reasons=[str(error)])
    try:
        from benchmarks import plot_layer_timeline as plot
        import matplotlib  # noqa: F401
    except ImportError:
        return dict(out, plots_skipped="matplotlib is not installed")
    for control, candidate in PLOTS:
        a, b, path = out["timelines"].get(control), out["timelines"].get(candidate), \
            root / f"nsys-{control}-vs-{candidate}.png"
        if a and b and a.get("status") == b.get("status") == "pass" and not path.exists():
            plot.draw(plot.pair_data(a, b, (control, candidate)), path)
            out["plots"][f"{control}/{candidate}"] = path.name
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("section", choices=("all", "correctness", "serving", "memory", "switch", "nsys"))
    parser.add_argument("--dir", type=Path, required=True, help="the recipe's output directory")
    parser.add_argument("--model", choices=("minimax_m3", "deepseek_v4"), default="minimax_m3")
    parser.add_argument("--swap-kernel", action="append", help="nsys read-path kernel names (non-MiniMax models)")
    parser.add_argument("--attention-kernel", action="append")
    parser.add_argument("--layer-end-kernel", action="append")
    parser.add_argument("--nsys-layers", help="layer IDs whose attention ends a period, e.g. 0-59")
    parser.add_argument("--output", type=Path, required=True, help="JSON report; never overwritten")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    named = (args.swap_kernel, args.attention_kernel, args.layer_end_kernel)
    profile = dict(swap=tuple(args.swap_kernel), attention=tuple(args.attention_kernel),
                   merge=tuple(args.layer_end_kernel)) if all(named) else None
    sections = dict(correctness=lambda: correctness(args.dir), serving=lambda: serving(args.dir),
                    memory=lambda: memory(args.dir), switch=lambda: switches(args.dir),
                    nsys=lambda: nsys(args.dir, args.model, profile, args.nsys_layers))
    chosen = sections if args.section == "all" else {args.section: sections[args.section]}
    report = dict(schema="hisparse-dynamic-offload-report-v1", model=args.model,
                  gate=(_json(args.dir / "gate.json") or {}).get("passed"),
                  build_receipt=(args.dir / "build.json").exists(), **{name: run() for name, run in chosen.items()})
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
