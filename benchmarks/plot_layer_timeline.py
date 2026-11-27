"""Plot two Nsight layer timelines of ``benchmarks.nsys_layer_timeline`` (control, candidate) to PNG, PDF or SVG.

    python -m benchmarks.plot_layer_timeline --control frozen_a.json --candidate dynamic.json \
        --labels "Frozen T40" "Dynamic" --output pair.png

Each column is one arm. The top row splits every layer of each step's critical rank (least TP wait; medians over
steps) into swap, attention, collective transfer, TP wait, other kernels and idle, and draws every rank's median TP
wait per layer over it; layers the candidate swaps are shaded in both columns. The rows below keep every step and
rank: whole-step time and TP wait per rank, with each step's critical rank marked. ``pair_data`` checks and
extracts the plotted values without matplotlib, which ``draw`` imports. Ported from issue #3's
``benchmarks/plot_timeline_pair.py`` (PR #10).
"""
import argparse
import json
from pathlib import Path

from benchmarks import nsys_layer_timeline as timeline

PARTS = ("swap", "attention", "collective", "tp_wait", "other", "idle")
COLORS = dict(swap="#bd5429", attention="#165a86", collective="#7aa6c2", tp_wait="#c43c6c", other="#9aa5ab",
              idle="#e3e7e9")


def _arm(summary, label):
    if summary.get("status") != "pass" or summary.get("critical") is None:
        raise ValueError(f"{label}: the timeline must pass and attribute TP waits ({summary.get('reasons')})")
    layers = summary["critical"]["layers"]
    parts = {part: [] for part in PARTS}
    for row in layers:
        for category in timeline.SWEEP + ("idle",):
            value = row[f"{category}_us"]
            if category == "collective":        # the wait is part of the collective kernels' time
                parts["tp_wait"].append(row["tp_wait_us"])
                value -= row["tp_wait_us"]
            parts[category].append(value)
    steps = summary["timeline"]
    return dict(label=label, layers=[row["layer"] for row in layers], parts=parts,
                steps=[t["step"] for t in steps], critical=[t["critical_rank"] for t in steps],
                whole_us=[t["whole_step_us"] for t in steps], tp_wait_us=[t["tp_wait_us"] for t in steps],
                rank_waits=[[row["tp_wait_us"] for row in rank["layers"]] for rank in summary["ranks"]],
                swapping=[row["layer"] for row in layers if row["swap_us"] > 0])


def pair_data(control, candidate, labels=("Control", "Candidate")):
    arms = [_arm(summary, label) for summary, label in zip((control, candidate), labels)]
    if arms[0]["layers"] != arms[1]["layers"] or control["tp_size"] != candidate["tp_size"]:
        raise ValueError("the two timelines must cover the same layers and TP size")
    return dict(arms=arms, tp_size=control["tp_size"], swapped=arms[1]["swapping"])


def draw(data, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(3, 2, figsize=(15, 9), sharex="row", sharey="row", layout="constrained")
    for column, arm in enumerate(data["arms"]):
        top, whole, wait = axes[:, column]
        bottom = [0.0] * len(arm["layers"])
        for layer in data["swapped"]:
            top.axvspan(layer - 0.5, layer + 0.5, color="#f4ead9", linewidth=0)
        for part in PARTS:
            top.bar(arm["layers"], arm["parts"][part], bottom=bottom, width=0.8, color=COLORS[part])
            bottom = [b + v for b, v in zip(bottom, arm["parts"][part])]
        for waits in arm["rank_waits"]:
            top.step(arm["layers"], waits, where="mid", color="black", linewidth=0.7, alpha=0.6)
        top.set_title(f"{arm['label']}: critical rank per layer (median µs)", loc="left", size=10)
        top.set_xlabel("layer")
        for metric, ax, name in (("whole_us", whole, "whole step"), ("tp_wait_us", wait, "TP wait")):
            for values in zip(*arm[metric]):
                ax.plot(arm["steps"], [v / 1e3 for v in values], linewidth=0.8, alpha=0.7)
            ax.scatter(arm["steps"], [arm[metric][i][r] / 1e3 for i, r in enumerate(arm["critical"])], s=10,
                       color="black", zorder=3)
            ax.set_title(f"{arm['label']}: {name} per rank (ms), critical rank marked", loc="left", size=10)
            ax.set_xlabel("decode step")
    fig.legend(handles=[Patch(color=COLORS[part], label=part) for part in PARTS]
               + [Patch(color="#f4ead9", label="swapped in the candidate"),
                  plt.Line2D([], [], color="black", linewidth=0.7, label="TP wait of each rank"),
                  plt.Line2D([], [], color="black", marker="o", linestyle="", label="critical rank")],
               loc="outside right upper", fontsize=8)
    fig.savefig(output)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--labels", nargs=2, default=("Control", "Candidate"))
    parser.add_argument("--output", type=Path, required=True, help="PNG, PDF or SVG; never overwritten")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    data = pair_data(*(json.loads(path.read_text()) for path in (args.control, args.candidate)), args.labels)
    draw(data, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
