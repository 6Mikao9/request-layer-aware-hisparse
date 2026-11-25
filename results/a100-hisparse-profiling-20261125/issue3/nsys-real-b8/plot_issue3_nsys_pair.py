"""Plot the verified Issue #3 Nsight pair aggregate to PNG, PDF, or SVG."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


METRICS = (
    ("read_path_us", "Native swap kernel"),
    ("exposed_wait_us", "Swap time without other kernels"),
    ("layer_us", "Merge end to merge end"),
)
STOCK = "#165a86"
T40 = "#bd5429"
RESIDENT = "#e3eee6"
INK = "#192730"


def checked_pair(data):
    if data["schema"] != "issue3-offline-nsys-pair-v1" or data["status"] != "paired":
        raise ValueError("Expected a paired Issue #3 offline Nsight aggregate")

    arms = data["arms"]
    if len(arms) != 2 or "stock" not in arms:
        raise ValueError("Expected stock and one T40 candidate")
    candidate = next(name for name in arms if name != "stock")
    timelines = []
    for name in ("stock", candidate):
        arm = arms[name]
        timeline = arm["graph_timeline"]
        if (arm["status"] != "verified" or timeline["status"] != "pass"
                or timeline["batch"] != data["batch"] or timeline["tp_size"] != 8
                or timeline["steps"] < 1):
            raise ValueError(f"Expected a verified TP8 Graph timeline for {name}")
        timelines.append(timeline)

    stock, t40 = timelines
    layers = stock["layer_ids"]
    if len(layers) != 60 or layers != list(range(layers[0], layers[0] + 60)):
        raise ValueError("Expected 60 consecutive layer IDs")
    offloaded = t40["native_swap_layers"]
    if (t40["layer_ids"] != layers or stock["native_swap_layers"] != layers
            or len(offloaded) != 40 or len(set(offloaded)) != 40
            or not set(offloaded).issubset(layers)):
        raise ValueError("Expected native swap on all 60 stock layers and 40 T40 layers")

    resident = sorted(set(layers) - set(offloaded))
    rows_by_arm = []
    medians = []
    for timeline in timelines:
        rows = timeline["bottleneck"]["layers"]
        if [row["layer"] for row in rows] != layers:
            raise ValueError("Bottleneck-rank rows must match the declared layer IDs")
        for row in rows:
            for key, _ in METRICS:
                if not math.isfinite(row[key]) or row[key] < 0:
                    raise ValueError(f"Invalid {key} for layer {row['layer']}")
        median = timeline["all_rank_graph_envelope_us"]["p50"]
        if not math.isfinite(median) or median <= 0:
            raise ValueError("All-rank Graph envelope medians must be positive and finite")
        rows_by_arm.append(rows)
        medians.append(median)
    return layers, resident, timelines, rows_by_arm, medians


def draw(data):
    layers, resident, timelines, rows_by_arm, medians = checked_pair(data)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(12, 7.5), layout="constrained")
    grid = fig.add_gridspec(4, 2, width_ratios=(3.4, 1.25), height_ratios=(1, 1, 1, 0.35))
    fig.suptitle("Stock vs. T40 decode Graph timing", x=0.03, ha="left", weight="bold", size=17)
    axes = []
    for index, (field, title) in enumerate(METRICS):
        ax = fig.add_subplot(grid[index, 0], sharex=axes[0] if axes else None)
        axes.append(ax)
        for layer in resident:
            ax.axvspan(layer - 0.5, layer + 0.5, facecolor=RESIDENT, linewidth=0)
        for rows, color, style in zip(rows_by_arm, (STOCK, T40), ("-", "--")):
            ax.plot(layers, [row[field] for row in rows], color=color,
                    linestyle=style, linewidth=1.8)
        ax.set_title(title, loc="left", weight="bold", size=11)
        ax.set_ylabel("µs")
        ax.set_xlim(layers[0] - 0.5, layers[-1] + 0.5)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="y", color="#cdd7dc", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if index < len(METRICS) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xticks([*layers[::10], layers[-1]])
            ax.set_xlabel("Model layer ID")

    summary = fig.add_subplot(grid[:2, 1])
    stock_ms, t40_ms = (value / 1000 for value in medians)
    summary.barh(("Stock", "T40"), (stock_ms, t40_ms), height=0.42, color=(STOCK, T40))
    summary.invert_yaxis()
    for position, value in enumerate((stock_ms, t40_ms)):
        summary.annotate(f"{value:.2f}", (value, position), xytext=(5, 0),
                         textcoords="offset points", va="center", weight="bold")
    summary.set_title("All-rank Graph envelope", loc="left", weight="bold", size=11)
    summary.set_xlabel("Median per replay (ms)")
    summary.set_xlim(0, max(stock_ms, t40_ms) * 1.28)
    summary.xaxis.set_major_locator(MaxNLocator(nbins=4))
    summary.grid(axis="x", color="#cdd7dc", linewidth=0.6)
    summary.set_axisbelow(True)
    summary.tick_params(axis="y", length=0)
    summary.spines[["top", "right", "left"]].set_visible(False)

    notes = fig.add_subplot(grid[2, 1])
    notes.set_axis_off()
    notes.text(
        0, 0.95,
        f"Batch {data['batch']} · TP8\n"
        f"Graph replays: {timelines[0]['steps']} stock, {timelines[1]['steps']} T40\n\n"
        "Envelope: first GPU node start to last\n"
        "GPU node end across all eight ranks.",
        va="top", linespacing=1.4, size=9,
    )
    footer = fig.add_subplot(grid[-1, :])
    footer.set_axis_off()
    footer.legend(handles=[
        Line2D([], [], color=STOCK, linewidth=1.8, label="Stock"),
        Line2D([], [], color=T40, linestyle="--", linewidth=1.8, label="T40"),
        Patch(facecolor=RESIDENT, label="20 T40 resident layers"),
    ], loc="upper left", bbox_to_anchor=(0, 1.09), ncol=3, frameon=False, borderaxespad=0)
    footer.text(
        0, 0,
        "Layer values are medians from the slowest rank within each replay. "
        "The first segment starts at that rank’s first kernel. Single profiled pair.",
        size=9, va="bottom",
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate", type=Path, help="Verified reducer JSON")
    parser.add_argument("--output", type=Path, required=True, help="Figure path (.png, .pdf, or .svg)")
    args = parser.parse_args()
    if args.output.suffix.lower() not in {".png", ".pdf", ".svg"}:
        parser.error("Output must end in .png, .pdf, or .svg")
    try:
        with args.aggregate.open(encoding="utf-8") as source:
            figure = draw(json.load(source))
        figure.savefig(args.output, dpi=220)
        plt.close(figure)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

