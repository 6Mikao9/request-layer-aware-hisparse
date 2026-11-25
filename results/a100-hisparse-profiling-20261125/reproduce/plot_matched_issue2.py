"""Plot anonymous equal-cohort B48 Graph profile evidence."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager


ARMS = (
    ("stock", "原生 HiSparse", "#214d80"),
    ("isolated_arena", "独立 arena", "#167d68"),
    ("shared_arena", "共享 arena", "#a84b2f"),
)
EVENTS = (
    ("decoder_us", "完整 decoder 层"),
    ("attention_us", "attention 模块"),
    ("swap_us", "融合换入区间"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("figure", type=Path)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    fontManager.addfont(str(args.font))
    name = FontProperties(fname=str(args.font)).get_name()
    plt.rcParams.update({"font.family": name, "font.size": 11, "axes.unicode_minus": False})
    with gzip.open(args.profile, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    arms = [row for row in ARMS if row[0] in data["arms"]]

    fig, grid = plt.subplots(2, 2, figsize=(12, 7.7), layout="constrained")
    for ax, (key, title) in zip(grid.flat, EVENTS):
        for arm, label, color in arms:
            points = data["arms"][arm]["per_layer_median_us"]
            ax.plot([p["layer"] for p in points], [p[key] / 1000 for p in points],
                    color=color, linewidth=1.8, label=label)
        ax.set(title=title, xlabel="模型层号", ylabel="耗时（ms）", xlim=(0, 59))
        ax.set_xticks([0, 10, 20, 30, 40, 50, 59])
        ax.set_ylim(bottom=0)
    grid[0][0].legend(loc="best", frameon=False)
    ax = grid[1][1]
    for arm, label, color in arms:
        ns = data["arms"][arm]["nsys"]
        points = ns["per_step"]
        ax.plot([p["step"] for p in points], [p["all_rank_span_ms"] for p in points],
                label=label, color=color, marker="o", markersize=5, linewidth=1.8)
    ax.set(title="Nsight Systems · 完整 GPU Graph", xlabel="生成步骤", ylabel="耗时（ms）")
    ax.set_xticks(range(17, 25))
    ax.set_ylim(bottom=0)
    for ax in grid.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dfe5e9")
        ax.set_axisbelow(True)
    fig.suptitle("MiniMax · 同前缀 B48 CUDA Graph 配对\n16–32K · TP8 · 60 层 · 相同热池容量",
                 fontsize=18, fontweight="bold")
    fig.savefig(args.figure, dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
