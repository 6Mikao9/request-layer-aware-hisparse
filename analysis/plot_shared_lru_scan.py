"""Plot per-case resolver timings from benchmark_shared_lru_scan JSON receipts."""
import argparse
import json
import math
import statistics
from pathlib import Path


ARMS = (("stock", "Native", "#444444", "o"),
        ("isolated", "Isolated arena", "#1676ab", "s"),
        ("shared", "Shared arena", "#c0500c", "^"))


def read_cases(paths):
    cases = {}
    environment = None
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("event") != "hisparse_resolver_scan_microbench":
            raise ValueError(f"Unexpected receipt type in {path}")
        env = tuple(doc[k] for k in ("device", "torch", "cuda", "arena_kernel_sha256", "dtype"))
        if environment is not None and environment != env:
            raise ValueError("Input receipts use different devices, software, kernels or data types")
        environment = env
        for row in doc["results"]:
            g = row["geometry"]
            key = (g["slots"], row["miss_blocks"] / g["blocks"], g["layers"], row["arm"])
            if key in cases:
                raise ValueError(f"Multiple measurements of the same case: {key}")
            counts = row["actual_counts_per_launch"]
            total = row["selected_rows_per_request"] * g["slots"]
            misses = g["slots"] * row["miss_blocks"] * g["block"]
            if counts != dict(hits=total - misses, misses=misses):
                raise ValueError(f"Unexpected hit/miss counts: {key}")
            samples = row["kernel_us"]["samples"]
            if not samples or any(not math.isfinite(t) or t <= 0 for t in samples):
                raise ValueError(f"Invalid kernel times: {key}")
            cases[key] = statistics.median(samples)
    if not cases:
        raise ValueError("No measured cases")
    return environment[0], cases


def plot(paths, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    device, cases = read_cases(paths)
    batches = sorted({b for b, _, _, _ in cases})
    fractions = sorted({f for _, f, _, _ in cases})
    layers = sorted({l for _, _, l, _ in cases})
    fig, panels = plt.subplots(len(batches), len(fractions), squeeze=False, sharey=True,
                               figsize=(max(5, 4 * len(fractions)), 2.3 * len(batches) + 1.9))
    for i, batch in enumerate(batches):
        for j, fraction in enumerate(fractions):
            ax = panels[i, j]
            for arm, label, color, marker in ARMS:
                values = [cases.get((batch, fraction, layer, arm), math.nan) for layer in layers]
                ax.plot(layers, values, color=color, marker=marker, label=label, linewidth=1.7)
            ax.set_title(f"Batch {batch}   ·   {fraction:.0%} misses", loc="left", fontsize=10)
            ax.set_yscale("log")
            ax.set_xticks(layers)
            ax.grid(axis="y", alpha=0.23, which="both")
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0:
                ax.set_ylabel("Median time (µs)")
            if i == len(batches) - 1:
                ax.set_xlabel("Layers in arena configuration")
    handles = [Line2D([], [], color=color, marker=marker, label=label)
               for arm, label, color, marker in ARMS if any(k[-1] == arm for k in cases)]
    fig.suptitle(f"CUDA Graph resolver time on {device}", x=0.08, ha="left", fontsize=14, weight="bold")
    fig.legend(handles=handles, ncol=len(handles), frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.036))
    fig.text(0.5, 0.015, "One resolver per replay. Native and isolated resolve one layer-local domain.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0.01, 0.10, 0.99, 0.95))
    with output.open("xb") as handle:
        fig.savefig(handle, format=output.suffix[1:], dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.receipts, args.output)
