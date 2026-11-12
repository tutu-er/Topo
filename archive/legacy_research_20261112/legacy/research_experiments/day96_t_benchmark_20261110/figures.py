"""Scientific plots from the independently verified finite one-day benchmark."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("noT", "T003", "T010")
LABELS = {"noT": "No joint tail", "T003": "Joint tail / alpha=.003", "T010": "Joint tail / alpha=.010"}
COLORS = {"noT": "#4477aa", "T003": "#ee7733", "T010": "#228833"}


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, destination, name):
    for suffix in ("png", "svg"):
        fig.savefig(destination / f"{name}.{suffix}", dpi=180)
    plt.close(fig)


def figures(output):
    output = Path(output).resolve()
    receipt = json.loads((output / "analysis/verification.json").read_text(encoding="utf-8"))
    if not receipt.get("passed") or receipt.get("verified") != 144:
        raise ValueError("Final figures require all 144 evaluations independently verified")
    destination = ROOT / "docs" / output.name
    destination.mkdir(parents=True, exist_ok=True)
    summary = {r["method"]: r for r in rows(output / "analysis/summary.csv")}
    xs = np.arange(3)
    colors = [COLORS[m] for m in METHODS]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.), layout="constrained")
    specs = (("F1", 1., "Mean clade F1", ".3f"), ("exact", 1., "Exact topology / 48 inputs", ".0f"),
             ("total_seconds", 1/3600, "Successful-run algorithm time (hours)", ".2f"))
    for ax, (field, factor, ylabel, fmt) in zip(axes, specs):
        values = [float(summary[m][field])*factor for m in METHODS]
        bars = ax.bar(xs, values, color=colors, width=.63)
        ax.bar_label(bars, labels=[format(v, fmt) for v in values], padding=4, fontsize=9)
        ax.set(xticks=xs, xticklabels=["No T", "T .003", "T .010"], ylabel=ylabel)
        ax.set_ylim(0, 1.10 if field == "F1" else (52 if field == "exact" else max(values)*1.18))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
    fig.suptitle("One continuous day / 96 slots: 48 paired inputs per method", fontsize=12)
    save(fig, destination, "overall_comparison")

    noise = rows(output / "analysis/noise.csv")
    sigmas = sorted({float(r["voltage_sigma"]) for r in noise})
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.1), layout="constrained")
    for method in METHODS:
        selected = sorted((r for r in noise if r["method"] == method), key=lambda r: float(r["voltage_sigma"]))
        x = np.array([float(r["voltage_sigma"])*100 for r in selected])
        values = ([float(r["F1"]) for r in selected], [100*float(r["exact"])/float(r["planned"]) for r in selected],
                  [float(r["total_seconds"])/float(r["verified"]) for r in selected])
        for ax, y in zip(axes, values):
            ax.plot(x, y, "o-", label=LABELS[method], color=COLORS[method], lw=1.8, ms=5)
    for ax, ylabel in zip(axes, ("Mean clade F1", "Exact topology recovery (%)", "Mean successful-run time (seconds)")):
        ax.set(xlabel="V/root meter noise SD (%)", ylabel=ylabel, xticks=np.array(sigmas)*100)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.18)
    axes[0].set_ylim(-.02, 1.04); axes[1].set_ylim(-2, 104)
    axes[2].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Noise-stratified paired comparison: 16 inputs per point", fontsize=12)
    save(fig, destination, "noise_comparison")

    fig, ax = plt.subplots(figsize=(9., 4.3), layout="constrained")
    categories = (("raw_pool_truth", "Whole true tree\nin raw own path"),
                  ("output_pool_truth", "Whole true tree\nin contracted/refit path"),
                  ("exit_connection_pool_truth_covered", "All true connections\nin exit MILP pool"))
    x = np.arange(len(categories)); width=.23
    for i, method in enumerate(METHODS):
        values = [float(summary[method][field]) for field, _ in categories]
        bars = ax.bar(x+(i-1)*width, values, width=width, color=COLORS[method], label=LABELS[method])
        ax.bar_label(bars, labels=[str(int(v)) for v in values], padding=3, fontsize=8)
    ax.set(xticks=x, xticklabels=[label for _, label in categories], ylabel="Inputs covered / 48", ylim=(0, 55))
    ax.set_title("Candidate coverage: a complete tree differs from a union of connections", fontsize=11)
    ax.legend(frameon=False, loc="upper center", ncols=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
    save(fig, destination, "candidate_coverage")
    (destination / "figure_receipt.json").write_text(json.dumps(dict(files=[f"{name}.{suffix}" for name in
        ("overall_comparison", "noise_comparison", "candidate_coverage") for suffix in ("png", "svg")],
        verification_sha256=__import__("hashlib").sha256((output / "analysis/verification.json").read_bytes()).hexdigest(),
        visual_review_pending=True, optimizer_calls=0), indent=2)+"\n", encoding="utf-8")
    print("Saved three PNG/SVG figure pairs; visual review pending", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    figures(parser.parse_args().output)
