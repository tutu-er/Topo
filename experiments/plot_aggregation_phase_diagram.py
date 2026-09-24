"""Plot the aggregation phase diagram from paired on/off sweep outputs.

Reads ``metrics.csv`` and ``paired_aggregation_gain.csv`` produced by
``experiments.run_aggregation_phase_diagram`` (the paired table is rebuilt
from ``metrics.csv`` when absent) and writes:

- ``phase_diagram_heatmap.png``: paired unified-F1 gain (on - off) heatmaps
  over clade size k x noise regime, one panel per data regime;
- ``paired_gain_curves.png``: paired gain curves over k per noise regime and
  data regime, with the flynn16 control as a reference line and the
  backbone/peripheral gain split.

Each figure annotates the theorem's three monotonicity predictions: the
paired gain should increase with clade size k, increase with measurement
noise, and decrease with sample size (sparse >= medium).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.run_aggregation_phase_diagram import _paired_gain_frame
from terminal_case33.utils.io import ensure_dir

NOISE_ORDER = ["low", "nominal", "high"]
REGIME_ORDER = ["sparse", "medium", "rich"]
PREDICTION_TEXT = (
    "定理预测：①配对增益随 clade 尺寸 k 单调增；"
    "②随噪声水平单调增；③随样本量减（sparse ≥ medium）"
)


def _setup_chinese_fonts() -> None:
    """Use installed CJK fonts and render minus signs correctly."""

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _load_paired(input_dirs: list[Path]) -> pd.DataFrame:
    """Load paired gain tables from one or more sweep output directories.

    Multiple directories support the per-case parallel layout (one
    ``--output`` subdirectory per case). The paired table is rebuilt from
    ``metrics.csv`` for any directory that lacks it.
    """

    frames = []
    for input_dir in input_dirs:
        paired_path = input_dir / "paired_aggregation_gain.csv"
        if paired_path.exists():
            frames.append(pd.read_csv(paired_path))
            continue
        metrics_path = input_dir / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"neither {paired_path} nor {metrics_path} exists"
            )
        frame = pd.read_csv(metrics_path)
        frame = frame.loc[frame["status"].eq("ok")].copy()
        frames.append(_paired_gain_frame(frame))
    paired = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if paired.empty:
        raise RuntimeError("no on/off paired conditions found")
    return paired


def _ordered(values: pd.Series, order: list[str]) -> list[str]:
    present = [value for value in order if value in set(values)]
    present.extend(sorted(set(values) - set(present)))
    return present


def _write_heatmap(output: Path, paired: pd.DataFrame) -> Path:
    """Draw k x noise paired-gain heatmaps, one panel per data regime."""

    grid = paired.loc[paired["case"].astype(str).str.startswith("grid")].copy()
    if grid.empty:
        grid = paired.copy()
    regimes = _ordered(grid["data_regime"], REGIME_ORDER)
    noises = _ordered(grid["noise_regime"], NOISE_ORDER)
    ks = sorted(grid["clade_size_k"].unique())
    fig, axes = plt.subplots(
        1,
        len(regimes),
        figsize=(4.6 * len(regimes) + 1.2, 4.2),
        squeeze=False,
        constrained_layout=True,
    )
    summary = (
        grid.groupby(["data_regime", "noise_regime", "clade_size_k"], as_index=False)[
            "unified_f1_gain"
        ]
        .mean()
    )
    vmax = max(float(np.nanmax(np.abs(summary["unified_f1_gain"]))), 0.05)
    image = None
    for axis, regime in zip(axes[0], regimes):
        matrix = np.full((len(ks), len(noises)), np.nan)
        for i, k in enumerate(ks):
            for j, noise in enumerate(noises):
                hit = summary.loc[
                    summary["data_regime"].eq(regime)
                    & summary["noise_regime"].eq(noise)
                    & summary["clade_size_k"].eq(k),
                    "unified_f1_gain",
                ]
                if not hit.empty:
                    matrix[i, j] = float(hit.iloc[0])
        image = axis.imshow(
            matrix,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
            origin="lower",
        )
        for i in range(len(ks)):
            for j in range(len(noises)):
                if np.isfinite(matrix[i, j]):
                    axis.text(
                        j,
                        i,
                        f"{matrix[i, j]:+.3f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )
        axis.set_xticks(range(len(noises)), noises)
        axis.set_yticks(range(len(ks)), [f"k={k}" for k in ks])
        axis.set_xlabel("噪声 regime")
        axis.set_title(f"数据 regime：{regime}")
    axes[0][0].set_ylabel("clade 尺寸 k")
    if image is not None:
        fig.colorbar(image, ax=list(axes[0]), label="配对 F1 增益（on − off）")
    fig.suptitle("聚合配对增益相图（k × 噪声 × 数据 regime）", fontsize=11)
    fig.text(
        0.5,
        -0.04,
        PREDICTION_TEXT,
        ha="center",
        va="top",
        fontsize=9,
    )
    path = output / "phase_diagram_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_gain_curves(output: Path, paired: pd.DataFrame) -> Path:
    """Draw paired-gain curves over k with the control case as reference."""

    grid = paired.loc[paired["case"].astype(str).str.startswith("grid")].copy()
    if grid.empty:
        grid = paired.copy()
    control = paired.loc[~paired["case"].astype(str).str.startswith("grid")]
    regimes = _ordered(grid["data_regime"], REGIME_ORDER)
    noises = _ordered(grid["noise_regime"], NOISE_ORDER)
    markers = {"low": "o", "nominal": "s", "high": "^"}
    fig, axes = plt.subplots(
        2,
        len(regimes),
        figsize=(4.6 * len(regimes), 7.6),
        squeeze=False,
        sharex=True,
        constrained_layout=True,
    )
    for col, regime in enumerate(regimes):
        top = axes[0][col]
        bottom = axes[1][col]
        for noise in noises:
            subset = grid.loc[
                grid["data_regime"].eq(regime) & grid["noise_regime"].eq(noise)
            ]
            if subset.empty:
                continue
            by_k = subset.groupby("clade_size_k", as_index=False).agg(
                gain=("unified_f1_gain", "mean"),
                backbone=("unified_backbone_f1_gain", "mean"),
                peripheral=("unified_peripheral_f1_gain", "mean"),
            )
            top.plot(
                by_k["clade_size_k"],
                by_k["gain"],
                marker=markers.get(noise, "o"),
                label=noise,
            )
            bottom.plot(
                by_k["clade_size_k"],
                by_k["backbone"],
                marker=markers.get(noise, "o"),
                linestyle="-",
                label=f"{noise} backbone",
            )
            bottom.plot(
                by_k["clade_size_k"],
                by_k["peripheral"],
                marker=markers.get(noise, "o"),
                linestyle="--",
                label=f"{noise} peripheral",
            )
        for axis, title in ((top, "整体 F1 配对增益"), (bottom, "backbone / peripheral 配对增益")):
            axis.axhline(0.0, color="0.4", linewidth=0.8)
            if not control.empty:
                control_gain = float(
                    control.loc[
                        control["data_regime"].eq(regime), "unified_f1_gain"
                    ].mean()
                )
                axis.axhline(
                    control_gain,
                    color="green",
                    linewidth=1.0,
                    linestyle=":",
                    label=f"flynn16 对照 ({control_gain:+.3f})",
                )
            axis.set_title(f"{title}（{regime}）", fontsize=10)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
        bottom.set_xlabel("clade 尺寸 k")
        top.set_ylabel("F1 增益（on − off）")
        bottom.set_ylabel("F1 增益（on − off）")
    fig.suptitle("聚合配对增益曲线（整体与 backbone/peripheral 拆分）", fontsize=11)
    fig.text(
        0.5,
        -0.03,
        PREDICTION_TEXT,
        ha="center",
        va="top",
        fontsize=9,
    )
    path = output / "paired_gain_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    """Parse command-line arguments and render the phase-diagram figures."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        default=["outputs/aggregation_phase_diagram"],
        help=(
            "one or more directories with metrics.csv / "
            "paired_aggregation_gain.csv (per-case parallel outputs merge)"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="figure output directory",
    )
    args = parser.parse_args()
    _setup_chinese_fonts()
    input_dirs = [Path(value) for value in args.input]
    output = ensure_dir(args.output)
    paired = _load_paired(input_dirs)
    heatmap = _write_heatmap(output, paired)
    curves = _write_gain_curves(output, paired)
    print(f"wrote {heatmap}")
    print(f"wrote {curves}")


if __name__ == "__main__":
    main()
