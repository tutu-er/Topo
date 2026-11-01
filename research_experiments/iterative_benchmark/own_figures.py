"""Scientific figures for the recorded four-algorithm, own-path comparison.

This module only draws supplied tables. It never loads a solver, fits a model,
selects a topology, or estimates runtime. PNG and vector PDF share one figure.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ALGORITHMS = ("NJ_matched", "pure_wzzt", "ordinary", "current_own")
CASES = ("paper15", "soumalas11", "flynn16", "pengwah18")
LABELS = {
    "NJ_matched": "NJ（同尺度后处理）",
    "pure_wzzt": "纯WZZT",
    "ordinary": "普通RNJ＋WZZT",
    "current_own": "当前主线（无整树备用）",
}
COLORS = {
    "NJ_matched": "#969ba6",
    "pure_wzzt": "#d18f39",
    "ordinary": "#4a9f86",
    "current_own": "#2265a2",
}


def _style():
    path = font_manager.findfont(
        font_manager.FontProperties(family="Microsoft YaHei"),
        fallback_to_default=False,
    )
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=path).get_name(),
        "font.size": 10,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })


def _axis(ax, title, *, percent=False):
    ax.set_title(title, pad=12)
    ax.set_ylim(0, 1.105)
    ax.set_yticks(np.arange(0, 1.01, .2))
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)


def _save(fig, dest, name):
    files = []
    for suffix in ("png", "pdf"):
        filename = f"{name}.{suffix}"
        options = {"dpi": 180} if suffix == "png" else {
            "metadata": {"CreationDate": None, "ModDate": None}
        }
        fig.savefig(dest / filename, **options)
        files.append(filename)
    plt.close(fig)
    return files


def _aggregate(observations, group):
    return observations.groupby(group, sort=False).agg(
        inputs=("exact", "size"), F1=("clade_f1", "mean"),
        exact=("exact", "sum"), rate=("exact", "mean"),
    )


def _overall(dest, observations):
    values = _aggregate(observations, "algorithm").loc[list(ALGORITHMS)]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), layout="constrained")
    x = np.arange(4)
    labels = ["NJ\n同尺度后处理", "纯WZZT", "普通\nRNJ＋WZZT", "当前主线\n无整树备用"]
    for ax, metric, title in zip(axes, ("F1", "rate"), ("平均拓扑F1", "完整恢复率")):
        ax.bar(x, values[metric], color=[COLORS[m] for m in ALGORITHMS], width=.60)
        ax.set_xticks(x, labels)
        _axis(ax, title, percent=metric == "rate")
        for i, row in enumerate(values.itertuples()):
            label = f"{row.F1:.4f}" if metric == "F1" else f"{int(row.exact)}/{row.inputs}\n{row.rate:.1%}"
            ax.text(i, getattr(row, metric) + .025, label, ha="center", va="bottom", color=COLORS[row.Index])
    fig.suptitle("四算法对照：同一批120个输入，第四算法仅保留自身搜索输出", fontsize=14)
    fig.supxlabel("108常规＋12压力输入；历史普通方法的求解预算与精度设置不同", fontsize=9, color="#555555")
    return _save(fig, dest, "four_algorithm_overall")


def _cases(dest, observations):
    values = _aggregate(observations, ["case", "algorithm"])
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.1), layout="constrained")
    x = np.arange(4)
    width = .19
    for j, algorithm in enumerate(ALGORITHMS):
        rows = values.xs(algorithm, level="algorithm").loc[list(CASES)]
        positions = x + (j - 1.5) * width
        for k, metric in enumerate(("F1", "rate")):
            axes[k].bar(positions, rows[metric], width=width, color=COLORS[algorithm], label=LABELS[algorithm])
            if algorithm in ("ordinary", "current_own"):
                for position, row in zip(positions, rows.itertuples()):
                    label = f"{row.F1:.3f}" if metric == "F1" else str(int(row.exact))
                    axes[k].text(position, getattr(row, metric) + .018, label, ha="center", fontsize=8.2, color=COLORS[algorithm])
    for ax, title, percent in zip(axes, ("各网络平均拓扑F1", "各网络完整恢复率（标注恢复个数）"), (False, True)):
        ax.set_xticks(x, CASES)
        _axis(ax, title, percent=percent)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("每个网络30个输入：第四算法F1均更高，完整恢复数存在局部退步", fontsize=14)
    return _save(fig, dest, "four_algorithm_cases")


def _noise(dest, observations):
    main = observations[observations.panel == "main"]
    values = _aggregate(main, ["voltage_sigma", "algorithm"])
    levels = (.0002, .0005, .001)
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), layout="constrained")
    markers = ("s", "^", "o", "D")
    for algorithm, marker in zip(ALGORITHMS, markers):
        rows = values.xs(algorithm, level="algorithm").loc[list(levels)]
        assert rows.inputs.eq(36).all()
        for k, metric in enumerate(("F1", "rate")):
            axes[k].plot(x, rows[metric], marker=marker, markersize=6,
                         linewidth=2.2 if algorithm == "current_own" else 1.8,
                         color=COLORS[algorithm], label=LABELS[algorithm])
            if algorithm in ("ordinary", "current_own"):
                for position, row in zip(x, rows.itertuples()):
                    offset = 12 if algorithm == "current_own" else -22
                    if algorithm == "ordinary" and (k == 0 and position == 2 or k == 1 and row.rate < .1):
                        offset = 12
                    label = f"{row.F1:.3f}" if metric == "F1" else f"{int(row.exact)}/36"
                    axes[k].annotate(label, (position, getattr(row, metric)), xytext=(0, offset),
                                     textcoords="offset points", ha="center", fontsize=9, color=COLORS[algorithm],
                                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": .5})
    for ax, title, percent in zip(axes, ("平均拓扑F1", "完整恢复率"), (False, True)):
        _axis(ax, title, percent=percent)
        ax.set_xticks(x, ("0.02%", "0.05%", "0.10%"))
        ax.set_xlim(-.13, 2.13)
        ax.set_xlabel("终端与根表电压相对噪声")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("噪声分组：第四算法的收益集中在中高噪声条件\n"
                 "每级36常规输入；合并N＝24/48/96；P/Q测量噪声固定0.5%", fontsize=12.5)
    return _save(fig, dest, "four_algorithm_noise")


def _zero_ablation(dest, history):
    row = history["zero_summary"].set_index("scope").loc["all"]
    fig, axes = plt.subplots(1, 3, figsize=(13.1, 4.7), layout="constrained", width_ratios=(1, 1, 1.12))
    x = np.arange(2)
    colors = ("#aeb8c4", COLORS["current_own"])
    values = ([row.F1_before, row.F1_after], [row.exact_before / row.inputs, row.exact_after / row.inputs])
    for k, metric in enumerate(values):
        axes[k].bar(x, metric, color=colors, width=.55)
        axes[k].set_xticks(x, ("无及时清零", "有及时清零"))
        _axis(axes[k], ("平均拓扑F1", "完整恢复率")[k], percent=k == 1)
        labels = [f"{v:.6f}" for v in metric] if k == 0 else [f"{int(row.exact_before)}/{int(row.inputs)}", f"{int(row.exact_after)}/{int(row.inputs)}"]
        for i, (v, label) in enumerate(zip(metric, labels)):
            axes[k].text(i, v + .025, label, ha="center", va="bottom", color=colors[i])
    counts = [row.wins, row.ties, row.losses]
    axes[2].bar(np.arange(3), counts, color=("#4a9f86", "#aeb8c4", "#bd6262"), width=.58)
    axes[2].set_xticks(np.arange(3), ("F1改善", "F1相同", "F1下降"))
    axes[2].set_ylim(0, row.inputs * 1.10)
    axes[2].set_ylabel("输入个数")
    axes[2].set_title("配对变化", pad=12)
    axes[2].grid(axis="y", alpha=.18)
    axes[2].set_axisbelow(True)
    for i, v in enumerate(counts):
        axes[2].text(i, v + row.inputs * .025, str(int(v)), ha="center")
    fig.suptitle("历史消融：及时释放零块（同预算、仅自身路径）", fontsize=14)
    fig.supxlabel("前后均保留联合末端与最终退出；完整RNJ备用和额外恢复分支均未计入", fontsize=9, color="#555555")
    return _save(fig, dest, "historical_zero_ablation")


def _raw_stages(dest, history):
    values = history["stage_summary"]
    values = values[values.scope == "all"].set_index("stage").loc[["raw_prefix", "raw_exit", "final_output"]]
    labels = ("原始退出前", "原始退出后", "收缩＋路径验证输出")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), layout="constrained")
    for k, metric in enumerate(("F1", "recovery_rate")):
        axes[k].plot(np.arange(3), values[metric], marker="o", markersize=7, linewidth=2, color=COLORS["current_own"])
        axes[k].set_xticks(np.arange(3), labels)
        axes[k].set_xlim(-.15, 2.15)
        _axis(axes[k], ("平均拓扑F1", "完整恢复率")[k], percent=k == 1)
        for i, row in enumerate(values.itertuples()):
            label = f"{row.F1:.4f}" if k == 0 else f"{int(row.exact)}/{int(row.inputs)}"
            axes[k].text(i, getattr(row, metric) + .025, label, ha="center", va="bottom", color=COLORS["current_own"])
    fig.suptitle("历史阶段诊断：原始状态与最终输出（非独立模块消融）", fontsize=14)
    fig.supxlabel("同一批120条历史路径；原始状态与最终输出的收缩、验证口径不同", fontsize=9, color="#555555")
    return _save(fig, dest, "historical_stage_diagnostic")


def render_figures(dest: Path, observations: pd.DataFrame, history: dict) -> list[str]:
    """Render five figures from supplied observations; return ten filenames."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    assert set(observations.algorithm) == set(ALGORITHMS)
    assert len(observations) == 480
    assert observations.groupby("algorithm").size().eq(120).all()
    assert not observations.duplicated(["algorithm", "job"]).any()
    _style()
    files = []
    files.extend(_overall(dest, observations))
    files.extend(_cases(dest, observations))
    files.extend(_noise(dest, observations))
    files.extend(_zero_ablation(dest, history))
    files.extend(_raw_stages(dest, history))
    return files
