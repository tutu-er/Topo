"""Scientific plots of complete R/T/E ablations and module-addition effects."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


FACTORS = ("R", "T", "E")
BITS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
COLORS = ("#a4acb8", "#4c9a84", "#cc983f", "#8a78aa", "#5c9cb3", "#5c83a3", "#a87c79", "#2265a2")


def style():
    font = font_manager.findfont(font_manager.FontProperties(family="Microsoft YaHei"), fallback_to_default=False)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(),
        "font.size": 10, "axes.unicode_minus": False, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, dest, name):
    result = []
    for suffix in ("png", "svg"):
        filename = name + "." + suffix
        fig.savefig(dest / filename, **({"dpi": 180} if suffix == "png" else {}))
        result.append(filename)
    plt.close(fig)
    return result


def mapping(protocol):
    return {tuple(int(bool(m[k])) for k in ("seed", "tail", "exit")): m["id"] for m in protocol["methods"]}


def overall(dest, tables, protocol):
    methods = mapping(protocol)
    values = tables["summary"].set_index("method").loc[[methods[b] for b in BITS]]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1), layout="constrained")
    x = np.arange(8)
    labels = ["基座\n000", "+R\n100", "+T\n010", "+E\n001", "+RT\n110", "+RE\n101", "+TE\n011", "完整\n111"]
    complete = values.complete.astype(bool).to_numpy()
    colors = [COLORS[i] if complete[i] else "#d3d6da" for i in range(8)]
    items = (("F1", "平均拓扑F1", False), ("exact_fraction_planned", "完整恢复数／计划输入数", True))
    for k, (metric, title, percent) in enumerate(items):
        ax = axes[k]
        ax.bar(x, values[metric].fillna(0.), color=colors, width=.65)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.16)
        ax.set_yticks(np.arange(0, 1.01, .2))
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(title, pad=12)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        for i, row in enumerate(values.itertuples()):
            if row.verified:
                label = f"{row.F1:.3f}" if k == 0 else f"{int(row.exact)}/{int(row.planned)}"
            else:
                label = "待完成"
            if not row.complete:
                label += f"\n核验{int(row.verified)}/{int(row.planned)}"
            ax.text(i, (getattr(row, metric) if row.verified else 0.) + .025, label, ha="center", va="bottom", fontsize=8)
    ax = axes[2]
    hours = values.total_seconds.fillna(0.) / 3600
    ax.bar(x, hours, color=colors, width=.65)
    ax.set_xticks(x, labels)
    ax.set_ylabel("小时（逐输入算法墙钟之和）")
    ax.set_title("累计已记录算法时间", pad=12)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    top = max(float(hours.max()), .1)
    ax.set_ylim(0, top * 1.22)
    for i, (row, value) in enumerate(zip(values.itertuples(), hours)):
        label = f"{value:.2f}" if row.verified else "待完成"
        ax.text(i, value + top * .025, label, ha="center", va="bottom", fontsize=8.5)
    all_complete = bool(complete.all())
    title = "完整因子消融：相同输入、预算与输出规则" if all_complete else "消融运行快照：灰色为未完成，保留计划分母"
    fig.suptitle(title, fontsize=14)
    fig.supxlabel("R＝RNJ结构指导；T＝≤5块联合末端；E＝最终退出；及时清零固定开启，无整树备用择优", fontsize=9)
    return save(fig, dest, "factorial_overall")


def marginal_heatmap(dest, tables, protocol):
    frame = tables["marginal_effects"]
    if frame.empty:
        return []
    contexts = {"R": ("T0E0", "T1E0", "T0E1", "T1E1"),
                "T": ("R0E0", "R1E0", "R0E1", "R1E1"),
                "E": ("R0T0", "R1T0", "R0T1", "R1T1")}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), layout="constrained")
    for ax, metric, title in zip(axes, ("delta_F1", "delta_recovery_rate"), ("加入模块的平均F1变化", "加入模块的完整恢复率变化")):
        matrix = np.full((3, 4), np.nan)
        counts = np.zeros((3, 4), int)
        for i, factor in enumerate(FACTORS):
            for j, context in enumerate(contexts[factor]):
                selected = frame[(frame.factor == factor) & (frame.context == context)]
                if len(selected):
                    matrix[i, j] = selected.iloc[0][metric]
                    counts[i, j] = int(selected.iloc[0].paired_inputs)
        bound = max(float(np.nanmax(abs(matrix))), .001)
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
        ax.set_yticks(np.arange(3), ("加入R", "加入T", "加入E"))
        ax.set_xticks(np.arange(4), ("其余全关", "第一个开启", "第二个开启", "其余全开"))
        ax.set_title(title, pad=12)
        for i, factor in enumerate(FACTORS):
            for j, context in enumerate(contexts[factor]):
                if counts[i, j]:
                    value = matrix[i, j]
                    delta = f"{value:+.4f}" if metric == "delta_F1" else f"{value*100:+.2f}个百分点"
                    label = f"{context}\n{delta}\n配对{counts[i,j]}/{len(protocol['jobs'])}"
                    color = "white" if abs(value) > .55 * bound else "#202020"
                else:
                    label, color = f"{context}\n待配对", "#555555"
                ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=.84, pad=.02)
    fig.suptitle("模块收益取决于已有模块：固定其余开关的成对对照", fontsize=14)
    return save(fig, dest, "conditional_marginals")


def addition_orders(dest, tables, protocol):
    frame = tables["addition_orders"]
    if frame.empty:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.1), layout="constrained")
    colors = plt.get_cmap("tab10").colors
    for index, (order, group) in enumerate(frame.groupby("order", sort=False)):
        group = group.sort_values("step")
        for k, metric in enumerate(("F1", "recovery_rate")):
            axes[k].plot(group.step, group[metric], marker="o", markersize=4.5, linewidth=1.6, color=colors[index], label=order)
    for k, ax in enumerate(axes):
        ax.set_xticks(range(4), ("基座", "加入第1项", "加入第2项", "全部加入"))
        ax.set_ylim(0, 1.105)
        ax.set_yticks(np.arange(0, 1.01, .2))
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.set_title(("平均拓扑F1", "完整恢复率")[k], pad=12)
        if k:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False, fontsize=9)
    paired = int(frame.paired_inputs.min())
    fig.suptitle(f"六种模块加入顺序：完整八配置配对{paired}/{len(protocol['jobs'])}输入", fontsize=14)
    fig.supxlabel("这里改变实验加入顺序，单次算法内部仍按RNJ预选→末端→退出执行", fontsize=9)
    return save(fig, dest, "module_addition_orders")


def shapley(dest, tables, protocol):
    frame = tables["shapley"]
    if frame.empty:
        return []
    frame = frame.set_index("factor").loc[list(FACTORS)]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.8), layout="constrained")
    for ax, metric, scale, title in zip(axes, ("shapley_F1", "shapley_exact", "shapley_seconds"), (1., 1., 1/3600),
                                      ("平均F1贡献", "净完整恢复个数贡献", "累计耗时贡献（小时）")):
        values = frame[metric] * scale
        ax.bar(np.arange(3), values, color=("#4c9a84", "#cc983f", "#2265a2"), width=.58)
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.set_xticks(np.arange(3), FACTORS)
        ax.set_title(title, pad=12)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        top = max(float(abs(values).max()), .001)
        ax.set_ylim(min(float(values.min()), 0.)-top*.25, max(float(values.max()), 0.)+top*.25)
        for i, value in enumerate(values):
            label = f"{value:+.4f}" if metric == "shapley_F1" else f"{value:+.2f}"
            ax.text(i, value + (top * .04 if value >= 0 else -top * .04), label,
                    ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
    paired = int(frame.paired_inputs.min())
    fig.suptitle(f"六种加入顺序的平均边际贡献：配对{paired}/{len(protocol['jobs'])}输入", fontsize=14)
    fig.supxlabel("三个模块贡献相加等于完整组合减去基座；允许负贡献与负耗时（节省）", fontsize=9)
    return save(fig, dest, "shapley_contributions")


def render_figures(dest, tables, protocol):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    style()
    files = []
    for routine in (overall, marginal_heatmap, addition_orders, shapley):
        files.extend(routine(dest, tables, protocol))
    return files
