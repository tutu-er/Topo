"""Public diagnostic tables and standard static figures; raw scales stay private."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ... import io


def group_name(row):
    return row["cohort"]+"/"+row["method"]


def summarize(rows):
    valid = [r for r in rows if r["status"] == "complete"]
    covered = [r for r in valid if r["truth_covered"]]
    ambiguous = [r for r in valid if r["near_pair_003"]]
    result = dict(planned=len(rows), complete=len(valid), failures=len(rows)-len(valid),
        correct=sum(r["map_correct"] for r in valid), native_correct=sum(r["native_correct"] for r in valid),
        truth_covered=len(covered), missing_truth=len(valid)-len(covered),
        covered_but_wrong=sum(not r["map_correct"] for r in covered),
        posterior_changes_native=sum(not r["map_matches_native"] for r in valid),
        single_topology=sum(r["distinct_topologies"] == 1 for r in valid),
        mean_top_probability=float(np.mean([r["p_map"] for r in valid])) if valid else None,
        accuracy=sum(r["map_correct"] for r in valid)/len(valid) if valid else None,
        mean_true_probability=float(np.mean([r["p_true"] for r in valid])) if valid else None,
        mean_brier=float(np.mean([r["brier"] for r in valid])) if valid else None,
        mean_negative_log_true_given_covered=float(np.mean([-r["log_p_true"] for r in covered])) if covered else None,
        mean_negative_log_true_all="infinite: truth absent from closed library" if len(covered)<len(valid) else
            float(np.mean([-r["log_p_true"] for r in valid])) if valid else None,
        max_independent_probability_error=max((r["max_probability_error"] for r in valid), default=None),
        near_pair_003=dict(count=len(ambiguous), correct=sum(r["map_correct"] for r in ambiguous),
            mean_top_probability=float(np.mean([r["p_map"] for r in ambiguous])) if ambiguous else None,
            mean_pair_probability=float(np.mean([r["pair_probability"] for r in ambiguous])) if ambiguous else None))
    result["confidence_gates"] = {}
    for gate in (.9, .95, .99):
        subset = [r for r in valid if r["p_map"] >= gate]
        result["confidence_gates"][str(gate)] = dict(count=len(subset),
            correct=sum(r["map_correct"] for r in subset), wrong=sum(not r["map_correct"] for r in subset),
            missing_truth=sum(not r["truth_covered"] for r in subset),
            precision=sum(r["map_correct"] for r in subset)/len(subset) if subset else None)
    return result


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(rows)


def publish(out, protocol, rows):
    public = out/"public"; public.mkdir()
    names = sorted({group_name(r) for r in rows})
    groups = {name: summarize([r for r in rows if group_name(r) == name]) for name in names}
    summary = dict(status="complete", groups=groups,
        primary_combined=summarize([r for r in rows if r["cohort"] == "full48"]),
        supplement=summarize([r for r in rows if r["cohort"] == "strict96"]),
        scope=protocol["rules"]["scope"], fits=0, optimizers=0, new_observations=0,
        warning="Model-assumption plug-in weights in a finite frozen library; no empirical confidence guarantee")
    io.write(public/"summary.json", summary)
    write_csv(public/"per_library.csv", rows)
    gates = [dict(group=name, gate=gate, **record) for name, value in groups.items()
             for gate, record in value["confidence_gates"].items()]
    write_csv(public/"confidence_gates.csv", gates)
    bins = []
    edges = protocol["rules"]["reliability_edges"]
    for name in names:
        valid = [r for r in rows if r["status"] == "complete" and group_name(r) == name]
        for lo, hi in zip(edges[:-1], edges[1:]):
            subset = [r for r in valid if lo <= r["p_map"] and (r["p_map"] < hi or hi == 1.)]
            bins.append(dict(group=name, lower=lo, upper=hi, count=len(subset),
                confidence=float(np.mean([r["p_map"] for r in subset])) if subset else None,
                accuracy=float(np.mean([r["map_correct"] for r in subset])) if subset else None,
                wrong=sum(not r["map_correct"] for r in subset),
                missing_truth=sum(not r["truth_covered"] for r in subset)))
    write_csv(public/"reliability_bins.csv", bins)
    write_csv(public/"near_pairs.csv", [r for r in rows if r.get("near_pair_003")])
    io.write(public/"PROTOCOL.json", dict(rules=protocol["rules"], expected=protocol["expected"],
        source_code_commit_at_lock=protocol["code_commit_at_lock"], environment=protocol["environment"],
        inputs=[dict(id=r["id"], cohort=r["cohort"], method=r["method"],
                result_sha256=r["result"]["sha256"],
                observed_sha256=r.get("observed", r.get("validation"))["sha256"]) for r in protocol["requests"]]))
    figures(public, names, groups, bins)
    report(public, summary, rows)


def figures(public, names, groups, bins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, len(names), figsize=(11, 3.6), squeeze=False)
    for ax, name in zip(axes[0], names):
        data = [r for r in bins if r["group"] == name and r["count"]]
        ax.plot([0, 1], [0, 1], "--", color=".65", label="identity")
        ax.scatter([r["confidence"] for r in data], [r["accuracy"] for r in data], s=60, color="#28608e")
        for r in data:
            ax.annotate("n="+str(r["count"]), (r["confidence"], r["accuracy"]),
                        xytext=(-8, 9), textcoords="offset points", ha="right", fontsize=8)
        ax.set(xlim=(-.03, 1.03), ylim=(-.03, 1.07), title=name,
               xlabel="Mean maximum plug-in weight", ylabel="Observed MAP accuracy")
        ax.grid(alpha=.2)
    fig.suptitle("Descriptive confidence bins on completed histories", fontsize=12)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(public/("reliability."+suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(names)); correct = []; wrong = []; missing = []
    for name in names:
        gate = groups[name]["confidence_gates"]["0.9"]
        correct.append(gate["correct"]); missing.append(gate["missing_truth"])
        wrong.append(gate["wrong"]-gate["missing_truth"])
    ax.bar(x, correct, label="Correct", color="#43846d")
    ax.bar(x, missing, bottom=correct, label="Wrong: truth absent", color="#c45850")
    ax.bar(x, wrong, bottom=np.array(correct)+missing, label="Wrong: truth in bank", color="#dca23f")
    ax.set(xticks=x, xticklabels=names, ylabel="Frozen libraries", title="Outcomes when maximum weight is at least 90%")
    ax.legend(loc="upper right", frameon=False); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(public/("high_confidence_outcomes."+suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


def report(public, summary, rows):
    groups = summary["groups"]
    lines = ["# 共同残差尺度的后验权重：既有算例验证", "",
        "2026-11-12。只对已完成的冻结候选和原验证观测重新评分；没有重拟合、结构求解、实际新日或 DR。", "",
        "协议锁定 `sigma²=min MSE`、`s=2 sigma²/N`，均匀拓扑先验；不按结果调温度。N 为所有节点×验证时槽的标量残差数。", "",
        "同一活动拓扑保存多个参数点时，按训练 MSE 最小、原序号择一；所有点保留在私有审计中，不在验证集重拟合。另核验本次 MAP 与原整库选择的差异。", "",
        "## 实际结果", "",
        "| 数据／方法 | 完成／计划 | MAP完整恢复 | 库含真拓扑 | 最高权重≥90%：正确／入选 | 错误高置信 |", "|---|---:|---:|---:|---:|---:|"]
    for name, g in groups.items():
        gate = g["confidence_gates"]["0.9"]
        lines.append(f"| {name} | {g['complete']}/{g['planned']} | {g['correct']}/{g['planned']} | {g['truth_covered']}/{g['complete']} | {gate['correct']}/{gate['count']} | {gate['wrong']} |")
    lines += ["", "历史主组每例训练96＋验证288，48输入来自四网络、四组种子、三档电压噪声；两方法复用同一观测。补充组只有四例，总96点、前64训练后32验证，固定使用原相对门槛0的已完成结果。两组分开报告。", "",
        "## 与原MSE选择及候选覆盖的关系", ""]
    for name, g in groups.items():
        lines.append(f"- {name}：MAP与原选择拓扑不同 {g['posterior_changes_native']} 例；库含真但MAP选错 {g['covered_but_wrong']} 例；库缺真 {g['missing_truth']} 例；只有一个不同拓扑 {g['single_topology']} 例。")
    lines += ["", "共同正尺度和均匀先验使MAP仍按MSE排序。验证对象是权重的数值与解释，不预期仅靠归一化提高恢复率。库缺真时真实拓扑权重为0，其无条件对数评分为无穷差；未用裁剪抹掉这一项。", "",
        "## 原.003难区分门槛", ""]
    for name, g in groups.items():
        a = g["near_pair_003"]
        lines.append(f"- {name}：原.003相对门槛内 {a['count']} 例，其中MAP正确 {a['correct']} 例；平均整库最高权重 {a['mean_top_probability']}，前两者单独归一化平均权重 {a['mean_pair_probability']}。")
    lines += ["", "该门槛用于描述既有难区分候选，不修改本次后验公式，也不筛除不利样本。单对概率只条件于前两者，与整库概率分开。", "",
        "## 已核验与解释范围", "",
        "独立检查逐支持重建P/Q预测，并用未平移高斯对数似然重算权重；另做解析二模型、相同损失、单位缩放、样本累积、极端数值和零方差检查。实际误差和测试记录见验收文件。", "",
        "这些是已研究历史的回溯诊断，不是独立概率校准。节点／时槽全部独立是本次刻意采用的简单假设；最高权重不是现场准确率保证。真拓扑标签仅在所有权重及源码／输入哈希封存后进入评价。活动large96批次没有纳入。", "",
        "原参数、MSE、方差、完整点表及逐项审计留在私有outputs；公开表只给候选数、概率及完整拓扑恢复指标。失败和零方差保留计划分母，未加任意噪声下限。", "",
        "![置信度分组](reliability.png)", "", "![高权重结果](high_confidence_outcomes.png)", ""]
    (public/"REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
