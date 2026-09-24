"""Postprocess M7 results and audit preserved inputs; no model fitting."""
from pathlib import Path
import hashlib
import json
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import networkx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "theft_wzzt/outputs/theft"
summary = json.loads((OUT / "m7_summary.json").read_text(encoding="utf-8"))
assert summary["status"] == "complete", summary["missing_or_failed"]
assert len(summary["rows"]) == 55
before = json.loads((OUT / "m7_before_manifest.json").read_text(encoding="utf-8"))
planned = {"theft_wzzt\\experiments\\run_theft.py", "theft_wzzt/experiments/run_theft.py"}
changed, preserved = [], []
for rel, digest in before["sha256"].items():
    path = ROOT / rel
    same = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
    (preserved if same else changed).append(rel)
assert set(changed) <= planned, changed
for rel, digest in summary["context"]["sha256"].items():
    assert hashlib.sha256((ROOT / "theft_wzzt" / rel).read_bytes()).hexdigest() == digest, rel
rows = summary["rows"]
for r in rows:
    assert r["status"] == "ok"
    for key in ("h0_diagnostics", "h1_diagnostics", "h0_fixed_diagnostics"):
        d = r[key]
        assert d["status"] == 0 and d["mip_gap"] <= 1e-7
        assert d["max_feasibility_error"] <= 2e-6 and d["max_product_error"] <= 2e-6
    env = np.asarray(r["amplitude_envelope"])
    assert np.all(env[0] <= env[1]) and np.all(env[1] <= env[2])
    assert r["balance_identity_max_error"] < 1e-10
baseline_checks = {}
for exp in ("exp2", "exp3"):
    old = json.loads((OUT / f"{exp}.json").read_text(encoding="utf-8"))
    new = next(r for r in rows if r["loss_model"] == "L0" and r["experiment"] == exp)
    diff = new["gain"] - old["gain"]
    assert abs(diff) < 1e-5
    baseline_checks[exp] = diff
original_null = np.array(summary["baseline_summary"]["null_gains"])
rank_comparison = []
for group in summary["groups"]:
    ng = np.array(group["null_gains"])
    ranks = [(1 + np.count_nonzero(original_null >= gain - 1e-7)) / (len(original_null) + 1)
             for gain in ng]
    rank_comparison.append({"loss_model": group["loss_model"], "loss_bias": group["loss_bias"],
                            "q95_false_alarms": group["false_alarms_fixed"],
                            "rank_005_false_alarms": int(np.count_nonzero(np.array(ranks) <= .05)),
                            "null_n": len(ng), "original_calibrated_ranks": ranks})
validation = dict(summary["validation"], original_rank_rule_comparison=rank_comparison,
                  preserved_original_count=len(preserved),
                  planned_original_changes=changed, preserved_paths=preserved,
                  original_gain_differences=baseline_checks,
                  summed_unique_run_seconds=sum(r["seconds_total"] for r in rows if "reused_from" not in r),
                  runtime={"python": sys.executable, "version": sys.version, "platform": platform.platform(),
                           "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
                           "networkx": networkx.__version__})
(OUT / "m7_validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")

plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
groups = [g for g in summary["groups"] if g["loss_model"] == "L2"]
bias = np.array([g["loss_bias"] for g in groups]) * 100
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), layout="constrained")
ax = axes[0, 0]
fpr = np.array([g["false_positive_fraction_fixed"] for g in groups]) * 100
ax.plot(bias, fpr, "o-", color="#b4473a", lw=2)
for b, f, g in zip(bias, fpr, groups):
    ax.annotate(f"{g['false_alarms_fixed']}/5", (b, f), xytext=(0, 7), textcoords="offset points", ha="center")
ax.set(title="固定原阈值下的 null 误报", ylabel="样本误报比例（%）", ylim=(-5, 115))
colors = {"exp2": "#246b8e", "exp3": "#bb711b"}
labels = {"exp2": "exp2：内部节点", "exp3": "exp3：末端节点"}
for exp in ("exp2", "exp3"):
    events = [next(r for r in g["events"] if r["experiment"] == exp) for g in groups]
    for ax, key, scale in ((axes[0, 1], "gain_margin_fixed", 1),
                           (axes[1, 0], "amplitude_active_relative_mae", 100),
                           (axes[1, 1], "active_time_region_coverage", 100)):
        ax.plot(bias, [r[key] * scale for r in events], "o-", color=colors[exp], label=labels[exp], lw=2)
axes[0, 1].set(title="事件告警余量", ylabel="gain − 5.455")
axes[0, 1].axhline(0, color="#777777", ls="--", lw=1)
axes[0, 1].legend(frameon=False)
axes[1, 0].set(title="活动时段幅值误差", ylabel="相对 MAE（%）")
axes[1, 1].set(title="活动时段区域覆盖", ylabel="6 个活动时刻的覆盖比例（%）", ylim=(-5, 105))
for ax in axes.flat:
    ax.set_xlabel("线损偏置 β（%，同时扰动 P/Q）")
    ax.set_xticks(bias)
    ax.grid(alpha=.18)
fig.suptitle("M7：线损系统偏置敏感性（paper15，辨识树）\n每点 5 个独立 null；每类事件 1 个复制，仅作探索性证据", fontsize=14)
fig.savefig(OUT / "m7_sensitivity.png", dpi=180)
fig.savefig(OUT / "m7_sensitivity.svg")
plt.close(fig)

l1 = next(g for g in summary["groups"] if g["loss_model"] == "L1")
l0 = next(g for g in summary["groups"] if g["loss_model"] == "L0")
l1null = [r for r in rows if r["loss_model"] == "L1" and r["experiment"] == "null"]
lines = ["# M7 执行结果与结论", "", "日期：2026-09-19。详见同日审查记录中的数学边界与实现说明。", "",
         f"已完成 55 个条件记录，其中 7 个零偏置条件复用 L0，共 48 个独立 H0/H1 比较及 48 次固定 R/X 的 H0 诊断。",
         f"原固定 q95={summary['fixed_q95']:.9f}；新 null 使用复制 30–34，与原校准复制 10–29 分离。", "",
         "## 主要观察", "",
         f"- L0 与 L1 的独立 null 均为 {l0['false_alarms_fixed']}/5、{l1['false_alarms_fixed']}/5 误报。",
         f"- L1 在 5 个 null 上的线损相对 L0 的 MAE 中位数为 {np.median([r['loss_relative_mae_vs_notheft'] for r in l1null]):.2%}，这是整体近似误差，不能只解释为参数误差。"]
for e in l1["events"]:
    lines.append(f"- L1 {e['experiment']}：gain={e['gain']:.3f}，告警={e['alarm_fixed']}，区域集合覆盖={e['location_covered']}，"
                 f"逐活动时刻区域覆盖={e['active_time_region_coverage']:.2%}，幅值相对 MAE={e['amplitude_active_relative_mae']:.2%}。")
lines += ["", "## 幅值偏差分解", "",
          "下表为活动时段平均相对**有符号偏差**的分量，各列之和等于平均幅值偏差；不等于相对 MAE。",
          "此处活动时段未发生零截断，真值仅用于事后解释。", "",
          "|估计器/事件|偷电增量线损|L0−估计线损|表计合成误差|合计|",
          "|---|---:|---:|---:|---:|"]
for r in rows:
    if r["loss_model"] in ("L0", "L1") and r["experiment"] != "null":
        truth = np.array(r["amplitude_truth_pu"])
        active = truth > 0
        actual_loss = np.array(r["loss_p_actual_eval"])
        reference = np.array(r["loss_p_notheft_eval"])
        estimate = np.array(r["loss_p_estimate"])
        meter_error = np.array(r["meter_error_eval"])
        parts = [float(np.mean(v[active] / truth[active]))
                 for v in (actual_loss-reference, reference-estimate, meter_error)]
        observed_bias = np.mean((np.array(r["amplitude_envelope"])[1, active] - truth[active]) / truth[active])
        assert abs(sum(parts) - observed_bias) < 1e-8
        lines.append(f"|{r['loss_model']}/{r['experiment']}|" + "|".join(f"{v:.3%}" for v in parts + [sum(parts)]) + "|")
lines += ["", "## 系统偏置结果", "",
          "|β|固定阈值误报数|变体 q95|变体回代误报比例|exp2 告警 / 区域集合覆盖|exp3 告警 / 区域集合覆盖|",
          "|---:|---:|---:|---:|---|---|"]
for g in groups:
    events = {r["experiment"]: r for r in g["events"]}
    lines.append(f"|{g['loss_bias']:+.0%}|{g['false_alarms_fixed']}/5|{g['variant_q95_exploratory']:.3f}|"
                 f"{g['variant_q95_resubstitution_fpr']:.0%}|"
                 + "|".join(f"{events[e]['alarm_fixed']} / {events[e]['location_covered']}" for e in ("exp2", "exp3")) + "|")
lines += ["", "附带比较原经验秩规则 rank≤0.05（不改变上述 q95 主报告规则）：", "",
          "|β|q95 误报数|秩规则误报数|",
          "|---:|---:|---:|"]
for g in rank_comparison:
    if g["loss_model"] == "L2":
        lines.append(f"|{g['loss_bias']:+.0%}|{g['q95_false_alarms']}/{g['null_n']}|{g['rank_005_false_alarms']}/{g['null_n']}|")
lines += ["", "变体 q95 与回代误报比例使用同一批 5 个 null，只有描述性意义，不能解释为独立重校准效果。",
          "", "![M7 敏感性曲线](D:/0-github_workspace/Topo/theft_wzzt/outputs/theft/m7_sensitivity.png)", "",
          "## 验证与保护", "",
          "- 单测 16/16 通过（原 12 项未改，新增 4 项），60.73 秒。首次新增测试导入冲突已经修复。",
          f"- 完成的所有求解均通过原模型全局最优状态及直接可行性/乘积/目标校验；最大 MIP gap={validation['max_mip_gap']:.3g}，"
          f"最大可行性误差={validation['max_feasibility_error']:.3g}，最大乘积误差={validation['max_product_error']:.3g}。",
          f"- 带符号平衡误差分解最大误差={validation['max_balance_identity_error']:.3g}。",
          f"- H0 重估收益相对 L0 的最大变化仅 {validation['max_abs_h0_refit_benefit_change']:.3g}；与其不依赖线损输入的代数分析一致，不能称为线损误差吸收率。",
          f"- {len(preserved)} 个受保护原文件哈希不变；原有文件中仅计划内的 run_theft.py 修改。原输出、辨识树缓存、MILP、原单测及 pilot 四文件保留。",
          "- L0 的 exp2/exp3 gain 与原输出差值均小于 1e-5。新增结果全部使用 m7_ 前缀，没有覆盖原 summary。",
          f"- 唯一条件的记录耗时合计 {validation['summed_unique_run_seconds']/60:.2f} 分钟（不含单测、编码和报告时间）。", "",
          "## 尚不能得出的结论", "",
          "不能由每点 5 个 null 与每类 1 个事件推出总体误报率、漏报率或可接受的最大线损误差；不能把零误报网格区间认定为现场安全范围。",
          "±25% 包络不是置信区间；其真值覆盖仅在数据中实测，不能由定义自动保证。区域覆盖允许祖先区域，不能解释为精确定位到真实物理节点或用户。",
          "本轮没有实施 exp5/exp6 交互实验、外层迭代、蒙特卡洛幅值区间或大样本独立重校准。", "",
          "## 详细产物", "",
          "- `theft_wzzt/outputs/theft/m7_report.md`：所有估计器的幅值与定位表。",
          "- `theft_wzzt/outputs/theft/m7_summary.json`：55 个条件及逐时包络与求解诊断。",
          "- `theft_wzzt/outputs/theft/m7_validation.json`：保护检查、数值校验、运行库版本。",
          "- `theft_wzzt/outputs/theft/m7_run_manifest.json`：输入和代码签名。",
          "- `scripts/m7_analysis_20260919.py`：本报告与图表的可复现后处理。"]
(ROOT / "docs/theft_m7_results_20260919.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({k:v for k,v in validation.items() if k != "preserved_paths"}, indent=2, ensure_ascii=False))
