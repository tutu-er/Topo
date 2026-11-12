"""Readable conclusions and saved-trajectory diagnostics; no new fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ORDER = ("R0T0E0", "R1T0E0", "R0T1E0", "R0T0E1", "R1T1E0", "R1T0E1", "R0T1E1", "R1T1E1")
NAMES = ("WZZT基座", "+RNJ结构指导", "+联合末端", "+最终退出", "+RNJ＋联合末端",
         "+RNJ＋最终退出", "+联合末端＋最终退出", "全部模块（原主线）")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def markdown(frame):
    def cell(v):
        return f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v)
    return "\n".join(["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns),
                      *["| " + " | ".join(cell(v) for v in row) + " |"
                        for row in frame.itertuples(index=False, name=None)]])


def summarize(output):
    from .diagnostics import collect_diagnostics

    output = Path(output).resolve()
    source, delivery = output / "analysis", ROOT / "docs" / output.name
    receipt = read(source / "verification.json")
    assert receipt["passed"] and receipt["verified"] == 960 and receipt["pending"] == 0
    frame = pd.read_csv(source / "per_model.csv")
    summary = pd.read_csv(source / "summary.csv").set_index("method").loc[list(ORDER)]
    assert len(frame) == 960 and frame.status.eq("verified").all()
    assert frame.groupby("method").size().eq(120).all()
    display = pd.DataFrame({"配置": NAMES, "R/T/E": [m.replace("R", "").replace("T", "").replace("E", "") for m in ORDER],
                            "F1": summary.F1.to_numpy(), "完整恢复": [f"{int(v)}/120" for v in summary.exact],
                            "恢复率": [f"{v:.2%}" for v in summary.recovery_rate_completed],
                            "累计小时": summary.total_seconds.to_numpy() / 3600})
    display.to_csv(delivery / "readable_summary.csv", index=False, lineterminator="\n")
    diagnostics = collect_diagnostics(output)
    for name, value in diagnostics.items():
        if isinstance(value, pd.DataFrame):
            for folder in (source, delivery):
                value.to_csv(folder / (name + ".csv"), index=False, lineterminator="\n")
    write(delivery / "diagnostic_verification.json", diagnostics["metadata"])
    shared = frame.groupby("job").common_seconds.first().sum()
    assert (frame.groupby("job").common_seconds.max() - frame.groupby("job").common_seconds.min()).abs().max() < 1e-10
    cost = dict(planned_evaluations=960, input_conditions=120, nominal_cap_method_hours=3840,
                summed_method_seconds=float(frame.total_seconds.sum()), fresh_seconds=float(frame.fresh_seconds.sum()),
                shared_frontend_once_seconds=float(shared), unique_component_seconds=float(frame.fresh_seconds.sum() + shared),
                manager_wall_seconds=float(read(output / "state.json")["batch_wall_seconds"]),
                wall_scope="manager scheduler excludes freeze and two prior integration runs; those runs are included in method costs")
    write(delivery / "cost_accounting.json", cost)
    assert abs(cost["summed_method_seconds"] - cost["unique_component_seconds"] - 7 * shared) < 1e-6
    paired = pd.read_csv(source / "marginal_inputs.csv")
    t = paired[(paired.factor == "T") & (paired.context == "R1E1")]
    assert len(t) == 120 and int(t.new_exact.sum()) == 0 and int(t.lost_exact.sum()) == 6
    shapley = pd.read_csv(source / "shapley.csv").set_index("factor").loc[["R", "T", "E"]]
    assert abs(shapley.shapley_F1.sum() - (summary.loc["R1T1E1", "F1"] - summary.loc["R0T0E0", "F1"])) < 1e-12
    phi = pd.DataFrame({"模块": ["R：RNJ结构指导", "T：联合末端", "E：最终退出"],
                        "平均F1贡献": shapley.shapley_F1.to_numpy(),
                        "净完整恢复贡献": shapley.shapley_exact.to_numpy(),
                        "累计小时贡献": shapley.shapley_seconds.to_numpy() / 3600})
    cases = pd.read_csv(source / "cases.csv")
    cases = cases[cases.method.isin(("R1T0E1", "R1T1E1"))][["case", "method", "F1", "exact", "total_seconds"]]
    lost = diagnostics["lost_exact_cases"]
    assert len(lost) == 6
    text = "# 完整消融结论：RNJ结构指导与最终退出互补，独立联合末端可以省去\n\n"
    text += "已完成同一批120输入×8组合=960次新配置运行。960/960通过核验，5972次MILP均达到设定精度，313个独立固定家族LP、30567项检查通过；失败、无效、超时及预算超限均为0。全模块输出120/120复现此前无RNJ整树备用的主线。\n\n"
    text += "R是RNJ结构指导（少量种子、几何候选池与热启动），T是≤5直属块独立联合末端，E是一次最终结构退出。及时清零固定开启，所有组合使用相同惩罚、收缩和自身路径验证规则；没有完整RNJ树输出备用。公共QP标定与标量f0保持固定，因此基座是公共标定下的WZZT，不等同于早期无惩罚纯WZZT。\n\n"
    text += "## 八组合完整成绩\n\n" + markdown(display) + "\n\n![八组合](factorial_overall.png)\n\n"
    text += "本批最高F1与完整恢复率来自R＋E：F1=0.828022，66/120（55%），累计7.7349小时；全部模块为0.821755，60/120（50%），7.3570小时。去掉独立T阶段净增6个完整恢复、F1增加0.006266，增加累计22.67分钟（相对全模块约5.14%）。这是本批固定配置下的质量/成本取舍，没有自动更改运行默认。\n\n"
    text += "## 每个模块的条件作用\n\n![条件边际](conditional_marginals.png)\n\n"
    text += "R在四种上下文中均提高平均F1；其收益在存在E时更大。E在没有R时只净增0或1个完整恢复；有R时净增15或22个，且四种上下文均未损失已有完整恢复。R/E互补的F1交互为T关闭时+0.154815、T开启时+0.138920，表明不能把各模块的单独增益直接相加。\n\n"
    text += "T在没有E、有R时带来F1+0.008597、净增1个完整恢复并节省约17.53分钟；其他三个上下文的平均F1均下降。已有R和E时，加入T为3胜109平8负，完整恢复0新增、6损失，省时22.67分钟。T不宜在当前参数下固定作为默认阶段。E内部仍含小块候选，所以关闭独立T并不等于移除全部小块重构能力。\n\n"
    text += "## 六种加入顺序与平均贡献\n\n![加入顺序](module_addition_orders.png)\n\n" + markdown(phi) + "\n\n![平均贡献](shapley_contributions.png)\n\n"
    text += "指定累积链R→T→E的完整恢复数为30→44→45→60，看起来每一步都改善；R→E→T却为30→44→66→60，揭示T最后加入时的负作用。平均贡献按全部六种模块加入顺序计算，三个数相加恰为全模块减去基座的成绩差。这里改变的是实验配置的模块加入顺序；算法内部仍按预选、前向/末端、最终退出的时序执行，并未新运行六套时序重排策略。\n\n"
    text += "## 为什么T会损失6个完整恢复\n\n"
    text += "逐例保存轨迹证实，这6例在有T与无T两侧的最终退出均未被接收；有T一侧的联合末端均接收了一次更新，随后以joint_tail_gap_met结束。无T一侧继续原始单候选扩展，验证选到了正确中间解。有T的输出候选库不含这个正确家族或正确正权重拓扑，且最终验证MAE更高。6例的真连接均已包含在两侧E的MILP候选域中，不能归因为缺失真连接；训练目标和接收门槛也未把正确树恢复到输出路径。问题表现为替代并截断有用的逐步候选路径，而非求解超时或验证在已有正确候选中选错。\n\n"
    text += "其中4个soumalas11常规输入，T的训练MAE更低，却保留5个内部集合而真实为4个，显示局部训练改善伴随过度细分。此诊断来自同参数保存轨迹，不证明所有小块联合模型都无效。\n\n"
    text += "六例详情见lost_exact_cases.csv。按网络比较：\n\n" + markdown(cases) + "\n\n"
    text += "关闭T后soumalas11从17增至22个完整恢复、flynn16从11增至12；paper15和pengwah18完整恢复数不变。pengwah18平均F1从0.839194降至0.834799，说明省去T并非逐网络每项指标都改善。弱激励仍为F1=0.071429、0/4；此问题未解决。\n\n"
    text += "## RNJ的贡献不能只归到两个小块\n\n"
    text += "退出候选池覆盖全部真连接：无R的两个E配置均为41/120，有R的两个均为105/120。R显著改善了结构候选覆盖；但完整覆盖不是恢复保证，R＋E仍仅恢复66/120。R的本轮开关同时控制种子、候选池和热启动，不能把全部增益归到预选小块，也没有分别识别这三个子功能的贡献。\n\n"
    text += "## 预算、时间和范围\n\n"
    text += f"每配置/输入总上限4小时，总名义上限3840方法小时；单步WZZT≤1小时、固定结构LP≤300秒，统一gap=0.0001、单线程、8个worker。参数为λ=0.003f0、响应界[0,2]、收缩0.24；前向预算规则相同，关闭E不会获得更多前向预算。\n\n实际方法时间之和{cost['summed_method_seconds']/3600:.4f}小时，其中公共前端按每方法计入；不重复计公共前端的唯一组件时间{cost['unique_component_seconds']/3600:.4f}小时。调度墙钟{cost['manager_wall_seconds']/60:.2f}分钟，不含冻结和调度前两次试运行；试运行已计入方法成本。方法时间为逐输入算法墙钟之和，包含求解、重拟合及输出选择，不含Python导入。\n\n"
    text += "本轮未调参、未叠加恢复分支，原方案均保留。结果支持本批数据下优先采用R＋原始WZZT及时清零＋E，再统一收缩/重拟合/自身验证；独立T保留作对照。该120输入已用于多轮研究，并非新盲测；候选域MILP最优性与物理拓扑恢复是不同的结论。\n\n"
    text += "可复现入口、原始阶段记录和实际执行源见RECOVERY.md与records.zip；完整原始表、条件边际和交互项见RESULTS.md及CSV。本结论图均有PNG和SVG版本。\n"
    (delivery / "CONCLUSIONS.md").write_text(text, encoding="utf-8", newline="\n")
    # Preserve the actually executed post-analysis/plotting version separately
    # from the preregistered optimization snapshot, which remains immutable.
    archive_source = output / "analysis_source"
    manifest = read(archive_source / "manifest.json")
    for name in ("diagnostics.py", "summarize.py", "figures.py"):
        path = ROOT / "research_experiments/full_ablation" / name
        shutil.copy2(path, archive_source / name)
        manifest["files"][name] = sha(path)
    manifest["reason"] = "analysis receipt fix, trajectory diagnostics/readable summary, and figure legend layout only; frozen optimizers/protocol unchanged"
    write(archive_source / "manifest.json", manifest)
    print(display.to_string(index=False))
    print(json.dumps(cost, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    summarize(parser.parse_args().output)
