"""Build compact tables and scientific figures from audited per-job results."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import zipfile

from .common import read, sha, write


CONDITION_LABELS={"ideal_ac":"无量测噪声", "reference":"参考条件", "high_voltage_noise":"高电压噪声", "higher_load":"1.5 倍负荷", "low_xr":"低 X/R"}


def table(frame, columns):
    def fmt(value):
        import pandas as pd
        if pd.isna(value):
            return "—"
        if isinstance(value,float):
            return f"{value:.4g}"
        return str(value)
    lines=["|"+"|".join(columns)+"|", "|"+"|".join("---" for _ in columns)+"|"]
    lines += ["|"+"|".join(fmt(row[c]) for c in columns)+"|" for row in frame.to_dict("records")]
    return "\n".join(lines)


def report(output, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    output=Path(output).resolve()
    destination=Path(destination).resolve()
    protocol=read(output/"protocol.json")
    overview=read(destination/"overview.json")
    frame=pd.read_csv(destination/"per_job.csv")
    aggregate=pd.read_csv(destination/"aggregate.csv")
    colors={"NJ":"#2457a7","WZZT":"#d67517","RNJ+WZZT":"#269d77"}
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.5),sharex=True,sharey=True,layout="constrained")
    for ax,condition in zip(axes.flat,protocol["conditions"]):
        for method in protocol["methods"]:
            part=aggregate[(aggregate.condition==condition)&(aggregate.method==method)].sort_values("N")
            ax.plot(part.N,part.clade_f1_all_planned,marker="o",label=method,color=colors[method])
        ax.set_title(condition.replace("_"," "))
        ax.set_xscale("log",base=2)
        ax.set_xticks(protocol["sample_sizes"],labels=[str(n) for n in protocol["sample_sizes"]])
        ax.set_ylim(-.02,1.04)
        ax.grid(alpha=.25)
        ax.set_xlabel("Total training samples across 3 days")
        ax.set_ylabel("Mean clade F1 across 4 feeders")
    axes.flat[-1].axis("off")
    handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles,labels,loc="center",frameon=False)
    fig.suptitle("Paired topology benchmark: one training draw per condition; missing models count as zero")
    fig.savefig(destination/"f1_by_sample_size.png",dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(4,5,figsize=(16,11),sharex=True,sharey=True,layout="constrained")
    for i,case in enumerate(protocol["cases"]):
        for j,condition in enumerate(protocol["conditions"]):
            ax=axes[i,j]
            for method in protocol["methods"]:
                part=frame[(frame.case==case)&(frame.condition==condition)&(frame.method==method)].sort_values("N")
                ax.plot(part.N,part.clade_f1.fillna(0),marker="o",ms=3,label=method,color=colors[method])
                incomplete=part[part.status!="complete"]
                if len(incomplete):
                    ax.scatter(incomplete.N,incomplete.clade_f1.fillna(0),facecolors="none",edgecolors="red",s=65,zorder=5)
            ax.set_title(f"{case} / {condition}",fontsize=9)
            ax.set_xscale("log",base=2)
            ax.set_xticks(protocol["sample_sizes"],labels=[str(n) for n in protocol["sample_sizes"]])
            ax.set_ylim(-.02,1.04)
            ax.grid(alpha=.2)
    fig.suptitle("Clade F1 per feeder and condition; red circles mark incomplete/failed search")
    fig.savefig(destination/"f1_all_cases.png",dpi=170)
    plt.close(fig)
    # Small exact source closure makes the historical producer recoverable,
    # including its metadata logging collision. No observations are archived.
    archive=destination/"producer_source.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in sorted((output/"source").rglob("*.py")):
            z.write(path,path.relative_to(output/"source").as_posix())
    shutil.copy2(output/"protocol.json",destination/"protocol.json")
    shutil.copy2(output/"frozen_manifest.json",destination/"frozen_manifest.json")
    # Physical input envelope, including actual rather than prescribed drops.
    envelope=[]
    for case in protocol["cases"]:
        for condition in protocol["conditions"]:
            diag=read(output/"data"/case/condition/"diagnostics.json")
            train=[d for d in diag if d["split"]=="train"]
            envelope.append(dict(case=case,condition=condition,
                voltage_min=min(d["voltage_min"] for d in train),
                voltage_max=max(d["voltage_max"] for d in train),
                mean_drop_pct=100*np.mean([d["mean_drop"] for d in train]),
                max_drop_pct=100*max(d["max_drop"] for d in train)))
    envelope=pd.DataFrame(envelope)
    envelope.to_csv(destination/"physical_envelope.csv",index=False,encoding="utf-8-sig")
    empty=frame[(frame.method=="RNJ+WZZT")&(frame.groups==0)]
    paired=frame.pivot(index=["case","condition","N"],columns="method",values="clade_f1")
    comparison={}
    for method in ("WZZT","RNJ+WZZT"):
        diff=paired[method]-paired.NJ
        comparison[method]=dict(better_than_NJ=int((diff>1e-12).sum()),equal_to_NJ=int((abs(diff)<=1e-12).sum()),
                                worse_than_NJ=int((diff< -1e-12).sum()),paired=int(diff.notna().sum()))
    write(destination/"paired_comparison.json",comparison)
    reference=frame[frame.condition=="reference"]
    ref_table=reference.pivot(index=["case","N"],columns="method",values="clade_f1").reset_index()
    report_lines=["# WZZT 多数据量系统测试", "",
        f"首轮计划 240 项方法任务，已返回 {overview['returned']}/240；正常完成 {overview['complete']}/240，状态分布为 {overview['statuses']}。可评价模型 {overview['models']}/240。",
        "", "## 测试范围", "",
        "四个网络 paper15、soumalas11、flynn16、pengwah18；训练总点数为 24、48、96、288，分别在三个物理日取 8、16、32、96 点。每个物理日先按 96 点生成，再作嵌套抽样；独立测试集固定为另三个 96 点物理日。每个条件只有一组训练和测试随机路径，跨算例均值用于本轮比较。",
        "", "五种条件为无量测噪声、参考条件、高电压噪声、1.5 倍负荷和低 X/R。参考条件的 P/Q 相对噪声标准差为 0.5%，末端及根电压均为 0.02%；高电压噪声将两类电压噪声改为 0.1%。低 X/R 保持 R 不变，干线和服务线 X/R 分别为 0.30、0.06，是受控敏感性分析。根电压物理过程均值 1.02、sigma=0.003；所有条件采用实际阻抗倍率 1，无额外隐藏源。",
        "", "## 方法与终止条件", "",
        "NJ 使用有序 R/X 回归和带观测根的经典 NJ，RX75 内边收缩系数 0.16。WZZT 从单点簇开始；RNJ+WZZT 额外加入最多两个通过严格 bootstrap 检查的稳定末端块，并共享块内成员变量。每个块使用 100 次同步移动块 bootstrap，块长度随采样间隔调整到约一小时。",
        "", f"RNJ+WZZT 有 {len(empty)}/80 项未选出稳定块，保留空初值退化情况。该配置沿用现有稳定末端块算法；完整 RNJ 树初始化是另一种算法配置，本轮未混入该名称。",
        "", "WZZT 每步最多 60 秒，结构搜索总计最多 180 秒，包含前端及 RNJ bootstrap。目标改善阈值为 1e-10 + 0.001×当前损失。若可行改善 g>阈值，且求解上、下界差不超过 0.5g，则接受；若有效下界证明最大可能改善不超过阈值，则终止；其他状态保留为未完成。请求相对 gap=1e-4，较大改善通过回调提前返回。",
        "", "系数初始上界为 2，触及 99.5% 时倍增并从初始簇重启，最多四次扩张。全路径使用同一训练 R/X 校准的 RX75 下限。只在固定重拟合确认不恶化目标时删除非初始零权簇；正权簇保持兼容约束。路径最多含 2n-1 个簇，前向步数另限为 4n。",
        "", "结构搜索后，在每个路径点用训练数据作局部共享阻抗 AC 拟合，最多 12 次函数评价。训练 AC 电压 MSE 最小值加一个日块标准误范围内，选簇数最少、然后最早的点。三个物理日为三个误差块，测试数据在选择之后才由独立评价器读取。AC 拟合单独计时，因此端到端时间可超过 180 秒；进程保护上限为 600 秒。",
        "", "## 评价口径", "",
        "主要结构指标为非平凡下游簇 F1，与完整物理真簇比较。NJ 按其收缩树计分；WZZT 按固定 RX75 分辨率下的有效簇计分，同时保存家族 F1、正权簇 F1 和最后路径点 F1。R/X 误差为原估计器矩阵的相对 Frobenius 误差：NJ 使用稠密回归矩阵，WZZT 使用原子矩阵。",
        "", "AC 测试误差为训练选定模型和训练拟合阻抗在独立三个测试日上的相对电压 RMSE。NJ 额外采用固定树 L1 及相同 AC 拟合，以统一物理预测参数口径，原 NJ 拓扑与稠密矩阵指标保持独立记录。物理输入误差使用测试真实 P/Q、根电压及末端电压，量测输入误差使用对应带噪数据。",
        "", "## 数据量与结构精度", "", "![不同数据量下的平均 F1](f1_by_sample_size.png)", "",
        table(aggregate,["condition","N","method","planned","returned","complete","clade_f1_all_planned","test_ac_rmse","elapsed_seconds"]),
        "", "## 参考条件逐算例结果", "", table(ref_table,["case","N","NJ","WZZT","RNJ+WZZT"]),
        "", "## 完成情况与核验", "",
        f"数据核验通过：{overview['data_audit_passed']}。求解原始约束、整数性、目标值和判据重算：{overview['raw_audit_passed']}，共 {overview['raw_attempts']} 次扩展求解。所有失败、超时和未完成任务保留在 per_job.csv 的计划分母中。未完成模型按停止时已有路径评分，状态列同时保留。",
        "", "首轮冻结的生产者有一处日志字段名冲突：系数上界覆盖了保存的 bound 键。求解回调与决策在覆盖前已完成；独立评价器从原路径当前损失与保留的 gain_possible 恢复实际求解下界，并重算判据。solver_audit.json 逐项记录恢复值；原始 source 和日志保留，后续驱动已采用不同字段名。",
        "", "本轮 normal complete 只表示声明的前向搜索规则完成；优化界覆盖当前有界单步问题。AC 低误差和 bootstrap 稳定度作为诊断使用。四个网络是仓库构建的文献启发合成网络；工程数据、额外源及多随机种子重复另行测试。",
        "", "## 交付与复现", "",
        "逐项结果见 per_job.csv，完整条件汇总见 aggregate.csv，物理电压和实际压降见 physical_envelope.csv，原始扩展求解核验见 solver_audit.json，冻结输入清单见 frozen_manifest.json。producer_source.zip 保留真正运行的源代码闭包。",
        "", "```powershell", "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_systematic.prepare outputs/<新目录>",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_systematic.run outputs/<新目录>",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_systematic.evaluate outputs/<新目录> docs/<新报告目录>",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_systematic.report outputs/<新目录> docs/<新报告目录>", "```", "",
        "冻结历史结果可将 producer_source.zip 解压到该运行的 source 目录，由冻结控制器执行。新驱动修复了日志字段名与循环检测的规范化；原历史冻结源仍可精确核对。"]
    (destination/"README.md").write_text("\n".join(report_lines)+"\n",encoding="utf-8")
    write(destination/"delivery_manifest.json",{p.name:sha(p) for p in destination.iterdir() if p.is_file() and p.name!="delivery_manifest.json"})
    print("report",destination,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("destination",type=Path)
    args=parser.parse_args()
    report(args.output,args.destination)
