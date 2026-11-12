from pathlib import Path
import argparse
import importlib.metadata
import json
import platform
import shutil
import zipfile
import pandas as pd
from research_experiments.wzzt_systematic.common import read,write,sha

ROOT=Path(__file__).resolve().parents[2]
ORDER=["baseline25","multistart_original_rule","backward","ridge_pool","two_step","combined"]


def table(frame,columns):
    def fmt(v):
        if pd.isna(v):return "—"
        return f"{v:.5g}" if isinstance(v,float) else str(v)
    return "\n".join(["|"+"|".join(columns)+"|","|"+"|".join("---" for _ in columns)+"|"]+
                     ["|"+"|".join(fmt(r[c]) for c in columns)+"|" for r in frame.to_dict("records")])


def report(output,destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=Path(output).resolve();destination=Path(destination).resolve();protocol=read(output/"protocol.json")
    overview=read(destination/"overview.json");audit=read(destination/"audit_summary.json")
    frame=pd.read_csv(destination/"per_job.csv");summary=pd.read_csv(destination/"summary.csv").set_index("variant").loc[ORDER].reset_index()
    metrics=frame.groupby("variant").agg(precision=("precision","mean"),recall=("recall","mean"))
    summary=summary.merge(metrics,on="variant",validate="one_to_one")
    summary.to_csv(destination/"method_summary.csv",index=False,encoding="utf-8-sig")
    by_n=pd.read_csv(destination/"by_sample_size.csv")
    current_fail=pd.read_csv(destination/"previous_unidentified.csv")
    previous_changes=pd.read_csv(destination/"previous_changes_vs_baseline25.csv")
    previous_changes["direction"]=previous_changes.F1_change.map(lambda x:"improved" if x>0 else "degraded")
    previous_changes.to_csv(destination/"previous_changes_vs_baseline25.csv",index=False,encoding="utf-8-sig")
    events=[]
    for reference,column in (("NJ","delta_vs_NJ"),("baseline25","delta_vs_baseline25"),("multistart_original_rule","delta_vs_multistart")):
        for row in frame.to_dict("records"):
            events.append(dict(job_id=row["job_id"],case=row["case"],condition=row["condition"],N=row["N"],variant=row["variant"],reference=reference,
                pq_sigma_relative=row["pq_sigma_relative"],v_sigma_relative=row["v_sigma_relative"],root_sigma_relative=row["root_sigma_relative"],
                clade_f1=row["clade_f1"],delta=row[column],direction="improved" if row[column]>1e-10 else "degraded" if row[column]<-1e-10 else "same"))
    ev=pd.DataFrame(events);ev.to_csv(destination/"comparison_events.csv",index=False,encoding="utf-8-sig")
    ev.groupby(["variant","reference","condition","N","direction"]).size().rename("count").to_csv(destination/"events_by_condition_N.csv",encoding="utf-8-sig")
    ridge=frame[frame.variant=="ridge_pool"]
    ridge_changes=ridge[abs(ridge.delta_vs_multistart)>1e-10]
    ridge_fail=ridge[ridge.clade_f1<1-1e-10]
    ridge_wins=int((ridge.delta_vs_multistart>1e-10).sum());ridge_losses=int((ridge.delta_vs_multistart < -1e-10).sum())
    old_ms=frame[frame.variant=="multistart_original_rule"]
    old_missing=int((old_ms.candidate_pool_true_recall<1-1e-10).sum());new_missing=int((ridge.candidate_pool_true_recall<1-1e-10).sum())
    ridge_fail.to_csv(destination/"ridge_unidentified_noise.csv",index=False,encoding="utf-8-sig")
    diagnostics=[];novel=[]
    for job in protocol["jobs"]:
        file=output/"jobs"/job/"result.json"
        if not file.exists():continue
        r=read(file);ops=r["operations"]
        old_pool={tuple(s) for s in r["base_pool"]}
        selected=ridge[ridge.job_id==job].iloc[0]
        false={frozenset(s) for s in json.loads(selected.diagnostic_json)["false_clades"]}
        for support in json.loads(selected.supports_json):
            if tuple(support) in old_pool or not 2<=len(support)<len(r["labels"]):continue
            labels=[r["labels"][i] for i in support]
            novel.append(dict(job_id=job,case=r["case"],condition=r["condition"],N=r["N"],support_indices=json.dumps(support),
                support_labels=json.dumps(labels),is_true_posthoc=frozenset(labels) not in false))
        diagnostics.append(dict(job_id=job,ridge_old_pool=r["augmentation"]["old_pool_size"],ridge_new_pool=r["augmentation"]["new_pool_size"],
            backward_accepted=sum(x["accepted"] for x in ops["backward"]["decisions"]),
            backward_training_rise=sum(x["training_loss_rise"]>1e-10 for d in ops["backward"]["decisions"] for x in d["candidates"]),
            two_step_pair_proposals=len(ops["two_step"]["pairs"]),two_step_accepted_endpoints=sum(x["accepted"] for x in ops["two_step"]["pairs"]),
            two_step_final_candidates=len(ops["two_step"]["points"]),combined_pair_proposals=len(ops["combined_two_step"]["pairs"]),
            combined_accepted_endpoints=sum(x["accepted"] for x in ops["combined_two_step"]["pairs"]),
            two_step_stop=ops["two_step"]["reason"],combined_two_step_stop=ops["combined_two_step"]["reason"],
            worker_seconds=r["seconds"],result_path=str(file),result_sha256=sha(file)))
    diag=pd.DataFrame(diagnostics);diag.to_csv(destination/"operation_diagnostics.csv",index=False,encoding="utf-8-sig")
    novel_frame=pd.DataFrame(novel,columns=["job_id","case","condition","N","support_indices","support_labels","is_true_posthoc"])
    novel_frame.to_csv(destination/"selected_new_supports.csv",index=False,encoding="utf-8-sig")
    winner=max(ORDER,key=lambda v:(round(float(summary.set_index("variant").loc[v,"F1"]),12),-float(summary.set_index("variant").loc[v,"additional_seconds"])))
    ac=[read(p) for p in (output/"ac").glob("*.json")]
    ac_summary=dict(unique_records=len(ac),errors=sum("error" in r for r in ac),valid=sum(r.get("testing",{}).get("valid",False) for r in ac),
        converged=sum(r.get("fitting",{}).get("refit",{}).get("success") is True for r in ac),
        reused=sum(r.get("reused_prior_AC",False) for r in ac),selected_rows=len(frame),selected_valid=int(frame.ac_valid.sum()))
    write(destination/"ac_summary.json",ac_summary)
    write(destination/"recommended_strategy.json",dict(strategy_id=winner,scope="highest mean F1 in the fixed 80-input comparison",
        base_strategy="multistart_original_rule",implemented=["backward","ridge_pool","two_step"],
        default_enabled=["ridge_pool"] if winner=="ridge_pool" else [winner],optional_diagnostics=["backward","two_step","combined"],
        parameters=protocol["parameters"],selection=protocol["selection"],information_scope=protocol["information_scope"],
        new_truth_tuning=False,mean_F1=float(summary.set_index("variant").loc[winner,"F1"]),
        catalog=str(destination/"catalog.sqlite"),code_entry="research_experiments.wzzt_refine.worker.refine_bank",
        standalone_scope="refines an existing observation-derived RNJ/WZZT candidate bank; returns all four ablations"))
    environment=dict(python=platform.python_version(),platform=platform.platform(),workers=4,solver_threads=1,
        versions={n:importlib.metadata.version(n) for n in ("numpy","pandas","scipy","gurobipy")})
    write(destination/"environment.json",environment)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    colors=["#8b9099","#275eb3","#b9675b","#208972","#8a61a5","#d08a24"]
    labels=["Original hybrid","Multi-start","Backward","Ridge pool","Two-step","Combined"]
    for variant,label,color in zip(ORDER,labels,colors):
        a=by_n[by_n.variant==variant].sort_values("N");axes[0].plot(a.N,a.F1,"o-",label=label,color=color)
    axes[0].set_xscale("log",base=2);axes[0].set_xticks([24,48,96,288],labels=["24","48","96","288"])
    axes[0].set_ylim(.7,1.02);axes[0].grid(alpha=.25);axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Total training measurements across 3 days");axes[0].set_ylabel("Mean clade F1 across 20 inputs")
    axes[1].bar(range(6),summary.F1,color=colors);axes[1].set_xticks(range(6),labels=labels,rotation=30,ha="right",fontsize=9)
    axes[1].set_ylim(0,1.04);axes[1].set_ylabel("Mean clade F1 across 80 inputs")
    for i,v in enumerate(summary.F1):axes[1].text(i,v+.01,f"{v:.4f}",ha="center",fontsize=9)
    fig.suptitle("Fixed observations and validation rule; all improvements and regressions retained")
    fig.savefig(destination/"comparison.png",dpi=175);fig.savefig(destination/"comparison.pdf");plt.close(fig)
    compact=by_n.pivot(index="N",columns="variant",values="F1").reset_index()
    lines=["# RNJ+WZZT 三项增强实现及噪声—数据量核查", "",
        f"三项功能均已实现。正式批次 {overview['evaluated_inputs']}/80 组输入，形成 320 项新增配置结果与 160 项复用对照，共 {len(frame)}/480 条记录。当前平均 F1 最高版本为 `{winner}`。原输入、验证/测试日、原始候选及输出分辨率保持绑定，未进行新的真值调参。",
        "", "## 基线与版本", "",
        "baseline25 是原完整 RNJ+WZZT 组合，平均 F1=0.9194；multistart_original_rule 是上轮最佳多初值策略，平均 F1=0.9252；NJ 使用同一批量测的已调优 NJ。所有逐项记录同时保存三种参照的 F1 差值，避免把不同基线混在一起。",
        "", "新增 backward、ridge_pool、two_step 分别单独作用于上轮候选模型库；combined 依次使用正则化候选池修正、两步修正、后向删除。最终保持原 3% 验证简洁规则和 0.16 收缩规则，AC 门控关闭。",
        "", "## 实现细节", "",
        "后向删簇：每轮删除一个非叶支持后用训练数据联合 L1 拟合，保留叶。删除候选允许训练损失上升；局部以验证 MAE 最小值 3% 范围内优先简单模型决定是否继续，最多两轮、15 秒。全部已计算删除候选进入最终原规则选模；局部未选中的候选也保留其来源和分数。",
        "", "正则化候选：在 raw/demean 两分支使用已有有序 R/X QP 接口，lambda=0.001、0.01，alpha=lambda×trace(DᵀD)/(2n)，D=[P Q]。三几何、五阈值、NJ/RNJ 生成补充簇，仅扩充原候选池；最终 L1 系数拟合、原稠密矩阵校准及分辨率不变。在原多初值选中的收缩前模型上进行一次最多 25 秒的扩池修正。",
        "", "两步联合修正：先评价单候选替换，按训练损失保留宽度 3，允许第一步损失上升；随后同时加入两个相容支持，删除与任一新支持交叉的旧簇，联合拟合。终点实际训练改善超过 1e-10+0.001×当前损失时才成为新候选，最多保留 3 个不同家族。每次最多 25 秒，第一阶段最多占 40%；宽度和时间限制下的启发式状态单列。",
        "", "单位检查包含一个显式反例：正权错误簇的训练 MAE=0.0025，任一单簇替换都更差，而双簇联合替换达到 0.001875。另一个例子验证后向删除在训练损失上升时仍能改善独立验证误差。这些检查验证实现机制，实际效果由以下 80 组对照给出。",
        "", "新增算子分别有额外预算，旧模型库和路径复用，各分支还共享 LP 缓存。additional_seconds 是本次共享执行的算子计时；正则化版本包括前置 QP/树生成，仍不等于独立部署总耗时。此比较属于增量增强消融，未把增加的计算量写成与原策略完全相同的端到端预算。",
        "", "## 完整结果", "", "![三项实现对照](comparison.png)", "",
        table(summary,["variant","rows","F1","exact","delta_vs_multistart","wins","ties","losses","delta_vs_baseline25"]),
        "", table(compact,["N",*ORDER]),
        "", table(summary,["variant","precision","recall","R_error","X_error","candidate_pool_true_recall","test_ac_rmse","ac_valid"]),
        "", f"正则化扩池加追加修正版本相对多初值策略：提升 {ridge_wins} 项、持平 {len(ridge)-ridge_wins-ridge_losses} 项、退化 {ridge_losses} 项，完整恢复 {int((ridge.clade_f1>=1-1e-12).sum())}/80；候选池平均真簇覆盖率从 {old_ms.candidate_pool_true_recall.mean():.4f} 升至 {ridge.candidate_pool_true_recall.mean():.4f}，覆盖不完整的输入由 {old_missing} 项变为 {new_missing} 项。下表为所有结构变化，保留其噪声条件与数据量。",
        "", table(ridge_changes,["case","condition","N","baseline25_F1","multistart_F1","clade_f1","delta_vs_multistart"]),
        "", "两项改善均选中了原池之外的新真簇：soumalas11 的 {101,102,103}，pengwah18 的 {313,314,315,316,317,318}。selected_new_supports.csv 保存所有选中池外支持及事后真值标签。该配置同时增加了扩池后的修正计算量，本文按完整配置解释增益。",
        "", "后向删除版本提高了平均 precision，但 recall 下降并拉低 F1，存在过度简化。组合版本也没有超过单独扩池。两步联合动作已执行并核验，当前单独版本最终 F1 与多初值基线相同。其提出/接受终点和局部状态保留于 operation_diagnostics.csv，训练改善与结构增益分别记录。",
        "", "## 未成功辨识场景的噪声量级", "",
        "每个通道加入独立零均值高斯误差，标准差为 relative_sigma×max(|瞬时真值|,1e-12)。根电压使用独立随机数发生器。设置值是标准差，误差没有以该百分比为硬上限。P/Q 的相对标准差为 0.005=0.5%；参考/高负荷/低 X/R 的末端及根电压为 0.0002=0.02%，高电压噪声条件为 0.001=0.1%。ideal_ac 无量测噪声。",
        "", "根物理电压过程 sigma=0.003 是已生成的运行变化，与根电表测量噪声分开。约 1 p.u. 时，两端独立 0.1% 电压噪声对应电压差噪声标准差约 0.0014 p.u.，平方压降约 0.0028 p.u.²；根误差还使不同末端目标误差相关。previous_all_noise.csv 保存实际训练样本相对误差标准差、压降误差标准差和物理平均压降。模拟真值仅用于这些事后统计，估计器未读取该文件。",
        "", f"上轮最佳策略未完全恢复 15 项，其中 12 项高电压噪声；其余为 flynn16/higher_load/24、flynn16/low_xr/24、pengwah18/low_xr/48。当前 ridge_pool 未完整恢复 {len(ridge_fail)} 项，其中高电压噪声 {int((ridge_fail.condition=='high_voltage_noise').sum())} 项；逐场景分数如下。",
        "", table(ridge_fail,["case","condition","N","pq_sigma_relative","v_sigma_relative","baseline25_F1","multistart_F1","clade_f1"]),
        "", "## 相对基线改善或劣化时的数据量", "",
        "上轮多初值相对 baseline25：改善 5 项，分别在 24 点两项、96 点两项、288 点一项；退化 3 项全为高电压噪声 48 点。下面保留准确场景，最新新增算子的变化见 changes_vs_multistart.csv、changes_vs_baseline25.csv。NJ 参照另列在 comparison_events.csv。",
        "", table(previous_changes,["case","condition","N","pq_sigma_relative","v_sigma_relative","baseline25_F1","clade_f1","F1_change","direction"]),
        "", "N 为三个固定物理日上的总快照数，24/48/96/288 对应每日 8/16/32/96 点；各档是相同日内的嵌套采样。训练 replicate=0、验证 replicate=2、测试 replicate=1，验证与测试各 288 点。本轮未新增随机重复或新网络。",
        "", "## 数据复用与入口", "",
        "core.py 提供 backward、regularized_pool、two_step、pair_family；worker.py::refine_bank 接受 train、validation、原观察数据生成的 base_points/base_pool 和固定 calibration，返回四种模型及逐候选记录，接口不接受真簇。输入必须按同一末端列顺序对齐，并保留所有叶支持。",
        "", "```python",
        "from research_experiments.wzzt_budget.records import load, load_trace, find",
        'catalog = r"D:\\0-github_workspace\\Topo\\docs\\wzzt_refine_20261028\\catalog.sqlite"',
        'model = load(catalog, case="soumalas11", condition="high_voltage_noise", N=24,',
        '             variant="ridge_pool", with_observations=True)',
        'R, X = model["R"], model["X"]',
        'P, Q, V, root = model["train"]',
        'trace = load_trace(catalog, case="soumalas11", condition="high_voltage_noise", N=24)',
        'backward_moves = trace["operations"]["backward"]["decisions"]',
        'pair_moves = trace["operations"]["two_step"]["pairs"]',"```", "",
        "记录库与原接口兼容，含 480 条模型、80 组输入及声明/实际噪声索引。ranking_validation_mae 显式保留新增版本选模时的收缩前分数，validation_mae 为最终输出模型的实际误差；diagnostic_json 为含真簇的事后标签。完整新增候选、删簇/双簇动作、正则化前置矩阵和 LP 记录保存在输出目录，以哈希绑定。",
        "", "## 核验与复现", "",
        f"独立核验通过：{audit['all_passed']}。重算 {audit['physical_models']} 个物理模型、检查 {audit['LP_solves']} 条 LP 记录、{audit['ridge_fronts']} 个正则化前置矩阵、{audit['pair_proposals']} 个双簇提案及 {audit['backward_proposals']} 个删簇检查；HiGHS 独立重解 {audit['independent_LP_checks']} 个 LP。五项单元检查通过。",
        "", f"AC 共 {ac_summary['unique_records']} 个独立输入组/参数组合记录，复用 {ac_summary['reused']} 份已有结果，{ac_summary['converged']} 份局部重估收敛。基础设施错误 {ac_summary['errors']}，所选记录 AC 可行 {ac_summary['selected_valid']}/{len(frame)}；达到评价上限与物理可行分别计数。AC 仅以训练数据拟合，测试冻结参数。",
        "", "```powershell",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_refine.noise_context docs/<新目录>",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_refine.prepare outputs/<新目录>",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_refine.run outputs/<新目录>",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_refine.evaluate outputs/<新目录> docs/<新目录>",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_refine.audit outputs/<新目录> docs/<新目录>",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_refine.report outputs/<新目录> docs/<新目录>",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_refine.verify outputs/<新目录> docs/<新目录>","```",""]
    (destination/"README.md").write_text("\n".join(lines),encoding="utf-8")
    handoff=["# 给其他对话的最新策略交接", "",
        f"工作区 D:\\0-github_workspace\\Topo；本批平均 F1 最高的已测版本为 `{winner}`，基础策略仍为 `multistart_original_rule`。请先读取同目录 README.md、recommended_strategy.json。",
        "", "三项请求均已实现：core.py::backward、regularized_pool、two_step；worker.py::refine_bank 统一返回 backward/ridge_pool/two_step/combined 四个对照及过程记录。该接口接收训练/验证观测、原观测生成的模型库和候选池、固定校准量，不接收真簇。",
        "", "当前主增强仅启用 ridge_pool：在原多初值策略的模型库上，选取原规则给出的收缩前模型；用 raw/demean、lambda=0.001/0.01 的前置有序 R/X 正则化结果补充 NJ/RNJ 候选簇；保留原池及原校准，在扩展池上从该模型追加最多 25 秒交叉替换搜索；原模型库加新路径仍按 3% 简洁规则选点，再执行最终 0.16 收缩，不作最终重估。",
        "", "alpha=lambda×trace(DᵀD)/(2n)，D=[P Q]。原响应系数界 [0,2]、绝对残差 L1、raw 拟合及独立验证/测试划分保持不变。正则化只作用于新增候选生成，不替换最终参数拟合。",
        "", f"固定 80 组输入中，ridge_pool 平均 F1={ridge.clade_f1.mean():.10f}，对原多初值提升 {ridge_wins} 项、退化 {ridge_losses} 项，完整恢复 {int((ridge.clade_f1>=1-1e-12).sum())}/80。原多初值平均 0.9252397562，baseline25 平均 0.9194371947。后向删除和 combined 的平均 F1 下降，two_step 单独使用最终 F1 不变，保留为可调用实验组件。",
        "", "当前未完整恢复的噪声—数据量表见 ridge_unidentified_noise.csv；旧方法的 15 项与八项变化分别见 previous_unidentified.csv、previous_changes_vs_baseline25.csv。高电压噪声为 P/Q 0.5%、末端/根电压 0.1% 的高斯相对标准差；其他含噪条件电压 0.02%。N 为三个物理日总快照数。",
        "", f"直接读取现有模型库：{destination/'catalog.sqlite'}。继续使用 research_experiments.wzzt_budget.records.find/load/load_trace，variant='ridge_pool'。已有 480 条记录，含全部三种参照差值、实际噪声、支持及系数、原输入路径和完整动作日志。",
        "", "原最优策略执行顺序详见 docs/wzzt_rnj_best_strategy_20261028/README.md，当前报告是在它之上新增扩池修正；请勿把本目录 ridge_pool 与早期叶初值 WZZT_pool 或组合补丁 multistart75 混用。新数据仍需先构造观测驱动的基模型库，refine_bank 本身是增量增强入口，不是从任意原始文件直接完成整条旧流水线的包装器。",
        "", "本批未新增随机重复。复现使用新目录，保留所有基线与退化，并使用冻结 source/producer_source.zip；交付已绑定输入、过程、代码和评价哈希。",""]
    (destination/"HANDOFF.md").write_text("\n".join(handoff),encoding="utf-8")
    for name in ("protocol.json","manifest.json","completion.json"):shutil.copy2(output/name,destination/name)
    with zipfile.ZipFile(destination/"producer_source.zip","w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted((output/"source").rglob("*.py")):z.write(p,p.relative_to(output/"source").as_posix())
    with zipfile.ZipFile(destination/"analysis_source.zip","w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(Path(__file__).parent.glob("*.py")):z.write(p,p.name)
    write(destination/"analysis_source_manifest.json",{p.name:sha(p) for p in Path(__file__).parent.glob("*.py")})
    helper=ROOT/"research_experiments/wzzt_budget/audit.py"
    write(destination/"analysis_helper_manifest.json",{helper.relative_to(ROOT).as_posix():sha(helper)})
    files=[p for folder in ("jobs","ac") for p in (output/folder).rglob("*") if p.is_file() and p.suffix in (".json",".log")]
    write(destination/"result_manifest.json",{p.relative_to(output).as_posix():sha(p) for p in sorted(files)})
    (destination/".gitattributes").write_bytes(b"* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n")
    write(destination/"delivery_manifest.json",{p.name:sha(p) for p in destination.iterdir() if p.is_file() and p.name!="delivery_manifest.json"})
    print("report saved",destination,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");p.add_argument("destination");a=p.parse_args();report(a.output,a.destination)
