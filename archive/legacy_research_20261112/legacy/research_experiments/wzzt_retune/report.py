from pathlib import Path
import argparse
import shutil
import zipfile
import json
import pandas as pd
from research_experiments.wzzt_systematic.common import read,write,sha
from .selection import METHODS

DISPLAY={"NJ":"NJ tuned","WZZT_free":"Free WZZT tuned","WZZT_pool":"Finite-pool WZZT control","RNJ_WZZT":"Full RNJ + WZZT"}
ROOT=Path(__file__).resolve().parents[2]


def table(frame,columns):
    def fmt(value):
        if pd.isna(value):return "—"
        if isinstance(value,float):return f"{value:.4g}"
        return str(value)
    return "\n".join(["|"+"|".join(columns)+"|","|"+"|".join("---" for _ in columns)+"|"]+
                     ["|"+"|".join(fmt(r[c]) for c in columns)+"|" for r in frame.to_dict("records")])


def report(output,destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=Path(output).resolve();destination=Path(destination).resolve()
    overview=read(destination/"overview.json");protocol=read(output/"protocol.json")
    chosen=read(destination/"chosen_global.json");audit=read(destination/"audit_summary.json")
    paired=read(destination/"same_source_summary.json")
    paired_rows=pd.read_csv(destination/"same_source_ablation.csv")
    paired_by_n=pd.read_csv(destination/"same_source_by_N.csv")
    frame=pd.read_csv(destination/"per_job.csv");aggregate=pd.read_csv(destination/"by_sample_size.csv")
    old=pd.read_csv(ROOT/"docs/wzzt_systematic_20261027/per_job.csv")
    colors={"NJ":"#2457a7","WZZT_free":"#d67517","WZZT_pool":"#8b59ad","RNJ_WZZT":"#269d77"}
    mapping={"NJ":"NJ","WZZT_free":"WZZT","WZZT_pool":"WZZT","RNJ_WZZT":"RNJ+WZZT"}
    fig,axes=plt.subplots(2,2,figsize=(11.8,8),sharex=True,sharey=True,layout="constrained")
    for ax,method in zip(axes.flat,METHODS):
        current=aggregate[aggregate.method==method].sort_values("N")
        baseline=old[old.method==mapping[method]].groupby("N").clade_f1.mean()
        prior="Round 1 stable-block hybrid" if method=="RNJ_WZZT" else "Round 1 NJ" if method=="NJ" else "Round 1 free WZZT"
        ax.plot(baseline.index,baseline.values,"--o",color="gray",label=prior)
        ax.plot(current.N,current.F1,"-o",color=colors[method],label=DISPLAY[method])
        ax.set_title(DISPLAY[method]);ax.grid(alpha=.25);ax.set_ylim(-.02,1.03)
        ax.set_xscale("log",base=2);ax.set_xticks([24,48,96,288],labels=["24","48","96","288"])
        ax.set_xlabel("Total training measurements across 3 days");ax.set_ylabel("Mean clade F1 across 20 input groups")
        ax.legend(fontsize=9,loc="lower left")
    fig.suptitle("Same observations; one global F1-tuned configuration per method (in-sample benchmark)")
    fig.savefig(destination/"before_after.png",dpi=170);fig.savefig(destination/"before_after.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout="constrained")
    columns=["native_RNJ_F1","same_family_L1_F1","L1_resolution_F1","final_hybrid_F1"]
    values=[paired["mean"][c] for c in columns]
    axes[0].bar(range(4),values,color=["#707780","#558bb7","#3c9b93","#269d77"])
    axes[0].set_xticks(range(4),labels=["Matched\nRNJ source","Joint L1\nzero pruning","Weak-branch\ncontraction","Structural\ncorrection"])
    axes[0].set_ylim(0,1.04);axes[0].set_ylabel("Mean clade F1 across 80 inputs")
    axes[0].set_title("Same RNJ source: paired component ablation")
    for i,v in enumerate(values):axes[0].text(i,v+.015,f"{v:.4f}",ha="center")
    axes[1].plot(paired_by_n.N,paired_by_n.standalone_tuned_RNJ_F1,"o-",label="Standalone RNJ tuned",color="#707780")
    axes[1].plot(paired_by_n.N,paired_by_n.final_hybrid_F1,"o-",label="Full RNJ + WZZT",color="#269d77")
    nj=aggregate[aggregate.method=="NJ"].sort_values("N")
    axes[1].plot(nj.N,nj.F1,"o--",label="NJ tuned",color="#2457a7")
    axes[1].set_xscale("log",base=2);axes[1].set_xticks([24,48,96,288],labels=["24","48","96","288"])
    axes[1].set_ylim(.65,1.02);axes[1].grid(alpha=.25);axes[1].legend(fontsize=9)
    axes[1].set_xlabel("Total measurements across 3 days");axes[1].set_ylabel("Mean clade F1 across 20 inputs")
    axes[1].set_title("Same inputs: independent common-config baselines")
    fig.suptitle("Outcome-tuned benchmark; source attribution retained")
    fig.savefig(destination/"same_source.png",dpi=170);fig.savefig(destination/"same_source.pdf");plt.close(fig)
    summary=frame.groupby("method").agg(rows=("job_id","count"),models=("model_available","sum"),F1=("clade_f1","mean"),
        old_F1=("round1_F1","mean"),change=("F1_change","mean"),test_ac_rmse=("test_ac_rmse","mean"),
        R_error=("R_error","mean"),X_error=("X_error","mean"),physical_test_ac_rmse=("physical_test_ac_rmse","mean"),
        ac_valid=("ac_valid","sum"),exact=("clade_f1",lambda x:int((x>=1-1e-12).sum())))
    casewise=pd.read_csv(destination/"casewise_results.csv").groupby("method").clade_f1.mean()
    loco=pd.read_csv(destination/"leave_one_case_out.csv").groupby("method").clade_f1.mean()
    summary["casewise_tuned_F1"]=casewise;summary["exploratory_LOCO_F1"]=loco
    summary=summary.reset_index();summary.to_csv(destination/"method_summary.csv",index=False,encoding="utf-8-sig")
    ac_records=[read(p) for p in (output/"ac").glob("*.json")]
    ac_summary=dict(planned=320,records=len(ac_records),infrastructure_errors=sum("error" in r for r in ac_records),
        training_valid=sum(r.get("fitting",{}).get("valid",False) for r in ac_records),
        testing_valid=sum(r.get("testing",{}).get("valid",False) for r in ac_records),
        local_refit_converged=sum(r.get("fitting",{}).get("refit",{}).get("success") is True for r in ac_records),
        max_function_evaluations=12)
    write(destination/"ac_summary.json",ac_summary)
    scenario=frame.groupby(["case","condition","method"]).clade_f1.mean().unstack("method").reset_index()
    scenario.to_csv(destination/"by_case_condition.csv",index=False,encoding="utf-8-sig")
    parameters=pd.DataFrame([dict(method=m,configuration=json.dumps(chosen[m]["config"],ensure_ascii=False,sort_keys=True),
                                 trial_count=chosen[m]["config_count"]) for m in METHODS])
    free=frame[frame.method=="WZZT_free"]
    hybrid=frame[frame.method=="RNJ_WZZT"]
    diagnosis=dict(free_certified_stops=int(free.free_stop_certified.astype(str).str.lower().eq("true").sum()),
        free_rows=len(free),free_stop_reasons=free.search_stop.value_counts().to_dict(),
        hybrid_selected_origins=hybrid.selected_origin.value_counts().to_dict(),
        free_rows_with_novel_true_clades=int((free.selected_true_clades_outside_initial_pool>0).sum()),
        free_novel_true_clades=int(free.selected_true_clades_outside_initial_pool.sum()),
        mean_candidate_pool_true_recall=float(frame.candidate_pool_true_recall.mean()))
    write(destination/"diagnosis.json",diagnosis)
    report_lines=["# WZZT 早期结果核查与第二轮系统调优", "",
        f"本轮 {overview['returned_datasets']}/80 组输入返回，四种配置形成 {overview['models']}/320 个可评价模型。主结果对每种方法选一个共同配置，使用本算例库真实簇 F1 调参。训练和测试量测与首轮逐字节一致；新增 replicate=2 的独立验证日。",
        "", "## 早期为什么表现好", "",
        "早期四例 F1=1 来自 RNJ 白名单、每日去均值、末端收缩和独立验证，单步预算为 1800 秒；当时三个网络的 RNJ 已 F1=1，四例的池加冻结支持均覆盖所有真簇。单树候选彼此层状相容，在覆盖真簇时不会出现自由域的交叉簇阻挡。低样本压力实验中的明显收益主要来自 L1 删除假细分。",
        "", "上轮同时更换自由域、初始化、预算和选点：288 点纯 WZZT 只完成 5/20，完整 RNJ 家族也被少量稳定末端块替代，训练 AC 一标准误规则还有误选反例。具体证据、可确认的机制和未分离的因素见 [早期结果核查](early_results_audit.md)。",
        "", "## 本轮配置与调参边界", "",
        "同一 80 组输入覆盖 paper15、soumalas11、flynn16、pengwah18，五种条件及 24/48/96/288 点。N 对应固定三个物理日的采样密度。原 replicate=0 训练与 replicate=1 测试不变，验证独立使用 replicate=2；无额外隐藏源。",
        "", "NJ 同样允许在原始/每日去均值、R/RX75/X 几何和五档收缩阈值中面向结果调参。自由域 WZZT 保持叶初值和全部相容支撑域，增加数据候选给出的合法热启动；限时可保留有实际改善的可行解，优化证书单列。有限池 WZZT 作为额外对照，比较有无交叉簇替换。RNJ+WZZT 恢复完整普通 RNJ 家族，并允许零权删除及交叉旧簇替换。",
        "", "本轮有限池固定为两种预处理、三种几何下 NJ/RNJ 簇的并集，比早期单树白名单更宽，也保留跨候选交叉风险。preprocess 参数控制所选分支的回归树库及 L1 拟合，两种分支仍共享候选并集。完整 RNJ 方法包含固定族联合 L1 模型库及一次修正搜索，选中模型的来源单独统计。候选、初值和热启动均不使用真簇；真实 F1 只用于外层配置选择。",
        "", "每种方法的共同配置覆盖全部 80 组输入；逐网络配置只在同网络的 20 组输入中共享，另列为探索性结果。不是逐实例挑最高真值 F1 后拼接。结构 F1 为算例调优成绩；独立测试日预测误差和留一网络重选结果均作补充。",
        "", table(parameters,["method","configuration","trial_count"]),
        "", "结构搜索参数：固定族 LP 系数界为 2、单次 LP 最多 15 秒；候选池前向及修正每次最多 25 秒；自由域每种预处理最多 40 秒、单次 MILP 最多 5 秒、请求 gap=1%，物理改善阈值 1e-10+0.001×当前损失。原始、每日去均值分别生成路径，再选一个共同配置。初始星形模型触界不自动丢弃路径，在线 RX75 二元正权下限改为公开的后处理分辨率网格。",
        "", "组内共享候选库、LP 缓存及前置回归。per_job.csv 同时保存局部搜索、共享模型库和整组时间；局部搜索秒数不能直接解释为单独部署某方法的全部耗时。机器同时最多四个工作进程，硬限制每组 420 秒。",
        "", "## 主结果", "", "![同观测调优前后](before_after.png)", "",
        table(summary,["method","rows","models","old_F1","F1","change","exact","test_ac_rmse","ac_valid"]),
        "", "参数误差为相对 Frobenius 比例，0.1 表示 10%。NJ 使用稠密前置 R/X，结构方法使用分辨率处理后的原子矩阵；AC 列使用统一训练后处理。", "",
        table(summary,["method","R_error","X_error","test_ac_rmse","physical_test_ac_rmse","ac_valid"]),
        "", "有限池和完整 RNJ 初值包含算法域/初始化变化；其差值应按相应配置解释，不能全归因于放宽 MIP gap。NJ 基线也进行了同库结果调参。",
        "", table(aggregate,["N","method","planned","models","round1_F1","F1","F1_change","exact_recoveries"]),
        "", "## 同数据、同 RNJ 来源的配合机制", "",
        "![RNJ 与 WZZT 的同源配合](same_source.png)", "",
        f"独立调优的普通 RNJ 在相同 30 个几何/预处理/收缩组合中选择一个共同配置，平均 F1={paired['mean']['standalone_tuned_RNJ_F1']:.4f}，配置为 {paired['standalone_RNJ_common_config']}。完整 RNJ+WZZT 为 {paired['mean']['final_hybrid_F1']:.4f}，比较未改变训练、验证或测试观测。",
        "", "进一步固定最终组合模型所对应的初始 RNJ 来源，逐输入依次比较原生簇、同簇联合 L1 后零权删除、弱支路收缩及结构修正。这里的 RNJ 来源已经由组合方法的验证选模确定，故属于机制分解；独立调优的普通 RNJ 基线另列。",
        "", table(paired_by_n,["N","native_RNJ_F1","same_family_L1_F1","L1_resolution_F1","final_hybrid_F1","standalone_tuned_RNJ_F1"]),
        "", f"同源平均 F1：{paired['mean']['native_RNJ_F1']:.4f} → {paired['mean']['same_family_L1_F1']:.4f} → {paired['mean']['L1_resolution_F1']:.4f} → {paired['mean']['final_hybrid_F1']:.4f}。三部分增量分别为 {paired['mean']['L1_delta']:.4f}、{paired['mean']['resolution_delta']:.4f}、{paired['mean']['search_delta']:.4f}。结构修正相对相同初值收缩模型的簇变化合计为 {paired['structural_edits']}，共 5 项选中修正路径、75 项选中固定族重估。",
        "", f"固定原始 raw/RX75/0.16 RNJ 来源、所有输入不改变这项配置时，原生簇 F1={paired['mean']['default_RNJ_F1']:.4f}，联合 L1 加 0.16 分辨率后 F1={paired['mean']['default_RNJ_L1_resolution_F1']:.4f}。这一对照表明收益并非完全来自扩大初始树模型库。完整组合另有验证选模及局部修正收益。",
        "", "建议的测试主线是保留完整普通 RNJ 家族、允许联合拟合将假细分权重归零，再收缩弱支路并允许有限的交叉簇替换；初始固定族候选与全部修正路径共用相同验证规则。稳定末端块初始化与叶初值自由搜索继续单列。上述局部收益按实际来源报告，75 项固定族选择不计为新的搜索发现。",
        "", f"完整组合相对独立调优 RNJ：提升 {int((paired_rows.final_hybrid_F1>paired_rows.standalone_tuned_RNJ_F1+1e-10).sum())} 项、持平 {int((abs(paired_rows.final_hybrid_F1-paired_rows.standalone_tuned_RNJ_F1)<=1e-10).sum())} 项、退化 {int((paired_rows.final_hybrid_F1<paired_rows.standalone_tuned_RNJ_F1-1e-10).sum())} 项。退化发生在 flynn16/high_voltage_noise/N24，F1 从 0.1429 降至 0，说明小样本高噪声的验证选点仍有反例。",
        "", "## 结果调参的敏感性", "",
        table(summary,["method","F1","casewise_tuned_F1","exploratory_LOCO_F1"]),
        "", "casewise_tuned_F1 允许每个已知网络选择一个共享配置，属于额外适配能力；LOCO 是同一结果库中的探索性留一网络重选。参数网格全部保留于 parameter_grid.csv，主结果仍只采用全矩阵共同配置。path_grid_oracle_f1 仅是所保存路径和选点网格的事后上限，不作为主成绩。",
        "", "## 搜索与初始化的实际作用", "",
        f"自由域最终有 {diagnosis['free_certified_stops']}/{diagnosis['free_rows']} 项达到当前单步无实质改善证书或家族上限；其余停止原因单列于 diagnosis.json。可行解限时保留具有启发式性质，不能写为全树最优。",
        "", f"完整 RNJ 方法选中来源：{diagnosis['hybrid_selected_origins']}。自由域所选模型中，{diagnosis['free_rows_with_novel_true_clades']} 项包含原始有限池外的新真簇，共 {diagnosis['free_novel_true_clades']} 个。原池平均真簇覆盖率为 {diagnosis['mean_candidate_pool_true_recall']:.4f}。这些数值区分固定族重估、候选筛选与真正的池外发现。",
        "", "## 独立核验与证据", "",
        f"核验通过：{audit['all_passed']}。数据检查 {audit['data_checks']} 项；检查原生 LP 求解记录 {audit['LP_solve_records']} 条，独立重算 {audit['physical_model_points']} 个模型点、{audit['free_primal_vectors']} 份原始自由域向量；另用 SciPy/HiGHS 重解 {audit['independent_LP_checks']} 个固定族 LP，覆盖四网络、24/288 点及四种方法。热启动、最终采用解和原始求解解分别检查全部约束、边界及整数性。",
        "", "AC 在已选拓扑上统一作训练数据共享阻抗局部重估，最多 12 次函数评价；随后冻结参数评价原独立测试日。AC 可行、局部优化收敛和结构 F1 分别记录，不将预测低误差当成正确拓扑证书。",
        "", f"AC 记录 {ac_summary['records']}/{ac_summary['planned']}，基础设施错误 {ac_summary['infrastructure_errors']}，训练可行 {ac_summary['training_valid']}，测试可行 {ac_summary['testing_valid']}，局部重估收敛 {ac_summary['local_refit_converged']}。达到 12 次评价上限但潮流可行的结果仍保留。首次 AC 后处理出现 Windows 多进程导入路径冲突，失败记录保存在 ac_infrastructure_failed；修正后仅重做 AC 后处理，结构模型、输入及调参配置未改，详见 ac_recovery.json。",
        "", "## 逐算例条件结果", "",
        table(scenario,["case","condition",*METHODS]),
        "", "## 文件与复现", "",
        "per_job.csv 为 320 项主结果，by_sample_size.csv 和 by_case_condition.csv 为配对汇总；selected_models.json 保存全矩阵共同配置对应模型；parameter_grid.csv 保存全部参数尝试，chosen_global.json 与 chosen_casewise.json 区分选择层次。audit_details.json、输入清单及源码快照用于复核。原大体积模型和原始向量保留在运行输出目录，result_manifest.json 绑定其字节哈希。",
        "", "same_source_ablation.csv 为 80 项同源机制分解，standalone_RNJ_grid.json 保留普通 RNJ 独立调参网格，same_source_summary.json 记录逐项结构修正。",
        "", "```powershell",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_retune.prepare outputs/<新目录>",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_retune.run outputs/<新目录>",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_retune.evaluate outputs/<新目录> docs/<新目录>",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_retune.audit outputs/<新目录> docs/<新目录> --independent-lp",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_retune.ablation outputs/<新目录> docs/<新目录>",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_retune.report outputs/<新目录> docs/<新目录>","```",""]
    report_lines[-2:-2]=["& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_retune.verify_delivery outputs/<新目录> docs/<新目录>"]
    (destination/"README.md").write_text("\n".join(report_lines),encoding="utf-8")
    for name in ("protocol.json","manifest.json","completion.json"):
        shutil.copy2(output/name,destination/name)
    if (output/"ac_recovery.json").exists():shutil.copy2(output/"ac_recovery.json",destination/"ac_recovery.json")
    pilot=ROOT/"outputs/wzzt_retune_20261028_pilot/lp_backend_comparison.json"
    if pilot.exists():shutil.copy2(pilot,destination/"lp_backend_comparison.json")
    with zipfile.ZipFile(destination/"producer_source.zip","w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for p in sorted((output/"source").rglob("*.py")):archive.write(p,p.relative_to(output/"source").as_posix())
    with zipfile.ZipFile(destination/"analysis_source.zip","w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for p in sorted(Path(__file__).parent.glob("*.py")):archive.write(p,p.name)
    write(destination/"analysis_source_manifest.json",{p.name:sha(p) for p in Path(__file__).parent.glob("*.py")})
    sources=[ROOT/"outputs/mainline/metrics.json",ROOT/"rnj_wzzt_core/docs/advantage_stress_results_20260910.md",
             ROOT/"docs/wzzt_case_results_20261027/README.md",ROOT/"docs/wzzt_systematic_20261027/per_job.csv"]
    write(destination/"historical_source_manifest.json",{p.relative_to(ROOT).as_posix():sha(p) for p in sources})
    files=[p for folder in ("jobs","ac","ac_infrastructure_failed","validation") for p in (output/folder).rglob("*") if p.is_file() and p.suffix in {".json",".npz",".log"}]
    if (output/"ac_recovery.json").exists():files.append(output/"ac_recovery.json")
    write(destination/"result_manifest.json",{p.relative_to(output).as_posix():sha(p) for p in sorted(files)})
    (destination/".gitattributes").write_bytes(b"* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n")
    write(destination/"delivery_manifest.json",{p.name:sha(p) for p in destination.iterdir() if p.is_file() and p.name!="delivery_manifest.json"})
    print("report saved",destination,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");p.add_argument("destination");a=p.parse_args();report(a.output,a.destination)
