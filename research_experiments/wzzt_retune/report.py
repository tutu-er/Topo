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
    frame=pd.read_csv(destination/"per_job.csv");aggregate=pd.read_csv(destination/"by_sample_size.csv")
    old=pd.read_csv(ROOT/"docs/wzzt_systematic_20261027/per_job.csv")
    colors={"NJ":"#2457a7","WZZT_free":"#d67517","WZZT_pool":"#8b59ad","RNJ_WZZT":"#269d77"}
    mapping={"NJ":"NJ","WZZT_free":"WZZT","WZZT_pool":"WZZT","RNJ_WZZT":"RNJ+WZZT"}
    fig,axes=plt.subplots(2,2,figsize=(11.8,8),sharex=True,sharey=True,layout="constrained")
    for ax,method in zip(axes.flat,METHODS):
        current=aggregate[aggregate.method==method].sort_values("N")
        baseline=old[old.method==mapping[method]].groupby("N").clade_f1.mean()
        ax.plot(baseline.index,baseline.values,"--o",color="gray",label="Round 1 baseline")
        ax.plot(current.N,current.F1,"-o",color=colors[method],label=DISPLAY[method])
        ax.set_title(DISPLAY[method]);ax.grid(alpha=.25);ax.set_ylim(-.02,1.03)
        ax.set_xscale("log",base=2);ax.set_xticks([24,48,96,288],labels=["24","48","96","288"])
        ax.set_xlabel("Total training measurements across 3 days");ax.set_ylabel("Mean clade F1 across 20 input groups")
        ax.legend(fontsize=9,loc="lower left")
    fig.suptitle("Same observations; one global F1-tuned configuration per method (in-sample benchmark)")
    fig.savefig(destination/"before_after.png",dpi=170);fig.savefig(destination/"before_after.pdf");plt.close(fig)
    summary=frame.groupby("method").agg(rows=("job_id","count"),models=("model_available","sum"),F1=("clade_f1","mean"),
        old_F1=("round1_F1","mean"),change=("F1_change","mean"),test_ac_rmse=("test_ac_rmse","mean"),
        ac_valid=("ac_valid","sum"),exact=("clade_f1",lambda x:int((x>=1-1e-12).sum())))
    casewise=pd.read_csv(destination/"casewise_results.csv").groupby("method").clade_f1.mean()
    loco=pd.read_csv(destination/"leave_one_case_out.csv").groupby("method").clade_f1.mean()
    summary["casewise_tuned_F1"]=casewise;summary["exploratory_LOCO_F1"]=loco
    summary=summary.reset_index();summary.to_csv(destination/"method_summary.csv",index=False,encoding="utf-8-sig")
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
        "", "有限池和完整 RNJ 初值包含算法域/初始化变化；其差值应按相应配置解释，不能全归因于放宽 MIP gap。NJ 基线也进行了同库结果调参。",
        "", table(aggregate,["N","method","planned","models","round1_F1","F1","F1_change","exact_recoveries"]),
        "", "## 结果调参的敏感性", "",
        table(summary,["method","F1","casewise_tuned_F1","exploratory_LOCO_F1"]),
        "", "casewise_tuned_F1 允许每个已知网络选择一个共享配置，属于额外适配能力；LOCO 是同一结果库中的探索性留一网络重选。参数网格全部保留于 parameter_grid.csv，主结果仍只采用全矩阵共同配置。path_grid_oracle_f1 仅是所保存路径和选点网格的事后上限，不作为主成绩。",
        "", "## 搜索与初始化的实际作用", "",
        f"自由域最终有 {diagnosis['free_certified_stops']}/{diagnosis['free_rows']} 项达到当前单步无实质改善证书或家族上限；其余停止原因单列于 diagnosis.json。可行解限时保留具有启发式性质，不能写为全树最优。",
        "", f"完整 RNJ 方法选中来源：{diagnosis['hybrid_selected_origins']}。自由域所选模型中，{diagnosis['free_rows_with_novel_true_clades']} 项包含原始有限池外的新真簇，共 {diagnosis['free_novel_true_clades']} 个。原池平均真簇覆盖率为 {diagnosis['mean_candidate_pool_true_recall']:.4f}。这些数值区分固定族重估、候选筛选与真正的池外发现。",
        "", "## 独立核验与证据", "",
        f"核验通过：{audit['all_passed']}。数据检查 {audit['data_checks']} 项；检查原生 LP 求解记录 {audit['LP_solve_records']} 条，独立重算 {audit['physical_model_points']} 个模型点、{audit['free_primal_vectors']} 份原始自由域向量；另用 SciPy/HiGHS 重解 {audit['independent_LP_checks']} 个固定族 LP，覆盖四网络、24/288 点及四种方法。热启动、最终采用解和原始求解解分别检查全部约束、边界及整数性。",
        "", "AC 在已选拓扑上统一作训练数据共享阻抗局部重估，最多 12 次函数评价；随后冻结参数评价原独立测试日。AC 可行、局部优化收敛和结构 F1 分别记录，不将预测低误差当成正确拓扑证书。",
        "", "## 逐算例条件结果", "",
        table(scenario,["case","condition",*METHODS]),
        "", "## 文件与复现", "",
        "per_job.csv 为 320 项主结果，by_sample_size.csv 和 by_case_condition.csv 为配对汇总；selected_models.json 保存全矩阵共同配置对应模型；parameter_grid.csv 保存全部参数尝试，chosen_global.json 与 chosen_casewise.json 区分选择层次。audit_details.json、输入清单及源码快照用于复核。原大体积模型和原始向量保留在运行输出目录，result_manifest.json 绑定其字节哈希。",
        "", "```powershell",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_retune.prepare outputs/<新目录>",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_retune.run outputs/<新目录>",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_retune.evaluate outputs/<新目录> docs/<新目录>",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_retune.audit outputs/<新目录> docs/<新目录> --independent-lp",
        "& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_retune.report outputs/<新目录> docs/<新目录>","```",""]
    (destination/"README.md").write_text("\n".join(report_lines),encoding="utf-8")
    for name in ("protocol.json","manifest.json","completion.json"):
        shutil.copy2(output/name,destination/name)
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
    files=[p for folder in ("jobs","ac","validation") for p in (output/folder).rglob("*") if p.is_file() and p.suffix in {".json",".npz",".log"}]
    write(destination/"result_manifest.json",{p.relative_to(output).as_posix():sha(p) for p in sorted(files)})
    (destination/".gitattributes").write_bytes(b"* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n")
    write(destination/"delivery_manifest.json",{p.name:sha(p) for p in destination.iterdir() if p.is_file() and p.name!="delivery_manifest.json"})
    print("report saved",destination,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");p.add_argument("destination");a=p.parse_args();report(a.output,a.destination)
