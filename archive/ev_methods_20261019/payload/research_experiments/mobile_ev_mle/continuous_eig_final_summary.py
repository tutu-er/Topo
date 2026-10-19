"""Generated H-only artifacts -> compact five-version comparison, no MC."""
from hashlib import sha256
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
from .continuous_eig_sampling import save_json
from .continuous_eig_diagnostic_summary import analyze


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def snapshot_validation(directory):
    directory=Path(directory)
    config=json.loads((directory/"config_before_MC.json").read_text())
    result=[]
    for field in ("source_sha256","math_sha256"):
        for name,wanted in config.get(field,{}).items():
            given=directory/"source_snapshot"/name
            result.append(dict(name=name,sha256_expected=wanted,
                sha256_snapshot=None if not given.exists() else digest(given),
                matches=given.exists() and digest(given)==wanted))
    return dict(all_match=all(r["matches"] for r in result),files=result,
        config_sha256=digest(directory/"config_before_MC.json"))


def final_analysis(directory):
    directory=Path(directory);summary=json.loads((directory/"summary.json").read_text())
    levels={}
    for level in (32,128):
        runs=[]
        for repeat in range(summary["independent_repeats"]):
            row=json.loads((directory/f"repeat_{repeat}_L{level}.json").read_text())
            modes=[]
            for raw,var in zip(row["diagnostics"]["modes"],row["variance_diagnostics"]["modes"]):
                modes.append(dict(var,log_mode_evidence=raw["log_mode_evidence"],
                    numerical_display_mass=raw["numerical_posterior_mode_mass"],
                    supported_theta_count=raw["theta_count"]-raw["prior_outside_theta_count"],
                    finite_joint_theta_count=raw["finite_joint_theta_count"],zero_theta_time_count=raw["zero_theta_time_count"],
                    prior_outside_theta_count=raw["prior_outside_theta_count"],
                    background_outside_point_count=raw["background_outside_point_count"],
                    master_rectangle_outside_point_count=raw["master_rectangle_outside_point_count"],
                    source_prior_outside_supported_point_count=raw["source_prior_outside_supported_point_count"],
                    inverse_failure_in_supported_domain_count=raw["inverse_failure_in_supported_domain_count"],
                    outside_background_master_inverse_unusable_count=raw["invalid_inverse_count"],
                    outside_inverse_count_is_numerical_failure=False,
                    proposal_roundoff=raw["proposal"].get("sampling_roundoff_diagnostics"),
                    guided_proposal_diagnostics=raw["guided_latent_proposal"]))
            runs.append(dict(repeat=repeat,seed=row["seed"],global_ESS=row["diagnostics"]["total_ess"],
                maximum_particle_weight=row["diagnostics"]["maximum_particle_weight"],
                log_evidence=row["log_evidence"],status=row["diagnostics"]["status"],
                failure_reasons=row["diagnostics"]["failure_reasons"],modes=modes,
                stratified_theta_relative_SE_conditional_on_fixed_latent_cloud=row["variance_diagnostics"]["stratified_theta_relative_SE_conditional_on_fixed_latent_cloud"],
                unconditional_beta_latent_variance_decomposition_identified=False))
        levels[str(level)]=dict(runs=runs,repeat_summary=summary["levels"][str(level)])
    return dict(directory=str(directory),status=summary["status"],seconds=summary["seconds"],levels=levels,
        paired=summary["paired"],posterior_accepted=False,predictive_MC_executed=False,
        source_validation=snapshot_validation(directory),postprocessing_only=True,new_MC_executed=False)


def produce(stage):
    stage=Path(stage)
    result={}
    for version in ("v1","v2","v3","v4"):
        directory=stage/f"continuous_eig_Honly_{version}"
        result[version]=analyze(directory)
        result[version]["source_validation"]=snapshot_validation(directory)
        result[version]["seconds"]=json.loads((directory/"summary.json").read_text())["seconds"]
    result["v5"]=final_analysis(stage/"continuous_eig_Honly_v5")
    result["engineering_pilot"]=json.loads((stage/"continuous_eig_tempered_pilot_v1/summary.json").read_text())
    result["scope"]=dict(all_versions_preserved=True,seven_modes_always_preserved=True,
        fixed_prior_and_physical_model=True,H_updates_each_final_level=1,
        actual_future_truth_noise_not_formal_inputs=True,incidental_schema_exposure_disclosed=True,
        posterior_accepted=False,predictive_EIG_or_action_ranking_reported=False,
        formal_C_D_stress_modified=False,primary_modified=False,git_executed=False)
    save_json(stage/"continuous_eig_Honly_all_five_comparison.json",result)
    lines=["# 连续 proper 模型：H-only 数值诊断交付", "",
        "五版均保留，proper prior、AC 物理模型和 Gaussian meter 目标未改。最终目标始终为原 π(β)L_H(β)，历史只更新一次。工程 pilot 的随机 Lhat^κ 权重不进入最终 estimator，也不是精确 tempered posterior。", "",
        "|版本|每模式 β / 每时刻 L|三重复 global ESS|最大权重范围|logZ 重复范围|秒|", "|---|---:|---|---:|---:|---:|"]
    for version in ("v1","v2","v3","v4"):
        row=result[version];runs=row["runs"];weights=[r["maximum_particle_weight"] for r in runs]
        ess=', '.join(f"{r['global_ESS']:.3f}" for r in runs)
        lines.append(f"|{version}|2048 / 32|{ess}|{min(weights):.4f}–{max(weights):.4f}|{row['log_evidence_range']:.6f}|{row['seconds']:.2f}|")
    for level in (32,128):
        row=result["v5"]["levels"][str(level)];runs=row["runs"];weights=[r["maximum_particle_weight"] for r in runs]
        ess=', '.join(f"{r['global_ESS']:.3f}" for r in runs)
        lines.append(f"|v5 L{level}|4096 / {level}|{ess}|{min(weights):.4f}–{max(weights):.4f}|{row['repeat_summary']['log_evidence_range']:.6f}|{result['v5']['seconds']:.2f}|")
    lines.extend(["", "v5 用每重复相同 fresh β/logq 和同一 per-time L128 nuisance cloud 的前32项，分别重算 H 权重。配对 logZ128−logZ32："+
        ", ".join(f"{r['logZ_delta_128_minus_32']:.6f}" for r in result["v5"]["paired"])+"。", "",
        "最终三重复中，clade_02_strong 的 conditional ESS 为437.32、259.06、193.94；另六模式均约1，未达到20。第3重复 L128 最大权重0.0521也超过0.05。故即使 global ESS 全部超过100，最终仍明确 failed。", "",
        "L128 的 clade_02_strong 条件加权 per-time latent ESS 平均90.58–92.76/128、relative SE 平均0.0570–0.0597、joint likelihood relative SE 样本矩0.2371–0.2474；其他模式的 joint 样本矩0.8311–2.4485。每模式原始 per-time 数组和独立重复证据跨度均保留。", "",
        "原 gate 保持 global ESS≥100、每个模式 conditional ESS≥20、最大粒子权重≤0.05，并要求重复/积分稳定；显示质量很小不能免检。所有模式 conditional ESS 和 latent ESS/relSE 保存在 JSON。", "",
        "θ 层 sample SE 条件于固定 latent cloud；latent 层 relSE 是样本矩诊断，由随机 H 权重选择，不能当误差界。跨 β 共用 latent CRN 使无条件 β/latent 方差分解无法从单个 bank 唯一识别。独立 repeats 和配对 doubling 提供额外诊断。", "",
        "实际支持内 inverse / finite target 数值失败强制终止并留档。outside background/master 支持的 inverse-unusable 计数属于 proposal 零目标事件，不能当支持内 numerical failure，也不能等同模型概率零。", "",
        "未接受 posterior，未运行 predictive EIG 或给动作排名。未改 formal C/D/stress 或 primary；无 Git。初始 trial schema 意外暴露已在数学 ledger 说明；正式 reader 只访问 sanitized H-only 六数组，NPZ future/latent/noise 未索引。", "",
        "继续的可核查瓶颈：按每模式 ESS、paired log evidence 与 conditional latent variance 判断 β proposal 是否仍偏离；不能删 rare mode、复用 pilot weights、把 finite-MC 零当模型拒绝，或在未再次冻结配置前继续大规模 MC。", "",
        "源代码与数学/测试交付清单见 continuous_eig_handoff_manifest.json；各版本 config/source_snapshot/rawbanks 保留 staging，raw NPZ 不推送。"])
    (stage/"continuous_eig_Honly_results_zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({version:dict(status=result[version]["status"],source_hashes_match=result[version]["source_validation"]["all_match"])
                     for version in ("v1","v2","v3","v4","v5")}))
    return result


if __name__=="__main__":
    produce(Path(__file__).parent)
