"""Saved-record formal RX report; persist rank thresholds before evaluation.

Only standard-library code, ev_rx_analyze and ev_dynamic_collect are imported.
There are no fits, AC calculations, strategy calls, or solvers in this module.
The calibration report is written and read back before evaluation is opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from . import ev_dynamic_collect as collector
from . import ev_rx_analyze as rx


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _threshold_bytes(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")


def save_thresholds(path,report,*,existing_reference=None):
    """Persist immutable thresholds; refuse differing existing/reference bytes."""
    path=Path(path);data=_threshold_bytes(report)
    if existing_reference is not None and Path(existing_reference).is_file():
        if Path(existing_reference).read_bytes()!=data:
            raise ValueError("computed threshold bytes differ from prior saved calibration thresholds")
    if path.exists():
        if path.read_bytes()!=data:raise ValueError("existing output thresholds differ; keep them and choose another directory")
    else:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    saved=collector.read_json(path)
    if saved!=report:raise ValueError("threshold readback differs")
    return dict(path=str(path.resolve()),sha256=_sha(path),thresholds_id=saved["thresholds_id"],
        prior_threshold_reference=None if existing_reference is None else str(Path(existing_reference).resolve()),
        prior_threshold_reference_verified=bool(existing_reference is not None and Path(existing_reference).is_file()))


def _diagnostics(domain):
    records=list(domain["records"].values());n=domain["planned_histories"]
    spectrum=[]
    for record in records:
        p=record.get("parameter",{});values=p.get("local_projected_shared_singular_values")
        finite=[float(v) for v in values or [] if collector._finite(v) and v>=0]
        spectrum.append(dict(seed=record["seed"],family=record["family"],values=values,
            coordinate_system=p.get("information_coordinate_system"),
            original_physical_scale=p.get("common_physical_scale"),
            minimum=min(finite) if finite else None,maximum=max(finite) if finite else None,
            condition_number=max(finite)/min(finite) if finite and min(finite)>0 else None,
            ratio_constraints=p.get("ratio_constraints"),
            scope="true candidate; original physical R/X scales after unconstrained local source-span projection; not confidence coverage"))
    labels=sorted({label for record in records for label in record.get("response",{})})
    response={}
    for name in labels:
        response[name]={}
        for channel in rx.CHANNELS:
            rows=[r.get("response",{}).get(name,{}) for r in records]
            response[name][channel]={key:rx.stats([
                r.get("channels",{}).get(channel,{}).get(key) if r.get("success") and r.get("saved_arithmetic_verified") else None
                for r in rows],n) for key in ("RMSE","relative_L2")}
    return dict(parameter_error=domain["parameter_error"],line_response_error=response,
        source_profile_spectra=spectrum,
        source_profile_minimum=rx.stats([r["minimum"] for r in spectrum],n),
        source_profile_condition_number=rx.stats([r["condition_number"] for r in spectrum],n),
        PQ_error_scale_by_history=[dict(seed=r["seed"],by_candidate=r.get("PQ_error_scale",{})) for r in records])


def _decision_contrast(contrast,decisions):
    indexed={str(r["seed"]):r for r in decisions}
    correct=[];rejected=[];unresolved=[]
    for row in contrast["rows"]:
        local=indexed.get(row["seed"])
        if local is None:
            correct.append(None);rejected.append(None);unresolved.append(None);continue
        a,b=local["methods"][contrast["left"]],local["methods"][contrast["right"]]
        row["delta_unique_correct"]=int(a["unique_correct"])-int(b["unique_correct"])
        row["delta_false_reject"]=int(a["false_reject"])-int(b["false_reject"])
        row["delta_unresolved"]=int(a["unresolved"])-int(b["unresolved"])
        correct.append(row["delta_unique_correct"]);rejected.append(row["delta_false_reject"]);unresolved.append(row["delta_unresolved"])
    n=contrast["delta_M"]["total"]
    contrast.update(delta_unique_correct=rx.stats(correct,n),delta_false_reject=rx.stats(rejected,n),
                    delta_unresolved=rx.stats(unresolved,n))
    return contrast


def contrasts(domain,decisions=()):
    """Within-history comparisons; every planned independent seed stays present."""
    result={}
    def add(label,a,b):
        result[label]=_decision_contrast(rx._contrast(domain["records"],a,b,domain["planned_seeds"],1e-4),decisions)
    for d in rx.DOMAINS:
        for baseline in ("square","linear","mean"):
            add("nearest_"+d+"_vs_"+baseline,"nearest_"+d,baseline+"_"+d)
    for prefix in rx.PREFIXES:
        for d in rx.DOMAINS:
            for baseline in ("constant","highlow"):
                add(prefix+"_"+d+"_vs_"+baseline,prefix+"_"+d,baseline)
        for label,a,b in (
            ("discharge_fixed","v2g_fixed","charge_fixed"),
            ("discharge_mobile","v2g_mobile","charge_mobile"),
            ("movement_charge","charge_mobile","charge_fixed"),
            ("movement_v2g","v2g_mobile","v2g_fixed")):
            add(prefix+"_"+label,prefix+"_"+a,prefix+"_"+b)
    return result


def _compact_domain(domain,decisions=()):
    keys=("directory","status","planned_seeds","observed_seeds","parsed_seeds","pending_seeds",
        "input_failed_seeds","input_errors","planned_histories","observed_histories","complete_histories",
        "failed_histories","planned_method_slots","observed_method_slots","complete_method_slots",
        "failed_method_slots","pending_method_slots","method_count","methods","policy_id",
        "config_sha256","source_manifest_sha256","input_files","freeze_sha256","runtime_seconds","cost")
    output={k:domain[k] for k in keys}
    output.update(diagnostics=_diagnostics(domain),paired_contrasts=contrasts(domain,decisions),
        observed_family_counts=dict(Counter(r["family"] for r in domain["records"].values())),
        candidate_omissions=sum(r["candidate_miss"] for r in domain["records"].values()),
        per_history=[dict(seed=r["seed"],family=r["family"],status=r["status"],seconds=r["seconds"],cost=r["cost"],
            parameter=r["parameter"],methods={m:dict(status=v["status"],M=v["M"],clean_future_M=v["clean_future_M"],
                energy=v["energy"],design=dict(selected_origin=v["design"]["selected_origin"],
                    selected_is_new_MILP=v["design"]["selected_is_new_MILP"],
                    substantive_new_MILP=v["design"]["substantive_new_MILP"],
                    predicted_gain_over_initial=v["design"]["predicted_gain_over_initial"],
                    proxy=v["design"]["proxy"],selected_AC_task_count=v["design"]["selected_prediction"]["observed"],
                    selected_AC_all_tasks=v["design"]["selected_prediction"]["complete"],
                    near_zero_pairs=v["design"]["selected_prediction"]["near_zero_pairs"])) for m,v in r["methods"].items()})
                for r in domain["records"].values()])
    return output


def render(report):
    def fmt(v):return "pending" if v is None else f"{v:.5g}"
    cal,eva=report["calibration"],report["evaluation"]
    n=eva["observed_histories"]
    lines=["# R/X包络与最近竞争者：独立校准和评价","",
        f"状态：{report['status']}。校准独立历史 {cal['observed_histories']}/{cal['planned_histories']}，评价 {n}/{eva['planned_histories']}。每历史19种方法，缓存或相同计划不增加历史数。","",
        "阈值在读取评价数据之前保存并回读核对。α=0.1，每方法、每真实候选族分别校准；每族9份历史，阈值为该族最大校准分数。控制依赖声明模拟分布内的历史可交换及真候选覆盖，对校准与新历史联合重复成立。没有同时控制19种策略的任意误拒。","",
        "范围为声明的平衡三末端、四候选闭合模拟族。M与模型距离不是正确概率；本批误拒次数也不是现场风险保证。","",
        "|方法|平均M|中位M|唯一真|误拒|未决|失败保留全部|有效评分/计划|",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in rx.METHODS:
        summary=eva["methods"][method];decision=report["decisions_by_method"].get(method,{}).get("overall",{})
        lines.append(f"|{method}|{fmt(summary['M']['mean'])}|{fmt(summary['M']['median'])}|{decision.get('unique_correct','pending')}|{decision.get('false_reject','pending')}|{decision.get('unresolved','pending')}|{decision.get('failed_keep_all','pending')}|{summary['M']['valid']}/{summary['planned']}|")
    lines += ["","唯一真表示校准后只保留真实候选；误拒可与未决同时发生。失败按原规则保留候选，单列原始失败分母。","",
        "|同域nearest−对照|平均ΔM|正/同/负|唯一真数量差|误拒数量差|有效/计划|",
        "|---|---:|---:|---:|---:|---:|"]
    for d in rx.DOMAINS:
        for baseline in ("square","linear","mean"):
            c=eva["paired_contrasts"]["nearest_"+d+"_vs_"+baseline];m=c["delta_M"]
            lines.append(f"|{d} vs {baseline}|{fmt(m['mean'])}|{m['positive']}/{m['ties']}/{m['negative']}|{fmt(c['delta_unique_correct']['sum'])}|{fmt(c['delta_false_reject']['sum'])}|{m['valid']}/{m['total']}|")
    lines += ["","ΔM并列容差1e−4，只做配对描述。constant/highlow及各前缀放电/移动差值和资源差异全部保存在JSON；跨域收益同时改变吞吐或驾驶条件。","",
        "|评价真候选训练误差|平均相对L2|有效/计划|","|---|---:|---:|"]
    for key in ("R","X"):
        s=eva["diagnostics"]["parameter_error"][key];lines.append(f"|{key}|{fmt(s['mean'])}|{s['valid']}/{s['total']}|")
    lines += ["","|评价代理与新计划|代理达标/可用|新MILP/实质新增|完整AC12任务/计划|净取电/吞吐均值(kWh)|驾驶/人员均值(kWh/min)|",
        "|---|---:|---:|---:|---:|---:|"]
    for prefix in ("square","nearest"):
        for d in rx.DOMAINS:
            s=eva["methods"][prefix+"_"+d];e=s["energy"]
            novelty="未保存标志" if s["new_MILP"] is None else f"{s['new_MILP']}/{s['substantive_new_MILP']}"
            lines.append(f"|{prefix}_{d}|{s['proxy_completed']}/{s['proxy_available']}|{novelty}|{s['selected_AC_all_pairs']}/{s['planned']}|{fmt(e['grid_energy_kwh']['mean'])}/{fmt(e['throughput_kwh']['mean'])}|{fmt(e['driving_kwh']['mean'])}/{fmt(e['worker_minutes']['mean'])}|")
    lines += ["","source-profile弱谱采用共同物理R/X尺度；它是访问点处消去源后的局部导数，不是置信域。保存的五类EV响应误差只在离线使用真实输入/源隔离线路误差。", "",
        f"校准/评价研究工作流耗时总和：{fmt(cal['runtime_seconds']['sum'])}/{fmt(eva['runtime_seconds']['sum'])}秒；包括训练、多方法和离线诊断，不是单一策略部署耗时。生产/独立回放AC、QP计数与各项求解失败另见JSON。此只读汇总新增AC、拟合与策略调用均为0。", "",
        "输入哈希、阈值字节核对、逐历史决策、全19方法分母、全部配对、参数/响应/弱谱、能源和求解界见formal_summary.json及独立thresholds.json/evaluation.json。",""]
    return "\n".join(lines)


def run_formal(calibration_directory,evaluation_directory,output_directory):
    caldir,evaldir,out=map(Path,(calibration_directory,evaluation_directory,output_directory))
    if out.resolve() in (caldir.resolve(),evaldir.resolve()):
        raise ValueError("formal outputs need a separate directory from raw inputs")
    # This is the only pre-threshold collection call. No evaluation path is read.
    thresholds=collector.collect(caldir,mode="calibrate",expected_per_family=9,alpha=.1)
    prior=caldir/"thresholds.json"
    saved=save_thresholds(out/"thresholds.json",thresholds,existing_reference=prior if prior.is_file() else None)
    # The persisted file, rather than the in-memory report, binds evaluation.
    caldomain=rx.read_domain(caldir,phase="calibration")
    evaldomain=rx.read_domain(evaldir,phase="evaluation")
    evaluation=(collector.collect(evaldir,mode="evaluate",thresholds=out/"thresholds.json")
                if evaldomain["parsed_seeds"] else None)
    if evaluation is not None:collector.write_json(out/"evaluation.json",evaluation)
    else:collector.write_json(out/"evaluation.json",dict(status="pending",planned_seeds=evaldomain["planned_seeds"],thresholds_id=saved["thresholds_id"]))
    if _sha(out/"thresholds.json")!=saved["sha256"]:raise ValueError("threshold file changed during evaluation")
    decisions=[] if evaluation is None else evaluation["decisions"]
    complete=not caldomain["pending_seeds"] and not evaldomain["pending_seeds"] and evaluation is not None
    status=("complete" if complete and caldomain["status"]==evaldomain["status"]=="complete" else
            "complete_with_failures" if complete else "partial")
    report=dict(schema="ev_rx_formal_saved_summary_v1",status=status,alpha=.1,
        thresholds_persisted_before_any_evaluation_read=True,threshold_file=saved,
        calibration=_compact_domain(caldomain),evaluation=_compact_domain(evaldomain,decisions),
        decisions_by_method={} if evaluation is None else evaluation["evaluation"],decisions=decisions,
        calibration_thresholds=thresholds["calibration"],thresholds_id=saved["thresholds_id"],
        policy_id=thresholds["policy_id"],scope=collector.SCOPE,
        candidate_family=list(collector.FAMILIES),declared_network="balanced three-terminal synthetic network",
        independent_unit="one history; methods, model pairs, duplicate curves, and noise reuse do not increase n",
        calibration_is_methodwise=True,simultaneous_multi_strategy_error_control=False,
        topology_correctness_probabilities_reported=False,
        analyzer=dict(source_sha256=_sha(__file__),rx_analyzer_sha256=_sha(rx.__file__),collector_sha256=_sha(collector.__file__),
            new_AC_calls=0,new_fit_calls=0,new_selector_calls=0,imports="standard library plus pure saved-record analysis/collector"))
    collector.write_json(out/"formal_summary.json",report)
    (out/"formal_summary.md").write_text(render(report),encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-directory",type=Path,required=True)
    parser.add_argument("--evaluation-directory",type=Path,required=True)
    parser.add_argument("--output-directory",type=Path,required=True)
    args=parser.parse_args()
    report=run_formal(args.calibration_directory,args.evaluation_directory,args.output_directory)
    print(json.dumps(dict(status=report["status"],calibration=report["calibration"]["observed_histories"],
        evaluation=report["evaluation"]["observed_histories"],thresholds_id=report["thresholds_id"],output=str(args.output_directory.resolve()))))


if __name__=="__main__":main()
