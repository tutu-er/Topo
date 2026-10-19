"""Pure saved-record RX comparison; no strategy, AC, fit, or solver imports.

Expected histories and all 19 methods stay in the denominator while a batch
is running.  Summaries describe development evidence, not calibrated decisions.
Truth is used only in saved offline parameter/response diagnostics and M.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

from . import ev_dynamic_collect as collector

DOMAINS = ("charge_fixed", "v2g_fixed", "charge_mobile", "v2g_mobile")
PREFIXES = ("linear", "square", "mean", "nearest")
METHODS = ("none", "constant", *(p+"_"+d for p in PREFIXES for d in DOMAINS), "highlow")
CHANNELS = ("v", "master_p", "master_q")


def _hash(value):
    return collector.object_hash(value)


def _finite(value):
    return collector._finite(value)


def stats(values, denominator, *, tolerance=0.):
    valid = [float(v) for v in values if _finite(v)]
    if len(valid)>denominator:
        raise ValueError("values exceed independent-history denominator")
    return dict(total=denominator, valid=len(valid), unavailable=denominator-len(valid),
        mean=statistics.fmean(valid) if valid else None,
        median=statistics.median(valid) if valid else None,
        minimum=min(valid) if valid else None, maximum=max(valid) if valid else None,
        sum=math.fsum(valid) if valid else None,
        positive=sum(v>tolerance for v in valid),negative=sum(v < -tolerance for v in valid),
        ties=sum(abs(v)<=tolerance for v in valid),sign_tolerance=tolerance)


def _flat(value):
    if isinstance(value,list):
        return [v for item in value for v in _flat(item)]
    if not _finite(value): raise ValueError("nonfinite numeric array")
    return [float(value)]


def _rms_difference(a,b):
    x,y=_flat(a),_flat(b)
    if len(x)!=len(y) or not x: raise ValueError("numeric arrays differ in size")
    return math.sqrt(math.fsum((s-t)**2 for s,t in zip(x,y))/len(x))


def _pair_prediction(prediction, candidate_names, zero_tolerance):
    expected={(a,b) for a in candidate_names for b in candidate_names if a!=b}
    rows=prediction.get("pairs",[]) if isinstance(prediction,dict) else []
    pairs=[];seen=set();bad=[]
    for row in rows:
        key=(row.get("a"),row.get("b"));value=row.get("added")
        if key not in expected or key in seen: bad.append("unexpected_or_duplicate_pair")
        seen.add(key)
        good=row.get("success") is True and _finite(value) and value>=0
        if not good: bad.append("failed_or_invalid_pair")
        pairs.append(dict(name=row.get("name"),a=key[0],b=key[1],half_SSE=value if good else None,
                          success=good))
    complete=bool(expected and seen==expected and len(rows)==len(expected) and not bad)
    vals=[r["half_SSE"] for r in pairs if r["success"]]
    minima={a:min(r["half_SSE"] for r in pairs if r["a"]==a)
            for a in candidate_names} if complete else {}
    return dict(expected=len(expected),observed=len(rows),complete=complete,failures=bad,
        missing_pairs=[list(p) for p in sorted(expected-seen)],pairs=pairs,
        exact_zero_pairs=sum(v==0 for v in vals),near_zero_pairs=sum(v<=zero_tolerance for v in vals),
        zero_tolerance=zero_tolerance,reference_minima=minima,
        mean_half_SSE=statistics.fmean(vals) if complete else None,
        nearest_half_SSE=statistics.fmean(minima.values()) if complete else None)


def _parameter_diagnostic(trial):
    true=trial.get("true_name");fits=trial.get("training_result",{}).get("fits",[])
    row=next((r for r in fits if r.get("candidate",r.get("name"))==true),None)
    result=dict(saved_relative_error=trial.get("training_parameter_error",{}),available=False)
    if row is None or not row.get("success") or not trial.get("truth"):
        return result
    truth=trial["truth"]
    if row.get("supports")!=truth.get("supports"):
        result["failure"]="true-candidate edge order differs from physical truth";return result
    for key,label in (("r","R"),("x","X")):
        fitted,actual=_flat(row[key]),_flat(truth[key])
        if len(fitted)!=len(actual):raise ValueError("true parameter arrays differ in size")
        norm=math.sqrt(math.fsum(v*v for v in actual));err=math.sqrt(math.fsum((a-b)**2 for a,b in zip(fitted,actual)))
        result[label+"_absolute_L2_pu"]=err
        result[label+"_relative_L2"]=err/norm if norm else None
        saved=result["saved_relative_error"].get(label)
        if _finite(saved) and not math.isclose(saved,result[label+"_relative_L2"],rel_tol=1e-10,abs_tol=1e-12):
            raise ValueError("saved true parameter error disagrees with arrays")
    diagnostic=row.get("diagnostics",{})
    result.update(available=True,
        X_over_R=[x/r for x,r in zip(row["x"],row["r"])],
        local_projected_shared_singular_values=diagnostic.get("local_projected_shared_singular_values"),
        information_coordinate_system=diagnostic.get("information_coordinate_system","legacy physical R/X/source"),
        common_physical_scale=diagnostic.get("common_physical_scale"),
        ratio_constraints=diagnostic.get("ratio_constraints"),training_true_SSE=row.get("loss"))
    return result


def _response_diagnostics(saved):
    result={}
    for name,row in saved.items():
        check=dict(row)
        if row.get("success") is True:
            try:
                for channel in CHANNELS:
                    value=row["channels"][channel]
                    rmse=_rms_difference(value["fitted_increment"],value["true_increment"])
                    true=_flat(value["true_increment"]);fit=_flat(value["fitted_increment"])
                    rel=math.sqrt(math.fsum((a-b)**2 for a,b in zip(fit,true)))/max(math.sqrt(math.fsum(v*v for v in true)),1e-14)
                    if not math.isclose(rmse,value["RMSE"],abs_tol=1e-12,rel_tol=1e-9) or not math.isclose(rel,value["relative_L2"],abs_tol=1e-12,rel_tol=1e-9):
                        raise ValueError("saved response error differs from increment arrays")
                check["saved_arithmetic_verified"]=True
            except (ValueError,KeyError,TypeError) as error:
                check.update(success=False,analysis_failure=str(error),saved_arithmetic_verified=False)
        result[name]=check
    return result


def _method_diagnostic(trial,name,zero_tolerance,substantive_tolerance):
    selection=trial.get("selection",{});record=selection.get("methods",{}).get(name,{})
    square=name.startswith("square_");model_objective=name.startswith(("mean_","nearest_"))
    raw=(record.get("exchange") if square else record.get("raw")) or record.get("raw",{}) or {}
    prediction=selection.get("predictions",{}).get(name,{})
    bank=[c["name"] for c in trial.get("candidates",[])]
    final_pairs=_pair_prediction(prediction,bank,zero_tolerance)
    raw_pairs=_pair_prediction(record.get("raw_prediction",{}),bank,zero_tolerance)
    score,initial=(record.get("Q"),record.get("best_initial_Q")) if square else (record.get("selected_score"),record.get("initial_best_score"))
    gain=score-initial if _finite(score) and _finite(initial) else None
    flag_available=model_objective and "selected_is_new_MILP_proposal" in record
    selected_new=record.get("selected_is_new_MILP_proposal") is True if flag_available else None
    required=raw.get("required_pairs",[])
    expected_names={r["name"] for r in final_pairs["pairs"]}
    proxy_source=record if square else raw
    gap,lower,upper=(proxy_source.get(k) for k in ("gap","objective_lower","objective_upper"))
    target=raw.get("target_gap")
    if square and all(_finite(v) for v in (lower,upper)):
        options=record.get("exchange_configuration",{})
        absolute,relative=options.get("absolute_gap"),options.get("relative_gap")
        if all(_finite(v) for v in (absolute,relative)):
            target=absolute+relative*max(1e-12,abs(lower),abs(upper))
    gap_checked=bool(all(_finite(v) for v in (gap,target,lower,upper)) and target>=0
        and gap>=-1e-8 and upper>=lower-1e-8 and gap<=target+1e-8
        and math.isclose(gap,upper-lower,abs_tol=1e-8,rel_tol=1e-8))
    return dict(selected_origin=record.get("selected_origin"),selected_is_new_MILP=selected_new,
        new_MILP_flag_available=flag_available,
        selected_score=score,initial_best_score=initial,predicted_gain_over_initial=gain,
        substantive_new_MILP=bool(selected_new and _finite(gain) and gain>substantive_tolerance) if flag_available else None,
        score_units="weighted kW squared" if square else ("half whitened SSE" if model_objective else "not applicable"),
        substantive_gain_tolerance=substantive_tolerance,
        inheritance_protected=record.get("inheritance_protected"),failures=record.get("failures",[]),
        proxy=dict(available=bool(raw),record_field="exchange" if square and record.get("exchange") else ("raw" if raw else "not applicable"),
            status=proxy_source.get("status",raw.get("status")),declared_completed=proxy_source.get("completed") is True,
            completed=proxy_source.get("completed") is True and gap_checked,gap_completion_arithmetic_pass=gap_checked,
            gap=gap,target_gap=target,objective_lower=lower,
            objective_upper=upper,required_pair_count=len(required),
            topology_task_requirement_applicable=model_objective,
            all_expected_pairs_required=bool(final_pairs["complete"] and required and
                len(required)==len(set(required))==len(bank)*(len(bank)-1) and set(required)==expected_names) if model_objective else None,
            inner_qp_failures=raw.get("inner_qp_failures"),qp_calls=raw.get("qp_calls"),
            final_control_segments=raw.get("final_control_segments"),seconds=raw.get("seconds")),
        selected_prediction=final_pairs,raw_prediction=raw_pairs,
        witness_gate=record.get("proxy_AC_check",{}).get("success"))


def _extract(trial,standard,zero_tolerance,substantive_tolerance):
    names=[c["name"] for c in trial.get("candidates",[])]
    methods={m["name"]:m for m in trial.get("methods",[])}
    output=dict(seed=standard["seed"],family=standard["family"],status=trial.get("status"),
        candidate_miss=standard["candidate_miss"],parameter=_parameter_diagnostic(trial),
        response=_response_diagnostics(trial.get("line_response_diagnostics",{})),PQ_error_scale=trial.get("PQ_error_scale",{}),
        fingerprints={k:_hash(trial.get(k)) for k in
            ("training","validation","future_meter","training_meter","validation_meter","background","truth","candidates")},
        methods={},cost=trial.get("actual_solver_calls",{}),seconds=trial.get("seconds"),
        model_task_count=trial.get("selection",{}).get("model_task_count"),candidate_count=len(names))
    for name in METHODS:
        row=methods.get(name,{})
        saved_clean=row.get("clean_future_M")
        clean=saved_clean if row.get("clean_future_complete") is True and _finite(saved_clean) else None
        output["methods"][name]=dict(standard["methods"][name],clean_future_M=clean,
            clean_future_scope=row.get("clean_future_scope"),power_sha256=_hash(row.get("power_kw")),
            physical_sha256=_hash(row.get("physical")),
            physical_available=isinstance(row.get("physical"),dict) and bool(row["physical"]),
            design=_method_diagnostic(trial,name,zero_tolerance,substantive_tolerance))
    return output


def read_domain(directory,*,phase="development",zero_tolerance=1e-8,substantive_tolerance=1e-4):
    directory=Path(directory);freeze=collector.read_json(directory/"freeze.json")
    identity=collector._identity(freeze);methods=collector.expected_methods(freeze["config"])
    if set(methods)!=set(METHODS) or len(methods)!=19:
        raise ValueError("RX freeze must declare all 18 select modes plus highlow")
    planned=[str(s) for s in freeze["config"].get(phase+"_seeds",[])]
    if not planned or len(set(planned))!=len(planned):raise ValueError("missing distinct planned phase seeds")
    records={};standard=[];manifest=[];errors=[];input_failed=set()
    for path in sorted(directory.glob("trial*.json.gz")):
        try:
            before=path.stat();data=path.read_bytes();after=path.stat()
            digest=hashlib.sha256(data).hexdigest()
            manifest.append(dict(name=path.name,sha256=digest,bytes=len(data)))
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise ValueError("record changed during read")
            trial=json.loads(gzip.decompress(data))
            if collector._identity(trial)!=identity:raise ValueError("trial/freeze identity mismatch")
            if trial.get("source_unchanged") is not True:raise ValueError("scientific source changed during trial")
            if any(trial.get("selection",{}).get(k,False) is not False for k in ("uses_truth","uses_validation","uses_future_noise")):
                raise ValueError("selector information boundary violated")
            if any(trial.get(k,False) is not False for k in ("validation_used_by_selector","future_noise_used_by_selector")):
                raise ValueError("trial information boundary violated")
            seed=str(trial.get("seed"))
            if seed not in planned or seed in records:raise ValueError("unplanned or duplicate history")
            parsed=collector.make_record(trial,methods)
            extracted=_extract(trial,parsed,zero_tolerance,substantive_tolerance)
            records[seed]=extracted;standard.append(parsed)
        except (ValueError,KeyError,TypeError,OSError,EOFError,gzip.BadGzipFile) as error:
            seed_hint=path.name.removeprefix("trial_").removesuffix(".json.gz")
            if seed_hint in planned:input_failed.add(seed_hint)
            errors.append(dict(file=path.name,seed_hint=seed_hint,error=type(error).__name__+": "+str(error)))
    input_failed-=set(records)
    pending=[s for s in planned if s not in records and s not in input_failed]
    methods_summary={}
    for name in METHODS:
        rows=[r["methods"][name] for r in records.values()]
        designs=[r["design"] for r in rows]
        proxy=[r["proxy"] for r in designs]
        methods_summary[name]=dict(planned=len(planned),observed=len(rows)+len(input_failed),pending=len(pending),
            complete=sum(r["status"]=="complete" for r in rows),failed=sum(r["status"]!="complete" for r in rows)+len(input_failed),
            input_record_failures=len(input_failed),
            failure_reasons=dict(Counter(v for r in rows for v in set(r["failure_reasons"]))),
            M=stats([r["M"] for r in rows],len(planned)),
            clean_future_M=stats([r["clean_future_M"] for r in rows],len(planned)),
            energy={k:stats([r["energy"].get(k) for r in rows],len(planned)) for k in
                ("grid_energy_kwh","throughput_kwh","driving_kwh","worker_minutes","moves")},
            selected_origin=dict(Counter(str(r["selected_origin"]) for r in designs)),
            new_MILP=sum(r["selected_is_new_MILP"] is True for r in designs) if any(r["new_MILP_flag_available"] for r in designs) else None,
            substantive_new_MILP=sum(r["substantive_new_MILP"] is True for r in designs) if any(r["new_MILP_flag_available"] for r in designs) else None,
            new_MILP_flags_available=sum(r["new_MILP_flag_available"] for r in designs),
            proxy_available=sum(r["available"] for r in proxy),proxy_completed=sum(r["completed"] for r in proxy),
            proxy_gap=stats([r["gap"] for r in proxy],len(planned)),
            proxy_target=stats([r["target_gap"] for r in proxy],len(planned)),
            proxy_all_pairs_required=sum(r["all_expected_pairs_required"] is True for r in proxy),
            selected_AC_all_pairs=sum(r["selected_prediction"]["complete"] for r in designs),
            near_zero_pairs=stats([r["selected_prediction"]["near_zero_pairs"]
                if r["selected_prediction"]["complete"] else None for r in designs],len(planned)))
    n=len(planned)
    status="partial" if pending else ("complete" if not errors and all(r["status"]=="complete" for r in records.values()) else "complete_with_failures")
    result=dict(directory=str(directory.resolve()),status=status,planned_seeds=planned,
        observed_seeds=sorted(set(records)|input_failed),parsed_seeds=sorted(records),pending_seeds=pending,
        input_failed_seeds=sorted(input_failed),input_errors=errors,
        planned_histories=n,observed_histories=len(records)+len(input_failed),complete_histories=sum(r["status"]=="complete" for r in records.values()),
        failed_histories=sum(r["status"]!="complete" for r in records.values())+len(input_failed),
        planned_method_slots=19*n,observed_method_slots=19*(len(records)+len(input_failed)),
        complete_method_slots=sum(m["complete"] for m in methods_summary.values()),
        failed_method_slots=sum(m["failed"] for m in methods_summary.values()),pending_method_slots=19*len(pending),
        method_count=19,methods=methods_summary,records=records,**identity,
        config=freeze["config"],
        input_files=manifest,freeze_sha256=hashlib.sha256((directory/"freeze.json").read_bytes()).hexdigest(),
        collector_observed_summary=collector._summary(standard,methods),
        parameter_error={k:stats([r["parameter"].get(k+"_relative_L2") for r in records.values()],n) for k in ("R","X")},
        runtime_seconds=stats([r["seconds"] for r in records.values()],n),
        cost={k:stats([r["cost"].get(k) for r in records.values()],n)
              for k in sorted({key for r in records.values() for key in r["cost"]})})
    return result


def _contrast(records,left,right,planned,tolerance):
    rows=[]
    for seed in planned:
        row=records.get(seed);a=row["methods"].get(left) if row else None;b=row["methods"].get(right) if row else None
        good=bool(a and b and a["status"]==b["status"]=="complete" and _finite(a["M"]) and _finite(b["M"]))
        clean=bool(good and _finite(a.get("clean_future_M")) and _finite(b.get("clean_future_M")))
        rows.append(dict(seed=seed,complete=good,pending=row is None,
            left_status=None if a is None else a["status"],right_status=None if b is None else b["status"],
            delta_M=a["M"]-b["M"] if good else None,
            delta_clean_future_M=a["clean_future_M"]-b["clean_future_M"] if clean else None,
            energy_delta={k:a["energy"][k]-b["energy"][k] if a and b and _finite(a["energy"].get(k)) and _finite(b["energy"].get(k)) else None
                for k in ("grid_energy_kwh","throughput_kwh","driving_kwh","worker_minutes","moves")}))
    return dict(left=left,right=right,rows=rows,delta_M=stats([r["delta_M"] for r in rows],len(planned),tolerance=tolerance),
        delta_clean_future_M=stats([r["delta_clean_future_M"] for r in rows],len(planned),tolerance=tolerance),
        energy_delta={k:stats([r["energy_delta"][k] for r in rows],len(planned)) for k in rows[0]["energy_delta"]})


def paired_domains(box,cone,*,tolerance=1e-4):
    planned=box["planned_seeds"]
    if set(planned)!=set(cone["planned_seeds"]):raise ValueError("paired directories must declare the same history seeds")
    config_differences=[k for k in set(box["config"])|set(cone["config"]) if box["config"].get(k)!=cone["config"].get(k)]
    config_ok=config_differences==["training_domain"] or set(config_differences)=={"training_domain"}
    manifest_ok=box["source_manifest_sha256"]==cone["source_manifest_sha256"]
    rows=[]
    for seed in planned:
        a,b=box["records"].get(seed),cone["records"].get(seed)
        checks={k:a["fingerprints"][k]==b["fingerprints"][k] for k in a["fingerprints"]} if a and b else {}
        paired=bool(a and b and config_ok and manifest_ok and all(checks.values()) and a["family"]==b["family"])
        row=dict(seed=seed,paired=paired,pending=seed in box["pending_seeds"] or seed in cone["pending_seeds"],
            input_record_failed=seed in box["input_failed_seeds"] or seed in cone["input_failed_seeds"],input_pairing_checks=checks,
            R_error_delta=(b["parameter"].get("R_relative_L2")-a["parameter"].get("R_relative_L2"))
                if paired and _finite(a["parameter"].get("R_relative_L2")) and _finite(b["parameter"].get("R_relative_L2")) else None,
            X_error_delta=(b["parameter"].get("X_relative_L2")-a["parameter"].get("X_relative_L2"))
                if paired and _finite(a["parameter"].get("X_relative_L2")) and _finite(b["parameter"].get("X_relative_L2")) else None,
            methods={},common_response={})
        for name in METHODS:
            aa=a["methods"][name] if a else {};bb=b["methods"][name] if b else {}
            good=paired and aa.get("status")==bb.get("status")=="complete" and _finite(aa.get("M")) and _finite(bb.get("M"))
            row["methods"][name]=dict(complete=good,delta_M=bb["M"]-aa["M"] if good else None,
                delta_clean_future_M=bb["clean_future_M"]-aa["clean_future_M"] if good and _finite(aa.get("clean_future_M")) and _finite(bb.get("clean_future_M")) else None)
        for name in ("constant","highlow"):
            aa=a["response"].get(name,{}) if a else {};bb=b["response"].get(name,{}) if b else {}
            common=bool(paired and aa.get("success") and bb.get("success")
                and a["methods"][name]["physical_available"] and b["methods"][name]["physical_available"]
                and a["methods"][name]["power_sha256"]==b["methods"][name]["power_sha256"]
                and a["methods"][name]["physical_sha256"]==b["methods"][name]["physical_sha256"])
            channels={}
            for channel in CHANNELS:
                x=aa.get("channels",{}).get(channel,{});y=bb.get("channels",{}).get(channel,{})
                same_increment=bool(common and x.get("true_increment")==y.get("true_increment"))
                channels[channel]=dict(complete=same_increment,
                    box_RMSE=x.get("RMSE") if same_increment else None,cone_RMSE=y.get("RMSE") if same_increment else None,
                    delta_RMSE=y["RMSE"]-x["RMSE"] if same_increment and _finite(x.get("RMSE")) and _finite(y.get("RMSE")) else None,
                    delta_relative_L2=y["relative_L2"]-x["relative_L2"] if same_increment and _finite(x.get("relative_L2")) and _finite(y.get("relative_L2")) else None)
            row["common_response"][name]=dict(common_curve_and_actual_physics=common,channels=channels)
        rows.append(row)
    n=len(planned)
    return dict(sign="cone minus box; negative parameter/response error is improvement",planned_histories=n,
        paired_histories=sum(r["paired"] for r in rows),pending_histories=sum(r["pending"] for r in rows),
        pairing_failures=sum(not r["paired"] and not r["pending"] for r in rows),
        config_differences=sorted(config_differences),allowed_config_difference=config_ok,source_manifest_same=manifest_ok,rows=rows,
        R_error_delta=stats([r["R_error_delta"] for r in rows],n),X_error_delta=stats([r["X_error_delta"] for r in rows],n),
        methods={name:dict(delta_M=stats([r["methods"][name]["delta_M"] for r in rows],n,tolerance=tolerance),
            delta_clean_future_M=stats([r["methods"][name]["delta_clean_future_M"] for r in rows],n,tolerance=tolerance)) for name in METHODS},
        common_response={name:{c:dict(stats([r["common_response"][name]["channels"][c]["delta_RMSE"] for r in rows],n),
            box_RMSE=stats([r["common_response"][name]["channels"][c].get("box_RMSE") for r in rows],n),
            cone_RMSE=stats([r["common_response"][name]["channels"][c].get("cone_RMSE") for r in rows],n),
            delta_relative_L2=stats([r["common_response"][name]["channels"][c]["delta_relative_L2"] for r in rows],n)) for c in CHANNELS}
                         for name in ("constant","highlow")})


def analyze(box_directory,cone_directory,*,phase="development",zero_tolerance=1e-8,comparison_tolerance=1e-4):
    if not all(_finite(v) and v>=0 for v in (zero_tolerance,comparison_tolerance)):
        raise ValueError("tolerances must be finite and nonnegative")
    groups={name:read_domain(path,phase=phase,zero_tolerance=zero_tolerance,substantive_tolerance=comparison_tolerance)
            for name,path in (("box",box_directory),("cone",cone_directory))}
    if groups["box"]["config"].get("training_domain")!="box" or groups["cone"]["config"].get("training_domain")!="cone":
        raise ValueError("box/cone directory labels differ from frozen training_domain")
    paired=paired_domains(groups["box"],groups["cone"],tolerance=comparison_tolerance)
    for group in groups.values():
        group["nearest_contrasts"]={d+"_vs_"+p:_contrast(group["records"],"nearest_"+d,p+"_"+d,
            group["planned_seeds"],comparison_tolerance) for d in DOMAINS for p in ("square","mean","linear")}
    complete=all(g["status"]=="complete" for g in groups.values()) and paired["paired_histories"]==paired["planned_histories"]
    return dict(schema="ev_rx_paired_saved_analysis_v1",phase=phase,status="complete" if complete else "partial_or_failed",
        independent_unit="one shared base history; box/cone, methods and noise reuses are paired",calibrated=False,
        no_rejection_or_topology_probability_claim=True,groups=groups,paired=paired,
        analyzer=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            collector_sha256=hashlib.sha256(Path(collector.__file__).read_bytes()).hexdigest(),
            imports="standard library and pure saved-record collector only",new_fit_calls=0,new_AC_calls=0,new_selector_calls=0),
        scope="Saved development diagnostics only; no policy, threshold, raw record, or scientific source is modified. M and design distance are not topology probabilities.")


def markdown(report):
    def fmt(value):return "pending" if value is None else f"{value:.5g}"
    groups=report["groups"];paired=report["paired"]
    lines=["# R/X包络：开发配对只读分析","",f"状态：{report['status']}。配对历史 {paired['paired_histories']}/{paired['planned_histories']}；统计单位为共享基础历史。未完成条件保留pending，未校准新策略，不报告拒绝率或正确概率。","",
        "|条件|已读/计划历史|完整/失败方法槽|待到达方法槽|耗时总和(s)|",
        "|---|---:|---:|---:|---:|"]
    for name,g in groups.items():lines.append(f"|{name}|{g['observed_histories']}/{g['planned_histories']}|{g['complete_method_slots']}/{g['failed_method_slots']}|{g['pending_method_slots']}|{fmt(g['runtime_seconds']['sum'])}|")
    lines += ["","每历史完整分母19方法；运行时读取错误、训练/未来失败均保留在JSON。成本为完整研究工作流，不能当作单策略部署耗时。","",
        "|方法|box平均M|cone平均M|配对cone−box|有效配对/计划|","|---|---:|---:|---:|---:|"]
    for name in METHODS:
        s=paired["methods"][name]["delta_M"]
        lines.append(f"|{name}|{fmt(groups['box']['methods'][name]['M']['mean'])}|{fmt(groups['cone']['methods'][name]['M']['mean'])}|{fmt(s['mean'])}|{s['valid']}/{s['total']}|")
    lines += ["","|同域最近竞争者−对照|box平均ΔM|cone平均ΔM|cone正/同/负|有效/计划|","|---|---:|---:|---:|---:|"]
    for d in DOMAINS:
        for p in ("square","mean","linear"):
            a=groups["box"]["nearest_contrasts"][d+"_vs_"+p]["delta_M"]
            b=groups["cone"]["nearest_contrasts"][d+"_vs_"+p]["delta_M"]
            lines.append(f"|{d} vs {p}|{fmt(a['mean'])}|{fmt(b['mean'])}|{b['positive']}/{b['ties']}/{b['negative']}|{b['valid']}/{b['total']}|")
    lines += ["","|真候选训练相对L2误差|box均值|cone均值|配对cone−box|","|---|---:|---:|---:|"]
    for key in ("R","X"):
        lines.append(f"|{key}|{fmt(groups['box']['parameter_error'][key]['mean'])}|{fmt(groups['cone']['parameter_error'][key]['mean'])}|{fmt(paired[key+'_error_delta']['mean'])}|")
    lines += ["","|共同曲线：仅线路参数导致的增量响应误差|电压RMSE box→cone|电压RMSE变化|根P RMSE变化|根Q RMSE变化|","|---|---:|---:|---:|---:|"]
    for name,channels in paired["common_response"].items():
        v=channels["v"]
        lines.append(f"|{name}|{fmt(v['box_RMSE']['mean'])}→{fmt(v['cone_RMSE']['mean'])}|"+"|".join(fmt(channels[c]["mean"]) for c in CHANNELS)+"|")
    lines += ["","只有同命令、同实际物理响应且保存真增量一致的constant/highlow才进入上表；这些真值诊断只离线使用。其他动作随策略改变，不能归为参数单因素。","",
        "|cone最近竞争者域|新MILP/实质新计划|代理达标/可用|完整AC12任务/计划|平均近零对数|","|---|---:|---:|---:|---:|"]
    for d in DOMAINS:
        s=groups["cone"]["methods"]["nearest_"+d]
        lines.append(f"|{d}|{s['new_MILP']}/{s['substantive_new_MILP']}|{s['proxy_completed']}/{s['proxy_available']}|{s['selected_AC_all_pairs']}/{s['planned']}|{fmt(s['near_zero_pairs']['mean'])}|")
    lines += ["","|cone域：资源均值|nearest净取电/吞吐(kWh)|square净取电/吞吐(kWh)|nearest驾驶(kWh)/人员(min)|nearest移动总次数|","|---|---:|---:|---:|---:|"]
    for d in DOMAINS:
        a=groups["cone"]["methods"]["nearest_"+d]["energy"];b=groups["cone"]["methods"]["square_"+d]["energy"]
        lines.append(f"|{d}|{fmt(a['grid_energy_kwh']['mean'])}/{fmt(a['throughput_kwh']['mean'])}|{fmt(b['grid_energy_kwh']['mean'])}/{fmt(b['throughput_kwh']['mean'])}|{fmt(a['driving_kwh']['mean'])}/{fmt(a['worker_minutes']['mean'])}|{fmt(a['moves']['sum'])}|")
    lines += ["","逐对距离、零对阈值、代理间隙、继承来源、能量/吞吐/驾驶和人员成本、全部失败、源码/输入哈希见summary.json。均值依据当前有效记录；不因部分完成而补值或外推。",""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-directory",type=Path,required=True)
    parser.add_argument("--cone-directory",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True,help="derived output directory")
    parser.add_argument("--phase",choices=("development","calibration","evaluation"),default="development")
    args=parser.parse_args()
    if args.output.resolve() in (args.box_directory.resolve(),args.cone_directory.resolve()):
        raise ValueError("derived outputs must have their own directory")
    report=analyze(args.box_directory,args.cone_directory,phase=args.phase)
    args.output.mkdir(parents=True,exist_ok=True)
    collector.write_json(args.output/"summary.json",report)
    (args.output/"summary.md").write_text(markdown(report),encoding="utf-8")
    print(json.dumps(dict(status=report["status"],paired=report["paired"]["paired_histories"],
        planned=report["paired"]["planned_histories"],output=str(args.output.resolve()))))


if __name__=="__main__":main()
