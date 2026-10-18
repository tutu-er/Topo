"""Package completed reference-network evidence without copying every large raw file.

The package command is gated on statistical_complete, audit.json and summary.json.
It does not fit, plan, tune, regenerate observations or modify scientific files.
The self-test command uses synthetic dictionaries or an explicitly named saved
development record.  Evaluation records are read only by the gated package step.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import importlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath
import shutil
import sys
import uuid
import zipfile


DEFAULT_ROOT=Path(__file__).resolve().parents[2]
PHASES=("development","calibration","evaluation")
OMITTED=("historical and future meter arrays outside the registered full development fixtures",
    "full per-time fitted source sequences", "optimizer restarts and duplicated residuals",
    "dense operation vectors, branch-and-bound iterations and solver matrices",
    "stored response Jacobians and forecasting fit internals", "development-only clean refits")


def digest(path):
    h=sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def read(path):
    path=Path(path);data=path.read_bytes()
    if path.suffix==".gz":data=gzip.decompress(data)
    return json.loads(data)


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if hasattr(value,"tolist"):return clean(value.tolist())
    if hasattr(value,"item"):return clean(value.item())
    return value


def canonical(value):
    return json.dumps(clean(value),ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(",",":")).encode("utf-8")


def object_digest(value):return sha256(canonical(value)).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);data=canonical(value)+b"\n"
    if path.suffix==".gz":data=gzip.compress(data,compresslevel=9,mtime=0)
    path.write_bytes(data)


def pick(value,keys):
    return {k:deepcopy(value[k]) for k in keys if k in value}


def compact_entry(entry):
    result=pick(entry,("loss","success","usable","error","selected_origin","optimization_failure",
        "global_optimality_certified","loss_scope"))
    if entry.get("state"):
        result["state"]=pick(entry["state"],("r","x"))
    return result


def compact_history(history):
    if history is None:return None
    result=pick(history,("complete","candidate_names","failures","counters","seconds","scope"))
    result["candidates"]={name:compact_entry(entry) for name,entry in history.get("candidates",{}).items()}
    return result


def compact_refit(refit):
    if refit is None:return None
    result=pick(refit,("complete","candidate_names","losses","failures","counters","seconds",
        "keep_all_on_failure","retained_on_failure","scope","shared_lines_history_and_future"))
    result["comparisons"]={}
    for name,row in refit.get("comparisons",{}).items():
        result["comparisons"][name]=pick(row,("success","error","joint_no_worse_than_fixed","same_line_vector_all_times"))
        for route in ("fixed","joint"):
            if route in row:result["comparisons"][name][route]=compact_entry(row[route])
    return result


def compact_passive(passive):
    if not passive:return passive
    result=pick(passive,("schema","seconds","n_arrived","n_terminal","generated_count","retained_count",
        "truncated_count","failures","budget","source_inputs_empty","source_estimated_by_wzzt",
        "extra_likelihood","ranking_is_AC_evidence","coverage","disputed_nodes","disputed_supports"))
    result["native"]={kind:[pick(row,("origin","supports")) for row in passive.get("native",{}).get(kind,[])]
        for kind in ("rnj","wzzt")}
    return result


def compact_record(raw):
    """Retain the sufficient inputs read by every primary statistical helper.

    The shape deliberately remains compatible with reference_benchmark.analysis.
    The result supports statistics, including R/X error, but is not an AC
    physical-replay record; predeclared original fixtures and the local originals
    supply that separate evidence.
    """
    result=pick(raw,("spec","config","freeze_identity","source_unchanged","status","seconds","failures",
        "truth_evaluation_only","candidate_structures"))
    result["passive_pre"]=compact_passive(raw.get("passive_pre",{}))
    planning=raw.get("planning",{});reduced=pick(planning,("cost","failures","tasks"))
    reduced["history"]=compact_history(planning.get("history",{}))
    context=planning.get("context",{});reduced["context"]=pick(context,("ac_calls",))
    if "forecast" in context:
        reduced["context"]["forecast"]=pick(context["forecast"],("p","q","root_v","source_by_candidate"))
    reduced["plans"]={name:pick(plan,("power","action_key","domain","physicalcheck","fallback_reason","primary","passive_control"))
        for name,plan in planning.get("plans",{}).items()}
    reduced["nominal_AC"]={name:pick(row,("incremental_score","total_score","failure"))
        for name,row in planning.get("nominal_AC",{}).items()}
    reduced["families"]={}
    for method,domains in planning.get("families",{}).items():
        reduced["families"][method]={domain:pick(row,("completed","unfinished","success","score","objective_lower",
            "objective_upper","gap","failures","seconds","best_parent_score","inheritance_protected",
            "power_kw","task_values","target_gap","n_iterations","selected_origin")) for domain,row in domains.items()}
    result["planning"]=reduced
    actions={}
    for key,action in raw.get("evaluation",{}).get("actions",{}).items():
        item=pick(action,("methods","power","primary","physical","physical_generation_failed","failures",
            "candidate_union","new_candidates","truth_inputs_evaluation_only","pre_refit_seconds"))
        item["pre_refit"]=compact_refit(action.get("pre_refit"))
        item["post_refit"]=compact_refit(action.get("post_refit"))
        item["post_history"]=compact_history(action.get("post_history"))
        item["post_passive"]=compact_passive(action.get("post_passive"))
        actions[key]=item
    result["evaluation"]=dict(actions=actions)
    result["compact_evidence"]=dict(schema="reference_benchmark_statistical_inputs_v1",
        statistical_replay_only=True,replaces_original_raw=False,
        discarded=list(OMITTED),truth_scope="evaluation labels only; never planning inputs")
    return result


def first_difference(a,b,path="$",atol=0.):
    a,b=clean(a),clean(b)
    if isinstance(a,dict) and isinstance(b,dict):
        if set(a)!=set(b):return path+" keys "+repr(sorted(set(a)^set(b)))
        for key in a:
            difference=first_difference(a[key],b[key],path+"."+key,atol)
            if difference:return difference
        return None
    if isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):return path+" length"
        for i,(left,right) in enumerate(zip(a,b)):
            difference=first_difference(left,right,path+"[%d]"%i,atol)
            if difference:return difference
        return None
    if isinstance(a,(int,float)) and isinstance(b,(int,float)) and not isinstance(a,bool) and not isinstance(b,bool):
        return None if abs(a-b)<=atol else path+" numeric"
    return None if a==b else path+" value"


def load_analysis(source_root):
    root=Path(source_root).resolve()
    expected=root/"research_experiments/mobile_ev_mle/reference_benchmark/analysis.py"
    if not expected.exists():raise FileNotFoundError(expected)
    sys.path[:0]=[str(root),str(root/"rnj_wzzt_core")]
    module=importlib.import_module("research_experiments.mobile_ev_mle.reference_benchmark.analysis")
    if Path(module.__file__).resolve()!=expected.resolve():
        raise RuntimeError("postprocessing imported from an unexpected source tree: "+str(module.__file__))
    return module


def condition_sufficiency(raw,reduced,analysis,thresholds,methods):
    names=list(dict.fromkeys([*methods,*raw.get("planning",{}).get("plans",{})]))
    expected=analysis.decision_rows([raw],thresholds,methods=names)
    actual=analysis.decision_rows([reduced],thresholds,methods=names)
    difference=first_difference(expected,actual)
    if difference:raise RuntimeError("compact decisions differ at "+difference)
    left,right=analysis.condition_detail(raw,methods),analysis.condition_detail(reduced,methods)
    difference=first_difference(left,right)
    if difference:raise RuntimeError("compact condition statistics differ at "+difference)
    return dict(decision_sha256=object_digest(expected),condition_detail_sha256=object_digest(left),
        main_methods=len(methods),saved_plan_methods=len(names),passed=True,
        forecast_error=left.get("forecast_error"))


def guard_output(output):
    output=Path(output)
    required=("controller.json","freeze.json","environment.json","source_snapshot.zip","thresholds.json","audit.json","summary.json","results.md")
    missing=[name for name in required if not (output/name).is_file()]
    if not (output/"controller.json").is_file():raise RuntimeError("controller receipt absent; packaging is not enabled")
    controller=read(output/"controller.json")
    if controller.get("status")!="statistical_complete":
        raise RuntimeError("packaging waits for controller statistical_complete; current status="+str(controller.get("status")))
    phases=controller.get("phases",{})
    for phase in PHASES:
        counts=phases.get(phase,{})
        if not counts or counts.get("remaining")!=0 or counts.get("computed")!=counts.get("total"):
            raise RuntimeError("controller condition receipt is incomplete: "+phase)
    if missing:raise RuntimeError("completed postprocessing files missing: "+repr(missing))
    audit=read(output/"audit.json");summary=read(output/"summary.json");frozen=read(output/"freeze.json")
    if not audit.get("passed"):raise RuntimeError("audit has unresolved issues; package must not be labelled verified")
    if summary.get("policy_identity")!=frozen.get("identity") or controller.get("freeze_identity")!=frozen.get("identity"):
        raise RuntimeError("controller, summary and freeze refer to different scientific versions")
    return controller,frozen,summary


def safe_member(name):
    p=PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
        raise ValueError("unsafe archive member: "+name)
    return p


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise RuntimeError("duplicate archive member")
        for name in names:safe_member(name)
        manifest=json.loads(archive.read("manifest.json"))
        expected=set(manifest["files"])|{"manifest.json"}
        if set(names)!=expected:raise RuntimeError("archive members differ from manifest")
        for name,row in manifest["files"].items():
            data=archive.read(name)
            if len(data)!=row["bytes"] or sha256(data).hexdigest()!=row["sha256"]:
                raise RuntimeError("archive byte verification failed: "+name)
    return dict(passed=True,files=len(manifest["files"]),archive_sha256=digest(path),archive_bytes=Path(path).stat().st_size)


def replay_statistics(bundle,source_root,*,write_receipt=True):
    """Recompute all primary tables, intervals and paired statistics; no AC calls."""
    bundle=Path(bundle);analysis=load_analysis(source_root)
    thresholds=read(bundle/"statistics/thresholds.json")
    frozen=read(bundle/"provenance/freeze.json");config=frozen["config"];methods=tuple(config["methods"]);records={}
    def load_phase(phase):
        values=[read(path) for path in sorted((bundle/"compact"/phase).glob("*.json.gz"))]
        values.sort(key=lambda r:(r["spec"]["cluster_index"],analysis.network_name(r["spec"]),r["spec"]["source_mode"]))
        return values
    # Even replay reconstructs the calibration rule before opening evaluation
    # compact records. The original frozen threshold remains the authority.
    records["calibration"]=load_phase("calibration")
    recal=analysis.calibration_thresholds(records["calibration"],methods=methods,alpha=config["alpha"],source_modes=config["source_modes"])
    for key,value in recal.items():
        difference=first_difference(value,thresholds[key])
        if difference:raise RuntimeError("replayed calibration differs: "+key+" "+difference)
    expected=read(bundle/"statistics/summary.json.gz")
    if digest(Path(analysis.__file__))!=expected.get("analysis_sha256") or tuple(expected["methods"])!=methods:
        raise RuntimeError("statistical replay source/methods differ from the recorded summary")
    for phase in ("development","evaluation"):records[phase]=load_phase(phase)
    B=int(expected["statistical_bootstrap"]);seed=int(expected["statistical_seed"]);checks={}
    for phase in PHASES:
        actual=analysis.phase_summary(records[phase],thresholds,methods,bootstrap=B,seed=seed)
        difference=first_difference(actual,expected["phases"][phase])
        if difference:raise RuntimeError("replayed phase differs: "+phase+" "+difference)
        checks[phase]=dict(passed=True,n_conditions=len(records[phase]),phase_summary_sha256=object_digest(actual))
    spatial=analysis.small_spatial_ablation(records["evaluation"],thresholds,bootstrap=B,seed=seed)
    difference=first_difference(spatial,expected.get("spatial_ablation"))
    if difference:raise RuntimeError("replayed spatial ablation differs: "+difference)
    receipt=dict(schema="reference_benchmark_statistical_replay_v1",passed=True,phases=checks,
        spatial_ablation_passed=True,calibration_source_blind_passed=True,bootstrap=B,seed=seed,
        raw_AC_replay_performed=False,fitting_calls=0,planning_calls=0,
        original_threshold_identity_preserved=thresholds.get("identity"),
        original_raw_byte_identity_recreated=False,
        compared_calibration_payload_only=True,
        scope="all main numerical statistical objects exactly regenerated; original threshold file identity is preserved, not regenerated from compact bytes; original full AC audit is separate")
    if write_receipt:write(bundle/"statistics/replay_receipt.json",receipt)
    return receipt


def verify_snapshot(path,frozen):
    expected=frozen.get("source_sha256",{})
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(expected):
            raise RuntimeError("source snapshot member manifest differs from freeze")
        for name in names:
            safe_member(name)
            if sha256(archive.read(name)).hexdigest()!=expected[name]:
                raise RuntimeError("source snapshot hash differs: "+name)
    return dict(passed=True,files=len(expected))


def concise_report(summary,audit,frozen):
    config=frozen["config"];evaluation=summary["phases"]["evaluation"]
    lines=["# 参考网络 EV 测试：完成批次的固定统计", "",
        "冻结版本：`"+str(frozen["identity"])+"`。统计单位为互不重叠的日期窗，同日期的网络及源条件配对。",
        "", "|网络|策略|条件数|唯一正确|唯一错误|未决|计算/物理不可用|严格精确|F1均值|收缩诊断精确|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for network,group in evaluation.get("by_network",{}).items():
        for method in summary["methods"]:
            row=group["tables"][method]["postJNT"];clades=row.get("clades",{});f1=clades.get("f1",{}).get("mean")
            lines.append("|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|"%(network,method,row["n_total_conditions"],
                row["unique_correct"],row["unique_wrong"],row["unresolved"],row["fit_failure"],clades.get("exact",0),
                "—" if f1 is None else "%.4f"%f1,clades.get("contracted_diagnostic",{}).get("exact",0)))
    lines += ["", "网络与数据范围：CIGRE 采用原居民子网及原变压器；case33 使用原标称电压、额定 P/Q 与线路。paper11/paper15 为项目兼容树，去掉旧线路放大并加入公开型变压器；它们属于已声明的修改算例，不作为作者原算例完整复现。",
        "", "逐时 P/Q 来自 SimBench 1.6.3 的六条 LV 聚合曲线，固定种子分配并允许复用。历史中心化 P/Q 设计秩至多 12；32 节点不等于 32 条独立住户实测曲线。开发在 Q1、校准在 Q2/Q3、评价在 Q4，跨季节结果按本模拟分布解释。源和计量噪声为已声明仿真，尚无现场大规模验证。",
        "", "P-SQ 为同可行域平方强基线；G-SQ/G-AQ 加入历史候选的响应差信息。F± 固定地点双向；无后缀表示允许移动双向。移动域可保留父计划，四嵌套域的规划分数在同一目标下检查继承；实际 AC 收益另列，规划分数不自动等于排错收益。",
        "", "每拓扑一组历史 R/X 用于规划，未来负荷由开发期选定的因果预测器生成。preFIX 固定历史线路；preJNT 在新增计量到达后共享历史和新增数据联合拟合；postJNT 再加入观测驱动候选。表为 postJNT，其他路线、同域配对差与区间均完整保存在 summary.json.gz。",
        "", "成本中分别保留共享历史拟合、预测/响应系数、各域求解、父计划继承和实际量测后拟合；相同动作的缓存执行属于计算复用，不能作为多个独立物理试验。完整失败、候选漏真、未决分母和求解界不因缓存或继承删除。没有实际执行的纯充电域仅报告代理规划值。",
        "", "阈值先由校准记录冻结，再读取评价；未知源条件下取预定各源校准阈值的最大值。收缩近零边仅为固定阈值诊断，严格主终点保持不变。", "",
        "完整 AC 数值审计：%s；前向回放 %s 次；本后处理重新拟合 0 次、重新规划 0 次。"%("通过" if audit.get("passed") else "存在问题",audit.get("forward_AC_replay_calls",0)),
        "", "代表记录固定为每网络×源的最小开发 index，不按表现选取。核心包保留完整物理回放字段，并逐块记录删去的优化重启状态；全量原始代表单独附件，所有原始记录清单及字节 hash 保留。", ""]
    return "\n".join(lines)


def physical_projection(raw):
    omitted=[]
    def visit(value,path="$",in_restart=False):
        if isinstance(value,dict):
            result={}
            for key,item in value.items():
                child=path+"."+key
                if in_restart and key in ("requested_initial_state","initial_state","attained_state"):
                    data=canonical(item)
                    omitted.append(dict(path=child,sha256=sha256(data).hexdigest(),bytes=len(data),
                        reason="optimizer restart trajectory state; final candidate state, outcomes, failures and bounds retained"))
                else:result[key]=visit(item,child,in_restart)
            return result
        if isinstance(value,list):
            return [visit(item,path+"["+str(i)+"]",in_restart or path.endswith(".restarts")) for i,item in enumerate(value)]
        return value
    return visit(raw),omitted


def physical_equivalence(raw,reduced,analysis):
    before=analysis.old.Audit();after=analysis.old.Audit()
    analysis.audit_record(raw,before);analysis.audit_record(reduced,after)
    left,right=before.record(),after.record();difference=first_difference(left,right)
    if difference:raise RuntimeError("physical fixture audit differs at "+difference)
    if not left["passed"]:raise RuntimeError("representative physical audit has issues; retain failures and review before delivery")
    return dict(passed=True,original=left,projected=right,original_audit_sha256=object_digest(left),
        projected_audit_sha256=object_digest(right),fitting_calls=0,planning_calls=0)


def encode_shared(value,objects):
    """Lossless content addressing; equal blocks are stored once, including states."""
    if isinstance(value,dict):result={key:encode_shared(item,objects) for key,item in value.items()}
    elif isinstance(value,list):result=[encode_shared(item,objects) for item in value]
    else:return value
    data=canonical(result)
    if len(data)<2048:return result
    key=sha256(data).hexdigest()
    if key in objects and objects[key]!=result:raise RuntimeError("content hash collision")
    objects[key]=result
    return {"$evidence_sha256":key}


def decode_shared(value,objects,cache=None):
    cache={} if cache is None else cache
    if isinstance(value,dict) and set(value)=={"$evidence_sha256"}:
        key=value["$evidence_sha256"]
        if key not in cache:
            block=objects[key]
            if object_digest(block)!=key:raise RuntimeError("physical content block hash failed: "+key)
            cache[key]=decode_shared(block,objects,cache)
        return cache[key]
    if isinstance(value,dict):return {key:decode_shared(item,objects,cache) for key,item in value.items()}
    if isinstance(value,list):return [decode_shared(item,objects,cache) for item in value]
    return value


def physical_audit(bundle,source_root):
    bundle=Path(bundle);analysis=load_analysis(source_root)
    saved=read(bundle/"raw_fixtures/shared_records.json.gz");selection=read(bundle/"raw_fixtures/selection.json")
    results=[]
    for fixture in selection["fixtures"]:
        relative=fixture["original_path"];raw=decode_shared(saved["roots"][relative],saved["objects"])
        if object_digest(raw)!=fixture["physical_record_sha256"]:raise RuntimeError("reconstructed physical record hash differs: "+relative)
        audit=analysis.old.Audit();analysis.audit_record(raw,audit);row=audit.record()
        if object_digest(row)!=fixture["physical_audit_sha256"]:raise RuntimeError("physical replay audit differs: "+relative)
        results.append(dict(path=relative,passed=row["passed"],forward_AC_replay_calls=row["forward_AC_replay_calls"]))
    return dict(passed=all(r["passed"] for r in results),fixtures=results,fitting_calls=0,planning_calls=0)


def deterministic_zip(bundle,path):
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for item in sorted(Path(bundle).rglob("*")):
            if item.is_file():
                info=zipfile.ZipInfo(item.relative_to(bundle).as_posix(),date_time=(1980,1,1,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(info,item.read_bytes(),compresslevel=9)


def restored_audit(archive,destination):
    restored=restore(archive,Path(destination)/"restore_check")
    restored_bundle=Path(restored["destination"]);tool=restored_bundle/"postprocessing/package_evidence.py"
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    checks={}
    for mode in ("replay","physical-audit"):
        result=subprocess.run([sys.executable,str(tool),mode,str(restored_bundle),"--source-root",restored["source_root"]],
            cwd=restored_bundle,capture_output=True,text=True,encoding="utf-8",env=environment)
        log=Path(destination)/("restored_"+mode+".log");log.write_text(result.stdout+result.stderr,encoding="utf-8")
        if result.returncode:raise RuntimeError("restored "+mode+" failed; inspect "+str(log))
        value=json.loads(result.stdout.strip().splitlines()[-1])
        if not value.get("passed"):raise RuntimeError("restored "+mode+" did not pass")
        checks[mode]=value
    receipt=dict(passed=True,byte_restore=restored,independent_process_checks=checks,fitting_calls=0,planning_calls=0)
    write(Path(destination)/"restore_audit.json",receipt)
    return receipt


def instructions():
    return """# Evidence restore and replay

The core archive contains the complete frozen source snapshot, exact prepared V2
windows, data source/licence receipts, all compact statistical records, complete
summary/threshold/audit objects, and one predeclared physical fixture for every
network/source combination (minimum development index, selected without scores).
Each physical fixture retains observations, final R/X and source states, failures,
solver outcomes/bounds and denominators. Only restart requested/initial/attained
states are omitted; original path, canonical JSON hash and bytes are recorded.
Repeated physical blocks are losslessly content addressed. The full original
representatives are in a separate local raw attachment; no large file is sent to
Git automatically. raw_sha256.json lists every original condition byte hash.

Commands (versions in provenance/environment.json):

    python package_evidence.py verify EVIDENCE.zip
    python package_evidence.py restore EVIDENCE.zip NEW_EMPTY_DIRECTORY
    python NEW_EMPTY_DIRECTORY/postprocessing/package_evidence.py replay NEW_EMPTY_DIRECTORY --source-root NEW_EMPTY_DIRECTORY/source
    python NEW_EMPTY_DIRECTORY/postprocessing/package_evidence.py physical-audit NEW_EMPTY_DIRECTORY --source-root NEW_EMPTY_DIRECTORY/source

Restore verifies every outer byte hash and every frozen source file after extract.
Replay reconstructs calibration before evaluation, then all primary phase tables,
intervals, paired tests, R/X, forecasting, cost and ordering summaries. It performs
zero fitting, zero planning and zero AC. physical-audit reconstructs the registered
physical fixtures and reruns forward AC only, comparing each numerical audit with
its full-original receipt. All future truth remains labelled evaluation only.

The large SimBench CSV files are not duplicated. inputs/prepared.json.gz contains
the exact windows used. The official source URLs, checksums, package version and
ODbL/DbCL licence text are retained. Calibration compact bytes differ from the
original full raw files: replay compares the pure numerical calibration payload,
not the original raw-file identity. The original threshold identity is copied
unchanged and never recomputed or replaced. The full representative attachment
preserves original gzip bytes; core shared blocks restore the projected JSON
object exactly, not omitted restart trajectories or the original full gzip.
Six native LV aggregate profiles are reused;
this is not 32 independent residential time series. Development/calibration/
evaluation are Q1/Q2-Q3/Q4. Standard native networks and declared paper-style
modified compatibility cases are distinguished in the report. Optional SINTEF
metadata is source-audit-only and did not enter this scientific batch.
"""


def package(output,destination,*,source_root=DEFAULT_ROOT,max_mib=20.,verify_statistics=True):
    output=Path(output).resolve();destination=Path(destination).resolve();source_root=Path(source_root).resolve()
    controller,frozen,summary=guard_output(output)
    snapshot_receipt=verify_snapshot(output/"source_snapshot.zip",frozen)
    if destination.exists():raise FileExistsError("delivery destination must be new: "+str(destination))
    analysis=load_analysis(source_root)
    if digest(Path(analysis.__file__))!=summary.get("analysis_sha256"):
        raise RuntimeError("current analysis differs from the completed summary")
    thresholds=read(output/"thresholds.json");config=frozen["config"];methods=tuple(summary["methods"])
    prepared=Path(controller["prepared"])
    if digest(prepared)!=frozen["prepared_sha256"]:raise RuntimeError("prepared input hash differs from freeze")
    paths=[]
    for phase in PHASES:
        current=sorted((output/phase).glob("*.json.gz"))
        declared=analysis.condition_manifest(config,phase)
        if len(current)!=len(declared):raise RuntimeError("raw condition count differs: "+phase)
        paths.extend(current)
    destination.mkdir(parents=True);bundle=destination/"bundle";bundle.mkdir()
    sources={}
    def copy_file(src,relative):
        src=Path(src);target=bundle/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
        sources[relative]=dict(kind="byte-exact copy",original_sha256=digest(src))
    for name in ("freeze.json","environment.json","source_snapshot.zip"):
        copy_file(output/name,"provenance/"+name)
    for name in ("thresholds.json","audit.json","results.md"):
        copy_file(output/name,"statistics/"+name)
    # Preserve full summary losslessly, but avoid a large text duplicate.
    target=bundle/"statistics/summary.json.gz";target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(gzip.compress((output/"summary.json").read_bytes(),compresslevel=9,mtime=0))
    sources["statistics/summary.json.gz"]=dict(kind="lossless gzip of complete summary",original_sha256=digest(output/"summary.json"))
    copy_file(prepared,"inputs/prepared.json.gz")
    copy_file(prepared.parent/"manifest.json","inputs/manifest.json")
    for name in ("load_sources.md","case_sources.md","paper_original_availability.md","data_inventory.json",
            "household_profile_origin.json","SimBench_1.6.3_LICENSE.txt","MATPOWER_7.1_LICENSE.txt",
            "pandapower_v3.2.1_LICENSE.txt","upstream_licenses.json","integration_review.md",
            "profiles_validation.json","data_sources_validation.json","input_metadata_correction.json",
            "case33bw_matpower_7.1.m","cigre_networks_pandapower_v3.2.1.py"):
        src=source_root/"audit"/name
        if src.is_file():copy_file(src,"provenance/data_sources/"+name)
    for relative in ("data_external/README.md","data_external/simbench/acquisition.json",
            "data_external/sintef_cineldi_14528192/acquisition.json"):
        src=source_root/relative
        if src.is_file():copy_file(src,"provenance/"+relative)
    for name in ("diagnostics.json","diagnostics.md","diagnostics_receipt.json"):
        src=output/name
        if src.is_file():copy_file(src,"statistics/"+name)
    for name in ("diagnostics.py","diagnostics_selftest_receipt.json","package_selftest_receipt.json"):
        src=Path(__file__).parent/name
        if src.is_file():copy_file(src,"postprocessing/"+name)
    copy_file(Path(analysis.__file__),"postprocessing/reference_benchmark_analysis.py")
    copy_file(Path(__file__),"postprocessing/package_evidence.py")
    receipt=pick(controller,("schema","status","phases","freeze_identity","controller_policy_sha256",
        "requested_phases","started_utc","updated_utc","finished_utc","workers","per_worker_threads","exceptions"))
    receipt.update(original_controller_sha256=digest(output/"controller.json"),
        executor_paths_and_lock_tokens_omitted=True)
    write(bundle/"provenance/controller_statistical_complete.json",receipt)
    raw_registry={};sufficiency=[];seen_conditions=set()
    expected_conditions={(row["phase"],row["cluster_index"],row["network"],row["source_mode"],row["n"])
        for phase in PHASES for row in analysis.condition_manifest(config,phase)}
    for path in paths:
        relative=path.relative_to(output).as_posix();raw=read(path);reduced=compact_record(raw)
        if raw.get("freeze_identity")!=frozen["identity"] or raw.get("source_unchanged") is not True:
            raise RuntimeError("original raw version differs from completed freeze: "+relative)
        spec=raw["spec"];condition=(spec["phase"],spec["cluster_index"],analysis.network_name(spec),spec["source_mode"],spec["n"])
        if condition not in expected_conditions or condition in seen_conditions:
            raise RuntimeError("unexpected or duplicated condition during packaging: "+repr(condition))
        seen_conditions.add(condition)
        proof=condition_sufficiency(raw,reduced,analysis,thresholds,methods)
        reduced["compact_evidence"].update(original_raw_sha256=digest(path),original_raw_bytes=path.stat().st_size,
            sufficient_statistics_verification=proof)
        write(bundle/"compact"/relative,reduced)
        raw_registry[relative]=dict(sha256=digest(path),bytes=path.stat().st_size,status=raw.get("status"),
            compact_path="compact/"+relative,spec=raw["spec"])
        sufficiency.append(dict(path=relative,**pick(proof,("decision_sha256","condition_detail_sha256","passed"))))
    if seen_conditions!=expected_conditions:raise RuntimeError("raw conditions missing from evidence")
    write(bundle/"raw_sha256.json",dict(schema="reference_benchmark_original_raw_manifest_v1",files=raw_registry,
        n_files=len(paths),total_original_bytes=sum(v["bytes"] for v in raw_registry.values()),
        scope="all original raw files remain local; compacts are additional statistical evidence"))
    fixtures=[];physical_objects={};physical_roots={};original_fixtures=[];physical_proofs=[]
    declared=analysis.condition_manifest(config,"development")
    for network in config["network_names"]:
        for source in config["source_modes"]:
            index=min(row["cluster_index"] for row in declared if row["network"]==network and row["source_mode"]==source)
            chosen=[relative for relative,row in raw_registry.items() if row["spec"]["phase"]=="development"
                and analysis.network_name(row["spec"])==network and row["spec"]["source_mode"]==source
                and row["spec"]["cluster_index"]==index]
            if len(chosen)!=1:raise RuntimeError("registered raw fixture missing or duplicated: "+repr((network,source,index)))
            relative=chosen[0];raw=read(output/relative);projected,omissions=physical_projection(raw)
            proof=physical_equivalence(raw,projected,analysis)
            physical_roots[relative]=encode_shared(projected,physical_objects)
            rebuilt=decode_shared(physical_roots[relative],physical_objects)
            if first_difference(projected,rebuilt):raise RuntimeError("physical block reconstruction differs")
            fixtures.append(dict(original_path=relative,selection="minimum declared development cluster index",spec=raw_registry[relative]["spec"],
                original_raw_sha256=raw_registry[relative]["sha256"],original_JSON_semantic_sha256=object_digest(raw),
                physical_record_sha256=object_digest(projected),full_gzip_identity_in_core=False,
                physical_audit_sha256=proof["projected_audit_sha256"],omitted_blocks=omissions))
            physical_proofs.append(dict(path=relative,**proof));original_fixtures.append((relative,output/relative))
    write(bundle/"raw_fixtures/shared_records.json.gz",dict(schema="lossless_physical_content_addressing_v1",roots=physical_roots,objects=physical_objects))
    write(bundle/"raw_fixtures/selection.json",dict(before_outcomes=True,fixtures=fixtures))
    write(bundle/"statistics/physical_projection_receipt.json",dict(passed=True,fixtures=physical_proofs,
        fitting_calls=0,planning_calls=0,scope="full original and core physical projection receive identical forward audits"))
    attachment=destination/"raw_representatives.zip"
    attachment_files={relative:dict(bytes=path.stat().st_size,sha256=digest(path)) for relative,path in original_fixtures}
    attachment_manifest=dict(schema="original_representatives_v1",files=attachment_files,
        selection="every network/source, minimum declared development index; independent of outcomes")
    with zipfile.ZipFile(attachment,"w",compression=zipfile.ZIP_STORED) as archive:
        for relative,path in original_fixtures:
            info=zipfile.ZipInfo(relative,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED
            archive.writestr(info,path.read_bytes())
        archive.writestr("manifest.json",canonical(attachment_manifest))
    attachment_receipt=verify_archive(attachment)
    write(bundle/"provenance/original_representatives_attachment.json",dict(filename=attachment.name,**attachment_receipt,
        automatically_added_to_Git=False,separate_local_attachment=True))
    write(bundle/"statistics/compact_sufficiency_receipt.json",dict(passed=True,n_conditions=len(sufficiency),
        conditions=sufficiency,scope="decision and complete condition-detail objects agree before aggregation",
        fitting_calls=0,planning_calls=0,AC_calls=0))
    (bundle/"RESTORE.md").write_text(instructions(),encoding="utf-8")
    (bundle/"statistics/concise_report.md").write_text(concise_report(summary,read(output/"audit.json"),frozen),encoding="utf-8")
    if verify_statistics:replay=replay_statistics(bundle,source_root)
    else:
        replay=dict(passed=False,not_run=True,reason="explicit skip; aggregate replay remains outstanding")
        write(bundle/"statistics/replay_receipt.json",replay)
    files={}
    for path in sorted(bundle.rglob("*")):
        if path.is_file():
            relative=path.relative_to(bundle).as_posix()
            files[relative]=dict(bytes=path.stat().st_size,sha256=digest(path),**sources.get(relative,{}))
    manifest=dict(schema="reference_benchmark_evidence_package_v1",policy_identity=frozen["identity"],files=files,
        created_utc=datetime.now(timezone.utc).isoformat(),original_run_name=output.name,
        original_raw_files=len(raw_registry),original_raw_bytes=sum(v["bytes"] for v in raw_registry.values()),
        complete_raw_fixtures="minimum predeclared development cluster index, every network/source condition; selected without reading performance",
        discarded_from_compact=list(OMITTED),compact_replaces_original=False,
        statistical_replay_passed=replay.get("passed",False),all_byte_checksums_required=True,
        original_CSV_included=False,official_profile_hashes=read(prepared.parent/"manifest.json"),
        target_max_archive_bytes=int(max_mib*1024*1024))
    write(bundle/"manifest.json",manifest)
    archive_path=destination/(output.name+"_evidence.zip")
    with zipfile.ZipFile(archive_path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                info=zipfile.ZipInfo(path.relative_to(bundle).as_posix(),date_time=(1980,1,1,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED
                archive.writestr(info,path.read_bytes(),compresslevel=9)
    verification=verify_archive(archive_path)
    verification.update(statistical_replay_passed=replay.get("passed",False),source_snapshot_verified=snapshot_receipt,
        original_representatives_attachment=str(attachment),original_representatives_bytes=attachment.stat().st_size,
        manifest_sha256=digest(bundle/"manifest.json"),target_max_archive_bytes=int(max_mib*1024*1024),
        within_size_target=archive_path.stat().st_size<=max_mib*1024*1024)
    write(destination/"package_receipt.json",verification)
    restore_receipt=restored_audit(archive_path,destination)
    bundle_files={path.relative_to(bundle).as_posix():dict(bytes=path.stat().st_size,sha256=digest(path))
        for path in sorted(bundle.rglob("*")) if path.is_file()}
    delivery=dict(schema="reference_delivery_manifest_v1",policy_identity=frozen["identity"],
        ready_for_review=bool(verification["within_size_target"] and replay.get("passed") and restore_receipt["passed"]),
        bundle_directory="bundle",bundle_files=bundle_files,
        core_archive=dict(path=archive_path.name,bytes=archive_path.stat().st_size,sha256=digest(archive_path),
            duplicates_bundle_contents=True,default_Git_inclusion=False),
        full_attachment=dict(path=attachment.name,bytes=attachment.stat().st_size,sha256=digest(attachment),
            byte_verification_passed=attachment_receipt["passed"],default_Git_inclusion=False,local_only=True),
        restore_audit=dict(path="restore_audit.json",bytes=(destination/"restore_audit.json").stat().st_size,
            sha256=digest(destination/"restore_audit.json"),passed=restore_receipt["passed"]),
        package_receipt=dict(path="package_receipt.json",sha256=digest(destination/"package_receipt.json")),
        audit_passed=True,statistical_replay_passed=replay.get("passed",False),
        scope="publisher may commit bundle/ and this manifest; large duplicate archives remain local")
    write(destination/"delivery_manifest.json",delivery)
    if not verification["within_size_target"]:
        raise RuntimeError("verified core package exceeds size target; all required evidence retained at "+str(archive_path))
    return dict(archive=str(archive_path),delivery_manifest=str(destination/"delivery_manifest.json"),
        ready_for_review=delivery["ready_for_review"],**verification)


def extract_checked(archive_path,destination):
    destination=Path(destination).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            relative=safe_member(name);target=destination.joinpath(*relative.parts).resolve()
            if destination not in target.parents and target!=destination:
                raise ValueError("archive extraction escaped destination")
            if name.endswith("/"):target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():raise FileExistsError(target)
            target.write_bytes(archive.read(name))


def restore(archive_path,destination):
    verification=verify_archive(archive_path);destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError("restore destination must be new")
    destination.mkdir(parents=True);extract_checked(archive_path,destination)
    source=destination/"source";source.mkdir()
    extract_checked(destination/"provenance/source_snapshot.zip",source)
    analysis=source/"research_experiments/mobile_ev_mle/reference_benchmark/analysis.py"
    saved=destination/"postprocessing/reference_benchmark_analysis.py"
    if analysis.exists():
        if digest(analysis)!=digest(saved):raise RuntimeError("saved analysis conflicts with frozen source")
    else:
        analysis.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(saved,analysis)
    frozen_path=destination/"provenance/freeze.json"
    if frozen_path.exists():
        frozen=read(frozen_path)
        if "source_sha256" in frozen:
            verify_snapshot(destination/"provenance/source_snapshot.zip",frozen)
            for name,expected in frozen["source_sha256"].items():
                if digest(source/name)!=expected:raise RuntimeError("restored source hash differs: "+name)
    return dict(destination=str(destination),source_root=str(source),**verification)


def self_test(source_root,sample_dev=None):
    """No scientific experiment; only dictionary projection and saved development data."""
    root=Path(source_root).resolve();analysis=load_analysis(root)
    fake=dict(spec=dict(phase="development",cluster_index=0,network="manual",n=4,source_mode="none"),config=dict(base_kw=20.),
        truth_evaluation_only=dict(supports=[],r=[],x=[]),candidate_structures=[dict(name="star",n=4,supports=[])],
        planning=dict(history=dict(candidates=dict(star=dict(loss=4.,success=True,usable=True))),
            plans=dict(OBS=dict(action_key="zero",domain="passive",power=[[0.]*4])),
            cost={},families={},tasks={}),evaluation=dict(actions={}),status="failed",failures={})
    condition_sufficiency(fake,compact_record(fake),analysis,{"source_blind_applied":{}},("OBS",))
    count=0
    if sample_dev:
        raw=read(sample_dev)
        if raw.get("spec",{}).get("phase")!="development":raise RuntimeError("self-test accepts saved development data only")
        methods=tuple(raw["config"]["methods"])
        proof=condition_sufficiency(raw,compact_record(raw),analysis,{"source_blind_applied":{}},methods)
        expected=analysis.phase_summary([raw],{"source_blind_applied":{}},methods,bootstrap=3,seed=20409999)
        actual=analysis.phase_summary([compact_record(raw)],{"source_blind_applied":{}},methods,bootstrap=3,seed=20409999)
        difference=first_difference(expected,actual)
        if difference:raise RuntimeError("self-test aggregate mismatch "+difference)
        projected,omissions=physical_projection(raw)
        physical_proof=physical_equivalence(raw,projected,analysis)
        objects={};encoded=encode_shared(projected,objects);rebuilt=decode_shared(encoded,objects)
        if first_difference(projected,rebuilt):raise RuntimeError("saved development physical reconstruction differs")
        physical_test=dict(omitted_blocks=len(omissions),shared_blocks=len(objects),passed=physical_proof["passed"],
            raw_gzip_bytes=Path(sample_dev).stat().st_size,
            projected_gzip_bytes=len(gzip.compress(canonical(projected),compresslevel=9,mtime=0)),
            shared_gzip_bytes=len(gzip.compress(canonical(dict(roots={"manual":encoded},objects=objects)),compresslevel=9,mtime=0)),
            forward_AC_replay_calls=physical_proof["original"]["forward_AC_replay_calls"]+physical_proof["projected"]["forward_AC_replay_calls"])
        count=1
    else:physical_test=dict(not_run=True)
    for name in ("../x","/x","C:/x","x\\y"):
        try:safe_member(name)
        except ValueError:pass
        else:raise RuntimeError("unsafe archive path accepted")
    # A fresh tool-only folder avoids altering science files or recursively
    # deleting a computed Windows path.  Tiny manual fixtures remain here.
    scratch=root/"outputs/reference_tools/tests"/("package_"+uuid.uuid4().hex[:12])
    bundle=scratch/"bundle";bundle.mkdir(parents=True)
    (bundle/"tiny.txt").write_bytes(b"manual byte checksum fixture\n")
    (bundle/"postprocessing").mkdir();(bundle/"provenance").mkdir()
    (bundle/"postprocessing/reference_benchmark_analysis.py").write_bytes(b"# restoration fixture only\n")
    source_fixture={"research_experiments/mobile_ev_mle/reference_benchmark/__init__.py":b"",
        "research_experiments/mobile_ev_mle/reference_benchmark/analysis.py":b"# restoration fixture only\n"}
    with zipfile.ZipFile(bundle/"provenance/source_snapshot.zip","w") as archive:
        for name,data in source_fixture.items():archive.writestr(name,data)
    write(bundle/"provenance/freeze.json",dict(source_sha256={name:sha256(data).hexdigest() for name,data in source_fixture.items()}))
    if not verify_snapshot(bundle/"provenance/source_snapshot.zip",read(bundle/"provenance/freeze.json"))["passed"]:
        raise RuntimeError("manual frozen snapshot verification failed")
    files={p.relative_to(bundle).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))
        for p in bundle.rglob("*") if p.is_file()}
    write(bundle/"manifest.json",dict(files=files,manual_fixture=True))
    archive_path=scratch/"manual.zip"
    with zipfile.ZipFile(archive_path,"w") as archive:
        for p in bundle.rglob("*"):
            if p.is_file():archive.write(p,p.relative_to(bundle).as_posix())
    if not verify_archive(archive_path)["passed"]:raise RuntimeError("manual archive verification failed")
    restored=restore(archive_path,scratch/"restored")
    if (Path(restored["source_root"])/"research_experiments/mobile_ev_mle/reference_benchmark/analysis.py").read_bytes()!=b"# restoration fixture only\n":
        raise RuntimeError("postprocessing restoration differs")
    tampered=scratch/"tampered.zip"
    with zipfile.ZipFile(archive_path) as source,zipfile.ZipFile(tampered,"w") as target:
        for name in source.namelist():target.writestr(name,b"bad" if name=="tiny.txt" else source.read(name))
    try:verify_archive(tampered)
    except RuntimeError:pass
    else:raise RuntimeError("tampered bytes passed verification")
    blocked=scratch/"running";write(blocked/"controller.json",dict(status="running"))
    try:guard_output(blocked)
    except RuntimeError:pass
    else:raise RuntimeError("unfinished controller passed package guard")
    # Exercise the complete replay entry point with three artificial records,
    # not with real calibration/evaluation samples or newly fitted models.
    replay_bundle=scratch/"artificial_replay";config=dict(methods=["OBS"],alpha=.1,source_modes=["none"])
    records={phase:[dict(deepcopy(fake),spec=dict(fake["spec"],phase=phase))] for phase in PHASES}
    thresholds=analysis.calibration_thresholds(records["calibration"],methods=("OBS",),source_modes=("none",))
    summary=dict(methods=["OBS"],analysis_sha256=digest(Path(analysis.__file__)),statistical_bootstrap=1,statistical_seed=20409999,
        phases={phase:analysis.phase_summary(values,thresholds,("OBS",),bootstrap=1,seed=20409999) for phase,values in records.items()},
        spatial_ablation=analysis.small_spatial_ablation(records["evaluation"],thresholds,bootstrap=1,seed=20409999))
    write(replay_bundle/"provenance/freeze.json",dict(config=config))
    write(replay_bundle/"statistics/thresholds.json",thresholds);write(replay_bundle/"statistics/summary.json.gz",summary)
    for phase,values in records.items():write(replay_bundle/"compact"/phase/"manual.json.gz",compact_record(values[0]))
    if not replay_statistics(replay_bundle,root)["passed"]:raise RuntimeError("artificial full statistical replay failed")
    return dict(passed=True,artificial_conditions=1,saved_development_conditions=count,
        archive_byte_check=True,tamper_rejected=True,safe_restore=True,incomplete_run_rejected=True,
        artificial_three_phase_replay=True,test_artifacts=str(scratch),fitting_calls=0,planning_calls=0,
        AC_calls=physical_test.get("forward_AC_replay_calls",0),
        scientific_calibration_files_read=0,evaluation_files_read=0,physical_projection=physical_test)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="mode",required=True)
    p=sub.add_parser("package");p.add_argument("output",type=Path);p.add_argument("destination",type=Path)
    p.add_argument("--source-root",type=Path,default=DEFAULT_ROOT);p.add_argument("--max-mib",type=float,default=20.)
    p.add_argument("--skip-full-replay",action="store_true")
    p=sub.add_parser("verify");p.add_argument("archive",type=Path)
    p=sub.add_parser("restore");p.add_argument("archive",type=Path);p.add_argument("destination",type=Path)
    p=sub.add_parser("replay");p.add_argument("bundle",type=Path);p.add_argument("--source-root",type=Path,required=True)
    p=sub.add_parser("physical-audit");p.add_argument("bundle",type=Path);p.add_argument("--source-root",type=Path,required=True)
    p=sub.add_parser("self-test");p.add_argument("--source-root",type=Path,default=DEFAULT_ROOT);p.add_argument("--sample-dev",type=Path)
    argv=sys.argv[1:]
    if argv and argv[0].startswith("--"):
        quick=argparse.ArgumentParser(description=__doc__)
        quick.add_argument("--output",type=Path,required=True);quick.add_argument("--destination",type=Path,required=True)
        quick.add_argument("--source-root",type=Path,default=DEFAULT_ROOT);quick.add_argument("--max-mib",type=float,default=20.)
        q=quick.parse_args(argv)
        value=package(q.output,q.destination,source_root=q.source_root,max_mib=q.max_mib)
        print(json.dumps(clean(value),ensure_ascii=False,sort_keys=True));return
    args=parser.parse_args(argv)
    if args.mode=="package":value=package(args.output,args.destination,source_root=args.source_root,max_mib=args.max_mib,verify_statistics=not args.skip_full_replay)
    elif args.mode=="verify":value=verify_archive(args.archive)
    elif args.mode=="restore":value=restore(args.archive,args.destination)
    elif args.mode=="replay":value=replay_statistics(args.bundle,args.source_root)
    elif args.mode=="physical-audit":value=physical_audit(args.bundle,args.source_root)
    else:
        value=self_test(args.source_root,args.sample_dev)
        value["tool_sha256"]=digest(Path(__file__))
        write(Path(__file__).with_name("package_selftest_receipt.json"),value)
    print(json.dumps(clean(value),ensure_ascii=False,sort_keys=True))


if __name__=="__main__":main()
