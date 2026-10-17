"""Package completed full-pipeline evidence without copying every large raw file.

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
from pathlib import Path, PurePosixPath
import shutil
import sys
import uuid
import zipfile


DEFAULT_ROOT=Path(__file__).resolve().parents[2]
PHASES=("development","calibration","evaluation")
OMITTED=("historical and future meter arrays outside the four full fixtures",
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

    The shape deliberately remains compatible with full_pipeline.analysis.
    The result supports statistics, including R/X error, but is not an AC
    physical-replay record; four original fixtures and the local originals
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
    result["compact_evidence"]=dict(schema="full_pipeline_statistical_inputs_v1",
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
    expected=root/"research_experiments/mobile_ev_mle/full_pipeline/analysis.py"
    if not expected.exists():raise FileNotFoundError(expected)
    sys.path[:0]=[str(root),str(root/"rnj_wzzt_core")]
    module=importlib.import_module("research_experiments.mobile_ev_mle.full_pipeline.analysis")
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
    counts=controller.get("counts",{})
    if counts.get("remaining",0)!=0 or counts.get("computed",0)!=counts.get("total",0):
        raise RuntimeError("controller condition receipt is incomplete")
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
        values.sort(key=lambda r:(r["spec"]["cluster_index"],r["spec"]["n"],r["spec"]["source_mode"]))
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
    receipt=dict(schema="full_pipeline_statistical_replay_v1",passed=True,phases=checks,
        spatial_ablation_passed=True,calibration_source_blind_passed=True,bootstrap=B,seed=seed,
        raw_AC_replay_performed=False,fitting_calls=0,planning_calls=0,
        scope="all main statistical objects exactly regenerated from retained loss/support/plan/forecast data; original full AC audit is a separate receipt")
    if write_receipt:write(bundle/"statistics/replay_receipt.json",receipt)
    return receipt


def instructions():
    return """# Evidence restore and statistical replay

This archive preserves the frozen source snapshot, postprocessing, exact used
input windows and statistics. All large original raw records remain in the
execution directory. raw_sha256.json lists every original byte SHA and size.
Compact files support statistical replay and do not replace those originals.
Four preregistered development cluster-0 files (n6/n8 x none/DER) are retained
without alteration for independent physical AC audit. Other full raw records
were audited locally; audit.json records that completed numerical audit.

Commands (Python environment versions are in provenance/environment.json):

    python package_evidence.py verify EVIDENCE.zip
    python package_evidence.py restore EVIDENCE.zip NEW_EMPTY_DIRECTORY
    python NEW_EMPTY_DIRECTORY/postprocessing/package_evidence.py replay NEW_EMPTY_DIRECTORY --source-root NEW_EMPTY_DIRECTORY/source

Restore verifies all byte hashes, extracts source_snapshot.zip into source/,
and installs the saved postprocessing analysis.py at its original module path.
Replay regenerates calibration thresholds from calibration compact files,
then all phase tables, date-cluster bootstrap intervals, paired tests, R/X,
forecast error, cost and score-ranking summaries. It does not refit, plan,
simulate or call AC. Future truth inside evidence is labelled evaluation only.

For physical audit, load the four raw_fixtures/development files and call the
saved full_pipeline.analysis.audit_record with a g_precision.analysis.Audit.
This performs forward AC replay only. Dense optimization states and observed
meters omitted from other compact files cannot be reconstructed from compact
statistics. Obtain their original files from the local execution directory
and verify raw_sha256.json before any additional physical replay.

The original 60MB OPSD CSV is not duplicated. inputs/prepared.json.gz contains
the exact selected raw windows; official README/datapackage and data inventory
retain attribution, source/version and the original CSV checksum.
"""


def package(output,destination,*,source_root=DEFAULT_ROOT,max_mib=20.,verify_statistics=True):
    output=Path(output).resolve();destination=Path(destination).resolve();source_root=Path(source_root).resolve()
    controller,frozen,summary=guard_output(output)
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
        expected=config[phase+"_clusters"]*len(config["terminal_counts"])*len(config["source_modes"])
        if len(current)!=expected:raise RuntimeError("raw condition count differs: "+phase)
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
    copy_file(prepared.parent/"data_inventory.json","inputs/data_inventory.json")
    metadata=source_root/"data_external/opsd_household_20200415"
    for name in ("README.md","datapackage.json"):copy_file(metadata/name,"inputs/official_"+name)
    copy_file(Path(analysis.__file__),"postprocessing/full_pipeline_analysis.py")
    copy_file(Path(__file__),"postprocessing/package_evidence.py")
    receipt=pick(controller,("schema","status","counts","freeze_identity","controller_policy_sha256",
        "requested_phases","started_utc","updated_utc","finished_utc","workers","per_worker_threads","exceptions"))
    receipt.update(original_controller_sha256=digest(output/"controller.json"),
        executor_paths_and_lock_tokens_omitted=True)
    write(bundle/"provenance/controller_statistical_complete.json",receipt)
    raw_registry={};sufficiency=[]
    for path in paths:
        relative=path.relative_to(output).as_posix();raw=read(path);reduced=compact_record(raw)
        if raw.get("freeze_identity")!=frozen["identity"] or raw.get("source_unchanged") is not True:
            raise RuntimeError("original raw version differs from completed freeze: "+relative)
        proof=condition_sufficiency(raw,reduced,analysis,thresholds,methods)
        reduced["compact_evidence"].update(original_raw_sha256=digest(path),original_raw_bytes=path.stat().st_size,
            sufficient_statistics_verification=proof)
        write(bundle/"compact"/relative,reduced)
        raw_registry[relative]=dict(sha256=digest(path),bytes=path.stat().st_size,status=raw.get("status"),
            compact_path="compact/"+relative,spec=raw["spec"])
        sufficiency.append(dict(path=relative,**pick(proof,("decision_sha256","condition_detail_sha256","passed"))))
    write(bundle/"raw_sha256.json",dict(schema="full_pipeline_original_raw_manifest_v1",files=raw_registry,
        n_files=len(paths),total_original_bytes=sum(v["bytes"] for v in raw_registry.values()),
        scope="all original raw files remain local; compacts are additional statistical evidence"))
    for n in config["terminal_counts"]:
        for source in config["source_modes"]:
            filename="history_000_n%d_%s.json.gz"%(n,source)
            copy_file(output/"development"/filename,"raw_fixtures/development/"+filename)
    write(bundle/"statistics/compact_sufficiency_receipt.json",dict(passed=True,n_conditions=len(sufficiency),
        conditions=sufficiency,scope="decision and complete condition-detail objects agree before aggregation",
        fitting_calls=0,planning_calls=0,AC_calls=0))
    (bundle/"RESTORE.md").write_text(instructions(),encoding="utf-8")
    if verify_statistics:replay=replay_statistics(bundle,source_root)
    else:
        replay=dict(passed=False,not_run=True,reason="explicit skip; aggregate replay remains outstanding")
        write(bundle/"statistics/replay_receipt.json",replay)
    files={}
    for path in sorted(bundle.rglob("*")):
        if path.is_file():
            relative=path.relative_to(bundle).as_posix()
            files[relative]=dict(bytes=path.stat().st_size,sha256=digest(path),**sources.get(relative,{}))
    manifest=dict(schema="full_pipeline_evidence_package_v1",policy_identity=frozen["identity"],files=files,
        created_utc=datetime.now(timezone.utc).isoformat(),original_run_name=output.name,
        original_raw_files=len(raw_registry),original_raw_bytes=sum(v["bytes"] for v in raw_registry.values()),
        complete_raw_fixtures="predeclared development cluster 0, every n/source condition; selected before outcomes",
        discarded_from_compact=list(OMITTED),compact_replaces_original=False,
        statistical_replay_passed=replay.get("passed",False),all_byte_checksums_required=True,
        original_CSV_included=False,official_CSV_sha256=config.get("raw_csv_sha256"),
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
    verification.update(statistical_replay_passed=replay.get("passed",False),manifest_sha256=digest(bundle/"manifest.json"),
        target_max_archive_bytes=int(max_mib*1024*1024),within_size_target=archive_path.stat().st_size<=max_mib*1024*1024)
    write(destination/"package_receipt.json",verification)
    if not verification["within_size_target"]:
        raise RuntimeError("verified package exceeds size target; required evidence retained at "+str(archive_path))
    return dict(archive=str(archive_path),**verification)


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
    analysis=source/"research_experiments/mobile_ev_mle/full_pipeline/analysis.py"
    if analysis.exists():raise RuntimeError("source snapshot unexpectedly contains a competing analysis.py")
    analysis.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(destination/"postprocessing/full_pipeline_analysis.py",analysis)
    return dict(destination=str(destination),source_root=str(source),**verification)


def self_test(source_root,sample_dev=None):
    """No scientific experiment; only dictionary projection and saved development data."""
    root=Path(source_root).resolve();analysis=load_analysis(root)
    fake=dict(spec=dict(phase="development",cluster_index=0,n=4,source_mode="none"),config=dict(base_kw=20.),
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
        count=1
    for name in ("../x","/x","C:/x","x\\y"):
        try:safe_member(name)
        except ValueError:pass
        else:raise RuntimeError("unsafe archive path accepted")
    # A fresh tool-only folder avoids altering science files or recursively
    # deleting a computed Windows path.  Tiny manual fixtures remain here.
    scratch=root/"outputs/full_pipeline_tools/tests"/("package_"+uuid.uuid4().hex[:12])
    bundle=scratch/"bundle";bundle.mkdir(parents=True)
    (bundle/"tiny.txt").write_bytes(b"manual byte checksum fixture\n")
    (bundle/"postprocessing").mkdir();(bundle/"provenance").mkdir()
    (bundle/"postprocessing/full_pipeline_analysis.py").write_bytes(b"# restoration fixture only\n")
    with zipfile.ZipFile(bundle/"provenance/source_snapshot.zip","w") as archive:
        archive.writestr("research_experiments/mobile_ev_mle/full_pipeline/__init__.py",b"")
    files={p.relative_to(bundle).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))
        for p in bundle.rglob("*") if p.is_file()}
    write(bundle/"manifest.json",dict(files=files,manual_fixture=True))
    archive_path=scratch/"manual.zip"
    with zipfile.ZipFile(archive_path,"w") as archive:
        for p in bundle.rglob("*"):
            if p.is_file():archive.write(p,p.relative_to(bundle).as_posix())
    if not verify_archive(archive_path)["passed"]:raise RuntimeError("manual archive verification failed")
    restored=restore(archive_path,scratch/"restored")
    if (Path(restored["source_root"])/"research_experiments/mobile_ev_mle/full_pipeline/analysis.py").read_bytes()!=b"# restoration fixture only\n":
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
        artificial_three_phase_replay=True,test_artifacts=str(scratch),fitting_calls=0,planning_calls=0,AC_calls=0,
        scientific_calibration_files_read=0,evaluation_files_read=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="mode",required=True)
    p=sub.add_parser("package");p.add_argument("output",type=Path);p.add_argument("destination",type=Path)
    p.add_argument("--source-root",type=Path,default=DEFAULT_ROOT);p.add_argument("--max-mib",type=float,default=20.)
    p.add_argument("--skip-full-replay",action="store_true")
    p=sub.add_parser("verify");p.add_argument("archive",type=Path)
    p=sub.add_parser("restore");p.add_argument("archive",type=Path);p.add_argument("destination",type=Path)
    p=sub.add_parser("replay");p.add_argument("bundle",type=Path);p.add_argument("--source-root",type=Path,required=True)
    p=sub.add_parser("self-test");p.add_argument("--source-root",type=Path,default=DEFAULT_ROOT);p.add_argument("--sample-dev",type=Path)
    args=parser.parse_args()
    if args.mode=="package":value=package(args.output,args.destination,source_root=args.source_root,max_mib=args.max_mib,verify_statistics=not args.skip_full_replay)
    elif args.mode=="verify":value=verify_archive(args.archive)
    elif args.mode=="restore":value=restore(args.archive,args.destination)
    elif args.mode=="replay":value=replay_statistics(args.bundle,args.source_root)
    else:value=self_test(args.source_root,args.sample_dev)
    print(json.dumps(clean(value),ensure_ascii=False,sort_keys=True))


if __name__=="__main__":main()
