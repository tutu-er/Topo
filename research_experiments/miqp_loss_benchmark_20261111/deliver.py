"""Archive complete MIQP evidence and paired L1 references, without raw arrays."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")


def recovery_ledger(output,protocol):
    """Read only the new MIQP host-restart debit, never the parent L1 ledger."""
    file=output/"recovery/host_restart_20261111/charges.json"
    if not file.exists():
        return dict(charges={},interrupted_attempts=0,total_interrupted_wall_upper_seconds=0.,
                    file=None,sha256=None,charge_scope="no new MIQP host-restart debit")
    ledger=read(file)
    assert ledger["protocol_sha256"]==sha(output/"protocol.json")
    assert ledger["catalog_sha256"]==sha(output/"catalog.json")
    expected={(job,method["id"]) for job in protocol["jobs"] for method in protocol["methods"]}
    charges=ledger["charges"]
    assert len(charges)==ledger["interrupted_attempts"]
    assert ledger["logical_models_planned"]==protocol["planned"]==96
    assert ledger["attempts_after_recovery_expected"]==protocol["planned"]+len(charges)
    for key,row in charges.items():
        assert key==row["job"]+"/"+row["method"] and (row["job"],row["method"]) in expected
        debit=float(row["interrupted_attempt_seconds"])
        assert math.isfinite(debit) and 0.<debit<protocol["limits"]["total"]
        assert row["charge_is_conservative_upper_bound"] is True and row["solver_failure"] is False
        assert abs(row["original_total_cap"]-protocol["limits"]["total"])<1e-7
        assert abs(row["remaining_total_cap"]+debit-row["original_total_cap"])<1e-7
        for name,digest in row["preserved_files"].items():
            assert sha(file.parent/"jobs"/row["job"]/row["method"]/name)==digest,name
    preserved=ledger["result_hashes_before_recovery"]
    assert len(preserved)==ledger["returned_preserved"]
    for name,digest in preserved.items():assert sha(output/name)==digest,name
    total=sum(float(row["interrupted_attempt_seconds"]) for row in charges.values())
    assert abs(total-ledger["total_interrupted_wall_upper_seconds"])<1e-6
    return dict(charges=charges,interrupted_attempts=len(charges),file=str(file),sha256=sha(file),
                total_interrupted_wall_upper_seconds=total,verified_system_boot_utc=ledger["verified_system_boot_utc"],
                preserved_successful_results=len(preserved),
                charge_scope="conservative upper bound: pre-spawn payload timestamp through verified subsequent host boot; downtime excluded")


def timing_row(job,method,result,common,parent_result,recovery,original_cap):
    """Charge the shared frontend once and the interrupted native attempt once."""
    assert abs(common["seconds"]-common["parent_common_seconds"]-common["qp_common_seconds"])<1e-7
    successful=float(result["total_seconds"])
    assert abs(successful-result["fresh_seconds"]-common["seconds"])<1e-7
    assert abs(float(result.get("successful_total_seconds",successful))-successful)<1e-7
    charge=recovery["charges"].get(job+"/"+method)
    debit=float(charge["interrupted_attempt_seconds"]) if charge else 0.
    assert abs(float(result.get("interrupted_attempt_seconds",0.))-debit)<1e-6
    assert int(result.get("interrupted_attempt_count",int(debit>0.)))==int(debit>0.)
    operational=successful+debit
    context=result.get("recovery",result.get("recovery_context",{}))
    recorded_charge_hash=result.get("recovery_charges_sha256") or context.get("recovery_charges_sha256")
    if context:
        assert recorded_charge_hash==recovery["sha256"]
        assert context["common_charged_once"] is True and context["optimizer_source_modified"] is False
    if charge:
        assert recorded_charge_hash==recovery["sha256"]
        assert "logical_total_seconds" in result and "operational_total_seconds" in result
        assert abs(float(result.get("original_total_cap",original_cap))-original_cap)<1e-7
    for key in ("logical_total_seconds","operational_total_seconds"):
        if key in result:assert abs(float(result[key])-operational)<1e-6,key
    remaining=original_cap-debit
    assert abs(float(result["limits"]["total"])-remaining)<1e-6
    parent_successful=float(parent_result["total_seconds"])
    parent_debit=float(parent_result.get("interrupted_attempt_seconds",0.))
    parent_operational=float(parent_result.get("logical_total_seconds",parent_successful))
    assert abs(parent_operational-parent_successful-parent_debit)<1e-6
    return dict(job=job,method=method,native_fresh_seconds=result["fresh_seconds"],
                charged_parent_common_seconds=common["parent_common_seconds"],
                charged_qp_common_seconds=common["qp_common_seconds"],
                total_successful_seconds=successful,successful_total_seconds=successful,
                new_MIQP_interrupted_attempts=int(charge is not None),
                new_MIQP_interrupted_wall_upper_seconds=debit,
                new_MIQP_interruption_charge_is_upper_bound=bool(charge),
                recovery_wrapper_used=bool(context),
                operational_total_seconds=operational,operational_total_upper_seconds=operational,
                operational_timing_scope="successful method + conservative interrupted-attempt upper debit; shared common charged once",
                original_total_cap_seconds=original_cap,remaining_successful_attempt_cap_seconds=remaining,
                successful_attempt_cap_exceeded=successful>remaining+1.,
                operational_upper_cap_exceeded=operational>original_cap+1.,
                new_MIQP_recovery_charges_sha256=recorded_charge_hash,
                parent_L1_successful_seconds=parent_successful,parent_L1_interrupted_proxy_seconds=parent_debit,
                parent_L1_operational_proxy_seconds=parent_operational,
                parent_L1_interrupted_attempts=int(parent_debit>0.))


def deliver(output):
    output=Path(output).resolve();protocol=read(output/"protocol.json")
    parent=Path(protocol["parent_output"]).resolve();analysis=output/"analysis"
    verification=read(analysis/"verification.json");parent_verification=read(parent/"analysis/verification.json")
    assert verification["passed"] and verification["complete"]
    assert verification["new_native_verified"]==verification["reused_parent_verified"]==96
    assert verification["new_native_failed"]==verification["new_native_invalid"]==verification["new_native_pending"]==0
    assert verification["protocol_sha256"]==sha(output/"protocol.json")
    assert verification["parent_analysis_verification_sha256"]==sha(parent/"analysis/verification.json")
    assert parent_verification["passed"] and parent_verification["complete"] and parent_verification["verified"]==144
    for name,digest in protocol["source_hashes"].items():assert sha(output/"source"/name)==digest,name
    for name,digest in protocol["source_artifact_hashes"].items():assert sha(output/"source"/name)==digest,name
    docs=ROOT/"docs"/output.name;docs.mkdir(parents=True,exist_ok=True)
    figures=read(docs/"figure_receipt.json");visual=read(docs/"visual_review.json")
    assert visual["passed"] and figures["verification_sha256"]==sha(analysis/"verification.json")
    for name,digest in figures["input_csv_hashes"].items():assert sha(analysis/name)==digest,name
    for name,digest in figures["figure_sha256"].items():assert sha(docs/name)==digest,name
    for file in analysis.iterdir():
        if file.is_file():shutil.copy2(file,docs/file.name)
    parent_docs=ROOT/"docs"/parent.name
    parent_archive=parent_docs/"records.zip";parent_archive_receipt=read(parent_docs/"archive_receipt.json")
    assert sha(parent_archive)==parent_archive_receipt["sha256"]
    methods=[m["id"] for m in protocol["methods"]]
    recovery=recovery_ledger(output,protocol)
    if recovery["interrupted_attempts"]:
        for name in ("capture_recovery.py","recovery_run.py","recovery_worker.py"):
            assert Path(__file__).with_name(name).is_file(),name
    timing_rows=[];paired_hashes={}
    for job in protocol["jobs"]:
        common=read(output/"common"/job/"common.json")
        for method in methods:
            file=output/"jobs"/job/method/"result.json";result=read(file)
            assert result["protocol_sha256"]==sha(output/"protocol.json")
            parent_file=parent/"jobs"/job/method/"result.json"
            timing_rows.append(timing_row(job,method,result,common,read(parent_file),recovery,protocol["limits"]["total"]))
            paired_hashes[job+"/"+method]=sha(parent_file)
    assert sum(row["new_MIQP_interrupted_attempts"] for row in timing_rows)==recovery["interrupted_attempts"]
    assert abs(sum(row["new_MIQP_interrupted_wall_upper_seconds"] for row in timing_rows)-recovery["total_interrupted_wall_upper_seconds"])<1e-6
    with (docs/"timing_components.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,timing_rows[0]);writer.writeheader();writer.writerows(timing_rows)
    unique_common=[read(output/"common"/job/"common.json") for job in protocol["jobs"]]
    timing=dict(new_native_evaluations=96,nominal_method_hours=384.,per_evaluation_cap_seconds=14400.,
                new_successful_method_seconds=sum(r["total_successful_seconds"] for r in timing_rows),
                new_operational_total_seconds=sum(r["operational_total_seconds"] for r in timing_rows),
                new_operational_total_upper_seconds=sum(r["operational_total_upper_seconds"] for r in timing_rows),
                new_operational_total_is_conservative_upper_bound=bool(recovery["interrupted_attempts"]),
                new_interrupted_attempts=recovery["interrupted_attempts"],
                new_interrupted_wall_upper_seconds=recovery["total_interrupted_wall_upper_seconds"],
                new_total_native_attempts_including_interrupted=96+recovery["interrupted_attempts"],
                new_interruption_timing_scope=recovery["charge_scope"],
                new_recovery_charges_sha256=recovery["sha256"],
                new_operational_upper_cap_exceeded=sum(r["operational_upper_cap_exceeded"] for r in timing_rows),
                new_successful_attempt_cap_exceeded=sum(r["successful_attempt_cap_exceeded"] for r in timing_rows),
                new_fresh_native_seconds=sum(r["native_fresh_seconds"] for r in timing_rows),
                charged_parent_common_seconds=sum(r["charged_parent_common_seconds"] for r in timing_rows),
                charged_qp_common_seconds=sum(r["charged_qp_common_seconds"] for r in timing_rows),
                unique_parent_common_seconds=sum(c["parent_common_seconds"] for c in unique_common),
                unique_qp_common_seconds=sum(c["qp_common_seconds"] for c in unique_common),
                common_charge_rule="the shared frontend and QP preprocessing are each charged once per method/input",
                interrupted_charge_rule="new MIQP worker upper debit added exactly once to successful total; not added to native_fresh or common components again",
                parent_paired_L1_successful_method_seconds=sum(r["parent_L1_successful_seconds"] for r in timing_rows),
                parent_paired_L1_interrupted_proxy_seconds=sum(r["parent_L1_interrupted_proxy_seconds"] for r in timing_rows),
                parent_paired_L1_operational_proxy_seconds=sum(r["parent_L1_operational_proxy_seconds"] for r in timing_rows),
                parent_paired_L1_interrupted_attempts=sum(r["parent_L1_interrupted_attempts"] for r in timing_rows),
                parent_L1_timing_scope="paired 96 old L1 runs only: original estimated interruption proxy, distinct from new MIQP boot-time upper debits; never added to MSE totals",
                recovery_evidence={key:value for key,value in recovery.items() if key!="charges"})
    timing["methods"]=[dict(method=method,evaluations=sum(r["method"]==method for r in timing_rows),
        successful_method_seconds=sum(r["successful_total_seconds"] for r in timing_rows if r["method"]==method),
        interrupted_attempts=sum(r["new_MIQP_interrupted_attempts"] for r in timing_rows if r["method"]==method),
        interrupted_wall_upper_seconds=sum(r["new_MIQP_interrupted_wall_upper_seconds"] for r in timing_rows if r["method"]==method),
        operational_total_upper_seconds=sum(r["operational_total_upper_seconds"] for r in timing_rows if r["method"]==method))
        for method in methods]
    write(docs/"timing_components.json",timing)
    parent_reference=dict(output=str(parent),protocol_sha256=sha(parent/"protocol.json"),
                          native_analysis_receipt_sha256=sha(parent/"analysis/verification.json"),
                          original_archive_path=str(parent_archive),original_archive_sha256=sha(parent_archive),
                          original_archive_bytes=parent_archive.stat().st_size,parent_zip_copied=False,
                          paired_method_runs=96,paired_result_sha256=paired_hashes)
    write(docs/"parent_reference.json",parent_reference)
    write(docs/"environment.json",dict(python=sys.version,recorded_utc=datetime.now(timezone.utc).isoformat(),
                                     source="per-native-result environment and runtime overlay receipts retained in archive"))
    files={}
    excluded={".npz",".pyc",".pem",".key",".lic"}
    for top in ("source","analysis_source","analysis","common","jobs","recovery"):
        for file in (output/top).rglob("*"):
            if file.is_file() and file.suffix.lower() not in excluded and not file.name.endswith(".tmp"):
                files["run/"+file.relative_to(output).as_posix()]=file
    for file in output.iterdir():
        if file.is_file() and file.suffix in {".json",".log"}:
            files["run/"+file.name]=file
    for job in protocol["jobs"]:
        for method in methods:
            for name in ("result.json","candidates.json","recovery.json"):
                file=parent/"jobs"/job/method/name
                if file.exists():files["parent_paired/"+file.relative_to(parent).as_posix()]=file
        for file in (parent/"analysis/jobs"/job).glob("*.json"):
            files["parent_paired/"+file.relative_to(parent).as_posix()]=file
    for name in ("protocol.json","catalog.json","analysis/verification.json","analysis/summary.csv","analysis_source/analyze.py"):
        file=parent/name;files["parent_reference/"+name]=file
    for file in Path(__file__).parent.iterdir():
        if file.is_file() and file.suffix.lower() not in excluded:files["actual_driver/"+file.name]=file
    for file in docs.iterdir():
        if file.is_file() and file.name not in {"records.zip","manifest.json","archive_receipt.json"}:
            files["report/"+file.name]=file
    hashes={name:sha(file) for name,file in sorted(files.items())};archive=docs/"records.zip"
    if recovery["interrupted_attempts"]:
        assert "run/recovery/host_restart_20261111/charges.json" in files
        for name in ("capture_recovery.py","recovery_run.py","recovery_worker.py"):
            assert "actual_driver/"+name in files,name
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as bundle:
        for name,file in sorted(files.items()):bundle.write(file,name)
        bundle.writestr("manifest.json",json.dumps(hashes,indent=2)+"\n")
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert all(hashlib.sha256(bundle.read(name)).hexdigest()==digest for name,digest in hashes.items())
        assert not any(Path(name).suffix.lower() in excluded for name in bundle.namelist())
        assert not any(name.endswith("records.zip") for name in bundle.namelist())
    with (analysis/"solver.csv").open(encoding="utf-8",newline="") as stream:solver=list(csv.DictReader(stream))
    receipt=dict(bytes=archive.stat().st_size,sha256=sha(archive),files=len(hashes),zip_crc_passed=True,
                 all_manifest_hashes_passed=True,new_native_verified=96,reused_parent_verified=96,
                 new_solver_timeouts=sum(r["fit_loss"]=="scaled_mse" and r["status"]=="9" for r in solver),
                 failed=verification["new_native_failed"],invalid=verification["new_native_invalid"],pending=verification["new_native_pending"],
                 budget_exceeded=verification["budget_exceeded"],successful_attempt_budget_exceeded=verification["successful_attempt_budget_exceeded"],
                 new_successful_method_seconds=timing["new_successful_method_seconds"],
                 new_operational_total_upper_seconds=timing["new_operational_total_upper_seconds"],
                 new_interrupted_attempts=recovery["interrupted_attempts"],
                 new_interrupted_wall_upper_seconds=recovery["total_interrupted_wall_upper_seconds"],
                 new_recovery_charges_sha256=recovery["sha256"],
                 recovery_subtree_archived=any(name.startswith("run/recovery/") for name in files),
                 new_interruption_cost_is_conservative_upper_bound=bool(recovery["interrupted_attempts"]),
                 parent_zip_copied=False,raw_measurement_arrays_bundled=False)
    write(docs/"archive_receipt.json",receipt)
    write(docs/"manifest.json",{p.name:sha(p) for p in docs.iterdir() if p.is_file() and p.name!="manifest.json"})
    print(json.dumps(receipt));return receipt


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",nargs="?",type=Path,default=ROOT/"outputs/miqp_loss_benchmark_20261111")
    deliver(parser.parse_args().output)
