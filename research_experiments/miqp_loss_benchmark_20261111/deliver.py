"""Archive complete MIQP evidence and paired L1 references, without raw arrays."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
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
    timing_rows=[];paired_hashes={}
    for job in protocol["jobs"]:
        common=read(output/"common"/job/"common.json")
        assert abs(common["seconds"]-common["parent_common_seconds"]-common["qp_common_seconds"])<1e-7
        for method in methods:
            file=output/"jobs"/job/method/"result.json";result=read(file)
            assert result["protocol_sha256"]==sha(output/"protocol.json")
            assert abs(result["total_seconds"]-result["fresh_seconds"]-common["seconds"])<1e-7
            timing_rows.append(dict(job=job,method=method,native_fresh_seconds=result["fresh_seconds"],
                                    charged_parent_common_seconds=common["parent_common_seconds"],
                                    charged_qp_common_seconds=common["qp_common_seconds"],total_successful_seconds=result["total_seconds"]))
            parent_file=parent/"jobs"/job/method/"result.json"
            paired_hashes[job+"/"+method]=sha(parent_file)
    with (docs/"timing_components.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,timing_rows[0]);writer.writeheader();writer.writerows(timing_rows)
    unique_common=[read(output/"common"/job/"common.json") for job in protocol["jobs"]]
    timing=dict(new_native_evaluations=96,nominal_method_hours=384.,per_evaluation_cap_seconds=14400.,
                new_successful_method_seconds=sum(r["total_successful_seconds"] for r in timing_rows),
                new_fresh_native_seconds=sum(r["native_fresh_seconds"] for r in timing_rows),
                charged_parent_common_seconds=sum(r["charged_parent_common_seconds"] for r in timing_rows),
                charged_qp_common_seconds=sum(r["charged_qp_common_seconds"] for r in timing_rows),
                unique_parent_common_seconds=sum(c["parent_common_seconds"] for c in unique_common),
                unique_qp_common_seconds=sum(c["qp_common_seconds"] for c in unique_common),
                common_charge_rule="the shared frontend and QP preprocessing are each charged once per method/input",
                parent_L1_timing_scope="successful and interrupted operational timings retained separately in paired records/analysis")
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
    for top in ("source","analysis_source","analysis","common","jobs"):
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
                 parent_zip_copied=False,raw_measurement_arrays_bundled=False)
    write(docs/"archive_receipt.json",receipt)
    write(docs/"manifest.json",{p.name:sha(p) for p in docs.iterdir() if p.is_file() and p.name!="manifest.json"})
    print(json.dumps(receipt));return receipt


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",nargs="?",type=Path,default=ROOT/"outputs/miqp_loss_benchmark_20261111")
    deliver(parser.parse_args().output)
