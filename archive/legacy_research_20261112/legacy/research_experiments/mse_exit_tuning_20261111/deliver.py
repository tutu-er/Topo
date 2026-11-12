"""Archive verified gate replays, calibration decisions and source references."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n",encoding="utf-8",newline="\n")
def deliver(output):
    output=Path(output).resolve();p=read(output/"protocol.json");v=read(output/"analysis/verification.json")
    assert v["passed"] and v["complete"] and v["inputs"]==48 and v["variant_rows"]==288
    assert v["new_native_MIQP_calls"]==0 and sha(output/"selection.json")==v["selection_file_sha256"]
    parent=Path(p["parent_output"]);pd=ROOT/"docs"/parent.name
    assert sha(pd/"records.zip")==p["parent_archive_sha256"]
    for n,h in p["controller_hashes"].items():assert sha(output/"controller_source"/n)==h
    for j,h in p["parent_result_sha256"].items():assert sha(parent/"jobs"/j/"T003/result.json")==h
    d=ROOT/"docs"/output.name
    f=read(d/"figure_receipt.json");visual=read(d/"visual_review.json")
    assert visual["passed"] and not f["visual_review_pending"]
    assert f["verification_sha256"]==sha(output/"analysis/verification.json")
    for n,h in f["files"].items():assert sha(d/n)==h
    for x in (output/"analysis").iterdir():
        if x.is_file():shutil.copy2(x,d/x.name)
    for n in ("selection.json","protocol.json","completion.json"):shutil.copy2(output/n,d/n)
    write(d/"parent_reference.json",dict(parent_output=str(parent),parent_archive_sha256=p["parent_archive_sha256"],
        parent_archive_path=str(pd/"records.zip"),parent_zip_copied=False,
        reused_MIQP_results=48,parent_results_sha256=p["parent_result_sha256"],
        native_source_manifest=p["source_hashes"]))
    files={}
    for x in output.rglob("*"):
        if x.is_file() and x.suffix.lower() not in {".npz",".pyc",".lic",".pem",".key"} and not x.name.endswith(".tmp"):
            files["run/"+x.relative_to(output).as_posix()]=x
    for x in Path(__file__).parent.iterdir():
        if x.is_file():files["actual_driver/"+x.name]=x
    for x in d.iterdir():
        if x.is_file() and x.name not in {"records.zip","manifest.json","archive_receipt.json"}:
            files["report/"+x.name]=x
    hashes={n:sha(x) for n,x in sorted(files.items())};z=d/"records.zip"
    with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as bundle:
        for n,x in sorted(files.items()):bundle.write(x,n)
        bundle.writestr("manifest.json",json.dumps(hashes,indent=2)+"\n")
    with zipfile.ZipFile(z) as bundle:
        assert bundle.testzip() is None
        assert all(hashlib.sha256(bundle.read(n)).hexdigest()==h for n,h in hashes.items())
    receipt=dict(passed=True,files=len(files),bytes=z.stat().st_size,sha256=sha(z),
        manifest_hashes_passed=True,zip_crc_passed=True,inputs=48,views=288,new_MIQP_calls=0,
        new_native_QP_fits=v["replay_new_native_QP_fits"],
        chosen_tau=v["full48_selected_tau"],selection_sha256=sha(output/"selection.json"),
        prospective_validation=False,parent_archive_sha256=p["parent_archive_sha256"],
        created_utc=datetime.now(timezone.utc).isoformat())
    write(d/"archive_receipt.json",receipt)
    write(d/"manifest.json",{x.name:sha(x) for x in d.iterdir() if x.is_file() and x.name!="manifest.json"})
    print(json.dumps(receipt))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("output",type=Path);deliver(p.parse_args().output)
