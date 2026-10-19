"""Archive a completed ranking run and all imported local scientific sources."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
import scipy

from research_experiments.mobile_ev_mle.case33_quality_fixed_rx import experiment as base

PREFIX="research_experiments/mobile_ev_mle/case33_candidate_ranking/"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--evidence",type=Path,required=True)
    args=parser.parse_args(); evidence=args.evidence
    evidence.mkdir(parents=True,exist_ok=True)
    for name in ("experiment", "audit_saved"):
        spec=importlib.util.spec_from_file_location("snapshot_"+name,Path(__file__).with_name(name+".py"))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=Path(base.lk.__file__).resolve().parents[2]
    files={root/"terminal_case33/data/case33bw_raw.py"}
    for module in list(sys.modules.values()):
        value=getattr(module,"__file__",None)
        if value:
            p=Path(value).resolve()
            if p.suffix==".py" and root in p.parents and "site-packages" not in str(p):
                files.add(p)
    own={p for p in Path(__file__).resolve().parent.iterdir()
         if p.is_file() and p.suffix in (".py",".md",".ini")}
    files.update(own)
    records=[]
    archive_paths=set()
    with zipfile.ZipFile(evidence/"source_snapshot.zip","w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files,key=str):
            relative=PREFIX+p.name if p in own else p.relative_to(root).as_posix()
            if relative in archive_paths:
                raise ValueError("duplicate source archive member: "+relative)
            archive_paths.add(relative)
            blob=p.read_bytes();z.writestr(relative,blob)
            records.append(dict(path=relative,sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob)))
    base.save(evidence/"source_manifest.json",dict(files=records,python=sys.version,executable=sys.executable,
        numpy=np.__version__,scipy=scipy.__version__,source_snapshot_sha256=sha(evidence/"source_snapshot.zip")))
    config=json.loads((args.results/"configuration.json").read_text(encoding="utf-8"))
    shutil.copyfile(config["input"],evidence/"input_archive.zip")
    assert sha(evidence/"input_archive.zip")==config["input_sha256"]
    with zipfile.ZipFile(evidence/"full_results.zip","w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(args.results.rglob("*.json")):
            if p.name!="progress.json":
                z.write(p,p.relative_to(args.results).as_posix())
    for name in ("summary.json","configuration.json","execution.json","independent_audit.json",
                 "canonical_replay_receipt.json","validation.json"):
        p=args.results/name
        if p.exists():shutil.copyfile(p,evidence/name)
    base.save(evidence/"package_integrity.json",dict(files=[dict(name=p.name,sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted(evidence.iterdir()) if p.is_file() and p.name!="package_integrity.json"]))
    print(json.dumps(dict(files=len(records),evidence_files=len(list(evidence.iterdir())))))


if __name__=="__main__":main()
