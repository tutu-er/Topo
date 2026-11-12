from pathlib import Path
from datetime import datetime,timezone
import argparse
import shutil
from research_experiments.wzzt_systematic.common import read,write,sha

ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/"outputs/wzzt_retune_20261028_round2"


def prepare(output,only=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    for p,h in read(PARENT/"manifest.json")["local"].items():
        if sha(PARENT/p)!=h:raise ValueError("parent input/source changed: "+p)
    shutil.copytree(PARENT/"source",output/"source",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    shutil.copytree(Path(__file__).parent,output/"source/research_experiments/wzzt_budget",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    parent_protocol=read(PARENT/"protocol.json")
    jobs=[j for j in parent_protocol["jobs"] if only is None or any(s in j for s in only)]
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),parent=str(PARENT),baseline=parent_protocol["baseline"],
        jobs=jobs,planned=len(jobs),cases=parent_protocol["cases"],conditions=parent_protocol["conditions"],
        sizes=parent_protocol["sizes"],config=dict(preprocess="raw",resolution=.16,selection="validation_simple_3pct",repair=True),
        variants=["baseline25","long75","final_score","post_refit","multistart75"],
        budgets=dict(original=25,long_single_start=75,multistart_original=25,extra_starts=2,per_extra_start=25,
                     fixed_LP=15,workers=4,worker_hard_seconds=300),
        data_policy="reuse byte-identical train, validation, test and data-derived candidate pool; no new truth tuning",
        algorithm_change_policy="long75 only increases search time; remaining patches are separately labelled",
        time_scope="search caps exclude shared LP bank and final-model postprocessing; original 25s trajectory reused for multistart")
    external={}
    for job in jobs:
        p=read(PARENT/"jobs"/job/"payload.json")
        parent_result=PARENT/"jobs"/job/"result.json"
        p["parent_result"]=str(parent_result)
        d=output/"jobs"/job;d.mkdir(parents=True);write(d/"payload.json",p)
        for target in [Path(p["train"]),Path(p["validation"]),parent_result]:external[str(target)]=sha(target)
        for name in ("test_observed.npz","test_truth.npz"):
            f=Path(parent_protocol["baseline"])/"data"/p["case"]/p["condition"]/name;external[str(f)]=sha(f)
    write(output/"protocol.json",protocol)
    write(output/"manifest.json",dict(local={p.relative_to(output).as_posix():sha(p) for p in output.rglob("*") if p.is_file()},external=external))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");p.add_argument("--only",nargs="+");a=p.parse_args();prepare(a.output,a.only)
