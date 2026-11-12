from pathlib import Path
from datetime import datetime,timezone
import argparse
import shutil
from research_experiments.wzzt_systematic.common import read,write,sha

ROOT=Path(__file__).resolve().parents[2]
PRIOR=ROOT/"outputs/wzzt_rnj_budget_20261028_full"


def prepare(output,only=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    for p,h in read(PRIOR/"manifest.json")["local"].items():
        if sha(PRIOR/p)!=h:raise ValueError("prior source/input changed: "+p)
    shutil.copytree(PRIOR/"source",output/"source",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    shutil.copytree(Path(__file__).parent,output/"source/research_experiments/wzzt_refine",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    old=read(PRIOR/"protocol.json")
    jobs=[j for j in old["jobs"] if only is None or any(s in j for s in only)]
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),prior=str(PRIOR),parent=old["parent"],baseline=old["baseline"],
        jobs=jobs,planned=len(jobs),cases=old["cases"],conditions=old["conditions"],sizes=old["sizes"],
        variants=["baseline25","multistart_original_rule","backward","ridge_pool","two_step","combined"],
        selection=dict(preprocess="raw",resolution=.16,rule="validation_simple_3pct",refit_after_final_contraction=False),
        parameters=dict(backward_max_deletions=2,ridge_strengths=[.001,.01],ridge_alpha="lambda * trace(D.T D)/(2*n), D=[P Q]",
            beam_width=3,beam_depth=2,beam_first_phase_fraction=.4,training_gain_absolute=1e-10,training_gain_relative=.001),
        budgets=dict(backward=15,ridge_repair=25,two_step=25,combined_two_step=25,combined_backward=15,
                     fixed_LP=15,workers=4,worker_hard_seconds=300),
        information_scope="estimator reads training/validation observations and prior observation-derived banks only",
        changes_scope="incremental refinements on the exact prior candidate bank; old paths reused; extra candidate and search cost disclosed",
        truth_tuning=False,shared_LP_cache=True,test_replicate=1,validation_replicate=2,train_replicate=0)
    external={}
    for job in jobs:
        p=read(PRIOR/"jobs"/job/"payload.json");p["prior_result"]=str(PRIOR/"jobs"/job/"result.json")
        d=output/"jobs"/job;d.mkdir(parents=True);write(d/"payload.json",p)
        paths=[Path(p[k]) for k in ("train","validation","parent_result","prior_result")]
        paths+=[Path(p["train"]).with_name(n) for n in ("train_truth.npz","test_observed.npz","test_truth.npz")]
        for f in paths:external[str(f)]=sha(f)
    write(output/"protocol.json",protocol)
    write(output/"manifest.json",dict(local={p.relative_to(output).as_posix():sha(p) for p in output.rglob("*") if p.is_file()},external=external))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");p.add_argument("--only",nargs="+");a=p.parse_args();prepare(a.output,a.only)
