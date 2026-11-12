from pathlib import Path
from time import perf_counter
import sys
import traceback
from research_experiments.wzzt_systematic.common import read,write,observed
from research_experiments.wzzt_retune.engine import PoolLP,search
from .core import scored,select_pre,finalize,backward,regularized_pool,two_step

VARIANTS=("backward","ridge_pool","two_step","combined")


def refine_bank(train,validation,base_points,base_pool,calibration,*,root_label=1,checkpoint=None):
    """Refine an observation-derived RNJ/WZZT bank; never accepts truth."""
    if not base_points:raise ValueError("a nonempty observation-derived base bank is required")
    labels=list(train[0].columns)
    if root_label in labels or len(set(labels))!=len(labels):raise ValueError("unique terminal labels excluding root required")
    for obs in (train,validation):
        if any(list(f.columns)!=labels or not f.index.equals(obs[3].index) for f in obs[:3]):
            raise ValueError("P/Q/V/root indices and ordered terminal columns must align")
    for p in base_points:
        if len(p["supports"])!=len(p["r"]) or len(p["r"])!=len(p["x"]):raise ValueError("support/weight lengths differ")
        if any([i] not in [list(s) for s in p["supports"]] for i in range(len(labels))):raise ValueError("all singleton leaves required")
        if any(not 0<=w<=2 for w in p["r"]+p["x"]):raise ValueError("base weights must lie in [0,2]")
    started=perf_counter();base=[scored(p,validation) for p in base_points]
    chosen=select_pre(base);original_lp=PoolLP(train,base_pool,"raw")
    result=dict(models={},candidates={},operations={},calibration=calibration,base_pool=base_pool,
                base_candidate_count=len(base),truth_used_by_estimator=False)
    try:
        b=backward(original_lp,chosen["supports"],validation)
        result["operations"]["backward"]=b
        result["candidates"]["backward"]=b["points"]
        result["models"]["backward"]=finalize(base+b["points"],train,validation,calibration)
        t=two_step(original_lp,chosen["supports"],validation)
        result["operations"]["two_step"]=t
        result["candidates"]["two_step"]=t["points"]
        result["models"]["two_step"]=finalize(base+t["points"],train,validation,calibration)
        result["LP_original"]=original_lp.solves
    finally:original_lp.close()
    if checkpoint is not None:checkpoint(result)
    augmentation=regularized_pool(train,base_pool,root_label=root_label)
    result["augmentation"]=augmentation
    augmented_lp=PoolLP(train,augmentation["pool"],"raw")
    try:
        trajectory=search(augmented_lp,chosen["supports"],seconds=25,repair=True)
        ridge_points=[scored(p,validation,"ridge_pool_repair") for p in trajectory["path"]]
        result["operations"]["ridge_pool"]=trajectory
        result["candidates"]["ridge_pool"]=ridge_points
        result["models"]["ridge_pool"]=finalize(base+ridge_points,train,validation,calibration)
        if checkpoint is not None:checkpoint(result)
        combined_base=base+ridge_points
        combined_start=select_pre(combined_base)
        joint=two_step(augmented_lp,combined_start["supports"],validation)
        for p in joint["points"]:p["origin"]="combined/"+p["origin"]
        combined_base+=joint["points"]
        after=select_pre(combined_base)
        deletion=backward(augmented_lp,after["supports"],validation)
        for p in deletion["points"]:p["origin"]="combined/"+p["origin"]
        all_new=ridge_points+joint["points"]+deletion["points"]
        result["operations"].update(combined_two_step=joint,combined_backward=deletion)
        result["candidates"]["combined"]=all_new
        result["models"]["combined"]=finalize(base+all_new,train,validation,calibration)
        result["LP_augmented"]=augmented_lp.solves
    finally:augmented_lp.close()
    result["seconds"]=perf_counter()-started
    return result


def run(payload_file):
    payload_file=Path(payload_file);cfg=read(payload_file);out=payload_file.parent
    parent=read(cfg["parent_result"]);prior=read(cfg["prior_result"])
    train=observed(cfg["train"],cfg["N"]);validation=observed(cfg["validation"])
    points=list(parent["methods"]["raw"]["RNJ_WZZT"])
    for name,trajectory in prior["searches"].items():
        if name.startswith("multistart_"):points.extend(scored(p,validation,name) for p in trajectory["path"])
    header=dict(job_id=cfg["job_id"],case=cfg["case"],condition=cfg["condition"],N=cfg["N"],labels=parent["labels"])
    result=refine_bank(train,validation,points,parent["pool"],parent["calibrations"]["raw"],
        checkpoint=lambda r:write(out/"checkpoint.json",dict(r,**header,status="checkpoint")))
    result.update(job_id=cfg["job_id"],case=cfg["case"],condition=cfg["condition"],N=cfg["N"],labels=parent["labels"],status="returned")
    write(out/"result.json",result)
    print(cfg["job_id"],"done",round(result["seconds"],2),"pool",len(result["base_pool"]),len(result["augmentation"]["pool"]),flush=True)


if __name__=="__main__":
    try:run(sys.argv[1])
    except Exception as e:
        write(Path(sys.argv[1]).parent/"error.json",dict(error=f"{type(e).__name__}: {e}",traceback=traceback.format_exc()));raise
