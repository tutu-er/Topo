from pathlib import Path
from time import perf_counter
import sys
import traceback
from research_experiments.wzzt_systematic.common import read,write,observed,matrices,linear_loss
from research_experiments.wzzt_retune.engine import PoolLP,search,canonical
from research_experiments.wzzt_retune.selection import select

CONFIG=dict(preprocess="raw",resolution=.16,selection="validation_simple_3pct",repair=True)
VARIANTS=("baseline25","long75","final_score","post_refit","multistart75")


def measured(point,obs,validation,*,origin=None):
    p=dict(point)
    r,x=matrices(obs[0].shape[1],p["supports"],p["r"],p["x"])
    p["pre_contraction_objective"]=p.get("objective")
    p["objective"]=linear_loss(obs,r,x)["mae"]
    p["validation_mae"]=linear_loss(validation,r,x)["mae"]
    if origin is not None:p["origin"]=origin
    return p


def choose(points,fraction=.03):
    best=min(p["validation_mae"] for p in points)
    eligible=[(i,p) for i,p in enumerate(points) if p["validation_mae"]<=best*(1+fraction)+1e-12]
    index,point=min(eligible,key=lambda item:(len(item[1]["supports"]),item[1]["validation_mae"],item[0]))
    return dict(point,selection_index=index)


def contract(point,cal):
    threshold=.16*cal["median_root_depth"]
    which=[i for i,s in enumerate(point["supports"]) if len(s)==1 or
           cal["a"]*point["r"][i]+cal["b"]*point["x"][i]+1e-10>=threshold]
    return dict(point,supports=[point["supports"][i] for i in which],
                r=[point["r"][i] for i in which],x=[point["x"][i] for i in which])


def run(payload_file):
    payload_file=Path(payload_file);cfg=read(payload_file);out=payload_file.parent
    old=read(cfg["parent_result"])
    obs,validation=observed(cfg["train"],cfg["N"]),observed(cfg["validation"])
    raw=old["methods"]["raw"];cal=old["calibrations"]["raw"]
    bank=[p for p in raw["RNJ_WZZT"] if p.get("origin")=="tree_bank"]
    start=next(p for p in bank if p["config"]=="raw_RX75_0.16")
    started=perf_counter();lp=PoolLP(obs,old["pool"],"raw")
    result=dict(job_id=cfg["job_id"],case=cfg["case"],condition=cfg["condition"],N=cfg["N"],
        labels=old["labels"],calibration=cal,pool=old["pool"],config=CONFIG,
        truth_used_by_worker=False,models={},candidates={},searches={})
    # Recreate only this branch's LP bank. Data, pool, initial family, objective,
    # coefficient bounds, gain rule and selection rule all stay fixed.
    for p in raw["NJ"]+bank:lp.fit(p["native_family"])
    old_selected=select(old,"RNJ_WZZT",CONFIG)
    result["models"]["baseline25"]=measured(old_selected,obs,validation)
    long=search(lp,start["native_family"],seconds=75,repair=True)
    result["searches"]["long75"]=long
    long_points=bank+[measured(dict(p,origin="tree_initialized_correction",config="RX75_0.16_search75"),obs,validation) for p in long["path"]]
    temporary=dict(old,methods={"raw":{"RNJ_WZZT":long_points}})
    result["models"]["long75"]=measured(select(temporary,"RNJ_WZZT",CONFIG),obs,validation)
    # A score must describe the actual contracted model being returned.
    scored=[measured(contract(p,cal),obs,validation,origin=p["origin"]+"/final_score") for p in long_points]
    result["candidates"]["final_score"]=scored
    result["models"]["final_score"]=choose(scored)
    refitted=[]
    for i,p in enumerate(scored):
        fit=measured(lp.fit(p["supports"]),obs,validation,origin=p["origin"]+"/refit")
        fit.update(source_index=i,source_config=long_points[i].get("config"),source_family=p["supports"])
        refitted.append(fit)
    repaired=scored+refitted
    result["candidates"]["post_refit"]=repaired
    result["models"]["post_refit"]=choose(repaired)
    write(out/"checkpoint.json",result)
    # Same nominal 75 s search allowance: retain the original 25 s path, then
    # spend at most 2 x 25 s at distinct validation-ranked, contracted RNJ starts.
    # Initial-source selection sees validation data only and never topology truth.
    initial_options=refitted[:len(bank)]
    ranked=sorted(enumerate(initial_options),key=lambda t:(t[1]["validation_mae"],len(t[1]["supports"]),t[0]))
    seen={canonical(start["native_family"])};extra=[]
    for index,p in ranked:
        family=canonical(p["supports"])
        if family in seen:continue
        seen.add(family);extra.append((index,p))
        if len(extra)==2:break
    multi_raw=bank+[dict(p,origin="original25_correction") for p in raw["RNJ_search"]["path"]]
    for serial,(index,p) in enumerate(extra):
        trajectory=search(lp,p["supports"],seconds=25,repair=True)
        result["searches"][f"multistart_{serial}"]=dict(trajectory,source_index=index,
            source_config=bank[index]["config"],source_family=p["supports"])
        multi_raw.extend(dict(pt,origin=f"multistart_{serial}",source_config=bank[index]["config"]) for pt in trajectory["path"])
    multi_points=[]
    for i,p in enumerate(multi_raw):
        cut=measured(contract(p,cal),obs,validation,origin=p["origin"]+"/final_score")
        multi_points.append(cut)
        fitted=measured(lp.fit(cut["supports"]),obs,validation,origin=p["origin"]+"/refit")
        fitted.update(source_index=i,source_config=p.get("source_config",p.get("config")))
        multi_points.append(fitted)
    result["candidates"]["multistart75"]=multi_points
    result["models"]["multistart75"]=choose(multi_points)
    result.update(LP_solves=lp.solves,seconds=perf_counter()-started,
        original_search_seconds=raw["RNJ_search"]["seconds"],original_search_stop=raw["RNJ_search"]["reason"],
        long_start_objective_error=abs(long["path"][0]["objective"]-raw["RNJ_search"]["path"][0]["objective"]),
        status="returned")
    lp.close();write(out/"result.json",result)
    print(cfg["job_id"],"done",round(result["seconds"],2),long["reason"],flush=True)


if __name__=="__main__":
    try:run(sys.argv[1])
    except Exception as e:
        write(Path(sys.argv[1]).parent/"error.json",dict(error=f"{type(e).__name__}: {e}",traceback=traceback.format_exc()))
        raise
