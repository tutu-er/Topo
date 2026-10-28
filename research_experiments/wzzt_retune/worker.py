from pathlib import Path
import sys
from time import perf_counter
import traceback
import numpy as np

from research_experiments.wzzt_systematic.common import observed,linear_loss,matrices,read,write
from .engine import PoolLP,build_bank,search,free_search


def run(payload):
    payload=Path(payload)
    cfg=read(payload);out=payload.parent
    obs=observed(cfg["train"],cfg["N"])
    validation=observed(cfg["validation"])
    labels=list(obs[0].columns)
    started=perf_counter()
    bank,pool,fronts,calibrations=build_bank(obs)
    result=dict(job_id=cfg["job_id"],case=cfg["case"],condition=cfg["condition"],N=cfg["N"],labels=labels,
        methods={},fronts=fronts,calibrations=calibrations,pool=pool,
        truth_used_by_worker=False,source_input_scope="train and independent validation observations only")
    for preprocessing in ("raw","demean"):
        t=perf_counter()
        lp=PoolLP(obs,pool,preprocessing)
        group=dict(NJ=[],WZZT_pool={},RNJ_WZZT=[],WZZT_free={})
        for item in bank:
            if item["preprocess"]!=preprocessing:continue
            point=dict(lp.fit(item["family"]),config=item["config"],method=item["method"],
                       native_family=item["family"],origin="tree_bank")
            rr,xx=matrices(len(labels),point["supports"],point["r"],point["x"])
            point["validation_mae"]=linear_loss(validation,rr,xx)["mae"]
            if item["method"]=="NJ":group["NJ"].append(point)
            else:group["RNJ_WZZT"].append(point)
        for correction in (False,True):
            trajectory=search(lp,[(i,) for i in range(len(labels))],seconds=25,repair=correction)
            group["WZZT_pool"][str(correction)]=trajectory
        selected=next(item for item in bank if item["preprocess"]==preprocessing and item["method"]=="RNJ"
                      and item["mode"]=="RX75" and item["tolerance"]==.16)
        corrected=search(lp,selected["family"],seconds=25,repair=True)
        for p in corrected["path"]:group["RNJ_WZZT"].append(dict(p,origin="tree_initialized_correction",config="RX75_0.16_search"))
        group["RNJ_search"]=corrected
        group["pool_seconds"]=perf_counter()-t
        result["methods"][preprocessing]=group
        write(out/"checkpoint.json",result)
        # Pool variants are frozen before free-domain discovery adds supports.
        group["WZZT_free"]=free_search(lp,out/f"free_{preprocessing}",seconds=40,per_solve=5)
        for family in (group["RNJ_WZZT"],group["WZZT_pool"]["False"]["path"],
                       group["WZZT_pool"]["True"]["path"],group["WZZT_free"]["path"]):
            for point in family:
                rr,xx=matrices(len(labels),point["supports"],point["r"],point["x"])
                point["validation_mae"]=linear_loss(validation,rr,xx)["mae"]
        group["LP_solves"]=lp.solves
        group["seconds"]=perf_counter()-t
        result["methods"][preprocessing]=group
        write(out/"checkpoint.json",result)
        lp.close()
        print(cfg["job_id"],preprocessing,"pool",len(pool),"free points",len(group["WZZT_free"]["path"]),flush=True)
    result["seconds"]=perf_counter()-started
    result["status"]="returned"
    write(out/"result.json",result)


if __name__=="__main__":
    try:run(sys.argv[1])
    except Exception as e:
        write(Path(sys.argv[1]).parent/"error.json",dict(error=f"{type(e).__name__}: {e}",traceback=traceback.format_exc()))
        raise
