from time import perf_counter
from research_experiments.wzzt_retune.engine import PoolLP,build_bank,search,canonical
from research_experiments.wzzt_refine.core import scored,select_pre,finalize,regularized_pool,backward,two_step
from .common import PoolView,validate_observations


def native_model(point,train,validation):
    from research_experiments.wzzt_systematic.common import matrices,linear_loss
    p=dict(point);weights={tuple(s):(r,x) for s,r,x in zip(p["supports"],p["r"],p["x"])}
    p.update(supports=p["native_family"],r=[weights.get(tuple(s),(0,0))[0] for s in p["native_family"]],
             x=[weights.get(tuple(s),(0,0))[1] for s in p["native_family"]])
    r,x=matrices(train[0].shape[1],p["supports"],p["r"],p["x"])
    p["objective"]=linear_loss(train,r,x)["mae"]
    return scored(p,validation)


def infer(train,validation,*,targeted=False,extensions=True,checkpoint=None):
    """Fresh observations -> current pipeline and fixed controls; no truth input."""
    validate_observations(train,validation)
    t0=perf_counter();bank,pool,fronts,calibrations=build_bank(train);cal=calibrations["raw"]
    result=dict(labels=list(train[0].columns),models={},candidates={},searches={},LP={},pool=pool,
        fronts=fronts,calibration=cal,truth_used_by_estimator=False,parameter_source="frozen 20261028 settings")
    lp=PoolLP(train,pool,"raw")
    try:
        rnj=[];nj=[]
        for item in bank:
            if item["preprocess"]!="raw":continue
            fit=scored(dict(lp.fit(item["family"]),config=item["config"],native_family=item["family"],origin="tree_bank"),validation)
            (rnj if item["method"]=="RNJ" else nj).append(fit)
        result["rnj_bank"]=rnj
        result["models"]["NJ"]=native_model(next(p for p in nj if p["config"]=="raw_RX75_0.16"),train,validation)
        result["models"]["RNJ_native"]=native_model(next(p for p in rnj if p["config"]=="raw_RX75_0.24"),train,validation)
        result["models"]["RNJ_L1_bank"]=finalize(rnj,train,validation,cal)
        initial=next(p for p in rnj if p["config"]=="raw_RX75_0.16")
        main=search(lp,initial["native_family"],seconds=25,repair=True)
        main_points=[scored(p,validation,"primary25") for p in main["path"]]
        baseline=rnj+main_points;result["searches"]["primary25"]=main
        result["models"]["baseline25"]=finalize(baseline,train,validation,cal)
        prepared=[]
        for i,p in enumerate(rnj):
            cut=finalize([p],train,validation,cal)
            fit=scored(lp.fit(cut["supports"]),validation,"extra_start_prepare")
            prepared.append((i,fit))
        prepared.sort(key=lambda v:(v[1]["validation_mae"],len(v[1]["supports"]),v[0]))
        seen={canonical(initial["native_family"])};extra=[]
        for i,p in prepared:
            family=canonical(p["supports"])
            if family in seen:continue
            seen.add(family);extra.append((i,p))
            if len(extra)==2:break
        multi=list(baseline)
        for serial,(index,p) in enumerate(extra):
            trajectory=search(lp,p["supports"],seconds=25,repair=True)
            result["searches"][f"extra{serial}"]=dict(trajectory,source_config=rnj[index]["config"])
            multi.extend(scored(p,validation,f"extra{serial}") for p in trajectory["path"])
        result["models"]["multistart"]=finalize(multi,train,validation,cal)
        result["base_points"]=multi
        chosen=select_pre(multi)
        result["base_pipeline_seconds"]=perf_counter()-t0
        if checkpoint is not None:checkpoint(result)
        if targeted:
            deletion=backward(lp,chosen["supports"],validation,seconds=15,max_deletions=1)
            result["searches"]["delete_one_round"]=deletion
            result["candidates"]["delete_all"]=deletion["points"]
            result["models"]["delete_all"]=finalize(multi+deletion["points"],train,validation,cal)
            for method,fraction in (("delete_guard0",0.),("delete_guard1pct",.01),("delete_guard3pct",.03)):
                points=[p for p in deletion["points"] if p["validation_mae"]<=chosen["validation_mae"]*(1+fraction)+1e-12]
                result["candidates"][method]=points
                result["models"][method]=finalize(multi+points,train,validation,cal)
        result["LP"]["base"]=lp.solves
    finally:lp.close()
    if targeted:
        for width in (3,5):
            branch=PoolLP(train,pool,"raw")
            try:
                trajectory=two_step(branch,chosen["supports"],validation,seconds=25,width=width)
                name=f"two_step_w{width}";result["searches"][name]=trajectory
                result["candidates"][name]=trajectory["points"]
                result["models"][name]=finalize(multi+trajectory["points"],train,validation,cal)
                result["LP"][name]=branch.solves
            finally:branch.close()
            if checkpoint is not None:checkpoint(result)
    if extensions:
        augmentation=regularized_pool(train,pool);result["augmentation"]=augmentation
        domains={"restart_control":pool,"ridge_union":augmentation["pool"]}
        for method,strength in (("ridge_1e3",.001),("ridge_1e2",.01)):
            members=list(pool)
            for front in augmentation["fronts"]:
                if front["strength"]==strength:
                    for tree in front["tree_families"]:members.extend(tree["family"])
            domains[method]=canonical(members)
        result["domains"]=domains
        # Each matched 25-second extension starts with its own fresh LP/cache.
        # The common union library keeps variable order and inactive columns equal.
        for method in ("restart_control","ridge_1e3","ridge_1e2","ridge_union"):
            branch=PoolLP(train,augmentation["pool"],"raw")
            try:
                trajectory=search(PoolView(branch,domains[method]),chosen["supports"],seconds=25,repair=True)
                points=[scored(p,validation,method) for p in trajectory["path"]]
                result["searches"][method]=trajectory;result["candidates"][method]=points
                result["models"][method]=finalize(multi+points,train,validation,cal)
                result["LP"][method]=branch.solves
            finally:branch.close()
            if checkpoint is not None:checkpoint(result)
    result["seconds"]=perf_counter()-t0
    return result
