"""Explicit result-based benchmark tuning; no per-instance oracle primary."""
from itertools import product
import numpy as np
from research_experiments.wzzt_systematic.common import matrices

METHODS=("NJ","WZZT_free","WZZT_pool","RNJ_WZZT")


def configurations(method):
    if method=="NJ":
        return [dict(preprocess=p,geometry=g,collapse=t) for p,g,t in product(("raw","demean"),("R","RX75","X"),(0.,.04,.08,.16,.24))]
    return [dict(preprocess=p,resolution=t,selection=s,repair=r)
            for p,t,s,r in product(("raw","demean"),(0.,.04,.08,.12,.16,.24),
                ("last","validation_min","validation_simple_1pct","validation_simple_3pct"),
                (False,True) if method=="WZZT_pool" else (False,))]


def select(result,method,config):
    group=result["methods"][config["preprocess"]]
    if method=="NJ":
        key=f"{config['preprocess']}_{config['geometry']}_{config['collapse']:g}"
        point=next(p for p in group["NJ"] if p["config"]==key)
        lookup={tuple(s):(r,x) for s,r,x in zip(point["supports"],point["r"],point["x"])}
        family=point["native_family"]
        r=[lookup.get(tuple(s),(0.,0.))[0] for s in family]
        x=[lookup.get(tuple(s),(0.,0.))[1] for s in family]
        return dict(point,supports=family,r=r,x=x,selection_index=0)
    if method=="WZZT_free":points=group[method]["path"]
    elif method=="WZZT_pool":points=group[method][str(config["repair"])]["path"]
    else:points=group[method]
    rule=config["selection"]
    if rule=="last":index=len(points)-1
    else:
        best=min(range(len(points)),key=lambda i:points[i]["validation_mae"])
        fraction=0 if rule=="validation_min" else .01 if rule=="validation_simple_1pct" else .03
        eligible=[i for i,p in enumerate(points) if p["validation_mae"]<=points[best]["validation_mae"]*(1+fraction)+1e-12]
        index=min(eligible,key=lambda i:(len(points[i]["supports"]),points[i]["validation_mae"],i)) if fraction else best
    point=points[index]
    cal=result["calibrations"][config["preprocess"]]
    threshold=cal["median_root_depth"]*config["resolution"]
    keep=[i for i,s in enumerate(point["supports"]) if len(s)==1 or cal["a"]*point["r"][i]+cal["b"]*point["x"][i]+1e-10>=threshold]
    return dict(point,supports=[point["supports"][i] for i in keep],r=[point["r"][i] for i in keep],
                x=[point["x"][i] for i in keep],selection_index=index)


def score(result,method,config,truth):
    from research_experiments.rnj.audit_source_ac96_benchmark import clade_metrics,relative_error
    point=select(result,method,config)
    labels=result["labels"]
    clades=[{labels[i] for i in s} for s in point["supports"]]
    metrics=clade_metrics(clades,truth["clades"],labels)
    r,x=matrices(len(labels),point["supports"],point["r"],point["x"])
    if method=="NJ":
        r=np.asarray(result["fronts"][config["preprocess"]]["R"])
        x=np.asarray(result["fronts"][config["preprocess"]]["X"])
    metrics.update(R_error=relative_error(r,truth["R"]),X_error=relative_error(x,truth["X"]),
                   validation_mae=point["validation_mae"],support_count=len(point["supports"]))
    return point,metrics
