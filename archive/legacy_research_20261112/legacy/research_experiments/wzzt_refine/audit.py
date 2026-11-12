from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from research_experiments.wzzt_systematic.common import read,write,sha,observed
from research_experiments.wzzt_retune.engine import compatible,canonical,arrays
from research_experiments.wzzt_budget.audit import independent_lp
from .core import ridge_alpha,pair_family


def audit(output,destination):
    output=Path(output).resolve();destination=Path(destination).resolve();protocol=read(output/"protocol.json");manifest=read(output/"manifest.json")
    frame=pd.read_csv(destination/"per_job.csv")
    checks=[dict(check="frozen inputs and producer",passed=all(sha(output/p)==h for p,h in manifest["local"].items()) and all(sha(Path(p))==h for p,h in manifest["external"].items()))]
    physical=[];LP=[];front_checks=[];pair_checks=[];backward_checks=[];references=[]
    for job in protocol["jobs"]:
        d=output/"jobs"/job;payload=read(d/"payload.json")
        if not (d/"result.json").exists():checks.append(dict(check=job+" completed",passed=False));continue
        result=read(d/"result.json");obs=observed(payload["train"],payload["N"]);val=observed(payload["validation"])
        checks.append(dict(check=job+" observation-only estimator",passed=result["truth_used_by_estimator"] is False))
        oldpool={tuple(s) for s in result["base_pool"]};newpool={tuple(s) for s in result["augmentation"]["pool"]}
        checks.append(dict(check=job+" original candidate pool retained",passed=oldpool<=newpool))
        for name in ("LP_original","LP_augmented"):
            rows=result[name]
            LP.append(dict(job_id=job,kind=name,count=len(rows),passed=all(r["status"]==2 and
                max(r["max_constraint_violation"],r["max_bound_violation"])<=1e-7 and abs(r["physical"]-r["objective"])<=1e-8 for r in rows)))
        points=list(result["models"].values())+[p for group in result["candidates"].values() for p in group]
        for name,operation in result["operations"].items():
            points+=operation.get("path",[])
            if "backward" in name:
                for decision in operation["decisions"]:
                    family={tuple(s) for s in decision["current_supports"]}
                    for p in decision["candidates"]:
                        subset={tuple(s) for s in p["supports"]}
                        backward_checks.append(dict(job_id=job,kind=name,passed=subset<=family and tuple(p["removed"]) not in subset and
                            all((i,) in subset for i in range(len(result["labels"]))) and
                            p["training_objective"]>=decision["current_objective"]-1e-8))
                    if decision["accepted"]:
                        best=min([decision["current_validation_mae"]]+[p["validation_mae"] for p in decision["candidates"]])
                        chosen=decision["candidates"][decision["selected_index"]-1]
                        backward_checks.append(dict(job_id=job,kind=name,passed=chosen["validation_mae"]<=1.03*best+1e-12 and
                            len(chosen["supports"])<len(decision["current_supports"])))
            if "two_step" in name:
                for row in operation["pairs"]:
                    family=pair_family(canonical(operation["initial"]),tuple(row["first"]),tuple(row["second"]))
                    accepted=operation["current_objective"]-row["endpoint_objective"]>1e-10+.001*operation["current_objective"]
                    pair_checks.append(dict(job_id=job,kind=name,passed=family==canonical(row["family"]) and accepted==row["accepted"]))
        seen=set()
        for p in points:
            key=str((p["supports"],p["r"],p["x"]))
            if key in seen:continue
            seen.add(key);family=canonical(p["supports"])
            legal=len(family)==len(p["supports"]) and all((i,) in family for i in range(len(result["labels"]))) and all(compatible(family[:i],s) for i,s in enumerate(family))
            bounded=all(-1e-9<=w<=2+1e-9 for w in p["r"]+p["x"])
            errors=[]
            for data,name in ((obs,"objective"),(val,"validation_mae")):
                if name not in p:continue
                P,Q,V,root=data;target=root.to_numpy()[:,None]**2-V.to_numpy()**2;prediction=np.zeros_like(target)
                pp,qq=P.to_numpy(),Q.to_numpy()
                for s,r,x in zip(p["supports"],p["r"],p["x"],strict=True):prediction[:,s]+=(r*pp[:,s].sum(axis=1)+x*qq[:,s].sum(axis=1))[:,None]
                errors.append(abs(float(np.mean(abs(prediction-target)))-p[name]))
            physical.append(dict(job_id=job,passed=legal and bounded and max(errors)<1e-8,max_score_error=max(errors)))
        for front in result["augmentation"]["fronts"]:
            p,q,_=arrays(obs,front["preprocess"]);alpha,scale=ridge_alpha(p,q,front["strength"])
            r,x=np.asarray(front["R"]),np.asarray(front["X"])
            front_checks.append(dict(job_id=job,preprocess=front["preprocess"],strength=front["strength"],
                passed=bool(abs(alpha-front["alpha"])<1e-12 and abs(scale-front["gram_scale"])<1e-12 and
                    front["diagnostics"].get("success",False) and np.all(r>=-1e-8) and np.all(x>=-1e-8) and
                    np.allclose(r,r.T,atol=1e-10) and np.allclose(x,x.T,atol=1e-10))))
        if payload["condition"]=="reference" and payload["N"] in (24,288):
            for name in ("backward","ridge_pool","two_step","combined"):
                candidates=result["candidates"][name]
                point=candidates[0] if candidates else result["operations"]["ridge_pool"]["path"][0]
                references.append(dict(job_id=job,variant=name,**independent_lp(point,obs)))
    prior=pd.read_csv(Path(__file__).resolve().parents[2]/"docs/wzzt_rnj_budget_20261028/per_job.csv")
    for variant in ("baseline25","multistart_original_rule"):
        joined=frame[frame.variant==variant].merge(prior[prior.variant==variant][["job_id","clade_f1"]],on="job_id",suffixes=("","_prior"),validate="one_to_one")
        checks.append(dict(check=variant+" exact published score reproduction",passed=len(joined)==80 and np.allclose(joined.clade_f1,joined.clade_f1_prior)))
    checks.append(dict(check="all declared model denominators",passed=len(frame)==480 and frame.model_available.all() and not frame.duplicated(["job_id","variant"]).any()))
    groups=(checks,physical,LP,front_checks,pair_checks,backward_checks,references)
    summary=dict(all_passed=all(r["passed"] for group in groups for r in group),checks=len(checks),physical_models=len(physical),
        LP_solves=sum(r["count"] for r in LP),ridge_fronts=len(front_checks),pair_proposals=len(pair_checks),backward_proposals=len(backward_checks),
        independent_LP_checks=len(references),inputs=frame.job_id.nunique(),model_rows=len(frame))
    write(destination/"audit_summary.json",summary);write(destination/"audit_details.json",dict(checks=checks,physical=physical,LP=LP,
        fronts=front_checks,pairs=pair_checks,backward=backward_checks,independent=references))
    print(summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");p.add_argument("destination");a=p.parse_args();audit(a.output,a.destination)
