"""Independent sparse-primal and physical-atom audit of the retuned run."""
from pathlib import Path
import argparse
import sys
import numpy as np
from scipy.sparse import csr_matrix,eye,hstack,vstack
from scipy.optimize import linprog
from research_experiments.wzzt_systematic.common import read,write,sha,observed
from research_experiments.wzzt_solver_tuning.profile_gaps import primal_audit
from .engine import arrays,compatible
from .selection import select


def audit(output,destination,*,partial=False,independent_lp=False):
    output=Path(output).resolve();destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    manifest=read(output/"manifest.json")
    checks=[];lp_summaries=[];point_checks=[];primal_checks=[];independent=[]
    checks.append(dict(check="frozen local inputs/source",passed=all(sha(output/p)==h for p,h in manifest["local"].items())))
    checks.append(dict(check="unchanged round1 training/test bytes",passed=all(sha(Path(p))==h for p,h in manifest["external"].items())))
    protocol=read(output/"protocol.json")
    for job in protocol["jobs"]:
        directory=output/"jobs"/job
        ended=any((directory/n).exists() for n in ("result.json","error.json","timeout.json"))
        if partial and not ended:continue
        if not (directory/"result.json").exists():
            checks.append(dict(check=job+" dataset returned",passed=False));continue
        result=read(directory/"result.json");payload=read(directory/"payload.json")
        obs=observed(payload["train"],payload["N"])
        validation=observed(payload["validation"])
        checks.append(dict(check=job+" validation has independent injections",passed=not np.array_equal(obs[0].to_numpy(),validation[0].to_numpy())))
        for preprocessing,group in result["methods"].items():
            p,q,y=arrays(obs,preprocessing)
            records=group["LP_solves"]
            lp_ok=all(r["status"]==2 and max(r["max_constraint_violation"],r["max_bound_violation"])<=1e-7
                      and abs(r["physical"]-r["objective"])<=1e-8 for r in records)
            lp_summaries.append(dict(job_id=job,preprocess=preprocessing,count=len(records),passed=lp_ok,
                max_constraint_violation=max((r["max_constraint_violation"] for r in records),default=0.),
                max_objective_error=max((abs(r["physical"]-r["objective"]) for r in records),default=0.)))
            points=group["NJ"]+group["RNJ_WZZT"]+group["WZZT_pool"]["False"]["path"]+group["WZZT_pool"]["True"]["path"]+group["WZZT_free"]["path"]
            seen=set()
            for index,point in enumerate(points):
                signature=str((point["supports"],point["r"],point["x"]))
                if signature in seen:continue
                seen.add(signature)
                family=[tuple(s) for s in point["supports"]]
                prediction=np.zeros_like(y)
                for support,r,x in zip(family,point["r"],point["x"],strict=True):
                    contribution=r*p[:,support].sum(axis=1)+x*q[:,support].sum(axis=1)
                    prediction[:,support]+=contribution[:,None]
                objective=float(np.mean(abs(prediction-y)))
                legal=(len(set(family))==len(family) and all((i,) in family for i in range(p.shape[1]))
                       and all(compatible(family[:i],s) for i,s in enumerate(family)))
                bounded=all(-1e-9<=w<=2.+1e-9 for w in point["r"]+point["x"])
                point_checks.append(dict(job_id=job,preprocess=preprocessing,point=index,
                    passed=bool(legal and bounded and abs(objective-point["objective"])<=1e-8),
                    objective_error=abs(objective-point["objective"])))
        for folder in sorted(directory.glob("free_*/step*")):
            with np.load(folder/"model.npz") as data:raw=dict(data)
            info=read(folder/"result.json")
            for name in ("start","primal","selected_primal"):
                file=folder/(name+".npz")
                if not file.exists():continue
                with np.load(file) as data:primal=data["primal"]
                record=primal_audit(raw,primal)
                expected=info["primal_audit"]["feasible"] if name=="primal" else True
                valid=record["feasible"]==expected
                target=info["objective"] if name=="primal" else info.get("selected_upper") if name=="selected_primal" else None
                if target is not None:
                    valid=valid and bool(np.isclose(record["objective_recomputed"],target,atol=1e-8,rtol=1e-12))
                primal_checks.append(dict(job_id=job,file=file.relative_to(output).as_posix(),passed=bool(valid),**record))
    if independent_lp:
        chosen=read(destination/"chosen_global.json")
        for job in protocol["jobs"]:
            directory=output/"jobs"/job;payload=read(directory/"payload.json")
            if payload["condition"]!="reference" or payload["N"] not in (24,288) or not (directory/"result.json").exists():continue
            result=read(directory/"result.json");obs=observed(payload["train"],payload["N"])
            for method,value in chosen.items():
                cfg=dict(value["config"])
                if method!="NJ":cfg["resolution"]=0.
                point=select(result,method,cfg)
                p,q,y=arrays(obs,cfg["preprocess"])
                n=y.size;family=point["supports"]
                columns=[]
                for source in (p,q):
                    for support in family:
                        feature=np.zeros_like(y);feature[:,support]=source[:,support].sum(axis=1)[:,None]
                        columns.append(feature.ravel())
                features=csr_matrix(np.column_stack(columns))
                absolute=eye(n,format="csr")
                reference=linprog(np.r_[np.zeros(2*len(family)),np.ones(n)/n],
                    A_ub=vstack((hstack((features,-absolute)),hstack((-features,-absolute))),format="csr"),
                    b_ub=np.r_[y.ravel(),-y.ravel()],bounds=[(0.,2.)]*(2*len(family))+[(0.,None)]*n,
                    method="highs",options=dict(time_limit=30,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
                independent.append(dict(job_id=job,method=method,passed=bool(reference.success and abs(reference.fun-point["objective"])<=1e-8),
                    status=int(reference.status),independent_objective=float(reference.fun) if reference.fun is not None else None,
                    saved_objective=point["objective"]))
    summary=dict(data_checks=len(checks),LP_solve_records=sum(r["count"] for r in lp_summaries),
        LP_summary_checks=len(lp_summaries),physical_model_points=len(point_checks),free_primal_vectors=len(primal_checks),
        independent_LP_checks=len(independent),all_passed=all(r["passed"] for group in (checks,lp_summaries,point_checks,primal_checks,independent) for r in group))
    write(destination/"audit_summary.json",summary)
    write(destination/"audit_details.json",dict(data=checks,LP=lp_summaries,points=point_checks,free_primal=primal_checks,independent_LP=independent))
    print(summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");p.add_argument("destination");p.add_argument("--partial",action="store_true");p.add_argument("--independent-lp",action="store_true");a=p.parse_args()
    audit(a.output,a.destination,partial=a.partial,independent_lp=a.independent_lp)
