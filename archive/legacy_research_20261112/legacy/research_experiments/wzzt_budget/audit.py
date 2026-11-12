from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,eye,hstack,vstack
from scipy.optimize import linprog
from research_experiments.wzzt_systematic.common import read,write,sha,observed
from research_experiments.wzzt_retune.engine import compatible


def independent_lp(point,obs):
    p,q,v,root=obs;y=root.to_numpy()[:,None]**2-v.to_numpy()**2
    columns=[]
    for source in (p.to_numpy(),q.to_numpy()):
        for support in point["supports"]:
            f=np.zeros_like(y);f[:,support]=source[:,support].sum(axis=1)[:,None];columns.append(f.ravel())
    features=csr_matrix(np.column_stack(columns));absolute=eye(y.size,format="csr");k=len(columns)
    r=linprog(np.r_[np.zeros(k),np.ones(y.size)/y.size],
        A_ub=vstack((hstack((features,-absolute)),hstack((-features,-absolute))),format="csr"),
        b_ub=np.r_[y.ravel(),-y.ravel()],bounds=[(0,2)]*k+[(0,None)]*y.size,method="highs",
        options=dict(time_limit=30,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    return dict(passed=bool(r.success and abs(r.fun-point["objective"])<1e-8),status=int(r.status),
                reference_objective=float(r.fun) if r.fun is not None else None,saved_objective=point["objective"])


def audit(output,destination):
    output=Path(output).resolve();destination=Path(destination).resolve();m=read(output/"manifest.json");protocol=read(output/"protocol.json")
    checks=[dict(check="frozen producer, payloads, observed inputs and parent results",passed=all(sha(output/p)==h for p,h in m["local"].items()) and all(sha(Path(p))==h for p,h in m["external"].items()))]
    physical=[];LP=[];references=[];decisions=0
    for job in protocol["jobs"]:
        d=output/"jobs"/job
        if not (d/"result.json").exists():checks.append(dict(check=job+" returned",passed=False));continue
        result=read(d/"result.json");payload=read(d/"payload.json")
        obs=observed(payload["train"],payload["N"]);validation=observed(payload["validation"])
        checks.append(dict(check=job+" identical initial LP optimum",passed=result["long_start_objective_error"]<1e-8 and result["truth_used_by_worker"] is False))
        LP.append(dict(job_id=job,count=len(result["LP_solves"]),passed=all(r["status"]==2 and
            max(r["max_constraint_violation"],r["max_bound_violation"])<1e-7 and abs(r["physical"]-r["objective"])<1e-8 for r in result["LP_solves"])))
        points=list(result["models"].values())+[p for group in result["candidates"].values() for p in group]
        for name,trajectory in result["searches"].items():
            points.extend(trajectory["path"])
            for row in trajectory["decisions"]:
                decisions+=1
                best=min((c["objective"] for c in row["candidates"]),default=row["current"])
                gain=row["current"]-best if row["candidates"] else 0.
                checks.append(dict(check=job+"/"+name+" gain",passed=abs(row["gain"]-gain)<1e-8))
            if trajectory["reason"]=="pool_local_optimum":checks.append(dict(check=job+"/"+name+" full sweep",passed=trajectory["decisions"][-1]["full_sweep"]))
        seen=set()
        for p in points:
            key=str((p["supports"],p["r"],p["x"]))
            if key in seen:continue
            seen.add(key);family=[tuple(s) for s in p["supports"]]
            legal=all((i,) in family for i in range(len(result["labels"]))) and len(set(family))==len(family) and all(compatible(family[:i],s) for i,s in enumerate(family))
            bounded=all(-1e-9<=w<=2+1e-9 for w in p["r"]+p["x"])
            errors=[]
            for data,name in ((obs,"objective"),(validation,"validation_mae")):
                if name not in p:continue
                P,Q,V,root=data;y=root.to_numpy()[:,None]**2-V.to_numpy()**2;prediction=np.zeros_like(y)
                for s,r,x in zip(p["supports"],p["r"],p["x"],strict=True):
                    v=r*P.to_numpy()[:,s].sum(axis=1)+x*Q.to_numpy()[:,s].sum(axis=1);prediction[:,s]+=v[:,None]
                errors.append(abs(float(np.mean(abs(prediction-y)))-p[name]))
            physical.append(dict(job_id=job,passed=legal and bounded and max(errors)<1e-8,max_score_error=max(errors)))
        if payload["condition"]=="reference" and payload["N"] in (24,288):
            for name in ("long75","multistart_0"):
                if name in result["searches"]:references.append(dict(job_id=job,search=name,**independent_lp(result["searches"][name]["path"][-1],obs)))
    frame=pd.read_csv(destination/"per_job.csv")
    old=pd.read_csv(Path(__file__).resolve().parents[2]/"docs/wzzt_retune_20261028/per_job.csv")
    comparison=frame[frame.variant=="baseline25"].merge(old[old.method=="RNJ_WZZT"][["job_id","clade_f1"]],on="job_id",validate="one_to_one",suffixes=("","_published"))
    checks.append(dict(check="published baseline reproduced for all 80 inputs",passed=len(comparison)==80 and np.allclose(comparison.clade_f1,comparison.clade_f1_published)))
    checks.append(dict(check="all final models retained",passed=len(frame)==560 and frame.model_available.all() and not frame.duplicated(["job_id","variant"]).any()))
    summary=dict(all_passed=all(r["passed"] for group in (checks,physical,LP,references) for r in group),checks=len(checks),
        physical_models=len(physical),LP_solves=sum(r["count"] for r in LP),independent_LP_checks=len(references),search_decisions=decisions,
        dataset_rows=frame.job_id.nunique(),selected_models=len(frame))
    write(destination/"audit_summary.json",summary);write(destination/"audit_details.json",dict(checks=checks,physical=physical,LP=LP,independent=references))
    print(summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");p.add_argument("destination");a=p.parse_args();audit(a.output,a.destination)
