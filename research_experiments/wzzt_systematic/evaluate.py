"""Post-fit scoring and independent evidence audit; never called by workers."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

from .common import (ac_diagnostic, indices, linear_loss, matrices, observed, read, sha, write)


def audit_data(output):
    import numpy as np
    import pandas as pd
    protocol=read(output/"protocol.json")
    checks=[]
    def require(name, condition):
        checks.append(dict(check=name, passed=bool(condition)))
    manifest=read(output/"frozen_manifest.json")
    require("all frozen source and data hashes", all(sha(output/p)==h for p,h in manifest.items()))
    require("240 unique planned jobs",len(protocol["jobs"])==len(set(protocol["jobs"]))==240)
    require("nested samples",all(set(indices(a))<set(indices(b)) for a,b in zip(protocol["sample_sizes"][:-1],protocol["sample_sizes"][1:])))
    for job in protocol["jobs"]:
        payload=read(output/"jobs"/job/"payload.json")
        require(job+" observed training path only", not any("test" in str(v) or "truth" in str(v) for k,v in payload.items() if k!="solve"))
    for case in protocol["cases"]:
        base=output/"data"/case
        reference=np.load(base/"reference/train_truth.npz")
        ideal=np.load(base/"ideal_ac/train_truth.npz")
        heavy=np.load(base/"higher_load/train_truth.npz")
        low=np.load(base/"low_xr/train_truth.npz")
        noisy=np.load(base/"high_voltage_noise/train_truth.npz")
        for key in ("P","Q","V","root"):
            require(case+" ideal/reference same physical "+key,np.array_equal(reference[key],ideal[key]))
            require(case+" high-noise/reference same physical "+key,np.array_equal(reference[key],noisy[key]))
        for key in ("P","Q"):
            require(case+" load stress "+key,np.allclose(heavy[key],1.5*reference[key],rtol=1e-14,atol=0.))
            require(case+" low X/R injection isolated "+key,np.array_equal(low[key],reference[key]))
        branches=pd.read_csv(base/"low_xr/truth_network/branches.csv")
        original=pd.read_csv(base/"reference/truth_network/branches.csv")
        require(case+" low X/R unchanged resistance",np.array_equal(branches.r_ohm,original.r_ohm))
        target=np.where(branches.branch_type.str.contains("service",case=False),.06,.30)
        require(case+" low X/R specified reactance",np.allclose(branches.x_ohm,branches.r_ohm*target,rtol=1e-14))
        for condition in protocol["conditions"]:
            directory=base/condition
            train=np.load(directory/"train_observed.npz")
            test=np.load(directory/"test_observed.npz")
            require(case+condition+" independent train/test injections",not np.array_equal(train["P"],test["P"]))
            require(case+condition+" all physical AC points converged",all(x["converged"]==96 for x in read(directory/"diagnostics.json")))
            if condition=="ideal_ac":
                for key in ("P","Q","V","root"):
                    require(case+" ideal exact meter "+key,np.array_equal(train[key],ideal[key]))
        for handle in (reference,ideal,heavy,low,noisy):
            handle.close()
    return dict(passed=all(c["passed"] for c in checks),checks=checks,check_count=len(checks))


def evaluate(output, destination, *, partial=False):
    import numpy as np
    import pandas as pd
    output=Path(output).resolve()
    destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(output/"source"))
    from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth, clade_metrics, relative_error
    from research_experiments.rnj.rx75_edge_floor import classify_rx75_atoms
    from research_experiments.wzzt_solver_tuning.profile_gaps import primal_audit
    from research_experiments.wzzt_solver_tuning.policy import extension_decision
    protocol=read(output/"protocol.json")
    data_audit=audit_data(output)
    write(destination/"data_audit.json",data_audit)
    rows=[]
    checks=[]
    path_checks=[]
    test_cache=output/"postfit_test"
    test_cache.mkdir(exist_ok=True)
    for job in protocol["jobs"]:
        directory=output/"jobs"/job
        payload=read(directory/"payload.json")
        ended=any((directory/name).exists() for name in ("result.json","error.json","hard_timeout.json"))
        if partial and not ended:
            continue
        row=dict(job_id=job,case=payload["case"],condition=payload["condition"],N=payload["N"],method=payload["method"],
                 returned=(directory/"result.json").exists(),status="missing", model_available=False)
        model=read(directory/"result.json") if row["returned"] else (read(directory/"checkpoint.json") if (directory/"checkpoint.json").exists() else None)
        if not row["returned"]:
            row["status"]="hard_timeout" if (directory/"hard_timeout.json").exists() else "failed" if (directory/"error.json").exists() else "missing"
            if (directory/"error.json").exists():
                row["error"]=read(directory/"error.json")["error"]
        if not model or not model.get("path"):
            rows.append(row)
            continue
        selected=model.get("selected_index")
        selection_fallback=selected is None
        selected=len(model["path"])-1 if selected is None else selected
        point=model["path"][selected]
        labels=model["labels"]
        data=output/"data"/payload["case"]/payload["condition"]
        truth=physical_truth(data/"truth_network",labels)
        test=observed(data/"test_observed.npz")
        true_test=observed(data/"test_truth.npz")
        train=observed(data/"train_observed.npz",payload["N"])
        p,q,v,root=train
        target_y=root.to_numpy()[:,None]**2-v.to_numpy()**2
        epoch_paths=[b["path"] for b in model.get("bound_expansions",[])]+[model["path"]]
        for epoch,path in enumerate(epoch_paths):
            for point_index,pp in enumerate(path):
                prediction=np.zeros_like(target_y)
                for support,rr,xx in zip(pp["supports"],pp["r"],pp["x"],strict=True):
                    feature=rr*p.iloc[:,support].sum(axis=1).to_numpy()+xx*q.iloc[:,support].sum(axis=1).to_numpy()
                    prediction[:,support]+=feature[:,None]
                objective=float(np.mean(abs(prediction-target_y)))
                family=[set(s) for s in pp["supports"]]
                valid=abs(objective-pp["search_objective"])<=1e-8
                valid=valid and all(a<=b or b<=a or a.isdisjoint(b) for i,a in enumerate(family) for b in family[i+1:])
                if point_index:
                    valid=valid and objective<=path[point_index-1]["search_objective"]+1e-8
                path_checks.append(dict(job_id=job,epoch=epoch,point=point_index,passed=bool(valid),physical_l1=objective))
        row.update(model_available=True,stop_reason=model.get("stop_reason"),
            search_seconds=model.get("search_seconds"),elapsed_seconds=model.get("elapsed_seconds"),
            bootstrap_seconds=model.get("bootstrap_seconds",0.),groups=len(model.get("groups",[])),
            group_members=sum(map(len,model.get("groups",[]))),path_points=len(model["path"]),
            selected_index=selected,selection_fallback=selection_fallback,
            bound_expansions=len(model.get("bound_expansions",[])),final_bound=model.get("final_bound"))
        if row["returned"]:
            row["status"]=model["status"]
        def score(p, native=False):
            r,x=matrices(len(labels),p["supports"],p["r"],p["x"])
            classification=classify_rx75_atoms(p["supports"],p["r"],p["x"],model["calibration"])
            effective=[{labels[i] for i in p["supports"][j]} for j in classification["retained_indices"]]
            family=[{labels[i] for i in s} for s in p["supports"]]
            active=[{labels[i] for i in s} for s,rr,xx in zip(p["supports"],p["r"],p["x"],strict=True) if max(rr,xx)>1e-9]
            if native:
                effective=model["NJ_clades"]
                r,x=np.asarray(model["NJ_R"]),np.asarray(model["NJ_X"])
            return dict(**clade_metrics(effective,truth["clades"],labels),
                        family_f1=clade_metrics(family,truth["clades"],labels)["clade_f1"],
                        active_f1=clade_metrics(active,truth["clades"],labels)["clade_f1"],
                        R_error=relative_error(r,truth["R"]),X_error=relative_error(x,truth["X"]),
                        linear_test_mae=linear_loss(test,r,x)["mae"])
        row.update(score(point,payload["method"]=="NJ"))
        last=score(model["path"][-1],payload["method"]=="NJ")
        row["last_clade_f1"]=last["clade_f1"]
        train_ac=point.get("ac",{})
        row["train_ac_valid"]=train_ac.get("valid",False)
        row["train_ac_refit_success"]=train_ac.get("refit",{}).get("success")
        row["train_ac_rmse"]=train_ac.get("relative_voltage_rmse")
        cache=test_cache/f"{job}.json"
        result_digest=sha(directory/"result.json") if row["returned"] else sha(directory/"checkpoint.json")
        if cache.exists() and read(cache).get("result_sha256")==result_digest:
            test_ac=read(cache)
        elif train_ac.get("valid"):
            test_ac=dict(result_sha256=result_digest,
                measured=ac_diagnostic(labels,point["supports"],train_ac["r_refit"],train_ac["x_refit"],test,refit=False),
                physical=ac_diagnostic(labels,point["supports"],train_ac["r_refit"],train_ac["x_refit"],true_test,refit=False))
            write(cache,test_ac)
        else:
            test_ac=dict(measured={"valid":False},physical={"valid":False})
        row.update(test_ac_valid=test_ac["measured"]["valid"],test_ac_rmse=test_ac["measured"].get("relative_voltage_rmse"),
                   test_physical_ac_rmse=test_ac["physical"].get("relative_voltage_rmse"))
        for target in sorted(directory.glob("epoch*_step*")):
            info=read(target/"result.json")
            with np.load(target/"model.npz") as saved:
                raw=dict(saved)
            primal=np.load(target/"primal.npz")["primal"] if (target/"primal.npz").exists() else None
            audit=primal_audit(raw,primal)
            saved_feasible=info["primal_audit"]["feasible"]
            passed=audit["feasible"]==saved_feasible
            if primal is not None and info["objective"] is not None:
                passed=passed and abs(audit["objective_recomputed"]-info["objective"])<=1e-8
            # Re-evaluate the saved decision against the original current loss.
            epoch_path=next((b["path"] for b in model.get("bound_expansions",[]) if b["epoch"]==info["epoch"]),model["path"])
            current=epoch_path[info["iteration"]]["search_objective"]
            # Original round-1 producer overwrote only the log's "bound" key
            # with the coefficient box. Search used the correct solver bound.
            # Recover the exact lower bound from the independently stored
            # possible-gain interval, and retain an explicit repair receipt.
            legacy="coefficient_bound" not in info
            lower=(current-info["gain_possible"] if info["gain_possible"] is not None else None) if legacy else info["bound"]
            passed=passed and abs(info["gain_tolerance"]-(protocol["solve"]["absolute_gain"]+protocol["solve"]["relative_gain"]*current))<=1e-12
            decision=extension_decision(current,info["objective"],lower,feasible=audit["feasible"],
                relative_gain=protocol["solve"]["relative_gain"],uncertainty_fraction=protocol["solve"]["uncertainty_fraction"])
            passed=passed and decision["decision"]==info["decision"]
            checks.append(dict(job_id=job,attempt=target.name,passed=bool(passed),audit=audit,
                               metadata_bound_recovered=legacy,solver_lower_bound=lower,current_objective=current))
        rows.append(row)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with (destination/"per_job.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fields)
        writer.writeheader()
        writer.writerows(rows)
    frame=pd.DataFrame(rows)
    aggregates=[]
    for keys,group in frame.groupby(["condition","N","method"],sort=False):
        record=dict(zip(("condition","N","method"),keys),planned=len(group),returned=int(group.returned.sum()),
                    complete=int((group.status=="complete").sum()),models=int(group.model_available.sum()))
        for metric in ("clade_f1","last_clade_f1","R_error","X_error","test_ac_rmse","test_physical_ac_rmse","elapsed_seconds","search_seconds"):
            if metric in group:
                record[metric]=float(pd.to_numeric(group[metric],errors="coerce").mean())
        if "clade_f1" in group:
            record["clade_f1_all_planned"]=float(group.clade_f1.fillna(0).mean())
        aggregates.append(record)
    pd.DataFrame(aggregates).to_csv(destination/"aggregate.csv",index=False,encoding="utf-8-sig")
    completed=len(frame)
    overview=dict(planned=240,evaluated=completed,returned=int(frame.returned.sum()),
        complete=int((frame.status=="complete").sum()),models=int(frame.model_available.sum()),
        statuses=frame.status.value_counts().to_dict(),
        data_audit_passed=data_audit["passed"],raw_attempts=len(checks),
        raw_audit_passed=all(c["passed"] for c in checks),path_points=len(path_checks),
        path_audit_passed=all(c["passed"] for c in path_checks),scope=protocol["scope"])
    write(destination/"overview.json",overview)
    write(destination/"solver_audit.json",dict(passed=all(c["passed"] for c in checks),checks=checks,check_count=len(checks)))
    write(destination/"path_audit.json",dict(passed=all(c["passed"] for c in path_checks),checks=path_checks,check_count=len(path_checks)))
    print(overview,flush=True)
    return frame


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("destination",type=Path)
    parser.add_argument("--partial",action="store_true")
    args=parser.parse_args()
    evaluate(args.output,args.destination,partial=args.partial)
