"""Independent raw-vector checks and post-hoc physical scores for the pilot.

Physical truth is first read here, after all candidate generation/AC selections.
This program never changes or selects the archived candidate paths.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from .profile_gaps import ROOT, RUN, RUNTIME, JOBS, read, write, sha


def audit_raw(raw, primal):
    activity = np.zeros(len(raw["constraint_lower"]))
    # Independent sparse matrix evaluation, separate from the producer's bincount.
    from scipy.sparse import coo_matrix
    matrix = coo_matrix((raw["constraint_data"],(raw["constraint_row_index"],raw["constraint_col_index"])),
        shape=(len(activity),len(primal))).tocsr()
    activity = matrix @ primal
    assert np.isfinite(primal).all()
    assert np.all(activity >= raw["constraint_lower"]-1e-7)
    assert np.all(activity <= raw["constraint_upper"]+1e-7)
    assert np.all(primal >= raw["lower_bounds"]-1e-7)
    assert np.all(primal <= raw["upper_bounds"]+1e-7)
    integer = raw["integrality"] != 0
    assert np.max(abs(primal[integer]-np.rint(primal[integer])),initial=0) <= 1e-7
    return float(raw["objective_coefficients"]@primal)


def csv_write(path, rows):
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT/"outputs/wzzt_solver_tuning_20261027")
    args=parser.parse_args(); output=args.root/"reports"
    output.mkdir(exist_ok=True)
    sys.path.insert(0,str(RUNTIME))
    from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth,clade_metrics,relative_error
    from .ac_screen import select_ac_one_se, atom_network
    from .analyze_profiles import observations
    from rnj_wzzt.models.ac_powerflow import _oriented_tree, _solve_ac_power_flow_snapshot_oriented
    checks=[]; profile_rows=[]; path_rows=[]; outcome_rows=[]
    def check(name,value):
        checks.append(dict(check=name,passed=bool(value)))
        if not value:raise AssertionError(name)
    profile=args.root/"gap_profiles";protocol=read(profile/"protocol.json")
    check("all nine gap solves returned",read(profile/"completion.json")["returned"]==9)
    for model in protocol["models"]:
        raw=dict(np.load(ROOT/model["raw"]))
        for gap in protocol["gaps"]:
            p=profile/f"{model['job_id']}__gap_{gap:g}";result=read(p/"result.json")
            primal=np.load(p/"primal.npz")["primal"]
            objective=audit_raw(raw,primal)
            check(f"{p.name}: raw objective",abs(objective-result["objective"])<1e-10)
            check(f"{p.name}: bounds bracket archived optimum",result["bound"]<=model["historical_objective"]+1e-10
                and model["historical_objective"]<=result["objective"]+1e-10)
            profile_rows.append({k:result[k] for k in ("job_id","requested_gap","runtime_seconds","raw_status",
                "objective","bound","gap","gap_target_met","objective_excess_over_historical","gain_lower","gain_upper")})
    for family in ("bounded_paths","refined_paths","single_pass_paths"):
        base=args.root/family;protocol=read(base/"protocol.json")
        check(f"{family}: all jobs returned",read(base/"completion.json")["returned"]==len(protocol["jobs"]))
        for name,expected in protocol["input_sha256"].items():
            path=ROOT/name
            if name.replace("\\", "/").startswith("research_experiments/wzzt_solver_tuning/"):
                frozen=base/"producer_source_snapshot"/path.name
                if frozen.exists():path=frozen
            check(f"{family}: hash {name}",sha(path)==expected)
        for job in protocol["jobs"]:
            directory=base/job; result=read(directory/"result.json")
            labels=read(RUN/"jobs"/job/"model.json")["terminal_labels"]
            observed=observations(job)
            truth=physical_truth(RUN/"evaluation_network"/job.split("__")[0],labels)
            calibration=read(RUN/"jobs"/job/"rx75_calibration.json")
            for attempt in sorted(directory.glob("attempt*/result.json")):
                info=read(attempt); raw=dict(np.load(attempt.parent/"model.npz"))
                if (attempt.parent/"primal.npz").exists():
                    objective=audit_raw(raw,np.load(attempt.parent/"primal.npz")["primal"])
                    check(f"{family}/{job}/{attempt.parent.name}: raw objective",abs(objective-info["objective"])<1e-10)
                if info["decision"]=="accept_bounded_improvement":
                    check(f"{family}/{job}/{attempt.parent.name}: >=2/3 gain guarantee",
                        info["gain_guaranteed"]>info["gain_tolerance"] and
                        info["gain_guaranteed"]>=2/3*info["gain_possible"]-1e-10)
                if info["decision"]=="stop_bound_proves_no_material_gain":
                    check(f"{family}/{job}/{attempt.parent.name}: no-gain bound",
                        info["gain_possible"]<=info["gain_tolerance"]+1e-12)
            for i,point in enumerate(result["path"]):
                r,x=np.asarray(point["r"]),np.asarray(point["x"])
                predicted=[]; active=[]; family_clades=[];matrices=[]
                for s,rv,xv in zip(point["supports"],r,x,strict=True):
                    labelled=[labels[j] for j in s]
                    family_clades.append(labelled)
                    if max(rv,xv)>1e-8:active.append(labelled)
                    if len(s)==1 or calibration["a"]*rv+calibration["b"]*xv+1e-10>=calibration["tau"]:
                        predicted.append(labelled)
                for values in (r,x):
                    matrix=np.zeros((len(labels),len(labels)))
                    for s,w in zip(point["supports"],values,strict=True):matrix[np.ix_(s,s)]+=w
                    matrices.append(matrix)
                ac=point["ac"]
                check(f"{family}/{job}/path{i}: objective reproduction",abs(point["physical_l1_objective"]-point["search_objective"])<1e-8)
                if ac["valid"]:
                    check(f"{family}/{job}/path{i}: AC balance",ac["max_active_balance_error"]<1e-8)
                    # Independent scalar kernel: reproduce all 96 voltage
                    # residuals and both P/Q balances from saved AC-fit weights.
                    net=atom_network(labels,point["supports"],ac["r_refit"],ac["x_refit"])
                    nodes,parent,children,impedances=_oriented_tree(net)
                    obs_p,obs_q,obs_v,obs_root=observed
                    prediction=[];max_balance=0.
                    for t in range(len(obs_p)):
                        p_values=obs_p.iloc[t].to_dict();q_values=obs_q.iloc[t].to_dict()
                        snap=_solve_ac_power_flow_snapshot_oriented(net,nodes,parent,children,impedances,
                            p_values,q_values,v_root=obs_root.iloc[t],max_iter=200,tol=1e-11)
                        assert snap.converged
                        prediction.append([abs(snap.bus_voltage[label]) for label in labels])
                        head=sum(v for edge,v in snap.branch_power_from.items() if edge[0]==net.root_bus)
                        load=sum(p_values.values())+1j*sum(q_values.values())
                        max_balance=max(max_balance,abs(head-load-sum(snap.branch_loss.values())))
                    mse=float(np.mean(((np.asarray(prediction)-obs_v.to_numpy())/obs_v.to_numpy())**2))
                    check(f"{family}/{job}/path{i}: independent scalar AC voltage",abs(mse-ac["relative_voltage_mse"])<1e-12)
                    check(f"{family}/{job}/path{i}: independent complex AC balance",max_balance<1e-8)
                path_rows.append(dict(family=family,job_id=job,path_index=i,atoms=len(r),
                    resolution_f1=clade_metrics(predicted,truth["clades"],labels)["clade_f1"],
                    active_f1=clade_metrics(active,truth["clades"],labels)["clade_f1"],
                    family_f1=clade_metrics(family_clades,truth["clades"],labels)["clade_f1"],
                    r_error_percent=100*relative_error(matrices[0],truth["R"]),
                    x_error_percent=100*relative_error(matrices[1],truth["X"]),
                    l1=point["search_objective"],ac_voltage_rmse=ac.get("relative_voltage_rmse"),
                    ac_refit_success=ac.get("refit",{}).get("success"),
                    ac_selected=i==result["ac_one_se_index"],last=i==result["last_index"]))
            check(f"{family}/{job}: AC selection reproducible",select_ac_one_se(result["path"])==result["ac_one_se_index"])
            last=next(r for r in path_rows if r["family"]==family and r["job_id"]==job and r["last"])
            chosen=next(r for r in path_rows if r["family"]==family and r["job_id"]==job and r["ac_selected"])
            outcome_rows.append(dict(family=family,job_id=job,returned=True,stop_reason=result["stop_reason"],
                runtime_seconds=result["runtime_seconds"],path_points=len(result["path"]),attempts=len(result["attempts"]),
                last_resolution_f1=last["resolution_f1"],ac_selected_index=chosen["path_index"],
                ac_selected_resolution_f1=chosen["resolution_f1"],last_ac_rmse=last["ac_voltage_rmse"],
                selected_ac_rmse=chosen["ac_voltage_rmse"],
                incomplete=result["stop_reason"]!="stop_bound_proves_no_material_gain"))
    csv_write(output/"gap_profiles.csv",profile_rows)
    csv_write(output/"path_points.csv",path_rows)
    csv_write(output/"outcomes.csv",outcome_rows)
    write(output/"independent_audit.json",dict(passed=True,check_count=len(checks),failed_count=0,checks=checks,
        scope="all pilot raw constraints/integrality/objectives; gain-bound decisions; AC balance; producer/input hashes; post-hoc truth scoring",
        independent_test_set=False,production_changed=False,
        audit_producer_sha256=sha(Path(__file__)),
        physical_evaluation_sha256=sha(ROOT/"research_experiments/rnj/audit_source_ac96_benchmark.py"),
        runtime_manifest_sha256=sha(RUNTIME/"runtime_manifest.json")))
    evidence={}
    for path in args.root.rglob("*"):
        if path.is_file() and output not in path.parents:
            evidence[str(path.relative_to(ROOT))]=dict(sha256=sha(path),bytes=path.stat().st_size)
    write(output/"raw_manifest.json",evidence)
    print(json.dumps(dict(passed=True,checks=len(checks),profile_rows=len(profile_rows),path_rows=len(path_rows),outcomes=outcome_rows),ensure_ascii=False))


if __name__=="__main__":
    main()
