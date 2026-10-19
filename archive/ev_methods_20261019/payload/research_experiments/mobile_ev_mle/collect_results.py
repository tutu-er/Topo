"""Aggregate completed frozen cohorts and export compact reviewable evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .run_study import dump,calibrate,summarize

DIAG=("initial_loss","local_shared_rank","shared_parameter_count","local_identifiability_full_column_rank",
      "active_bound_count","restart_loss_spread","ac_equation_residual",
      "power_balance_residual","min_voltage","max_voltage","global_optimality_certified",
      "seconds","restarts","failure_kind","physical_voltage_feasible")


def compact_fit(row):
    result={k:v for k,v in row.items() if k!="diagnostics"}
    result["diagnostics"]={k:row["diagnostics"].get(k) for k in DIAG}
    return result


def compact_trial(t,role):
    result={k:t[k] for k in ("seed","case","n","seconds","selection_uses_truth",
                             "selection_uses_future_noise","historical_failures")}
    result["role"]=role
    result["true_family"]=t["truth"]["supports"]
    result["truth_r_pu"]=t["truth"]["r_pu"]
    result["truth_x_pu"]=t["truth"]["x_pu"]
    result["historical_fits"]=[compact_fit(r) for r in t["historical_fits"]]
    result["passive_diagnostics"]=t["proposal"]["passive"]["diagnostics"]
    result["pre_omission_proposal_bank"]=t["proposal"]["bank"]
    result["fitted_candidate_bank"]=[dict(name=f["name"],supports=f["supports"])
                                    for f in t["historical_fits"]]
    design=t["design"]
    result["design"]={k:v for k,v in design.items() if k!="scores"}
    result["design"]["scores"]=[]
    for row in design.get("scores",[]):
        s={k:v for k,v in row.items() if k!="competitor_fits"}
        s["competitor_fits"]=[]
        for f in row.get("competitor_fits",[]):
            s["competitor_fits"].append({k:v for k,v in f.items() if k!="diagnostics"})
        result["design"]["scores"].append(s)
    result["records"]=[]
    for r in t["records"]:
        rr={k:v for k,v in r.items() if k!="fits"}
        rr["fits"]=[compact_fit(f) for f in r.get("fits",[])]
        retained=r.get("retained_candidates",[])
        families=[set(tuple(s) for s in f["supports"] if 1<len(s)<t["n"])
                  for f in r.get("fits",[]) if f["name"] in retained]
        rr["common_clades"]=list(map(list,sorted(set.intersection(*families)))) if families else []
        result["records"].append(rr)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    source=Path(args.input);target=Path(args.output)
    config=json.loads((source/"config.json").read_text(encoding="utf-8"))
    calibration=[];evaluation=[]
    manifest=[]
    for role,seeds,collection in (
        ("calibration",config["calibration_seeds"],calibration),
        ("evaluation",config["evaluation_seeds"],evaluation)):
        for case in ("strong","weak"):
            for seed in seeds:
                path=source/f"{case}_{seed}.json"
                t=json.loads(path.read_text(encoding="utf-8"))
                if t["seed"]!=seed or t["case"]!=case or len(t["records"])!=5:
                    raise RuntimeError("Inconsistent trial: "+str(path))
                collection.append(t)
                npz=source/f"{case}_{seed}_observations.npz"
                manifest.append(dict(role=role,case=case,seed=seed,
                    record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    observations_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
                    record_bytes=path.stat().st_size,observations_bytes=npz.stat().st_size))
    thresholds=calibrate(calibration,.1)
    rows=summarize(evaluation,thresholds)
    all_trials=calibration+evaluation
    summary=dict(config=config,thresholds=thresholds,summary=rows,
        completed_trials=len(all_trials),method_records=5*len(all_trials),
        calibration_total_seconds=sum(t["seconds"] for t in calibration),
        evaluation_total_seconds=sum(t["seconds"] for t in evaluation),
        design_mean_seconds=float(np.mean([t["design"]["elapsed_seconds"] for t in evaluation])),
        failed_inner_actions=sum(t["design"].get("failed_action_count",0) for t in all_trials),
        negative_increment_actions=sum(any("historical_baseline_instability" in f for f in s["failures"])
                                      for t in all_trials for s in t["design"].get("scores",[])),
        formal_main_source_commit="434b2d3",
        orchestration_commit="f741aac",
        optimization="same bounded sparse local solver in every statistical trial",
        statistical_scope="declared synthetic mixture and paired-complete pipeline; no topology probabilities")
    dump(source/"thresholds.json",thresholds)
    dump(source/"summary.json",summary)
    dump(target/"formal_summary.json",summary)
    dump(target/"formal_records.json",
         [compact_trial(t,"calibration") for t in calibration]+
         [compact_trial(t,"evaluation") for t in evaluation])
    dump(target/"formal_artifact_manifest.json",manifest)
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
