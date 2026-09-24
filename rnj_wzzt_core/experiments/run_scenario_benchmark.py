"""Audit explicit scenario suites without bootstrap, MILP, or hyperparameter search.

Physical days always contain 96 points. Fitting uses nested 8/16/32/96-point
subsets; the separate test replicate always retains all 96 points per scenario.
The unchanged estimator sets latent common mode to zero: unobserved-root results
are deliberate model-mismatch diagnostics, not a proposed estimator for that case.
"""
from __future__ import annotations
import os
for option in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[option] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
sys.dont_write_bytecode = True
CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))
import numpy as np
import pandas as pd
from rnj_wzzt.scenario.settings import SCENARIO_SUITES, ROOT_OBSERVATIONS
from rnj_wzzt.scenario.simulation import _simulate_pool, _terminal_buses
from rnj_wzzt.estimation.multiscenario import fit_projected_sensitivity, preprocess_scenarios
from rnj_wzzt.estimation.preprocessing import RECIPE
from rnj_wzzt.models.lin_distflow import build_reduced_sensitivity_matrices
from rnj_wzzt.graph.sensitivity_geometry import sensitivity_geometry
from rnj_wzzt.graph.rooted_neighbor_joining import rooted_neighbor_joining
from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
from rnj_wzzt.data.paper_style_case_bank import CASE_BUILDERS
from rnj_wzzt.reporting import _family_score, _truth_nontrivial_clades


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def matrix_error(a,b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def run_job(job):
    destination=Path(job["output"])/job["suite"]/job["case"]/f"repeat_{job['repeat']:02d}"/job["observation"]
    destination.mkdir(parents=True, exist_ok=True)
    previous=destination/"result.json"
    if previous.exists():
        old=json.loads(previous.read_text(encoding="utf-8"))
        if old["job"]!=job: raise ValueError(f"Different cached configuration: {destination}")
        return old
    start=perf_counter()
    all_sets={}
    arrays={}
    physical=[]
    for offset,split in enumerate(("train","test")):
        net,raw=_simulate_pool(job["case"],96,2*job["repeat"]+offset,3,job["pq_noise_rel"],job["voltage_noise_rel"],scenario_suite=job["suite"],root_observation=job["observation"],root_meter_noise_rel=job["root_meter_noise_rel"],root_sigma=job["root_sigma"],impedance_scale=job["impedance_scale"])
        all_sets[split]=raw
        for index,s in enumerate(raw):
            prefix=f"{split}_{index}_"
            for key in ("P_terminal","Q_terminal","V_terminal","drop_target","P_true","Q_true","V_terminal_true","root_voltage_true"):
                if key in s: arrays[prefix+key]=np.asarray(s[key])
            root=np.asarray(s["root_voltage_true"])
            v=np.asarray(s["V_terminal_true"])
            observed=s.get("root_voltage")
            if observed is not None: arrays[prefix+"root_voltage_observed"]=np.asarray(observed)
            drop=root[:,None]**2-v*v
            measured=np.asarray(s["drop_target"])
            # Target disturbance includes unknown physical root when unobserved.
            disturbance=measured-drop
            signal=float(np.sqrt(np.mean((drop-drop.mean(axis=0))**2)))
            raw_error=float(np.sqrt(np.mean(disturbance**2)))
            error=float(np.sqrt(np.mean((disturbance-disturbance.mean(axis=0))**2)))
            row={"case":job["case"],"suite":job["suite"],"repeat":job["repeat"],"observation":job["observation"],"split":split,"scenario":index,"name":s["name"],"impedance_scale":net.metadata["impedance_scale"],"root_std_pu":float(np.std(root,ddof=1)),"root_min_pu":float(root.min()),"root_max_pu":float(root.max()),"root_sq_std":float(np.std(root*root,ddof=1)),"root_lag1":float(np.corrcoef(root[:-1],root[1:])[0,1]) if np.std(root)>0 else None,"terminal_v_min_pu":float(v.min()),"terminal_v_max_pu":float(v.max()),"terminal_v_std_rms":float(np.sqrt(np.mean(np.var(v,axis=0)))),"dynamic_drop_rms":signal,"raw_target_disturbance_rms":raw_error,"dynamic_target_disturbance_rms":error,"signal_to_disturbance_amplitude":signal/max(error,1e-15),"root_meter_error_rms":None if observed is None else float(np.sqrt(np.mean((np.asarray(observed)-root)**2))),"voltage_outside_0p93_1p07_count":int(np.sum((v<.93)|(v>1.07))),"ac_converged":True}
            physical.append(row)
    labels=_terminal_buses(net);n=len(labels)
    truth=np.asarray(build_reduced_sensitivity_matrices(net,labels,voltage_model="squared-voltage"))
    truth_clades=_truth_nontrivial_clades(net,labels)
    arrays.update(R_true=truth[0],X_true=truth[1],terminals=np.asarray(labels))
    np.savez_compressed(destination/"observations.npz",**arrays)
    rows=[]
    for count in job["sample_counts"]:
        indices=np.arange(0,96,96//count)
        selected=[]
        for s in all_sets["train"]:
            selected.append({"name":s["name"],**{k:s[k].iloc[indices].copy() for k in ("P_terminal","Q_terminal","drop_target")}})
        row={"case":job["case"],"suite":job["suite"],"repeat":job["repeat"],"observation":job["observation"],"samples_per_scenario":count,"total_training_rows":3*count,"test_rows":288,"terminal_count":n,"status":"failed","latent_common_mode_fitted":False}
        diag={}
        try:
            tick=perf_counter()
            r,x,r2,condition=fit_projected_sensitivity(preprocess_scenarios(selected,RECIPE),diagnostics=diag)
            row["fit_seconds"]=perf_counter()-tick
            b=np.vstack([np.asarray(s["drop_target"].mean())-np.asarray(s["P_terminal"].mean())@r.T-np.asarray(s["Q_terminal"].mean())@x.T for s in selected])
            trace={"R":r,"X":x,"intercepts":b,"indices":indices}
            for split,data in (("train",selected),("test",all_sets["test"])):
                residual=np.vstack([np.asarray(s["drop_target"])-np.asarray(s["P_terminal"])@r.T-np.asarray(s["Q_terminal"])@x.T-b[j] for j,s in enumerate(data)])
                row[f"{split}_rmse"]=float(np.sqrt(np.mean(residual**2)))
                trace[f"{split}_residual"]=residual
            row.update(status="success",r_relative_error=matrix_error(r,truth[0]),x_relative_error=matrix_error(x,truth[1]),train_r2=r2,condition=condition)
            for mode,label in (("R","r"),("X","x"),("RX_75R_25X","rx75")):
                geo=sensitivity_geometry(r,x,mode)
                tau=.16*max(float(np.median(geo.root_depths)),1e-12)
                tree=rooted_neighbor_joining(geo.shared_paths,geo.root_depths,labels,int(net.root_bus),group_tolerance=tau)
                clades=rooted_clades(tree.edges,int(net.root_bus),labels)
                row[f"rnj_{label}_f1"]=_family_score(clades,truth_clades)["nontrivial_support_f1"]
                row[f"rnj_{label}_exact"]=int(clades==truth_clades)
                trace[f"rnj_{label}_edges"]=np.asarray(tree.edges)
            np.savez_compressed(destination/f"fit_N{count}.npz",**trace)
        except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
            row.update(status="failed",error_type=type(exc).__name__,error=str(exc))
        write_json(destination/f"fit_N{count}_diagnostics.json",diag)
        rows.append(row)
    result={"job":job,"physical":physical,"rows":rows,"seconds":perf_counter()-start,"scenario_settings":all_sets["train"][0]["scenario_settings"]}
    write_json(previous,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("outputs/scenario_benchmark"))
    parser.add_argument("--cases",nargs="+",choices=tuple(CASE_BUILDERS),default=list(CASE_BUILDERS))
    parser.add_argument("--suites",nargs="+",choices=tuple(SCENARIO_SUITES),default=list(SCENARIO_SUITES))
    parser.add_argument("--root-observations",nargs="+",choices=tuple(ROOT_OBSERVATIONS),default=list(ROOT_OBSERVATIONS))
    parser.add_argument("--samples",nargs="+",type=int,default=[8,16,32,96])
    parser.add_argument("--repeats",type=int,default=1)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--pq-noise-rel",type=float,default=.005)
    parser.add_argument("--voltage-noise-rel",type=float,default=.0002)
    parser.add_argument("--root-meter-noise-rel",type=float,default=.0002)
    parser.add_argument("--root-sigma",type=float)
    parser.add_argument("--impedance-scale",type=float)
    args=parser.parse_args()
    if args.repeats<1 or args.workers<1 or any(k<4 or 96%k for k in args.samples): parser.error("positive repeats/workers and sample counts >=4 dividing 96 are required")
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),*sorted((CORE/"rnj_wzzt").rglob("*.py"))]
    manifest={str(p.relative_to(CORE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    mp=output/"source_manifest.json"
    if mp.exists() and json.loads(mp.read_text(encoding="utf-8"))!=manifest: raise ValueError("Source changed; choose a fresh output directory")
    config={**vars(args),"output":str(output),"estimator":"current ordered squared-loss regression, latent c=0","split":"train replicate=2*k; test replicate=2*k+1","test_intercepts":"training only","physical_voltage_screen":"0.93..1.07 pu is a diagnostic screen, not a full standards compliance test","noise_parameters":"Gaussian standard deviations, not instrument maximum-error specifications"}
    config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()}
    cp=output/"config.json"
    if cp.exists() and json.loads(cp.read_text(encoding="utf-8"))!=config: raise ValueError("Configuration changed; choose a fresh output directory")
    if not mp.exists() and any(output.rglob("result.json")):raise ValueError("Cached results lack source manifest")
    write_json(mp,manifest);write_json(cp,config)
    jobs=[dict(output=str(output),case=case,suite=suite,repeat=repeat,observation=obs,sample_counts=args.samples,pq_noise_rel=args.pq_noise_rel,voltage_noise_rel=args.voltage_noise_rel,root_meter_noise_rel=args.root_meter_noise_rel,root_sigma=args.root_sigma,impedance_scale=args.impedance_scale) for suite in args.suites for case in args.cases for repeat in range(args.repeats) for obs in args.root_observations]
    completed=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for future in as_completed([ex.submit(run_job,j) for j in jobs]):
            result=future.result();completed.append(result)
            pd.DataFrame([r for v in completed for r in v["rows"]]).to_csv(output/"metrics.csv",index=False)
            pd.DataFrame([r for v in completed for r in v["physical"]]).to_csv(output/"physical_metrics.csv",index=False)
            print(json.dumps({"done":len(completed),"total":len(jobs),"case":result["job"]["case"],"suite":result["job"]["suite"],"observation":result["job"]["observation"],"seconds":result["seconds"]}),flush=True)
    write_json(output/"run_summary.json",{"conditions":len(completed),"physical_scenario_records":sum(len(x["physical"]) for x in completed),"fit_rows":sum(len(x["rows"]) for x in completed),"fit_failures":sum(r["status"]!="success" for x in completed for r in x["rows"])})

if __name__=="__main__":main()
