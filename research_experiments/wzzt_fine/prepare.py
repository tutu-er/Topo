from pathlib import Path
from datetime import datetime,timezone
from dataclasses import replace
import argparse
import shutil
import sys
import numpy as np
from research_experiments.wzzt_systematic.common import read,write,sha,CASES
from .common import SIZES,SIGMAS,MAIN,TARGET,AC_METHODS,sample_indices

ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/"outputs/wzzt_refine_20261028_full"


def prepare(output,only=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    parent_manifest=read(PARENT/"manifest.json")
    for p,h in parent_manifest["local"].items():
        if p.startswith("source/") and sha(PARENT/p)!=h:raise ValueError("frozen producer changed: "+p)
    shutil.copytree(PARENT/"source",output/"source",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    shutil.copytree(Path(__file__).parent,output/"source/research_experiments/wzzt_fine",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    sys.path.insert(0,str(output/"source"))
    from rnj_wzzt.scenario import simulation as sim
    all_jobs=[f"{case}__b{bundle}__v{int(round(sigma*10000)):04d}__N{N}" for case in CASES for bundle in range(3) for sigma in SIGMAS for N in SIZES]
    jobs=[j for j in all_jobs if only is None or any(s in j for s in only)]
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,planned=len(jobs),full_matrix=len(jobs)==180,
        cases=CASES,sizes=SIZES,voltage_sigmas=SIGMAS,pq_sigma=.005,root_process_sigma=.003,root_mean=1.02,
        bundles=[dict(bundle=b,train=10+10*b,validation=11+10*b,test=12+10*b) for b in range(3)],
        main_methods=MAIN,target_methods=TARGET,target_rule="voltage sigma .001 and total N in [24,48]",AC_methods=AC_METHODS,
        expected_main_rows=len(jobs)*len(MAIN),expected_full_main_rows=1620,expected_full_target_rows=144,
        samples_per_day={N:sample_indices(N).tolist() for N in SIZES},physical_days=3,physical_points_per_day=96,
        common_random_numbers="same physical data and standard-normal errors across noise levels within bundle; nested sample sets across N",
        independent_units="three independent training/validation/test bundles; noise/N conditions are paired, not independent repeats",
        frozen_parameters=dict(raw_fit=True,resolution=.16,validation_fraction=.03,coefficient_bound=2.,ridge_strengths=[.001,.01],
            primary25=True,max_extra_starts=2,per_extra_seconds=25,extension_seconds=25,NJ_geometry="RX75",NJ_collapse=.16,RNJ_native_collapse=.24),
        target_parameters=dict(deletion_rounds=1,guards=[None,0.,.01,.03],beam_widths=[3,5],beam_seconds=25),
        budgets=dict(workers=4,worker_hard_seconds=360,fixed_LP_seconds=15),
        truth_tuning=False,baseline_parameter_model="same fixed-tree joint L1 refit for parameter/prediction comparison; native NJ/RNJ families retained for structural scoring",
        model_runtime_source=str(PARENT/"source"),scope="fresh repeated noise-size panel on four fixed synthetic networks; parameters not retuned")
    wanted=set(jobs)
    for case in CASES:
        if not any(j.startswith(case+"__") for j in wanted):continue
        context=sim._prepare_case_context(case,96,scenario_suite="reference",root_observation="noisy",root_meter_noise_rel=.0002,root_sigma=.003,impedance_scale=1.)
        network=output/"data"/case/"truth_network";network.mkdir(parents=True)
        context.net.buses.to_csv(network/"buses.csv",index=False);context.net.branches.to_csv(network/"branches.csv",index=False)
        write(network/"metadata.json",dict(root_bus=int(context.net.root_bus),base_kv=float(context.net.base_kv),base_mva=float(context.net.base_mva)))
        for bundle in range(3):
            if not any(j.startswith(f"{case}__b{bundle}__") for j in wanted):continue
            bundle_dir=output/"data"/case/f"b{bundle}";bundle_dir.mkdir(parents=True)
            physical={}
            for split,replicate in (("train",10+10*bundle),("validation",11+10*bundle),("test",12+10*bundle)):
                physical[split]=[];truth={k:[] for k in ("P","Q","V","root")};diag=[]
                for day in range(3):
                    base=sim._build_base_trajectory(context,day,replicate)
                    ac,voltage=sim._solve_physical_ac(context,base.p_base,base.q_base,base.root_true)
                    p,q=base.p_base.loc[:,context.terminals],base.q_base.loc[:,context.terminals]
                    physical[split].append((base,p,q,voltage))
                    for k,value in zip(truth,(p,q,voltage,base.root_true)):truth[k].append(value.to_numpy())
                    diag.append(dict(day=day,replicate=replicate,profile=base.profile,seed=base.shifted_seed,
                        converged=int(ac["converged"].sum()),voltage_min=float(voltage.min().min()),voltage_max=float(voltage.max().max())))
                np.savez_compressed(bundle_dir/f"{split}_truth.npz",labels=context.terminals,**{k:np.stack(v) for k,v in truth.items()})
                write(bundle_dir/f"{split}_physical.json",diag)
            for sigma in SIGMAS:
                ids=[f"{case}__b{bundle}__v{int(round(sigma*10000)):04d}__N{N}" for N in SIZES]
                if not any(j in wanted for j in ids):continue
                folder=bundle_dir/f"v{int(round(sigma*10000)):04d}";folder.mkdir()
                obs_context=replace(context,settings=dict(context.settings,root_meter_noise_rel=sigma))
                for split,days in physical.items():
                    measured={k:[] for k in ("P","Q","V","root")};settings=[]
                    for base,p,q,v in days:
                        obs=sim._observe_terminal_channels(obs_context,base,p,q,v,pq_noise_rel=.005,v_noise_rel=sigma)
                        for k,name in zip(measured,("p_measured","q_measured","voltage_measured","root_observed")):measured[k].append(obs[name].to_numpy())
                        settings.append(dict(obs["scenario_settings"],actual_pq_sigma_relative=.005,actual_voltage_sigma_relative=sigma))
                    np.savez_compressed(folder/f"{split}_observed.npz",labels=context.terminals,**{k:np.stack(v) for k,v in measured.items()})
                    write(folder/f"{split}_meters.json",settings)
                for N,job in zip(SIZES,ids):
                    if job not in wanted:continue
                    d=output/"jobs"/job;d.mkdir(parents=True)
                    write(d/"payload.json",dict(job_id=job,case=case,bundle=bundle,condition=f"v{int(round(sigma*10000)):04d}",N=N,
                        voltage_sigma=sigma,pq_sigma=.005,targeted=sigma==.001 and N in (24,48),
                        train=str(folder/"train_observed.npz"),validation=str(folder/"validation_observed.npz"),test=str(folder/"test_observed.npz"),
                        train_truth=str(bundle_dir/"train_truth.npz"),validation_truth=str(bundle_dir/"validation_truth.npz"),test_truth=str(bundle_dir/"test_truth.npz"),
                        truth_network=str(network)))
            print("PREPARED",case,bundle,flush=True)
    write(output/"protocol.json",protocol)
    write(output/"manifest.json",{p.relative_to(output).as_posix():sha(p) for p in output.rglob("*") if p.is_file()})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");p.add_argument("--only",nargs="+");a=p.parse_args();prepare(a.output,a.only)
