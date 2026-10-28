from pathlib import Path
from dataclasses import replace
from datetime import datetime,timezone
import shutil
import sys
import numpy as np
from research_experiments.wzzt_systematic.common import read,write,sha,CASES,CONDITIONS,SIZES

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"outputs/wzzt_systematic_20261027_round1"


def prepare(output):
    for relative,digest in read(BASE/"frozen_manifest.json").items():
        if sha(BASE/relative)!=digest:raise ValueError("baseline input/source changed: "+relative)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(BASE/"source",output/"source",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    shutil.copytree(Path(__file__).parent,output/"source/research_experiments/wzzt_retune",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    shutil.copy2(ROOT/"research_experiments/wzzt_systematic/common.py",output/"source/research_experiments/wzzt_systematic/common.py")
    sys.path.insert(0,str(output/"source"))
    from rnj_wzzt.scenario import simulation as sim
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),baseline=str(BASE),data_policy="byte-identical training/test observations from round 1",
        cases=CASES,conditions=CONDITIONS,sizes=SIZES,methods=["NJ","WZZT_free","WZZT_pool","RNJ_WZZT"],
        methods_scope={"NJ":"common outcome-tuned NJ geometry/preprocessing/collapse", "WZZT_free":"singleton free-domain MILP, legal LP warm start, feasible-gain fallback, no online floor",
        "WZZT_pool":"finite data-derived pool, singleton init, exact fixed-family LP enumeration; correction on/off grid",
        "RNJ_WZZT":"full ordinary RNJ families, joint L1 fit, deletable supports and crossing-support replacement; finite pool"},
        tuning=dict(uses_benchmark_topology_truth=True,primary="one global configuration per method across all 80 conditions",
            secondary="casewise common configuration across 20 conditions; exploratory",
            preprocess=["raw","demean"],geometry=["R","RX75","X"],nj_collapse=[0.,.04,.08,.16,.24],
            resolution=[0.,.04,.08,.12,.16,.24],selection=["last","validation_min","validation_simple_1pct","validation_simple_3pct"]),
        budgets=dict(pool_search_seconds=25,free_search_seconds_per_preprocess=40,free_extension_seconds=5,
                     fixed_LP_seconds=15,worker_hard_seconds=420,workers=4),
        validation_replicate=2,test_replicate=1,train_replicate=0,
        differences_from_baseline=["validation data added; training/test unchanged","retained paths and useful incumbents on timeout",
            "warm starts satisfy original free-domain constraints","initial singleton coefficient touch does not restart the path",
            "two preprocessing variants; no online RX75 binary floor; post-fit resolution tuned",
            "finite pool and full RNJ initialization are explicitly separate algorithm variants"],jobs=[])
    external={}
    for case in CASES:
        for condition,settings in CONDITIONS.items():
            directory=output/"validation"/case/condition;directory.mkdir(parents=True)
            context=sim._prepare_case_context(case,96,scenario_suite="reference",root_observation="noisy",
                root_meter_noise_rel=settings["v_noise"],root_sigma=.003,impedance_scale=1.)
            if settings["low_xr"]:
                service=context.net.branches.branch_type.eq("service")
                context.net.branches.loc[:,"x_ohm"]=context.net.branches.r_ohm*np.where(service,.06,.30)
            pool={key:[] for key in ("P","Q","V","root")}
            for day in range(3):
                base=sim._build_base_trajectory(context,day,2)
                base=replace(base,p_base=base.p_base*settings["load_factor"],q_base=base.q_base*settings["load_factor"])
                ac,v=sim._solve_physical_ac(context,base.p_base,base.q_base,base.root_true)
                d=sim._observe_terminal_channels(context,base,base.p_base,base.q_base,v,pq_noise_rel=settings["pq_noise"],v_noise_rel=settings["v_noise"])
                for key,name in zip(pool,("p_measured","q_measured","voltage_measured","root_observed")):pool[key].append(d[name].to_numpy())
            target=directory/"observed.npz"
            np.savez_compressed(target,labels=context.terminals,**{k:np.stack(v) for k,v in pool.items()})
            for name in ("train_observed.npz","test_observed.npz"):
                file=BASE/"data"/case/condition/name;external[str(file)]=sha(file)
            for size in SIZES:
                job=f"{case}__{condition}__N{size}"
                folder=output/"jobs"/job;folder.mkdir(parents=True)
                write(folder/"payload.json",dict(job_id=job,case=case,condition=condition,N=size,
                    train=str(BASE/"data"/case/condition/"train_observed.npz"),validation=str(target)))
                protocol["jobs"].append(job)
    write(output/"protocol.json",protocol)
    files=[p for p in output.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    write(output/"manifest.json",dict(local={p.relative_to(output).as_posix():sha(p) for p in files},external=external))


if __name__=="__main__":prepare(sys.argv[1])
