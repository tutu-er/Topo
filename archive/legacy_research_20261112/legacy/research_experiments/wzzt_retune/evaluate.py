"""Common-configuration F1 tuning followed by frozen-parameter AC testing."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import hashlib
import json
import shutil
import sys
import numpy as np
import pandas as pd

from research_experiments.wzzt_systematic.common import read,write,observed,sha
from .selection import METHODS,configurations,score,select


def choose_index(f1,validation=None,counts=None,rows=None):
    f1=np.asarray(f1,dtype=float)
    if rows is not None:
        f1=f1[rows]
        if validation is not None:validation=np.asarray(validation)[rows]
        if counts is not None:counts=np.asarray(counts)[rows]
    means=np.mean(f1,axis=0)
    val=np.mean(validation,axis=0) if validation is not None else np.zeros(len(means))
    sizes=np.mean(counts,axis=0) if counts is not None else np.zeros(len(means))
    return max(range(len(means)),key=lambda i:(round(float(means[i]),12),-float(val[i]),-float(sizes[i]),-i))


def ac_task(task):
    sys.path.insert(0,task["source"])
    from research_experiments.wzzt_solver_tuning.ac_screen import ac_score
    point=task["point"]
    obs=observed(task["train"],task["N"])
    fitting=ac_score(task["labels"],point["supports"],point["r"],point["x"],obs,refit=True,max_nfev=12)
    testing={"valid":False};physical={"valid":False}
    if fitting["valid"]:
        test=observed(task["test"])
        testing=ac_score(task["labels"],point["supports"],fitting["r_refit"],fitting["x_refit"],test,refit=False)
        true_test=observed(task["true_test"])
        physical=ac_score(task["labels"],point["supports"],fitting["r_refit"],fitting["x_refit"],true_test,refit=False)
    return dict(key=task["key"],fitting=fitting,testing=testing,physical=physical,model_sha256=task["model_sha256"])


def evaluate(output,destination,*,partial=False,skip_ac=False):
    output=Path(output).resolve();destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    protocol=read(output/"protocol.json");base=Path(protocol["baseline"])
    # Parent scoring imports no rnj_wzzt code. Pin the runtime only inside AC
    # children; otherwise Windows spawn can import the older frozen research
    # package before this later analysis module is available.
    from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
    jobs=[];results={};truths={}
    for job in protocol["jobs"]:
        directory=output/"jobs"/job
        ended=any((directory/n).exists() for n in ("result.json","error.json","timeout.json"))
        if partial and not ended:continue
        payload=read(directory/"payload.json");jobs.append(payload)
        result=read(directory/"result.json") if (directory/"result.json").exists() else read(directory/"checkpoint.json") if (directory/"checkpoint.json").exists() else None
        results[job]=result
        key=(payload["case"],payload["condition"])
        if key not in truths:
            labels=list(observed(payload["train"],payload["N"])[0].columns)
            truths[key]=physical_truth(base/"data"/key[0]/key[1]/"truth_network",labels)
    if not jobs:raise ValueError("no completed datasets")
    grids=[];chosen={};casewise={};matrices_by_method={}
    chosen_points=[];rows=[];case_rows=[];loco_rows=[]
    cases=np.array([p["case"] for p in jobs])
    for method in METHODS:
        configs=configurations(method)
        f1=np.zeros((len(jobs),len(configs)));val=np.full_like(f1,1e9);counts=np.full_like(f1,1e9)
        for i,payload in enumerate(jobs):
            result=results[payload["job_id"]]
            if result is None:continue
            for j,cfg in enumerate(configs):
                try:
                    _,metric=score(result,method,cfg,truths[(payload["case"],payload["condition"])])
                except KeyError:continue
                f1[i,j]=metric["clade_f1"];val[i,j]=metric["validation_mae"];counts[i,j]=metric["support_count"]
        best=choose_index(f1,val,counts)
        chosen[method]=dict(config=configs[best],mean_F1=float(f1[:,best].mean()),config_count=len(configs),
                            tuning_used_topology_truth=True,scope="one shared configuration across all planned input groups")
        matrices_by_method[method]=(configs,f1,val,counts)
        for j,cfg in enumerate(configs):
            grids.append(dict(method=method,config_index=j,config_json=json.dumps(cfg,sort_keys=True),mean_F1=float(f1[:,j].mean()),
                              exact_recoveries=int((f1[:,j]>=1-1e-12).sum()),mean_validation_mae=float(val[:,j].mean())))
        casewise[method]={}
        for case in sorted(set(cases)):
            mask=cases==case;index=choose_index(f1,val,counts,mask)
            casewise[method][case]=dict(config=configs[index],mean_F1=float(f1[mask,index].mean()))
            for i in np.flatnonzero(mask):
                case_rows.append(dict(job_id=jobs[i]["job_id"],method=method,case=case,N=jobs[i]["N"],condition=jobs[i]["condition"],
                                      clade_f1=float(f1[i,index]),config_json=json.dumps(configs[index],sort_keys=True)))
            held=choose_index(f1,val,counts,~mask) if len(set(cases))>1 else best
            for i in np.flatnonzero(mask):
                loco_rows.append(dict(job_id=jobs[i]["job_id"],method=method,case=case,N=jobs[i]["N"],condition=jobs[i]["condition"],
                                      clade_f1=float(f1[i,held]),config_json=json.dumps(configs[held],sort_keys=True),scope="exploratory leave-one-case-out"))
        for i,payload in enumerate(jobs):
            job=payload["job_id"];result=results[job];directory=output/"jobs"/job
            row=dict(job_id=job,case=payload["case"],condition=payload["condition"],N=payload["N"],method=method,
                     worker_returned=(directory/"result.json").exists(),model_available=False,clade_f1=0.,
                     config_json=json.dumps(configs[best],sort_keys=True),path_grid_oracle_f1=float(f1[i].max()))
            if result is not None:
                try:point,metric=score(result,method,configs[best],truths[(payload["case"],payload["condition"])])
                except KeyError:point=None
                if point is not None:
                    row.update(metric,model_available=True,selected_index=point["selection_index"],preprocess=configs[best]["preprocess"],
                               dataset_seconds=result.get("seconds"),selected_origin=point.get("origin","forward_path"))
                    group=result["methods"][configs[best]["preprocess"]]
                    if method=="WZZT_free":search=group[method]
                    elif method=="WZZT_pool":search=group[method][str(configs[best]["repair"])]
                    elif method=="RNJ_WZZT":search=group["RNJ_search"]
                    else:search=dict(reason="NJ_tree_completed",seconds=0)
                    row.update(search_stop=search["reason"],search_seconds=search["seconds"],
                        shared_bank_and_pool_seconds=group["pool_seconds"],
                        free_stop_certified=search["reason"] in {"certified_no_gain","family_limit"} if method=="WZZT_free" else None,
                        bound_touched=bool(max(point["r"]+point["x"])>=1.99))
                    universe=set(result["labels"])
                    pool={frozenset(result["labels"][k] for k in s) for s in result["pool"] if 2<=len(s)<len(universe)}
                    true=truths[(payload["case"],payload["condition"])]["clades"]
                    row["candidate_pool_true_recall"]=len(pool&true)/len(true) if true else 1.
                    predicted={frozenset(result["labels"][k] for k in s) for s in point["supports"] if 2<=len(s)<len(universe)}
                    row["selected_clades_outside_initial_pool"]=len(predicted-pool)
                    row["selected_true_clades_outside_initial_pool"]=len((predicted-pool)&true)
                    chosen_points.append(dict(key=f"{job}__{method}",job_id=job,method=method,config=configs[best],point=point,
                        labels=result["labels"],N=payload["N"],train=payload["train"],
                        test=str(base/"data"/payload["case"]/payload["condition"]/"test_observed.npz"),
                        true_test=str(base/"data"/payload["case"]/payload["condition"]/"test_truth.npz"),source=str(output/"source")))
            rows.append(row)
    write(destination/"chosen_global.json",chosen)
    write(destination/"chosen_casewise.json",casewise)
    pd.DataFrame(grids).to_csv(destination/"parameter_grid.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(case_rows).to_csv(destination/"casewise_results.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(loco_rows).to_csv(destination/"leave_one_case_out.csv",index=False,encoding="utf-8-sig")
    write(destination/"selected_models.json",chosen_points)
    if not skip_ac:
        acdir=output/"ac";acdir.mkdir(exist_ok=True)
        tasks=[];cache={}
        for task in chosen_points:
            digest=hashlib.sha256(json.dumps(task["point"],sort_keys=True).encode()).hexdigest()
            task["model_sha256"]=digest
            path=acdir/(task["key"]+".json")
            previous=read(path) if path.exists() else None
            if previous is not None and previous.get("model_sha256")==digest and "error" not in previous:
                cache[task["key"]]=previous
            else:
                if previous is not None and "error" in previous:
                    archive=output/"ac_infrastructure_failed"
                    archive.mkdir(exist_ok=True)
                    if not (archive/path.name).exists():shutil.copy2(path,archive/path.name)
                tasks.append(task)
        if (output/"ac_infrastructure_failed").exists():
            write(output/"ac_recovery.json",dict(reason="Windows spawn analysis import was shadowed by older frozen package",
                preserved_records=len(list((output/"ac_infrastructure_failed").glob("*.json"))),
                structure_search_repeated=False,hyperparameters_changed=False))
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures={executor.submit(ac_task,task):task for task in tasks}
            for future in as_completed(futures):
                task=futures[future]
                try:record=future.result()
                except Exception as e:record=dict(key=task["key"],model_sha256=task["model_sha256"],error=str(e),fitting={"valid":False},testing={"valid":False},physical={"valid":False})
                cache[task["key"]]=record;write(acdir/(task["key"]+".json"),record)
        for row in rows:
            record=cache.get(row["job_id"]+"__"+row["method"],{})
            row.update(ac_valid=record.get("testing",{}).get("valid",False),
                test_ac_rmse=record.get("testing",{}).get("relative_voltage_rmse"),
                physical_test_ac_rmse=record.get("physical",{}).get("relative_voltage_rmse"),
                ac_refit_success=record.get("fitting",{}).get("refit",{}).get("success"))
    frame=pd.DataFrame(rows)
    baseline=pd.read_csv(Path(__file__).resolve().parents[2]/"docs/wzzt_systematic_20261027/per_job.csv")
    old={(r.case,r.condition,r.N,r.method):r.clade_f1 for r in baseline.itertuples()}
    mapping={"NJ":"NJ","WZZT_free":"WZZT","WZZT_pool":"WZZT","RNJ_WZZT":"RNJ+WZZT"}
    frame["round1_F1"]=[old[(r.case,r.condition,r.N,mapping[r.method])] for r in frame.itertuples()]
    frame["F1_change"]=frame.clade_f1-frame.round1_F1
    frame.to_csv(destination/"per_job.csv",index=False,encoding="utf-8-sig")
    aggregate=frame.groupby(["N","method"]).agg(planned=("job_id","count"),models=("model_available","sum"),
        F1=("clade_f1","mean"),round1_F1=("round1_F1","mean"),F1_change=("F1_change","mean"),
        exact_recoveries=("clade_f1",lambda x:int((x>=1.-1e-12).sum())))
    if "test_ac_rmse" in frame:
        aggregate=aggregate.join(frame.groupby(["N","method"]).agg(test_ac_rmse=("test_ac_rmse","mean"),ac_valid=("ac_valid","sum")))
    aggregate.to_csv(destination/"by_sample_size.csv",encoding="utf-8-sig")
    overview=dict(planned_datasets=80,evaluated_datasets=len(jobs),returned_datasets=sum((output/"jobs"/j["job_id"]/"result.json").exists() for j in jobs),
        planned_method_rows=320,method_rows=len(frame),models=int(frame.model_available.sum()),
        mean_F1=frame.groupby("method").clade_f1.mean().to_dict(),
        scope="result-tuned benchmark scores; one common configuration per method; not a blind topology test")
    write(destination/"overview.json",overview)
    print(overview,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("output");parser.add_argument("destination");parser.add_argument("--partial",action="store_true");parser.add_argument("--skip-ac",action="store_true")
    a=parser.parse_args();evaluate(a.output,a.destination,partial=a.partial,skip_ac=a.skip_ac)
