from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import hashlib
import json
import sqlite3
import sys
import pandas as pd
from research_experiments.wzzt_systematic.common import read,write,sha,observed,matrices,linear_loss
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth,clade_metrics,relative_error

ROOT=Path(__file__).resolve().parents[2]
VARIANTS=("baseline25","multistart_original_rule","backward","ridge_pool","two_step","combined")


def signature(point):
    return hashlib.sha256(json.dumps({k:point[k] for k in ("supports","r","x")},sort_keys=True).encode()).hexdigest()


def ac_task(task):
    sys.path.insert(0,task["source"])
    from research_experiments.wzzt_solver_tuning.ac_screen import ac_score
    p=task["point"]
    fit=ac_score(task["labels"],p["supports"],p["r"],p["x"],observed(task["train"],task["N"]),refit=True,max_nfev=12)
    tests={name:{"valid":False} for name in ("testing","physical","validation")}
    if fit["valid"]:
        for name,path in (("testing",task["test"]),("physical",task["test_truth"]),("validation",task["validation"])):
            tests[name]=ac_score(task["labels"],p["supports"],fit["r_refit"],fit["x_refit"],observed(path),refit=False)
    return dict(key=task["key"],model_signature=task["model_signature"],fitting=fit,**tests,reused_prior_AC=False)


def evaluate(output,destination,*,partial=False,skip_ac=False):
    output=Path(output).resolve();destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    protocol=read(output/"protocol.json");prior=Path(protocol["prior"]);base=Path(protocol["baseline"])
    olddir=ROOT/"docs/wzzt_rnj_budget_20261028"
    oldmodels={(p["job_id"],p["variant"]):p["point"] for p in read(olddir/"selected_models.json") if p["variant"] in VARIANTS[:2]}
    olddatasets={d["job_id"]:d for d in read(olddir/"datasets.json")}
    oldrows=pd.read_csv(olddir/"per_job.csv");oldrows=oldrows.set_index(["job_id","variant"])
    nj=pd.read_csv(ROOT/"docs/wzzt_retune_20261028/per_job.csv");nj=nj[nj.method=="NJ"].set_index("job_id").clade_f1
    noise=pd.read_csv(destination/"previous_all_noise.csv").set_index("job_id")
    rows=[];selected=[];datasets=[];truths={};tasks={};missing=[]
    for job in protocol["jobs"]:
        d=output/"jobs"/job
        ended=any((d/n).exists() for n in ("result.json","error.json","timeout.json"))
        if partial and not ended:continue
        payload=read(d/"payload.json");parent=read(payload["parent_result"])
        result=read(d/"result.json") if (d/"result.json").exists() else read(d/"checkpoint.json") if (d/"checkpoint.json").exists() else {}
        returned=(d/"result.json").exists()
        if not returned:missing.append(job)
        labels=parent["labels"];train=observed(payload["train"],payload["N"]);validation=observed(payload["validation"])
        testfile=base/"data"/payload["case"]/payload["condition"]/"test_observed.npz";test=observed(testfile)
        key=payload["case"],payload["condition"]
        if key not in truths:truths[key]=physical_truth(testfile.parent/"truth_network",labels)
        truth=truths[key];true=truth["clades"]
        dataset=dict(olddatasets[job]);tracefile=next((d/n for n in ("result.json","checkpoint.json","error.json","timeout.json") if (d/n).exists()),None)
        dataset.update(result_path=str(tracefile) if tracefile else None,result_sha256=sha(tracefile) if tracefile else None,
            prior_result_path=payload["prior_result"],noise_stats_json=noise.loc[job].to_json())
        datasets.append(dataset)
        models={v:oldmodels[(job,v)] for v in VARIANTS[:2]};models.update(result.get("models",{}))
        for variant in VARIANTS:
            row=dict(job_id=job,case=key[0],condition=key[1],N=payload["N"],variant=variant,
                worker_returned=returned,model_available=variant in models,clade_f1=0.,
                baseline25_F1=float(oldrows.loc[(job,"baseline25"),"clade_f1"]),
                multistart_F1=float(oldrows.loc[(job,"multistart_original_rule"),"clade_f1"]),NJ_F1=float(nj[job]))
            for column in ("pq_sigma_relative","v_sigma_relative","root_sigma_relative","P_realized_relative_std","Q_realized_relative_std",
                           "V_realized_relative_std","root_realized_relative_std","drop_noise_std_pu","squared_drop_noise_std","true_mean_drop_pct"):
                row[column]=float(noise.loc[job,column])
            if variant in models:
                point=models[variant];r,x=matrices(len(labels),point["supports"],point["r"],point["x"])
                clades={frozenset(labels[k] for k in s) for s in point["supports"] if 2<=len(s)<len(labels)}
                row.update(clade_metrics(clades,true,labels),R_error=relative_error(r,truth["R"]),X_error=relative_error(x,truth["X"]),
                    validation_mae=linear_loss(validation,r,x)["mae"],train_mae=linear_loss(train,r,x)["mae"],linear_test_mae=linear_loss(test,r,x)["mae"],
                    ranking_validation_mae=point.get("ranking_validation_mae"),selected_origin=point.get("origin"),support_count=len(point["supports"]),
                    model_signature=signature(point),supports_json=json.dumps(point["supports"]),r_json=json.dumps(point["r"]),x_json=json.dumps(point["x"]))
                if variant not in VARIANTS[:2] and (abs(row["validation_mae"]-point["validation_mae"])>1e-10 or abs(row["train_mae"]-point["objective"])>1e-10):
                    raise AssertionError("selected physical score mismatch")
                pool=result.get("augmentation",{}).get("pool",parent["pool"]) if variant in ("ridge_pool","combined") else parent["pool"]
                pooled={frozenset(labels[i] for i in s) for s in pool if 2<=len(s)<len(labels)}
                oldpool={frozenset(labels[i] for i in s) for s in parent["pool"] if 2<=len(s)<len(labels)}
                false=clades-true;omitted=true-clades
                blocking=[(f,t) for f in false for t in omitted if f&t and not(f<=t or t<=f)]
                row.update(candidate_pool_true_recall=len(pooled&true)/len(true),pool_size=len(pool),pool_true_added=len((pooled-oldpool)&true),
                    pool_false_added=len((pooled-oldpool)-true),blocking_pairs=len(blocking),
                    diagnostic_json=json.dumps(dict(false_clades=[sorted(s) for s in false],omitted_true_clades=[sorted(s) for s in omitted],
                        blocking_pairs=[dict(false=sorted(f),omitted_true=sorted(t)) for f,t in blocking],truth_used_only_posthoc=True)))
                if variant in VARIANTS[:2]:
                    row.update(search_stop=oldrows.loc[(job,variant),"search_stop"],additional_seconds=0.,search_seconds=float(oldrows.loc[(job,variant),"search_seconds"]))
                else:
                    ops=result.get("operations",{});aug=result.get("augmentation",{}).get("seconds",0.)
                    names={"backward":["backward"],"ridge_pool":["ridge_pool"],"two_step":["two_step"],"combined":["ridge_pool","combined_two_step","combined_backward"]}[variant]
                    extra=sum(ops.get(n,{}).get("seconds",0.) for n in names)+(aug if variant in ("ridge_pool","combined") else 0.)
                    row.update(search_stop=json.dumps({n:ops.get(n,{}).get("reason","unavailable") for n in names}),additional_seconds=extra,
                        search_seconds=float(oldrows.loc[(job,"multistart_original_rule"),"search_seconds"])+extra,
                        shared_dataset_seconds=result.get("seconds"),new_candidate_models=len(result.get("candidates",{}).get(variant,[])))
                task=dict(key=job+"__"+signature(point),job_id=job,variant=variant,point=point,labels=labels,N=payload["N"],train=payload["train"],
                    validation=payload["validation"],test=str(testfile),test_truth=str(testfile.with_name("test_truth.npz")),source=str(output/"source"),model_signature=signature(point))
                selected.append(task);tasks[task["key"]]=task
            rows.append(row)
    if not skip_ac:
        acdir=output/"ac";acdir.mkdir(exist_ok=True);cache={};todo=[];reused=0
        for key,task in tasks.items():
            file=acdir/(key+".json");priorfile=prior/"ac"/(key+".json")
            if file.exists() and "error" not in read(file):cache[key]=read(file)
            elif priorfile.exists() and "error" not in read(priorfile):
                record=dict(read(priorfile),reused_prior_AC=True,prior_record=str(priorfile));cache[key]=record;write(file,record);reused+=1
            else:todo.append(task)
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures={executor.submit(ac_task,t):t for t in todo}
            for f in as_completed(futures):
                t=futures[f]
                try:record=f.result()
                except Exception as e:record=dict(key=t["key"],model_signature=t["model_signature"],error=str(e),testing={"valid":False},fitting={"valid":False})
                cache[t["key"]]=record;write(acdir/(t["key"]+".json"),record)
        for row in rows:
            record=cache.get(row["job_id"]+"__"+row.get("model_signature",""),{})
            row.update(ac_valid=record.get("testing",{}).get("valid",False),test_ac_rmse=record.get("testing",{}).get("relative_voltage_rmse"),
                physical_test_ac_rmse=record.get("physical",{}).get("relative_voltage_rmse"),validation_ac_rmse=record.get("validation",{}).get("relative_voltage_rmse"),
                ac_refit_success=record.get("fitting",{}).get("refit",{}).get("success"))
    frame=pd.DataFrame(rows)
    for reference,column in (("baseline25","baseline25_F1"),("multistart","multistart_F1"),("NJ","NJ_F1")):
        frame["delta_vs_"+reference]=frame.clade_f1-frame[column]
    frame.to_csv(destination/"per_job.csv",index=False,encoding="utf-8-sig")
    summary=frame.groupby("variant").agg(rows=("job_id","count"),models=("model_available","sum"),F1=("clade_f1","mean"),
        exact=("clade_f1",lambda x:int((x>=1-1e-12).sum())),delta_vs_multistart=("delta_vs_multistart","mean"),
        wins=("delta_vs_multistart",lambda x:int((x>1e-10).sum())),ties=("delta_vs_multistart",lambda x:int((abs(x)<=1e-10).sum())),
        losses=("delta_vs_multistart",lambda x:int((x < -1e-10).sum())),delta_vs_baseline25=("delta_vs_baseline25","mean"),
        R_error=("R_error","mean"),X_error=("X_error","mean"),candidate_pool_true_recall=("candidate_pool_true_recall","mean"),
        linear_test_mae=("linear_test_mae","mean"),additional_seconds=("additional_seconds","mean"))
    if not skip_ac:summary=summary.join(frame.groupby("variant").agg(ac_valid=("ac_valid","sum"),test_ac_rmse=("test_ac_rmse","mean"),ac_refit_success=("ac_refit_success","sum")))
    summary.to_csv(destination/"summary.csv",encoding="utf-8-sig")
    frame.groupby(["N","variant"]).agg(rows=("job_id","count"),F1=("clade_f1","mean"),delta_vs_multistart=("delta_vs_multistart","mean"),
        delta_vs_baseline25=("delta_vs_baseline25","mean"),wins=("delta_vs_multistart",lambda x:int((x>1e-10).sum())),
        losses=("delta_vs_multistart",lambda x:int((x < -1e-10).sum()))).to_csv(destination/"by_sample_size.csv",encoding="utf-8-sig")
    frame.groupby(["case","condition","variant"]).clade_f1.mean().unstack("variant").to_csv(destination/"by_case_condition.csv",encoding="utf-8-sig")
    frame[frame.clade_f1<1-1e-10].to_csv(destination/"unidentified.csv",index=False,encoding="utf-8-sig")
    frame[abs(frame.delta_vs_multistart)>1e-10].to_csv(destination/"changes_vs_multistart.csv",index=False,encoding="utf-8-sig")
    frame[abs(frame.delta_vs_baseline25)>1e-10].to_csv(destination/"changes_vs_baseline25.csv",index=False,encoding="utf-8-sig")
    write(destination/"selected_models.json",selected);write(destination/"datasets.json",datasets)
    db=destination/"catalog.sqlite";temporary=db.with_suffix(".tmp")
    with sqlite3.connect(temporary) as connection:
        pd.DataFrame(datasets).to_sql("datasets",connection,index=False,if_exists="replace")
        frame.drop(columns=["case","condition","N"]).to_sql("selected_models",connection,index=False,if_exists="replace")
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS model_key ON selected_models(job_id,variant)")
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS dataset_key ON datasets(job_id)")
        pd.DataFrame([dict(key="protocol",value=json.dumps(protocol)),dict(key="output",value=str(output))]).to_sql("metadata",connection,index=False,if_exists="replace")
    connection.close();temporary.replace(db)
    write(destination/"overview.json",dict(planned_inputs=protocol["planned"],evaluated_inputs=len(datasets),planned_models=protocol["planned"]*len(VARIANTS),
        selected_records=len(frame),available_models=int(frame.model_available.sum()),incomplete_inputs=missing,
        mean_F1=frame.groupby("variant").clade_f1.mean().to_dict(),unique_AC_models=len(tasks),new_truth_tuning=False))
    print(summary.to_string(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");p.add_argument("destination");p.add_argument("--partial",action="store_true");p.add_argument("--skip-ac",action="store_true")
    a=p.parse_args();evaluate(a.output,a.destination,partial=a.partial,skip_ac=a.skip_ac)
