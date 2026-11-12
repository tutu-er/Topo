from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import hashlib
import json
import sqlite3
import sys
import numpy as np
import pandas as pd
from research_experiments.wzzt_systematic.common import read,write,sha,observed,matrices,linear_loss
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth,clade_metrics,relative_error
from .variants import multistart_original_rule,ac_gate

VARIANTS=("baseline25","long75","final_score","post_refit","multistart75","multistart_original_rule")


def signature(point):
    return hashlib.sha256(json.dumps({k:point[k] for k in ("supports","r","x")},sort_keys=True).encode()).hexdigest()


def ac_task(task):
    sys.path.insert(0,task["source"])
    from research_experiments.wzzt_solver_tuning.ac_screen import ac_score
    p=task["point"]
    previous=task.get("previous",{})
    fit=previous.get("fitting") or ac_score(task["labels"],p["supports"],p["r"],p["x"],observed(task["train"],task["N"]),refit=True,max_nfev=12)
    test=previous.get("testing",{"valid":False});physical=previous.get("physical",{"valid":False});validation={"valid":False}
    if fit["valid"]:
        if not test.get("valid"):test=ac_score(task["labels"],p["supports"],fit["r_refit"],fit["x_refit"],observed(task["test"]),refit=False)
        if not physical.get("valid"):physical=ac_score(task["labels"],p["supports"],fit["r_refit"],fit["x_refit"],observed(task["test_truth"]),refit=False)
        validation=ac_score(task["labels"],p["supports"],fit["r_refit"],fit["x_refit"],observed(task["validation"]),refit=False)
    return dict(key=task["key"],model_signature=task["model_signature"],fitting=fit,testing=test,physical=physical,
                validation=validation,reused_parent_AC=previous.get("reused_parent_AC",False),
                training_refit_reused=bool(previous.get("fitting")))


def evaluate(output,destination,*,partial=False,skip_ac=False):
    output=Path(output).resolve();destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    protocol=read(output/"protocol.json");parent=Path(protocol["parent"]);base=Path(protocol["baseline"])
    parent_models=read(Path(__file__).resolve().parents[2]/"docs/wzzt_retune_20261028/selected_models.json")
    parent_lookup={p["job_id"]:p for p in parent_models if p["method"]=="RNJ_WZZT"}
    rows=[];datasets=[];selected=[];truths={};tasks={};missing=[]
    for job in protocol["jobs"]:
        d=output/"jobs"/job;payload=read(d/"payload.json")
        if not (d/"result.json").exists():
            missing.append(job)
            if partial:continue
            for variant in VARIANTS:rows.append(dict(job_id=job,case=payload["case"],condition=payload["condition"],N=payload["N"],variant=variant,model_available=False,clade_f1=0.))
            continue
        result=read(d/"result.json");labels=result["labels"]
        train,validation=observed(payload["train"],payload["N"]),observed(payload["validation"])
        result["models"]["multistart_original_rule"]=multistart_original_rule(result,read(payload["parent_result"]),train,validation)
        test_path=base/"data"/payload["case"]/payload["condition"]/"test_observed.npz";test=observed(test_path)
        key=payload["case"],payload["condition"]
        if key not in truths:truths[key]=physical_truth(base/"data"/key[0]/key[1]/"truth_network",labels)
        truth=truths[key];true=truth["clades"]
        y=train[3].to_numpy()[:,None]**2-train[2].to_numpy()**2
        drop=train[3].to_numpy()[:,None]-train[2].to_numpy()
        stats=dict(observed_drop_min=float(drop.min()),observed_drop_median=float(np.median(drop)),observed_drop_max=float(drop.max()),
                   squared_drop_std=float(y.std()),P_std=float(train[0].to_numpy().std()),Q_std=float(train[1].to_numpy().std()),
                   training_rows=len(train[0]),validation_rows=len(validation[0]),test_rows=len(test[0]))
        datasets.append(dict(job_id=job,case_name=key[0],condition_name=key[1],N=payload["N"],labels_json=json.dumps(labels),
            train_path=payload["train"],validation_path=payload["validation"],test_path=str(test_path),
            train_sha256=sha(payload["train"]),validation_sha256=sha(payload["validation"]),test_sha256=sha(test_path),
            result_path=str(d/"result.json"),result_sha256=sha(d/"result.json"),
            input_stats_json=json.dumps(stats),condition_json=json.dumps(protocol["conditions"][key[1]])))
        pool={frozenset(labels[k] for k in s) for s in result["pool"] if 2<=len(s)<len(labels)}
        for variant in VARIANTS:
            p=result["models"][variant]
            clades={frozenset(labels[k] for k in s) for s in p["supports"] if 2<=len(s)<len(labels)}
            metric=clade_metrics(clades,true,labels);r,x=matrices(len(labels),p["supports"],p["r"],p["x"])
            train_mae=linear_loss(train,r,x)["mae"];validation_mae=linear_loss(validation,r,x)["mae"]
            if abs(validation_mae-p["validation_mae"])>1e-10 or abs(train_mae-p["objective"])>1e-10:raise AssertionError("selected-model score mismatch")
            false=clades-true;omitted=true-clades
            blocking=[(f,t) for f in false for t in omitted if f&t and not(f<=t or t<=f)]
            diagnostic=dict(false_clades=[sorted(s) for s in false],omitted_true_clades=[sorted(s) for s in omitted],
                blocking_pairs=[dict(false=sorted(f),omitted_true=sorted(t)) for f,t in blocking],
                truth_used_only_posthoc=True)
            searches=result["searches"]
            if variant=="baseline25":stop=result["original_search_stop"];seconds=result["original_search_seconds"]
            elif variant.startswith("multistart"):
                extra=[v for k,v in searches.items() if k.startswith("multistart_")]
                stop=json.dumps([result["original_search_stop"]]+[v["reason"] for v in extra]);seconds=result["original_search_seconds"]+sum(v["seconds"] for v in extra)
            else:stop=searches["long75"]["reason"];seconds=searches["long75"]["seconds"]
            row=dict(job_id=job,case=key[0],condition=key[1],N=payload["N"],variant=variant,model_available=True,**metric,
                validation_mae=validation_mae,train_mae=train_mae,linear_test_mae=linear_loss(test,r,x)["mae"],
                R_error=relative_error(r,truth["R"]),X_error=relative_error(x,truth["X"]),support_count=len(p["supports"]),
                selected_origin=p.get("origin"),search_stop=stop,search_seconds=seconds,shared_dataset_seconds=result["seconds"],
                candidate_pool_true_recall=len(pool&true)/len(true) if true else 1.,blocking_pairs=len(blocking),
                diagnostic_json=json.dumps(diagnostic,sort_keys=True),model_signature=signature(p),
                supports_json=json.dumps(p["supports"]),r_json=json.dumps(p["r"]),x_json=json.dumps(p["x"]))
            rows.append(row)
            task=dict(key=job+"__"+signature(p),job_id=job,variant=variant,labels=labels,point=p,N=payload["N"],
                train=payload["train"],validation=payload["validation"],test=str(test_path),test_truth=str(test_path.with_name("test_truth.npz")),source=str(output/"source"),model_signature=signature(p))
            selected.append(task);tasks[task["key"]]=task
    frame=pd.DataFrame(rows)
    if not skip_ac:
        acdir=output/"ac";acdir.mkdir(exist_ok=True);cache={};todo=[]
        for key,task in tasks.items():
            file=acdir/(key+".json")
            previous=read(file) if file.exists() else None
            if previous is not None and "error" not in previous and previous.get("model_signature")==task["model_signature"] and "validation" in previous:cache[key]=previous;continue
            if previous is not None and "error" not in previous:task["previous"]=previous
            parent_point=parent_lookup[task["job_id"]]["point"]
            if signature(parent_point)==task["model_signature"]:
                oldfile=parent/"ac"/(task["job_id"]+"__RNJ_WZZT.json")
                record=dict(read(oldfile),key=key,model_signature=task["model_signature"],reused_parent_AC=True,parent_record=str(oldfile))
                task["previous"]=record
            todo.append(task)
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures={executor.submit(ac_task,t):t for t in todo}
            for future in as_completed(futures):
                task=futures[future]
                try:record=future.result()
                except Exception as e:record=dict(key=task["key"],model_signature=task["model_signature"],error=str(e),fitting={"valid":False},testing={"valid":False})
                cache[task["key"]]=record;write(acdir/(task["key"]+".json"),record)
        for row in rows:
            record=cache.get(row["job_id"]+"__"+row.get("model_signature",""),{})
            row.update(ac_valid=record.get("testing",{}).get("valid",False),test_ac_rmse=record.get("testing",{}).get("relative_voltage_rmse"),
                physical_test_ac_rmse=record.get("physical",{}).get("relative_voltage_rmse"),
                validation_ac_rmse=record.get("validation",{}).get("relative_voltage_rmse"),
                ac_refit_success=record.get("fitting",{}).get("refit",{}).get("success"),reused_parent_AC=record.get("reused_parent_AC",False))
        # The gate compares only the baseline and original-rule multi-start
        # alternatives, using validation observations and training-fitted AC
        # parameters. Test and topology scores never enter this decision.
        additions=[];selected_lookup={(t["job_id"],t["variant"]):t for t in selected}
        for job in sorted(set(r["job_id"] for r in rows)):
            candidates={r["variant"]:r for r in rows if r["job_id"]==job and r.get("model_available")}
            if "baseline25" not in candidates:continue
            old,new=candidates["baseline25"],candidates["multistart_original_rule"]
            accepted=ac_gate(cache[job+"__"+old["model_signature"]],cache[job+"__"+new["model_signature"]])
            origin="multistart_original_rule" if accepted else "baseline25"
            chosen=dict(candidates[origin],variant="multistart_ac_gate",gate_adopted=accepted,gate_source=origin)
            chosen["selected_source_search_seconds"]=chosen["search_seconds"]
            chosen["search_seconds"]=new["search_seconds"]
            additions.append(chosen);selected.append(dict(selected_lookup[(job,origin)],variant="multistart_ac_gate",gate_adopted=accepted))
        rows.extend(additions)
        frame=pd.DataFrame(rows)
    baseline=frame[frame.variant=="baseline25"].set_index("job_id").clade_f1
    frame["F1_change"]=frame.clade_f1-frame.job_id.map(baseline)
    frame.to_csv(destination/"per_job.csv",index=False,encoding="utf-8-sig")
    frame.groupby(["N","variant"]).agg(rows=("job_id","count"),F1=("clade_f1","mean"),F1_change=("F1_change","mean"),
        validation_mae=("validation_mae","mean"),linear_test_mae=("linear_test_mae","mean"),search_seconds=("search_seconds","mean")).to_csv(destination/"by_sample_size.csv",encoding="utf-8-sig")
    summary=frame.groupby("variant").agg(rows=("job_id","count"),models=("model_available","sum"),F1=("clade_f1","mean"),F1_change=("F1_change","mean"),
        exact=("clade_f1",lambda x:int((x>=1-1e-12).sum())),wins=("F1_change",lambda x:int((x>1e-10).sum())),
        ties=("F1_change",lambda x:int((abs(x)<=1e-10).sum())),losses=("F1_change",lambda x:int((x < -1e-10).sum())),
        R_error=("R_error","mean"),X_error=("X_error","mean"),validation_mae=("validation_mae","mean"),linear_test_mae=("linear_test_mae","mean"),search_seconds=("search_seconds","mean"))
    if not skip_ac:summary=summary.join(frame.groupby("variant").agg(ac_valid=("ac_valid","sum"),test_ac_rmse=("test_ac_rmse","mean"),physical_test_ac_rmse=("physical_test_ac_rmse","mean"),ac_refit_success=("ac_refit_success","sum")))
    summary.to_csv(destination/"summary.csv",encoding="utf-8-sig")
    frame.groupby(["case","condition","variant"]).clade_f1.mean().unstack("variant").to_csv(destination/"by_case_condition.csv",encoding="utf-8-sig")
    write(destination/"selected_models.json",selected)
    write(destination/"datasets.json",datasets)
    file=destination/"catalog.sqlite";temporary=file.with_suffix(".tmp")
    with sqlite3.connect(temporary) as connection:
        pd.DataFrame(datasets).to_sql("datasets",connection,index=False,if_exists="replace")
        frame.drop(columns=["case","condition","N"]).to_sql("selected_models",connection,index=False,if_exists="replace")
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS model_key ON selected_models(job_id,variant)")
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS dataset_key ON datasets(job_id)")
        pd.DataFrame([dict(key="protocol",value=json.dumps(protocol)),dict(key="output",value=str(output)),dict(key="scope",value="selected models, observed inputs and separately labelled posthoc truth diagnostics")]).to_sql("metadata",connection,index=False,if_exists="replace")
    connection.close()
    temporary.replace(file)
    write(destination/"overview.json",dict(planned=protocol["planned"],returned=len(datasets),selected_rows=len(frame),missing=missing,
        variants=VARIANTS if skip_ac else (*VARIANTS,"multistart_ac_gate"),retuned_parameters=False,mean_F1=frame.groupby("variant").clade_f1.mean().to_dict(),
        unique_AC_models=len(tasks),new_AC_tasks=None if skip_ac else len(todo)))
    print(summary.to_string(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");p.add_argument("destination");p.add_argument("--partial",action="store_true");p.add_argument("--skip-ac",action="store_true")
    a=p.parse_args();evaluate(a.output,a.destination,partial=a.partial,skip_ac=a.skip_ac)
