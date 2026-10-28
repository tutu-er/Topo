from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import hashlib
import json
import sqlite3
import sys
import numpy as np
import pandas as pd
from research_experiments.wzzt_systematic.common import read,write,sha,matrices,linear_loss
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth,clade_metrics,relative_error
from .common import observed,sample_indices,MAIN,TARGET,AC_METHODS


def signature(point):
    return hashlib.sha256(json.dumps({k:point[k] for k in ('supports','r','x')},sort_keys=True).encode()).hexdigest()


def ac_task(t):
    sys.path.insert(0,t['source'])
    from research_experiments.wzzt_solver_tuning.ac_screen import ac_score
    p=t['point'];fit=ac_score(t['labels'],p['supports'],p['r'],p['x'],observed(t['train'],t['N']),refit=True,max_nfev=12)
    test={'valid':False};physical={'valid':False}
    if fit['valid']:
        test=ac_score(t['labels'],p['supports'],fit['r_refit'],fit['x_refit'],observed(t['test']),refit=False)
        physical=ac_score(t['labels'],p['supports'],fit['r_refit'],fit['x_refit'],observed(t['test_truth']),refit=False)
    return dict(key=t['key'],model_signature=t['model_signature'],fitting=fit,testing=test,physical=physical)


def evaluate(output,destination,*,partial=False,skip_ac=False):
    output=Path(output).resolve();destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    protocol=read(output/'protocol.json');rows=[];datasets=[];selected=[];truths={};tasks={};incomplete=[]
    for job in protocol['jobs']:
        d=output/'jobs'/job
        ended=any((d/n).exists() for n in ('result.json','error.json','timeout.json'))
        if partial and not ended:continue
        payload=read(d/'payload.json');done=(d/'result.json').exists()
        result=read(d/'result.json') if done else read(d/'checkpoint.json') if (d/'checkpoint.json').exists() else {}
        if not done:incomplete.append(job)
        train,validation,test=observed(payload['train'],payload['N']),observed(payload['validation']),observed(payload['test'])
        labels=list(train[0].columns);case=payload['case']
        if case not in truths:truths[case]=physical_truth(Path(payload['truth_network']),labels)
        truth=truths[case];true=truth['clades'];noise={}
        train_true=observed(payload['train_truth'],payload['N'])
        for name,a,b in zip(('P','Q','V','root'),train,train_true):
            noise[name+'_realized_relative_std']=float(np.std((a.to_numpy()-b.to_numpy())/np.maximum(abs(b.to_numpy()),1e-12)))
        ds=dict(job_id=job,case_name=case,condition_name=payload['condition'],bundle=payload['bundle'],N=payload['N'],
            voltage_sigma=payload['voltage_sigma'],pq_sigma=payload['pq_sigma'],labels_json=json.dumps(labels),
            train_indices_json=json.dumps(sample_indices(payload['N']).tolist()),noise_stats_json=json.dumps(noise),
            input_contract='three physical days; ordered terminal P/Q/V and measured root',truth_network=payload['truth_network'])
        for split in ('train','validation','test'):
            ds[split+'_path']=payload[split];ds[split+'_sha256']=sha(payload[split]);ds[split+'_truth_path']=payload[split+'_truth']
        path=next((d/n for n in ('result.json','checkpoint.json','error.json','timeout.json') if (d/n).exists()),None)
        ds.update(result_path=str(path) if path else None,result_sha256=sha(path) if path else None)
        datasets.append(ds)
        variants=MAIN+(TARGET if payload['targeted'] else ())
        for variant in variants:
            row=dict(job_id=job,case=case,condition=payload['condition'],bundle=payload['bundle'],N=payload['N'],
                voltage_sigma=payload['voltage_sigma'],pq_sigma=payload['pq_sigma'],variant=variant,panel='main' if variant in MAIN else 'targeted',
                worker_returned=done,model_available=variant in result.get('models',{}),clade_f1=0.,ac_planned=variant in AC_METHODS,**noise)
            if row['model_available']:
                p=result['models'][variant];r,x=matrices(len(labels),p['supports'],p['r'],p['x'])
                clades={frozenset(labels[i] for i in s) for s in p['supports'] if 2<=len(s)<len(labels)}
                row.update(clade_metrics(clades,true,labels),R_error=relative_error(r,truth['R']),X_error=relative_error(x,truth['X']),
                    train_mae=linear_loss(train,r,x)['mae'],validation_mae=linear_loss(validation,r,x)['mae'],linear_test_mae=linear_loss(test,r,x)['mae'],
                    ranking_validation_mae=p.get('ranking_validation_mae'),support_count=len(p['supports']),selected_origin=p.get('origin'),
                    model_signature=signature(p),supports_json=json.dumps(p['supports']),r_json=json.dumps(p['r']),x_json=json.dumps(p['x']),
                    base_pipeline_seconds=result.get('base_pipeline_seconds'),shared_worker_seconds=result.get('seconds'),
                    parameter_model='joint fixed-tree L1; native NJ/RNJ families for structure')
                if abs(row['train_mae']-p['objective'])>1e-8 or abs(row['validation_mae']-p['validation_mae'])>1e-8:raise AssertionError('physical model score mismatch')
                pool=result.get('domains',{}).get(variant,result['pool']);oldpool={tuple(s) for s in result['pool']}
                pooled={frozenset(labels[i] for i in s) for s in pool if 2<=len(s)<len(labels)}
                false=clades-true;omitted=true-clades
                blocking=[(f,t) for f in false for t in omitted if f&t and not(f<=t or t<=f)]
                row.update(pool_size=len(pool),candidate_pool_true_recall=len(pooled&true)/len(true),
                    selected_new_true_clades=sum(tuple(s) not in oldpool and frozenset(labels[i] for i in s) in true for s in p['supports']),
                    diagnostic_json=json.dumps(dict(false_clades=[sorted(s) for s in false],omitted_true_clades=[sorted(s) for s in omitted],
                        blocking_pairs=[dict(false=sorted(f),omitted_true=sorted(t)) for f,t in blocking],truth_used_only_posthoc=True)))
                if variant in result['searches']:
                    row['search_stop']=result['searches'][variant]['reason'];row['search_seconds']=result['searches'][variant]['seconds']
                elif variant in ('multistart','baseline25'):
                    names=[n for n in result['searches'] if n=='primary25' or (variant=='multistart' and n.startswith('extra'))]
                    row['search_stop']=json.dumps({n:result['searches'][n]['reason'] for n in names});row['search_seconds']=sum(result['searches'][n]['seconds'] for n in names)
                elif variant.startswith('delete'):
                    row['search_stop']=result['searches']['delete_one_round']['reason'];row['search_seconds']=result['searches']['delete_one_round']['seconds']
                else:row['search_stop']='tree_bank_complete';row['search_seconds']=0.
                model=dict(job_id=job,variant=variant,point=p,model_signature=signature(p));selected.append(model)
                if variant in AC_METHODS:
                    key=job+'__'+signature(p)
                    tasks[key]=dict(key=key,job_id=job,point=p,model_signature=signature(p),N=payload['N'],labels=labels,
                        train=payload['train'],test=payload['test'],test_truth=payload['test_truth'],source=str(output/'source'))
            rows.append(row)
    cache={}
    if not skip_ac:
        folder=output/'ac';folder.mkdir(exist_ok=True);todo=[]
        for key,t in tasks.items():
            file=folder/(key+'.json')
            if file.exists() and 'error' not in read(file):cache[key]=read(file)
            else:todo.append(t)
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures={executor.submit(ac_task,t):t for t in todo}
            for future in as_completed(futures):
                t=futures[future]
                try:r=future.result()
                except Exception as e:r=dict(key=t['key'],model_signature=t['model_signature'],error=str(e),testing={'valid':False},fitting={'valid':False})
                cache[t['key']]=r;write(folder/(t['key']+'.json'),r)
    for row in rows:
        record=cache.get(row['job_id']+'__'+row.get('model_signature',''),{}) if row['ac_planned'] else {}
        row.update(ac_evaluated=bool(record),ac_valid=record.get('testing',{}).get('valid') if record else None,
            test_ac_rmse=record.get('testing',{}).get('relative_voltage_rmse'),physical_test_ac_rmse=record.get('physical',{}).get('relative_voltage_rmse'),
            ac_refit_success=record.get('fitting',{}).get('refit',{}).get('success'))
    frame=pd.DataFrame(rows)
    for reference in ('NJ','baseline25','multistart','restart_control'):
        lookup=frame[frame.variant==reference].set_index('job_id').clade_f1
        frame['delta_vs_'+reference]=frame.clade_f1-frame.job_id.map(lookup)
    frame.to_csv(destination/'per_job.csv',index=False,encoding='utf-8-sig')
    write(destination/'datasets.json',datasets)
    with (destination/'selected_models.jsonl').open('w',encoding='utf-8') as stream:
        for model in selected:stream.write(json.dumps(model,ensure_ascii=False)+'\n')
    main=frame[frame.panel=='main'];target=frame[frame.panel=='targeted']
    def aggregate(data,groups):
        return data.groupby(groups).agg(rows=('job_id','count'),models=('model_available','sum'),F1=('clade_f1','mean'),
            exact=('clade_f1',lambda x:int((x>=1-1e-12).sum())),delta_vs_multistart=('delta_vs_multistart','mean'),
            wins=('delta_vs_multistart',lambda x:int((x>1e-10).sum())),ties=('delta_vs_multistart',lambda x:int((abs(x)<=1e-10).sum())),
            losses=('delta_vs_multistart',lambda x:int((x < -1e-10).sum())),delta_vs_restart=('delta_vs_restart_control','mean'),
            pool_recall=('candidate_pool_true_recall','mean'),R_error=('R_error','mean'),X_error=('X_error','mean'),linear_test_mae=('linear_test_mae','mean'),
            AC_planned=('ac_planned','sum'),AC_evaluated=('ac_evaluated','sum'),AC_valid=('ac_valid',lambda x:int(x.eq(True).sum())),test_ac_rmse=('test_ac_rmse','mean'))
    aggregate(main,['variant']).to_csv(destination/'main_summary.csv',encoding='utf-8-sig')
    aggregate(main,['voltage_sigma','N','variant']).to_csv(destination/'by_noise_N.csv',encoding='utf-8-sig')
    aggregate(main,['bundle','variant']).to_csv(destination/'by_bundle.csv',encoding='utf-8-sig')
    aggregate(main,['case','voltage_sigma','N','variant']).to_csv(destination/'by_case_noise_N.csv',encoding='utf-8-sig')
    if len(target):aggregate(target,['variant']).to_csv(destination/'target_summary.csv',encoding='utf-8-sig')
    frame[abs(frame.delta_vs_multistart)>1e-10].to_csv(destination/'changes_vs_multistart.csv',index=False,encoding='utf-8-sig')
    frame[frame.clade_f1<1-1e-10].to_csv(destination/'unidentified.csv',index=False,encoding='utf-8-sig')
    file=destination/'catalog.sqlite';temporary=file.with_suffix('.tmp')
    with sqlite3.connect(temporary) as connection:
        pd.DataFrame(datasets).to_sql('datasets',connection,index=False,if_exists='replace')
        frame.drop(columns=['case','condition','N','bundle','voltage_sigma','pq_sigma']).to_sql('selected_models',connection,index=False,if_exists='replace')
        connection.execute('CREATE UNIQUE INDEX IF NOT EXISTS model_key ON selected_models(job_id,variant)')
        connection.execute('CREATE UNIQUE INDEX IF NOT EXISTS input_key ON datasets(job_id)')
        pd.DataFrame([dict(key='protocol',value=json.dumps(protocol)),dict(key='output',value=str(output))]).to_sql('metadata',connection,index=False,if_exists='replace')
    connection.close();temporary.replace(file)
    overview=dict(planned_inputs=protocol['planned'],evaluated_inputs=len(datasets),returned_inputs=len(datasets)-len(incomplete),incomplete_inputs=incomplete,
        main_rows=len(main),target_rows=len(target),models=int(frame.model_available.sum()),selected_records=len(frame),
        main_F1=main.groupby('variant').clade_f1.mean().to_dict(),new_truth_tuning=False,AC_unique_models=len(tasks))
    write(destination/'overview.json',overview);print(overview,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('destination');p.add_argument('--partial',action='store_true');p.add_argument('--skip-ac',action='store_true')
    a=p.parse_args();evaluate(a.output,a.destination,partial=a.partial,skip_ac=a.skip_ac)
