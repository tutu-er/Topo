from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import json
import sqlite3
import sys
import numpy as np
import pandas as pd
from research_experiments.wzzt_systematic.common import read,write,sha,matrices,linear_loss
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth,clade_metrics,relative_error
from research_experiments.wzzt_fine.evaluate import signature
from .common import observed,MAIN,AC_METHODS,indices


def ac_task(t):
    sys.path.insert(0,t['source'])
    from research_experiments.wzzt_solver_tuning.ac_screen import ac_score
    p=t['point'];fit=ac_score(t['labels'],p['supports'],p['r'],p['x'],observed(t['train'],t['N']),refit=True,max_nfev=12)
    test=physical={'valid':False}
    if fit['valid']:
        test=ac_score(t['labels'],p['supports'],fit['r_refit'],fit['x_refit'],observed(t['test']),refit=False)
        physical=ac_score(t['labels'],p['supports'],fit['r_refit'],fit['x_refit'],observed(t['test_truth']),refit=False)
    return dict(key=t['key'],model_signature=t['model_signature'],fitting=fit,testing=test,physical=physical)


def evaluate(output,destination,*,partial=False,skip_ac=False):
    output=Path(output).resolve();destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    protocol=read(output/'protocol.json');rows=[];datasets=[];selected=[];truths={};tasks={};incomplete=[]
    for j,job in enumerate(protocol['jobs']):
        folder=output/'jobs'/job
        ended=any((folder/n).exists() for n in ('result.json','error.json','timeout.json'))
        if partial and not ended:continue
        p=read(folder/'payload.json');returned=(folder/'result.json').exists()
        error=read(folder/'error.json').get('error','') if (folder/'error.json').exists() else ''
        failure_kind='fixed_family_LP_time_limit' if 'fixed-family LP status 9' in error else 'worker_hard_limit' if (folder/'timeout.json').exists() else 'worker_error' if error else None
        result=read(folder/'result.json') if returned else read(folder/'checkpoint.json') if (folder/'checkpoint.json').exists() else {}
        if not returned:incomplete.append(job)
        labels=[];truth=None;noise={}
        if not p['data_error']:
            train,validation,test=observed(p['train'],p['N']),observed(p['validation']),observed(p['test'])
            true_train=observed(p['train_truth'],p['N']);labels=list(train[0].columns)
            if p['truth_network'] not in truths:truths[p['truth_network']]=physical_truth(Path(p['truth_network']),labels)
            truth=truths[p['truth_network']]
            for name,a,b in zip(('P','Q','V','root'),train,true_train):noise[name+'_realized_relative_std']=float(np.std((a.to_numpy()-b.to_numpy())/np.maximum(abs(b.to_numpy()),1e-12)))
        ds={k:p[k] for k in ('job_id','panel','case','condition','bundle','N','voltage_sigma','pq_sigma','truth_network','data_error')}
        ds.update(labels_json=json.dumps(labels),train_indices_json=json.dumps(indices(p['N']).tolist()),noise_stats_json=json.dumps(noise),failure_kind=failure_kind,failure_message=error)
        for split in ('train','validation','test'):
            ds[split+'_path']=p[split];ds[split+'_truth_path']=p[split+'_truth'];ds[split+'_sha256']=sha(p[split]) if Path(p[split]).exists() else None
        file=next((folder/n for n in ('result.json','checkpoint.json','error.json','timeout.json') if (folder/n).exists()),None)
        ds.update(result_path=str(file) if file else None,result_sha256=sha(file) if file else None);datasets.append(ds)
        for method in p['methods']:
            row={k:p[k] for k in ('job_id','panel','case','condition','bundle','N','voltage_sigma','pq_sigma')}
            row.update(variant=method,method_group='main' if method in MAIN else 'ablation',worker_returned=returned,
                model_available=method in result.get('models',{}),physical_data_valid=not bool(p['data_error']),clade_f1=0.,
                ac_planned=p['ac_planned'] and method in AC_METHODS,worker_failure_kind=failure_kind,**noise)
            if row['model_available'] and truth is not None:
                point=result['models'][method];R,X=matrices(len(labels),point['supports'],point['r'],point['x'])
                predicted={frozenset(labels[i] for i in s) for s in point['supports'] if 2<=len(s)<len(labels)}
                row.update(clade_metrics(predicted,truth['clades'],labels),R_error=relative_error(R,truth['R']),X_error=relative_error(X,truth['X']),
                    train_mae=linear_loss(train,R,X)['mae'],validation_mae=linear_loss(validation,R,X)['mae'],linear_test_mae=linear_loss(test,R,X)['mae'],
                    ranking_validation_mae=point.get('ranking_validation_mae'),support_count=len(point['supports']),selected_origin=point.get('origin'),
                    model_signature=signature(point),supports_json=json.dumps(point['supports']),r_json=json.dumps(point['r']),x_json=json.dumps(point['x']),
                    shared_worker_seconds=result.get('seconds'),base_pipeline_seconds=result.get('base_pipeline_seconds'))
                if abs(row['train_mae']-point['objective'])>1e-8 or abs(row['validation_mae']-point['validation_mae'])>1e-8:raise AssertionError('saved physical score mismatch')
                domain=result.get('domains',{}).get(method,result.get('augmentation',{}).get('pool',result['pool']) if method.startswith('ridge') else result['pool'])
                pooled={frozenset(labels[i] for i in s) for s in domain if 2<=len(s)<len(labels)}
                missing=truth['clades']-predicted;false=predicted-truth['clades']
                crossing=[dict(false=sorted(f),missed=sorted(t)) for f in false for t in missing if f&t and not(f<=t or t<=f)]
                row.update(pool_size=len(domain),candidate_pool_true_recall=len(pooled&truth['clades'])/len(truth['clades']),
                    selected_new_true_clades=sum(tuple(s) not in {tuple(t) for t in result['pool']} and frozenset(labels[i] for i in s) in truth['clades'] for s in point['supports']),
                    diagnostic_json=json.dumps(dict(false_clades=[sorted(s) for s in false],missed_true_clades=[sorted(s) for s in missing],
                        crossing_pairs=crossing,truth_used_only_posthoc=True)))
                name={'WZZT_leaf_path':'leaf25','WZZT_leaf_bank':'leaf25','multistart_guard0':'multistart_delete','ridge_guard0':'ridge_delete'}.get(method,method)
                search=result.get('searches',{}).get(name)
                if search:row.update(wzzt_search_status=search['reason'],search_seconds=search['seconds'])
                elif method in ('multistart','baseline25'):
                    found={k:v for k,v in result['searches'].items() if k=='primary25' or (method=='multistart' and k.startswith('extra'))}
                    row.update(wzzt_search_status=json.dumps({k:v['reason'] for k,v in found.items()}),search_seconds=sum(v['seconds'] for v in found.values()))
                else:row.update(wzzt_search_status='tree_bank_complete',search_seconds=0.)
                selected.append(dict(job_id=job,variant=method,point=point,model_signature=signature(point)))
                if row['ac_planned']:
                    key=job+'__'+signature(point);tasks[key]=dict(key=key,point=point,model_signature=signature(point),N=p['N'],labels=labels,
                        train=p['train'],test=p['test'],test_truth=p['test_truth'],source=str(output/'source'))
            rows.append(row)
        if (j+1)%200==0:print('READ',j+1,flush=True)
    cache={}
    if not skip_ac:
        acdir=output/'ac';acdir.mkdir(exist_ok=True);todo=[]
        for key,t in tasks.items():
            file=acdir/(key+'.json')
            if file.exists() and 'error' not in read(file):cache[key]=read(file)
            else:todo.append(t)
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures={executor.submit(ac_task,t):t for t in todo}
            for i,future in enumerate(as_completed(futures)):
                t=futures[future]
                try:value=future.result()
                except Exception as e:value=dict(key=t['key'],model_signature=t['model_signature'],error=str(e),testing={'valid':False},fitting={'valid':False})
                cache[t['key']]=value;write(acdir/(t['key']+'.json'),value)
                if (i+1)%50==0:print('AC',i+1,'/',len(todo),flush=True)
    for row in rows:
        a=cache.get(row['job_id']+'__'+row.get('model_signature',''),{}) if row['ac_planned'] else {}
        row.update(ac_evaluated=bool(a),ac_valid=a.get('testing',{}).get('valid') if a else None,
            test_ac_rmse=a.get('testing',{}).get('relative_voltage_rmse'),physical_test_ac_rmse=a.get('physical',{}).get('relative_voltage_rmse'),
            ac_refit_success=a.get('fitting',{}).get('refit',{}).get('success'))
    frame=pd.DataFrame(rows)
    for method in ('NJ','baseline25','multistart','restart_control','ridge_union'):
        lookup=frame[frame.variant==method].set_index('job_id').clade_f1
        frame['delta_vs_'+method]=frame.clade_f1-frame.job_id.map(lookup)
    frame.to_csv(destination/'per_job.csv',index=False,encoding='utf-8-sig');write(destination/'datasets.json',datasets)
    with (destination/'selected_models.jsonl').open('w',encoding='utf-8') as s:
        for model in selected:s.write(json.dumps(model,ensure_ascii=False)+'\n')
    def aggregate(groups):
        return frame.groupby(groups).agg(rows=('job_id','count'),available=('model_available','sum'),F1=('clade_f1','mean'),
            exact=('clade_f1',lambda x:int((x>=1-1e-12).sum())),delta_vs_multistart=('delta_vs_multistart','mean'),
            wins=('delta_vs_multistart',lambda x:int((x>1e-10).sum())),ties=('delta_vs_multistart',lambda x:int((abs(x)<=1e-10).sum())),
            losses=('delta_vs_multistart',lambda x:int((x < -1e-10).sum())),R_error=('R_error','mean'),X_error=('X_error','mean'),
            test_mae=('linear_test_mae','mean'),AC_planned=('ac_planned','sum'),AC_evaluated=('ac_evaluated','sum'),
            AC_valid=('ac_valid',lambda x:int(x.eq(True).sum())),test_ac_rmse=('test_ac_rmse','mean'))
    for name,groups in (('panel_summary',['panel','variant']),('by_noise_N',['panel','voltage_sigma','N','variant']),
                        ('by_condition',['panel','condition','variant']),('by_bundle',['panel','bundle','variant']),('by_case',['panel','case','variant'])):
        aggregate(groups).to_csv(destination/(name+'.csv'),encoding='utf-8-sig')
    frame[frame.clade_f1<1-1e-12].to_csv(destination/'unidentified.csv',index=False,encoding='utf-8-sig')
    frame[abs(frame.delta_vs_multistart)>1e-10].to_csv(destination/'changes_vs_multistart.csv',index=False,encoding='utf-8-sig')
    file=destination/'catalog.sqlite';tmp=file.with_suffix('.tmp')
    with sqlite3.connect(tmp) as conn:
        pd.DataFrame(datasets).to_sql('datasets',conn,index=False,if_exists='replace')
        frame.drop(columns=['panel','case','condition','N','bundle','voltage_sigma','pq_sigma','physical_data_valid']).to_sql('selected_models',conn,index=False,if_exists='replace')
        conn.execute('CREATE UNIQUE INDEX model_key ON selected_models(job_id,variant)');conn.execute('CREATE UNIQUE INDEX input_key ON datasets(job_id)')
    conn.close()
    tmp.replace(file)
    overview=dict(planned_inputs=protocol['planned'],evaluated_inputs=len(datasets),returned_inputs=len(datasets)-len(incomplete),
        incomplete_inputs=incomplete,planned_model_rows=protocol['planned_model_rows'],model_rows=len(frame),available_models=int(frame.model_available.sum()),
        AC_planned=int(frame.ac_planned.sum()),AC_evaluated=int(frame.ac_evaluated.sum()),AC_valid=int(frame.ac_valid.eq(True).sum()),AC_unique_models=len(tasks),new_truth_tuning=False)
    overview['completed_inputs']=sum(any((output/'jobs'/d['job_id']/n).exists() for n in ('result.json','error.json','timeout.json')) for d in datasets)
    overview['error_inputs']=sum((output/'jobs'/d['job_id']/'error.json').exists() for d in datasets)
    overview['timeout_inputs']=sum((output/'jobs'/d['job_id']/'timeout.json').exists() for d in datasets)
    overview['fixed_LP_time_limit_inputs']=sum(d['failure_kind']=='fixed_family_LP_time_limit' for d in datasets)
    write(destination/'overview.json',overview);print(overview,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('destination');p.add_argument('--partial',action='store_true');p.add_argument('--skip-ac',action='store_true')
    a=p.parse_args();evaluate(a.output,a.destination,partial=a.partial,skip_ac=a.skip_ac)
