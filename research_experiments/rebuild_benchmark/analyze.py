"""Frozen-parameter held-out scoring; never selects by test loss or truth."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import pandas as pd

from .run import ROOT,METHODS,read,sha,write
from .engine import prepare,loss
from research_experiments.wzzt_paper.common import observed
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth,clade_metrics
from research_experiments.local_rebuild.verify import independent_loss
from rnj_wzzt.estimation import laminar_l1_milp as core


def score(point,data,truth,labels,pool):
    if point is None:return dict(model_available=False,clade_f1=0.,exact=False)
    for name,d in data.items():
        value=independent_loss(d.p[0],d.q[0],d.target[0],point)
        if name=='train' and 'objective' in point and abs(value-point['objective'])>1e-8:
            raise AssertionError('saved physical training objective mismatch')
    predicted={frozenset(labels[i] for i in s) for s,r,x in zip(point['supports'],point['r'],point['x'])
               if 1<len(s)<len(labels) and max(r,x)>1e-9}
    pos={label:i for i,label in enumerate(labels)}
    truepos={tuple(sorted(pos[v] for v in s)) for s in truth['clades']}
    predpos={tuple(sorted(pos[v] for v in s)) for s in predicted}
    R,X=core.build_matrices_from_atoms(len(labels),point['supports'],point['r'],point['x'])
    return dict(model_available=True,**clade_metrics(predicted,truth['clades'],labels),
        exact=predicted==truth['clades'],train_mae=loss(data['train'],point),validation_mae=loss(data['validation'],point),
        test_mae=loss(data['test'],point),R_relative=float(np.linalg.norm(R-truth['R'])/np.linalg.norm(truth['R'])),
        X_relative=float(np.linalg.norm(X-truth['X'])/np.linalg.norm(truth['X'])),
        internal=len(predicted),candidate_pool_true_recall=len(set(pool)&truepos)/len(truepos) if truepos else 1.,
        novel_true=len((predpos&truepos)-set(pool)),novel_false=len((predpos-truepos)-set(pool)),
        bound_active=bool(max([*point['r'],*point['x']],default=0.)>=1.999))


def views(result,params,shared):
    own=params['selected']['parameters'];path=params['selected_path']['parameters']
    key=lambda p:f"{p['resolution']:g}_{p['validation_fraction']:g}"
    return dict(tuned_bank=result['selected'][key(own)]['point'],
        tuned_path=result['selected_path'][key(path)]['point'],
        shared_bank=result['selected'][key(shared)]['point'],default_bank=result['selected']['0.16_0.03']['point'],
        legacy_bank=result['legacy'],raw_endpoint=result['trajectory']['point'])


def aggregate(frame):
    result=[]
    for (panel,view,method),g in frame.groupby(['panel','view','method']):
        available=g[g.model_available]
        row=dict(panel=panel,view=view,method=method,inputs=len(g),available=int(g.model_available.sum()),
                 F1=float(g.clade_f1.mean()),exact=int(g.exact.sum()))
        for name in ('train_mae','validation_mae','test_mae','R_relative','X_relative','internal',
                     'search_seconds','elapsed_seconds','standalone_estimate_seconds','candidate_pool_true_recall','novel_true','novel_false'):
            if name in g:row[name]=float(available[name].mean()) if len(available) else None
        result.append(row)
    return pd.DataFrame(result)


def paired(frame):
    rows=[];blocks=[]
    for (panel,view),g in frame.groupby(['panel','view']):
        pivot=g.pivot(index='job',columns='method',values='clade_f1')
        for reference in METHODS[:-1]:
            delta=pivot['rebuild_RNJ']-pivot[reference]
            rows.append(dict(panel=panel,view=view,reference=reference,inputs=len(delta),mean_delta=float(delta.mean()),
                wins=int((delta>1e-10).sum()),ties=int((abs(delta)<=1e-10).sum()),losses=int((delta< -1e-10).sum())))
            if panel=='main':
                for bundle in (2,3,4):
                    jobs=set(g[g.bundle==bundle].job)
                    values=delta.loc[list(jobs)]
                    blocks.append(dict(panel=panel,view=view,reference=reference,bundle=bundle,inputs=len(values),mean_delta=float(values.mean())))
    return pd.DataFrame(rows),pd.DataFrame(blocks)


def analyze(output,destination):
    output=Path(output).resolve();destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    protocol=read(output/'protocol.json');frozen=read(output/'frozen_method_parameters.json');shared=read(output/'frozen_parameters.json')['parameters']
    completion=read(output/'evaluation_completion.json')
    assert completion['planned']==len(protocol['evaluation'])*len(METHODS)
    metadata=read(output/'evaluation_catalog.json');records=[];checks=0;events=[];failures=[]
    source=output/'source'
    for file,digest in protocol['source_hashes'].items():assert sha(source/file)==digest;checks+=1
    assert not set(protocol['development'])&set(protocol['evaluation'])
    truth_cache={};data_cache={}
    for job in protocol['evaluation']:
        p=metadata[job];common_file=output/'common'/job/'result.json'
        common=read(common_file) if common_file.exists() else None
        data={s:prepare(observed(p[s],p['N'] if s=='train' else None)) for s in ('train','validation','test')}
        labels=list(data['train'].labels)
        if p['truth_network'] not in truth_cache:truth_cache[p['truth_network']]=physical_truth(Path(p['truth_network']),labels)
        truth=truth_cache[p['truth_network']];pool=tuple(tuple(s) for s in common['pool']) if common else ()
        header={k:p[k] for k in ('panel','case','condition','bundle','N','voltage_sigma','pq_sigma')}
        for method in METHODS:
            directory=output/'evaluation'/job/method;file=directory/'result.json'
            r=read(file) if file.exists() else None
            if r:
                payload=read(directory/'payload.json')
                assert payload['frozen_sha256']==sha(output/'frozen_parameters.json')
                assert payload['frozen_method_sha256']==sha(output/'frozen_method_parameters.json')
                assert r['truth_used'] is False and r['test_opened'] is False and sha(common_file)==r['common_sha256']
                assert sha(p['train'])==r['input_hashes']['train'] and sha(p['validation'])==r['input_hashes']['validation']
                checks+=6
                previous=None
                for point in r['trajectory']['path']:
                    value=independent_loss(data['train'].p[0],data['train'].q[0],data['train'].target[0],point)
                    assert abs(value-point['objective'])<=1e-8
                    if previous is not None:
                        assert value<previous['objective']-1e-10
                        if method.startswith('original'):assert set(map(tuple,previous['supports']))<=set(map(tuple,point['supports']))
                    previous=point;checks+=1
                for e in r['trajectory']['attempts']:
                    if 'status' not in e:continue
                    assert e['warm_start']['valid'] and e['bound_consistent_with_known_upper']
                    if e['decoded_objective'] is not None and e['dual_bound'] is not None:
                        assert e['dual_bound']<=e['decoded_objective']+1e-8
                    events.append(dict(job=job,method=method,panel=p['panel'],status=e['status'],
                        objective=e['objective'],dual_bound=e['dual_bound'],gap=e['relative_gap'],accepted=e['accepted'],
                        seconds=e['seconds'],variables=e['variable_count'],binaries=e['binary_count'],constraints=e['constraint_count'],
                        max_violation=e['audit'].get('maximum_violation'),incumbent_valid=e['audit']['valid']))
                    checks+=2
                for certificate in r['LP']:
                    assert certificate['status']==2 and max(certificate['max_constraint_violation'],certificate['max_bound_violation'])<=1e-7
                    checks+=1
                choices=views(r,frozen['methods'][method],shared)
            else:
                choices={view:None for view in ('tuned_bank','tuned_path','shared_bank','default_bank','legacy_bank','raw_endpoint')}
                error=read(directory/'error.json') if (directory/'error.json').exists() else read(directory/'timeout.json') if (directory/'timeout.json').exists() else {'error':'missing result'}
                failure=dict(error);failure.update(job=job,method=method,shared_preprocessing_failure=common is None)
                log=directory/'worker.log'
                failure['last_log_line']=log.read_text(encoding='utf-8',errors='replace').strip().splitlines()[-1] if log.exists() else None
                if common is None:
                    commonlog=output/'common'/job/'worker.log'
                    failure['frontend_last_log_line']=commonlog.read_text(encoding='utf-8',errors='replace').strip().splitlines()[-1] if commonlog.exists() else None
                failures.append(failure)
            for view,point in choices.items():
                row=dict(header,job=job,method=method,view=view,**score(point,data,truth,labels,pool))
                if r:row.update(search_seconds=r['trajectory']['seconds'],elapsed_seconds=r['elapsed_seconds'],
                    common_seconds=r['common_seconds'],standalone_estimate_seconds=r['elapsed_seconds']+r['common_seconds'],
                    stop_reason=r['trajectory']['stop_reason'],MIP_count=sum('status' in a for a in r['trajectory']['attempts']),
                    MIP_timeouts=sum(a.get('status')==9 for a in r['trajectory']['attempts']),
                    LP_count=len(r['LP']),accepted_steps=len(r['trajectory']['path'])-1,
                    search_budget_overrun=max(0.,r['trajectory']['seconds']-20.),
                    worker_result_sha256=sha(file))
                if r:
                    q=r['trajectory']['path'][0]
                    row['initial_zero_structural_constraints']=sum(len(s)>1 and max(x,y)<=1e-9
                        for s,x,y in zip(q['supports'],q['r'],q['x']))
                    if view in ('tuned_bank','shared_bank','default_bank'):
                        policy=frozen['methods'][method]['selected']['parameters'] if view=='tuned_bank' else shared if view=='shared_bank' else dict(resolution=.16,validation_fraction=.03)
                        selected=r['selected'][f"{policy['resolution']:g}_{policy['validation_fraction']:g}"]
                        row['selected_from_RNJ_bank']=selected['index']<len(common['bank'])
                    elif view=='tuned_path':row['selected_from_RNJ_bank']=False
                records.append(row);checks+=3 if point else 1
    frame=pd.DataFrame(records);summary=aggregate(frame);contrasts,blocks=paired(frame)
    frame.to_csv(destination/'per_model.csv',index=False);summary.to_csv(destination/'summary.csv',index=False)
    contrasts.to_csv(destination/'paired.csv',index=False);blocks.to_csv(destination/'paired_bundle.csv',index=False)
    pd.DataFrame(events).to_csv(destination/'MIP_events.csv',index=False)
    main=frame[(frame.panel=='main')&(frame.view=='tuned_bank')]
    main.groupby(['voltage_sigma','N','method']).agg(inputs=('job','count'),F1=('clade_f1','mean'),exact=('exact','sum'),test_MAE=('test_mae','mean')).reset_index().to_csv(destination/'main_cells.csv',index=False)
    main.groupby(['case','method']).agg(inputs=('job','count'),F1=('clade_f1','mean'),exact=('exact','sum')).reset_index().to_csv(destination/'main_cases.csv',index=False)
    pd.DataFrame(failures).to_json(destination/'failures.json',orient='records',force_ascii=False,indent=2)
    write(destination/'verification.json',dict(passed=True,checks=checks,planned_inputs=140,planned_workers=560,
        returned_workers=completion['returned'],worker_failures=len(failures),rows=len(frame),
        MIP_calls=len(events),MIP_timeouts=sum(e['status']==9 for e in events),
        source_manifest_sha256=sha(output/'protocol.json'),frozen_parameters_sha256=sha(output/'frozen_parameters.json'),
        frozen_method_parameters_sha256=sha(output/'frozen_method_parameters.json'),
        observation_scope='historical synthetic AC observations, held out from this tuning; not new field data',
        timing_scope='four concurrent single-thread searches; common fresh frontend added per model as an estimate, not a separate sequential timing replay',
        independence='main has three bundle blocks; N/noise cells within a block are paired, not independent repeats'))
    print('PRIMARY ANALYSIS',len(frame),'rows',len(failures),'failures',checks,'checks',flush=True)
    print(summary[(summary.panel=='main')&(summary.view=='tuned_bank')].to_string(index=False),flush=True)
    return frame,summary


def supplements(output,destination):
    output=Path(output).resolve();destination=Path(destination).resolve()
    metadata=read(output/'evaluation_catalog.json');params=read(output/'frozen_method_parameters.json');shared=read(output/'frozen_parameters.json')['parameters']
    extra=read(output/'analysis_addendum.json');records=[]
    for phase in ('budget','augmented_pool'):
        methods=('original_leaf','rebuild_RNJ') if phase=='budget' else ('pool_RNJ',)
        for job in extra['jobs']:
            p=metadata[job];common=read(output/'common'/job/'result.json');data={s:prepare(observed(p[s],p['N'] if s=='train' else None)) for s in ('train','validation','test')}
            truth=physical_truth(Path(p['truth_network']),common['labels']);pool=tuple(map(tuple,common['pool']))
            for method in methods:
                file=output/phase/job/method/'result.json';r=read(file) if file.exists() else None
                for view,point in (views(r,params['methods'][method],shared).items() if r else [('tuned_bank',None),('tuned_path',None)]):
                    row=dict(phase=phase,job=job,case=p['case'],method=method,view=view,**score(point,data,truth,common['labels'],pool))
                    if r:row.update(search_seconds=r['trajectory']['seconds'],elapsed_seconds=r['elapsed_seconds'],
                                    stop_reason=r['trajectory']['stop_reason'],MIP_timeouts=sum(a.get('status')==9 for a in r['trajectory']['attempts']))
                    records.append(row)
    pd.DataFrame(records).to_csv(destination/'supplement_per_model.csv',index=False)
    print('SUPPLEMENTS',len(records),'rows',flush=True)


def crosssolver(output,destination):
    from research_experiments.local_rebuild.matched import FiniteLP
    output=Path(output).resolve();destination=Path(destination).resolve();extra=read(output/'analysis_addendum.json')
    meta=read(output/'evaluation_catalog.json');params=read(output/'frozen_method_parameters.json');shared=read(output/'frozen_parameters.json')['parameters'];checks=[]
    for job in extra['jobs']:
        p=meta[job];data=prepare(observed(p['train'],p['N']))
        for method in ('original_leaf','rebuild_RNJ'):
            r=read(output/'evaluation'/job/method/'result.json');point=views(r,params['methods'][method],shared)['tuned_bank']
            bank=FiniteLP(data,point['supports'],mandatory=point['supports'],seconds=20.)
            refit=bank.fit(point['supports']);difference=refit['objective']-point['objective']
            assert abs(difference)<=1e-8
            checks.append(dict(job=job,method=method,Gurobi_objective=point['objective'],HiGHS_objective=refit['objective'],difference=difference,certificates=bank.solves))
    write(destination/'crosssolver.json',dict(passed=True,checks=checks));print('CROSS SOLVER',len(checks),'passed',flush=True)


def historical(output,destination):
    from .engine import Bank
    from .run import validation_candidates,choose
    output=Path(output).resolve();destination=Path(destination).resolve();extra=read(output/'analysis_addendum.json')
    meta=read(output/'evaluation_catalog.json');params=read(output/'frozen_method_parameters.json');records=[]
    for job in extra['jobs']:
        p=meta[job];common=read(output/'common'/job/'result.json');data={s:prepare(observed(p[s],p['N'] if s=='train' else None)) for s in ('train','validation','test')}
        directory=ROOT/'outputs/wzzt_paper_20261029_full/free_domain'/job;old=read(directory/'result.json');oldpayload=read(directory/'payload.json')
        assert sha(oldpayload['train'])==common['source_input_hashes']['train'] and sha(oldpayload['validation'])==common['source_input_hashes']['validation']
        assert old['truth_used_by_estimator'] is False
        truth=physical_truth(Path(p['truth_network']),common['labels']);pool=tuple(map(tuple,common['pool']))
        bank=Bank(data['train'],[s for q in [*common['bank'],*old['search']['path']] for s in q['supports']])
        try:
            for view,name in (('legacy_path','WZZT_free_path'),('legacy_bank','WZZT_free_bank')):
                records.append(dict(job=job,case=p['case'],view=view,**score(old['models'][name],data,truth,common['labels'],pool),search_seconds=old['search']['seconds'],source_sha256=sha(directory/'result.json')))
            for view,key,points in (('refit_path','selected_path',old['search']['path']),('refit_bank','selected',[*common['bank'],*old['search']['path']])):
                config=params['methods']['original_leaf'][key]['parameters']
                candidates=validation_candidates(bank,points,data['validation'],common['calibration'],config['resolution'])
                selected=choose(candidates,config['validation_fraction'])['point']
                records.append(dict(job=job,case=p['case'],view=view,**score(selected,data,truth,common['labels'],pool),search_seconds=old['search']['seconds'],source_sha256=sha(directory/'result.json')))
        finally:bank.close()
    pd.DataFrame(records).to_csv(destination/'historical_reference.csv',index=False);print('HISTORICAL',len(records),'rows',flush=True)


def additional(output,destination):
    """Keep post-primary controls and LP-budget recovery outside primary rows."""
    output=Path(output).resolve();destination=Path(destination).resolve()
    meta=read(output/'evaluation_catalog.json');params=read(output/'frozen_method_parameters.json')
    shared=read(output/'frozen_parameters.json')['parameters'];records=[];events=[];failures=[];checks=0
    for phase in ('recovery','fullwarm'):
        protocol=read(output/f'{phase}_protocol.json')
        assert protocol.get('parameters_sha256',protocol.get('frozen_parameters_sha256'))==sha(output/'frozen_method_parameters.json')
        for name,digest in protocol['source_hashes'].items():
            assert sha(output/f'source_{phase}'/'research_experiments/rebuild_benchmark'/name)==digest
        methods=METHODS if phase=='recovery' else ('original_leaf',)
        for job in protocol['jobs']:
            p=meta[job];commonfile=output/('recovery_common' if phase=='recovery' else 'common')/job/'result.json'
            common=read(commonfile) if commonfile.exists() else None
            data={s:prepare(observed(p[s],p['N'] if s=='train' else None)) for s in ('train','validation','test')}
            labels=list(data['train'].labels);truth=physical_truth(Path(p['truth_network']),labels)
            pool=tuple(map(tuple,common['pool'])) if common else ()
            for method in methods:
                directory=output/phase/job/method;file=directory/'result.json';r=read(file) if file.exists() else None
                if r:
                    assert not r['truth_used'] and not r['test_opened'] and sha(commonfile)==r['common_sha256']
                    assert r['input_hashes']==common['source_input_hashes'];checks+=4
                    previous=None
                    for point in r['trajectory']['path']:
                        value=independent_loss(data['train'].p[0],data['train'].q[0],data['train'].target[0],point)
                        assert abs(value-point['objective'])<=1e-8
                        if previous is not None:assert value<previous['objective']-1e-10
                        previous=point;checks+=1
                    for e in r['trajectory']['attempts']:
                        if 'status' not in e:continue
                        assert e['warm_start']['valid'] and e['bound_consistent_with_known_upper']
                        assert e['dual_bound'] is None or e['decoded_objective'] is None or e['dual_bound']<=e['decoded_objective']+1e-8
                        events.append(dict(phase=phase,job=job,method=method,status=e['status'],accepted=e['accepted'],
                            gap=e['relative_gap'],incumbent_valid=e['audit']['valid']));checks+=2
                    for q in r['LP']:
                        assert q['status']==2 and max(q['max_constraint_violation'],q['max_bound_violation'])<=1e-7;checks+=1
                    choices=views(r,params['methods'][method],shared)
                else:
                    choices={'tuned_bank':None,'tuned_path':None}
                    failure=dict(phase=phase,job=job,method=method,shared_preprocessing_failure=common is None)
                    if (directory/'error.json').exists():failure.update(read(directory/'error.json'))
                    failures.append(failure)
                for view,point in choices.items():
                    row=dict(phase=phase,job=job,method=method,view=view,panel=p['panel'],N=p['N'],voltage_sigma=p['voltage_sigma'],
                        **score(point,data,truth,labels,pool))
                    if r:row.update(search_seconds=r['trajectory']['seconds'],elapsed_seconds=r['elapsed_seconds'],
                        stop_reason=r['trajectory']['stop_reason'],MIP_timeouts=sum(a.get('status')==9 for a in r['trajectory']['attempts']))
                    records.append(row)
    frame=pd.DataFrame(records);frame.to_csv(destination/'additional_per_model.csv',index=False)
    frame.groupby(['phase','panel','view','method']).agg(inputs=('job','count'),available=('model_available','sum'),
        F1=('clade_f1','mean'),exact=('exact','sum'),test_mae=('test_mae','mean')).reset_index().to_csv(destination/'additional_summary.csv',index=False)
    pd.DataFrame(events).to_csv(destination/'additional_MIP_events.csv',index=False)
    write(destination/'additional_verification.json',dict(passed=True,checks=checks,failures=failures,
        primary_rows_replaced=0,fullwarm_declared_after_primary_scoring=True))
    print('ADDITIONAL',len(frame),'rows',len(failures),'failures',checks,'checks',flush=True)


def diagnostics(output,destination):
    from .engine import Bank
    from .run import validation_candidates,choose
    output=Path(output).resolve();destination=Path(destination).resolve();meta=read(output/'evaluation_catalog.json')
    params=read(output/'frozen_method_parameters.json');raw=[];bankrows=[];steps=[]
    final=pd.read_csv(destination/'per_model.csv');final=final[(final.panel=='main')&(final.view=='tuned_bank')].set_index(['job','method'])
    for job,p in meta.items():
        if job not in set(final.index.get_level_values('job')):continue
        common=read(output/'common'/job/'result.json');labels=common['labels'];pool=tuple(map(tuple,common['pool']))
        data={s:prepare(observed(p[s],p['N'] if s=='train' else None)) for s in ('train','validation','test')}
        truth=physical_truth(Path(p['truth_network']),labels)
        bank=Bank(data['train'],pool)
        try:
            candidates={}
            for method in METHODS:
                r=read(output/'evaluation'/job/method/'result.json');config=params['methods'][method]['selected']['parameters']
                resolution=config['resolution']
                if resolution not in candidates:candidates[resolution]=validation_candidates(bank,common['bank'],data['validation'],common['calibration'],resolution)
                bankpoint=choose(candidates[resolution],config['validation_fraction'])['point']
                baseline=score(bankpoint,data,truth,labels,pool);selected=final.loc[(job,method)]
                bankrows.append(dict(job=job,method=method,bank_F1=baseline['clade_f1'],final_F1=selected.clade_f1,
                    F1_delta=selected.clade_f1-baseline['clade_f1'],test_relative_change=selected.test_mae/baseline['test_mae']-1,
                    selected_from_bank=selected.selected_from_RNJ_bank))
                initial=score(r['trajectory']['path'][0],data,truth,labels,pool)
                endpoint=score(r['trajectory']['point'],data,truth,labels,pool)
                raw.append(dict(job=job,method=method,F1_before=initial['clade_f1'],F1_after=endpoint['clade_f1'],
                    F1_delta=endpoint['clade_f1']-initial['clade_f1'],
                    training_relative_change=endpoint['train_mae']/initial['train_mae']-1,
                    test_relative_change=endpoint['test_mae']/initial['test_mae']-1,
                    TP_delta=endpoint['matched_clades']-initial['matched_clades'],
                    FP_delta=(endpoint['predicted_clades']-endpoint['matched_clades'])-(initial['predicted_clades']-initial['matched_clades'])))
                previous=initial
                for k,point in enumerate(r['trajectory']['path'][1:],1):
                    q=score(point,data,truth,labels,pool)
                    steps.append(dict(job=job,method=method,step=k,F1_delta=q['clade_f1']-previous['clade_f1'],
                        training_relative_change=q['train_mae']/previous['train_mae']-1,
                        test_relative_change=q['test_mae']/previous['test_mae']-1,
                        TP_delta=q['matched_clades']-previous['matched_clades'],
                        FP_delta=(q['predicted_clades']-q['matched_clades'])-(previous['predicted_clades']-previous['matched_clades'])))
                    previous=q
        finally:bank.close()
    for name,rows in (('raw_changes',raw),('accepted_steps',steps),('bank_contribution',bankrows)):
        pd.DataFrame(rows).to_csv(destination/f'{name}.csv',index=False)
    print('DIAGNOSTICS',len(raw),'raw comparisons',len(steps),'accepted steps',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('primary','supplements','crosssolver','historical','additional','diagnostics'))
    parser.add_argument('output');parser.add_argument('destination');a=parser.parse_args()
    dict(primary=analyze,supplements=supplements,crosssolver=crosssolver,historical=historical,additional=additional,diagnostics=diagnostics)[a.command](a.output,a.destination)
