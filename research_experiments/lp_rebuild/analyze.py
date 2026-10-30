"""Independent audits and post-freeze scoring; no test-based parameter choice."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .run import ROOT,PARENT,read,write,sha
from .engine import signal_coefficients,criterion,local_families
from research_experiments.rebuild_benchmark.engine import prepare,Bank,regions
from research_experiments.rebuild_benchmark.analyze import score
from research_experiments.local_rebuild.model import compatible
from research_experiments.local_rebuild.verify import independent_loss
from research_experiments.wzzt_paper.common import observed
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth


def old_diagnostics(output):
    output=Path(output).resolve();destination=output/'analysis';destination.mkdir(exist_ok=True)
    meta=read(PARENT/'evaluation_catalog.json');coverage=[];strength=[];witnesses=[];old_jobs=set(read(PARENT/'protocol.json')['evaluation'])
    for job,p in meta.items():
        if p['panel']!='main' or job not in old_jobs:continue
        common=read(PARENT/'common'/job/'result.json');labels=common['labels']
        r=read(PARENT/'evaluation'/job/'rebuild_RNJ/result.json');point=r['trajectory']['point']
        truth=physical_truth(Path(p['truth_network']),labels);cal=common['calibration']
        initial=set(map(tuple,r['trajectory']['path'][0]['supports']))
        for s,rr,xx in zip(point['supports'],point['r'],point['x']):
            if 1<len(s)<len(labels):strength.append(dict(job=job,correct=frozenset(labels[i] for i in s) in truth['clades'],
                added=tuple(s) not in initial,normalized_depth=(cal['a']*rr+cal['b']*xx)/cal['median_root_depth']))
        if p['bundle'] not in (2,3) or p['N']!=48 or p['voltage_sigma'] not in (.0002,.001):continue
        train=prepare(observed(p['train'],p['N']));bank=Bank(train,common['pool'])
        try:
            best=None;count=0
            for c in tuple(bank.lp.pool):
                if len(c)<2 or list(c) in point['supports'] or c in point['supports']:continue
                family=[s for s in point['supports'] if compatible(s,c)]
                q=bank.fit([*family,c]);count+=1
                if best is None or q['objective']<best[1]['objective']-1e-12:best=c,q
            covered=False;local_domains=[]
            for v,blocks,virtual in regions(point['supports'],train.n,4,np.asarray(common['kernel'])):
                augmented=[*point['supports'],v] if virtual and list(v) not in point['supports'] and v not in point['supports'] else point['supports']
                from research_experiments.local_rebuild.model import build_model
                _,m=build_model(bank.scaled,augmented,v,blocks,bound=2.)
                local_domains.append(m)
            if best is not None:
                target=set(map(tuple,best[1]['supports']))
                covered=any(target<=set(m['all_supports']) and all(compatible(s,t) for s in target for t in m['keep']) for m in local_domains)
            gain=None if best is None else point['objective']-best[1]['objective'];gate=1e-10+.001*point['objective']
            coverage.append(dict(job=job,checked_pool_moves=count,regions=len(local_domains),best_candidate=None if best is None else best[0],
                gain=gain,threshold=gate,exceeds_gate=gain is not None and gain>gate,covered_by_one_local_region=covered,
                joint_stop=r['trajectory']['stop_reason']))
            witnesses.append(dict(job=job,train_sha256=sha(p['train']),source_result_sha256=sha(PARENT/'evaluation'/job/'rebuild_RNJ/result.json'),
                current=point,best_point=None if best is None else best[1],LP=bank.lp.solves,
                domains=[{k:m[k] for k in ('keep','released','candidates','region','blocks')} for m in local_domains]))
        finally:bank.close()
    pd.DataFrame(coverage).to_csv(destination/'old_domain_coverage.csv',index=False)
    pd.DataFrame(strength).to_csv(destination/'old_edge_strength.csv',index=False)
    write(destination/'old_domain_witnesses.json',dict(cases=witnesses,scope='training-only post-hoc diagnosis of earlier endpoints; no new tuning/evaluation observations'))
    print('OLD DIAGNOSTIC',len(coverage),'complete pool scans',len(strength),'internal supports',flush=True)


def check_floor(point,data,cal,config,reference):
    checks=0
    for s,r,x in zip(point['supports'],point['r'],point['x']):
        if not 1<len(s)<data.n or config['floor']==0:continue
        if config['mode']=='depth':value=(cal['a']*r+cal['b']*x)/cal['median_root_depth']
        else:
            gr,gx=signal_coefficients(data.p[0],data.q[0],tuple(s),cal);value=(gr*r+gx*x)/reference
            z=np.zeros(data.n);z[list(s)]=1.
            effect=(r*data.p[0][:,s].sum(axis=1)+x*data.q[0][:,s].sum(axis=1))[:,None]*z
            effect=effect-effect.mean(axis=0)
            assert np.sqrt(np.mean(effect**2))+1e-8>=config['floor']*reference
        assert value+1e-7>=config['floor'];checks+=1
    return checks


def evaluate(output):
    from rnj_wzzt.estimation import laminar_l1_milp as core
    output=Path(output).resolve();destination=output/'analysis';destination.mkdir(exist_ok=True)
    protocol=read(output/'protocol.json');frozen=read(output/'frozen.json');meta=read(output/'catalog.json')
    completion=read(output/'evaluation_completion.json');checks=0;records=[];failures=[];events=[]
    assert completion['planned']==len(protocol['evaluation'])*len(frozen['evaluation_settings'])
    assert frozen['protocol_sha256']==sha(output/'protocol.json')
    for name,digest in protocol['source_hashes'].items():assert sha(output/'source'/name)==digest;checks+=1
    for job in protocol['evaluation']:
        p=meta[job];common=read(output/'common'/job/'result.json');labels=common['labels']
        data={s:prepare(observed(p[s],p['N'] if s=='train' else None)) for s in ('train','validation','test')}
        truth=physical_truth(Path(p['truth_network']),labels);pool=tuple(map(tuple,common['pool']));results={}
        for cid,config in frozen['evaluation_settings'].items():
            file=output/'evaluation'/job/cid/'result.json';r=read(file) if file.exists() else None;results[cid]=r
            if r is None:
                errorfile=file.with_name('error.json');failures.append(dict(job=job,config_id=cid,error=read(errorfile) if errorfile.exists() else 'missing result'));continue
            assert not r['truth_used'] and not r['test_opened'] and r['frozen_sha256']==sha(output/'frozen.json')
            assert sha(output/'common'/job/'result.json')==r['common_sha256']
            assert r['input_hashes']['train']==sha(p['train']) and r['input_hashes']['validation']==sha(p['validation']);checks+=6
            previous=None
            for point in r['trajectory']['path']:
                assert core.is_laminar_family(point['supports'])
                assert all(np.isfinite(v) and -1e-9<=v<=2.+1e-9 for v in [*point['r'],*point['x']]);checks+=2
                value=independent_loss(data['train'].p[0],data['train'].q[0],data['train'].target[0],point)
                assert abs(value-point['objective'])<1e-8
                obj=criterion(point,data['train'].n,config['sparsity']*r['reference_MAE'])
                if previous is not None:assert obj<previous-1e-10
                previous=obj;checks+=1+check_floor(point,data['train'],common['calibration'],config,r['reference_MAE'])
            for point in (r['selected']['point'],r['selected_path']['point']):
                assert core.is_laminar_family(point['supports']);checks+=1
                checks+=check_floor(point,data['train'],common['calibration'],config,r['reference_MAE'])
            for q in r['LP']:
                assert q['status'] in (2,3)
                if q['status']==2:assert max(q['max_constraint_violation'],q['max_bound_violation'],q.get('response_floor_violation',0.))<=1e-7
                checks+=1
            for event in r['trajectory']['attempts']:
                if 'status' in event:
                    assert event['warm_start']['valid'] and event['audit']['valid'] and event['bound_consistent_with_known_upper'];checks+=3
                    if event.get('dual_bound') is not None and event.get('decoded_objective') is not None:
                        assert event['dual_bound']<=event['decoded_objective']+1e-8;checks+=1
                if event.get('kind')=='joint' and event['complete']:
                    assert event['checked']==event['candidate_families'] and event['lower_bound'] is not None;checks+=1
                events.append(dict(job=job,config_id=cid,**event))
        for method,cid in frozen['methods'].items():
            r=results[cid]
            for view,key in (('bank','selected'),('path','selected_path'),('raw',None)):
                point=r[key]['point'] if r and key else r['trajectory']['point'] if r else None
                row=dict(job=job,case=p['case'],bundle=p['bundle'],voltage_sigma=p['voltage_sigma'],method=method,config_id=cid,view=view,
                    **score(point,data,truth,labels,pool))
                if r:row.update(search_seconds=r['trajectory']['seconds'],elapsed_seconds=r['elapsed_seconds'],
                    total_estimate_seconds=r['elapsed_seconds']+r['common_seconds'],LPs=len(r['LP']),
                    infeasible_families=sum(q['status']==3 for q in r['LP']),stop_reason=r['trajectory']['stop_reason'],
                    selected_from_common_bank=r['selected']['index']<r['bank_count'] if view=='bank' else False,
                    fallback=r['initial_fallback'],worker_sha256=sha(output/'evaluation'/job/cid/'result.json'))
                records.append(row)
    frame=pd.DataFrame(records);frame.to_csv(destination/'per_model.csv',index=False)
    summary=frame.groupby(['view','method','config_id']).agg(inputs=('job','count'),available=('model_available','sum'),F1=('clade_f1','mean'),
        exact=('exact','sum'),train_MAE=('train_mae','mean'),validation_MAE=('validation_mae','mean'),test_MAE=('test_mae','mean'),
        R_error=('R_relative','mean'),X_error=('X_relative','mean'),internal=('internal','mean'),search_median=('search_seconds','median'),
        total_estimate_median=('total_estimate_seconds','median')).reset_index()
    summary.to_csv(destination/'summary.csv',index=False)
    pairs=[]
    for view,g in frame.groupby('view'):
        pivot=g.pivot(index='job',columns='method',values='clade_f1')
        for method in ('enum_control','best_floor','best_joint','best_hybrid'):
            for reference in ('pool','native_joint','native_corrected','original_fullwarm','enum_control'):
                if method==reference:continue
                d=pivot[method]-pivot[reference]
                pairs.append(dict(view=view,method=method,reference=reference,inputs=len(d),mean_delta=d.mean(),
                    wins=int((d>1e-10).sum()),ties=int((d.abs()<=1e-10).sum()),losses=int((d< -1e-10).sum())))
    pd.DataFrame(pairs).to_csv(destination/'paired.csv',index=False)
    frame[frame.view=='bank'].groupby(['case','method']).agg(inputs=('job','count'),F1=('clade_f1','mean'),exact=('exact','sum')).reset_index().to_csv(destination/'by_case.csv',index=False)
    frame[frame.view=='bank'].groupby(['bundle','method']).agg(inputs=('job','count'),F1=('clade_f1','mean'),exact=('exact','sum')).reset_index().to_csv(destination/'by_bundle.csv',index=False)
    write(destination/'events.json',events);write(destination/'verification.json',dict(passed=True,checks=checks,planned=completion['planned'],returned=completion['returned'],
        failures=failures,protocol_sha256=sha(output/'protocol.json'),frozen_sha256=sha(output/'frozen.json'),
        analysis_source_sha256=sha(__file__),
        alias_configurations=frozen['methods'],independence='two bundle blocks, conditions paired; identical selected configurations are aliases, not additional trials'))
    print('EVALUATED',len(frame),'rows',checks,'checks',len(failures),'failures',flush=True)
    print(summary[summary.view=='bank'].to_string(index=False),flush=True)


def crosssolver(output):
    from scipy.optimize import linprog
    from scipy.sparse import csc_matrix,eye,hstack,vstack
    from rnj_wzzt.estimation import laminar_l1_milp as core
    output=Path(output).resolve();protocol=read(output/'protocol.json');frozen=read(output/'frozen.json');meta=read(output/'catalog.json');records=[]
    for job in protocol['evaluation']:
        p=meta[job];data=prepare(observed(p['train'],p['N']));common=read(output/'common'/job/'result.json');cal=common['calibration']
        for method in ('best_floor','best_hybrid'):
            cid=frozen['methods'][method];r=read(output/'evaluation'/job/cid/'result.json');point=r['selected']['point'];config=r['config']
            supports=point['supports'];fp,fq=core._fixed_atom_features(data,supports);features=np.column_stack((fp,fq));y=data.target[0].ravel();m,k=features.shape[0],len(supports)
            identity=eye(m,format='csc');A=vstack([hstack([csc_matrix(features),-identity]),hstack([-csc_matrix(features),-identity])],format='csc');rhs=np.r_[y,-y]
            rows=[];extra=[]
            for i,s in enumerate(supports):
                if not 1<len(s)<data.n or config['floor']==0:continue
                if config['mode']=='depth':gr,gx=cal['a']/cal['median_root_depth'],cal['b']/cal['median_root_depth']
                else:gr,gx=[v/r['reference_MAE'] for v in signal_coefficients(data.p[0],data.q[0],tuple(s),cal)]
                row=np.zeros(2*k+m);row[i]=-gr;row[k+i]=-gx;rows.append(row);extra.append(-config['floor'])
            if rows:A=vstack([A,csc_matrix(np.asarray(rows))],format='csc');rhs=np.r_[rhs,extra]
            q=linprog(np.r_[np.zeros(2*k),np.ones(m)/m],A_ub=A,b_ub=rhs,bounds=[(0,2)]*(2*k)+[(0,None)]*m,
                method='highs-ds',options=dict(time_limit=20.,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            assert q.success and q.status==0 and abs(q.fun-point['objective'])<=1e-8
            violation=float(np.max(A@q.x-rhs,initial=0.));assert violation<=1e-7
            records.append(dict(job=job,method=method,config_id=cid,status=int(q.status),Gurobi_objective=point['objective'],
                HiGHS_objective=float(q.fun),difference=float(q.fun-point['objective']),maximum_violation=violation))
    write(output/'analysis/crosssolver.json',dict(passed=True,records=records));print('CROSS SOLVER',len(records),'passed',flush=True)


def floor_path_diagnostic(output):
    from .engine import FloorBank
    output=Path(output).resolve();frozen=read(output/'frozen.json');meta=read(output/'catalog.json')
    f=pd.read_csv(output/'analysis/per_model.csv');f=f[f.view=='bank'].pivot(index='job',columns='method',values='clade_f1')
    jobs=sorted(f.index[f.best_floor<f.enum_control-1e-10]);records=[]
    for job in jobs:
        p=meta[job];common=read(output/'common'/job/'result.json');train=prepare(observed(p['train'],p['N']));val=prepare(observed(p['validation']))
        source={name:read(output/'evaluation'/job/frozen['methods'][name]/'result.json') for name in ('enum_control','best_floor')}
        comparisons=[]
        for fitname in ('enum_control','best_floor'):
            config=source[fitname]['config'];bank=FloorBank(train,common['pool'],common['calibration'],floor=config['floor'],mode=config['mode'],reference=source[fitname]['reference_MAE'])
            try:
                for familyname,r in source.items():
                    q=bank.fit(r['selected']['point']['supports'])
                    from research_experiments.rebuild_benchmark.engine import loss
                    comparisons.append(dict(constraint=fitname,family=familyname,point=q,validation_MAE=loss(val,q)))
            finally:bank.close()
        records.append(dict(job=job,post_hoc=True,selection='every held-out input where selected floor F1 is lower than no-floor enumeration; no parameter change',
            results=comparisons,selected_from_bank={k:r['selected']['index']<r['bank_count'] for k,r in source.items()}))
    write(output/'analysis/floor_path_counterexamples.json',dict(cases=records));print('FLOOR PATH DIAGNOSTIC',len(records),'cases',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('old','evaluation','crosssolver','floor_path'));parser.add_argument('output');a=parser.parse_args()
    dict(old=old_diagnostics,evaluation=evaluate,crosssolver=crosssolver,floor_path=floor_path_diagnostic)[a.command](a.output)
