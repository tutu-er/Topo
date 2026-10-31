"""Post-freeze F1/exact-recovery scoring and independent regularized audits."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .run import PARENT,read,write,sha,result_choice,config_id
from .engine import criterion,count
from .diagnose import error_type
from research_experiments.joint_five.analyze import independent_lp,size_errors,supports
from research_experiments.rebuild_benchmark.engine import prepare
from research_experiments.rebuild_benchmark.analyze import score
from research_experiments.local_rebuild.verify import independent_loss
from research_experiments.local_rebuild.model import compatible
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
from research_experiments.wzzt_paper.common import observed


def audit_development(output):
    output=Path(output);protocol=read(output/'protocol.json');catalog=read(output/'catalog.json');checks=0
    for job in protocol['development']:
        meta=catalog[job];data=prepare(observed(meta['train'],meta['N']))
        for cid in protocol['settings']:
            r=read(output/'development'/job/cid/'result.json');t=r['trajectory'];previous=float('inf')
            for point in t['path']:
                value=independent_loss(data.p[0],data.q[0],data.target[0],point)
                assert abs(value-point['objective'])<1e-8
                score_value=criterion(point,data.n,t['penalty']);assert score_value<=previous+1e-9
                previous=score_value;checks+=2
            for event in t['attempts']:
                if event.get('status') is not None:
                    assert event['audit']['valid'] and event['warm_start']['valid'] and event['bound_consistent_with_known_upper'];checks+=3
                if event.get('regularized_bound') is not None:
                    assert event['regularized_bound']<=event['decoded_score']+1e-8;checks+=1
    write(output/'development_verification.json',dict(passed=True,checks=checks,searches=len(protocol['development'])*len(protocol['settings'])))
    return checks


def development_score(output):
    protocol=read(output/'protocol.json');catalog=read(output/'catalog.json');records=[]
    assert (output/'frozen.json').exists()
    for job in protocol['development']:
        meta=catalog[job];common=read(output/'common'/job/'result.json');labels=common['labels']
        data={s:prepare(observed(meta[s],meta['N'] if s=='train' else None)) for s in ('train','validation','test')}
        truth=physical_truth(Path(meta['truth_network']),labels)
        for mode in ('search_only','selection_only','both'):
            for coefficient in protocol['grid']:
                cid=config_id(0. if mode=='selection_only' else coefficient)
                result=read(output/'development'/job/cid/'result.json');q=result_choice(result,mode,coefficient)['bank']
                records.append(dict(job=job,case=meta['case'],bundle=meta['bundle'],mode=mode,coefficient=coefficient,
                    **score(q['point'],data,truth,labels,tuple(map(tuple,common['pool'])))))
    frame=pd.DataFrame(records);frame.to_csv(output/'analysis/development_models.csv',index=False)
    frame.groupby(['mode','coefficient']).agg(inputs=('job','count'),F1=('clade_f1','mean'),exact=('exact','sum'),
        tuning_MAE=('test_mae','mean'),internal=('internal','mean')).reset_index().to_csv(output/'analysis/development_summary.csv',index=False)


def analyze(output):
    output=Path(output);dest=output/'analysis';dest.mkdir(exist_ok=True)
    protocol=read(output/'protocol.json');frozen=read(output/'frozen.json');catalog=read(output/'catalog.json')
    assert frozen['protocol_sha256']==sha(output/'protocol.json') and not frozen['truth_used'] and not frozen['evaluation_test_used']
    assert sha(PARENT/'protocol.json')==protocol['parent_protocol_sha256']
    checks=4;rows=[];errors=[];events=[];failures=[];cross=[];new_errors=[];baseline_references=[];coverage=[]
    for name,digest in protocol['source_hashes'].items():
        assert sha(output/'source'/name)==digest;checks+=1
    for job in protocol['evaluation']:
        meta=catalog[job];common_path=output/'common'/job/'result.json';common=read(common_path);labels=common['labels'];n=len(labels)
        assert sha(common_path)==protocol['frontend_reused'][job];checks+=1
        data={s:prepare(observed(meta[s],meta['N'] if s=='train' else None)) for s in ('train','validation','test')}
        truth=physical_truth(Path(meta['truth_network']),labels);pos={v:i for i,v in enumerate(labels)}
        true={tuple(sorted(pos[v] for v in t)) for t in truth['clades']};results={}
        for cid,config in frozen['evaluation_settings'].items():
            file=output/'evaluation'/job/cid/'result.json'
            if not file.exists():
                failures.append(dict(job=job,config_id=cid,error=read(file.with_name('error.json'))));results[cid]=None;continue
            r=read(file);results[cid]=r;t=r['trajectory'];penalty=t['penalty']
            assert r['frozen_sha256']==sha(output/'frozen.json') and r['protocol_sha256']==sha(output/'protocol.json')
            assert not r['truth_used'] and not r['test_opened'] and r['common_sha256']==sha(common_path)
            assert r['input_hashes']==dict(train=sha(meta['train']),validation=sha(meta['validation']));checks+=5
            assert abs(penalty-config['coefficient']*t['reference_MAE'])<1e-12;checks+=1
            previous=float('inf')
            for point in t['path']:
                value=independent_loss(data['train'].p[0],data['train'].q[0],data['train'].target[0],point)
                assert abs(value-point['objective'])<1e-8
                objective=criterion(point,n,penalty);assert objective<=previous+1e-9;previous=objective;checks+=2
            for event in t['attempts']:
                if event.get('status') is not None:
                    assert event['audit']['valid'] and event['warm_start']['valid'] and event['bound_consistent_with_known_upper'];checks+=3
                if event.get('regularized_bound') is not None:
                    assert event['regularized_bound']<=event['decoded_score']+1e-8;checks+=1
                if event['kind'] in ('rnj_pre_joint','wzzt_tail_joint'):
                    assert all(len(q['blocks'])<=5 for q in event['regions']);checks+=1
                events.append(dict(job=job,panel=meta['panel'],case=meta['case'],config_id=cid,coefficient=config['coefficient'],**event))
            for lp in r['LP']:
                assert lp['status']==2 and max(lp['max_constraint_violation'],lp['max_bound_violation'])<=1e-7;checks+=1
            for stage,point in t['states'].items():
                retained=supports(point,n,True);wrong=retained-true
                blocked={s for s in true if any(not compatible(s,t) for t in wrong)}
                coverage.append(dict(job=job,panel=meta['panel'],case=meta['case'],config_id=cid,coefficient=config['coefficient'],stage=stage,
                    retained=len(retained),true_present=len(retained&true),false_compatible=sum(error_type(s,true)[0]=='compatible_overresolution' for s in wrong),
                    false_cross=sum(error_type(s,true)[0]=='crosses_truth' for s in wrong),blocked_true=len(blocked)))
            choices=[r['selected_plain'],r['selected_regular'],*r['extra_selection'].values()]
            for choice in choices:
                for q in choice.values():
                    assert abs(q['selection_score']-(q['validation']+q['selection_penalty']*count(q['point'],n)))<1e-10;checks+=1
        original_file=PARENT/'evaluation'/job/'baseline'/'result.json';original=read(original_file);highs_cache={}
        assert original['common_sha256']==sha(common_path) and original['protocol_sha256']==sha(PARENT/'protocol.json');checks+=2
        baseline_references.append(dict(job=job,path=str(original_file),sha256=sha(original_file)))
        specs=[('original_rnj_wzzt',None),*frozen['methods'].items()]
        for method,spec in specs:
            base=dict(job=job,panel=meta['panel'],case=meta['case'],bundle=meta['bundle'],N=meta['N'],
                      condition=meta['condition'],voltage_sigma=meta['voltage_sigma'],method=method)
            if spec is None:
                r=original;t=r['trajectory'];choice=dict(bank=r['selected'],path=r['selected_path']);coefficient=0.;cid='previous_baseline'
                origin=r['selected_origin']
            else:
                r=results[spec['config_id']];cid=spec['config_id'];coefficient=spec['coefficient']
                if r is None:
                    for view in ('bank','path','endpoint'):
                        rows.append(dict(**base,view=view,model_available=False,clade_f1=0.,exact=False))
                    continue
                t=r['trajectory'];choice=result_choice(r,spec['mode'],coefficient);origin=choice['bank']['origin']
            for view,point in (('bank',choice['bank']['point']),('path',choice['path']['point']),('endpoint',t['point'])):
                metric=score(point,data,truth,labels,tuple(map(tuple,common['pool'])))
                rows.append(dict(**base,view=view,config_id=cid,coefficient=coefficient,**metric,
                    search_seconds=t['seconds'],elapsed_seconds=r['elapsed_seconds'],total_estimate_seconds=r['elapsed_seconds']+r['common_seconds'],
                    selected_origin=origin if view=='bank' else view,stop_reason=t['stop_reason'],
                    penalized_search=0. if spec is None else t['penalty'],
                    selection_score=choice['bank'].get('selection_score',choice['bank']['validation']) if view=='bank' else None))
                for q in size_errors(point,true,n):
                    errors.append(dict(**base,view=view,**q))
                if view=='bank':
                    for s in supports(point,n)-true:
                        kind,blocked=error_type(s,true)
                        new_errors.append(dict(**base,support=s,size='small' if len(s)<=5 else 'large',error_type=kind,blocked_true=blocked,blocked_count=len(blocked)))
            point=choice['bank']['point'];key=tuple(sorted(map(tuple,point['supports'])))
            if key not in highs_cache:
                check=independent_lp(data['train'],point);highs_cache[key]=check
                cross.append(dict(job=job,family=key,**check))
            assert abs(highs_cache[key]['objective']-point['objective'])<1e-8;checks+=1
    frame=pd.DataFrame(rows);frame.to_csv(dest/'per_model.csv',index=False)
    frame.groupby(['panel','view','method']).agg(inputs=('job','count'),available=('model_available','sum'),
        F1=('clade_f1','mean'),exact=('exact','sum'),train_MAE=('train_mae','mean'),validation_MAE=('validation_mae','mean'),
        test_MAE=('test_mae','mean'),internal=('internal','mean'),search_median=('search_seconds','median')).reset_index().to_csv(dest/'summary.csv',index=False)
    frame.groupby(['panel','view','case','method']).agg(inputs=('job','count'),F1=('clade_f1','mean'),exact=('exact','sum')).reset_index().to_csv(dest/'cases.csv',index=False)
    error_frame=pd.DataFrame(errors);error_frame.to_csv(dest/'size_errors.csv',index=False)
    totals=error_frame.groupby(['panel','view','method','size'])[['predicted','truth','TP','FP','FN']].sum().reset_index()
    totals['false_discovery_rate']=totals.FP/totals.predicted.replace(0,np.nan);totals['miss_rate']=totals.FN/totals.truth.replace(0,np.nan)
    totals.to_csv(dest/'size_summary.csv',index=False)
    pd.DataFrame(new_errors).to_csv(dest/'final_error_types.csv',index=False)
    pd.DataFrame(new_errors).groupby(['panel','method','size','error_type']).size().rename('occurrences').reset_index().to_csv(dest/'final_error_summary.csv',index=False)
    coverage_frame=pd.DataFrame(coverage);coverage_frame.to_csv(dest/'stage_coverage.csv',index=False)
    coverage_frame.groupby(['panel','config_id','stage'])[['retained','true_present','false_compatible','false_cross','blocked_true']].sum().reset_index().to_csv(dest/'coverage_summary.csv',index=False)
    paired=[]
    for (panel,view),g in frame.groupby(['panel','view']):
        pivot=g.pivot(index='job',columns='method',values='clade_f1')
        for method in frozen['methods']:
            if method=='joint_unregularized':
                continue
            for reference in ('original_rnj_wzzt','joint_unregularized'):
                delta=pivot[method]-pivot[reference]
                paired.append(dict(panel=panel,view=view,method=method,reference=reference,inputs=len(delta),mean_delta=delta.mean(),
                    wins=int((delta>1e-9).sum()),ties=int((abs(delta)<=1e-9).sum()),losses=int((delta< -1e-9).sum())))
    pd.DataFrame(paired).to_csv(dest/'paired.csv',index=False)
    write(dest/'events.json',events);write(dest/'crosssolver.json',dict(passed=True,unique_families=len(cross),records=cross))
    write(dest/'baseline_references.json',baseline_references)
    write(dest/'verification.json',dict(passed=True,checks=checks,planned=read(output/'evaluation_completion.json')['planned'],
        returned=read(output/'evaluation_completion.json')['returned'],failures=failures,unique_independent_LP=len(cross),
        scored_bank_outputs=len(protocol['evaluation'])*(len(frozen['methods'])+1),analysis_source_sha256=sha(__file__)))
    audit_development(output);development_score(output)
    print(pd.read_csv(dest/'summary.csv').to_string(index=False),flush=True)
    print('VERIFIED',checks,'checks;',len(cross),'unique independent LPs',flush=True)
