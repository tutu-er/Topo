"""Line-count penalty on explicit structural choices, in original MAE units."""
from __future__ import annotations

from time import perf_counter
import numpy as np

from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.joint_five.model import build_joint, open_small, remainder
from research_experiments.joint_five.search import search as unregularized_search
from research_experiments.local_rebuild.model import canonical, compatible
from research_experiments.rebuild_benchmark.engine import SCALE, solve, reconstruction_start, any_extension, loss
from research_experiments.wzzt_retune.engine import extension_start
from research_experiments.lp_rebuild.engine import select_output as plain_output


def count(point, n):
    """Retained proper internal supports; fixed coarse supports count as constants."""
    return sum(1<len(s)<n for s in point['supports'])


def criterion(point, n, penalty):
    return point['objective']+penalty*count(point,n)


def joint_fit(bank, current, keep, *, penalty, seconds, stage, relative=.001):
    then=perf_counter();model,meta=build_joint(bank.scaled,keep)
    if not set(canonical(current['supports']))<=set(meta['all_supports']):
        raise ValueError('previous tree is missing from the regularized domain')
    for i in range(meta['selection'].start,meta['selection'].stop):
        model.variables.objective[i]=penalty*SCALE
    constant=penalty*sum(1<len(s)<bank.data.n for s in meta['keep'])
    before=criterion(current,bank.data.n,penalty);point=current
    event=dict(kind=stage,solve_started=False,accepted=False,before=current['objective'],before_score=before,
        penalty=penalty,protected_count_cost=constant,candidate_count=len(meta['candidates']),keep=meta['keep'],
        candidates=meta['candidates'],regions=meta['regions'],conflicts=meta['conflicts'],
        maximum_remainder=meta['maximum_remainder'],skipped_large_regions=meta['skipped_large_regions'])
    remaining=seconds-(perf_counter()-then)
    if meta['candidates'] and remaining>.05:
        start=reconstruction_start(bank,model,meta,current)
        values,record=solve(bank,model,start,seconds=remaining);event.update(record,solve_started=True)
        if values is not None:
            indices=[*range(len(meta['keep'])),*[len(meta['keep'])+i for i,a in enumerate(values[meta['selection']]) if a>.5]]
            r,x=values[model.r],values[model.x]
            indices=[i for i in indices if i<len(meta['keep']) or max(r[i],x[i])>1e-9]
            point=bank.fit([meta['all_supports'][i] for i in indices],preserve=True)
            retained=[s for s,r,x in zip(point['supports'],point['r'],point['x'])
                      if tuple(s) in meta['keep'] or max(r,x)>1e-9]
            if len(retained)!=len(point['supports']):
                point=bank.fit(retained,preserve=True)
            if criterion(point,bank.data.n,penalty)>record['objective']+constant+1e-8:
                raise RuntimeError('canonical regularized refit exceeds the feasible MIP score')
    elif not meta['candidates']:
        point=bank.fit(meta['keep'],preserve=True);event['no_candidates']=True
    score=criterion(point,bank.data.n,penalty);gain=before-score;gate=1e-10+relative*current['objective']
    event.update(decoded_objective=point['objective'],decoded_score=score,gain=gain,threshold=gate,accepted=gain>gate,
        regularized_bound=None if event.get('dual_bound') is None else event['dual_bound']+constant,
        regularized_incumbent=None if event.get('objective') is None else event['objective']+constant,
        wall_seconds=perf_counter()-then,scope='conditional five-block count-regularized joint domain')
    return (point if gain>gate else current),event


def original_step(bank,current,*,penalty,deadline,per_mip=2.,relative=.001):
    family=canonical(current['supports']);n=bank.data.n;new=any_extension(family,n,bank.lp.pool)
    if new is None:
        return current,dict(kind='wzzt_extension',accepted=False,structurally_complete=True)
    model=core.build_extension_model(bank.scaled,family,r_upper_bound=2.,x_upper_bound=2.)
    activation=model.variables.add('activate_new',1,upper=1.,integral=True,objective=penalty*SCALE)
    for weight in (model.new_r,model.new_x):
        model.rows.add({weight.start:1.,activation.start:-2.},upper=0.)
    warm=current;checked=0
    def clean_new(q,c):
        i=list(map(tuple,q['supports'])).index(c)
        if max(q['r'][i],q['x'][i])<=1e-9:
            return bank.fit([s for s in q['supports'] if tuple(s)!=c],preserve=True)
        return q
    for c in bank.lp.pool:
        if c in family or not all(compatible(c,s) for s in family):
            continue
        if deadline-perf_counter()<per_mip+.1:
            break
        q=clean_new(bank.fit([*family,c],preserve=True),c);checked+=1
        if criterion(q,n,penalty)<criterion(warm,n,penalty)-1e-12:
            warm,new=q,c
    if deadline-perf_counter()<=.05:
        return current,dict(kind='wzzt_extension',accepted=False,time_exhausted=True)
    start=extension_start(model,family,new,warm,bank.lp.p,bank.lp.q,bank.lp.y)
    start[activation]=float(new in canonical(warm['supports']))
    values,event=solve(bank,model,start,seconds=min(per_mip,deadline-perf_counter()));best=warm
    if values is not None and values[activation.start]>.5:
        c=core._round_and_validate_support(values[model.z],family,n)
        q=clean_new(bank.fit([*family,c],preserve=True),c)
        if criterion(q,n,penalty)<criterion(best,n,penalty):
            best=q
    before=criterion(current,n,penalty);score=criterion(best,n,penalty);gate=1e-10+relative*current['objective']
    constant=penalty*count(current,n)
    event.update(kind='wzzt_extension',warm_checked=checked,before=current['objective'],before_score=before,
        decoded_objective=best['objective'],decoded_score=score,gain=before-score,threshold=gate,accepted=before-score>gate,
        penalty=penalty,protected_count_cost=constant,
        regularized_bound=None if event.get('dual_bound') is None else event['dual_bound']+constant,
        added=[s for s in best['supports'] if s not in family])
    return (best if event['accepted'] else current),event


def search(bank,initial,*,coefficient,seconds=20.,pre_seconds=5.,per_mip=2.,relative=.001):
    if coefficient==0:
        result=unregularized_search(bank,initial,pre_joint=True,tail_joint=True,seconds=seconds,
                                   pre_seconds=pre_seconds,per_mip=per_mip,relative=relative)
        result.update(coefficient=0.,penalty=0.,reference_MAE=result['path'][0]['objective'])
        return result
    then=perf_counter();deadline=then+seconds;n=bank.data.n
    initial=canonical([*initial,*((i,) for i in range(n)),tuple(range(n))]);raw=bank.fit(initial,preserve=True)
    reference=max(raw['objective'],1e-12);penalty=coefficient*reference
    current=raw;path=[raw];events=[];states=dict(rnj=raw);reason='wall_limit'
    keep,released=open_small(initial,n)
    current,event=joint_fit(bank,current,keep,penalty=penalty,seconds=min(pre_seconds,max(0.,deadline-perf_counter())),stage='rnj_pre_joint')
    event['released']=released;events.append(event)
    if event['accepted']:
        path.append(current)
    states['after_pre']=current
    for iteration in range(3*n):
        if deadline-perf_counter()<=.05:
            break
        if remainder(current['supports'],n)<=5:
            states['before_tail']=current;keep,released=open_small(current['supports'],n)
            current,event=joint_fit(bank,current,keep,penalty=penalty,seconds=deadline-perf_counter(),stage='wzzt_tail_joint')
            event.update(released=released,iteration=iteration);events.append(event)
            if event['accepted']:
                path.append(current)
            reason='joint_tail_gap_met' if event.get('status')==2 else 'joint_tail_no_candidates' if event.get('no_candidates') else 'joint_tail_unresolved'
            break
        current,event=original_step(bank,current,penalty=penalty,deadline=deadline,per_mip=per_mip,relative=relative)
        event['iteration']=iteration;events.append(event)
        if event['accepted']:
            path.append(current)
        else:
            reason='structurally_complete' if event.get('structurally_complete') else 'wall_limit' if event.get('time_exhausted') else 'no_regularized_gain'
            break
    else:
        reason='step_limit'
    states['endpoint']=current
    return dict(point=current,path=path,attempts=events,states=states,stop_reason=reason,seconds=perf_counter()-then,
        coefficient=coefficient,penalty=penalty,reference_MAE=reference,maximum_remainder=remainder(current['supports'],n),tree_global_optimal=False)


def select_output(bank,points,validation,*,penalty=0.,resolution=.16):
    if penalty==0:
        q=plain_output(bank,points,validation,resolution=resolution)
        q.update(selection_score=q['validation'],selection_penalty=0.,internal=count(q['point'],bank.data.n))
        return q
    candidates=[];cal=bank.calibration
    for i,point in enumerate(points):
        family=[s for s,r,x in zip(point['supports'],point['r'],point['x']) if len(s)==1 or
                cal['a']*r+cal['b']*x+1e-10>=resolution*cal['median_root_depth']]
        fitted=bank.fit(family);value=loss(validation,fitted);k=count(fitted,bank.data.n)
        candidates.append(dict(point=fitted,validation=value,selection_score=value+penalty*k,
                               selection_penalty=penalty,internal=k,index=i))
    best=min(q['selection_score'] for q in candidates)
    return min((q for q in candidates if q['selection_score']<=best+1e-12),key=lambda q:(q['internal'],q['validation'],q['index']))
