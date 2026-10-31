"""Release provisional boundaries, keeping the physical measurement objective."""
from __future__ import annotations

from time import perf_counter
import numpy as np

from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.local_rebuild.model import build_model, canonical
from research_experiments.rebuild_benchmark.engine import SCALE, regions, reconstruction_start, solve, loss
from research_experiments.rnj_count.engine import criterion, original_step


def clean_zero(bank, point, *, hard=()):
    protected=set(canonical(hard)); before=canonical(point['supports'])
    keep=[s for s,r,x in zip(point['supports'],point['r'],point['x'])
          if len(s)==1 or tuple(s) in protected or max(r,x)>1e-9]
    if len(keep)==len(before):
        return point,[]
    # A numerical threshold has no unconditional zero-loss guarantee.
    proposed=bank.fit(keep,preserve=True)
    if proposed['objective']>point['objective']+1e-10:
        return point,[]
    return proposed,[s for s in before if s not in canonical(proposed['supports'])]


def local_fit(bank,current,region,blocks,*,penalty,seconds,relative=.001,hard=()):
    then=perf_counter();model,meta=build_model(bank.scaled,current['supports'],region,blocks,hard=hard,cuts='subset')
    for col in range(meta['selection'].start,meta['selection'].stop):
        model.variables.objective[col]=penalty*SCALE
    constant=penalty*sum(1<len(s)<bank.data.n for s in meta['keep'])
    before=criterion(current,bank.data.n,penalty)
    event=dict(kind='revocable_local',region=region,blocks=blocks,released=meta['released'],
        keep=meta['keep'],candidates=meta['candidates'],candidate_count=len(meta['candidates']),
        before=current['objective'],before_score=before,protected_count_cost=constant,
        penalty=penalty,accepted=False,solve_started=False)
    point=current;remaining=seconds-(perf_counter()-then)
    if remaining>.05 and meta['candidates']:
        start=reconstruction_start(bank,model,meta,current)
        values,record=solve(bank,model,start,seconds=remaining)
        event.update(record,solve_started=True)
        if values is not None:
            indices=[*range(len(meta['keep'])),*[len(meta['keep'])+k for k,a in enumerate(values[meta['selection']]) if a>.5]]
            family=[meta['all_supports'][i] for i in indices]
            point=bank.fit(family,preserve=True)
            point,removed=clean_zero(bank,point,hard=hard)
            event['removed_zero']=removed
            if criterion(point,bank.data.n,penalty)>record['objective']+constant+1e-8:
                raise RuntimeError('decoded regularized score exceeds its feasible solver upper bound')
    gate=1e-10+relative*current['objective'];score=criterion(point,bank.data.n,penalty)
    bound=None if event.get('dual_bound') is None else event['dual_bound']+constant
    event.update(decoded_objective=point['objective'],decoded_score=score,gain=before-score,threshold=gate,
        accepted=before-score>gate,regularized_bound=bound,
        regularized_incumbent=None if event.get('objective') is None else event['objective']+constant,
        bound_excludes_gain=bound is not None and before-bound<=gate,
        deleted=[s for s in current['supports'] if tuple(s) not in canonical(point['supports'])],
        added=[s for s in point['supports'] if tuple(s) not in canonical(current['supports'])],
        wall_seconds=perf_counter()-then)
    return (point if event['accepted'] else current),event


def neighborhoods(point,n,kernel,*,hard=()):
    """Prefer frontiers opening old boundaries; only training geometry is used."""
    family=canonical(point['supports']);protected=set(canonical(hard))
    def released(q):
        v,blocks,_=q
        return sum(set(s)<set(v) and not any(set(s)<=set(b) for b in blocks) and s not in protected for s in family)
    choices=regions(family,n,5,kernel)
    return sorted(choices,key=lambda q:(-released(q),-len(q[0]),-len(q[1]),q[0],q[1]))


def search(bank,initial,kernel,*,kind='revocable',coefficient=.003,seconds=20.,pre_seconds=5.,per_mip=2.,relative=.001):
    then=perf_counter();deadline=then+seconds;n=bank.data.n
    initial=canonical([*initial,*((i,) for i in range(n)),tuple(range(n))])
    raw=bank.fit(initial,preserve=True);reference=max(raw['objective'],1e-12);penalty=coefficient*reference
    current=raw;path=[raw];events=[];states=dict(rnj=raw);tried=set();forward_no_gain=False;reason='wall_limit'
    current,removed=(current,[]) if kind=='locked' else clean_zero(bank,current)
    if removed:
        path.append(current)
    states['after_zero']=current
    def local_phase(until,stage):
        nonlocal current
        accepted=0
        while until-perf_counter()>.08:
            family=canonical(current['supports'])
            todo=[q for q in neighborhoods(current,n,kernel) if (family,q) not in tried]
            if not todo:
                return accepted,True
            q=todo[0];tried.add((family,q));v,blocks,virtual=q
            current,event=local_fit(bank,current,v,blocks,penalty=penalty,
                seconds=min(per_mip,until-perf_counter()),relative=relative)
            event.update(stage=stage,virtual_region=virtual);events.append(event)
            if event['accepted']:
                path.append(current);accepted+=1
        return accepted,False
    if kind=='revocable':
        local_phase(min(deadline,then+pre_seconds),'rnj_pre')
    elif kind not in ('zero_only','locked'):
        raise ValueError(kind)
    states['after_pre']=current
    for iteration in range(4*n):
        if deadline-perf_counter()<=.08:
            break
        if not forward_no_gain:
            current,event=original_step(bank,current,penalty=penalty,deadline=deadline,per_mip=per_mip,relative=relative)
            event.update(stage='wzzt',iteration=iteration);events.append(event)
            if event['accepted']:
                path.append(current)
                current,removed_now=(current,[]) if kind=='locked' else clean_zero(bank,current)
                if removed_now:
                    path.append(current)
            else:
                forward_no_gain=True
        if kind in ('zero_only','locked'):
            if forward_no_gain:
                reason='forward_no_gain_or_unresolved';break
            continue
        gained,complete=local_phase(min(deadline,perf_counter()+per_mip),'wzzt_repair')
        if gained:
            forward_no_gain=False
        elif complete and forward_no_gain:
            relevant=[e for e in events if e['kind']=='revocable_local' and
                canonical(e['keep']+e['released'])==canonical(current['supports'])]
            reason='declared_neighborhood_bound_gate' if relevant and all(e['bound_excludes_gain'] for e in relevant) else 'neighborhood_scan_unresolved'
            break
    else:
        reason='step_limit'
    states['endpoint']=current
    return dict(point=current,path=path,attempts=events,states=states,stop_reason=reason,
        seconds=perf_counter()-then,penalty=penalty,coefficient=coefficient,reference_MAE=reference,
        removed_initial_zero=removed,kind=kind,tree_global_optimal=False,
        scope='five-block current-subtree frontiers and bounded sibling subsets, provisional supports releasable')
