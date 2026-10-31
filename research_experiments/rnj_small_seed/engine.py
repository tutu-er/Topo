"""Sparse temporary RNJ seeds; largest-frontier WZZT; joint tail and exit.

All observations remain at original terminals. Seed groups are atomic only
during the forward stage. Exit windows explicitly split their old boundaries.
"""
from __future__ import annotations
from itertools import combinations
from math import ceil
from time import perf_counter
import numpy as np
from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.local_rebuild.model import canonical,compatible
from research_experiments.rebuild_benchmark.engine import hierarchy_children,SCALE,solve,reconstruction_start,loss
from research_experiments.wzzt_retune.engine import extension_start
from research_experiments.rnj_count.engine import criterion,joint_fit
from research_experiments.joint_five.model import open_small,remainder
from research_experiments.joint_exit.model import candidate_pool,build,solve_checkpoints,physical_weights
from research_experiments.revocable.engine import clean_zero


def seed_family(bank,initial,*,mode,reference,deadline,max_seeds=2):
    full=bank.fit(initial,preserve=True);n=bank.data.n;records=[]
    permitted=lambda s: (2<=len(s)<=int(mode[-1])) if mode.startswith('small') else 5<len(s)<n
    candidates=canonical(s for s in initial if permitted(s)) if mode!='none' else ()
    complete=True
    for C in candidates:
        if deadline-perf_counter()<=.1:complete=False;break
        q=bank.fit([s for s in initial if tuple(s)!=C],preserve=True)
        cost=q['objective']-full['objective']
        if cost< -1e-8:raise RuntimeError('subset LP improved the reference full-family optimum')
        records.append(dict(support=C,deletion_cost=cost))
    chosen=[]
    for r in sorted(records,key=lambda q:(-q['deletion_cost'],len(q['support']),q['support'])):
        C=r['support']
        if r['deletion_cost']<=.003*reference+1e-10:continue
        if mode.startswith('small') and any(set(C)&set(B) for B in chosen):continue
        chosen.append(C)
        if len(chosen)>=max_seeds:break
    family=canonical([*((i,) for i in range(n)),tuple(range(n)),*chosen])
    return bank.fit(family,preserve=True),dict(mode=mode,seeds=chosen,rankings=records,ranking_complete=complete,
        reference_MAE=reference,full_reference_MAE=full['objective'],truth_used=False)


def frontier_model(data,family,V,blocks,penalty,*,large):
    model=core.build_extension_model(data,canonical(family),r_upper_bound=2.,x_upper_bound=2.)
    activation=model.variables.add('activate_new',1,upper=1.,integral=True,objective=penalty*SCALE)
    for w in (model.new_r,model.new_x):model.rows.add({w.start:1.,activation.start:-2.},upper=0.)
    for i in range(data.n):
        if i not in V:model.variables.upper[model.z.start+i]=0.
    for B in blocks:
        for i in B[1:]:model.rows.add({model.z.start+i:1.,model.z.start+B[0]:-1.},lower=0.,upper=0.)
    lower=max(2,ceil(len(blocks)/2)) if large else 2
    model.rows.add({model.z.start+B[0]:1. for B in blocks},lower=float(lower),upper=float(len(blocks)-1))
    return model,activation,lower


def forward(bank,current,*,penalty,deadline,per_mip=2.):
    then=perf_counter();family=canonical(current['supports']);children=hierarchy_children(family,bank.data.n)
    V,blocks=min(((V,B) for V,B in children.items() if len(B)>5),key=lambda q:(-len(q[1]),-len(q[0]),q[0]))
    trials=[]
    for large in (True,False):
        if deadline-perf_counter()<=.1:break
        model,a,lower=frontier_model(bank.scaled,family,V,blocks,penalty,large=large)
        new=tuple(sorted(i for B in blocks[:lower] for i in B));warm=current;checked=0
        # A few observation-only starts; the MIP domain remains all eligible unions.
        legal=[]
        for C in bank.seed_start_pool:
            indices=[k for k,B in enumerate(blocks) if set(B)<=set(C)]
            if lower<=len(indices)<len(blocks) and set(C)=={i for k in indices for i in blocks[k]} and C not in family:
                legal.append(C)
        for C in legal[:4]:
            if deadline-perf_counter()<per_mip+.1:break
            q=bank.fit([*family,C],preserve=True);checked+=1
            if criterion(q,bank.data.n,penalty)<criterion(warm,bank.data.n,penalty):warm,new=q,C
        if deadline-perf_counter()<=.05:break
        start=extension_start(model,family,new,warm,bank.lp.p,bank.lp.q,bank.lp.y)
        start[a]=float(new in canonical(warm['supports']))
        values,event=solve(bank,model,start,seconds=min(per_mip,deadline-perf_counter()));point=warm
        if values is not None and values[a.start]>.5:
            C=core._round_and_validate_support(values[model.z],family,bank.data.n)
            q=bank.fit([*family,C],preserve=True)
            if criterion(q,bank.data.n,penalty)<criterion(point,bank.data.n,penalty):point=q
        before=criterion(current,bank.data.n,penalty);after=criterion(point,bank.data.n,penalty);gate=1e-10+.001*current['objective']
        accepted=before-after>gate
        trials.append(dict(**event,large_first=large,minimum_child_blocks=lower,warm_checked=checked,
            accepted=accepted,before_score=before,score=after,threshold=gate,added=[s for s in point['supports'] if tuple(s) not in family],
            region=V,blocks=blocks,decoded_objective=point['objective'],fixed_count_cost=penalty*sum(1<len(s)<bank.data.n for s in family)))
        if accepted:return point,dict(kind='coarse_wzzt',accepted=True,trials=trials,seconds=perf_counter()-then)
    return current,dict(kind='coarse_wzzt',accepted=False,trials=trials,seconds=perf_counter()-then,stop='no_gain_or_unresolved')


def prefix(bank,initial,*,mode,reference,seconds=20.):
    then=perf_counter();deadline=then+seconds;penalty=.003*reference
    current,seeding=seed_family(bank,initial,mode=mode,reference=reference,deadline=deadline)
    path=[current];events=[];reason='wall_limit';initial_max=remainder(current['supports'],bank.data.n)
    for iteration in range(3*bank.data.n):
        if deadline-perf_counter()<=.1:break
        if remainder(current['supports'],bank.data.n)<=5:
            keep,released=open_small(current['supports'],bank.data.n)
            current,e=joint_fit(bank,current,keep,penalty=penalty,seconds=deadline-perf_counter(),stage='joint_tail')
            e.update(released=released,iteration=iteration);events.append(e)
            if e['accepted']:path.append(current)
            reason='joint_tail_gap_met' if e.get('status')==2 else 'joint_tail_unresolved';break
        current,e=forward(bank,current,penalty=penalty,deadline=deadline);e['iteration']=iteration;events.append(e)
        if e['accepted']:path.append(current)
        else:reason='coarse_no_gain_or_unresolved';break
    return dict(point=current,path=path,attempts=events,seeding=seeding,seconds=perf_counter()-then,stop_reason=reason,
        initial_maximum_remainder=initial_max,maximum_remainder=remainder(current['supports'],bank.data.n),reference_MAE=reference,
        coefficient=.003,penalty=penalty,tree_global_optimal=False)


def repair_windows(initial,kernel,n):
    """Open every maximal <=3-terminal RNJ region with two nearby terminals.

    This is fixed from the common training frontend, independent of selected
    seed mode. All pairs are also available, including across seed boundaries.
    """
    small=canonical(s for s in initial if 2<=len(s)<=3)
    maximal=[s for s in small if not any(set(s)<set(t) for t in small)]
    D=np.diag(kernel)[:,None]+np.diag(kernel)[None,:]-2*kernel
    windows=[];pool=set(combinations(range(n),2))
    for B in maximal:
        near=sorted((i for i in range(n) if i not in B),key=lambda i:(float(np.mean(D[i,list(B)])),i))[:5-len(B)]
        W=tuple(sorted([*B,*near]));windows.append(W)
        pool.update(C for k in range(2,len(W)+1) for C in combinations(W,k) if len(C)<n)
    return canonical(pool),canonical(windows)


def exit_repair(bank,initial,current,kernel,*,reference,prefix_seconds):
    then=perf_counter();base,domain=candidate_pool(bank,initial,current,kernel,'compact')
    extra,windows=repair_windows(initial,kernel,bank.data.n);pool=canonical([*base,*extra]);penalty=.003*reference
    model,meta=build(bank.scaled,pool,penalty=penalty);construction=perf_counter()-then
    snapshots,record=solve_checkpoints(bank,model,meta,current,prefix_seconds=prefix_seconds+construction,budgets=(40.,))
    s=snapshots[40.];values=s.pop('values');from research_experiments.rebuild_benchmark.engine import audit_vector
    audit=audit_vector(model,values)
    if not audit['valid']:raise RuntimeError('invalid exit incumbent')
    indices=[*range(len(meta['keep'])),*[len(meta['keep'])+k for k,v in enumerate(values[meta['selection']]) if v>.5]]
    rr,rp=physical_weights(values[model.r]);xx,xp=physical_weights(values[model.x])
    raw=dict(supports=[meta['all_supports'][i] for i in indices],r=rr[indices].tolist(),x=xx[indices].tolist());raw['objective']=loss(bank.data,raw)
    before=criterion(current,bank.data.n,penalty);raw_score=criterion(raw,bank.data.n,penalty)
    if raw_score>audit['objective']+1e-8:raise RuntimeError('raw physical score exceeds audited epigraph')
    accepted=prefix_seconds+construction<40 and before-raw_score>1e-10+.001*current['objective']
    point=bank.fit(raw['supports'],preserve=True) if accepted else current
    if accepted:point,_=clean_zero(bank,point)
    score=criterion(point,bank.data.n,penalty)
    if score>before+1e-8 or (s['bound'] is not None and s['bound']>score+1e-8):raise RuntimeError('invalid exit score/bound')
    return dict(point=point,raw_point=raw,accepted=accepted,before_score=before,score=score,raw_score=raw_score,audit=audit,
        coefficient_projection_max=max(rp,xp),**s),{**record,**domain,'pool':pool,'pool_count':len(pool),'repair_windows':windows,
        'additional_supports':len(set(pool)-set(base)),'construction_seconds':construction}
