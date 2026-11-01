"""RNJ-small-seed mainline with immediate cleanup and one stall fork.

Validation selects a complete unmasked/masked continuation after training.
The masked support is never made a permanent prior on other branches.
"""
from time import perf_counter
import numpy as np
from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.local_rebuild.model import canonical
from research_experiments.rnj_small_seed.engine import prefix, exit_repair
from research_experiments.rnj_count.engine import criterion
from research_experiments.flexible_exit.model import MaskBank, replay
from research_experiments.joint_exit.selection import choose
from .zero import release_zero_atoms


def stopped_without_gain(trajectory):
    if not trajectory['attempts']:return False
    last=trajectory['attempts'][-1]
    return not last.get('accepted',False) and not last.get('time_exhausted',False)


def weakest_deletion(bank, point, *, hard=(), deadline):
    """Lowest refitted deletion cost, training only; not line reliability."""
    protected=set(canonical(hard));records=[];complete=True
    for C in canonical(point['supports']):
        if len(C) in (1,bank.data.n) or C in protected:continue
        if deadline-perf_counter()<=.1:complete=False;break
        q=bank.fit([D for D in point['supports'] if tuple(D)!=C],preserve=True)
        cost=q['objective']-point['objective']
        if cost < -2e-7:raise RuntimeError('Deleting a fixed LP column improved the certified MAE.')
        records.append(dict(block=C,cost=cost,point=q))
    best=min(records,key=lambda r:(r['cost'],len(r['block']),r['block'])) if records else None
    return best,dict(complete=complete,candidates=[dict(block=r['block'],cost=r['cost']) for r in records],truth_used=False)


def finish_exit(bank, common, trajectory, *, reference, deadline, hard=()):
    current=trajectory['point'];remaining=deadline-perf_counter()
    if remaining<=.1:return trajectory,dict(accepted=False,skipped='deadline'),{}
    q,event=exit_repair(bank,common['initial'],current,np.asarray(common['kernel']),reference=reference,
        prefix_seconds=0.,warm_candidates=common['bank'],total_seconds=remaining,
        forbidden=bank.forbidden,hard=hard)
    point,zeros=release_zero_atoms(bank,q['point'],hard=hard,stage='final_exit')
    path=list(trajectory['path'])
    if q['accepted']:path.append(q['point'])
    if any(e['accepted'] for e in zeros):path.append(point)
    return {**trajectory,'point':point,'path':path,'zero_events':[*trajectory.get('zero_events',[]),*zeros]},q,event


def validation_rows(bank, points, validation, *, source, hard=(), resolution=.24):
    cal=bank.calibration;protected=set(canonical(hard));rows=[]
    for index,point in enumerate(points):
        family=canonical(C for C,r,x in zip(point['supports'],point['r'],point['x']) if len(C)==1 or tuple(C) in protected or
            cal['a']*r+cal['b']*x+1e-10>=resolution*cal['median_root_depth'])
        fitted=bank.fit(family,preserve=True)
        fitted,_=release_zero_atoms(bank,fitted,hard=hard,stage='output_refit')
        R,X=core.build_matrices_from_atoms(bank.data.n,fitted['supports'],fitted['r'],fitted['x'])
        value=sum(float(abs(y-p@R.T-q@X.T).sum()) for p,q,y in zip(validation.p,validation.q,validation.target))/validation.observation_count
        rows.append(dict(point=fitted,validation=value,index=index,source=source,
            internal=sum(1<len(C)<bank.data.n for C in fitted['supports'])))
    return rows


def select_outputs(choices,rnj_rows=()):
    candidates=[dict(row,index=i) for i,row in enumerate(choices.values())]
    own=choose(candidates,'strict')
    proposed=[dict(row,index=i) for i,(name,row) in enumerate(choices.items()) if name!='baseline']
    proposal=choose(proposed,'strict') if proposed else None
    pooled=[dict(row,index=i) for i,row in enumerate([*rnj_rows,*choices.values()])]
    return proposal,own,choose(pooled,'strict')


def fit_mainline(data,validation,common,*,reference,seconds=14400.,prefix_seconds=7200.,per_mip=3600.,lp_seconds=300.,
                 post_reserve=600.,recovery=True,mask_branch=True,hard=(),heartbeat=None,checkpoint=None):
    """Run the four-stage mainline; at most one training-selected deletion.

    One shared wall budget covers the baseline and both counterfactual forks.
    Existing initial_supports in the public core API keep their hard semantics.
    RNJ seeds here are temporary unless explicitly listed in hard.
    """
    then=perf_counter();deadline=then+seconds;structural_deadline=deadline-min(post_reserve,seconds/5)
    hard=canonical(hard);penalty=.003*reference
    bank=MaskBank(data,[*common['pool'],*hard],common['calibration'],deadline=deadline,seconds=lp_seconds)
    bank.joint_exit_seed_pool=canonical(common['pool']);bank.seed_start_pool=bank.joint_exit_seed_pool
    branches={};banks=[]
    try:
        if checkpoint:checkpoint(dict(stage='prefix_started'))
        t=prefix(bank,common['initial'],mode='small3',reference=reference,
            seconds=min(prefix_seconds,max(.1,(structural_deadline-perf_counter())/2)),per_mip=per_mip,
            strategy='original',zero_release=True,hard=hard)
        if checkpoint:checkpoint(dict(stage='prefix_complete',prefix=t))
        stalled=stopped_without_gain(t);selected=None;ranking=dict(complete=True,candidates=[])
        if recovery and stalled and structural_deadline-perf_counter()>.1:
            selected,ranking=weakest_deletion(bank,t['point'],hard=hard,deadline=structural_deadline)
        forks=['release']+(['masked'] if mask_branch else []) if selected else []
        remaining=structural_deadline-perf_counter()
        baseline_deadline=perf_counter()+remaining/(1+len(forks))
        base,exit_record,event=finish_exit(bank,common,t,reference=reference,deadline=baseline_deadline,hard=hard)
        branches['baseline']=dict(trajectory=base,exit=exit_record,exit_event=event)
        if checkpoint:checkpoint(dict(stage='baseline_exit_complete',prefix=t,branches=branches,ranking=ranking))
        # No validation quantities have been read to choose the deletion or train forks.
        for index,name in enumerate(forks):
            if structural_deadline-perf_counter()<=.1:break
            phase_deadline=perf_counter()+(structural_deadline-perf_counter())/(len(forks)-index)
            C=tuple(selected['block']);forbidden=(C,) if name=='masked' else ()
            pool=canonical(D for D in [*common['pool'],*t['point']['supports']] if tuple(D) not in forbidden)
            child=MaskBank(data,pool,common['calibration'],forbidden=forbidden,deadline=deadline,seconds=lp_seconds)
            banks.append(child);child.joint_exit_seed_pool=pool;child.seed_start_pool=pool
            initial=child.fit([D for D in t['point']['supports'] if tuple(D)!=C],preserve=True)
            child_trajectory=replay(child,initial,penalty=penalty,
                deadline=perf_counter()+(phase_deadline-perf_counter())/2,per_mip=per_mip,heartbeat=heartbeat,
                zero_release=True,hard=hard)
            child_trajectory,exit_record,event=finish_exit(child,common,child_trajectory,
                reference=reference,deadline=phase_deadline,hard=hard)
            if name=='masked':
                assert all(C not in canonical(p['supports']) for p in child_trajectory['path'])
            branches[name]=dict(trajectory=child_trajectory,exit=exit_record,exit_event=event,
                block=C,forbidden=forbidden,temporary_drop_L=initial['objective'],
                temporary_drop_J=criterion(initial,data.n,penalty),parent_J=criterion(t['point'],data.n,penalty))
            if checkpoint:checkpoint(dict(stage=name+'_complete',prefix=t,branches=branches,ranking=ranking))
        choices={}
        for name,record in branches.items():
            active_bank=bank if name=='baseline' else banks[forks.index(name)]
            rows=validation_rows(active_bank,record['trajectory']['path'],validation,source=name,hard=hard)
            choices[name]=choose(rows,'strict')
        rnj_points=[p for p in common['bank'] if set(hard)<=set(canonical(p['supports']))]
        rnj_rows=validation_rows(bank,rnj_points,validation,source='complete_rnj',hard=hard)
        proposal,own,final=select_outputs(choices,rnj_rows)
        _,_,baseline_pooled=select_outputs(dict(baseline=choices['baseline']),rnj_rows)
        elapsed=perf_counter()-then
        return dict(prefix=t,branches=branches,choices=choices,new_branch=proposal,own_output=own,with_rnj_fallback=final,
            baseline_with_rnj=baseline_pooled,
            recovery_triggered=bool(selected),recovery_attempts=1 if selected else 0,stalled=stalled,
            selected_block=selected['block'] if selected else None,ranking=ranking,
            mask_adopted=final['source']=='masked',seconds=elapsed,total_cap=seconds,total_cap_exceeded=elapsed>seconds+1.,
            hard=hard,truth_used=False,test_opened=False,
            LP_failures=[*bank.failures,*[row for b in banks for row in b.failures]])
    finally:
        for child in banks:child.close()
        bank.close()
