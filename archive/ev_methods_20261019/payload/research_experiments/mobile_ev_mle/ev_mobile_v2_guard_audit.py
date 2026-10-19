"""Independent arithmetic for the mobile per-reference AC protection rule.

Fit losses and derivatives are physically replayed by ev_mobile_v2_review.
Here every nearest gap is rebuilt from all K-by-K retained fit losses. The
saved guard success flags and saved aggregation minima are not used as proof.
No fitting or optimization is invoked.
"""
from __future__ import annotations
import numpy as np


def require(ok,message):
    if not ok:raise ValueError(message)


def close(a,b,message,*,atol=2e-7,rtol=2e-9):
    x,y=np.asarray(a,float),np.asarray(b,float)
    require(x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
        and np.allclose(x,y,atol=atol,rtol=rtol),message)


def nearest(profile,names):
    """Derive all signed nearest gaps from retained total H+predicted-E losses."""
    K=len(names);references=profile.get('references',[]);expected={(a,b) for a in names for b in names if a!=b}
    require(len(references)==K and {r['a'] for r in references}==set(names),'guard reference denominator')
    fits={r['a']:r['fits'] for r in references}
    require(all(set(row)==set(names) for row in fits.values()),'guard all K-by-K fits required')
    pairs=profile.get('pairs',[])
    require(len(pairs)==len(expected) and {(p['a'],p['b']) for p in pairs}==expected,'guard all directed competitors required')
    if not all(p.get('success') for p in pairs):return {}
    gaps={}
    for a in names:
        require(all('loss' in row for row in fits[a].values()),'guard attained losses absent')
        values={b:.5*(fits[a][b]['loss']-fits[a][a]['loss']) for b in names if b!=a}
        for pair in pairs:
            if pair['a']==a:close(pair['half_gap'],values[pair['b']],'guard pair differs from retained fit losses')
        gaps[a]=min(values.values())
    saved=profile.get('aggregation',{}).get('nearest',[])
    require(len(saved)==K and {r['a'] for r in saved}==set(names),'guard saved nearest denominator')
    for row in saved:close(row['half_gap'],gaps[row['a']],'guard saved nearest differs from all rivals')
    return gaps


def guard(profile,floor,tolerance,names,saved=None):
    require(np.isfinite(tolerance) and tolerance>=0,'finite nonnegative guard tolerance')
    gaps=nearest(profile,names);complete=bool(profile.get('success') and set(gaps)==set(names))
    numeric={a:tolerance*(1+abs(floor[a])) for a in names}
    slacks={a:None if a not in gaps else gaps[a]-floor[a] for a in names}
    violated=[a for a in names if slacks[a] is None or slacks[a]<-numeric[a]]
    success=complete and not violated
    if saved is not None:
        require(saved.get('success')==success and saved.get('complete')==complete,'saved AC guard status')
        require(set(saved['reference_floor'])==set(names) and set(saved['reference_values'])==set(gaps),'saved guard reference coverage')
        for a in names:
            close(saved['reference_floor'][a],floor[a],'saved guard floor')
            close(saved['numerical_tolerance'][a],numeric[a],'saved guard tolerance')
            if a in gaps:
                close(saved['reference_values'][a],gaps[a],'saved guard actual value')
                close(saved['slacks'][a],slacks[a],'saved guard actual slack')
            else:require(saved['slacks'][a] is None,'incomplete guard slack must be null')
        require(set(saved['violated_references'])==set(violated),'saved guard violated references')
        close(saved['tolerance_coefficient'],tolerance,'guard tolerance coefficient')
    return success,gaps


def linear_step(solved,current,floor,tolerance,names,domain):
    groups=current['aggregation']['active_groups'];ordered=[g[0]['a'] for g in groups]
    require(len(ordered)==len(names) and set(ordered)==set(names),'guard active reference coverage')
    require(solved['reference_names']==ordered,'guard MILP reference order')
    current_gaps=nearest(current,names);u=np.asarray(current['power_kw'],float);op=domain.as_dict()
    # The frozen driver passes trust_kw=None; its documented default is the
    # whole [-pmax,+pmax] power span. Guarded solver records omit trust_kw.
    trust=float(solved.get('trust_kw',2*np.max(np.asarray(domain.config.p_charge_max_kw))))
    lo=np.maximum(op['control_lower'],u.ravel()-trust);hi=np.minimum(op['control_upper'],u.ravel()+trust)
    gradients=[np.asarray(p['gradient_per_kw'],float).ravel() for group in groups for p in group]
    bound=max(1.,max(float(abs(g)@np.maximum(abs(lo-u.ravel()),abs(hi-u.ravel()))) for g in gradients))
    ideal=np.array([floor[a]-current_gaps[a] for a in ordered]);numeric=np.array([tolerance*(1+abs(floor[a])) for a in ordered])
    effective=np.maximum(-bound,ideal-numeric)
    close(solved['ideal_gain_lower'],ideal,'guard MILP ideal lower gain')
    close(solved['numerical_guard_tolerance'],numeric,'guard MILP numerical allowance')
    close(solved['effective_gain_lower'],effective,'guard MILP effective lower gain')
    if solved.get('power_kw') is not None:
        delta=np.asarray(solved['power_kw']).ravel()-u.ravel()
        gains=np.array([min(float(np.asarray(p['gradient_per_kw']).ravel()@delta) for p in group) for group in groups])
        close(solved['attained_reference_gains'],gains,'guard MILP attained per-reference gains')
        violation=float(np.max(np.maximum(effective-gains,0.)))
        close(solved['linear_reference_floor_violation'],violation,'guard MILP lower violation')
        if solved.get('success'):require(violation<=1e-7,'successful guard MILP violates lower gain constraint')


def audit_record(record,*,expected_guard=True,domain=None):
    selection=record['selection'];enabled=bool(selection.get('capped_reference_floor',False))
    require(enabled==bool(expected_guard),'selection guard differs from frozen experiment declaration')
    result=dict(expected_guard=bool(expected_guard),guard_checks=0,guard_MILP_checks=0,
        rejected_initials=0,rejected_proposals=0,legal_fixed_fallbacks=0)
    if not expected_guard:return result
    if domain is None:
        from .ev_mobile_v2_review import _domains
        domain=_domains(record)['mobile']
    names=[c['name'] for c in record['candidates']];metadata=selection['methods'];methods={m['name']:m for m in record['methods']}
    parent=metadata['jointgrad_fixed'].get('raw',{});raw=metadata['jointgrad_mobile'].get('raw',{})
    parent_profile=parent.get('selected_profile',{})
    if not parent.get('success') or not parent_profile.get('success'):
        require(not metadata['jointgrad_mobile'].get('success') and methods['jointgrad_mobile'].get('planning_failure'),
            'missing parent reference evidence must remain a planning failure')
        close(methods['jointgrad_mobile']['power_kw'],methods['jointgrad_fixed']['power_kw'],'missing-parent fixed fallback')
        result['legal_fixed_fallbacks']+=1;return result
    parent_gaps=nearest(parent_profile,names);cap=float(selection['task_cap'])
    floor={a:min(parent_gaps[a],cap) for a in names};tolerance=float(raw.get('guard_tolerance_coefficient',1e-7))
    require(tolerance==1e-7,'guard numerical coefficient differs from declared frozen policy')
    if not raw.get('success'):
        require(not metadata['jointgrad_mobile'].get('success') and methods['jointgrad_mobile'].get('planning_failure'),
            'incomplete guarded planning must retain a failure record')
        close(methods['jointgrad_mobile']['power_kw'],methods['jointgrad_fixed']['power_kw'],'incomplete guarded fixed fallback')
        result['legal_fixed_fallbacks']+=1;return result
    require(raw.get('reference_guard_enabled') and raw.get('reference_floor_parent')=='jointgrad_fixed','guard parent/enabled metadata')
    close(raw['reference_floor_cap'],cap,'guard cap equals declared task cap')
    require(set(raw['reference_floor'])==set(names),'guard floor reference coverage')
    for a in names:close(raw['reference_floor'][a],floor[a],'guard parent minimum/cap construction')
    checked={}
    for initial in raw['initial_candidates']:
        if 'profile' in initial and initial['profile'].get('success'):
            ok,_=guard(initial['profile'],floor,tolerance,names,initial.get('reference_guard'));result['guard_checks']+=1
            require(initial.get('reference_guard') is not None and bool(initial['success'])==ok,'initial guard acceptance')
            if ok:checked[initial['origin']]=initial['profile']
            else:result['rejected_initials']+=1
        elif initial.get('near_duplicate') and initial.get('success'):
            require(initial.get('canonical_origin') in checked,'near duplicate lacks guarded canonical action')
        else:require(not initial.get('success'),'successful initial lacks complete guarded profile')
    for start in raw['starts']:
        require(start['origin'] in checked,'guarded local start lacks guarded initial')
        current=checked[start['origin']]
        for row in start['iterations']:
            solved=row['raw'];linear_step(solved,current,floor,tolerance,names,domain);result['guard_MILP_checks']+=1
            if not solved.get('success') or row.get('repeated_AC_score_avoided'):continue
            trial=row.get('raw_AC_joint_profile');require(trial is not None,'guarded proposal lacks its complete AC attempt')
            ok,_=guard(trial,floor,tolerance,names,row.get('reference_guard'));result['guard_checks']+=1
            require(row.get('reference_guard') is not None,'guarded proposal missing saved guard details')
            should=bool(trial.get('success') and ok and trial['score']>current['score']+1e-9+1e-10*abs(current['score']))
            require(bool(row['accepted'])==should,'guarded complete-AC acceptance')
            if should:current=trial
            elif not ok:result['rejected_proposals']+=1
    ok,_=guard(raw['selected_profile'],floor,tolerance,names,raw.get('selected_reference_guard'));result['guard_checks']+=1
    require(ok and raw.get('selected_reference_guard') is not None,'final mobile action violates a reference floor')
    close(raw['selected_profile']['power_kw'],methods['jointgrad_mobile']['power_kw'],'guard final physical action')
    return result
