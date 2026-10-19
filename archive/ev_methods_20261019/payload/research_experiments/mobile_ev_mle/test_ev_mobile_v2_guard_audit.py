"""Arithmetic tampering checks; independent of the planner's guard helper."""
from copy import deepcopy
import numpy as np
import pytest

from . import ev_mobile_v2_guard_audit as audit
from . import ev_mobile_cases_v2 as cases


CAP=2*1.6448536269514722**2
NAMES=('star','clade_01','clade_02','clade_12')


def profile(gaps):
    u=np.zeros((8,3));u[:,0]=1.5;references=[];pairs=[]
    for a in NAMES:
        fits={b:dict(loss=100. if b==a else 100.+2*gaps[a]) for b in NAMES}
        references.append(dict(a=a,fits=fits))
        for b in NAMES:
            if b!=a:pairs.append(dict(a=a,b=b,success=True,half_gap=gaps[a],gradient_per_kw=np.ones((8,3)).tolist()))
    return dict(success=True,power_kw=u.tolist(),references=references,pairs=pairs,
        score=float(np.mean(list(gaps.values()))),aggregation=dict(
            nearest=[dict(a=a,half_gap=gaps[a]) for a in NAMES],
            active_groups=[[p for p in pairs if p['a']==a] for a in NAMES]))


def guard_record(p,floor):
    values={row['a']:row['half_gap'] for row in p['aggregation']['nearest']}
    numeric={a:1e-7*(1+abs(floor[a])) for a in NAMES};slack={a:values[a]-floor[a] for a in NAMES}
    bad=[a for a in NAMES if slack[a]<-numeric[a]]
    return dict(success=not bad,complete=True,reference_floor=dict(floor),reference_values=values,
        numerical_tolerance=numeric,slacks=slack,violated_references=bad,tolerance_coefficient=1e-7)


def fixture():
    gaps=dict(zip(NAMES,(1.,8.,.5,-.1)));parent=profile(gaps);floor={a:min(v,CAP) for a,v in gaps.items()}
    mobile=dict(success=True,reference_guard_enabled=True,reference_floor_parent='jointgrad_fixed',
        reference_floor_cap=CAP,reference_floor=floor,guard_tolerance_coefficient=1e-7,
        initial_candidates=[dict(origin='fixed_parent',success=True,profile=deepcopy(parent),
            reference_guard=guard_record(parent,floor))],starts=[],selected_profile=deepcopy(parent),
        selected_reference_guard=guard_record(parent,floor))
    record=dict(config=cases.case_config('remote8'),candidates=[dict(name=a) for a in NAMES],
        selection=dict(capped_reference_floor=True,task_cap=CAP,methods=dict(
            jointgrad_fixed=dict(success=True,raw=dict(success=True,selected_profile=parent)),
            jointgrad_mobile=dict(success=True,raw=mobile))),
        methods=[dict(name=a,power_kw=parent['power_kw'],planning_failure=None)
            for a in ('jointgrad_fixed','jointgrad_mobile')])
    return record,parent,floor,cases.service_domain('remote8',mobile=True)


def test_parent_gap_cap_initial_final_guard_and_negative_floors():
    record,_,floor,domain=fixture();result=audit.audit_record(record,domain=domain)
    assert result['guard_checks']==2
    assert floor['clade_01']==CAP and floor['clade_12']==-.1


@pytest.mark.parametrize('tamper',['flag','parent_cap','floor','initial','final'])
def test_guard_flags_caps_floors_and_saved_success_are_not_accepted_as_proof(tamper):
    record,parent,floor,domain=fixture();raw=record['selection']['methods']['jointgrad_mobile']['raw']
    if tamper=='flag':record['selection']['capped_reference_floor']=False
    elif tamper=='parent_cap':raw['reference_floor_cap']+=1.
    elif tamper=='floor':raw['reference_floor']['star']+=.1
    elif tamper=='initial':raw['initial_candidates'][0]['reference_guard']['success']=False
    else:
        bad=profile(dict(zip(NAMES,(.7,12.,.5,-.1))))
        raw['selected_profile']=bad;raw['selected_reference_guard']=guard_record(bad,floor)
    with pytest.raises(ValueError):audit.audit_record(record,domain=domain)


def step(current,floor):
    ordered=list(NAMES);gaps={r['a']:r['half_gap'] for r in current['aggregation']['nearest']}
    ideal=np.array([floor[a]-gaps[a] for a in ordered]);numeric=np.array([1e-7*(1+abs(floor[a])) for a in ordered])
    return dict(success=True,power_kw=current['power_kw'],reference_names=ordered,
        ideal_gain_lower=ideal,numerical_guard_tolerance=numeric,effective_gain_lower=ideal-numeric,
        attained_reference_gains=np.zeros(4),linear_reference_floor_violation=0.)


def test_each_reference_linear_lower_bound_and_explicit_tolerance_replay():
    _,parent,floor,domain=fixture();audit.linear_step(step(parent,floor),parent,floor,1e-7,NAMES,domain)


@pytest.mark.parametrize('field',['ideal_gain_lower','numerical_guard_tolerance','effective_gain_lower','attained_reference_gains'])
def test_tampered_milp_reference_lower_or_attained_gains_rejected(field):
    _,parent,floor,domain=fixture();raw=step(parent,floor);raw[field]=raw[field]+.01
    with pytest.raises(ValueError):audit.linear_step(raw,parent,floor,1e-7,NAMES,domain)


def test_larger_average_does_not_override_a_failed_reference_guard():
    record,parent,floor,domain=fixture();raw=record['selection']['methods']['jointgrad_mobile']['raw']
    trial=profile(dict(zip(NAMES,(.7,20.,.5,-.1))))
    row=dict(raw=step(parent,floor),accepted=True,raw_AC_joint_profile=trial,
        reference_guard=guard_record(trial,floor))
    raw['starts']=[dict(origin='fixed_parent',iterations=[row])]
    assert trial['score']>parent['score']
    with pytest.raises(ValueError,match='acceptance'):audit.audit_record(record,domain=domain)


def test_incomplete_parent_keeps_failed_status_and_exact_fixed_fallback():
    record,_,_,domain=fixture();meta=record['selection']['methods']
    meta['jointgrad_fixed']['raw']['success']=False
    meta['jointgrad_mobile']=dict(success=False,raw=dict(failure='parent unavailable'))
    record['methods'][1]['planning_failure']='parent unavailable'
    assert audit.audit_record(record,domain=domain)['legal_fixed_fallbacks']==1
    record['methods'][1]['planning_failure']=None
    with pytest.raises(ValueError):audit.audit_record(record,domain=domain)
