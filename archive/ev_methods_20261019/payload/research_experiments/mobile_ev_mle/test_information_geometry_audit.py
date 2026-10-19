"""Tests of the independent full-nuisance AC geometry rebuild."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from . import information_geometry_audit as audit
from .information_context import meter_to_record
from .likelihood import Observations, RadialCandidate, ac_predict
from .simulation import model
from .structured_information import InformationCenter, FutureForecast, StructuredInformationEvaluator


def test_full_projection_normalizes_units_and_retains_rank_guard():
    a=np.array([[2.,1.],[1.,3.],[4.,2.],[0.,5.]])
    b=np.c_[[1.,0.,1.,0.],[0.,1.,0.,1.],[1.,0.,1.,0.],np.zeros(4)]
    result=audit.full_project(a,b)
    rescaled=audit.full_project(a,b*np.array([1e-12,1e8,3e-5,1.]))
    assert result['rank_record']['rank']==2
    assert result['rank_record']['zero_column_count']==1
    np.testing.assert_allclose(result['information'],rescaled['information'],rtol=1e-13,atol=1e-12)
    near=audit.full_project(np.ones((2,2)),np.array([[1.,1.],[0.,2e-10]]))
    assert not near['rank_record']['rank_guard_passed']


def test_joint_projection_shares_lines_but_separates_states():
    rng=np.random.default_rng(452)
    h,n=rng.normal(size=(11,6)),rng.normal(size=(9,7))
    result=audit.joined_geometry(h,n,4,(0,1))
    a=np.vstack((h[:,:2],n[:,:2]))
    b=np.block([[h[:,2:4],h[:,4:],np.zeros((11,3))],
                [n[:,2:4],np.zeros((9,2)),n[:,4:]]])
    residual=a-b@np.linalg.lstsq(b,a,rcond=1e-10)[0]
    np.testing.assert_allclose(result['information'],residual.T@residual,rtol=1e-12,atol=1e-12)
    # Giving H and N separate line parameters expands nuisance freedom and
    # destroys shared-line information; this catches block-diagonal RX errors.
    wrong=audit.full_project(h[:,:2],h[:,2:])['information']+audit.full_project(n[:,:2],n[:,2:])['information']
    assert np.linalg.norm(result['information']-wrong)>.1
    assert np.linalg.eigvalsh(result['information']-wrong).min()>-1e-10


@pytest.fixture
def trial():
    star=RadialCandidate.from_supports('star',3)
    r=np.array([.004,.020,.023,.025]);x=np.array([.003,.013,.014,.015])
    rng=np.random.default_rng(593)
    p=.12+rng.normal(0,.005,(12,3));q=.04+rng.normal(0,.003,(12,3))
    sp=.08+rng.normal(0,.004,12);sq=.025+rng.normal(0,.002,12);root=np.ones(12)
    physical=ac_predict(star,r,x,p,q,sp,sq,root)
    history=Observations(p=p,q=q,v=physical['v'],root_v=root,master_p=physical['master_p'],master_q=physical['master_q'])
    meter=model(3);future=FutureForecast(p=np.tile(p.mean(axis=0),(4,1)),q=np.tile(q.mean(axis=0),(4,1)),
        source_p=np.full(4,sp.mean()),source_q=np.full(4,sq.mean()),root_v=np.ones(4))
    centers=[];saved=[]
    for support in sorted(audit.SUPPORTS):
        candidate=RadialCandidate.from_supports('clade_'+''.join(map(str,support)),3,[support])
        old={s:i for i,s in enumerate(star.supports)}
        cr=np.array([r[old[s]] if s in old else 0. for s in candidate.supports])
        cx=np.array([x[old[s]] if s in old else 0. for s in candidate.supports])
        center=InformationCenter(id='h0_'+candidate.name,anchor_id='h0',target_support=support,candidate=candidate,r=cr,x=cx,
            history_p=p,history_q=q,history_source_p=sp,history_source_q=sq,history_root_v=root,future=future)
        centers.append(center)
        saved.append(dict(id=center.id,anchor_id=center.anchor_id,target_support=list(support),name=candidate.name,
            supports=[list(s) for s in candidate.supports],r=cr.tolist(),x=cx.tolist(),history_p=p.tolist(),history_q=q.tolist(),
            history_source_p=sp.tolist(),history_source_q=sq.tolist(),history_root_v=root.tolist(),
            future={k:getattr(future,k).tolist() for k in ('p','q','source_p','source_q','root_v')},metadata={}))
    ev=StructuredInformationEvaluator(history,centers,meter,q_over_p=.2)
    pp=np.zeros((4,3));pp[:,0]=[.3,0.,.3,0.]
    plan=dict(site=0,p=pp.tolist(),q=(.2*pp).tolist())
    evaluation=ev.evaluate(0,20*pp[:,0],with_gradient=False)
    assert evaluation['success']
    return dict(seed=593,quality='uniform',run_id='self_contained',history_id='h0',
        history_observations={k:None if getattr(history,k) is None else np.asarray(getattr(history,k)).tolist() for k in
            ('p','q','v','root_v','master_p','master_q','source_p','source_q')},
        measurement_models=dict(history=meter_to_record(meter),future=meter_to_record(meter)),
        config=dict(information=dict(rcond=1e-10,scales_pu=[.003,.003])),device_constraints=dict(q_over_p=.2),
        selection=dict(activated=True,centres=saved,searches=dict(seeds=dict(success=False,error='fixture_not_generated'),nominal=dict(success=False,error='fixture_not_generated'),robust=dict(success=True,selected_site=0,score=evaluation['score'],
            plan=plan,evaluation=deepcopy(evaluation),site_plans={'0':plan},site_evaluations={'0':evaluation}))))


def test_saved_trial_is_independent_and_catches_tampering(trial,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('auditor called evaluator')
    monkeypatch.setattr(StructuredInformationEvaluator,'evaluate',forbidden)
    result=audit.audit_trial_geometry(trial)
    assert result['success'],result['errors']
    assert result['maximum_matrix_relative_error']<1e-9
    tampered=deepcopy(trial)
    tampered['selection']['searches']['robust']['site_evaluations']['0']['targets'][0]['information_joint'][0][0]*=1.01
    result=audit.audit_trial_geometry(tampered)
    assert not result['success']
    assert any('information_joint' in error for error in result['errors'])


def test_ev_total_pq_once_with_recorded_ratio_and_shared_source(trial,monkeypatch):
    seen=[];original=audit.history_white_jacobian
    def captured(candidate,fit,obs,meter,**kwargs):
        seen.append((fit.p.copy(),fit.q.copy(),fit.source_p.copy(),len(fit.p)))
        return original(candidate,fit,obs,meter,**kwargs)
    monkeypatch.setattr(audit,'history_white_jacobian',captured)
    result=audit.audit_trial_geometry(trial)
    assert result['success'],result['errors']
    c=trial['selection']['centres'][0];f=c['future'];expected=np.array([.3,0.,.3,0.])
    ev=[row for row in seen if row[3]==4 and np.max(row[0][:,0]-np.array(f['p'])[:,0])>.01]
    assert len(ev)==3
    for p,q,source,_ in ev:
        np.testing.assert_allclose(p[:,0]-np.array(f['p'])[:,0],expected,atol=1e-15)
        np.testing.assert_allclose(q[:,0]-np.array(f['q'])[:,0],.2*expected,atol=1e-15)
        np.testing.assert_array_equal(source,f['source_p'])


def test_all_three_supports_and_positive_tasks_are_required(trial):
    trial['selection']['centres'].pop()
    result=audit.audit_trial_geometry(trial)
    assert not result['success']
    assert any('anchor_complete_supports' in error for error in result['errors'])


def test_v2_saved_task_weights_and_non_mle_anchor_meaning(trial):
    weights=[.1,.3,.6]
    rows=[]
    for center,weight in zip(trial['selection']['centres'],weights):
        center['metadata']['historical_anchor_kind']='feasible_perturbed_H_point'
        rows.append(dict(target_support=center['target_support'],weight=weight))
    trial['selection']['target_weight_record']=dict(target_records=rows)
    trial['selection']['searches']['uniform']=dict(success=False,error='fixture_not_generated')
    search=trial['selection']['searches']['robust']
    for ev in (search['evaluation'],search['site_evaluations']['0']):
        for target,weight in zip(ev['targets'],weights):
            target['target_task_weight']=weight
            target['metadata']['historical_anchor_kind']='feasible_perturbed_H_point'
        ev['pairs'][0]['score']=sum(t['score']*w for t,w in zip(ev['targets'],weights))
        ev['pairs'][0]['target_task_weights']=weights
        ev['score']=ev['pairs'][0]['score']
    search['score']=search['evaluation']['score']
    result=audit.audit_trial_geometry(trial)
    assert result['success'],result['errors']
    assert all(t['anchor_kind']=='feasible_perturbed_H_point' for e in result['evaluations'] for t in e['targets'])
    assert all(t['target_task_weight'] in weights for e in result['evaluations'] for t in e['targets'])


def test_archive_cannot_claim_success_without_evaluations(trial):
    search=trial['selection']['searches']['robust']
    search['site_evaluations']={}
    result=audit.audit_trial_geometry(trial)
    assert not result['success']
    assert 'robust:successful_search_missing_saved_evaluations' in result['errors']


def test_tiny_engineering_weight_uses_strict_tolerance(trial):
    trial['selection']['searches']['robust']['site_evaluations']['0']['targets'][0]['weight'][0][0] *= 1.00001
    result=audit.audit_trial_geometry(trial)
    assert not result['success']
    assert any(error.endswith(':weight') for error in result['errors'])


def test_required_weight_failure_is_retained_as_unavailable(trial):
    trial['selection']['searches']={}
    trial['selection']['information_not_started_reason']='required_target_weight_failure'
    trial['selection']['target_weight_record']=dict(success=False,failures=[dict(candidate='clade_01',reason='H_fit_failure')])
    result=audit.audit_trial_geometry(trial)
    assert result['success'],result['errors']
    assert result['geometry_coverage_status']=='partially_unavailable'
    assert result['independent_AC_rebuild_calls']==0
    assert result['target_weight_record']['failures'][0]['reason']=='H_fit_failure'
    assert result['information_not_started_reason']=='required_target_weight_failure'
