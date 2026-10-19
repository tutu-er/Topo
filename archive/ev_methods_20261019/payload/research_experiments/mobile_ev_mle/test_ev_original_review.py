"""Targeted tamper/denominator tests; no selector, generator or fitting."""
from copy import deepcopy
import base64
import hashlib
import inspect

import numpy as np
import pytest

from . import ev_original_review as review
from .likelihood import MeasurementModel, Observations, RadialCandidate
from .ev_mobile_service_domain import MobileServiceConfig, build_mobile_service_domain
from .information_fit_recovery import write_json


def meter():
    return MeasurementModel(.01, .02, .001, .03, .04,
                            bias_p=.0001, bias_q=-.0002, bias_v=.0003)


def fit_fixture():
    c=RadialCandidate.from_supports('star',3,())
    r=np.full(c.m,.005);x=np.full(c.m,.001);p=np.full((2,3),.1);q=np.full((2,3),.02)
    sp=np.full(2,.03);sq=np.full(2,.01);m=meter()
    phy=review.forward_physics(c,r,x,p,q,sp,sq,np.ones(2))
    means={k:phy[k]+(getattr(m,'bias_'+k) if k in ('p','q','v') else 0)
           for k in ('p','q','v','master_p','master_q')}
    obs=Observations(**means,root_v=np.ones(2))
    normalizer=sum(np.log(2*np.pi*np.broadcast_to(getattr(m,k+'_sigma'),np.shape(v))**2).sum()
                   for k,v in means.items())
    row=dict(name='star',supports=[list(s) for s in c.supports],r=r,x=x,p=p,q=q,source_p=sp,source_q=sq,
        bias_p=m.bias_p,bias_q=m.bias_q,bias_v=m.bias_v,predicted=means,loss=0.,half_loss=0.,
        loglik=-.5*normalizer,r_ohm=r*8,x_ohm=x*8,success=True,global_MLE_certified=False)
    return c,m,phy,obs,row


def test_saved_fit_replay_has_no_ac_solver_or_new_fit_dependency(monkeypatch):
    c,m,phy,obs,row=fit_fixture()
    from . import likelihood
    def forbidden(*a,**kw):raise AssertionError('new AC model/fit invoked')
    monkeypatch.setattr(likelihood,'ac_predict',forbidden)
    monkeypatch.setattr(likelihood,'fit_candidate',forbidden)
    audit=review.replay_fits([row],{'star':c},obs,m,{},complete=True,config={'zbase_ohm':8},domain='numeric')
    assert audit['replayed_states']==1 and audit['audits'][0]['success']
    assert audit['audits'][0]['full_SSE']<1e-20


@pytest.mark.parametrize('key,value', [('loglik',1.),('r_ohm',[1,1,1,1]),('half_loss',1.)])
def test_saved_gaussian_and_unit_tampering_rejected(key,value):
    c,m,_,obs,row=fit_fixture();row[key]=value
    with pytest.raises(ValueError):
        review.replay_fits([row],{'star':c},obs,m,{},complete=True,config={'zbase_ohm':8},domain='numeric')


def test_failed_candidate_remains_in_coverage_denominator():
    c,m,_,obs,row=fit_fixture();other=RadialCandidate.from_supports('other',3,((0,1),))
    with pytest.raises(ValueError,match='disappear'):
        review.replay_fits([row],{'star':c,'other':other},obs,m,{},complete=False,config={'zbase_ohm':8},domain='numeric')
    audit=review.replay_fits([row],{'star':c,'other':other},obs,m,{'other':'solver failure'},
        complete=False,config={'zbase_ohm':8},domain='numeric')
    assert audit['replayed_states']==1


def test_original_qp_residual_and_tangent_use_absolute_physical_box():
    model=dict(c=[-.25],B=[[0.]],A0=[[1.]],At=[[[0.]]],
        domain=dict(physical_lower=[0.],physical_upper=[1.],base_center=[0.],shift=[[1.]]))
    u=np.array([.2]);xi=np.array([.5]);z=xi-u;r=np.array([.05]);primal=float(r@r);g=2*r
    correction=float(np.minimum(g*(0-xi),g*(1-xi)).sum())
    guard=256*np.finfo(float).eps*(1+abs(primal)+abs(correction)+float(abs(g)@(np.array([0.])+1+2*abs(xi))))
    saved=dict(physical_parameter=xi,z=z,residual=r,primal_upper=primal,
        tangent_lower_numeric=max(0.,min(primal,primal+correction-guard)),numeric_guard=guard,success=True)
    out=review.replay_qp(model,u,saved);assert out['lower']<=out['upper']
    bad=deepcopy(saved);bad['physical_parameter']=[-.1]
    with pytest.raises(ValueError,match='physical box'):review.replay_qp(model,u,bad)
    bad=deepcopy(saved);bad['z']=[.5]
    with pytest.raises(ValueError,match='moving map'):review.replay_qp(model,u,bad)
    fallback=dict(saved,success=False,fallback=True,tangent_lower_numeric=0.)
    fallback.pop('numeric_guard')
    assert review.replay_qp(model,u,fallback)['declared_exception_fallback']


def test_affine_witness_checks_whole_box_and_keeps_cross_terms():
    model=dict(c=[.1],B=[[.2,.3]],A0=[[1.]],At=[[[.5]],[[.7]]],
        domain=dict(physical_lower=[0.],physical_upper=[1.],base_center=[.2],
                    shift=[[.1,.2]],action_lower=[-1.,-1.],action_upper=[1.,1.]))
    good=dict(physical_intercept=[.5],physical_slope=[[.1,.1]],validation={'method':'complete_control_box'})
    upper,valid=review.witness_box_upper(model,good,set(),0.,.5)
    assert valid
    a,B,Q=review.polynomial(model,good);assert (0,1) in Q
    for u in ([-1,-1],[-1,1],[0,0],[1,-1],[1,1]):
        u=np.asarray(u);xi=np.asarray(good['physical_intercept'])+np.asarray(good['physical_slope'])@u
        exact=np.asarray(model['c'])+np.asarray(model['B'])@u+(np.asarray(model['A0'])+np.einsum('k,krp->rp',u,np.asarray(model['At'])))@(xi-np.asarray(model['domain']['base_center'])-np.asarray(model['domain']['shift'])@u)
        polynomial=a+B@u+sum(v*u[t]*u[s] for (t,s),v in Q.items())
        np.testing.assert_allclose(polynomial,exact)
        assert .5*float(exact@exact)<=upper+1e-10
    bad=dict(good,physical_slope=[[1.,1.]])
    with pytest.raises(ValueError,match='global physical'):review.witness_box_upper(model,bad,set(),0.,.5)


def test_operation_replay_checks_full_x_and_independent_path_energy():
    cfg=MobileServiceConfig(n_slots=2,n_sites=3,dt_hours=1.,p_charge_max_kw=1.,
        allow_discharge=False,battery_capacity_kwh=10.,initial_soc=.5,soc_min=0.,soc_max=1.,
        terminal_energy_kwh=6.,eta_charge=1.,eta_discharge=1.,mobile=False,
        driver_budget=0.,max_moves=0,max_throughput_kwh=2.)
    d=build_mobile_service_domain(cfg);x=np.zeros(d.model.n_variables);ix=d.indices
    x[ix.connected_arc[:,0]]=1;x[ix.charge[:,0]]=.5;x[ix.u[:,0]]=.5;x[ix.mode]=1
    x[ix.energy]=[5,5.5,6]
    saved=d.check(x);assert saved['feasible']
    assert review.replay_operation_record(saved,cfg)['grid_energy_kwh']==pytest.approx(1)
    bad=deepcopy(saved);bad['route'][0]['origin']=1
    with pytest.raises(ValueError,match='continuity'):review.replay_operation_record(bad,cfg)
    bad=deepcopy(saved);bad['energy_kwh'][1]+=1
    with pytest.raises(ValueError,match='energy trajectory'):review.replay_operation_record(bad,cfg)


def test_true_future_power_includes_ev_exactly_once_and_noiseless_is_separate():
    c,m,_,_,row=fit_fixture();H=2;T=2
    truth=dict(family=(),r=row['r'],x=row['x'],p=np.full((4,3),.1),q=np.full((4,3),.02),
        source_p=np.full(4,.03),source_q=np.full(4,.01),root_v=np.ones(4),
        white={k:np.zeros((4,3) if k in ('p','q','v') else 4) for k in review.CHANNELS})
    power=np.array([[6,0,0],[0,-2,0.]])
    phy=review.forward_physics(c,row['r'],row['x'],truth['p'][H:]+power/20,truth['q'][H:],truth['source_p'][H:],truth['source_q'][H:],np.ones(T))
    trial=dict(config=dict(history=H,horizon=T,base_kw=20.,dt_hours=.25),truth=truth)
    future=dict(power_kw=power,physical={k:phy[k] for k in ('p','q','v','master_p','master_q')},
        observations=None,grid_net_kwh=1.,grid_throughput_kwh=2.)
    assert review.future_audit(trial,future,m)['independent_AC_replays']==1
    bad=deepcopy(future);bad['physical']['p']=bad['physical']['p']+power/20
    with pytest.raises(ValueError,match='EV once'):review.future_audit(trial,bad,m)


def test_snapshot_validates_old_bytes_and_reports_later_current_version(tmp_path):
    payload=b'frozen scientific version\n';digest=hashlib.sha256(payload).hexdigest();name='absent_review_test_source.py'
    write_json(tmp_path/'source_snapshot.json.gz',{name:dict(sha256=digest,bytes_base64=base64.b64encode(payload).decode())})
    result=review.source_audit(tmp_path,dict(source_sha256={name:digest}))
    assert result['success'] and not result['current_source_matches']
    assert result['current_source_mismatches'][0]['current_sha256'] is None
    with pytest.raises(ValueError,match='snapshot bytes'):
        review.source_audit(tmp_path,dict(source_sha256={name:'0'*64}))


def test_reviewer_does_not_import_scientific_selector_or_generator():
    import ast
    source=inspect.getsource(review);tree=ast.parse(source)
    imported=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(n.endswith(('ev_original_study','ev_original_domain_milp')) for n in imported)
    names={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not {'solve','solve_prices','fit_candidate','ac_predict','run_trial','make_case'}&names


def test_reference_prediction_is_historical_not_future_true_parameters():
    c,m,_,obs,row=fit_fixture();T=2
    bg=dict(p=np.tile(np.mean(obs.p-m.bias_p,axis=0),(T,1)),
            q=np.tile(np.mean(obs.q-m.bias_q,axis=0),(T,1)),
            source_p=np.full(T,np.mean(row['source_p'])),source_q=np.full(T,np.mean(row['source_q'])))
    model=dict(a='star',reference_r=row['r'],reference_x=row['x'],reference_background=bg,future_root_v=np.ones(T))
    review.history_reference_audit(model,obs,m,{'star':row},{'horizon':T})
    bad=deepcopy(model);bad['reference_r']=np.asarray(bad['reference_r'])*1.01
    with pytest.raises(ValueError,match='H fit'):review.history_reference_audit(bad,obs,m,{'star':row},{'horizon':T})
    bad=deepcopy(model);bad['reference_background']['source_p']+=.001
    with pytest.raises(ValueError,match='from H'):review.history_reference_audit(bad,obs,m,{'star':row},{'horizon':T})


def test_backend_sign_and_partial_start_mean_submission_only():
    master=dict(backend='gurobi',scale=2.,numeric_guard=.001,solver_best_bound=6.,surrogate_incumbent=5.,
        bound_consistent=True,status=1,warm_start=dict(filled_variables=5,provided_to_solver=True,
            operation_validation=dict(feasible=True)),backend_details=dict(objective_sense='min',objective_transform='none',
            backend_requested='gurobi',backend_used='gurobi',fallback_used=False,start_provided_count=5,
            start_supplied=True,start_used=True,gurobi_status=9,threads=1,seed=0,backend_seconds=.5),
        compressions=[dict(product_cost_error_upper=1.,residual_square_error_upper=2.,
            qr_upward_cost_guard=.1,pwl_score_error_upper=1.6)])
    audit=review.master_backend_audit(master,dict(exchange=dict(backend='gurobi')))
    assert audit['raw_MIN_lower']<=audit['raw_MIN_incumbent'] and audit['warm_start_submitted']
    assert 'submission' in audit['interpretation']
    bad=deepcopy(master);bad['backend_details']['objective_transform']='negated'
    with pytest.raises(ValueError,match='minimization'):review.master_backend_audit(bad,dict(exchange=dict(backend='gurobi')))
    bad=deepcopy(master);bad['backend_details']['start_provided_count']=0
    with pytest.raises(ValueError,match='count'):review.master_backend_audit(bad,dict(exchange=dict(backend='gurobi')))


def test_noiseless_distance_retains_missing_history_and_future_candidates():
    _,m,_,obs,row=fit_fixture()
    future=Observations(**{k:np.asarray(row['predicted'][k])[1:] for k in ('p','q','v','master_p','master_q')},root_v=[1.])
    detail=dict(candidate='star',success=False,reason='required_historical_fit_failure',
        attained_joint_SSE=0.,added_half_SSE=None,future_channel_SSE={k:0. for k in row['predicted']},historical_SSE=0.)
    audit=review.noiseless_distance_audit(detail,[row],{},future,m,1)
    assert audit['available'] and not audit['historical_baseline_available']
    bad=dict(detail,added_half_SSE=0.)
    with pytest.raises(ValueError,match='missing historical'):review.noiseless_distance_audit(bad,[row],{},future,m,1)
    absent=dict(candidate='star',success=False,reason='required_future_fit_missing',
        attained_joint_SSE=None,added_half_SSE=None,future_channel_SSE=None,historical_SSE=None)
    assert not review.noiseless_distance_audit(absent,[],{},future,m,1)['available']


def test_asset_unit_audit_uses_public_constants_and_explicit_simulation_lengths():
    c=RadialCandidate.from_supports('truth',3,())
    config=dict(base_kv=.4,base_kw=20.,zbase_ohm=8.,asset=dict(R_ohm_km=[.642,.225,.208],
        X_ohm_km=[.083,.080,.080],declared_length_km=[.025,.150],length_is_synthetic_family=True,source='declared primary source'),
        asset_bounds=dict(r_bounds=[.208*.025/8,.642*.150/8],x_bounds=[.080*.025/8,.083*.150/8]))
    assets=[dict(support=s,cable_mm2=50,length_km=.1) for s in c.supports]
    trial=dict(config=config,truth=dict(family=(),assets=assets,r=[.642*.1/8]*c.m,x=[.083*.1/8]*c.m))
    assert review.asset_unit_audit(trial)['edges']==4
    bad=deepcopy(trial);bad['truth']['r'][0]*=1000
    with pytest.raises(ValueError,match='ohm/km'):review.asset_unit_audit(bad)
