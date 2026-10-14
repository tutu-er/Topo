"""Independent AC, bounds, boundary, scenario and counterfactual pairing checks."""
from dataclasses import replace
import json
import numpy as np
import pytest
from . import likelihood as lk
from .candidates import RadialCandidate
from .gradient_price import _models_for_future
from .symmetric_ac_local import mean_and_jacobian,whiten_observations
from .ev_original_ac_check import build_multisite_context
from . import ev_prediction_ensemble as en


def point(candidate,history,meter,r=.004,x=.002):
    t,n=history.t,history.n
    p=history.p-np.asarray(meter.bias_p);q=history.q-np.asarray(meter.bias_q)
    sp=np.full(t,.1);sq=np.full(t,.03)
    r=np.broadcast_to(r,(candidate.m,)).copy();x=np.broadcast_to(x,(candidate.m,)).copy()
    g=mean_and_jacobian(candidate,r,x,p,q,sp,sq,history.root_v,meter,channels=en._observed_channels(history))
    cost=float(np.sum((g['mean']-whiten_observations(history,meter,en._observed_channels(history)))**2))
    predicted=dict(g['physical'])
    for k in ('p','q','v'):predicted[k]=predicted[k]+getattr(meter,'bias_'+k)
    norm=sum(np.log(2*np.pi*np.broadcast_to(getattr(meter,k+'_sigma'),np.asarray(getattr(history,k)).shape)**2).sum()
        for k in en._observed_channels(history))
    return lk.FitResult(candidate.name,r,x,p,q,sp,sq,np.broadcast_to(meter.bias_p,(n,)).copy(),
        np.broadcast_to(meter.bias_q,(n,)).copy(),np.broadcast_to(meter.bias_v,(n,)).copy(),
        cost,float(-.5*(cost+norm)),True,1,'feasible AC test point',0,0.,predicted,
        dict(global_optimality_certified=False))


@pytest.fixture
def sample():
    bank=[RadialCandidate.from_supports('star',3,()),
          RadialCandidate.from_supports('clade_01',3,((0,1),))]
    meter=lk.MeasurementModel(p_sigma=.003,q_sigma=.003,v_sigma=.0005,master_p_sigma=.003,master_q_sigma=.003,
        bias_p=np.array([-.0001,0,.0001]),bias_q=np.array([.0001,0,-.0001]),bias_v=np.array([-2e-5,0,2e-5]))
    common=np.array([.11,.13,.12,.14])
    p=common[:,None]*np.ones((1,3));q=.3*p
    physical=lk.ac_predict(bank[0],np.full(4,.004),np.full(4,.002),p,q,np.full(4,.1),np.full(4,.03),np.ones(4))
    history=lk.Observations(p=p+meter.bias_p,q=q+meter.bias_q,v=physical['v']+meter.bias_v,root_v=np.ones(4),
        master_p=physical['master_p'],master_q=physical['master_q'])
    fits={c.name:point(c,history,meter) for c in bank}
    opts=dict(r_bounds=(.001,.02),x_bounds=(.001,.01),p_bounds=(-1.,1.),q_bounds=(-1.,1.),
        source_bounds=(-.5,.5),voltage_bounds=(.85,1.15),max_nfev=80,tol=1e-9,linear_solver='exact',multistart=1)
    explanations=dict(success=True,explanations={n:[dict(id=n+'__center',candidate=n,fit=f,kind='test_center')] for n,f in fits.items()})
    scenarios=en.build_forecast_scenarios(history,meter,fits,2)
    return history,bank,fits,meter,opts,explanations,scenarios


def test_fixed_lines_are_exact_not_a_narrow_box(sample):
    H,bank,fits,meter,opts,_,_=sample;c=bank[0];fit=fits[c.name];boxes=en._boxes(c,H,opts)
    r=fit.r.copy();r[-1]+=.0001
    alt,cost=en._profile_fixed_lines(c,H,meter,en._state(fit),r,fit.x,boxes,max_nfev=80,tol=1e-10,voltage_bounds=(.85,1.15))
    assert alt.success and np.array_equal(alt.r,r) and np.array_equal(alt.x,fit.x)
    g=mean_and_jacobian(c,alt.r,alt.x,alt.p,alt.q,alt.source_p,alt.source_q,H.root_v,meter)
    independent=float(np.sum((g['mean']-whiten_observations(H,meter))**2))
    assert alt.loss==pytest.approx(independent,abs=1e-9)
    assert cost['ac_calls']>=1 and alt.diagnostics['source_node']==1


def test_history_builder_checks_actual_AC_and_retains_small_explanations(sample):
    H,bank,fits,meter,opts,_,_=sample
    answer=en.build_history_explanations(H,bank,fits,meter,fit_options=opts,radius_fraction=.01,max_backtracks=0,history_gate_half_sse=10)
    assert answer['success'],answer['diagnostics']
    assert set(answer['explanations'])=={c.name for c in bank}
    for c in bank:
        kept=answer['explanations'][c.name];assert 1<=len(kept)<=3
        for row in kept:
            en._feasible_point(row['fit'],en._boxes(c,H,opts))
            g=en._geometry(c,row['fit'],H,meter,en._observed_channels(H))
            cost=float(np.sum((g['mean']-whiten_observations(H,meter))**2))
            assert row['fit'].loss==pytest.approx(cost,abs=1e-8)
            assert row['history_gap_half_sse']<=10
    assert not answer['diagnostics']['historical_support_is_confidence_set']
    json.dumps(answer['records'],allow_nan=False)


def test_activity_projection_keeps_blocked_original_ray(sample):
    H,bank,fits,meter,opts,_,_=sample;c=bank[0]
    lower=point(c,H,meter,r=.001,x=.001)
    answer=en.build_history_explanations(H,[c],{c.name:lower},meter,fit_options=opts,
        radius_fraction=.01,max_backtracks=0,history_gate_half_sse=100)
    assert answer['success']
    attempts=answer['diagnostics']['candidate_diagnostics'][c.name]['attempts']
    assert any(a['reason']=='active_boundary_blocks_original_ray' for a in attempts)
    profiled=[a for a in attempts if 'profile' in a]
    assert profiled and any(a['direction_was_line_tangent_projected'] for a in profiled)
    assert any(a['retained'] for a in profiled)
    for a in profiled:
        if a['direction_was_line_tangent_projected']:
            assert not a['original_smallest_profile_direction']
            assert np.min(a['actual_line_direction'])>=0


def test_missing_required_center_is_kept_as_failure(sample):
    H,bank,fits,meter,opts,_,_=sample
    out=en.build_history_explanations(H,bank,{'star':fits['star']},meter,fit_options=opts,max_explanations=1)
    assert not out['success'] and out['explanations']['clade_01']==[]
    assert 'clade_01' in out['diagnostics']['failures']


def test_profile_exception_keeps_spent_AC_calls(sample,monkeypatch):
    H,bank,fits,meter,opts,_,_=sample
    def failing(*args,call_counter=None,**kwargs):
        call_counter['ac_calls']+=3
        raise RuntimeError('deliberate profile AC failure')
    monkeypatch.setattr(en,'_profile_fixed_lines',failing)
    out=en.build_history_explanations(H,[bank[0]],fits,meter,fit_options=opts,max_backtracks=0)
    attempts=out['diagnostics']['candidate_diagnostics']['star']['attempts']
    actual=[a for a in attempts if a.get('reason')=='profile_exception']
    assert actual and all(a['ac_calls_spent']==3 for a in actual)
    assert out['diagnostics']['ac_calls']==1+3*len(actual)
    assert out['diagnostics']['fit_calls']==len(actual)


def test_unanchored_or_free_calibration_is_rejected(sample):
    H,_,fits,meter,_,_,_=sample
    bad=replace(H,master_p=None,master_q=None)
    with pytest.raises(ValueError,match='source requires'):en.build_forecast_scenarios(bad,meter,fits,2)
    with pytest.raises(ValueError,match='calibrated'):en.build_forecast_scenarios(H,replace(meter,fit_bias=True),fits,2)
    with pytest.raises(TypeError):en.build_forecast_scenarios({'history':H,'truth':'unavailable'},meter,fits,2)


def test_forecast_scenarios_are_H_only_deterministic_and_have_all_channels(sample,monkeypatch):
    H,_,fits,meter,_,_,_=sample
    def prohibited(*a,**k):raise AssertionError('no RNG belongs in the H-only three-point construction')
    monkeypatch.setattr(np.random,'default_rng',prohibited)
    out=en.build_forecast_scenarios(H,meter,fits,2)
    assert len(out['scenarios'])==3 and sum(s['weight'] for s in out['scenarios'])==1.
    assert [s['control_gain'][0,0] for s in out['scenarios']]==pytest.approx([1.,.95,1.05])
    for scene in out['scenarios']:
        assert set(scene['offsets'])==set(fits)
        assert scene['source_interface']==1
    diag=out['diagnostics']
    assert not diag['uses_truth'] and not diag['uses_future_noise'] and not diag['predictive_distribution_calibrated']
    assert 'not posterior' in diag['scenario_weights_meaning']


def test_declared_floors_do_not_shrink_flat_history_uncertainty(sample):
    H,_,fits,meter,_,_,_=sample
    flat=replace(H,p=np.tile(H.p[0],(H.t,1)),q=np.tile(H.q[0],(H.t,1)))
    constant={n:replace(f,source_p=np.full(H.t,.1),source_q=np.full(H.t,.03)) for n,f in fits.items()}
    out=en.build_forecast_scenarios(flat,meter,constant,2,load_floor_pu=.004,source_floor_pu=.007)
    mode=np.asarray(out['diagnostics']['mode_records']['star']['declared_mode_pu'])
    assert np.min(abs(mode[:6]))>=.004 and np.min(abs(mode[6:]))>=.007


def test_duplicate_center_bundles_are_exactly_merged_with_aliases(sample):
    _,bank,_,_,_,explanations,scenarios=sample
    bundles=en._bundles(bank,explanations,scenarios,(0,1),np.zeros((2,3)),20.)
    assert len(bundles)==3
    assert sum(b['weight'] for b in bundles)==pytest.approx(1.)
    assert all(len(b['aliases'])==2 for b in bundles)
    assert all(b['alias_records'][1]['missing_flat_fallback_candidates']==['star','clade_01'] for b in bundles)


def test_one_real_alternative_prevents_bundle_merge(sample):
    H,bank,_,meter,_,explanations,scenarios=sample
    alt=point(bank[0],H,meter,r=.0041)
    explanations['explanations']['star'].append(dict(id='star__flat',fit=alt,kind='test_flat'))
    bundles=en._bundles(bank,explanations,scenarios,(0,1),np.zeros((2,3)),20.)
    assert len(bundles)==6 and all(len(b['aliases'])==1 for b in bundles)
    assert sum(b['weight'] for b in bundles)==pytest.approx(1.)


def test_override_none_preserves_default_and_gain_scales_control_map(sample):
    H,bank,fits,meter,opts,_,scenarios=sample;ref=np.zeros((2,3));ref[:,0]=2.
    default=build_multisite_context(H,bank,fits,meter,meter,ref,0.,6.)
    explicit=build_multisite_context(H,bank,fits,meter,meter,ref,0.,6.,forecast_overrides=None)
    assert default['success'] and explicit['success']
    for a,b in zip(default['models'],explicit['models']):
        for key in ('c','B','A0','At'):assert np.array_equal(a[key],b[key])
    overrides={}
    for c in bank:
        f=fits[c.name];overrides[c.name]=dict(p=np.full((2,3),.12)+.95*ref/20.,
            q=np.full((2,3),.04),source_p=np.full(2,np.mean(f.source_p)),source_q=np.full(2,np.mean(f.source_q)),
            root_v=np.ones(2),control_gain=np.full((2,3),.95))
    built=build_multisite_context(H,bank,fits,meter,meter,ref,0.,6.,forecast_overrides=overrides)
    assert built['success'],built['diagnostics']
    model=built['models'][0];U=np.asarray(model['domain']['shift'])[-16:]
    assert (U@ref.ravel()).reshape(2,8)[:,:3]==pytest.approx(.95*ref/20.)
    assert built['forecasts']['star']['p']==pytest.approx(overrides['star']['p'])


def test_contexts_preserve_all_pairs_weights_and_original_H_baselines(sample):
    H,bank,fits,meter,opts,explanations,scenarios=sample
    alternative=point(bank[0],H,meter,r=.0041)
    explanations['explanations']['star'].append(dict(id='star__flat',fit=alternative,kind='test_flat'))
    out=en.build_ensemble_contexts(H,bank,fits,meter,meter,np.zeros((2,3)),0.,6.,
        explanations=explanations,scenarios=scenarios,line_bounds=dict(r_bounds=opts['r_bounds'],x_bounds=opts['x_bounds']))
    assert out['success'],out['diagnostics']
    assert len(out['contexts'])==6 and len(out['models'])==12
    assert len({m['name'] for m in out['models']})==12
    assert sum(m['task_weight'] for m in out['models'])==pytest.approx(1.)
    assert all(m['competitor_history_cost']==fits[m['b']].loss for m in out['models'])
    assert any(m['expansion_competitor_history_cost']!=m['competitor_history_cost'] for m in out['models'])
    assert not out['diagnostics']['alternative_futures_stacked']
    json.dumps(out['context_records'],allow_nan=False)


def test_override_candidate_or_physical_root_cannot_disappear(sample):
    H,bank,fits,meter,_,_,_=sample
    bad={'star':dict(p=np.zeros((2,3)),q=np.zeros((2,3)),source_p=np.zeros(2),source_q=np.zeros(2),root_v=np.ones(2))}
    out=build_multisite_context(H,bank,fits,meter,meter,np.zeros((2,3)),0.,6.,forecast_overrides=bad)
    assert not out['success'] and out['models']==[]
    assert 'every candidate' in out['diagnostics']['failure']


def test_each_counterfactual_fits_only_H_plus_one_future(sample,monkeypatch):
    H,bank,fits,meter,opts,explanations,scenarios=sample
    fm,jm=_models_for_future(H,meter,2,meter)
    calls=[];original=en.io.fit_all_candidates
    def counted(candidates,observations,meter,**kwargs):
        calls.append(observations.t)
        return original(candidates,observations,meter,**kwargs)
    monkeypatch.setattr(en.io,'fit_all_candidates',counted)
    u=np.zeros((2,3));u[:,0]=1.
    result=en.evaluate_ensemble_action(H,bank,fits,meter,fm,jm,u,explanations=explanations,scenarios=scenarios,
        ensemble_axes=(0,),fit_options=opts)
    assert result['success'],result['failures']
    assert calls==[H.t+2]*6
    assert len(result['pairs'])==6 and result['required_fit_states']==12
    assert result['fit_calls']==12 and result['returned_fit_states']==12
    assert sum(r['weight'] for r in result['pairs'])==pytest.approx(1.)
    expected=sum(r['weight']*.5*(r['joint_loss']-fits[r['b']].loss) for r in result['pairs'])
    assert result['mean_added_score']==pytest.approx(expected)
    for record in result['records']:
        source=record['a'];scene=next(s for s in scenarios['scenarios'] if record['ensemble_id'].endswith(s['id']))
        expected=scene['background']['p']+scene['sign']*scene['offsets'][source]['p']+scene['control_gain']*u/20.
        assert record['forecast_state']['p']==pytest.approx(expected)
        assert np.asarray(record['predicted_observations']['p'])==pytest.approx(expected+meter.bias_p)
    assert not result['alternative_futures_stacked']


def test_failed_joint_model_retains_pairs_and_has_no_aggregate_score(sample,monkeypatch):
    H,bank,fits,meter,opts,explanations,scenarios=sample;fm,jm=_models_for_future(H,meter,2,meter)
    def fail(candidates,observations,meter,**kwargs):
        return {},{c.name:'deliberate_failure' for c in candidates},dict(fit_calls=len(candidates),ac_calls=7)
    monkeypatch.setattr(en.io,'fit_all_candidates',fail)
    result=en.evaluate_ensemble_action(H,bank,fits,meter,fm,jm,np.zeros((2,3)),explanations=explanations,
        scenarios=scenarios,ensemble_axes=(0,),fit_options=opts)
    assert not result['success'] and result['mean_added_score'] is None
    assert len(result['pairs'])==6 and all(not r['success'] and r['added'] is None for r in result['pairs'])
    assert result['fit_calls']==12 and result['fit_ac_calls']==42 and result['forecast_calls']==6


def test_mismatched_joint_H_weights_are_rejected(sample):
    H,bank,fits,meter,opts,explanations,scenarios=sample;fm,jm=_models_for_future(H,meter,2,meter)
    with pytest.raises(ValueError,match='same H and one future'):
        en.evaluate_ensemble_action(H,bank,fits,meter,fm,replace(jm,p_sigma=np.asarray(jm.p_sigma)*2),
            np.zeros((2,3)),explanations=explanations,scenarios=scenarios,fit_options=opts)


def test_multi_explanation_gain_physics_survives_same_witness_recenter(sample):
    from .ev_original_ac_check import make_ac_context,evaluate_ac_witness,recenter_at_absolute
    H,bank,fits,meter,opts,explanations,scenarios=sample
    alt=point(bank[0],H,meter,r=.0041)
    explanations['explanations']['star'].append(dict(id='star__flat',fit=alt,kind='test_flat'))
    ref=np.zeros((2,3));ref[:,0]=1.
    built=en.build_ensemble_contexts(H,bank,fits,meter,meter,ref,-6.,6.,
        explanations=explanations,scenarios=scenarios)
    assert built['success']
    model=next(m for m in built['models'] if m['a']=='star' and m['ensemble_id']=='axis_1__minus')
    context=make_ac_context(model,H,bank,fits,meter,meter)
    assert np.array_equal(context['reference_r'],alt.r)
    assert not np.array_equal(context['reference_r'],fits['star'].r)
    u=np.array([[.5,0.,0.],[0.,.75,0.]])
    U=context['future_control_map']
    extra=(U@u.ravel()).reshape(2,8)
    assert extra[:,:3]==pytest.approx(.95*u/20.)
    assert np.max(abs(extra[:,3:]))==0.
    xi=np.asarray(model['domain']['base_center'])+np.asarray(model['domain']['shift'])@u.ravel()
    updated=recenter_at_absolute(model,u.ravel(),xi,context)
    assert updated['task_weight']==model['task_weight']
    assert updated['ensemble_id']==model['ensemble_id']
    newcontext=make_ac_context(updated,H,bank,fits,meter,meter)
    assert np.array_equal(newcontext['reference_r'],alt.r)
    assert np.array_equal(newcontext['future_control_map'],U)
    audit=evaluate_ac_witness(updated,u.ravel(),xi,newcontext)
    assert audit['error_norm']<1e-9
    expected_background=next(c for c in built['contexts'] if c['id']==model['ensemble_id'])['forecast_background']['star']
    assert audit['reference_future_state']['p']==pytest.approx(expected_background['p']+.95*u/20.)
    assert audit['reference_future_state']['source_p']==pytest.approx(expected_background['source_p'])
