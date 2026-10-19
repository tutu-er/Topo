"""Independent arithmetic fixtures and tampering; no optimizer or new fit."""
from copy import deepcopy
import numpy as np
import pytest
from . import ev_compact_review as review
from . import ev_original_review as old
from .likelihood import concat_observations,RadialCandidate
from .test_ev_original_review import fit_fixture


def prediction_fixture():
    _,meter,physical,history,row=fit_fixture()
    candidates={n:RadialCandidate.from_supports(n,3,()) for n in ('A','B')}
    hfits={}
    for name in candidates:
        hfits[name]=dict(deepcopy(row),name=name)
    joint=[]
    for name in candidates:
        fit=dict(deepcopy(row),name=name,loglik=2*row['loglik'])
        for k in ('p','q','source_p','source_q'):fit[k]=np.concatenate([row[k],row[k]])
        fit['predicted']={k:np.concatenate([v,v]) for k,v in row['predicted'].items()}
        joint.append(fit)
    obs={k:getattr(history,k) for k in ('p','q','v','root_v','master_p','master_q','source_p','source_q')}
    background=dict(p=physical['p'],q=physical['q'],source_p=physical['source_p'],
                    source_q=physical['source_q'],root_v=physical['root_v'])
    context={None:dict(weight=1.,fits=hfits,control_gain=np.ones((2,3)),
                      forecast_background={n:deepcopy(background) for n in candidates})}
    records=[dict(a=n,forecast_state=deepcopy(background),success=True,
        predicted_observations=obs,fits=deepcopy(joint),failures={}) for n in candidates]
    pairs=[dict(a=a,b=b,success=True,added=0.,joint=0.,self_relative=0.)
           for a in candidates for b in candidates if a!=b]
    saved=dict(success=True,power_kw=np.zeros((2,3)),records=records,pairs=pairs,
        uses_truth=False,uses_future_noise=False,score=0.,mean_added_score=0.,worst_added_score=0.)
    config=dict(horizon=2,base_kw=20.,zbase_ohm=8.)
    return saved,history,meter,candidates,hfits,config,context,physical


def audit(saved=None):
    fixture=prediction_fixture();s,h,m,cs,hf,cfg,ctx,_=fixture
    return review.audit_prediction(saved or s,h,m,m,m,cs,hf,cfg,'numeric',ctx)


def test_full_prediction_saved_states_replay_without_ac_or_fit(monkeypatch):
    from . import likelihood
    def forbidden(*a,**kw):raise AssertionError('AC optimizer/model called')
    monkeypatch.setattr(likelihood,'fit_candidate',forbidden)
    monkeypatch.setattr(likelihood,'ac_predict',forbidden)
    out=audit()
    assert out['replayed_fit_states']==4 and out['directed_tasks']==2


@pytest.mark.parametrize('change',[
    lambda s:s['records'][0]['forecast_state']['p'].__setitem__((0,0),.3),
    lambda s:s['records'][0]['fits'][0].update(loglik=1.),
    lambda s:s['pairs'][0].update(added=1.),
    lambda s:s['pairs'].pop(),lambda s:s['records'].pop(),
    lambda s:s.update(alternative_futures_stacked=True),
    lambda s:s.update(uses_truth=True)])
def test_prediction_tampering_is_rejected(change):
    saved=prediction_fixture()[0];change(saved)
    with pytest.raises(ValueError):audit(saved)


def test_failed_forecast_remains_in_reference_and_pair_denominator():
    s,h,m,cs,hf,cfg,ctx,_=prediction_fixture()
    s.update(success=False);s['records'][0].update(success=False,predicted_observations=None,fits=[])
    for pair in s['pairs']:
        if pair['a']=='A':pair.update(success=False)
    out=review.audit_prediction(s,h,m,m,m,cs,hf,cfg,'numeric',ctx)
    assert not out['success'] and out['reference_cells']==2 and out['directed_tasks']==2


def test_cap_is_after_signed_history_adjusted_scores():
    saved=dict(success=True,pairs=[dict(added=-3.,weight=.25),dict(added=100.,weight=.75)])
    assert review.score(saved)==pytest.approx(74.25)
    assert review.score(saved,2.)==pytest.approx(.75)
    assert review.score(dict(success=False),2.) is None


@pytest.mark.parametrize('method',['capped_v2g_mobile','refined_v2g_mobile'])
def test_capped_policy_uses_v2g_mobile_physical_domain(method):
    from dataclasses import asdict
    from .ev_original_study import CONFIG
    one=review.service_config(CONFIG,method)
    two=review.service_config(CONFIG,'v2g_mobile')
    assert one.allow_discharge and one.mobile
    for name,value in asdict(one).items():
        np.testing.assert_equal(value,asdict(two)[name])


def same_witness_fixture():
    s,h,m,cs,hf,cfg,ctx,physical=prediction_fixture()
    n=3;H=2;T=2;state=np.column_stack((physical['p'],physical['q'],physical['source_p'],physical['source_q'])).ravel()
    xi=np.r_[hf['B']['r'],hf['B']['x'],state,state]
    channels=('p','q','v','master_p','master_q')
    length=2*len(old.whiten(physical,m,channels))
    qp=dict(physical_parameter=xi,residual=np.zeros(length),primal_upper=0.,tangent_lower_numeric=0.)
    raw=dict(u=np.zeros(6),required_pairs=['A__to__B','B__to__A'],best_evaluation_index=0,
        evaluations=[dict(qp_results=[deepcopy(qp),deepcopy(qp)])])
    ranges={k:[physical['voltage_min'],physical['voltage_max']] for k in
            ('competitor_history','competitor_future','reference_future')}
    checks=[dict(name=name,check=dict(complete_AC_residual=np.zeros(length),proxy_residual=np.zeros(length),
        proxy_minus_AC=np.zeros(length),complete_AC_cost=0.,proxy_cost=0.,max_channel_RMS=0.,
        voltage_ranges=deepcopy(ranges),complete_AC_voltage_feasible=True)) for name in raw['required_pairs']]
    return raw,checks,cs,h,m,hf,cfg,ctx


def test_same_witness_exact_ac_replay_reports_matrix_gap_honestly():
    raw,checks,cs,h,m,hf,cfg,ctx=same_witness_fixture()
    out=review.audit_qp_boxes_and_same_witness(raw,checks,cs,h,m,m,hf,cfg,'numeric',ctx)
    assert out==dict(box_norm_states=2,same_witness_AC_states=6,full_matrix_QP_states=0)


@pytest.mark.parametrize('change',[
    lambda raw,checks:raw['evaluations'][0]['qp_results'][0]['physical_parameter'].__setitem__(-1,.6),
    lambda raw,checks:checks[0]['check']['complete_AC_residual'].__setitem__(0,.1),
    lambda raw,checks:checks[0]['check'].update(max_channel_RMS=1.),
    lambda raw,checks:checks[0].update(name='B__to__A')])
def test_same_witness_or_box_tampering_detected(change):
    raw,checks,cs,h,m,hf,cfg,ctx=same_witness_fixture();change(raw,checks)
    with pytest.raises(ValueError):review.audit_qp_boxes_and_same_witness(raw,checks,cs,h,m,m,hf,cfg,'numeric',ctx)


def test_prediction_cache_key_handles_tiny_mip_roundoff_only():
    assert review.key([[1.,0.]])==review.key([[1.+1e-12,-0.]])
    assert review.key([[1.,0.]])!=review.key([[1.+1e-6,0.]])


def test_v2_command_limit_is_distinct_from_physical_rating():
    from .ev_original_study import CONFIG
    physical=deepcopy(CONFIG);ops=deepcopy(CONFIG);ops['device']['power_kw']=6/1.05
    trial=dict(config=dict(physical=physical,command_power_kw=6/1.05,
        ensemble={'tracking_fraction':.05}),operations_config=ops)
    assert review.operation_config(trial)['device']['power_kw']==pytest.approx(6/1.05)
    assert trial['config']['physical']['device']['power_kw']==6
    trial['operations_config']['device']['power_kw']=6
    with pytest.raises(ValueError):review.operation_config(trial)


def test_method_sets_remain_v1_ten_and_v2_eleven():
    assert len(review.method_names({'config':{}}))==10
    names=list(review.METHODS)+['refined_v2g_mobile']
    assert len(review.method_names({'config':{'methods':names}}))==11
    with pytest.raises(ValueError):review.method_names({'config':{'methods':names+['unknown']}})
