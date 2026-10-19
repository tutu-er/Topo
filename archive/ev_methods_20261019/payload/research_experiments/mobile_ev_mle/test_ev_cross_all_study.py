import inspect
from types import SimpleNamespace
import numpy as np
import pytest
from research_experiments.mobile_ev_mle import ev_cross_all_study as study
from research_experiments.mobile_ev_mle.ev_bilinear_score_exchange import charging_service_domain
from research_experiments.mobile_ev_mle.ev_flexibility_study import device
from research_experiments.mobile_ev_mle.ev_service_milp import canonical_service_plan,check_service_plan

def test_same_regime_high_correlation_is_for_both_p_and_q():
    high,_,_,_,a=study.make_development_path(987103,((0,1),),True)
    low,_,_,_,b=study.make_development_path(987104,((0,1),),False)
    off=np.triu_indices(3,1)
    assert np.min(a['p_correlation'][off])>.98
    assert np.min(a['q_correlation'][off])>.98
    assert a['population_correlation_p']>.99 and a['population_correlation_q']>.99
    assert b['population_correlation_p']==pytest.approx(.5)
    assert b['population_correlation_q']==pytest.approx(.2)
    assert np.mean(a['p_correlation'][off])>np.mean(b['p_correlation'][off])
    assert np.mean(a['q_correlation'][off])>np.mean(b['q_correlation'][off])
    assert a['historical_and_future_same_regime'] and not a['standard_or_measured_data']

def test_seed_replays_and_truth_future_inputs_are_not_selector_arguments():
    first=study.make_development_path(987107,((1,2),),True)
    second=study.make_development_path(987107,((1,2),),True)
    for key in ('p','q','r','x','source_p','source_q'):
        np.testing.assert_array_equal(getattr(first[0],key),getattr(second[0],key))
    for key in first[3]:np.testing.assert_array_equal(first[3][key],second[3][key])
    assert set(inspect.signature(study.select_curves).parameters).isdisjoint(
        {'truth','path','white','future_observations','true_candidate'})

def test_quality_and_random_price_lp_have_same_terminal_battery_service():
    service=device();d=charging_service_domain(0,4,3,service,power_scale=20.)
    quality=np.array([1.,9.,9.,1.]);random=np.array([.4,-.5,.1,.2])
    for prices in (quality,random):
        s=study.linear_curve(prices,d['lower'],d['upper'],d['constraints'])
        assert s['success']
        plan=canonical_service_plan(0,s['u']/20.,3,service)
        check=check_service_plan(plan,service)
        assert check['feasible']
        assert np.sum(s['u'])*.25==pytest.approx(3.)
        assert plan.soc[-1]*service.battery_capacity*20.==pytest.approx(26.85)
    s=study.linear_curve(quality,d['lower'],d['upper'],d['constraints'])
    assert quality@s['u']==pytest.approx(6*(9+9))

def test_difficult_pair_failure_is_propagated_instead_of_deleted(monkeypatch):
    calls=[]
    def broken(history,candidates,*args,**kwargs):
        calls.append(tuple(c.name for c in candidates))
        return {'success':False,'diagnostics':{'required_pair_failure':True}}
    monkeypatch.setattr(study,'build_pair_models',broken)
    candidates=[SimpleNamespace(name=n) for n in ['star','clade_01','clade_02','clade_12']]
    with pytest.raises(ValueError,match='required pair context'):
        study.build_all_context(None,candidates,{},None,None,site=0,lower=np.zeros(4),upper=np.ones(4)*6)
    assert len(calls)==1

def test_declared_families_include_missing_previous_case_and_independent_base_units():
    families=[case[0] for case in study.CASES]
    assert families.count(((1,2),))==2 and families.count(())==2
    assert len(set(study.CONFIG['seeds']))==8
    assert len(study.CONFIG['quality'])==2
    assert len(study.METHODS)==7
