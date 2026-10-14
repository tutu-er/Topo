"""Physical probing-task and controller-interface checks."""
from dataclasses import replace
import inspect
import numpy as np
import pytest
from .ev_mobile_service_domain import MobileServiceConfig,build_mobile_service_domain
from .ev_original_study import service_domain,realize_control,select_controls,validate_evaluation_gate,CONFIG,make_case,prepare,evaluate

def test_controller_signature_has_no_truth_or_future_noise():
    assert tuple(inspect.signature(select_controls).parameters)==('history','bank','hfits','hm','fm','jm','seed','domain')

def test_same_battery_constant_is_legal_in_all_four_resource_domains():
    u=np.zeros((6,3));u[:,0]=2.
    for mobile in (False,True):
        for discharge in (False,True):
            d=service_domain(mobile=mobile,discharge=discharge)
            r=realize_control(d,u)
            assert r['success']
            assert r['audit']['energy_kwh'][-1]==pytest.approx(CONFIG['device']['terminal_energy_kwh'])
            assert r['audit']['travel_energy_kwh']==pytest.approx(0.)

def test_trip_without_destination_probing_is_excluded_by_declared_task():
    cfg=MobileServiceConfig(n_slots=3,n_sites=2,dt_hours=1.,p_charge_max_kw=1.,allow_discharge=False,
        battery_capacity_kwh=10.,initial_soc=.5,soc_min=0.,soc_max=1.,eta_charge=1.,eta_discharge=1.,
        terminal_energy_kwh=7.,travel_energy_kwh=0.,setup_slots=1,terminal_site=1,max_moves=1)
    u=np.array([[1.,0.],[1.,0.],[0.,0.]])
    old=realize_control(build_mobile_service_domain(cfg),u)
    assert old['success']
    proper=realize_control(build_mobile_service_domain(replace(cfg,min_arrival_throughput_kwh=.1)),u)
    assert not proper['success'] and proper['status']==2

def test_arrival_task_still_allows_a_real_remote_probe():
    d=service_domain(mobile=True,discharge=True)
    prices=np.zeros((6,3));prices[:,1]=1.
    r=d.solve_prices(prices,time_limit=3.)
    a=d.check(r)
    assert a['feasible'] and a['moves']==1
    assert np.sum(np.abs(np.asarray(a['u_kw'])[:,1]))*.25>=.25-1e-7
    assert a['travel_energy_kwh']==pytest.approx(.2)
    assert a['energy_kwh'][-1]==pytest.approx(26.85)

def test_arrival_task_requires_nonnegative_finite_amount():
    for value in (-1.,np.inf,np.nan):
        with pytest.raises(ValueError):
            build_mobile_service_domain(MobileServiceConfig(min_arrival_throughput_kwh=value))

def test_independent_evaluation_requires_current_passed_gate_and_disjoint_histories():
    gate=dict(ready_for_independent_evaluation=True,source_sha256={'model':'abc'},pilot_seeds=[1,2],
        evaluation_scope='independent descriptive AC evaluation; no calibrated rejection claim')
    assert validate_evaluation_gate(gate,{'model':'abc'},[3,4])
    for update,hashes,seeds in [({'ready_for_independent_evaluation':False},{'model':'abc'},[3]),
            ({},{'model':'changed'},[3]),({},{'model':'abc'},[2]),({'evaluation_scope':'formal risk'},{'model':'abc'},[3])]:
        with pytest.raises(ValueError):validate_evaluation_gate({**gate,**update},hashes,seeds)

def test_missing_historical_fit_does_not_drop_future_candidate_or_crash_record():
    case=make_case(988995,());prepared=prepare(case,'asset')
    missing=prepared['candidates'][0].name
    del prepared['fits'][missing]
    rows=evaluate(case,prepared,{'none':np.zeros((6,3))},'asset')
    assert len(rows)==1
    audit=rows[0]['noiseless_AC_refit']
    assert not audit['all_historical_baselines']
    assert len(audit['candidate_distances'])==4
    row=next(r for r in audit['candidate_distances'] if r['candidate']==missing)
    assert row['added_half_SSE'] is None
    assert row['reason']=='required_historical_fit_failure'
