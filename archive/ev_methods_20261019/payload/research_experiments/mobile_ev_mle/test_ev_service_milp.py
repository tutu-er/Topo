"""Independent conservation, service, setup, bounds and search tests."""
from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest
from . import ev_service_milp as module
from .ev_service_milp import (EVServiceConstraints, SearchSettings,
    constant_site_plan, solve_site_step, check_service_plan, operational_data,
    optimize_service_curves, search_record, canonical_service_plan)


def constraints(**changes):
    # 60 kWh battery, 24 kWh pre-driving energy, +2.85 kWh service,
    # preserving the earlier charge-only 3 kWh grid-side baseline.
    return replace(EVServiceConstraints(terminal_energy=1.3425),**changes)


def test_default_constant_delivers_declared_battery_service():
    c=constraints(); r=constant_site_plan(0,4,1,c)
    assert r['success']
    np.testing.assert_allclose(r['plan'].p[:,0],[.15]*4,atol=1e-10)
    v=r['validation']
    assert v['terminal_battery_energy']==pytest.approx(1.3425)
    assert v['grid_charge_pu_hours']==pytest.approx(.15)
    assert v['service_battery_gain']==pytest.approx(.1425)
    assert v['efficiency_loss_pu_hours']==pytest.approx(.0075)
    assert abs(v['accounting_error']) < 1e-12


def test_linear_charging_step_is_actual_lp_with_dual_bound():
    r=solve_site_step(0,4,1,constraints(),gradients=[[1.,0,2.,0]])
    assert r['success'] and r['solver_kind']=='LP' and not r['binary_modes']
    np.testing.assert_allclose(r['plan'].p[:,0],[.3,0,.3,0],atol=1e-10)
    assert r['incumbent']==pytest.approx(.9)
    assert r['best_bound']==pytest.approx(.9)
    assert abs(r['absolute_gap'])<1e-9
    assert r['lp_dual_stationarity_inf']<1e-9
    assert r['lp_dual_sign_error']<1e-9


def test_max_min_lp_has_independent_interior_solution():
    c=constraints(terminal_energy=1.2+.95*.075)
    r=solve_site_step(0,2,1,c,gradients=[[2.,1.],[1.,2.]])
    np.testing.assert_allclose(r['plan'].p[:,0],[.15,.15],atol=1e-10)
    assert r['incumbent']==pytest.approx(.45)
    assert r['best_bound']==pytest.approx(.45)


def test_offsets_and_trust_radius_use_same_net_power_expression():
    c=constraints(terminal_energy=1.2+.95*.075)
    r=solve_site_step(0,2,1,c,base_power=[.12,.18],trust_radius=.02,
        intercepts=[-2.],gradients=[[1.,0.]])
    np.testing.assert_allclose(r['plan'].p[:,0],[.14,.16],atol=1e-10)
    assert r['incumbent']==pytest.approx(-1.98)
    assert r['best_bound']==pytest.approx(-1.98)
    assert r['trust_ok']


def test_v2g_fixed_terminal_maximum_negative_slot_is_hand_computable():
    c=constraints(p_discharge_max=.3)
    r=solve_site_step(0,4,1,c,gradients=[[0.,0.,0.,-1.]])
    assert r['success'] and r['solver_kind']=='MILP' and r['binary_modes']
    # Three max charging slots give .21375pu-h battery energy. To leave
    # .1425pu-h, discharge .07125pu-h battery: power -.07125*.95/.25.
    np.testing.assert_allclose(r['plan'].p[:,0],[.3,.3,.3,-.27075],atol=1e-9)
    v=r['validation']
    assert v['grid_discharge_pu_hours']==pytest.approx(.0676875)
    assert v['grid_net_pu_hours']==pytest.approx(.1573125)
    assert v['terminal_battery_energy']==pytest.approx(1.3425)
    assert v['efficiency_loss_pu_hours']==pytest.approx(.0148125)
    assert r['best_bound']>=r['incumbent']-1e-8


def test_v2g_zero_battery_service_replenishes_round_trip_loss():
    c=constraints(terminal_energy=1.2,p_discharge_max=.3)
    r=solve_site_step(0,2,1,c,gradients=[[1.,-1.]])
    assert r['success']
    np.testing.assert_allclose(r['plan'].p[:,0],[.3,-.27075],atol=1e-9)
    v=r['validation']
    assert v['grid_net_pu_hours']==pytest.approx(.0073125)
    assert v['battery_net_charging_gain']==pytest.approx(0,abs=1e-10)
    assert v['terminal_battery_energy']==pytest.approx(1.2)


def test_negative_battery_task_uses_discharge_efficiency():
    c=constraints(terminal_energy=1.2-.05,p_discharge_max=.3)
    r=constant_site_plan(0,2,1,c)
    assert r['success']
    np.testing.assert_allclose(r['plan'].p[:,0],[-.095,-.095])
    assert r['validation']['grid_discharge_pu_hours']==pytest.approx(.0475)
    assert r['validation']['efficiency_loss_pu_hours']==pytest.approx(.0025)


def test_same_terminal_is_preserved_when_moving_requires_more_charge():
    c=constraints(); static=constant_site_plan(0,4,2,c); mobile=constant_site_plan(1,4,2,c)
    assert static['success'] and mobile['success']
    assert static['validation']['terminal_battery_energy']==pytest.approx(mobile['validation']['terminal_battery_energy'])
    assert mobile['validation']['grid_charge_pu_hours']==pytest.approx((.1425+.01)/.95)
    assert mobile['validation']['post_drive_energy']==pytest.approx(1.19)
    np.testing.assert_allclose(mobile['plan'].p[:,1],[0,*([(.1425+.01)/(.95*.75)]*3)])


def test_staff_coverage_recovers_access_without_changing_device_or_task():
    access=np.ones((4,3),bool); access[:2,2]=False
    c=constraints(availability=access)
    failed=solve_site_step(2,4,3,c,gradients=[[0.,0,1,1]])
    assert not failed['success'] and failed['status']=='infeasible'
    access[1,2]=True
    recovered=solve_site_step(2,4,3,replace(c,availability=access),gradients=[[0,1,2,3]])
    assert recovered['success']
    assert recovered['validation']['terminal_battery_energy']==pytest.approx(1.3425)
    assert recovered['validation']['travel_energy_pu_hours']==pytest.approx(.01)


def test_setup_requires_contiguous_worker_window_after_vehicle_arrival():
    worker=np.array([[True],[False],[True],[True]])
    c=constraints(terminal_energy=1.2+.01,setup_slots=2,worker_setup_available=worker)
    d=operational_data(0,4,1,c)
    assert d['setup_start']==2 and d['ready_slot']==4
    r=solve_site_step(0,4,1,c)
    assert not r['success'] and r['status']=='operational_infeasible'


def test_worker_can_leave_after_setup_and_charging_continues():
    worker=np.array([[True],[False],[False],[False]])
    c=constraints(setup_slots=1,worker_setup_available=worker)
    r=solve_site_step(0,4,1,c,gradients=[[100,1,2,3]])
    assert r['success'] and r['setup_start']==0 and r['ready_slot']==1
    assert r['plan'].p[0,0]==pytest.approx(0)
    assert np.sum(r['plan'].p[1:,0])==pytest.approx(.6)
    assert r['plan'].p[3,0]>0


def test_one_setup_slot_after_driving_makes_short_window_service_infeasible():
    r=solve_site_step(1,4,2,constraints(setup_slots=1))
    assert not r['success'] and r['status']=='infeasible'


def test_setup_cannot_be_done_before_vehicle_arrival_or_at_unavailable_site():
    worker=np.zeros((4,2),bool); worker[0,1]=True
    c=constraints(worker_setup_available=worker,setup_slots=1)
    r=solve_site_step(1,4,2,c)
    assert not r['success'] and r['status']=='operational_infeasible'
    worker[1,1]=True; access=np.ones((4,2),bool); access[1,1]=False
    r=solve_site_step(1,4,2,replace(c,worker_setup_available=worker,availability=access))
    assert not r['success'] and r['status']=='operational_infeasible'


def test_v2g_has_no_export_when_one_slot_lost_and_task_is_large():
    # Even for ideal 3 available slots, the 2.85kWh gain consumes two
    # full 6kW slots. Actual driving energy makes remaining slot positive.
    c=constraints(p_discharge_max=.3)
    r=solve_site_step(1,4,2,c,gradients=[[0,0,0,-1]])
    assert r['success']
    assert r['validation']['grid_discharge_pu_hours']==pytest.approx(0)
    assert np.all(r['plan'].p[:,1]>=-1e-9)
    assert r['plan'].p[3,1]==pytest.approx(.01/(.95*.25))


def test_departure_removes_late_prices_and_fixed_terminal_keeps_service():
    c=constraints(terminal_energy=1.2+.05,departure_slot=2)
    r=solve_site_step(0,4,1,c,gradients=[[0,1,100,100]])
    assert r['success']
    np.testing.assert_allclose(r['plan'].p[2:,0],0)
    np.testing.assert_array_equal(r['plan'].sites,[0,0,-1,-1])
    assert r['validation']['terminal_battery_energy']==pytest.approx(1.25)


def test_return_trip_reserve_checked_separately_from_terminal_charging():
    good=constraints(return_energy=.02,departure_reserve=1.30)
    r=constant_site_plan(0,4,1,good)
    assert r['success']
    assert r['validation']['post_return_battery_energy']==pytest.approx(1.3225)
    bad=solve_site_step(0,4,1,replace(good,departure_reserve=1.33))
    assert not bad['success'] and 'reserve' in bad['termination']


def test_pre_drive_soc_cannot_be_repaired_by_subsequent_charging():
    c=constraints(initial_soc=.101,terminal_energy=.4,soc_min=.1)
    r=solve_site_step(1,4,2,c)
    assert not r['success'] and 'driving' in r['termination']


def test_variable_curve_feasible_when_uniform_cap_is_too_low():
    c=constraints(terminal_energy=1.2+.95*.075,p_charge_max=np.array([[.1],[.3]]))
    assert not constant_site_plan(0,2,1,c)['success']
    r=solve_site_step(0,2,1,c,gradients=[[1,0]])
    assert r['success']
    np.testing.assert_allclose(r['plan'].p[:,0],[.1,.2])


def test_validator_recomputes_energy_path_and_terminal_independently():
    c=constraints(); r=constant_site_plan(0,4,2,c); plan=r['plan']
    plan.p[0,1]=.01
    assert 'multiple_sites' in check_service_plan(plan,c)['issues']
    plan.p[0,1]=0; plan.p[0,0]+=.01
    issues=check_service_plan(plan,c)['issues']
    assert 'soc_dynamics' in issues and 'terminal_battery_service' in issues


def test_validator_catches_simultaneous_solver_flows():
    c=constraints(p_discharge_max=.3); r=solve_site_step(0,4,1,c,gradients=[[1,0,1,0]])
    r['plan'].solver['split_charge_pu'][0]+=.01
    r['plan'].solver['split_discharge_pu'][0]+=.01
    checked=check_service_plan(r['plan'],c)
    assert not checked['feasible'] and 'simultaneous_or_wrong_split_flow' in checked['issues']


def test_failure_and_feasible_time_limit_bound_are_retained(monkeypatch):
    original=module._frozen_solver
    def limited(*args,**kwargs):
        result=original(*args,**kwargs); result.status=1; result.success=False
        result.message='declared time limit'; result.mip_dual_bound=-1.5
        return result
    monkeypatch.setattr(module,'_frozen_solver',limited)
    r=solve_site_step(0,4,1,constraints(),gradients=[[1,0,1,0]])
    assert r['success'] and r['status']=='feasible_incumbent'
    assert r['best_bound']==pytest.approx(1.5)
    assert r['absolute_gap']==pytest.approx(.9)
    assert r['termination']=='declared time limit'


def test_bad_dual_evidence_is_not_treated_as_optimal_lp(monkeypatch):
    original=module._frozen_solver
    def invalid(*args,**kwargs):
        result=original(*args,**kwargs);result.mip_dual_bound=None
        result.lp_dual_stationarity_inf=.1
        return result
    monkeypatch.setattr(module,'_frozen_solver',invalid)
    r=solve_site_step(0,4,1,constraints(),gradients=[[1,0,1,0]])
    assert not r['success'] and r['status']=='invalid_incumbent'
    assert r['best_bound'] is None


def test_solver_exception_is_saved(monkeypatch):
    def fail(*args,**kwargs):raise RuntimeError('declared LP failure')
    monkeypatch.setattr(module,'_frozen_solver',fail)
    r=solve_site_step(0,4,1,constraints())
    assert not r['success'] and r['status']=='solver_failure'
    assert 'declared LP failure' in r['termination']


@pytest.mark.parametrize('changes,match',[
    ({'terminal_energy':3.},'SOC'),({'terminal_energy':float('nan')},'finite'),
    ({'setup_slots':.5},'integers'),({'worker_setup_available':[[2],[0],[0],[0]]},'bool'),
    ({'departure_slot':5},'outside'),({'return_energy':-.1},'nonnegative'),
    ({'departure_reserve':-.1},'nonnegative'),({'p_charge_max':-.1},'nonnegative'),
])
def test_invalid_public_inputs_raise(changes,match):
    with pytest.raises(ValueError,match=match):solve_site_step(0,4,1,constraints(**changes))


class AnalyticEvaluator:
    horizon=4; base_kva=20.; q_over_p=0.;history=SimpleNamespace(n=2)
    def __init__(self,failed=None):self.calls=0;self.failed=failed
    def evaluate(self,site,power_kw):
        self.calls+=1
        if site==self.failed:return dict(success=False,score=None,pairs=[],failures=['retained AC failure'])
        u=np.asarray(power_kw)/20.;score=float(u@u)+site*.001
        pair=dict(score=score,gradient_kw=2*u/20.,pq_only_gradient_kw=2*u/20.)
        return dict(success=True,score=score,pairs=[pair],fit_calls=1,total_ac_calls=2)


def test_continuous_optimizer_generates_feasible_curves_and_rechecks_score():
    c=constraints(travel_slots=np.zeros((2,2),int),travel_energy=np.zeros((2,2)))
    evaluator=AnalyticEvaluator()
    r=optimize_service_curves(evaluator,c,settings=SearchSettings(max_iterations=2))
    assert r['success'] and r['selected_site']==1
    assert r['score']==pytest.approx(.181)
    assert r['gain_over_best_constant']==pytest.approx(.09)
    assert check_service_plan(r['plan'],c)['feasible']
    assert not r['global_design_optimality_certified']
    assert r['fit_calls']==evaluator.calls and r['total_ac_calls']==2*evaluator.calls
    assert any(s['kind']=='optimized_alternating_seed' for record in r['site_records'] for s in record['starts'])
    json.dumps(search_record(r))


def test_continuous_optimizer_retains_failed_necessary_evaluation():
    c=constraints(travel_slots=np.zeros((2,2),int),travel_energy=np.zeros((2,2)))
    r=optimize_service_curves(AnalyticEvaluator(failed=1),c)
    assert r['success'] and r['selected_site']==0
    assert r['failures'] and len(r['site_records'])==2
    assert 1 not in r['site_plans']


def test_very_small_discrete_audit_is_only_an_independent_comparator():
    c=constraints(terminal_energy=1.2+.95*.075)
    r=solve_site_step(0,2,1,c,gradients=[[2,1],[1,2]])
    discrete=max(min(np.array([[2,1],[1,2]])@np.array([x,.3-x])) for x in [0,.1,.2,.3])
    assert discrete==pytest.approx(.4) and r['incumbent']==pytest.approx(.45)


def test_expanded_access_guard_rebuilds_metadata_without_changing_injection():
    access=np.ones((4,3),bool);access[:2,2]=False
    old=constraints(terminal_energy=1.25,availability=access)
    plan=solve_site_step(2,4,3,old,gradients=[[0,0,1,0]])['plan']
    assert check_service_plan(plan,old)['feasible']
    expanded=replace(old,availability=np.ones((4,3),bool))
    assert 'connection_path' in check_service_plan(plan,expanded)['issues']
    kept=canonical_service_plan(2,plan.p[:,2],3,expanded,solver=plan.solver)
    assert check_service_plan(kept,expanded)['feasible']
    np.testing.assert_array_equal(kept.p,plan.p)
    np.testing.assert_array_equal(kept.q,plan.q)
    np.testing.assert_allclose(kept.soc,plan.soc,rtol=0.,atol=1e-12)
    # Rebuilding metadata is not permission to ignore changed task physics.
    mobility=np.full((3,3),.02);np.fill_diagonal(mobility,0)
    changed=replace(expanded,travel_energy=mobility)
    bad=canonical_service_plan(2,plan.p[:,2],3,changed)
    assert 'terminal_battery_service' in check_service_plan(bad,changed)['issues']
