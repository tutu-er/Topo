"""Hand-checkable resource/LP fixtures, with no feeder, AC or identifier calls."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from research_experiments.wzzt_dr_selection.dr_day_modules.day_schedule import (
    Battery, ResourceDay, DayModule, LinearTask, DT, SLOTS, shift_shape,
    shift_module, ev_resource, ev_reallocation_module, spatial_module,
    complementary_modules, evaluate_schedule, optimize_amplitudes,
    decompose_terminal_baseline, load_archived_ev_domain,
)


def resource(name="load", node=201, *, price=.1, tariff=0., available=True):
    return ResourceDay(name,node,np.full(SLOTS,.2),np.full(SLOTS,.4),
                       np.full(SLOTS,-1.),np.full(SLOTS,1.),available=available,
                       incentive_per_kwh=price,electricity_price_per_kwh=tariff,
                       declaration_source="synthetic hand-checkable unit fixture")


def test_unequal_shift_windows_conserve_per_resource_energy():
    profile=shift_shape(range(8,12),range(20,28))
    assert np.all(profile[8:12]==-1)
    assert np.all(profile[20:28]==.5)
    r=resource()
    m=shift_module("shift","load",range(8,12),range(20,28),.2)
    result=evaluate_schedule([r],[m],[.2])
    assert result["resource_feasible"]
    row=result["resources"]["load"]
    assert row["reduced_kwh"]==pytest.approx(.2)
    assert row["rebound_kwh"]==pytest.approx(.2)
    assert row["net_kwh"]==pytest.approx(0.,abs=1e-12)
    assert result["total_incentive_payment"]==pytest.approx(.02)


def test_resource_baseline_is_unique_and_der_residual_may_be_negative():
    r=resource()
    with pytest.raises(ValueError,match="unique baseline"):
        evaluate_schedule([r,r],[],[])
    total=np.full((SLOTS,1),-.1)
    pieces=decompose_terminal_baseline(total,[201],[r])
    np.testing.assert_allclose(pieces["nonflexible_kw"],-.3)
    assert pieces["negative_nonflexible_allowed"]
    assert not pieces["resource_capability_inferred_from_total"]


def test_resource_rho_scales_the_aggregate_not_individual_modules():
    r=resource()
    h=shift_shape(range(8,12),range(16,20))
    modules=[DayModule("first",{"load":h},.1),DayModule("second",{"load":h},.1)]
    full=evaluate_schedule([r],modules,[.1,.1])
    actual=evaluate_schedule([r],modules,[.1,.1],rho={"load":.3})
    np.testing.assert_allclose(actual["resources"]["load"]["executed_delta_kw"],
                               .3*np.asarray(full["resources"]["load"]["planned_delta_kw"]))
    assert actual["resource_feasible"]
    assert actual["total_incentive_payment"]==pytest.approx(.3*full["total_incentive_payment"])


def test_cancelling_modules_are_not_charged_twice():
    r=resource()
    h=shift_shape(range(8,12),range(16,20))
    modules=[DayModule("down_up",{"load":h},.2),DayModule("up_down",{"load":-h},.2)]
    result=evaluate_schedule([r],modules,[.2,.2])
    assert result["resource_feasible"]
    assert result["total_incentive_payment"]==pytest.approx(0.)
    assert result["resources"]["load"]["reduced_kwh"]==pytest.approx(0.)
    np.testing.assert_allclose(result["node_increments"]["201"]["delta_kw"],0.)


def test_full_plan_is_checked_even_if_realized_rho_is_zero():
    r=resource()
    h=shift_shape(range(8,12),range(16,20))
    modules=[DayModule("first",{"load":h},.2),DayModule("second",{"load":h},.2)]
    result=evaluate_schedule([r],modules,[.2,.2],rho={"load":0.})
    assert result["resources"]["load"]["actual_resource_feasible"]
    assert not result["resources"]["load"]["full_plan_feasible"]
    assert not result["resource_feasible"]


def test_availability_is_not_extended_by_a_module():
    available=np.zeros(SLOTS,bool);available[8:20]=True
    r=resource(available=available)
    m=shift_module("outside","load",range(0,4),range(4,8),.2)
    assert not evaluate_schedule([r],[m],[.1])["resource_feasible"]
    solution=optimize_amplitudes([r],[m],amplitude_cost_per_kw=[-1.],time_limit_seconds=2.)
    assert solution["status"]=="optimal"
    assert solution["module_amplitudes_kw"]["outside"]==pytest.approx(0.)
    assert solution["AC_calls"]==solution["WZZT_calls"]==0


def test_ev_reallocation_keeps_soc_deadline_and_all_delivery_fractions():
    available=np.zeros(SLOTS,bool);available[8:16]=True
    baseline=np.zeros(SLOTS);baseline[8:12]=.2
    battery=Battery(capacity_kwh=1.,initial_kwh=.5,terminal_kwh=.68,
                    soc_min=.1,soc_max=.9,eta_charge=.9)
    r=ev_resource("ev",202,baseline,.4,available,battery,incentive_per_kwh=.1)
    m=ev_reallocation_module("later",r,range(8,12),range(12,16),.2)
    for rho in (0.,.3,1.):
        result=evaluate_schedule([r],[m],[.2],rho={"ev":rho})
        assert result["resource_feasible"]
        row=result["resources"]["ev"]
        assert row["terminal_energy_kwh"]==pytest.approx(.68)
        assert min(row["soc"])>=.1-1e-10
        assert max(row["soc"])<=.9+1e-10
        assert result["total_incentive_payment"]<=result["full_delivery_incentive_payment"]+1e-12
    with pytest.raises(ValueError,match="outside"):
        ev_reallocation_module("bad",r,range(0,4),range(12,16),.2)


def test_composed_plan_power_is_jointly_constrained_by_lp():
    r=resource(price=0.)
    h=shift_shape(range(8,12),range(16,20))
    modules=[DayModule("first",{"load":h},.2),DayModule("second",{"load":h},.2)]
    result=optimize_amplitudes([r],modules,amplitude_cost_per_kw=[-1.,-1.],
                               time_limit_seconds=2.)
    assert result["status"]=="optimal"
    assert sum(result["module_amplitudes_kw"].values())==pytest.approx(.2)
    assert result["execution"]["resource_feasible"]


def test_cost_and_budget_use_net_delivered_reduction():
    tariff=np.zeros(SLOTS);tariff[8:12]=1.;tariff[16:20]=.1
    r=resource(price=.1,tariff=tariff)
    m=shift_module("economic_shift","load",range(8,12),range(16,20),.2)
    result=optimize_amplitudes([r],[m],incentive_budget=.01,time_limit_seconds=2.)
    assert result["status"]=="optimal"
    assert result["module_amplitudes_kw"]["economic_shift"]==pytest.approx(.1)
    assert result["execution"]["total_incentive_payment"]==pytest.approx(.01)
    assert result["execution"]["total_incremental_electricity_bill"]==pytest.approx(-.09)
    assert result["direct_operating_objective"]==pytest.approx(-.08)
    assert result["objective_bound"]==pytest.approx(-.08)
    assert result["independent_primal_checks_pass"]


def test_linear_service_task_and_infeasible_budget_are_retained():
    r=resource(price=.1)
    m=shift_module("shift","load",range(8,12),range(16,20),.2)
    coefficient=np.zeros(SLOTS);coefficient[8:12]=-DT
    task=LinearTask("requested_peak_shift",{"load":coefficient},lower=.1,units="kWh")
    good=optimize_amplitudes([r],[m],tasks=[task],incentive_budget=.01,time_limit_seconds=2.)
    assert good["status"]=="optimal"
    assert good["module_amplitudes_kw"]["shift"]==pytest.approx(.1)
    bad=optimize_amplitudes([r],[m],tasks=[task],incentive_budget=0.,time_limit_seconds=2.)
    assert bad["solver_status"]==2
    assert not bad["solver_reported_optimal"]
    assert bad["status"]=="no_feasible_incumbent"
    assert bad["module_amplitudes_kw"] is None


def test_zero_activity_and_fixed_subset_need_no_binary():
    r=resource(price=.1)
    m=shift_module("shift","load",range(8,12),range(16,20),.2)
    result=optimize_amplitudes([r],[m],selected_module_ids=[],time_limit_seconds=2.)
    assert result["status"]=="optimal"
    assert result["module_amplitudes_kw"]=={"shift":pytest.approx(0.)}
    assert result["execution"]["total_incentive_payment"]==pytest.approx(0.)
    assert "not WZZT" in result["objective_scope"]


def test_same_opposed_and_complementary_patterns_need_no_topology():
    h=shift_shape(range(8,12),range(16,20))
    r1,r2,r3=resource("a",201),resource("b",202),resource("c",203)
    same=spatial_module("same",h,{"a":1,"b":1},.1)
    opposed=spatial_module("opposed",h,{"a":1,"b":-1},.1)
    together=evaluate_schedule([r1,r2],[same],[.1])
    opposite=evaluate_schedule([r1,r2],[opposed],[.1])
    assert abs(sum(together["node_increments"][str(n)]["delta_kw"][8] for n in (201,202)))>.1
    assert sum(opposite["node_increments"][str(n)]["delta_kw"][8] for n in (201,202))==pytest.approx(0.)
    later=shift_shape(range(40,44),range(48,52))
    modules=complementary_modules("two",["a","b","c"],waveforms=(h,later),amplitude_upper_kw=.1)
    result=evaluate_schedule([r1,r2,r3],modules,[.1,.1])
    assert result["resource_feasible"]
    assert len(modules)==2


def test_dr_ev_pair_only_balances_active_power():
    h=shift_shape(range(8,12),range(16,20))
    dr=resource("dr",201)
    battery=Battery(10.,1.,5.8)
    ev=ev_resource("ev",202,np.full(SLOTS,.2),.4,np.ones(SLOTS,bool),battery)
    module=spatial_module("mixed",h,{"dr":1,"ev":-1},.1)
    result=evaluate_schedule([dr,ev],[module],[.1])
    assert result["resource_feasible"]
    at=lambda name:sum(v[name][8] for v in result["node_increments"].values())
    assert at("delta_kw")==pytest.approx(0.)
    assert at("delta_kvar")==pytest.approx(-.035)


def test_external_ev_adapter_requires_explicit_hashes_and_does_not_write_archive(tmp_path):
    (tmp_path/"ev_operations_milp.py").write_text("class LinearMILP: pass\n",encoding="utf-8")
    (tmp_path/"ev_mobile_service_domain.py").write_text(
        "from .ev_operations_milp import LinearMILP\n"
        "class MobileServiceConfig:\n"
        " def __init__(self,**kwargs): self.__dict__.update(kwargs)\n"
        "def build_mobile_service_domain(config): return config\n",encoding="utf-8")
    expected={name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
              for name in ("ev_operations_milp.py","ev_mobile_service_domain.py")}
    with pytest.raises(ValueError,match="hashes"):
        load_archived_ev_domain(tmp_path,{})
    domain,metadata=load_archived_ev_domain(tmp_path,expected)
    assert domain.n_slots==96 and domain.n_sites==1
    assert not domain.mobile and not domain.allow_discharge
    assert metadata["solver_calls"]==0 and not metadata["packaged_dependency"]
    assert not (tmp_path/"__pycache__").exists()
    with pytest.raises(ValueError,match="charge-only"):
        load_archived_ev_domain(tmp_path,expected,allow_discharge=True)


def test_invalid_profiles_windows_and_rho_are_rejected():
    with pytest.raises(ValueError,match="zero daily"):
        DayModule("nonconserving",{"load":np.ones(SLOTS)},.2)
    with pytest.raises(ValueError,match="disjoint"):
        shift_shape([1,2],[2,3])
    with pytest.raises(ValueError,match="spatial signs"):
        spatial_module("bad",shift_shape([1],[2]),{"load":True},.2)
    with pytest.raises(ValueError,match="EV resources"):
        ResourceDay("ev",201,np.full(SLOTS,.2),.4,-1.,1.,kind="ev")
    r=resource();m=shift_module("shift","load",range(8,12),range(16,20),.2)
    with pytest.raises(ValueError,match="rho"):
        evaluate_schedule([r],[m],[.1],rho={"load":1.1})

def test_complementary_temporal_counterexample_and_partial_delivery():
    """Input rank illustrates the prior labeling error, never a recovery metric."""
    resources=[resource("a",201),resource("b",202),resource("c",203)]
    first=shift_shape(range(8,12),range(16,20))
    second=shift_shape(range(40,44),range(48,52))
    matrix=lambda result:np.column_stack([
        result["node_increments"][str(node)]["delta_kw"] for node in (201,202,203)])
    # Previous same-waveform composition: h(t)*(a1*v1+a2*v2), only rank one.
    old=[spatial_module("old_first",first,{"a":-1,"b":1},.1),
         spatial_module("old_second",first,{"b":1,"c":-1},.1)]
    prior=evaluate_schedule(resources,old,[.1,.1])
    assert prior["resource_feasible"] and np.linalg.matrix_rank(matrix(prior))==1
    for duplicate in (first,2.*first):
        with pytest.raises(ValueError,match="collinear"):
            complementary_modules("bad",["a","b","c"],(first,duplicate),.1)
    with pytest.raises(ValueError,match="two explicit"):
        complementary_modules("old_api",["a","b","c"],first,.1)
    modules=complementary_modules("new",["a","b","c"],(first,second),.1)
    full=evaluate_schedule(resources,modules,[.1,.1])
    assert full["resource_feasible"] and np.linalg.matrix_rank(matrix(full))==2
    partial=evaluate_schedule(resources,modules,[.1,.1],rho={"a":0.,"b":1.,"c":0.})
    assert partial["resource_feasible"] and np.linalg.matrix_rank(matrix(partial))==1
    one=evaluate_schedule(resources,modules,[.1,0.])
    zero=evaluate_schedule(resources,modules,[0.,0.])
    assert one["resource_feasible"] and np.linalg.matrix_rank(matrix(one))==1
    assert zero["resource_feasible"] and np.linalg.matrix_rank(matrix(zero))==0


def test_declared_service_deadline_prevents_post_deadline_rebound():
    # The EV is connected until24:00, but its planned service is due at20:00.
    available=np.ones(SLOTS,bool)
    baseline=np.zeros(SLOTS);baseline[76:80]=.2
    battery=Battery(1.,.5,.68,soc_min=.1,soc_max=.9,eta_charge=.9)
    ev=ev_resource("ev",202,baseline,.4,available,battery,service_deadline_slot=80)
    late=ev_reallocation_module("late",ev,range(76,80),range(84,88),.2)
    bad=evaluate_schedule([ev],[late],[.2])
    row=bad["resources"]["ev"]
    assert row["terminal_energy_kwh"]==pytest.approx(.68)  # Daily total alone passes.
    assert row["planned_deadline_energy_kwh"]==pytest.approx(.5)
    assert row["actual_deadline_energy_kwh"]==pytest.approx(.5)
    assert not row["planned_deadline_pass"] and not row["actual_deadline_pass"]
    assert not bad["resource_feasible"]
    assert bad["increment_only_payload"]
    assert "original_bases_not_added_again" not in bad
    assert "not evaluated" in bad["double_counting_bridge_check"]
    # A zero actual fraction cannot hide an invalid fully promised plan.
    absent=evaluate_schedule([ev],[late],[.2],rho={"ev":0.})
    assert absent["resources"]["ev"]["actual_deadline_pass"]
    assert not absent["resources"]["ev"]["planned_deadline_pass"]
    # The same deadline boundary enters the operating LP without a special rule.
    solved=optimize_amplitudes([ev],[late],amplitude_cost_per_kw=[-1.],time_limit_seconds=2.)
    assert solved["status"]=="optimal" and solved["independent_primal_checks_pass"]
    assert solved["module_amplitudes_kw"]["late"]==pytest.approx(0.)
    assert solved["execution"]["resources"]["ev"]["actual_deadline_pass"]
    assert solved["execution"]["resources"]["ev"]["planned_deadline_pass"]
    late_baseline=np.zeros(SLOTS);late_baseline[84:88]=.2
    with pytest.raises(ValueError,match="planned deadline"):
        ev_resource("bad_base",202,late_baseline,.4,available,battery,service_deadline_slot=80)


def test_multiple_windows_need_a_planned_deadline_and_boundary_validation():
    available=np.zeros(SLOTS,bool);available[8:12]=True;available[76:80]=True
    baseline=np.zeros(SLOTS);baseline[76:80]=.2
    battery=Battery(1.,.5,.68,soc_min=.1,soc_max=.9,eta_charge=.9)
    with pytest.raises(ValueError,match="explicit deadline"):
        ev_resource("ev",202,baseline,.4,available,battery)
    ev=ev_resource("ev",202,baseline,.4,available,battery,service_deadline_slot=80)
    assert ev.service_deadline_slot==80
    lo,hi=ev.energy_bounds()
    assert lo[79]==pytest.approx(0.) and hi[79]==pytest.approx(0.)
    assert evaluate_schedule([ev],[],[])["resource_feasible"]
    for deadline in (0,97,80.,True):
        with pytest.raises(ValueError,match="integer boundary"):
            ev_resource("invalid",202,baseline,.4,available,battery,
                        service_deadline_slot=deadline)
