"""Physical unit invariance and nonzero mobility/setup operation tests."""
import numpy as np
import pytest
from .ev_physical_cases import case33_local_standard
from .ev_ev_ess_spatial_audit import (physical_response,electrical_audit,
    service_case,operation_audit,solve_affine_curve)
from .ev_bilinear_score_exchange import charging_service_domain
from .ev_service_milp import canonical_service_plan,check_service_plan


@pytest.fixture(scope='module')
def electricity():return electrical_audit()


@pytest.fixture(scope='module')
def operations():return operation_audit()


def test_EV_and_ESS_equal_for_identical_grid_PQ(electricity):
    assert electricity['same_location_same_PQ_equivalence_max']==0.
    assert len(electricity['rows'])==6 and electricity['ac_calls']==25
    for r in electricity['rows']:
        assert max(r['same_PQ_EV_ESS_max_differences'].values())==0.
        assert r['grid_energy_kwh']==3.


def test_consistent_rebase_preserves_full_physical_AC(electricity):
    assert electricity['consistent_rebase_voltage_max']<1e-12
    for r in electricity['rows']:
        for check in r['rebase_checks']:
            assert check['max_master_p_kw_difference']<1e-8
            assert check['max_master_q_kvar_difference']<1e-8
            assert check['ac_equation_residual']<2e-10
            assert check['power_balance_residual']<2e-10


def test_EV_scale_and_noise_are_declared_not_old_synthetic_scale(electricity):
    alternating=[r for r in electricity['rows'] if r['curve']=='alternating']
    for r in alternating:
        assert r['units']['base_kw']==1000.
        assert np.max(r['ratio_to_site_static_p_kw'])==pytest.approx(.05 if r['site']==0 else 1/15)
        assert np.asarray(r['delta_v_noise_units'])==pytest.approx(np.asarray(r['delta_terminal_v_pu'])/.00012)
        assert .01<r['max_abs_delta_v_noise_units']<.5
    assert electricity['case']['field_PQV_available'] is False


def test_location_changes_each_terminal_response(electricity):
    rows={r['site']:r for r in electricity['rows'] if r['curve']=='alternating'}
    for site,r in rows.items():
        dv=np.asarray(r['delta_terminal_v_pu'])
        assert np.all(dv[[0,2]]<0) and np.all(dv[[1,3]]==0)
        assert np.argmin(dv[0])==site
    assert rows[1]['max_abs_delta_v_noise_units']<rows[0]['max_abs_delta_v_noise_units']
    assert rows[1]['max_abs_delta_v_noise_units']<rows[2]['max_abs_delta_v_noise_units']


def test_worker_window_positive_and_negative_with_same_battery_service(operations):
    rows={r['context']['kind']:r for r in operations['rows']}
    early=rows['ev_remote_preposition_early_worker'];late=rows['ev_remote_preposition_late_worker']
    assert early['allowed_slots']==3 and late['allowed_slots']==2
    assert early['service_feasible'] and not late['service_feasible']
    assert early['required_grid_energy_kwh']==pytest.approx(3.+.2/.95)
    assert early['context']['travel_before_observation_slots']==1
    assert early['context']['trip_energy_kwh']==.2
    assert early['context']['total_declared_task_hours']==1.25
    for p in early['plans'].values():
        assert p['terminal_energy_kwh']==pytest.approx(26.85)
        assert p['validation']['feasible']


def test_in_window_travel_plus_setup_is_infeasible_without_free_staff_shortcut(operations):
    rows={r['context']['kind']:r for r in operations['rows']}
    r=rows['ev_remote_in_window_early_worker']
    assert r['operations']['arrival_slot']==1 and r['operations']['ready_slot']==2
    assert r['allowed_slots']==2
    assert r['maximum_grid_energy_kwh']==3.
    assert r['required_grid_energy_kwh']>3.
    assert not r['service_feasible']
    assert r['plans']['quality_lp']['status']==2


def test_two_slots_at_initial_site_remove_power_curve_freedom(operations):
    r=next(r for r in operations['rows'] if r['context']['kind']=='ev_initial_late_worker')
    assert r['allowed_slots']==2 and r['service_feasible']
    for p in r['plans'].values():
        assert p['u_kw']==pytest.approx([0.,0.,6.,6.])
        assert p['terminal_energy_kwh']==pytest.approx(26.85)


def test_public_affine_prices_have_independent_LP_gap_and_operation_validation(operations):
    feasible=[r for r in operations['rows'] if r['service_feasible']]
    assert len(feasible)==5
    for r in feasible:
        for method in ('quality_lp','random_price_lp'):
            p=r['plans'][method]
            assert p['primal_dual_gap_numeric']==pytest.approx(0.,abs=1e-10)
            assert p['dual_stationarity_inf']<1e-10
            assert p['operation_violation']<1e-10
            assert p['validation']['feasible']
            assert p['complete_curve_enumeration'] is False


def test_service_units_can_be_rebased_without_changing_kW_or_SOC():
    for kind in ('fixed_ess_initial_connected','ev_remote_preposition_early_worker'):
        plans=[]
        for base in (20.,1000.):
            service,site,_=service_case(kind,base_kw=base)
            d=charging_service_domain(site,4,3,service,power_scale=base)
            solved=solve_affine_curve([1.,9.,1.,9.],d)
            plan=canonical_service_plan(site,solved['u_kw']/base,3,service)
            check=check_service_plan(plan,service)
            plans.append((solved['u_kw'],plan.soc,check['terminal_battery_energy']*base))
        assert plans[0][0]==pytest.approx(plans[1][0])
        assert plans[0][1]==pytest.approx(plans[1][1])
        assert plans[0][2]==pytest.approx(plans[1][2])


def test_invalid_physical_and_operation_inputs_are_rejected():
    case=case33_local_standard()
    with pytest.raises(ValueError):physical_response(case,3,[1.])
    with pytest.raises(ValueError):physical_response(case,0,[np.nan])
    with pytest.raises(ValueError):physical_response(case,0,[1.],base_mva=0.)
    with pytest.raises(ValueError):service_case('missing_case')
