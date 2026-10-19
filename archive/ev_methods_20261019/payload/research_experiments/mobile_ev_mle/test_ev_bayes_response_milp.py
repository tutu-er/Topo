"""Independent parent-envelope, grid and operation checks for the new master."""
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import LinearConstraint

from . import ev_bayes_response_milp as b
from . import ev_bayes_small_milp as old
from .test_ev_bayes_small_milp import bank, domain
from .ev_mobile_service_domain import MobileServiceConfig, build_mobile_service_domain


def dense_problem():
    covariance = np.asarray([[[[1.,.35],[.35,2.]], [[2.,-.2],[-.2,.8]], [[1.5,.1],[.1,1.2]]]])
    model = bank([[[.2,-.3],[.8,.5],[-.4,.9]]], cov=covariance, prior=[.2,.3,.5])
    J = np.asarray([[[[1.,.3],[.2,1.]], [[.2,-.4],[.7,.1]], [[-.3,.6],[.8,-.2]]]])
    return model, J


def test_parent_envelope_tightens_old_proxy_and_is_parent_exact_to_roundoff():
    model, J = dense_problem()
    p = np.array([[.217,-.133]])
    op = domain([-1.,-1.], [1.,1.])
    previous = old.build_bc_master(op, model, J, p, segments=2, exp_segments=3)
    current = b.build_bc_master(op, model, J, p, segments=2, exp_segments=3)
    before = old.evaluate_proxy(previous, p)
    after = b.evaluate_proxy(current, p)
    assert before['actual_proxy_error'] > 1e-3
    assert abs(after['actual_proxy_error']) < 3e-13
    assert current['anchor_proxy_error'] == after['actual_proxy_error']
    assert current['n_square_binaries'] == previous['n_square_binaries']
    assert current['n_exp_binaries'] == 0
    for a in current['pair_audits']:
        assert a['parent_B'] in a['exp_grid']
        assert a['max_branch_count'] <= 3


def test_lower_distance_and_one_sided_error_hold_on_independent_dense_grid():
    model, J = dense_problem()
    p = np.array([[.2,-.1]])
    master = b.build_bc_master(domain([-1.,-1.], [1.,1.]), model, J, p, segments=3, exp_segments=7)
    assert not master['shared_control_square']['used']
    for u in product(np.linspace(-1., 1., 13), repeat=2):
        answer = b.evaluate_proxy(master, np.asarray([u]))
        assert answer['C'] == pytest.approx(old.frozen_bc_value(model, J, p, np.asarray([u]))['C'], abs=2e-15)
        assert -3e-13 <= answer['actual_proxy_error'] <= master['proxy_uniform_error']+3e-13
        for value, audit in zip(answer['pairs'], master['pair_audits']):
            assert value['B_lower'] <= value['B']+3e-13
            assert value['B_lower'] >= audit['covariance_term']
            assert value['B_tangent'] <= value['B']+3e-13
            assert value['B_chord']-value['B'] <= value['epsilon_B']+3e-13


def test_zero_mean_gradient_keeps_quadratic_escape_and_frozen_improvement():
    model = bank([[[0.],[0.]]])
    J = np.asarray([[[[1.]], [[-1.]]]])
    result = b.solve_bc_milp(domain([-1.], [1.]), model, J, [[0.]], backend='scipy')
    assert result['success'] and result['physical_audit']['feasible']
    assert abs(result['power_kw'][0,0]) == pytest.approx(1., abs=1e-8)
    assert result['frozen_C'] == pytest.approx(.5*np.exp(-.5), abs=1e-11)
    assert result['frozen_C_nonincrease']
    assert result['anchor_proxy_error'] < 1e-12


def test_binary_microenumeration_checks_global_master_optimum_and_bound_direction():
    model, J = dense_problem()
    op = domain([-1.,-1.], [1.,1.], integer=[1,1], constraints=[LinearConstraint([[1.,1.]], 0., 0.)])
    p = np.zeros((1,2))
    master = b.build_bc_master(op, model, J, p, segments=3, exp_segments=7)
    feasible = [np.array([[a,-a]]) for a in (-1.,0.,1.)]
    values = [b.evaluate_proxy(master, u) for u in feasible]
    answer = b.solve_bc_milp(op, model, J, p, segments=3, exp_segments=7, backend='scipy')
    assert answer['success']
    assert answer['proxy_value'] == pytest.approx(min(v['proxy'] for v in values), abs=2e-9)
    assert answer['solver_best_bound'] == pytest.approx(answer['proxy_value'], abs=2e-9)
    assert answer['frozen_C_optimum_lower'] <= min(v['C'] for v in values)+1e-10
    assert answer['frozen_C'] <= b.frozen_bc_value(model, J, p, p)['C']+1e-12


def test_exact_max_graph_cannot_inflate_distance_under_decreasing_objective():
    model, J = dense_problem()
    anchor = np.array([[.2,-.1]])
    # Fix both controls at a point where the three max branches differ.
    for u in ((-.7,.6), (.5,.1), (.2,-.1)):
        u = np.asarray([u])
        op = domain([-1.,-1.], [1.,1.], constraints=[LinearConstraint(np.eye(2), u.ravel(), u.ravel())])
        answer = b.solve_bc_milp(op, model, J, anchor, segments=2, exp_segments=5, backend='scipy')
        expected = b.evaluate_proxy(b.build_bc_master(op, model, J, anchor, segments=2, exp_segments=5), u)
        assert answer['success']
        assert answer['solver_incumbent'] == pytest.approx(expected['proxy'], abs=2e-9)


def test_covariance_floor_avoids_large_shifted_chord_artifact():
    model = bank([[[0.],[0.]]], cov=[[[[1.]], [[9.]]]])
    J = np.zeros((1,2,1,1))
    answer = b.solve_bc_milp(domain([-1.], [1.]), model, J, [[.3]], backend='scipy')
    expected_B = .5*np.log(5./3.)
    assert answer['success']
    assert answer['frozen_C'] == pytest.approx(.5*np.exp(-expected_B), abs=3e-14)
    assert answer['proxy_value'] == pytest.approx(answer['frozen_C'], abs=1e-12)
    assert answer['n_exp_binaries'] == 0


def test_full_mobile_domain_shared_squares_and_parent_with_travel_are_kept():
    cfg = MobileServiceConfig(n_slots=4,n_sites=2,dt_hours=.5,p_charge_max_kw=2.,p_discharge_max_kw=0.,
        allow_discharge=False,battery_capacity_kwh=10.,initial_soc=.4,terminal_energy_kwh=5.,
        eta_charge=1.,eta_discharge=1.,setup_slots=0,travel_slots=1,travel_energy_kwh=.1,
        terminal_site=1,mobile=True,max_moves=1)
    op = build_mobile_service_domain(cfg)
    parent = np.array([[1.1,0.],[0.,0.],[0.,1.1],[0.,0.]])
    J = np.zeros((1,2,4,8))
    for t in range(4):
        J[0,0,t,2*t:2*t+2] = [1.,2.]
    model = bank(np.zeros((1,2,4)))
    result = b.solve_bc_milp(op, model, J, parent, segments=2, backend='scipy')
    assert result['success'] and result['physical_audit']['feasible']
    assert result['shared_control_square']['used']
    assert result['shared_control_square']['zero_product_certificate']['valid']
    assert result['original_operation_constraints_preserved']
    decoded = op.decode(result['operation_x'])
    assert decoded['energy_kwh'][-1] == pytest.approx(5., abs=1e-7)
    assert sum(arc.kind == 'travel' for arc in decoded['selected_arcs']) == 1
    assert result['frozen_C_nonincrease']
    master=b.build_bc_master(op,model,J,parent,segments=2)
    start,start_audit=b._operation_start(master,op,result['operation_x'])
    dimension=master['operation_dimension']
    assert dimension==len(result['operation_x']) and dimension>parent.size
    assert np.array_equal(start[:dimension],result['operation_x']) and np.isnan(start[dimension:]).all()
    assert start_audit['physical']['feasible']


def test_power_trust_region_and_native_failure_fields_are_preserved(monkeypatch):
    model = bank([[[0.],[0.]]])
    J = np.asarray([[[[1.]], [[-1.]]]])
    answer = b.solve_bc_milp(domain([-1.], [1.]), model, J, [[.1]], power_radius=.2, backend='scipy')
    assert answer['success'] and -.1-1e-8 <= answer['power_kw'][0,0] <= .3+1e-8
    monkeypatch.setattr(b, 'solve_linear_mip', lambda *a, **k: SimpleNamespace(x=None, fun=None, status=4,
        message='intentional failure', mip_dual_bound=-100., mip_gap=None))
    failed = b.solve_bc_milp(domain([-1.], [1.]), model, J, [[.1]])
    assert not failed['success'] and failed['failure'] == 'no_solver_incumbent'
    assert failed['solver_best_bound'] is None and failed['operation_x'] is None


def test_finite_log_prior_preserves_rare_pairs_and_probability_bound():
    from .ev_bayes_small_probability import bhattacharyya_bank
    model = bank(np.zeros((1,3,1)), prior=[1.,1e-200,1e-200])
    model['log_prior'] = np.log(model['prior'])
    J = np.zeros((1,3,1,1))
    answer = b.solve_bc_milp(domain([-1.], [1.]), model, J, [[0.]], backend='scipy')
    assert answer['success'] and len(answer['pair_audits']) == 3
    assert answer['pair_audits'][-1]['weight'] == pytest.approx(1e-200, abs=0., rel=1e-12)
    assert answer['frozen_C'] == pytest.approx(bhattacharyya_bank(model)['bound'], abs=1e-210)


def test_exp_parent_close_to_old_knot_has_stable_chord_slopes():
    grid = np.array([0., .3, .3+1e-14, 1.])
    slopes = [r[2] for r in b._exp_chords(grid)]
    assert np.diff(slopes).min() >= -1e-14
    for x in np.linspace(0.,1.,101):
        envelope = max(slope*x+intercept for _, _, slope, intercept in b._exp_chords(grid))
        expected = np.interp(x, grid, np.exp(-grid))
        assert envelope == pytest.approx(expected, abs=3e-15)


def test_gurobi_parent_envelope_matches_independent_scalar_solution():
    from .ev_original_mip_backend import gurobi_available
    if not gurobi_available()['available']:
        pytest.skip('Gurobi unavailable')
    model = bank([[[.2],[0.]]])
    J = np.asarray([[[[1.]], [[-1.]]]])
    answer = b.solve_bc_milp(domain([-1.], [1.]), model, J, [[.13]], backend='gurobi',operation_start=np.array([.13]))
    assert answer['success'] and answer['backend'] == 'gurobi'
    assert abs(answer['power_kw'][0,0]) == pytest.approx(1., abs=1e-7)
    assert answer['frozen_C_nonincrease'] and answer['anchor_proxy_error'] < 1e-12
    assert answer['start_supplied'] and answer['start_used'] and answer['start_provided_count']==1
    assert 'acceptance or improvement is not inferred' in answer['start_used_meaning']


def test_old_and_new_submit_identical_complete_operation_prefix_only(monkeypatch):
    model,J=dense_problem();op=domain([-1.,-1.],[1.,1.]);p=np.array([[.2,-.1]])
    calls=[]
    def solver(c,integrality,bounds,constraints,**options):
        calls.append((np.asarray(c),integrality,bounds,constraints,options))
        return SimpleNamespace(x=None,fun=None,status=1,message='test time limit',mip_dual_bound=0.,mip_gap=None,
            start_used=False,start_policy='test backend: only submission inspected',
            start_used_meaning='submission does not establish acceptance')
    monkeypatch.setattr(b,'solve_linear_mip',solver)
    for fn in (b.solve_bc_milp,b.solve_legacy_bc_milp):
        result=fn(op,model,J,p,operation_start=p.ravel(),seconds=2.5)
        start=calls[-1][-1]['start']
        assert np.array_equal(start[:2],p.ravel()) and np.isnan(start[2:]).all()
        assert len(start)==len(calls[-1][0])
        assert calls[-1][-1]['time_limit']==2.5
        assert result['start_supplied'] and result['start_provided_count']==2 and not result['start_used']
        assert result['operation_start_audit']['operation']['feasible']
    # Legacy wrapper supplies the unchanged original matrix, bounds and cost.
    original=old.build_bc_master(op,model,J,p)
    cost,integ,bounds,constraints,_=calls[-1]
    assert np.array_equal(cost,original['c']) and np.array_equal(integ,original['integrality'])
    assert np.array_equal(bounds.lb,original['bounds'].lb) and np.array_equal(bounds.ub,original['bounds'].ub)
    assert np.array_equal(constraints.A.toarray(),original['constraints'].A.toarray())
    assert np.array_equal(constraints.lb,original['constraints'].lb) and np.array_equal(constraints.ub,original['constraints'].ub)


def test_legacy_and_parent_envelope_same_start_have_valid_backend_results():
    model=bank([[[.2],[0.]]]);J=np.asarray([[[[1.]], [[-1.]]]])
    op=domain([-1.],[1.]);p=np.array([[.13]])
    for fn in (b.solve_bc_milp,b.solve_legacy_bc_milp):
        result=fn(op,model,J,p,backend='scipy',operation_start=p.ravel())
        assert result['success'] and result['physical_audit']['feasible']
        assert result['start_supplied'] and not result['start_used']
        assert 'explicitly ignored' in result['start_policy']
        assert result['solver_best_bound']<=result['proxy_value']+1e-8
    expected=old.solve_bc_milp(op,model,J,p,backend='scipy')
    repeated=b.solve_legacy_bc_milp(op,model,J,p,backend='scipy',operation_start=p.ravel())
    assert repeated['proxy_value']==pytest.approx(expected['proxy_value'],abs=1e-10)
    assert repeated['frozen_C']==pytest.approx(expected['frozen_C'],abs=1e-10)


def test_invalid_parent_start_cannot_be_silently_used_or_truncated():
    model=bank([[[0.],[0.]]]);J=np.asarray([[[[1.]], [[-1.]]]])
    for fn in (b.solve_bc_milp,b.solve_legacy_bc_milp):
        with pytest.raises(ValueError,match='complete finite original'):
            fn(domain([-1.],[1.]),model,J,[[0.]],operation_start=np.array([0.,0.]))
        with pytest.raises(ValueError,match='violates'):
            fn(domain([-1.],[1.]),model,J,[[0.]],operation_start=np.array([2.]))
        with pytest.raises(ValueError,match='violates'):
            fn(domain([-1.],[1.]),model,J,[[0.]],power_radius=.1,operation_start=np.array([.5]))
