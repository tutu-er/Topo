"""Small independent action audits; enumeration is confined to these tests."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import LinearConstraint

from . import ev_mobile_v4_bounded as b
from . import ev_mobile_v4_master as m


def _domain(lo, hi, integer=None, constraints=()):
    return dict(lower=np.asarray(lo, float), upper=np.asarray(hi, float),
                integrality=np.zeros(len(lo), int) if integer is None else integer,
                constraints=constraints, u_map=np.eye(len(lo)),
                control_lower=np.asarray(lo, float), control_upper=np.asarray(hi, float))


def _pairs(B, J, parameter_domain, *, residual=None, affine=None, own=(0., 0.)):
    B = np.asarray(B, float)
    if residual is None: residual = np.zeros(B.shape[0])
    return [b.build_pair(a, c, residual, B, J, np.zeros((1, B.shape[1])), parameter_domain,
        self_interval=own, affine_control=affine)
        for a, c in [('a', 'b'), ('b', 'a')]]


def test_scalar_explanation_exchange_attains_binary_audit_optimum():
    pairs = _pairs([[1.]], [[-1.]], b.make_domain([-.5], [.5]))
    domain = _domain([0.], [1.], integer=[1])
    possible = [np.array([[0.]]), np.array([[1.]])]
    truth = max(b.evaluate_pairs(pairs, u)['lower'] for u in possible)
    result = m.solve_bounded_milp(pairs, domain, initial_operations={'parent': [0.]},
        segments=2, max_iterations=3, backend='scipy', qp_backend='scipy', solver_seconds=5.)
    assert result['success'] and result['inheritance_protected']
    assert result['selected_origin'].startswith('bounded_MILP_')
    assert result['power_kw'][0, 0] == pytest.approx(1.)
    assert result['objective_lower'] <= truth+1e-9 <= result['objective_upper']+1e-9
    assert result['objective_upper'] == pytest.approx(.125, abs=2e-8)
    assert result['completed']
    assert result['MILP_calls'] == 2
    assert all(len(group) == 2 for group in result['witnesses'])


def test_new_witnesses_add_rows_but_do_not_repeat_square_variables():
    pairs = _pairs(np.eye(2), -np.ones((2, 1)), b.make_domain([-1.], [1.]))
    domain = _domain([-1., -1.], [1., 1.])
    one = m.build_master(pairs, [[np.array([0.])]]*2, domain, segments=3)
    three = m.build_master(pairs, [[np.array([-.5]), np.array([0.]), np.array([.5])]]*2,
                           domain, segments=3)
    assert len(one['objective']) == len(three['objective'])
    assert one['n_segment_binary'] == three['n_segment_binary'] == 6
    assert three['constraints'].A.shape[0]-one['constraints'].A.shape[0] == 4


def test_two_time_shared_parameter_tiny_exhaustive_audit():
    pairs = _pairs(np.eye(2), -np.ones((2, 1)), b.make_domain([-.2], [.2]),
                   affine=[.1, -.2], own=(.3, .3))
    domain = _domain([0., 0.], [1., 1.], integer=[1, 1],
                     constraints=[LinearConstraint([[1., 1.]], 1., 1.)])
    possible = [np.array([[0., 1.]]), np.array([[1., 0.]])]
    exact = [b.evaluate_pairs(pairs, u)['upper'] for u in possible]
    result = m.solve_bounded_milp(pairs, domain, initial_operations={'parent': [0., 1.]},
        segments=2, max_iterations=3, backend='scipy', qp_backend='scipy', solver_seconds=5.)
    assert result['selected']['score_upper'] == pytest.approx(max(exact), abs=1e-8)
    assert result['objective_upper'] >= max(exact)-1e-8
    assert result['objective_lower'] <= max(exact)+1e-8
    np.testing.assert_allclose(result['power_kw'], [[1., 0.]], atol=1e-9)


def test_original_operation_binary_and_nonidentity_mapping_remain_intact():
    pairs = _pairs([[1., 2.]], np.zeros((1, 0)), b.make_domain([], []))
    operation = dict(lower=[0., -1., -1.], upper=[1., 1., 1.], integrality=[1, 0, 0],
        constraints=[LinearConstraint([[1., 0., 0.]], 1., 1.)],
        u_map=np.array([[0., 1., 0.], [0., 0., 1.]]),
        control_lower=[-1., -1.], control_upper=[1., 1.])
    result = m.solve_bounded_milp(pairs, operation, initial_operations={'parent': [1., .2, -.4]},
        backend='scipy', qp_backend='scipy', max_iterations=1, solver_seconds=5.)
    assert result['success']
    assert result['operation_x'][0] == 1.
    np.testing.assert_allclose(result['operation_x'][1:], result['power_kw'].ravel())
    assert result['objective_upper'] >= 4.5-1e-8


def test_parent_retained_on_solver_failure_and_valid_box_upper(monkeypatch):
    pairs = _pairs([[1.]], [[-1.]], b.make_domain([-.1], [.1]), own=(2., 2.))
    monkeypatch.setattr(m, 'solve_linear_mip', lambda *a, **kw: SimpleNamespace(
        x=None, status=4, message='deliberate unavailable solver', mip_dual_bound=None, fun=None))
    result = m.solve_bounded_milp(pairs, _domain([-1.], [1.]),
        initial_operations={'parent': [.7]}, qp_backend='scipy', backend='scipy')
    assert result['success'] and result['inheritance_protected']
    np.testing.assert_allclose(result['power_kw'], [[.7]])
    assert result['selected_origin'] == 'initial_operation:parent'
    assert {f['kind'] for f in result['failures']} == {'master_solver_failure', 'master_no_incumbent'}
    assert result['objective_upper'] >= .405-2.
    assert result['objective_lower'] <= result['selected']['score_upper']


def test_interval_overlap_does_not_replace_parent_without_proof():
    # Deliberately wide self intervals persist even for exact inner QPs.
    pairs = _pairs([[1.]], np.zeros((1, 0)), b.make_domain([], []), own=(0., 100.))
    result = m.solve_bounded_milp(pairs, _domain([-1.], [1.]),
        initial_operations={'parent': [.1]}, qp_backend='scipy', backend='scipy', max_iterations=1)
    np.testing.assert_allclose(result['power_kw'], [[.1]])
    assert len(result['proposals']) == 1
    assert abs(result['proposals'][0]['power_kw'][0, 0]) == pytest.approx(1.)
    assert result['objective_upper'] >= .5
    assert result['inheritance_protected']


def test_illegal_physical_explanation_is_rejected():
    pairs = _pairs([[1.]], [[1.]], b.make_domain([0.], [1.]))
    with pytest.raises(ValueError, match='legal original-domain'):
        m.build_master(pairs, [[[-.1]], [[0.]]], _domain([0.], [1.]))


def test_master_witness_upper_covers_exact_distance_at_all_small_grid_points():
    pairs = _pairs([[1., -.4], [.2, 1.]], [[-1.], [-.5]], b.make_domain([-.3], [.3]),
                   residual=[.5, -.2], affine=[-.3, .1], own=(.2, .3))
    domain = _domain([-1., -.5], [1., .5])
    master = m.build_master(pairs, [[[-.3], [0.], [.3]]]*2, domain, segments=4)
    for x in np.linspace(-1., 1., 9):
        for y in np.linspace(-.5, .5, 7):
            start = m.reconstruct_start(master, np.array([x, y]))
            assert m._master_audit(master, start, 1e-7)['feasible']
            upper = float(-master['objective']@start)
            exact = b.evaluate_pairs(pairs, [[x, y]])
            assert upper >= exact['upper']-1e-8


def test_exact_parameter_domain_ratio_and_support_bounds_in_master():
    Z = b.make_domain([-.2, -.2], [.2, .2], LinearConstraint([[1., -2.]], 0., .1))
    pairs = _pairs(np.eye(2), np.eye(2), Z)
    domain = _domain([-1., -1.], [1., 1.], integer=[1, 1])
    result = m.solve_bounded_milp(pairs, domain, initial_operations={'parent': [0., 0.]},
        backend='scipy', qp_backend='scipy', max_iterations=2, solver_seconds=5.)
    truth = max(b.evaluate_pairs(pairs, [[i, j]])['upper'] for i in [-1., 0., 1.] for j in [-1., 0., 1.])
    assert result['objective_upper'] >= truth-1e-8
    assert result['objective_lower'] <= truth+1e-8
    assert all(Z.audit(z)['feasible'] for group in result['witnesses'] for z in group)


def test_complete_mobile_domain_proves_zero_products_and_shares_all_pair_squares():
    from .ev_mobile_service_domain import MobileServiceConfig, build_mobile_service_domain
    from .ev_mobile_v3_domain import tighten_operation_bounds
    cfg = MobileServiceConfig(n_slots=4, n_sites=2, dt_hours=1.,
        p_charge_max_kw=1., p_discharge_max_kw=1., battery_capacity_kwh=10.,
        initial_soc=.5, soc_min=0., soc_max=1., terminal_energy_kwh=5.,
        eta_charge=1., eta_discharge=1., travel_energy_kwh=0., travel_slots=1,
        setup_slots=1, max_moves=1)
    physical = build_mobile_service_domain(cfg)
    operation = tighten_operation_bounds(physical)
    B = np.zeros((4, 8))
    for t in range(4): B[t, 2*t:2*t+2] = [1., 2.]
    Z = b.make_domain([], [])
    pairs = [b.build_pair(a, c, np.zeros(4), B, np.zeros((4, 0)), np.zeros((4, 2)), Z)
             for a in ['a', 'b', 'c'] for c in ['a', 'b', 'c'] if a != c]
    witnesses = [[np.zeros(0)] for pair in pairs]
    compact = m.build_master(pairs, witnesses, operation, segments=3)
    generic = m.build_master(pairs, witnesses, operation, segments=3, shared_control_squares=False)
    proof = compact['shared_control_square']
    assert proof['used'] and proof['zero_product_certificate']['valid']
    assert len(proof['needed_zero_products']) == 2
    assert compact['n_segment_binary'] == 18
    assert generic['n_segment_binary'] == 72
    assert compact['segment_binary_saving'] == 54
    # An independently constructed legal nonzero full route/energy vector.
    x = np.zeros(physical.model.n_variables)
    x[physical.indices.connected_arc[:, 0]] = 1.
    x[physical.indices.energy] = [5., 6., 5., 5., 5.]
    x[physical.indices.u[0, 0]], x[physical.indices.u[1, 0]] = 1., -1.
    x[physical.indices.charge[0, 0]] = 1.; x[physical.indices.discharge[1, 0]] = 1.
    x[physical.indices.mode[0]] = 1.
    assert physical.check(x)['feasible']
    start = m.reconstruct_start(compact, x)
    assert m._master_audit(compact, start, 1e-7)['feasible']
    u = np.asarray(operation['u_map']@x).reshape(4, 2)
    assert -compact['objective']@start >= b.evaluate_pairs(pairs, u)['upper']-1e-8


def test_false_exclusion_label_is_not_accepted_as_a_zero_product_proof():
    pairs = _pairs([[1., 2.]], np.zeros((1, 0)), b.make_domain([], []))
    domain = _domain([-1., -1.], [1., 1.])
    domain['zero_product_pairs'] = [(0, 1)]
    domain['zero_product_proof'] = 'claimed_single_EV_without_actual_links'
    master = m.build_master(pairs, [[np.zeros(0)]]*2, domain)
    assert not master['shared_control_square']['used']
    assert master['shared_control_square']['reason'].startswith('zero_product_certificate_failed')
    assert master['pair_audits'][0]['form'] == 'generic_affine_factor_square'


def test_unsupported_real_cross_time_term_safely_uses_generic_master():
    pairs = _pairs([[1., .2], [.3, 1.]], np.zeros((2, 0)), b.make_domain([], []))
    master = m.build_master(pairs, [[np.zeros(0)]]*2, _domain([-1., -1.], [1., 1.]))
    assert not master['shared_control_square']['used']
    assert master['shared_control_square']['reason'] == 'unsupported_nonzero_Gram_cross_term'
    for point in [np.array([1., 1.]), np.array([-1., 1.]), np.array([.2, -.3])]:
        start = m.reconstruct_start(master, point)
        assert -master['objective']@start >= b.evaluate_pairs(pairs, point)['upper']-1e-8


def test_numeric_cross_dot_is_enclosed_instead_of_silently_dropped():
    B = [[1., 1.], [1., -1.+1e-15]]
    pairs = _pairs(B, np.zeros((2, 0)), b.make_domain([], []))
    master = m.build_master(pairs, [[np.zeros(0)]]*2, _domain([-1., -1.], [1., 1.]), segments=2)
    assert master['shared_control_square']['used']
    assert master['shared_control_square']['enclosed_numeric_cross_terms'] == 2
    assert master['pair_audits'][0]['quadratic_rounding_upper'] > 0.
    for point in [np.array([1., 1.]), np.array([-1., 1.])]:
        start = m.reconstruct_start(master, point)
        assert -master['objective']@start >= b.evaluate_pairs(pairs, point)['upper']-1e-12


def test_zero_control_derivatives_do_not_create_unused_square_binaries():
    pairs = _pairs(np.zeros((1, 3)), np.zeros((1, 0)), b.make_domain([], []))
    master = m.build_master(pairs, [[np.zeros(0)]]*2, _domain([-1.]*3, [1.]*3))
    assert master['shared_control_square']['used']
    assert master['n_segment_binary'] == 0
