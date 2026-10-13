"""Independent theoretical checks for nuisance-projected branch information.

The main action selector does not import this module. Only the audit below
uses a tiny grid; the demonstrated power step uses the existing operational
MILP. All distance guarantees here are for the declared linear models.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
from pathlib import Path
import json
import platform
import numpy as np
import scipy
from scipy.optimize import lsq_linear

from .single_site_milp import SingleSiteConstraints, solve_site_step


ENGINEERING_SCALE = .003
OMEGA = .5*np.diag([ENGINEERING_SCALE**2, ENGINEERING_SCALE**2])


def projected_information(signal, nuisance, *, rank_rtol=1e-10):
    """Independent SVD projection in original whitened parameter units."""
    a, b = np.asarray(signal, float), np.asarray(nuisance, float)
    if (a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.isfinite(rank_rtol) or rank_rtol <= 0):
        raise ValueError('finite matching two-dimensional matrices and positive rank tolerance required')
    if b.shape[1]:
        q, singular, vt = np.linalg.svd(b, full_matrices=False)
        tolerance = rank_rtol*singular[0] if len(singular) else 0.
        rank = int(np.count_nonzero(singular > tolerance))
        q = q[:, :rank]
        coefficients = ((vt[:rank].T/singular[:rank])@(q.T@a)
                        if rank else np.zeros((b.shape[1], a.shape[1])))
        residual = a-q@(q.T@a)
    else:
        singular, tolerance, rank = np.empty(0), 0., 0
        coefficients = np.empty((0, a.shape[1]))
        residual = a.copy()
    info = residual.T@residual
    return dict(information=.5*(info+info.T), residual=residual,
                coefficients=coefficients, nuisance_rank=rank,
                nuisance_singular_values=singular, rank_tolerance=float(tolerance))


def _scale_factor(omega):
    omega = np.asarray(omega, float)
    if (omega.ndim != 2 or omega.shape[0] != omega.shape[1]
            or not np.isfinite(omega).all()
            or not np.allclose(omega, omega.T, rtol=0, atol=1e-15)):
        raise ValueError('finite symmetric second-moment matrix required')
    values, vectors = np.linalg.eigh(omega)
    tol = 1e-12*max(float(np.linalg.norm(omega, 2)), np.finfo(float).tiny)
    if np.min(values, initial=0.) < -tol:
        raise ValueError('positive semidefinite second moment required')
    return vectors@np.diag(np.sqrt(np.maximum(values, 0.)))


def information_half_value(signal, nuisance, omega=OMEGA):
    a, b = np.asarray(signal, float), np.asarray(nuisance, float)
    s = _scale_factor(omega)
    if a.ndim != 2 or a.shape[1] != s.shape[0]:
        raise ValueError('signal and second moment dimensions differ')
    fit = projected_information(a@s, b)
    return .5*float(np.sum(fit['residual']**2))


def compensation_price(signal, nuisance, d_signal, d_nuisance, omega=OMEGA):
    """Envelope derivative including the changing nuisance compensation."""
    a, b, da, db = [np.asarray(x, float) for x in
                     (signal, nuisance, d_signal, d_nuisance)]
    if da.shape != a.shape or db.shape != b.shape or not np.isfinite(da).all() or not np.isfinite(db).all():
        raise ValueError('finite derivatives with matching shapes required')
    s = _scale_factor(omega)
    fit = projected_information(a@s, b)
    residual, coefficients = fit['residual'], fit['coefficients']
    signal_only = float(np.sum(residual*(da@s)))
    compensation = -float(np.sum(residual*(db@coefficients)))
    return dict(price=signal_only+compensation, signal_term=signal_only,
                compensation_term=compensation, nuisance_rank=fit['nuisance_rank'])


def _four_slot_data(h, a):
    if not np.isfinite([h, a]).all() or h < 0 or a <= 1:
        raise ValueError('nonnegative history precision and a>1 required')
    return np.array([1., a, a, 1.]), float(h+2+2*a)


def four_slot_information(power, h=10., a=4.):
    power = np.asarray(power, float)
    w, total = _four_slot_data(h, a)
    if power.shape != (4,) or not np.isfinite(power).all():
        raise ValueError('four finite powers required')
    return float(w@(power*power)-(w@power)**2/total)


def four_slot_gradient(power, h=10., a=4.):
    power = np.asarray(power, float)
    w, total = _four_slot_data(h, a)
    if power.shape != (4,) or not np.isfinite(power).all():
        raise ValueError('four finite powers required')
    return 2*w*(power-float(w@power)/total)


def four_slot_global_shape(h=10., a=4.):
    """Analytic three vertex classes; no list of power curves is searched."""
    _, total = _four_slot_data(h, a)
    classes = dict(low_low=2-4/total,
                   low_high=a+1-(a+1)**2/total,
                   high_high=2*a-4*a*a/total)
    power = np.array([0., 1., 1., 0.]) if h >= a-1 else np.array([1., 1., 0., 0.])
    return dict(power=power, information=four_slot_information(power, h, a),
                vertex_class_values=classes, high_minus_mixed=(a-1)*(h+1-a)/total,
                optimal_class='high_high' if h > a-1 else 'low_high' if h < a-1 else 'tie')


def operational_price_step(power, h=10., a=4., *, trust_radius=1.):
    """Same operation MILP selects the affine price step, not a curve grid."""
    power = np.asarray(power, float)
    constraints = SingleSiteConstraints(
        dt_hours=1., p_charge_max=1., p_discharge_max=0., energy=2.,
        battery_capacity=10., initial_soc=.2, soc_min=0., soc_max=1.,
        eta_charge=1., eta_discharge=1., availability=np.ones((4, 1), bool),
        travel_slots=np.zeros((1, 1), int), travel_energy=np.zeros((1, 1)))
    result = solve_site_step(0, 4, 1, constraints, base_power=power,
        trust_radius=trust_radius, intercepts=[four_slot_information(power, h, a)],
        gradients=four_slot_gradient(power, h, a)[None, :], time_limit=5.)
    plan = result.get('plan')
    return dict(success=bool(result['success']), status=result['status'],
                power=None if plan is None else plan.p[:, 0].tolist(),
                actual_information=None if plan is None else four_slot_information(plan.p[:, 0], h, a),
                incumbent=result.get('incumbent'), best_bound=result.get('best_bound'),
                absolute_gap=result.get('absolute_gap'), termination=result.get('termination'),
                validation=result.get('validation'),
                bound_scope='frozen affine operational price problem')


def tiny_grid_audit(h, a, global_value):
    """Only independent audit enumerates this 19-point declared grid."""
    rows = []
    for power in product((0., .5, 1.), repeat=4):
        if abs(sum(power)-2.) < 1e-12:
            value = four_slot_information(power, h, a)
            rows.append(dict(power=list(power), information=value,
                             below_analytic_global=value <= global_value+1e-12))
    return rows


def run_audit(seed=20261014):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=(18, 2)), rng.normal(size=(18, 3))
    fit = projected_information(a, b)
    direction = np.array([.003, .002])
    adjustment = np.linalg.lstsq(b, -a@direction, rcond=1e-10)[0]
    direct = float(np.sum((a@direction+b@adjustment)**2))
    projected = float(direction@fit['information']@direction)
    positive_directions = np.array([[ENGINEERING_SCALE, 0.], [0., ENGINEERING_SCALE]])
    individual_squared = [float(v@fit['information']@v) for v in positive_directions]
    average_half = .5*float(np.mean(individual_squared))
    trace_half = information_half_value(a, b)
    second_moment = positive_directions.T@positive_directions/2
    covariance = second_moment-np.outer(positive_directions.mean(0), positive_directions.mean(0))

    da, db = rng.normal(size=a.shape)*.2, rng.normal(size=b.shape)*.2
    price = compensation_price(a, b, da, db)
    step = 1e-5
    plus = information_half_value(a+step*da, b+step*db)
    minus = information_half_value(a-step*da, b-step*db)
    price_fd = (plus-minus)/(2*step)
    unit_increment = (plus-trace_half)/step

    ah, an = rng.normal(size=(12, 2)), rng.normal(size=(9, 2))
    sh, sn = rng.normal(size=(12, 2)), rng.normal(size=(9, 2))
    lh, ln = rng.normal(size=(12, 3)), rng.normal(size=(9, 2))
    historical_b = np.c_[sh, lh]
    joint_a = np.r_[ah, an]
    joint_b = np.r_[np.c_[sh, lh, np.zeros((12, 2))],
                    np.c_[sn, np.zeros((9, 3)), ln]]
    ih = projected_information(ah, historical_b)['information']
    ijn = projected_information(joint_a, joint_b)['information']
    increment_eigenvalues = np.linalg.eigvalsh(ijn-ih)
    h_local = projected_information(np.c_[ah, sh], lh)['residual']
    n_local = projected_information(np.c_[an, sn], ln)['residual']
    two_stage = projected_information(np.r_[h_local[:, :2], n_local[:, :2]],
                                      np.r_[h_local[:, 2:], n_local[:, 2:]])['information']
    rank_def_b = np.c_[b[:, 0], 2*b[:, 0], np.zeros(18)]
    rank_def_fit = projected_information(a, rank_def_b)
    rank_def_direct = a+rank_def_b@np.linalg.lstsq(rank_def_b, -a, rcond=1e-10)[0]

    boundary_signal = .5
    boundary_fit = lsq_linear(np.ones((1, 1)), np.array([-boundary_signal]), bounds=(0., np.inf),
                              tol=1e-12, lsmr_tol=1e-12)
    boundary_cone = float(np.sum((boundary_signal+boundary_fit.x)**2))
    free_boundary = projected_information(np.ones((1, 1)), np.ones((1, 1)))['information'][0, 0]
    weak_info = projected_information(np.array([[1., 0.], [2., 0.]]), np.ones((2, 1)))['information']
    ev_minus_zero = np.diag([1.5**2-1., .5**2-1.])

    strong = four_slot_global_shape(10., 4.)
    weak = four_slot_global_shape(0., 4.)
    constant = np.full(4, .5)
    alternating = np.array([1., 0., 1., 0.])
    strong_step = operational_price_step(constant)
    local_step = operational_price_step(alternating)
    strong_grid = tiny_grid_audit(10., 4., strong['information'])
    weak_grid = tiny_grid_audit(0., 4., weak['information'])
    threshold_checks = []
    for h in (0., .2, 3., 10.):
        for precision in (1.5, 2., 4., 7.):
            shape = four_slot_global_shape(h, precision)
            classes = shape['vertex_class_values']
            threshold_checks.append(dict(h=h, a=precision,
                value_error=abs(shape['information']-max(classes.values())),
                difference_error=abs(classes['high_high']-classes['low_high']-shape['high_minus_mixed'])))
    nonlinear = [dict(zeta=z, nuisance=float(np.sqrt(z)), true_mean=z,
                      competing_mean=float(np.sqrt(z)**2), linear_squared_distance=z*z,
                      actual_squared_distance=float((z-np.sqrt(z)**2)**2),
                      nuisance_over_zeta=float(1/np.sqrt(z))) for z in (.1, .01, .001)]

    checks = dict(
        projection_equals_independent_least_squares=abs(direct-projected) < 1e-14,
        positive_direction_average_equals_trace=abs(average_half-trace_half) < 1e-15,
        declared_omega_is_second_moment=np.allclose(second_moment, OMEGA, atol=0., rtol=1e-14),
        omega_is_not_covariance=bool(np.linalg.norm(second_moment-covariance) > 1e-6),
        complete_compensation_price_matches_fd=abs(price['price']-price_fd) < 1e-12,
        omitted_compensation_changes_price=abs(price['price']-price['signal_term']) > 1e-8,
        expected_unit_increment_approaches_price=abs(unit_increment-price['price']) < 1e-9,
        history_added_information_psd=bool(increment_eigenvalues.min() > -1e-12),
        two_stage_projection_equals_joint=np.max(abs(two_stage-ijn)) < 1e-11,
        rank_deficient_projection_correct=bool(rank_def_fit['nuisance_rank'] == 1 and
            np.max(abs(rank_def_fit['residual']-rank_def_direct)) < 1e-12),
        ev_minus_no_ev_not_psd=bool(np.linalg.eigvalsh(ev_minus_zero).min() < 0),
        active_cone_exceeds_free_projection=bool(boundary_fit.success and abs(boundary_cone-.25) < 1e-10
                                                 and abs(free_boundary) < 1e-14),
        pure_x_direction_unresolved=bool(weak_info[0, 0] > 0 and abs(weak_info[1, 1]) < 1e-14),
        strong_milp_selects_global_shape=bool(strong_step['success'] and
            np.max(abs(np.asarray(strong_step['power'])-strong['power'])) < 1e-8 and
            strong_step['validation']['feasible']),
        strong_information_exceeds_constant_and_seeds=bool(strong['information'] >
            max(four_slot_information(constant), four_slot_information(alternating))),
        local_seed_can_stay_suboptimal=bool(local_step['success'] and
            abs(local_step['actual_information']-four_slot_information(alternating)) < 1e-10 and
            local_step['actual_information'] < strong['information']-1e-8),
        weak_precision_heuristic_fails=bool(four_slot_information([0, 1, 1, 0], 0., 4.) < weak['information']),
        zero_constant_gradient_with_positive_excitation=bool(np.max(abs(four_slot_gradient(constant, 0., 4.))) == 0.
                                                              and weak['information'] > 0),
        tiny_grid_is_bounded_by_analytic_shapes=bool(len(strong_grid) == len(weak_grid) == 19 and
            all(row['below_analytic_global'] for row in strong_grid+weak_grid)),
        global_class_threshold_formula=bool(max(row['value_error'] for row in threshold_checks) < 1e-12 and
            max(row['difference_error'] for row in threshold_checks) < 1e-12),
        nonlinear_counterexample_retained=bool(all(row['linear_squared_distance'] > 0 and
            row['actual_squared_distance'] < 1e-28 for row in nonlinear))
    )
    checks = {key: bool(value) for key, value in checks.items()}
    return dict(version='structured_information_theory_checks_v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(), seed=seed,
        status='passed' if all(checks.values()) else 'failed',
        check_count=len(checks), passed_count=sum(bool(v) for v in checks.values()), checks=checks,
        config=dict(engineering_scale_pu=ENGINEERING_SCALE, omega_pu_squared=OMEGA.tolist(),
                    matrix_price_fd_step=step, h_strong=10., h_weak=0., a=4.,
                    audit_grid=[0., .5, 1.], operational_power_units='normalized pu',
                    operational_dt_hours=1., operational_energy_pu_hours=2.),
        projection=dict(independent_squared_distance=direct, projected_squared_distance=projected,
                        information_pu_inverse_squared=fit['information'].tolist(),
                        positive_direction_squared_distances=individual_squared,
                        trace_half_distance=trace_half, second_moment=second_moment.tolist(),
                        covariance=covariance.tolist()),
        price=dict(**price, finite_difference=price_fd, absolute_error=abs(price['price']-price_fd),
                   finite_expected_unit_increment=unit_increment),
        history_increment=dict(eigenvalues=increment_eigenvalues.tolist(),
                               two_stage_max_error=float(np.max(abs(two_stage-ijn)))),
        rank_deficiency=dict(nuisance_rank=rank_def_fit['nuisance_rank'],
                             pure_x_information=weak_info.tolist()),
        boundary=dict(free_distance=float(free_boundary), cone_squared_distance=boundary_cone,
                      solver_success=bool(boundary_fit.success)),
        ev_minus_zero_information=ev_minus_zero.tolist(),
        strong=dict(global_power=strong['power'].tolist(), global_information=strong['information'],
                    class_values=strong['vertex_class_values'], constant_information=four_slot_information(constant),
                    alternating_information=four_slot_information(alternating),
                    price_step=strong_step, local_seed_step=local_step, grid_rows=strong_grid),
        weak=dict(global_power=weak['power'].tolist(), global_information=weak['information'],
                  class_values=weak['vertex_class_values'], constant_information=four_slot_information(constant, 0., 4.),
                  constant_gradient=four_slot_gradient(constant, 0., 4.).tolist(),
                  precision_heuristic_information=four_slot_information([0, 1, 1, 0], 0., 4.), grid_rows=weak_grid),
        threshold_rows=threshold_checks, nonlinear_counterexample=nonlinear,
        theory_scope='declared local linear models and the analytic scalar four-slot shape',
        full_AC_global_optimality_certified=False,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261014)
    args = parser.parse_args()
    evidence = run_audit(args.seed)
    root = Path(__file__).resolve().parent
    evidence['source_hashes'] = {name: sha256((root/name).read_bytes()).hexdigest() for name in
        ('structured_information_checks.py', 'test_structured_information_checks.py',
         'structured_information_math.md', 'structured_information_literature.json', 'single_site_milp.py')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=evidence['status'], passed=evidence['passed_count'],
                         total=evidence['check_count'], output=str(args.output))))
    return 0 if evidence['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())