"""Independent two-slot flexibility checks, separate from frozen AC selectors.

The selector maximizes an exact rank-one profiled linear-model distance using
one sign disjunction, battery states, continuous power, and joint site choice.
Toy sensitivities are declared inputs; no future truth or noise enters design.
This is an analytic benchmark, not a new deployed full-AC EV selector.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import argparse
import json
import platform

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


@dataclass(frozen=True)
class CurveTask:
    """All power in kW, energy in kWh; exactly two time slots.

    Arrays broadcast to (2, number_of_sites). Driving happens before slot 0,
    with its energy deducted once. One connection site is chosen for the round.
    terminal_energy is an exact battery-service task, not grid-side net energy.
    """
    n_sites: int = 1
    dt_hours: float = 1.
    p_charge_max: object = 1.
    p_discharge_max: object = 0.
    eta_charge: float = .9
    eta_discharge: float = .9
    initial_energy: float = 2.
    terminal_energy: float = 2.45
    energy_min: float = 0.
    energy_max: float = 4.
    availability: object = True
    travel_slots: object = 0
    driving_energy: object = 0.

    def arrays(self):
        if not isinstance(self.n_sites, (int, np.integer)) or self.n_sites < 1:
            raise ValueError('positive integer n_sites required')
        scalars = np.asarray([self.dt_hours, self.eta_charge, self.eta_discharge,
            self.initial_energy, self.terminal_energy, self.energy_min,
            self.energy_max], float)
        if not np.isfinite(scalars).all() or self.dt_hours <= 0:
            raise ValueError('finite scalars and positive duration required')
        if not 0 < self.eta_charge <= 1 or not 0 < self.eta_discharge <= 1:
            raise ValueError('efficiencies must lie in (0,1]')
        if not (self.energy_min <= self.initial_energy <= self.energy_max
                and self.energy_min <= self.terminal_energy <= self.energy_max):
            raise ValueError('initial and terminal battery energy outside limits')
        shape = (2, self.n_sites)
        cp = np.broadcast_to(np.asarray(self.p_charge_max, float), shape).copy()
        dp = np.broadcast_to(np.asarray(self.p_discharge_max, float), shape).copy()
        access = np.broadcast_to(np.asarray(self.availability), shape).copy()
        tau = np.broadcast_to(np.asarray(self.travel_slots, float), (self.n_sites,)).copy()
        drive = np.broadcast_to(np.asarray(self.driving_energy, float), (self.n_sites,)).copy()
        if not all(np.isfinite(a).all() for a in (cp, dp, tau, drive)):
            raise ValueError('operational arrays must be finite')
        if (cp < 0).any() or (dp < 0).any() or (drive < 0).any():
            raise ValueError('power and driving energy must be nonnegative')
        if (tau < 0).any() or not np.equal(tau, np.floor(tau)).all():
            raise ValueError('travel duration must be a nonnegative slot count')
        if not np.isin(access, [False, True]).all():
            raise ValueError('availability must be boolean')
        allowed = access.astype(bool)
        for j in range(self.n_sites):
            allowed[:min(2, int(tau[j])), j] = False
        return cp*allowed, dp*allowed, allowed, tau.astype(int), drive


def profile_signal(signal, history_precision=0.):
    """min_b ||signal-b*1||^2 + h*b^2, with unit white future errors.

    h is information on the same shared offset from history, not a free prior.
    """
    signal = np.asarray(signal, float)
    h = float(history_precision)
    if signal.ndim != 1 or not len(signal) or not np.isfinite(signal).all():
        raise ValueError('finite nonempty signal vector required')
    if not np.isfinite(h) or h < 0:
        raise ValueError('history_precision must be finite and nonnegative')
    b = float(signal.sum()/(len(signal)+h))
    residual = signal-b
    distance = float(residual@residual+h*b*b)
    return dict(distance_squared=distance, offset=b,
                future_residual=residual.tolist(), historical_penalty=h*b*b,
                closed_form=float(signal@signal-signal.sum()**2/(len(signal)+h)))


def local_rx_information(p, q, history_precision=0., sigma=1.):
    """LinDistFlow voltage-row information after shared-offset compensation."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.shape != q.shape or p.ndim != 1 or not len(p):
        raise ValueError('matching nonempty P/Q curves required')
    if not np.isfinite(p).all() or not np.isfinite(q).all() or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('finite curves and positive sigma required')
    h = float(history_precision)
    if not np.isfinite(h) or h < 0:
        raise ValueError('history_precision must be finite and nonnegative')
    a = np.column_stack([p, q])/sigma
    # h is already a dimensionless precision relative to a unit white future row.
    projector = np.eye(len(p))-np.ones((len(p), len(p)))/(len(p)+h)
    return a.T@projector@a


def battery_accounting(net_power, task, site=0):
    """Recompute service and loss from decoded net power, not split variables."""
    u = np.asarray(net_power, float)
    if u.shape != (2,) or not np.isfinite(u).all() or not 0 <= site < task.n_sites:
        raise ValueError('two finite powers and a valid site required')
    _, _, _, _, drive = task.arrays()
    charged = task.dt_hours*float(np.maximum(u, 0).sum())
    discharged = task.dt_hours*float(np.maximum(-u, 0).sum())
    loss = (1-task.eta_charge)*charged+(1/task.eta_discharge-1)*discharged
    initial_after_driving = task.initial_energy-float(drive[site])
    e = np.r_[initial_after_driving, initial_after_driving+np.cumsum(
        task.dt_hours*(task.eta_charge*np.maximum(u, 0)
                       -np.maximum(-u, 0)/task.eta_discharge))]
    return dict(grid_charge_kwh=charged, grid_export_kwh=discharged,
                grid_net_kwh=charged-discharged, grid_throughput_kwh=charged+discharged,
                conversion_loss_kwh=loss, driving_energy_kwh=float(drive[site]),
                battery_energy_kwh=e.tolist(), battery_gain_kwh=float(e[-1]-task.initial_energy),
                balance_residual_kwh=float(charged-discharged-
                    (e[-1]-task.initial_energy)-float(drive[site])-loss))


def _operational_model(task):
    cc, dd, allowed, tau, driving = task.arrays()
    n = task.n_sites
    cp = np.arange(2*n).reshape(2, n)
    dp = np.arange(2*n, 4*n).reshape(2, n)
    site = np.arange(4*n, 5*n)
    offset = 5*n
    mode = np.arange(offset, offset+2) if np.any(dd > 0) else None
    if mode is not None:
        offset += 2
    energy = np.arange(offset, offset+3)
    nv = offset+3
    lower, upper = np.zeros(nv), np.full(nv, np.inf)
    integer = np.zeros(nv, int)
    upper[cp.ravel()] = cc.ravel()
    upper[dp.ravel()] = dd.ravel()
    upper[site] = 1.
    integer[site] = 1 if n > 1 else 0
    if n == 1:
        lower[site] = 1.
    if mode is not None:
        upper[mode] = 1.
        integer[mode] = 1
    lower[energy] = task.energy_min
    upper[energy] = task.energy_max
    rows, lo, hi = [], [], []
    def row(terms, a, b):
        vector = np.zeros(nv)
        for index, coefficient in terms:
            vector[index] += coefficient
        rows.append(vector)
        lo.append(a)
        hi.append(b)
    row([(idx, 1.) for idx in site], 1., 1.)
    for t in range(2):
        for j in range(n):
            row([(cp[t,j], 1.), (site[j], -cc[t,j])], -np.inf, 0.)
            row([(dp[t,j], 1.), (site[j], -dd[t,j])], -np.inf, 0.)
        if mode is not None:
            row([(idx, 1.) for idx in cp[t]]+[(mode[t], -float(cc[t].max()))], -np.inf, 0.)
            row([(idx, 1.) for idx in dp[t]]+[(mode[t], float(dd[t].max()))], -np.inf, float(dd[t].max()))
        row([(energy[t+1], 1.), (energy[t], -1.)]
            +[(idx, -task.dt_hours*task.eta_charge) for idx in cp[t]]
            +[(idx, task.dt_hours/task.eta_discharge) for idx in dp[t]], 0., 0.)
    row([(energy[0], 1.)]+[(site[j], driving[j]) for j in range(n)],
        task.initial_energy, task.initial_energy)
    row([(energy[-1], 1.)], task.terminal_energy, task.terminal_energy)
    return dict(charge=cp, discharge=dp, site=site, mode=mode, energy=energy,
        lower=lower, upper=upper, integer=integer, rows=np.asarray(rows),
        row_lower=np.asarray(lo), row_upper=np.asarray(hi), cc=cc, dd=dd,
        allowed=allowed, travel_slots=tau, driving=driving)


def solve_rank_one_curve(task, sensitivity=None, *, time_limit=10.):
    """Joint-site MILP for max .5*(a_0*u_0-a_1*u_1)^2.

    One direction binary selects +r or -r; maximizing its amplitude z is
    equivalent to the squared distance. This exact reduction uses two white
    rows and an unanchored shared offset (h=0). It is not a full-AC certificate.
    """
    started = perf_counter()
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('positive finite time_limit required')
    model = _operational_model(task)
    a = np.ones((2, task.n_sites)) if sensitivity is None else np.broadcast_to(
        np.asarray(sensitivity, float), (2, task.n_sites)).copy()
    if not np.isfinite(a).all():
        raise ValueError('sensitivity must be finite')
    n0 = len(model['lower'])
    r = np.zeros(n0+2)
    for t in range(2):
        sign = 1. if t == 0 else -1.
        r[model['charge'][t]] = sign*a[t]
        r[model['discharge'][t]] = -sign*a[t]
    bound = float(np.sum(np.abs(a)*np.maximum(model['cc'], model['dd'])))
    big_m = 2*bound
    z, direction = n0, n0+1
    lower = np.r_[model['lower'], 0., 0.]
    upper = np.r_[model['upper'], bound, 1.]
    integer = np.r_[model['integer'], 0, 1]
    rows = np.pad(model['rows'], ((0,0),(0,2)))
    # direction=1 activates z<=r; direction=0 activates z<=-r.
    positive = -r.copy(); positive[z] = 1.; positive[direction] = big_m
    negative = r.copy(); negative[z] = 1.; negative[direction] = -big_m
    rows = np.vstack([rows, positive, negative])
    row_lo = np.r_[model['row_lower'], -np.inf, -np.inf]
    row_hi = np.r_[model['row_upper'], big_m, 0.]
    objective = np.zeros(len(lower)); objective[z] = -1.
    solver_attempts = []
    result = None
    for presolve in (True, False):
        attempt_started = perf_counter()
        result = milp(objective, integrality=integer, bounds=Bounds(lower, upper),
            constraints=LinearConstraint(rows, row_lo, row_hi),
            options={'time_limit': float(time_limit), 'mip_rel_gap': 1e-10,
                     'presolve': presolve})
        solver_attempts.append(dict(presolve=presolve, status=int(result.status),
            message=str(result.message), seconds=perf_counter()-attempt_started,
            same_constraint_matrix=True, same_objective=True))
        # Preserve the initial computational failure. Only status=4 is retried
        # without presolve; mathematical infeasibility is never retried away.
        if result.status != 4:
            break
    record = dict(status=int(result.status), message=str(result.message),
        seconds=perf_counter()-started, solver_attempts=solver_attempts,
        success=False, site=None, net_power_kw=None,
        distance_squared=None, amplitude=None, amplitude_best_bound=None,
        distance_best_bound=None, absolute_amplitude_gap=None,
        big_m=big_m, declared_contrast_absolute_bound=bound,
        formulation='rank_one_sign_disjunction_MILP', history_precision=0.,
        includes_complete_curve_library=False, includes_joint_site_choice=True,
        sensitivity=a.tolist(), operational_constraints=task_config(task))
    if result.x is None:
        return record
    solution = np.asarray(result.x, float)
    chosen = int(np.argmax(solution[model['site']]))
    u_matrix = solution[model['charge']]-solution[model['discharge']]
    u = u_matrix[:,chosen]
    decoded_r = float(np.sum(a[0]*u_matrix[0])-np.sum(a[1]*u_matrix[1]))
    amplitude = abs(decoded_r)
    account = battery_accounting(u, task, chosen)
    inequality_violation = max(float(np.max(np.maximum(rows@solution-row_hi, 0))),
                              float(np.max(np.maximum(row_lo-rows@solution, 0))))
    bounds_violation = max(float(np.max(np.maximum(lower-solution, 0))),
                          float(np.max(np.maximum(solution-upper, 0))))
    integrality_violation = float(np.max(np.abs(solution[integer == 1]-np.rint(solution[integer == 1]))))
    # Independently validate the decoded net curve; silent simultaneous cycling
    # would change the battery accounting and must remain a failure.
    physical_violation = max(abs(account['battery_energy_kwh'][-1]-task.terminal_energy),
        max(0., task.energy_min-min(account['battery_energy_kwh'])),
        max(0., max(account['battery_energy_kwh'])-task.energy_max),
        float(np.max(np.maximum(u-model['cc'][:,chosen],0))),
        float(np.max(np.maximum(-u-model['dd'][:,chosen],0))),
        float(np.max(np.abs(np.delete(u_matrix, chosen, axis=1)))) if task.n_sites > 1 else 0.,
        abs(account['balance_residual_kwh']))
    best = getattr(result, 'mip_dual_bound', None)
    if best is not None and np.isfinite(best):
        best = -float(best)
    else:
        best = None
    gap = None if best is None else best-amplitude
    record.update(site=chosen, net_power_kw=u.tolist(), net_power_by_site_kw=u_matrix.tolist(),
        amplitude=amplitude, distance_squared=.5*amplitude**2,
        amplitude_best_bound=best,
        distance_best_bound=None if best is None else .5*max(best,0.)**2,
        absolute_amplitude_gap=gap, signed_contrast=decoded_r,
        direction_binary=float(solution[direction]), z=float(solution[z]),
        solver_mip_gap=None if getattr(result,'mip_gap',None) is None else float(result.mip_gap),
        inequality_violation=inequality_violation, bounds_violation=bounds_violation,
        integrality_violation=integrality_violation,
        decoded_physical_violation=physical_violation, battery=account)
    tolerance = 2e-7
    record['success'] = bool(result.status == 0 and max(inequality_violation,
        bounds_violation, integrality_violation, physical_violation,
        abs(float(solution[z])-amplitude)) <= tolerance
        and (gap is None or gap >= -tolerance))
    return record


def one_site_one_way_lp_audit(task):
    """Independent signed LP audit for a two-slot, one-site charging task.

    Only two projection directions are checked, with no trajectory library.
    """
    if task.n_sites != 1 or np.any(np.asarray(task.p_discharge_max) != 0):
        raise ValueError('audit requires one site and charge-only task')
    cc, _, _, _, driving = task.arrays()
    energy = (task.terminal_energy-task.initial_energy+driving[0])/task.eta_charge
    answers = []
    for sign in (-1., 1.):
        g = sign*np.array([1., -1.])
        # Independent cumulative-energy rows, no use of the MILP builder.
        after = task.initial_energy-driving[0]
        cumulative = np.tril(np.ones((2,2)))*task.dt_hours*task.eta_charge
        inequality = np.vstack([cumulative, -cumulative])
        rhs = np.r_[np.repeat(task.energy_max-after,2), np.repeat(after-task.energy_min,2)]
        fit = linprog(-g, A_ub=inequality, b_ub=rhs,
            A_eq=[[task.dt_hours, task.dt_hours]], b_eq=[energy],
            bounds=[(0.,float(cap)) for cap in cc[:,0]], method='highs')
        answers.append(dict(sign=sign, status=int(fit.status), success=bool(fit.success),
            amplitude=None if fit.x is None else float(g@fit.x),
            net_power_kw=None if fit.x is None else fit.x.tolist()))
    return answers


def task_config(task):
    cc, dd, allowed, tau, drive = task.arrays()
    return dict(n_sites=int(task.n_sites), dt_hours=float(task.dt_hours),
        p_charge_max_kw=cc.tolist(), p_discharge_max_kw=dd.tolist(),
        eta_charge=float(task.eta_charge), eta_discharge=float(task.eta_discharge),
        initial_energy_kwh=float(task.initial_energy), terminal_energy_kwh=float(task.terminal_energy),
        energy_min_kwh=float(task.energy_min), energy_max_kwh=float(task.energy_max),
        effective_availability=allowed.tolist(), travel_slots=tau.tolist(), driving_energy_kwh=drive.tolist())


def run_checks():
    """Predeclared analytic examples, retaining positive and negative cases."""
    one_way = CurveTask()
    bidirectional = replace(one_way, p_discharge_max=1.)
    ideal = replace(one_way, eta_charge=1., eta_discharge=1., terminal_energy=2.5)
    staff_old = replace(ideal, availability=[[True],[False]])
    staff_new = replace(ideal, availability=[[True],[True]])
    high_priority_old = replace(ideal, availability=[[True],[False]])
    high_priority_shift = replace(ideal, availability=[[False],[True]])
    moving = replace(ideal, n_sites=2, travel_slots=[0,1], driving_energy=[0.,.1])
    travel_failure = replace(moving, travel_slots=[0,2])
    no_headroom = replace(bidirectional, terminal_energy=3.8)
    cases = {
        'one_way_terminal_service': solve_rank_one_curve(one_way),
        'v2g_same_terminal_service': solve_rank_one_curve(bidirectional),
        'staff_original_window': solve_rank_one_curve(staff_old, [[1.],[3.]]),
        'staff_added_window': solve_rank_one_curve(staff_new, [[1.],[3.]]),
        'staff_original_high_window': solve_rank_one_curve(high_priority_old, [[3.],[1.]]),
        'staff_forced_shift_loses_window': solve_rank_one_curve(high_priority_shift, [[3.],[1.]]),
        'mobile_local_blind_spot': solve_rank_one_curve(moving, [[0.,1.],[0.,1.]]),
        'mobile_travel_exhausts_window': solve_rank_one_curve(travel_failure, [[1.,10.],[1.,10.]]),
        'v2g_no_charge_headroom': solve_rank_one_curve(no_headroom),
        'all_sites_operationally_infeasible': solve_rank_one_curve(replace(ideal, travel_slots=2)),
    }
    constant = [.25,.25]
    history = {str(h): profile_signal(constant,h) for h in (0.,1.,4.)}
    rx_pure = local_rx_information([0.,1.], [0.,0.], 1.)
    rx_fixed = local_rx_information([0.,1.], [0.,.4], 1.)
    # Small independent grid audits only the one-way analytic line segment.
    grid = np.linspace(0.,.5,51)
    grid_scores = [.5*(first-(.5-first))**2 for first in grid]
    source_files = [Path(__file__), Path(__file__).with_name('test_ev_flexibility_analytic.py'),
                    Path(__file__).with_name('ev_flexibility_math.md')]
    return dict(scope='declared_two_slot_profiled_linear_examples_not_full_AC',
        version='ev_flexibility_analytic_v1', selector_uses_true_topology=False,
        selector_uses_future_noise=False, random_seed=None,
        cases=cases, constant_curve_kw=constant, constant_profile_by_history_precision=history,
        one_way_signed_LP_audit=one_site_one_way_lp_audit(one_way),
        independent_small_grid_audit=dict(points=51, full_curve_library_used_by_selector=False,
            role='independent_one_dimensional_toy_audit_only', maximum_distance_squared=max(grid_scores)),
        lin_distflow_pure_P_information=rx_pure.tolist(),
        lin_distflow_fixed_power_factor_information=rx_fixed.tolist(),
        zero_gradient_constant=dict(curve=constant, gradient=[0.,0.],
            distance_squared=0., optimized_distance_squared=cases['one_way_terminal_service']['distance_squared'],
            interpretation='a stay-on-tie affine-price iteration can stall at constant'),
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in source_files if p.exists()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run_checks(), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
