"""Continuous Gaussian DR with a sufficient bound for the original L1 margin.

No scenarios or integer decisions occur in this optimizer. All residual and
margin quantities must use the same units, preferably micro p.u.^2. The
reference residual distribution is fixed during optimization. Nominal mean
power limits are not hard support limits for the unbounded Gaussian response.

Settlement is explicitly for *both* directions of planned modulation:
    price = abs(mean_u)/elasticity,
    expected/planned fee = dt * sum(price * abs(mean_u)).
This is not the older reduction-only realized-energy settlement contract.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import norm


_STATUS = {2: 'optimal', 3: 'infeasible', 4: 'infeasible_or_unbounded',
           5: 'unbounded', 9: 'time_limit', 12: 'numeric', 13: 'suboptimal'}


def prepare(problem):
    """Validate and expand a history-only coefficient problem."""
    g = np.asarray(problem['G_gradient'], float)
    if g.ndim != 2 or not g.size or not np.isfinite(g).all():
        raise ValueError('finite resource-by-time G_gradient required')
    shape = g.shape
    def shaped(key, default=None):
        value = problem[key] if key in problem else default
        a = np.asarray(value, float)
        if a.shape == (shape[0],):
            a = a[:, None]
        return np.broadcast_to(a, shape).copy()
    lower, upper = shaped('u_lower', -np.inf), shaped('u_upper', np.inf)
    lam, cv = shaped('elasticities', .5), shaped('cv', .1)
    if (np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper)
            or not np.isfinite(lam).all() or np.any(lam <= 0)
            or not np.isfinite(cv).all() or np.any(cv < 0)):
        raise ValueError('invalid nominal bounds, elasticities, or response CV')
    mu = np.asarray(problem['reference_mean'], float).reshape(-1)
    if not mu.size or not np.isfinite(mu).all():
        raise ValueError('finite nonempty reference residual mean required')
    if 'reference_noise' in problem:
        L = np.asarray(problem['reference_noise'], float)
        if L.ndim != 2 or L.shape[0] != mu.size or not np.isfinite(L).all():
            raise ValueError('reference_noise must be residual-by-factor')
        std = np.linalg.norm(L, axis=1)
        if 'reference_std' in problem and not np.allclose(
                std, problem['reference_std'], rtol=1e-10, atol=1e-12):
            raise ValueError('reference std disagrees with coefficient matrix')
    else:
        std = np.asarray(problem['reference_std'], float).reshape(-1)
    if std.shape != mu.shape or not np.isfinite(std).all() or np.any(std < 0):
        raise ValueError('invalid reference residual standard deviations')
    sign = np.asarray(problem.get('reference_sign', np.where(mu >= 0, 1., -1.)), float)
    if sign.shape != mu.shape or not np.isin(sign, [-1., 1.]).all():
        raise ValueError('reference signs must be fixed +/-1')
    b = np.asarray(problem['G_noise_base'], float).reshape(-1)
    numbers = np.array([problem['G_mean_zero'], problem['target'],
                        problem.get('alpha', .4/mu.size), problem.get('beta', 1.02),
                        problem.get('slot_hours', .25), problem.get('epsilon_G', .05),
                        problem.get('epsilon_F', .05)], float)
    if not np.isfinite(numbers).all() or not np.isfinite(b).all():
        raise ValueError('finite coefficients required')
    mean0, target, alpha, beta, dt, epsg, epsf = numbers
    if min(alpha, beta, dt) <= 0 or min(epsg, epsf) <= 0 or epsg+epsf > .1+1e-14:
        raise ValueError('positive scales and a total risk at most .1 required')
    epsj = np.asarray(problem.get('reference_risks', np.full(mu.size, epsf/mu.size)), float)
    if (epsj.shape != mu.shape or np.any(epsj <= 0) or np.any(epsj >= .5)
            or not np.isfinite(epsj).all() or epsj.sum() > epsf+1e-14):
        raise ValueError('reference risks must be positive fixed allocations')
    t = np.maximum(0., -sign*mu+norm.ppf(1-epsj)*std)
    return dict(shape=shape, gradient=g, lower=lower, upper=upper, elasticity=lam,
                cv=cv, reference_mean=mu, reference_std=std, reference_sign=sign,
                reference_risks=epsj, reference_flip_bounds=t,
                flip_penalty=float(2*alpha*beta*t.sum()), alpha=float(alpha),
                beta=float(beta), G_noise_base=b, G_mean_zero=float(mean0),
                target=float(target), dt=float(dt), epsilon_G=float(epsg),
                epsilon_F=float(epsf))


def evaluate_coefficients(problem, mean_u, *, corrected=True):
    """Recompute the conic condition from the physical mean curve, not auxiliaries."""
    p = prepare(problem)
    u = np.asarray(mean_u, float)
    if u.shape != p['shape'] or not np.isfinite(u).all():
        raise ValueError('invalid mean_u')
    a = np.abs(u)
    prices = a/p['elasticity']
    base_var = float(p['G_noise_base']@p['G_noise_base'])
    dr_coeff = p['gradient']*p['cv']*a
    dr_var = float(np.square(dr_coeff).sum())
    mean = float(p['G_mean_zero']+np.sum(p['gradient']*u))
    sd = math.sqrt(base_var+dr_var)
    risk = p['epsilon_G'] if corrected else .1
    quantile = float(norm.ppf(1-risk))
    penalty = p['flip_penalty'] if corrected else 0.
    bound = mean-quantile*sd-penalty
    lower_violation = float(np.max(p['lower']-u, initial=0.))
    upper_violation = float(np.max(u-p['upper'], initial=0.))
    return dict(G_mean=mean, G_std=sd, G_variance_base=base_var,
                G_variance_DR=dr_var, gaussian_quantile=quantile,
                signed_quantile_bound=mean-quantile*sd,
                flip_penalty=penalty, corrected_margin_bound=bound,
                target=p['target'], target_surplus=bound-p['target'],
                nominal_bound_violation=max(0., lower_violation, upper_violation),
                cost=float(p['dt']*np.sum(np.square(a)/p['elasticity'])),
                prices=prices.tolist(), mean_u=u.tolist(),
                mean_net_energy_kwh=(p['dt']*u.sum(axis=1)).tolist(),
                epsilon_G=risk, epsilon_F=p['epsilon_F'] if corrected else None,
                risk_sum=risk+p['epsilon_F'] if corrected else .1,
                scope=('sufficient 90% condition for original L1 margin under '
                       'declared Gaussian law' if corrected else
                       'naive signed-Gaussian ablation; no original-margin guarantee'),
                settlement='planned bidirectional modulation: dt*sum(abs(u)^2/lambda)',
                response_law='u+cv*abs(u)*independent_standard_normal_per_slot')


def box_bound(problem, *, corrected=True):
    """Cheap valid upper bound on the best sufficient margin in a nominal box.

    The mean is maximized coordinatewise and DR variance is omitted. Thus this
    can reject the sufficient SOCP target, not the original nonlinear margin
    or physical usefulness of DR. It may be loose but does not run a solver.
    """
    p = prepare(problem)
    if not (np.isfinite(p['lower']).all() and np.isfinite(p['upper']).all()):
        return dict(status='unavailable_unbounded_nominal_box')
    upper_mean = float(p['G_mean_zero']+np.maximum(
        p['gradient']*p['lower'], p['gradient']*p['upper']).sum())
    z = float(norm.ppf(1-(p['epsilon_G'] if corrected else .1)))
    floor = z*float(np.linalg.norm(p['G_noise_base']))
    penalty = p['flip_penalty'] if corrected else 0.
    bound = upper_mean-floor-penalty
    return dict(status='computed', maximum_signed_mean=upper_mean,
                unavoidable_gaussian_deduction=floor, flip_penalty=penalty,
                sufficient_margin_upper_bound=bound, target=p['target'],
                proves_sufficient_model_infeasible=bound<p['target']-1e-7,
                scope='upper bound for this signed sufficient condition only')


def _finite(model, key):
    try:
        value = float(model.getAttr(key))
        return value if math.isfinite(value) and abs(value) < 1e90 else None
    except Exception:
        return None


def solve_coefficients(problem, *, corrected=True, mode='min_cost', seconds=30., log_path=None):
    """Solve a convex QCP/SOCP, preserving original-form feasibility evidence.

    mode=max_bound is a finite-nominal-box diagnostic of the maximum sufficient
    margin bound. It is not a minimum-fee solution and must not be relabeled.
    """
    if mode not in ('min_cost', 'max_bound'):
        raise ValueError('mode must be min_cost or max_bound')
    p = prepare(problem)
    if mode == 'max_bound' and not (np.isfinite(p['lower']).all() and np.isfinite(p['upper']).all()):
        raise ValueError('max_bound requires finite nominal mean-power limits')
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('positive solve time required')
    import gurobipy as gp
    from gurobipy import GRB
    started = time.perf_counter()
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model('gaussian_signed_margin_with_flip_compensation', env=env)
    try:
        for key, val in dict(OutputFlag=0, Threads=1, Seed=0, NonConvex=0,
                             NumericFocus=2, FeasibilityTol=1e-8,
                             OptimalityTol=1e-8, BarQCPConvTol=1e-8,
                             DualReductions=0, TimeLimit=float(seconds)).items():
            model.setParam(key, val)
        if log_path is not None:
            path = Path(log_path); path.parent.mkdir(parents=True, exist_ok=True)
            model.setParam('LogFile', str(path)); model.setParam('OutputFlag', 1)
            model.setParam('LogToConsole', 0)
        count = int(np.prod(p['shape']))
        lo, hi = p['lower'].ravel(), p['upper'].ravel()
        g, cv, lam = p['gradient'].ravel(), p['cv'].ravel(), p['elasticity'].ravel()
        u = model.addVars(count, lb={j:float(lo[j]) for j in range(count)},
                          ub={j:float(hi[j]) for j in range(count)}, name='mean_kw')
        a = model.addVars(count, lb=0., name='mean_amplitude_kw')
        for j in range(count):
            model.addConstr(a[j] >= u[j])
            model.addConstr(a[j] >= -u[j])
        base_sd = float(np.linalg.norm(p['G_noise_base']))
        sd = model.addVar(lb=0., name='signed_margin_std')
        nonzero = [j for j in range(count) if g[j]*cv[j] != 0.]
        if not nonzero:
            model.addConstr(sd == base_sd)
        else:
            vv = model.addVars(len(nonzero)+1, lb=-GRB.INFINITY, name='noise_coeff')
            model.addConstr(vv[0] == base_sd)
            for k, j in enumerate(nonzero, 1):
                model.addConstr(vv[k] == float(g[j]*cv[j])*a[j])
            model.addQConstr(gp.quicksum(vv[k]*vv[k] for k in range(len(nonzero)+1)) <= sd*sd,
                             name='gaussian_std_soc')
        mean = p['G_mean_zero']+gp.quicksum(float(g[j])*u[j] for j in range(count) if g[j] != 0.)
        z = float(norm.ppf(1-(p['epsilon_G'] if corrected else .1)))
        penalty = p['flip_penalty'] if corrected else 0.
        bound = mean-z*sd-penalty
        cost = gp.quicksum(float(p['dt']/lam[j])*a[j]*a[j] for j in range(count))
        if mode == 'min_cost':
            model.addConstr(bound >= p['target'], name='original_margin_90pct_sufficient')
            model.setObjective(cost, GRB.MINIMIZE)
        else:
            model.setObjective(bound, GRB.MAXIMIZE)
        model.optimize()
        result = dict(status=_STATUS.get(int(model.Status), f'status_{model.Status}'),
                      status_code=int(model.Status), solutions=int(model.SolCount),
                      mode=mode, corrected=bool(corrected),
                      solver_reported_optimal=int(model.Status)==2,
                      objective=_finite(model,'ObjVal') if model.SolCount else None,
                      solver_bound=_finite(model,'ObjBound'),
                      runtime_seconds=float(model.Runtime), wall_seconds=time.perf_counter()-started,
                      nonconvex=0, scenarios=0, binary_variables=int(model.NumBinVars),
                      variables=int(model.NumVars), linear_constraints=int(model.NumConstrs),
                      quadratic_constraints=int(model.NumQConstrs),
                      flip_penalty=p['flip_penalty'] if corrected else 0.,
                      correction_sum_t=float(p['reference_flip_bounds'].sum()),
                      alpha=p['alpha'], beta=p['beta'], target=p['target'],
                      mean_u=None, audit=None, budget=None,
                      box_diagnostic=box_bound(problem, corrected=corrected),
                      covariance_scope='baseline G factors retain shared model errors; DR slot factors independent',
                      nominal_limits_only=True, exact_energy_equality_enforced=False,
                      gaussian_support_unbounded=True)
        if model.SolCount:
            out = np.array([u[j].X for j in range(count)]).reshape(p['shape'])
            amplitude = np.array([a[j].X for j in range(count)]).reshape(p['shape'])
            audit = evaluate_coefficients(problem, out, corrected=corrected)
            result.update(mean_u=out.tolist(), audit=audit,
                          auxiliary_amplitude_excess=float(np.max(amplitude-np.abs(out), initial=0.)),
                          auxiliary_cost=float(p['dt']*np.sum(amplitude**2/p['elasticity'])),
                          original_form_target_feasible=audit['target_surplus']>=-1e-7,
                          original_form_nominal_feasible=audit['nominal_bound_violation']<=1e-7)
        return result
    finally:
        model.dispose(); env.dispose()
