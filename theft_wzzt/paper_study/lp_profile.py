"""Stable-location positive-activity model as H+1 compact L1 LPs.
Activity u_t=1[a_t>0] is a declared assumption, not optimized. H0 is included
explicitly. All candidate LPs share data, scales and R/X bounds.
"""
from time import perf_counter
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack
from theft_wzzt.theft.theft_model import predict
from theft_wzzt.theft.credibility import evaluate


def fit_location_lp(tree, data, bounds, location=None):
    data.validate(len(tree.observed))
    z, c = tree.incidence()
    e = len(z)
    active = (data.amplitude > 0).astype(float) if location is not None else np.zeros(len(data.p))
    downstream = active[:, None]*c[:, tree.candidates.index(location)] if location is not None else np.zeros((len(data.p), e))
    fp = data.p @ z.T + data.amplitude[:, None]*downstream
    fq = data.q @ z.T + data.extra_q[:, None]*downstream
    ar = (fp[:, None, :]*z.T[None, :, :]).reshape(-1, e)
    ax = (fq[:, None, :]*z.T[None, :, :]).reshape(-1, e)
    design = csr_matrix(np.hstack((ar, ax))/data.voltage_scale)
    y = data.y.ravel()/data.voltage_scale
    n = len(y)
    lo, hi = [np.broadcast_to(np.asarray(b, float), (e,)) for b in bounds]
    if np.any(lo < 0) or np.any(lo >= hi) or not np.all(np.isfinite([lo, hi])):
        raise ValueError('Invalid parameter bounds')
    a_ub = vstack((hstack((design, -eye(n))), hstack((-design, -eye(n)))), format='csr')
    b_ub = np.concatenate((y, -y))
    costs = np.r_[np.zeros(2*e), np.ones(n)]
    lows = np.r_[lo, lo, np.zeros(n)]
    highs = np.r_[hi, hi, np.full(n, np.inf)]
    result = linprog(costs, A_ub=a_ub, b_ub=b_ub,
                     bounds=list(zip(lows, highs)), method='highs')
    if not result.success:
        raise RuntimeError(f'LP did not reach optimum: {result.message}')
    r, x = result.x[:e], result.x[e:2*e]
    selection = np.zeros((len(data.p), len(tree.candidates)), int)
    if location is not None:
        selection[:, tree.candidates.index(location)] = active.astype(int)
    prediction = predict(tree, data, r, x, selection)
    score = evaluate(data, prediction, data.amplitude*active)
    dual = float(b_ub @ result.ineqlin.marginals + lows @ result.lower.marginals + highs[:2*e] @ result.upper.marginals[:2*e])
    diagnostics = dict(status=int(result.status), objective_error=abs(score['voltage_loss']-result.fun),
        scaled_constraint_violation=float(max(0, np.max(a_ub @ result.x-b_ub))),
        primal_dual_gap=abs(float(result.fun)-dual), iterations=int(result.nit))
    if diagnostics['objective_error'] > 2e-5 or diagnostics['scaled_constraint_violation'] > 2e-6 or diagnostics['primal_dual_gap'] > 2e-5:
        raise RuntimeError(f'LP direct validation failed: {diagnostics}')
    return dict(**score, diagnostics=diagnostics, r=r.tolist(), x=x.tolist())


def lp_profile(tree, data, bounds):
    started = perf_counter()
    h0 = fit_location_lp(tree, data, bounds)
    rows = []
    best, label = h0, None
    for candidate in tree.candidates:
        fit = fit_location_lp(tree, data, bounds, candidate)
        rows.append(dict(location=candidate, loss=fit['loss'], diagnostics=fit['diagnostics']))
        if fit['loss'] < best['loss']-1e-8:
            best, label = fit, candidate
    return dict(method='stable_positive_lp', h0_loss=h0['loss'], h1_loss=best['loss'],
        gain=h0['loss']-best['loss'], gain_voltage=h0['voltage_loss']-best['voltage_loss'],
        gain_balance=h0['balance_loss']-best['balance_loss'], best_location=label,
        locations=[label if a > 0 else None for a in data.amplitude], profile=rows,
        r=best['r'], x=best['x'], h0_diagnostics=h0['diagnostics'], h1_diagnostics=best['diagnostics'],
        seconds=perf_counter()-started,
        method_certificate='global minimum over H0 and all stable-location fixed-positive-activity LPs; direct objective and LP dual checked')
