"""Exact fixed-RX L1 baseline and stable-location profile extensions.
No oracle inputs; existing MILP and calibration implementation are untouched.
"""
from time import perf_counter
import numpy as np
from theft_wzzt.theft.theft_model import TheftTree, fit_theft_milp, predict
from theft_wzzt.theft.credibility import evaluate


def fixed_profile(tree, data, weights, *, stable=False):
    started = perf_counter()
    r, x = weights
    z, c = tree.incidence()
    base = (data.p @ z.T * r + data.q @ z.T * x) @ z
    no_cost = np.abs(data.y-base).sum(1)/data.voltage_scale + np.abs(data.balance)/data.balance_scale
    costs = []
    for j in range(len(tree.candidates)):
        effect = ((data.amplitude[:, None]*r + data.extra_q[:, None]*x)*c[:, j]) @ z
        active = np.abs(data.y-base-effect).sum(1)/data.voltage_scale + np.abs(data.balance-data.amplitude)/data.balance_scale
        costs.append(np.where(data.amplitude > 0, active, no_cost))
    costs = np.array(costs).T
    local_gain = np.maximum(no_cost[:, None]-costs, 0)
    h = int(np.argmax(local_gain.sum(0)))
    choice = np.full(len(data.p), h) if stable else local_gain.argmax(1)
    selection = np.zeros((len(data.p), len(tree.candidates)), int)
    for t, j in enumerate(choice):
        if local_gain[t, j] > 1e-10:
            selection[t, j] = 1
    s0 = evaluate(data, base, np.zeros(len(data.p)))
    s1 = evaluate(data, predict(tree, data, r, x, selection), data.amplitude*selection.sum(1))
    return dict(method='fixed_stable' if stable else 'fixed_free',
        h0_loss=s0['loss'], h1_loss=s1['loss'], gain=s0['loss']-s1['loss'],
        gain_voltage=s0['voltage_loss']-s1['voltage_loss'], gain_balance=s0['balance_loss']-s1['balance_loss'],
        best_location=tree.candidates[h] if stable and selection.any() else None,
        locations=[tree.candidates[int(row.argmax())] if row.any() else None for row in selection],
        profile=[dict(location=label, loss=float(np.minimum(costs[:, j], no_cost).sum())) for j, label in enumerate(tree.candidates)],
        seconds=perf_counter()-started, method_certificate='exact finite L1 enumeration with fixed R/X')


def stable_profile(tree, data, bounds, *, optional_activity=True, time_limit=60.):
    started = perf_counter()
    h0 = fit_theft_milp(tree, data, allow_theft=False, weight_bounds=bounds, time_limit=time_limit)
    s0 = evaluate(data, h0.prediction, h0.balance_prediction)
    best, best_tree, best_label = h0, tree, None
    rows = []
    for label in tree.candidates:
        candidate = TheftTree(tree.edges, tree.observed, (label,))
        forced = None if optional_activity else {i: 0 if a > 0 else -1 for i, a in enumerate(data.amplitude)}
        fit = fit_theft_milp(candidate, data, fixed_locations=forced, weight_bounds=bounds, time_limit=time_limit)
        rows.append(dict(location=label, loss=fit.objective, diagnostics=fit.diagnostics))
        if fit.objective < best.objective-1e-8:
            best, best_tree, best_label = fit, candidate, label
    s1 = evaluate(data, best.prediction, best.balance_prediction)
    return dict(method='stable_optional' if optional_activity else 'stable_positive',
        h0_loss=s0['loss'], h1_loss=s1['loss'], gain=s0['loss']-s1['loss'],
        gain_voltage=s0['voltage_loss']-s1['voltage_loss'], gain_balance=s0['balance_loss']-s1['balance_loss'],
        best_location=best_label, locations=best.selected_locations(best_tree), profile=rows,
        h0_diagnostics=h0.diagnostics, h1_diagnostics=best.diagnostics, seconds=perf_counter()-started,
        method_certificate='H0 union all one-location subproblems; optional activity MILP' if optional_activity else 'H0 union fixed-positive-activity location LPs')
