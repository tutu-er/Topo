"""Topology-free price-design baseline and a separate empirical DR evaluator.

The planner sees means and historical meter inputs only. Realized response
factors are supplied later by an evaluator. This module neither runs WZZT nor
interprets its ordinary D-optimal score as a topology probability.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

MENUS = ("P_2", "P_5", "P_10", "P_15")
BANDS = {"before": slice(0, 16), "event": slice(16, 18), "after": slice(18, 24)}
NO_REWARD = "no_reward"


def _matrix(values, name, *, hours=None, nonnegative=False):
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or not a.shape[0] or not a.shape[1]:
        raise ValueError(f"{name} must be a nonempty time-by-node matrix")
    if hours is not None and a.shape[0] != hours:
        raise ValueError(f"{name} must have {hours} hourly rows")
    if not np.isfinite(a).all() or (nonnegative and np.any(a < 0)):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return a


@dataclass(frozen=True)
class MenuModel:
    mean_factors: Mapping[str, np.ndarray]
    prices: Mapping[str, np.ndarray]
    mean_status: str = "provisional training-only means; future predictive adequacy is not established"

    def __post_init__(self):
        for menu in MENUS:
            for table, label in [(self.mean_factors, "factor"), (self.prices, "price")]:
                v = np.asarray(table[menu], dtype=float)
                if v.shape != (24,) or not np.isfinite(v).all() or np.any(v < 0):
                    raise ValueError(f"{menu} {label} must contain 24 finite nonnegative values")

    @classmethod
    def load(cls, mean_csv, price_csv):
        factors = {m: np.full(24, np.nan) for m in MENUS}
        prices = {m: np.full(24, np.nan) for m in MENUS}
        seen_factors, seen_prices = set(), set()
        with Path(mean_csv).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["menu"] not in MENUS:
                    continue
                key = (row["menu"], row["band"])
                if key in seen_factors or row["band"] not in BANDS:
                    raise ValueError("duplicate or unknown menu-band mean")
                seen_factors.add(key)
                factors[row["menu"]][BANDS[row["band"]]] = float(row["factor"])
        with Path(price_csv).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["Price_signal"] not in MENUS:
                    continue
                hour = int(row["Hour"])
                key = (row["Price_signal"], hour)
                if key in seen_prices or not 1 <= hour <= 24:
                    raise ValueError("duplicate or invalid price hour")
                seen_prices.add(key)
                prices[key[0]][hour - 1] = float(row["Experiment_price_NOK_kWh"])
        return cls(factors, prices)


def predefined_candidates(n_nodes):
    """No reward, four uniform menus, and four menus at each single node."""
    if not isinstance(n_nodes, (int, np.integer)) or n_nodes < 1:
        raise ValueError("n_nodes must be a positive integer")
    result = [{"name": NO_REWARD, "menus": [NO_REWARD] * n_nodes}]
    for menu in MENUS:
        result.append({"name": "all_" + menu, "menus": [menu] * n_nodes})
        for node in range(n_nodes):
            labels = [NO_REWARD] * n_nodes
            labels[node] = menu
            result.append({"name": f"node_{node}_{menu}", "menus": labels})
    return result


def _labels(assignment, n_nodes):
    labels = list(assignment["menus"] if isinstance(assignment, dict) else assignment)
    if len(labels) != n_nodes or any(x not in (*MENUS, NO_REWARD) for x in labels):
        raise ValueError("one known menu or no_reward is required per node")
    return labels


def _menu_arrays(labels, model):
    price = np.column_stack([np.zeros(24) if m == NO_REWARD else model.prices[m] for m in labels])
    factor = np.column_stack([np.ones(24) if m == NO_REWARD else model.mean_factors[m] for m in labels])
    return price, factor


def settlement(price, settlement_baseline_kw, actual_p_kw):
    """One-hour records; floor signed accumulated credits once at horizon end.

    Each column is one declared settlement account (terminal aggregator here).
    A one-day test horizon is explicit, not the historical iFlex final payout.
    """
    baseline = _matrix(settlement_baseline_kw, "settlement_baseline_kw", hours=24, nonnegative=True)
    actual = _matrix(actual_p_kw, "actual_p_kw", hours=24, nonnegative=True)
    p = _matrix(price, "price", hours=24, nonnegative=True)
    if baseline.shape != actual.shape or p.shape != actual.shape:
        raise ValueError("settlement arrays must share shape")
    credits = np.sum(p * (baseline - actual), axis=0)
    paid = np.maximum(credits, 0.0)
    # actual>=0 and nonnegative signal imply this deterministic bound.
    upper = np.sum(p * baseline, axis=0)
    if np.any(paid > upper + 1e-9):
        raise AssertionError("settlement exceeded its nonnegative-load bound")
    return credits, paid, float(upper.sum())


def operating_checks(p_kw, baseline_p_kw, *, power_limits_kw=None, max_daily_energy_change_fraction=None):
    p = _matrix(p_kw, "p_kw", hours=24, nonnegative=True)
    baseline = _matrix(baseline_p_kw, "baseline_p_kw", hours=24, nonnegative=True)
    if p.shape != baseline.shape:
        raise ValueError("baseline and response shapes differ")
    energy0 = baseline.sum(axis=0)
    delta = p.sum(axis=0) - energy0
    fraction = np.divide(delta, energy0, out=np.zeros_like(delta), where=energy0 > 0)
    if np.any((energy0 == 0) & (p.sum(axis=0) > 0)):
        raise ValueError("response creates load on a zero-baseline account")
    violations = []
    if power_limits_kw is not None:
        limits = np.broadcast_to(np.asarray(power_limits_kw, dtype=float), (p.shape[1],))
        if not np.isfinite(limits).all() or np.any(limits < 0):
            raise ValueError("power limits must be finite and nonnegative")
        nodes = np.flatnonzero(np.any(p > limits[None, :] + 1e-9, axis=0))
        if len(nodes):
            violations.append({"kind": "power", "node_positions": nodes.tolist(),
                               "max_excess_kw": float(np.maximum(p - limits, 0).max())})
    if max_daily_energy_change_fraction is not None:
        cap = float(max_daily_energy_change_fraction)
        if not np.isfinite(cap) or cap < 0:
            raise ValueError("daily energy change bound must be nonnegative")
        nodes = np.flatnonzero(np.abs(fraction) > cap + 1e-9)
        if len(nodes):
            violations.append({"kind": "daily_energy_change", "node_positions": nodes.tolist()})
    return {"valid": not violations, "violations": violations, "clipped_samples": 0,
            "total_peak_kw": float(p.sum(axis=1).max()),
            "daily_energy_change_kwh_per_node": delta.tolist(),
            "daily_energy_change_fraction_per_node": fraction.tolist()}


def plan_prices(forecast_p_kw, forecast_q_kvar, model, *, budget_nok,
                history_p_kw=None, history_q_kvar=None, candidates=None,
                settlement_baseline_kw=None, power_limits_kw=None,
                max_daily_energy_change_fraction=None, objective="d_opt",
                ridge=1e-6, budget_mode="conservative"):
    """Enumerate predeclared prices using only forecast means and history.

    d_opt is the ordinary input-design baseline, not a proposed WZZT score.
    RMS scales are fixed from historical columns before any candidate is scored.
    Conservative budget uses full positive settlement baseline, not E[payment].
    """
    p = _matrix(forecast_p_kw, "forecast_p_kw", hours=24, nonnegative=True)
    q = _matrix(forecast_q_kvar, "forecast_q_kvar", hours=24)
    if p.shape != q.shape:
        raise ValueError("forecast P/Q shapes differ")
    budget = float(budget_nok)
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and nonnegative")
    if budget_mode not in ("conservative", "mean_response"):
        raise ValueError("unknown budget mode")
    baseline = p if settlement_baseline_kw is None else _matrix(settlement_baseline_kw, "settlement_baseline_kw", hours=24, nonnegative=True)
    if baseline.shape != p.shape:
        raise ValueError("settlement baseline shape differs")
    scales = None
    if objective == "d_opt":
        hp = _matrix(history_p_kw, "history_p_kw", nonnegative=True)
        hq = _matrix(history_q_kvar, "history_q_kvar")
        if hp.shape != hq.shape or hp.shape[1] != p.shape[1]:
            raise ValueError("historical P/Q dimensions differ from forecast")
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("D-opt ridge must be strictly positive")
        history = np.column_stack((hp, hq))
        scales = np.sqrt(np.mean(history ** 2, axis=0))
        scales = np.where(scales > 0, scales, 1.0)
        scaled = history / scales
        information = scaled.T @ scaled + float(ridge) * np.eye(2 * p.shape[1])
    elif objective != "peak":
        raise ValueError("objective must be d_opt or peak")
    audits, feasible = [], []
    candidates = predefined_candidates(p.shape[1]) if candidates is None else list(candidates)
    for assignment in candidates:
        labels = _labels(assignment, p.shape[1])
        prices, factors = _menu_arrays(labels, model)
        pp, qq = p * factors, q * factors
        _, payments, upper = settlement(prices, baseline, pp)
        physical = operating_checks(pp, p, power_limits_kw=power_limits_kw,
                                    max_daily_energy_change_fraction=max_daily_energy_change_fraction)
        mean_payment = float(payments.sum())
        budget_quantity = upper if budget_mode == "conservative" else mean_payment
        if objective == "d_opt":
            x = np.column_stack((pp, qq)) / scales
            sign, score = np.linalg.slogdet(information + x.T @ x)
            if sign <= 0 or not np.isfinite(score):
                raise ArithmeticError("regularized information matrix is numerically invalid")
        else:
            score = -float(pp.sum(axis=1).max())
        audit = {"name": str(assignment.get("name", "unnamed")), "menus": labels,
                 "score": float(score), "objective": objective,
                 "payment_at_mean_response_nok": mean_payment,
                 "conservative_payment_bound_nok": upper, "budget_quantity_nok": budget_quantity,
                 "budget_feasible": budget_quantity <= budget + 1e-9, "operating": physical}
        audit["feasible"] = audit["budget_feasible"] and physical["valid"]
        audits.append(audit)
        if audit["feasible"]:
            # Deterministic tie order is cost, then original candidate order.
            feasible.append((float(score), -budget_quantity, -len(audits), audit, pp, qq))
    if not feasible:
        raise ValueError("no declared assignment passes planning constraints")
    selected = max(feasible, key=lambda x: x[:3])
    audit, pp, qq = selected[3:]
    return {"selected_assignment": {"name": audit["name"], "menus": audit["menus"]},
            "predicted_p_kw": pp, "predicted_q_kvar": qq, "selected_score": audit["score"],
            "selected_audit": audit, "candidates": audits, "objective": objective,
            "history_column_scales": None if scales is None else scales.tolist(),
            "ridge": float(ridge) if objective == "d_opt" else None,
            "budget_nok": budget, "budget_mode": budget_mode,
            "mean_response_status": model.mean_status,
            "forecast_baseline_operating": operating_checks(
                p, p, power_limits_kw=power_limits_kw,
                max_daily_energy_change_fraction=max_daily_energy_change_fraction),
            "constraint_scope": "mean prediction only; actual violations remain in evaluation denominators",
            "scope": "optimal only over declared finite assignments; no topology guarantee"}


def draw_per_menu_banks(banks, seed, n_nodes):
    """Draw full 24h node trajectories, coupled by a shared U across menus.

    Banks are evaluator-only; different menu banks may have different lengths.
    No sample recentering or per-hour resampling is performed.
    """
    u = float(np.random.default_rng(seed).random())
    factors, metadata = {}, {"seed": int(seed), "shared_uniform": u, "draws": {}}
    for menu in MENUS:
        a = np.asarray(banks[menu], dtype=float)
        if a.ndim != 3 or not a.shape[0] or a.shape[1] != 24 or a.shape[2] not in (1, n_nodes):
            raise ValueError(f"{menu} bank must be samples x 24 x nodes (or one shared column)")
        if not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError(f"{menu} bank contains invalid response factors")
        ix = min(int(u * len(a)), len(a) - 1)
        factors[menu] = np.broadcast_to(a[ix], (24, n_nodes)).copy()
        metadata["draws"][menu] = {"index": ix, "bank_size": len(a), "shared_node_factor": a.shape[2] == 1}
    return factors, metadata


def realize_response(actual_baseline_p_kw, actual_baseline_q_kvar, assignment, model,
                     realized_factors_by_menu, *, settlement_baseline_kw,
                     power_limits_kw=None, max_daily_energy_change_fraction=None,
                     budget_nok=None):
    """Apply evaluator-only realized factors; no clipping or hidden correction.

    Q changes at each account's baseline P/Q ratio, an explicit modeled behavior.
    Inputs must describe positive consumption, not net load including DER.
    """
    p = _matrix(actual_baseline_p_kw, "actual_baseline_p_kw", hours=24, nonnegative=True)
    q = _matrix(actual_baseline_q_kvar, "actual_baseline_q_kvar", hours=24)
    if p.shape != q.shape:
        raise ValueError("actual P/Q shapes differ")
    labels = _labels(assignment, p.shape[1])
    prices, _ = _menu_arrays(labels, model)
    factors = np.ones_like(p)
    for i, menu in enumerate(labels):
        if menu == NO_REWARD:
            continue
        f = _matrix(realized_factors_by_menu[menu], menu + " realized factors", hours=24, nonnegative=True)
        if f.shape != p.shape:
            raise ValueError("realized factor shape differs from actual baseline")
        factors[:, i] = f[:, i]
    pp, qq = p * factors, q * factors
    credits, paid, upper = settlement(prices, settlement_baseline_kw, pp)
    audit = operating_checks(pp, p, power_limits_kw=power_limits_kw,
                             max_daily_energy_change_fraction=max_daily_energy_change_fraction)
    audit.update({"settlement_horizon_hours": 24, "settlement_accounts": "one terminal aggregator per column",
                  "settlement_floor": "once at horizon end per account", "reactive_response": "same multiplicative factor as P",
                  "counterfactual_status": "semi-synthetic response applied to real baseline trajectories",
                  "payment_bound_nok": upper,
                  "violations_policy": "retain realized trajectory and report invalid; no clipping or resampling"})
    if budget_nok is not None:
        audit["budget_valid"] = float(paid.sum()) <= float(budget_nok) + 1e-9
        if not audit["budget_valid"]:
            audit["valid"] = False
            audit["violations"].append({"kind": "settlement_budget", "excess_nok": float(paid.sum()) - float(budget_nok)})
    return {"p_kw": pp, "q_kvar": qq, "credits_nok_per_node": credits,
            "payment_nok_per_node": paid, "total_payment_nok": float(paid.sum()),
            "audit": audit}
