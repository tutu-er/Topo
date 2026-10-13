"""Independent linear toy audit: future P/Q uncertainty changes EV design.

The main selector calls the existing MILP, with continuous powers and a
time-expanded operations network. Closed forms are used only by independent
tests/audit output. This toy has no AC physics, topology truth or future noise.
All scores are squared whitened residual norms (SSE), not half-SSE or EIG.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
import inspect
import platform

import numpy as np
import scipy


from research_experiments.mobile_ev_mle import history_qp_acceleration
from research_experiments.mobile_ev_mle.ev_information_milp import (
    FrozenPair, adaptive_search, solve_pair,
)
from research_experiments.mobile_ev_mle.ev_operations_milp import (
    Arc, EVConfig, EVPlan, check_plan,
)


def operations_config():
    return EVConfig(n_slots=2, n_sites=1, dt_hours=1., p_max_kw=6.,
        net_grid_energy_kwh=6., eta_charge=1., eta_discharge=1.,
        travel_energy_kwh=0., allow_discharge=False,
        max_throughput_kwh=6.)


def fixed_power_pair():
    return FrozenPair("fixed_future_PQ", np.zeros(2), np.diag([2., 1.5]),
        np.zeros((2, 0)), np.zeros((0, 0)), np.zeros(0), np.zeros(0),
        metadata={"scope": "linear toy only", "future_power_profiled": False})


def free_power_pair():
    # First two rows are future power-meter residuals. The last two rows are
    # responses allowing the same future power correction to alter them.
    return FrozenPair("profiled_future_PQ", np.zeros(4),
        np.array([[0., 0.], [0., 0.], [2., 0.], [0., 1.5]]),
        np.array([[1., 0.], [0., 1.], [4., 0.], [0., .1]]),
        np.zeros((0, 2)), np.full(2, -10.), np.full(2, 10.),
        metadata={"scope": "linear toy only", "future_power_profiled": True})


def shared_line_pair():
    # One historical line-parameter penalty z^2; the SAME abstract line
    # correction enters both future rows. This is not a shared source example.
    return FrozenPair("shared_line_with_history", np.zeros(3),
        np.array([[0., 0.], [1., 0.], [0., 1.]]), np.ones((3, 1)),
        np.zeros((0, 1)), np.array([-10.]), np.array([10.]),
        metadata={"scope": "abstract scalar line correction", "history_rows": 1})


def offset_center_pairs():
    return [FrozenPair("offset_center_" + str(int(offset)), np.array([offset]),
        np.array([[1., -1.]]), np.zeros((1, 0)), np.zeros((0, 0)),
        np.zeros(0), np.zeros(0)) for offset in (-6., 6.)]


def one_site_plan(powers, config=None):
    config = operations_config() if config is None else config
    u = np.asarray(powers, float).reshape(2, 1)
    pp, pm = np.maximum(u, 0.), np.maximum(-u, 0.)
    energy0 = config.initial_soc * config.battery_capacity_kwh
    energy = np.r_[energy0, energy0 + np.cumsum(config.dt_hours * (
        config.eta_charge * pp[:, 0] - pm[:, 0] / config.eta_discharge))]
    return EVPlan((Arc(0, 1, 0, 0), Arc(1, 2, 0, 0)), np.ones(2),
        pp, pm, u, energy)


def select(config, pairs):
    """MILP-only selection API: no truth, observations or future noise inputs."""
    return adaptive_search(config, pairs, initial_us=([3., 3.],),
        max_iterations=16, segments=2, max_segments=64, solver_seconds=5.,
        total_seconds=60., absolute_gap=1e-6, relative_gap=1e-7, seed=0)


def _qp_record(pair, powers):
    result = solve_pair(pair, powers)
    return {key: result[key] for key in (
        "success", "status", "q_lower", "q_upper", "numeric_gap", "optimality",
        "roundoff_guard", "z", "acceleration", "semantics")}


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _search_record(result, config):
    record = {key: result[key] for key in (
        "success", "status", "proxy_lower", "proxy_upper", "absolute_gap",
        "relative_gap", "records", "solver_records", "failures", "qp_count",
        "cut_counts", "witness_counts", "final_segments", "pair_hashes", "seconds",
        "seed", "enumerated_complete_actions", "bound_scope", "selector_uses_truth",
        "selector_uses_future_noise")}
    record["plan"] = result["plan"].as_dict() if result["plan"] is not None else None
    record["operations_audit"] = (check_plan(result["plan"], config)
        if result["plan"] is not None else None)
    return record


def run(output_path=None):
    config = operations_config()
    fixed, free, shared = fixed_power_pair(), free_power_pair(), shared_line_pair()
    fixed_result = select(config, [fixed])
    free_result = select(config, [free])
    centers = offset_center_pairs()
    robust_result = select(config, centers)
    baseline = one_site_plan([3., 3.], config)
    baseline_fixed = _qp_record(fixed, baseline.u_kw)
    baseline_free = _qp_record(free, baseline.u_kw)
    cross = {}
    for name, result in (("fixed_selected", fixed_result), ("free_selected", free_result)):
        if result["plan"] is not None:
            cross[name] = {
                "fixed_metric": _qp_record(fixed, result["plan"].u_kw),
                "profiled_metric": _qp_record(free, result["plan"].u_kw)}
    source_paths = [Path(__file__),
        Path(inspect.getfile(FrozenPair)), Path(inspect.getfile(EVConfig)),
        Path(inspect.getfile(history_qp_acceleration))]
    test_path = Path(__file__).with_name("test_future_power_metric_check.py")
    if test_path.is_file():
        source_paths.append(test_path)
    gains = {}
    for name, pair, result, baseline_score in (
            ("fixed_metric", fixed, fixed_result, baseline_fixed),
            ("profiled_metric", free, free_result, baseline_free)):
        if result["plan"] is None:
            gains[name] = dict(status="selected_plan_unavailable",
                               gain_lower=None, gain_upper=None)
            continue
        point = _qp_record(pair, result["plan"].u_kw)
        gains[name] = dict(
            gain_lower=point["q_lower"]-baseline_score["q_upper"],
            gain_upper=point["q_upper"]-baseline_score["q_lower"],
            selected_point_qp=point,
            interval_scope="same selected control QP bounds minus baseline QP bounds",
            global_optimum_gain_upper=result["proxy_upper"]-baseline_score["q_lower"])
    if "fixed_selected" in cross:
        fixed_under_profile = cross["fixed_selected"]["profiled_metric"]
        gains["fixed_selected_under_profiled_metric"] = dict(
            gain_lower=fixed_under_profile["q_lower"]-baseline_free["q_upper"],
            gain_upper=fixed_under_profile["q_upper"]-baseline_free["q_lower"])
    record = dict(
        scope="linear toy; not AC, topology accuracy, calibrated test power, or EIG",
        score_definition="SSE: squared whitened residual norm; not half-SSE",
        experiment="future power profiling can reverse an MILP optimized EV timing",
        config=asdict(config),
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        source_sha256={str(path): sha256(path.read_bytes()).hexdigest() for path in source_paths},
        matrix_hashes={pair.name: pair.fingerprint() for pair in [fixed, free, shared, *centers]},
        fixed_milp=_search_record(fixed_result, config),
        profiled_milp=_search_record(free_result, config),
        multi_center_milp=_search_record(robust_result, config),
        constant_charge_baseline=dict(plan=baseline.as_dict(), operations_audit=check_plan(baseline, config),
            fixed_metric=baseline_fixed, profiled_metric=baseline_free),
        gain_vs_constant_charge=gains,
        cross_evaluation=cross,
        shared_line_audit=dict(balanced=_qp_record(shared, [3., 3.]),
            pulse=_qp_record(shared, [6., 0.])),
        audit_method="independent closed forms in tests; no action/route grid",
        selector_enumerated_complete_actions=0,
        selector_inputs_exclude_truth_and_future_noise=True)
    result = _json_safe(record)
    if output_path is not None:
        path = Path(output_path)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    destination = (Path(__file__).parent / "evidence" / "ev_metric_revision" /
                   "future_power_metric_check.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = run(destination)
    print(json.dumps({"output": str(destination), "scope": result["scope"],
        "fixed": {key: result["fixed_milp"][key] for key in ("status", "proxy_lower", "proxy_upper", "seconds")},
        "profiled": {key: result["profiled_milp"][key] for key in ("status", "proxy_lower", "proxy_upper", "seconds")},
        "fixed_u": result["fixed_milp"]["plan"]["u_kw"],
        "profiled_u": result["profiled_milp"]["plan"]["u_kw"],
        "center_u": result["multi_center_milp"]["plan"]["u_kw"]}, ensure_ascii=False))
