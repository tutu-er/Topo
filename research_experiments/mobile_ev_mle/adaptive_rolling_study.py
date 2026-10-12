"""Two-round history-only EV shadow controller and separate synthetic evaluator.

Round 1 plans eight slots but executes two. Their actual meters and known
executed EV power (tracking=1) enter H. Round 2 fits this enlarged H, repairs
the passive proposals, and replans the six remaining slots with carried SOC,
site, remaining GRID energy, throughput and terminal constraints. The complete
three-terminal star/cherry audit bank remains in every round; gates never
prune it. A new policy ID and independent full-trajectory calibration are
required. Scores maximize over BOTH required joint-fit rounds; any fit failure
is retained as +inf. Local fits and nonlinear rollback are not global MLE or
nonlinear separation certificates.

plan_round has no generator, true parameters, future observations or noise
arguments. Only rolling_trial creates truth and calls observe, after a plan is
returned. The smallest declared controller supports one-slot allowed travel;
the prefix interpreter records longer-arc transit accurately but refuses to
replan it as a fictitious arrival. This is a synthetic shadow policy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

try:
    from .likelihood import concat_observations
    from .candidates import propose, normalize_family, RadialCandidate
    from .simulation import generate, observe, model, truth_metadata, BASE_KW
    from .design import EVAction
    from .run_study import dump, fit_bank, fit_record, fingerprint
    from .adaptive_history import build_history_context, build_frozen_pairs, evaluate_frozen_pair_actual
    from .ev_operations_milp import EVConfig, EVPlan, Arc, operation_model, decode_plan, check_plan
    from .ev_information_milp import FrozenPair, solve_pair, adaptive_search
    from .adaptive_statistics import policy_trial_scores
    from .adaptive_milp_study import DEFAULT_POLICY, POPULATION, _physical_admission
except ImportError:
    from research_experiments.mobile_ev_mle.likelihood import concat_observations
    from research_experiments.mobile_ev_mle.candidates import propose, normalize_family, RadialCandidate
    from research_experiments.mobile_ev_mle.simulation import generate, observe, model, truth_metadata, BASE_KW
    from research_experiments.mobile_ev_mle.design import EVAction
    from research_experiments.mobile_ev_mle.run_study import dump, fit_bank, fit_record, fingerprint
    from adaptive_history import build_history_context, build_frozen_pairs, evaluate_frozen_pair_actual
    from ev_operations_milp import EVConfig, EVPlan, Arc, operation_model, decode_plan, check_plan
    from research_experiments.mobile_ev_mle.ev_information_milp import FrozenPair, solve_pair, adaptive_search
    from adaptive_statistics import policy_trial_scores
    from research_experiments.mobile_ev_mle.adaptive_milp_study import DEFAULT_POLICY, POPULATION, _physical_admission


ROLLING_VERSION = "rolling_two_round_shadow_v1"
FAMILIES = ((), ((0, 1),), ((0, 2),), ((1, 2),))
ROLLING_DEFAULTS = {**DEFAULT_POLICY, "version": ROLLING_VERSION,
    "execute_first_slots": 2, "tracking_gain": 1., "run_wzzt": True,
    "rolling_audit_bank": "all_four_star_cherries_every_round",
    "gates_influence_shadow_policy": False}


def _identifier(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=lambda a:np.asarray(a).tolist(),
        separators=(",", ":")).encode()).hexdigest()


def rolling_policy_spec(policy=None):
    p = {**ROLLING_DEFAULTS, **(policy or {})}
    if (p["tracking_gain"] != 1. or p["gates_influence_shadow_policy"]
            or p["execute_first_slots"] != 2 or p["future_points"] != 8):
        raise ValueError("this frozen shadow policy requires tracking=1, 2+6 slots and no gate-based pruning")
    return p


def policy_identifiers(policy, config, *, stress=None):
    # IDs are intentionally incompatible with the one-round law. The root
    # manifest must additionally freeze dependency source hashes before study.
    return dict(policy_id=_identifier(dict(kind=ROLLING_VERSION, policy=policy, ev=asdict(config),
        mandatory_fit_rounds="H+2 then H+8", required_candidate_bank=FAMILIES)),
        population_id=_identifier(dict(base=POPULATION, observation_policy=ROLLING_VERSION,
            history_points=policy["history_points"], future_points=policy["future_points"],
            declared_stress=stress, full_trace_history_unit=True)), requires_independent_rolling_calibration=True,
        may_reuse_one_round_gates=False)


@dataclass(frozen=True)
class PrefixExecution:
    slots: int
    u_kw: np.ndarray
    p_plus_kw: np.ndarray
    p_minus_kw: np.ndarray
    energy_kwh: np.ndarray
    sites: tuple[int, ...]
    next_site: int | None
    in_transit_destination: int | None
    arrival_remaining_slots: int
    grid_energy_kwh: float
    throughput_kwh: float
    driving_energy_kwh: float
    remaining_grid_energy_kwh: float


@dataclass(frozen=True)
class RollingDecision:
    plan: EVPlan
    config: EVConfig
    candidates: tuple
    context: dict
    record: dict


def execute_prefix(plan, config, slots, *, tolerance=2e-6):
    """Interpret actual prefix; deduct driving once, at departure, as the MILP."""
    if not isinstance(slots, (int, np.integer)) or not 1 <= slots <= config.n_slots:
        raise ValueError("prefix length must be an integer within the plan")
    checked = check_plan(plan, config, tolerance=tolerance)
    if not checked["feasible"]:
        raise ValueError("cannot execute invalid plan: "+str(checked["issues"]))
    pp, pm, u = plan.p_plus_kw[:slots].copy(), plan.p_minus_kw[:slots].copy(), plan.u_kw[:slots].copy()
    movement = np.zeros(slots)
    next_site, destination, remaining = config.initial_site, None, 0
    for arc in (Arc(**row) for row in checked["route"]):
        if arc.departure >= slots:
            break
        if arc.is_travel:
            movement[arc.departure] += arc.travel_energy_kwh
        if arc.arrival <= slots:
            next_site = arc.destination
        elif arc.departure < slots < arc.arrival:
            next_site, destination, remaining = None, arc.destination, arc.arrival-slots
    changes = config.dt_hours*(config.eta_charge*pp.sum(axis=1)-pm.sum(axis=1)/config.eta_discharge)-movement
    energy = np.r_[config.initial_soc*config.battery_capacity_kwh,
        config.initial_soc*config.battery_capacity_kwh+np.cumsum(changes)]
    if np.max(abs(energy-plan.energy_kwh[:slots+1]), initial=0.) > tolerance:
        raise ValueError("prefix energy is inconsistent with independently checked plan")
    grid = float(config.dt_hours*u.sum())
    return PrefixExecution(int(slots), u, pp, pm, energy, tuple(checked["sites"][:slots]),
        next_site, destination, remaining, grid, float(config.dt_hours*(pp+pm).sum()),
        float(movement.sum()), float(config.net_grid_energy_kwh-grid))


def remaining_config(config, execution, *, tolerance=2e-6):
    """Carry public battery/location/task state; never reset energy or travel."""
    if not 0 < execution.slots < config.n_slots:
        raise ValueError("a positive remaining horizon is required")
    if execution.next_site is None or execution.arrival_remaining_slots:
        raise ValueError("pending transit is recorded; this minimal controller only replans at a node")
    access = config.arrays()[0][execution.slots:].copy()
    target = execution.remaining_grid_energy_kwh
    if not config.allow_discharge and target < -tolerance:
        raise ValueError("executed charging exceeded the charge-only grid task")
    if abs(target) < tolerance:
        target = 0.
    throughput = None if config.max_throughput_kwh is None else config.max_throughput_kwh-execution.throughput_kwh
    if throughput is not None and throughput < -tolerance:
        raise ValueError("executed throughput exceeded the original task bound")
    result = replace(config, n_slots=config.n_slots-execution.slots,
        initial_site=int(execution.next_site), initial_soc=float(execution.energy_kwh[-1]/config.battery_capacity_kwh),
        net_grid_energy_kwh=float(target), access=access,
        max_throughput_kwh=None if throughput is None else max(0., throughput))
    result.arrays()
    return result


def append_executed_history(history, observed, execution, known_history_ev_kw=None):
    """Append only executed meters; known EV is deducted only in forecasting."""
    if observed.t != execution.slots or observed.n != history.n or execution.u_kw.shape != observed.p.shape:
        raise ValueError("executed measurements and actual EV prefix must match")
    known = (np.zeros_like(history.p) if known_history_ev_kw is None
        else np.asarray(known_history_ev_kw, float))
    if known.shape != history.p.shape or not np.isfinite(known).all():
        raise ValueError("known historical actual EV power must match history")
    return concat_observations(history, observed), np.vstack((known, execution.u_kw))


def combine_executed_plans(first, second, original_config, first_execution):
    """Build and independently verify the realized full route, including travel."""
    k = first_execution.slots
    if first_execution.next_site is None:
        raise ValueError("cannot stitch a pending transit as an arrived node")
    prefix_arcs = tuple(a for a in first.selected_arcs if a.departure < k)
    if any(a.arrival > k for a in prefix_arcs):
        raise ValueError("prefix contains transit beyond its boundary")
    shifted = tuple(Arc(a.departure+k, a.arrival+k, a.origin, a.destination, a.travel_energy_kwh)
                    for a in second.selected_arcs)
    arcs = prefix_arcs+shifted
    plan = EVPlan(arcs, np.ones(len(arcs)), np.vstack((first.p_plus_kw[:k], second.p_plus_kw)),
        np.vstack((first.p_minus_kw[:k], second.p_minus_kw)), np.vstack((first.u_kw[:k], second.u_kw)),
        np.r_[first_execution.energy_kwh, second.energy_kwh[1:]])
    check = check_plan(plan, original_config, tolerance=2e-6)
    if not check["feasible"]:
        raise ValueError("realized rolling route failed physical check: "+str(check["issues"]))
    return plan, check


def _audit_bank(history, run_wzzt):
    if history.n != 3:
        raise ValueError("rolling pilot is confined to three-terminal star/cherry class")
    proposed, metadata = propose(history, run_wzzt=run_wzzt, wzzt_time_limit=.3)
    if any(normalize_family(c.supports, 3) not in FAMILIES for c in proposed):
        raise ValueError("out-of-class proposal cannot be silently dropped from the audit bank")
    candidates = tuple(RadialCandidate.from_supports("star" if not f else
        "clade_"+"".join(map(str, f[0])), 3, f) for f in FAMILIES)
    return candidates, {**metadata, "rolling_full_audit_bank": True,
        "audit_bank_order": [c.name for c in candidates], "gates_prune_candidates": False}


def _feasible_seed(config, seconds):
    operations, indices = operation_model(config)
    # A public operational seed: minimize driving energy/time, with no topology.
    objective = {int(index):-(arc.travel_energy_kwh+1e-3*(arc.arrival-arc.departure))
                 for index, arc in zip(indices.arc_index, indices.arcs) if arc.is_travel}
    result = operations.solve(objective, time_limit=max(.05, seconds))
    if result.x is None:
        raise RuntimeError("remaining EV task has no operational incumbent: "+str(result.message))
    plan = decode_plan(result, indices)
    check = check_plan(plan, config, tolerance=2e-6)
    if not check["feasible"]:
        raise RuntimeError("operational seed failed physical verification")
    return plan, dict(status=int(result.status), message=str(result.message), operational=check,
        seed_uses_topology=False, seed_uses_future_data=False)


def plan_round(history, meter, config, *, policy=None, seed=0, historical_fits=None,
               known_history_ev_kw=None):
    """Pure H-only controller for either 8-slot or carried 6-slot public task."""
    started = perf_counter()
    p = rolling_policy_spec(policy)
    if config.n_sites != 3 or config.n_slots not in (6, 8) or config.allow_discharge:
        raise ValueError("initial rolling pilot uses charge-only three-site 8/6-slot tasks")
    _, _, _, travel, _, allowed = config.arrays()
    if np.any(travel[allowed] != 1):
        raise ValueError("minimal rolling policy requires one-slot allowed travel")
    candidates, proposal = _audit_bank(history, p["run_wzzt"])
    baseline, seed_record = _feasible_seed(config, p["solver_seconds"])
    context = build_history_context(history, candidates, meter, horizon=config.n_slots,
        historical_fits=historical_fits, max_centers=p["max_centers"],
        max_attempts_per_candidate=p["max_attempts"], center_loss_slack=p["history_loss_slack"],
        line_trust_fraction=p["line_trust"], source_delta_bound=p["source_delta"],
        multistart=p["fit_multistart"], max_nfev=p["fit_max_nfev"], seed=seed+100,
        passive_metadata=proposal, known_history_ev_kw=known_history_ev_kw,
        forecast_support=[dict(name=p["forecast"], baseline=p["forecast"], source_baseline=p["forecast"])])
    record = dict(history=context["summary"], proposal=proposal, policy=p, ev_config=asdict(config),
        operational_seed=seed_record, selector_uses_truth=False, selector_uses_future_noise=False,
        all_four_candidates_retained=True, gates_used_for_pruning=False,
        known_history_ev_is_actual_tracking1=True, actual_rollback_is_nonlinear_certificate=False)
    plan = baseline
    try:
        raw = build_frozen_pairs(context, baseline.u_kw, source_mode="local_refit")
        pairs = [FrozenPair(**row) for row in raw]
        searched = adaptive_search(config, pairs, initial_us=[baseline.u_kw, np.zeros_like(baseline.u_kw)],
            max_iterations=p["iterations"], segments=p["segments"], max_segments=p["max_segments"],
            solver_seconds=p["solver_seconds"], total_seconds=p["search_seconds"],
            relative_gap=p["relative_gap"], absolute_gap=p["absolute_gap"], seed=seed)
        record["proxy_search"] = {k:v for k,v in searched.items() if k != "plan"}
        if not searched["success"] or searched["plan"] is None:
            raise RuntimeError("rolling proxy search failed: "+str(searched.get("failures")))
        proposed = searched["plan"]
        admission = _physical_admission(context, raw, proposed.u_kw, limits=p["post_selection_voltage_limits"])
        record["prediction_admission"] = admission
        if not admission["feasible"]:
            raise RuntimeError("selected prediction violated physical admission")
        if not check_plan(proposed, config, tolerance=2e-6)["feasible"]:
            raise RuntimeError("selected EV plan violated operations")
        plan = proposed
        rollback = []
        none = [solve_pair(pair, np.zeros_like(plan.u_kw)) for pair in pairs]
        record["none_proxy_lower"] = min(s["q_lower"] for s in none)
        record["none_proxy_upper"] = min(s["q_upper"] for s in none)
        record["proxy_gain_lower_vs_none"] = searched["proxy_lower"]-record["none_proxy_upper"]
        for given, frozen in zip(raw, pairs):
            solved = solve_pair(frozen, plan.u_kw)
            try:
                actual = evaluate_frozen_pair_actual(context, given, plan.u_kw, solved["z"])
                rollback.append(dict(pair=given["name"], inner_success=bool(solved["success"]),
                    **{k:v for k,v in actual.items() if k not in ("actual_residual", "affine_residual")}))
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                rollback.append(dict(pair=given["name"], physically_feasible=False, error=str(error)))
        record.update(status="complete", actual_rollback=rollback,
            all_required_rollback_pairs_preserved=len(rollback)==len(raw),
            nonlinear_rollback_failure_count=sum(not row.get("physically_feasible", False) for row in rollback),
            nonlinear_rollback_is_infimum=False,
            rollback_scope="feasible-point evaluation, not nonlinear box infimum or separation certificate")
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        record.update(status="selection_failure_operational_fallback", error=str(error))
    record.update(actual_plan=plan.as_dict(), operational=check_plan(plan, config, tolerance=2e-6),
        selected_u_kw=plan.u_kw, total_selection_seconds=perf_counter()-started)
    return RollingDecision(plan, config, candidates, context, record)


def _history_initial(context):
    result = {}
    for name, centres in context["centres"].items():
        optimized = [c for c in centres if c["optimized"]]
        if optimized or centres:
            chosen = min(optimized or centres, key=lambda c:c["history_loss"])
            result[name] = chosen["fit"]
    return result


def _evaluation_path(seed, policy, stress):
    streams = np.random.SeedSequence(seed).spawn(3)
    generator = int(streams[0].generate_state(1)[0]); selector_seed = int(streams[1].generate_state(1)[0])
    rng = np.random.default_rng(streams[2]); topology = int(rng.integers(4))
    case = "strong" if rng.random() < POPULATION["strong_probability"] else "weak"
    path = generate(generator, case, history_count=policy["history_points"],
        future_count=policy["future_points"], truth_family=FAMILIES[topology])
    path.r *= rng.uniform(.8, 1.2, len(path.r)); path.x *= rng.uniform(.8, 1.2, len(path.x))
    h = path.history_count
    if stress == "bias": path.bias["v"] += np.array([-.0006, 0., .0006])
    elif stress == "source_drift": path.source_p[h:] += .10; path.source_q[h:] += .035
    elif stress == "load_trend": path.p[h:] += np.linspace(0., .05, policy["future_points"])[:, None]
    elif stress is not None: raise ValueError("unsupported declared rolling stress")
    return path, selector_seed, topology, case


def rolling_trial(seed, *, policy=None, stress=None, output=None):
    """Evaluator only: observe after action selection; mandatory H+2/H+8 fits."""
    started = perf_counter(); p = rolling_policy_spec(policy)
    access = np.ones((8, 3), bool); access[:2, 2] = False
    original = EVConfig(access=access)
    identifiers = policy_identifiers(p, original, stress=stress)
    path, selector_seed, topology, case = _evaluation_path(seed, p, stress)
    history, _ = observe(path, slice(0, path.history_count)); meter = model(3)
    known_ev = np.zeros_like(history.p)
    observed_arrays = {"history_"+key:getattr(history, key) for key in ("p", "q", "v", "master_p", "master_q", "root_v")}
    decisions, rounds, losses, successes, initial = [], [], [], [], None
    config, executed, first_execution = original, 0, None
    for round_number, count in enumerate((2, 6), start=1):
        decision = plan_round(history, meter, config, policy=p, seed=selector_seed+round_number,
            historical_fits=initial, known_history_ev_kw=known_ev)
        decisions.append(decision)
        prefix = execute_prefix(decision.plan, config, count)
        action = EVAction("rolling_round_"+str(round_number), prefix.u_kw/BASE_KW,
            np.zeros_like(prefix.u_kw), np.asarray(prefix.sites), "rolling_refit_milp")
        rows = slice(path.history_count+executed, path.history_count+executed+count)
        observed, physical = observe(path, rows, action)
        history, known_ev = append_executed_history(history, observed, prefix, known_ev)
        warm = _history_initial(decision.context)
        fitted, failures = fit_bank(decision.candidates, history, meter, initial=warm,
            seed=selector_seed+2000+round_number*100, multistart=p["fit_multistart"], max_nfev=p["fit_max_nfev"])
        round_losses = [float(fitted[c.name].loss) if c.name in fitted else np.inf for c in decision.candidates]
        round_success = [bool(c.name in fitted and fitted[c.name].success) for c in decision.candidates]
        losses.append(round_losses); successes.append(round_success)
        fit_rows = [fit_record(c, fitted[c.name]) for c in decision.candidates if c.name in fitted]
        true_index = next(i for i,c in enumerate(decision.candidates)
            if normalize_family(c.supports, 3)==normalize_family(path.candidate.supports, 3))
        row = dict(round=round_number, executed_slots=count, total_executed_slots=executed+count,
            observed_history_points=history.t, losses=round_losses, success=round_success,
            fits=fit_rows, fit_failures=failures, joint_observation_hash=fingerprint(history.p, history.q, history.v,
                history.master_p, history.master_q), true_index=true_index, selection_status=decision.record["status"],
            complete=all(round_success), voltage_min=float(physical["v"].min()), voltage_max=float(physical["v"].max()),
            operational_voltage_ok=bool(physical["v"].min()>=.9 and physical["v"].max()<=1.1),
            executed=asdict(prefix), is_global_MLE=False)
        if all(round_success):
            best = int(np.argmin(round_losses)); row.update(winner=decision.candidates[best].name,
                wrong_winner=best!=true_index, true_loss=round_losses[true_index],
                true_gap=round_losses[true_index]-min(round_losses))
            tf = fitted[decision.candidates[true_index].name]
            row["impedance_relative_rmse"] = float(np.sqrt(np.mean(np.r_[((tf.r-path.r)/path.r)**2, ((tf.x-path.x)/path.x)**2])))
        rounds.append(row)
        for key in ("p", "q", "v", "master_p", "master_q", "root_v"):
            observed_arrays[f"round_{round_number}_observed_{key}"] = getattr(observed, key)
            observed_arrays[f"round_{round_number}_joint_{key}"] = getattr(history, key)
        for name, fit in fitted.items():
            for key in ("p", "q", "source_p", "source_q"):
                observed_arrays[f"round_{round_number}__{name}__fit_{key}"] = getattr(fit, key)
        executed += count
        initial = {name:f for name,f in fitted.items() if f.success}
        if round_number == 1:
            first_execution = prefix; config = remaining_config(config, prefix)
    realized, physical_check = combine_executed_plans(decisions[0].plan, decisions[1].plan, original, first_execution)
    observed_arrays["known_actual_ev_kw"] = known_ev
    observed_arrays["realized_u_kw"] = realized.u_kw
    score = policy_trial_scores(np.asarray(losses)[None], true_index,
        success=np.asarray(successes)[None], coverage_ok=True)
    result = dict(history_id=str(seed), seed=seed, policy_seed=selector_seed, policy=p, **identifiers,
        stress=stress, case=case, truth_topology=topology, true_index=true_index,
        candidate_names=[c.name for c in decisions[0].candidates],
        selections=[d.record for d in decisions], rounds=rounds, score_record=asdict(score),
        realized_plan=realized.as_dict(), operational=physical_check,
        selector_uses_truth=False, selector_uses_future_noise=False, gates_prune_candidates=False,
        truth=truth_metadata(path), independent_history_unit=True, mandatory_joint_fit_rounds=2,
        seconds=perf_counter()-started)
    if output is not None:
        out = Path(output); dump(out/f"rolling_{seed}.json", result)
        np.savez_compressed(out/f"rolling_{seed}_observations.npz", **observed_arrays)
    return result
