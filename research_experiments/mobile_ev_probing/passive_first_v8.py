"""Passive-first finite-candidate controller, using unchanged v6/v7 primitives.

This is an operational composition of existing certificates/designs, not an AC
solver or a new confidence theory. Truth enters only simulation/evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy

import design_v7 as d
import local_shared_source_v6 as v6

HISTORY_MODES = ("flat", "single_node", "rich")
SEEDS = (8101, 8102)


def classify_fit(result):
    """Exclusion needs a valid positive lower bound; solver failure is retained."""
    lb = result.get("reliable_lower_bound")
    ub = result.get("verified_upper_bound")
    if result.get("exact_box_corrected_dual_verified") and lb is not None and lb > 0:
        return "excluded"
    if result.get("exact_primal_verified") and ub == 0:
        return "compatible_verified"
    return "unresolved"


def screen_history(history, candidates=d.NAMES, box=d.BOX, solver=None):
    """Only declared historical observations are visible to this API."""
    solve = v6.solve_lp if solver is None else solver
    records = {}
    for h in candidates:
        try:
            fit = solve(history, h, parameter_box=box)
        except (RuntimeError, ValueError) as exc:
            fit = dict(status="exception_unresolved", message=str(exc),
                       reliable_lower_bound=None, verified_upper_bound=None)
        records[h] = dict(classification=classify_fit(fit), result=fit)
    retained = [h for h in candidates if records[h]["classification"] != "excluded"]
    if len(retained) == 1 and records[retained[0]]["classification"] != "compatible_verified":
        # LP primal repair may leave an arbitrarily small positive tau. Do not
        # round it to zero: establish exact zero-slack feasibility in the
        # algebraically equivalent projected historical domain instead.
        h = retained[0]
        poly = d.build_history_polytope(history, h, box)
        witness = d.solve_qp(poly, d.midpoint_objective())
        records[h]["feasibility_witness"] = witness
        if witness.get("exact_feasible_verified") and poly.exact_feasible(witness["x"]):
            records[h]["classification"] = "compatible_verified"
    if not retained:
        state = "history_inconsistent_with_declared_family"
    elif len(retained) == 1:
        state = ("singleton_verified_skip_ev" if records[retained[0]]["classification"] == "compatible_verified"
                 else "singleton_unverified_numerically_unresolved")
    else:
        state = "additional_evidence_needed"
    return dict(state=state, retained=retained, records=records,
                conditional_scope="declared finite family and linear/source/error/parameter domains",
                is_topology_probability=False)


def design_retained(history, retained, background=None, box=d.BOX):
    """Historical local/committee T scores restricted to retained candidates.

No numerical failure removes a candidate. If any historical domain cannot
produce an exactly feasible representative or any score lacks a certificate,
the action is unresolved, rather than silently redesigning a smaller bank.
    """
    if len(retained) < 2:
        return dict(state="design_not_applicable", reason="fewer than two retained candidates")
    polys = {h: d.build_history_polytope(history, h, box) for h in retained}
    reps = {h: dict(candidate=h, **d.solve_qp(p, d.midpoint_objective())) for h, p in polys.items()}
    if any(r.get("lower") is None for r in reps.values()):
        return dict(state="design_numerically_unresolved", representatives=reps,
                    retained=list(retained), reason="missing exact feasible historical representative")
    best = min(r["upper"] for r in reps.values())
    possible = [h for h in retained if reps[h]["lower"] <= best + d.REFERENCE_TIE]
    local = dict(selected=possible[0], possible=possible, unique_resolved=len(possible) == 1,
                 tolerance=d.REFERENCE_TIE)
    records, scores = [], {}
    for r, ref in reps.items():
        scores[r] = {}
        for site in range(3):
            name = f"site{site + 1}_endpoint"
            p, _ = d.action_inputs(site, "endpoint", background)
            fits = []
            for h, poly in polys.items():
                if h == r:
                    continue
                fit = d.solve_qp(poly, d.design_objective(poly, ref, p), reps[h]["x"], scale=1e4)
                records.append(dict(reference=r, candidate=h, action=name, result=fit))
                fits.append(fit)
            if any(q.get("lower") is None for q in fits):
                return dict(state="design_numerically_unresolved", representatives=reps,
                            retained=list(retained), qp_records=records,
                            reason="missing certified score; no candidate removed")
            scores[r][name] = dict(lower=min(q["lower"] for q in fits), upper=min(q["upper"] for q in fits))
    committee = {
        name: dict(lower=v6._down(sum((d.f(scores[r][name]["lower"]) for r in retained), F()) / len(retained)),
                   upper=v6._up(sum((d.f(scores[r][name]["upper"]) for r in retained), F()) / len(retained)))
        for name in scores[retained[0]]
    }
    return dict(state="probe_selected", retained=list(retained), representatives=reps,
                reference_selection=local, scores_by_reference=scores, qp_records=records,
                committee_scores=committee, committee_selection=d.rank_intervals(committee),
                local_selection=d.rank_intervals(scores[local["selected"]]),
                committee_semantics="equal design weights; not topology probabilities")


def plan(history, candidates=d.NAMES, background=None, box=d.BOX, solver=None):
    began = perf_counter()
    screening = screen_history(history, candidates, box, solver)
    out = dict(screening=screening, state=screening["state"], selected_action=None,
               selection_inputs="historical observations, declared box, candidate bank, future background P only")
    if screening["state"] == "additional_evidence_needed":
        design = design_retained(history, screening["retained"], background, box)
        out["design"] = design
        out["state"] = design["state"]
        if design["state"] == "probe_selected":
            out["selected_action"] = design["committee_selection"]["selected"]
    out["seconds"] = perf_counter() - began
    return out


def history_inputs(mode):
    p = np.full((12, 3), .2)
    if mode == "single_node":
        p[:, 0] += np.tile([0., .02, .02, 0.], 3)
    elif mode == "rich":
        p += np.tile([[.02, 0., 0.], [0., .02, 0.], [0., 0., .02], [0., 0., 0.]], (3, 1))
    elif mode != "flat":
        raise ValueError(mode)
    return p


def simulate(topology, theta, seed, mode, site=0, policy="none", epsilon=1e-5):
    """Exactly paired source/error draws across future charging policies."""
    rng = np.random.default_rng(seed)
    if policy == "none":
        newp, ev = np.full((24, 3), .2), np.zeros(24)
    else:
        newp, ev = d.action_inputs(site, policy)
    p = np.r_[history_inputs(mode), newp]
    delta = rng.uniform(-.02, .02, 36)
    noise = rng.uniform(-.8 * epsilon, .8 * epsilon, (36, 3))
    exact = [[sum((c*d.f(v) for c, v in zip(d.prediction_coeff(pt, .12, topology, i), theta)), F())
              + d.f(theta[0])*d.f(delta[t]) for i in range(3)] for t, pt in enumerate(p)]
    y0 = np.array([[float(v) for v in row] for row in exact])
    y = y0 + noise
    error = np.array([[float(d.f(y[t, i])-exact[t][i]) for i in range(3)] for t in range(36)])
    obs = dict(P_total=p, Y_drop=y, source_nominal=np.full(36, .12),
               source_error_bound=np.full(36, .1), drop_error_bound=np.full((36, 3), epsilon),
               batch_id=np.r_[np.zeros(12, int), np.ones(24, int)])
    soc = .4 + np.r_[0., np.cumsum(ev * 10 * .25 * .95)] / 50
    truth = dict(topology=topology, theta=list(theta), seed=seed, history_mode=mode,
                 source_delta=delta.tolist(), noise_draw=noise.tolist(), actual_error=error.tolist(), Y_true=y0.tolist(),
                 ev_pu=ev.tolist(), policy=policy, site=site + 1,
                 energy_kwh=float(ev.sum() * 10 * .25), soc_range=[float(soc.min()), float(soc.max())],
                 checks=dict(source=bool(np.max(abs(delta)) <= .1), noise=bool(np.max(abs(error)) <= epsilon),
                             coefficient_box=all(l <= x <= u for x, (l, u) in zip(theta, d.BOX)),
                             ev=bool(np.max(abs(ev)) <= .3 and soc.min() >= .2 and soc.max() <= .9)))
    return obs, truth


def evaluate_case(topology, profile, seed, mode, output):
    name = f"passive_{topology}_{profile}_{mode}_{seed}"
    obs0, _ = simulate(topology, d.PROFILES[profile], seed, mode)
    history = d.observation_slice(obs0, slice(0, 12))
    hp = output / f"{name}_history.npz"
    np.savez_compressed(hp, **history)
    decision = plan(history)
    records, inputs, stats = [], [], {}
    planned = decision.get("selected_action")
    site = int(planned[4]) - 1 if planned else 0
    # Resolved cases actually use no EV. Forced-charge comparators would add
    # no identification value; skip them and record zero deployed energy.
    policies = ("none", "constant", "endpoint") if planned else ("none",)
    for policy in policies:
        obs, truth = simulate(topology, d.PROFILES[profile], seed, mode, site, policy)
        op, tp = output / f"{name}_{policy}_observations.npz", output / f"{name}_{policy}_truth.json"
        np.savez_compressed(op, **obs)
        d.dump(tp, truth)
        inputs.append(dict(policy=policy, observations=op.name, observations_hash=d.sha(op),
                           truth=tp.name, truth_hash=d.sha(tp), checks=truth["checks"]))
        fits = {h: v6.solve_lp(obs, h, parameter_box=d.BOX) for h in d.NAMES}
        for h, fit in fits.items():
            records.append(dict(policy=policy, candidate=h, result=fit))
        wrong = [r for h, r in fits.items() if h != topology]
        stats[policy] = dict(excluded_wrong=sum(classify_fit(r) == "excluded" for r in wrong),
                            nearest_wrong_lower=min(r["reliable_lower_bound"] for r in wrong),
                            nearest_wrong_upper=min(r["verified_upper_bound"] for r in wrong),
                            truth_excluded=classify_fit(fits[topology]) == "excluded",
                            energy_kwh=truth["energy_kwh"])
    before = decision["screening"]
    return dict(name=name, topology=topology, profile=profile, seed=seed, history_mode=mode,
                history_input=dict(path=hp.name, sha256=d.sha(hp)), decision=decision,
                evaluation=records, inputs=inputs,
                summary=dict(historical_wrong_excluded=sum(h != topology and r["classification"] == "excluded"
                             for h, r in before["records"].items()),
                             historical_truth_retained=topology in before["retained"], policies=stats,
                             local_reference_correct=(decision.get("design", {}).get("reference_selection", {}).get("selected") == topology)
                             if planned else None,
                             deployed_energy_kwh=stats["endpoint"]["energy_kwh"] if planned else 0.))


def aggregate(episodes):
    groups = {}
    for mode in HISTORY_MODES:
        for profile in d.PROFILES:
            es = [e for e in episodes if e["history_mode"] == mode and e["profile"] == profile]
            groups[f"{mode}/{profile}"] = summarize(es)
    return dict(overall=summarize(episodes), groups=groups)


def summarize(episodes):
    probe = [e for e in episodes if e["decision"]["selected_action"]]
    allfits = [r["result"] for e in episodes for r in e["evaluation"]]
    allfits += [r["result"] for e in episodes for r in e["decision"]["screening"]["records"].values()]
    qps = [r["result"] for e in probe for r in e["decision"]["design"]["qp_records"]]
    refs = [r for e in probe for r in e["decision"]["design"]["representatives"].values()]
    feasibility = [r["feasibility_witness"] for e in episodes for r in e["decision"]["screening"]["records"].values()
                   if "feasibility_witness" in r]
    return dict(episodes=len(episodes), history_singletons=sum(e["decision"]["state"] == "singleton_verified_skip_ev" for e in episodes),
                probe_count=len(probe), numerical_unresolved=sum(e["decision"]["state"] == "design_numerically_unresolved" for e in episodes),
                history_inconsistent=sum(e["decision"]["state"] == "history_inconsistent_with_declared_family" for e in episodes),
                no_ev_final_singletons=sum(e["summary"]["policies"]["none"]["excluded_wrong"] == 2 for e in episodes),
                deployed_policy_final_singletons=sum(e["summary"]["policies"]["endpoint" if e["decision"]["selected_action"] else "none"]["excluded_wrong"] == 2 for e in episodes),
                probing_constant_singletons=sum(e["summary"]["policies"]["constant"]["excluded_wrong"] == 2 for e in probe),
                probing_endpoint_singletons=sum(e["summary"]["policies"]["endpoint"]["excluded_wrong"] == 2 for e in probe),
                probing_mean_constant_margin=float(np.mean([e["summary"]["policies"]["constant"]["nearest_wrong_lower"] for e in probe])) if probe else None,
                probing_mean_endpoint_margin=float(np.mean([e["summary"]["policies"]["endpoint"]["nearest_wrong_lower"] for e in probe])) if probe else None,
                all_truths_retained=all(e["summary"]["historical_truth_retained"] and not any(s["truth_excluded"] for s in e["summary"]["policies"].values()) for e in episodes),
                wrong_local_references=sum(e["summary"]["local_reference_correct"] is False for e in probe),
                selection_seconds_total=sum(e["decision"]["seconds"] for e in episodes),
                selection_seconds_max=max((e["decision"]["seconds"] for e in episodes), default=0),
                energy_kwh=sum(e["summary"]["deployed_energy_kwh"] for e in episodes),
                avoided_energy_vs_always_ev_kwh=9 * (len(episodes)-len(probe)),
                lp_count=len(allfits), all_lps_certified=all(r.get("exact_primal_verified") and r.get("exact_box_corrected_dual_verified") for r in allfits),
                max_lp_gap=max((r.get("exact_gap", float("inf")) for r in allfits), default=0),
                qp_count=len(qps), reference_qp_count=len(refs),
                singleton_feasibility_qp_count=len(feasibility),
                max_qp_gap=max((r.get("gap", float("inf")) for r in qps), default=0),
                max_reference_qp_gap=max((r.get("gap", float("inf")) for r in refs), default=0))


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "passive_frozen_protocol.json"
    if protocol_path.exists():
        raise RuntimeError("refusing to overwrite frozen passive-first experiment")
    protocol = dict(version=8, created=datetime.now(timezone.utc).isoformat(),
                    central_claim="certified passive screening can avoid unnecessary EV and target remaining ambiguity",
                    model="unchanged v7 linear P-only known common source; no AC or PQ measurement errors",
                    histories=HISTORY_MODES, seeds=SEEDS, profiles=d.PROFILES, box=d.BOX,
                    history_samples=12, future_samples=24, time_step_hours=.25, epsilon=1e-5,
                    history_variation_pu=.02, actual_source_uniform=[-.02, .02], source_budget=.1,
                    future_policies=["none", "same-site constant 1.5 kW", "committee-site endpoint 0/3 kW"],
                    energy_kwh=9, site_selection="equal committee T over retained history-compatible candidates",
                    source_hash=d.sha(__file__), v7_hash=d.sha(d.__file__), v6_hash=d.sha(v6.__file__),
                    python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    interpretation="descriptive matched synthetic grid; two seeds are not population confidence intervals")
    d.dump(protocol_path, protocol)
    cases = [(h, p, s, m) for h in d.NAMES for p in d.PROFILES for s in SEEDS for m in HISTORY_MODES]
    episodes = []
    for i, case in enumerate(cases):
        ep = evaluate_case(*case, output)
        episodes.append(ep)
        print(i + 1, len(cases), ep["name"], ep["decision"]["state"], ep["decision"]["selected_action"], flush=True)
    report = dict(protocol=protocol, episodes=episodes, summary=aggregate(episodes),
                  source_hash_unchanged=d.sha(__file__) == protocol["source_hash"])
    d.dump(output / "passive_results.json", report)
    d.dump(output / "passive_summary.json", report["summary"])
    print(report["summary"]["overall"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
