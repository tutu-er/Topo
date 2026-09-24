"""Theft-identification experiments (M5/M6) and loss sensitivity (M7).

Default: identified tree and weight bounds from the saved clean mainline fit.
M7 uses separate m7_* outputs and preserves the original null calibration.

Stages write JSON per run into outputs/theft/ so long runs can be resumed:

    python -m experiments.run_theft null       # clean-scenario gain calibration
    python -m experiments.run_theft exp2       # single experiment
    python -m experiments.run_theft report     # assemble the summary table

Experiments:
  1 normal            no theft (null member)
  2 internal          theft at hidden internal bus
  3 terminal          unmetered extra at a metered terminal
  4 switching         location changes mid-window
  5 ambiguous         theft on an unobserved spur branch (must NOT localize)
  6 biased            no theft, model uses 10% wrong impedance (must NOT alarm)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from theft_wzzt.theft.credibility import compare_h0_h1
from theft_wzzt.theft.loss_models import amplitude_envelope, estimate_loss
from theft_wzzt.theft.scan import scan_locations
from theft_wzzt.theft.identified_tree import (
    edge_weight_bounds,
    fit_identified_tree,
    load_identified,
    region_label,
    save_identified,
    to_theft_tree,
)
from theft_wzzt.theft.theft_model import (
    TheftData,
    data_from_scenario,
    fit_theft_milp,
    nominal_edge_weights,
    tree_from_network,
)
from theft_wzzt.theft.theft_simulation import (
    TheftSpec,
    add_unobserved_spur,
    simulate_theft_scenarios,
)

OUTPUT = PACKAGE_ROOT / "outputs" / "theft"
CASE = "paper15"
WINDOW = (44, 52)          # 8 time points, theft active in the middle for exps 2-5
AMP_KW = 8.0
KAPPA = 0.4
BOUNDS_FACTOR = (0.75, 1.25)
TIME_LIMIT = 240.0
NULL_REPLICATES = 20
NULL_REPLICATE_OFFSET = 10  # training/validation used replicates 0/1
TREE_MODE = "identified"    # "identified" = realistic chain; "true" = oracle
IDENTIFIED_CACHE = OUTPUT / "identified_tree.json"
BOUNDS_FACTOR_IDENTIFIED = (0.5, 2.0)
INTERNAL_BUS = 2
TERMINAL_BUS = 108
SPUR_PARENT = 5
SPUR_BUS = 900


def _window(data: TheftData) -> TheftData:
    a, b = WINDOW
    return TheftData(data.p[a:b], data.q[a:b], data.y[a:b], data.balance[a:b],
                     data.amplitude[a:b], data.extra_q[a:b],
                     data.voltage_scale, data.balance_scale)


def _event_amplitude(on_mask: np.ndarray) -> tuple[float, ...]:
    return tuple(AMP_KW * on_mask.astype(float))


def _event_mask(start: int, stop: int, n: int = 96) -> np.ndarray:
    mask = np.zeros(n)
    mask[start:stop] = 1.0
    return mask


def _scenario(exp: str, replicate: int):
    """Return (net, scenario, truth_locations, truth_mode) for one experiment id.

    truth_mode "all": every truth location must appear in the detected set.
    truth_mode "any": the truth is an equivalence class; hitting ANY member is
    correct localization (exp5: an unobserved spur and its parent share the
    exact same response column, so 900 and 5 are unresolvable by construction).
    """

    if exp in ("null", "exp1"):
        net, s = simulate_theft_scenarios(CASE, None, replicate=replicate)
        truth = []
    elif exp == "exp6":
        net, s = simulate_theft_scenarios(CASE, None, replicate=replicate,
                                          impedance_scale=1.1)
        truth = []
    elif exp == "exp2":
        net, s = simulate_theft_scenarios(
            CASE, (TheftSpec(INTERNAL_BUS, _event_amplitude(_event_mask(45, 51)), KAPPA),),
            replicate=replicate)
        truth = [INTERNAL_BUS]
    elif exp == "exp3":
        net, s = simulate_theft_scenarios(
            CASE, (TheftSpec(TERMINAL_BUS, _event_amplitude(_event_mask(45, 51)), KAPPA),),
            replicate=replicate)
        truth = [TERMINAL_BUS]
    elif exp == "exp4":
        first = _event_mask(45, 48)
        second = _event_mask(48, 51)
        net, s = simulate_theft_scenarios(
            CASE,
            (TheftSpec(INTERNAL_BUS, _event_amplitude(first), KAPPA),
             TheftSpec(TERMINAL_BUS, _event_amplitude(second), KAPPA)),
            replicate=replicate)
        truth = [INTERNAL_BUS, TERMINAL_BUS]
    elif exp == "exp5":
        net, s = simulate_theft_scenarios(
            CASE, (TheftSpec(SPUR_BUS, _event_amplitude(_event_mask(45, 51)), KAPPA),),
            replicate=replicate,
            case_modifier=lambda n: add_unobserved_spur(n, SPUR_PARENT, SPUR_BUS))
        truth = [SPUR_BUS, SPUR_PARENT]  # equivalence class: response columns identical
    else:
        raise ValueError(f"unknown experiment {exp}")
    mode = "any" if exp == "exp5" else "all"
    return net, s, truth, mode


def run_detection(exp: str, replicate: int, tree_mode: str = TREE_MODE, *,
                  loss_model: str | None = None, loss_bias: float = 0.0,
                  loss_ratio=None) -> dict:
    started = time.perf_counter()
    net, scenario, truth, mode = _scenario(exp, replicate)
    if tree_mode == "identified":
        # Realistic chain: tree and weight bands come from the mainline fit on
        # CLEAN data. exp6's impedance drift is then a physics-vs-model mismatch
        # the refit must absorb without a false alarm. Truth buses are mapped to
        # region labels on the identified tree for evaluation.
        if IDENTIFIED_CACHE.exists():
            identified = load_identified(IDENTIFIED_CACHE)
        else:
            identified = fit_identified_tree(CASE)
            save_identified(identified, IDENTIFIED_CACHE)
        tree, weights = to_theft_tree(identified)
        lo, hi = edge_weight_bounds(weights, BOUNDS_FACTOR_IDENTIFIED)
        truth = sorted({region_label(bus, net, identified) for bus in truth})
        scan_weights = weights
    elif tree_mode == "true":
        # Oracle chain: true tree, bounds around true edge weights.
        tree = tree_from_network(net)
        r_nom, x_nom = nominal_edge_weights(net)
        lo = BOUNDS_FACTOR[0] * np.minimum(r_nom, x_nom)
        hi = BOUNDS_FACTOR[1] * np.maximum(r_nom, x_nom)
        scan_weights = (r_nom, x_nom)
    else:
        raise ValueError(f"unknown tree_mode {tree_mode}")

    if loss_model is None:
        data = _window(data_from_scenario(scenario))
    else:
        loss_p, loss_q = estimate_loss(
            scenario, scan_weights, tree, model=loss_model, bias=loss_bias,
            seed=20260919 + replicate, ratio=loss_ratio)
        data = _window(data_from_scenario(
            scenario, loss_p_estimate=loss_p, loss_q_estimate=loss_q))

    record = {"experiment": exp, "replicate": replicate, "case": CASE,
              "tree_mode": tree_mode,
              "window": list(WINDOW), "truth_locations": truth, "truth_mode": mode,
              "voltage_scale": data.voltage_scale, "balance_scale": data.balance_scale,
              "amplitude_max_pu": float(data.amplitude.max())}

    comparison, h0, h1 = compare_h0_h1(tree, data, weight_bounds=(lo, hi),
                                       time_limit=TIME_LIMIT)
    record.update({
        "h0_loss": comparison["h0"]["loss"], "h1_loss": comparison["h1"]["loss"],
        "gain": comparison["gain"],
        "h1_scaled_voltage_mae": comparison["h1"]["voltage_scaled_mae"],
        "h1_scaled_balance_mae": comparison["h1"]["balance_scaled_mae"],
        "h1_seconds": comparison["h1_diagnostics"]["seconds"],
        "h1_mip_gap": comparison["h1_diagnostics"]["mip_gap"],
        "h1_binary_variables": comparison["h1_diagnostics"]["binary_variables"],
        "h1_locations": sorted({int(l) for l in h1.selected_locations(tree) if l is not None}),
    })

    # Baseline A: frozen-weight closed-form scan (same weights as the bounds center)
    scan = scan_locations(tree, data, scan_weights[0], scan_weights[1], kappa=KAPPA)
    order = np.argsort(-scan["gain"], axis=1)
    record["scan_top3"] = [
        [[int(tree.candidates[i]), float(scan["gain"][t, i])] for i in order[t, :3]]
        for t in range(len(data.p))
    ]

    # Baseline B: master-balance only (threshold fixed later by the null stage)
    record["balance_max"] = float(data.balance.max())
    record["balance_mean"] = float(data.balance.mean())

    if loss_model is not None:
        fixed = fit_theft_milp(tree, data, allow_theft=False, fixed_rx=scan_weights,
                               weight_bounds=(lo, hi), time_limit=TIME_LIMIT)
        record.update(_m7_metrics(scenario, data, loss_p, h0, h1, fixed, tree,
                                  truth, tree_mode))
        record.update(loss_model=loss_model, loss_bias=loss_bias,
                      loss_ratio=loss_ratio,
                      loss_q_estimate=loss_q.iloc[slice(*WINDOW)].tolist())

    record["seconds_total"] = time.perf_counter() - started
    return record


def _m7_metrics(scenario, data, loss_p, h0, h1, fixed, tree, truth, tree_mode):
    """Truth is read only AFTER fitting, exclusively for evaluation."""
    window = slice(*WINDOW)
    env = amplitude_envelope(scenario, loss_p).iloc[window]
    actual = sum((r["amplitude_pu"].to_numpy() for r in scenario["theft_truth"]),
                 np.zeros(len(scenario["P_terminal"])))[window]
    active = actual > 0
    delta = data.amplitude - actual
    selected = h1.selected_locations(tree)
    acceptable = set(truth)
    for label in truth:
        acceptable |= _ancestor_labels(label, tree_mode)
    observed = set(selected) - {None}
    reference_loss = scenario["loss_p_notheft"].iloc[window].to_numpy()
    actual_loss = scenario["loss_p_true"].iloc[window].to_numpy()
    meter_error = ((scenario["P0_measured"] - scenario["P0_true"])
                   - (scenario["P_terminal"] - scenario["P_true"]).sum(axis=1)
                   ).iloc[window].to_numpy()
    lp = loss_p.iloc[window].to_numpy()
    propagation = actual + actual_loss - lp + meter_error
    return {
        "h0_diagnostics": h0.diagnostics, "h1_diagnostics": h1.diagnostics,
        "h0_fixed_diagnostics": fixed.diagnostics,
        "h0_fixed_loss": fixed.objective,
        "h0_refit_benefit": fixed.objective - h0.objective,
        "h1_r": h1.r.tolist(), "h1_x": h1.x.tolist(),
        "h1_locations_by_time": selected,
        "location_covered": bool(acceptable & observed) if truth else None,
        "active_time_region_coverage": float(np.mean([
            selected[i] in acceptable for i in np.flatnonzero(active)])) if active.any() else None,
        "location_width": len(observed),
        "location_extra": sorted(observed - acceptable) if truth else [],
        "locations_readable": _readable(sorted(observed), tree_mode),
        "amplitude_truth_pu": actual.tolist(),
        "amplitude_mae_pu": float(np.abs(delta).mean()),
        "amplitude_bias_pu": float(delta.mean()),
        "amplitude_active_relative_mae": float(np.mean(np.abs(delta[active]) / actual[active]))
                                         if active.any() else None,
        "amplitude_inactive_max_pu": float(data.amplitude[~active].max()) if (~active).any() else None,
        "amplitude_envelope": env[["lower", "point", "upper"]].to_numpy().T.tolist(),
        "amplitude_envelope_order": ["lower", "point", "upper"],
        "amplitude_rel_halfwidth": [float(v) if np.isfinite(v) else None
                                    for v in env.relative_halfwidth],
        "amplitude_envelope_active_coverage": float(np.mean(
            (env.lower.to_numpy()[active] <= actual[active])
            & (actual[active] <= env.upper.to_numpy()[active]))) if active.any() else None,
        "envelope_beta": [-0.25, 0.25],
        "loss_p_estimate": lp.tolist(), "loss_p_notheft_eval": reference_loss.tolist(),
        "loss_p_actual_eval": actual_loss.tolist(), "meter_error_eval": meter_error.tolist(),
        "signed_balance": data.balance.tolist(),
        "loss_relative_bias_vs_notheft": float(np.sum(lp - reference_loss) / np.sum(reference_loss)),
        "loss_relative_mae_vs_notheft": float(np.sum(np.abs(lp - reference_loss)) / np.sum(reference_loss)),
        "balance_identity_max_error": float(np.max(np.abs(data.balance - propagation))),
    }


def run_null(replicates: int = NULL_REPLICATES, tree_mode: str = TREE_MODE) -> list[dict]:
    rows = []
    for rep in range(replicates):
        record = run_detection("null", replicate=NULL_REPLICATE_OFFSET + rep,
                               tree_mode=tree_mode)
        rows.append(record)
        _save(record, f"null_rep{rep}")
        print(f"null rep{rep}: gain={record['gain']:.3f} balance_max={record['balance_max']:.5f} "
              f"({record['seconds_total']:.0f}s)", flush=True)
    return rows


def _save(record: dict, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"{name}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


def _load(name: str) -> dict:
    return json.loads((OUTPUT / f"{name}.json").read_text(encoding="utf-8"))


def _ancestor_labels(label: int, tree_mode: str) -> set[int]:
    """Regions strictly coarser than ``label`` (ancestors); detecting one is
    region-level-consistent localization, not a false positive."""
    if tree_mode != "identified":
        return set()  # true-tree mode: keep strict semantics (bus labels)
    identified = load_identified(IDENTIFIED_CACHE)
    if label in identified.terminals:
        clade = frozenset({label})
    else:
        clade = identified.supports[-(label + 2)] if label < -1 else None
        if clade is None:
            return set()
    ancestors = set()
    for s in identified.supports:
        if len(s) > 1 and clade < s:
            ancestors.add(identified.label_of(s))
    return ancestors


def _readable(labels, tree_mode: str):
    if tree_mode != "identified":
        return [int(l) for l in labels]
    identified = load_identified(IDENTIFIED_CACHE)
    out = []
    for l in labels:
        if l in identified.terminals:
            out.append(int(l))
        elif l == -1:
            out.append("root")
        else:
            out.append("clade" + str(sorted(identified.supports[-(l + 2)])))
    return out


def run_report() -> dict:
    names = sorted(OUTPUT.glob("null_rep*.json"), key=lambda p: int(p.stem.split("rep")[1]))
    nulls = [json.loads(p.read_text(encoding="utf-8")) for p in names]
    null_gains = np.array([r["gain"] for r in nulls])
    null_balance = np.array([r["balance_max"] for r in nulls])
    gain_threshold = float(np.quantile(null_gains, 0.95))
    balance_threshold = float(np.quantile(null_balance, 0.95))
    rows = []
    for exp in ("exp1", "exp2", "exp3", "exp4", "exp5", "exp6"):
        r = _load(exp)
        rank = float((1 + np.count_nonzero(null_gains >= r["gain"] - 1e-7)) / (len(null_gains) + 1))
        truth = set(r["truth_locations"])
        locs = set(r["h1_locations"])
        # Localization is reported as set coverage/width, not a single point.
        # For no-theft experiments localization is not applicable: correctness
        # is "no alarm", and any tiny-amplitude selection is noise, not a claim.
        tree_mode = r.get("tree_mode", "true")
        acceptable = set()
        for label in truth:
            acceptable.add(label)
            acceptable |= _ancestor_labels(label, tree_mode)
        if not truth:
            covered = None
        elif r.get("truth_mode", "all") == "any":
            covered = bool(acceptable & locs)  # equivalence class: any member hit
        else:
            covered = all(t in locs or bool(_ancestor_labels(t, tree_mode) & locs)
                          for t in truth)
        extras = sorted(locs - acceptable)
        rows.append({
            "experiment": exp,
            "gain": r["gain"], "gain_null_q95": gain_threshold,
            "calibrated_rank": rank, "alarm": bool(r["gain"] > gain_threshold),
            "h1_locations": r["h1_locations"], "truth_locations": r["truth_locations"],
            "location_covered": covered,
            "location_extra": extras,
            "locations_readable": _readable(sorted(locs), tree_mode),
            "truth_readable": _readable(sorted(truth), tree_mode),
            "location_width": len(locs),
            "balance_alarm": r["balance_max"] > balance_threshold,
            "h1_seconds": r["h1_seconds"], "mip_gap": r["h1_mip_gap"],
        })
    summary = {"min_resolvable_alpha": 1.0 / (len(null_gains) + 1),
               "gain_threshold_q95": gain_threshold,
               "balance_threshold_q95": balance_threshold,
               "null_gains": null_gains.tolist(), "rows": rows}
    _save(summary, "summary")
    return summary


# M7 evaluation nulls are disjoint from the unchanged 10..29 calibration.
M7_NULL_REPS = tuple(range(30, 35))
M7_BIASES = (-0.5, -0.25, 0.0, 0.25, 0.5)
M7_SCHEMA = 1


def _m7_name(model, bias, exp, rep):
    tag = f"_{bias:+.2f}".replace("+", "p").replace("-", "m").replace(".", "") if model == "L2" else ""
    return f"m7_{model}{tag}_{exp}_rep{rep}"


def _m7_specs(stage):
    cases = [("null", rep) for rep in M7_NULL_REPS] + [("exp2", 100), ("exp3", 100)]
    if stage == "m7_l1":
        return ([(model, 0.0, exp, rep) for model in ("L0", "L1") for exp, rep in cases]
                + [(model, 0.0, exp, rep) for model in ("L3", "L4")
                   for exp, rep in (("null", 30), ("exp2", 100), ("exp3", 100))])
    if stage == "m7_bias":
        return [("L2", bias, exp, rep) for bias in M7_BIASES for exp, rep in cases]
    raise ValueError(stage)


def _m7_context():
    if not IDENTIFIED_CACHE.exists():
        raise FileNotFoundError("M7 requires the existing identified_tree.json; no mainline refit")
    null_paths = sorted(OUTPUT.glob("null_rep*.json"))
    nulls = [json.loads(p.read_text(encoding="utf-8")) for p in null_paths]
    if (len(nulls) != NULL_REPLICATES or {r["replicate"] for r in nulls} != set(range(10, 30))
            or any(r.get("tree_mode") != "identified" or r["case"] != CASE
                   or r["window"] != list(WINDOW) for r in nulls)):
        raise ValueError("M7 requires 20 existing identified paper15 nulls, replicates 10..29")
    baseline = _load("summary")
    threshold = float(np.quantile([r["gain"] for r in nulls], 0.95))
    if not np.isclose(threshold, baseline["gain_threshold_q95"], rtol=0, atol=1e-10):
        raise ValueError("Original summary and null calibration do not agree")
    paths = [*sorted((PACKAGE_ROOT / "theft_wzzt").rglob("*.py")), *sorted((PACKAGE_ROOT / "experiments").glob("*.py")),
             IDENTIFIED_CACHE, OUTPUT / "summary.json", *null_paths]
    hashes = {str(p.relative_to(PACKAGE_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths}
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"signature": signature, "sha256": hashes, "fixed_q95": threshold,
            "calibration_replicates": sorted(r["replicate"] for r in nulls),
            "evaluation_null_replicates": list(M7_NULL_REPS), "schema": M7_SCHEMA}


def _m7_run_one(spec, context):
    model, bias, exp, rep = spec
    name = _m7_name(*spec)
    path = OUTPUT / f"{name}.json"
    if path.exists():
        old = _load(name)
        if old.get("signature") != context["signature"] or old.get("schema") != M7_SCHEMA:
            raise ValueError(f"Stale M7 checkpoint {path}; preserve and review before reuse")
        if old.get("status") == "ok":
            print(f"resume {name}", flush=True)
            return old
    try:
        if model == "L2" and bias == 0:
            record = dict(_m7_run_one(("L0", 0.0, exp, rep), context))
            record.update(loss_model=model, loss_bias=bias,
                          reused_from=_m7_name("L0", 0.0, exp, rep))
        else:
            ratio = None
            if model == "L4":
                # Freeze the oracle ratio on a separate clean historical replicate.
                _, training, _, _ = _scenario("null", 10)
                denominator = np.maximum(training["P_terminal"].sum(axis=1), 0).sum()
                ratio = [float(v.sum() / denominator) for v in estimate_loss(training)]
            record = run_detection(exp, rep, "identified", loss_model=model,
                                   loss_bias=bias, loss_ratio=ratio)
            if model == "L4":
                record["ratio_calibration"] = "oracle L0, separate clean replicate 10, full day"
        record.update(status="ok", schema=M7_SCHEMA, signature=context["signature"])
    except Exception as exc:
        _save({"status": "failed", "schema": M7_SCHEMA, "signature": context["signature"],
               "spec": list(spec), "error": repr(exc)}, name)
        raise
    _save(record, name)
    print(f"{name}: gain={record['gain']:.6f}, time={record['seconds_total']:.1f}s", flush=True)
    return record


def run_m7(stage):
    context = _m7_context()
    _save(context, "m7_run_manifest")
    failures = []
    for spec in _m7_specs(stage):
        try:
            _m7_run_one(spec, context)
        except Exception as exc:
            failures.append({"spec": list(spec), "error": repr(exc)})
            print(f"FAILED {spec}: {exc}", flush=True)
    if failures:
        raise RuntimeError(f"M7 incomplete; {len(failures)} failed conditions: {failures}")


def run_m7_report():
    context = _m7_context()
    records, missing = [], []
    for spec in _m7_specs("m7_l1") + _m7_specs("m7_bias"):
        name = _m7_name(*spec)
        if not (OUTPUT / f"{name}.json").exists():
            missing.append(name)
            continue
        row = _load(name)
        if (row.get("status") != "ok" or row.get("signature") != context["signature"]
                or row.get("schema") != M7_SCHEMA):
            missing.append(name)
            continue
        records.append(dict(row, checkpoint=name))
    fixed = context["fixed_q95"]
    baselines = {(r["experiment"], r["replicate"]): r for r in records if r["loss_model"] == "L0"}
    for row in records:
        row.update(alarm_fixed=bool(row["gain"] > fixed), gain_margin_fixed=row["gain"] - fixed)
        base = baselines.get((row["experiment"], row["replicate"]))
        if base is not None:
            row["h0_refit_benefit_over_L0_h0_loss"] = row["h0_refit_benefit"] / base["h0_loss"]
            row["h0_refit_benefit_change_from_L0"] = row["h0_refit_benefit"] - base["h0_refit_benefit"]
            row["h1_r_shift_l2_from_L0"] = float(np.linalg.norm(np.array(row["h1_r"]) - base["h1_r"]))
            row["h1_x_shift_l2_from_L0"] = float(np.linalg.norm(np.array(row["h1_x"]) - base["h1_x"]))
    groups = []
    for model, bias in [(m, 0.0) for m in ("L0", "L1", "L3", "L4")] + [("L2", b) for b in M7_BIASES]:
        group = [r for r in records if r["loss_model"] == model and r["loss_bias"] == bias]
        null = [r for r in group if r["experiment"] == "null"]
        ng = np.array([r["gain"] for r in null])
        q95 = float(np.quantile(ng, .95)) if len(ng) >= 5 else None
        for row in group:
            row["variant_q95_exploratory"] = q95
            row["alarm_variant_q95"] = bool(row["gain"] > q95) if q95 is not None else None
        events = [r for r in group if r["experiment"] != "null"]
        groups.append({
            "loss_model": model, "loss_bias": bias, "null_n": len(ng),
            "null_gains": ng.tolist(), "false_alarms_fixed": int(np.count_nonzero(ng > fixed)),
            "false_positive_fraction_fixed": float(np.mean(ng > fixed)) if len(ng) else None,
            "variant_q95_exploratory": q95,
            "variant_q95_resubstitution_fpr": float(np.mean(ng > q95)) if q95 is not None else None,
            "min_resolvable_variant_rank": 1 / (len(ng) + 1) if len(ng) else None,
            "events": [{k: r[k] for k in ("experiment", "gain", "alarm_fixed", "alarm_variant_q95",
                         "location_covered", "active_time_region_coverage", "amplitude_active_relative_mae",
                         "location_width", "gain_margin_fixed")} for r in events],
        })
    summary = {
        "status": "complete" if not missing else "incomplete", "missing_or_failed": missing,
        "fixed_q95": fixed, "context": context,
        "baseline_summary": _load("summary"), "groups": groups, "rows": records,
        "notes": [
            "Fixed alarm rule is gain > original empirical q95, matching M5/M6 run_report.",
            "Null 30..34 is held out from original calibration 10..29; 5 samples are exploratory.",
            "Variant q95 and resubstitution FPR use the SAME 5 nulls; not independent calibrated FPR.",
            "Variant null rank resolution is 1/6, insufficient for a 5% rank test.",
            "The envelope is conditional on a declared loss band, excludes meter error; not a confidence interval.",
            "L1-L0 includes approximation, fitted tree/weights and measurement error; not isolated parameter error.",
            "L0/L2/L3 are oracle controls; L4 uses a ratio calibrated from separate oracle clean data.",
            "H0 refit benefit is voltage-fit flexibility, independent of external loss; not loss-error absorption.",
            "L2 perturbs both P and Q losses with the same relative bias; L3 shares one random factor.",
            "Regional ancestor coverage does not establish unique physical-node or household localization.",
            "Each exp2/exp3 has one replicate; no population detection probability or safe error bound is established.",
        ],
    }
    summary["validation"] = {
        "completed_rows": len(records), "expected_rows": 55,
        "unique_fits": sum("reused_from" not in r for r in records),
        "max_mip_gap": max((r["h1_mip_gap"] for r in records), default=None),
        "max_feasibility_error": max((r[d]["max_feasibility_error"] for r in records
                                      for d in ("h0_diagnostics", "h1_diagnostics", "h0_fixed_diagnostics")), default=None),
        "max_product_error": max((r[d]["max_product_error"] for r in records
                                  for d in ("h0_diagnostics", "h1_diagnostics", "h0_fixed_diagnostics")), default=None),
        "max_balance_identity_error": max((r["balance_identity_max_error"] for r in records), default=None),
        "max_abs_h0_refit_benefit_change": max((abs(r["h0_refit_benefit_change_from_L0"])
                                               for r in records if "h0_refit_benefit_change_from_L0" in r), default=None),
    }
    _save(summary, "m7_summary")
    _m7_write_markdown(summary)
    return summary


def _m7_write_markdown(summary):
    lines = ["# M7 幅值误差与线损敏感性结果", "", f"状态：{summary['status']}；固定阈值：{summary['fixed_q95']:.9f}。",
             "", "null 使用独立复制 30–34；exp2/exp3 各使用复制 100，窗口为 44:52。",
             "L2 同时扰动 P/Q 线损。告警沿用原汇总的 gain > q95 规则。", "",
             "|估计器|偏置|固定阈值误报数|变体 q95（探索性）|exp2 gain / 告警 / 区域覆盖|exp3 gain / 告警 / 区域覆盖|",
             "|---|---:|---:|---:|---|---|"]
    for group in summary["groups"]:
        cells = []
        for exp in ("exp2", "exp3"):
            r = next((r for r in group["events"] if r["experiment"] == exp), None)
            cells.append("未完成" if r is None else f"{r['gain']:.3f} / {r['alarm_fixed']} / {r['location_covered']}")
        q = group["variant_q95_exploratory"]
        lines.append(f"|{group['loss_model']}|{group['loss_bias']:+.2f}|{group['false_alarms_fixed']}/{group['null_n']}|"
                     + ("—" if q is None else f"{q:.3f}") + "|" + "|".join(cells) + "|")
    lines += ["", "## 幅值与定位细节", "",
              "|估计器|偏置|事件|活动时段幅值相对 MAE|活动时段区域覆盖|±25% 包络真值覆盖|定位集合宽度|",
              "|---|---:|---|---:|---:|---:|---:|"]
    for row in summary["rows"]:
        if row["experiment"] != "null":
            lines.append(f"|{row['loss_model']}|{row['loss_bias']:+.2f}|{row['experiment']}|"
                         f"{row['amplitude_active_relative_mae']:.2%}|{row['active_time_region_coverage']:.2%}|"
                         f"{row['amplitude_envelope_active_coverage']:.2%}|{row['location_width']}|")
    lines += ["", "## 解释边界", "",
              "- 5 个 null 仅作探索；变体 q95 的回代误报率使用同一批样本，不是独立验证。",
              "- 包络仅传播声明的线损误差带，未覆盖表计噪声和全部偷电增量线损，不是置信区间。",
              "- L1 与 L0 的差含树/权重误差、潮流近似和量测误差，不能单独归因为参数误差。",
              "- L0/L2/L3 使用反事实真值；L4 比例从独立清洁复制 10 的 L0 标定，仅是理想对照。",
              "- H0 中线损仅改变与 R/X 无关的平衡损失；固定/重估差是电压拟合收益，不能称线损误差吸收率。",
              "- 区域覆盖允许祖先区域，不意味着逐户唯一定位；exp2/exp3 各一例不支持总体检出率结论。",
              "", "## 验证", "", "```json", json.dumps(summary["validation"], ensure_ascii=False, indent=2), "```", "",
              "逐时包络、幅值、误差分解、求解诊断和文件哈希见同目录 m7_summary.json 与 m7_run_manifest.json。"]
    if summary["missing_or_failed"]:
        lines += ["", "未完成：" + ", ".join(summary["missing_or_failed"])]
    (OUTPUT / "m7_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["null", "report", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6",
                                              "m7_l1", "m7_bias", "m7_report"])
    parser.add_argument("--replicates", type=int, default=NULL_REPLICATES)
    parser.add_argument("--tree", choices=["identified", "true"], default=TREE_MODE)
    args = parser.parse_args()
    if args.stage.startswith("m7_"):
        if args.tree != "identified":
            parser.error("M7 currently requires --tree identified")
        if args.stage == "m7_report":
            summary = run_m7_report()
            print(json.dumps(summary["validation"], indent=2), flush=True)
            if summary["status"] != "complete":
                raise SystemExit("M7 report incomplete; inspect missing_or_failed")
        else:
            run_m7(args.stage)
    elif args.stage == "null":
        run_null(args.replicates, tree_mode=args.tree)
    elif args.stage == "report":
        summary = run_report()
        for row in summary["rows"]:
            print(json.dumps(row), flush=True)
    else:
        record = run_detection(args.stage, replicate=100, tree_mode=args.tree)
        _save(record, args.stage)
        print(json.dumps({k: v for k, v in record.items() if k != "scan_top3"}, indent=1))


if __name__ == "__main__":
    main()
