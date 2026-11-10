"""Prepare a node-restricted case and inspect frozen coalition quotes.

The original pricing solver is reused unchanged. All case arrays, execution
contracts and detailed audits stay in the caller's private output directory.
Run from this file with --help; this script contains no grid or truth loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))
from problem import load_problem
from solve import reference, write_arrays, write_json


def inputs(case, quote, adapter):
    problem = load_problem(case)
    with np.load(quote, allow_pickle=False) as data:
        x = data["x"].copy()
    with np.load(adapter, allow_pickle=False) as data:
        nodes = np.asarray(data["resource_node_index"], int)
    labels = np.asarray(problem.node_labels)[nodes]
    if x.shape != problem.shape or labels.shape != (problem.device.R,):
        raise ValueError("Quote or physical adapter does not match this case")
    return problem, x, labels


def prepare(args):
    problem, old, labels = inputs(args.case, args.quote, args.adapter)
    protocol = json.loads((Path(__file__).with_name("node_scope_protocol.json")).read_text())
    keep = np.isin(labels, protocol["allowed_nodes"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "prepared.json").exists():
        raise ValueError("Preparation already recorded; do not overwrite")
    with np.load(problem.arrays_path, allow_pickle=False) as data:
        original = {name:data[name].copy() for name in data.files}
    keys_r_first = {"resource_lower_kw", "resource_upper_kw", "resource_cost",
        "resource_efficiency", "resource_energy_lower_kwh", "resource_energy_upper_kwh",
        "resource_reset", "resource_terminal", "resource_future_lower_kwh",
        "resource_future_upper_kwh"}
    restricted = {}
    for key, value in original.items():
        if key in keys_r_first:
            restricted[key] = value[keep]
        elif key in ("effect_coarse", "effect_fine"):
            restricted[key] = value[:, keep]
        elif key != "warm_requests":
            restricted[key] = value
    for start in protocol["optimization"]["starts"]:
        folder = out / ("case_" + start)
        folder.mkdir(exist_ok=False)
        config = json.loads(json.dumps(problem.config))
        config.update(case_id=problem.config["case_id"] + "_101_105_" + start,
            run_purpose="user_requested_node_scope_ablation", approved_for_new_optimization=True,
            resource_ids=[r for r, allowed in zip(problem.resource_ids, keep) if allowed],
            solver=protocol["optimization"]["per_start"])
        config["scope_ablation"] = dict(allowed_nodes=protocol["allowed_nodes"],
            weight_reason=protocol["fixed"]["weight_reason"], no_capacity_redistribution=True)
        restricted["warm_requests"] = old[keep] if start == "projected_old_AB" else np.zeros_like(old[keep])
        write_json(folder / "case.json", config)
        write_arrays(folder / "arrays.npz", **restricted)
        check = load_problem(folder)
        if check.threshold != problem.threshold:
            raise AssertionError("Node restriction changed the selection threshold")
    only_a, only_b = old.copy(), old.copy()
    only_a[~keep], only_b[keep] = 0., 0.
    contracts = {}
    for name, x in (("old_AB", old), ("old_A_only", only_a),
                    ("old_B_only", only_b), ("zero", np.zeros_like(old))):
        point = problem.nominal(problem.requests_to_values(x))
        contracts[name] = point["mean_u"]
    write_arrays(out / "frozen_counterfactuals.npz", **contracts)
    write_arrays(out / "resource_mapping.npz", keep=keep, resource_labels=labels)
    write_json(out / "prepared.json", dict(protocol=protocol,
        sources={p.name:reference(p) for p in [Path(__file__), Path(__file__).with_name("node_scope_protocol.json"),
            CORE/"problem.py", CORE/"probability.py", CORE/"solve.py"]},
        inputs={"case":reference(problem.case_path), "arrays":reference(problem.arrays_path),
            "quote":reference(args.quote), "adapter":reference(args.adapter)},
        retained_resources=int(keep.sum()), retained_active_requests=int(check.active.size),
        original_resources=problem.device.R, original_active_requests=int(problem.active.size),
        threshold=problem.threshold, new_day_weight=problem.new_weight))


def folded_mean(a, sigma):
    ratio = np.divide(a, sigma, out=np.zeros_like(a), where=sigma>0)
    result = sigma*np.sqrt(2/np.pi)*np.exp(-.5*ratio*ratio) + a*(2*ndtr(ratio)-1)
    return np.where(sigma>0, result, np.abs(a))


def summarize(problem, x):
    values = problem.requests_to_values(x)
    point = problem.nominal(values)
    a_c, a_f, sd = point["a_C"], point["a_F"], problem.sigma.ravel()
    lc, lf = float(folded_mean(a_c, sd).mean()), float(folded_mean(a_f, sd).mean())
    gap = lc-lf
    band = -(problem.kappa-1)*lf
    return dict(expected_LC=lc, expected_LF=lf, expected_unweighted_gap=gap,
        band_component=band, expected_D=gap+band,
        candidate_prediction_distance=float(np.abs(a_c-a_f).mean()),
        fee=point["fee"], device=problem.audit_delivery(point["mean_u"]),
        probability=problem.audit_probability(values),
        max_abs_effect_difference=float(np.max(np.abs(problem.effect["C"]-problem.effect["F"]))))


def analyze(args):
    problem, old, labels = inputs(args.case, args.quote, args.adapter)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"coalitions.json").exists():
        raise ValueError("Completed coalition analysis already exists")
    keep = np.isin(labels, [101,102,103,104,105])
    a, b = old.copy(), old.copy()
    a[~keep], b[keep] = 0., 0.
    results = {}
    for name, x in (("zero", np.zeros_like(old)), ("A", a), ("B", b), ("AB", old)):
        results[name] = summarize(problem, x)
        write_json(out/"coalitions_partial.json", results)
    for node in sorted(set(labels[~keep])):
        x = old.copy()
        x[labels == node] = 0.
        results["without_"+str(node)] = summarize(problem, x)
        write_json(out/"coalitions_partial.json", results)
    write_json(out/"coalitions.json", dict(results=results,
        source=reference(Path(__file__)), case=reference(problem.case_path),
        arrays=reference(problem.arrays_path), quote=reference(args.quote),
        meaning="Saved quote ablations only; parameters and prices of retained resources are unchanged"))


def freeze(args):
    original, _, labels = inputs(args.case, args.quote, args.adapter)
    out = Path(args.out)
    if (out/"policy_freeze.json").exists():
        raise ValueError("Policy already frozen")
    keep = np.isin(labels, [101,102,103,104,105])
    rows = []
    for name in ("projected_old_AB", "zero"):
        folder = out / ("optimization_"+name)
        row = json.loads((folder/"summary.json").read_text())
        rows.append((name,row,folder/"quote.npz"))
    qualified = [r for r in rows if r[1]["qualified"]]
    selected = min(qualified, key=lambda r:r[1]["fee"]) if qualified else max(rows,key=lambda r:r[1]["probability"])
    name, summary, path = selected
    with np.load(path,allow_pickle=False) as data:
        x = np.zeros(original.shape)
        x[keep] = data["x"]
        expected_u = data["mean_u"].copy()
    point = original.nominal(original.requests_to_values(x))
    if not np.allclose(point["mean_u"][keep], expected_u, atol=1e-12, rtol=0):
        raise AssertionError("Restricted device map does not lift to original resources")
    if np.max(np.abs(point["mean_u"][~keep])) != 0:
        raise AssertionError("Inactive-node delivery is not zero")
    with np.load(out/"frozen_counterfactuals.npz",allow_pickle=False) as data:
        contracts = {key:data[key].copy() for key in data.files}
    contracts["new_A"] = point["mean_u"]
    write_arrays(out/"frozen_execution_contracts.npz", **contracts)
    write_arrays(out/"new_A_quote.npz", x=x, mean_u=point["mean_u"],
        unit_reward=original.unit_reward(point["mean_u"]))
    write_json(out/"policy_freeze.json", dict(selected_start=name, selected_summary=summary,
        starts={n:s for n,s,_ in rows}, contracts=reference(out/"frozen_execution_contracts.npz"),
        quote=reference(out/"new_A_quote.npz"),
        no_actual_future_or_truth_access=True, new_actual_execution_started=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare","analyze","freeze"])
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--quote", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"prepare":prepare,"analyze":analyze,"freeze":freeze}[args.action](args)
