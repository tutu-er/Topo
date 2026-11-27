"""Compare candidate pools by exhaustive tree enumeration on four synthetic feeder inputs.

From topology_study: python -B -m ptid.runs.pool_audit --out results.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
from time import perf_counter

from ptid.method.identify import Settings, identify
from ptid.method.model import BOUND, balance
from ptid.method.sets import basis, canonical, compatible

import gurobipy as gp
import numpy as np
import scipy
from scipy.optimize import lsq_linear


SEED, HOURS, METERS = 20261127, 96, 4
EPS = 1e-6                     # comparisons in H units
SETTINGS = Settings(short_seconds=10, long_seconds=10, node_seconds=10,
                    start_seconds=2, evidence=False)


def readings(excitation, hidden):
    """Matched synthetic linear-model observations. Ground truth is returned separately."""
    rng = np.random.default_rng(SEED)
    day = np.sin(2*np.pi*np.arange(HOURS)/24)[:, None]
    p = .04 + .006*day + excitation*.004*rng.normal(size=(HOURS, METERS))
    q = .3*p + excitation*.0015*rng.normal(size=p.shape)
    up = .008 + .001*day[:, 0] + excitation*.001*rng.normal(size=HOURS)
    uq = .35*up + excitation*.0003*rng.normal(size=HOURS)
    n = METERS + int(hidden)
    p_all = np.column_stack((p, up)) if hidden else p
    q_all = np.column_stack((q, uq)) if hidden else q
    truth = ((0, 1), (2, 3, 4) if hidden else (2, 3))
    y = np.zeros_like(p)
    for line in (*basis(n), *truth):
        r = .06 if len(line) == n else .045 if line in truth else .025
        indices = [i for i in line if i < METERS]
        flow = (r*p_all[:, line] + .5*r*q_all[:, line]).sum(axis=1)
        y[:, indices] += flow[:, None]
    y += .00022*(rng.normal(size=(HOURS, 1)) + rng.normal(size=y.shape))
    v0sq = np.full(HOURS, 1.02**2)
    ratio = np.sqrt(v0sq[:, None]/(v0sq[:, None]-y))
    data = dict(P=p, Q=q, Y=y, V0sq=v0sq, Ppcc=(p*ratio).sum(axis=1) + (up if hidden else 0),
                Qpcc=(q*ratio).sum(axis=1) + (uq if hidden else 0))
    return data, truth


def all_trees(lines, tree=()):
    """Generate each compatible family once by adding only later compatible lines."""
    yield tree
    for i, line in enumerate(lines):
        remaining = [other for other in lines[i+1:] if compatible(line, other)]
        yield from all_trees(remaining, (*tree, line))


def enumerate_scores(data, info, full, hidden):
    """Independent explicit design, box least squares and a supporting-plane lower bound."""
    p, q = data["P"], data["Q"]
    if hidden:
        up, uq = balance(**data)
        p, q = np.column_stack((p, up)), np.column_stack((q, uq))
    base = basis(p.shape[1])
    gamma, sigma = info["gamma"], np.sqrt(info["sigma2"])
    y = data["Y"] - gamma*data["Y"].mean(axis=1, keepdims=True)
    target = ((y-y.mean(axis=0))/sigma).ravel()
    columns = {}
    for line in (*base, *full):
        mask = np.isin(np.arange(METERS), line).astype(float)
        weight = (mask-gamma*mask.mean())/sigma
        blocks = []
        for flows in (p, q):
            flow = flows[:, line].sum(axis=1)
            blocks.append(((flow-flow.mean())[:, None]*weight).ravel())
        columns[line] = np.column_stack(blocks)

    scores = {}
    for tree in all_trees(full):
        design = np.hstack([columns[line] for line in (*base, *tree)])
        fit = lsq_linear(design, target, bounds=(0, BOUND), method="bvls", tol=1e-11)
        if not fit.success:
            raise RuntimeError(f"Bounded fit failed for {tree}: {fit.message}")
        theta = np.clip(fit.x, 0., BOUND)
        residual = design@theta-target
        upper = .5*(residual@residual + info["N"]*info["alpha"]*len(tree))
        gradient = design.T@residual
        # Minimise the tangent at the feasible theta over [0, BOUND] coordinatewise.
        lower = upper + np.minimum(-theta*gradient, (BOUND-theta)*gradient).sum()
        scores[tree] = (float(lower), float(upper))
    return scores


def meter_groups(tree):
    return frozenset(group for line in tree if 1 < len(group := tuple(i for i in line if i < METERS)) < METERS)


def interval(rows):
    return [min(row[k] for row in rows) for k in (0, 1)] if rows else None


def hidden_support_path(tree):
    """Meter descendants on internal supports carrying the hidden leaf, from root to leaf."""
    carrying = sorted((line for line in tree if METERS in line), key=len, reverse=True)
    return tuple(tuple(i for i in line if i != METERS) for line in carrying)


def near_optimal_summary(scores, hidden):
    best_lower, best_upper = interval(list(scores.values()))
    inside = sorted((tree for tree, (_, upper) in scores.items() if upper <= best_lower + SETTINGS.m - EPS),
                    key=lambda tree: scores[tree][1])
    outside = sum(lower > best_upper + SETTINGS.m + EPS for lower, _ in scores.values())
    summary = dict(trees_within=len(inside), trees_outside=outside,
                   trees_on_numerical_boundary=len(scores) - len(inside) - outside,
                   distinct_meter_group_sets_within=len({meter_groups(tree) for tree in inside}),
                   distinct_hidden_internal_support_paths_within=None)
    if hidden:
        summary["distinct_hidden_internal_support_paths_within"] = len({hidden_support_path(t) for t in inside})
        if inside:
            first = inside[0]
            alternative = next((t for t in inside if meter_groups(t) == meter_groups(first)
                                and hidden_support_path(t) != hidden_support_path(first)), None)
            if alternative is not None:
                summary["same_meter_groups_different_paths"] = [
                    dict(tree=t, meter_groups=sorted(meter_groups(t)), internal_support_path=hidden_support_path(t),
                         H=scores[t], H_minus_best=[max(0., scores[t][0] - best_upper), scores[t][1] - best_lower])
                    for t in (first, alternative)
                ]
    return summary


def compare_domain(pool, scores, ref, targets, true_groups):
    pool_set = set(pool)
    available = [(tree, bounds) for tree, bounds in scores.items() if pool_set.issuperset(tree)]
    best_tree = min(available, key=lambda pair: pair[1][1])[0]
    reference = scores[ref]
    ref_groups = meter_groups(ref)
    rows = []
    for target in targets:
        with_group, without_group = [], []
        for tree, bounds in available:
            (with_group if target in meter_groups(tree) else without_group).append(bounds)
        present, absent = interval(with_group), interval(without_group)
        opposite = absent if target in ref_groups else present
        if present is None:
            verdict = "not_covered"
        elif opposite[1] <= reference[0]+SETTINGS.m-EPS:
            verdict = "open"
        elif opposite[0] > reference[1]+SETTINGS.m+EPS:
            verdict = "decided"
        else:
            verdict = "unsettled"
        rows.append(dict(group=target, in_reference=target in ref_groups, true=target in true_groups,
                         present_H=present, absent_H=absent, verdict=verdict))
    inside = [r for r in rows if r["in_reference"]]
    return dict(pool=pool, families=len(available), best_tree=best_tree, best_H=interval([b for _, b in available]),
                truth_groups_covered=len(true_groups & meter_groups(pool)), inside_decided=sum(
                    r["verdict"] == "decided" for r in inside),
                inside_open=sum(r["verdict"] == "open" for r in inside), rows=rows)


def audit(excitation, hidden):
    start = perf_counter()
    data, truth = readings(excitation, hidden)
    result = identify(**data, st=SETTINGS)
    id_seconds = perf_counter()-start
    if result["gate"]["load"] != hidden:
        raise ValueError(f"Unexpected gate decision: {result['gate']}")
    leaves = METERS+int(hidden)
    full = tuple(line for k in range(2, leaves) for line in combinations(range(leaves), k))
    initial = canonical(result["first"]["domain"]["lines"])
    final = canonical(result["pool"])
    ref = canonical(result["first"]["tree"])
    assert set(ref) <= set(initial) <= set(final) <= set(full)
    scores = enumerate_scores(data, result["model"], full, hidden)
    assert len(scores) == {4: 26, 5: 236}[leaves]
    max_width = max(upper-lower for lower, upper in scores.values())
    selected_error = abs(result["model"]["N"]/2*result["model"]["crit"] - scores[canonical(result["lines"])][1])
    first_error = abs(result["model"]["N"]/2*result["first"]["crit"] - scores[ref][1])
    if max(max_width, first_error, selected_error) > EPS:
        raise ValueError(f"Independent scores exceed tolerance: {max_width}, {first_error}, {selected_error}")
    groups = sorted(meter_groups(full), key=lambda g: (len(g), g))
    domains = {name: compare_domain(pool, scores, ref, groups, meter_groups(truth))
               for name, pool in [("initial", initial), ("cumulative", final), ("all", full)]}
    transitions = []
    for first, middle, last in zip(*(domain["rows"] for domain in domains.values())):
        flip_key = "absent_H" if first["in_reference"] else "present_H"
        flips = [row[flip_key] for row in (first, middle, last) if row[flip_key] is not None]
        assert all(b[1] <= a[1]+EPS for a, b in zip(flips, flips[1:]))
        if first["in_reference"] and first["verdict"] == "decided" and last["verdict"] == "open":
            transitions.append(first["group"])
    first = result["first"]
    if first["bound"] is not None:
        assert first["bound"]*result["model"]["N"]/2 <= domains["initial"]["best_H"][1]+EPS
    final_rows = compare_domain(final, scores, canonical(result["lines"]), groups, meter_groups(truth))["rows"]
    final_oracle = {row["group"]: row["verdict"] for row in final_rows}
    labels_checked = 0
    for label in result["labels"]:
        if label["label"] in ("open", "decided"):
            assert label["label"] == final_oracle[tuple(label["grouping"])], label
            assert label["threshold_excluded"] == (final_oracle[tuple(label["grouping"])] == "decided"), label
            labels_checked += 1
    round_statuses = [dict(anchored=rec["solve"]["status"], pricing_complete=rec["price"]["certified"],
                           unproved_nodes=rec["price"]["unproved"]) for rec in result["rounds"] if "solve" in rec]
    return dict(excitation=excitation, hidden=hidden, truth=truth, true_tree_H=scores[canonical(truth)], first_tree=ref,
                reference_H=scores[ref], final_tree=canonical(result["lines"]), gate=result["gate"],
                input_sha256={k: hashlib.sha256(v.astype("<f8").tobytes()).hexdigest() for k, v in data.items()},
                first_status=first["status"], first_bound_crit=first["bound"], scope=result["scope"],
                stop=result["stop"], round_statuses=round_statuses, final_labels_checked=labels_checked,
                final_labels_unsettled=sum(l["label"] == "unsettled" for l in result["labels"]),
                initial_inside_decided_to_full_open=transitions, domains=domains,
                full_near_optimal=near_optimal_summary(scores, hidden),
                largest_fit_interval_width_H=max_width, largest_difference_vs_identifier_H=max(first_error, selected_error),
                identify_seconds=id_seconds, seconds=perf_counter()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="save detailed JSON; otherwise print it")
    args = parser.parse_args()
    run_dir = Path(__file__).resolve().parent
    sources = [Path(__file__), *(run_dir.parent/"method").glob("*.py")]
    output = dict(seed=SEED, hours=HOURS, settings=asdict(SETTINGS), score_tolerance_H=EPS,
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                gurobi=gp.gurobi.version()),
                  source_sha256={p.name: hashlib.sha256(p.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
                                 for p in sorted(sources)}, cases=[])
    for excitation in (1., .15):
        for hidden in (False, True):
            output["cases"].append(audit(excitation, hidden))
    report = json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False)
    if args.out:
        args.out.write_text(report+"\n", encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
