"""Small paired node-refinement and line-penalty comparison. Run from topology_study."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import gurobipy as gp
import numpy as np
import scipy
from scipy.optimize import lsq_linear

from ptid.method.identify import Settings, identify
from ptid.method.model import BOUND, Fitter, Model, parameters_per_line
from ptid.method.sets import basis, canonical, children, compatible, unions


SEED, METERS, HOURS = 20261127, 11, 96
WALL_LIMIT = 120              # seconds per arm, including Python startup and all verification
RELEASE_SECONDS = 60          # separate, post-decision time allowance per anchor
LIMITS = dict(K=9, repairs=2, rounds=1, short_seconds=4, long_seconds=4,
              flip_seconds=4, node_seconds=4, start_seconds=1, evidence=False)


def readings(scale):
    """A trunk with ten children, one of them serving meters 2 and 8 (zero-based)."""
    rng = np.random.default_rng(SEED)
    daily = np.sin(2*np.pi*np.arange(HOURS)/24)[:, None]
    p = .022 + .005*daily + scale*.004*rng.normal(size=(HOURS, METERS))
    q = .3*p + scale*.0015*rng.normal(size=p.shape)
    truth = ((2, 8),)
    # Twice the line impedances in per unit, as in the linearised voltage model.
    y = np.zeros_like(p)
    for line in (*basis(METERS), *truth):
        resistance = .045 if line in truth else .06 if len(line) == METERS else .025
        drop = resistance*(p[:, line]+.5*q[:, line]).sum(axis=1)
        y[:, line] += drop[:, None]
    y += .00022*(rng.normal(size=(HOURS, 1))+rng.normal(size=y.shape))
    return dict(P=p, Q=q, Y=y, V0sq=np.full(HOURS, 1.02**2)), truth


def score_check(data, model, tree):
    """Refit the returned tree independently on its explicit, centred voltage design."""
    cols = []
    for line in (*basis(METERS), *tree):
        z = np.isin(np.arange(METERS), line).astype(float)
        weight = z-model["gamma"]*z.mean()
        for key in ("P", "Q"):
            flow = data[key][:, line].sum(1)
            cols.append(((flow-flow.mean())[:, None]*weight).ravel())
    y = data["Y"]-model["gamma"]*data["Y"].mean(axis=1, keepdims=True)
    fit = lsq_linear(np.column_stack(cols), (y-y.mean(0)).ravel(), bounds=(0, BOUND), method="bvls")
    if not fit.success:
        raise RuntimeError(f"Independent final fit failed: {fit.message}")
    return float(fit.cost/model["sigma2"] + model["N"]/2*model["alpha"]*len(tree))


def run_arm(scale, p_s, enabled):
    data, truth = readings(scale)
    st = Settings(**LIMITS, p_s=p_s, node_refinement=enabled)
    result = identify(**data, st=st)
    first, model = result["first"], result["model"]
    half = model["N"]/2
    initial_pool = canonical(first["domain"]["lines"])
    final = canonical(result["lines"])
    h_final = half*model["crit"]
    h_check = score_check(data, model, final)
    if abs(h_check-h_final) > 1e-5:
        raise ValueError(f"Independent final H differs: {h_check} vs {h_final}")
    summaries, additions = [], []
    for rnd in result["rounds"]:
        if "solve" not in rnd:
            continue
        solve = rnd["solve"]
        domain = canonical(solve["domain"]["lines"])
        # Without a hidden leaf each required group has a single support.
        anchors = canonical(group[0] for group in solve["domain"]["require"])
        tree = canonical(solve["tree"])
        price = rnd.get("price") or {}
        for addition in price.get("added", []):
            addition = tuple(addition)
            node, kids = min(((node, kids) for node, kids in children(tree, METERS).items()
                              if set(addition) < set(node)), key=lambda pair: len(pair[0]))
            parts = [k for k in kids if set(k) <= set(addition)]
            if set(addition) != set().union(*map(set, parts)) or not 2 <= len(parts) < len(kids):
                raise ValueError("An accepted line is not a proper union of current children")
            additions.append(dict(line=addition, in_initial=addition in initial_pool,
                                  in_pre_pricing_domain=addition in domain, node=node, width=len(kids),
                                  method="MIQP" if len(kids)>st.K else "enumeration",
                                  physical=addition in truth, retained=addition in final))
            tree = canonical([*tree, addition])
        repeated = price.get("re_solve")
        summaries.append(dict(round=rnd["round"], domain_size=len(domain), anchors=anchors,
                              false_anchors=[a for a in anchors if a not in truth],
                              missing_truth_crossing_anchor=[c for c in truth if c not in domain
                                                              and any(not compatible(a,c) for a in anchors)],
                              anchors_in_final_tree=[a for a in anchors if a in final],
                              anchors_in_final_anchors=[a for a in anchors if a in result["anchors"]],
                              tree_status=solve["status"], tree_H=half*solve["crit"],
                              tree_bound_H=None if solve["bound"] is None else half*solve["bound"],
                              re_solve=None if repeated is None else dict(status=repeated["status"],
                                  H=half*repeated["crit"], bound_H=None if repeated["bound"] is None
                                  else half*repeated["bound"]),
                              scanned_widths=sorted((len(k) for k in children(solve["tree"], METERS).values()
                                                     if len(k)>2), reverse=True),
                              last_pricing_wide_nodes=price.get("wide_nodes", []),
                              last_pricing_enumerated=price.get("enumerated"),
                              last_pricing_unproved=price.get("unproved"),
                              still_paying=price.get("still_paying")))
    negative = [x["grouping"] for x in result["labels"] if x["gap"][1] is not None and x["gap"][1] < -1e-6]
    statuses = Counter("local" if row["status"] is None else str(row["status"]) for row in result["labels"])
    return dict(scale=scale, p_s=p_s, node_refinement=enabled, observed_p_s=parameters_per_line(data["P"], data["Q"]),
                input_sha256={k: hashlib.sha256(v.astype("<f8").tobytes()).hexdigest() for k, v in data.items()},
                truth=truth, initial_pool=initial_pool, true_lines_in_initial=[c for c in truth if c in initial_pool],
                warm_tree=first["warm_tree"], warm_H=half*first["warm_crit"], first_tree=first["tree"],
                first_status=first["status"], first_H=half*first["crit"],
                first_bound_H=None if first["bound"] is None else half*first["bound"],
                rounds=summaries, additions=additions, final_tree=final,
                flip_witnesses=[{key: rnd[key] for key in ("round", "flip_witness", "after_flip_witness") if key in rnd}
                                for rnd in result["rounds"]],
                final_true_positive=len(set(truth)&set(final)), final_false_positive=len(set(final)-set(truth)),
                final_false_negative=len(set(truth)-set(final)), final_H=h_final, independent_H=h_check,
                final_anchors=result["anchors"], final_better_flip_groups=negative, scope=result["scope"],
                final_label_statuses=dict(statuses), final_labels=dict(Counter(x["label"] for x in result["labels"])),
                stop=result["stop"], identification_seconds=result["timing"]["total"])


def audit_anchors(arm, pool):
    """Release one first-round anchor in the common first tree and fit each local crossing replacement."""
    tree = canonical(arm["first_tree"])
    anchors = arm["rounds"][0]["anchors"]
    if not anchors:
        return []
    data, _ = readings(arm["scale"])
    model = Model(**data, p_s=arm["p_s"])
    half = model.N/2
    reference = half*Fitter(model, tree).crit(tree)
    if abs(reference-arm["first_H"]) > 1e-6:
        raise ValueError("Anchor audit used a different reference fit")
    audit = []
    for anchor in canonical(anchors):
        start = perf_counter()
        parent = min((c for c in (*tree, tuple(range(METERS))) if set(anchor) < set(c)), key=len)
        base = canonical(c for c in tree if c != anchor)
        kids = children(base, METERS)[parent]
        lines = [c for c in unions(kids) if not compatible(c, anchor)]
        tested = competitive = outside = competitive_outside = 0
        best = {"in_initial_pool": None, "outside_initial_pool": None}
        for i in range(0, len(lines), 32):
            if perf_counter()-start >= RELEASE_SECONDS:
                break
            chunk = lines[i:i+32]
            fit = Fitter(model, [*base, *chunk])
            for line in chunk:
                if perf_counter()-start >= RELEASE_SECONDS:
                    break
                rival = canonical([*base, line])
                delta = half*fit.crit(rival)-reference
                tested += 1
                outside += line not in pool
                competitive += delta <= Settings.m
                competitive_outside += line not in pool and delta <= Settings.m
                key = "in_initial_pool" if line in pool else "outside_initial_pool"
                if best[key] is None or delta < best[key]["delta_H"]:
                    best[key] = dict(line=line, tree=rival, delta_H=delta)
        audit.append(dict(anchor=anchor, parent=parent, children_after_release=len(kids),
                          candidates=len(lines), tested=tested, tested_outside_pool=outside, competitive=competitive,
                          competitive_outside_pool=competitive_outside, best=best,
                          complete=tested == len(lines), seconds=perf_counter()-start))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write full experiment JSON")
    parser.add_argument("--arm", nargs=3, metavar=("SCALE","P_S","NODE"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.arm:
        scale, p_s, node = args.arm
        print(json.dumps(run_arm(float(scale), int(p_s), bool(int(node))), allow_nan=False))
        return
    if args.out is None:
        parser.error("--out is required")
    source = Path(__file__).resolve()
    sources = [source, *(p for p in (source.parents[1]/"method").glob("*.py") if p.name != "probe.py")]
    document = dict(seed=SEED, meters=METERS, hours=HOURS, wall_limit_seconds=WALL_LIMIT,
                    posthoc_seconds_per_anchor=RELEASE_SECONDS, anchor_checks=[],
                    base_settings=asdict(Settings(**LIMITS)),
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                                       scipy=scipy.__version__, gurobi=gp.gurobi.version()),
                    source_sha256={p.name: hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
                                   for p in sorted(sources)}, initial_pools={}, arms=[])
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PYTHONHASHSEED="0")
    for scale in (1., .15):
        for p_s in (1, 2):
            pair = []
            for enabled in (False, True):
                start = perf_counter()
                command = [sys.executable, "-B", "-m", "ptid.runs.node_pairs", "--arm",
                           str(scale), str(p_s), str(int(enabled))]
                try:
                    proc = subprocess.run(command, timeout=WALL_LIMIT, capture_output=True, text=True, env=environment)
                    if proc.returncode:
                        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
                    arm = dict(outcome="completed", **json.loads(proc.stdout.splitlines()[-1]))
                except (subprocess.TimeoutExpired, RuntimeError) as error:
                    arm = dict(outcome="timeout" if isinstance(error, subprocess.TimeoutExpired) else "error",
                               scale=scale, p_s=p_s, node_refinement=enabled, detail=str(error))
                arm["wall_seconds"] = perf_counter()-start
                document["arms"].append(arm)
                pair.append(arm)
                print(scale, p_s, enabled, arm["outcome"], f'{arm["wall_seconds"]:.2f}s', flush=True)
            if all(a["outcome"] == "completed" for a in pair):
                for key in ("input_sha256", "initial_pool", "warm_tree", "first_tree"):
                    if pair[0][key] != pair[1][key]:
                        raise ValueError(f"Unmatched pair at scale {scale}, p_s={p_s}: {key}")
                for key in ("warm_H", "first_H"):
                    if abs(pair[0][key]-pair[1][key]) > 1e-6:
                        raise ValueError(f"Unmatched starting scores: {key}")
                initial = set(map(tuple, pair[0]["initial_pool"]))
                document["anchor_checks"].append(dict(scale=scale, p_s=p_s,
                    released=audit_anchors(pair[0], initial)))
            for arm in pair:
                if arm["outcome"] == "completed":
                    pool = arm.pop("initial_pool")
                    digest = hashlib.sha256(json.dumps(pool).encode()).hexdigest()
                    document["initial_pools"][digest] = pool
                    arm["initial_pool_sha256"] = digest
    changed = [p.name for p in sources if hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
               != document["source_sha256"][p.name]]
    if changed:
        raise RuntimeError(f"Code changed during calculation: {changed}")
    args.out.write_text(json.dumps(document, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
