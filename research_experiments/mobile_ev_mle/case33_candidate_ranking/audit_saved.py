"""Independent replay of fixed-equipment full-case33 ranking evidence."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from research_experiments.mobile_ev_mle.case33_quality_fixed_rx.audit_saved import PhysicalReplay
from research_experiments.mobile_ev_mle.reference_benchmark.networks import build_case33
from research_experiments.mobile_ev_mle.case33_quality_fixed_rx import experiment as base


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(value, message):
    if not value:
        raise AssertionError(message)


def traversal(edges):
    adj = {i: [] for i in range(1, 34)}
    for row in edges:
        a, b = row["from_bus"], row["to_bus"]
        require(a != b and a in adj and b in adj, "invalid endpoint")
        adj[a].append((b, row["branch_id"]))
        adj[b].append((a, row["branch_id"]))
    parent = {1: (None, None)}
    queue = [1]
    for a in queue:
        for b, branch in adj[a]:
            if b == parent[a][0]:
                continue
            require(b not in parent, "cycle")
            parent[b] = (a, branch)
            queue.append(b)
    require(len(parent) == 33 and len(edges) == 32, "not a 33-bus tree")
    return parent, adj


def pair_records(ac, proposed, names):
    rows = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            da = ac[i]-ac[j]
            dm = proposed[i]-proposed[j]
            tolerance = 1e-7 * (1+max(abs(ac[i]), abs(ac[j]), abs(proposed[i]), abs(proposed[j])))
            sa = 0 if abs(da) <= tolerance else (1 if da > 0 else -1)
            sm = 0 if abs(dm) <= tolerance else (1 if dm > 0 else -1)
            rows.append(dict(a=names[i], b=names[j], AC_gap=da, proposed_gap=dm,
                gap_error=dm-da, numerical_tolerance=tolerance,
                AC_tie=sa == 0, proposed_tie=sm == 0, inversion=sa*sm < 0,
                includes_true=names[i] == "true" or names[j] == "true"))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="Relocated identical original input archive")
    args = parser.parse_args()
    output = args.output
    began = time.perf_counter()
    modspec = importlib.util.spec_from_file_location("ranking_target", Path(__file__).with_name("experiment.py"))
    target = importlib.util.module_from_spec(modspec)
    modspec.loader.exec_module(target)
    conf = read(output/"configuration.json")
    candidates = read(output/"candidate_catalog.json")
    obs = read(output/"observations.json")
    details = read(output/"rankings.json")
    summary = read(output/"summary.json")
    execution = read(output/"execution.json")
    require(len(candidates) == 10 and len(obs) == 51, "incorrect frozen count")
    require(execution["candidate_observations"] == 510 and execution["method_slots"] == 3060, "missing result slots")
    original_input = args.input if args.input is not None else Path(conf["input"])
    require(hashlib.sha256(original_input.read_bytes()).hexdigest() == conf["input_sha256"], "input archive changed")
    require(hashlib.sha256(Path(__file__).with_name("protocol.md").read_bytes()).hexdigest() == conf["scientific_protocol_sha256"], "protocol changed")
    original = build_case33()
    parent, adjacent = traversal(original["branches"])
    byid = {e["branch_id"]: e for e in original["branches"]}
    replays = {}
    for candidate in candidates:
        edges = candidate["branches"]
        traversal(edges)
        require(len({e["branch_id"] for e in edges}) == 32, "duplicate equipment ID")
        name = candidate["name"]
        if name == "true":
            require(edges == original["branches"], "true network differs")
        elif candidate["family"] == "switch":
            index = int(name.split("_")[1])-1
            tie = original["metadata"]["normally_open_ties"][index]
            nextid = 33+index
            missing = set(byid)-{e["branch_id"] for e in edges}
            require(len(missing) == 1, "switch removes wrong edge count")
            edgelook = {e["branch_id"]: e for e in edges}
            for ident, row in byid.items():
                if ident not in missing:
                    require(edgelook[ident] == row, "switched existing device changed")
            new = edgelook[nextid]
            require(all(new[k] == tie[k] for k in ("from_bus", "to_bus", "r_ohm", "x_ohm")), "tie data changed")
            # A second BFS obtains the true-tree path between tie terminals.
            start, finish = tie["from_bus"], tie["to_bus"]
            route = {start: (None, None)}
            queue = [start]
            for a in queue:
                for b, ident in adjacent[a]:
                    if b not in route:
                        route[b] = (a, ident)
                        queue.append(b)
            path = []
            at = finish
            while at != start:
                at, ident = route[at]
                path.append(ident)
            require(missing == {max(path)}, "wrong predeclared cycle cut")
        else:
            leaf = candidate["edit"]["leaf"]
            par, ident = parent[leaf]
            grand = parent[par][0]
            require(len(adjacent[leaf]) == 1 and grand is not None, "wrong reattachment leaf")
            expected = deepcopy(original["branches"])
            for row in expected:
                if row["branch_id"] == ident:
                    row.update(from_bus=grand, to_bus=leaf)
            require(edges == expected, "wiring device or wrong endpoints changed")
        net = deepcopy(original)
        net["branches"] = edges
        replays[name] = PhysicalReplay(net)

    method_count = reference_count = 0
    method_failures = []
    max_self = max_back = max_output = max_decomposition = 0.
    # Reconstruct every score using the independent physical branch-table model.
    for candidate in candidates:
        name = candidate["name"]
        replay = replays[name]
        for observation in obs:
            rec = read(output/"raw"/name/(observation["id"]+".json"))
            require(all(rec[k] == observation[k] for k in ("id", "state", "power_sigma", "seed", "clean")), "observation identity differs")
            zo, yo, sz, sy = [np.asarray(observation[k], float) for k in ("zobs", "yobs", "sigma_z", "sigma_y")]
            ref = rec["methods"]["ac/EIV"]
            starts = ref["starts"]+rec["extra_AC_starts"]
            valid = [s for s in starts if s.get("failure") is None]
            best = min(valid, key=lambda s: s["score"])
            qualified = (best["success"] and best["scaled_gradient"] <= 1e-7
                and max(s["score"] for s in valid)-min(s["score"] for s in valid) <= 1e-7*(1+best["score"]))
            require(bool(qualified) == rec["reference_qualified"], "reference qualification mismatch")
            reference_count += qualified
            for key, item in rec["methods"].items():
                method_count += 1
                if not item["success"]:
                    method_failures.append([name, observation["id"], key])
                if "z" not in item:
                    continue
                z = np.asarray(item["z"], float)
                if key.endswith("/FIX"):
                    require(np.array_equal(z, zo), "FIX changed measured PQ")
                yp = replay.output(z, item["backend"])
                ya = replay.output(z, "ac")
                rz = (z-zo)/sz
                rm, ra = (yp-yo)/sy, (ya-yo)/sy
                own, back = .5*(rz@rz+rm@rm), .5*(rz@rz+ra@ra)
                es, eb = abs(own-item["score"]), abs(back-item["AC_back_score"])
                max_self, max_back = max(max_self, es), max(max_back, eb)
                require(es <= 1e-7*(1+abs(own)), "independent self score differs")
                require(eb <= 1e-7*(1+abs(back)), "independent AC back score differs")
                max_decomposition = max(max_decomposition, abs((own-ref["score"])-((own-back)+(back-ref["score"]))))
                max_output = max(max_output, item["independent_AC_max_error"])
                if key == "ac/EIV":
                    require(back <= rec["methods"]["ac/FIX"]["score"]+1e-7*(1+abs(back)), "AC EIV worse than feasible FIX")

    strict = inversions = ties = top_mismatch = 0
    for detail in details:
        ac = np.asarray(detail["reference_scores"], float)
        proposed = np.asarray(detail["proposed_scores"], float)
        names = detail["names"]
        require(detail["available"] == (np.isfinite(ac).all() and np.isfinite(proposed).all()), "availability differs")
        if not detail["available"]:
            continue
        pairs = pair_records(ac, proposed, names)
        require(len(pairs) == len(detail["pairs"]), "pair denominator differs")
        for p, stored in zip(pairs, detail["pairs"]):
            for key in p:
                if isinstance(p[key], float):
                    require(abs(p[key]-stored[key]) < 1e-10*(1+abs(p[key])), "pair numeric differs")
                else:
                    require(p[key] == stored[key], "pair decision differs")
            strict += not p["AC_tie"] and not p["proposed_tie"]
            inversions += p["inversion"]
            ties += p["AC_tie"] or p["proposed_tie"]
        minimum_ac, minimum_m = min(ac), min(proposed)
        top_ac = [names[i] for i, v in enumerate(ac) if abs(v-minimum_ac) <= 1e-7*(1+max(abs(v), abs(minimum_ac)))]
        top_m = [names[i] for i, v in enumerate(proposed) if abs(v-minimum_m) <= 1e-7*(1+max(abs(v), abs(minimum_m)))]
        require(top_ac == detail["AC_top"] and top_m == detail["proposed_top"], "top candidate set differs")
        require(detail["top1_set_equal"] == (top_ac == top_m), "top1 agreement differs")
        chosen = names.index(top_m[0])
        require(abs(ac[chosen]-minimum_ac-detail["AC_winner_regret"]) < 1e-10, "AC winner regret differs")
        top_mismatch += top_ac != top_m
        true_index = names.index("true")
        rank_a = 1+sum(v < ac[true_index]-1e-7*(1+max(abs(v), abs(ac[true_index]))) for v in ac)
        rank_m = 1+sum(v < proposed[true_index]-1e-7*(1+max(abs(v), abs(proposed[true_index]))) for v in proposed)
        require(rank_a == detail["true_AC_rank"] and rank_m == detail["true_proposed_rank"], "true rank differs")
    require(len(details) == 51*3*2*2*2, "missing ranking conditions")
    # Audit every reported group/qualification denominator using saved rows.
    group_keys = ("bank", "mode", "backend", "score_type", "state", "power_sigma", "clean")
    for group in summary["groups"]:
        allrows = [d for d in details if all(d[k] == group[k] for k in group_keys)]
        good = [d for d in allrows if d["available"] and d["methods_successful"] and d["reference_qualified"]]
        pairs = [p for d in good for p in d["pairs"]]
        require(group["total"] == len(allrows) and group["qualified"] == len(good), "group denominator differs")
        require(group["pair_count"] == len(pairs), "group pair denominator differs")
        require(group["inversions"] == sum(p["inversion"] for p in pairs), "group inversion differs")
        require(group["top1_equal"] == sum(d["top1_set_equal"] for d in good), "group top1 differs")

    # Independent centered directional derivatives in standardized PQ units,
    # for each candidate and state. No optimization and no fit uses true PQ.
    derivative_rows = []
    rng = np.random.default_rng(2026101833)
    directions = [rng.normal(size=64) for _ in range(2)]
    directions = [v/np.linalg.norm(v) for v in directions]
    for candidate in candidates:
        name = candidate["name"]
        ctx = target.make_context(candidate)
        replay = replays[name]
        for observation in [o for o in obs if o["clean"]]:
            z, sz, sy = [np.asarray(observation[k], float) for k in ("zobs", "sigma_z", "sigma_y")]
            for backend in ("linear", "i2", "ac"):
                _, analytic = base.response(ctx, backend, z, True)
                for number, direction in enumerate(directions):
                    dz = sz*direction
                    h = .05
                    fd = (replay.output(z+h*dz, backend)-replay.output(z-h*dz, backend))/(2*h)
                    exact = analytic@dz
                    error = float(np.max(abs((fd-exact)/sy)))
                    scale = float(max(1., np.max(abs(exact/sy))))
                    require(error/scale < 3e-6, "independent directional derivative differs")
                    derivative_rows.append(dict(candidate=name, state=observation["state"],
                        backend=backend, direction=number, white_max_error=error, scale=scale))
    result = dict(success=True, seconds=time.perf_counter()-began,
        candidate_count=len(candidates), observations=len(obs), method_slots=method_count,
        reference_qualified=reference_count, method_failures=method_failures,
        max_independent_self_score_difference=max_self,
        max_independent_AC_back_score_difference=max_back,
        max_recorded_independent_AC_output_difference=max_output,
        max_decomposition_roundoff=max_decomposition,
        ranking_conditions=len(details), paired_comparisons=strict+ties,
        strict_comparisons=strict, tie_comparisons=ties, inversions=inversions,
        top1_mismatches=top_mismatch, directional_checks=len(derivative_rows),
        max_directional_white_error=max(r["white_max_error"] for r in derivative_rows),
        scope="independent physical branch-table replay; known equipment RX; 48 paired noisy conditions plus 3 clean diagnostics")
    base.save(output/"independent_audit.json", result)
    base.save(output/"independent_directional_checks.json", derivative_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
