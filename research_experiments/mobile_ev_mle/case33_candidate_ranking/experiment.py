"""Predeclared full-case33 candidate-ranking audit with reused noisy meters."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
from time import perf_counter
import zipfile

import numpy as np

from research_experiments.mobile_ev_mle.case33_quality_fixed_rx import experiment as base
from research_experiments.mobile_ev_mle.ev_fixed_response import FixedResponse
from research_experiments.mobile_ev_mle.reference_benchmark.representation import encode_physical_tree

VERSION = "case33_known_equipment_candidate_ranking_v1"
INPUT_RELATIVE = "research_experiments/mobile_ev_mle/case33_quality_fixed_rx/evidence/raw_results.zip"
GRAD_TOL = 1e-7
SCORE_TOL = 1e-7


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    base.save(path, value)


def rooted(branches):
    adjacency = {i: [] for i in range(1, 34)}
    for e in branches:
        a, b = e["from_bus"], e["to_bus"]
        adjacency[a].append((b, e["branch_id"]))
        adjacency[b].append((a, e["branch_id"]))
    parents = {1: (None, None)}
    todo = [1]
    for a in todo:
        for b, edge in adjacency[a]:
            if b == parents[a][0]:
                continue
            if b in parents:
                raise ValueError("cycle")
            parents[b] = (a, edge)
            todo.append(b)
    if len(parents) != 33 or len(branches) != 32:
        raise ValueError("not a connected 33-bus tree")
    return parents, adjacency


def path_edges(branches, a, b):
    _, adjacency = rooted(branches)
    previous = {a: (None, None)}
    queue = [a]
    for v in queue:
        for w, edge in adjacency[v]:
            if w not in previous:
                previous[w] = (v, edge)
                queue.append(w)
    path = []
    while b != a:
        b, edge = previous[b]
        path.append(edge)
    return path


def catalog():
    """Uses only static public equipment and tree; never observations or scores."""
    net = base.make_context()["network"]
    original = net["branches"]
    candidates = [dict(name="true", family="original", branches=deepcopy(original), edit={})]
    for i, tie in enumerate(net["metadata"]["normally_open_ties"], 1):
        opened = max(path_edges(original, tie["from_bus"], tie["to_bus"]))
        new = deepcopy(tie)
        new.update(branch_id=32+i, status=True, branch_type="original_tie_closed")
        edges = [deepcopy(e) for e in original if e["branch_id"] != opened] + [new]
        candidates.append(dict(name=f"switch_{i}", family="switch", branches=edges,
            edit=dict(close_tie=[tie["from_bus"],tie["to_bus"]], open_branch=opened)))
    parents, adjacency = rooted(original)
    for leaf in sorted(b for b in adjacency if b != 1 and len(adjacency[b]) == 1):
        parent, edge_id = parents[leaf]
        grandparent = parents[parent][0]
        if grandparent is None:
            raise ValueError("declared reattachment has no grandparent")
        edges = deepcopy(original)
        for e in edges:
            if e["branch_id"] == edge_id:
                e.update(from_bus=grandparent, to_bus=leaf)
        candidates.append(dict(name=f"wiring_{leaf}", family="wiring_record_error", branches=edges,
            edit=dict(leaf=leaf, old_parent=parent, new_parent=grandparent, branch=edge_id)))
    for candidate in candidates:
        rooted(candidate["branches"])
        root_edges = [e for e in candidate["branches"] if e["branch_id"] == 1]
        assert len(root_edges) == 1 and root_edges[0] == original[0]
        candidate["equipment_hash"] = hashlib.sha256(json.dumps(candidate["branches"],
            sort_keys=True).encode()).hexdigest()
    assert len(candidates) == 10
    return candidates


def make_context(candidate):
    ctx = base.make_context()
    ctx["network"] = deepcopy(ctx["network"])
    ctx["network"]["branches"] = candidate["branches"]
    net = ctx["network"]
    encoded = encode_physical_tree(candidate["branches"], ctx["ids"], net["root_bus"],
        net["base_kv"], net["base_kw"], source_bus=2, name=candidate["name"])
    ctx["encoded"] = encoded
    ctx["backends"] = {m: FixedResponse(encoded.candidate, encoded.r, encoded.x, method=m)
                       for m in ("linear", "i2", "ac")}
    return ctx


def observations(archive, *, seed_limit=None):
    records = []
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if not name.endswith(".json"):
                continue
            item = json.loads(z.read(name))
            if not isinstance(item, dict) or not all(k in item for k in ("zobs", "yobs", "sigma_z", "sigma_y")):
                continue
            if seed_limit is not None and not item["clean"] and item["seed"] not in base.CONFIG["seeds"][:seed_limit]:
                continue
            keys = ("state", "power_sigma", "seed", "clean", "zobs", "yobs", "sigma_z", "sigma_y")
            record = {k: item[k] for k in keys}
            record["id"] = Path(name).stem
            records.append(record)
    return records


def compact_starts(starts):
    return [{k:v for k,v in s.items() if k not in ("z", "w")} for s in starts]


def fit_observation(ctx, observation):
    """Receives only noisy meters, weights and labels. No true P/Q or true outputs."""
    result = {k:observation[k] for k in ("id", "state", "power_sigma", "seed", "clean")}
    zo, yo, sz, sy = [np.asarray(observation[k], float) for k in ("zobs", "yobs", "sigma_z", "sigma_y")]
    result.update(methods={}, failures=[])
    for backend in ("linear", "i2", "ac"):
        for mode in ("FIX", "EIV"):
            key = backend+"/"+mode
            try:
                item = base.fit_method(ctx, backend, mode, zo, yo, sz, sy)
            except Exception as exc:
                item = dict(success=False, failure=repr(exc), backend=backend, mode=mode)
            result["methods"][key] = item
            if not item["success"]:
                result["failures"].append(key)
    ref = result["methods"]["ac/EIV"]
    extra = []
    if "z" in ref:
        for backend in ("linear", "i2"):
            item = result["methods"][backend+"/EIV"]
            if "z" in item:
                extra.append(base.solve_start(ctx, "ac", zo, yo, sz, sy,
                    (item["z"]-zo)/sz, "lm", "audit_AC_from_"+backend))
        valid = [s for s in ref["starts"]+extra if s.get("failure") is None]
        best = min(valid, key=lambda s:s["score"])
        ref.update(z=best["z"], score=best["score"], gradient_inf=best["gradient_inf"],
            selected_start=best["label"], success=best["success"], scaled_gradient=best["scaled_gradient"],
            start_score_spread=max(s["score"] for s in valid)-min(s["score"] for s in valid),
            audit_seconds=ref.get("audit_seconds",0)+sum(s["seconds"] for s in extra))
        result["reference_qualified"] = bool(best["success"] and best["scaled_gradient"] <= GRAD_TOL
            and ref["start_score_spread"] <= SCORE_TOL*(1+best["score"]))
    else:
        result["reference_qualified"] = False
    result["extra_AC_starts"] = compact_starts(extra)
    for key, item in result["methods"].items():
        if "z" not in item:
            continue
        try:
            own, parts = base.score(ctx, item["backend"], item["z"], zo, yo, sz, sy)
            back, back_parts = base.score(ctx, "ac", item["z"], zo, yo, sz, sy)
            exact, physical = base.independent_output(ctx, item["z"])
            encoded, _ = base.response(ctx, "ac", item["z"])
            item.update(score=own, parts=parts, AC_back_score=back, AC_back_parts=back_parts,
                independent_AC_max_error=float(np.max(abs(exact-encoded))), physical=physical)
            if item.get("gradient_inf") is not None:
                item["scaled_gradient"] = item["gradient_inf"]/(1+np.sqrt(2*own))
        except Exception as exc:
            item.update(success=False, postcheck_failure=repr(exc))
            result["failures"].append(key+"/postcheck")
        item["starts"] = compact_starts(item.get("starts", []))
    return result


def candidate_worker(candidate, data, destination):
    ctx = make_context(candidate)
    result = []
    directory = Path(destination)/candidate["name"]
    directory.mkdir(parents=True, exist_ok=True)
    for i, observation in enumerate(data):
        path = directory/(observation["id"]+".json")
        if path.exists():
            item = json.loads(path.read_text(encoding="utf-8"))
        else:
            item = fit_observation(ctx, observation)
            save(path, item)
        result.append(item)
        save(directory/"progress.json", dict(completed=i+1, total=len(data)))
    return candidate["name"], result


def ranking(reference, proposed, names):
    a = np.asarray(reference, float); m = np.asarray(proposed, float)
    if not np.isfinite(a).all() or not np.isfinite(m).all():
        return dict(available=False)
    top_a = [i for i,x in enumerate(a) if abs(x-min(a)) <= SCORE_TOL*(1+max(abs(x),abs(min(a))))]
    top_m = [i for i,x in enumerate(m) if abs(x-min(m)) <= SCORE_TOL*(1+max(abs(x),abs(min(m))))]
    pairs = []
    for i,j in combinations(range(len(names)), 2):
        da, dm = a[i]-a[j], m[i]-m[j]
        tolerance = SCORE_TOL*(1+max(abs(a[i]),abs(a[j]),abs(m[i]),abs(m[j])))
        sign_a = int(da > tolerance)-int(da < -tolerance)
        sign_m = int(dm > tolerance)-int(dm < -tolerance)
        pairs.append(dict(a=names[i], b=names[j], AC_gap=da, proposed_gap=dm,
            gap_error=dm-da, numerical_tolerance=tolerance, AC_tie=sign_a==0,
            proposed_tie=sign_m==0, inversion=sign_a*sign_m<0,
            includes_true=(names[i]=="true" or names[j]=="true")))
    true = names.index("true")
    rank_a = 1+sum(x < a[true]-SCORE_TOL*(1+max(abs(x),abs(a[true]))) for x in a)
    rank_m = 1+sum(x < m[true]-SCORE_TOL*(1+max(abs(x),abs(m[true]))) for x in m)
    # A set-valued tie is reported explicitly. The deterministic winner uses static catalog order.
    chosen = top_m[0]
    return dict(available=True, top1_set_equal=top_a==top_m,
        AC_top=[names[i] for i in top_a], proposed_top=[names[i] for i in top_m],
        true_AC_rank=rank_a, true_proposed_rank=rank_m,
        AC_winner_regret=float(a[chosen]-min(a)), pairs=pairs,
        weakest_pair=min(pairs,key=lambda p:abs(p["AC_gap"])),
        inversions=sum(p["inversion"] for p in pairs), pair_count=len(pairs),
        true_pair_inversions=sum(p["inversion"] for p in pairs if p["includes_true"]))


def summarize(candidates, data, results):
    banks = dict(switch=[c["name"] for c in candidates if c["family"] in ("original", "switch")],
                 wiring=[c["name"] for c in candidates if c["family"] in ("original", "wiring_record_error")],
                 union=[c["name"] for c in candidates])
    indexed = {k:{s["id"]:s for s in samples} for k,samples in results.items()}
    details = []
    for obs in data:
        for bank,names in banks.items():
            for mode in ("FIX", "EIV"):
                ac = [indexed[k][obs["id"]]["methods"]["ac/"+mode] for k in names]
                qualified = all(s.get("success",False) for s in ac)
                if mode == "EIV":
                    qualified &= all(indexed[k][obs["id"]]["reference_qualified"] for k in names)
                a = [s.get("score") for s in ac]
                for backend in ("linear", "i2"):
                    items = [indexed[k][obs["id"]]["methods"][backend+"/"+mode] for k in names]
                    for score_type,field in (("self","score"),("AC_back","AC_back_score")):
                        m = [s.get(field) for s in items]
                        available = all(s.get("success",False) for s in items)
                        detail = {k:obs[k] for k in ("id","state","power_sigma","seed","clean")}
                        detail.update(bank=bank, mode=mode, backend=backend, score_type=score_type,
                            reference_qualified=bool(qualified), methods_successful=available,
                            reference_scores=a, proposed_scores=m, names=names)
                        detail.update(ranking(a,m,names))
                        details.append(detail)
    groups = []
    keys = ("bank","mode","backend","score_type","state","power_sigma","clean")
    for key in sorted({tuple(d[k] for k in keys) for d in details}):
        group = [d for d in details if tuple(d[k] for k in keys)==key]
        valid = [d for d in group if d["available"] and d["methods_successful"] and d["reference_qualified"]]
        pairs = [p for d in valid for p in d["pairs"]]
        tp = [p for p in pairs if p["includes_true"]]
        out = dict(zip(keys,key))
        out.update(total=len(group), qualified=len(valid), top1_equal=sum(d["top1_set_equal"] for d in valid),
            inversions=sum(p["inversion"] for p in pairs), pair_count=len(pairs),
            true_pair_inversions=sum(p["inversion"] for p in tp), true_pair_count=len(tp),
            AC_true_top=sum("true" in d["AC_top"] for d in valid),
            proposed_true_top=sum("true" in d["proposed_top"] for d in valid),
            AC_ties=sum(p["AC_tie"] for p in pairs), proposed_ties=sum(p["proposed_tie"] for p in pairs),
            max_regret=max((d["AC_winner_regret"] for d in valid),default=None),
            min_AC_pair_gap=min((abs(p["AC_gap"]) for p in pairs),default=None),
            max_absolute_pair_gap_error=max((abs(p["gap_error"]) for p in pairs),default=None))
        groups.append(out)
    return dict(version=VERSION, banks=banks, groups=groups), details


def run(args):
    destination = Path(args.output); destination.mkdir(parents=True,exist_ok=True)
    archive = Path(args.input)
    candidates = catalog()
    # Persist unscored static catalog before opening any measured observation.
    catalog_path = destination/"candidate_catalog.json"
    if catalog_path.exists() and json.loads(catalog_path.read_text(encoding="utf-8")) != candidates:
        raise ValueError("existing output has another candidate catalog; choose a fresh directory")
    save(catalog_path,candidates)
    data = observations(archive,seed_limit=args.seed_limit)
    configuration = dict(version=VERSION, input=str(archive), input_sha256=digest(archive),
        input_config=base.CONFIG, seed_limit=args.seed_limit, observations=len(data),
        candidates=len(candidates), scientific_protocol_sha256=digest(Path(__file__).with_name("protocol.md")),
        reference_normalized_gradient_tolerance=GRAD_TOL,reference_score_tolerance=SCORE_TOL,
        R_X_fixed=True, source_P_Q_fixed=[0,0], workers=args.workers, executable=sys.executable)
    config_path = destination/"configuration.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        for key in configuration:
            if key not in ("input","workers","executable") and previous[key] != configuration[key]:
                raise ValueError("cached results disagree with configuration: "+key)
        if (destination/"execution.json").exists():
            raise ValueError("run already completed; audit saved results or choose a fresh directory")
    save(config_path,configuration)
    save(destination/"observations.json",data)
    began=perf_counter(); results={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(candidate_worker,c,data,str(destination/"raw")) for c in candidates]
        for future in as_completed(futures):
            name,items=future.result(); results[name]=items
            print(f"{name}: {len(items)} observations done",flush=True)
    summary,details=summarize(candidates,data,results)
    save(destination/"summary.json",summary); save(destination/"rankings.json",details)
    all_samples=[s for samples in results.values() for s in samples]
    all_methods=[m for s in all_samples for m in s["methods"].values()]
    save(destination/"execution.json",dict(seconds=perf_counter()-began,candidate_observations=len(all_samples),
        method_slots=len(all_methods),method_failures=sum(not m["success"] for m in all_methods),
        reference_qualified=sum(s["reference_qualified"] for s in all_samples),
        max_independent_AC_error=max(m.get("independent_AC_max_error",0) for m in all_methods),
        primary_seconds_by_method={key:sum(s["methods"][key].get("primary_seconds",0) for s in all_samples)
            for key in all_samples[0]["methods"]},
        audit_seconds=sum(m.get("audit_seconds",0) for m in all_methods)))
    print(json.dumps(json.loads((destination/"execution.json").read_text()),indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--seed-limit",type=int)
    run(parser.parse_args())
