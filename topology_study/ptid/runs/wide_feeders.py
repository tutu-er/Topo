"""Paired wide-junction AC feeders. From topology_study run python -m ptid.runs.wide_feeders."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import gurobipy as gp
import numpy as np
import scipy

from ptid.cases import physics
from ptid.cases.paths import OUTPUTS
from ptid.method import builders, identify as identifier
from ptid.method.model import Fitter, Model
from ptid.method.sets import canonical, children, compatible
from ptid.runs.four_offline import digest, settings
from ptid.runs.passive_baselines import run_baseline
from ptid.runs.random_dev_compare import cached_resistance

SEED = 20261129
HOURS = 192
BLOCK_COUNT = 12
TARGET_BLOCKS = (0, 2, 4, 6, 8, 10, 11)
STUDY = Path(__file__).resolve().parents[2]
ST = settings()
PHASES = ("controlled_B", "controlled_C", "first", "normal_B", "normal_C")


def block_layout(n):
    size = n // BLOCK_COUNT
    return [tuple(range(j*size, (j+1)*size)) for j in range(BLOCK_COUNT)]


def observations(n, shared, *, include_circuit=False):
    """Generate measured arrays and offline evaluation data for one declared physical circuit."""
    blocks = block_layout(n)
    target = tuple(i for j in TARGET_BLOCKS for i in blocks[j])
    labels = list(range(1000, 1000+n))
    lines = []
    def edge(a, b, r, x):
        lines.append(dict(from_bus=a, to_bus=b, r_ohm=r, x_ohm=x, status=True))
    edge(0, 1, .016, .008)
    if shared:
        edge(1, 2, .036, .014)
    for j, block in enumerate(blocks):
        edge(2 if shared and j in TARGET_BLOCKS else 1, 100+j, .040, .016)
        for i in block:
            edge(100+j, labels[i], .020, .008)
    nodes = sorted({e[s] for e in lines for s in ("from_bus", "to_bus")})
    net = dict(root_bus=0, buses=[dict(bus_id=b) for b in nodes], branches=lines,
               base_kv=.4, base_mva=.2)
    rng = np.random.default_rng([SEED, n])
    day = np.sin(2*np.pi*np.arange(4*HOURS)/96)
    perturb = rng.normal(size=(HOURS, n)).repeat(4, axis=0)
    p = .5/n*np.clip(1+.2*day[:, None]+.32*perturb+.04*rng.normal(size=perturb.shape), .12, 2.5)
    q = .32*p + .09*.5/n*rng.normal(size=(HOURS, n)).repeat(4, axis=0)
    v0 = 1.02+.002*day
    order, v, s, _ = physics.power_flow(net, labels, p, q, v0)
    true = dict(P=p, Q=q, V=v[:, [order.index(b) for b in labels]], V0=v0, S_pcc=s)
    meter = physics.read(true, physics.Meters(n, [SEED, n]), 1)
    data = dict(P=meter["P"], Q=meter["Q"], Y=meter["V0"][:, None]**2-meter["V"]**2, V0sq=meter["V0"]**2)
    truth = canonical([*blocks, *([target] if shared else [])])
    topology = children(truth, n)
    evidence = dict(shared=shared, meters=n, physical_blocks=blocks, target=target,
                    true_groups=truth, physical_max_children=max(map(len, topology.values())),
                    target_edge=([1,2] if shared else None), target_physical_depth=(2 if shared else None),
                    target_group_level=(1 if shared else None),
                    voltage_range_pu=[float(v.min()),float(v.max())],
                    max_transformer_va_pu=float(abs(s).max()), observed_sha256=digest(data))
    if include_circuit:
        evidence.update(network=net, meter_bus_ids=labels, seed=SEED,
                        power_base_mva=.2, voltage_base_kv=.4, hours=HOURS, physical_samples_per_hour=4,
                        aggregation="four physical quarter hours averaged before applying the hourly meter error",
                        meter_model="ptid.cases.physics.Meters default gains and noise",
                        meter_seed_key=[SEED,n], load_seed_key=[SEED,n])
    return data, evidence


def export_inputs(directory, sizes):
    """Keep the arrays supplied to an identifier separate from the simulation and scoring truth."""
    directory=directory.resolve()
    if directory.is_relative_to(STUDY/"experiments"):
        raise ValueError("Export generated inputs to an output directory outside the published experiments")
    for n in sizes:
        for shared in (True,False):
            name=f"{n}-"+("shared" if shared else "no-shared")
            folder=directory/name
            folder.mkdir(exist_ok=True,parents=True)
            data, evidence=observations(n,shared,include_circuit=True)
            measured=folder/"observations.npz"
            if measured.exists():
                with np.load(measured,allow_pickle=False) as saved:
                    if set(saved.files)!=set(data) or any(not np.array_equal(saved[k],data[k]) for k in data):
                        raise ValueError(f"Existing measurement file differs: {measured}")
            else:
                with tempfile.NamedTemporaryFile(dir=folder,suffix=".npz",delete=False) as temp:
                    np.savez_compressed(temp,**data)
                Path(temp.name).replace(measured)
            evidence["observations_file_sha256"]=sha256(measured.read_bytes()).hexdigest()
            metadata=folder/"circuit.json"
            if metadata.is_file():
                if json.loads(metadata.read_text(encoding="utf-8"))!=json.loads(json.dumps(evidence)):
                    raise ValueError(f"Existing circuit metadata differs: {metadata}")
            else:
                write(metadata,evidence)
            print(name,"observations",list(data["P"].shape),evidence["observations_file_sha256"],flush=True)


def tree_record(tree, source):
    n = source["meters"]
    groups = set(canonical(tree))
    truth, target = set(canonical(source["true_groups"])), tuple(source["target"])
    widths = {group:len(kids) for group,kids in children(groups, n).items() if kids}
    layers = {line:1+sum(set(line)<set(parent) for parent in truth) for line in truth}
    return dict(tree=canonical(groups), found=len(truth & groups), extra=len(groups-truth),
                missed=len(truth-groups), f1=2*len(truth & groups)/(len(truth)+len(groups)),
                target_in_tree=target in groups, root_side_hits=sum(line in groups for line in truth if len(line)*2>n),
                root_side_truth=sum(len(line)*2>n for line in truth),
                top_two_hits=sum(g in groups for g,d in layers.items() if d<=2),
                top_two_truth=sum(d<=2 for d in layers.values()), root_children=widths[tuple(range(n))],
                direct_children=[dict(node=k,count=v) for k,v in widths.items()])


def compact_solve(info, source):
    return dict(pool_size=info["pool"], domain_hash=sha256(repr(canonical(info["domain"]["lines"])).encode()).hexdigest(),
                required_blocks=info["domain"]["require"],
                warm=tree_record(info["warm_tree"],source), warm_crit=info["warm_crit"],
                incumbent=None if info["solver_tree"] is None else tree_record(info["solver_tree"],source),
                incumbent_crit=info["solver_crit"], retained=tree_record(info["tree"],source),
                status=info["status"], bound=info["bound"], crit=info["crit"],
                domain_gap_closed=info["gap_closed"], miqp_seconds=info["miqp_seconds"])


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",",":"))+"\n", encoding="utf-8")
    temp.replace(path)


def audit(report):
    """Regenerate observations and independently refit returned trees on an explicit voltage design."""
    from ptid.runs.node_mechanism import Reference
    from ptid.method.anchors import anchored_universe
    checks = {}
    for name, phases in report["cases"].items():
        n, shape = name.split("-",1)
        data, source = observations(int(n), shape=="shared")
        if phases["first"].get("source",{}).get("observed_sha256")!=digest(data):
            raise ValueError(f"Observation digest differs for {name}")
        model = Model(**data)
        reference = Reference(data,model)
        fits = {}
        result = dict(prefix_pairs=[], independent_fits=0, max_crit_difference=0., checked_node_bounds=0,
                      verified_re_solve_additions={})
        for family in ("controlled","normal"):
            a, b = (phases[family+"_"+m] for m in ("B","C"))
            if a.get("initial") and b.get("initial"):
                if a["initial"]!=b["initial"] or a["measurement_sha256"]!=b["measurement_sha256"]:
                    raise ValueError(f"Different starting trees or observations for {name} {family}")
            for step in ("labels","anchored_tree"):
                left, right = (row.get("shared_prefix",{}).get(step) for row in (a,b))
                if not right:
                    continue
                if not left or left["cache_sha256"]!=right["cache_sha256"] or left["origin"]!="computed" or right["origin"]!="reused_B":
                    raise ValueError(f"Unequal common prefix for {name} {family} {step}")
                result["prefix_pairs"].append([family,step])
            for row in (a,b):
                if len(row.get("solves",[])) > 1:
                    first, repeated = row["solves"][:2]
                    anchors = [g[0] for g in first["required_blocks"]]
                    warm_pool = source["physical_blocks"] if family=="controlled" else phases["first"]["first_pool"]
                    universe,_ = anchored_universe(anchors, int(n), warm_pool, ST.K)
                    initial = row["initial"]["tree"]
                    universe = canonical([*universe,*initial])
                    proposed = [s["selected"] for s in row["pricing"][:ST.repairs] if s["selected"] is not None]
                    enlarged = canonical([*universe,*proposed])
                    for pool, record in [(universe,first),(enlarged,repeated)]:
                        if sha256(repr(pool).encode()).hexdigest()!=record["domain_hash"]:
                            raise ValueError(f"Re-solve domain differs for {name} {row['phase']}")
                    result["verified_re_solve_additions"][row["phase"]]=proposed
                for node in row.get("node_solves",[]):
                    feasible, bound = node.get("feasible_crit"),node.get("lower_bound")
                    if feasible is not None and bound is not None:
                        if bound>feasible+1e-7:
                            raise ValueError(f"Lower bound exceeds feasible objective for {name}")
                        result["checked_node_bounds"]+=1
                if "final" not in row:
                    continue
                tree = canonical(row["final"]["tree"])
                if tree not in fits:
                    fits[tree] = reference.score(tree)
                low,high = fits[tree]
                difference = abs(high-row["final_crit"])
                if difference>1e-7 or high-low>1e-7:
                    raise ValueError(f"Independent bounded least-squares fit differs at {name}")
                result["max_crit_difference"]=max(result["max_crit_difference"],difference)
        result["independent_fits"]=len(fits)
        checks[name]=result
    for name in checks:
        if not name.endswith("-no-shared"):
            continue
        other = name.replace("-no-shared","-shared")
        left,right=(report["cases"][key]["first"]["source"]["observed_sha256"] for key in (name,other))
        if any((left[k]==right[k])!=(k!="Y") for k in left):
            raise ValueError(f"Unexpected pairing of meter streams for {name}")
    return checks


def audit_time_limits(report, base):
    """Verify the fixed node against its short run and refit feasible trees without an integer solve."""
    from itertools import combinations
    from ptid.runs.node_mechanism import Reference

    seconds = report["settings"]["identifier"]["node_seconds"]
    baseline = base["settings"]["identifier"]["node_seconds"]
    if baseline != report["settings"]["fixed_reference_budget"] or seconds <= baseline:
        raise ValueError("Incorrect reference time limit")
    expected = base["settings"]["identifier"].copy()
    expected.update({k: seconds for k in ("short_seconds", "long_seconds", "flip_seconds", "node_seconds")})
    if report["settings"]["identifier"] != expected:
        raise ValueError("Identifier settings differ beyond the four solver limits")
    for field in ("seed", "hours", "quarter_hours", "target_blocks", "observation_channels"):
        if report["settings"][field] != base["settings"][field]:
            raise ValueError(f"Data settings differ for {field}")

    checks = {}
    for name, phases in report["cases"].items():
        n, shape = name.split("-", 1)
        if shape != "shared":
            raise ValueError(f"Unexpected sensitivity case {name}")
        n = int(n)
        short = base["cases"][name]
        phase_names = ("fixed_node", "controlled_B", "controlled_C")
        if any(phases.get(k, {}).get("status") != "completed" for k in phase_names):
            raise ValueError(f"Incomplete sensitivity stages for {name}")
        data, source = observations(n, True)
        observed = source["observed_sha256"]
        if observed != short["first"]["source"]["observed_sha256"]:
            raise ValueError(f"Regenerated observations differ from the reference for {name}")
        a, b = (phases[k] for k in phase_names[1:])
        model = Model(**data)
        reference = Reference(data, model)
        fitted = {}
        differences = []

        def score(tree):
            groups = canonical(tree)
            if groups not in fitted:
                if any(not compatible(c, d) for c, d in combinations(groups, 2)):
                    raise ValueError(f"Crossing groups in {name}")
                low, high = reference.score(groups)
                if not np.isfinite([low, high]).all() or abs(high-low) > 1e-7:
                    raise ValueError(f"Continuous fit is unresolved for {name}")
                fitted[groups] = high
            return fitted[groups]

        def agree(value, recorded):
            difference = abs(value-recorded)
            if not np.isfinite(difference) or difference > 1e-7:
                raise ValueError(f"Independent fit differs for {name}: {value} versus {recorded}")
            differences.append(difference)

        def node_score(node):
            tree, parent = canonical(node["input_tree"]), tuple(node["parent"])
            kids = children(tree, n)[parent]
            if len(kids) != node["children"]:
                raise ValueError(f"Different fixed-node width for {name}")
            before = score(tree)
            upper = before + model.alpha
            line = node["returned"]
            if line is not None:
                line = set(line)
                parts = [k for k in kids if set(k) <= line]
                if not 2 <= len(parts) < len(kids) or line != set().union(*parts):
                    raise ValueError(f"Returned line is not an admissible union for {name}")
                feasible = score([*tree, line])
                agree(feasible, node["feasible_crit"])
                upper = min(upper, feasible)
            bound = node["lower_bound"]
            if bound is not None and (not np.isfinite(bound) or bound > upper + 1e-7):
                raise ValueError(f"Fixed-node bound exceeds its feasible score for {name}")
            return before, upper

        expected_tree = canonical(source["physical_blocks"])
        for key in phase_names:
            if phases[key]["measurement_sha256"] != observed:
                raise ValueError(f"Different observations for {name} {key}")
        for k in phase_names[1:]:
            if short[k]["measurement_sha256"] != observed or short[k]["initial"] != phases[k]["initial"]:
                raise ValueError(f"Cross-budget initialisation differs for {name} {k}")
            if canonical(phases[k]["initial"]["tree"]) != expected_tree or phases[k]["target_in_initial_pool"]:
                raise ValueError(f"Unexpected controlled initial tree for {name}")
        if a["initial"] != b["initial"] or a["solves"][0] != b["solves"][0]:
            raise ValueError(f"Different B/C common tree for {name}")
        for step in ("labels", "anchored_tree"):
            left, right = (arm["shared_prefix"][step] for arm in (a, b))
            if left["origin"] != "computed" or right != {**left, "origin": "reused_B"}:
                raise ValueError(f"Different B/C common step {step} for {name}")

        old = short["controlled_C"]["node_solves"][0]
        fixed = phases["fixed_node"]["fixed"]
        if fixed["solver_limit"] != seconds or any(old[k] != fixed[k] for k in ("input_tree", "parent", "children")):
            raise ValueError(f"Different reference node or input tree for {name}")
        node_score(old)
        before, upper = node_score(fixed)
        agree(before, fixed["tree_crit"])
        agree(upper, fixed["feasible_upper"])
        if fixed["returned"] is not None:
            agree(before-fixed["feasible_crit"], fixed["gain"])
        if fixed["status"] == gp.GRB.OPTIMAL:
            if fixed["lower_bound"] is None or upper-fixed["lower_bound"] > max(1e-7, abs(upper)*1e-4):
                raise ValueError(f"Optimal fixed-node status conflicts with its bound for {name}")
        reused = []
        for arm in (a, b):
            for record in arm["solves"]:
                for tree_key, value_key in (("warm", "warm_crit"), ("incumbent", "incumbent_crit"), ("retained", "crit")):
                    if record[tree_key] is not None:
                        agree(score(record[tree_key]["tree"]), record[value_key])
            for node in arm["node_solves"]:
                node_score(node)
                same_fixed_input = all(node[k] == fixed[k] for k in ("input_tree", "parent", "children"))
                if same_fixed_input != ("reused_fixed_node_seconds" in node):
                    raise ValueError(f"Fixed-node timing or identity differs for {name}")
                if "reused_fixed_node_seconds" in node:
                    for k in ("input_tree", "parent", "children", "returned", "objective", "feasible_crit", "status", "lower_bound"):
                        if node[k] != fixed[k]:
                            raise ValueError(f"Cached fixed-node result differs in {name} {k}")
                    agree(node["reused_fixed_node_seconds"], fixed["seconds"])
                    reused.append(arm["phase"])
            final = arm["final"]
            computed = tree_record(final["tree"], source)
            if any(computed[k] != final[k] for k in ("found", "extra", "missed", "f1", "target_in_tree")):
                raise ValueError(f"Final topology scores differ for {name}")
            agree(score(final["tree"]), arm["final_crit"])
        checks[name] = dict(phase_count=len(phase_names), independent_fits=len(fitted),
                            max_fit_difference=max(differences), fixed_width=fixed["children"],
                            fixed_lower=fixed["lower_bound"], fixed_upper=upper,
                            fixed_reused_by=reused, fixed_compute_seconds=fixed["seconds"],
                            same_anchored_tree_across_budgets=short["controlled_B"]["solves"][0]["retained"] == a["solves"][0]["retained"])
    return checks


def fixed_node_trial(model, tree, parent, seconds):
    from ptid.method.search import node_miqp
    kids=children(tree,model.n)[tuple(parent)]
    before=Fitter(model,tree).crit(tree)
    began=perf_counter()
    line,obj,status,bound=node_miqp(model,tree,kids,seconds,return_bound=True)
    elapsed=perf_counter()-began
    candidate=None if line is None else Fitter(model,[*tree,line]).crit([*tree,line])
    upper=min(before+model.alpha, candidate) if candidate is not None else before+model.alpha
    if bound is not None and bound>upper+1e-7:
        raise ArithmeticError("A fixed-node lower bound exceeds the refitted feasible upper bound")
    return dict(parent=parent, children=len(kids), input_tree=tree, status=status, returned=line, objective=obj,
                feasible_crit=candidate, lower_bound=bound, feasible_upper=upper,
                tree_crit=before, gain=None if candidate is None else before-candidate,
                no_improvement=bool(bound is not None and bound>=before-1e-7),
                seconds=elapsed, solver_limit=seconds)


def worker(n, shared, phase, stage_dir):
    data, source = observations(n, shared)
    path = stage_dir/(phase+".json")
    result = dict(phase=phase, status="running", source=source) if phase=="first" else dict(phase=phase,status="running")
    def save():
        write(path, result)
    save()
    model = Model(**data)
    budget_file=stage_dir/"budget.json"
    st=ST
    if budget_file.is_file():
        config=json.loads(budget_file.read_text(encoding="utf-8"))
        seconds=config["seconds"]
        st=replace(ST,short_seconds=seconds,long_seconds=seconds,flip_seconds=seconds,node_seconds=seconds)
        if digest(data)!=config["measurement_sha256"]:
            raise ValueError("Different measurements from the frozen short-budget job")
    if phase=="fixed_node":
        previous=config["fixed"]
        fixed=fixed_node_trial(model,canonical(previous["input_tree"]),previous["parent"],seconds)
        if fixed["children"]!=previous["children"]:
            raise ValueError("Different child groups from the frozen short-budget node")
        result.update(status="completed", **{"fixed":fixed}, measurement_sha256=digest(data))
        save()
        return
    if phase == "first":
        result["baselines"] = {}
        with cached_resistance(stage_dir/"arms", "warm") as _:
            for method in ("nj", "rnj"):
                out = run_baseline(method, **data)
                if out["tree"] is None:
                    raise RuntimeError(f"{method}: {out.get('error')}")
                result["baselines"][method] = tree_record(out["tree"],source)
                save()
            starts = builders.builder_trees(**data)
        pool = canonical(g for tree in starts for g in tree)
        result.update(builders={m:tree_record(t,source) for m,t in zip(builders.NAMES,starts)},
                      first_pool=pool, target_in_first_pool=tuple(source["target"]) in pool)
        save()
        tree, first = identifier.solve(model,pool,list(dict.fromkeys(frozenset(t) for t in starts)),st.short_seconds,st)
        result["first_tree_miqp"] = compact_solve(first,source)
        write(stage_dir/"first_state.json", dict(tree=tree,pool=pool,source=digest(data),
                    initial_solve={k:first[k] for k in ("status","crit","bound","gap_closed")}))
    else:
        controlled = phase.startswith("controlled")
        if controlled:
            # Supplied terminal groups are engineering input for this arm; no inter-block line is supplied.
            tree = pool = canonical(block_layout(n))
            initial_solve = None
        else:
            initial = json.loads((stage_dir/"first_state.json").read_text(encoding="utf-8"))
            if initial["source"]!=digest(data):
                raise ValueError("Different measurements in paired first tree")
            tree, pool = map(canonical, (initial["tree"], initial["pool"]))
            initial_solve = initial.get("initial_solve")
        result.update(initial=tree_record(tree,source), target_in_initial_pool=tuple(source["target"]) in pool,
                      measurement_sha256=digest(data), shared_prefix={}, solves=[], pricing=[], node_solves=[])
        save()
        original_solve, original_search = identifier.solve, identifier.best_addition
        original_label = identifier.label_groupings
        from ptid.method import search
        original_node = search.node_miqp
        def common_step(name, inputs, call):
            """Run the common prefix once in B and replay exactly those results in C."""
            key = sha256(json.dumps([digest(data), inputs], sort_keys=True).encode()).hexdigest()
            cache = stage_dir/(phase.rsplit("_",1)[0]+"_"+name+".pkl")
            if phase.endswith("C"):
                if not cache.is_file():
                    raise RuntimeError(f"Paired B did not finish common {name}")
                saved_key, value = pickle.loads(cache.read_bytes())
                if key != saved_key:
                    raise ValueError(f"Different paired {name} inputs")
                origin = "reused_B"
            else:
                value = call()
                binary = cache.with_suffix(".pending")
                binary.write_bytes(pickle.dumps((key, value),protocol=5))
                binary.replace(cache)
                origin = "computed"
            result["shared_prefix"][name] = dict(input_sha256=key, cache_sha256=sha256(cache.read_bytes()).hexdigest(),origin=origin)
            return value
        def label(*args, **kwargs):
            if "labels" in result["shared_prefix"]:
                return original_label(*args,**kwargs)
            inputs = [canonical(args[1]),canonical(args[2]),*args[3:6],args[7]]
            return common_step("labels",inputs,lambda:original_label(*args,**kwargs))
        def solve(*args, **kw):
            if not result["solves"]:
                inputs = [canonical(args[1]),[canonical(start) for start in args[2]],args[3],args[5]]
                candidate,info = common_step("anchored_tree",inputs,lambda:original_solve(*args,**kw))
            else:
                candidate,info = original_solve(*args, **kw)
            result["solves"].append(compact_solve(info, source))
            save()
            return candidate,info
        def node(model_,tree_,kids,seconds,seed=7,*,return_bound=False):
            began = perf_counter()
            row = dict(parent=sorted(i for block in kids for i in block), children=len(kids),
                       input_tree=canonical(tree_), status="running", feasible_crit=None, lower_bound=None)
            result["node_solves"].append(row)
            save()
            fixed_path=stage_dir/"fixed_node.json"
            saved_fixed=json.loads(fixed_path.read_text(encoding="utf-8")).get("fixed") if fixed_path.exists() else None
            if (saved_fixed and saved_fixed["solver_limit"]==seconds and saved_fixed["parent"]==row["parent"]
                    and canonical(saved_fixed["input_tree"])==canonical(tree_) and seed==7):
                line=saved_fixed["returned"]
                line=None if line is None else tuple(line)
                objective,status,bound=(saved_fixed[k] for k in ("objective","status","lower_bound"))
                row["reused_fixed_node_seconds"]=saved_fixed["seconds"]
            else:
                line,objective,status,bound = original_node(model_,tree_,kids,seconds,seed,return_bound=True)
            refit = None if line is None else Fitter(model_,[*tree_,line]).crit([*tree_,line])
            row.update(returned=line, objective=objective, feasible_crit=refit, status=status,
                       lower_bound=bound, seconds=perf_counter()-began)
            if bound is not None and refit is not None and bound>refit+search.PRICING_TOL:
                row.update(lower_bound=None, numerical_bound_conflict=True)
            save()
            return (line,objective,status,bound) if return_bound else (line,objective,status)
        def scan(model_,tree_,*args,**kwargs):
            line,info = original_search(model_,tree_,*args,**kwargs)
            result["pricing"].append(dict(input=tree_record(tree_,source), selected=line, gain=info["gain"],
                                          enumerated=info["enumerated"], examined=info["evaluated"],
                                          wide=info["wide_nodes"], certified=bool(info["certified"])))
            save()
            return line,info
        with (patch.object(identifier,"label_groupings",label), patch.object(identifier,"solve",solve),
              patch.object(identifier,"best_addition",scan), patch.object(search,"node_miqp",node)):
            out = identifier.refine(model,tree,pool,replace(st,wide_node_miqp=phase.endswith("C")),initial_solve=initial_solve)
        final = canonical(out["tree"])
        result.update(final=tree_record(final,source), final_crit=Fitter(model,final).crit(final),
                      stop=out["stop"], labels=[dict(group=g,label=v["label"]) for g,v in out["labels"].items() if g==tuple(source["target"])])
    result["status"]="completed"
    save()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes",type=int,nargs="+",choices=(24,36),default=[24,36])
    ap.add_argument("--out",type=Path)
    ap.add_argument("--workers",type=int,choices=(1,2),default=2)
    ap.add_argument("--phase-seconds",type=int)
    modes=ap.add_mutually_exclusive_group()
    modes.add_argument("--check",type=Path,help="recompute the checks on an existing report")
    ap.add_argument("--reference",type=Path,help="short-budget reference for --check (default: published results.json)")
    modes.add_argument("--export-inputs",type=Path,help="only generate separate observation and circuit files")
    modes.add_argument("--time-sensitivity",type=Path,help="use the two controlled shared cases and fixed nodes from this short-budget report")
    ap.add_argument("--solver-seconds",type=int,help="Tree, Node and flip limits for the time sensitivity")
    ap.add_argument("--worker",nargs=4,metavar=("METERS","SHARED","PHASE","DIRECTORY"),help=argparse.SUPPRESS)
    args=ap.parse_args()
    if args.reference and not args.check:
        ap.error("use --reference with --check")
    if args.check:
        report=json.loads(args.check.read_text(encoding="utf-8"))
        if "fixed_reference_budget" in report["settings"]:
            source=args.reference or STUDY/"experiments"/"wide_feeders"/"results.json"
            checks=audit_time_limits(report,json.loads(source.read_text(encoding="utf-8")))
        else:
            if args.reference:
                ap.error("the short-budget report has no external reference")
            checks=audit(report)
        print(json.dumps(checks,indent=2))
        return
    if args.worker:
        n,sh,phase,work=args.worker
        worker(int(n),bool(int(sh)),phase,Path(work))
        return
    args.phase_seconds=args.phase_seconds or (max(180,12*(args.solver_seconds or 4)) if args.time_sensitivity else 180)
    if len(set(args.sizes))!=len(args.sizes) or args.phase_seconds<=0:
        ap.error("use distinct sizes and a positive stage limit")
    if args.export_inputs:
        export_inputs(args.export_inputs,args.sizes)
        return
    if bool(args.time_sensitivity)!=bool(args.solver_seconds) or (args.solver_seconds is not None and args.solver_seconds<=4):
        ap.error("use --time-sensitivity and --solver-seconds greater than 4 together")
    output=(args.out or OUTPUTS/"topology_study"/("wide_feeders.json" if not args.time_sensitivity else
                                         f"wide_feeders_{args.solver_seconds}s.json")).resolve()
    base=json.loads(args.time_sensitivity.read_text(encoding="utf-8")) if args.time_sensitivity else None
    if args.time_sensitivity and output==args.time_sensitivity.resolve():
        ap.error("use a separate file for the sensitivity results")
    output.parent.mkdir(exist_ok=True,parents=True)
    settings=dict(seed=SEED, hours=HOURS, quarter_hours=4*HOURS, phase_seconds=args.phase_seconds, workers=args.workers,
                  identifier=asdict(ST), target_blocks=TARGET_BLOCKS, root_side="more than half of metered leaves",
                  observation_channels=["P","Q","Y","V0sq"],python=platform.python_version(),numpy=np.__version__,
                  scipy=scipy.__version__,gurobi=gp.gurobi.version())
    results={}
    with tempfile.TemporaryDirectory() as work:
        root=Path(work)
        phases=PHASES
        cases=[(n,s) for n in args.sizes for s in (True,False)]
        if base:
            phases=("fixed_node","controlled_B","controlled_C")
            cases=[(n,True) for n in args.sizes]
            settings["fixed_reference_budget"]=4
            settings["identifier"].update({k:args.solver_seconds for k in ("short_seconds","long_seconds","flip_seconds","node_seconds")})
            for n,_ in cases:
                key=f"{n}-shared"
                prior=base["cases"][key]
                folder=root/key
                folder.mkdir()
                write(folder/"budget.json",dict(seconds=args.solver_seconds,measurement_sha256=prior["first"]["source"]["observed_sha256"],
                           fixed=prior["controlled_C"]["node_solves"][0]))
        for phase in phases:
            def launch(case):
                n,shared=case
                key=f"{n}-"+("shared" if shared else "no-shared")
                folder=root/key
                folder.mkdir(exist_ok=True)
                if phase.startswith("normal") and not (folder/"first_state.json").is_file():
                    return key,dict(phase=phase,status="first_tree_unavailable")
                cmd=[sys.executable,"-B","-m","ptid.runs.wide_feeders","--worker",str(n),str(int(shared)),phase,str(folder)]
                t=perf_counter()
                try:
                    process=subprocess.run(cmd,cwd=STUDY,timeout=args.phase_seconds,capture_output=True,text=True,errors="replace")
                    error=process.stderr[-1200:] if process.returncode else None
                    status="failed" if process.returncode else "completed"
                except subprocess.TimeoutExpired:
                    status,error="wall_limit",None
                path=folder/(phase+".json")
                row=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                row.update(phase=phase,status=status,wall_seconds=perf_counter()-t)
                if error:
                    row["error"]=error
                print(key,phase,status,round(row["wall_seconds"],1),flush=True)
                return key,row
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for key,row in pool.map(launch,cases):
                    results.setdefault(key,{})[phase]=row
                    write(output,dict(settings=settings,cases=results))
    report=dict(settings=settings,cases=results)
    if not base:
        report["checks"]=audit(report)
    elif all(r["status"]=="completed" for c in results.values() for r in c.values()):
        report["checks"]=audit_time_limits(report,base)
    write(output,report)
    if any(r["status"]=="failed" for c in results.values() for r in c.values()):
        raise SystemExit(1)


if __name__=="__main__":
    main()
