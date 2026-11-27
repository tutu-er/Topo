"""Repeat affected random-feeder jobs with more solver time; keep the original observations and model."""
from __future__ import annotations

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import pickle
import subprocess
import sys
from time import perf_counter, sleep
from unittest.mock import patch

from ptid.cases import library
from ptid.runs.four_offline import digest, metrics, save, settings
from ptid.runs.random_dev_compare import cached_resistance, contract, selected


@contextmanager
def feeder_lock(path):
    """OS releases this lock if a timed worker is killed; two queues never solve the same feeder twice."""
    with path.open("a+b") as lock:
        if not lock.tell():
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
                    break
                except OSError as exc:
                    if exc.errno not in (11,13,36):raise
                    sleep(2)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX)
        yield


def published_rows():
    original=Path(__file__).resolve().parents[2]/"experiments"/"four_feeders"/"random_comparison.json"
    records=json.loads(original.read_text(encoding="utf-8"))["rows"]
    return {(r["case"],r["method"]):r for r in records}


def selections(arms,all_verified=False):
    chosen = []
    reference=published_rows()
    for name in selected():
        if reference.get((name,"nj"),{}).get("status")!="ok":
            continue
        file = arms / f"{name}__proposed.json"
        if not file.is_file():
            raise FileNotFoundError(f"Missing the original per-feeder Proposed receipt: {file}")
        old = json.loads(file.read_text(encoding="utf-8"))
        if all_verified or old["status"] == "wall_limit" or (old["status"] == "ok" and old["score"]["f1"] < 1
                and (not old["first"]["gap_closed"] or not old["scope"]["restricted"])):
            chosen.append(name)
    priority = {"medium":0,"large":1,"huge":2,"small":3}
    high_priority = ("dev_medium_002_head","dev_large_005_none","dev_large_006_head")
    return sorted(chosen, key=lambda n:(high_priority.index(n) if n in high_priority else len(high_priority),
                                        priority[library.kind(n)], n))


def run_case(name, outputs, seconds, cache):
    """Record every returned tree/flip/price subproblem, and resume only identical calls."""
    import gurobipy as gp
    module = import_module("ptid.method.identify")
    labels = import_module("ptid.method.labels")
    pricing = import_module("ptid.method.search")
    arms = outputs/"topology_study_random_dev"/"arms"
    library.LIBRARY = outputs/"case_library_20261125"
    rec = library.open_case(name)
    old = json.loads((arms/f"{name}__proposed.json").read_text(encoding="utf-8"))
    nj = json.loads((arms/f"{name}__nj.json").read_text(encoding="utf-8"))
    history = library.readings(rec)
    data = {k:history[k] for k in ("P","Q","Y","V0sq")}
    hashes = digest(data)
    if nj["history_sha256"] != hashes or old["run_contract"] != contract(name,"proposed",120):
        raise ValueError(f"Input, model, seed, or original settings changed on {name}")
    st = replace(settings(), short_seconds=seconds, long_seconds=seconds, node_seconds=seconds, flip_seconds=seconds)
    spec = dict(st=asdict(st), inputs=hashes, original=old["run_contract"])
    folder = cache/sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    folder.mkdir(parents=True,exist_ok=True)
    progress = dict(case=name, settings=asdict(st), history_sha256=hashes, steps=[], labels=Counter(), wide_nodes=[],
                    completed_flips=0, restored_calls=0, elapsed_s=0)
    checked_steps = []
    started = perf_counter()
    def update(stage):
        progress.update(current=stage, elapsed_s=perf_counter()-started)
        save(cache/(name+".progress.json"),progress)

    def remember(kind, key, compute):
        # A feeder has a single fixed no-PCC model. Key all other subproblem arguments, including the
        # exact current tree, warm starts, cutoff, and local-witness value for each flip.
        signature = sha256(pickle.dumps((kind,key),protocol=5)).hexdigest()
        dest = folder/(signature+".pkl")
        if dest.is_file():
            answer = pickle.loads(dest.read_bytes())
            progress["restored_calls"]+=1
        else:
            update(kind)
            answer = compute()
            save(dest,answer)
        return answer

    solve_tree, flip_one, node_one = module.solve, labels.flip_test, pricing.node_miqp
    def solve(model,pool,starts,limit,settings,require=()):
        ordinal = len(progress["steps"])
        tree,info = remember("tree",(ordinal,pool,starts,limit,require),
                             lambda:solve_tree(model,pool,starts,limit,settings,require))
        checked_steps.append(info)
        progress["steps"].append(dict(tree=tree, H=info["crit"]*model.N/2,
            lower_H=None if info["bound"] is None else info["bound"]*model.N/2,
            warm_H=info["warm_crit"]*model.N/2, warm_tree=info["warm_tree"], pool=info["pool"], status=info["status"],
            raw_miqp_tree=info.get("solver_tree"), raw_miqp_H=(None if info.get("solver_crit") is None
                                                            else info["solver_crit"]*model.N/2),
            gap_closed=info["gap_closed"], miqp_seconds=info["miqp_seconds"],
            domain_sha256=sha256(json.dumps(info["domain"],sort_keys=True).encode()).hexdigest(),
            required_groups=len(info["domain"]["require"])))
        update("tree_returned")
        return tree,info
    def flip(model,*args,**kwargs):
        result = remember("flip",(args,kwargs),lambda:flip_one(model,*args,**kwargs))
        progress["labels"][result["label"]]+=1
        progress["completed_flips"]+=1
        update("flip_returned")
        return result
    def node(model,*args,**kwargs):
        ans = remember("node",(args,kwargs),lambda:node_one(model,*args,**kwargs))
        progress["wide_nodes"].append(dict(selected=ans[0], objective=ans[1],status=ans[2],
                                          bound=ans[3] if len(ans)>3 else None))
        update("node_returned")
        return ans
    update("building_initial_trees")
    with ExitStack() as stack:
        stack.enter_context(cached_resistance(arms,name))
        for module_, label, sub in ((module,"solve",solve),(labels,"flip_test",flip),(pricing,"node_miqp",node)):
            stack.enter_context(patch.object(module_,label,sub))
        fit = module.identify(**data,st=st)
    for step,checked in zip(progress["steps"],checked_steps):
        # A later flip can invalidate an earlier solver bound.
        step.update(status=checked["status"], gap_closed=checked["gap_closed"],
                    lower_H=None if checked["bound"] is None else checked["bound"]*fit["model"]["N"]/2)
    truth=library.truth(rec)
    result=dict(case=name, status="ok", meters=rec["n"], record_sha256=old["run_contract"]["record_sha256"],
                settings=asdict(st), history_sha256=hashes, elapsed_s=perf_counter()-started,
                tree=fit["tree"], score=metrics(fit["tree"],truth), H=fit["model"]["crit"]*fit["model"]["N"]/2,
                model=fit["model"], scope=fit["scope"], stop=fit["stop"], steps=progress["steps"], wide_nodes=progress["wide_nodes"],
                node_additions=[r["price"]["added"] for r in fit["rounds"] if "price" in r],
                flip_outcomes=progress["labels"], final_labels=dict(Counter(row["label"] for row in fit["labels"])),
                restored_calls=progress["restored_calls"], gurobi=list(gp.gurobi.version()))
    save(cache/(name+".json"),result)
    update("finished")
    return result


def replay_narrow_additions(model, st, steps, directory):
    """Validate older receipts against each enumerated move and the resulting restricted domain."""
    from ptid.method.search import best_addition
    from ptid.method.sets import lines_of
    if st.rounds!=1 or len(steps)!=3:
        return {}
    domains={}
    for snapshot in directory.glob("*.pkl"):
        pair=pickle.loads(snapshot.read_bytes())
        if isinstance(pair,tuple) and len(pair)==2 and isinstance(pair[1],dict) and "domain" in pair[1]:
            domain=pair[1]["domain"]
            domains[sha256(json.dumps(domain,sort_keys=True).encode()).hexdigest()]=domain
    older=domains.get(steps[1].get("domain_sha256"))
    newer=domains.get(steps[2].get("domain_sha256"))
    if not older or not newer or older["require"]!=newer["require"]:
        return {}
    tree=steps[1]["tree"]
    moves={}
    for ordinal in range(st.repairs):
        line,info=best_addition(model,tree,st.K,st.node_seconds,wide_node_miqp=False)
        if info["unproved"]:
            return {}
        if line is None:
            break
        moves[line]=dict(order=ordinal+1,enumerated_nodes=info["enumerated"],
                         candidates=info["evaluated"],gain_H=info["gain"]*model.N/2)
        tree=lines_of([*tree,line],model.leaves)
    if set(map(tuple,older["lines"]))|set(moves)!=set(map(tuple,newer["lines"])):
        raise ValueError("The replayed narrow-node additions produce a different candidate domain")
    return moves


def collect(names, outputs, seconds, out):
    """Re-score the old and new trees on identical verified inputs and retain each restricted lower bound."""
    from ptid.method.model import Model
    from ptid.runs.four_offline import independent_H
    from ptid.runs.root_ablation import physical_edges
    library.LIBRARY=outputs/"case_library_20261125"
    base=outputs/"topology_study_random_dev"
    cache=base/f"budget_{seconds}s"
    new_settings=replace(settings(),short_seconds=seconds,long_seconds=seconds,node_seconds=seconds,flip_seconds=seconds)
    rows=[]
    baseline_scores={}
    environment=None
    affected=set(selections(base/"arms"))
    reference=published_rows()
    for name in names:
        def read(path):
            return json.loads(path.read_text(encoding="utf-8"))
        record=library.open_case(name)
        truth=library.truth(record)
        edges=physical_edges(record)
        if set(edges)!=truth:raise ValueError(f"Physical grouping mismatch on {name}")
        outside = truth.copy()
        baselines={m:read(base/"arms"/f"{name}__{m}.json") for m in ("rg","nj","rnj","park")}
        for method in ("nj","rnj"):
            outside.difference_update(map(tuple,baselines[method]["tree"]))
        joint=dict(physical_first_two=sorted(g for g in outside if any(e["physical_depth"]<=2 for e in edges[g])),
                   identifiable_first_two=sorted(g for g in outside if sum(set(g)<set(h) for h in truth)<=1),
                   majority=sorted(g for g in outside if 2*len(g)>record["n"]))
        def group_hits(tree):
            estimate=set(map(tuple,tree))
            return {k:[g for g in groups if g in estimate] for k,groups in joint.items()}
        samples=library.readings(record)
        data={k:samples[k] for k in ("P","Q","Y","V0sq")}
        h=digest(data)
        old=read(base/"arms"/f"{name}__proposed.json")
        control=old["run_contract"]
        if control!=contract(name,"proposed",120):
            raise ValueError(f"Original input/settings hash differs: {name}")
        recorded={k:control[k] for k in ("calculation_sha256","python","numpy","scipy","gurobi","sampling")}
        if environment is not None and recorded!=environment:
            raise ValueError(f"Mixed calculation source, environment, or sampling: {name}")
        environment=recorded
        for method,baseline in baselines.items():
            published=reference[name,method]
            if (baseline["status"]!="ok" or baseline["record_sha256"]!=control["record_sha256"]
                    or h!=baseline["history_sha256"] or h!=published["history_sha256"] or
                    metrics(baseline["tree"],truth)!=baseline["score"] or baseline["score"]!=published["score"]):
                raise ValueError(f"Baseline input, record or score differs: {name}, {method}")
        if old["status"]!=reference[name,"proposed"]["status"] or old.get("score")!=reference[name,"proposed"].get("score"):
            raise ValueError(f"Original Proposed receipt differs from the published score/status: {name}")
        baseline_scores[name]={method:baseline["score"] for method,baseline in baselines.items()}
        row=dict(case=name,meters=record["n"],record_sha256=control["record_sha256"],observations=h,
                 role="originally_time_suspect" if name in affected else "originally_closed_paired_control",
                 baseline_scores=baseline_scores[name],
                 original_status=old["status"],original_seconds=old["wall_seconds"],original_tree=old.get("tree"),
                 original_score=old.get("score"),original_scope=old.get("scope"),original_first=old.get("first"))
        if old.get("tree") is not None:
            if old["history_sha256"]!=h or metrics(old["tree"],truth)!=old["score"]:
                raise ValueError(f"Original tree or score differs: {name}")
            row["original_H"]=old["model"]["crit"]*old["model"]["N"]/2
            row["original_nearby_recoveries"]=group_hits(old["tree"])
        row["joint_nj_rnj_misses"]=joint
        if "completed_restricted_solves" in old:
            row["original_completed_steps"]=[{k:s[k] for k in ("tree","status","crit","bound","warm_crit","gap_closed","pool")}
                for s in old["completed_restricted_solves"] or []]
        file=cache/(name+".json")
        path=cache/(name+".attempts.json")
        row["attempts"]=read(path) if path.is_file() else []
        if file.is_file():
            result=read(file)
            if (h!=result["history_sha256"] or asdict(new_settings)!=result["settings"] or
                    result["record_sha256"]!=control["record_sha256"] or metrics(result["tree"],truth)!=result["score"]):
                raise ValueError(f"Relaxed receipt inconsistent: {name}")
            m=Model(data["P"],data["Q"],data["Y"],data["V0sq"],None,new_settings.lam,p_s=new_settings.p_s)
            check=independent_H(data,dict(model=dict(gamma=m.gamma,sigma2=m.sigma2,alpha=m.alpha,N=m.N),
                                     gate={"load":False},tree=[tuple(t) for t in result["tree"]]))
            if abs(check-result["H"])>1e-4:raise ValueError((name,check,result["H"]))
            row["new"]={k:result[k] for k in ("status","seconds","tree","H","score","scope","stop","steps",
                          "wide_nodes","flip_outcomes","final_labels","restored_calls") if k in result}
            row["new"].update(final_call_elapsed_s=result["elapsed_s"], independent_H=check, N=m.N)
            row["new"]["nearby_recoveries"]=group_hits(result["tree"])
            steps=row["new"]["steps"]
            if any("raw_miqp_tree" not in s for s in steps):
                # The first receipts predate the raw-incumbent field. Recover it from the
                # original numerical subproblem, not from the post-MIQP local search.
                spec=dict(st=asdict(new_settings),inputs=h,original=control)
                directory=cache/sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
                snapshots=[]
                for p in directory.glob("*.pkl"):
                    pair=pickle.loads(p.read_bytes())
                    if isinstance(pair,tuple) and len(pair)==2 and isinstance(pair[1],dict) and "domain" in pair[1]:
                        snapshots.append(pair[1])
                for s in steps:
                    if "raw_miqp_tree" in s:continue
                    matches=[a for a in snapshots if a["pool"]==s["pool"] and
                        set(map(tuple,a["tree"]))==set(map(tuple,s["tree"])) and abs(a["crit"]*m.N/2-s["H"])<1e-6]
                    if not matches or len({json.dumps(a.get("solver_tree")) for a in matches})!=1:
                        raise ValueError(f"Ambiguous original raw incumbent in {name}")
                    a=matches[0]
                    s["raw_miqp_tree"]=a.get("solver_tree")
                    s["raw_miqp_H"]=None if a.get("solver_crit") is None else a["solver_crit"]*m.N/2
            first_warm=set(map(tuple,steps[0]["warm_tree"]))
            end=set(map(tuple,result["tree"]))
            for step in steps:
                step["warm_score"]=metrics(step["warm_tree"],truth)
                step["after_optional_local_score"]=metrics(step["tree"],truth)
                step["raw_miqp_score"]=None if step["raw_miqp_tree"] is None else metrics(step["raw_miqp_tree"],truth)
            additions=result.get("node_additions")
            accepted=None if additions is None else {tuple(g) for batch in additions for g in batch}
            for node in row["new"]["wide_nodes"]:
                group=None if node["selected"] is None else tuple(node["selected"])
                node.update(identifiable_true_group=group in truth,kept_in_final=group in end,
                            accepted_in_pricing=None if accepted is None else group in accepted)
            row["new"]["joint_miss_absent_at_first_warm_present_at_final"]=[]
            targets=sorted(outside & (end-first_warm))
            replayed={}
            if targets and additions is None and not result["wide_nodes"]:
                spec=dict(st=asdict(new_settings),inputs=h,original=control)
                directory=cache/sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
                replayed=replay_narrow_additions(m,new_settings,steps,directory)
            for target in targets:
                phases=[dict(warm=target in set(map(tuple,s["warm_tree"])),
                             raw_miqp=None if s["raw_miqp_tree"] is None else target in set(map(tuple,s["raw_miqp_tree"])),
                             after_optional_local_search=target in set(map(tuple,s["tree"]))) for s in steps]
                detail=dict(group=target,steps=phases,
                    root_scopes=[key for key,targets in joint.items() if target in targets], physical_edges=edges[target],
                    wide_node_returned=any(n.get("selected") is not None and tuple(n["selected"])==target for n in result["wide_nodes"]),
                    accepted_node_addition=None if additions is None else any(target in set(map(tuple,a)) for a in additions))
                if target in replayed:
                    detail.update(accepted_node_addition=True,verified_narrow_pricing=replayed[target])
                row["new"]["joint_miss_absent_at_first_warm_present_at_final"].append(detail)
        elif (cache/(name+".progress.json")).is_file():
            r=read(cache/(name+".progress.json"))
            row["new"]=dict(status=row["attempts"][-1]["status"] if row["attempts"] else "running",
                            step_reached=r["current"],steps=r["steps"],wide_nodes=r["wide_nodes"],flips=r["labels"])
        else:
            row["new"]=dict(status="not_started")
        fixed=sorted((p for p in cache.glob(f"{name}.fixed.*.json")),key=lambda p:p.name)
        if fixed:
            row["fixed_domain_extensions"]=[json.loads(p.read_text(encoding="utf-8")) for p in fixed]
        rows.append(row)
    finished={r["case"] for r in rows if r["new"]["status"]=="ok"}
    def score_panel(subset):
        complete=[r for r in rows if r["case"] in finished and r["case"] in subset]
        data={}
        for method in ("rg","nj","rnj","park","proposed"):
            vals=[r["new"]["score"] if method=="proposed" else baseline_scores[r["case"]][method] for r in complete]
            data[method]=dict(n=len(vals),mean_F1=sum(v["f1"] for v in vals)/len(vals) if vals else None,
                              exact_trees=sum(v["exact_match"] for v in vals))
        return dict(declared=len(subset),paired_feeders=sorted(r["case"] for r in complete),methods=data)
    subsets={"all_verified_requested":set(names),"originally_time_suspect":set(names)&affected,
             "originally_closed_paired_control":set(names)-affected}
    subsets.update({s:{n for n in names if library.kind(n)==s} for s in ("small","medium","large","huge")})
    root_summary={}
    for scope in ("physical_first_two","identifiable_first_two","majority"):
        complete=[r for r in rows if r["case"] in finished]
        missing_count=sum(len(r["joint_nj_rnj_misses"][scope]) for r in complete)
        recovered_count=sum(len(r["new"]["nearby_recoveries"][scope]) for r in complete)
        after_warm=sum(scope in detail["root_scopes"] for r in complete
                       for detail in r["new"]["joint_miss_absent_at_first_warm_present_at_final"])
        root_summary[scope]=dict(paired_feeders=len(complete),joint_baseline_misses=missing_count,
                                 recovered_final=recovered_count,already_in_first_warm=recovered_count-after_warm,
                                 absent_from_first_warm_but_final=after_warm)
    missing=[name for name in selected() if not (library.LIBRARY/"dev"/(name+".pkl")).is_file()]
    metadata=dict(environment=environment,solver_threads=1,original_wall_s=120,original_settings=asdict(settings()),new_settings=asdict(new_settings),
        preregistered_feeders=len(selected()),original_sha_unverified=missing,
        comparison={k:score_panel(subset) for k,subset in subsets.items()},
        root_side_joint_misses=root_summary,
        selection="Seventeen originally timed-out or open-gap/lower-than-perfect-F1 jobs. With --all-verified, include the four previously gap-closed small control cases so all 21 originally verified inputs use the same solver settings.",
        timing="Each attempt records its own wall allowance and elapsed time. Parallel queues can wait for the same feeder lock, so attempt durations can overlap. final_call_elapsed_s measures the successful solver run after loading the input and acquiring its lock. Identical completed subproblems resume from checkpoints. The resistance estimate uses the verified numeric cache from the original comparison.",
        scope="Tree lower_H is the checked bound for that pool and its required anchors. Node objective/bound are the solver's raw per-observation criterion for its single-addition problem. The final single_line scope is the pricing procedure's checked result. Flip labels can use a different domain. Checkpoint trees before wall termination are feasible intermediates.",
        paired_node_ablation=False,
        attribution="Normal-run stage membership can show where a grouping first appeared. Raw MIQP incumbents and output after local cleanup are recorded separately. Without paired B/C arms, a difference between final trees is not attributed to wide-node pricing.")
    # One line per feeder keeps the published evidence readable without repeating array indentation.
    def compact(value):
        return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(",",":"))
    lines=["{",*(f'{compact(key)}:{compact(value)},' for key,value in metadata.items()),'"rows":[']
    lines.append(",\n".join(compact(row) for row in rows))
    lines.append("]}")
    out.parent.mkdir(parents=True,exist_ok=True)
    temporary=out.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines)+"\n",encoding="utf-8")
    temporary.replace(out)
    return rows


def extend_tree(name, outputs, seconds, extended, stage, case_dir):
    """Continue on one saved candidate domain from its best admissible recorded tree."""
    from ptid.method.model import Model,Fitter
    from ptid.method.miqp import tree_miqp
    from ptid.method.sets import lines_of,compatible
    from itertools import combinations
    arms=outputs/"topology_study_random_dev"/"arms"
    library.LIBRARY=outputs/"case_library_20261125"
    rec=library.open_case(name)
    old=json.loads((arms/f"{name}__proposed.json").read_text(encoding="utf-8"))
    control=old["run_contract"]
    st=replace(settings(),short_seconds=seconds,long_seconds=seconds,node_seconds=seconds,flip_seconds=seconds)
    samples=library.readings(rec)
    data={k:samples[k] for k in ("P","Q","Y","V0sq")}
    h=digest(data)
    if control!=contract(name,"proposed",120) or h!=json.loads((arms/f"{name}__nj.json").read_text(encoding="utf-8"))["history_sha256"]:
        raise ValueError(f"Input or source mismatch: {name}")
    folder=case_dir/sha256(json.dumps(dict(st=asdict(st),inputs=h,original=control),sort_keys=True).encode()).hexdigest()
    info=json.loads((case_dir/(name+".progress.json")).read_text(encoding="utf-8"))["steps"][stage]
    model=Model(data["P"],data["Q"],data["Y"],data["V0sq"],None,st.lam,p_s=st.p_s)
    desired=lines_of(info["tree"],model.leaves)
    saved=[]
    for f in folder.glob("*.pkl"):
        record=pickle.loads(f.read_bytes())
        if (isinstance(record,tuple) and len(record)==2 and isinstance(record[1],dict)
                and "domain" in record[1] and record[1]["pool"]==info["pool"]
                and abs(record[1]["crit"]*model.N/2-info["H"])<1e-6 and lines_of(record[0],model.leaves)==desired):
            saved.append(record[1]["domain"])
    if not saved:raise ValueError(f"No matching stored restricted domain for {name}")
    if "domain_sha256" in info:
        saved=[d for d in saved if sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()==info["domain_sha256"]]
    else:
        saved=[d for d in saved if stage!=0 or not d["require"]]
    # The first solve stores require=(); a subsequent solve with no anchors stores [].
    # Both encode the same constraints. Compare their serialized domains, including every line.
    if not saved or len({json.dumps(d,sort_keys=True) for d in saved})!=1:
        raise ValueError(f"Cannot unambiguously select the required domain on {name}")
    domain=saved[0]; allowed=set(map(tuple,domain["lines"])); required=domain["require"]
    fit=Fitter(model,domain["lines"])
    starts={"relaxed_tree":desired}
    if old.get("tree") is not None:starts["original_final"]=old["tree"]
    for i,r in enumerate(old.get("completed_restricted_solves") or []):starts[f"original_checkpoint_{i}"]=r["tree"]
    domain_hash=sha256(json.dumps(domain,sort_keys=True).encode()).hexdigest()
    for previous in sorted(case_dir.glob(f"{name}.fixed.{stage}.*.json")):
        earlier=json.loads(previous.read_text(encoding="utf-8"))
        if earlier["limit_s"]<extended and earlier["inputs"]==h and earlier["domain_sha256"]==domain_hash:
            starts[f"fixed_{earlier['limit_s']}s"]=earlier["tree"]
    feasible={}
    for key,tree in starts.items():
        lines=set(map(tuple,tree))
        if (lines<=allowed and all(any(tuple(t) in lines for t in group) for group in required)
                and all(compatible(a,b) for a,b in combinations(lines,2))):
            feasible[key]=float(fit.crit(lines))
    best=min(feasible,key=feasible.get)
    warm=lines_of(starts[best],model.leaves)
    t=perf_counter()
    ans=tree_miqp(model,domain["lines"],extended,require=required,seed=st.seed,warm=warm,known=feasible[best],
                  tighten_at=feasible[best] if st.tighten else None)
    tree=warm
    if ans["family"] is not None:
        candidate=lines_of(ans["family"],model.leaves)
        if fit.crit(candidate)<fit.crit(warm):tree=candidate
    value=fit.crit(tree); low=ans["bound"]
    if low is not None and low>value+1e-5*(1+abs(value)):raise ValueError(f"Conflicting fixed-domain bound: {name}")
    half=model.N/2
    result=dict(stage=stage,limit_s=extended,elapsed_s=perf_counter()-t,inputs=h,domain_sha256=domain_hash,
        pool=len(allowed),required_groups=len(required),
        admissible_starts_H={k:v*half for k,v in feasible.items()},warm_start=best,tree=tree,H=value*half,
        score=metrics(tree,library.truth(rec)),status=ans["status"],lower_H=None if low is None else min(low,value)*half,
        relaxed_lower_H=info["lower_H"],relaxed_H=info["H"],gap_closed=(low is not None and
        ans["status"] in (2,9) and value-min(value,low)<=max(1e-7,1e-4*abs(value))))
    dest=case_dir/f"{name}.fixed.{stage}.{extended}.json"
    save(dest,result)
    print(name,stage,"fixed-domain",result["warm_start"],result["H"],result["lower_H"],flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs",type=Path,required=True)
    ap.add_argument("--seconds",type=int,default=30,help="Per-call Tree, Node and flip solver limit")
    ap.add_argument("--wall",type=int,default=1200,help="Wall allowance per attempt; completed subproblems can resume")
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--jobs",nargs="+")
    ap.add_argument("--all-verified",action="store_true",help="Include upper-bound and gap-closed small controls: all 21 verified feeders")
    ap.add_argument("--child",action="store_true")
    ap.add_argument("--collect",action="store_true",help="Re-score and publish current paired records without solving")
    ap.add_argument("--fixed-seconds",type=int,help="Extra Tree solve on a saved domain, retaining the best compatible original or relaxed incumbent")
    ap.add_argument("--stage",type=int,default=0,help="Index of the saved Tree call for --fixed-seconds")
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    if min(args.seconds,args.wall,args.workers) < 1:
        ap.error("Solver time, wall time and worker count must be positive")
    outputs=args.outputs.resolve()
    arms=outputs/"topology_study_random_dev"/"arms"
    if not arms.is_dir() or not (outputs/"case_library_20261125"/"dev").is_dir():
        ap.error(f"Expected the original per-arm receipts in {arms} and records in {outputs/'case_library_20261125'/'dev'}")
    names=selections(arms,args.all_verified)
    if args.jobs:
        if not set(args.jobs)<=set(names):
            ap.error("Choose a verified affected feeder, or pass --all-verified for an originally closed control")
        names=args.jobs
    case_dir=outputs/"topology_study_random_dev"/f"budget_{args.seconds}s"
    case_dir.mkdir(exist_ok=True,parents=True)
    if args.collect:
        if not args.out:ap.error("--collect requires --out")
        collect(names,outputs,args.seconds,args.out)
        return
    if args.fixed_seconds:
        if not args.jobs or args.fixed_seconds<=args.seconds:ap.error("Give --jobs and a larger fixed-domain allowance")
        for name in names:
            extend_tree(name,outputs,args.seconds,args.fixed_seconds,args.stage,case_dir)
        return
    if args.child:
        if len(names)!=1:ap.error("One feeder per child")
        name=names[0]
        with feeder_lock(case_dir/(name+".lock")):
            if not (case_dir/(name+".json")).is_file():
                run_case(name,outputs,args.seconds,case_dir)
        return
    def one(name):
        out=case_dir/(name+".json")
        if not out.is_file():
            cmd=[sys.executable,"-B","-m","ptid.runs.random_budget","--child","--outputs",str(outputs),
                 "--seconds",str(args.seconds),"--jobs",name]
            if args.all_verified:cmd.append("--all-verified")
            t=perf_counter()
            error=None
            try:
                proc=subprocess.run(cmd,timeout=args.wall)
                if proc.returncode or not out.is_file():
                    raise RuntimeError(f"Child {proc.returncode}: {name}")
                state="ok"
            except subprocess.TimeoutExpired:
                state="wall_limit"
            except (RuntimeError,OSError) as exc:
                state="error"
                error=str(exc)
            elapsed=perf_counter()-t
            log=case_dir/(name+".attempts.json")
            past=json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
            save(log,[*past,dict(wall_s=args.wall,elapsed_s=elapsed,status=state,error=error)])
            print(name,state,round(elapsed,1),flush=True)
        return out.is_file()
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        rows=list(workers.map(one,names))
    if args.out:
        collect(names,outputs,args.seconds,args.out)
    return int(not all(rows))


if __name__ == "__main__":
    raise SystemExit(main())
