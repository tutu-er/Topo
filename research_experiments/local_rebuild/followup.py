"""Frozen small matched-domain follow-up, including unsuccessful comparisons."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import scipy

from .run import ROOT, clade_score, from_observed, scenes, sha, write, core
from .model import build_model, canonical, compatible, frontiers, prediction_loss, solve_rebuild, unions
from .matched import FiniteLP, all_hierarchies, gate, rnj_initial, single_closure
from .diagnose import diagnose, select_points


def random_truth(n, rng):
    blocks = [(i,) for i in range(n)]
    family = list(blocks)
    while len(blocks) > 1:
        i, j = sorted(rng.choice(len(blocks), 2, replace=False))
        merged = tuple(sorted([*blocks[i], *blocks[j]]))
        blocks = [s for k, s in enumerate(blocks) if k not in (i, j)] + [merged]
        family.append(merged)
    return canonical(family)


def small_data(seed, sigma, output):
    n = 5
    rng = np.random.default_rng(2026103000 + seed)
    truth = random_truth(n, rng)
    rr, xx = rng.uniform(.02, .2, (2, len(truth)))
    R, X = core.build_matrices_from_atoms(n, truth, rr, xx)
    prepared, arrays = [], {}
    for split, t in (("train", 12), ("validation", 64), ("test", 128)):
        common = rng.uniform(.008, .02, (t, 1))
        p = .4 * common + .6 * rng.uniform(.005, .025, (t, n))
        q = p * rng.uniform(.25, .75, (t, n))
        clean_y = p @ R.T + q @ X.T
        v = np.sqrt(1. - clean_y)
        measured_p = p * (1 + .005 * rng.normal(size=p.shape))
        measured_q = q * (1 + .005 * rng.normal(size=q.shape))
        measured_root = 1 + sigma * rng.normal(size=(t, 1))
        measured_v = v * (1 + sigma * rng.normal(size=v.shape))
        y = measured_root ** 2 - measured_v ** 2
        prepared.append(scenes(measured_p, measured_q, y))
        arrays.update({split+"_P":measured_p, split+"_Q":measured_q, split+"_Y":y})
    np.savez_compressed(output, **arrays)
    return prepared, truth


def scores(point, splits, truth):
    return {**clade_score(point, splits[0].n, truth), **{name:prediction_loss(data, point["supports"],
            point["r"], point["x"]) for name, data in zip(("train", "validation", "test"), splits)}}


def landscape(output):
    """All 236 starts are examined, without choosing a start by its true F1."""
    rows = []
    n = 5
    mandatory = canonical([*((i,) for i in range(n)), tuple(range(n))])
    local = [s for _, s in unions(tuple((i,) for i in range(n)))]
    domain = canonical([*mandatory, *local])
    for seed in range(3):
        for sigma in (.0002, .001):
            ident = f"n5_seed{seed}_v{sigma:g}"
            row = dict(id=ident, seed=seed, voltage_sigma=sigma, status="failed")
            try:
                splits, truth = small_data(seed, sigma, output / (ident+".npz"))
                train, validation, test = splits
                bank = FiniteLP(train, domain, mandatory=mandatory)
                oracle = [bank.fit([*mandatory, *f]) for f in all_hierarchies(n)]
                best = min(oracle, key=lambda p:(p["objective"], len(p["supports"]), p["supports"]))
                # Only training loss chooses the first witness and all stops.
                starts, witnesses, mathematical = [], [], []
                for i, start in enumerate(oracle):
                    fitted = single_closure(bank, start["supports"], relative=.001)
                    endpoint = fitted["point"]
                    gap = endpoint["objective"] - best["objective"]
                    unlocked = fitted["stationary"] and gap > gate(endpoint["objective"])
                    last = fitted["sweeps"][-1]
                    entry = dict(start_index=i, endpoint=endpoint, joint_gain=gap,
                                 best_single_gain=last.get("best_gain"), unlocked=unlocked)
                    starts.append(entry)
                    if unlocked:
                        witnesses.append(entry)
                        if last.get("best_gain", 0.) <= 1e-8:
                            mathematical.append(entry)
                initial = rnj_initial(train)
                rnj_fit = bank.fit(initial["family"])
                rnj_single = single_closure(bank, rnj_fit["supports"], relative=.001)
                joint = solve_rebuild(train, rnj_fit["supports"], range(n), tuple((i,) for i in range(n)),
                                      hard=mandatory, seconds=5.)
                comparison = dict(RNJ=scores(rnj_fit, splits, truth), single=scores(rnj_single["point"],splits,truth),
                                  joint=scores(best,splits,truth))
                witness = mathematical[0] if mathematical else witnesses[0] if witnesses else None
                if witness:
                    witness = dict(witness, single_score=scores(witness["endpoint"],splits,truth),
                                   joint_score=scores(best,splits,truth),
                                   kind="nonpositive_single_gain" if witness in mathematical else "gain_threshold_stall")
                # A predeclared penalty grid uses validation only. These six
                # datasets remain a pilot, including every unfavorable row.
                penalties = []
                for relative_lambda in (0., .001, .005, .01):
                    lam = relative_lambda * rnj_fit["objective"]
                    selected = min(oracle, key=lambda p:(p["objective"] + lam *
                        sum(1 < len(s) < n for s in p["supports"]), len(p["supports"]), p["supports"]))
                    penalties.append(dict(relative_lambda=relative_lambda, lambda_value=lam,
                                          point=selected, score=scores(selected,splits,truth)))
                choice = min(range(len(penalties)), key=lambda i:(penalties[i]["score"]["validation"],
                            len(penalties[i]["point"]["supports"]), i))
                # Truth is consulted only in the evaluators above, after all
                # training fits and witness selection; it does not affect paths.
                row.update(status="completed", domain=domain, hierarchy_count=len(oracle),
                    data_sha256=sha(output/(ident+".npz")), truth=truth,
                    initial=initial, rnj_single=rnj_single, joint_MILP=joint,
                    enumeration_best=best, MILP_vs_enumeration=None if joint["point"] is None else
                    joint["point"]["objective"]-best["objective"],
                    comparison=comparison, starts=starts, stationary_trap_starts=len(witnesses),
                    nonpositive_gain_trap_starts=len(mathematical), first_witness=witness,
                    penalties=penalties, validation_selected_penalty=choice,
                    LP_certificates=bank.solves, truth_used_by_search=False)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            write(output/"landscape.json",rows)
            print(ident,row["status"], 'traps',row.get("stationary_trap_starts"),
                  'nonpositive',row.get("nonpositive_gain_trap_starts"),row.get("error",""),flush=True)
    return rows


def real_regions(output):
    """Fresh bundle 1: two structure-selected five-block regions per network."""
    parent=ROOT/"outputs/wzzt_paper_20261029_full/jobs"
    rows=[]
    for case in ("paper15","soumalas11","flynn16","pengwah18"):
        ident=f"main__{case}__b1__reference__v0005__N48"
        directory=parent/ident
        row=dict(id=ident,case=case,status="failed")
        try:
            payload=json.loads((directory/"payload.json").read_text(encoding="utf-8"))
            saved=json.loads((directory/"result.json").read_text(encoding="utf-8"))
            train=from_observed(payload["train"],48)
            # This is the actual deployment location: the selected raw family
            # BEFORE contraction, not the already-contracted final output.
            bank_points=saved["base_points"]+saved["candidates"]["ridge_union"]
            eligible=[p for p in bank_points if p["validation_mae"]<=min(q["validation_mae"] for q in bank_points)*1.03+1e-12]
            initial=min(eligible,key=lambda p:(len(p["supports"]),p["validation_mae"]))
            candidates=sorted([(v,b) for v,b in frontiers(initial["supports"],train.n,max_blocks=5)
                               if len(b)==5],key=lambda pair:(-len(pair[0]),pair[0],pair[1]))
            selected=[]
            for v,b in candidates:
                if not any(v==other[0] for other in selected):selected.append((v,b))
                if len(selected)==2:break
            searches=[]
            for v,blocks in selected:
                _,meta=build_model(train,initial["supports"],v,blocks)
                keep=meta["keep"]
                allowed=[s for s in meta["candidates"] if all(compatible(s,t) for t in keep)]
                domain=canonical([*keep,*allowed])
                lp=FiniteLP(train,domain,mandatory=keep,seconds=3.)
                start=lp.fit(initial["supports"])
                single=single_closure(lp,start["supports"],relative=.001,seconds=20.)
                # Rebuild starts at the single-move endpoint. A better joint
                # solution now specifically establishes an unlocked stop.
                joint=solve_rebuild(train,single["point"]["supports"],v,blocks,hard=keep,seconds=5.)
                gain=None if joint["point"] is None else single["point"]["objective"]-joint["point"]["objective"]
                searches.append(dict(region=v,blocks=blocks,domain=domain,mandatory=keep,start=start,
                    single=single,joint=joint,joint_gain=gain,
                    unlocked=bool(single["stationary"] and gain is not None and gain>gate(single["point"]["objective"])),
                    LP_certificates=lp.solves))
                print(case,'region',v,'stop',single["stop_reason"],'gain',gain,flush=True)
            # Independent evaluation begins after every training region ends.
            validation,test=from_observed(payload["validation"]),from_observed(payload["test"])
            from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
            labels=saved["labels"];positions={v:i for i,v in enumerate(labels)}
            truth=canonical([tuple(positions[v] for v in c) for c in physical_truth(Path(payload["truth_network"]),labels)["clades"]])
            splits=(train,validation,test)
            for search in searches:
                points=[search["start"],*search["single"]["path"][1:]]
                joint=search["joint"]["point"]
                search["scores"]=dict(start=scores(search["start"],splits,truth),single=scores(search["single"]["point"],splits,truth),
                                        joint=None if joint is None else scores(joint,splits,truth))
                paths={"single":points,"with_joint":points+([joint] if search["unlocked"] else [])}
                pool=canonical(s for path in paths.values() for p in path for s in p["supports"])
                refit=FiniteLP(train,pool,mandatory=[(i,) for i in range(train.n)])
                policies={}
                for method,path in paths.items():
                    for order in ("pre","post_refit"):
                        choice=select_points(path,validation,saved["calibration"],order=order,refit_bank=refit)
                        policies[method+"_"+order]=dict(choice=choice,score=scores(choice["point"],splits,truth))
                search["policies"]=policies
                search["refit_LP_certificates"]=refit.solves
            row.update(status="completed",initial=initial,searches=searches,truth=truth,
                       data_sha256={s:sha(payload[s]) for s in ("train","validation","test")},
                       parent_result_sha256=sha(directory/"result.json"),truth_used_by_search=False)
        except Exception as exc:
            row["error"]=f"{type(exc).__name__}: {exc}"
        rows.append(row);write(output/"real_matched.json",rows)
        print(case,row["status"],row.get("error",""),flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("--phase",choices=("all","small","real","diagnosis"),default="all")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    protocol=args.output/("protocol_"+args.phase+".json")
    if protocol.exists():raise FileExistsError("a frozen phase already exists")
    source_names=("matched.py","followup.py","diagnose.py","model.py","run.py")
    write(protocol,dict(created_utc=datetime.now(timezone.utc).isoformat(),phase=args.phase,
        python=sys.version,scipy=scipy.__version__,numpy=np.__version__,platform=platform.platform(),
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        sources={n:sha(Path(__file__).with_name(n)) for n in source_names},
        settings=dict(small_datasets=6,n=5,all_starts=236,train=12,validation=64,test=128,
            seeds=[0,1,2],voltage_sigmas=[.0002,.001],pq_sigma=.005,
            common_relative_gate=.001,absolute_gate=1e-10,bound=2.,LP_seconds=3.,MILP_seconds=5.,
            real_bundle=1,real_cases=4,real_regions_per_case=2,real_blocks=5,
            real_region_rule="largest region then lexicographic, distinct regions; no truth selection",
            real_start="existing 3pct selected raw bank BEFORE contraction",
            penalty_grid=[0.,.001,.005,.01],selection="validation only, test evaluation afterwards"),
        command="python -m research_experiments.local_rebuild.followup "+" ".join(sys.argv[1:])))
    snapshot=args.output/("source_phase1" if args.phase=="all" else "source_"+args.phase)
    snapshot.mkdir(exist_ok=False)
    for name in source_names:
        (snapshot/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    started=perf_counter()
    if args.phase in ("all","diagnosis"):
        destination=args.output/"diagnosis_refit";destination.mkdir(exist_ok=False)
        diagnose(destination,ROOT/"outputs/local_tree_rebuild_20261030_final")
    if args.phase in ("all","small"):landscape(args.output)
    if args.phase in ("all","real"):real_regions(args.output)
    write(args.output/("completion_"+args.phase+".json"),dict(seconds=perf_counter()-started))


if __name__=="__main__":main()
