"""Development experiment driver (pre-lock; not Stage M evidence).

    python -m rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments build <condition>
    python -m rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments run <condition> [--paths 4]

Inputs and trajectory records go to outputs/hourly_probing_dev_20261120/<condition>/
(ignored by Git).
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")   # one BLAS thread per worker; launch with these set too

import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
import traceback

import numpy as np

from . import bank, config, run
from . import scenario as sc

ROOT = Path(__file__).resolve().parents[3]/"outputs"/"hourly_probing_dev_20261120"

SEEDS = {"soumalas11": [(s, (96, 48)) for s in range(1, 25)]+[(s, (48, 24)) for s in range(101, 113)],
         "pengwah18": [(s, (96, 48)) for s in range(1, 17)]+[(s, (48, 24)) for s in range(101, 109)]}

SEEDS_1D = {c: [(s, (96, 48)) for s in range(1, 17)] for c in ("soumalas11", "pengwah18", "flynn16", "paper15")}
SEEDS_CONF = {c: [(s, (96, 48)) for s in range(17, 33)] for c in ("soumalas11", "pengwah18", "flynn16", "paper15")}
SEEDS_HARD = {c: [(s, (96, 48)) for s in range(33, 65)] for c in ("soumalas11", "pengwah18", "flynn16", "paper15")}
SEEDS_LOCK = {c: [(s, (96, 48)) for s in range(65, 97)] for c in ("soumalas11", "pengwah18", "flynn16", "paper15")}

# hard-case test (DEVELOPMENT_RUNS section 16): refined pipeline, linear DR prices, budget and V re-scaled
HARD = dict(meter_bias_sigma=.0005, meter_offsets=True, trust_gate=True, root_whitening=True,
            cost_model="linear", price_down_cny=4., price_up_cny=1.2,
            hourly_cap_cny=20., daily_cap_cny=100., cert_value_cny=1000.)

STRESS_SEEDS = {"soumalas11": [(s, (96, 48)) for s in range(1, 5)]+[(s, (48, 24)) for s in range(101, 113)],
                "pengwah18": [(s, (96, 48)) for s in range(1, 7)]+[(s, (48, 24)) for s in range(101, 109)]}

CONDITIONS = {
    "main": dict(settings={}, rebuild=True),
    "unbalanced": dict(settings=dict(unbalanced=True), rebuild=True),
    "meter_bias": dict(settings=dict(meter_bias_sigma=.001), rebuild=True),
    "shortline": dict(settings={}, rebuild=True, short_line=1.5*.003),
    "low_adoption": dict(settings=dict(battery_share=.10, ev_share=.05), rebuild=False),
    "rmin_records": dict(settings=dict(r_min_ohm=.01), rebuild=False, refit_floor=True),
    "lead0": dict(settings=dict(gate_lead_hours=0), rebuild=False),
    "tail1": dict(settings=dict(tail_order=1.), rebuild=False),
    # 2026-11-20 follow-up: single phase only, typical meter bias 0.05 %, revised step 1
    "bias_typical": dict(seeds="1d", settings=dict(meter_bias_sigma=.0005), rebuild=True),
    "v2_main": dict(seeds="1d", settings=dict(meter_offsets=True), rebuild=True, step1="v2"),
    "v2_bias_typical": dict(seeds="1d", settings=dict(meter_bias_sigma=.0005, meter_offsets=True), rebuild=True, step1="v2"),
    "v2_bias_stress": dict(seeds="1d", settings=dict(meter_bias_sigma=.001, meter_offsets=True), rebuild=True, step1="v2"),
    "v2_shortline": dict(seeds="1d", settings=dict(meter_bias_sigma=.0005, meter_offsets=True), rebuild=True, step1="v2",
                         short_line=.005),
    # development variant: best-effort targeted planner (no union-bound tail), main-case inputs reused
    "v2_besteffort": dict(seeds="1d", settings=dict(meter_bias_sigma=.0005, meter_offsets=True, tail_guarantee=False),
                          rebuild=False, base="v2_bias_typical"),
    "v2_shortline_besteffort": dict(seeds="1d", settings=dict(meter_bias_sigma=.0005, meter_offsets=True,
                                                              tail_guarantee=False),
                                    rebuild=False, base="v2_shortline"),
    # head-to-head comparison of the certified planner (DEVELOPMENT_RUNS section 13), arms config.CERT_ARMS
    "cmp_main": dict(seeds="1d", settings=dict(meter_bias_sigma=.0005, meter_offsets=True, trust_gate=True),
                     rebuild=False, base="v2_bias_typical"),
    "cmp_short01": dict(seeds="1d", settings=dict(meter_bias_sigma=.0005, meter_offsets=True, trust_gate=True),
                        rebuild=True, step1="v2", short_line=.01),
    "cmp_lowflex": dict(seeds="1d", settings=dict(meter_bias_sigma=.0005, meter_offsets=True, trust_gate=True,
                                                  battery_share=.10, ev_share=.05),
                        rebuild=False, base="v2_bias_typical"),
    "cmp_highflex": dict(seeds="1d", settings=dict(meter_bias_sigma=.0005, meter_offsets=True, trust_gate=True,
                                                   battery_share=.50, ev_share=.40, enrolled_share=.50),
                         rebuild=False, base="v2_bias_typical"),
    # development round 2 (2026-11-22): GLS scoring for the shared root-meter error, strong greedy
    "dev_main": dict(seeds="1d", settings=dict(meter_bias_sigma=.0005, meter_offsets=True, trust_gate=True,
                                               root_whitening=True), rebuild=False, base="v2_bias_typical"),
    "dev_short01": dict(seeds="1d", settings=dict(meter_bias_sigma=.0005, meter_offsets=True, trust_gate=True,
                                                  root_whitening=True), rebuild=False, base="cmp_short01"),
    # confirmation on fresh seeds 17-32 (pre-registered in DEVELOPMENT_RUNS section 15), full refined pipeline
    "conf_main": dict(seeds="conf", settings=dict(meter_bias_sigma=.0005, meter_offsets=True, trust_gate=True,
                                                  root_whitening=True), rebuild=True, step1="v2"),
    "conf_short01": dict(seeds="conf", settings=dict(meter_bias_sigma=.0005, meter_offsets=True, trust_gate=True,
                                                     root_whitening=True), rebuild=True, step1="v2", short_line=.01),
    # baseline pipeline on the same fresh seeds: first-round bank, ordinary least squares, no offsets
    "conf_base_main": dict(seeds="conf", settings=dict(meter_bias_sigma=.0005), rebuild=True),
    "conf_base_short01": dict(seeds="conf", settings=dict(meter_bias_sigma=.0005), rebuild=True, short_line=.01),
    # hard cases on new seeds 33-64, ambiguous (S1) inputs only (pre-registered in section 16)
    "hard_short01": dict(seeds="hard", settings=dict(HARD), rebuild=True, step1="v2", short_line=.01),
    "hard_short005": dict(seeds="hard", settings=dict(HARD), rebuild=True, step1="v2", short_line=.005),
    "hard_noisy": dict(seeds="hard", settings=dict(HARD, meter_bias_sigma=.001, voltage_sigma=.001),
                       rebuild=True, step1="v2"),
    # confirmation of the lock on fresh seeds 65-96, natural ambiguous inputs and their truth-removed
    # copies (pre-registered in DEVELOPMENT_RUNS section 18); settings identical to the hard cases
    "lock_short01": dict(seeds="lock", settings=dict(HARD), rebuild=True, step1="v2", short_line=.01),
    "lock_short005": dict(seeds="lock", settings=dict(HARD), rebuild=True, step1="v2", short_line=.005),
    "lock_noisy": dict(seeds="lock", settings=dict(HARD, meter_bias_sigma=.001, voltage_sigma=.001),
                       rebuild=True, step1="v2"),
}


def settings_for(condition):
    return config.Settings().with_(**CONDITIONS[condition]["settings"])


def _build_one(condition, case, seed, history):
    try:
        cond = CONDITIONS[condition]
        s = settings_for(condition)
        out = ROOT/condition/"inputs"
        out.mkdir(parents=True, exist_ok=True)
        path = out/f"{case}_s{seed}.pkl"
        if path.exists():
            inp = bank.load(path)
        else:
            inp = bank.build(case, seed, s, history=history, short_line=cond.get("short_line"),
                             step1=cond.get("step1", "v1"))
            bank.save(inp, path)
        return dict(name=inp.name, case=case, seed=seed, history=list(history), **inp.meta["passive"],
                    bank=inp.meta["bank"], identifier_bank=inp.meta["identifier_bank"],
                    identifier_bank_has_truth=inp.meta["identifier_bank_has_truth"],
                    validation_mse_correct=inp.meta["validation_mse_correct"],
                    truth_in_bank=inp.truth_in_bank, short_line=inp.meta.get("short_line"))
    except Exception as exc:
        return dict(name=f"{case}_s{seed}", error=repr(exc), trace=traceback.format_exc())


def derive(condition, base=None):
    """Conditions that reuse another condition's inputs with other resources, floors or planner settings."""
    base = base or CONDITIONS[condition].get("base", "main")
    s = settings_for(condition)
    src, dst = ROOT/base/"inputs", ROOT/condition/"inputs"
    dst.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(src.glob("*.pkl")):
        inp = bank.load(path)
        if inp.name.endswith("_missing"):
            continue
        inp = copy.copy(inp)
        inp.res = sc.resources(s, inp.hist["P"].mean(axis=0)*inp.base_kw, inp.base_kw)
        if CONDITIONS[condition].get("refit_floor"):
            zbase = inp.network["base_kv"]**2/inp.network["base_mva"]
            fits, J, _ = bank.passive_fit(inp.families, inp.n, inp.hist, 2*s.r_min_ohm/zbase, s.upper_bound)
            inp.theta0 = [f.theta.copy() for f in fits]
            inp.meta = dict(inp.meta); inp.meta["passive"] = bank.classify(inp, s, J)
        bank.save(inp, dst/f"{inp.name}.pkl")
        rows.append(dict(name=inp.name, case=inp.case, seed=inp.seed, **inp.meta["passive"]))
    return rows


def build(condition, workers=14, seeds=None):
    cond = CONDITIONS[condition]
    (ROOT/condition).mkdir(parents=True, exist_ok=True)
    if not cond["rebuild"]:
        rows = derive(condition)
    else:
        jobs = []
        default = {"1d": SEEDS_1D, "conf": SEEDS_CONF, "hard": SEEDS_HARD, "lock": SEEDS_LOCK}.get(cond.get("seeds"), SEEDS)
        for case, lst in (seeds or default).items():
            for seed, hist in lst:
                jobs.append((condition, case, seed, tuple(hist)))
        rows = []
        with ProcessPoolExecutor(workers) as pool:
            futures = [pool.submit(_build_one, *j) for j in jobs]
            for f in as_completed(futures):
                rows.append(f.result()); print(json.dumps({k: rows[-1].get(k) for k in ("name", "stratum", "error")}),
                                               flush=True)
    rows.sort(key=lambda r: (r.get("case", ""), r.get("seed", 0)))
    (ROOT/condition/"inputs.json").write_text(json.dumps(rows, indent=1, default=str))
    return rows


def select(condition, per_stratum=8, s3_from_drop=True):
    """Up to `per_stratum` inputs per stratum and network, ascending seed (PROTOCOL section 3)."""
    s = settings_for(condition)
    rows = [r for r in json.loads((ROOT/condition/"inputs.json").read_text()) if "error" not in r]
    chosen = []
    for case in dict.fromkeys(r["case"] for r in rows):
        for stratum in ("S1", "S2", "S3"):
            pick = [r for r in rows if r["case"] == case and r["stratum"] == stratum][:per_stratum]
            chosen += [r["name"] for r in pick]
        if s3_from_drop:
            have = sum(1 for r in rows if r["case"] == case and r["stratum"] == "S3")
            donors = [r for r in rows if r["case"] == case and r["stratum"] in ("S1", "S2")]
            donors.sort(key=lambda r: (r["stratum"] != "S1", r["seed"]))
            for r in donors[:max(per_stratum-have, 0)]:
                path = ROOT/condition/"inputs"/f"{r['name']}_missing.pkl"
                if not path.exists():
                    inp = bank.drop_truth(bank.load(ROOT/condition/"inputs"/f"{r['name']}.pkl"), s)
                    bank.save(inp, path)
                chosen.append(r["name"]+"_missing")
    return chosen


def missing_copies(condition, names):
    """Truth-removed copies of the given natural inputs (created if needed); returns their names."""
    s = settings_for(condition)
    out = []
    for nm in names:
        path = ROOT/condition/"inputs"/f"{nm}_missing.pkl"
        if not path.exists():
            bank.save(bank.drop_truth(bank.load(ROOT/condition/"inputs"/f"{nm}.pkl"), s), path)
        out.append(nm+"_missing")
    return out


def _run_input(condition, name, arms, paths):
    s = settings_for(condition)
    inp = bank.load(ROOT/condition/"inputs"/f"{name}.pkl")
    recs = []
    for path in range(paths):
        base = None
        for arm in arms:
            if arm in ("frozen_replay", "random_matched_P", "qbc_matched_P"):
                if base is None:
                    continue
                rec = run.run(inp, arm, path, s, replay=base["schedule"])
            else:
                rec = run.run(inp, arm, path, s)
                if arm in ("t_opt_P", "cert_P") and base is None:
                    base = rec                     # matched arms replay the first planner's hours and costs
            rec["condition"] = condition
            recs.append(rec)
    return recs


def run_condition(condition, arms=config.ARMS, paths=4, workers=14, names=None, tag="runs"):
    names = names or select(condition)
    out = ROOT/condition/f"{tag}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line); done.add(r["input"])
    todo = [n for n in names if n not in done]
    started = time.time()
    with ProcessPoolExecutor(workers) as pool, out.open("a") as fh:
        futures = {pool.submit(_run_input, condition, n, tuple(arms), paths): n for n in todo}
        for f in as_completed(futures):
            try:
                for rec in f.result():
                    for key in ("gates",):
                        rec[key] = [{k: v for k, v in g.items()} for g in rec[key]]
                    fh.write(json.dumps(rec, default=_json)+"\n")
                fh.flush()
                print(f"{futures[f]} done {time.time()-started:.0f}s", flush=True)
            except Exception:
                print(f"{futures[f]} FAILED\n{traceback.format_exc()}", flush=True)


def _json(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(type(value))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("build", "run", "select"))
    ap.add_argument("condition", choices=sorted(CONDITIONS))
    ap.add_argument("--paths", type=int, default=4)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--arms", default=",".join(config.ARMS))
    ap.add_argument("--per-stratum", type=int, default=8)
    ap.add_argument("--subset", choices=("all", "stress"), default="all")
    ap.add_argument("--strata", default="S1,S2,S3", help="run only inputs of these strata")
    ap.add_argument("--no-missing", action="store_true", help="skip the truth-removed copies (natural strata only)")
    ap.add_argument("--tag", default="runs", help="results file <tag>.jsonl (separate files for sweeps)")
    ap.add_argument("--truth-missing", action="store_true",
                    help="run the truth-removed copy of every selected natural input instead of the input itself")
    a = ap.parse_args()
    if a.command == "build":
        build(a.condition, a.workers, STRESS_SEEDS if a.subset == "stress" else None)
    elif a.command == "select":
        print(select(a.condition, a.per_stratum))
    else:
        names = select(a.condition, a.per_stratum)
        keep = set(a.strata.split(","))
        strata = {r["name"]: r["stratum"] for r in json.loads((ROOT/a.condition/"inputs.json").read_text())
                  if "error" not in r}
        names = [nm for nm in names if (strata.get(nm) if not nm.endswith("_missing") else "S3") in keep]
        if a.no_missing or a.truth_missing:
            names = [nm for nm in names if not nm.endswith("_missing")]
        if a.truth_missing:
            names = missing_copies(a.condition, names)
        run_condition(a.condition, a.arms.split(","), a.paths, a.workers, names, tag=a.tag)


if __name__ == "__main__":
    main()
