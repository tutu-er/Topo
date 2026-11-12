"""Zero-DR certificates conditional on a frozen historical target being correct.

No fitting, optimization, simulation, or future observations. The existing
loader supplies all .24/QP candidates and the unchanged device contract.
Spectral calculation is exactly the original fixed-Gaussian exponential bound.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import zipfile

import numpy as np
import scipy
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from research_experiments.wzzt_dr_selection.dr_pricing.mse_existing_cases import core


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                    allow_nan=False)+"\n", encoding="utf-8")


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def spectral_pair(data, competitor, temporal):
    """Target is data.current; return strict exclusion sufficient bounds."""
    target = data["current"]
    H1, H0 = data["H"][[target, competitor]]
    T, n = data["forecast"].shape[0], len(data["labels"])
    residual = data["history_drop"]-data["history_power"]@H1.T
    sd = np.maximum(np.sqrt(np.mean(residual**2, axis=0)),
                    2*np.sqrt(2)*data["voltage_sigma"])
    s = float(sd.max())
    if s <= 0:
        raise ValueError("Degenerate original remaining-voltage error")
    lam = .25
    t = lam*(1-2*lam*(sd/s)**2)
    sigma = .05*np.max(np.abs(data["forecast"]), axis=0)
    dh = H1-H0
    F = (dh[:, :n]*sigma[:n]+dh[:, n:]*sigma[n:])/s
    V, singular, _ = np.linalg.svd(np.sqrt(t)[:, None]*F, full_matrices=True)
    r, U = temporal
    eigen = 1+2*r[:, None]*singular[None, :]**2
    transformed = U.T@((data["forecast"]@dh.T/s)*np.sqrt(t))@V
    q = float(np.sum(transformed**2/eigen))
    half_logdet = float(.5*np.log(eigen).sum())
    total = data["history_drop"].size+T*n
    delta = float((total*core.STRICT_SCALED_EPSILON/data["K"]
                   -data["history_sse"][competitor]+data["history_sse"][target])/(s*s))
    score = q+half_logdet
    log_fail = min(0., lam*delta-score)
    model, other = data["models"][target], data["models"][competitor]
    same_family = model["point"]["supports"] == other["point"]["supports"]
    equal_response = bool(np.array_equal(H1, H0))
    # Equal predictions give equal scores forever. Original fixed tie ordering
    # then keeps the historical winner; no strict exclusion is necessary.
    if equal_response:
        if (other["internal"], competitor) < (model["internal"], target):
            raise ValueError("Historical winner conflicts with exact-response tie order")
        log_fail = -np.inf
    return dict(competitor=other["id"], same_family=bool(same_family),
        equal_response=equal_response, q=q, half_logdet=half_logdet,
        score=score, lambda_=lam, delta=delta, scale_pu2=s,
        historical_MSE_advantage=float(other["history_mse"]-model["history_mse"]),
        log_failure_upper=None if equal_response else log_fail,
        pair_success_lower=1. if equal_response else float(-np.expm1(log_fail)),
        pair_90_sufficient=bool(equal_response or log_fail <= np.log(.1)),
        strict_margin=score-lam*delta-np.log(10))


def joint(rows):
    logs = [r["log_failure_upper"] for r in rows if not r["equal_response"]]
    log_sum = float(logsumexp(logs)) if logs else -np.inf
    return dict(competitors=len(rows),
        log_union_failure_upper=None if not logs else min(0., log_sum),
        success_lower=1. if not logs else float(-np.expm1(min(0., log_sum))),
        sufficient_90=bool(log_sum <= np.log(.1)),
        union_log_margin=None if not logs else float(np.log(.1)-log_sum),
        unproved_pair_count=sum(not r["pair_90_sufficient"] for r in rows))


def dense_check(data, pair):
    """Independent old dense implementation; no control factor is needed."""
    view = dict(data, current=next(i for i, m in enumerate(data["models"])
                                  if m["id"] == pair["competitor"]),
                reference=data["current"])
    gate = core.build_gate(view, full=False)
    old_q = float(gate.offset@gate.offset)
    # Original gate.h includes -half_logdet, while spectral score includes it.
    old_margin = old_q-float(gate.h)
    error = abs(old_margin-pair["strict_margin"])
    if error > 1e-8*max(1., abs(old_margin)):
        raise AssertionError((old_margin, pair["strict_margin"], error))
    return dict(competitor=pair["competitor"], dense_margin=old_margin,
                spectral_margin=pair["strict_margin"], absolute_error=error)


def prepare(args, out):
    manifest, catalog = read(args.manifest), read(args.catalog)
    pq_sigma = float(read(args.source_config)["physics"]["pq_sigma"])
    eligible = [r for r in manifest["all48_inventory"]
        if r["selected_declared_errors"]["all_supports"]["errors"] == 0
        and r["full_repair_available_in_output_library"]
        and r["oracle_declared_errors"]["all_supports"]["errors"] == 0]
    if len(eligible) != 40:
        raise ValueError("Locked qualification requires exactly40 existing inputs")
    requests = []
    for row in eligible:
        selected, original = row["selected"], catalog[row["job"]]
        result = ref(selected["result"]["path"])
        if result["sha256"] != selected["result"]["sha256"]:
            raise ValueError("Fitted result changed: "+row["job"])
        # Worker receives no test path, true supports, truth-network path,
        # oracle model id, or model error counts.
        safe = {k: original[k] for k in ("case", "labels", "train", "validation",
                                        "voltage_sigma")}
        safe["pq_sigma"] = pq_sigma
        requests.append(dict(job=row["job"], case=row["case"], target=selected["model_id"],
            result=result, catalog=safe,
            history=[ref(safe["train"]), ref(safe["validation"])]))
    source_paths = [Path(__file__)]
    for folder in ("mse_existing_cases", "paper15_exponential",
                   "../dr_heterogeneous_pilot", "../dr_heterogeneous_resources"):
        source_paths.extend((HERE.parent/folder).resolve().rglob("*.py"))
    sources = {str(p.relative_to(REPO)): ref(p)["sha256"] for p in sorted(set(source_paths))}
    protocol = dict(id="conditional_zero_screen_v1", status="frozen_before_calculation",
        created_utc=datetime.now(timezone.utc).isoformat(), expected_cases=40,
        target="historical strict-MSE winner, conditional on this fixed target being correct",
        eligibility="previous offline truth manifest only; no truth fields sent to calculation",
        forecast="all96train+288validation rows; four observed days mean; newday96slots",
        candidates="all original .24-contracted/QP-refitted output points; parameters fixed",
        primary="family: fixed target point beats every different-support candidate",
        secondary="fixed parameter point: beats every other parameter point",
        probability="exponential pair bounds+union; sufficient, not exact selector probability",
        beta=.9, lambda_=.25, power_sd="original .05*channel max absolute forecast",
        power_dependence="same-node P/Q factor; independent nodes; temporal AR1 .7",
        voltage_sd="target historical node RMS or2sqrt2*voltage_sigma, whichever larger; independent",
        response="zero DR, hence no uncertain delivery term", fee=0,
        validation="first case in each feeder: worst different-family pair checked against original dense gate",
        seed=None, source_sha256=sources, manifest=ref(args.manifest), catalog=ref(args.catalog),
        original_configuration=ref(args.source_config),
        allocation=ref(args.allocation), resource_library=ref(args.library),
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
        forbidden="no test/newfuture, refit, optimizer, AC, new topology or parameter changes")
    write(out/"requests.json", requests)
    write(out/"protocol.json", protocol)
    with zipfile.ZipFile(out/"source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(set(source_paths)):
            archive.write(p, p.relative_to(REPO))
    return requests


def run(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/"complete.json").exists():
        print(json.dumps(read(out/"summary.json"), ensure_ascii=False)); return
    lock = out/"run.lock"
    fd = os.open(lock, os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        if (out/"protocol.json").exists():
            protocol = read(out/"protocol.json")
            for p, expected in protocol["source_sha256"].items():
                if ref(REPO/p)["sha256"] != expected:
                    raise ValueError("Source drift on resume: "+p)
            requests = read(out/"requests.json")
        else:
            requests = prepare(args, out)
        allocations, library = read(args.allocation)["cases"], read(args.library)
        allocation = {r["case"]: r for r in allocations}
        R = .7**np.abs(np.arange(96)[:, None]-np.arange(96)[None, :])
        temporal = np.linalg.eigh(R)
        results = read(out/"rows.json") if (out/"rows.json").exists() else []
        done = {r["job"] for r in results}
        audited = {r["case"] for r in results if r.get("dense_audit")}
        for request in requests:
            if request["job"] in done:
                continue
            start = time.perf_counter()
            row = dict(job=request["job"], case=request["case"], target=request["target"])
            try:
                d = core.load_history_case(request["result"]["path"], request["catalog"],
                                           allocation[request["case"]], library)
                if request["target"] not in d["models"][d["current"]]["model_ids"]:
                    raise ValueError("Target is not original historical strict-MSE winner")
                pairs = [spectral_pair(d, j, temporal) for j in range(len(d["models"]))
                         if j != d["current"]]
                different = [p for p in pairs if not p["same_family"]]
                row.update(status="complete", models=len(d["models"]),
                    train_rows=len(d["train_power"]), validation_rows=len(d["history_power"]),
                    voltage_sigma=d["voltage_sigma"], family=joint(different),
                    point=joint(pairs), pairs=pairs)
                if request["case"] not in audited and different:
                    row["dense_audit"] = dense_check(d, min(different,
                        key=lambda p: p["pair_success_lower"]))
                    audited.add(request["case"])
            except Exception:
                row.update(status="failed", error=traceback.format_exc())
            row["seconds"] = time.perf_counter()-start
            results.append(row)
            write(out/"rows.json", results)
            write(out/"checkpoint.json", dict(returned=len(results), expected=40))
            print(json.dumps({k: row[k] for k in ("job", "status", "seconds")}), flush=True)
        good = [r for r in results if r["status"] == "complete"]
        summary = dict(expected=40, returned=len(results), complete=len(good),
            failures=len(results)-len(good),
            family_90=sum(r["family"]["sufficient_90"] for r in good),
            point_90=sum(r["point"]["sufficient_90"] for r in good),
            exact_equal_response_pairs=sum(p["equal_response"] for r in good for p in r["pairs"]),
            pairs=sum(len(r["pairs"]) for r in good),
            different_family_pairs=sum(r["family"]["competitors"] for r in good),
            by_case={name: dict(complete=sum(r["case"]==name for r in good),
                family_90=sum(r["case"]==name and r["family"]["sufficient_90"] for r in good))
                for name in sorted({r["case"] for r in results})},
            total_seconds=sum(r["seconds"] for r in results),
            new_fits=0, optimization_calls=0, new_AC=0, random_draws=0)
        write(out/"summary.json", summary)
        write(out/"complete.json", dict(status="complete", returned=len(results), expected=40))
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        lock.unlink()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--source-config", required=True)
    p.add_argument("--allocation", default=str(HERE.parent.parent/"dr_heterogeneous_resources/case_allocations.json"))
    p.add_argument("--library", default=str(HERE.parent.parent/"dr_heterogeneous_resources/library.json"))
    run(p.parse_args())
