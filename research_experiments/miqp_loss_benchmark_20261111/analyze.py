"""Independent, offline audit of the isolated L1 versus scaled-MSE experiment.

This module does not import Gurobi or the native optimizer.  NumPy replays
saved full models, and SciPy solves bounded least-squares problems solely as
post-analysis checks.  Truth and test observations are opened only here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from itertools import combinations
import json
import os
from pathlib import Path
from time import perf_counter

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("noT", "T003")
VIEWS = ("primary_mae", "secondary_mse")
METADATA = ("panel", "case", "bundle", "seed_index", "N", "voltage_sigma", "condition")
QP_ATOL = 2e-8
QP_RTOL = 2e-5
REPLAY_ATOL = 1e-9


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    # Match the portable frozen-pool hash convention, including ASCII escaping.
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                    encoding="utf-8", newline="\n")


def close(a, b, *, atol=REPLAY_ATOL, rtol=1e-7, message="Values differ"):
    a, b = float(a), float(b)
    if not np.isfinite(a) or not np.isfinite(b) or abs(a-b) > atol+rtol*max(abs(a), abs(b)):
        raise ValueError(f"{message}: {a!r} versus {b!r}")


def canonical(supports):
    return tuple(sorted({tuple(map(int, C)) for C in supports}, key=lambda C: (len(C), C)))


def load_split(path):
    with np.load(path, allow_pickle=False) as archive:
        labels = archive["labels"].astype(int).tolist(); n = len(labels)
        raw = [np.asarray(archive[k], float) for k in ("P", "Q", "V")]
        if not all(a.ndim == 3 and a.shape[1:] == (96, n) for a in raw):
            raise ValueError("Expected complete 96-slot days.")
        p, q, voltage = [a.reshape(-1, n) for a in raw]
        root = np.asarray(archive["root"], float).reshape(-1)
    y = root[:, None]**2-voltage**2
    if p.shape != q.shape or p.shape != y.shape or not all(np.isfinite(a).all() for a in (p, q, y)):
        raise ValueError("Invalid saved observations.")
    return dict(p=p, q=q, y=y, labels=labels, n=n, rows=len(p))


def family(point, n):
    supports = tuple(tuple(map(int, C)) for C in point["supports"])
    if len(supports) != len(set(supports)) or any(
            not C or C != tuple(sorted(set(C))) or any(i < 0 or i >= n for i in C) for C in supports):
        raise ValueError("Malformed or repeated support.")
    for C, D in combinations(map(set, supports), 2):
        if C & D and not (C <= D or D <= C):
            raise ValueError("A saved full model is not laminar.")
    r, x = np.asarray(point["r"], float), np.asarray(point["x"], float)
    if r.shape != x.shape or r.shape != (len(supports),) or not np.isfinite(np.r_[r, x]).all():
        raise ValueError("Invalid response coefficients.")
    if np.min(np.r_[r, x], initial=0.) < -1e-9 or np.max(np.r_[r, x], initial=0.) > 2.+1e-9:
        raise ValueError("Response coefficients violate the unchanged [0,2] box.")
    return supports


def matrices(point, n):
    supports = family(point, n)
    z = np.asarray([[i in C for i in range(n)] for C in supports], float).reshape(len(supports), n)
    return (z.T*np.asarray(point["r"]))@z, (z.T*np.asarray(point["x"]))@z


def scores(split, point):
    R, X = matrices(point, split["n"])
    residual = split["y"]-split["p"]@R.T-split["q"]@X.T
    return dict(mae=float(np.mean(abs(residual))), mse=float(np.mean(residual**2)),
                sse=float(np.sum(residual**2)), observations=int(residual.size))


def topology(point, n):
    return frozenset(tuple(C) for C, r, x in zip(point["supports"], point["r"], point["x"])
                     if 1 < len(C) < n and max(r, x) > 1e-9)


def support_count(point, n):
    """Native count includes retained zero proper supports, just as the model."""
    return sum(1 < len(C) < n for C in point["supports"])


def fingerprint(point):
    return tuple(sorted((tuple(C), float(r), float(x))
                        for C, r, x in zip(point["supports"], point["r"], point["x"])))


def features(split, supports):
    n = split["n"]
    z = np.asarray([[i in C for i in range(n)] for C in supports], float).reshape(len(supports), n)
    columns = [((a@row)[:, None]*row).ravel() for a in (split["p"], split["q"]) for row in z]
    return np.column_stack(columns) if columns else np.zeros((split["y"].size, 0))


def box_certificate(F, y, theta):
    """A convex supporting-plane lower bound, valid also for a rank-deficient F."""
    theta = np.asarray(theta, float)
    residual = F@theta-y
    value = float(np.mean(residual**2)); gradient = 2.*(F.T@residual)/len(y)
    lower = value+float(np.minimum(-theta*gradient, (2.-theta)*gradient).sum())
    # The projected unit-gradient map is a box KKT residual, in physical units.
    projected = theta-np.clip(theta-gradient, 0., 2.)
    return dict(mse=value, lower_bound=max(0., lower), gap=max(0., value-max(0., lower)),
                projected_gradient_max=float(np.max(abs(projected), initial=0.)),
                bound_violation=float(max(np.max(-theta, initial=0.), np.max(theta-2., initial=0.))))


def independent_qp(split, supports):
    """Independent bounded LS; no LP epigraph or native fitting code is reused."""
    supports = canonical(supports)
    F, y = features(split, supports), split["y"].ravel()
    started = perf_counter()
    if F.shape[1]:
        solved = lsq_linear(F, y, bounds=(0., 2.), method="bvls", tol=1e-12, max_iter=20000)
        if not solved.success:
            raise RuntimeError("Independent bounded LS did not converge: "+str(solved.message))
        theta = np.clip(solved.x, 0., 2.)
        status, message, iterations = int(solved.status), str(solved.message), int(solved.nit)
    else:
        theta = np.zeros(0); status, message, iterations = 3, "empty fixed family", 0
    certificate = box_certificate(F, y, theta)
    if certificate["bound_violation"] > 1e-12:
        raise ValueError("Independent bounded-LS point is infeasible.")
    return dict(supports=[list(C) for C in supports], theta=theta.tolist(), status=status,
                message=message, iterations=iterations, rank=int(np.linalg.matrix_rank(F)),
                features=F.shape[1], observations=len(y), seconds=perf_counter()-started,
                solver="SciPy lsq_linear BVLS; physical bounded squared loss", **certificate)


def fixed_qp_check(point, split, profile, K):
    supports = family(point, split["n"])
    # Order the independently evaluated native coefficients to the profile family.
    lookup = {C: (float(r), float(x)) for C, r, x in zip(supports, point["r"], point["x"])}
    ordered = canonical(supports)
    theta = np.r_[[lookup[C][0] for C in ordered], [lookup[C][1] for C in ordered]]
    native = box_certificate(features(split, ordered), split["y"].ravel(), theta)
    optimum = float(profile["mse"])
    tolerance = QP_ATOL+QP_RTOL*abs(K*optimum)
    if K*(native["mse"]-profile["lower_bound"]) > tolerance:
        raise ValueError("A QP-refitted point exceeds the independent bounded-LS certificate tolerance.")
    if K*(optimum-native["mse"]) > tolerance:
        raise ValueError("Independent bounded LS is worse than the saved feasible point.")
    return dict(native_mse=native["mse"], independent_mse=optimum,
                scaled_objective_excess=K*(native["mse"]-optimum),
                scaled_certificate_gap=K*(native["mse"]-profile["lower_bound"]),
                scaled_tolerance=tolerance, native_projected_gradient_max=native["projected_gradient_max"],
                native_box_gap=native["gap"])


def physical_truth(output, meta, labels):
    """Load only the frozen, NumPy/Pandas truth helper, by an absolute file path."""
    path = output/"source/research_experiments/rnj/audit_source_ac96_benchmark.py"
    spec = spec_from_file_location("miqp_offline_physical_truth", path)
    module = module_from_spec(spec); spec.loader.exec_module(module)
    truth = module.physical_truth(Path(meta["truth_network"]), labels)
    positions = {v: i for i, v in enumerate(labels)}
    truth["indexed_clades"] = frozenset(tuple(sorted(positions[v] for v in C)) for C in truth["clades"])
    return truth


def metrics(point, splits, truth):
    n = splits["train"]["n"]; predicted = topology(point, n); target = truth["indexed_clades"]
    tp, fp, fn = len(predicted & target), len(predicted-target), len(target-predicted)
    R, X = matrices(point, n)
    result = dict(clade_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1., exact=predicted == target,
                  true_positive=tp, false_positive=fp, false_negative=fn,
                  predicted_clades=len(predicted), true_clades=len(target),
                  R_relative=float(np.linalg.norm(R-truth["R"])/np.linalg.norm(truth["R"])),
                  X_relative=float(np.linalg.norm(X-truth["X"])/np.linalg.norm(truth["X"])))
    for name, split in splits.items():
        result.update({name+"_"+key: value for key, value in scores(split, point).items() if key != "observations"})
    return result


def choose(rows, score_values):
    values = {int(row["index"]): float(value) for row, value in zip(rows, score_values)}
    best = min(values.values())
    eligible = [row for row in rows if values[int(row["index"])] <= best+1e-12]
    return min(eligible, key=lambda row: (int(row["internal"]), values[int(row["index"])], int(row["index"])))


def solver_records(result):
    for event in result.get("trajectory", {}).get("attempts", []):
        for call in event.get("trials", [event]):
            if call.get("status") is not None:
                yield dict(stage=event.get("kind", "unknown"), **call)
    event = result.get("exit_event", {})
    if event.get("status") is not None:
        yield dict(stage="final_exit", **event)


def audit_events(result, *, K, reference, squared, n):
    calls = list(solver_records(result)); audits = 0
    for call in calls:
        incumbent = call.get("objective", call.get("final_objective"))
        bound = call.get("dual_bound", call.get("bound", call.get("final_bound")))
        if incumbent is not None and bound is not None and float(bound) > float(incumbent)+2e-8:
            raise ValueError("Saved solver lower bound exceeds its incumbent.")
        for key in ("audit", "warm_start"):
            audit = call.get(key)
            if isinstance(audit, dict) and "valid" in audit and not audit["valid"]:
                raise ValueError("Native feasibility or warm-start audit failed.")
        if squared:
            if call.get("solver") != "gurobi_convex_MIQP":
                raise ValueError("Squared structural call is not the explicit convex MIQP backend.")
            close(call["loss_scale"], K, message="MIQP K changed")
        audits += 1
    trajectory = result["trajectory"]
    penalty = .003*reference
    close(result["reference"], reference, atol=1e-12, message="Fixed normalization reference")
    for event in trajectory.get("attempts", []):
        if "penalty" in event:
            close(event["penalty"], penalty, atol=1e-12, message="Line penalty changed")
        if all(key in event for key in ("before", "before_score", "decoded_objective", "decoded_score", "gain", "threshold")):
            close(event["gain"], event["before_score"]-event["decoded_score"], message="Gain replay")
            close(event["threshold"], 1e-10+.001*event["before"], atol=1e-12, message="Gate unit replay")
            if bool(event["accepted"]) != bool(event["gain"] > event["threshold"]):
                raise ValueError("Saved acceptance does not use the declared gate.")
            if penalty > 0:
                for score_key, loss_key in (("before_score", "before"), ("decoded_score", "decoded_objective")):
                    count = (float(event[score_key])-float(event[loss_key]))/penalty
                    if abs(count-round(count)) > 1e-5 or not -.00001 <= count <= n-2+.00001:
                        raise ValueError("Penalized objective is inconsistent with an internal-support count.")
            regularized_bound = event.get("regularized_bound")
            if regularized_bound is not None and regularized_bound > min(event["before_score"], event["decoded_score"])+2e-8:
                raise ValueError("Regularized bound does not include the same protected-count constant.")
            audits += 4
    exit_record = result.get("exit_record", {})
    if isinstance(exit_record.get("raw_point"), dict):
        for point_key, score_key in (("raw_point", "raw_score"), ("point", "score")):
            point = exit_record[point_key]
            close(exit_record[score_key], point["objective"]+penalty*support_count(point, n),
                  message="E count-regularized objective replay")
        audit = exit_record.get("audit", {})
        if not audit.get("valid", False) or float(exit_record["raw_score"]) > float(audit["objective"])+2e-8:
            raise ValueError("E raw physical objective exceeds its audited squared epigraph objective.")
        bound = exit_record.get("bound")
        if bound is not None and float(bound) > float(exit_record["score"])+2e-8:
            raise ValueError("E saved bound exceeds its returned regularized score.")
        audits += 4
    zeros = trajectory.get("zero_events", [])
    for event in zeros:
        accepted = event["after"] <= event["before"]+event.get("loss_tolerance", 1e-10)
        if bool(event["accepted"]) != bool(accepted):
            raise ValueError("Zero cleanup gate changed units or acceptance.")
    tails = [event for event in trajectory.get("attempts", []) if event.get("kind") == "joint_tail"]
    if result["method"] == "noT" and tails:
        raise ValueError("noT unexpectedly executed T.")
    return dict(MIPs=len(calls), MIP_timeouts=sum(c["status"] == 9 for c in calls),
                gap_met=sum(c["status"] == 2 for c in calls),
                model_solver_failure_events=sum(c["status"] in (3, 4, 5, 12) for c in calls),
                tail_calls=len(tails), tail_accepted=sum(bool(e.get("accepted")) for e in tails),
                tail_seconds=sum(float(e.get("wall_seconds", 0.)) for e in tails),
                exit_accepted=bool(result.get("exit_record", {}).get("accepted")),
                exit_seconds=float(trajectory.get("final_exit_seconds", 0.)),
                zero_release_events=sum(bool(e["accepted"]) for e in zeros),
                zero_released_sets=sum(len(e.get("removed", [])) for e in zeros if e["accepted"]),
                stop_reason=trajectory.get("stop_reason"), audited_event_checks=audits), calls


def timing(result, protocol):
    raw = result.get("timing", result)
    fresh, common, total = (float(raw[k]) for k in ("fresh_seconds", "common_seconds", "total_seconds"))
    if not all(np.isfinite(v) and v >= 0 for v in (fresh, common, total)):
        raise ValueError("Invalid saved timing.")
    close(total, fresh+common, atol=1e-7, message="Timing sum")
    charge = float(result.get("interrupted_attempt_seconds", 0.))
    logical = float(result.get("logical_total_seconds", total+charge))
    close(logical, total+charge, atol=1e-7, message="Operational timing sum")
    original_cap = float(result.get("original_total_cap", protocol["limits"]["total"]))
    current_cap = float(result.get("limits", protocol["limits"])["total"])
    if current_cap+charge > original_cap+1.:
        raise ValueError("An interrupted attempt received a restored budget.")
    return dict(total_seconds=total, common_seconds=common, fresh_seconds=fresh,
                operational_total_seconds=logical, interrupted_attempt_seconds=charge,
                cap_exceeded=logical > original_cap+1., successful_cap_exceeded=total > current_cap+1.)


def loaded_sources(result, result_file, output, protocol):
    source = (output/"source").resolve()
    for stage, receipt in result["loaded_source_receipts"].items():
        file = result_file.with_name(f"loaded_sources_{stage}.json")
        if sha(file) != receipt["sha256"]:
            raise ValueError("Loaded-source receipt changed.")
        value = read(file)
        if not value["passed"] or len(value["loaded_modules"]) != int(receipt["modules"]):
            raise ValueError("Incomplete loaded-source receipt.")
        for module in value["loaded_modules"].values():
            path = Path(module["path"]).resolve()
            if not path.is_relative_to(source) or path.relative_to(source).as_posix() != module["relative_path"]:
                raise ValueError("A runtime project import escaped the frozen snapshot.")
            if sha(path) != module["sha256"] or protocol["source_hashes"][module["relative_path"]] != module["sha256"]:
                raise ValueError("A loaded runtime module hash changed.")
    overlay = result["runtime_overlay"]
    close(overlay["loss_scale"], result["normalization"]["K"], message="Overlay K")
    if overlay.get("ridge") != 0 or overlay.get("positive_impedance_floor") != 0:
        raise ValueError("The objective comparison introduced an unplanned regularizer or floor.")
    if (not overlay["frozen_native_sources_verified"] or len(overlay["native_module_sources"]) != 6
            or len(overlay["aliases"]) != 14):
        raise ValueError("Missing explicit frozen runtime-alias manifest.")
    overlay_file = result_file.with_name("runtime_overlay.json")
    if "runtime_overlay_sha256" in result:
        if sha(overlay_file) != result["runtime_overlay_sha256"]:
            raise ValueError("Runtime overlay receipt changed.")
    elif read(overlay_file) != overlay:
        raise ValueError("Common runtime overlay differs from its saved manifest.")


def audit_qp_records(records, *, K, observations):
    for record in records:
        if record["solver"] != "gurobi_QP" or record["status"] != 2:
            raise ValueError("A fixed-family QP did not return native optimal status.")
        close(record["loss_scale"], K, message="Fixed-QP loss scale")
        close(record["objective_scale"], 1000., message="Fixed-QP native objective scale")
        close(record["quadratic_coefficient"], K/(observations*1000.), atol=1e-15,
              message="Fixed-QP quadratic coefficient")
        close(record["objective"]/1000., record["normalized_squared_loss"], atol=1e-8,
              message="Fixed-QP native objective replay")
        close(record["normalized_squared_loss"], K*record["physical_mse"], atol=1e-12,
              message="Fixed-QP physical normalization")
        if record["ridge"] != 0 or record["positive_impedance_floor"] != 0 or record["cache_kind"] != "QP only":
            raise ValueError("A fixed-family QP changed the objective-only contract.")


def parent_receipt(parent, job, method, file, protocol_hash):
    verification_file = parent/"analysis/verification.json"
    verified_file = parent/"analysis/jobs"/job/"verified.json"
    verification, verified = read(verification_file), read(verified_file)
    if not verification["passed"] or not verification["complete"] or verification["planned"] != 144 or verification["verified"] != 144:
        raise ValueError("The parent L1 experiment must have all 144 configurations independently verified.")
    if verification["protocol_sha256"] != protocol_hash or verified["protocol_sha256"] != protocol_hash:
        raise ValueError("Parent analysis refers to a different protocol.")
    if verification["analyzer_sha256"] != verified["analyzer_sha256"]:
        raise ValueError("Parent per-input and aggregate analyzers differ.")
    analyzer = parent/"analysis_source/analyze.py"
    if sha(analyzer) != verification["analyzer_sha256"]:
        raise ValueError("Parent analysis does not match its authoritative frozen post-analysis source.")
    if verified["raw_hashes"].get(method) != sha(file):
        raise ValueError("Parent result bytes differ from their independent verification.")
    matches = [row for row in verified["rows"] if row["method"] == method]
    if len(matches) != 1 or matches[0]["status"] != "verified":
        raise ValueError("Parent result has not passed independent verification.")
    return dict(analysis_verification_sha256=sha(verification_file), job_verification_sha256=sha(verified_file),
                analyzer_sha256=verification["analyzer_sha256"], result_sha256=sha(file),
                parent_complete=bool(verification["complete"]), parent_passed=bool(verification["passed"]),
                parent_planned=verification["planned"], parent_verified=verification["verified"])


def verify_export(result_file, result, splits, labels, K):
    """Independent loss replay in addition to the portable export's SHA check."""
    file = result_file.with_name("candidates.json")
    if sha(file) != result["candidate_export_sha256"]:
        raise ValueError("Candidate export file changed.")
    pool = read(file)
    if digest({key: value for key, value in pool.items() if key != "pool_sha256"}) != pool["pool_sha256"]:
        raise ValueError("Candidate full-payload hash does not reproduce.")
    if pool["labels"] != labels or pool.get("fit_loss") != "scaled_mse":
        raise ValueError("Export lost label order or squared-fit semantics.")
    ids = set()
    for row in pool["candidates"]:
        ids.add(row["candidate_id"])
        if row.get("frozen_weights") is not True or row.get("labels") != labels:
            raise ValueError("A portable candidate is not a frozen full model in the declared label order.")
        ordered = sorted(zip(row["point"]["supports"], row["point"]["r"], row["point"]["x"]),
                         key=lambda item: (len(item[0]), tuple(item[0])))
        canonical_point = dict(supports=[list(C) for C, _, _ in ordered],
                               r=[float(r) for _, r, _ in ordered], x=[float(x) for _, _, x in ordered])
        if digest(canonical_point) != row["candidate_id"]:
            raise ValueError("A portable candidate's support/parameter identifier changed.")
        R, X = matrices(row["point"], len(labels))
        if not np.array_equal(R, np.asarray(row["R"])) or not np.array_equal(X, np.asarray(row["X"])):
            raise ValueError("Saved portable R/X do not reproduce from the frozen coefficients.")
        train, validation = scores(splits["train"], row["point"]), scores(splits["validation"], row["point"])
        for name, values in (("train", train), ("validation", validation)):
            for metric in ("mae", "mse", "sse"):
                close(row[f"{name}_{metric}"], values[metric], atol=1e-12, message="Export physical loss replay")
        close(row["point"]["objective"], train["mae"], atol=1e-12, message="Portable objective remains MAE")
        close(row["native_internal_objective"], K*train["mse"], atol=1e-12, message="Export internal objective")
    for key, selected in (("native_selected", result["selected"]), ("native_mse_view", result["selected_mse_view"])):
        anchor = pool["native_anchors"][key]
        if anchor["candidate_id"] not in ids:
            raise ValueError("A selected model anchor is missing.")
        matches = [row for row in pool["candidates"] if row["candidate_id"] == anchor["candidate_id"]]
        if len(matches) != 1 or fingerprint(matches[0]["point"]) != fingerprint(selected["point"]):
            raise ValueError("A selected anchor changed its fitted R/X coefficients.")
    provenance = pool["provenance"]
    if any(provenance.get(k) is not False for k in ("truth_used_in_fitting", "test_used_in_generation", "future_DR_used")):
        raise ValueError("Export provenance violates data-only frozen-parameter generation.")
    return dict(candidate_export_sha256=sha(file), pool_sha256=pool["pool_sha256"],
                exported_models=len(pool["candidates"]), selected_anchors_verified=True)


def analyze_job(args):
    output, job = args; output = Path(output)
    protocol, catalog = read(output/"protocol.json"), read(output/"catalog.json")
    parent = Path(protocol["parent_output"]); parent_protocol = read(parent/"protocol.json")
    meta = catalog[job]; header = dict(job=job, **{k: meta.get(k) for k in METADATA})
    rows = [dict(**header, fit_loss=loss, method=method, status="pending", available=False,
                 returned=(base/"jobs"/job/method/"result.json").exists())
            for loss, base in (("mae", parent), ("scaled_mse", output)) for method in METHODS]
    record = dict(job=job, rows=rows, views=[], solver=[], errors=[], fixed_QPs=[], fixed_checks=[],
                  parent_analysis={}, raw_hashes={}, checks=0, protocol_sha256=sha(output/"protocol.json"),
                  analyzer_sha256=sha(__file__))
    profiles = {}
    try:
        if meta.get("data_error"):
            raise ValueError("Frozen input generation failed: "+str(meta["data_error"]))
        input_hashes = {name: sha(meta[name]) for name in ("train", "validation", "test")}
        if input_hashes != {k: protocol["input_hashes"][job][k] for k in input_hashes}:
            raise ValueError("Frozen observations changed.")
        splits = {name: load_split(meta[name]) for name in input_hashes}
        if {key: s["rows"] for key, s in splits.items()} != dict(train=96, validation=288, test=96):
            raise ValueError("Unexpected split length.")
        labels = splits["train"]["labels"]; n = len(labels)
        if any(s["labels"] != labels for s in splits.values()):
            raise ValueError("Split label order differs.")
        truth = physical_truth(output, meta, labels)
        record["truth_source_hashes"] = {name: sha(Path(meta["truth_network"])/name)
                                         for name in ("metadata.json", "buses.csv", "branches.csv")}
        parent_common_file = parent/"common"/job/"common.json"
        parent_common = read(parent_common_file)
        if sha(parent_common_file) != protocol["parent_common_hashes"][job]:
            raise ValueError("Frozen parent preprocessing changed.")
        reference_family = canonical([*parent_common["initial"], *((i,) for i in range(n)), tuple(range(n))])
        def profile(supports):
            key = canonical(supports)
            if key not in profiles:
                profiles[key] = independent_qp(splits["train"], key)
            return profiles[key]
        ref_profile = profile(reference_family)
        f01, f02 = float(parent_common["reference"]), float(ref_profile["mse"])
        independently_profiled_f02 = f02
        if not (f01 > 0 and f02 > 0):
            raise ValueError("Positive reference losses are required by the frozen protocol.")
        K = f01/f02
        common_file = output/"common"/job/"common.json"
        common = read(common_file) if common_file.exists() else None
        if common:
            norm = common["normalization"]
            close(norm["f0_mae"], f01, atol=1e-12, message="Parent MAE reference")
            close(norm["K"]*norm["f0_mse"], f01, atol=1e-12, message="Normalization identity")
            if norm["reference_family_sha256"] != digest(reference_family):
                raise ValueError("Squared reference family differs from the declared parent family.")
            K = float(norm["K"])
            close(norm["f0_mse"], f02, atol=QP_ATOL/K, rtol=QP_RTOL, message="Independent reference QP")
            f02 = float(norm["f0_mse"])
            if common["parent_common_sha256"] != sha(parent_common_file) or common["warm_weights_reused"]:
                raise ValueError("Warm QP preprocessing reused LP weights or changed its parent.")
            close(common["seconds"], common["parent_common_seconds"]+common["qp_common_seconds"], message="Common timing")
            if common.get("QP_failures") or common.get("LP") or common.get("LP_failures"):
                raise ValueError("New common preprocessing contains failed or reused LP fits.")
            if len(common["bank"]) != len(parent_common["bank"]):
                raise ValueError("Warm-family bank length changed.")
            for new, old in zip(common["bank"], parent_common["bank"]):
                if canonical(new["supports"]) != canonical(old["supports"]):
                    raise ValueError("QP warm preprocessing changed a parent-provided family.")
            for point in [common["reference_point"], *common["bank"]]:
                close(point["objective"], K*scores(splits["train"], point)["mse"], message="Common K*MSE replay")
                check = fixed_qp_check(point, splits["train"], profile(point["supports"]), K)
                record["fixed_checks"].append(dict(kind="common", **check))
            physical = common["reference_qp_point_physical_mse"]
            close(physical["objective"], scores(splits["train"], physical)["mse"], atol=1e-12, message="Cold physical QP")
            if physical["reference_fit_uses_global_loss_scale"] is not False:
                raise ValueError("The cold reference QP reused a previous global K.")
            audit_qp_records([physical["audit"]], K=1., observations=splits["train"]["y"].size)
            audit_qp_records(common["QP"], K=K, observations=splits["train"]["y"].size)
            loaded_sources(common, common_file, output, protocol)
            record["common_sha256"] = sha(common_file)
        record["normalization"] = dict(f0_mae=f01, f0_mse=f02,
                                        independently_profiled_f0_mse=independently_profiled_f02, K=K,
                                        source="native common QP" if common else "post-analysis reference QP only")
        true_family = canonical([*truth["indexed_clades"], *((i,) for i in range(n)), tuple(range(n))])
        true_profile = profile(true_family)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        for row in rows:
            row.update(status="invalid", error=message)
        record["errors"].append(dict(job=job, kind="input_or_common", detail=message))
        record["fixed_QPs"] = list(profiles.values())
        return record

    for row in rows:
        squared = row["fit_loss"] == "scaled_mse"; method = row["method"]
        base, source_protocol = (output, protocol) if squared else (parent, parent_protocol)
        file = base/"jobs"/job/method/"result.json"
        if not file.exists():
            error = file.with_name("error.json")
            if error.exists():
                row.update(status="failed", error=read(error))
            continue
        record["raw_hashes"][row["fit_loss"]+"/"+method] = sha(file)
        try:
            result = read(file)
            if result["job"] != job or result["method"] != method:
                raise ValueError("Saved job or method differs.")
            if result["input_hashes"] != {k: input_hashes[k] for k in ("train", "validation")}:
                raise ValueError("Result observations changed.")
            if result.get("truth_used") is not False or result.get("test_opened") is not False or result.get("final_RNJ_fallback"):
                raise ValueError("Result violates the data-only own-search boundary.")
            if result["protocol_sha256"] != sha(base/"protocol.json"):
                raise ValueError("Result protocol changed.")
            if squared:
                if common is None or result["common_sha256"] != sha(common_file):
                    raise ValueError("Missing or changed common QP preprocessing.")
                if result["fit_loss"] != "scaled_mse" or result.get("QP_failures") or result.get("LP_failures"):
                    raise ValueError("Squared fit metadata or fitting failures are invalid.")
                close(result["normalization"]["K"], K, message="Method normalization")
                loaded_sources(result, file, output, protocol)
                audit_qp_records(result["QP"], K=K, observations=splits["train"]["y"].size)
                if result["parent"]["protocol_sha256"] != protocol["parent_protocol_sha256"] or result["parent"]["common_sha256"] != protocol["parent_common_hashes"][job]:
                    raise ValueError("Native result lost its parent protocol/common SHA chain.")
            else:
                record["parent_analysis"][method] = parent_receipt(parent, job, method, file, sha(parent/"protocol.json"))
                if result["common_sha256"] != sha(parent_common_file):
                    raise ValueError("Parent common preprocessing changed.")
            trajectory = result["trajectory"]; output_rows = result["output_candidates"]
            if len({int(v["index"]) for v in output_rows}) != len(output_rows) or not output_rows:
                raise ValueError("Output-library indices are missing or repeated.")
            for point in [*trajectory["path"], trajectory["point"], *[v["point"] for v in output_rows]]:
                train_score = scores(splits["train"], point)
                expected = K*train_score["mse"] if squared else train_score["mae"]
                close(point["objective"], expected, message="Native training objective replay")
                if squared:
                    for key in ("mae", "mse", "sse"):
                        close(point["train_"+key], train_score[key], message="Saved physical training loss")
                    close(point["native_internal_objective"], expected, message="Saved internal training objective")
                    close(point["loss_scale"], K, message="Saved point normalization")
                    if point["objective_kind"] != "K*MSE" or point["fit_loss"] != "scaled_mse":
                        raise ValueError("A native point changed the declared squared-loss objective semantics.")
                    check = fixed_qp_check(point, splits["train"], profile(point["supports"]), K)
                    record["fixed_checks"].append(dict(kind="method", method=method, **check))
                record["checks"] += 3
            if squared and result.get("exit_record", {}).get("raw_point"):
                raw = result["exit_record"]["raw_point"]
                close(raw["objective"], K*scores(splits["train"], raw)["mse"], message="Raw E internal loss")
            mae_values, squared_values = [], []
            for candidate in output_rows:
                values = scores(splits["validation"], candidate["point"])
                close(candidate["validation"], values["mae"], atol=1e-10, message="Primary MAE validation replay")
                if candidate["internal"] != support_count(candidate["point"], n):
                    raise ValueError("Output selector uses a different support count.")
                if squared:
                    for key in ("mae", "mse", "sse"):
                        close(candidate["validation_"+key], values[key], message="Saved physical validation loss")
                    close(candidate["validation_scaled_mse"], K*values["mse"], message="Scaled validation MSE")
                mae_values.append(values["mae"]); squared_values.append(K*values["mse"])
            chosen = {"primary_mae": choose(output_rows, mae_values), "secondary_mse": choose(output_rows, squared_values)}
            for view, selected in chosen.items():
                saved = result["selected"] if view == "primary_mae" else result.get("selected_mse_view")
                if saved is not None and (int(saved["index"]) != int(selected["index"]) or fingerprint(saved["point"]) != fingerprint(selected["point"])):
                    raise ValueError("Saved historical selector does not reproduce from the same fitted library.")
            event_metrics, calls = audit_events(result, K=K, reference=f01, squared=squared, n=n)
            record["solver"].extend(dict(job=job, fit_loss=row["fit_loss"], method=method, **call) for call in calls)
            time_metrics = timing(result, source_protocol)
            coverage_sets = [topology(v, n) for v in trajectory["path"]]
            output_sets = [topology(v["point"], n) for v in output_rows]
            union = frozenset().union(*coverage_sets, *output_sets)
            coverage = dict(raw_pool_has_true_tree=truth["indexed_clades"] in coverage_sets,
                            output_pool_has_true_tree=truth["indexed_clades"] in output_sets,
                            own_pool_union_has_true_connections=truth["indexed_clades"] <= union,
                            own_pool_union_true_recall=len(truth["indexed_clades"] & union)/len(truth["indexed_clades"]) if truth["indexed_clades"] else 1.)
            measured = metrics(chosen["primary_mae"]["point"], splits, truth)
            row.update(status="verified", available=True, **measured, **event_metrics, **time_metrics, **coverage,
                       selected_index=int(chosen["primary_mae"]["index"]), path_length=len(trajectory["path"]),
                       true_family_training_MSE=true_profile["mse"], f0_mae=f01, f0_mse=f02, K=K,
                       selected_minus_true_training_MSE=measured["train_mse"]-true_profile["mse"])
            for view, selected in chosen.items():
                record["views"].append(dict(**header, fit_loss=row["fit_loss"], method=method, view=view,
                    status="verified", returned=True, selected_index=int(selected["index"]), **metrics(selected["point"], splits, truth),
                    **time_metrics))
            if squared:
                row.update(verify_export(file, result, splits, labels, K))
            record["checks"] += 20
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            row.update(status="invalid", available=False, error=message)
            record["views"] = [v for v in record["views"] if not (v["fit_loss"] == row["fit_loss"] and v["method"] == method)]
            record["errors"].append(dict(job=job, method=method, fit_loss=row["fit_loss"], kind="verification", detail=message))
    record["fixed_QPs"] = list(profiles.values())
    return record


def aggregate(frame, keys):
    rows = []
    for key, group in frame.groupby(keys, dropna=False, sort=False):
        if not isinstance(key, tuple): key = (key,)
        ok = group[group.status.eq("verified")]
        item = dict(zip(keys, key), planned=len(group), verified=len(ok), pending=int(group.status.eq("pending").sum()),
                    returned=int(group.returned.sum()),
                    failed=int(group.status.eq("failed").sum()), invalid=int(group.status.eq("invalid").sum()),
                    complete=len(ok) == len(group))
        item.update(F1=float(ok.clade_f1.mean()) if len(ok) else np.nan,
                    exact=int(ok.exact.sum()) if len(ok) else 0,
                    exact_fraction_planned=float(ok.exact.sum())/len(group) if len(ok) else 0.,
                    recovery_rate_completed=float(ok.exact.mean()) if len(ok) else np.nan,
                    total_seconds=float(ok.total_seconds.sum()) if len(ok) else np.nan,
                    operational_total_seconds=float(ok.operational_total_seconds.sum()) if len(ok) else np.nan)
        for name in ("train_mae", "train_mse", "validation_mae", "validation_mse", "test_mae", "test_mse"):
            item[name] = float(ok[name].mean()) if len(ok) else np.nan
        rows.append(item)
    return pd.DataFrame(rows)


def paired_tables(views, jobs, *, draws=20000):
    good = views[views.status.eq("verified")].set_index(["job", "fit_loss", "method", "view"])
    inputs, summaries, bootstrap = [], [], []
    rng = np.random.default_rng(20261111)
    for method in METHODS:
        for view in VIEWS:
            paired = []
            for job in jobs:
                key1, key2 = (job, "mae", method, view), (job, "scaled_mse", method, view)
                header = dict(job=job, method=method, view=view, before="mae", after="scaled_mse")
                if key1 not in good.index or key2 not in good.index:
                    inputs.append(dict(**header, status="incomplete")); continue
                a, b = good.loc[key1], good.loc[key2]
                item = dict(**header, status="paired", case=a["case"], bundle=int(a.bundle), seed_index=int(a.seed_index),
                    voltage_sigma=float(a.voltage_sigma), delta_F1=float(b.clade_f1-a.clade_f1),
                    delta_exact=int(b.exact)-int(a.exact), delta_seconds=float(b.total_seconds-a.total_seconds),
                    new_exact=bool(b.exact and not a.exact), lost_exact=bool(a.exact and not b.exact))
                paired.append(item); inputs.append(item)
            data = pd.DataFrame(paired)
            summary = dict(method=method, view=view, planned=len(jobs), paired=len(data), complete=len(data) == len(jobs))
            if len(data):
                summary.update(delta_F1=float(data.delta_F1.mean()), delta_exact=int(data.delta_exact.sum()),
                    delta_recovery_rate=float(data.delta_exact.mean()), delta_total_seconds=float(data.delta_seconds.sum()),
                    F1_wins=int((data.delta_F1 > 1e-10).sum()), F1_ties=int((abs(data.delta_F1) <= 1e-10).sum()),
                    F1_losses=int((data.delta_F1 < -1e-10).sum()), new_exact=int(data.new_exact.sum()), lost_exact=int(data.lost_exact.sum()))
            summaries.append(summary)
            blocks = data.groupby("seed_index").agg(n=("job", "size"), F1=("delta_F1", "mean"),
                exact=("delta_exact", "mean"), seconds=("delta_seconds", "sum")) if len(data) else pd.DataFrame()
            complete = len(data) == 48 and len(blocks) == 4 and bool((blocks.n == 12).all())
            receipt = dict(method=method, view=view, paired_inputs=len(data), complete=complete,
                shared_seed_blocks=len(blocks), resamples=draws, seed=20261111,
                scheme="resample four shared seed_index blocks, keeping all four cases and three sigma levels together")
            if complete:
                values = blocks[["F1", "exact", "seconds"]].to_numpy(float)
                sampled = values[rng.integers(0, 4, size=(draws, 4))].mean(axis=1)
                for i, name in enumerate(("delta_F1", "delta_recovery_rate", "delta_total_seconds")):
                    interval = np.quantile(sampled[:, i]*(4 if i == 2 else 1), [.025, .975])
                    receipt[name+"_lower95"], receipt[name+"_upper95"] = map(float, interval)
            receipt["interpretation"] = "descriptive four-seed resampling; fixed four networks; unavailable for a pilot or incomplete panel"
            bootstrap.append(receipt)
    return dict(paired_inputs=pd.DataFrame(inputs), paired_summary=pd.DataFrame(summaries), block_bootstrap=pd.DataFrame(bootstrap))


def check_protocol(output):
    protocol = read(output/"protocol.json"); parent = Path(protocol["parent_output"])
    if protocol["planned"] != 96 or protocol["planned_inputs"] != 48 or tuple(m["id"] for m in protocol["methods"]) != METHODS:
        raise ValueError("This analyzer requires the fixed 48-input, two-method MIQP protocol.")
    if protocol["fit_loss"] != "scaled_mse" or protocol["observations_regenerated"] or protocol["parent_source_bytes_modified"]:
        raise ValueError("Unplanned objective or data/source change.")
    if sha(output/"catalog.json") != protocol["catalog_sha256"] or sha(parent/"catalog.json") != protocol["parent_catalog_sha256"]:
        raise ValueError("New or parent input catalog changed.")
    if sha(parent/"protocol.json") != protocol["parent_protocol_sha256"]:
        raise ValueError("Parent protocol changed.")
    parent_analysis = read(parent/"analysis/verification.json")
    if (not parent_analysis["passed"] or not parent_analysis["complete"]
            or parent_analysis["planned"] != 144 or parent_analysis["verified"] != 144
            or parent_analysis["protocol_sha256"] != protocol["parent_protocol_sha256"]):
        raise ValueError("Complete, independently passed parent L1 verification is required.")
    if sha(parent/"analysis_source/analyze.py") != parent_analysis["analyzer_sha256"]:
        raise ValueError("The parent authoritative frozen post-analysis source changed.")
    for base, hashes in ((output, protocol["source_hashes"]), (parent, protocol["parent_source_hashes"])):
        for name, expected in hashes.items():
            if sha(base/"source"/name) != expected:
                raise ValueError("Frozen source changed: "+str(base/name))
    for name, expected in protocol["source_artifact_hashes"].items():
        if sha(output/"source"/name) != expected:
            raise ValueError("Frozen new package artifact changed: "+name)
    for name, expected in protocol["parent_source_hashes"].items():
        if protocol["source_hashes"].get(name) != expected or sha(output/"source"/name) != expected:
            raise ValueError("A parent source byte was changed by the new snapshot: "+name)
    catalog = read(output/"catalog.json")
    if len(protocol["jobs"]) != len(catalog) or len(catalog) != 48 or set(protocol["jobs"]) != set(catalog):
        raise ValueError("The planned 48 input denominator changed.")
    return protocol, catalog


def analyze(output, *, workers=2, use_cache=True):
    output = Path(output).resolve(); started = perf_counter()
    protocol, catalog = check_protocol(output); jobs = protocol["jobs"]
    destination = output/"analysis"; destination.mkdir(parents=True, exist_ok=True)
    parent = Path(protocol["parent_output"])
    parent_verification = parent/"analysis/verification.json"
    parent_analysis_sha = sha(parent_verification) if parent_verification.exists() else None
    source_signature = dict(analyzer_sha256=sha(__file__), protocol_sha256=sha(output/"protocol.json"),
                            parent_analysis_sha256=parent_analysis_sha,
                            parent_analyzer_sha256=sha(parent/"analysis_source/analyze.py"))
    results, todo = {}, []
    for job in jobs:
        raw = {loss+"/"+method: sha(base/"jobs"/job/method/"result.json")
               for loss, base in (("mae", parent), ("scaled_mse", output)) for method in METHODS
               if (base/"jobs"/job/method/"result.json").exists()}
        common = output/"common"/job/"common.json"
        signature = dict(**source_signature, raw_hashes=raw, common_sha256=sha(common) if common.exists() else None)
        cache = destination/"jobs"/job/"verified.json"
        saved = read(cache) if use_cache and cache.exists() else None
        if saved and saved.get("cache_signature") == signature and all(row["status"] == "verified" for row in saved["rows"]):
            results[job] = saved
        else:
            todo.append((str(output), job))
    def save(record):
        record["cache_signature"] = dict(**source_signature, raw_hashes=record["raw_hashes"],
                                          common_sha256=record.get("common_sha256"))
        results[record["job"]] = record
        write(destination/"jobs"/record["job"]/"verified.json", record)
    if workers == 1:
        for record in map(analyze_job, todo): save(record)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for record in executor.map(analyze_job, todo): save(record)
    records = [results[job] for job in jobs]
    frame = pd.DataFrame([row for record in records for row in record["rows"]])
    # Preserve absent views explicitly. One fit has two selectors, not two solves.
    available_views = {(v["job"], v["fit_loss"], v["method"], v["view"]): v for r in records for v in r["views"]}
    views = pd.DataFrame([available_views.get((r["job"], r["fit_loss"], r["method"], view),
        dict(**{k: r.get(k) for k in ("job", *METADATA, "fit_loss", "method", "status", "returned")}, view=view))
        for r in frame.to_dict("records") for view in VIEWS])
    for table in (frame, views):
        for name in ("clade_f1", "exact", "total_seconds", "operational_total_seconds", "train_mae", "train_mse",
                     "validation_mae", "validation_mse", "test_mae", "test_mse"):
            if name not in table: table[name] = np.nan
    tables = dict(per_model=frame, view_inputs=views,
                  summary=aggregate(views, ["fit_loss", "method", "view"]),
                  cases=aggregate(views, ["case", "fit_loss", "method", "view"]),
                  noise=aggregate(views, ["voltage_sigma", "fit_loss", "method", "view"]),
                  case_noise=aggregate(views, ["case", "voltage_sigma", "fit_loss", "method", "view"]))
    tables.update(paired_tables(views, jobs))
    tables["fixed_QP_checks"] = pd.DataFrame([dict(job=r["job"], **check) for r in records for check in r["fixed_checks"]])
    tables["fixed_QP_profiles"] = pd.DataFrame([dict(job=r["job"], **{k: v for k, v in profile.items() if k not in ("theta", "supports")})
                                               for r in records for profile in r["fixed_QPs"]])
    tables["solver"] = pd.DataFrame([dict(job=r["job"], fit_loss=c["fit_loss"], method=c["method"], stage=c["stage"],
        status=c["status"], seconds=c.get("seconds"), relative_gap=c.get("relative_gap", c.get("gap")),
        incumbent=c.get("objective", c.get("final_objective")), bound=c.get("dual_bound", c.get("final_bound")))
        for r in records for c in r["solver"]])
    tables["failures"] = pd.DataFrame([e for r in records for e in r["errors"]])
    mechanism_rows = []
    for (loss_name, method), group in frame.groupby(["fit_loss", "method"], sort=False):
        ok = group[group.status.eq("verified")]
        item = dict(fit_loss=loss_name, method=method, planned=len(group), verified=len(ok))
        for name in ("tail_calls", "tail_accepted", "tail_seconds", "exit_accepted", "exit_seconds",
                     "zero_release_events", "zero_released_sets", "MIPs", "MIP_timeouts", "gap_met",
                     "raw_pool_has_true_tree", "output_pool_has_true_tree", "own_pool_union_has_true_connections"):
            item[name] = float(ok[name].sum()) if len(ok) else 0.
        mechanism_rows.append(item)
    tables["mechanisms"] = pd.DataFrame(mechanism_rows)
    for name, table in tables.items(): table.to_csv(destination/(name+".csv"), index=False, lineterminator="\n")
    new, old = frame[frame.fit_loss.eq("scaled_mse")], frame[frame.fit_loss.eq("mae")]
    new_verified, old_verified = int(new.status.eq("verified").sum()), int(old.status.eq("verified").sum())
    errors = [e for r in records for e in r["errors"]]
    budget_exceeded = int(frame.loc[frame.status.eq("verified"), "cap_exceeded"].sum()) if "cap_exceeded" in frame else 0
    successful_budget_exceeded = int(frame.loc[frame.status.eq("verified"), "successful_cap_exceeded"].sum()) if "successful_cap_exceeded" in frame else 0
    receipt = dict(passed=new_verified == old_verified == 96 and not errors and not budget_exceeded and not successful_budget_exceeded,
        complete=new_verified == old_verified == 96, new_native_planned=96, new_native_returned=int(new.returned.sum()),
        new_native_verified=new_verified, new_native_failed=int(new.status.eq("failed").sum()),
        new_native_invalid=int(new.status.eq("invalid").sum()), new_native_pending=int(new.status.eq("pending").sum()),
        reused_parent_planned=96, reused_parent_returned=int(old.returned.sum()), reused_parent_verified=old_verified,
        combined_logical_models_planned=192, output_selector_views_planned=384,
        independent_QP_profiles=sum(len(r["fixed_QPs"]) for r in records),
        independent_QP_profile_count_scope="unique per-input family profiles represented in this audit; no native solver calls",
        fixed_QP_checks=sum(len(r["fixed_checks"]) for r in records), budget_exceeded=budget_exceeded,
        successful_attempt_budget_exceeded=successful_budget_exceeded,
        all_planned_denominators_preserved=True, truth_and_test_used_only_in_post_analysis=True,
        no_complete_RNJ_output_fallback=True, original_L1_source_unchanged=True,
        primary_validation="strict physical MAE on method-specific .24-contracted/refitted saved library",
        secondary_validation="strict K*MSE on exactly the same saved library, no new fitting",
        fixed_QP_tolerance=dict(scaled_absolute=QP_ATOL, relative=QP_RTOL),
        normalization="K=f0_MAE/f0_MSE; independent bounded-LS reference check; all coefficients remain physical",
        source_hashes_verified=True, protocol_sha256=sha(output/"protocol.json"), catalog_sha256=sha(output/"catalog.json"),
        parent_protocol_sha256=protocol["parent_protocol_sha256"], parent_catalog_sha256=protocol["parent_catalog_sha256"],
        parent_analysis_verification_sha256=parent_analysis_sha, analyzer_sha256=sha(__file__),
        parent_analyzer_sha256=sha(parent/"analysis_source/analyze.py"),
        bootstrap="four globally shared seed_index blocks; unavailable until all 48 paired inputs exist",
        runtime_scope="successful method time includes shared common costs; interrupted L1 wall charges reported separately",
        elapsed_seconds=perf_counter()-started)
    write(destination/"verification.json", receipt)
    text = "# L1 与平方误差的独立配对核查\n\n"
    text += f"新MIQP核验 {new_verified}/96，复用L1核验 {old_verified}/96。完整状态：{receipt['complete']}。\n\n"
    text += "每种损失、每种结构固定48输入；试运行和未完成结果保留原分母。两种验证视图使用同一输出库，不增加拟合次数。\n\n"
    text += "主结果见 summary.csv，配对差异见 paired_summary.csv。源文件、数据、父分析与逐结果SHA链保存在 verification.json 和 jobs/*/verified.json。\n\n"
    text += "独立QP只核查固定家族的平方损失最优性；结构搜索证书仍按各原生MIQP的实际上下界和状态报告。\n"
    (destination/"RESULTS.md").write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=ROOT/"outputs/miqp_loss_benchmark_20261111")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8: parser.error("Use 1..8 independent offline audit workers.")
    analyze(args.output, workers=args.workers, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
