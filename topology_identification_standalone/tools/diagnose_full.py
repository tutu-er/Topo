"""Optional post-audit true-family oracles; never used for search or selection.

One independent bounded least-squares fit is performed on every true family.
The decomposition concerns the frozen data/model/count penalty. It does not
separate measurement noise from AC/linear-model bias or prove identifiability.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import lsq_linear


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def flag(value):
    return str(value).lower() in {"true", "1"}


def oracle_fit(a, train, validation, test, family, reference, bound, alpha):
    started = perf_counter(); columns = []
    for power in (train["p"], train["q"]):
        for C in family:
            z = np.zeros(train["n"]); z[list(C)] = 1.
            columns.append(((power@z)[:, None]*z).ravel())
    F = np.column_stack(columns)/np.sqrt(reference)
    target = train["y"].ravel()/np.sqrt(reference)
    fitted = lsq_linear(F, target, bounds=(0., bound), method="bvls", tol=1e-12, max_iter=5000)
    if not fitted.success:
        raise RuntimeError("True-family BVLS failed: "+fitted.message)
    theta = np.clip(fitted.x, 0., bound); size = len(family)
    # Solve the same box problem in residual-centred coordinates once. This
    # reduces cancellation for tiny reference errors, without adding a ridge
    # or changing any feasible coefficient or statistical objective.
    Fl, yl = F.astype(np.longdouble), target.astype(np.longdouble)
    residual = Fl@theta.astype(np.longdouble)-yl
    refinement = lsq_linear(F, -np.asarray(residual, float), bounds=(-theta, bound-theta),
                            method="bvls", tol=1e-12, max_iter=5000)
    if not refinement.success:
        raise RuntimeError("Oracle residual refinement failed: "+refinement.message)
    trial = np.clip(theta+refinement.x, 0., bound)
    if np.mean((Fl@trial.astype(np.longdouble)-yl)**2) <= np.mean(residual**2)+1e-12:
        theta = trial
    point = dict(supports=[list(C) for C in family], r=theta[:size].tolist(), x=theta[size:].tolist())
    certificate = a.qp_certificate(train, point, reference, bound)
    scores = {name: a.score(observations, point, bound)
              for name, observations in (("train", train), ("validation", validation), ("test", test))}
    a.close(scores["train"]["mse"]/reference, certificate["objective"], "oracle objective", atol=1e-9)
    count = len(a.internal(point, train["n"]))
    return dict(status="verified", point=point, theta=theta.tolist(), scores=scores,
        proper_count=count, reference_mse=reference, J_upper=scores["train"]["mse"]/reference+alpha*count,
        J_lower=certificate["lower_bound"]+alpha*count, certificate=certificate,
        solver="SciPy lsq_linear BVLS", solver_status=int(fitted.status), iterations=int(fitted.nit),
        continuous_BVLS_calls=2, residual_refinement_status=int(refinement.status),
        tolerance=1e-12, max_iterations=5000, coefficient_bound=bound,
        zero_coefficients_retained=True, seconds=perf_counter()-started)


def decomposition(a, train, validation, test, point, truth, oracle, alpha, bound, coverage):
    reference = oracle["reference_mse"]; count = len(a.internal(point, train["n"]))
    scores = {name: a.score(obs, point, bound) for name, obs in (("train", train), ("validation", validation), ("test", test))}
    fit_term = (scores["train"]["mse"]-oracle["scores"]["train"]["mse"])/reference
    count_term = alpha*(count-oracle["proper_count"]); delta = fit_term+count_term
    if count == oracle["proper_count"] and count_term != 0.:
        raise AssertionError("Equal-cardinality count contribution must be exactly zero")
    J = scores["train"]["mse"]/reference+alpha*count
    a.close(delta, J-oracle["J_upper"], "objective decomposition", atol=1e-9)
    epsilon = 1e-8*(1+abs(J)+abs(oracle["J_upper"]))
    preference = ("point_below_true_family_lower_bound" if J < oracle["J_lower"]-epsilon else
                  "true_family_feasible_point_better" if J > oracle["J_upper"]+epsilon else "within_certificate_tolerance")
    topology = a.topology(point, truth, train["n"])
    pool_missing = coverage.get("E_pool_missing_true_connections", "")
    missing = int(pool_missing) if pool_missing not in ("", None) else None
    saved_true = flag(coverage["output_bank_contains_true"]) if "output_bank_contains_true" in coverage else None
    return dict(k_point=count, k_true=oracle["proper_count"], fit_term=fit_term, count_term=count_term,
        delta_J=delta, J_point=J, J_true_upper=oracle["J_upper"], J_true_lower=oracle["J_lower"],
        preference_with_certificate=preference, preference_numeric_tolerance=epsilon,
        equal_cardinality=count == oracle["proper_count"],
        topology_exact=topology["exact"], F1=topology["F1"], FP=topology["FP"], FN=topology["FN"],
        classify_pool_missing=missing > 0 if missing is not None else None,
        true_in_saved_library=saved_true, MSE_selects_wrong=saved_true and not topology["exact"],
        E_pool_missing_true_connections=missing,
        **{name+"_mse": values["mse"] for name, values in scores.items()})


def diagnose(output):
    started = perf_counter(); output = Path(output).resolve(); runtime = output/"runtime"; analysis = output/"analysis"
    protocol = read(output/"protocol.json"); verification = read(analysis/"verification.json")
    completion = read(output/"completion.json")
    if not (verification.get("passed") and verification.get("complete") and verification.get("planned") == 480
            and verification.get("pending") == 0 and verification.get("native_queue_finished")
            and verification.get("MAE_used_for_final_selection") is False
            and completion.get("methods", {}).get("pending", 0) == 0):
        raise RuntimeError("Run this diagnostic only after all 480 attempts and the independent main audit finish")
    helper = (output/verification["analyzer_source_archive"].replace("\\", "/")).resolve()
    if not helper.is_relative_to(output):
        raise ValueError("The archived analysis helper escapes this output")
    if hashlib.sha256(helper.read_bytes()).hexdigest() != verification["analyzer_sha256"]:
        raise ValueError("The archived independent helper changed before import")
    spec = importlib.util.spec_from_file_location("independent_main_audit", helper)
    a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
    if a.sha(helper) != verification["analyzer_sha256"] or a.sha(output/"protocol.json") != verification["protocol_sha256"]:
        raise ValueError("The audited protocol/helper changed")
    for name in ("per_model.csv", "coverage.csv"):
        if a.sha(analysis/name) != verification["artifacts"][name]:
            raise ValueError("An audited analysis table changed")
    for relative, expected in protocol["runtime_file_hashes"].items():
        if a.sha(a.local(runtime.resolve(), relative)) != expected:
            raise ValueError("Frozen runtime/data changed: "+relative)
    for key, expected in verification["native_result_sha256"].items():
        if a.sha(output/"runs"/key/"result.json") != expected:
            raise ValueError("An audited native result changed: "+key)
    if len(protocol["jobs"]) != 48 or len(verification["native_result_sha256"]) != 480:
        raise ValueError("Uniform 48-family diagnosis requires the full frozen panel")
    own = Path(__file__).resolve(); own_hash = a.sha(own)
    verification_hash = a.sha(analysis/"verification.json")
    archived = output/"analysis_source"/own_hash/own.name
    archived.parent.mkdir(parents=True, exist_ok=True); archived.write_bytes(own.read_bytes())
    models = {(r["job"], r["method"]): r for r in table(analysis/"per_model.csv")}
    covered = {(r["job"], r["method"]): r for r in table(analysis/"coverage.csv")}
    catalog = read(runtime/"data/catalog.json"); config = protocol["config"]
    bound, alpha = float(config["response_bound"]), float(config["alpha"])
    oracles, rows, fit_seconds = {}, [], 0.
    for job in protocol["jobs"]:
        meta = catalog[job]; common_file = output/"common"/job/"common.json"
        if a.sha(common_file) != verification["common_sha256"][job]:
            raise ValueError("Audited common preprocessing changed")
        common = read(common_file)
        splits = {s: a.load_observations(a.local(runtime.resolve(), meta[s])) for s in ("train", "validation", "test")}
        truth = a.truth_family(runtime.resolve(), meta, splits["train"]["labels"])
        family = a.canonical(set(truth) | a.bases(splits["train"]["n"]), splits["train"]["n"])
        attempt_started = perf_counter()
        try:
            if common.get("status") != "complete":
                raise RuntimeError("The common normalization reference is unavailable")
            oracle = oracle_fit(a, splits["train"], splits["validation"], splits["test"], family,
                                float(common["reference_mse"]), bound, alpha)
        except Exception as error:
            oracle = dict(status="failed", error=str(error), seconds=perf_counter()-attempt_started,
                          family=[list(C) for C in family])
        fit_seconds += oracle["seconds"]
        oracles[job] = dict(case=meta["case"], seed_index=meta["seed_index"],
                            voltage_sigma=meta["voltage_sigma"], labels=list(splits["train"]["labels"]), **oracle)
        for method in protocol["methods"]:
            record = read(output/"runs"/job/method/"result.json"); main = models[job, method]
            base = dict(job=job, method=method, case=meta["case"], seed_index=meta["seed_index"],
                        voltage_sigma=meta["voltage_sigma"], method_status=main["status"])
            available = main["status"] == "verified" or (main["status"] == "budget_exceeded"
                         and flag(main.get("optimizer_ledger_available")) and not main.get("model_audit_error"))
            if oracle["status"] != "verified" or not available or "result" not in record:
                rows.append(dict(base, origin="final_selected", comparison_status="unavailable",
                                 reason="oracle_failed" if oracle["status"] != "verified" else "no_audited_final_model"))
                continue
            result = record["result"]
            points = [("final_selected", result["selected"]["point"], None, "QP_closed_final")]
            for event in result.get("events", []):
                if event["stage"] == "E":
                    if event.get("point") is not None:
                        points.append(("E_proposal", event["point"], event["accepted"], "QP_closed_proposal"))
                    if event.get("raw_point") is not None:
                        points.append(("E_raw", event["raw_point"], event["accepted"], "native_feasible_incumbent"))
            for origin, point, accepted, fit_kind in points:
                values = decomposition(a, splits["train"], splits["validation"], splits["test"], point,
                    truth, oracle, alpha, bound, covered.get((job, method), {}))
                rows.append(dict(base, origin=origin, comparison_status="available", point_fit=fit_kind,
                    E_accepted=accepted, actual_final_selection=origin == "final_selected",
                    matches_final_selected_model=a.fingerprint(point) == a.fingerprint(result["selected"]["point"]), **values))
    a.write_json(analysis/"true_family_oracles.json", dict(planned=48, fits=oracles, used_by_algorithm=False))
    a.write_csv(analysis/"objective_decomposition.csv", rows)
    for key, expected in verification["native_result_sha256"].items():
        if a.sha(output/"runs"/key/"result.json") != expected:
            raise ValueError("Native search/selection changed during oracle diagnosis")
    if (a.sha(own) != own_hash or a.sha(helper) != verification["analyzer_sha256"]
            or a.sha(analysis/"verification.json") != verification_hash
            or any(name == "gurobipy" or name.startswith("gurobipy.") for name in sys.modules)):
        raise RuntimeError("Diagnostic source changed or imported a native optimizer")
    metadata = dict(protocol_sha256=verification["protocol_sha256"],
        analysis_verification_sha256=verification_hash,
        diagnostic_source_sha256=own_hash, source_archive=archived.relative_to(output).as_posix(),
        helper_source_sha256=a.sha(helper), helper_source_archive=helper.relative_to(output).as_posix(),
        artifacts={name: a.sha(analysis/name) for name in ("true_family_oracles.json", "objective_decomposition.csv")},
        used_by_algorithm=False, native_solver_calls=0, oracle_family_fits_planned=48,
        oracle_family_fits_verified=sum(v["status"] == "verified" for v in oracles.values()),
        oracle_family_fits_failed=sum(v["status"] != "verified" for v in oracles.values()),
        final_rows=sum(r["origin"] == "final_selected" for r in rows), all_rows=len(rows),
        total_diagnostic_seconds=perf_counter()-started, oracle_fit_seconds=fit_seconds,
        continuous_BVLS_calls_verified_oracles=2*sum(v["status"] == "verified" for v in oracles.values()),
        sign_convention="delta_J = J_point - J_true_upper; positive favours the feasible true-family oracle",
        preference_scope="point-vs-true-fixed-family box certificate; raw E incumbents need not be QP-optimal",
        causal_noise_bias_attribution_attempted=False, alpha_retuned=False, primary_selection_unchanged=True)
    a.write_json(analysis/"oracle_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    result = diagnose(parser.parse_args().output)
    raise SystemExit(0 if result["oracle_family_fits_verified"] == 48 else 2)
