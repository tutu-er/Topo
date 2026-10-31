"""Read-only independent arithmetic audit; no planner/identifier import or solve.

Only explicitly synthetic qualification amplitudes/quotes and aggregates are
exported. No household, network-parameter, P/Q/V or AC trajectories are copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


RUNS = ("release_socp", "release_misocp", "release_socp_day", "release_misocp_day")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit(root, repo):
    package = repo/"research_experiments/wzzt_dr_selection/dr_conic_pricing"
    rows = []
    for name in RUNS:
        folder = root/name
        p, r, v, protocol = (read(folder/f) for f in
                            ("problem.json", "optimization.json", "validation.json", "protocol.json"))
        assert p["metadata"]["calibrated"] is False
        for f, digest in protocol["source_sha256"].items():
            assert hashlib.sha256((package/f).read_bytes()).hexdigest() == digest, f
        canonical = (json.dumps(p, sort_keys=True, ensure_ascii=False, allow_nan=False,
                               separators=(",", ":"))+"\n").encode()
        assert hashlib.sha256(canonical).hexdigest() == protocol["canonical_problem_sha256"]
        h, f, d, g, mu, x = (np.asarray(value, float) for value in
                             (p["profiles"], p["background_factors"], p["response_factors"],
                              p["benefit_directions"], p["mu0"], r["amplitudes"]))
        assert r["primary"]["status"] == r["secondary"]["status"] == "optimal"
        assert r["status"] == "optimal" and r["feasible_incumbent"]
        lam, e = np.asarray(p["elasticities"]), np.asarray(p["reduction_hours"])
        price = x/lam
        q = np.abs(d).sum(axis=1)
        response = (h*x)@d
        if p["benefit_mode"] == "incremental":
            mean, loading = g@(h@x), g@response
        else:
            mean = np.asarray(p["benefit_offsets"])+g@(mu+h@x)
            loading = g@(f+response)
        std = np.linalg.norm(loading, axis=1)/np.sqrt(3.)
        eps = np.asarray(p["risk_allocations"])
        lcb = mean-np.sqrt((1-eps)/eps)*std
        support_lower = mean-np.abs(loading).sum(axis=1)
        fee = np.sum(e/lam*x*x)
        safe_fee = np.sum(e/lam*(1+q)*x*x)
        exact_fee = fee+np.abs((e/lam*x*x)@d).sum()
        A, b = np.asarray(p["device_matrix"]), np.asarray(p["device_rhs"])
        residual = A@x+np.abs((A*x)@d).sum(axis=1)-b
        assert np.min(x) >= 0
        tol = r["numerical_check_tolerance"]
        assert np.max(price-np.asarray(p["price_upper"]), initial=0) <= tol
        assert np.max((1+q)*x-np.asarray(p["actual_amplitude_upper"]), initial=0) <= tol
        assert residual.max(initial=0) <= tol
        assert exact_fee <= safe_fee+tol <= p["budget"]+2*tol
        for group in p["exclusive_groups"]:
            assert np.count_nonzero(x[group] > tol) <= 1
        assert np.allclose(r["prices"], price, rtol=0, atol=1e-12)
        assert abs(float(np.min(lcb))-r["achieved_benefit_lower_bound"]) <= 1e-12
        assert float(np.min(lcb)) >= r["secondary_benefit_floor"]-tol
        assert abs(r["evaluation"]["expected_cost"]-fee) <= 1e-12
        assert v["samples"] == v["successes"]+v["failures"]
        assert v["samples"] == v["paired_no_dr"]["samples"]
        assert v["device_failures"] == v["budget_failures"] == v["amplitude_failures"] == 0
        if p["metadata"]["slots"] == 96:
            hp = h[:h.shape[0]//2].reshape(96, 2, len(x))
            assert np.max(np.abs(hp.sum(axis=0))) <= 1e-12
        rows.append(dict(
            run=name, mode=r["mode"], slots=p["metadata"]["slots"], source="explicit synthetic qualification",
            calibrated=False, amplitudes=x.tolist(), prices=price.tolist(),
            positive_contract_ids=r["positive_contract_ids"], status=r["status"],
            primary=r["primary"], secondary=r["secondary"],
            reliable_benefit=float(np.min(lcb)), benefit_means=mean.tolist(),
            benefit_standard_deviations=std.tolist(), support_lower_bounds=support_lower.tolist(),
            expected_cost=float(fee), hard_budget_bound=float(safe_fee), exact_box_worst_cost=float(exact_fee),
            budget=p["budget"], maximum_device_violation=float(max(0, residual.max(initial=0))),
            model_size=r["model_size"], compilation=r["compilation"],
            validation_samples=v["samples"], validation_seed=v["seed"], successes=v["successes"],
            failures=v["failures"], budget_failures=0, device_failures=0, amplitude_failures=0,
            no_dr_successes=v["paired_no_dr"]["successes"], mean_actual_payment=v["mean_payment"],
            maximum_actual_payment=v["maximum_payment"], confidence=p["confidence"],
            canonical_problem_sha256=protocol["canonical_problem_sha256"],
            source_sha256=protocol["source_sha256"], environment=protocol["environment"],
            optimization_sha256=hashlib.sha256((folder/"optimization.json").read_bytes()).hexdigest(),
            validation_sha256=hashlib.sha256((folder/"validation.json").read_bytes()).hexdigest()))
    xml = ET.parse(root/"final_regression_v4.xml").getroot()
    tests = list(xml.iter("testcase"))
    assert len(tests) == 81 and not list(xml.iter("failure")) and not list(xml.iter("error"))
    return dict(schema_version=1, status="complete", runs=rows,
                qualification_runs=4, new_tests=56, existing_compatibility_tests=25,
                final_test_total=81, final_test_passed=81,
                test_receipt_sha256=hashlib.sha256((root/"final_regression_v4.xml").read_bytes()).hexdigest(),
                tested_model_scope="conservative reliable excitation, resource constraints and incentive budget",
                topology_identification_runs=0, ac_runs=0, measured_calibration=False,
                test_corrections=[
                    "Initial 36-test stage: 34 pass, two status13 cone failures; conic convergence setting documented and one-dimensional norm linearized.",
                    "Initial 54-test integration: 53 pass, one tiny useless fee residual; secondary objective changed to equivalent linear fee norm.",
                    "First compatibility CLI invocation had a malformed pytest report argument and ran zero tests; corrected invocation passed25.",
                    "Initial 96-slot MISOCP secondary reported optimal but failed unsquared benefit recomputation; retained primary. Three unchanged-model diagnostics checkedMIQCP0/NumericFocus1/NumericFocus2; finalNumericFocus2passes. All original receipts retained privately."],
                replicate_scope="Earlier numerical/debug reruns reuse factor seeds and are not additional independent paths.",
                probability_scope="excitation success under the declared law; never complete topology recovery")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.runs_root, args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "qualification_runs", "final_test_passed", "topology_identification_runs")}))


if __name__ == "__main__":
    main()
