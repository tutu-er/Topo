"""Reproducible one-thread hand checks of the isolated squared-loss backend.

Only tiny synthetic observations are used.  --source selects immutable native
code; every imported native module must originate inside it.  No benchmark job,
real observation array or still-running optimizer is read or modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(source):
    source = Path(source).resolve()
    if not source.is_dir():
        raise ValueError("An existing frozen source directory is required")
    started = perf_counter()
    sys.path[:0] = [str(source), str(source / "rnj_wzzt_core")]
    # A live regular package can otherwise take precedence over a frozen
    # namespace package even when the frozen root is first on sys.path.
    import research_experiments
    research_experiments.__path__ = [str(source / "research_experiments"), *research_experiments.__path__]
    backend_file = Path(__file__).with_name("squared_backend.py")
    spec = importlib.util.spec_from_file_location("verified_squared_backend", backend_file)
    backend = importlib.util.module_from_spec(spec); spec.loader.exec_module(backend)
    from rnj_wzzt.estimation import laminar_l1_milp as core
    core_file = Path(core.__file__).resolve()
    if not core_file.is_relative_to(source):
        raise AssertionError(f"The linear constraint builder is outside frozen source: {core_file}")
    # Known minimizer: r=1.5, mean squared residual=(.5^2+.5^2)/2=.25.
    single = core._PreparedScenarios((0,), (np.ones((2, 1)),), (np.zeros((2, 1)),),
                                     (np.array([[1.], [2.]]),))
    one_reference = backend.reference_fit(single, [(0,)], seconds=15.)
    assert abs(one_reference["mse"]-.25) < 1e-8 and abs(one_reference["r"][0]-1.5) < 1e-7
    # Nonzero-residual three-leaf check catches an accidentally retained L1 term.
    P = np.eye(3); Q = np.zeros((3, 3)); z = np.array([0., 1., 1.]); Y = P+np.outer(z, z)
    data = core._PreparedScenarios((0, 1, 2), (P,), (Q,), (Y,))
    basis = backend.canonical([(0,), (1,), (2,), (0, 1, 2)])
    wrong = backend.canonical([*basis, (0, 1)])
    reference = backend.reference_fit(data, wrong, seconds=15.)
    full_index = list(reference["supports"]).index((0, 1, 2))
    full_r = reference["r"][full_index]
    assert abs(reference["mse"]-4/27) < 1e-8 and abs(full_r-1/3) < 1e-7
    K = (2/9)/(4/27)
    manifest = backend.install(K)
    assert len(manifest["aliases"]) == 14
    for item in manifest["native_module_sources"].values():
        assert Path(item["path"]).resolve().is_relative_to(source)
        assert sha(item["path"]) == item["sha256"]
    calibration = dict(a=1., b=1., median_root_depth=1.)
    bank_one = backend.SquaredBank(single, [(0,)], calibration, seconds=15.)
    try:
        one_fitted = bank_one.fit([(0,)], preserve=True)
        assert abs(one_fitted["objective"]-.375) < 1e-8
        one_audit = bank_one.lp.solves[-1]
    finally:
        bank_one.close()
    pool = backend.canonical([*basis, (0, 1), (0, 2), (1, 2)])
    bank = backend.SquaredBank(data, pool, calibration, seconds=15.)
    try:
        initial = bank.fit(basis, preserve=True)
        wrong_fitted = bank.fit(wrong, preserve=True)
        assert abs(wrong_fitted["objective"]-2/9) < 1e-8
        penalty = .003*initial["objective"]
        from research_experiments.rnj_count.engine import original_step, joint_fit
        fitted, forward = original_step(bank, initial, penalty=penalty, deadline=perf_counter()+30., per_mip=15.)
        assert forward["accepted"] and forward["status"] == 2 and forward["solver"] == "gurobi_convex_MIQP"
        assert backend.mse(data, fitted) < 1e-8 and (1, 2) in backend.canonical(fitted["supports"])
        assert abs(forward["objective"]-penalty) < 1e-7
        # T must also call the squared structural solver and fixed-family QP.
        # The current wrong old atom is released by the singleton/full keep.
        tail_point, tail = joint_fit(bank, wrong_fitted, basis, penalty=penalty, seconds=15., stage="toy_joint_tail")
        assert tail["accepted"] and tail["status"] == 2 and tail["solver"] == "gurobi_convex_MIQP"
        assert backend.mse(data, tail_point) < 1e-8 and (1, 2) in backend.canonical(tail_point["supports"])
        assert abs(tail["decoded_score"]-penalty) < 1e-7 and abs(tail["regularized_bound"]-penalty) < 1e-7
        from research_experiments.joint_exit.model import build, solve_checkpoints
        model, meta = build(bank.scaled, pool, penalty=penalty)
        snapshots, exit_record = solve_checkpoints(bank, model, meta, initial, prefix_seconds=0., budgets=(15.,))
        exit_audit = backend.audit_vector(model, snapshots[15.]["values"])
        assert exit_record["status"] == 2 and exit_audit["valid"]
        assert abs(exit_audit["objective"]-penalty) < 1e-7 and abs(exit_record["final_bound"]-penalty) < 1e-7
        qp_audits = list(bank.lp.solves)
    finally:
        bank.close()
    # Installation only replaces in-process aliases; all native files stay intact.
    source_unchanged = all(sha(item["path"]) == item["sha256"] for item in manifest["native_module_sources"].values())
    assert source_unchanged
    return dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(), seconds=perf_counter()-started,
                script_sha256=sha(__file__), backend_sha256=sha(backend_file), source=str(source),
                linear_constraint_builder=dict(path=str(core_file), sha256=sha(core_file)),
                native_origins_within_requested_source=True, native_sources_unchanged=source_unchanged,
                install_manifest=manifest, normalization=dict(f0_MAE=2/9, f0_MSE=4/27, K=K),
                hand_checks=dict(one_node_MSE=one_reference["mse"], one_node_expected_MSE=.25,
                                 one_node_r=one_reference["r"][0], one_node_expected_r=1.5,
                                 one_node_normalized_QP=one_fitted["objective"],
                                 wrong_family_MSE=reference["mse"], wrong_family_expected_MSE=4/27,
                                 wrong_family_full_r=full_r, wrong_family_expected_full_r=1/3,
                                 wrong_family_normalized_objective=wrong_fitted["objective"], expected_normalized_wrong=2/9,
                                 forward_MSE=backend.mse(data, fitted), tail_MSE=backend.mse(data, tail_point),
                                 expected_MIQP_objective=penalty, forward_objective=forward["objective"],
                                 forward_bound=forward["dual_bound"], tail_score=tail["decoded_score"],
                                 tail_bound=tail["regularized_bound"], exit_objective=exit_audit["objective"], exit_bound=exit_record["final_bound"]),
                statuses=dict(one_reference_QP=one_reference["audit"]["status"], three_leaf_reference_QP=reference["audit"]["status"],
                              normalized_one_QP=one_audit["status"], forward_MIQP=forward["status"],
                              tail_MIQP=tail["status"], final_exit_MIQP=exit_record["status"]),
                forward_record=forward, tail_record=tail, exit_record=exit_record, exit_audit=exit_audit,
                fixed_family_QP_audits=[one_reference["audit"], reference["audit"], one_audit, *qp_audits],
                fixed_family_solver_calls=3+len(qp_audits), structural_solver_calls=3, solver_threads=1,
                ridge=0., positive_impedance_floor=0., real_benchmark_jobs_started=0,
                existing_optimizers_stopped=0, physical_truth_read=False,
                x_identifiability="Q is zero in these toys; x coefficients are not asserted unique")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/day96_t_benchmark_20261110/source")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/miqp_loss_benchmark_20261111/backend_verification.json")
    args = parser.parse_args()
    receipt = verify(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: receipt[key] for key in ("passed", "seconds", "hand_checks", "statuses", "fixed_family_solver_calls", "structural_solver_calls")}))
