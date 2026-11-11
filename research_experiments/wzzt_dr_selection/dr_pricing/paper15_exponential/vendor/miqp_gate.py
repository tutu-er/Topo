"""Physical-MSE exit gates and a data-only audit of frozen E proposals.

Standard library only. No observations, truth, F1, fitting code or solver is
used to define a gate. The audit reuses SHA-verified saved objectives/weights;
it does not generate, refit, validate or accept a new topology itself.

Public API:
  pure_mse_gate(old_mse, new_mse, *, K, tau=.001, epsilon_internal=1e-10)
  old_regularized_gate(old_mse, new_mse, old_count, new_count, *, K,
                       f0_mse, alpha=.003, tau=.001, epsilon_internal=1e-10)
Both return a dict with accepted, delta_mse, relative_improvement and the
physical-MSE threshold. Callers separately enforce time/feasibility guards.

  audit_saved(output, *, method='T003', grid=TAU_GRID)
returns an evidence dict without writing files. CLI --write is explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median

TAU_GRID = (0., 1e-4, 1e-3, 3e-3, 1e-2)
ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "research_experiments/full_ablation/engine.py",
    "research_experiments/rnj_count/engine.py",
    "research_experiments/rnj_small_seed/engine.py",
    "research_experiments/joint_exit/model.py",
    "research_experiments/iterative_exit/zero.py",
    "research_experiments/revocable/engine.py",
    "research_experiments/miqp_loss_benchmark_20261111/run.py",
    "research_experiments/miqp_loss_benchmark_20261111/squared_backend.py",
)


def _nonnegative(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(name+" must be finite and nonnegative.")
    return value


def _positive(value, name):
    value = _nonnegative(value, name)
    if value == 0:
        raise ValueError(name+" must be positive.")
    return value


def _count(value, name):
    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError(name+" must be a nonnegative integer.")
    return int(value)


def pure_mse_gate(old_mse, new_mse, *, K, tau=.001, epsilon_internal=1e-10):
    """Accept iff M_old-M_new > epsilon_internal/K + tau*M_old.

    M is unscaled physical mean squared error, K is the already-frozen positive
    training-loss scale, and tau is dimensionless. A tie is rejected. K enters
    this pure-MSE rule only through conversion of the existing numerical floor.
    """
    old_mse = _nonnegative(old_mse, "old_mse"); new_mse = _nonnegative(new_mse, "new_mse")
    K = _positive(K, "K"); tau = _nonnegative(tau, "tau")
    epsilon_internal = _nonnegative(epsilon_internal, "epsilon_internal")
    delta = old_mse-new_mse; epsilon = epsilon_internal/K; threshold = epsilon+tau*old_mse
    relative = delta/old_mse if old_mse > 0 else (0. if delta == 0 else None)
    return dict(rule="pure_physical_mse_relative", accepted=delta > threshold,
                old_mse=old_mse, new_mse=new_mse, delta_mse=delta,
                relative_improvement=relative,
                relative_improvement_kind="finite" if relative is not None else "negative_infinite_zero_baseline",
                K=K, tau=tau, epsilon_internal=epsilon_internal, epsilon_physical=epsilon,
                threshold_physical=threshold, margin_physical=delta-threshold)


def old_regularized_gate(old_mse, new_mse, old_count, new_count, *, K,
                         f0_mse, alpha=.003, tau=.001, epsilon_internal=1e-10):
    """Reproduce the old final-E numeric gate in physical MSE units.

    k counts retained proper internal supports, including selected zero atoms.
    With K=f0_MAE/f0_MSE, the old score divided by K is M+alpha*f0_MSE*k.
    Positive delta_count therefore means net deletion, not addition.
    """
    result = pure_mse_gate(old_mse, new_mse, K=K, tau=tau, epsilon_internal=epsilon_internal)
    old_count, new_count = _count(old_count, "old_count"), _count(new_count, "new_count")
    f0_mse, alpha = _nonnegative(f0_mse, "f0_mse"), _nonnegative(alpha, "alpha")
    delta_count = old_count-new_count; reward = alpha*f0_mse*delta_count
    gain = result["delta_mse"]+reward
    result.update(rule="old_count_regularized_in_physical_mse_units", accepted=gain > result["threshold_physical"],
        old_count=old_count, new_count=new_count, delta_count=delta_count,
        f0_mse=f0_mse, alpha=alpha, count_reward_physical=reward,
        regularized_gain_physical=gain, margin_physical=gain-result["threshold_physical"])
    return result


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _point(point):
    rows = sorted(zip(point["supports"], point["r"], point["x"]), key=lambda row: (len(row[0]), tuple(row[0])))
    return dict(supports=[list(C) for C, _, _ in rows], r=[float(r) for _, r, _ in rows], x=[float(x) for _, _, x in rows])


def _close(a, b, message):
    a, b = float(a), float(b)
    if not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > 1e-10+1e-7*max(abs(a), abs(b)):
        raise ValueError(message)


def _before_exit(result):
    """Locate the exact saved prefix endpoint using frozen finish append rules.

    An accepted E appends one point. The outer final-exit zero-cleanup appends
    one more point iff any deletion was accepted. No nearest-loss matching.
    """
    trajectory, record = result["trajectory"], result["exit_record"]
    if not isinstance(record.get("raw_point"), dict):
        raise ValueError("A saved raw final-E proposal is required.")
    outer_zero = any(row.get("stage") == "final_exit" and row.get("accepted")
                     for row in trajectory.get("zero_events", []))
    index = len(trajectory["path"])-1-int(bool(record["accepted"]))-int(outer_zero)
    if index < 0:
        raise ValueError("Final-E append counts exceed the saved path length.")
    endpoint_index = index+int(bool(record["accepted"]))
    if _point(trajectory["path"][endpoint_index]) != _point(record["point"]):
        raise ValueError("The exact final-E endpoint does not reproduce from the saved path.")
    if _point(trajectory["path"][-1]) != _point(trajectory["point"]):
        raise ValueError("The saved final path and trajectory endpoint differ.")
    return trajectory["path"][index], index, outer_zero


def _distribution(rows):
    relative = [row["relative_improvement"] for row in rows if row["relative_improvement"] is not None]
    return dict(n=len(rows), delta_count=dict(sorted(Counter(str(row["delta_count"]) for row in rows).items())),
        raw_mse_improves=sum(row["delta_mse"] > row["epsilon_physical"] for row in rows),
        raw_mse_worsens=sum(-row["delta_mse"] > row["epsilon_physical"] for row in rows),
        raw_mse_numerical_ties=sum(abs(row["delta_mse"]) <= row["epsilon_physical"] for row in rows),
        retained_mse_worsens=sum(row["retained_point_mse"]-row["old_mse"] > row["epsilon_physical"] for row in rows),
        relative_improvement_min=min(relative) if relative else None,
        relative_improvement_median=median(relative) if relative else None,
        relative_improvement_max=max(relative) if relative else None,
        actual_selected_E_added=sum(row["actual_selected_E_added"] for row in rows))


def audit_saved(output, *, method="T003", grid=TAU_GRID):
    """Read-only numeric-gate audit of all 48 verified frozen raw E proposals.

    Current raw proposals still come from the old count-regularized MIQP.
    Counterfactual gate decisions do not assert that new pure-MSE-optimal
    proposals or new contracted/QP-refitted output-library rows are available.
    """
    output = Path(output).resolve()
    if method not in ("T003", "noT"):
        raise ValueError("Only a declared parent MIQP method may be audited.")
    protocol, catalog = _read(output/"protocol.json"), _read(output/"catalog.json")
    verification = _read(output/"analysis/verification.json")
    if (not verification["passed"] or not verification["complete"] or verification["new_native_verified"] != 96
            or verification["reused_parent_verified"] != 96 or protocol["planned"] != 96 or len(protocol["jobs"]) != 48):
        raise ValueError("The complete 96-new/96-parent verification is required.")
    if verification["protocol_sha256"] != _sha(output/"protocol.json") or verification["catalog_sha256"] != _sha(output/"catalog.json"):
        raise ValueError("The verified frozen protocol/catalog changed.")
    source_hashes = {}
    for name in SOURCE_FILES:
        measured = _sha(output/"source"/name)
        if measured != protocol["source_hashes"][name]:
            raise ValueError("A frozen gate/control-flow source changed: "+name)
        source_hashes[name] = measured
    grid = tuple(_nonnegative(value, "tau grid value") for value in grid)
    if len(set(grid)) != len(grid) or not grid:
        raise ValueError("A nonempty unique tau grid is required.")
    rows, result_hashes, audit_hashes = [], {}, {}
    for job in protocol["jobs"]:
        file = output/"jobs"/job/method/"result.json"; result = _read(file)
        audit_file = output/"analysis/jobs"/job/"verified.json"; audit = _read(audit_file)
        if (audit["protocol_sha256"] != verification["protocol_sha256"] or audit["analyzer_sha256"] != verification["analyzer_sha256"]
                or audit["raw_hashes"]["scaled_mse/"+method] != _sha(file)):
            raise ValueError("A native result differs from its current independent verification.")
        result_hashes[job] = _sha(file); audit_hashes[job] = _sha(audit_file)
        before, before_index, outer_zero = _before_exit(result)
        record = result["exit_record"]; raw = record["raw_point"]
        K = _positive(result["normalization"]["K"], "saved K")
        f0_mse = _positive(result["normalization"]["f0_mse"], "saved f0_mse")
        _close(K*f0_mse, result["reference"], "The frozen reference normalization differs.")
        old_mse, new_mse = float(before["train_mse"]), float(raw["objective"])/K
        _close(before["objective"], K*old_mse, "The saved prefix point does not use K*MSE.")
        n = len(catalog[job]["labels"])
        count = lambda point: sum(1 < len(C) < n for C in point["supports"])
        old = old_regularized_gate(old_mse, new_mse, count(before), count(raw), K=K, f0_mse=f0_mse)
        _close(record["before_score"], K*(old_mse+.003*f0_mse*count(before)), "Old E before-score does not reproduce.")
        _close(record["raw_score"], K*(new_mse+.003*f0_mse*count(raw)), "Old E raw-score does not reproduce.")
        if old["accepted"] != bool(record["accepted"]):
            raise ValueError("The stored E decision has an additional budget/control guard; numeric-only audit is insufficient.")
        selected_index = int(result["selected"]["index"])
        selected_E = bool(record["accepted"]) and before_index < selected_index < len(result["trajectory"]["path"])
        rows.append(dict(job=job, method=method, old_accepted=bool(record["accepted"]), old_mse=old_mse,
            raw_proposal_mse=new_mse, retained_point_mse=float(record["point"]["train_mse"]),
            old_count=count(before), raw_proposal_count=count(raw), delta_count=old["delta_count"],
            delta_mse=old["delta_mse"], relative_improvement=old["relative_improvement"],
            K=K, f0_mse=f0_mse, epsilon_physical=old["epsilon_physical"],
            old_regularized_gain_physical=old["regularized_gain_physical"], old_threshold_physical=old["threshold_physical"],
            before_path_index=before_index, actual_selected_index=selected_index,
            actual_selected_E_added=selected_E, final_zero_append=outer_zero,
            before_parameter_sha256=_digest(_point(before)), raw_proposal_parameter_sha256=_digest(_point(raw)),
            retained_parameter_sha256=_digest(_point(record["point"])), E_solver_status=result["exit_event"].get("status")))
    summaries = []
    for tau in grid:
        qualify = lambda row: pure_mse_gate(row["old_mse"], row["raw_proposal_mse"], K=row["K"], tau=tau)["accepted"]
        newly_allowed = [row["job"] for row in rows if not row["old_accepted"] and qualify(row)]
        newly_rejected = [row["job"] for row in rows if row["old_accepted"] and not qualify(row)]
        summaries.append(dict(tau=tau, raw_proposals_passing=sum(qualify(row) for row in rows),
            old_rejected_newly_allowed=len(newly_allowed), old_accepted_newly_rejected=len(newly_rejected),
            actual_selected_E_points_removed=sum(row["old_accepted"] and not qualify(row) and row["actual_selected_E_added"] for row in rows),
            newly_allowed_jobs=newly_allowed, newly_rejected_jobs=newly_rejected))
    return dict(passed=True, method=method, planned=48, audited=len(rows), parameter_selection_performed=False,
        truth_used=False, F1_used=False, observations_opened=False, new_fit_calls=0, new_native_optimizer_calls=0,
        old_alpha=.003, old_tau=.001, contraction_resolution_unchanged=.24, epsilon_internal=1e-10,
        old_formula="delta_M + .003*f0_MSE*(k_old-k_raw) > 1e-10/K + .001*M_old",
        pure_formula="delta_M > 1e-10/K + tau*M_old", count_definition="all retained proper internal supports, including selected zeros",
        sign_definition="delta_M=M_old-M_raw; delta_count=k_old-k_raw; positive values mean fit improvement/net deletion",
        all=_distribution(rows), old_accepted=_distribution([r for r in rows if r["old_accepted"]]),
        old_rejected=_distribution([r for r in rows if not r["old_accepted"]]), grid=summaries, rows=rows,
        protocol_sha256=_sha(output/"protocol.json"), catalog_sha256=_sha(output/"catalog.json"),
        native_verification_sha256=_sha(output/"analysis/verification.json"), native_analyzer_sha256=verification["analyzer_sha256"],
        result_sha256=result_hashes, per_input_verification_sha256=audit_hashes,
        source_manifest_sha256=_digest(protocol["source_hashes"]), source_sha256=source_hashes,
        gate_helper_sha256=_sha(__file__),
        interpretation="gate-only counterfactual on old regularized raw proposals; all own-path points remain available to global validation; newly allowed raw families require their own QP/contraction/validation evaluation before claiming a changed final model")


def numerical_examples():
    """Meaningful unit examples, independent of benchmark outcomes."""
    count_tradeoff = dict(old_mse=1., new_mse=1.0005, old_count=3, new_count=2, K=1., f0_mse=1.)
    old = old_regularized_gate(**count_tradeoff)
    pure = pure_mse_gate(count_tradeoff["old_mse"], count_tradeoff["new_mse"], K=1.)
    if not old["accepted"] or pure["accepted"]:
        raise AssertionError("Count reduction should explain the old acceptance of a 0.05% MSE increase.")
    equal_old = old_regularized_gate(1., .998, 3, 3, K=2., f0_mse=.4)
    equal_pure = pure_mse_gate(1., .998, K=2.)
    if (equal_old["accepted"], equal_old["margin_physical"]) != (equal_pure["accepted"], equal_pure["margin_physical"]):
        raise AssertionError("Equal-count old and pure gates must coincide at the same tau.")
    a = pure_mse_gate(1., .998, K=2.)
    b = pure_mse_gate(100., 99.8, K=.02)
    _close(b["threshold_physical"], 100*a["threshold_physical"], "Physical MSE unit mapping failed.")
    if a["accepted"] != b["accepted"]:
        raise AssertionError("Changing MSE units with inverse K must preserve the decision.")
    internal = pure_mse_gate(1., .998, K=20., epsilon_internal=1e-9)
    if a["threshold_physical"] != internal["threshold_physical"]:
        raise AssertionError("Internal objective scaling must map the numeric epsilon as well.")
    tie = pure_mse_gate(2., 1., K=1., tau=.5, epsilon_internal=0.)
    if tie["accepted"]:
        raise AssertionError("The gate is strict, so equality must be rejected.")
    return dict(passed=True, count_reduction_accepts_MSE_increase=dict(old=old, pure=pure),
        equal_count_equivalence=True, physical_unit_rescaling=dict(original=a, rescaled=b),
        internal_scale_with_mapped_epsilon=True, strict_tie_rejected=True,
        scale_note="Relative improvement is dimensionless; changing internal objective scale while leaving epsilon_internal fixed changes only the converted numeric floor.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, help="Read-only audit of a complete frozen MIQP output.")
    parser.add_argument("--method", choices=("T003", "noT"), default="T003")
    parser.add_argument("--examples", action="store_true", help="Run only small standard-library numerical examples.")
    parser.add_argument("--write", type=Path, help="Explicit destination for the returned evidence JSON.")
    args = parser.parse_args()
    if not args.examples and args.audit is None:
        parser.error("Choose --examples or --audit OUTPUT.")
    evidence = audit_saved(args.audit, method=args.method) if args.audit is not None else numerical_examples()
    if args.examples and args.audit is not None:
        evidence["numerical_examples"] = numerical_examples()
    payload = json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(payload, encoding="utf-8", newline="\n")
    else:
        print(payload, end="")
