"""Offline audit of archived shared-square pilots; no optimizer is called.

Run as a package module from the repository root. The archived drivers retain
their original machine paths as provenance; this audit uses repository-relative
evidence and package imports instead.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .adaptive_history import evaluate_frozen_pair_actual
from .candidates import RadialCandidate
from .likelihood import MeasurementModel, Observations

BASE = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = BASE / "evidence" / "shared_power_square_upper_compact"
SUPPORTS = {"clade_02": ((0, 2),), "clade_01": ((0, 1),),
            "star": (), "clade_12": ((1, 2),)}


def _original_bytes(path):
    """Read an original artifact, transparently undoing archive-only gzip."""
    if path.exists():
        return path.read_bytes()
    return gzip.decompress(path.with_name(path.name + ".gz").read_bytes())


def _json(path):
    return json.loads(_original_bytes(path))


def _packed(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def _sha(path):
    return hashlib.sha256(_original_bytes(path)).hexdigest()


def verify_archive_manifest(root=DEFAULT_ARCHIVE):
    root = Path(root)
    manifest = _json(root / "archive_manifest.json")
    count = 0
    total = 0
    for entry in manifest["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("manifest entry escapes archive")
        assert path.stat().st_size == entry["bytes"]
        assert _sha(path) == entry["sha256"], entry["path"]
        if entry.get("compression") == "gzip-original-bytes":
            original = gzip.decompress(path.read_bytes())
            assert len(original) == entry["original_bytes"]
            assert hashlib.sha256(original).hexdigest() == entry["original_sha256"]
        count += 1
        total += entry["bytes"]
    assert count == manifest["file_count"] and total == manifest["total_bytes"]
    return {"manifest_files_verified": count, "manifest_bytes_verified": total}


def verify_single_upper_saved(root):
    """Replay the preceding same-pool single-upper H1 comparison."""
    root = Path(root)
    manifest = _json(root / "pair_manifest.json")
    summary = _json(root / "comparison_summary.json")
    assert _sha(root / "reconstructed_pairs_and_witnesses.npz") == manifest["archive_sha256"]
    before = _json(root / "source_hashes_before.json")
    after = _json(root / "source_hashes_after.json")
    assert before["frozen_core_sha256"] == after["frozen_core_sha256"] and after["unchanged"]
    assert before["new_module_sha256"] == after["new_module_sha256"]
    with np.load(root / "reconstructed_pairs_and_witnesses.npz", allow_pickle=False) as data:
        pairs = [{key: data[f"pair_{i}_{key}"].copy()
                  for key in ("c", "B", "A", "F", "lower", "upper", "scale", "witness_0")}
                 for i in range(24)]
    for name, pair in zip(manifest["names"], pairs):
        digest = hashlib.sha256(name.encode())
        for key in ("c", "B", "A", "F", "lower", "upper", "scale"):
            array = pair[key]
            digest.update(str(array.shape).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        assert digest.hexdigest() == manifest["fingerprints"][name]
        for row in pair["B"]:
            assert len(np.unique(np.flatnonzero(row != 0.0) // 3)) <= 1
        assert np.all(pair["witness_0"] >= pair["lower"]) and np.all(pair["witness_0"] <= pair["upper"])
    max_primal = max_tangent = 0.0
    point_lowers = []
    for filename in ("constant_qp.json", "none_qp.json", "shared_selected_point_qp.json"):
        point = _json(root / filename)
        assert point["complete"] and len(point["pair_results"]) == 24
        for pair, record in zip(pairs, point["pair_results"]):
            primal, tangent = replay_qp_record(pair, {**record, "u_kw": point["u_kw"]})
            max_primal = max(max_primal, primal)
            max_tangent = max(max_tangent, tangent)
        assert min(record["q_lower"] for record in point["pair_results"]) == point["q_lower"]
        assert min(record["q_upper"] for record in point["pair_results"]) == point["q_upper"]
        point_lowers.append(point["q_lower"])
    snapshots = {}
    for label in ("old", "shared"):
        raw = gzip.decompress((root / f"{label}_model_snapshot.json.gz").read_bytes())
        assert hashlib.sha256(raw).hexdigest() == summary[label]["snapshot_sha256"]
        snapshot = snapshots[label] = json.loads(raw)
        assert len(snapshot["lower"]) == summary[label]["model_variables"]
        assert len(snapshot["rows"]) == summary[label]["model_rows"]
    assert summary["old"]["global_best_bound"] is None and summary["old"]["incumbent"] is None
    assert summary["shared"]["global_best_bound"] >= max(point_lowers) - 1e-6
    with np.load(root / "master_incumbents.npz", allow_pickle=False) as data:
        x = data["shared_master_x"].copy()
    snapshot = snapshots["shared"]
    shared = summary["shared"]
    record = {"n_variables": shared["model_variables"], "n_rows": shared["model_rows"],
              "objective_value": shared["incumbent"] / shared["scale"],
              "objective_bound": shared["global_best_bound"] / shared["scale"],
              "raw_minimization_fun": shared["raw_minimization_fun"],
              "raw_minimization_mip_dual_bound": shared["raw_minimization_mip_dual_bound"]}
    row, integer = verify_snapshot_incumbent(snapshot, record, x)
    assert all(0 in knots for knots in snapshot["metadata"]["knots"])
    return {"status": "PASS", "optimizer_called": False, "saved_QP_records_replayed": 72,
            "MILP_snapshots_verified": 2, "max_primal_difference": max_primal,
            "max_tangent_difference": max_tangent, "max_model_row_violation": row,
            "max_model_integrality_violation": integer,
            "old_missing_incumbent_and_bound_preserved": True}


def _restore_context(saved):
    context = {
        "history": Observations(**saved["history"]),
        "model": MeasurementModel(**saved["measurement_model"]),
        "config": saved["config"], "channels": tuple(saved["channels"]),
        "scenarios": saved["scenarios"], "centres": {},
    }
    for scenario in context["scenarios"]:
        for key in ("p0", "q0", "root_v"):
            scenario[key] = np.asarray(scenario[key])
    for name, rows in saved["centres"].items():
        candidate = RadialCandidate.from_supports(name, 3, SUPPORTS[name])
        context["centres"][name] = [
            {"candidate": candidate, "beta": np.asarray(row["beta"]),
             "history_state": np.asarray(row["history_state"]),
             "fit": SimpleNamespace(**{key: np.asarray(value)
                                       for key, value in row["fit"].items()})}
            for row in rows
        ]
    return context


def replay_qp_record(pair, record):
    """Recompute feasible primal and box-tangent lower values from saved data."""
    control = np.asarray(record["u_kw"]).ravel()
    nuisance = np.asarray(record["z"])
    assert np.isfinite(control).all() and np.isfinite(nuisance).all()
    assert np.all(nuisance >= pair["lower"]) and np.all(nuisance <= pair["upper"])
    matrix = np.vstack([pair["A"], pair["F"]])
    constant = np.r_[pair["c"] + pair["B"] @ control, np.zeros(len(pair["F"]))]
    residual = constant + matrix @ nuisance
    primal = float(residual @ residual)
    gradient = 2 * matrix.T @ residual
    correction = np.minimum(gradient * (pair["lower"] - nuisance),
                            gradient * (pair["upper"] - nuisance))
    tangent = max(0, primal + float(correction.sum()) - record["roundoff_guard"])
    primal_difference = abs(primal - record["q_upper"])
    tangent_difference = abs(tangent - record["q_lower"])
    assert primal_difference < 1e-8 * max(1, abs(primal))
    assert tangent_difference < 1e-8 * max(1, abs(tangent))
    assert record["numeric_gap"] == record["q_upper"] - record["q_lower"]
    return primal_difference, tangent_difference


def verify_snapshot_incumbent(snapshot, record, x):
    """Check saved MILP primal rows, integrality, objective, and bound direction."""
    assert len(snapshot["lower"]) == record["n_variables"]
    assert len(snapshot["rows"]) == record["n_rows"]
    row_violation = 0.0
    integer_violation = 0.0
    if x is not None:
        violations = []
        for value, lower, upper in zip(x, snapshot["lower"], snapshot["upper"]):
            violations.extend([float(lower) - value, value - float(upper)])
        for row, lower, upper in zip(snapshot["rows"], snapshot["row_lower"], snapshot["row_upper"]):
            value = math.fsum(float(coefficient) * x[int(key)] for key, coefficient in row.items())
            violations.extend([float(lower) - value, value - float(upper)])
        row_violation = max(0, max(violations))
        integer_violation = max((abs(x[i] - round(x[i]))
                                 for i, integer in enumerate(snapshot["integer"]) if integer), default=0)
        assert row_violation < 2e-6 and integer_violation < 2e-6
        objective = math.fsum(float(coefficient) * x[int(key)]
                              for key, coefficient in snapshot["objective"].items())
        assert abs(objective - record["objective_value"]) < 1e-8
        bound = record.get("objective_bound")
        if bound is not None:
            assert bound >= record["objective_value"] - 1e-8
    else:
        assert record.get("objective_value") is None, "incumbent objective without saved x"
    for positive, negative in (("objective_value", "raw_minimization_fun"),
                               ("objective_bound", "raw_minimization_mip_dual_bound")):
        if record.get(positive) is not None and record.get(negative) is not None:
            assert abs(record[positive] + record[negative]) < 1e-8
    return row_violation, integer_violation


def verify_honly_saved(root, *, verify_sources=True, replay_ac=True):
    root = Path(root)
    for entry in _json(root / "evidence_manifest.json")["files"]:
        assert len(_original_bytes(root / entry["path"])) == entry["bytes"]
        assert _sha(root / entry["path"]) == entry["sha256"]
    before = _json(root / "code_and_input_hashes_before.json")
    after = _json(root / "code_hashes_after.json")
    assert before["frozen_core"] == after["frozen_core"]
    assert before["new_shared_module"] == after["new_shared_module"]
    assert after["original_global_upper_restored"] and after["original_MILP_solve_restored"]
    if verify_sources:
        for name, expected in before["frozen_core"].items():
            assert _sha((BASE / name).resolve()) == expected, name
        assert _sha(BASE / "shared_power_square_upper.py") == before["new_shared_module"]
    metrics = {"max_primal_difference": 0.0, "max_tangent_difference": 0.0,
               "max_model_row_violation": 0.0, "max_model_integrality_violation": 0.0,
               "max_actual_or_affine_AC_cost_difference": 0.0,
               "max_taylor_error_norm_difference": 0.0}
    qp_count = model_count = ac_count = exceptions = 0
    cases = []
    upper_without_bound = upper_with_bound = nonzero_optimal_gaps = 0
    for ident in ("710009", "720001"):
        case = root / f"H_{ident}"
        manifest = _json(case / "pair_manifest.json")
        assert manifest["pair_count"] == 24
        assert _sha(case / "frozen_pairs.npz") == manifest["archive_sha256"]
        with np.load(case / "frozen_pairs.npz", allow_pickle=False) as data:
            pairs = [{key: data[f"pair_{i}_{key}"].copy()
                      for key in ("c", "B", "A", "F", "lower", "upper", "scale")}
                     for i in range(24)]
        for name, pair in zip(manifest["names"], pairs):
            digest = hashlib.sha256(name.encode())
            for key in ("c", "B", "A", "F", "lower", "upper", "scale"):
                array = pair[key]
                digest.update(str(array.shape).encode())
                digest.update(np.ascontiguousarray(array).tobytes())
            assert digest.hexdigest() == manifest["fingerprints"][name]
            for row in pair["B"]:
                assert len(np.unique(np.flatnonzero(row != 0.0) // 3)) <= 1
        pair_map = dict(zip(manifest["names"], pairs))

        def replay_records(records):
            nonlocal qp_count, exceptions
            for record in records:
                if "exception" in record:
                    exceptions += 1
                    continue
                primal, tangent = replay_qp_record(pair_map[record["pair"]], record)
                metrics["max_primal_difference"] = max(metrics["max_primal_difference"], primal)
                metrics["max_tangent_difference"] = max(metrics["max_tangent_difference"], tangent)
                qp_count += 1

        baselines = _packed(case / "initial_action_QP.json.gz")
        for baseline in baselines.values():
            replay_records(baseline["records"])
        context = _restore_context(_packed(case / "AC_replay_context.json.gz")) if replay_ac else None
        for label in ("original", "shared"):
            target = case / label
            adapter = _json(target / "adapter_audit.json")
            assert adapter["same_code_object"] and adapter["keyword_defaults_copied"]
            assert adapter["original_global_upper_unmodified"]
            expected = {"solve_pair"} if label == "original" else {"solve_pair", "upper_master"}
            assert set(adapter["changed_globals"]) == expected
            replay_records(_packed(target / "qp_trace.json.gz"))
            point = _json(target / "selected_point_qp.json")
            replay_records(point["records"])
            result = _json(target / "search_result.json")
            audit = _json(target / "selected_audit.json")
            assert result["raw_global_gap"] == result["proxy_upper"] - result["proxy_lower"]
            assert result["relative_gap"] == result["absolute_gap"] / max(1, abs(result["proxy_lower"]))
            assert audit["absolute_stop_tolerance"] == 1e-4 + .001 * max(1, abs(result["proxy_lower"]))
            assert result["status"] == "proxy_gap_met"
            assert result["absolute_gap"] <= audit["absolute_stop_tolerance"]
            assert audit["raw_controls"]["clipping_or_rounding_performed"] is False
            for key, baseline in baselines.items():
                assert audit["gain_intervals_vs_initial_actions"][key] == [
                    point["q_lower"] - baseline["q_upper"], point["q_upper"] - baseline["q_lower"]]
            if label == "shared":
                gain = audit["gain_intervals_vs_initial_actions"]["constant"]
                assert gain[0] < 0 < gain[1]
            if replay_ac:
                saved_ac = _json(target / "full_AC_rollback.json")
                control = np.asarray(audit["raw_controls"]["u_kw"])
                for pair, metadata, qp, saved in zip(pairs, manifest["raw_pair_metadata"], point["records"], saved_ac):
                    replay = evaluate_frozen_pair_actual(context, {**pair, **metadata}, control, np.asarray(qp["z"]))
                    for key in ("actual_soft_cost", "affine_soft_cost"):
                        metrics["max_actual_or_affine_AC_cost_difference"] = max(
                            metrics["max_actual_or_affine_AC_cost_difference"], abs(replay[key] - saved[key]))
                    metrics["max_taylor_error_norm_difference"] = max(
                        metrics["max_taylor_error_norm_difference"],
                        abs(replay["joint_taylor_error_norm"] - saved["joint_taylor_error_norm"]))
                    assert bool(replay["physically_feasible"]) == bool(saved["physically_feasible"])
                    assert abs(replay["actual_soft_cost"] - sum(replay["actual_history_costs"])
                               - replay["actual_future_distance_squared"]) < 1e-8
                    assert saved["nonlinear_infimum_computed"] is False
                    assert saved["is_nonlinear_design_bound"] is False
                    ac_count += 1
            with np.load(target / "master_incumbents.npz", allow_pickle=False) as data:
                incumbents = {key: data[key].copy() for key in data.files}
            for record in _json(target / "solver_calls.json"):
                raw = gzip.decompress((target / record["snapshot_file"]).read_bytes())
                assert hashlib.sha256(raw).hexdigest() == record["snapshot_sha256"]
                snapshot = json.loads(raw)
                row, integer = verify_snapshot_incumbent(snapshot, record, incumbents.get(f"model_{record['call']}_x"))
                metrics["max_model_row_violation"] = max(metrics["max_model_row_violation"], row)
                metrics["max_model_integrality_violation"] = max(metrics["max_model_integrality_violation"], integer)
                if record["class"] == "upper_original":
                    assert record["status"] == 1
                    assert record["objective_value"] is None and record["objective_bound"] is None
                    upper_without_bound += 1
                if record["class"] == "upper_shared":
                    assert record["objective_value"] is not None and record["objective_bound"] is not None
                    upper_with_bound += 1
                    nonzero_optimal_gaps += int(record["status"] == 0 and record["mip_gap"] > 0)
                model_count += 1
            cases.append({"history": ident, "backend": label,
                          "proxy_lower": result["proxy_lower"], "proxy_upper": result["proxy_upper"],
                          "raw_global_gap": result["raw_global_gap"],
                          "gain_vs_constant": audit["gain_intervals_vs_initial_actions"]["constant"]})
    assert qp_count == 696 and model_count == 16
    assert ac_count == (96 if replay_ac else 0)
    assert upper_without_bound == upper_with_bound == 3
    assert nonzero_optimal_gaps == 1
    assert metrics["max_actual_or_affine_AC_cost_difference"] < 1e-8
    assert metrics["max_taylor_error_norm_difference"] < 1e-10
    return {"status": "PASS", "fit_optimizer_or_selector_called": False,
            "QP_records_replayed": qp_count, "MILP_snapshots_verified": model_count,
            "full_AC_saved_witness_records_replayed": ac_count,
            "source_hashes_checked": len(before["frozen_core"]) if verify_sources else 0,
            "QP_exceptions_preserved": exceptions,
            "original_upper_missing_incumbent_and_bound": upper_without_bound,
            "shared_upper_with_incumbent_and_bound": upper_with_bound,
            "optimal_status_with_nonzero_mip_gap_preserved": nonzero_optimal_gaps,
            **metrics, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_ARCHIVE)
    arguments = parser.parse_args()
    integrity = verify_archive_manifest(arguments.evidence_root)
    result = verify_honly_saved(arguments.evidence_root / "originals" / "Honly_search_v1")
    comparison = verify_single_upper_saved(arguments.evidence_root / "originals" / "H_710009_compare")
    print(json.dumps({**integrity, **result, "single_upper_comparison": comparison}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
