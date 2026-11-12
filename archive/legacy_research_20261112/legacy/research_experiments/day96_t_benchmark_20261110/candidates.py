"""Export finite, history-fitted own-search models for later frozen-parameter DR.

This module never fits weights, calls a solver, searches supports, reads truth,
or caps the pool. Output contraction is a separate topology map: the exported
R/X predictor always uses every original response coefficient.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math

import numpy as np


ZERO = 1e-9
ALLOWED_SOURCES = {"own_path", "own_search", "singleton", "wzzt_path", "joint_tail", "final_exit"}


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def digest(value):
    """Order-stable JSON hash, retaining the actual floating-point values."""
    payload = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _arrays(data, n):
    if isinstance(data, Mapping):
        values = [data[key] for key in ("p", "q", "y")]
    else:
        values = list(data)
        if len(values) != 3:
            raise ValueError("Scoring data must contain p, q and squared-voltage-drop y.")
    result = []
    for value in values:
        array = np.asarray(value, dtype=float)
        if array.ndim < 2 or array.shape[-1] != n or not array.size or not np.isfinite(array).all():
            raise ValueError("Scoring arrays must have finite (..., time, terminal) shape.")
        result.append(array.reshape(-1, n))
    if any(array.shape != result[0].shape for array in result):
        raise ValueError("p, q and y must align exactly.")
    return tuple(result)


def _array_digest(arrays):
    value = hashlib.sha256()
    for name, array in zip(("p", "q", "y"), arrays):
        array = np.ascontiguousarray(array, dtype="<f8")
        value.update(name.encode("ascii"))
        value.update(str(array.shape).encode("ascii"))
        value.update(array.tobytes())
    return value.hexdigest()


def response_matrices(point, n):
    z = np.asarray([[i in support for i in range(n)] for support in point["supports"]], dtype=float)
    return ((z.T * np.asarray(point["r"], dtype=float)) @ z,
            (z.T * np.asarray(point["x"], dtype=float)) @ z)


def loss(point, data, n):
    p, q, y = _arrays(data, n)
    R, X = response_matrices(point, n)
    return float(np.mean(np.abs(y - p @ R.T - q @ X.T)))


def _point(raw, n):
    supports = [tuple(int(i) for i in support) for support in raw["supports"]]
    if not supports or len(supports) != len(set(supports)):
        raise ValueError("Every fitted family must contain unique nonempty supports.")
    if any(not C or C != tuple(sorted(set(C))) or min(C) < 0 or max(C) >= n for C in supports):
        raise ValueError("Support indices must be sorted, unique and inside the terminal order.")
    sets = list(map(set, supports))
    if any(C & D and not (C <= D or D <= C) for i, C in enumerate(sets) for D in sets[i+1:]):
        raise ValueError("Each exported complete model must be laminar.")
    r, x = np.asarray(raw["r"], float), np.asarray(raw["x"], float)
    if r.shape != (len(supports),) or x.shape != r.shape or not np.isfinite(np.r_[r, x]).all():
        raise ValueError("Every support must have finite fitted r and x coefficients.")
    if np.min(np.r_[r, x]) < -ZERO or np.max(np.r_[r, x]) > 2 + ZERO:
        raise ValueError("This benchmark exports the declared [0,2] response bounds only.")
    indices = sorted(range(len(supports)), key=lambda i: (len(supports[i]), supports[i]))
    return dict(supports=[list(supports[i]) for i in indices],
                r=[float(r[i]) for i in indices], x=[float(x[i]) for i in indices])


def _choose(rows, *, fraction, tie_key, tolerance):
    best = min(row["validation_mae"] for row in rows)
    eligible = [row for row in rows if row["validation_mae"] <= (1 + fraction)*best + tolerance]
    chosen = min(eligible, key=lambda row: (row[tie_key], row["validation_mae"], row["index"]))
    return dict(candidate_id=chosen["candidate_id"], index=chosen["index"],
                validation_mae=chosen["validation_mae"], best_validation_mae=best,
                relative_band=fraction, absolute_tolerance=tolerance, tie_count=tie_key,
                eligible_candidate_ids=[row["candidate_id"] for row in eligible],
                parameter_update=False, contraction_used_in_selection=False)


def export_candidates(points, *, labels, train, validation, provenance,
                      calibration=None, root_label=0, base_kw=None,
                      contraction_resolution=.16, band_fraction=.03, fit_loss_tolerance=1e-9):
    """Serialize raw own-path points and explicitly separate output topology.

    ``points`` accepts raw fitted dictionaries or wrappers containing ``point``;
    wrappers carry source/stage/path_index metadata. Required provenance asserts
    weights_fit_split='train', truth_used_in_fitting=False,
    test_used_in_generation=False, future_DR_used=False, and SHA256 input_hashes
    for train/validation. These are caller assertions, independently supplemented
    by array hashes and recomputed losses, not proof of estimator provenance.

    Same families with different coefficients remain distinct. Only identical
    canonical support/coefficient dictionaries merge, with all origins retained.
    No candidate is dropped by a relative validation cutoff in this export.
    """
    labels = [int(label) for label in labels]
    n = len(labels)
    if not n or len(set(labels)) != n:
        raise ValueError("An explicit unique terminal label order is required.")
    provenance = _json_value(dict(provenance))
    if provenance.get("weights_fit_split") != "train":
        raise ValueError("Only history-training-fitted coefficients may be exported.")
    for key in ("truth_used_in_fitting", "test_used_in_generation", "future_DR_used"):
        if provenance.get(key) is not False:
            raise ValueError(f"Explicit data-only provenance assertion required: {key}=False.")
    hashes = provenance.get("input_hashes", {})
    for key in ("train", "validation"):
        value = hashes.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
            raise ValueError(f"A SHA256 source hash is required for {key}.")
    for value in (contraction_resolution, band_fraction, fit_loss_tolerance):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Resolutions, bands and tolerances must be finite and nonnegative.")
    if base_kw is not None and (not math.isfinite(float(base_kw)) or float(base_kw) <= 0):
        raise ValueError("The declared power base must be positive.")
    training, validating = _arrays(train, n), _arrays(validation, n)
    if calibration is not None:
        calibration = {key: float(calibration[key]) for key in ("a", "b", "median_root_depth")}
        if any(not math.isfinite(v) or v <= 0 for v in calibration.values()):
            raise ValueError("Output contraction requires positive frozen calibration values.")

    rows, seen = [], {}
    for path_index, wrapper in enumerate(points):
        raw = wrapper.get("point", wrapper)
        source = wrapper.get("source", "own_path")
        if source not in ALLOWED_SOURCES:
            raise ValueError(f"External or complete-RNJ fallback source is excluded: {source!r}.")
        origin = dict(source=source, stage=wrapper.get("stage", "own_path"),
                      path_index=wrapper.get("path_index", path_index))
        if origin["stage"] in {"complete_rnj", "raw_rnj", "rnj_backup", "rnj_fallback", "rnj_bank"}:
            raise ValueError("Complete RNJ fallback stages are excluded from this own-search pool.")
        for key in ("parent_candidate", "iteration", "event_index"):
            if key in wrapper:
                origin[key] = _json_value(wrapper[key])
        point = _point(raw, n)
        train_mae, validation_mae = loss(point, training, n), loss(point, validating, n)
        if "objective" in raw and (not math.isfinite(float(raw["objective"])) or
                                   abs(float(raw["objective"]) - train_mae) > fit_loss_tolerance):
            raise ValueError("Saved fitted training loss does not match the supplied training observations.")
        signature = digest(point)
        if signature in seen:
            rows[seen[signature]]["origins"].append(origin)
            continue
        R, X = response_matrices(point, n)
        positive = [C for C, r, x in zip(point["supports"], point["r"], point["x"])
                    if 1 < len(C) < n and max(r, x) > ZERO]
        effective_count = sum(len(C) == 1 or max(r, x) > ZERO
                              for C, r, x in zip(point["supports"], point["r"], point["x"]))
        output = None if calibration is None else [C for C, r, x in zip(point["supports"], point["r"], point["x"])
                    if 1 < len(C) < n and max(r, x) > ZERO and
                    calibration["a"]*r + calibration["b"]*x + 1e-10 >= contraction_resolution*calibration["median_root_depth"]]
        point["objective"] = train_mae
        point["validation_mae"] = validation_mae
        row = dict(index=len(rows), candidate_id=signature, family_sha256=digest(point["supports"]),
                   parameter_sha256=signature, prediction_sha256=digest(dict(R=R.tolist(), X=X.tolist())),
                   point=point, R=R.tolist(), X=X.tolist(), labels=labels,
                   train_mae=train_mae, validation_mae=validation_mae,
                   support_count=len(point["supports"]), effective_support_count=effective_count,
                   proper_internal_count=sum(1 < len(C) < n for C in point["supports"]),
                   positive_internal_count=len(positive), positive_topology=positive,
                   output_topology=output, origins=[origin], frozen_weights=True,
                   fit_loss_recomputed=True, output_contraction_refit=False)
        seen[signature] = len(rows)
        rows.append(row)
    if not rows:
        raise ValueError("No fitted own-search candidate was supplied.")
    policy = dict(resolution=contraction_resolution, calibration=calibration,
                  refit=False, predictor_changed=False,
                  meaning="Topology-only map; frozen raw R/X continue to define predictions.")
    payload = dict(schema_version=1, scope="finite_own_search_complete_fitted_models",
                complete_topology_or_parameter_coverage_claimed=False,
                labels=labels, root_label=int(root_label), base_kw=None if base_kw is None else float(base_kw),
                units=dict(P="p.u.", Q="p.u.", Y="p.u.^2", R_X="squared-voltage response coefficients; not ohms"),
                target_convention="Y=V_root^2-V_terminal^2", prediction_convention="Y=P@R.T+Q@X.T",
                frozen_before_DR=True, weights_frozen=True, future_refit_allowed=False,
                complete_RNJ_output_fallback=False, pool_capped=False,
                dispatch_base_available=base_kw is not None, provenance=provenance,
                provenance_is_caller_assertion=True,
                data_array_sha256=dict(train=_array_digest(training), validation=_array_digest(validating)),
                sample_counts=dict(train=training[0].shape[0], validation=validating[0].shape[0]),
                output_mapping=policy, candidates=rows,
                selections=dict(strict=_choose(rows, fraction=0., tie_key="positive_internal_count", tolerance=1e-12),
                                band3_proper_internal=_choose(rows, fraction=band_fraction, tie_key="positive_internal_count", tolerance=1e-12),
                                neighbor_style_band3=_choose(rows, fraction=band_fraction, tie_key="support_count", tolerance=1e-12)),
                candidate_set_sha256=digest([row["candidate_id"] for row in rows]))
    payload["pool_sha256"] = digest(payload)
    return payload


def verify_export(export, *, expected_sha256=None, train=None, validation=None):
    """Check the full frozen payload and optionally replay historical losses.

    An expected externally frozen digest is needed to detect deliberate digest
    replacement; the embedded digest alone detects accidental mutation.
    """
    stored = export.get("pool_sha256")
    measured = digest({key: value for key, value in export.items() if key != "pool_sha256"})
    if not stored or stored != measured or (expected_sha256 is not None and measured != expected_sha256):
        raise ValueError("Frozen candidate export hash mismatch.")
    if not export.get("weights_frozen") or not export.get("frozen_before_DR") or export.get("future_refit_allowed") is not False:
        raise ValueError("Frozen-parameter candidate contract was changed.")
    if export.get("complete_RNJ_output_fallback") is not False:
        raise ValueError("A complete-RNJ output fallback is outside this export contract.")
    labels = export["labels"]
    n = len(labels)
    rows = export["candidates"]
    if digest([row["candidate_id"] for row in rows]) != export["candidate_set_sha256"]:
        raise ValueError("Candidate-set hash mismatch.")
    supplied = {name: _arrays(value, n) for name, value in (("train", train), ("validation", validation)) if value is not None}
    for name, arrays in supplied.items():
        if _array_digest(arrays) != export["data_array_sha256"][name]:
            raise ValueError(f"Historical {name} arrays differ from the frozen export.")
    for row in rows:
        point = _point(row["point"], n)
        signature = digest(point)
        if signature != row["candidate_id"] or signature != row["parameter_sha256"]:
            raise ValueError("A candidate's frozen response coefficients changed.")
        if row["labels"] != labels or row["family_sha256"] != digest(point["supports"]):
            raise ValueError("Terminal order or candidate family hash mismatch.")
        R, X = response_matrices(point, n)
        if np.asarray(row["R"]).shape != (n, n) or np.asarray(row["X"]).shape != (n, n):
            raise ValueError("Frozen response matrices must align with the terminal order.")
        if not np.allclose(R, row["R"], rtol=0., atol=1e-12) or not np.allclose(X, row["X"], rtol=0., atol=1e-12):
            raise ValueError("Frozen R/X does not reproduce the support/weight response.")
        if row["prediction_sha256"] != digest(dict(R=row["R"], X=row["X"])):
            raise ValueError("Frozen prediction-matrix hash mismatch.")
        for name, arrays in supplied.items():
            if abs(loss(point, arrays, n) - row[name + "_mae"]) > 1e-10:
                raise ValueError(f"A candidate's {name} loss does not reproduce.")
    return dict(passed=True, candidates_verified=len(rows),
                complete_payload_sha256=measured, response_replayed=True,
                historical_losses_replayed=sorted(supplied), new_fit_calls=0)


def coverage_diagnostics(export, true_topology):
    """Post-analysis only; never use truth to filter or choose exported models."""
    truth = {tuple(sorted(int(i) for i in C)) for C in true_topology}
    predicted = [{tuple(C) for C in row["positive_topology"]} for row in export["candidates"]]
    output = [{tuple(C) for C in row["output_topology"]} for row in export["candidates"]
              if row["output_topology"] is not None]
    union = set().union(*predicted)
    return dict(post_analysis_only=True, candidates=len(predicted),
                any_raw_model_exact=any(C == truth for C in predicted),
                any_output_model_exact=any(C == truth for C in output) if output else None,
                true_connection_union_recall=len(union & truth)/len(truth) if truth else 1.,
                all_true_connections_in_union=truth <= union,
                candidate_positive_counts=[len(C) for C in predicted])
