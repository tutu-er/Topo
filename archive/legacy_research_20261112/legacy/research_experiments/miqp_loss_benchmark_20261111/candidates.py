"""MAE-compatible frozen exports of models fitted with scaled squared loss.

No fits or support search occur here. Only export copies receive an MAE
``objective``; the native points and their QP coefficients remain unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from research_experiments.day96_t_benchmark_20261110 import candidates as base

digest = base.digest
response_matrices = base.response_matrices
coverage_diagnostics = base.coverage_diagnostics


def physical_scores(point, data, n):
    p, q, y = base._arrays(data, n)
    R, X = response_matrices(point, n)
    residual = y - p @ R.T - q @ X.T
    return dict(mae=float(np.mean(np.abs(residual))),
                mse=float(np.mean(residual**2)), sse=float(np.sum(residual**2)),
                observations=int(residual.size))


def _scale(normalization):
    K = float(normalization["K"])
    if not math.isfinite(K) or K <= 0:
        raise ValueError("A positive frozen squared-loss scale is required.")
    return K


def _anchor(row, n):
    point = row.get("point", row)
    return digest(base._point(point, n))


def export_candidates(points, *, selected, selected_mse_view, normalization,
                      labels, train, validation, **kwargs):
    """Include raw path, fitted output library and both historical selectors.

    Callers provide all output-library rows among ``points``. The two selected
    fitted models are appended again as anchors; exact duplicates merge while
    retaining their origins. No complete-RNJ fallback is admitted.
    """
    n, K = len(labels), _scale(normalization)
    wrappers = [deepcopy(p if "point" in p else dict(point=p)) for p in points]
    for name, row in (("native_selected", selected), ("native_mse_view", selected_mse_view)):
        wrappers.append(dict(point=deepcopy(row["point"]), source="own_search",
                             stage=name, path_index=int(row["index"])))
    native = {}
    for wrapper in wrappers:
        point = wrapper["point"]
        score = physical_scores(point, train, n)
        saved = float(point["objective"])
        expected = K * score["mse"]
        if not math.isfinite(saved) or abs(saved-expected) > 1e-9+1e-7*abs(expected):
            raise ValueError("A native QP objective does not reproduce K times training MSE.")
        key = _anchor(point, n)
        native.setdefault(key, []).append(saved)
        # The old frozen-parameter contract defines objective as observed MAE.
        # This detached copy is the only place that definition is changed.
        point["objective"] = score["mae"]
    exported = base.export_candidates(wrappers, labels=labels, train=train,
                                      validation=validation, **kwargs)
    for row in exported["candidates"]:
        training = physical_scores(row["point"], train, n)
        validating = physical_scores(row["point"], validation, n)
        row.update(fit_loss="scaled_mse", score_loss="mae", loss_scale=K,
                   train_mse=training["mse"], train_sse=training["sse"],
                   validation_mse=validating["mse"], validation_sse=validating["sse"],
                   native_internal_objective=K*training["mse"],
                   native_saved_objectives=native[row["candidate_id"]],
                   historical_score_recomputed=True)
        row["point"].update(train_mae=training["mae"], train_mse=training["mse"],
                            train_sse=training["sse"], fit_loss="scaled_mse",
                            native_internal_objective=K*training["mse"], loss_scale=K)
    anchors = {name: dict(candidate_id=_anchor(row, n),
                          output_library_index=int(row["index"]),
                          validation_mae=float(row.get("validation_mae", row["validation"])),
                          validation_mse=float(row["validation_mse"]))
               for name, row in (("native_selected", selected), ("native_mse_view", selected_mse_view))}
    available = {row["candidate_id"] for row in exported["candidates"]}
    if any(row["candidate_id"] not in available for row in anchors.values()):
        raise AssertionError("The actual selected fitted point is absent from the frozen pool.")
    exported.update(extension_schema="miqp_squared_fit_mae_score_v1", fit_loss="scaled_mse",
                    historical_selection_score="mae", normalization=deepcopy(normalization),
                    native_anchors=anchors, output_library_included=True,
                    portable_pool_selectors_equal_native_output_selector=False,
                    export_score_is_not_fit_objective=True,
                    primary_selector="native strict MAE on .24-contracted, QP-refitted output library",
                    secondary_selector="strict K*MSE on the identical fitted output library; no new fit")
    exported["pool_sha256"] = digest({k: v for k, v in exported.items() if k != "pool_sha256"})
    return exported


def verify_export(export, *, expected_sha256=None, train=None, validation=None):
    receipt = base.verify_export(export, expected_sha256=expected_sha256,
                                 train=train, validation=validation)
    if export.get("extension_schema") != "miqp_squared_fit_mae_score_v1":
        raise ValueError("Unsupported squared-fit export extension.")
    K, n = _scale(export["normalization"]), len(export["labels"])
    for row in export["candidates"]:
        if row.get("fit_loss") != "scaled_mse" or row.get("score_loss") != "mae":
            raise ValueError("Training-loss and historical-score meanings were conflated.")
        if abs(row["point"]["objective"]-row["train_mae"]) > 1e-12:
            raise ValueError("The portable objective must retain its actual MAE meaning.")
        if abs(row["native_internal_objective"]-K*row["train_mse"]) > 1e-12:
            raise ValueError("Scaled squared-loss metadata changed.")
        for split, arrays in (("train", train), ("validation", validation)):
            if arrays is None:
                continue
            scores = physical_scores(row["point"], arrays, n)
            for metric in ("mse", "sse"):
                if abs(scores[metric]-row[f"{split}_{metric}"]) > 1e-12+1e-10*abs(scores[metric]):
                    raise ValueError(f"The saved {split} squared loss does not reproduce.")
    ids = {row["candidate_id"] for row in export["candidates"]}
    if any(a["candidate_id"] not in ids for a in export["native_anchors"].values()):
        raise ValueError("A native selected anchor is absent.")
    return dict(**receipt, squared_losses_replayed=sorted(
        name for name, value in (("train", train), ("validation", validation)) if value is not None),
        selected_anchors_verified=True, fit_and_score_losses_separate=True)
