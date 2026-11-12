"""The agreed plug-in formula; no fitting, truth, file access, or random draws."""
from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.special import logsumexp

from ...history import freeze_history


def weights(mse, cells):
    """Uniform prior, sigma²=min(MSE), s=2*sigma²/N; retain log probabilities."""
    mse = np.asarray(mse, dtype=float)
    if (mse.ndim != 1 or not mse.size or not np.isfinite(mse).all()
            or np.any(mse < 0) or isinstance(cells, bool) or int(cells) != cells or cells <= 0):
        raise ValueError("Finite nonnegative losses and a positive scalar residual count required")
    best = float(mse.min())
    if best == 0:
        raise ValueError("zero_variance: no artificial error floor is supplied")
    log_likelihood = -.5*int(cells)*((mse-best)/best)
    log_probability = log_likelihood-logsumexp(log_likelihood)
    return dict(sigma2=best, s=2*best/int(cells), cells=int(cells),
                probability=np.exp(log_probability), log_probability=log_probability,
                log_likelihood_shifted=log_likelihood)


def active_topology(point, labels, zero):
    return tuple(sorted(tuple(sorted(labels[i] for i in C))
                        for C, r, x in zip(point["supports"], point["r"], point["x"], strict=True)
                        if 1 < len(C) < len(labels) and max(r, x) > zero))


def topology_id(clades):
    return hashlib.sha256(json.dumps(clades, separators=(",", ":")).encode()).hexdigest()


def score(result, training, validation):
    """One training-selected frozen point per active topology, then uniform prior.

    All original parameter points remain in the output audit. Representative
    selection uses training MSE only, before validation losses are consumed.
    """
    if tuple(training.labels) != tuple(validation.labels):
        raise ValueError("Training and validation labels differ")
    history = freeze_history(result, validation)
    bank = result["frozen_candidates"]
    zero = float(result["config"]["zero_tol"])
    groups, point_rows = {}, []
    power = np.column_stack((training.p, training.q))
    for index, (candidate, H) in enumerate(zip(bank, history.H, strict=True)):
        clades = active_topology(candidate["point"], training.labels, zero)
        identity = topology_id(clades)
        train_mse = float(np.square(training.y-power@H.T).mean())
        point_rows.append(dict(index=index, topology_id=identity, active_clades=clades,
                               train_mse=train_mse, validation_mse=float(history.sse[index]/history.cells)))
        groups.setdefault(identity, []).append(index)
    representatives = [min(indices, key=lambda i: (point_rows[i]["train_mse"], i))
                       for indices in groups.values()]
    representatives.sort()
    posterior = weights([point_rows[i]["validation_mse"] for i in representatives], history.cells)
    rows = []
    for ordinal, index in enumerate(representatives):
        row = dict(point_rows[index], member_indices=groups[point_rows[index]["topology_id"]],
                   probability=float(posterior["probability"][ordinal]),
                   log_probability=float(posterior["log_probability"][ordinal]))
        rows.append(row)
    ranking = sorted(range(len(rows)), key=lambda i: (-rows[i]["probability"],
                    rows[i]["validation_mse"], rows[i]["index"]))
    first = rows[ranking[0]]
    second = rows[ranking[1]] if len(ranking) > 1 else None
    minimum = min(row["validation_mse"] for row in rows)
    if first["validation_mse"] != minimum:
        raise AssertionError("MAP must minimize MSE for uniform priors and common positive scale")
    native = point_rows[int(result["selected"]["index"])]
    if second:
        threshold = max(result["reference_mse"]*result["config"]["validation_tie"],
                        .003*second["validation_mse"])
        gap = second["validation_mse"]-first["validation_mse"]
        ambiguous = bool(gap <= threshold)
        pair_log_odds = history.cells*gap/(2*posterior["sigma2"])
        pair_probability = float(1/(1+np.exp(-pair_log_odds)))
    else:
        ambiguous, pair_probability, gap = False, None, None
    return dict(status="complete", cells=history.cells, training_points=training.points,
        validation_points=validation.points, labels=list(training.labels), raw_points=len(bank),
        distinct_topologies=len(rows), uniform_prior_over="distinct active topologies",
        representative_rule="minimum training MSE then original index; no refit",
        sigma2=posterior["sigma2"], s=posterior["s"], models=rows, all_frozen_points=point_rows,
        map_topology_id=first["topology_id"], map_index=first["index"], p_map=first["probability"],
        native_topology_id=native["topology_id"], native_index=native["index"],
        map_matches_native_topology=first["topology_id"] == native["topology_id"],
        near_pair_003=ambiguous, pair_probability=pair_probability, first_second_gap=gap,
        max_probability_sum_error=abs(sum(row["probability"] for row in rows)-1.),
        parameter_sha256=history.parameter_sha256, refitted=False)


def independent_check(result, training, validation, scored):
    """Direct support-by-support prediction and unshifted Gaussian likelihood."""
    mse = {}
    train_mse = {}
    for i, row in enumerate(result["frozen_candidates"]):
        for name, obs, dest in (("train", training, train_mse), ("validation", validation, mse)):
            predicted = np.zeros_like(obs.y)
            for C, r, x in zip(row["point"]["supports"], row["point"]["r"], row["point"]["x"], strict=True):
                term = r*obs.p[:, C].sum(axis=1)+x*obs.q[:, C].sum(axis=1)
                predicted[:, C] += term[:, None]
            residual = obs.y-predicted
            dest[i] = float(np.sum(residual.astype(np.longdouble)**2)/residual.size)
    indices = [m["index"] for m in scored["models"]]
    for model in scored["models"]:
        expected = min(model["member_indices"], key=lambda i: (train_mse[i], i))
        # A fit-level numerical tie may have indistinguishable training loss;
        # check optimality up to floating arithmetic rather than changing it.
        if train_mse[model["index"]] > train_mse[expected]+1e-10*max(train_mse[expected], 1e-30):
            raise AssertionError("Representative did not follow training-only choice")
    loss = np.array([mse[i] for i in indices], dtype=np.longdouble)
    variance = min(loss)
    ll = -np.longdouble(validation.y.size)*loss/(2*variance)
    ll -= max(ll)
    expected = np.exp(ll); expected /= sum(expected)
    actual = np.array([m["probability"] for m in scored["models"]], dtype=np.longdouble)
    error = float(max(abs(actual-expected)))
    mse_error = max(abs(mse[p["index"]]-p["validation_mse"]) for p in scored["all_frozen_points"])
    if error > 1e-8 or mse_error > 1e-10*max(mse.values()):
        raise AssertionError(("independent prediction/posterior mismatch", error, mse_error))
    return dict(status="passed", maximum_probability_error=error,
                maximum_physical_mse_error=mse_error, candidates_checked=len(mse))
