"""Read-only comparison of two distinct saved, training-fitted topologies.

No refitting, candidate generation, statistical confidence, or file access.
Time blocks must be supplied explicitly; a row count does not identify days.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

if __package__:
    from .optimization import predict
    from .trees import is_laminar
else:
    from optimization import predict
    from trees import is_laminar


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def competition(result, validation, *, blocks=None):
    """Return actual primary and best distinct alternative, with frozen weights.

    Models are referenced by index in result['frozen_candidates']. Numerical
    zero auxiliary clades do not create a second topology. Their original
    coefficients and complete point snapshots remain unchanged in that bank.
    """
    bank = result["frozen_candidates"]
    reference = float(result["reference_mse"])
    config = result["config"]
    tie, zero = float(config["validation_tie"]), float(config["zero_tol"])
    labels, n = tuple(validation.labels), validation.n
    if not bank or not np.isfinite(reference) or reference <= 0:
        raise ValueError("A nonempty frozen library and positive reference are required")
    stages = ["baseline" if result["method"] in {"rnj", "nj"} else "initial"]
    stages += [e["stage"] for e in result.get("events", []) if e.get("accepted")]
    rows, losses = [], {}
    for index, candidate in enumerate(bank):
        if candidate["index"] != index or tuple(candidate["labels"]) != labels:
            raise ValueError("Frozen candidate indices or terminal labels changed")
        point = candidate["point"]
        family = [tuple(C) for C in point["supports"]]
        r, x = np.asarray(point["r"], float), np.asarray(point["x"], float)
        if (len(family) != len(set(family)) or not is_laminar(family)
                or r.shape != (len(family),) or x.shape != r.shape
                or not np.isfinite(r).all() or not np.isfinite(x).all()
                or np.any(r < 0) or np.any(x < 0)
                or any(not C or min(C) < 0 or max(C) >= n for C in family)):
            raise ValueError("Invalid frozen model")
        active = tuple(sorted(C for C, rr, xx in zip(family, r, x, strict=True)
                              if 1 < len(C) < n and max(rr, xx) > zero))
        clades = sorted([sorted(labels[i] for i in C) for C in active])
        residual = validation.y-predict(validation, point)
        losses[index] = np.mean(residual**2, axis=1)
        mse = float(np.mean(losses[index])); scaled = mse/reference
        if (not np.isfinite(scaled) or not np.isclose(scaled, candidate["validation_scaled_mse"],
                                                     rtol=1e-10, atol=1e-12)):
            raise ValueError("Candidate validation score is stale or uses different data")
        identity = _hash(dict(labels=sorted(labels), clades=clades))
        snapshot = dict(supports=[list(C) for C in family], r=r.tolist(), x=x.tolist())
        stage = stages[index] if index < len(stages) else "singleton_protection"
        rows.append(dict(index=index, topology_id=identity, active_clades=clades,
                         active_internal=len(active), stored_internal=sum(1 < len(C) < n for C in family),
                         validation_mse=mse, validation_scaled_mse=scaled, source_stage=stage,
                         coefficient_sha256=_hash(snapshot)))

    def choose(candidates):
        best = min(c["validation_scaled_mse"] for c in candidates)
        return min((c for c in candidates if c["validation_scaled_mse"] <= best+tie),
                   key=lambda c: (c["stored_internal"], c["validation_scaled_mse"], c["index"]))

    primary = rows[int(result["selected"]["index"])]
    selected_point = result["selected"]["point"]
    selected_snapshot = dict(supports=[list(C) for C in selected_point["supports"]],
                             r=list(map(float, selected_point["r"])), x=list(map(float, selected_point["x"])))
    if _hash(selected_snapshot) != primary["coefficient_sha256"]:
        raise ValueError("The actual selected point differs from its frozen candidate")
    if choose(rows)["index"] != primary["index"]:
        raise ValueError("The actual primary does not follow the declared validation-MSE rule")
    alternatives = [c for c in rows if c["topology_id"] != primary["topology_id"]]
    secondary = choose(alternatives) if alternatives else None
    compared = [primary] + ([secondary] if secondary else [])
    block_rows = []
    if blocks is not None:
        occupied = np.zeros(validation.points, int)
        for block in blocks:
            start, stop = int(block["start"]), int(block["stop"])
            if start < 0 or stop <= start or stop > validation.points:
                raise ValueError("Validation blocks must be nonempty in-range intervals")
            occupied[start:stop] += 1
            first = float(np.mean(losses[primary["index"]][start:stop]))
            second = float(np.mean(losses[secondary["index"]][start:stop])) if secondary else None
            block_rows.append(dict(name=str(block["name"]), start=start, stop=stop, points=stop-start,
                                   mse_primary=first, mse_secondary=second,
                                   delta_mse=second-first if secondary else None))
        if not np.all(occupied == 1):
            raise ValueError("Explicit blocks must partition validation exactly once")
    if secondary is None:
        stability = "no_distinct_saved_alternative"
    elif len(block_rows) < 2:
        stability = "insufficient_blocks_for_stability_assessment"
    else:
        differences = [b["delta_mse"]/reference for b in block_rows]
        stability = ("primary_wins_all_observed_blocks" if all(d > tie for d in differences)
                     else "observed_blocks_not_strictly_separated")
    first_set = set(map(tuple, primary["active_clades"]))
    second_set = set(map(tuple, secondary["active_clades"])) if secondary else set()
    crossings = [(list(C), list(D)) for C in first_set for D in second_set
                 if set(C) & set(D) and set(C)-set(D) and set(D)-set(C)] if secondary else []
    return dict(primary_index=primary["index"], secondary_index=secondary["index"] if secondary else None,
                distinct_topologies=len({c["topology_id"] for c in rows}), models=compared,
                scope="saved_training_fitted_candidates_only", library_scan_complete=True,
                topology_identity="proper_clades_with_response_weight_above_numerical_zero_tolerance",
                zero_tolerance=zero, validation_points=validation.points,
                validation_mse_gap=secondary["validation_mse"]-primary["validation_mse"] if secondary else None,
                normalized_mse_gap=secondary["validation_scaled_mse"]-primary["validation_scaled_mse"] if secondary else None,
                only_primary=[list(C) for C in sorted(first_set-second_set)] if secondary else [],
                only_secondary=[list(C) for C in sorted(second_set-first_set)] if secondary else [],
                crossing_pairs=sorted(crossings), blocks=block_rows, observed_stability=stability,
                statistical_ambiguity="not_calibrated", global_identification="not_established",
                search_stop_reason=result.get("stop_reason"), refitted_on_validation=False)
