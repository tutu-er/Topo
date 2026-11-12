"""Frozen native candidate bank and the unchanged final MSE selector."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json

import numpy as np

from .identification import algorithm
from .identification.competition import competition
from .identification.optimization import response_matrices
from .dr.selection_mse import choose_mse, required_new_sse_gap
from .observations import fingerprint
from .observations import Observation


def fit_observed(training, validation, *, method="111", config=None):
    """Use the existing identifier; retain every native own-path candidate."""
    result = algorithm.fit(training, validation, method=method, config=config)
    result["handoff"] = dict(schema="rnj_wzzt_dr_v1",
        training_sha256=fingerprint(training), validation_sha256=fingerprint(validation),
        parameter_rule="native accepted path and singleton protection, no contraction",
        selection_rule="validation_MSE/reference_MSE, original numerical ties")
    return result


@dataclass(frozen=True)
class FrozenHistory:
    labels: tuple
    H: np.ndarray
    sse: np.ndarray
    cells: int
    internal: np.ndarray
    topology_ids: tuple
    models: tuple
    reference_mse: float
    epsilon_scaled: float
    current: int
    parameter_sha256: str
    observed: object

    @property
    def K(self):
        return 1. / self.reference_mse

    def select(self, new):
        """Equal weight for every old validation and new-day residual cell."""
        if tuple(new.labels) != self.labels:
            raise ValueError("New-day terminal order differs from frozen history")
        power = np.column_stack((new.p, new.q))
        sse = self.sse + np.array([np.square(new.y-power@H.T).sum() for H in self.H])
        decision = choose_mse(sse/(self.cells+new.y.size), self.internal, K=self.K,
                              epsilon_scaled=self.epsilon_scaled)
        return dict(decision, total_cells=self.cells+new.y.size, refitted=False,
                    topology_id=self.topology_ids[decision["selected"]])

    def required_gap(self, target, competitor, new_cells):
        return required_new_sse_gap(self.sse[competitor], self.sse[target],
            self.cells, new_cells, K=self.K, epsilon_scaled=self.epsilon_scaled)


def freeze_history(result, validation):
    """Verify native selection, coefficient matrices and history before DR."""
    stamp = result.get("handoff", {}).get("validation_sha256")
    if stamp is not None and stamp != fingerprint(validation):
        raise ValueError("Historical observations changed after fitting")
    competition(result, validation)  # Original full-bank validation and tie rule.
    bank = result["frozen_candidates"]
    matrices, models, topologies, counts, snapshots = [], [], [], [], []
    zero = result["config"]["zero_tol"]
    power = np.column_stack((validation.p, validation.q))
    for i, row in enumerate(bank):
        R, X = response_matrices(row["point"], validation.n)
        if not all(np.allclose(row[name], a, rtol=1e-12, atol=1e-12)
                   for name, a in (("R", R), ("X", X))):
            raise ValueError("Stored R/X differs from frozen support weights")
        H = np.column_stack((R, X))
        point = {k: deepcopy(row["point"][k]) for k in ("supports", "r", "x")}
        snapshots.append(point)
        active = sorted(tuple(C) for C, r, x in zip(point["supports"], point["r"], point["x"], strict=True)
                        if 1 < len(C) < validation.n and max(r, x) > zero)
        identity = json.dumps(active, separators=(",", ":"))
        internal = sum(1 < len(C) < validation.n for C in point["supports"])
        mse = float(np.square(validation.y-power@H.T).mean())
        models.append(dict(id=i, point=point, H=H, topology_id=identity,
                           internal=internal, history_mse=mse))
        matrices.append(H); topologies.append(identity); counts.append(internal)
    H = np.stack(matrices)
    sse = np.array([m["history_mse"]*validation.y.size for m in models])
    internal = np.array(counts)
    reference = float(result["reference_mse"])
    epsilon = float(result["config"]["validation_tie"])
    selected = choose_mse(sse/validation.y.size, internal, K=1/reference,
                          epsilon_scaled=epsilon)["selected"]
    if selected != result["selected"]["index"]:
        raise ValueError("DR historical winner differs from native final selection")
    for a in (H, sse, internal):
        a.setflags(write=False)
    digest = hashlib.sha256(json.dumps(snapshots, sort_keys=True, allow_nan=False).encode()).hexdigest()
    observed = Observation(validation.p.copy(), validation.q.copy(), validation.y.copy(), validation.labels)
    for a in (observed.p, observed.q, observed.y):
        a.setflags(write=False)
    return FrozenHistory(tuple(validation.labels), H, sse, validation.y.size, internal,
                         tuple(topologies), tuple(models), reference, epsilon, selected,
                         digest, observed)
