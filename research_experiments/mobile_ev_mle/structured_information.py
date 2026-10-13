"""Local information prices for a few uncertain internal branches.

No competing nonlinear fit is performed during curve search.  Historical and
future AC Jacobians contain one shared R/X vector and separate local P/Q and
same-interface source states.  Free local nuisance compensation is removed
by rank-revealing projections.  Final discrimination must still use the full
joint AC likelihood; this criterion is a local average squared-signal design.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
from time import perf_counter
from typing import Mapping
import numpy as np
from .symmetric_ac_local import CHANNELS, mean_and_jacobian


class InformationRankError(ValueError):
    """A required projection changed rank or lies near the declared cutoff."""


class InformationInitializationError(ValueError):
    """Required center setup failed, with every already attempted AC call."""
    def __init__(self, message, *, ac_calls, seconds, center_id, cause_kind):
        self.ac_calls = self.initialization_ac_calls = int(ac_calls)
        self.initialization_seconds = float(seconds)
        self.failed_center_id, self.cause_kind = str(center_id), str(cause_kind)
        self.ac_call_count_available = True
        super().__init__(message+f" [initialization AC calls={self.ac_calls}, center={self.failed_center_id}]")

    def record(self):
        return dict(kind=type(self).__name__, message=str(self),
            initialization_ac_calls=self.ac_calls,
            initialization_seconds=self.initialization_seconds,
            failed_center_id=self.failed_center_id, cause_kind=self.cause_kind,
            ac_call_count_available=True)


def _readonly(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FutureForecast:
    p: np.ndarray
    q: np.ndarray
    source_p: np.ndarray
    source_q: np.ndarray
    root_v: np.ndarray | float
    model: object = None
    provenance: str = "history_only"

    def __post_init__(self):
        for name in ("p", "q", "source_p", "source_q", "root_v"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


@dataclass(frozen=True)
class InformationCenter:
    id: str
    anchor_id: str
    target_support: tuple[int, ...]
    candidate: object
    r: np.ndarray
    x: np.ndarray
    history_p: np.ndarray
    history_q: np.ndarray
    history_source_p: np.ndarray
    history_source_q: np.ndarray
    history_root_v: np.ndarray | float
    future: FutureForecast | Mapping
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "target_support", tuple(sorted(self.target_support)))
        for name in ("r", "x", "history_p", "history_q", "history_source_p",
                     "history_source_q", "history_root_v"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _array(value, shape, name):
    try:
        result = np.broadcast_to(np.asarray(value, float), shape).copy()
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must broadcast to {shape}") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _weight(value):
    result = .5*np.diag([.003**2, .003**2]) if value is None else np.asarray(value, float)
    if (result.shape != (2, 2) or not np.isfinite(result).all()
            or not np.allclose(result, result.T, atol=1e-14, rtol=1e-12)):
        raise ValueError("weight must be a finite symmetric 2 by 2 matrix")
    if np.linalg.eigvalsh(result).min() < -1e-15 or np.trace(result) <= 0:
        raise ValueError("weight must be positive semidefinite with positive trace")
    return result.copy()


def _target_weights(value, expected):
    if value is None:
        return {support: 1/len(expected) for support in sorted(expected)}
    if not isinstance(value, Mapping):
        raise ValueError("target_weights must map every target support to a positive weight")
    result = {}
    for raw_support, raw_weight in value.items():
        try:
            support = tuple(sorted(int(j) for j in raw_support))
            valid_indices = all(int(j) == j for j in raw_support)
            number = float(raw_weight)
        except (ValueError, TypeError, OverflowError):
            raise ValueError("target weights require integer supports and finite positive values")
        if not valid_indices or support in result or not np.isfinite(number) or number <= 0:
            raise ValueError("target weights require distinct supports and a strictly positive floor")
        result[support] = number
    if set(result) != expected or not np.isclose(sum(result.values()), 1., atol=1e-12, rtol=1e-12):
        raise ValueError("all three target weights must be present and sum to one")
    return result


def project_columns(signal, nuisance, *, rcond=1e-10, rank_guard_factor=3.):
    """Return signal left after free nuisance compensation, in original rows.

    Nuisance columns are normalized for numerical rank decisions.  This does
    not change their mathematical span or impose a physical regularizer.
    Exact null columns and stable rank deficiency are allowed.  Near-cutoff
    singular values are retained as an explicit failure, never ridge-fixed.
    """
    a, b = np.asarray(signal, float), np.asarray(nuisance, float)
    if (a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not 0 < rcond < 1 or rank_guard_factor < 1):
        raise ValueError("finite matching matrices and valid rank controls required")
    scales = np.linalg.norm(b, axis=0)
    scales = np.where(scales > 0, scales, 1.)
    if b.shape[1]:
        u, singular, vt = np.linalg.svd(b/scales, full_matrices=False)
    else:
        u, singular, vt = np.zeros((len(a), 0)), np.zeros(0), np.zeros((0, 0))
    cutoff = rcond*singular[0] if len(singular) else 0.
    rank = int(np.count_nonzero(singular > cutoff))
    if len(singular) and cutoff > 0:
        near = (singular >= cutoff/rank_guard_factor) & (singular <= cutoff*rank_guard_factor)
        if np.any(near):
            raise InformationRankError("projection singular value near declared rank cutoff")
    basis = u[:, :rank]
    projected = a-basis@(basis.T@a)
    coefficients = ((vt[:rank].T/singular[:rank])@(basis.T@a))/scales[:, None] if rank else np.zeros((b.shape[1], a.shape[1]))
    orthogonal_error = float(np.linalg.norm((b/scales).T@projected, ord="fro"))
    return dict(projected=projected, coefficients=coefficients, rank=rank,
                singular_values=singular.tolist(), column_scales=scales.tolist(),
                rank_cutoff=float(cutoff), projection_rcond=float(rcond),
                rank_guard_factor=float(rank_guard_factor),
                nuisance_orthogonality_error=orthogonal_error)


def efficient_information(line_jacobian, target_columns, *, weight=None,
                          rcond=1e-10, rank_guard_factor=3.):
    """Eliminate all non-target shared lines from state-profiled AC rows."""
    line = np.asarray(line_jacobian, float)
    target = tuple(int(i) for i in target_columns)
    if (line.ndim != 2 or len(target) != 2 or len(set(target)) != 2
            or any(i < 0 or i >= line.shape[1] for i in target)):
        raise ValueError("two distinct target R/X columns required")
    other = tuple(i for i in range(line.shape[1]) if i not in target)
    result = project_columns(line[:, target], line[:, other], rcond=rcond,
                             rank_guard_factor=rank_guard_factor)
    residual = result["projected"]
    information = residual.T@residual
    information = .5*(information+information.T)
    omega = _weight(weight)
    result.update(information=information, score=.5*float(np.trace(omega@information)),
                  target_columns=target, other_columns=other)
    return result


def information_from_blocks(history_line, history_state, future_line, future_state,
                            target_columns, *, weight=None, rcond=1e-10,
                            rank_guard_factor=3.):
    """Two-stage profiling exactly retains H/N sharing of every line column."""
    opts = dict(rcond=rcond, rank_guard_factor=rank_guard_factor)
    hp = project_columns(history_line, history_state, **opts)
    fp = project_columns(future_line, future_state, **opts)
    joined = np.vstack((hp["projected"], fp["projected"]))
    info = efficient_information(joined, target_columns, weight=weight, **opts)
    return dict(history_projection=hp, future_projection=fp,
                joint_line_jacobian=joined, joint_information=info)


class StructuredInformationEvaluator:
    """History-only average target information; worst retained anchor.

    Targets have fixed positive task weights within an anchor (uniform by
    default).  The model-weight matrix
    is a declared engineering direction scale, not topology confidence.
    Each evaluation counts only newly attempted AC predictions.  Setup calls
    (historical and no-EV profiles) are separately recorded in metadata.
    """
    def __init__(self, history, centers, history_model, *, future_model=None,
                 base_kva=20., q_over_p=0., weight=None, target_weights=None, fd_step_kw=.01,
                 rcond=1e-10, rank_guard_factor=3., voltage_limits=(.9, 1.1),
                 r_bounds=(1e-6, .1), x_bounds=(1e-6, .1),
                 p_bounds=(-1., 1.), q_bounds=(-1., 1.), source_bounds=(-.5, .5)):
        self.history, self.history_model = history, history_model
        self.centers = tuple(centers)
        if history.n != 3:
            raise ValueError("this stage requires the declared three-terminal target family")
        if not self.centers or len({c.id for c in self.centers}) != len(self.centers):
            raise ValueError("nonempty uniquely identified information centers required")
        if history_model.fit_bias:
            raise ValueError("independently calibrated fixed biases required")
        if (not np.isfinite(base_kva) or base_kva <= 0 or not np.isfinite(q_over_p)
                or not np.isfinite(fd_step_kw) or fd_step_kw <= 0):
            raise ValueError("positive base power/FD step and finite Q/P required")
        self.base_kva, self.q_over_p = float(base_kva), float(q_over_p)
        self.weight, self.fd_step_kw = _weight(weight), float(fd_step_kw)
        self.rcond, self.rank_guard_factor = float(rcond), float(rank_guard_factor)
        self.bounds = dict(r=r_bounds, x=x_bounds, p=p_bounds, q=q_bounds,
                           source=source_bounds, voltage=voltage_limits)
        for name, bounds in self.bounds.items():
            if len(bounds) != 2 or not np.isfinite(bounds).all() or not bounds[0] < bounds[1]:
                raise ValueError(f"invalid {name} bounds")
        self.channels = CHANNELS+tuple(name for name in ("source_p", "source_q")
                                      if getattr(history, name) is not None)
        if history.master_p is None or history.master_q is None:
            raise ValueError("full P/Q/V information requires master P/Q anchors")
        self.horizon = len(np.asarray(_get(self.centers[0].future, "p")))
        if self.horizon < 1:
            raise ValueError("positive future horizon required")
        expected = {tuple(s) for s in combinations(range(history.n), 2)}
        self.target_weights = _target_weights(target_weights, expected)
        self.uniform_target_rule = target_weights is None
        anchors = {}
        for center in self.centers:
            anchors.setdefault(center.anchor_id, []).append(center)
        for name, group in anchors.items():
            if len(group) != len(expected) or {c.target_support for c in group} != expected:
                raise ValueError(f"anchor {name} must retain all three distinct target supports")
        self.anchors = tuple(anchors)
        self._models, self._contexts = {}, {}
        self._ac_calls = 0
        started = perf_counter()
        try:
            for center in self.centers:
                if center.candidate.n != history.n or center.target_support not in center.candidate.supports:
                    raise ValueError("center target must be an internal candidate support")
                e = center.candidate.supports.index(center.target_support)
                if len(center.target_support) in (1, history.n):
                    raise ValueError("target must be a proper internal edge")
                if center.r.shape != (center.candidate.m,) or center.x.shape != center.r.shape:
                    raise ValueError("center R/X shape must match all shared lines")
                self._models[center.id] = _get(center.future, "model") or future_model or history_model
                if self._models[center.id].fit_bias:
                    raise ValueError("fixed future biases required")
                target = (e, center.candidate.m+e)
                other = tuple(i for i in range(2*center.candidate.m) if i not in target)
                for name in ("r", "x"):
                    values = getattr(center, name)
                    lo, hi = self.bounds[name]
                    if not np.isfinite(values).all() or values[e] < 0 or values[e] > hi:
                        raise ValueError("nonnegative structural geometry edge required")
                    keep = np.delete(values, e)
                    if np.any(keep < lo) or np.any(keep > hi):
                        raise ValueError("other center lines violate the declared physical domain")
                geometry = self._geometry(center, future=False)
                hp = self._profile(geometry)
                hinfo = efficient_information(hp["projected"], target, weight=self.weight,
                    rcond=self.rcond, rank_guard_factor=self.rank_guard_factor)
                self._contexts[center.id] = dict(target=target, other=other,
                    history_geometry=geometry, history_projection=hp, history_information=hinfo)
                zero = self._point(center, np.zeros(self.horizon), 0)
                self._contexts[center.id]["zero"] = zero
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            raise InformationInitializationError(str(error), ac_calls=self._ac_calls,
                seconds=perf_counter()-started, center_id=center.id,
                cause_kind=type(error).__name__) from error
        self.initialization_ac_calls = self._ac_calls
        self.initialization_seconds = perf_counter()-started

    def _profile(self, geometry):
        return project_columns(geometry["line_jacobian"], geometry["state_jacobian"],
            rcond=self.rcond, rank_guard_factor=self.rank_guard_factor)

    def _geometry(self, center, *, future, power_kw=None, site=0):
        candidate = center.candidate
        if future:
            given = center.future
            shape = (self.horizon, self.history.n)
            p, q = _array(_get(given, "p"), shape, "future p"), _array(_get(given, "q"), shape, "future q")
            p[:, site] += np.asarray(power_kw)/self.base_kva
            q[:, site] += self.q_over_p*np.asarray(power_kw)/self.base_kva
            sp = _array(_get(given, "source_p"), (self.horizon,), "future source P")
            sq = _array(_get(given, "source_q"), (self.horizon,), "future source Q")
            root = _array(_get(given, "root_v"), (self.horizon,), "future root V")
            model = self._models[center.id]
        else:
            shape = self.history.p.shape
            p, q = _array(center.history_p, shape, "H p"), _array(center.history_q, shape, "H q")
            sp = _array(center.history_source_p, (self.history.t,), "H source P")
            sq = _array(center.history_source_q, (self.history.t,), "H source Q")
            root = _array(center.history_root_v, (self.history.t,), "H root V")
            model = self.history_model
        for name, values in (("p", p), ("q", q), ("source", sp), ("source", sq)):
            if np.any(values < self.bounds[name][0]) or np.any(values > self.bounds[name][1]):
                raise ValueError(f"required {name} expansion violates physical domain")
        if np.any(root <= 0):
            raise ValueError("root voltage must be positive")
        self._ac_calls += 1
        geometry = mean_and_jacobian(candidate, center.r, center.x, p, q, sp, sq, root,
                                     model, channels=self.channels)
        voltages = abs(geometry["physical"]["voltage"])
        if np.any(voltages < self.bounds["voltage"][0]) or np.any(voltages > self.bounds["voltage"][1]):
            raise ValueError("required AC expansion violates voltage domain")
        return geometry

    def _point(self, center, power_kw, site):
        ctx = self._contexts[center.id]
        geometry = self._geometry(center, future=True, power_kw=power_kw, site=site)
        fp = self._profile(geometry)
        joined = np.vstack((ctx["history_projection"]["projected"], fp["projected"]))
        info = efficient_information(joined, ctx["target"], weight=self.weight,
            rcond=self.rcond, rank_guard_factor=self.rank_guard_factor)
        return dict(geometry=geometry, future_projection=fp, information=info,
                    joint_line_jacobian=joined,
                    rank_signature=(fp["rank"], info["rank"]))

    def _active(self, center, point):
        physical = point["geometry"]["physical"]
        edge = self._contexts[center.id]["target"][0]
        result = {"target_geometry_lower_boundary": bool(center.r[edge] <= 1e-12 or center.x[edge] <= 1e-12)}
        for name in ("r", "x"):
            values = getattr(center, name)
            bound = self.bounds[name]
            tolerance = 1e-7*max(bound[1]-bound[0], 1e-4)
            others = np.delete(values, edge)
            result[name+"_other_lower"] = int(np.count_nonzero(others-bound[0] <= tolerance))
            result[name+"_other_upper"] = int(np.count_nonzero(bound[1]-others <= tolerance))
        for name in ("p", "q", "source_p", "source_q"):
            values = physical[name]
            bound = self.bounds["source" if name.startswith("source") else name]
            tolerance = 1e-7*max(bound[1]-bound[0], 1e-4)
            result[name+"_lower"] = int(np.count_nonzero(values-bound[0] <= tolerance))
            result[name+"_upper"] = int(np.count_nonzero(bound[1]-values <= tolerance))
        return result

    def _rank_check(self, base, neighbor):
        if neighbor["rank_signature"] != base["rank_signature"]:
            raise InformationRankError("required future state/shared-line projection changed rank during price differentiation")

    def evaluate(self, site, power_kw, *, with_gradient=True, initial_fits=None):
        """Evaluate one continuous curve, retaining all required center failures."""
        started, before = perf_counter(), self._ac_calls
        if not isinstance(site, (int, np.integer)) or not 0 <= site < self.history.n:
            raise ValueError("site outside terminal set")
        power = _array(power_kw, (self.horizon,), "EV power kW")
        targets, failures = [], []
        for center in self.centers:
            try:
                ctx = self._contexts[center.id]
                point = self._point(center, power, int(site))
                matrix = point["information"]["information"]
                hmatrix = ctx["history_information"]["information"]
                zmatrix = ctx["zero"]["information"]["information"]
                delta = matrix-zmatrix
                gradient = np.zeros(self.horizon)
                fd_records = []
                if with_gradient:
                    residual = point["information"]["projected"]
                    coefficients = point["information"]["coefficients"]
                    future_rows = point["future_projection"]["projected"].shape[0]
                    future_residual = residual[-future_rows:]
                    for time in range(self.horizon):
                        plus, minus = power.copy(), power.copy()
                        plus[time] += self.fd_step_kw
                        minus[time] -= self.fd_step_kw
                        gp, gm = self._point(center, plus, int(site)), self._point(center, minus, int(site))
                        self._rank_check(point, gp)
                        self._rank_check(point, gm)
                        dj = (gp["future_projection"]["projected"]-gm["future_projection"]["projected"])/(2*self.fd_step_kw)
                        change = dj[:, ctx["target"]]-dj[:, ctx["other"]]@coefficients
                        gradient[time] = float(np.trace(self.weight@future_residual.T@change))
                        fd_records.append(dict(time=time, step_kw=self.fd_step_kw,
                            ranks_plus=list(gp["rank_signature"]), ranks_minus=list(gm["rank_signature"])))
                if not np.isfinite(gradient).all():
                    raise FloatingPointError("nonfinite required information price")
                targets.append(dict(id=center.id, anchor_id=center.anchor_id,
                    target_support=list(center.target_support), candidate=center.candidate.name,
                    target_task_weight=self.target_weights[center.target_support],
                    target_columns=list(ctx["target"]), score=.5*float(np.trace(self.weight@delta)),
                    gradient_kw=gradient.tolist(), weight=self.weight.tolist(),
                    information_history=hmatrix.tolist(), information_joint=matrix.tolist(),
                    information_zero_EV=zmatrix.tolist(), information_EV_delta=delta.tolist(),
                    information_future_gain=(matrix-hmatrix).tolist(),
                    future_gain_min_eigenvalue=float(np.linalg.eigvalsh(matrix-hmatrix).min()),
                    EV_delta_eigenvalues=np.linalg.eigvalsh(delta).tolist(),
                    axes_weighted_half_distance=(.5*np.diag(self.weight)*np.diag(matrix)).tolist(),
                    axes_EV_delta=(.5*np.diag(self.weight)*np.diag(delta)).tolist(),
                    history_state_rank=ctx["history_projection"]["rank"],
                    history_line_nuisance_rank=ctx["history_information"]["rank"],
                    future_state_rank=point["future_projection"]["rank"],
                    joint_line_nuisance_rank=point["information"]["rank"],
                    future_state_projection={k:v for k,v in point["future_projection"].items() if k not in ("projected", "coefficients")},
                    joint_line_projection={k:v for k,v in point["information"].items() if k not in ("projected", "coefficients", "information", "score")},
                    active_boundaries=self._active(center, point),
                    history_active_boundaries=self._active(center, {"geometry": ctx["history_geometry"]}),
                    rank_signature_zero_EV=list(ctx["zero"]["rank_signature"]),
                    rank_change_relative_to_zero_EV=point["rank_signature"] != ctx["zero"]["rank_signature"],
                    price_differences=fd_records, metadata=dict(center.metadata)))
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                failures.append(dict(center_id=center.id, anchor_id=center.anchor_id,
                    target_support=list(center.target_support), kind=type(error).__name__, message=str(error)))
        complete = len(targets) == len(self.centers) and not failures
        pairs = []
        if complete:
            for anchor in self.anchors:
                group = [t for t in targets if t["anchor_id"] == anchor]
                if self.uniform_target_rule:
                    # Preserve the original v1 arithmetic when no weight rule is supplied.
                    score = float(np.mean([t["score"] for t in group]))
                    g = np.mean([t["gradient_kw"] for t in group], axis=0).tolist()
                else:
                    score = float(sum(t["target_task_weight"]*t["score"] for t in group))
                    g = sum((t["target_task_weight"]*np.asarray(t["gradient_kw"]) for t in group),
                            np.zeros(self.horizon)).tolist()
                pairs.append(dict(anchor_id=anchor, score=score,
                    gradient_kw=g, pq_only_gradient_kw=g, gradient_mode="same_full_information_no_PQ_ablation",
                    target_ids=[t["id"] for t in group],
                    target_task_weights=[t["target_task_weight"] for t in group]))
        return dict(success=complete, score=min(p["score"] for p in pairs) if pairs else None,
            pairs=pairs, targets=targets, failures=failures, fit_calls=0,
            total_ac_calls=self._ac_calls-before, seconds=perf_counter()-started,
            criterion="target_average_full_AC_local_information_EV_increment_worst_anchor",
            with_gradient=bool(with_gradient), local_linear_nuisance_refit=True,
            nonlinear_competitor_refit_during_selection=False,
            full_joint_AC_refit_required_for_final_evaluation=True,
            information_is_confidence_region=False, target_weights_are_topology_probabilities=False,
            target_aggregation="uniform" if self.uniform_target_rule else "fixed_H_only_positive_task_weights",
            minimum_target_task_weight=min(self.target_weights.values()),
            global_design_optimality_certified=False,
            initialization_ac_calls=self.initialization_ac_calls,
            rank_policy="column_scaled_SVD_with_fixed_rcond_and_explicit_rank_change_failure",
            physical_prediction_scope="history_only_forecasts_and_fixed_geometry_centers")