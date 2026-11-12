"""Independent full-domain Benders prototype for ONE wzzT extension.

No production module is imported or modified. The master chooses an arbitrary
nonempty support, laminar with the supplied old family. A lifted LP jointly
refits all old/new R/X weights. SciPy/HiGHS is the only solver dependency.
This is an experimental outer-loop implementation, not a full-tree optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable, Sequence
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, vstack


@dataclass
class BendersCut:
    cut_constant: float
    cut_gradient: np.ndarray
    generation_y: np.ndarray
    dual_value: float
    primal_objective: float


@dataclass
class SubproblemResult:
    objective: float
    cut_constant: float
    cut_gradient: np.ndarray
    dual_objective: float
    primal_dual_gap: float
    r_values: np.ndarray
    x_values: np.ndarray
    intercepts: np.ndarray
    u_r: np.ndarray
    u_x: np.ndarray
    primal_feasibility_error: float
    dual_free_variable_error: float


@dataclass
class BendersResult:
    support: tuple[int, ...] | None
    objective: float | None
    lower_bound: float
    upper_bound: float
    absolute_gap: float | None
    proven_optimal: bool
    stop_reason: str
    iterations: int
    elapsed_seconds: float
    r_values: np.ndarray = field(default_factory=lambda: np.empty(0))
    x_values: np.ndarray = field(default_factory=lambda: np.empty(0))
    intercepts: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    trace: list[dict] = field(default_factory=list)
    cuts: list[BendersCut] = field(default_factory=list)


class SubproblemSolveError(RuntimeError):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(message)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _nonnegative(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return value


def _laminar(left, right):
    a, b = set(left), set(right)
    return not (a & b) or a <= b or b <= a


class _Rows:
    def __init__(self):
        self.ri, self.ci, self.data, self.lower, self.upper = [], [], [], [], []

    def add(self, coefficients, lower=-np.inf, upper=np.inf):
        row = len(self.upper)
        for column, value in coefficients.items():
            if value:
                self.ri.append(row)
                self.ci.append(column)
                self.data.append(float(value))
        self.lower.append(float(lower))
        self.upper.append(float(upper))
        return row

    def matrix(self, columns):
        return coo_matrix(
            (self.data, (self.ri, self.ci)),
            shape=(len(self.upper), columns), dtype=float,
        ).tocsc()


class BendersExtensionProblem:
    """Prepared, fixed-bound model; y follows ``pairs`` (upper triangle).

    Input schema matches core: each scenario has P_terminal, Q_terminal,
    drop_target of shape (time, terminal), plus optional name. DataFrame
    columns are aligned by label; inconsistent timestamps are rejected.
    """

    def __init__(self, scenarios: Sequence[dict], supports: Iterable[Iterable[int]],
                 *, r_upper_bound: float, x_upper_bound: float):
        self.r_bound = _positive(r_upper_bound, "r_upper_bound")
        self.x_bound = _positive(x_upper_bound, "x_upper_bound")
        if not scenarios:
            raise ValueError("at least one scenario is required")
        first = scenarios[0]["P_terminal"]
        first_array = np.asarray(first, dtype=float)
        if first_array.ndim != 2 or min(first_array.shape) < 1:
            raise ValueError("P_terminal must be a nonempty two-dimensional table")
        self.n = first_array.shape[1]
        self.labels = tuple(first.columns) if hasattr(first, "columns") else tuple(range(self.n))
        if len(set(self.labels)) != self.n:
            raise ValueError("terminal labels must be unique")
        self.p, self.q, self.d = [], [], []
        for scenario in scenarios:
            raw_p = scenario["P_terminal"]
            blocks = []
            for key in ("P_terminal", "Q_terminal", "drop_target"):
                raw = scenario[key]
                if hasattr(raw, "columns"):
                    if len(raw.columns) != self.n or set(raw.columns) != set(self.labels):
                        raise ValueError(f"{key} terminal labels disagree")
                    if hasattr(raw_p, "index") and not raw.index.equals(raw_p.index):
                        raise ValueError("scenario timestamps must agree")
                    raw = raw.loc[:, list(self.labels)]
                block = np.asarray(raw, dtype=float)
                if block.ndim != 2 or block.shape[0] == 0 or block.shape[1] != self.n:
                    raise ValueError(f"{key} has an invalid shape")
                if not np.isfinite(block).all():
                    raise ValueError("scenario data must be finite")
                blocks.append(block.copy())
            if any(a.shape != blocks[0].shape for a in blocks):
                raise ValueError("P/Q/target shapes must agree")
            self.p.append(blocks[0]); self.q.append(blocks[1]); self.d.append(blocks[2])
        normalized = []
        for support in supports:
            members = tuple(support)
            if not members or any(not isinstance(i, (int, np.integer)) for i in members):
                raise ValueError("supports must contain integer terminal indices")
            item = tuple(sorted(set(int(i) for i in members)))
            if item[0] < 0 or item[-1] >= self.n or item in normalized:
                raise ValueError("duplicate or out-of-range support")
            if any(not _laminar(item, old) for old in normalized):
                raise ValueError("old supports must be laminar")
            normalized.append(item)
        self.supports = tuple(normalized)
        self.k, self.scenario_count = len(self.supports), len(self.p)
        self.pairs = tuple((i, j) for i in range(self.n) for j in range(i, self.n))
        self.pair_lookup = {pair: j for j, pair in enumerate(self.pairs)}
        self.pair_count = len(self.pairs)
        self.target = np.concatenate([a.reshape(-1) for a in self.d])
        self.observation_count = len(self.target)
        self._build_subproblem()
        self._build_master()

    def _build_subproblem(self):
        k, p, sn = self.k, self.pair_count, self.scenario_count * self.n
        cursor = 0
        self.blocks = {}
        lower, upper = [], []
        for name, size, lo, hi in (
            ("r", k + 1, 0, self.r_bound),
            ("x", k + 1, 0, self.x_bound),
            ("u_r", p, 0, self.r_bound),
            ("u_x", p, 0, self.x_bound),
            ("b", sn, -np.inf, np.inf),
            ("e", self.observation_count, 0, np.inf),
        ):
            self.blocks[name] = slice(cursor, cursor + size)
            cursor += size
            lower.extend([lo] * size); upper.extend([hi] * size)
        self.variable_count = cursor
        self.sub_lower, self.sub_upper = np.array(lower), np.array(upper)
        self.sub_c = np.zeros(cursor)
        self.sub_c[self.blocks["e"]] = 1.0 / self.observation_count
        rows, predictions = _Rows(), _Rows()
        groups, observation = [], 0
        for scenario, (pb, qb, db) in enumerate(zip(self.p, self.q, self.d)):
            for t in range(len(pb)):
                for i in range(self.n):
                    group = scenario * self.n + i
                    pred = {self.blocks["b"].start + group: 1.0}
                    for atom, support in enumerate(self.supports):
                        if i in support:
                            pred[self.blocks["r"].start + atom] = pb[t, list(support)].sum()
                            pred[self.blocks["x"].start + atom] = qb[t, list(support)].sum()
                    for j in range(self.n):
                        pair = self.pair_lookup[(min(i, j), max(i, j))]
                        pred[self.blocks["u_r"].start + pair] = pb[t, j]
                        pred[self.blocks["u_x"].start + pair] = qb[t, j]
                    predictions.add(pred)
                    positive, negative = pred.copy(), {col: -v for col, v in pred.items()}
                    positive[self.blocks["e"].start + observation] = -1.0
                    negative[self.blocks["e"].start + observation] = -1.0
                    rows.add(positive, upper=db[t, i])
                    rows.add(negative, upper=-db[t, i])
                    groups.append(group)
                    observation += 1
        rhs_rows, rhs_columns, rhs_values = [], [], []
        for weight, product, bound in (("r", "u_r", self.r_bound), ("x", "u_x", self.x_bound)):
            new = self.blocks[weight].start + k
            for pair in range(p):
                u = self.blocks[product].start + pair
                rows.add({u: 1, new: -1}, upper=0)
                upper_row = rows.add({u: 1}, upper=0)
                lower_row = rows.add({new: 1, u: -1}, upper=bound)
                rhs_rows.extend([upper_row, lower_row])
                rhs_columns.extend([pair, pair])
                rhs_values.extend([bound, -bound])
        self.sub_a = rows.matrix(cursor)
        self.sub_rhs = np.array(rows.upper)
        self.rhs_y = coo_matrix((rhs_values, (rhs_rows, rhs_columns)),
                               shape=(len(rows.upper), p)).tocsc()
        self.prediction = predictions.matrix(cursor)
        self.groups = np.array(groups)

    def _build_master(self):
        n, p, k = self.n, self.pair_count, self.k
        self.y_slice = slice(n, n + p)
        relation_start = n + p
        self.theta_index = n + p + 3 * k
        size = self.theta_index + 1
        rows = _Rows()
        rows.add({i: 1 for i in range(n)}, lower=1)
        for pair, (i, j) in enumerate(self.pairs):
            y = n + pair
            if i == j:
                rows.add({y: 1, i: -1}, lower=0, upper=0)
            else:
                rows.add({y: 1, i: -1}, upper=0)
                rows.add({y: 1, j: -1}, upper=0)
                rows.add({y: 1, i: -1, j: -1}, lower=-1)
        for atom, support in enumerate(self.supports):
            sub, sup, dis = [relation_start + 3 * atom + offset for offset in range(3)]
            rows.add({sub: 1, sup: 1, dis: 1}, lower=1, upper=1)
            for i in range(n):
                if i not in support:
                    rows.add({i: 1, sub: 1}, upper=1)
                else:
                    rows.add({sup: 1, i: -1}, upper=0)
                    rows.add({i: 1, dis: 1}, upper=1)
            rows.add({i: -1 if i in support else 1 for i in range(n)},
                     lower=1 - len(support))
        self.master_a = rows.matrix(size)
        self.master_lower = np.array(rows.lower)
        self.master_upper = np.array(rows.upper)
        self.master_c = np.zeros(size); self.master_c[-1] = 1
        self.master_integrality = np.zeros(size, dtype=np.uint8)
        self.master_integrality[:n] = 1
        self.master_integrality[relation_start:self.theta_index] = 1
        hi = np.ones(size); hi[-1] = np.inf
        self.master_bounds = Bounds(np.zeros(size), hi)

    def _dual_cut(self, result, y):
        """A bound-aware Lagrangian minorant, with residual duals repaired.

        Using inf over finite variable bounds compensates stationarity errors
        for weights/products. Residual duals are projected onto the intercept
        zero-sum constraints and their L1 box before constructing the cut.
        Certificates, like HiGHS itself, are numerical (not rational proofs).
        """
        marginal = np.minimum(np.asarray(result.ineqlin.marginals, dtype=float), 0.0)
        count = self.observation_count
        pi = marginal[:2 * count:2] - marginal[1:2 * count:2]
        for group in range(self.scenario_count * self.n):
            indices = np.flatnonzero(self.groups == group)
            values = pi[indices] - np.mean(pi[indices])
            if len(values) == 1:
                values[:] = 0
            else:
                values[-1] = -np.sum(values[:-1])
                peak = np.max(np.abs(values))
                if peak:
                    values *= min(1.0, (1.0 - 1e-12) / (count * peak))
                values[-1] = -np.sum(values[:-1])
            pi[indices] = values
        marginal[:2 * count:2] = np.minimum(pi, 0)
        marginal[1:2 * count:2] = -np.maximum(pi, 0)
        reduced = self.sub_c - self.sub_a.T @ marginal
        free_error = float(np.max(np.abs(reduced[self.blocks["b"]])))
        if free_error > 1e-10 or np.min(reduced[self.blocks["e"]]) < -1e-12:
            raise SubproblemSolveError(4, "could not construct a feasible numerical dual cut")
        finite = np.isfinite(self.sub_upper)
        correction = np.sum(np.minimum(reduced[finite] * self.sub_lower[finite],
                                       reduced[finite] * self.sub_upper[finite]))
        constant = float(marginal @ self.sub_rhs + correction)
        gradient = np.asarray(self.rhs_y.T @ marginal).reshape(-1)
        return constant, gradient, float(constant + gradient @ y), free_error

    def solve_subproblem(self, y, time_limit=None):
        y = np.asarray(y, dtype=float)
        if y.shape != (self.pair_count,) or not np.isfinite(y).all():
            raise ValueError("y must be a finite vector ordered by problem.pairs")
        if np.any(y < 0) or np.any(y > 1):
            raise ValueError("y must lie in [0,1]")
        options = {"primal_feasibility_tolerance": 1e-9,
                   "dual_feasibility_tolerance": 1e-9}
        if time_limit is not None:
            options["time_limit"] = _positive(time_limit, "time_limit")
        rhs = self.sub_rhs + self.rhs_y @ y
        result = linprog(self.sub_c, A_ub=self.sub_a, b_ub=rhs,
                         bounds=list(zip(self.sub_lower, self.sub_upper)),
                         method="highs", options=options)
        if result.status != 0 or result.x is None or not np.isfinite(result.x).all():
            raise SubproblemSolveError(int(result.status), str(result.message))
        value = result.x
        violation = max(0.0, float(np.max(self.sub_a @ value - rhs)),
                        float(np.max(self.sub_lower - value)),
                        float(np.max(value - self.sub_upper)))
        if violation > 1e-7:
            raise SubproblemSolveError(4, f"LP feasibility error {violation}")
        # Recompute the actual loss instead of trusting residual slack values.
        objective = float(np.mean(np.abs(self.target - self.prediction @ value)))
        constant, gradient, dual, free_error = self._dual_cut(result, y)
        if dual > objective + 1e-7 * max(1, abs(objective)):
            raise SubproblemSolveError(4, "dual cut exceeds the primal objective")
        return SubproblemResult(
            objective, constant, gradient, dual, float(objective - dual),
            value[self.blocks["r"]].copy(), value[self.blocks["x"]].copy(),
            value[self.blocks["b"]].reshape(self.scenario_count, self.n).copy(),
            value[self.blocks["u_r"]].copy(), value[self.blocks["u_x"]].copy(),
            violation, free_error,
        )

    def solve_master(self, cuts, time_limit=None):
        if cuts:
            extra = _Rows()
            for cut in cuts:
                coeff = {self.theta_index: -1.0}
                coeff.update({self.y_slice.start + j: g for j, g in enumerate(cut.cut_gradient)})
                extra.add(coeff, upper=-cut.cut_constant)
            matrix = vstack([self.master_a, extra.matrix(len(self.master_c))], format="csc")
            lower = np.concatenate([self.master_lower, extra.lower])
            upper = np.concatenate([self.master_upper, extra.upper])
        else:
            matrix, lower, upper = self.master_a, self.master_lower, self.master_upper
        options = {"mip_rel_gap": 0.0, "mip_abs_gap": 1e-10}
        if time_limit is not None:
            options["time_limit"] = _positive(time_limit, "time_limit")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected.*", category=RuntimeWarning)
            return milp(self.master_c, integrality=self.master_integrality,
                        bounds=self.master_bounds,
                        constraints=LinearConstraint(matrix, lower, upper), options=options)


def solve_best_laminar_extension_benders(
    scenarios, supports, *, r_upper_bound, x_upper_bound,
    max_iterations=200, time_limit=60.0,
    absolute_tolerance=1e-8, relative_tolerance=1e-8,
):
    """Solve one unrestricted laminar extension with global LB/UB reporting.

    ``time_limit`` is the entire call's budget, including model construction;
    individual solver calls receive only the remaining time. An unfinished
    result may contain an incumbent, but ``proven_optimal`` remains false.
    Bounds stay fixed and no candidate pool, no-good cut or core patch is used.
    """
    started = perf_counter()
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    absolute_tolerance = _nonnegative(absolute_tolerance, "absolute_tolerance")
    relative_tolerance = _nonnegative(relative_tolerance, "relative_tolerance")
    budget = None if time_limit is None else _positive(time_limit, "time_limit")
    problem = BendersExtensionProblem(scenarios, supports,
                                     r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound)
    cuts, trace, seen = [], [], set()
    lb, ub = 0.0, np.inf
    best_support, best_fit = None, None
    stop, optimal = "iteration_limit", False

    def remaining():
        return None if budget is None else budget - (perf_counter() - started)

    def requested_tolerance():
        return absolute_tolerance + relative_tolerance * max(abs(ub), 1e-15)

    def converged():
        return np.isfinite(ub) and abs(ub - lb) <= requested_tolerance()

    for iteration in range(1, int(max_iterations) + 1):
        left = remaining()
        if left is not None and left <= 0:
            stop = "time_limit"; break
        master_started = perf_counter()
        master = problem.solve_master(cuts, time_limit=left)
        master_seconds = perf_counter() - master_started
        bound = getattr(master, "mip_dual_bound", None)
        if bound is not None and np.isfinite(bound):
            lb = max(lb, float(bound))
        if master.status == 2:
            if not cuts:
                lb, stop = np.inf, "no_feasible_extension"
            else:
                stop = "master_infeasible_after_cuts"
            break
        if np.isfinite(ub) and lb > ub + requested_tolerance():
            stop = "inconsistent_bounds"; break
        if converged():
            optimal, stop = True, "optimal_within_tolerance"; break
        if master.x is None or master.status not in (0, 1):
            stop = "master_time_limit" if master.status == 1 else "master_failure"
            break
        z_raw = master.x[:problem.n]
        z = np.rint(z_raw).astype(int)
        support = tuple(int(i) for i in np.flatnonzero(z))
        if (np.max(np.abs(z_raw - z)) > 1e-6 or not support
                or support in problem.supports
                or any(not _laminar(support, old) for old in problem.supports)):
            stop = "invalid_master_incumbent"; break
        y = np.array([z[i] * z[j] for i, j in problem.pairs], dtype=float)
        if np.max(np.abs(master.x[problem.y_slice] - y)) > 1e-6:
            stop = "invalid_master_products"; break
        left = remaining()
        if left is not None and left <= 0:
            stop = "time_limit"; break
        sub_started = perf_counter()
        try:
            fitted = problem.solve_subproblem(y, time_limit=left)
        except SubproblemSolveError as exc:
            stop = "subproblem_time_limit" if exc.status == 1 else "subproblem_failure"
            trace.append({"iteration": iteration, "support": list(support),
                          "failure": str(exc), "status": stop})
            break
        sub_seconds = perf_counter() - sub_started
        # Reconstruct the exact integer-support model for a feasible UB.
        rv = np.clip(fitted.r_values, 0, problem.r_bound)
        xv = np.clip(fitted.x_values, 0, problem.x_bound)
        total_error = 0.0
        for s, (pb, qb, db) in enumerate(zip(problem.p, problem.q, problem.d)):
            predicted = np.broadcast_to(fitted.intercepts[s], db.shape).copy()
            for k, atom in enumerate((*problem.supports, support)):
                members = list(atom)
                contribution = rv[k] * pb[:, members].sum(axis=1) + xv[k] * qb[:, members].sum(axis=1)
                predicted[:, members] += contribution[:, None]
            total_error += float(np.abs(db - predicted).sum())
        feasible_objective = total_error / problem.observation_count
        if feasible_objective < ub:
            ub, best_support, best_fit = feasible_objective, support, fitted
            best_fit.r_values, best_fit.x_values = rv, xv
        cut = BendersCut(fitted.cut_constant, fitted.cut_gradient.copy(), y.copy(),
                         fitted.dual_objective, fitted.objective)
        cuts.append(cut)
        trace.append({
            "iteration": iteration, "support": list(support),
            "master_status": int(master.status), "master_objective": float(master.fun),
            "lower_bound": lb, "upper_bound": ub, "absolute_gap": abs(ub - lb), "bound_inversion": max(0.0, lb - ub),
            "subproblem_objective": fitted.objective, "feasible_objective": feasible_objective,
            "subproblem_dual": fitted.dual_objective, "primal_dual_gap": fitted.primal_dual_gap,
            "primal_feasibility_error": fitted.primal_feasibility_error,
            "dual_free_variable_error": fitted.dual_free_variable_error,
            "master_seconds": master_seconds, "subproblem_seconds": sub_seconds,
            "elapsed_seconds": perf_counter() - started,
        })
        if lb > ub + requested_tolerance():
            stop = "inconsistent_bounds"; break
        if converged():
            optimal, stop = True, "optimal_within_tolerance"; break
        if support in seen:
            stop = "numerical_stall"; break
        seen.add(support)
    gap = abs(ub - lb) if np.isfinite(ub) and np.isfinite(lb) else None
    return BendersResult(
        best_support, None if best_fit is None else float(ub), float(lb), float(ub), gap,
        optimal, stop, len(cuts), perf_counter() - started,
        np.empty(0) if best_fit is None else best_fit.r_values,
        np.empty(0) if best_fit is None else best_fit.x_values,
        np.empty((0, 0)) if best_fit is None else best_fit.intercepts,
        trace, cuts,
    )
