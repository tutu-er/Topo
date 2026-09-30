"""Fit shared R/X laminar atoms with L1 loss and a forward support search.

R = sum_k r_k z_k z_k.T and X = sum_k x_k z_k z_k.T share binary
terminal-clade supports. Each extension searches a new support and jointly
refits all weights. Initial supports remain fixed; new supports are optimized
over all nonempty, distinct extensions compatible with the current family.
The observed-root target is Y = V_root**2 - V_terminal**2, with prediction
P @ R.T + Q @ X.T and no fitted measurement offsets.

Data validation, standard matrix construction, solver dispatch and path
selection live here. SciPy/HiGHS is the default; gurobi_milp provides the
optional Gurobi adapter. Exactness covers one bounded extension only;
the complete forward path remains greedy.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from math import sqrt
from time import perf_counter
from typing import Hashable, Iterable, Sequence
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array


# Data and result records.

IndexSupport = tuple[int, ...]
_EXACT_MIP_REL_GAP_TOLERANCE = 1e-10
_EXACT_MIP_ABS_GAP_TOLERANCE = 1e-10
_SUPPORT_INTEGRAL_TOLERANCE = 1e-6
_COEFFICIENT_BOUND_TOLERANCE = 1e-8
_OBJECTIVE_REPRODUCTION_REL_TOLERANCE = 1e-7


@dataclass(frozen=True)
class SolverDiagnostics:
    """One LP/MILP solve, with SciPy-compatible status and native provenance."""

    status: int
    success: bool
    message: str
    objective: float | None
    dual_bound: float | None
    mip_gap: float | None
    node_count: int | None
    runtime_seconds: float
    variable_count: int
    binary_variable_count: int
    constraint_count: int
    solver: str = "highs"
    raw_status: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FixedSupportSolution:
    """L1 optimum for a fixed ordered list of supports."""

    supports: tuple[IndexSupport, ...]
    r_values: np.ndarray
    x_values: np.ndarray
    objective: float
    diagnostics: SolverDiagnostics


@dataclass(frozen=True)
class ExtensionSolution:
    """Exact best admissible one-support extension of a fixed family."""

    support: IndexSupport | None
    r_values: np.ndarray
    x_values: np.ndarray
    objective: float | None
    diagnostics: SolverDiagnostics


@dataclass(frozen=True)
class LaminarL1PathPoint:
    """One fully corrected model on the forward path."""

    iteration: int
    support_indices: tuple[IndexSupport, ...]
    support_labels: tuple[tuple[Hashable, ...], ...]
    r_values: np.ndarray
    x_values: np.ndarray
    r_matrix: np.ndarray
    x_matrix: np.ndarray
    train_mae: float
    validation_mae: float | None
    validation_se: float | None
    accepted_gain: float | None
    solver: SolverDiagnostics


@dataclass(frozen=True)
class LaminarL1Result:
    """Selected model and the complete auditable forward path."""

    terminal_labels: tuple[Hashable, ...]
    support_indices: tuple[IndexSupport, ...]
    support_labels: tuple[tuple[Hashable, ...], ...]
    r_values: np.ndarray
    x_values: np.ndarray
    r_matrix: np.ndarray
    x_matrix: np.ndarray
    train_mae: float
    validation_mae: float | None
    selected_path_index: int
    path: tuple[LaminarL1PathPoint, ...]
    attempted_extensions: tuple[ExtensionSolution, ...]
    stop_reason: str
    r_upper_bound: float
    x_upper_bound: float
    bound_expansions: int

    def summary(self) -> dict:
        return {
            "terminal_labels": list(self.terminal_labels),
            "support_labels": [list(item) for item in self.support_labels],
            "r_values": self.r_values.tolist(),
            "x_values": self.x_values.tolist(),
            "train_mae": self.train_mae,
            "validation_mae": self.validation_mae,
            "selected_path_index": self.selected_path_index,
            "path_length": len(self.path),
            "stop_reason": self.stop_reason,
            "r_upper_bound": self.r_upper_bound,
            "x_upper_bound": self.x_upper_bound,
            "bound_expansions": self.bound_expansions,
            "solver": self.path[self.selected_path_index].solver.solver,
        }


@dataclass(frozen=True)
class _PreparedScenarios:
    labels: tuple[Hashable, ...]
    p: tuple[np.ndarray, ...]
    q: tuple[np.ndarray, ...]
    target: tuple[np.ndarray, ...]

    @property
    def n(self) -> int:
        return len(self.labels)

    @property
    def observation_count(self) -> int:
        return sum(block.shape[0] * self.n for block in self.p)


# Input validation and support rules.


def _require_positive_finite(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _require_nonnegative_finite(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return result


def _validate_solver_parameters(
    *,
    r_upper_bound: float,
    x_upper_bound: float,
    time_limit: float | None,
    mip_rel_gap: float | None = None,
) -> tuple[float, float, float | None, float | None]:
    r_bound = _require_positive_finite("r_upper_bound", r_upper_bound)
    x_bound = _require_positive_finite("x_upper_bound", x_upper_bound)
    checked_time_limit = (
        None
        if time_limit is None
        else _require_positive_finite("time_limit", time_limit)
    )
    checked_gap = (
        None
        if mip_rel_gap is None
        else _require_nonnegative_finite("mip_rel_gap", mip_rel_gap)
    )
    return r_bound, x_bound, checked_time_limit, checked_gap


def _validate_solver(solver: str) -> None:
    if solver not in ("highs", "gurobi"):
        raise ValueError("solver must be 'highs' or 'gurobi'")


def solver_diagnostics_prove_optimality(
    diagnostics: SolverDiagnostics,
    *,
    relative_gap_tolerance: float = _EXACT_MIP_REL_GAP_TOLERANCE,
    absolute_gap_tolerance: float = _EXACT_MIP_ABS_GAP_TOLERANCE,
) -> bool:
    """Return whether the solver certified the requested numerical optimum."""

    if diagnostics.status != 0 or not diagnostics.success:
        return False
    if diagnostics.binary_variable_count == 0:
        return True
    if diagnostics.objective is not None and diagnostics.dual_bound is not None:
        if np.isfinite(diagnostics.objective) and np.isfinite(diagnostics.dual_bound):
            absolute_gap = abs(
                float(diagnostics.objective) - float(diagnostics.dual_bound)
            )
            if absolute_gap <= absolute_gap_tolerance:
                return True
    # A status-zero MIP result without either a finite primal/dual certificate
    # or a reported relative gap is not enough for this module's deliberately
    # strict "exact extension" claim. Older SciPy/HiGHS combinations may omit
    # these attributes; callers then receive the incumbent objective only, not
    # a support that is labelled exact.
    if diagnostics.mip_gap is None:
        return False
    return bool(
        np.isfinite(diagnostics.mip_gap)
        and diagnostics.mip_gap >= 0.0
        and diagnostics.mip_gap <= relative_gap_tolerance
    )


def _objectives_agree(left: float, right: float) -> bool:
    scale = max(1.0, abs(float(left)), abs(float(right)))
    return bool(
        abs(float(left) - float(right))
        <= _OBJECTIVE_REPRODUCTION_REL_TOLERANCE * scale
    )


def _matrix_from_scenario(value, labels: tuple[Hashable, ...], key: str, index=None) -> np.ndarray:
    if hasattr(value, "loc") and hasattr(value, "columns"):
        if not value.columns.is_unique or set(value.columns) != set(labels):
            raise ValueError(f"{key} must contain the same unique terminal columns")
        if not value.index.is_unique or (index is not None and set(value.index) != set(index)):
            raise ValueError(f"{key} must contain the same unique time indices")
        array = value.loc[value.index if index is None else index, list(labels)].to_numpy(dtype=float)
    else:
        array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{key} must be a two-dimensional table")
    return array


def _prepare_scenarios(scenarios: Sequence[dict] | _PreparedScenarios) -> _PreparedScenarios:
    """Validate external tables once; reuse prepared blocks within a fit."""
    if isinstance(scenarios, _PreparedScenarios):
        return scenarios
    if not scenarios:
        raise ValueError("at least one scenario is required")
    first_p = scenarios[0]["P_terminal"]
    if hasattr(first_p, "columns"):
        labels = tuple(first_p.columns.tolist())
    else:
        first_array = np.asarray(first_p)
        if first_array.ndim != 2:
            raise ValueError("P_terminal must be two-dimensional")
        labels = tuple(range(first_array.shape[1]))
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("terminal columns must be nonempty and unique")

    p_blocks: list[np.ndarray] = []
    q_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    for scenario in scenarios:
        reference = scenario["P_terminal"]
        index = reference.index if hasattr(reference, "columns") else None
        p = _matrix_from_scenario(reference, labels, "P_terminal", index)
        q = _matrix_from_scenario(scenario["Q_terminal"], labels, "Q_terminal", index)
        target = _matrix_from_scenario(scenario["drop_target"], labels, "drop_target", index)
        if p.shape != q.shape or p.shape != target.shape:
            raise ValueError("P_terminal, Q_terminal, and drop_target must have equal shape")
        if p.shape[0] == 0:
            raise ValueError("each scenario must contain at least one time sample")
        if p.shape[1] != len(labels):
            raise ValueError("all scenarios must use the same terminal count")
        if not (np.isfinite(p).all() and np.isfinite(q).all() and np.isfinite(target).all()):
            raise ValueError("scenario data must contain only finite values")
        p_blocks.append(p)
        q_blocks.append(q)
        target_blocks.append(target)
    return _PreparedScenarios(
        labels=labels,
        p=tuple(p_blocks),
        q=tuple(q_blocks),
        target=tuple(target_blocks),
    )


def normalize_supports(
    supports: Iterable[Iterable[int]], n: int
) -> tuple[IndexSupport, ...]:
    """Validate, sort, and canonicalize positional supports."""

    normalized: list[IndexSupport] = []
    seen: set[IndexSupport] = set()
    for support in supports:
        item = tuple(sorted({int(index) for index in support}))
        if not item:
            raise ValueError("supports must be nonempty")
        if item[0] < 0 or item[-1] >= n:
            raise ValueError("support index is outside the terminal range")
        if item in seen:
            raise ValueError("duplicate support")
        seen.add(item)
        normalized.append(item)
    if not is_laminar_family(normalized):
        raise ValueError("supports must form a laminar family")
    return tuple(normalized)


def is_laminar_family(supports: Iterable[Iterable[int]]) -> bool:
    """Return whether every support pair is nested or disjoint."""

    family = [set(item) for item in supports]
    for left_index, left in enumerate(family):
        for right in family[left_index + 1 :]:
            overlap = left & right
            if overlap and not (left <= right or right <= left):
                return False
    return True


def is_admissible_extension(
    support: Iterable[int], existing_supports: Iterable[Iterable[int]], n: int
) -> bool:
    """Check nonemptiness, uniqueness, range, and laminar compatibility."""

    candidate = tuple(sorted({int(index) for index in support}))
    if not candidate or candidate[0] < 0 or candidate[-1] >= n:
        return False
    family = [tuple(sorted(set(item))) for item in existing_supports]
    if candidate in family:
        return False
    return is_laminar_family([*family, candidate])


def _round_and_validate_support(
    z_values: np.ndarray,
    existing_supports: tuple[IndexSupport, ...],
    n: int,
) -> IndexSupport:
    values = np.asarray(z_values, dtype=float)
    if values.shape != (n,) or not np.isfinite(values).all():
        raise RuntimeError("MILP returned an invalid support vector")
    rounded = np.rint(values)
    if float(np.max(np.abs(values - rounded))) > _SUPPORT_INTEGRAL_TOLERANCE:
        raise RuntimeError("MILP support vector violates the integrality tolerance")
    support = tuple(int(index) for index in np.flatnonzero(rounded > 0.5))
    if not is_admissible_extension(support, existing_supports, n):
        raise RuntimeError("MILP returned a nonempty/unique/laminar-infeasible support")
    return support


# Matrix reconstruction, evaluation and path selection.


def build_matrices_from_atoms(
    n: int,
    supports: Sequence[Sequence[int]],
    r_values: Sequence[float],
    x_values: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Construct symmetric R/X matrices from nonnegative support atoms.

    This low-level constructor intentionally permits crossing supports so it
    can also build adversarial matrices for tests. The estimators themselves
    call :func:`normalize_supports` and therefore require a laminar family.
    """

    if not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)
    support_values = tuple(tuple(support) for support in supports)
    r_array = np.asarray(tuple(r_values), dtype=float)
    x_array = np.asarray(tuple(x_values), dtype=float)
    if r_array.ndim != 1 or x_array.ndim != 1:
        raise ValueError("atom coefficients must be one-dimensional")
    if len(support_values) != len(r_array) or len(support_values) != len(x_array):
        raise ValueError("support and coefficient counts must agree")
    if not (np.isfinite(r_array).all() and np.isfinite(x_array).all()):
        raise ValueError("atom coefficients must be finite")
    if np.any(r_array < 0.0) or np.any(x_array < 0.0):
        raise ValueError("atom coefficients must be nonnegative")
    r_matrix = np.zeros((n, n), dtype=float)
    x_matrix = np.zeros((n, n), dtype=float)
    for atom_index, (support, r_value, x_value) in enumerate(
        zip(support_values, r_array, x_array, strict=True)
    ):
        index_tuple = tuple(int(value) for value in support)
        if not index_tuple:
            raise ValueError(f"support {atom_index} must be nonempty")
        if len(set(index_tuple)) != len(index_tuple):
            raise ValueError(f"support {atom_index} contains duplicate indices")
        if min(index_tuple) < 0 or max(index_tuple) >= n:
            raise ValueError(f"support {atom_index} is outside the terminal range")
        index = np.asarray(index_tuple, dtype=int)
        r_matrix[np.ix_(index, index)] += float(r_value)
        x_matrix[np.ix_(index, index)] += float(x_value)
    return r_matrix, x_matrix


def estimate_atom_upper_bounds(
    scenarios: Sequence[dict] | _PreparedScenarios, *, safety_factor: float = 2.0, floor: float = 1e-8
) -> tuple[float, float]:
    """Derive data-only atom bounds from the current dense OLS coefficient scale."""

    if not np.isfinite(safety_factor) or safety_factor <= 1.0:
        raise ValueError("safety_factor must exceed one")
    floor = _require_positive_finite("floor", floor)
    prepared = _prepare_scenarios(scenarios)
    design = np.hstack([np.vstack(prepared.p), np.vstack(prepared.q)])
    target = np.vstack(prepared.target)
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    n = prepared.n
    r_raw = coefficients[:n, :].T
    x_raw = coefficients[n : 2 * n, :].T
    r_symmetric = np.maximum(0.0, 0.5 * (r_raw + r_raw.T))
    x_symmetric = np.maximum(0.0, 0.5 * (x_raw + x_raw.T))
    r_scale = max(float(np.max(r_symmetric)), float(floor))
    x_scale = max(float(np.max(x_symmetric)), float(floor))
    return safety_factor * r_scale, safety_factor * x_scale


def evaluate_l1_matrices(
    scenarios: Sequence[dict] | _PreparedScenarios,
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    *,
    blocks_per_scenario: int = 4,
) -> tuple[float, float, np.ndarray]:
    """Evaluate the physical prediction P R.T + Q X.T without refitting."""

    prepared = _prepare_scenarios(scenarios)
    if not isinstance(blocks_per_scenario, (int, np.integer)) or blocks_per_scenario < 1:
        raise ValueError("blocks_per_scenario must be a positive integer")
    r_value = np.asarray(r_matrix, dtype=float)
    x_value = np.asarray(x_matrix, dtype=float)
    if r_value.shape != (prepared.n, prepared.n) or x_value.shape != r_value.shape:
        raise ValueError("R/X shape does not match validation terminals")
    if not (np.isfinite(r_value).all() and np.isfinite(x_value).all()):
        raise ValueError("R/X must contain only finite values")
    absolute_parts: list[np.ndarray] = []
    block_means: list[float] = []
    for p_block, q_block, target_block in zip(prepared.p, prepared.q, prepared.target, strict=True):
        absolute = np.abs(target_block - p_block @ r_value.T - q_block @ x_value.T)
        absolute_parts.append(absolute.reshape(-1))
        for time_indices in np.array_split(np.arange(len(absolute)), blocks_per_scenario):
            if time_indices.size:
                block_means.append(float(np.mean(absolute[time_indices, :])))
    all_absolute = np.concatenate(absolute_parts)
    mae = float(np.mean(all_absolute))
    block_array = np.asarray(block_means, dtype=float)
    standard_error = (
        float(np.std(block_array, ddof=1) / sqrt(len(block_array)))
        if len(block_array) > 1
        else 0.0
    )
    return mae, standard_error, block_array


def _touches_bound(values: np.ndarray, bound: float, fraction: float) -> bool:
    return bool(values.size and np.max(values) >= fraction * bound)


def _choose_path_point(path: Sequence[LaminarL1PathPoint]) -> int:
    if not path or path[0].validation_mae is None:
        return len(path) - 1
    means = np.asarray([point.validation_mae for point in path], dtype=float)
    best = int(np.argmin(means))
    threshold = means[best] + float(path[best].validation_se or 0.0)
    eligible = np.flatnonzero(means <= threshold)
    return min(
        (int(index) for index in eligible),
        key=lambda index: (len(path[index].support_indices), index),
    )




# Standard LP/MILP variables and constraints.


class _VariableBuilder:
    """Create a flat MILP vector while retaining named slices."""

    def __init__(self) -> None:
        self.objective: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.integrality: list[int] = []
        self.slices: dict[str, slice] = {}

    def add(
        self,
        name: str,
        size: int,
        *,
        lower: float | np.ndarray = 0.0,
        upper: float | np.ndarray = np.inf,
        objective: float | np.ndarray = 0.0,
        integral: bool = False,
    ) -> slice:
        if name in self.slices:
            raise ValueError(f"duplicate variable block: {name}")
        start = len(self.objective)
        stop = start + int(size)
        block = slice(start, stop)
        self.slices[name] = block
        self.lower.extend(np.broadcast_to(lower, (size,)).astype(float).tolist())
        self.upper.extend(np.broadcast_to(upper, (size,)).astype(float).tolist())
        self.objective.extend(np.broadcast_to(objective, (size,)).astype(float).tolist())
        self.integrality.extend([1 if integral else 0] * size)
        return block

    @property
    def size(self) -> int:
        return len(self.objective)


class _ConstraintBuilder:
    """Sparse ranged-row builder for scipy.optimize.LinearConstraint."""

    def __init__(self) -> None:
        self.row_index: list[int] = []
        self.col_index: list[int] = []
        self.data: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(
        self,
        entries: dict[int, float],
        *,
        lower: float = -np.inf,
        upper: float = np.inf,
    ) -> None:
        row = len(self.lower)
        for col, value in entries.items():
            if value != 0.0:
                self.row_index.append(row)
                self.col_index.append(int(col))
                self.data.append(float(value))
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    def build(self, variable_count: int) -> LinearConstraint:
        matrix = coo_array(
            (self.data, (self.row_index, self.col_index)),
            shape=(len(self.lower), variable_count),
            dtype=float,
        ).tocsc()
        return LinearConstraint(
            matrix,
            np.asarray(self.lower, dtype=float),
            np.asarray(self.upper, dtype=float),
        )

    @property
    def size(self) -> int:
        return len(self.lower)


def _fixed_atom_features(
    prepared: _PreparedScenarios, supports: tuple[IndexSupport, ...]
) -> tuple[np.ndarray, np.ndarray]:
    if not supports:
        empty = np.empty((prepared.observation_count, 0), dtype=float)
        return empty, empty.copy()
    indicators = np.zeros((len(supports), prepared.n))
    for indicator, support in zip(indicators, supports, strict=True):
        indicator[list(support)] = 1.0

    def features(blocks):
        # Each column is vec((P_s z_k) z_k.T), likewise for Q_s.
        return np.vstack([
            np.column_stack([((block @ z)[:, None] * z).ravel() for z in indicators])
            for block in blocks
        ])

    return features(prepared.p), features(prepared.q)


def _add_absolute_residual_constraints(
    prepared: _PreparedScenarios,
    supports: tuple[IndexSupport, ...],
    model: L1Model,
    *,
    new_u_r_block: slice | None = None,
    new_u_x_block: slice | None = None,
    pair_lookup: dict[tuple[int, int], int] | None = None,
    source_terms: SourceTermProvider | None = None,
) -> None:
    """Impose prediction - e <= target <= prediction + e for each observation.

    An optional provider returns affine source coefficients and a constant for
    (scenario, time, output). It may use existing non-residual variables only;
    source variables and their constraints must be allocated before this call.
    """
    rows = model.rows
    if source_terms is not None and not callable(source_terms):
        raise ValueError("source_terms must be callable or None")
    p_features, q_features = _fixed_atom_features(prepared, supports)
    has_new_atom = new_u_r_block is not None and new_u_x_block is not None
    if has_new_atom and pair_lookup is None:
        raise AssertionError("pair lookup is required for a new atom")
    observation = 0
    for scenario_index, (p_block, q_block, target) in enumerate(
        zip(prepared.p, prepared.q, prepared.target, strict=True)
    ):
        for time_index in range(p_block.shape[0]):
            for output_index in range(prepared.n):
                prediction: dict[int, float] = {}
                for weights, features in ((model.r, p_features), (model.x, q_features)):
                    prediction.update({
                        weights.start + k: float(value)
                        for k, value in enumerate(features[observation])
                    })
                if has_new_atom:
                    for input_index in range(prepared.n):
                        pair = (min(output_index, input_index), max(output_index, input_index))
                        pair_index = pair_lookup[pair]
                        prediction[new_u_r_block.start + pair_index] = float(p_block[time_index, input_index])
                        prediction[new_u_x_block.start + pair_index] = float(q_block[time_index, input_index])

                value = float(target[time_index, output_index])
                if source_terms is not None:
                    variable_count, row_count = model.variables.size, rows.size
                    terms = source_terms(model, scenario_index, time_index, output_index)
                    if model.variables.size != variable_count or model.rows.size != row_count:
                        raise ValueError("source_terms must not add variables or constraints")
                    try:
                        coefficients, constant = terms
                    except (TypeError, ValueError) as exc:
                        raise ValueError("source_terms must return (coefficient mapping, constant)") from exc
                    if not isinstance(coefficients, Mapping):
                        raise ValueError("source coefficients must be a mapping")
                    try:
                        constant = float(constant)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise ValueError("source constant must be finite") from exc
                    if not np.isfinite(constant):
                        raise ValueError("source constant must be finite")
                    for column, coefficient in coefficients.items():
                        if isinstance(column, (bool, np.bool_)) or not isinstance(column, (int, np.integer)):
                            raise ValueError("source columns must be integer variable indices, not bool")
                        column = int(column)
                        if not 0 <= column < variable_count:
                            raise ValueError("source columns must refer to already allocated variables")
                        if model.absolute.start <= column < model.absolute.stop:
                            raise ValueError("source terms cannot use absolute residual variables")
                        try:
                            coefficient = float(coefficient)
                        except (TypeError, ValueError, OverflowError) as exc:
                            raise ValueError("source coefficients must be finite") from exc
                        if not np.isfinite(coefficient):
                            raise ValueError("source coefficients must be finite")
                        combined = prediction.get(column, 0.0) + coefficient
                        if not np.isfinite(combined):
                            raise ValueError("combined prediction coefficients must be finite")
                        prediction[column] = combined
                    value -= constant
                    if not np.isfinite(value):
                        raise ValueError("source-adjusted residual target must be finite")

                # The row builder drops zero coefficients for both residual sides.
                error = model.absolute.start + observation
                rows.add({**prediction, error: 1.0}, lower=value)
                rows.add({**prediction, error: -1.0}, upper=value)
                observation += 1


@dataclass
class L1Model:
    """Sparse model and the variable slices needed to read its solution."""

    variables: _VariableBuilder
    rows: _ConstraintBuilder
    r: slice
    x: slice
    absolute: slice
    new_r: slice | None = None
    new_x: slice | None = None
    z: slice | None = None


SourceTermProvider = Callable[[L1Model, int, int, int], tuple[Mapping[int, float], float]]


@dataclass(frozen=True)
class _ExtensionBlocks:
    """Allocated atom-product and relation blocks for one support extension."""

    pairs: tuple[tuple[int, int], ...]
    pair_lookup: dict[tuple[int, int], int]
    y: slice
    u_r: slice
    u_x: slice
    relations: slice


def _base_model(
    prepared: _PreparedScenarios,
    atom_count: int,
    *,
    r_upper_bound: float,
    x_upper_bound: float,
    prefix: str = "",
) -> L1Model:
    """Create the shared weights and L1 residual variables."""
    variables = _VariableBuilder()
    r = variables.add(f"{prefix}r", atom_count, upper=r_upper_bound)
    x = variables.add(f"{prefix}x", atom_count, upper=x_upper_bound)
    absolute = variables.add(
        "absolute_residual",
        prepared.observation_count,
        objective=1.0 / prepared.observation_count,
    )
    return L1Model(variables, _ConstraintBuilder(), r, x, absolute)


def build_fixed_model(
    prepared: _PreparedScenarios,
    supports: tuple[IndexSupport, ...],
    *,
    r_upper_bound: float,
    x_upper_bound: float,
) -> L1Model:
    """Fit every existing atom weight with an L1 loss."""
    model = _base_model(
        prepared, len(supports), r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound
    )
    _add_absolute_residual_constraints(prepared, supports, model)
    return model


def _add_atom_product_constraints(
    rows: _ConstraintBuilder,
    pairs: tuple[tuple[int, int], ...],
    *,
    z: slice,
    y: slice,
    products: tuple[tuple[slice, slice, float], ...],
) -> None:
    """Linearize y_ij = z_i*z_j and each bounded weight times y_ij."""
    for pair_index, (i, j) in enumerate(pairs):
        y_index = y.start + pair_index
        z_i = z.start + i
        z_j = z.start + j
        # Binary z fixes y, so y itself can remain continuous.
        if i == j:
            rows.add({y_index: 1.0, z_i: -1.0}, lower=0.0, upper=0.0)
        else:
            rows.add({y_index: 1.0, z_i: -1.0}, upper=0.0)
            rows.add({y_index: 1.0, z_j: -1.0}, upper=0.0)
            rows.add({y_index: 1.0, z_i: -1.0, z_j: -1.0}, lower=-1.0)

        # Keep R then X within each pair, preserving the solver row order.
        for product, weight, upper_bound in products:
            product_index = product.start + pair_index
            rows.add({product_index: 1.0, y_index: -upper_bound}, upper=0.0)
            rows.add({product_index: 1.0, weight.start: -1.0}, upper=0.0)
            rows.add(
                {product_index: 1.0, weight.start: -1.0, y_index: -upper_bound},
                lower=-upper_bound,
            )


def _add_laminar_extension_constraints(
    rows: _ConstraintBuilder,
    n: int,
    supports: tuple[IndexSupport, ...],
    z: slice,
    relations: slice,
) -> None:
    """Require a new, nested-or-disjoint support relative to every old atom."""
    for atom_index, support in enumerate(supports):
        support_set = set(support)
        subset_relation = relations.start + 3 * atom_index
        superset_relation = subset_relation + 1
        disjoint_relation = subset_relation + 2
        rows.add(
            {subset_relation: 1.0, superset_relation: 1.0, disjoint_relation: 1.0},
            lower=1.0,
            upper=1.0,
        )
        for terminal in range(n):
            z_index = z.start + terminal
            if terminal not in support_set:
                rows.add({z_index: 1.0, subset_relation: 1.0}, upper=1.0)
            else:
                rows.add({superset_relation: 1.0, z_index: -1.0}, upper=0.0)
                rows.add({z_index: 1.0, disjoint_relation: 1.0}, upper=1.0)

        # Hamming distance from each old support must be at least one.
        duplicate_row = {
            z.start + terminal: -1.0 if terminal in support_set else 1.0
            for terminal in range(n)
        }
        rows.add(duplicate_row, lower=1.0 - len(support_set))


def _base_extension_model(
    prepared: _PreparedScenarios,
    supports: tuple[IndexSupport, ...],
    *,
    r_upper_bound: float,
    x_upper_bound: float,
) -> tuple[L1Model, _ExtensionBlocks]:
    """Allocate the original extension variables without adding constraints."""
    pairs = tuple((i, j) for i in range(prepared.n) for j in range(i, prepared.n))
    pair_lookup = {pair: index for index, pair in enumerate(pairs)}
    model = _base_model(
        prepared, len(supports), r_upper_bound=r_upper_bound,
        x_upper_bound=x_upper_bound, prefix="old_",
    )
    variables = model.variables
    model.new_r = variables.add("new_r", 1, upper=r_upper_bound)
    model.new_x = variables.add("new_x", 1, upper=x_upper_bound)
    model.z = variables.add("z", prepared.n, upper=1.0, integral=True)
    y = variables.add("y", len(pairs), upper=1.0)
    u_r = variables.add("u_r", len(pairs), upper=r_upper_bound)
    u_x = variables.add("u_x", len(pairs), upper=x_upper_bound)
    relations = variables.add("relations", 3 * len(supports), upper=1.0, integral=True)
    return model, _ExtensionBlocks(pairs, pair_lookup, y, u_r, u_x, relations)


def _add_extension_constraints(
    prepared: _PreparedScenarios,
    supports: tuple[IndexSupport, ...],
    model: L1Model,
    blocks: _ExtensionBlocks,
    *,
    r_upper_bound: float,
    x_upper_bound: float,
) -> None:
    """Add the original nonempty-support, product and laminar rows in order."""
    rows = model.rows
    rows.add({model.z.start + i: 1.0 for i in range(prepared.n)}, lower=1.0)
    _add_atom_product_constraints(
        rows,
        blocks.pairs,
        z=model.z,
        y=blocks.y,
        products=((blocks.u_r, model.new_r, r_upper_bound), (blocks.u_x, model.new_x, x_upper_bound)),
    )
    _add_laminar_extension_constraints(rows, prepared.n, supports, model.z, blocks.relations)


def build_extension_model(
    prepared: _PreparedScenarios,
    supports: tuple[IndexSupport, ...],
    *,
    r_upper_bound: float,
    x_upper_bound: float,
) -> L1Model:
    """Search a new zz.T atom while jointly refitting all accepted weights."""
    model, blocks = _base_extension_model(
        prepared, supports, r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound,
    )
    _add_absolute_residual_constraints(
        prepared, supports, model,
        new_u_r_block=blocks.u_r,
        new_u_x_block=blocks.u_x,
        pair_lookup=blocks.pair_lookup,
    )
    _add_extension_constraints(
        prepared, supports, model, blocks,
        r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound,
    )
    return model


# Solver dispatch, fixed-support LP and one-support MILP.


def _diagnostics(result, elapsed: float, variables: _VariableBuilder, rows: _ConstraintBuilder) -> SolverDiagnostics:
    def optional(name, cast=float):
        value = getattr(result, name, None)
        return None if value is None else cast(value)

    return SolverDiagnostics(
        status=int(result.status),
        success=bool(result.success),
        message=str(result.message),
        objective=optional("fun"),
        dual_bound=optional("mip_dual_bound"),
        mip_gap=optional("mip_gap"),
        node_count=optional("mip_node_count", int),
        runtime_seconds=float(elapsed),
        variable_count=variables.size,
        binary_variable_count=int(np.count_nonzero(variables.integrality)),
        constraint_count=rows.size,
        solver=getattr(result, "solver", "highs"),
        raw_status=getattr(result, "raw_status", None),
    )


def _run_milp(
    variables: _VariableBuilder,
    rows: _ConstraintBuilder,
    *,
    time_limit: float | None,
    mip_rel_gap: float,
    presolve: bool,
    disp: bool,
    solver: str = "highs",
):
    """Build one standard-form input, dispatch it, then collect diagnostics."""
    _validate_solver(solver)
    solve = milp
    if solver == "gurobi":
        from .gurobi_milp import solve_gurobi_milp as solve

    options: dict[str, float | bool | int] = {
        "mip_rel_gap": float(mip_rel_gap),
        # SciPy forwards this native HiGHS option even though some SciPy
        # releases do not list it in ``milp``'s public option schema.
        "mip_abs_gap": _EXACT_MIP_ABS_GAP_TOLERANCE,
        "presolve": bool(presolve),
        "disp": bool(disp),
    }
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    started = perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Unrecognized options detected: .*mip_abs_gap.*",
            category=RuntimeWarning,
        )
        result = solve(
            c=np.asarray(variables.objective, dtype=float),
            integrality=np.asarray(variables.integrality, dtype=np.uint8),
            bounds=Bounds(
                np.asarray(variables.lower, dtype=float),
                np.asarray(variables.upper, dtype=float),
            ),
            constraints=rows.build(variables.size),
            options=options,
        )
    elapsed = perf_counter() - started
    return result, _diagnostics(result, elapsed, variables, rows)


def _bounded_weights(values: np.ndarray, upper_bound: float) -> np.ndarray:
    """Validate solver bounds, then remove only tolerated boundary roundoff."""
    values = np.asarray(values, dtype=float)
    if (not np.isfinite(values).all()
            or np.any(values < -_COEFFICIENT_BOUND_TOLERANCE)
            or np.any(values > upper_bound + _COEFFICIENT_BOUND_TOLERANCE)):
        raise RuntimeError("solver atom coefficients violate their finite nonnegative bounds")
    return np.clip(values, 0.0, upper_bound)


def _solve_fixed_prepared(
    prepared: _PreparedScenarios,
    supports: tuple[IndexSupport, ...],
    *,
    r_upper_bound: float,
    x_upper_bound: float,
    time_limit: float | None,
    presolve: bool,
    disp: bool,
    solver: str = "highs",
) -> FixedSupportSolution:
    model = build_fixed_model(
        prepared, supports, r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound
    )
    result, diagnostics = _run_milp(
        model.variables, model.rows, time_limit=time_limit,
        mip_rel_gap=0.0, presolve=presolve, disp=disp, solver=solver,
    )
    if result.x is None or result.fun is None or not solver_diagnostics_prove_optimality(diagnostics):
        raise RuntimeError(
            "fixed-support L1 LP was not proven optimal: "
            f"status={diagnostics.status}, gap={diagnostics.mip_gap}, "
            f"message={diagnostics.message}"
        )
    return FixedSupportSolution(
        supports=supports,
        r_values=_bounded_weights(result.x[model.r], r_upper_bound),
        x_values=_bounded_weights(result.x[model.x], x_upper_bound),
        objective=float(result.fun),
        diagnostics=diagnostics,
    )


def solve_fixed_support_l1(
    scenarios: Sequence[dict],
    supports: Iterable[Iterable[int]],
    *,
    r_upper_bound: float,
    x_upper_bound: float,
    time_limit: float | None = None,
    presolve: bool = True,
    disp: bool = False,
    solver: str = "highs",
) -> FixedSupportSolution:
    """Fit fixed supports using ``solver='highs'`` or ``solver='gurobi'``."""

    _validate_solver(solver)
    r_bound, x_bound, checked_time_limit, _ = _validate_solver_parameters(
        r_upper_bound=r_upper_bound,
        x_upper_bound=x_upper_bound,
        time_limit=time_limit,
    )
    prepared = _prepare_scenarios(scenarios)
    family = normalize_supports(supports, prepared.n)
    return _solve_fixed_prepared(
        prepared, family, r_upper_bound=r_bound, x_upper_bound=x_bound,
        time_limit=checked_time_limit, presolve=presolve, disp=disp, solver=solver,
    )


def _solve_extension_prepared(
    prepared: _PreparedScenarios,
    supports: tuple[IndexSupport, ...],
    *,
    r_upper_bound: float,
    x_upper_bound: float,
    time_limit: float | None,
    mip_rel_gap: float,
    presolve: bool,
    disp: bool,
    solver: str = "highs",
) -> ExtensionSolution:
    model = build_extension_model(
        prepared, supports,
        r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound,
    )
    result, diagnostics = _run_milp(
        model.variables, model.rows, time_limit=time_limit,
        mip_rel_gap=mip_rel_gap, presolve=presolve, disp=disp, solver=solver,
    )
    has_incumbent = (
        result.x is not None and result.fun is not None
        and np.isfinite(result.fun) and np.isfinite(result.x).all()
    )
    if not has_incumbent or not solver_diagnostics_prove_optimality(diagnostics):
        # Keep the objective and bound for the outer no-gain check, but never
        # expose an unproved incumbent as an exact support.
        return ExtensionSolution(
            support=None,
            r_values=np.empty(0),
            x_values=np.empty(0),
            objective=float(result.fun) if has_incumbent else None,
            diagnostics=diagnostics,
        )
    support = _round_and_validate_support(
        np.asarray(result.x[model.z], dtype=float), supports, prepared.n
    )
    r_values = _bounded_weights(np.r_[result.x[model.r], result.x[model.new_r]], r_upper_bound)
    x_values = _bounded_weights(np.r_[result.x[model.x], result.x[model.new_x]], x_upper_bound)
    r_matrix, x_matrix = build_matrices_from_atoms(
        prepared.n, (*supports, support), r_values, x_values,
    )
    objective = float(sum(
        np.abs(y - p @ r_matrix.T - q @ x_matrix.T).sum()
        for p, q, y in zip(prepared.p, prepared.q, prepared.target, strict=True)
    ) / prepared.observation_count)
    if not _objectives_agree(objective, float(result.fun)):
        raise RuntimeError(
            "physical prediction does not reproduce the optimal MILP objective: "
            f"milp={result.fun:.12g}, physical={objective:.12g}"
        )
    return ExtensionSolution(
        support=support, r_values=r_values, x_values=x_values,
        objective=objective,
        diagnostics=diagnostics,
    )


def solve_best_laminar_extension_l1(
    scenarios: Sequence[dict],
    supports: Iterable[Iterable[int]],
    *,
    r_upper_bound: float,
    x_upper_bound: float,
    time_limit: float | None = None,
    mip_rel_gap: float = 0.0,
    presolve: bool = True,
    disp: bool = False,
    solver: str = "highs",
) -> ExtensionSolution:
    """Search for the best one-atom L1 extension.

    Binary membership variables search every nonempty new support compatible
    with the existing laminar family.
    ``solver`` selects HiGHS (default) or the optional Gurobi backend.
    """

    _validate_solver(solver)
    r_bound, x_bound, checked_time_limit, checked_gap = _validate_solver_parameters(
        r_upper_bound=r_upper_bound,
        x_upper_bound=x_upper_bound,
        time_limit=time_limit,
        mip_rel_gap=mip_rel_gap,
    )
    prepared = _prepare_scenarios(scenarios)
    family = normalize_supports(supports, prepared.n)
    return _solve_extension_prepared(
        prepared, family,
        r_upper_bound=r_bound, x_upper_bound=x_bound,
        time_limit=checked_time_limit, mip_rel_gap=float(checked_gap),
        presolve=presolve, disp=disp, solver=solver,
    )


# Fully corrective forward path.


def _make_path_point(
    *,
    iteration: int,
    prepared: _PreparedScenarios,
    solution: FixedSupportSolution,
    validation_scenarios: Sequence[dict] | _PreparedScenarios | None,
    validation_blocks: int,
    accepted_gain: float | None,
    solver: SolverDiagnostics,
) -> LaminarL1PathPoint:
    r_matrix, x_matrix = build_matrices_from_atoms(
        prepared.n,
        solution.supports,
        solution.r_values,
        solution.x_values,
    )
    validation_mae: float | None = None
    validation_se: float | None = None
    if validation_scenarios is not None:
        validation_mae, validation_se, _ = evaluate_l1_matrices(
            validation_scenarios,
            r_matrix,
            x_matrix,
            blocks_per_scenario=validation_blocks,
        )
    return LaminarL1PathPoint(
        iteration=iteration,
        support_indices=solution.supports,
        support_labels=tuple(tuple(prepared.labels[i] for i in support) for support in solution.supports),
        r_values=solution.r_values.copy(),
        x_values=solution.x_values.copy(),
        r_matrix=r_matrix,
        x_matrix=x_matrix,
        train_mae=float(solution.objective),
        validation_mae=validation_mae,
        validation_se=validation_se,
        accepted_gain=accepted_gain,
        solver=solver,
    )


def fit_laminar_l1_sensitivity(
    scenarios: Sequence[dict],
    *,
    validation_scenarios: Sequence[dict] | None = None,
    initial_supports: Iterable[Iterable[int]] | None = None,
    max_atoms: int | None = None,
    r_upper_bound: float | None = None,
    x_upper_bound: float | None = None,
    bound_safety_factor: float = 2.0,
    bound_expansion_factor: float = 2.0,
    max_bound_expansions: int = 4,
    bound_active_fraction: float = 0.995,
    improvement_abs_tol: float = 1e-10,
    improvement_rel_tol: float = 1e-8,
    zero_tolerance: float = 1e-9,
    validation_blocks: int = 4,
    time_limit: float | None = None,
    mip_rel_gap: float = 0.0,
    presolve: bool = True,
    disp: bool = False,
    solver: str = "highs",
) -> LaminarL1Result:
    """Fit a forward path of exact bounded one-atom laminar L1 MILPs.

    A non-optimal extension (for example a time limit) is never accepted.
    Each extension searches all nonempty new supports compatible with the
    existing laminar family.
    ``initial_supports`` are fixed topology blocks: their weights are jointly
    refitted, but the supports themselves are retained and constrain every
    later extension.  If validation scenarios are supplied, the returned model
    is the smallest path point within one standard error of the minimum
    validation MAE, starting from that initial family.
    ``solver`` selects ``highs`` or ``gurobi`` for both extension MILPs and
    fixed-support LP refits. It does not change the mathematical model.
    """

    _validate_solver(solver)
    prepared = _prepare_scenarios(scenarios)
    validation_prepared = None
    if validation_scenarios is not None:
        validation_prepared = _prepare_scenarios(validation_scenarios)
        if validation_prepared.labels != prepared.labels:
            raise ValueError("training and validation terminal labels must agree")
    if max_atoms is None:
        max_atoms = 2 * prepared.n - 1
    if not isinstance(max_atoms, (int, np.integer)):
        raise ValueError("max_atoms must be an integer")
    max_atoms = int(max_atoms)
    if max_atoms < 0 or max_atoms > 2 * prepared.n - 1:
        raise ValueError("max_atoms must lie between zero and 2*n-1")
    initial_family = normalize_supports(
        () if initial_supports is None else initial_supports,
        prepared.n,
    )
    if len(initial_family) > max_atoms:
        raise ValueError("initial_supports cannot exceed max_atoms")
    if not np.isfinite(bound_safety_factor) or bound_safety_factor <= 1.0:
        raise ValueError("bound_safety_factor must exceed one")
    if not np.isfinite(bound_active_fraction) or not (
        0.0 < bound_active_fraction < 1.0
    ):
        raise ValueError("bound_active_fraction must lie strictly between zero and one")
    if not np.isfinite(bound_expansion_factor) or bound_expansion_factor <= 1.0:
        raise ValueError("bound_expansion_factor must exceed one")
    if not isinstance(max_bound_expansions, (int, np.integer)) or max_bound_expansions < 0:
        raise ValueError("max_bound_expansions must be a nonnegative integer")
    max_bound_expansions = int(max_bound_expansions)
    improvement_abs_tol = _require_nonnegative_finite(
        "improvement_abs_tol", improvement_abs_tol
    )
    improvement_rel_tol = _require_nonnegative_finite(
        "improvement_rel_tol", improvement_rel_tol
    )
    zero_tolerance = _require_nonnegative_finite("zero_tolerance", zero_tolerance)
    if not isinstance(validation_blocks, (int, np.integer)) or validation_blocks < 1:
        raise ValueError("validation_blocks must be a positive integer")
    validation_blocks = int(validation_blocks)

    if r_upper_bound is None or x_upper_bound is None:
        estimated_r, estimated_x = estimate_atom_upper_bounds(
            prepared, safety_factor=bound_safety_factor
        )
        if r_upper_bound is None:
            r_upper_bound = estimated_r
        if x_upper_bound is None:
            x_upper_bound = estimated_x
    r_bound, x_bound, time_limit, checked_gap = _validate_solver_parameters(
        r_upper_bound=float(r_upper_bound),
        x_upper_bound=float(x_upper_bound),
        time_limit=time_limit,
        mip_rel_gap=mip_rel_gap,
    )
    mip_rel_gap = float(checked_gap)

    attempts: list[ExtensionSolution] = []
    stop_reason = "max_atoms"
    bound_expansions = 0
    supports: tuple[IndexSupport, ...] = initial_family
    frozen_support_count = len(initial_family)

    def refit(family):
        return _solve_fixed_prepared(
            prepared, family, r_upper_bound=r_bound, x_upper_bound=x_bound,
            time_limit=time_limit, presolve=presolve, disp=disp, solver=solver,
        )

    def path_point(solution, iteration, *, gain=None, solver=None):
        return _make_path_point(
            iteration=iteration, prepared=prepared, solution=solution,
            validation_scenarios=validation_prepared, validation_blocks=validation_blocks,
            accepted_gain=gain, solver=solution.diagnostics if solver is None else solver,
        )

    def initialize_path():
        base = refit(initial_family)
        return base, [path_point(base, frozen_support_count)]

    current, path = initialize_path()
    seen_families: set[frozenset[IndexSupport]] = {frozenset(initial_family)}
    iteration = frozen_support_count + 1
    while iteration <= max_atoms:
        gain_tolerance = improvement_abs_tol + improvement_rel_tol * max(
            abs(current.objective), np.finfo(float).eps
        )

        extension = _solve_extension_prepared(
            prepared,
            supports,
            r_upper_bound=r_bound,
            x_upper_bound=x_bound,
            time_limit=time_limit,
            mip_rel_gap=mip_rel_gap,
            presolve=presolve,
            disp=disp,
            solver=solver,
        )
        attempts.append(extension)
        if extension.diagnostics.status == 2:
            stop_reason = "no_feasible_extension"
            break
        if not solver_diagnostics_prove_optimality(extension.diagnostics):
            # current - lower_bound bounds the best possible extension gain.
            lower_bound = extension.diagnostics.dual_bound
            if (
                lower_bound is not None and np.isfinite(lower_bound)
                and current.objective - float(lower_bound) <= gain_tolerance
            ):
                stop_reason = "no_significant_one_atom_gain_certified_by_dual_bound"
            else:
                stop_reason = "extension_not_proven_optimal"
            break
        if extension.support is None or extension.objective is None:
            stop_reason = "extension_without_incumbent"
            break

        extended_supports = (*supports, extension.support)
        # The MILP already jointly optimized every old and new coefficient.
        candidate = FixedSupportSolution(
            supports=extended_supports, r_values=extension.r_values,
            x_values=extension.x_values, objective=extension.objective,
            diagnostics=extension.diagnostics,
        )
        provisional_gain = current.objective - candidate.objective
        if provisional_gain <= gain_tolerance:
            stop_reason = "no_significant_one_atom_gain"
            break

        r_active = _touches_bound(candidate.r_values, r_bound, bound_active_fraction)
        x_active = _touches_bound(candidate.x_values, x_bound, bound_active_fraction)
        if r_active or x_active:
            if bound_expansions >= max_bound_expansions:
                stop_reason = "bound_expansion_limit"
                break
            if r_active:
                r_bound *= bound_expansion_factor
            if x_active:
                x_bound *= bound_expansion_factor
            bound_expansions += 1

            # A larger coefficient domain can alter every earlier greedy
            # support choice.  Restarting is required for a coherent exact
            # forward path under the final bounds.
            supports = initial_family
            current, path = initialize_path()
            seen_families = {frozenset(initial_family)}
            iteration = frozen_support_count + 1
            continue

        # Remove near-zero atoms only when a re-solve preserves the L1 optimum.
        active_indices = list(range(frozen_support_count))
        active_indices.extend(
            index
            for index, (r_value, x_value) in enumerate(
                zip(candidate.r_values, candidate.x_values, strict=True)
            )
            if index >= frozen_support_count
            and (r_value > zero_tolerance or x_value > zero_tolerance)
        )
        if len(active_indices) < len(extended_supports):
            reduced_supports = tuple(extended_supports[index] for index in active_indices)
            reduced = refit(reduced_supports)
            if reduced.objective <= candidate.objective + gain_tolerance:
                extended_supports = reduced_supports
                candidate = reduced

        gain = current.objective - candidate.objective
        if gain <= gain_tolerance:
            stop_reason = "no_significant_one_atom_gain_after_pruning"
            break

        family_signature = frozenset(extended_supports)
        if family_signature in seen_families:
            stop_reason = "repeated_family_after_zero_pruning"
            break
        seen_families.add(family_signature)
        supports = extended_supports
        current = candidate
        path.append(
            path_point(current, iteration, gain=float(gain), solver=extension.diagnostics)
        )
        if len(supports) >= 2 * prepared.n - 1:
            stop_reason = "laminar_family_limit"
            break
        iteration += 1

    selected_index = _choose_path_point(path)
    selected = path[selected_index]
    return LaminarL1Result(
        terminal_labels=prepared.labels,
        support_indices=selected.support_indices,
        support_labels=selected.support_labels,
        r_values=selected.r_values.copy(),
        x_values=selected.x_values.copy(),
        r_matrix=selected.r_matrix.copy(),
        x_matrix=selected.x_matrix.copy(),
        train_mae=selected.train_mae,
        validation_mae=selected.validation_mae,
        selected_path_index=selected_index,
        path=tuple(path),
        attempted_extensions=tuple(attempts),
        stop_reason=stop_reason,
        r_upper_bound=r_bound,
        x_upper_bound=x_bound,
        bound_expansions=bound_expansions,
    )


__all__ = [
    "ExtensionSolution",
    "FixedSupportSolution",
    "LaminarL1PathPoint",
    "LaminarL1Result",
    "SolverDiagnostics",
    "SourceTermProvider",
    "build_matrices_from_atoms",
    "estimate_atom_upper_bounds",
    "evaluate_l1_matrices",
    "fit_laminar_l1_sensitivity",
    "is_admissible_extension",
    "is_laminar_family",
    "normalize_supports",
    "solver_diagnostics_prove_optimality",
    "solve_best_laminar_extension_l1",
    "solve_fixed_support_l1",
]
