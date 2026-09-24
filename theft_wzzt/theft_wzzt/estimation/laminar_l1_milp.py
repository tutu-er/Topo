"""Exact one-atom laminar L1 sensitivity regression with SciPy/HiGHS.

The estimator represents the reduced sensitivity matrices as

    R = sum_k r_k z_k z_k.T,    X = sum_k x_k z_k z_k.T,

where every ``z_k`` is a binary terminal-clade indicator.  Selected supports
are required to be laminar: two supports are nested or disjoint.  At every
forward step, a MILP searches all admissible nonempty supports and refits all
previous atom weights and scenario-by-terminal intercepts with an L1 loss.

"Exact" means globally optimal for one bounded MILP extension when HiGHS
returns status 0 and the reported primal/dual bound or relative gap satisfies
this module's strict certificate check. The complete forward path is greedy
and is not claimed to be the globally optimal K-atom model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from time import perf_counter
from typing import Hashable, Iterable, Sequence
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array


IndexSupport = tuple[int, ...]

_EXACT_MIP_REL_GAP_TOLERANCE = 1e-10
_EXACT_MIP_ABS_GAP_TOLERANCE = 1e-10
_SUPPORT_INTEGRAL_TOLERANCE = 1e-6
_OBJECTIVE_REPRODUCTION_REL_TOLERANCE = 1e-7


@dataclass(frozen=True)
class SolverDiagnostics:
    """Auditable information returned by one HiGHS LP/MILP solve."""

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

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FixedSupportSolution:
    """L1 optimum for a fixed ordered list of supports."""

    supports: tuple[IndexSupport, ...]
    r_values: np.ndarray
    x_values: np.ndarray
    intercepts: np.ndarray
    objective: float
    diagnostics: SolverDiagnostics


@dataclass(frozen=True)
class ExtensionSolution:
    """Exact best admissible one-support extension of a fixed family."""

    support: IndexSupport | None
    r_values: np.ndarray
    x_values: np.ndarray
    intercepts: np.ndarray
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
    intercepts: np.ndarray
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
    intercepts: np.ndarray
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
            "intercepts": self.intercepts.tolist(),
            "train_mae": self.train_mae,
            "validation_mae": self.validation_mae,
            "selected_path_index": self.selected_path_index,
            "path_length": len(self.path),
            "stop_reason": self.stop_reason,
            "r_upper_bound": self.r_upper_bound,
            "x_upper_bound": self.x_upper_bound,
            "bound_expansions": self.bound_expansions,
        }


@dataclass(frozen=True)
class _PreparedScenarios:
    labels: tuple[Hashable, ...]
    names: tuple[str, ...]
    p: tuple[np.ndarray, ...]
    q: tuple[np.ndarray, ...]
    target: tuple[np.ndarray, ...]

    @property
    def n(self) -> int:
        return len(self.labels)

    @property
    def scenario_count(self) -> int:
        return len(self.p)

    @property
    def observation_count(self) -> int:
        return sum(block.shape[0] * self.n for block in self.p)


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


def solver_diagnostics_prove_optimality(
    diagnostics: SolverDiagnostics,
    *,
    relative_gap_tolerance: float = _EXACT_MIP_REL_GAP_TOLERANCE,
    absolute_gap_tolerance: float = _EXACT_MIP_ABS_GAP_TOLERANCE,
) -> bool:
    """Return whether HiGHS certified the requested numerical optimum."""

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


def _dual_bound_certifies_no_gain(
    current_objective: float,
    diagnostics: SolverDiagnostics,
    gain_tolerance: float,
) -> bool:
    """Use a minimization dual bound to rule out a material extension gain."""

    dual_bound = diagnostics.dual_bound
    return bool(
        dual_bound is not None
        and np.isfinite(dual_bound)
        and current_objective - float(dual_bound) <= gain_tolerance
    )


def _objectives_agree(left: float, right: float) -> bool:
    scale = max(1.0, abs(float(left)), abs(float(right)))
    return bool(
        abs(float(left) - float(right))
        <= _OBJECTIVE_REPRODUCTION_REL_TOLERANCE * scale
    )


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


def _matrix_from_scenario(value, labels: tuple[Hashable, ...], key: str) -> np.ndarray:
    if hasattr(value, "loc") and hasattr(value, "columns"):
        try:
            array = value.loc[:, list(labels)].to_numpy(dtype=float)
        except KeyError as exc:
            raise ValueError(f"{key} columns do not match the first scenario") from exc
    else:
        array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{key} must be a two-dimensional table")
    return array


def _prepare_scenarios(scenarios: Sequence[dict]) -> _PreparedScenarios:
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
    if not labels:
        raise ValueError("at least one terminal is required")

    p_blocks: list[np.ndarray] = []
    q_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    names: list[str] = []
    for scenario_index, scenario in enumerate(scenarios):
        p = _matrix_from_scenario(scenario["P_terminal"], labels, "P_terminal")
        q = _matrix_from_scenario(scenario["Q_terminal"], labels, "Q_terminal")
        target = _matrix_from_scenario(scenario["drop_target"], labels, "drop_target")
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
        names.append(str(scenario.get("name", f"scenario_{scenario_index}")))
    return _PreparedScenarios(
        labels=labels,
        names=tuple(names),
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


def normalize_support_pool(
    supports: Iterable[Iterable[int]], n: int
) -> tuple[IndexSupport, ...]:
    """Canonicalize a finite candidate pool without requiring mutual laminarity.

    Candidate supports may cross each other because the MILP chooses only one
    extension at a time. Each chosen support must still be laminar with the
    already accepted family.
    """

    normalized: list[IndexSupport] = []
    seen: set[IndexSupport] = set()
    for support in supports:
        item = tuple(sorted({int(index) for index in support}))
        if not item:
            raise ValueError("candidate supports must be nonempty")
        if item[0] < 0 or item[-1] >= n:
            raise ValueError("candidate support index is outside the terminal range")
        if item not in seen:
            seen.add(item)
            normalized.append(item)
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


def _fixed_atom_features(
    prepared: _PreparedScenarios, supports: tuple[IndexSupport, ...]
) -> tuple[np.ndarray, np.ndarray]:
    if not supports:
        empty = np.empty((prepared.observation_count, 0), dtype=float)
        return empty, empty.copy()
    p_parts: list[np.ndarray] = []
    q_parts: list[np.ndarray] = []
    for p_block, q_block in zip(prepared.p, prepared.q, strict=True):
        block_p: list[np.ndarray] = []
        block_q: list[np.ndarray] = []
        for support in supports:
            indicator = np.zeros(prepared.n, dtype=float)
            indicator[list(support)] = 1.0
            block_p.append(((p_block @ indicator)[:, None] * indicator[None, :]).reshape(-1))
            block_q.append(((q_block @ indicator)[:, None] * indicator[None, :]).reshape(-1))
        p_parts.append(np.column_stack(block_p))
        q_parts.append(np.column_stack(block_q))
    return np.vstack(p_parts), np.vstack(q_parts)


def _target_and_scenario_output(prepared: _PreparedScenarios) -> tuple[np.ndarray, np.ndarray]:
    target = np.concatenate([block.reshape(-1) for block in prepared.target])
    scenario_output: list[np.ndarray] = []
    for scenario_index, block in enumerate(prepared.target):
        time_count = block.shape[0]
        output = np.tile(np.arange(prepared.n), time_count)
        scenario_output.append(scenario_index * prepared.n + output)
    return target, np.concatenate(scenario_output)


def _diagnostics(result, elapsed: float, variables: _VariableBuilder, rows: _ConstraintBuilder) -> SolverDiagnostics:
    objective = getattr(result, "fun", None)
    return SolverDiagnostics(
        status=int(result.status),
        success=bool(result.success),
        message=str(result.message),
        objective=None if objective is None else float(objective),
        dual_bound=(
            None
            if getattr(result, "mip_dual_bound", None) is None
            else float(result.mip_dual_bound)
        ),
        mip_gap=(None if getattr(result, "mip_gap", None) is None else float(result.mip_gap)),
        node_count=(
            None
            if getattr(result, "mip_node_count", None) is None
            else int(result.mip_node_count)
        ),
        runtime_seconds=float(elapsed),
        variable_count=variables.size,
        binary_variable_count=int(np.count_nonzero(variables.integrality)),
        constraint_count=rows.size,
    )


def _run_milp(
    variables: _VariableBuilder,
    rows: _ConstraintBuilder,
    *,
    time_limit: float | None,
    mip_rel_gap: float,
    presolve: bool,
    disp: bool,
):
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
        result = milp(
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


def _add_absolute_residual_constraints(
    prepared: _PreparedScenarios,
    variables: _VariableBuilder,
    rows: _ConstraintBuilder,
    *,
    r_block: slice,
    x_block: slice,
    intercept_block: slice,
    absolute_block: slice,
    supports: tuple[IndexSupport, ...],
    new_u_r_block: slice | None = None,
    new_u_x_block: slice | None = None,
    pair_lookup: dict[tuple[int, int], int] | None = None,
) -> None:
    target, scenario_output = _target_and_scenario_output(prepared)
    p_features, q_features = _fixed_atom_features(prepared, supports)
    observation = 0
    for scenario_index, (p_block, q_block) in enumerate(
        zip(prepared.p, prepared.q, strict=True)
    ):
        for time_index in range(p_block.shape[0]):
            for output_index in range(prepared.n):
                prediction: dict[int, float] = {
                    intercept_block.start + int(scenario_output[observation]): 1.0
                }
                for atom_index in range(len(supports)):
                    p_value = p_features[observation, atom_index]
                    q_value = q_features[observation, atom_index]
                    if p_value != 0.0:
                        prediction[r_block.start + atom_index] = float(p_value)
                    if q_value != 0.0:
                        prediction[x_block.start + atom_index] = float(q_value)
                if new_u_r_block is not None and new_u_x_block is not None:
                    if pair_lookup is None:
                        raise AssertionError("pair lookup is required for a new atom")
                    for input_index in range(prepared.n):
                        pair = (min(output_index, input_index), max(output_index, input_index))
                        pair_index = pair_lookup[pair]
                        p_value = p_block[time_index, input_index]
                        q_value = q_block[time_index, input_index]
                        if p_value != 0.0:
                            prediction[new_u_r_block.start + pair_index] = float(p_value)
                        if q_value != 0.0:
                            prediction[new_u_x_block.start + pair_index] = float(q_value)

                positive = dict(prediction)
                positive[absolute_block.start + observation] = 1.0
                rows.add(positive, lower=float(target[observation]))

                negative = dict(prediction)
                negative[absolute_block.start + observation] = -1.0
                rows.add(negative, upper=float(target[observation]))
                observation += 1


def _extract_intercepts(
    vector: np.ndarray, block: slice, prepared: _PreparedScenarios
) -> np.ndarray:
    return np.asarray(vector[block], dtype=float).reshape(prepared.scenario_count, prepared.n)


def _solve_fixed_prepared(
    prepared: _PreparedScenarios,
    supports: tuple[IndexSupport, ...],
    *,
    r_upper_bound: float,
    x_upper_bound: float,
    time_limit: float | None,
    presolve: bool,
    disp: bool,
) -> FixedSupportSolution:
    atom_count = len(supports)
    variables = _VariableBuilder()
    r_block = variables.add("r", atom_count, upper=r_upper_bound)
    x_block = variables.add("x", atom_count, upper=x_upper_bound)
    intercept_block = variables.add(
        "intercepts", prepared.scenario_count * prepared.n, lower=-np.inf, upper=np.inf
    )
    absolute_block = variables.add(
        "absolute_residual",
        prepared.observation_count,
        objective=1.0 / prepared.observation_count,
    )
    rows = _ConstraintBuilder()
    _add_absolute_residual_constraints(
        prepared,
        variables,
        rows,
        r_block=r_block,
        x_block=x_block,
        intercept_block=intercept_block,
        absolute_block=absolute_block,
        supports=supports,
    )
    result, diagnostics = _run_milp(
        variables,
        rows,
        time_limit=time_limit,
        mip_rel_gap=0.0,
        presolve=presolve,
        disp=disp,
    )
    if (
        result.x is None
        or result.fun is None
        or not solver_diagnostics_prove_optimality(diagnostics)
    ):
        raise RuntimeError(
            "fixed-support L1 LP was not proven optimal: "
            f"status={diagnostics.status}, gap={diagnostics.mip_gap}, "
            f"message={diagnostics.message}"
        )
    return FixedSupportSolution(
        supports=supports,
        r_values=np.asarray(result.x[r_block], dtype=float),
        x_values=np.asarray(result.x[x_block], dtype=float),
        intercepts=_extract_intercepts(result.x, intercept_block, prepared),
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
) -> FixedSupportSolution:
    """Solve the fully corrective L1 regression for fixed positional supports."""

    r_bound, x_bound, checked_time_limit, _ = _validate_solver_parameters(
        r_upper_bound=r_upper_bound,
        x_upper_bound=x_upper_bound,
        time_limit=time_limit,
    )
    prepared = _prepare_scenarios(scenarios)
    family = normalize_supports(supports, prepared.n)
    return _solve_fixed_prepared(
        prepared,
        family,
        r_upper_bound=r_bound,
        x_upper_bound=x_bound,
        time_limit=checked_time_limit,
        presolve=presolve,
        disp=disp,
    )


def _upper_pairs(n: int) -> tuple[tuple[tuple[int, int], ...], dict[tuple[int, int], int]]:
    pairs = tuple((i, j) for i in range(n) for j in range(i, n))
    return pairs, {pair: index for index, pair in enumerate(pairs)}


def _solve_extension_prepared(
    prepared: _PreparedScenarios,
    supports: tuple[IndexSupport, ...],
    *,
    candidate_supports: tuple[IndexSupport, ...] | None,
    r_upper_bound: float,
    x_upper_bound: float,
    time_limit: float | None,
    mip_rel_gap: float,
    presolve: bool,
    disp: bool,
) -> ExtensionSolution:
    atom_count = len(supports)
    admissible_candidates = (
        None
        if candidate_supports is None
        else tuple(
            support
            for support in candidate_supports
            if is_admissible_extension(support, supports, prepared.n)
        )
    )
    pairs, pair_lookup = _upper_pairs(prepared.n)
    pair_count = len(pairs)
    variables = _VariableBuilder()
    old_r = variables.add("old_r", atom_count, upper=r_upper_bound)
    old_x = variables.add("old_x", atom_count, upper=x_upper_bound)
    intercepts = variables.add(
        "intercepts", prepared.scenario_count * prepared.n, lower=-np.inf, upper=np.inf
    )
    absolute = variables.add(
        "absolute_residual",
        prepared.observation_count,
        objective=1.0 / prepared.observation_count,
    )
    new_r = variables.add("new_r", 1, upper=r_upper_bound)
    new_x = variables.add("new_x", 1, upper=x_upper_bound)
    z_block = variables.add("z", prepared.n, upper=1.0, integral=True)
    candidate_selector = (
        None
        if admissible_candidates is None
        else variables.add(
            "candidate_selector",
            len(admissible_candidates),
            upper=1.0,
            integral=True,
        )
    )
    y_block = variables.add("y", pair_count, upper=1.0)
    u_r_block = variables.add("u_r", pair_count, upper=r_upper_bound)
    u_x_block = variables.add("u_x", pair_count, upper=x_upper_bound)
    relation_block = variables.add("relations", 3 * atom_count, upper=1.0, integral=True)

    rows = _ConstraintBuilder()
    _add_absolute_residual_constraints(
        prepared,
        variables,
        rows,
        r_block=old_r,
        x_block=old_x,
        intercept_block=intercepts,
        absolute_block=absolute,
        supports=supports,
        new_u_r_block=u_r_block,
        new_u_x_block=u_x_block,
        pair_lookup=pair_lookup,
    )

    # Nonempty candidate support.
    rows.add({z_block.start + i: 1.0 for i in range(prepared.n)}, lower=1.0)

    # Optional finite-domain encoding: choose exactly one admissible proposal
    # and make z equal its incidence vector. This retains a genuine MILP and
    # a global optimum certificate relative to the supplied candidate pool.
    if candidate_selector is not None:
        rows.add(
            {
                candidate_selector.start + index: 1.0
                for index in range(len(admissible_candidates))
            },
            lower=1.0,
            upper=1.0,
        )
        for terminal in range(prepared.n):
            row = {z_block.start + terminal: 1.0}
            for index, support in enumerate(admissible_candidates):
                if terminal in support:
                    row[candidate_selector.start + index] = -1.0
            rows.add(row, lower=0.0, upper=0.0)

    # y_ij = z_i z_j.  y can remain continuous because binary z fixes it.
    for pair_index, (i, j) in enumerate(pairs):
        y_index = y_block.start + pair_index
        z_i = z_block.start + i
        z_j = z_block.start + j
        if i == j:
            rows.add({y_index: 1.0, z_i: -1.0}, lower=0.0, upper=0.0)
        else:
            rows.add({y_index: 1.0, z_i: -1.0}, upper=0.0)
            rows.add({y_index: 1.0, z_j: -1.0}, upper=0.0)
            rows.add({y_index: 1.0, z_i: -1.0, z_j: -1.0}, lower=-1.0)

        # Exact bounded products uR = new_r*y and uX = new_x*y.
        u_r = u_r_block.start + pair_index
        rows.add({u_r: 1.0, y_index: -r_upper_bound}, upper=0.0)
        rows.add({u_r: 1.0, new_r.start: -1.0}, upper=0.0)
        rows.add(
            {u_r: 1.0, new_r.start: -1.0, y_index: -r_upper_bound},
            lower=-r_upper_bound,
        )
        u_x = u_x_block.start + pair_index
        rows.add({u_x: 1.0, y_index: -x_upper_bound}, upper=0.0)
        rows.add({u_x: 1.0, new_x.start: -1.0}, upper=0.0)
        rows.add(
            {u_x: 1.0, new_x.start: -1.0, y_index: -x_upper_bound},
            lower=-x_upper_bound,
        )

    # New support must be nested with or disjoint from every old support.
    for atom_index, support in enumerate(supports):
        support_set = set(support)
        subset_relation = relation_block.start + 3 * atom_index
        superset_relation = subset_relation + 1
        disjoint_relation = subset_relation + 2
        rows.add(
            {
                subset_relation: 1.0,
                superset_relation: 1.0,
                disjoint_relation: 1.0,
            },
            lower=1.0,
            upper=1.0,
        )
        for terminal in range(prepared.n):
            z_index = z_block.start + terminal
            if terminal not in support_set:
                rows.add({z_index: 1.0, subset_relation: 1.0}, upper=1.0)
            else:
                rows.add({superset_relation: 1.0, z_index: -1.0}, upper=0.0)
                rows.add({z_index: 1.0, disjoint_relation: 1.0}, upper=1.0)

        # Hamming distance from the old support is at least one.
        duplicate_row: dict[int, float] = {}
        for terminal in range(prepared.n):
            duplicate_row[z_block.start + terminal] = (
                -1.0 if terminal in support_set else 1.0
            )
        rows.add(duplicate_row, lower=1.0 - len(support_set))

    result, diagnostics = _run_milp(
        variables,
        rows,
        time_limit=time_limit,
        mip_rel_gap=mip_rel_gap,
        presolve=presolve,
        disp=disp,
    )
    if (
        result.x is None
        or result.fun is None
        or not np.isfinite(result.fun)
        or not np.isfinite(result.x).all()
    ):
        return ExtensionSolution(
            support=None,
            r_values=np.empty(0),
            x_values=np.empty(0),
            intercepts=np.empty((prepared.scenario_count, prepared.n)),
            objective=None,
            diagnostics=diagnostics,
        )
    if not solver_diagnostics_prove_optimality(diagnostics):
        # A time-limited incumbent is deliberately not exposed as an exact
        # support. This also covers a nominal status-zero result for which the
        # requested strict gap cannot be reconstructed from diagnostics. Its
        # dual bound remains available for a certified no-gain decision in the
        # outer algorithm.
        return ExtensionSolution(
            support=None,
            r_values=np.empty(0),
            x_values=np.empty(0),
            intercepts=np.empty((prepared.scenario_count, prepared.n)),
            objective=float(result.fun),
            diagnostics=diagnostics,
        )
    support = _round_and_validate_support(
        np.asarray(result.x[z_block], dtype=float), supports, prepared.n
    )
    combined_r = np.concatenate(
        [np.asarray(result.x[old_r], dtype=float), [float(result.x[new_r.start])]]
    )
    combined_x = np.concatenate(
        [np.asarray(result.x[old_x], dtype=float), [float(result.x[new_x.start])]]
    )
    return ExtensionSolution(
        support=support,
        r_values=combined_r,
        x_values=combined_x,
        intercepts=_extract_intercepts(result.x, intercepts, prepared),
        objective=float(result.fun),
        diagnostics=diagnostics,
    )


def solve_best_laminar_extension_l1(
    scenarios: Sequence[dict],
    supports: Iterable[Iterable[int]],
    *,
    candidate_supports: Iterable[Iterable[int]] | None = None,
    r_upper_bound: float,
    x_upper_bound: float,
    time_limit: float | None = None,
    mip_rel_gap: float = 0.0,
    presolve: bool = True,
    disp: bool = False,
) -> ExtensionSolution:
    """Search for the best one-atom L1 extension.

    With no candidate pool the search covers every admissible subset.
    Otherwise it is exact over the supplied finite candidate pool.
    """

    r_bound, x_bound, checked_time_limit, checked_gap = _validate_solver_parameters(
        r_upper_bound=r_upper_bound,
        x_upper_bound=x_upper_bound,
        time_limit=time_limit,
        mip_rel_gap=mip_rel_gap,
    )
    prepared = _prepare_scenarios(scenarios)
    family = normalize_supports(supports, prepared.n)
    candidate_pool = (
        None
        if candidate_supports is None
        else normalize_support_pool(candidate_supports, prepared.n)
    )
    return _solve_extension_prepared(
        prepared,
        family,
        candidate_supports=candidate_pool,
        r_upper_bound=r_bound,
        x_upper_bound=x_bound,
        time_limit=checked_time_limit,
        mip_rel_gap=float(checked_gap),
        presolve=presolve,
        disp=disp,
    )


def estimate_atom_upper_bounds(
    scenarios: Sequence[dict], *, safety_factor: float = 2.0, floor: float = 1e-8
) -> tuple[float, float]:
    """Derive data-only atom bounds from the current dense OLS coefficient scale."""

    if not np.isfinite(safety_factor) or safety_factor <= 1.0:
        raise ValueError("safety_factor must exceed one")
    floor = _require_positive_finite("floor", floor)
    prepared = _prepare_scenarios(scenarios)
    design_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for scenario_index, (p_block, q_block, target_block) in enumerate(
        zip(prepared.p, prepared.q, prepared.target, strict=True)
    ):
        indicator = np.zeros((p_block.shape[0], prepared.scenario_count), dtype=float)
        indicator[:, scenario_index] = 1.0
        design_parts.append(np.hstack([p_block, q_block, indicator]))
        target_parts.append(target_block)
    design = np.vstack(design_parts)
    target = np.vstack(target_parts)
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
    scenarios: Sequence[dict],
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    *,
    blocks_per_scenario: int = 4,
    fixed_intercepts: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """Evaluate MAE with fixed intercepts or profiled scenario/output medians.

    Passing ``fixed_intercepts`` is required for genuine held-out prediction.
    The ``None`` mode is retained as an explicitly profiled nuisance-intercept
    diagnostic for callers that provide a separate calibration interpretation.
    """

    prepared = _prepare_scenarios(scenarios)
    if not isinstance(blocks_per_scenario, (int, np.integer)) or blocks_per_scenario < 1:
        raise ValueError("blocks_per_scenario must be a positive integer")
    r_value = np.asarray(r_matrix, dtype=float)
    x_value = np.asarray(x_matrix, dtype=float)
    if r_value.shape != (prepared.n, prepared.n) or x_value.shape != r_value.shape:
        raise ValueError("R/X shape does not match validation terminals")
    intercept_values: np.ndarray | None = None
    if fixed_intercepts is not None:
        intercept_values = np.asarray(fixed_intercepts, dtype=float)
        expected_shape = (prepared.scenario_count, prepared.n)
        if intercept_values.shape != expected_shape:
            raise ValueError(
                f"fixed_intercepts must have shape {expected_shape}"
            )
        if not np.isfinite(intercept_values).all():
            raise ValueError("fixed_intercepts must contain only finite values")
    absolute_parts: list[np.ndarray] = []
    block_means: list[float] = []
    for scenario_index, (p_block, q_block, target_block) in enumerate(
        zip(prepared.p, prepared.q, prepared.target, strict=True)
    ):
        raw_residual = target_block - p_block @ r_value.T - q_block @ x_value.T
        intercept = (
            np.median(raw_residual, axis=0, keepdims=True)
            if intercept_values is None
            else intercept_values[scenario_index][None, :]
        )
        absolute = np.abs(raw_residual - intercept)
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


def _labels_for_supports(
    supports: tuple[IndexSupport, ...], labels: tuple[Hashable, ...]
) -> tuple[tuple[Hashable, ...], ...]:
    return tuple(tuple(labels[index] for index in support) for support in supports)


def _make_path_point(
    *,
    iteration: int,
    prepared: _PreparedScenarios,
    solution: FixedSupportSolution,
    validation_scenarios: Sequence[dict] | None,
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
            fixed_intercepts=solution.intercepts,
        )
    return LaminarL1PathPoint(
        iteration=iteration,
        support_indices=solution.supports,
        support_labels=_labels_for_supports(solution.supports, prepared.labels),
        r_values=solution.r_values.copy(),
        x_values=solution.x_values.copy(),
        intercepts=solution.intercepts.copy(),
        r_matrix=r_matrix,
        x_matrix=x_matrix,
        train_mae=float(solution.objective),
        validation_mae=validation_mae,
        validation_se=validation_se,
        accepted_gain=accepted_gain,
        solver=solver,
    )


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


def fit_laminar_l1_sensitivity(
    scenarios: Sequence[dict],
    *,
    validation_scenarios: Sequence[dict] | None = None,
    initial_supports: Iterable[Iterable[int]] | None = None,
    candidate_supports: Iterable[Iterable[int]] | None = None,
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
) -> LaminarL1Result:
    """Fit a forward path of exact bounded one-atom laminar L1 MILPs.

    A non-optimal extension (for example a time limit) is never accepted.
    If candidate supports are supplied, each forward extension is globally
    optimal over that finite data-derived pool; otherwise it searches all
    admissible subsets.
    ``initial_supports`` are fixed topology blocks: their weights are jointly
    refitted, but the supports themselves are retained and constrain every
    later extension.  If validation scenarios are supplied, the returned model
    is the smallest path point within one standard error of the minimum
    validation MAE, starting from that initial family.
    """

    prepared = _prepare_scenarios(scenarios)
    if validation_scenarios is not None:
        validation_prepared = _prepare_scenarios(validation_scenarios)
        if validation_prepared.labels != prepared.labels:
            raise ValueError("training and validation terminal labels must agree")
        if validation_prepared.names != prepared.names:
            raise ValueError(
                "training and validation must contain the same scenarios in the same order "
                "when training intercepts are reused"
            )
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
    candidate_pool = (
        None
        if candidate_supports is None
        else normalize_support_pool(candidate_supports, prepared.n)
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
            scenarios, safety_factor=bound_safety_factor
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

    def initialize_path() -> tuple[FixedSupportSolution, list[LaminarL1PathPoint]]:
        base = _solve_fixed_prepared(
            prepared,
            initial_family,
            r_upper_bound=r_bound,
            x_upper_bound=x_bound,
            time_limit=time_limit,
            presolve=presolve,
            disp=disp,
        )
        return base, [
            _make_path_point(
                iteration=frozen_support_count,
                prepared=prepared,
                solution=base,
                validation_scenarios=validation_scenarios,
                validation_blocks=validation_blocks,
                accepted_gain=None,
                solver=base.diagnostics,
            )
        ]

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
            candidate_supports=candidate_pool,
            r_upper_bound=r_bound,
            x_upper_bound=x_bound,
            time_limit=time_limit,
            mip_rel_gap=mip_rel_gap,
            presolve=presolve,
            disp=disp,
        )
        attempts.append(extension)
        if extension.diagnostics.status == 2:
            stop_reason = "no_feasible_extension"
            break
        if not solver_diagnostics_prove_optimality(extension.diagnostics):
            if _dual_bound_certifies_no_gain(
                current.objective, extension.diagnostics, gain_tolerance
            ):
                stop_reason = "no_significant_one_atom_gain_certified_by_dual_bound"
            else:
                stop_reason = "extension_not_proven_optimal"
            break
        if extension.support is None or extension.objective is None:
            stop_reason = "extension_without_incumbent"
            break

        candidate_supports = (*supports, extension.support)
        polished = _solve_fixed_prepared(
            prepared,
            candidate_supports,
            r_upper_bound=r_bound,
            x_upper_bound=x_bound,
            time_limit=time_limit,
            presolve=presolve,
            disp=disp,
        )
        if not _objectives_agree(polished.objective, extension.objective):
            raise RuntimeError(
                "fixed-support LP does not reproduce the optimal MILP objective: "
                f"milp={extension.objective:.12g}, fixed={polished.objective:.12g}"
            )
        provisional_gain = current.objective - polished.objective
        if provisional_gain <= gain_tolerance:
            stop_reason = "no_significant_one_atom_gain"
            break

        r_active = _touches_bound(polished.r_values, r_bound, bound_active_fraction)
        x_active = _touches_bound(polished.x_values, x_bound, bound_active_fraction)
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

        # Remove exactly inactive atoms only when a re-solve preserves the L1 optimum.
        active_indices = list(range(frozen_support_count))
        active_indices.extend(
            index
            for index, (r_value, x_value) in enumerate(
                zip(polished.r_values, polished.x_values, strict=True)
            )
            if index >= frozen_support_count
            and (r_value > zero_tolerance or x_value > zero_tolerance)
        )
        if len(active_indices) < len(candidate_supports):
            reduced_supports = tuple(candidate_supports[index] for index in active_indices)
            reduced = _solve_fixed_prepared(
                prepared,
                reduced_supports,
                r_upper_bound=r_bound,
                x_upper_bound=x_bound,
                time_limit=time_limit,
                presolve=presolve,
                disp=disp,
            )
            if reduced.objective <= polished.objective + gain_tolerance:
                candidate_supports = reduced_supports
                polished = reduced

        gain = current.objective - polished.objective
        if gain <= gain_tolerance:
            stop_reason = "no_significant_one_atom_gain_after_polishing"
            break

        family_signature = frozenset(candidate_supports)
        if family_signature in seen_families:
            stop_reason = "repeated_family_after_zero_pruning"
            break
        seen_families.add(family_signature)
        supports = candidate_supports
        current = polished
        path.append(
            _make_path_point(
                iteration=iteration,
                prepared=prepared,
                solution=current,
                validation_scenarios=validation_scenarios,
                validation_blocks=validation_blocks,
                accepted_gain=float(gain),
                solver=extension.diagnostics,
            )
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
        intercepts=selected.intercepts.copy(),
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
    "build_matrices_from_atoms",
    "estimate_atom_upper_bounds",
    "evaluate_l1_matrices",
    "fit_laminar_l1_sensitivity",
    "is_admissible_extension",
    "is_laminar_family",
    "normalize_support_pool",
    "normalize_supports",
    "solver_diagnostics_prove_optimality",
    "solve_best_laminar_extension_l1",
    "solve_fixed_support_l1",
]
