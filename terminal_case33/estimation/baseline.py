"""Complete constrained multi-scenario RNJ baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from terminal_case33.estimation.multiscenario import (
    fit_projected_sensitivity,
    preprocess_scenarios,
)
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.graph.rooted_neighbor_joining import (
    RootedTreeResult,
    rooted_neighbor_joining,
)
from terminal_case33.graph.sensitivity_geometry import sensitivity_geometry


@dataclass(frozen=True)
class CompleteBaselineResult:
    """All fitted quantities needed to audit the baseline end to end."""

    terminals: tuple[int, ...]
    root_bus: int
    r_matrix: pd.DataFrame
    x_matrix: pd.DataFrame
    d_r: pd.DataFrame
    d_x: pd.DataFrame
    distance: pd.DataFrame
    root_depths: pd.Series
    delta_v_residuals: dict[str, pd.DataFrame]
    residual_rmse: float
    r2_score: float
    condition_number: float
    objective_value: float
    tree: RootedTreeResult
    rooted_clades: frozenset[frozenset[int]]
    preprocessing: str
    distance_mode: str
    constraint_mode: str
    alpha: float
    tolerance_factor: float


def _preprocessing_recipe(name: str) -> dict:
    recipes = {
        "raw": {"name": "raw", "kind": "raw"},
        "daily_demean": {"name": "daily_demean", "kind": "demean"},
        "difference": {"name": "difference", "kind": "difference"},
    }
    if name not in recipes:
        raise ValueError(f"unknown preprocessing {name!r}; choose one of {sorted(recipes)}")
    return recipes[name]


def _distance_and_depth(
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build distance diagnostics and root depths from fitted R/X."""

    geometry = sensitivity_geometry(r_matrix, x_matrix, mode)
    return geometry.distance, geometry.root_depths, geometry.d_r, geometry.d_x


def fit_complete_rnj_baseline(
    scenarios: list[dict],
    root_bus: int,
    preprocessing: str = "raw",
    distance_mode: str = "RX_75R_25X",
    constraint_mode: str = "ordered",
    alpha: float = 0.0,
    tolerance_factor: float = 0.16,
) -> CompleteBaselineResult:
    """Fit constrained R/X, expose voltage residuals, and reconstruct RNJ.

    The load-positive squared-voltage-drop model is

    ``Y_s = P_s R.T + Q_s X.T + delta_V_s``.

    Y is the squared-voltage drop from the observed root. No terminal offset
    is fitted; explicit temporal preprocessing remains available for research.

    ``delta_V_s`` is not an additional degree of freedom. It is the fitted
    random/model residual and the objective includes
    ``0.5 * sum_s ||delta_V_s||_F^2`` plus optional R/X ridge penalty.
    """

    if not scenarios:
        raise ValueError("at least one scenario is required")
    if tolerance_factor < 0.0:
        raise ValueError("tolerance_factor must be nonnegative")
    terminals = [int(column) for column in scenarios[0]["P_terminal"].columns]
    if len(terminals) < 2:
        raise ValueError("RNJ baseline requires at least two terminals")
    for scenario in scenarios:
        if [int(column) for column in scenario["P_terminal"].columns] != terminals:
            raise ValueError("all scenarios must have identical terminal columns")

    recipe = _preprocessing_recipe(preprocessing)
    prepared = preprocess_scenarios(scenarios, recipe)
    r_matrix, x_matrix, r2_score, condition_number = fit_projected_sensitivity(
        prepared,
        alpha=alpha,
        constraint_mode=constraint_mode,
    )

    residuals: dict[str, pd.DataFrame] = {}
    residual_sum = 0.0
    sample_count = 0
    for index, scenario in enumerate(prepared):
        name = str(scenario.get("name", f"scenario_{index}"))
        p = scenario["P_terminal"].loc[:, terminals].to_numpy(dtype=float)
        q = scenario["Q_terminal"].loc[:, terminals].to_numpy(dtype=float)
        target_frame = scenario["drop_target"].loc[:, terminals]
        target = target_frame.to_numpy(dtype=float)
        physical_prediction = p @ r_matrix.T + q @ x_matrix.T
        residual = target - physical_prediction
        residuals[name] = pd.DataFrame(
            residual,
            index=target_frame.index,
            columns=terminals,
        )
        residual_sum += float(np.sum(residual**2))
        sample_count += residual.size

    objective = 0.5 * residual_sum
    objective += 0.5 * alpha * float(np.sum(r_matrix**2) + np.sum(x_matrix**2))
    residual_rmse = float(np.sqrt(residual_sum / max(sample_count, 1)))

    geometry = sensitivity_geometry(
        r_matrix,
        x_matrix,
        distance_mode,
    )
    distance, root_depths = geometry.distance, geometry.root_depths
    d_r, d_x = geometry.d_r, geometry.d_x
    tolerance = tolerance_factor * max(float(np.median(root_depths)), 1e-12)
    # RNJ ranks the direct normalized R/X shared-path score. ``distance`` is
    # retained for diagnostics and other additive-distance methods.
    tree = rooted_neighbor_joining(
        geometry.shared_paths,
        root_depths,
        terminals,
        int(root_bus),
        tolerance,
    )
    labels = pd.Index(terminals, name="bus_id")
    return CompleteBaselineResult(
        terminals=tuple(terminals),
        root_bus=int(root_bus),
        r_matrix=pd.DataFrame(r_matrix, index=labels, columns=labels),
        x_matrix=pd.DataFrame(x_matrix, index=labels, columns=labels),
        d_r=pd.DataFrame(d_r, index=labels, columns=labels),
        d_x=pd.DataFrame(d_x, index=labels, columns=labels),
        distance=pd.DataFrame(distance, index=labels, columns=labels),
        root_depths=pd.Series(root_depths, index=labels, name="root_depth"),
        delta_v_residuals=residuals,
        residual_rmse=residual_rmse,
        r2_score=float(r2_score),
        condition_number=float(condition_number),
        objective_value=float(objective),
        tree=tree,
        rooted_clades=frozenset(rooted_clades(tree.edges, int(root_bus), terminals)),
        preprocessing=preprocessing,
        distance_mode=distance_mode,
        constraint_mode=constraint_mode,
        alpha=float(alpha),
        tolerance_factor=float(tolerance_factor),
    )
