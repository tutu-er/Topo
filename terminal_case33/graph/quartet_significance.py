"""Bootstrap significance tests for rooted latent-tree clades."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np

from terminal_case33.estimation.multiscenario import (
    fit_projected_sensitivity,
    preprocess_scenarios,
)
from terminal_case33.models.lin_distflow import impedance_distance_from_reduced_R


@dataclass(frozen=True)
class QuartetCladeEvidence:
    """Four-point evidence for one rooted terminal clade."""

    clade: frozenset[int]
    estimate: float
    lower_confidence_bound: float
    upper_confidence_bound: float
    positive_support: float
    quartet_count: int
    replicate_count: int
    accepted: bool

    def to_dict(self) -> dict:
        """Return a JSON-serializable record."""

        record = asdict(self)
        record["clade"] = sorted(self.clade)
        return record


def _distance_and_depth(
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    distance_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    d_r = impedance_distance_from_reduced_R(r_matrix)
    d_x = impedance_distance_from_reduced_R(x_matrix)

    def scale(value: np.ndarray) -> float:
        positive = value[value > 0.0]
        return max(float(np.mean(positive)) if positive.size else 1.0, 1e-12)

    if distance_mode == "R":
        return d_r, np.diag(r_matrix)
    if distance_mode == "X":
        return d_x, np.diag(x_matrix)
    weights = {
        "RX_equal_normalized": (0.5, 0.5),
        "RX_75R_25X": (0.75, 0.25),
        "RX_25R_75X": (0.25, 0.75),
    }
    if distance_mode not in weights:
        raise ValueError(f"unknown distance mode {distance_mode!r}")
    weight_r, weight_x = weights[distance_mode]
    scale_r, scale_x = scale(d_r), scale(d_x)
    return (
        weight_r * d_r / scale_r + weight_x * d_x / scale_x,
        weight_r * np.diag(r_matrix) / scale_r
        + weight_x * np.diag(x_matrix) / scale_x,
    )


def _normalized_augmented_distance(
    distance: np.ndarray,
    root_depths: np.ndarray,
) -> np.ndarray:
    """Append the observed root and normalize one distance replicate."""

    value = np.asarray(distance, dtype=float)
    depths = np.asarray(root_depths, dtype=float)
    if value.shape != (len(depths), len(depths)):
        raise ValueError("distance and root_depths shapes do not match")
    value = np.maximum(0.0, 0.5 * (value + value.T))
    np.fill_diagonal(value, 0.0)
    positive = value[np.triu_indices(len(value), 1)]
    positive = positive[positive > 0.0]
    scale = max(float(np.median(positive)) if positive.size else 1.0, 1e-12)
    augmented = np.zeros((len(value) + 1, len(value) + 1), dtype=float)
    augmented[:-1, :-1] = value / scale
    augmented[-1, :-1] = np.maximum(depths, 0.0) / scale
    augmented[:-1, -1] = augmented[-1, :-1]
    return augmented


def quartet_gaps_for_clade(
    distance: np.ndarray,
    root_depths: np.ndarray,
    terminals: list[int],
    clade: frozenset[int],
) -> np.ndarray:
    """Return four-point gaps supporting ``clade | complement``.

    The known feeder root is added to the complement as an observed anchor.
    For an additive tree and a valid split, both cross sums exceed the within
    sum. The returned gap is the smaller cross sum minus the within sum.
    """

    position = {int(node): index for index, node in enumerate(terminals)}
    if not clade <= set(position):
        raise ValueError("clade contains nodes outside terminals")
    inside = [position[node] for node in sorted(clade)]
    outside = [position[node] for node in terminals if node not in clade]
    outside.append(len(terminals))
    if len(inside) < 2 or len(outside) < 2:
        return np.empty(0, dtype=float)
    augmented = _normalized_augmented_distance(distance, root_depths)
    gaps = []
    for left_a, left_b in combinations(inside, 2):
        for right_a, right_b in combinations(outside, 2):
            within = augmented[left_a, left_b] + augmented[right_a, right_b]
            cross_one = augmented[left_a, right_a] + augmented[left_b, right_b]
            cross_two = augmented[left_a, right_b] + augmented[left_b, right_a]
            gaps.append(min(cross_one, cross_two) - within)
    return np.asarray(gaps, dtype=float)


def evaluate_quartet_clades(
    candidate_clades: set[frozenset[int]],
    terminals: list[int],
    distance_samples: list[np.ndarray],
    depth_samples: list[np.ndarray],
    confidence_level: float = 0.90,
    minimum_positive_support: float = 0.80,
    minimum_effect: float = 0.002,
    within_replicate_quantile: float = 0.25,
) -> tuple[set[frozenset[int]], dict[frozenset[int], QuartetCladeEvidence]]:
    """Filter clades using bootstrap lower bounds on four-point gaps."""

    if len(distance_samples) != len(depth_samples) or not distance_samples:
        raise ValueError("distance_samples and depth_samples must be nonempty and aligned")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if not 0.0 <= minimum_positive_support <= 1.0:
        raise ValueError("minimum_positive_support must lie in [0, 1]")
    if minimum_effect < 0.0:
        raise ValueError("minimum_effect must be nonnegative")
    if not 0.0 <= within_replicate_quantile <= 1.0:
        raise ValueError("within_replicate_quantile must lie in [0, 1]")

    accepted: set[frozenset[int]] = set()
    evidence: dict[frozenset[int], QuartetCladeEvidence] = {}
    alpha = 1.0 - confidence_level
    for clade in sorted(candidate_clades, key=lambda value: (len(value), tuple(sorted(value)))):
        replicate_statistics = []
        quartet_count = 0
        for distance, depths in zip(distance_samples, depth_samples, strict=True):
            gaps = quartet_gaps_for_clade(distance, depths, terminals, clade)
            quartet_count = max(quartet_count, len(gaps))
            if gaps.size:
                replicate_statistics.append(
                    float(np.quantile(gaps, within_replicate_quantile))
                )
        statistics = np.asarray(replicate_statistics, dtype=float)
        if statistics.size:
            estimate = float(np.median(statistics))
            lower = float(np.quantile(statistics, alpha))
            upper = float(np.quantile(statistics, confidence_level))
            support = float(np.mean(statistics > minimum_effect))
        else:
            estimate = lower = upper = float("nan")
            support = 0.0
        is_accepted = bool(
            statistics.size
            and np.isfinite(lower)
            and lower > minimum_effect
            and support >= minimum_positive_support
        )
        record = QuartetCladeEvidence(
            clade=clade,
            estimate=estimate,
            lower_confidence_bound=lower,
            upper_confidence_bound=upper,
            positive_support=support,
            quartet_count=quartet_count,
            replicate_count=int(statistics.size),
            accepted=is_accepted,
        )
        evidence[clade] = record
        if is_accepted:
            accepted.add(clade)
    return accepted, evidence


def bootstrap_quartet_significance(
    scenarios: list[dict],
    candidate_clades: set[frozenset[int]],
    root_bus: int,
    distance_mode: str = "RX_75R_25X",
    replicates: int = 30,
    block_fraction: float = 0.70,
    confidence_level: float = 0.90,
    minimum_positive_support: float = 0.80,
    minimum_effect: float = 0.002,
    within_replicate_quantile: float = 0.25,
    preprocess_recipe: dict | None = None,
    seed: int = 0,
) -> tuple[set[frozenset[int]], dict[frozenset[int], QuartetCladeEvidence]]:
    """Fit block-subsampled distance matrices and test RNJ clades.

    ``preprocess_recipe`` lets quartet validation use the same preprocessing
    recipe as the topology proposal stage. With ``None``, no filtering is used.
    """

    del root_bus
    if not scenarios:
        raise ValueError("at least one scenario is required")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0.5 <= block_fraction <= 1.0:
        raise ValueError("block_fraction must lie in [0.5, 1]")
    prepared = (
        preprocess_scenarios(scenarios, preprocess_recipe)
        if preprocess_recipe is not None
        else scenarios
    )
    terminals = [int(column) for column in prepared[0]["P_terminal"].columns]
    rng = np.random.default_rng(seed)
    distance_samples: list[np.ndarray] = []
    depth_samples: list[np.ndarray] = []
    for _ in range(replicates):
        sampled = []
        for scenario in prepared:
            count = len(scenario["P_terminal"])
            block_size = max(4, min(count, int(round(block_fraction * count))))
            start = int(rng.integers(0, count))
            positions = (start + np.arange(block_size)) % count
            sampled.append(
                {
                    "name": scenario.get("name", "scenario"),
                    "P_terminal": scenario["P_terminal"].iloc[positions],
                    "Q_terminal": scenario["Q_terminal"].iloc[positions],
                    "drop_target": scenario["drop_target"].iloc[positions],
                }
            )
        r_matrix, x_matrix, _r2, _condition = fit_projected_sensitivity(
            sampled,
            constraint_mode="ordered",
        )
        distance, depths = _distance_and_depth(r_matrix, x_matrix, distance_mode)
        distance_samples.append(distance)
        depth_samples.append(depths)
    return evaluate_quartet_clades(
        candidate_clades,
        terminals,
        distance_samples,
        depth_samples,
        confidence_level=confidence_level,
        minimum_positive_support=minimum_positive_support,
        minimum_effect=minimum_effect,
        within_replicate_quantile=within_replicate_quantile,
    )
