"""Fix selected RX75-RNJ blocks, then search new laminar L1-MILP supports.

Fit raw squared-voltage drops relative to the observed root, without terminal
offsets. Defaults use three noisy AC scenarios, 96 samples per scenario,
ordered R/X least-squares QP, RX75 score, and RNJ tolerance factor 0.16.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from rnj_wzzt.graph.bootstrap import (
    _boundary_cherries, _moving_block_bootstrap_copy, _select_disjoint,
)
from rnj_wzzt.scenario.simulation import _integer, _simulate_pool, _terminal_buses
from rnj_wzzt.scenario.settings import _finite_number, resolve_scenario_settings
from rnj_wzzt.reporting import (
    _milp_result_row, _rnj_candidate_rows, _rnj_summary_row,
    _truth_nontrivial_clades,
)
from rnj_wzzt.data.paper_style_case_bank import CASE_BUILDERS
from rnj_wzzt.estimation.laminar_l1_milp import (
    _validate_solver, fit_laminar_l1_sensitivity,
)
from rnj_wzzt.estimation.multiscenario import (
    align_scenarios,
    fit_projected_sensitivity,
)
from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
from rnj_wzzt.graph.rooted_neighbor_joining import rooted_neighbor_joining
from rnj_wzzt.graph.sensitivity_geometry import sensitivity_geometry


DEFAULT_OUTPUT = Path("outputs/reference")
DEFAULT_CASES = tuple(CASE_BUILDERS)
SensitivityFit = tuple[np.ndarray, np.ndarray, float, float]


@dataclass(frozen=True)
class _CaseData:
    name: str
    training: list[dict]
    validation: list[dict]
    terminals: list[int]
    root: int
    truth_clades: set[frozenset[int]]
    truth_cherries: set[frozenset[int]]


@dataclass(frozen=True)
class _RnjSelection:
    full_clades: set[frozenset[int]]
    full_cherries: set[frozenset[int]]
    confidence: dict[frozenset[int], float]
    selected: list[frozenset[int]]
    r2_score: float
    condition: float
    elapsed: float


def _validate_run_options(
    *,
    scenario_suite: str,
    samples_per_scenario: int,
    scenario_count: int,
    bootstrap_replicates: int,
    block_length: int,
    confidence_threshold: float,
    maximum_candidate_count: int,
    tolerance_factor: float,
    time_limit: float,
    coefficient_bound: float,
    root_observation: str,
) -> None:
    """Reject invalid orchestration settings before creating output files."""

    _integer(samples_per_scenario, "samples_per_scenario", 1)
    _integer(scenario_count, "scenario_count", 1)
    _integer(bootstrap_replicates, "bootstrap_replicates", 1)
    _integer(block_length, "block_length", 1)
    _integer(maximum_candidate_count, "maximum_candidate_count", 0)
    confidence = _finite_number(confidence_threshold, "confidence_threshold")
    _finite_number(tolerance_factor, "tolerance_factor")
    _finite_number(time_limit, "time_limit", positive=True)
    _finite_number(coefficient_bound, "coefficient_bound", positive=True)
    if confidence > 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if scenario_suite == "legacy" and samples_per_scenario < 96:
        raise ValueError("at least 96 samples per scenario are required")
    if scenario_suite != "legacy" and (
        samples_per_scenario < 4
        or 96 % samples_per_scenario != 0
    ):
        raise ValueError("new scenario suites require 4..96 samples dividing 96")
    if root_observation not in {"exact", "noisy"}:
        raise ValueError("the model requires observed root voltage (exact or noisy)")


def _scenario_manifest_rows(
    *,
    case_name: str,
    split: str,
    scenarios: list[dict],
    impedance_scale: float,
    root_observation: str,
) -> list[dict]:
    """Describe actual scenario inputs without estimator-only details."""

    return [
        {
            "case": case_name,
            "split": split,
            "name": scenario["name"],
            "impedance_scale": float(impedance_scale),
            "root_observation": root_observation,
            "samples": len(scenario["P_terminal"]),
        }
        for scenario in scenarios
    ]


def _rx75_rnj_clades(
    scenarios: list[dict],
    terminals: list[int],
    root: int,
    *,
    tolerance_factor: float,
    sensitivity_fit: SensitivityFit | None = None,
) -> tuple[set[frozenset[int]], float, float]:
    if sensitivity_fit is None:
        sensitivity_fit = fit_projected_sensitivity(scenarios)
    r_matrix, x_matrix, r2_score, condition = sensitivity_fit
    geometry = sensitivity_geometry(r_matrix, x_matrix, "RX_75R_25X")
    scale = max(float(np.median(geometry.root_depths)), 1e-12)
    tree = rooted_neighbor_joining(
        geometry.shared_paths,
        geometry.root_depths,
        terminals,
        root,
        group_tolerance=tolerance_factor * scale,
    )
    return rooted_clades(tree.edges, root, terminals), float(r2_score), float(condition)


def _select_boundary_blocks(
    scenarios: list[dict],
    terminals: list[int],
    root: int,
    *,
    bootstrap_replicates: int,
    block_length: int,
    confidence_threshold: float,
    maximum_candidate_count: int,
    tolerance_factor: float,
    seed: int,
    sensitivity_fit: SensitivityFit | None = None,
) -> tuple[set[frozenset[int]], dict[frozenset[int], float], list[frozenset[int]], float, float]:
    full_clades, r2_score, condition = _rx75_rnj_clades(
        scenarios,
        terminals,
        root,
        tolerance_factor=tolerance_factor,
        sensitivity_fit=sensitivity_fit,
    )
    full_cherries = _boundary_cherries(full_clades)
    counter: Counter = Counter()
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap_replicates):
        sampled = _moving_block_bootstrap_copy(scenarios, rng, block_length)
        sampled_clades, _, _ = _rx75_rnj_clades(
            sampled,
            terminals,
            root,
            tolerance_factor=tolerance_factor,
        )
        counter.update(_boundary_cherries(sampled_clades))
    confidence = {
        clade: counter[clade] / bootstrap_replicates
        for clade in set(counter) | full_cherries
    }
    stable = {
        clade
        for clade in full_cherries
        if confidence[clade] >= confidence_threshold
    }
    selected = _select_disjoint(stable, confidence, maximum_candidate_count)
    return full_clades, confidence, selected, r2_score, condition


def _prepare_case(
    case_name: str,
    *,
    samples_per_scenario: int,
    scenario_count: int,
    training_replicate: int,
    validation_replicate: int,
    pq_noise_rel: float,
    voltage_noise_rel: float,
    scenario_options: dict,
    root_observation: str,
) -> tuple[_CaseData, list[dict]]:
    """Simulate, preprocess, and label one independent train/validation pair."""

    net, training_raw = _simulate_pool(
        case_name,
        samples_per_scenario,
        training_replicate,
        scenario_count,
        pq_noise_rel,
        voltage_noise_rel,
        **scenario_options,
    )
    _, validation_raw = _simulate_pool(
        case_name,
        samples_per_scenario,
        validation_replicate,
        scenario_count,
        pq_noise_rel,
        voltage_noise_rel,
        **scenario_options,
    )
    scenario_rows = []
    for split, raw in (("training", training_raw), ("validation", validation_raw)):
        scenario_rows.extend(
            _scenario_manifest_rows(
                case_name=case_name,
                split=split,
                scenarios=raw,
                impedance_scale=net.metadata["impedance_scale"],
                root_observation=root_observation,
            )
        )

    training = align_scenarios(training_raw)
    validation = align_scenarios(validation_raw)
    terminals = _terminal_buses(net)
    truth_clades = _truth_nontrivial_clades(net, terminals)
    return (
        _CaseData(
            name=case_name,
            training=training,
            validation=validation,
            terminals=terminals,
            root=int(net.root_bus),
            truth_clades=truth_clades,
            truth_cherries=_boundary_cherries(truth_clades),
        ),
        scenario_rows,
    )


def _select_case_rnj(
    case: _CaseData,
    *,
    bootstrap_replicates: int,
    block_length: int,
    confidence_threshold: float,
    maximum_candidate_count: int,
    tolerance_factor: float,
    seed: int,
) -> _RnjSelection:
    """Fit R/X once, then bootstrap and select RNJ boundary blocks."""

    started = perf_counter()
    sensitivity_fit = fit_projected_sensitivity(case.training)
    full_clades, confidence, selected, r2_score, condition = (
        _select_boundary_blocks(
            case.training,
            case.terminals,
            case.root,
            bootstrap_replicates=bootstrap_replicates,
            block_length=block_length,
            confidence_threshold=confidence_threshold,
            maximum_candidate_count=maximum_candidate_count,
            tolerance_factor=tolerance_factor,
            seed=seed,
            sensitivity_fit=sensitivity_fit,
        )
    )
    return _RnjSelection(
        full_clades=full_clades,
        full_cherries=_boundary_cherries(full_clades),
        confidence=confidence,
        selected=selected,
        r2_score=r2_score,
        condition=condition,
        elapsed=perf_counter() - started,
    )


def _leaf_singletons(terminal_count: int) -> list[tuple[int, ...]]:
    """Return the structurally known leaf-edge atoms for a terminal-only tree."""

    if terminal_count < 1:
        raise ValueError("terminal_count must be positive")
    return [(index,) for index in range(terminal_count)]


def _fit_milp_variants(
    case: _CaseData,
    selection: _RnjSelection,
    *,
    run_baseline: bool,
    coefficient_bound: float,
    time_limit: float,
    milp_solver: str = "highs",
) -> Iterator[dict]:
    """Search for new supports and yield one auditable row per variant."""

    position = {label: index for index, label in enumerate(case.terminals)}
    selected_supports = [
        tuple(position[label] for label in sorted(clade))
        for clade in selection.selected
    ]
    leaf_singletons = _leaf_singletons(len(case.terminals))
    # The MILP API treats every initial support as fixed. The selected RNJ
    # boundary blocks are therefore retained while the MILP extends the tree.
    variants = [("rx75_rnj_plus_milp", [*leaf_singletons, *selected_supports])]
    if run_baseline:
        variants.insert(0, ("milp_only", leaf_singletons))

    for initializer, supports in variants:
        started = perf_counter()
        result = fit_laminar_l1_sensitivity(
            case.training,
            validation_scenarios=case.validation,
            initial_supports=supports,
            r_upper_bound=coefficient_bound,
            x_upper_bound=coefficient_bound,
            time_limit=time_limit,
            solver=milp_solver,
        )
        elapsed = perf_counter() - started
        predicted = {
            frozenset(support)
            for support in result.support_labels
            if 1 < len(support) < len(case.terminals)
        }
        yield _milp_result_row(
            case_name=case.name,
            initializer=initializer,
            result=result,
            selected=[] if initializer == "milp_only" else selection.selected,
            supports=supports,
            predicted=predicted,
            truth_clades=case.truth_clades,
            elapsed=elapsed,
        )


def run(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    cases: tuple[str, ...] = DEFAULT_CASES,
    scenario_count: int = 3,
    samples_per_scenario: int = 96,
    training_replicate: int = 0,
    validation_replicate: int = 1,
    pq_noise_rel: float = 0.005,
    voltage_noise_rel: float = 0.0002,
    tolerance_factor: float = 0.16,
    bootstrap_replicates: int = 100,
    block_length: int = 4,
    confidence_threshold: float = 0.75,
    maximum_candidate_count: int = 2,
    selection_only: bool = False,
    run_baseline: bool = False,
    time_limit: float = 1800.0,
    coefficient_bound: float = 2.0,
    scenario_suite: str = "reference",
    root_observation: str | None = None,
    root_meter_noise_rel: float | None = None,
    root_sigma: float | None = None,
    impedance_scale: float | None = None,
    milp_solver: str = "highs",
) -> dict:
    """Run the Python computation API and define defaults for every entrypoint.

    The default reference workflow fits observed-root voltage drops, selects
    RNJ boundary blocks, and extends their supports with MILP. Explicit
    ``selection_only`` and ``run_baseline`` enable diagnostics and comparison.
    """
    _validate_solver(milp_solver)
    scenario_options = dict(
        scenario_suite=scenario_suite, root_observation=root_observation,
        root_meter_noise_rel=root_meter_noise_rel, root_sigma=root_sigma,
        impedance_scale=impedance_scale,
    )
    scenario_settings = resolve_scenario_settings(**scenario_options)
    _validate_run_options(
        scenario_suite=scenario_suite,
        samples_per_scenario=samples_per_scenario,
        scenario_count=scenario_count,
        bootstrap_replicates=bootstrap_replicates,
        block_length=block_length,
        confidence_threshold=confidence_threshold,
        maximum_candidate_count=maximum_candidate_count,
        tolerance_factor=tolerance_factor,
        time_limit=time_limit,
        coefficient_bound=coefficient_bound,
        root_observation=scenario_settings["root_observation"],
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict] = []
    summary_rows: list[dict] = []
    milp_rows: list[dict] = []
    scenario_rows: list[dict] = []

    for case_name in cases:
        case, manifest = _prepare_case(
            case_name,
            samples_per_scenario=samples_per_scenario,
            scenario_count=scenario_count,
            training_replicate=training_replicate,
            validation_replicate=validation_replicate,
            pq_noise_rel=pq_noise_rel,
            voltage_noise_rel=voltage_noise_rel,
            scenario_options=scenario_options,
            root_observation=scenario_settings["root_observation"],
        )
        scenario_rows.extend(manifest)
        selection = _select_case_rnj(
            case,
            bootstrap_replicates=bootstrap_replicates,
            block_length=block_length,
            confidence_threshold=confidence_threshold,
            maximum_candidate_count=maximum_candidate_count,
            tolerance_factor=tolerance_factor,
            seed=20_260_903 + training_replicate,
        )
        summary_rows.append(
            _rnj_summary_row(
                case_name=case_name,
                scenario_count=scenario_count,
                samples_per_scenario=samples_per_scenario,
                r2_score=selection.r2_score,
                condition=selection.condition,
                selection_seconds=selection.elapsed,
                full_clades=selection.full_clades,
                full_cherries=selection.full_cherries,
                selected=selection.selected,
                confidence=selection.confidence,
                truth_clades=case.truth_clades,
                truth_cherries=case.truth_cherries,
            )
        )
        candidate_rows.extend(
            _rnj_candidate_rows(
                case_name=case_name,
                confidence=selection.confidence,
                full_cherries=selection.full_cherries,
                selected=selection.selected,
                truth_cherries=case.truth_cherries,
                confidence_threshold=confidence_threshold,
            )
        )
        pd.DataFrame(summary_rows).to_csv(output / "rnj_summary.csv", index=False)
        pd.DataFrame(candidate_rows).to_csv(output / "rnj_boundary_candidates.csv", index=False)

        if selection_only:
            continue
        for row in _fit_milp_variants(
            case,
            selection,
            run_baseline=run_baseline,
            coefficient_bound=coefficient_bound,
            time_limit=time_limit,
            milp_solver=milp_solver,
        ):
            milp_rows.append(row)
            pd.DataFrame(milp_rows).to_csv(output / "milp_results.csv", index=False)

    payload = {
        "config": {
            "cases": list(cases),
            "scenario_count": scenario_count,
            "samples_per_scenario": samples_per_scenario,
            "training_replicate": training_replicate,
            "validation_replicate": validation_replicate,
            "pq_noise_rel": pq_noise_rel,
            "voltage_noise_rel": voltage_noise_rel,
            "preprocessing": "raw",
            "constraint_mode": "ordered",
            "distance_mode": "RX_75R_25X",
            "tolerance_factor": tolerance_factor,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_kind": "circular_moving_block",
            "block_length": block_length,
            "confidence_threshold": confidence_threshold,
            "maximum_candidate_count": maximum_candidate_count,
            "selection_only": selection_only,
            "run_baseline": run_baseline,
            "support_search_mode": "unrestricted",
            "time_limit_seconds_per_extension": time_limit,
            "milp_solver": milp_solver,
            "coefficient_bound": coefficient_bound,
            "truth_used_for_candidate_generation": False,
            "scenario_settings": scenario_settings,
        },
        "scenarios": scenario_rows,
        "rnj": summary_rows,
        "milp": milp_rows,
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    """Delegate the historical module entrypoint to the shared CLI parser."""
    from rnj_wzzt.cli import main as cli_main

    cli_main(runner=run)


if __name__ == "__main__":
    main()
