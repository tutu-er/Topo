"""Noisy-data sweep for common deltaV regularization.

All simulated measurements include noise. The experiment compares the standard
fast R/X fit with a variant that includes a regularized common residual
``delta_v(t)`` in the squared-voltage-drop equation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import (
    resource_assignment as _resource_assignment,
    selected_scenario_defs as _selected_scenarios,
    simulate_case_scenario as _simulate_case_scenario,
    terminal_buses as _terminal_buses,
)
from terminal_case33.estimation.recipes import apply_preprocessing_recipe
from experiments.run_large_sweep_filter_distance import (
    _distance_candidates,
    _fast_delta_v_regularized_multiscenario_fit,
    _fast_projected_multiscenario_fit,
)
from terminal_case33.graph.terminal_tree import terminal_equivalent_minimum_distance_tree
from terminal_case33.data.paper_style_case_bank import CASE_BUILDERS
from terminal_case33.graph.metrics import edge_precision_recall_f1
from terminal_case33.models.lin_distflow import build_reduced_sensitivity_matrices, impedance_distance_from_reduced_R
from terminal_case33.utils.io import ensure_dir, write_json


def _recipe_grid() -> list[dict]:
    return [
        {"name": "raw_drop", "kind": "raw"},
        {"name": "rolling_highpass_w288", "kind": "rolling_highpass", "window": 288},
    ]


def _simulate_noisy_scenarios(
    *,
    case_key: str,
    scenario_count: int,
    t_count: int,
    pq_noise_rel: float,
    v_noise_rel: float,
    root_voltage_mean: float,
) -> tuple[object, list[int], list[dict], list[tuple[int, int]]]:
    """Simulate noisy AC measurements for one case and data-volume condition."""

    net = CASE_BUILDERS[case_key]()
    assignment = _resource_assignment(case_key, net)
    terminals = _terminal_buses(net)
    R_true, _X_true = build_reduced_sensitivity_matrices(net, terminals, voltage_model="squared-voltage")
    true_mst = terminal_equivalent_minimum_distance_tree(terminals, impedance_distance_from_reduced_R(R_true))
    scenarios = []
    for item in _selected_scenarios(max_scenarios=scenario_count):
        sim = _simulate_case_scenario(
            net=net,
            assignment=assignment,
            T=t_count,
            seed=int(item["seed"]),
            pq_noise_rel=pq_noise_rel,
            v_noise_rel=v_noise_rel,
            root_voltage_mean=root_voltage_mean,
            root_voltage_sigma=float(item["root_voltage_sigma"]),
            profile_scenario=str(item["profile_scenario"]),
        )
        scenarios.append(
            {
                "name": str(item["name"]),
                "V_terminal": sim["V_terminal"],
                "P_terminal": sim["P_terminal"],
                "Q_terminal": sim["Q_terminal"],
                "root_voltage": sim["root_voltage"],
                "drop_target": sim["drop_target"],
            }
        )
    return net, terminals, scenarios, true_mst


def _preprocess_scenarios(scenarios: list[dict], recipe: dict, t_count: int) -> list[dict]:
    fitted = []
    for scenario in scenarios:
        P = apply_preprocessing_recipe(scenario["P_terminal"], recipe, t_count)
        Q = apply_preprocessing_recipe(scenario["Q_terminal"], recipe, t_count)
        D = apply_preprocessing_recipe(scenario["drop_target"], recipe, t_count)
        common = P.index.intersection(Q.index).intersection(D.index)
        fitted.append(
            {
                "name": scenario["name"],
                "V_terminal": scenario["V_terminal"].loc[common],
                "P_terminal": P.loc[common],
                "Q_terminal": Q.loc[common],
                "drop_target": D.loc[common],
            }
        )
    return fitted


def run(
    output_dir: str | Path = "outputs/delta_v_regularization_test",
    cases: list[str] | None = None,
    scenario_counts: list[int] | None = None,
    t_counts: list[int] | None = None,
    delta_penalties: list[float] | None = None,
    recipes: list[str] | None = None,
    distance_modes: list[str] | None = None,
    alphas: list[float] | None = None,
    pq_noise_rel: float = 0.005,
    v_noise_rel: float = 0.0002,
    root_voltage_mean: float = 1.02,
    alpha: float = 0.0,
) -> dict:
    """Run the noisy deltaV regularization sweep."""

    out_dir = ensure_dir(output_dir)
    selected_cases = cases or ["pengwah18"]
    selected_counts = scenario_counts or [1, 2, 3, 5]
    selected_t = t_counts or [96, 240, 480]
    selected_penalties = delta_penalties or [1.0, 10.0, 100.0, 1000.0]
    recipe_lookup = {recipe["name"]: recipe for recipe in _recipe_grid()}
    selected_recipes = [
        recipe_lookup[name]
        for name in (recipes or ["raw_drop", "rolling_highpass_w288"])
    ]
    selected_modes = distance_modes or ["R", "RX_75R_25X"]
    selected_alphas = alphas or [0.0]
    rows = []
    for case_key in selected_cases:
        for scenario_count in selected_counts:
            for t_count in selected_t:
                _net, terminals, scenarios, true_mst = _simulate_noisy_scenarios(
                    case_key=case_key,
                    scenario_count=scenario_count,
                    t_count=t_count,
                    pq_noise_rel=pq_noise_rel,
                    v_noise_rel=v_noise_rel,
                    root_voltage_mean=root_voltage_mean,
                )
                true_set = {tuple(sorted(edge)) for edge in true_mst}
                for recipe in selected_recipes:
                    fitted = _preprocess_scenarios(scenarios, recipe, t_count)
                    fit_results = []
                    for alpha_value in selected_alphas:
                        fit_results.append(
                            (
                                "standard",
                                np.nan,
                                float(alpha_value),
                                *_fast_projected_multiscenario_fit(fitted, alpha=float(alpha_value)),
                                0.0,
                            )
                        )
                        for penalty in selected_penalties:
                            fit_results.append(
                                (
                                    "delta_v_regularized",
                                    float(penalty),
                                    float(alpha_value),
                                    *_fast_delta_v_regularized_multiscenario_fit(
                                        fitted,
                                        alpha=float(alpha_value),
                                        delta_penalty=float(penalty),
                                    ),
                                )
                            )
                    for method, penalty, alpha_value, R_hat, X_hat, r2_score, condition_number, delta_rms in fit_results:
                        distances = _distance_candidates(R_hat, X_hat)
                        for distance_mode in selected_modes:
                            pred = terminal_equivalent_minimum_distance_tree(terminals, distances[distance_mode])
                            score = edge_precision_recall_f1(pred, true_mst)
                            rows.append(
                                {
                                    "case": case_key,
                                    "scenario_count": scenario_count,
                                    "t_count": t_count,
                                    "total_samples": scenario_count * t_count,
                                    "recipe": recipe["name"],
                                    "method": method,
                                    "alpha": alpha_value,
                                    "delta_penalty": penalty,
                                    "distance_mode": distance_mode,
                                    "terminal_mst_f1": float(score["f1"]),
                                    "correct_edges": int(len(set(pred) & true_set)),
                                    "total_edges": int(len(true_mst)),
                                    "r2_score": float(r2_score),
                                    "condition_number": float(condition_number),
                                    "delta_v_rms": float(delta_rms),
                                }
                            )
                print(
                    f"[deltaV] case={case_key} sc={scenario_count} T={t_count} "
                    f"rows={len(rows)}",
                    flush=True,
                )
                pd.DataFrame(rows).to_csv(out_dir / "delta_v_results_partial.csv", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "delta_v_results.csv", index=False)
    best = (
        result.sort_values("terminal_mst_f1", ascending=False)
        .groupby(["case", "scenario_count", "t_count"], as_index=False)
        .head(1)
    )
    best.to_csv(out_dir / "best_by_condition.csv", index=False)
    by_method = (
        result.groupby(["method", "alpha", "delta_penalty"], dropna=False, as_index=False)
        .agg(
            n=("terminal_mst_f1", "size"),
            mean_f1=("terminal_mst_f1", "mean"),
            max_f1=("terminal_mst_f1", "max"),
            high_success_rate=("terminal_mst_f1", lambda values: float((values >= 0.9).mean())),
            exact_success_rate=("terminal_mst_f1", lambda values: float((values >= 0.999999).mean())),
            mean_delta_v_rms=("delta_v_rms", "mean"),
        )
        .sort_values("mean_f1", ascending=False)
    )
    by_method.to_csv(out_dir / "summary_by_method.csv", index=False)
    by_samples = (
        best.groupby("total_samples", as_index=False)
        .agg(
            n=("terminal_mst_f1", "size"),
            mean_best_f1=("terminal_mst_f1", "mean"),
            high_success_rate=("terminal_mst_f1", lambda values: float((values >= 0.9).mean())),
            exact_success_rate=("terminal_mst_f1", lambda values: float((values >= 0.999999).mean())),
        )
        .sort_values("total_samples")
    )
    by_samples.to_csv(out_dir / "best_by_total_samples.csv", index=False)
    metrics = {
        "cases": selected_cases,
        "scenario_counts": selected_counts,
        "t_counts": selected_t,
        "delta_penalties": selected_penalties,
        "alphas": selected_alphas,
        "recipes": [recipe["name"] for recipe in selected_recipes],
        "distance_modes": selected_modes,
        "pq_noise_rel": pq_noise_rel,
        "v_noise_rel": v_noise_rel,
        "row_count": int(len(result)),
        "mean_best_f1": float(best["terminal_mst_f1"].mean()) if len(best) else float("nan"),
        "max_best_f1": float(best["terminal_mst_f1"].max()) if len(best) else float("nan"),
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# DeltaV Regularization Test\n\n"
        "All measurements include noise. The deltaV term is a regularized common residual per time sample.\n\n"
        "## By Method\n\n"
        + by_method.to_string(index=False)
        + "\n\n## Best By Total Samples\n\n"
        + by_samples.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/delta_v_regularization_test")
    parser.add_argument("--cases", nargs="*", choices=list(CASE_BUILDERS), default=None)
    parser.add_argument("--scenario-counts", nargs="*", type=int, default=None)
    parser.add_argument("--T-counts", nargs="*", type=int, default=None)
    parser.add_argument("--delta-penalties", nargs="*", type=float, default=None)
    parser.add_argument("--recipes", nargs="*", default=None)
    parser.add_argument("--distance-modes", nargs="*", default=None)
    parser.add_argument("--alphas", nargs="*", type=float, default=None)
    parser.add_argument("--pq-noise-rel", type=float, default=0.005)
    parser.add_argument("--v-noise-rel", type=float, default=0.0002)
    args = parser.parse_args()
    run(
        output_dir=args.output,
        cases=args.cases,
        scenario_counts=args.scenario_counts,
        t_counts=args.T_counts,
        delta_penalties=args.delta_penalties,
        recipes=args.recipes,
        distance_modes=args.distance_modes,
        alphas=args.alphas,
        pq_noise_rel=args.pq_noise_rel,
        v_noise_rel=args.v_noise_rel,
    )


if __name__ == "__main__":
    main()
