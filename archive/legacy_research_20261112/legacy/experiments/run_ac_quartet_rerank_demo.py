"""Evaluate RNJ candidate AC reranking and quartet filtering on noisy AC cases."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.common import simulate_case
from terminal_case33.estimation.multiscenario import fit_projected_sensitivity
from terminal_case33.graph.quartet_significance import bootstrap_quartet_significance
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.pipeline.ac_likelihood import rank_candidate_trees_ac
from terminal_case33.pipeline.rnj_candidates import generate_rnj_candidates
from terminal_case33.utils.io import ensure_dir, write_json


def _f1(prediction: set[frozenset[int]], truth: set[frozenset[int]]) -> float:
    matched = len(prediction & truth)
    precision = matched / len(prediction) if prediction else float(not truth)
    recall = matched / len(truth) if truth else float(not prediction)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def run(
    output_dir: str | Path,
    cases: list[str],
    scenario_count: int,
    sample_count: int,
    quartet_replicates: int,
) -> dict:
    output = ensure_dir(output_dir)
    rows = []
    evidence_rows = []
    for case in cases:
        net, scenarios = simulate_case(
            case,
            scenario_count,
            sample_count,
            pq_noise_rel=0.005,
            root_voltage_mean=1.02,
        )
        terminals = [int(column) for column in scenarios[0]["P_terminal"].columns]
        truth = rooted_clades(net.closed_edges(), net.root_bus, terminals)
        training, validation = scenarios[:-1], [scenarios[-1]]
        r_matrix, x_matrix, _r2, _condition = fit_projected_sensitivity(
            training,
            constraint_mode="ordered",
        )
        baseline = generate_rnj_candidates(
            r_matrix,
            x_matrix,
            terminals,
            net.root_bus,
            distance_modes=("RX_75R_25X",),
            tolerance_factors=(0.24,),
        )[0]
        candidates = generate_rnj_candidates(r_matrix, x_matrix, terminals, net.root_bus)
        reranked = rank_candidate_trees_ac(
            candidates,
            training,
            validation,
            net,
            pq_noise_relative_std=0.005,
            voltage_noise_relative_std=0.0002,
        )
        selected_clades = set(reranked.selected.candidate.clades)
        quartet_clades, evidence = bootstrap_quartet_significance(
            training,
            selected_clades,
            net.root_bus,
            replicates=quartet_replicates,
            confidence_level=0.90,
            minimum_positive_support=0.80,
            minimum_effect=0.002,
            seed=20260714,
        )
        candidate_f1 = [_f1(set(candidate.clades), truth) for candidate in candidates]
        rows.append(
            {
                "case": case,
                "terminal_count": len(terminals),
                "true_clade_count": len(truth),
                "candidate_count": len(candidates),
                "baseline_f1": _f1(set(baseline.clades), truth),
                "ac_selected_f1": _f1(selected_clades, truth),
                "quartet_filtered_f1": _f1(quartet_clades, truth),
                "candidate_oracle_f1": max(candidate_f1),
                "exact_candidate_present": any(set(candidate.clades) == truth for candidate in candidates),
                "selected_distance_mode": reranked.selected.candidate.distance_mode,
                "selected_tolerance_factor": reranked.selected.candidate.tolerance_factor,
                "selected_voltage_rmse": reranked.selected.voltage_rmse,
                "selected_ac_nll": reranked.selected.standardized_negative_log_likelihood,
                "quartet_kept": len(quartet_clades),
                "quartet_input": len(selected_clades),
            }
        )
        for record in evidence.values():
            evidence_rows.append({"case": case, **record.to_dict()})
        print(
            f"[{case}] candidates={len(candidates)} baseline={rows[-1]['baseline_f1']:.4f} "
            f"ac={rows[-1]['ac_selected_f1']:.4f} quartet={rows[-1]['quartet_filtered_f1']:.4f}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(evidence_rows).to_csv(output / "quartet_evidence.csv", index=False)
    summary = {
        "case_count": len(frame),
        "scenario_count": scenario_count,
        "sample_count": sample_count,
        "pq_noise_relative_std": 0.005,
        "voltage_noise_relative_std": 0.0002,
        "quartet_replicates": quartet_replicates,
        "mean_baseline_f1": float(frame["baseline_f1"].mean()),
        "mean_ac_selected_f1": float(frame["ac_selected_f1"].mean()),
        "mean_quartet_filtered_f1": float(frame["quartet_filtered_f1"].mean()),
        "mean_candidate_oracle_f1": float(frame["candidate_oracle_f1"].mean()),
        "exact_candidate_coverage": float(frame["exact_candidate_present"].mean()),
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/ac_quartet_rerank_demo")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["paper15", "soumalas11", "flynn16", "pengwah18"],
    )
    parser.add_argument("--scenario-count", type=int, default=5)
    parser.add_argument("--sample-count", type=int, default=288)
    parser.add_argument("--quartet-replicates", type=int, default=20)
    args = parser.parse_args()
    run(
        args.output,
        args.cases,
        args.scenario_count,
        args.sample_count,
        args.quartet_replicates,
    )


if __name__ == "__main__":
    main()
