"""Run BIC-regularized AC reranking with quartet and edge-effect validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.common import simulate_case
from terminal_case33.estimation.multiscenario import fit_projected_sensitivity
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.pipeline.rnj_candidates import generate_rnj_candidates
from terminal_case33.pipeline.topology_validation import select_and_validate_topology
from terminal_case33.utils.io import ensure_dir, write_json


def _f1(prediction: set[frozenset[int]], truth: set[frozenset[int]]) -> float:
    matched = len(prediction & truth)
    precision = matched / len(prediction) if prediction else float(not truth)
    recall = matched / len(truth) if truth else float(not prediction)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def run(output_dir: str | Path, cases: list[str], sample_count: int) -> dict:
    output = ensure_dir(output_dir)
    rows = []
    edge_rows = []
    for case_index, case in enumerate(cases):
        net, scenarios = simulate_case(case, 5, sample_count, 0.005, 1.02)
        terminals = [int(column) for column in scenarios[0]["P_terminal"].columns]
        truth = rooted_clades(net.closed_edges(), net.root_bus, terminals)
        r_matrix, x_matrix, _r2, _condition = fit_projected_sensitivity(
            scenarios[:-1],
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
        result = select_and_validate_topology(
            scenarios,
            net,
            bic_complexity_weight=1.0,
            quartet_replicates=20,
            minimum_edge_strength_ratio=0.02,
            seed=20260714 + case_index,
        )
        candidate_oracle = max(
            _f1(set(score.candidate.clades), truth)
            for score in result.ac_rerank.ranked
        )
        rows.append(
            {
                "case": case,
                "baseline_f1": _f1(set(baseline.clades), truth),
                "ac_bic_f1": _f1(set(result.raw_clades), truth),
                "quartet_only_f1": _f1(set(result.quartet_clades), truth),
                "validated_f1": _f1(set(result.validated_clades), truth),
                "candidate_oracle_f1": candidate_oracle,
                "candidate_count": result.ac_rerank.candidate_count,
                "raw_clade_count": len(result.raw_clades),
                "validated_clade_count": len(result.validated_clades),
                "selected_mode": result.ac_rerank.selected.candidate.distance_mode,
                "selected_tolerance": result.ac_rerank.selected.candidate.tolerance_factor,
                "voltage_rmse": result.ac_rerank.selected.voltage_rmse,
                "edge_strength_threshold": result.edge_strength_threshold,
            }
        )
        for clade, evidence in result.quartet_evidence.items():
            edge_rows.append(
                {
                    "case": case,
                    **evidence.to_dict(),
                    "edge_strength": result.clade_edge_strength.get(clade, 0.0),
                    "kept": clade in result.validated_clades,
                }
            )
        print(
            f"[{case}] baseline={rows[-1]['baseline_f1']:.4f} "
            f"ac_bic={rows[-1]['ac_bic_f1']:.4f} "
            f"validated={rows[-1]['validated_f1']:.4f}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(output / "edge_evidence.csv", index=False)
    summary = {
        "case_count": len(frame),
        "sample_count": sample_count,
        "scenario_count": 5,
        "mean_baseline_f1": float(frame["baseline_f1"].mean()),
        "mean_ac_bic_f1": float(frame["ac_bic_f1"].mean()),
        "mean_quartet_only_f1": float(frame["quartet_only_f1"].mean()),
        "mean_validated_f1": float(frame["validated_f1"].mean()),
        "mean_candidate_oracle_f1": float(frame["candidate_oracle_f1"].mean()),
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/validated_topology_demo")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["paper15", "soumalas11", "flynn16", "pengwah18"],
    )
    parser.add_argument("--sample-count", type=int, default=288)
    args = parser.parse_args()
    run(args.output, args.cases, args.sample_count)


if __name__ == "__main__":
    main()
