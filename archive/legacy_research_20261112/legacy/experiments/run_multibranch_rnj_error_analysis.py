"""RNJ error analysis on multi-branch benchmark and simulated feeders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import resource_assignment, selected_scenario_defs, simulate_case_scenario
from experiments.run_current_method_line_error_analysis import _physical_clade_lines
from experiments.run_latent_tree_diagnostics import _distance_and_depth, _set_score
from experiments.run_rooted_hierarchical_aggregation import RECIPE
from terminal_case33.data.case33_der_assignment import build_case33_der_assignment
from terminal_case33.data.case33bw_raw import load_raw_case33bw
from terminal_case33.data.paper_style_case_bank import CASE_BUILDERS
from terminal_case33.estimation.multiscenario import fit_projected_sensitivity, preprocess_scenarios
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.graph.rooted_neighbor_joining import rooted_neighbor_joining, shared_paths_from_distances
from terminal_case33.scenario.terminalize import terminalize_case33
from terminal_case33.utils.io import ensure_dir, write_json


CASE_KEYS = ("ieee33_hybrid", "paper15", "flynn16", "pengwah18")
CASE_SOURCES = {
    "ieee33_hybrid": "IEEE/MATPOWER case33bw topology; terminal-load-only hybrid transformation",
    "paper15": "multi-branch synthetic LV feeder",
    "flynn16": "balanced multi-branch synthetic LV feeder",
    "pengwah18": "unbalanced multi-lateral synthetic LV feeder",
}


def _members_label(members: set[int] | frozenset[int]) -> str:
    return " ".join(str(node) for node in sorted(members))


def _edge_list(edges) -> list[list[int]]:
    return [[int(edge[0]), int(edge[1])] for edge in edges]


def _build_case(case_key: str):
    if case_key == "ieee33_hybrid":
        raw = load_raw_case33bw(include_tie_lines=False)
        net = terminalize_case33(
            raw,
            mode="hybrid_leaf",
            service_impedance_mode="scaled_original",
            seed=0,
        )
        return net, build_case33_der_assignment(raw)
    net = CASE_BUILDERS[case_key]()
    return net, resource_assignment(case_key, net)


def _simulate_pool(
    case_key: str,
    t_count: int,
    replicate: int,
    maximum_scenarios: int,
    pq_noise_rel: float,
    v_noise_rel: float,
):
    net, assignment = _build_case(case_key)
    scenarios = []
    definitions = selected_scenario_defs(max_scenarios=maximum_scenarios)
    for definition in definitions:
        shifted_seed = int(definition["seed"]) + 100_003 * replicate
        simulation = simulate_case_scenario(
            net=net,
            assignment=assignment,
            T=t_count,
            seed=shifted_seed,
            pq_noise_rel=pq_noise_rel,
            v_noise_rel=v_noise_rel,
            root_voltage_mean=1.02,
            root_voltage_sigma=float(definition["root_voltage_sigma"]),
            profile_scenario=str(definition["profile_scenario"]),
        )
        if not simulation["ac_converged"]:
            raise RuntimeError(f"AC power flow did not converge for {case_key}")
        scenarios.append(
            {
                "name": f"{definition['name']}_rep{replicate}",
                "P_terminal": simulation["P_terminal"],
                "Q_terminal": simulation["Q_terminal"],
                "V_terminal": simulation["V_terminal"],
                "root_voltage": simulation["root_voltage"],
                "drop_target": simulation["drop_target"],
            }
        )
    return net, scenarios


def _extra_relation(extra: frozenset[int], true_clades: set[frozenset[int]]):
    nearest = max(true_clades, key=lambda truth: len(extra & truth) / max(len(extra | truth), 1))
    similarity = len(extra & nearest) / max(len(extra | nearest), 1)
    if extra < nearest:
        relation = "spurious_refinement_inside_true_clade"
    elif extra > nearest:
        relation = "spurious_superset_of_true_clade"
    else:
        relation = "cross_branch_terminal_substitution"
    return nearest, relation, similarity


def run(
    output_dir: str | Path = "outputs/multibranch_rnj_error_analysis_20260902",
    cases: list[str] | None = None,
    scenario_counts: list[int] | None = None,
    t_counts: list[int] | None = None,
    replicates: int = 3,
    pq_noise_rel: float = 0.005,
    v_noise_rel: float = 0.0002,
    tolerance_factor: float = 0.16,
) -> dict:
    out_dir = ensure_dir(output_dir)
    selected_cases = cases or list(CASE_KEYS)
    selected_scenario_counts = scenario_counts or [3, 5]
    selected_t_counts = t_counts or [48, 96, 288]
    maximum_scenarios = max(selected_scenario_counts)
    condition_rows = []
    missing_rows = []
    extra_rows = []
    topology_records = []
    total = len(selected_cases) * len(selected_t_counts) * replicates * len(selected_scenario_counts)
    completed = 0

    for case_key in selected_cases:
        for t_count in selected_t_counts:
            for replicate in range(replicates):
                net, scenario_pool = _simulate_pool(
                    case_key,
                    t_count,
                    replicate,
                    maximum_scenarios,
                    pq_noise_rel,
                    v_noise_rel,
                )
                terminals = net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"].astype(int).tolist()
                true_edges = net.closed_edges()
                true_clades = rooted_clades(true_edges, net.root_bus, terminals)
                physical_lines = _physical_clade_lines(net, terminals)
                for scenario_count in selected_scenario_counts:
                    fitted = preprocess_scenarios(scenario_pool[:scenario_count], RECIPE)
                    r_matrix, x_matrix, r2_score, condition_number = fit_projected_sensitivity(
                        fitted,
                        constraint_mode="ordered",
                    )
                    distance, depth, _ = _distance_and_depth(r_matrix, x_matrix, "RX_75R_25X")
                    scale = max(float(np.median(depth)), 1e-12)
                    tree = rooted_neighbor_joining(
                        shared_paths_from_distances(distance, depth),
                        depth,
                        terminals,
                        net.root_bus,
                        tolerance_factor * scale,
                    )
                    predicted_clades = rooted_clades(tree.edges, net.root_bus, terminals)
                    precision, recall, f1_score = _set_score(predicted_clades, true_clades)
                    missing = true_clades - predicted_clades
                    extra = predicted_clades - true_clades
                    if not missing and not extra:
                        pattern = "exact"
                    elif not missing:
                        pattern = "pure_oversegmentation"
                    elif not extra:
                        pattern = "pure_undersegmentation"
                    else:
                        pattern = "mixed_missing_and_extra"
                    condition_id = f"{case_key}_r{replicate}_s{scenario_count}_t{t_count}"
                    condition_rows.append(
                        {
                            "condition_id": condition_id,
                            "case": case_key,
                            "source": CASE_SOURCES[case_key],
                            "replicate": replicate,
                            "scenario_count": scenario_count,
                            "t_count": t_count,
                            "samples": scenario_count * t_count,
                            "terminal_count": len(terminals),
                            "true_clade_count": len(true_clades),
                            "predicted_clade_count": len(predicted_clades),
                            "clade_precision": precision,
                            "clade_recall": recall,
                            "clade_f1": f1_score,
                            "exact_recovery": predicted_clades == true_clades,
                            "failure_pattern": pattern,
                            "missing_true_clades": len(missing),
                            "extra_predicted_clades": len(extra),
                            "r2_score": r2_score,
                            "condition_number": condition_number,
                        }
                    )
                    for clade in missing:
                        line = physical_lines[clade]
                        missing_rows.append(
                            {
                                "condition_id": condition_id,
                                "case": case_key,
                                "clade_members": _members_label(clade),
                                **line,
                            }
                        )
                    for clade in extra:
                        nearest, relation, similarity = _extra_relation(clade, true_clades)
                        extra_rows.append(
                            {
                                "condition_id": condition_id,
                                "case": case_key,
                                "extra_clade_members": _members_label(clade),
                                "nearest_true_members": _members_label(nearest),
                                "relation": relation,
                                "jaccard_similarity": similarity,
                                "terminal_symmetric_difference": len(clade ^ nearest),
                            }
                        )
                    topology_records.append(
                        {
                            "condition_id": condition_id,
                            "case": case_key,
                            "root": int(net.root_bus),
                            "terminals": terminals,
                            "true_edges": _edge_list(true_edges),
                            "predicted_edges": _edge_list(tree.edges),
                            "true_clades": [sorted(clade) for clade in sorted(true_clades, key=lambda c: (len(c), tuple(sorted(c))))],
                            "predicted_clades": [sorted(clade) for clade in sorted(predicted_clades, key=lambda c: (len(c), tuple(sorted(c))))],
                            "missing_clades": [sorted(clade) for clade in sorted(missing, key=lambda c: (len(c), tuple(sorted(c))))],
                            "extra_clades": [sorted(clade) for clade in sorted(extra, key=lambda c: (len(c), tuple(sorted(c))))],
                        }
                    )
                    completed += 1
                print(f"[multibranch-rnj] {completed}/{total} case={case_key} T={t_count} rep={replicate}", flush=True)

    conditions = pd.DataFrame(condition_rows)
    missing_frame = pd.DataFrame(missing_rows)
    extra_frame = pd.DataFrame(extra_rows)
    conditions.to_csv(out_dir / "conditions.csv", index=False)
    missing_frame.to_csv(out_dir / "missing_true_clades.csv", index=False)
    extra_frame.to_csv(out_dir / "extra_predicted_clades.csv", index=False)
    (out_dir / "topology_records.json").write_text(json.dumps(topology_records, ensure_ascii=False, indent=2), encoding="utf-8")

    by_case = (
        conditions.groupby(["case", "source"], as_index=False)
        .agg(
            runs=("condition_id", "size"),
            failures=("exact_recovery", lambda values: int((~values).sum())),
            exact_rate=("exact_recovery", "mean"),
            mean_f1=("clade_f1", "mean"),
            missing_true_clades=("missing_true_clades", "sum"),
            extra_predicted_clades=("extra_predicted_clades", "sum"),
        )
    )
    by_case.to_csv(out_dir / "summary_by_case.csv", index=False)
    failed = conditions.loc[~conditions["exact_recovery"]]
    failure_patterns = (
        failed.groupby(["case", "failure_pattern"], as_index=False)
        .agg(conditions=("condition_id", "size"))
    )
    failure_patterns.to_csv(out_dir / "failure_patterns.csv", index=False)
    region_summary = (
        missing_frame.groupby(["case", "region"], as_index=False)
        .agg(missing_true_clades=("condition_id", "size"))
        if len(missing_frame)
        else pd.DataFrame(columns=["case", "region", "missing_true_clades"])
    )
    region_summary.to_csv(out_dir / "missing_region_summary.csv", index=False)
    extra_summary = (
        extra_frame.groupby(["case", "relation"], as_index=False)
        .agg(extra_clades=("condition_id", "size"))
        if len(extra_frame)
        else pd.DataFrame(columns=["case", "relation", "extra_clades"])
    )
    extra_summary.to_csv(out_dir / "extra_type_summary.csv", index=False)
    metrics = {
        "method": "ordered_RX75_RNJ",
        "cases": selected_cases,
        "condition_count": int(len(conditions)),
        "failure_count": int(len(failed)),
        "exact_rate": float(conditions["exact_recovery"].mean()),
        "mean_f1": float(conditions["clade_f1"].mean()),
        "summary_by_case": by_case.to_dict(orient="records"),
        "failure_patterns": failure_patterns.to_dict(orient="records"),
        "missing_region_summary": region_summary.to_dict(orient="records"),
        "extra_type_summary": extra_summary.to_dict(orient="records"),
    }
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/multibranch_rnj_error_analysis_20260902")
    parser.add_argument("--cases", nargs="*", choices=CASE_KEYS, default=None)
    parser.add_argument("--scenario-counts", nargs="*", type=int, default=None)
    parser.add_argument("--T-counts", nargs="*", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=3)
    args = parser.parse_args()
    run(
        output_dir=args.output,
        cases=args.cases,
        scenario_counts=args.scenario_counts,
        t_counts=args.T_counts,
        replicates=args.replicates,
    )


if __name__ == "__main__":
    main()
