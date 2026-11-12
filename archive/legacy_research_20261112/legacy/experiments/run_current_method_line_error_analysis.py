"""Locate physical line regions responsible for current RNJ topology errors."""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from experiments.run_latent_tree_diagnostics import _distance_and_depth, _set_score
from experiments.run_matrix_constraint_large_sweep import _simulate_pool, _terminal_buses
from experiments.run_rooted_hierarchical_aggregation import RECIPE
from terminal_case33.data.paper_style_case_bank import CASE_BUILDERS
from terminal_case33.estimation.multiscenario import fit_projected_sensitivity, preprocess_scenarios
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.graph.rooted_neighbor_joining import rooted_neighbor_joining, shared_paths_from_distances
from terminal_case33.utils.io import ensure_dir, write_json


def _members_label(members: set[int] | frozenset[int]) -> str:
    return " ".join(str(node) for node in sorted(members))


def _physical_clade_lines(net, terminals: list[int]) -> dict[frozenset[int], dict]:
    """Map each identifiable rooted terminal clade to its physical feeder edge."""

    oriented = net.orient_from_root()
    children: dict[int, list[int]] = {int(bus): [] for bus in net.buses["bus_id"]}
    parent: dict[int, int] = {}
    edge_data: dict[tuple[int, int], dict] = {}
    for row in oriented.itertuples(index=False):
        upstream, child = int(row.parent), int(row.child)
        children[upstream].append(child)
        parent[child] = upstream
        edge_data[(upstream, child)] = {
            "r_ohm": float(row.r_ohm),
            "x_ohm": float(row.x_ohm),
            "branch_type": str(row.branch_type),
        }
    terminal_set = set(terminals)
    bus_type = net.buses.set_index("bus_id")["bus_type"].astype(str).to_dict()
    descendants: dict[int, set[int]] = {}
    hidden_descendants: dict[int, int] = {}

    def visit(node: int) -> tuple[set[int], int]:
        found = {node} if node in terminal_set else set()
        hidden_below = 0
        for child in children[node]:
            child_terminals, child_hidden = visit(child)
            found.update(child_terminals)
            hidden_below += child_hidden + int(bus_type.get(child) == "hidden_internal")
        descendants[node] = found
        hidden_descendants[node] = hidden_below
        return found, hidden_below

    visit(int(net.root_bus))
    physical_graph = net.to_networkx_graph()
    records: dict[frozenset[int], dict] = {}
    for child, upstream in parent.items():
        members = frozenset(descendants[child])
        if not 1 < len(members) < len(terminals):
            continue
        if upstream == net.root_bus:
            region = "root_adjacent_backbone"
        elif hidden_descendants[child] == 0:
            region = "peripheral_cluster_feeder"
        else:
            region = "intermediate_hidden_backbone"
        records[members] = {
            "parent_bus": upstream,
            "child_bus": child,
            "region": region,
            "root_edge_distance": int(nx.shortest_path_length(physical_graph, net.root_bus, child)),
            "clade_size": len(members),
            "clade_members": _members_label(members),
            **edge_data[(upstream, child)],
        }
    return records


def _terminal_signatures(edges: list[tuple] | tuple[tuple, ...], root: int, terminals: list[int]) -> dict[int, dict]:
    """Return each terminal's sibling neighborhood and rooted clade chain."""

    graph = nx.Graph()
    graph.add_edges_from((int(edge[0]), int(edge[1])) for edge in edges)
    terminal_set = set(terminals)
    clades = rooted_clades(edges, root, terminals)
    result = {}
    for terminal in terminals:
        neighbor = next(iter(graph.neighbors(terminal)))
        siblings = {
            int(node)
            for node in graph.neighbors(neighbor)
            if node in terminal_set and node != terminal
        }
        chain = {clade for clade in clades if terminal in clade}
        result[terminal] = {
            "siblings": siblings,
            "chain": chain,
        }
    return result


def run(
    output_dir: str | Path = "outputs/current_method_line_error_analysis",
    cases: list[str] | None = None,
    scenario_counts: list[int] | None = None,
    t_counts: list[int] | None = None,
    replicates: int = 2,
    pq_noise_rel: float = 0.005,
    v_noise_rel: float = 0.0002,
    tolerance_factor: float = 0.16,
) -> dict:
    """Analyze line-region errors for ordered RX75 RNJ in normal conditions."""

    out_dir = ensure_dir(output_dir)
    selected_cases = cases or list(CASE_BUILDERS)
    selected_scenario_counts = scenario_counts or [3, 5, 10]
    selected_t_counts = t_counts or [48, 96, 288]
    if min(selected_scenario_counts) < 3:
        raise ValueError("this primary analysis requires at least three scenarios")
    maximum_scenarios = max(selected_scenario_counts)
    condition_rows: list[dict] = []
    line_rows: list[dict] = []
    extra_rows: list[dict] = []
    terminal_rows: list[dict] = []
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
                terminals = _terminal_buses(net)
                true_edges = net.closed_edges()
                true_clades = rooted_clades(true_edges, net.root_bus, terminals)
                physical_lines = _physical_clade_lines(net, terminals)
                true_terminal_signatures = _terminal_signatures(true_edges, net.root_bus, terminals)
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
                    condition_id = f"{case_key}_r{replicate}_s{scenario_count}_t{t_count}"
                    condition_rows.append(
                        {
                            "condition_id": condition_id,
                            "case": case_key,
                            "replicate": replicate,
                            "scenario_count": scenario_count,
                            "t_count": t_count,
                            "samples": scenario_count * t_count,
                            "clade_precision": precision,
                            "clade_recall": recall,
                            "clade_f1": f1_score,
                            "exact_recovery": predicted_clades == true_clades,
                            "missing_true_lines": len(missing),
                            "extra_predicted_lines": len(extra),
                            "r2_score": r2_score,
                            "condition_number": condition_number,
                        }
                    )
                    for clade in true_clades:
                        line_rows.append(
                            {
                                "condition_id": condition_id,
                                "case": case_key,
                                "replicate": replicate,
                                "scenario_count": scenario_count,
                                "t_count": t_count,
                                "recovered": clade in predicted_clades,
                                **physical_lines[clade],
                            }
                        )
                    for clade in extra:
                        nearest = max(
                            true_clades,
                            key=lambda truth: len(clade & truth) / max(len(clade | truth), 1),
                        )
                        similarity = len(clade & nearest) / max(len(clade | nearest), 1)
                        if clade < nearest:
                            relation = "spurious_refinement_inside_true_clade"
                        elif clade > nearest:
                            relation = "spurious_superset_of_true_clade"
                        else:
                            relation = "cross_branch_terminal_substitution"
                        extra_rows.append(
                            {
                                "condition_id": condition_id,
                                "case": case_key,
                                "replicate": replicate,
                                "scenario_count": scenario_count,
                                "t_count": t_count,
                                "extra_clade_size": len(clade),
                                "extra_clade_members": _members_label(clade),
                                "nearest_true_members": _members_label(nearest),
                                "nearest_true_region": physical_lines[nearest]["region"],
                                "nearest_true_parent": physical_lines[nearest]["parent_bus"],
                                "nearest_true_child": physical_lines[nearest]["child_bus"],
                                "jaccard_similarity": similarity,
                                "terminal_substitutions": len(clade ^ nearest),
                                "relation": relation,
                            }
                        )
                    predicted_signatures = _terminal_signatures(tree.edges, net.root_bus, terminals)
                    for terminal in terminals:
                        truth = true_terminal_signatures[terminal]
                        predicted = predicted_signatures[terminal]
                        terminal_rows.append(
                            {
                                "condition_id": condition_id,
                                "case": case_key,
                                "replicate": replicate,
                                "scenario_count": scenario_count,
                                "t_count": t_count,
                                "terminal_bus": terminal,
                                "true_siblings": _members_label(truth["siblings"]),
                                "predicted_siblings": _members_label(predicted["siblings"]),
                                "sibling_neighborhood_correct": truth["siblings"] == predicted["siblings"],
                                "rooted_path_signature_correct": truth["chain"] == predicted["chain"],
                            }
                        )
                    completed += 1
                print(
                    f"[line-errors] {completed}/{total} case={case_key} T={t_count} replicate={replicate}",
                    flush=True,
                )

    conditions = pd.DataFrame(condition_rows)
    lines = pd.DataFrame(line_rows)
    extras = pd.DataFrame(extra_rows)
    terminal_status = pd.DataFrame(terminal_rows)
    conditions.to_csv(out_dir / "condition_errors.csv", index=False)
    lines.to_csv(out_dir / "physical_line_recovery.csv", index=False)
    extras.to_csv(out_dir / "extra_predicted_clades.csv", index=False)
    terminal_status.to_csv(out_dir / "terminal_attachment_status.csv", index=False)

    region_summary = (
        lines.groupby("region", as_index=False)
        .agg(
            line_opportunities=("recovered", "size"),
            recovered=("recovered", "sum"),
        )
        .assign(
            missed=lambda frame: frame["line_opportunities"] - frame["recovered"],
            miss_rate=lambda frame: frame["missed"] / frame["line_opportunities"],
        )
        .sort_values("miss_rate", ascending=False)
    )
    region_summary.to_csv(out_dir / "error_region_summary.csv", index=False)
    line_summary = (
        lines.groupby(
            ["case", "parent_bus", "child_bus", "region", "clade_size", "clade_members"],
            as_index=False,
        )
        .agg(opportunities=("recovered", "size"), recovered=("recovered", "sum"))
        .assign(
            misses=lambda frame: frame["opportunities"] - frame["recovered"],
            miss_rate=lambda frame: frame["misses"] / frame["opportunities"],
        )
        .sort_values(["miss_rate", "misses"], ascending=False)
    )
    line_summary.to_csv(out_dir / "physical_line_miss_rates.csv", index=False)
    terminal_summary = (
        terminal_status.groupby("case", as_index=False)
        .agg(
            terminal_opportunities=("terminal_bus", "size"),
            sibling_neighborhood_accuracy=("sibling_neighborhood_correct", "mean"),
            rooted_path_signature_accuracy=("rooted_path_signature_correct", "mean"),
        )
    )
    terminal_summary.to_csv(out_dir / "terminal_attachment_summary.csv", index=False)
    extra_summary = (
        extras.groupby("relation", as_index=False)
        .agg(extra_lines=("condition_id", "size"))
        .sort_values("extra_lines", ascending=False)
    )
    extra_summary.to_csv(out_dir / "extra_line_type_summary.csv", index=False)
    condition_by_case = (
        conditions.assign(error=lambda frame: ~frame["exact_recovery"])
        .groupby("case", as_index=False)
        .agg(
            runs=("error", "size"),
            error_conditions=("error", "sum"),
            error_rate=("error", "mean"),
            mean_clade_f1=("clade_f1", "mean"),
        )
    )
    condition_by_case.to_csv(out_dir / "condition_error_by_case.csv", index=False)
    error_conditions = conditions.loc[~conditions["exact_recovery"]]
    metrics = {
        "method": "ordered_RX75_RNJ",
        "normal_regime": "scenario_count >= 3",
        "condition_count": int(len(conditions)),
        "error_condition_count": int(len(error_conditions)),
        "error_condition_rate": float(len(error_conditions) / len(conditions)),
        "mean_f1_all_normal": float(conditions["clade_f1"].mean()),
        "mean_f1_error_conditions": float(error_conditions["clade_f1"].mean()),
        "missing_true_line_count": int(conditions["missing_true_lines"].sum()),
        "extra_predicted_line_count": int(conditions["extra_predicted_lines"].sum()),
        "region_summary": region_summary.to_dict(orient="records"),
        "extra_line_type_summary": extra_summary.to_dict(orient="records"),
        "terminal_attachment_summary": terminal_summary.to_dict(orient="records"),
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# Current method line-error analysis\n\n"
        "Physical hidden-node labels are not comparable across latent-tree reconstructions. "
        "Each identifiable backbone line is therefore represented by its downstream rooted "
        "terminal clade. Singleton service lines are assessed by terminal sibling neighborhoods "
        "and complete rooted clade-chain signatures.\n\n"
        f"- normal conditions: {len(conditions)}\n"
        f"- conditions with any topology error: {len(error_conditions)}\n"
        f"- mean F1 over normal conditions: {conditions['clade_f1'].mean():.4f}\n"
        f"- mean F1 conditional on an error: {error_conditions['clade_f1'].mean():.4f}\n\n"
        "## Error region summary\n\n"
        + region_summary.to_string(index=False)
        + "\n\n## Highest physical-line miss rates\n\n"
        + line_summary.head(20).to_string(index=False)
        + "\n\n## Extra predicted line types\n\n"
        + extra_summary.to_string(index=False)
        + "\n\n## Terminal attachment summary\n\n"
        + terminal_summary.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/current_method_line_error_analysis")
    parser.add_argument("--cases", nargs="*", choices=list(CASE_BUILDERS), default=None)
    parser.add_argument("--scenario-counts", nargs="*", type=int, default=None)
    parser.add_argument("--T-counts", nargs="*", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--pq-noise-rel", type=float, default=0.005)
    parser.add_argument("--v-noise-rel", type=float, default=0.0002)
    parser.add_argument("--tolerance-factor", type=float, default=0.16)
    args = parser.parse_args()
    run(
        output_dir=args.output,
        cases=args.cases,
        scenario_counts=args.scenario_counts,
        t_counts=args.T_counts,
        replicates=args.replicates,
        pq_noise_rel=args.pq_noise_rel,
        v_noise_rel=args.v_noise_rel,
        tolerance_factor=args.tolerance_factor,
    )


if __name__ == "__main__":
    main()
