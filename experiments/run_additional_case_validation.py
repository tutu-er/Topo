"""Validate the unified pipeline on independent synthetic and open-data cases."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import simulate_case_scenario
from terminal_case33.data.norwegian_industrial import (
    DATASET_DOI,
    canonicalize_norwegian_terminal_radial,
    load_norwegian_active_power,
    load_norwegian_industrial_radial,
)
from terminal_case33.data.paper_style_case_bank import (
    build_case_bank_resource_assignment,
)
from terminal_case33.data.small_terminal_lv import build_small_terminal_lv_case
from terminal_case33.estimation.preprocessing import (
    squared_voltage_drop_from_observed_root,
)
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.models.ac_powerflow import solve_ac_power_flow_timeseries
from terminal_case33.models.network import TerminalizedNetwork
from terminal_case33.pipeline.unified_topology_pipeline import (
    UnifiedTopologyConfig,
    identify_topology_unified,
)
from terminal_case33.utils.io import ensure_dir, write_json


def _clade_metrics(predicted: set[frozenset[int]], truth: set[frozenset[int]]) -> dict:
    """Return precision, recall, F1, and exact recovery for rooted clades."""

    true_positive = len(predicted & truth)
    precision = true_positive / len(predicted) if predicted else float(not truth)
    recall = true_positive / len(truth) if truth else float(not predicted)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "exact": predicted == truth,
    }


def _small6_scenarios(
    scenario_count: int,
    sample_count: int,
    pq_noise_rel: float,
    v_noise_rel: float,
) -> tuple[TerminalizedNetwork, list[dict], dict]:
    """Build heterogeneous synthetic days on the independent six-terminal case."""

    net = build_small_terminal_lv_case()
    assignment = build_case_bank_resource_assignment(net)
    profiles = ("default", "high_wind", "no_solar", "high_load", "storm_front")
    scenarios = []
    voltage_min = float("inf")
    voltage_max = float("-inf")
    for index in range(scenario_count):
        simulation = simulate_case_scenario(
            net=net,
            assignment=assignment,
            T=sample_count,
            seed=4100 + 503 * index,
            pq_noise_rel=pq_noise_rel,
            v_noise_rel=v_noise_rel,
            root_voltage_mean=1.02,
            root_voltage_sigma=0.0008,
            profile_scenario=profiles[index % len(profiles)],
        )
        voltage_min = min(voltage_min, float(simulation["V_true"].min().min()))
        voltage_max = max(voltage_max, float(simulation["V_true"].max().max()))
        scenarios.append(
            {
                "name": f"small6_{profiles[index % len(profiles)]}_{index}",
                "P_terminal": simulation["P_terminal"],
                "Q_terminal": simulation["Q_terminal"],
                "V_terminal": simulation["V_terminal"],
                "root_voltage": simulation["root_voltage"],
                "drop_target": simulation["drop_target"],
            }
        )
    return net, scenarios, {
        "source": "built-in independent six-terminal synthetic LV feeder",
        "active_power": "heterogeneous synthetic demand/PV/wind/generator profiles",
        "reactive_power": "profile generator with resource-specific Q behavior",
        "voltage": "nonlinear radial AC backward-forward sweep",
        "true_voltage_min_pu": voltage_min,
        "true_voltage_max_pu": voltage_max,
    }


def _norwegian_scenarios(
    radial: int,
    scenario_count: int,
    hours_per_scenario: int,
    pq_noise_rel: float,
    v_noise_rel: float,
) -> tuple[TerminalizedNetwork, list[dict], dict]:
    """Build AC-voltage scenarios from open Norwegian hourly active-load data."""

    raw_net = load_norwegian_industrial_radial(radial=radial)
    net = canonicalize_norwegian_terminal_radial(raw_net)
    active = load_norwegian_active_power(net)
    if len(active) < hours_per_scenario:
        raise ValueError("Norwegian source has fewer rows than one requested scenario")
    starts = np.linspace(
        0,
        len(active) - hours_per_scenario,
        scenario_count,
        dtype=int,
    )
    scenarios = []
    source_ranges = []
    selected_active = []
    voltage_min = float("inf")
    voltage_max = float("-inf")
    terminals = net.load_buses()
    for scenario_index, start in enumerate(starts):
        p_true = active.iloc[start : start + hours_per_scenario].copy()
        selected_active.append(p_true)
        rng = np.random.default_rng(7300 + 101 * radial + scenario_index)
        q_ratio = rng.uniform(0.20, 0.36, size=len(terminals))
        phase = rng.uniform(0.0, 2.0 * np.pi, size=len(terminals))
        hour = np.arange(hours_per_scenario, dtype=float)[:, None]
        q_modulation = 1.0 + 0.08 * np.sin(2.0 * np.pi * hour / 24.0 + phase[None, :])
        q_true = pd.DataFrame(
            p_true.to_numpy(dtype=float) * q_ratio[None, :] * q_modulation,
            index=p_true.index,
            columns=terminals,
        )
        root = pd.Series(
            1.02 + rng.normal(0.0, 0.0008, size=hours_per_scenario),
            index=p_true.index,
            name="V_root_true",
        )
        ac = solve_ac_power_flow_timeseries(net, p_true, q_true, v_root=root)
        if not bool(ac["converged"].all()):
            raise RuntimeError(f"AC power flow failed for Norwegian radial {radial}")
        v_true = ac["V_bus_mag"].loc[:, terminals]
        p_meas = p_true + rng.normal(
            0.0,
            pq_noise_rel * np.maximum(np.abs(p_true), 1e-12),
            size=p_true.shape,
        )
        q_meas = q_true + rng.normal(
            0.0,
            pq_noise_rel * np.maximum(np.abs(q_true), 1e-12),
            size=q_true.shape,
        )
        v_meas = v_true + rng.normal(
            0.0,
            v_noise_rel * np.maximum(np.abs(v_true), 1e-12),
            size=v_true.shape,
        )
        voltage_min = min(voltage_min, float(v_true.min().min()))
        voltage_max = max(voltage_max, float(v_true.max().max()))
        source_ranges.append(
            {"start": str(p_true.index[0]), "end": str(p_true.index[-1])}
        )
        scenarios.append(
            {
                "name": f"norwegian_r{radial}_block_{scenario_index}",
                "P_terminal": p_meas,
                "Q_terminal": q_meas,
                "V_terminal": v_meas,
                "root_voltage": root,
                "drop_target": squared_voltage_drop_from_observed_root(v_meas, root),
            }
        )
    selected_p = pd.concat(selected_active, axis=0)
    selected_total_p = selected_p.sum(axis=1)
    return net, scenarios, {
        "source": f"Norwegian industrial distribution grid, DOI {DATASET_DOI}",
        "network_form": "canonical identifiable tree: hidden zero-Z edges contracted, degree-2 hidden chains suppressed, terminal zero-Z connectors replaced by 0.0005+j0.0004 pu service impedance",
        "raw_bus_count": len(raw_net.buses),
        "canonical_bus_count": len(net.buses),
        "raw_hidden_count": len(raw_net.hidden_buses()),
        "canonical_hidden_count": len(net.hidden_buses()),
        "selected_total_p_min_pu": float(selected_total_p.min()),
        "selected_total_p_mean_pu": float(selected_total_p.mean()),
        "selected_total_p_max_pu": float(selected_total_p.max()),
        "active_power": "published hourly smart-meter active energy interpreted as average kW",
        "reactive_power": "synthetic time-varying Q/P ratio in [0.20, 0.36]",
        "voltage": "synthetic nonlinear radial AC power flow on the published topology and R/X",
        "root_voltage": "synthetic observed slack magnitude; no added root-meter noise",
        "source_ranges": source_ranges,
        "true_voltage_min_pu": voltage_min,
        "true_voltage_max_pu": voltage_max,
    }


def _run_case(
    case: str,
    scenario_count: int,
    synthetic_sample_count: int,
    norwegian_hours: int,
    pq_noise_rel: float,
    v_noise_rel: float,
    quartet_replicates: int,
    aggregation_replicates: int,
) -> dict:
    """Run one additional case and return auditable topology metrics."""

    if case == "small6":
        net, scenarios, provenance = _small6_scenarios(
            scenario_count,
            synthetic_sample_count,
            pq_noise_rel,
            v_noise_rel,
        )
    elif case.startswith("norwegian_r"):
        radial = int(case.removeprefix("norwegian_r"))
        net, scenarios, provenance = _norwegian_scenarios(
            radial,
            scenario_count,
            norwegian_hours,
            pq_noise_rel,
            v_noise_rel,
        )
    else:
        raise ValueError(f"unknown case {case!r}")

    terminals = [int(column) for column in scenarios[0]["P_terminal"].columns]
    truth = set(rooted_clades(net.closed_edges(), net.root_bus, terminals))
    config = UnifiedTopologyConfig(
        validation_scenario_count=1,
        quartet_replicates=quartet_replicates,
        nj_edge_bootstrap_replicates=aggregation_replicates,
    )
    started = time.perf_counter()
    result = identify_topology_unified(
        scenarios,
        net,
        pq_noise_relative_std=pq_noise_rel,
        voltage_noise_relative_std=v_noise_rel,
        config=config,
        seed=20260715 + len(case),
    )
    elapsed = time.perf_counter() - started
    ac = result.ac_validated.ac_rerank
    candidate_metrics = [
        _clade_metrics(set(score.candidate.clades), truth) for score in ac.ranked
    ]
    methods = {
        "ordered_base": set(result.base.rooted_clades),
        "gtls_map": set(result.gtls_posterior.selected.projection.candidate.clades),
        "shared_ac_pre_quartet": set(result.ac_validated.raw_clades),
        "unified": set(result.final_clades),
    }
    graph = net.to_networkx_graph()
    hidden = net.hidden_buses()
    return {
        "case": case,
        "terminal_count": len(terminals),
        "hidden_count": len(hidden),
        "hidden_degree2_count": sum(graph.degree(node) == 2 for node in hidden),
        "truth_clade_count": len(truth),
        "scenario_count": len(scenarios),
        "samples_per_scenario": len(scenarios[0]["P_terminal"]),
        "pq_noise_rel": pq_noise_rel,
        "v_noise_rel": v_noise_rel,
        "selection_source": result.selection_source,
        "candidate_count": ac.candidate_count,
        "candidate_oracle_f1": max(item["f1"] for item in candidate_metrics),
        "candidate_contains_exact": any(item["exact"] for item in candidate_metrics),
        "aggregation_rerun": result.aggregation_rerun,
        "applied_cluster_count": len(result.applied_aggregation_clades),
        "elapsed_seconds": elapsed,
        "provenance": provenance,
        "metrics": {
            name: _clade_metrics(prediction, truth)
            for name, prediction in methods.items()
        },
    }


def _write_report(results: list[dict], output: Path) -> None:
    """Write CSV, JSON, and a concise Markdown report."""

    rows = []
    for result in results:
        row = {key: value for key, value in result.items() if key not in {"metrics", "provenance"}}
        for method, metrics in result["metrics"].items():
            row[f"{method}_f1"] = metrics["f1"]
            row[f"{method}_exact"] = metrics["exact"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    write_json(output / "results.json", {"cases": results})
    lines = [
        "# Additional-case validation",
        "",
        "All voltages are generated by nonlinear radial AC power flow. Measurement noise is relative to each instantaneous measured quantity.",
        "",
        "| case | terminals | scenarios x samples | candidates | AC F1 | unified F1 | exact | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        ac_f1 = result["metrics"]["shared_ac_pre_quartet"]["f1"]
        final = result["metrics"]["unified"]
        lines.append(
            f"| {result['case']} | {result['terminal_count']} | "
            f"{result['scenario_count']} x {result['samples_per_scenario']} | "
            f"{result['candidate_count']} | {ac_f1:.4f} | {final['f1']:.4f} | "
            f"{final['exact']} | {result['elapsed_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Provenance boundary",
            "",
            "`small6` is fully synthetic. The Norwegian cases use published topology, R/X, and hourly active-load records; Q, root voltage, and terminal voltage are synthetic. They are therefore semi-synthetic topology tests, not field PQV validation.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run requested additional cases."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["small6", "norwegian_r2"])
    parser.add_argument("--scenario-count", type=int, default=3)
    parser.add_argument("--synthetic-samples", type=int, default=96)
    parser.add_argument("--norwegian-hours", type=int, default=168)
    parser.add_argument("--pq-noise-rel", type=float, default=0.005)
    parser.add_argument("--v-noise-rel", type=float, default=0.0002)
    parser.add_argument("--quartet-replicates", type=int, default=20)
    parser.add_argument("--aggregation-replicates", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("outputs/additional_case_validation"))
    args = parser.parse_args()
    output = ensure_dir(args.output)
    results = []
    for index, case in enumerate(args.cases, start=1):
        print(f"[{index}/{len(args.cases)}] {case}", flush=True)
        result = _run_case(
            case,
            args.scenario_count,
            args.synthetic_samples,
            args.norwegian_hours,
            args.pq_noise_rel,
            args.v_noise_rel,
            args.quartet_replicates,
            args.aggregation_replicates,
        )
        results.append(result)
        print(
            f"  unified_f1={result['metrics']['unified']['f1']:.4f} "
            f"elapsed={result['elapsed_seconds']:.1f}s",
            flush=True,
        )
    _write_report(results, output)
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
