"""Tests for evidence coupling and deterministic pipeline gates."""

import numpy as np
import pandas as pd

from terminal_case33.data.small_terminal_lv import build_small_terminal_lv_case
from terminal_case33.estimation.preprocessing import squared_voltage_drop_from_observed_root
from terminal_case33.graph.quartet_significance import QuartetCladeEvidence
from terminal_case33.graph.rooted_hierarchy import rooted_tree_from_clades
from terminal_case33.models.ac_powerflow import solve_ac_power_flow_timeseries
from terminal_case33.pipeline.unified_topology_pipeline import (
    UnifiedTopologyConfig,
    build_clade_evidence,
    identify_topology_unified,
)


def _quartet(
    clade: frozenset[int],
    upper: float,
    accepted: bool = False,
) -> QuartetCladeEvidence:
    return QuartetCladeEvidence(
        clade=clade,
        estimate=upper,
        lower_confidence_bound=upper,
        upper_confidence_bound=upper,
        positive_support=float(accepted),
        quartet_count=4,
        replicate_count=10,
        accepted=accepted,
    )


def _small_ac_scenarios() -> tuple[object, list[dict]]:
    net = build_small_terminal_lv_case()
    terminals = net.load_buses()
    buses = net.buses.set_index("bus_id")
    base = np.array(
        [float(buses.loc[node, "pd_kw"]) / (1000.0 * net.base_mva) for node in terminals]
    )
    scenarios = []
    for seed in (11, 22, 33):
        rng = np.random.default_rng(seed)
        count = 32
        p_values = base[None, :] * rng.lognormal(
            0.0,
            0.25,
            (count, len(terminals)),
        )
        q_values = p_values * rng.uniform(0.20, 0.38, p_values.shape)
        p = pd.DataFrame(p_values, columns=terminals)
        q = pd.DataFrame(q_values, columns=terminals)
        root = pd.Series(1.02 + 0.001 * np.sin(np.linspace(0.0, 4.0 * np.pi, count)))
        ac = solve_ac_power_flow_timeseries(net, p, q, v_root=root)
        voltage = ac["V_bus_mag"].loc[:, terminals]
        scenarios.append(
            {
                "name": f"small_{seed}",
                "P_terminal": p,
                "Q_terminal": q,
                "V_terminal": voltage,
                "root_voltage": root,
                "drop_target": squared_voltage_drop_from_observed_root(
                    voltage,
                    root,
                ),
            }
        )
    return net, scenarios


def test_aggregation_requires_confirmation_and_rejects_contradiction() -> None:
    confirmed = frozenset({10, 11})
    ordered_only = frozenset({12, 13})
    contradicted = frozenset({14, 15})
    evidence = build_clade_evidence(
        {confirmed, ordered_only, contradicted},
        ac_raw_clades={contradicted},
        ac_validated_clades={contradicted},
        gtls_map_clades={confirmed, contradicted},
        gtls_marginals={confirmed: 0.7, contradicted: 0.8},
        ordered_base_clades={confirmed, ordered_only, contradicted},
        quartet_evidence={
            confirmed: _quartet(confirmed, 0.1, accepted=True),
            contradicted: _quartet(contradicted, -0.08),
        },
    )

    assert evidence[confirmed].stable_for_aggregation
    assert evidence[confirmed].source_support_count == 3
    assert not evidence[ordered_only].stable_for_aggregation
    assert evidence[ordered_only].source_support_count == 1
    assert evidence[contradicted].materially_contradicted
    assert not evidence[contradicted].stable_for_aggregation


def test_unified_pipeline_reranks_one_shared_candidate_pool_with_ac() -> None:
    net, scenarios = _small_ac_scenarios()
    result = identify_topology_unified(
        scenarios,
        net,
        pq_noise_relative_std=0.005,
        voltage_noise_relative_std=0.0002,
        config=UnifiedTopologyConfig(
            validation_scenario_count=1,
            quartet_replicates=2,
            gtls_temporal_block_length=4,
            nj_edge_bootstrap_replicates=2,
            nj_edge_support_threshold=0.50,
            nj_edge_minimum_boundary_margin=0.0,
        ),
        seed=17,
    )

    assert result.selection_source.startswith("ac_")
    assert result.pre_quartet_clades == result.ac_validated.raw_clades
    assert result.final_clades == result.ac_validated.validated_clades
    assert result.applied_aggregation_clades <= result.stable_aggregation_clades
    assert result.peripheral_proposals is not None
    assert set(result.peripheral_proposals.clusters_by_mode) == {
        "nj_stable",
        "nj_rg_consensus",
    }
    assert all(
        mode.startswith(("nj_stable_", "nj_rg_consensus_")) for mode in result.detector_aggregations
    )
    assert all(item.clusters for item in result.detector_aggregations.values())
    rooted_tree_from_clades(
        result.final_clades,
        result.base.terminals,
        result.base.root_bus,
    )
