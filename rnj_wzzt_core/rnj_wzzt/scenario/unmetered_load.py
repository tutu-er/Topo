"""One extra physical load, invisible to the existing terminal P/Q meters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rnj_wzzt.models.network import TerminalizedNetwork
from rnj_wzzt.scenario.settings import _finite_number
from rnj_wzzt.scenario.simulation import (
    SCENARIO_DEFINITIONS,
    _build_base_trajectory,
    _diagnostics,
    _integer,
    _observe_terminal_channels,
    _prepare_case_context,
    _solve_physical_ac,
)


@dataclass(frozen=True)
class UnmeteredLoadSpec:
    """Extra consumption at an existing non-root bus on the full 96-point grid."""

    bus_id: int
    p_kw: pd.Series
    q_kvar: pd.Series


@dataclass
class ScenarioBundle:
    """Keep estimator-visible measurements separate from simulation truth."""

    observed: dict
    truth: dict
    diagnostics: dict


def _source_on_grid(
    source: UnmeteredLoadSpec | None, net: TerminalizedNetwork, index: pd.Index,
) -> tuple[int | None, pd.Series, pd.Series]:
    p_zero = pd.Series(0.0, index=index, name="P_unmetered_true")
    q_zero = pd.Series(0.0, index=index, name="Q_unmetered_true")
    if source is None:
        return None, p_zero, q_zero
    if not isinstance(source, UnmeteredLoadSpec):
        raise ValueError("source must be an UnmeteredLoadSpec or None")
    if isinstance(source.bus_id, (bool, np.bool_)) or not isinstance(source.bus_id, (int, np.integer)):
        raise ValueError("source bus_id must be an integer physical bus ID")
    bus_id = int(source.bus_id)
    if bus_id == int(net.root_bus) or bus_id not in set(net.buses["bus_id"]):
        raise ValueError("source bus_id must identify an existing non-root bus")
    for name, values in (("p_kw", source.p_kw), ("q_kvar", source.q_kvar)):
        if not isinstance(values, pd.Series) or not values.index.equals(index):
            raise ValueError(f"source {name} must be a Series on the full physical time index")
    try:
        p_kw = source.p_kw.to_numpy(dtype=float, copy=True)
        q_kvar = source.q_kvar.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("source P/Q must contain numeric values") from exc
    if not np.isfinite(p_kw).all() or not np.isfinite(q_kvar).all() or (p_kw < 0.0).any():
        raise ValueError("source P must be finite and nonnegative; Q must be finite")
    base_kw = 1000.0 * net.base_mva
    p_pu = pd.Series(p_kw / base_kw, index=index, name="P_unmetered_true")
    q_pu = pd.Series(q_kvar / base_kw, index=index, name="Q_unmetered_true")
    effective_bus = bus_id if np.any(p_kw != 0.0) or np.any(q_kvar != 0.0) else None
    return effective_bus, p_pu, q_pu


def simulate_unmetered_scenario(
    case_key: str,
    *,
    scenario_index: int = 0,
    replicate: int = 0,
    source: UnmeteredLoadSpec | None = None,
    t_count: int = 96,
    scenario_suite: str = "reference",
    pq_noise_rel: float = 0.0,
    v_noise_rel: float = 0.0,
    master_p_noise_rel: float = 0.0,
    master_q_noise_rel: float = 0.0,
    root_observation: str | None = None,
    root_meter_noise_rel: float | None = None,
    root_sigma: float | None = None,
    impedance_scale: float | None = None,
) -> tuple[TerminalizedNetwork, ScenarioBundle]:
    """Simulate one full AC day with an extra physical load and honest terminal meters.

    P and Q in ``source`` use kW/kvar on all 96 physical time points. All
    DataFrames and AC powers use per unit on ``net.base_mva``. Observations are
    uniformly subsampled only after the full physical day and its noise are made.
    ``truth`` always retains that full 96-point day and must not enter a fitter.
    """

    scenario_index = _integer(scenario_index, "scenario_index", 0)
    if scenario_index >= len(SCENARIO_DEFINITIONS):
        raise ValueError("scenario_index is outside the available scenario definitions")
    replicate = _integer(replicate, "replicate", 0)
    t_count = _integer(t_count, "t_count", 4)
    if 96 % t_count:
        raise ValueError("t_count must divide the 96-point physical day")
    if scenario_suite == "legacy":
        raise ValueError("unmetered scenarios require a full-grid non-legacy suite")
    pq_noise_rel = _finite_number(pq_noise_rel, "pq_noise_rel")
    v_noise_rel = _finite_number(v_noise_rel, "v_noise_rel")
    master_p_noise_rel = _finite_number(master_p_noise_rel, "master_p_noise_rel")
    master_q_noise_rel = _finite_number(master_q_noise_rel, "master_q_noise_rel")

    context = _prepare_case_context(
        case_key, t_count,
        scenario_suite=scenario_suite, root_observation=root_observation,
        root_meter_noise_rel=root_meter_noise_rel, root_sigma=root_sigma,
        impedance_scale=impedance_scale,
    )
    net = context.net
    base = _build_base_trajectory(context, scenario_index, replicate)
    p_meter_true = base.p_base.copy()
    q_meter_true = base.q_base.copy()
    root_true = base.root_true.copy()
    source_bus, source_p, source_q = _source_on_grid(source, net, p_meter_true.index)

    bus_ids = net.buses["bus_id"].astype(int).tolist()
    p_physical = p_meter_true.reindex(columns=bus_ids, fill_value=0.0).copy()
    q_physical = q_meter_true.reindex(columns=bus_ids, fill_value=0.0).copy()
    if source_bus is not None:
        p_physical.loc[:, source_bus] += source_p.to_numpy()
        q_physical.loc[:, source_bus] += source_q.to_numpy()

    ac, voltage_true = _solve_physical_ac(context, p_physical, q_physical, root_true)
    day = _observe_terminal_channels(
        context, base, p_meter_true, q_meter_true, voltage_true,
        pq_noise_rel=pq_noise_rel, v_noise_rel=v_noise_rel,
    )
    settings = day["scenario_settings"]
    head_edges = [edge["label"] for edge in ac["metadata"]["branch_edges"]
                  if edge["from_bus"] == int(net.root_bus)]
    if not head_edges:
        raise RuntimeError("the network has no root outgoing branch for the master meter")
    p0_true = ac["branch_p_from_pu"][head_edges].sum(axis=1).rename("P0_true")
    q0_true = ac["branch_q_from_pu"][head_edges].sum(axis=1).rename("Q0_true")
    loss_p = ac["branch_loss_p_pu"].sum(axis=1).rename("P_loss_true")
    loss_q = ac["branch_loss_q_pu"].sum(axis=1).rename("Q_loss_true")
    p_balance = p0_true - p_physical.sum(axis=1) - loss_p
    q_balance = q0_true - q_physical.sum(axis=1) - loss_q
    p_balance_max = float(p_balance.abs().max())
    q_balance_max = float(q_balance.abs().max())
    if max(p_balance_max, q_balance_max) > 1e-7:
        raise RuntimeError("AC feeder-head power balance failed")

    # A new RNG stream keeps the existing terminal and root-voltage meters paired.
    master_seed = 80_000_019 + 100_003 * replicate + scenario_index
    master_rng = np.random.default_rng(master_seed)
    p0_measured = pd.Series(
        p0_true.to_numpy() + master_rng.normal(
            0.0, master_p_noise_rel * np.maximum(np.abs(p0_true), 1e-12), size=len(p0_true),
        ), index=p0_true.index, name="P0_measured",
    )
    q0_measured = pd.Series(
        q0_true.to_numpy() + master_rng.normal(
            0.0, master_q_noise_rel * np.maximum(np.abs(q0_true), 1e-12), size=len(q0_true),
        ), index=q0_true.index, name="Q0_measured",
    )

    indices = context.sample_indices
    settings["pq_noise_rel"] = pq_noise_rel
    settings["v_noise_rel"] = v_noise_rel
    settings["master_p_noise_rel"] = master_p_noise_rel
    settings["master_q_noise_rel"] = master_q_noise_rel
    settings["master_meter_seed"] = master_seed
    observed = {
        "name": base.name,
        "P_terminal": day["p_measured"].iloc[indices].copy(),
        "Q_terminal": day["q_measured"].iloc[indices].copy(),
        "V_terminal": day["voltage_measured"].iloc[indices].copy(),
        "root_voltage": day["root_observed"].iloc[indices].copy(),
        "drop_target": day["drop_target"].iloc[indices].copy(),
        "P0_measured": p0_measured.iloc[indices].copy(),
        "Q0_measured": q0_measured.iloc[indices].copy(),
        "root_observation": context.settings["root_observation"],
        "source_sample_indices": indices.tolist(),
        "scenario_settings": settings,
    }
    truth = {
        "source_bus_id": source_bus,
        "P_unmetered": source_p,
        "Q_unmetered": source_q,
        "P_terminal_base": p_meter_true,
        "Q_terminal_base": q_meter_true,
        "P_physical": p_physical,
        "Q_physical": q_physical,
        "root_voltage_true": root_true,
        "V_terminal_true": voltage_true,
        "P0_true": p0_true,
        "Q0_true": q0_true,
        "P_loss_true": loss_p,
        "Q_loss_true": loss_q,
        "ac": ac,
    }
    diagnostics = _diagnostics(
        p_meter_true, q_meter_true, voltage_true, root_true,
        day["p_measured"], day["q_measured"], day["voltage_measured"],
        day["root_observed"], day["drop_target"], settings,
    )
    diagnostics.update({
        "master_meter_seed": master_seed,
        "p_balance_max_abs_pu": p_balance_max,
        "q_balance_max_abs_pu": q_balance_max,
        "source_energy_kwh": float(source_p.sum() * net.base_mva * 1000.0 * 24.0 / 96.0),
    })
    return net, ScenarioBundle(observed=observed, truth=truth, diagnostics=diagnostics)
