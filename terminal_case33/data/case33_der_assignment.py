"""DER/load-type assignment for IEEE case33 nodes.

The assignment is deterministic and intended for reproducible topology-
identification simulations. It keeps the original case33 load at every non-root
bus and adds a node-level DER category that can be converted into time-series
net-load profiles using public PV, wind, and generator profile datasets.
"""

from __future__ import annotations

import pandas as pd

from terminal_case33.data.case33bw_raw import RawCase33, load_raw_case33bw


_DATASET_CANDIDATES = {
    "base_load": "OPSD time_series / local smart-meter profile; use as demand multiplier",
    "pv": "NREL PVDAQ or NSRDB-derived PV profile; generation subtracts from load",
    "wind": "NREL WIND Toolkit or OPSD wind profile; generation subtracts from load",
    "small_generator": "dispatchable small generator / CHP / diesel profile; negative load when dispatched",
}


def build_case33_der_assignment(raw: RawCase33 | None = None) -> pd.DataFrame:
    """Assign load/DER profile classes to every case33 bus.

    Returns:
        DataFrame with original load, DER category, suggested profile source,
        DER capacity, and sign convention. Positive ``pd_kw``/``qd_kvar`` remain
        demand. DER active power is interpreted as negative load when forming
        net injections.
    """

    raw = raw or load_raw_case33bw(include_tie_lines=False)
    buses = raw.buses.set_index("bus_id")

    # The choices below are deterministic and deliberately heterogeneous:
    # - PV is assigned to residential/long-lateral users where rooftop PV is plausible.
    # - Wind is assigned to electrically remote/lateral nodes to create smoother,
    #   weather-correlated negative-load profiles.
    # - Small generators are assigned to heavier loads, representing CHP/diesel/biogas.
    pv_buses = {4, 5, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23}
    wind_buses = {11, 12, 13, 14, 15, 16, 17, 26, 27, 28, 29, 32, 33}
    small_gen_buses = {6, 24, 25, 30, 31}

    records = []
    for bus_id, row in buses.iterrows():
        bus_id = int(bus_id)
        pd_kw = float(row.pd_kw)
        qd_kvar = float(row.qd_kvar)
        if bus_id == raw.root_bus:
            category = "root"
            der_kw = 0.0
            note = "observable feeder root; no customer load assigned"
            source = "not applicable"
        elif bus_id in pv_buses:
            category = "pv"
            der_kw = round(0.65 * pd_kw, 3)
            note = "rooftop PV-like negative load with midday peak and irradiance-driven variability"
            source = _DATASET_CANDIDATES["pv"]
        elif bus_id in wind_buses:
            category = "wind"
            der_kw = round(0.55 * pd_kw, 3)
            note = "small wind-like negative load with weather-correlated stochastic generation"
            source = _DATASET_CANDIDATES["wind"]
        elif bus_id in small_gen_buses:
            category = "small_generator"
            der_kw = round(0.45 * pd_kw, 3)
            note = "dispatchable small generator, CHP, diesel, or biogas unit at heavier customer node"
            source = _DATASET_CANDIDATES["small_generator"]
        else:
            category = "base_load"
            der_kw = 0.0
            note = "ordinary demand-only node using the original case33 load"
            source = _DATASET_CANDIDATES["base_load"]

        records.append(
            {
                "bus_id": bus_id,
                "pd_kw": pd_kw,
                "qd_kvar": qd_kvar,
                "profile_class": category,
                "der_capacity_kw": der_kw,
                "der_capacity_ratio_to_load": 0.0 if pd_kw == 0.0 else der_kw / pd_kw,
                "profile_source_candidate": source,
                "net_load_sign_convention": "net_P_load = demand_P - der_P_generation",
                "note": note,
            }
        )
    return pd.DataFrame(records)


def assignment_summary(assignment: pd.DataFrame) -> dict:
    """Summarize counts and capacity by profile class."""

    grouped = assignment.groupby("profile_class", dropna=False).agg(
        bus_count=("bus_id", "count"),
        total_demand_kw=("pd_kw", "sum"),
        total_der_capacity_kw=("der_capacity_kw", "sum"),
    )
    return grouped.reset_index().to_dict(orient="records")
