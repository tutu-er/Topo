"""Small cases derived from published parameters, with explicit unit provenance.

This module does not estimate physical impedance bounds. The case33 local case
keeps five published line impedances and three static load entries, deletes
other branches and hidden-node loads, and is a newly derived simulation case.
It is neither the complete standard feeder nor field PQV measurement data.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np

from .candidates import RadialCandidate
from terminal_case33.data.case33bw_raw import load_raw_case33bw


BASE_KV = 12.66
BASE_MVA = 1.0
BASE_KW = BASE_MVA * 1000.0
ZBASE_OHM = BASE_KV ** 2 / BASE_MVA
TERMINAL_BUSES = (4, 19, 23)
SOURCE_EDGES = ((1, 2), (2, 3), (3, 4), (2, 19), (3, 23))
SOURCE_URL = "https://matpower.org/docs/ref/matpower6.0/case33bw.html"
SOURCE_RELATIVE_PATH = "terminal_case33/data/case33bw_raw.py"


def power_kw_to_pu(value, *, base_kw=BASE_KW):
    """Convert kW (or kvar) on a stated kVA base; no global old-study base."""
    if not np.isfinite(base_kw) or base_kw <= 0:
        raise ValueError("base_kw must be finite and positive")
    values = np.asarray(value, float)
    if not np.isfinite(values).all():
        raise ValueError("power must be finite")
    return values / float(base_kw)


def impedance_ohm_to_pu(value, *, base_kv=BASE_KV, base_mva=BASE_MVA):
    """Convert ohms using kV line voltage and total three-phase MVA."""
    if not np.isfinite(base_kv) or not np.isfinite(base_mva) or base_kv <= 0 or base_mva <= 0:
        raise ValueError("bases must be finite and positive")
    values = np.asarray(value, float)
    if not np.isfinite(values).all():
        raise ValueError("impedance must be finite")
    return values / (float(base_kv) ** 2 / float(base_mva))


@dataclass(frozen=True)
class PhysicalCase:
    candidate: object
    r: np.ndarray
    x: np.ndarray
    p0: np.ndarray
    q0: np.ndarray
    base_kv: float
    base_mva: float
    base_kw: float
    zbase_ohm: float
    provenance: dict


def case33_local_standard():
    """Five-line standard-data derivative with source mapping in solver order.

    p0/q0 are static references only. Any generated history/future variation,
    root voltage, source injection and meter errors must be labelled separately.
    """
    raw = load_raw_case33bw(include_tie_lines=False)
    candidate = RadialCandidate.from_supports("case33_derived_clade_02", 3, ((0, 2),))
    source_by_support = {
        (0, 1, 2): (1, 2), (0, 2): (2, 3),
        (0,): (3, 4), (1,): (2, 19), (2,): (3, 23),
    }
    ordered_supports = [tuple(s) for s in candidate.supports]
    if set(ordered_supports) != set(source_by_support) or len(ordered_supports) != 5:
        raise RuntimeError("solver support ordering does not cover five source lines")
    branch_map = {(int(row.from_bus), int(row.to_bus)): row
                  for row in raw.branches.itertuples(index=False)}
    edge_rows = []
    for support in ordered_supports:
        edge = source_by_support[support]
        row = branch_map[edge]
        edge_rows.append(dict(support=list(support), source_edge=list(edge),
            source_branch_id=int(row.branch_id), r_ohm=float(row.r_ohm), x_ohm=float(row.x_ohm)))
    r = impedance_ohm_to_pu([row["r_ohm"] for row in edge_rows])
    x = impedance_ohm_to_pu([row["x_ohm"] for row in edge_rows])
    buses = raw.buses.set_index("bus_id")
    p_kw = buses.loc[list(TERMINAL_BUSES), "pd_kw"].to_numpy(float)
    q_kvar = buses.loc[list(TERMINAL_BUSES), "qd_kvar"].to_numpy(float)
    source_file = Path(__file__).resolve().parents[2] / SOURCE_RELATIVE_PATH
    body = source_file.read_bytes()
    zeroed_loads = [dict(source_bus=bus, removed_p_kw=float(buses.loc[bus, "pd_kw"]),
                         removed_q_kvar=float(buses.loc[bus, "qd_kvar"])) for bus in (2, 3)]
    omitted = [int(bus) for bus in buses.index if bus not in (1, 2, 3, *TERMINAL_BUSES)]
    record = dict(name="case33_five_line_standard_derivative", data_kind="standard_parameters_derived_simulation",
        field_PQV_available=False, source_url=SOURCE_URL, source_path=SOURCE_RELATIVE_PATH,
        source_sha256=hashlib.sha256(body).hexdigest(),
        source_lf_sha256=hashlib.sha256(body.replace(b"\r\n", b"\n")).hexdigest(),
        source_voltage_kv=float(raw.base_kv), source_local_base_mva=float(raw.base_mva),
        new_base_kv=BASE_KV, new_base_mva=BASE_MVA, new_base_kw=BASE_KW, zbase_ohm=ZBASE_OHM,
        root_source_bus=1, terminal_source_buses=list(TERMINAL_BUSES),
        ordered_edges=edge_rows, source_load_p_kw=p_kw.tolist(), source_load_q_kvar=q_kvar.tolist(),
        hidden_loads_zeroed=zeroed_loads, omitted_source_buses=omitted,
        omitted_branch_count=int(len(raw.branches) - len(edge_rows)),
        changes=["retained exactly five published closed-line ohmic impedances",
            "zeroed original loads at retained hidden buses 2 and 3",
            "deleted other branches and their loads", "rebased all powers and impedances consistently"],
        impedance_scale_factor=1.0, fitted_impedance_lower_bound_from_assets=None,
        missing_asset_information=["line lengths", "cable type", "parameter uncertainty", "temperature"],
        generated_quantities_require_explicit_label=["load time series", "root voltage", "source P/Q", "meter errors"],
        not_complete_original_feeder=True,
        ev_6kw_pu=float(power_kw_to_pu(6.0)),
        ev_6kw_to_terminal_load_ratios=(6.0 / p_kw).tolist(),
        old_synthetic_6kw_pu=6.0 / 20.0,
        old_synthetic_6kw_to_nominal_terminal_load_ratio=6.0 / (0.12 * 20.0))
    return PhysicalCase(candidate, r, x, power_kw_to_pu(p_kw), power_kw_to_pu(q_kvar),
                        BASE_KV, BASE_MVA, BASE_KW, ZBASE_OHM, record)


def case33_local_record():
    """Return a JSON-compatible case record without a solver or AC call."""
    case = case33_local_standard()
    return dict(**case.provenance, candidate_name=case.candidate.name,
        supports=[list(s) for s in case.candidate.supports], r_pu=case.r.tolist(), x_pu=case.x.tolist(),
        p0_pu=case.p0.tolist(), q0_pu=case.q0.tolist())
