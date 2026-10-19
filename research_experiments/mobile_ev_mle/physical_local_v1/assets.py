"""New simulator assets; the old paper-style builders remain byte-unchanged.

These are synthetic layouts with traceable 50 Hz conductors, not reproductions
of the named authors' original asset tables. Use cable_v1 for new experiments.
The returned physical R/X are simulator inputs, never estimator inputs.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from ..reference_benchmark.networks import (
    _assignments, _finalize, _source, build_network as legacy_network,
)


CATALOG_PATH = Path(__file__).with_name("catalog.json")
PAPER_NAMES = ("paper15", "paper11", "flynn16", "pengwah18")
PROFILES = ("legacy_unscaled", "cable_v1", "mixed_v1")


def catalog():
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def canonical(name):
    return {"soumalas11": "paper11", "samem11": "paper11",
            "case33bw": "case33"}.get(name, name)


def _other_paper(name):
    functions = {"flynn16": "build_flynn_balanced_lv_16_case",
                 "pengwah18": "build_pengwah_mixed_lv_18_case"}
    relative = "rnj_wzzt_core/rnj_wzzt/data/paper_style_case_bank.py"
    tree, record = _source(relative)
    fun = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == functions[name])
    assigned = _assignments(fun.body)
    hidden = ast.literal_eval(assigned["hidden"])
    raw_loads = assigned["terminals"]
    # The two legacy builders enumerate a literal list of P/Q pairs. Parse
    # that limited form without executing the builder or arbitrary expressions.
    if isinstance(raw_loads, ast.ListComp):
        pairs = ast.literal_eval(raw_loads.generators[0].iter.args[0])
        first_bus = ast.literal_eval(raw_loads.elt.elts[0].left)
        loads = [(first_bus+i, float(p), float(q)) for i, (p, q) in enumerate(pairs)]
    else:
        loads = ast.literal_eval(raw_loads)
    lines = ast.literal_eval(assigned["edges"])
    call = next(n.value for n in fun.body if isinstance(n, ast.Return))
    old_scale = ast.literal_eval(next(k.value for k in call.keywords if k.arg == "impedance_scale"))
    buses = [dict(bus_id=1, pd_kw=0., qd_kvar=0., nominal_kv=.4, source_label="root")]
    buses += [dict(bus_id=i, pd_kw=0., qd_kvar=0., nominal_kv=.4,
                   source_label=f"hidden_{i}") for i in hidden]
    buses += [dict(bus_id=i, pd_kw=float(p), qd_kvar=float(q), nominal_kv=.4,
                   source_label=str(i)) for i, p, q in loads]
    edges = []
    for i, (u, v, metres, kind) in enumerate(lines, 1):
        rp, xp = (.55, .38) if kind == "service" else (.42, .34)
        edges.append(dict(branch_id=i, from_bus=u, to_bus=v, r_ohm=rp*metres/1000.,
            x_ohm=xp*metres/1000., status=True, branch_type=kind,
            length_km=metres/1000., rating_a=None))
    return _finalize(dict(name=name, root_bus=1, base_kv=.4, base_mva=.2,
        source_bus_suggestion=1, buses=buses, branches=edges,
        provenance=dict(**record, source_url=None, original_paper_parameters=False,
            data_kind="project_synthetic_paper_style_compatibility", field_PQV=False,
            old_impedance_multiplier=old_scale, new_impedance_multiplier=1.,
            modifications=[f"undo original synthetic multiplier {old_scale}"],
            line_parameter_note="legacy synthetic R/X-balanced line library"),
        metadata=dict(declared_terminalization="none", original_internal_loads_moved=0)))


def build_network(name, profile="cable_v1"):
    """Preserve the physical root/layout/loads/lengths; change declared assets.

    case33 and CIGRE ignore the LV profile and retain their published R/X.
    There is no user-independent claim about actual local cable installation.
    """
    name = canonical(name)
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    if name in ("case33", "cigre_residential"):
        return legacy_network(name)
    if name not in PAPER_NAMES:
        raise ValueError("unknown physical case")
    net = (legacy_network(name) if name in ("paper11", "paper15") else _other_paper(name))
    net = deepcopy(net)
    if profile == "legacy_unscaled":
        net["metadata"]["asset_profile"] = profile
        return net
    cat = catalog()
    for e in net["branches"]:
        if e["branch_type"] == "service":
            typ = "NAYY 4x50 SE"
        elif profile == "cable_v1":
            typ = "NAYY 4x150 SE" if e["from_bus"] == net["root_bus"] else "NAYY 4x120 SE"
        else:
            typ = ("94-AL1/15-ST1A 0.4" if e["from_bus"] == net["root_bus"]
                   else "48-AL1/8-ST1A 0.4")
        row = cat["types"][typ]
        e.update(line_type=typ, r_ohm=row["r_ohm_per_km"]*e["length_km"],
            x_ohm=row["x_ohm_per_km"]*e["length_km"], rating_a=1000.*row["max_i_ka"],
            shunt_capacitance_nf_per_km=row["c_nf_per_km"],
            conductor_reference_celsius=20, physical_asset_source=cat["source_url"])
    net["provenance"].update(
        data_kind="synthetic_layout_with_public_conductor_assets",
        line_parameter_note="50 Hz catalog assets times retained lengths; no Z scaling or random jitter",
        conductor_catalog_sha256=hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(),
        conductor_source_url=cat["source_url"], conductor_source_version=cat["source_version"],
        catalog_assignment_is_scenario=True, original_paper_parameters=False,
        field_PQV=False, temperature_estimated=False,
        shunt_model="series-only AC; catalog C retained and charging-Q bound reported")
    net["provenance"]["modifications"] += [
        "replace legacy synthetic line library by role-based catalog assets; retain all original lengths/loads"]
    net["metadata"].update(asset_profile=profile, frequency_hz=50,
        conductor_temperature_celsius=20, added_transformer=False,
        original_layout_and_nominal_loads_retained=True)
    return _finalize(net)


def ratio_summary(net):
    """Separate line ratios from transformer ratios; lengths do not change X/R."""
    edges = [e for e in net["branches"] if e["r_ohm"] > 0]
    x_r = np.array([e["x_ohm"]/e["r_ohm"] for e in edges])
    return dict(edges=len(edges), minimum=float(x_r.min()), median=float(np.median(x_r)),
        maximum=float(x_r.max()), distinct_ratio_count=len(set(np.round(x_r, 10))),
        ratio_values=sorted(set(np.round(x_r, 10).tolist())),
        legacy_domain_excluded_edges=sum(not .25 <= z <= 1.5 for z in x_r))


def public_lv_ratio_domain():
    """A declared broad catalog envelope, without a true network argument.

    This is for the stated LV asset family, not a universal field constraint.
    It covers all catalog entries and the legacy comparison, including X/R<.25.
    """
    ratios = [z["x_ohm_per_km"]/z["r_ohm_per_km"] for z in catalog()["types"].values()]
    return dict(ratio_min=.05, ratio_max=2., source="declared widened 50 Hz LV catalog envelope",
                catalog_ratio_min=min(ratios), catalog_ratio_max=max(ratios),
                physical_impedance_lower_bound=0., field_coverage_claim=False)


def charging_q_bound_kvar(net, vmax_pu=1.15):
    """3-phase charging Q at a uniform Vmax, using per-phase catalog C."""
    if any("shunt_capacitance_nf_per_km" not in e for e in net["branches"]):
        return None  # A missing legacy capacitance is not evidence of zero Q.
    capacitance_nf = sum(e.get("shunt_capacitance_nf_per_km", 0.)*e.get("length_km", 0.)
                         for e in net["branches"])
    return float(2*np.pi*50*capacitance_nf*net["base_kv"]**2*vmax_pu**2*1e-6)


__all__ = ["build_network", "catalog", "ratio_summary", "public_lv_ratio_domain",
           "charging_q_bound_kvar", "PAPER_NAMES", "PROFILES"]
