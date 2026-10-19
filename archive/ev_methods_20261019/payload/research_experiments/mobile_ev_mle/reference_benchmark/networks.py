"""Source-traceable physical trees, retaining loads at their physical buses.

This module performs no power flow, fitting, leaf conversion, or network I/O.
P/Q values are static nominal loads, not measured time series. A downstream
adapter may encode internal meters as *zero*-impedance virtual observation
leaves; this module never adds service impedance or an upstream line.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
CIGRE_URL = ("https://raw.githubusercontent.com/e2nIEE/pandapower/v3.2.1/"
             "pandapower/networks/cigre_networks.py")
CIGRE_SHA256 = "04817b6bdb11074310fc1e59aa91a27bf5b335a3c1236cb56b655fc73a6a1480"
CASE33_UPSTREAM_URL = "https://raw.githubusercontent.com/MATPOWER/matpower/7.1/data/case33bw.m"
CASE33_URL = CASE33_UPSTREAM_URL
CASE33_UPSTREAM_SHA256 = "b40831eeb444669ae876e2996f0dda9f05cd83e81b314b8dfca51e4890cca95d"
NETWORK_NAMES = ("case33", "cigre_residential", "paper11", "paper15")

# Numbers transcribed from create_cigre_network_lv in the pinned public source.
# The complete residential section is retained, including its R1 internal load.
_CIGRE_LINES = (
    (1, 2, .035, "UG1"), (2, 3, .035, "UG1"),
    (3, 4, .035, "UG1"), (4, 5, .035, "UG1"),
    (5, 6, .035, "UG1"), (6, 7, .035, "UG1"),
    (7, 8, .035, "UG1"), (8, 9, .035, "UG1"),
    (9, 10, .035, "UG1"), (3, 11, .030, "UG3"),
    (4, 12, .035, "UG3"), (12, 13, .035, "UG3"),
    (13, 14, .035, "UG3"), (14, 15, .030, "UG3"),
    (6, 16, .030, "UG3"), (9, 17, .030, "UG3"),
    (10, 18, .030, "UG3"),
)
_CIGRE_TYPES = {"UG1": (.162, .0832, 1.), "UG3": (.822, .0847, 1.)}
_CIGRE_LOADS = ((1, .190, .062449980), (11, .01425, .004683748),
               (15, .0494, .016236995), (16, .05225, .017173744),
               (17, .03325, .010928746), (18, .04465, .014675745))
_CIGRE_TRAFO = dict(sn_mva=.5, vn_hv_kv=20., vn_lv_kv=.4,
    vkr_percent=1., vk_percent=4.123106, pfe_kw=0., i0_percent=0.,
    shift_degree=30., tap_pos=0.)


def _source(relative: str):
    path = ROOT / relative
    body = path.read_bytes()
    return ast.parse(body.decode("utf-8-sig")), dict(
        source_path=relative, source_sha256=hashlib.sha256(body).hexdigest(),
        source_bytes=len(body))


def _assignments(statements):
    result = {}
    for node in statements:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        else:
            continue
        if isinstance(target, ast.Name):
            result[target.id] = node.value
    return result


def _finalize(network):
    """Check physical preservation and emit one JSON-serializable schema."""
    buses, edges = network["buses"], network["branches"]
    ids = [row["bus_id"] for row in buses]
    if len(ids) != len(set(ids)) or network["root_bus"] not in ids:
        raise ValueError("unique physical bus IDs and present root required")
    graph = {i: [] for i in ids}
    for edge in edges:
        if not edge.get("status", True):
            raise ValueError("branches must contain only the selected closed tree")
        u, v = edge["from_bus"], edge["to_bus"]
        if u not in graph or v not in graph or u == v:
            raise ValueError("invalid physical branch endpoints")
        if any(not math.isfinite(edge[k]) or edge[k] < 0 for k in ("r_ohm", "x_ohm")):
            raise ValueError("finite nonnegative physical R/X required")
        graph[u].append(v); graph[v].append(u)
    reached = {network["root_bus"]}
    stack = list(reached)
    while stack:
        for v in graph[stack.pop()]:
            if v not in reached:
                reached.add(v); stack.append(v)
    if len(edges) != len(ids) - 1 or reached != set(ids):
        raise ValueError("connected radial physical network required")
    observations = [row["bus_id"] for row in buses
                    if row["bus_id"] != network["root_bus"]
                    and (row["pd_kw"] != 0 or row["qd_kvar"] != 0)]
    network.update(schema="physical_reference_tree_v1", observation_bus_ids=observations,
        base_kw=1000. * network["base_mva"],
        total_pd_kw=sum(row["pd_kw"] for row in buses),
        total_qd_kvar=sum(row["qd_kvar"] for row in buses))
    # A physical interface is exported explicitly; the experiment may declare
    # a different interface before freezing but cannot let each fit move it.
    network["source_bus"] = network["source_bus_suggestion"]
    network["metadata"].update(physical_bus_count=len(ids), physical_edge_count=len(edges),
        observation_count=len(observations),
        observed_internal_buses=[i for i in observations if len(graph[i]) > 1],
        zero_impedance_branches=[edge["branch_id"] for edge in edges
                                 if edge["r_ohm"] == edge["x_ohm"] == 0.],
        synthetic_positive_service_edges_added=0,
        static_loads_are_measurement_time_series=False)
    network["physical_sha256"] = hashlib.sha256(json.dumps(
        dict(buses=buses, branches=edges, root_bus=network["root_bus"],
             base_kv=network["base_kv"], base_mva=network["base_mva"]),
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return network


def build_case33():
    """Complete 33-bus Baran-Wu closed tree; all 32 original loads retained."""
    tree, record = _source("terminal_case33/data/case33bw_raw.py")
    assignments = _assignments(tree.body)
    loads = ast.literal_eval(assignments["_BUS_LOADS"])
    edges = ast.literal_eval(assignments["_CLOSED_BRANCHES"])
    ties = ast.literal_eval(assignments["_TIE_BRANCHES"])
    return _finalize(dict(name="case33", root_bus=1, base_kv=12.66, base_mva=10.,
        buses=[dict(bus_id=i, pd_kw=float(p), qd_kvar=float(q), nominal_kv=12.66,
                    source_label=str(i)) for i, p, q in loads],
        branches=[dict(branch_id=j, from_bus=u, to_bus=v, r_ohm=float(r),
            x_ohm=float(x), status=True, branch_type="original_closed_line",
            length_km=None, rating_a=None) for j, (u, v, r, x) in enumerate(edges, 1)],
        source_bus_suggestion=2,
        provenance=dict(**record, data_kind="complete_standard_network_static_parameters",
            source_url=CASE33_URL, publication_doi="10.1109/61.25627",
            upstream_source_url=CASE33_UPSTREAM_URL,
            upstream_source_sha256=CASE33_UPSTREAM_SHA256,
            source_version="MATPOWER 7.1", original_source_base_mva=10.,
            local_power_base_mva=10., power_base_rebased=False,
            R_X_unit="ohm", P_Q_unit="kW_kvar", all_original_loads_retained=True,
            modifications=["closed radial state retained; five normally open ties excluded",
                           "original MATPOWER 7.1 10 MVA power base retained; physical ohms and kW unchanged"],
            field_PQV=False, load_time_series_source=None),
        metadata=dict(normally_open_ties=[dict(from_bus=u, to_bus=v, r_ohm=r,
            x_ohm=x, status=False) for u, v, r, x in ties],
            declared_terminalization="none", original_internal_loads_moved=0,
            ev_connection_note="MV bus injections represent LV service transformed to this base")))


def build_cigre_residential():
    """Full CIGRE residential section, with original transformer series Z.

    Bus R0's 20 kV reference is referred to 0.4 kV. The physical ideal ratio
    and 30-degree phase shift are removed by this balanced common-base change;
    zero no-load loss/current and neutral tap make the series model exact for
    voltage magnitudes, P/Q and losses in this balanced benchmark. It is not
    an unbalanced four-wire model. R1's 190 kW load is not removed or moved.
    """
    tr = deepcopy(_CIGRE_TRAFO)
    zbase = tr["vn_lv_kv"] ** 2 / tr["sn_mva"]
    rpu = tr["vkr_percent"] / 100.
    xpu = math.sqrt((tr["vk_percent"] / 100.) ** 2 - rpu ** 2)
    load_by_bus = {i: (p * 1000., q * 1000.) for i, p, q in _CIGRE_LOADS}
    buses = [dict(bus_id=i, pd_kw=load_by_bus.get(i, (0., 0.))[0],
        qd_kvar=load_by_bus.get(i, (0., 0.))[1], nominal_kv=.4,
        original_nominal_kv=20. if i == 0 else .4, source_label=f"R{i}")
        for i in range(19)]
    edges = [dict(branch_id=1, from_bus=0, to_bus=1, r_ohm=rpu * zbase,
        x_ohm=xpu * zbase, status=True, branch_type="physical_transformer_lv_equivalent",
        length_km=None, rating_a=tr["sn_mva"] * 1000. / (math.sqrt(3.) * .4),
        transformer=tr)]
    for j, (u, v, length, kind) in enumerate(_CIGRE_LINES, 2):
        r, x, max_i_ka = _CIGRE_TYPES[kind]
        edges.append(dict(branch_id=j, from_bus=u, to_bus=v,
            r_ohm=r * length, x_ohm=x * length, status=True,
            branch_type=kind, length_km=length, rating_a=max_i_ka * 1000.,
            shunt_capacitance_nf_per_km=0.))
    return _finalize(dict(name="cigre_residential", root_bus=0, base_kv=.4,
        base_mva=.5, source_bus_suggestion=1, buses=buses, branches=edges,
        provenance=dict(data_kind="published_standard_residential_subnetwork",
            source_url=CIGRE_URL, source_sha256=CIGRE_SHA256,
            source_version="pandapower v3.2.1 create_cigre_network_lv",
            original_reference="CIGRE Task Force C6.04.02 benchmark systems, 2014",
            read_locations="residential bus/line/load tables and R0-R1 transformer",
            R_X_unit="ohm_referred_to_LV", P_Q_unit="kW_kvar",
            all_residential_loads_retained=True, field_PQV=False,
            load_time_series_source=None, modifications=[
                "select complete residential section; industrial/commercial sections excluded",
                "contract closed zero-impedance MV switch to R0 reference",
                "refer transformer primary reference to LV; remove ideal ratio/phase shift",
                "retain transformer series impedance, no-load zero loss and neutral tap"]),
        metadata=dict(transformer=tr, original_MV_root_kv=20.,
            referred_root_voltage_kv=.4, declared_terminalization="none",
            original_internal_loads_moved=0, omitted_sections=["industrial", "commercial"],
            all_original_residential_lines_retained=True,
            no_load_loss_kw=0., no_load_current_percent=0.,
            rating_note="published max_i_ka=1.0 line field; not an independently verified cable ampacity")))


def build_paper_compatibility(name):
    """Use old synthetic topology/static loads but undo its 5.x Z multiplier."""
    if name not in ("paper11", "paper15"):
        raise ValueError("paper compatibility case must be paper11 or paper15")
    relative = ("rnj_wzzt_core/rnj_wzzt/data/paper_style_case_bank.py" if name == "paper11"
                else "rnj_wzzt_core/rnj_wzzt/data/paper_style_terminal_lv.py")
    tree, record = _source(relative)
    function_name = ("build_soumalas_style_lv_11_case" if name == "paper11"
                     else "build_paper_style_terminal_lv_case")
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name)
    assigned = _assignments(function.body)
    loads = ast.literal_eval(assigned["terminals" if name == "paper11" else "terminal_specs"])
    hidden = ast.literal_eval(assigned["hidden"])
    if name == "paper11":
        lines = ast.literal_eval(assigned["edges"])
        call = next(n.value for n in function.body if isinstance(n, ast.Return))
        old_scale = ast.literal_eval(next(k.value for k in call.keywords if k.arg == "impedance_scale"))
    else:
        old_scale = ast.literal_eval(assigned["impedance_scale"])
        # Each tuple has (u,v,*z(r_per_km,x_per_km,length_m),kind,length_m).
        lines = []
        for item in assigned["edge_specs"].elts:
            u, v = (ast.literal_eval(item.elts[i]) for i in (0, 1))
            length = ast.literal_eval(item.elts[4])
            kind = ast.literal_eval(item.elts[3])
            lines.append((u, v, length, kind))
    buses = [dict(bus_id=1, pd_kw=0., qd_kvar=0., nominal_kv=.4, source_label="root")]
    buses += [dict(bus_id=i, pd_kw=0., qd_kvar=0., nominal_kv=.4,
                   source_label=f"hidden_{i}") for i in hidden]
    buses += [dict(bus_id=i, pd_kw=float(p), qd_kvar=float(q), nominal_kv=.4,
                   source_label=str(i)) for i, p, q in loads]
    edges = []
    for j, (u, v, length_m, kind) in enumerate(lines, 1):
        r, x = (.55, .38) if kind == "service" else (.42, .34)
        edges.append(dict(branch_id=j, from_bus=u, to_bus=v,
            r_ohm=r * length_m / 1000., x_ohm=x * length_m / 1000.,
            status=True, branch_type=kind, length_km=length_m / 1000., rating_a=None))
    return _finalize(dict(name=name, root_bus=1, base_kv=.4, base_mva=.2,
        buses=buses, branches=edges, source_bus_suggestion=1,
        provenance=dict(**record, data_kind="project_synthetic_paper_style_compatibility",
            source_url=None, original_paper_parameters=False,
            old_impedance_multiplier=old_scale, new_impedance_multiplier=1.,
            modifications=[f"undo original synthetic impedance multiplier {old_scale}",
                           "retain exact previous topology, physical lengths and static P/Q"],
            field_PQV=False, load_time_series_source=None,
            line_parameter_note="synthetic R/X-balanced line library; not original authors' cable data"),
        metadata=dict(declared_terminalization="none", original_internal_loads_moved=0,
            upstream_common_positive_line_added=False)))


def build_network(name):
    """Return a new physical-tree dict. No selector or simulator data enter."""
    aliases = {"case33bw": "case33", "cigre_lv_residential": "cigre_residential",
               "soumalas11_unscaled": "paper11", "paper11_unscaled": "paper11",
               "paper15_unscaled": "paper15"}
    name = aliases.get(name, name)
    if name == "case33":
        return build_case33()
    if name == "cigre_residential":
        return build_cigre_residential()
    if name in ("paper11", "paper15"):
        return build_paper_compatibility(name)
    raise ValueError(f"unknown reference network: {name!r}")


def verify_cigre_source(path):
    """Independent AST check against the complete frozen upstream source bytes.

    Only literal call arguments are inspected; the downloaded code is never
    imported or executed. Returns the verified byte hash and row counts.
    """
    body = Path(path).read_bytes()
    if hashlib.sha256(body).hexdigest() != CIGRE_SHA256:
        raise ValueError("CIGRE upstream source byte hash mismatch")
    tree = ast.parse(body.decode("utf-8-sig"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "create_cigre_network_lv")
    lines, loads, transformer = [], [], None
    types, last_data = {}, None
    for node in fn.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "line_data":
                last_data = ast.literal_eval(node.value)
        call = node.value if isinstance(node, ast.Expr) else None
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        kw = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
        if call.func.id == "create_std_type":
            kind = kw.get("name")
            if kind in _CIGRE_TYPES:
                types[kind] = (last_data["r_ohm_per_km"], last_data["x_ohm_per_km"], last_data["max_i_ka"])
        if len(call.args) < 2 or not isinstance(call.args[1], ast.Name):
            continue
        name = call.args[1].id
        if not name.startswith("busR"):
            continue
        i = int(name[4:])
        if call.func.id == "create_line":
            j = int(call.args[2].id[4:])
            lines.append((i, j, kw["length_km"], kw["std_type"]))
        elif call.func.id == "create_load":
            loads.append((i, kw["p_mw"], kw["q_mvar"]))
        elif call.func.id == "create_transformer_from_parameters":
            transformer = {k: kw[k] for k in _CIGRE_TRAFO}
    if tuple(lines) != _CIGRE_LINES or tuple(loads) != _CIGRE_LOADS:
        raise ValueError("CIGRE selected line/load table differs from upstream")
    if types != _CIGRE_TYPES or transformer != _CIGRE_TRAFO:
        raise ValueError("CIGRE electrical library or transformer differs from upstream")
    return dict(source_sha256=CIGRE_SHA256, residential_line_count=len(lines),
                residential_load_count=len(loads), all_tables_match=True)


def verify_case33_source(path):
    """Check baseMVA, every load and closed branch against MATPOWER 7.1."""
    body = Path(path).read_bytes()
    if hashlib.sha256(body).hexdigest() != CASE33_UPSTREAM_SHA256:
        raise ValueError("case33 upstream source byte hash mismatch")
    source = body.decode("utf-8-sig")
    def block(name):
        match = re.search(r"mpc\." + name + r"\s*=\s*\[(.*?)\];", source, re.S)
        if match is None:
            raise ValueError("missing MATPOWER source table")
        result = []
        for line in match.group(1).splitlines():
            line = line.split("%")[0].replace(";", " ").strip()
            if line:
                result.append([float(value) for value in line.split()])
        return result
    net = build_case33()
    base_match = re.search(r"^\s*mpc\.baseMVA\s*=\s*([0-9.eE+-]+)\s*;", source, re.M)
    if base_match is None:
        raise ValueError("missing MATPOWER source baseMVA")
    source_base_mva = float(base_match.group(1))
    if source_base_mva != net["base_mva"]:
        raise ValueError("case33 baseMVA differs from pinned upstream")
    loads = [(int(row[0]), row[2], row[3]) for row in block("bus")]
    expected_loads = [(row["bus_id"], row["pd_kw"], row["qd_kvar"]) for row in net["buses"]]
    branches = [(int(row[0]), int(row[1]), row[2], row[3])
                for row in block("branch") if row[10] == 1]
    expected_branches = [(row["from_bus"], row["to_bus"], row["r_ohm"], row["x_ohm"])
                         for row in net["branches"]]
    if loads != expected_loads or branches != expected_branches:
        raise ValueError("case33 physical table differs from upstream")
    return dict(source_sha256=CASE33_UPSTREAM_SHA256,
                source_base_mva=source_base_mva, local_base_mva=net["base_mva"],
                power_base_rebased=False, bus_count=len(loads),
                closed_branch_count=len(branches), all_tables_match=True)
