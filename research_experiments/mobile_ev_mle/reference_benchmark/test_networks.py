"""Physical-data conservation and upstream-source checks; no AC/solver calls."""
import math
import hashlib
import pytest

from . import networks
from .networks import (ROOT, NETWORK_NAMES, build_network, verify_cigre_source,
                       verify_case33_source, CIGRE_SHA256, _finalize)


@pytest.mark.parametrize("name", NETWORK_NAMES)
def test_physical_graph_is_tree_and_every_load_is_observed(name):
    n = build_network(name)
    assert len(n["branches"]) == len(n["buses"]) - 1
    assert len(n["physical_sha256"]) == 64
    assert n["base_kw"] == 1000 * n["base_mva"]
    loads = {b["bus_id"] for b in n["buses"] if b["pd_kw"] or b["qd_kvar"]}
    assert set(n["observation_bus_ids"]) == loads - {n["root_bus"]}
    assert n["metadata"]["synthetic_positive_service_edges_added"] == 0
    assert not n["metadata"]["static_loads_are_measurement_time_series"]


def test_case33_keeps_full_original_parameters_and_internal_loads():
    n = build_network("case33")
    assert (len(n["buses"]), len(n["branches"]), len(n["observation_bus_ids"])) == (33, 32, 32)
    assert n["total_pd_kw"] == 3715.
    assert n["total_qd_kvar"] == 2300.
    assert n["base_kv"] == 12.66
    assert n["base_mva"] == 10.
    assert n["provenance"]["original_source_base_mva"] == 10.
    assert n["provenance"]["local_power_base_mva"] == 10.
    assert n["provenance"]["power_base_rebased"] is False
    assert n["provenance"]["source_version"] == "MATPOWER 7.1"
    assert n["provenance"]["source_url"] == networks.CASE33_UPSTREAM_URL
    assert n["branches"][0]["r_ohm"] == .0922
    assert n["branches"][0]["x_ohm"] == .0470
    assert 2 in n["metadata"]["observed_internal_buses"]
    assert 30 in n["metadata"]["observed_internal_buses"]
    assert next(b for b in n["buses"] if b["bus_id"] == 30)["qd_kvar"] == 600.
    assert len(n["metadata"]["normally_open_ties"]) == 5
    assert n["provenance"]["all_original_loads_retained"]


def test_cigre_retains_transformer_and_internal_aggregate_load():
    n = build_network("cigre_residential")
    assert (len(n["buses"]), len(n["branches"])) == (19, 18)
    assert n["observation_bus_ids"] == [1, 11, 15, 16, 17, 18]
    assert n["metadata"]["observed_internal_buses"] == [1]
    b = next(b for b in n["buses"] if b["bus_id"] == 1)
    assert b["pd_kw"] == 190.
    assert b["qd_kvar"] == pytest.approx(62.449980)
    tr = n["branches"][0]
    assert tr["from_bus"] == 0 and tr["to_bus"] == 1
    assert tr["r_ohm"] == pytest.approx(.0032)
    assert tr["x_ohm"] == pytest.approx(math.sqrt((4.123106 / 100.) ** 2 - .01 ** 2) * .32)
    assert n["total_pd_kw"] == pytest.approx(383.8)
    assert n["metadata"]["original_internal_loads_moved"] == 0
    assert n["provenance"]["source_sha256"] == CIGRE_SHA256


@pytest.mark.parametrize("name,count,scale,first_r", [
    ("paper11", 11, 5.2, .42 * .064),
    ("paper15", 15, 5., .42 * .092),
])
def test_compatibility_undoes_only_impedance_multiplier(name, count, scale, first_r):
    n = build_network(name)
    assert len(n["observation_bus_ids"]) == count
    assert n["branches"][0]["r_ohm"] == pytest.approx(first_r)
    assert n["provenance"]["old_impedance_multiplier"] == scale
    assert n["provenance"]["new_impedance_multiplier"] == 1.
    assert not n["provenance"]["original_paper_parameters"]
    assert not n["metadata"]["upstream_common_positive_line_added"]


def test_exports_are_independent_mutable_copies():
    a = build_network("cigre_residential")
    a["branches"][0]["transformer"]["sn_mva"] = 99.
    a["buses"][1]["pd_kw"] = -99.
    b = build_network("cigre_residential")
    assert b["branches"][0]["transformer"]["sn_mva"] == .5
    assert b["buses"][1]["pd_kw"] == 190.


def test_closed_tree_validation_rejects_lost_edge():
    n = build_network("case33")
    n["branches"].pop()
    with pytest.raises(ValueError, match="connected radial"):
        _finalize(n)


def test_zero_physical_edge_is_allowed_without_perturbation():
    n = build_network("case33")
    n["branches"][0]["r_ohm"] = n["branches"][0]["x_ohm"] = 0.
    n = _finalize(n)
    assert n["metadata"]["zero_impedance_branches"] == [1]
    assert n["branches"][0]["r_ohm"] == n["branches"][0]["x_ohm"] == 0.


def test_upstream_cigre_source_literal_tables_match():
    source = ROOT / "audit/cigre_networks_pandapower_v3.2.1.py"
    if not source.exists():
        pytest.skip("optional original source byte fixture is archived with evidence")
    assert verify_cigre_source(source)["all_tables_match"]


def test_upstream_matpower_complete_tables_match():
    source = ROOT / "audit/case33bw_matpower_7.1.m"
    if not source.exists():
        pytest.skip("optional original source byte fixture is archived with evidence")
    receipt = verify_case33_source(source)
    assert receipt["all_tables_match"]
    assert receipt["source_base_mva"] == receipt["local_base_mva"] == 10.
    assert receipt["power_base_rebased"] is False


def test_upstream_matpower_base_is_parsed_not_assumed(tmp_path, monkeypatch):
    source = ROOT / "audit/case33bw_matpower_7.1.m"
    if not source.exists():
        pytest.skip("optional original source byte fixture is archived with evidence")
    body = source.read_bytes().replace(b"mpc.baseMVA = 10;", b"mpc.baseMVA = 100;")
    assert body != source.read_bytes()
    changed = tmp_path / "changed_base.m"
    changed.write_bytes(body)
    monkeypatch.setattr(networks, "CASE33_UPSTREAM_SHA256", hashlib.sha256(body).hexdigest())
    with pytest.raises(ValueError, match="baseMVA differs"):
        verify_case33_source(changed)


def test_bad_upstream_hash_is_not_silently_accepted(tmp_path):
    bad = tmp_path / "source.py"
    bad.write_text("# changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_cigre_source(bad)


def test_unknown_case_rejected():
    with pytest.raises(ValueError, match="unknown reference network"):
        build_network("favourable_custom_truth")
