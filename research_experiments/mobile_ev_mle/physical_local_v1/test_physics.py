"""Independent physics and data-boundary tests; no MLE performance assertions."""
from copy import deepcopy
import numpy as np
import pytest

from .assets import (build_network, ratio_summary, public_lv_ratio_domain,
                     charging_q_bound_kvar, PAPER_NAMES, PROFILES)
from .context import rooted_scope, compress_zero_injection_series, port_reduction_oracle
from .experiment import trajectory, simulate, inference_payload, context_audit, standard_reference
from ..reference_benchmark.networks import build_network as legacy_network
from ..reference_benchmark.representation import physical_tree_ac


@pytest.mark.parametrize("name", PAPER_NAMES)
@pytest.mark.parametrize("profile", ("cable_v1","mixed_v1"))
def test_revised_assets_preserve_layout_loads_and_length(name,profile):
    old=build_network(name,"legacy_unscaled");new=build_network(name,profile)
    assert new["buses"]==old["buses"]
    assert new["root_bus"]==old["root_bus"]
    for a,b in zip(old["branches"],new["branches"]):
        for key in ("from_bus","to_bus","length_km","branch_type"):
            assert a[key]==b[key]
        assert b["physical_asset_source"].endswith("/v3.2.1/pandapower/std_types.py")
        assert b["rating_a"]>0
    ratios=ratio_summary(new);domain=public_lv_ratio_domain()
    assert ratios["distinct_ratio_count"]==3
    assert domain["ratio_min"]<=ratios["minimum"]<=ratios["maximum"]<=domain["ratio_max"]
    assert ratios["legacy_domain_excluded_edges"]==len(new["observation_bus_ids"])
    assert charging_q_bound_kvar(new)>0
    assert charging_q_bound_kvar(old) is None


@pytest.mark.parametrize("name", ("case33","cigre_residential"))
def test_published_standard_assets_remain_exactly_unchanged(name):
    assert build_network(name)==legacy_network(name)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("name", ("paper15","paper11"))
def test_root_context_retains_side_loads_and_exact_port_losses(name,profile):
    net=build_network(name,profile);data=trajectory(net,34991,total=12);state=simulate(net,data)
    record=context_audit(net,data,state)
    assert record["scope"]["root_bus"]==1
    assert record["scope"]["paths"][0][0]==1
    assert record["scope"]["side_subtree_ports"]
    assert record["candidate_count"]==4
    assert record["candidate_measured_bus_counts"]==[len(net["observation_bus_ids"])]*4
    assert record["all_candidate_context_supports_preserved"]
    assert record["exact_port_v_max_pu"]<1e-9
    assert record["exact_port_master_max_pu"]<1e-9
    assert record["correct_boundary_phase_aligned_max_pu"]<1e-9
    assert record["wrong_root_v_rmse_pu"]>record["naive_port_v_rmse_pu"]>1e-10
    assert record["exact_port_metadata"]["estimator_input_allowed"] is False
    payload=inference_payload(net,data,state,8)
    assert not set(payload).intersection(("r","x","truth","network","future","port"))
    assert len(payload["observed"]["p"])==8
    assert not np.array_equal(payload["root_meter"][0],data["root_v"][:8])


def test_series_reduction_is_exact_and_does_not_delete_an_injection_or_meter():
    net=build_network("cigre_residential");ids=net["observation_bus_ids"]
    by={b["bus_id"]:b for b in net["buses"]}
    p=np.array([[by[b]["pd_kw"]*.65 for b in ids]]);q=np.array([[by[b]["qd_kvar"]*.65 for b in ids]])
    a=physical_tree_ac(net["branches"],ids,net["root_bus"],net["base_kv"],net["base_kw"],p,q,net["source_bus"])
    reduced=compress_zero_injection_series(net)
    assert len(reduced["buses"])<len(net["buses"])
    assert set(ids)<=set(b["bus_id"] for b in reduced["buses"])
    b=physical_tree_ac(reduced["branches"],ids,net["root_bus"],net["base_kv"],net["base_kw"],p,q,net["source_bus"])
    np.testing.assert_allclose(a["measured_voltage"],b["measured_voltage"],atol=1e-10,rtol=0)
    np.testing.assert_allclose(a["master_p"]+1j*a["master_q"],b["master_p"]+1j*b["master_q"],atol=1e-10,rtol=0)
    protected=compress_zero_injection_series(net,protected_buses=[2])
    assert 2 in [z["bus_id"] for z in protected["buses"]]


def test_ordinary_paths_and_meter_noise_are_paired_across_asset_profiles():
    a=build_network("paper15","legacy_unscaled");b=build_network("paper15","cable_v1")
    da,db=trajectory(a,34991),trajectory(b,34991)
    for key in ("p_kw","q_kvar","root_v","pv_kw"):
        np.testing.assert_array_equal(da[key],db[key])
    for key in da["white"]:
        np.testing.assert_array_equal(da["white"][key],db["white"][key])
    np.testing.assert_array_equal(da["p_kw"][:,0],da["demand_kw"][:,0]-da["pv_kw"])


def test_nominal_case33_matches_the_published_base_and_xr_is_not_LV_uniform():
    record=standard_reference("case33")
    assert record["nominal_P_kw"]==3715
    assert record["nominal_Q_kvar"]==2300
    assert abs(record["voltage_min"]-.91309048)<1e-7
    assert record["ratios"]["maximum"]>3
    assert record["series_error_max_pu"]<1e-10


def test_bad_scope_and_asset_names_are_rejected():
    net=build_network("paper15")
    with pytest.raises(ValueError):rooted_scope(net,[999999])
    with pytest.raises(ValueError):build_network("paper15","random_target_separation")
