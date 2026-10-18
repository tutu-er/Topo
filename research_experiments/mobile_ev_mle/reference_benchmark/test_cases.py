"""Boundary and physical conservation checks on complete reference networks."""
from copy import deepcopy
import numpy as np
import pytest

from .cases import CHANNELS, native_network, make_case, planning_inputs, actual_observation
from .networks import NETWORK_NAMES, build_network


def _inputs(name, mode="PV", *, h=8, t=4):
    network = native_network(name)
    k = np.arange(h + t)[:, None]
    factors = .65 + .03 * k + .01 * np.arange(6)[None, :]
    window = dict(index=0, phase="development", timestamps=[f"time_{i}" for i in range(h + t)],
        p_relative=factors.tolist(), q_relative=(.8 + .1 * factors).tolist(),
        pv_relative=np.linspace(0., .8, h + t).tolist(), history_rows=h, horizon_rows=t,
        forecast_cutoff_utc=f"time_{h-1}")
    prepared = dict(profile_names=[f"profile_{i}" for i in range(6)], assignments={name:{
        str(bus):f"profile_{j%6}" for j,bus in enumerate(network["observation_bus_ids"])}})
    config = dict(H=h, T=t, fit_multistart=2, fit_max_nfev=100, fit_tol=1e-8)
    spec = dict(network=name, source_mode=mode, seed=1200, physical_seed=777)
    return spec, window, prepared, config


def _white(t,n):
    return {k:np.zeros((t,n) if k in ("p","q","v") else t) for k in CHANNELS}


@pytest.mark.parametrize("name", NETWORK_NAMES)
def test_native_full_networks_with_future_EV_match_independent_physical_AC(name):
    c = make_case(*_inputs(name))
    h,t,n=c["H"],c["T"],c["history"].n
    assert c["physics_audit"]["history"]["passed"]
    power=np.zeros((t,n));power[:,0]=[0.,3.,-1.,2.]
    obs,physical,bg=actual_observation(c,power,_white(t,n))
    assert physical["reference_audit"]["passed"]
    assert physical["reference_audit"]["voltage_max_error_pu"] < 2e-10
    np.testing.assert_allclose(obs.p*c["base_kw"],c["p"][h:]*c["base_kw"]+power)
    np.testing.assert_allclose(obs.q,c["q"][h:])
    np.testing.assert_array_equal(bg["p"],c["p"][h:])
    assert c["provenance"]["EV_Q"] == 0.
    assert c["fit_options"]["r_bounds"][0] == c["fit_options"]["x_bounds"][0] == 0.


@pytest.mark.parametrize("name", ("paper11","paper15"))
def test_paper_extension_is_explicit_without_changing_any_old_line(name):
    original=build_network(name);new=native_network(name)
    assert len(new["buses"])==len(original["buses"])+1
    assert new["source_bus"]==original["root_bus"]
    assert new["root_bus"] != original["root_bus"]
    assert new["branches"][1:]==original["branches"]
    assert new["provenance"]["added_transformer"]["rating_is_scenario_choice"]
    assert not new["provenance"]["added_transformer"]["original_paper_asset"]


def test_original_nominal_units_and_profiles_are_not_history_renormalized():
    args=_inputs("case33");c=make_case(*args)
    n=c["network"];buses={b["bus_id"]:b for b in n["buses"]}
    assert c["base_kw"]==10000.
    for j,bus in enumerate(c["observation_bus_ids"]):
        factor=np.array(args[1]["p_relative"])[:,j%6]
        np.testing.assert_allclose(c["p"][:,j]*c["base_kw"],buses[bus]["pd_kw"]*factor)
    assert c["provenance"]["scaling"]["history_normalization"] is False
    assert 2 in c["observation_bus_ids"] and 30 in c["observation_bus_ids"]


def test_source_is_PQ_at_same_interface_and_unknown_domain_does_not_use_mode():
    pv=make_case(*_inputs("cigre_residential","PV"))
    none=make_case(*_inputs("cigre_residential","none"))
    np.testing.assert_allclose(pv["source_q"],.2*pv["source_p"])
    assert np.min(pv["source_p"]) < 0
    assert np.all(none["source_p"]==0) and np.all(none["source_q"]==0)
    assert pv["fit_options"]==none["fit_options"]
    assert pv["provenance"]["source"]["physical_bus"]==1
    cap=min(.3*pv["nominal_p_kw"].sum(),80.)
    assert pv["provenance"]["source"]["active_capacity_kw"]==cap
    assert pv["provenance"]["source"]["apparent_power_box_envelope_kva"] >= np.hypot(1.1*cap,.4*cap)-1e-12


def test_future_truth_and_source_label_do_not_enter_planning_allowlist():
    c=make_case(*_inputs("paper11"));original=planning_inputs(c)
    other=deepcopy(c)
    for key in ("p","q","source_p","source_q","root_v"):
        other[key][c["H"]:]+=999
    other["truth"]["r"]*=20
    other["spec"]["source_mode"]="evaluation_oracle"
    newer=planning_inputs(other)
    assert set(newer)==set(original)
    for key in ("p","q","v","master_p","master_q"):
        np.testing.assert_array_equal(getattr(newer["history"],key),getattr(original["history"],key))
    forbidden={"truth","network","p","q","source_p","source_q","source_mode","spec","future_timestamps","physics_audit","encoded_tree","provenance"}
    assert not forbidden.intersection(newer)
    assert set(newer)=={"history","history_meter","future_meter","timestamps","horizon","base_kw","public_parameters"}
    assert set(newer["public_parameters"])=={"base_kv","base_kw","observation_bus_ids","units","site_interpretation"}
    assert "nominal_p_kw" not in newer and "nominal_q_kvar" not in newer


def test_history_future_identical_meter_precision_and_zero_bias():
    c=make_case(*_inputs("case33"))
    for channel in CHANNELS:
        np.testing.assert_array_equal(getattr(c["history_meter"],channel+"_sigma"),getattr(c["future_meter"],channel+"_sigma"))
    assert c["history_meter"].v_sigma==.0005
    assert c["history_meter"].fit_bias is False
    np.testing.assert_allclose(c["history_meter"].p_sigma*c["base_kw"],.002*np.maximum(c["nominal_p_kw"],1.))


def test_white_noise_applied_once_and_clean_is_evaluation_only():
    c=make_case(*_inputs("paper11"));t,n=c["T"],c["history"].n
    white={k:np.ones_like(v) for k,v in _white(t,n).items()}
    noisy,physical,_=actual_observation(c,np.zeros((t,n)),white)
    clean,_,_=actual_observation(c,np.zeros((t,n)),white,clean=True)
    np.testing.assert_allclose(noisy.v-clean.v,c["future_meter"].v_sigma)
    np.testing.assert_array_equal(clean.v,physical["v"])
    assert "clean" not in planning_inputs(c)


def test_public_line_upper_uses_entire_inventory_not_true_paths():
    c=make_case(*_inputs("case33"))
    domain=c["provenance"]["fit_domain"]
    assert domain["r_ohm_upper"]==sum(b["r_ohm"] for b in c["network"]["branches"])
    assert domain["x_ohm_upper"]==sum(b["x_ohm"] for b in c["network"]["branches"])
    assert not domain["actual_virtual_edge_pattern_used"]


def test_signed_net_P_and_capacitive_Q_are_preserved_without_clipping():
    spec,w,p,cfg=_inputs("paper11")
    w["p_relative"][1][0]=-.1
    w["q_relative"][2][1]=-.2
    c=make_case(spec,w,p,cfg)
    assert c["p"][1,0] < 0.
    assert c["q"][2,1] < 0.


@pytest.mark.parametrize("broken",("shape","assignment","pv"))
def test_invalid_prepared_input_fails_explicitly(broken):
    spec,w,p,cfg=_inputs("paper11")
    if broken=="shape":w["p_relative"]=w["p_relative"][:-1]
    elif broken=="assignment":p["assignments"]["paper11"].pop("101")
    else:w["pv_relative"][0]=1.3
    with pytest.raises(ValueError):make_case(spec,w,p,cfg)
