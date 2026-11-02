"""Semantic checks for frozen DR resource declarations, without identification."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("resource_builder",HERE/"build_allocations.py")
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
CFG=json.loads((HERE/"library.json").read_text(encoding="utf-8"))


def fixture_nodes():
    return builder.allocate("fixture",[7,3,11],np.array([[2.,4.,6.],[3.,5.,7.]]),CFG)[0]


def test_column_order_does_not_choose_assets():
    p=np.array([[2.,4.,6.],[3.,5.,7.]])
    first,_=builder.allocate("fixture",[7,3,11],p,CFG)
    second,_=builder.allocate("fixture",[11,7,3],p[:,[2,0,1]],CFG)
    second={r["node_label"]:r for r in second}
    for r in first:
        assert r["mix"]==second[r["node_label"]]["mix"]
        assert r["inventory_kw"]==pytest.approx(second[r["node_label"]]["inventory_kw"])
        assert r["cost_factor"]==second[r["node_label"]]["cost_factor"]


def test_uniform_load_scaling_scales_power_and_energy():
    p=np.ones((3,96,3))*np.array([2.,4.,6.])
    first,_=builder.allocate("fixture",[7,3,11],p,CFG)
    second,_=builder.allocate("fixture",[7,3,11],5*p,CFG)
    for a,b in zip(first,second):
        for key in ("inventory_kw","ev_charger_kw","ev_grid_task_kwh","ev_battery_kwh","battery_energy_kwh"):
            assert b[key]==pytest.approx(5*a[key])


def test_no_invented_positive_floor_for_export_only_nodes():
    rows,stat=builder.allocate("fixture",[1,2],-np.ones((288,2)),CFG)
    assert stat["total_kw"]==0
    assert all(r["inventory_kw"]==0 for r in rows)
    assert all(all(builder.check_node(r,CFG).values()) for r in rows)


def test_inventory_is_not_symmetric_EV_headroom():
    n=fixture_nodes()[0];p=builder.profiles(n,CFG)
    expected=n["ev_charger_kw"]*20/(7*14)
    assert -p["delta_lower_kw"][2,0]==pytest.approx(expected)
    assert p["delta_upper_kw"][2,0]==pytest.approx(n["ev_charger_kw"]-expected)
    assert np.all(p["delta_upper_kw"][2,32:72]==0)
    assert np.all(p["delta_lower_kw"][2,32:72]==0)


def test_EV_midnight_and_departure_energies():
    n=fixture_nodes()[0];p=builder.profiles(n,CFG)
    morning=.25*sum(p["baseline_kw"][2,:32]);evening=.25*sum(p["baseline_kw"][2,72:])
    assert morning==pytest.approx(n["ev_grid_task_kwh"]*8/14)
    assert evening==pytest.approx(n["ev_grid_task_kwh"]*6/14)
    assert all(builder.check_node(n,CFG).values())


def test_battery_equal_charge_discharge_energy_is_not_terminal_neutral():
    n=fixture_nodes()[0];eta=np.sqrt(CFG["battery"]["roundtrip_efficiency"])
    p=n["battery_power_kw"]*.1
    assert .25*p*(eta-1/eta)<0
    assert .25*(eta*p/.85-p/eta)==pytest.approx(0,abs=1e-12)


def test_cost_and_uncertainty_vary_by_time_and_type():
    n=fixture_nodes()[0];p=builder.profiles(n,CFG)
    assert p["cost_full_activation"][0,76]>p["cost_full_activation"][0,4]
    assert p["relative_sd"][0,76]>p["relative_sd"][3,76]
    assert p["relative_sd"][2,28]==pytest.approx(1.5*p["relative_sd"][2,4])


def test_user_curtail_and_shift_are_different_domains():
    n=fixture_nodes()[0];p=builder.profiles(n,CFG)
    assert (p["delta_upper_kw"][1]==0).all()
    assert (p["delta_upper_kw"][0]>0).all()
    assert n["curtailable_kw"]/(n["curtailable_kw"]+n["shiftable_kw"])==pytest.approx(.25)


def test_cost_perspective_is_invariant_to_equal_share_split():
    A=2.;u=.7;c=.5
    original=c*u*u/A
    split=sum(c*(u*w)**2/(A*w) for w in [.2,.3,.5])
    assert original==pytest.approx(split)


def test_prescribed_covariance_is_positive_semidefinite():
    # Three resources, two at the same node; temporal marginal variance one.
    nodes=np.repeat([1,1,2],4);resources=np.repeat([0,1,2],4);slots=np.tile(np.arange(4),3)
    for key,phikey in [("main_variance_weights","main_ar1_phi"),("stress_variance_weights","stress_ar1_phi")]:
        a,b,c,d=CFG["response"][key];phi=CFG["response"][phikey]
        covariance=a+b*(nodes[:,None]==nodes[None,:])+(c+d*(resources[:,None]==resources[None,:]))*phi**np.abs(slots[:,None]-slots[None,:])
        np.testing.assert_allclose(np.diag(covariance),1)
        assert np.linalg.eigvalsh(covariance).min()>-1e-12


def test_invalid_training_cannot_silently_drop_nodes():
    with pytest.raises(ValueError):builder.allocate("fixture",[1,1],np.ones((10,2)),CFG)
    with pytest.raises(ValueError):builder.allocate("fixture",[1,2],np.array([[np.nan,1.]]),CFG)


def test_checks_are_json_serializable():
    for n in fixture_nodes():
        checks=builder.check_node(n,CFG)
        assert all(checks.values())
        json.dumps(checks,allow_nan=False)
