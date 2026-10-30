import numpy as np
from .coded_shapes import pair_template,contract_quantities,atom_response,complementary_inputs,feasible_on_declared_device_band


def test_energy_and_power_match_original_contract():
    for kind in ('same','opposed','alternating_opposed'):
        q=contract_quantities(pair_template(kind))
        assert np.allclose(q['max_power_kw'],.2)
        assert np.allclose(q['reduced_energy_kWh'],.2)
        assert np.allclose(q['added_energy_kWh'],.2)
        assert np.allclose(q['net_energy_kWh'],0)
    reserve=np.full((8,2),.2)
    assert feasible_on_declared_device_band(pair_template(),np.ones((8,2)),3.6,-reserve,reserve)
    assert not feasible_on_declared_device_band(pair_template(),np.zeros((8,2)),3.6,-reserve,reserve)


def test_common_ancestor_cancel_and_partial_response_leak():
    profile=pair_template('opposed')
    assert np.allclose(atom_response([0,1],profile),0)
    partial=profile*np.array([1,0])
    assert np.max(abs(atom_response([0,1],partial)))==.2
    assert np.max(abs(atom_response([0],profile)))==.2


def test_two_rounds_avoid_exciting_only_one_crossing_direction():
    c,d={1,2},{2,3,4}
    inputs,_=complementary_inputs(c,d,6)
    assert np.allclose(inputs.sum(axis=1),0)
    assert np.allclose(inputs[:,sorted(c)].sum(axis=1),[0,1])
    assert np.allclose(inputs[:,sorted(d)].sum(axis=1),[1,0])
    assert np.allclose(atom_response(range(6),inputs),0)
