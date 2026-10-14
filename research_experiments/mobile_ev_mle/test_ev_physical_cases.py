"""Independent units and provenance checks; no new AC strategy experiment."""
import numpy as np
import pytest
from .ev_physical_cases import (BASE_KW, case33_local_standard, case33_local_record,
                               power_kw_to_pu, impedance_ohm_to_pu)


def test_standard_source_mapping_matches_solver_order():
    case = case33_local_standard()
    expected = {(0, 1, 2): ((1, 2), .0922, .047), (0, 2): ((2, 3), .493, .2511),
                (0,): ((3, 4), .366, .1864), (1,): ((2, 19), .164, .1565),
                (2,): ((3, 23), .4512, .3083)}
    assert case.candidate.supports[0] == (0, 1, 2)
    assert set(map(tuple, case.candidate.supports)) == set(expected)
    for i, support in enumerate(case.candidate.supports):
        edge, r, x = expected[tuple(support)]
        record = case.provenance["ordered_edges"][i]
        assert tuple(record["source_edge"]) == edge
        assert record["support"] == list(support)
        assert case.r[i] * (12.66 ** 2) == pytest.approx(r)
        assert case.x[i] * (12.66 ** 2) == pytest.approx(x)


def test_load_units_and_realistic_EV_power_scale():
    case = case33_local_standard()
    assert case.base_kw == BASE_KW == 1000.0
    assert case.base_mva == 1.0 and case.zbase_ohm == pytest.approx(160.2756)
    np.testing.assert_allclose(case.p0, [.12, .09, .09], rtol=0, atol=1e-15)
    np.testing.assert_allclose(case.q0, [.08, .04, .05], rtol=0, atol=1e-15)
    assert power_kw_to_pu(6.0) == pytest.approx(.006)
    assert case.provenance["old_synthetic_6kw_pu"] == pytest.approx(.3)
    np.testing.assert_allclose(case.provenance["ev_6kw_to_terminal_load_ratios"], [.05, 1/15, 1/15])
    assert case.provenance["old_synthetic_6kw_to_nominal_terminal_load_ratio"] == pytest.approx(2.5)


def test_consistent_rebase_preserves_first_order_physical_voltage_scale():
    r_ohm, p_kw, v_kv = .493, 6., 12.66
    products = []
    for mva in (.02, 1., 10.):
        products.append(float(impedance_ohm_to_pu(r_ohm, base_kv=v_kv, base_mva=mva)
                              * power_kw_to_pu(p_kw, base_kw=mva * 1000.)))
    np.testing.assert_allclose(products, [r_ohm * p_kw / (1000. * v_kv ** 2)] * 3, rtol=1e-14)


def test_provenance_preserves_model_changes_and_does_not_create_floor():
    record = case33_local_record()
    assert record["data_kind"] == "standard_parameters_derived_simulation"
    assert record["not_complete_original_feeder"] and not record["field_PQV_available"]
    assert record["fitted_impedance_lower_bound_from_assets"] is None
    assert record["impedance_scale_factor"] == 1.0
    assert record["omitted_branch_count"] == 27
    assert sum(v["removed_p_kw"] for v in record["hidden_loads_zeroed"]) == 190.0
    assert sum(v["removed_q_kvar"] for v in record["hidden_loads_zeroed"]) == 100.0
    assert len(record["source_sha256"]) == len(record["source_lf_sha256"]) == 64


@pytest.mark.parametrize("base_kw", [0., -1., float("nan"), float("inf")])
def test_power_conversion_rejects_invalid_base(base_kw):
    with pytest.raises(ValueError):
        power_kw_to_pu(6., base_kw=base_kw)


def test_ohm_conversion_rejects_nonfinite_impedance():
    with pytest.raises(ValueError):
        impedance_ohm_to_pu([.1, float("nan")])
