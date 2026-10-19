import numpy as np
import pytest

from .ev_lv_parameter_family import public_lv_parameter_family


def test_pinned_public_matrices_and_explicit_meters_convert_to_pu():
    r = public_lv_parameter_family([50, 100], ["C201", "C250"])
    np.testing.assert_allclose(r["r_ohm_per_km"], [1.29408, .145584])
    np.testing.assert_allclose(r["x_ohm_per_km"], [.3188825754725975, .2486104937994243])
    assert r["zbase_ohm"] == pytest.approx(8.)
    np.testing.assert_allclose(r["r"], [1.29408 * .05 / 8, .145584 * .1 / 8])
    np.testing.assert_allclose(r["x"], np.asarray(r["x_ohm_per_km"]) * [.05, .1] / 8)
    p = r["provenance"]
    assert not p["field_pqv"] and not p["fitted_impedance_floor_derived"]
    assert max(p["balanced_relative_errors"]) < 1e-12
    assert len(p["source_sha256"]) == 64


def test_impedance_tends_to_zero_as_individual_positive_length_decreases():
    full = public_lv_parameter_family([100], ["C201"])
    short = public_lv_parameter_family([1e-6], ["C201"])
    assert 0 < short["r"][0] < 1e-9
    assert short["r"][0] == pytest.approx(full["r"][0] * 1e-8)
    assert short["x"][0] == pytest.approx(full["x"][0] * 1e-8)


def test_power_and_impedance_base_changes_preserve_physical_voltage_scale():
    a = public_lv_parameter_family([50], ["C201"], base_mva=.02)
    b = public_lv_parameter_family([50], ["C201"], base_mva=.04)
    assert a["r"][0] * (6 / a["base_kw"]) == pytest.approx(
        b["r"][0] * (6 / b["base_kw"]))


def test_asymmetric_public_matrix_requires_four_wire_model():
    with pytest.raises(ValueError, match="four-wire"):
        public_lv_parameter_family([50], ["OH1"])


@pytest.mark.parametrize("length,codes", [([0], ["C201"]), ([-1], ["C201"]),
                                         ([float("nan")], ["C201"]),
                                         ([50], ["missing"])])
def test_missing_or_invalid_public_parameters_rejected(length, codes):
    with pytest.raises(ValueError):
        public_lv_parameter_family(length, codes)
