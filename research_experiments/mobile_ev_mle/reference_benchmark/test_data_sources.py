"""Unit and boundary checks independent of optional downloaded data."""
import csv
import hashlib
import io
import zipfile

import numpy as np
import pytest

from . import data_sources as ds


def _simbench(tmp_path):
    (tmp_path / "LoadProfile.csv").write_text(
        "time;H0-A_pload;H0-A_qload;G0-A_pload;G0-A_qload\n"
        "01.01.2016 00:00;0.2;0.3;0.7;0.9\n"
        "01.01.2016 00:15;0.4;0.8;0.6;0.1\n"
        "01.01.2016 00:30;0.1;0.6;0.5;0.2\n", encoding="utf-8")
    return tmp_path


def test_independent_pq_scaling_and_order(tmp_path):
    root = _simbench(tmp_path)
    profiles = ds.load_simbench_profiles(root, ["G0-A", "H0-A"])
    net = {"observation_bus_ids": [9, 3], "buses": [
        {"bus_id": 3, "pd_kw": 100., "qd_kvar": 60.},
        {"bus_id": 9, "pd_kw": 50., "qd_kvar": 5.}]}
    result = ds.apply_standard_profiles(net, profiles, {9: "H0-A", 3: "G0-A"})
    np.testing.assert_allclose(result["p_kw"][0], [10., 70.])
    np.testing.assert_allclose(result["q_kvar"][0], [1.5, 54.])
    assert result["dt_hours"] == .25
    assert result["provenance"]["unique_profile_count"] == 2


def test_duplicate_assignment_counted_not_independent(tmp_path):
    p = ds.load_simbench_profiles(_simbench(tmp_path), ["H0-A"])
    net = {"observation_bus_ids": [1, 2], "buses": [
        {"bus_id": i, "pd_kw": 10., "qd_kvar": 3.} for i in (1, 2)]}
    out = ds.apply_standard_profiles(net, p, {1: "H0-A", 2: "H0-A"})
    assert out["provenance"]["profile_reuse"] == {"H0-A": 2}
    with pytest.raises(ValueError):
        ds.load_simbench_profiles(tmp_path, ["H0-A", "H0-A"])
    with pytest.raises(ValueError):
        ds.apply_standard_profiles(net, p, {1: "H0-A"})


def test_usage_does_not_mislabel_ehv_as_household(tmp_path):
    (tmp_path / "Load.csv").write_text(
        "profile;pLoad;voltLvl;subnet\nHS0;120;1;EHV1\nH0-A;0.003;7;LV1.101\n")
    usage = ds.simbench_profile_usage(tmp_path)
    assert usage["HS0"]["voltage_levels"] == [1]
    assert usage["H0-A"]["nominal_p_mw_max"] == .003


def test_no_future_normalization(tmp_path):
    root = _simbench(tmp_path)
    a = ds.load_simbench_profiles(root, ["H0-A"])
    path = root / "LoadProfile.csv"
    path.write_text(path.read_text().replace("00:30;0.1;0.6", "00:30;10000;60000"))
    b = ds.load_simbench_profiles(root, ["H0-A"])
    np.testing.assert_array_equal(a["p_relative"][:2], b["p_relative"][:2])
    np.testing.assert_array_equal(a["q_relative"][:2], b["q_relative"][:2])


@pytest.mark.parametrize("values", [
    ["2016-01-01 00:00", "2016-01-01 00:00"],
    ["2016-01-01 00:00", "2016-01-01 00:15", "2016-01-01 00:45"],
])
def test_irregular_or_duplicate_rejected(values):
    with pytest.raises(ValueError):
        ds._time_axis(values)


def test_german_dst_native_intervals_preserved():
    spring, dt = ds._time_axis(["27.03.2016 01:45", "27.03.2016 03:00"],
        date_format="%d.%m.%Y %H:%M", expected_hours=.25, local_timezone="Europe/Berlin")
    assert dt == .25 and spring[0].endswith("+00:00")
    hours = ["01:45", "02:00", "02:15", "02:30", "02:45", "02:00", "02:15", "02:30", "02:45", "03:00"]
    fall, dt = ds._time_axis(["30.10.2016 " + x for x in hours],
        date_format="%d.%m.%Y %H:%M", expected_hours=.25, local_timezone="Europe/Berlin")
    assert dt == .25 and len(set(fall)) == 10


def _sintef_fixture(tmp_path, monkeypatch):
    path = tmp_path / "data.zip"
    base = "dataset/39_bus_semi_urban_reference_grid/"
    entries = {
        "load_bus_extra.csv": "bus_i,Consumer type,ID\n30,house,1\n30,house,2\n7,house,3\n",
        "p_load.csv": "Date,30,30,7\n2021-01-01 00:00:00,1,2,4\n2021-01-01 01:00:00,3,4,8\n",
        "q_load.csv": "Date,30,30.1,7\n2021-01-01 00:00:00,0.2,0.4,0.8\n2021-01-01 01:00:00,0.6,0.8,1.6\n",
        "mpc_bus.csv": "bus_i,type,basekV\n1,3,0.23\n30,1,0.23\n7,1,0.23\n",
        "mpc_branch.csv": "fbus,tbus,r,x\n1,30,0.1,0.02\n1,7,0.2,0.03\n",
        "mpc_base_mva.csv": "Base MVA\n0.05\n",
    }
    with zipfile.ZipFile(path, "w") as z:
        for name, content in entries.items():
            z.writestr(base + name, content)
    monkeypatch.setattr(ds, "SINTEF_ZIP_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def test_sintef_repeated_bus_columns_preserved(tmp_path, monkeypatch):
    path = _sintef_fixture(tmp_path, monkeypatch)
    out = ds.load_sintef_raw(path, "39_bus_semi_urban_reference_grid")
    assert out["observation_bus_ids"] == (30, 7)
    np.testing.assert_allclose(out["p_raw"], [[3, 4], [7, 8]])
    np.testing.assert_allclose(out["q_raw"], [[.6, .8], [1.4, 1.6]])
    assert out["provenance"]["consumer_count"] == 3
    assert out["provenance"]["constant_Q_over_P"]
    assert out["dt_hours"] == 1.
    assert not out["provenance"]["direct_AC_units_ready"]


def test_sintef_unit_requires_explicit_basis(tmp_path, monkeypatch):
    out = ds.load_sintef_raw(_sintef_fixture(tmp_path, monkeypatch), ds.SINTEF_GRIDS[0])
    with pytest.raises(ValueError):
        ds.convert_sintef_power_units(out, declared_unit="MW", declaration_basis="")
    m = ds.convert_sintef_power_units(out, declared_unit="MW", declaration_basis="test declaration")
    pu = ds.convert_sintef_power_units(out, declared_unit="pu_on_declared_base", declaration_basis="test declaration")
    np.testing.assert_allclose(m["p_kw"], out["p_raw"] * 1000)
    np.testing.assert_allclose(pu["p_kw"], out["p_raw"] * 50)


def test_sintef_digest_and_unknown_grid_rejected(tmp_path, monkeypatch):
    path = _sintef_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        ds.load_sintef_raw(path, "unknown")
    monkeypatch.setattr(ds, "SINTEF_ZIP_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        ds.load_sintef_raw(path, ds.SINTEF_GRIDS[0])


def test_window_keeps_native_sampling_and_no_filling():
    x = {"time": ("2021-01-01T00:00:00", "2021-01-01T01:00:00", "2021-01-01T02:00:00"),
         "dt_hours": 1., "p_raw": np.array([[1.], [2.], [3000.]])}
    y = ds.window(x, "2021-01-01 00:00", "2021-01-01 02:00")
    assert y["dt_hours"] == 1. and len(y["time"]) == 2
    np.testing.assert_array_equal(y["p_raw"], [[1.], [2.]])


def test_window_utc_bounds(tmp_path):
    x = ds.load_simbench_profiles(_simbench(tmp_path), ["H0-A"])
    y = ds.window(x, "2015-12-31 23:00", "2015-12-31 23:30")
    assert len(y["time"]) == 2 and y["time"][0].endswith("+00:00")


def test_profile_rank_exposes_duplicated_excitation():
    x = np.array([[1., 2.], [2., 4.], [4., 8.]])
    d = ds.profile_diagnostics(x, .2 * x)
    assert d["centered_scaled_rank_1e_10"] == 1
    assert d["centered_PQ_columns"] == 4
