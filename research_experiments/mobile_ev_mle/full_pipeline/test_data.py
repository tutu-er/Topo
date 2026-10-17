"""Independent unit/causality tests; the large raw CSV is not required."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from . import data


COL = "DE_KN_residential1_grid_import"
OTHER = "DE_KN_residential2_grid_import"


def csv_fixture(tmp_path, energy, *, marks=None, timestamps=None):
    count = len(energy)
    stamps = pd.date_range("2016-01-01", periods=count, freq="15min", tz="UTC") if timestamps is None else timestamps
    frame = pd.DataFrame(dict(utc_timestamp=stamps,
        interpolated=[""] * count if marks is None else marks,
        **{COL: energy, OTHER: np.arange(count) * .5}))
    path = tmp_path / "meter.csv"
    frame.to_csv(path, index=False)
    return path


def test_cumulative_kwh_to_kw_without_normalization(tmp_path):
    path = csv_fixture(tmp_path, [100., 100.25, 100.75, 101.5, 102.5])
    series = data.load_opsd_power(path, [COL])
    np.testing.assert_allclose(series.p_kw[:, 0], [1., 2., 3., 4.])
    assert series.valid.all()
    assert series.csv_line_numbers.tolist() == [3, 4, 5, 6]
    assert series.provenance["raw_sha256"] == data.file_sha256(path)
    assert series.timestamps[0] == data.utc64("2016-01-01T00:15Z")


@pytest.mark.parametrize("separator", [";", " | "])
def test_selected_interpolation_marks_discard_both_endpoints(tmp_path, separator):
    marks = ["", "", separator.join(["DE_KN_residential1_dishwasher", COL.upper()]), "", ""]
    series = data.load_opsd_power(csv_fixture(tmp_path, np.arange(5), marks=marks), [COL])
    assert series.valid[:, 0].tolist() == [True, False, False, True]
    assert np.isnan(series.p_kw[1:3]).all()


def test_unrelated_mark_does_not_discard_selected_column(tmp_path):
    path = csv_fixture(tmp_path, np.arange(4), marks=["", OTHER, "", ""])
    assert data.load_opsd_power(path, [COL]).valid.all()


def test_missing_reset_and_zero_are_distinct(tmp_path):
    series = data.load_opsd_power(csv_fixture(tmp_path, [3., 3., np.nan, 4., 1., 2.]), [COL])
    assert series.valid[:, 0].tolist() == [True, False, False, False, True]
    assert series.p_kw[0, 0] == 0
    assert series.provenance["counts"]["reset_intervals"] == [1]


def test_irregular_clock_not_silently_differenced(tmp_path):
    stamps = pd.to_datetime(["2016-01-01T00:00Z", "2016-01-01T00:15Z", "2016-01-01T01:15Z"])
    series = data.load_opsd_power(csv_fixture(tmp_path, [0., 1., 2.], timestamps=stamps), [COL])
    assert series.valid[:, 0].tolist() == [True, False]


def test_duplicate_household_and_wrong_raw_hash_rejected(tmp_path):
    path = csv_fixture(tmp_path, np.arange(4))
    with pytest.raises(ValueError, match="distinct"):
        data.load_opsd_power(path, [COL, COL])
    with pytest.raises(ValueError, match="SHA256"):
        data.load_opsd_power(path, [COL], expected_sha256="0" * 64)


def test_causal_fill_prefix_is_unaffected_by_future():
    x = [np.nan, 2., np.nan, np.nan, np.nan, 10.]
    filled, mask = data.causal_forward_fill(x, max_gap=2)
    assert np.isnan(filled[0]) and np.isnan(filled[4])
    np.testing.assert_allclose(filled[1:4], [2., 2., 2.])
    short, _ = data.causal_forward_fill(x[:4], max_gap=2)
    np.testing.assert_equal(filled[:4], short)
    assert mask.tolist() == [False, False, True, True, False, False]


def clean_series(count=60):
    stamps = pd.date_range("2016-01-01", periods=count, freq="15min").to_numpy()
    power = np.arange(count, dtype=float)[:, None]
    return data.PowerSeries(stamps, power, np.ones_like(power, bool), (COL,), np.arange(3, 3 + count), {})


def test_chronological_windows_disjoint_and_future_sealed():
    series = clean_series()
    split = data.CalendarSplit("development", "2016-01-01", "2016-01-02")
    indices = data.chronological_windows(series, H=10, T=2, split=split, count=3)
    assert indices == [0, 12, 24]
    windows = [data.select_window(series, phase="development", start_utc=series.timestamps[i],
        history_slots=10, future_slots=2, split=split) for i in indices]
    assert data.assert_windows_disjoint(windows)
    np.testing.assert_array_equal(windows[0]["history"]["p_kw"][:, 0], np.arange(10))
    np.testing.assert_array_equal(windows[0]["future"]["p_kw"][:, 0], [10, 11])
    assert windows[0]["provenance"]["simulator_only_future"]


def test_missing_windows_cannot_be_cherry_picked_by_load_size():
    series = clean_series()
    altered = data.PowerSeries(series.timestamps, series.p_kw * 100 + 7,
        series.valid, series.columns, series.csv_line_numbers, {})
    assert data.chronological_windows(series, 10, 2) == data.chronological_windows(altered, 10, 2)
    assert data.RESIDENTIAL_COLUMNS == data.MIXED8_COLUMNS[:6]
    assert len(set(data.MIXED8_COLUMNS)) == 8


def test_split_boundaries_and_explicit_overlap_rejected():
    series = clean_series()
    with pytest.raises(ValueError, match="overlap"):
        data.register_calendar_windows(series, data.CalendarSplit("development", "2016-01-01", "2016-01-02"),
            history_slots=10, future_slots=2, start_dates=[series.timestamps[0], series.timestamps[5]])
    with pytest.raises(ValueError, match="crosses"):
        data.select_window(series, phase="development", start_utc=series.timestamps[0], history_slots=10,
            future_slots=2, split=data.CalendarSplit("development", "2016-01-01", "2016-01-01T02:00Z"))
    with pytest.raises(ValueError, match="overlap"):
        data.validate_calendar_splits([data.CalendarSplit(name, "2016-01-01", "2016-02-01") for name in data.PHASES])


def test_synthetic_Q_is_explicit_and_prefix_causal():
    p = np.full((20, 2), 2.)
    stamps = pd.date_range("2016-01-01", periods=20, freq="15min", tz="UTC")
    full = data.synthetic_reactive_power(p, stamps, seed=4)
    prefix = data.synthetic_reactive_power(p[:10], stamps[:10], seed=4)
    np.testing.assert_allclose(full["q_kvar"][:10], prefix["q_kvar"])
    np.testing.assert_allclose(full["q_kvar"], p * np.tan(np.arccos(full["power_factor"])))
    assert full["metadata"]["synthetic"] and not full["metadata"]["measured_Q"]
