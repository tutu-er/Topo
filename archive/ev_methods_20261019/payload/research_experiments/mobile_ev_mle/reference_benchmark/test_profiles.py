"""Calendar selection, profile reuse and forecast leakage boundary checks."""
from copy import deepcopy
import gzip
import json

import numpy as np
import pandas as pd
import pytest

from . import profiles as pr


def _clock():
    return tuple(x.isoformat() for x in pd.date_range(
        "2016-01-01", "2017-01-01", inclusive="left", freq="15min", tz="UTC"))


def test_calendar_windows_native_nonoverlap_and_quarters():
    clock = _clock()
    windows = pr.calendar_windows(clock)
    assert {k: len(v) for k, v in windows.items()} == pr.DEFAULT_COUNTS
    intervals = []
    for phase, rows in windows.items():
        lo, hi = map(pd.Timestamp, pr.PHASE_BOUNDS[phase])
        for w in rows:
            s, e = w["start_row"], w["stop_row"]
            assert e - s == 204
            assert pd.Timestamp(clock[s]) >= lo
            assert pd.Timestamp(clock[e - 1]) < hi
            assert w["forecast_cutoff_utc"] == clock[s + 191]
            intervals.append((s, e))
    intervals.sort()
    assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))


def test_invalid_clock_and_overlap_fail():
    with pytest.raises(ValueError):
        pr.calendar_windows(_clock()[::2])
    with pytest.raises(ValueError):
        pr.calendar_windows(_clock(), counts=dict(development=100, calibration=16, evaluation=12))
    with pytest.raises(ValueError):
        pr.calendar_windows(_clock(), history=96)


def _net(name="mock", n=8):
    return dict(name=name, observation_bus_ids=list(range(1, n + 1)),
        buses=[dict(bus_id=i, pd_kw=100. + i, qd_kvar=30. + i) for i in range(1, n + 1)],
        physical_sha256="fixture", base_kw=1000., total_pd_kw=1000., total_qd_kvar=300.,
        provenance={"data_kind": "unit_fixture"}, metadata={})


def test_assignment_fixed_seed_reuse_and_no_hs():
    nets = {"b": _net("b", 32), "a": _net("a", 6)}
    before = deepcopy(nets)
    a = pr.profile_assignment(nets, 123)
    b = pr.profile_assignment(dict(reversed(list(nets.items()))), 123)
    assert a == b and nets == before
    assert set(a["b"].values()) == set(pr.PROFILE_NAMES)
    assert len(set(a["b"].values())) == 6 and len(a["b"]) == 32
    assert all(not n.startswith("HS") for values in a.values() for n in values.values())


def _development_window():
    time = list(pd.date_range("2016-02-01", periods=204, freq="15min", tz="UTC").map(lambda x: x.isoformat()))
    t = np.arange(204)[:, None]
    return dict(phase="development", index=0, timestamps=time,
        p_relative=(.5 + .1 * np.sin(t / 6 + np.arange(6))).tolist(),
        q_relative=(.05 * np.cos(t / 8 + np.arange(6))).tolist())


def test_forecaster_uses_only_development_history_not_planned_future():
    a = _development_window()
    b = deepcopy(a)
    b["p_relative"][192:] = (np.ones((12, 6)) * 1e12).tolist()
    b["q_relative"][192:] = (np.ones((12, 6)) * -1e12).tolist()
    x, y = pr.select_forecasters([a]), pr.select_forecasters([b])
    assert x == y
    assert x["nonnegative"] is False and x["p"]["selection"]["nonnegative"] is False
    assert x["q"]["selection"]["nonnegative"] is False
    assert x["p"]["selection"]["origins"] == [96, 120, 144, 168]
    c = deepcopy(a); c["phase"] = "evaluation"
    with pytest.raises(ValueError):
        pr.select_forecasters([c])


def test_history_rank_reveals_reused_curves():
    w = _development_window()
    net = _net(n=32)
    assignment = pr.profile_assignment({"mock": net}, 12)["mock"]
    d = pr._node_history_diagnostics(net, assignment, w)
    assert d["assigned_nodes"] == 32 and d["unique_PQ_profile_count"] == 6
    assert d["centered_scaled_rank_1e_10"] <= 12
    assert d["history_only"] and d["absolute_P_correlation_max"] > .999999


def _fake_sources(monkeypatch):
    clock = _clock(); t = np.arange(len(clock))[:, None]
    p = .5 + .1 * np.sin(t / 8 + np.arange(6))
    q = .1 * np.cos(t / 7 + np.arange(6))
    data = dict(time=clock, dt_hours=.25, profile_names=pr.PROFILE_NAMES,
        p_relative=p, q_relative=q, provenance=dict(sha256="source_fixture"))
    pv = dict(time=clock, dt_hours=.25, p_relative=np.maximum(0., np.sin(t[:, 0] / 12)),
        provenance=dict(sha256="pv_fixture"))
    monkeypatch.setattr(pr, "load_simbench_profiles", lambda *args: data)
    monkeypatch.setattr(pr, "load_simbench_pv", lambda *args: pv)
    monkeypatch.setattr(pr, "build_network", lambda name: _net(name))
    return data


def test_prepare_schema_restore_and_evaluation_perturbation(tmp_path, monkeypatch):
    data = _fake_sources(monkeypatch)
    config = dict(network_names=["mock"], phase_counts={k: 1 for k in pr.PHASE_BOUNDS})
    a = pr.prepare(tmp_path / "a", config)
    with gzip.open(tmp_path / "a/prepared.json.gz", "rt", encoding="utf-8") as f:
        restored = json.load(f)
    assert a == restored
    assert a["assignments"] == a["assignment"]
    assert a["forecast_configs"]["p"] == a["forecast_config_p"]
    w = a["windows"]["evaluation"][0]
    assert np.asarray(w["p_relative"]).shape == (204, 6)
    assert len(w["pv_relative"]) == len(w["timestamps"]) == 204
    # Changes anywhere in evaluation must not alter the frozen forecast model.
    e = pr.calendar_windows(data["time"], counts=config["phase_counts"])["evaluation"][0]
    data["p_relative"][e["start_row"]:e["stop_row"]] = 1e6
    data["q_relative"][e["start_row"]:e["stop_row"]] = -1e6
    b = pr.prepare(tmp_path / "b", config)
    assert a["forecast_selection"] == b["forecast_selection"]
    assert a["assignments"] == b["assignments"]
    assert a["windows"]["evaluation"][0]["timestamps"] == b["windows"]["evaluation"][0]["timestamps"]
    with pytest.raises(FileExistsError):
        pr.prepare(tmp_path / "a", config)
