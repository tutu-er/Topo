"""Historical boundaries, declarations, one tiny worker and an owned timeout."""
from copy import deepcopy
import gzip
import json
import sys

import numpy as np
import pytest

from ..information_context import meter_to_record
from ..gradient_price_study import observations_record
from .test_fixed_source import fixture
from . import benchmark as b


def source_fixture(tmp_path):
    candidate, data, meter, options, state = fixture(periods=4)
    declaration = dict(name=candidate.name, n=candidate.n, supports=candidate.supports)
    row = dict(loss=2., seconds=1., success=True, state=state)
    raw = dict(spec=dict(network="paper11", phase="development", cluster_index=0, source_mode="none"),
        observed_history=observations_record(data), history_meter=meter_to_record(meter),
        candidate_structures=[declaration], fit_options=options,
        freeze_identity="fixture_identity", planning=dict(history=dict(candidates={candidate.name: row}),
            forecast_values=[999999., -888888.]),
        evaluation=dict(secret_future=[1234567.], complex_brackets='[}]"\\',
                        actual_true_lines=[45678.]),
        truth_evaluation_only=dict(r=[987654.]), future_meter=dict(v_sigma=123456.))
    source = tmp_path/"source"; source.mkdir()
    path = source/"history_000_paper11_none.json.gz"
    path.write_bytes(gzip.compress(b._bytes(raw)))
    return source, raw


def package_fixture(tmp_path):
    source, raw = source_fixture(tmp_path)
    path = tmp_path/"history_only.json.gz"
    package = b.export_history(source, path, networks=("paper11", "case33"))
    return path, package, raw


def test_allowlist_projection_never_decodes_excluded_numbers(tmp_path, monkeypatch):
    source, raw = source_fixture(tmp_path)
    original = json.JSONDecoder.raw_decode
    decoded_starts = []
    def spy(self, text, idx=0):
        decoded_starts.append(text[idx:idx+60])
        return original(self, text, idx)
    monkeypatch.setattr(json.JSONDecoder, "raw_decode", spy)
    projected = b.project_source(source/"history_000_paper11_none.json.gz")
    assert set(projected) == set(b.SOURCE_PROJECTION)
    assert set(projected["planning"]) == {"history"}
    assert "evaluation" not in projected and "future_meter" not in projected
    assert not any(value.startswith('{"secret_future"') or value.startswith("1234567") or
                   value.startswith("987654") or value.startswith("999999") for value in decoded_starts)


def test_export_hash_boundary_and_missing_network(tmp_path):
    path, package, _ = package_fixture(tmp_path)
    saved = b.load_history_package(path)
    assert saved["payload_sha256"] == package["payload_sha256"]
    assert list(saved["items"]) == ["paper11"]
    assert saved["fits_started"] == 0
    assert saved["missing"][0]["network"] == "case33"
    text = b._bytes(saved).decode()
    assert "secret_future" not in text and "987654" not in text and "999999" not in text
    assert not saved["data_boundary"]["excluded_value_decoding"]
    selected = saved["items"]["paper11"]["selected_candidates"]
    assert len(selected) == 1
    assert selected[0]["roles"] == ["lowest_history_loss", "slow_history_fit"]
    corrupted = deepcopy(saved); corrupted["items"]["paper11"]["history_length"] = 99
    b._write(path, corrupted)
    with pytest.raises(ValueError, match="content hash"):
        b.load_history_package(path)


def test_task_order_alternates_without_starting_fits(tmp_path):
    _, package, _ = package_fixture(tmp_path)
    tasks = b.make_tasks(package, prefixes=(2,), algorithms=("fixed_ref", "fixed_2d"), repeats=3)
    assert [task["algorithm"] for task in tasks] == ["fixed_ref", "fixed_2d", "fixed_2d",
                                                    "fixed_ref", "fixed_ref", "fixed_2d"]
    assert len({task["task_id"] for task in tasks}) == 6
    with pytest.raises(ValueError, match="prefix exceeds"):
        b.make_tasks(package, prefixes=(192,))


def test_prefix_initializer_and_warm_are_explicitly_distinct(tmp_path):
    _, package, _ = package_fixture(tmp_path)
    tasks = b.make_tasks(package, prefixes=(2,), algorithms=("fixed_2d",),
                         starts=b.START_MODES, repeats=1)
    item = package["items"]["paper11"]
    for task in tasks:
        candidate, data, meter, options, initial, record = b._reconstruct(item, task)
        assert data.t == 2 and candidate.n == 3
        assert len(initial["source_p"]) == 2
        assert np.asarray(meter.v_sigma).shape == (2, 3)
        assert record["uses_full_history_fitted_state"] == (task["start"] == "historical_warm")
        if task["start"] == "historical_warm":
            assert record["training_time_count"] == 4
            assert "not a prefix-only" in record["temporal_interpretation"]


def test_tiny_worker_replays_gradient_and_channel_sse(tmp_path):
    path, package, _ = package_fixture(tmp_path)
    task = b.make_tasks(package, prefixes=(2,), algorithms=("fixed_2d",),
                        starts=("historical_warm",), repeats=1)[0]
    result = b.run_worker(path, task)
    assert result["status"] == "returned", result
    assert result["solver_success"]
    assert result["audit"]["available"]
    assert result["audit"]["loss"] < 1e-10
    assert sum(result["audit"]["channel_sse"].values()) == pytest.approx(result["audit"]["loss"])
    assert result["audit"]["reported_loss_absolute_difference"] < 1e-10
    assert result["audit"]["ac_equation_residual"] < 2e-10
    assert "per period" in result["budget_scope"]
    assert not result["future_used"] and not result["truth_used"]


def test_worker_wrong_identity_is_retained_exception(tmp_path):
    path, package, _ = package_fixture(tmp_path)
    task = b.make_tasks(package, prefixes=(2,), algorithms=("fixed_2d",), repeats=1)[0]
    task["package_sha256"] = "wrong"
    result = b.run_worker(path, task)
    assert result["status"] == "exception"
    assert "package hash" in result["exception"]


def test_owned_worker_timeout_is_retained_and_cached(tmp_path, monkeypatch):
    path, package, _ = package_fixture(tmp_path)
    task = b.make_tasks(package, prefixes=(2,), algorithms=("fixed_2d",), repeats=1)[0]
    monkeypatch.setattr(b, "_worker_command", lambda *args:
                        [sys.executable, "-c", "import time; time.sleep(5)"])
    output = tmp_path/"bounded"
    result = b.run_task_subprocess(path, task, output, timeout_seconds=.05)
    assert result["status"] == "timeout"
    assert result["retained_in_denominator"] and result["loss"] is None
    assert result["process_wall_seconds"] < 3.
    cached = b.run_task_subprocess(path, task, output, timeout_seconds=.05)
    assert cached == result


def test_run_lock_prevents_duplicate_scheduler(tmp_path):
    output = tmp_path/"locked"; output.mkdir()
    (output/"benchmark.lock").write_text("existing-controller")
    with pytest.raises(FileExistsError):
        b.run_matrix(tmp_path/"unused", [], output)
