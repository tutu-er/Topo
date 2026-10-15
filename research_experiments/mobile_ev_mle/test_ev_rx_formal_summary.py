"""Saved-record fixtures only; no AC, simulator, fitter, or selector."""
from copy import deepcopy
import hashlib

import pytest

from . import ev_dynamic_collect as collector
from . import ev_rx_formal_summary as formal
from .test_ev_rx_analyze import FAMILIES, METHODS, _trial, _write


def _inputs(tmp_path, *, evaluation_count=12, missing_method=False):
    cal, eva = tmp_path/"cal", tmp_path/"eval"
    calibration = list(range(1001, 1037))
    evaluation = list(range(2001, 2013))
    config = dict(select_modes=list(METHODS[:-1]), training_domain="cone",
                  calibration_seeds=calibration, evaluation_seeds=evaluation,
                  mean_policy_ablation=False, diagnostics=False)
    sources = {"fixture_only.py": "a"*64}
    freeze = dict(config=config, source_sha256=sources,
                  policy_id=collector.object_hash(dict(config=config, source_sha256=sources)))
    for directory, seeds in ((cal, calibration), (eva, evaluation[:evaluation_count])):
        _write(directory/"freeze.json", freeze)
        for i, seed in enumerate(seeds):
            trial = _trial(deepcopy(freeze), seed, cone=True)
            trial["family_index"] = i % 4
            trial["true_name"] = FAMILIES[i % 4]
            if directory == eva and i == 0 and missing_method:
                trial["methods"] = [m for m in trial["methods"] if m["name"] != "nearest_v2g_mobile"]
                trial["status"] = "partial_failure"
            _write(directory/f"trial_{seed}.json.gz", trial)
    return cal, eva


def _hashes(*directories):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in directories for p in directory.iterdir() if p.is_file()}


def test_threshold_persisted_before_eval_and_exact_prior_bytes(tmp_path, monkeypatch):
    cal, eva = _inputs(tmp_path)
    thresholds = collector.collect(cal, mode="calibrate", expected_per_family=9, alpha=.1)
    collector.write_json(cal/"thresholds.json", thresholds)
    before = _hashes(cal, eva)
    out = tmp_path/"report"
    original_collect, original_read = collector.collect, formal.rx.read_domain
    accesses = []

    def assert_saved(directory, label):
        if directory.resolve() == eva.resolve():
            assert (out/"thresholds.json").read_bytes() == (cal/"thresholds.json").read_bytes()
            accesses.append(label)

    def wrapped_collect(directory, **kwargs):
        assert_saved(directory, "collect")
        return original_collect(directory, **kwargs)

    def wrapped_read(directory, **kwargs):
        assert_saved(directory, "analyze")
        return original_read(directory, **kwargs)

    monkeypatch.setattr(collector, "collect", wrapped_collect)
    monkeypatch.setattr(formal.rx, "read_domain", wrapped_read)
    report = formal.run_formal(cal, eva, out)
    assert accesses == ["analyze", "collect"]
    assert report["status"] == "complete"
    assert report["calibration"]["planned_histories"] == 36
    assert report["evaluation"]["planned_histories"] == 12
    assert report["evaluation"]["complete_method_slots"] == 12*19
    assert len(report["decisions_by_method"]) == 19
    assert report["threshold_file"]["prior_threshold_reference_verified"]
    assert report["analyzer"]["new_AC_calls"] == report["analyzer"]["new_fit_calls"] == 0
    assert len(report["evaluation"]["paired_contrasts"]) == 60
    nearest = report["evaluation"]["paired_contrasts"]["nearest_v2g_mobile_vs_square"]
    # Star receives +3; each of the other true families receives -3.
    assert nearest["delta_M"]["mean"] == -1.5
    assert nearest["delta_M"]["total"] == 12
    assert report["evaluation"]["methods"]["square_v2g_fixed"]["proxy_available"] == 12
    assert "19种方法" in (out/"formal_summary.md").read_text(encoding="utf-8")
    assert before == _hashes(cal, eva)


def test_pending_and_method_failure_keep_full_denominator(tmp_path):
    cal, eva = _inputs(tmp_path, evaluation_count=11, missing_method=True)
    report = formal.run_formal(cal, eva, tmp_path/"report")
    assert report["status"] == "partial"
    domain = report["evaluation"]
    assert domain["pending_method_slots"] == 19
    assert domain["failed_method_slots"] == 1
    assert domain["observed_histories"] == 11 and domain["planned_histories"] == 12
    method = domain["methods"]["nearest_v2g_mobile"]
    assert method["M"]["valid"] == 10 and method["M"]["total"] == 12
    decisions = report["decisions_by_method"]["nearest_v2g_mobile"]["overall"]
    assert decisions["failed_keep_all"] == 1 and decisions["total"] == 11
    paired = domain["paired_contrasts"]["nearest_v2g_mobile_vs_square"]
    assert paired["delta_M"]["valid"] == 10 and paired["delta_M"]["total"] == 12


def test_incomplete_calibration_never_reads_evaluation(tmp_path, monkeypatch):
    cal, eva = _inputs(tmp_path)
    (cal/"trial_1001.json.gz").unlink()
    original_collect = collector.collect

    def wrapped(directory, **kwargs):
        assert directory.resolve() != eva.resolve(), "evaluation read before valid thresholds"
        return original_collect(directory, **kwargs)

    monkeypatch.setattr(collector, "collect", wrapped)
    with pytest.raises(ValueError, match="independent histories per true family"):
        formal.run_formal(cal, eva, tmp_path/"report")
    assert not (tmp_path/"report"/"thresholds.json").exists()


def test_refuse_different_existing_or_prior_threshold_bytes(tmp_path):
    cal, eva = _inputs(tmp_path)
    report = collector.collect(cal, mode="calibrate", expected_per_family=9, alpha=.1)
    out = tmp_path/"report"
    out.mkdir()
    (out/"thresholds.json").write_text("preserved", encoding="utf-8")
    with pytest.raises(ValueError, match="existing output thresholds differ"):
        formal.run_formal(cal, eva, out)
    assert (out/"thresholds.json").read_text() == "preserved"
    (cal/"thresholds.json").write_text("different", encoding="utf-8")
    with pytest.raises(ValueError, match="prior saved calibration"):
        formal.save_thresholds(tmp_path/"other"/"thresholds.json", report,
                               existing_reference=cal/"thresholds.json")
    assert not (tmp_path/"other").exists()
