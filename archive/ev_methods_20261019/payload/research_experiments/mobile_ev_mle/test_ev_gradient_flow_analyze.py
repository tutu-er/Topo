"""Pure saved-record fixtures; no AC, fitter, selector or simulation execution."""
from copy import deepcopy
import hashlib

import pytest

from . import ev_dynamic_collect as collector
from . import ev_gradient_flow_analyze as analysis
from .test_ev_rx_analyze import FAMILIES, _trial, _write


def inputs(tmp_path, *, development_count=None, evaluation_count=16, fail=False):
    planned_dev = list(range(3001, 3005))
    cal_seeds, eval_seeds = list(range(1001, 1037)), list(range(2001, 2017))
    cfg = dict(select_modes=[m for m in analysis.METHODS if m != "highlow"],
        development_seeds=planned_dev, calibration_seeds=cal_seeds, evaluation_seeds=eval_seeds,
        mean_policy_ablation=False, diagnostics=False)
    sources = {"fixture_only.py": "a"*64}
    freeze = dict(config=cfg, source_sha256=sources,
        policy_id=collector.object_hash(dict(config=cfg, source_sha256=sources)))
    dirs = {p:tmp_path/p for p in ("development", "calibration", "evaluation")}
    for phase, seeds in (("development", planned_dev[:development_count or 0]),
                         ("calibration", cal_seeds), ("evaluation", eval_seeds[:evaluation_count])):
        _write(dirs[phase]/"freeze.json", freeze)
        for i, seed in enumerate(seeds):
            raw = _trial(deepcopy(freeze), seed)
            raw.update(family_index=i%4, true_name=FAMILIES[i%4],
                training_top2=dict(names=["star", "clade_01"], miss_true=i%4 >= 2))
            base = deepcopy(raw["methods"][0])
            raw["methods"] = []
            raw["selection"]["methods"] = {}
            for j, name in enumerate(analysis.METHODS):
                row = deepcopy(base)
                row["name"] = name
                future = row["noisy"]["fits"]
                for fit in future:
                    fit["loss"] = 3. if fit["candidate"] == raw["true_name"] else 9.+j
                row["loss_by_candidate"] = {fit["candidate"]:fit["loss"]+1 for fit in future}
                row["M"] = 6.+j
                row["runtime"] = {k:1.+j for k in analysis.RUNTIMES}
                if fail and phase == "evaluation" and i == 0 and name == "gradac_v2g_mobile":
                    row["status"] = "selection_failure"
                    row["power_kw"] = None
                raw["methods"].append(row)
                raw["selection"]["methods"][name] = dict(selected_origin="MILP", iterations=[1, 2], failures=[])
            _write(dirs[phase]/f"trial_{seed}.json.gz", raw)
    return dirs


def hashes(*directories):
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for d in directories for p in d.iterdir() if p.is_file()}


def test_formal_threshold_precedes_eval_read_and_planned_denominator(tmp_path, monkeypatch):
    dirs = inputs(tmp_path, evaluation_count=15, fail=True)
    before = hashes(*dirs.values())
    out = tmp_path/"report"
    original_collect, original_read = collector.collect, collector.read_json
    reads = []

    def check(path):
        if dirs["evaluation"] in path.resolve().parents or path.resolve() == dirs["evaluation"].resolve():
            assert (out/"thresholds.json").exists()
            reads.append(str(path))

    def wrapped_collect(path, **kwargs):
        check(path)
        return original_collect(path, **kwargs)

    def wrapped_read(path):
        check(path)
        return original_read(path)

    monkeypatch.setattr(collector, "collect", wrapped_collect)
    monkeypatch.setattr(collector, "read_json", wrapped_read)
    report = analysis.analyze(out, calibration=dirs["calibration"], evaluation=dirs["evaluation"])
    assert reads
    assert report["status"] == "partial"
    assert report["calibration"]["planned_histories"] == 36
    evaluation = report["evaluation"]
    assert evaluation["planned_histories"] == 16
    assert evaluation["observed_histories"] == 15
    assert evaluation["pending_method_slots"] == 13
    assert evaluation["failed_method_slots"] == 1
    method = evaluation["methods"]["gradac_v2g_mobile"]
    assert method["M"]["total"] == 16 and method["M"]["valid"] == 14
    assert method["runtime"]["planning_standalone_seconds"]["total"] == 16
    assert evaluation["training_top2_miss"] == 7
    assert len(evaluation["paired_contrasts"]) == 17
    paired = evaluation["paired_contrasts"]["gradac_vs_square_v2g_mobile"]["delta_M"]
    assert paired["valid"] == 14 and paired["total"] == 16 and paired["positive"] == 14
    decision = report["decisions_by_method"]["gradac_v2g_mobile"]["overall"]
    assert decision["failed_keep_all"] == 1 and decision["total"] == 15
    assert report["analyzer"]["new_AC_calls"] == 0
    assert before == hashes(*dirs.values())
    assert "训练前二漏真" in (out/"summary.md").read_text(encoding="utf-8")


def test_development_partial_full_methods_and_no_calibration(tmp_path):
    dirs = inputs(tmp_path, development_count=3, evaluation_count=0)
    report = analysis.analyze(tmp_path/"report", development=dirs["development"])
    assert report["status"] == "partial"
    dev = report["development"]
    assert dev["planned_method_slots"] == 52 and dev["complete_method_slots"] == 39
    assert dev["candidate_omissions"] == 0 and dev["training_top2_miss"] == 1
    assert "thresholds_id" not in report
    assert dev["paired_contrasts"]["gradi2_vs_gradac_v2g_fixed"]["delta_M"]["median"] == 1.
    assert "pair_scope" in dev["records"]["3001"]["methods"]["pairac_v2g_fixed"]


def test_empty_development_preserves_all_pending_histories(tmp_path):
    dirs = inputs(tmp_path, development_count=0, evaluation_count=0)
    result = analysis.analyze(tmp_path/"report", development=dirs["development"])
    assert result["status"] == "partial"
    assert result["development"]["pending_method_slots"] == 52
    assert result["development"]["methods"]["constant"]["M"]["unavailable"] == 4


def test_incomplete_calibration_no_evaluation_read(tmp_path, monkeypatch):
    dirs = inputs(tmp_path)
    (dirs["calibration"]/"trial_1001.json.gz").unlink()
    original = collector.collect

    def wrapped(path, **kwargs):
        assert path.resolve() != dirs["evaluation"].resolve()
        return original(path, **kwargs)

    monkeypatch.setattr(collector, "collect", wrapped)
    with pytest.raises(ValueError, match="independent histories per true family"):
        analysis.analyze(tmp_path/"report", calibration=dirs["calibration"], evaluation=dirs["evaluation"])
    assert not (tmp_path/"report"/"thresholds.json").exists()


def test_thresholds_conflict_preserves_existing_file(tmp_path):
    dirs = inputs(tmp_path)
    out = tmp_path/"report"
    out.mkdir()
    (out/"thresholds.json").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="existing output thresholds differ"):
        analysis.analyze(out, calibration=dirs["calibration"], evaluation=dirs["evaluation"])
    assert (out/"thresholds.json").read_text() == "keep"


def test_contrast_tolerance_and_min_max():
    row = analysis.stats([-2., -5e-7, 0., 5e-7, 3., None], 7)
    assert (row["negative"], row["ties"], row["positive"]) == (1, 3, 1)
    assert row["minimum"] == -2. and row["maximum"] == 3.
    assert row["unavailable"] == 2 and row["total"] == 7
    with pytest.raises(ValueError, match="denominator"):
        analysis.stats([1., 2.], 1)


def test_eight_per_family_has_infinite_point_one_threshold():
    assert collector.rank_threshold(list(range(8)), alpha=.1)["threshold"] == "inf"
    assert collector.rank_threshold(list(range(9)), alpha=.1)["threshold"] == 8.


def test_empty_evaluation_still_checks_frozen_policy(tmp_path):
    dirs = inputs(tmp_path, evaluation_count=0)
    path = dirs["evaluation"]/"freeze.json"
    changed = collector.read_json(path)
    changed["config"]["changed_strategy"] = True
    changed["policy_id"] = collector.object_hash(dict(config=changed["config"], source_sha256=changed["source_sha256"]))
    _write(path, changed)
    with pytest.raises(ValueError, match="frozen policy differs"):
        analysis.analyze(tmp_path/"report", calibration=dirs["calibration"], evaluation=dirs["evaluation"])
    assert (tmp_path/"report"/"thresholds.json").exists()


def test_invalid_runtime_is_exposed(tmp_path):
    dirs = inputs(tmp_path, development_count=1, evaluation_count=0)
    path = dirs["development"]/"trial_3001.json.gz"
    record = collector.read_json(path)
    record["methods"][0]["runtime"]["dependency_seconds"] = -1.
    _write(path, record)
    with pytest.raises(ValueError, match="invalid runtime"):
        analysis.analyze(tmp_path/"report", development=dirs["development"])


def test_iteration_profiles_and_payment_events_not_copied():
    log = dict(iterations=[dict(before_score=1., candidate_score=2., after_score=2., accepted=True,
        gradient_profile={"large_array":[1]*100}, candidate_profile={"large_array":[2]*100},
        MILP=dict(seconds=.1, linear_mip_gap=0., power_kw=[[2.]*100], predicted_directional_gain=1.))],
        counters=dict(cache_payment_log=[dict(cache_hit=False, paid_pair_fits=12, actual_seconds=.2)],
            source_kkt_by_method=dict(ac=dict(successful_fits=12, scaled_max=1e-7, scaled_count_gt_1e_minus_5=0))))
    iterations, summary = analysis._iterations(log)
    assert summary["count"] == summary["accepted"] == 1
    assert "gradient_profile" not in iterations[0] and "power_kw" not in iterations[0]["MILP"]
    counter = analysis._counters(log)
    assert "cache_payment_log" not in counter and counter["cache_payment_summary"]["paid_pair_fits"] == 12
    assert counter["source_kkt_by_method"]["ac"]["scaled_max"] == 1e-7


def test_timing_audit_deduplicates_diamond_dependencies_and_exposes_error():
    timing = {
        "square_v2g_fixed":dict(planning_actual_seconds=1., dependency_seconds=.1,
            planning_standalone_seconds=1.1, dependencies=[], actual_calls=dict(AC_batches=1)),
        "square_v2g_mobile":dict(planning_actual_seconds=2., dependency_seconds=1.1,
            planning_standalone_seconds=3.1, dependencies=["square_v2g_fixed"], actual_calls=dict(AC_batches=2)),
        "gradac_v2g_fixed":dict(planning_actual_seconds=3., dependency_seconds=1.3,
            planning_standalone_seconds=4.3, dependencies=["square_v2g_fixed"], context_seconds=.2,
            actual_calls=dict(AC_batches=3), context_calls=dict(AC_batches=5),
            charged_ancestors=["square_v2g_fixed"], standalone_calls=dict(AC_batches=9)),
        "gradac_v2g_mobile":dict(planning_actual_seconds=4., dependency_seconds=6.3,
            planning_standalone_seconds=10.3, dependencies=["square_v2g_mobile", "gradac_v2g_fixed"],
            context_seconds=.2, actual_calls=dict(AC_batches=4), context_calls=dict(AC_batches=5),
            charged_ancestors=["square_v2g_fixed", "square_v2g_mobile", "gradac_v2g_fixed"],
            standalone_calls=dict(AC_batches=15))}
    trial = dict(selection=dict(timing=timing, public_setup_seconds=.1),
        common_runtime=dict(training_and_candidates_seconds=2., validation_seconds=1., forecast_seconds=.5,
            training_calls=dict(AC_batches=10), validation_calls=dict(AC_batches=20)),
        methods=[dict(name="gradac_v2g_mobile", noisy=dict(seconds=.5, cost=dict(ac_calls=30)),
            runtime=dict(common_history_pipeline_seconds=3.5, operational_future_fit_seconds=.5,
                planning_standalone_seconds=10.3, final_operation_audit_seconds=.1,
                end_to_end_standalone_seconds=14.4, standalone_calls=dict(AC_batches=15), standalone_AC_batches=75))])
    report = analysis.timing_audit(trial)
    assert report["success"] and report["checks"] >= 16
    timing["gradac_v2g_mobile"]["dependency_seconds"] += 1.
    report = analysis.timing_audit(trial)
    assert not report["success"]
    assert any(r["label"] == "gradac_v2g_mobile/dependency" for r in report["failures"])
