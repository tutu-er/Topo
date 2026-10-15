"""Small saved-record checks; no simulation, fitting, MILP or AC imports."""
from copy import deepcopy

import pytest

from . import ev_policy_calibration as cal
from . import ev_fixed_train_collect as collect


def _setup(tmp_path, condition="weak8"):
    config = dict(training_points=8 if condition == "weak8" else 32, methods=list(collect.METHODS),
        calibration_seeds=list(range(100,136)), evaluation_seeds=list(range(200,212)))
    if condition == "rich32":config["training_support"]={"kind":"declared synthetic independent training"}
    freeze = dict(config=config, source_sha256={"selector.py":"1"*64})
    freeze["policy_id"] = cal.freeze_identity(freeze)["policy_id"]
    directories = {}
    for phase in ("calibration", "evaluation"):
        directory = tmp_path/condition/phase;directory.mkdir(parents=True)
        collect.write_json(directory/"freeze.json", freeze);directories[phase]=directory
    return directories, freeze


def _trial(freeze, seed, truth="star", score=0., condition="weak8", cap_extra=0.):
    names = collect.CANDIDATES
    def fits(losses):
        return [dict(candidate=n,name=n,success=True,loss=losses[n],r=[.003],x=[.001]) for n in names]
    train_losses={n:50.+i for i,n in enumerate(names)}
    val_losses={n:1. for n in names}
    methods=[]
    for method in collect.METHODS:
        future={n:1. for n in names};future[truth]+=score+(cap_extra if method==collect.PRIMARY[1] else 0.)
        methods.append(dict(name=method,status="complete",power_kw=[[0.,0.,0.]],
            noisy=dict(all_complete=True,failures={},fits=fits(future)),
            combined_loss={n:val_losses[n]+future[n] for n in names}))
    record=dict(schema="test_fixed_training_saved_record",**deepcopy(freeze),seed=seed,
        true_name=truth,candidate_miss=False,candidates=[dict(name=n) for n in names],source_unchanged=True,
        validation_used_by_selector=False,truth_used_by_selector=False,future_noise_used_by_selector=False,
        training_result=dict(all_complete=True,failures={},fits=fits(train_losses)),
        validation_result=dict(all_complete=True,failures={},fits=fits(val_losses)),methods=methods)
    if condition != "weak8":record["condition"]=condition
    return record


def _save(directory, row):
    path=directory/("trial_"+str(row["seed"])+".json.gz");collect.write_json(path,row);return path


def _calibration(directories, freeze, condition="weak8", cap_extra=0.):
    for i,seed in enumerate(range(100,136)):
        _save(directories["calibration"],_trial(freeze,seed,collect.CANDIDATES[i%4],float(i//4),condition,cap_extra))


def test_exact_combined_score_excludes_training_and_rejects_changed_lines_or_bad_sum(tmp_path):
    _, freeze=_setup(tmp_path)
    row=_trial(freeze,100,score=4.)
    identity=cal.freeze_identity(freeze)
    adapt=lambda r:collect.make_standard_record(r,reference_id="test",**identity)
    standard=adapt(row)
    assert standard["methods"]["constant"]["losses"]["star"]==6.
    assert standard["methods"]["constant"]["losses"]["star"] != 6.+53.
    assert standard["independent_unit_id"]=="100" and standard["training_condition"]=="weak8"
    changed=deepcopy(row);changed["methods"][1]["noisy"]["fits"][0]["r"][0]+=.0001
    assert adapt(changed)["methods"]["constant"]["status"]=="changed_training_lines_fit_failure"
    changed=deepcopy(row);changed["methods"][1]["combined_loss"]["star"]+=1.
    assert adapt(changed)["methods"]["constant"]["status"]=="combined_fixed_fit_loss_mismatch_failure"


def test_calibration_first_lock_nine_per_family_and_primary_max_then_offline_replay(tmp_path,monkeypatch):
    directories, freeze=_setup(tmp_path);_calibration(directories,freeze,cap_extra=10.)
    before=collect.collect_condition(directories["calibration"],condition="weak8")
    assert before["actual_calibration"]==36 and before["actual_evaluation"]==0
    assert before["phase_balance"]["calibration"]["prespecified_balanced_design"]
    for candidate in before["primary_family"]["calibration"]["thresholds"]["declared"].values():
        assert candidate["__family__"]["n"]==9 and candidate["__family__"]["rank"]==9
        assert candidate["__family__"]["threshold"]==18.
    _save(directories["evaluation"],_trial(freeze,200,"star",9.,cap_extra=10.))
    final=collect.collect_condition(directories["calibration"],directories["evaluation"],condition="weak8",stage_before=before)
    assert final["stage_before_calibration_exactly_preserved"]
    assert final["primary_family"]["evaluation"]["totals"]["any_method_false_rejections"]==1
    changed_source=deepcopy(before)
    changed_source["collector_manifest"]["source_sha256"]["ev_fixed_train_collect.py"]="0"*64
    with pytest.raises(ValueError,match="collector/calibrator source changed"):
        collect.collect_condition(directories["calibration"],directories["evaluation"],condition="weak8",stage_before=changed_source)
    bundle=deepcopy(final["standard_records"])
    def no_read(*a,**k):raise AssertionError("offline replay read raw data")
    monkeypatch.setattr(collect,"read_json",no_read)
    replay=collect.risk_report_from_standard_records(bundle)
    for group in ("individual_policies","primary_family"):assert replay[group]==final[group]
    bundle["evaluation"][0]["methods"]["none"]["status"]="tampered"
    with pytest.raises(ValueError,match="hash mismatch"):collect.risk_report_from_standard_records(bundle)


def test_training_operation_and_missing_fit_failures_keep_all_candidates_and_denominators(tmp_path):
    directories,freeze=_setup(tmp_path);_calibration(directories,freeze)
    training=_trial(freeze,200);training["training_result"]["all_complete"]=False;training["methods"]=[]
    _save(directories["evaluation"],training)
    operation=_trial(freeze,201);operation["methods"][0].update(status="selection_failure",power_kw=None)
    operation["methods"][1]["noisy"]["fits"].pop()
    _save(directories["evaluation"],operation)
    report=collect.collect_condition(directories["calibration"],directories["evaluation"],condition="weak8")
    stats=report["individual_policies"]["evaluation"]
    assert stats["methods"]["none"]["total"]==2
    assert stats["methods"]["none"]["operation_failures"]==1
    assert stats["methods"]["none"]["fit_failures"]==1
    assert stats["methods"]["constant"]["fit_failures"]==2
    assert all(d["retained"]==list(collect.CANDIDATES) for d in stats["records"][0]["methods"].values())
    assert stats["records"][1]["methods"]["none"]["retained"]==list(collect.CANDIDATES)
    assert stats["records"][1]["methods"]["constant"]["retained"]==list(collect.CANDIDATES)


def test_candidate_miss_is_separate_and_blocks_coverage_calibration(tmp_path):
    directories,freeze=_setup(tmp_path);_calibration(directories,freeze)
    path=directories["calibration"]/"trial_100.json.gz"
    row=collect.read_json(path);row.update(true_name=None,candidate_miss=True);collect.write_json(path,row)
    report=collect.collect_condition(directories["calibration"],condition="weak8")
    for group in ("individual_policies","primary_family"):
        calibration=report[group]["calibration"]
        assert calibration["coverage_failures"]==1 and not calibration["rejection_enabled"]
        assert all(q["infinite"] for candidate in calibration["thresholds"]["declared"].values() for q in candidate.values())
    assert report["standard_records"]["calibration"][0]["true_name"]=="__uncovered_truth__"


def test_conditions_are_separate_and_paired_units_count_once(tmp_path):
    reports=[]
    for condition in collect.CONDITIONS:
        directories,freeze=_setup(tmp_path,condition)
        _save(directories["calibration"],_trial(freeze,100,condition=condition))
        _save(directories["evaluation"],_trial(freeze,200,condition=condition))
        reports.append(collect.collect_condition(directories["calibration"],directories["evaluation"],condition=condition))
        if condition=="rich32":
            with pytest.raises(ValueError,match="cannot be pooled"):
                collect.collect_condition(directories["calibration"],condition="weak8")
    paired=collect.paired_condition_units(reports)
    assert paired["condition_record_count"]==4 and paired["distinct_base_history_units"]==2 and paired["paired_base_units"]==2
    assert not paired["cross_condition_any_error_control"]
    assert all(not r["cross_condition_any_error_control"] for r in reports)
    assert reports[0]["primary_family"]["calibration"]["calibration_id"] != reports[1]["primary_family"]["calibration"]["calibration_id"]


def test_source_mismatch_duplicate_units_and_changed_calibration_lock_rejected(tmp_path):
    directories,freeze=_setup(tmp_path);_calibration(directories,freeze)
    before=collect.collect_condition(directories["calibration"],condition="weak8")
    row=_trial(freeze,200);row["source_unchanged"]=False;path=_save(directories["evaluation"],row)
    with pytest.raises(ValueError,match="source was changed"):
        collect.collect_condition(directories["calibration"],directories["evaluation"],condition="weak8")
    row["source_unchanged"]=True;collect.write_json(path,row)
    cpath=directories["calibration"]/"trial_100.json.gz"
    changed=collect.read_json(cpath)
    for method in changed["methods"]:
        method["noisy"]["fits"][0]["loss"]+=20.
        method["combined_loss"][collect.CANDIDATES[0]]+=20.
    collect.write_json(cpath,changed)
    with pytest.raises(ValueError,match="calibration changed"):
        collect.collect_condition(directories["calibration"],directories["evaluation"],condition="weak8",stage_before=before)
    collect.write_json(directories["evaluation"]/"trial_duplicate.json.gz",row)
    with pytest.raises(ValueError,match="duplicate or unplanned"):
        collect.collect_condition(directories["calibration"],directories["evaluation"],condition="weak8")
