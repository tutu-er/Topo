"""Small saved-record fixtures; no science fitter, strategy, or simulator runs."""
from copy import deepcopy
import json

import pytest

from . import ev_dynamic_collect as collect


def _freeze(path, *, changed=False):
    config = dict(select_modes=["none", "constant", "charge_fixed"], diagnostics=False,
                  mean_policy_ablation=False, calibration_seeds=list(range(100,136)),
                  evaluation_seeds=list(range(200,212)), forecast="changed" if changed else "ar1")
    sources = {"science.py": "ab"*32}
    frozen = dict(config=config, source_sha256=sources,
                  policy_id=collect.object_hash(dict(config=config, source_sha256=sources)))
    path.mkdir(parents=True, exist_ok=True)
    collect.write_json(path/"freeze.json", frozen)
    return frozen


def _fit(name, loss, *, fixed=False):
    return dict(name=name, candidate=name, success=True, loss=loss,
                r=[.003], x=[.001], diagnostics=dict(line_parameters_fixed=fixed))


def _block(losses, *, fixed):
    return dict(all_complete=True, failures={},
                fits=[_fit(name, loss, fixed=fixed) for name,loss in losses.items()],
                cost=dict(ac_calls=10, fit_calls=4, solver_calls=8), seconds=.01)


def _trial(frozen, seed, family_index, score, *, failed=None, omit=False):
    family = collect.FAMILIES[family_index]
    names = [name for name in collect.FAMILIES if not (omit and name == family)]
    other = next(name for name in names if name != family)
    losses = {name:12. for name in names}
    losses[other] = 10.
    if family in names:
        losses[family] = 10.+score
    val = {name:1. for name in names}
    future = {name:value-1. for name,value in losses.items()}
    rows = []
    for method in collect.expected_methods(frozen["config"]):
        row = dict(name=method, status="complete", power_kw=[[0.,0.,0.]],
            operation=dict(success=True, grid_energy_kwh=3., charging_kwh=3.5,
                           discharging_kwh=.5,throughput_kwh=4., driving_kwh=0.,
                           worker_minutes=0., moves=0),
            loss_by_candidate=deepcopy(losses), noisy=_block(future,fixed=True),
            noiseless=_block(future,fixed=True))
        if method == failed:
            row.update(status="selection_failure", power_kw=None)
        rows.append(row)
    return dict(**frozen, schema="ev_ordinary_dynamic_v1", seed=seed,
        family_index=family_index, true_name=None if omit else family,
        candidate_miss=omit, candidates=[dict(name=name) for name in names],
        methods=rows, training_result=_block({name:30. for name in names},fixed=False),
        validation_result=_block(val,fixed=True), source_unchanged=True,
        training_parameter_error=dict(R=.1,X=.2), seconds=1.,
        actual_solver_calls=dict(total_AC_batches=20,total_AC_snapshots=200,AC_failures=0),
        selection=dict(uses_validation=False,uses_truth=False,uses_future_noise=False,
            methods={"charge_fixed":dict(selected_origin="MILP",
                selected_is_new_MILP_proposal=True, failures=[])}))


def _save(path, frozen, seed, family, score, **kwargs):
    value = _trial(frozen,seed,family,score,**kwargs)
    collect.write_json(path/("trial_"+str(seed)+".json.gz"),value)
    return value


def _calibration(path, *, missing=False):
    frozen = _freeze(path)
    for family in range(4):
        for i in range(9):
            seed = 100+family*9+i
            value = _save(path,frozen,seed,family,float(i),
                          failed="design_charge_fixed" if missing and family == 0 and i == 8 else None)
            if missing and family == 1 and i == 8:
                value["methods"][0]["loss_by_candidate"] = {}
                collect.write_json(path/("trial_"+str(seed)+".json.gz"),value)
    return frozen


def test_rank_quantile_and_missing_scores():
    q = collect.rank_threshold(list(range(9)),alpha=.1)
    assert (q["n"],q["rank"],q["threshold"]) == (9,9,8.)
    assert collect.rank_threshold([0.]*9,alpha=.01)["threshold"] == "inf"
    assert collect.rank_threshold([0.]*8+[None],alpha=.1)["threshold"] == "inf"
    assert collect.rank_threshold([0.]*8+["inf"],alpha=.1)["missing_or_infinite"] == 1
    with pytest.raises(ValueError,match="alpha"):
        collect.rank_threshold([0.],alpha=0.)


def test_balanced_calibration_has_separate_methods_families_hashes(tmp_path):
    _calibration(tmp_path/"cal")
    result = collect.collect(tmp_path/"cal",mode="calibrate")
    assert result["summary"]["independent_histories"] == 36
    assert result["summary"]["family_counts"] == {family:9 for family in collect.FAMILIES}
    for method in result["methods"]:
        assert {v["threshold"] for v in result["calibration"][method].values()} == {8.}
    assert len(result["trial_files"]) == 36 and all(len(r["sha256"]) == 64 for r in result["trial_files"])
    assert result["thresholds_id"] == collect.object_hash({k:v for k,v in result.items() if k != "thresholds_id"})
    assert result["collector"]["new_fit_calls"] == 0
    assert result["rank_retention_requires_true_candidate_coverage"] is True
    assert result["candidate_omission_counted_in_overall_false_reject"] is True
    assert result["rank_retention_requires_within_true_family_exchangeability"] is True
    assert result["calibration_is_methodwise"] is True
    assert result["simultaneous_multi_strategy_error_control"] is False
    assert result["summary"]["methods"]["design_charge_fixed"]["selected_origin"] == {"new":36}
    assert result["summary"]["training_parameter_error"]["R"]["mean"] == pytest.approx(.1)


def test_failure_missing_score_becomes_infinity_and_json_portable(tmp_path):
    _calibration(tmp_path/"cal",missing=True)
    result = collect.collect(tmp_path/"cal",mode="calibrate")
    assert result["calibration"]["design_charge_fixed"]["star"]["threshold"] == "inf"
    assert result["calibration"]["none"]["clade_01"]["threshold"] == "inf"
    assert result["summary"]["methods"]["design_charge_fixed"]["failed"] == 1
    assert result["summary"]["methods"]["none"]["M"]["valid"] == 35
    collect.write_json(tmp_path/"thresholds.json",result)
    assert collect.read_json(tmp_path/"thresholds.json") == result
    json.dumps(result,allow_nan=False)


def test_evaluation_keep_all_failure_and_omission_counted_in_false_reject(tmp_path):
    frozen = _calibration(tmp_path/"cal")
    thresholds = collect.collect(tmp_path/"cal",mode="calibrate")
    _freeze(tmp_path/"eval")
    _save(tmp_path/"eval",frozen,200,0,9.)
    _save(tmp_path/"eval",frozen,201,1,100.,failed="constant")
    _save(tmp_path/"eval",frozen,202,2,0.,omit=True)
    result = collect.collect(tmp_path/"eval",mode="evaluate",thresholds=thresholds)
    const = result["evaluation"]["constant"]["overall"]
    assert const["total"] == 3 and const["failed_keep_all"] == 1
    assert const["false_reject"] == 2 and const["candidate_omissions"] == 1
    assert result["rank_retention_requires_true_candidate_coverage"] is True
    assert result["candidate_omission_counted_in_overall_false_reject"] is True
    failure = result["decisions"][1]["methods"]["constant"]
    assert set(failure["retained"]) == set(collect.FAMILIES)
    assert failure["failed_keep_all"] and not failure["false_reject"]
    assert result["evaluation"]["none"]["by_true_family"]["clade_02"]["false_reject"] == 1
    assert result["summary"]["methods"]["constant"]["M"]["total"] == 3
    assert result["summary"]["methods"]["constant"]["M"]["valid"] == 1


def test_reject_frozen_policy_or_calibration_seed_overlap(tmp_path):
    frozen = _calibration(tmp_path/"cal")
    thresholds = collect.collect(tmp_path/"cal",mode="calibrate")
    changed = _freeze(tmp_path/"changed",changed=True)
    _save(tmp_path/"changed",changed,200,0,0.)
    with pytest.raises(ValueError,match="frozen policy differs"):
        collect.collect(tmp_path/"changed",mode="evaluate",thresholds=thresholds)
    _freeze(tmp_path/"overlap")
    _save(tmp_path/"overlap",frozen,100,0,0.)
    with pytest.raises(ValueError,match="seeds overlap"):
        collect.collect(tmp_path/"overlap",mode="evaluate",thresholds=thresholds)
    bad = deepcopy(thresholds)
    bad["calibration"]["none"]["star"]["threshold"] = 999.
    with pytest.raises(ValueError,match="content/hash"):
        collect.collect(tmp_path/"overlap",mode="evaluate",thresholds=bad)


def test_changed_lines_missing_method_and_partial_pairs_preserve_denominators(tmp_path):
    frozen = _freeze(tmp_path/"data")
    a = _save(tmp_path/"data",frozen,200,0,1.)
    b = _save(tmp_path/"data",frozen,201,1,2.)
    b["methods"][2]["noisy"]["fits"][0]["r"] = [.004]
    b["methods"] = [r for r in b["methods"] if r["name"] != "highlow"]
    collect.write_json(tmp_path/"data"/"trial_201.json.gz",b)
    result = collect.collect(tmp_path/"data")
    design = result["summary"]["methods"]["design_charge_fixed"]
    assert design["M"]["total"] == 2 and design["M"]["valid"] == 1 and design["failed"] == 1
    assert design["failure_reasons"]["training_lines_changed"] == 1
    assert result["summary"]["methods"]["highlow"]["failed"] == 1
    pair = result["summary"]["paired_contrasts"]["charge_fixed_vs_constant"]["delta_M"]
    assert pair["total"] == 2 and pair["valid"] == 1 and pair["zero"] == 1
    assert result["summary"]["actual_solver_calls"]["total_AC_batches"]["sum"] == 40
    assert len(result["records"][0]["training_lines"]) == 4
