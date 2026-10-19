"""Thin real-fit integration and non-anticipating driver/snapshot contracts."""
import base64
from copy import deepcopy
import hashlib

import numpy as np

from . import likelihood as lk
from . import ev_fixed_train_study as study
from . import gradient_price_study as io
from .ev_fixed_train_fit import subset_model, subset_observations


def _small_case():
    candidate = lk.RadialCandidate.from_supports("pair01", 3, [(0, 1)])
    rng = np.random.default_rng(117)
    t = 12
    p, q = rng.uniform(.03, .14, (t, 3)), rng.uniform(-.01, .07, (t, 3))
    r, x = np.linspace(.003, .009, candidate.m), np.linspace(.001, .004, candidate.m)
    sp, sq = rng.uniform(.025, .075, t), rng.uniform(-.005, .025, t)
    root = rng.uniform(.997, 1.004, t)
    meter = lk.MeasurementModel(.0005, .0004, .0001, .001, .001)
    phy = lk.ac_predict(candidate, r, x, p, q, sp, sq, root)
    observations = lk.Observations(p, q, phy["v"], root, phy["master_p"], phy["master_q"])
    options = dict(multistart=1, max_nfev=250, tol=1e-10,
        r_bounds=(1e-6, .1), x_bounds=(1e-6, .1), source_p_bounds=(-.5, .5), source_q_bounds=(-.5, .5))
    return candidate, observations, meter, options


def test_real_fit_bank_aggregates_independent_review_and_fixes_validation_theta():
    candidate, obs, meter, options = _small_case()
    train = subset_observations(obs, slice(0, 8))
    tm = subset_model(meter, obs, slice(0, 8))
    trained, tr = study.fit_bank([candidate], train, tm, options=options, seed=314)
    assert tr["all_complete"] and not tr["failures"]
    assert len(tr["fits"]) == len(tr["independent_review"]) == 1
    # review_fixed_fit already supplies candidate; dictionary aggregation must
    # preserve it without raising duplicate-key errors or losing the fit.
    assert tr["independent_review"][0]["candidate"] == candidate.name
    assert tr["independent_review"][0]["success"]
    assert tr["cost"]["fit_calls"] == 1
    validation = subset_observations(obs, slice(8, None))
    vm = subset_model(meter, obs, slice(8, None))
    checked, vr = study.fit_bank([candidate], validation, vm, trained=trained,
        options=options, seed=315)
    assert vr["all_complete"] and vr["independent_review"][0]["trained_impedance_checked"]
    np.testing.assert_array_equal(checked[candidate.name].r, trained[candidate.name].r)
    np.testing.assert_array_equal(checked[candidate.name].x, trained[candidate.name].x)
    np.testing.assert_array_equal(checked[candidate.name].p, validation.p)


def test_driver_passes_only_train_and_adds_ev_once_retaining_failure_and_curve_cache(monkeypatch):
    seed = 990101
    actual_case = study.base.make_case(seed, study.base.FAMILIES[0])
    bank = [lk.RadialCandidate.from_supports(name, 3, family) for name, family in zip(
        ("star", "clade_01", "clade_02", "clade_12"), study.base.FAMILIES)]
    config = deepcopy(study.CONFIG)
    config["training"]["multistart"] = 1
    config["source_profile"]["multistart"] = 1
    captured = {}

    def proposal(train, **kwargs):
        assert train.t == 8
        np.testing.assert_array_equal(train.p, actual_case["history"].p[:8])
        return bank, {"test_scope": "deterministic complete bank; WZZT tested elsewhere"}

    def selector(train, candidates, trained, train_meter, future_meter, **kwargs):
        assert train.t == 8 and all(f.p.shape == (8, 3) for f in trained.values())
        assert not ({"validation", "truth", "future_noise", "case"} & set(kwargs))
        np.testing.assert_array_equal(train.v, actual_case["history"].v[:8])
        captured["training_times"] = train.t
        choices = {}
        power = np.zeros((6, 3));power[:, 0] = 2.
        for method in study.METHODS:
            choices[method] = None if method == "random_mobile" else (
                np.zeros_like(power) if method == "none" else power.copy())
        return choices, {"mock_selector": True, "purpose": "driver boundary/cache test"}

    monkeypatch.setattr(study, "propose", proposal)
    monkeypatch.setattr(study, "select_fixed_controls", selector)
    record = study.run_trial(seed, 0, config=config)
    assert captured["training_times"] == 8
    assert record["training_indices"] == list(range(8))
    assert record["validation_indices"] == list(range(8, 12))
    np.testing.assert_array_equal(record["validation"]["p"], actual_case["history"].p[8:12])
    assert not record["validation_used_by_selector"] and record["source_unchanged"]
    assert len(record["methods"]) == 11 and record["unique_actual_curves"] == 2
    rows = {r["name"]:r for r in record["methods"]}
    assert rows["random_mobile"]["status"] == "selection_failure"
    assert rows["random_mobile"]["power_kw"] is None and record["status"] == "partial_failure"
    assert sum(r.get("actual_fit_cache_hit", False) for r in record["methods"]) == 8
    assert not rows["none"]["actual_fit_cache_hit"] and not rows["constant"]["actual_fit_cache_hit"]
    assert rows["quality_fixed"]["actual_fit_cache_hit"]
    expected = actual_case["p"][12:]+np.asarray(rows["constant"]["power_kw"])/study.base.BASE_KW
    np.testing.assert_allclose(rows["constant"]["physical"]["p"], expected, atol=1e-16, rtol=0)
    expected_meter_p = expected+np.asarray(actual_case["fm"].p_sigma)*actual_case["white"]["p"][12:]
    expected_meter_p += np.asarray(actual_case["fm"].bias_p)
    np.testing.assert_allclose(rows["constant"]["observations"]["p"], expected_meter_p, atol=1e-16, rtol=0)
    np.testing.assert_array_equal(rows["constant"]["physical"]["q"], actual_case["q"][12:])


def test_fit_bank_retains_nonconverged_candidate_and_failure_denominator(monkeypatch):
    candidate, obs, meter, options = _small_case()
    bad = lk.RadialCandidate.from_supports("forced_budget_stop", 3, [(0, 1)])
    actual_fit = study.fit_fixed_inputs

    def bounded_fit(candidate, *args, **kwargs):
        if candidate.name == bad.name:
            kwargs["max_nfev"] = 1
        return actual_fit(candidate, *args, **kwargs)

    monkeypatch.setattr(study, "fit_fixed_inputs", bounded_fit)
    fits, record = study.fit_bank([candidate, bad], obs, meter, options=options, seed=316)
    assert len(fits) == len(record["fits"]) == record["cost"]["fit_calls"] == 2
    assert set(fits) == {candidate.name, bad.name}
    assert not record["all_complete"] and record["failures"] == {bad.name:"fit_nonconverged"}
    assert len(record["independent_review"]) == 1
    assert not fits[bad.name].success and np.isfinite(fits[bad.name].loss)
    assert record["fits"][1]["diagnostics"]["restarts"][0]["status"] == 0


def test_actual_freeze_snapshot_restores_source_bytes_including_repository_core(tmp_path):
    directory = tmp_path/"freeze"
    config = {"version":"thin_freeze_snapshot_test", "selector_used":False}
    freeze = study.freeze_directory(directory, config=config)
    assert study.freeze_directory(directory, config=config) == freeze
    snapshot = io.read_json(directory/"source_snapshot.json.gz")
    assert freeze["source_count"] == len(snapshot["source_sha256"]) == len(snapshot["source_base64"])
    assert snapshot["source_sha256"] == freeze["source_sha256"]
    core_names = [name for name in freeze["source_sha256"] if name.startswith("rnj_wzzt_core/")]
    assert core_names
    for name, expected_sha in freeze["source_sha256"].items():
        source = study.HERE.parents[1]/name if name.startswith("rnj_wzzt_core/") else study.HERE/name
        restored = base64.b64decode(snapshot["source_base64"][name])
        assert restored == source.read_bytes()
        assert hashlib.sha256(restored).hexdigest() == expected_sha
