"""Tamper, failure-denominator and saved-model checks for AC evidence replay."""
from copy import deepcopy
from dataclasses import asdict
import base64
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pytest
from . import information_replay as replay
from .information_context import quality_models, meter_to_record, embed_star
from .test_information_context import fixture
from .likelihood import Observations, ac_predict, concat_observations
from .single_site_milp import SingleSiteConstraints, SingleSitePlan, constant_site_plan


def record_obs(obs):
    return {key: None if getattr(obs, key) is None else np.asarray(getattr(obs, key)).tolist()
            for key in ("p", "q", "v", "root_v", "master_p", "master_q", "source_p", "source_q")}


def saved_trial():
    history, meter, candidates, fits = fixture()
    h = history.t
    star, fit = candidates[0], fits["star"]
    seed = 883001
    streams = np.random.SeedSequence(seed).spawn(5)
    seeds = {name: int(stream.generate_state(1)[0]) for name, stream in zip(
        ("generator", "policy", "evaluation_noise", "public_quality", "design_integration"), streams)}
    public = np.random.default_rng(seeds["public_quality"])
    public.choice(("uniform", "temporal", "spatial"), p=[1/3]*3)
    permutation = public.permutation(4).tolist()
    future_model, joint_model, quality = quality_models(history, meter, "spatial", row_permutation=permutation)
    noise = np.random.default_rng(seeds["evaluation_noise"])
    white = {name: noise.standard_normal((4, 3) if name in ("p", "q", "v") else (4,))
             for name in ("p", "q", "v", "master_p", "master_q")}
    p = np.vstack((fit.p, np.tile(fit.p.mean(axis=0), (4, 1))))
    q = np.vstack((fit.q, np.tile(fit.q.mean(axis=0), (4, 1))))
    sp = np.r_[fit.source_p, np.full(4, fit.source_p.mean())]
    sq = np.r_[fit.source_q, np.full(4, fit.source_q.mean())]
    truth = dict(supports=[], r_pu=fit.r.tolist(), x_pu=fit.x.tolist(), p=p.tolist(), q=q.tolist(),
                 source_p=sp.tolist(), source_q=sq.tolist())
    constraints = SingleSiteConstraints()
    constant = constant_site_plan(0, 4, 3, constraints)["plan"]
    zero = SingleSitePlan(0, np.zeros((4, 3)), np.zeros((4, 3)), np.zeros(4, int), np.full(5, constraints.initial_soc))
    def fitted_row(candidate, observed, model, actual_p, actual_q, actual_sp, actual_sq):
        if candidate.name == "star":
            r, x = fit.r.copy(), fit.x.copy()
        else:
            r, x, target = embed_star(star, fit, candidate)
            e = candidate.supports.index(target)
            r[e] = x[e] = 1e-6
        physical = ac_predict(candidate, r, x, actual_p, actual_q, actual_sp, actual_sq)
        predicted, cost, normalizer = {}, 0., 0.
        for channel in ("p", "q", "v", "master_p", "master_q"):
            mean = physical[channel]+(getattr(model, "bias_"+channel) if channel in ("p", "q", "v") else 0)
            sigma = np.broadcast_to(getattr(model, channel+"_sigma"), mean.shape)
            predicted[channel] = mean.tolist()
            cost += float(np.sum(((mean-getattr(observed, channel))/sigma)**2))
            normalizer += float(np.sum(np.log(2*np.pi*sigma**2)))
        return dict(name=candidate.name, candidate=candidate.name, supports=[list(s) for s in candidate.supports],
            r=r.tolist(), x=x.tolist(), p=actual_p.tolist(), q=actual_q.tolist(), source_p=actual_sp.tolist(), source_q=actual_sq.tolist(),
            bias_p=np.asarray(model.bias_p).tolist(), bias_q=np.asarray(model.bias_q).tolist(), bias_v=np.asarray(model.bias_v).tolist(),
            loss=cost, half_loss=.5*cost, loglik=-.5*(cost+normalizer), success=True,
            predicted=predicted, global_MLE_certified=False,
            diagnostics=dict(global_optimality_certified=False, min_voltage=float(abs(physical["voltage"]).min()), max_voltage=float(abs(physical["voltage"]).max())))
    hfits = [fitted_row(candidate, history, meter, fit.p, fit.q, fit.source_p, fit.source_q) for candidate in candidates]
    cache, rows = {}, []
    for method in replay.METHODS:
        plan = zero if method == "none" else constant
        action = plan.summary()
        action = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in action.items()}
        physical = ac_predict(star, fit.r, fit.x, p[h:]+plan.p, q[h:]+plan.q, sp[h:], sq[h:])
        values = {name: physical[name]+(getattr(future_model, "bias_"+name) if name in ("p", "q", "v") else 0)
                  +np.broadcast_to(getattr(future_model, name+"_sigma"), physical[name].shape)*white[name] for name in white}
        future = Observations(**values)
        joint = concat_observations(history, future)
        key = plan.p.tobytes()
        cached = key in cache
        if not cached:
            cache[key] = [fitted_row(candidate, joint, joint_model, np.vstack((fit.p, p[h:]+plan.p)),
                np.vstack((fit.q, q[h:]+plan.q)), sp, sq) for candidate in candidates]
        joint_fits = cache[key]
        losses = {row["name"]: row["loss"] for row in joint_fits}
        minimum = min(losses.values())
        rows.append(dict(method=method, status="complete", action=action, fits=deepcopy(joint_fits), losses=losses,
            gaps={name: value-minimum for name, value in losses.items()}, winner=min(losses, key=lambda name:(losses[name], name)),
            true_gap=losses["star"]-minimum, signed_margin_full_SSE=min(value for name, value in losses.items() if name != "star")-losses["star"],
            observations=record_obs(future), equivalent_action_fit_cache_hit=cached,
            fit_calls=0 if cached else 4, ac_calls=0,
            operational_voltage_ok=True, voltage_min=float(abs(physical["voltage"]).min()), voltage_max=float(abs(physical["voltage"]).max())))
    config = dict(future_points=4, population=dict(quality_mass=[1/3]*3))
    source = {}
    policy = hashlib.sha256(json.dumps(dict(config=config, source_sha256=source), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(seed=seed, history_id=str(seed), run_id=str(seed)+"_spatial", case="star", quality="spatial",
        config=config, source_sha256=source, policy_id=policy, seeds=seeds, quality_record=quality,
        history_observations=record_obs(history), measurement_models=dict(history=meter_to_record(meter), future=meter_to_record(future_model), joint=meter_to_record(joint_model)),
        candidate_names=[candidate.name for candidate in candidates], historical_fits=hfits, historical_failures={},
        historical_cost=dict(ac_calls=0), selection=dict(selector_ac_calls=0, selector_uses_truth=False, selector_uses_future_noise=False,
            methods={method:dict(status="optimized") for method in replay.METHODS}),
        selection_uses_truth=False, selection_uses_future_noise=False, truth=truth, true_covered=True, true_name="star",
        device_constraints=asdict(constraints), records=rows, complete=True,
        relative_score=max(row["true_gap"] for row in rows), calibration_score_is_infinite=False,
        actual_joint_fit_calls=8, joint_unique_action_count=2, total_ac_calls=0,
        white_evaluation_noise={name:value.tolist() for name,value in white.items()})


def test_self_contained_saved_actual_meter_replay():
    trial = saved_trial()
    result = replay.audit_trial(trial)
    assert result["success"], result["errors"]
    assert result["recorded_fit_replay_count"] == 44
    assert result["actual_unique_fit_replay_count"] == 12
    assert result["quality_audit"]["V_raw_read_count"] == 60


@pytest.mark.parametrize("what", ["sigma", "bias", "fit_loss", "noise", "operation", "cache", "count"])
def test_tampered_record_detected(what):
    trial = saved_trial()
    if what == "sigma":
        trial["measurement_models"]["joint"]["v_sigma"][-1][0] *= 2
    elif what == "bias":
        trial["measurement_models"]["future"]["bias_v"][0] *= .5
    elif what == "fit_loss":
        trial["records"][0]["fits"][0]["loss"] += 1.
    elif what == "noise":
        trial["white_evaluation_noise"]["v"][0][0] += 1.
    elif what == "operation":
        trial["records"][1]["action"]["soc"][-1] += .05
    elif what == "cache":
        trial["records"][2]["equivalent_action_fit_cache_hit"] = False
    else:
        trial["actual_joint_fit_calls"] += 4
    result = replay.audit_trial(trial)
    assert not result["success"] and result["errors"]


def test_new_declared_method_is_not_hardcoded_to_ten():
    trial = saved_trial()
    row = deepcopy(trial["records"][-1])
    row["method"] = "uniform_information"
    trial["records"].append(row)
    trial["selection"]["methods"][row["method"]] = dict(status="optimized")
    result = replay.audit_trial(trial)
    assert result["success"], result["errors"]
    assert len(result["methods"]) == 11


def test_calibration_decisions_and_overlap_checked():
    trial = saved_trial()
    gate = dict(policy_id=trial["policy_id"], calibration_history_ids=["940001"], relative=dict(vacuous=False, threshold=1e6))
    for row in trial["records"]:
        row.update(retained_candidates=trial["candidate_names"], false_rejection=False, only_truth_retained=False, unresolved=True)
    trial["any_method_false_rejection"] = False
    result = replay.audit_trial(trial, calibration=gate)
    assert result["success"], result["errors"]
    trial["records"][0]["false_rejection"] = True
    assert not replay.audit_trial(trial, calibration=gate)["success"]
    trial["records"][0]["false_rejection"] = False
    gate["calibration_history_ids"] = [trial["history_id"]]
    assert "calibration_evaluation_overlap" in replay.audit_trial(trial, calibration=gate)["errors"]


def test_recorded_joint_failure_keeps_infinite_calibration_mass():
    trial = saved_trial()
    failed = trial["records"][-1]
    failed.update(status="joint_fit_failure")
    failed["fits"][0]["success"] = False
    # Different action is unnecessary for this failure fixture; retain cache consistency.
    for row in trial["records"][1:]:
        row["fits"][0]["success"] = False
        row["losses"].pop("star")
        row["status"] = "joint_fit_failure"
    trial.update(complete=False, calibration_score_is_infinite=True, relative_score=None)
    result = replay.audit_trial(trial)
    assert result["success"], result["errors"]
    assert result["calibration_score_is_infinite"] and result["recomputed_relative_score"] is None


def test_rank_replay_uses_independent_history_and_failure_mass():
    audits = [dict(history_id=str(index), calibration_score_is_infinite=index == 18,
                   recomputed_relative_score=None if index == 18 else float(index)) for index in range(19)]
    saved = dict(calibration_history_ids=[str(index) for index in range(19)], failed_history_count=1, all_history_count=19,
                 relative=dict(rank=18, sample_count=19, vacuous=False, threshold=17., rank_tail_bound=.1))
    result = replay.audit_calibration(audits, saved, .1)
    assert result["success"] and result["failed_history_count"] == 1
    audits[-1]["history_id"] = "0"
    assert not replay.audit_calibration(audits, saved, .1)["success"]


def test_source_snapshot_check_is_read_only(tmp_path):
    root = tmp_path/"research_experiments"/"mobile_ev_mle"
    root.mkdir(parents=True)
    original = b"x = 1\n"
    path = root/"unit.py"
    path.write_bytes(original)
    hashes = {"unit.py":hashlib.sha256(original).hexdigest()}
    snap = tmp_path/"snapshot.json"
    replay.write_json(snap, dict(source_sha256=hashes, files={"unit.py":base64.b64encode(original).decode()}))
    assert replay.audit_source(hashes, source_root=root, snapshot=snap)["success"]
    path.write_bytes(b"x = 2\n")
    before = path.read_bytes()
    assert not replay.audit_source(hashes, source_root=root, snapshot=snap)["success"]
    assert replay.audit_source(hashes, source_root=root, snapshot=snap, check_current_bytes=False)["success"]
    assert path.read_bytes() == before
    assert replay.audit_source(hashes, source_root=root, snapshot=snap)["restoration_attempted"] is False