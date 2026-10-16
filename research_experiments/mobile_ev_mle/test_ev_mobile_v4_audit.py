"""Arithmetic, failure-denominator and tamper checks for read-only v4 audit."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_mobile_v4_audit as audit
from . import ev_mobile_v4_domain as nested
from .ev_mobile_service_domain import MobileServiceConfig


def predictive(power, *, score=1., objective_id="objective"):
    names = ("a", "b")
    nearest = [dict(a=name, half_gap=score, active=[other])
               for name, other in (("a", "b"), ("b", "a"))]
    references = [dict(a=a, success=True, fits={b: dict(success=True, gradient_reliable=True,
        loss=1. if b == a else 1. + 2. * score) for b in names}) for a in names]
    pairs = [dict(a=a, b=b, success=True, own_loss=1., rival_loss=1. + 2. * score,
                  half_gap=score, gradient_per_kw=np.zeros_like(power).tolist())
             for a in names for b in names if a != b]
    return dict(success=True, objective_id=objective_id, power_kw=np.asarray(power).tolist(),
        source_interface=1, uses_truth=False, uses_future_noise=False, uses_observed_D=True,
        score=score, nearest=nearest, references=references, pairs=pairs,
        aggregation=dict(success=True, score=score, nearest=deepcopy(nearest)))


def guard(before=1., after=1.):
    slack = after - before
    violated = [] if slack >= -1e-7 * (1. + abs(before)) else ["a", "b"]
    return dict(success=not violated, complete=True, slacks={"a": slack, "b": slack},
                violated=violated, tolerance=1e-7, floor={"a": before, "b": before},
                expected_references=["a", "b"])


def _fixture(monkeypatch, *, fixed_states=True):
    bank = {name: SimpleNamespace(name=name, n=1, m=1, supports=((0,),)) for name in ("a", "b")}
    D = dict(p=[[.1]] * 100, q=[[.03]] * 100, v=[[1.]] * 100,
             master_p=[.1] * 100, master_q=[.03] * 100, root_v=[1.] * 100)
    future = {key: values[:1] for key, values in D.items()}
    meter = dict(bias_p=0., bias_q=0., bias_v=0., v_sigma=1., master_p_sigma=1., master_q_sigma=1.)
    normalizer = 101 * 3 * np.log(2. * np.pi)
    state = dict(r=[.01], x=[.003], source_p=[0.] * 100, source_q=[0.] * 100)
    trained = [dict(name=name, supports=[[0]], loss=0., success=True, **deepcopy(state),
        p=deepcopy(D["p"]), q=deepcopy(D["q"]),
        predicted={k:deepcopy(D[k]) for k in ("p", "q", "v", "master_p", "master_q")}) for name in bank]
    blocks = {label: dict(loss=0., residual=np.zeros((T, 3)).tolist(), time_count=T,
        channel_loss=dict(v=0., master_p=0., master_q=0.)) for label, T in (("observed", 100), ("future", 1))}
    score = dict(loss=0., normalizer=normalizer, loglik=-.5 * normalizer, blocks=blocks,
                 line_domain_feasible=True, source_box_feasible=True, physical_voltage_feasible=True, usable=True)
    joint = dict(**deepcopy(score), success=True,
                 state={**deepcopy(state), "source_p":[0.] * 101, "source_q":[0.] * 101},
                 feasible_witness=deepcopy(score), selected_origin="local_optimizer")
    fixed = deepcopy(score)
    if fixed_states:
        fixed["state"] = deepcopy(joint["state"])
    config = MobileServiceConfig(n_slots=1, n_sites=1, dt_hours=1.,
        battery_capacity_kwh=2., initial_soc=.5, terminal_energy_kwh=1.,
        soc_min=0., soc_max=1., max_moves=1)
    domains = nested.build_nested_domains(config)
    power = [[0.]]; key = nested.action_key(power)
    profile = predictive(power)
    methods, families = {}, {}
    for family in audit.FAMILIES:
        logs = {}
        for domain, model in domains.items():
            x = np.zeros(model.model.n_variables)
            x[model.indices.energy] = 1.
            connected = next(i for i, arc in zip(model.indices.arc, model.indices.arcs)
                             if arc.kind == "connected" and arc.origin == 0)
            x[connected] = 1.
            physical = model.check(x)
            assert physical["feasible"]
            methods[family + "__" + domain] = dict(domain=domain, family=family,
                power_kw=power, operation_x=x.tolist(), operation=physical,
                action_key=key, planning_success=True, planning_failure=None,
                nominal=dict(success=True, score=1., nearest=deepcopy(profile["nearest"])))
            parent_names = nested.PARENTS[domain]
            if family == "square":
                logs[domain] = dict(Q=0., parent_Q=[0.] * len(parent_names), scalar_monotonicity_verified=True)
            else:
                logs[domain] = dict(domain=domain, guard=family == "guarded", planning_failure=None,
                    direct_parents=[dict(domain=p, score=1., action_key=key) for p in parent_names],
                    initials=[dict(origin="square_initializer", score=1., action_key=key,
                        reference_floor=guard() if parent_names else None, accepted=not parent_names)],
                    parent_score=1., parent_action=key, steps=[], selected_score=1., selected_action=key,
                    selected_origin="square_initializer", parent_best_score=1. if parent_names else None,
                    scalar_monotonicity_verified=True, selected_reference_gaps=deepcopy(profile["nearest"]))
        families[family] = dict(logs=logs)
        if family != "square":
            families[family].update(objective_id="objective", cache={key:deepcopy(profile)})
    comparisons = {name:dict(fixed=deepcopy(fixed), joint=deepcopy(joint), fixed_source_success=True) for name in bank}
    record = dict(spec=dict(index=0, phase="development", bank="n3", H=96, T=1, regimes=["regular"]),
        config=dict(base_kw=20., source_interface=1), observed=D, observed_meter=meter, future_meter=meter,
        operation_config=asdict(config), training_result=dict(fits=trained, failures={}),
        methods=methods, families=families, reporting_cache={key:deepcopy(profile)}, true_name="a", status="complete",
        actual=dict(regular=dict(actions={key:dict(future=future, evaluation=dict(complete=True, losses={"a":0.,"b":0.},
            comparisons=comparisons, M=0.))})))
    calls = []
    def replay(candidate, state, blocks, cfg):
        calls.append(candidate.name)
        return dict(loss=0., normalizer=sum(len(obs["p"]) for _,obs,_ in blocks)*3*np.log(2*np.pi),
            loglik=-.5*sum(len(obs["p"]) for _,obs,_ in blocks)*3*np.log(2*np.pi),
            line_domain_feasible=True, source_box_feasible=True, physical_voltage_feasible=True,
            max_AC_equation=0., max_power_balance=0.,
            blocks={name:dict(loss=0.,residual=np.zeros((len(obs["p"]),3)),channel_loss=dict(v=0.,master_p=0.,master_q=0.))
                    for name,obs,meter in blocks})
    monkeypatch.setattr(audit, "_bank", lambda record:bank)
    monkeypatch.setattr(audit.old, "replay_state", replay)
    return record, calls


def test_all_method_slots_kept_when_actions_share_one_physical_replay(monkeypatch):
    record,calls = _fixture(monkeypatch)
    result = audit.audit_record(record)
    assert result["success"], result["errors"]
    assert len(calls) == 6  # two observed + two fixed + two joint.
    assert result["metrics"]["operation_checks"] == 12
    assert result["metrics"]["actual_method_candidate_slots"] == 24
    assert result["metrics"]["actual_joint_unique_states"] == 2
    assert result["metrics"]["actual_fixed_unique_states"] == 2
    assert result["metrics"]["predictive_profiles_checked"] == 1
    assert result["optimizer_calls"] == result["production_AC_calls"] == result["new_simulation_calls"] == 0


def test_fixed_source_state_gap_is_explicit_despite_valid_residual_arithmetic(monkeypatch):
    record,calls = _fixture(monkeypatch, fixed_states=False)
    result = audit.audit_record(record)
    assert not result["success"]
    assert result["available_checks_passed"]
    assert result["complete_joint_physical_coverage"]
    assert result["metrics"]["fixed_missing_state_slots"] == 2
    assert result["metrics"]["fixed_residual_arithmetic_checks"] == 2
    assert len(result["missing"]) == 2
    assert len(calls) == 4


def test_fallback_and_failed_optimizer_keep_original_denominators(monkeypatch):
    record,_ = _fixture(monkeypatch)
    record["methods"]["guarded__mobile_bidir"]["planning_success"] = False
    record["methods"]["guarded__mobile_bidir"]["planning_failure"] = "family budget exhausted; parent retained"
    action = next(iter(record["actual"]["regular"]["actions"].values()))
    action["evaluation"]["comparisons"]["b"]["joint"]["success"] = False
    action["evaluation"]["complete"] = False
    result = audit.audit_record(record)
    assert result["success"], result["errors"]
    assert result["metrics"]["actual_method_candidate_slots"] == 24
    assert result["metrics"]["recorded_joint_fit_failure_slots"] == 12
    assert result["metrics"]["recorded_inference_failure_method_slots"] == 12
    assert result["metrics"]["recorded_planning_failure_slots"] == 1
    assert any(row["kind"] == "actual_joint_fit_failure" for row in result["recorded_failures"])


@pytest.mark.parametrize("tamper", ["SSE", "prediction", "domain", "route", "hash", "complete", "nominal", "fixed_line"])
def test_saved_physical_and_arithmetic_tampering_detected(monkeypatch, tamper):
    record,_ = _fixture(monkeypatch)
    method = record["methods"]["aligned__fixed_charge"]
    action = next(iter(record["actual"]["regular"]["actions"].values()))
    if tamper == "SSE":
        action["evaluation"]["comparisons"]["a"]["joint"]["loss"] = 1.
    elif tamper == "prediction":
        record["training_result"]["fits"][0]["predicted"]["v"][0] = [1.01]
    elif tamper == "domain":
        method["operation_x"][0] = 1.1
    elif tamper == "route":
        method["operation"]["route"][0]["departure"] = 1
    elif tamper == "hash":
        method["action_key"] = "wrong"
    elif tamper == "complete":
        action["evaluation"]["complete"] = False
    elif tamper == "nominal":
        method["nominal"]["score"] = 2.
    elif tamper == "fixed_line":
        action["evaluation"]["comparisons"]["a"]["fixed"]["state"]["r"] = [.02]
    result = audit.audit_record(record)
    assert not result["success"]
    assert result["errors"]


@pytest.mark.parametrize("tamper", ["gap", "mean", "missing_pair", "objective", "reference", "truth"])
def test_predictive_pair_and_aggregation_checks_reject_tamper(tamper):
    profile = predictive([[0.]])
    if tamper == "gap": profile["pairs"][0]["half_gap"] = 2.
    elif tamper == "mean": profile["score"] = 2.
    elif tamper == "missing_pair": profile["pairs"].pop()
    elif tamper == "objective": profile["objective_id"] = "different"
    elif tamper == "reference": profile["references"][0]["fits"].pop("b")
    elif tamper == "truth": profile["uses_truth"] = True
    with pytest.raises(ValueError):
        audit.check_predictive_profile(profile, ["a", "b"], key=nested.action_key([[0.]]), objective_id="objective")


def test_guard_checks_anchor_slacks_and_rejects_hidden_reference_sacrifice():
    before = predictive([[0.]], score=1.)
    after = predictive([[1.]], score=.5)
    rejected = guard(1., .5)
    assert not audit.check_reference_guard(rejected, before, after, ["a", "b"])
    rejected["success"] = True
    with pytest.raises(ValueError, match="guard acceptance"):
        audit.check_reference_guard(rejected, before, after, ["a", "b"])


def test_parent_acceptance_and_common_objective_reject_tamper(monkeypatch):
    record,_ = _fixture(monkeypatch)
    record["families"]["guarded"]["logs"]["mobile_bidir"]["direct_parents"][0]["score"] = 5.
    result = audit.audit_record(record)
    assert not result["success"]
    assert any("parent" in row["check"] for row in result["errors"])


def test_dev2_initializer_retains_parent_inside_relative_numerical_tolerance(monkeypatch):
    record,_ = _fixture(monkeypatch)
    family = record["families"]["aligned"]
    log = family["logs"]["fixed_bidir"]
    candidate = predictive([[1.]], score=1. + 1.05e-9)
    key = nested.action_key(candidate["power_kw"])
    family["cache"][key] = candidate
    log["domain_soft_quota_seconds"] = 75.
    log["initials"] = [dict(origin="square_initializer", score=candidate["score"], action_key=key,
        reference_floor=guard(1., candidate["score"]), accepted=False)]
    result = audit.check_family("aligned", family["logs"], record["methods"], family["cache"],
                                 ["a", "b"], "objective")
    assert result["domain_checks"] == 4


def test_cold_cache_duplicate_times_can_differ_without_scientific_difference(monkeypatch):
    record,_ = _fixture(monkeypatch)
    for index, family in enumerate(("aligned", "guarded")):
        profile = next(iter(record["families"][family]["cache"].values()))
        profile["seconds"] = index + 1.
        profile["references"][0]["fits"]["a"]["seconds"] = index + 3.
    result = audit.audit_record(record)
    assert result["success"], result["errors"]


def test_fatal_record_retains_predeclared_denominator(monkeypatch):
    record,_ = _fixture(monkeypatch)
    result = audit.audit_record(dict(spec=record["spec"], status="fatal_failure", failure="recorded exception"))
    assert not result["success"]
    assert result["expected"]["method_slots"] == 12
    assert result["expected"]["actual_method_candidate_slots"] == 24
    assert result["recorded_failure"] == "recorded exception"


def test_batch_keeps_missing_histories_and_protects_existing_output(monkeypatch, tmp_path):
    record,_ = _fixture(monkeypatch)
    directory = tmp_path / "batch"
    (directory / "development").mkdir(parents=True)
    frozen = dict(policy_id="p", source_sha256={},
        specifications=dict(development=[record["spec"], {**record["spec"], "index":1}]))
    audit.old.write(directory / "freeze.json", frozen)
    record["policy_id"] = "p"
    audit.old.write(directory / "development" / "000.json.gz", record)
    output = tmp_path / "audit.json"
    result = audit.run(directory, output)
    assert not result["success"] and not result["exact_case_coverage"]
    assert result["available_checks_passed"]
    assert result["expected_case_slots"] == 2 and result["saved_case_slots"] == 1
    with pytest.raises(FileExistsError):
        audit.run(directory, output)
