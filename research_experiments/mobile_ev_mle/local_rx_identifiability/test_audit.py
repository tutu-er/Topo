"""Audit failures must be detected independently of a saved success flag."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .. import likelihood as lk
from .cases import candidate_bank, public_domain
from .audit import audit_directory, _bootstrap_recompute, _key


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()).encode()
    path.write_bytes(gzip.compress(body, mtime=0) if path.suffix == ".gz" else body)


def _load(path):
    body = path.read_bytes()
    return json.loads(gzip.decompress(body) if path.suffix == ".gz" else body)


def _pilot_fixture(tmp_path):
    specs = [dict(network="paper15", regime="ordinary", seed=31001, N=96, boundary=b) for b in ("K", "U")]
    protocol = Path(__file__).with_name("protocol.md")
    _save(tmp_path/"configuration.json", dict(tasks=specs,
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest()))
    count = 96
    root = 1.01 + .005*np.sin(np.arange(count)*2*np.pi/count)
    p = np.tile([.03, .04, .035], (count, 1))
    q = .35*p
    sigma = dict(p=np.full(3, .0003), q=np.full(3, .00015), v=np.full(3, .0005))
    candidates = candidate_bank()
    true_prediction = lk.ac_predict(candidates[0], np.full(4, .04), np.full(4, .025), p, q, root_v=root)
    obs = dict(p=p, q=q, v=true_prediction["v"])
    domain = public_domain()
    domain.update(power_caps=np.full(3, .15))
    for spec in specs:
        fits, certificates = {}, {}
        for c in candidates:
            r, x = np.full(c.m, .04), np.full(c.m, .025)
            prediction = lk.ac_predict(c, r, x, p, q, root_v=root)
            loss = float(np.sum(((prediction["v"]-obs["v"])/sigma["v"])**2))
            fits[c.name] = dict(status="qualified", qualified=True, r=r, x=x, p=p, q=q,
                root_v=root, v=prediction["v"], loss=loss, truth_used=False,
                information=dict(rank=6 if spec["boundary"] == "U" and c.name == "star" else 2*c.m))
            certificates[c.name] = dict(alpha_results=[dict(alpha=.05, status="completed",
                raw=dict(sse_lower=0., threshold=100., reject=False),
                residual=dict(qp_lower=0., threshold=100., reject=False))])
        key = _key(spec)
        inference_path = tmp_path/"inference"/(key+".json.gz")
        inference = dict(spec=spec, observed=obs, sigma=sigma, domain=domain,
            root_known=root if spec["boundary"] == "K" else None,
            fits=fits, certificates=certificates, truth_used=False)
        _save(inference_path, inference)
        _save(tmp_path/"truth"/(key+".json.gz"), dict(candidate_name="star", **obs, root_v=root))
        _save(tmp_path/"results"/(key+".json.gz"), dict(spec=spec, status="completed",
            inference_sha256=hashlib.sha256(inference_path.read_bytes()).hexdigest(), diagnostics={}))
    return specs


def _reseal(tmp_path, spec, inference):
    key = _key(spec)
    path = tmp_path/"inference"/(key+".json.gz")
    _save(path, inference)
    result_path = tmp_path/"results"/(key+".json.gz")
    result = _load(result_path)
    result["inference_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _save(result_path, result)


def test_valid_saved_physical_states_replay_without_any_fit(tmp_path):
    _pilot_fixture(tmp_path)
    report = audit_directory(tmp_path)
    assert report["audit_pass"], report["issues"]
    assert report["configured_slots"] == 2
    assert report["base_condition_trajectories"] == 1
    assert report["independent_random_families"] == 1
    assert "independent_base_trajectories" not in report
    assert report["fit_calls"] == 0
    assert report["newly_generated_samples"] == 0
    assert report["configured_candidate_slots"] == 8
    assert all(f["independent_voltage_max_error"] < 2e-11
               for slot in report["slots"] for f in slot["fits"].values())
    assert (tmp_path/"audit.json").is_file()


def test_loss_tampering_is_found_even_after_hash_reseal(tmp_path):
    specs = _pilot_fixture(tmp_path)
    path = tmp_path/"inference"/(_key(specs[0])+".json.gz")
    inference = _load(path)
    inference["fits"]["star"]["loss"] += 1.
    _reseal(tmp_path, specs[0], inference)
    report = audit_directory(tmp_path)
    assert not report["audit_pass"]
    assert "loss_mismatch" in {issue["code"] for issue in report["issues"]}
    assert len(report["slots"]) == 2


def test_truth_field_leak_found_even_when_hashes_match(tmp_path):
    specs = _pilot_fixture(tmp_path)
    path = tmp_path/"inference"/(_key(specs[1])+".json.gz")
    inference = _load(path)
    inference["provenance"] = dict(truth=dict(r=[.1]))
    _reseal(tmp_path, specs[1], inference)
    report = audit_directory(tmp_path)
    assert not report["audit_pass"]
    assert "truth_field_leak" in {issue["code"] for issue in report["issues"]}


def test_scientific_failure_remains_a_slot(tmp_path):
    specs = _pilot_fixture(tmp_path)
    path = tmp_path/"results"/(_key(specs[1])+".json.gz")
    _save(path, dict(spec=specs[1], status="failed", error="retained solver failure"))
    report = audit_directory(tmp_path)
    assert report["failed_slots"] == 1
    assert report["completed_slots"] == 1
    assert report["configured_slots"] == 2
    assert len(report["slots"]) == 2
    assert report["slots"][1]["failure_preserved"]


def test_unknown_boundary_rank_inflation_is_found(tmp_path):
    specs = _pilot_fixture(tmp_path)
    path = tmp_path/"inference"/(_key(specs[1])+".json.gz")
    inference = _load(path)
    inference["fits"]["star"]["information"]["rank"] = 8
    _reseal(tmp_path, specs[1], inference)
    report = audit_directory(tmp_path)
    assert "unknown_boundary_rank" in {issue["code"] for issue in report["issues"]}


def test_bootstrap_failed_or_nonstationary_runs_count_in_upper_tail():
    records = [dict(sse=2., success=True, stationary=True),
               dict(sse=20., success=True, stationary=True),
               dict(sse=0., success=False, stationary=True),
               dict(sse=0., success=True, stationary=False),
               dict(sse=None, success=True, stationary=True)]
    result = _bootstrap_recompute(10., records)
    assert result == dict(pvalue=5/6, failed=3, valid=2, replicates=5, upper_tail=4)


def test_condition_trajectories_are_distinguished_from_randomness_families(tmp_path):
    tasks = [dict(network=network, regime=regime, seed=seed, N=n, boundary=boundary)
             for network in ("paper15", "paper11") for regime in ("ordinary", "correlated")
             for seed in (31001, 31002, 31003) for n in (96, 288) for boundary in ("K", "U")]
    protocol = Path(__file__).with_name("protocol.md")
    _save(tmp_path/"configuration.json", dict(tasks=tasks,
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest()))
    # Missing evidence is intentionally retained as errors; only count metadata
    # is tested here, without creating or fitting any scientific observations.
    result = audit_directory(tmp_path)
    assert not result["audit_pass"]
    assert result["configured_slots"] == 48
    assert result["base_condition_trajectories"] == 12
    assert result["independent_random_families"] == 6
    assert len(result["slots"]) == 48
