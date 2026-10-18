"""Seed replay and relative-statistic accounting; no scientific fits run."""
import gzip
import hashlib
import json

import numpy as np
import pytest

from .relative_bootstrap import (CANDIDATE_NAMES, REPEATS, _key,
                                 measurement_sha256, relative_statistic,
                                 replay_measurements, run_one, tasks)


def _sealed(network="paper15", regime="ordinary"):
    spec = dict(network=network, regime=regime, seed=31001, N=96, boundary="K")
    p = np.arange(12, dtype=float).reshape(4, 3)/1000.+.02
    fits = {name:dict(qualified=True, loss=loss, p=p.tolist(), q=(p*.3).tolist(),
                     v=(1.-p/10).tolist())
            for name, loss in zip(CANDIDATE_NAMES, (20., 10., 12., 14.))}
    return dict(spec=spec, fits=fits, truth_used=False,
                sigma=dict(p=[.0002]*3, q=[.0001]*3, v=[.0005]*3),
                root_known=[1.01]*4, domain=dict())


@pytest.mark.parametrize("network,regime", [("paper11", "ordinary"), ("paper15", "correlated")])
@pytest.mark.parametrize("candidate", CANDIDATE_NAMES)
def test_replayed_noise_exactly_matches_original_literal_rng_order(network, regime, candidate):
    sealed = _sealed(network, regime)
    index = list(CANDIDATE_NAMES).index(candidate)
    rng = np.random.default_rng(np.random.SeedSequence([31001, 170, index,
        11 if network == "paper11" else 15, 0 if regime == "ordinary" else 1]))
    for b, actual in replay_measurements(sealed, candidate, repeats=3):
        expected = {k:np.asarray(sealed["fits"][candidate][k])
                    + rng.normal(size=np.shape(sealed["fits"][candidate][k]))*sealed["sigma"][k]
                    for k in ("p", "q", "v")}
        for k in expected:
            np.testing.assert_array_equal(actual[k], expected[k])
        assert measurement_sha256(actual) == measurement_sha256(expected)
    assert b == 2


def test_statistic_requires_all_qualified_and_subtracts_best_candidate():
    sealed = _sealed()
    result = relative_statistic(sealed["fits"], "star")
    assert result["valid"] and result["statistic"] == 10.
    sealed["fits"]["pair12"]["qualified"] = False
    result = relative_statistic(sealed["fits"], "star")
    assert not result["valid"] and result["statistic"] is None
    assert result["invalid_candidates"] == ["pair12"]


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(obj).encode()
    path.write_bytes(gzip.compress(body, mtime=0) if path.suffix == ".gz" else body)


def test_saved_own_records_reused_and_rival_calls_match_frozen_rule(tmp_path):
    sealed = _sealed()
    spec = sealed["spec"]
    key = _key(spec)
    source = tmp_path/"inference"/(key+".json.gz")
    original_path = tmp_path/"bootstrap"/(key+"_star.json")
    _write(source, sealed)
    records = [dict(sse=15., success=True, stationary=True, seconds=.1) for _ in range(REPEATS)]
    records[3] = dict(sse=None, success=False, stationary=False, seconds=.1)
    original = dict(spec=spec, candidate="star", status="completed", observed_sse=20.,
                    records=records, decision=dict(pvalue=.5), seconds=4.)
    _write(original_path, original)
    before = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, original_path)}
    calls = []
    def fake_fit(candidate, boot, sigma, domain, **kwargs):
        calls.append((candidate.name, kwargs))
        assert candidate.name != "star"
        assert kwargs["multistart"] == 2 and kwargs["max_iter"] == 160
        assert kwargs["seed"] == 31001+CANDIDATE_NAMES.index(candidate.name)
        return dict(status="qualified", qualified=True,
                    loss={"pair01":12., "pair02":13., "pair12":14.}[candidate.name],
                    seconds=.001, ac_calls=2, projected_gradient=0.)
    summary = run_one(tmp_path, spec, "star", fit_function=fake_fit)
    assert summary["status"] == "completed"
    assert summary["extra_fit_calls"] == 38*3
    assert summary["extra_ac_calls"] == 38*3*2
    assert summary["reused_own_fit_count"] == 39
    assert summary["decision"]["pvalue"] == .05
    assert summary["decision"]["failed_or_nonstationary_replicates"] == 1
    assert len(calls) == 38*3
    assert not (tmp_path/"truth").exists()
    for path, digest in before.items():
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == digest
    saved = json.loads((tmp_path/"relative_bootstrap"/(key+"_star.json")).read_bytes())
    assert len(saved["records"]) == 39
    assert saved["records"][3]["valid"] is False
    assert saved["records"][3]["sse"] is None
    assert saved["records"][3]["skipped_competitor_fit_count"] == 3
    assert all(row["sse"] == 3. for row in saved["records"] if row["valid"])


def test_only_four_declared_histories_and_sixteen_candidate_slots():
    assert len(tasks()) == 16
    assert len({(_key(spec)) for spec, _ in tasks()}) == 4
    assert all(spec["N"] == 96 and spec["boundary"] == "K" and spec["seed"] == 31001
               for spec, _ in tasks())
