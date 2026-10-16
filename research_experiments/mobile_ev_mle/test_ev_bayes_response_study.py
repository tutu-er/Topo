"""Independent integration checks for the frozen experiment boundary."""
import numpy as np

from . import ev_bayes_response_study as study
from . import ev_bayes_response_cases as cases
from .symmetric_ac_local import mean_and_jacobian, whiten_observations


def test_random_stream_namespaces_are_disjoint_and_reproducible():
    values = []
    for phase in ("development", "calibration", "evaluation"):
        for spec in cases.specifications(phase):
            for tag in (1000, 2000, 3000, 4000, 5000, 6000):
                values.append(tuple(study.stream_rng(spec["seed"], tag).integers(2**32, size=4)))
    assert len(values) == len(set(values))
    assert np.array_equal(study.stream_rng(1, 2000).normal(size=10),
                          study.stream_rng(1, 2000).normal(size=10))


def test_credible_membership_is_minimal_and_exposes_uncertainty():
    p = np.array([[.91, .03, .03, .03], [.4, .35, .2, .05], [.25]*4])
    members = study.credible_membership(p, .9)
    assert members.sum(axis=1).tolist() == [1, 3, 4]
    assert np.all((members*p).sum(axis=1) >= .9)


def test_future_meter_reconstruction_and_ev_counted_once():
    spec = cases.specifications("development")[0]
    case = cases.make_case(spec)
    sl = slice(case["future_start"], None)
    power = np.zeros((8, 3)); power[:, 0] = 2.
    f = mean_and_jacobian(case["truth"]["candidate"], case["truth"]["r"],
        case["truth"]["x"], case["p"][sl]+power/case["base_kw"], case["q"][sl],
        case["source_p"][sl], case["source_q"][sl], case["root_v"][sl],
        case["fm"], with_jacobian=False)
    white = study.stream_rng(spec["seed"], 4000).normal(size=len(f["mean"]))
    obs = study.first_observation(f["physical"], case["fm"], white)
    assert np.allclose(whiten_observations(obs, case["fm"]), f["mean"]+white,
                       atol=1e-10, rtol=1e-12)
    assert np.allclose(f["physical"]["p"]-case["p"][sl], power/case["base_kw"])


def test_actual_summary_does_not_count_repeated_noise_as_new_histories():
    T, n, K = 8, 3, 4
    d = T*(3*n+2)
    bank = dict(means=np.zeros((1,K,d)), covariances=np.broadcast_to(np.eye(d),(1,K,d,d)),
                prior=np.array([.91,.03,.03,.03]), log_prior=np.log([.91,.03,.03,.03]), weights=np.ones(1))
    row = study.evaluate_actual(bank, np.zeros(d), np.zeros((32,d)), true_index=1)
    assert row["n_independent_histories"] == 1 and row["n_noise"] == 32
    assert row["summary"]["true_rejected"] == 1
    assert row["summary"]["wrong_unique"] == 1
    assert np.allclose(row["samples"]["posterior"], bank["prior"])
    assert row["summary"]["voltage_interval90_coverage"] == 1
