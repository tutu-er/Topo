from copy import deepcopy

import numpy as np
import pytest

from . import ev_response_ranking_diagnose as diag
from .ev_fixed_train_fit import fit_fixed_inputs
from .information_context import meter_to_record
from .likelihood import MeasurementModel, Observations, RadialCandidate, ac_predict


def test_factorial_main_interactions_and_average_identity():
    values = {}
    for mask in diag.SCENARIOS:
        t, l, s = map(int, mask)
        values[mask] = 5+2*t-3*l+4*s+7*t*l-2*t*s+6*l*s+8*t*l*s
    out = diag.factorial(values)
    assert out["effects"] == dict(theta=2., load=-3., source=4., theta_load=7., theta_source=-2., load_source=6., theta_load_source=8.)
    assert out["total_change"] == 22.
    assert abs(out["identity_error"]) < 1e-12 and abs(out["average_identity_error"]) < 1e-12
    assert out["average_replacement"]["theta"] == pytest.approx(2+7/2-2/2+8/3)


def test_rank_reversal_ties_and_small_denominators():
    out = diag.rank_agreement(dict(a=1., b=2., c=3.), dict(a=3., b=2., c=1.))
    assert out["spearman"] == pytest.approx(-1.) and out["counts"]["discordant"] == 3
    tie = diag.rank_agreement(dict(a=1., b=1.), dict(a=1., b=2.))
    assert tie["spearman"] is None and tie["counts"]["predicted_tie"] == 1
    assert diag.rank_agreement(dict(a=1.), dict(a=2.))["pairs"] == 0


def test_EV_increment_removes_additive_background_shift():
    actual = np.array([[.99, .98, .97, .4, .2]])
    base = np.array([[1., 1., 1., .1, .1]])
    predicted, predicted_base = actual+.02, base+.02
    error = diag.response_errors(predicted, predicted_base, actual, base, np.ones((1, 5)), 3)
    assert error["absolute"]["whitened_SSE"] > 0.
    assert error["increment"]["whitened_SSE"] < 1e-28


def _small_saved_trial():
    n, T = 3, 2
    bank = [RadialCandidate.from_supports("star", n, ()),
        *[RadialCandidate.from_supports("clade_"+"".join(map(str, pair)), n, (pair,))
            for pair in ((0, 1), (0, 2), (1, 2))]]
    p = np.array([[.10, .11, .12], [.11, .10, .13]])
    q = p*.25; source = np.array([[.09, .03], [.10, .04]])
    u = np.array([[2., 0., 0.], [1., 0., 0.]])
    rtrue = np.array([.010, .012, .009, .011]); xtrue = rtrue*.25
    physical = ac_predict(bank[0], rtrue, xtrue, p+u/20, q, source[:, 0], source[:, 1], np.ones(T))
    meter = MeasurementModel(.001, .001, .0005, .002, .002,
        bias_p=[.0001, 0., -.0001], bias_q=[0., .0001, 0.], bias_v=[.00001, 0., -.00001])
    obs = Observations(p=physical["p"]+meter.bias_p, q=q+meter.bias_q,
        v=physical["v"]+meter.bias_v, master_p=physical["master_p"], master_q=physical["master_q"], root_v=np.ones(T))
    config = dict(training_points=1, validation_points=1, horizon=2,
        profile=dict(multistart=2, max_nfev=150, tol=1e-10),
        source=dict(source_p_bounds=(-.5, .5), source_q_bounds=(-.5, .5)))
    training, validation, clean = [], [], []
    for index, candidate in enumerate(bank):
        r = np.full(candidate.m, .0105); x = r*.25
        initial = type("Initial", (), dict(r=r, x=x, source_p=np.full(T, .095), source_q=np.full(T, .035)))()
        fitted = fit_fixed_inputs(candidate, obs, meter, fixed_r=r, fixed_x=x, initial=initial,
            seed=7+600+index, **config["profile"], **config["source"])
        assert fitted.success
        training.append(dict(candidate=candidate.name, r=r, x=x, source_p=np.full(3, .095), source_q=np.full(3, .035)))
        validation.append(dict(candidate=candidate.name, loss=float(index+1)))
        clean.append(dict(candidate=candidate.name, loss=fitted.loss))
    losses = {f["candidate"]:f["loss"]+validation[index]["loss"] for index,f in enumerate(clean)}
    M = min(v for name,v in losses.items() if name != "star")-losses["star"]
    row = dict(power_kw=u, status="complete", M=M, clean_future_M=M, operation={}, noiseless=dict(fits=clean))
    return dict(seed=7, policy_id="fixture", config=config, true_name="star", candidates=[dict(name=c.name, supports=c.supports) for c in bank],
        training=dict(p=[[.1, .1, .1]]), future_meter=meter_to_record(meter),
        training_result=dict(fits=training), validation_result=dict(fits=validation),
        truth=dict(r=rtrue, x=xtrue), forecast=dict(p=p*.95, q=q*.96, root=np.ones(T),
            source_by_candidate=dict(star=dict(source_p=source[:, 0]+.003, source_q=source[:, 1]-.002))),
        background=dict(p_kw=np.r_[p[:1], p[:1], p]*20, q_kvar=np.r_[q[:1], q[:1], q]*20,
            source_p_kw=np.r_[source[:1, 0], source[:1, 0], source[:, 0]]*20,
            source_q_kvar=np.r_[source[:1, 1], source[:1, 1], source[:, 1]]*20, root_v=np.ones(4)),
        methods=[dict(name=name, **deepcopy(row)) for name in diag.METHODS])


def test_small_complete_AC_endpoint_and_line_freezing():
    trial = _small_saved_trial()
    before = [(r["r"].copy(), r["x"].copy()) for r in trial["training_result"]["fits"]]
    out = diag.diagnose_trial(trial)
    assert out["success"] and out["unique_actions"] == 1
    action = next(iter(out["actions"].values()))
    assert action["endpoint_audit"]["success"]
    assert abs(action["endpoint_audit"]["difference"]) < 1e-7
    assert action["scenarios"]["111"]["own_future_SSE"] > 0.
    assert action["scenarios"]["000"]["own_future_SSE"] == 0.
    assert action["scenarios"]["111"]["response_error"]["increment"]["whitened_SSE"] == 0.
    assert out["source_profiles_completed"] == 28 and out["source_profiles_failed"] == 0
    for row, (r, x) in zip(trial["training_result"]["fits"], before):
        assert np.array_equal(row["r"], r) and np.array_equal(row["x"], x)


def test_same_curve_aliases_and_invalid_cache_values_retained():
    trial = _small_saved_trial()
    actions, missing = diag.select_saved_actions(trial)
    assert len(actions) == 1 and not missing
    assert len(next(iter(actions.values()))["aliases"]) == 5
    trial["methods"][1]["clean_future_M"] += 1.
    with pytest.raises(ValueError, match="different clean losses"):
        diag.select_saved_actions(trial)
