"""Independent fixed-input replay and strict parameter/data tamper checks."""
from copy import deepcopy

import numpy as np
import pytest

from . import ev_fixed_train_review as review
from .likelihood import MeasurementModel, Observations, RadialCandidate


def fixture():
    candidate = RadialCandidate.from_supports("cherry", 3, ((0, 1),))
    r = np.full(candidate.m, .006)
    x = np.full(candidate.m, .003)
    meter = MeasurementModel(.02, .03, .001, .005, .007,
                             bias_p=[.001, -.002, .003], bias_q=-.001, bias_v=.0002)
    # These are noisy P/Q readings, not the original latent physical loads.
    p_read = np.array([[.121, .083, .109], [.108, .141, .094]])
    q_read = np.array([[.041, .023, .034], [.025, .038, .041]])
    p = p_read - np.asarray(meter.bias_p)
    q = q_read - np.asarray(meter.bias_q)
    sp, sq = np.array([.03, -.01]), np.array([.012, .008])
    phy = review.forward_physics(candidate, r, x, p, q, sp, sq, np.ones(2))
    means = {k: phy[k] + (getattr(meter, "bias_" + k) if k in ("p", "q", "v") else 0.)
             for k in ("p", "q", "v", "master_p", "master_q")}
    voltage_noise = np.array([[.0003, -.0001, .0002], [-.0002, .0001, .0004]])
    observations = Observations(p_read, q_read, means["v"] + voltage_noise, root_v=np.ones(2),
                                master_p=means["master_p"] + [.001, -.002],
                                master_q=means["master_q"] + [-.003, .001])
    channel_sse, normalizer = {}, 0.
    for channel in review.OBJECTIVE_CHANNELS:
        observed = getattr(observations, channel)
        sigma = np.broadcast_to(getattr(meter, channel + "_sigma"), observed.shape)
        channel_sse[channel] = float(np.sum(((means[channel] - observed)/sigma)**2))
        normalizer += float(np.sum(np.log(2*np.pi*sigma**2)))
    loss = sum(channel_sse.values())
    row = dict(name=candidate.name, supports=candidate.supports, r=r, x=x, p=p, q=q,
               source_p=sp, source_q=sq, bias_p=meter.bias_p, bias_q=meter.bias_q,
               bias_v=meter.bias_v, predicted=means, loss=loss, half_loss=.5*loss,
               loglik=-.5*(loss+normalizer), success=True,
               diagnostics=dict(source_interface=1, objective_channels=list(review.OBJECTIVE_CHANNELS)),
               global_MLE_certified=False)
    return candidate, observations, meter, row


def test_noisy_PQ_are_exactly_fixed_and_old_likelihood_is_not_called(monkeypatch):
    candidate, observations, meter, row = fixture()
    from . import likelihood
    def forbidden(*args, **kwargs):
        raise AssertionError("production model/fitter called by replay")
    monkeypatch.setattr(likelihood, "ac_predict", forbidden)
    monkeypatch.setattr(likelihood, "fit_candidate", forbidden)
    result = review.review_fixed_fit(candidate, observations, meter, row,
                                     trained_r=row["r"], trained_x=row["x"])
    assert result["success"] and result["P_Q_fixed_exactly"]
    assert result["trained_impedance_checked"]
    assert set(result["channel_SSE"]) == {"v", "master_p", "master_q"}
    assert result["full_SSE"] == pytest.approx(row["loss"], abs=1e-12)
    # P/Q sigma is absent from this conditional working objective.
    other_meter = MeasurementModel(100., 200., meter.v_sigma, meter.master_p_sigma, meter.master_q_sigma,
                                   bias_p=meter.bias_p, bias_q=meter.bias_q, bias_v=meter.bias_v)
    other = review.review_fixed_fit(candidate, observations, other_meter, row)
    assert other["full_SSE"] == result["full_SSE"]
    assert other["loglik"] == result["loglik"]


@pytest.mark.parametrize("channel", ["p", "q"])
def test_adjusting_reading_to_improve_other_channels_is_rejected(channel):
    candidate, observations, meter, row = fixture()
    row[channel] = row[channel].copy()
    row[channel][0, 0] += 1e-14
    with pytest.raises(ValueError, match="fixed corrected reading"):
        review.review_fixed_fit(candidate, observations, meter, row)


@pytest.mark.parametrize("channel", ["r", "x"])
def test_validation_impedance_matches_training_exactly(channel):
    candidate, observations, meter, row = fixture()
    trained_r, trained_x = row["r"].copy(), row["x"].copy()
    row[channel] = row[channel].copy()
    row[channel][0] += 1e-15
    with pytest.raises(ValueError, match="trained .* changed"):
        review.review_fixed_fit(candidate, observations, meter, row,
                                 trained_r=trained_r, trained_x=trained_x)


@pytest.mark.parametrize("field", ["loss", "loglik", "half_loss"])
def test_objective_tampering_is_rejected(field):
    candidate, observations, meter, row = fixture()
    row[field] += 1.
    with pytest.raises(ValueError, match="SSE|loglik"):
        review.review_fixed_fit(candidate, observations, meter, row)


@pytest.mark.parametrize("damage", ["bias", "source_interface", "objective", "predicted"])
def test_model_or_prediction_tampering_is_rejected(damage):
    candidate, observations, meter, row = fixture()
    row = deepcopy(row)
    if damage == "bias":
        row["bias_p"] = [.002, -.002, .003]
    elif damage == "source_interface":
        row["diagnostics"]["source_interface"] = 2
    elif damage == "objective":
        row["diagnostics"]["objective_channels"].append("p")
    else:
        row["predicted"]["v"][0, 0] += 1e-5
    with pytest.raises(ValueError):
        review.review_fixed_fit(candidate, observations, meter, row)


def test_actual_fixed_input_fitter_state_passes_independent_replay():
    from .ev_fixed_train_fit import fit_fixed_inputs
    from .gradient_price_study import fit_record
    candidate, observations, meter, row = fixture()
    fit = fit_fixed_inputs(candidate, observations, meter,
                           fixed_r=row["r"], fixed_x=row["x"], multistart=1, max_nfev=100)
    assert fit.success
    saved = fit_record(candidate, fit)
    result = review.review_fixed_fit(candidate, observations, meter, saved,
                                     trained_r=row["r"], trained_x=row["x"])
    assert result["success"]
    assert result["full_SSE"] == pytest.approx(fit.loss, abs=1e-9)
    assert fit.diagnostics["joint_parameter_count"] == 2*observations.t
