"""Independent AC/objective replay for the fixed-training-impedance study.

P/Q readings, after the externally fixed bias correction, are inputs rather
than fitted latent loads.  The conditional working objective contains only
terminal voltage magnitude and root active/reactive-power errors.  This module
does not invoke a fitter, the production AC predictor, or its old PQ likelihood.
"""
from __future__ import annotations

import numpy as np

from .ev_original_review import forward_physics

OBJECTIVE_CHANNELS = ("v", "master_p", "master_q")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, shape, name):
    try:
        result = np.broadcast_to(np.asarray(value, float), shape)
    except (ValueError, TypeError) as error:
        raise ValueError(name + " has wrong dimensions") from error
    _require(np.isfinite(result).all(), name + " is nonfinite")
    return result


def _close(actual, expected, name, *, atol=1e-6, rtol=5e-8):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    _require(actual.shape == expected.shape and np.isfinite(actual).all()
             and np.isfinite(expected).all()
             and np.allclose(actual, expected, atol=atol, rtol=rtol), name)


def review_fixed_fit(candidate, observations, model, row, *, trained_r=None, trained_x=None):
    """Replay one finite saved fit; raise ValueError on any inconsistency.

    ``row`` uses ``gradient_price_study.fit_record`` fields.  ``trained_r/x``
    must both be supplied for validation/future states: byte-restored float
    arrays must agree exactly with the training parameter arrays, rather than
    merely being close.  Omitting both permits training-state objective replay.

    The reported log likelihood is the working conditional Gaussian score
    ``-.5 * (SSE + sum(log(2*pi*sigma**2)))`` over the three objective channels.
    No P/Q or independent-source-reading density is included.  No optimality
    certificate follows from replaying a saved finite state.
    """
    n, t, m = candidate.n, observations.t, candidate.m
    _require(observations.n == n, "observation/candidate terminal count")
    _require(not model.fit_bias and model.calibration is None, "bias must be externally fixed; no joint calibration likelihood")
    _require(all(getattr(observations, k) is None and getattr(model, k + "_sigma") is None
                 for k in ("source_p", "source_q")), "separate source telemetry is outside this model")
    _require(not row.get("global_MLE_certified", False), "local fit labelled global MLE")
    _require(row.get("name", row.get("candidate")) == candidate.name, "candidate name")
    _require(tuple(map(tuple, row["supports"])) == candidate.supports, "saved support/edge order")
    _require(candidate.parents[0] < 0 and candidate.supports[0] == tuple(range(n)),
             "source receiving node must follow the full-support root edge")
    diagnostics = row.get("diagnostics", {})
    for saved in (row, diagnostics):
        for key in ("source_interface", "source_node"):
            if key in saved:
                _require(saved[key] == 1, "physical source interface must be 1")
        for key in ("objective_channels", "measurement_channels"):
            if key in saved:
                _require(tuple(saved[key]) == OBJECTIVE_CHANNELS, "objective channel declaration")

    r, x = (_array(row[k], (m,), k) for k in ("r", "x"))
    _require(np.all(r >= 0) and np.all(x >= 0), "nonnegative impedance required")
    _require((trained_r is None) == (trained_x is None), "supply both trained R and X")
    fixed_impedance = trained_r is not None
    if fixed_impedance:
        for key, value, trained in (("R", r, trained_r), ("X", x, trained_x)):
            target = _array(trained, (m,), "trained " + key)
            _require(np.array_equal(value, target), "trained " + key + " changed")

    fixed_load = {}
    for channel in ("p", "q", "v"):
        bias = _array(getattr(model, "bias_" + channel), (t, n), "fixed " + channel + " bias")
        saved_bias = _array(row["bias_" + channel], (t, n), "saved " + channel + " bias")
        _require(np.array_equal(saved_bias, bias), "saved " + channel + " bias changed")
        if channel in ("p", "q"):
            expected = np.asarray(getattr(observations, channel), float) - bias
            value = _array(row[channel], (t, n), channel)
            _require(np.array_equal(value, expected), channel.upper() + " is not the fixed corrected reading")
            fixed_load[channel] = value
    sp, sq = (_array(row[k], (t,), k) for k in ("source_p", "source_q"))
    root = _array(observations.root_v, (t,), "known root voltage")
    _require(np.all(root > 0), "positive root voltage")
    physical = forward_physics(candidate, r, x, fixed_load["p"], fixed_load["q"], sp, sq, root)

    channel_sse, sigmas, residuals, prediction_errors = {}, {}, {}, {}
    normalizer = 0.
    for channel in OBJECTIVE_CHANNELS:
        observed = getattr(observations, channel)
        _require(observed is not None, "missing objective channel: " + channel)
        observed = np.asarray(observed, float)
        sigma_value = getattr(model, channel + "_sigma")
        _require(sigma_value is not None, "missing sigma: " + channel)
        sigma = _array(sigma_value, observed.shape, channel + " sigma")
        _require(np.all(sigma > 0), "positive sigma required: " + channel)
        prediction = physical[channel]
        if channel == "v":
            prediction = prediction + _array(model.bias_v, (t, n), "fixed v bias")
        white = (prediction - observed) / sigma
        channel_sse[channel] = float(np.sum(white**2))
        sigmas[channel] = sigma.tolist()
        residuals[channel] = white.tolist()
        normalizer += float(np.sum(np.log(2*np.pi*sigma**2)))

    saved_predictions = row.get("predicted", {})
    for channel in ("p", "q", *OBJECTIVE_CHANNELS):
        _require(channel in saved_predictions, "missing saved prediction: " + channel)
        expected = physical[channel]
        if channel in ("p", "q", "v"):
            expected = expected + _array(getattr(model, "bias_" + channel), (t, n), channel + " bias")
        value = _array(saved_predictions[channel], expected.shape, "predicted " + channel)
        error = float(np.max(abs(value - expected)))
        prediction_errors[channel] = error
        _require(error <= 2e-10, "saved prediction disagrees with branch replay: " + channel)

    cost = sum(channel_sse.values())
    loglik = -.5*(cost + normalizer)
    _close(row["loss"], cost, "fixed-input SSE mismatch")
    _close(row["loglik"], loglik, "conditional Gaussian loglik mismatch")
    if "half_loss" in row:
        _close(row["half_loss"], .5*cost, "half-SSE mismatch")
    return dict(success=True, candidate=candidate.name, declared_fit_success=bool(row.get("success", False)),
                trained_impedance_checked=fixed_impedance, P_Q_fixed_exactly=True, source_interface=1,
                objective_channels=list(OBJECTIVE_CHANNELS), full_SSE=cost, half_SSE=.5*cost,
                loglik=loglik, Gaussian_normalizer=normalizer, channel_SSE=channel_sse,
                sigma=sigmas, whitened_residual=residuals, prediction_max_abs_errors=prediction_errors,
                AC_equation_residual=physical["equation_residual"],
                AC_balance_residual=physical["balance_residual"],
                voltage_min=physical["voltage_min"], voltage_max=physical["voltage_max"],
                iterations=physical["iterations"],
                implementation="independent_backward_branch_currents_forward_voltage_drops",
                loglik_scope="conditional working Gaussian score given corrected P/Q readings; only v/root P/root Q",
                no_new_fit=True, no_production_AC_predictor=True, global_MLE_certified=False)
