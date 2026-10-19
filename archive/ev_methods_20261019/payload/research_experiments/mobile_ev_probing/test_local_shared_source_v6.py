"""Independent contrast/witness checks for the small declared linear experiment.

Only eight small LP solves are used. Expected constants come from observable
Y2-Y3 slopes and explicit feasible points, not from a second optimizer.
"""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from . import local_shared_source_v6 as study


_AUDIT = []
_BOX = ((.01, .05), (.005, .03), (.01, .12), (.01, .12), (.01, .12))
_PAIR = {"cherry12": (0, 1), "cherry13": (0, 2), "cherry23": (1, 2)}


def _predict(p, pair, theta, nominal, w):
    """Direct rational physical prediction; does not call the LP builder."""
    a, b, *c = theta
    return [a * (sum(p) + nominal) + (b * sum(p[j] for j in pair) if i in pair else 0)
            + c[i] * p[i] + (w[i] if isinstance(w, tuple) else w) for i in range(3)]


def _independent_observations(policy, *, epsilon=1e-14, noise=False,
                              correlated_source=False, branch=F(".01")):
    ev = np.full(24, .15) if policy == "constant" else np.tile([0., .3, .3, 0.], 6)
    u = np.r_[np.zeros(12), ev]
    p = np.column_stack((.2 + u, np.full(36, .2), np.full(36, .2)))
    delta = (.02 * np.where(u > 0., 1., -1.) if correlated_source
             else .02 * np.sin(np.arange(36)))
    nominal = np.full(36, .12)
    theta = (F(".02"), branch, F(".06"), F(".06"), F(".06"))
    exact = [_predict(tuple(map(F, pt)), (0, 1), theta, F(st), theta[0] * F(dt))
             for pt, st, dt in zip(p, nominal, delta)]
    errors = np.zeros((36, 3))
    if noise:
        errors = .8 * epsilon * np.resize(np.array([-1., .5, 1., -.5, 0.]), (36, 3))
    y = np.array([[float(value + F(errors[t, i])) for i, value in enumerate(row)]
                  for t, row in enumerate(exact)])
    actual_error = [[F(y[t, i]) - exact[t][i] for i in range(3)] for t in range(36)]
    assert max(abs(value) for row in actual_error for value in row) <= F(epsilon)
    return dict(P_total=p, Y_drop=y, source_nominal=nominal,
                source_error_bound=np.full(36, .1), drop_error_bound=np.full((36, 3), epsilon),
                batch_id=np.r_[np.zeros(12, int), np.ones(24, int)]), theta, delta, exact


def _solve(obs, candidate, **kwargs):
    result = study.solve_lp(obs, candidate, **kwargs)
    _AUDIT.append(dict(candidate=candidate, arguments=kwargs, result=result))
    assert result["solver_success"]
    assert result["exact_primal_verified"] and result["exact_box_corrected_dual_verified"]
    assert result["reliable_lower_bound"] <= result["verified_upper_bound"]
    assert result["exact_gap"] < 1e-11
    return result


@pytest.fixture(scope="session", autouse=True)
def _save_independent_lp_evidence():
    yield
    source = Path(study.__file__)
    output = source.with_name("results_v6") / "independent_lp_checks.json"
    output.write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="eight small LPs plus independent exact-rational contrast/feasible-point checks",
        n_lp=len(_AUDIT), records=_AUDIT), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_exact_contrast_lower_bounds_are_attained_and_box_erases_them():
    """Check the analytic proof using rational arithmetic and explicit points."""
    true_theta = tuple(map(F, (".02", ".01", ".06", ".06", ".06")))
    for length in (F(".15"), F(".3")):
        witnesses = {
            "cherry13": ((F(".0225"), F(".005"), F(".0625"),
                          F(".0725") + F(".01875") * length,
                          F(".0425") - F(".01875") * length), F("-.0003"), F(".0075")),
            "cherry23": ((F(".025"), F(".005"), F(".065"),
                          F(".055") + F(".0125") * length,
                          F(".035") - F(".0125") * length), F("-.0006"), F(".005")),
        }
        for name, (theta, offset, slope) in witnesses.items():
            # This symbolic proposition uses the declared decimal domain.
            # Stored binary-float LP endpoints are certified separately below.
            assert all(F(str(lo)) <= value <= F(str(hi)) for value, (lo, hi) in zip(theta, _BOX))
            for u in (F(0), length):
                p = (F(".2") + u, F(".2"), F(".2"))
                for delta in (F("-.02"), F(0), F(".02")):
                    truth = _predict(p, (0, 1), true_theta, F(".12"), F(".02") * delta)
                    w = offset + F(".02") * delta
                    predicted = _predict(p, _PAIR[name], theta, F(".12"), w)
                    residual = tuple(t - v for t, v in zip(truth, predicted))
                    assert residual == (0, slope * (u - length / 2), -slope * (u - length / 2))
                    assert max(map(abs, residual)) == slope * length / 2
                    assert abs(w) <= theta[0] * F(".1")
                    box_w = tuple(w + value for value in residual)
                    assert max(map(abs, box_w)) <= theta[0] * F(".1")
                    assert _predict(p, _PAIR[name], theta, F(".12"), box_w) == truth
                    # Coarse measurement uncertainty alone also makes these
                    # wrong shared-source candidates compatible (no extra tau).
                    assert max(map(abs, residual)) < F(".002")


def test_four_lp_values_match_independent_analytic_constants():
    for policy, length in (("constant", .15), ("abba", .3)):
        observed, *_ = _independent_observations(policy)
        for candidate, slope_gap in (("cherry13", .015), ("cherry23", .01)):
            result = _solve(observed, candidate)
            expected = slope_gap * length / 4
            assert result["status"] == "excluded"
            assert abs(result["reliable_lower_bound"] - expected) < 1e-12
            assert abs(result["verified_upper_bound"] - expected) < 1e-12


def test_same_observations_with_parameter_reset_lose_constant_probe_certificate():
    observed, *_ = _independent_observations("constant")
    result = _solve(observed, "cherry13", parameter_sharing="reset")
    assert result["status"] == "unresolved"
    assert result["reliable_lower_bound"] == 0
    assert result["verified_upper_bound"] < 1e-12
    assert len(result["theta_by_batch"]) == 2


def test_coordinate_source_box_retains_wrong_tree_and_true_model_is_feasible():
    observed, theta, delta, _ = _independent_observations("abba", epsilon=1e-5, noise=True)
    result = _solve(observed, "cherry13", source_model="box")
    assert result["status"] == "unresolved"
    assert result["reliable_lower_bound"] == 0
    assert result["verified_upper_bound"] < 1e-12
    model = study.build_lp(observed, "cherry12")
    point = list(theta) + [theta[0] * F(d) for d in delta] + [F(0)]
    assert all(F(l) <= v <= F(u) for v, l, u in zip(point, model.lower, model.upper))
    assert all(sum(a * x for a, x in zip(row, point)) <= rhs
               for row, rhs in zip(model.exact_rows, model.exact_right_sides))


def test_correlated_source_and_bounded_adversarial_meter_error_preserve_separation():
    epsilon = 1e-5
    observed, _, _, exact = _independent_observations(
        "abba", epsilon=epsilon, noise=True, correlated_source=True)
    for p, y in zip(observed["P_total"], exact):
        assert y[1] - y[2] == F(".01") * (F(p[0]) + F(p[1]))
    result = _solve(observed, "cherry13")
    d = .015 * .3 / 4
    assert result["status"] == "excluded"
    assert result["reliable_lower_bound"] >= d - 2 * epsilon - 1e-12
    assert result["verified_upper_bound"] <= d + 1e-12


def test_nonzero_but_weak_branch_can_remain_unresolved():
    observed, *_ = _independent_observations("abba", epsilon=1e-5, branch=F(".000001"))
    box = (_BOX[0], (1e-6, .03), *_BOX[2:])
    result = _solve(observed, "cherry13", parameter_box=box)
    assert result["status"] == "unresolved"
    assert result["reliable_lower_bound"] == 0
    assert result["verified_upper_bound"] < 1e-12
    # At the zero-edge boundary the three labelled candidate matrices are
    # identical for every input, not only for the selected probe waveform.
    theta = (F(".02"), F(0), F(".06"), F(".06"), F(".06"))
    p = (F(".51"), F(".18"), F(".29"))
    predictions = [_predict(p, pair, theta, F(".12"), F(".0002")) for pair in _PAIR.values()]
    assert predictions[0] == predictions[1] == predictions[2]
