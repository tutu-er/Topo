"""Statistical helpers checked with finite enumeration and analytic cases."""

from math import comb, exp

import numpy as np
import pytest

from rnj_wzzt.estimation.ac_validation import (
    binomial_exceedance_test,
    chi_square_wls_diagnostic,
    empirical_upper_tail_pvalue,
    one_sided_rate_lower_bound,
)


@pytest.mark.parametrize("scores", [[0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 2.0, 4.0, 4.0]])
def test_rank_pvalues_are_superuniform_under_exchangeable_index_enumeration(scores):
    # Conditional on the unordered scores, each held-out index is equally likely.
    values = np.asarray(scores)
    pvalues = np.array([
        empirical_upper_tail_pvalue(value, np.delete(values, i))
        for i, value in enumerate(values)
    ])
    for alpha in (0.0, 0.19, 0.2, 0.4, 0.75, 1.0):
        assert np.mean(pvalues <= alpha) <= alpha + 1e-14


def test_rank_resolution_ties_and_input_immutability():
    null_scores = np.array([1.0, 2.0, 2.0, 4.0])
    original = null_scores.copy()
    assert empirical_upper_tail_pvalue(5.0, null_scores) == pytest.approx(0.2)
    assert empirical_upper_tail_pvalue(2.0, null_scores) == pytest.approx(0.8)
    assert empirical_upper_tail_pvalue(0.0, null_scores) == 1.0
    np.testing.assert_array_equal(null_scores, original)


@pytest.mark.parametrize("score,null_scores", [
    (-1.0, [1.0]), (np.nan, [1.0]), (np.inf, [1.0]), (True, [1.0]),
    (10**1000, [1.0]),
    (1.0, []), (1.0, [[1.0]]), (1.0, [np.inf]), (1.0, [np.nan]),
    (1.0, [-1.0]), (1.0, [True]), (1.0, [1.0 + 1j]), (1.0, ["1"]),
])
def test_rank_rejects_invalid_scores_without_dropping_samples(score, null_scores):
    with pytest.raises(ValueError):
        empirical_upper_tail_pvalue(score, null_scores)


def test_binomial_tails_match_independent_finite_sums_and_boundaries():
    expected = sum(comb(10, k) for k in range(8, 11)) / 2**10
    assert binomial_exceedance_test(8, 10, 0.5) == pytest.approx(expected)
    assert binomial_exceedance_test(2, 10, 0.5, alternative="less") == pytest.approx(expected)
    assert binomial_exceedance_test(1, 10, 0.5, alternative="two-sided") == pytest.approx(22 / 1024)
    assert binomial_exceedance_test(0, 10, 0.0) == 1.0
    assert binomial_exceedance_test(1, 10, 0.0) == 0.0
    assert binomial_exceedance_test(10, 10, 1.0) == 1.0
    assert binomial_exceedance_test(9, 10, 1.0, alternative="less") == 0.0


def test_one_sided_binomial_bound_has_finite_sample_coverage():
    # Exhaust the sample space for each Bernoulli rate, independently of beta.ppf.
    trials = 7
    confidence = 0.9
    bounds = [one_sided_rate_lower_bound(k, trials, confidence_level=confidence)
              for k in range(trials + 1)]
    assert bounds == sorted(bounds)
    for rate in np.linspace(0.0, 1.0, 101):
        coverage = sum(
            comb(trials, k) * rate**k * (1 - rate)**(trials - k)
            for k in range(trials + 1) if bounds[k] <= rate + 1e-14
        )
        assert coverage >= confidence - 1e-12


def test_all_success_lower_bound_has_analytic_sample_size_interpretation():
    assert one_sided_rate_lower_bound(0, 59) == 0.0
    assert one_sided_rate_lower_bound(59, 59) == pytest.approx(0.05 ** (1 / 59))
    assert one_sided_rate_lower_bound(58, 58) < 0.95
    assert one_sided_rate_lower_bound(59, 59) > 0.95
    assert one_sided_rate_lower_bound(299, 299) > 0.99


@pytest.mark.parametrize("successes,trials", [(0, 0), (-1, 2), (3, 2), (1.0, 2), (1, 2.0), (True, 2), (1, False)])
def test_count_helpers_reject_invalid_counts(successes, trials):
    with pytest.raises(ValueError):
        binomial_exceedance_test(successes, trials, 0.05)
    with pytest.raises(ValueError):
        one_sided_rate_lower_bound(successes, trials)


@pytest.mark.parametrize("rate", [-0.1, 1.1, np.nan, np.inf, True])
def test_binomial_test_rejects_invalid_null_rates(rate):
    with pytest.raises(ValueError):
        binomial_exceedance_test(1, 3, rate)


def test_binomial_test_rejects_unknown_alternative():
    with pytest.raises(ValueError):
        binomial_exceedance_test(1, 3, 0.5, alternative="unexpected")


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, np.nan, True])
def test_rate_bound_rejects_invalid_confidence_levels(confidence):
    with pytest.raises(ValueError):
        one_sided_rate_lower_bound(1, 3, confidence_level=confidence)


def test_chi_square_diagnostic_matches_exact_df2_formula_without_claiming_acceptance():
    # chi-square_2 survival is exp(-L/2); L is twice least_squares.cost.
    diagnostic = chi_square_wls_diagnostic(6.0, 8, 6)
    assert diagnostic.dof == 2
    assert diagnostic.reduced_loss == 3.0
    assert diagnostic.approx_pvalue == pytest.approx(exp(-3.0))
    assert not hasattr(diagnostic, "reject")
    assert chi_square_wls_diagnostic(0.0, 8, 6).approx_pvalue == 1.0


def test_saturated_fit_has_no_residual_test():
    diagnostic = chi_square_wls_diagnostic(0.0, 8, 8)
    assert diagnostic.dof == 0
    assert diagnostic.approx_pvalue is None
    assert diagnostic.reduced_loss is None


@pytest.mark.parametrize("loss,count,rank", [
    (-1.0, 8, 6), (np.nan, 8, 6), (np.inf, 8, 6), (True, 8, 6),
    (1.0, 0, 0), (1.0, 8, 9), (1.0, 8, -1), (1.0, 8.0, 6),
    (1.0, 8, 6.0), (1.0, True, 0),
])
def test_chi_square_diagnostic_rejects_invalid_inputs(loss, count, rank):
    with pytest.raises(ValueError):
        chi_square_wls_diagnostic(loss, count, rank)
