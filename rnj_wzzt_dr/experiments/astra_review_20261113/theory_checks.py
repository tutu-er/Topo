"""Deterministic checks of recovery-QCQP derivation, separate from AC evidence."""
import json

import numpy as np
from scipy.stats import chi2, ncx2, norm


def checks():
    count = 0
    # Independent direct CDF versus existence of the minimal auxiliary variable.
    # Nonzero margin also covers deterministic ties in the intended selection.
    margin = .01
    for alpha in (.2, .1, .05, .001):
        z = norm.isf(alpha)
        for mu in (-3., -.01, .01, .2, 1., 5., 20.):
            for variance in (0., .02, 1., 9.):
                success = float(mu >= margin) if variance == 0 else norm.cdf((mu-margin)/np.sqrt(variance))
                auxiliary = mu >= margin+z*np.sqrt(variance)
                assert auxiliary == bool(success >= 1-alpha)
                count += 1

    h = .6454865441781983
    branches = {}
    for probability in (.9, .95):
        z = norm.ppf(probability)
        r = np.sqrt(z*z-2*h)
        low, high = (z-r)**2/2, (z+r)**2/2
        assert abs(norm.cdf((h+low)/np.sqrt(2*low))-probability) < 1e-12
        assert abs(norm.cdf((h+high)/np.sqrt(2*high))-probability) < 1e-12
        assert norm.cdf((h+h)/np.sqrt(2*h)) < probability
        # A high-distance-only formulation wrongly excludes the small-J branch.
        assert low/2 < high and norm.cdf((h+low/2)/np.sqrt(low)) > probability
        branches[str(probability)] = dict(low=float(low), high=float(high))

    # Posterior score scaling must not change augmented MAP with consistent c0.
    for losses in ((1., 3., 6.), (5., 3., 2.), (7., 8., 9.)):
        for budget in (1.5, 4., 10.):
            expected = int(np.argmin(np.r_[budget, losses]))
            for score_variance in (.001, .2, 10.):
                selected = int(np.argmax(-np.r_[budget, losses]/(2*score_variance)))
                assert selected == expected
                count += 1

    # Independent isotropic benchmark for the noncentral Gaussian energy bound.
    # For tau²I, exact chi-square tails are available without Monte Carlo.
    worst_tail_ratio = 0.
    for m in (1, 4, 40, 1440):
        for noncentrality in (0., 1., 10., 100.):
            for alpha in (.2, .05, .02, .001):
                x = np.log(1/alpha)
                threshold = m+noncentrality+2*np.sqrt(x*(m+2*noncentrality))+2*x
                tail = ncx2.sf(threshold, m, noncentrality)
                assert tail <= alpha*(1+1e-10)
                worst_tail_ratio = max(worst_tail_ratio, float(tail/alpha))
                count += 1
    # Failure-budget obstruction exists even with exactly zero fitted bias.
    B = chi2.ppf(.9, 20)
    assert chi2.sf(B, 20) > .02
    assert ncx2.sf(B, 20, 1.) > chi2.sf(B, 20)
    return dict(status='passed', scalar_comparisons=count, pair_branches=branches,
                minimum_pair_probability=float(norm.cdf(np.sqrt(2*h))),
                worst_rejection_bound_tail_over_alpha=worst_tail_ratio,
                scope='Deterministic mathematical checks; zero AC days and no optimizer claims')


if __name__ == '__main__':
    print(json.dumps(checks(), indent=2))
