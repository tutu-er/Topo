"""Independent low-dimensional checks for bounded line Gaussian integration."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr, log_ndtr
from scipy.stats import truncnorm

from .ev_bayes_response_truncation import (
    line_prior, line_prior_mass, bounded_line_moments, contains_eta,
    eta_to_rx, eta_jacobian, _log_interval, _truncated_quantile,
)


def domain(r=(-1., 1.), ratio=(-1., 1.), x=(-3., 3.)):
    return dict(edges=1, log_R=np.asarray([r]), log_ratio=np.asarray([ratio]), log_X=np.asarray([x]))


def reference_2d(mean, cov, bounds):
    """Ordinary quadrature with analytic conditional b moments, independent of GHK."""
    lr, lb, lx = (np.asarray(bounds[name])[0] for name in ('log_R', 'log_ratio', 'log_X'))
    lo, hi = max(lr[0], lx[0]-lb[1]), min(lr[1], lx[1]-lb[0])
    sa = np.sqrt(cov[0, 0])
    sb = np.sqrt(cov[1, 1]-cov[0, 1]**2/cov[0, 0])
    def integrand(a, p, q):
        mb = mean[1]+cov[1, 0]/cov[0, 0]*(a-mean[0])
        aa, bb = (max(lb[0], lx[0]-a)-mb)/sb, (min(lb[1], lx[1]-a)-mb)/sb
        pa, pb = np.exp(-aa*aa/2)/np.sqrt(2*np.pi), np.exp(-bb*bb/2)/np.sqrt(2*np.pi)
        z = ndtr(bb)-ndtr(aa)
        if q == 1:
            z = mb*z+sb*(pa-pb)
        elif q == 2:
            z = (mb*mb+sb*sb)*z+2*mb*sb*(pa-pb)+sb*sb*(aa*pa-bb*pb)
        return a**p*z*np.exp(-.5*((a-mean[0])/sa)**2)/(np.sqrt(2*np.pi)*sa)
    def integ(p, q):
        points = [v for v in (lx[0]-lb[0], lx[0]-lb[1], lx[1]-lb[0], lx[1]-lb[1]) if lo < v < hi]
        return quad(lambda a: integrand(a, p, q), lo, hi, points=sorted(set(points)), epsabs=1e-12, epsrel=1e-11)[0]
    mass = integ(0, 0)
    mu = np.asarray([integ(1, 0), integ(0, 1)])/mass
    raw = np.asarray([[integ(2, 0), integ(1, 1)], [integ(1, 1), integ(0, 2)]])/mass
    return np.log(mass), mu, raw-np.outer(mu, mu)


def test_public_prior_is_untruncated_log_interval_prior_and_original_domain():
    r, x, ratio = [.03, .4], [.01, .3], [.2, 2.]
    p = line_prior(2, r, x, ratio)
    assert p['mean'] == pytest.approx(np.tile([np.log(r).mean(), np.log(ratio).mean()], 2))
    assert np.diag(p['cov']) == pytest.approx(np.tile([np.diff(np.log(r))[0]**2/12, np.diff(np.log(ratio))[0]**2/12], 2))
    expected = reference_2d(p['mean'][:2], p['cov'][:2, :2], {k:(v[:1] if isinstance(v, np.ndarray) else 1 if k == 'edges' else v) for k, v in p['bounds'].items()})[0]
    assert p['log_mass'] == pytest.approx(2*expected, abs=2e-10)
    assert p['normalization']['edge_relative_quad_error'].max() < 1e-8
    result = bounded_line_moments(p['mean'], p['cov'], p['bounds'], seed=121, return_samples=True)
    assert result['status'] == 'ok'
    assert contains_eta(result['samples'], p['bounds'], tolerance=1e-14).all()
    assert result['log_mass'] == pytest.approx(p['log_mass'], abs=4e-4)


def test_one_dimensional_moments_and_mass_with_deterministic_other_coordinate():
    spec = domain(r=(-1., 2.), x=(-5., 5.))
    result = bounded_line_moments([0., 0.], np.diag([1., 0.]), spec, seed=33)
    mu, var = truncnorm.stats(-1., 2., moments='mv')
    assert result['status'] == 'ok'
    assert result['log_mass'] == pytest.approx(np.log(ndtr(2.)-ndtr(-1.)), abs=1e-13)
    assert result['mean'] == pytest.approx([mu, 0.], abs=3e-5)
    assert result['cov'] == pytest.approx(np.diag([var, 0.]), abs=5e-5)
    assert result['diagnostics']['factor']['rank'] == 1


@pytest.mark.parametrize('correlation', [0., .7, -.85])
def test_correlated_two_dimensional_quadrature(correlation):
    spec = domain(r=(-1., 1.2), ratio=(-.8, 1.), x=(-.7, .9))
    mu = np.array([-.2, .15])
    cross = correlation*np.sqrt(.8*.6)
    cov = np.array([[.8, cross], [cross, .6]])
    z, mean, variance = reference_2d(mu, cov, spec)
    result = bounded_line_moments(mu, cov, spec, seed=44, power=11, max_power=13, return_samples=True)
    assert result['status'] == 'ok'
    assert result['log_mass'] == pytest.approx(z, abs=2e-4)
    assert result['mean'] == pytest.approx(mean, abs=3e-4)
    assert result['cov'] == pytest.approx(variance, abs=4e-4)
    assert result['weights'].sum() == pytest.approx(1.)
    assert np.einsum('n,ni->i', result['weights'], result['samples']) == pytest.approx(result['mean'], abs=3e-14)
    assert contains_eta(result['samples'][result['weights'] > 0.], spec, tolerance=1e-14).all()


def test_rank_one_active_sum_boundary_reduces_to_scalar_truncation():
    # b=a exactly, and the original X interval truncates 2a to [-.3,.5].
    spec = domain(x=(-.3, .5))
    result = bounded_line_moments([0., 0.], np.ones((2, 2)), spec, seed=52,
                                  power=12, max_power=14, return_samples=True)
    mu, var = truncnorm.stats(-.15, .25, moments='mv')
    assert result['status'] == 'ok'
    assert result['diagnostics']['factor']['rank'] == 1
    assert result['log_mass'] == pytest.approx(np.log(ndtr(.25)-ndtr(-.15)), abs=7e-4)
    assert result['mean'] == pytest.approx([mu, mu], abs=3e-4)
    assert result['cov'] == pytest.approx(np.full((2, 2), var), abs=2e-4)
    assert np.array_equal(result['samples'][:, 0], result['samples'][:, 1])


def test_far_tail_mass_stays_in_log_domain_and_moments_finite():
    spec = domain(r=(40., 41.), x=(39., 42.))
    result = bounded_line_moments([0., 0.], np.diag([1., 0.]), spec, seed=65, power=12, max_power=14)
    z = quad(lambda t: np.exp(-40*t-t*t/2), 0., 1., epsabs=1e-14)[0]
    first = quad(lambda t: t*np.exp(-40*t-t*t/2), 0., 1., epsabs=1e-14)[0]/z
    second = quad(lambda t: t*t*np.exp(-40*t-t*t/2), 0., 1., epsabs=1e-14)[0]/z
    assert result['status'] == 'ok'
    assert result['mass'] == 0.  # exp(log Z) underflows; log Z is primary.
    assert np.isfinite(result['log_mass'])
    assert result['log_mass'] == pytest.approx(log_ndtr(-40.), abs=2e-12)
    assert result['mean'][0] == pytest.approx(40+first, abs=2e-5)
    assert result['cov'][0, 0] == pytest.approx(second-first*first, abs=4e-6)


def test_narrow_interval_probability_and_quantile_do_not_cancel():
    aa, bb = np.array([1e-20, 40.]), np.array([2e-20, 40.+1e-10])
    mass = _log_interval(aa, bb)
    center = aa+(bb-aa)/2
    expected = np.log(bb-aa)-.5*center**2-.5*np.log(2*np.pi)
    assert mass == pytest.approx(expected, abs=1e-12)
    quantiles = _truncated_quantile(np.array([.25, .75]), aa, bb)
    assert np.isfinite(quantiles).all()
    assert quantiles[0] == pytest.approx(1.25e-20, abs=1e-34)
    assert np.all(quantiles >= aa) and np.all(quantiles <= bb)


def test_independent_scrambles_reproducibility_and_reported_precision():
    spec = domain(x=(-.5, .8))
    args = ([.1, -.3], [[1., .6], [.6, 1.2]], spec)
    a = bounded_line_moments(*args, seed=141, return_samples=True)
    b = bounded_line_moments(*args, seed=141, return_samples=True)
    c = bounded_line_moments(*args, seed=142, return_samples=True)
    for name in ('samples', 'weights', 'mean', 'cov'):
        assert np.array_equal(a[name], b[name])
    assert a['log_mass'] == b['log_mass']
    assert not np.array_equal(a['samples'], c['samples'])
    d = a['diagnostics']
    assert len(set(d['replicate_seeds'])) == 8
    assert d['relative_se'] < .01 and d['ess_fraction'] > .02
    assert d['mean_se'].shape == (2,) and d['covariance_se'].shape == (2, 2)
    assert a['status'] == 'ok'


def test_adaptive_precision_failure_is_explicit_and_estimate_retained():
    result = bounded_line_moments([0., .1], [[1., .7], [.7, 1.]], domain(x=(-.4, .6)),
        seed=8, power=4, max_power=6, replicates=4, relative_se_tolerance=0.,
        mean_se_tolerance=0., covariance_se_tolerance=0., min_ess=0.)
    assert result['status'] == 'precision_not_met'
    assert np.isfinite(result['log_mass']) and np.isfinite(result['mean']).all()
    assert [h['power'] for h in result['diagnostics']['history']] == [4, 5, 6]
    assert result['diagnostics']['points'] == 4*2**6
    assert result['samples'] is None and result['weights'] is None


def test_deterministic_allowed_and_impossible_support_are_distinguished():
    good = bounded_line_moments([.1, .2], np.zeros((2, 2)), domain(), min_ess=0.)
    bad = bounded_line_moments([2., .2], np.zeros((2, 2)), domain(), min_ess=0.)
    assert good['status'] == 'ok' and good['log_mass'] == 0.
    assert good['mean'] == pytest.approx([.1, .2], abs=1e-14)
    assert np.max(np.abs(good['cov'])) < 1e-26
    assert bad['status'] == 'zero_mass' and bad['log_mass'] == -np.inf
    assert np.isnan(bad['mean']).all()


def test_eta_physical_maps_and_both_jacobian_row_orders():
    eta = np.array([np.log(.2), np.log(.5), np.log(.4), np.log(1.5)])
    r, x = eta_to_rx(eta)
    assert r == pytest.approx([.2, .4]) and x == pytest.approx([.1, .6])
    for order in ('interleaved', 'stacked'):
        J = eta_jacobian(r, x, ordering=order)
        def physical(v):
            rr, xx = eta_to_rx(v)
            return np.column_stack((rr, xx)).ravel() if order == 'interleaved' else np.r_[rr, xx]
        finite = np.column_stack([(physical(eta+1e-6*np.eye(4)[j])-physical(eta-1e-6*np.eye(4)[j]))/2e-6 for j in range(4)])
        assert J == pytest.approx(finite, abs=1e-10)


@pytest.mark.parametrize('kind', ['empty', 'indefinite', 'nonsymmetric', 'badrep', 'correlated_quad'])
def test_invalid_domains_or_covariances_fail_explicitly(kind):
    if kind == 'empty':
        with pytest.raises(ValueError, match='zero-volume|empty'):
            line_prior(1, [1., 2.], [20., 30.], [.5, 1.])
    elif kind == 'indefinite':
        with pytest.raises(ValueError, match='semidefinite|variance'):
            bounded_line_moments([0., 0.], [[1., 2.], [2., 1.]], domain())
    elif kind == 'nonsymmetric':
        with pytest.raises(ValueError, match='symmetric'):
            bounded_line_moments([0., 0.], [[1., .1], [.3, 1.]], domain())
    elif kind == 'badrep':
        with pytest.raises(ValueError, match='two independent'):
            bounded_line_moments([0., 0.], np.eye(2), domain(), replicates=1)
    else:
        with pytest.raises(ValueError, match='diagonal'):
            line_prior_mass([0., 0.], [[1., .1], [.1, 1.]], domain())
