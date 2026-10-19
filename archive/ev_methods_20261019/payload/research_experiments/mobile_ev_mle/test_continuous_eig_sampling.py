"""Independent algebra, density, support, information-flow and pairing checks."""
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

from research_experiments.mobile_ev_mle import continuous_eig_sampling as s
from research_experiments.mobile_ev_mle import continuous_ac_inverse as ac


@dataclass
class Candidate:
    name: str
    supports: tuple
    n: int = 3

    @property
    def m(self):
        return len(self.supports)


def candidates():
    result = []
    for name, pair in zip(s.TOPOLOGY_NAMES, ((), (0,1), (0,2), (1,2))):
        supports = ((0,1,2),)+((pair,) if pair else ())+((0,), (1,), (2,))
        result.append(Candidate(name, supports))
    return tuple(result)


def toy_component():
    c = Candidate("two_dim_toy", ((0,1,2),))
    return s.PriorComponent("toy", 0, None, c, .25, [1.,2.], [.8,1.6], [1.2,2.4])


class GuardedMap:
    def __init__(self, allowed):
        self.allowed = allowed
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        if key not in self.allowed:
            raise AssertionError("forbidden actual future/truth/latent/noise access")
        return self.allowed[key]

    def __iter__(self):
        raise AssertionError("NPZ enumeration or whole-file expansion is forbidden")

    def keys(self):
        raise AssertionError("NPZ enumeration is forbidden")


class ToyEngine:
    """Proper smooth background and bounded source, voltage independent of PQ."""
    seen_r = []

    @staticmethod
    def inverse_ac(candidate, r, x, p, q, master_p, master_q, *, root_v=1.):
        shape = np.broadcast_shapes(r.shape[:-1], p.shape[:-1], np.shape(master_p), np.shape(root_v))
        ToyEngine.seen_r.append(np.array(r, copy=True))
        value = np.broadcast_to(np.asarray(r)[..., 0], shape)
        # Allows a beta-dependent V test while preserving a valid density map.
        v = np.ones(shape+(3,))-.01*value[..., None]
        return dict(v=v, source_p=np.broadcast_to(master_p, shape),
                    source_q=np.broadcast_to(master_q, shape), jacobian_abs=np.ones(shape),
                    valid=np.ones(shape, bool), high_branch=np.ones(shape, bool))

    @staticmethod
    def background_logpdf(p, q):
        return (norm.logpdf(p, .12, .003)+norm.logpdf(q, .04, .003)).sum(axis=-1)

    source_logpdf = staticmethod(ac.source_logpdf)

    @staticmethod
    def forward_ac(candidate, r, x, p, q, source_p, source_q, *, root_v=1.):
        shape = p.shape[:-1]
        return dict(v=np.ones_like(p)-.01*np.asarray(r)[0], master_p=source_p,
                    master_q=source_q, valid=np.ones(shape, bool))


def history(t=2, meter=None):
    meter = s.MeterSpec() if meter is None else meter
    return s.MeterObservation(np.full((t,3), .12)+meter.bias_p,
        np.full((t,3), .04)+meter.bias_q, np.ones((t,3))-.00005+meter.bias_v,
        np.full(t,.1), np.full(t,.035))


def test_star_not_duplicated_and_canonical_independent_boxes():
    cs = s.declared_components(candidates())
    assert len(cs) == 7
    assert sum(c.topology == 0 for c in cs) == 1
    assert [c.mass for c in cs] == [.25]+[.125]*6
    assert [sum(c.mass for c in cs if c.topology == h) for h in range(4)] == [.25]*4
    for c in cs:
        np.testing.assert_allclose(c.lower, .8*c.nominal_beta)
        np.testing.assert_allclose(c.upper, 1.2*c.nominal_beta)
        assert len(c.nominal_beta) == 2*c.candidate.m
    weak = cs[1]
    strong = cs[2]
    assert weak.nominal_beta[1] == .00003 and weak.nominal_beta[weak.candidate.m+1] == .00002
    assert strong.nominal_beta[1] == .012 and strong.nominal_beta[strong.candidate.m+1] == .007


def test_lazy_history_access_whitelist_never_enumerates_future_noise():
    h = history()
    allowed = {"history_"+key:getattr(h,key) for key in (*s.CHANNELS,"root_v")}
    guarded = GuardedMap(allowed)
    extracted = s.history_from_arrays(guarded)
    assert guarded.reads == list(s.HISTORY_KEYS)
    assert extracted.identity() == h.identity()


def test_entropy_and_zero_ess_preserve_zero_probability():
    assert s.entropy([1,0,0,0]) == 0
    assert float(s.log_ess([-np.inf,-np.inf])) == 0
    assert float(s.log_ess(np.log([.1,.2,.7]))) == pytest.approx(1/(.01+.04+.49))


def test_product_truncated_normal_normalization_and_full_mixture_density():
    c = toy_component()
    p = s.MixtureBoxProposal(c, [1.,2.], [.06,.2], .2)
    marginal_product = 1.
    for i in range(2):
        lo,hi = c.lower[i],c.upper[i]
        z = norm.cdf((hi-p.mean[i])/p.sigma[i])-norm.cdf((lo-p.mean[i])/p.sigma[i])
        marginal_product *= quad(lambda b: norm.pdf((b-p.mean[i])/p.sigma[i])/p.sigma[i]/z, lo, hi)[0]
    assert marginal_product == pytest.approx(1., abs=1e-11)
    x = np.array([[1.,2.],[.81,2.3],[.79,2.]])
    normalization = np.prod(norm.cdf((c.upper-p.mean)/p.sigma)-norm.cdf((c.lower-p.mean)/p.sigma))
    independent = np.prod(norm.pdf((x-p.mean)/p.sigma)/p.sigma,axis=-1)/normalization
    supported = np.isfinite(c.prior_logpdf(x))
    manual = np.where(supported,.2*np.exp(c.prior_logconstant)+.8*independent,0.)
    np.testing.assert_allclose(np.exp(p.logpdf(x)),manual,rtol=1e-13,atol=0)


def test_untruncated_correlated_density_includes_scaled_jacobian():
    c = toy_component()
    means = np.array([[1.,1.],[.98,1.02]])
    cov = np.array([[[.01,.004],[.004,.02]],[[.03,-.003],[-.003,.012]]])
    p = s.GaussianUniformProposal(c,means,cov,.2)
    x = np.array([[1.,2.],[.7,2.],[1.2,2.4]])
    gaussian = sum(multivariate_normal.pdf(x/c.nominal_beta,mean=m,cov=k) for m,k in zip(means,cov))/2/np.prod(c.nominal_beta)
    uniform = np.exp(c.prior_logpdf(x))
    np.testing.assert_allclose(np.exp(p.logpdf(x)),.2*uniform+.8*gaussian,rtol=1e-13,atol=1e-14)
    assert np.isfinite(p.logpdf(x)[1])  # outside target still positive proposal


def test_untruncated_sampling_keeps_outside_prior_and_importance_integrates_one():
    c = toy_component()
    p = s.GaussianUniformProposal(c,[[1.,1.]],[[[.018,.007],[.007,.025]]],.2)
    beta,lq = p.sample(50000,np.random.default_rng(17813))
    outside = ~np.isfinite(c.prior_logpdf(beta))
    assert outside.sum() > 1000
    w = np.exp(c.prior_logpdf(beta)-lq)
    assert np.all(w[outside] == 0)
    # Target integral = 1; tolerance is independent Monte Carlo standard error.
    assert abs(w.mean()-1.) < 5*w.std(ddof=1)/np.sqrt(len(w))


def test_h_only_profile_covariance_formula_and_not_prior():
    c = toy_component()
    center = dict(beta=[1.01,2.02], profile_precision=[[300.,-80.],[-80.,100.]])
    p = s.GaussianUniformProposal.from_history_centers(c,[center],ridge=75.)
    scale = c.nominal_beta
    expected = np.linalg.inv(scale[:,None]*np.array(center["profile_precision"])*scale[None,:]+75*np.eye(2))
    np.testing.assert_allclose(p.covariances_phi[0],expected,rtol=1e-13)
    assert p.payload()["engineering_covariance_is_prior_or_confidence_region"] is False
    np.testing.assert_allclose(p.means_phi[0],[1.01,1.01])
    outside = dict(beta=[10.,.01],profile_precision=center["profile_precision"])
    clipped = s.GaussianUniformProposal.from_history_centers(c,[outside])
    np.testing.assert_allclose(clipped.means_phi[0],[1.2,.8])


def test_latent_estimator_gaussian_cancellation_and_time_product():
    c = s.declared_components(candidates())[0]
    beta = c.nominal_beta[None]
    meter = s.MeterSpec()
    h = history(2,meter)
    z = np.random.default_rng(13171).standard_normal((2,16000,8))
    estimate = s.latent_likelihood(c,beta,h,meter,z,ToyEngine)
    log_terminal = (3*norm.logpdf(0,0,np.sqrt(.003**2+.0006**2))+
                    3*norm.logpdf(0,0,np.sqrt(.003**2+.0004**2)))
    log_voltage = 3*norm.logpdf(0,0,.00012)
    log_source = -np.log(.07*.03)
    log_one = log_terminal+log_voltage+log_source
    np.testing.assert_allclose(estimate.log_by_time[0],log_one,atol=.006)
    assert estimate.loglik[0] == pytest.approx(estimate.log_by_time.sum())
    assert np.all(estimate.time_ess > 14000)


def test_prior_outside_points_remain_zero_and_never_enter_inverse():
    c = s.declared_components(candidates())[0]
    beta = np.stack([c.nominal_beta,2*c.nominal_beta])
    ToyEngine.seen_r = []
    result = s.latent_likelihood(c,beta,history(1),s.MeterSpec(),np.zeros((1,1,8)),ToyEngine)
    assert result.prior_outside_theta_count == 1
    assert np.isfinite(result.loglik[0]) and result.loglik[1] == -np.inf
    assert ToyEngine.seen_r[0].shape[0] == 1


def test_future_EV_removed_exactly_once_before_background_density():
    c = s.declared_components(candidates())[0]
    meter = s.MeterSpec()
    u = s.PlannedAction("known",[[.05,0,0]],[[0,0,0]])
    h = history(1,meter)
    future = s.MeterObservation(h.p+u.p,h.q,h.v,h.master_p,h.master_q,h.root_v)
    with_action = s.latent_likelihood(c,c.nominal_beta[None],future,meter,np.zeros((1,1,8)),ToyEngine,action=u)
    base = s.latent_likelihood(c,c.nominal_beta[None],h,meter,np.zeros((1,1,8)),ToyEngine)
    np.testing.assert_allclose(with_action.loglik,base.loglik,atol=1e-12)
    wrong = s.latent_likelihood(c,c.nominal_beta[None],future,meter,np.zeros((1,1,8)),ToyEngine)
    assert wrong.loglik[0] < base.loglik[0]-100


def test_proof_required_and_failed_H_never_triggers_predictive_by_default():
    cs = s.declared_components(candidates())
    ps = [s.MixtureBoxProposal(c,c.nominal_beta,.08*c.nominal_beta,1.) for c in cs]
    with pytest.raises(ValueError,match="proof"):
        s.build_history_bank(cs,ps,history(),s.MeterSpec(),ToyEngine,n_theta=4,n_latent=4)
    bank = s.build_history_bank(cs,ps,history(),s.MeterSpec(),ToyEngine,n_theta=4,n_latent=4,
        proof_accepted=True,minimum_total_ess=1000.)
    assert bank.diagnostics["status"] == "failed" and len(bank.blocks) == 7
    with pytest.raises(ValueError,match="ESS"):
        s.evaluate_paired_eig(bank,[],s.MeterSpec(),ToyEngine,n_outer=2,n_latent=2)


def test_shared_HN_beta_topology_strength_sum_and_mass_reweighting():
    cs = s.declared_components(candidates())
    ps = [s.MixtureBoxProposal(c,c.nominal_beta,.08*c.nominal_beta,1.) for c in cs]
    bank = s.build_history_bank(cs,ps,history(1),s.MeterSpec(),ToyEngine,n_theta=8,n_latent=16,
        proof_accepted=True,minimum_mode_ess=1.,minimum_total_ess=1.,maximum_weight=1.)
    ToyEngine.seen_r = []
    u = s.PlannedAction("none",np.zeros((1,3)),np.zeros((1,3)))
    d = s.predictive_logdensities(bank,history(1),u,s.MeterSpec(),ToyEngine,np.zeros((1,2,8)))
    assert len(ToyEngine.seen_r) == 7
    for actual,b in zip(ToyEngine.seen_r,bank.blocks):
        np.testing.assert_array_equal(actual[:,0,0],b.beta[:,:b.component.candidate.m])
    assert d["log_mixture"] == pytest.approx(logsumexp(d["log_topology_joint"]))
    np.testing.assert_allclose(d["log_by_topology"]+np.log(bank.topology_mass),d["log_topology_joint"])
    same = s.reweight_history_bank(bank,star_mass=.25,weak_mass=.5)
    np.testing.assert_allclose(same["topology_mass"],bank.topology_mass,atol=1e-14)


def test_outer_generation_paired_common_random_variables_and_fresh_prior():
    cs = s.declared_components(candidates())
    ps = [s.MixtureBoxProposal(c,c.nominal_beta,.08*c.nominal_beta,1.) for c in cs]
    meter = s.MeterSpec()
    bank = s.build_history_bank(cs,ps,history(1),meter,ToyEngine,n_theta=8,n_latent=8,
        proof_accepted=True,minimum_mode_ess=1.,minimum_total_ess=1.,maximum_weight=1.)
    crn = s.sample_future_crn(bank,8,2,np.random.default_rng(3181))
    assert np.all((crn.base_p >= .104) & (crn.base_p <= .136))
    assert np.all((crn.base_q >= .033) & (crn.base_q <= .047))
    assert np.all((crn.source_p >= .065) & (crn.source_p <= .135))
    assert np.all((crn.source_q >= .02) & (crn.source_q <= .05))
    a = s.PlannedAction("a",np.zeros((2,3)),np.zeros((2,3)))
    b = s.PlannedAction("b",[[.01,0,0],[.01,0,0]],np.zeros((2,3)))
    ya = s.predictive_observations(bank,crn,a,meter,ToyEngine)
    yb = s.predictive_observations(bank,crn,b,meter,ToyEngine)
    for left,right in zip(ya,yb):
        np.testing.assert_allclose(right.p-left.p,b.p,atol=1e-16)
        np.testing.assert_array_equal(left.q,right.q)
        np.testing.assert_array_equal(left.v,right.v)


def test_no_negative_EIG_clipping_or_failure_conditioning_in_serializer():
    value = s.json_safe(dict(eig=-.002,raw=[-.3,.2],failed=float("nan"),logzero=-np.inf))
    assert value == dict(eig=-.002,raw=[-.3,.2],failed=None,logzero=None)


def test_all_zero_H_retains_seven_modes_and_reports_posterior_unavailable():
    class ZeroBackground(ToyEngine):
        @staticmethod
        def background_logpdf(p,q):
            return np.full(p.shape[:-1],-np.inf)
    cs = s.declared_components(candidates())
    ps = [s.MixtureBoxProposal(c,c.nominal_beta,.08*c.nominal_beta,1.) for c in cs]
    bank = s.build_history_bank(cs,ps,history(),s.MeterSpec(),ZeroBackground,n_theta=4,n_latent=4,
        proof_accepted=True)
    assert len(bank.blocks) == 7
    assert bank.diagnostics["status"] == "failed"
    assert bank.summary()["posterior_available"] is False
    assert bank.summary()["topology_mass"] is None
    assert all(r["numerical_zero_mode_estimate"] for r in bank.diagnostics["modes"])
    with pytest.raises(ValueError,match="no posterior"):
        s.sample_future_crn(bank,2,2,np.random.default_rng(13))


def test_negative_noisy_voltage_keeps_untruncated_Gaussian_density():
    c = s.declared_components(candidates())[0]
    meter = s.MeterSpec(v_sigma=.2)
    h = history(1,meter)
    noisy = s.MeterObservation(h.p,h.q,[[-.1,0.,-.2]],h.master_p,h.master_q,1.)
    value = s.latent_likelihood(c,c.nominal_beta[None],noisy,meter,np.zeros((1,1,8)),ToyEngine)
    assert np.isfinite(value.loglik[0])
    expected_v = np.ones(3)-.01*c.nominal_beta[0]
    log_expected = (ToyEngine.background_logpdf(h.p-meter.bias_p,h.q-meter.bias_q)[0]+
                    ac.source_logpdf(.1,.035)+norm.logpdf(noisy.v[0]-meter.bias_v,expected_v,.2).sum())
    assert value.loglik[0] == pytest.approx(log_expected)
    with pytest.raises(ValueError,match="root"):
        s.MeterObservation(h.p,h.q,h.v,h.master_p,h.master_q,-1.)


@pytest.mark.parametrize("failure",["valid","high_branch","certified_domain","physical_voltage"])
def test_supported_inverse_failure_cannot_be_relabelled_zero_likelihood(failure):
    class Broken(ToyEngine):
        @staticmethod
        def inverse_ac(*args,**kwargs):
            response = ToyEngine.inverse_ac(*args,**kwargs)
            if failure == "physical_voltage":
                response["v"] = -response["v"]
            else:
                response[failure] = np.zeros_like(response["valid"])
            return response
    c = s.declared_components(candidates())[0]
    with pytest.raises(RuntimeError,match="supported proposal points"):
        s.latent_likelihood(c,c.nominal_beta[None],history(1),s.MeterSpec(),np.zeros((1,1,8)),Broken)


def test_nominal_root_and_action_contract_rejected_before_engine():
    c = s.declared_components(candidates())[0]
    h = history(1)
    wrong_root = s.MeterObservation(h.p,h.q,h.v,h.master_p,h.master_q,1.01)
    with pytest.raises(ValueError,match="root=1"):
        s.latent_likelihood(c,c.nominal_beta[None],wrong_root,s.MeterSpec(),np.zeros((1,1,8)),ToyEngine)
    actions = [s.PlannedAction("q",[[0,0,0]],[[.01,0,0]]),
               s.PlannedAction("multi",[[.01,.01,0]],[[0,0,0]]),
               s.PlannedAction("negative",[[-.01,0,0]],[[0,0,0]]),
               s.PlannedAction("large",[[.31,0,0]],[[0,0,0]])]
    for action in actions:
        with pytest.raises(ValueError,match="inverse proof"):
            s.latent_likelihood(c,c.nominal_beta[None],h,s.MeterSpec(),np.zeros((1,1,8)),ToyEngine,action=action)


def test_true_prior_outside_source_is_legitimate_zero_without_numerical_failure():
    class SourceOutside(ToyEngine):
        @staticmethod
        def inverse_ac(*args,**kwargs):
            result = ToyEngine.inverse_ac(*args,**kwargs)
            result["source_p"] = np.full_like(result["source_p"],.2)
            return result
    c = s.declared_components(candidates())[0]
    result = s.latent_likelihood(c,c.nominal_beta[None],history(1),s.MeterSpec(),np.zeros((1,1,8)),SourceOutside)
    assert result.loglik[0] == -np.inf


def test_topology_conditional_logdensity_survives_display_mass_underflow():
    cs = s.declared_components(candidates())
    ps = [s.MixtureBoxProposal(c,c.nominal_beta,.08*c.nominal_beta,1.) for c in cs]
    bank = s.build_history_bank(cs,ps,history(1),s.MeterSpec(),ToyEngine,n_theta=4,n_latent=4,
        proof_accepted=True,minimum_mode_ess=1.,minimum_total_ess=1.,maximum_weight=1.)
    blocks = tuple(replace(b,logweight=b.logweight-(1000 if b.component.topology > 0 else 0)) for b in bank.blocks)
    logz = float(logsumexp(np.concatenate([b.logweight for b in blocks])))
    masses = np.array([sum(np.exp(logsumexp(b.logweight)-logz) for b in blocks if b.component.topology == h)
                       for h in range(4)])
    weighted = replace(bank,blocks=blocks,log_evidence=logz,topology_mass=masses)
    assert np.any(weighted.topology_mass[1:] == 0)
    assert np.isfinite(weighted.topology_logmass).all()
    action = s.PlannedAction("none",np.zeros((1,3)),np.zeros((1,3)))
    result = s.predictive_logdensities(weighted,history(1),action,s.MeterSpec(),ToyEngine,np.zeros((1,4,8)))
    assert np.isfinite(result["log_by_topology"]).all()


def test_shared_common_background_cannot_be_replaced_by_independent_marginals():
    exact_nominal = .008/(.024*.008**6)
    assert np.exp(ac.background_logpdf([.12]*3,[.04]*3)) == pytest.approx(exact_nominal)
    # Every P is in its individual support, yet no shared common c can
    # simultaneously explain terminal offsets differing by more than .008.
    assert ac.background_logpdf([.110,.130,.120],[.04]*3) == -np.inf
    class CommonBackground(ToyEngine):
        background_logpdf = staticmethod(ac.background_logpdf)
    c = s.declared_components(candidates())[0]
    meter = s.MeterSpec()
    base = history(1,meter)
    value = s.latent_likelihood(c,c.nominal_beta[None],base,meter,np.zeros((1,1,8)),CommonBackground)
    expected = np.log(exact_nominal)+ac.source_logpdf(.1,.035)+3*norm.logpdf(0,0,.00012)
    assert value.loglik[0] == pytest.approx(expected)
