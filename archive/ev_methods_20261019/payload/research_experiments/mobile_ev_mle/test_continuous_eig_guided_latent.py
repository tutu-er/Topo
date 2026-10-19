"""Analytic linear evidence, nonlinear quadrature and proposal contracts."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import multivariate_normal,norm

from research_experiments.mobile_ev_mle import continuous_eig_sampling as s
from research_experiments.mobile_ev_mle.continuous_eig_guided_latent import WhiteGaussianGuidance,guided_latent_likelihood
from research_experiments.mobile_ev_mle.test_continuous_eig_sampling import candidates,history,ToyEngine


def linear_problem():
    A=np.array([[.3,.1,0.,-.2],[0.,.2,.25,.1]])
    v0=np.array([1.,.95]);sigma=np.array([.2,.15]);y=np.array([.8,1.1])
    guidance=WhiteGaussianGuidance.build(A/sigma[:,None],(y-v0)/sigma)
    return guidance,A,v0,sigma,y


def test_linear_guided_gaussian_is_exact_posterior_and_restored_target_evidence_constant():
    g,A,v0,sigma,y=linear_problem()
    rng=np.random.default_rng(311013)
    w=rng.normal(size=(200,4))
    logv=norm.logpdf(y,v0+w@A.T,sigma).sum(axis=-1)
    restored=g.logpdf_baseline(w)-g.logpdf_guided(w)+logv
    exact=multivariate_normal.logpdf(y,v0,np.diag(sigma**2)+A@A.T)
    np.testing.assert_allclose(restored,exact,rtol=0,atol=2e-13)
    K=np.eye(4)+(A/sigma[:,None]).T@(A/sigma[:,None])
    np.testing.assert_allclose(g.precision_cholesky@g.precision_cholesky.T,K,rtol=1e-14)


def test_full_mixture_restores_original_meter_gaussian_linear_integral():
    g,A,v0,sigma,y=linear_problem()
    rng=np.random.default_rng(321013)
    z=rng.normal(size=(50000,4));u=rng.random(50000)
    w,ratio=g.sample_mixture(z,u,.2)
    weights=np.exp(ratio+norm.logpdf(y,v0+w@A.T,sigma).sum(axis=-1))
    exact=multivariate_normal.pdf(y,v0,np.diag(sigma**2)+A@A.T)
    assert abs(weights.mean()-exact)<5*weights.std(ddof=1)/np.sqrt(len(weights))
    expected=np.logaddexp(np.log(.2)+g.logpdf_baseline(w),np.log(.8)+g.logpdf_guided(w))
    np.testing.assert_allclose(g.logpdf_mixture(w,.2),expected,rtol=0,atol=3e-15)
    assert np.exp(ratio).max()<=5.+1e-12


def test_gaussian_guidance_density_normalizes_and_samples_covariance():
    g=WhiteGaussianGuidance.build([[1.7]],[.9])
    integral=quad(lambda w:float(np.exp(g.logpdf_guided(np.array([[w]])))[0]),-np.inf,np.inf)[0]
    assert integral==pytest.approx(1.,abs=1e-9)
    draws=g.sample_guided(np.random.default_rng(331013).normal(size=(40000,1)))[:,0]
    expected_var=1/(1+1.7**2)
    assert abs(draws.mean()-g.mean[0])<5*np.sqrt(expected_var/len(draws))
    assert draws.var()==pytest.approx(expected_var,rel=.025)


def test_nonlinear_true_voltage_recovers_independent_original_meter_quadrature():
    v0,A,sigma,y=1.,.3,.2,.9
    g=WhiteGaussianGuidance.build([[A/sigma]],[(y-v0)/sigma])
    physical=lambda w:v0+A*w+.2*w*w
    exact=quad(lambda w:norm.pdf(w)*norm.pdf(y,physical(w),sigma),-np.inf,np.inf,epsabs=1e-11)[0]
    rng=np.random.default_rng(341013)
    w,ratio=g.sample_mixture(rng.normal(size=(60000,1)),rng.random(60000),.2)
    # The target uses the quadratic response, even though q is linear guidance.
    weights=np.exp(ratio+norm.logpdf(y,physical(w[:,0]),sigma))
    assert abs(weights.mean()-exact)<5*weights.std(ddof=1)/np.sqrt(len(weights))
    incorrect=norm.pdf(y,v0,np.sqrt(sigma**2+A**2))
    assert abs(exact-incorrect)>.03


def test_precision_positive_with_large_residual_and_nearly_dependent_H_without_jitter():
    H=np.array([[1e5,1e5],[1e5,1e5+1e-3]])
    g=WhiteGaussianGuidance.build(H,[1e6,-1e6])
    assert np.isfinite(g.mean).all() and np.isfinite(g.precision_cholesky).all()
    draws=g.sample_guided(np.random.default_rng(351013).normal(size=(100,2)))
    assert np.isfinite(g.logpdf_guided(draws)).all()
    with pytest.raises(ValueError,match="finite"):
        WhiteGaussianGuidance.build([[np.nan]],[1.])


def test_extreme_finite_residual_keeps_log_mixture_ratio_representable():
    g=WhiteGaussianGuidance.build([[.2]],[1e6])
    z=np.array([[0.],[1.],[-1.],[2.]])
    w,ratio=g.sample_mixture(z,[.9,.9,.1,.9],.2)
    assert np.isfinite(w).all() and np.isfinite(ratio).all()
    assert np.isfinite(g.logpdf_mixture(w)).all()
    assert ratio.max()<=np.log(5)+1e-10


def test_negative_Gaussian_voltage_reading_keeps_full_real_observation_support():
    g=WhiteGaussianGuidance.build([[.2/.1]],[(-.2-1.)/.1])
    # Negative y is a valid residual; q remains a normalized proper Gaussian.
    assert np.isfinite(g.mean).all()
    assert g.logpdf_mixture(np.array([[0.],[1.],[-3.]])).shape==(3,)
    exact=norm.pdf(-.2,1.,np.sqrt(.1**2+.2**2))
    w=np.array([[0.],[1.],[-3.]])
    restored=np.exp(g.logpdf_baseline(w)-g.logpdf_guided(w)+norm.logpdf(-.2,1.+.2*w[:,0],.1))
    np.testing.assert_allclose(restored,exact,rtol=2e-13)


def test_zero_guidance_matches_original_nonlinear_target_and_meter_cancellation():
    c=s.declared_components(candidates())[0];meter=s.MeterSpec();obs=history(2,meter)
    z=np.random.default_rng(361013).normal(size=(2,64,8));u=np.random.default_rng(371013).random((2,64))
    original=s.latent_likelihood(c,c.nominal_beta[None],obs,meter,z,ToyEngine)
    guided=guided_latent_likelihood(c,c.nominal_beta[None],obs,meter,z,u,ToyEngine)
    np.testing.assert_allclose(guided.loglik,original.loglik,rtol=0,atol=1e-12)
    np.testing.assert_allclose(guided.time_ess,original.time_ess,rtol=1e-12)
    assert guided.guidance_diagnostics["meter_ratio_restored"] is True
    assert guided.guidance_diagnostics["linear_V_used_in_target"] is False


def test_guidance_invalid_inside_target_support_raises_not_fallback_or_zero():
    class Broken(ToyEngine):
        @staticmethod
        def inverse_ac(*args,**kwargs):
            out=ToyEngine.inverse_ac(*args,**kwargs);out["valid"]=np.zeros_like(out["valid"]);return out
    c=s.declared_components(candidates())[0]
    with pytest.raises(RuntimeError,match="cannot assign zero density or guidance fallback"):
        guided_latent_likelihood(c,c.nominal_beta[None],history(1),s.MeterSpec(),np.zeros((1,2,8)),np.zeros((1,2)),Broken)


def test_guidance_unavailable_outside_true_support_retains_baseline_and_model():
    class Outside(ToyEngine):
        @staticmethod
        def inverse_ac(*args,**kwargs):
            out=ToyEngine.inverse_ac(*args,**kwargs)
            out["valid"]=np.zeros_like(out["valid"])
            out["v"]=np.full_like(out["v"],np.nan)
            return out
        @staticmethod
        def background_logpdf(p,q):
            return np.full(p.shape[:-1],-np.inf)
    c=s.declared_components(candidates())[0]
    out=guided_latent_likelihood(c,c.nominal_beta[None],history(1),s.MeterSpec(),np.zeros((1,2,8)),np.zeros((1,2)),Outside)
    assert out.loglik[0]==-np.inf
    assert out.guidance_diagnostics["guidance_fallback_outside_target_support_count"]==1
    assert out.guidance_diagnostics["fallback_is_normalized_baseline_not_model_deletion"] is True


def test_beta_outside_prior_retained_without_guidance_engine_access():
    c=s.declared_components(candidates())[0]
    values=np.stack([c.nominal_beta,2*c.nominal_beta])
    out=guided_latent_likelihood(c,values,history(1),s.MeterSpec(),np.zeros((1,2,8)),np.zeros((1,2)),ToyEngine)
    assert out.prior_outside_theta_count==1 and out.loglik[1]==-np.inf and np.isfinite(out.loglik[0])


def test_history_bank_explicit_guided_evaluator_updates_H_once_and_preserves_all_modes():
    cs=s.declared_components(candidates())
    ps=[s.MixtureBoxProposal(c,c.nominal_beta,.08*c.nominal_beta,1.) for c in cs]
    h=history(1);meter=s.MeterSpec();called=[]
    def evaluator(c,b,h,m,z,e):
        called.append(c.name)
        return guided_latent_likelihood(c,b,h,m,z,np.zeros(z.shape[:2]),e)
    kwargs=dict(n_theta=4,n_latent=8,seed=381013,proof_accepted=True,
                minimum_mode_ess=1.,minimum_total_ess=1.,maximum_weight=1.)
    baseline=s.build_history_bank(cs,ps,h,meter,ToyEngine,**kwargs)
    guided=s.build_history_bank(cs,ps,h,meter,ToyEngine,latent_evaluator=evaluator,**kwargs)
    assert called==[c.name for c in cs]
    assert len(guided.blocks)==7 and guided.summary()["H_updates"]==1
    assert guided.log_evidence==pytest.approx(baseline.log_evidence,abs=1e-12)


def test_finite_astronomical_noisy_V_raises_numeric_failure_not_Gaussian_probability_zero():
    c=s.declared_components(candidates())[0];h=history(1);meter=s.MeterSpec()
    huge=s.MeterObservation(h.p,h.q,np.full((1,3),1e300),h.master_p,h.master_q,1.)
    with pytest.raises(FloatingPointError):
        guided_latent_likelihood(c,c.nominal_beta[None],huge,meter,np.zeros((1,2,8)),np.zeros((1,2)),ToyEngine)
    with pytest.raises(FloatingPointError):
        s.latent_likelihood(c,c.nominal_beta[None],huge,meter,np.zeros((1,2,8)),ToyEngine)


def test_actual_prior_outside_zero_does_not_trigger_false_numeric_failure():
    class Zero(ToyEngine):
        @staticmethod
        def background_logpdf(p,q):
            return np.full(p.shape[:-1],-np.inf)
    c=s.declared_components(candidates())[0];h=history(1);meter=s.MeterSpec()
    huge=s.MeterObservation(h.p,h.q,np.full((1,3),1e300),h.master_p,h.master_q,1.)
    out=guided_latent_likelihood(c,c.nominal_beta[None],huge,meter,np.zeros((1,2,8)),np.zeros((1,2)),Zero)
    assert out.loglik[0]==-np.inf and out.background_outside_point_count==2


def test_integration_prefix_uses_identical_per_time_samples_and_counts():
    c=s.declared_components(candidates())[0];h=history(2);m=s.MeterSpec()
    rng=np.random.default_rng(401013);z=rng.normal(size=(2,128,8));u=rng.random((2,128))
    full=guided_latent_likelihood(c,c.nominal_beta[None],h,m,z,u,ToyEngine,prefix_samples=(32,))
    direct=guided_latent_likelihood(c,c.nominal_beta[None],h,m,z[:,:32],u[:,:32],ToyEngine)
    prefix=full.prefix_estimates[32]
    np.testing.assert_allclose(prefix.loglik,direct.loglik,atol=1e-12,rtol=0)
    np.testing.assert_allclose(prefix.time_ess,direct.time_ess,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(prefix.time_positive_count,direct.time_positive_count)
    assert prefix.samples_per_time==32 and full.samples_per_time==128
