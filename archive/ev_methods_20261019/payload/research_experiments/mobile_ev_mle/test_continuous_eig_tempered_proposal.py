import numpy as np
from scipy.special import logsumexp
from research_experiments.mobile_ev_mle.continuous_eig_proposals import SequentialGaussianBoxProposal
from research_experiments.mobile_ev_mle.continuous_eig_tempered_proposal import fit_tempered_proposal
from research_experiments.mobile_ev_mle.test_continuous_eig_sampling import toy_component


def setup():
    c=toy_component();C=np.array([[.01,.003],[.003,.02]])
    previous=SequentialGaussianBoxProposal(c,[[1.,1.]],C[None],.2)
    x=np.linspace(.81,1.19,20);X,Y=np.meshgrid(x,x)
    phi=np.column_stack((X.ravel(),Y.ravel()));beta=phi*c.nominal_beta
    return c,previous,phi,beta


def test_weighted_population_moments_regularization_and_exact_normalized_new_q():
    c,previous,phi,beta=setup();q=c.prior_logpdf(beta)
    loglik=-30*(phi[:,0]-1)**2-70*(phi[:,1]-1)**2
    result=fit_tempered_proposal(c,previous,beta,q,loglik,.25)
    assert result.diagnostics["status"]=="proposal_updated"
    w=np.exp(.25*loglik-logsumexp(.25*loglik));mean=w@phi;delta=phi-mean
    C=delta.T@(w[:,None]*delta)
    expected=.9*C+.1*previous.covariances_phi[0]+1e-6*np.eye(2)
    np.testing.assert_allclose(result.proposal.means_phi[0],mean,rtol=1e-13)
    np.testing.assert_allclose(result.proposal.covariances_phi[0],expected,rtol=1e-13)
    draws,lq=result.proposal.sample(40000,np.random.default_rng(391013))
    iw=np.exp(c.prior_logpdf(draws)-lq)
    assert abs(iw.mean()-1)<5*iw.std(ddof=1)/np.sqrt(len(iw))
    assert result.diagnostics["exact_tempered_posterior"] is False


def test_one_effective_sample_never_manufactures_a_covariance_and_fallback_identity():
    c,previous,phi,beta=setup();q=c.prior_logpdf(beta)
    loglik=np.full(len(beta),-1e6);loglik[0]=0.
    result=fit_tempered_proposal(c,previous,beta,q,loglik,.1)
    assert result.proposal is previous
    assert result.diagnostics["conditional_ESS"]==1
    assert result.diagnostics["mode_preserved"] is True
    assert result.diagnostics["new_proposal_fitted"] is False


def test_all_zero_or_nonfinite_input_retains_mode_and_last_valid_proposal():
    c,previous,phi,beta=setup();q=c.prior_logpdf(beta)
    zero=fit_tempered_proposal(c,previous,beta,q,np.full(len(beta),-np.inf),.1)
    assert zero.proposal is previous and zero.diagnostics["conditional_ESS"]==0
    badq=q.copy();badq[0]=np.nan
    failed=fit_tempered_proposal(c,previous,beta,badq,np.zeros(len(beta)),.1)
    assert failed.proposal is previous and failed.diagnostics["failure_reason"]=="nonfinite_input_numerics"


def test_kappa_zero_ignores_random_zero_likelihood_instead_of_nan_zero_times_inf():
    c,previous,phi,beta=setup();q=c.prior_logpdf(beta)
    result=fit_tempered_proposal(c,previous,beta,q,np.full(len(beta),-np.inf),0.)
    assert result.diagnostics["status"]=="proposal_updated"
    assert abs(result.diagnostics["conditional_ESS"]-len(beta))<1e-9


def test_prior_over_full_q_enters_moments_not_likelihood_power_alone():
    c,previous,phi,beta=setup();lq=previous.logpdf(beta)
    ll=np.zeros(len(beta));result=fit_tempered_proposal(c,previous,beta,lq,ll,.3)
    logw=c.prior_logpdf(beta)-lq;w=np.exp(logw-logsumexp(logw))
    np.testing.assert_allclose(result.proposal.means_phi[0],w@phi,rtol=1e-12)
    assert result.diagnostics["engineering_sum_is_model_evidence"] is False
