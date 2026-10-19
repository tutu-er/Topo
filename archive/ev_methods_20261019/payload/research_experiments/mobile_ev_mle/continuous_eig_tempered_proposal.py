"""Weighted finite-bank moments define q only, never an exact posterior."""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from .continuous_eig_sampling import log_ess
from .continuous_eig_proposals import SequentialGaussianBoxProposal


@dataclass(frozen=True)
class ProposalFit:
    proposal: object
    diagnostics: dict


def fit_tempered_proposal(component,previous,beta,logq,loglik,kappa,*,minimum_ess=None,
                           old_covariance_mass=.1,diagonal_floor=1e-6):
    """Pilot pi/q*Lhat**kappa moments; fallback preserves last valid q exactly."""
    d=len(component.lower)
    minimum_ess=d+2 if minimum_ess is None else float(minimum_ess)
    beta,logq,loglik=np.asarray(beta,float),np.asarray(logq,float),np.asarray(loglik,float)
    if not 0 <= kappa <= 1 or not 0 < old_covariance_mass < 1 or diagonal_floor <= 0:
        raise ValueError("predeclared kappa/regularization parameters invalid")
    diag=dict(kappa=float(kappa),component=component.name,minimum_ESS=minimum_ess,
        exact_tempered_posterior=False,random_Lhat_power_is_exact_marginal_power=False,
        proposal_is_prior_MLE_or_CI=False,mode_preserved=True,
        old_covariance_mass=old_covariance_mass,diagonal_floor=diagonal_floor,
        covariance_previous_is_actual_sequential_distribution_covariance=False)
    def fail(reason,ess=None):
        diag.update(status="fallback_last_valid_proposal",failure_reason=reason,conditional_ESS=ess,
                    previous_proposal_reused_unchanged=True,new_proposal_fitted=False)
        return ProposalFit(previous,diag)
    if beta.ndim!=2 or beta.shape[1]!=d or logq.shape!=(len(beta),) or loglik.shape!=(len(beta),):
        return fail("input_shape")
    if not np.isfinite(beta).all() or not np.isfinite(logq).all() or np.isnan(loglik).any() or np.isposinf(loglik).any():
        return fail("nonfinite_input_numerics")
    prior=component.prior_logpdf(beta)
    logw=prior-logq if kappa==0 else prior-logq+kappa*loglik
    total=float(logsumexp(logw));ess=float(log_ess(logw))
    diag.update(theta_count=len(beta),finite_loglik_count=int(np.isfinite(loglik).sum()),
                zero_MC_likelihood_count=int(np.isneginf(loglik).sum()),
                log_engineering_weight_sum=total,engineering_sum_is_model_evidence=False)
    if not np.isfinite(total):
        return fail("all_random_engineering_weights_zero_not_model_rejection",ess)
    if ess < minimum_ess:
        return fail("effective_sample_count_below_dimension_plus_two",ess)
    w=np.exp(logw-total);phi=beta/component.nominal_beta
    mean=w@phi;centered=phi-mean
    covariance=centered.T@(w[:,None]*centered)
    covariance=.5*(covariance+covariance.T)
    previous_covariance=np.mean(previous.covariances_phi,axis=0)
    new_covariance=(1-old_covariance_mass)*covariance+old_covariance_mass*previous_covariance+diagonal_floor*np.eye(d)
    new_covariance=.5*(new_covariance+new_covariance.T)
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all() or not np.isfinite(new_covariance).all():
        return fail("nonfinite_weighted_moments",ess)
    try:
        L=np.linalg.cholesky(new_covariance)
        new=SequentialGaussianBoxProposal(component,mean[None],new_covariance[None],.2)
    except (ValueError,np.linalg.LinAlgError,FloatingPointError) as error:
        return fail("covariance_or_normalized_proposal_failure: "+str(error),ess)
    diag.update(status="proposal_updated",failure_reason=None,conditional_ESS=ess,
        weighted_phi_mean=mean.tolist(),weighted_phi_covariance=covariance.tolist(),
        previous_engineering_covariance=previous_covariance.tolist(),new_covariance=new_covariance.tolist(),
        new_covariance_eigenvalues=np.linalg.eigvalsh(new_covariance).tolist(),
        minimum_Cholesky_diagonal=float(np.diag(L).min()),
        previous_proposal_reused_unchanged=False,new_proposal_fitted=True,
        normalized_weights_sum=float(w.sum()),uniform_prior_coverage_mass=.2,
        weighted_covariance_is_unbiased_or_confidence_region=False)
    return ProposalFit(new,diag)
