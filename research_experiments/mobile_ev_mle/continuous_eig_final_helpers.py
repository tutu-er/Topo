"""Chunking and pure reassembly: each L level targets original H once."""
from dataclasses import replace
import numpy as np
from scipy.special import logsumexp
from . import continuous_eig_sampling as s
from .continuous_eig_guided_latent import GuidedLatentEstimate,guided_latent_likelihood


def concatenate_estimates(items):
    first=items[0];diags=[x.guidance_diagnostics for x in items];diag=dict(diags[0])
    for key in ("guidance_available_beta_time_count","guidance_fallback_outside_target_support_count"):
        diag[key]=sum(d[key] for d in diags)
    diag["minimum_precision_cholesky_diagonal"]=min(d["minimum_precision_cholesky_diagonal"] for d in diags)
    for key in ("maximum_absolute_white_mean","maximum_log_meter_ratio"):
        diag[key]=max(d[key] for d in diags)
    diag.update(beta_chunks=len(items),chunking_changes_only_memory=True)
    return GuidedLatentEstimate(np.concatenate([x.loglik for x in items]),np.concatenate([x.log_by_time for x in items]),
        np.concatenate([x.time_ess for x in items]),np.concatenate([x.time_positive_count for x in items]),
        sum(x.invalid_inverse_count for x in items),first.samples_per_time,sum(x.prior_outside_theta_count for x in items),
        sum(x.background_outside_point_count for x in items),sum(x.master_rectangle_outside_point_count for x in items),
        sum(x.source_prior_outside_supported_point_count for x in items),diag)


def paired_chunked_likelihood(component,beta,obs,meter,z,u,engine,*,chunk_size=256):
    full,prefix=[],[]
    for start in range(0,len(beta),chunk_size):
        result=guided_latent_likelihood(component,beta[start:start+chunk_size],obs,meter,z,u,engine,
            baseline_mass=.2,fd_white_step=.02,prefix_samples=(32,))
        full.append(result);prefix.append(result.prefix_estimates[32])
    return concatenate_estimates(full),concatenate_estimates(prefix)


def bank_with_new_H_estimates(template,estimates):
    """Reuse theta/logq only. Never multiply old H weights by a new H factor."""
    blocks=[];rows=[]
    for old in template.blocks:
        c=old.component;result=estimates[c.name];B=len(old.beta)
        lw=np.log(c.mass)-np.log(B)+c.prior_logpdf(old.beta)-old.logq+result.loglik
        blocks.append(replace(old,history=result,logweight=lw))
        row=result.diagnostics();mode_total=float(logsumexp(lw))
        row.update(component=c.name,topology=c.topology,mode_ess=float(s.log_ess(lw)),
            conditional_particle_ESS=float(s.log_ess(lw)),log_mode_evidence=mode_total-np.log(c.mass),
            numerical_zero_mode_estimate=not np.isfinite(mode_total),numerical_zero_is_exact_model_rejection=False,
            proposal=old.proposal.payload(),mode_preserved=True,
            single_time_marginals_are_joint_history_evidence=False)
        time_terms=(c.prior_logpdf(old.beta)-old.logq-np.log(B))[:,None]+result.log_by_time
        row["log_single_time_marginal_evidence"]=logsumexp(time_terms,axis=0).tolist()
        if np.isfinite(mode_total):
            w=np.exp(lw-mode_total);active=w>0;L=result.samples_per_time
            with np.errstate(divide="ignore",invalid="ignore"):
                relative=np.sqrt(np.maximum(L/result.time_ess-1,0)/(L-1))
            row["H_weighted_mean_time_MC_relative_SE"]=np.sum(w[active,None]*relative[active],axis=0).tolist()
        else:
            row["H_weighted_mean_time_MC_relative_SE"]=None
        rows.append(row)
    flat=np.concatenate([b.logweight for b in blocks]);logz=float(logsumexp(flat))
    topology=np.zeros(4) if not np.isfinite(logz) else np.array([
        sum(np.exp(logsumexp(b.logweight)-logz) for b in blocks if b.component.topology==h) for h in range(4)])
    ess=float(s.log_ess(flat));maximum=0. if not np.isfinite(logz) else float(np.exp(flat.max()-logz))
    limits=template.diagnostics;problems=[]
    if not np.isfinite(logz):problems.append("all_mode_likelihood_estimates_zero_no_posterior_available")
    if any(r["numerical_zero_mode_estimate"] for r in rows):problems.append("one_or_more_mode_likelihood_estimates_zero_not_exact_rejection")
    if any(r["mode_ess"]<limits["minimum_mode_ess"] for r in rows):problems.append("one_or_more_mode_ESS_below_frozen_minimum")
    if ess<limits["minimum_total_ess"]:problems.append("overall_ESS_below_frozen_minimum")
    if maximum>limits["maximum_weight_limit"]:problems.append("maximum_particle_weight_above_frozen_maximum")
    for row,b in zip(rows,blocks):
        row["numerical_posterior_mode_mass"]=0. if not np.isfinite(logz) else float(np.exp(logsumexp(b.logweight)-logz))
    diagnostics=dict(limits);diagnostics.update(status="failed" if problems else "passed_initial_ESS",
        failure_reasons=problems,total_ess=ess,maximum_particle_weight=maximum,modes=rows,
        previous_H_weights_reused=False,H_likelihood_replaced_not_multiplied=True)
    return s.HistoryBank(tuple(blocks),logz,topology,diagnostics,template.history_hash,template.meter_hash,
                         template.seed,template.proof_accepted)
