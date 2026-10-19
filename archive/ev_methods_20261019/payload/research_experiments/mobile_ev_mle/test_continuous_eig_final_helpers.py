import numpy as np
from scipy.special import logsumexp
from research_experiments.mobile_ev_mle import continuous_eig_sampling as s
from research_experiments.mobile_ev_mle.continuous_eig_guided_latent import guided_latent_likelihood
from research_experiments.mobile_ev_mle.continuous_eig_final_helpers import paired_chunked_likelihood,bank_with_new_H_estimates
from research_experiments.mobile_ev_mle.test_continuous_eig_sampling import candidates,history,ToyEngine


def test_chunking_preserves_same_theta_time_kernels_and_prefix():
    c=s.declared_components(candidates())[0];m=s.MeterSpec();h=history(2)
    b=np.stack([c.nominal_beta*(.9+.02*i) for i in range(7)])
    rng=np.random.default_rng(411013);z=rng.normal(size=(2,128,8));u=rng.random((2,128))
    direct=guided_latent_likelihood(c,b,h,m,z,u,ToyEngine,prefix_samples=(32,))
    full,prefix=paired_chunked_likelihood(c,b,h,m,z,u,ToyEngine,chunk_size=3)
    np.testing.assert_allclose(full.loglik,direct.loglik,atol=1e-12,rtol=0)
    np.testing.assert_allclose(prefix.loglik,direct.prefix_estimates[32].loglik,atol=1e-12,rtol=0)
    assert full.guidance_diagnostics["beta_chunks"]==3


def test_bank_reassembly_uses_prior_over_q_times_new_H_once_not_old_H_weights():
    cs=s.declared_components(candidates());m=s.MeterSpec();h=history(1)
    ps=[s.MixtureBoxProposal(c,c.nominal_beta,.08*c.nominal_beta,1.) for c in cs]
    template=s.build_history_bank(cs,ps,h,m,ToyEngine,n_theta=4,n_latent=4,seed=421013,
        proof_accepted=True,minimum_mode_ess=1.,minimum_total_ess=1.,maximum_weight=1.)
    rng=np.random.default_rng(431013);z=rng.normal(size=(1,128,8));u=rng.random((1,128));new={}
    for b in template.blocks:
        _,new[b.component.name]=paired_chunked_likelihood(b.component,b.beta,h,m,z,u,ToyEngine,chunk_size=2)
    bank=bank_with_new_H_estimates(template,new)
    manual=[]
    for b in template.blocks:
        c=b.component;manual.append(np.log(c.mass)-np.log(len(b.beta))+c.prior_logpdf(b.beta)-b.logq+new[c.name].loglik)
    assert abs(bank.log_evidence-logsumexp(np.concatenate(manual)))<1e-12
    assert bank.summary()["H_updates"]==1 and len(bank.blocks)==7
    assert bank.diagnostics["H_likelihood_replaced_not_multiplied"] is True
