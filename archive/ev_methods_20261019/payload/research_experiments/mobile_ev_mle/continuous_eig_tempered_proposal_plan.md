# H-only tempered engineering pilot: fixed model, fresh final target

The parent has frozen kappa=[.001,.003,.01,.03,.1,.3,1],2048 theta/mode,
32 guided latent proposals/time,stage seeds28201301–28201307,BelowNormal,
threads1. Seven modes remain present. The soft pilot wall budget is20min,
checked after each complete stage; checkpoint and stop before another stage
if exceeded. Parent's outer deadline is01:00UTC. Final independent integration
and predictive MC have not been authorized.

## 1. Pilot is only proposal engineering

Within each mode, draw a fresh theta batch from its current fully normalized
uniform-plus-sequential-conditional-TN q. Draw fresh independent-time Gaussian
and mixture-uniform latent streams, and evaluate the same exact nonlinear
target with the restored meter-density ratio. Let Lhat_H denote the positive
or zero Monte Carlo product of time means, not an exact marginal likelihood.

At stage kappa, engineering weights are

    logw=log pi(beta)-log q(beta)+kappa*log Lhat_H(beta).

The uniform impedance prior pi is unchanged. These weights are not reported
as topology posterior probabilities, confidence regions, MLEs or EIG. In
particular E[Lhat^kappa] generally differs from L^kappa; for0<kappa<1 the
concavity bias is downward. The random pilot therefore does not sample an
exact tempered Bayesian posterior. It supplies numerical proposal moments.

Each stage uses a fresh beta/latent batch. No old particle weights are bridged
or accumulated. Fit q parameters from this batch only; the next stage's
importance denominator is its entire updated mixture density. All mode
masses and physical priors remain fixed. A final experiment, if parent later
authorizes it, must freeze q and use fresh independent beta/latent streams
with final weights pi/q*Lhat_H once. Pilot weights never enter that estimate.

## 2. Weighted moment fit and failure rule

Set phi=beta/canonical and normalize the finite engineering weights by
logsumexp. Its discrete weighted mean and covariance are

    mu=sum_i w_i phi_i,
    Cw=sum_i w_i (phi_i-mu)(phi_i-mu)^T.

These are moments of the finite random engineering bank, not unbiased moments
or an uncertainty certificate for a mathematical posterior. Define Cprev as
the arithmetic mean of the current proposal's engineering covariance matrices
(not its actual truncated distribution covariance). The proposed new Gaussian
conditional parameters are

    Cnext=.9*Cw+.1*Cprev+1e-6*I,
    mean_next=mu.

Use one sequential conditional-TN component with these parameters and the
same .2 uniform prior-box coverage. This preserves exact normalized q and
all past-dependent interval normalizers and beta Jacobian. The covariance
formula is predeclared proposal regularization; it does not change pi.

Require conditional engineering ESS>=d+2, finite moments and a successful
Cholesky of Cnext. If any fails, record the failure, retain the exact last
valid proposal unchanged, and keep the mode. No covariance is manufactured
from one effective particle. No covariance eigenvalue clipping is applied.

## 3. Why the smaller first temperature was chosen

Read-only postprocessing of the existing V4 banks found that kappa=.1 would
give most modes only ESS1–3.15; even the numerically highest-quality mode had
ESS18.12/1.56/4.79 across its three saved batches. Most moment fits would fail
the d+2 threshold immediately. With kappa=.001 every existing mode/batch had
ESS17.89–307.94. This supported the parent's smaller fixed start. It is a
diagnostic on existing samples, not a new pilot or a guarantee that fresh
batches will pass. ESS need not be monotone in kappa because pi/q varies.

## 4. Practical limits to inspect

Sequential truncation changes actual q mean/covariance, so Gaussian conditional
parameters fitted from moments do not make the resulting supported proposal
moment-matched automatically. Near a prior face, a Gaussian mean inside the
box can move farther inside after conditional truncation. Also the1e-6 scaled
diagonal floor sets a Gaussian width near.001, which may be too broad for
extremely sharp boundary concentrations in poor-fit modes. These are possible
efficiency limits, not permission to tune this frozen pilot after results.

Record stage/mode seed, source/input hashes, q before/after, ESS, weighted
moments, covariance spectrum/Cholesky, latent ESS/relative-SE diagnostics,
finite/zero likelihood counts and every fallback. If supported-domain target
numerics fail, preserve the ledger and stop; never exclude a model. A final
H bank is unavailable until fresh integration passes its own ESS/repeat and
integration-sensitivity checks. No posterior entropy or predictive ranking is
accepted from the pilot.
