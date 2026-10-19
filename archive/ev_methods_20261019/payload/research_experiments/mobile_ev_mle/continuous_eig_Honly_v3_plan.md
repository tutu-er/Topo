# Sequential conditional truncated-Gaussian proposal: math first, not yet MC

This changes only the H-only engineering importance proposal. The seven-mode
proper nominal prior, physical inverse model, measurement likelihood, one H
update and formal C/D/stress policies remain unchanged. No predictive MC is
authorized. Parent authorization is required before actual-H v3 integration.

## 1. Proposal definition and normalization

For one existing H-only engineering center and covariance, let
phi=beta/canonical. Permute coordinates by decreasing scaled covariance
diagonal, with stable index tie-breaking, so broad coordinates precede narrow
ones. The permutation has absolute determinant one. In permuted coordinates
write covariance=C=LL^T and center=mu. For j=1,...,d define

    m_j(eta_previous)=mu_j+sum_{i<j} L_ji eta_i,
    a_j=(lower_phi_j-m_j)/L_jj,
    b_j=(upper_phi_j-m_j)/L_jj,
    Z_j=Phi(b_j)-Phi(a_j).

Draw eta_j conditionally from a standard normal truncated to [a_j,b_j], then
set phi_j=m_j+L_jj eta_j. Every draw lies in the declared prior box. At a given
phi, recover eta=L^(-1)(phi-mu), recompute all past-dependent a_j,b_j,Z_j, and
evaluate

    log q_phi(phi)=sum_j[-eta_j^2/2-log(2pi)/2-log L_jj-log Z_j].

Each conditional phi_j density integrates to one on its prescribed interval
for every past phi sequence. Repeated integration from the last coordinate
to the first therefore proves joint normalization. Positivity of every
diagonal L_jj gives full box support. The proposal uses the correlations of
the engineering Gaussian conditionals, but its final mean/covariance generally
do not equal the original mu/C after truncation.

This is **not** a multivariate normal conditioned on the entire box. That
distribution would require a single multivariate box-probability normalizer.
Here the product of normalizers depends on the earlier coordinates. Omitting
that dependence would produce incorrect importance weights.

Convert to beta density by

    log q_beta(beta)=log q_phi(beta/canonical)-sum_j log canonical_j.

For the retained two original H profile covariance components, use the v2
local selected center and their respective C matrices. The complete proposal
is .2 uniform prior box + .8 equally split over these sequential components.
Its full mixture density, not the sampled component density, enters
`priorpdf(beta)/q_beta(beta)`. Existing component prior masses are unchanged.

## 2. Stable tail normalizers and sampling

For b<=0 use

    logZ=logPhi(b)+log(-expm1(logPhi(a)-logPhi(b))).

For a>=0 evaluate the same difference in survival coordinates,

    logZ=logPhi(-a)+log(-expm1(logPhi(-b)-logPhi(-a))).

For a<0<b, use the stable positive erf difference
`Z=(erf(b/sqrt(2))-erf(a/sqrt(2)))/2`. Deep-tail logPhi comes from log_ndtr.
No tiny box probability is exponentiated before taking its logarithm.

With an interior uniform u, negative-tail sampling uses

    logP=logaddexp(logPhi(a),log(u)+logZ), eta=ndtri_exp(logP).

Positive-tail sampling uses

    logS=logaddexp(logPhi(-b),log(u)+logZ), eta=-ndtri_exp(logS).

Cross-zero intervals can use the negative-tail formula with stable logZ.
For upper-half cross-zero quantiles the corresponding log-survival expression
uses `(1-u)*Z` to avoid a last floating uniform causing CDF=1 rounding.
Only last-bit rounding is corrected to the closed interval; no out-of-domain
sample is discarded or redrawn. Unsupported/nonfinite covariance, interval,
normalizer, or density is a numerical failure to retain, not a hidden exclusion.
Every rounding correction is recorded. The allowed error is at most
`128*machine_epsilon*max(1,abs(value),abs(bounds))`; final beta multiplication
is checked in phi units so tiny weak canonical impedances cannot hide an
appreciable relative correction. Larger excursions raise a computational
failure. Nonfinite means are explicitly rejected.

## 3. Independent tests required before actual-H MC

1. Two-dimensional conditional integrals and full joint quadrature equal one.
2. Density matches an independent explicit two-dimensional conditional-normal
   formula, including its second-coordinate past-dependent Z.
3. Samples remain in the box; correlations retain the engineering sign in
   a centered example, while no claim equates them to the original covariance.
4. Exact mixture importance ratio integrates the uniform target to one;
   beta-scale/permutation Jacobians are included.
5. Deep positive and negative tail intervals remain finite and generate within
   bounds without exponentiating underflowed normalizers.
6. All prior modes remain present and no actual future/truth field is used.

Actual-H candidate remains B=2048/mode,L=32/time,three fresh independent
repeats. Root has now frozen fresh seeds26101301,26101302,26101303 and authorized
this H-only run after the density/roundoff tests. The v2 selected means are
reused without further optimization. Complete source/configuration snapshots
will be saved before MC. Existing v1/v2 failures and exact source snapshots
remain intact; predictive MC remains disabled.

## 4. What v2 established and what it did not

All seven bounded Powell searches completed without numerical failure or budget
stops. They are local fixed-cloud searches, not global maxima. Fresh final IS
still failed: global ESS2.234/3.709/5.356,maxweight .656/.470/.369,log evidence
range .8877. Evidence variation improved but did not meet the predeclared ESS
criteria. No predictive MC was run or accepted topology ranking reported.

For the numerically highest-quality mode, the selected center's fixed-cloud
per-time latent ESS median is24.79/32 and relativeSE median .0968. In fresh IS,
mode-conditional H-weighted per-time relativeSE is approximately .138–.151,
while the full beta cloud's median latent ESS is only2.47–2.91. All modes still
have roughly 380–457 supported beta points out of2048 because the untruncated
Gaussian mostly escapes the prior box. Several selected coordinates sit near
faces and the engineering covariance has strong thin correlated directions;
these observations motivate a supported conditional proposal.

Other modes still have conditional latent ESS near one and severe voltage-tail
importance degeneracy. V3 may improve beta coverage without curing their
latent integration. Even if beta ESS improves, single-time variance, independent
repeat evidence and integration doubling must still be checked. A voltage-aware
latent proposal would require restoring the metering-density/proposal-density
ratio; the original Gaussian cancellation cannot be reused after changing the
latent proposal. That is a separate future design, not part of this v3 step.
