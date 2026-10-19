# Voltage-guided latent Gaussian integration proposal: math before MC

Parent authorization currently covers mathematics, implementation and analytic
toy tests only. Actual H integration and predictive MC remain disabled. The
v3 beta proposal/selected centers, seven proper priors and physical likelihood
are unchanged. This is a numerical integration proposal, not a physical model,
a fitted prior, an extra H update or a policy change.

## 1. Exact target and the proposal that changes

For each supported beta and time, let x=(terminal actual P[3],Q[3],masterP,
masterQ). The original eight calibrated metering Gaussian factors define
`p0(x)=N(m0,Sigma0)` with mean equal to the calibrated P/Q/master observations.
Let L0=diag(the eight known meter standard deviations), x=m0+L0*w. Then
`p0(w)=N(0,I8)`. The complete per-time model integral remains

    f(beta) = E_p0 [pV(yV | V_true(beta,x))*pbackground(P-uP,Q-uQ)*
                    psource(s_true(beta,x))*abs(det ds/dS0)].

The inverse/proof-domain restrictions and exact common-c marginal density are
unchanged. Known EV appears once in actual terminal total and is subtracted
once before background prior evaluation.

At the known calibrated meter mean m0, compute physical V0 and a finite-
difference derivative J=dV/dx. These quantities define only a proposal. Set

    H=SigmaV^(-1/2)*J*L0,
    r=SigmaV^(-1/2)*(yV_calibrated-V0),
    K=I8+H^T H,
    mu=K^(-1) H^T r,
    qg(w)=N(mu,K^(-1)).

K is positive definite because v^T K v=||v||^2+||Hv||^2>0. A Cholesky factor
of K is used; do not form a cancellation-prone covariance difference. Sample
qg by solving `L_K^T*(w-mu)=z`. Its normalized density is

    log qg(w)=.5 logdet K -4 log(2pi)-.5 (w-mu)^T K(w-mu).

Use the fully normalized mixture `q(w)=.2*p0(w)+.8*qg(w)` and account for its
full mixture density, including when a sample was generated from one branch.
For every draw, the exact importance integrand is

    p0(w)/q(w) * pV(yV | V_true(beta,m0+L0*w)) *
      pbackground(P-uP,Q-uQ)*psource(s_true)*abs(det ds/dS0).

The original metering factors no longer cancel to one. Their ratio is restored
explicitly. Both x densities carry the same L0 Jacobian, which cancels in
`p0(x)/q(x)=p0(w)/q(w)`. All V/background/source/Jacobian constants remain.
The ratio is at most five because of the .2 baseline mixture coverage.

Finite-difference errors change q and variance only; the target evaluates the
full nonlinear inverse AC voltage at every sampled x. A linear V likelihood
must never replace that target. Beta is shared between H and any prospective
future; H is updated once. Independent proposal draws across times make the
product of time means unbiased before logs/self-normalization, as before.

## 2. Domain and design failures

Observed Gaussian V readings may be any finite real number. Known physical
root remains one, and certified EV inputs remain single-node P[0,.3],Q=0.
Physical latent voltages remain positive. No noisy observation is truncated,
clipped or redrawn.

The guidance anchor or FD points need not lie in the background/source prior.
If their high-voltage response is unavailable outside true target support,
record `guidance_unavailable_outside_target_support` and use H=0,mu=0,K=I
for that beta/time. This makes q exactly p0 and retains the model. Within the
proved background/master/parameter domain, an invalid inverse remains a
computational failure and raises; it cannot become zero density or a fallback.

At sampled target points, finite background density plus the master rectangle
and supported beta require valid/high/certified positive physical response and
positive finite Jacobian. A failure there raises. Actual background/source/
master/parameter-domain outside proposals contribute legitimate zero target
weight, counted separately. Nonfinite proposal matrices/densities, Cholesky
failure or astronomically unrepresentable residuals are numerical failures;
no jitter, covariance clipping or hidden removal is applied.

## 3. Independent checks before actual H authorization

For linear V=V0+A*w and Gaussian V noise, the analytic integral is
`N(yV;V0,SigmaV+A*A^T)`. The unmixtured guided Gaussian is the exact posterior
for this linear toy, so `p0(w)/qg(w)*pV(yV|V0+A*w)` is constant and equals
that analytic evidence at arbitrary points. The .2/.8 mixture estimator must
recover the same integral. A nonlinear toy must recover the original meter-
proposal quadrature using the restored ratio, without substituting linear V.

Tests also check normalized densities, positive precision, large residual and
nearly dependent H stability, full support for negative Gaussian V readings,
all support-vs-computational-failure classifications, and preservation of the
existing target when H=0. These tests do not establish acceptable ESS on
actual H. A later actual run requires new frozen seeds/config/source hashes,
the same mode ledger and independent repeats; no old pilot weights may enter.

## 4. Root-frozen H-only v4 diagnostic

Root accepted the density ratio/white Cholesky construction and authorized
B=2048 per mode,L=32 per time,three independent repeats with fresh seeds
27101301,27101302,27101303. The v3 sequential beta proposal and v2 selected
centers/profile covariances remain frozen without reoptimization. Each repeat
uses independent normal and mixture-uniform streams, independent across time
and paired across beta. Old H particle weights are not reused. Every target
background/source-supported finite physical point must have finite logratio,
logV,logJacobian and total log target; overflow raises numerical_failure.
Finite V readings as large as1e300 therefore cannot silently become zero
Gaussian probability. True prior-outside points still contribute zero.

The current48 analytic, density, nonlinear-target and domain tests passed
before MC. Exact source/math/input hashes and complete configuration will be
saved before the v4 run. If any supported numerical failure occurs, preserve
its ledger/all seven modes and stop; do not exclude a model. Same initial ESS
thresholds remain20 conditional mode ESS,100 global ESS,.05 maximum weight.
These are numerical diagnostics, not probability calibration. Predictive MC
and latent doubling remain disabled until another explicit root freeze.
