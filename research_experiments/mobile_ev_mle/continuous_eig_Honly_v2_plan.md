# H-only importance pilot v2: frozen-model proposal repair, not yet executed

The v1 target and all physical priors remain identical. This plan is a proposal
diagnostic only, does not alter C/D/stress, and does not run predictive EIG.
Root must freeze execution before this plan is run.

## Existing H-only evidence

The parent-authorized H710009 v1 used seven modes, 2048 theta draws/mode,
32 independent latent draws/time, three independent repeats, and exact
Gaussian+uniform proposal density. All seven modes were retained in all three
repeats. No mode likelihood estimate was exactly zero in finite arithmetic.
All inverse invalid/certification counts were zero in the executed v1 engine.

All three runs failed the predeclared ESS criteria. Global ESS was approximately
one, maximum particle weight approximately one, and conditional mode ESS ranged
from 1 to 1.00891. The log evidence range over independent repeats was 25.5612,
equivalent to an evidence ratio of about 1.26e11. This is not a reliable
posterior approximation or a basis for accepted topology/action ranking.

Existing unconstrained H fit centers mostly lie outside the declared impedance
boxes. Clipping puts every coordinate on a prior face for star and most modes;
the other modes have 8 or 9 out of 10 coordinates on faces. Individual scaled
proposal standard deviations span approximately .0451 to .1155. Between 1599
and 1668 out of 2048 theta samples per mode were outside the target prior and
correctly retained at zero target weight. Only 380–449 supported samples/mode
remained. This is proposal undercoverage of the interior, not a change to the
prior or a reason to redraw outside samples.

Median positive per-time latent ESS ranges approximately 1.005–2.817 out of 32.
Sample-moment relative SE of a per-time likelihood mean on finite-joint-theta
samples is approximately .578–.997. The product across 12 independent times
therefore remains highly variable. V1 exhibits both beta proposal and latent
importance degeneracy. Merely increasing outer predictive samples would not
repair either problem. No predictive run is requested at this stage.

## Entry contract repaired before any v2 execution

The current sampler permits any finite noisy V observation because the meter
noise is an untruncated additive Gaussian; physical latent/forward voltages
must remain positive. Known physical root must equal one. Commands must have
QEV=0 and single-site PEV in [0,.3] per time. These conditions match the accepted
uniform high-voltage inverse proof.

For supported beta, finite background density and master powers inside the
proved rectangle, inverse numerical/high-branch/certificate/nonpositive-physical
voltage/nonpositive-Jacobian failure raises a computational failure. It is
never relabeled zero target density. Gaussian proposals outside a real
background/source/parameter/master support still correctly contribute zero.
These cases are separately counted. Twenty-two sampler tests passed after
these contract repairs; the executed v1 numerical inputs had no such failures.
Its exact old source/math snapshots remain in `continuous_eig_Honly_v1` and
match every source/math hash in `config_before_MC.json`.

## Proposed local engineering center

For each of seven modes, take its highest *H likelihood* supported particle
from v1 (maximize stored loghistory, not normalized posterior mass and not
importance-weight rank). Use it only as a starting point. Save its index,
beta, likelihood and input/source hashes.

Freeze a fresh H-only latent Gaussian cloud, L=32/time, independently across
time, with one of the seven proposed optimization seeds 24101301–24101307.
Within scaled phi=beta/canonical in [.8,1.2]^d, minimize

    objective(phi) = -log fhat_H(canonical*phi; fixed latent cloud).

Use bounded Powell with xtol=.0005, ftol=.001, maxfev=2000 and a hard
60-second budget per mode checked before each objective evaluation. The
common-c intersection density is piecewise smooth, so Powell is a reasonable
local engineering choice. Save every attempted evaluation's finite/zero status
and the best feasible point. If time stops, retain the best evaluated point
and the stopped status. Do not call success or the selected point a global
likelihood maximum or a posterior mode. A failed mode remains present.

The objective reuses a fixed latent cloud for a smoother deterministic local
search. This does not make its result exact; the optimizer can exploit this
cloud's errors. That result sets q's center only. It never changes the target
prior or an H likelihood used in the final importance weights.

## Proposed final independent integration

Use .2 prior-box uniform plus .8 untruncated Gaussian, split over the existing
two H-only engineering profile covariances, now centered at the locally
selected beta. The scaled covariance remains `(D Iprofile D+75I)^(-1)`.
Save complete centers, covariance matrices and exact q density. Samples outside
the prior remain at zero target weight and are not redrawn.

Use fresh independent theta and latent streams for final IS; suggested frozen
seeds are 25101301–25101303. Retain B=2048/mode,L=32/time, three repeats, and
the same ESS/maximum-weight criteria. None of the old v1 particle weights enter
the final estimator. Then, only if useful and root authorizes, double latent
L to 64 with paired theta streams to isolate latent integration sensitivity.
The optimization cloud must not be reused as a final likelihood cloud.

Report all conditional mode ESS, global ESS, maximum weight, supported-theta
counts, per-time MC relative SE, repeated evidence spread and single-time
marginal evidence variation. Fail again if mode ESS remains insufficient or
repeats disagree; do not accept a posterior merely because proposal centers
now sit inside the prior box. Correlated profile covariance may still miss the
proper-prior posterior geometry, and raw Gaussian metering proposals may
still need voltage-informed tilting with its exact density accounted for.
Such a further change would need a new declared plan and fresh integration.
