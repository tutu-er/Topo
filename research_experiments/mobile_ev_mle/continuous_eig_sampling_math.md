# Proper continuous EIG sampler: declared model and integration contract

This is an independent diagnostic of the frozen nominal synthetic population.
It does not modify C/D/stress, a deployed selector, a threshold, or a formal
evaluation. The mathematical statements below are self-derived. They are not
claims about a field prior or topology correctness in service.

## 1. One proper prior, before H

The four reduced three-terminal topologies have probability 1/4 each. Star has
one impedance component: its weak/strong label has no electrical effect and is
integrated out, not duplicated. Every cherry has weak and strong components
with conditional probabilities 1/2. Thus seven components have masses
`[.25, .125, .125, .125, .125, .125, .125]` when grouped by topology.

For every canonical edge, R and X independently have uniform priors on
`[.8 R0, 1.2 R0]` and `[.8 X0, 1.2 X0]`. Full-support R0/X0 are .005/.0035;
terminal i has R0=.028+.003i and X0=.011+.002i. A weak cherry edge has
.00003/.00002 and a strong cherry edge .012/.007. Edge ordering is the actual
candidate canonical ordering. All parameters and powers are pu.

At each time, independently of beta and other times,

    c ~ U[-.012,.012]; ePi,eQi iid ~ U[-.004,.004],
    bPi=.12+c+ePi; bQi=.04+.25c+eQi,
    sourceP ~ U[.065,.135]; sourceQ ~ U[.02,.05].

Bias/root are fixed, known calibrated quantities; EV tracking is one. Actual
terminal totals are `P=bP+uP,Q=bQ+uQ`; the known EV command is included once.
Independent Gaussian meter standard deviations are P=.0006,Q=.0004,
V=.00012,masterP=masterQ=.001, independent across channel and time.

The shared c can be integrated exactly: let

    lower=max(-.012, max_i(Pbase_i-.12-.004),
                        max_i(4*(Qbase_i-.04-.004))),
    upper=min( .012, min_i(Pbase_i-.12+.004),
                        min_i(4*(Qbase_i-.04+.004))).

The six-dimensional background density is
`max(upper-lower,0)/(.024*.008**6)`. It is not the product of independent
terminal marginal densities. The source density is the indicator of its
rectangle divided by `.07*.03`.

## 2. Change of variables and cancellation

For fixed component, beta and root, the physical forward model maps
`(P,Q,sP,sQ)` to terminal voltage magnitudes and master actual powers
`S0=(masterP,masterQ)`. On the declared high-voltage domain, the inverse agent
must establish a one-to-one differentiable mapping from `(P,Q,S0)` to source
and voltage, and return `J=abs(det d(sP,sQ)/d(masterP,masterQ))`.
Acceptance of that proof is a prerequisite for calling this the complete model
integral. A successful pointwise AC solve alone is insufficient.

For observations yt and known action ut, the likelihood at fixed beta is

    ft(beta,ut) = integral qG(P,Q,S0|yt) *
       pV(yt.V | V(beta,P,Q,S0)) *
       pbackground(P-ut.P,Q-ut.Q) * psource(s(beta,P,Q,S0)) * J
       dP dQ dS0.

Here `qG` is the product of the eight Gaussian P/Q/master meter densities,
centered at calibrated observations. It is also the proposal, so those eight
meter densities cancel exactly. Their constants are not accidentally dropped:
the qG expectation carries the correct normalizations. Voltage densities,
background/source densities, and J retain all constants.

Draw independent proposal clouds at different times and use

    fhat(beta,H)=product_t mean_l K(beta,t,l).

Each time mean is unbiased; independence across t makes the product unbiased
for the joint likelihood. Sharing a cloud across beta or actions is valid CRN,
but sharing the same random cloud across times generally destroys this
unbiasedness. Implement in log space with logsumexp, preserving exact zeros.
Gaussian points outside the supported inverse/prior domain contribute zero;
numerical engine failures inside the declared supported domain are failures,
not legitimate exclusions.

## 3. H-only importance proposal, never a prior

Within component k, use the frozen proposal

    qk(beta)=a*Uniform(prior box)+(1-a)*product_j TN(mu_j,sd_j,box_j).

The center may be derived from H-only fit summaries, then clipped into the
component prior box. The default sd is .08 times the component canonical
parameter and a=.2. Both the normal truncation constants and the full mixture
density must be included. Uniform mass provides full support. The exact weight
is `priorpdf(beta)/qk(beta)`; a fitted MLE is never assigned prior probability.

For Bk independent beta samples, component evidence is estimated by

    Zhat_k=mean_b [priorpdf(beta_b)/qk(beta_b)*fhat_H(beta_b)].

The unnormalized particle weight is
`mode_mass_k/Bk * priorpdf/qk * fhat_H`. Normalize once over all seven modes.
H enters once; no extra WZZT likelihood is added. Failed/zero-weight components
remain in the ledger. Zero mode evidence is a numerical approximation failure
when it cannot be distinguished from proposal undercoverage, not an exact
rejection of that model.

The random fhat_H yields a self-normalized auxiliary importance approximation.
Only as both beta and latent integration grow does it approximate the exact
posterior. Component ESS, overall ESS, maximum normalized weight, per-time
latent ESS and zero counts are necessary diagnostics. A small ESS cannot be
repaired by silently dropping modes or reporting only the best topology.

## 4. Prospective EIG, shared beta and fresh hypothetical future

With the H particle bank frozen, draw component/beta from its normalized H
weights. Then draw fresh iid future background/source from the declared proper
prior, solve forward AC with `base+command`, and add fresh independent meter
noise. These are hypothetical predictive observations; actual evaluation
future/truth/noise arrays are never needed.

For each hypothetical Y and each H particle, estimate the independent future
likelihood by the same change of variables with the EV command subtracted
once from latent actual terminal totals before evaluating background density.
Let Wh be the H topology mass. Then

    p_hat_h(Y|H,u)=sum_{b:topology=h} w_b fhat_N(beta_b,u) / Wh,
    p_hat(Y|H,u)=sum_b w_b fhat_N(beta_b,u),
    EIG_hat=mean_outer log[p_hat_sampled_h(Y|H,u)/p_hat(Y|H,u)].

The topology sum includes both strength components. H and N share the same
beta particle; integrating a new independent beta for N would be a different
model. Future latent clouds are independent of the H clouds. Negative finite
sample EIG estimates are retained. An entropy-reduction estimate can be saved
as a separate diagnostic, but finite nested integration does not make it
identical to the log-ratio estimate.

Across actions, pair the same sampled topology/beta, future c/eP/eQ/source,
meter-normal noise and future latent proposal normals. Independent repeats
use independent streams. Pairing changes variance, not the model. Sampling
from the finite H importance bank introduces posterior discretization error;
log of estimated future integrals introduces nested Jensen bias. Neither is
removed by an outer standard error. Double one integration level at a time,
retain paired differences, and compare independent-repeat variability with
the differences due to doubling. Report failed if the minimum ESS or integral
stability criteria are unmet. No clipping of negative EIG is permitted.

## 5. Sensitivity and freeze

Changing only topology/strength masses reweights existing seven component
evidence and H particle weights, provided the boxes/densities remain identical.
Widening beta support changes prior and proposal integration: it requires a
new integral, not a relabeling of old particle weights.

Suggested first useful budget, requiring root authorization: H710009,
constant_fixed/fixed_mle_milp/refit_milp; B=256 per mode, latent L=16,
outer=128, three independent repeats, then double one level. H integration and
ESS are checked before predictive MC. Root freezes the actual size before any
MC run. Unit tests and small analytic mocks are correctness checks, not an
actual-H EIG pilot. Math/API/proof status and hashes are saved before MC.

## 6. Information-flow ledger

Formal implementation contracts accept sanitized inputs containing only six
H meter arrays, calibrated model quantities, H-only center beta summaries,
and already planned action P/Q. NPZ access is lazy and restricted to
history_p/q/v/master_p/master_q/root_v; no future/latent/noise keys are indexed.

During initial schema inspection, this agent mistakenly printed the first 45
lines of trial_710009.json; this exposed case/truth_topology/true_index. These
fields are excluded from the formal inputs and are not used for any proposal,
action, prior, budget, result filtering or numerical selection. H710009 and the
three actions were already specified by the parent before that inspection.
No actual future/noise NPZ field has been indexed. Therefore the formal reader
and estimator are isolated, but this work cannot claim the agent never saw
truth. This incidental inspection does not alter the original frozen C/D/stress
evaluation. Subsequent actual-H inputs must be supplied sanitized by root.

## 7. Root-frozen first actual-H diagnostic

The parent has frozen H710009, B=2048 beta proposals per component, L=32
independent latent proposals per time, and three independent H-only repeats.
Predictive MC remains disabled pending the H diagnostics. Inputs are only the
parent-sanitized `outputs/continuous_eig_Honly/H_710009.json` and `.npz`.

The actual-H proposal is .2 uniform prior box plus .8 split equally over all
available H-only fitted centers. Center k is clipped into the current mode's
prior box. In scaled phi=beta/canonical coordinates, its engineering covariance
is `(D Iprofile D + 75 I)^(-1)`, where D=diag(canonical). The Gaussian is
untruncated: outside-prior samples remain in the bank with exact zero target
weight, and are not redrawn. Its beta-coordinate density includes
`1/product(canonical)`. Profile precision controls proposal shape only and is
not a prior or confidence-region claim. The product-TN implementation remains
an independently tested baseline and is not used in this H pilot.

Declared independent H seeds are 22101301,22101302,22101303. Predeclared initial
failure thresholds are conditional ESS <20 in any of seven modes, global ESS
<100, or maximum normalized particle weight >.05. Any zero mode estimate also
fails; it is not interpreted as zero mathematical posterior mass. These
thresholds are practical initial diagnostics, not bounds on integration error.
Independent repeats, per-time sample-moment relative standard errors and
single-time marginal-evidence fluctuations are retained; the latter are not
joint H evidence because beta is shared across times.

## 8. Current sampler contract and executed result

The repaired current observation type accepts any finite noisy V reading,
including negative values, because the Gaussian measurement distribution is
untruncated. The known physical root is required to equal one. EV commands
must be single-site P in [0,.3] per time with Q=0, exactly the proof envelope.
Within supported beta, finite background density and the master rectangle,
an inverse numerical/branch/certificate/nonpositive physical-V or nonpositive
Jacobian failure raises a computational failure. It cannot be silently counted
as probability zero. Background, source, master-rectangle and parameter-domain
outside proposals remain legitimate zero target weights, with separate counts.

The executed H-only v1 failed: all global ESS values were approximately one,
all maximum weights approximately one, and evidence varied by 25.5612 in log
scale over three independent runs. No numerical zero mode was dropped. V1 had
zero invalid inverse counts, so the later contract validation changes do not
alter its stored numerical values. Its exact executed source/math snapshots
match its pre-MC hashes. No predictive MC was run. A v2 proposal-repair plan
is saved separately and requires root's execution freeze.

The current sampler's 24 independent tests cover exact mixture density and
coordinate Jacobian, proper importance normalization, time-factorized Gaussian
cancellation, common-c background usage through the inverse module, one EV
addition/subtraction, H/N shared beta, allowed NPZ field access, retained
outside-prior samples, untruncated negative noisy V, supported-domain numerical
failure, and stable topology conditional log density under display underflow.
Small analytic/mock tests verify algebra and contracts; they do not establish
adequate ESS or a usable EIG estimate for actual H.

## 9. Subsequent proposal diagnostics and current final freeze

V2 executed the separately frozen H-only bounded local optimization of
engineering means and then used fresh independent importance samples.
V3 retained those means/covariances and replaced the untruncated Gaussian
branch by the normalized sequential-conditional TN proposal. V4 retained V3
beta proposals and used the exact-density voltage-guided latent mixture.
All three retained the original proper prior/physical target and failed the
predeclared all-mode ESS gates; all source snapshots and numerical failures
or zero-target events remain archived. The separate plans give the full
normalizers, Jacobians, restoration ratio and numerical-domain contracts.

The H-only engineering pilot subsequently used the frozen schedule
kappa=.001,.003,.01,.03,.1,.3,1, fresh theta/latent samples at every stage,
and ESS-gated weighted moment fitting. Random Lhat_H^kappa is explicitly
not an exact tempered-posterior density. Its weights are never reused for
the final historical update. Failed moment fitting retains the exact last
valid normalized proposal and every mode.

The final parent freeze is seeds28301301-03, B=4096 per mode and independent
per-time L128 guided latent samples; L32 is the first32 prefix of the same
per-time kernel. At each level the final estimator independently constructs
pi(beta)/q(beta) times that level's original H likelihood once. It does not
multiply two H likelihoods or reuse pilot particle weights. Fixed beta chunks
of256 change memory use only. The original globalESS100, every-modeESS20
and maximum-weight.05 gates remain in force, including rare displayed modes.

There are56 independently run sampler/proposal/guidance/moment/prefix tests
at the final freeze. Conditional theta-weight SE is computed conditional on
the common latent cloud. Conditional weighted latent ESS/SE is a sample
diagnostic selected by random H weights, not an error bound. CRN across beta
precludes a unique unconditional beta/latent variance decomposition from one
bank; fresh repeats and paired integration doubling are retained separately.
Final outcomes and compact comparisons belong in
continuous_eig_Honly_all_five_comparison.json and
continuous_eig_Honly_results_zh.md. Predictive MC remains unexecuted.
