Passive WZZT to AC evidence bridge, version 9
=============================================

Scope
-----
passive_bridge_v9.py accepts observations and proposes replaceable laminar
support families. The current constrained sensitivity estimator, RNJ, and
source-aware WZZT are actually called; this is not a hard-coded three-tree bank.
No production default, WZZT optimizer, or frozen version-8 experiment is changed.
The AC verifier consumes the resulting bank separately. Ranking and exclusion
remain separate fields; this bridge leaves every ac_evidence field null.

Data and notation
-----------------
p, q, y have shape (T,n). P and Q are total measured terminal powers in pu,
including any EV; y_ti = Vroot_t^2 - Vi_t^2 is the measured squared-voltage drop.
Known kappa = X/R is used for this bridge's effective input u=P+kappa*Q.
A support C is the set of 0-based terminal indices below one tree edge;
b_C is its 0/1 membership vector. Supports must be laminar: disjoint or nested.
Candidates return nontrivial supports only. The AC model constructor adds the
all-terminal trunk and singleton edges and determines its own edge order.

1. Passive sensitivity and RNJ proposals
----------------------------------------
The current fit_projected_sensitivity solves the symmetric, nonnegative,
diagonal-ordered squared loss problem for Y = P R + Q X. Here it is called with
P=u, Q=0 so its R represents the known-X/R effective sensitivity. The zero-Q
channel is deliberately rank deficient and its infinite augmented condition is
reported as unavailable; effective-input rank/condition are reported separately.

Two observation-only preprocessing choices are used: raw (u,Y), and temporal
centering (u-mean_t(u),Y-mean_t(Y)). Centering is only a proposal heuristic; it
does not assert that a varying physical source or AC losses have disappeared.
Each estimate is passed through current sensitivity_geometry and
rooted_neighbor_joining at tolerance factors 0, 0.08, 0.16 by default.
RNJ supports are hypotheses, not calibrated certificates.

2. Current source-aware WZZT, restarted from singletons
-----------------------------------------------------
The actual fit_source_aware_wzzt call always sets certified_supports=().
The model's sensitivities are

    R = sum_C r_C b_C b_C^T,    X = sum_C x_C b_C b_C^T,
    g_R = sum_C h_C r_C b_C,    g_X = sum_C h_C x_C b_C,
    predicted y_t = R p_t + X q_t + g_R sP_t + g_X sQ_t.

The common hidden-source path h is refitted in every accepted step. Its source
allowance is an output-wise response relaxation, not an exact shared-source
AC law. WZZT also allows separate R/X here. These distinctions are acceptable
for candidate generation; neither supplies AC exclusion evidence.

The objective is mean absolute adjusted voltage-drop residual. The algorithm
starts with singleton atoms, searches one additional compatible atom, refits
the whole accepted family, and may prune zero-weight atoms. Solver checks apply
to each fixed/one-extension domain. The forward path is greedy and does not
prove a globally optimal topology. Timeout or nonoptimal stages retain earlier
accepted proposals and diagnostics; a failed WZZT call does not remove RNJ.
The time limit is per solve, not a whole-pipeline wall-time guarantee.

3. Replaceable proposals and finite local expansion
--------------------------------------------------
prior_proposals=(family1,family2,...) can carry old, uncertified RNJ structures.
They are scored as candidates and never sent to WZZT as fixed or certified
atoms. Neighborhoods of observation-ranked seeds contain:

  - deletion of one clade;
  - rooted local interchange C -> (C\A) union D, where A is a direct child of
    C and D is a sibling of C;
  - permutation of two terminal labels throughout a family.

The local-interchange formula keeps all other clades and laminarity. It can
replace an erroneous cherry, rather than only add structures compatible with
that error. The default uses the best four base hypotheses as neighborhood
seeds; this is finite local search, not coverage of every radial topology.

4. One common proposal ranking, separate from AC evidence
-------------------------------------------------------
Let H = I - 11^T/n remove the terminal-common component at each time. For a
fixed family F, the ranking fit is bounded nonnegative least squares:

  minimize over 0 <= theta_C <= coefficient_bound
      sum_t || H y_t - sum_C theta_C H b_C b_C^T u_t ||_2^2.

The reported score is the resulting mean absolute residual. The trunk term
vanishes under H, so only the proposed nontrivial and singleton atoms are fit.
This profiles out arbitrary common responses for ranking. Noncommon AC loss
responses, actual power-input errors, and model uncertainty are not certified.
No rank, fit success, or positive score in this bridge excludes a topology.

Base candidates receive capacity priority. Remaining slots use ranked local
neighbors. generated_count, retained_count, base_count, and discarded_base_count
make truncation explicit. A discarded candidate has not been excluded. A
singleton survivor later reported by AC is conditional on the supplied bank
and the physical/error domains; no global topology probability is implied.

Interface
---------
from research_experiments.mobile_ev_probing.passive_bridge_v9 import build_passive_candidates

bank = build_passive_candidates(
    p, q, y, kappa=.5,
    source_p_nominal=.12, source_q_nominal=.036,
    source_p_error=.1, source_q_error=.03,
    max_candidates=12, wzzt_time_limit=2.,
    prior_proposals=(),
)

for row in bank['candidates']:
    candidate = RadialCandidate.from_supports(row['id'], p.shape[1], row['supports'])
    # AC evidence must now be computed from the original measurements.

No evaluation topology, true impedance, true hidden-source realization, or
future observation is accepted by build_passive_candidates.

Tests and limits
----------------
Command from repository root:
  D:\apps\miniconda3\envs\Topo\python.exe -m pytest research_experiments/mobile_ev_probing/test_passive_bridge_v9.py -q -p no:cacheprovider

14 tests passed on 2026-10-12. Tests include actual six-terminal current RNJ
recovery, actual WZZT singleton restart, correction of a wrong old cherry,
laminar local moves, input preservation, failure retention, and an independent
common-response invariance check. This is module verification, not measured
network performance or a proof that the finite proposal bank contains truth.
