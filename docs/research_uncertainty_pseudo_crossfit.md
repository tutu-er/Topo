# RNJ uncertainty, joint pseudo-parent voltage, and three-stage cross-fitting

## Scope and benchmark contract

This note evaluates three extensions of the current terminal-only latent-tree
pipeline. The standard pilot uses three distinct day scenarios, 96 synchronized
15-minute samples per day, radial AC power flow at every sample, 0.5% relative
P/Q Gaussian measurement noise, 0.02% relative voltage-magnitude noise, and a
time-varying observed root voltage with 0.0008 p.u. standard deviation. No
noise-free measurement set is supplied to an estimator.

The three extensions address different failure modes:

1. uncertainty-aware RNJ addresses unstable discrete joins caused by an
   uncertain additive-distance estimate;
2. joint pseudo-parent estimation addresses bias in the voltage assigned to a
   contracted peripheral region;
3. three-stage cross-fitting addresses adaptive reuse of the same days for
   region discovery, parameter estimation, and topology selection.

They are not interchangeable, and the pilot keeps their outputs separately
ablatable.

## 1. Uncertainty-aware RNJ

For reduced sensitivity matrices, RNJ directly uses

\[
S_{ij}=c_RR_{ij}+c_XX_{ij},\qquad H_i=S_{ii}.
\]

For a generic additive-distance input this score can still be obtained from
`(H_i+H_j-d_ij)/2`, but doing so for R/X only reconstructs the original matrix
entry. Exact RNJ repeatedly joins the active pair with largest shared path. Ni and
Tatikonda prove exact recovery for additive inputs and, for general trees with
minimum logical edge length `Delta`, a sufficient robustness condition

\[
\lVert \widehat d-d\rVert_\infty < \frac{\Delta}{4}.
\]

The current code uses a fixed tolerance but does not estimate the left-hand
side. The extension obtains moving-block bootstrap replicates
`S^(1),...,S^(B)` from the already noisy days. It then:

1. chooses the next pair by the lower confidence bound of `S_ij`;
2. contracts a third active node `k` into the same multifurcation unless the
   lower confidence bound of `S_ij-min(S_ik,S_jk)` exceeds the grouping
   tolerance;
3. propagates every bootstrap replicate through contraction, rather than
   replacing uncertainty by one scalar standard error;
4. reports ordinary clade bootstrap support as a separate diagnostic.

This is a conservative engineering adaptation of RNJ robustness and tree
bootstrap ideas. It is not BIONJ: BIONJ uses an explicit covariance model to
reduce distance-update variance, whereas this implementation changes the RNJ
decision rule using empirical block uncertainty. The relevant primary sources
are [Ni and Tatikonda (2008)](https://arxiv.org/abs/0809.0158),
[Felsenstein (1985)](https://onlinelibrary.wiley.com/doi/10.1111/j.1558-5646.1985.tb00420.x),
and [Gascuel (1997)](https://academic.oup.com/mbe/article/14/7/685/1119804).

Expected benefit: fewer false positive hidden clades when competing shared
paths overlap within sampling uncertainty. Expected cost: lower recall and a
more unresolved tree when short internal edges are close to the uncertainty
radius. This makes it appropriate as an additional candidate or confidence
gate, not yet as an unconditional replacement for point RNJ.

## 2. Joint pseudo-parent squared voltage

The former contraction first estimates global R/X and then reconstructs the
pseudo-parent voltage with a fixed blend of mean terminal voltage and an
estimated service drop. That plug-in order treats the global sensitivity error
as fixed and can leak it directly into the second-level regression.

For a cluster `C`, the extension jointly fits

\[
y_i(t)=v_0^2(t)-v_i^2(t)
=g_C(t)+\sum_{j\in C}R^C_{ij}p_j(t)
+\sum_{j\in C}X^C_{ij}q_j(t)+e_i(t),
\]

where `g_C(t)=v_0^2(t)-v_C^2(t)` is a free, per-sample common mode. The
objective is

\[
\min_{R^C,X^C,g_C}
\sum_t\left\|y_C(t)-g_C(t)\mathbf 1
-R^Cp_C(t)-X^Cq_C(t)\right\|_2^2
+\lambda(\lVert R^C\rVert_F^2+\lVert X^C\rVert_F^2),
\]

subject to symmetric, nonnegative, diagonally ordered R/X matrices. Given R/X,
`g_C(t)` is the mean residual across cluster terminals. Given `g_C`, projected
gradient updates R/X. No temporal smoothing prior is imposed.

There is a necessary gauge ambiguity: adding `c 11'` to a local sensitivity
and subtracting `c sum_j p_j(t)` from `g_C(t)` leaves the fit unchanged. The
implementation anchors the 20th-percentile off-diagonal common-path component at zero,
which defines the pseudo-parent at the shallowest represented common path. The
estimated pseudo voltage is then

\[
\widehat v_C^2(t)=v_0^2(t)-\widehat g_C(t).
\]

Simultaneous sensitivity and transformer-voltage estimation has direct
precedent in [Fang et al. (2024)](https://minerva-access.unimelb.edu.au/server/api/core/bitstreams/26bfcb36-fcc3-4676-9b6d-9099b38c3e13/content).
The present implementation applies the same joint-variable principle locally
to a pseudo-parent and explicitly states the gauge that is otherwise hidden.

Expected benefit: better pseudo voltage when the initial service-drop estimate
is biased but within-cluster voltage contrasts are informative. Main risk: a
free `g_C(t)` can absorb real common injection signal; therefore cluster
membership must be credible, and held-out AC validation remains necessary.

## 3. Three-stage cross-fitting

The current two-way split prevents AC validation leakage, but topology-aware
cluster discovery and pseudo-parameter estimation can still use the same
training days. Because the discovered cluster is a discrete function of noise,
evaluating its pseudo fit on those same days is optimistic.

For three days, fold `k` assigns disjoint roles:

\[
(D_k,E_k,V_k)=(\text{discovery},\text{estimation},\text{validation}),
\]

and rotates the roles over all three days. `D_k` proposes NJ/RG peripheral
regions only. `E_k` estimates Ordered R/X, uncertainty-aware RNJ, joint
pseudo-parent voltage, and candidate trees. `V_k` is used only by fixed-tree
EIV fitting plus AC predictive likelihood. The final answer is one of the three
valid fold winners, selected as the topology medoid with maximum mean clade
Jaccard agreement; independent clade voting is not used because it could
produce a non-laminar set.

A second pilot uses disjoint contiguous two-hour blocks from every day. Each
stage then receives 96 samples spanning all three operating scenarios, while
no row appears in two stages of the same fold. This reduces day-to-day
covariate shift, but adjacent blocks remain temporally dependent, so it is a
practical small-data diagnostic rather than an independence guarantee.

Cross-fitting is established as a way to reduce overfitting from estimated
nuisance components in [Chernozhukov et al. (2018)](https://arxiv.org/abs/1608.00060).
Our use is narrower and does **not** inherit double-machine-learning inference
theory: there is no Neyman-orthogonal topology score, and the target is
discrete. It should be described as honest cyclic sample splitting for
adaptive candidate generation.

Expected benefit: a more credible estimate of generalization and less tendency
to select a pseudo cluster because it fits the same noise used to discover it.
Expected cost: with only three days, each stage receives one day, increasing
variance. More days would permit grouped folds and should improve this method.

## Standard 3x96 pilot result

Across `paper15`, `soumalas11`, `flynn16`, and `pengwah18`, mean rooted-clade
F1 was 0.9833 for Ordered RNJ, 0.9833 for joint pseudo-parent aggregation,
0.9643 for uncertainty-aware RNJ, 0.9641 for fixed de-embedding, 0.7730 for
whole-day three-stage cross-fitting, and 0.7639 for block-disjoint
cross-fitting. The baseline was already exact in three of four cases.

Joint pseudo-parent estimation improved true-boundary squared-voltage RMSE in
`paper15` (0.01898 to 0.00379) and `soumalas11` (0.00910 to 0.00658), but
worsened it in `pengwah18` (0.00287 to 0.01302). Both cross-fitting variants
produced three different fold winners in every case. These observations support
optional candidate integration for the first two extensions and reject
three-stage cross-fitting as the default at only three days.
## Integration decision

- Add uncertainty-aware RNJ to the candidate pool and expose its clade support;
  do not make it the sole reconstruction path until recall is tested more
  broadly.
- Keep joint pseudo-parent voltage as an optional aggregation mode. Promote it
  only when it improves true-boundary voltage RMSE or held-out AC score; a lower
  in-sample residual alone is insufficient.
- Keep three-stage cross-fitting as an evaluation and selection mode. For the
  standard three-day case it is deliberately strict and may underperform the
  two-way all-data flow because each fold estimates from one day.

The terminal-only identifiability limit remains unchanged: degree-2 hidden
chains are represented by one logical edge, not uniquely recovered as physical
segments. Terminal-only topology learning and hidden-node recovery assumptions
are discussed by [Deka, Backhaus, and Chertkov (2016)](https://arxiv.org/abs/1608.05031).

## Reproduction

```powershell
cd D:\0-github_workspace\Topo
python -m experiments.run_research_extensions_3x96 `
  --bootstrap-replicates 6 `
  --output-dir outputs/research_extensions_3x96
```

The command writes `method_results.csv`, `case_diagnostics.csv`,
`method_summary.csv`, and `summary.json`.
