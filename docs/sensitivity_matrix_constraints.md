# Reduced sensitivity matrix constraints

## 1. Radial LinDistFlow matrix structure

For all non-root buses of a radial feeder, the squared-voltage sensitivities
can be written as

$$
R=2A^{-1}\operatorname{diag}(r)A^{-\mathsf T},\qquad
X=2A^{-1}\operatorname{diag}(x)A^{-\mathsf T},
$$

where $A$ is the reduced branch-bus incidence matrix oriented from the root.
Thus $R/2$ and $X/2$ are inverse grounded weighted Laplacians. For an observed
terminal subset $L$, the measured matrix is the principal block $R_{LL}$.
Its inverse is a Schur complement of the full grounded Laplacian, hence a
Kron-reduced grounded Laplacian.

Under positive-edge LinDistFlow, the exact properties are:

1. $R=R^\mathsf T$ and $X=X^\mathsf T$.
2. All entries are nonnegative shared-path impedances.
3. The matrices are positive definite when terminal service paths have
   positive impedance. Zero edges produce repeated or semidefinite structure.
4. For every $i\ne j$,
   $R_{ii}>R_{ij}$ and $R_{jj}>R_{ij}$, with analogous inequalities for $X$.
   Equality indicates a zero terminal segment or non-minimal representation.
5. The derived distance
   $d_{ij}=R_{ii}+R_{jj}-2R_{ij}$ is positive.
6. $R_{LL}^{-1}$ and $X_{LL}^{-1}$ are symmetric grounded M-matrices:
   positive definite, with nonpositive off-diagonals and nonnegative row sums.
7. Exact terminal distances satisfy the four-point tree-metric condition.

Items 1--5 are direct matrix constraints. Item 6 is a stronger inverse-domain
condition. Item 7 is nonconvex because all tree metrics form a union of
topology-specific cones.

## 2. Treatment in the literature

### Pengwah et al. 2024

Pengwah et al., Topology Identification of Distribution Networks With Partial
Smart Meter Coverage, formulate impedance estimation as constrained least
squares. Equations (9)--(11) explicitly impose

$$
R^{SS}=(R^{SS})^\mathsf T,\qquad
X^{SS}=(X^{SS})^\mathsf T,
$$

and $R_{kk}\ge R_{kl}$ for all relevant customer pairs, with the analogous
constraint on $X$. Their reason is the same shared-path interpretation: the
common path of two customers cannot exceed either customer's root path. The
estimated resistance matrix is then converted to
$D_{kl}=R_{kk}+R_{ll}-2R_{kl}$ before recursive grouping.
DOI: https://doi.org/10.1109/TPWRD.2024.3354292

### Flynn et al. 2023

Flynn et al. estimate the sensitivity matrix jointly with time-varying
transformer voltage. Equations (5)--(6) retain the physical constraints of
the earlier constrained estimator and add regularization on transformer
voltage variation. Their topology stage also rejects direct meter-to-meter
physical edges and forces updated distances to remain positive.
DOI: https://doi.org/10.1109/TSG.2023.3239650

### Park, Deka, and Chertkov

The minimal-observability work proves that terminal voltage/injection data
identify a reduced radial graph when all leaves are observed and hidden-node
conditions hold. The inverse reduced-Laplacian representation explains
positive definiteness and why degree-2 or zero-edge structures cannot be
separately identified. Source: https://arxiv.org/abs/1710.10727

### Broader structural and statistical methods

- The topology-learning tutorial by Deka, Kekatos, and Cavraro distinguishes
  least squares, convex programs, graph searches, and mixed-integer radiality
  constraints. Matrix inequalities are an outer physical relaxation, not an
  exact hidden-tree constraint. Source: https://arxiv.org/abs/2206.10837
- Laurent, Brouillon, and Ferrari-Trecate formulate a maximum-likelihood /
  weighted total least-squares estimator with noise on both measured sides.
  This addresses errors-in-variables bias that projection cannot remove.
  Source: https://arxiv.org/abs/2210.02217
- Grid Topology Identification With Hidden Nodes via Structured Norm
  Minimization decomposes observed inverse covariance into sparse, low-rank,
  and correction components. This is relevant when hidden injections violate
  strict zero injection.
  Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC9223390/
- Talkington et al. show localized block structure in secondary-network
  sensitivity matrices and use eigenvalue-inclusion analysis to quantify
  block dominance. DOI: https://doi.org/10.1016/j.epsr.2023.109788
- Liu et al. decompose sensitivity into a topology-related principal
  component and a small secondary correction, estimated through two quadratic
  programs. This is useful when exact LinDistFlow constraints conflict with
  AC model error.
  Source: https://xuebao.sjtu.edu.cn/CN/abstract/abstract46834.shtml

## 3. Implemented constraint levels

Implementation: terminal_case33/estimation/matrix_constraints.py

### basic

The historical projection is

$$
M\leftarrow[(M+M^\mathsf T)/2]_+.
$$

It can still produce negative distances because an off-diagonal element may
exceed its diagonal.

### ordered, current default

This mode enforces

$$
M=M^\mathsf T,\quad M\ge0,\quad
M_{ii}-M_{ij}\ge\epsilon,\quad
M_{jj}-M_{ij}\ge\epsilon.
$$

The default margin is scale relative:
$\epsilon=10^{-6}\operatorname{median}(\operatorname{diag}M)$. A zero,
unidentifiable matrix remains zero rather than receiving artificial diagonal
values.

After the first feasible projection, projected-gradient iterations minimize
the original multi-scenario regression objective. Scenario intercepts are
recomputed exactly after every step, and backtracking prevents objective
increase.

### tree_covariance

This mode additionally imposes

$$
M\succeq\epsilon_\lambda I
$$

by eigenvalue projection. It guarantees nonnegative distances and removes
negative eigenvalues, but it is a stronger outer relaxation and may increase
topology error when AC model mismatch dominates.

### Diagnostics

sensitivity_matrix_diagnostics reports symmetry error, minimum entry,
minimum eigenvalue, minimum diagonal gap, minimum derived distance, positive
off-diagonal fraction in the pseudoinverse, maximum positive precision
off-diagonal, and minimum precision row sum.

four_point_violation_summary reports mean, maximum, and 95th-percentile
violations of the additive-tree four-point condition.

## 4. Fixed ablation result

The experiment contains 16 noisy conditions over paper15, soumalas11,
flynn16, and pengwah18, with one or three scenarios and 96 or 288 samples.

| constraint | mean clade F1 | min F1 | diagonal gap | min eigenvalue | mean four-point violation |
|---|---:|---:|---:|---:|---:|
| basic | 0.8974 | 0.5000 | can be negative | can be negative | 0.1396 |
| ordered | 0.8630 | 0.3333 | positive | may be negative | 0.1230 |
| tree_covariance | 0.8689 | 0.5714 | positive | positive | 0.1035 |

Hard constraints improve matrix feasibility and tree-metric consistency, but
do not uniformly improve strict topology F1. The strongest mode improves
three conditions, leaves eleven unchanged, and degrades two. The main
degradation occurs in single-scenario paper15.

This demonstrates that AC-to-linear model error and poor excitation cannot be
fixed by matrix feasibility alone. A nonphysical basic estimate can
occasionally produce a correct RNJ ordering through compensating errors.
Every experiment should therefore report both topology accuracy and matrix
feasibility.

## 5. Recommended refinement

1. Use ordered when physical matrix validity is mandatory.
2. During research evaluation, retain basic, ordered, and tree_covariance as
   explicit candidates instead of hiding the tradeoff.
3. Select candidates without topology labels using blocked holdout voltage
   error plus bootstrap distance stability, not training residual alone.
4. After RNJ selects a topology, refit nonnegative edge weights on that fixed
   tree. This enforces the exact tree cone rather than only an outer matrix
   approximation.
5. Replace ordinary least squares by weighted TLS/MLE when meter variances
   are available.
6. For three-phase feeders, use block sensitivity and phase-coupled block
   symmetry/PSD constraints.

Run:

    python -m experiments.run_matrix_constraint_ablation

Outputs are written to outputs/matrix_constraint_ablation/.
## 6. Large noisy-AC sweep

The larger test uses four feeders, 48/96/288 samples per scenario, one, three,
five, or ten coupled scenarios, and two independent profile/noise replicates.
Every voltage trace is generated by the AC power-flow solver. Measurement
noise is always present: P/Q noise is 0.5% of the instantaneous local
magnitude and voltage noise is 0.02% of the instantaneous terminal voltage
magnitude. The test contains 96 data conditions, 288 constrained fits, and
2,592 RNJ reconstructions over three distance modes and three tolerances.

The primary normal regime is defined before topology scoring as at least three
coupled scenarios. Single-scenario conditions are retained as a separate
stress test rather than being deleted after observing F1. For fixed
RX_75R_25X distance and tolerance factor 0.16, the normal-regime results are:

| constraint | mean clade F1 | minimum F1 | exact recovery | mean sibling F1 |
|---|---:|---:|---:|---:|
| basic | 0.9681 | 0.8000 | 69.4% | 0.9694 |
| ordered | 0.9681 | 0.7692 | 72.2% | 0.9734 |
| tree_covariance | 0.9638 | 0.7143 | 68.1% | 0.9649 |

In the normal regime, relative R/X errors fall from 0.233/0.714 for basic to
0.224/0.670 for ordered and 0.215/0.647 for tree_covariance. Mean four-point
violation falls from 0.0895 to 0.0842 and 0.0752. With 288 samples and at least
three scenarios, all modes reach mean F1 0.9972 and 95.8% exact recovery.

The separate single-scenario stress regime has mean F1 0.659, 0.636, and 0.654
for basic, ordered, and tree_covariance. These values are not included in the
primary normal metric. This split is based on data availability, not on the
observed output score, so it does not introduce outcome-dependent filtering.

Run:

    python -m experiments.run_matrix_constraint_large_sweep

Outputs are written to outputs/matrix_constraint_large_sweep/.
## 7. Fair literature-inspired comparison

The comparison uses the same noisy AC-generated P/Q/V samples, terminal set,
observed root voltage, daily demeaning, scenario counts, sample counts, and
random replicates for every method. The common primary metric is unrooted
terminal-split F1, because it is defined for NJ, RG, and RNJ alike. Root
partition accuracy and rooted-clade F1 are reported separately.

Normal-regime results over 72 matched data conditions are:

| method | mean split F1 | minimum F1 | exact split recovery | root partition success |
|---|---:|---:|---:|---:|
| Flynn 2023 root-corrected proxy | 0.9672 | 0.8000 | 69.4% | 95.8% |
| proposed ordered RNJ | 0.9661 | 0.7273 | 72.2% | 100.0% |
| proposed PSD RNJ | 0.9621 | 0.6667 | 68.1% | 100.0% |
| controlled ordered RG | 0.7417 | 0.2857 | 4.2% | 1.4% |
| Pengwah 2024 ordered-RG proxy | 0.6554 | 0.1818 | 1.4% | 0.0% |
| Park/Deka RG baseline | 0.6541 | 0.1538 | 0.0% | 1.4% |
| controlled ordered NJ | 0.6298 | 0.5556 | 0.0% | 25.0% |
| Soumalas 2017 NJ proxy | 0.6037 | 0.2222 | 0.0% | 25.0% |

The word proxy is essential. The project does not claim an exact reproduction
of Soumalas' integer Pruefer procedure or Flynn's full iterative graph-learning
implementation. The controlled NJ/RG/RNJ rows isolate reconstruction behavior
using the same ordered matrix and RX distance. RG receives the known root and
root-terminal depths directly; NJ uses the root depths for post-reconstruction
root placement because classical NJ treats all input labels as leaves.

Run:

    python -m experiments.run_fair_literature_comparison

Outputs are written to outputs/fair_literature_comparison/.
## 8. Current-method line error anatomy

The ordered RX75 RNJ method was examined over the 72 normal conditions. Twenty
conditions were not exact, although their conditional mean clade F1 remained
0.8852. Physical hidden-node labels are algorithm-local, so each identifiable
physical line is compared through its downstream rooted terminal clade.

Across all conditions there were only three missing true clade-lines but 26
extra predicted clade-lines. Every extra line was a proper subclade of an
already-correct true clade: the dominant failure is spurious refinement, or an
extra hidden branching level, rather than loss of the main feeder backbone.
All 216 root-adjacent backbone opportunities were recovered. Peripheral
cluster feeders were recovered in 195 of 198 opportunities. The three misses
all occurred in flynn16: edge 3--6 feeding terminals 207/208/209 once, and edge
2--4 feeding terminals 201/202/203 twice.

Terminal sibling-neighborhood accuracy over all normal conditions is 93.33%,
and complete rooted path-signature accuracy is 93.43%. Pengwah18 has the lowest
terminal sibling accuracy, 89.51%, while paper15 reaches 97.41%. The present
case bank has root-adjacent and one-level peripheral hidden feeders but no deep
multi-level intermediate hidden backbone. Therefore the result supports
backbone preservation on the current cases only; deeper feeders require an
additional case-bank extension.

Run:

    python -m experiments.run_current_method_line_error_analysis

Outputs are written to outputs/current_method_line_error_analysis/.
## 9. Did RNJ move errors from the center to the edge?

A controlled comparison gives NJ, RG, and RNJ the same ordered R/X estimate and
the same RX75 distance over the 72 normal conditions. RG and RNJ receive the
known root depths; classical NJ remains unrooted. Physical lines are compared
through canonical terminal splits.

| method | mean split F1 | exact recovery | missing true splits | extra predicted splits |
|---|---:|---:|---:|---:|
| NJ | 0.6298 | 0.0% | 1 | 469 |
| RG | 0.7417 | 4.2% | 15 | 268 |
| RNJ | 0.9661 | 72.2% | 3 | 26 |

RNJ recovers all 198 central root-backbone split opportunities. RG misses four
central splits, while NJ misses none but introduces hundreds of extra binary
splits. Peripheral true-split miss rates are 0.5%, 5.6%, and 1.5% for NJ, RG,
and RNJ. Hence RNJ does not improve the center by sacrificing the edge; it
reduces both error classes. The apparent shift occurs because central and
large-split errors are suppressed so strongly that the residual errors are
mostly small peripheral refinements. Of RNJ's 26 extra splits, 20 contain at
most three terminals and only six contain four or more. NJ and RG produce 339
and 182 such small extra splits, respectively.

The earlier statement that errors were mainly internal used a coarse metric:
every extra latent split was called an internal-edge error even when it split a
small terminal cluster. The physical-region analysis distinguishes true
root-backbone loss from over-resolving a terminal clade. Under this definition,
current RNJ backbone recovery is strong and residual errors are predominantly
peripheral over-segmentation.

Run:

    python -m experiments.run_reconstructor_error_region_comparison

Outputs are written to outputs/reconstructor_error_region_comparison/.

## 10. One-layer latent-edge detection before RNJ

A terminal sibling pair is not necessarily an edge-induced peripheral clade:
its common parent may also feed another hidden subtree. Contracting such a pair
duplicates one physical boundary and invalidates the pseudo-node model. The
first layer therefore uses complete rooted clades rather than nearest-terminal
pairs.

For NJ, the unrooted latent tree is rooted by minimizing the mismatch between
its terminal path lengths and the measured root-to-terminal depths. For RG, the
known root is included directly. A candidate clade C is retained only if its
bootstrap frequency is at least 0.875 and

    margin(C) = [q_0.2{s_ij: i,j in C} - max_{k notin C} median_{i in C}s_ik]
                / median_i(s_ii) >= 0.10,

where s_ij is the estimated shared-path depth. The margin rejects stable
subgroups that do not have a separate upstream edge.

On 16 noisy AC conditions (four cases, 3/5 scenarios, 48/96 samples), baseline
RNJ has mean clade F1 0.9450 and exact recovery 50.0%. RG-clade contraction
reaches F1 0.9566 and exact recovery 62.5%; NJ-clade contraction reaches F1
0.9526 and exact recovery 68.75%; consensus reaches F1 0.9525 and exact recovery
62.5% with 100% selected-clade precision. The gain is real but modest and
case-dependent: soumalas11 improves most, paper15 is unchanged, and pengwah18
remains the difficult case.

Run:

    python -m experiments.run_edge_detector_then_rnj

Outputs are written to outputs/edge_detector_then_rnj/.

## 11. Probabilistic edge persistence and one-level root decoupling

Raw hidden-node labels are not comparable across NJ/RG/RNJ bootstrap samples.
The stable object is therefore an identifiable rooted terminal clade C induced
by a physical hidden edge. Relative to the topology fitted from the original
noisy measurements, the empirical persistence probability is

    p_hat_m(C) = B^(-1) sum_b I{C belongs to T_m^(b)},

where m is NJ, RG, or RNJ and T_m^(b) is reconstructed after adding a second
controlled measurement perturbation. The procedure deliberately does not call
p_hat a Bayesian posterior. Matrix-tree marginals would require a fixed
candidate graph with persistent hidden-node labels; a terminal-only
matrix-tree distribution instead describes an equivalent terminal tree, not
the latent physical clades evaluated here.

The root-decoupling step uses the estimated shared paths S. A common root mode

    c_hat = q_0.10({S_ij: i < j})

is removed by S' = max(S-c_hat,0) and h' = max(diag(S)-c_hat,0). RNJ is rerun
with the physical root, producing a first-level branch partition. Each branch
is reconstructed independently with NJ, RG, or RNJ. The RNJ grouping tolerance
is kept in absolute impedance units:

    tau_local = tau_global,
    factor_local = tau_global / median(h_local).

Rescaling the tolerance by local depth without this correction creates false
extra hidden levels.

The final sweep uses noisy AC-generated measurements with P/Q noise 0.5% and
voltage noise 0.02%, four cases, scenario counts 1/3/5, 24/48/96 samples, two
data replicates, and 16 perturbation replicates, for 72 conditions. Normal
multiscenario results are:

| method | mean rooted-clade F1 | exact recovery |
|---|---:|---:|
| baseline RNJ | 0.8837 | 41.67% |
| root-decoupled RNJ | 0.8930 | 45.83% |
| baseline RG | 0.6105 | 0% |
| root-decoupled RG | 0.7185 | 0% |
| baseline NJ | 0.5849 | 0% |
| root-decoupled NJ | 0.6230 | 0% |

At persistence threshold 0.75, normal-regime RNJ stable edges have precision
0.9491 and recall 0.9115. Threshold 0.875 increases precision to 0.9555 but
reduces recall to 0.8713. Root-edge decoupling reduces the overall root-edge
Brier score from 0.0969 to 0.0700 and the mean probability assigned to false
root edges from 0.4111 to 0.2478. It improves the root layer and suppresses
common-mode artifacts, but it cannot repair the poor excitation of a single
scenario; those conditions remain a stress test.

Run:

    python -m experiments.run_probabilistic_root_decoupling

Outputs are written to outputs/probabilistic_root_decoupling/.

## 12. Physical pseudo modes versus latent common-mode fitting

### 12.1 Physical pseudo P/Q/V

For an estimated first-level branch C_g, the hard aggregation is

    P_g(t) = sum_{i in C_g} P_i(t),
    Q_g(t) = sum_{i in C_g} Q_i(t).

For child i, the branch-boundary squared voltage is back-calculated as

    v_hat_g^(i)(t) = v_i(t)
                     + Delta R_i,g P(t)
                     + Delta X_i,g Q(t).

The robust deembedded estimate is the median over children. To account for
sensitivity error, it is shrunk toward the direct mean terminal squared
voltage:

    v_hat_g(t;lambda)
      = mean_{i in C_g} v_i(t)
        + lambda [median_i v_hat_g^(i)(t)
                  - mean_{i in C_g} v_i(t)],

with lambda in {0,0.25,0.50,1}. The test also includes an AC-truth pseudo
voltage as a nondeployable upper-bound diagnostic.

This construction produces one distinct P_g/Q_g/v_g time series per estimated
root branch. Each v_g is used as the local root voltage when the branch
sensitivity and hidden topology are refitted.

### 12.2 Common mode as a fitted variable

Hard pseudo voltage is avoided by treating the branch common root drop m_g,s(t)
as a variable:

    min_{R_g,X_g,{m_g,s}}
      sum_s ||Y_g,s - P_g,s R_g^T - Q_g,s X_g^T
                    - m_g,s 1^T||_F^2
      + lambda_prior sum_s ||m_g,s - m_tilde_g,s||_2^2
      + lambda_time sum_s ||D m_g,s||_2^2,

subject to symmetric, nonnegative, ordered R_g and X_g. Here Y_g,s contains
root-to-terminal squared-voltage drops, m_tilde is the physical pseudo-voltage
prior, and D is the first-difference matrix.

For fixed R/X, the mode update has the closed form

    [n_g I + lambda_prior I + lambda_time D^T D] m_g,s
      = sum_i residual_i + lambda_prior m_tilde_g,s.

R/X and common modes are alternated. The cross-terminal residual dispersion
provides a mode standard-error diagnostic. With lambda_prior=0, the method is
equivalent to removing an estimated branch-specific common voltage mode before
local sensitivity recovery. A nonzero prior trades topology fit against
physical voltage-mode accuracy.

### 12.3 AC benchmark

The final test has four cases, scenario counts 1/3/5, 24/48/96 samples, two
data replicates, P/Q noise 0.5%, voltage noise 0.02%, and AC-generated PQV.

Normal multiscenario topology results are:

| method | mean F1 | exact recovery |
|---|---:|---:|
| baseline RNJ | 0.8837 | 41.67% |
| latent free smooth mode | 0.9457 | 43.75% |
| latent pseudo prior 0.25 | 0.9409 | 37.50% |
| latent pseudo prior 1 | 0.9198 | 31.25% |
| physical local shrink 0.25 | 0.8158 | 18.75% |
| physical local AC-truth voltage | 0.8847 | 43.75% |

The operational pseudo-voltage errors are:

| voltage construction | magnitude RMSE (p.u.) | squared-voltage RMSE |
|---|---:|---:|
| mean terminal voltage | 0.01029 | 0.02014 |
| shrink 0.25 | 0.00525 | 0.01030 |
| shrink 0.50 | 0.00314 | 0.00621 |
| full deembedding | 0.01088 | 0.02169 |

Aggregate pseudo P and Q have relative RMSE 3.67% and 6.60% against true AC
root-branch flow. This is larger than meter noise because simple summation
omits branch losses.

A stronger pseudo prior improves voltage-mode truth RMSE from 0.00366 for the
free mode to 0.00128 at prior weight 5, but reduces topology F1 to 0.8947.
Therefore the best topology result uses the smooth free common mode. The
physical pseudo mode is useful as a weak prior and diagnostic, not as a hard
replacement for the branch boundary voltage.

The single-scenario latent fit reaches F1 0.7702 versus baseline 0.3803, but its
median design condition number is about 2.5e4 and exact recovery remains only
20.83%. It should remain a stress result rather than the headline metric.

Run:

    python -m experiments.run_layered_voltage_mode_comparison

Outputs are written to outputs/layered_voltage_mode_comparison/.

## 13. Meaning of the prior and the pure-free ablation

The pseudo-voltage prior is the operational estimate

    m_tilde_g(t) = v_0(t)^2 - v_hat_g(t)^2,

obtained from terminal P/Q/V and estimated sensitivity. It is not AC truth and
does not reveal topology labels. The penalty lambda_prior ||m_g-m_tilde_g||^2
only discourages the latent common mode from moving far from this noisy
physical proxy.

The pure-free test sets

    lambda_prior = 0,
    lambda_time = 0,
    alpha_RX = 0.

After increasing alternating iterations from 8 to 20, its normal-regime result
over 48 conditions is:

| method | mean F1 | exact recovery | precision | recall |
|---|---:|---:|---:|---:|
| baseline RNJ | 0.8837 | 41.67% | 0.8329 | 0.9531 |
| free smooth mode | 0.9457 | 43.75% | 0.9042 | 0.9965 |
| pure free unregularized | 0.9275 | 58.33% | 0.8947 | 0.9688 |

Pure free has the highest exact-recovery rate but lower mean F1 than smooth
free. Its failures are less frequent and more severe. In single-scenario
stress it reaches F1 0.5055, compared with 0.7702 for smooth free.

Without a prior or temporal constraint, the decomposition is not unique. For
compatible vectors a and b,

    R'_g = R_g + 1 a^T,
    X'_g = X_g + 1 b^T,
    m'_g(t) = m_g(t) - P_g(t)a - Q_g(t)b

can preserve the same voltage prediction. Symmetry and ordered projection
reduce but do not generally remove this ambiguity. Consequently, pure-free
topology can be useful while its estimated R/X and common mode are not yet
physically interpretable.

On the exact same 48 conditions, the literature-inspired proxy comparison has
mean rooted-clade F1 0.8919 for tree-covariance RNJ, 0.8837 for ordered RNJ,
and 0.8820 for the Flynn root-corrected proxy. Both free-mode variants improve
these project-local proxies. This is not evidence that they outperform the
original publications, whose case construction, observability, metrics, and
algorithm details differ.

The base projected regression minimizes the squared residual

    ||Y - A B||_F^2 + alpha ||B||_F^2

under matrix constraints. This is equivalent to introducing an implicit error
E = Y-AB with a quadratic ||E||_F^2 penalty. There is currently no separately
parameterized delta V variable, no temporal regularizer on delta V, and no
errors-in-variables correction for noisy P/Q. Adding only an explicit delta V
with equality Y=AB+delta V and penalty lambda||delta V||^2 is algebraically
the same least-squares model. A materially different error model requires
GLS/covariance weighting, Huber or L1 residuals, structured temporal residuals,
or TLS/IV treatment of P/Q measurement error.
