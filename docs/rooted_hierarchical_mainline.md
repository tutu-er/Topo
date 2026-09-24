# Rooted hierarchical topology-identification mainline

## Aggregation object

The aggregation unit is no longer an unrooted terminal set. It is a `PseudoCluster` representing a reliable downstream rooted clade:

- `pseudo_id`: temporary pseudo-boundary label;
- `members`: observed terminals downstream of the boundary;
- `confidence`: bootstrap frequency of the rooted clade;
- `frozen_clades`: reliable local rooted clades that must survive expansion;
- `frozen_sibling_pairs`: reliable terminal sibling relations that must survive expansion.

Candidate clades come from root-constrained RNJ. A clade is eligible only when its bootstrap confidence exceeds 0.9, its size is 2--5, and it does not overlap another selected clade. Large peripheral clades are selected first; stable nested local clades and sibling pairs are retained as frozen structures.

## Pseudo measurements

For cluster C,

\[
P_C(t)=\sum_{i\in C}P_i(t),\qquad
Q_C(t)=\sum_{i\in C}Q_i(t).
\]

The former voltage proxy was the arithmetic mean of terminal squared voltages. The revised proxy estimates a boundary sensitivity row from descendant R/X rows. For terminal i,

\[
\widehat v_C^{(i)}(t)
=v_i(t)+\bigl(\widehat R_i-\widehat R_C\bigr)P(t)
          +\bigl(\widehat X_i-\widehat X_C\bigr)Q(t).
\]

The terminal estimates are combined by a median. Because estimated R/X errors can make full deembedding noisier than the original mean, the implemented proxy uses shrinkage:

\[
\widehat v_C^{\text{shrink}}
=\widehat v_C^{\text{mean}}
+\lambda\left(widehat v_C^{\text{deembed}}-widehat v_C^{\text{mean}}\right).
\]

The tested fixed value is lambda=0.25. It is an empirical case-bank result, not yet a label-free optimum.

## Reconstruction

1. Fit terminal R/X and run root-constrained RNJ.
2. Perturb already noisy measurements and estimate rooted-clade and sibling stability.
3. Freeze reliable peripheral clades and sibling pairs.
4. Contract selected clades to pseudo boundary nodes.
5. Build pseudo P/Q/V with voltage shrinkage.
6. Fit pseudo R/X and run RNJ only on the reduced rooted backbone.
7. Expand pseudo clades to terminal sets and reinsert every frozen local structure unchanged.

The second RNJ stage cannot alter a frozen peripheral clade or sibling pair.

## Results

The large experiment contains 24 operating/data conditions and five voltage proxies, giving 120 hierarchy results. It selected 78 peripheral clades; all 78 are true physical rooted clades under the simulation reference.

| voltage proxy | full clade F1 | remaining internal clade F1 | delta from terminal RNJ | frozen clade preservation | frozen sibling preservation |
|---|---:|---:|---:|---:|---:|
| mean squared voltage | 0.9472 | 0.8992 | +0.0230 | 1.0 | 1.0 |
| 25% deembedding shrinkage | 0.9513 | 0.9265 | +0.0271 | 1.0 | 1.0 |
| 50% shrinkage | 0.9356 | 0.9028 | +0.0114 | 1.0 | 1.0 |
| full deembedding | 0.9158 | 0.8352 | -0.0084 | 1.0 | 1.0 |

Full deembedding is worse because terminal R/X row errors are amplified by subtraction. A small correction improves the boundary signal while controlling variance.

For 1 scenario and 96 samples, the remaining internal-clade F1 is 0.522; for 1 scenario and 288 samples it is 1.0. Once stable peripheral structures are frozen, the residual low-data failures are mainly on the reduced internal backbone. This is consistent with the earlier visual observation that high-confidence endpoint clusters are reliable, while distinguishing it from the stronger claim that every terminal attachment is identifiable.


## One-layer pseudo plus local re-identification

A non-recursive local variant is now implemented. Cluster selection still
contracts only one set of disjoint peripheral clades. After pseudo-backbone
RNJ, each selected cluster is reconstructed exactly once; no second pseudo
layer is created inside a cluster.

Two local methods are available:

- voltage_refit: use the estimated pseudo-boundary voltage as the local
  observed root and fit a new local R/X model;
- distance_reroot: retain the full-data terminal distance estimate, remove
  the estimated common path above the cluster boundary, and rerun RNJ with the
  pseudo node as the local root.

The second method is more stable because pseudo-voltage error is not passed
through another sensitivity regression. The selected cross-case local
tolerance factor is 0.56. It was selected from the current case bank and is
not a label-free universal constant.

For 60 independent noisy 288-point days, the frozen baseline has mean clade
F1 0.9898 and exact recovery 0.8833. Local distance rerooting has mean F1
0.9690 and exact recovery 0.7500. For the 48-condition data-volume sweep, the
corresponding mean F1 values are 0.9073 and 0.9021. Local rerooting improves
3 conditions, is unchanged in 41, and degrades 4. In particular, it raises
the mean F1 to 1.0 for 3 scenarios x 96 samples and 3 scenarios x 288 samples,
but it is less robust on several single-day Pengwah cases.

Consequently this local stage is retained as an explicit experimental option,
not as an unconditional replacement for frozen high-confidence local clades.
The command is:

    python -m experiments.run_pseudo_local_reidentification --local-method distance_reroot --local-tolerance-factor 0.56

The pseudo-voltage regression comparison remains available with
--local-method voltage_refit.
