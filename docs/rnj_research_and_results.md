# RNJ research and benchmark results

## Why the old and new scores differ

A terminal-equivalent minimum-distance tree connects observed terminals directly. If terminals a, b, and c share one hidden parent h, the equivalent tree must select two artificial terminal-terminal edges from ab, ac, and bc. Recovering those edges correctly shows that the terminal cluster and local distance ordering are correct, but it does not recover the physical edges a-h, b-h, c-h or the hidden parent h.

NJ/RG hidden-tree evaluation is stricter:

- terminal split metrics evaluate internal hidden branches;
- sibling metrics require terminals to share the same reconstructed hidden parent;
- terminal limb errors evaluate pendant-edge length;
- root partition metrics require the known feeder root to induce the correct main-lateral partition.

Therefore the earlier observation remains valid at the terminal-equivalent level: endpoint clusters were often identified correctly. It does not imply that the full hidden-node tree was recovered.

## RNJ formulation

For reduced R/X matrices, Rooted Neighbor-Joining now uses the direct score

\[
S_{ij}=c_RR_{ij}+c_XX_{ij},\qquad H_i=S_{ii},
\]

which is the root-to-LCA shared-path length. In the R-only mode this is exactly
`R_ij`; the X-only mode uses `X_ij`; normalized mixed modes use the same
coefficients as their distance diagnostics. Constructing a distance and then
computing a separate `rho` returns the same matrix entry and has been removed
from the sensitivity-matrix path.

The implementation repeatedly joins the active pair with the largest shared path, groups additional active nodes whose shared path differs by less than a tolerance, and retains the supplied feeder-root label. A known-root stopping constraint connects remaining top-level groups directly to the root when their maximum shared path is below the same tolerance. This prevents a small positive common-mode bias in estimated R/X from creating a false trunk above every feeder lateral.

Primary references:

- J. Ni, H. Xie, S. Tatikonda, and Y. R. Yang, Efficient and Dynamic Routing Topology Inference From End-to-End Measurements, IEEE/ACM Transactions on Networking, 2010.
- J. Ni and S. Tatikonda, Network Tomography Based on Additive Metrics, IEEE Transactions on Information Theory, 2011.
- L. Lampe and M. Ahmed, Power grid topology inference using power line communications, IEEE SmartGridComm, 2013, for a power-network application of rooted neighbor joining.

## Large test

The benchmark contains 504 RNJ runs:

- four feeders;
- 1/3/5 operating scenarios;
- 48/96/288 samples per scenario;
- raw and daily-demeaned fitting;
- seven grouping-tolerance factors;
- 0.5% P/Q noise and 0.02% voltage-magnitude noise;
- AC power-flow-generated measurements.

| method/configuration | split F1 | sibling F1 | rooted clade F1 | root success |
|---|---:|---:|---:|---:|
| unrooted NJ, previous 144-run test | 0.592 | 0.459 | not rooted | 0.222 post-hoc |
| unrooted RG, previous 144-run test | 0.652 | 0.698 | not rooted | 0.222 post-hoc |
| RNJ, tolerance factor 0.08 | 0.780 | 0.773 | 0.782 | 0.361 |
| RNJ, tolerance factor 0.16 | 0.846 | 0.869 | 0.846 | 0.750 |

The factor multiplies the median estimated root depth. Factor 0.16 is the retrospectively best single fixed value over this case bank and is not a label-free selection result.

At 5 scenarios and 288 samples, factor 0.16 recovers split, sibling, rooted-clade, and root-partition scores of 1.0 on all four feeders. At 1 scenario and 48 samples, performance remains poor: split F1 is 0.256 and root success is 0.0. Root constraints improve structural consistency but do not replace sufficient excitation and data volume.

The next step should estimate the RNJ threshold from bootstrap uncertainty or a conservative lower bound on physical line impedance rather than selecting it from true topology labels.
