# Edge-Stability Clustering for Low-Data Topology Recovery

This note records the perturbation-stability idea implemented in
`experiments/run_stability_cluster_aggregation_demo.py`.

## Basic Procedure

Given already noisy measured data `Z = (P, Q, V)`, first identify a reference
MST from the unperturbed measured data:

```text
T0 = MST(dhat(Z)).
```

Then generate perturbed copies

```text
Z^(b) = Z + xi^(b),  b = 1, ..., B,
```

where `xi^(b)` is an additional synthetic measurement perturbation. For each
copy:

1. estimate the reduced sensitivity matrices;
2. build the terminal distance matrix;
3. run MST;
4. count whether each reference edge `e in T0` appears again.

The empirical edge confidence is

```text
chat_e = (1 / B) sum_b 1{e in T^(b)},  e in T0.
```

Edges with `chat_e >= tau` are treated as high-confidence local edges. Edges
that appear only after perturbation are not used for clustering. The demo
then contracts these edges into pseudo clusters, aggregates P/Q and mean squared
voltage inside each cluster, identifies a pseudo-node MST, and expands it back
to a terminal-equivalent tree.

## Error-Analysis View

Let the estimated distance be

```text
dhat_e = d_e + epsilon_e,
```

where `d_e` is the ideal additive distance and `epsilon_e` collects AC-model
mismatch, finite-sample regression error, preprocessing distortion, and
measurement noise. MST selects an edge when its perturbed distance remains below
competing cut edges. For a true edge `e`, define the local separation margin

```text
gamma_e = min_{f in C(e)} d_f - d_e,
```

where `C(e)` is the set of competing edges across the cut induced by removing
`e` from the true tree. If `gamma_e` is large compared with the standard
deviation of `epsilon_f - epsilon_e`, then `e` is stable under perturbation.

Under a Gaussian approximation,

```text
Pr(e selected) roughly increases with gamma_e / sqrt(Var(epsilon_f - epsilon_e)).
```

So stability is not just "short edge" selection. It estimates a signal-to-noise
margin of the edge-ranking decision. This is why stable edges are good
aggregation candidates: collapsing them removes variables whose relative order
is already reliable and reduces the condition number of the next regression
layer.

## Caveat

Stability is a confidence proxy, not a correctness certificate. A biased
estimator can produce a wrong but stable edge. Therefore the threshold `tau`
must not be too low, and the contracted graph should still be evaluated by the
expanded terminal-equivalent tree. In the first low-data demo, thresholds around
`0.65-0.75` were better than aggressive contraction.
