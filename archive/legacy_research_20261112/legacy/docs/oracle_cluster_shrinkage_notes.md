# Oracle Cluster Voltage Lifting with Shrinkage

This note records the shrinkage variant used in
`experiments/run_oracle_cluster_lifting_large_validation.py`.

## Estimator

For a terminal child `i` in an oracle terminal cluster with hidden parent `h`,
write the squared-voltage drop across the service line as

```text
v_i(t) = v_h(t) - delta_i(t) + eps_i(t)
```

where `v_i = |V_i|^2`, `v_h = |V_h|^2`, `delta_i` is the true service drop, and
`eps_i` is voltage measurement noise plus AC-model mismatch after preprocessing.
The full service-drop lifting estimator uses

```text
vhat_h,full(t) = mean_i [v_i(t) + deltahat_i(t)].
```

The shrinkage estimator inserts a scalar `lambda`:

```text
vhat_h,lambda(t) = mean_i [v_i(t) + lambda * deltahat_i(t)],
0 <= lambda <= 1.
```

The script currently tests `lambda = 0.25, 0.50, 0.75`. The two endpoints are:

- `lambda = 0`: simple cluster mean voltage, close to `mean_vsq`;
- `lambda = 1`: full estimated service-drop lifting, close to `estimated_drop_mean`.

## Error Decomposition

Let the service-drop estimate satisfy

```text
deltahat_i(t) = delta_i(t) + eta_i(t),
```

where `eta_i` is the service-drop estimation error. Then

```text
vhat_h,lambda(t) - v_h(t)
  = mean_i [-(1 - lambda) delta_i(t) + lambda eta_i(t) + eps_i(t)].
```

Ignoring covariance terms gives the approximate mean-square error

```text
MSE(lambda)
  approx (1 - lambda)^2 S_delta^2
        + lambda^2 S_eta^2
        + S_eps^2,
```

where `S_delta^2` is the mean squared residual service drop left uncorrected,
`S_eta^2` is the variance of the estimated correction error after averaging
within the cluster, and `S_eps^2` is the voltage noise contribution.

The approximate oracle-optimal shrinkage is therefore

```text
lambda_star approx S_delta^2 / (S_delta^2 + S_eta^2).
```

So full correction is useful only when the service drop is large relative to the
correction error. If the local service sensitivity is noisy or ill-conditioned,
smaller `lambda` reduces variance and can improve MST edge ordering even when it
slightly worsens the fitted R/X magnitude.

## Empirical Result

The full validation output is in
`outputs/oracle_cluster_lifting_large_validation_shrinkage/`.

Across 216 case/filter/time/distance tasks:

```text
estimated_drop_shrink_025  mean F1 = 0.9531, edge recovery = 0.9493
mean_vmag                  mean F1 = 0.9420, edge recovery = 0.9366
estimated_drop_shrink_075  mean F1 = 0.9369, edge recovery = 0.9376
estimated_drop_mean        mean F1 = 0.9360, edge recovery = 0.9366
original_terminal          mean F1 = 0.7509, edge recovery = 0.7444
```

The current best fixed choice is therefore `estimated_drop_shrink_025`. This
matches the error model: the service-drop correction contains useful signal, but
its estimation noise is large enough that only a modest correction is optimal
for edge ranking.

