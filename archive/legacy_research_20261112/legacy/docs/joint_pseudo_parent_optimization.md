# Joint pseudo-parent aggregation: identifiable form and tuning

## 1. Model and identifiability

For a terminal cluster with `m` meters, define the measured squared-voltage
drop matrix

\[
Y_s = v_{0,s}^2\mathbf 1^\top - V_s^2.
\]

The joint local model used by the implementation is

\[
Y_s = g_s\mathbf 1^\top + P_sR_\ell^\top + Q_sX_\ell^\top + E_s,
\]

where `g_s(t)` is the root-to-pseudo-parent common drop and `R_l`, `X_l`
describe only the paths below that pseudo-parent.

The model has an unavoidable gauge freedom. For arbitrary scalars `c_R,c_X`,

\[
R_\ell' = R_\ell+c_R\mathbf 1\mathbf 1^\top,\quad
X_\ell' = X_\ell+c_X\mathbf 1\mathbf 1^\top,
\]

\[
g_s'(t)=g_s(t)-c_R\mathbf 1^\top p_s(t)
                 -c_X\mathbf 1^\top q_s(t)
\]

produce exactly the same fitted terminal voltages. Equivalently, right
multiplication by

\[
M=I-\frac{1}{m}\mathbf 1\mathbf 1^\top
\]

eliminates the common mode:

\[
Y_sM=P_sR_\ell^\top M+Q_sX_\ell^\top M+E_sM.
\]

Therefore terminal data identify local contrasts, but do not identify the
absolute pseudo-parent voltage without a boundary gauge or an external prior.

## 2. Regularized joint objective

The improved solver minimizes

\[
\begin{aligned}
\min_{R_\ell,X_\ell,\{g_s\}}
&\sum_s\|Y_s-g_s\mathbf 1^\top-P_sR_\ell^\top-Q_sX_\ell^\top\|_F^2\\
&+\lambda_A\bigl(\|R_\ell-R_A\|_F^2+\|X_\ell-X_A\|_F^2\bigr)\\
&+\lambda_P\sum_s\|g_s-g_s^{\rm deembed}\|_2^2
 +\lambda_S\sum_s\|Dg_s\|_2^2,
\end{aligned}
\]

subject to symmetric, nonnegative and diagonally ordered local sensitivities:

\[
R_\ell=R_\ell^\top\ge 0,\quad
X_\ell=X_\ell^\top\ge 0,\quad
(R_\ell)_{ii}\ge(R_\ell)_{ij},\quad
(X_\ell)_{ii}\ge(X_\ell)_{ij}.
\]

`R_A,X_A` are the global ordered-fit submatrices after removing one common
boundary component. `D` is a first-difference operator applied separately to
each day, so no artificial smoothing is introduced across day boundaries.
The common mode can use mean, median, or Huber profiling.

The common-mode update is closed form for quadratic profiling:

\[
\bigl[(m+\lambda_P)I+\lambda_SD^\top D\bigr]g_s
=\sum_{i=1}^m z_{s,i}+\lambda_Pg_s^{\rm deembed},
\]

where `z` is the terminal residual after subtracting the current local R/X
drop. R/X are updated by projected gradient. `anchor_strength` is scaled by
the data curvature, making it more portable than a raw ridge coefficient.

## 3. Parameter roles

| parameter | role | practical finding |
|---|---|---|
| `gauge_quantile` | one-time common-path estimate | 0.05-0.10 changes the physical level, but cannot be selected by local residual alone |
| `optimization_gauge_quantile` | keeps the local gauge fixed during iteration | use 0.0 for a clear lower-envelope convention; the old 0.20 loop was numerically identical in this sweep |
| `anchor_strength` | shrink local R/X to the global ordered fit | 0.10 is conservative |
| `prior_weight` | anchor the absolute common mode to de-embedding | useful with robust profiling; too strong alone can change RNJ splits |
| `smoothness_weight` | suppress sample-scale common-mode noise | apply per day; 1.0 is the tested preset |
| `common_mode_method` | row-wise common-mode profile | Huber improves voltage error under occasional local mismatch |
| `joint_blend_with_deembedded` | squared-voltage shrinkage | 0.75 is a useful alternate candidate, not a universal optimum |

The named `regularized_joint_vsq` preset uses gauge 0.05, zero lower-envelope
optimization gauge, anchor 0.10, prior 1.0, smoothness 1.0, and Huber profiling.

## 4. 3 days x 96 points results

Conditions are unchanged from the standard benchmark: radial AC power flow at
every sample, 15-minute resolution, P/Q relative noise 0.5%, V relative noise
0.02% of instantaneous local voltage, and root-voltage variation. Three
bootstrap detector seeds were used for each of paper15, soumalas11, flynn16,
and pengwah18.

Eleven conditions produced at least one NJ/RG consensus cluster. One flynn16
replicate safely skipped aggregation because no consensus cluster was found.

| method | mean rooted-clade F1 | exact rate | mean true pseudo V2 RMSE |
|---|---:|---:|---:|
| de-embedded, weight 0.15 | 0.9470 | 54.5% | 0.01007 |
| de-embedded, weight 0.50 | 0.9939 | 90.9% | 0.00455 |
| unregularized joint | 0.9939 | 90.9% | 0.00793 |
| regularized Huber joint | 0.9939 | 90.9% | 0.00331 |
| regularized joint/de-embedded 0.75 blend | 0.9939 | 90.9% | 0.00338 |

Relative to the unregularized joint estimate, the regularized Huber form
reduces the diagnostic pseudo-parent squared-voltage RMSE by about 58%. It does
not increase mean F1 further because the remaining failure is upstream: one
pengwah18 detector realization supplies only two consensus clusters, and every
pseudo-voltage configuration remains at F1 0.933. A voltage estimator cannot
recover a cluster absent from the candidate set.

## 5. Deployment recommendation

1. Keep `deembedded_vsq` with weight 0.50 as the default. It is cheap,
   label-free, and has the same topology result as the best joint variants in
   this sweep.
2. Add `regularized_joint_vsq` as an alternate candidate when pseudo-parent
   voltage accuracy matters or when downstream AC likelihood is already used
   to rank candidates.
3. Do not select the absolute gauge solely by local residual minimization. The
   gauge invariance makes that criterion incapable of validating the physical
   pseudo-parent depth.
4. Treat strong prior/smoothing as a coupled preset. `prior=1, smooth=1` with
   ordinary mean profiling reduced pengwah18 F1 in two detector realizations.
5. Improve the stable-cluster candidate recall separately. This is now the
   limiting stage in the remaining non-exact condition.

## 6. Interfaces and reproducibility

```python
result = run_ordered_two_level_aggregation(
    scenarios,
    root_bus,
    clusters,
    pseudo_voltage_mode="regularized_joint_vsq",
    deembedding_weight=0.50,
)
```

Any preset parameter can be overridden through `joint_fit_options`. The unified
pipeline exposes `aggregation_pseudo_voltage_mode` and
`aggregation_deembedding_weight`.

```powershell
python -m experiments.run_joint_pseudo_parent_optimization `
  --bootstrap-replicates 6 `
  --detector-replicates 3 `
  --output-dir outputs/joint_pseudo_parent_optimization_3seeds
```

Detailed rows, adaptive label-free selections, and aggregate summaries are in
`outputs/joint_pseudo_parent_optimization_3seeds/`.
