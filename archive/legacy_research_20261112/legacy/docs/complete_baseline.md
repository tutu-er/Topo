# Complete constrained RNJ baseline

## Scope

The baseline is the smallest complete end-to-end method used before pseudo
aggregation, latent voltage modes, bootstrap stability, or paper-specific
extensions. It consumes only noisy terminal P/Q/V and the observed root
voltage. Ground-truth topology is used only after fitting to calculate metrics.

## Measurement model

For scenario s, define the observed squared-voltage drop

\[
Y_s(t,i)=V_{0,s}(t)^2-V_{i,s}(t)^2.
\]

With positive P/Q denoting load, the fitted model is

\[
Y_s=P_sR^\top+Q_sX^\top+\mathbf 1\gamma_s^\top+\delta V_s.
\]

The scenario fixed effect \(\gamma_s\) absorbs a constant offset after temporal
preprocessing. The residual \(\delta V_s\) includes meter noise, LinDistFlow
approximation error, AC loss terms, and unmodelled effects.

The baseline solves

\[
\min_{R,X,\{\gamma_s\}}
\frac12\sum_s\|\delta V_s\|_F^2
+\frac\alpha2(\|R\|_F^2+\|X\|_F^2),
\]

subject to symmetric, nonnegative, ordered R/X matrices. The default is
\(\alpha=0\). Therefore the baseline already has an implicit quadratic
\(\|\delta V\|_F^2\) error term, but it does not use a separate structured
error variable, covariance weighting, or P/Q errors-in-variables correction.

## Distance and RNJ

The terminal distances are

\[
d^R_{ij}=R_{ii}+R_{jj}-2R_{ij},\qquad
d^X_{ij}=X_{ii}+X_{jj}-2X_{ij}.
\]

The default mixed distance and root depth are

\[
d_{ij}=0.75\frac{d^R_{ij}}{\overline d^R}
       +0.25\frac{d^X_{ij}}{\overline d^X},
\]

\[
h_i=0.75\frac{R_{ii}}{\overline d^R}
    +0.25\frac{X_{ii}}{\overline d^X}.
\]

The shared-path matrix is

\[
S_{ij}=\frac{h_i+h_j-d_{ij}}2.
\]

RNJ reconstructs a rooted hidden tree from \(S\) and \(h\), retaining the
physical root label. The grouping tolerance is

\[
\tau=0.16\operatorname{median}(h).
\]

## Public API

```python
from terminal_case33.estimation import fit_complete_rnj_baseline

result = fit_complete_rnj_baseline(
    scenarios,
    root_bus=1,
    preprocessing="daily_demean",
    distance_mode="RX_75R_25X",
    constraint_mode="ordered",
    alpha=0.0,
    tolerance_factor=0.16,
)
```

The result exposes R, X, dR, dX, mixed distance, root depths, scenario fixed
effects, every \(\delta V_s\) residual sample, objective value, condition
number, RNJ edges, and rooted terminal clades.

## Reproducible benchmark

```powershell
python -m experiments.run_complete_baseline `
  --scenario-counts 3 5 `
  --T-counts 24 48 96 `
  --data-replicates 2
```

The 48 conditions use four cases, 3/5 coupled daily scenarios, 24/48/96 samples
per scenario, two independent data replicates, 0.5% P/Q noise, 0.02% voltage
noise, and AC-generated P/Q/V.

| metric | value |
|---|---:|
| mean rooted-clade F1 | 0.8837 |
| exact recovery | 41.67% |
| precision | 0.8329 |
| recall | 0.9531 |
| root-partition exact rate | 87.50% |
| mean residual RMSE in squared-voltage drop | 0.00347 p.u. |
| mean regression R2 | 0.8420 |
| median condition number | 16765.7 |

Outputs are written under `outputs/complete_baseline/`. Each condition has its
own auditable matrices, residuals, fixed effects, tree edges, clades, and fit
metrics.
