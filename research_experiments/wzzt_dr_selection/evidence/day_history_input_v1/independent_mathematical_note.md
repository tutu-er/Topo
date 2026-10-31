# Post-audit mathematical note

This note was derived after the independent numerical receipt. It changes no original or audited result.

For a nonzero positive-proportion pair x,y, let c=(x' y)/(norm(x) norm(y))>0.
The unconstrained zero-intercept least-squares alpha is positive, so the nonnegative constraint is inactive.

Then:
- residual_RMS^2 = RMS(x)^2 (1-c^2);
- plug-in noise_RMS^2 = sigma^2 RMS(x)^2 (1+c^2);
- residual/noise = sqrt((1-c^2)/(sigma^2 (1+c^2))).

The current ratio is therefore a monotone transformation of uncentered cosine similarity. It is not a new structural statistic. With equal sigma=.005 in both directions, reversing target and predictor has the same mathematical ratio. The two saved directions are a numerical consistency check, not independent information.

At ratio <= k, c^2 >= (1-k^2 sigma^2)/(1+k^2 sigma^2), provided k sigma < 1:
- k=1: c >= 0.9999750003, residual_RMS/RMS(x) <= 0.0070709794 (about 0.7071%);
- k=2: c >= 0.9999000050, residual_RMS/RMS(x) <= 0.0141414286 (about 1.4141%).

Consequences:
1. No full-day pair within 2x means no pair meets this very stringent near-proportional criterion. It does not mean the loads are uncorrelated.
2. This is an observed plug-in reference based on independent multiplicative meter errors and an alpha estimated from the same data. It is not a confidence test.
3. Different noise variances, correlated measurement errors or constrained alpha would need a different expression.
4. Multiplication by 200 cancels from alpha and the dimensionless ratios. The producer 0.2 MVA base still matters for the reported physical kW/kvar residuals and comparisons with device capability; the independent raw samples verified those physical values.
5. Pairwise nonproportionality does not rule out multivariate parameter compensation, P/Q mixing, or weak-block ambiguity.

The P_only_2/Q_only_2 names in the original derived aggregate denote marginal channel totals, including both_PQ_2. They are not exclusive categories. Parent will preserve the old hashed file and issue a separately named derivative with explicit channel-total names.

Sampling scope: only eight predetermined pair/window/view records were recomputed from raw data (first/last job, first pair, P/Q both directions). All 60,620 detail records and 560 case-window summaries were independently reaggregated and internally checked. This is not a claim of independent raw-data recomputation for all 80 observed files.
