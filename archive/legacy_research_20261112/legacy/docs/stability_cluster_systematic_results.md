# Stability Cluster Systematic Test Results

Output directory:
`outputs/stability_cluster_systematic_test/`

The sweep uses the unperturbed measured-data MST as the reference tree. For each
condition, extra synthetic perturbations are added to the already noisy measured
P/Q/V data, and the confidence of a reference edge is its repeat frequency under
perturbation.

## Conditions

- Cases: `paper15`, `soumalas11`, `flynn16`, `pengwah18`
- Scenario counts: `1, 2, 3`
- Samples per scenario: `96, 240, 480`
- Extra voltage perturbation relative std: `0.00005, 0.0001, 0.0002`
- Extra P/Q perturbation relative std: `0.0025`
- Replicates per condition: `60`
- Thresholds: `0.55, 0.65, 0.75, 0.85`
- Distance: `RX_75R_25X`
- Preprocessing: `raw_drop`

Success definitions:

- exact success: `F1 >= 1.0`
- high success: `F1 >= 0.9`
- improvement: expanded-cluster F1 is larger than baseline F1

## Main Result

Using the best threshold per case/data/noise condition:

```text
mean baseline F1 = 0.7980
mean expanded F1 = 0.8208
mean improvement = +0.0228
not-worse rate = 1.0000
```

So this stability-cluster strategy is conservative in this sweep: it almost
never harms the base tree, but the average improvement is modest unless the data
regime is very sparse.

## By Data Length

```text
T=96:  baseline 0.7047 -> expanded 0.7419, delta +0.0371
T=240: baseline 0.8177 -> expanded 0.8404, delta +0.0226
T=480: baseline 0.8716 -> expanded 0.8801, delta +0.0085
```

The benefit is largest when the time series is short. With more samples, the
baseline MST is already stable and the aggregation layer has less room to help.

## By Total Sample Count

Here total sample count means `scenario_count * T`. The table uses the best
threshold for each case/data/noise condition.

```text
total samples   baseline F1   expanded F1   delta
96              0.4875        0.5846        +0.0972
192             0.7987        0.8130        +0.0143
240             0.7729        0.8055        +0.0326
288             0.8281        0.8281        +0.0000
480             0.8261        0.8438        +0.0177
720             0.9086        0.9293        +0.0207
960             0.8464        0.8513        +0.0049
1440            0.8880        0.8880        +0.0000
```

The improvement is not monotone in total samples because scenario diversity and
time resolution matter differently. However, the largest gain is clearly at the
lowest sample count. Once the baseline is already above roughly `0.85-0.90`, the
aggregation layer mainly preserves the result instead of creating a large jump.

## By Scenario Count

```text
1 scenario:  baseline 0.7136 -> expanded 0.7637, delta +0.0501
2 scenarios: baseline 0.8056 -> expanded 0.8169, delta +0.0113
3 scenarios: baseline 0.8749 -> expanded 0.8818, delta +0.0069
```

This is consistent with the original motivation: perturbation confidence is most
useful in the low-data regime.

## By Case

```text
paper15:    0.7937 -> 0.8254, delta +0.0317
soumalas11: 0.7889 -> 0.7926, delta +0.0037
flynn16:    0.9037 -> 0.9136, delta +0.0099
pengwah18:  0.7059 -> 0.7516, delta +0.0458
```

`pengwah18` benefits most because its baseline has more unstable intermediate
edges. `soumalas11` changes little: its errors are less fixable by contracting
terminal-local stable edges.

## Threshold Behavior

Fixed-threshold averages across all rows:

```text
tau=0.55: mean F1 0.8056, delta +0.0075, not-worse 1.0000
tau=0.65: mean F1 0.8045, delta +0.0065, not-worse 0.9630
tau=0.75: mean F1 0.8032, delta +0.0052, not-worse 0.9167
tau=0.85: mean F1 0.8022, delta +0.0042, not-worse 0.8426
```

In this sweep, `tau=0.55` is the safest fixed threshold. Higher thresholds have
higher stable-edge precision but lower contraction, so the final expanded-tree
benefit is smaller and can occasionally degrade.
