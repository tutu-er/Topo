# Experiment M results (PROTOCOL_MIQP.md)

## Certificates at scale: node-by-node against the global MIQP (2026-11-25)

On the identifiable truth of transformer-level networks (`eulv_large`, day 0, 192 slots; one thread; scratchpad
`bench3.py`). "No-gain" proves that no compatible line anywhere pays its price; "force" finds the strongest split
at every node. Every node MIQP was proved optimal (status 2) on every network.

| Network | Meters | Nodes | Widest node (blocks) | Node-by-node, no-gain / force | Global time-free MIQP | WZZT-R set-up (SLSQP guidance) |
| --- | --- | --- | --- | --- | --- | --- |
| td400858 feeder59 | 24 | 7 | 6 | 1.1 s / 1.2 s | 83 s | 1 s |
| td401346 | 26 | 2 | 5 | 0.4 s / 0.4 s | 150 s | 1 s |
| td400858 | 34 | 10 | 6 | 2.9 s / 2.8 s | 648 s | 19 s |
| td400291 | 44 | 10 | 9 | 3.1 s / 3.2 s | time limit 1,200 s | 67 s |
| td009471 | 53 | 10 | 8 | 8.7 s / 8.8 s | not run | 177 s |
| td006145 | 63 | 13 | 7 | 13.6 s / 12.8 s | not run | 22 min |
| td008742 | 73 | 16 | 6 | 22.6 s / 21.3 s | not run | 78 min |
| td008690 | 77 | 21 | 6 | 19.6 s / 19.1 s | not run | 3.4 h |

Each node MIQP has at most 585 variables whatever the network size, so the certificate grows only with the number
of nodes. The set-up column is dominated by the guidance's ordered matrix fit with SciPy SLSQP (timed at 34 and 44
meters: 11 s and 73 s of guidance against 1 s and 5 s for the noise estimate); the Gurobi QP option solves the same
fit in seconds (`trees.ordered_qp_gurobi`), and the frozen method's default is not yet switched.

## Amendment D: root lines and terminal lines, WZZT-R against RNJ, NJ and RG (2026-11-25)

Run: `compare_builders.py` (best-fitting R/X matrix by the ordered P-and-Q fit on WZZT-R's slots, per-meter
demeaned, Gurobi QP; RNJ, NJ, RG at 3 weights x 5 tolerances), `analyse_builders.py`; output
`outputs/wzzt_evidence_20261124/builders/` (`analysis_20261125.txt`). No errors (33 + 520 inputs).
"Best" is each builder's setting with the best pooled F1 on that input set, chosen after the fact.

**Confirmatory set `eulv_16d`** (33 real feeders, 16-day histories, 242 true internal lines):

| Line class | Lines | WZZT-R | RNJ default / best | NJ default / best | RG default / best |
| --- | --- | --- | --- | --- | --- |
| Near root (f > 1/2) | 88 | **80.7** | 28.4 / 37.5 | 17.0 / 37.5 | 17.0 / 22.7 |
| near root, short (ρ < 0.08) | 18 | 55.6 | 11.1 / 16.7 | 0.0 / 16.7 | 0.0 / 5.6 |
| Terminal side (f ≤ 1/2) | 154 | **50.6** | 25.3 / 36.4 | 22.1 / 35.7 | 14.9 / 16.9 |
| terminal side, long | 129 | 55.0 | 26.4 / 39.5 | 24.8 / 39.5 | 17.1 / 19.4 |
| All lines | 242 | 61.6 | 26.4 / 36.8 | 20.2 / 36.4 | 15.7 / 19.0 |
| Precision | | **86.1** | 28.3 / 25.8 | 31.2 / 27.2 | 27.1 / 25.8 |
| False lines per input | | 0.73 | 4.9 / 7.8 | 3.3 / 7.1 | 3.1 / 4.0 |

Recall in %, pooled over inputs; best settings: RNJ and NJ weight 0.75, tolerance 0.04; RG weight 0.75, tolerance 0.

- **HR1 (root lines): supported.** WZZT-R's per-input recall on near-root lines exceeds every builder's, at the
  default and at the best setting (30 inputs; mean differences 43–60 points; Holm-adjusted p ≤ 4.1e-4).
- **HR2 (terminal lines within 10 points): not supported.** On long terminal-side lines the best builder (RNJ,
  tolerance 0.04) is 15.5 points below WZZT-R [−27.0, −3.5]. The distance-based builders are not adequate on
  their own even near the terminals, though the gap there is about a third of the gap near the root.
- **HR3 (mechanism): direction as predicted, underpowered.** In the middle and high evidence tertiles the
  builders recall short lines 14–43 points less often than long ones (NJ and RG default intervals exclude 0);
  WZZT-R's differences are +5.6 and −6.2 points with wide intervals (only 8 and 11 short lines in those bands).

**Exploratory set, Phase 3** (already seen): real feeders (328 inputs, 2,405 lines) near root 73.9 vs ≤ 32.8 for
any builder setting, terminal side 42.8 vs ≤ 30.5, precision 86.1 vs ≤ 25.9; synthetic feeders contain no
near-root lines (terminal side 84.6 vs ≤ 75.5). The earlier "83 of 84" figure counted only data-supported lines;
on all identifiable-truth lines WZZT-R finds 55% of short near-root lines (builders ≤ 19%). Lines the data do not
support are declined by the score, by design.

**Where WZZT-R's accepted lines come from** (W events, `eulv_16d`, 175 accepted lines): the RNJ pool of its own
guidance (a P-only fit under the GLS and offset transform, 4 tolerances) 125, of them 58 of the 82 near-root
lines; the screen 29; sibling pairs 11; residual clusters 9; RNJ on the residual 1. The final MIQP check ran 32
times (31 proved optimal, median 2.2 s; one 120 s time limit) and never found a better line. So RNJ trees mostly
*propose* the lines, and the whole-map score *decides* which to keep.

**Limit of the comparison.** The builders use the standard P-and-Q matrix, not the transformed P-only matrix that
WZZT-R's own guidance uses (better: its RNJ reference tree reached 44.5% of near-root supported lines on the
Phase 3 real feeders against 26.9% for the standard RNJ). Builders on WZZT-R's matrix are the fair next check.

**W by MIQP alone (the original design), Phase 2 ablation `R_wmiqp`.** On the 130 development inputs (≤ 24
meters) the per-step MIQP was fast (771 solves, 98% proved optimal, median 4.5 s; it beat the best pool line in
221 steps) but not more accurate: synthetic exact 33 against 32, real supported 208 against 228 (p = 0.07), real
precision 79.2% against 85.4%, median time 51/184 s against 42/116 s. On wide early steps of large networks it
does not finish (first step provable up to 29 meters, 121–247 s; 48% gap at 34 meters).
