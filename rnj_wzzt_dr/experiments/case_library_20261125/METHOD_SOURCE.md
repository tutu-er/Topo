# WZZT-C with one unmetered load (development note, 2026-11-25)

Code: `identification/source.py`, tests: `test_source.py`. Nothing here is registered; every run so far used development
cases.

## Model

The user's source term: one unmetered load with power u_t = PCC - meters - losses, and h_e = 1 on the lines of the path
from the PCC to it:

    Source_t = sum_e h_e (r_e uP_t + x_e uQ_t) z_e

so a line C carries its meters' power plus h_C u_t. All errors are of the squared voltage (L2); r and x stay free per
line and both enter every fit. The identifiers keep the linear model; the data are AC.

## Unmetered power

u_t = PCC_t - sum of meters - losses. Losses are computed **without any topology** from the meters' own voltages:
loss_t = sum_i P_i (V0 - V_i) / V_i (the exact balance S_PCC - sum S_i = sum (V0 - V_i) conj(I_i) with small angles).
On AC data of `paper15_street` this is within 6% of the true loss. The same formula is used for Q.

## S1, robust: the source is removed

For any path, the source adds rho_i uP_t + chi_i uQ_t to meter i, with rho, chi unknown. The residual operator of
WZZT-C already removes each meter's constant; it now also removes the time series uP and uQ. Every MIQP, the Gram
statistics and the certificate are unchanged. It returns a tree and no location. (Test: the residual equals that of a
fit with free source coefficients for every meter.)

## S2, located: the source is one more leaf

The unmetered load is leaf n of a tree over n + 1 leaves: **its power is known, its voltage is not.** A line carries u
exactly when its set contains leaf n. In a tree the sets that contain a given leaf are nested, so they are the path from
the PCC to the source; laminarity over n + 1 leaves is the only constraint needed. The pool MIQP therefore chooses the
tree and the path together, exactly, with the same coefficients r_C, x_C: the path costs no parameter, and the lines of
the path must explain the source with the same r, x that they need for their meters.

- Candidates: every line of S1's sources and of S1's tree, with and without leaf n, and leaf n below each single meter.
- Warm start: S1's tree with the source at its best place (a fit per place).
- Output: tree, path, the cost in half-BIC units of every other place of the source, and the evidence that there is a
  source at all (the same tree with u set to zero).

**Why this matters for the MIQP's novelty.** A leaf without a voltage has no row and no diagonal entry in the path
matrix, so RNJ, NJ and RG cannot place it: they can only be given the corrected matrix of the metered leaves (S1's
projection), which is what the comparison gives them. The whole-map formulation treats the leaf like any other.

## What is not done

- No node certificate for S2 (S1's tree is certified; S2 is the exact optimum over its candidates only).
- One source, not several; the source's own service line is not identifiable (no voltage behind it) and not modelled.
- The place of a source at the head (just after the trunk) changes no voltage except through the trunk, so it is
  detected by the PCC balance, not by the voltages; its evidence is small by construction.
- The S2 candidates come from S1's sources; a source-specific candidate generator does not exist.
