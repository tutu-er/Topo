# Price objective after the shared-zero counterexample

Full historical-family minimum response distance is zero when both sides can
choose the same zero-extra-block explanation. That rejects that objective, not
the possibility of buying useful observations at the actual positive model.

One alternative worth falsifying is the extra final error relative to the
best feasible menu for the same historical hypothesis. Let h be a complete
historically compatible structure/parameter explanation and let F_h^+ denote
its positive effective blocks. For a precommitted menu m, define

R_m(h) = E_omega [ full_error(A_W(history + predicted_data(m,omega;h)), F_h^+) ].

Consider choosing the menu with the smallest

G_m = sup_h [ R_m(h) - min_{m' feasible} R_{m'}(h) ].

The menu comparison is outside the response expectation: its comparator knows
the hypothetical model but cannot predict the participation realization. It
does not read the actual test topology. Report absolute R as well as G; a small
extra error can coexist with large absolute errors.

## A positive mathematical example, not a WZZT result

With synchronous noiseless history, consider a positive pair12, positive
pair23 and a zero proper block. Use an ideal decoder that recovers an internal
block when a participating node exposes its cross-node response, otherwise
keeps no proper block. For per-event participation.75:

| Hypothesis | repeat node1 | repeat node3 | rotate1 then3 |
|---|---:|---:|---:|
| pair12 |1/16|1|1/4|
| pair23 |1|1/16|1/4|
| zero proper |0|0|0|

The best model-conditional risks are1/16,1/16,0. Worst extra errors are15/16,
15/16 and3/16. A zero explanation therefore does not force this criterion to
be flat. This is a simple-exploration/ideal-decoder example. The actual WZZT
gain threshold, candidate admission, validation and contraction may invalidate
the table and must be included before any method claim.

## Counterexamples and computational boundary

If no action allows a weak block to survive final selection, all menus can
have the same high absolute error and zero extra error. Noise can also make
the zero model's risks differ across menus by creating different false blocks.
Do not assume zero models always have zero extra error.

Keep continuous zero/weak weights and all complete historical families.
Equivalent declarations with a new zero coefficient may be merged by effective
support, but structures only observationally equal on the limited history must
remain distinct.

For a finite collection of legal parameter explanations, its maximum extra
error is only a lower bound on the continuous-domain worst value. It may expose
a bad menu, but it cannot certify a robust-optimal price. Full-domain claims
need valid output-region bounds or global bounds. Nonunique LP optima, RNJ
rebuilding and discontinuous final structural errors make that difficult.

No new price batch or claim of a solved optimizer follows from this note. The
next bounded check may test whether the actual final-decision operator gives
any non-flat ranking, with the corrected selection rule declared first. If
the only positive result belongs to the ideal decoder or ordinary rotation,
say so rather than rename that exploration as a WZZT method.

## Independent measurement-model issue

When TRUE q=kappa*p throughout a toy case, only K=R+kappa*X is physically
identified by noiseless data. On measured inputs the original regression is
K*p_m + X*(q_m-kappa*p_m), whose second input is metering error in this case.
Fitting K alone would use extra known-ratio information and change the observed
measurement model; it is not a neutral rewrite or a WZZT innovation. No K-only
fit has been run here. For heterogeneous background Q this reduction generally
fails even when the DR increment alone has fixed ratio.

The geometric contraction score a*r+b*x is constant along a fixed-K fiber only
if b=kappa*a. Otherwise two predictively equivalent decompositions can produce
different output blocks. Re-scoring the actual raw/contracted predictions on
the same validation observations is the simple first diagnostic; it does not
repair the independent shared-zero price degeneracy.
