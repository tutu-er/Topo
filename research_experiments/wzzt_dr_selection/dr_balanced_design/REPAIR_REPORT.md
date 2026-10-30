# B: completion, selector repair and new response/noise test

**A simple final-selector change preserves substantially more information than
the fixed3% simplicity band. It also adds false blocks. The change improves
WZZT in a new behavior/measurement test, but the same paid observations still
give identical strict outputs from RNJ plus an optional common root. A new
WZZT-specific price optimizer has not been demonstrated.**

## A separately recorded completion, not an extended first batch

After the user's instruction to solve problems and continue testing, a new
completion protocol froze only the original91 incomplete jobs. Original menus,
actions, seeds, natural input, history, DR budget and identification rules are
unchanged. All original evidence remains read-only, including its3000-LP end
receipt. The217 completed original mappings are unchanged.

The completion executes90 pending trajectories, reuses the one already delivered
trajectory and its completed preprocessing, and reuses15 completed observation
fits. It produces76 additional record files. New compute is1234 LPs/69.03s,
within its separate1800-LP/600s limit. Original plus completion is4234 LPs.
The merged panel is now308/308 complete with262 distinct observed fits and no
partial W scans. It remains two behavior/measurement streams on the same known
natural trajectory, not308 independent observations.

Independent completion audit has23,996 passing checks and recounts all4234 LP
receipts. The first audit incorrectly equated91 jobs with91 distinct fits; that
failed audit is retained. Corrected audit uses87 distinct completion fit IDs,
including reuse, and no experiment was rerun. Saved/raw/final L1 replay error0.

## What the selector repair changes

The original final selector admits raw candidates within3% of the lowest
validation L1, prefers the simplest, then contracts geometric weak blocks
without rechecking their validation loss. The tested alternative is:

1. Keep the actual original fitted candidates and accepted paths.
2. Recompute the same complete validation L1 for every raw candidate and its
   original contracted variant.
3. Select the minimum loss, with only an absolute1e-12 numerical tie; ties
   prefer fewer declared supports, original index, then raw.
4. Do not apply an additional unvalidated contraction afterwards.

All continuous weights were already jointly optimized from zero. This change
does not refit weights, modify search acceptance, change the candidate bank,
or use the true topology in selection. The production default is unchanged;
selection_rules.py and confirm.py implement the research comparison.

Four rules were declared before diagnostic execution: original3%+contraction,
original3% raw, strict raw, and strict raw-or-rescored-contract. Old186 fit
selections are hash-matched cache entries. Only76 new fits require3762 new raw
point scores and their3762 contracted scores; no new LP is used for this replay.
Across all262 fits,786 method comparisons reproduce original selections and
scores exactly. Strict raw and strict raw-or-contract choose the same outputs;
every strict winner is raw. Removing contraction alone does not fix WZZT.

## Complete development panel:308 matched mappings

| Method | Original errors / full recovery | Strict errors / full recovery |
|---|---:|---:|
| RNJ |529 /0|609 /0|
| Finite-pool WZZT |442 /12|282 /126|
| RNJ plus optional root |293 /70|268 /135|

Strict W is better178 times, equal90, worse40. Its root errors fall216->27,
but leaf errors rise4->43 and extra false blocks rise24->88. In the zero-proper
group, false internal blocks rise5/44->25/44. The full-error total still includes
all zero/weak cases; none is removed to improve the apparent result.

This diagnoses a real simplicity-selection bottleneck, not a new price design
or an independently validated topology confidence statement.

## One prelocked new behavior/measurement test

The selector source is frozen before generating two new behavior streams934031/32
and measurement streams934131/32. Keep natural884001, all288/288 old observations,
480/480 merged observations, the same seven zero/weak/strong true models, prices,
participation law, task amplitude, coefficient domain and payment cap1.6.

Five fixed policies are frozen by independent baseline seeds before actual
generation: zero, repeated single2, different-single rotation2, repeated opposed
pair1 and different opposed-pair1. There is no test-best menu selection and no
new optimized-price claim. All70/70 mappings finish,54 different observations,
592 LPs/41.91s, within1200 LPs/600s. No failures or partial scans. The streams
provide new behavior/measurement variation, not a new natural-load population.

| Method | Original errors / full recovery | Strict errors / full recovery |
|---|---:|---:|
| RNJ |134 /0|146 /0|
| Finite-pool WZZT |116 /2|78 /24|
| RNJ plus optional root |57 /14|61 /25|

The same selector change improves W in37, leaves22 equal and worsens11. It is
not uniformly beneficial to other methods; root-augmented RNJ has lower total
error under its original rule than under strict selection in this test.

The improvement is concentrated in the common root: W root errors55->7,
while internal-pair errors remain52. Extra false blocks increase5->23 and
leaf errors9->19. There are37 correct saved W candidates, all3%-eligible;
strict selection retains24, leaving13 such cases unrecovered. It converts23
previously wrong outputs to correct ones and makes one previously correct
output wrong. This does not establish better internal-block price guidance.

Each fixed policy below has14 mappings. Results are strict W final selections:

| Policy | Errors | Full recovery | Actual four-event payment r0 / r1 |
|---|---:|---:|---:|
| Zero |32|2/14|0 /0|
| Repeated single2 |19|3/14|1.2 /1.6|
| Single rotation2 |7|7/14|1.2 /1.6|
| Repeated opposed pair1 |11|5/14|0.8 /1.2|
| Different opposed pair1 |9|7/14|0.6 /1.2|

Every56 paid mapping has exactly the same strict effective support from W and
root-augmented RNJ. The remaining method differences are7 zero-policy mappings.
Different opposed pairs do not beat simple single rotation; repeated-pair and
different-pair cost and participation realizations must also be reported.
No sevenfold multiplication of deployment costs across counterfactual models.

## Interpretation and next method question

The tests support preserving validation distinctions rather than automatically
rounding them into a3% simplicity band. They also show the cost of overreporting
weak or false blocks. Keep both error types visible. This is a selector repair
and a necessary baseline check, not the paper's WZZT/DR novelty.

The full-history shared-zero witness remains: conservative whole-family minimum
separation is zero for all22 menus. Changing the final selector does not fix
that price objective. A next bounded investigation may compare the final extra
errors of a menu against the best feasible menu under each historical model,
as derived in REGRET_REVIEW.md. It must use actual acceptance/selection, retain
zero/weak parameters and distinguish finite counterexample bounds from a
continuous-domain optimum. No new price batch has been run for this idea.

## Recovery and privacy

Local source stages:01e20e89 (separate completion),2cb5da79 (selector and new
confirmation). No production defaults or previous frozen evidence are changed.

Private cache roots:

- wzzt_b_completion_v1:91 updates and merged308 results; original root untouched.
- wzzt_b_completion_audit/run_v2:independent corrected audit; run_v1 retained.
- wzzt_b_selection_stage:217-map development replay and seven selector tests.
- wzzt_b_selection_complete_stage:cache-preserving308-map extension.
- wzzt_b_confirm_v1:frozen new70-map protocol/actions, observed fits and receipts.
- wzzt_b_confirm_audit:independent new-test replay.

All temporal P/Q/V, control and device arrays remain private. Public evidence
is explicit aggregate JSON only. Git changes are local while the prior request
to approve the GitHub destination remains unanswered; no retry or alternative
external upload is attempted.

Independent confirmation audit performs15,221 passing checks, recounts all592
optimal LP receipts, checks125 accepted and112 stopped steps, and recomputes
5334 raw/contract validation scores with difference0. Its first replay had a
tuple-versus-JSON-list support comparison error; that source/receipt is retained.
Only serialization normalization changed, with no experiment or selector rerun.
