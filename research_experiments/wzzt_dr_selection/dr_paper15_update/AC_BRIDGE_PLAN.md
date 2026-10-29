# Truth-free AC bridge for candidate-family refitting

2026-10-30. Scratch implementation only; production code unchanged. No paper15 batch has been run by this subtask.

## Recommendation

Use `ac_family_refit.py` as the minimal physics update to the existing fixed-family WZZT L1 refit, serving RNJ bank candidates and WZZT proposals equally. It retains the observed squared-voltage-drop L1 target and all nonnegative coefficients, so it can be inserted into an explicitly named AC-corrected search without relabeling another proxy as line reliability.

The old `terminal_case33.pipeline.ac_likelihood.rank_candidate_trees_ac` remains callable. Its `reference_net` is used by `_candidate_network` only for `base_kv` and `base_mva`, not reference physical branches. Candidates can be built from clades via `research_experiments.rnj.graph_adapters.rooted_tree_from_clades` and `RNJCandidate`; the full-terminal support must be passed as `include_hidden_root_stem=True` rather than discarded as a nontrivial-clade metric. The public function fits an EIV/least-squares model and performs one AC correction, so it is a useful separate comparison, but not exactly the current WZZT L1 objective.

The newer `rnj_wzzt.estimation.ac_measurement_fit.fit_ac_pqv` is also callable, but its fixed-topology latent-P/Q likelihood requires positive absolute pu sigma and explicit finite physical-ohm impedance bounds. Such bounds are not currently identified merely by WZZT response weights. Do not introduce arbitrary finite bounds to obtain an advantage. The scratch bridge therefore uses the existing LP with no finite bound by default; an explicit inherited bound2 is supported for matched comparisons.

## Supplied functions

1. `family_network(point, labels, *, base_kv, base_mva, root_label=1)` builds one physical edge per supplied support. The parent of a support is its smallest strict superset, or the observed root. Singleton nodes keep the meter labels; internal IDs are fresh negative integers. It validates laminarity and retains exact zero impedances. No truth topology, physical line table, or reference impedance is input.
2. Conversion is `r_ohm = .5*r_coefficient*(base_kv**2/base_mva)` and similarly for X. This matches the squared-voltage response convention and existing AC candidate code. P/Q and root/terminal V must be per unit on the same explicit bases.
3. `predict_ac(point, normalized_obs, ...)` calls the current core radial backward-forward solver on raw observed P/Q and observed root magnitude. It returns AC squared-voltage-drop MAE, voltage RMSE and convergence receipts. Initial/trial AC failure is explicit.
4. `fit_family_ac(obs, family, ..., outer_iterations=3, line_steps=(1,.5,.25))` starts from the supplied point, or a joint fixed-family LP. For each iteration it computes `correction = y_AC - y_linear`, jointly refits every R/X coefficient against `y_observed - correction`, and accepts the first line-search trial with strictly lower actual training AC L1. No held-out observations are accepted by this function. The default domain is zero to infinity. This is local iterative correction, not global nonlinear fitting.

`native_point` and `initial_native_point` retain every supplied support and its coefficients, including zero and weak values. The returned effective `point` applies the explicitly recorded inherited `zero_tolerance=1e-9`; callers may request zero tolerance. This is a reporting/pruning tolerance, not a coefficient lower bound.

## Integration requirements

- Existing `dr_pool_v2.selection.measure` and `package` recompute LINEAR train/validation loss from R/X. They must not be reused unchanged for an AC score. Score every RNJ/WZZT candidate and final contracted result by `predict_ac`; keep original linear results separately.
- Reconstruct crossing deletion exactly as current `engine.search`, then refit all retained and new coefficients. Preserve the existing strict search acceptance threshold in squared-voltage-drop units. Inner AC monotonicity is not itself a structural-recovery claim.
- A failed initial AC prediction returns `success=False`, `objective=None` and the original parameter point. Do not substitute linear loss as if it were an AC score. Failed trials retain the last converged candidate with full failure diagnostics.
- Raw observed P/Q are held fixed. This bridge does not correct errors-in-variables statistically, fit fixed meter gains, search source locations or reproduce the full `fit_ac_pqv` likelihood.
- AC uses the same balanced single-phase-equivalent convention as the current core and frozen paper15 simulator. Unmetered source loads are not included by this bridge; an experiment with such sources needs an explicit fixed source input before use.
- Give RNJ-only, linear WZZT and AC-corrected WZZT the same candidate information, train/validation split, bases, and parameter domain. AC reranking of an existing candidate bank is not a greedy-search repair.

## Bounded checks executed

`test_ac_family_refit.py`: 4 passed in 0.178s.

- Independent two-bus closed-form receiving voltage checks the 1/2 factor and physical-ohm conversion.
- Exact zero root/internal/terminal edges remain zero and the unimpeded terminal equals root voltage.
- Misordered P/Q labels and crossing families are rejected.
- A genuinely AC-generated five-terminal data set, fixed topology and no meter noise, has decreased training AC L1 after joint unbounded correction; validation is not accepted as an input. This is an interface/mechanism check, not paper15 recovery evidence.

Previously completed 75 AC API tests were not repeated. Only these four new helper checks were run.
