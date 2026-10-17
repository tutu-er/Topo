# G-precision frozen study

This report separates EV planning from final shared-line refitting. One history is one statistical unit; its two future regimes are paired tasks.

Status: **complete**. Saved-state audit: **passed**.

|Phase|Method|History records|Joint noisy errors / available|Failed tasks|Unresolved incl. failure|Joint clean errors / available|
|---|---|---:|---:|---:|---:|---:|
|development|OBS|2/2|4/4|0|0|4/4|
|development|CC|2/2|4/4|0|0|4/4|
|development|HL|2/2|4/4|0|0|4/4|
|development|P-SQ|2/2|2/4|0|0|2/4|
|development|G-SQ|2/2|0/4|0|0|0/4|
|development|G-AQ|2/2|0/4|0|0|0/4|
|evaluation|OBS|8/8|4/16|0|0|4/16|
|evaluation|CC|8/8|4/16|0|0|4/16|
|evaluation|HL|8/8|4/16|0|0|4/16|
|evaluation|P-SQ|8/8|2/16|0|0|2/16|
|evaluation|G-SQ|8/8|1/16|0|0|0/16|
|evaluation|G-AQ|8/8|2/16|0|0|0/16|

P-SQ maximizes precision-weighted power squares. G-SQ uses AC-response difference squares; G-AQ also retains the current response difference and its signed linear term. All three optimize the same nested EV domains. Their solver bounds apply to the frozen polynomial surrogate, not to nonlinear AC classification.

For each G policy, the paired summary separately records surrogate improvement, nominal full-AC response distance, and the true-topology margin (half the SSE difference to its strongest competitor). They are different quantities. The true label is used only after plans are sealed.

Clean removes only the new meter noise. Historical data, trained parameters and ordinary forecast errors remain unchanged. Unresolved means a numerical SSE tie or a retained failure; no probability calibration or confidence-set coverage is claimed.

OBS keeps the same measurement duration but performs no battery service. CC, HL and the three MILP policies share the declared battery task. Failures and pending histories remain explicit. Development is diagnostic; evaluation is not used to tune the policy.

Audit forward-AC calls: 1920; fitting/planning calls: 0. Audit replay cost is reported separately from the experiment's saved computation costs.

Per-policy deployment costs include dependencies: history and baseline completion for all three MILP methods; P-SQ additionally needs its own family; each G method also needs P-SQ inheritance, AC-response setup and its own family. Unallocated selection time is reported as zero-to-all common overhead, giving a cost allocation range. Inference uses that policy's noisy-refit time averaged across the two regimes, charging a shared action in full to each separate deployment. Clean diagnostics and the other G family are excluded from this deployment range. The original batch costs are listed separately.

|Phase|Policy|Planning mean allocation range (s)|Noisy inference mean (s)|Deployment mean allocation range (s)|
|---|---|---:|---:|---:|
|development|P-SQ|4.561–4.626|1.560|6.121–6.186|
|development|G-SQ|9.028–9.093|1.432|10.460–10.525|
|development|G-AQ|10.840–10.905|1.547|12.387–12.452|
|evaluation|P-SQ|4.596–4.660|1.329|5.925–5.989|
|evaluation|G-SQ|8.746–8.811|1.332|10.078–10.142|
|evaluation|G-AQ|9.795–9.859|1.350|11.145–11.209|

