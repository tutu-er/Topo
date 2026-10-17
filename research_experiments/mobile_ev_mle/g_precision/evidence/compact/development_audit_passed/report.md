# G-precision frozen study

This report separates EV planning from final shared-line refitting. One history is one statistical unit; its two future regimes are paired tasks.

Status: **pending**. Saved-state audit: **passed**.

|Phase|Method|History records|Joint noisy errors / available|Failed tasks|Unresolved incl. failure|Joint clean errors / available|
|---|---|---:|---:|---:|---:|---:|
|development|OBS|2/2|4/4|0|0|4/4|
|development|CC|2/2|4/4|0|0|4/4|
|development|HL|2/2|4/4|0|0|4/4|
|development|P-SQ|2/2|2/4|0|0|2/4|
|development|G-SQ|2/2|0/4|0|0|0/4|
|development|G-AQ|2/2|0/4|0|0|0/4|
|evaluation|OBS|0/8|0/0|0|0|0/0|
|evaluation|CC|0/8|0/0|0|0|0/0|
|evaluation|HL|0/8|0/0|0|0|0/0|
|evaluation|P-SQ|0/8|0/0|0|0|0/0|
|evaluation|G-SQ|0/8|0/0|0|0|0/0|
|evaluation|G-AQ|0/8|0/0|0|0|0/0|

P-SQ maximizes precision-weighted power squares. G-SQ uses AC-response difference squares; G-AQ also retains the current response difference and its signed linear term. All three optimize the same nested EV domains. Their solver bounds apply to the frozen polynomial surrogate, not to nonlinear AC classification.

For each G policy, the paired summary separately records surrogate improvement, nominal full-AC response distance, and the true-topology margin (half the SSE difference to its strongest competitor). They are different quantities. The true label is used only after plans are sealed.

Clean removes only the new meter noise. Historical data, trained parameters and ordinary forecast errors remain unchanged. Unresolved means a numerical SSE tie or a retained failure; no probability calibration or confidence-set coverage is claimed.

OBS keeps the same measurement duration but performs no battery service. CC, HL and the three MILP policies share the declared battery task. Failures and pending histories remain explicit. Development is diagnostic; evaluation is not used to tune the policy.

Audit forward-AC calls: 384; fitting/planning calls: 0. Audit replay cost is reported separately from the experiment's saved computation costs.

