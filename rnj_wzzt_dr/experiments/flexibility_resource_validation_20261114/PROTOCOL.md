# Voltage, active procurement and evidence-based Q sizing

Locked before new optimizer or execution results, 2026-11-14.

## Questions and fixed decisions

Audit every saved full-network AC state in the existing 480-record study, including hidden buses, voltage bounds 0.95--1.05 pu, current, branch ratings if present, power balance and active procurement. A normalization base is not equipment nameplate capacity. Missing thermal ratings remain unknown.

Retain the original frozen candidates, historical validation, strict full-native MSE selection, representative probability model and all ten historical-mean direction guards. Do not change fitting, noise, target mass, risk, price or threshold to obtain a result.

Retain the 15 historical PV/battery participant locations and their Q device IDs/order. The other three measured nodes have no new participation contract. For each participant set q_new=max(q_old, tier(Q95)), where Q95 is the absolute Q 95th percentile of the original 192 training samples, and tiers are 1, 2, 3, 5, 7.5, 10, 15, 20 kvar. Existing 303/304 contracts remain at least 5 kvar. Their original 5 kvar was a declared scenario, not estimated from Q95. Other original contracts remain at least 1 kvar. No response matrix, truth, new-day load or execution result enters the sizing formula. Prior experiments motivated the retained resource domain; this is a development change, not an outcome-blind original design.

Model additional independent bidirectional compensators. Assume nameplate Q at least contract/0.8, round up to 3/5/10/15/20/30 kvar equipment tiers. The present case uses only vendor-documented 3/5/10 kvar classes. 80% is an engineering headroom assumption, not a manufacturer requirement. Remote signed dispatch and all-day availability are explicit service assumptions. Do not infer inverter spare kVA or night service from registered PV kW. This gives 36 kvar contracts and 63 kvar example equipment, compared with the old 23 kvar contracts.

P contracts and all prices stay unchanged. Keep P and Q quantities separate. Equipment, installation, capacity reservation and auxiliary energy are excluded from the illustrative activation fee and must be identified as such.

## Paired and fresh-path checks

1. Paired resource development: updated configuration, four forecasts, eight execution/meter paths each, seed 20261115. Compare to the completed old23 results, reusing the same random streams. These are previously used observations.
2. Fresh-path check: freeze both configurations before execution, four new forecasts and eight paths per forecast, seed 2026111409, comparing old23 and new36. Both share each forecast, background, delivery multipliers and meter streams. This provides fresh conditional paths on the same previously studied physical day, not new independent physical days.
3. Each quote gets 300 additional seconds, including current seed generation/loading/auditing and model construction. Single solver thread, at most four processes. Prior solved plans are transferred by exact resource ID/node/unit and are rechecked on the current domain and evidence. Old computation is separately disclosed. The same forecast is required in the paired stage; fresh stages permit old actions as templates with full rechecking and do not call them pre-certified.
4. Execute only no procurement, the absolute-plus-guard plan, and the same-planned-fee rotation. Unresolved planning uses an explicit zero fallback. No old-increment optimization is necessary for this question.

Report all quotes, bounds/gaps, seed retention or improvement, exceptions, failed AC and completed paths. Count correct-to-wrong and wrong-to-correct changes, correct/wrong high weights, mean clean and measured evidence gains, forecast/actual J, active/reactive quantities, fees, all-bus voltages and available thermal metrics. Counts alone are not an accuracy guarantee.

Assess whether the paired updated recovery is at least the old 26/32 without losing previously correct cases. On fresh paths compare new36 with both no action and old23, including adverse flips and wrong high weights. Record these comparisons even if the new configuration loses; do not retune on fresh outcomes. Preserve the old23 configuration if a larger contract fails to maintain its advantage. Any further design must be separately named and the consumed confirmation paths treated as development data.

## Independent verification and preservation

All raw P/Q/V, historical source statistics, fitted matrices, actions, quotes and execution paths remain in ignored outputs. Public artifacts contain source code, declared synthetic device capacities and aggregate results only. Save input/source/environment hashes, complete markers and locks. Never overwrite old completed batches. Check all terminal and hidden-bus AC values directly from saved arrays; independently recompute strict selection and evidence, delivered=requested times the locked multiplier (with original P availability clipping), resource feasibility, fees, shared random streams and candidate/source identity. No identifier refitting.
