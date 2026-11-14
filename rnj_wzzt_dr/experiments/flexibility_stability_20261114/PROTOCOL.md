# Fixed comparison of capacity procurement and stable actions

Locked before new planning or execution. Date: 2026-11-14. The user requested an assessment of the flexibility interpretation and experiments seeking reasonable, better identification results. This is an isolated study; old completed files and production defaults remain unchanged.

## Question and fixed identification problem

Q-only procurement is a valid reactive flexibility service. No minimum P purchase is imposed. Test whether (a) historically registered P offers at additional relevant terminals, (b) paying for available capacity as well as activation, or (c) refusing economically negligible action replacement improves the delivered identification evidence.

Retain the Pengwah18 2024-10-14 enriched measured-P/Q case, complete 192-row training and 96-row validation history, frozen candidate parameters and shared sigma, original representative absolute-recovery and ten direction-protection constraints, original full five-point native-bank MSE selection, load day, AC bridge, P/Q/V error laws and whole-day response law. No fitting, shortened history, true-line parameter access in pricing, changed identification threshold or removal of adverse paths.

The physical evaluator owns actual new-day P/Q, network and voltage. It creates a planning forecast by adding the original 5% multiplicative noise to actual P/Q. Optimizers receive only that forecast, historical observations, frozen candidates and resource contracts. Actual execution uses the original network, 0.5% P/Q and 0.05% voltage errors, and one Uniform(.9,1) response multiplier per resource for the whole day.

## Resource variants

Both variants offer the fixed 15-node, 36 kvar Q capability. The meaning is maximum available capability, with the purchased daily reserve chosen by the plan; these devices are declared independent compensators, not inferred spare PV kVA. Positive Q is absorption, negative Q injection.

- `original_p10_q36`: original seven P contracts, 10% of the registered capacities.
- `expanded_p10_q36`: append seven registered P contracts at 301/303/304, selected from frozen candidate P-response differences and registration before outcomes. Preserve 10% participation and all original time/energy constraints. These additions are PV at 301/303/304, thermal load at 301/303/304 and EV at 304. Total P nameplate contract rises from approximately 12.455 to 16.624 kW; the actually available time-dependent envelope is reported separately.

Retain the conservative PV net-export proxy, including its zero availability where a node has no net export. Registered PV alone does not establish device-side curtailment headroom. No fabricated PV generation profile or compulsory active purchase is added. Original seven P resources stay first; new P resources append; the original 15 Q resources retain their independent response stream and order. Q rotation directions keep their original identity-based ordering despite the added P columns.

## Prices and policies

All strategies are evaluated under the same illustrative two-part bill:

`C_total = C_activation(u) + kappa_P sum s_P + kappa_Q sum s_Q`,

where `s_r = max_h |u_hr|`, `kappa_P = 0.5 yuan/(kW service-day)` and `kappa_Q = 0.5 yuan/(kvar service-day)`. These are separately dimensioned example bids for the declared resource availability window, not calibrated local-market prices. Q offers [-s,s] and pays once for the one-sided rating; PV remains one-directional. Activation prices are unchanged, including Q at 0.05 yuan/kvarh. Reserve payment is based on the requested plan and is not reduced when actual delivery is below the request. Device investment, converter auxiliary consumption and network losses are not silently included in these payments; network losses are separately reported.

The activation-only optimizer minimizes the activation part; the capacity optimizer minimizes the full bill, adding only peak-capacity variables and linear inequalities. Solver bounds keep their own objective units; an activation bound is not relabeled as a total-objective bound. Both retain the same nonconvex evidence constraints.

For each resource/forecast, generate the existing current-forecast LP seed portfolio with a 2-second setting. First solve the activation objective for 150 seconds to produce a common anchor. Then give activation refinement and capacity procurement 300 seconds each, with the identical seed pool plus that anchor. Each method's reported computation includes the shared preparation and anchor cost, approximately 450 seconds plus recorded overhead; actual computation shares the first stage once. Use one solver thread and at most four forecast workers. Log model time, raw/native bounds, original-form feasibility, initialization and retained points. No previous 300/900-second quote or execution is selected as a new seed.

Evaluate six sealed policies:

1. No purchase.
2. The 150-second anchor, including explicit zero fallback if unresolved.
3. Refined activation-only plan.
4. Stable activation: retain the same-forecast feasible anchor unless refinement saves at least 0.01 yuan in activation fee. This currency-resolution rule is fixed before the new results. If no feasible anchor exists, report that fact and use the refined plan if available.
5. Capacity plus activation plan.
6. Candidate-independent rotation scaled to the capacity plan's total planned bill, with any insufficient pattern capacity reported.

All six policies pay the same declared capacity and activation prices in evaluation. The stable rule has no additional optimization; its point, reference, threshold and applicable activation-objective bound are recorded. Plan selection never reads execution or correctness labels.

## Development and confirmation

- Development seed 2026111415: four new forecasts, 32 paired execution paths each, both resource variants, six policies: 1536 execution records.
- Confirmation seed 2026111416: four further forecasts, 64 paired paths each, both resource variants, six policies: 3072 execution records.

All policies, prices, resource rules, solver budgets and success criteria are fixed here. Confirmation runs after the development integrity checks pass, without changing those settings based on development outcomes. These are new forecast/error streams on one already known physical day; neither stage creates new independent network/day cases. Resource variants share the same worlds. Preserve all losses, errors, timeouts, zero fallbacks and negative evidence changes.

Primary outcomes are native internal-topology correctness, new/lost correct paths, and wrong topology with relative weight >=.9. Report full-response-support exactness separately. Also record mean true weight, clean-AC and measured log-odds increments, model-bias/noise decomposition, requested/reserved/delivered P and Q, component fees, voltage, current, power balance and losses. Record whether any P was actually used rather than assuming expansion forces participation. A favorable result should improve identification without increasing wrong high-weight outputs; costs and adverse paths remain visible even if this criterion fails.

## Integrity and preservation

Freeze code/input/environment/protocol hashes before each stage. Seal every action and capacity vector before execution. Independently recompute full-native selection, paired streams, resource feasibility, purchased peaks and settlement, and stored AC balance/voltages. Check both preparation and solve timings. Keep raw P/Q/V, fitted numbers, individual quote/action trajectories and all solver logs under ignored outputs. Public commits contain isolated code, protocol, source notes and aggregate reports. Preserve completed results; never restart them as new confirmation.
