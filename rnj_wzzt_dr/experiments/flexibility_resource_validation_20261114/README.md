# Distribution Flexibility Services: resource and operating checks

This isolated study checks the prior 23 kvar result, assigns node-level reactive contracts from historical load scale, and compares the updated 36 kvar scenario. The frozen identifier, fitted parameters, error laws, score, conditional probability threshold and direction protection remain unchanged. [PROTOCOL.md](PROTOCOL.md) fixes the comparison before new results. Raw P/Q/V, historical load statistics, fitted coefficients and action/price trajectories stay in ignored `outputs/`.

## Existing operating results

The independent stored-array audit covers all 480 earlier strategy executions, including zero fallbacks. All 480 pass physical algebra, declared resource constraints and fee checks. Each path has 96 samples at 26 buses: 18 measured terminals, seven hidden nodes and the root. The full 1,198,080 voltage values range from 1.0014948 to 1.0233905 pu, with no excursion outside the study's 0.95--1.05 reporting band.

For the 32 prior 23 kvar absolute-plus-guard executions, all-bus voltage is 1.0015382--1.0233905 pu, hidden-bus voltage 1.0029845--1.0229040 pu, maximum branch current 60.443714 A, and maximum root apparent power 135.195524 kVA. The 200 kVA quantity is a normalization base. The source network has no branch ampacity or transformer nameplate, so thermal loading and overload status remain unavailable. Independent KCL and complex-power balance maximum errors over all 480 records are 2.7732e-12 pu and 2.0148e-9 kVA.

The previous optimized 23 kvar plan buys essentially no active service: mean P movement is 3.14e-9 kWh, versus Q movement 39.340321 kvarh. The same-fee rotation uses 3.700550 kWh and 29.721833 kvarh. Q dispatch increases mean network losses from 13.382213 to 13.507205 kWh; the additional 0.124991 kWh is also visible in root imports. Compensator auxiliary consumption is not included.

The first audit attempt treated cached zero-action states as local files and had 160 path errors. The corrected audit follows the sealed cache reference and time slice; all 480 pass. Both attempts are retained, with the valid report under `outputs/flexibility_resource_operational_audit_20261114_v1/corrected_receipt_paths/`.

## Node-level capacity rule

For the existing 15 participating terminals, round the absolute training-Q 95th percentile upward to the fixed contract tiers 1, 2, 3, 5, 7.5 kvar (larger tiers exist in the implementation), and retain any larger previous contract. Only the original 192 training P/Q samples and historical registration enter this calculation. This sample quantile sets an engineering size; it is not a 95% availability guarantee. Keeping the existing 303/304 contracts at 5 kvar is an explicit continuity decision, not a value inferred from their Q quantiles.

Assume new or separately reserved independent bidirectional compensators, with contract capacity no greater than 80% of declared equipment rating; select the next 3/5/10 kvar equipment class. The 80% ceiling is a design assumption. The resulting case actually reserves at most two-thirds of each device's declared rating.

| Terminal | Q contract, +/- kvar | Example equipment, kvar |
|---|---:|---:|
| 301 | 2 | 3 |
| 302 | 2 | 3 |
| 303 | 5 | 10 |
| 304 | 5 | 10 |
| 305 | 0 | 0 |
| 306 | 2 | 3 |
| 307 | 3 | 5 |
| 308 | 2 | 3 |
| 309 | 2 | 3 |
| 310 | 2 | 3 |
| 311 | 2 | 3 |
| 312 | 2 | 3 |
| 313 | 0 | 0 |
| 314 | 3 | 5 |
| 315 | 0 | 0 |
| 316 | 2 | 3 |
| 317 | 1 | 3 |
| 318 | 1 | 3 |
| Total | 36 | 63 |

305/313/315 have no contract under the retained participation rule; additional equipment could physically be installed there. These are synthetic engineering capacities, not a measured SVG inventory. The old 23 kvar domain is contained in the new domain. Resource IDs, ordering, windows, prices and P contracts are retained, preserving paired random delivery streams.

Official manufacturer evidence supports the hardware classes: the [SMone 3 kvar product page](https://smone.pl/produkty/sm-asvg-3-04-4l-cw) specifies 400 V three-phase equipment and Modbus; the [SMone manual](https://smone.pl/files/SVG%20ASVG%20Manual%2020251212v1.6.7.pdf), table 3.1 on PDF page 5, lists 5 and 10 kvar models. The [YIY SVG page](https://www.yiyelec.com/power-quality-static-var-generator-svg-three-phase-product/) describes bidirectional inductive/capacitive compensation. Product webpages and relevant manual specification/control pages were read, not the entire manual. Ratings and communications support do not establish that a particular installed device accepts the study's arbitrary signed-Q schedule: remote control, all-day reserve and participation are explicit service assumptions.

PV registration alone is insufficient to infer this Q capability. Even a hypothetical inverter with S=1.1 P_rated has only sqrt(1.1^2-1) P_rated reactive headroom at full active output. PV and batteries may share an inverter. [SMA's Q on Demand 24/7 description](https://manuals.sma.de/STP50JP40/en-US/5810005643.html) also requires enabled night operation and grid-supplied auxiliaries for that product. The present independent-compensator scenario avoids assuming those features for every registered PV system.

## Why active procurement is zero here

Seven original P contracts remain at four terminals 306/307/309/318. They do not cover the three principal Q locations 301/303/304. PV is limited to four daytime hours; EV and hot-water moves have restoration constraints. Device registration supports a scale, while signed flexible availability remains an aggregate contract assumption.

An independent centre-radius upper bound on the near-pair J, relaxing all energy coupling and direction protection, proves the current P-only contracts insufficient: across the four current sealed forecasts, the 10% participation upper bound is 0.276569--0.290944, and the rebuilt 25% contract upper bound is 0.282141--0.296630. Both are below the buffered absolute requirement 6.732100205. The 25% PV bounds are rebuilt against the same forecast net-export proxy; eight resource-hours differ from simply multiplying the 10% limits by 2.5. This identifies available positions and model response differences as an important limitation, independent of relative P/Q prices. It does not prove all possible active-service locations are ineffective.

## New validation

The paired phase reuses four known forecasts and eight paths per forecast, with new36 and the completed old23 reference. The fresh phase fixes both resource configurations and uses four new forecasts and eight new paths each, on the same known physical load day. Both configurations receive a 300-second additional solver budget per quote in the fresh phase. Only absolute-plus-guard, no action and same-planned-fee rotation are executed; no identifier refit occurs. Old optimization costs are reported separately from current seed loading and search.

All 12 new quotations returned constraint-feasible plans and all 288 new strategy executions completed. Independent physical/resource audits and the original independent native-selection/random-stream verifier passed 96/96 paired and 192/192 fresh records, with zero errors. The paired stream/zero-observation arrays match the old run; fresh variants share identical baseline observations and delivery/meter streams. The audit includes hidden buses.

| Paths | Configuration/strategy | Internal recovery | True weight >=.9 | Wrong weight >=.9 | Mean true weight | Mean activation fee/yuan |
|---|---|---:|---:|---:|---:|---:|
| Paired 32 | No action | 18 | 0 | 1 | .500662 | 0 |
| Paired 32 | Old23, previous 180s quotes | 26 | 19 | 2 | .801322 | 1.967016 |
| Paired 32 | New36, additional 300s quotes | 26 | 24 | 1 | .834877 | 1.825866 |
| Paired 32 | New36 same-fee rotation | 19 | 0 | 1 | .512646 | 1.828386 |
| Fresh 32 | No action | 16 | 1 | 1 | .460604 | 0 |
| Fresh 32 | Old23, 300s budget | 25 | 19 | 7 | .726063 | 2.387236 |
| Fresh 32 | New36, 300s budget | 27 | 23 | 3 | .828354 | 1.789239 |
| Fresh 32 | New36 same-fee rotation | 18 | 1 | 1 | — | 1.791489 |

New36 preserves aggregate recovery on the previously used paths and improves fresh-path recovery versus old23 by two cases. It does not preserve every old correct path: paired new/old has one gain and one loss; fresh has three gains and one loss. Against zero action the paired comparison has nine gains and one loss, and fresh eleven gains with zero losses. Thus the predeclared no-loss-to-old criterion fails in both phases; the old23 configuration is retained rather than replaced as universally inferior. These are 64 error paths on one physical day, with the first 32 already used; they are not 64 independent network/day cases. Full response-support matching remains 0 in every arm because the fitted common term remains.

One paired loss is instructive: old23 true weight .884233 becomes .294453 under new36. Clean-AC true-direction evidence gain increases from 3.211328 to 4.491694, while the measured gain changes from 1.450196 to -1.456796. Shared noise arrays do not produce the same likelihood perturbation when the action changes the candidate prediction difference. All such adverse paths are retained.

| Quantity, mean true-direction increment | Paired old23 | Paired new36 | Fresh old23 | Fresh new36 |
|---|---:|---:|---:|---:|
| Actual prediction distance J | 5.824522 | 5.821372 | 6.477718 | 5.871516 |
| Combined frozen-model/AC bias term | -2.138769 | -.968647 | -3.005417 | -.132697 |
| Clean-AC log odds | 3.685753 | 4.852725 | 3.472302 | 5.738819 |
| Measured log odds | 4.272573 | 5.591543 | 3.281442 | 5.288745 |
| Paths with negative measured increment | 4/32 | 4/32 | 7/32 | 3/32 |

New36 mainly uses Q at 301/302/303/304, with P still numerically zero. Fresh mean Q delivery falls from 47.744725 to 35.784774 kvarh while mean clean evidence increases. The improvement is associated with less adverse model-bias projection, not newly accurate fitted R/X. All clean-AC gains are positive in these saved samples; measured reversals remain. Correctness statistics and the posterior-like weights are separate measures.

The 288 new executions have zero voltage excursions in the 0.95--1.05 band. The fresh whole-batch voltage range is 1.0015201--1.0233905 pu. New36 optimized paths have maximum branch current 60.443714 A and maximum root apparent power 135.195524 kVA, with fresh mean losses 13.492674 kWh versus 13.382213 without action and 13.525653 for old23. Thermal limits remain unknown as stated above; the all-bus voltage check includes seven hidden buses. Fresh optimized P delivery is 4.74e-9 kWh, numerically zero.

All 12 quotes reach TIME_LIMIT with valid lower bound zero and gap 100%; none proves minimum cost. Ten materially improve their verified initial point, one improves only 1.79e-8 yuan, and one retains its old point exactly. Actual charged time is 300.087--313.694 seconds, and all exceed the protocol's 0.05-second timing tolerance. Report the 300-second setting and the recorded elapsed time separately. Gurobi's [TimeLimit documentation](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#parameter:TimeLimit) explains that termination can require additional computations; the records here do not isolate which termination step caused each overrun. No strict wall-clock deadline is claimed. Paired resource comparisons also include extra optimization time; fresh old23/new36 is the matched-budget comparison.

The fresh P-only upper-bound check agrees with the previous conclusion: 10% contracts give J upper bounds .269685--.282819, and reconstructed 25% contracts .275246--.288485, all below 6.732100205. P alone is inadequate at the retained sites, while its marginal contribution in a different joint P/Q plan is not ruled out by this bound.

Result sources: `outputs/flexibility_resource_analysis_20261114_v1/REPORT.json`; execution roots `outputs/flexibility_resource_{paired,fresh}_20261114_v1`; independent checks `outputs/flexibility_resource_{paired,fresh}_operational_audit_20261114_v1` and `outputs/flexibility_resource_{paired,fresh}_verify_20261114_v1`. The first operational-audit attempt and all old completed experiments are preserved.

Procurement fees retain the illustrative 0.05 yuan/kvarh activation price. Full 36 kvar activation for 24 hours would cost 43.2 yuan. Actual activation, device investment, capacity reservation and auxiliary energy are different quantities; only activation is optimized in this experiment.

## Entrypoints

`resources.py` freezes node capacities and input hashes; `run.py` performs the two predeclared phases; `operational_audit.py` independently checks saved electrical states and procurement; [p_reachability.py](../flexibility_reaudit_20261114/p_reachability.py) bounds the P-only distance without optimization. `analyze.py` recomputes the full-native choices and true-direction evidence from saved observations and produces aggregate reports. The existing `reactive_flex_20261114/verify.py` replays candidate scoring, actions, costs and random streams independently. See the protocol for scope and budgets.

```powershell
python -m rnj_wzzt_dr.experiments.flexibility_resource_validation_20261114.resources --out outputs/flexibility_resource_spec_NEW
python -m rnj_wzzt_dr.experiments.flexibility_resource_validation_20261114.run --phase paired --placements outputs/flexibility_resource_spec_NEW/paired_placements.json --out outputs/flexibility_resource_paired_NEW
python -m rnj_wzzt_dr.experiments.flexibility_resource_validation_20261114.run --phase fresh --placements outputs/flexibility_resource_spec_NEW/fresh_placements.json --out outputs/flexibility_resource_fresh_NEW
python -m rnj_wzzt_dr.experiments.flexibility_resource_validation_20261114.analyze --paired-run outputs/flexibility_resource_paired_NEW --fresh-run outputs/flexibility_resource_fresh_NEW --out outputs/flexibility_resource_analysis_NEW
```

These commands reproduce the fixed protocol and seeds; repeating them does not create a new independent confirmation. Use distinct output paths and retain old completed records. Relevant seven existing resource/solver tests passed before execution, and the two completed batches passed the independent checks described above.
