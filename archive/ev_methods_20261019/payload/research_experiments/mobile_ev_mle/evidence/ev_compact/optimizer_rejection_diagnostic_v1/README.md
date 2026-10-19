# Optimizer stability of four saved false-rejection events

The four already observed individual-policy events were refitted with the same saved H+actual-future observations, Gaussian meter model, calibrated biases, fixed root voltages, candidate bank, R/X box and common-source interface 1. No new history, noise, EV action or decision was generated. All four candidates were retained for each event.

Each candidate used its original full joint physical state as the first start and seven additional seeded R/X perturbations. Budget: eight starts, max_nfev=500 per start, tol=1e-10, exact linear algebra; original evaluation used two starts and max_nfev=250. The original feasible state remains an explicit incumbent even if the fitter's standard interior clipping changes an R/X boundary coordinate by about 1e-10. No future truth is used in initialization or fitting.

`gap` below is the saved true candidate SSE minus the smallest candidate SSE. Expanded-search values take the best independently feasible attained state including the original state. Original thresholds are shown only to locate the four recorded events; the new values are not substituted into the calibrated procedure.

| History / method | Original gap | Expanded-search gap | Gap change | Original threshold |
|---|---:|---:|---:|---:|
| 989201 / none | .884747792262 | .884747792019 | −2.44e−10 | .769763500085 |
| 989205 / charge_fixed | 1.079152273523 | 1.079152273520 | −3.04e−12 | .619737703543 |
| 989210 / none | .498201394617 | .498201394607 | −1.03e−11 | .091759539899 |
| 989211 / none | .442692377313 | .442692377311 | −2.05e−12 | .384898283982 |

Across all 16 candidates, the largest attained SSE improvement is 7.40e-10. All 128 local software stops and all 16 returned best fits report success. Maximum returned optimality is .00323670. These four events are numerically stable under the specified larger search: the observed changes are far smaller than their original excess over the thresholds. This diagnostic supplies no evidence that the original limited local search caused these four false rejections. It does not establish global MLEs or numerical stability for all histories.

The run used 2,008 fit-internal AC calls; solver time was about 15.5 seconds and total instrumented/replay time about 45.8 seconds on the recorded machine. `protocol.json` binds all 72 frozen scientific sources, diagnostic helper hashes, original raw hashes, risk record, configuration and seeds. All remained unchanged. Raw, thresholds, decisions, policy and evaluation counts are preserved.

Independent backward-current/forward-voltage arithmetic replays the 16 original states plus 272 new/returned saved states (288 total), checks the exact asset R/X box and complete P/Q/V/root-meter Gaussian residuals and loglikelihood, and checks the CSV costs and retained incumbent. The fitted source remains a latent state constrained through the common physical interface and root meters; there is no extra source telemetry in these cases. `review_full.json` records this separate saved-data review; it invokes no fitting or selection. Detailed restart states, initial clipping shifts, R/X, stopping status, optimality and costs are in the four compressed condition files. `candidate_comparison.csv` is the concise 16-row comparison. The mathematical scope is given in `ev_compact_rejection_optimizer_diagnostic_math.md`.
