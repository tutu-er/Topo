import re
p = r"C:\0-workspace\Topo\repo\rnj_wzzt_dr\experiments\hourly_probing_20261118\DEVELOPMENT_RUNS.md"
s = open(p, encoding="utf-8").read()
start = s.index("### 11. Revised step 1: does the bank contain the truth? (PARTIAL, stopped 2026-11-20)")
end = s.index("**A defect found and fixed during the runs (2026-11-21).**")
new = """### 11. Revised step 1: does the bank contain the truth?

`coverage.py`: 240 inputs, 60 per feeder (Soumalas11, Pengwah18, Flynn16, Paper15). 1-day histories on seeds 1–8 and 3-day histories on seeds 1–4, each under ideal meters, 0.05% bias, 0.1% bias, and 0.05% bias with one internal segment shortened to 0.01 Ω or 0.005 Ω. Truth is used only for evaluation.

The revised step 1 (`bank.identifier_families`, `bank.closure`, `model.demean_rows`) has three changes, all using only metered data:
1. **Per-meter offsets.** Every refit gets a free constant per voltage meter, fitted by per-terminal demeaning (exactly equivalent, unit-tested). The identifier also runs once on demeaned data.
2. **Keep what the 0.08 contraction deletes.** The identifier's path before output contraction joins the bank, from both the levels run and the demeaned run.
3. **BIC neighbourhood closure.** Every one-edit neighbour (add, drop or move one line) of each candidate within 6 BIC units of the leader is added and refitted (loss term, offsets, resistance floor), repeated until no candidate within the margin is left unexpanded (caps: 4 rounds, 1,500 candidates). The bank is then pruned to a BIC gap of 30.

Bank labels: A = identifier output as used so far; A+ = A plus the leader's neighbours (the first round's bank); C = A + pre-contraction paths + demeaned run; D = C closed, with offsets (the revised step 1); D0 = the same closure without offsets. "Leader" is the BIC leader of D or D0.

| History | Meter bias | Short line | n | Truth in A | A+ | C | **D** | D0 | Leader correct, D | Leader correct, D0 | Truth's BIC gap in D, median (max) | Adequacy ratio of leader, D / D0 median (D0 max) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 day | 0 | — | 32 | 29 | 32 | 30 | **32** | 32 | 30 | 31 | 0 (3.0) | 0.99 / 0.99 (1.13) |
| 1 day | 0.05% | — | 32 | 10 | 18 | 25 | **32** | 32 | 30 | 22 | 0 (3.0) | 0.99 / 1.03 (1.19) |
| 1 day | 0.1% | — | 32 | 1 | 6 | 24 | **31** | 26 | 29 | 7 | 0 (3.0) | 0.99 / 1.14 (1.42) |
| 1 day | 0.05% | 0.01 Ω | 32 | 4 | 10 | 16 | **30** | 28 | 15 | 11 | 0.4 (7.1) | 0.99 / 1.03 (1.19) |
| 1 day | 0.05% | 0.005 Ω | 32 | 1 | 8 | 4 | **31** | 26 | 1 | 1 | 5.2 (8.0) | 0.99 / 1.03 (1.19) |
| 3 days | 0 | — | 16 | 16 | 16 | 16 | **16** | 16 | 16 | 16 | 0 (0) | 1.00 / 1.00 (1.06) |
| 3 days | 0.05% | — | 16 | 10 | 15 | 16 | **16** | 16 | 16 | 9 | 0 (0) | 1.00 / 1.03 (1.12) |
| 3 days | 0.1% | — | 16 | 0 | 2 | 16 | **16** | 16 | 16 | 1 | 0 (0) | 1.00 / 1.18 (1.36) |
| 3 days | 0.05% | 0.01 Ω | 16 | 6 | 11 | 13 | **16** | 16 | 13 | 5 | 0 (5.2) | 1.00 / 1.03 (1.13) |
| 3 days | 0.05% | 0.005 Ω | 16 | 2 | 11 | 3 | **16** | 15 | 3 | 1 | 4.7 (8.6) | 1.00 / 1.03 (1.13) |
| **All** | | | **240** | **79** | **129** | **163** | **236** | **223** | **169** | **104** | | |

Per feeder, D contained the truth in 60 of 60 inputs on Soumalas11, Pengwah18 and Paper15, and 56 of 60 on Flynn16.

Findings:
- **The revised bank contained the truth in 236 of 240 inputs** against 79 of 240 for the identifier output alone and 129 for the first round's bank. All four misses are Flynn16 inputs in which a closure cap bound (round cap in 2, size cap in 2). The caps bound in only 7 of 240 inputs; raising them costs seconds.
- **With a 1-day history and ideal meters the current step 1 is nearly fine** (truth in A in 29 of 32; in A+ in 32 of 32). The first round's ambiguous and truth-missing inputs all came from 12-hour histories or meter bias.
- **Offsets are decisive under bias.** At 0.05% bias (1 day), the leader of D was correct in 30 of 32 against 22 of 32 without offsets; at 0.1%, 29 against 7. Without offsets the closure also loses the truth in some biased inputs (D0 26 of 32 at 0.1%).
- **A longer history helps.** With 3 days, D's leader was correct in every input without a short line, at all bias levels.
- **Short lines stay in the bank but are not resolved passively.** At 0.005 Ω the truth is about 5 BIC units behind a merged topology (at most 8.6), whose voltages differ by at most about 0.1%; at 0.01 Ω the leader is right in about half the inputs (15 of 32 at 1 day, 13 of 16 at 3 days).
- **Adequacy check.** With offsets the leader's ratio has a median of 0.99–1.00 in every condition. Without offsets it rises with bias (median 1.14–1.18 and up to 1.42 at 0.1%), so a ratio test against the meter class flags stress-level bias. At typical bias it stays below about 1.2, too close to 1 to detect reliably.
- **Cost.** Two identifier runs took a median of 294 s (at most 914 s). The closure with offsets took a median of 6.7 s (at most 76 s), and closed banks held 55–2,332 candidates. The probing runs cap the bank at 150 by BIC order.
- **No bank can be guaranteed to contain the truth.** The closure guarantees only that every candidate within the margin has had all of its one-edit neighbours tried, and that no cap bound. The truth is found whenever a chain of single edits through plausible candidates leads to it. A bound cap or a high adequacy ratio is the signal that it may not have been.

"""
s = s[:start]+new+s[end:]
open(p, "w", encoding="utf-8").write(s)
print("ok")
