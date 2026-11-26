# Sealed identification test: results (I7, opened once on 2026-11-26)

**What was tested.** The registered predecessor method, `identification/final.py` at 71684ff (PROTOCOL_ID.md), on the
sealed `test` stream: 240 cases on 60 new feeders, run once, 2026-11-25 22:02 to 2026-11-26 12:14 (14 h 12 min, nine
workers). The method, the runner and the case builder were not changed after registration (no commits to
`identification/`, `sealed_test.py` or `build.py` since 71684ff). The registered analysis
(`sealed_test analyse --stream test`) was run once; its full printout is the appendix.

**What it does not test.** The method of draft v12 (`identification_v8/`: load gate, labels by flip tests, anchored
rounds) and probing. The draft's Section VII lists the differences; this result does not carry over to them.

## Primary endpoints (PROTOCOL_ID.md section 5)

Feeder-cluster bootstrap within size class, equal class weight, Holm over H1 to H4 at 5%. F1 in points.

| | Comparison | Cases | Estimate | 95% interval | p | Holds |
| --- | --- | --- | --- | --- | --- | --- |
| H1 | final method without PCC minus RNJ (blind, 0.04), no source | 60 | +26.9 | [23.4, 30.4] | 0.0002 | yes |
| H2 | S2 (with PCC) minus RNJ (projected, 0.08), with a source | 180 | +29.3 | [26.0, 32.7] | 0.0002 | yes |
| H3 | S2 minus the final method without PCC, same cases | 180 | +8.7 | [6.9, 10.7] | 0.0002 | yes |
| H4 | true load place within 4.6 of the best; false alarms without a load | 180 + 60 | coverage 96.7%, false alarms 0.0% | [93.9, 98.9]; [0.0, 0.0] | 0.0002 | yes |

p = 0.0002 is the smallest value 10,000 resamples can give. The paper's claim, "better than the distance methods, and
better because it knows the unmetered load" (H1 to H3), **holds**. The location claim (H4) **holds**.

## Secondary (no decision rests on them)

- **Failures:** 0 unreadable cases and 0 method failures, for every method and part.
- **No source, all 60 cases, pooled:**
  - final method: F1 70.1 (precision 90.7, recall 57.1);
  - without the structural prior: F1 67.5;
  - WZZT-R: F1 62.5;
  - RNJ at its development tolerance: F1 38.2 (precision 34.1, recall 43.5, 15.6 false lines per case).
- **With a source, all 180 cases, pooled:**
  - S2: F1 72.1 (precision 91.5, recall 59.4);
  - S1: F1 69.6;
  - without the PCC: F1 61.2;
  - RNJ projected: F1 37.7.
- **By size:** the gap to the distance methods grows with feeder size. No source, case-F1 from small to huge:
  - final method: 77.5, 75.6, 70.9, 62.9;
  - RNJ: 60.9, 52.3, 36.7, 29.3.
- **Final method without the PCC against WZZT-R** (no source, H1 bootstrap): +5.4 points [+3.3, +7.6].
- **Exact trees:** 2 of 240 (S2, small). The method is precise but conservative: recall is 52 to 69% by size.
- **Load evidence above 4.6:**
  - by place: end 100%, middle 93%, head 50%. A load at the head is hard to tell apart from the PCC, but its place is
    still covered in 100% of cases;
  - by share: 2% → 65%, 5% → 85%, 10% → 93%.
- **Certified trees:**
  - our method: 240 of 240, both without the PCC and with it;
  - WZZT-R: 20 of 60 (0 of 30 at large and huge).
- **Time** (median, one solver thread per case):
  - small: 3 s without the PCC, 7 s with it;
  - huge: 1,909 s without, 2,100 s with;
  - WZZT-R at huge: 4,900 s.
- **Margin calibration** (pool-step lines, blind; share true): margin [0, 2) 63%, [2, 4.6) 72%, [4.6, 10) 78%,
  10 and above 90%.
  A margin above 4.6 (Bayes factor 100) is therefore **not** 99% reliable for this predecessor's local margins. The
  draft v12 labels use flip-test gaps, which are a different and more conservative quantity, but their calibration has
  to be measured (test SR3 in `paper/PROPOSALS_UNDER_TEST.md`) before "decided" can be read as roughly 1% wrong.
- **Likely lines** (blind tree, no source): 5.0 listed per case, 44% true. They hold 131 of the 477 true lines the trees
  missed.

## Run conditions

- Nine workers on a 16-thread (8-core) machine, as registered.
- During the run, a few short development scripts ran beside it, each on one core for under a minute (checks on
  development cases for the v12 draft). Their total is a few minutes, against 14 hours of run time.
- The searches stop at wall-clock limits, so the machine load is part of the method's conditions (PROTOCOL_ID.md
  section 8).

## Appendix: the registered analysis printout

```
test: 240 cases, 0 unreadable, method failures: {'blind': 0, 'source_aware': 0, 'plain': 0, 'wzzt_r': 0, 'blind_builders': 0, 'projected_builders': 0, 'likely_blind': 0, 'likely_s1': 0}

PRIMARY (feeder-cluster bootstrap within size class, equal class weight, Holm over H1-H4 at 5%)
  H1: estimate 26.9, p 0.0002, cases 60, sign_ok True, holds True, ci [23.4, 30.4]
  H2: estimate 29.3, p 0.0002, cases 180, sign_ok True, holds True, ci [26.0, 32.7]
  H3: estimate 8.7, p 0.0002, cases 180, sign_ok True, holds True, ci [6.9, 10.7]
  H4: coverage 96.7, false_alarms 0.0, p 0.0002, cases 240, sign_ok True, holds True, coverage_ci [93.9, 98.9], false_alarm_ci [0.0, 0.0]
  claim 'better than the distance methods, and better because it knows the unmetered load': HOLDS; location claim (H4): HOLDS

SECONDARY (pooled counts; no decision rests on them)
  no source, small (15)
    blind                        F1  78.7  prec  95.9  rec  66.7  case-F1  77.5  root   30/37  exact   0  false/case 0.2
    plain                        F1  75.6  prec  97.0  rec  61.9  case-F1  74.0  root   29/37  exact   0  false/case 0.1
    wzzt_r                       F1  76.3  prec  97.1  rec  62.9  case-F1  74.9  root   29/37  exact   0  false/case 0.1
    blind|rnj|0.04 (dev-best)    F1  60.8  prec  54.1  rec  69.5  case-F1  60.9  root   25/37  exact   0  false/case 4.1
    blind|nj|0.04 (dev-best)     F1  57.3  prec  51.9  rec  63.8  case-F1  57.2  root   24/37  exact   0  false/case 4.1
    blind|rg|0.16 (dev-best)     F1  38.3  prec  54.4  rec  29.5  case-F1  38.8  root   13/37  exact   0  false/case 1.7
    rnj best in hindsight: 0.16 (case-F1 62.9)
    nj best in hindsight: 0.08 (case-F1 61.2)
    rg best in hindsight: 0.08 (case-F1 39.6)
  no source, medium (15)
    blind                        F1  76.6  prec  95.3  rec  64.0  case-F1  75.6  root   53/69  exact   0  false/case 0.4
    plain                        F1  73.4  prec  96.6  rec  59.3  case-F1  72.2  root   50/69  exact   0  false/case 0.3
    wzzt_r                       F1  73.7  prec  97.4  rec  59.3  case-F1  72.4  root   50/69  exact   0  false/case 0.2
    blind|rnj|0.04 (dev-best)    F1  52.5  prec  45.8  rec  61.4  case-F1  52.3  root   40/69  exact   0  false/case 9.1
    blind|nj|0.04 (dev-best)     F1  47.3  prec  43.0  rec  52.4  case-F1  46.7  root   29/69  exact   0  false/case 8.7
    blind|rg|0.16 (dev-best)     F1  22.3  prec  37.5  rec  15.9  case-F1  22.5  root    6/69  exact   0  false/case 3.3
    rnj best in hindsight: 0.0 (case-F1 53.2)
    nj best in hindsight: 0.04 (case-F1 46.7)
    rg best in hindsight: 0.08 (case-F1 22.9)
  no source, large (15)
    blind                        F1  71.2  prec  91.8  rec  58.2  case-F1  70.9  root  81/106  exact   0  false/case 1.1
    plain                        F1  69.0  prec  92.8  rec  54.9  case-F1  68.3  root  76/106  exact   0  false/case 0.9
    wzzt_r                       F1  67.4  prec  91.6  rec  53.3  case-F1  67.2  root  74/106  exact   0  false/case 1.0
    blind|rnj|0.04 (dev-best)    F1  35.1  prec  31.1  rec  40.2  case-F1  36.7  root  40/106  exact   0  false/case 18.1
    blind|nj|0.04 (dev-best)     F1  30.5  prec  28.3  rec  33.0  case-F1  32.0  root  33/106  exact   0  false/case 17.1
    blind|rg|0.16 (dev-best)     F1  10.2  prec  20.0  rec   6.9  case-F1  10.4  root   8/106  exact   0  false/case 5.6
    rnj best in hindsight: 0.04 (case-F1 36.7)
    nj best in hindsight: 0.04 (case-F1 32.0)
    rg best in hindsight: 0.16 (case-F1 10.4)
  no source, huge (15)
    blind                        F1  65.1  prec  87.0  rec  52.0  case-F1  62.9  root   74/97  exact   0  false/case 2.7
    plain                        F1  62.7  prec  87.2  rec  48.9  case-F1  61.3  root   70/97  exact   0  false/case 2.5
    wzzt_r                       F1  51.4  prec  85.1  rec  36.8  case-F1  50.8  root   55/97  exact   0  false/case 2.2
    blind|rnj|0.04 (dev-best)    F1  29.9  prec  27.0  rec  33.5  case-F1  29.3  root   21/97  exact   0  false/case 31.0
    blind|nj|0.04 (dev-best)     F1  29.0  prec  27.4  rec  30.8  case-F1  28.5  root   22/97  exact   0  false/case 27.9
    blind|rg|0.16 (dev-best)     F1   6.2  prec  12.7  rec   4.1  case-F1   6.2  root    2/97  exact   0  false/case 9.7
    rnj best in hindsight: 0.04 (case-F1 29.3)
    nj best in hindsight: 0.04 (case-F1 28.5)
    rg best in hindsight: 0.16 (case-F1 6.2)
  no source, all (60)
    blind                        F1  70.1  prec  90.7  rec  57.1  case-F1  71.7  root 238/309  exact   0  false/case 1.1
    plain                        F1  67.5  prec  91.4  rec  53.5  case-F1  68.9  root 225/309  exact   0  false/case 0.9
    wzzt_r                       F1  62.5  prec  90.9  rec  47.6  case-F1  66.3  root 208/309  exact   0  false/case 0.9
    blind|rnj|0.0                F1  36.1  prec  30.5  rec  44.2  case-F1  42.9  root 134/309  exact   1  false/case 18.6
    blind|rnj|0.04 (dev-best)    F1  38.2  prec  34.1  rec  43.5  case-F1  44.8  root 126/309  exact   0  false/case 15.6
    blind|rnj|0.08               F1  37.9  prec  37.3  rec  38.5  case-F1  44.6  root 108/309  exact   0  false/case 12.0
    blind|rnj|0.16               F1  33.8  prec  41.3  rec  28.6  case-F1  41.0  root  74/309  exact   0  false/case 7.5
    blind|rnj|0.24               F1  28.5  prec  42.9  rec  21.4  case-F1  36.1  root  62/309  exact   0  false/case 5.3
    blind|nj|0.0                 F1  34.4  prec  29.1  rec  42.1  case-F1  39.8  root 125/309  exact   0  false/case 19.0
    blind|nj|0.04 (dev-best)     F1  35.3  prec  32.9  rec  38.2  case-F1  41.1  root 108/309  exact   0  false/case 14.4
    blind|nj|0.08                F1  33.7  prec  38.7  rec  29.9  case-F1  40.1  root  76/309  exact   0  false/case 8.8
    blind|nj|0.16                F1  23.1  prec  45.3  rec  15.5  case-F1  28.9  root  27/309  exact   0  false/case 3.5
    blind|nj|0.24                F1  11.5  prec  42.5  rec   6.6  case-F1  16.3  root   8/309  exact   0  false/case 1.7
    blind|rg|0.0                 F1  12.0  prec  13.6  rec  10.8  case-F1  17.8  root  35/309  exact   0  false/case 12.7
    blind|rg|0.04                F1  12.9  prec  16.4  rec  10.7  case-F1  19.0  root  34/309  exact   0  false/case 10.1
    blind|rg|0.08                F1  13.3  prec  19.1  rec  10.2  case-F1  19.6  root  31/309  exact   0  false/case 8.1
    blind|rg|0.16 (dev-best)     F1  13.5  prec  25.2  rec   9.3  case-F1  19.5  root  29/309  exact   0  false/case 5.1
    blind|rg|0.24                F1  11.5  prec  28.3  rec   7.2  case-F1  16.4  root  17/309  exact   0  false/case 3.4
    rnj best in hindsight: 0.04 (case-F1 44.8)
    nj best in hindsight: 0.04 (case-F1 41.1)
    rg best in hindsight: 0.08 (case-F1 19.6)
  with a source, small (45)
    s2                           F1  80.3  prec  95.6  rec  69.2  case-F1  79.5  root  93/111  exact   2  false/case 0.2
    s1                           F1  78.3  prec  96.7  rec  65.7  case-F1  77.2  root  89/111  exact   0  false/case 0.2
    blind                        F1  75.2  prec  92.2  rec  63.5  case-F1  74.2  root  85/111  exact   0  false/case 0.4
    projected|rnj|0.08 (dev-best) F1  61.4  prec  58.9  rec  64.1  case-F1  61.9  root  68/111  exact   0  false/case 3.1
    projected|nj|0.04 (dev-best) F1  57.9  prec  52.2  rec  65.1  case-F1  58.0  root  74/111  exact   0  false/case 4.2
    projected|rg|0.24 (dev-best) F1  29.5  prec  50.0  rec  21.0  case-F1  29.2  root  18/111  exact   0  false/case 1.5
    rnj best in hindsight: 0.08 (case-F1 61.9)
    nj best in hindsight: 0.08 (case-F1 61.9)
    rg best in hindsight: 0.04 (case-F1 36.4)
  with a source, medium (45)
    s2                           F1  77.8  prec  95.2  rec  65.8  case-F1  76.9  root 162/207  exact   0  false/case 0.4
    s1                           F1  75.5  prec  95.2  rec  62.6  case-F1  74.8  root 156/207  exact   0  false/case 0.4
    blind                        F1  71.1  prec  91.6  rec  58.0  case-F1  70.3  root 148/207  exact   0  false/case 0.7
    projected|rnj|0.08 (dev-best) F1  52.0  prec  50.2  rec  54.0  case-F1  51.1  root  99/207  exact   0  false/case 6.8
    projected|nj|0.04 (dev-best) F1  46.4  prec  42.3  rec  51.3  case-F1  45.9  root  86/207  exact   0  false/case 8.8
    projected|rg|0.24 (dev-best) F1  19.3  prec  40.2  rec  12.7  case-F1  19.2  root  15/207  exact   0  false/case 2.4
    rnj best in hindsight: 0.04 (case-F1 51.6)
    nj best in hindsight: 0.04 (case-F1 45.9)
    rg best in hindsight: 0.08 (case-F1 20.8)
  with a source, large (45)
    s2                           F1  72.2  prec  91.8  rec  59.5  case-F1  71.9  root 243/318  exact   0  false/case 1.1
    s1                           F1  70.0  prec  91.7  rec  56.6  case-F1  69.9  root 233/318  exact   0  false/case 1.0
    blind                        F1  63.8  prec  84.3  rec  51.3  case-F1  63.4  root 219/318  exact   0  false/case 2.0
    projected|rnj|0.08 (dev-best) F1  34.1  prec  33.4  rec  34.7  case-F1  35.4  root 100/318  exact   0  false/case 14.1
    projected|nj|0.04 (dev-best) F1  29.2  prec  26.9  rec  31.8  case-F1  30.5  root  88/318  exact   0  false/case 17.6
    projected|rg|0.24 (dev-best) F1   8.3  prec  21.4  rec   5.1  case-F1   8.4  root  21/318  exact   0  false/case 3.8
    rnj best in hindsight: 0.08 (case-F1 35.4)
    nj best in hindsight: 0.04 (case-F1 30.5)
    rg best in hindsight: 0.16 (case-F1 9.7)
  with a source, huge (45)
    s2                           F1  68.0  prec  88.8  rec  55.1  case-F1  65.7  root 220/291  exact   0  false/case 2.4
    s1                           F1  65.2  prec  88.5  rec  51.6  case-F1  62.9  root 213/291  exact   0  false/case 2.3
    blind                        F1  52.6  prec  73.5  rec  40.9  case-F1  51.3  root 184/291  exact   0  false/case 5.0
    projected|rnj|0.08 (dev-best) F1  29.3  prec  29.5  rec  29.2  case-F1  28.6  root  63/291  exact   0  false/case 23.9
    projected|nj|0.04 (dev-best) F1  26.9  prec  25.5  rec  28.3  case-F1  26.6  root  57/291  exact   0  false/case 28.2
    projected|rg|0.24 (dev-best) F1   4.2  prec  11.9  rec   2.5  case-F1   3.8  root   1/291  exact   0  false/case 6.4
    rnj best in hindsight: 0.04 (case-F1 29.7)
    nj best in hindsight: 0.0 (case-F1 26.8)
    rg best in hindsight: 0.04 (case-F1 5.5)
  with a source, all (180)
    s2                           F1  72.1  prec  91.5  rec  59.4  case-F1  73.5  root 718/927  exact   2  false/case 1.0
    s1                           F1  69.6  prec  91.5  rec  56.2  case-F1  71.2  root 691/927  exact   0  false/case 1.0
    blind                        F1  61.2  prec  81.8  rec  48.8  case-F1  64.8  root 636/927  exact   0  false/case 2.0
    projected|rnj|0.0            F1  36.7  prec  31.0  rec  44.8  case-F1  42.6  root 405/927  exact   3  false/case 18.5
    projected|rnj|0.04           F1  37.8  prec  33.7  rec  43.0  case-F1  43.7  root 386/927  exact   0  false/case 15.7
    projected|rnj|0.08 (dev-best) F1  37.7  prec  37.2  rec  38.2  case-F1  44.3  root 330/927  exact   0  false/case 12.0
    projected|rnj|0.16           F1  33.2  prec  40.4  rec  28.2  case-F1  39.9  root 226/927  exact   0  false/case 7.7
    projected|rnj|0.24           F1  27.0  prec  40.6  rec  20.3  case-F1  34.1  root 169/927  exact   0  false/case 5.5
    projected|nj|0.0             F1  33.1  prec  28.0  rec  40.5  case-F1  38.9  root 353/927  exact   0  false/case 19.3
    projected|nj|0.04 (dev-best) F1  33.9  prec  31.6  rec  36.7  case-F1  40.2  root 305/927  exact   0  false/case 14.7
    projected|nj|0.08            F1  33.0  prec  37.8  rec  29.4  case-F1  39.7  root 215/927  exact   0  false/case 9.0
    projected|nj|0.16            F1  22.5  prec  43.4  rec  15.2  case-F1  28.9  root  74/927  exact   0  false/case 3.7
    projected|nj|0.24            F1  11.3  prec  41.8  rec   6.5  case-F1  15.6  root  19/927  exact   0  false/case 1.7
    projected|rg|0.0             F1  11.6  prec  13.0  rec  10.4  case-F1  16.8  root 101/927  exact   0  false/case 12.9
    projected|rg|0.04            F1  12.3  prec  15.5  rec  10.2  case-F1  17.7  root  97/927  exact   0  false/case 10.3
    projected|rg|0.08            F1  12.5  prec  17.9  rec   9.6  case-F1  18.0  root  88/927  exact   0  false/case 8.2
    projected|rg|0.16            F1  12.3  prec  22.9  rec   8.4  case-F1  17.4  root  81/927  exact   0  false/case 5.3
    projected|rg|0.24 (dev-best) F1  10.7  prec  26.0  rec   6.7  case-F1  15.1  root  55/927  exact   0  false/case 3.5
    rnj best in hindsight: 0.08 (case-F1 44.3)
    nj best in hindsight: 0.04 (case-F1 40.2)
    rg best in hindsight: 0.08 (case-F1 18.0)

  final method without the PCC against WZZT-R (no-source cases; feeder bootstrap as for H1; no decision rests on it)
    F1 blind minus wzzt_r: +5.4 points [+3.3, +7.6] (60 cases)
    F1 plain minus wzzt_r: +2.6 points [+1.0, +4.3] (60 cases)

  by source place / share / kind: S2 case-F1, blind case-F1, coverage, evidence > 4.6
    source=end (60): S2 75.1  blind 59.3  coverage 97%  evidence > 4.6 100%
    source=head (60): S2 72.3  blind 71.2  coverage 100%  evidence > 4.6 50%
    source=middle (60): S2 73.1  blind 63.9  coverage 93%  evidence > 4.6 93%
    share=0.02 (60): S2 72.5  blind 70.1  coverage 92%  evidence > 4.6 65%
    share=0.05 (60): S2 74.0  blind 66.9  coverage 100%  evidence > 4.6 85%
    share=0.1 (60): S2 74.0  blind 57.4  coverage 98%  evidence > 4.6 93%
    kind=block (60): S2 74.1  blind 61.5  coverage 97%  evidence > 4.6 95%
    kind=house (60): S2 73.8  blind 69.7  coverage 95%  evidence > 4.6 67%
    kind=lamp (60): S2 72.7  blind 63.2  coverage 98%  evidence > 4.6 82%

  certified and time (seconds per case, one solver thread, several cases in parallel)
    small  blind        certified 60/60  median      3  p90      6  max     60
    small  source_aware certified 60/60  median      7  p90     41  max     88
    small  wzzt_r       certified 15/15  median     16  p90     49  max     58
    medium blind        certified 60/60  median     29  p90    219  max    271
    medium source_aware certified 60/60  median    121  p90    452  max    534
    medium wzzt_r       certified 5/15  median    384  p90    448  max    470
    large  blind        certified 60/60  median    417  p90   1608  max   1794
    large  source_aware certified 60/60  median    573  p90   1932  max   2077
    large  wzzt_r       certified 0/15  median    541  p90    769  max   1084
    huge   blind        certified 60/60  median   1909  p90   2064  max   2153
    huge   source_aware certified 60/60  median   2100  p90   2499  max   3271
    huge   wzzt_r       certified 0/15  median   4900  p90   7677  max   7810

  likely lines (identification/likely.py, excluded candidate lines with evidence above -4.6; secondary output)
    blind tree, no source      small   15 cases,  2.5 listed per case,  48.6% true (18/37), missed true lines on the list 18/35
    blind tree, no source      medium  15 cases,  4.3 listed per case,  40.0% true (26/65), missed true lines on the list 26/68
    blind tree, no source      large   15 cases,  4.5 listed per case,  47.8% true (32/67), missed true lines on the list 32/128
    blind tree, no source      huge    15 cases,  8.5 listed per case,  43.0% true (55/128), missed true lines on the list 55/246
    blind tree, no source      all     60 cases,  5.0 listed per case,  44.1% true (131/297), missed true lines on the list 131/477
    blind tree, with a source  small   45 cases,  2.5 listed per case,  49.1% true (55/112), missed true lines on the list 55/115
    blind tree, with a source  medium  45 cases,  4.7 listed per case,  35.7% true (76/213), missed true lines on the list 76/238
    blind tree, with a source  large   45 cases,  4.6 listed per case,  48.3% true (99/205), missed true lines on the list 99/447
    blind tree, with a source  huge    45 cases,  8.3 listed per case,  33.9% true (127/375), missed true lines on the list 127/909
    blind tree, with a source  all    180 cases,  5.0 listed per case,  39.4% true (357/905), missed true lines on the list 357/1709
    S1 tree, no source         small   15 cases,  2.9 listed per case,  41.9% true (18/43), missed true lines on the list 18/37
    S1 tree, no source         medium  15 cases,  3.8 listed per case,  47.4% true (27/57), missed true lines on the list 27/69
    S1 tree, no source         large   15 cases,  4.1 listed per case,  55.7% true (34/61), missed true lines on the list 34/132
    S1 tree, no source         huge    15 cases,  8.6 listed per case,  43.4% true (56/129), missed true lines on the list 56/247
    S1 tree, no source         all     60 cases,  4.8 listed per case,  46.6% true (135/290), missed true lines on the list 135/485
    S1 tree, with a source     small   45 cases,  2.7 listed per case,  43.4% true (53/122), missed true lines on the list 53/108
    S1 tree, with a source     medium  45 cases,  4.2 listed per case,  43.7% true (83/190), missed true lines on the list 83/212
    S1 tree, with a source     large   45 cases,  4.4 listed per case,  53.5% true (107/200), missed true lines on the list 107/398
    S1 tree, with a source     huge    45 cases,  8.7 listed per case,  42.7% true (168/393), missed true lines on the list 168/745
    S1 tree, with a source     all    180 cases,  5.0 listed per case,  45.4% true (411/905), missed true lines on the list 411/1463

  margin calibration (blind, pool-step lines): share true by margin class
    margin [-inf, 0.0): 0/2 true (0%)
    margin [0.0, 2.0): 92/147 true (63%)
    margin [2.0, 4.6): 125/174 true (72%)
    margin [4.6, 10.0): 223/286 true (78%)
    margin [10.0, inf): 1682/1863 true (90%)
```
