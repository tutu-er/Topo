# Case library (2026-11-25)

Reconstructed cases for the identification and probing studies. The aim is cases that are **reasonable first**. Regimes
where the MIQP-based method should help are included because real feeders have them, not because they favour it, and
regimes where simple methods should do as well are kept next to them.

## 1. What was wrong with the old cases

| | Old cases (4) | Real EU LV feeders, 13-25 connections (15) |
| --- | --- | --- |
| Lines leaving the transformer side by side | 2 to 4 | 1 (a trunk) |
| Nested depth (inner lines on one path) | 2 | median 10 (5-95%: 7 to 13) |
| Inner lines with more than half of the meters | **0** | median **5** (5-95%: 2 to 6) |
| Children of a branching point | 3 to 4 | median 2 |
| Resistance of inner lines | 20 to 40 milliohm | median 7 milliohm |
| Loads | one smooth daily law for all | real customers |
| Unmetered load | none | possible |
| Impedance | metadata mention a scale of 4.5 to 5.2 | physical length x conductor |

The old cases have no line close to the transformer that carries most of the meters. That is the hard part of a real
feeder and the part the MIQP is meant for, so the old cases could not test it.

**Impedance scale.** The library uses lengths in metres and catalogue conductors, with no scale factor. The old files at
scale 1 are consistent with this (0.42 ohm/km, a 70 mm2 aluminium line). At scale 5 the same 92 m line would have
2.1 ohm/km, a 16 mm2 service conductor used as a main line, and drops of 9 to 13%. Scale 1 is kept for the star cases.

## 2. Feeders

| Set | Feeders | What it is |
| --- | --- | --- |
| **Star** (named) | `soumalas11`, `paper15`, `flynn16`, `pengwah18` `_star` | the old cases, lines unchanged |
| **Street** (named) | the same four, `_street` | same meters, customers, service lines and laterals, rearranged along one street |
| **Random** | `dev_...`, `test_...`: small 10-15, medium 16-25, large 26-40, huge 41-60 meters | new street feeders |

**Star and street are a controlled pair:** they share meters, loads and service lines; only the arrangement differs. The
street rule (`feeders.street_plan`):
1. The lines that left the transformer side by side become consecutive sections of one street, in the old order and with
   their old lengths, in a main cable (150 then 95 mm2). The first one is the trunk.
2. Where more than two lines left one point, they leave one by one along the street: services first, then laterals from
   the smallest to the largest, the street itself last. New sections get lengths of real street sections.

**Random feeders** (`feeders.random_plan`): a trunk, then poles along a street. At a pole one to three connections
(75/20/5%) or, with 15% chance, a side street (at most two levels). Three styles in equal shares: urban (buried 185 and
150 mm2 cable, 60-160 customers, 40% of the connections looped in at the pole), suburban (150 and 95 mm2, 40-120
customers) and rural (overhead 95 and 50 mm2 bundles, 25-70 customers, lengths doubled). Length laws (trunk median 50 m,
section 20 m, service 7.5 m) and conductors come from the 84 real feeders with at least 4 connections. A draw is
rejected if its largest voltage drop is outside 1-7% or a conductor is overloaded. Nodes that no meter can place are
merged.

Real and random feeders share the statistics the generator was set to; section 6 shows them side by side.

## 3. Loads, physics, meters

- **Loads:** real hourly readings of single customers (8,087 records, 20 days). Customers above 14.5 kW peak (0.4%, more
  than a 63 A single-phase supply) are not used. Each hour is split over four quarter-hours at random with the hour's
  energy unchanged. The split is an assumption; hourly readings do not depend on it except through averaging.
- **Reactive power:** each customer has a mean power factor of 0.90-0.99 that drifts. The dataset's constant 0.95 would
  make Q proportional to P.
- **Physics:** AC power flow, balanced single-phase equivalent, constant power, at 15 minutes. Identifiers never see it:
  they keep the linear model with r and x per line.
- **Meters:** voltage gain error 0.05% (fixed) and noise 0.05%; P and Q gain error 0.3% (fixed; 0.2% at the PCC) and
  noise 0.5%. The PCC's V, P and Q are metered. The P/Q gain error is new; the old cases had only noise.
- **Resolution:** hourly readings (averages of the four quarter-hours) are the default, with 6 + 2 days (144 + 48
  values); quarter-hourly readings (192 values, 2 days) are available from the same case.

## 4. With and without an unmetered load

Every feeder comes as four twins with identical loads: no unmetered load, and one unmetered load at the **head** (just
after the trunk), the **middle** or the **end** of the feeder. Its mean power is 2, 5 or 10% of the metered mean power
and it is street lighting (19:00-06:00), three customers without a meter, or a pump/charger in blocks. Over nine feeders
every location meets every size with every kind exactly once. The PCC's P and Q are stored in **every** twin, so a
source-aware method must also recognise that there is no source.

This default grid is mine; the user has not chosen one.

## 5. Sets, and how a favourable case set is kept honest

| Set | Contents | Use |
| --- | --- | --- |
| `dev` | 4 star + 4 street feeders x 6 load draws, 24 random feeders; 4 twins each (288 cases) | development, tuning, comparison |
| `test` | 60 random feeders (15 per size; new feeder, customer and load random numbers); 4 twins each (240 cases) | sealed; opened once, by a registered protocol |

- Development and test share **no customer** (even and odd record numbers) and no feeder. The sealed files' SHA-256
  values are committed in `manifest_test.json`.
- **The generator was fixed from real-feeder statistics and engineering limits, and committed, before any identifier ran
  on any of its cases.** After that commit nothing in `conductors.py`, `feeders.py`, `loads.py`, `physics.py` or the sets
  in `build.py` changes because of a method's result; an error found later is fixed, logged in section 7 and rebuilt.
- Every regime is reported, also the star cases, where nothing has to be proven, and the cases without a source.
- The random feeders share the generator's assumptions, so the real feeders stay the second test family.
- Whole-map evidence (how many true lines the data support) uses the truth and the data, not an identifier; it is
  computed on development cases only.

## 6. Check of the development cases (no identifier involved)

10% / median / 90%. "Supported" lines: the BIC of the true tree gets worse when the line is removed.

| | Star | Street | Random small | medium | large | huge | Real (33) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Meters | 11/16/18 | 11/16/18 | 10/12/14 | 18/24/25 | 30/34/36 | 44/49/59 | 8/12/19 |
| Inner lines | 4/6/7 | 9/14/16 | 7/8/10 | 12/17/19 | 24/26/29 | 32/38/41 | 5/8/15 |
| Lines with more than half of the meters | **0** | 2/5/6 | 3/4/5 | 2/4/8 | 5/8/12 | 6/8/15 | 1/3/6 |
| Nested depth | 2 | 7/8/12 | 6/7/9 | 8/10/17 | 12/16/22 | 18/20/26 | 4/7/11 |
| Root-side line, milliohm | - | 1.3/9.4/21.6 | 1.1/6.7/17.9 | 1.9/4.5/16.9 | 1.9/8.5/27.3 | 2.5/7.0/24.4 | 1.7/6.7/16.8 |
| Largest drop, % | 1.2/1.7/2.3 | 1.9/4.5/6.1 | 1.6/1.9/2.1 | 1.6/2.3/4.3 | 2.2/3.1/3.9 | 3.9/4.3/6.0 | not computed |
| Linear misfit / meter noise | 0.004 | 0.028 | 0.006 | 0.008 | 0.018 | 0.035 | - |
| Supported, root-side lines | - | 76 of 90 | 13 of 17 | 14 of 21 | 31 of 43 | 34 of 48 | - |
| Supported, other lines | 138 of 138 | 123 of 186 | 9 of 24 | 31 of 56 | 32 of 100 | 58 of 147 | - |

- The lowest voltage is 0.932 pu and no line is overloaded.
- Misfit: the best linear model on the true tree against noiseless AC data. It is at most 6% of the meter noise without
  a source (so the linear model is adequate), but reaches 0.67 (median) and 13 (largest) with a source that is ignored:
  the source is the larger model error, and that is why the twins exist.
- The unmetered load stands 3.5/8.7/16.9 random reading errors above zero in every hourly PCC balance, and losses are
  0.4-2.1% of the PCC power, the same size as the smallest source; losses must therefore be estimated.
- Not every line is identifiable: many service-side lines are below what the meters resolve.

## 7. Log

- 2026-11-25, after the generator was committed (`draw.py`, `compare_dev.py` and the sealed set's manifest followed; the
  generator is unchanged): the first development comparison showed that the four named street feeders are fully binary
  (14 inner lines for 16 meters; real feeders about 12), which suits a builder that always returns a full binary tree
  (RNJ with tolerance 0 gets the best F1 there). The rule of section 5 says not to change cases after a method's result,
  so they stay, and this is reported with every street result.
- The "supported" label comes from a BIC judge of the same kind as the MIQP's own score. It favours methods that use
  that score, so recall of all true lines, precision and F1 are always reported next to it.
- 2026-11-25, before any identifier ran: the first draft was too weak (2 root-side lines for 18 meters where real
  feeders have 5) and some customers were commercial (voltages down to 0.77 pu). Changed: at most 14.5 kW per customer,
  fewer side streets, thicker service cables, equal shares of the three styles, named streets screened like the random
  ones.

## 8. Known limits

Single phase, balanced; no missing readings, no clock errors, no PV; one unmetered load; the 15-minute split is assumed;
the generator's side streets and poles are simpler than real ones; random and named feeders use one 200 kVA base and
customers from one Spanish town.
