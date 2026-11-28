"""N6, probing a root line (experiments/node_miqp/README.md): N4's probing on the main case of N5.

Twelve laterals of 3 homes each; a short root line of r ohm from the trunk end feeds 7 to 10 of them (more than half,
drawn from the seed); controls have none. After an 8-day history the node MIQP at the root junction (12 children:
4082 unions, beyond enumeration) proposes the best union S* and its flip test decides it or leaves it open. Only an
open question is probed (as in the method): one extra day, hourly gates, the competing trees are the flat junction,
T + S* and every one-lateral change of S* (node_probe.hypotheses), orders chosen by eq. (15) with psi = 2000 CNY / 4.6.

Arms (paired: same loads and meter noise in every hour):
  passive   no order;
  greedy1   the draft's rule: one home per hour, caps 20 CNY per hour and 100 per day;
  greedy3   aggregated orders of up to 3 homes per hour, ranked by evidence per CNY, the same caps (N4's best arm);
  random3   random orders at as many homes in greedy3's hours;
  greedy6   up to 6 homes per hour with raised caps, 40 CNY per hour and 200 per day (more cost for more excitation);
  random6   random orders at as many homes in greedy6's hours, the same caps.
Homes that can respond: 40%, up to 3 kW either way (a battery or an EV charger through an aggregator); prices 1.2
CNY/kWh up, 4 CNY/kWh down; every tree's predicted voltages must stay in [0.96, 1.035] pu.

Settings varied (each with root lines of r = 0.004, 0.006, 0.008 ohm and controls):
  base         voltage noise 0.05% per reading, own load variation 0.10 (loads that move together);
  noisy        voltage noise 0.10% (a lower meter class);
  varied       own load variation 0.20 (more passive excitation).

Pilot (6 inputs, base setting, r = 0.003, 0.005, 0.008, seeds 61-62; not part of the results): at 0.003 ohm the root
line's true evidence was below zero (-2: with loads that move together there is nothing to find), at 0.005 about 7
and at 0.008 about 35; 5 of 6 were open after the history; greedy6 decided 2, greedy1 1, greedy3, random and passive
none. The levels were therefore moved to 0.004, 0.006 and 0.008 ohm.

Predictions written before the full run (after the pilot): aggregated orders decide more
root lines than passive operation and than random orders of the same cost, in every setting; raised caps decide more
again where the base caps run out; no arm decides a wrong line; noisy meters leave more open, varied loads fewer.

Run from topology_study:
    python -m ptid.runs.node_probe_roots run --workers 8     # experiments/node_miqp/probe_roots.jsonl (resumable)
    python -m ptid.runs.node_probe_roots summary
    python -m ptid.runs.node_probe_roots run --settings base --r 0.005 --seeds 61 --out probe_roots_try.jsonl
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

import numpy as np

from ptid.runs.node_probe import exists, leader, overlap, paired, probe_arms, verdict
from ptid.runs.node_study import (Feeder, Simulator, draw_shared, feeder_dict, model_of, node_scan, read_rows, score,
                                  truth, widest, write_row)

OUT = Path(__file__).resolve().parents[2]/"experiments"/"node_miqp"/"probe_roots.jsonl"
LATERALS, PER, HISTORY = 12, 3, 192
SETTINGS = {"base": dict(v_noise=5e-4, excitation=.10),
            "noisy": dict(v_noise=1e-3, excitation=.10),
            "varied": dict(v_noise=5e-4, excitation=.20)}
R_LEVELS = (0., .004, .006, .008)       # after the pilot (see the docstring)
SEEDS = tuple(range(61, 69))
ARMS = ("passive", "greedy1", "greedy3", "random3", "greedy6", "random6")
CAPS = {"greedy6": (40., 200.), "random6": (40., 200.)}
VALUE = 2000.


def root_size(seed):
    return 7+seed % 4                       # 7 to 10 of the 12 laterals


def instances(settings, r_levels, seeds):
    for name in settings:
        for r in r_levels:
            for seed in seeds:
                yield Feeder(blocks=LATERALS, per=PER, r_shared=r, hours=HISTORY, seed=seed,
                             shared=draw_shared(LATERALS, seed, root_size(seed)) if r > 0 else (), **SETTINGS[name])


def setting_of(feeder):
    return next(k for k, s in SETTINGS.items()
                if s["v_noise"] == feeder["v_noise"] and s["excitation"] == feeder["excitation"])


def one(f, seconds):
    t0 = time.perf_counter()
    sim = Simulator(f)
    hist = sim.history()
    model, load = model_of(hist)
    true, flat, target = truth(f)
    start = node_scan(model, flat, seconds)
    _, kids = widest(model, flat)
    S0 = frozenset(a for a, kid in enumerate(kids) if start["line"] and set(kid) <= set(start["line"]))
    row = dict(feeder=feeder_dict(f), load_modelled=load, target=target,
               true_evidence=None if target is None else score(model, flat)-score(model, true),
               start=start, start_verdict=verdict(start, target), arms={})
    if row["start_verdict"] == "open":      # a decided question is not probed
        row["arms"] = probe_arms(f, sim, hist, flat, kids, S0, target, ARMS, seconds, CAPS)
    row["seconds"] = time.perf_counter()-t0
    return row


def run(args):
    out = Path(args.out)
    done = {r["feeder"]["name"] for r in read_rows(out)}
    todo = [f for f in instances(args.settings, args.r, args.seeds) if f.name not in done]
    print(f"{len(todo)} inputs to run ({len(done)} done) -> {out}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(one, f, args.seconds): f for f in todo}
        for job in as_completed(jobs):
            row = job.result()
            write_row(out, row)
            print(f"{row['feeder']['name']}: start {row['start_verdict']}; "
                  + ", ".join(f"{k} {v['verdict']} ({v['hours']} h, {v['cost']:.0f} CNY)"
                              for k, v in row["arms"].items()) + f"; {row['seconds']:.0f}s", flush=True)


def table(g, arms):
    """The arms' results over the open inputs g (rows of a markdown table), and the paired comparisons."""
    seg = [r for r in g if r["target"] is not None]
    print("| arm | exact line decided right | decided wrong | existence decided right | leader right | "
          "overlap with the line (mean) | hours with orders per day | homes per day | kWh per day | CNY per day | "
          "CNY per extra decided line |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    base = sum(r["arms"]["passive"]["verdict"] == "right" for r in g)
    for arm in arms:
        v = [r["arms"][arm]["verdict"] for r in g]
        mean = lambda k: np.mean([r["arms"][arm][k] for r in g])
        extra = v.count("right")-base
        per_line = f"{mean('cost')*len(g)/extra:.0f}" if arm != "passive" and extra > 0 else "-"
        print(f"| {arm} | {v.count('right')} | {v.count('wrong')} | "
              f"{sum(r['arms'][arm]['exists'] == 'right' for r in g)} | "
              f"{sum(r['arms'][arm]['leader'] == 'right' for r in g)} | "
              f"{np.mean([r['arms'][arm]['overlap'] for r in seg]) if seg else float('nan'):.2f} | "
              f"{mean('hours'):.1f} | {mean('homes'):.1f} | {mean('kwh'):.1f} | {mean('cost'):.1f} | {per_line} |")
    for arm, other in (("greedy3", "passive"), ("greedy3", "random3"), ("greedy1", "passive"), ("greedy3", "greedy1"),
                       ("greedy6", "passive"), ("greedy6", "random6"), ("greedy6", "greedy3")):
        if arm in arms and other in arms:
            print("  "+paired(g, arm, other, "verdict"))


def summary(args):
    rows = read_rows(args.out)
    for r in rows:                          # the verdicts again, from the saved scans
        r["setting"] = setting_of(r["feeder"])
        r["start_verdict"] = verdict(r["start"], r["target"])
        for a in r["arms"].values():
            a["verdict"], a["leader"] = verdict(a["end"], r["target"]), leader(a["end"], r["target"])
            a["exists"], a["overlap"] = exists(a["end"], r["target"]), overlap(a["end"], r["target"])
    opened = [r for r in rows if r["arms"]]
    arms = [a for a in ARMS if opened and a in opened[0]["arms"]]
    print(f"{len(rows)} inputs from {args.out}; {len(opened)} open after the history and probed\n")
    print("After the history (before probing): decided right / wrong / open")
    print("| setting | r (ohm) | inputs | true evidence (median) | right | wrong | open |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in [k for k in SETTINGS if any(r["setting"] == k for r in rows)]:
        for rr in sorted({r["feeder"]["r_shared"] for r in rows}):
            g = [r for r in rows if r["setting"] == name and r["feeder"]["r_shared"] == rr]
            if not g:
                continue
            ev = [r["true_evidence"] for r in g if r["true_evidence"] is not None]
            v = [r["start_verdict"] for r in g]
            print(f"| {name} | {rr:g} | {len(g)} | {np.median(ev) if ev else float('nan'):.1f} | {v.count('right')} | "
                  f"{v.count('wrong')} | {v.count('open')} |")
    print("\nexact line decided = the root line named and every one-lateral change beaten by 4.6 (node flip test);"
          "\nexistence = some union beats the flat junction by 4.6; leader = the best union without the margin.\n")
    groups = [("All open inputs", opened)]
    groups += [(f"Setting {k}", [r for r in opened if r["setting"] == k]) for k in SETTINGS]
    groups += [(f"r = {rr:g} ohm (all settings)", [r for r in opened if r["feeder"]["r_shared"] == rr])
               for rr in sorted({r["feeder"]["r_shared"] for r in opened})]
    for title, g in groups:
        if not g:
            continue
        print(f"{title}: {len(g)} inputs ({sum(r['target'] is not None for r in g)} with a root line, "
              f"{sum(r['target'] is None for r in g)} controls)")
        table(g, arms)
        print()
    s = [r["seconds"] for r in rows]
    print(f"Seconds per input (median / max): {np.median(s):.0f} / {max(s):.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("run", "summary"))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--settings", type=lambda s: tuple(s.split(",")), default=tuple(SETTINGS))
    p.add_argument("--r", type=lambda s: tuple(float(x) for x in s.split(",")), default=R_LEVELS)
    p.add_argument("--seeds", type=lambda s: tuple(int(x) for x in s.split(",")), default=SEEDS)
    p.add_argument("--seconds", type=float, default=60., help="node MIQP time limit per solve")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    run(args) if args.command == "run" else summary(args)


if __name__ == "__main__":
    main()
