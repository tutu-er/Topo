"""N4, probing an open node question (experiments/node_miqp/README.md).

After a short history (2 days), the node MIQP at the wide junction proposes the best union S*. At every hourly gate
of one extra day the competing trees are the flat junction T, T + S* and T + S' for every union S' that differs from
S* in one child: each such rival is one block's membership question. S* is kept current: one-child moves at every
gate while one scores better, and the node MIQP again every 6 hours. The operator may buy, each hour, one block of
active power at one participating meter (a home battery or an EV charger through an aggregator: 40% of meters, up to
3 kW either way). The block is chosen by eq. (15) of the draft:

    J(u) = psi * sum_k [kappa(m - e_k, q_k(u)) - kappa(m - e_k, q_k(0))] - c(u),

over the rivals k of the leading tree with gap e_k < m (kappa: the expected evidence, counted up to what is still
needed; q_k: the predicted separation, eq. 13), bought only when J > 0. psi = V / m with V = 2000 CNY per resolved
question, the value fixed in the earlier hourly-probing study (2026-11-23). Prices 1.2 CNY/kWh up, 4 CNY/kWh down, at
most 20 CNY per hour and 100 CNY per day; every tree's predicted voltages must stay in [0.96, 1.035] pu.

Arms (paired: same loads, same meter noise in every hour): passive (no block), greedy (eq. 15), random (a random
feasible block in the hours the greedy arm bought). At the end of the day the node MIQP and its flip test run again on
all the data; the decision is right when it names the hidden segment (present, right union) or certifies its absence,
wrong when it names a wrong union or a segment that is not there, and open otherwise.

The tree around the junction is the true one (blocks known): the test isolates the node question.

Run from topology_study:
    python -m ptid.runs.node_probe run --workers 8       # experiments/node_miqp/probe.jsonl (resumable)
    python -m ptid.runs.node_probe summary
    python -m ptid.runs.node_probe run --seeds 21 --r 0.008 --excitation 0.1 --out probe_try.jsonl   # one input
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

import numpy as np

from ptid.method import probe
from ptid.runs.node_study import (BASE_KW, M, Feeder, Simulator, draw_shared, feeder_dict, join, model_of, node_scan,
                                  read_rows, score, truth, widest, write_row)

OUT = Path(__file__).resolve().parents[2]/"experiments"/"node_miqp"/"probe.jsonl"
VALUE = 2000.                       # CNY per resolved question (earlier study's locked value)
PS = probe.ProbeSettings(psi=VALUE/M)
R_LEVELS = (0., .006, .010, .014)
EXCITATION = (.10,)
HISTORY = 192
SEEDS = tuple(range(21, 33))
SHARE, FLEX_KW = .4, 3.


def participants(f):
    rng = np.random.default_rng([20261128, 5, f.n, f.seed])
    room = np.where(rng.random(f.n) < SHARE, FLEX_KW/BASE_KW, 0.)
    return room, room                       # up, down headroom (pu)


def union(kids, S):
    return tuple(sorted(i for a in S for i in kids[a]))


def hypotheses(model, flat, kids, S, rescan, seconds):
    """The competing trees at a gate: the flat junction, the leading union S (child indices) and every union that
    differs from it in one child. S is first improved: by the node MIQP when `rescan`, then by one-child moves while
    one scores better. Returns (S, {name: lines})."""
    d = len(kids)
    ok = lambda U: 2 <= len(U) <= d-1
    if rescan:
        best = node_scan(model, flat, seconds, flip=False)
        if best["line"] is not None:
            S = frozenset(a for a, kid in enumerate(kids) if set(kid) <= set(best["line"]))
    while True:
        moves = [S ^ {a} for a in range(d) if ok(S ^ {a})]
        scored = {U: score(model, (*flat, union(kids, U))) for U in [S, *moves]}
        top = min(scored, key=scored.get)
        if top == S:
            break
        S = top
    trees = {"flat": flat, "S": (*flat, union(kids, S))}
    for a in range(d):
        if ok(S ^ {a}):
            trees[f"S^{a}"] = (*flat, union(kids, S ^ {a}))
    return S, trees


def band_ok(preds, u, fc, sigma2):
    """Every tree's predicted V0^2 - V^2 stays inside the voltage band (with z sigma_V of margin) under order u,
    and the PCC stays within the transformer rating."""
    sv = np.sqrt(sigma2)/2
    lo, hi = fc["v0sq"]-(PS.band[1]-PS.z*sv)**2, fc["v0sq"]-(PS.band[0]+PS.z*sv)**2
    if fc["pcc"]+u.sum() > PS.transformer:
        return False
    return all(np.all((t["y"]+t["dy"]@u >= lo) & (t["y"]+t["dy"]@u <= hi)) for t in preds)


def choose_order(model, lead, rivals, now, fc, up, down, budget, meters, ratio=False):
    """Eq. (15) over orders of at most `meters` participating homes in one hour, built greedily: the best one-home
    offer, then the home whose offer raises J most, while J rises. meters=1 is the draft's one-meter block.
    ratio=True ranks the offers by their expected evidence per CNY (psi * value gain / price) instead of by J, still
    adding a home only when its value gain exceeds its price: with the daily cap binding, ranking by J spends the cap in
    the first hours on the dearest offers. Returns (J, u or None, cost, homes)."""
    seps = [(now[k], *probe.separation(lead, rivals[k], model.sigma2)) for k in rivals]
    value = lambda u: sum(probe.term(s, (d0+G@u)@S@(d0+G@u), M, PS.variant) for s, d0, G, S in seps)
    zero = np.zeros(model.n)
    passive = value(zero)
    offers = list(probe.feasible_blocks(model, lead, rivals, fc, up, down, budget, BASE_KW, PS))
    preds = [lead, *rivals.values()]
    u, cost, J, homes = zero.copy(), 0., 0., []
    for _ in range(meters):
        best, now_value = None, value(u)
        for i, a, c in offers:
            if i in homes or cost+c > budget+1e-9:
                continue
            v = u.copy()
            v[i] += a
            if not band_ok(preds, v, fc, model.sigma2):
                continue
            gain = PS.psi*(value(v)-now_value)
            Jv = PS.psi*(value(v)-passive)-(cost+c)
            if gain <= c or Jv <= J:
                continue
            rank = gain/c if ratio else Jv
            if best is None or rank > best[0]:
                best = (rank, Jv, i, a, c)
        if best is None:
            break
        _, J, i, a, c = best
        u[i] += a
        cost += c
        homes.append(int(i))
    return (J, u, cost, homes) if homes else (0., None, 0., [])


def random_order(model, lead, rivals, fc, up, down, budget, count, rng):
    """`count` random one-home offers at distinct homes, within the budget and the voltage band (50 tries)."""
    offers = list(probe.feasible_blocks(model, lead, rivals, fc, up, down, budget, BASE_KW, PS))
    preds = [lead, *rivals.values()]
    for _ in range(50):
        if not offers:
            break
        pick = rng.permutation(len(offers))
        u, cost, homes = np.zeros(model.n), 0., []
        for k in pick:
            i, a, c = offers[k]
            if i in homes or cost+c > budget+1e-9:
                continue
            v = u.copy()
            v[i] += a
            if band_ok(preds, v, fc, model.sigma2):
                u, cost, homes = v, cost+c, homes+[int(i)]
            if len(homes) == count:
                break
        if len(homes) == count:
            return u, cost, homes
    return None, 0., []


ARMS = {"passive": None, "greedy1": 1, "greedy3": 3, "random3": 3}


def day(f, sim, hist, flat, kids, S, arm, plan=None, seconds=60.):
    """One probing day from the leading union S. plan: {hour: homes} of the paired greedy arm (for random3).
    Returns the data with the day, and the log of the gates."""
    up, down = participants(f)
    data, spent, log, settled = hist, 0., [], 0
    rng = np.random.default_rng([20261128, 6, f.seed, f.n])
    for h in range(24):
        model, _ = model_of(data)
        S, trees = hypotheses(model, flat, kids, S, h > 0 and h % 6 == 0, seconds)
        H = {k: score(model, t) for k, t in trees.items()}
        lead = min(H, key=H.get)
        gaps = {k: H[k]-H[lead] for k in H if k != lead}
        settled = settled+1 if all(g >= M for g in gaps.values()) else 0
        entry = dict(hour=h, lead=lead, S=sorted(S), gaps=gaps, order=None, cost=0., J=None)
        budget = min(PS.cap_hour, PS.cap_day-spent)
        u = None
        wanted = ARMS[arm] if not arm.startswith("random") else len((plan or {}).get(h, []))
        if wanted and settled < 2 and budget > 1e-6:
            fc = probe.forecast(data, data["P"], data["Q"], f.hours+h)
            pred = {k: probe.prediction(model, t, fc) for k, t in trees.items()}
            rivals = {k: pred[k] for k, g in gaps.items() if g < M}
            if rivals:
                if arm.startswith("greedy"):
                    now = {k: dict(high=gaps[k], kind="inside") for k in rivals}
                    J, u, cost, homes = choose_order(model, pred[lead], rivals, now, fc, up, down, budget, wanted,
                                                     ratio=arm == "greedy3")
                    entry["J"] = float(J)
                else:
                    u, cost, homes = random_order(model, pred[lead], rivals, fc, up, down, budget, wanted, rng)
                if u is not None:
                    entry.update(order={int(i): float(u[i]) for i in homes}, cost=float(cost))
        spent += entry["cost"]
        log.append(entry)
        data = join(data, sim.hours(f.hours+h, 1, u))
    return data, log


def same(line, target):
    return line is not None and target is not None and tuple(line) == tuple(target)


def verdict(scan, target):
    if scan["decision"] == "present":
        return "right" if same(scan["line"], target) else "wrong"
    if scan["decision"] == "absent":
        return "right" if target is None else "wrong"
    return "open"


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
    plan = None
    for arm in ARMS:
        data, log = day(f, sim, hist, flat, kids, S0, arm, plan, seconds)
        if arm == "greedy3":
            plan = {e["hour"]: list(e["order"]) for e in log if e["order"]}
        end_model, _ = model_of(data)
        end = node_scan(end_model, flat, seconds)
        row["arms"][arm] = dict(end=end, verdict=verdict(end, target), leader=leader(end, target),
                                cost=sum(e["cost"] for e in log), hours=sum(e["order"] is not None for e in log),
                                homes=sum(len(e["order"] or {}) for e in log),
                                kwh=sum(abs(a) for e in log for a in (e["order"] or {}).values())*BASE_KW, log=log)
    row["seconds"] = time.perf_counter()-t0
    return row


def leader(scan, target):
    """Accuracy without the margin: right when the best union pays and is the segment's grouping, or, without a
    segment, when no union pays."""
    pays = scan["gain"] is not None and scan["gain"] > 0
    if target is None:
        return "right" if not pays else "wrong"
    return "right" if pays and same(scan["line"], target) else "wrong"


def instances(seeds, r_levels, excitation, hours):
    for e in excitation:
        for r in r_levels:
            for seed in seeds:
                yield Feeder(blocks=12, per=2, shared=draw_shared(12, seed) if r > 0 else (), r_shared=r,
                             excitation=e, hours=hours, seed=seed)


def run(args):
    out = Path(args.out)
    done = {r["feeder"]["name"] for r in read_rows(out)}
    todo = [f for f in instances(args.seeds, args.r, args.excitation, args.hours) if f.name not in done]
    print(f"{len(todo)} inputs to run ({len(done)} done) -> {out}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(one, f, args.seconds): f for f in todo}
        for job in as_completed(jobs):
            row = job.result()
            write_row(out, row)
            a = row["arms"]
            print(f"{row['feeder']['name']}: start {row['start_verdict']}; "
                  + ", ".join(f"{k} {v['verdict']}/{v['leader']} ({v['hours']} h, {v['cost']:.0f} CNY)"
                              for k, v in a.items()) + f"; {row['seconds']:.0f}s", flush=True)


def exists(scan, target):
    """Whether a hidden line exists at the junction, decided at margin M without naming its union: right when some
    union beats the flat junction by M and there is a segment, wrong when there is none; open otherwise."""
    if scan["gain"] is not None and scan["gain"] >= M:
        return "right" if target is not None else "wrong"
    return "open"


def overlap(scan, target):
    """Jaccard similarity of the best union (if it pays) with the segment's grouping."""
    if target is None or scan["line"] is None or scan["gain"] is None or scan["gain"] <= 0:
        return 0.
    a, b = set(scan["line"]), set(target)
    return len(a & b)/len(a | b)


def paired(g, arm, other, key):
    better = sum(r["arms"][arm][key] == "right" and r["arms"][other][key] != "right" for r in g)
    worse = sum(r["arms"][arm][key] != "right" and r["arms"][other][key] == "right" for r in g)
    p = f", sign test p = {sign_p(better, worse):.3g}" if better+worse else ""
    return f"{arm} vs {other} ({key}): right only with {arm} {better}, only with {other} {worse}{p}"


def summary(args):
    rows = read_rows(args.out)
    for r in rows:                                   # the verdicts again, from the saved scans
        r["start_verdict"] = verdict(r["start"], r["target"])
        for a in r["arms"].values():
            a["verdict"], a["leader"] = verdict(a["end"], r["target"]), leader(a["end"], r["target"])
            a["exists"], a["overlap"] = exists(a["end"], r["target"]), overlap(a["end"], r["target"])
    print(f"{len(rows)} inputs from {args.out}")
    print("decided = the node decision at margin 4.6 with its flip test (right: names the segment, or certifies none);"
          "\nleader = the best union without the margin (right: it pays and is the segment, or none pays without one)\n")
    opened = [r for r in rows if r["start_verdict"] == "open"]
    groups = [("all inputs", rows), ("open after the history", opened)]
    extra = [r for r in opened if r["feeder"]["seed"] not in SEEDS]
    if extra:      # inputs added after the first set's results (seeds outside 21-32), reported apart as well
        groups += [("open after the history, first set (seeds 21-32)",
                    [r for r in opened if r["feeder"]["seed"] in SEEDS]),
                   ("open after the history, added set (other seeds)", extra)]
    for title, g in groups:
        print(f"{title}: {len(g)} inputs (segments {sum(r['target'] is not None for r in g)}, "
              f"controls {sum(r['target'] is None for r in g)})")
        print("| arm | union decided right | decided wrong | open | existence decided right | existence wrong | "
              "leader right | overlap with segment (mean) | probing hours per day | homes per day | kWh per day | "
              "CNY per day |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        seg = [r for r in g if r["target"] is not None]
        for arm in ARMS:
            v = [r["arms"][arm]["verdict"] for r in g]
            x = [r["arms"][arm]["exists"] for r in g]
            mean = lambda k: np.mean([r["arms"][arm][k] for r in g]) if g else 0.
            ov = np.mean([r["arms"][arm]["overlap"] for r in seg]) if seg else float("nan")
            print(f"| {arm} | {v.count('right')} | {v.count('wrong')} | {v.count('open')} | {x.count('right')} | "
                  f"{x.count('wrong')} | {sum(r['arms'][arm]['leader'] == 'right' for r in g)} | {ov:.2f} | "
                  f"{mean('hours'):.1f} | {mean('homes'):.1f} | {mean('kwh'):.1f} | {mean('cost'):.1f} |")
        for arm, other in (("greedy3", "passive"), ("greedy3", "random3"), ("greedy1", "passive"),
                           ("greedy3", "greedy1")):
            print(paired(g, arm, other, "verdict"))
        for arm, other in (("greedy3", "passive"), ("greedy3", "random3"), ("greedy1", "passive")):
            print(paired(g, arm, other, "exists"))
        print(paired(g, "greedy3", "passive", "leader"))
        print()
    print("By segment (open after the history): decided right / wrong")
    print("| r (ohm) | inputs | true evidence (median) | " + " | ".join(ARMS) + " | greedy3 CNY |")
    print("| ---: | ---: | ---: | " + " | ".join("---" for _ in ARMS) + " | ---: |")
    open_rows = groups[1][1]
    for rr in sorted({r["feeder"]["r_shared"] for r in open_rows}):
        g = [r for r in open_rows if r["feeder"]["r_shared"] == rr]
        cell = lambda arm: (f"{sum(r['arms'][arm]['verdict'] == 'right' for r in g)} / "
                            f"{sum(r['arms'][arm]['verdict'] == 'wrong' for r in g)}")
        ev = [r["true_evidence"] for r in g if r["true_evidence"] is not None]
        print(f"| {rr:g} | {len(g)} | {np.median(ev) if ev else float('nan'):.1f} | "
              + " | ".join(cell(a) for a in ARMS) + f" | {np.mean([r['arms']['greedy3']['cost'] for r in g]):.1f} |")


def sign_p(a, b):
    """Two-sided exact sign test (McNemar on discordant pairs)."""
    from math import comb
    n, k = a+b, min(a, b)
    return min(1., 2*sum(comb(n, i) for i in range(k+1))/2**n)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("run", "summary"))
    p.add_argument("--out", default=str(OUT))
    floats = lambda s: tuple(float(x) for x in s.split(","))
    p.add_argument("--seeds", type=lambda s: tuple(int(x) for x in s.split(",")), default=SEEDS)
    p.add_argument("--r", type=floats, default=R_LEVELS)
    p.add_argument("--excitation", type=floats, default=EXCITATION)
    p.add_argument("--hours", type=int, default=HISTORY, help="history before the probing day")
    p.add_argument("--seconds", type=float, default=60., help="node MIQP time limit per solve")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    run(args) if args.command == "run" else summary(args)


if __name__ == "__main__":
    main()
