"""Paired one- and two-day demand-response experiments on the four saved AC networks.

From topology_study with its Gurobi environment, run e.g.
    python -m ptid.runs.four_probe --case soumalas11 --days 1
Private caches and detailed AC trajectories are kept beside the study outputs.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

for env_name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[env_name] = "1"

import numpy as np

from ptid.cases import library, physics
from ptid.cases.world import BATTERY, ENROLLED, SOC, World
from ptid.method import probe as pr
from ptid.method.identify import identify, label_rows, refine
from ptid.method.labels import DECIDED, OPEN, grouping
from ptid.method.model import Fitter
from ptid.method.sets import canonical
from ptid.runs.four_feeders import CASES, history, load_feeder, meter_stream
from ptid.runs.four_records import cache_root
from ptid.runs.four_offline import digest, settings
from ptid.runs.identify_cases import f1

ST = settings()

def write_json(path, value):
    def encode(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, default=encode, allow_nan=False, indent=1)+"\n", encoding="utf-8")
    tmp.replace(path)


def atomic_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(pickle.dumps(value, protocol=5))
    tmp.replace(path)


class DeviceSchedule(World):
    """Mapped registered capacities with R1 participation, duration, availability and efficiency."""

    def __init__(self, rec):
        ctx = rec["resource_context"]
        if isinstance(ctx, (str, Path)):
            ctx = json.loads(Path(ctx).read_text(encoding="utf-8"))
        registrations = {int(row["feeder"]): row for row in ctx["resource_metadata_at_history_start"]}
        mapped = {int(row["label"]): row for row in ctx["node_mapping"]}
        self.base_kw = rec["network"]["base_kw"]
        self.bat_kw, self.ev_kw = [], []
        for label in rec["labels"]:
            row = mapped[int(label)]
            reg = registrations[int(row["feeder_id"])]
            scale = ENROLLED*float(row["scale"])
            self.bat_kw.append(scale*float(reg["batteries_kW"] or 0.))
            self.ev_kw.append(scale*float(reg["EV_chargers_kW"] or 0.))
        self.bat_kw, self.ev_kw = map(np.asarray, (self.bat_kw, self.ev_kw))
        self.bat_kwh = self.bat_kw*BATTERY[2]/BATTERY[1]
        self.soc = np.full(len(self.ev_kw), .5)

    def headroom(self, hour, pending=None):
        return super().headroom(hour%24, pending)

    def advance(self, hour, u):
        up, down = self.headroom(hour)
        if np.any(u > up+1e-10) or np.any(-u > down+1e-10):
            raise ArithmeticError(f"resource headroom exceeded at {hour}")
        self.soc = self.after(self.soc, u)
        if np.any(self.soc < SOC[0]-1e-10) or np.any(self.soc > SOC[1]+1e-10):
            raise ArithmeticError(f"battery SOC out of range at {hour}")


class ObservedDay:
    def __init__(self, rec, hours, ps):
        self.rec, self.hours, self.ps = rec, hours, ps
        self.devices = DeviceSchedule(rec)
        self.meters = meter_stream(rec)
        self.trace = []
        self.zero_voltage_error = self.zero_supply_error = 0.

    def deliver(self, hour, u):
        rec = self.rec
        start = 8*96+hour*4
        idx = slice(start, start+4)
        base = rec["true"]
        p = base["P"][idx]+u[None, :]
        q, v0 = base["Q"][idx], base["V0"][idx]
        order, v, s, _ = physics.power_flow(rec["network"], rec["labels"], p, q, v0)
        positions = {bus: j for j, bus in enumerate(order)}
        measured = v[:, [positions[i] for i in rec["labels"]]]
        refpos = {int(bus): j for j, bus in enumerate(rec["bus_ids"])}
        vbase = base["V_all"][idx][:, [refpos[b] for b in order]]
        sbase = base["S_pcc"][idx]
        if not np.any(u):
            self.zero_voltage_error = max(self.zero_voltage_error, float(np.max(np.abs(v-vbase))))
            self.zero_supply_error = max(self.zero_supply_error, float(np.max(np.abs(s-sbase))))
        vmin, vmax = self.ps.band
        violation = (v < vmin) | (v > vmax)
        base_violation = (vbase < vmin) | (vbase > vmax)
        self.devices.advance(hour, u)
        self.trace.append(dict(v=v.copy(), p=s.real.copy(), q=s.imag.copy(), soc=self.devices.soc.copy(),
                               new_v=int((violation & ~base_violation).sum()), all_v=int(violation.sum()),
                               base_v=int(base_violation.sum()),
                               new_s=int(((np.abs(s) > self.ps.transformer) & (np.abs(sbase) <= self.ps.transformer)).sum()),
                               all_s=int((np.abs(s) > self.ps.transformer).sum()),
                               base_s=int((np.abs(sbase) > self.ps.transformer).sum()),
                               order=list(order), action=u.copy()))
        r = self.meters.read(p.mean(0), q.mean(0), measured.mean(0), v0.mean(), s.mean())
        return dict(P=r["P"], Q=r["Q"], Y=r["V0"]**2-r["V"]**2, V0sq=r["V0"]**2,
                    Ppcc=r["Ppcc"], Qpcc=r["Qpcc"])


def views(model, tree, now, fc):
    rivals = {k: pr.prediction(model, s["rival"], fc) for k, s in now.items() if s["label"] == OPEN}
    return pr.prediction(model, tree, fc), rivals


def full_tree_block(model, tree, now, lead, rivals, fc, data, fp, fq, up, down, budget, base_kw, ps):
    """Legacy strong-greedy whole-tree score (V=2,000) on the common fitted flip-witness bank."""
    pool = {canonical(s["rival"]) for s in now.values() if s["label"] == OPEN}
    leader = canonical(tree)
    pool.add(leader)
    fit = Fitter(model, [c for family in pool for c in family])
    score = {family: fit.crit(family)*model.N/2 for family in pool}
    floor = min(score.values())
    active = [family for family in sorted(pool, key=lambda t: (score[t], t)) if score[family]-floor < 25.]
    if leader not in active:
        active.append(leader)
    forecasts = [pr.forecast(data, fp, fq, hour) for hour in range(24)]

    def expected_voltages(family):
        phi = [pr.prediction(model, family, f)["y"] for f in forecasts]
        return np.sqrt(np.maximum(0., np.array([f["v0sq"] for f in forecasts])[:, None]-phi))

    reps, voltage, masses = [leader], [expected_voltages(leader)], [0.]
    for family in active:
        v = voltage[0] if family == leader else expected_voltages(family)
        hit = next((i for i, row in enumerate(voltage) if np.max(np.abs(v-row)) <= .0005), None)
        weight = math.exp(-(score[family]-floor))
        if hit is not None:
            masses[hit] += weight
        elif len(reps) < 10:
            reps.append(family)
            voltage.append(v)
            masses.append(weight)
    weights = np.asarray(masses)/sum(masses)
    predictions = [pr.prediction(model, family, fc) for family in reps]
    terms = []
    for i, a in enumerate(predictions):
        for j, b in enumerate(predictions[:i]):
            d, G, W = pr.separation(a, b, model.sigma2)
            terms.append((math.sqrt(weights[i]*weights[j]), d, G, W))
    risk0 = sum(weight*math.exp(-float(d@W@d)/8) for weight, d, _, W in terms)
    best = (0., None, 0.)
    for i, signed, cost in pr.feasible_blocks(model, lead, rivals, fc, up, down, budget, base_kw, ps):
        risk = 0.
        for weight, d, G, W in terms:
            delta = d+signed*G[:, i]
            risk += weight*math.exp(-float(delta@W@delta)/8.)
        value = 2000.*(risk0-risk)-cost
        if value > best[0]+1e-12:
            best = value, signed*np.eye(model.n)[i], cost
    return best


def fixed_schedule(data, model, fp, fq, tree, now, devices, ps, m):
    """Freeze all forecasts and fitted rivals at h=0. Roll forward only planned SOC and day budget."""
    shadow = copy.copy(devices)
    shadow.soc = devices.soc.copy()
    spend, planned = {}, {}
    zero = np.zeros(model.n)
    for h in range(1, ps.hours):
        fc = pr.forecast(data, fp, fq, h)
        lead, rivals = views(model, tree, now, fc)
        up, down = shadow.headroom(h)
        day = h//24
        value, u, cost = pr.choose_block(model, lead, rivals, now, fc, up, down,
                                        min(ps.cap_hour, ps.cap_day-spend.get(day, 0.)), shadow.base_kw, ps, m)
        if u is not None:
            planned[h] = (value, u, cost)
            spend[day] = spend.get(day, 0.)+cost
        shadow.advance(h, zero if u is None else u)
    return planned


def prepared_gate(data, load, state, hour, cache, st, ps, room):
    tree, pool, rows, targets = state
    model, fp, fq = pr.setup(data, load, st)
    h = hashlib.sha256()
    for key in sorted(data):
        h.update(np.ascontiguousarray(data[key]).tobytes())
    h.update(repr((canonical(tree), canonical(pool), targets, asdict(st), load)).encode())
    key = h.hexdigest()
    path = cache/f"{key}.pkl"
    hit = path.exists()
    if hit:
        tree, pool, rows, targets, now, stop, statuses = pickle.loads(path.read_bytes())
    else:
        stop, statuses = "initial", []
        if hour:
            result = refine(model, tree, pool, st)
            tree, pool = result["tree"], result["pool"]
            rows = label_rows(result["labels"], result["anchors"])
            stop = result["stop"]
            for rnd in result["rounds"]:
                if "solve" in rnd:
                    statuses.append(dict(status=rnd["solve"]["status"], gap_closed=rnd["solve"]["gap_closed"],
                                         wide_nodes=rnd["price"]["wide_nodes"]))
        place = pr.placement_rival(model, tree) if load else None
        fc = pr.forecast(data, fp, fq, hour+1)
        if targets is None:
            lead = pr.prediction(model, tree, fc)
            targets = pr.day_targets(model, tree, rows, place, lead, fc, room, ps, st.m)
        rows = pr.relabel_outside(model, tree, pool, rows, targets, st)
        now = pr.states(model, tree, rows, place, targets, st.m)
        atomic_pickle(path, (tree, pool, rows, targets, now, stop, statuses))
    return model, fp, fq, (tree, pool, rows, targets), now, key, hit, stop, statuses


def evaluate_targets(tree, labels, reference):
    counts = Counter()
    for target, label in labels.items():
        if target == "place":
            continue
        if label != DECIDED:
            counts["unresolved"] += 1
        elif (target in tree) == (target in reference):
            counts["correct"] += 1
        else:
            counts["wrong"] += 1
    return {k: counts[k] for k in ("correct", "wrong", "unresolved")}


def identify_day(rec, data0, first, policy, ps, private, name, start_day, final_path, token):
    st = ST
    day = ObservedDay(rec, ps.hours, ps)
    devices = day.devices
    room = np.any([np.add(*devices.headroom(t)) > 0 for t in range(24)], axis=0)
    data = {k: np.asarray(v) for k, v in data0.items()}
    n = len(room)
    state = first["lines"], first["pool"], first["labels"], None
    pending = np.zeros(n)
    expenses, log = Counter(), []
    stopped, decided_before, schedule = None, False, None
    for h in range(ps.hours+1):
        start = time.perf_counter()
        model, fp, fq, state, now, key, hit, reason, statuses = prepared_gate(
            data, first["gate"]["load"], state, h, private/"gates"/name/token, st, ps, room)
        tree, pool, rows, targets = state
        meter_tree = canonical(g for g in (grouping(C, n) for C in tree) if g)
        labels = {k: s["label"] for k, s in now.items()}
        decided = all(v == DECIDED for v in labels.values())
        if h == 0 and policy == "P2" and targets:
            schedule = fixed_schedule(data, model, fp, fq, tree, now, devices, ps, st.m)
        if stopped is None and (not targets or (decided and decided_before)):
            stopped = h
        decided_before = decided
        gate = dict(hour=h, tree=meter_tree, labels={str(k): v for k, v in labels.items()},
                    gap_upper={str(k): s["high"] for k, s in now.items()}, gate_key=key,
                    status=reason, solves=statuses, trust=None, purchase=None, forecast=None)
        u = np.zeros(n)
        if h+1 < ps.hours and stopped is None and not decided and policy != "P0":
            gate["trust"] = pr.trust(model, tree, ps)
            if gate["trust"] <= ps.kappa_A:
                fc = pr.forecast(data, fp, fq, h+1)
                lead, rivals = views(model, tree, now, fc)
                up, down = devices.headroom(h+1, pending)
                budget = min(ps.cap_hour, ps.cap_day-expenses[(h+1)//24])
                args = model, lead, rivals, now, fc, up, down, budget, devices.base_kw, ps, st.m
                value, action, cost = 0., None, 0.
                if policy == "P1":
                    value, action, cost = full_tree_block(model, tree, now, lead, rivals, fc, data, fp, fq,
                                                          up, down, budget, devices.base_kw, ps)
                elif policy == "P2":
                    fixed = (schedule or {}).get(h+1)
                    if fixed is not None:
                        _, a, _ = fixed
                        i = int(np.flatnonzero(a)[0])
                        intervals = pr.feasible_ranges(model, lead, rivals, fc, up, down, budget, devices.base_kw, ps)
                        if any(idx == i and sign == np.sign(a[i]) and lo-1e-12 <= abs(a[i]) <= hi+1e-12
                               for idx, sign, _, lo, hi in intervals):
                            value, action, cost = fixed
                elif policy == "P3" or policy == "last24" and h+1 >= 24:
                    value, action, cost = pr.choose_block(*args)
                if action is not None:
                    u = action
                    i = int(np.flatnonzero(u)[0])
                    expenses[(h+1)//24] += cost
                    gate["purchase"] = dict(deliver=h+1, meter=int(rec["labels"][i]), kw=u[i]*devices.base_kw,
                                            cost=cost, value=value)
                    gate["forecast"] = dict(lead_v=np.sqrt(np.maximum(0., fc["v0sq"]-lead["y"]-lead["dy"]@u)),
                        rival_v={str(k): np.sqrt(np.maximum(0., fc["v0sq"]-r["y"]-r["dy"]@u))
                                 for k, r in rivals.items()})
        gate["seconds"] = time.perf_counter()-start
        gate["cached"] = hit
        log.append(gate)
        if h == 0:
            initial_targets = list(targets)
        if h < ps.hours:
            row = day.deliver(h, pending)
            data = {k: np.concatenate([v, np.asarray(row[k])[None]]) for k, v in data.items()}
            pending = u
        print(f"{name} {policy} {ps.cap_day:g} {h}/{ps.hours} targets={len(targets)} cache={int(hit)}", flush=True)
    truth = library.truth(rec)
    exact = library.exact_truth(rec)
    for g in log:
        mtree = set(tuple(x) for x in g["tree"])
        glabels = {k: g["labels"][str(k)] for k in initial_targets}
        g.update(reference=evaluate_targets(mtree, glabels, truth), exact=evaluate_targets(mtree, glabels, exact),
                 f1=f1(mtree, truth), f1_exact=f1(mtree, exact))
    trace = day.trace
    totals = {k: sum(t[k] for t in trace) for k in ("new_v", "all_v", "base_v", "new_s", "all_s", "base_s")}
    acfile = private/"ac"/(final_path.stem+".npz")
    acfile.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(acfile, bus_ids=trace[0]["order"], meter_ids=rec["labels"],
                        voltage=np.concatenate([t["v"] for t in trace]), p_pcc_pu=np.concatenate([t["p"] for t in trace]),
                        q_pcc_pu=np.concatenate([t["q"] for t in trace]), soc=np.stack([t["soc"] for t in trace]),
                        action_pu=np.stack([t["action"] for t in trace]), **{"observed_"+k:v for k,v in data.items()})
    for h, item in enumerate(trace):
        ordered = None if h == 0 else log[h-1]["purchase"]
        expected = np.zeros(n)
        if ordered is not None:
            expected[rec["labels"].index(ordered["meter"])] = ordered["kw"]/devices.base_kw
        if not np.allclose(item["action"], expected, rtol=0, atol=1e-12):
            raise ArithmeticError(f"Order and delivery differ at {h}")
    gatefile = private/"ac"/(final_path.stem+"_gates.json")
    write_json(gatefile, log)
    summary = [{k:g[k] for k in ("hour", "reference", "exact", "trust", "purchase")} for g in log]
    result = dict(case=name, start_day=start_day, policy=policy, hours=ps.hours, config=asdict(ps), identifier=asdict(st),
                  targets=initial_targets, stop_buying=stopped, hours_delivered=len(trace), gates=len(log),
                  paid=sum(expenses.values()), day_spend=dict(expenses), orders=sum(g["purchase"] is not None for g in log),
                  abs_energy_kwh=sum(np.abs(t["action"]).sum()*devices.base_kw for t in trace),
                  correct=log[-1]["reference"], correct_exact=log[-1]["exact"], initial_f1=log[0]["f1"],
                  final_f1=log[-1]["f1"], f1_exact=log[-1]["f1_exact"], ac_violations=totals,
                  min_voltage=min(t["v"].min() for t in trace), max_voltage=max(t["v"].max() for t in trace),
                  max_apparent_pu=max(np.hypot(t["p"],t["q"]).max() for t in trace),
                  min_soc=min(t["soc"].min() for t in trace), max_soc=max(t["soc"].max() for t in trace),
                  applied_resource=dict(participation=ENROLLED, battery_kw_sum=devices.bat_kw.sum(),
                                        ev_kw_sum=devices.ev_kw.sum(), battery_duration_h=BATTERY[2]/BATTERY[1]),
                  zero_v_error_pu=day.zero_voltage_error, zero_complex_s_error_pu=day.zero_supply_error,
                  trace_file=acfile.name, trace_sha256=hashlib.sha256(acfile.read_bytes()).hexdigest(),
                  gate_file=gatefile.name, source_hashes=rec["source_hashes"], trajectory=summary)
    if any(cost > ps.cap_day+1e-7 for cost in expenses.values()) or day.zero_voltage_error > 1e-7 or day.zero_supply_error > 1e-7:
        raise ArithmeticError("invalid delivered budget or baseline AC replay")
    write_json(final_path, result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", choices=CASES, required=True)
    ap.add_argument("--days", choices=(1, 2), type=int, default=1)
    ap.add_argument("--budget", type=float, default=100.)
    ap.add_argument("--policies", nargs="+", choices=("P0","P1","P2","P3","last24"))
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--out", "--public", dest="out", type=Path, help="Working result directory (default: local cache/runs)")
    args = ap.parse_args()
    args.cache = args.cache or cache_root()
    args.out = args.out or args.cache/"runs"
    date = "2024-01-31" if args.days == 1 else "2024-01-30"
    policies = args.policies or (["P0","P1","P2","P3"] if args.days == 1 else ["P0","last24","P3"])
    rec = load_feeder(args.case, start_day=date, execution_days=args.days)
    data = history(rec)
    cache_initial = args.cache/"first"/f"{args.case}{'' if args.days==1 else '_'+date}.pkl"
    sidecar = cache_initial.with_suffix(".json")
    if cache_initial.exists() and sidecar.exists():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        binary = cache_initial.read_bytes()
        if meta["input_sha256"] != digest(data) or meta["cache_sha256"] != hashlib.sha256(binary).hexdigest():
            raise ValueError(f"Initial cache disagrees with the requested measurements: {cache_initial}")
        first = pickle.loads(binary)
    elif args.days == 2:
        first = identify(**data, st=ST)
        atomic_pickle(cache_initial, first)
        write_json(sidecar, dict(case=args.case, input_sha256=digest(data),
                                 cache_sha256=hashlib.sha256(cache_initial.read_bytes()).hexdigest()))
    else:
        raise FileNotFoundError(f"complete the paired offline run first: {cache_initial}")
    if first["settings"] != asdict(ST):
        raise ValueError("Initial and rolling identifier settings must match")
    ps = pr.ProbeSettings(hours=24*args.days, cap_day=args.budget)
    token = hashlib.sha256()
    for path in (Path(__file__), Path(pr.__file__), Path(__file__).with_name("four_feeders.py")):
        token.update(path.read_bytes())
    token.update(pickle.dumps(first))
    for policy in policies:
        out = args.out/f"probe_{args.case}_{date}_{args.budget:g}_{policy}.json"
        identify_day(rec, data, first, policy, ps, args.cache, args.case, date, out, token.hexdigest()[:16])


if __name__ == "__main__":
    main()
