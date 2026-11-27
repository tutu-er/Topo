"""Summarise paired, completed four-network Probe runs; explain zero purchases from saved inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

from ptid.runs.four_probe import DeviceSchedule, ST, cache_root, load_feeder, np, pr, views, write_json
from ptid.runs.four_records import input_records, read_record, record_names
from ptid.method.labels import DECIDED
from ptid.runs.four_feeders import history
from ptid.runs.four_offline import digest


def bid_check(row, rec, cache, hourly=None):
    """Replay score inputs from the saved P3 path, without rerunning the identifier."""
    trace_file = cache/"ac"/row["trace_file"]
    gates = json.loads((cache/"ac"/row["gate_file"]).read_text(encoding="utf-8"))
    ps = pr.ProbeSettings(**row["config"])
    devices = DeviceSchedule(rec)
    zeros = np.zeros(rec["n"])
    with np.load(trace_file, allow_pickle=False) as archive:
        observed = {k[len("observed_"):]:archive[k] for k in archive.files if k.startswith("observed_")}
        actions = archive["action_pu"]
        soc = archive["soc"]
    expenses = Counter()
    categories = Counter()
    best_offers = []
    break_even = None
    for h, g in enumerate(gates):
        if h+1 >= ps.hours:
            continue
        if g["trust"] is None:
            categories["no_eligible_open_target_or_stopped"] += 1
            continue
        if g["trust"] > ps.kappa_A:
            categories["trust_gate_closed"] += 1
            continue
        paths = list((cache/"gates"/row["case"]).glob("*/"+g["gate_key"]+".pkl"))
        if not paths:
            raise FileNotFoundError(g["gate_key"])
        tree, _, _, _, now, _, _ = pickle.loads(paths[0].read_bytes())
        data = {k:v[:8*24+h] for k,v in observed.items()}
        model, fp, fq = pr.setup(data, row["load_model"], ST)
        fc = pr.forecast(data, fp, fq, h+1)
        lead, rivals = views(model, tree, now, fc)
        devices.soc = np.full(rec["n"], .5) if h == 0 else soc[h-1].copy()
        up, down = devices.headroom(h+1, actions[h])
        budget = min(ps.cap_hour, ps.cap_day-expenses[(h+1)//24])
        seps = {k:pr.separation(lead, rival, model.sigma2) for k, rival in rivals.items()}
        passive = sum(pr.term(now[k], float(d@W@d), ST.m, ps.variant) for k,(d,_,W) in seps.items())
        best = cheapest_information = None
        for i, signed, cost in pr.feasible_blocks(model, lead, rivals, fc, up, down, budget, devices.base_kw, ps):
            gain = -passive
            for k, (d,G,W) in seps.items():
                delta = d+G[:,i]*signed
                gain += pr.term(now[k], float(delta@W@delta), ST.m, ps.variant)
            net = ps.psi*gain-cost
            if gain > 0 and (cheapest_information is None or cost/gain < cheapest_information["cny_per_unit"]):
                cheapest_information = dict(hour=h, meter=int(rec["labels"][i]), signed_kw=signed*devices.base_kw,
                                            information_gain=gain, cost_cny=cost, cny_per_unit=cost/gain)
            if best is None or net > best["net_cny"]:
                best = dict(hour=h, meter=int(rec["labels"][i]), signed_kw=signed*devices.base_kw,
                            information_gain=gain, cost_cny=cost, net_cny=net)
        actual = g["purchase"]
        if cheapest_information is not None:
            if break_even is None or cheapest_information["cny_per_unit"] < break_even["cny_per_unit"]:
                break_even = cheapest_information
            if hourly is not None:
                hourly.append(cheapest_information)
        if best is not None:
            best_offers.append(best)
        if best is None:
            category = "empty_feasible_menu"
        elif best["net_cny"] > 0:
            category = "positive_net_bid"
        else:
            category = "feasible_bids_below_cost"
        categories[category] += 1
        predicted_kw = zeros.copy()
        if best is not None and best["net_cny"] > 0:
            predicted_kw[rec["labels"].index(best["meter"])] = best["signed_kw"]
        if not np.allclose(predicted_kw, actions[h+1]*devices.base_kw, atol=1e-7, rtol=1e-10):
            raise ArithmeticError(f"replayed offer disagrees with delivery: {row['case']} h={h}")
        if actual is not None:
            expenses[(h+1)//24] += actual["cost"]
    return dict(eligible_order_gates=ps.hours-1, gate_counts=dict(categories),
                lowest_cost_per_positive_unit_on_observed_path=break_even,
                highest_nonzero_offer=max(best_offers,key=lambda r:r["net_cny"]) if best_offers else None)


def illustrate(case, public, cache, source):
    """An observed P3 trajectory, using hypothetical offers and their actual clearance result."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    row = read_record(source, f"probe_{case}_2024-01-31_100_P3.json")
    rec = load_feeder(case)
    row["load_model"] = pickle.loads((cache/"first"/f"{case}.pkl").read_bytes())["gate"]["load"]
    hourly = []
    qc = bid_check(row, rec, cache, hourly)
    gates = json.loads((cache/"ac"/row["gate_file"]).read_text(encoding="utf-8"))
    target = tuple(next(g for g in row["targets"] if g != "place"))
    identifier = row["identifier"]
    psi = row["config"]["psi"]
    hours = np.array([g["hour"] for g in gates])
    gaps = [g["gap_upper"][str(target)] for g in gates]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.labelcolor":"#263341","text.color":"#142534"})
    fig, axes = plt.subplots(2, 1, figsize=(8.4,5.8), sharex=True)
    fig.subplots_adjust(left=.11,right=.97,bottom=.12,top=.82,hspace=.32)
    fig.suptitle(f"{case.capitalize()}  |  31 January 2024",x=.11,y=.97,ha="left",fontsize=17,fontweight="bold")
    names = ", ".join(str(rec["labels"][i]) for i in target)
    fig.text(.11,.88,f"Meter group {{{names}}} began the day unresolved. "
                      f"{row['orders']} orders executed for {row['paid']:g} CNY.")
    ax = axes[0]
    ax.plot(hours,gaps,"o-",ms=3,lw=1.6,color="#245a78",label="Best available flip-witness gap")
    ax.axhline(identifier["m"],ls="--",lw=1,color="#9b520b",label=f"Decision margin ({identifier['m']:g})")
    ax.set_ylabel("Flip gap (score units)")
    ax.legend(loc="center",bbox_to_anchor=(.70,.74),frameon=False,fontsize=9)
    bx = axes[1]
    bx.plot([q["hour"] for q in hourly],[q["cost_cny"] for q in hourly],"s-",ms=3,color="#7b352e",
            label="Quoted price of the hypothetical offer")
    bx.plot([q["hour"] for q in hourly],[psi*q["information_gain"] for q in hourly],"o-",ms=3,
            color="#1b715d",label=f"Expected gain valued at {psi:g} CNY/unit")
    bx.set_ylabel("CNY (log scale)")
    bx.set_yscale("log")
    bx.set_xlabel("Decision hour (local time)")
    bx.set_xticks(np.arange(0,25,4))
    bx.set_xlim(-.4,24.3)
    bx.legend(loc="best",frameon=False,fontsize=9)
    for a in axes:
        a.grid(axis="y",color="#d9dfe1",lw=.65)
        a.set_axisbelow(True)
    out = public/"probe_offer_trajectory.png"
    fig.savefig(out,dpi=175)
    plt.close(fig)
    write_json(public/"probe_offer_trajectory.json",dict(case=case,policy="P3",date=row["start_day"],
        meter_group=[rec["labels"][i] for i in target],m=identifier["m"],psi=psi,actual_orders=row["orders"],
        actual_cny=row["paid"],hours=hours.tolist(),flip_witness_upper_gap=gaps,
        hourly_lowest_cost_per_positive_information=hourly,decision_check=qc))


def summarise(source, cache):
    results = [read_record(source, name) for name in record_names(source, "probe_*_2024-*.json")]
    if not results:
        raise FileNotFoundError(f"No probe trajectories in {source}")
    detail = []
    counter = Counter()
    records = {}
    initial_states = {}
    initial_snapshots = []
    for r in results:
        key = (r["case"], r["start_day"], r["hours"])
        if key not in records:
            records[key] = load_feeder(r["case"], start_day=r["start_day"], execution_days=r["hours"]//24)
        rec = records[key]
        if rec["source_hashes"] != r["source_hashes"]:
            raise ValueError("Source archives differ from those used for this trajectory")
        counter[(r["start_day"], int(r["config"]["cap_day"]), r["policy"])] += 1
        g = r["correct"]
        assert r["hours"] == r["hours_delivered"] and r["gates"] == r["hours"]+1
        assert sum(g.values()) == sum(t != "place" for t in r["targets"])
        assert r["ac_violations"]["new_v"] <= r["ac_violations"]["all_v"]
        def skim(k):
            return {field:k[field] for field in ("correct","wrong","unresolved")}
        item = dict(case=r["case"], date=r["start_day"], hours=r["hours"], cap=r["config"]["cap_day"],
                    policy=r["policy"], targets=len(r["targets"]), initial_f1=r["initial_f1"]["f1"],
                    final_f1=r["final_f1"]["f1"], final=skim(g), final_exact=skim(r["correct_exact"]),
                    correct_24=r["trajectory"][24]["reference"], cost=r["paid"], orders=r["orders"],
                    stop_buying=r["stop_buying"], ac_violations=r["ac_violations"])
        gates = json.loads((cache/"ac"/r["gate_file"]).read_text(encoding="utf-8"))
        if key not in initial_states:
            initial = cache/"first"/(r["case"]+("" if r["hours"] == 24 else "_"+r["start_day"])+".pkl")
            raw = initial.read_bytes()
            expected = digest(history(rec))
            raw_hash = hashlib.sha256(raw).hexdigest()
            manifest = json.loads(initial.with_suffix(".json").read_text(encoding="utf-8"))
            if manifest["input_sha256"] != expected or manifest["cache_sha256"] != raw_hash:
                raise ValueError("Initial-result manifest does not match this history or fit")
            fitted = pickle.loads(raw)
            initial_states[key] = fitted, expected
            initial_snapshots.append(dict(case=r["case"], start_day=r["start_day"], history_rows=8*24,
                                          cache=initial.name, input_sha256=expected, cache_sha256=raw_hash,
                                          settings=fitted["settings"], tree=fitted["tree"],
                                          source_hashes=rec["source_hashes"]))
        initial_fit, expected = initial_states[key]
        if initial_fit["settings"] != r["identifier"] or list(map(list, initial_fit["tree"])) != gates[0]["tree"]:
            raise ValueError("Experimental trajectory did not begin from the common matched estimate")
        tracepath = cache/"ac"/r["trace_file"]
        if hashlib.sha256(tracepath.read_bytes()).hexdigest() != r["trace_sha256"]:
            raise ValueError("AC trajectory checksum mismatch")
        with np.load(tracepath, allow_pickle=False) as arc:
            prefix = {k:arc["observed_"+k][:8*24] for k in expected}
            if digest(prefix) != expected:
                raise ValueError("Initial measurement history differs across these runs")
        solves = [solve for gate in gates for solve in gate["solves"]]
        item["rolling_anchored_tree_solves"] = dict(
            status_counts=dict(Counter(str(s["status"]) for s in solves)),
            gap_closed=sum(bool(s["gap_closed"]) for s in solves), total=len(solves))
        item["runtime"] = dict(cached_gates=sum(g["cached"] for g in gates),
                               gate_seconds_including_cache=sum(g["seconds"] for g in gates))
        per_target = []
        for target in r["targets"]:
            if target == "place":
                continue
            group = tuple(target)
            present = group in rec["truth"]
            correct_at = []
            for gate in gates:
                decided = gate["labels"][str(group)] == DECIDED
                if decided and (target in gate["tree"]) == present:
                    correct_at.append(gate["hour"])
            last_incorrect = max((h for h in range(r["hours"]+1) if h not in correct_at), default=-1)
            per_target.append(dict(group=group, true=present, final_inside=target in gates[-1]["tree"],
                                   final_label=gates[-1]["labels"][str(group)],
                                   first_correct=min(correct_at, default=None),
                                   continuously_correct_from=last_incorrect+1 if last_incorrect < r["hours"] else None))
        item["group_outcomes"] = per_target
        if r["policy"] == "P3":
            r["load_model"] = initial_fit["gate"]["load"]
            item["p3_bid_check"] = bid_check(r, rec, cache)
        detail.append(item)
    equal_trajectories = 0
    for i, a in enumerate(results):
        for b in results[i+1:]:
            if a["case"] != b["case"] or a["start_day"] != b["start_day"]:
                continue
            if a["targets"] != b["targets"]:
                raise AssertionError("initial targets differ within the paired comparison")
            with np.load(cache/"ac"/a["trace_file"]) as araw, np.load(cache/"ac"/b["trace_file"]) as braw:
                if np.array_equal(araw["action_pu"], braw["action_pu"]):
                    for key in ("observed_Y","observed_P","observed_Q","observed_V0sq","observed_Ppcc",
                                "observed_Qpcc","soc","voltage","p_pcc_pu","q_pcc_pu"):
                        if not np.array_equal(araw[key], braw[key]):
                            raise AssertionError(f"paired measurements or AC state differ: {key}")
                    equal_trajectories += 1
                    if a["correct"] != b["correct"] or a["final_f1"] != b["final_f1"]:
                        raise AssertionError("different terminal results on an identical action/measurement path")
    return dict(completed=len(detail), initial_snapshots=initial_snapshots,
                count_by_setting={str(k):v for k,v in sorted(counter.items())},
                equal_action_pairs_verified=equal_trajectories, rows=detail)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--public", dest="out", type=Path, help="Working result directory (default: local cache/runs)")
    parser.add_argument("--records", type=Path, help="Trajectory records to summarise (default: working results or published evidence)")
    parser.add_argument("--cache",type=Path)
    parser.add_argument("--illustrate", choices=("soumalas11","paper15","flynn16","pengwah18"))
    args = parser.parse_args()
    args.cache = args.cache or cache_root()
    out = args.out or args.cache/"runs"
    source = args.records or args.out or input_records(out, "probe_*_2024-*.json")
    report = summarise(source,args.cache)
    write_json(out/"probe_summary.json",report)
    if args.illustrate:
        illustrate(args.illustrate,out,args.cache,source)
    print("Completed",report["completed"],"paired checks",report["equal_action_pairs_verified"])
