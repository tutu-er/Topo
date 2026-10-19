"""Paired physical asset/root-context checks, not a new EV policy experiment.

Run: python -m research_experiments.mobile_ev_mle.physical_local_v1.experiment
     --output outputs/physical_lv_v1_main
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback

import numpy as np

from .. import likelihood as lk
from ..local_rx_identifiability.cases import _ar, LOCAL_SELECTION
from ..reference_benchmark.representation import (
    physical_tree_ac, encode_physical_tree, impedance_base_ohm, _orient,
)
from .assets import (PAPER_NAMES, PROFILES, build_network, ratio_summary,
                     public_lv_ratio_domain, charging_q_bound_kvar)
from .context import (rooted_scope, compress_zero_injection_series,
                      port_reduction_oracle, local_refinements)


ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                  indent=2, allow_nan=False)+"\n", encoding="utf-8")


def trajectory(network, seed, total=768):
    """Ordinary daily/AR load, fixed before testing any line-profile outcome."""
    code = {"paper15": 15, "paper11": 11, "flynn16": 16, "pengwah18": 18}[network["name"]]
    rng = lambda c: np.random.default_rng(np.random.SeedSequence([int(seed), code, c]))
    ids = network["observation_bus_ids"]
    by_bus = {b["bus_id"]: b for b in network["buses"]}
    p0 = np.array([by_bus[b]["pd_kw"] for b in ids])
    q0 = np.array([by_bus[b]["qd_kvar"] for b in ids])
    n = len(ids); hours = np.arange(total)/4.
    phase = .15+2*np.pi*np.arange(n)/n
    common = .84+.13*np.cos(2*np.pi*(hours-19)/24)+.06*np.cos(4*np.pi*(hours-7)/24)
    common += .035*_ar(rng(1), total, rho=.9)[:, 0]
    pf = common[:, None]+.07*np.sin(2*np.pi*hours[:, None]/24+phase)
    pf += .055*_ar(rng(2), total, width=n, rho=.65)
    qf = (.85+.10*np.cos(2*np.pi*(hours-18)/24))[:, None]
    qf = qf+.06*np.sin(2*np.pi*hours[:, None]/24+phase+.8)
    qf += .045*_ar(rng(4), total, rho=.85)
    qf += .085*_ar(rng(3), total, width=n, rho=.7)
    demand = pf*p0; p = demand.copy(); q = qf*q0
    sun = np.maximum(0., np.sin(np.pi*((hours % 24)-6)/12))
    pv = .15*p0[0]*sun*(.90+.04*np.tanh(_ar(rng(5), total, rho=.9)[:, 0]))
    p[:, 0] -= pv
    root = 1.01+.008*np.sin(2*np.pi*(hours-3)/24)
    root += .003*np.tanh(_ar(rng(6), total, rho=.97)[:, 0])
    sigma = dict(p=.01*p0, q=.01*np.maximum(abs(q0), .2*p0), v=np.full(n, .0005), root=.0005)
    white = {k:rng(11+i).standard_normal((total, n)) for i, k in enumerate(("p", "q", "v"))}
    white["root"] = rng(14).standard_normal(total)
    return dict(p_kw=p, q_kvar=q, root_v=root, nominal_p_kw=p0, nominal_q_kvar=q0,
                pv_kw=pv, demand_kw=demand, sigma=sigma, white=white,
                provenance="synthetic daily/AR ordinary loads, not measured field profiles")


def simulate(network, data):
    return physical_tree_ac(network["branches"], network["observation_bus_ids"],
        network["root_bus"], network["base_kv"], network["base_kw"], data["p_kw"],
        data["q_kvar"], network["source_bus"], root_v=data["root_v"])


def inference_payload(network, data, state, n):
    """Only arrived noisy readings and known scales; no physical tree or truth."""
    obs = {"p":data["p_kw"]+data["sigma"]["p"]*data["white"]["p"],
           "q":data["q_kvar"]+data["sigma"]["q"]*data["white"]["q"],
           "v":state["v"]+data["sigma"]["v"]*data["white"]["v"]}
    base = network["base_kw"]
    return dict(observed={k:(v[:n]/base if k in ("p", "q") else v[:n]).copy()
                          for k, v in obs.items()},
        sigma={k:(v/base if k in ("p", "q") else v).copy()
               for k, v in data["sigma"].items() if k != "root"},
        root_meter=((data["root_v"]+.0005*data["white"]["root"])[:n].copy(), .0005),
        base_kw=base, base_kv=network["base_kv"],
        ratio_domain=public_lv_ratio_domain(), history_rows=n,
        measurement_bus_ids=list(network["observation_bus_ids"]), truth_used=False,
        root_information="noisy original physical root meter; no cut-boundary oracle")


def _rms(a):
    return float(np.sqrt(np.mean(abs(np.asarray(a))**2)))


def context_audit(network, data, state):
    selection = LOCAL_SELECTION[network["name"]]
    focus = selection["terminals"]
    scope = rooted_scope(network, focus)
    chosen = [network["observation_bus_ids"].index(b) for b in focus]
    target = state["v"][:, chosen]
    port, record = port_reduction_oracle(network, focus, data["p_kw"], data["q_kvar"],
                                        data["root_v"], state, include_losses=True)
    naive, _ = port_reduction_oracle(network, focus, data["p_kw"], data["q_kvar"],
                                    data["root_v"], state, include_losses=False)
    exact_error = np.column_stack([abs(port["voltage_by_bus"][b])-target[:, j]
                                   for j, b in enumerate(focus)])
    naive_error = np.column_stack([abs(naive["voltage_by_bus"][b])-target[:, j]
                                   for j, b in enumerate(focus)])
    small = {selection["root"], selection["hidden"], *focus}
    edges = [e for e in network["branches"] if e["from_bus"] in small and e["to_bus"] in small]
    correct = physical_tree_ac(edges, focus, selection["root"], network["base_kv"],
        network["base_kw"], data["p_kw"][:, chosen], data["q_kvar"][:, chosen],
        selection["root"], root_v=abs(state["voltage_by_bus"][selection["root"]]))
    wrong = physical_tree_ac(edges, focus, selection["root"], network["base_kv"],
        network["base_kw"], data["p_kw"][:, chosen], data["q_kvar"][:, chosen],
        selection["root"], root_v=data["root_v"])
    encoded = encode_physical_tree(network["branches"], network["observation_bus_ids"],
        network["root_bus"], network["base_kv"], network["base_kw"], network["source_bus"])
    bank = local_refinements(encoded.candidate, chosen)
    phase = state["voltage_by_bus"][selection["root"]]/abs(state["voltage_by_bus"][selection["root"]])
    cut_complex_error = np.column_stack([correct["voltage_by_bus"][b]*phase-state["voltage_by_bus"][b]
                                         for b in focus])
    result = dict(scope=scope, exact_port_metadata=record, candidate_count=len(bank),
        candidate_measured_bus_counts=[c.n for c in bank],
        all_candidate_context_supports_preserved=all(set(encoded.candidate.supports)<=set(c.supports) for c in bank),
        exact_port_v_max_pu=float(np.max(abs(exact_error))),
        exact_port_master_max_pu=float(max(np.max(abs(port["master_p"]-state["master_p"])),
                                          np.max(abs(port["master_q"]-state["master_q"])))),
        correct_boundary_v_max_pu=float(np.max(abs(correct["v"]-target))),
        correct_boundary_phase_aligned_max_pu=float(np.max(abs(cut_complex_error))),
        naive_port_v_rmse_pu=_rms(naive_error),
        naive_port_v_rmse_sigma=_rms(naive_error)/.0005,
        naive_port_v_max_pu=float(np.max(abs(naive_error))),
        wrong_root_v_rmse_pu=_rms(wrong["v"]-target),
        wrong_root_v_rmse_sigma=_rms(wrong["v"]-target)/.0005,
        wrong_root_v_max_pu=float(np.max(abs(wrong["v"]-target))),
        old_local_meters=len(focus), full_context_meters=len(network["observation_bus_ids"]),
        measurement_information_budget_changed=True, evaluation_only=True,
        naive_error_by_time=naive_error, wrong_error_by_time=wrong["v"]-target)
    if max(result[k] for k in ("exact_port_v_max_pu", "exact_port_master_max_pu",
                               "correct_boundary_phase_aligned_max_pu")) > 1e-9:
        raise AssertionError("exact local-equivalent audit exceeded tolerance")
    return result


def one_condition(name, profile, seed, folder):
    start = perf_counter(); net = build_network(name, profile)
    data = trajectory(net, seed); state = simulate(net, data)
    encoded = encode_physical_tree(net["branches"], net["observation_bus_ids"], net["root_bus"],
                                  net["base_kv"], net["base_kw"], net["source_bus"])
    response = lk.ac_predict(encoded.candidate, encoded.r, encoded.x,
        data["p_kw"]/net["base_kw"], data["q_kvar"]/net["base_kw"], root_v=data["root_v"])
    encoded_error = float(np.max(abs(response["v"]-state["v"])))
    compressed = compress_zero_injection_series(net)
    cs = simulate(compressed, data)
    series_error = float(max(np.max(abs(cs["v"]-state["v"])),
        np.max(abs(cs["master_p"]-state["master_p"])), np.max(abs(cs["master_q"]-state["master_q"]))))
    if max(encoded_error, series_error, state["ac_equation_residual"], state["power_balance_residual"]) > 1e-9:
        raise AssertionError("independent full-tree audit failed")
    local = context_audit(net, data, state) if name in LOCAL_SELECTION else None
    path = folder/"raw"/f"{name}_{profile}_{seed}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = inference_payload(net, data, state, 672)
    np.savez_compressed(path, p_kw=data["p_kw"], q_kvar=data["q_kvar"], root_v=data["root_v"],
        v_true=state["v"], p_obs=payload["observed"]["p"], q_obs=payload["observed"]["q"],
        v_obs=payload["observed"]["v"], root_obs=payload["root_meter"][0])
    _, parent, edge_for, order = _orient(net["branches"], net["root_bus"])
    zbase = impedance_base_ohm(net["base_kv"], net["base_kw"])
    ibase = net["base_kw"]/(np.sqrt(3)*net["base_kv"])
    ratios = ratio_summary(net); rows=[]
    for count in (288, 672):
        volts = state["v"][:count]; root = data["root_v"][:count]
        drop = root[:, None]-volts
        edge_drop=[]; utilization=[]
        for b in order[1:]:
            a=parent[b]; e=net["branches"][edge_for[b]]
            edge_drop.append(abs(state["voltage_by_bus"][a][:count])-abs(state["voltage_by_bus"][b][:count]))
            if e.get("rating_a"):
                z=complex(e["r_ohm"],e["x_ohm"])/zbase
                current=(state["voltage_by_bus"][a][:count]-state["voltage_by_bus"][b][:count])/z
                utilization.append(float(np.max(abs(current))*ibase/e["rating_a"]))
        pair=np.column_stack([volts[:,j]-volts[:,k] for j in range(volts.shape[1]) for k in range(j)])
        row=dict(network=name, profile=profile, seed=seed, N=count, status="completed",
            voltage_min=float(volts.min()), voltage_max=float(volts.max()),
            root_drop_mean_pu=float(drop.mean()), root_drop_rms_pu=_rms(drop), root_drop_max_pu=float(drop.max()),
            edge_drop_rms_pu=_rms(edge_drop), edge_drop_max_pu=float(np.max(edge_drop)),
            terminal_difference_rms_pu=_rms(pair),
            max_ampacity_fraction=max(utilization) if utilization else None,
            operational_voltage_ok=bool(volts.min()>=.9 and volts.max()<=1.1),
            ampacity_ok=bool(max(utilization)<=1.) if utilization else None,
            omitted_shunt_q_bound_kvar=charging_q_bound_kvar(net),
            ratio_min=ratios["minimum"], ratio_max=ratios["maximum"], ratio_distinct=ratios["distinct_ratio_count"],
            old_ratio_domain_excluded_edges=ratios["legacy_domain_excluded_edges"],
            full_physical_buses=len(net["buses"]), full_physical_edges=len(net["branches"]),
            full_meters=len(net["observation_bus_ids"]),
            encoded_vs_independent_v_max_pu=encoded_error, series_error_max_pu=series_error,
            ac_equation_residual=state["ac_equation_residual"], power_balance_residual=state["power_balance_residual"],
            raw_file=str(path.relative_to(folder)), raw_sha256=sha(path))
        if local:
            row.update({k:v for k,v in local.items() if k not in ("scope", "exact_port_metadata", "naive_error_by_time", "wrong_error_by_time")})
            row.update(naive_port_v_rmse_pu=_rms(local["naive_error_by_time"][:count]),
                       wrong_root_v_rmse_pu=_rms(local["wrong_error_by_time"][:count]))
            row["naive_port_v_rmse_sigma"]=row["naive_port_v_rmse_pu"]/.0005
            row["wrong_root_v_rmse_sigma"]=row["wrong_root_v_rmse_pu"]/.0005
        rows.append(row)
    return rows, dict(network=name, profile=profile, seed=seed, seconds=perf_counter()-start,
        ac_forward_trajectories=7 if local else 3, mle_fits=0, ev_planning_calls=0,
        state_status="complete", independent_ac_iterations=state["iterations"],
        context_metadata=None if local is None else local["scope"])


def standard_reference(name):
    net=build_network(name);ids=net["observation_bus_ids"];by={b["bus_id"]:b for b in net["buses"]}
    data=dict(p_kw=np.array([[by[b]["pd_kw"] for b in ids]]),
              q_kvar=np.array([[by[b]["qd_kvar"] for b in ids]]),root_v=np.ones(1))
    state=simulate(net,data);compressed=compress_zero_injection_series(net);other=simulate(compressed,data)
    return dict(network=name, nominal_P_kw=net["total_pd_kw"], nominal_Q_kvar=net["total_qd_kvar"],
        voltage_min=float(state["v"].min()),voltage_max=float(state["v"].max()),
        root_drop_max_pu=float(1-state["v"].min()),ratios=ratio_summary(net),
        physical_buses=len(net["buses"]),compressed_buses=len(compressed["buses"]),
        series_error_max_pu=float(max(np.max(abs(state["v"]-other["v"])),
                                     np.max(abs(state["master_p"]-other["master_p"])))),
        power_balance_residual=state["power_balance_residual"],source=net["provenance"])


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True)
    parser.add_argument("--pilot",action="store_true");args=parser.parse_args()
    folder=Path(args.output)
    if folder.exists():raise FileExistsError("use a fresh output folder")
    folder.mkdir(parents=True)
    files=[Path(__file__).with_name(z) for z in ("assets.py","catalog.json","context.py","experiment.py","protocol.md","math.md")]
    files += [ROOT/z for z in ("research_experiments/mobile_ev_mle/likelihood.py",
        "research_experiments/mobile_ev_mle/reference_benchmark/networks.py",
        "research_experiments/mobile_ev_mle/reference_benchmark/representation.py",
        "research_experiments/mobile_ev_mle/local_rx_identifiability/cases.py",
        "rnj_wzzt_core/rnj_wzzt/data/paper_style_terminal_lv.py",
        "rnj_wzzt_core/rnj_wzzt/data/paper_style_case_bank.py",
        "terminal_case33/data/case33bw_raw.py")]
    # Include the actually imported legacy AC/helper closure as well as static
    # asset sources; numerical code outside ROOT is identified by environment.
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path:
            path=Path(path).resolve()
            if path.suffix==".py" and path.is_relative_to(ROOT):
                files.append(path)
    files=sorted(set(files))
    config=dict(schema="physical_local_assets_root_context_v1",pilot=args.pilot,
        networks=list(PAPER_NAMES),profiles=list(PROFILES),seeds=[34001] if args.pilot else list(range(35001,35005)),
        prefixes=[288,672],total=768,holdout=[672,768],dt_minutes=15,
        sigma_rule="P=.01P0; Q=.01max(abs(Q0),.2P0); V/root=.0005pu; known independent Gaussian",
        root_information="noisy original-root meter",measurement_bias=0,
        public_lv_domain=public_lv_ratio_domain(),
        goal="physical dataset and root-context validation; not MLE or EV performance",
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        python=sys.version,numpy=np.__version__,platform=platform.platform())
    write(folder/"configuration.json",config)
    all_rows=[];runs=[];started=perf_counter()
    for name in PAPER_NAMES:
        for profile in PROFILES:
            net=build_network(name,profile);write(folder/"networks"/f"{name}_{profile}.json",net)
            for seed in config["seeds"]:
                try:
                    rows,run=one_condition(name,profile,seed,folder);all_rows.extend(rows);runs.append(run)
                except Exception as exc:
                    failure=dict(network=name,profile=profile,seed=seed,status="failed",error=repr(exc),traceback=traceback.format_exc())
                    all_rows += [dict(failure,N=n) for n in config["prefixes"]]
                    runs.append(failure)
                print(name,profile,seed,all_rows[-1]["status"],flush=True)
                write(folder/"progress.json",dict(completed_conditions=len(runs),expected_conditions=12*len(config["seeds"])))
    standard=[standard_reference(z) for z in ("case33","cigre_residential")]
    summary=dict(config_sha256=sha(folder/"configuration.json"),conditions=len(runs),
        expected_conditions=12*len(config["seeds"]),rows=len(all_rows),
        failed_conditions=sum(r.get("status")=="failed" for r in runs),
        seconds=perf_counter()-started,standard_references=standard,rows_data=all_rows,runs=runs,
        inference_performance_tested=False,mle_fits=0,ev_planning_calls=0)
    write(folder/"results.json",summary)
    fields=sorted(set().union(*(r.keys() for r in all_rows)))
    with (folder/"conditions.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(all_rows)
    write(folder/"completed.json",dict(conditions=len(runs),failed=summary["failed_conditions"],
        results_sha256=sha(folder/"results.json"),seconds=summary["seconds"]))
    print(json.dumps({k:v for k,v in summary.items() if k not in ("standard_references","rows_data","runs")}),flush=True)


if __name__=="__main__":main()
