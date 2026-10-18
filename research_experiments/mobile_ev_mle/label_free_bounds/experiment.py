"""Observed-only case33 bounds audit, with simulator truth in separate files.

Only ``generate_history`` and ``audit_history`` read the physical generating
parameters.  ``run_history`` accepts a noisy meter file and public candidate /
equipment-domain files; its signature does not expose simulator truth.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy

from research_experiments.mobile_ev_mle.case33_candidate_ranking import experiment as ranking
from research_experiments.mobile_ev_mle.case33_quality_fixed_rx import experiment as base
from research_experiments.mobile_ev_mle.reference_benchmark.representation import (
    impedance_base_ohm, physical_tree_ac,
)


VERSION = "label_free_case33_bounds_v1"
CONFIG = dict(
    version=VERSION, seeds=[20261018071, 20261018072], samples=[96, 288],
    source_modes=["none", "pcc_PQ"], sample_minutes=15, alpha=.05,
    nominal_P_noise_fraction=.01, nominal_Q_noise_fraction=.01,
    voltage_sigma_pu=.0005, root_voltage_pu=1., root_voltage_known=True,
    pv_buses=[18, 25, 33], pv_weights=[.35, .30, .35],
    pv_capacity_fraction=.60, pv_reactive_power="zero; unity power factor",
    load_profile="synthetic ordinary daily profile plus independent bounded AR(1) fluctuations",
    source_cap_fraction_of_nominal_apparent=.15,
    asset_modulus_min_fraction=.50, asset_modulus_max_fraction=1.50,
    voltage_domain=[.85, 1.10], ratio_X_over_R=[.25, 4.],
    nodal_apparent_cap_rule="2*nominal_apparent+installed_PV_capacity",
    threshold_calibration="none; raw independent Gaussian joint-noise chi-square bound",
    candidate_parameters="unknown, shared over all samples; no generating R/X passed to bound builder",
    data_kind="synthetic time series on complete standard case33; no field measurements",
)


def plain(value):
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, np.generic): return plain(value.item())
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode()
    if path.suffix == ".gz":
        with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as stream:
            stream.write(data)
    else:
        path.write_bytes(data)


def read(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific_source_hashes():
    root=Path(__file__).resolve().parents[3]
    relative=["research_experiments/mobile_ev_mle/label_free_bounds/experiment.py",
        "research_experiments/mobile_ev_mle/label_free_bounds/bounds.py",
        "research_experiments/mobile_ev_mle/reference_benchmark/representation.py",
        "research_experiments/mobile_ev_mle/reference_benchmark/networks.py",
        "research_experiments/mobile_ev_mle/case33_candidate_ranking/experiment.py",
        "terminal_case33/data/case33bw_raw.py"]
    return {name:digest(root/name) for name in relative}


def public_assets():
    """One predeclared equipment prior, derived only from the public asset table.

    The prior is additional engineering information.  Unknown field P/Q/V
    alone cannot establish a positive conductor-impedance lower bound.
    """
    ctx = base.make_context(); net = ctx["network"]
    zbase = impedance_base_ohm(net["base_kv"], net["base_kw"])
    assets = net["branches"] + net["metadata"]["normally_open_ties"]
    magnitudes = np.array([np.hypot(e["r_ohm"], e["x_ohm"])/zbase for e in assets])
    ratios = np.array([e["x_ohm"]/e["r_ohm"] for e in assets])
    ratio_lo, ratio_hi = CONFIG["ratio_X_over_R"]
    if ratios.min() < ratio_lo or ratios.max() > ratio_hi:
        raise ValueError("Predeclared X/R domain does not cover the public assets; stop rather than tune it")
    capacity = CONFIG["pv_capacity_fraction"]*ctx["p0"].sum()
    pv = np.zeros(len(ctx["ids"]))
    for bus, weight in zip(CONFIG["pv_buses"], CONFIG["pv_weights"]):
        pv[ctx["ids"].index(bus)] = capacity*weight
    caps = (2*np.hypot(ctx["p0"], ctx["q0"])+pv)/net["base_kw"]
    source_cap = (CONFIG["source_cap_fraction_of_nominal_apparent"]*
                  np.hypot(ctx["p0"], ctx["q0"]).sum()/net["base_kw"])
    candidates = []
    for i, original in enumerate(ranking.catalog()):
        parents, _ = ranking.rooted(original["branches"])
        candidates.append(dict(id=f"c{i:02d}", root=net["root_bus"],
            edges=[[parents[bus][0],bus] for bus in sorted(parents) if bus!=net["root_bus"]],
            observed_nodes=ctx["ids"], source_node=2,
            description=original["family"], edit=original["edit"]))
    return dict(version=VERSION, network="case33", base_kw=net["base_kw"],
        base_kv=net["base_kv"], zbase_ohm=zbase, observed_nodes=ctx["ids"],
        candidates=candidates,
        domain=dict(z_min=float(CONFIG["asset_modulus_min_fraction"]*magnitudes.min()),
            z_max=float(CONFIG["asset_modulus_max_fraction"]*magnitudes.max()),
            v_min=CONFIG["voltage_domain"][0], v_max=CONFIG["voltage_domain"][1],
            power_caps={str(bus): float(cap) for bus, cap in zip(ctx["ids"], caps)},
            ratio_min=ratio_lo, ratio_max=ratio_hi, source_power_cap=float(source_cap)),
        provenance=net["provenance"],
        asset_prior_note="Global physical-edge bounds from all public closed and tie assets; hypothetical engineering prior, not estimated from unknown field P/Q/V",
        physical_edge_scope="native case33 edges; observed internal loads retained; no virtual zero-impedance meter edges",
        public_asset_summary=dict(count=len(assets), min_modulus_pu=float(magnitudes.min()),
            max_modulus_pu=float(magnitudes.max()), min_X_over_R=float(ratios.min()),
            max_X_over_R=float(ratios.max())), nominal_p_kw=ctx["p0"],
        nominal_q_kvar=ctx["q0"], installed_pv_kw=pv)


def ar1(rng, shape, persistence=.82):
    out = np.empty(shape); out[0] = rng.standard_normal(shape[1:])
    for t in range(1, shape[0]):
        out[t] = persistence*out[t-1]+np.sqrt(1-persistence**2)*rng.standard_normal(shape[1:])
    return np.clip(out, -2.5, 2.5)


def generate_history(samples, seed, source_mode):
    if source_mode not in CONFIG["source_modes"]:
        raise ValueError("unknown declared source mode")
    ctx = base.make_context(); net = ctx["network"]; ids = ctx["ids"]
    base_kw = net["base_kw"]; n = len(ids)
    # Shared load/noise streams make the source scenarios paired.  T is part
    # of the seed so that the 96- and 288-observation histories are independent.
    streams = np.random.SeedSequence([seed, samples]).spawn(4)
    rng_load, rng_pv, rng_source, rng_noise = map(np.random.default_rng, streams)
    hour = np.arange(samples)/4 % 24
    common = (.62+.12*np.exp(-((hour-8)/2.2)**2)+
              .30*np.exp(-((hour-19)/3.3)**2))
    p_factor = np.clip(common[:, None]*(1+.075*ar1(rng_load, (samples, n))), .25, 1.35)
    q_factor = np.clip(p_factor*(1+.06*ar1(rng_load, (samples, n))), .20, 1.50)
    p_load = ctx["p0"]*p_factor; q_load = ctx["q0"]*q_factor
    public = public_assets(); pv_cap = np.asarray(public["installed_pv_kw"])
    daylight = np.maximum(np.sin(np.pi*(hour-6)/12), 0)
    cloud = np.clip(.88+.045*ar1(rng_pv, (samples, 1)), .65, 1.)
    pv = daylight[:, None]*cloud*pv_cap[None, :]
    p_kw = p_load-pv; q_kvar = q_load
    source_p = np.zeros(samples); source_q = np.zeros(samples)
    if source_mode == "pcc_PQ":
        nominal = float(np.hypot(ctx["p0"], ctx["q0"]).sum())
        source_p = nominal*(.065*np.sin(2*np.pi*(hour-11)/24)+
                            .012*ar1(rng_source, (samples, 1))[:, 0])
        source_q = nominal*(.035*np.cos(2*np.pi*(hour-9)/24)+
                            .007*ar1(rng_source, (samples, 1))[:, 0])
    physical = physical_tree_ac(net["branches"], ids, net["root_bus"], net["base_kv"],
        base_kw, p_kw, q_kvar, source_bus=2, source_p_kw=source_p,
        source_q_kvar=source_q, root_v=CONFIG["root_voltage_pu"])
    p, q, v = p_kw/base_kw, q_kvar/base_kw, physical["v"]
    sigma_p = CONFIG["nominal_P_noise_fraction"]*np.maximum(ctx["p0"], 1.)/base_kw
    sigma_q = CONFIG["nominal_Q_noise_fraction"]*np.maximum.reduce(
        [np.abs(ctx["q0"]), .2*ctx["p0"], np.ones(n)])/base_kw
    sigma_v = np.full(n, CONFIG["voltage_sigma_pu"])
    ep, eq, ev = [rng_noise.standard_normal((samples, n)) for _ in range(3)]
    history_id = f"case33_T{samples}_seed{seed}_{source_mode}"
    meters = dict(version=VERSION, id=history_id, observed_nodes=ids, samples=samples,
        source_mode=source_mode, source_projection=source_mode!="none", source_node=2,
        p=p+sigma_p*ep, q=q+sigma_q*eq, v=v+sigma_v*ev,
        sigma_p=sigma_p, sigma_q=sigma_q, sigma_v=sigma_v,
        root_v=np.ones(samples), sigma_root=None,
        observation_note="Only noisy local P/Q and voltage magnitudes; fixed known root voltage; no root power meter",
        time_minutes=np.arange(samples)*CONFIG["sample_minutes"])
    truth = dict(version=VERSION, id=history_id, generating_candidate_id="c00",
        p=p, q=q, v=v, source_p=source_p/base_kw, source_q=source_q/base_kw,
        branch_table=net["branches"],
        standardized_noise_squared=float(np.sum(ep**2)+np.sum(eq**2)+np.sum(ev**2)),
        power_balance_residual=physical["power_balance_residual"],
        ac_equation_residual=physical["ac_equation_residual"],
        iterations=physical["iterations"], p_load_kw=p_load, q_load_kvar=q_load,
        pv_kw=pv, seed=seed)
    return meters, truth


def generate_bundle(destination, *, pilot=False):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    assets_path = destination/"public_assets.json"
    save(assets_path, public_assets()); save(destination/"config.json", CONFIG)
    histories = []
    settings = [(t, s, mode) for t in CONFIG["samples"] for s in CONFIG["seeds"]
                for mode in CONFIG["source_modes"]]
    if pilot: settings = settings[:1]
    for samples, seed, mode in settings:
        meters, truth = generate_history(samples, seed, mode)
        meter_path = destination/"meters"/(meters["id"]+".json.gz")
        truth_path = destination/"truth"/(meters["id"]+".json.gz")
        save(meter_path, meters); save(truth_path, truth)
        histories.append(dict(id=meters["id"], meter_file=str(meter_path.relative_to(destination)),
            truth_file=str(truth_path.relative_to(destination)), meter_sha256=digest(meter_path),
            truth_sha256=digest(truth_path)))
    save(destination/"manifest.json", dict(version=VERSION, histories=histories,
        public_assets_sha256=digest(assets_path), configuration=CONFIG,
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
            platform=platform.platform(), executable=sys.executable,
            thread_environment={k:os.environ.get(k) for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})))
    return histories


def run_history(meter_path, public_assets_path, result_path, *, alpha=.05):
    """The whole decision stage: no truth path or true-topology argument."""
    from .bounds import TreeSpec, PhysicalDomain, Measurements, certify_candidate
    began = perf_counter(); meter = read(meter_path); assets = read(public_assets_path)
    if tuple(meter["observed_nodes"]) != tuple(assets["observed_nodes"]):
        raise ValueError("meter / public candidate observation order mismatch")
    domain_record = deepcopy(assets["domain"])
    domain_record["power_caps"] = {int(k):v for k,v in domain_record["power_caps"].items()}
    if meter["source_mode"] == "none": domain_record["source_power_cap"] = 0.
    domain = PhysicalDomain(**domain_record)
    measurements = Measurements(**{k:np.asarray(meter[k], float) for k in
        ("p", "q", "v", "root_v", "sigma_p", "sigma_q", "sigma_v")},
        sigma_root=meter["sigma_root"])
    records = []; hashes=scientific_source_hashes()
    for item in assets["candidates"]:
        start = perf_counter()
        try:
            tree = TreeSpec(root=item["root"], edges=tuple(tuple(e) for e in item["edges"]),
                observed_nodes=tuple(item["observed_nodes"]),
                source_node=item["source_node"] if meter["source_mode"]!="none" else None)
            result = certify_candidate(tree, measurements, domain,
                source_projection=bool(meter["source_projection"]), alpha=alpha)
            if hasattr(result, "__dataclass_fields__"):
                from dataclasses import asdict
                result = asdict(result)
            records.append(dict(candidate_id=item["id"], success=True, result=result,
                ac_feasible_point=ac_feasible_point(item,meter,assets,result),
                seconds=perf_counter()-start))
        except Exception as exc:
            records.append(dict(candidate_id=item["id"], success=False, failure=repr(exc),
                seconds=perf_counter()-start))
        save(result_path, dict(version=VERSION, history_id=meter["id"], alpha=alpha,
            status="running", completed_candidates=len(records), total_candidates=len(assets["candidates"]),
            candidate_results=records, meter_sha256=digest(meter_path),
            public_assets_sha256=digest(public_assets_path), selector_read_paths=[str(meter_path), str(public_assets_path)],
            scientific_source_hashes=hashes,
            seconds=perf_counter()-began))
    out = dict(version=VERSION, history_id=meter["id"], alpha=alpha, status="complete",
        completed_candidates=len(records), total_candidates=len(assets["candidates"]),
        candidate_results=records, meter_sha256=digest(meter_path),
        public_assets_sha256=digest(public_assets_path),
        scientific_source_hashes=hashes,
        selector_read_paths=[str(meter_path), str(public_assets_path)], seconds=perf_counter()-began)
    save(result_path, out)
    return out


def ac_feasible_point(candidate,meter,assets,certificate):
    """A raw-noise SSE *upper* bound at one physical parameter point.

    Measured P/Q are a particular latent-state choice and consequently have
    zero P/Q residual at this point.  The whole model still permits P/Q
    corrections; this optional feasible point is not a conditional-MLE claim.
    A large value cannot reject a candidate.
    """
    began=perf_counter(); domain=assets["domain"]; m=len(candidate["edges"])
    theta=np.asarray(certificate["theta"],float); zbase=assets["zbase_ohm"]
    branches=[dict(from_bus=a,to_bus=b,r_ohm=float(theta[e]*zbase),
                   x_ohm=float(theta[m+e]*zbase),status=True)
              for e,(a,b) in enumerate(candidate["edges"])]
    p,q=np.asarray(meter["p"]),np.asarray(meter["q"])
    caps=np.array([domain["power_caps"][str(b)] for b in assets["observed_nodes"]])
    try:
        physical=physical_tree_ac(branches,assets["observed_nodes"],candidate["root"],
            assets["base_kv"],assets["base_kw"],p*assets["base_kw"],q*assets["base_kw"],
            source_bus=meter["source_node"],root_v=np.asarray(meter["root_v"]))
        all_v=np.concatenate([np.asarray(a) for a in physical["voltage_by_bus"].values()])
        modulus=np.hypot(theta[:m],theta[m:]); ratio=theta[m:]/theta[:m]
        coverage=dict(nodal_power=bool(np.all(np.hypot(p,q)<=caps[None,:]+1e-12)),
            voltage=bool(np.min(abs(all_v))>=domain["v_min"] and np.max(abs(all_v))<=domain["v_max"]),
            modulus=bool(np.all(modulus>=domain["z_min"]-1e-12) and np.all(modulus<=domain["z_max"]+1e-12)),
            ratio=bool(np.all(ratio>=domain["ratio_min"]-1e-12) and np.all(ratio<=domain["ratio_max"]+1e-12)),
            source=True)
        in_domain=all(coverage.values())
        sse=float(np.sum(((physical["v"]-np.asarray(meter["v"]))/np.asarray(meter["sigma_v"]))**2))
        return dict(available=True,feasible_for_declared_AC_domain=in_domain,
            domain_coverage=coverage,
            whitened_AC_sse_upper=sse if in_domain else None,
            point_sse_without_domain_claim=sse,
            ac_noise_ball_feasible=bool(in_domain and sse<=certificate["chi_square_threshold"]),
            latent_PQ_choice="noisy meter readings",source_PQ_choice=[0.,0.],
            fitted_or_MLE=False,physical_equation_residual=physical["ac_equation_residual"],
            power_balance_residual=physical["power_balance_residual"],
            voltage_min=float(np.min(abs(all_v))),voltage_max=float(np.max(abs(all_v))),
            seconds=perf_counter()-began)
    except Exception as exc:
        return dict(available=False,failure=repr(exc),seconds=perf_counter()-began)


def audit_history(meter_path, truth_path, public_assets_path, result_path):
    """Read truth only after candidate decisions have been saved."""
    result = read(result_path)
    if result["status"] != "complete": raise ValueError("decision stage must finish before truth audit")
    meter, truth, assets = read(meter_path), read(truth_path), read(public_assets_path)
    dom = assets["domain"]; p, q, v = [np.asarray(truth[k]) for k in ("p", "q", "v")]
    caps = np.array([dom["power_caps"][str(b)] for b in assets["observed_nodes"]])
    zbase = assets["zbase_ohm"]
    z = np.array([np.hypot(e["r_ohm"], e["x_ohm"])/zbase for e in truth["branch_table"]])
    ratio = np.array([e["x_ohm"]/e["r_ohm"] for e in truth["branch_table"]])
    cap_source = dom["source_power_cap"] if meter["source_mode"]!="none" else 0.
    coverage = dict(voltage=bool(v.min()>=dom["v_min"] and v.max()<=dom["v_max"]),
        nodal_power=bool(np.all(np.hypot(p,q)<=caps[None,:]+1e-12)),
        source_power=bool(np.max(np.hypot(truth["source_p"],truth["source_q"]))<=cap_source+1e-12),
        impedance=bool(z.min()>=dom["z_min"] and z.max()<=dom["z_max"]),
        ratio=bool(ratio.min()>=dom["ratio_min"] and ratio.max()<=dom["ratio_max"]))
    records = result["candidate_results"]
    rejected, retained, failed = [], [], []
    residual_rejected,residual_retained=[],[]
    for row in records:
        if not row["success"]:
            failed.append(row["candidate_id"]); continue
        item = row["result"]
        rejection = item.get("rejected", item.get("reject", False))
        (rejected if rejection else retained).append(row["candidate_id"])
        (residual_rejected if item.get("residual_chi_reject",False) else residual_retained).append(row["candidate_id"])
    generating = truth["generating_candidate_id"]
    from scipy.stats import chi2
    dimension = p.size*3
    return dict(history_id=meter["id"], samples=meter["samples"], source_mode=meter["source_mode"],
        public_domain_coverage=coverage, all_domain_assumptions_covered=all(coverage.values()),
        generating_candidate_id=generating, true_candidate_retained=generating in retained,
        uniquely_retained_true=retained==[generating] and not failed,
        rejected=rejected, retained=retained, failures=failed,
        unresolved=len(retained)!=1 or bool(failed), candidate_count=len(records),
        wrong_rejection_count=len([k for k in rejected if k!=generating]),
        residual_chi=dict(rejected=residual_rejected,retained=residual_retained,
            true_candidate_retained=generating in residual_retained,
            uniquely_retained_true=residual_retained==[generating] and not failed,
            unresolved=len(residual_retained)!=1 or bool(failed),
            wrong_rejection_count=len([k for k in residual_rejected if k!=generating])),
        noise_squared=truth["standardized_noise_squared"], noise_dimension=dimension,
        chi_square_quantile=float(chi2.ppf(1-result["alpha"],dimension)),
        joint_noise_event=bool(truth["standardized_noise_squared"]<=chi2.ppf(1-result["alpha"],dimension)),
        truth_voltage_min=float(v.min()), truth_voltage_max=float(v.max()),
        generated_nodal_max_to_cap=float(np.max(np.hypot(p,q)/caps[None,:])),
        source_max_to_cap=0. if cap_source==0 else float(np.max(np.hypot(truth["source_p"],truth["source_q"]))/cap_source),
        ac_equation_residual=truth["ac_equation_residual"],
        power_balance_residual=truth["power_balance_residual"], seconds=result["seconds"])


def execute_bundle(destination, *, workers=1, alpha=.05):
    destination = Path(destination); manifest = read(destination/"manifest.json")
    jobs = [(destination/row["meter_file"], destination/"public_assets.json",
             destination/"results"/(row["id"]+".json")) for row in manifest["histories"]]
    hashes=scientific_source_hashes()
    def reusable(path):
        if not path.exists(): return False
        record=read(path)
        return record.get("status")=="complete" and record.get("scientific_source_hashes")==hashes
    if workers<=1:
        for meter_path, assets_path, result_path in jobs:
            if reusable(result_path): continue
            run_history(meter_path,assets_path,result_path,alpha=alpha)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_history,m,a,r,alpha=alpha) for m,a,r in jobs
                       if not reusable(r)]
            for future in as_completed(futures): future.result()
    # Post-decision audits have their own files and never modify the decisions.
    audits = []
    for row in manifest["histories"]:
        audit = audit_history(destination/row["meter_file"],destination/row["truth_file"],
            destination/"public_assets.json",destination/"results"/(row["id"]+".json"))
        save(destination/"audits"/(row["id"]+".json"),audit); audits.append(audit)
    summary = dict(version=VERSION,status="complete",histories=len(audits),
        domain_covered=sum(a["all_domain_assumptions_covered"] for a in audits),
        true_retained=sum(a["true_candidate_retained"] for a in audits),
        unique_true=sum(a["uniquely_retained_true"] for a in audits),
        unresolved=sum(a["unresolved"] for a in audits),
        candidate_failures=sum(len(a["failures"]) for a in audits),
        candidate_count=sum(a["candidate_count"] for a in audits),
        wrong_candidates_rejected=sum(a["wrong_rejection_count"] for a in audits),
        residual_chi=dict(true_retained=sum(a["residual_chi"]["true_candidate_retained"] for a in audits),
            unique_true=sum(a["residual_chi"]["uniquely_retained_true"] for a in audits),
            unresolved=sum(a["residual_chi"]["unresolved"] for a in audits),
            wrong_candidates_rejected=sum(a["residual_chi"]["wrong_rejection_count"] for a in audits)),
        total_seconds=sum(a["seconds"] for a in audits),audits=audits,
        interpretation="Public reference-feeder finite-candidate experiment; no known-true calibration histories used; simulator truth only checked after decisions")
    save(destination/"summary.json",summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--mode",choices=["generate","run","all"],default="all")
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args(argv)
    if args.mode in ("generate","all"): generate_bundle(args.output,pilot=args.pilot)
    if args.mode in ("run","all"):
        out=execute_bundle(args.output,workers=args.workers)
        print(json.dumps({k:v for k,v in out.items() if k!="audits"},ensure_ascii=False))


if __name__=="__main__": main()
