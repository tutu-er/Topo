"""Read-only statistics for native reference-network EV benchmarks.

No fitting, planning or regenerated measurements occur here.  Calibration
uses calibration records only; evaluation is opened only after its threshold
file has been atomically frozen.  All decisions concern a finite candidate
pool and attained conditional local AC likelihood values.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from ..g_precision import analysis as old
from .. import likelihood as lk
from . import io

METHODS = ("OBS", "CC", "HL", "RAND", "P-SQ", "G-SQ", "G-AQ",
    "P-SQ/F±", "G-SQ/F±", "G-AQ/F±")
ROUTES = ("history", "preFIX", "preJNT", "postJNT")
ALPHA = .1


def signature(supports, n):
    """Canonical nontrivial clades; full/root and leaf edges are implicit."""
    universe=set(range(int(n))); out=set()
    for raw in supports:
        s=tuple(sorted(set(int(j) for j in raw)))
        if not s or not set(s)<=universe:
            raise ValueError("invalid terminal support")
        if 1<len(s)<n:
            out.add(s)
    sets=list(map(set,out))
    if any(a&b and not(a<=b or b<=a) for a in sets for b in sets):
        raise ValueError("crossing clades")
    return tuple(sorted(out,key=lambda s:(-len(s),s)))


def clade_metrics(predicted, truth, n):
    a,b=set(signature(predicted,n)),set(signature(truth,n)); hit=len(a&b)
    precision=hit/len(a) if a else 1.
    recall=hit/len(b) if b else 1.
    f1=2*hit/(len(a)+len(b)) if a or b else 1.
    return dict(exact=a==b,precision=precision,recall=recall,f1=f1,
        predicted_clades=len(a),true_clades=len(b),matched_clades=hit)


def threshold_quantile(values, alpha=ALPHA):
    """Split-calibration order statistic, with infinity when data are scarce.

    Delta=(S_true-min_k S_k)/2, where S is conditional output SSE.  The
    threshold retains candidates satisfying (S_k-min S)/2 <= threshold.
    Exchangeability is a simulation assumption, not a universal guarantee.
    """
    if not 0<float(alpha)<1:
        raise ValueError("alpha must lie strictly between zero and one")
    array=np.asarray(list(values),float)
    if not np.isfinite(array).all() or np.any(array < -1e-8):
        raise ValueError("finite nonnegative calibration Delta values required")
    array=np.maximum(array,0.);m=len(array)
    order=int(math.ceil((m+1)*(1-float(alpha))-1e-12))
    finite=order<=m
    return dict(value=float(np.sort(array)[order-1]) if finite else None,
        finite=finite,n=m,order_statistic_1_based=order,alpha=float(alpha),
        unavailable_value_means="infinity; retain all candidates",
        score="Delta=(conditional_SSE_true-min_candidate_SSE)/2")


def threshold_value(threshold):
    if isinstance(threshold,dict):
        return float(threshold["value"]) if threshold.get("finite") else math.inf
    if threshold is None:
        return math.inf
    value=float(threshold)
    if math.isnan(value) or value<0:
        raise ValueError("nonnegative threshold required")
    return value


def _structures(record, action=None):
    source=(action or {}).get("candidate_union")
    if source is None:
        source=record.get("candidate_structures")
    if source is None:
        source=record.get("passive_pre",{}).get("bank",[])
    if isinstance(source,dict):
        source=[dict(name=k,**v) for k,v in source.items()]
    n=int(record["spec"]["n"])
    result={}
    for row in source:
        result[row["name"]]=dict(name=row["name"],n=int(row.get("n",n)),
            supports=row.get("full_supports",row["supports"]))
    return result


def _action(record, method):
    matches=[(key,a) for key,a in record.get("evaluation",{}).get("actions",{}).items()
        if method in a.get("methods",[])]
    if len(matches)==1:
        return matches[0]
    key=record.get("planning",{}).get("plans",{}).get(method,{}).get("action_key")
    actions=record.get("evaluation",{}).get("actions",{})
    return (key,actions[key]) if key in actions else (None,None)


def _entry_good(entry):
    return bool(entry and entry.get("success",False) and entry.get("usable",True)
        and old.number(entry.get("loss")) is not None)


def route_view(record, method, route):
    if route not in ROUTES:
        raise ValueError("unknown inference route")
    _,action=_action(record,method)
    structures=_structures(record,action if route=="postJNT" else None)
    entries={};failure=None;physical_failure=False;service_voltage_failure=False;calculation_domain_failure=False
    if route=="history":
        entries=record.get("planning",{}).get("history",{}).get("candidates",{})
    elif action is None:
        failure="plan has no completed actual-measurement record"
    else:
        component="fixed" if route=="preFIX" else "joint"
        pre=action.get("pre_refit",{}) or {}
        entries={name:row.get(component,{}) for name,row in pre.get("comparisons",{}).items()}
        if route=="postJNT":
            post=action.get("post_refit",{}) or {}
            for name,row in post.get("comparisons",{}).items():
                if name in entries:
                    raise ValueError("post_refit must contain only newly proposed candidates")
                entries[name]=row.get("joint",{})
        calculation_domain_failure=bool(action.get("physical_generation_failed")
            or action.get("physical",{}).get("voltage_domain_satisfied") is False
            or "actual_voltage_domain" in action.get("failures",{}))
        service_voltage_failure=bool("actual_service_voltage" in action.get("failures",{})
            or action.get("physical",{}).get("service_voltage_09_11_satisfied") is False)
        physical_failure=calculation_domain_failure or service_voltage_failure
        if calculation_domain_failure:
            failure="actual AC generation or numerical/model voltage-domain check failed"
        elif service_voltage_failure:
            failure="executed plan violated declared 0.9..1.1 service-voltage bounds; mathematical AC fits retained"
    failures=[name for name in structures if not _entry_good(entries.get(name))]
    if not structures:
        failure=failure or "candidate pool unavailable"
    losses={name:float(entries[name]["loss"]) for name in structures if _entry_good(entries.get(name))}
    truth=record.get("truth_evaluation_only",{})
    n=int(record["spec"]["n"])
    true_key=signature(truth.get("supports",[]),n) if truth else None
    true_names=[name for name,row in structures.items() if signature(row["supports"],n)==true_key]
    if len(true_names)>1:
        raise ValueError("duplicate structural candidates in saved pool")
    return dict(route=route,structures=structures,entries=entries,losses=losses,
        true_name=true_names[0] if true_names else None,
        truth_present=bool(true_names),truth_known=bool(truth),fit_available=not failures and not failure,
        failed_candidates=failures,failure=failure,physical_failure=physical_failure,
        service_voltage_failure=service_voltage_failure,calculation_domain_failure=calculation_domain_failure)


def decide(view, threshold, *, tie_tolerance=1e-8):
    names=list(view["structures"]);losses=view["losses"]
    available=bool(view["fit_available"] and set(losses)==set(names) and names)
    true=view.get("true_name");truth_present=bool(view.get("truth_present"))
    truth_known=bool(view.get("truth_known",True))
    value=threshold_value(threshold);predicted=None;ties=[];delta=margin=None
    if available:
        best=min(losses.values());eps=tie_tolerance*(1+abs(best))
        ties=sorted(name for name,x in losses.items() if x<=best+eps)
        if len(ties)==1:
            predicted=ties[0]
        retained=sorted(name for name,x in losses.items() if (x-best)/2<=value+eps)
        if truth_present:
            delta=max(0.,(losses[true]-best)/2)
            competitors=[x for name,x in losses.items() if name!=true]
            margin=(min(competitors)-losses[true])/2 if competitors else None
    else:
        # Failed candidate optimizations cannot be used as exclusions.
        retained=sorted(names)
    unresolved=not available or len(retained)!=1
    unique_correct=not unresolved and retained[0]==true
    unique_wrong=not unresolved and truth_known and retained[0]!=true
    return dict(available=available,conditional_available=available and truth_present,
        truth_present=truth_present,truth_known=truth_known,true_name=true,predicted=predicted,argmin_ties=ties,
        numerical_tie=available and len(ties)>1,
        point_error=(predicted!=true) if predicted is not None and truth_known else None,
        conditional_point_error=(predicted!=true) if predicted is not None and truth_present else None,
        retained=retained,unresolved=unresolved,unique_correct=unique_correct,
        unique_wrong=unique_wrong,false_reject=(true not in retained) if truth_present else None,
        Delta=delta,true_margin_half_SSE=margin,candidate_count=len(names),
        retained_count=len(retained),fit_failure=not available,
        physical_failure=bool(view.get("physical_failure")),
        service_voltage_failure=bool(view.get("service_voltage_failure")),
        calculation_domain_failure=bool(view.get("calculation_domain_failure")),
        optimization_failure=bool(view.get("failed_candidates") and not view.get("failure")),
        failed_candidates=view.get("failed_candidates",[]),failure=view.get("failure"),
        threshold_value=None if not math.isfinite(value) else value,
        threshold_is_infinite=not math.isfinite(value),
        eliminated_wrong_candidates=sum(name!=true and name not in retained for name in names),
        wrong_candidate_count=sum(name!=true for name in names))


def network_name(spec):
    """A physical network label, with explicit fallback for small test records."""
    return str(spec.get("network", "n"+str(int(spec["n"]))))


def cell_key(method,n,source_mode,route,network=None):
    return "|".join(map(str,(method,network or "n"+str(int(n)),int(n),source_mode,route)))


def source_blind_key(method,n,route,network=None):
    return "|".join(map(str,(method,network or "n"+str(int(n)),int(n),route)))


def declared_network_sizes(config):
    """Use the frozen network list/static constructors, never completed files."""
    names=tuple(config.get("network_names",()))
    if names:
        if len(names)!=len(set(names)):
            raise ValueError("duplicate declared network names")
        supplied=config.get("network_observation_counts")
        if supplied is not None:
            if set(supplied)!=set(names):raise ValueError("network count declaration differs from network list")
            sizes={name:int(supplied[name]) for name in names}
        else:
            from .networks import build_network
            sizes={name:len(build_network(name)["observation_bus_ids"]) for name in names}
    else:
        # Small fixtures and retained legacy-schema replay only.
        sizes={"n"+str(int(n)):int(n) for n in config["terminal_counts"]}
    if not sizes or any(n<2 for n in sizes.values()):
        raise ValueError("declared nonempty networks with at least two measurements required")
    return sizes


def phase_count(config,phase):
    if "phase_counts" in config:
        count=int(config["phase_counts"][phase])
    else:count=int(config[phase+"_clusters"])
    if count<1:raise ValueError("positive declared date-window count required")
    return count


def condition_manifest(config,phase):
    sizes=declared_network_sizes(config)
    return [dict(phase=phase,cluster_index=i,network=name,n=n,source_mode=source)
        for i in range(phase_count(config,phase)) for name,n in sizes.items()
        for source in config["source_modes"]]


def calibration_thresholds(records, *, methods=METHODS, alpha=ALPHA,source_modes=None):
    cells=defaultdict(list); seen=set()
    records=list(records)
    for record in records:
        spec=record["spec"]
        if spec.get("phase") not in ("calibration","cal"):
            raise ValueError("calibration thresholds may read calibration records only")
        condition=(spec["cluster_index"],network_name(spec),spec["source_mode"])
        if condition in seen:
            raise ValueError("duplicate calibration condition")
        seen.add(condition)
        for method in methods:
            for route in ROUTES:
                key=cell_key(method,spec["n"],spec["source_mode"],route,network_name(spec))
                decision=decide(route_view(record,method,route),math.inf)
                cells[key].append(decision)
    result={}
    for key,rows in sorted(cells.items()):
        valid=[r["Delta"] for r in rows if r["conditional_available"]]
        result[key]=dict(**threshold_quantile(valid,alpha),n_all=len(rows),
            candidate_missing=sum(r["truth_known"] and not r["truth_present"] for r in rows),
            truth_unknown=sum(not r["truth_known"] for r in rows),
            fit_failure=sum(r["fit_failure"] for r in rows),
            conditional_usable=len(valid))
    source_modes=tuple(sorted({r["spec"]["source_mode"] for r in records})) if source_modes is None else tuple(source_modes)
    blind={}
    for method in methods:
        for network,n in sorted({(network_name(r["spec"]),int(r["spec"]["n"])) for r in records}):
            for route in ROUTES:
                strata={source:result.get(cell_key(method,n,source,route,network),
                    dict(**threshold_quantile([],alpha),n_all=0,candidate_missing=0,
                        truth_unknown=0,fit_failure=0,conditional_usable=0)) for source in source_modes}
                values=[threshold_value(v) for v in strata.values()]
                value=max(values,default=math.inf);finite=math.isfinite(value)
                key=source_blind_key(method,n,route,network)
                blind[key]=dict(value=value if finite else None,finite=finite,
                    source_modes=list(source_modes),alpha=float(alpha),
                    network=network,n=n,
                    source_thresholds={source:dict(cell_key=cell_key(method,n,source,route,network),**row)
                        for source,row in strata.items()},
                    n_independent_date_clusters=len({r["spec"]["cluster_index"] for r in records if network_name(r["spec"])==network}),
                    rule="maximum of preregistered source-stratum thresholds; any infinite stratum makes the applied threshold infinite",
                    lookup="method, physical network, n, route only; actual source label is never read")
    return dict(schema="reference_benchmark_thresholds_v1",alpha=float(alpha),cells=result,
        cells_meaning="source-stratified diagnostic thresholds; never selected using an evaluation source label",
        source_blind_applied=blind,
        source_phase="calibration",n_condition_records=len(records),
        n_independent_date_clusters=len({r["spec"]["cluster_index"] for r in records}),
        scope="conditional on candidate inclusion and required route computations; attained conditional local likelihood",
        coverage_assumption="exchangeable independent date clusters within each frozen n/source/method/route cell",
        probabilities_are_not_topology_posteriors=True)


def decision_rows(records, thresholds, *, methods=METHODS):
    out=[]
    for record in records:
        spec=record["spec"]
        for method in methods:
            for route in ROUTES:
                key=source_blind_key(method,spec["n"],route,network_name(spec))
                view=route_view(record,method,route)
                row=decide(view,thresholds.get("source_blind_applied",{}).get(key))
                out.append(dict(spec=spec,cluster_index=spec["cluster_index"],n=spec["n"],network=network_name(spec),
                    source_mode=spec["source_mode"],method=method,route=route,
                    applied_threshold_key=key,source_label_used_for_threshold=False,
                    zero_equivalence=zero_equivalence_diagnostic(record,view),**row))
    return out


def _rate(numerator, denominator):
    return numerator/denominator if denominator else None


def summarize_decisions(rows):
    rows=list(rows);total=len(rows)
    available=[r for r in rows if r["available"]]
    conditional=[r for r in rows if r["conditional_available"]]
    point=[r for r in available if r["point_error"] is not None]
    conditional_point=[r for r in conditional if r["conditional_point_error"] is not None]
    present=[r for r in rows if r["truth_present"]]
    errors=sum(r["point_error"] is True for r in point)
    conditional_errors=sum(r["conditional_point_error"] is True for r in conditional_point)
    false_reject=sum(r["false_reject"] is True for r in present)
    return dict(n_total_conditions=total,n_independent_date_clusters=len({r["cluster_index"] for r in rows}),
        candidate_missing=sum(r["truth_known"] and not r["truth_present"] for r in rows),
        truth_unknown=sum(not r["truth_known"] for r in rows),
        fit_failure=sum(r["fit_failure"] for r in rows),available=len(available),
        required_route_unavailable=sum(r["fit_failure"] for r in rows),
        physical_failure=sum(r.get("physical_failure",False) for r in rows),
        service_voltage_failure=sum(r.get("service_voltage_failure",False) for r in rows),
        calculation_domain_failure=sum(r.get("calculation_domain_failure",False) for r in rows),
        optimization_failure=sum(r.get("optimization_failure",False) for r in rows),
        conditional_available=len(conditional),numerical_ties=sum(r["numerical_tie"] for r in rows),
        top1_wrong=errors,top1_evaluable=len(point),top1_error_rate=_rate(errors,len(point)),
        top1_wrong_all_denominator_rate=_rate(errors,total),
        conditional_top1_wrong=conditional_errors,conditional_top1_evaluable=len(conditional_point),
        conditional_top1_error_rate=_rate(conditional_errors,len(conditional_point)),
        unique_correct=sum(r["unique_correct"] for r in rows),
        unique_wrong=sum(r["unique_wrong"] for r in rows),unresolved=sum(r["unresolved"] for r in rows),
        false_reject=false_reject,truth_present=len(present),
        false_reject_given_inclusion=_rate(false_reject,len(present)),
        false_reject_all_denominator_rate=_rate(false_reject,total),
        false_reject_given_usable_inclusion=_rate(sum(r["false_reject"] is True for r in conditional),len(conditional)),
        eliminated_wrong_candidates=sum(r["eliminated_wrong_candidates"] for r in rows),
        wrong_candidate_count=sum(r["wrong_candidate_count"] for r in rows),
        infinite_threshold_conditions=sum(r["threshold_is_infinite"] for r in rows),
        structural_refinement_coverage=sum(r.get("zero_equivalence",{}).get("structural_refinement_coverage") is True for r in rows),
        legal_fitted_contraction_coverage=sum(r.get("zero_equivalence",{}).get("fitted_contraction_truth_coverage") is True for r in rows),
        strict_missing_but_legal_contraction_covered=sum(not r["truth_present"] and r.get("zero_equivalence",{}).get("fitted_contraction_truth_coverage") is True for r in rows),
        contraction_diagnostics_do_not_replace_strict_endpoint=True)


def cluster_bootstrap(rows, statistic, *, seed=2026101801, replicates=2000, confidence=.95):
    """Resample complete date clusters; statistic returns a finite scalar/None."""
    rows=list(rows)
    groups=defaultdict(list)
    for row in rows:
        groups[row["cluster_index"]].append(row)
    keys=sorted(groups,key=str);rng=np.random.default_rng(seed);values=[]
    estimate=statistic(list(rows))
    if not keys:
        return dict(estimate=estimate,lower=None,upper=None,n_clusters=0,valid_replicates=0,
            requested_replicates=replicates,seed=seed)
    for _ in range(replicates):
        sample=[r for index in rng.integers(0,len(keys),len(keys)) for r in groups[keys[index]]]
        value=statistic(sample)
        if old.number(value) is not None:
            values.append(float(value))
    tail=(1-confidence)/2
    lower,upper=np.quantile(values,[tail,1-tail]) if values else (None,None)
    return dict(estimate=estimate,lower=lower,upper=upper,n_clusters=len(keys),
        valid_replicates=len(values),requested_replicates=replicates,seed=seed,
        confidence=confidence,unit="date cluster",method="percentile paired cluster bootstrap")


def paired_rows(rows, left="P-SQ", right="G-SQ", *, route="postJNT"):
    index={}
    for row in rows:
        if row["route"]!=route or row["method"] not in (left,right):
            continue
        key=(row["cluster_index"],row.get("network",network_name(row.get("spec",row))),row["n"],row["source_mode"])
        if row["method"] in index.setdefault(key,{}):
            raise ValueError("duplicate paired decision row")
        index[key][row["method"]]=row
    return [dict(cluster_index=key[0],network=key[1],n=key[2],source_mode=key[3],
        left=pair.get(left),right=pair.get(right)) for key,pair in sorted(index.items(),key=lambda x:str(x[0]))]


def paired_summary(rows, left="P-SQ", right="G-SQ", *, route="postJNT", replicates=2000, seed=2026101801):
    pairs=paired_rows(rows,left,right,route=route)
    def rate_delta(sample,metric):
        usable=[p for p in sample if p["left"] is not None and p["right"] is not None]
        return old.mean(float(p["right"][metric])-float(p["left"][metric]) for p in usable)
    complete=[p for p in pairs if p["left"] is not None and p["right"] is not None]
    evaluable=[p for p in complete if p["left"]["conditional_point_error"] is not None
        and p["right"]["conditional_point_error"] is not None]
    def error_delta(sample):
        usable=[p for p in sample if p["left"] is not None and p["right"] is not None
            and p["left"]["conditional_point_error"] is not None and p["right"]["conditional_point_error"] is not None]
        return old.mean(float(p["right"]["conditional_point_error"])-float(p["left"]["conditional_point_error"]) for p in usable)
    return dict(left=left,right=right,route=route,n_all_pairs=len(pairs),
        n_both_recorded=len(complete),n_conditional_point_pairs=len(evaluable),
        right_fixes_left=sum(p["left"]["conditional_point_error"] and not p["right"]["conditional_point_error"] for p in evaluable),
        right_breaks_left=sum(not p["left"]["conditional_point_error"] and p["right"]["conditional_point_error"] for p in evaluable),
        conditional_point_error_delta=cluster_bootstrap(pairs,error_delta,replicates=replicates,seed=seed),
        unique_correct_delta=cluster_bootstrap(pairs,lambda s:rate_delta(s,"unique_correct"),replicates=replicates,seed=seed),
        unresolved_delta=cluster_bootstrap(pairs,lambda s:rate_delta(s,"unresolved"),replicates=replicates,seed=seed),
        difference_sign="right minus left; negative error and positive unique-correct favor right")


def native_topology_metrics(record):
    truth=record.get("truth_evaluation_only")
    if not truth:
        return dict(truth_available=False)
    n=int(record["spec"]["n"]);native=record.get("passive_pre",{}).get("native",{})
    variants={}
    for kind in ("rnj","wzzt"):
        variants[kind]=[dict(origin=r.get("origin"),**clade_metrics(r["supports"],truth["supports"],n))
            for r in native.get(kind,[])]
    chosen={}
    rnjs=native.get("rnj",[])
    if rnjs:
        row=next((r for r in rnjs if r.get("origin")=="rnj:raw_effective_pq:tolerance=0"),rnjs[0])
        chosen["RNJ_raw_tolerance0"]=dict(origin=row.get("origin"),**clade_metrics(row["supports"],truth["supports"],n))
    wzzt=native.get("wzzt",[])
    if wzzt:
        row=max(wzzt,key=lambda r:int(r["origin"].rsplit(":",1)[1]))
        chosen["WZZT_last_accepted"]=dict(origin=row.get("origin"),**clade_metrics(row["supports"],truth["supports"],n))
    return dict(truth_available=True,selected_native=chosen,variants=variants,
        rule="raw RNJ tolerance zero, or first available fallback; final accepted WZZT; no truth-based selection")


def rx_error(record,view):
    true=view.get("true_name");truth=record.get("truth_evaluation_only",{})
    entry=view.get("entries",{}).get(true,{}) if true else {}
    state=entry.get("state")
    if not state or not view.get("truth_present"):
        return dict(available=False,reason="true topology absent or fitted state unavailable")
    try:
        truth_supports=[tuple(s) for s in truth["supports"]]
        candidate=lk.RadialCandidate.from_supports(true,record["spec"]["n"],view["structures"][true]["supports"])
        if len(truth_supports)!=len(truth["r"]) or len(truth_supports)!=len(truth["x"]):
            raise ValueError("true edge/support arrays do not align")
        rows={s:i for i,s in enumerate(truth_supports)}
        order=[rows[s] for s in candidate.supports]
        output=dict(available=True,fit_success=_entry_good(entry),n_edges=len(order),
            zero_edge_rule="absolute errors for exact-zero encoded meter edges; relative errors only on positive truth edges")
        actual_r=np.asarray(truth["r"],float)[order]
        actual_x=np.asarray(truth["x"],float)[order]
        output["virtual_zero_edge_count"]=int(np.count_nonzero((actual_r==0)&(actual_x==0)))
        for key in ("r","x"):
            actual=np.asarray(truth[key],float)[order];estimate=np.asarray(state[key],float)
            if (estimate.shape!=actual.shape or not np.isfinite(estimate).all()
                    or not np.isfinite(actual).all() or np.any(actual<0)):
                raise ValueError("finite aligned nonnegative truth edge values required")
            error=estimate-actual
            positive=actual>0;zero=actual==0
            relative=np.abs(error[positive])/actual[positive]
            abszero=np.abs(error[zero]);absolute=np.abs(error)
            output[key]=dict(
                mean_absolute_relative=float(np.mean(relative)) if relative.size else None,
                median_absolute_relative=float(np.median(relative)) if relative.size else None,
                maximum_absolute_relative=float(np.max(relative)) if relative.size else None,
                rmse_pu=float(np.sqrt(np.mean(error*error))),
                n_positive_edges=int(positive.sum()),n_zero_edges=int(zero.sum()),
                positive_edges_relative=dict(n_edges=int(positive.sum()),
                    mean=float(np.mean(relative)) if relative.size else None,
                    median=float(np.median(relative)) if relative.size else None,
                    maximum=float(np.max(relative)) if relative.size else None),
                all_edges_absolute_pu=dict(n_edges=len(error),mean=float(absolute.mean()),
                    maximum=float(absolute.max()),rmse=float(np.sqrt(np.mean(error*error)))),
                zero_truth_edges_absolute_pu=dict(n_edges=int(zero.sum()),
                    mean=float(abszero.mean()) if abszero.size else None,
                    maximum=float(abszero.max()) if abszero.size else None,
                    rmse=float(np.sqrt(np.mean(error[zero]**2))) if abszero.size else None))
        return output
    except Exception as exc:
        return dict(available=False,reason=type(exc).__name__+": "+str(exc))


def contracted_supports(shape,state,tolerance):
    """Contract only fitted near-zero nontrivial edges, using a frozen pu tol."""
    if not np.isfinite(tolerance) or tolerance<0:raise ValueError("finite nonnegative contraction tolerance required")
    c=lk.RadialCandidate.from_supports(shape["name"],shape["n"],shape["supports"])
    r,x=np.asarray(state["r"],float),np.asarray(state["x"],float)
    if r.shape!=(c.m,) or x.shape!=r.shape or not np.isfinite(r).all() or not np.isfinite(x).all():
        raise ValueError("fitted line arrays must align with complete candidate supports")
    support=[];removed=[]
    for s,z in zip(c.supports,np.hypot(r,x)):
        if 1<len(s)<c.n:
            (removed if z<=tolerance else support).append(tuple(s))
    return signature(support,c.n),removed


def zero_equivalence_diagnostic(record,view):
    """Evaluation-only coverage diagnostics; primary topology tests unchanged."""
    truth=record.get("truth_evaluation_only")
    tolerance=float(record.get("config",{}).get("zero_equivalence_tolerance_pu",1e-8))
    out=dict(tolerance_impedance_magnitude_pu=tolerance,truth_known=bool(truth),
        structural_refinement_coverage=None,fitted_contraction_truth_coverage=None,
        structural_refinement_candidates=[],fitted_contraction_truth_candidates=[],
        legal_converged_states_assessed=0,state_failures={},
        scope="diagnostic only; no truth-based candidate insertion, fit, label selection, or threshold change")
    if not truth:return out
    n=int(record["spec"]["n"]);key=signature(truth["supports"],n);true_set=set(key)
    for name,shape in view["structures"].items():
        if true_set<=set(signature(shape["supports"],n)):
            out["structural_refinement_candidates"].append(name)
        entry=view["entries"].get(name,{})
        if not _entry_good(entry) or not entry.get("state"):continue
        try:
            contracted,_=contracted_supports(shape,entry["state"],tolerance)
            out["legal_converged_states_assessed"]+=1
            if contracted==key:out["fitted_contraction_truth_candidates"].append(name)
        except Exception as exc:out["state_failures"][name]=repr(exc)
    out["structural_refinement_coverage"]=bool(out["structural_refinement_candidates"])
    if out["legal_converged_states_assessed"]:
        out["fitted_contraction_truth_coverage"]=bool(out["fitted_contraction_truth_candidates"])
    return out


def final_topology_metrics(record,method,route):
    """Point-estimate clade F1; retain failures/ties as unscored conditions."""
    view=route_view(record,method,route)
    decision=decide(view,math.inf)
    truth=record.get("truth_evaluation_only")
    chosen=decision["predicted"]
    out=dict(available=False,method=method,route=route,
        cluster_index=record["spec"]["cluster_index"],network=network_name(record["spec"]),
        n=int(record["spec"]["n"]),source_mode=record["spec"]["source_mode"],
        fit_failure=decision["fit_failure"],numerical_tie=decision["numerical_tie"],
        truth_present=decision["truth_present"],predicted=chosen)
    if not truth or chosen is None or not decision["available"]:
        out["reason"]="missing truth, unavailable required fit/physics, or tied point labels"
        return out
    metrics=clade_metrics(view["structures"][chosen]["supports"],truth["supports"],record["spec"]["n"])
    out=dict(out,available=True,**metrics)
    tolerance=float(record.get("config",{}).get("zero_equivalence_tolerance_pu",1e-8))
    try:
        contracted,removed=contracted_supports(view["structures"][chosen],view["entries"][chosen]["state"],tolerance)
        out["contracted"]=dict(available=True,tolerance_impedance_magnitude_pu=tolerance,
            removed_nontrivial_supports=removed,removed_count=len(removed),
            **clade_metrics(contracted,truth["supports"],record["spec"]["n"]))
    except Exception as exc:out["contracted"]=dict(available=False,reason=repr(exc))
    return out


def summarize_clades(rows,*,bootstrap=2000,seed=2026101801):
    rows=list(rows);valid=[r for r in rows if r.get("available")]
    result=dict(n_total_conditions=len(rows),n_date_clusters=len({r["cluster_index"] for r in rows}),
        scored=len(valid),unscored=len(rows)-len(valid),
        fit_failure=sum(r.get("fit_failure",False) for r in rows),
        numerical_ties=sum(r.get("numerical_tie",False) for r in rows),
        exact=sum(r["exact"] for r in valid),
        exact_rate_all=_rate(sum(r["exact"] for r in valid),len(rows)),
        precision=old.distribution(r["precision"] for r in valid),
        recall=old.distribution(r["recall"] for r in valid),
        f1=old.distribution(r["f1"] for r in valid),
        failed_or_tied_routes_are_not_assigned_zero_f1=True)
    result["mean_f1_interval"]=cluster_bootstrap(rows,
        lambda x:old.mean(r["f1"] for r in x if r.get("available")),
        replicates=bootstrap,seed=seed)
    contracted=[dict(r["contracted"],cluster_index=r["cluster_index"]) for r in rows
        if r.get("available") and r.get("contracted",{}).get("available")]
    result["contracted_diagnostic"]=dict(scored=len(contracted),unscored=len(rows)-len(contracted),
        exact=sum(r["exact"] for r in contracted),
        exact_rate_all=_rate(sum(r["exact"] for r in contracted),len(rows)),
        f1=old.distribution(r["f1"] for r in contracted),
        tolerance_impedance_magnitude_pu=sorted({r["tolerance_impedance_magnitude_pu"] for r in contracted}),
        strict_endpoint_unchanged=True,
        mean_f1_interval=cluster_bootstrap(rows,
            lambda x:old.mean(r["contracted"]["f1"] for r in x
                if r.get("available") and r.get("contracted",{}).get("available")),
            replicates=bootstrap,seed=seed))
    return result


def _rank(values):
    array=np.asarray(values,float);order=np.argsort(array,kind="stable");result=np.empty(len(array),float)
    i=0
    while i<len(order):
        j=i+1
        while j<len(order) and array[order[j]]==array[order[i]]:j+=1
        result[order[i:j]]=(i+j-1)/2;i=j
    return result


def spearman(left,right):
    if len(left)<3:
        return None
    x,y=_rank(left),_rank(right)
    if not np.ptp(x) or not np.ptp(y):
        return None
    return float(np.corrcoef(x,y)[0,1])


def score_ranking(record):
    plans=record.get("planning",{}).get("plans",{})
    tasks=record.get("planning",{}).get("tasks",{})
    output={}
    for objective in ("G-SQ","G-AQ"):
        rows=[]
        if not tasks.get(objective):
            continue
        for method,plan in plans.items():
            if "power" not in plan:
                continue
            surrogate,_=old.task_scores(tasks[objective],plan["power"])
            decision=decide(route_view(record,method,"preJNT"),math.inf)
            nominal_key="incremental_score" if objective=="G-SQ" else "total_score"
            nominal=old.number(record.get("planning",{}).get("nominal_AC",{}).get(method,{}).get(nominal_key))
            rows.append(dict(method=method,surrogate=surrogate,nominal_AC=nominal,
                actual_true_margin_half_SSE=decision["true_margin_half_SSE"],
                actual_available=decision["conditional_available"]))
        valid=[r for r in rows if r["actual_available"] and r["actual_true_margin_half_SSE"] is not None]
        nominal=[r for r in valid if r["nominal_AC"] is not None]
        baseline=next((r for r in rows if r["method"]=="P-SQ"),None)
        own=next((r for r in rows if r["method"]==objective),None)
        delta={}
        if baseline and own:
            for key in ("surrogate","nominal_AC","actual_true_margin_half_SSE"):
                x,y=old.number(baseline[key]),old.number(own[key])
                delta[key]=y-x if x is not None and y is not None else None
        output[objective]=dict(rows=rows,n_paired_actual=len(valid),
            spearman_surrogate_vs_actual=spearman([r["surrogate"] for r in valid],
                [r["actual_true_margin_half_SSE"] for r in valid]),
            spearman_nominal_AC_vs_actual=spearman([r["nominal_AC"] for r in nominal],
                [r["actual_true_margin_half_SSE"] for r in nominal]),
            own_minus_P_SQ=delta,
            scope="same saved final plans scored under one frozen objective; true margin evaluation only; shared actions remain ties")
    return output


def forecast_errors(record):
    context=record.get("planning",{}).get("context",{})
    forecast=context.get("forecast",{});actions=record.get("evaluation",{}).get("actions",{})
    actual=next((v.get("truth_inputs_evaluation_only") for v in actions.values()
        if v.get("truth_inputs_evaluation_only")),None)
    if not actual or not forecast:
        return dict(available=False)
    base=float(record.get("config",{}).get("base_kw",record.get("provenance",{}).get("base_kw",20.)))
    def errors(predicted,true):
        x,y=np.asarray(predicted,float),np.asarray(true,float)
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            return dict(available=False)
        d=(x-y)*base
        return dict(available=True,mean_error_kw=float(np.mean(d)),mae_kw=float(np.mean(np.abs(d))),
            rmse_kw=float(np.sqrt(np.mean(d*d))))
    result=dict(available=True,load={key:errors(forecast[key],actual[key]) for key in ("p","q")},source={})
    for name,row in forecast.get("source_by_candidate",{}).items():
        result["source"][name]={key:errors(row[key],actual[key]) for key in ("source_p","source_q")}
    return result


def cone_contains(r,x,bounds,tolerance=1e-10):
    """Closed ratio cone includes exact (0,0), without dividing by R."""
    r,x=np.asarray(r,float),np.asarray(x,float)
    if r.shape!=x.shape or not np.isfinite(r).all() or not np.isfinite(x).all():return False
    if bounds is None:return True
    lo,hi=map(float,bounds)
    eps=tolerance*(1+np.maximum(abs(r),abs(x)))
    return bool(np.all(x>=lo*r-eps) and np.all(x<=hi*r+eps))


def _replay_state(candidate,entry,obs,meters,history_t,options,audit,label,*,history_only=False):
    state=entry.get("state")
    if not state:return
    audit.ac_calls+=1
    physical=lk.ac_predict(candidate,np.asarray(state["r"]),np.asarray(state["x"]),
        obs["p"]-meters["bias_p"],obs["q"]-meters["bias_q"],
        np.asarray(state["source_p"]),np.asarray(state["source_q"]),obs["root_v"])
    residual=dict(v=(physical["v"]+meters["bias_v"]-obs["v"])/meters["v_sigma"],
        master_p=(physical["master_p"]-obs["master_p"])/meters["master_p_sigma"],
        master_q=(physical["master_q"]-obs["master_q"])/meters["master_q_sigma"])
    total=0.
    slices=[("history",slice(None,history_t))]
    if not history_only:slices.append(("future",slice(history_t,None)))
    for block,index in slices:
        channels={key:float(np.sum(values[index]**2)) for key,values in residual.items()}
        loss=sum(channels.values());total+=loss
        saved=entry.get("blocks",{}).get(block,{})
        audit.close("AC_block_SSE",loss,saved.get("loss"),dict(context=label,block=block))
        for key,value in channels.items():
            audit.close("AC_channel_SSE",value,saved.get("channel_loss",{}).get(key),dict(context=label,block=block,channel=key))
    audit.close("AC_total_SSE",total,entry.get("loss"),label)
    audit.check("AC_equations",physical["ac_equation_residual"]<=1e-8 and physical["power_balance_residual"]<=1e-8,label)
    audit.maxima["AC_equation_residual"]=max(audit.maxima["AC_equation_residual"],physical["ac_equation_residual"])
    audit.maxima["AC_power_balance_residual"]=max(audit.maxima["AC_power_balance_residual"],physical["power_balance_residual"])
    if entry.get("usable",False):
        checks=dict(r=old.in_bounds(state["r"],options.get("r_bounds")),x=old.in_bounds(state["x"],options.get("x_bounds")),
            ratio=cone_contains(state["r"],state["x"],options.get("ratio_bounds")),
            source_p=old.in_bounds(state["source_p"],options.get("source_p_bounds")),
            source_q=old.in_bounds(state["source_q"],options.get("source_q_bounds")),
            voltage=old.in_bounds(np.abs(physical["voltage"]),options.get("voltage_bounds")))
        audit.check("AC_usable_bounds",all(checks.values()),dict(context=label,checks=checks))


def audit_record(record,audit):
    """Forward-only replay of saved states and actually executed measurements."""
    if "observed_history" not in record:
        return
    historical,hm=old.obs_arrays(record["observed_history"],record["history_meter"])
    H=len(historical["p"]);options=record.get("fit_options",{})
    pre=_structures(record)
    old.audit_operations(record,audit)
    _audit_coefficients(record,audit)
    for method,domains in record.get("planning",{}).get("families",{}).items():
        for domain,result in domains.items():
            score=old.number(result.get("score"));parent=old.number(result.get("best_parent_score"))
            lower=old.number(result.get("objective_lower"));upper=old.number(result.get("objective_upper"))
            if score is not None and lower is not None and upper is not None:
                audit.check("fixed_surrogate_bounds",lower<=score+1e-6 and score<=upper+1e-6,
                    dict(method=method,domain=domain,score=score,lower=lower,upper=upper))
            if score is not None and parent is not None:
                audit.check("parent_plan_inheritance",score+1e-8*(1+abs(parent))>=parent,
                    dict(method=method,domain=domain,score=score,parent_score=parent))
    cost=deployment_costs(record,())
    signed=cost["signed_unallocated_seconds"]
    if signed is not None:
        audit.check("selection_cost_component_closure",signed>=-1e-5,
            dict(spec=record["spec"],signed_residual=signed,components=cost["components_seconds"]))
    for name,entry in record.get("planning",{}).get("history",{}).get("candidates",{}).items():
        if not entry.get("state") or name not in pre:continue
        try:
            c=lk.RadialCandidate.from_supports(name,pre[name]["n"],pre[name]["supports"])
            _replay_state(c,entry,historical,hm,H,options,audit,dict(kind="history",candidate=name,spec=record["spec"]),history_only=True)
        except Exception as exc:
            audit.check("AC_replay_exception",False,dict(spec=record["spec"],candidate=name,exception=repr(exc)))
    first_inputs=None;first_white=None
    for key,action in record.get("evaluation",{}).get("actions",{}).items():
        if "observations" not in action:continue
        inputs=action.get("truth_inputs_evaluation_only",{})
        white=action.get("actual_white_noise",{})
        if first_inputs is None:first_inputs=inputs;first_white=white
        else:
            for channel in ("p","q","source_p","source_q","root_v"):
                audit.close("paired_actual_background",inputs.get(channel),first_inputs.get(channel),dict(action=key,channel=channel))
            for channel in ("p","q","v","master_p","master_q"):
                audit.close("paired_future_white_noise",white.get(channel),first_white.get(channel),dict(action=key,channel=channel))
        obs,meters=old.join_blocks([(historical,hm),old.obs_arrays(action["observations"],record["future_meter"])])
        structures=_structures(record,action)
        for stage in ("pre_refit","post_refit"):
            for name,comparison in (action.get(stage) or {}).get("comparisons",{}).items():
                if name not in structures:
                    audit.check("saved_candidate_structure",False,dict(action=key,candidate=name));continue
                c=lk.RadialCandidate.from_supports(name,structures[name]["n"],structures[name]["supports"])
                for route in ("fixed","joint"):
                    entry=comparison.get(route,{})
                    if not entry.get("state"):continue
                    try:_replay_state(c,entry,obs,meters,H,options,audit,dict(spec=record["spec"],action=key,stage=stage,candidate=name,route=route))
                    except Exception as exc:
                        audit.check("AC_replay_exception",False,dict(spec=record["spec"],action=key,stage=stage,candidate=name,route=route,exception=repr(exc)))
        for name,entry in (action.get("post_history") or {}).get("candidates",{}).items():
            if not entry.get("state"):continue
            try:
                row=structures[name];c=lk.RadialCandidate.from_supports(name,row["n"],row["supports"])
                _replay_state(c,entry,historical,hm,H,options,audit,dict(kind="post_history",action=key,candidate=name),history_only=True)
            except Exception as exc:
                audit.check("AC_replay_exception",False,dict(action=key,candidate=name,exception=repr(exc)))
        _audit_actual_measurement(record,key,action,audit)


def _audit_coefficients(record,audit):
    context=record.get("planning",{}).get("context",{})
    responses=context.get("zero_responses",{})
    if not responses:
        return
    local=deepcopy(record);target=local["planning"]["context"]["zero_responses"]
    try:
        if any("jacobian" not in row for row in responses.values()):
            forecast=context["forecast"]
            for name,row in responses.items():
                if "jacobian" in row:continue
                shape=_structures(record)[name];c=lk.RadialCandidate.from_supports(name,shape["n"],shape["supports"])
                state=context["states"][name];source=forecast["source_by_candidate"][name]
                physical=lk.ac_predict(c,np.asarray(state["r"]),np.asarray(state["x"]),
                    np.asarray(forecast["p"]),np.asarray(forecast["q"]),
                    np.asarray(source["source_p"]),np.asarray(source["source_q"]),np.asarray(forecast["root_v"]))
                audit.ac_calls+=1
                mean=np.column_stack((physical["v"],physical["master_p"],physical["master_q"]))
                audit.close("zero_response_replay",mean,row["mean"],dict(candidate=name))
                derivative=lk._response_derivatives(c,physical,True)
                columns=slice(2*c.m,2*c.m+c.n)
                target[name]["jacobian"]=np.concatenate((derivative["v"][:,:,columns],
                    derivative["master_p"][:,None,columns],derivative["master_q"][:,None,columns]),axis=1)/float(record.get("config",{}).get("base_kw",20.))
        old.audit_coefficients(local,audit)
    except Exception as exc:
        audit.check("coefficient_replay_exception",False,dict(spec=record["spec"],exception=repr(exc)))


def _audit_actual_measurement(record,key,action,audit):
    truth=record.get("truth_evaluation_only",{});inputs=action.get("truth_inputs_evaluation_only",{})
    if not truth or not inputs:return
    try:
        n=record["spec"]["n"];c=lk.RadialCandidate.from_supports("replay_truth",n,truth["supports"])
        base=float(record.get("config",{}).get("base_kw",20.))
        p=np.asarray(inputs["p"])+np.asarray(action["power"])/base
        physical=lk.ac_predict(c,np.asarray(truth["r"]),np.asarray(truth["x"]),p,np.asarray(inputs["q"]),
            np.asarray(inputs["source_p"]),np.asarray(inputs["source_q"]),np.asarray(inputs["root_v"]))
        audit.ac_calls+=1;white=action.get("actual_white_noise")
        if not isinstance(white,dict):
            raise ValueError("actual_white_noise must be stored by channel")
        for channel in ("p","q","v","master_p","master_q"):
            value=np.asarray(physical[channel])+np.asarray(record["future_meter"][channel+"_sigma"])*np.asarray(white[channel])
            if channel in ("p","q","v"):
                value+=np.asarray(record["future_meter"].get("bias_"+channel,0.))
            audit.close("actual_meter_generation",value,action["observations"][channel],dict(spec=record["spec"],action=key,channel=channel),atol=1e-10)
        audit.check("actual_AC_equations",physical["ac_equation_residual"]<=1e-8 and physical["power_balance_residual"]<=1e-8,dict(action=key))
    except Exception as exc:
        audit.check("actual_AC_replay_exception",False,dict(spec=record["spec"],action=key,exception=repr(exc)))


def freeze_thresholds(output,*,alpha=ALPHA,methods=METHODS,expected_conditions=None):
    """This function opens calibration files only, never evaluation files."""
    output=Path(output);paths=sorted((output/"calibration").glob("*.json.gz"))
    records=[io.read(path) for path in paths]
    if expected_conditions is not None and len(records)!=expected_conditions:
        raise RuntimeError("calibration incomplete: %s/%s conditions"%(len(records),expected_conditions))
    policy=io.read(output/"freeze.json")
    config=policy.get("config",{})
    if config.get("network_names"):
        declared=condition_manifest(config,"calibration")
        expected={(r["cluster_index"],r["network"],r["source_mode"],r["n"]) for r in declared}
        actual={(r["spec"]["cluster_index"],network_name(r["spec"]),r["spec"]["source_mode"],int(r["spec"]["n"])) for r in records}
        if actual!=expected or len(records)!=len(expected):
            raise RuntimeError("calibration records differ from frozen condition manifest")
    result=calibration_thresholds(records,methods=methods,alpha=alpha,
        source_modes=policy.get("config",{}).get("source_modes"))
    result.update(calibration_file_sha256={p.relative_to(output).as_posix():io.digest(p) for p in paths},
        analysis_sha256=io.digest(Path(__file__)),policy_identity=policy.get("identity"))
    identity=sha256(json.dumps(io.serial(result),sort_keys=True).encode()).hexdigest()
    result["identity"]=identity
    path=output/"thresholds.json"
    if path.exists():
        if io.read(path)!=io.serial(result):raise RuntimeError("frozen calibration thresholds differ; keep existing evidence")
    else:io.write(path,result)
    return result


def deployment_costs(record,methods):
    """Measured component closure; fixed aliases use the full-family upper bound."""
    costs=record.get("planning",{}).get("cost",{})
    groups={
        "passive":("passive_pre_seconds","passive_seconds"),
        "history":("history_seconds",),"baseline":("baseline_completion_seconds",),
        "random":("random_seconds",),"P-SQ":("P-SQ_seconds",),
        "responses":("response_setup_seconds",),"G-SQ":("G-SQ_seconds",),"G-AQ":("G-AQ_seconds",)}
    components={name:next((float(costs[k]) for k in keys if old.number(costs.get(k)) is not None),None)
        for name,keys in groups.items()}
    total=old.number(costs.get("selection_total_seconds"));component_sum=sum(v for v in components.values() if v is not None)
    signed_residual=total-component_sum if total is not None else None
    residual=max(0.,signed_residual) if signed_residual is not None else None
    output={}
    for method in methods:
        base=method.split("/")[0];deps=["passive","history"]
        if base in ("CC","HL","P-SQ","G-SQ","G-AQ"):deps.append("baseline")
        if base=="RAND":deps.extend(["baseline","random"])
        if base in ("P-SQ","G-SQ","G-AQ"):deps.append("P-SQ")
        if base in ("G-SQ","G-AQ"):deps.extend(["responses",base])
        complete=all(components.get(k) is not None for k in deps)
        lower=sum(components[k] for k in deps) if complete else None
        upper=lower+residual if lower is not None and residual is not None else None
        _,action=_action(record,method);action=action or {}
        inference_components=dict(pre_refit=old.number((action.get("pre_refit") or {}).get("seconds")),
            passive_update=old.number((action.get("post_passive") or {}).get("seconds")),
            new_history=old.number((action.get("post_history") or {}).get("seconds")),
            new_refit=old.number((action.get("post_refit") or {}).get("seconds")))
        # No newly proposed candidate means there is no additional AC fit.
        new_count=len(_structures(record,action))-len(_structures(record)) if action else None
        if new_count==0:
            for key in ("new_history","new_refit"):
                if inference_components[key] is None:inference_components[key]=0.
        infer_complete=all(value is not None for value in inference_components.values())
        infer=sum(inference_components.values()) if infer_complete else None
        fixed_alias="/" in method
        output[method]=dict(component_names=deps,components_seconds={k:components[k] for k in deps},
            selection_component_lower_seconds=None if fixed_alias else lower,selection_component_upper_seconds=upper,
            full_family_component_seconds=lower,
            fixed_alias_full_family_upper_bound=fixed_alias,
            inference_components_seconds=inference_components,inference_complete=infer_complete,
            inference_seconds=infer,
            deployment_upper_seconds=upper+infer if upper is not None and infer is not None else None,
            deployment_lower_seconds=None if fixed_alias else lower+infer if lower is not None and infer is not None else None,
            new_candidate_count=new_count,
            scope="standalone dependency components, shared actions charged in full; fixed aliases conservatively charged all four domains of their base family; other methods excluded")
    return dict(components_seconds=components,selection_total_seconds=total,
        component_sum_seconds=component_sum,signed_unallocated_seconds=signed_residual,
        policies=output,batch_seconds=old.number(record.get("seconds")))


def condition_detail(record,methods):
    planning=record.get("planning",{});families=[]
    for method,domains in planning.get("families",{}).items():
        for domain,result in domains.items():
            families.append(dict(method=method,domain=domain,completed=result.get("completed"),
                unfinished=result.get("unfinished"),success=result.get("success"),
                score=result.get("score"),objective_lower=result.get("objective_lower"),
                objective_upper=result.get("objective_upper"),gap=result.get("gap"),
                failures=result.get("failures",[]),seconds=result.get("seconds")))
    estimates={}
    for method in methods:
        estimates[method]={route:rx_error(record,route_view(record,method,route)) for route in ROUTES}
    counters=Counter(planning.get("history",{}).get("counters",{}))
    counters["ac_calls"]+=planning.get("context",{}).get("ac_calls",0)
    for action in record.get("evaluation",{}).get("actions",{}).values():
        for key in ("pre_refit","post_history","post_refit"):
            counters.update((action.get(key) or {}).get("counters",{}))
    return dict(spec=record["spec"],status=record.get("status"),failures=record.get("failures",{}),
        native=native_topology_metrics(record),forecast_error=forecast_errors(record),
        rx=estimates,
        final_topology={m:{route:final_topology_metrics(record,m,route) for route in ROUTES} for m in methods},
        score_ranking=score_ranking(record),solver=families,
        costs=deployment_costs(record,methods),scientific_counters=dict(counters),
        plans={m:dict(action_key=p.get("action_key"),domain=p.get("domain"),
            physicalcheck=p.get("physicalcheck"),fallback_reason=p.get("fallback_reason"),
            primary=p.get("primary",m in methods)) for m,p in planning.get("plans",{}).items()},
        unique_actual_actions=len(record.get("evaluation",{}).get("actions",{})))


def phase_summary(records,thresholds,methods,*,bootstrap=2000,seed=2026101801):
    rows=decision_rows(records,thresholds,methods=methods);details=[condition_detail(r,methods) for r in records]
    tables={};by_condition={};by_network={}
    for method in methods:
        tables[method]={}
        for route in ROUTES:
            subset=[r for r in rows if r["method"]==method and r["route"]==route]
            cell=summarize_decisions(subset)
            cell["clades"]=summarize_clades([d["final_topology"][method][route] for d in details],
                bootstrap=bootstrap,seed=seed)
            cell["intervals"]=dict(
                conditional_top1_error=cluster_bootstrap(subset,lambda x:summarize_decisions(x)["conditional_top1_error_rate"],replicates=bootstrap,seed=seed),
                unique_correct_all=cluster_bootstrap(subset,lambda x:old.mean(float(r["unique_correct"]) for r in x),replicates=bootstrap,seed=seed),
                false_reject_given_inclusion=cluster_bootstrap(subset,lambda x:summarize_decisions(x)["false_reject_given_inclusion"],replicates=bootstrap,seed=seed))
            tables[method][route]=cell
            conditions=sorted({(r["network"],r["n"],r["source_mode"]) for r in subset})
            for network,n,source in conditions:
                by_condition[cell_key(method,n,source,route,network)]=summarize_decisions(
                    [r for r in subset if r["network"]==network and r["source_mode"]==source])
    for network in sorted({network_name(r["spec"]) for r in records}):
        group=[r for r in records if network_name(r["spec"])==network]
        nd=[d for d in details if network_name(d["spec"])==network]
        nt={}
        for method in methods:
            nt[method]={}
            for route in ROUTES:
                chosen=[r for r in rows if r["network"]==network and r["method"]==method and r["route"]==route]
                cell=summarize_decisions(chosen)
                cell["clades"]=summarize_clades([d["final_topology"][method][route] for d in nd],
                    bootstrap=bootstrap,seed=seed)
                nt[method][route]=cell
        by_network[network]=dict(n_condition_records=len(group),
            n_independent_date_clusters=len({r["spec"]["cluster_index"] for r in group}),
            n=sorted({int(r["spec"]["n"]) for r in group}),tables=nt)
    paired={m:paired_summary(rows,"P-SQ",m,replicates=bootstrap,seed=seed) for m in methods if m!="P-SQ"}
    same_domain={}
    for suffix in ("","/F±"):
        for base in ("G-SQ","G-AQ"):
            left,right="P-SQ"+suffix,base+suffix
            if left in methods and right in methods:
                same_domain[right]=paired_summary(rows,left,right,replicates=bootstrap,seed=seed)
    spatial={base:paired_summary(rows,base+"/F±",base,replicates=bootstrap,seed=seed)
        for base in ("P-SQ","G-SQ","G-AQ") if base in methods and base+"/F±" in methods}
    inference={};updating={}
    for method in methods:
        remap=[];updates=[]
        for r in rows:
            if r["method"]==method and r["route"] in ("preFIX","preJNT"):
                remap.append(dict(r,method=r["route"],route="inference"))
            if r["method"]==method and r["route"] in ("preJNT","postJNT"):
                updates.append(dict(r,method=r["route"],route="update"))
        inference[method]=paired_summary(remap,"preFIX","preJNT",route="inference",replicates=bootstrap,seed=seed)
        updating[method]=paired_summary(updates,"preJNT","postJNT",route="update",replicates=bootstrap,seed=seed)
        updatepairs=paired_rows(updates,"preJNT","postJNT",route="update")
        updating[method]["candidate_coverage_recovered"]=sum(not p["left"]["truth_present"] and p["right"]["truth_present"] for p in updatepairs if p["left"] and p["right"])
    native={}
    for name in ("RNJ_raw_tolerance0","WZZT_last_accepted"):
        saved=[d["native"].get("selected_native",{}).get(name) for d in details]
        available=[r for r in saved if r is not None]
        native[name]=dict(n_total_conditions=len(records),available=len(available),missing=len(records)-len(available),
            exact=sum(r["exact"] for r in available),exact_rate_all=_rate(sum(r["exact"] for r in available),len(records)),
            f1=old.distribution(r["f1"] for r in available))
    solver=[r for d in details for r in d["solver"]]
    solver_summary={}
    for method,domain in sorted({(r["method"],r["domain"]) for r in solver}):
        subset=[r for r in solver if r["method"]==method and r["domain"]==domain]
        solver_summary[method+"|"+domain]=dict(n_conditions=len(subset),
            completed=sum(r["completed"] is True for r in subset),unfinished=sum(r["unfinished"] is True for r in subset),
            raw_failure_conditions=sum(bool(r["failures"]) for r in subset),
            gap=old.distribution(r["gap"] for r in subset),seconds=old.distribution(r["seconds"] for r in subset))
    rx={m:{route:{key:old.distribution(d["rx"][m][route].get(key,{}).get("mean_absolute_relative") for d in details
            if d["rx"][m][route].get("available") and d["rx"][m][route].get("fit_success")) for key in ("r","x")}
            for route in ROUTES} for m in methods}
    rx_absolute={m:{route:{key:dict(
        all_edges_rmse_pu=old.distribution(d["rx"][m][route].get(key,{}).get("rmse_pu") for d in details
            if d["rx"][m][route].get("available") and d["rx"][m][route].get("fit_success")),
        zero_truth_edges_absolute_mean_pu=old.distribution(d["rx"][m][route].get(key,{}).get("zero_truth_edges_absolute_pu",{}).get("mean") for d in details
            if d["rx"][m][route].get("available") and d["rx"][m][route].get("fit_success")),
        condition_edge_denominators=[dict(spec=d["spec"],positive=d["rx"][m][route].get(key,{}).get("n_positive_edges"),zero=d["rx"][m][route].get(key,{}).get("n_zero_edges"))
            for d in details if d["rx"][m][route].get("available")]) for key in ("r","x")}
        for route in ROUTES} for m in methods}
    return dict(n_condition_records=len(records),n_independent_date_clusters=len({r["spec"]["cluster_index"] for r in records}),
        record_status=dict(Counter(r.get("status","missing") for r in records)),tables=tables,by_condition=by_condition,by_network=by_network,
        native=native,paired_vs_mobile_P_SQ=paired,paired_same_domain_vs_P_SQ=same_domain,
        paired_fixed_to_optional_mobile=spatial,
        paired_joint_vs_fixed=inference,paired_candidate_update=updating,solver=solver_summary,rx=rx,rx_absolute=rx_absolute,
        counters=dict(sum((Counter(d["scientific_counters"]) for d in details),Counter())),
        decision_rows=rows,condition_details=details,
        uncertainty_scope="date-cluster percentile bootstrap; chronological simulation under the declared generator, not a topology posterior or a guarantee of nominal coverage")


def _load_phase(output,phase,config,audit):
    records={}
    for path in sorted((output/phase).glob("*.json.gz")):
        value=io.read(path);spec=value["spec"]
        key=(int(spec["cluster_index"]),network_name(spec),spec["source_mode"])
        if key in records:raise ValueError("duplicate saved condition "+repr(key))
        if spec["phase"]!=phase:raise ValueError("saved record phase differs from directory")
        records[key]=value
    declared=condition_manifest(config,phase)
    expected={(r["cluster_index"],r["network"],r["source_mode"]) for r in declared}
    lookup={(r["cluster_index"],r["network"],r["source_mode"]):r for r in declared}
    for key,value in records.items():
        if key in lookup:
            audit.check("condition_measurement_count",int(value["spec"]["n"])==lookup[key]["n"],
                dict(phase=phase,condition=key,declared_n=lookup[key]["n"],stored_n=value["spec"]["n"]))
    audit.check("condition_manifest",set(records)==expected,dict(phase=phase,missing=sorted(expected-set(records)),unexpected=sorted(set(records)-expected)))
    for key in sorted(expected-set(records)):
        records[key]=dict(spec=dict(lookup[key]),
            status="missing",failures=dict(record="condition file missing"))
    return [records[key] for key in sorted(records)]


def small_spatial_ablation(records,thresholds,*,bootstrap=2000,seed=2026101801):
    """Four optimized domains: proxy comparisons and actually executed plans.

    Charge-only proxy solutions are not turned into fictitious AC outcomes.
    The main batch executes fixed/mobile bidirectional plans; all conditions
    participate, with no legacy four-date or n=6 filter.
    """
    records=list(records);proxy={};actual={}
    edges=(("fixed_charge","fixed_bidir"),("fixed_charge","mobile_charge"),
           ("fixed_bidir","mobile_bidir"),("mobile_charge","mobile_bidir"))
    for base in ("P-SQ","G-SQ","G-AQ"):
        comparisons={}
        for left,right in edges:
            pairs=[]
            for record in records:
                family=record.get("planning",{}).get("families",{}).get(base,{})
                l=old.number(family.get(left,{}).get("score"));r=old.number(family.get(right,{}).get("score"))
                pairs.append(dict(spec=record["spec"],available=l is not None and r is not None,
                    left=l,right=r,delta=r-l if l is not None and r is not None else None))
            valid=[p for p in pairs if p["available"]]
            comparisons[left+"__to__"+right]=dict(n_all_conditions=len(pairs),n_paired_scores=len(valid),
                unavailable=len(pairs)-len(valid),delta=old.distribution(p["delta"] for p in valid),
                decrease=sum(p["delta"] < -1e-8*(1+abs(p["left"])) for p in valid),
                by_network={name:old.distribution(p["delta"] for p in valid if network_name(p["spec"])==name)
                    for name in sorted({network_name(r["spec"]) for r in records})},
                rows=pairs,scope="same frozen planning objective only; not observed AC discrimination")
        proxy[base]=comparisons
        methods=(base+"/F±",base)
        if any(_action(r,methods[0])[1] is not None or _action(r,methods[1])[1] is not None for r in records):
            rows=decision_rows(records,thresholds,methods=methods)
            actual[base]=paired_summary(rows,*methods,route="postJNT",replicates=bootstrap,seed=seed)
    charge_only=("P-SQ/F+","P-SQ/M+","G-SQ/F+","G-SQ/M+","G-AQ/F+","G-AQ/M+")
    executed={m:sum(_action(r,m)[1] is not None for r in records) for m in charge_only}
    return dict(n_condition_records=len(records),n_date_clusters=len({r["spec"]["cluster_index"] for r in records}),
        optimized_domains=["fixed_charge","fixed_bidir","mobile_charge","mobile_bidir"],
        proxy_parent_child_comparisons=proxy,actual_fixed_bidir_to_optional_mobile=actual,
        charge_only_actual_record_counts=executed,
        charge_only_actual_comparison_available=all(executed.values()) if records else False,
        scope="all declared networks/source conditions; four proxy domains, separate actual bidirectional comparison; unexecuted plans are not extra failures or AC results")


def report_text(summary,audit):
    lines=["# 原生参考网络WZZT—AC—EV测试", "", "统计单位是日期簇，同一时间窗的各参考网络及两种源条件配对。主表包含候选漏真和计算失败；条件误判率只使用真结构在池中且该路线可用的条件。", "",
        "|阶段|策略|条件数/日期簇|历史池漏真|preJNT错/可用含真|postJNT错/可用含真|post唯一正确|post未决|post不可用|", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for phase,result in summary["phases"].items():
        for method in summary["methods"]:
            pre=result["tables"][method]["preJNT"];post=result["tables"][method]["postJNT"]
            lines.append("|%s|%s|%d/%d|%d|%d/%d|%d/%d|%d|%d|%d|"%(phase,method,result["n_condition_records"],result["n_independent_date_clusters"],
                pre["candidate_missing"],pre["conditional_top1_wrong"],pre["conditional_top1_evaluable"],post["conditional_top1_wrong"],post["conditional_top1_evaluable"],
                post["unique_correct"],post["unresolved"],post["fit_failure"]))
    lines.extend(["", "各网络分别列出postJNT结果。F1只在点标签和必要计算可用时评分，未评分数量保留：", "",
        "|阶段|网络|策略|条件数|严格精确/全部|F1均值|F1评分/未评分|收缩诊断精确/全部|唯一正确|唯一错误|未决|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for phase,result in summary["phases"].items():
        for network,group in result.get("by_network",{}).items():
            for method in summary["methods"]:
                cell=group["tables"][method]["postJNT"];c=cell["clades"]
                mean=c["f1"].get("mean")
                lines.append("|%s|%s|%s|%d|%d/%d|%s|%d/%d|%d/%d|%d|%d|%d|"%(
                    phase,network,method,cell["n_total_conditions"],c["exact"],c["n_total_conditions"],
                    "—" if mean is None else "%.4f"%mean,c["scored"],c["unscored"],
                    c["contracted_diagnostic"]["exact"],c["n_total_conditions"],
                    cell["unique_correct"],cell["unique_wrong"],cell["unresolved"]))
    lines.extend(["", "preFIX固定历史R/X；preJNT让原候选共享历史与新增计量重新拟合；postJNT再并入观测驱动的新候选。联合拟合收益、候选更新收益与EV规划收益分别报告。F±为固定地点双向，未加后缀的MILP为允许移动的双向域。", "",
        "集合判定阈值只由独立校准期的半SSE差生成。可用校准不足时阈值为无穷并保留全池；这些未决保留在分母。时间顺序的数据可能存在分布变化，名义alpha不自动保证评价期误拒绝率。", "",
        "实际判定按策略、物理网络、末端数与拟合路线查统一阈值，取预定源条件各自校准阈值的最大值，不读取实际源类别。源分层阈值仅作诊断；同日期的不同网络及源条件仍只计一个日期簇。", "",
        "零边诊断保留原严格终点。额外报告结构细化覆盖，并按事前固定标幺阻抗幅值阈值收缩既定点标签的近零非平凡边，再算F1；不据真值换标签、调阈值或改保留集合。R/X相对误差只对正真值边计算，所有边绝对误差和零边估计另报。", "",
        "四嵌套域代理值和父计划继承另列；只有实际执行并生成量测的动作才比较完整AC辨识。未执行的纯充电域不补成虚构测量结果。", "",
        "审计：%s；保存状态的完整AC前向回放次数%d，重新拟合0次，重新规划0次。"%("通过" if audit["passed"] else "发现问题",audit["forward_AC_replay_calls"]),
        "详细原生拓扑、clade F1、R/X、求解界、费用、预测误差、实际收益排序和日期簇配对区间见summary.json。候选评分、似然差与bootstrap频率均不表示拓扑正确概率。", ""])
    return "\n".join(lines)


def finish(output,*,bootstrap=None):
    output=Path(output);frozen=io.read(output/"freeze.json");config=frozen["config"]
    methods=tuple(config.get("methods",METHODS));audit=old.Audit()
    payload={k:v for k,v in frozen.items() if k!="identity"}
    identity=sha256(json.dumps(io.serial(payload),sort_keys=True).encode()).hexdigest()
    audit.check("freeze_identity",identity==frozen.get("identity"),dict(computed=identity,stored=frozen.get("identity")))
    for name,digest in frozen.get("source_sha256",{}).items():
        path=io.ROOT/name;audit.check("source_hash",path.exists() and io.digest(path)==digest,dict(path=name))
    audit.check("selector_truth_flag",frozen.get("selector_reads_truth") is False,"frozen selector boundary")
    count=len(condition_manifest(config,"calibration"))
    # Crucial sequencing: read only calibration, persist its immutable rule,
    # and only then load evaluation records. No evaluation input tunes a rule.
    thresholds=freeze_thresholds(output,alpha=config.get("alpha",ALPHA),methods=methods,expected_conditions=count)
    phases={};B=int(config.get("statistical_bootstrap",2000) if bootstrap is None else bootstrap)
    for phase in ("development","calibration","evaluation"):
        records=_load_phase(output,phase,config,audit)
        for record in records:
            try:audit_record(record,audit)
            except Exception as exc:audit.check("record_audit_exception",False,dict(spec=record["spec"],exception=repr(exc)))
        phases[phase]=phase_summary(records,thresholds,methods,bootstrap=B,seed=int(config.get("statistical_seed",2026101801)))
        if phase=="evaluation":
            spatial_ablation=small_spatial_ablation(records,thresholds,bootstrap=B,
                seed=int(config.get("statistical_seed",2026101801)))
    value=dict(schema="reference_benchmark_summary_v1",policy_identity=frozen.get("identity"),
        analysis_sha256=io.digest(Path(__file__)),threshold_identity=thresholds["identity"],methods=methods,
        phases=phases,spatial_ablation=spatial_ablation,
        expected_conditions_by_phase={p:len(condition_manifest(config,p)) for p in phases},
        declared_network_sizes=declared_network_sizes(config),
        statistical_seed=int(config.get("statistical_seed",2026101801)),statistical_bootstrap=B,
        limits=["finite, truncated candidate pool; candidate miss is reported",
            "conditional output likelihood holds measured P/Q fixed; input meter noise is not propagated",
            "one historical line fit per candidate; point forecasts; no calibrated response covariance",
            "attained local AC likelihood, not a certified global optimum",
            "published reference network parameters, six native LV aggregate P/Q profiles with explicit reuse, declared source and synthetic meter noise",
            "paper11/paper15 are project paper-style compatibility cases with an explicit transformer extension, not author-original case replication",
            "shared six-profile basis has historical centered P/Q rank at most 12; larger networks are correlated low-excitation benchmarks, not independent heterogeneous customer samples",
            "strict topology and fitted near-zero contraction diagnostics are separate; neither changes primary calibrated decisions",
            "calibration coverage is conditional on inclusion and required computations, under the declared simulation distribution"])
    audited=audit.record();io.write(output/"audit.json",audited);io.write(output/"summary.json",value)
    (output/"results.md").write_text(report_text(value,audited),encoding="utf-8")
    return value,audited


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("output",nargs="?")
    parser.add_argument("--output",dest="output_option");parser.add_argument("--bootstrap",type=int)
    args=parser.parse_args();output=args.output_option or args.output
    if not output:parser.error("output directory required")
    summary,audit=finish(output,bootstrap=args.bootstrap)
    print(json.dumps(dict(policy_identity=summary["policy_identity"],audit_passed=audit["passed"],
        audit_issues=len(audit["issues"]),evaluation_conditions=summary["phases"]["evaluation"]["n_condition_records"])))


if __name__=="__main__":main()


__all__=["METHODS","ROUTES","signature","clade_metrics","threshold_quantile",
    "calibration_thresholds","route_view","decide","decision_rows","summarize_decisions",
    "cluster_bootstrap","paired_rows","paired_summary","freeze_thresholds","finish",
    "network_name","declared_network_sizes","phase_count","condition_manifest",
    "rx_error","cone_contains","final_topology_metrics","summarize_clades",
    "contracted_supports","zero_equivalence_diagnostic","small_spatial_ablation"]
