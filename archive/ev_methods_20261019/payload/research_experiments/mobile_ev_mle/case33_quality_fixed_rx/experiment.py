"""Known-line case33: paired input-noise and power-flow score diagnostic.

Run as a package module, or as a script with the Topo source root on PYTHONPATH.
Only the generator sees true P/Q. Fitters receive known lines and noisy meters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import sys
import zipfile

import numpy as np
import scipy
from scipy.optimize import least_squares

from research_experiments.mobile_ev_mle import likelihood as lk
from research_experiments.mobile_ev_mle.ev_fixed_response import FixedResponse
from research_experiments.mobile_ev_mle.reference_benchmark.networks import build_case33
from research_experiments.mobile_ev_mle.reference_benchmark.representation import (
    encode_physical_tree, physical_tree_ac,
)


CONFIG = dict(
    version="case33_known_lines_quality_factorial_v1",
    states=[dict(name="evening", p_load=1., q_load=1., pv_factor=0.),
            dict(name="day_pv", p_load=.65, q_load=.70, pv_factor=.80),
            dict(name="reverse_pv", p_load=.40, q_load=.50, pv_factor=1.)],
    pv_capacity_fraction=.60, pv_buses=[18,25,33], pv_weights=[.35,.30,.35],
    power_sigmas=[.002,.01], voltage_sigma=.0005, root_sigma_fraction=.001,
    seeds=list(range(2026101801,2026101809)),
    max_nfev=400, tolerance=1e-12, reference_gradient_tolerance=1e-6,
    reference_score_tolerance=1e-7,
    source_P_Q_fixed=[0.,0.], root_v=1., fitted_line_parameters=False,
    backend_names=["linear","i2","ac"], modes=["FIX","EIV"],
)


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [plain(v) for v in value]
    return value


def save(path, value):
    Path(path).write_text(json.dumps(plain(value), ensure_ascii=False, indent=2,
                                   allow_nan=False)+"\n", encoding="utf-8")


def make_context():
    net=build_case33(); ids=list(net["observation_bus_ids"])
    by_bus={b["bus_id"]:b for b in net["buses"]}
    p0=np.array([by_bus[b]["pd_kw"] for b in ids])
    q0=np.array([by_bus[b]["qd_kvar"] for b in ids])
    encoded=encode_physical_tree(net["branches"],ids,net["root_bus"],
        net["base_kv"],net["base_kw"],source_bus=2,name="case33_known_lines")
    return dict(network=net, ids=ids, p0=p0, q0=q0, encoded=encoded,
        backends={m:FixedResponse(encoded.candidate,encoded.r,encoded.x,method=m)
                  for m in CONFIG["backend_names"]})


def response(ctx, backend, z, derivatives=False):
    n=len(ctx["ids"])
    out=ctx["backends"][backend].evaluate(np.asarray(z[:n])[None,:],
        np.asarray(z[n:])[None,:],np.zeros(1),np.zeros(1),np.ones(1),
        derivatives=derivatives)
    jac=out["jac"][0,:,:2*n] if derivatives else None
    return out["values"][0],jac


def independent_output(ctx,z):
    net=ctx["network"]; n=len(ctx["ids"]); base=net["base_kw"]
    out=physical_tree_ac(net["branches"],ctx["ids"],net["root_bus"],
        net["base_kv"],base,np.asarray(z[:n])*base,np.asarray(z[n:])*base,
        source_bus=2,root_v=1.)
    y=np.r_[out["v"][0],out["master_p"][0],out["master_q"][0]]
    return y,dict(ac_equation_residual=out["ac_equation_residual"],
        power_balance_residual=out["power_balance_residual"])


def generate_state(ctx, state):
    base=ctx["network"]["base_kw"]; p=state["p_load"]*ctx["p0"]
    q=state["q_load"]*ctx["q0"]; pv=np.zeros_like(p)
    capacity=CONFIG["pv_capacity_fraction"]*ctx["p0"].sum()
    for bus,weight in zip(CONFIG["pv_buses"],CONFIG["pv_weights"]):
        pv[ctx["ids"].index(bus)]=capacity*weight*state["pv_factor"]
    z=np.r_[p-pv,q]/base
    y,physics=independent_output(ctx,z)
    direct,_=response(ctx,"ac",z)
    err=float(np.max(abs(y-direct)))
    if err>1e-10: raise AssertionError("generator and encoded AC disagree")
    return dict(name=state["name"],state=state,z_true=z,y_true=y,
        p_load_kw=p,q_load_kvar=q,pv_kw=pv,pv_installed_kw=float(capacity),
        net_p_kw=float((p-pv).sum()),net_q_kvar=float(q.sum()),
        root_p_kw=float(y[-2]*base),root_q_kvar=float(y[-1]*base),
        v_min=float(y[:-2].min()),v_max=float(y[:-2].max()),
        generator_AC_agreement=err,physics=physics)


def sigmas(ctx,power_sigma):
    base=ctx["network"]["base_kw"]; p0=ctx["p0"];q0=ctx["q0"]
    sz=np.r_[power_sigma*np.maximum(p0,1.),
        power_sigma*np.maximum.reduce([abs(q0),.2*p0,np.ones_like(p0)])]/base
    sy=np.r_[np.full(len(p0),CONFIG["voltage_sigma"]),
        CONFIG["root_sigma_fraction"]*max(float(p0.sum()),20.)/base,
        CONFIG["root_sigma_fraction"]*max(float(abs(q0).sum()),.2*float(p0.sum()),20.)/base]
    return sz,sy


def score(ctx,backend,z,zobs,yobs,sz,sy):
    yp,_=response(ctx,backend,z)
    rpq=(z-zobs)/sz; ry=(yp-yobs)/sy
    return .5*float(rpq@rpq+ry@ry),dict(
        pq=.5*float(rpq@rpq),v=.5*float(ry[:-2]@ry[:-2]),
        root_p=.5*float(ry[-2]**2),root_q=.5*float(ry[-1]**2))


def solve_start(ctx,backend,zobs,yobs,sz,sy,start,algorithm,label):
    began=perf_counter(); cached={}; calls=dict(value=0,jacobian=0)
    def value(w):
        if "w" not in cached or not np.array_equal(cached["w"],w):
            yp,j= response(ctx,backend,zobs+sz*w,True)
            cached.clear(); cached.update(w=w.copy(), yp=yp, j=j)
            calls["value"]+=1;calls["jacobian"]+=1
        return cached["yp"],cached["j"]
    def fun(w):
        yp,_=value(w)
        return np.r_[w,(yp-yobs)/sy]
    def jac(w):
        _,j=value(w)
        return np.vstack([np.eye(len(w)),j*sz[None,:]/sy[:,None]])
    try:
        fit=least_squares(fun,np.asarray(start,float),jac=jac,method=algorithm,
            ftol=CONFIG["tolerance"],xtol=CONFIG["tolerance"],gtol=CONFIG["tolerance"],
            max_nfev=CONFIG["max_nfev"])
        gradient=float(np.max(abs(fit.jac.T@fit.fun)))
        return dict(label=label,algorithm=algorithm,w=fit.x,z=zobs+sz*fit.x,
            score=.5*float(fit.fun@fit.fun),gradient_inf=gradient,
            scaled_gradient=gradient/(1+np.linalg.norm(fit.fun)),
            status=int(fit.status),success=bool(fit.success),message=fit.message,
            nfev=int(fit.nfev),njev=None if fit.njev is None else int(fit.njev),
            seconds=perf_counter()-began,calls=calls,failure=None)
    except Exception as exc:
        return dict(label=label,algorithm=algorithm,success=False,
            seconds=perf_counter()-began,calls=calls,failure=repr(exc))


def fit_method(ctx,backend,mode,zobs,yobs,sz,sy):
    if mode=="FIX":
        began=perf_counter(); s,parts=score(ctx,backend,zobs,zobs,yobs,sz,sy)
        return dict(backend=backend,mode=mode,z=zobs.copy(),score=s,parts=parts,
            starts=[],primary_seconds=perf_counter()-began,audit_seconds=0.,
            success=True,gradient_inf=None,unknown_states=0)
    starts=[]
    for label,algorithm,w in [
        ("primary_observed","lm",np.zeros_like(zobs)),
        ("audit_other_algorithm","trf",np.zeros_like(zobs)),
        ("audit_perturbed","lm",.25*np.sin(np.arange(len(zobs))+1.))]:
        starts.append(solve_start(ctx,backend,zobs,yobs,sz,sy,w,algorithm,label))
    valid=[s for s in starts if s.get("failure") is None]
    if not valid: return dict(backend=backend,mode=mode,success=False,starts=starts)
    best=min(valid,key=lambda s:s["score"])
    s,parts=score(ctx,backend,best["z"],zobs,yobs,sz,sy)
    return dict(backend=backend,mode=mode,z=best["z"],score=s,parts=parts,
        starts=starts,selected_start=best["label"],primary_seconds=starts[0]["seconds"],
        audit_seconds=sum(s["seconds"] for s in starts[1:]),
        success=bool(best["success"]),gradient_inf=best["gradient_inf"],
        unknown_states=len(zobs),start_score_spread=float(max(v["score"] for v in valid)-min(v["score"] for v in valid)))


def run_sample(ctx,generated,state_index,power_sigma,seed,*,clean=False):
    ztrue=generated["z_true"]; ytrue=generated["y_true"]
    sz,sy=sigmas(ctx,power_sigma)
    rng=np.random.default_rng(np.random.SeedSequence([seed,state_index]))
    ez=rng.standard_normal(len(ztrue)); ey=rng.standard_normal(len(ytrue))
    if clean: ez*=0.;ey*=0.
    zobs=ztrue+sz*ez; yobs=ytrue+sy*ey
    result=dict(state=generated["name"],power_sigma=power_sigma,seed=seed,clean=clean,
        zobs=zobs,yobs=yobs,sigma_z=sz,sigma_y=sy,standardized_noise_z=ez,
        standardized_noise_y=ey,methods={},reference_audits=[],failures=[])
    for backend in CONFIG["backend_names"]:
        for mode in CONFIG["modes"]:
            key=backend+"/"+mode
            try:
                result["methods"][key]=fit_method(ctx,backend,mode,zobs,yobs,sz,sy)
            except Exception as exc:
                result["methods"][key]=dict(backend=backend,mode=mode,success=False,failure=repr(exc))
            if not result["methods"][key]["success"]:result["failures"].append(key)
    ref=result["methods"]["ac/EIV"]
    if not ref["success"]: return result
    for backend in ("linear","i2"):
        item=result["methods"][backend+"/EIV"]
        if item["success"]:
            extra=solve_start(ctx,"ac",zobs,yobs,sz,sy,(item["z"]-zobs)/sz,
                "lm","audit_AC_from_"+backend)
            result["reference_audits"].append(extra)
            if extra.get("failure") is None and extra["score"]<ref["score"]:
                ref.update(z=extra["z"],score=extra["score"],gradient_inf=extra["gradient_inf"],
                    selected_start=extra["label"])
    all_ac=ref["starts"]+result["reference_audits"]
    valid=[s for s in all_ac if s.get("failure") is None]
    agreement=max(s["score"] for s in valid)-min(s["score"] for s in valid)
    result["reference_qualified"]=bool(ref["gradient_inf"]<=CONFIG["reference_gradient_tolerance"]
        and agreement<=CONFIG["reference_score_tolerance"]*(1+ref["score"]))
    result["reference_start_score_spread"]=agreement
    result["oracle_true_state_noise_score"],_=score(ctx,"ac",ztrue,zobs,yobs,sz,sy)
    for key,item in result["methods"].items():
        if not item["success"]: continue
        z=item["z"]; self_score,parts=score(ctx,item["backend"],z,zobs,yobs,sz,sy)
        ac_back,ac_parts=score(ctx,"ac",z,zobs,yobs,sz,sy)
        exact_y,physical=independent_output(ctx,z)
        encoded_y,_=response(ctx,"ac",z)
        refscore=result["methods"]["ac/"+item["mode"]]["score"]
        model_y,_=response(ctx,item["backend"],z)
        n=len(ctx["ids"]); base=ctx["network"]["base_kw"]
        item.update(score=self_score,parts=parts,AC_back_score=ac_back,AC_back_parts=ac_parts,
            reference_score=refscore,signed_score_error=self_score-refscore,
            absolute_score_error=abs(self_score-refscore),
            relative_score_error=abs(self_score-refscore)/max(refscore,1.),
            model_score_bias=self_score-ac_back,state_fitting_loss=ac_back-refscore,
            AC_back_regret_relative=(ac_back-refscore)/max(refscore,1.),
            pq_recovery_whitened_rmse=float(np.sqrt(np.mean(((z-ztrue)/sz)**2))),
            p_recovery_rmse_kw=float(np.sqrt(np.mean((z[:n]-ztrue[:n])**2))*base),
            q_recovery_rmse_kvar=float(np.sqrt(np.mean((z[n:]-ztrue[n:])**2))*base),
            model_vs_AC_v_max_pu=float(np.max(abs(model_y[:-2]-encoded_y[:-2]))),
            model_vs_AC_root_p_kw=float((model_y[-2]-encoded_y[-2])*base),
            model_vs_AC_root_q_kvar=float((model_y[-1]-encoded_y[-1])*base),
            independent_AC_max_error=float(np.max(abs(exact_y-encoded_y))),
            independent_physics=physical,
            full_EIV_reference_excess=ac_back-ref["score"])
        identity=(item["model_score_bias"]+item["state_fitting_loss"])-item["signed_score_error"]
        if abs(identity)>1e-8*(1+abs(self_score)):raise AssertionError("score decomposition")
        if item["independent_AC_max_error"]>1e-9:raise AssertionError("independent AC replay")
    for backend in CONFIG["backend_names"]:
        corrected=result["methods"][backend+"/EIV"];fixed=result["methods"][backend+"/FIX"]
        if corrected["success"] and fixed["success"]:
            if corrected["score"]>fixed["score"]+1e-7*(1+fixed["score"]):
                raise AssertionError("correction lost its zero-correction feasible point")
    return result


def derivative_checks(ctx,states):
    records=[]; rng=np.random.default_rng(55019)
    for state in states:
        sz,_=sigmas(ctx,.002);z=state["z_true"]
        for backend in CONFIG["backend_names"]:
            _,j=response(ctx,backend,z,True);errors=[]
            for _ in range(6):
                d=sz*rng.normal(size=len(z));eps=1e-3
                plus,_=response(ctx,backend,z+eps*d);minus,_=response(ctx,backend,z-eps*d)
                finite=(plus-minus)/(2*eps);analytic=j@d
                errors.append(float(np.max(abs(finite-analytic))/max(np.max(abs(analytic)),1e-12)))
            maximum=max(errors)
            if maximum>2e-5:raise AssertionError(("directional derivative",backend,maximum))
            records.append(dict(state=state["name"],backend=backend,directions=6,
                max_relative_error=maximum))
    return records


def summarize(samples):
    groups={}
    for result in samples:
        if result["clean"]:continue
        group=result["state"]+"/"+str(result["power_sigma"])
        groups.setdefault(group,{})
        for method,item in result["methods"].items():groups[group].setdefault(method,[]).append(item)
    report={}
    fields=["score","AC_back_score","reference_score","absolute_score_error",
        "relative_score_error","model_score_bias","state_fitting_loss","AC_back_regret_relative",
        "full_EIV_reference_excess","pq_recovery_whitened_rmse","p_recovery_rmse_kw",
        "q_recovery_rmse_kvar","primary_seconds","audit_seconds","model_vs_AC_v_max_pu",
        "model_vs_AC_root_p_kw","model_vs_AC_root_q_kvar"]
    for group,methods in groups.items():
        report[group]={}
        for method,items in methods.items():
            good=[i for i in items if i["success"]]
            row=dict(total=len(items),success=len(good),failures=len(items)-len(good))
            for f in fields:
                if good:
                    values=np.array([i[f] for i in good])
                    row[f]=dict(mean=float(values.mean()),median=float(np.median(values)),
                                min=float(values.min()),max=float(values.max()))
            report[group][method]=row
    return dict(groups=report,sample_count=len(samples),
        noisy_samples=sum(not s["clean"] for s in samples),
        clean_samples=sum(s["clean"] for s in samples),
        method_slots=sum(len(s["methods"]) for s in samples),
        failed_method_slots=sum(len(s["failures"]) for s in samples),
        reference_unqualified=sum(not s.get("reference_qualified",False) for s in samples),
        all_references_qualified=all(s.get("reference_qualified",False) for s in samples),
        global_MLE_certified=False)


def source_manifest(output):
    root=Path(lk.__file__).resolve().parents[2]
    files={Path(__file__).resolve(), root/"terminal_case33/data/case33bw_raw.py"}
    files.update(p for p in Path(__file__).resolve().parent.iterdir()
                 if p.is_file() and p.suffix in (".py",".md",".ini"))
    for module in list(sys.modules.values()):
        value=getattr(module,"__file__",None)
        if value:
            p=Path(value).resolve()
            if p.suffix==".py" and (root in p.parents) and "site-packages" not in str(p):files.add(p)
    records=[]
    with zipfile.ZipFile(output/"source_snapshot.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files,key=str):
            relative=p.relative_to(root).as_posix() if root in p.parents else "case33_quality_fixed_rx/"+p.name
            data=p.read_bytes();archive.writestr(relative,data)
            records.append(dict(path=str(p),archive_path=relative,bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest()))
    save(output/"source_manifest.json",dict(files=records,python=sys.version,
        executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,
        snapshot_sha256=hashlib.sha256((output/"source_snapshot.zip").read_bytes()).hexdigest()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seeds",type=int,default=8)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if not 1<=args.seeds<=len(CONFIG["seeds"]):parser.error("seeds must be between 1 and 8")
    ctx=make_context();states=[generate_state(ctx,s) for s in CONFIG["states"]]
    save(args.output/"configuration.json",CONFIG)
    save(args.output/"execution.json",dict(actual_seeds=CONFIG["seeds"][:args.seeds],
        scope="known true topology/RX; three single-slot states; no unknown source",
        truth_used_in_fitting="only deliberately supplied topology and line impedances",
        PQ_truth_used_for_initialization=False,global_minimum_certified=False))
    save(args.output/"case33.json",dict(network=ctx["network"],encoded=ctx["encoded"].record()))
    save(args.output/"states.json",states)
    checks=derivative_checks(ctx,states);save(args.output/"derivative_checks.json",checks)
    samples=[];began=perf_counter()
    for i,state in enumerate(states):
        for noise in CONFIG["power_sigmas"]:
            for seed in CONFIG["seeds"][:args.seeds]:
                result=run_sample(ctx,state,i,noise,seed);samples.append(result)
                save(args.output/(state["name"]+"_"+str(noise)+"_"+str(seed)+".json"),result)
            print(json.dumps(dict(state=state["name"],sigma=noise,completed=len(samples),
                seconds=round(perf_counter()-began,3))),flush=True)
    for i,state in enumerate(states):
        result=run_sample(ctx,state,i,.002,CONFIG["seeds"][0],clean=True);samples.append(result)
        save(args.output/(state["name"]+"_clean.json"),result)
    summary=summarize(samples);summary["seconds_with_audits"]=perf_counter()-began
    save(args.output/"summary.json",summary);source_manifest(args.output)
    print(json.dumps({k:v for k,v in summary.items() if k!="groups"}),flush=True)


if __name__=="__main__":main()
