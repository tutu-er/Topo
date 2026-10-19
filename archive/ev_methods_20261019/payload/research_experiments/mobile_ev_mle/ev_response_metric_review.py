"""Independent saved-data review of the strict two-model contrast experiment.

Only the separate branch-current/voltage sweep reconstructs physical values.
No fit, selector, optimizer or production AC predictor is called. Legacy
four-model/name conventions are adapted explicitly in a private review view.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from . import ev_dynamic_review as dynamic
from . import ev_gradient_flow_review as profile_review
from . import ev_fixed_train_design as fixed
from .ev_original_review import forward_physics

HERE = Path(__file__).resolve().parent
NAMES = ("star", "clade_01")
METHODS = ("none", "constant", "highlow", "square", "nearest", "gradac",
           "contrast_joint", "contrast_box", "contrast_load")
V2G = frozenset(METHODS[3:])
SUPPORTS = {"star":((0,1,2),(0,),(1,),(2,)),
    "clade_01":((0,1,2),(0,1),(0,),(1,),(2,))}


def require(value, message):
    if not value: raise ValueError(message)


def close(actual, expected, label, *, atol=2e-7, rtol=5e-8):
    return dynamic._close(actual, expected, label, atol=atol, rtol=rtol)


def line_id(r, x):
    h = hashlib.sha256()
    for values in (r, x):
        a = np.ascontiguousarray(values, dtype="<f8")
        h.update(repr(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def strict_pair_schema(trial):
    bank = trial.get("candidates", [])
    require([r["name"] for r in bank] == list(NAMES), "strict predeclared candidate order")
    for row in bank:
        require(tuple(map(tuple, row["supports"])) == SUPPORTS[row["name"]], "strict declared candidate supports")
    cfg, selection = trial["config"], trial["selection"]
    require(cfg.get("declared_candidates") == list(NAMES) and cfg.get("wzzt_used") is False,
        "candidate declaration must precede simulation; WZZT disabled")
    require(selection.get("declared_candidates") == list(NAMES) and selection.get("WZZT_used") is False,
        "selector pair declaration differs")
    require(cfg.get("source_interface") == 1 and cfg.get("PQ_inputs_fixed") is True,
        "same physical source interface and fixed measured PQ required")
    fits = {r.get("candidate", r.get("name")):r for r in trial["training_result"]["fits"]}
    require(len(trial["training_result"]["fits"])==2 and set(fits) == set(NAMES), "both training candidates must remain exactly once")
    identities = {name:line_id(fits[name]["r"], fits[name]["x"]) for name in NAMES}
    checked = 0
    for record in selection.get("methods", {}).values():
        for model in record.get("scenario_models", [record.get("model", {})]):
            if not model: continue
            require(model.get("names") == list(NAMES), "contrast model pair differs")
            require(model.get("line_sha256") == identities, "contrast theta differs from training snapshot")
            checked += 1
    methods = [r["name"] for r in trial.get("methods", [])]
    require(len(methods) == len(set(methods)) == len(METHODS) and set(methods) == set(METHODS),
        "all nine methods, including failures, must remain in the denominator")
    require(trial.get("true_name") in NAMES and trial.get("candidate_miss") is False,
        "true candidate coverage metadata differs")
    return dict(candidate_count=2, supports_verified=True, line_snapshots_checked=checked,
        method_denominator=len(methods), failed_methods=sum(r.get("status") != "complete" for r in trial["methods"]))


def adapted_reviews(trial, freeze, snapshot):
    source = dynamic._source_audit(freeze, snapshot)
    require(source["source_count"] == 101 and source["current_source_matches"], "all 101 frozen sources must match current bytes")
    require(trial.get("source_sha256") == freeze["source_sha256"] and trial.get("config") == freeze["config"]
        and trial.get("policy_id") == freeze["policy_id"], "trial/freeze identity differs")
    schema = strict_pair_schema(trial)
    view = deepcopy(trial)
    require("configuration" not in view["selection"] or view["selection"]["configuration"] == fixed.CONFIG,
        "saved device configuration differs from verified frozen CONFIG")
    view["selection"]["configuration"] = deepcopy(fixed.CONFIG)
    view["review_adapter"] = dict(configuration_from="verified frozen ev_fixed_train_design.CONFIG",
        original_raw_unchanged=True, strict_two_candidates=True)
    base = dynamic.review_trial(view, freeze=freeze, snapshot=snapshot)
    superseded = []
    rows = {r["name"]:r for r in view["methods"]}
    for check in base["checks"]:
        name = check["name"]
        method = name.removeprefix("operation/") if name.startswith("operation/") else None
        if (method in V2G and check["status"] == "failed"
                and "discharge in charge-only comparison" in check.get("message", "")):
            superseded.append(deepcopy(check))
            result = dynamic._operation(dict(rows[method], name=method+"_v2g_fixed"), fixed.CONFIG)
            check.clear(); check.update(name=name, status="passed", result=result,
                adapter="predeclared fixed V2G method; legacy name convention only")
    base["legacy_superseded_checks"] = superseded
    base["success"] = bool(base["checks"]) and all(c["status"] == "passed" for c in base["checks"])
    base["check_counts"] = dict(Counter(c["status"] for c in base["checks"]))
    profiles = profile_review.review_profile_trial(view)
    replaced = [r for r in profiles["checks"] if r["name"] == "planning_schema"]
    require(len(replaced) == 1, "legacy profile schema check is absent or duplicated")
    errors = profiles["errors"]
    legacy = [e for e in errors if e["name"] == "planning_schema"]
    require(not legacy or all("four-model bank" in e["message"] for e in legacy),
        "unexpected legacy schema failure cannot be suppressed")
    profiles["legacy_superseded_checks"] = deepcopy(replaced)
    replaced[0].clear(); replaced[0].update(name="planning_schema", status="passed", result=schema,
        adapter="independently checked predeclared two-candidate schema")
    profiles["errors"] = [e for e in errors if e["name"] != "planning_schema"]
    profiles["success"] = bool(profiles["counts"]["unique_profiles"]) and not profiles["errors"]
    return view, dict(source_audit=source, strict_pair_schema=schema, base_review=base, profile_review=profiles,
        success=base["success"] and profiles["success"], raw_records_modified=False,
        configuration_adapter=view["review_adapter"])


class Replay:
    """Physical response/central derivative cache, independent of production AC."""
    def __init__(self, trial, step=1e-3):
        self.trial, self.step = trial, step
        self.forecast = trial["forecast"]; self.root = np.asarray(self.forecast["root"], float)
        self.T, self.n = np.shape(self.forecast["p"])
        self.bank = {r["name"]:dynamic._candidate(r["name"], r["supports"], self.n) for r in trial["candidates"]}
        self.lines = {r.get("candidate", r.get("name")):r for r in trial["training_result"]["fits"]}
        meter = dynamic._meter(trial["future_meter"])
        self.sigma = np.column_stack((np.broadcast_to(meter.v_sigma, (self.T,self.n)),
            np.broadcast_to(meter.master_p_sigma, (self.T,)), np.broadcast_to(meter.master_q_sigma, (self.T,))))
        self.sources = {name:np.column_stack((s["source_p"], s["source_q"]))
            for name,s in self.forecast["source_by_candidate"].items()}
        self.bounds = np.asarray([trial["config"]["source"][k] for k in ("source_p_bounds", "source_q_bounds")], float)
        require(np.array_equal(self.bounds, np.array([[-.5,.5],[-.5,.5]])), "original physical source box must be unchanged")
        self.cache, self.jcache = {}, {}; self.counts = Counter(independent_forward_batches=0,
            forward_cache_hits=0, derivative_cache_hits=0, derivative_columns=0)

    def key(self, name, p, q, u, source):
        return name+":"+dynamic._identity([p,q,u,source])

    def forward(self, name, p, q, u, source=None):
        source = self.sources[name] if source is None else np.asarray(source, float)
        key = self.key(name,p,q,u,source)
        if key in self.cache:
            self.counts["forward_cache_hits"] += 1; return self.cache[key]
        line = self.lines[name]
        phy = forward_physics(self.bank[name], line["r"], line["x"], np.asarray(p)+np.asarray(u)/20., q,
            source[:,0],source[:,1],self.root)
        out = dict(values=np.column_stack((phy["v"],phy["master_p"],phy["master_q"])),
            margin=float(min(phy["voltage_min"]-.85,1.15-phy["voltage_max"])))
        self.cache[key] = out; self.counts["independent_forward_batches"] += 1
        return out

    def derivative(self, name, p, q, u, *, kind):
        source = self.sources[name]
        key = self.key(name,p,q,u,source)+"/"+kind
        if key in self.jcache:
            self.counts["derivative_cache_hits"] += 1; return self.jcache[key]
        width = self.n if kind == "P" else 2
        answer = np.empty((self.T,self.n+2,width))
        for j in range(width):
            plus, minus = np.asarray(p if kind=="P" else source,float).copy(), np.asarray(p if kind=="P" else source,float).copy()
            plus[:,j] += self.step; minus[:,j] -= self.step
            if kind == "P":
                a,b = self.forward(name,plus,q,u), self.forward(name,minus,q,u)
            else:
                a,b = self.forward(name,p,q,u,plus), self.forward(name,p,q,u,minus)
            answer[:,:,j] = (a["values"]-b["values"])/(2*self.step)
            self.counts["derivative_columns"] += 1
        self.jcache[key] = answer
        return answer


def verify_pressure(trial):
    """Rebuild the declared five training-only PQ stress paths, without fitting."""
    selection, forecast = trial["selection"], trial["forecast"]
    saved = selection.get("load_pressure_scenarios")
    if saved is None:
        require(selection.get("load_pressure",{}).get("success") is False, "missing pressure paths without declared failure")
        return dict(available=False, retained_failure=True)
    meta = forecast["metadata"]; ar = meta["ar1"]
    p,q = (np.asarray(trial["training"][k],float) for k in ("p","q"))
    meter = dynamic._meter(trial["training_meter"])
    p,q = p-np.asarray(meter.bias_p), q-np.asarray(meter.bias_q)
    pf = np.clip(np.divide(np.maximum(p,0),np.hypot(np.maximum(p,0),np.maximum(q,0)),
        out=np.full_like(p,.97),where=p>0),.94,.99)
    pp = np.clip(np.asarray(ar["p"]["intercept"])+np.asarray(ar["p"]["phi"])*p[:-1],.05/20,12/20)
    pff = np.clip(np.asarray(ar["pf"]["intercept"])+np.asarray(ar["pf"]["phi"])*pf[:-1],.94,.99)
    errors = np.column_stack((p[1:]-pp,q[1:]-pp*np.tan(np.arccos(pff))))
    errors -= errors.mean(axis=0); cov = errors.T@errors/max(len(errors)-2,1)
    eigenvalues,eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(-eigenvalues,kind="stable"); eigenvalues=np.maximum(eigenvalues[order],0); eigenvectors=eigenvectors[:,order]
    for j in range(eigenvectors.shape[1]):
        if eigenvectors[np.argmax(abs(eigenvectors[:,j])),j] < 0: eigenvectors[:,j] *= -1
    report = selection["load_pressure"]
    close(report["covariance"],cov,"training PQ covariance",atol=1e-14,rtol=1e-10)
    close(report["eigenvalues"],eigenvalues,"training covariance eigenvalues",atol=1e-14,rtol=1e-9)
    close(report["eigenvectors"],eigenvectors,"training covariance eigenvectors",atol=1e-8,rtol=1e-7)
    scale = np.sqrt(np.asarray(meta["forecast_steps"],float))
    nominal_p,nominal_q = np.asarray(forecast["p"]), np.asarray(forecast["q"])
    expected = [("nominal",nominal_p,nominal_q)]
    n=nominal_p.shape[1]
    for pc in range(2):
        direction=np.sqrt(eigenvalues[pc])*eigenvectors[:,pc]
        for sign in (1,-1):
            delta=sign*scale[:,None]*direction
            pv=np.clip(nominal_p+delta[:,:n],.05/20,12/20)
            qv=np.clip(nominal_q+delta[:,n:],pv*np.tan(np.arccos(.99)),pv*np.tan(np.arccos(.94)))
            expected.append((f"pc{pc+1}_"+("plus" if sign>0 else "minus"),pv,qv))
    require([r["name"] for r in saved] == [r[0] for r in expected], "all five pressure paths must remain")
    for row,(name,pv,qv) in zip(saved,expected):
        close(row["p"],pv,"training-only pressure P "+name,atol=1e-13,rtol=1e-10)
        close(row["q"],qv,"training-only pressure Q "+name,atol=1e-13,rtol=1e-10)
        m=row["metadata"]
        require(m.get("training_only") is True and all(m.get(k) is False for k in
            ("uses_validation","uses_truth","uses_future_noise","statistical_coverage_claimed")), "pressure provenance")
    return dict(available=True, paths=5, covariance_rebuilt_from_training=True, coverage_claimed=False)


def verify_model(replay, model, p, q, *, projection, rcond, nominal_weights=None):
    T,n = replay.T,replay.n
    require(model.get("names") == list(NAMES), "model names")
    require(model.get("line_sha256") == {name:line_id(replay.lines[name]["r"],replay.lines[name]["x"]) for name in NAMES}, "model fixed lines")
    u,w = np.asarray(model["anchor_kw"],float),np.asarray(model["weights"],float)
    require(u.shape==(T,n) and w.shape==(T,n+2) and np.isfinite([u]).all() and np.isfinite(w).all(), "anchor and weight dimensions")
    square = next(row["power_kw"] for row in replay.trial["methods"] if row["name"]=="square")
    close(u,square,"public squared-power anchor",atol=0,rtol=0)
    close(model["sigma"],replay.sigma,"public output sigmas",atol=0,rtol=0)
    norms = np.linalg.norm(w,axis=1)
    require(np.all((abs(norms-1)<1e-8)|(norms<1e-12)), "contrast direction must have unit or zero norm")
    if nominal_weights is not None:
        close(w,nominal_weights,"all load scenarios keep nominal w",atol=0,rtol=0)
    values = {name:replay.forward(name,p,q,u)["values"] for name in NAMES}
    delta = (values[NAMES[0]]-values[NAMES[1]])/replay.sigma
    Js = {name:replay.derivative(name,p,q,u,kind="source")/replay.sigma[:,:,None] for name in NAMES}
    Jp = {name:replay.derivative(name,p,q,u,kind="P")/replay.sigma[:,:,None] for name in NAMES}
    beta = np.zeros((T,T*n)); rank_rows=[]; derivative_errors=[]
    require(len(model["direction_records"])==T,"all direction rows required")
    lo,hi = replay.bounds[:,0],replay.bounds[:,1]
    for t,row in enumerate(model["direction_records"]):
        beta[t,t*n:(t+1)*n]=w[t]@(Jp[NAMES[0]][t]-Jp[NAMES[1]][t])/20.
        Ga,Gb=Js[NAMES[0]][t],Js[NAMES[1]][t]
        leakage_a,leakage_b=w[t]@Ga,w[t]@Gb
        derivative_errors.append(max(close(row["source_leakage_a"],leakage_a,"source A derivative",atol=1e-6,rtol=1e-7),
            close(row["source_leakage_b"],leakage_b,"source B derivative",atol=1e-6,rtol=1e-7)))
        G=np.column_stack((Ga,Gb)) if projection=="joint" else (Ga+Gb)/2
        singular=np.linalg.svd(G,compute_uv=False)
        saved_s=np.asarray(row["source_singular_values"],float)
        close(saved_s,singular,"independent source singular values",atol=1e-6,rtol=1e-8)
        cutoff=float(rcond*saved_s[0]); close(row["rank_cutoff"],cutoff,"source rank cutoff",atol=1e-13,rtol=1e-10)
        require(row["source_rank"]==int(np.sum(saved_s>cutoff)),"saved rank/cutoff arithmetic")
        fd_rank=int(np.sum(singular>rcond*singular[0])); require(fd_rank==row["source_rank"],"independent source numerical rank")
        coeff=np.r_[row["source_leakage_a"],-np.asarray(row["source_leakage_b"])]
        lower=np.r_[lo-replay.sources[NAMES[0]][t],lo-replay.sources[NAMES[1]][t]]
        upper=np.r_[hi-replay.sources[NAMES[0]][t],hi-replay.sources[NAMES[1]][t]]
        low=float(np.sum(coeff*np.where(coeff>=0,lower,upper)))
        high=float(np.sum(coeff*np.where(coeff>=0,upper,lower)))
        close([row["source_support_low"],row["source_support_high"]],[low,high],"original-box exact support",atol=1e-10,rtol=1e-8)
        close([model["source_low"][t],model["source_high"][t]],[low,high],"model/row source support",atol=1e-10,rtol=1e-8)
        if nominal_weights is None:
            projected_leak = np.r_[row["source_leakage_a"],row["source_leakage_b"]] if projection=="joint" else (
                np.asarray(row["source_leakage_a"])+np.asarray(row["source_leakage_b"]))/2
            require(np.linalg.norm(projected_leak)<=cutoff+1e-8,"nominal weight outside retained source-null directions")
        rank_rows.append(dict(time=t,rank=fd_rank,rank_cutoff=cutoff,smallest_singular=float(singular[-1]),
            absolute_distance_to_rank_cutoff=float(min(abs(singular-cutoff)))))
    beta_error=close(model["beta"],beta,"independent beta per kW",atol=1e-6,rtol=1e-7)
    alpha=np.einsum("to,to->t",w,delta)-np.asarray(model["beta"])@u.ravel()
    close(model["alpha"],alpha,"anchor alpha",atol=1e-7,rtol=1e-8)
    require(model.get("global_AC_source_elimination_certified") is False,"affine support cannot certify AC source elimination")
    return dict(time_rows=T,unit_weights=True,source_support_original_box=True,
        beta_max_absolute_FD_error=beta_error,source_derivative_max_absolute_FD_error=max(derivative_errors),
        source_rank_rows=rank_rows,FD_step_pu=replay.step,
        derivative_scope="simultaneous same-column perturbations across independent snapshots")


def gap_values(d, low, high):
    return np.maximum(np.maximum(0.,np.asarray(d)+low),-np.asarray(d)-high)


def verify_evaluation(replay, model, p, q, u, saved):
    a,b=model["names"]; w=np.asarray(model["weights"]); sig=replay.sigma
    fa,fb=replay.forward(a,p,q,u),replay.forward(b,p,q,u)
    d=np.einsum("to,to->t",w,(fa["values"]-fb["values"])/sig)
    affine=np.asarray(model["alpha"])+np.asarray(model["beta"])@np.asarray(u).ravel()
    low,high=np.asarray(model["source_low"]),np.asarray(model["source_high"])
    gap,agap=gap_values(d,low,high),gap_values(affine,low,high)
    for key,expected in (("scalar_AC_contrast",d),("scalar_affine_contrast",affine),
        ("scalar_affine_error",d-affine),("source_adjusted_gaps",gap),("affine_source_adjusted_gaps",agap)):
        close(saved[key],expected,"independent "+key,atol=1e-7,rtol=1e-8)
    score=float(gap.sum()); close(saved["score"],score,"independent fixed-w AC score",atol=1e-7,rtol=1e-8)
    close(saved["nominal_projected_half_SSE"],float(d@d)/2,"nominal projected half SSE",atol=2e-7,rtol=1e-8)
    close(saved["affine_source_distance_lower_bound_from_L1"],float(agap.sum()**2)/(2*len(agap)),"affine L1 lower bound",atol=2e-7,rtol=1e-8)
    close(saved["source_adjusted_AC_contrast_proxy_squared"],score**2/(2*len(gap)),"AC proxy squared is separate",atol=2e-7,rtol=1e-8)
    require(saved.get("full_AC_source_distance_certificate") is False,"AC contrast proxy is not a source-profile certificate")
    margin=min(fa["margin"],fb["margin"])
    close(saved["terminal_and_internal_voltage_margin"],margin,"internal/terminal voltage margin",atol=3e-10,rtol=0)
    require(saved.get("success") == bool(margin>=0),"physical evaluation status")
    source_error=0.
    for name in (a,b):
        J=replay.derivative(name,p,q,u,kind="source")/sig[:,:,None]
        leak=np.einsum("to,tos->ts",w,J)
        source_error=max(source_error,close(saved["source_derivative_at_action"][name],leak,
            "source derivative at actual action",atol=1e-6,rtol=1e-7))
    return dict(score=score,affine_score=float(agap.sum()),success=bool(margin>=0),
        source_derivative_max_absolute_FD_error=source_error)


def review_contrast_trial(trial):
    replay=Replay(trial); checks=[]; errors=[]; counts=Counter(models=0,evaluations=0,accepted_candidates=0,retained_failures=0)
    def check(label, function):
        try:
            out=function(); checks.append(dict(name=label,status="passed",result=out)); return out
        except (ValueError,KeyError,TypeError,IndexError,AssertionError,np.linalg.LinAlgError) as exc:
            message=type(exc).__name__+":"+str(exc); errors.append(dict(name=label,message=message))
            checks.append(dict(name=label,status="failed",message=message)); return None
    check("training_only_load_pressure",lambda:verify_pressure(trial))
    actions={r["name"]:np.asarray(r["power_kw"],float) for r in trial["methods"] if r.get("power_kw") is not None}
    p,q=(np.asarray(trial["forecast"][k],float) for k in ("p","q"))
    scenarios={"nominal":(p,q)}
    for row in trial["selection"].get("load_pressure_scenarios",[]) or []:
        scenarios[row["name"]]=(np.asarray(row["p"],float),np.asarray(row["q"],float))
    for method in ("contrast_joint","contrast_box","contrast_load"):
        record=trial["selection"].get("methods",{}).get(method,{})
        if not record.get("success"):
            counts["retained_failures"]+=1
            check(method+"/failure_preserved",lambda method=method,record=record:
                require(method in actions and bool(record.get("failure") or record.get("failures")),"failed contrast plan disappeared"))
            continue
        models=record.get("scenario_models",[record["model"]]); names=record["scenario_names"]
        require(len(models)==len(names)==(5 if method=="contrast_load" else 1),"scenario denominator")
        require(names==(["nominal","pc1_plus","pc1_minus","pc2_plus","pc2_minus"] if method=="contrast_load" else ["nominal"]),"scenario order")
        projection="joint" if method=="contrast_joint" else "bounded_average"
        for index,(name,model) in enumerate(zip(names,models)):
            pp,qq=scenarios[name]
            check(method+"/model/"+name,lambda model=model,pp=pp,qq=qq,index=index:
                verify_model(replay,model,pp,qq,projection=projection,rcond=record["configuration"]["rcond"],
                    nominal_weights=models[0]["weights"] if index else None))
            counts["models"]+=1
        require(dynamic._identity(record["model"])==dynamic._identity(models[0]),"nominal duplicate model payload differs")
        def evaluate(u,saved,label,eval_models=None):
            model_list=models if eval_models is None else eval_models
            entries=saved.get("scenario_evaluations")
            entries=([entries[name] for name in names] if isinstance(entries,dict) else entries) if entries is not None else [saved]
            require(len(entries)==len(model_list),"all AC scenario evaluations required")
            values=[]
            for name,model,entry in zip(names,model_list,entries):
                pp,qq=scenarios[name]
                value=check(label+"/"+name,lambda model=model,pp=pp,qq=qq,entry=entry:
                    verify_evaluation(replay,model,pp,qq,u,entry))
                counts["evaluations"]+=1
                if value is None:return None
                values.append(value)
            score=min(v["score"] for v in values)
            close(saved["score"],score,"minimum-scenario AC score",atol=2e-7,rtol=1e-8)
            return score
        initial=[]
        for index,row in enumerate(record["initial_candidates"]):
            if row.get("evaluation",{}).get("success"):
                origin=row["origin"]
                require(origin in actions,"declared baseline origin missing")
                value=evaluate(actions[origin],row["evaluation"],method+"/initial/"+str(index))
                if value is not None:initial.append((value,origin))
        if not initial: continue
        atol,rtol=(record["configuration"][k] for k in ("acceptance_atol","acceptance_rtol"))
        best=initial[0]
        for value,origin in initial[1:]:
            if value>best[0]+atol+rtol*abs(best[0]):best=(value,origin)
        check(method+"/initial_best_score",lambda:close(record["initial_best_score"],best[0],"initial best same-metric score"))
        milp=record.get("MILP",{})
        if milp.get("success"):
            def audit_milp():
                alpha=np.concatenate([m["alpha"] for m in models]);beta=np.vstack([m["beta"] for m in models])
                lo=np.concatenate([m["source_low"] for m in models]);hi=np.concatenate([m["source_high"] for m in models])
                R=len(alpha);T=replay.T
                require(milp["added_binary_count"]==3*R== (90 if method=="contrast_load" else 18),"exact interval selector binary count")
                require(milp["added_continuous_count"]==R+int(len(models)>1),"interval auxiliary count")
                lower=np.zeros((T,replay.n));upper=np.zeros_like(lower)
                lower[:,0]=-fixed.CONFIG["command_power_kw"];upper[:,0]=fixed.CONFIG["command_power_kw"]
                dl=alpha+(beta*np.where(beta>=0,lower.ravel(),upper.ravel())).sum(axis=1)
                dh=alpha+(beta*np.where(beta>=0,upper.ravel(),lower.ravel())).sum(axis=1)
                vu=np.maximum(np.maximum(0.,dh+lo),-dl-hi)
                L=np.column_stack((np.zeros(R),dl+lo,-dh-hi))
                close(milp["contrast_lower"],dl,"scalar affine control lower",atol=1e-8,rtol=1e-8)
                close(milp["contrast_upper"],dh,"scalar affine control upper",atol=1e-8,rtol=1e-8)
                close(milp["gap_upper"],vu,"gap upper",atol=1e-8,rtol=1e-8)
                close(milp["big_M"],vu[:,None]-L,"valid big M",atol=1e-8,rtol=1e-8)
                u=np.asarray(milp["power_kw"])
                gap=gap_values(alpha+beta@u.ravel(),lo,hi)
                value=min(float(gap[i*T:(i+1)*T].sum()) for i in range(len(models)))
                close(milp["proxy_value"],value,"affine MILP incumbent",atol=2e-6,rtol=1e-8)
                close(milp["solver_value"],value,"solver/independent affine objective",atol=2e-6,rtol=1e-8)
                require(milp.get("global_AC_gap") is None and "affine" in milp["bound_scope"],"MILP bound is affine only")
                if milp.get("proxy_best_bound") is not None:
                    require(value<=milp["proxy_best_bound"]+1e-6+1e-8*abs(value),"MILP incumbent exceeds best bound")
                return dict(affine_incumbent=value,best_bound=milp.get("proxy_best_bound"),added_binaries=3*R)
            check(method+"/MILP",audit_milp)
        if record.get("candidate_evaluation") is not None:
            candidate=evaluate(np.asarray(milp["power_kw"]),record["candidate_evaluation"],method+"/candidate")
            if candidate is not None:
                accepted=bool(record["candidate_evaluation"]["success"] and candidate>best[0]+atol+rtol*abs(best[0]))
                check(method+"/accepted",lambda accepted=accepted:require(record["candidate_accepted"]==accepted,"same-metric strict AC acceptance"))
                if accepted:best=(candidate,"contrast_MILP");counts["accepted_candidates"]+=1
        value=evaluate(actions[method],record["selected_evaluation"],method+"/selected")
        if value is not None:
            check(method+"/selected_score",lambda:close(record["selected_score"],value,"selected independent AC contrast score"))
            check(method+"/best_retained",lambda:require(abs(value-best[0])<=2e-7+1e-8*abs(best[0]) and record["selected_origin"]==best[1],"selected same-metric best plan/origin"))
        # Post-planning diagnostic tables contain all executed actions, including
        # no EV. They use the same fixed direction and never choose a new plan.
        diagnostic=trial.get("planning_diagnostics",{})
        label="joint" if method=="contrast_joint" else "box"
        if method=="contrast_load":
            dm=diagnostic.get("load_models",{})
            diagnostic_models=[dm[name] for name in names] if set(dm)==set(names) else []
        else:
            dm=diagnostic.get("models",{}).get(label)
            diagnostic_models=[dm] if dm is not None else []
        require(len(diagnostic_models)==len(models),"post-planning diagnostic model denominator")
        for index,(name,model) in enumerate(zip(names,diagnostic_models)):
            pp,qq=scenarios[name]
            check(method+"/diagnostic_model/"+name,lambda model=model,pp=pp,qq=qq,index=index:
                verify_model(replay,model,pp,qq,projection=projection,rcond=record["configuration"]["rcond"],
                    nominal_weights=diagnostic_models[0]["weights"] if index else None))
            counts["models"]+=1
        for action in METHODS:
            row=diagnostic.get("methods",{}).get(action,{})
            saved=row.get(method,{})
            require(saved,"missing diagnostic action "+action+"/"+method)
            if saved.get("success"):
                evaluate(actions[action],saved,method+"/diagnostic_action/"+action,diagnostic_models)
            else:
                counts["retained_failures"]+=1
                check(method+"/diagnostic_failure/"+action,lambda saved=saved:
                    require(bool(saved.get("failure") or saved.get("scenario_evaluations")),"failed diagnostic action lacks reason"))
    return dict(success=bool(counts["models"]) and not errors,checks=checks,errors=errors,counts=dict(counts),
        independent_forward_counts=dict(replay.counts),no_fit=True,no_selector=True,no_production_AC_predictor=True)


def review_directory(directory, output=None):
    directory = Path(directory); freeze = dynamic._read(directory/"freeze.json")
    snapshot = dynamic._read(directory/"source_snapshot.json.gz")
    tool_names=("ev_response_metric_review.py","test_ev_response_metric_review.py",
        "ev_dynamic_review.py","ev_gradient_flow_review.py","ev_original_review.py",
        "ev_fixed_train_review.py")
    tools={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in tool_names}
    rows = []
    for path in sorted(directory.glob("trial_*.json.gz")):
        try:
            trial=dynamic._read(path)
            _, report = adapted_reviews(trial, freeze, snapshot)
            report["contrast_review"]=review_contrast_trial(trial)
            report["success"] = report["success"] and report["contrast_review"]["success"]
        except (ValueError, KeyError, TypeError, IndexError, np.linalg.LinAlgError) as exc:
            report = dict(success=False, review_exception=type(exc).__name__+":"+str(exc))
        report.update(input_path=str(path.resolve()), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        rows.append(report)
    totals={name:Counter() for name in ("base","profile","contrast","contrast_forward")}
    for row in rows:
        totals["base"].update(row.get("base_review",{}).get("counts",{}))
        totals["profile"].update(row.get("profile_review",{}).get("counts",{}))
        totals["contrast"].update(row.get("contrast_review",{}).get("counts",{}))
        totals["contrast_forward"].update(row.get("contrast_review",{}).get("independent_forward_counts",{}))
    source_unchanged=all(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest for name,digest in tools.items())
    result = dict(schema="strict_binary_response_metric_review_v1", success=bool(rows) and source_unchanged and all(r["success"] for r in rows),
        trial_count=len(rows), verified_trials=sum(r["success"] for r in rows), reviews=rows,
        no_new_fit=True, no_selector=True, no_optimizer=True, no_production_AC_predictor=True,
        policy_id=freeze["policy_id"],source_count=len(freeze["source_sha256"]),
        freeze_sha256=hashlib.sha256((directory/"freeze.json").read_bytes()).hexdigest(),
        snapshot_sha256=hashlib.sha256((directory/"source_snapshot.json.gz").read_bytes()).hexdigest(),
        reviewer_sha256=tools["ev_response_metric_review.py"],review_tool_sha256=tools,
        reviewer_source_unchanged=source_unchanged,counts={k:dict(v) for k,v in totals.items()})
    if output is not None:
        output = Path(output)
        require(output.resolve() not in {p.resolve() for p in directory.glob("trial_*.json.gz")}
            | {(directory/"freeze.json").resolve(), (directory/"source_snapshot.json.gz").resolve()}, "output would overwrite raw evidence")
        output.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(dynamic._serial(result), ensure_ascii=False, sort_keys=True, allow_nan=False)+"\n").encode()
        output.write_bytes(gzip.compress(data, mtime=0) if output.suffix == ".gz" else data)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); result = review_directory(args.directory, args.output)
    print(json.dumps({k:result[k] for k in ("success", "trial_count", "verified_trials")}))
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__": main()
