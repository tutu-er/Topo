"""Historical-set constrained local T design: small synthetic study only.

Design APIs accept historical observations and declared future P only. Full
source/measurement LP evaluation is delegated to the unchanged v6 module.
The convex QP has an exact-rational tangent/support lower certificate and an
exact feasible upper certificate. It is not a probability or full-grid method.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import LinearConstraint, linprog, minimize

import local_shared_source_v6 as v6


NAMES = tuple(v6.CANDIDATES)
BOX = ((.01, .05), (.000001, .03), (.01, .12), (.01, .12), (.01, .12))
PROFILES = {
    "symmetric": (.02, .01, .06, .06, .06),
    "asymmetric": (.027, .018, .035, .081, .052),
    "weak": (.02, .000001, .06, .06, .06),
}
FORMAL_SEEDS = (7201, 7202, 7203)
PILOT_SEEDS = (601, 602)
SCORE_TIE = 1e-10
REFERENCE_TIE = 1e-7


def f(x):
    return F(float(x))


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prediction_coeff(p, source_nominal, h, i):
    z = v6.CANDIDATES[h]
    return (sum(map(f, p), F()) + f(source_nominal),
            sum((f(p[k]) for k in z), F()) if i in z else F(),
            *(f(p[i]) if k == i else F() for k in range(3)))


@dataclass
class Polytope:
    exact_a: tuple
    exact_b: tuple
    a: np.ndarray
    b_outer: np.ndarray
    b_inner: np.ndarray
    offset: tuple
    width: tuple
    candidate: str

    def exact_feasible(self, x):
        xf = tuple(map(f, x))
        return (all(0 <= z <= 1 for z in xf) and
                all(sum((c*z for c, z in zip(row, xf)), F()) <= b
                    for row, b in zip(self.exact_a, self.exact_b)))

    def theta(self, x):
        return np.array([float(l + w*f(z)) for l, w, z in zip(self.offset, self.width, x)])


def build_history_polytope(observed, candidate, box=BOX):
    """Exact elimination of one shared source w_t via interval intersection.

    Output inequalities in normalized x=(theta-lower)/(upper-lower), [0,1]^5.
    Original rational coefficients are retained, outward float rows bound the
    support LP, tightened rows merely help find a feasible incumbent.
    """
    p, y = np.asarray(observed["P_total"]), np.asarray(observed["Y_drop"])
    eps = np.broadcast_to(observed["drop_error_bound"], y.shape)
    sn = np.broadcast_to(observed["source_nominal"], (len(p),))
    sb = np.broadcast_to(observed["source_error_bound"], (len(p),))
    offset = tuple(f(lo) for lo, hi in box)
    width = tuple(f(hi)-f(lo) for lo, hi in box)
    unique = {}
    def add(co, rhs):
        row = tuple(c*w for c,w in zip(co,width))
        rhs -= sum((c*l for c,l in zip(co,offset)), F())
        scale = max(map(abs, row), default=F(1)) or F(1)
        row = tuple(c/scale for c in row); rhs /= scale
        unique[row] = min(unique.get(row, rhs), rhs)
    for t in range(len(p)):
        rows = [prediction_coeff(p[t], sn[t], candidate, i) for i in range(3)]
        for i in range(3):
            for j in range(3):
                if i != j:
                    add(tuple(b-a for a,b in zip(rows[i], rows[j])),
                        f(y[t,j])-f(y[t,i])+f(eps[t,i])+f(eps[t,j]))
            plus = list(rows[i]); plus[0] -= f(sb[t])
            minus = [-v for v in rows[i]]; minus[0] -= f(sb[t])
            add(tuple(plus), f(y[t,i])+f(eps[t,i]))
            add(tuple(minus), -f(y[t,i])+f(eps[t,i]))
    erows, rhs = tuple(unique.keys()), tuple(unique.values())
    a = np.array([[float(v) for v in row] for row in erows])
    guards = [sum((abs(f(c)-ce) for c,ce in zip(row, ex)), F()) for row,ex in zip(a,erows)]
    bout = np.array([v6._up(b+g) for b,g in zip(rhs,guards)])
    # Strict numerical interior used only for candidate construction. This is
    # not the domain used in the certified support lower bound.
    binward = np.array([v6._down(b-g)-2e-10 for b,g in zip(rhs,guards)])
    return Polytope(erows, rhs, a, bout, binward, offset, width, candidate)


@dataclass
class Quadratic:
    rows: tuple
    targets: tuple
    weights: tuple

    def exact_value_gradient(self, x):
        xf = tuple(map(f, x)); val = F(); grad = [F()]*5
        for row, target, weight in zip(self.rows, self.targets, self.weights):
            r = sum((a*z for a,z in zip(row,xf)), F())-target
            val += weight*r*r
            for k in range(5):
                grad[k] += 2*weight*r*row[k]
        return val, tuple(grad)

    def arrays(self):
        return (np.array([[float(v) for v in row] for row in self.rows]),
                np.array(list(map(float,self.targets))), np.array(list(map(float,self.weights))))


def midpoint_objective():
    return Quadratic(tuple(tuple(F(i==j) for j in range(5)) for i in range(5)),
                     (F(1,2),)*5, (F(1),)*5)


def design_objective(poly, reference, p_new):
    """||C d||^2 = (1/3) sum_{i<j}(d_i-d_j)^2; no irrational C needed."""
    rname, rtheta = reference["candidate"], reference["theta"]
    patterns, counts = np.unique(np.asarray(p_new), axis=0, return_counts=True)
    rows, targets, weights = [], [], []
    for p, count in zip(patterns, counts):
        cr = [prediction_coeff(p, 0., rname, i) for i in range(3)]
        ch = [prediction_coeff(p, 0., poly.candidate, i) for i in range(3)]
        for i,j in ((0,1),(0,2),(1,2)):
            d = tuple(a-b for a,b in zip(ch[i],ch[j]))
            target = sum(((a-b)*f(t) for a,b,t in zip(cr[i],cr[j],rtheta)), F())
            target -= sum((a*l for a,l in zip(d,poly.offset)), F())
            rows.append(tuple(a*w for a,w in zip(d,poly.width)))
            targets.append(target); weights.append(F(int(count),3))
    return Quadratic(tuple(rows),tuple(targets),tuple(weights))


def solve_qp(poly, objective, start=None, *, scale=1.):
    """Convex feasible upper plus global tangent/support lower; not status-only."""
    began = perf_counter()
    a, target, weights = objective.arrays()
    def fun(x):
        r = a@x-target
        return scale*float(np.dot(weights,r*r))
    def jac(x):
        return scale*2*a.T@(weights*(a@x-target))
    feas = linprog(np.zeros(5), A_ub=poly.a, b_ub=poly.b_inner,
                   bounds=[(0.,1.)]*5, method="highs",
                   options={"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9})
    if not feas.success:
        return dict(status="history_infeasible_or_numerically_thin", lower=None, upper=None,
                    solver_message=feas.message, seconds=perf_counter()-began)
    x0 = np.asarray(start if start is not None and poly.exact_feasible(start) else feas.x)
    sol = minimize(fun,x0,jac=jac,method="SLSQP",bounds=[(0.,1.)]*5,
                   constraints=[LinearConstraint(poly.a,-np.inf,poly.b_inner)],
                   options={"ftol":1e-12,"maxiter":500})
    candidates = [np.clip(sol.x,0.,1.),np.clip(feas.x,0.,1.)]
    if start is not None:
        candidates.append(np.clip(start,0.,1.))
    feasible = [x for x in candidates if poly.exact_feasible(x)]
    if not feasible:
        return dict(status="no_exact_feasible_incumbent",lower=None,upper=None,
                    solver_message=sol.message, seconds=perf_counter()-began)
    x = min(feasible,key=fun)
    value, gradient = objective.exact_value_gradient(x)
    support = linprog(np.array(list(map(float,gradient))), A_ub=poly.a,b_ub=poly.b_outer,
                      bounds=[(0.,1.)]*5,method="highs",
                      options={"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9})
    if not support.success:
        return dict(status="support_solver_failure",lower=None,upper=v6._up(value),x=x.tolist())
    # Any lambda<=0 gives a valid lower bound for min g'x over the outer
    # history polytope. Box correction uses the exact gradient, not float g.
    lam = tuple(f(min(0.,v)) for v in support.ineqlin.marginals)
    linear_lower = sum((f(b)*l for b,l in zip(poly.b_outer,lam)),F())
    reduced = tuple(g-sum((f(row[k])*l for row,l in zip(poly.a,lam)),F())
                    for k,g in enumerate(gradient))
    linear_lower += sum((min(F(),z) for z in reduced),F())
    lower = max(F(),value-sum((g*f(z) for g,z in zip(gradient,x)),F())+linear_lower)
    if lower > value:
        raise RuntimeError("invalid QP lower/upper interval")
    return dict(status="certified_interval",lower=v6._down(lower),upper=v6._up(value),
                gap=v6._up(value-lower),x=x.tolist(),theta=poly.theta(x).tolist(),
                exact_lower=str(lower),exact_upper=str(value),
                exact_feasible_verified=True,exact_tangent_dual_verified=True,
                raw_solver_success=bool(sol.success),raw_solver_message=str(sol.message),
                iterations=int(sol.nit),seconds=perf_counter()-began)


def history_representatives(history, box=BOX):
    polys = {h:build_history_polytope(history,h,box) for h in NAMES}
    reps = {}
    for h,poly in polys.items():
        fit = solve_qp(poly,midpoint_objective())
        reps[h] = dict(candidate=h,**fit)
    valid = [h for h in NAMES if reps[h].get("lower") is not None]
    if not valid:
        raise RuntimeError("no historical reference")
    best_upper = min(reps[h]["upper"] for h in valid)
    possible = [h for h in valid if reps[h]["lower"] <= best_upper+REFERENCE_TIE]
    chosen = possible[0]
    return polys,reps,dict(selected=chosen,possible=possible,
                          unique_resolved=len(possible)==1,tolerance=REFERENCE_TIE)


def action_inputs(site,policy,background=None):
    p = np.full((24,3),.2) if background is None else np.array(background,float,copy=True)
    if policy == "constant":
        ev = np.full(24,.15)
    elif policy == "endpoint":
        ev = np.tile([0.,.3,.3,0.],6)
    else:
        raise ValueError(policy)
    p[:,site] += ev
    return p,ev


def rank_intervals(scores,tolerance=SCORE_TIE):
    lo = max(scores[h]["lower"] for h in scores)
    possible = [h for h in scores if scores[h]["upper"] >= lo-tolerance]
    return dict(selected=possible[0],possible=possible,unique_resolved=len(possible)==1,tolerance=tolerance)


def design_scores(history,background=None,box=BOX):
    """No simulation truth or future outputs enter this function."""
    polys,reps,local = history_representatives(history,box)
    records,score_by_reference = [],{}
    for r,ref in reps.items():
        if ref.get("lower") is None:
            continue
        score_by_reference[r] = {}
        for site in range(3):
            for policy in ("constant","endpoint"):
                name = f"site{site+1}_{policy}"
                p,_ = action_inputs(site,policy,background)
                fits = []
                for h,poly in polys.items():
                    if h == r or reps[h].get("lower") is None:
                        continue
                    fit = solve_qp(poly,design_objective(poly,ref,p),reps[h]["x"],scale=1e6)
                    records.append(dict(reference=r,candidate=h,action=name,result=fit))
                    fits.append(fit)
                if not fits or any(v.get("lower") is None for v in fits):
                    raise RuntimeError("design lacks certified rival score")
                score_by_reference[r][name] = dict(lower=min(v["lower"] for v in fits),
                                                   upper=min(v["upper"] for v in fits))
    local_scores = score_by_reference[local["selected"]]
    committee = {name:dict(lower=float(np.mean([score_by_reference[r][name]["lower"] for r in score_by_reference])),
                          upper=float(np.mean([score_by_reference[r][name]["upper"] for r in score_by_reference])))
                 for name in local_scores}
    endpoint_local = {k:v for k,v in local_scores.items() if k.endswith("endpoint")}
    endpoint_committee = {k:v for k,v in committee.items() if k.endswith("endpoint")}
    return dict(reference_selection=local,representatives=reps,qp_records=records,
                scores_by_reference=score_by_reference,local_scores=local_scores,
                committee_scores=committee,local_selection=rank_intervals(endpoint_local),
                committee_selection=rank_intervals(endpoint_committee),
                committee_semantics="equal design weights, not topology probabilities")


def simulate(topology,theta,seed,site,policy,*,epsilon=1e-5,rich_history=False,correlated=False):
    """Only this generator/evaluator sees truth; it returns separate objects."""
    rng = np.random.default_rng(seed)
    oldp = np.full((12,3),.2)
    if rich_history:
        oldp += np.tile(np.array([[.02,0,0],[0,.02,0],[0,0,.02],[0,0,0]]),(3,1))
    newp,ev = action_inputs(site,policy)
    p = np.r_[oldp,newp]
    delta = rng.uniform(-.02,.02,36)
    if correlated:
        delta[12:] = (.02/.15)*(ev-.15)
    noise = rng.uniform(-.8*epsilon,.8*epsilon,(36,3))
    exact_y = [[sum((c*f(v) for c,v in zip(prediction_coeff(pt,.12,topology,i),theta)),F())+f(theta[0])*f(delta[t])
                for i in range(3)] for t,pt in enumerate(p)]
    y0 = np.array([[float(v) for v in row] for row in exact_y])
    y = y0+noise
    actualerr = np.array([[float(f(y[t,i])-exact_y[t][i]) for i in range(3)] for t in range(36)])
    obs = dict(P_total=p,Y_drop=y,source_nominal=np.full(36,.12),source_error_bound=np.full(36,.1),
               drop_error_bound=np.full((36,3),epsilon),batch_id=np.r_[np.zeros(12,int),np.ones(24,int)])
    soc = .4+np.r_[0.,np.cumsum(ev*10*.25*.95)]/50.
    truth = dict(topology=topology,theta=list(theta),seed=seed,source_delta=delta.tolist(),
                 actual_error=actualerr.tolist(),Y_true=y0.tolist(),ev_pu=ev.tolist(),
                 energy_kwh=float(ev.sum()*10*.25),soc_range=[float(soc.min()),float(soc.max())],
                 coefficient_box=BOX,epsilon=epsilon,rich_history=rich_history,correlated=correlated,
                 checks=dict(source=bool(abs(delta).max()<=.1),noise=bool(abs(actualerr).max()<=epsilon),
                             ev=bool(abs(ev).max()<=.3 and soc.min()>=.2 and soc.max()<=.9),
                             energy=bool(abs(ev.sum()*10*.25-9)<1e-12)))
    return obs,truth


def observation_slice(obs,selector):
    return {k:np.asarray(v)[selector].copy() for k,v in obs.items()}


def evaluate_actions(topology,theta,seed,*,epsilon=1e-5,rich_history=False,correlated=False,output=None,prefix=""):
    first,_ = simulate(topology,theta,seed,0,"constant",epsilon=epsilon,rich_history=rich_history,correlated=correlated)
    history = observation_slice(first,slice(0,12))
    design = design_scores(history)
    records,inputs = [],[]
    for site in range(3):
        for policy in ("constant","endpoint"):
            action = f"site{site+1}_{policy}"
            obs,truth = simulate(topology,theta,seed,site,policy,epsilon=epsilon,rich_history=rich_history,correlated=correlated)
            if output:
                ofile = output/f"{prefix}_{action}_observations.npz"
                tfile = output/f"{prefix}_{action}_evaluation_truth.json"
                np.savez_compressed(ofile,**obs); dump(tfile,truth)
                inputs.append(dict(action=action,observations=ofile.name,observations_hash=sha(ofile),
                                   truth=tfile.name,truth_hash=sha(tfile),domain_checks=truth["checks"]))
            for h in NAMES:
                records.append(dict(action=action,candidate=h,result=v6.solve_lp(obs,h,parameter_box=BOX)))
    return dict(design=design,evaluation=records,inputs=inputs,
                summary=summarize(topology,design,records))


def summarize(truth,design,records):
    actionstats = {}
    for action in dict.fromkeys(r["action"] for r in records):
        subset = [r for r in records if r["action"]==action]
        wrong = [r["result"] for r in subset if r["candidate"]!=truth]
        actionstats[action] = dict(excluded_wrong=sum(r["status"]=="excluded" for r in wrong),
            closest_wrong_lower=min(r["reliable_lower_bound"] for r in wrong),
            closest_wrong_upper=min(r["verified_upper_bound"] for r in wrong),
            truth_excluded=any(r["result"]["status"]=="excluded" for r in subset if r["candidate"]==truth))
    endpoint = {k:dict(lower=v["closest_wrong_lower"],upper=v["closest_wrong_upper"])
                for k,v in actionstats.items() if k.endswith("endpoint")}
    oracle = rank_intervals(endpoint,1e-9)
    selected = {m:design[m+"_selection"]["selected"] for m in ("local","committee")}
    return dict(actionstats=actionstats,oracle_margin_sites=oracle,selected=selected,
                selected_reference=design["reference_selection"]["selected"],
                reference_correct=design["reference_selection"]["selected"]==truth,
                selection_hits_oracle={m:a in oracle["possible"] for m,a in selected.items()})


def protocol():
    return dict(version=7,created=datetime.now(timezone.utc).isoformat(),
        central_claim="history-only local T site design improves actual local discrimination, tested rather than assumed",
        model="three-terminal linear squared-drop; one known common source direction; P only; no AC",
        box=BOX,profiles=PROFILES,topologies=NAMES,seeds=FORMAL_SEEDS,pilot_seeds=PILOT_SEEDS,
        historical_samples=12,new_samples=24,background_pu=[.2,.2,.2],epsilon=1e-5,
        source_budget=.1,actual_source_uniform=[-.02,.02],actual_noise_fraction=.8,
        actions=[f"site{s}_{p}" for s in (1,2,3) for p in ("constant","endpoint")],
        ev=dict(energy_kwh=9,slots=24,hours_per_slot=.25,kw_max=3,constant_kw=1.5,q=0,
                battery_kwh=50,efficiency=.95,initial_soc=.4,soc_limits=[.2,.9]),
        local_reference="minimum normalized distance to box midpoint; interval overlap with tolerance breaks lexicographically",
        committee="uniform average of per-reference minimum rival squared scores; design weights only",
        score_tie=SCORE_TIE,reference_tie=REFERENCE_TIE,
        certificate="exact rational feasible QP upper; convex tangent plus box-corrected outer LP dual lower",
        evaluation="unchanged v6 full common-source joint LP with exact rational primal/dual certificates",
        source_hash=sha(__file__),v6_hash=sha(v6.__file__),python=platform.python_version(),
        numpy=np.__version__,scipy=scipy.__version__)


def aggregate(episodes):
    metrics = {}
    for method in ("local","committee","fixed_site1"):
        data = []
        for ep in episodes:
            s = ep["summary"]
            action = "site1_endpoint" if method=="fixed_site1" else s["selected"][method]
            d = s["actionstats"][action]
            c = s["actionstats"][action.replace("endpoint","constant")]
            fixed = s["actionstats"]["site1_endpoint"]
            data.append((d,c,fixed,action in s["oracle_margin_sites"]["possible"]))
        metrics[method] = dict(episodes=len(data),both_wrong_excluded=sum(d["excluded_wrong"]==2 for d,c,fx,o in data),
            total_wrong_excluded=sum(d["excluded_wrong"] for d,c,fx,o in data),
            selected_constant_both_wrong_excluded=sum(c["excluded_wrong"]==2 for d,c,fx,o in data),
            margin_better_than_fixed=sum(d["closest_wrong_lower"]>fx["closest_wrong_upper"]+1e-9 for d,c,fx,o in data),
            margin_worse_than_fixed=sum(d["closest_wrong_upper"]+1e-9<fx["closest_wrong_lower"] for d,c,fx,o in data),
            oracle_margin_hits=sum(o for d,c,fx,o in data),
            mean_closest_wrong_lower=float(np.mean([d["closest_wrong_lower"] for d,c,fx,o in data])))
    qps = [r["result"] for e in episodes for r in e["design"]["qp_records"]]
    lps = [r["result"] for e in episodes for r in e["evaluation"]]
    return dict(methods=metrics,episodes=len(episodes),qp_count=len(qps),lp_count=len(lps),
        all_qps_certified=all(q["status"]=="certified_interval" for q in qps),
        max_qp_gap=max(q.get("gap",float("inf")) for q in qps),
        all_lps_certified=all(q.get("exact_primal_verified") and q.get("exact_box_corrected_dual_verified") for q in lps),
        max_lp_gap=max(q.get("exact_gap",float("inf")) for q in lps),
        truth_never_excluded=not any(d["truth_excluded"] for e in episodes for d in e["summary"]["actionstats"].values()),
        correct_local_references=sum(e["summary"]["reference_correct"] for e in episodes),
        local_unique_ranking=sum(e["design"]["local_selection"]["unique_resolved"] for e in episodes),
        committee_unique_ranking=sum(e["design"]["committee_selection"]["unique_resolved"] for e in episodes))


def run(mode,output):
    output.mkdir(parents=True,exist_ok=True)
    config=protocol()
    if mode=="pilot":
        cases=[("cherry12","symmetric",601),("cherry23","asymmetric",602)]
        dump(output/"pilot_protocol.json",config)
    else:
        path=output/"frozen_protocol.json"
        if path.exists():
            raise RuntimeError("refusing to overwrite an existing frozen protocol")
        dump(path,config)
        cases=[(h,p,s) for h in NAMES for p in PROFILES for s in FORMAL_SEEDS]
    episodes=[]
    for k,(h,profile,seed) in enumerate(cases):
        name=f"{mode}_{h}_{profile}_{seed}"
        ep=evaluate_actions(h,PROFILES[profile],seed,output=output,prefix=name)
        ep.update(name=name,topology=h,profile=profile,seed=seed)
        dump(output/f"{name}_results.json",ep)
        episodes.append(ep)
        print(json.dumps(dict(done=k+1,total=len(cases),case=name,summary=ep["summary"]["selected"])),flush=True)
    report=dict(protocol=config,episodes=episodes,summary=aggregate(episodes),source_hash_unchanged=sha(__file__)==config["source_hash"])
    dump(output/f"{mode}_results.json",report)
    print(json.dumps(report["summary"],indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("pilot","formal"),default="pilot")
    parser.add_argument("--output",type=Path,default=Path(__file__).with_name("results_v7"))
    args=parser.parse_args()
    run(args.mode,args.output)
