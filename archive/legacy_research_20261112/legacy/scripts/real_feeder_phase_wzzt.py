"""Phase-block wzzT extension, isolated from the scalar and production models.

For symmetric cables Z_phase-neutral = z1 I + zn 11T. Each physical/reduced
branch shares ONE downstream-meter mask across its four real parameters.
Theft is a stationary balanced three-phase source with known Q/P = .4.
Normal history and the current window share topology/weights. All observations
keep their measured absolute root reference. No free voltage intercept exists.
"""
from __future__ import annotations
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

try:
    from scripts import real_feeder_joint_theft as base
except ModuleNotFoundError:
    import real_feeder_joint_theft as base

ANGLES = np.array([0, -2*np.pi/3, 2*np.pi/3])
ROTATION = np.exp(1j*(ANGLES[None, :]-ANGLES[:, None]))


def phase_sample(feeder, seed, count, balanced, condition="clean", source_bus=5):
    # Same seeds, loads and bus-level meter errors as scalar v1 for paired data.
    rng = np.random.default_rng(seed); snapshot = feeder.snapshot.copy()
    if balanced:
        snapshot[:] = snapshot.sum(axis=1, keepdims=True)/3
    factor = rng.uniform(.55, 1.15, (count, len(feeder.meters), 1))
    loads = np.zeros((count, 53, 3), complex)
    loads[:, feeder.meters] = factor*snapshot[feeder.meters]
    reactive = rng.uniform(-.08, .15, factor.shape)*abs(snapshot[feeder.meters].real)
    loads[:, feeder.meters] += 1j*reactive
    root = 230+.8*np.sin(np.linspace(0, 4*np.pi, count))+rng.normal(0, .1, count)
    amplitude = np.zeros(count)
    if condition == "persistent":
        amplitude[:] = 15
    elif condition == "intermittent":
        amplitude[rng.choice(count, count//2, replace=False)] = 30
    elif condition != "clean":
        raise ValueError(condition)
    actual = loads.copy(); actual[:, source_bus] += amplitude[:, None]*(1+.4j)/3
    volt, head, loss, audit = base.ac_flow(feeder, actual, root)
    pp = loads[:, feeder.meters].real*(1+rng.normal(0, .005, (count, len(feeder.meters), 1)))
    qq = loads[:, feeder.meters].imag*(1+rng.normal(0, .005, (count, len(feeder.meters), 1)))
    mags = abs(volt[:, feeder.meters, :3]-volt[:, feeder.meters, 3:])
    mags *= 1+rng.normal(0, .0002, mags.shape)
    yp = (root[:, None, None]**2-mags**2)/2000
    p0 = head.real*(1+rng.normal(0, .002, count))
    return dict(p=pp.sum(axis=2), q=qq.sum(axis=2), y=yp.sum(axis=2),
                pp=pp, qq=qq, yp=yp, root_v=root, p0=p0, loss=loss.real,
                amplitude=amplitude, audit=audit)


def features(data, supports):
    n = data["p"].shape[1]
    z = np.array([[j in s for j in range(n)] for s in supports], float)
    fp = np.einsum("tna,en->tea", data["pp"], z)
    fq = np.einsum("tna,en->tea", data["qq"], z)
    neutral = np.einsum("ab,teb->tea", ROTATION, fp-1j*fq)
    # Re[(rn+j*xn)*neutral] = rn*Re(neutral)-xn*Im(neutral).
    edge = np.stack([fp, fq, neutral.real, -neutral.imag], axis=-1)
    matrix = np.einsum("teak,en->tnaek", edge, z).reshape(-1, len(supports)*4)
    return z, matrix


def phase_predict(data, supports, weights, amplitude=None, source=None):
    z, matrix = features(data, supports)
    weights = np.asarray(weights)
    result = (matrix @ weights.ravel()).reshape(data["yp"].shape)
    if source is not None:
        b = np.array([supports[source] <= s for s in supports])
        response = ((weights[:, 0]+.4*weights[:, 1])*b) @ z
        result += amplitude[:, None, None]/3*response[None, :, None]
    return result


def fit_phase(data, supports, amplitude=None, fixed_supports=None, seconds=45,
              atom_penalty=.00015, weight_penalty=.001):
    z, design = features(data, supports)
    y = data["yp"].ravel(); t,n,_ = data["yp"].shape; e=len(supports)
    w = np.arange(4*e).reshape(e,4); u=np.arange(4*e,5*e); s=u+e
    vr=s+e; vx=vr+e; slack=np.arange(8*e,8*e+len(y)); size=int(slack[-1]+1)
    lower=np.zeros(size); upper=np.full(size,np.inf)
    upper[np.r_[w.ravel(),vr,vx,u,s]]=1
    integrality=np.zeros(size,int); integrality[np.r_[u,s]]=1
    cost=np.zeros(size); cost[slack]=base.SCALE/len(y)
    cost[w.ravel()]=base.SCALE*weight_penalty
    for k,support in enumerate(supports):
        if len(support)==1: lower[u[k]]=1
        else: cost[u[k]]=base.SCALE*atom_penalty
        if fixed_supports is not None: lower[u[k]]=upper[u[k]]=int(k in fixed_supports)
    joint=amplitude is not None
    if amplitude is None: amplitude=np.zeros(t); upper[s]=0
    a=np.array([[h<=edge for h in supports] for edge in supports],float)
    ri=[];ci=[];val=[];lows=[];highs=[]
    def add(coef,lo=-np.inf,hi=np.inf):
        row=len(lows)
        for k,v in coef.items():
            if v: ri.append(row);ci.append(int(k));val.append(float(v))
        lows.append(lo);highs.append(hi)
    for i,j in combinations(range(e),2):
        if supports[i]&supports[j] and not (supports[i]<=supports[j] or supports[j]<=supports[i]):
            add({u[i]:1,u[j]:1},hi=1)
    add({k:1 for k in s},int(joint),int(joint))
    for k in range(e):
        add({s[k]:1,u[k]:-1},hi=0)
        for weight in w[k]: add({weight:1,u[k]:-1},hi=0)
        b={s[h]:a[k,h] for h in range(e) if a[k,h]}
        for weight,product in ((w[k,0],vr[k]),(w[k,1],vx[k])):
            add({product:1,**{j:-v for j,v in b.items()}},hi=0)
            add({product:1,weight:-1},hi=0)
            add({product:1,weight:-1,**{j:-v for j,v in b.items()}},lo=-1)
    for row in range(len(y)):
        sample=row//(n*3); terminal=(row//3)%n
        coef={k:design[row,k] for k in np.flatnonzero(design[row])}
        for k in np.flatnonzero(z[:,terminal]):
            coef[vr[k]]=amplitude[sample]/3;coef[vx[k]]=.4*amplitude[sample]/3
        add({**coef,slack[row]:-1},hi=y[row]);add({**coef,slack[row]:1},lo=y[row])
    mat=coo_matrix((val,(ri,ci)),shape=(len(lows),size)).tocsc()
    started=perf_counter()
    opt=milp(cost,integrality=integrality,bounds=Bounds(lower,upper),
             constraints=LinearConstraint(mat,lows,highs),
             options=dict(time_limit=seconds,mip_rel_gap=1e-6))
    result=dict(status="optimal" if opt.status==0 else "incomplete",solver_status=int(opt.status),
                seconds=perf_counter()-started,message=opt.message,
                mip_gap=float(opt.mip_gap) if getattr(opt,"mip_gap",None) is not None else None)
    if opt.x is None: return result
    values=opt.x;chosen=np.flatnonzero(values[s]>.5);source=int(chosen[0]) if len(chosen) else None
    active=list(map(int,np.flatnonzero(values[u]>.5)));weights=values[w]
    predicted=phase_predict(data,supports,weights,amplitude,source)
    direct=float(np.mean(abs(predicted-data["yp"])))
    penalized=direct+atom_penalty*sum(len(supports[k])>1 for k in active)+weight_penalty*float(weights.sum())
    activity=mat@values
    feasibility=max(float(np.max(np.maximum(np.asarray(lows)-activity,0))),float(np.max(np.maximum(activity-np.asarray(highs),0))),
                    float(np.max(np.maximum(lower-values,0))),float(np.max(np.maximum(values-upper,0))),
                    float(np.max(abs(values[np.r_[u,s]]-np.rint(values[np.r_[u,s]])))))
    b=a@np.rint(values[s]);product=float(max(np.max(abs(values[vr]-weights[:,0]*b)),np.max(abs(values[vx]-weights[:,1]*b))))
    objective_error=abs(penalized-float(opt.fun)/base.SCALE)
    assert feasibility<3e-6 and product<3e-6 and objective_error<3e-5
    result.update(source=source,source_support=None if source is None else sorted(supports[source]),
                  active=active,weights=weights.tolist(),train_mae=direct,
                  audit=dict(feasibility=feasibility,product_error=product,objective_error=objective_error))
    return result


def phase_loss(data,supports,fit):
    """First-order phase and neutral I^2R losses, using estimated weights only."""
    z,_=features(data,supports)
    current=np.einsum("tna,en->tea",data["pp"]-1j*data["qq"],z)*np.exp(1j*ANGLES)[None,None,:]
    weights=np.array(fit["weights"])
    losses=(abs(current)**2).sum(axis=2)@weights[:,0]+abs(current.sum(axis=2))**2@weights[:,2]
    return losses*1000/data["root_v"]**2


def combine(history,current):
    return {k:np.concatenate([history[k],current[k]]) for k in ("p","q","y","pp","qq","yp","root_v","p0")}


def evaluate(feeder,supports,fit,holdout,source_bus):
    if "weights" not in fit:return
    fit["r"]=[w[0] for w in fit["weights"]];fit["x"]=[w[1] for w in fit["weights"]]
    base.evaluate(feeder,holdout,supports,fit,holdout,source_bus)
    estimate=phase_predict(holdout,supports,fit["weights"])
    fit["heldout_phase_mae_v_approx"]=float(np.mean(abs(estimate-holdout["yp"]))*2000/(2*230))
    truth,_=base.truth_atoms(feeder);n=len(feeder.meters)
    z=np.array([[j in s for j in range(n)] for s in supports],float)
    for name,component,excitation in (("r","real",holdout["p"]),("x","imag",holdout["q"])):
        true_mat=np.zeros((n,n))
        for s,w in truth.items():
            row=np.array([j in s for j in range(n)],float);true_mat+=getattr(w,component)*np.outer(row,row)
        estimated=z.T@(np.array(fit[name])[:,None]*z)
        fit[name+"_excitation_weighted_error"]=float(np.linalg.norm(excitation@(estimated-true_mat))/np.linalg.norm(excitation@true_mat))


def main(output,seeds=2,count=64,seconds=45,source_bus=5):
    output=output.resolve();assert output.is_relative_to(base.ROOT);output.mkdir(parents=True,exist_ok=True)
    feeder=base.read_feeder()
    protocol=dict(dataset=feeder.metadata,seeds=seeds,train_samples=count,history_samples=128,holdout_samples=96,source_bus=source_bus,
                  conditions=["clean","persistent","intermittent"],normal_history_shared_in_all_fits=True,
                  root_voltage="known at every sample; all three phase voltage references retained; no intercept",
                  amplitude="history phase/neutral I2R, scalar scale fitted only to normal head-minus-meters balance",
                  loss_scale="clip least-squares normal-data scalar to [0.2,5]; does not use theft/loss truth",
                  stationary_balanced_three_phase_theft=True,known_q_over_p=.4,
                  atom_penalty=.00015,weight_penalty=.001,conditional_source_count=1,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  base_script_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest())
    path=output/"protocol.json"
    if path.exists():assert json.loads(path.read_text())==protocol,"Changed protocol: use a new output folder"
    else:path.write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    for seed in range(seeds):
        for balanced in (True,False):
            prefix=f"seed{seed}_{'balanced' if balanced else 'unbalanced'}"
            history=phase_sample(feeder,74000+seed,128,balanced)
            holdout=phase_sample(feeder,78000+seed,96,balanced)
            hpool=base.supports_from_measurements(history)
            path=output/(prefix+"_history.json")
            if path.exists():histfit=json.loads(path.read_text())
            else:
                histfit=fit_phase(history,hpool,seconds=seconds)
                path.write_text(json.dumps(histfit,indent=2),encoding="utf-8")
            if "weights" not in histfit:print(prefix+" no historical incumbent",flush=True);continue
            hist_loss=phase_loss(history,hpool,histfit)
            hist_balance=history["p0"]-history["p"].sum(axis=1)
            scale=float(np.clip(hist_loss@hist_balance/max(hist_loss@hist_loss,1e-12),.2,5))
            for condition in protocol["conditions"]:
                path=output/(prefix+"_"+condition+".json")
                if path.exists():continue
                current=phase_sample(feeder,76000+seed,count,balanced,condition,source_bus)
                pool=base.supports_from_measurements(history,current)
                losshat=scale*phase_loss(current,hpool,histfit)
                amp=np.maximum(current["p0"]-current["p"].sum(axis=1)-losshat,0)
                combined=combine(history,current);amplitude=np.r_[np.zeros(len(history["p"])),amp]
                fits={};fits["topology_only"]=fit_phase(combined,pool,seconds=seconds)
                fixed=fits["topology_only"].get("active")
                if fixed is not None:fits["sequential"]=fit_phase(combined,pool,amplitude,fixed,seconds)
                fits["joint"]=fit_phase(combined,pool,amplitude,seconds=seconds)
                for fitted in fits.values():evaluate(feeder,pool,fitted,holdout,source_bus)
                result=dict(seed=seed,balanced=balanced,condition=condition,supports=[sorted(s) for s in pool],
                            ac_audit=current["audit"],loss_scale=scale,loss_estimate_mae_kw=float(np.mean(abs(losshat-current["loss"]))),
                            amplitude_mae_kw=float(np.mean(abs(amp-current["amplitude"]))),amplitude_mean_kw=float(amp.mean()),
                            true_amplitude_mean_kw=float(current["amplitude"].mean()),fits=fits)
                path.write_text(json.dumps(result,indent=2),encoding="utf-8")
                print(json.dumps(dict(case=prefix+"_"+condition,amplitude_mae_kw=result['amplitude_mae_kw'],fits={k:dict(status=v['status'],seconds=round(v['seconds'],2),source=v.get('source_support'),correct=v.get('source_region_correct'),r_error=v.get('r_relative_error'),holdout=v.get('heldout_phase_mae_v_approx')) for k,v in fits.items()})),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seeds",type=int,default=2);parser.add_argument("--samples",type=int,default=64)
    parser.add_argument("--seconds",type=float,default=45);parser.add_argument("--source-bus",type=int,default=5)
    args=parser.parse_args();main(args.output,args.seeds,args.samples,args.seconds,args.source_bus)
