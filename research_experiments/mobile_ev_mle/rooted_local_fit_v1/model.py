"""Observed-only interfaces, common convex line domain, and full AC blocks."""
from __future__ import annotations

import numpy as np
from .. import likelihood as lk
from ..physical_local_v1.assets import build_network
from ..physical_local_v1.experiment import trajectory, simulate
from ..physical_local_v1.context import local_refinements
from ..reference_benchmark.representation import encode_physical_tree
from ..local_rx_identifiability.cases import LOCAL_SELECTION, candidate_bank


def public_domain(old=False):
    inventory = [build_network(n, p) for n in ("paper15", "paper11")
                 for p in ("cable_v1", "mixed_v1")]
    cap_ohm = 2*max(e["r_ohm"] for net in inventory for e in net["branches"])
    return dict(r_cap=cap_ohm/.8, r_cap_ohm=cap_ohm,
                ratio_min=.25 if old else .05, ratio_max=1.5 if old else 2.,
                power_cap=.2, v_min=.9, v_max=1.1, root_min=.97, root_max=1.05,
                base_kw=200., base_kv=.4, impedance_lower_bound=0.,
                provenance="twice maximum R over both complete public layouts and both catalog profiles")


def prepare_case(name, seed, count, context="FULL"):
    net = build_network(name, "cable_v1"); data = trajectory(net, seed)
    state = simulate(net, data); selected = LOCAL_SELECTION[name]
    all_ids = net["observation_bus_ids"]; focus = [all_ids.index(b) for b in selected["terminals"]]
    encoded = encode_physical_tree(net["branches"], all_ids, net["root_bus"], .4, 200., net["source_bus"])
    observed = {k:data[{"p":"p_kw", "q":"q_kvar"}.get(k, "root_v")]
                for k in ("p", "q")}
    observed["v"] = state["v"]
    observed = {k:(np.asarray(v)+data["sigma"][k]*data["white"][k])/(200. if k in ("p", "q") else 1.)
                for k,v in observed.items()}
    sigma = {k:np.asarray(data["sigma"][k])/(200. if k in ("p", "q") else 1.) for k in ("p", "q", "v")}
    root_meter = data["root_v"]+.0005*data["white"]["root"]
    if context == "CUT":
        bank = candidate_bank()
        bank = tuple(lk.RadialCandidate.from_supports("base" if i == 0 else c.name, 3, c.supports) for i,c in enumerate(bank))
        observed = {k:v[:,focus] for k,v in observed.items()}; sigma = {k:v[focus] for k,v in sigma.items()}
        rows = [e for e in net["branches"] if e["from_bus"] in {selected["root"],selected["hidden"],*selected["terminals"]}
                and e["to_bus"] in {selected["root"],selected["hidden"],*selected["terminals"]}]
        cut = encode_physical_tree(rows, selected["terminals"], selected["root"], .4, 200., selected["hidden"])
        rt,xt = cut.r,cut.x; true_p=data["p_kw"][:,focus]/200.; true_q=data["q_kvar"][:,focus]/200.
        true_root = abs(state["voltage_by_bus"][selected["root"]]); true_v=state["v"][:,focus]
        root_meter=None; zeros=[]; positions=list(range(3)); path=list(range(cut.candidate.m))
    else:
        # Only this structural template is given; no line values enter payload.
        base = lk.RadialCandidate.from_supports("base", len(all_ids), encoded.candidate.supports)
        bank = local_refinements(base, focus); rt,xt=encoded.r,encoded.x
        true_p=data["p_kw"]/200.; true_q=data["q_kvar"]/200.; true_root=data["root_v"]; true_v=state["v"]
        zeros=[tuple(range(len(all_ids)))]; positions=focus
        path=[i for i,s in enumerate(base.supports) if any(j in s for j in focus) and tuple(s) not in zeros]
    mask = positions if context == "FOCUS-V" else list(range(bank[0].n))
    payload=dict(observed={k:v[:count].copy() for k,v in observed.items()}, sigma=sigma,
        root_meter=None if root_meter is None else (root_meter[:count].copy(),.0005),
        voltage_positions=mask, candidates=bank, fixed_zero_supports=zeros,
        domain=public_domain(context == "OLD-RATIO"), history_rows=count,
        context=context, focus_positions=positions, truth_used=False)
    hold=dict(payload, observed={k:v[672:768].copy() for k,v in observed.items()},
        root_meter=None if root_meter is None else (root_meter[672:768].copy(),.0005), history_rows=96)
    truth=dict(r=rt,x=xt,p=true_p,q=true_q,root_v=true_root,v=true_v,
               path_edges=path, focus_positions=positions, name=name, seed=seed,
               actual_candidate="base", physical_network=net)
    return payload,hold,truth


def line_map(candidate, payload):
    active=np.array([i for i,s in enumerate(candidate.supports)
                     if tuple(s) not in map(tuple,payload["fixed_zero_supports"])],int)
    m,d=candidate.m,len(active);domain=payload["domain"]
    out=np.zeros((2*m,2*d)); cap=domain["r_cap"]
    out[active,np.arange(d)]=cap;out[active,d+np.arange(d)]=cap
    out[m+active,np.arange(d)]=cap*domain["ratio_min"]
    out[m+active,d+np.arange(d)]=cap*domain["ratio_max"]
    return out,active


def project(point):
    point=np.asarray(point,float);d=len(point)//2
    a,b=point[:d],point[d:];pa,pb=np.maximum(a,0),np.maximum(b,0)
    outside=pa+pb>1
    pa[outside]=np.clip((a[outside]-b[outside]+1)/2,0,1)
    pb[outside]=1-pa[outside]
    return np.r_[pa,pb]


def linear_support(gradient):
    d=len(gradient)//2
    return float(np.minimum(0,np.minimum(gradient[:d],gradient[d:])).sum())


def scales(payload):
    return {k:np.broadcast_to(np.asarray(payload["sigma"][k]),payload["observed"][k].shape).copy()
            for k in ("p","q","v")}


def evaluate(candidate,payload,theta,local,*,derivatives=True,fixed_lines=None):
    obs=payload["observed"];sig=scales(payload);n=candidate.n;t=len(local)
    transform,active=line_map(candidate,payload)
    r,x=np.split(transform@theta,2) if fixed_lines is None else tuple(map(np.asarray,fixed_lines))
    p=obs["p"]+sig["p"]*local[:,:n];q=obs["q"]+sig["q"]*local[:,n:2*n]
    root=1+.01*local[:,-1];physical=lk.ac_predict(candidate,r,x,p,q,root_v=root)
    vpos=np.asarray(payload["voltage_positions"],int)
    pieces=[local[:,:2*n],(physical["v"][:,vpos]-obs["v"][:,vpos])/sig["v"][:,vpos]]
    if payload["root_meter"] is not None:
        b,bs=payload["root_meter"];pieces.append(((root-np.asarray(b))/bs)[:,None])
    residual=np.c_[*pieces]
    result=dict(residual=residual,p=p,q=q,root_v=root,physical=physical,r=r,x=x)
    if not derivatives:return result
    dv=lk._response_derivatives(candidate,physical,False)["v"];m=candidate.m
    shared=np.zeros((t,residual.shape[1],0 if fixed_lines is not None else transform.shape[1]))
    if fixed_lines is None:shared[:,2*n:2*n+len(vpos)]=(dv[:,vpos,:2*m]@transform)/sig["v"][:,vpos,None]
    c=np.zeros((t,residual.shape[1],2*n+1));c[:,:2*n,:2*n]=np.eye(2*n)
    c[:,2*n:2*n+len(vpos),:n]=dv[:,vpos,2*m:2*m+n]*sig["p"][:,None,:]/sig["v"][:,vpos,None]
    c[:,2*n:2*n+len(vpos),n:2*n]=dv[:,vpos,2*m+n:]*sig["q"][:,None,:]/sig["v"][:,vpos,None]
    db=(physical["v"]-2*np.einsum('tnj,tj->tn',dv[:,:,2*m:],np.c_[p,q]))/root[:,None]
    c[:,2*n:2*n+len(vpos),-1]=.01*db[:,vpos]/sig["v"][:,vpos]
    if payload["root_meter"] is not None:c[:,-1,-1]=.01/payload["root_meter"][1]
    return dict(result,shared=shared,state=c)
