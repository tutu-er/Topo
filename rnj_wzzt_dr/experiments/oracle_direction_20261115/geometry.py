"""Quadratic expansions for an explicitly oracle, frozen-matrix comparison."""
from __future__ import annotations
import numpy as np


def expand(response,forecast,control_map,hours,repeat,sigma2):
    response,forecast,B=map(lambda x:np.asarray(x,float),(response,forecast,control_map))
    base=forecast@response.T
    effect=response@B
    q=np.broadcast_to(repeat*(effect.T@effect)/(2*sigma2),(hours,B.shape[1],B.shape[1])).copy()
    a=(base@effect/sigma2).reshape(hours,repeat,B.shape[1]).sum(axis=1)
    return q,a,float(np.sum(base*base)/(2*sigma2))


def build(case,score,forecast,Hstar,target,competitors,base_kw):
    H=np.asarray([r['H'] for r in score['native_bank']],float)
    sigma2=float(score['sigma2']);B=case.response_effect(np.eye(2*case.N),base_kw)
    if (len(set(competitors))!=len(competitors) or not competitors or target in competitors
            or Hstar.shape!=H[target].shape):raise ValueError('Aligned nonempty foreign candidate bank required')
    E=Hstar-H[target]
    eq,ea,ec=expand(E,forecast,B,case.H,case.repeat,sigma2)
    D=H[target][None]-H[np.asarray(competitors)]
    parts=[expand(d,forecast,B,case.H,case.repeat,sigma2) for d in D]
    jq,ja,jc=map(np.asarray,zip(*parts))
    scales=np.maximum(1.,jc)
    # Valid outer bound on each full predicted difference; energy constraints dropped.
    amplitude=np.maximum(abs(case.lower),abs(case.upper))
    base=np.einsum('tc,knc->ktn',forecast,D)
    effect=np.einsum('knc,cr->knr',D,B)
    radius=np.repeat(np.einsum('knr,hr->khn',abs(effect),amplitude),case.repeat,axis=1)
    jupper=np.sum((abs(base)+radius)**2,axis=(1,2))/(2*sigma2)
    zupper=max(0.,float(np.min((jupper-jc)/scales)))
    return dict(J_q=jq,J_a=ja,J_c=jc,e_q=eq,e_a=ea,e_c=ec,j_scales=scales,
        z_box_upper=zupper,J_box_upper=jupper,Hstar=Hstar,Hc=H[target],D=D,
        forecast=forecast,control_map=B,sigma2=sigma2,base_kw=float(base_kw),target_native=int(target),
        competitor_native_indices=list(map(int,competitors)))


def direct(case,geometry,u,*,mode):
    X=geometry['forecast']+case.shift_pq_pu(u,geometry['base_kw'])
    E=geometry['Hstar']-geometry['Hc']
    error=float(np.sum((X@E.T)**2)/(2*geometry['sigma2']))
    distance=np.sum(np.einsum('tc,knc->ktn',X,geometry['D'])**2,axis=(1,2))/(2*geometry['sigma2'])
    z_j=(distance-geometry['J_c'])/geometry['j_scales']
    z_distance=float(z_j.min());relative_error=None if geometry['e_c']==0 else 1-error/geometry['e_c']
    z=z_distance if mode=='distance_only' or relative_error is None else min(z_distance,relative_error)
    return dict(J=distance.tolist(),J0=geometry['J_c'].tolist(),e=error,e0=float(geometry['e_c']),
        relative_error_reduction=relative_error,relative_distance_gain_by_competitor=z_j.tolist(),
        direct_z=float(z),distance_only_z=z_distance,
        error_improvement_is_strict=bool(relative_error is not None and relative_error>1e-6),
        all_distances_increased=bool(z_distance>1e-6))
