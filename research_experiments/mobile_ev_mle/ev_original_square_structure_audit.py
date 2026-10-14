"""Read-only check of shared-control-square compression on saved witnesses.

It does not solve a MILP, fit AC, choose controls, or modify the frozen proxy.
All arrays are recomputed directly from saved c/B/A0/At and affine witnesses.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .information_fit_recovery import read_json,write_json


def gamma(n):
    e=np.finfo(float).eps/2
    return n*e/(1-n*e)


def audit_witness(model,witness,actions):
    a0=np.asarray(model['A0'],float);at=np.asarray(model['At'],float)
    c=np.asarray(model['c'],float);B=np.asarray(model['B'],float);dom=model['domain']
    base=np.asarray(dom['base_center'],float);S=np.asarray(dom['shift'],float)
    intercept=np.asarray(witness['physical_intercept'],float);T=np.asarray(witness['physical_slope'],float)
    K=B.shape[1];times=np.asarray(model['control_time_site'])[:,0]
    ubar=np.maximum(abs(np.asarray(dom['action_lower'])),abs(np.asarray(dom['action_upper'])))
    z=intercept-base;a=c+a0@z;C=B+np.einsum('kmn,n->mk',at,z)
    G=C.T@C;overlap=abs(C).T@abs(C)
    cross=[(i,j) for i in range(K) for j in range(i+1,K) if times[i]!=times[j]]
    same=[(i,j) for i in range(K) for j in range(i+1,K) if times[i]==times[j]]
    slope_error=float(np.max(abs(T-S)));m,n=a0.shape
    kappa=float(a@a);linear=2*C.T@a;d=np.diag(G).copy()
    # Standard floating-point dot-product bounds; input arrays are the frozen
    # proxy coefficients, whose modelling/finite-difference error is separate.
    ez=np.finfo(float).eps*(abs(intercept)+abs(base))
    formgamma=gamma(2*n+8)
    ea=abs(a0)@ez+formgamma*(abs(c)+abs(a0)@abs(z))
    eC=np.einsum('kmn,n->mk',abs(at),ez)+formgamma*(abs(B)+np.einsum('kmn,n->mk',abs(at),abs(z)))
    e=ea+eC@ubar;M=abs(a)+abs(C)@ubar
    expansion=float(2*M@e+e@e)
    dotgamma=gamma(2*m+8)
    dG=dotgamma*overlap
    coefficient=float(dotgamma*(abs(a)@abs(a)+2*(abs(C).T@abs(a))@ubar+(np.sum(C*C,axis=0)*ubar**2).sum()))
    crossed=float(sum(2*(abs(G[i,j])+dG[i,j])*ubar[i]*ubar[j] for i,j in cross))
    error_guard=expansion+coefficient+crossed
    checks=[]
    for raw in actions:
        u=np.asarray(raw,float).ravel()
        if u.shape!=(K,):raise ValueError('saved control dimension')
        zactual=intercept+T@u-base-S@u
        residual=c+B@u+(a0+np.einsum('k,kmn->mn',u,at))@zactual
        square=float(residual@residual);diagonal=float(kappa+linear@u+d@(u*u))
        same_term=float(sum(2*G[i,j]*u[i]*u[j] for i,j in same))
        checks.append(dict(full_square=square,diagonal_square=diagonal,
            difference=square-diagonal,same_time_cross_contribution=same_term,
            largest_same_time_product=float(max((abs(u[i]*u[j]) for i,j in same),default=0)),
            guarded_diagonal_upper=diagonal+error_guard,
            guarded_diagonal_dominates=bool(square<=diagonal+error_guard+abs(same_term)+1e-9)))
    step=(np.asarray(dom['action_upper'])-np.asarray(dom['action_lower']))/4
    return dict(name=model['name'],witness_origin=witness['origin'],residual_rows=m,physical_columns=n,controls=K,
        affine_slope_equals_shift=bool(np.array_equal(T,S)),max_affine_slope_error=slope_error,
        control_history_column_max=float(np.max(abs(C[:model['history_rows']]),initial=0.)),
        cross_time_gram_max=float(max((abs(G[i,j]) for i,j in cross),default=0)),
        cross_time_absolute_overlap_max=float(max((overlap[i,j] for i,j in cross),default=0)),
        same_time_gram_max=float(max((abs(G[i,j]) for i,j in same),default=0)),
        diagonal_min=float(d.min()),diagonal_max=float(d.max()),kappa=kappa,
        squared_cost_numerical_upward_guard=error_guard,
        numerical_guard_parts=dict(expansion=expansion,dot_coefficients=coefficient,cross_time=crossed),
        four_segment_chord_squared_cost_error_upper=float(d@(step**2)/4),
        saved_action_checks=checks,
        globally_box_feasible=bool(witness['validation']['valid']))


def audit_directory(directory,output=None):
    directory=Path(directory);freeze=read_json(directory/'freeze.json');records=[];identities=[];rounds=[]
    for path in sorted(directory.glob('trial_*.json.gz')):
        trial=read_json(path)
        if trial['source_sha256']!=freeze['source_sha256']:raise ValueError('frozen source map mismatch')
        identities.append(dict(seed=trial['seed'],path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for method in ('charge_fixed','v2g_fixed','charge_mobile','v2g_mobile'):
            detail=trial['selection']['methods'][method];contexts=[]
            if detail.get('before_recenter') is not None:contexts.append(('before_recenter',trial['selection']['context_models'],detail['before_recenter']))
            contexts.append(('final',detail.get('recenter_models') or trial['selection']['context_models'],detail['raw']))
            for label,models,exchange in contexts:
                if len(models)!=12 or len(exchange['witnesses'])!=12:raise ValueError('required pair denominator')
                actions=[x['u'] for x in exchange['evaluations']]
                rounds.append(dict(seed=trial['seed'],method=method,round=label,pairs=12,witnesses=sum(map(len,exchange['witnesses'])),
                    previous_residual_square_binaries=[x['master'].get('n_residual_binary') for x in exchange['iterations']],
                    proposed_control_square_segment_binaries_at_most=4*models[0]['control_shape'][0]*models[0]['control_shape'][1]))
                for model,witnesses in zip(models,exchange['witnesses']):
                    for witness in witnesses:
                        row=audit_witness(model,witness,actions);row.update(seed=trial['seed'],method=method,round=label);records.append(row)
    checks=[a for r in records for a in r['saved_action_checks']]
    result=dict(success=all(r['affine_slope_equals_shift'] and r['cross_time_absolute_overlap_max']==0 and r['control_history_column_max']==0 and r['diagonal_min']>=0
        and r['globally_box_feasible'] for r in records) and all(a['guarded_diagonal_dominates'] and a['largest_same_time_product']==0 for a in checks),
        independent_pilot_histories=len(identities),exchange_rounds=len(rounds),witness_count=len(records),saved_witness_action_comparisons=len(checks),
        source_sha256=freeze['source_sha256'],trial_hashes=identities,
        maximum_cross_time_gram=max(r['cross_time_gram_max'] for r in records),
        maximum_cross_time_absolute_overlap=max(r['cross_time_absolute_overlap_max'] for r in records),
        maximum_same_time_gram=max(r['same_time_gram_max'] for r in records),
        maximum_history_control_column=max(r['control_history_column_max'] for r in records),
        maximum_same_time_action_product=max(a['largest_same_time_product'] for a in checks),
        maximum_abs_saved_square_difference=max(abs(a['difference']) for a in checks),
        maximum_numerical_upward_guard=max(r['squared_cost_numerical_upward_guard'] for r in records),
        rounds=rounds,witnesses=records,
        scope='read-only exact-algebra/floating-arithmetic audit of saved affine witnesses; no new MILP, AC fit, selector, or global nonlinear AC bound',
        reviewer=dict(file=Path(__file__).name,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),science_source=False))
    if output:write_json(output,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True,type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    result=audit_directory(a.directory,a.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','trial_hashes','rounds','witnesses')}))


if __name__=='__main__':main()
