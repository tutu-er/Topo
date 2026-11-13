"""Training-only frozen-mean corrections; historical validation diagnostics.

Never opens simulator truth or new-day observations. Corrected means are
planning auxiliaries: the native H bank and final MSE selection are unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter
import traceback
import numpy as np

from ... import io
from ...history import freeze_history
from ..unified_posterior_dr_20261112.history_scan import load_request
from .catalog import resources
from .solver import target_set

METHODS=('nominal','constant','ridge_0.01','ridge_0.1','ridge_1','svd_0.05')


def fitted_means(H,X,Y):
    """Return method -> (K,intercept,receipt), fitted only to training arrays."""
    H,X,Y=map(lambda a:np.asarray(a,float),(H,X,Y))
    anchor=X.mean(0);Xc=X-anchor
    residual=Y[None]-np.einsum('td,knd->ktn',X,H)
    average=residual.mean(1);centered=residual-average[:,None]
    U,s,Vt=np.linalg.svd(Xc,full_matrices=False)
    results={'nominal':(H.copy(),np.zeros_like(average),dict()),
        'constant':(H.copy(),average.copy(),dict())}
    for rho in (.01,.1,1.):
        lam=rho*s[0]**2
        inverse=(Vt.T*(s/(s*s+lam)))@U.T
        B=np.einsum('dt,ktn->knd',inverse,centered)
        results[f'ridge_{rho:g}']=(H+B,average-np.einsum('knd,d->kn',B,anchor),
            dict(ridge_fraction=rho,ridge_penalty=float(lam)))
    keep=s/s[0]>=.05
    inverse=(Vt[keep].T/s[keep])@U[:,keep].T
    B=np.einsum('dt,ktn->knd',inverse,centered)
    results['svd_0.05']=(H+B,average-np.einsum('knd,d->kn',B,anchor),
        dict(retained_directions=int(keep.sum()),singular_ratio_cutoff=.05))
    return results,dict(training_anchor=anchor,training_residual_mean=average,
        singular_values=s,singular_ratios=s/s[0],condition_number=float(s[0]/s[-1]))


def evaluate(H,K,offset,X,Y,sigma2,pairs):
    """Expected pair evidence vs observed evidence on held-out validation."""
    records=[]
    nominal=np.einsum('td,knd->ktn',X,H)
    corrected=np.einsum('td,knd->ktn',X,K)+offset[:,None]
    loss=np.sum((Y[None]-nominal)**2,axis=(1,2))
    for i,j in pairs:
        d=nominal[i]-nominal[j]
        J=float(np.sum(d*d)/(2*sigma2))
        actual=float((loss[j]-loss[i])/(2*sigma2))
        predicted=J+float(np.sum((corrected[i]-nominal[i])*d)/sigma2)
        # Independent identity: expected contrast for frozen scoring under K.
        expanded=float((np.sum((corrected[i]-nominal[j])**2)-np.sum((corrected[i]-nominal[i])**2))/(2*sigma2))
        if not np.isclose(expanded,predicted,rtol=1e-9,atol=1e-8):
            raise ArithmeticError('Expected frozen-score identity failed')
        newdistance=float(np.sum((corrected[i]-corrected[j])**2)/(2*sigma2))
        records.append(dict(i=i,j=j,observed_logodds_increment=actual,predicted_mean_increment=predicted,
            error=predicted-actual,absolute_error=abs(predicted-actual),
            falsely_positive=bool(predicted>1e-8 and actual< -1e-8),
            wrong_sign=bool(predicted*actual< -1e-16),
            original_distance=J,corrected_mean_distance=newdistance,
            mean_distance_ratio=None if J<1e-20 else newdistance/J))
    a=lambda f:np.array([r[f] for r in records],float)
    return records,dict(pairs=len(records),mean_absolute_error=float(a('absolute_error').mean()),
        root_mean_squared_error=float(np.sqrt(np.mean(a('error')**2))),
        mean_error=float(a('error').mean()),falsely_positive=int(a('falsely_positive').sum()),
        wrong_sign=int(a('wrong_sign').sum()),
        median_mean_distance_ratio=float(np.median([r['mean_distance_ratio'] for r in records if r['mean_distance_ratio'] is not None])),
        validation_prediction_mse=float(np.mean((Y[None]-corrected)**2)))


def pq_constraints(entry,X):
    n=X.shape[1]//2;p,q=X[:,:n],X[:,n:]
    k=np.sum(p*q,0)/np.sum(p*p,0)
    e=q-p*k;rmse=np.sqrt(np.mean(e*e,0))
    case=resources(entry,scale=2.)
    mapping=np.zeros((n,len(case.resources)))
    injection=np.zeros((2*n,len(case.resources)))
    for r,node in enumerate(case.node):
        mapping[node,r]=case.qratio[r]-k[node]
        injection[node,r]=1.;injection[n+node,r]=case.qratio[r]
    rank=int(np.linalg.matrix_rank(mapping,tol=1e-10))
    _,_,Vt=np.linalg.svd(mapping,full_matrices=True)
    kernel=Vt[rank:].T
    return dict(node_slopes=k,training_Q_minus_kP_RMSE_pu=rmse,
        per_resource_Q_minus_kP_per_unit_action=[float(case.qratio[r]-k[node]) for r,node in enumerate(case.node)],
        resources=len(case.resources),resource_node_count=len(set(case.node)),
        exact_consistency_rank=rank,exact_consistency_linear_nullity=len(case.resources)-rank,
        exact_consistency_injection_dimensions=int(np.linalg.matrix_rank(injection@kernel,tol=1e-10)),
        individually_exact_resources=int(np.sum(np.abs([case.qratio[r]-k[node] for r,node in enumerate(case.node)])<1e-8)),
        per_resource=[dict(kind=r['kind'],node_label=entry['labels'][int(case.node[j])],
            qratio=float(case.qratio[j]),historical_kappa=float(k[case.node[j]])) for j,r in enumerate(case.resources)],
        scope='Nullity concerns one-period linear equalities only; bounds, EV energy and thermal dynamics may further reduce feasible actions. This is not a procurement solve.')


def one(entry,split):
    began=perf_counter()
    if entry['pool']!='large96':raise ValueError('Protocol supports the existing large96 historical split only')
    native,train,valid=load_request(entry['request'])
    reference=entry['score']
    if io.ref(reference['path'])['sha256']!=reference['sha256']:raise ValueError('Score source drift')
    score=io.read(reference['path'])
    frozen=freeze_history(native,valid);H=frozen.H
    X=np.c_[train.p,train.q];V=np.c_[valid.p,valid.q]
    sigma2=float(score['sigma2'])
    fits,training=fitted_means(H,X,train.y)
    pairs=[(i,j) for i in range(len(H)) for j in range(len(H)) if frozen.topology_ids[i]!=frozen.topology_ids[j]]
    rep_targets,_=target_set([p['history_sse'] for p in score['parameters']],sigma2,.99)
    raw_targets={int(score['parameters'][i]['index']) for i in rep_targets}
    target_pairs=[pair for pair in pairs if pair[0] in raw_targets]
    results={};models={}
    for method,(K,b,receipt) in fits.items():
        allpairs,full=evaluate(H,K,b,V,valid.y,sigma2,pairs)
        _,target=evaluate(H,K,b,V,valid.y,sigma2,target_pairs)
        results[method]=dict(all_pairs=full,target_pairs=target,pairs=allpairs,fit=receipt)
        models[method]=dict(K=K,intercept=b)
    return dict(id=entry['id'],case=entry['case'],partition=split,train_points=len(X),validation_points=len(V),
        native_points=len(H),target_points=sorted(raw_targets),sigma2=sigma2,source_score=reference,
        observed_source=entry['request']['observed'],training=training,pq=pq_constraints(entry,X),
        results=results,planning_models=models,truth_read=False,new_day_observations_read=False,
        seconds=perf_counter()-began)


def aggregate(rows,scope='all_pairs'):
    out={}
    baseline={r['id']:r['results']['nominal'][scope]['mean_absolute_error'] for r in rows}
    for method in METHODS:
        data=[r['results'][method][scope] for r in rows]
        out[method]=dict(histories=len(rows),pairs=sum(d['pairs'] for d in data),
            equal_history_mean_absolute_error=float(np.mean([d['mean_absolute_error'] for d in data])),
            median_history_absolute_error=float(np.median([d['mean_absolute_error'] for d in data])),
            equal_history_root_mean_squared_error=float(np.mean([d['root_mean_squared_error'] for d in data])),
            histories_improved=sum(r['results'][method][scope]['mean_absolute_error']<baseline[r['id']]-1e-9 for r in rows),
            histories_worsened=sum(r['results'][method][scope]['mean_absolute_error']>baseline[r['id']]+1e-9 for r in rows),
            false_positive_pairs=sum(d['falsely_positive'] for d in data),wrong_sign_pairs=sum(d['wrong_sign'] for d in data),
            median_history_mean_distance_ratio=float(np.median([d['median_mean_distance_ratio'] for d in data])),
            mean_validation_prediction_mse=float(np.mean([d['validation_prediction_mse'] for d in data])))
    return out


def run(development,inventory,destination):
    if destination.exists():raise FileExistsError('Use a new private diagnostic folder')
    destination.mkdir(parents=True)
    (destination/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    dev=io.read(development)
    if len(dev)!=32 or any(e['history_partition']!='development' for e in dev):
        raise ValueError('Use the original fixed 32 development histories')
    post=[e for e in io.read(inventory) if e['id']=='pressure__input043__101']
    io.write(destination/'protocol.json',dict(source=io.ref(__file__),source_snapshot=io.ref(destination/'source_snapshot.py'),development=io.ref(development),inventory=io.ref(inventory),
        methods=METHODS,main_ridge_fraction=.1,sensitivity_only=[.01,1.],svd_cutoff=.05,
        training='original first 64 observed rows only',evaluation='original 32 validation rows; no future observations or truth',
        pairs='All ordered different-topology native parameter pairs, plus original 99%-mass representative targets versus all native competitors.',
        summary='Equal weight to each development history, preserve per-pair errors and failures.',
        posthoc_only_ids=[e['id'] for e in post],policy_changes=0,new_AC_days=0,optimization_solves=0))
    rows=[];failures=[]
    for split,entries in (('development',dev),('posthoc_only',post)):
        for entry in entries:
            try:
                result=one(entry,split);rows.append(result)
                io.write(destination/'private'/(entry['id']+'.json'),result)
            except Exception as exc:
                failures.append(dict(id=entry['id'],partition=split,error=repr(exc),traceback=traceback.format_exc()))
    good=[r for r in rows if r['partition']=='development']
    summary=dict(planned_development=len(dev),completed_development=len(good),failures=failures,
        all_pairs=aggregate(good),target_pairs=aggregate(good,'target_pairs'),
        condition_numbers=[r['training']['condition_number'] for r in good],
        pq_exact_consistency_nullities=[r['pq']['exact_consistency_linear_nullity'] for r in good],
        pq_exact_consistency_injection_dimensions=[r['pq']['exact_consistency_injection_dimensions'] for r in good],
        pq_individually_exact_resources=[r['pq']['individually_exact_resources'] for r in good],
        posthoc_only={r['id']:{m:{k:v for k,v in r['results'][m].items() if k!='pairs'} for m in METHODS} for r in rows if r['partition']=='posthoc_only'},
        protocol=io.ref(destination/'protocol.json'))
    io.write(destination/'summary.json',summary);io.write(destination/'complete.json',dict(summary=io.ref(destination/'summary.json')))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development',type=Path,default=Path('outputs/astra_inventory_20261114/development32.json'))
    p.add_argument('--inventory',type=Path,default=Path('outputs/astra_inventory_20261114/mainline400.json'))
    p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();s=run(a.development.resolve(),a.inventory.resolve(),a.destination.resolve())
    print('completed',s['completed_development'],'failed',len(s['failures']))
    for m,r in s['all_pairs'].items():print(m,r)
