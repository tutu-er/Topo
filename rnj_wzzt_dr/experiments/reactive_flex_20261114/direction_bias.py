"""Signed voltage and action-response errors from completed saved executions.

No fitting, optimizer, AC calculation, or physical-network parameter is used.
The clean saved root and terminal voltage are the offline comparison reference.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .placement import checked, read, ref


def measured(path):
    with np.load(path, allow_pickle=False) as z:
        return np.c_[z['true_P'], z['true_Q']], z['true_V'].copy(), z['true_root'].copy()


def predicted_voltage(X, H, root):
    Y = X@H.T
    squared = root[:, None]**2-Y
    if np.any(squared <= 0):
        raise ArithmeticError('Frozen prediction gives nonpositive squared voltage')
    return Y, np.sqrt(squared)


def stats(predicted, actual):
    p, a = np.asarray(predicted).ravel(), np.asarray(actual).ravel()
    if not len(a):
        return dict(cells=0)
    e = p-a
    pp, aa, pa = float(p@p), float(a@a), float(p@a)
    cosine = float(np.clip(pa/np.sqrt(pp*aa), -1., 1.)) if pp*aa > 1e-40 else None
    pc, ac = p-p.mean(), a-a.mean()
    cp, ca = float(pc@pc), float(ac@ac)
    relevant = np.maximum(np.abs(p), np.abs(a)) > 1e-12
    return dict(cells=len(a), predicted_mean=float(p.mean()), actual_mean=float(a.mean()),
        signed_error_mean=float(e.mean()), mean_absolute_error=float(np.abs(e).mean()),
        RMS_error=float(np.sqrt(e@e/len(e))), minimum_error=float(e.min()), maximum_error=float(e.max()),
        relative_error=float(np.linalg.norm(e)/np.sqrt(aa)) if aa > 1e-28 else None,
        amplitude_k_actual_on_predicted=pa/pp if pp > 1e-28 else None,
        cosine=cosine, angle_degrees=float(np.degrees(np.arccos(cosine))) if cosine is not None else None,
        pearson_correlation=float(pc@ac/np.sqrt(cp*ca)) if cp*ca > 1e-40 else None,
        active_comparison_cells=int(relevant.sum()),
        opposite_direction_cells=int(((p*a < -1e-24)&relevant).sum()),
        signed_error_positive_cells=int((e > 1e-12).sum()),
        signed_error_negative_cells=int((e < -1e-12).sum()))


def run(directory, base_kw=200.):
    directory = Path(directory).resolve()
    completion = read(directory/'complete.json')
    if completion['status'] != 'complete':
        raise ValueError('Saved batch must be complete')
    entries_ref = ref(directory/'inputs.json')
    entries = read(entries_ref['path'])
    contexts, baselines, data = {}, {}, defaultdict(list)
    max_baseline_difference = max_voltage_identity_error = 0.
    references=[]
    for entry in entries:
        score_ref = checked(entry['score'])
        score=read(score_ref['path'])
        contexts[entry['id']]=dict(entry=entry,score=score,reference=score_ref,
            H={i:np.asarray(score['native_bank'][i]['H'],float) for i in (2,4)})
    for path in sorted(directory.glob('fraction*/*/forecast*/results.json')):
        for row in read(path):
            if row['status'] != 'complete':
                raise ValueError('Do not silently drop failed execution rows')
            ctx=contexts[row['id']]
            rd=path.parent/f"execution{row['repeat']:02d}"
            source=rd/row['strategy']/'observations.npz'
            zero=rd/'no_dfs'/'observations.npz'
            X0,V0,root0=measured(zero)
            X,V,root=measured(source)
            if not np.array_equal(root,root0):
                raise ValueError('Root voltage differs between paired clean executions')
            if row['id'] not in baselines:
                baselines[row['id']]=(X0,V0,root0,ref(zero))
            else:
                a,b,c,_=baselines[row['id']]
                max_baseline_difference=max(max_baseline_difference,float(np.max(np.abs(X0-a))),
                    float(np.max(np.abs(V0-b))),float(np.max(np.abs(root0-c))))
            actualY=root[:,None]**2-V**2
            actualY0=root0[:,None]**2-V0**2
            model_data={}
            for i,H in ctx['H'].items():
                Yhat,Vhat=predicted_voltage(X,H,root)
                Yhat0,Vhat0=predicted_voltage(X0,H,root0)
                residual=actualY-Yhat
                identity=residual/(Vhat+V)
                max_voltage_identity_error=max(max_voltage_identity_error,float(np.max(np.abs(identity-(Vhat-V)))))
                model_data[i]=dict(deltaYhat=Yhat-Yhat0,deltaVhat=Vhat-Vhat0)
            data[row['id'],row['strategy']].append(dict(deltaX=X-X0,deltaY=actualY-actualY0,
                deltaV=V-V0,models=model_data,source=ref(source)))
    if max_baseline_difference > 1e-13 or max_voltage_identity_error > 1e-13:
        raise ArithmeticError('Baseline or signed voltage identity check failed')
    baseline_rows=[]
    for identifier,(X,V,root,source) in baselines.items():
        ctx=contexts[identifier]; labels=ctx['score']['labels']; Y=root[:,None]**2-V**2
        models={}
        for i,H in ctx['H'].items():
            yh,vh=predicted_voltage(X,H,root)
            models[str(i)]=dict(voltage=stats(vh,V),squared_voltage_difference=stats(yh,Y),
                mean_r_Yactual_minus_Yhat=float(np.mean(Y-yh)),
                nodes=[dict(node_label=int(l),voltage=stats(vh[:,k],V[:,k]),
                    squared_voltage_difference=stats(yh[:,k],Y[:,k]),
                    mean_r_Yactual_minus_Yhat=float(np.mean(Y[:,k]-yh[:,k]))) for k,l in enumerate(labels)])
        baseline_rows.append(dict(id=identifier,source=source,score=ctx['reference'],
            actual_background_count=1,background_points=len(X),models=models))
    summaries=[]
    for (identifier,strategy), records in sorted(data.items()):
        ctx=contexts[identifier]; labels=ctx['score']['labels']; n=len(labels)
        dx=np.concatenate([r['deltaX'] for r in records]); dy=np.concatenate([r['deltaY'] for r in records]); dv=np.concatenate([r['deltaV'] for r in records])
        models={}
        for i in (2,4):
            py=np.concatenate([r['models'][i]['deltaYhat'] for r in records])
            pv=np.concatenate([r['models'][i]['deltaVhat'] for r in records])
            nodes=[]
            for k,label in enumerate(labels):
                item=dict(node_label=int(label),deltaY=stats(py[:,k],dy[:,k]),deltaV=stats(pv[:,k],dv[:,k]),
                    mean_delta_r_Yactual_minus_Yhat=float(np.mean(dy[:,k]-py[:,k])),by_local_Q_sign={})
                for name,sign in [('absorption_positive',1.),('injection_negative',-1.)]:
                    mask=sign*dx[:,n+k] > 1e-8
                    other_q=np.abs(dx[:,n:]).sum(axis=1)-np.abs(dx[:,n+k])
                    isolated=mask&(other_q < 1e-10)&(np.abs(dx[:,:n]).sum(axis=1)<1e-10)
                    item['by_local_Q_sign'][name]=dict(cells=int(mask.sum()),
                        mean_local_deltaQ_kvar=float(dx[mask,n+k].mean()*base_kw) if mask.any() else None,
                        mean_other_nodes_abs_deltaQ_kvar=float(other_q[mask].mean()*base_kw) if mask.any() else None,
                        deltaV=stats(pv[mask,k],dv[mask,k]),deltaY=stats(py[mask,k],dy[mask,k]),
                        isolated_this_node_only_cells=int(isolated.sum()),
                        isolated_deltaV=stats(pv[isolated,k],dv[isolated,k]),
                        qualification='Q sign groups describe actual joint schedules; only isolated subgroup has zero other P/Q action')
                nodes.append(item)
            models[str(i)]=dict(deltaY=stats(py,dy),deltaV=stats(pv,dv),
                mean_delta_r_Yactual_minus_Yhat=float(np.mean(dy-py)),nodes=nodes)
        summaries.append(dict(id=identifier,strategy=strategy,paths=len(records),
            clean_deltaP_norm=float(np.linalg.norm(dx[:,:n])),clean_deltaQ_norm=float(np.linalg.norm(dx[:,n:])),
            models=models,sources=[r['source'] for r in records]))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),run=str(directory),source=ref(__file__),
        inputs=entries_ref,completion=ref(directory/'complete.json'),
        conditions=dict(no_refit=True,no_optimizer=True,no_AC=True,no_physical_line_parameters_read=True,
            clean_reference='Saved true_P/Q/V/root from the already completed AC executions',
            positive_Q='Increase net reactive absorption; negative is reactive injection',
            fixed_bias='Baseline Vhat minus Vactual at the same clean power and root voltage',
            response_bias='(Vhat_after-Vhat_before)-(Vactual_after-Vactual_before)',
            k='dot(deltaYhat,deltaYactual)/dot(deltaYhat,deltaYhat); diagnostic scalar only, not a recalibration',
            r_identity='r=Yactual-Yhat=(Vhat-Vactual)*(Vhat+Vactual)',
            action_identity='delta r equals deltaYactual-deltaYhat; converting this to deltaV requires the before/after voltage denominators',
            single_background='Repeated forecasts and meter streams share one clean baseline; it is counted once per arm',
            conditional_Q_groups='Other-node Q may act concurrently; no isolated causal response is asserted except for explicitly isolated subsets'),
        maximum_baseline_replay_difference=max_baseline_difference,
        maximum_signed_voltage_identity_error=max_voltage_identity_error,
        baseline=baseline_rows,action=summaries)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.run)
    args.out.mkdir(parents=True,exist_ok=False)
    with (args.out/'direction_bias.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=ref(args.out/'direction_bias.json'),baseline_entries=len(result['baseline']),
        strategy_entries=len(result['action']),maximum_identity_error=result['maximum_signed_voltage_identity_error'])))


if __name__=='__main__':
    main()
