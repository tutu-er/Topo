"""Rescore completed development data after removing each row's mean voltage.

Diagnostic criterion change only: no optimization, AC, fitting or new data.
The full native bank and original strict numerical selection tolerance stay
fixed. Meter-variance diagnostics use saved physical values offline only.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import shutil
import numpy as np

from ... import io
from ...dr.selection_mse import choose_mse
from .data import load_entry


def centered(values):
    return values-np.mean(values,axis=-1,keepdims=True)


def loss_vectors(bank_H,power,Y):
    residual=Y[None]-np.einsum('tc,knc->ktn',power,bank_H)
    raw=np.sum(residual**2,axis=(1,2))
    relative=np.sum(centered(residual)**2,axis=(1,2))
    return raw,relative


def choose(loss,counts,cells,score):
    # Keep the original strict SSE tie width. Using N-1 as a denominator
    # without also rescaling K would silently change that width.
    return choose_mse(np.asarray(loss)/cells,counts,K=1/score['reference_mse'],
        epsilon_scaled=score['validation_tie'])['selected']


def meter_variance(d,root,voltage,voltage_sigma,score_sigma2):
    """Exact conditional variance of d.Y/score_sigma2 from voltage meters."""
    energy=float(np.sum(d*d))
    if energy<=1e-30:
        return None
    a=4*voltage_sigma**2+2*voltage_sigma**4
    root_term=float(np.sum(root**4*np.sum(d,axis=1)**2))
    terminal_term=float(np.sum(voltage**4*d*d))
    exact=a*(root_term+terminal_term)/score_sigma2**2
    nominal=energy/score_sigma2
    dc=centered(d);relative_energy=float(np.sum(dc*dc))
    relative_exact=a*float(np.sum(voltage**4*dc*dc))/score_sigma2**2
    relative_nominal=relative_energy/score_sigma2
    square_mean=voltage_sigma**2*float(np.sum(d*(root[:,None]**2-voltage**2)))/score_sigma2
    return dict(exact_voltage_meter_logodds_variance=exact,declared_iid_logodds_variance=nominal,
        variance_ratio=exact/nominal,root_variance_fraction=root_term/(root_term+terminal_term),
        relative_distance_fraction=relative_energy/energy,
        projected_variance_ratio_using_old_score_scale=None if relative_nominal<=1e-30 else relative_exact/relative_nominal,
        squaring_mean_shift_in_declared_sd=square_mean/np.sqrt(nominal),
        near_unit_prediction_ratio=(a/score_sigma2)*(1+float(np.sum(np.sum(d,axis=1)**2))/energy))


def distribution(values):
    a=np.asarray([x for x in values if x is not None and np.isfinite(x)],float)
    if not len(a):
        return dict(n=0)
    q=np.quantile(a,[0,.1,.5,.9,1])
    return dict(n=len(a),minimum=float(q[0]),p10=float(q[1]),median=float(q[2]),p90=float(q[3]),
        maximum=float(q[4]),mean=float(a.mean()))


def aggregate(rows):
    result={}
    for strategy in sorted({r['strategy'] for r in rows}):
        own=[r for r in rows if r['strategy']==strategy]
        completed=sum(r.get('status','complete')=='complete' for r in own)
        result[strategy]=dict(n=len(own),complete_rows=completed,failed_or_missing_rows=len(own)-completed,
            original_correct=sum(r['original_correct'] for r in own),
            relative_correct=sum(r['relative_correct'] for r in own),
            original_wrong=completed-sum(r['original_correct'] for r in own),
            relative_wrong=completed-sum(r['relative_correct'] for r in own),
            relative_rescues=sum(r['relative_correct'] and not r['original_correct'] for r in own),
            relative_losses=sum(r['original_correct'] and not r['relative_correct'] for r in own),
            selected_topology_changed=sum(r['selected_topology_changed'] for r in own),
            meter_variance={name:distribution([r['meter'][name] for r in own if r['meter'] is not None])
                for name in ('variance_ratio','root_variance_fraction','relative_distance_fraction',
                    'projected_variance_ratio_using_old_score_scale','squaring_mean_shift_in_declared_sd')})
    return result


def secondary_lock(run_dir,path):
    if path is None:
        raise ValueError('Non-development rescoring requires an explicit prelocked secondary-metric protocol')
    lock=io.read(path)
    if lock.get('metric')!='relative_voltage_native_mse' or lock.get('primary_unchanged') is not True:
        raise ValueError('Secondary metric lock must preserve the original primary criterion')
    allowed={str(Path(p).resolve()).lower() for p in lock['allowed_runs']}
    if str(run_dir.resolve()).lower() not in allowed:
        raise ValueError('Batch absent from secondary metric lock')
    if not lock.get('created_utc') or len(lock.get('development_evidence',[]))<2:
        raise ValueError('Lock requires timestamp and the two development evidence references')
    for reference in lock['development_evidence']:
        if io.ref(reference['path'])['sha256']!=reference['sha256']:
            raise ValueError('Secondary selection development evidence changed')
    return io.ref(path)


def run(run_dir,out,lock=None):
    if out.exists():
        raise ValueError('Preserve earlier diagnostic output')
    p=io.read(run_dir/'protocol.json')
    locked=None if p['phase']=='development' else secondary_lock(run_dir,lock)
    if not (run_dir/'complete.json').exists():
        raise ValueError('Use completed development data only')
    entries=io.read(run_dir/'inputs.json');rows=[];history=[];errors=[];identities=defaultdict(float)
    for entry in entries:
        score,_,validation,truth=load_entry(entry)
        bank=score['native_bank'];H=np.asarray([v['H'] for v in bank]);counts=[v['internal'] for v in bank]
        raw_h,relative_h=loss_vectors(H,np.c_[validation.p,validation.q],validation.y)
        stored_h=np.asarray([v['history_sse'] for v in bank])
        identities['historical_sse']=max(identities['historical_sse'],float(np.max(abs(raw_h-stored_h))))
        h0=choose(raw_h,counts,score['cells'],score);hc=choose(relative_h,counts,score['cells'],score)
        history.append(dict(id=entry['id'],role=entry['role'],original_correct=bank[h0]['topology_id']==truth['truth_id'],
            relative_correct=bank[hc]['topology_id']==truth['truth_id']))
        reps=np.asarray([param['H'] for param in score['parameters']])
        target=score['map_index'];other=[j for j in range(len(reps)) if j!=target]
        runner=min(other,key=lambda j:score['parameters'][j]['history_sse']) if other else None
        for day in range(p['days']):
            d=run_dir/'days'/entry['id']/f'day{day:03d}'
            if not (d/'results.json').exists():
                expected=['no_dfs']+[name for mode in p['modes'] for name in (mode,mode+'_rotation')]
                saved=[dict(strategy=name,status='missing_task') for name in expected]
            else:
                saved=[r for r in io.read(d/'results.json') if r['physics']=='AC']
            if not any(r['status']=='complete' for r in saved):
                true_root=None
            else:
                with np.load(d/'background.npz') as z:
                    true_root=z['root'].copy()
            for r in saved:
                if r['status']!='complete':
                    errors.append(dict(id=entry['id'],day=day,strategy=r['strategy'],error='Execution incomplete'))
                    rows.append(dict(id=entry['id'],day=day,role=entry['role'],case=entry['case'],strategy=r['strategy'],
                        status=r['status'],original_correct=False,relative_correct=False,selected_topology_changed=False,meter=None))
                    continue
                with np.load(d/(r['strategy']+'_AC.npz')) as z:
                    P=np.c_[z['P'],z['Q']];Y=z['Y'].copy();V=z['V'].copy();true_V=z['true_V'].copy()
                new_raw,new_relative=loss_vectors(H,P,Y)
                cells=score['cells']+Y.size
                old=choose(raw_h+new_raw,counts,cells,score)
                rel=choose(relative_h+new_relative,counts,cells,score)
                if old!=r['native_selected_index']:
                    errors.append(dict(id=entry['id'],day=day,strategy=r['strategy'],error='Original native selection reconstruction differs'))
                identities['root_cancellation']=max(identities['root_cancellation'],float(np.max(abs(centered(Y)+centered(V*V)))))
                projected_H=H-H.mean(axis=1,keepdims=True)
                alternate=centered(Y)[None]-np.einsum('tc,knc->ktn',P,projected_H)
                identities['prediction_projection']=max(identities['prediction_projection'],float(np.max(abs(new_relative-np.sum(alternate**2,axis=(1,2))))))
                meter=None if runner is None else meter_variance(P@(reps[target]-reps[runner]).T,
                    true_root,true_V,entry['voltage_sigma'],score['sigma2'])
                rows.append(dict(id=entry['id'],day=day,role=entry['role'],case=entry['case'],strategy=r['strategy'],status='complete',
                    original_correct=bank[old]['topology_id']==truth['truth_id'],relative_correct=bank[rel]['topology_id']==truth['truth_id'],
                    original_native_index=int(old),relative_native_index=int(rel),
                    selected_topology_changed=bank[old]['topology_id']!=bank[rel]['topology_id'],
                    historical_target=target,historical_runner=runner,meter=meter))
    relative_by_strategy={}
    table={(r['id'],r['day'],r['strategy']):r for r in rows}
    for strategy in sorted({r['strategy'] for r in rows}):
        if strategy=='no_dfs':
            continue
        own=[r for r in rows if r['strategy']==strategy]
        relative_by_strategy[strategy]=dict(n=len(own),
            new_recoveries=sum(r['relative_correct'] and not table[r['id'],r['day'],'no_dfs']['relative_correct'] for r in own),
            lost_recoveries=sum(not r['relative_correct'] and table[r['id'],r['day'],'no_dfs']['relative_correct'] for r in own))
    result=dict(source=io.ref(__file__),protocol=io.ref(run_dir/'protocol.json'),secondary_metric_lock=locked,
        metric_role='Development scoring diagnostic' if locked is None else 'Prelocked secondary scoring ablation; original primary outcome remains unchanged',
        fixed_history_rows=sorted({e['validation_points'] for e in entries}),new_rows_per_day=96,
        overall=aggregate(rows),by_role={role:aggregate([r for r in rows if r['role']==role]) for role in sorted({r['role'] for r in rows})},
        by_case={case:aggregate([r for r in rows if r['case']==case]) for case in sorted({r['case'] for r in rows})},
        relative_strategy_vs_relative_no_dfs=relative_by_strategy,
        history=dict(n=len(history),original_correct=sum(r['original_correct'] for r in history),relative_correct=sum(r['relative_correct'] for r in history),
            rescued=sum(r['relative_correct'] and not r['original_correct'] for r in history),lost=sum(r['original_correct'] and not r['relative_correct'] for r in history)),
        identities=dict(identities),errors=errors,passed=not errors and max(identities.values(),default=0.)<1e-10,
        scope='Scoring diagnostic uses residuals minus their within-time terminal mean. Same full native parameter bank, full old validation, full new day and original SSE tie width. Frozen controls, no replanning/refitting/simulation and no extra independent paths. Historical winner-versus-runner defines meter comparisons before the new day. True root/terminal voltage used only for offline variance diagnostics. Non-development batches require an explicit secondary-metric lock based on prior development evidence.',
        interpretation='Any raw-to-relative recovery difference is a scoring-rule change, not a flexibility benefit. Voltage-only variance is exact conditional on the saved physical state and measured power; it excludes power-meter uncertainty when power is not conditioned upon and frozen-parameter bias. Projected variance uses the old common score scale for comparison, not a newly calibrated likelihood.')
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'source_snapshot.py')
    result['source_snapshot']=io.ref(out/'source_snapshot.py')
    io.write(out/'report.json',result);io.write(out/'rows.json',rows);io.write(out/'history.json',history)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--secondary-lock',type=Path)
    args=parser.parse_args();result=run(args.run.resolve(),args.out.resolve(),args.secondary_lock)
    import json
    print(json.dumps(dict(history=result['history'],overall={k:v for k,v in result['overall'].items() if not k.endswith('_rotation')},
        identities=result['identities'],errors=result['errors'],passed=result['passed']),indent=2))
