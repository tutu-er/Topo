"""Frozen Gaussian DR plans evaluated with the existing nonlinear AC producer.

No optimization, identifier fit, clipping, resampling or validation-rule change.
Private trajectories are never part of the public summary.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,hashlib,json,pickle,platform,sys,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import scipy

REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
PRIVATE=Path(r'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
sys.path.insert(0,str(REPO))
from research_experiments.wzzt_dr_selection.dr_newday_weighted.weighted_selection import score
from research_experiments.wzzt_dr_selection.dr_frozen_distance.frozen_pool import _power

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(r):
    p=Path(r['path'])
    if sha(p)!=r['sha256']:raise ValueError('input drift '+str(p))
    return p
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def dump(p,x):
    with Path(p).open('wb') as f:pickle.dump(x,f,protocol=4)
def unpickle(p):
    with Path(p).open('rb') as f:return pickle.load(f)

def sources(source_root):
    paths=list(source_root.rglob('*.py'))+list(Path(__file__).parent.glob('*.py'))
    for s in ['dr_competition_bridge/physical_bridge.py','end_to_end/physical_case.py',
              'dr_frozen_distance/frozen_pool.py','dr_newday_weighted/weighted_selection.py',
              'dr_pool_v2/selection.py','dr_pool_v2/observations.py']:
        paths.append(REPO/'research_experiments/wzzt_dr_selection'/s)
    return sorted(set(p.resolve() for p in paths))

def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    parent=PRIVATE/'wzzt_conditional_dr_stage';old=read(parent/'run_v1/protocol.json')
    inputs=old['inputs'];pool=read(checked(inputs['pool']))
    high_ref=ref(parent/'run_v1/price_freeze.json');low_ref=ref(parent/'delivery_v1/price_freeze.json')
    high=read(checked(high_ref));low=read(checked(low_ref))
    policies={'zero':high['policies']['zero']}
    for tier,plans in [('high',high),('low',low)]:
        for kind in ['continuous','rotation']:policies[kind+'_'+tier]=plans['policies'][kind]
    for name,v in policies.items():
        u=np.asarray(v['mean_u'],float)
        if u.shape!=(2,96) or not np.isfinite(u).all():raise ValueError('bad frozen plan')
        if not np.isclose(.25*np.sum(u*u/.5),v['planned_cost'],rtol=1e-12):raise ValueError('cost mismatch')
    old_physical=read(PRIVATE/'wzzt_day_effect_stage/actual_v1/protocol.json')
    source_root=Path(old_physical['source_root']);metadata=old_physical['source_metadata'];checked(metadata)
    files=sources(source_root)
    # Prespecified equal representation of all three original profiles; fresh replicates.
    days=[dict(day=i,profile_index=i%3,replicate=930000+97*i,response_seed=2026110210+i) for i in range(12)]
    identities={(d['profile_index'],d['replicate']) for d in days};old_ids=set();examined=0
    for f in PRIVATE.rglob('*request*.json'):
        try:d=read(f)
        except (ValueError,OSError):continue
        if isinstance(d,dict) and d.get('case')=='paper15' and 'profile_index' in d and 'replicate' in d:
            old_ids.add((int(d['profile_index']),int(d['replicate'])));examined+=1
    if identities&old_ids:raise ValueError('previously observed physical identity')
    protocol=dict(id='conditional_ac_v1',status='frozen',created_utc=now(),case='paper15',
        condition='paper15_b2_v0005',inputs=inputs,high_prices=high_ref,low_prices=low_ref,
        source_root=str(source_root),source_metadata=metadata,
        source_sha256={str(f):sha(f) for f in files},labels=pool['labels'],root_label=1,base_kw=200.,
        resource_labels=[103,101],history_rows=[288,288],slots=96,days=days,
        voltage_sigma=.0005,pq_sigma=.005,root_physical_sigma=.003,impedance_scale=1.,
        observation_variants=['none','v','pq','both'],main_observation='both',
        response='U=u*(1+sqrt(.025/3)*Z); independent normal per node-slot; same Z across policies',
        reactive_ratio=.35,dt_hours=.25,cv=float(np.sqrt(.025/3)),
        fee='prepaid .25*sum(u^2/.5), no execution clawback; currency uncalibrated',
        execution='no clipping, no rejection/redraw of resource or voltage limit excursions; retain AC failures',
        rule=dict(history_gate=True,new_weight=2,new_fraction=.03,history_fraction=.03),
        identification='reuse frozen historical pair R/X and fixed .16 output map; no new fit or W search',
        truth_gate='all predictive selections frozen before evaluate_supports; true graph used by AC producer only',
        independent_unit='12 new physical-background replicates, each one whole response; three fixed profile strata; no probability certification',
        prior_request_audit=dict(examined=examined,distinct=len(old_ids),overlap=0),
        environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,platform=platform.platform()))
    write(root/'protocol.json',protocol)
    write(root/'price_freeze.json',dict(status='frozen',policies=policies,parents=[high_ref,low_ref],new_price_solves=0))
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for i,f in enumerate(files):z.write(f,f'{i:03d}_{f.name}')
    # Observed historical denominator only; no forecast or future rows used.
    packet=unpickle(checked(inputs['history']));hist=packet['history'];den={}
    for split in ['training','validation']:
        ps=np.stack([_power(s,pool['labels'])[0] for s in hist[split]])*200
        pos=[pool['labels'].index(i) for i in [103,101]]
        den[split]=dict(days=len(ps),network_kwh_per_day=float(ps.sum(axis=(1,2)).mean()*.25),
            controlled_kwh_per_day=float(ps[:,:,pos].sum(axis=(1,2)).mean()*.25),
            network_mean_kw=float(ps.sum(axis=2).mean()),controlled_mean_kw=float(ps[:,:,pos].sum(axis=2).mean()))
    write(root/'history_load_summary.json',den)
    write(root/'run_lock.json',dict(status='prepared',pid=None,updated_utc=now()))
    print('frozen',len(days),'new days;',len(policies),'policies',flush=True)

def config(root):
    root=Path(root);p=read(root/'protocol.json');f=read(root/'price_freeze.json')
    for path,digest in p['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('source drift '+path)
    for key in ['high_prices','low_prices','source_metadata']:checked(p[key])
    for r in p['inputs'].values():checked(r)
    return root,p,f

def observe(root):
    from ac_bridge import open_runtime,build_background,execute_policy
    root,p,f=config(root)
    if (root/'observations_frozen.json').exists():raise RuntimeError('observations already complete')
    runtime=open_runtime(p['source_root'],p['source_metadata'],repo_root=REPO,
        protocol_ref=ref(root/'protocol.json'),labels=p['labels'],case=p['case'],
        v_sigma=p['voltage_sigma'],pq_sigma=p['pq_sigma'])
    write(root/'run_lock.json',dict(status='running',pid=os.getpid(),updated_utc=now()))
    rows=[]
    for d in p['days']:
        base=build_background(runtime,d['profile_index'],d['replicate'])
        z=np.random.default_rng(d['response_seed']).standard_normal((2,96))
        for name,v in f['policies'].items():
            record=f"day{d['day']:02d}_{name}";folder=root/'actual'/record;folder.mkdir(parents=True,exist_ok=True)
            out=folder/'result.pkl';receipt=folder/'receipt.json'
            if receipt.exists():
                r=read(receipt)
                if 'packet' in r:checked(r['packet'])
                rows.append(r);continue
            try:
                result=execute_policy(runtime,base,np.asarray(v['mean_u']),z,resource_labels=p['resource_labels'],
                    cv=p['cv'],day_offset=6+d['day'],name=record)
                dump(out,result)
                r=dict(record_id=record,**d,policy=name,status='observed',packet=ref(out),physical=result['summary'],
                    fee=float(v['planned_cost']))
            except Exception as e:
                r=dict(record_id=record,**d,policy=name,status='failed',fee=float(v['planned_cost']),
                    error=repr(e),traceback=traceback.format_exc())
            write(receipt,r);rows.append(r)
            write(root/'checkpoint.json',dict(completed=len(rows),planned=60,last_record=record))
        print('observed day',d['day'],len(rows),'/60',flush=True)
    write(root/'observations_frozen.json',dict(status='frozen',rows=rows,protocol=ref(root/'protocol.json'),price_freeze=ref(root/'price_freeze.json')))
    write(root/'run_lock.json',dict(status='observations_frozen',pid=None,updated_utc=now()))

def evaluate(root):
    root,p,f=config(root);pool=read(checked(p['inputs']['pool']));ledger=read(root/'observations_frozen.json')
    checked(ledger['price_freeze']);rows=[]
    for physical in ledger['rows']:
        result=unpickle(checked(physical['packet'])) if physical['status']=='observed' else None
        for variant in p['observation_variants']:
            row={k:v for k,v in physical.items() if k not in ['packet','traceback']};row['noise']=variant
            scene=None if result is None else result['scenes'].get(variant)
            if scene is None:
                row.update(status='unresolved');rows.append(row);continue
            try:
                s=score(pool,scene,history_gate=True)
                q=next(q for q in s['rules'] if q['new_day_weight']==2 and q['new_day_fraction']==.03)
                losses={x['candidate_id']:x['new_validation_mae'] for x in s['loss_rows']}
                outfile=root/'scores'/f"{physical['record_id']}_{variant}.json";write(outfile,s)
                row.update(status='predicted',score=ref(outfile),selected=q['selected_candidate_id'],
                    coarse_excluded='coarse' not in q['eligible_candidate_ids'],refined_excluded='refined' not in q['eligible_candidate_ids'],
                    coarse_margin_micro=1e6*q['exit_margins']['coarse'],D_micro=1e6*(losses['coarse']-1.03*losses['refined']),
                    coarse_L_micro=1e6*losses['coarse'],refined_L_micro=1e6*losses['refined'],
                    distance_micro=1e6*s['new_prediction_pair_distance_l1'])
            except Exception as e:row.update(status='unresolved',evaluation_error=repr(e))
            rows.append(row)
    write(root/'predictions_frozen.json',dict(status='frozen',rows=rows,truth_read=False,observations=ref(root/'observations_frozen.json')))
    from ac_bridge import open_runtime
    runtime=open_runtime(p['source_root'],p['source_metadata'],repo_root=REPO,
        protocol_ref=ref(root/'protocol.json'),labels=p['labels'],case=p['case'],
        v_sigma=p['voltage_sigma'],pq_sigma=p['pq_sigma'])
    write(root/'truth_private.json',runtime['legacy'].evaluate_supports([],runtime['context'].net))
    truth={tuple(x) for x in read(root/'truth_private.json')['missed_blocks']}
    def errs(supports):
        s={tuple(sorted(x)) for x in supports};return dict(missing=len(truth-s),extra=len(s-truth),errors=len(s^truth))
    models={}
    for m in pool['models']:
        raw=[[pool['labels'][i] for i in b] for b in m['point']['supports']]
        models[m['candidate_id']]=dict(raw=errs(raw),final=errs(m['frozen_output_support_labels']))
    for r in rows:
        if r['status']=='predicted':r.update(status='complete',**models[r['selected']]['final'],raw_errors=models[r['selected']]['raw']['errors'])
    for r in rows:
        z=next(x for x in rows if x['day']==r['day'] and x['noise']==r['noise'] and x['policy']=='zero')
        if r['status']==z['status']=='complete':r.update(delta_D_micro=r['D_micro']-z['D_micro'],delta_errors=r['errors']-z['errors'])
    tables=[];pairs=[]
    for variant in p['observation_variants']:
        for name in f['policies']:
            g=[r for r in rows if r['noise']==variant and r['policy']==name];valid=[r for r in g if r['status']=='complete']
            mean=lambda k:float(np.mean([r[k] for r in valid])) if valid else None
            tables.append(dict(noise=variant,policy=name,planned=12,complete=len(valid),unresolved=12-len(valid),
                recoveries=sum(r['errors']==0 for r in valid),full_errors=sum(r['errors'] for r in valid),
                raw_errors=sum(r['raw_errors'] for r in valid),coarse_excluded=sum(r['coarse_excluded'] for r in valid),
                refined_excluded=sum(r['refined_excluded'] for r in valid),mean_D_micro=mean('D_micro'),
                mean_delta_D_micro=mean('delta_D_micro'),mean_coarse_L_micro=mean('coarse_L_micro'),
                mean_refined_L_micro=mean('refined_L_micro'),mean_margin_micro=mean('coarse_margin_micro'),
                mean_distance_micro=mean('distance_micro'),fee_per_day=f['policies'][name]['planned_cost'],
                improved_vs_zero=sum(r['delta_errors']<0 for r in valid),worse_vs_zero=sum(r['delta_errors']>0 for r in valid)))
        for tier in ['high','low']:
            ps=[]
            for d in p['days']:
                a=next(r for r in rows if r['noise']==variant and r['day']==d['day'] and r['policy']=='continuous_'+tier)
                b=next(r for r in rows if r['noise']==variant and r['day']==d['day'] and r['policy']=='rotation_'+tier)
                if a['status']==b['status']=='complete':ps.append(a['errors']-b['errors'])
            pairs.append(dict(noise=variant,tier=tier,better=sum(x<0 for x in ps),worse=sum(x>0 for x in ps),same=sum(x==0 for x in ps),unresolved=12-len(ps)))
    summary=dict(status='complete',tables=tables,pairs=pairs,models=models,planned_AC_days=60,
        observed_AC_days=sum(r['status']=='observed' for r in ledger['rows']),planned_selections=240,
        complete_selections=sum(r['status']=='complete' for r in rows),independent_backgrounds=12,profile_strata=3,
        new_fits=0,new_W_search=0,new_optimizations=0,scope='nonlinear AC simulation and fresh explicit metering noise; frozen historical two-candidate final selection only')
    write(root/'scored_private.json',dict(rows=rows))
    write(root/'summary.json',summary)
    write(root/'ended.json',dict(status='complete',summary=ref(root/'summary.json'),scored=ref(root/'scored_private.json')))
    write(root/'run_lock.json',dict(status='complete',pid=None,updated_utc=now()))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','observe','evaluate']);a.add_argument('--output',type=Path,required=True)
    args=a.parse_args();globals()[args.command](args.output)
