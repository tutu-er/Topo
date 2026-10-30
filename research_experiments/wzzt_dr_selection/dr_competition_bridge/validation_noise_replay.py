"""Frozen-fit validation-noise diagnostic: no LP, AC, actual future or truth."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse, gzip, hashlib, importlib.util, json, platform, shutil, sys, traceback
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import numpy as np

SCRATCH=Path(r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
DEFAULT_PARENT=SCRATCH/'wzzt_dr_competition_bridge_v2'
DEFAULT_OUT=SCRATCH/'wzzt_dr_competition_validation_noise_v4'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Fraction):return str(x)
    return x
def write(p,x):Path(p).write_text(json.dumps(plain(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def family(p,keep=None):
    return {tuple(s) for i,(s,r,x) in enumerate(zip(p['supports'],p['r'],p['x'])) if (keep is None or i in keep) and (r>0 or x>0)}
def matrices(p):
    r=np.zeros((15,15));x=r.copy()
    for s,rv,xv in zip(p['supports'],p['r'],p['x']):
        if rv<0 or xv<0:raise ValueError('negative fitted coefficient')
        r[np.ix_(s,s)]+=rv;x[np.ix_(s,s)]+=xv
    return r,x

def run(parent,out):
    if out.exists():raise FileExistsError('Existing diagnostic retained; no overwrite/resume')
    cfg=read(parent/'protocol.json');case=next(r for r in cfg['cases'] if r['voltage_sigma']==.001)
    folder=parent/'planning'/case['job_id']/'response_table';plan=read(folder/'plan_result.json')
    if plan['status']!='complete' or plan['training_pairs_completed']!=48:raise ValueError('complete v2 table required')
    maskfiles=sorted((folder/'training_masks').glob('*.json'))
    if len(maskfiles)!=48:raise ValueError('exactly48 existing records required')
    helper=Path(__file__).resolve().parents[0]/'pricing_tables.py'
    if not helper.exists():
        helper=Path(r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo\research_experiments\wzzt_dr_selection\dr_competition_bridge\pricing_tables.py')
    expected=next(v for p,v in plan['identity']['source_sha256'].items() if Path(p).name=='pricing_tables.py')
    if sha(helper)!=expected:raise RuntimeError('pure price helper differs from frozen v2')
    files=[parent/'protocol.json',folder/'plan_result.json',folder/'plan_protocol.json',Path(case['existing_result']['path']),
           Path(case['train']['path']),Path(case['validation']['path']),folder.parent/'stochastic_input_local.json',folder.parent/'stochastic_input_common.json',*maskfiles]
    oldhash=cfg['input_sha256']
    for p in (Path(case['existing_result']['path']),Path(case['train']['path']),Path(case['validation']['path'])):
        if sha(p)!=oldhash[str(p)]:raise RuntimeError('old source hash changed: '+str(p))
    out.mkdir();(out/'source').mkdir();shutil.copyfile(__file__,out/'source/validation_noise_replay.py');shutil.copyfile(helper,out/'source/pricing_tables.py')
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),stage='validation_noise_replay_v4',
        scope='Frozen v2 fitted points; old validation untouched; noisy predicted validation only, no training noise/refit',
        inputs_sha256={str(p):sha(p) for p in files},source_sha256={str(p):sha(p) for p in (out/'source/validation_noise_replay.py',out/'source/pricing_tables.py')},
        noise_seeds=list(range(620010,620018)),sigmaV=.001,sigmaRoot=.001,sigmaPQ=.005,
        noise='independent Gaussian relative P/Q/terminal voltage; one shared root error per time and event; same standard draws across masks/hypotheses/prices',
        raw_prediction='rootnominal1; V=sqrt(1-RP-XQ), no AC or clipping',
        units=dict(P='pu on200kW base',Q='pu on200kvar base',V='pu',drop='pu2',interval_hours=.25),
        first_gate='sigma0 all768 hypothesis/mask errors must reproduce v2 exactly before noisy scoring',
        unchanged=dict(historical_train_rows=288,historical_validation_rows=288,future_validation_rows=192,price_levels=[0,1,2],participation=[0,.5,.75],four_event_budget=1.6,waveform='.2kW/.07kvar19-20h with20-21h rebound'),
        price_objective='equal3 historical fitted models and8 fixed predictive-noise draws, exact participation integration',
        interpretation='conditional forward-model stress diagnostic, not field calibration, truth probability or full training-noise simulation',
        solver_calls=0,AC_calls=0,new_actual_future_or_topology_read=False)
    write(out/'protocol.json',protocol)
    spec=importlib.util.spec_from_file_location('_frozen_price_tables',out/'source/pricing_tables.py');pricing=importlib.util.module_from_spec(spec);spec.loader.exec_module(pricing)
    receipt=dict(status='started',protocol_sha256=sha(out/'protocol.json'),phase='sigma0',completed_fit_records=0,errors=[],solver_calls=0,AC_calls=0)
    write(out/'run_receipt.json',receipt)
    losses=np.full((9,3,16,16,2),np.nan);selected=np.full(losses.shape,-1,dtype=np.int16);maes=np.full(losses.shape,np.nan)
    context={}
    try:
        old=read(case['existing_result']['path']);labels=old['labels'];cal=old['calibration'];hist=[]
        for role in ('train','validation'):
            with np.load(case[role]['path'],allow_pickle=False) as a:
                if set(a.files)!={'labels','P','Q','V','root'}:raise ValueError('unexpected old observed channels')
                if a['labels'].astype(int).tolist()!=labels:raise ValueError('old label order mismatch')
                hist.append({k:np.array(a[k],float) for k in ('P','Q','V','root')})
        if any(d['P'].shape!=(3,96,15) for d in hist):raise ValueError('complete old histories required')
        P=np.concatenate([h['P'] for h in hist]).mean(axis=0);Q=np.concatenate([h['Q'] for h in hist]).mean(axis=0)
        P=np.repeat(P[None],4,axis=0);Q=np.repeat(Q[None],4,axis=0)
        positions=[labels.index(v) for v in plan['identity']['active_labels']]
        for mask in range(4):
            for j,pos in enumerate(positions):
                if mask>>j&1:
                    P[mask,76:80,pos]-=.001;P[mask,80:84,pos]+=.001
                    Q[mask,76:80,pos]-=.00035;Q[mask,80:84,pos]+=.00035
        vp=hist[1]['P'].reshape(288,15);vq=hist[1]['Q'].reshape(288,15);vy=(hist[1]['root'][:,:,None]**2-hist[1]['V']**2).reshape(288,15)
        records=[read(p) for p in maskfiles];keys=[(i,j) for i in range(4) for j in range(4)];keypos={k:i for i,k in enumerate(keys)}
        if any(r['status']!='complete' for r in records):raise ValueError('incomplete fitted record')
        references={tuple(r['key']):r for r in plan['response_table']};zero_checks=0
        for phase,seeds,offset in (('sigma0',[None],0),('prediction_noise',protocol['noise_seeds'],1)):
            receipt.update(phase=phase);write(out/'run_receipt.json',receipt)
            zp,zq,zv,zr=[np.zeros((len(seeds),2,96,n)) for n in (15,15,15,1)]
            for s,seed in enumerate(seeds):
                if seed is None:continue
                for event in range(2):
                    for stream,a in enumerate((zp,zq,zv,zr)):
                        a[s,event]=np.random.default_rng(np.random.SeedSequence([seed,event,stream])).normal(size=a.shape[2:])
            pq=0. if phase=='sigma0' else .005;voltage=0. if phase=='sigma0' else .001
            po=P[None,None]+pq*np.maximum(abs(P),1e-12)[None,None]*zp[:,:,None]
            qo=Q[None,None]+pq*np.maximum(abs(Q),1e-12)[None,None]*zq[:,:,None]
            root_observed=1.+voltage*zr[:,:,None]
            for record in records:
                m=record['model_serial'];ti=keypos[tuple(record['training_mask_key'])];context=dict(phase=phase,model=m,training_key=record['training_mask_key'])
                h=old['base_points'][record['raw_index']];rh,xh=matrices(h);target=family(h)
                raw_y=P@rh.T+Q@xh.T
                if (1.-raw_y<=0).any():raise ValueError('nonpositive predicted squared voltage; no clipping')
                v=np.sqrt(1.-raw_y);vo=v[None,None]+voltage*np.maximum(v,1e-12)[None,None]*zv[:,:,None]
                if (vo<=0).any() or (root_observed<=0).any():raise ValueError('nonpositive noisy predicted voltage')
                yo=root_observed**2-vo**2;points=record['candidate_points'];kernels=[matrices(p) for p in points]
                rstack=np.stack([k[0] for k in kernels]);xstack=np.stack([k[1] for k in kernels])
                past=abs(np.einsum('tj,kij->kti',vp,rstack)+np.einsum('tj,kij->kti',vq,xstack)-vy).sum(axis=(1,2))
                residual=np.einsum('semtj,kij->ksemti',po,rstack,optimize=True)
                residual+=np.einsum('semtj,kij->ksemti',qo,xstack,optimize=True);residual-=yo[None]
                future=abs(residual).sum(axis=(-1,-2));errors=[]
                for p in points:
                    keep=[i for i,(s,r,x) in enumerate(zip(p['supports'],p['r'],p['x'])) if len(s)==1 or cal['a']*r+cal['b']*x+1e-10>=.16*cal['median_root_depth']]
                    errors.append(len(family(p,keep)^target))
                for vi,(a,b) in enumerate(keys):
                    scores=(past[:,None]+future[:,:,0,a]+future[:,:,1,b])/7200
                    for seed_i in range(len(seeds)):
                        for alg,limit in enumerate((len(points),15)):
                            values=scores[:limit,seed_i];best=values.min();eligible=np.flatnonzero(values<=1.03*best+1e-12)
                            chosen=min(eligible,key=lambda i:(len(points[i]['supports']),values[i],int(i)))
                            losses[offset+seed_i,m,ti,vi,alg]=errors[chosen];selected[offset+seed_i,m,ti,vi,alg]=chosen;maes[offset+seed_i,m,ti,vi,alg]=values[chosen]
                            if phase=='sigma0':
                                ref=references[tuple(record['training_mask_key'])+(a,b)]['wzzt_by_hypothesis' if alg==0 else 'RNJ_by_hypothesis'][m]
                                if errors[chosen]!=ref:raise AssertionError('sigma0 state did not reproduce v2')
                        if phase=='sigma0':zero_checks+=1
                receipt['completed_fit_records']+=1
            if phase=='sigma0':
                if zero_checks!=768:raise AssertionError('incomplete sigma0 gate')
                write(out/'sigma0_replay.json',dict(status='passed',hypothesis_states=zero_checks,algorithm_comparisons=2*zero_checks,solver_calls=0))
        if not np.isfinite(losses).all():raise RuntimeError('incomplete conditional tables; do not integrate partial values')
        menus=pricing.enumerate_menus(2);singles=np.array([x for x in menus if np.count_nonzero(np.any(x>0,axis=0))<=1]);results={}
        for law in ('local','common'):
            mean=losses[1:,:,:,:,0].mean(axis=(0,1));table={keys[t]+keys[v]:float(mean[t,v]) for t in range(16) for v in range(16)}
            scored=pricing.score_menus(menus,table,law,tie_seed=530014);one=pricing.score_menus(singles,table,law,tie_seed=530014)
            dopt=np.asarray(read(folder.parent/f'stochastic_input_{law}.json')['menu'])[:,positions]
            per_seed=[]
            for s in range(8):
                z=losses[s+1,:,:,:,0].mean(axis=0);small={keys[t]+keys[v]:float(z[t,v]) for t in range(16) for v in range(16)}
                score=pricing.score_menus(menus,small,law,tie_seed=530014);per_seed.append(dict(seed=protocol['noise_seeds'][s],selected_menu=score['selected_menu'],best=score['best_expected_loss']))
            values=[r['expected_loss'] for r in scored['records']];zero=np.zeros((2,2),dtype=int)
            results[law]=dict(scored=scored,flat=max(values)-min(values)<=1e-12,loss_range=[min(values),max(values)],
                no_reward_loss=pricing.expected_loss(table,zero,law),within_pair_single=one,
                pair_stochastic_Dopt_menu=dopt,pair_stochastic_Dopt_loss=pricing.expected_loss(table,dopt,law),
                differs_from_no_reward=not np.array_equal(scored['selected_menu'],zero),differs_from_single=not np.array_equal(scored['selected_menu'],one['selected_menu']),
                differs_from_pair_stochastic_Dopt=not np.array_equal(scored['selected_menu'],dopt),per_noise_draw=per_seed)
        payload=dict(axis_order=['sigma0_then8seeds','hypothesis','training_key','validation_key','algorithm_WZZT_RNJ'],keys=keys,noise_seeds=protocol['noise_seeds'],
            hypotheses=[dict(model_serial=m,raw_index=next(r['raw_index'] for r in records if r['model_serial']==m)) for m in range(3)],
            losses=losses,selected_candidate_indices=selected,selected_raw_validation_mae=maes,labels=labels,active_labels=plan['identity']['active_labels'])
        (out/'tables.json.gz').write_bytes(gzip.compress(json.dumps(plain(payload),separators=(',',':'),allow_nan=False).encode(),mtime=0))
        write(out/'price_scores.json',results)
        receipt.update(status='complete',sigma0_states_reproduced=768,noisy_hypothesis_states=8*3*256,
            risk_mean_by_hypothesis=losses[1:,:,:,:,0].mean(axis=(0,2,3)),price_summary={law:{k:results[law][k] for k in ('flat','loss_range','no_reward_loss','differs_from_no_reward','differs_from_single','differs_from_pair_stochastic_Dopt')} for law in results})
        print(json.dumps(plain(receipt['price_summary']),ensure_ascii=False),flush=True)
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,message=str(error),context=context,traceback=traceback.format_exc())
        receipt['errors'].append(dict(context=context,type=type(error).__name__,message=str(error)))
        np.savez_compressed(out/'partial_tables.npz',losses=losses,selected=selected,maes=maes)
        print(json.dumps(dict(status='failed',context=context,error=str(error))),flush=True)
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(),environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),actual_future_read=False,truth_or_network_read=False,training_parameters_refitted=False)
        write(out/'run_receipt.json',receipt)
        manifest={p.relative_to(out).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()}
        write(out/'manifest.json',manifest)
    return receipt

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--parent',type=Path,default=DEFAULT_PARENT);a.add_argument('--out',type=Path,default=DEFAULT_OUT);args=a.parse_args()
    result=run(args.parent,args.out);raise SystemExit(0 if result['status']=='complete' else 2)
