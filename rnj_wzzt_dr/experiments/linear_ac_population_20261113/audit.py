"""Independent all-row forecast/meter/likelihood audits and sampled AC replay."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_name]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import numpy as np
from ... import io
from ..unified_posterior_dr_20261112.history_scan import ROOT,require,sha,utc
from ..unified_posterior_dr_20261112.physical import open_runtime,generate_background
from ..expected_logodds_admm_20261113.run import REFERENCE


def run(out,phase,shard=0,shards=1):
    protocol=io.read(out/'protocol.json');inputs={x['request']['id']:x for x in io.read(out/'inputs.json')}
    require((out/phase/'complete.json').exists(),'Finish phase before auditing')
    audit_target=out/phase/('independent_audit.json' if shards==1 else f'audit_part{shard}.json')
    require(not audit_target.exists(),'Keep prior audit')
    for rel,ref in protocol['sources'].items():require(sha(ROOT/rel)==ref['sha256'],'Source changed: '+rel)
    truth={x['id']:x for x in io.read(REFERENCE/'evaluation.json')};modules={}
    max_meter=max_forecast=max_probability=max_gain=max_ac=0.;rows_checked=contexts=plans_checked=ac_replays=0;failures=0
    audited=[]
    directories=[d for d in sorted((out/phase/'jobs').iterdir()) if d.is_dir()]
    for index,directory in enumerate(directories):
        if index%shards!=shard:continue
        if not directory.is_dir():continue
        audited.append(directory.name)
        entry=inputs[directory.name];score=io.read(entry['scores']['path']);require(sha(entry['scores']['path'])==entry['scores']['sha256'],'History score changed')
        rows=io.read(directory/'results.json');target=entry['target'];n=len(score['labels']);scale=np.array(entry['scale'])
        bykey={(r['day'],r['strategy']):r for r in rows};failures+=sum(r['status']!='complete' for r in rows)
        for d in sorted(directory.glob('day*')):
            if not (d/'observed.npz').exists():continue
            day=int(d.name[3:]);meta=io.read(d/'generation.json');seal=io.read(d/'policy_seal.json')
            for key in ('planner','plans'):require(sha(seal[key]['path'])==seal[key]['sha256'],'Frozen policy file changed')
            require(not seal['true_voltage_or_line_parameters_in_planner'],'Future channels reached planner')
            plans=io.read(d/'plans.json')
            with np.load(d/'physical.npz',allow_pickle=False) as physical,np.load(d/'observed.npz',allow_pickle=False) as observed,np.load(d/'planner.npz',allow_pickle=False) as packet:
                true=np.column_stack((physical['baseline_P'],physical['baseline_Q']))
                predicted=true*(1+protocol['prediction_relative_sd']*physical['forecast_noise'])
                error=float(np.max(abs(predicted-packet['forecast'])));max_forecast=max(max_forecast,error)
                require(error<1e-12,'Forecast is not true new day plus prescribed noise')
                names=observed['strategy'].tolist();require(names==physical['strategy'].tolist(),'Observed/physical policy order differs')
                for i,strategy in enumerate(names):
                    row=bykey[day,strategy];require(row['status']=='complete','Observed row not marked complete')
                    x=packet[strategy];dp=np.repeat(x,4,axis=0)*scale
                    require(np.max(abs(physical['P'][i]-physical['baseline_P']-dp))<1e-12,'Linear P delivery differs')
                    require(np.max(abs(physical['Q'][i]-physical['baseline_Q']-protocol['qratio']*dp))<1e-12,'Linear Q delivery differs')
                    expectations=[physical['P'][i]*(1+meta['pq_sigma']*physical['standardized_meter_P']),
                        physical['Q'][i]*(1+meta['pq_sigma']*physical['standardized_meter_Q']),
                        physical['V'][i]*(1+meta['voltage_sigma']*physical['standardized_meter_V']),
                        physical['true_root']*(1+meta['voltage_sigma']*physical['standardized_meter_root'])]
                    error=max(float(np.max(abs(observed[k][i]-value))) for k,value in zip(('P','Q','V','root'),expectations,strict=True))
                    max_meter=max(max_meter,error);require(error<1e-12,'Original meter noise differs')
                    P=observed['P'][i].astype(np.longdouble);Q=observed['Q'][i].astype(np.longdouble)
                    Y=observed['root'][i,:,None].astype(np.longdouble)**2-observed['V'][i].astype(np.longdouble)**2
                    losses=[]
                    for p in score['parameters']:
                        prediction=np.zeros((96,n),dtype=np.longdouble);point=p['point']
                        for C,r,v in zip(point['supports'],point['r'],point['x'],strict=True):
                            contribution=np.longdouble(r)*P[:,C].sum(axis=1)+np.longdouble(v)*Q[:,C].sum(axis=1)
                            prediction[:,C]+=contribution[:,None]
                        losses.append(np.sum((Y-prediction)**2)+np.longdouble(p['history_sse']))
                    losses=np.array(losses);weights=np.exp(-(losses-losses.min())/(2*np.longdouble(score['sigma2'])));weights/=weights.sum()
                    error=float(np.max(abs(weights-np.exp(observed['log_probability'][i]))));max_probability=max(max_probability,error)
                    require(error<1e-7,'Independent full-support posterior differs')
                    best=int(np.argmax(weights));ids=[m['topology_id'] for m in score['models']]
                    ti=next((j for j,identity in enumerate(ids) if identity==truth[directory.name]['truth_id']),None)
                    require(best==row['selected_index'] and row['map_correct']==(ti==best),'MAP fields differ')
                    require(abs(float(weights[target])-row['p_target'])<1e-7,'Target posterior field differs')
                    if ti is not None:require(abs(float(weights[ti])-row['p_true'])<1e-7,'True posterior field differs')
                    order=np.argsort(-weights,kind='stable');count=min(int(np.searchsorted(np.cumsum(weights[order]),.9))+1,len(order))
                    kept=np.flatnonzero(weights>=weights[order[count-1]])
                    require(len(kept)==row['retained_count'] and row['correct_unique']==(len(kept)==1 and ti in kept),'Credible-set decision differs')
                    require(row['wrong_unique']==(len(kept)==1 and ti not in kept),'Wrong unique decision differs')
                    require(abs(float(np.mean(x*x))-row['cost'])<1e-12,'Applied fee differs')
                    if strategy.startswith('optimized_'):
                        label=strategy.split('_',1)[1];plan=plans[label]
                        if plan['qualified'] and entry['competitors']:
                            full=[]
                            H=np.array([p['H'] for p in score['parameters']]);shift=np.column_stack((dp,protocol['qratio']*dp))
                            for j in entry['competitors']:
                                D=H[target]-H[j];base=packet['forecast']@D.T;delta=shift@D.T
                                full.append(float(np.sum(2*base*delta+delta*delta)/(2*score['sigma2'])))
                            error=float(np.max(abs(np.array(full)-plan['gains'])));max_gain=max(max_gain,error)
                            gamma=next(g for g in protocol['gamma'] if f'g{g:.6g}'==label)
                            require(error<1e-6*max(1.,max(abs(np.array(full)))) and min(full)>=gamma-1e-8,'Original QCQP gain differs')
                            require(plan['lower_bound']<=plan['cost']+1e-8,'Bound exceeds fee')
                            if plan['global_optimality_verified']:require(plan['relative_gap']<=protocol['global_gap_tolerance'],'Global-gap label differs')
                        partner=packet['rotation_'+label];require(abs(float(np.mean(partner*partner))-row['cost'])<1e-12,'Rotation fee not matched')
                        plans_checked+=1
                    rows_checked+=1
                # Deterministic replay of existing paths; no fresh noise or new day.
                if day==0 and (phase!='population' or entry['index']%25==0):
                    cohort=entry['request']['cohort']
                    if cohort not in modules:modules[cohort]=open_runtime(out,cohort)[0]
                    parameters={k:meta[k] for k in ('case','replicate','voltage_sigma','pq_sigma','root_sigma','study_seed','profile','level')}
                    background=generate_background(modules[cohort],parameters)
                    require(background['metadata']['clean_observation_sha256']==meta['clean_observation_sha256'],'Baseline AC replay differs')
                    for i in range(len(names)):
                        voltage,_=modules[cohort]._ac_voltages(background['network'],background['labels'],physical['P'][i],physical['Q'][i],physical['true_root'])
                        error=float(np.max(abs(voltage-physical['V'][i])));max_ac=max(max_ac,error)
                        require(error<1e-10,'Saved voltage is not original AC output');ac_replays+=1
                contexts+=1
        # An AC failure still has a frozen quote. Check its original QCQP
        # independently, without fabricating a missing voltage/posterior.
        for row in rows:
            if row['status']!='AC_or_scoring_failed' or not row['strategy'].startswith('optimized_'):continue
            d=directory/f"day{row['day']:03d}";label=row['strategy'].split('_',1)[1];plan=io.read(d/'plans.json')[label]
            with np.load(d/'planner.npz',allow_pickle=False) as packet:
                x=packet[row['strategy']];dp=np.repeat(x,4,axis=0)*scale
                shift=np.column_stack((dp,protocol['qratio']*dp));H=np.array([p['H'] for p in score['parameters']]);full=[]
                for j in entry['competitors']:
                    D=H[target]-H[j];base=packet['forecast']@D.T;delta=shift@D.T
                    full.append(float(np.sum(2*base*delta+delta*delta)/(2*score['sigma2'])))
                gamma=next(g for g in protocol['gamma'] if f'g{g:.6g}'==label)
                error=float(np.max(abs(np.array(full)-plan['gains'])));max_gain=max(max_gain,error)
                require(min(full)>=gamma-1e-8 and error<1e-6*max(1.,max(abs(np.array(full)))),'Failed-AC quote fails QCQP gain')
                require(abs(float(np.mean(x*x))-row['cost'])<1e-12,'Failed-AC quote fee differs')
                plans_checked+=1
    receipt=dict(status='passed',created_utc=utc(),phase=phase,contexts_checked=contexts,rows_checked=rows_checked,
        failed_rows_retained=failures,plans_checked=plans_checked,maximum_forecast_error=max_forecast,
        maximum_meter_error=max_meter,maximum_probability_error=max_probability,maximum_gain_error=max_gain,
        maximum_AC_replay_error=max_ac,AC_replay_calls=ac_replays,new_independent_days=0,
        all_consumed_source_hashes_unchanged=True,source=io.ref(__file__),audited_inputs=audited)
    io.write(audit_target,receipt);print({k:v for k,v in receipt.items() if k not in ('source','audited_inputs')},flush=True)
    return receipt


def parallel(out,phase,workers):
    require(not (out/phase/'independent_audit.json').exists(),'Keep prior audit')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run,out,phase,k,workers) for k in range(workers)]
        results=[f.result() for f in futures]
    ids=[identity for r in results for identity in r['audited_inputs']]
    expected=[r['id'] for r in io.read(out/phase/'complete.json')['receipts']]
    require(len(ids)==len(set(ids)) and set(ids)==set(expected),'Audit partition misses or duplicates inputs')
    result=dict(status='passed',created_utc=utc(),phase=phase,audited_inputs=len(ids),
        all_consumed_source_hashes_unchanged=True,source=io.ref(__file__),independent_shards=workers)
    for key in ('contexts_checked','rows_checked','failed_rows_retained','plans_checked','AC_replay_calls','new_independent_days'):
        result[key]=sum(r[key] for r in results)
    for key in ('maximum_forecast_error','maximum_meter_error','maximum_probability_error','maximum_gain_error','maximum_AC_replay_error'):
        result[key]=max(r[key] for r in results)
    io.write(out/phase/'independent_audit.json',result);print({k:v for k,v in result.items() if k!='source'},flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--phase',required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();parallel(a.run.resolve(),a.phase,a.workers) if a.workers>1 else run(a.run.resolve(),a.phase)
