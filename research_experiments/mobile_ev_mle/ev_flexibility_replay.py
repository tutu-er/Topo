"""Independent saved-data audit and descriptive EV flexibility comparisons.

No control selection, new nonlinear fitting, calibration or data generation is
performed. Branch-current backward/forward sweeps replay all available saved
states. Three quality rows are averaged within H before paired summaries.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter
import numpy as np
from .information_fit_recovery import read_json, write_json, exact_candidate, independent_replay
from .information_replay import _obs, _meter, _quality_audit
from .likelihood import concat_observations, RadialCandidate

HERE=Path(__file__).resolve().parent
METHODS=('none','constant_fixed','quality_charge','charge_fixed','charge_random',
         'charge_mobile','v2g_fixed','v2g_mobile','charge_staff')
QUALITY=('uniform','temporal','spatial')
INFORMATION=METHODS[3:]
PAIRS=(('charge_fixed','constant_fixed'),('charge_mobile','charge_fixed'),
       ('v2g_fixed','charge_fixed'),('v2g_mobile','charge_mobile'),
       ('charge_staff','charge_mobile'),('charge_mobile','quality_charge'),
       ('charge_mobile','none'))
CHANNELS=('p','q','v','master_p','master_q','source_p','source_q')


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual,expected,label,*,atol=2e-9,rtol=2e-9):
    a,b=np.asarray(actual,float),np.asarray(expected,float)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a,b,atol=atol,rtol=rtol),label)


def fingerprint(values):
    digest=hashlib.sha256()
    for name in ('p','q','v','master_p','master_q'):
        a=np.asarray(values[name]);digest.update(str(a.shape).encode());digest.update(a.astype('<f8').tobytes())
    return digest.hexdigest()


def selection_kind(row,selection):
    if row.get('status')=='runner_exception':return 'runner_exception'
    if row['method'] not in INFORMATION:return 'diagnostic' if row['method']=='none' else 'baseline'
    if not selection.get('activated',False):return 'not_activated'
    status=row['selection']['status']
    if status=='information_optimized':return 'information_optimized'
    if status=='expanded_domain_incumbent_guard':return 'expanded_domain_guard'
    return 'fallback_after_design_failure'


def source_audit(directory,freeze):
    hashes=freeze['source_sha256'];require(bool(hashes),'nonempty frozen source map')
    snapshot_path=directory/'source_snapshot.json.gz'
    if not snapshot_path.exists():snapshot_path=directory/'science_source_snapshot.json.gz'
    snapshot=read_json(snapshot_path)
    if snapshot.get('encoding')=='gzip_hex':
        require(snapshot.get('hashes')==hashes,'source snapshot hash map')
        decode=lambda value:gzip.decompress(bytes.fromhex(value))
    elif snapshot.get('schema')=='source_bytes_base64':
        require(snapshot.get('policy_id')==freeze['policy_id'],'formal source snapshot identity')
        decode=lambda value:base64.b64decode(value['bytes_base64'],validate=True)
    else:raise ValueError('unsupported frozen source byte snapshot')
    require(set(snapshot.get('sources',{}))==set(hashes),'source snapshot coverage')
    rows=[]
    for name,expected in hashes.items():
        target=(HERE/name).resolve();target.relative_to(HERE.parent.resolve())
        data=decode(snapshot['sources'][name])
        require(hashlib.sha256(data).hexdigest()==expected,'snapshot source bytes:'+name)
        require(target.is_file() and sha(target)==expected,'current frozen source bytes:'+name)
        rows.append(dict(name=name,sha256=expected))
    return dict(success=True,source_count=len(rows),files=rows,snapshot_sha256=sha(snapshot_path))


def validate_coverage(trials,freeze,count):
    require(tuple(freeze['methods'])==METHODS,'method family differs from declared9')
    require(tuple(freeze['quality'])==QUALITY,'quality family differs from declared3')
    seeds=[str(v) for v in freeze['seeds'][:count]]
    expected={(h,q) for h in seeds for q in QUALITY};actual=[]
    for trial in trials:
        key=(trial['history_id'],trial['quality']);actual.append(key)
        require(trial['run_id']==key[0]+'_'+key[1],'run identity')
        require(trial['config']==freeze['config'] and trial['source_sha256']==freeze['source_sha256'],'trial config/source freeze')
        require(trial['old_threshold_applied'] is False,'old calibration threshold used')
        require(trial['selector_uses_truth'] is False and trial['selector_uses_future_noise'] is False,'selector isolation flags')
        require(trial['selection']['selector_uses_truth'] is False and trial['selection']['selector_uses_future_noise'] is False,'selection isolation flags')
        require(trial['independent_history_unit'] and trial['multiple_quality_rows_are_not_independent_histories'],'history statistical unit')
        names=[r['method'] for r in trial['records']]
        require(len(names)==9 and set(names)==set(METHODS),'method coverage/duplicates')
        require(trial['history_hash']==fingerprint(trial['history_observations']),'history observation fingerprint')
    require(len(actual)==len(expected) and set(actual)==expected,'H by quality Cartesian coverage/duplicates')
    for h in seeds:
        rows=[r for r in trials if r['history_id']==h]
        first=rows[0]
        keys=('name','supports','loss','loglik','r','x','p','q','source_p','source_q','success')
        signature=lambda t:(t['history_hash'],t['candidates'],[{k:f[k] for k in keys} for f in t['historical_fits']])
        require(all(signature(t)==signature(first) for t in rows),'same-H fits/candidates changed:'+h)
        require(all(t['truth']==first['truth'] and t['white_future_noise']==first['white_future_noise'] for t in rows),'same-H evaluator path/noise changed:'+h)
    return seeds



def validate_formal_coverage(trials,freeze,phase):
    require(freeze.get('schema')=='ev_service_formal_v1','formal envelope schema')
    require(tuple(freeze['methods'])==METHODS,'formal9 method family')
    expected={str(v['seed']):v for v in freeze['schedule'][phase]}
    require(len(trials)==len(expected) and {t['history_id'] for t in trials}==set(expected),'formal independent H coverage/duplicates')
    wrapper={'ev_flexibility_formal.py','ev_flexibility_formal_protocol.md'}
    policy={k:v for k,v in freeze['source_sha256'].items() if k not in wrapper}
    for trial in trials:
        label=expected[trial['history_id']];formal=trial.get('formal',{})
        require(formal.get('policy_id')==freeze['policy_id'] and formal.get('phase')==phase and formal.get('labels')==label,'formal identity/labels')
        require(formal.get('runner_kind')=='real' and formal.get('one_quality_per_H') is True and formal.get('old_threshold_applied') is False,'real single-quality formal input')
        require(trial['source_sha256']==policy,'formal unchanged scientific policy source')
        require(trial['run_id']==trial['history_id']+'_'+trial['quality'],'formal run identity')
        require(trial['case']==label['case'] and trial['quality']==label['quality'],'formal frozen case/quality')
        require(trial['selector_uses_truth'] is False and trial['selector_uses_future_noise'] is False,'formal selector isolation flags')
        require(len(trial['records'])==9 and {r['method'] for r in trial['records']}==set(METHODS),'formal9 method coverage')
        if not trial.get('runner_exception'):
            require(trial['config']==freeze['policy_config'],'formal policy config')
            require(trial['history_hash']==fingerprint(trial['history_observations']),'formal historical meter fingerprint')
            require(trial['truth']['supports']==label['family'],'formal declared true family')
    return list(expected)


def operation_audit(action,device,*,diagnostic=False):
    p,q=np.asarray(action['p'],float),np.asarray(action['q'],float)
    require(p.shape==(4,3) and q.shape==p.shape,'action P/Q shape')
    require(np.isfinite(p).all() and np.isfinite(q).all(),'finite action')
    if diagnostic:
        close(p,np.zeros_like(p),'noEV power');close(q,np.zeros_like(q),'noEV reactive power')
        close(action['soc'],np.full(5,float(device['initial_soc'])),'noEV battery state')
        return dict(feasible=True,diagnostic=True,grid_charge_kwh=0.,grid_discharge_kwh=0.,throughput_kwh=0.,drive_kwh=0.,curve_kw=p.tolist())
    site=int(action['site']);require(0<=site<3,'site index')
    close(np.delete(p,site,axis=1),np.zeros((4,2)),'single injection location')
    close(q,float(device['q_over_p'])*p,'Q/P rule')
    dt=float(device['dt_hours']);capacity=float(device['battery_capacity']);eta_c=float(device['eta_charge']);eta_d=float(device['eta_discharge'])
    def array(name,shape,default):
        value=device.get(name);value=default if value is None else value
        return np.broadcast_to(np.asarray(value),shape).copy()
    access=array('availability',(4,3),True).astype(bool)
    worker=array('worker_setup_available',(4,3),True).astype(bool)
    travel=array('travel_slots',(3,3),np.ones((3,3),int)-np.eye(3,dtype=int)).astype(int)
    mobility=array('travel_energy',(3,3),.01*(np.ones((3,3))-np.eye(3)))
    setup=array('setup_slots',(3,),0).astype(int);departure=array('departure_slot',(3,),4).astype(int)
    returned=array('return_energy',(3,),0.)
    initial_site=int(device['initial_site']);arrival=int(travel[initial_site,site]);expense=float(mobility[initial_site,site])
    duration=int(setup[site]);finish=int(departure[site]);start=None
    if duration==0:start=arrival if arrival<=finish else None
    else:
        for t in range(arrival,finish-duration+1):
            if worker[t:t+duration,site].all() and access[t:t+duration,site].all():start=t;break
    ready=5 if start is None else start+duration
    require(ready<finish,'arrival/setup outside usable connection')
    allowed=access[:,site].copy();allowed[:min(ready,4)]=False;allowed[finish:]=False
    close(action['sites'],np.where(allowed,site,-1),'connection path',atol=0.,rtol=0.)
    caps_c=array('p_charge_max',(4,3),.3)[:,site]*allowed;caps_d=array('p_discharge_max',(4,3),0.)[:,site]*allowed
    u=p[:,site];require(np.all(u<=caps_c+2e-7) and np.all(u>=-caps_d-2e-7),'power/access caps')
    cp,dp=np.maximum(u,0),np.maximum(-u,0)
    initial=float(device['initial_soc'])*capacity-expense
    energy=np.r_[initial,initial+np.cumsum(dt*(eta_c*cp-dp/eta_d))]
    close(action['soc'],energy/capacity,'battery energy recursion',atol=2e-7)
    require(np.all(energy>=float(device['soc_min'])*capacity-2e-7) and np.all(energy<=float(device['soc_max'])*capacity+2e-7),'SOC bounds')
    require(abs(energy[-1]-float(device['terminal_energy']))<=2e-7,'fixed terminal battery service')
    require(energy[-1]-returned[site]>=max(float(device['departure_reserve']),float(device['soc_min'])*capacity)-2e-7,'return reserve')
    solver=action.get('solver',{})
    if 'split_charge_pu' in solver or 'split_discharge_pu' in solver:
        close(solver.get('split_charge_pu',[]),cp,'charge split exclusion',atol=2e-7)
        close(solver.get('split_discharge_pu',[]),dp,'discharge split exclusion',atol=2e-7)
    charge,discharge=dt*cp.sum(),dt*dp.sum();loss=(1-eta_c)*charge+(1/eta_d-1)*discharge
    require(abs(energy[-1]-(float(device['initial_soc'])*capacity-expense+charge-discharge-loss))<=2e-7,'independent energy accounting')
    return dict(feasible=True,diagnostic=False,site=site,ready_slot=ready,grid_charge_kwh=float(20*charge),grid_discharge_kwh=float(20*discharge),grid_net_kwh=float(20*(charge-discharge)),throughput_kwh=float(20*(charge+discharge)),loss_kwh=float(20*loss),drive_kwh=20*expense,terminal_energy_kwh=float(20*energy[-1]),post_return_energy_kwh=float(20*(energy[-1]-returned[site])),curve_kw=(20*u).tolist())


def future_measurement_audit(trial,row):
    truth=trial['truth'];action=row['action'];physical=row['physical'];n=3
    p=np.asarray(truth['p'],float)[12:]+np.asarray(action['p'],float)
    q=np.asarray(truth['q'],float)[12:]+np.asarray(action['q'],float)
    close(physical['p'],p,'EV exactly once in total P');close(physical['q'],q,'EV exactly once in total Q')
    meter=_meter(trial['measurement_models']['future']);obs=_obs(row['observations'])
    means={}
    for channel in ('p','q','v','master_p','master_q'):
        means[channel]=np.asarray(physical[channel],float)
        if channel in ('p','q','v'):means[channel]=means[channel]+getattr(meter,'bias_'+channel)
        sigma=np.broadcast_to(getattr(meter,channel+'_sigma'),means[channel].shape)
        white=np.asarray(trial['white_future_noise'][channel],float)
        close(getattr(obs,channel),means[channel]+sigma*white,'generated future meter:'+channel,atol=5e-12,rtol=5e-12)
    candidate=RadialCandidate.from_supports('evaluation_truth',n,truth['supports'])
    numerical=dict(r=truth['r_pu'],x=truth['x_pu'],p=p,q=q,source_p=np.asarray(truth['source_p'])[12:],source_q=np.asarray(truth['source_q'])[12:],predicted=means,**{key:getattr(meter,key) for key in ('bias_p','bias_q','bias_v')})
    zero_obs=_obs(dict(**means,root_v=np.ones(4),source_p=None,source_q=None))
    audit=independent_replay(candidate,numerical,zero_obs,meter)
    require(audit['success'] and audit['full_SSE']<1e-12,'physical generation branch sweep')
    require(abs(row['voltage_min']-audit['voltage_min'])<=2e-10 and abs(row['voltage_max']-audit['voltage_max'])<=2e-10,'stored voltage limits')
    return audit


def fit_audits(fits,candidates,observations,meter,failures,*,complete):
    names=[r['name'] for r in fits];require(len(names)==len(set(names)) and set(names)<=set(candidates),'saved candidate fit coverage')
    if complete:require(set(names)==set(candidates) and not failures,'complete fit family missing/failed')
    else:require(set(names)|set(failures)==set(candidates),'failed fit family loses candidates')
    audits=[];unavailable=[]
    for row in fits:
        candidate=exact_candidate(candidates[row['name']],3)
        require(row['supports']==candidates[row['name']]['supports'],'fit edge parameter order')
        if any(row.get(k) is None for k in ('r','x','p','q','source_p','source_q','loss','loglik')):
            require(not row['success'],'successful fit has unavailable numerical state');unavailable.append(row['name']);continue
        replay=independent_replay(candidate,row,observations,meter)
        require(replay['success'],'candidate independent AC/SSE/loglik:'+row['name']+':'+str(replay['errors']))
        audits.append(dict(name=row['name'],declared_fit_success=row['success'],**replay))
    return dict(states_replayed=len(audits),unavailable_failed_states=unavailable,audits=audits)


def audit_trial(trial):
    if trial.get('runner_exception'):
        require(all(r['status']=='runner_exception' for r in trial['records']) and not trial['candidates'] and not trial['historical_fits'],'runner exception must preserve all9 unavailable methods')
        require(trial['total_ac_calls'] is None,'unknown runner cost replaced by zero')
        return dict(run_id=trial['run_id'],history_id=trial['history_id'],quality=trial['quality'],case=trial['case'],success=True,history=dict(states_replayed=0,unavailable_failed_states=[],audits=[]),methods=[dict(method=r['method'],status='runner_exception',selection_kind='runner_exception',selection_status='runner_exception') for r in trial['records']],saved_fit_states_replayed=0,new_unique_joint_fit_states_replayed=0,physical_states_replayed=0,algorithm_ac_calls=None,declared_runner_exception=trial['runner_exception'],scope='original runner failure retained; no numerical state and unknown cost preserved')
    history=_obs(trial['history_observations']);hm=_meter(trial['measurement_models']['history']);fm=_meter(trial['measurement_models']['future']);jm=_meter(trial['measurement_models']['joint'])
    candidates={r['name']:r for r in trial['candidates']};require(len(candidates)==len(trial['candidates'])==4,'four retained star/cherry candidates')
    quality_errors=[];_quality_audit({**trial,'quality_record':trial['selection']['public_quality']},history,hm,fm,jm,quality_errors);require(not quality_errors,'public measurement model:'+str(quality_errors))
    historical=fit_audits(trial['historical_fits'],candidates,history,hm,trial['historical_failures'],complete=not trial['historical_failures'])
    selection=trial['selection'];devices=selection['devices'];config=trial['config']
    for center in selection.get('centres',[]):
        for channel in ('p','q','source_p','source_q','root_v'):
            old=np.asarray(center['history_'+channel],float);future=np.asarray(center['future'][channel],float)
            close(future,np.broadcast_to(old.mean(axis=0),future.shape),'H-only mean forecast:'+channel,atol=1e-12,rtol=1e-12)
    family=lambda supports:tuple(sorted(tuple(sorted(s)) for s in supports if 1<len(s)<3))
    true_names=[name for name,c in candidates.items() if family(c['supports'])==family(trial['truth']['supports'])]
    require(len(true_names)==1,'declared local bank lacks unique true support')
    actual_true_name=true_names[0]
    for device in devices.values():
        for key in ('dt_hours','p_charge_max','eta_charge','eta_discharge','terminal_energy'):
            config_key='terminal_energy_pu_h' if key=='terminal_energy' else key
            require(device[key]==config[config_key],'actual device differs from CONFIG:'+key)
    require(devices['charge']['p_discharge_max']==0 and devices['v2g']['p_discharge_max']==config['p_discharge_max'],'charge/V2G domain')
    rows=[];cache={};all_states=historical['states_replayed'];actual_states=0;physical_states=0
    for row in trial['records']:
        name=row['method'];require(row['selection']==selection['methods'][name],'saved method selection disagrees')
        require(row['true_name']==actual_true_name and row['true_covered'],'true support label/coverage')
        result=dict(method=name,status=row['status'],selection_kind=selection_kind(row,selection),selection_status=row['selection']['status'])
        action=row['action']
        if action is None:
            require(row['status']=='operational_failure','missing action without operational failure');rows.append(result);continue
        device=devices['v2g' if name.startswith('v2g') else 'staff' if name=='charge_staff' else 'charge']
        try:operations=operation_audit(action,device,diagnostic=name=='none')
        except ValueError as error:
            require(row['status']=='operational_failure','unexpected infeasible saved operation:'+str(error));result['declared_failure_verified']=str(error);rows.append(result);continue
        result['operations']=operations
        if row['status'] in ('complete','joint_fit_failure'):
            result['physical']=future_measurement_audit(trial,row);physical_states+=1
            joint=concat_observations(history,_obs(row['observations']))
            result['fits']=fit_audits(row['fits'],candidates,joint,jm,row['fit_failures'],complete=row['status']=='complete')
            all_states+=result['fits']['states_replayed']
            key=(np.asarray(action['p'],dtype='<f8').tobytes(),np.asarray(action['q'],dtype='<f8').tobytes())
            if row['cache_hit']:
                require(key in cache and row['fits']==cache[key],'fit cache changes equivalent actions')
                require(row['ac_calls']==0 and row['fit_calls']==0,'cache hit charged new fits')
            else:
                require(key not in cache,'duplicate action unexpectedly refitted');cache[key]=row['fits'];actual_states+=result['fits']['states_replayed']
                require(row['fit_calls']==4,'all retained candidate fit attempts')
            if row['status']=='complete':
                loss={f['name']:float(f['loss']) for f in row['fits']};require(loss==row['losses'],'saved losses omit/change fits')
                winner=min(loss,key=lambda n:(loss[n],n));require(row['winner']==winner,'winner recomputation')
                true_name=row['true_name'];require(true_name in loss and row['true_covered'],'true candidate missing')
                margin=min(v for n,v in loss.items() if n!=true_name)-loss[true_name]
                require(np.isclose(margin,row['signed_margin_full_SSE'],atol=1e-8,rtol=1e-10),'true signed margin')
                require(np.isclose(loss[true_name]-min(loss.values()),row['true_gap'],atol=1e-8,rtol=1e-10),'true gap')
                fitted=next(f for f in row['fits'] if f['name']==true_name)
                error=np.sqrt(np.mean(np.r_[(np.asarray(fitted['r'])-trial['truth']['r_pu'])**2,(np.asarray(fitted['x'])-trial['truth']['x_pu'])**2]))
                require(np.isclose(error,row['impedance_absolute_rmse_pu'],atol=1e-10,rtol=1e-9),'true candidate impedance RMSE')
                result.update(signed_margin_full_SSE=margin,minimum_SSE_is_truth=winner==true_name,predicted_score=row['selection'].get('predicted_score') if name in INFORMATION else row['selection'].get('information_score'))
        else:require(row['status'] in ('operational_failure','physical_failure'),'unrecognized method failure status')
        result.update(ac_calls=row.get('ac_calls',0),fit_calls=row.get('fit_calls',0),seconds=row['seconds'],cache_hit=row.get('cache_hit',False));rows.append(result)
    counted=(0 if trial.get('historical_cost_cache_hit',False) else trial['historical_cost']['ac_calls'])+selection['selector_ac_calls']+sum(r.get('ac_calls',0) for r in trial['records'])
    require(counted==trial['total_ac_calls'],'actual algorithm AC count with shared H')
    for guard in selection.get('expanded_domain_guards',[]):
        if guard['retained']:
            expanded,base=guard['method'],guard['from_method'];by={r['method']:r for r in trial['records']}
            require(guard['candidate_feasible'] and by[expanded]['action']['site']==by[base]['action']['site'],'expanded-domain guard site/feasibility')
            for channel in ('p','q'):
                close(by[expanded]['action'][channel],by[base]['action'][channel],'expanded guard preserves physical injection:'+channel,atol=1e-12,rtol=1e-12)
            require(by[expanded]['selection']['predicted_score']==guard['candidate_score'],'expanded guard score copied')
            require(guard['old_score'] is None or guard['candidate_score']>guard['old_score'],'guard score direction')
    return dict(run_id=trial['run_id'],history_id=trial['history_id'],quality=trial['quality'],case=trial['case'],success=True,history=historical,methods=rows,saved_fit_states_replayed=all_states,new_unique_joint_fit_states_replayed=actual_states,physical_states_replayed=physical_states,algorithm_ac_calls=counted,scope='algorithm AC counts exclude evaluator truth generation; available failed fit states included; no optimizer invoked')


def grouped_pairs(trials,left,right,*,quality_count=3):
    groups={};failures=[]
    for trial in trials:
        by={r['method']:r for r in trial['records']};a,b=by[left],by[right]
        if a['status']=='complete' and b['status']=='complete':
            groups.setdefault(trial['history_id'],[]).append(float(a['signed_margin_full_SSE']-b['signed_margin_full_SSE']))
        else:failures.append(dict(history_id=trial['history_id'],quality=trial['quality'],left_status=a['status'],right_status=b['status']))
    values={h:float(np.mean(v)) for h,v in groups.items() if len(v)==quality_count}
    array=np.asarray(list(values.values()));n=len(array)
    return dict(left=left,right=right,planned_H=len(set(t['history_id'] for t in trials)),complete_H=n,within_H_quality_count=quality_count,H_mean_effects=values,mean_effect=None if not n else float(array.mean()),between_H_descriptive_SE=None if n<2 else float(array.std(ddof=1)/np.sqrt(n)),positive_H=int((array>0).sum()),negative_H=int((array<0).sum()),zero_H=int((array==0).sum()),failed_conditions=failures,scope='predeclared heterogeneous development H; within-H quality mean first; descriptive only, no population risk or significance claim')


def summary(trials,replays,*,quality_count=3):
    n=len(trials);methods={};csv_rows=[]
    for name in METHODS:
        raw=[next(r for r in t['records'] if r['method']==name) for t in trials]
        checked=[next(r for r in t['methods'] if r['method']==name) for t in replays]
        states=Counter(r['status'] for r in raw);kinds=Counter(r['selection_kind'] for r in checked)
        complete=[r for r in checked if r['status']=='complete'];oper=[r['operations'] for r in checked if 'operations' in r and not r['operations']['diagnostic']]
        methods[name]=dict(condition_denominator=n,status_counts=dict(states),selection_counts=dict(kinds),minimum_SSE_truth_count=sum(r['minimum_SSE_is_truth'] for r in complete),complete_count=len(complete),mean_signed_margin=None if not complete else float(np.mean([r['signed_margin_full_SSE'] for r in complete])),mean_grid_charge_kwh=None if not oper else float(np.mean([r['grid_charge_kwh'] for r in oper])),mean_grid_discharge_kwh=None if not oper else float(np.mean([r['grid_discharge_kwh'] for r in oper])),mean_throughput_kwh=None if not oper else float(np.mean([r['throughput_kwh'] for r in oper])),mean_loss_kwh=None if not oper else float(np.mean([r['loss_kwh'] for r in oper])),selected_sites=dict(Counter(str(r['action']['site']) for r in raw if r.get('action') is not None)),mean_curve_kw=None if not oper else np.mean([r['curve_kw'] for r in oper],axis=0).tolist())
    for trial,replay in zip(trials,replays):
        raw={r['method']:r for r in trial['records']}
        for row in replay['methods']:
            r=raw[row['method']];op=row.get('operations',{});selected=r.get('selection',trial['selection']['methods'][row['method']])
            csv_rows.append(dict(history_id=trial['history_id'],case=trial['case'],quality=trial['quality'],method=row['method'],status=row['status'],selection_kind=row['selection_kind'],selection_status=row['selection_status'],site=None if r.get('action') is None else r['action']['site'],signed_margin_full_SSE=row.get('signed_margin_full_SSE'),minimum_SSE_is_truth=row.get('minimum_SSE_is_truth'),predicted_score=row.get('predicted_score'),requested_random_site=trial['selection'].get('random_site') if row['method']=='charge_random' else None,selection_failure_reason=selected.get('reason'),grid_charge_kwh=op.get('grid_charge_kwh'),grid_discharge_kwh=op.get('grid_discharge_kwh'),throughput_kwh=op.get('throughput_kwh'),loss_kwh=op.get('loss_kwh'),drive_kwh=op.get('drive_kwh'),terminal_energy_kwh=op.get('terminal_energy_kwh'),ac_calls=row.get('ac_calls',0),fit_calls=row.get('fit_calls',0),cache_hit=row.get('cache_hit',False),p_pu=None if r.get('action') is None else json.dumps(r['action']['p']),seconds=r.get('seconds')))
    contributions=[]
    for trial in trials:
        for tag,run in trial['selection'].get('searches',{}).items():
            for site in run.get('site_records',[]):
                if 'selected_score' in site:
                    contributions.append(dict(history_id=trial['history_id'],quality=trial['quality'],hardware=tag,site=site['site'],site_constant_score=site['constant_score'],best_optimized_seed_score=site['best_seed_score'],selected_score=site['selected_score'],seed_gain_over_constant=None if site['constant_score'] is None else site['best_seed_score']-site['constant_score'],price_gain_over_best_seed=site['gradient_gain_over_best_seed'],selected_gain_over_constant=site['gain_over_site_constant'],accepted_price_steps=sum(bool(s['accepted']) for s in site['steps'])))
    deployed=[]
    for trial in trials:
        for row in trial['records']:
            name=row['method'];kind=selection_kind(row,trial['selection'])
            if name not in INFORMATION or kind not in ('information_optimized','expanded_domain_guard'):continue
            source=row['selection'].get('from_method',name)
            tag='v2g' if source.startswith('v2g') else 'staff' if source=='charge_staff' else 'charge'
            site=row['action']['site'];run=trial['selection']['searches'][tag]
            trace=next(r for r in run['site_records'] if r['site']==site)
            deployed.append(dict(history_id=trial['history_id'],quality=trial['quality'],method=name,trace_source_method=source,expanded_guard=kind=='expanded_domain_guard',site=site,constant_score=trace['constant_score'],best_optimized_seed_score=trace['best_seed_score'],price_gain_over_best_seed=trace['gradient_gain_over_best_seed'],seed_gain_over_constant=None if trace['constant_score'] is None else trace['best_seed_score']-trace['constant_score'],selected_score=row['selection']['predicted_score']))
    contribution_summary={}
    for name in INFORMATION:
        subset=[r for r in deployed if r['method']==name]
        contribution_summary[name]=dict(planned_conditions=n,available_design_traces=len(subset),positive_price_gain_count=sum(r['price_gain_over_best_seed']>1e-6 for r in subset),mean_price_gain=None if not subset else float(np.mean([r['price_gain_over_best_seed'] for r in subset])),max_price_gain=None if not subset else float(max(r['price_gain_over_best_seed'] for r in subset)),mean_seed_gain=None if not subset else float(np.mean([r['seed_gain_over_constant'] for r in subset if r['seed_gain_over_constant'] is not None])))
    search_failures={tag:dict(Counter(f['kind'] for t in trials for f in t['selection'].get('searches',{}).get(tag,{}).get('failures',[]))) for tag in ('charge','v2g','staff')}
    result=dict(stage='independent_saved_data_development_replay',success=True,independent_H=len(set(t['history_id'] for t in trials)),quality_conditions=n,quality_conditions_per_H=quality_count,method_conditions=9*n,old_threshold_applied=False,source_file_count=21,methods=methods,paired_H_means=[grouped_pairs(trials,*pair,quality_count=quality_count) for pair in PAIRS],search_contributions=contributions,deployed_curve_contributions=deployed,method_contribution_summary=contribution_summary,search_failure_counts=search_failures,random_requested_sites=dict(Counter(str(t['selection'].get('random_site')) for t in trials)),counts=dict(saved_H_fit_copies=sum(r['history']['states_replayed'] for r in replays),unique_H_fits=sum(r['history']['states_replayed'] for r in replays if quality_count==1 or r['quality']=='uniform'),saved_joint_fit_states=sum(r['saved_fit_states_replayed']-r['history']['states_replayed'] for r in replays),unique_joint_fit_states=sum(r['new_unique_joint_fit_states_replayed'] for r in replays),physical_response_states=sum(r['physical_states_replayed'] for r in replays),actual_algorithm_AC_calls=None if any(t['total_ac_calls'] is None for t in trials) else sum(t['total_ac_calls'] for t in trials),known_algorithm_AC_calls_sum=sum(t['total_ac_calls'] for t in trials if t['total_ac_calls'] is not None),histories_with_unknown_algorithm_AC_counts=sum(t['total_ac_calls'] is None for t in trials),actual_algorithm_AC_scope='H fitting once unless actually recomputed, all planning and new joint candidate fits; evaluator truth generation excluded'),all_stage_gates_passed=all(t.get('gate_passed',False) for t in trials),case_H_counts=dict(Counter(next(t['case'] for t in trials if t['history_id']==h) for h in sorted(set(t['history_id'] for t in trials)))),whole_research_complete=False,scope='6 fixed development H or2-H pilot; minimum-SSE ranking is a diagnostic, no calibrated topology rejection/probability; mean prediction and declared public quality; staff means additional external connection windows, setup0; no joint worker routes')
    return result,csv_rows


def analysis_text(result):
    lines=['# EV灵活性证据：独立回算与比较','',f"{result['independent_H']}个独立历史、{result['quality_conditions']}质量条件、{result['method_conditions']}方法条件。每H有{result['quality_conditions_per_H']}个质量条件，先在同H内平均，配对分母按H计。旧校准阈值未使用；这里只报告开发数据的最小SSE排序和有符号裕量。",'', '| 方法 | AC完成/条件 | 真候选最小SSE | 信息优化/guard/回退/未启用 | 平均充/放电kWh |','|---|---:|---:|---|---:|']
    for name,m in result['methods'].items():
        c=m['selection_counts'];a=m['mean_grid_charge_kwh'];b=m['mean_grid_discharge_kwh']
        energy='诊断' if a is None else f'{a:.4f}/{b:.4f}'
        lines.append(f"| {name} | {m['complete_count']}/{m['condition_denominator']} | {m['minimum_SSE_truth_count']} | {c.get('information_optimized',0)}/{c.get('expanded_domain_guard',0)}/{c.get('fallback_after_design_failure',0)}/{c.get('not_activated',0)} | {energy} |")
    lines += ['', '同终端电池服务是公平比较口径；移动和充放电损耗会改变网侧电量。恒充、优化初值、后续价格步贡献保存在summary的search_contributions。信息优化成功仅指局部代理设计得到可行曲线，最终AC裕量另行评价。', '', '| 配对（前者减后者） | 完整H/计划H | 每H平均裕量差 | 正/负/零H |','|---|---:|---:|---|']
    for p in result['paired_H_means']:
        v='未取得' if p['mean_effect'] is None else f"{p['mean_effect']:.6f}"
        lines.append(f"| {p['left']} − {p['right']} | {p['complete_H']}/{p['planned_H']} | {v} | {p['positive_H']}/{p['negative_H']}/{p['zero_H']} |")
    lines += ['', '| 最终选用曲线 | 设计追踪/条件 | 初值平均改进 | 价格步平均/最大改进 | 价格步有正改进 |','|---|---:|---:|---:|---:|']
    for name,c in result['method_contribution_summary'].items():
        seed='—' if c['mean_seed_gain'] is None else f"{c['mean_seed_gain']:.6f}"
        price='—' if c['mean_price_gain'] is None else f"{c['mean_price_gain']:.6f}/{c['max_price_gain']:.6f}"
        lines.append(f"| {name} | {c['available_design_traces']}/{c['planned_conditions']} | {seed} | {price} | {c['positive_price_gain_count']} |")
    lines += ['', '这些H的案例预先固定，表中均值和H间离散程度用于开发判断。方法是否稳定改善排错及误拒，仍需冻结策略后的新独立校准和评价。所有失败保留在原trial与CSV，失败条件不会变成额外独立样本。', '', 'EV与同接入点ESS采用相同电池能力、同注入曲线时，AC响应相同；本批ESS为固定V2G响应的等价别名。人员合作只扩大外部接入窗口，接线时间为0的理想条件已声明。未实施联合人员路线。P/Q控制只改变P，Q=0；不能据这些结果声称所有X方向都获得强信息。', '', f"独立重放：保存H拟合副本{result['counts']['saved_H_fit_copies']}（独立H拟合{result['counts']['unique_H_fits']}），保存联合拟合{result['counts']['saved_joint_fit_states']}，实际唯一联合拟合{result['counts']['unique_joint_fit_states']}，真实响应{result['counts']['physical_response_states']}。AC调用计数仅含算法拟合与规划，排除仿真真值生成。", '']
    return '\n'.join(lines)


DROP_SELECTION_DIAGNOSTICS=frozenset(('local_scaled_singular_values','local_projected_shared_singular_values'))


def compact_trial(value,*,path=(),removed=None,selection_only=False):
    """Remove only the two named long SVD diagnostics.

    All candidate-fit states, all other fit diagnostics, power,
    source, physical, bounds, ranks, gradients and failures are retained.
    """
    removed=[] if removed is None else removed
    if isinstance(value,dict):
        result={}
        for key,child in value.items():
            child_path=path+(key,)
            if key in DROP_SELECTION_DIAGNOSTICS and (not selection_only or path and path[0]=='selection'):
                removed.append(dict(path='/'.join(map(str,child_path)),sha256=hashlib.sha256(json.dumps(child,sort_keys=True,separators=(',',':')).encode()).hexdigest()))
            else:result[key]=compact_trial(child,path=child_path,removed=removed,selection_only=selection_only)
        return result
    if isinstance(value,list):return [compact_trial(child,path=path+(i,),removed=removed,selection_only=selection_only) for i,child in enumerate(value)]
    return value


def export_compact(directory,output):
    """Create a new compact export, retaining original files and their SHA."""
    directory,output=Path(directory),Path(output)
    require(directory.resolve()!=output.resolve(),'compact must use a new directory')
    output.mkdir(parents=True,exist_ok=True);rows=[]
    snapshot='source_snapshot.json.gz' if (directory/'source_snapshot.json.gz').exists() else 'science_source_snapshot.json.gz'
    for name in ('freeze.json',snapshot):
        destination=output/name
        if destination.exists():require(sha(destination)==sha(directory/name),'different prior compact source file')
        else:shutil.copyfile(directory/name,destination)
    freeze=read_json(directory/'freeze.json')
    paths=sorted(directory.glob('trial_*.json.gz')) if freeze.get('schema')!='ev_service_formal_v1' else sorted(p for phase in ('calibration','evaluation') for p in (directory/phase).glob('trial_*.json.gz'))
    for path in paths:
        removed=[];data=compact_trial(read_json(path),removed=removed)
        encoded=gzip.compress((json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode(),mtime=0)
        relative=path.relative_to(directory);target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():require(target.read_bytes()==encoded,'different prior compact trial; preserve and stop')
        else:target.write_bytes(encoded)
        require(read_json(target)==data,'compact serialization changed trial fields')
        rows.append(dict(name=relative.as_posix(),raw_sha256=sha(path),raw_bytes=path.stat().st_size,compact_sha256=sha(target),compact_bytes=target.stat().st_size,removed=removed))
    require(rows,'no original trials for compact export')
    manifest=dict(schema='EV_diagnostic_compact_v2',raw_directory=str(directory.resolve()),field_policy='only2 named long singular-value diagnostics in all logs; all fit state, likelihood, rank summary, 2x2 information, operation and gradient fields retained exactly',files=rows,raw_total_bytes=sum(r['raw_bytes'] for r in rows),compact_total_bytes=sum(r['compact_bytes'] for r in rows),original_files_unchanged=True)
    write_json(output/'compact_manifest.json',manifest)
    return manifest


def verify_compact_manifest(directory):
    path=directory/'compact_manifest.json'
    if not path.exists():return None
    manifest=read_json(path);require(manifest['schema'] in ('EV_selection_diagnostic_compact_v1','EV_diagnostic_compact_v2'),'unknown compact schema')
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('trial_*.json.gz')};require(actual=={r['name'] for r in manifest['files']},'compact manifest Cartesian file coverage')
    raw=Path(manifest['raw_directory']);verified=0
    for row in manifest['files']:
        target=directory/row['name'];require(sha(target)==row['compact_sha256'],'compact bytes differ from manifest')
        original=raw/row['name']
        if original.exists():
            require(sha(original)==row['raw_sha256'],'original raw trial changed')
            removed=[];expected=compact_trial(read_json(original),removed=removed,selection_only=manifest['schema']=='EV_selection_diagnostic_compact_v1')
            require(expected==read_json(target) and removed==row['removed'],'compact changed fields outside declared diagnostic policy')
            verified+=1
    return dict(success=True,manifest_sha256=sha(path),file_count=len(actual),raw_files_available_and_verified=verified,raw_total_bytes=manifest['raw_total_bytes'],compact_total_bytes=manifest['compact_total_bytes'],lossless_other_fields=True,raw_SHA_retained=True)


def compare_scientific_summaries(left,right):
    ignored={'input_sha256','seconds','compact_manifest'}
    a={k:v for k,v in left.items() if k not in ignored};b={k:v for k,v in right.items() if k not in ignored}
    require(a==b,'compact versus original scientific results differ')
    return dict(success=True,scientific_payload_exactly_equal=True,ignored_metadata=sorted(ignored))

def audit_directory(directory,*,count=6,output=None,phase=None):
    directory=Path(directory);freeze=read_json(directory/'freeze.json');began=perf_counter()
    if phase is None:require(count in (2,6),'declared2-H pilot or6-H development only')
    require(phase in (None,'calibration','evaluation'),'valid formal phase')
    data=directory if phase is None else directory/phase
    paths=sorted(data.glob('trial_*.json.gz'));trials=[read_json(p) for p in paths]
    compact=verify_compact_manifest(directory)
    require(paths,'no saved trials')
    if phase is None:validate_coverage(trials,freeze,count)
    else:validate_formal_coverage(trials,freeze,phase)
    sources=source_audit(directory,freeze)
    replays=[audit_trial(t) for t in trials];result,csv_rows=summary(trials,replays,quality_count=3 if phase is None else 1)
    result['phase']=phase or 'development';result['source_file_count']=sources['source_count']
    if phase is not None:
        result['scope']='formal single-quality saved-data numeric replay; calibrated retained sets and risk are evaluated separately by the frozen formal wrapper'
        for pair in result['paired_H_means']:pair['scope']='one quality per independent H in declared frozen simulation mixture; saved-margin comparison, not a topology probability'
    result['source_audit']=sources;result['input_sha256']={p.name:sha(p) for p in paths};result['freeze_sha256']=sha(directory/'freeze.json')
    result['replay_module_sha256']=sha(Path(__file__));result['compact_manifest']=compact;result['seconds']=perf_counter()-began
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        write_json(output/'summary.json',result);write_json(output/'independent_replay.json.gz',dict(success=True,optimizer_invoked=False,trials=replays,counts=result['counts'],input_sha256=result['input_sha256'],freeze_sha256=result['freeze_sha256'],replay_module_sha256=result['replay_module_sha256']))
        with (output/'methods.csv').open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
        (output/'analysis.md').write_text(analysis_text(result),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,default=HERE/'evidence/ev_flexibility/v1_development');parser.add_argument('--count',type=int,default=6);parser.add_argument('--output',type=Path);parser.add_argument('--compact-output',type=Path);parser.add_argument('--phase',choices=('calibration','evaluation'))
    args=parser.parse_args()
    directory=args.directory
    if args.compact_output is not None:
        export_compact(directory,args.compact_output);directory=args.compact_output
    output=args.output or (directory/'independent' if args.phase is None else directory/args.phase/'independent');result=audit_directory(directory,count=args.count,output=output,phase=args.phase)
    print(json.dumps(dict(success=result['success'],independent_H=result['independent_H'],quality_conditions=result['quality_conditions'],counts=result['counts'],output=str(output)),ensure_ascii=False))


if __name__=='__main__':main()
