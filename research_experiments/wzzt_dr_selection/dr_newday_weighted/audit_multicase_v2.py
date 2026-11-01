"""Independent post-freeze physical/billing audit of a bounded multi-case DR test.

No identifier, optimizer, physical producer or project selector is imported.
Private arrays are opened only after observations, predictive selections and
scored ledgers all exist. Static topology supports are hash-checked, not parsed.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,pickle,traceback
from pathlib import Path
import numpy as np

REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
TOL=1.e-7

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stamp(value):return datetime.fromisoformat(value.replace('Z','+00:00'))
def reference(item,parent=None):
    if not isinstance(item,dict) or not {'path','sha256'}<=set(item):raise ValueError('path/hash reference required')
    path=Path(item['path']);path=(path if path.is_absolute() else Path(parent or '.')/path).resolve()
    if sha(path)!=item['sha256']:raise AssertionError('bound artifact hash drift: '+str(path))
    return path
def unpickle(path):
    with Path(path).open('rb') as stream:return pickle.load(stream)
def vector(value):return np.broadcast_to(np.asarray(value,float),(96,)).copy()
def fingerprint(arrays):
    h=hashlib.sha256()
    for value in arrays:
        a=np.ascontiguousarray(value,float);h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def resource_check(dto,delta,check,key):
    base=vector(dto['baseline_kw']);prefix=.25*np.cumsum(delta)
    check(key+'_net_energy',abs(float(prefix[-1]))<=TOL)
    check(key+'_power_bounds',np.max(vector(dto.get('power_lower_kw',0.))-base-delta)<=TOL
          and np.max(base+delta-vector(dto['power_upper_kw']))<=TOL)
    check(key+'_energy_bounds',np.max(vector(dto['prefix_delta_lower_kwh'])-prefix)<=TOL
          and np.max(prefix-vector(dto['prefix_delta_upper_kwh']))<=TOL)
    avail=np.broadcast_to(np.asarray(dto.get('available',True),bool),(96,))
    check(key+'_availability',np.max(np.abs(delta[~avail]),initial=0.)<=TOL)
    battery=dto.get('battery')
    if battery:
        e=battery['initial_kwh']+battery.get('eta_charge',1.)*.25*np.cumsum(base+delta)
        check(key+'_battery_soc',e.min()>=battery.get('soc_min',0.)*battery['capacity_kwh']-TOL
              and e.max()<=battery.get('soc_max',1.)*battery['capacity_kwh']+TOL)
        check(key+'_battery_terminal',abs(e[-1]-battery['terminal_kwh'])<=TOL)
        if dto.get('service_deadline_slot') is not None:
            check(key+'_battery_deadline',abs(e[dto['service_deadline_slot']-1]-battery['terminal_kwh'])<=TOL)


def plan_settlement(plan,z,expected_nodes,check,prefix):
    """Rebuild actual amplitudes, device checks and fees without runtime code."""
    mean=np.asarray(plan['amplitudes'],float);prices=np.asarray(plan['prices'],float)
    lam=np.asarray(plan['elasticities'],float);d=np.asarray(plan['response_factors'],float)
    modules=plan['modules'];resources={r['resource_id']:r for r in plan['resources']}
    n=len(modules);z=np.asarray(z,float)
    check(prefix+'_contract_dimensions',mean.shape==prices.shape==lam.shape==(n,) and d.shape==(n,2) and z.shape==(2,))
    check(prefix+'_finite_domain',all(np.isfinite(v).all() for v in [mean,prices,lam,d,z])
          and np.min(mean,initial=0.)>=0 and np.min(prices,initial=0.)>=0 and np.max(prices,initial=0.)<=2+1e-10
          and np.min(lam,initial=1.)>0 and np.max(np.abs(z),initial=0.)<=1)
    check(prefix+'_declared_nodes',set(r['node_label'] for r in resources.values())==set(expected_nodes))
    check(prefix+'_mean_price_law',np.allclose(mean,lam*prices,atol=1e-12,rtol=0))
    check(prefix+'_nonnegative_response_law',np.all(np.abs(d).sum(axis=1)<=1))
    actual=mean*(1+d@z);grouped={key:np.zeros(96) for key in resources};quotes={};hours=[]
    units={}
    for key,r in resources.items():
        base=vector(r['baseline_kw']);number=float(base[0]/.2)
        check(prefix+'_'+key+'_declared_aggregate',np.all(base==base[0]) and 1<=number<=5
              and abs(number-round(number))<1e-10 and np.allclose(vector(r['power_upper_kw']),2*base,rtol=0,atol=1e-12)
              and np.allclose(vector(r['prefix_delta_lower_kwh']),-2*base,rtol=0,atol=1e-12)
              and np.allclose(vector(r['prefix_delta_upper_kwh']),2*base,rtol=0,atol=1e-12))
        units[key]=int(round(number))
    for j,module in enumerate(modules):
        check(prefix+f'_module{j}_one_resource',len(module['profiles'])==1)
        key,shape=next(iter(module['profiles'].items()));shape=np.asarray(shape,float)
        check(prefix+f'_module{j}_shape',key in resources and shape.shape==(96,) and np.isfinite(shape).all() and abs(shape.sum())<1e-10)
        check(prefix+f'_module{j}_inventory_elasticity',abs(lam[j]-.1*units[key])<1e-12
              and abs(module['amplitude_upper_kw']-.2*units[key])<1e-12)
        check(prefix+f'_module{j}_actual_amplitude',actual[j]>=-TOL and actual[j]<=module['amplitude_upper_kw']+TOL)
        if mean[j]>0:
            check(prefix+f'_module{j}_exclusive',key not in quotes);quotes[key]=prices[j]
        grouped[key]+=shape*actual[j];hours.append(float(.25*np.maximum(-shape,0.).sum()))
    hours=np.asarray(hours)
    fee=float((hours*prices)@actual);expected=float((hours*prices)@mean)
    hard=float(np.sum(hours*lam*(1+np.abs(d).sum(axis=1))*prices**2))
    check(prefix+'_hard_budget',fee>=-TOL and fee<=hard+1e-9 and hard<=1.6+1e-9)
    for key,r in resources.items():resource_check(r,grouped[key],check,prefix+'_actual_'+key)
    netfee=sum(float(.25*np.maximum(-grouped[key],0.).sum()*quotes.get(key,0.)) for key in resources)
    check(prefix+'_independent_net_fee',abs(netfee-fee)<1e-9)
    for corner in [(-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)]:
        amp=mean*(1+d@np.asarray(corner));cg={key:np.zeros(96) for key in resources}
        for j,module in enumerate(modules):
            key,shape=next(iter(module['profiles'].items()));cg[key]+=np.asarray(shape)*amp[j]
            check(prefix+f'_corner{corner}_amp{j}',amp[j]>=-TOL and amp[j]<=module['amplitude_upper_kw']+TOL)
        for key,r in resources.items():resource_check(r,cg[key],check,prefix+f'_corner{corner}_'+key)
        cf=sum(float(.25*np.maximum(-cg[key],0.).sum()*quotes.get(key,0.)) for key in resources)
        check(prefix+f'_corner{corner}_budget',cf<=hard+1e-9)
    return dict(resources=resources,grouped=grouped,actual=actual,factors=z,fee=fee,
                expected_fee=expected,hard_fee=hard,inventory_units=units)


def audit(root):
    root=Path(root).resolve();folder=root/'actual'
    needed=[folder/n for n in ['observations_frozen.json','predictions_frozen.json','scored.json']]
    if any(not p.is_file() for p in needed):
        return dict(status='waiting_for_frozen_predictions',missing=[p.name for p in needed if not p.exists()],
            private_arrays_opened=False,static_truth_supports_parsed=False,new_AC=0,new_solver_calls=0,new_fits=0)
    checks=[]
    def check(name,condition):
        checks.append(dict(name=name,passed=bool(condition)))
        if not condition:raise AssertionError(name)
    c=read(root/'protocol.json');f=read(root/'price_freeze.json')
    observations=read(needed[0]);projections=read(needed[1]);scored=read(needed[2])
    check('master_frozen',c['status']=='frozen' and f['status']=='frozen')
    if 'protocol' in f:check('master_protocol_binding',reference(f['protocol'])==root/'protocol.json')
    for p,h in c['source_sha256'].items():check('source_'+Path(p).name,sha(p)==h)
    if 'price_freeze' in observations:check('observation_quote_binding',reference(observations['price_freeze'])==root/'price_freeze.json')
    if 'observations_frozen' in projections:check('projection_observation_binding',reference(projections['observations_frozen'])==needed[0])
    if 'predictions_frozen' in scored:check('score_prediction_binding',reference(scored['predictions_frozen'])==needed[1])
    check('all_ledgers_frozen_before_audit',observations['status']=='frozen' and projections['status']=='frozen')
    check('same_record_denominators',len(observations['rows'])==len(projections['rows'])==len(scored['rows']))
    projections_by={r['record_id']:r for r in projections['rows']};scored_by={r['record_id']:r for r in scored['rows']}
    check('unique_record_ids',len(projections_by)==len(scored_by)==len(observations['rows']))
    registry={};histories=[]
    for case in c['cases']:
        name=case['name'];runtime=read(reference(case['runtime_protocol']));pool=read(reference(case['pool']))
        source=read(reference(case['case_protocol']));reference(case['design']);qualification=read(reference(case['qualification']))
        reference(case['problem'])
        check(name+'_runtime_frozen',runtime['status']=='frozen' and runtime['case']==case['case'])
        check(name+'_original_noise',runtime['voltage_sigma']==source['voltage_sigma'] and runtime['pq_sigma']==source.get('pq_sigma',.005)==.005)
        check(name+'_same_bases_labels',runtime['base_kw']==pool['base_kw']==source['base_kw']
              and runtime['root_label']==pool['root_label']==source['root_label'] and runtime['labels']==pool['labels']==source['labels'])
        check(name+'_full_history',pool['train_rows']==pool['validation_rows']==288 and pool['new_fit_calls']==pool['new_search_calls']==0)
        raw=unpickle(reference(pool['history_packet']));history=raw['history'] if 'history' in raw else raw
        check(name+'_six_full_days',len(history['training'])==len(history['validation'])==3
              and all(len(s['P_terminal'])==96 for s in history['training']+history['validation']))
        labels=runtime['labels'];base=runtime['base_kw'];pm=np.concatenate([s['P_terminal'].to_numpy() for s in history['training']+history['validation']])*base
        qm=np.concatenate([s['Q_terminal'].to_numpy() for s in history['training']+history['validation']])*base
        models=[]
        for m in pool['models']:
            matrices=[]
            for key in ('r','x'):
                matrix=np.zeros((len(labels),len(labels)))
                for block,value in zip(m['point']['supports'],m['point'][key],strict=True):matrix[np.ix_(block,block)]+=value
                matrices.append(matrix)
            check(name+'_'+m['candidate_id']+'_parameter_matrix',np.array_equal(matrices[0],np.asarray(m['R'])) and np.array_equal(matrices[1],np.asarray(m['X'])))
            models.append(matrices)
        dr,dx=models[1][0]-models[0][0],models[1][1]-models[0][1]
        plans={policy:read(reference(item)) for policy,item in f['cases'][name]['plans'].items()}
        check(name+'_policy_scope',set(plans)=={'zero','rotation','optimized'})
        check(name+'_declared_sites',set(runtime['expected_resource_nodes'])==set(qualification['selected_labels']))
        for policy,plan in plans.items():
            check(name+'_'+policy+'_plan_sites',set(r['node_label'] for r in plan['resources'])==set(runtime['expected_resource_nodes']))
            for dto in plan['resources']:
                i=labels.index(dto['node_label']);bp=vector(dto['baseline_kw']);ratio=vector(dto.get('reactive_ratio',.35))
                check(name+'_'+policy+'_'+dto['resource_id']+'_past_containment',pm[:,i].min()>=bp.max()-1e-10 and qm[:,i].min()>=(bp*ratio).max()-1e-10)
        histories.append(dict(condition=name,case=case['case'],history_training_rows=288,history_validation_rows=288,
            nodes=runtime['expected_resource_nodes'],voltage_sigma=runtime['voltage_sigma']))
        registry[name]=dict(case=case,runtime=runtime,pool=pool,plans=plans,dr=dr,dx=dx,labels=labels,base=base)
    expected={(name,policy,path) for name,cfg in registry.items() for policy in cfg['plans'] for path in cfg['runtime']['paths']}
    check('complete_predeclared_records',len(observations['rows'])==len(expected)
          and {(r['condition'],r['policy'],r['path']) for r in observations['rows']}==expected)
    rows=[];natural={};noise={};backgrounds={};ac_attempted=0;valid_points=0
    for ledger in observations['rows']:
        rid,name,policy,path=ledger['record_id'],ledger['condition'],ledger['policy'],ledger['path']
        cfg=registry[name];runtime=cfg['runtime'];plan=cfg['plans'][policy]
        row=dict(record_id=rid,condition=name,case=cfg['case']['case'],policy=policy,path=path,status=ledger['status'],
            actual_payment=ledger.get('actual_payment'),known_actual_payment=ledger.get('actual_payment') or 0.,
            unknown_payment_events=0 if ledger.get('actual_payment') is not None else 1,
            AC_attempted=False,valid_observation_rows=0,frozen_distance_physical=None,frozen_distance_observed=None)
        check(rid+'_ledger_projection_present',rid in projections_by and rid in scored_by)
        if not ledger.get('receipt'):
            rows.append(row);continue
        receipt=read(reference(ledger['receipt']));request=read(reference(ledger['request']))
        check(rid+'_price_before_future',stamp(f['created_utc'])<=stamp(receipt['started_utc']))
        check(rid+'_receipt_request_hash',receipt['request_sha256_expected']==ledger['request']['sha256'])
        check(rid+'_runtime_binding',reference(request['protocol'])==reference(cfg['case']['runtime_protocol']))
        check(rid+'_quote_binding',request['plan']==f['cases'][name]['plans'][policy])
        check(rid+'_pool_binding',request['frozen_pool']==cfg['case']['pool'])
        check(rid+'_same_noise',request['voltage_sigma']==runtime['voltage_sigma'] and request['pq_sigma']==runtime['pq_sigma'])
        check(rid+'_not_refit',receipt['identifier_run'] is False and receipt['parameter_fitting_run'] is False
              and receipt['old_or_new_pool_parameters_parsed'] is False and not receipt.get('frozen_pool_parsed',False))
        for file,bound in receipt.get('private_files',{}).items():check(rid+'_private_hash_'+file,sha(Path(ledger['physical_output'])/file)==bound['sha256'])
        z=np.asarray(request['factors'],float)
        expected_z=np.random.default_rng(np.random.SeedSequence([runtime['response_seed'],path])).uniform(-1,1,2)
        check(rid+'_prelocked_response_seed',np.array_equal(z,expected_z))
        check(rid+'_prelocked_physical_replicate',path in runtime['paths'] and request['replicate']==runtime['replicate_base']+10*path)
        settled=plan_settlement(plan,z,runtime['expected_resource_nodes'],check,rid)
        row.update(known_actual_payment=receipt.get('known_actual_payment',row['known_actual_payment']),
            unknown_payment_events=receipt.get('unknown_payment_events',row['unknown_payment_events']),
            actual_payment=receipt.get('actual_payment'),AC_attempted=receipt.get('AC_attempted',False))
        if receipt.get('actual_payment') is not None:
            check(rid+'_independent_fee',abs(receipt['actual_payment']-settled['fee'])<1e-9)
            check(rid+'_hard_fee',abs(receipt['worst_payment']-settled['hard_fee'])<1e-9)
        ac_attempted+=int(receipt.get('AC_attempted',False))
        pp=Path(ledger['physical_output'])/'physical_truth.pkl'
        if not pp.exists():rows.append(row);continue
        private=unpickle(pp)
        if 'natural_P_pu' not in private:rows.append(row);continue
        p0,q0,root0=np.asarray(private['natural_P_pu']),np.asarray(private['natural_Q_pu']),np.asarray(private['root_pu'])
        minimum_p=minimum_q=float('inf');labels=cfg['labels'];base=cfg['base']
        for dto in settled['resources'].values():
            i=labels.index(dto['node_label']);bp=vector(dto['baseline_kw']);bq=vector(dto.get('reactive_ratio',.35))*bp
            minimum_p=min(minimum_p,float((p0[:,i]*base-bp).min()));minimum_q=min(minimum_q,float((q0[:,i]*base-bq).min()))
        component=minimum_p>=-1e-8 and minimum_q>=-1e-8
        check(rid+'_component_report',receipt.get('contained_component_pass')==component)
        if 'minimum_nonflexible_p_kw' in receipt:
            check(rid+'_component_values',abs(receipt['minimum_nonflexible_p_kw']-minimum_p)<1e-10 and abs(receipt['minimum_nonflexible_q_kvar']-minimum_q)<1e-10)
        key=(name,path);shared=(p0,q0,root0,z)
        if key in natural:check(rid+'_natural_response_CRN',all(np.array_equal(a,b) for a,b in zip(shared,natural[key],strict=True)))
        else:natural[key]=shared;backgrounds[key]=fingerprint(shared[:3])
        row.update(component_supported=component,minimum_nonflexible_P_kw=minimum_p,minimum_nonflexible_Q_kvar=minimum_q)
        if receipt['status']!='complete':rows.append(row);continue
        check(rid+'_successful_component',component)
        packet=unpickle(reference(ledger['packet']))
        check(rid+'_newday_only',len(packet['validation'])==1 and len(packet['validation'][0]['P_terminal'])==96 and 'training' not in packet)
        scene=packet['validation'][0];p,q=np.asarray(private['P_pu']),np.asarray(private['Q_pu']);dp=np.zeros_like(p);dq=np.zeros_like(q)
        for resource,delta in settled['grouped'].items():
            dto=settled['resources'][resource];i=labels.index(dto['node_label']);dp[:,i]+=delta;dq[:,i]+=vector(dto.get('reactive_ratio',.35))*delta
        check(rid+'_active_increment_once',np.allclose((p-p0)*base,dp,atol=1e-9,rtol=0) and np.allclose(dp,private['delta_kw'],atol=1e-12,rtol=0))
        check(rid+'_reactive_increment_once',np.allclose((q-q0)*base,dq,atol=1e-9,rtol=0) and np.allclose(dq,private['delta_kvar'],atol=1e-12,rtol=0))
        check(rid+'_actual_amplitude',np.allclose(private['actual_amplitudes'],settled['actual'],atol=1e-12,rtol=0))
        operation=receipt['checks']
        check(rid+'_AC_and_voltage',operation['ac_all_96_converged'] and operation['electrical_equations_pass'] and operation['voltage_limits_pass']
              and operation['branch_ampacity_checked'] is False)
        measured=[scene['P_terminal'].to_numpy(),scene['Q_terminal'].to_numpy(),scene['V_terminal'].to_numpy(),scene['V_root_observed'].to_numpy()]
        true=[p,q,np.asarray(private['V_pu']),root0]
        relative=[np.divide(a-b,b,out=np.zeros_like(b,dtype=float),where=np.abs(b)>1e-12) for a,b in zip(measured,true,strict=True)]
        if key in noise:check(rid+'_measurement_CRN',all(np.allclose(a,b,atol=5e-14,rtol=0) for a,b in zip(relative,noise[key],strict=True)))
        else:noise[key]=relative
        def distance(ap,aq):return float(np.abs(np.asarray(ap)@cfg['dr'].T+np.asarray(aq)@cfg['dx'].T).mean())
        qphys,qobs=distance(p,q),distance(measured[0],measured[1]);score=read(reference(projections_by[rid]['scores']))
        reported=score.get('new_prediction_pair_distance_l1',score.get('new_distance'))
        if reported is None:reported=projections_by[rid].get('new_distance')
        check(rid+'_observed_distance_replay',reported is not None and abs(qobs-reported)<1e-12)
        valid_points+=96;row.update(valid_observation_rows=96,frozen_distance_physical=qphys,frozen_distance_observed=qobs,
            natural_physical_distance=distance(p0,q0),status='complete')
        rows.append(row)
    for row in rows:
        zero=next((r for r in rows if (r['condition'],r['path'],r['policy'])==(row['condition'],row['path'],'zero')),None)
        if zero and row['frozen_distance_physical'] is not None and zero['frozen_distance_physical'] is not None:
            gp=row['frozen_distance_physical']-zero['frozen_distance_physical'];go=row['frozen_distance_observed']-zero['frozen_distance_observed']
            bound=f['cases'][row['condition']];gamma=bound['gamma_native']
            row.update(paired_physical_distance_gain=gp,paired_observed_distance_gain=go,
                physical_positive_gain=gp>1e-12,observed_positive_gain=go>1e-12,
                physical_target_met=gp>=gamma-1e-12 if bound['positive_gamma'] else None,
                observed_target_met=go>=gamma-1e-12 if bound['positive_gamma'] else None)
    table=[]
    for name in registry:
        for policy in f['cases'][name]['plans']:
            rr=[r for r in rows if r['condition']==name and r['policy']==policy];n=len(rr)
            valid=[r for r in rr if r['frozen_distance_physical'] is not None]
            table.append(dict(condition=name,case=registry[name]['case']['case'],policy=policy,denominator=n,valid=len(valid),failed_or_unresolved=n-len(valid),
                known_payment_total=sum(r['known_actual_payment'] for r in rr),unknown_payment_events=sum(r['unknown_payment_events'] for r in rr),
                mean_payment=sum(r['actual_payment'] for r in rr)/n if n and all(r['actual_payment'] is not None for r in rr) else None,
                physical_distance_gains=[r.get('paired_physical_distance_gain') for r in rr],observed_distance_gains=[r.get('paired_observed_distance_gain') for r in rr],
                physical_positive_count=sum(r.get('physical_positive_gain',False) for r in rr),observed_positive_count=sum(r.get('observed_positive_gain',False) for r in rr),
                physical_target_count=sum(r.get('physical_target_met') is True for r in rr),observed_target_count=sum(r.get('observed_target_met') is True for r in rr)))
    return dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,check_count=len(checks),passed_checks=len(checks),failures=[],
        rows=rows,table=table,historical_conditions=histories,condition_count=len(registry),requested_records=len(observations['rows']),
        completed_records=sum(r['status']=='complete' for r in rows),AC_attempted_days=ac_attempted,valid_AC_points=valid_points,
        distinct_natural_case_paths=len(set(backgrounds.values())),case_paths=len(backgrounds),new_AC=0,new_solver_calls=0,new_fits=0,
        static_truth_supports_parsed=False,source_sha256=sha(__file__),
        input_sha256={str(p.relative_to(root)):sha(p) for p in [root/'protocol.json',root/'price_freeze.json',*needed]},
        scope='independent physical/billing/frozen-distance replay; no topology success probability or all-tree coverage',
        boundaries=['Repeated histories of one network are not distinct physical networks.',
            'Inventory and whole-day response factors are declared synthetic, not individual user calibration.',
            'Thermal ampacity is not checked by the inherited physical checker.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve original independent audit')
    try:result=audit(a.root)
    except Exception as error:
        result=dict(status='audit_failed',error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),new_AC=0,new_solver_calls=0,new_fits=0,source_sha256=sha(__file__))
    if result['status'].startswith('waiting'):
        print(json.dumps(result));return 3
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result.get(k) for k in ['status','check_count','completed_records','valid_AC_points','message']}))
    return 0 if result['status']=='complete' else 2

if __name__=='__main__':raise SystemExit(main())
