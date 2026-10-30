"""Coded DR on a fixed simulated non-V2G task, keeping natural P/Q unchanged.

Pure template/device functions run without physical data. The evaluator is
price-hash guarded and reuses the original physical producer only after freeze.
Device feasibility is a declared simulation model, not empirical EV calibration.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_SOURCE=Path(r'D:\0-github_workspace\Topo\outputs\wzzt_paper_20261029_full\source')
DEVICE=dict(model='fixed_simulated_nonV2G_task',baseline_kw=.2,baseline_kvar=.07,
            power_lower_kw=0.,power_upper_kw=.4,window_start_hour=19.,deadline_hour=21.,
            initial_energy_kWh=0.,terminal_energy_kWh=.4,energy_lower_kWh=0.,
            energy_upper_kWh=.4,efficiency=1.,interval_hours=.25,
            Q_increment_over_P_increment=.35,empirically_calibrated=False,
            decomposition='strict nonnegative residual P/Q in the device window')
WINDOW=np.arange(76,84)
_LEGACY=None


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _legacy(repo_root):
    global _LEGACY
    path=Path(repo_root).resolve()/'research_experiments/wzzt_dr_selection/dr_competition_bridge/physical_bridge.py'
    if _LEGACY is None or Path(_LEGACY.__file__).resolve()!=path:
        spec=importlib.util.spec_from_file_location('_coded_original_physical_bridge',path)
        if spec is None or spec.loader is None:raise RuntimeError('original physical bridge not found')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);_LEGACY=module
    return _LEGACY


def waveforms(template,directions,active_positions):
    """directions signs map explicitly; alternating is only within active pair."""
    d=np.asarray(directions,int)
    if d.ndim!=1 or not np.isin(d,[-1,0,1]).all():raise ValueError('explicit per-node direction required')
    if template not in ('same','opposed','alternating_opposed'):raise ValueError('unknown coded template')
    base=np.tile(np.r_[np.full(4,-.2),np.full(4,.2)][:,None],(1,len(d)))
    if template=='alternating_opposed':
        base[:,list(active_positions)]=np.array([-.2,.2]*4)[:,None]
    return base*d[None,:]


def device_precheck(p_pu,q_pu,offered,delta_kw,*,base_kw=200.,declaration=DEVICE):
    """Offered-node diagnostics; execution additionally requires accepted nodes.

    Missing declarations are unresolved. A strict decomposition failure at a DER
    terminal means this demand-only simulation declaration does not apply there;
    it does not prove that a real DER-equipped terminal cannot provide DR.
    """
    p,q=np.asarray(p_pu,float),np.asarray(q_pu,float)
    offered=np.asarray(offered,bool);delta=np.asarray(delta_kw,float)
    if p.shape!=(96,len(offered)) or q.shape!=p.shape or delta.shape!=(8,len(offered)):
        raise ValueError('96-point natural P/Q and eight-point device delta required')
    required=('baseline_kw','baseline_kvar','power_lower_kw','power_upper_kw','initial_energy_kWh',
              'terminal_energy_kWh','energy_lower_kWh','energy_upper_kWh','interval_hours','efficiency')
    if declaration is None or any(k not in declaration for k in required):
        return dict(status='unresolved',node_status=['unresolved' if v else 'not_offered' for v in offered],
                    device_declaration_complete=False,quote_qualification_pass=False)
    if declaration['efficiency']!=1.:
        return dict(status='unresolved',node_status=['unresolved' if v else 'not_offered' for v in offered],
                    reason='this fixed task does not implement efficiency conversion',quote_qualification_pass=False)
    b=float(declaration['baseline_kw']);q0=float(declaration['baseline_kvar']);dt=float(declaration['interval_hours'])
    power=b+delta;energy=float(declaration['initial_energy_kWh'])+np.cumsum(power,axis=0)*dt
    pmin=(p[WINDOW]*base_kw).min(axis=0);qmin=(q[WINDOW]*base_kw).min(axis=0)
    decomposition=(pmin>=b-1e-12)&(qmin>=q0-1e-12)
    power_ok=(power>=declaration['power_lower_kw']-1e-12).all(axis=0)&(power<=declaration['power_upper_kw']+1e-12).all(axis=0)
    energy_ok=(energy>=declaration['energy_lower_kWh']-1e-12).all(axis=0)&(energy<=declaration['energy_upper_kWh']+1e-12).all(axis=0)
    terminal_ok=abs(energy[-1]-declaration['terminal_energy_kWh'])<=1e-12
    available=decomposition&power_ok&energy_ok&terminal_ok
    statuses=['available' if good else 'not_available' for good in available]
    statuses=[value if quoted else 'not_offered' for value,quoted in zip(statuses,offered)]
    return dict(status='available' if all((~offered)|available) else 'not_available',
        node_status=statuses,device_declaration_complete=True,quote_qualification_pass=bool(((~offered)|available).all()),
        min_natural_window_P_kw=pmin.tolist(),min_natural_window_Q_kvar=qmin.tolist(),
        decomposition_pass=decomposition.tolist(),device_power_pass=power_ok.tolist(),
        device_energy_pass=energy_ok.tolist(),terminal_energy_pass=terminal_ok.tolist(),
        declared_device_baseline_kw=b,declared_device_baseline_kvar=q0,
        device_total_energy_kWh=energy[-1].tolist(),strict_residual_nonnegative=True,
        field_device_feasibility='unknown; synthetic declaration only')


def apply_coded_response(p_pu,q_pu,prices,participants,directions,template,active_positions,*,base_kw=200.,declaration=DEVICE):
    """Pure whole-event application; fails explicitly instead of changing Z."""
    p,q=np.asarray(p_pu,float),np.asarray(q_pu,float)
    price=np.asarray(prices,int);z=np.asarray(participants,bool);direction=np.asarray(directions,int)
    if price.shape!=z.shape or price.shape!=direction.shape or not np.isin(price,[0,1,2]).all():raise ValueError('price/mask/direction mismatch')
    if np.any(z&(price==0)) or np.any((price>0)&~np.isin(direction,[-1,1])):raise ValueError('invalid accepted direction or zero-price participation')
    full_delta=waveforms(template,direction,active_positions)
    qualification=device_precheck(p,q,price>0,full_delta,base_kw=base_kw,declaration=declaration)
    committed=.2*float(price@z)
    unavailable=[i for i,flag in enumerate(z) if flag and qualification['node_status'][i]!='available']
    result=dict(status='complete',quote_qualification=qualification,accepted_participants=z.astype(int).tolist(),
        committed_payment=committed,actual_payment=None,
        contracted_reduction_kWh=(.2*z).tolist(),contracted_rebound_kWh=(.2*z).tolist(),
        promised_terminal_task_kWh=(.4*z).tolist(),energy_return_pass=False,
        execution_mask_unchanged=True,no_power_clipping=True,
        actual_settlement='payment only known when full contracted template delivered',
        simulation_device_declaration=dict(DEVICE),device_feasibility_scope='synthetic contract only; actual EV feasibility unknown')
    if unavailable:
        result.update(status='contract_unresolved' if qualification['status']=='unresolved' else 'contract_not_available',
                      unavailable_accepted_positions=unavailable,executed_participants=None,
                      delivered_reduction_kWh=None,delivered_rebound_kWh=None)
        return None,None,result
    delta=full_delta*z[None,:]
    actual_p=p.copy();actual_q=q.copy()
    actual_p[WINDOW]+=delta/base_kw;actual_q[WINDOW]+=.35*delta/base_kw
    net=delta.sum(axis=0)*.25
    reduced=np.maximum(-delta,0).sum(axis=0)*.25;added=np.maximum(delta,0).sum(axis=0)*.25
    assert np.allclose(reduced,.2*z,atol=1e-12) and np.allclose(added,.2*z,atol=1e-12)
    result.update(executed_participants=z.astype(int).tolist(),delivered_reduction_kWh=reduced.tolist(),
        delivered_rebound_kWh=added.tolist(),net_energy_change_kWh=net.tolist(),
        energy_return_pass=bool(np.max(abs(net),initial=0)<=1e-12),
        max_absolute_net_power_kw=float(np.max(abs(actual_p))*base_kw),
        capacity_pass=bool((abs(actual_p)*base_kw<=45.+1e-12).all()),
        nonfinite_values=bool(not np.isfinite(actual_p).all() or not np.isfinite(actual_q).all()))
    # Actual delivery payment is made known only after physical execution checks.
    return actual_p,actual_q,result


def _lock(lock_path,expected_sha,policy,legacy):
    if sha(lock_path)!=expected_sha:raise RuntimeError('frozen price lock hash mismatch')
    lock=json.loads(Path(lock_path).read_text(encoding='utf-8'))
    if lock.get('status')!='frozen':raise RuntimeError('price lock is not frozen')
    expected_device=dict(baseline_kw=.2,baseline_kvar=.07,upper_kw=.4,start_hour=19,end_hour=21,
                         initial_energy_kWh=0.,final_energy_kWh=.4,efficiency=1.,non_V2G=True)
    declaration=lock.get('device_contract')
    if not isinstance(declaration,dict) or any(declaration.get(k)!=v for k,v in expected_device.items()):
        raise RuntimeError('missing or different fixed simulated device declaration')
    labels=[int(v) for v in lock['labels']]
    plan=legacy.validate_plan(lock['plans'][policy],len(labels))
    directions=np.asarray(lock['directions'][policy],int)
    if directions.shape!=plan.shape or not np.isin(directions,[-1,0,1]).all():raise ValueError('bad explicit direction matrix')
    if np.any((plan>0)&~np.isin(directions,[-1,1])):raise ValueError('offered node requires an explicit signed direction')
    template=lock['templates'][policy]
    if template not in ('same','opposed','alternating_opposed'):raise ValueError('bad declared template')
    active=[labels.index(int(v)) for v in lock['active_labels']]
    if len(active)!=2 or len(set(active))!=2:raise ValueError('exactly two distinct active positions required')
    return plan,directions,template,active,labels,lock


def generate_actual_events(lock_path,expected_lock_sha256,policy,*,path,dependence='local',
    voltage_sigma=.001,case='paper15',source_root=DEFAULT_SOURCE,replicate_base=20000,
    event_profiles=(0,2,0,2),participation_seed=90410000,repo_root=None):
    """Post-freeze physical future. No true network/trajectory before lock check."""
    repo_root=Path.cwd() if repo_root is None else Path(repo_root)
    legacy=_legacy(repo_root)
    plan,directions,template,active,labels,lock=_lock(lock_path,expected_lock_sha256,policy,legacy)
    if case!='paper15' or voltage_sigma not in (0.,.0002,.0005,.001):raise ValueError('outside original paper15/noise domain')
    if replicate_base<20000 or path<0 or int(path)!=path:raise ValueError('new actual replicate stream required')
    if len(event_profiles)!=4 or any(v not in (0,1,2) for v in event_profiles):raise ValueError('bad event profiles')
    sim,runtime=legacy._simulation_runtime(source_root)
    physical_checks,checks_source=legacy._physical_check_function(repo_root)
    runtime[str(checks_source)]=sha(checks_source);runtime[str(Path(legacy.__file__).resolve())]=sha(legacy.__file__)
    context=sim._prepare_case_context(case,96,scenario_suite='reference',root_observation='noisy',
        root_meter_noise_rel=voltage_sigma,root_sigma=.003,impedance_scale=1.)
    if context.terminals!=labels:raise ValueError('frozen lock terminal order differs')
    context.net.check_terminal_load_only(strict=True)
    output=dict(status='complete',training=[],validation=[],training_event_ids=[],validation_event_ids=[],
        truth=[],event_records=[],physical_network=context.net,labels=labels,runtime_sha256=runtime,
        committed_payment=0.,known_actual_payment=0.,actual_payment=0.,unknown_payment_events=0,
        worst_payment=float(.4*plan.sum()),plan_lock_sha256=expected_lock_sha256,
        policy=policy,path=int(path),dependence=dependence,template=template,
        directions=directions.tolist(),device_declaration=dict(DEVICE),old_test_or_truth_files_read=False)
    for event in range(4):
        replicate=int(replicate_base)+10*int(path)+event
        z=legacy.draw_participation(plan,event,path,dependence,seed=participation_seed)
        committed=.2*float(plan[event%2]@z)
        record=dict(event=event,role='training' if event<2 else 'validation',status='started',
            replicate=replicate,profile_index=int(event_profiles[event]),price_row=plan[event%2].tolist(),
            directions=directions[event%2].tolist(),template=template,participants=z.astype(int).tolist(),
            accepted_participants=z.astype(int).tolist(),actual_payment=None,committed_payment=committed,
            contracted_reduction_kWh=(.2*z).tolist(),contracted_rebound_kWh=(.2*z).tolist(),
            delivery_status='not_yet_verified',checks=dict(energy_return_pass=False))
        output['committed_payment']+=committed
        try:
            base=sim._build_base_trajectory(context,int(event_profiles[event]),replicate)
            p,q,checks=apply_coded_response(base.p_base.to_numpy(),base.q_base.to_numpy(),
                plan[event%2],z,directions[event%2],template,active,base_kw=200.)
            record['checks']=checks;record['profile_seed']=int(base.shifted_seed);record['profile']=str(base.profile)
            if checks['status']!='complete':
                record.update(status=checks['status'],delivery_status='not_delivered_or_unresolved',
                              unavailable_accepted_positions=checks.get('unavailable_accepted_positions',[]))
                output['status']='physical_failed'
            else:
                pframe=pd.DataFrame(p,index=base.p_base.index,columns=labels)
                qframe=pd.DataFrame(q,index=base.q_base.index,columns=labels)
                ac,v=sim._solve_physical_ac(context,pframe,qframe,base.root_true)
                equations=physical_checks(context.net,pframe,qframe,ac,(.9,1.1),1e-8)
                measured=sim._observe_terminal_channels(context,base,pframe,qframe,v,pq_noise_rel=.005,v_noise_rel=voltage_sigma)
                scene=legacy.scene(measured['p_measured'].to_numpy(),measured['q_measured'].to_numpy(),
                    measured['voltage_measured'].to_numpy(),measured['root_observed'].to_numpy(),labels,
                    f'coded_{policy}_{dependence}_p{path}_e{event}',6+event)
                scene['actual_event_id']=event
                checks.update(electrical_equations_pass=equations['electrical_equations_pass'],
                    ac_converged_samples=int(ac['converged'].sum()),
                    voltage_min_pu=float(v.to_numpy().min()),voltage_max_pu=float(v.to_numpy().max()),
                    voltage_limits_pass=equations['voltage_limits_pass'],voltage_violation_cells=equations['voltage_violation_cells'],
                    max_nodal_p_residual_pu=equations['max_nodal_p_residual_pu'],
                    max_nodal_q_residual_pu=equations['max_nodal_q_residual_pu'],
                    max_distflow_identity_error_pu2=equations['max_distflow_identity_error_pu2'],
                    branch_ampacity_checked=False,actual_payment=committed)
                split='training' if event<2 else 'validation'
                output[split].append(scene);output[split+'_event_ids'].append(event)
                output['truth'].append(dict(event=event,P_pu=p,Q_pu=q,V_pu=v.to_numpy(),root_pu=base.root_true.to_numpy(),
                    participants=z,ac=ac,natural_P_pu=base.p_base.to_numpy(),natural_Q_pu=base.q_base.to_numpy()))
                record.update(status='complete',delivery_status='complete',actual_payment=committed,
                    checks=checks,delivered_reduction_kWh=checks['delivered_reduction_kWh'],
                    delivered_rebound_kWh=checks['delivered_rebound_kWh'])
                output['known_actual_payment']+=committed
        except Exception as error:
            record.update(status='physical_failed',delivery_status='unresolved',error_type=type(error).__name__,message=str(error))
            output['status']='physical_failed'
        if record['actual_payment'] is None:output['unknown_payment_events']+=1
        output['event_records'].append(record)
    output['actual_payment']=None if output['unknown_payment_events'] else output['known_actual_payment']
    output['budget_pass']=bool(output['committed_payment']<=1.6+1e-12 and output['worst_payment']<=1.6+1e-12)
    output['all_operation_checks_pass']=bool(output['status']=='complete' and all(all(r['checks'][k] for k in
        ('energy_return_pass','capacity_pass','voltage_limits_pass','electrical_equations_pass')) for r in output['event_records']))
    output['quote_qualification_pass']=all(r.get('checks',{}).get('quote_qualification',{}).get('quote_qualification_pass',False) for r in output['event_records'])
    return output


def evaluate_supports(predicted_support_labels,net):
    if _LEGACY is None:raise RuntimeError('evaluate only after frozen physical generation')
    return _LEGACY.evaluate_supports(predicted_support_labels,net)


def pure_checks():
    p=np.full((96,3),.02);q=np.full((96,3),.008)
    for template in ('same','opposed','alternating_opposed'):
        direction=np.array([1,1,1]) if template=='same' else np.array([1,-1,1])
        response_p,response_q,r=apply_coded_response(p,q,np.array([1,1,0]),np.array([1,1,0],bool),direction,template,[0,1])
        assert r['status']=='complete' and r['energy_return_pass']
        assert np.max(abs((response_p-p)*200.))<=.2+1e-12
        assert np.allclose((response_q-q)*200.,.35*(response_p-p)*200.)
        assert np.allclose(r['delivered_reduction_kWh'],[.2,.2,0])
        assert np.allclose(r['quote_qualification']['device_total_energy_kWh'],[.4,.4,.4])
    signed=waveforms('opposed',np.array([1,-1,1]),[0,1]);assert np.allclose(signed[:,:2].sum(axis=1),0)
    low=p.copy();low[WINDOW,0]=.0005
    _,_,declined=apply_coded_response(low,q,np.array([1,0,0]),np.array([0,0,0],bool),np.ones(3,int),'same',[0,1])
    assert declined['status']=='complete' and not declined['quote_qualification']['quote_qualification_pass']
    a,b,failed=apply_coded_response(low,q,np.array([1,0,0]),np.array([1,0,0],bool),np.ones(3,int),'same',[0,1])
    assert a is None and b is None and failed['status']=='contract_not_available' and failed['actual_payment'] is None
    assert failed['accepted_participants']==[1,0,0] and failed['committed_payment']==.2
    unknown=device_precheck(p,q,np.array([1,0,0],bool),signed,declaration=None)
    assert unknown['status']=='unresolved'
    return dict(status='passed',pure_arrays=True,new_profiles=0,AC_calls=0,LP_calls=0,
        synthetic_contract_feasibility_only=True,actual_device_feasibility='unverified')
