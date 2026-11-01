"""Hash-gated one-day physical validation of a frozen DR price plan.

No identifier or fitted pool is imported or parsed.  The pool file is checked
by hash only.  Original AC/measurement primitives are loaded after request,
protocol, plan, resource, payment and source gates.  One call generates exactly
96 observations and never adds them to training or adjusts historical weights.

Plan JSON fields: resources, modules, amplitudes, prices, elasticities,
response_factors.  All resource/module entries are the existing constructor
DTOs.  ``amplitudes`` are mean kW, a=lambda*price; the actual full-day response
is a*(1+D*z). Request z is ``factors`` (two prelocked numbers in[-1,1]).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sys
import traceback


PRODUCER_MODULES = (
    'rnj_wzzt.scenario.simulation', 'rnj_wzzt.scenario.profiles',
    'rnj_wzzt.scenario.settings', 'rnj_wzzt.data.paper_style_case_bank',
    'rnj_wzzt.data.paper_style_terminal_lv', 'rnj_wzzt.models.network',
    'rnj_wzzt.models.ac_powerflow',
)
HELPER_FILES = (
    'research_experiments/wzzt_dr_selection/dr_day_modules/day_schedule.py',
    'research_experiments/wzzt_dr_selection/dr_day_modules/day_physical_worker.py',
    'research_experiments/wzzt_dr_selection/dr_competition_bridge/coded_shapes.py',
    'research_experiments/wzzt_dr_selection/dr_competition_bridge/physical_bridge.py',
    'research_experiments/wzzt_dr_selection/end_to_end/physical_case.py',
)


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def write_pickle(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_bytes(pickle.dumps(value,protocol=4));tmp.replace(path)


def resolve(path,parent):
    p=Path(path);return (p if p.is_absolute() else parent/p).resolve()


def reference(item,parent,name):
    if not isinstance(item,dict) or not {'path','sha256'}<=set(item):
        raise ValueError(name+' requires path and sha256')
    p=resolve(item['path'],parent)
    if sha(p)!=item['sha256']:raise RuntimeError(name+' hash mismatch')
    return p


def integer(value,name,minimum=0):
    if isinstance(value,bool) or not isinstance(value,int) or value<minimum:
        raise ValueError(name+' must be integer >= '+str(minimum))
    return value


def contract_scope(request,protocol):
    """Bind only explicit multi-case gates; original defaults stay unchanged.

    The caller verifies the protocol hash before this pure check. Resource
    inventories and response laws remain in the separately hash-bound plan.
    """
    allowed=protocol.get('allowed_voltage_sigma',[.0005])
    if (not isinstance(allowed,list) or not allowed
            or any(isinstance(v,bool) or not isinstance(v,(int,float)) or v not in (.0002,.0005) for v in allowed)
            or len(set(allowed))!=len(allowed)):
        raise ValueError('declare allowed_voltage_sigma as distinct original .0002/.0005 values')
    if request.get('voltage_sigma') not in allowed or request.get('pq_sigma')!=.005:
        raise ValueError('retain protocol-declared original voltage noise andPQ.005')
    expected=protocol.get('expected_resource_nodes',[201,213])
    if (not isinstance(expected,list) or not expected
            or any(isinstance(v,bool) or not isinstance(v,int) or v not in request['labels'] for v in expected)
            or len(set(expected))!=len(expected)):
        raise ValueError('declared resource nodes must be distinct observed terminal labels')
    if 'expected_resource_nodes' in request and request['expected_resource_nodes']!=expected:
        raise ValueError('request resource nodes differ from hash-bound protocol')
    request['expected_resource_nodes']=list(expected)
    return request


def load_request(request_path,expected_sha):
    """All source/reference gates precede parsing a plan or loading a producer."""
    request_path=Path(request_path).resolve()
    if not expected_sha or sha(request_path)!=expected_sha:
        raise RuntimeError('request differs from frozen expected hash')
    request=json.loads(request_path.read_text(encoding='utf-8'));parent=request_path.parent
    if request.get('status')!='frozen' or request.get('phase','test')!='test':
        raise ValueError('frozen test request required')
    for key in ['case','record_id','policy']:
        if not isinstance(request.get(key),str) or not request[key]:
            raise ValueError(key+' must be a nonempty string')
    labels=request.get('labels')
    if not isinstance(labels,list) or not labels or len(set(labels))!=len(labels) or any(isinstance(v,bool) or not isinstance(v,int) for v in labels):
        raise ValueError('explicit unique integer terminal labels required')
    request['root_label']=integer(request.get('root_label'),'root_label')
    if request['root_label'] in labels:raise ValueError('root cannot be terminal')
    request['path']=integer(request.get('path'),'path')
    request['replicate']=integer(request.get('replicate'),'replicate',20000)
    request['day_offset']=integer(request.get('day_offset',6+request['path']),'day_offset',6)
    profile=integer(request.get('profile_index'),'profile_index')
    if profile not in (0,1,2):raise ValueError('use an original profile index0/1/2')
    factors=request.get('factors')
    if not isinstance(factors,list) or len(factors)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>1 for v in factors):
        raise ValueError('two prelocked continuous response factors in[-1,1] required')
    base=float(request.get('base_kw',float('nan')))
    if not math.isfinite(base) or base<=0:raise ValueError('positive base_kw required')
    request['base_kw']=base
    budget=float(request.get('incentive_budget',request.get('budget',float('nan'))))
    if budget!=1.6:raise ValueError('one-day hard budget remains1.6')
    request['incentive_budget']=budget
    protocol_path=reference(request.get('protocol'),parent,'protocol')
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    if protocol.get('status')!='frozen':raise ValueError('protocol must be frozen')
    contract_scope(request,protocol)
    if 'voltage_sigma' in protocol and protocol['voltage_sigma']!=request['voltage_sigma']:
        raise ValueError('protocol noise differs from request')
    if 'pq_sigma' in protocol and protocol['pq_sigma']!=request['pq_sigma']:
        raise ValueError('protocol P/Q noise differs from request')
    plan_item=request.get('plan',request.get('plan_packet'))
    if 'plan' in request and 'plan_packet' in request and request['plan']!=request['plan_packet']:
        raise ValueError('conflicting plan references')
    plan_path=reference(plan_item,parent,'price plan')
    pool_path=reference(request.get('frozen_pool'),parent,'frozen historical pool')
    # Do not parse the pool: worker has no predictive model/identifier input.
    if isinstance(protocol.get('plans'),dict) and request['policy'] in protocol['plans']:
        if reference(protocol['plans'][request['policy']],protocol_path.parent,'protocol price plan')!=plan_path:
            raise ValueError('request plan differs from protocol plan')
    if isinstance(protocol.get('frozen_pool'),dict):
        if reference(protocol['frozen_pool'],protocol_path.parent,'protocol frozen pool')!=pool_path:
            raise ValueError('request historical pool differs from protocol pool')
    repo=resolve(request['repo_root'],parent);source_root=resolve(request['source_root'],parent)
    expected=request.get('source_sha256')
    if not isinstance(expected,dict) or not expected:raise ValueError('explicit frozen source closure required')
    checked={}
    for file,digest in expected.items():
        path=resolve(file,parent)
        if str(path) in checked and checked[str(path)]!=digest:raise ValueError('conflicting source aliases')
        if sha(path)!=digest:raise RuntimeError('source hash mismatch: '+str(path))
        checked[str(path)]=digest
    required=[Path(__file__).resolve()]+[repo/file for file in HELPER_FILES]
    required += [source_root.joinpath(*name.split('.')).with_suffix('.py') for name in PRODUCER_MODULES]
    if any(str(path.resolve()) not in checked for path in required):
        raise ValueError('source closure omits worker/resource/physical dependency')
    protocol_sources=protocol.get('source_sha256')
    if not isinstance(protocol_sources,dict):raise ValueError('protocol must bind source closure')
    protocol_sources={str(resolve(p,protocol_path.parent)):h for p,h in protocol_sources.items()}
    if any(protocol_sources.get(p)!=h for p,h in checked.items()):
        raise ValueError('request sources differ from protocol source closure')
    metadata=dict(request.get('source_metadata',{}))
    metadata['path']=str(reference(metadata,parent,'source metadata'))
    request.update(repo_root=str(repo),source_root=str(source_root),source_metadata=metadata,
                   checked_source_sha256=checked,plan_path=str(plan_path),frozen_pool_path=str(pool_path),
                   protocol_path=str(protocol_path),request_path=str(request_path))
    return request


def audit_plan(request):
    """Pure generic resource/response/fee audit, including all four box corners."""
    import numpy as np
    sys.path.insert(0,request['repo_root'])
    from research_experiments.wzzt_dr_selection.dr_day_modules.day_schedule import (
        Battery,DayModule,ResourceDay,evaluate_schedule,
    )
    plan=json.loads(Path(request['plan_path']).read_text(encoding='utf-8'))
    modules=[DayModule(**m) for m in plan['modules']];n=len(modules)
    mean=np.asarray(plan['amplitudes'],float);prices=np.asarray(plan['prices'],float)
    elasticities=np.asarray(plan['elasticities'],float)
    d=np.asarray(plan['response_factors'],float)
    if n==0 or any(v.shape!=(n,) for v in [mean,prices,elasticities]) or d.shape!=(n,2):
        raise ValueError('aligned module/mean/price/elasticity/two-factor arrays required')
    if any(not np.isfinite(v).all() for v in [mean,prices,elasticities,d]) or (mean<0).any() or (prices<0).any() or (prices>2+1e-10).any() or (elasticities<=0).any() or (np.abs(d).sum(axis=1)>1).any():
        raise ValueError('invalid continuous contract domain')
    if not np.allclose(mean,elasticities*prices,rtol=0,atol=1e-10):
        raise ValueError('mean amplitude must equal frozen elasticity times price')
    factors=np.asarray(request['factors'],float);actual=mean*(1+d@factors)
    quotes={};hours=np.empty(n)
    for j,module in enumerate(modules):
        if len(module.profiles)!=1:raise ValueError('each price contract must meter one resource')
        key,shape=next(iter(module.profiles.items()))
        hours[j]=.25*np.maximum(-np.asarray(shape,float),0.).sum()
        if hours[j]<=0:raise ValueError('contract needs positive compensated reduction')
        if mean[j]>0:
            if key in quotes:raise ValueError('inactive alternative types must be exactly zero')
            quotes[key]=float(prices[j])
    resources=[]
    for dto in plan['resources']:
        dto=dict(dto)
        if dto.get('battery') is not None:dto['battery']=Battery(**dto['battery'])
        dto['incentive_per_kwh']=quotes.get(dto['resource_id'],0.)
        resources.append(ResourceDay(**dto))
    mapping={r.resource_id:r.node_label for r in resources}
    if len(mapping)!=len(resources) or set(quotes)-set(mapping):raise ValueError('contract resource declarations inconsistent')
    if set(mapping.values())!=set(request.get('expected_resource_nodes',[201,213])):
        raise ValueError('resource nodes differ from hash-bound declared scope')
    if any(r.node_label not in request['labels'] for r in resources):raise ValueError('resource outside observed terminals')
    hard=float(np.sum(hours*elasticities*(1+np.abs(d).sum(axis=1))*prices**2))
    if hard>request['incentive_budget']+1e-9:raise ValueError('analytical hard payment exceeds1.6')
    expected_audit=evaluate_schedule(resources,modules,mean.tolist(),rho=None)
    actual_audit=evaluate_schedule(resources,modules,actual.tolist(),rho=None)
    if not expected_audit['resource_feasible'] or not actual_audit['resource_feasible']:
        raise ValueError('mean or actual response violates generic resource constraints')
    corner_records=[]
    for z1,z2 in [(-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)]:
        amp=mean*(1+d@np.array([z1,z2]))
        corner=evaluate_schedule(resources,modules,amp.tolist(),rho=None)
        if not corner['resource_feasible']:raise ValueError('support corner violates resource/module constraints')
        if corner['total_incentive_payment']>hard+1e-9:raise AssertionError('corner fee exceeds analytical cap')
        corner_records.append(dict(factors=[z1,z2],maximum_violation=corner['maximum_violation'],
                                   payment=corner['total_incentive_payment']))
    fee=float((hours*prices)@actual);expected=float((hours*prices)@mean)
    if abs(fee-actual_audit['total_incentive_payment'])>1e-9 or abs(expected-expected_audit['total_incentive_payment'])>1e-9:
        raise AssertionError('independent per-contract/net-resource billing mismatch')
    if fee<0 or fee>hard+1e-9:raise AssertionError('realized fee outside hard payment cap')
    detail=dict(resources=resources,modules=modules,audit=actual_audit,expected_audit=expected_audit,
                expected_amplitudes=mean,actual_amplitudes=actual,prices=prices,
                response_factors=d,elasticities=elasticities,factors=factors,corners=corner_records)
    payments=dict(actual_payment=fee,known_actual_payment=fee,unknown_payment_events=0,
                  expected_payment=expected,worst_payment=hard,
                  exact_box_worst_payment=max(r['payment'] for r in corner_records),budget_pass=True)
    return plan,detail,payments


def contained_component_check(natural_p,natural_q,resources,labels,base_kw):
    """Check declared flexible P/Q components fit inside the natural node load."""
    import numpy as np
    positions={label:i for i,label in enumerate(labels)};bp={};bq={}
    for resource in resources:
        bp[resource.node_label]=bp.get(resource.node_label,0.)+np.asarray(resource.baseline_kw)
        bq[resource.node_label]=bq.get(resource.node_label,0.)+np.asarray(resource.reactive_ratio)*np.asarray(resource.baseline_kw)
    pmin=qmin=float('inf')
    for label in bp:
        pos=positions[label]
        pmin=min(pmin,float(np.min(np.asarray(natural_p)[:,pos]*base_kw-bp[label])))
        qmin=min(qmin,float(np.min(np.asarray(natural_q)[:,pos]*base_kw-bq[label])))
    report=dict(minimum_nonflexible_p_kw=pmin,minimum_nonflexible_q_kvar=qmin,
                contained_component_pass=bool(pmin>=-1e-8 and qmin>=-1e-8))
    if not report['contained_component_pass']:
        error=ValueError('actual natural load cannot contain the predeclared flexible component')
        error.component_report=report
        raise error
    return report


def _legacy(repo):
    path=Path(repo)/HELPER_FILES[3]
    spec=importlib.util.spec_from_file_location('_one_day_original_physical_bridge',path)
    if spec is None or spec.loader is None:raise RuntimeError('existing physical bridge unavailable')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def run(request,output,receipt):
    import numpy as np
    import pandas as pd
    plan,detail,payments=audit_plan(request)
    from research_experiments.wzzt_dr_selection.dr_day_modules.day_physical_worker import node_increment_arrays,scalar_resource_report
    receipt.update(**payments,case=request['case'],record_id=request['record_id'],policy=request['policy'],
                   path=request['path'],profile_index=request['profile_index'],replicate=request['replicate'],
                   protocol_sha256=request['protocol']['sha256'],plan_sha256=sha(request['plan_path']),
                   frozen_pool_sha256=sha(request['frozen_pool_path']),frozen_pool_parsed=False,
                   source_sha256=request['checked_source_sha256'],
                   voltage_sigma=request['voltage_sigma'],pq_sigma=request['pq_sigma'],
                   incentive_budget=request['incentive_budget'],corners=detail['corners'],
                   resource_reports=scalar_resource_report(detail['audit']),
                   baseline_semantics='contained component of unchanged natural terminal load; no baseline added',
                   actual_payment_scope='realized contract response fee retained even if subsequent physical delivery/checks fail')
    write_json(output/'events_receipt.json',receipt)
    legacy=_legacy(request['repo_root'])
    metadata=legacy.frozen_source_metadata(request,request['case'])
    if metadata['root_label']!=request['root_label'] or abs(metadata['base_kw']-request['base_kw'])>1e-12:
        raise ValueError('request bases differ from hash-bound metadata')
    sim,runtime=legacy._simulation_runtime(Path(request['source_root']))
    for name in PRODUCER_MODULES:
        path=Path(request['source_root']).joinpath(*name.split('.')).with_suffix('.py').resolve()
        if runtime[name]!=request['checked_source_sha256'][str(path)]:raise RuntimeError('loaded producer differs from frozen source')
    checker,checker_source=legacy._physical_check_function(request['repo_root'])
    if sha(checker_source)!=request['checked_source_sha256'][str(checker_source.resolve())]:raise RuntimeError('loaded physical checker differs from frozen source')
    context=sim._prepare_case_context(request['case'],96,scenario_suite='reference',root_observation='noisy',
        root_meter_noise_rel=request['voltage_sigma'],root_sigma=.003,impedance_scale=1.)
    legacy.validate_context_metadata(context,metadata,request['labels'])
    context.net.check_terminal_load_only(strict=True)
    receipt.update(runtime_sha256=runtime,base_kw=metadata['base_kw'],base_kv=metadata['base_kv'],base_mva=metadata['base_mva'])
    private=dict(status='started',resource_audit=detail['audit'],expected_amplitudes=detail['expected_amplitudes'],
        actual_amplitudes=detail['actual_amplitudes'],prices=detail['prices'],factors=detail['factors'])
    scenes=[];ac_attempted=False
    try:
        base=sim._build_base_trajectory(context,request['profile_index'],request['replicate'])
        p0,q0=base.p_base.to_numpy(),base.q_base.to_numpy()
        private.update(natural_P_pu=p0,natural_Q_pu=q0,root_pu=base.root_true.to_numpy())
        component=contained_component_check(p0,q0,detail['resources'],request['labels'],metadata['base_kw'])
        receipt.update(component)
        dp,dq=node_increment_arrays(detail['audit'],request['labels'])
        physical_p,physical_q=p0+dp/metadata['base_kw'],q0+dq/metadata['base_kw']
        if physical_p.shape!=(96,len(request['labels'])) or physical_q.shape!=physical_p.shape or not np.isfinite(physical_p).all() or not np.isfinite(physical_q).all():
            raise ValueError('finite96-row original producer/increment arrays required')
        target_positions=[request['labels'].index(r.node_label) for r in detail['resources']]
        if np.min(physical_p[:,target_positions]) < -1e-10:raise ValueError('DR overdraws whole actual terminal active load')
        p_error=float(np.max(np.abs((physical_p-p0)*metadata['base_kw']-dp),initial=0.))
        q_error=float(np.max(np.abs((physical_q-q0)*metadata['base_kw']-dq),initial=0.))
        if max(p_error,q_error)>1e-9:raise AssertionError('resource increment was not added exactly once')
        receipt.update(increment_only_bridge_pass=True,max_p_increment_error_kw=p_error,
                       max_q_increment_error_kvar=q_error,whole_target_terminal_nonnegative=True,
                       profile_seed=int(base.shifted_seed))
        pf=pd.DataFrame(physical_p,index=base.p_base.index,columns=request['labels'])
        qf=pd.DataFrame(physical_q,index=base.q_base.index,columns=request['labels'])
        ac_attempted=True
        ac,voltage=sim._solve_physical_ac(context,pf,qf,base.root_true)
        equations=checker(context.net,pf,qf,ac,(.9,1.1),1e-8)
        measured=sim._observe_terminal_channels(context,base,pf,qf,voltage,pq_noise_rel=request['pq_sigma'],v_noise_rel=request['voltage_sigma'])
        observed=legacy.scene(measured['p_measured'].to_numpy(),measured['q_measured'].to_numpy(),
            measured['voltage_measured'].to_numpy(),measured['root_observed'].to_numpy(),
            request['labels'],'one_day_'+request['record_id'],request['day_offset'])
        scenes.append(observed)
        checks=dict(electrical_equations_pass=bool(equations['electrical_equations_pass']),
            voltage_limits_pass=bool(equations['voltage_limits_pass']),
            ac_converged_samples=int(ac['converged'].sum()),ac_all_96_converged=int(ac['converged'].sum())==96,
            voltage_min_pu=float(voltage.to_numpy().min()),voltage_max_pu=float(voltage.to_numpy().max()),
            voltage_violation_cells=int(equations['voltage_violation_cells']),
            max_nodal_p_residual_pu=float(equations['max_nodal_p_residual_pu']),
            max_nodal_q_residual_pu=float(equations['max_nodal_q_residual_pu']),
            max_distflow_identity_error_pu2=float(equations['max_distflow_identity_error_pu2']),
            branch_ampacity_checked=False)
        passed=checks['electrical_equations_pass'] and checks['voltage_limits_pass'] and checks['ac_all_96_converged']
        receipt.update(status='complete' if passed else 'physical_checks_failed',checks=checks,
                       all_operation_checks_pass=passed,observed_validation_rows=96)
        private.update(P_pu=physical_p,Q_pu=physical_q,V_pu=voltage.to_numpy(),ac=ac,delta_kw=dp,delta_kvar=dq)
    except Exception as error:
        receipt.update(status='physical_failed',all_operation_checks_pass=False,error_type=type(error).__name__,
                       message=str(error),traceback=traceback.format_exc(),failure_retained=True)
        if hasattr(error,'component_report'):receipt.update(error.component_report)
    receipt.update(completed_utc=now(),observed_validation_days=len(scenes),
                   observed_validation_rows=sum(len(s['P_terminal']) for s in scenes),
                   physical_event_attempted=1,AC_attempted=ac_attempted)
    private['status']=receipt['status']
    packet=dict(validation=scenes,metadata=dict(status=receipt['status'],observed_only=True,
        training_rows=0,validation_rows=receipt['observed_validation_rows'],terminal_labels=request['labels'],
        base_kw=metadata['base_kw'],base_kv=metadata['base_kv'],base_mva=metadata['base_mva'],root_label=metadata['root_label'],
        case=request['case'],path=request['path'],record_id=request['record_id'],interval_hours=.25,
        synthetic_dates_are_order_labels=True,protocol_sha256=request['protocol']['sha256'],
        request_sha256=receipt['request_sha256_expected'],frozen_pool_sha256=receipt['frozen_pool_sha256']))
    write_pickle(output/'packet.pkl',packet)
    write_pickle(output/'physical_truth.pkl',private)
    if ac_attempted:
        evaluation=legacy.evaluate_supports([],context.net)
        write_json(output/'truth_evaluation.json',dict(true_supports=evaluation['missed_blocks'],
            terminal_labels=request['labels'],full_effective_truth_blocks=evaluation['full_effective_truth_blocks'],
            zero_impedance_merged=True,arbitrarily_weak_positive_lines_retained=True,
            use='evaluator only after frozen predictive selection; never price planning or parameter fitting'))
    for name in ['packet.pkl','physical_truth.pkl','truth_evaluation.json']:
        path=output/name
        if path.exists():receipt['private_files'][name]=dict(sha256=sha(path),bytes=path.stat().st_size)
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True)
    p.add_argument('--expected-sha',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args(argv)
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('retain existing one-day output')
    output.mkdir(parents=True,exist_ok=True)
    receipt=dict(status='started',started_utc=now(),request_sha256_expected=args.expected_sha,
        physical_events_expected=1,physical_rows_expected=96,actual_payment=None,known_actual_payment=0.,unknown_payment_events=1,
        no_clipping=True,no_resampling=True,automatic_retry=False,identifier_run=False,parameter_fitting_run=False,
        old_or_new_pool_parameters_parsed=False,old_test_or_truth_files_read=False,private_files={})
    write_json(output/'events_receipt.json',receipt)
    try:
        request=load_request(args.request,args.expected_sha)
        receipt=run(request,output,receipt)
    except Exception as error:
        receipt.update(status='worker_failed',completed_utc=now(),error_type=type(error).__name__,
            message=str(error),traceback=traceback.format_exc(),failure_retained=True)
    write_json(output/'events_receipt.json',receipt)
    print(json.dumps({key:receipt.get(key) for key in ['status','observed_validation_rows','actual_payment','worst_payment','budget_pass','message']},ensure_ascii=False,allow_nan=False))
    return 0 if receipt['status']=='complete' else 2


if __name__=='__main__':
    raise SystemExit(main())
