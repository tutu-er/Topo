"""192-fit/96-validation histories, frozen equipment and paired new-day tests.

Planning reads only historical observations, frozen candidates, the old sealed
forecast and the fixed equipment specification. New-day noise/AC/true network
are opened only after all three quotations and actions are sealed.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback

import numpy as np

from ... import io
from ...observations import Observation
from ..equipment_flex_20261115 import run as equipment
from ..equipment_flex_20261115.domain import EquipmentCase
from ..distribution_gate_20261114.run import planning_context
from ..distribution_gate_20261114 import solver as engine
from ..single_target_20261115.evidence import SingleTargetEvidence
from ..node_time_pq_20261115.run import historical_band
from ..node_time_pq_20261115.domain import NodeTimePQCase
from ..flexibility_stability_20261114 import solver as prices


paired = equipment.paired
old = paired.old
ROOT = old.ROOT
MODES = tuple(equipment.MODES)
POLICIES = ('no_dfs', *('single_'+mode for mode in MODES))
SEED = 2026111521
PQ_LEVELS = (0., .0025, .005, .01, .02)


def history_repeat(entry):
    value = int(entry.get('history_repeat', entry.get('noise_repeat', -1)))
    if value not in range(4):
        raise ValueError('Expected one of the four locked history-noise repeats')
    if 'noise_repeat' in entry and int(entry['noise_repeat']) != value:
        raise ValueError('Conflicting history-noise repeat labels')
    return value


def pq_sd(entry):
    value = float(entry.get('pq_relative_sd', entry['meter']['additional_PQ_relative_sd']))
    if value not in PQ_LEVELS or value != float(entry['meter']['additional_PQ_relative_sd']):
        raise ValueError('PQ noise differs from the locked five-level grid')
    if float(entry['meter']['additional_V_relative_sd']) != .0005:
        raise ValueError('Voltage/root meter error must stay .0005 across PQ levels')
    return value


def selected_entries(entries, smoke=False):
    """Smoke selection is fixed by input labels, never by fit/AC performance."""
    identities = [entry['id'] for entry in entries]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate prepared input IDs')
    for entry in entries:
        history_repeat(entry); pq_sd(entry)
        if entry['case'] not in ('pengwah18','soumalas11'):
            raise ValueError('Unexpected physical network in the locked grid')
        if (int(entry['training_rows']), int(entry['validation_rows']),
                int(entry.get('new_rows', entry.get('new_points', 96)))) != (192, 96, 96):
            raise ValueError('This study requires 192 fitting, 96 validation and 96 new-day rows')
    cells=[(entry['case'],history_repeat(entry),pq_sd(entry)) for entry in entries]
    if len(set(cells))!=len(cells):
        raise ValueError('Duplicate network/history-repeat/PQ-noise cells')
    if not smoke:
        return deepcopy(entries)
    chosen = [entry for entry in entries if history_repeat(entry)==0 and pq_sd(entry)==.005]
    if {entry['case'] for entry in chosen} != {'pengwah18', 'soumalas11'} or len(chosen)!=2:
        raise ValueError('Smoke requires both predeclared h00/PQ.005 inputs; no replacement based on outcomes')
    return deepcopy(chosen)


def device_multipliers(case, entry, repeat, seed=SEED):
    """Five PQ levels share each physical-device realization and P/Q scalar."""
    ids = list(dict.fromkeys(row['physical_device_id'] for row in case.resources))
    eta = np.empty(case.R)
    records = []
    for device in ids:
        value, source = equipment.keyed_multiplier(seed,
            int(entry.get('original_input_index', entry['index'])), repeat, device,
            forecast_index=history_repeat(entry))
        columns = [j for j, row in enumerate(case.resources) if row['physical_device_id']==device]
        eta[columns] = value
        records.append(dict(physical_device_id=device, columns=columns, value=value, keyed_stream=source))
    return eta, records


def evaluation_noises(arrays, repeat, n):
    """Extract already sealed independent evaluation draws; no new RNG draw."""
    result=[]
    for name in ('P', 'Q', 'V', 'root'):
        value=np.asarray(arrays[name], float)
        expected=(16,96) if name=='root' else (16,96,n)
        if value.shape!=expected or not np.isfinite(value).all() or not 0<=repeat<16:
            raise ValueError('Invalid sealed evaluation-noise array: '+name)
        result.append(value[repeat].copy())
    return result


def source_snapshot(out, entries):
    """Snapshot runtime imports, excluding concurrently developed postprocessing."""
    paths={Path(__file__).resolve(),Path(__file__).with_name('PROTOCOL.md'),Path(__file__).with_name('MODEL.md')}
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename: continue
        path=Path(filename).resolve()
        if (path.suffix=='.py' and path.is_relative_to(ROOT) and 'outputs' not in path.parts
                and not path.name.startswith('test_')
                and path.name not in ('audit.py','report.py','plot.py','probability_panel.py')):
            paths.add(path)
    for case in {entry['case'] for entry in entries}:
        for name in ('metadata.json','buses.csv','branches.csv'):
            paths.add(ROOT/'topology_identification_standalone'/'cases'/case/name)
    for path in sorted(paths):
        destination=out/'source_snapshot'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
    return [io.ref(path) for path in sorted(paths)]


def load_index(index):
    document=io.read(index)
    if isinstance(document,list):
        raise ValueError('Use the data manifest so failed fits and planned denominators remain visible')
    value=document['inputs']
    entries=io.read(old.checked(value)) if isinstance(value,dict) and 'path' in value else value
    if not isinstance(entries,list): raise ValueError('Data manifest inputs must be a list or a checked JSON reference')
    if document.get('status') not in (None,'complete') or document.get('source_unchanged') is not True:
        raise ValueError('Complete unchanged data preparation required')
    # The successful subset cannot replace the original forty-attempt ledger.
    attempts=document.get('attempts',[])
    if isinstance(attempts,dict) and 'path' in attempts:
        attempts=io.read(old.checked(attempts))
    expected={(c,h,q) for c in ('pengwah18','soumalas11') for h in range(4) for q in PQ_LEVELS}
    cells=[(r['task']['case'],history_repeat(r['task']),pq_sd(r['task'])) for r in attempts]
    if len(cells)!=40 or set(cells)!=expected or int(document.get('planned_inputs',-1))!=40:
        raise ValueError('The data ledger must retain all forty predeclared cells, including failures')
    successful={r['task']['id'] for r in attempts if r['status']=='complete'}
    if successful!={entry['id'] for entry in entries}:
        raise ValueError('Successful data entries do not match the complete attempt ledger')
    old.checked(document['protocol'])
    return document,entries


def validate_preflight(reference,index):
    if reference is None: raise ValueError('Passed independent smoke audit required before formal preparation')
    audit=io.read(reference)
    if audit.get('status')!='passed' or audit.get('errors'):
        raise ValueError('Independent preflight audit did not pass')
    previous=io.read(old.checked(audit['source_protocol']))
    if not previous.get('smoke') or previous['input_index']!=io.ref(index):
        raise ValueError('Smoke must use this same completed input index')
    expected=int(previous['expected_rows'])
    if expected!=8 or int(audit.get('audited_rows',-1))!=expected:
        raise ValueError('Smoke audit must cover both fixed inputs and all eight policy rows')
    completion=io.read(Path(audit['source_protocol']['path']).parent/'complete.json')
    if completion.get('failed_rows') or completion.get('failed_tasks') or completion.get('quote_exceptions'):
        raise ValueError('Smoke contains execution, task or quote implementation exceptions')
    paired.check_refs(previous['sources'])
    return io.ref(reference)


def context_planning(entry):
    planning=planning_context(entry,entry['base_kw'])
    with np.load(old.checked(entry['forecast_ref']),allow_pickle=False) as z:
        forecast=z[entry.get('forecast_key','forecast00')].copy()
    spec=io.read(old.checked(entry['equipment_spec']))
    physical=EquipmentCase(spec,planning['labels'],forecast,planning['base'])
    metadata=io.read(old.checked(entry['node_time_bandwidth']))
    arrays=paired.arrays(old.checked(metadata['arrays']))
    case=NodeTimePQCase(physical,arrays,metadata)
    return planning,forecast,physical,case


def prepare(index,out,*,smoke=False,seconds=1200.,workers=3,preflight=None):
    index,out=Path(index).resolve(),Path(out).resolve()
    if out.exists(): raise FileExistsError('Preserve prior prepared results: '+str(out))
    if not np.isfinite(seconds) or seconds<=0 or not 1<=workers<=3:
        raise ValueError('Positive finite quotation budget and 1..3 workers required')
    manifest,all_entries=load_index(index)
    entries=selected_entries(all_entries,smoke)
    preflight_ref=None if smoke else validate_preflight(preflight,index)
    out.mkdir(parents=True)
    ready,failures=[],[]
    fixed_specs,forecasts={},{}
    for entry in entries:
        try:
            # This path never opens physical_cache, new_day or evaluation_noise.
            for ref in (entry['score'],entry['native_copy'],entry['files']['train.npz'],
                        entry['files']['validation.npz'],entry['files']['observations_sealed.json'],
                        entry['forecast_ref'],entry['equipment_spec'],entry['resource_context']): old.checked(ref)
            physical_key=(entry['case'],entry['start_date'])
            for registry,key in ((fixed_specs,'equipment_spec'),(forecasts,'forecast_ref')):
                previous=registry.setdefault(physical_key,entry[key])
                if previous!=entry[key]: raise ValueError('Equipment or forecast changed across same-source noise levels')
            arrays,metadata=historical_band(entry)
            folder=out/'prepared'/entry['id'];folder.mkdir(parents=True)
            np.savez_compressed(folder/'bandwidth.npz',**arrays)
            metadata['arrays']=io.ref(folder/'bandwidth.npz')
            io.write(folder/'bandwidth.json',metadata)
            entry['node_time_bandwidth']=io.ref(folder/'bandwidth.json')
            planning,forecast,physical,case=context_planning(entry)
            probe=SingleTargetEvidence(case,planning['score'],planning['val'],forecast,planning['base'],
                mode='normal_fixed',meter_relative_sd=.0005,forecast_relative_sd=.05)
            entry['planned_history_target_native']=int(probe.native_indices[probe.targets[0]])
            entry['historical_native_index']=int(planning['score']['native_index'])
            entry['historical_p_map']=float(planning['score']['p_map'])
            if entry['planned_history_target_native']!=entry['historical_native_index']:
                raise ValueError('Historical target differs from native strict validation selection')
            if not case.audit(np.zeros(case.shape))['passed']:
                raise ValueError('Zero action fails physical equipment or fresh historical band')
            io.write(folder/'resource_case.json',physical.config)
            io.write(folder/'baseline_device_states.json',physical.device_states(np.zeros(physical.shape)))
            io.write(folder/'baseline_decomposition.json',physical.baseline_decomposition())
            entry.update(prepared_resource_case=io.ref(folder/'resource_case.json'),
                baseline_device_states=io.ref(folder/'baseline_device_states.json'),
                baseline_decomposition=io.ref(folder/'baseline_decomposition.json'))
            ready.append(entry)
        except Exception as exc:
            failures.append(dict(id=entry['id'],stage='planning_preparation',error=repr(exc),traceback=traceback.format_exc()))
    io.write(out/'inputs.json',ready)
    data_failures=manifest.get('failures',[])
    if isinstance(data_failures,dict) and 'path' in data_failures:
        data_failures=io.read(old.checked(data_failures))
    io.write(out/'preparation_failures.json',dict(data_failures=data_failures,planning_failures=failures))
    planned=2 if smoke else 40
    repeats=1 if smoke else 16
    protocol=dict(created_utc=old.utc(),phase='history288_PQ_only_noise_development',smoke=bool(smoke),
        input_index=io.ref(index),inputs=io.ref(out/'inputs.json'),data_protocol=manifest.get('protocol'),
        preparation_failures=io.ref(out/'preparation_failures.json'),planned_inputs=planned,
        available_fitted_inputs=len(all_entries),prepared_inputs=len(ready),
        expected_quotes=planned*3,expected_rows=planned*4*repeats,
        expected_DR_rows=planned*3*repeats,expected_zero_rows=planned*repeats,
        executable_expected_quotes=len(ready)*3,executable_expected_rows=len(ready)*4*repeats,
        unavailable_inputs=planned-len(ready),unavailable_policy_rows=(planned-len(ready))*4*repeats,
        seed=SEED,forecasts=1,repeats=repeats,history_repeats=4,pq_relative_sd=list(PQ_LEVELS),
        meter_relative_sd=.0005,forecast_relative_sd=.05,modes=list(MODES),policies=list(POLICIES),
        workers=1 if smoke else workers,solver_seconds=2. if smoke else float(seconds),
        training_rows=192,validation_rows=96,new_rows=96,scoring_rows='96 historical validation + 96 new-day rows; training is never rescored',
        preflight_audit=preflight_ref,oracle=False,
        objective='Minimize activation; requested capacity plus delivered activation reported separately',
        reservation_prices=dict(kappaP=.5,kappaQ=.5),
        target_rule='One strict historical native winner; all foreign native parameter points remain competitors',
        guard_rule='Original forward scalar-mean guards only; no new gate or RX refit',
        seed_rule='One shared zero point only; no old optimizer action, no candidate-dependent additional seed',
        bandwidth_rule='Recompute original node/hour affine-PQ residual max-minus-min band from this input training192 and timestamps only',
        equipment_rule='Byte-identical frozen four-class equipment specification per physical network-day; no extra SVG or Q95 sizing',
        noise_rule='Sealed data evaluation_noise split1/path0..15; same network/history-repeat arrays for five PQ SDs; voltage/root SD fixed.0005',
        response_rule='One keyed Uniform(.9,1) scalar for each whole physical device P/Q trajectory; no noise-level key',
        zero_rule='Freshly score no_dfs against this grid cell frozen bank; original zero physical AC cache only may be reused',
        fallback='Every failed quotation executes explicit zero and remains in the denominator',
        resume_rule='Completed quote/action checkpoints can be reused; interrupted searches without a completed checkpoint require review and cannot reset the budget',
        independence='Two previously studied physical network-days; four history noise draws and sixteen conditional execution paths per level, not forty independent dates',
        driver_relative_path=Path(__file__).resolve().relative_to(ROOT).as_posix(),environment=io.environment())
    protocol['sources']=source_snapshot(out,entries)
    protocol.update(old._git_receipt([Path(ref['path']) for ref in protocol['sources']]))
    io.write(out/'protocol.json',protocol)
    return protocol


def load_quote_checkpoint(directory,name,protocol_ref):
    checkpoint=directory/(name+'_completed.json')
    if checkpoint.exists():
        record=io.read(checkpoint)
        if record['protocol']!=protocol_ref: raise ValueError('Quote checkpoint belongs to a different protocol')
        quote=io.read(old.checked(record['plan']))
        with np.load(old.checked(record['action']),allow_pickle=False) as z: action=z['action'].copy()
        return quote,action
    if (directory/(name+'_started.json')).exists():
        raise RuntimeError('Interrupted quotation needs reviewed remaining budget; refusing to restart: '+name)
    if (directory/(name+'_plan.json')).exists() or (directory/(name+'_action.npz')).exists():
        raise RuntimeError('Incomplete quotation checkpoint must be preserved: '+name)
    return None


def load_policy_checkpoint(folder,seal_ref):
    if not folder.exists(): return None
    path=folder/'receipt.json'
    if not path.exists(): raise RuntimeError('Partial execution directory must be preserved: '+str(folder))
    receipt=io.read(path)
    if receipt.get('planning_seal')!=seal_ref:
        raise ValueError('Execution receipt belongs to a different sealed plan')
    if receipt['status']=='complete':
        for key in ('observations','pair_evidence','pair_changes','physical_state'):
            old.checked(receipt[key])
        old.checked(receipt['device_states']['json']);old.checked(receipt['device_states']['arrays'])
    return receipt


def execute(entry,protocol,directory,planning,physical,case,forecast,controls,plans):
    seal=io.read(directory/'seal.json');seal_ref=io.ref(directory/'seal.json')
    paired.check_refs([seal[k] for k in ('planner','plans','score','train','validation','equipment_spec',
        'resource_case','shared_initial_points','bandwidth_json','bandwidth_npz','baseline_device_states')])
    # First future data / true-network read in the execution path.
    ctx=old.context(entry)
    if ctx['score']!=planning['score'] or ctx['base']!=planning['base']:
        raise ValueError('Evaluation frozen bank differs from planning')
    noise_ref=entry['evaluation_noise']
    all_noise=paired.arrays(old.checked(noise_ref))
    H,C,hist,sigma2=ctx['H'],ctx['common_H'],ctx['history'],ctx['score']['sigma2']
    predicted={name:old.pair_components(H,C,hist,sigma2,forecast+physical.shift_pq_pu(u,ctx['base']))
               for name,u in controls.items()}
    zero_prediction=predicted['no_dfs']
    baseline_clean=ctx['root'][:,None]**2-ctx['baseline_state']['terminal_voltage_pu']**2
    zero_grid_value=old.grid_value(physical,ctx['baseline_state']['root_supply_kw'])
    rows=[]
    for repeat in range(protocol['repeats']):
        rd=directory/f'execution{repeat:02d}';rd.mkdir(exist_ok=True)
        noise=evaluation_noises(all_noise,repeat,physical.N)
        eta,lineage=device_multipliers(physical,entry,repeat,protocol['seed'])
        random_path=rd/'paired_randomness.npz'
        expected=dict(eta=eta,**{f'noise{i}':z for i,z in enumerate(noise)})
        if random_path.exists():
            saved=paired.arrays(random_path)
            if set(saved)!=set(expected) or any(not np.array_equal(saved[k],v) for k,v in expected.items()):
                raise ValueError('Saved paired randomness changed on resume')
        else: np.savez_compressed(random_path,**expected)
        lineage_path=rd/'randomness_lineage.json'
        lineage_record=dict(evaluation_noise=noise_ref,noise_path=repeat,history_repeat=history_repeat(entry),
            devices=lineage,noise_level_not_in_random_key=True,all_modes_share_observation_noise=True,
            physical_PQ_share_device_eta=True,history_and_evaluation_streams_separate=True)
        if lineage_path.exists():
            if io.read(lineage_path)!=lineage_record: raise ValueError('Randomness lineage changed')
        else: io.write(lineage_path,lineage_record)
        random_ref=io.ref(random_path)
        baseline=None;zero_ob=None
        for name,request in controls.items():
            folder=rd/name
            resumed=load_policy_checkpoint(folder,seal_ref)
            if resumed is not None:
                rows.append(resumed)
                if name=='no_dfs' and resumed['status']=='complete':
                    zero_ob=paired.arrays(old.checked(resumed['observations']))
                    baseline=paired.arrays(old.checked(resumed['pair_evidence']))
                continue
            folder.mkdir()
            is_zero=name=='no_dfs';quote=None if is_zero else plans[name]
            row=dict(id=entry['id'],case=entry['case'],start_date=entry['start_date'],
                original_id=entry['original_id'],index=entry['index'],original_input_index=entry.get('original_input_index',entry['index']),
                history_repeat=history_repeat(entry),noise_repeat=history_repeat(entry),pq_relative_sd=pq_sd(entry),
                forecast_index=0,repeat=repeat,strategy=name,parent_strategy=name,
                mode=None if is_zero else quote['mode'],guard_kind='equipment_node_hour_pq',is_rotation=False,
                quote_zero_fallback=False if is_zero else quote['zero_fallback'],
                quote_new_constraints_feasible=None if is_zero else quote['new_gate_feasible'],
                planning_seal=seal_ref,paired_randomness_source=random_ref,
                randomness_lineage=io.ref(lineage_path),evaluation_noise_source=noise_ref,
                baseline_source=None if is_zero else io.ref(rd/'no_dfs'/'receipt.json'))
            try:
                if not is_zero and (baseline is None or zero_ob is None):
                    raise RuntimeError('Fresh same-cell no_dfs evaluation failed; paired comparison unavailable')
                delivered,delivery=physical.execute(request,ctx['p']*ctx['base'],eta)
                states=equipment.save_states(folder,physical.device_states(request),physical.device_states(delivered))
                requested_domain,delivered_domain=physical.audit(request),physical.audit(delivered)
                if not requested_domain['passed'] or not delivered_domain['passed']:
                    raise ArithmeticError('Equipment domain violated')
                pp,qq,state,ac,state_ref=old.physical_state(ctx,physical,delivered,folder/'full_ac.npz')
                v,root=state['terminal_voltage_pu'],ctx['root']
                pm,qm,y,vm,rm=old.meter(pp,qq,v,root,noise,pq_sigma=pq_sd(entry),voltage_sigma=.0005)
                if is_zero and repeat==0:
                    prepared=paired.arrays(old.checked(entry['files']['new_day.npz']))
                    for key,value in (('P',pm),('Q',qm),('Y',y),('V',vm),('root',rm)):
                        if not np.array_equal(value,prepared[key]):
                            raise ArithmeticError('Evaluation repeat0 differs from sealed new_day: '+key)
                clean_y=root[:,None]**2-v**2
                observed=Observation(pm,qm,y,ctx['labels'],vm,rm)
                values,ll,new_sse=old.score_observation(ctx,observed)
                components=old.pair_components(H,C,hist,sigma2,np.c_[pp,qq],clean_y,observed,ctx['history_without_common'])
                if is_zero:
                    baseline=components
                    zero_ob=dict(true_P=pp,true_Q=qq,true_V=v,true_root=root,P=pm,Q=qm,Y=y,V=vm,root=rm)
                changes,summary=old.evidence_summary(ctx,baseline,components,zero_prediction,predicted[name],None,None)
                np.savez_compressed(folder/'pair_evidence.npz',**components)
                np.savez_compressed(folder/'pair_changes.npz',**changes)
                np.savez_compressed(folder/'observations.npz',P=pm,Q=qm,Y=y,V=vm,root=rm,labels=ctx['labels'],
                    true_P=pp,true_Q=qq,true_V=v,true_root=root,requested_action=request,delivered_action=delivered,
                    native_history_sse=hist,native_new_sse=new_sse,native_total_sse=hist+new_sse,representative_log_probability=ll)
                delta_power=np.c_[pp,qq]-np.c_[ctx['p'],ctx['q']]
                increment_y=clean_y-baseline_clean
                Kerror=np.mean((increment_y[None]-np.einsum('tc,knc->ktn',delta_power,ctx['means']))**2,axis=(1,2))/sigma2
                repH=np.asarray([p['H'] for p in ctx['score']['parameters']])
                Herror=np.mean((increment_y[None]-np.einsum('tc,knc->ktn',delta_power,repH))**2,axis=(1,2))/sigma2
                planned=prices.cost_breakdown(physical,request);settled=prices.cost_breakdown(physical,request,delivered)
                voltage=state['voltage_magnitude_pu'];amounts=physical.amounts(delivered)
                row.update(status='complete',**values,delivery=delivery,evidence=summary,physical_state=state_ref,AC=ac,
                    observations=io.ref(folder/'observations.npz'),pair_evidence=io.ref(folder/'pair_evidence.npz'),
                    pair_changes=io.ref(folder/'pair_changes.npz'),device_states=states,
                    requested_equipment_audit=requested_domain,delivered_equipment_audit=delivered_domain,
                    planned_band=case.band_audit(request),delivered_band=case.band_audit(delivered),
                    planned_cost=planned,settled_cost=settled,planned_fee_yuan=planned['total_yuan'],settled_fee_yuan=settled['total_yuan'],
                    requested_quantities=physical.amounts(request),delivered_quantities=amounts,
                    delivered_abs_kwh=amounts['active_abs_kwh'],delivered_abs_kvarh=amounts['reactive_abs_kvarh'],
                    all_bus_voltage_min=float(voltage.min()),all_bus_voltage_max=float(voltage.max()),
                    voltage_outside_095_105_cells=int(((voltage<.95)|(voltage>1.05)).sum()),
                    baseline_root_energy_value_yuan=zero_grid_value,
                    root_energy_value_yuan=old.grid_value(physical,state['root_supply_kw']),
                    root_energy_value_change_yuan=old.grid_value(physical,state['root_supply_kw'])-zero_grid_value,
                    reused_zero_physical_cache=bool(np.count_nonzero(delivered)==0),
                    fresh_score_against_current_frozen_bank=True,
                    actual_increment_response_error_H_over_sigma2_by_representative=Herror.tolist(),
                    actual_increment_response_error_K_over_sigma2_by_representative=Kerror.tolist())
            except Exception as exc:
                row.update(status='execution_failed',error=repr(exc),traceback=traceback.format_exc())
            io.write(folder/'receipt.json',row);rows.append(row)
            io.write(directory/'results.json',rows)
    io.write(directory/'results.json',rows)
    return rows


def run_case(entry):
    protocol,out=old.WORK['p'],old.WORK['out'];directory=out/'cases'/entry['id']
    protocol_ref=io.ref(out/'protocol.json')
    if (directory/'complete.json').exists():
        done=io.read(directory/'complete.json')
        if done['protocol']!=protocol_ref: raise ValueError('Completed case belongs to different protocol')
        paired.check_refs([done['results'],done['plans']]);return done
    directory.mkdir(parents=True,exist_ok=True)
    started=directory/'started.json'
    if not started.exists(): io.write(started,dict(pid=os.getpid(),created_utc=old.utc(),protocol=protocol_ref))
    elif io.read(started)['protocol']!=protocol_ref: raise ValueError('Case protocol changed')
    began=perf_counter();tick=perf_counter()
    planning,forecast,physical,case=context_planning(entry)
    if physical.config!=io.read(old.checked(entry['prepared_resource_case'])):
        raise ValueError('Prepared physical equipment changed')
    zero=np.zeros(case.shape);seeds={'zero':zero}
    if not case.audit(zero)['passed']: raise ValueError('Common zero seed is not resource feasible')
    seed_receipt=directory/'shared_initial_receipt.json'
    if not seed_receipt.exists():
        io.write(directory/'resource_case.json',physical.config)
        for key,name in (('equipment_spec','equipment_spec.json'),('baseline_device_states','baseline_device_states.json'),
                         ('baseline_decomposition','baseline_decomposition.json')):
            shutil.copyfile(old.checked(entry[key]),directory/name)
        band=io.read(old.checked(entry['node_time_bandwidth']))
        shutil.copyfile(old.checked(band['arrays']),directory/'bandwidth.npz')
        io.write(directory/'bandwidth.json',band)
        np.savez_compressed(directory/'shared_initial_points.npz',**seeds)
        shared=perf_counter()-tick
        io.write(seed_receipt,dict(seconds=shared,rule='One zero point only; no previous optimization actions',
            protocol=protocol_ref,points=io.ref(directory/'shared_initial_points.npz')))
    else:
        saved=io.read(seed_receipt)
        if saved['protocol']!=protocol_ref: raise ValueError('Seed protocol changed')
        points=paired.arrays(old.checked(saved['points']))
        if set(points)!=set(seeds) or not np.array_equal(points['zero'],zero): raise ValueError('Shared zero seed changed')
        shared=float(saved['seconds'])
    initial_ref=io.ref(directory/'shared_initial_points.npz')
    controls={'no_dfs':zero};plans={};models={}
    for mode in MODES:
        name='single_'+mode
        resumed=load_quote_checkpoint(directory,name,protocol_ref)
        if resumed is not None:
            plans[name],controls[name]=resumed
            model=SingleTargetEvidence(case,planning['score'],planning['val'],forecast,planning['base'],
                mode=mode,meter_relative_sd=.0005,forecast_relative_sd=.05)
            if plans[name]['new_gate_feasible'] and not (case.audit(controls[name])['passed'] and model.verify_expansion(controls[name])['feasible']):
                raise ValueError('Resumed completed quote fails original constraints')
            models[mode]=model;continue
        tick=perf_counter();model=None
        io.write(directory/(name+'_started.json'),dict(pid=os.getpid(),created_utc=old.utc(),mode=mode,
            protocol=protocol_ref,allowed_seconds=protocol['solver_seconds']))
        try:
            model=SingleTargetEvidence(case,planning['score'],planning['val'],forecast,planning['base'],
                mode=mode,meter_relative_sd=.0005,forecast_relative_sd=.05)
            remaining=max(.001,protocol['solver_seconds']-shared-(perf_counter()-tick))
            quote,point=engine.plan(case,planning['score'],planning['val'],forecast,planning['base'],
                mode=mode,seconds=remaining,env=old.WORK['env'],seed_points=seeds,
                record_directory=directory/(name+'_diagnostics'),meter_relative_sd=.0005,forecast_relative_sd=.05,
                evidence_factory=SingleTargetEvidence)
            if bool(quote['feasible'])!=(point is not None): raise ValueError('Point/quote feasibility mismatch')
            if point is not None and not (case.audit(point)['passed'] and model.verify_expansion(point)['feasible']):
                raise ArithmeticError('Returned action fails original physical/band/probability audit')
        except Exception as exc:
            quote,point=dict(status='exception',feasible=False,error=repr(exc),traceback=traceback.format_exc()),None
        action=zero.copy() if point is None else physical.array(point).copy()
        quote.update(mode=mode,guard_kind='equipment_node_hour_pq',new_gate_feasible=point is not None,
            zero_fallback=point is None,executed_source='zero_fallback' if point is None else 'verified_equipment_node_hour_band',
            executed_probability_check=dict(feasible=False,status='evidence_construction_failed') if model is None else model.verify_expansion(action),
            executed_cost=prices.cost_breakdown(physical,action),target_native=entry['planned_history_target_native'],
            planned_band=case.band_audit(action),equipment_audit=physical.audit(action),initial_points=initial_ref,
            shared_preparation_seconds=shared,requested_quote_budget_seconds=protocol['solver_seconds'],
            quote_wall_seconds=perf_counter()-tick,total_method_seconds_with_shared_preparation=perf_counter()-tick+shared,
            band_encoding=case.band_encoding,prior_optimizer_time_reused=False,
            raw_bound_scope='Activation cost for this grid cell frozen bank, equipment domain, encoded fresh training band and original single-target probability constraints')
        np.savez_compressed(directory/(name+'_action.npz'),action=action)
        io.write(directory/(name+'_plan.json'),quote)
        io.write(directory/(name+'_completed.json'),dict(protocol=protocol_ref,
            plan=io.ref(directory/(name+'_plan.json')),action=io.ref(directory/(name+'_action.npz'))))
        controls[name],plans[name],models[mode]=action,quote,model
        print(entry['id'],name,quote['status'],'fee',quote.get('cost_yuan'),'fallback',point is None,flush=True)
    seal_path=directory/'seal.json'
    if not seal_path.exists():
        np.savez_compressed(directory/'planner.npz',forecast=forecast,**controls)
        io.write(directory/'plans.json',plans)
        io.write(directory/'prediction_audits.json',{mode:{name:None if model is None else model.verify_expansion(u)
            for name,u in controls.items()} for mode,model in models.items()})
        io.write(directory/'requested_device_states.json',{name:physical.device_states(u) for name,u in controls.items()})
        io.write(seal_path,dict(created_utc=old.utc(),protocol=protocol_ref,
            planner=io.ref(directory/'planner.npz'),plans=io.ref(directory/'plans.json'),score=entry['score'],
            train=entry['files']['train.npz'],validation=entry['files']['validation.npz'],
            equipment_spec=io.ref(directory/'equipment_spec.json'),resource_case=io.ref(directory/'resource_case.json'),
            baseline_device_states=io.ref(directory/'baseline_device_states.json'),baseline_decomposition=io.ref(directory/'baseline_decomposition.json'),
            bandwidth_json=io.ref(directory/'bandwidth.json'),bandwidth_npz=io.ref(directory/'bandwidth.npz'),
            sampling_source=entry['files']['observations_sealed.json'],shared_initial_points=initial_ref,
            prediction_audits=io.ref(directory/'prediction_audits.json'),requested_device_states=io.ref(directory/'requested_device_states.json'),
            planned_history_target_native=entry['planned_history_target_native'],planner_actual_voltage_or_network_inputs=False))
    else:
        seal=io.read(seal_path)
        if seal['protocol']!=protocol_ref: raise ValueError('Sealed protocol changed')
        saved=paired.arrays(old.checked(seal['planner']))
        if not np.array_equal(saved['forecast'],forecast) or any(not np.array_equal(saved[k],u) for k,u in controls.items()):
            raise ValueError('Quoted controls differ from the saved planning seal')
        if io.read(old.checked(seal['plans']))!=plans: raise ValueError('Sealed quote receipts changed')
    rows=execute(entry,protocol,directory,planning,physical,case,forecast,controls,plans)
    good=[row for row in rows if row['status']=='complete']
    result=dict(status='complete',id=entry['id'],protocol=protocol_ref,rows=len(rows),failed_rows=len(rows)-len(good),
        expected_rows=4*protocol['repeats'],quote_exceptions=sum(q['status']=='exception' for q in plans.values()),
        feasible_quotes=sum(q['feasible'] for q in plans.values()),zero_fallbacks=sum(q['zero_fallback'] for q in plans.values()),
        zero_sufficient_quotes=sum(q['status']=='zero_sufficient' for q in plans.values()),
        reused_ac_rows=sum(bool(row['reused_zero_physical_cache']) for row in good),
        new_ac_rows=sum(not row['reused_zero_physical_cache'] for row in good),seconds=perf_counter()-began,
        shared_preparation_seconds=shared,results=io.ref(directory/'results.json'),plans=io.ref(directory/'plans.json'))
    io.write(directory/'complete.json',result)
    return result


def run(out):
    out=Path(out).resolve();protocol=io.read(out/'protocol.json')
    paired.check_refs([protocol['inputs'],*protocol['sources']])
    if (out/'lock.json').exists() or (out/'complete.json').exists():
        raise RuntimeError('Preserve active locks and completed runs')
    io.write(out/'lock.json',dict(pid=os.getpid(),created_utc=old.utc()))
    receipts,failures=[],[]
    try:
        with ProcessPoolExecutor(max_workers=protocol['workers'],initializer=old.initialize,initargs=(str(out),)) as pool:
            tasks={pool.submit(run_case,entry):entry for entry in io.read(old.checked(protocol['inputs']))}
            for task in as_completed(tasks):
                try: receipts.append(task.result())
                except Exception as exc:
                    failures.append(dict(id=tasks[task]['id'],error=repr(exc),traceback=traceback.format_exc(),
                        expected_policy_rows=4*protocol['repeats']))
                io.write(out/'progress.json',dict(receipts=receipts,failed_tasks=failures,
                    expected_rows=protocol['expected_rows'],unavailable_policy_rows=protocol['unavailable_policy_rows']))
        result=dict(status='complete' if not failures and protocol['unavailable_inputs']==0 else 'incomplete',
            receipts=receipts,failed_tasks=failures,expected_rows=protocol['expected_rows'],
            planned_inputs=protocol['planned_inputs'],prepared_inputs=protocol['prepared_inputs'],
            unavailable_inputs=protocol['unavailable_inputs'],preparation_failures=protocol['preparation_failures'],
            **{key:sum(r[key] for r in receipts) for key in ('rows','failed_rows','quote_exceptions','feasible_quotes',
                'zero_fallbacks','zero_sufficient_quotes','reused_ac_rows','new_ac_rows')},
            source_unchanged=all(io.ref(ref['path'])==ref for ref in protocol['sources']))
        result['missing_policy_rows']=result['expected_rows']-result['rows']
        io.write(out/'complete.json',result)
        return result
    finally: (out/'lock.json').unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--seconds',type=float,default=1200.)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--preflight',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not args.resume:
        if args.index is None: parser.error('--index required for preparation')
        prepare(args.index,args.out,smoke=args.smoke,seconds=args.seconds,workers=args.workers,preflight=args.preflight)
    if not args.prepare_only: print(run(args.out))


if __name__=='__main__': main()
