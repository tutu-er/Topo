"""Independent bookkeeping audit for a frozen single-bank 11-method batch.

No AC fitter or planner is imported. It checks loader declarations, history
counts, isolation flags, saved cost arithmetic and planning-ancestor charges.
Optional full raw records permit checking selection metadata and dependencies.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import math
from pathlib import Path

from . import ev_reliability_v2 as rel
from . import ev_systematic_analyze as base

EXPECTED_METHODS=('none','constant','highlow','square_fixed','square_mobile','scalar_fixed',
    'scalar_mobile','cap_fixed','cap_mobile','jointgrad_fixed','jointgrad_mobile')
FALSE_FLAGS=frozenset(('uses_truth','uses_validation','uses_future_noise','truth_used_in_fit',
    'selector_uses_truth','selector_uses_validation'))
FORBIDDEN_SELECTION_KEYS=frozenset(('truth','true_name','future_observations','future_noise','validation'))


def _finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _close(left,right):
    return _finite(left) and _finite(right) and abs(left-right)<=1e-8*(1+abs(right))


def _metadata_flags(value,path='selection'):
    counts=Counter();errors=[]
    if isinstance(value,dict):
        for key,row in value.items():
            label=path+'.'+key
            if key in FALSE_FLAGS:
                counts[key]+=1
                if row is not False:errors.append(label+' must be false')
            if key in FORBIDDEN_SELECTION_KEYS:errors.append(label+' contains a forbidden input field')
            if isinstance(row,(dict,list,tuple)):
                sub,bad=_metadata_flags(row,label);counts.update(sub);errors.extend(bad)
    elif isinstance(value,(list,tuple)):
        for i,row in enumerate(value):
            if isinstance(row,(dict,list,tuple)):
                sub,bad=_metadata_flags(row,path+'['+str(i)+']');counts.update(sub);errors.extend(bad)
    return counts,errors


def _cost_record(record, *, raw=None):
    errors=[];checks=Counter();methods=base._methods(record);common=record.get('common_cost',{})
    status=record.get('status');incomplete=status!='complete'
    if not _finite(common.get('training_seconds')):
        errors.append('training cost missing or nonfinite')
    for name,method in methods.items():
        planning=method.get('planning_seconds');own=method.get('planning_actual_seconds')
        if incomplete and not _finite(planning):continue
        if not _finite(planning) or planning<0:errors.append(name+': invalid planning cost')
        if not _finite(own) or own<0:errors.append(name+': invalid own planning cost')
        if _finite(planning) and _finite(own) and planning+1e-10<own:
            errors.append(name+': standalone cost omits own planning work')
        for mode,value in method['inference'].items():
            fields=(common.get('training_seconds'),planning,value.get('seconds'),common.get('fixed_history_seconds'))
            if incomplete and not all(_finite(v) for v in fields):continue
            if not all(_finite(v) for v in fields):
                errors.append(name+'/'+mode+': missing constituent mode cost');continue
            expected=sum(fields)
            if mode=='joint_total':
                if not _finite(common.get('joint_history_seconds')):
                    errors.append(name+'/'+mode+': missing joint history cost');continue
                expected+=common['joint_history_seconds']
            checks['inference_total_cost']+=1
            if not _close(value.get('total_seconds'),expected):
                errors.append(name+'/'+mode+': mode total cost arithmetic mismatch')
            checks['joint_witness_dependency_flag']+=1
            if value.get('joint_pays_fixed_source_witness_dependencies') is not (mode=='joint_total'):
                errors.append(name+'/'+mode+': witness dependency flag differs')
        values=(common.get('training_seconds'),planning,method.get('evaluation_seconds'),
                common.get('combined_adapter_preparation_seconds'))
        if all(_finite(v) for v in values):
            checks['all_inference_total_cost']+=1
            if not _close(method.get('total_seconds'),sum(values)):
                errors.append(name+': total cost arithmetic mismatch')
        elif not incomplete:errors.append(name+': missing all-inference cost constituent')
    if raw is not None and raw.get('selection'):
        selection=raw['selection'];meta=selection.get('methods',{})
        if set(meta)!=set(methods):errors.append('selection method set differs')
        context=selection.get('context_seconds')
        flags,bad=_metadata_flags(selection);checks.update({'flag_'+k:v for k,v in flags.items()});errors.extend(bad)
        for name in methods:
            if name not in meta:continue
            row=meta[name];ancestors=set();visiting=set()
            def visit(key):
                if key in visiting:raise ValueError('planning dependency cycle')
                if key in ancestors:return
                if key not in meta:raise ValueError('unknown planning ancestor '+key)
                visiting.add(key)
                for parent in meta[key].get('dependencies',[]):visit(parent)
                visiting.remove(key);ancestors.add(key)
            try:
                for parent in row.get('dependencies',[]):visit(parent)
                constituents=[meta[k].get('planning_actual_seconds') for k in ancestors]
                constituents.append(row.get('planning_actual_seconds'))
                uses_context=name.startswith(('scalar_','cap_','jointgrad_'))
                if uses_context:constituents.append(context)
                checks['planning_ancestor_cost']+=1
                if not all(_finite(v) for v in constituents) or not _close(row.get('planning_seconds'),sum(constituents)):
                    errors.append(name+': ancestor/context charge mismatch')
                if 'standalone_ancestors' in row and set(row['standalone_ancestors'])!=ancestors:
                    errors.append(name+': declared ancestor set differs from closure')
                if not _close(methods[name].get('planning_seconds'),row.get('planning_seconds')):
                    errors.append(name+': method and selection costs differ')
            except ValueError as exc:errors.append(name+': '+str(exc))
    return dict(success=not errors,checks=dict(checks),errors=errors)


def audit_phase(phase, *, expected_count, raw_directory=None):
    records=phase['records'];errors=[];checks=Counter();rows=[];raw_files=[]
    methods=rel.declared_methods(phase)
    if methods!=EXPECTED_METHODS:errors.append('frozen method order/set differs from the declared eleven')
    if len(records)!=expected_count:errors.append('history count differs from expected')
    if len({r['seed'] for r in records})!=len(records):errors.append('independent histories require unique seeds')
    config=phase.get('manifest',{}).get('config',{})
    for key in ('selector_uses_truth','selector_uses_validation'):
        checks['freeze_selector_flags']+=1
        if config.get(key) is not False:errors.append('freeze '+key+' must be false')
    if config.get('PQ_direct_inputs') is not True or config.get('root_and_bias_fixed') is not True:
        errors.append('freeze direct-PQ/fixed-root/bias model differs')
    n_values=set();banks=set();method_counts={name:Counter() for name in methods};costs={name:defaultdict(list) for name in methods}
    for record in records:
        errors_for_row=[];parameters,structural_error=rel.candidate_parameters(record)
        if structural_error:errors_for_row.append(structural_error)
        else:
            n_values.update(p['n'] for p in parameters.values())
            banks.add(tuple(sorted(p['structural_key'] for p in parameters.values())))
        if record.get('source_unchanged') is not True:errors_for_row.append('record source snapshot flag missing/false')
        if record.get('status')=='complete' and record.get('line_parameters_unchanged_during_planning') is not True:
            errors_for_row.append('trained reference parameter immutability flag missing/false')
        raw=None
        if raw_directory is not None:
            candidates=list(Path(raw_directory).glob('*_'+str(record['seed'])+'.json.gz'))
            if len(candidates)!=1:errors_for_row.append('expected one full raw history')
            else:
                path=candidates[0];raw=base._read(path)
                raw_files.append(dict(seed=record['seed'],path=str(path.resolve()),sha256=sha256(path.read_bytes()).hexdigest()))
                if (raw['case_id'],raw['seed'],raw['phase'])!=(record['case_id'],record['seed'],record['phase']):
                    errors_for_row.append('raw/projection case identity differs')
        cost=_cost_record(record,raw=raw);checks.update(cost['checks']);errors_for_row.extend(cost['errors'])
        for method in record['methods']:
            name=method['name'];count=method_counts[name]
            count['histories']+=1;count['planning_failure']+=bool(method.get('planning_failure'))
            count['operation_failure']+=not bool(method.get('operation',{}).get('success'))
            count['moves']+=int(method.get('operation',{}).get('audit',{}).get('moves',0))
            for mode,value in method['inference'].items():
                count['incomplete_'+mode]+=not bool(value.get('complete'))
                if _finite(value.get('total_seconds')):costs[name][mode].append(value['total_seconds'])
            if _finite(method.get('planning_seconds')):costs[name]['planning'].append(method['planning_seconds'])
            if _finite(method.get('operation',{}).get('driver_cost')):costs[name]['driver_minutes'].append(method['operation']['driver_cost'])
        rows.append(dict(seed=record['seed'],case_id=record['case_id'],status=record['status'],success=not errors_for_row,errors=errors_for_row))
    if len(n_values)!=1 or len(banks)!=1:errors.append('batch must contain a single physical candidate bank')
    return dict(schema='ev_mobile_v2_analysis_audit_v1',phase=phase['phase'],history_count=len(records),
        expected_count=expected_count,methods=list(methods),terminal_counts=sorted(n_values),
        source_files=phase['files'],raw_files=raw_files,identity=phase['identity'],
        success=not errors and all(r['success'] for r in rows),global_errors=errors,checks=dict(checks),
        method_failures={k:dict(v) for k,v in method_counts.items()},
        method_costs={k:{mode:base._stats(v) for mode,v in modes.items()} for k,modes in costs.items()},
        histories=rows,scope='flags and saved arithmetic only; no claim that flags alone prove absence of data leakage')


def audit_pair(calibration,evaluation,*,calibration_raw=None,evaluation_raw=None,calibration_count=96,evaluation_count=48):
    cal=audit_phase(calibration,expected_count=calibration_count,raw_directory=calibration_raw)
    ev=audit_phase(evaluation,expected_count=evaluation_count,raw_directory=evaluation_raw)
    errors=[]
    try:base._identity_match(calibration['identity'],evaluation['identity'])
    except ValueError as exc:errors.append(str(exc))
    overlap=sorted({r['seed'] for r in calibration['records']}&{r['seed'] for r in evaluation['records']})
    if overlap:errors.append('calibration and evaluation histories overlap')
    if cal['terminal_counts']!=ev['terminal_counts']:errors.append('calibration/evaluation physical banks differ')
    return dict(success=cal['success'] and ev['success'] and not errors,errors=errors,overlapping_seeds=overlap,
                calibration=cal,evaluation=ev)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration',type=Path,required=True);parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--calibration-raw',type=Path);parser.add_argument('--evaluation-raw',type=Path)
    parser.add_argument('--calibration-count',type=int,default=96);parser.add_argument('--evaluation-count',type=int,default=48)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=audit_pair(rel.read_phase(args.calibration,expected_phase='calibration'),rel.read_phase(args.evaluation,expected_phase='evaluation'),
        calibration_raw=args.calibration_raw,evaluation_raw=args.evaluation_raw,calibration_count=args.calibration_count,evaluation_count=args.evaluation_count)
    base._write(args.output,result)
    if not result['success']:raise SystemExit(1)


if __name__=='__main__':main()
