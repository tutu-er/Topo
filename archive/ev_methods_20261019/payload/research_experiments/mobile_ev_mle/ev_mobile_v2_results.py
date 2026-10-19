"""Compact descriptive results for the frozen eleven-policy EV study.

This module changes no score, rule, action or fitted parameter. It distinguishes
paired case-history slots from independent base seeds and direct gradient
changes from inherited plans. Formal calibration/evaluation is performed by
ev_reliability_v2; this module consumes those already frozen thresholds.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from hashlib import sha256
import itertools
from pathlib import Path

from . import ev_reliability_v2 as rel
from . import ev_systematic_analyze as base

MOBILE_FAMILIES=('square','scalar','cap','jointgrad')


def history_key(record):
    return str(record['case_id'])+':'+str(record['seed'])


def curve_difference(a,b, *, net=False):
    if not isinstance(a,list) or not isinstance(b,list) or len(a)!=len(b):return None
    if any(not isinstance(x,list) or not isinstance(y,list) or len(x)!=len(y) for x,y in zip(a,b)):return None
    if net:return max((abs(sum(x)-sum(y)) for x,y in zip(a,b)),default=0.)
    return max((abs(x-y) for ar,br in zip(a,b) for x,y in zip(ar,br)),default=0.)


def _near(a,b):
    return base._finite(a) and base._finite(b) and abs(a-b)<=base.TIE_ABS+base.TIE_REL*max(abs(a),abs(b))


def _delta(a,b):
    return a-b if base._finite(a) and base._finite(b) else None


def _wtl(rows):
    counts=Counter();values=[]
    for left,right in rows:
        if not base._finite(left) or not base._finite(right):counts['incomplete']+=1;continue
        value=left-right;values.append(value)
        counts['tie' if _near(left,right) else 'win' if value>0 else 'loss']+=1
    return dict(denominator=len(rows),valid_pairs=len(values),
        **{key:counts[key] for key in ('win','tie','loss','incomplete')},delta=base._stats(values),
        tie_atol=base.TIE_ABS,tie_rtol=base.TIE_REL)


def _decisions(record,thresholds):
    if thresholds is None:return None
    return {rule:{name:{mode:rel.decision(record,name,mode,rule,thresholds['rules'][rule][name][mode])
                for mode in base.INFERENCE} for name in thresholds['methods']} for rule in rel.RULES}


def _decision_counts(rows):
    counts=Counter(row['category'] for row in rows)
    return dict(denominator=len(rows),
        **{key:counts[key] for key in ('unique_correct','unique_wrong','unresolved','failed_keep_all','candidate_miss')},
        false_reject=sum(r['false_reject'] is True for r in rows),
        empty_set=sum(not r['retained'] for r in rows),
        retained_count=base._stats(len(r['retained']) for r in rows))


def _guard_metadata(value):
    if not isinstance(value,dict):return None
    result={key:deepcopy(value.get(key)) for key in ('success','complete','reference_floor',
        'reference_values','slacks','numerical_tolerance','tolerance_coefficient',
        'tolerance_rule','violated_references','scope')}
    slacks=value.get('slacks') or {};numeric=value.get('numerical_tolerance') or {}
    actual=[v for v in slacks.values() if base._finite(v)]
    effective=[v+numeric[k] for k,v in slacks.items() if base._finite(v) and base._finite(numeric.get(k))]
    result.update(min_raw_slack=min(actual) if actual else None,
                  min_effective_slack=min(effective) if effective else None)
    return result


def _gradient_metadata(raw):
    starts=raw.get('starts',[]);attempts=[r for start in starts for r in start.get('iterations',[])]
    failure_reasons=Counter(str(r.get('failure')) for r in attempts if r.get('failure'))
    counters=raw.get('counters',{})
    initial_guards=[r['reference_guard'] for r in raw.get('initial_candidates',[]) if isinstance(r.get('reference_guard'),dict)]
    proposal_guards=[r['reference_guard'] for r in attempts if isinstance(r.get('reference_guard'),dict)]
    nearest=raw.get('selected_profile',{}).get('aggregation',{}).get('nearest',[])
    return dict(success=bool(raw.get('success')),selected_origin=raw.get('selected_origin'),
        selected_score=raw.get('selected_score'),initial_best_score=raw.get('initial_best_score'),
        selected_reference_nearest_gaps={row['a']:row['half_gap'] for row in nearest},
        selected_predictive_gain=_delta(raw.get('selected_score'),raw.get('initial_best_score')),
        direct_gradient_selected=raw.get('selected_origin')=='joint_gradient_MILP',
        gradient_MILP_calls=raw.get('gradient_MILP_calls',len(attempts)),
        gradient_MILP_accepted=raw.get('gradient_MILP_accepted',sum(bool(r.get('accepted')) for r in attempts)),
        logged_gradient_calls=len(attempts),logged_accepted=sum(bool(r.get('accepted')) for r in attempts),
        accepted_not_final=bool(any(r.get('accepted') for r in attempts) and raw.get('selected_origin')!='joint_gradient_MILP'),
        initial_plan_count=len(raw.get('initial_candidates',[])),
        initial_complete_count=sum(bool(r.get('success')) for r in raw.get('initial_candidates',[])),
        reference_guard_enabled=raw.get('reference_guard_enabled'),
        reference_guard_requested=raw.get('reference_guard_requested',raw.get('reference_guard_enabled')),
        reference_floor=deepcopy(raw.get('reference_floor')),
        reference_floor_parent=raw.get('reference_floor_parent'),reference_floor_cap=raw.get('reference_floor_cap'),
        guard_tolerance_coefficient=raw.get('guard_tolerance_coefficient'),
        initial_guard_checked=len(initial_guards),
        initial_guard_rejected_complete=sum(g.get('complete') is True and g.get('success') is False for g in initial_guards),
        initial_guard_incomplete=sum(g.get('complete') is not True for g in initial_guards),
        gradient_attempts_not_accepted=sum(not bool(r.get('accepted')) for r in attempts),
        proposal_guard_checked=len(proposal_guards),
        proposal_guard_rejected_complete=sum(g.get('complete') is True and g.get('success') is False for g in proposal_guards),
        proposal_guard_incomplete=sum(g.get('complete') is not True for g in proposal_guards),
        selected_reference_guard=_guard_metadata(raw.get('selected_reference_guard')),
        completed_requested_gradient_iterations=raw.get('completed_requested_gradient_iterations'),
        time_budget_exceeded=raw.get('time_budget_exceeded'),
        failure=raw.get('failure'),gradient_step_failure_reasons=dict(failure_reasons),
        counters={key:counters.get(key) for key in ('profile_evaluations','cache_hits','joint_fit_calls',
            'fixed_source_witness_fit_calls','actual_profile_seconds')})


def prediction_rank_diagnostic(phase,selection_metadata,*,tolerance=1e-7):
    """Pure post-hoc projection: mean task tradeoff versus prediction reversal.

    Takes already projected records/metadata, reads no files and calls no fit,
    optimizer or selector. True labels identify the realized reference only
    after the action has been selected. g/Phi use half-SSE; M uses full SSE.
    """
    if not base._finite(tolerance) or tolerance<0:raise ValueError('finite nonnegative diagnostic tolerance required')
    rows=[]
    for record in phase['records']:
        metadata=selection_metadata.get('rows',{}).get(history_key(record),{})
        f=metadata.get('jointgrad_fixed',{}).get('gradient') or {}
        m=metadata.get('jointgrad_mobile',{}).get('gradient') or {}
        truth=record.get('true_name');fg=f.get('selected_reference_nearest_gaps') or {}
        mg=m.get('selected_reference_nearest_gaps') or {}
        fixed=base._mode(record,'jointgrad_fixed','joint_total');mobile=base._mode(record,'jointgrad_mobile','joint_total')
        mean=_delta(m.get('selected_score'),f.get('selected_score'))
        true=_delta(mg.get(truth),fg.get(truth));actual=_delta(mobile.get('M'),fixed.get('M'))
        complete=bool(fixed['complete'] and mobile['complete'])
        if not complete:actual=None
        mean_reversal=mean is not None and actual is not None and mean>tolerance and actual<-tolerance
        tradeoff=mean is not None and true is not None and mean>tolerance and true<-tolerance
        true_reversal=true is not None and actual is not None and true>tolerance and actual<-tolerance
        rows.append(dict(key=history_key(record),case_id=record['case_id'],seed=record['seed'],true_name=truth,
            delta_mean_predictive=mean,delta_true_reference_predictive=true,delta_actual_M=actual,
            inference_complete=complete,metadata_available=bool(f and m),true_reference_available=true is not None,
            mean_sacrifices_true_reference=bool(tradeoff),mean_prediction_reversal=bool(mean_reversal),
            true_reference_prediction_reversal=bool(true_reversal),
            mean_reversal_with_tradeoff=bool(mean_reversal and true is not None and true<-tolerance),
            mean_reversal_with_true_prediction_increase=bool(mean_reversal and true_reversal)))
    def counts(items):
        return dict(denominator=len(items),metadata_missing=sum(not r['metadata_available'] for r in items),
            true_reference_missing=sum(not r['true_reference_available'] for r in items),
            inference_incomplete=sum(not r['inference_complete'] for r in items),
            mean_M_comparable=sum(r['delta_mean_predictive'] is not None and r['delta_actual_M'] is not None for r in items),
            all_three_comparable=sum(all(r[k] is not None for k in
                ('delta_mean_predictive','delta_true_reference_predictive','delta_actual_M')) for r in items),
            **{name:sum(r[name] for r in items) for name in ('mean_sacrifices_true_reference',
                'mean_prediction_reversal','true_reference_prediction_reversal','mean_reversal_with_tradeoff',
                'mean_reversal_with_true_prediction_increase')})
    return dict(schema='EV_predictive_rank_posthoc_v1',tolerance=tolerance,by_history=rows,overall=counts(rows),
        by_case={name:counts([r for r in rows if r['case_id']==name]) for name in sorted({r['case_id'] for r in rows})},
        scope='Post-hoc signs only; prediction tradeoffs, noisy outcomes and different refitted parameters remain separate.',
        units=dict(delta_mean_predictive='half-SSE',delta_true_reference_predictive='half-SSE',delta_actual_M='SSE'),
        uses_future_labels_only_after_action=True,changes_selection=False)


def extract_selection_metadata(phase, raw_directory):
    """Project raw selection logs once, without retaining their large AC states."""
    raw_directory=Path(raw_directory);rows={};files=[]
    for record in phase['records']:
        filename='case_'+str(record.get('index',0)).zfill(3)+'_'+str(record['seed'])+'.json.gz'
        path=raw_directory/filename
        if not path.exists():
            choices=list(raw_directory.glob('*_'+str(record['seed'])+'.json.gz'))
            matches=[]
            for choice in choices:
                value=base._read(choice)
                if value.get('case_id')==record['case_id']:matches.append(choice)
            if len(matches)!=1:raise ValueError('expected exactly one raw record for '+history_key(record))
            path=matches[0]
        raw=base._read(path)
        if (raw['case_id'],raw['seed'])!=(record['case_id'],record['seed']):raise ValueError('raw/projection history differs')
        selection=raw.get('selection',{});methods=selection.get('methods',{});projected={}
        for name in rel.declared_methods(phase):
            row=methods.get(name);payload={} if row is None else row.get('raw',{})
            projected[name]=dict(metadata_available=row is not None,planning_success=None if row is None else row.get('success'),
                selected_origin=payload.get('selected_origin'),selected_score=payload.get('selected_score'),
                initial_best_score=payload.get('initial_best_score'),failure=payload.get('failure'),
                standalone_ancestors=[] if row is None else row.get('standalone_ancestors',[]),
                dependencies=[] if row is None else row.get('dependencies',[]))
            if name.startswith('jointgrad_'):
                projected[name]['gradient']=_gradient_metadata(payload) if row is not None else None
        rows[history_key(record)]=projected
        files.append(dict(key=history_key(record),path=str(path.resolve()),sha256=sha256(path.read_bytes()).hexdigest()))
    return dict(schema='ev_mobile_v2_selection_projection_v1',rows=rows,source_files=files,
        scope='primitive selection metadata only; excludes AC states and future meter arrays')


def _history(record,methods,thresholds,selection):
    raw=base._methods(record);decisions=_decisions(record,thresholds);values={}
    metadata=selection.get('rows',{}).get(history_key(record),{}) if selection else {}
    for name in methods:
        row=raw[name];op=row.get('operation',{});audit=op.get('audit',{});perr=row.get('parameter_error',{})
        values[name]=dict(planning_failure=row.get('planning_failure'),operation_success=op.get('success'),
            moves=audit.get('moves'),driver_minutes=op.get('driver_cost',audit.get('driver_cost')),
            travel_energy_kwh=audit.get('travel_energy_kwh'),grid_energy_kwh=audit.get('grid_energy_kwh'),
            throughput_kwh=audit.get('throughput_kwh'),conversion_loss_kwh=audit.get('conversion_loss_kwh'),
            planning_seconds=row.get('planning_seconds'),planning_actual_seconds=row.get('planning_actual_seconds'),
            fixed_R_error=perr.get('training_R'),fixed_X_error=perr.get('training_X'),
            joint_R_error=perr.get('joint_R'),joint_X_error=perr.get('joint_X'),
            inference={mode:base._mode(record,name,mode) for mode in base.INFERENCE},
            selection=deepcopy(metadata.get(name)))
        if decisions is not None:
            values[name]['decisions']={rule:{mode:dict(category=decisions[rule][name][mode]['category'],
                retained=decisions[rule][name][mode]['retained'],false_reject=decisions[rule][name][mode]['false_reject'])
                for mode in base.INFERENCE} for rule in rel.RULES}
    if 'jointgrad_mobile' in metadata:
        mobile=values['jointgrad_mobile'].get('selection')
        fixed=values['jointgrad_fixed'].get('selection')
        if mobile and mobile.get('gradient'):
            mobile['gradient']['inherits_fixed_gradient']=bool(mobile['selected_origin']=='fixed_parent'
                and fixed and fixed.get('gradient') and fixed['gradient']['direct_gradient_selected'])
    pairs={}
    for family in MOBILE_FAMILIES:
        left=family+'_fixed';right=family+'_mobile'
        if left not in raw or right not in raw:continue
        a=values[left];b=values[right]
        pair=dict(fixed=left,mobile=right,moves=b['moves'],driver_minutes=b['driver_minutes'],
            power_matrix_max_difference_kw=curve_difference(raw[left].get('power_kw'),raw[right].get('power_kw')),
            net_profile_max_difference_kw=curve_difference(raw[left].get('power_kw'),raw[right].get('power_kw'),net=True),
            delta_M={mode:_delta(b['inference'][mode]['M'],a['inference'][mode]['M']) for mode in base.INFERENCE},
            decision_changes={})
        if decisions is not None:
            for rule in rel.RULES:
                pair['decision_changes'][rule]={}
                for mode in base.INFERENCE:
                    x=decisions[rule][left][mode];y=decisions[rule][right][mode]
                    same_losses=x['complete'] and y['complete'] and set(x['losses'])==set(y['losses']) and all(
                        _near(x['losses'][k],y['losses'][k]) for k in x['losses'])
                    pair['decision_changes'][rule][mode]=dict(fixed=x['category'],mobile=y['category'],
                        fixed_retained=x['retained'],mobile_retained=y['retained'],
                        retained_changed=x['retained']!=y['retained'],same_all_candidate_losses=same_losses,
                        same_losses_different_retained=bool(same_losses and x['retained']!=y['retained']),
                        correct_gain=y['category']=='unique_correct' and x['category']!='unique_correct',
                        correct_loss=x['category']=='unique_correct' and y['category']!='unique_correct',
                        false_reject_reduced=x['false_reject'] is True and y['false_reject'] is False,
                        false_reject_increased=x['false_reject'] is False and y['false_reject'] is True)
        pairs[family]=pair
    return dict(key=history_key(record),case_id=record['case_id'],seed=record['seed'],true_name=record['true_name'],
        H=record.get('config',{}).get('training_points'),status=record['status'],candidate_miss=record['candidate_miss'],
        methods=values,mobility_pairs=pairs)


def _gradient_summary(rows,name):
    present=[r['methods'][name]['selection']['gradient'] for r in rows if r['methods'][name].get('selection')
        and r['methods'][name]['selection'].get('gradient') is not None]
    if not name.startswith('jointgrad_'):return None
    origins=Counter(r.get('selected_origin') or 'unavailable_or_failed' for r in present)
    selected_guards=[r['selected_reference_guard'] for r in present if r.get('selected_reference_guard') is not None]
    floor_values=defaultdict(list)
    for r in present:
        for key,value in (r.get('reference_floor') or {}).items():
            if base._finite(value):floor_values[key].append(value)
    return dict(denominator=len(rows),metadata_available=len(present),metadata_missing=len(rows)-len(present),
        selected_origins=dict(origins),direct_gradient_selected=sum(r['direct_gradient_selected'] for r in present),
        inherits_fixed_gradient=sum(bool(r.get('inherits_fixed_gradient')) for r in present),
        accepted_step_without_final_gradient=sum(r['accepted_not_final'] for r in present),
        gradient_MILP_calls=sum(r['gradient_MILP_calls'] for r in present),
        gradient_MILP_accepted=sum(r['gradient_MILP_accepted'] for r in present),
        reference_guard_enabled=sum(r.get('reference_guard_enabled') is True for r in present),
        reference_guard_requested=sum(r.get('reference_guard_requested') is True for r in present),
        reference_floors={key:base._stats(values) for key,values in sorted(floor_values.items())},
        guard_counts={key:sum(r.get(key,0) for r in present) for key in ('initial_guard_checked',
            'initial_guard_rejected_complete','initial_guard_incomplete','gradient_attempts_not_accepted',
            'proposal_guard_checked','proposal_guard_rejected_complete','proposal_guard_incomplete')},
        selected_guard_checked=len(selected_guards),
        selected_guard_passed=sum(g.get('success') is True for g in selected_guards),
        selected_guard_failed=sum(g.get('success') is False for g in selected_guards),
        selected_guard_missing_when_requested=sum(r.get('reference_guard_requested') is True
            and r.get('selected_reference_guard') is None for r in present),
        selected_guard_min_raw_slack=base._stats(g['min_raw_slack'] for g in selected_guards),
        selected_guard_min_effective_slack=base._stats(g['min_effective_slack'] for g in selected_guards),
        selected_predictive_gain=base._stats(r['selected_predictive_gain'] for r in present),
        budget_overrun=sum(bool(r['time_budget_exceeded']) for r in present),
        completed_requested_iterations=sum(r['completed_requested_gradient_iterations'] is True for r in present),
        counters={key:base._stats(r['counters'].get(key) for r in present) for key in
            ('profile_evaluations','cache_hits','joint_fit_calls','fixed_source_witness_fit_calls','actual_profile_seconds')})


def _summarize(rows,methods,has_rules):
    seeds=defaultdict(list)
    for row in rows:seeds[row['seed']].append(row['case_id'])
    result=dict(case_history_slots=len(rows),unique_base_seeds=len(seeds),
        paired_seed_clusters={str(k):v for k,v in sorted(seeds.items()) if len(v)>1},
        status_counts=dict(Counter(r['status'] for r in rows)),
        truth_counts=dict(Counter(r['true_name'] for r in rows)),methods={},mobility={})
    for name in methods:
        values=[r['methods'][name] for r in rows]
        item=dict(denominator=len(rows),planning_failure=sum(bool(r['planning_failure']) for r in values),
            operation_failure=sum(r['operation_success'] is False for r in values),
            actual_moved=sum(base._finite(r['moves']) and r['moves']>0 for r in values),
            moves=base._stats(r['moves'] for r in values),driver_minutes=base._stats(r['driver_minutes'] for r in values),
            planning_seconds=base._stats(r['planning_seconds'] for r in values),
            planning_actual_seconds=base._stats(r['planning_actual_seconds'] for r in values),
            parameter_error={key:base._stats(r[key] for r in values) for key in
                ('fixed_R_error','fixed_X_error','joint_R_error','joint_X_error')},inference={},
            selection_origins=dict(Counter((r['selection'] or {}).get('selected_origin') or 'unavailable_or_public' for r in values)),
            gradient=_gradient_summary(rows,name))
        baseline='square_mobile' if name.endswith('_mobile') else 'square_fixed'
        for mode in base.INFERENCE:
            inf=[r['inference'][mode] for r in values]
            item['inference'][mode]=dict(denominator=len(rows),complete=sum(r['complete'] for r in inf),
                incomplete=sum(not r['complete'] for r in inf),candidate_miss=sum(r['candidate_miss'] for r in inf),
                M=base._stats(r['M'] for r in inf),total_seconds=base._stats(r['total_seconds'] for r in inf),
                raw_unique_correct=sum(r['minimum_candidates']==[history['true_name']] for r,history in zip(inf,rows)),
                versus_same_mobility_square=_wtl([(r['methods'][name]['inference'][mode]['M'],
                    r['methods'][baseline]['inference'][mode]['M']) for r in rows]))
        if has_rules:
            item['rules']={rule:{mode:_decision_counts([r['decisions'][rule][mode] for r in values])
                for mode in base.INFERENCE} for rule in rel.RULES}
        result['methods'][name]=item
    for family in MOBILE_FAMILIES:
        if family+'_fixed' not in methods or family+'_mobile' not in methods:continue
        pairs=[r['mobility_pairs'][family] for r in rows]
        item=dict(denominator=len(rows),actual_moved=sum(base._finite(r['moves']) and r['moves']>0 for r in pairs),
            same_power_matrix=sum(base._finite(r['power_matrix_max_difference_kw']) and r['power_matrix_max_difference_kw']<=1e-6 for r in pairs),
            same_net_profile=sum(base._finite(r['net_profile_max_difference_kw']) and r['net_profile_max_difference_kw']<=1e-6 for r in pairs),
            driver_minutes=base._stats(r['driver_minutes'] for r in pairs),inference={})
        for mode in base.INFERENCE:
            fixed_name=family+'_fixed';mobile_name=family+'_mobile'
            item['inference'][mode]=_wtl([(r['methods'][mobile_name]['inference'][mode]['M'],
                r['methods'][fixed_name]['inference'][mode]['M']) for r in rows])
            item['inference'][mode]['by_actual_move']={label:_wtl([
                (r['methods'][mobile_name]['inference'][mode]['M'],r['methods'][fixed_name]['inference'][mode]['M'])
                for r in rows if (base._finite(r['mobility_pairs'][family]['moves']) and r['mobility_pairs'][family]['moves']>0)==moved])
                for label,moved in (('moved',True),('not_moved_or_unknown',False))}
        if has_rules:
            item['rules']={}
            for rule in rel.RULES:
                item['rules'][rule]={}
                for mode in base.INFERENCE:
                    changes=[r['decision_changes'][rule][mode] for r in pairs]
                    item['rules'][rule][mode]=dict(denominator=len(rows),
                        transitions=dict(Counter(r['fixed']+' -> '+r['mobile'] for r in changes)),
                        **{key:sum(r[key] for r in changes) for key in ('retained_changed','same_losses_different_retained',
                            'correct_gain','correct_loss','false_reject_reduced','false_reject_increased')})
        result['mobility'][family]=item
    return result


def analyze(phase, *, thresholds=None, selection_metadata=None):
    """Describe already-loaded data; evaluation callers first save/calibrate rules."""
    methods=rel.declared_methods(phase)
    if thresholds is not None:
        base._identity_match(thresholds['identity'],phase['identity'])
        if tuple(thresholds['methods'])!=methods:raise ValueError('threshold and result methods differ')
    if phase['phase']=='evaluation' and thresholds is None:
        raise ValueError('evaluation result analysis requires frozen calibrated thresholds')
    histories=[_history(r,methods,thresholds,selection_metadata) for r in phase['records']]
    cases=defaultdict(list);truths=defaultdict(list);lengths=defaultdict(list)
    for row in histories:cases[row['case_id']].append(row);truths[str(row['true_name'])].append(row);lengths[str(row['H'])].append(row)
    return dict(schema='ev_mobile_v2_results_v1',phase=phase['phase'],identity=phase['identity'],
        methods=list(methods),source_files=phase['files'],has_calibrated_rules=thresholds is not None,
        selection_metadata_source_files=[] if selection_metadata is None else selection_metadata.get('source_files',[]),
        count_unit='case-history slots; independent base seeds reported separately; no IID intervals for paired slots',
        score_scope='M uses raw SSE, never AIC-adjusted SSE; rule decisions use their separately calibrated scores',
        attribution_scope='moving and power changes are paired jointly; same-loss decision changes may come from thresholds',
        old_69_explanation='old square/scalar curves matched 25/72, first candidate and retained sets 72/72; old ties do not imply identical new actions',
        overall=_summarize(histories,methods,thresholds is not None),
        by_case={k:_summarize(v,methods,thresholds is not None) for k,v in sorted(cases.items())},
        by_truth={k:_summarize(v,methods,thresholds is not None) for k,v in sorted(truths.items())},
        by_H={k:_summarize(v,methods,thresholds is not None) for k,v in sorted(lengths.items())},by_history=histories)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--phase',choices=('development','calibration','evaluation'),required=True)
    parser.add_argument('--thresholds',type=Path);parser.add_argument('--raw-directory',type=Path)
    parser.add_argument('--selection-metadata',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.phase=='evaluation' and not args.thresholds:parser.error('evaluation requires previously saved thresholds')
    # Thresholds are deliberately loaded before the evaluation directory.
    thresholds=None if args.thresholds is None else base._read(args.thresholds)
    phase=rel.read_phase(args.directory,expected_phase=args.phase)
    selection=None if args.selection_metadata is None else base._read(args.selection_metadata)
    if args.raw_directory:
        if selection is not None:parser.error('choose raw directory or saved selection metadata')
        selection=extract_selection_metadata(phase,args.raw_directory)
        base._write(args.output.with_name(args.output.stem+'_selection.json'),selection)
    result=analyze(phase,thresholds=thresholds,selection_metadata=selection)
    base._write(args.output,result)


if __name__=='__main__':main()
