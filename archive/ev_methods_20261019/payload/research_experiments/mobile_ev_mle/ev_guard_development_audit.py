"""Read completed old/guard development batches once and save a small audit.

This is descriptive development evidence. Unchanged policy components can still
produce different finite-time MILP plans under different computation loads.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path

from . import ev_mobile_v2_results as results
from . import ev_reliability_v2 as rel
from . import ev_systematic_analyze as base


def load_completed(directories,counts):
    records=[];metadata={};files=[];manifest=None
    for directory,count in zip(map(Path,directories),counts):
        paths=sorted(directory.glob('case_*.json.gz'))
        checkpoint=base._read(directory/'checkpoint.json')
        if len(paths)!=count or checkpoint.get('completed')!=count:
            raise ValueError('wait for complete development batch: '+str(directory))
        freeze=base._read(directory/'freeze.json')
        if manifest is not None and manifest!=freeze:raise ValueError('development parts have different frozen source/config')
        manifest=freeze
        for path in paths:
            raw=base._read(path)
            if raw['phase']!='development':raise ValueError('development-only audit')
            record=base.project_analysis_record(raw);records.append(record)
            projected={}
            for name,row in raw.get('selection',{}).get('methods',{}).items():
                payload=row.get('raw',{})
                projected[name]=dict(metadata_available=True,planning_success=row.get('success'),
                    selected_origin=payload.get('selected_origin'),selected_score=payload.get('selected_score'),
                    initial_best_score=payload.get('initial_best_score'),failure=payload.get('failure'),
                    dependencies=row.get('dependencies',[]),standalone_ancestors=row.get('standalone_ancestors',[]))
                if name.startswith('jointgrad_'):projected[name]['gradient']=results._gradient_metadata(payload)
            metadata[results.history_key(record)]=projected
            files.append(dict(path=str(path.resolve()),sha256=sha256(path.read_bytes()).hexdigest()))
    phase=dict(records=records,phase='development',manifest=manifest,
        identity=base._stable_identity(manifest,records),methods=manifest['config']['methods'],files=files)
    rel.declared_methods(phase)
    return phase,dict(rows=metadata,source_files=files)


def _stats(value):
    return {key:value.get(key) for key in ('n','mean','median','min','max')}


def _small_overall(value):
    methods={}
    for name,row in value['methods'].items():
        mode=row['inference']['joint_total']
        methods[name]=dict(denominator=row['denominator'],planning_failure=row['planning_failure'],
            operation_failure=row['operation_failure'],joint_inference_incomplete=mode['incomplete'],
            raw_unique_correct=mode['raw_unique_correct'],joint_M=_stats(mode['M']),
            planning_seconds=_stats(row['planning_seconds']),joint_total_seconds=_stats(mode['total_seconds']),
            moved_histories=row['actual_moved'],driver_minutes=_stats(row['driver_minutes']),
            selected_origins=row['selection_origins'])
        if row['gradient'] is not None:
            gradient=row['gradient']
            methods[name]['B']=dict(gradient_MILP_calls=gradient['gradient_MILP_calls'],
                gradient_MILP_accepted=gradient['gradient_MILP_accepted'],
                direct_gradient_selected=gradient['direct_gradient_selected'],
                inherits_fixed_gradient=gradient['inherits_fixed_gradient'],
                accepted_step_without_final_gradient=gradient['accepted_step_without_final_gradient'],
                metadata_missing=gradient['metadata_missing'],
                reference_guard_enabled=gradient['reference_guard_enabled'],
                reference_guard_requested=gradient['reference_guard_requested'],
                guard_counts=gradient['guard_counts'],selected_guard_checked=gradient['selected_guard_checked'],
                selected_guard_passed=gradient['selected_guard_passed'],selected_guard_failed=gradient['selected_guard_failed'],
                selected_guard_missing_when_requested=gradient['selected_guard_missing_when_requested'],
                selected_guard_min_raw_slack=_stats(gradient['selected_guard_min_raw_slack']),
                selected_guard_min_effective_slack=_stats(gradient['selected_guard_min_effective_slack']),
                selected_predictive_gain=_stats(gradient['selected_predictive_gain']),
                budget_overrun=gradient['budget_overrun'],
                joint_fit_calls=_stats(gradient['counters']['joint_fit_calls']))
    mobility={}
    for family,row in value['mobility'].items():
        mode=row['inference']['joint_total']
        mobility[family]=dict(denominator=row['denominator'],moved_histories=row['actual_moved'],
            same_power_matrix=row['same_power_matrix'],same_net_profile=row['same_net_profile'],
            driver_minutes=_stats(row['driver_minutes']),joint_M_mobile_minus_fixed={key:mode[key]
                for key in ('win','tie','loss','incomplete','valid_pairs')},delta_M=_stats(mode['delta']))
    return dict(case_history_slots=value['case_history_slots'],unique_base_seeds=value['unique_base_seeds'],
        paired_seed_clusters=value['paired_seed_clusters'],status_counts=value['status_counts'],
        truth_counts=value['truth_counts'],methods=methods,mobility=mobility)


def _failure_details(report):
    rows=[]
    for history in report['by_history']:
        failures={}
        for name,value in history['methods'].items():
            if value['planning_failure'] or value['operation_success'] is False or not value['inference']['joint_total']['complete']:
                selection=value.get('selection') or {};gradient=selection.get('gradient')
                failures[name]=dict(planning_failure=value['planning_failure'],operation_success=value['operation_success'],
                    joint_complete=value['inference']['joint_total']['complete'],M=value['inference']['joint_total']['M'],
                    selection_failure=selection.get('failure'),selected_origin=selection.get('selected_origin'),
                    guard_requested=None if gradient is None else gradient.get('reference_guard_requested'),
                    guard_enabled=None if gradient is None else gradient.get('reference_guard_enabled'),
                    selected_guard=None if gradient is None else gradient.get('selected_reference_guard'))
        if failures:
            rows.append(dict(case_id=history['case_id'],seed=history['seed'],true_name=history['true_name'],methods=failures))
    return rows


def compare(old_phase,new_phase,old_report,new_report):
    old={results.history_key(r):r for r in old_phase['records']};new={results.history_key(r):r for r in new_phase['records']}
    if set(old)!=set(new):raise ValueError('old and guard development histories must match')
    compared={}
    for name in rel.declared_methods(new_phase):
        delta=[];curves=0;rank_same=0;missing=0;planning_changed=0
        for key,row in new.items():
            before=old[key];a=base._methods(before)[name];b=base._methods(row)[name]
            difference=results.curve_difference(a.get('power_kw'),b.get('power_kw'))
            curves+=difference is not None and difference<=1e-6
            mode_a=base._mode(before,name,'joint_total');mode_b=base._mode(row,name,'joint_total')
            delta.append((mode_b['M'],mode_a['M']))
            rank_same+=mode_a['complete'] and mode_b['complete'] and mode_a['minimum_candidates']==mode_b['minimum_candidates']
            missing+=not(mode_a['complete'] and mode_b['complete'])
            planning_changed+=bool(a.get('planning_failure'))!=bool(b.get('planning_failure'))
        compared[name]=dict(denominator=len(new),same_power_matrix=curves,same_raw_first=rank_same,
            incomplete_pair=missing,planning_failure_status_changed=planning_changed,new_minus_old_M=results._wtl(delta))
    changes=[]
    old_hist={r['key']:r for r in old_report['by_history']}
    for row in new_report['by_history']:
        before=old_hist[row['key']]
        a=before['methods']['jointgrad_mobile'];b=row['methods']['jointgrad_mobile']
        guard=(b.get('selection') or {}).get('gradient') or {}
        fixed=b['inference']['joint_total'];oldfixed=before['methods']['jointgrad_fixed']['inference']['joint_total']
        changes.append(dict(case_id=row['case_id'],seed=row['seed'],true_name=row['true_name'],
            old_mobile_M=a['inference']['joint_total']['M'],guard_mobile_M=b['inference']['joint_total']['M'],
            old_mobile_minus_fixed=before['mobility_pairs']['jointgrad']['delta_M']['joint_total'],
            guard_mobile_minus_fixed=row['mobility_pairs']['jointgrad']['delta_M']['joint_total'],
            old_moves=a['moves'],guard_moves=b['moves'],old_planning_failure=a['planning_failure'],
            guard_planning_failure=b['planning_failure'],guard_enabled=guard.get('reference_guard_enabled'),
            guard_requested=guard.get('reference_guard_requested'),selected_guard=guard.get('selected_reference_guard'),
            guard_origin=(b.get('selection') or {}).get('selected_origin')))
    return dict(methods=compared,B_mobile_histories=changes,
        attribution='Guard, computation load and finite-time MILP plans changed together; this is not a guard-only causal comparison.')


def run(root):
    root=Path(root);by_bank={};all_files=[]
    for bank,parts,counts in (('n3',('n3_part0','n3_part1'),(10,10)),('n4',('n4',),(8,))):
        old_dirs=[root/('mobile_v2_dev_'+p) for p in parts]
        new_dirs=[root/('mobile_v2_guard_dev_'+p) for p in parts]
        old,old_metadata=load_completed(old_dirs,counts);new,new_metadata=load_completed(new_dirs,counts)
        old_report=results.analyze(old,selection_metadata=old_metadata)
        new_report=results.analyze(new,selection_metadata=new_metadata)
        by_bank[bank]=dict(original=_small_overall(old_report['overall']),guard=_small_overall(new_report['overall']),
            original_policy_id=old['manifest'].get('policy_id'),guard_policy_id=new['manifest'].get('policy_id'),
            configured_planning_seconds=dict(original=old['manifest']['config'].get('planning_seconds'),
                                              guard=new['manifest']['config'].get('planning_seconds')),
            guard_by_case={k:_small_overall(v) for k,v in new_report['by_case'].items()},
            guard_by_truth={k:_small_overall(v) for k,v in new_report['by_truth'].items()},
            original_failures=_failure_details(old_report),guard_failures=_failure_details(new_report),
            comparison=compare(old,new,old_report,new_report))
        all_files.extend(old['files']+new['files'])
    return dict(schema='ev_guard_development_audit_v1',by_bank=by_bank,source_files=all_files,
        scope='All 28 guard development slots retained; 16 unique base seeds across the two physical banks.',
        no_formal_evaluation_read=True,no_rules_changed=True,
        computational_confounding='Development used concurrent solver/audit loads. Thirty-second soft deadlines and finite-time incumbents can alter even unchanged policy curves. Formal runs will use two compute workers without competing audits.',
        missing_information='No calibrated decision counts are inferred from development raw ranks or M.')


def markdown(report):
    lines=['# Guard开发完整回放','',
        '全部28个开发槽保留，对应16个基础seed；n3为20槽/8 seed，n4为8槽/8 seed。未读取正式评价。',
        '原版与guard比较同时受到计算负载、30秒软预算和MILP有限时限解的影响，不能全部归因于保护约束。','',
        '表中“原始正确”是完整最终拟合SSE唯一第一，不是校准后唯一识别。规划失败即使采用合法回退也照计。','']
    for bank,value in report['by_bank'].items():
        current=value['guard'];lines.extend([f'## {bank}：{current["case_history_slots"]}槽/{current["unique_base_seeds"]} seed','',
            '|方法|规划失败|最终联合拟合未完成|原始正确|平均M|平均规划秒|实际移动|',
            '|---|---:|---:|---:|---:|---:|---:|'])
        for name,row in current['methods'].items():
            mean=row['joint_M']['mean'];seconds=row['planning_seconds']['mean']
            lines.append('|'+ '|'.join(map(str,(name,row['planning_failure'],row['joint_inference_incomplete'],
                row['raw_unique_correct'],'—' if mean is None else f'{mean:.3f}',
                '—' if seconds is None else f'{seconds:.2f}',row['moved_histories'])))+'|')
        mobile=current['methods']['jointgrad_mobile']['B'];pair=current['mobility']['jointgrad']
        lines.extend(['',f'B移动相对B固定点：{pair["joint_M_mobile_minus_fixed"]}；驾驶分钟均值{pair["driver_minutes"]["mean"]}。',
            f'B移动梯度MILP调用{mobile["gradient_MILP_calls"]}、接受{mobile["gradient_MILP_accepted"]}、最终直接梯度计划{mobile["direct_gradient_selected"]}。',
            f'保护请求{mobile["reference_guard_requested"]}、启用{mobile["reference_guard_enabled"]}、最终检查{mobile["selected_guard_checked"]}、通过{mobile["selected_guard_passed"]}、缺失{mobile["selected_guard_missing_when_requested"]}；{mobile["guard_counts"]}。','',
            '原版→guard中未改变算法的基线也应核对曲线：'])
        for name in ('square_fixed','square_mobile','scalar_fixed','scalar_mobile','cap_fixed','cap_mobile'):
            row=value['comparison']['methods'][name]
            lines.append(f'- {name}：曲线相同{row["same_power_matrix"]}/{row["denominator"]}；规划失败状态变化{row["planning_failure_status_changed"]}。')
        lines.append('')
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=run(args.root);base._write(args.output.with_suffix('.json'),report)
    args.output.with_suffix('.md').write_text(markdown(report),encoding='utf-8')


if __name__=='__main__':main()
