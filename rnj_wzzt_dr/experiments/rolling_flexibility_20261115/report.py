"""Compact summaries of independently audited rolling runs; no new simulation."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


DT=.25
STEPS=96


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ref(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(reference):
    path=Path(reference['path']).resolve()
    if ref(path)!=reference: raise ValueError('Audited summary reference changed: '+str(path))
    return path


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def csv_write(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def stats(values):
    values=[float(v) for v in values if v is not None]
    if not values:return dict(n=0,mean=None,minimum=None,maximum=None,total=None)
    a=np.asarray(values)
    if not np.isfinite(a).all():raise ValueError('Nonfinite audited summary value')
    return dict(n=len(a),mean=float(a.mean()),minimum=float(a.min()),maximum=float(a.max()),total=float(a.sum()))


def key(row): return row['id'],row['policy'],int(row['repeat'])


def quantity(row,kind,axis):
    return row.get('quantities',{}).get(kind,{}).get(axis)


def full_sum(values,expected):
    values=list(values)
    return float(sum(values)) if len(values)==expected and all(v is not None for v in values) else None


def final_declaration(snapshot,feedback_state,beta):
    """Postprocessing decision only; this label was not emitted by the runner."""
    if not 0<beta<1:raise ValueError('Require a fixed confidence threshold in (0,1)')
    if len(snapshot['weights'])<2:return 'pool_unresolved'
    if feedback_state=='pause':return 'model_unresolved'
    return 'confirmed' if float(snapshot['target_weight'])>=beta else 'undecided'


def trajectory_summary(entry,policy,repeat,item,rows,beta):
    rows=sorted(rows,key=lambda r:r['step'])
    if len({r['step'] for r in rows})!=len(rows):raise ValueError('Repeated audited quarter-hour row')
    final={} if item is None else item['final']
    evaluation=final.get('evaluation') or {}
    complete=bool(item and item['status']=='complete' and item['audited_steps']==STEPS)
    true_families=(item or {}).get('true_family_ids')
    if true_families is None and rows:true_families=rows[0].get('true_family_ids')
    profile_true=evaluation.get('profile_true_weight')
    if profile_true is None and complete and true_families is not None:
        profile_true=sum(rows[-1]['profile_weights'].get(g,0.) for g in true_families)
    legacy=evaluation.get('p_true')
    declaration=(final_declaration(final['final_snapshot'],rows[-1]['feedback']['state'],beta)
                 if complete else 'unavailable')
    switches=[dict(step=int(r['step']),after_elapsed_minutes=15*(int(r['step'])+1),
        before_native=int(r['before_target']),after_native=int(r['after_target']))
        for r in rows if r['before_target']!=r['after_target']]
    call={}
    for kind in ('service','extra_requested','extra_delivered','total_requested','total_actual'):
        call[kind]={axis:full_sum((quantity(r,kind,axis) for r in rows),len(rows))
                    for axis in ('P_kwh','Q_kvarh')}
    voltage=[r['operational']['all_bus_voltage'] for r in rows]
    quote_stages=[s for r in rows for s in r.get('quote_stages',[])]
    result=dict(id=entry['id'],case=entry['case'],start_date=entry['start_date'],
        history_repeat=entry.get('history_repeat'),pq_relative_sd=entry.get('pq_relative_sd'),policy=policy,repeat=repeat,
        status='missing' if item is None else item['status'],complete=complete,
        expected_steps=STEPS,audited_steps=len(rows),missing_steps=STEPS-len(rows),
        internal_truth_covered=(item or {}).get('internal_truth_covered'),
        full_support_truth_covered=(item or {}).get('full_support_truth_covered'),
        historical_target_correct=evaluation.get('historical_target_correct'),
        final_internal_correct=evaluation.get('native_map_correct') if complete else None,
        final_full_support_exact=evaluation.get('native_full_support_exact') if complete else None,
        final_native=evaluation.get('native_selected_index') if complete else None,
        final_declaration=declaration,declaration_beta=beta,
        declaration_source='New postprocessing label based on the stored final snapshot/feedback and the predeclared beta; not a runtime field',
        selected_family_profile_weight=final.get('final_snapshot',{}).get('target_weight') if complete else None,
        correct_confirmation=bool(declaration=='confirmed' and evaluation.get('native_map_correct') is True),
        wrong_confirmation=bool(declaration=='confirmed' and evaluation.get('native_map_correct') is False),
        profile_true_weight=profile_true if complete else None,
        profile_true_pass_beta=None if profile_true is None or not complete else bool(profile_true>=beta),
        legacy_representative_true_weight=legacy if complete else None,
        legacy_weight_status='Available in audited final evaluation' if legacy is not None else 'Not present in audited final schema; not inferred from profile weight',
        true_family_ids=true_families,
        target_switch_count=len(switches),target_switches=switches,
        nonzero_extra_rounds=final.get('nonzero_extra_rounds'),service_recovery_rounds=final.get('service_recovery_rounds'),
        quantities=call,quantity_scope='DT times sum of absolute per-device changes; service and extra can cancel, so their absolute quantities need not add to total',
        activation_yuan=final.get('activation_yuan'),availability_yuan=final.get('availability_yuan'),
        regulation_total_yuan=final.get('service_yuan'),root_energy_yuan=final.get('root_energy_yuan'),
        fee_scope='Total delivered activation plus nonrefundable daily capacity; firm service is included, not a separately additive fee',
        compute_seconds=final.get('compute_seconds'),shared_preparation_seconds=final.get('shared_preparation_seconds'),
        wall_seconds=final.get('wall_seconds'),quote_status_counts=dict(Counter(r['quote_status'] for r in rows)),
        optimizer_rounds=sum(bool(r.get('optimizer_invoked')) for r in rows),
        quote_exceptions=final.get('quote_exceptions',sum(bool(r.get('quote_exception')) for r in rows)),
        compute_budget_overrun_rounds=sum(r.get('compute_budget_overrun') is True for r in rows),
        budget_field_available_rounds=sum(r.get('compute_budget_overrun') is not None for r in rows),
        pause_rounds=sum(r['feedback']['state']=='pause' for r in rows),
        caution_rounds=sum(r['feedback']['state']=='caution' for r in rows),
        gate_feasible_rounds=sum(r.get('new_gate_feasible') is True for r in rows),
        native_final_status_counts=dict(Counter(str(r['certificate'].get('native_status')) for r in rows if r.get('certificate'))),
        quote_stage_count=len(quote_stages),bound_conflicts=sum(bool((r.get('certificate') or {}).get('bound_conflict')) for r in rows),
        minimum_voltage=min((v['minimum'] for v in voltage),default=None),maximum_voltage=max((v['maximum'] for v in voltage),default=None),
        voltage_violation_rounds=sum(r['operational']['cells_below_095']+r['operational']['cells_above_105']>0 for r in rows),
        maximum_branch_current_A=max((r['operational']['maximum_branch_current_A'] for r in rows),default=None),
        maximum_root_kVA=max((r['operational']['root_S_kVA']['maximum'] for r in rows),default=None),
        failure=final.get('failure'),source=None if item is None else item['source'])
    return result


def summarize(audit,protocol,entries,records):
    policies=protocol['policies'];repeats=int(protocol['repeats']);beta=float(protocol['config']['beta'])
    expected={(e['id'],p,k) for e in entries for p in policies for k in range(repeats)}
    if len(expected)!=audit['expected_trajectories']:raise ValueError('Original planned trajectory denominator differs')
    indexed={key(t):t for t in audit['trajectories']};grouped=defaultdict(list)
    if len(indexed)!=len(audit['trajectories']) or not set(indexed)<=expected:
        raise ValueError('Duplicate or unplanned audited trajectory')
    for row in records:
        if key(row) not in expected or not 0<=int(row['step'])<STEPS:raise ValueError('Unplanned audited step')
        grouped[key(row)].append(row)
    trajectories=[trajectory_summary(e,p,k,indexed.get((e['id'],p,k)),grouped[(e['id'],p,k)],beta)
        for e in entries for k in range(repeats) for p in policies]
    bykey={key(t):t for t in trajectories};comparisons=[]
    for t in trajectories:
        baseline=bykey.get((t['id'],'no_dfs',t['repeat']))
        paired=bool(baseline and baseline['complete'] and t['complete'])
        comparison=dict(id=t['id'],policy=t['policy'],repeat=t['repeat'],paired_complete=paired,
            new_recovery=None,lost_recovery=None,changed_final_native=None,profile_true_change=None,
            regulation_cost_change_yuan=None,root_energy_change_yuan=None)
        if paired:
            comparison.update(new_recovery=bool(t['final_internal_correct'] and not baseline['final_internal_correct']),
                lost_recovery=bool(baseline['final_internal_correct'] and not t['final_internal_correct']),
                changed_final_native=t['final_native']!=baseline['final_native'],
                profile_true_change=None if t['profile_true_weight'] is None or baseline['profile_true_weight'] is None else t['profile_true_weight']-baseline['profile_true_weight'],
                regulation_cost_change_yuan=t['regulation_total_yuan']-baseline['regulation_total_yuan'],
                root_energy_change_yuan=t['root_energy_yuan']-baseline['root_energy_yuan'])
        comparisons.append(comparison)
    groups=[]
    for policy in policies:
        items=[t for t in trajectories if t['policy']==policy];pairs=[p for p in comparisons if p['policy']==policy]
        groups.append(dict(policy=policy,expected=len(items),complete=sum(t['complete'] for t in items),
            failed_or_missing=sum(not t['complete'] for t in items),
            internal_correct=sum(t['final_internal_correct'] is True for t in items),
            full_support_exact=sum(t['final_full_support_exact'] is True for t in items),
            correct_confirmations=sum(t['correct_confirmation'] for t in items),
            wrong_confirmations=sum(t['wrong_confirmation'] for t in items),
            unresolved=sum(t['final_declaration'] in ('undecided','pool_unresolved','model_unresolved') for t in items),
            declaration_counts=dict(Counter(t['final_declaration'] for t in items)),
            paired_complete=sum(p['paired_complete'] for p in pairs),
            new_recoveries=sum(p['new_recovery'] is True for p in pairs),lost_recoveries=sum(p['lost_recovery'] is True for p in pairs),
            changed_final_native=sum(p['changed_final_native'] is True for p in pairs),
            profile_true_weight=stats(t['profile_true_weight'] for t in items),
            profile_true_change=stats(p['profile_true_change'] for p in pairs),
            legacy_representative_true_weight=stats(t['legacy_representative_true_weight'] for t in items),
            nonzero_extra_rounds=sum(t['nonzero_extra_rounds'] or 0 for t in items),
            service_recovery_rounds=sum(t['service_recovery_rounds'] or 0 for t in items),
            missing_call_count_trajectories=sum(t['nonzero_extra_rounds'] is None for t in items),
            target_switches=sum(t['target_switch_count'] for t in items),
            activation_yuan=stats(t['activation_yuan'] for t in items),availability_yuan=stats(t['availability_yuan'] for t in items),
            regulation_total_yuan=stats(t['regulation_total_yuan'] for t in items),root_energy_yuan=stats(t['root_energy_yuan'] for t in items),
            root_energy_change_yuan=stats(p['root_energy_change_yuan'] for p in pairs),
            compute_seconds=stats(t['compute_seconds'] for t in items),
            quote_status_counts=dict(sum((Counter(t['quote_status_counts']) for t in items),Counter())),
            quote_exceptions=sum(t['quote_exceptions'] for t in items),bound_conflicts=sum(t['bound_conflicts'] for t in items),
            native_final_status_counts=dict(sum((Counter(t['native_final_status_counts']) for t in items),Counter())),
            compute_budget_overrun_rounds=sum(t['compute_budget_overrun_rounds'] for t in items),
            quantities={kind:{axis:stats(t['quantities'][kind][axis] for t in items) for axis in ('P_kwh','Q_kvarh')}
                        for kind in ('service','extra_delivered','total_actual')}))
    no_nonzero=all(t['nonzero_extra_rounds']==0 for t in trajectories)
    has_nonzero=any((t['nonzero_extra_rounds'] or 0)>0 for t in trajectories)
    no_service=all(t['service_recovery_rounds']==0 for t in trajectories)
    return dict(trajectories=trajectories,groups=groups,paired_comparisons=comparisons,
        expected_trajectories=len(expected),completed_trajectories=sum(t['complete'] for t in trajectories),
        expected_rounds=len(expected)*STEPS,audited_rounds=len(records),missing_rounds=len(expected)*STEPS-len(records),
        histories=len(entries),physical_network_days=len({(e['case'],e['start_date']) for e in entries}),
        independent_new_physical_days=0,repeats=repeats,dt_hours=DT,
        no_nonzero_identification_dispatch=no_nonzero,has_nonzero_identification_dispatch=has_nonzero,
        no_firm_service_adjustment=no_service,
        interpretation=('All executed extra identification actions were zero; this batch provides no realized nonzero procurement benefit evidence.' if no_nonzero else
                        'Nonzero executed actions exist; paired final recovery and losses, service repair and complete costs must be read together.' if has_nonzero else
                        'No nonzero dispatch is confirmed in available records; missing trajectories prevent an all-zero conclusion.'),
        uncertainty_scope='Known development histories and previously used physical days; quarter hours and repeated paths are not independent history inputs',
        probability_scope='Per-replan conditional nominal bounds only; no validated whole-day adaptive 90% coverage',
        weight_scope='Profile family minimum and legacy representative-normalized weights are separate quantities; absent legacy values are not imputed',
        bound_scope='Per-quote remaining-horizon activation plus additional capacity; bounds cannot be summed into a closed-loop day-optimality certificate')


def fmt(v):
    if v is None:return '—'
    if isinstance(v,bool):return '是' if v else '否'
    if isinstance(v,float):return f'{v:.6g}'
    return str(v)


def markdown_table(columns,rows):
    return '\n'.join(['|'+'|'.join(columns)+'|','|'+'|'.join(['---']*len(columns))+'|']+
        ['|'+'|'.join(fmt(v).replace('|','/') for v in row)+'|' for row in rows])


def figure(records,trajectories,out):
    if not records or not any(r.get('true_family_ids') is not None and 'quantities' in r for r in records):
        return dict(status='unavailable',reason='Audited schema lacks per-step true-family identities or action quantities')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    identifiers=list(dict.fromkeys(t['id'] for t in trajectories));policies=list(dict.fromkeys(t['policy'] for t in trajectories))
    fig,axes=plt.subplots(len(identifiers),3,figsize=(14,3.4*len(identifiers)),squeeze=False,constrained_layout=True)
    colors={p:plt.get_cmap('tab10')(k) for k,p in enumerate(policies)}
    for i,identifier in enumerate(identifiers):
        for policy in policies:
            paths=sorted({r['repeat'] for r in records if r['id']==identifier and r['policy']==policy})
            for repeat in paths:
                rows=sorted((r for r in records if r['id']==identifier and r['policy']==policy and r['repeat']==repeat),key=lambda r:r['step'])
                x=[r['step']+1 for r in rows]
                y=[sum(r['profile_weights'].get(g,0.) for g in r['true_family_ids']) if r.get('true_family_ids') is not None else np.nan for r in rows]
                label=policy if len(paths)==1 else policy+f' / path {repeat}'
                axes[i,0].plot(x,y,color=colors[policy],label=label,alpha=.85)
                for j,axis in enumerate(('P_kwh','Q_kvarh'),1):
                    y=[np.nan if quantity(r,'extra_delivered',axis) is None else quantity(r,'extra_delivered',axis)/DT for r in rows]
                    axes[i,j].plot(x,y,color=colors[policy],label=label,alpha=.85)
        axes[i,0].set_title(identifier,fontsize=9);axes[i,0].set_ylim(-.02,1.02)
        axes[i,0].set_ylabel('True-family profile weight')
        axes[i,1].set_ylabel('Extra delivered |P| (kW)');axes[i,2].set_ylabel('Extra delivered |Q| (kvar)')
        for ax in axes[i]:ax.set_xlabel('Completed 15-minute steps');ax.set_xlim(0,96);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7,loc='best')
    fig.suptitle('Known-development replay: profile evidence and extra identification dispatch',fontsize=12)
    path=out/'rolling_profiles_and_calls.png';fig.savefig(path,dpi=160);plt.close(fig)
    return dict(status='complete',image=ref(path),scope='Each line is one audited conditional trajectory; service-repair quantities are reported separately in tables')


def run(audit_path,out):
    audit_path,out=Path(audit_path).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve earlier report outputs')
    audit=read(audit_path)
    if audit.get('status')!='passed' or audit.get('errors'):
        raise ValueError('An independently passed audit is required; scientific failures remain in its trajectories')
    protocol=read(checked(audit['source_protocol']));records=read(checked(audit['records']))
    entries=read(checked(protocol['inputs']))
    for trajectory in audit['trajectories']:checked(trajectory['source'])
    result=summarize(audit,protocol,entries,records)
    if result['audited_rounds']!=audit['audited_rounds'] or result['missing_rounds']!=audit['missing_rounds']:
        raise ValueError('Report and independent audit denominators differ')
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'report_source.py')
    result.update(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),source_audit=ref(audit_path),
        source_protocol=audit['source_protocol'],source_records=audit['records'],source_inputs=protocol['inputs'],
        report_source=ref(__file__),audit_errors=audit.get('errors',[]),original_failures=audit.get('original_failures',[]),
        configuration=protocol['config'])
    compact=[]
    for t in result['trajectories']:
        row={k:v for k,v in t.items() if not isinstance(v,(list,dict))}
        row['target_switch_steps']=[s['step'] for s in t['target_switches']]
        for kind,data in t['quantities'].items():
            for axis,value in data.items():row[kind+'_'+axis]=value
        compact.append(row)
    csv_write(out/'trajectories.csv',compact);csv_write(out/'paired_comparisons.csv',result['paired_comparisons'])
    csv_write(out/'target_switches.csv',[dict(id=t['id'],policy=t['policy'],repeat=t['repeat'],**s) for t in result['trajectories'] for s in t['target_switches']])
    rounds=[];quotes=[]
    for r in records:
        row=dict(id=r['id'],policy=r['policy'],repeat=r['repeat'],step=r['step'],completed_step=r['step']+1,
            elapsed_minutes=15*(r['step']+1),before_target=r['before_target'],after_target=r['after_target'],
            internal_correct=r['internal_correct'],full_support_exact=r['full_support_exact'],
            profile_true_weight=None if r.get('true_family_ids') is None else sum(r['profile_weights'].get(g,0.) for g in r['true_family_ids']),
            feedback=r['feedback']['state'],quote_status=r['quote_status'],gate_feasible=r.get('new_gate_feasible'),
            activation_yuan=r['fees']['activation_yuan'],availability_increment_yuan=r['fees']['availability_increment_yuan'],
            root_energy_yuan=r['fees']['root_energy_yuan'])
        for kind in ('service','extra_requested','extra_delivered','total_requested','total_actual'):
            for axis in ('P_kwh','Q_kvarh'):row[kind+'_'+axis]=quantity(r,kind,axis)
        rounds.append(row)
        quotes.append(dict(id=r['id'],policy=r['policy'],repeat=r['repeat'],step=r['step'],status=r['quote_status'],
            optimizer_invoked=r.get('optimizer_invoked'),gate_feasible=r.get('new_gate_feasible'),
            quote_seconds=r.get('quote_seconds'),round_planning_seconds=r.get('round_planning_seconds'),
            round_budget_seconds=r.get('round_budget_seconds'),round_wall_seconds=r.get('round_wall_seconds'),
            cumulative_compute_seconds=r.get('cumulative_compute_seconds'),compute_budget_overrun=r.get('compute_budget_overrun'),
            exception=r.get('quote_exception'),certificate=r.get('certificate'),stages=r.get('quote_stages'),
            probability_box_checks=r.get('probability_box_checks')))
    csv_write(out/'rounds.csv',rounds);csv_write(out/'quotes.csv',quotes)
    write(out/'quotes.json',quotes)
    try:result['figure']=figure(records,result['trajectories'],out)
    except Exception as exc:result['figure']=dict(status='failed',error=repr(exc))
    text=['# 15分钟滚动灵活性：独立审计汇总','',
        f"本批为已知开发背景重放：{result['histories']}组历史、{result['physical_network_days']}个已使用物理网络日；不是新日期独立确认。每轮15分钟，完整轨迹96轮。",
        f"原定{result['expected_trajectories']}条策略轨迹、{result['expected_rounds']}轮；完整轨迹{result['completed_trajectories']}，审计{result['audited_rounds']}轮，缺失{result['missing_rounds']}轮。失败和缺失仍留在原定分母。",'',
        '所有策略均无实际额外辨识调用，本批没有实现非零采购收益；目标和权重变化来自持续观测。' if result['no_nonzero_identification_dispatch'] else
        '本批含实际辨识调用；下表同时报告相对零采购新增、丢失以及全部费用，不以平均权重代替完整拓扑恢复。' if result['has_nonzero_identification_dispatch'] else
        '已保存记录尚未确认非零辨识调用；存在缺失轨迹，不能据此断言所有策略均零调用。','',
        markdown_table(['策略','SSE第一名内部正确/原定轨迹','正确确认/错误确认/未决','完整支持正确','新增/丢失（完整配对）','额外调用轮数','履约补能轮数','调用/预留/合计 元','根节点购售电账本 元'],
            [[g['policy'],f"{g['internal_correct']}/{g['expected']}",f"{g['correct_confirmations']}/{g['wrong_confirmations']}/{g['unresolved']}",g['full_support_exact'],f"{g['new_recoveries']}/{g['lost_recoveries']}（{g['paired_complete']}）",g['nonzero_extra_rounds'],g['service_recovery_rounds'],
              '/'.join(fmt(g[k]['total']) for k in ('activation_yuan','availability_yuan','regulation_total_yuan')),fmt(g['root_energy_yuan']['total'])] for g in result['groups']]),'',
        f"确认/未决是本报告根据已存输出新增的判定层，未回改运行结果。沿用事前β={protocol['config']['beta']:g}：候选族不足2个为pool_unresolved；末轮反馈pause为model_unresolved；其余只有选中族profile权重达到β才confirmed，否则undecided。第一名错误而权重不足β，应记为第一名错误但未确认，不计高信心误识别。",'',
        '费用为已发生前缀累计；缺失数据不补零。预留费按已承诺尾计划的日峰值逐次补差且不退还，调用费按实际总交付计算。履约服务和额外辨识动作可能互相抵消，二者绝对调用量不应直接相加为总调用量。根节点账本是总购售电示例费用，与辨识采购费用分列。','',
        markdown_table(['输入','策略','路径','第一名内部正确','最终判定','选中族权重','完整支持','真族profile权重','旧代表库真权重','选中参数点变化轮号（0起）'],
            [[t['id'],t['policy'],t['repeat'],t['final_internal_correct'],t['final_declaration'],t['selected_family_profile_weight'],t['final_full_support_exact'],t['profile_true_weight'],t['legacy_representative_true_weight'],','.join(str(s['step']) for s in t['target_switches']) or '无'] for t in result['trajectories']]),'',
        'profile权重先累计每个原生点SSE、再对同族取最小值；旧代表库归一化权重另列，缺字段标“—”，不相互代替。参数点变化发生在所列第k轮观测到达后，即已完成(k+1)个15分钟点；不是第k小时。原始前后原生点见target_switches.csv；同族可能有多个参数点，因此该计数不直接当作拓扑族切换数。','',
        markdown_table(['策略','末次报价状态计数','原生末次状态','报价异常','界冲突','计算预算超出轮数','累计计算秒'],
            [[g['policy'],json.dumps(g['quote_status_counts'],ensure_ascii=False),json.dumps(g['native_final_status_counts']),g['quote_exceptions'],g['bound_conflicts'],g['compute_budget_overrun_rounds'],g['compute_seconds']['total']] for g in result['groups']]),'',
        '每轮首个可行方案与进度约束重求解的各阶段保存在quotes.json/CSV；若旧审计未提供阶段或预算字段则留空。报价界对应当轮剩余时域的调用费加新增预留费，不能相加成全天闭环全局最优界。原生不可行、未决、服务回退和暂停均按原记录保留。','',
        '每轮概率门槛只是声明模型下的条件名义界，本批没有核验全天自适应90%覆盖。实际设备服务采用明确的本地精确履约假设；它与额外辨识请求的设备响应倍率分开。电压、电流和根节点容量见审计；缺少热额定时不宣称线路热容量合格。','',
        '公开汇总没有把15分钟轮数或重复路径当作独立历史输入。详细逐轮调用、profile、费用和失败见CSV与JSON。']
    if result['figure']['status']=='complete':text+=['','![累计profile权重和额外实际调用](rolling_profiles_and_calls.png)']
    if result['original_failures']:text+=['','原始失败：',json.dumps(result['original_failures'],ensure_ascii=False,indent=2)]
    (out/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    write(out/'report.json',result)
    write(out/'complete.json',dict(report=ref(out/'report.json'),markdown=ref(out/'REPORT.md'),figure=result['figure']))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=run(args.audit,args.out)
    print({k:result[k] for k in ('status','expected_trajectories','completed_trajectories','audited_rounds','missing_rounds','no_nonzero_identification_dispatch')})


if __name__=='__main__':main()
