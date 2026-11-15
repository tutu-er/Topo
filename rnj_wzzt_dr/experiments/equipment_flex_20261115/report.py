"""Paired equipment-resource readout from a passed saved-array audit only.

This module never fits, plans or simulates. Original free-Q/band runs and zero
observations are reused references, not additional independent trials.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ... import io
from ..frozen_rx_guard_20261115 import report as shared
from ..single_target_20261115 import report as single


class Postprocess(shared.Postprocess):
    """Keep each row's actual resource map, including changed device columns."""
    def context(self, row):
        c = super().context(row)
        folder = Path(row['source']).parents[1]
        config = self.reader.json(folder/'resource_case.json')
        units = config['action_units']
        mapping = np.zeros((2*c['n'], len(config['resources'])))
        for j, (resource, unit) in enumerate(zip(config['resources'], units, strict=True)):
            node = c['labels'].index(int(resource['node_label']))
            if unit not in ('kW', 'kvar'):
                raise ValueError('Unknown action column unit')
            mapping[node+(c['n'] if unit == 'kvar' else 0), j] = 1/c['entry']['base_kw']
        c['map'] = mapping
        return c


def renamed(policy, origin):
    if origin == 'zero':
        if policy != 'no_dfs':
            raise ValueError('Unexpected zero-reference policy')
        return policy
    return origin+'__'+policy.removeprefix('single_')


def number(value, digits=4):
    return '—' if value is None else f'{value:.{digits}g}'


def device_metrics(states, kind):
    """Quantities are increments; operating energy/state is separately named."""
    p=np.array([s['p_kw'] for s in states],float);q=np.array([s['q_kvar'] for s in states],float)
    p0=np.array([s['baseline_p_kw'] for s in states],float);q0=np.array([s['baseline_q_kvar'] for s in states],float)
    dp,dq=p-p0,q-q0
    answer=dict(paths=len(states),active_abs_kwh_mean=float(np.mean(abs(dp).sum(1)/4)),
        active_increase_kwh_mean=float(np.mean(np.maximum(dp,0).sum(1)/4)),
        active_decrease_kwh_mean=float(np.mean(np.maximum(-dp,0).sum(1)/4)),
        reactive_abs_kvarh_mean=float(np.mean(abs(dq).sum(1)/4)),
        reactive_increase_kvarh_mean=float(np.mean(np.maximum(dq,0).sum(1)/4)),
        reactive_decrease_kvarh_mean=float(np.mean(np.maximum(-dq,0).sum(1)/4)),
        active_peak_kw_max=float(abs(dp).max(initial=0)),reactive_peak_kvar_max=float(abs(dq).max(initial=0)),
        physical_consumption_kwh_mean=float(np.mean(np.maximum(p,0).sum(1)/4)),
        physical_injection_kwh_mean=float(np.mean(np.maximum(-p,0).sum(1)/4)),
        baseline_consumption_kwh_mean=float(np.mean(np.maximum(p0,0).sum(1)/4)),
        baseline_injection_kwh_mean=float(np.mean(np.maximum(-p0,0).sum(1)/4)))
    if kind=='battery':
        soc=np.array([s['soc'] for s in states],float)
        answer.update(charge_grid_kwh_mean=answer['physical_consumption_kwh_mean'],
            discharge_grid_kwh_mean=answer['physical_injection_kwh_mean'],soc_minimum=float(soc.min()),soc_maximum=float(soc.max()),
            terminal_soc_error_abs_max=max(abs(s['terminal_error']) for s in states))
    elif kind=='ev':
        answer.update(departure_energy_kwh_mean=float(np.mean([s['delivered_battery_energy_kwh'] for s in states])),
            required_departure_energy_kwh=states[0]['required_battery_energy_kwh'],
            departure_energy_error_abs_kwh_max=max(abs(s['delivered_battery_energy_kwh']-s['required_battery_energy_kwh']) for s in states))
    elif kind=='thermal':
        temperature=np.array([s['temperature_C'] for s in states],float)
        answer.update(temperature_minimum_C=float(temperature.min()),temperature_maximum_C=float(temperature.max()),
            terminal_temperature_error_abs_C_max=max(abs(s['terminal_error']) for s in states))
    if kind in ('pv','battery'):
        answer['apparent_power_max_kva']=max(max(s['apparent_power_kva']) for s in states)
        answer['maximum_apparent_power_fraction']=max(max(s['apparent_power_kva'])/s['inverter_rating_kva'] for s in states)
    return answer


def device_readout(reader,entries,root,rows,modes):
    catalog=[];devices=[];classes=[]
    for entry in entries:
        tag=entry['id'];seal=reader.json(root/'cases'/tag/'seal.json');spec=reader.json(seal['equipment_spec'])
        for d in spec['devices']:
            catalog.append(dict(input_id=tag,device_id=d['id'],kind=d['kind'],node_label=d['node_label'],
                feeder_id=d['feeder_id'],registered_source_kw=d['registered_kw'],registration_date=d['registration_date'],
                source_to_network_scale=d['source_to_network_scale'],scaled_registered_kw=d['scaled_registered_kw'],
                participation_fraction=d['participation_fraction'],power_kw=d['power_kw'],inverter_kva=d.get('inverter_kva'),
                reactive_cap_kvar=d.get('reactive_cap_kvar',0.),storage_energy_kwh=d.get('energy_kwh'),
                water_volume_litres=d.get('water_litres'),heat_capacity_kwh_per_C=d.get('heat_capacity_kwh_per_C'),
                hourly_temperature_retention=d.get('hourly_temperature_retention'),
                connected_clock_hours=d.get('connected_clock_hours'),charging_clock_hours=d.get('charging_clock_hours'),
                discharging_clock_hours=d.get('discharging_clock_hours'),reactive_clock_windows=d.get('reactive_clock_windows'),
                P_fee_positive_yuan_per_kwh=d['fee_positive_yuan_per_kwh'],P_fee_negative_yuan_per_kwh=d['fee_negative_yuan_per_kwh'],
                Q_fee_yuan_per_kvarh=spec['reactive_fee_yuan_per_kvarh'] if d['kind'] in ('pv','battery') else None,
                source_equipment_spec=seal['equipment_spec'],count_scope=d['count_interpretation']))
        for mode in modes:
            matching=[r for r in rows if r['id']==tag and r['strategy']=='single_'+mode]
            good=[r for r in matching if r['audit_status']=='passed']
            for d in spec['devices']:
                value=dict(input_id=tag,mode=mode,device_id=d['id'],kind=d['kind'],node_label=d['node_label'],
                    expected_paths=len(matching),completed_paths=len(good),power_kw=d['power_kw'],
                    inverter_kva=d.get('inverter_kva'),reactive_cap_kvar=d.get('reactive_cap_kvar',0.))
                for phase in ('requested','delivered'):
                    value[phase]=device_metrics([r[phase+'_devices']['states'][d['id']] for r in good],d['kind']) if good else None
                value['has_nonzero_call']=bool(good and (value['requested']['active_peak_kw_max']>1e-8 or value['requested']['reactive_peak_kvar_max']>1e-8))
                devices.append(value)
            for kind in ('pv','ev','thermal','battery'):
                ids=[d['id'] for d in spec['devices'] if d['kind']==kind]
                selected=[r for r in devices if r['input_id']==tag and r['mode']==mode and r['kind']==kind]
                value=dict(input_id=tag,mode=mode,kind=kind,aggregate_contracts=len(ids),expected_paths=len(matching),completed_paths=len(good))
                for phase in ('requested','delivered'):
                    if not good:value[phase]=None;continue
                    measures={key:sum(r[phase][key] for r in selected) for key in
                        ('active_abs_kwh_mean','active_increase_kwh_mean','active_decrease_kwh_mean',
                         'reactive_abs_kvarh_mean','reactive_increase_kvarh_mean','reactive_decrease_kvarh_mean')}
                    # Coincident absolute aggregate peak; do not sum each device's noncoincident maximum.
                    for axis,unit in (('p','kw'),('q','kvar')):
                        samples=np.array([[np.asarray(r[phase+'_devices']['states'][d][axis+'_'+unit])-
                            r[phase+'_devices']['states'][d]['baseline_'+axis+'_'+unit] for d in ids] for r in good])
                        measures[('active' if axis=='p' else 'reactive')+'_coincident_abs_peak_'+unit+'_max']=float(abs(samples).sum(1).max(initial=0))
                    value[phase]=measures
                classes.append(value)
    return dict(catalog=catalog,by_device=devices,by_class=classes,
        sign_convention='P/Q positive is consumption; increments are relative to declared device baseline. Separate physical energy includes that baseline.',
        peak_convention='Device maximum over time and paths; class peak is simultaneous sum of absolute device increments. Neither is inferred installed hardware count.')


def write_csv(path,rows):
    import csv,json
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)


def self_test():
    zeros=np.zeros(96);load=np.ones(96)*2
    state=dict(p_kw=load,q_kvar=zeros,baseline_p_kw=load,baseline_q_kvar=zeros,
        delivered_battery_energy_kwh=45.6,required_battery_energy_kwh=45.6)
    value=device_metrics([state],'ev')
    assert value['active_abs_kwh_mean']==0 and value['physical_consumption_kwh_mean']==48
    assert value['departure_energy_error_abs_kwh_max']==0
    action=zeros.copy();action[:4]=2;action[4:8]=-2
    battery=dict(p_kw=action,q_kvar=zeros,baseline_p_kw=zeros,baseline_q_kvar=zeros,
        soc=[.5,.6,.5],terminal_error=0.,apparent_power_kva=abs(action),inverter_rating_kva=2.)
    value=device_metrics([battery],'battery')
    assert value['charge_grid_kwh_mean']==value['discharge_grid_kwh_mean']==2.
    assert value['active_abs_kwh_mean']==4. and value['soc_minimum']==.5 and value['soc_maximum']==.6
    assert value['maximum_apparent_power_fraction']==1.
    thermal=dict(p_kw=load,q_kvar=zeros,baseline_p_kw=load,baseline_q_kvar=zeros,
        temperature_C=[47.5,48.,47.5],terminal_error=0.)
    value=device_metrics([thermal],'thermal')
    assert value['temperature_maximum_C']==48. and value['terminal_temperature_error_abs_C_max']==0.
    return dict(status='passed',checks=['baseline_energy_not_procurement','EV_departure_energy','quarter_hour_battery_energy',
        'increment_and_SOC_ranges','inverter_loading','thermal_terminal_state'])


def markdown(data):
    lines = ['# 四类设备资源的配对试验', '',
        ('本文件是每设置1条路径、每报价2秒的接线测试，结果不代表正式16条路径试验。' if data['smoke'] else '本文件为预先设定的设备配置配对开发试验。'), '',
        f"新执行 {data['new_rows']} 条；引用旧带方案 {data['band_baseline_rows']} 条、共同零采购 {data['zero_rows']} 条。",
        '三种设置对应两个已研究网络日；使用原配对误差路径。设备配置改变，冻结候选、R/X与最终评分不变。', '',
        '|输入|方案|内部结构正确|完整响应支持正确|错误权重≥.9|平均真族权重|调用费/元|预留费/元|总费/元|实际P/kWh|实际Q/kvarh|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for case in data['cases']:
        for name, value in case['strategies'].items():
            lines.append('|'+ '|'.join([case['id'], name, f"{value['recovered']}/{value['n']}",
                f"{value['full_support_recovered']}/{value['n']}", str(value['wrong_high']),
                number(value['p_true']['mean']), number(value['settled_activation_yuan']['mean']),
                number(value['settled_availability_yuan']['mean']), number(value['settled_total_yuan']['mean']),
                number(value['quantities']['delivered_P_kwh']['mean']),
                number(value['quantities']['delivered_Q_kvarh']['mean'])])+'|')
    lines += ['', '|输入|新设备方案|对零采购新增/丢失|对旧带方案新增/丢失|', '|---|---|---:|---:|']
    for case in data['cases']:
        for mode in data['modes']:
            name='equipment__'+mode
            zero=case['comparisons']['no_dfs'][name]
            band=case['comparisons']['old_band'][name]
            lines.append(f"|{case['id']}|{mode}|{zero['new_recoveries']}/{zero['lost_recoveries']}|{band['new_recoveries']}/{band['lost_recoveries']}|")
    lines += ['', '## 预测与实际证据', '',
        '固定历史赢家及其历史最强异族，区分预测输入的K均值、实际输入的K均值、干净AC和量测评分。全库比较另保存在JSON；真值仅用于选后评价。', '',
        '|输入|新方案|预测ΔL|实际输入K ΔL|干净AC ΔL|量测ΔL|', '|---|---|---:|---:|---:|---:|']
    for case in data['cases']:
        for mode in data['modes']:
            value=case['strategies']['equipment__'+mode].get('historical_fixed_pair')
            inc=(value or {}).get('increments', {})
            lines.append('|'+ '|'.join([case['id'], mode]+[number(inc.get(k,{}).get('mean')) for k in
                ('forecast_K_logodds','actual_input_K_logodds','clean_logodds','observed_logodds')])+'|')
    lines += ['', '## 设备与电气结果', '',
        '|输入|方案|原设备约束违规路径|交付带超出路径|全母线电压pu|越限路径|最大支路A|最大根kVA|平均网损变化kWh|',
        '|---|---|---:|---:|---|---:|---:|---:|---:|']
    for row in data['operational_summary']:
        lines.append('|'+ '|'.join([row['id'],row['policy'],str(row['device_failed_rows']),
            str(row['delivered_band_failed_rows']),number(row['voltage_minimum'])+'/'+number(row['voltage_maximum']),
            str(row['voltage_failed_rows']),number(row['maximum_branch_current_A']),
            number(row['maximum_root_S_kVA']),number(row['active_loss_change_kwh']['mean'])])+'|')
    lines += ['', '### 分设备类型的增量调用', '',
        '以下均相对于已封存设备基线；原有充电、热水用电和PV发电不计作本次新增采购。能量为路径平均，峰值为同时刻各设备绝对增量之和的最大值。', '',
        '|输入|模式|设备类型|请求P/kWh|交付P/kWh|请求Q/kvarh|交付Q/kvarh|请求P峰/kW|交付P峰/kW|请求Q峰/kvar|交付Q峰/kvar|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in data['device_readout']['by_class']:
        a,b=row['requested'] or {},row['delivered'] or {}
        lines.append('|'+ '|'.join([row['input_id'],row['mode'],row['kind']]+[number(v) for v in
            (a.get('active_abs_kwh_mean'),b.get('active_abs_kwh_mean'),a.get('reactive_abs_kvarh_mean'),b.get('reactive_abs_kvarh_mean'),
             a.get('active_coincident_abs_peak_kw_max'),b.get('active_coincident_abs_peak_kw_max'),a.get('reactive_coincident_abs_peak_kvar_max'),b.get('reactive_coincident_abs_peak_kvar_max'))])+'|')
    lines += ['', '实际发生增量采购的设备：', '',
        '|输入|模式|设备/节点|请求P/kWh|交付P/kWh|请求Q/kvarh|交付Q/kvarh|请求Q峰/kvar|交付Q峰/kvar|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in data['device_readout']['by_device']:
        if not row['has_nonzero_call']:continue
        a,b=row['requested'],row['delivered']
        lines.append('|'+ '|'.join([row['input_id'],row['mode'],row['device_id']]+[number(v) for v in
            (a['active_abs_kwh_mean'],b['active_abs_kwh_mean'],a['reactive_abs_kvarh_mean'],b['reactive_abs_kvarh_mean'],a['reactive_peak_kvar_max'],b['reactive_peak_kvar_max'])])+'|')
    lines += ['', '### 实际设备运行状态', '',
        '储能充放电量是设备总运行量；EV离站能量、热水温度含原基线需求。CSV同时保留请求状态与实际状态、每项分母及全部零调用设备。', '',
        '|输入|模式|设备|充电/放电kWh|SOC或温度范围|期末状态误差|EV离站能量/所需kWh|EV误差kWh|',
        '|---|---|---|---|---|---:|---|---:|']
    for row in data['device_readout']['by_device']:
        if row['kind']=='pv':continue
        b=row['delivered'] or {};kind=row['kind']
        charge='/'.join(number(b.get(k)) for k in ('charge_grid_kwh_mean','discharge_grid_kwh_mean')) if kind=='battery' else '—'
        interval=('/'.join(number(b.get(k)) for k in ('soc_minimum','soc_maximum')) if kind=='battery' else
            '/'.join(number(b.get(k)) for k in ('temperature_minimum_C','temperature_maximum_C'))+'℃' if kind=='thermal' else '—')
        terminal=b.get('terminal_soc_error_abs_max') if kind=='battery' else b.get('terminal_temperature_error_abs_C_max')
        ev='/'.join(number(b.get(k)) for k in ('departure_energy_kwh_mean','required_departure_energy_kwh')) if kind=='ev' else '—'
        lines.append(f"|{row['input_id']}|{row['mode']}|{row['device_id']}|{charge}|{interval}|{number(terminal)}|{ev}|{number(b.get('departure_energy_error_abs_kwh_max'))}|")
    lines += ['', '设备约束包括设备总P/Q、固定充放电时窗、SOC或热状态及额定视在容量；P/Q带是另加的规划限制。电压为采购后AC核验，未进入采购目标；线路和变压器热额定资料不足。', '',
        '## 模型门槛与实际结果', '',
        '联合概率下界是声明的K均值及高斯误差模型内的保守界。逐对均分风险验收比“失败概率界之和≤.1”更强，两者分别记录。', '',
        '|输入|模式|零动作normal联合下界|零动作normal逐对通过|本模式零点联合下界|新门槛通过|外盒负上界|实际内部恢复|',
        '|---|---|---:|---|---:|---|---:|---|']
    cases={c['id']:c for c in data['cases']}
    for q in data['quote_records']:
        z=q.get('zero_direct_gate') or {};normal=q.get('zero_normal_direct_gate') or {}
        negative=[r['upper'] for r in q.get('independent_resource_box_checks',[]) if r['impossible']]
        s=cases[q['id']]['strategies']['equipment__'+q['base_mode']]
        lines.append('|'+ '|'.join([q['id'],q['base_mode'],number(normal.get('worst_recovery_lower')),
            str(normal.get('feasible')),number(z.get('worst_recovery_lower')),str(q['new_gate_feasible']),
            number(min(negative) if negative else None),f"{s['recovered']}/{s['n']}"] )+'|')
    lines += ['', '负外盒上界只证明固定λ和逐对风险份额这一充分二次约束无解；它放松了设备能量、SOC、逆变器圆和P/Q带，故证书适用于原设备域。零回退继续计入实际恢复分母。', '',
        '## 求解与费用证据', '',
        '目标仍为调用费，报告的总费还包括按请求峰值支付的容量费。上下界只针对本轮设备域及数值编码；旧独立Q域的下界不继承。', '',
        '|输入|模式|状态|新门槛通过|零回退|原生状态|费用上界|有效下界|相对界差|可行初值|报价秒数|',
        '|---|---|---|---|---|---|---:|---:|---:|---|---:|']
    for quote in data['quote_records']:
        c=quote.get('certificate') or {};initial=quote.get('initial') or {}
        lines.append('|'+ '|'.join([quote['id'],quote['policy'],str(quote.get('status')),str(quote['new_gate_feasible']),
            str(quote['zero_fallback']),str(quote.get('native_status')),
            number(c.get('feasible_upper_bound_yuan')),number(c.get('lower_bound_yuan')),
            number(c.get('relative_gap')),str(initial.get('name') if initial.get('feasible') else None),
            number(quote.get('quote_wall_seconds',quote.get('seconds')))])+'|')
    lines += ['', '求解器状态和相对费用界差按原记录保留；独立审计不把小界差改写为全局最优。原概率式与设备约束验收、带数值buffer的严格非负性分别核对。', '',
        '|输入|模式|最小概率buffer余量|按记录容差通过编码域|独立严格全局最优证书|历史费用界冲突|',
        '|---|---|---:|---|---|---|']
    for q in data['quote_records']:
        c=q['certificate'];history=c.get('native_bound_history') or {}
        lines.append('|'+ '|'.join([q['id'],q['base_mode'],number(c.get('minimum_original_probability_buffer_margin'),7),
            str(c.get('encoded_domain_feasible')),str(c.get('audit_global_optimality_supported',False)),
            str(history.get('recorded_bound_conflict_with_verified_buffered_point'))])+'|')
    lines += ['', '|输入|模式|原节点群日电费/元|新节点群日电费/元|日电费变化/元|服务费加日电费变化/元|', '|---|---|---:|---:|---:|---:|']
    for row in data['operational_summary']:
        lines.append(f"|{row['id']}|{row['policy']}|{number(row['baseline_root_energy_yuan']['mean'],10)}|{number(row['root_energy_yuan']['mean'],10)}|{number(row['root_energy_change_yuan']['mean'])}|{number(row['service_plus_energy_change_yuan']['mean'])}|")
    lines += ['', '节点群整日购售电总费、灵活性服务费及二者增量分别列示；服务费不是整日电费。根购售电费用使用示例分时价格，未作实证电价校准。服务费加电费变化不作本轮最小化目标。共享准备每输入只计一次，旧初值生成的历史耗时另列。', '',
        f"任务失败 {len(data['original_failed_tasks'])}；报价异常 {data['original_quote_exceptions']}；执行失败 {data['original_execution_failed_rows']}；独立审计错误 {len(data['audit_errors'])}。",
        f"完整原生候选SSE复算最大误差 {data['maximum_native_SSE_replay_error']:.3g}。",
        '设备额定值、运行基线、参与比例、响应和价格的假设来源保留在设备清单；登记容量不等于实测可调能力。', '']
    return '\n'.join(lines)


def report(audit_path, out):
    reader=shared.Reader();audit=reader.json(audit_path)
    if audit.get('status')!='passed':
        raise ValueError('Passed independent audit required')
    protocol=reader.json(audit['source_protocol'])
    root=Path(audit['source_protocol']['path']).parent
    complete=reader.json(root/'complete.json')
    entries=reader.json(audit['source_inputs']);entrymap={e['id']:e for e in entries}
    coverage={c['id']:c for c in audit['input_coverage']}
    if set(coverage)!=set(entrymap):
        raise ValueError('Input/coverage denominator differs')
    modes=list(protocol['modes']);repeats=int(protocol['repeats'])
    post=Postprocess(reader,entrymap,coverage,root,baseline_only=True)
    enriched=[];raw={}
    for origin,field in [('zero','zero_records'),('band','band_baseline_records'),('equipment','records')]:
        rows=reader.lines(audit[field]);raw[origin]=rows
        for row in rows:
            value=post.enrich(row,'new_execution' if origin=='equipment' else 'reused_'+origin)
            value['original_policy']=value['strategy'];value['strategy']=renamed(value['strategy'],origin)
            for key in ('requested_devices','delivered_devices','planning_band','delivered_band'):
                value[key]=row.get(key)
            enriched.append(value)
    expected=len(entries)*len(modes)*repeats
    if (len(raw['equipment'])!=expected or len(raw['band'])!=expected
            or len(raw['zero'])!=len(entries)*repeats or audit['audited_rows']!=expected):
        raise ValueError('New/reused denominator differs')
    if len({shared.key(r) for r in enriched})!=len(enriched):
        raise ValueError('Duplicate paired row')
    cases=[]
    for tag,e in entrymap.items():
        rows=[r for r in enriched if r['id']==tag]
        summary={name:single.summarize([r for r in rows if r['strategy']==name]) for name in sorted({r['strategy'] for r in rows})}
        comparisons={'no_dfs':{},'old_band':{}}
        for mode in modes:
            name='equipment__'+mode
            comparisons['no_dfs'][name]=shared.paired(rows,name,'no_dfs')
            comparisons['old_band'][name]=shared.paired(rows,name,'band__'+mode)
        cases.append(dict(id=tag,dimensions={k:e.get(k) for k in
            ('case','start_date','stride','meter_multiplier','training_rows','validation_rows')},
            coverage=coverage[tag],strategies=summary,comparisons=comparisons))
    zero={(r['id'],r['repeat']):r for r in raw['zero']};operation=[]
    for tag in entrymap:
        for mode in modes:
            name='single_'+mode;rows=[r for r in raw['equipment'] if r['id']==tag and r['strategy']==name]
            good=[r for r in rows if r['audit_status']=='passed']
            operation.append(dict(id=tag,policy=name,n=len(rows),completed=len(good),
                device_failed_rows=sum(not r['delivered_devices']['passed'] for r in good),
                delivered_band_failed_rows=sum(not r['delivered_band']['passed'] for r in good),
                voltage_failed_rows=sum(not r['voltage_screen_passed'] for r in good),
                voltage_minimum=min((r['electrical']['all_bus_voltage']['minimum'] for r in good),default=None),
                voltage_maximum=max((r['electrical']['all_bus_voltage']['maximum'] for r in good),default=None),
                maximum_branch_current_A=max((r['electrical']['maximum_branch_current_A'] for r in good),default=None),
                maximum_root_S_kVA=max((r['electrical']['root_S_kVA']['maximum'] for r in good),default=None),
                root_energy_yuan=shared.stats([r['root_energy_value_yuan'] for r in good]),
                baseline_root_energy_yuan=shared.stats([r['baseline_root_energy_value_yuan'] for r in good]),
                root_energy_change_yuan=shared.stats([r['root_energy_value_change_yuan'] for r in good]),
                service_plus_energy_change_yuan=shared.stats([r.get('service_plus_root_energy_change_yuan',r['settled_cost']['total_yuan']+r['root_energy_value_change_yuan']) for r in good]),
                actual_net_negative_background_P_cells=max((r.get('actual_net_baseline_decomposition',{}).get('negative_background_P_cells',0) for r in good),default=0),
                active_loss_change_kwh=shared.stats([r['electrical']['active_loss_kwh']-zero[tag,r['repeat']]['electrical']['active_loss_kwh'] for r in good])))
    quotes=reader.lines(audit['quotes']);reader.checked(__file__)
    device_summary=device_readout(reader,entries,root,raw['equipment'],modes)
    timing=[]
    for tag in entrymap:
        qs=[q for q in quotes if q['id']==tag];shared_seconds={q.get('shared_preparation_seconds',0.) for q in qs}
        if len(shared_seconds)!=1:raise ValueError('Shared preparation differs across same-case quotes')
        preparation=next(iter(shared_seconds))
        timing.append(dict(id=tag,shared_preparation_seconds_once=preparation,
            quote_wall_seconds_sum=sum(q.get('quote_wall_seconds',q.get('seconds',0.)) for q in qs),
            total_method_seconds_including_shared_once=preparation+sum(q.get('quote_wall_seconds',q.get('seconds',0.)) for q in qs),
            scope='Serial method work, excluding AC and audit; not parallel batch elapsed wall time.'))
    if not reader.unchanged():
        raise ValueError('Inputs changed during report')
    result=dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),source_audit=io.ref(audit_path),
        source_protocol=audit['source_protocol'],source_inputs=audit['source_inputs'],smoke=protocol.get('smoke',False),
        modes=modes,repeats=repeats,cases=cases,path_records=enriched,operational_summary=operation,
        quote_records=quotes,quote_summary=single.quotation_summary(quotes),timing=timing,
        new_rows=expected,band_baseline_rows=len(raw['band']),zero_rows=len(raw['zero']),
        new_ac_rows=complete.get('new_ac_rows'),reused_ac_rows=complete.get('reused_ac_rows'),
        original_failed_tasks=audit.get('original_failed_tasks',[]),
        original_quote_exceptions=audit.get('original_quote_exceptions',0),
        original_execution_failed_rows=audit.get('original_execution_failed_rows',0),
        audit_errors=audit.get('task_errors',[]),maximum_native_SSE_replay_error=post.maximum_error,
        device_checks=audit.get('device_checks'),device_readout=device_summary,sources_unchanged=True,
        statistical_scope=f'Three conditions, two previously studied physical backgrounds, {repeats} reused paired paths per condition; not independent confirmation.',
        bound_scope='New device-domain activation-fee QCQP only. Total settlement is not the optimized objective; no inherited old free-Q cost bound.',
        input_refs=[dict(path=p,sha256=h) for p,h in reader.refs.items()])
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    io.write(out/'report.json',result);(out/'REPORT.md').write_text(markdown(result),encoding='utf-8')
    import csv
    flat=[]
    for case in cases:
        for policy,s in case['strategies'].items():
            flat.append(dict(id=case['id'],policy=policy,n=s['n'],internal_recovery=s['recovered'],
                full_support_recovery=s['full_support_recovered'],wrong_high=s['wrong_high'],mean_p_true=s['p_true']['mean'],
                activation_yuan=s['settled_activation_yuan']['mean'],availability_yuan=s['settled_availability_yuan']['mean'],
                total_yuan=s['settled_total_yuan']['mean'],delivered_P_kwh=s['quantities']['delivered_P_kwh']['mean'],
                delivered_Q_kvarh=s['quantities']['delivered_Q_kvarh']['mean']))
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    write_csv(out/'resource_catalog.csv',device_summary['catalog'])
    io.write(out/'resource_catalog.json',device_summary['catalog'])
    device_rows=[]
    for row in device_summary['by_device']:
        flat={k:v for k,v in row.items() if k not in ('requested','delivered')}
        for phase in ('requested','delivered'):
            flat.update({phase+'_'+k:v for k,v in (row[phase] or {}).items()})
        device_rows.append(flat)
    write_csv(out/'device_calls_and_states.csv',device_rows)
    class_rows=[]
    for row in device_summary['by_class']:
        flat={k:v for k,v in row.items() if k not in ('requested','delivered')}
        for phase in ('requested','delivered'):flat.update({phase+'_'+k:v for k,v in (row[phase] or {}).items()})
        class_rows.append(flat)
    write_csv(out/'device_class_calls.csv',class_rows)
    io.write(out/'complete.json',dict(status='complete',report=io.ref(out/'report.json')))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit');parser.add_argument('--out');parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:print(self_test())
    else:
        if not args.audit or not args.out:parser.error('--audit and --out are required')
        print(report(args.audit,args.out)['status'])
