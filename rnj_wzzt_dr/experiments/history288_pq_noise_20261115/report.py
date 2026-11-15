"""Readout of passed, independent history/PQ-error audits.

Only sealed data are read. Noise repetitions are nested in two fixed physical
days; conditional execution counts are never labelled independent histories.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

from ... import io
from ..equipment_flex_20261115 import report as equipment
from ..single_target_20261115 import report as single
from ..frozen_rx_guard_20261115 import report as shared

ROOT=Path(__file__).resolve().parents[3]
HSTAR_ROOT=ROOT/'outputs/oracle_direction_execution_20261115_v1/cases'
HSTAR_IDS={'pengwah18':'pengwah18__2024-01-04__s2__m2p0','soumalas11':'soumalas11__2024-01-04__s2__m1p0'}


def native_matrix(point,n):
    H=np.zeros((n,2*n))
    for support,r,x in zip(point['supports'],point['r'],point['x'],strict=True):
        ix=np.ix_(support,support);H[:,:n][ix]+=r;H[:,n:][ix]+=x
    return H


def historical_summary(cells,view):
    models=[c['fit']['models'][view] for c in cells if c.get('fit')]
    return dict(n=len(cells),completed=len(models),failed=len(cells)-len(models),
        correct=sum(m['history_correct'] for m in models),
        full_support_correct=sum(m['history_full_support_exact'] for m in models),
        covered=sum(m['truth_in_native_bank'] for m in models),
        full_support_covered=sum(m['full_support_truth_in_native_bank'] for m in models),
        candidate_count=shared.stats([m['bank_size'] for m in models]),
        actual_fit_seconds=shared.stats([m['actual_fit_seconds'] for m in models]),
        stop_reason_counts=dict(Counter(m['stop_reason'] for m in models)),
        budget_exceeded=sum(m['total_cap_exceeded'] for m in models),
        denominator='Independent historical added-noise draws within one fixed physical background; not independent network-days.')


def pooled_strategy(rows):
    result=single.summarize(rows)
    # A native index is local to one freshly fitted bank. Never pool its error
    # or truth flag across independently fitted banks under the same integer.
    result.pop('actual_response_errors',None)
    result.pop('audited_winner_counts',None)
    for key in ('whole_bank_historical_target','whole_bank_fixed_true_target'):
        for law in (result.get(key) or {}).values():
            if isinstance(law,dict):law.pop('strongest_competitor_counts',None)
    result['candidate_identity_scope']='Candidate indices and R/X matrices are cell-local. Cross-fit summaries combine scalar outcomes, not candidate identities.'
    return result


def rx_diagnostics(reader,entry,fit):
    """Use only the pre-existing separately audited physical response matrix."""
    directory=HSTAR_ROOT/HSTAR_IDS[entry['case']]
    if not (directory/'geometry.json').exists():
        return dict(status='unavailable',reason='No previously sealed physical response matrix')
    metadata=reader.json(directory/'geometry.json');arrays=reader.arrays(metadata['arrays'])
    physical_hash=entry['physical_cache'].get('network_hash')
    if physical_hash is None:physical_hash=entry['physical_cache'].get('physical_network_hash')
    if physical_hash!=metadata['physical_network_hash'] or metadata['base_kw']!=entry['base_kw']:
        raise ValueError('Offline response matrix physical identity/base differs')
    n=len(entry['labels']);Hstar=arrays['Hstar']
    if Hstar.shape!=(n,2*n):raise ValueError('Offline response dimensions differ')
    original_score=reader.json(metadata['score'])
    if original_score['labels']!=entry['labels']:raise ValueError('Offline response label order differs')
    validation=reader.arrays(entry['files']['validation.npz']);observed=reader.arrays(entry['files']['new_day.npz'])
    clean=reader.arrays(entry['physical_cache']['original'])
    Xclean=np.c_[clean['P'][288:384],clean['Q'][288:384]]
    Yclean=clean['root'][288:384,None]**2-clean['V'][288:384]**2
    rows=[]
    for view in ('original','enriched'):
        model=fit['models'][view];native=reader.json(model['native'])
        good=[r for r in model['candidates'] if r['internal_correct']]
        true=min(good,key=lambda r:(r['validation_sse'],r['native_index']))['native_index'] if good else None
        for role,index in (('historical_winner',model['history_selected']),('best_true_by_validation',true)):
            if index is None:
                rows.append(dict(view=view,role=role,available=False,reason='No internally correct candidate'));continue
            H=native_matrix(native['frozen_candidates'][index]['point'],n);difference=H-Hstar
            answer=dict(view=view,role=role,available=True,native_index=index,
                internal_correct=model['candidates'][index]['internal_correct'],
                R_relative_frobenius=float(np.linalg.norm(difference[:,:n])/np.linalg.norm(Hstar[:,:n])),
                X_relative_frobenius=float(np.linalg.norm(difference[:,n:])/np.linalg.norm(Hstar[:,n:])),
                H_relative_frobenius=float(np.linalg.norm(difference)/np.linalg.norm(Hstar)),
                validation_mse=model['candidates'][index]['validation_mse'],
                new_day_clean_AC_mse=float(np.square(Yclean-Xclean@H.T).mean()),
                evaluation_path0_observed_mse=float(np.square(observed['Y']-np.c_[observed['P'],observed['Q']]@H.T).mean()))
            rows.append(answer)
    return dict(status='available',source=metadata['arrays'],source_metadata=io.ref(directory/'geometry.json'),rows=rows,
        scope='Offline linear physical response matrices in squared-voltage/per-unit convention. Clean AC residual also includes linearization error; these are not fitted ohm coefficients or planning inputs.')


def operation(rows):
    good=[r for r in rows if r['audit_status']=='passed']
    return dict(n=len(rows),completed=len(good),device_failed_rows=sum(not r['delivered_devices']['passed'] for r in good),
        delivered_band_failed_rows=sum(not r['delivered_band']['passed'] for r in good),
        voltage_failed_rows=sum(not r['voltage_screen_passed'] for r in good),
        voltage_minimum=min((r['electrical']['all_bus_voltage']['minimum'] for r in good),default=None),
        voltage_maximum=max((r['electrical']['all_bus_voltage']['maximum'] for r in good),default=None),
        maximum_branch_current_A=max((r['electrical']['maximum_branch_current_A'] for r in good),default=None),
        maximum_root_S_kVA=max((r['electrical']['root_S_kVA']['maximum'] for r in good),default=None),
        root_energy_yuan=shared.stats([r['root_energy_value_yuan'] for r in good]),
        baseline_root_energy_yuan=shared.stats([r['baseline_root_energy_value_yuan'] for r in good]),
        root_energy_change_yuan=shared.stats([r['root_energy_value_change_yuan'] for r in good]),
        service_plus_energy_change_yuan=shared.stats([r['service_plus_root_energy_change_yuan'] for r in good]),
        thermal_capacity_scope='Network line and transformer thermal ratings unavailable. Base200kVA is a normalization base.')


def markdown(result):
    f=equipment.number
    lines=['# 三天历史与 P/Q 附加误差试验','',
        ('本文件为短预算接线测试。' if result['smoke'] else '本文件为已知两个物理网络日上的配对开发试验。'),'',
        f"计划 {result['planned_inputs']} 组历史，已审计 {result['audited_rows']} 条策略记录；每组每策略 {result['repeats']} 条执行路径。",
        '每组重新拟合候选与 R/X，192点训练、96点验证；最终评分只累计96点验证和96点新日。V/root附加标准差固定0.05%，P/Q各档共用标准噪声。',
        '设备与预测保持相同；动作带从各组训练重新估计。因此比较的是整个流程对P/Q误差的敏感性，包含参数和动作带的变化。','',
        '|网络|P/Q标准差|原辨识正确/历史数|追加后正确/历史数|原/追加真族覆盖|原/追加完整支持覆盖|',
        '|---|---:|---:|---:|---:|---:|']
    for g in result['groups']:
        a,b=g['history']['original'],g['history']['enriched']
        lines.append(f"|{g['case']}|{100*g['pq_relative_sd']:g}%|{a['correct']}/{a['n']}|{b['correct']}/{b['n']}|{a['covered']}/{b['covered']}|{a['full_support_covered']}/{b['full_support_covered']}|")
    lines+=['','|网络|P/Q标准差|策略|内部正确/路径|完整支持正确|新增/丢失|错误权重≥.9|调用/预留/总费 元|交付P kWh/Q kvarh|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for g in result['groups']:
        for name,s in g['strategies'].items():
            pair=g['paired_vs_no_dfs'][name]
            lines.append('|'+ '|'.join([g['case'],f"{100*g['pq_relative_sd']:g}%",name,f"{s['recovered']}/{s['n']}",str(s['full_support_recovered']),
                f"{pair['new_recoveries']}/{pair['lost_recoveries']}",str(s['wrong_high']),
                '/'.join(f(s['settled_'+k+'_yuan']['mean']) for k in ('activation','availability','total')),
                f(s['quantities']['delivered_P_kwh']['mean'])+'/'+f(s['quantities']['delivered_Q_kvarh']['mean'])])+'|')
    lines+=['','4次历史噪声重复和每次16条执行路径分层保存于JSON/CSV；64条条件路径不等于64个独立物理日。缺候选、准备失败和执行失败保留在原定分母。',
        '0%表示未额外添加模拟P/Q误差；原实测来源的误差仍未知。','',
        '## 冻结响应矩阵误差','',
        '下表仅在评价时读取既有独立封存的物理线性R/X响应矩阵，报告相对Frobenius范数误差的历史重复平均。正确族列是在库中正确内部族里按验证MSE最低选取，缺库时保留缺失。',
        '|网络|P/Q标准差|原赢家R/X|追加赢家R/X|追加正确族R/X|正确族可用历史数|',
        '|---|---:|---:|---:|---:|---:|']
    for g in result['groups']:
        d=g['response_parameter_errors'];a=d['original']['historical_winner'];b=d['enriched']['historical_winner'];t=d['enriched']['best_true_by_validation']
        pair=lambda x:'/'.join(f(x[k]['mean']) for k in ('R_relative_frobenius','X_relative_frobenius'))
        lines.append(f"|{g['case']}|{100*g['pq_relative_sd']:g}%|{pair(a)}|{pair(b)}|{pair(t)}|{t['available']}/{t['planned_histories']}|")
    lines+=['','上述R/X是电压响应系数；干净AC残差还包含线性模型近似误差。逐候选角色的验证MSE、新日干净AC MSE和路径0量测MSE保存在rx_errors.csv。','',
        '## 预测与实际证据','',
        '每格固定历史赢家及其历史最强异族，再平均路径增量。四列分别对应预测输入的历史校正均值K、实际输入的同一K、干净AC和含表计误差的评分。',
        '|网络|P/Q标准差|策略|预测ΔL|实际输入K ΔL|干净AC ΔL|量测ΔL|负量测增量/有效路径|',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for g in result['groups']:
        for name,s in g['strategies'].items():
            v=s['historical_fixed_pair'];inc=v['increments']
            lines.append('|'+ '|'.join([g['case'],f"{100*g['pq_relative_sd']:g}%",name]+[f(inc[k]['mean']) for k in
                ('forecast_K_logodds','actual_input_K_logodds','clean_logodds','observed_logodds')]+[f"{v['negative_observed_increment_count']}/{v['n']}"])+'|')
    lines+=['','概率界仍以既有历史K与V/root误差模型为条件，未新增P/Q表计误差传播保证；固定λ和均分风险充分式无解，与实际恢复率分别汇报。','',
        '## 求解与设备核验','',
        f"报价 {result['quote_summary']['n']}/{result['expected_quotes']}；状态 {result['quote_summary']['status_counts']}；零回退 {result['quote_summary']['zero_fallback']}。",
        f"原生下界撤回 {result['quote_summary']['lower_bound_withdrawn']}，轨迹界冲突 {result['quote_summary']['bound_trajectory_conflict_count']}。每张报价的初值、时间、原生界、独立有效界与buffer残差保存在quotes表。",
        '优化目标仅为调用费。预留费按请求峰值核算，日购电费另列；Status2或数值gap不能替代严格原式和buffer证书。','',
        '|网络|P/Q标准差|策略|全母线电压范围|越限路径|设备违规路径|交付带超出路径|最大支路A|最大根kVA|',
        '|---|---:|---|---|---:|---:|---:|---:|---:|']
    for g in result['groups']:
        for name,v in g['operation'].items():
            lines.append('|'+ '|'.join([g['case'],f"{100*g['pq_relative_sd']:g}%",name,f(v['voltage_minimum'])+'/'+f(v['voltage_maximum']),
                str(v['voltage_failed_rows']),str(v['device_failed_rows']),str(v['delivered_band_failed_rows']),f(v['maximum_branch_current_A']),f(v['maximum_root_S_kVA'])])+'|')
    lines+=['','支路/变压器热额定未提供，200kVA是标幺基准。设备调用、SOC、EV需求和热水温度逐设备导出；设备基线运行与新增采购分开。',
        f"失败辨识 {len(result['original_failed_fits'])}；任务异常 {len(result['original_failed_tasks'])}；报价异常 {result['original_quote_exceptions']}；执行失败 {result['original_execution_failed_rows']}。",'']
    return '\n'.join(lines)


def report(audit_path,out):
    reader=shared.Reader();audit_path=Path(audit_path).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    audit=reader.json(audit_path)
    if audit['status']!='passed' or audit['task_errors'] or audit['audited_rows']!=audit['expected_rows']:
        raise ValueError('Complete passed independent audit required')
    p=reader.json(audit['source_protocol']);root=Path(audit['source_protocol']['path']).parent
    complete=reader.json(root/'complete.json');entries=reader.json(audit['source_inputs']);entrymap={e['id']:e for e in entries}
    manifest=reader.json(audit['source_data_index']);dp=reader.json(manifest['protocol'])
    tasks=[t for t in dp['tasks'] if not p['smoke'] or (t['history_repeat']==0 and t['pq_relative_sd']==.005)]
    taskmap={t['id']:t for t in tasks};coverage={r['id']:r for r in audit['input_coverage']}
    fits={r['id']:r for r in audit['fit_records']};raw=reader.lines(audit['records']);quotes=reader.lines(audit['quotes'])
    post=equipment.Postprocess(reader,entrymap,coverage,root,baseline_only=True)
    enriched=[]
    for row in raw:
        value=post.enrich(row,'new_histories_paired_execution');entry=taskmap[row['id']]
        value.update(case=entry['case'],pq_relative_sd=entry['pq_relative_sd'],history_repeat=entry['history_repeat'])
        enriched.append(value)
    names=p['policies'];cells=[];flat=[];rx_rows=[]
    successful_data={e['id']:e for e in manifest['inputs']}
    for tag,task in taskmap.items():
        rows=[r for r in enriched if r['id']==tag]
        value=dict(id=tag,case=task['case'],pq_relative_sd=task['pq_relative_sd'],history_repeat=task['history_repeat'],
            fit=fits.get(tag),coverage=coverage.get(tag),strategies={n:single.summarize([r for r in rows if r['strategy']==n]) for n in names},
            paired_vs_no_dfs={n:shared.paired(rows,n,'no_dfs') for n in names})
        if value['fit']:
            value['rx_diagnostics']=rx_diagnostics(reader,successful_data[tag],value['fit'])
            for r in value['rx_diagnostics'].get('rows',[]):rx_rows.append(dict(id=tag,case=task['case'],pq_relative_sd=task['pq_relative_sd'],history_repeat=task['history_repeat'],**r))
        cells.append(value)
        for view in ('original','enriched'):
            h=historical_summary([value],view)
            flat.append(dict(id=tag,case=task['case'],pq_relative_sd=task['pq_relative_sd'],history_repeat=task['history_repeat'],view='history_'+view,policy='',
                n=1,correct=h['correct'],completed=h['completed'],coverage=h['covered'],full_support_correct=h['full_support_correct'],denominator_unit='historical_fit'))
        for policy,s in value['strategies'].items():
            paired=value['paired_vs_no_dfs'][policy]
            flat.append(dict(id=tag,case=task['case'],pq_relative_sd=task['pq_relative_sd'],history_repeat=task['history_repeat'],view='execution',policy=policy,n=s['n'],correct=s['recovered'],completed=s['completed'],
                full_support_correct=s['full_support_recovered'],new_recoveries=paired['new_recoveries'],lost_recoveries=paired['lost_recoveries'],wrong_high=s['wrong_high'],
                mean_p_true=s['p_true']['mean'],activation_yuan=s['settled_activation_yuan']['mean'],availability_yuan=s['settled_availability_yuan']['mean'],total_yuan=s['settled_total_yuan']['mean'],
                delivered_P_kwh=s['quantities']['delivered_P_kwh']['mean'],delivered_Q_kvarh=s['quantities']['delivered_Q_kvarh']['mean'],denominator_unit='conditional_execution_path'))
    groups=[]
    for case,sd in sorted({(t['case'],t['pq_relative_sd']) for t in tasks}):
        cc=[c for c in cells if c['case']==case and c['pq_relative_sd']==sd];ids={c['id'] for c in cc}
        rows=[r for r in enriched if r['id'] in ids];rr=[r for r in raw if r['id'] in ids]
        rrxx=[r for r in rx_rows if r['case']==case and r['pq_relative_sd']==sd]
        response={}
        for view in ('original','enriched'):
            response[view]={}
            for role in ('historical_winner','best_true_by_validation'):
                matched=[r for r in rrxx if r['view']==view and r['role']==role and r['available']]
                response[view][role]=dict(available=len(matched),planned_histories=len(cc),
                    **{k:shared.stats([r[k] for r in matched]) for k in ('R_relative_frobenius','X_relative_frobenius','validation_mse','new_day_clean_AC_mse','evaluation_path0_observed_mse')})
        groups.append(dict(case=case,pq_relative_sd=sd,historical_repetitions=len(cc),history={v:historical_summary(cc,v) for v in ('original','enriched')},
            response_parameter_errors=response,
            strategies={n:pooled_strategy([r for r in rows if r['strategy']==n]) for n in names},
            paired_vs_no_dfs={n:shared.paired(rows,n,'no_dfs') for n in names},
            operation={n:operation([r for r in rr if r['strategy']==n]) for n in names}))
    device_entries=[e for e in entries if (root/'cases'/e['id']/'seal.json').exists()]
    device=equipment.device_readout(reader,device_entries,root,raw,p['modes'])
    device['missing_case_seals']=[e['id'] for e in entries if e not in device_entries]
    if not reader.unchanged():raise ValueError('Source changed during postprocessing')
    result=dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),source_audit=io.ref(audit_path),source_protocol=audit['source_protocol'],
        source_inputs=audit['source_inputs'],smoke=p['smoke'],planned_inputs=p['planned_inputs'],repeats=p['repeats'],modes=p['modes'],policies=names,
        audited_rows=audit['audited_rows'],expected_quotes=p['expected_quotes'],new_DR_rows=p['expected_DR_rows'],zero_rows=p['expected_zero_rows'],
        physical_backgrounds=2,cells=cells,groups=groups,path_records=enriched,quote_records=quotes,quote_summary=single.quotation_summary(quotes),
        device_readout=device,original_failed_fits=audit['original_failed_fits'],original_failed_tasks=audit['original_failed_tasks'],
        original_preparation_failures=audit['original_preparation_failures'],original_quote_exceptions=audit['original_quote_exceptions'],original_execution_failed_rows=audit['original_execution_failed_rows'],
        new_ac_rows=complete.get('new_ac_rows'),reused_ac_rows=complete.get('reused_ac_rows'),maximum_native_SSE_replay_error=post.maximum_error,
        scope='Two fixed physical backgrounds, independent added-history-noise repetitions; execution paths conditional on each frozen fit. Added-noise0 is not physically exact source measurement.',
        input_refs=[dict(path=k,sha256=v) for k,v in reader.refs.items()])
    out.mkdir(parents=True)
    io.write(out/'report.json',result);(out/'REPORT.md').write_text(markdown(result),encoding='utf8')
    equipment.write_csv(out/'summary.csv',flat);equipment.write_csv(out/'rx_errors.csv',rx_rows)
    equipment.write_csv(out/'resource_catalog.csv',device['catalog']);io.write(out/'resource_catalog.json',device['catalog'])
    equipment.write_csv(out/'device_calls_and_states.csv',device['by_device']);equipment.write_csv(out/'device_class_calls.csv',device['by_class'])
    equipment.write_csv(out/'quotes.csv',quotes)
    io.write(out/'complete.json',dict(status='complete',report=io.ref(out/'report.json')))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();print(report(args.audit,args.out)['status'])
