"""Compare the two predeclared guard rounds from completed, audited reports only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

MODES=('normal_fixed','mgf_fixed','mgf_forecast')
VARIANTS=('original','pq4','response025','response035')
LABELS={'original':'原方案','pq4':'P/Q 4','response025':'响应 0.25','response035':'响应 0.35'}


class Reader:
    def __init__(self):self.refs={}
    def checked(self,ref):
        p=Path(ref['path'] if isinstance(ref,dict) else ref).resolve();key=str(p)
        if key not in self.refs:self.refs[key]=hashlib.sha256(p.read_bytes()).hexdigest()
        if isinstance(ref,dict) and ref.get('sha256')!=self.refs[key]:raise ValueError('Source hash changed: '+key)
        return p
    def read(self,ref):return json.loads(self.checked(ref).read_text(encoding='utf8'))
    def unchanged(self):return all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in self.refs.items())


def identity(r):return r['id'],int(r['forecast']),int(r['repeat']),r['strategy']


def unique(rows):
    result={identity(r):r for r in rows}
    if len(result)!=len(rows):raise ValueError('Duplicate input/forecast/repeat/strategy row')
    return result


def baseline_identity(first,second):
    a=unique([r for r in first['path_records'] if r['data_origin']=='reused_baseline'])
    b=unique([r for r in second['path_records'] if r['data_origin']=='reused_baseline'])
    if set(a)!=set(b):raise ValueError('Rounds refer to different baseline paths')
    for k in a:
        if Path(a[k]['source']).resolve()!=Path(b[k]['source']).resolve():
            raise ValueError('Baseline observation source differs: '+str(k))
        # Both reports replay exactly the same sealed rows, not a fresh baseline draw.
        for field in ('audit_status','original_status','evaluation','planned_cost','settled_cost'):
            if a[k].get(field)!=b[k].get(field):raise ValueError('Baseline content differs: '+str(k)+' / '+field)
    return a


def correct(row):return row['audit_status']=='passed' and bool(row['evaluation'].get('native_correct'))


def paired_response(first,second,tag,mode):
    strategy=mode+'__response'
    def take(report):
        return {(r['forecast'],r['repeat']):r for r in report['path_records']
            if r['id']==tag and r['strategy']==strategy and r['data_origin']=='new_guard_execution'}
    a,b=take(first),take(second)
    if not a or set(a)!=set(b):raise ValueError('Tight/loose response paired denominator differs')
    gains=[];losses=[];changed=[]
    for k in a:
        if correct(b[k]) and not correct(a[k]):gains.append(list(k))
        if correct(a[k]) and not correct(b[k]):losses.append(list(k))
        if a[k]['evaluation'].get('native_selected_index')!=b[k]['evaluation'].get('native_selected_index'):changed.append(list(k))
    return dict(id=tag,base_mode=mode,n=len(a),tight_recovered=sum(correct(r) for r in a.values()),
        loose_recovered=sum(correct(r) for r in b.values()),new_recoveries=len(gains),lost_recoveries=len(losses),
        new_recovery_paths=gains,lost_recovery_paths=losses,changed_winner_paths=changed,
        tight_recorded_failures=sum(r['audit_status']!='passed' for r in a.values()),
        loose_recorded_failures=sum(r['audit_status']!='passed' for r in b.values()))


def number(value,digits=3):return '—' if value is None else f'{value:.{digits}f}'


def bankcell(summary,law):
    value=summary['whole_bank_fixed_true_target'].get(law)
    return '—' if value is None else number(value['minimum_logodds']['mean'])+' / '+number(value['minimum_logodds_increment']['mean'])


def bound_conflict(quote):
    cert=(quote or {}).get('certificate') or {};history=cert.get('native_bound_history') or {}
    return dict(lower_bound_withdrawn=bool(cert.get('lower_bound_withdrawn',(quote or {}).get('lower_bound_withdrawn'))),
        trajectory_conflict=bool(history.get('recorded_bound_conflict_with_verified_buffered_point')),
        raw_native_bound_minus_direct_fee_yuan=(quote or {}).get('raw_native_bound_minus_direct_fee_yuan'),
        native_bound_history=history or None,audit_global_optimality_supported=cert.get('audit_global_optimality_supported'))


def markdown(data):
    lines=['# 两轮冻结参数预算对照','']
    if data['interface_smoke']:lines+=['**仅接口检查：使用烟测报告，不据此评价方案效果。**','']
    c=data['counts'];lines += [f"旧基线只计 {c['baseline_rows_counted_once']} 行；第一轮新增 {c['first_new_rows']} 行，第二轮新增 {c['second_new_rows']} 行。",
        '两轮复用同一历史、物理日、预测和执行误差路径；第二轮统一将响应RMS档位从0.25改为0.35。重复引用的基线不增加样本数。',
        '恢复数按内部拓扑统计；完整支持还要求单节点和公共响应支路一致。预算通过与模型内概率门槛均不等同实际电网的恢复率保证。','',
        '|输入|原概率方法|调度限制|内部正确|完整支持|错误高权重|结算元|P kWh / Q kvarh|零回退|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in data['groups']:
        s=row['summary'];q=s['quantities']
        lines.append(f"|{row['id']}|{row['base_mode']}|{LABELS[row['variant']]}|{s['recovered']}/{s['n']}|{s['full_support_exact_count']}/{s['n']}|{s['wrong_high']}|{number(s['settled_total_yuan']['mean'])}|{number(q['delivered_P_kwh']['mean'])} / {number(q['delivered_Q_kvarh']['mean'])}|{s['zero_fallback_rows']}|")
    lines += ['', '## 响应预算放宽后的逐路径变化','',
        '|输入|原概率方法|紧/松内部正确|新增正确|丢失正确|紧/松执行失败|',
        '|---|---|---:|---:|---:|---:|']
    for p in data['paired_response']:
        lines.append(f"|{p['id']}|{p['base_mode']}|{p['tight_recovered']}/{p['loose_recovered']}（各{p['n']}）|{p['new_recoveries']}|{p['lost_recoveries']}|{p['tight_recorded_failures']}/{p['loose_recorded_failures']}|")
    lines += ['', '## 全库最弱优势：绝对值 / 相对零动作增量','',
        '|输入|原概率方法|调度限制|预测K|真实输入K|干净AC|量测|实际H/K响应误差|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for row in data['groups']:
        error=row['fixed_true_response_error'];error_text='—' if error is None else number(error['H_mean_square_over_sigma2']['mean'])+' / '+number(error['K_mean_square_over_sigma2']['mean'])
        lines.append('|'+row['id']+'|'+row['base_mode']+'|'+LABELS[row['variant']]+'|'+'|'.join(bankcell(row['summary'],law) for law in ('forecast_K','actual_input_K','clean','observed'))+'|'+error_text+'|')
    lines += ['', '固定目标为历史评分最好的真内部族参数点，真值仅用于评价定位。预测列取各竞争者期望赔率的最小值；实际列允许最强错误竞争者改变。实际H/K响应误差是动作响应增量预测残差的均方除评分方差，和设计参数方差预算分开。机器表还保留历史赢家目标口径、各代表参数误差、预算值及全部竞争者记录。','',
        '## 求解状态与费用界','',
        '|输入/方法/限制|原生码|报价可行|界撤回|轨迹冲突|原始LB−直接费|原始有效UB/LB/gap|',
        '|---|---:|---:|---:|---:|---:|---|']
    for row in data['groups']:
        q=row['quote'] or {};cert=q.get('certificate') or {};b=row['bound_conflict']
        upper=cert.get('feasible_upper_bound_yuan',q.get('cost_yuan'));lower=cert.get('lower_bound_yuan',q.get('lower_bound_yuan'));gap=cert.get('relative_gap',q.get('relative_gap'))
        lines.append('|'+row['id']+' / '+row['base_mode']+' / '+LABELS[row['variant']]+f"|{q.get('native_status')}|{q.get('new_gate_feasible',q.get('feasible'))}|{b['lower_bound_withdrawn']}|{b['trajectory_conflict']}|"+number(b['raw_native_bound_minus_direct_fee_yuan'],9)+'|'+number(upper)+' / '+number(lower)+' / '+number(gap)+'|')
    lines += ['', '以上费用界来自对应原问题或新增预算问题的独立审计证书，保留撤回及轨迹冲突。最终原生状态码不覆盖证书异常。两轮共享机器并行运行，耗时差不能单独归因于预算档位。','',
        '## 对应轮换对照','',
        '|输入|原概率方法|限制|轮换内部正确|轮换结算元|采购相对轮换新增/丢失|',
        '|---|---|---|---:|---:|---:|']
    for row in data['rotations']:
        s=row['summary'];p=row['procurement_vs_rotation']
        lines.append(f"|{row['id']}|{row['base_mode']}|{LABELS[row['variant']]}|{s['recovered']}/{s['n']}|{number(s['settled_total_yuan']['mean'])}|{p['new_recoveries']}/{p['lost_recoveries']}|")
    lines += ['', '轮换按原物理合同和费用匹配生成，不要求通过新增预算；其报价状态字段指对应采购方案。未匹配费用例和原始失败记录保留在机器表。','']
    return '\n'.join(lines)


def plot(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    font=font_manager.findfont('Microsoft YaHei',fallback_to_default=False)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':9,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    caseids=data['input_order'];lookup={(g['id'],g['base_mode'],g['variant']):g['summary'] for g in data['groups']}
    fig,axes=plt.subplots(len(caseids),3,figsize=(14.6,3.5*len(caseids)),squeeze=False)
    titles=('正态约束','固定功率指数约束','含预测误差指数约束');colors=('#89939e','#397bb8','#d48b35','#3a977c')
    for i,tag in enumerate(caseids):
        name=tag.replace('__2024-01-04__s2__','  ').replace('m0p5','噪声 ×0.5').replace('m1p0','噪声 ×1').replace('m2p0','噪声 ×2')
        for j,mode in enumerate(MODES):
            ax=axes[i,j];groups=[lookup[(tag,mode,v)] for v in VARIANTS];n=groups[0]['n'];pad=max(.06,.025*n)
            zero=data['baseline_controls'][tag]['no_dfs']['recovered'];ax.axhline(zero,color='#555555',ls='--',lw=1,label='零采购')
            for x,g in enumerate(groups):
                fallback=g['zero_fallback_rows']>0
                ax.bar(x,g['recovered'],width=.65,color=colors[x],hatch='//' if fallback else None,edgecolor='#555555' if fallback else 'none')
                ax.text(x,g['recovered']+pad,f"{g['recovered']}/{g['n']}",ha='center',va='bottom',fontsize=10)
                fee=g['settled_total_yuan']['mean'];label='费用缺失' if fee is None else f'¥{fee:.2f}'
                ax.text(x,-.115*n-.03,label+(' 回退' if fallback else ''),ha='center',fontsize=8)
            ax.set_title(name+'\n'+titles[j],fontsize=10)
            ax.set_xticks(np.arange(4),[LABELS[v] for v in VARIANTS]);ax.set_ylim(-.21*n-.2,n+max(.3,.18*n))
            ax.set_yticks(np.arange(0,n+1,max(1,n//4)));ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            if j==0:ax.set_ylabel('内部拓扑恢复次数')
            if i==0 and j==2:ax.legend(frameon=False,fontsize=8)
    fig.suptitle('冻结 R/X：两档响应预算与 P/Q 预算的配对比较'+('（仅接口检查）' if data['interface_smoke'] else ''),fontsize=16,y=.995)
    fig.text(.5,.006,'柱上数字为内部拓扑恢复次数；费用为平均实际结算总费用。斜线表示报价失败后零采购回退。\n同一历史及物理日重复使用相同误差路径；完整支持、逐路径丢失和费用界异常另表报告。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.96))
    fig.savefig(out/'comparison.png',dpi=180,bbox_inches='tight');fig.savefig(out/'comparison.pdf',bbox_inches='tight');plt.close(fig)
    return font


def compare(first_path,second_path,out,*,smoke=False):
    reader=Reader();first=reader.read(first_path);second=reader.read(second_path)
    for r in (first,second):
        if r.get('status')!='complete' or r.get('sources_unchanged') is not True:raise ValueError('Completed unchanged report required')
        if bool(r.get('smoke'))!=bool(smoke):raise ValueError('Smoke/formal report boundary mismatch')
        audit=reader.read(dict(path=r['source_audit'],sha256=r['audit']))
        if audit.get('status')!='passed':raise ValueError('Independent passed audit required')
    pa,pb=reader.read(first['source_protocol']),reader.read(second['source_protocol'])
    if not smoke:
        if first.get('kinds')!=['pq','response'] or second.get('kinds')!=['response']:raise ValueError('Expected two-guard first round and response-only second round')
        if first.get('response_rms')!=.25 or second.get('response_rms')!=.35 or first.get('pq_rho')!=4. or second.get('pq_rho')!=4.:raise ValueError('Unexpected locked budget levels')
        if (first['baseline_rows'],second['baseline_rows'],first['new_rows'],second['new_rows'])!=(384,384,576,288):raise ValueError('Formal 384 reused +576/+288 new denominator mismatch')
    for k in ('seed','forecasts','repeats','solver_seconds','target_mass','forecast_relative_sd','modes'):
        if pa.get(k)!=pb.get(k):raise ValueError('Round protocol mismatch: '+k)
    if tuple(first['modes'])!=MODES or tuple(second['modes'])!=MODES:raise ValueError('All three original modes required')
    baseline=baseline_identity(first,second)
    # Each reused observation/receipt must still match the hashes recorded in both reports.
    refs=[{str(Path(x['path']).resolve()):x['sha256'] for x in r['input_refs']} for r in (first,second)]
    for row in baseline.values():
        for name in ('observations.npz','receipt.json'):
            path=(Path(row['source'])/name).resolve();hashes=[m.get(str(path)) for m in refs]
            if not hashes[0] or hashes[0]!=hashes[1]:raise ValueError('Baseline source hashes differ: '+str(path))
            reader.checked(dict(path=str(path),sha256=hashes[0]))
    ea,eb=reader.read(first['source_inputs']),reader.read(second['source_inputs'])
    ea={e['id']:e for e in (ea['inputs'] if isinstance(ea,dict) else ea)};eb={e['id']:e for e in (eb['inputs'] if isinstance(eb,dict) else eb)}
    if set(ea)!=set(eb):raise ValueError('Different input sets')
    for tag in ea:
        for k in ('files','score','base_kw','physical_cache','resource_spec','p_resources','q_resources','forecast_ref','prior_seal','prior_results','initial_points'):
            if ea[tag].get(k)!=eb[tag].get(k):raise ValueError('Frozen input changed: '+tag+' / '+k)
    ca={c['id']:c for c in first['cases']};cb={c['id']:c for c in second['cases']}
    if set(ca)!=set(cb) or set(ca)!=set(ea):raise ValueError('Report case ledger differs')
    groups=[];rotations=[];pairing=[]
    def quote(report,tag,policy,old=False):
        rows=report['baseline_quotes' if old else 'quotes']['rows'];found=[q for q in rows if q['id']==tag and q['policy']==policy]
        if len(found)!=1:raise ValueError('Expected exactly one quote: '+tag+' / '+policy)
        return found[0]
    for tag in ca:
        for mode in MODES:
            specs=(('original',first,ca[tag],mode),('pq4',first,ca[tag],mode+'__pq'),
                ('response025',first,ca[tag],mode+'__response'),('response035',second,cb[tag],mode+'__response'))
            for variant,source,case,policy in specs:
                s=case['strategies'][policy];q=quote(source,tag,policy,variant=='original')
                target=case['comparison_targets']['fixed_true_target']
                groups.append(dict(id=tag,base_mode=mode,variant=variant,policy=policy,summary=s,quote=q,
                    fixed_true_target=target,fixed_true_response_error=s['actual_response_errors'].get(str(target)),
                    bound_conflict=bound_conflict(q),source_round='reused_baseline' if variant=='original' else 'first' if source is first else 'second'))
                rotations.append(dict(id=tag,base_mode=mode,variant=variant,policy=policy+'_rotation',
                    summary=case['strategies'][policy+'_rotation'],procurement_vs_rotation=case['paired_vs_same_fee_rotation'][policy]))
            pairing.append(paired_response(first,second,tag,mode))
    if len(baseline)!=first['baseline_rows'] or len(baseline)!=second['baseline_rows']:raise ValueError('Baseline denominator mismatch')
    reader.checked(__file__)
    data=dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),interface_smoke=bool(smoke),
        first=dict(path=str(Path(first_path).resolve()),sha256=reader.refs[str(Path(first_path).resolve())]),
        second=dict(path=str(Path(second_path).resolve()),sha256=reader.refs[str(Path(second_path).resolve())]),
        input_order=list(ca),counts=dict(baseline_rows_counted_once=len(baseline),first_new_rows=first['new_rows'],
            second_new_rows=second['new_rows'],unique_execution_records=None if smoke else len(baseline)+first['new_rows']+second['new_rows'],
            reused_baseline_duplicates_removed=len(baseline)),
        groups=groups,paired_response=pairing,rotations=rotations,
        baseline_controls={tag:{k:v for k,v in ca[tag]['strategies'].items() if v['data_origin_counts'].get('reused_baseline')} for tag in ca},
        timing=dict(first=first['case_timing'],second=second['case_timing'],shared_machine_concurrency_confounds_timing=True),
        original_failures=dict(first=first['recorded_failures'],second=second['recorded_failures']),
        task_errors=dict(first=first.get('task_errors',[]),second=second.get('task_errors',[])),
        reused_original_failed_tasks=first.get('original_failed_tasks',[]),
        reused_original_execution_failed_rows=first.get('original_execution_failed_rows',[]),
        unmatched_rotation_fees=dict(first=first['rotation_unmatched_fees'],second=second['rotation_unmatched_fees']),
        scope='Matched development replay; same physical days and random paths. Internal topology and full response support are distinct. No new solver, fit, AC or sampling invoked.',
        baseline_identity_verified=True,verified_input_refs=[dict(path=p,sha256=h) for p,h in reader.refs.items()])
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    data['font_path']=plot(data,out);data['sources_unchanged']=reader.unchanged()
    if not data['sources_unchanged']:raise ValueError('Source changed during comparison')
    (out/'report.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    (out/'REPORT.md').write_text(markdown(data),encoding='utf8')
    artifacts=[dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.iterdir()) if p.is_file()]
    (out/'complete.json').write_text(json.dumps(dict(status='complete',sources_unchanged=True,artifacts=artifacts),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--first',type=Path,required=True);p.add_argument('--second',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true',help='Interface check only; both inputs must be smoke reports')
    a=p.parse_args();r=compare(a.first,a.second,a.out,smoke=a.smoke);print(json.dumps(dict(status=r['status'],counts=r['counts'],sources_unchanged=r['sources_unchanged']),ensure_ascii=False))


if __name__=='__main__':main()
