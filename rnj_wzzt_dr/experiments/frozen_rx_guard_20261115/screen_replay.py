"""Post hoc development diagnostic: audit old plans, then replay old cached rows.

No new optimization, AC, fitting or random sampling. Decisions use only the
old audited probability gate and the already frozen historical guard matrices.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import fsum
from pathlib import Path

import numpy as np

MODES=('normal_fixed','mgf_fixed','mgf_forecast')
BUDGETS=(('pq4','pq',16.),('response025','response',.25**2),('response035','response',.35**2))
ROOT=Path(__file__).resolve().parents[3]


class Reader:
    def __init__(self):self.refs={}
    def checked(self,ref):
        p=Path(ref['path'] if isinstance(ref,dict) else ref).resolve();key=str(p)
        if key not in self.refs:self.refs[key]=sha256(p.read_bytes()).hexdigest()
        if isinstance(ref,dict) and ref.get('sha256')!=self.refs[key]:raise ValueError('Changed source: '+key)
        return p
    def read(self,ref):return json.loads(self.checked(ref).read_text(encoding='utf8'))
    def lines(self,ref):return [json.loads(line) for line in self.checked(ref).read_text(encoding='utf8').splitlines() if line.strip()]
    def arrays(self,ref):
        with np.load(self.checked(ref)) as z:return {k:z[k].copy() for k in z.files}
    def ref(self,path):
        p=self.checked(path);return dict(path=str(p),sha256=self.refs[str(p)])
    def unchanged(self):return all(sha256(Path(p).read_bytes()).hexdigest()==h for p,h in self.refs.items())


def budget_value(action,matrix,limit):
    x=np.asarray(action,float).ravel(order='C');Q=np.asarray(matrix,float)
    if Q.shape!=(len(x),len(x)) or not np.isfinite(Q).all() or not np.isfinite(x).all():raise ValueError('Invalid saved action/guard matrix')
    terms=(x[:,None]*Q*x[None,:]).ravel();value=fsum(terms.tolist());absolute=fsum(abs(terms).tolist())
    tolerance=64*np.finfo(float).eps*max(absolute,limit,np.finfo(float).tiny)
    return dict(value=value,limit=limit,slack=limit-value,roundoff_tolerance=tolerance,
        passed=bool(value<=limit+tolerance),within_solver_numerical_buffer=bool(value<=limit*(1.-1e-8)+tolerance))


def screen_decision(original_gate_passed,budget_rows):
    """This function has no outcomes, truth labels or alternate action search."""
    if not budget_rows:raise ValueError('At least one explicit budget required')
    budget_passed=all(r['passed'] for r in budget_rows)
    retain=bool(original_gate_passed and budget_passed)
    return dict(retain_original=retain,original_probability_gate_passed=bool(original_gate_passed),
        added_budget_passed=budget_passed,selected_source='original_plan' if retain else 'no_dfs',
        reason='all_static_checks_passed' if retain else 'original_probability_gate_failed' if not original_gate_passed else 'added_budget_failed')


def recovery(row):return row['audit_status']=='passed' and bool(row['evaluation'].get('native_correct'))


def summary(rows,units):
    good=[r for r in rows if r['audit_status']=='passed'];p=[i for i,u in enumerate(units) if u=='kW'];q=[i for i,u in enumerate(units) if u=='kvar']
    def mean(v):return float(np.mean(v)) if v else None
    return dict(n=len(rows),completed=len(good),recorded_failures=len(rows)-len(good),internal_correct=sum(recovery(r) for r in rows),
        full_support_exact_count=sum(bool(r['evaluation'].get('native_full_support_exact')) for r in good),
        true_high=sum(bool(r['evaluation'].get('true_high')) for r in good),wrong_high=sum(bool(r['evaluation'].get('wrong_high')) for r in good),
        p_true_mean=mean([r['evaluation']['p_true'] for r in good]),
        mean_settled_total_yuan=mean([r['settled_cost']['total_yuan'] for r in good]),
        mean_settled_activation_yuan=mean([r['settled_cost']['activation_yuan'] for r in good]),
        mean_settled_availability_yuan=mean([r['settled_cost']['availability_yuan'] for r in good]),
        mean_delivered_P_kwh=mean([sum(r['delivered'][i]['absolute_energy'] for i in p) for r in good]),
        mean_delivered_Q_kvarh=mean([sum(r['delivered'][i]['absolute_energy'] for i in q) for r in good]))


def comparison(rows,reference):
    a={(r['forecast'],r['repeat']):r for r in rows};b={(r['forecast'],r['repeat']):r for r in reference}
    if not a or set(a)!=set(b):raise ValueError('Paired execution paths differ')
    return dict(n=len(a),new_recoveries=sum(recovery(a[k]) and not recovery(b[k]) for k in a),
        lost_recoveries=sum(recovery(b[k]) and not recovery(a[k]) for k in a))


def markdown(report):
    lines=['# 旧计划筛查回放：事后开发诊断','',
        '规则固定为：原方案的原概率门槛通过，且既定P/Q或响应预算通过，就保留原方案；否则调用原零动作记录。没有在新保护域内重新搜索。',
        '全部选择先由封存历史、旧计划和原概率审计决定，再读取执行结果。诊断是在已知开发案例上提出，结果复用了旧缓存，不是新增独立实验，也不替代两轮重新优化的失败结果。','',
        '|输入|方法|预算|原概率通过|预算通过|筛查选择|原/筛查/零内部正确|完整支持正确|原/筛查结算元|相对原 新增/丢失|预算值/上限|',
        '|---|---|---|---:|---:|---|---:|---:|---:|---:|---|']
    def n(v):return '—' if v is None else f'{v:.4f}'
    for g in report['groups']:
        d=g['decision'];a=g['original'];s=g['screened'];z=g['no_dfs'];p=g['paired_vs_original']
        budget='; '.join(f"{r['id']}={r['value']:.6g}/{r['limit']:.6g}" for r in d['budgets'])
        lines.append(f"|{g['id']}|{g['base_mode']}|{g['guard']}|{d['original_probability_gate_passed']}|{d['added_budget_passed']}|{d['selected_source']}|{a['internal_correct']}/{s['internal_correct']}/{z['internal_correct']}（各{s['n']}）|{s['full_support_exact_count']}/{s['n']}|{n(a['mean_settled_total_yuan'])}/{n(s['mean_settled_total_yuan'])}|{p['new_recoveries']}/{p['lost_recoveries']}|{budget}|")
    lines += ['', '三库的完整支持覆盖：','']
    for c in report['input_coverage']:
        lines.append(f"- {c['id']}：内部真族在库={c['truth_in_native_bank']}；完整支持真族在库={c['full_support_truth_in_native_bank']}。")
    lines += ['', '拒绝后执行零动作不等于原概率目标仍然满足；机器表单列原零点门槛状态。筛查只使用原数学预算加浮点舍入容差，不新增或扫描阈值。金额、交付及评分全部取自旧独立审计记录。','']
    return '\n'.join(lines)


def replay(run,out):
    reader=Reader();run=Path(run);protocol=reader.read(run/'protocol.json');entries=reader.read(protocol['inputs'])
    if isinstance(entries,dict):entries=entries['inputs']
    if len(entries)!=3 or protocol['pq_rho']!=4. or protocol['response_rms']!=.25:raise ValueError('Use the locked first-round three-input geometry')
    oldaudit=reader.read(protocol['source_audit'])
    if oldaudit.get('status')!='passed':raise ValueError('Passed old independent audit required')
    # Quote records contain only planning checks; execution-record file stays unopened here.
    quotes=reader.lines(oldaudit['quotes']);decisions=[];units={}
    reader.checked(__file__)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    for e in entries:
        tag=e['id'];prior=reader.read(e['prior_seal']);planner=reader.arrays(prior['planner'])
        config=reader.read(prior['resource_case']);units[tag]=config['action_units']
        g=reader.read(run/'cases'/tag/'guard_geometry.json');arrays=reader.arrays(g['arrays'])
        for mode in MODES:
            hits=[q for q in quotes if q['id']==tag and q['forecast']==0 and q['policy']==mode]
            if len(hits)!=1:raise ValueError('Original quote missing or duplicated')
            q=hits[0];gate=bool(q['direct_gate']['feasible'])
            if gate!=bool(q['executed_gate_feasible']):raise ValueError('Original audited gate fields disagree')
            action=planner[mode]
            for name,kind,limit in BUDGETS:
                rows=[]
                for spec in g['guards']:
                    if spec['kind']!=kind:continue
                    if spec['limit']!=(16. if kind=='pq' else .25**2):raise ValueError('Unexpected source geometry normalization')
                    rows.append(dict(id=spec['id'],kind=kind,matrix_key=spec['matrix_key'],**budget_value(action,arrays[spec['matrix_key']],limit)))
                d=screen_decision(gate,rows);key=mode if d['retain_original'] else 'no_dfs'
                decisions.append(dict(id=tag,base_mode=mode,guard=name,**d,budgets=rows,
                    selected_original_strategy=key,selected_action_source=dict(planner=prior['planner'],array_key=key),
                    action_sha256=sha256(np.ascontiguousarray(planner[key],dtype=np.float64).tobytes()).hexdigest(),
                    original_quote_flags={k:q.get(k) for k in ('new_gate_feasible','anchor_fallback','zero_fallback','source','status')},
                    zero_original_gate_passed=bool(q['zero_action_direct_gate']['feasible']),
                    probability_source=dict(ref=oldaudit['quotes'],id=tag,forecast=0,policy=mode)))
    decision_receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),stage='All decisions sealed before opening old execution records in this script',
        design_scope='Post hoc development diagnostic; these cases and earlier results were already known. Historical-only decision inputs do not make this an independent confirmation.',
        source_protocol=reader.ref(run/'protocol.json'),fixed_thresholds=[dict(id=n,kind=k,limit=l) for n,k,l in BUDGETS],
        decisions=decisions,planning_read_refs=[dict(path=p,sha256=h) for p,h in reader.refs.items()])
    (out/'decisions.json').write_text(json.dumps(decision_receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    decision_ref=reader.ref(out/'decisions.json')
    # Only now open the completed old execution outcomes. No new action is executed.
    ids={e['id'] for e in entries};oldrows=[r for r in reader.lines(oldaudit['records']) if r['id'] in ids]
    if len(oldrows)!=384:raise ValueError('Expected all384 old audited rows for the selected inputs')
    if any(r['audit_status'] not in ('passed','recorded_failure') for r in oldrows):raise ValueError('Unverified old outcome')
    lookup={}
    for r in oldrows:lookup.setdefault((r['id'],r['strategy']),[]).append(r)
    groups=[];replayed=[]
    for d in decisions:
        tag=d['id'];old=lookup[(tag,d['base_mode'])];zero=lookup[(tag,'no_dfs')];chosen=lookup[(tag,d['selected_original_strategy'])]
        if len(chosen)!=16 or len(old)!=16 or len(zero)!=16:raise ValueError('Original16 paired path denominator required')
        groups.append(dict(id=tag,base_mode=d['base_mode'],guard=d['guard'],decision=d,
            original=summary(old,units[tag]),screened=summary(chosen,units[tag]),no_dfs=summary(zero,units[tag]),
            paired_vs_original=comparison(chosen,old),paired_vs_no_dfs=comparison(chosen,zero)))
        for r in chosen:
            replayed.append(dict(id=tag,base_mode=d['base_mode'],guard=d['guard'],forecast=r['forecast'],repeat=r['repeat'],
                source_original_strategy=r['strategy'],source=r['source'],audit_status=r['audit_status'],
                evaluation=r['evaluation'],planned_cost=r['planned_cost'],settled_cost=r['settled_cost'],
                data_origin='Reused old independently audited execution record; no new simulation'))
    if not reader.unchanged():raise ValueError('Source changed during replay')
    result=dict(status='complete',scope='Post hoc development diagnostic; old cached execution rows only. Not a new experiment, new solve, current default policy or replacement of either failed reoptimization round.',
        decisions=decision_ref,original_baseline_rows=384,new_executions=0,virtual_replay_rows=len(replayed),
        groups=groups,rows=replayed,input_coverage=[c for c in oldaudit['input_coverage'] if c['id'] in ids],
        sources_unchanged=True,input_refs=[dict(path=p,sha256=h) for p,h in reader.refs.items()])
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    (out/'REPORT.md').write_text(markdown(result),encoding='utf8')
    (out/'complete.json').write_text(json.dumps(dict(status='complete',new_executions=0,sources_unchanged=True,
        decisions=decision_ref,report=reader.ref(out/'report.json')),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=ROOT/'outputs/frozen_rx_guard_execution_20261115_v1');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=replay(a.run,a.out)
    print(json.dumps(dict(status=r['status'],new_executions=0,groups=len(r['groups']),retained=sum(g['decision']['retain_original'] for g in r['groups']),sources_unchanged=r['sources_unchanged']),ensure_ascii=False))


if __name__=='__main__':main()
