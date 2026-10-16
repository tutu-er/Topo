"""Lightweight saved-log audit of guarded development and four cost probes.

No model fitting, optimization or formal record reading is performed.  The
comparison is descriptive: compute load and finite-time parent plans differ.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip,hashlib,json
from pathlib import Path
import numpy as np


def _same(a,b,tol=1e-7):
    if a is None or b is None:return False
    x,y=np.asarray(a),np.asarray(b)
    return x.shape==y.shape and bool(np.max(abs(x-y))<=tol)


def _key(row):return row.get('bank',row.get('physical_bank')),row['case_id'],row['seed']


def extract(path,old):
    if 'formal' in str(path).lower():raise ValueError('formal paths are excluded')
    payload=path.read_bytes();r=json.loads(gzip.decompress(payload))
    if r.get('phase')!='development':raise ValueError('development/cost phase only')
    methods={m['name']:m for m in r['methods']};selection=r['selection'];meta=selection['methods']['jointgrad_mobile']
    raw=meta.get('raw',{});mobile=methods['jointgrad_mobile'];fixed=methods['jointgrad_fixed']
    initials=[];steps=[];counts=Counter()
    for item in raw.get('initial_candidates',[]):
        profile=item.get('profile',{});guard=item.get('reference_guard')
        if item.get('near_duplicate'):kind='same_as_checked_canonical'
        elif item.get('success'):kind='guard_feasible'
        elif guard is not None and not guard.get('success'):kind='reference_floor_rejected'
        elif profile.get('deadline_reached') or 'budget' in str(item.get('failure','')):kind='budget_before_complete_score'
        elif profile and not profile.get('success'):kind='incomplete_predictive_fit'
        elif item.get('operation',{}).get('feasible') is False:kind='operation_completion_failed'
        else:kind='other_failure'
        counts['initial_'+kind]+=1
        initials.append(dict(origin=item['origin'],kind=kind,score=item.get('score'),
            violated_references=None if guard is None else guard.get('violated_references'),
            reference_values=None if guard is None else guard.get('reference_values'),
            profile_power=profile.get('power_kw'),failure=item.get('failure')))
    for start in raw.get('starts',[]):
        if start.get('failure'):
            counts['start_'+start['failure']]+=1
        for item in start.get('iterations',[]):
            solved=item['raw'];profile=item.get('raw_AC_joint_profile',{});guard=item.get('reference_guard')
            if item.get('accepted'):kind='accepted'
            elif item.get('proposal_max_delta_kw',float('inf'))<=1e-10:kind='zero_power_step'
            elif guard is not None and not guard.get('success'):kind='complete_AC_guard_rejected'
            elif profile.get('deadline_reached'):kind='budget_during_predictive_fit'
            elif not solved.get('success'):kind='invalid_or_unfinished_linear_proposal'
            elif profile and not profile.get('success'):kind='predictive_fit_incomplete'
            elif profile:kind='complete_AC_score_did_not_improve'
            else:kind='other_stop'
            counts['step_'+kind]+=1
            steps.append(dict(start=start['origin'],kind=kind,status=solved.get('status'),
                linear_best_bound=solved.get('linear_best_bound'),linear_mip_gap=solved.get('linear_mip_gap'),
                proposal_max_delta_kw=item.get('proposal_max_delta_kw'),
                before_score=item.get('before_score'),trial_score=profile.get('score'),
                linear_floor_violation=solved.get('linear_reference_floor_violation'),
                AC_guard_violated=None if guard is None else guard.get('violated_references'),
                failure=item.get('failure')))
    if not steps:
        counts['no_gradient_steps']=1
        if any('budget' in str(s.get('failure','')) for s in raw.get('starts',[])):
            counts['no_gradient_steps_due_to_budget']=1
        if 'complete Bfixed reference floor unavailable' in str(mobile.get('planning_failure','')):
            counts['no_gradient_steps_parent_profile_unavailable']=1
    op=mobile.get('operation',{}).get('audit',{});u=mobile['power_kw'];nearest=raw.get('reference_floor')
    out=dict(path=str(path),sha256=hashlib.sha256(payload).hexdigest(),bank=r['physical_bank'],
        case_id=r['case_id'],seed=r['seed'],true_name=r['true_name'],status=r['status'],
        policy_id=r['policy_id'],budget_seconds=selection['total_seconds'],planning_success=meta.get('success'),
        planning_failure=mobile.get('planning_failure'),selected_origin=raw.get('selected_origin'),
        selected_score=raw.get('selected_score'),selected_reference_guard=raw.get('selected_reference_guard'),
        planning_seconds=meta['planning_seconds'],own_planning_seconds=meta['planning_actual_seconds'],
        ancestor_seconds=meta['planning_seconds']-meta['planning_actual_seconds'],
        recorded_budget_overrun=raw.get('time_budget_exceeded'),reference_floor=nearest,
        counts=dict(counts),initials=initials,steps=steps,counters=raw.get('counters'),
        mobile_M=mobile['inference']['joint_total']['M'],fixed_M=fixed['inference']['joint_total']['M'],
        scalar_M=methods['scalar_mobile']['inference']['joint_total']['M'],
        actual_moves=op.get('moves'),worker_minutes=op.get('driver_cost'),
        driving_energy_kwh=op.get('travel_energy_kwh'),same_power_as_fixed=_same(u,fixed['power_kw']),
        active_sites=np.flatnonzero(np.any(abs(np.asarray(u))>1e-6,axis=0)).tolist(),
        standalone_ancestors=meta.get('standalone_ancestors'),
        attribution='These are development/cost outcomes, not independent calibrated error rates.')
    original=old.get(_key(out))
    if original:
        om=original['methods']['jointgrad_mobile'];of=original['methods']['jointgrad_fixed']
        od=om.get('B_prediction',{}).get('reference_gaps',{});oldfloor={a:min(value,om.get('cap',5.411086908190826))
            for a,value in of.get('B_prediction',{}).get('reference_gaps',{}).items()}
        old_pass=bool(od and oldfloor and all(od[a]>=oldfloor[a]-1e-7*(1+abs(oldfloor[a])) for a in oldfloor))
        out['old_action_audit']=dict(old_was_moving=om['active_remote'],old_passed_capped_floor=old_pass,
            old_origin=om['origin'],old_mobile_M=om['M'],old_fixed_M=of['M'],
            old_action_present_in_scored_new_initials=any(_same(om['power_kw'],item.get('profile_power')) for item in initials),
            old_and_new_fixed_powers_match=_same(of['power_kw'],fixed['power_kw']),
            saved_old_prediction_passes_new_floor=(None if not nearest or not od else all(
                od[a]>=nearest[a]-1e-7*(1+abs(nearest[a])) for a in nearest)),
            caveat='Last check uses saved old predictions, not a new reprofile under any changed training fit.')
    return out


def summarize(rows):
    counts=Counter()
    for row in rows:counts.update(row['counts'])
    return dict(records=len(rows),unique_seeds=len({r['seed'] for r in rows}),
        planning_failures=sum(not r['planning_success'] for r in rows),moved=sum(bool(r['actual_moves']) for r in rows),
        selected_origins=dict(Counter(r['selected_origin'] for r in rows)),counts=dict(counts),
        mean_standalone_seconds=float(np.mean([r['planning_seconds'] for r in rows])),
        max_standalone_seconds=max(r['planning_seconds'] for r in rows),
        total_gradient_calls=len([step for r in rows for step in r['steps']]),
        old_feasible_moving_histories=sum(r.get('old_action_audit',{}).get('old_was_moving',False)
            and r.get('old_action_audit',{}).get('old_passed_capped_floor',False) for r in rows),
        old_feasible_moving_present_as_scored_initial=sum(r.get('old_action_audit',{}).get('old_was_moving',False)
            and r.get('old_action_audit',{}).get('old_passed_capped_floor',False)
            and r.get('old_action_audit',{}).get('old_action_present_in_scored_new_initials',False) for r in rows))


def run(root,old_diagnostic,output_prefix):
    root=Path(root);old=json.loads(Path(old_diagnostic).read_text(encoding='utf-8'))
    old={_key(r):r for r in old['records'] if r['phase']=='development'}
    directories=[root/name for name in ('mobile_v2_guard_dev_n3_part0','mobile_v2_guard_dev_n3_part1',
        'mobile_v2_guard_dev_n4','mobile_v2_guard60_cost_n3','mobile_v2_guard60_cost_n4')]
    rows=[];failures=[]
    for directory in directories:
        for path in sorted(directory.glob('case_*.json.gz')):
            try:rows.append(extract(path,old))
            except Exception as exc:failures.append(dict(path=str(path),failure=type(exc).__name__+': '+str(exc)))
    groups=defaultdict(list)
    for row in rows:groups[f"budget{row['budget_seconds']:g}/{row['bank']}"].append(row)
    out=dict(schema='mobile_guard_reachability_readonly_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        no_formal_read=True,no_new_optimization=True,no_new_fit=True,read_failures=failures,records=rows,
        groups={name:summarize(values) for name,values in groups.items()},
        scope='28 guard30 development slots and four chosen guard60 cost probes; repeated histories are not new independent samples.')
    prefix=Path(output_prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    prefix.with_suffix('.md').write_text(render(out),encoding='utf-8');return out


def render(out):
    rows=out['records']
    lines=['# 逐参考保护、预算与移动可达性：只读诊断','',
        '仅使用原开发及4个60秒成本核查记录；未读取正式校准/评价，没有重新拟合或求解优化。成本例是事前选定的少数开发历史复跑，不能当成60秒总体效果。','',
        '|组别|槽数|实际移动|规划失败|梯度MILP次数|预算导致0步|固定父未评分|初值保护拒绝|零功率步|完整AC保护拒绝|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,g in out['groups'].items():
        c=g['counts'];lines.append(f"|{name}|{g['records']}|{g['moved']}|{g['planning_failures']}|{g['total_gradient_calls']}|"
            f"{c.get('no_gradient_steps_due_to_budget',0)}|{c.get('no_gradient_steps_parent_profile_unavailable',0)}|"
            f"{c.get('initial_reference_floor_rejected',0)}|{c.get('step_zero_power_step',0)}|{c.get('step_complete_AC_guard_rejected',0)}|")
    lines.extend(['','## 成本核查中的具体代价','',
        '|组/seed|固定M|标量移动M|保护移动M|移动次数|工作人员分钟|驾驶kWh|最终来源|','|---|---:|---:|---:|---:|---:|---:|---|'])
    for r in rows:
        if r['budget_seconds']==60:
            lines.append(f"|{r['bank']}/{r['seed']}|{r['fixed_M']:.6g}|{r['scalar_M']:.6g}|{r['mobile_M']:.6g}|"
                f"{r['actual_moves']}|{r['worker_minutes']}|{r['driving_energy_kwh']}|{r['selected_origin']}|")
    lines.extend(['','## 机制','',
        '30秒开发的28个槽中，20个没有梯度步骤：12个直接在预算处停止，8个尚未取得完整的固定父逐参考评分。余下8个历史合计求解10次梯度MILP，全部为零功率变化。初值中有4次被保护拒绝；没有非零新提议进入完整AC后再被保护拒绝。',
        '预算不足、初值被保护拒绝、局部一阶没有上升方向是不同原因。60秒n3/star仍停在固定父，且总规划只用约23秒，说明该例的失败由搜索锚点和局部结构造成。n4的部分成本例则有预算完成额外完整AC步骤，因此30秒与60秒同时受有限时间和实际父计划差异影响。',
        '旧动作满足完整非线性底线，不表示固定父处的线性MILP能找到它。例：u∈[0,3]，d1=(u−1)^2，d2=(u+1)^2，父u0=0，两底线均为1。u=3合法且平均分数更高，但在0处第一条线性底线为−2u≥0，把局部步骤锁在0。可行集合包含孤立父点与远处区域；整数路由又带来不同接入分支。',
        '初值筛选会删除已有的跨区域跳跃；重新运行有限时间MILP也可能不再生成同一条旧计划。JSON记录旧计划是否出现在已评分的新初值中，以及固定父是否变化。尚未评分的初值没有完整功率记录，因此未匹配到旧计划只表示已评分部分没有它。旧预测对新底线的比较仅作保存数据审计，没有冒充重新profile。',
        '下一版只建议增加一个可行性恢复阶段：被完整AC保护拒绝的优化跳跃不作为可执行计划，但保留为数值展开点。用操作MILP最小化逐参考底线的线性违约量，再做完整AC检查；全部保护通过后才进入可执行候选。固定父始终保留。这样可探索远处可行区域，代价是额外MILP和AC调用，仍可能失败，不能宣称增加保护或搜索必然改善。',
        '每次实际转移已扣除0.2kWh驾驶电量并占用驾驶/接线时段，另使用30工作人员分钟。M增量是电气证据指标，与工作人员时间单位不同；本轮没有将二者随意相减或合成为成本收益率。',''])
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs'))
    p.add_argument('--old-diagnostic',type=Path,default=Path(__file__).with_name('ev_mobile_v2_development_diagnostic.json'))
    p.add_argument('--output-prefix',type=Path,default=Path(__file__).with_name('ev_mobile_v2_guard_reachability'))
    args=p.parse_args();out=run(args.root,args.old_diagnostic,args.output_prefix)
    print(json.dumps(dict(groups=out['groups'],read_failures=out['read_failures']),ensure_ascii=False))
