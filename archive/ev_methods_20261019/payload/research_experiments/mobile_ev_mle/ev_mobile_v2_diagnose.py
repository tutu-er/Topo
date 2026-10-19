"""Read-only development diagnostics; never used by the EV selector.

Only saved pilot/development records are accepted.  This script neither fits
models nor proposes actions. Actual future outcomes are used solely to audit
the already selected plans, not to select policies or tune their constants.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


METHODS=('square_fixed','square_mobile','scalar_fixed','scalar_mobile',
    'cap_fixed','cap_mobile','jointgrad_fixed','jointgrad_mobile')


def _number(value):
    return None if value is None or not np.isfinite(value) else float(value)


def _physical_matches(power,methods,tol=1e-7):
    if power is None:return []
    p=np.asarray(power,float)
    return [name for name,m in methods.items() if np.asarray(m['power_kw']).shape==p.shape
        and np.max(abs(np.asarray(m['power_kw'])-p))<=tol]


def _compensation(profile):
    if not profile or not profile.get('success'):return None
    names=[r['a'] for r in profile['references']];F={};J={};delta_own=[];delta_rival=[];errors=[]
    optimality=[];active=[]
    for row in profile['references']:
        a=row['a'];own=row['fits'][a];da=own['feasible_witness']['loss']-own['loss'];delta_own.append(da)
        for b,fit in row['fits'].items():
            diagnostic=(fit.get('optimizer') or {}).get('diagnostics',{})
            if (fit.get('optimizer') or {}).get('optimality') is not None:
                optimality.append(fit['optimizer']['optimality'])
            if diagnostic.get('active_bound_count') is not None:active.append(diagnostic['active_bound_count'])
            if a==b:continue
            db=fit['feasible_witness']['loss']-fit['loss'];delta_rival.append(db)
            f=.5*(fit['feasible_witness']['loss']-own['feasible_witness']['loss'])
            j=.5*(fit['loss']-own['loss']);F[a,b]=f;J[a,b]=j
            errors.append(abs(j-(f+.5*(da-db))))
    return dict(fixed_training_theta_mean_nearest_gap=float(np.mean([min(F[a,b] for b in names if a!=b) for a in names])),
        shared_line_mean_nearest_gap=float(np.mean([min(J[a,b] for b in names if a!=b) for a in names])),
        mean_reference_loss_reduction=float(np.mean(delta_own)),
        mean_competitor_loss_reduction=float(np.mean(delta_rival)),
        maximum_pair_identity_error=max(errors,default=0.),
        max_optimizer_optimality=max(optimality,default=None),
        max_active_bound_count=max(active,default=None),
        fitted_models=len(names)**2,
        scope='same synthetic future and actual H; attained fixed witness minus attained shared-line loss')


def extract(path):
    source=path.read_bytes();r=json.loads(gzip.decompress(source))
    if r.get('phase') not in ('pilot','development'):
        raise ValueError('only declared pilot/development records are accepted')
    methods={m['name']:m for m in r.get('methods',[])};out={}
    scored=[]
    for name in ('jointgrad_fixed','jointgrad_mobile'):
        raw=r.get('selection',{}).get('methods',{}).get(name,{}).get('raw',{})
        for initial in raw.get('initial_candidates',[]):
            if initial.get('profile',{}).get('success'):scored.append(initial['profile'])
        for start in raw.get('starts',[]):
            for step in start.get('iterations',[]):
                if step.get('raw_AC_joint_profile',{}).get('success'):scored.append(step['raw_AC_joint_profile'])
    predicted={}
    for profile in scored:
        for name in _physical_matches(profile.get('power_kw'),methods):
            predicted.setdefault(name,dict(score=profile['score'],compensation=_compensation(profile),
                reference_gaps={row['a']:row['half_gap'] for row in profile.get('aggregation',{}).get('nearest',[])}))
    for name in METHODS:
        if name not in methods:
            out[name]=dict(available=False);continue
        m=methods[name];meta=r.get('selection',{}).get('methods',{}).get(name,{})
        raw=meta.get('raw',{});mode=m.get('inference',{}).get('joint_total',{})
        u=np.asarray(m['power_kw']);op=m.get('operation',{}).get('audit',{})
        steps=[step for start in raw.get('starts',[]) for step in start.get('iterations',[])]
        failures=[step.get('failure') for step in steps if step.get('failure')]
        cap=raw.get('cap');cap_selected=raw.get('selected_score')
        if cap is None:cap=r.get('selection',{}).get('task_cap')
        zero_power=sum(step.get('proposal_max_delta_kw',float('inf'))<=1e-10 for step in steps)
        cap_saturated=bool(name.startswith('cap') and cap is not None and cap_selected is not None
            and abs(cap_selected-cap)<=1e-7)
        initial_rows=[]
        initial_profiles=[]
        for initial in raw.get('initial_candidates',[]):
            matches=_physical_matches(initial.get('profile',{}).get('power_kw'),methods)
            if initial.get('profile',{}).get('success') and 'score' in initial['profile']:
                initial_profiles.append(initial['profile'])
            initial_rows.append(dict(origin=initial.get('origin'),score=initial.get('score'),methods=matches,
                success=initial.get('success',False),failure=initial.get('failure'),
                near_duplicate=initial.get('near_duplicate',False)))
        selected_matches_initial=any(np.max(abs(np.asarray(p['power_kw'])-u))<=1e-7 for p in initial_profiles)
        best_initial=max(initial_profiles,key=lambda p:p['score']) if initial_profiles else None
        best_matches=[] if best_initial is None else _physical_matches(best_initial['power_kw'],methods)
        initial_actual=None if not best_matches else methods[best_matches[0]].get('inference',{}).get('joint_total',{}).get('M')
        out[name]=dict(available=True,planning_success=meta.get('success',False),
            planning_failure=m.get('planning_failure'),operation_success=m.get('operation',{}).get('success',False),
            inference_complete=mode.get('complete',False),M=_number(mode.get('M')),
            loss_by_candidate=mode.get('losses'),origin=raw.get('selected_origin'),
            planning_seconds=_number(m.get('planning_seconds')),total_seconds=_number(mode.get('total_seconds')),
            score=_number(raw.get('selected_score')),same_square_fixed='square_fixed' in _physical_matches(u,methods),
            moves=op.get('moves'),driver_minutes=op.get('driver_cost'),
            active_sites=np.flatnonzero(np.any(abs(u)>1e-6,axis=0)).tolist(),
            active_remote=bool(u.shape[1]>1 and np.any(abs(u[:,1:])>1e-6)),
            power_kw=u.tolist(),cap_saturated=cap_saturated,cap=_number(cap),
            initial_best_score=_number(raw.get('initial_best_score')),initial_candidates=initial_rows,
            gradient_calls=raw.get('gradient_MILP_calls',0),gradient_accepted=raw.get('gradient_MILP_accepted',0),
            final_origin_gradient=raw.get('selected_origin')=='joint_gradient_MILP',
            final_gradient_physically_new=bool(raw.get('selected_origin')=='joint_gradient_MILP' and not selected_matches_initial),
            best_B_initial_matches=best_matches,
            power_max_delta_from_best_B_initial=(None if best_initial is None
                else float(np.max(abs(u-np.asarray(best_initial['power_kw']))))),
            B_score_gain_from_best_initial=(None if raw.get('selected_score') is None
                or raw.get('initial_best_score') is None else raw['selected_score']-raw['initial_best_score']),
            actual_M_gain_over_best_B_initial=(None if initial_actual is None or mode.get('M') is None
                else mode['M']-initial_actual),
            zero_power_steps=zero_power,step_failures=failures,
            start_failures=[s['failure'] for s in raw.get('starts',[]) if s.get('failure')],
            budget_exceeded=bool(raw.get('time_budget_exceeded')),counters=raw.get('counters'),
            B_prediction=predicted.get(name),parameter_error=m.get('parameter_error'))
    comparisons={}
    for family in ('square','scalar','cap','jointgrad'):
        f=out[family+'_fixed'];m=out[family+'_mobile']
        if not f.get('available') or not m.get('available'):continue
        actual=None if f['M'] is None or m['M'] is None else m['M']-f['M']
        pf=f.get('B_prediction');pm=m.get('B_prediction')
        prediction=None if not pf or not pm else pm['score']-pf['score']
        truth=r.get('true_name');true_prediction=(None if not pf or not pm or truth not in pf['reference_gaps']
            or truth not in pm['reference_gaps'] else pm['reference_gaps'][truth]-pf['reference_gaps'][truth])
        comparisons[family]=dict(actual_M_gain=actual,B_predicted_gain=prediction,
            B_predicted_true_reference_gain=true_prediction,
            actual_moves=m['moves'],active_remote=m['active_remote'],
            same_power=bool(np.max(abs(np.asarray(m['power_kw'])-np.asarray(f['power_kw'])))<=1e-7),
            prediction_order_reversal=bool(prediction is not None and actual is not None
                and prediction>1e-7 and actual<-1e-7),
            predicted_mean_sacrifices_true_reference=bool(prediction is not None and prediction>1e-7
                and true_prediction is not None and true_prediction<-1e-7),
            true_reference_prediction_order_reversal=bool(true_prediction is not None and true_prediction>1e-7
                and actual is not None and actual<-1e-7))
    return dict(path=str(path),sha256=hashlib.sha256(source).hexdigest(),bytes=len(source),
        phase=r['phase'],bank=r.get('physical_bank'),case_id=r['case_id'],index=r['index'],seed=r['seed'],
        training_points=r['config']['training_points'],true_name=r.get('true_name'),status=r.get('status'),
        policy_id=r.get('policy_id'),methods=out,mobile_fixed=comparisons)


def summarize(records):
    result=dict(records=len(records),unique_seeds=len({r['seed'] for r in records}),methods={},mobile_fixed={})
    for name in METHODS:
        rows=[r['methods'][name] for r in records];good=[m for m in rows if m.get('available')]
        observed_M=[m['M'] for m in good if m['M'] is not None]
        compensation=[m['B_prediction']['compensation'] for m in good if m.get('B_prediction')
            and m['B_prediction'].get('compensation')]
        result['methods'][name]=dict(available=len(good),missing=len(rows)-len(good),
            planning_failures=sum(not m['planning_success'] for m in good),
            operation_failures=sum(not m['operation_success'] for m in good),
            inference_failures=sum(not m['inference_complete'] for m in good),
            raw_correct=sum(m['M'] is not None and m['M']>0 for m in good),
            mean_M=float(np.mean(observed_M)) if observed_M else None,
            actual_moved=sum(bool(m['moves']) for m in good),active_remote=sum(m['active_remote'] for m in good),
            same_square_fixed=sum(m['same_square_fixed'] for m in good),
            capped_saturated=sum(m['cap_saturated'] for m in good),
            gradient_calls=sum(m['gradient_calls'] for m in good),gradient_accepted=sum(m['gradient_accepted'] for m in good),
            final_origin_gradient=sum(m['final_origin_gradient'] for m in good),
            final_gradient_physically_new=sum(m['final_gradient_physically_new'] for m in good),
            final_gradient_actual_gains=[m['actual_M_gain_over_best_B_initial'] for m in good
                if m['final_gradient_physically_new']],
            zero_power_steps=sum(m['zero_power_steps'] for m in good),
            step_failures=dict(Counter(reason for m in good for reason in m['step_failures'])),
            start_failures=dict(Counter(reason for m in good for reason in m['start_failures'])),
            origins=dict(Counter(m['origin'] for m in good)),
            compensation_profiles=len(compensation),
            mean_fixed_witness_predictive_score=(float(np.mean([p['fixed_training_theta_mean_nearest_gap'] for p in compensation]))
                if compensation else None),
            mean_shared_line_predictive_score=(float(np.mean([p['shared_line_mean_nearest_gap'] for p in compensation]))
                if compensation else None),
            mean_reference_loss_reduction=(float(np.mean([p['mean_reference_loss_reduction'] for p in compensation]))
                if compensation else None),
            mean_competitor_loss_reduction=(float(np.mean([p['mean_competitor_loss_reduction'] for p in compensation]))
                if compensation else None),
            max_compensation_identity_error=max([p['maximum_pair_identity_error'] for p in compensation],default=None),
            mean_planning_seconds=float(np.mean([m['planning_seconds'] for m in good])) if good else None)
    for family in ('square','scalar','cap','jointgrad'):
        rows=[r['mobile_fixed'][family] for r in records if family in r['mobile_fixed']]
        gains=[m['actual_M_gain'] for m in rows if m['actual_M_gain'] is not None]
        result['mobile_fixed'][family]=dict(paired=len(gains),win=sum(g>1e-7 for g in gains),
            tie=sum(abs(g)<=1e-7 for g in gains),loss=sum(g<-1e-7 for g in gains),
            mean_M_gain=float(np.mean(gains)) if gains else None,
            B_prediction_reversals=sum(m['prediction_order_reversal'] for m in rows),
            mean_sacrifices_true_reference=sum(m['predicted_mean_sacrifices_true_reference'] for m in rows),
            true_reference_prediction_reversals=sum(m['true_reference_prediction_order_reversal'] for m in rows),
            prediction_comparable=sum(m['B_predicted_gain'] is not None for m in rows))
    return result


def screen_reference_floors(records,*,capped=False):
    """Audit existing actions only; no new plan, truth-based threshold or fit."""
    rows=[]
    for row in records:
        if row['phase']!='development':continue
        fixed=row['methods']['jointgrad_fixed'];mobile=row['methods']['jointgrad_mobile']
        fp=fixed.get('B_prediction');mp=mobile.get('B_prediction')
        if not fp or not mp:
            rows.append(dict(seed=row['seed'],case_id=row['case_id'],available=False));continue
        cap=row['methods']['cap_fixed']['cap'];old=fp['reference_gaps'];new=mp['reference_gaps']
        floor={a:min(old[a],cap) if capped else old[a] for a in old}
        slacks={a:new[a]-floor[a] for a in floor}
        passed=all(slacks[a]>=-1e-7*(1+abs(floor[a])) for a in floor)
        rows.append(dict(available=True,seed=row['seed'],case_id=row['case_id'],bank=row['bank'],
            true_name=row['true_name'],passed=passed,active_remote=mobile['active_remote'],
            actual_M_gain=row['mobile_fixed']['jointgrad']['actual_M_gain'],
            prediction_order_reversal=row['mobile_fixed']['jointgrad']['prediction_order_reversal'],
            declared_cap=cap,reference_floors=floor,reference_slacks=slacks))
    def counts(group):
        gains=[r['actual_M_gain'] for r in group if r['actual_M_gain'] is not None]
        return dict(records=len(group),active_remote=sum(r['active_remote'] for r in group),
            M_wins=sum(v>1e-7 for v in gains),M_ties=sum(abs(v)<=1e-7 for v in gains),M_losses=sum(v<-1e-7 for v in gains),
            summed_M_gain=sum(gains),prediction_reversals=sum(r['prediction_order_reversal'] for r in group))
    available=[r for r in rows if r['available']]
    return dict(capped=capped,tolerance='1e-7*(1+abs(reference_floor))',
        scope='post-hoc feasibility of existing development B actions; reoptimized policy outcomes unknown',
        total=counts(available),passed=counts([r for r in available if r['passed']]),
        blocked=counts([r for r in available if not r['passed']]),
        unavailable=sum(not r['available'] for r in rows),records=rows)


def run(directories,output_prefix):
    paths=sorted({p for directory in directories for p in Path(directory).glob('case_*.json.gz')})
    records=[];read_failures=[]
    for path in paths:
        try:records.append(extract(path))
        except Exception as exc:read_failures.append(dict(path=str(path),failure=type(exc).__name__+': '+str(exc)))
    groups=defaultdict(list)
    for row in records:groups[row['phase']+'/'+row['bank']].append(row)
    result=dict(schema='mobile_v2_readonly_development_diagnostic_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),input_files=len(paths),read_failures=read_failures,
        total=summarize(records),groups={name:summarize(rows) for name,rows in groups.items()},records=records,
        interpretation=['Pilot and development are separate; paired training conditions do not create independent histories.',
            'Actual M uses truth only offline; it is a white SSE difference, not calibrated classification.',
            'A gradient accepted on a lower-scoring start is not necessarily the final selected action.',
            'An inherited scalar action remains a scalar contribution even if B accepts its predicted score.',
            'B fit optimality is a numerical diagnostic, not a global KKT or global AC certificate.'])
    result['reference_floor_screens']={kind:screen_reference_floors(records,capped=kind=='capped')
        for kind in ('strict','capped')}
    output_prefix=Path(output_prefix);output_prefix.parent.mkdir(parents=True,exist_ok=True)
    output_prefix.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 移动EV开发阶段只读诊断','',f'快照含{len(records)}条记录；读取失败{len(read_failures)}条。',
        '以下只比较已经选择的动作；未使用评价未来数据调参。M为错误候选最小SSE减真候选SSE。','']
    for name,group in result['groups'].items():
        lines.extend([f'## {name}', '',f"记录{group['records']}槽，基础种子{group['unique_seeds']}个。",'',
            '|方法|原始正确|实际移动|最终为梯度新动作|梯度MILP/接受|零功率步|截断饱和|',
            '|---|---:|---:|---:|---:|---:|---:|'])
        for method,m in group['methods'].items():
            lines.append(f"|{method}|{m['raw_correct']}/{m['available']}|{m['actual_moved']}|{m['final_origin_gradient']}|{m['gradient_calls']}/{m['gradient_accepted']}|{m['zero_power_steps']}|{m['capped_saturated']}|")
        lines.extend(['','|移动相对固定|M胜/平/负|平均M增量|B预测反序/可比|','|---|---:|---:|---:|'])
        for family,m in group['mobile_fixed'].items():
            lines.append(f"|{family}|{m['win']}/{m['tie']}/{m['loss']}|{m['mean_M_gain']:.5g}|{m['B_prediction_reversals']}/{m['prediction_comparable']}|")
        lines.append('')
    lines.extend(['## 平均目标与逐参考目标的取舍','',
        '|开发场景/seed|平均B增量|真参考预测增量|实际M增量|','|---|---:|---:|---:|'])
    for row in records:
        comparison=row['mobile_fixed'].get('jointgrad',{})
        if row['phase']=='development' and comparison.get('prediction_order_reversal'):
            lines.append(f"|{row['case_id']}/{row['seed']}|{comparison['B_predicted_gain']:.6g}|"
                f"{comparison['B_predicted_true_reference_gain']:.6g}|{comparison['actual_M_gain']:.6g}|")
    lines.extend(['','## 原已选动作的逐参考保护检查','',
        '只检查旧动作能否通过新增条件，未重新优化；以下计数不是新策略的评价结果。阈值沿用既定5.411规划尺度，未用真值调节。',
        '|规则|通过槽/移动槽|通过M胜/平/负|阻止正移动/负移动|通过累计M增量|',
        '|---|---:|---:|---:|---:|'])
    for kind,screen in result['reference_floor_screens'].items():
        p=screen['passed'];b=screen['blocked']
        lines.append(f"|{kind}|{p['records']}/{p['active_remote']}|{p['M_wins']}/{p['M_ties']}/{p['M_losses']}|"
            f"{b['M_wins']}/{b['M_losses']}|{p['summed_M_gain']:.6g}|")
    lines.extend(['## 机制和下一步','',
        '截断目标达到所有任务的上限后对进一步改善不再给价格。当前严格改善接受规则保留最先达到同分的平方父计划，因而A常不改变动作。',
        'B的零功率步通常是非零价格在当前操作顶点没有一阶上升方向，应与价格全零、预算耗尽区分。固定参数的凸平方距离也会有这个现象：max (u+0.1)^2，u∈[-1,1]，从u=-1开始的梯度LP仍选-1，而全局较好点为+1。',
        'B的参数补偿分解为g_J,ab=g_F,ab+(δ_a−δ_b)/2，其中δ为同一预测数据下固定合法见证损失减联合取得损失。全部对的恒等式误差及边界/收敛信息保存在JSON中。',
        '平均分数提高可以伴随某个参考模型的分数下降。表中真参考列仅用于离线解释；规划器没有真标签。负值说明平均目标本来就允许牺牲这类模型；正值而实际M下降，才是该参考预测与实际排序反序。',
        '原版开发诊断支持给B增加逐参考保护。严格要求d_a(u)>=d_a(u0)会阻止较多正移动；较适度的工作条件为d_a(u)>=min(d_a(u0),tau)，tau沿用既定规划尺度。局部MILP给每个参考的增量eta_a设置相应下界，并保留全部活动竞争分支；每次用完整AC检查所有参考的预测值。当前合法父计划仍可行。这个改进防止平均值掩盖单模型损失，但不保证真实M单调。',
        '本报告保留原版32条开发/pilot记录的事实。其后新增的可选capped_reference_floor版本已另行实现，默认关闭；效果须重新生成计划、冻结并独立校准评价，不能用本表的事后筛选计数代替。反向价格MILP有限跳跃可检查顶点的一阶停滞，作为次级搜索备选；没有全局最优保证。',
        'A若保留，可把未截断距离仅作同分次级目标；这主要改善饱和后的并列选择，不能把继承的平方或标量优势称为截断指标本身的贡献。',''])
    output_prefix.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,action='append',required=True)
    parser.add_argument('--output-prefix',type=Path,required=True);args=parser.parse_args()
    result=run(args.directory,args.output_prefix)
    print(json.dumps(dict(records=result['total']['records'],read_failures=result['read_failures'],
        groups={k:v['records'] for k,v in result['groups'].items()}),ensure_ascii=False))
