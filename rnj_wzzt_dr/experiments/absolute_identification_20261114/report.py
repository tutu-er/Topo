"""Aggregate locked absolute-recovery experiments, preserving failed denominators."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from ... import io

LABELS=dict(no_dfs='无服务',old_gamma2='旧增量2',nominal='单目标名义恢复',calibrated='单目标均值修正',
    symmetric='多目标绝对恢复',symmetric_guard='多目标＋方向保护',symmetric_full='全候选绝对恢复',symmetric_full_guard='全候选＋方向保护',
    multiplicative_absolute='乘性校正绝对恢复',capped_absolute='限幅校正绝对恢复',
    weighted_nominal='按难度分配误判额度',weighted_capped='按难度分配额度＋限幅校正')

def label(strategy):
    return LABELS.get(strategy,strategy) if not strategy.endswith('_rotation') else label(strategy[:-9])+'同费轮换'

def mean(values):
    values=[float(v) for v in values if v is not None and np.isfinite(v)]
    return None if not values else float(np.mean(values))

def cluster_interval(values,groups,seed):
    if not len(values):return None
    keys=sorted(set(groups));sums=np.array([sum(v for v,g in zip(values,groups) if g==key) for key in keys])
    counts=np.array([sum(g==key for g in groups) for key in keys])
    draws=np.random.default_rng(seed).integers(0,len(keys),(2000,len(keys)))
    estimates=sums[draws].sum(1)/counts[draws].sum(1)
    return np.quantile(estimates,[.025,.975]).tolist()

def day_interval(paired_differences,seed):
    """Resample complete paired days together, conditional on one history."""
    values=np.asarray(paired_differences,float)
    if not len(values):return None
    indices=np.random.default_rng(seed).integers(0,len(values),(2000,len(values)))
    return np.quantile(values[indices].mean(1),[.025,.975]).tolist()

def compare(rows,entries,days,a,b,physics,seed):
    table={(r['id'],r['day'],r['strategy']):r for r in rows if r['physics']==physics}
    diff=[];weights=[];weight_groups=[];recovery_groups=[]
    completed_pairs=0;baseline_correct=0;actual_correct=0
    voltage_new=voltage_removed=voltage_cells_change=0
    for e in entries:
        for day in range(days):
            x=table.get((e['id'],day,a),{});y=table.get((e['id'],day,b),{})
            xc=x.get('status')=='complete' and x.get('native_map_correct',False)
            yc=y.get('status')=='complete' and y.get('native_map_correct',False)
            diff.append(int(xc)-int(yc));recovery_groups.append(e['physical_history_group'])
            actual_correct+=int(xc);baseline_correct+=int(yc)
            if x.get('status')==y.get('status')=='complete':
                completed_pairs+=1
                xv=x.get('voltage_cells_outside_screen',0);yv=y.get('voltage_cells_outside_screen',0)
                voltage_new+=xv>0 and yv==0;voltage_removed+=yv>0 and xv==0;voltage_cells_change+=xv-yv
                if x.get('p_true') is not None and y.get('p_true') is not None:
                    weights.append(x['p_true']-y['p_true']);weight_groups.append(e['physical_history_group'])
    recovery=np.array(diff);weight=np.array(weights);n=len(diff)
    gain=int(np.sum(recovery>0));loss=int(np.sum(recovery<0))
    a_ci=binomtest(gain,n).proportion_ci(.975,method='exact');b_ci=binomtest(loss,n).proportion_ci(.975,method='exact')
    radius=float(np.sqrt(2*np.log(40)/n))
    return dict(strategy=a,reference=b,planned_pairs=n,complete_execution_pairs=completed_pairs,
        actual_correct=actual_correct,reference_correct=baseline_correct,new_recoveries=gain,lost_recoveries=loss,
        native_recovery_gain=float(recovery.mean()),
        recovery_gain_binomial_working_model95=[float(a_ci.low-b_ci.high),float(a_ci.high-b_ci.low)],
        recovery_gain_independent_bounded95=[max(-1.,float(recovery.mean())-radius),min(1.,float(recovery.mean())+radius)],
        recovery_gain_history_cluster_bootstrap95=cluster_interval(diff,recovery_groups,seed),
        paired_binomial_working_model_p=1. if gain+loss==0 else float(binomtest(gain,gain+loss,.5).pvalue),
        truth_probability_pairs=len(weight),mean_true_weight_gain=mean(weight),
        paired_voltage_complete_days=completed_pairs,new_voltage_violation_days=int(voltage_new),
        removed_voltage_violation_days=int(voltage_removed),voltage_violation_cells_change=int(voltage_cells_change),
        true_weight_gain_history_cluster_bootstrap95=cluster_interval(weights,weight_groups,seed+1),
        weight_improved_days=int(np.sum(weight>1e-12)),weight_declined_days=int(np.sum(weight< -1e-12)),
        worst_weight_change=None if not len(weight) else float(weight.min()),
        inference_scope='Exploratory unadjusted comparisons. The bounded Hoeffding interval permits different success probabilities across fixed histories, assuming independent future paths. Physical-history cluster bootstrap is approximate and can cancel within groups. Clopper-Pearson difference and paired binomial p-value are working-model diagnostics, not universal exact inference for the heterogeneous panel.')

def aggregate(rows,entries,days,strategies,physics,seed):
    ids={e['id'] for e in entries};own=[r for r in rows if r['id'] in ids and r['physics']==physics]
    result=dict(histories=len(entries),physical_history_groups=len({e['physical_history_group'] for e in entries}),
        planned_days=len(entries)*days,truth_containing_histories=sum(e['truth_covered'] for e in entries),strategies={},comparisons={})
    for strategy in strategies:
        attempted=[r for r in own if r['strategy']==strategy];complete=[r for r in attempted if r['status']=='complete']
        covered=[r for r in complete if r.get('p_true') is not None]
        summary=dict(planned_days=len(entries)*days,recorded_rows=len(attempted),complete_rows=len(complete),
            failed_or_missing_rows=len(entries)*days-len(complete),native_correct=sum(r['native_map_correct'] for r in complete),
            representative_correct=sum(r['map_correct'] for r in complete),
            baseline_historical_correct_histories=sum(e['native_correct'] for e in entries),
            mean_true_weight=mean([r['p_true'] for r in covered]),true_weight_denominator=len(covered),
            mean_target_weight=mean([r['p_target'] for r in complete]),
            true_weight_at_least90=sum(r['true_pass90'] for r in covered),
            mean_planned_fee=mean([r['planned_fee_yuan'] for r in attempted]),mean_settled_fee=mean([r['settled_fee_yuan'] for r in attempted]),
            mean_delivered_abs_kwh=mean([r['delivered_abs_kwh'] for r in attempted]),zero_fallbacks=sum(r['zero_fallback'] for r in attempted),
            zero_cost_days=sum(r['planned_fee_yuan']<=1e-9 for r in attempted),
            nonzero_cost_days=sum(r['planned_fee_yuan']>1e-9 for r in attempted),
            voltage_violation_cells=sum(r.get('voltage_cells_outside_screen',0) for r in complete),
            voltage_violation_days=sum(r.get('voltage_cells_outside_screen',0)>0 for r in complete),
            device_violations=sum(r.get('device') is not None and not r['device']['passed'] for r in complete),rejection={})
        for factor in ('1.25','2.0','4.0'):
            rr=[r['rejection'][factor] for r in complete]
            summary['rejection'][factor]=dict(correct=sum(r['correct'] for r in rr),wrong=sum(r['wrong'] for r in rr),
                rejected=sum(r['rejected'] for r in rr),failed_or_missing=len(entries)*days-len(rr),
                mean_true_weight=mean([r['p_true'] for r in rr]),mean_target_weight=mean([r['p_target'] for r in rr]))
        result['strategies'][strategy]=summary
        if strategy!='no_dfs':
            result['comparisons'][strategy+'_vs_no_dfs']=compare(own,entries,days,strategy,'no_dfs',physics,seed);seed+=2
        if strategy+'_rotation' in strategies:
            result['comparisons'][strategy+'_vs_rotation']=compare(own,entries,days,strategy,strategy+'_rotation',physics,seed);seed+=2
    return result

def summarize_solver(records,entries,modes,days,rows):
    ids={e['id'] for e in entries};result={}
    for mode in modes:
        own=[p for (identity,day,name),p in records.items() if identity in ids and name==mode]
        feasible=[p for p in own if p.get('feasible')];positive=[p for p in feasible if p.get('cost_yuan',0)>1e-9]
        result[mode]=dict(planned_quotes=len(entries)*days,recorded_quotes=len(own),feasible_quotes=len(feasible),
            statuses=dict(Counter(p.get('status','unknown') for p in own)),sources=dict(Counter(p.get('source','unavailable') for p in own)),
            zero_sufficient=sum(p.get('source')=='zero' for p in own),nonzero_feasible=len(positive),
            failed_or_missing_quotes=len(entries)*days-len(feasible),zero_fallbacks=sum(p.get('zero_fallback',False) for p in own),
            infeasibility_proved=sum(p.get('infeasibility_proved',False) for p in own),
            quote_exceptions=sum(p.get('status')=='exception' for p in own),
            unresolved_quotes=sum(p.get('status')=='unresolved' for p in own),
            initial_retained=sum(p.get('source') in ('verified_initial_retained','supporting_LP_initial_point_retained') for p in own),
            native_improved_initial=sum(p.get('source') in ('native_improved_initial','native_QCQP_improved_initial_point') for p in own),
            native_found_without_verified_initial=sum(p.get('source') in ('native_found','native_QCQP_found_feasible') for p in own),
            initial_improvement_yuan=mean([p.get('improvement_from_verified_initial_yuan') for p in own]),
            nonzero_bound_closed=sum(p.get('global_optimality_verified',False) for p in positive),
            all_bound_closed=sum(p.get('global_optimality_verified',False) for p in feasible),
            mean_relative_gap_nonzero=mean([p.get('relative_gap') for p in positive]),
            mean_valid_lower_bound_nonzero=mean([p.get('lower_bound_yuan') for p in positive]),
            mean_feasible_upper_bound_nonzero=mean([p.get('cost_yuan') for p in positive]),
            withdrawn_bounds=sum(p.get('lower_bound_withdrawn',False) for p in own),
            time_limits=sum(p.get('native_status')==9 for p in own),
            mean_driver_seconds=mean([p.get('driver_planning_seconds') for p in own]),
            mean_shared_seed_charged_seconds=mean([p.get('shared_seed_seconds',0) for p in own]),
            mean_total_charged_seconds=mean([p.get('driver_planning_seconds',0)+p.get('shared_seed_seconds',0) for p in own]),
            maximum_total_charged_seconds=max([p.get('driver_planning_seconds',0)+p.get('shared_seed_seconds',0) for p in own],default=None),
            same_fee_rotation_failures=sum(not p.get('rotation_same_fee',False) for p in own),
            certificate_bank_scopes=dict(Counter(p.get('constrained_bank','no_probability_certificate') for p in own)),
            scope='Valid bounds apply to the particular locked QCQP, not to the unapproximated identification probability or physical feasibility.')
    return result

def verify_seals(run,p,entries):
    errors=[];count=0;source_count=0;cases=set()
    if io.ref(run/'protocol.json')['sha256']!=io.read(run/'lock.json')['protocol']['sha256']:errors.append('protocol_hash')
    for relative,expected in p['sources'].items():
        snapshot=run/'source_snapshot'/relative;source_count+=1
        if not snapshot.exists() or io.ref(snapshot)['sha256']!=expected:errors.append('snapshot:'+relative)
    for e in entries:
        if io.ref(e['score']['path'])['sha256']!=e['score']['sha256']:errors.append('score:'+e['id'])
        for day in range(p['days']):
            d=run/'days'/e['id']/f'day{day:03d}'
            if (d/'seal.json').exists():
                seal=io.read(d/'seal.json');count+=1
                for key in ('planner','plans','protocol'):
                    if io.ref(seal[key]['path'])['sha256']!=seal[key]['sha256']:errors.append('seal:'+e['id']+':'+str(day)+':'+key)
                if seal.get('shared_points') and io.ref(seal['shared_points']['path'])['sha256']!=seal['shared_points']['sha256']:errors.append('seed_seal:'+e['id']+':'+str(day))
    return dict(passed=not errors,source_snapshots=source_count,policy_seals=count,errors=errors,
        scope='Protocol, frozen scores, source snapshots and sealed plans. No new optimization or AC runs.')

def report(run,destination):
    if not (run/'complete.json').exists():raise ValueError('Wait for all planned tasks, including failures, before reporting')
    p=io.read(run/'protocol.json');entries=io.read(run/'inputs.json');completion=io.read(run/'complete.json')
    rows=[r for path in sorted((run/'days').glob('*/day*/results.json')) for r in io.read(path)]
    plans={(path.parent.parent.name,int(path.parent.name[3:]),name):value for path in sorted((run/'days').glob('*/day*/plans.json')) for name,value in io.read(path).items()}
    strategies=['no_dfs']+[s for mode in p['modes'] for s in (mode,mode+'_rotation')]
    result=dict(source=io.ref(__file__),protocol=io.ref(run/'protocol.json'),inputs=io.ref(run/'inputs.json'),
        phase=p['phase'],completion=completion,planned_histories=len(entries),planned_independent_future_paths=len(entries)*p['days'],
        physics_copies_per_path=2,planned_strategy_rows=2*len(entries)*p['days']*len(strategies),recorded_strategy_rows=len(rows),
        failed_execution_rows=sum(r['status']!='complete' for r in rows),failed_tasks=len(completion['failed']),
        planned_optimization_quotes=len(entries)*p['days']*len(p['modes']),
        quote_exceptions=sum(v.get('status')=='exception' for v in plans.values()),
        sample_design='stratified_repeat44' if Path(p['input_ref']['path']).name=='repeat44.json' else 'fixed_inventory_panel',
        verification=verify_seals(run,p,entries),overall={},by_role={},by_case={},by_partition={},by_role_partition={},
        solver={},solver_by_role={},solver_by_case={},linear_ablation={})
    for physics in ('AC','linear'):
        result['overall'][physics]=aggregate(rows,entries,p['days'],strategies,physics,20261114)
        for field,target in [('role','by_role'),('case','by_case'),('history_partition','by_partition')]:
            result[target][physics]={}
            for index,value in enumerate(sorted({e[field] for e in entries})):
                group=[e for e in entries if e[field]==value]
                result[target][physics][value]=aggregate(rows,group,p['days'],strategies,physics,20261114+100*index)
        result['by_role_partition'][physics]={}
        for role,part in sorted({(e['role'],e['history_partition']) for e in entries}):
            group=[e for e in entries if e['role']==role and e['history_partition']==part]
            result['by_role_partition'][physics][role+'/'+part]=aggregate(rows,group,p['days'],strategies,physics,20261114)
    result['solver']=summarize_solver(plans,entries,p['modes'],p['days'],rows)
    for field,target in [('role','solver_by_role'),('case','solver_by_case')]:
        for value in sorted({e[field] for e in entries}):result[target][value]=summarize_solver(plans,[e for e in entries if e[field]==value],p['modes'],p['days'],rows)
    rowmap={(r['id'],r['day'],r['strategy'],r['physics']):r for r in rows}
    result['per_history']={}
    for entry in entries:
        per=dict(role=entry['role'],case=entry['case'],history_partition=entry['history_partition'],
            physical_history_group=entry['physical_history_group'],days=p['days'],physics={},
            inference_scope='Conditional on this frozen history and operating condition, treating new days as independent identically distributed trials. Each strategy has a Clopper-Pearson recovery interval; paired effects resample whole day pairs,2000times. A zero-width bootstrap interval with no observed discordance is not proof of zero effect; the independent bounded interval remains nondegenerate. These are not confidence intervals for generalization across400 histories. All comparisons exploratory, unadjusted.')
        for physics in ('AC','linear'):
            values={}
            for strategy in strategies:
                own=[rowmap.get((entry['id'],day,strategy,physics),{}) for day in range(p['days'])]
                base=[rowmap.get((entry['id'],day,'no_dfs',physics),{}) for day in range(p['days'])]
                success=[r.get('status')=='complete' and r.get('native_map_correct',False) for r in own]
                before=[r.get('status')=='complete' and r.get('native_map_correct',False) for r in base]
                difference=np.asarray(success,int)-np.asarray(before,int)
                weight_difference=[a['p_true']-b['p_true'] for a,b in zip(own,base)
                    if a.get('status')==b.get('status')=='complete' and a.get('p_true') is not None and b.get('p_true') is not None]
                ci=binomtest(sum(success),p['days']).proportion_ci(.95,method='exact')
                seed=int(sha256((entry['id']+'|'+physics+'|'+strategy).encode()).hexdigest()[:8],16)
                radius=float(np.sqrt(2*np.log(40)/p['days']))
                values[strategy]=dict(native_correct=sum(success),complete=sum(r.get('status')=='complete' for r in own),
                    new_recoveries=sum(a and not b for a,b in zip(success,before)),lost_recoveries=sum(b and not a for a,b in zip(success,before)),
                    conditional_native_recovery_binomial95=[float(ci.low),float(ci.high)],
                    paired_day_recovery_gain_bootstrap95=day_interval(difference,seed),
                    paired_recovery_gain_independent_bounded95=[max(-1.,float(difference.mean())-radius),min(1.,float(difference.mean())+radius)],
                    paired_day_true_weight_gain_bootstrap95=day_interval(weight_difference,seed+1),
                    true_weight_difference_pairs=len(weight_difference),mean_true_weight_difference=mean(weight_difference),
                    mean_true_weight=mean([r.get('p_true') for r in own]),mean_settled_fee=mean([r.get('settled_fee_yuan') for r in own]),
                    nonzero_cost_days=sum(r.get('planned_fee_yuan',0)>1e-9 for r in own),zero_fallbacks=sum(r.get('zero_fallback',False) for r in own))
            per['physics'][physics]=values
        result['per_history'][entry['id']]=per
    for strategy in strategies:
        rescued=lost=both=0;pdiff=[]
        for e in entries:
            for day in range(p['days']):
                a=rowmap.get((e['id'],day,strategy,'AC'),{});b=rowmap.get((e['id'],day,strategy,'linear'),{})
                ac=a.get('status')=='complete' and a.get('native_map_correct',False)
                linear=b.get('status')=='complete' and b.get('native_map_correct',False)
                rescued+=linear and not ac;lost+=ac and not linear
                if a.get('status')==b.get('status')=='complete':
                    both+=1
                    if a.get('p_true') is not None and b.get('p_true') is not None:pdiff.append(b['p_true']-a['p_true'])
        result['linear_ablation'][strategy]=dict(planned_pairs=len(entries)*p['days'],complete_pairs=both,
            linear_recovers_AC_fails=int(rescued),AC_recovers_linear_fails=int(lost),mean_linear_minus_AC_true_weight=mean(pdiff),
            interpretation='Same physical branches/PQ/meters/actions; only new-day voltage calculation changes. Original fitted candidates and AC history remain frozen.')
    result['statistical_scope']='Development rounds are method-development evidence. Confirmation must use fresh locked paths. Report independent physical-history groups; bootstrap clusters those groups,2000replicates. All pairwise comparisons exploratory and unadjusted. Missing/failed execution counts as nonrecovery; true weights use explicit truth-containing complete-row denominators.'
    destination.mkdir(parents=True,exist_ok=True);io.write(destination/'REPORT.json',result)
    lines=['# 绝对拓扑恢复实验完整汇总','',f"阶段：{p['phase']}。{len(entries)}个冻结历史输入，每个{p['days']}条新日，共{len(entries)*p['days']}条路径。AC与物理线性计算共享路径，不作为独立样本翻倍。",'',
        f"计划策略行{result['planned_strategy_rows']}，记录{len(rows)}，执行失败{result['failed_execution_rows']}，任务失败{result['failed_tasks']}；报价代码异常{result['quote_exceptions']}/{result['planned_optimization_quotes']}次。封存核验{'通过' if result['verification']['passed'] else '失败'}。",'']
    if result['quote_exceptions']:
        lines+=['**本批存在实现失败。异常报价直接回退零动作，后续电气计算成功不代表对应方法完成求解；受影响方法的本批结果仅记录实现故障，不能作为该方法的效果证据。**','']
    if result['sample_design']=='stratified_repeat44':
        lines+=['本批是按历史资格锁定的补充分组：全部19个弱优势、9个错误胜者但含真候选输入，加每网络2个明确正确和2个缺真对照。每历史重复32条独立新日。总体均值描述这个指定面板，不能与400输入总体未经加权合并成总体准确率；每历史结果另存本地REPORT.json的per_history。',
            '', '逐历史恢复率给出条件二项Clopper–Pearson95%区间，前提是同一固定历史条件下32条新日独立同分布；方案差值按配对新日一起抽样2000次。逐历史不使用只有一个历史组的分组bootstrap。未观察到恢复差异时，日bootstrap可能仍为零宽，另列非零宽的独立有界区间；这些区间描述该历史条件，不能外推为400历史的泛化区间。','']
    fmt=lambda x:'—' if x is None else f'{x:.2f}'
    def table(group):
        lines.extend(['|方案|完整恢复|代表模型恢复|含真库平均真权重|该权重分母|实际激活费/元|非零／零费／回退|','|---|---:|---:|---:|---:|---:|---:|'])
        for strategy,s in group['strategies'].items():
            weight='—' if s['mean_true_weight'] is None else f"{100*s['mean_true_weight']:.2f}%"
            lines.append(f"|{label(strategy)}|{s['native_correct']}/{s['planned_days']}|{s['representative_correct']}/{s['planned_days']}|{weight}|{s['true_weight_denominator']}|{fmt(s['mean_settled_fee'])}|{s['nonzero_cost_days']}/{s['zero_cost_days']}/{s['zero_fallbacks']}|")
        lines.extend(['','|方案相对无服务|新增／丢失恢复|恢复率差独立有界95%区间/百分点|平均真权重差95%分组bootstrap区间/百分点|提高／下降日|最坏权重变化/百分点|','|---|---:|---:|---:|---:|---:|'])
        for mode in p['modes']:
            c=group['comparisons'][mode+'_vs_no_dfs'];ri=c['recovery_gain_independent_bounded95'];wi=c['true_weight_gain_history_cluster_bootstrap95']
            lines.append(f"|{label(mode)}|{c['new_recoveries']}/{c['lost_recoveries']}|[{100*ri[0]:.2f},{100*ri[1]:.2f}]|{'—' if wi is None else f'[{100*wi[0]:.2f},{100*wi[1]:.2f}]'}|{c['weight_improved_days']}/{c['weight_declined_days']}|{fmt(None if c['worst_weight_change'] is None else 100*c['worst_weight_change'])}|")
    for physics in ('AC','linear'):
        lines+=['## '+physics+'：总体',''];table(result['overall'][physics]);lines+=['']
        lines+=['|方案相对无服务|新增电压越界日|消除电压越界日|越界测点数变化|配对完整日|','|---|---:|---:|---:|---:|']
        for mode in p['modes']:
            c=result['overall'][physics]['comparisons'][mode+'_vs_no_dfs']
            lines.append(f"|{label(mode)}|{c['new_voltage_violation_days']}|{c['removed_voltage_violation_days']}|{c['voltage_violation_cells_change']}|{c['paired_voltage_complete_days']}|")
        lines+=['']
        lines+=['### 候选资格分组','', '|类别|路径数|无服务恢复|'+'|'.join(label(m)+'恢复' for m in p['modes'])+'|','|---|---:|---:|'+'|'.join('---:' for _ in p['modes'])+'|']
        for role,g in result['by_role'][physics].items():lines.append('|'+role+'|'+str(g['planned_days'])+'|'+'|'.join(str(g['strategies'][m]['native_correct']) for m in ['no_dfs',*p['modes']])+'|')
        lines+=['','### 网络分组','', '|网络|路径数|无服务恢复|'+'|'.join(label(m)+'恢复' for m in p['modes'])+'|','|---|---:|---:|'+'|'.join('---:' for _ in p['modes'])+'|']
        for case,g in result['by_case'][physics].items():lines.append('|'+case+'|'+str(g['planned_days'])+'|'+'|'.join(str(g['strategies'][m]['native_correct']) for m in ['no_dfs',*p['modes']])+'|')
        for part,g in result['by_partition'][physics].items():
            lines+=['','### 历史分组：'+part,''];table(g)
        lines+=['']
    lines+=['## 求解与费用边界','', '|方案|可行／计划|零动作充分|非零最优闭合|保留初值|无需可行初值找到解|平均非零间隙|平均计入预算时间/秒|同费轮换不足|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode,s in result['solver'].items():lines.append(f"|{label(mode)}|{s['feasible_quotes']}/{s['planned_quotes']}|{s['zero_sufficient']}|{s['nonzero_bound_closed']}/{s['nonzero_feasible']}|{s['initial_retained']}|{s['native_found_without_verified_initial']}|{fmt(None if s['mean_relative_gap_nonzero'] is None else 100*s['mean_relative_gap_nonzero'])}%|{fmt(s['mean_total_charged_seconds'])}|{s['same_fee_rotation_failures']}|")
    lines+=['','|方案|代码异常|QCQP内不可行证明|预算内未决|求解器时间上限|','|---|---:|---:|---:|---:|']
    for mode,s in result['solver'].items():lines.append(f"|{label(mode)}|{s['quote_exceptions']}|{s['infeasibility_proved']}|{s['unresolved_quotes']}|{s['time_limits']}|")
    lines+=['','时间含本方法初始化；公共初值实际生成时间分别计入使用它的方案预算。只对当前QCQP的有效上下界讨论最优性；未决与模型内不可行分别记录，均不证明原辨识任务不可行。费用是原示例报价的激活补偿，未另加完整市场采购成本。','',
        '## 去掉AC非线性的配对比较','', '|方案|线性恢复而AC失败|AC恢复而线性失败|线性减AC平均真权重/百分点|','|---|---:|---:|---:|']
    for strategy,a in result['linear_ablation'].items():lines.append(f"|{label(strategy)}|{a['linear_recovers_AC_fails']}|{a['AC_recovers_linear_fails']}|{fmt(None if a['mean_linear_minus_AC_true_weight'] is None else 100*a['mean_linear_minus_AC_true_weight'])}|")
    lines+=['','线性消融使用真实物理网络线路的LinDistFlow响应，不使用候选Ht生成电压；原历史仍为AC观测、参数保持冻结。它只隔离新日的AC非线性，并不消除既有参数偏差、历史评分误差或候选缺失。','',
        '全部角色、网络、历史分组、同费轮换配对、拒绝1.25/2/4倍敏感性、电压越界、初值改进、有效费用界、超时与失败明细计数见本地REPORT.json。拒绝敏感性采用每种拓扑一个代表模型的增广MAP；主恢复计数采用完整原生参数库，二者分开。原始P/Q/V、拟合矩阵、逐日报价和行动轨迹留私有输出。','',
        '恢复率差的区间利用每条新日恢复差只能取−1、0、1，按Hoeffding界取均值±sqrt(2log(40)/n)，允许固定历史之间成功率不同，依赖各新日随机流独立；本界较保守。JSON保留二项工作模型区间和p值作为探索性对照，不宣称它们对异质总体具有精确覆盖。',
        '', '权重差区间按原物理历史组做2000次bootstrap；相同历史不同噪声/压力变体不会被当作独立历史组。这是近似分析，组数少或同组正负效果抵消时区间可能很窄，不能据此忽略逐日下降。全部比较为探索性且未作多重校正。开发轮仅用于方法选择，最终确认须另用锁定随机流。','']
    (destination/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--destination',type=Path,required=True)
    args=a.parse_args();r=report(args.run.resolve(),args.destination.resolve())
    print('phase',r['phase'],'paths',r['planned_independent_future_paths'],'rows',r['recorded_strategy_rows'],'verified',r['verification']['passed'])
