"""Read-only saved C optimization diagnosis. Standard library; no project imports."""
from pathlib import Path
import collections,csv,gzip,hashlib,io,json,math,statistics

BASE=Path(r'C:\Users\23761\.codex\worktrees\36a1\Topo\research_experiments\mobile_ev_mle')
INPUT=BASE/'outputs/adaptive_milp'
OUT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def stat(vals):
    a=list(vals)
    return dict(n=len(a),minimum=min(a),median=statistics.median(a),mean=statistics.mean(a),maximum=max(a)) if a else {'n':0}
def stratum(t):return 'star' if t['truth_topology']==0 else 'weak_cherry' if t['case']=='weak' else 'strong_cherry'
def interval_class(a,b):return 'refit_LB_gt_library_UB' if a[0]>b[1] else 'library_LB_gt_refit_UB' if b[0]>a[1] else 'overlap'

gates=load(INPUT/'evaluation_v3_audit/history_gates.json');gate_map={x['history_id']:x for x in gates}
audit=load(INPUT/'evaluation_v3_audit/audit_summary.json')
config=load(INPUT/'evaluation_v3/config.json')
frozen=load(INPUT/'frozen_C_execution.json')
extracted=[];rows=[];hashes={};solver=collections.Counter();stops=collections.Counter();iterations=collections.Counter()
classes=collections.Counter();outcomes=collections.Counter();joint=collections.Counter();strata=collections.defaultdict(collections.Counter)
counts=collections.Counter();bound_updates=[];selected_match_counts=collections.Counter()
for trial_path in sorted((INPUT/'evaluation_v3').glob('trial_*.json')):
    expected=audit['evaluation_manifest']['trial_sha256'][trial_path.name];observed=sha(trial_path);assert observed==expected
    hashes[trial_path.name]=observed
    t=load(trial_path);tid=t['history_id'];r=t['selection']['methods']['refit_milp'];l=t['selection']['methods']['legacy_library_proxy']
    assert r['success'] and r['status']=='complete' and not r['selector_uses_truth'] and not r['selector_uses_future_noise']
    matched=[v for v in r['records'] if v['u_kw']==r['selected_u_kw']];assert matched
    # JSON +0/-0 can yield two saved evaluations of one mathematical control.
    # Intersect valid intervals for the same selected control, never take the global design UB.
    selected_lo=max(v['q_lower'] for v in matched);selected_hi=min(v['q_upper'] for v in matched)
    assert selected_lo<=selected_hi and selected_lo==r['proxy_lower']
    assert all(min(v['q_lower'] for v in a['pair_values'])==a['q_lower'] and min(v['q_upper'] for v in a['pair_values'])==a['q_upper'] for a in matched)
    selected_match_counts[len(matched)]+=1
    raw_gap=r['proxy_upper']-r['proxy_lower'];abs_gap=max(0,raw_gap);rel_gap=abs_gap/max(1,abs(r['proxy_lower']))
    assert abs_gap==r['absolute_gap'] and rel_gap==r['relative_gap']
    tolerance=config['policy']['absolute_gap']+config['policy']['relative_gap']*max(1,abs(r['proxy_lower']))
    assert bool(abs_gap<=tolerance)==r['proxy_gap_met']
    stops[r['proxy_stop_reason']]+=1;iterations[(r['proxy_stop_reason'],len(r['solver_records'])//2)]+=1
    for s in r['solver_records']:
        solver[(s['kind'],s['status'],s['success'],s['incumbent'] is None,s['bound'] is None)]+=1
        if s['kind']=='secant_upper' and s['bound'] is not None:bound_updates.append(tid)
    library_complete=l['status']=='complete';library_actions=l['actions']
    counts['library_action_score_rows']+=len(library_actions)
    if library_complete:
        assert len(library_actions)==11 and all(a['success'] for a in library_actions)
        assert l['criterion']=='same frozen soft-history proxy as refit MILP'
        chosen=min(library_actions,key=lambda a:(-a['q_lower'],a['action']))
        assert chosen['action']==l['selected']
    else:chosen=None
    constant=next((a for a in library_actions if a['action']=='constant_site_0'),None)
    assert constant and constant['success']
    gain_lo=selected_lo-constant['q_upper'];gain_hi=selected_hi-constant['q_lower']
    counts['gain_interval_contains_zero']+=int(gain_lo<=0<=gain_hi)
    if r['proxy_gap_met']:
        counts['gap_met_gain_interval_contains_zero']+=int(gain_lo<=0<=gain_hi)
        counts['gap_met_tolerance_exceeds_positive_gain_upper']+=int(tolerance>max(0,gain_hi))
        counts['gap_met_only_relative_part_not_absolute']+=int(abs_gap>config['policy']['absolute_gap'])
    active=[a for a in r['actual_rollback'] if 'proxy_q_lower' in a and 'actual_soft_cost' in a]
    witness=min(active,key=lambda a:a['proxy_q_lower']) if len(active)==len(r['actual_rollback']) else None
    witness_history_share=(sum(witness['actual_history_costs'])/witness['actual_soft_cost']) if witness and witness['actual_soft_cost']>0 else None
    witness_future_share=(witness['actual_future_distance_squared']/witness['actual_soft_cost']) if witness and witness['actual_soft_cost']>0 else None
    rg=gate_map[tid]['methods']['refit_milp'];lg=gate_map[tid]['methods']['legacy_library_proxy']
    outcome=('both_unique_correct' if rg['unique_correct_retention'] and lg['unique_correct_retention'] else
             'refit_only_unique_correct' if rg['unique_correct_retention'] else 'library_only_unique_correct' if lg['unique_correct_retention'] else 'neither_unique_correct')
    outcomes[outcome]+=1
    cls=interval_class((selected_lo,selected_hi),(chosen['q_lower'],chosen['q_upper'])) if chosen else 'library_selection_failure'
    classes[cls]+=1;strata[stratum(t)][cls]+=1;joint[(cls,outcome)]+=1
    lib_advantage=chosen['q_lower']-selected_hi if chosen else None
    if cls=='library_LB_gt_refit_UB':
        for th in [1e-4,1e-3,1e-2,1.]:counts['library_advantage_gt_'+str(th)]+=int(lib_advantage>th)
        counts['library_advantage_gt_relative_1e-6']+=int(lib_advantage>1e-6*max(1,abs(selected_hi),abs(chosen['q_lower'])))
    if chosen:assert chosen['q_lower']<=r['proxy_upper']
    row={'history_id':tid,'stratum':stratum(t),'refit_selection_status':r['status'],'proxy_stop_reason':r['proxy_stop_reason'],
         'selected_record_matches':len(matched),'refit_global_proxy_lower':r['proxy_lower'],'refit_global_proxy_upper':r['proxy_upper'],
         'raw_global_gap':raw_gap,'global_absolute_gap':abs_gap,'global_relative_gap':rel_gap,'stopping_tolerance':tolerance,
         'selected_action_q_lower':selected_lo,'selected_action_q_upper':selected_hi,'selected_action_q_gap':selected_hi-selected_lo,
         'library_status':l['status'],'library_selected':l.get('selected'),'library_action_q_lower':chosen['q_lower'] if chosen else None,
         'library_action_q_upper':chosen['q_upper'] if chosen else None,'library_action_q_gap':chosen['q_upper']-chosen['q_lower'] if chosen else None,
         'interval_comparison':cls,'library_certified_action_advantage':lib_advantage,
         'constant_q_lower':constant['q_lower'],'constant_q_upper':constant['q_upper'],'gain_vs_constant_lower':gain_lo,
         'gain_vs_constant_upper':gain_hi,'gap_exceeds_gain_upper':abs_gap>max(0,gain_hi),
         'tolerance_exceeds_gain_upper':tolerance>max(0,gain_hi),'actual_active_witness_history_share':witness_history_share,
         'actual_active_witness_future_share':witness_future_share,'unique_correct_pair_category':outcome,
         'refit_unique_correct':rg['unique_correct_retention'],'library_unique_correct':lg['unique_correct_retention'],
         'refit_false_rejection':rg['false_rejection'],'library_false_rejection':lg['false_rejection']}
    rows.append(row)
    extracted.append({'history_id':tid,'stratum':stratum(t),'source_trial_sha256':observed,
                      'refit':{k:r[k] for k in ['proxy_lower','proxy_upper','absolute_gap','relative_gap','proxy_stop_reason','proxy_gap_met','selected_u_kw','solver_records']},
                      'refit_selected_record_intervals':[{'origin':a['origin'],'q_lower':a['q_lower'],'q_upper':a['q_upper']} for a in matched],
                      'legacy_library':l,'evaluation_events':{'refit_unique_correct':rg['unique_correct_retention'],
                        'library_unique_correct':lg['unique_correct_retention'],'refit_false_rejection':rg['false_rejection'],
                        'library_false_rejection':lg['false_rejection']},'diagnostic':row})
assert len(rows)==256 and len(hashes)==256
assert sum(r['refit_unique_correct'] for r in rows)==71 and sum(r['library_unique_correct'] for r in rows)==81
with (OUT/'raw_extracts.json.gz').open('xb') as f:f.write(gzip.compress(json.dumps(extracted,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8'),mtime=0))
with (OUT/'history_diagnostics.csv').open('x',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
numbers={'scope':'read-only frozen C saved optimization diagnosis; no optimization, policy change or recalibration',
         'statistical_unit':'one independent history; paired refit and library measurements','history_denominator':256,
         'complete_library_comparison_denominator':255,'library_failure_ids':[r['history_id'] for r in rows if r['library_status']!='complete'],
         'same_criterion_scope':'refit and all eleven library actions share frozen soft-history local-refit proxy; fixed-MLE criterion excluded',
         'interval_comparison_counts':dict(classes),'stratum_interval_comparison_counts':{s:dict(v) for s,v in strata.items()},
         'stop_reason_counts':dict(stops),'iteration_counts':[{'stop_reason':s,'iterations':i,'count':n} for (s,i),n in iterations.items()],
         'solver_records':[{'kind':k,'status':s,'success':v,'incumbent_missing':im,'bound_missing':bm,'count':n} for (k,s,v,im,bm),n in solver.items()],
         'secant_bound_update_count':len(bound_updates),'selected_record_match_counts':dict(selected_match_counts),
         'method_unique_correct_counts':{'refit':71,'library':81},'paired_unique_correct_counts':dict(outcomes),
         'proxy_interval_vs_actual_outcome':[{'interval_class':c,'outcome':o,'count':n} for (c,o),n in joint.items()],
         'scale_and_sensitivity_counts':dict(counts),'library_lower_exceeds_global_upper_count':0,
         'metric_summaries':{key:stat(r[key] for r in rows if r[key] is not None) for key in
             ['refit_global_proxy_lower','global_absolute_gap','global_relative_gap','selected_action_q_gap','library_action_q_gap',
              'gain_vs_constant_lower','gain_vs_constant_upper','actual_active_witness_history_share','actual_active_witness_future_share']},
         'selected_cases':{},'interpretation_limits':[
             'Numerical convex-QP/affine-proxy intervals are not interval-arithmetic or global nonlinear-AC certificates.',
             'Action q_upper bounds one fixed action inner minimization; global proxy_upper bounds maximization over the design domain.',
             'Strict separated intervals can have tiny absolute advantages; sensitivity counts do not redefine the frozen stopping rule.',
             'Saved active nonlinear rollback witness history/future shares are not a decomposition of the optimal affine q value.',
             'Actual unique-correct endpoint is not monotone in this proxy on the saved sample; no causal attribution of the whole ten-count difference.',
             'Descriptive stratification and examples never feed back into the frozen policy.']}
opposite=next(r['history_id'] for r in rows if r['interval_comparison']=='library_LB_gt_refit_UB' and r['unique_correct_pair_category']=='refit_only_unique_correct')
for tid in ['730007','730030',opposite]:numbers['selected_cases'][tid]=next(r for r in rows if r['history_id']==tid)
dump(OUT/'C_optimization_diagnostic_numbers.json',numbers)
inputs={'trial_sha256':hashes,'config_sha256':sha(INPUT/'evaluation_v3/config.json'),
        'audit_summary_sha256':sha(INPUT/'evaluation_v3_audit/audit_summary.json'),
        'history_gates_sha256':sha(INPUT/'evaluation_v3_audit/history_gates.json'),
        'frozen_execution_sha256':sha(INPUT/'frozen_C_execution.json'),'frozen_core_source_sha256':frozen['source_sha256'],
        'source_definitions_used':{},'raw_extracts_compressed_sha256':sha(OUT/'raw_extracts.json.gz'),
        'source_root':str(INPUT),'all_trial_hashes_match_original_manifest':True,'original_outputs_read_only':True}
for name in ['ev_information_milp.py','adaptive_milp_study.py','adaptive_history.py']:
    value=sha(BASE/name);assert value==frozen['source_sha256'][name];inputs['source_definitions_used'][name]=value
dump(OUT/'input_hashes.json',inputs)
table='\n'.join('| '+s+' | '+str(sum(v.values()))+' | '+ ' | '.join(str(v.get(c,0)) for c in ['refit_LB_gt_library_UB','library_LB_gt_refit_UB','overlap','library_selection_failure'])+' |' for s,v in strata.items())
cases='\n'.join('| '+tid+' | '+r['stratum']+' | '+format(r['refit_global_proxy_lower'],'.8g')+' | '+format(r['global_absolute_gap'],'.8g')+' | '+format(r['stopping_tolerance'],'.8g')+' | ['+format(r['gain_vs_constant_lower'],'.8g')+', '+format(r['gain_vs_constant_upper'],'.8g')+'] | '+r['proxy_stop_reason']+' |' for tid,r in numbers['selected_cases'].items())
text='''# C 阶段 MILP 与原动作库优化诊断

**C 的 71 对 81 次唯一正确保留差异同时包含求解受限与代理目标对实际终点不充分的问题，不能单独归因于“MILP 搜索域更大所以应更好”。** refit 未保证选中动作的代理值高于原 11 动作库：255 条完整比较中，原库动作下界高于 refit 动作上界 99 条，反向 16 条，区间重叠 140 条；另 1 条原库选址失败保留在全 256 历史分母内。所有数字来自既有 C trial，未调用优化器或更改冻结策略。

## 同一判据和两层界

源码 adaptive_milp_study.py:142–158 与 164–171 表明，原库遍历全部 11 个动作，用同一 frozen soft-history local-refit proxy 的 q_lower 最大者（动作名作确定性 tie-break）选址；refit 对同样的历史上下文、中心/情景/参数盒与冻结线性化构造进行 MILP 设计搜索。原库是有限基线动作库，未穷举 MILP 全设计域。fixed_mle_milp 固定中心、来源与阻抗范围，属于不同判据，本诊断不把它混入上述比较。

对固定动作 u，每 pair 内层盒约束凸 QP 的 q_lower 是数值切线下界，q_upper 是可行 primal 上界；跨 pair 取最小值，得到所选动作的代理区间。refit 的 proxy_lower 是已找到且经运行约束核查的动作 lower incumbent；proxy_upper 是全设计域最大化目标的上界，初值来自 control-box witness upper，之后仅有可用 secant-master bound 才能收紧。**全域 proxy_upper 不是所选动作 q_upper。** 本诊断从 selected_u_kw 匹配 records，重复的相同控制（包括 +0/−0）取区间交集，256 条均匹配且动作下界等于 proxy_lower。

## 求解状态与动作代理排序

refit 的外层保存 status=complete 共 256，表示选址与预测准入完成；原搜索停止原因须读 proxy_stop_reason：proxy_gap_met=128，budget_or_iteration_limit=128。direction_lower 共 515 次，均 HiGHS status=0 / success=true；secant_upper 共 515 次，均 status=1/time limit，且 incumbent、bound 全部缺失。因此没有一次 secant-master bound 更新，全域 UB 始终保留初始 box upper；“选址完成”不能解释为全设计域代理最优。

| 描述层 | 历史数 | refit动作LB > 库动作UB | 库动作LB > refit动作UB | 重叠 | 库失败 |
|---|---:|---:|---:|---:|---:|
'''+table+'''

99 条严格库优区间中，76 条优势 >1e−4，75 条 >0.01，69 条 >1；76 条超过 1e−6×max(1,|两动作代理值|)。这些仅说明数值优势尺度，不修改原规则。128 条 gap_met 中，21 条仍有库动作LB > refit动作UB，但优势多在小尺度；128 条预算/迭代停止中，对应 78 条。库动作下界超过 refit 全域 UB 为 0 条，未观察到这一类全域界矛盾。

## 停止 gap 与新增收益的尺度

源码 ev_information_milp.py:390、395–400 使用

`global_gap=max(0,proxy_upper−proxy_lower)`，`relative_gap=global_gap/max(1,|proxy_lower|)`，允许 gap 为 `1e−4 + 1e−3×max(1,|proxy_lower|)`。

q 的历史残差和未来候选差异一并进入目标（adaptive_history.py:423–427）。因此相对 gap 的分母是总代理代价，不是相对 constant 的新增收益。本诊断的动作增益区间为 `[refit动作LB−constant动作UB, refit动作UB−constant动作LB]`，不以全域 UB 充当 refit 动作分数。

128 条 gap_met 中，43 条动作增益区间包含 0；28 条允许 gap 大于正部增益上界；其中 126 条超过绝对 gap=1e−4，停止依赖相对项。本诊断按最小 proxy_q_lower 选取保存的非线性 rollback witness，future share 中位数约 3.27e−7，n=254，典型历史由历史项占据。这支持“按总代价相对停止可能不能分辨新增收益”这一机制解释；该 witness 比例不等于代理最优值的严格历史/未来分解。

| 历史 | 层 | refit总LB | 全域gap | 允许gap | 相对constant动作增益区间 | 停止原因 |
|---|---|---:|---:|---:|---|---|
'''+cases+'''

730007 的总 LB≈59.7551，允许 gap≈0.059855，而新增收益区间为 [−0.00238774,7.14e−7]，全域 gap≈0.00249398；已满足总代价相对停止，却未证实相对 constant 的正收益。730030 的库代理优势≈1317.06、全域 gap≈3672.64，属于明显预算/迭代未闭合例，库唯一正确而 refit 未唯一正确。

## 与 71 对 81 的实际终点关联

两者均唯一正确 67，refit 单独 4，库单独 14，两者均未唯一正确 171；净差为库多 10。库单独 14 条中，6 条代理区间严格库优，8 条重叠。refit 单独 4 条中，1 条代理区间严格库优，3 条重叠；严格 refit 代理优的 16 条没有出现 refit 单独唯一正确。这是保存样本中代理排序不足以决定实际唯一保留终点的直接反例，不能把更高代理值解释成更高拓扑正确概率。

所以求解受限有明确证据（99 条动作代理被基线分离压过、全部 secant 超时无界），同时当前代理与局部 AC 联合拟合/双门限终点不构成单调对应。现有证据没有识别出这 10 条净差各由多少求解误差、线性化、参数盒/历史项或门限效应造成；重叠不等于相同代理值。进一步归因需要预先设计的同判据精度/预算研究，此诊断没有开展新实验或依据评价结果反调策略。

raw_extracts.json.gz 保留全 256 的选中动作区间、完整 11 动作库记录（失败历史只有已保存 1 行）、global界/停止原因/solver记录与实际事件；history_diagnostics.csv 为逐历史表，numbers JSON 为统计与实例，input_hashes.json 保留 256 原 trial、配置、门限审查及冻结源码哈希。所有比较界仅针对声明冻结仿射代理与有限 pair，属于浮点数值证据，不能升级为完整非线性 AC 全局最优证书。
'''
with (OUT/'C_optimization_diagnostic.md').open('x',encoding='utf-8',newline='\n') as f:f.write(text)
print(json.dumps({'status':'complete','history_count':256,'interval_counts':dict(classes),'solver_records':numbers['solver_records'],
                  'selected_cases':list(numbers['selected_cases']),'files':str(OUT)},ensure_ascii=False))
