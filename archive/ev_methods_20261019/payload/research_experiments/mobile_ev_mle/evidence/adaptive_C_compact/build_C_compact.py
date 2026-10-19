"""Read frozen C inputs; write a new compact archive only. No project imports."""
from pathlib import Path
import collections, csv, gzip, hashlib, io, json, math

BASE = Path(r'C:\Users\23761\.codex\worktrees\36a1\Topo\research_experiments\mobile_ev_mle')
INPUT = BASE / 'outputs' / 'adaptive_milp'
OUT = Path(__file__).resolve().parent
STAGING = OUT.parent
CANON = ['star', 'clade_01', 'clade_02', 'clade_12']
EVENTS = ['false_rejection', 'truth_omission', 'truth_loss_event', 'unique_correct_retention',
          'algorithm_unique_correct_retention', 'relative_unique_correct_retention',
          'wrong_unique_retention', 'unresolved', 'empty_retained', 'selection_failure',
          'fit_failure', 'physical_failure', 'bank_adequacy_alarm',
          'any_wrong_candidate_eliminated', 'all_wrong_candidates_eliminated']

def sha(data): return hashlib.sha256(data).hexdigest()
def read_json(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write_new(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f: f.write(data)
def json_bytes(x): return (json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
def write_json(name, x): write_new(OUT/name, json_bytes(x))
def csv_bytes(rows, fields):
    s = io.StringIO(newline='')
    w = csv.DictWriter(s, fieldnames=fields, lineterminator='\n'); w.writeheader(); w.writerows(rows)
    return s.getvalue().encode('utf-8')
def bool_csv(x): return x in ('True', 'true', '1')
def stratum(case, topology): return 'star' if topology == 0 else 'weak_cherry' if case == 'weak' else 'strong_cherry'
def bitmask(values): return sum(int(v)<<i for i,v in enumerate(values))

if (OUT/'evidence_manifest.json').exists(): raise SystemExit('Refusing to overwrite an existing archive')
gates = read_json(INPUT/'calibration_v3/calibrated_gates.json')
audit = read_json(INPUT/'evaluation_v3_audit/audit_summary.json')
stored_gates = {r['history_id']:r for r in read_json(INPUT/'evaluation_v3_audit/history_gates.json')}
frozen = read_json(INPUT/'frozen_C_execution.json')
numbers = read_json(STAGING/'formal_C_independent_numbers.json')
methods = audit['evaluation_config']['methods']
relative = gates['calibration']['relative']['threshold']; absolute = gates['calibration']['absolute']['threshold']
assert len(methods)==7
archive_files=[]

def archive(source, target, compress=True, role='original frozen evidence'):
    data=source.read_bytes(); packed=gzip.compress(data, compresslevel=9, mtime=0) if compress else data
    write_new(OUT/target, packed)
    entry={'archive_path':target, 'source_path':str(source), 'role':role,
           'source_bytes':len(data), 'source_sha256':sha(data), 'archive_bytes':len(packed),
           'archive_sha256':sha(packed), 'compression':'gzip' if compress else 'none',
           'decompressed_bytes_are_original':True}
    if source.suffix=='.csv': entry['data_row_count']=len(list(csv.DictReader(io.StringIO(data.decode('utf-8-sig')))))
    archive_files.append(entry)
    assert (gzip.decompress(packed) if compress else packed)==data

selected = {
 'calibration_v3':['config.json','summary.json','calibrated_gates.json'],
 'evaluation_v3':['config.json','summary.json'],
 'evaluation_v3_audit':['audit_summary.json','history_gates.json','fits.csv','actions.csv','raw_bound_crossings.json'],
 'evaluation_v3_empirical':['empirical_design_summary.json'],
 'evaluation_v3_impedance':['impedance_component_summary.json'],
}
for folder, files in selected.items():
    for name in files:
        compress=(INPUT/folder/name).stat().st_size>100000
        archive(INPUT/folder/name, 'originals/'+folder+'/'+name+('.gz' if compress else ''), compress)
archive(INPUT/'frozen_C_execution.json', 'originals/frozen_C_execution.json', False)
archive(STAGING/'formal_C_independent_review.md', 'formal_C_independent_review.md', False, 'existing independent review, unchanged copy')
archive(STAGING/'formal_C_independent_numbers.json', 'formal_C_independent_numbers.json', False, 'existing independent numbers, unchanged copy')

# Verify every manifest member against the original read-only source. Archive hashes, not raw trials.
input_manifests={}; verification={}
for phase, folder in [('calibration','calibration_v3'),('evaluation','evaluation_v3')]:
    manifest=audit[phase+'_manifest']; input_manifests[phase]=manifest
    entries={'config.json':manifest['config_sha256'], **manifest['trial_sha256'], **manifest['observation_archive_sha256']}
    mismatches=[]
    for name, expected in entries.items():
        actual=sha((INPUT/folder/name).read_bytes())
        if actual!=expected: mismatches.append(name)
    assert not mismatches, (phase,mismatches)
    verification[phase+'_manifest']={'entries_verified':len(entries),'mismatches':mismatches}
assert gates['source_manifest']==audit['calibration_manifest']
core=[]
for name, expected in frozen['source_sha256'].items():
    p=(BASE/name).resolve(); actual=sha(p.read_bytes()); assert actual==expected, name
    core.append({'source_path':str(p),'relative_path':name,'frozen_sha256':expected,'observed_sha256':actual,'match':True})
empirical=read_json(INPUT/'evaluation_v3_empirical/empirical_design_summary.json')
impedance=read_json(INPUT/'evaluation_v3_impedance/impedance_component_summary.json')
post=[]
for name,expected in [('adaptive_result_audit.py',audit['collector_sha256']),
                      ('empirical_design_analysis.py',empirical['analysis_source_sha256']),
                      ('impedance_component_audit.py',impedance['source_sha256'])]:
    actual=sha((BASE/name).read_bytes()); assert actual==expected,name
    post.append({'relative_path':name,'frozen_sha256':expected,'observed_sha256':actual,'match':True})
write_json('source_manifests.json', {'raw_input_manifests':input_manifests,'frozen_core_sources':core,
                                  'postprocessing_sources':post,'verification':verification})

# Full calibration scores: recompute all-method/round maxima and retain failed histories as +inf.
scores=[]; evaluation_metadata=[]; calibration_ids=set()
for name in sorted(audit['calibration_manifest']['trial_sha256']):
    trial=read_json(INPUT/'calibration_v3'/name); tid=trial['history_id']; calibration_ids.add(tid)
    assert trial['independent_history_unit']
    true=trial['true_index']; covered=true is not None and true>=0
    failed=sum(not(s and math.isfinite(float(loss)) and float(loss)>=0)
               for r in trial['records'] for s,loss in zip(r['success'],r['losses']))
    complete=covered and failed==0
    rel=max(float(r['losses'][true])-min(map(float,r['losses'])) for r in trial['records']) if complete else math.inf
    ab=max(float(r['losses'][true]) for r in trial['records']) if complete else math.inf
    saved=trial['score_record']
    for key,value in [('relative_gap',rel),('absolute_loss',ab)]:
        sv=saved[key]; sv=math.inf if sv is None else float(sv)
        assert value==sv,(tid,key,value,sv)
    scores.append({'history_id':tid,'true_candidate':trial['candidate_names'][true] if covered else '',
                   'stratum':stratum(trial['case'],trial['truth_topology']), 'complete':int(complete),
                   'compute_failure':int(failed>0),'coverage_failure':int(not covered),
                   'failed_fit_count':failed,'relative_score':repr(rel),'absolute_score':repr(ab),
                   'source_trial_sha256':audit['calibration_manifest']['trial_sha256'][name]})
assert len(scores)==128 and sum(r['complete'] for r in scores)==127
assert sum(r['compute_failure'] for r in scores)==1
for field,gate in [('relative_score','relative'),('absolute_score','absolute')]:
    values=sorted(float(r[field]) for r in scores); n=len(values); alpha=gates['calibration'][gate]['alpha']
    rank=math.ceil((n+1)*(1-alpha)); assert rank==123
    assert values[rank-1]==gates['calibration'][gate]['threshold']
    assert sum(math.isinf(v) for v in values)==1
write_new(OUT/'calibration_scores.csv',csv_bytes(scores,list(scores[0])))
for name in sorted(audit['evaluation_manifest']['trial_sha256']):
    trial=read_json(INPUT/'evaluation_v3'/name); assert trial['independent_history_unit']
    evaluation_metadata.append({'history_id':trial['history_id'],'case':trial['case'],
                                'truth_topology':trial['truth_topology'],'true_candidate':trial['candidate_names'][trial['true_index']],
                                'candidate_names':trial['candidate_names'],'true_index':trial['true_index'],
                                'stratum':stratum(trial['case'],trial['truth_topology']),
                                'independent_history_unit':True,'selection_uses_truth':trial['selection_uses_truth'],
                                'selection_uses_future_noise':trial['selection_uses_future_noise']})
assert len(evaluation_metadata)==256 and not(calibration_ids & {x['history_id'] for x in evaluation_metadata})
metadata={r['history_id']:r for r in evaluation_metadata}
write_new(OUT/'evaluation_metadata.json.gz',gzip.compress(json_bytes(evaluation_metadata),mtime=0))

# Derive every method-history mask and event directly from archived CSV values.
with (INPUT/'evaluation_v3_audit/fits.csv').open(newline='',encoding='utf-8-sig') as f: fits=list(csv.DictReader(f))
with (INPUT/'evaluation_v3_audit/actions.csv').open(newline='',encoding='utf-8-sig') as f: actions=list(csv.DictReader(f))
assert len(fits)==7168 and len(actions)==1792
groups=collections.defaultdict(dict)
for r in fits:
    key=(r['history_id'],r['method'],r['round']); assert r['candidate'] not in groups[key]
    groups[key][r['candidate']]=r
action_map={(r['history_id'],r['method'],r['round']):r for r in actions}; assert len(action_map)==1792
masks=[]; counts={m:dict.fromkeys(EVENTS,0) for m in methods}; strata={s:{'denominator':sum(r['stratum']==s for r in evaluation_metadata),'methods':{m:dict.fromkeys(EVENTS,0) for m in methods}} for s in ['star','weak_cherry','strong_cherry']}
union=[]; mask_comparisons=0; event_comparisons=0; exceptions=[]
for tid in sorted(metadata):
    meta=metadata[tid]; true=CANON.index(meta['true_candidate']); any_false=False
    for method in methods:
        key=(tid,method,'0'); group=groups[key]; action=action_map[key]
        assert set(group)==set(CANON)
        losses=[float(group[n]['loss']) for n in CANON]
        complete=all(bool_csv(group[n]['success']) and math.isfinite(loss) and loss>=0 for n,loss in zip(CANON,losses))
        rel=[complete and loss-min(losses)>relative for loss in losses]
        ab=[complete and loss>absolute for loss in losses]; combined=[r or a for r,a in zip(rel,ab)]
        retained=[not x for x in combined]; relative_retained=[not x for x in rel]
        physical=bool_csv(action['actual_voltage_ok']); selection_failure=action['selection_status']=='selection_failure_fallback'
        wrong=[i for i in range(4) if i!=true]; eliminated=sum(combined[i] for i in wrong)
        correct=complete and physical and sum(retained)==1 and retained[true]
        events={'false_rejection':combined[true],'truth_omission':False,'truth_loss_event':combined[true],
          'unique_correct_retention':correct,'algorithm_unique_correct_retention':correct and not selection_failure,
          'relative_unique_correct_retention':complete and physical and sum(relative_retained)==1 and relative_retained[true],
          'wrong_unique_retention':sum(retained)==1 and not retained[true],'unresolved':sum(retained)>1,
          'empty_retained':sum(retained)==0,'selection_failure':selection_failure,'fit_failure':not complete,
          'physical_failure':not physical,'bank_adequacy_alarm':complete and min(losses)>absolute,
          'any_wrong_candidate_eliminated':eliminated>0,'all_wrong_candidates_eliminated':eliminated==3}
        stored=stored_gates[tid]['methods'][method]; order=meta['candidate_names']; gate=stored['gates'][0]
        for field,values in [('relative_reject',rel),('absolute_reject',ab),('combined_reject',combined)]:
            assert [values[CANON.index(n)] for n in order]==gate[field],(tid,method,field)
            mask_comparisons+=1
        for event,value in events.items():
            assert bool(stored[event])==bool(value),(tid,method,event)
            event_comparisons+=1; counts[method][event]+=int(value); strata[meta['stratum']]['methods'][method][event]+=int(value)
        any_false=any_false or events['false_rejection']
        row={'history_id':tid,'method':method,'round':0,'stratum':meta['stratum'],'truth_topology':meta['truth_topology'],
             'true_candidate':meta['true_candidate'],'candidate_order':'|'.join(CANON),
             'relative_reject_mask':bitmask(rel),'absolute_reject_mask':bitmask(ab),'combined_reject_mask':bitmask(combined),
             'retained_mask':bitmask(retained),'complete':int(complete),'operational_voltage_ok':int(physical),
             'wrong_candidates_eliminated':eliminated, **{e:int(v) for e,v in events.items()}}
        masks.append(row)
        if selection_failure or not complete or events['wrong_unique_retention']:
            exceptions.append({'history_id':tid,'method':method,'stratum':meta['stratum'],
                               'true_candidate':meta['true_candidate'],'retained_candidates':[n for n,r in zip(CANON,retained) if r],
                               'candidate_losses':dict(zip(CANON,losses)),'fit_success':{n:bool_csv(group[n]['success']) for n in CANON},
                               'selection_status':action['selection_status'],'fit_status':action['fit_status'],
                               'events':{e:int(v) for e,v in events.items()}})
    if any_false: union.append(tid)
assert len(masks)==1792 and len(union)==14
for method in methods:
    for e in EVENTS: assert counts[method][e]==audit['methods'][method]['events'][e]['count'],(method,e)
    for e in numbers['method_event_counts'][method]: assert counts[method][e]==numbers['method_event_counts'][method][e],(method,e)
for s,info in strata.items():
    expected=numbers['strata'][s]; assert info['denominator']==expected['histories']
    for method in methods:
        values=[info['methods'][method][e] for e in ['unique_correct_retention','false_rejection','unresolved','selection_failure','fit_failure']]
        assert values==expected['methods'][method],(s,method)
write_new(OUT/'history_method_masks.csv.gz',gzip.compress(csv_bytes(masks,list(masks[0])),mtime=0))
counts_payload={'statistical_unit':'independent generator history','history_denominator':256,'method_records':1792,'candidate_fits':7168,
                'candidate_bit_order':CANON,'bit_encoding':'bit i = candidate_order[i]; 1 means reject, retained_mask uses 1 means retained',
                'method_event_counts':counts,'strata':strata,'all_methods_false_rejection_union':{'count':14,'denominator':256,'history_ids':union},
                'independent_recomputation':{'mask_arrays_compared':mask_comparisons,'method_history_event_values_compared':event_comparisons,
                                            'disagreements':0,'full_summary_event_counts_match':True,'independent_review_counts_match':True}}
write_json('compact_counts.json',counts_payload)
weak=numbers['weak_cherry_random_absolute_gate_exception']
weak_row=next(r for r in masks if r['history_id']=='730121' and r['method']=='random_action')
assert weak_row['relative_reject_mask']==0 and weak_row['absolute_reject_mask']==7 and weak_row['unique_correct_retention']==1
write_json('exception_cases.json',{'failure_and_wrong_unique_cases':exceptions,'weak_cherry_random_absolute_gate_case':weak,
                                'calibration_failed_history_ids':[r['history_id'] for r in scores if not r['complete']]})

raw=INPUT/'evaluation_v3_audit/numerical_bounds.csv'
raw_record={'source_path':str(raw),'bytes':raw.stat().st_size,'sha256':sha(raw.read_bytes()),'archived':False,
            'reason':'7.9 MB raw numerical bounds retained at the original read-only source; crossing records archived separately',
            'evaluation_negative_gap_count':6,'maximum_relative_crossing':6.503972900110877e-7,
            'calibration_negative_gap_count':3,'raw_negative_gaps_not_clipped':True,
            'tolerance_acceptance_is_not_strict_order_or_optimality_certificate':True}
verification.update({'frozen_core_hashes_verified':len(core),'postprocessing_hashes_verified':len(post),
                     'calibration_total':128,'calibration_complete':127,'calibration_compute_failure':1,
                     'calibration_inf_score_counts':{'relative':1,'absolute':1},'calibration_eval_id_overlap':0,
                     'fit_rows':len(fits),'action_rows':len(actions),'mask_rows':len(masks),
                     'mask_array_comparisons':mask_comparisons,'event_value_comparisons':event_comparisons,'disagreements':0,
                     'all_method_union_false_rejection':14})

main=[]
for method in methods:
    c=counts[method]
    main.append('| '+method+' | '+ ' | '.join(str(c[e]) for e in ['unique_correct_retention','false_rejection','unresolved','wrong_unique_retention','selection_failure','fit_failure'])+' |')
stratified=[]
for s in ['star','weak_cherry','strong_cherry']:
    for method in methods:
        c=strata[s]['methods'][method]
        stratified.append('| '+s+' | '+str(strata[s]['denominator'])+' | '+method+' | '+' | '.join(str(c[e]) for e in ['unique_correct_retention','false_rejection','unresolved','wrong_unique_retention','selection_failure','fit_failure'])+' |')
report='''# 冻结 C 阶段结果与紧凑证据归档

C 已完成 cal128 / eval256。refit 相对 constant 的唯一正确保留增加 9/256，增益集中在 strong cherry；这批结果没有支持 star 或 weak cherry 的稳定唯一识别，也没有支持 refit 在七种方法中整体最好。七方法至少一次误拒真候选的历史为 14/256（5.46875%）。本结论仅针对冻结 nominal 模拟总体和固定数值策略。

本包从既有只读输出复制、压缩和独立重算，没有运行新 AC 优化、调整策略或门限、改动正式源码或覆盖旧输出。原始 JSON/NPZ 仍保留在来源目录，完整输入 manifest 和 15 个核心源码冻结 SHA256 见 source_manifests.json。已有独立审查原文及数字原样复制。

## 分母、失败与结果定义

评价单位为 256 条独立生成器历史。每条历史 7 方法、1 轮、4 候选，得到 1,792 条方法记录和 7,168 条候选拟合；方法/候选记录是历史内的配对测量，不是额外独立样本。所有计数保留完整 256 分母，事件可重叠。唯一正确保留要求真候选在库中、拟合完整、运行电压检查通过且联合保留集合仅含真候选。选址 fallback 另作失败标记，并另列 algorithm_unique_correct_retention。

| 方法 | 唯一正确保留 | 误拒真候选 | 未决 | 错误唯一保留 | 选址失败 | 拟合失败 |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(main)+'''

legacy_library_proxy 的选址失败 730085 与 random_action 的联合拟合失败 730180 均未删除；后者 clade_01、clade_02 的 optimizer success=false。错误唯一、空保留、真候选误拒不是同一事件；它们的逐历史列表和掩码可分别核查。全部方法的真候选遗漏、电压失败为 0。

## 冻结校准与拒绝规则

校准共有 128 历史，完整 127、计算失败 1。每条历史的 relative score 是七方法/所有声明轮次上“真 loss − 该方法最小候选 loss”的最大值，absolute score 是真 loss 的最大值；任何必要拟合失败在两组分数中记 +∞。calibration_scores.csv 保存全部 128 条（inf 文本表示正无穷），失败历史没有从校准分母中排除。

两门限分别使用 α=0.05，k=ceil((128+1)(1−0.05))=123。第 123 个顺序统计量为 relative=8.610311359932325、absolute=75.77255697515467。严格以大于门限拒绝；单门限秩尾界 (129−123)/129=6/129，合计 12/129=0.0930232558，冻结 nominal 总预算为 0.1。必要候选拟合失败时两拒绝掩码全 0、保留全部候选并记录失败。正常联合掩码为 relative OR absolute。

七方法联合误拒为 14/256；既有独立审查的 95% Clopper–Pearson 区间为 [3.02172%,9.00548%]。秩控制要求策略预先固定、同一声明模拟总体、历史独立/可交换，且对校准样本随机性与新历史共同取边际；它不是给定当前校准集后的条件保证、逐历史概率、各层风险保证或连续参数域统一保证。未用评价结果反调门限。

## 分层结果

以下各层分母由评价历史的真拓扑/案例确定，属于描述性分层。

| 层 | 分母 | 方法 | 唯一正确保留 | 误拒 | 未决 | 错误唯一 | 选址失败 | 拟合失败 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(stratified)+'''

weak 层 random 的一次唯一正确保留是 730121（真 clade_12）：真 loss=74.98534762927387，三个错候选 loss=78.18523061739937、78.18603612507276、78.19502542529348。relative gate 全保留，absolute 门限 75.77255697515467 恰好留下真候选；本例保存在 exception_cases.json，不能用它推断弱支路稳定分离。

## 配对比较的方向与结论边界

refit 对 constant 的唯一正确保留：两者都成功 62，refit 单独成功 9，constant 单独成功 0，两者都未成功 185；收益 +3.515625 个百分点。既有审查的保守配对 95% 区间为 [−0.26212,7.04877] 个百分点，双侧 exact McNemar p=0.00390625。多方法、多终点或事后分层比较另有比较族问题，不能把该 p 值当拓扑正确概率。

collector 的 generic paired_binary improved/worsened 仅表示事件 0→1 / 1→0。对误拒、未决、拟合失败，0→1 是不利变化。refit 对 constant 的误拒为新增 2、消除 0，频率差 +0.78125 个百分点，p=0.5；不能写成误拒改善。legacy 的唯一正确保留 81 高于 refit 的 71，故未证明 refit 七方法整体最优。

AC replay 全部 7,168 为 verified，验证的是保存目标值与方程重放；全局最优证书为 0。代理距离和实际 signed margin 的关联较弱（refit raw proxy 的 Spearman ρ=0.12801），不能把代理目标直接解释为新增未来辨别信息。阻抗统计的单位先在历史内合并边，再跨历史汇总；star 的 internal 成分不适用，不记零。

原 numerical_bounds.csv 为 7,920,099 字节，未复制进紧凑包，但其 SHA256 和来源路径保存在 evidence_manifest.json。评价负 raw gap 6 处、最大相对 crossing=6.503972900110877e−7；校准负 gap 3 处，原值与 crossing 记录保留。容差接受不等于严格上下界有序，也不提供非线性 AC 设计最优性证书。

## 复核入口

- history_method_masks.csv.gz 保存全部 1,792 方法历史的 relative/absolute/combined/retained 掩码与事件；bit 顺序固定为 star、clade_01、clade_02、clade_12，最低位为 star，拒绝掩码 1=拒绝，retained_mask 1=保留。
- originals/evaluation_v3_audit/fits.csv.gz 与 actions.csv.gz 解压后逐字节等同原 CSV，分别为 7,168 / 1,792 数据行。原始与压缩文件 SHA256 均在 evidence_manifest.json。
- compact_counts.json 保存独立重算方法/各层计数、联合误拒历史 ID。该重算与完整 audit_summary 的所有事件计数及既有独立审查一致；5,376 组门掩码、26,880 个事件值比较零差异。
- verify_C_compact.py 仅用 Python 标准库验证归档 SHA256、解压 SHA256、校准 rank、全历史掩码与计数，不调用项目源码或优化器。命令：python verify_C_compact.py。
- source_manifests.json 保存 cal257 / eval513 个输入文件哈希、15 个核心冻结哈希、3 个后处理哈希，本次均读原文件匹配。

证据强度止于冻结模拟总体、有限四候选类和保存的局部拟合；没有单次拓扑正确概率、现场泛化、连续全域识别或全局 MLE 结论。
'''
write_new(OUT/'stage_C_results.md',report.encode('utf-8'))
derived=[]
for p in sorted(OUT.iterdir()):
    if p.is_file() and p.name not in {x['archive_path'] for x in archive_files} and p.name!='evidence_manifest.json':
        data=p.read_bytes(); derived.append({'archive_path':p.name,'bytes':len(data),'sha256':sha(data),'role':'derived compact evidence or reproducible archive script'})
manifest={'schema':'frozen_C_compact_evidence_v1','source_root':str(INPUT),'archive_root':str(OUT),
          'scope':'read-only frozen C result packaging; no new optimization, tuning, original-output mutation or git change',
          'statistical_unit':'one independent generator history; all method/candidate measurements paired within history',
          'frozen_stage_commit':frozen['stage_commit'],'policy_id':gates['calibration']['policy_id'],
          'population_id':gates['calibration']['population_id'],'calibration':gates['calibration'],
          'calibration_failure_infinity_preserved':True,'evaluation_denominator':256,
          'archive_files':archive_files,'derived_files':derived,'unarchived_large_raw_bounds':raw_record,
          'verification':verification,'all_methods_union_false_rejection':{'count':14,'denominator':256,'rate':14/256,
              'scope':'frozen nominal simulation population; history-wise union of all seven fixed methods'},
          'paired_event_semantics':{'0_to_1':'event incidence increases; adverse for false_rejection/unresolved/fit_failure',
                                    '1_to_0':'event incidence decreases; beneficial for false_rejection/unresolved/fit_failure'},
          'global_MLE_claim':False,'topology_correct_probability_reported':False,
          'manifest_self_hash':'omitted to avoid a circular self-hash; external SHA256 may be computed after creation'}
write_json('evidence_manifest.json',manifest)
print(json.dumps({'archive':str(OUT),'files':len(list(OUT.rglob('*'))),'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),'verification':verification},ensure_ascii=False))
