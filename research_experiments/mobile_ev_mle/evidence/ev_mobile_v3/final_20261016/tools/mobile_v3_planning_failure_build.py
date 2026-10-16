"""Read four saved failure histories only; no model, fitter, solver imports."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent/'mobile_v3_pilot2'
TARGET=OUT/'planning_failure_audit_v1.json'
if TARGET.with_suffix('.md').exists():raise FileExistsError(TARGET)
summary_path=OUT/'summary_final_v1.json'
summary=json.loads(summary_path.read_bytes())
selected=[r for r in summary['phases']['evaluation']['rows'] if any(r['methods'][m]['planning_status']=='failed' for m in ('projection_fixed','projection_mobile'))]
assert len(selected)==4 and all(r['spec']['bank']=='n4' for r in selected)
rows=[];profiles={};source_files=[]
physical_keys=('usable','physical_voltage_feasible','source_box_feasible','line_domain_feasible','ac_equation_residual','power_balance_residual')


def physical_fields(row):return {k:row.get(k) for k in physical_keys}


def failed_fit(ref,b,fit):
    optimizer=fit.get('optimizer') or {};diagnostics=optimizer.get('diagnostics') or {}
    reason=fit.get('failure') or fit.get('optimizer_error') or optimizer.get('message') or ref.get('failure')
    if fit.get('attempted') is False:
        category='unattempted_wall_budget' if 'budget' in str(reason).lower() else 'unattempted_reference_failure'
    elif fit.get('fixed_source_success') is False:category='fixed_source_witness_failure_not_fully_logged'
    elif optimizer.get('status')==0 and 'maximum number of function evaluations' in str(optimizer.get('message','')).lower():category='joint_optimizer_max_nfev'
    elif fit.get('optimizer_error'):category='joint_optimizer_exception'
    elif fit.get('success') is False:category='joint_optimizer_other_failure'
    else:category='gradient_reliability_gate_failure'
    return dict(a=ref['a'],b=b,category=category,reason=reason,
        attempted=fit.get('attempted'),success=fit.get('success'),fixed_source_success=fit.get('fixed_source_success'),
        gradient_reliable=fit.get('gradient_reliable'),optimization_failure=fit.get('optimization_failure'),
        selected_origin=fit.get('selected_origin'),selected_loss=fit.get('loss'),selected_physical=physical_fields(fit),
        feasible_witness_loss=(fit.get('feasible_witness') or {}).get('loss'),
        feasible_witness_physical=physical_fields(fit.get('feasible_witness') or {}),
        optimizer={k:optimizer.get(k) for k in ('success','status','message','nfev','optimality','loss')},
        optimizer_physical=physical_fields(fit.get('optimizer_recomputed_score') or {}),
        diagnostics={k:diagnostics.get(k) for k in ('failure_kind','solver','solver_calls','total_nfev','active_bound_count','local_jacobian_rank','local_shared_rank','local_source_rank','seconds','ac_equation_residual','power_balance_residual')},
        active_ratio_bound_count=(diagnostics.get('ratio_constraints') or {}).get('active_ratio_bound_count'),
        restarts=[{k:r.get(k) for k in ('index','success','solver_success','status','message','nfev','optimality','initial_loss','loss','physical_voltage_feasible','ratio_domain_feasible')} for r in diagnostics.get('restarts',[])])


for selected_row in selected:
    spec=selected_row['spec'];path=OUT/'evaluation'/(str(spec['index']).zfill(3)+'.json.gz')
    data=path.read_bytes();raw=json.loads(gzip.decompress(data));assert raw['spec']==spec
    source=dict(path=str(path),sha256=sha256(data).hexdigest(),bytes=len(data));source_files.append(source)
    row=dict(spec=spec,true_name=raw['true_name'],methods=[])
    for method in ('projection_fixed','projection_mobile'):
        policy=raw['planning'][method]['raw'];executed=raw['methods'][method]
        assert not policy['success'] and executed['planning_success'] is False
        methodrow=dict(name=method,planning_success=policy['success'],failure=policy.get('failure'),
            seconds=policy.get('seconds'),soft_budget_seconds=policy['configuration']['total_seconds'],
            fallback_origin=policy.get('fallback_origin'),selected_origin=policy.get('selected_origin'),
            recorded_proposal_steps=len(policy.get('steps',[])),initials=[],
            executed_success=executed['success'],operation_feasible=executed['operation']['audit']['feasible'],
            final_evaluation_complete=executed['evaluation']['complete'],
            final_joint_success={b:c['joint']['post']['success'] for b,c in executed['evaluation']['comparisons'].items()},
            final_joint_optimizer_success={b:(c['joint']['post'].get('optimizer') or {}).get('success') for b,c in executed['evaluation']['comparisons'].items()})
        fallback_name='projection_fixed' if policy.get('fallback_origin')=='fixed_parent' else ('square_mobile' if method.endswith('mobile') else 'square_fixed')
        methodrow['fallback_matches_saved_parent_power']=executed['power_kw']==raw['methods'][fallback_name]['power_kw']
        for index,initial in enumerate(policy.get('initials',[])):
            profile=initial.get('profile') or {};payload=json.dumps(profile,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            key=sha256(payload).hexdigest()
            item=dict(origin=initial.get('origin'),success=initial.get('success'),failure=initial.get('failure'),
                operation_feasible=(initial.get('operation') or {}).get('feasible'),
                profile_sha256=key,profile_complete=profile.get('success'),profile_deadline_reached=profile.get('deadline_reached'),
                profile_seconds=profile.get('seconds'),joint_fit_calls=profile.get('joint_fit_calls'),
                fixed_source_witness_fit_calls=profile.get('fixed_source_witness_fit_calls'))
            methodrow['initials'].append(item)
            if key not in profiles:
                failures=[];allfits=[]
                for ref in profile.get('references',[]):
                    for b,fit in ref.get('fits',{}).items():
                        allfits.append((ref['a'],b,fit))
                        if not (fit.get('success') and fit.get('gradient_reliable')):failures.append(failed_fit(ref,b,fit))
                profiles[key]=dict(spec=spec,profile_sha256=key,appearances=[],complete=profile.get('success'),
                    deadline_reached=profile.get('deadline_reached'),reference_count=len(profile.get('references',[])),
                    reference_failures=[dict(a=r['a'],failure=r.get('failure')) for r in profile.get('references',[]) if r.get('failure')],
                    fit_denominator=len(allfits),successful_fit_count=sum(f.get('success') and f.get('gradient_reliable') for a,b,f in allfits),
                    attempted_fit_count=sum(f.get('attempted') is True for a,b,f in allfits),
                    fixed_source_witness_success_count=sum(f.get('fixed_source_success') is True for a,b,f in allfits),
                    all_attempted_saved_states_usable=all(f.get('usable') is True for a,b,f in allfits if f.get('attempted') is True),failed_fits=failures)
            profiles[key]['appearances'].append(dict(method=method,path=f'planning.{method}.raw.initials[{index}].profile'))
        row['methods'].append(methodrow)
    rows.append(row)

unique=list(profiles.values());failures=[f for p in unique for f in p['failed_fits']]
report=dict(schema='mobile_v3_predictive_planning_failure_audit_v1',created_utc=datetime.now(timezone.utc).isoformat(),
    scope='Only four independently generated evaluation histories whose saved projection planning failed; no refit, AC or policy change.',
    history_denominator=4,method_slot_denominator=8,unique_saved_predictive_profiles=len(unique),
    unique_predictive_fit_denominator=sum(p['fit_denominator'] for p in unique),unique_failed_fits=len(failures),
    attempted_fit_count=sum(p['attempted_fit_count'] for p in unique),
    successful_fixed_source_witness_count=sum(p['fixed_source_witness_success_count'] for p in unique),
    distinct_failed_problem_count=len({(p['spec']['seed'],f['a'],f['b']) for p in unique for f in p['failed_fits'] if f['category']=='joint_optimizer_max_nfev'}),
    shared_profile_groups=sum(len(p['appearances'])>1 for p in unique),
    failure_categories=dict(Counter(f['category'] for f in failures)),
    method_fallbacks=sum(not m['planning_success'] for r in rows for m in r['methods']),
    actual_operation_successes=sum(m['operation_feasible'] for r in rows for m in r['methods']),
    final_joint_successes=sum(v for r in rows for m in r['methods'] for v in m['final_joint_success'].values()),
    profile_duplication='Three nondeadline failed profiles are cached and reused by fixed/mobile. Index12 deadline profiles are not cached and are recomputed; distinct JSON SHA and attempted slots are retained. Method appearances are not independent histories.',
    distinction='The required predictive joint optimizer convergence gate failed before proposal generation. Feasible attained explanations remain; executable fallback and its final observed-data joint fits are separate.',
    histories=rows,profiles=unique,source_files=source_files,
    summary_source=dict(path=str(summary_path),sha256=sha256(summary_path.read_bytes()).hexdigest()),
    audited_source_contract=dict(JointPredictiveEngine='ev_mobile_v2_design.py:170-263; gradient_reliable requires joint success and fixed-source success; complete nondeadline profiles, including failures, are cached.',
        stage='ev_systematic_refit.py:160-190; optimizer failure keeps best feasible attainment but success=False.',
        fallback='ev_mobile_v3_design.py:57-66; incomplete parent profile returns feasible fallback before any proposal.'),
    generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
TARGET.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,allow_nan=False,indent=2),encoding='utf8')

text=['# 完整 AC 规划失败与可执行回退的区别','',
'独立评价中的 N4 四条历史使 projection_fixed/mobile 各有四条规划失败：index 8、11（H8），12、15（H96）。两方法合计八个失败槽仍计入分母。index 8、11、15 的移动方法复用固定方法的同一失败缓存；index 12 触发墙钟门，失败 profile 不缓存，移动方法又计算了一次同曲线。独立历史仍只有四条。','',
f"五个实际计算 profile 共记录 {report['unique_predictive_fit_denominator']} 个参考—解释槽，其中75项已尝试、70项成功。5次 joint 达到 nfev=250（对应4个独立参考—解释问题，index12重复一次）；另5槽在60秒软墙钟门后未尝试。两类原因分开记录。",'',
'|index/H/真值|失败的参考→解释|nfev|status|optimality|固定线路源成功|解释物理可用|原因|','|---|---|---:|---:|---:|---|---|---|']
for p in unique:
    for f in p['failed_fits']:
        s=p['spec'];o=f['optimizer'];optimality='—' if o['optimality'] is None else f"{o['optimality']:.6g}"
        text.append(f"|{s['index']}/H{s['H']}/{next(r['true_name'] for r in rows if r['spec']==s)}|{f['a']}→{f['b']}|{o['nfev']}|{o['status']}|{optimality}|{f['fixed_source_success']}|{f['selected_physical']['usable']}|{f['reason']}|")
text.extend(['',
'所有75项已尝试拟合的固定线路源拟合均成功，保存解释满足电压、源界与线路域。五次失败 joint 的 status=0、nfev=250，最优性指标分别约3.02、6.19、31.49（两次）、9.02；程序保留其物理可行解释，但优化 success=False，梯度可靠标记随之为 false。全 K×K 预测评分要求每项成功，因而不能得到完整规划评分。index12还出现5个明确记录为未尝试的槽；它们没有物理状态，不能写成物理失败。','',
'这些方法没有进入 MILP 新提案步骤（steps=0）。固定方法执行 square_parent，移动方法执行 fixed_parent；逐时功率与对应已保存父计划完全一致。八个回退动作均满足操作/SOC核查，随后对实际计量的 32 个候选 joint 拟合均成功。失败的是规划时的某个预测解释计算，物理操作和最终实际计量拟合是另两项检查。','',
'JSON 保存每个失败的 optimizer message/nfev/optimality、重启摘要、物理可用标记、可行证据损失、profile来源与字节SHA，以及八个方法槽的回退和最终拟合状态。未追加拟合、延长迭代或修改策略。',''])
TARGET.with_suffix('.md').write_text('\n'.join(text),encoding='utf8')
print(json.dumps({k:report[k] for k in ('history_denominator','method_slot_denominator','unique_saved_predictive_profiles','unique_predictive_fit_denominator','unique_failed_fits','failure_categories','actual_operation_successes','final_joint_successes')}),flush=True)
