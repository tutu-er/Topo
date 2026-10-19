"""Post-hoc stored-array audit only; no project solver or AC imports."""
from __future__ import annotations

from collections import Counter
from datetime import datetime,timezone,timedelta
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from time import perf_counter

import numpy as np

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
OLD=BASE/'scalar_profile_probe_v1'
DOMAINS=('charge_fixed','v2g_fixed','charge_mobile','v2g_mobile','refined_v2g_mobile','capped_v2g_mobile')
SEEDS=tuple(range(990101,990105))
RANK_TOL=1e-8
SCORE_TOL=1e-8
manifest={}


def read(path):
    data=path.read_bytes()
    manifest[str(path.relative_to(BASE))]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def stat(values):
    a=[float(v) for v in values if v is not None and math.isfinite(v)]
    return dict(n=len(a),mean=statistics.mean(a) if a else None,
        median=statistics.median(a) if a else None,max=max(a) if a else None)


def sign(a,b):
    tolerance=RANK_TOL*max(1.,abs(a),abs(b))
    return 0 if abs(a-b)<=tolerance else 1 if a>b else -1


def rankings(rows,aggregate=False):
    groups={}
    if aggregate:
        nodes={}
        for r in rows:nodes.setdefault((r['slot'],r['site'],r['power_kw']),[]).append(r)
        for (slot,site,power),rr in nodes.items():
            if len(rr)!=12 or not all(r['success'] for r in rr):continue
            groups.setdefault(slot,[]).append(dict(site=site,power_kw=power,
                **{k:statistics.mean(r[k] for r in rr) for k in ('AC_attained_D','direct_D_interp','residual_interp_squared','old_QP')}))
    else:
        for r in rows:
            if r['success']:groups.setdefault((r['slot'],r['site'],r['name']),[]).append(r)
    out={k:dict(comparisons=0,target_ties=0,strict_targets=0,opposite_order=0,predicted_ties=0,order_recovered=0)
        for k in ('direct_D_interp','residual_interp_squared','old_QP')}
    for group in groups.values():
        for a,b in itertools.combinations(group,2):
            target=sign(a['AC_attained_D'],b['AC_attained_D'])
            for key,count in out.items():
                pred=sign(a[key],b[key]);count['comparisons']+=1
                if not target:count['target_ties']+=1;continue
                count['strict_targets']+=1;count['opposite_order']+=pred*target<0
                count['predicted_ties']+=pred==0;count['order_recovered']+=pred==target
    return out


def main():
    began=perf_counter();original=read(OLD/'records.json.gz')
    payload=read(OLD/'training_only_inputs.json.gz');old_summary=read(OLD/'summary.json');read(OLD/'freeze.json')
    fm=payload['future_meter'];T=payload['forecast_horizon'];n=3
    sigma=dict(v=np.broadcast_to(np.asarray(fm['v_sigma'],float),(T,n)),
        master_p=np.broadcast_to(np.asarray(fm['master_p_sigma'],float),(T,)),
        master_q=np.broadcast_to(np.asarray(fm['master_q_sigma'],float),(T,)))
    residual_cache={};queries={};reconstruction=[]
    for query in original['queries']:
        key=(query['slot'],query['site'],round(query['power_kw'],8));queries[key]=query
        result=query['result'];fits=result['saved_fits'];targets=result['reference_observations']
        for pair in result['pairs']:
            if not pair['success']:continue
            saved=fits[pair['name']];target=targets[pair['a']];slot=query['slot']
            residual=np.r_[((np.asarray(saved['predicted']['v'])-np.asarray(target['v']))/sigma['v'][slot]).ravel(),
                (np.asarray(saved['predicted']['master_p'])-np.asarray(target['master_p']))/sigma['master_p'][slot],
                (np.asarray(saved['predicted']['master_q'])-np.asarray(target['master_q']))/sigma['master_q'][slot]]
            value=.5*float(residual@residual);error=value-pair['added']
            if abs(error)>1e-8*max(1.,abs(value),abs(pair['added'])):
                raise ValueError('saved residual/SSE mismatch')
            residual_cache[key+(pair['name'],)]=residual
            reconstruction.append(dict(slot=slot,site=query['site'],power_kw=query['power_kw'],
                name=pair['name'],reconstructed_D=value,saved_D=pair['added'],absolute_error=abs(error)))
    midpoint=[]
    knots=original['config']['knots_kw']
    for old in original['midpoint_comparison']:
        row=dict(slot=old['slot'],site=old['site'],power_kw=old['power_kw'],name=old['name'],success=False)
        if not old['success']:
            row['failure']='original comparison incomplete';midpoint.append(row);continue
        left,right=next((a,b) for a,b in zip(knots,knots[1:]) if a<old['power_kw']<b)
        key=lambda p:(old['slot'],old['site'],round(p,8),old['name'])
        if not all(key(p) in residual_cache for p in (left,right,old['power_kw'])):
            row['failure']='stored endpoint or midpoint residual missing';midpoint.append(row);continue
        weight=(old['power_kw']-left)/(right-left)
        r=(1-weight)*residual_cache[key(left)]+weight*residual_cache[key(right)]
        target_r=residual_cache[key(old['power_kw'])];error_r=r-target_r
        value=.5*float(r@r);target=old['AC_attained_D']
        identity_error=float(target_r@error_r+.5*(error_r@error_r))-(value-target)
        if abs(identity_error)>1e-8*max(1.,abs(value),abs(target)):
            raise ValueError('residual-square error identity failed')
        row.update(success=True,AC_attained_D=target,direct_D_interp=old['linear_interpolation_D'],
            old_QP=old['original_linear_proxy_D'],residual_interp_squared=value,
            residual_interpolated=r.tolist(),residual_at_midpoint=target_r.tolist(),
            residual_interpolation_error_norm=float(np.linalg.norm(error_r)),
            residual_interpolation_signed_D_error=float(value-target),
            residual_interpolation_absolute_D_error=float(abs(value-target)),
            direct_interpolation_absolute_D_error=old['interpolation_absolute_error'],
            old_QP_absolute_D_error=old['proxy_absolute_error'],error_identity_abs=float(abs(identity_error)))
        midpoint.append(row)
    complete=[r for r in midpoint if r['success']]
    profile_summary=dict(planned_midpoint_tasks=288,complete_midpoint_tasks=len(complete),
        residual_reconstructed_nodes=len(reconstruction),reconstruction_max_D_error=max(r['absolute_error'] for r in reconstruction),
        direct_distance_error=stat([r['direct_interpolation_absolute_D_error'] for r in complete]),
        residual_squared_error=stat([r['residual_interpolation_absolute_D_error'] for r in complete]),
        old_QP_error=stat([r['old_QP_absolute_D_error'] for r in complete]),
        residual_better_than_direct=sum(r['residual_interpolation_absolute_D_error']<r['direct_interpolation_absolute_D_error']-1e-9 for r in complete),
        residual_better_than_old_QP=sum(r['residual_interpolation_absolute_D_error']<r['old_QP_absolute_D_error']-1e-9 for r in complete),
        residual_worse_than_old_QP=sum(r['residual_interpolation_absolute_D_error']>r['old_QP_absolute_D_error']+1e-9 for r in complete),
        residual_signed_error_counts=dict(positive=sum(r['residual_interpolation_signed_D_error']>1e-9 for r in complete),
            zero=sum(abs(r['residual_interpolation_signed_D_error'])<=1e-9 for r in complete),
            negative=sum(r['residual_interpolation_signed_D_error']< -1e-9 for r in complete)),
        per_task_rankings=rankings(midpoint),mean_task_rankings=rankings(midpoint,aggregate=True))
    gate_rows=[]
    dirs={'weak8':BASE/'development_v2',**{c:BASE/'training_support_dev_v1'/c for c in ('rich8','rich32','rich96')}}
    for condition,directory in dirs.items():
        read(directory/'freeze.json')
        for seed in SEEDS:
            raw=read(directory/f'trial_{seed}.json.gz');selection=raw['selection']
            selected={m['name']:m.get('power_kw') for m in raw['methods']};assert raw['source_unchanged']
            for method in DOMAINS:
                record=selection['methods'][method];prediction=record.get('raw_prediction',{});cap=record.get('cap')
                score=None
                if prediction.get('success'):
                    pairs=prediction['pairs'];weight=sum(p.get('weight',1/len(pairs)) for p in pairs)
                    if abs(weight-1)>1e-10:raise ValueError('non-normalized raw AC score')
                    score=sum(p.get('weight',1/len(pairs))*(p['added'] if cap is None else min(p['added'],cap)) for p in pairs)
                initial=record.get('initial_best_score');delta=None if score is None or initial is None else score-initial
                tol=SCORE_TOL*max(1.,abs(score or 0.),abs(initial or 0.))
                u=record.get('raw',{}).get('u');chosen=selected.get(method)
                same=None if u is None or chosen is None else bool(np.allclose(np.asarray(u).ravel(),np.asarray(chosen).ravel(),atol=1e-8,rtol=0.))
                control_key=None if u is None else np.round(np.asarray(u,float).ravel(),8)
                if control_key is not None:control_key[control_key==0.]=0.  # canonicalize signed zero
                control_hash=None if control_key is None else hashlib.sha256(control_key.astype('<f8').tobytes()).hexdigest()
                check=record.get('proxy_AC_check',{})
                gate_rows.append(dict(condition=condition,seed=seed,method=method,
                    raw_gate_passed=bool(record.get('raw_proxy_gate')),raw_profile_success=bool(prediction.get('success')),
                    raw_operation_feasible=bool(record.get('raw_operation',{}).get('feasible')),
                    initial_best_origin=record.get('initial_best_origin'),selected_origin=record.get('selected_origin'),
                    raw_AC_score=score,initial_best_AC_score=initial,selected_AC_score=record.get('selected_score'),
                    raw_minus_initial=delta,comparison_tolerance=tol,
                    raw_higher_than_best_initial=delta is not None and delta>tol,
                    raw_equal_to_selected_power=same,proxy_completed=record.get('raw',{}).get('completed'),
                    raw_control_sha256_rounded_8=control_hash,
                    proxy_gap=record.get('raw',{}).get('gap'),
                    max_witness_RMS=max((p.get('max_channel_RMS',0.) for p in check.get('pairs',[])),default=None),
                    witness_failures=sum(not p.get('gate_passed',False) for p in check.get('pairs',[])),
                    original_method_failures=record.get('failures'),
                    initial_origins=[p.get('origin') for p in record.get('initial_candidates',[])]))
    def gate_stats(rows):
        failed=[r for r in rows if not r['raw_gate_passed']]
        comparable=[r for r in failed if r['raw_operation_feasible'] and r['raw_profile_success'] and r['raw_minus_initial'] is not None]
        higher=[r for r in comparable if r['raw_higher_than_best_initial']]
        return dict(planned_proposals=len(rows),gate_pass=sum(r['raw_gate_passed'] for r in rows),gate_fail=len(failed),
            failed_but_comparable_AC=len(comparable),gate_failed_AC_improvement_count=len(higher),
            failed_same_power_as_selected=sum(r['raw_equal_to_selected_power'] is True for r in failed),
            failed_distinct_power=sum(r['raw_equal_to_selected_power'] is False for r in failed),
            failed_raw_minus_initial=stat([r['raw_minus_initial'] for r in comparable]),
            unique_improvement_curves=len({(r['condition'],r['seed'],r['raw_control_sha256_rounded_8']) for r in higher}),
            improvement_condition_histories=len({(r['condition'],r['seed']) for r in higher}),
            improvement_asset_seeds=sorted({r['seed'] for r in higher}),
            improvement_method_counts=dict(Counter(r['method'] for r in higher)),
            failed_positive_deltas=[r for r in higher],
            failed_selected_origin_counts=dict(Counter(str(r['selected_origin']) for r in failed)))
    gate_summary=dict(paired_asset_histories=4,training_conditions=4,total_paired_conditions=16,
        proposals=96,overall=gate_stats(gate_rows),
        by_condition={c:gate_stats([r for r in gate_rows if r['condition']==c]) for c in dirs},
        comparison='raw attained training-only full-AC profile objective minus best initial objective; relative tolerance 1e-8',
        no_actual_future_or_validation_scores_used=True)
    after={name:dict(bytes=(BASE/name).stat().st_size,sha256=hashlib.sha256((BASE/name).read_bytes()).hexdigest()) for name in manifest}
    unchanged=manifest==after
    if not unchanged:raise ValueError('original input evidence changed during read-only posthoc audit')
    created=datetime.now(timezone.utc)
    output=dict(schema='stored_residual_interpolation_and_gate_posthoc_v1',created_utc=created.isoformat(),
        created_beijing=created.astimezone(timezone(timedelta(hours=8))).isoformat(),
        posthoc=True,proposed_after_observing_direct_distance_interpolation_result=True,
        prior_result=old_summary,independent_evaluation=False,new_AC_calls=0,new_fit_calls=0,new_QP_calls=0,new_MILP_calls=0,
        new_EV_plans=0,original_evidence_unchanged=unchanged,source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=manifest,profile_summary=profile_summary,gate_summary=gate_summary,
        residual_reconstruction_checks=reconstruction,midpoint_comparison=midpoint,gate_rows=gate_rows,
        seconds=perf_counter()-began,
        scope='posthoc empirical approximation and saved acceptance bookkeeping; no strategy or bound deployment')
    OUT.joinpath('records.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    OUT.joinpath('summary.json').write_text(json.dumps({k:output[k] for k in
        ('schema','created_beijing','posthoc','proposed_after_observing_direct_distance_interpolation_result','new_AC_calls',
        'new_fit_calls','new_QP_calls','new_MILP_calls','new_EV_plans','original_evidence_unchanged','profile_summary','gate_summary','seconds')},
        ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    OUT.joinpath('input_unchanged_proof.json').write_text(json.dumps(dict(before=manifest,after=after,
        exact_same_sha256=unchanged,source_code_sha256=output['source_code_sha256']),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    p=profile_summary;g=gate_summary['overall'];fmt=lambda v:'—' if v is None else f'{v:.6g}'
    text=f'''# 后验诊断：残差插值后平方，与AC误差门

本构造是在看到前一次“直接距离插值误差较大”之后提出的后验分析。全部计算只使用保存的记录；新AC、拟合、QP、MILP和EV计划均为0。原54节点记录、16个条件raw与冻结文件的读取前后SHA完全一致，证明见input_unchanged_proof.json。它不是新独立评价，也没有改变策略。

## 1. 保留平方曲率的插值

从每个保存的源profile拟合，用预测V/根P/Q减参考观测并除公开标准差，重构5维白化残差r。648节点任务的半SSE重构最大差为{p['reconstruction_max_D_error']:.6g}。对两端点残差作线性插值rI，再算DI=||rI||²/2；端点仍是原来5个预声明功率，检验仍是4个中点、全部288任务。

| 中点绝对距离误差 | 均值 | 中位数 | 最大值 |
|---|---:|---:|---:|
| 直接距离D插值 | {fmt(p['direct_distance_error']['mean'])} | {fmt(p['direct_distance_error']['median'])} | {fmt(p['direct_distance_error']['max'])} |
| 残差r插值后平方 | {fmt(p['residual_squared_error']['mean'])} | {fmt(p['residual_squared_error']['median'])} | {fmt(p['residual_squared_error']['max'])} |
| 原一次线性残差＋QP | {fmt(p['old_QP_error']['mean'])} | {fmt(p['old_QP_error']['median'])} | {fmt(p['old_QP_error']['max'])} |

残差插值比旧QP误差更小{p['residual_better_than_old_QP']}/288，更大{p['residual_worse_than_old_QP']}/288；逐任务与平均任务排序的翻转数、并列及完整分母见summary.json。残差插值不一定对应某个合法源的真实AC残差，因而不是距离上界或下界；这次样本误差也不是区间误差证书。

直觉：若原残差近似随功率线性变化，则距离本来近似二次。直接对距离作线性插值丢掉曲率，先插值残差再平方保留了它。后验改善只能作为下一步构造依据，不能算预先声明的新方法优势。

## 2. witness残差门是否真的挡住了更好的预测计划

读取weak8、rich8、rich32、rich96四条件×四配对资产×六个设计域，共96个proposal。比较每个raw计划的完整AC取得目标与它的最好初值，使用该方法原来的平均或cap20目标；未使用实际future/validation评分。保持全部失败和不可比情形。

AC门通过{g['gate_pass']}/96，失败{g['gate_fail']}/96；失败中可比的完整AC目标与可行操作有{g['failed_but_comparable_AC']}项。门失败但AC目标高于最好初值的计划有{g['gate_failed_AC_improvement_count']}项。失败中raw功率与最终已选功率相同{g['failed_same_power_as_selected']}项，不同{g['failed_distinct_power']}项。完整来源、分数差和相对1e−8容差都逐项保存。

这{g['gate_failed_AC_improvement_count']}个方法proposal含跨域重复，按条件/seed/功率曲线去重为{g['unique_improvement_curves']}条，涉及{g['improvement_condition_histories']}个配对条件、资产seed {g['improvement_asset_seeds']}；不是10个独立新历史。方法分布为{g['improvement_method_counts']}。例如rich32/990102的charge_mobile从64.6878提高到129.4105，差64.7227，而witness RMS=.31249被.25门否决；同历史v2g_mobile和refined共享另一条曲线，差36.5934。它们的实际未来效果未在此后验审计中计算。

门检查的是同一个代理源解释在AC中的残差误差；最小化后的AC距离还允许源再调。即使残差向量相差明显，距离可能变化很小：若e是残差误差，距离变化等于r'e+||e||²/2；接近源补偿方向的误差还可能被再次拟合吸收。因此误差门和距离精度不能互相代替。

## 3. 可核查的候选接受准则建议（尚未实施）

1. 先检查EV路线、功率、SOC、人员、源盒及各预测拓扑电压的可行性；全部12任务保留，任何必要任务失败则该候选不参加完整目标比较。
2. 对raw与确定性初值使用相同的固定训练R/X、P/Q输入和源拟合流程，独立重算完整AC取得目标。比较对象必须是同一个平均或cap目标。
3. 若raw的完整AC目标超过最好初值一个预先声明的数值容差，可将它记为“相同数值流程下的预测目标改进”；平局保留初值。该判断是局部取得值比较，不是全局AC最优性或实际排错收益的保证。
4. witness误差和代理gap继续记录，用于触发重新展开、缩小搜索或标记代理证据不足。若完整AC已核查候选，可单独研究把残差门从硬否决改为这类诊断；需要新开发与重新冻结。若本次没有真正挡住更好的候选，就没有证据把现有性能问题归因于门。
5. 更强的接受证明需要非线性profile的有效上下界，例如新计划的全任务下界高于旧计划上界；当前局部source拟合与插值均未提供这种证书。

这些建议没有修改任何已有接受规则，也没有新增优化结果。后续首先验证残差插值的活动边界/源分支问题和可用误差界，并继续检查预测指标与实际排错的关系。
'''
    OUT.joinpath('README.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(profile_summary=p,gate_summary={k:v for k,v in g.items() if k!='failed_positive_deltas'},original_evidence_unchanged=unchanged,
        seconds=output['seconds']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
