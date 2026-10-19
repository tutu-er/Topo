"""Bounded training-only scalar AC profile audit; no new controller is deployed.

Known rich32 development asset 990102 is a targeted approximation diagnostic.
The five knots and four midpoints are scalar FUNCTION queries, not feasible EV
plans. Complete nonlinear source profiles are attained local fits. Neither
their interpolation nor the sampled error is a certified global bound.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from . import gradient_price_study as io
from . import likelihood as lk
from .ev_fixed_train_design import (CONFIG as DEVICE,build_fixed_contexts,
                                    evaluate_fixed_action)
from .ev_fixed_train_fit import subset_model
from .ev_fixed_train_review import review_fixed_fit
from .information_context import meter_from_record,meter_to_record

HERE=Path(__file__).resolve().parent
CONFIG=dict(version='fixed_training_scalar_AC_profile_probe_v1',development_seed=990102,
    training_condition='rich32',slots=[0,1],sites=[0,1,2],
    knots_kw=[-5.7,-2.85,0.,2.85,5.7],midpoints_kw=[-4.275,-1.425,1.425,4.275],
    scalar_queries=54,required_tasks_per_query=12,
    source_profile=dict(multistart=2,max_nfev=180,tol=1e-10),
    source_p_bounds=[-.5,.5],source_q_bounds=[-.5,.5],
    base_kw=20.,ranking_relative_tolerance=1e-8,
    sum_check_relative_tolerance=1e-6,
    direction_selected_from_truth=False,validation_used=False,future_noise_used=False,
    independent_evaluation=False,complete_action_enumeration=False,
    feasible_EV_action_claim=False,new_policy_deployed=False,
    nonlinear_profile_is_global_minimum_certificate=False,
    interpolation_is_bound_certificate=False)


def digest(value):
    return hashlib.sha256(json.dumps(io.serial(value),sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def training_only_payload(record):
    """Whitelist is the sole computation boundary; all other raw fields ignored."""
    fields=('candidate','r','x','p','q','source_p','source_q','loss','success')
    return dict(training=record['training'],training_meter=record['training_meter'],
        future_meter=record['future_meter'],candidates=record['candidates'],
        training_fits=[{k:r[k] for k in fields} for r in record['training_result']['fits']],
        forecast_horizon=6,optimizer_seed=int(record['seeds']['policy'])+9000)


def restore_inputs(payload):
    if set(payload)!={'training','training_meter','future_meter','candidates','training_fits',
                       'forecast_horizon','optimizer_seed'}:
        raise ValueError('computation accepts only the training/public-model whitelist')
    training=lk.Observations(**payload['training']);tm=meter_from_record(payload['training_meter'])
    fm=meter_from_record(payload['future_meter'])
    bank=[lk.RadialCandidate.from_supports(c['name'],training.n,c['supports']) for c in payload['candidates']]
    fits={r['candidate']:SimpleNamespace(**{k:np.asarray(v,float) if k in
        ('r','x','p','q','source_p','source_q') else v for k,v in r.items()}) for r in payload['training_fits']}
    if len(bank)!=4 or len(bank)*(len(bank)-1)!=12 or not all(c.name in fits and fits[c.name].success for c in bank):
        raise ValueError('all four training candidates and fits are required')
    return training,tm,fm,bank,fits


def sliced_future_meter(model,T,n,slot):
    scaffold=lk.Observations(p=np.zeros((T,n)),q=np.zeros((T,n)),v=np.ones((T,n)),
        root_v=np.ones(T),master_p=np.zeros(T),master_q=np.zeros(T))
    return subset_model(model,scaffold,[int(slot)])


def _reference_observations(context,power):
    T,n=power.shape;meter=context['future_meter'];p=context['background']['p']+power/context['base_kw']
    q=context['background']['q'];targets={};calls=0
    for candidate in context['bank']:
        name=candidate.name;line=context['lines'][name];forecast=context['forecasts'][name]
        calls+=1
        phy=lk.ac_predict(candidate,line['r'],line['x'],p,q,forecast['source_p'],forecast['source_q'],context['root_v'])
        targets[name]=lk.Observations(p=p+lk._array(meter.bias_p,(T,n),'bias_p'),
            q=q+lk._array(meter.bias_q,(T,n),'bias_q'),v=phy['v']+lk._array(meter.bias_v,(T,n),'bias_v'),
            root_v=context['root_v'],master_p=phy['master_p'],master_q=phy['master_q'])
    return targets,calls


def audited_profile(context,power,seed):
    result=evaluate_fixed_action(context,power,fit_options=CONFIG['source_profile'],seed=seed)
    target,calls=_reference_observations(context,np.asarray(power,float))
    bank={c.name:c for c in context['bank']};fits={};reviews=[]
    for pair in result['pairs']:
        fitted=result.get('fit_results',{}).get(pair['name'])
        if fitted is None:continue
        row=io.fit_record(bank[pair['b']],fitted);fits[pair['name']]=row
        if not fitted.success:continue
        try:
            check=review_fixed_fit(bank[pair['b']],target[pair['a']],context['future_meter'],io.serial(row),
                trained_r=context['lines'][pair['b']]['r'],trained_x=context['lines'][pair['b']]['x'])
            reviews.append(dict(name=pair['name'],success=True,review=check))
        except (ValueError,AssertionError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            reviews.append(dict(name=pair['name'],success=False,error=type(exc).__name__+':'+str(exc)))
    result['saved_fits']=fits;result['reference_observations']={k:io.observations_record(v) for k,v in target.items()}
    result['independent_fit_reviews']=reviews;result['audit_target_forward_AC_calls']=calls
    result['independent_fit_replay_count']=len(reviews)
    result['independent_fit_reviews_passed']=len(reviews)==len(context['pairs']) and all(r['success'] for r in reviews)
    return result


def _stat(values):
    a=np.asarray([v for v in values if v is not None],float)
    return dict(n=len(a),mean=float(np.mean(a)) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,max=float(np.max(a)) if len(a) else None)


def interpolation_comparison(queries,knots=None):
    """At predeclared held-out midpoints, preserve all 24*12 task outcomes."""
    knots=CONFIG['knots_kw'] if knots is None else knots
    lookup={(q['slot'],q['site'],round(q['power_kw'],8)):q for q in queries};rows=[]
    for slot in CONFIG['slots']:
        for site in CONFIG['sites']:
            for left,right in zip(knots,knots[1:]):
                mid=(left+right)/2;key=lambda p:(slot,site,round(p,8))
                lo,hi,target=(lookup[key(p)] for p in (left,right,mid))
                l={r['name']:r for r in lo['result']['pairs']};h={r['name']:r for r in hi['result']['pairs']}
                for t in target['result']['pairs']:
                    name=t['name'];row=dict(slot=slot,site=site,power_kw=mid,name=name,success=False,
                        endpoint_success=[l[name].get('success'),h[name].get('success')],midpoint_success=t.get('success'))
                    if t.get('success') and l[name].get('success') and h[name].get('success') and t.get('proxy_QP',{}).get('success'):
                        observed=float(t['added']);interpolated=.5*(l[name]['added']+h[name]['added'])
                        linear_proxy=.5*t['proxy_QP']['primal_upper']
                        row.update(success=True,AC_attained_D=observed,linear_interpolation_D=float(interpolated),
                            original_linear_proxy_D=float(linear_proxy),interpolation_signed_error=float(interpolated-observed),
                            interpolation_absolute_error=float(abs(interpolated-observed)),
                            proxy_signed_error=float(linear_proxy-observed),proxy_absolute_error=float(abs(linear_proxy-observed)))
                    else:row['failure']='required endpoint/midpoint AC or midpoint proxy QP failed'
                    rows.append(row)
    return rows


def ranking_agreement(rows,*,aggregate=False):
    groups={}
    if aggregate:
        nodes={}
        for r in rows:nodes.setdefault((r['slot'],r['site'],r['power_kw']),[]).append(r)
        for (slot,site,power),rr in nodes.items():
            if len(rr)!=12 or not all(x['success'] for x in rr):continue
            groups.setdefault(slot,[]).append(dict(site=site,power_kw=power,
                **{k:float(np.mean([x[k] for x in rr])) for k in ('AC_attained_D','linear_interpolation_D','original_linear_proxy_D')}))
    else:
        for r in rows:
            if r['success']:groups.setdefault((r['slot'],r['site'],r['name']),[]).append(r)
    result={name:dict(comparisons=0,target_ties=0,strict_targets=0,opposite_order=0,predicted_ties=0,order_recovered=0)
        for name in ('linear_interpolation_D','original_linear_proxy_D')}
    def sign(x,y):
        eps=CONFIG['ranking_relative_tolerance']*max(1.,abs(x),abs(y))
        return 0 if abs(x-y)<=eps else 1 if x>y else -1
    for group in groups.values():
        for a,b in itertools.combinations(group,2):
            target=sign(a['AC_attained_D'],b['AC_attained_D'])
            for metric,c in result.items():
                pred=sign(a[metric],b[metric]);c['comparisons']+=1
                if not target:c['target_ties']+=1;continue
                c['strict_targets']+=1;c['opposite_order']+=pred*target<0
                c['predicted_ties']+=pred==0;c['order_recovered']+=pred==target
    return result


def run_probe(payload):
    """Only training/public payload reaches numerical queries."""
    began=perf_counter();train,tm,fm,bank,fits=restore_inputs(payload);T=payload['forecast_horizon'];n=train.n
    reference=np.zeros((T,n));reference[:,0]=(DEVICE['terminal_energy_kwh']-DEVICE['initial_energy_kwh'])/(
        DEVICE['eta_charge']*T*DEVICE['dt_hours'])
    full=build_fixed_contexts(train,bank,fits,tm,fm,-5.7,5.7,reference)
    if not full['success']:raise ValueError('training-only full context failed: '+str(full['diagnostics']))
    contexts={};sigma_keys={};matrix_checks=[]
    for slot in CONFIG['slots']:
        meter=sliced_future_meter(fm,T,n,slot);sigma_keys[digest(meter_to_record(meter))]=slot
        ctx=build_fixed_contexts(train,bank,fits,tm,meter,-5.7,5.7,reference[[slot]])
        if not ctx['success']:raise ValueError('scalar context failed')
        contexts[slot]=ctx
        rid=np.r_[np.arange(slot*n,(slot+1)*n),T*n+slot,T*(n+1)+slot]
        uid=np.arange(slot*n,(slot+1)*n);sid=np.arange(2*slot,2*slot+2)
        for a,b in zip(ctx['pairs'],full['pairs']):
            e=max(float(np.max(abs(a.c-b.c[rid]))),float(np.max(abs(a.B-b.B[np.ix_(rid,uid)]))),
                float(np.max(abs(a.A0-b.A0[np.ix_(rid,sid)]))))
            matrix_checks.append(dict(slot=slot,name=a.name,max_matrix_error=e,passed=e<1e-9))
    points=sorted(set(CONFIG['knots_kw']+CONFIG['midpoints_kw']));queries=[];cache={}
    for slot in CONFIG['slots']:
        for site in CONFIG['sites']:
            for power in points:
                u=np.zeros((1,n));u[0,site]=power
                result=audited_profile(contexts[slot],u,payload['optimizer_seed'])
                row=dict(slot=slot,site=site,power_kw=power,role='knot' if power in CONFIG['knots_kw'] else 'held_out_midpoint',
                    scalar_function_query=True,feasible_EV_action_claim=False,result=result)
                queries.append(row);cache[slot,site,round(power,8)]=result
    curves=[dict(name='predeclared_curve_A',sites=[0,1,2,0,1,2],powers=[5.7,-2.85,2.85,0.,5.7,-5.7]),
        dict(name='predeclared_curve_B',sites=[2,1,0,2,1,0],powers=[-4.275,1.425,4.275,-1.425,-2.85,2.85])]
    sum_checks=[]
    for curve in curves:
        power=np.zeros((T,n));pieces=[]
        for t,(site,p) in enumerate(zip(curve['sites'],curve['powers'])):
            power[t,site]=p;sig=digest(meter_to_record(sliced_future_meter(fm,T,n,t)))
            if sig not in sigma_keys:raise ValueError('unqueried time precision; no assumed caching equivalence')
            pieces.append(cache[sigma_keys[sig],site,round(p,8)])
        joint=audited_profile(full,power,payload['optimizer_seed']);rows=[]
        by_part=[{r['name']:r for r in x['pairs']} for x in pieces]
        for p in joint['pairs']:
            complete=bool(p['success'] and all(x[p['name']]['success'] for x in by_part))
            row=dict(name=p['name'],success=complete)
            if complete:
                value=sum(x[p['name']]['added'] for x in by_part);error=value-p['added']
                tol=CONFIG['sum_check_relative_tolerance']*max(1.,abs(value),abs(p['added']))
                row.update(summed_scalar_attained_D=float(value),full_horizon_attained_D=float(p['added']),
                    signed_error=float(error),absolute_error=float(abs(error)),tolerance=float(tol),passed=bool(abs(error)<=tol))
            rows.append(row)
        sum_checks.append(dict(**curve,power_kw=power,nonexecuted_diagnostic_curve=True,
            full_result=joint,pairs=rows,all_passed=all(r.get('passed',False) for r in rows)))
    comparison=interpolation_comparison(queries)
    all_results=[x['result'] for x in queries]+[x['full_result'] for x in sum_checks]
    counters={k:sum(x['cost'].get(k,0) for x in all_results) for k in ('fit_calls','solver_calls','ac_calls')}
    counters.update(context_AC_calls=full['diagnostics']['ac_calls']+sum(c['diagnostics']['ac_calls'] for c in contexts.values()),
        audit_target_forward_AC_calls=sum(x['audit_target_forward_AC_calls'] for x in all_results),
        independent_fit_replays=sum(x['independent_fit_replay_count'] for x in all_results))
    summary=dict(logical_queries=len(queries),planned_queries=54,planned_query_tasks=648,
        completed_query_tasks=sum(r['success'] for q in queries for r in q['result']['pairs']),
        failed_query_tasks=[dict(slot=q['slot'],site=q['site'],power_kw=q['power_kw'],name=r['name'],failure=r.get('failure'))
            for q in queries for r in q['result']['pairs'] if not r['success']],
        queries_with_all_tasks_complete=sum(q['result']['success'] for q in queries),
        independent_fit_replay_all_pass=all(x['independent_fit_reviews_passed'] for x in all_results),
        planned_midpoint_tasks=288,complete_midpoint_tasks=sum(r['success'] for r in comparison),
        interpolation_absolute_error=_stat([r.get('interpolation_absolute_error') for r in comparison]),
        original_proxy_absolute_error=_stat([r.get('proxy_absolute_error') for r in comparison]),
        interpolation_better_count=sum(r['interpolation_absolute_error']<r['proxy_absolute_error']-1e-9 for r in comparison if r['success']),
        interpolation_worse_count=sum(r['interpolation_absolute_error']>r['proxy_absolute_error']+1e-9 for r in comparison if r['success']),
        task_rankings=ranking_agreement(comparison),mean_task_rankings=ranking_agreement(comparison,aggregate=True),
        matrix_slicing_all_pass=all(r['passed'] for r in matrix_checks),sum_checks_all_pass=all(x['all_passed'] for x in sum_checks),
        sum_check_max_error=max((r.get('absolute_error',0.) for s in sum_checks for r in s['pairs']),default=None),
        costs=counters,seconds=perf_counter()-began,
        scope='local attained nonlinear scalar-source profiles and empirical interpolation checks; no AC bound or new policy')
    return dict(config=CONFIG,context_diagnostics=full['diagnostics'],scalar_context_diagnostics={k:v['diagnostics'] for k,v in contexts.items()},
        queries=queries,midpoint_comparison=comparison,matrix_slicing_checks=matrix_checks,sum_checks=sum_checks,summary=summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    raw_bytes=args.input.read_bytes();record=json.loads(__import__('gzip').decompress(raw_bytes))
    if record.get('seed')!=CONFIG['development_seed'] or record.get('condition')!='rich32':
        raise ValueError('this bounded diagnostic is predeclared only for rich32 seed990102')
    payload=training_only_payload(record);saved_sources=record['source_sha256'];del record
    sources=dict(saved_sources)
    for name,expected in saved_sources.items():
        path=HERE.parents[1]/name if name.startswith('rnj_wzzt_core/') else HERE/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('frozen source drift: '+name)
    sources[Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'freeze.json').exists():raise ValueError('existing diagnostic retained; choose another directory')
    freeze=dict(config=CONFIG,source_sha256=sources,input_raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        numerical_payload_sha256=digest(payload),payload_fields=list(payload),
        protocol_frozen_before_queries=True,known_development_case=True,independent_evaluation=False)
    io.write_json(args.output/'freeze.json',freeze)
    io.write_json(args.output/'training_only_inputs.json.gz',payload,compressed=True)
    snapshot={name:base64.b64encode((HERE.parents[1]/name if name.startswith('rnj_wzzt_core/') else HERE/name).read_bytes()).decode() for name in sources}
    io.write_json(args.output/'source_snapshot.json.gz',dict(source_sha256=sources,source_base64=snapshot),compressed=True)
    result=run_probe(payload)
    after={name:hashlib.sha256((HERE.parents[1]/name if name.startswith('rnj_wzzt_core/') else HERE/name).read_bytes()).hexdigest() for name in sources}
    result['source_unchanged']=after==sources
    io.write_json(args.output/'records.json.gz',result,compressed=True)
    io.write_json(args.output/'summary.json',dict(result['summary'],source_unchanged=result['source_unchanged']))
    s=result['summary'];interp=s['interpolation_absolute_error'];proxy=s['original_proxy_absolute_error']
    text=f'''# 固定训练参数的标量AC profile：定向近似检查

本例选择已知存在AC门问题的rich32开发990102，属于定向诊断，不是独立评价。计算入口只收到training_only_inputs.json.gz中的训练观测、训练拟合、候选和公开future计量精度；validation、未来真实负荷/源、真实线路和未来噪声均未传入数值计算。输入raw哈希与白名单哈希分别保存。

固定每个拓扑的训练R/X，源按时段在同一接口独立拟合，则各有向任务的完整未来距离按时段相加。单EV每槽最多一点后，可写为各时段、接入点、连续功率的标量函数之和。54个函数节点是2种公开计量精度×3点×9功率，其中5点是预声明插值端点、4点是留出中点。它们不满足完整SOC/人员服务，也没有被当作可执行计划或动作库。没有运行新的动作优化。

全部12有向任务保留：节点任务完成{s['completed_query_tasks']}/648，中点误差项完成{s['complete_midpoint_tasks']}/288，失败在records中逐项保留。原一次线性模型的QP和完整AC局部源profile分别计算。距离单位均为半个白化SSE。

| 中点绝对距离误差 | 均值 | 中位数 | 最大值 |
|---|---:|---:|---:|
| 五端点分段线性插值 | {interp['mean']:.6g} | {interp['median']:.6g} | {interp['max']:.6g} |
| 原一次线性残差代理的source-QP | {proxy['mean']:.6g} | {proxy['median']:.6g} | {proxy['max']:.6g} |

插值更小的中点任务为{s['interpolation_better_count']}/288，更大为{s['interpolation_worse_count']}/288。逐任务和平均任务排序的翻转、并列和分母见summary.json。中点仅检验这些预声明位置，不代表整区间误差上限。

两个预声明非执行曲线核对了按槽求和与原六槽evaluate_fixed_action：全部通过={s['sum_checks_all_pass']}，最大半SSE差{s['sum_check_max_error']:.6g}。单槽矩阵与原六槽对应行列一致={s['matrix_slicing_all_pass']}；全部保存拟合的独立物理/目标回放通过={s['independent_fit_replay_all_pass']}。每次非线性源拟合仍是局部取得值，该一致性不证明全局极小。

成本：{s['costs']}；记录耗时{s['seconds']:.3f}秒，不含最后写盘。records保留所有节点的源状态、R/X、标准化残差、目标、重启与停止信息、参考观测、独立回放，以及QP界。源码快照和原冻结源核对均保留，source_unchanged={result['source_unchanged']}。

这为下一步“连续功率的标量响应分段+路线/SOC MILP”提供构造检查。它不提供新的MILP最优性界、完整AC误差界、全局source最小值或EV效果优势。建立可靠上下界、处理活动源边界和在新开发样本验证仍是后续工作。
'''
    (args.output/'README.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(s,source_unchanged=result['source_unchanged']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
