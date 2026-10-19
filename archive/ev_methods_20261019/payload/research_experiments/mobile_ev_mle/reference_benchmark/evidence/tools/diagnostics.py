"""Read-only failure, score-discrepancy, and cost decomposition of frozen raws.

Formal records are opened only after statistical_complete and primary analysis.
No AC evaluation, fit, planning, calibration, threshold or decision is changed.
The explicit development-smoke mode reads at most two completed pilot histories.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import gzip
import importlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    path = Path(path); body = path.read_bytes()
    return json.loads(gzip.decompress(body) if path.suffix == '.gz' else body)


def number(value):
    return float(value) if isinstance(value, (int, float, np.number)) and math.isfinite(value) else None


def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(clean(value), ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n').encode('utf-8')
    tmp = path.with_name(path.name+'.tmp'); tmp.write_bytes(data); tmp.replace(path)


def analysis_module(source_root):
    for path in (Path(source_root), Path(source_root)/'rnj_wzzt_core'):
        if str(path) not in sys.path: sys.path.insert(0, str(path))
    return importlib.import_module('research_experiments.mobile_ev_mle.reference_benchmark.analysis')


def failure_tags(entry):
    entry = entry or {}; opt = entry.get('optimizer') or entry
    diag = opt.get('diagnostics') or {}; restarts = diag.get('restarts') or []
    messages = ' '.join(str(x) for x in [entry.get('error'), opt.get('message')]
                        +[r.get('message') for r in restarts])
    return dict(available=bool(entry.get('success', False) and entry.get('usable', True)
                               and number(entry.get('loss')) is not None),
        max_nfev=bool(opt.get('status') == 0 or any(r.get('status') == 0 for r in restarts)),
        reported_physical_domain_failure=bool(diag.get('failure_kind') == 'physical_domain'
            or entry.get('physical_voltage_feasible') is False
            or entry.get('source_box_feasible') is False or entry.get('line_domain_feasible') is False),
        AC_forward_computation_failure=any(s in messages.lower() for s in
            ('ac sweep failed', 'ac low-voltage iterate', 'nonfinite ac', 'ac fixed point')),
        computation_failure=diag.get('failure_kind') == 'computation',
        optimizer_status=opt.get('status'), optimality=number(opt.get('optimality')),
        missing_entry=not bool(entry))


def stage_failures(comparison):
    comparison = comparison or {}; starts = comparison.get('starts') or []
    future = []; historical = []; optimizers = []
    for start in starts:
        sources = start.get('source_fits') or {}
        if sources.get('future') is not None: future.append(sources['future'])
        if sources.get('history') is not None: historical.append(sources['history'])
        if start.get('optimizer') is not None: optimizers.append(start['optimizer'])
    return dict(future_source_attempted=bool(future),
        future_source_failed=any(not r.get('success', False) for r in future),
        historical_source_refit_failed=any(not r.get('success', False) for r in historical),
        joint_optimizer_attempted=bool(optimizers),
        joint_optimizer_failed=any(not r.get('success', False) for r in optimizers),
        joint_attainment_unusable=any((s.get('score') or {}).get('usable') is False for s in starts),
        joint_no_converged_attainment=bool((comparison.get('joint') or {}).get('optimization_failure')),
        exception=bool(comparison.get('error') or any(s.get('error') for s in starts)),
        starts=len(starts),
        future_budget_failed=any(failure_tags(r)['max_nfev'] for r in future),
        joint_budget_failed=any(failure_tags(r)['max_nfev'] for r in optimizers))


def route_decomposition(record, method, route, analysis):
    view = analysis.route_view(record, method, route)
    decision = analysis.decide(view, math.inf)
    _, action = analysis._action(record, method); action = action or {}
    old_h = record.get('planning', {}).get('history', {}).get('candidates', {})
    new_h = (action.get('post_history') or {}).get('candidates', {}) if route == 'postJNT' else {}
    histories = dict(old_h, **new_h)
    comparisons = dict((action.get('pre_refit') or {}).get('comparisons', {}))
    if route == 'postJNT': comparisons.update((action.get('post_refit') or {}).get('comparisons', {}))
    details = []
    for name in view['failed_candidates']:
        flags = stage_failures(comparisons.get(name))
        inherited = not failure_tags(histories.get(name))['available']
        # The compiled joint optimization_failure flag can cascade from an
        # unsuccessful historical source witness even when the actual joint
        # optimizer converged. Do not mislabel that inheritance as a new fail.
        independent = (flags['future_source_failed'] or (route != 'preFIX' and
            (flags['joint_optimizer_failed'] or flags['joint_attainment_unusable']))
            or (flags['exception'] and not inherited))
        details.append(dict(candidate=name, inherited_H_unavailable=inherited,
            future_or_joint_itself_failed=bool(independent), **flags))
    inherited = any(r['inherited_H_unavailable'] for r in details)
    own = any(r['future_or_joint_itself_failed'] for r in details)
    if view['fit_available']: primary='available'
    elif view['physical_failure']: primary='physical_execution_failure'
    elif inherited and own: primary='inherited_H_and_future_or_joint_failure'
    elif inherited: primary='inherited_H_failure_only'
    elif own: primary='future_or_joint_failure_only'
    else: primary='other_or_missing_computation'
    return dict(method=method, route=route, primary_unavailability=primary,
        available=bool(view['fit_available']), truth_present=view['truth_present'],
        conditional_available=decision['conditional_available'],
        true_margin_half_SSE=decision['true_margin_half_SSE'],
        inherited_H_unavailable=inherited, future_or_joint_itself_failed=bool(own),
        calculation_domain_failure=view['calculation_domain_failure'],
        service_voltage_failure=view['service_voltage_failure'],
        missing_action=not bool(action), failed_candidates=details,
        failure=view['failure'])


def compiled_scores(task_rows, powers):
    """The saved diagonal quadratic, batched without changing its min-pair score."""
    if not task_rows or not powers: return {}
    names = list(powers); u = np.stack([powers[k] for k in names])
    a = np.array([r['constant'] for r in task_rows])
    b = np.array([r['linear'] for r in task_rows]); d = np.array([r['quadratic'] for r in task_rows])
    values = a[None, :]+np.einsum('ktj,mtj->mk', b, u)+np.einsum('ktj,mtj->mk', d, u*u)
    return {name:float(np.min(values[i])) for i,name in enumerate(names)}


def sign(value, reference=0.):
    if value is None: return 'unavailable'
    tolerance=1e-8*(1+abs(reference))
    return 'up' if value > tolerance else 'down' if value < -tolerance else 'tie'


def condition(record, analysis):
    planning=record.get('planning') or {}; plans=planning.get('plans') or {}
    methods=list(record.get('config', {}).get('methods') or analysis.METHODS)
    history=planning.get('history') or {}; training=[]
    for name,entry in (history.get('candidates') or {}).items():
        training.append(dict(candidate=name, **failure_tags(entry)))
    routes={method:{r:route_decomposition(record,method,r,analysis)
                     for r in ('preFIX','preJNT','postJNT')} for method in methods}
    powers={name:np.asarray(p['power'],float) for name,p in plans.items() if p.get('power') is not None}
    objective_scores={name:compiled_scores(rows,powers) for name,rows in (planning.get('tasks') or {}).items()}
    pair_rows=[]; rankings=[]
    for objective in ('G-SQ','G-AQ'):
        score=objective_scores.get(objective,{})
        for suffix in ('','/F±'):
            left='P-SQ'+suffix;right=objective+suffix
            pair=dict(baseline=left,method=right,objective=objective,
                method_recorded=right in plans,baseline_recorded=left in plans,
                explicit_G_fallback=bool(plans.get(right,{}).get('fallback_reason')),
                fallback_reason=plans.get(right,{}).get('fallback_reason'),proxy_available=left in score and right in score)
            pair['exact_same_power']=bool(left in powers and right in powers and np.array_equal(powers[left],powers[right]))
            pair['same_power_within_1e_7_kw']=bool(left in powers and right in powers and np.max(abs(powers[left]-powers[right]))<=1e-7)
            if pair['proxy_available']:
                pair.update(proxy_baseline=score[left],proxy_own=score[right],proxy_delta=score[right]-score[left])
                pair['proxy_direction']=sign(pair['proxy_delta'],score[left])
            else: pair.update(proxy_delta=None,proxy_direction='unavailable')
            nominal_key='incremental_score' if objective=='G-SQ' else 'total_score'
            nominal=planning.get('nominal_AC') or {}
            values=[number(nominal.get(k,{}).get(nominal_key)) for k in (left,right)]
            pair['nominal_AC_delta']=values[1]-values[0] if all(x is not None for x in values) else None
            pair['nominal_AC_direction']=sign(pair['nominal_AC_delta'],values[0] or 0.)
            pair['actual']={}
            for route in ('preFIX','preJNT','postJNT'):
                l=routes.get(left,{}).get(route,{});r=routes.get(right,{}).get(route,{})
                x,y=l.get('true_margin_half_SSE'),r.get('true_margin_half_SSE')
                usable=bool(l.get('conditional_available') and r.get('conditional_available') and x is not None and y is not None)
                delta=y-x if usable else None
                pair['actual'][route]=dict(paired_available=usable,margin_delta=delta,
                    direction=sign(delta,x or 0.),baseline_available=l.get('available',False),
                    own_available=r.get('available',False),baseline_truth_present=l.get('truth_present',False),
                    own_truth_present=r.get('truth_present',False))
            pair_rows.append(pair)
        # Deduplicate actual action identities; aliases are not extra evidence.
        unique={}
        for method,plan in plans.items():
            key=plan.get('action_key')
            if key is not None and key not in unique and method in score:
                d=routes.get(method,{}).get('preJNT',{}); nom=number((planning.get('nominal_AC') or {}).get(method,{}).get(
                    'incremental_score' if objective=='G-SQ' else 'total_score'))
                unique[key]=dict(method=method,proxy=score[method],nominal=nom,
                    actual=d.get('true_margin_half_SSE') if d.get('conditional_available') else None)
        valid=[r for r in unique.values() if r['actual'] is not None]
        nominal=[r for r in valid if r['nominal'] is not None]
        rankings.append(dict(objective=objective,unique_actions=len(unique),actual_usable_actions=len(valid),
            spearman_proxy_vs_actual=analysis.spearman([r['proxy'] for r in valid],[r['actual'] for r in valid]),
            spearman_nominal_vs_actual=analysis.spearman([r['nominal'] for r in nominal],[r['actual'] for r in nominal])))
    actions=[]
    for key,action in (record.get('evaluation') or {}).get('actions',{}).items():
        counters=Counter();components={}
        for stage in ('pre_refit','post_passive','post_history','post_refit'):
            block=action.get(stage) or {};components[stage]=number(block.get('seconds'))
            for k,v in (block.get('counters') or {}).items():
                if number(v) is not None: counters[k]+=v
        actions.append(dict(action_key=key,methods=action.get('methods',[]),
            components_seconds=components,counters=dict(counters),
            pre_refit_wall_seconds=number(action.get('pre_refit_seconds')),
            new_candidates=len(action.get('new_candidates') or []),
            failures=action.get('failures') or {}))
    return dict(spec=record['spec'],record_status=record.get('status'),
        history_complete=history.get('complete',False),history_missing=not bool(history),training=training,
        routes=routes,pairs=pair_rows,rankings=rankings,
        actual_unique_actions=actions,
        shared_history_counters=history.get('counters') or {},
        shared_selection_cost=planning.get('cost') or {},
        dependency_cost_accounting=analysis.deployment_costs(record,methods),
        source_unchanged=record.get('source_unchanged'))


def stats(values):
    values=[v for v in values if number(v) is not None]
    return dict(n_known=len(values),total=float(sum(values)) if values else None,
                mean=float(np.mean(values)) if values else None,
                median=float(np.median(values)) if values else None,
                maximum=float(max(values)) if values else None)


def summarize(rows):
    training=[r for c in rows for r in c['training']]
    result=dict(conditions=len(rows),date_clusters=len({c['spec']['cluster_index'] for c in rows}),
        record_status=dict(Counter(str(c['record_status']) for c in rows)),
        training=dict(history_missing_conditions=sum(c['history_missing'] for c in rows),
            historical_unavailable_conditions=sum(not c['history_complete'] for c in rows),
            candidate_fits=len(training),candidate_unavailable=sum(not r['available'] for r in training),
            max_nfev_candidate_fits=sum(r['max_nfev'] for r in training),
            unavailable_max_nfev_candidate_fits=sum(r['max_nfev'] and not r['available'] for r in training),
            reported_physical_domain_failures=sum(r['reported_physical_domain_failure'] for r in training),
            AC_forward_computation_failures=sum(r['AC_forward_computation_failure'] for r in training)),
        routes={},pairs={},costs={})
    methods=sorted({m for c in rows for m in c['routes']})
    for method in methods:
        for route in ('preFIX','preJNT','postJNT'):
            data=[c['routes'][method][route] for c in rows if method in c['routes']]
            result['routes'][method+'|'+route]=dict(total=len(data),
                primary_unavailability=dict(Counter(r['primary_unavailability'] for r in data)),
                inherited_H_unavailable=sum(r['inherited_H_unavailable'] for r in data),
                future_or_joint_itself_failed=sum(r['future_or_joint_itself_failed'] for r in data),
                calculation_domain_failures=sum(r['calculation_domain_failure'] for r in data),
                service_voltage_failures=sum(r['service_voltage_failure'] for r in data))
    pair_methods=sorted({p['method'] for c in rows for p in c['pairs']})
    for method in pair_methods:
        data=[p for c in rows for p in c['pairs'] if p['method']==method]
        out=dict(total=len(data),explicit_G_fallback=sum(r['explicit_G_fallback'] for r in data),
            exact_same_power=sum(r['exact_same_power'] for r in data),
            same_power_within_1e_7_kw=sum(r['same_power_within_1e_7_kw'] for r in data),
            proxy_direction=dict(Counter(r['proxy_direction'] for r in data)),
            nominal_AC_direction=dict(Counter(r['nominal_AC_direction'] for r in data)),actual={})
        for route in ('preFIX','preJNT','postJNT'):
            valid=[r for r in data if r['actual'][route]['paired_available']]
            out['actual'][route]=dict(paired_available=len(valid),all_pairs=len(data),
                margin_direction=dict(Counter(r['actual'][route]['direction'] for r in valid)),
                proxy_up_but_actual_down=sum(r['proxy_direction']=='up' and r['actual'][route]['direction']=='down' for r in valid),
                nominal_up_but_actual_down=sum(r['nominal_AC_direction']=='up' and r['actual'][route]['direction']=='down' for r in valid),
                unavailable_due_to_candidate_miss=sum(not r['actual'][route]['baseline_truth_present'] or not r['actual'][route]['own_truth_present'] for r in data),
                unavailable_due_to_computation_or_physics=sum(not r['actual'][route]['baseline_available'] or not r['actual'][route]['own_available'] for r in data))
        result['pairs'][method]=out
    actions=[a for c in rows for a in c['actual_unique_actions']]
    counters=Counter()
    for a in actions: counters.update(a['counters'])
    result['costs']=dict(unique_actual_actions=len(actions),actual_action_counters=dict(counters),
        history_source_cache_hits_actual=int(counters.get('history_source_cache_hits',0)),
        shared_history_seconds=stats(c['shared_selection_cost'].get('history_seconds') for c in rows),
        shared_all_methods_selection_seconds=stats(c['shared_selection_cost'].get('selection_total_seconds') for c in rows),
        per_unique_action={stage:stats(a['components_seconds'][stage] for a in actions)
                           for stage in ('pre_refit','post_passive','post_history','post_refit')},
        dependency_deployment_upper_by_method={m:stats((c['dependency_cost_accounting']['policies'].get(m) or {}).get('deployment_upper_seconds') for c in rows) for m in methods},
        scope='measured shared preparation and unique-action costs; aliases never counted as new actions; dependency deployment accounting charges shared action in full, without amortization claims')
    return result


def report(result):
    lines=['# 失败、规划分数与费用诊断','',
        '只读现有记录；AC调用、重新拟合和重新规划均为0。此文件不改变主判决或阈值。', '',
        '模式：'+result['mode']+'。同一天的网络和源条件属于配对条件，日期簇才是统计单位。','']
    for phase,content in result['phases'].items():
        lines.extend(['## '+phase,'','|网络|条件|历史不可用|失败候选/候选拟合|达到预算|报告物理域失败|',
                      '|---|---:|---:|---:|---:|---:|'])
        for network,s in content['by_network'].items():
            t=s['training'];lines.append('|%s|%d|%d|%d/%d|%d|%d|'%(
                network,s['conditions'],t['historical_unavailable_conditions'],t['candidate_unavailable'],
                t['candidate_fits'],t['max_nfev_candidate_fits'],t['reported_physical_domain_failures']))
        lines.extend(['','|网络/方法|G回退|与P-SQ精确同动作|preJNT可比较|代理升而实际降|名义AC升而实际降|',
                      '|---|---:|---:|---:|---:|---:|'])
        for network,s in content['by_network'].items():
            for method,p in s['pairs'].items():
                a=p['actual']['preJNT'];lines.append('|%s/%s|%d/%d|%d/%d|%d/%d|%d|%d|'%(
                    network,method,p['explicit_G_fallback'],p['total'],p['exact_same_power'],p['total'],
                    a['paired_available'],a['all_pairs'],a['proxy_up_but_actual_down'],a['nominal_up_but_actual_down']))
        lines.append('')
    lines.extend(['主分解、每路线继承/自身失败、候选缺失、逐动作时间和源缓存计数见diagnostics.json。',
        '代理和实际比较使用同一保存目标；preJNT保持原候选池，postJNT还包含候选更新，不能单归为物理收益。',
        '同动作同时报告精确相等和1e-7 kW数值相等。后者只作诊断，不合并计量或改变判决。',
        '路线主原因按实际物理失败、历史与新增均失败、仅继承历史、仅新增自身失败、其他缺失顺序归类；另保留各原因重叠计数。',
        'joint汇总失败标记可能由历史见证失败继承；只有新增源或joint优化器本身失败才计为新增自身失败。',
        '费用保留共享准备与唯一实际动作；单策略依赖核算全额计入所需动作和历史准备，不按方法数摊分成standalone优势。',''])
    return '\n'.join(lines)


def run(output, source_root=ROOT, *, development_smoke=False):
    output=Path(output).resolve(); analysis=analysis_module(source_root)
    if development_smoke:
        phases=('development',);policy=None;identity=None
        paths=sorted((output/'development').glob('*.json.gz'))[:2]
        if len(paths)!=2:raise RuntimeError('development smoke requires exactly two available completed pilot records')
        phase_paths={'development':paths}
    else:
        controller=read(output/'controller.json')
        if controller.get('status')!='statistical_complete':
            raise RuntimeError('diagnostics cannot open formal raws before statistical_complete')
        if not (output/'summary.json').is_file() or not (output/'audit.json').is_file():
            raise RuntimeError('primary analysis must finish before diagnostics')
        policy=read(output/'freeze.json');identity=policy['identity'];phases=('development','calibration','evaluation')
        phase_paths={phase:sorted((output/phase).glob('*.json.gz')) for phase in phases}
        for phase,paths in phase_paths.items():
            if len(paths)!=len(analysis.condition_manifest(policy['config'],phase)):
                raise RuntimeError('formal phase incomplete: '+phase)
    result=dict(schema='reference_failure_score_cost_diagnostics_v1',mode='development smoke only' if development_smoke else 'frozen post-analysis',
        created_utc=datetime.now(timezone.utc).isoformat(),policy_identity=identity,
        tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        analysis_sha256=sha256(Path(analysis.__file__).read_bytes()).hexdigest(),
        raw_files={},phases={},AC_calls=0,fit_calls=0,planning_calls=0,
        changes_primary_decisions=False,changes_thresholds=False)
    for phase,paths in phase_paths.items():
        rows=[]
        for path in paths:
            raw=read(path)
            if raw.get('spec',{}).get('phase')!=phase:raise RuntimeError('raw phase mismatch')
            if not development_smoke and raw.get('freeze_identity')!=identity:raise RuntimeError('wrong frozen raw identity')
            if development_smoke and raw.get('status') not in ('complete','complete_with_retained_failures'):
                raise RuntimeError('pilot raw not complete')
            row=condition(raw,analysis);rows.append(row)
            if development_smoke:
                # Independent check of the batched quadratic against the frozen
                # scalar scorer; no optimization or new physical calculation.
                p=raw.get('planning',{});powers={k:np.asarray(v['power']) for k,v in p.get('plans',{}).items()}
                for objective,tasks in p.get('tasks',{}).items():
                    scores=compiled_scores(tasks,powers)
                    for name,power in powers.items():
                        expected,_=analysis.old.task_scores(tasks,power)
                        if expected is not None and abs(scores[name]-expected)>1e-8*(1+abs(expected)):
                            raise AssertionError('batched proxy differs from frozen scalar scorer')
            result['raw_files'][str(path.relative_to(output)).replace('\\','/')]=dict(sha256=sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
        groups=defaultdict(list)
        for row in rows:groups[str(row['spec'].get('network','unknown'))].append(row)
        result['phases'][phase]=dict(all_networks=summarize(rows),by_network={name:summarize(v) for name,v in sorted(groups.items())},conditions=rows)
    json_path=output/('diagnostics_development_smoke.json' if development_smoke else 'diagnostics.json')
    md_path=json_path.with_suffix('.md');write(json_path,result);md_path.write_bytes(report(result).encode('utf-8'))
    return result,json_path,md_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--development-smoke',type=Path,metavar='PILOT_OUTPUT')
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        if args.output or args.development_smoke:parser.error('--self-test is standalone')
        print(json.dumps(self_test(args.source_root)));return
    if bool(args.output)==bool(args.development_smoke):parser.error('choose --output or --development-smoke')
    result,path,md=run(args.development_smoke or args.output,args.source_root,
                       development_smoke=bool(args.development_smoke))
    print(json.dumps(dict(mode=result['mode'],json=str(path),markdown=str(md),
        conditions=sum(len(v['conditions']) for v in result['phases'].values()),
        AC_calls=0,fit_calls=0,planning_calls=0)))


def self_test(source_root=ROOT):
    from types import SimpleNamespace
    from tempfile import TemporaryDirectory
    checks=[]
    tasks=[dict(name='a',constant=3.,linear=[[1.,-2.],[0.,1.]],quadratic=[[2.,3.],[1.,1.]]),
           dict(name='b',constant=1.,linear=[[0.,1.],[2.,-1.]],quadratic=[[1.,2.],[4.,2.]])]
    powers={'left':np.array([[1.,0.],[0.,-2.]]),'right':np.array([[0.,-1.],[3.,0.]])}
    actual=compiled_scores(tasks,powers)
    for name,u in powers.items():
        expected=min(t['constant']+sum(t['linear'][i][j]*u[i,j]+t['quadratic'][i][j]*u[i,j]**2
            for i in range(2) for j in range(2)) for t in tasks)
        assert actual[name]==expected
    checks.append('batched_proxy_matches_independent_scalar_polynomial')
    assert failure_tags(dict(success=False,optimizer=dict(status=0,diagnostics={})))['max_nfev']
    assert failure_tags(dict(success=False,physical_voltage_feasible=False))['reported_physical_domain_failure']
    checks.append('budget_and_physical_tags')
    good=dict(success=True,usable=True,loss=1.)
    comparison=dict(joint=dict(optimization_failure=True),starts=[dict(
        source_fits=dict(history=dict(success=False),future=dict(success=True)),
        optimizer=dict(success=True),score=dict(usable=True))])
    raw=dict(planning=dict(history=dict(candidates={'c':dict(success=False)})),
             action=dict(pre_refit=dict(comparisons={'c':comparison})))
    base_view=dict(failed_candidates=['c'],fit_available=False,physical_failure=False,
        truth_present=True,calculation_domain_failure=False,service_voltage_failure=False,failure=None)
    module=SimpleNamespace(_action=lambda r,m:('x',r['action']),route_view=lambda r,m,route:dict(base_view),
        decide=lambda v,t:dict(conditional_available=False,true_margin_half_SSE=None))
    row=route_decomposition(raw,'G-SQ','preJNT',module)
    assert row['primary_unavailability']=='inherited_H_failure_only'
    assert not row['future_or_joint_itself_failed']
    checks.append('inherited_H_flag_not_counted_as_joint_optimizer_failure')
    raw['planning']['history']['candidates']['c']=good
    comparison['starts'][0]['source_fits']['history']['success']=True
    comparison['starts'][0]['source_fits']['future']['success']=False
    row=route_decomposition(raw,'G-SQ','preJNT',module)
    assert row['primary_unavailability']=='future_or_joint_failure_only'
    checks.append('actual_future_source_failure_is_separate')
    base_view['physical_failure']=True;base_view['service_voltage_failure']=True
    assert route_decomposition(raw,'G-SQ','preJNT',module)['primary_unavailability']=='physical_execution_failure'
    checks.append('physical_execution_failure_priority')
    with TemporaryDirectory(prefix='reference_diagnostics_guard_',dir=Path(__file__).parent) as temp:
        p=Path(temp).resolve()
        assert p.parent==Path(__file__).resolve().parent
        write(p/'controller.json',dict(status='running'))
        try:run(p,source_root)
        except RuntimeError as error:assert 'before statistical_complete' in str(error)
        else:raise AssertionError('formal running guard did not reject')
        write(p/'controller.json',dict(status='statistical_complete'))
        try:run(p,source_root)
        except RuntimeError as error:assert 'primary analysis must finish' in str(error)
        else:raise AssertionError('primary analysis guard did not reject')
        assert not (p/'diagnostics.json').exists()
    checks.append('formal_raw_access_guard_and_primary_analysis_guard')
    result=dict(passed=len(checks),checks=checks,AC_calls=0,fit_calls=0,planning_calls=0,
                tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    write(Path(__file__).with_name('diagnostics_selftest_receipt.json'),result)
    return result


if __name__=='__main__':main()
