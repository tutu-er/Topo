"""Development-only, saved-action diagnostics for nearest-competitor scores.

No new action, fit, QP or optimizer is constructed. The scores use saved local
full-AC attained counterfactual costs, not a new proxy optimality certificate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

SCHEMA = 'ev_compact_nearest_diagnostic_v1'
SCORES = ('mean_added', 'cap20_added', 'phi_add', 'phi_H', 'phi_self', 'actual_M', 'noiseless_M')


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def close(a, b, tol=1e-8):
    return abs(a-b) <= tol*max(1., abs(a), abs(b))


def _empty(reason):
    return dict(success=False, reason=reason, errors=[reason], self_errors=[reason],
        scores={k:None for k in SCORES[:5]}, task_components=[], competitor_means=[], reference_minima=[])


def score_prediction(prediction, historical_costs, candidates, *, cap=20.):
    """Compute equations (2)/(3), averaging scenarios before each minimum.

    Pair weights are the saved global task weights. Their conditional scenario
    distributions must agree across competitors within each reference. An
    absent difficult task makes the aggregate unavailable, not easier.
    """
    if not prediction or not prediction.get('pairs'):
        return _empty('no_saved_H_only_prediction')
    names = tuple(candidates)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError('distinct candidate names required')
    if set(historical_costs) != set(names) or any(number(v) is None for v in historical_costs.values()):
        return _empty('missing_historical_candidate_cost')
    costs = {k:float(v) for k,v in historical_costs.items()}
    fits, fit_keys, errors = {}, set(), []
    for record in prediction.get('records') or []:
        key = (record.get('ensemble_id') or 'default', record.get('a'))
        for fit in record.get('fits') or []:
            fit_key = (*key, fit.get('name'))
            if fit_key in fit_keys:
                errors.append('duplicate_saved_prediction_fit:'+str(fit_key))
                continue
            fit_keys.add(fit_key)
            if fit.get('success') and number(fit.get('loss')) is not None:
                fits[fit_key] = float(fit['loss'])
    components, self_errors, seen = [], [], set()
    own_costs = defaultdict(list)
    for pair in prediction['pairs']:
        a, b, context = pair.get('a'), pair.get('b'), pair.get('ensemble_id') or 'default'
        key = a, b, context
        if key in seen:
            errors.append('duplicate_task:'+str(key))
        seen.add(key)
        if a not in costs or b not in costs or a == b:
            errors.append('unknown_or_self_competitor:'+str(key))
            continue
        weight = number(pair.get('weight'))
        if weight is None or weight < 0:
            errors.append('invalid_task_weight:'+str(key))
        q = number(pair.get('joint_loss'))
        q_source = 'pair.joint_loss'
        if q is None:
            q, q_source = fits.get((context, a, b)), 'saved_prediction_fit.loss'
        qa = number(pair.get('self_loss'))
        qa_source = 'pair.self_loss'
        if qa is None:
            qa, qa_source = fits.get((context, a, a)), 'saved_self_fit.loss'
        if q is None:
            errors.append('missing_saved_competitor_joint_cost:'+str(key))
        if qa is None:
            self_errors.append('missing_saved_q_aa:'+str(key))
        else:
            own_costs[a, context].append(qa)
        added = .5*(q-costs[b]) if q is not None else None
        joint = .5*(q-costs[a]) if q is not None else None
        relative = .5*(q-qa) if q is not None and qa is not None else None
        for field, value in (('added', added), ('joint', joint)):
            stored = number(pair.get(field))
            if stored is not None and value is not None and not close(stored, value):
                errors.append('saved_'+field+'_disagrees_with_cost:'+str(key))
        stored_self = number(pair.get('self_relative'))
        if stored_self is not None and relative is not None and not close(stored_self, relative):
            self_errors.append('saved_self_relative_disagrees_with_cost:'+str(key))
        if pair.get('success') is not True:
            errors.append('unsuccessful_saved_task:'+str(key))
        components.append(dict(name=pair.get('name'), a=a, b=b, scenario=context, weight=weight,
            success=pair.get('success') is True, h_a=costs[a], h_b=costs[b],
            q_ab=q, q_aa=qa, q_ab_source=q_source if q is not None else 'unavailable',
            q_aa_source=qa_source if qa is not None else 'unavailable',
            added_h_b=added, joint_h_a=joint, self_relative=relative,
            stored_added=number(pair.get('added')), stored_joint=number(pair.get('joint')),
            stored_self_relative=stored_self))
    for key, values in own_costs.items():
        if any(not close(v, values[0]) for v in values):
            self_errors.append('different_q_aa_for_same_reference_scenario:'+str(key))
    if not close(sum(r['weight'] or 0. for r in components), 1.):
        errors.append('global_task_weights_not_normalized')
    grouped = defaultdict(list)
    for row in components:
        grouped[row['a'], row['b']].append(row)
    required = {(a,b) for a in names for b in names if a != b}
    if set(grouped) != required:
        errors.append('missing_or_extra_directed_competitor')
    pair_rows, reference_rows = [], []
    reference_weights = {}
    for a in names:
        ab_groups = [(b, grouped.get((a,b), [])) for b in names if b != a]
        masses = [sum(r['weight'] or 0. for r in group) for b,group in ab_groups]
        wa = sum(masses)
        reference_weights[a] = wa
        if wa <= 0 or any(not close(mass, wa/(len(names)-1)) for mass in masses):
            errors.append('unequal_or_missing_competitor_weight:'+a)
        scenario_law = None
        for (b, group), mass in zip(ab_groups, masses):
            law = {r['scenario']:(r['weight'] or 0.)/mass for r in group} if mass > 0 else {}
            if scenario_law is None:
                scenario_law = law
            elif set(law) != set(scenario_law) or any(not close(law[s],scenario_law[s]) for s in law if s in scenario_law):
                errors.append('scenario_law_differs_between_competitors:'+a)
            means = {}
            for field in ('added_h_b','joint_h_a','self_relative'):
                means[field] = (sum((r['weight'] or 0.)*r[field] for r in group)/mass
                    if group and mass > 0 and all(r[field] is not None and r['success'] for r in group) else None)
            pair_rows.append(dict(a=a, b=b, reference_weight=wa, pair_weight=mass,
                scenarios=len(group), scenario_weights=law, **means))
        summary = dict(a=a, reference_weight=wa)
        for field in ('added_h_b','joint_h_a','self_relative'):
            alternatives = [r for r in pair_rows if r['a'] == a]
            values = [r[field] for r in alternatives]
            value = min(values) if values and all(v is not None for v in values) else None
            summary['nearest_'+field] = value
            summary['nearest_'+field+'_candidates'] = ([r['b'] for r in alternatives if close(r[field],value)]
                if value is not None else [])
        reference_rows.append(summary)
    if prediction.get('success') is not True:
        errors.append('prediction_marked_incomplete')
    valid = not errors
    self_valid = valid and not self_errors
    scores = dict(mean_added=sum(r['weight']*r['added_h_b'] for r in components) if valid else None,
        cap20_added=sum(r['weight']*min(r['added_h_b'],cap) for r in components) if valid else None,
        phi_add=sum(r['reference_weight']*r['nearest_added_h_b'] for r in reference_rows) if valid else None,
        phi_H=sum(r['reference_weight']*r['nearest_joint_h_a'] for r in reference_rows) if valid else None,
        phi_self=sum(r['reference_weight']*r['nearest_self_relative'] for r in reference_rows) if self_valid else None)
    return dict(success=valid, self_success=self_valid, reason=None if valid else 'incomplete_saved_task_set',
        errors=errors, self_errors=self_errors, scores=scores, task_components=components,
        competitor_means=pair_rows, reference_minima=reference_rows,
        score_source='saved local full-AC attained H+counterfactual-future costs',
        historical_baseline='h_b for phi_add; h_a for phi_H; saved q_aa for phi_self')


def _flat(power):
    if not isinstance(power, (list,tuple)):
        return None
    result = []
    for item in power:
        if isinstance(item, (list,tuple)):
            part = _flat(item)
            if part is None:
                return None
            result.extend(part)
        elif number(item) is None:
            return None
        else:
            result.append(float(item))
    return result


def saved_prediction(trial, method, power, *, tolerance=1e-10):
    selection = trial.get('selection') or {}
    direct = (selection.get('predicted_AC') or {}).get(method)
    flat = _flat(power)
    if flat is None:
        return None, dict(source=None, reason='no_saved_executable_curve', max_power_difference=None)
    def difference(pred):
        other = _flat(pred.get('power_kw'))
        return max((abs(a-b) for a,b in zip(flat,other)), default=0.) if other is not None and len(other)==len(flat) else math.inf
    if direct is not None:
        error = difference(direct)
        if error <= tolerance:
            return direct, dict(source='named_selected_prediction', reason=None, max_power_difference=error)
        return None, dict(source=None, reason='named_prediction_curve_mismatch', max_power_difference=None)
    matches = [(p,difference(p)) for p in selection.get('all_prediction_records') or [] if difference(p)<=tolerance]
    if not matches:
        return None, dict(source=None, reason='no_saved_H_only_prediction_for_curve', max_power_difference=None)
    signatures = {json.dumps(p.get('pairs'),sort_keys=True,allow_nan=False) for p,error in matches}
    if len(signatures) != 1:
        return None, dict(source=None, reason='ambiguous_saved_prediction_for_curve', max_power_difference=None)
    pred, error = min(matches,key=lambda item:item[1])
    return pred, dict(source='saved_cache_same_curve', reason=None, max_power_difference=error)


def _margin(fits, true, names):
    bank = {f['name']:float(f['loss']) for f in fits or []
            if f.get('success') and number(f.get('loss')) is not None}
    return min(bank[b] for b in names if b != true)-bank[true] if true in bank and set(bank)==set(names) else None


def ranks(values):
    """Descending average ranks, retaining exact ties in saved scalar scores."""
    ordered = sorted((value,name) for name,value in values.items() if number(value) is not None)
    result, start = {}, 0
    while start < len(ordered):
        stop = start+1
        while stop < len(ordered) and ordered[stop][0] == ordered[start][0]:
            stop += 1
        average = len(ordered)-(start+stop-1)/2
        for value,name in ordered[start:stop]:
            result[name] = average
        start = stop
    return result


def spearman(left, right):
    common = sorted(set(left)&set(right))
    x, y = ranks({m:left[m] for m in common}), ranks({m:right[m] for m in common})
    common = sorted(set(x)&set(y))
    if len(common) < 2:
        return dict(n_methods=len(common), value=None, reason='fewer_than_two_available_methods')
    # Re-rank the common set when either metric has missing methods.
    x, y = ranks({m:left[m] for m in common}), ranks({m:right[m] for m in common})
    mx,my = mean(x.values()),mean(y.values())
    xx,yy = sum((x[m]-mx)**2 for m in common),sum((y[m]-my)**2 for m in common)
    value = sum((x[m]-mx)*(y[m]-my) for m in common)/math.sqrt(xx*yy) if xx>0 and yy>0 else None
    return dict(n_methods=len(common), value=value, reason=None if value is not None else 'constant_rank_vector')


def anchor_diagnostic(methods, reference_rows, candidates, *, anchor='charge_mobile', tolerance=1e-8):
    """Pointwise H-only acceptance set on saved actions, not an optimizer.

    Requires every reference's saved full-AC nearest self-relative score.
    Actual/noiseless margins, true-reference labels and future observations
    are not used. An upper proxy score above the anchor is not a lower bound.
    Caller must first ensure one history, one candidate set, the same scenario
    law and weights, and the same historical baselines for all compared actions.
    This small API receives scores only and cannot reconstruct those inputs.
    analyze_trial supplies one saved H at a time; its current-v2 weight/baseline
    comparability is additionally checked by the independent slim review.
    """
    names = tuple(candidates)
    bank = {r['method']:r for r in methods}
    grouped = defaultdict(dict)
    for row in reference_rows:
        if row['a'] in grouped[row['method']]:
            raise ValueError('duplicate reference/anchor record')
        grouped[row['method']][row['a']] = row
    def complete(method):
        refs = grouped.get(method,{})
        return bool(bank.get(method,{}).get('self_success') and set(refs)==set(names)
            and all(number(refs[a].get('nearest_self_relative')) is not None for a in names))
    anchor_complete = complete(anchor)
    decisions, components = [], []
    for method in bank:
        valid = anchor_complete and complete(method)
        violating = []
        for a in names:
            reference = grouped.get(method,{}).get(a,{})
            base = grouped.get(anchor,{}).get(a,{})
            value, threshold = number(reference.get('nearest_self_relative')), number(base.get('nearest_self_relative'))
            delta = value-threshold if value is not None and threshold is not None else None
            guard = tolerance*max(1.,abs(value),abs(threshold)) if valid else None
            passed = delta >= -guard if valid else None
            if passed is False:violating.append(a)
            components.append(dict(method=method, anchor=anchor, a=a, candidate_value=value,
                anchor_value=threshold, difference=delta, numerical_tolerance=guard,
                passes=passed, candidate_nearest=reference.get('nearest_self_relative_candidates'),
                anchor_nearest=base.get('nearest_self_relative_candidates')))
        decision = 'accepted' if valid and not violating else 'rejected' if valid else 'unknown'
        reason = ('all_reference_scores_at_least_anchor' if decision=='accepted' else
                  'one_or_more_reference_scores_below_anchor' if decision=='rejected' else
                  'missing_complete_anchor_prediction' if not anchor_complete else 'missing_complete_reference_prediction')
        decisions.append(dict(method=method,anchor=anchor,decision=decision,reason=reason,
            required_references=len(names),violating_references=violating,all_reference_predictions_available=valid))
    return dict(anchor=anchor,decisions=decisions,components=components,
        scope='saved H-only full-AC point acceptance; not a new MILP certificate or selector',
        future_truth_used=False,proxy_upper_above_anchor_is_sufficient=False)


def analyze_trial(trial):
    if trial.get('phase') != 'development':
        raise ValueError('development-only diagnostic; calibration/evaluation not accepted')
    names = [r['name'] for r in trial.get('proposal',{}).get('bank',[])]
    historical = {f['name']:f['loss'] for f in trial.get('historical_fits') or [] if f.get('success')}
    declared = (trial.get('delivery') or {}).get('declared_methods') or [r['method'] for r in trial.get('rows') or []]
    bank = {r['method']:r for r in trial.get('rows') or []}
    if len(bank) != len(trial.get('rows') or []):
        raise ValueError('duplicate saved method')
    methods, tasks, competitors, references = [], [], [], []
    for method in declared:
        row = bank.get(method) or {}
        prediction, match = saved_prediction(trial,method,row.get('power_kw'))
        score = score_prediction(prediction,historical,names)
        common = dict(seed=trial['seed'], method=method, true_reference=row.get('true_name'),
                      policy_id=trial.get('policy_id'))
        actual = _margin(row.get('fits'),row.get('true_name'),names) if row.get('status') == 'complete' else None
        noiseless = row.get('noiseless_AC_refit') or {}
        noiseless_margin = _margin(noiseless.get('fits'),row.get('true_name'),names) if noiseless.get('all_candidates') else None
        methods.append(dict(**common, status=row.get('status','missing_method'), prediction_match=match,
            prediction_success=score['success'], self_success=score.get('self_success',False),
            diagnostic_errors=score['errors'], self_errors=score['self_errors'],
            actual_M=actual, noiseless_M=noiseless_margin, **score['scores']))
        for target,key in ((tasks,'task_components'),(competitors,'competitor_means'),(references,'reference_minima')):
            target.extend(dict(**common,**piece) for piece in score[key])
    ranked = {}
    for metric in SCORES:
        rank = ranks({r['method']:r[metric] for r in methods})
        ranked[metric] = rank
        for r in methods:
            r['rank_'+metric] = rank.get(r['method'])
    correlations = []
    for metric in SCORES[:5]:
        for target in ('actual_M','noiseless_M'):
            correlations.append(dict(seed=trial['seed'], metric=metric, target=target,
                **spearman({r['method']:r[metric] for r in methods},{r['method']:r[target] for r in methods})))
    anchored = anchor_diagnostic(methods,references,names)
    return dict(seed=trial['seed'], policy_id=trial.get('policy_id'), methods=methods,
        task_components=tasks, competitor_means=competitors, reference_minima=references,
        correlations=correlations,
        anchor_decisions=[dict(seed=trial['seed'],**r) for r in anchored['decisions']],
        anchor_components=[dict(seed=trial['seed'],**r) for r in anchored['components']])


def _write_csv(path, rows):
    if not rows:
        path.write_text('',encoding='utf-8-sig')
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=keys);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def analyze_directory(directory, output):
    directory,output = Path(directory),Path(output)
    freeze_path = directory/'freeze.json'
    freeze = json.loads(freeze_path.read_bytes())
    if freeze.get('phase') != 'development':
        raise ValueError('development-only directory required')
    paths = sorted(directory.glob('trial_*.slim.json.gz'))
    if not paths:
        raise ValueError('saved slim development trials required')
    results,inputs,seen = [],[],set()
    for path in paths:
        raw = path.read_bytes();trial = json.loads(gzip.decompress(raw))
        seed = trial['seed']
        if seed in seen or seed not in freeze['seeds'] or trial.get('policy_id') != freeze['policy_id']:
            raise ValueError('unplanned/duplicate history or policy mismatch')
        seen.add(seed)
        results.append(analyze_trial(trial))
        inputs.append(dict(file=path.name,sha256=hashlib.sha256(raw).hexdigest(),
            source_raw_sha256=(trial.get('delivery') or {}).get('source_raw_sha256')))
        del trial
    output.mkdir(parents=True,exist_ok=True)
    tables = {key:[r for result in results for r in result[key]] for key in
              ('methods','task_components','competitor_means','reference_minima','correlations',
               'anchor_decisions','anchor_components')}
    for key,rows in tables.items():
        _write_csv(output/(key+'.csv'),rows)
    correlation_summary = []
    for metric in SCORES[:5]:
        for target in ('actual_M','noiseless_M'):
            selected = [r for r in tables['correlations'] if r['metric']==metric and r['target']==target]
            values = [r['value'] for r in selected if r['value'] is not None]
            correlation_summary.append(dict(metric=metric,target=target,history_denominator=len(selected),
                available_histories=len(values),median=median(values) if values else None,
                minimum=min(values) if values else None,maximum=max(values) if values else None))
    summary = dict(schema=SCHEMA,phase='development',policy_id=freeze['policy_id'],
        independent_histories=len(results),planned_histories=len(freeze['seeds']),
        seeds=sorted(seen), method_conditions=len(tables['methods']),
        prediction_available=sum(r['prediction_success'] for r in tables['methods']),
        self_available=sum(r['self_success'] for r in tables['methods']),
        unavailable=[dict(seed=r['seed'],method=r['method'],reason=r['prediction_match']['reason'],
            errors=r['diagnostic_errors']) for r in tables['methods'] if not r['prediction_success']],
        independent_unit='one H; rankings/correlations across its saved actions are descriptive',
        new_actions_selected=0,new_fits=0,new_optimizations=0,
        score_source='saved local full-AC attained counterfactual costs, not new QP/MILP optima',
        strategy_updated=False,calibration_or_evaluation_read=False,
        conclusion_scope='interpretation of existing development actions; no claim that optimizing the new criterion is effective',
        input_manifest=inputs,freeze_sha256=hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
        diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        correlation_summary=correlation_summary,
        anchor_diagnostic=dict(anchor='charge_mobile',
            accepted=sum(r['decision']=='accepted' for r in tables['anchor_decisions']),
            rejected=sum(r['decision']=='rejected' for r in tables['anchor_decisions']),
            unknown=sum(r['decision']=='unknown' for r in tables['anchor_decisions']),
            scope='existing H-only full-AC point acceptance; not a new selector or MILP certificate',
            future_truth_used=False,proxy_upper_above_anchor_is_sufficient=False))
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# 已有开发动作的最近竞争者诊断','',
        f"独立历史 {len(results)}/{len(freeze['seeds'])}；方法条件 {len(tables['methods'])}；保存预测可用 {summary['prediction_available']}，自相对可用 {summary['self_available']}。",
        '每个参考先对情景加权，再取最近竞争者；最后平均参考。phi_add减竞争者历史h_b；phi_H减参考历史h_a；phi_self减已保存的参考自身联合成本。','',
        '| 历史 | 方法 | mean | cap20 | phi_add | phi_H | phi_self | 实际M | 无噪M |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return '—' if v is None else f'{v:.5g}'
    for row in tables['methods']:
        lines.append('| '+str(row['seed'])+' | '+row['method']+' | '+' | '.join(fmt(row[k]) for k in SCORES)+' |')
    lines+=['','逐历史排序见methods.csv；每个任务及h_a/h_b/q_ab/q_aa见task_components.csv，最近竞争者身份及并列候选见reference_minima.csv。',
        '无保存预测的动作保留在分母且指标留空；不补拟合。相关系数按每历史的已有动作计算，无p值或独立样本扩增。','',
        f"以预声明的charge_mobile为锚，要求全部参考的最近自相对完整AC分数不下降：保存动作中接受 {summary['anchor_diagnostic']['accepted']}，拒绝 {summary['anchor_diagnostic']['rejected']}，未知 {summary['anchor_diagnostic']['unknown']}。逐参考差值见anchor_components.csv。",
        '这只是已存动作的点验收集合；代理upper高于锚不足以保证真实代理q达到锚，尚需有效lower或完整AC点验收。','',
        '这些是完整AC局部取得成本下的描述性诊断，没有用新指标选择动作或执行MILP；独立校准/评价未读取。','']
    (output/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=analyze_directory(args.directory,args.output)
    print(json.dumps({k:result[k] for k in ('independent_histories','method_conditions','prediction_available','self_available','new_fits','new_optimizations')}))


if __name__ == '__main__':
    main()
