"""Exploratory mechanism diagnosis of one already observed development reversal.

Only development history 001, shift, guarded MC versus FC is authorized.
This is separate from predeclared evaluation attribution, changes no policy,
and cannot serve as independent confirmation.  Generic target/evaluation
checkpoint functions are reused without changing their predeclared caller.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np

from . import ev_mobile_v4_attribution as core


METHODS=('guarded__fixed_charge','guarded__mobile_charge')
INDEX=1
REGIME='shift'
TARGET_SLOTS=16
CANDIDATE_SLOTS=64


def ready(root, predeclared_output):
    root=Path(root);frozen=core._read(root/'freeze.json')
    if core.study.source_hashes()!=frozen['source_sha256']:
        raise ValueError('scientific source differs from the frozen study')
    completed=core._read(Path(predeclared_output)/'summary.json')
    if completed.get('completed_target_slots')!=64 or completed.get('complete_factor_groups')!=4:
        raise ValueError('finish the original predeclared attribution before this exploratory case')
    specs=[s for s in frozen['specifications']['development'] if s['index']==INDEX]
    if len(specs)!=1 or specs[0].get('regimes')!=[REGIME]:
        raise ValueError('the selected development history/regime differs from its freeze')
    path=root/'development'/'001.json.gz';raw=core._read(path)
    if raw.get('spec')!=specs[0] or raw.get('policy_id')!=frozen['policy_id'] or raw.get('status')!='complete':
        raise ValueError('complete saved development history with matching frozen identity required')
    return frozen,raw,core._source(path),core._source(Path(predeclared_output)/'summary.json')


def prepare(raw,frozen,source):
    began=perf_counter();spec=raw['spec'];truth=raw['true_name']
    case=core.cases.make_case(spec['bank'],spec['seed'],spec['family_index'],REGIME)
    D=core.lk.concat_observations(case['train'],case['validation'])
    if core.fingerprint(core.io.observations_record(D))!=core.fingerprint(raw['observed']):
        raise ValueError('regenerated development D differs from the saved observations')
    if any(not np.array_equal(raw['truth'][k],case['truth'][k]) for k in ('r','x')):
        raise ValueError('regenerated development true line parameters differ')
    bank=tuple(case['bank']);names=[c.name for c in bank]
    if len(names)!=4:raise ValueError('retain all four candidates')
    candidate=next(c for c in bank if c.name==truth)
    if candidate.supports!=case['truth']['candidate'].supports:raise ValueError('true support mismatch')
    trained={r.get('candidate',r.get('name')):r for r in raw['training_result']['fits']}
    if set(trained)!=set(names) or any(r.get('success') is not True for r in trained.values()):
        raise ValueError('all saved observed-D references required; no new training')
    for c in bank:
        if tuple(map(tuple,trained[c.name]['supports']))!=c.supports:raise ValueError('saved reference edge order mismatch')
    forecast=raw['forecast'];source_forecast=forecast['source_by_candidate'][truth]
    predicted=dict(r=trained[truth]['r'],x=trained[truth]['x'],p=forecast['p'],q=forecast['q'],
        sp=source_forecast['source_p'],sq=source_forecast['source_q'],root=forecast['root_v'])
    start=case['future_start'];end=start+spec['T']
    true=dict(r=raw['truth']['r'],x=raw['truth']['x'],p=case['p'][start:end],q=case['q'][start:end],
        sp=case['source_p'][start:end],sq=case['source_q'][start:end],root=case['root_v'][start:end])
    if not np.array_equal(predicted['root'],true['root']):raise ValueError('undeclared root prediction difference')
    powers={};keys={};nominal={};actual={};verified=[]
    for m in METHODS:
        method=raw['methods'][m]
        if method.get('operation',{}).get('feasible') is not True:raise ValueError('saved action physically invalid')
        powers[m]=np.asarray(method['power_kw'],float);keys[m]=method['action_key']
        observed=core.cases.observe_future(case,powers[m])
        saved=raw['actual'][REGIME]['actions'][keys[m]]['future']
        if core.fingerprint(core.io.observations_record(observed))!=core.fingerprint(saved):
            raise ValueError('regenerated noisy fixed-action future differs')
        nominal[m]=dict(Phi=method['nominal']['score'],true_ref_m=core._reference_gap(method,truth))
        actual[m]=core._actual_halfM(raw,REGIME,m,names);verified.append(m)
    cfg=raw['config'];fcfg=frozen['config']
    options=core.refit._options({**cfg['training'],**cfg['source'],'ratio_bounds':cfg['ratio_bounds'],
        'multistart':fcfg['multistart_predictive'],'max_nfev':fcfg['max_nfev']})
    return dict(schema='mobile_v4_exploratory_development_gap_history_v1',spec=spec,true_name=truth,
        candidate_names=names,candidate_bank=[dict(name=c.name,n=c.n,supports=c.supports) for c in bank],
        observed=raw['observed'],observed_meter=raw['observed_meter'],future_meter=raw['future_meter'],trained=trained,
        predicted=predicted,true_by_regime={REGIME:true},powers=powers,action_keys=keys,
        nominal=nominal,observed_halfM={REGIME:actual},fit_seed=spec['seed']+4000,fit_options=options,
        base_kw=cfg['base_kw'],raw_source=source,source_interface=1,observed_D_retrained=False,planning_called=False,
        preparation_seconds=perf_counter()-began,
        verification=dict(observed_D_identical=True,true_lines_identical=True,common_root_identical=True,
            saved_noisy_fixed_action_futures_identical=verified),
        selection_basis='already observed development mean-score increase and actual separation decrease',
        independent_confirmation=False,used_to_change_policy=False)


def event(meta,slots):
    values={m:{} for m in METHODS};states={};deltas={}
    for bits in core.STATES:
        key=''.join(map(str,bits));state={}
        for m in METHODS:
            row=slots.get((m,key),{});value=row.get('halfM') if row.get('status')=='complete' else None
            state[m]=dict(status=row.get('status','not_run'),halfM=value)
            if value is not None:values[m][bits]=value
        if all(state[m]['halfM'] is not None for m in METHODS):
            deltas[bits]=state[METHODS[1]]['halfM']-state[METHODS[0]]['halfM']
        states[key]=dict(methods=state,delta_halfM=deltas.get(bits))
    subtract=lambda source,key:None if any(source.get(m,{}).get(key) is None for m in METHODS) else source[METHODS[1]][key]-source[METHODS[0]][key]
    phi=subtract(meta.get('nominal',{}),'Phi');nominal=subtract(meta.get('nominal',{}),'true_ref_m')
    observed=meta.get('observed_halfM',{}).get(REGIME,{})
    actual=None if any(observed.get(m) is None for m in METHODS) else observed[METHODS[1]]-observed[METHODS[0]]
    result=dict(index=INDEX,regime=REGIME,methods=METHODS,states=states,complete=len(deltas)==8,
        delta_nominal_Phi=phi,delta_nominal_true_ref_m=nominal,delta_actual_halfM=actual,
        selection_basis=meta.get('selection_basis'),independent_confirmation=False,used_to_change_policy=False)
    if len(deltas)==8:
        split=core.shapley_change(deltas)
        aggregation=None if nominal is None or phi is None else nominal-phi
        implementation=None if nominal is None else deltas[0,0,0]-nominal
        residual=None if actual is None else actual-deltas[1,1,1]
        terms=[aggregation,implementation,*split['contributions'].values(),residual]
        result.update(delta_000=deltas[0,0,0],delta_111=deltas[1,1,1],Shapley_delta_halfM=split,
            Shapley_method_halfM={m:core.shapley_change(v) for m,v in values.items()},
            metric_aggregation_discrepancy=aggregation,nominal_vs_same_inference_rule_discrepancy=implementation,
            observed_minus_111_residual=residual,
            full_gap_decomposition_error=None if phi is None or actual is None or any(t is None for t in terms) else sum(terms)-(actual-phi),
            residual_meaning='conditional P/Q input error, voltage/root-PQ metering noise and numerical AC refit differences')
    return result


def summarize(output):
    output=Path(output);history=output/'history_001'/'history.json.gz';slots={};rows=[]
    if history.exists():
        meta=core._read(history)
        for m in METHODS:
            for bits in core.STATES:
                key=''.join(map(str,bits));path=output/'history_001'/REGIME/(m+'_'+key+'.json')
                if path.exists():row=core._read(path);rows.append(row);slots[m,key]=row
        effect=event(meta,slots);prep=dict(seconds=meta.get('preparation_seconds',0.),counters=meta.get('preparation_counters',{}))
    else:
        effect=dict(index=INDEX,regime=REGIME,complete=False,failure='history not prepared')
        prep=core._read(output/'preparation_failure.json') if (output/'preparation_failure.json').exists() else {}
    caches=[core._read(p) for p in sorted((output/'fit_cache').glob('*.json.gz'))]
    targets=[core._read(p) for p in sorted((output/'targets').glob('*.json.gz'))]
    count=Counter();count.update(prep.get('counters',{}))
    for row in [*caches,*targets]:count.update(row.get('counters',{}))
    candidate={k:sum(r.get('candidate_slots',{}).get(k,0) for r in rows) for k in ('expected','attempted','success','failed','not_run')}
    missing=TARGET_SLOTS-len(rows);candidate['expected']+=4*missing;candidate['not_run']+=4*missing
    unique=[core.outcome_values(c['outcome'],c['true_name'],c['candidate_names']) for c in caches]
    return dict(schema='mobile_v4_exploratory_development_gap_summary_v1',independent_confirmation=False,
        selected_after_observing_development_reversal=True,used_to_change_policy=False,
        history_denominator=1,regime=REGIME,target_slot_denominator=TARGET_SLOTS,completed_target_slots=len(rows),
        candidate_slot_denominator=CANDIDATE_SLOTS,candidate_slots=candidate,
        slot_status=dict(Counter(r['status'] for r in rows)),unique_target_generations=len(targets),
        unique_evaluation_calls=len(caches),unique_attempted_joint_candidate_posts=sum(v['candidate_slots']['attempted'] for v in unique),
        unique_successful_joint_candidate_posts=sum(v['candidate_slots']['success'] for v in unique),
        cached_slot_reuses=sum(r.get('cache_reused',False) for r in rows),
        fit_seconds=sum(c.get('seconds',0.) for c in caches),target_generation_seconds=sum(t.get('seconds',0.) for t in targets),
        preparation_seconds=prep.get('seconds',0.),counters=dict(count),event=effect)


def render(summary,path,status):
    effect=summary['event'];lines=['# 开发001探索性反转归因','',
        '本例因已观察到开发反转而选中，用于解释该例机制，不是独立确认，也未用于调整冻结策略。只比较原shift未来的guarded MC与FC原动作，不更换正式评价样本。','',
        f"状态{status}；目标{summary['completed_target_slots']}/16，联合候选槽成功{summary['candidate_slots']['success']}/64，失败{summary['candidate_slots']['failed']}，未运行{summary['candidate_slots']['not_run']}。",'']
    if effect.get('complete'):
        c=effect['Shapley_delta_halfM']['contributions']
        for label,key in [('平均预测Φ差','delta_nominal_Phi'),('真参考预测m差','delta_nominal_true_ref_m'),('相同推断规则000差','delta_000'),('真因素111差','delta_111'),('带噪实际半M差','delta_actual_halfM'),('聚合差','metric_aggregation_discrepancy'),('预测流程到相同推断规则差','nominal_vs_same_inference_rule_discrepancy'),('实测减111残余','observed_minus_111_residual')]:
            lines.append(f"- {label}：{effect.get(key)}")
        lines += ['',f"线路/背景/源贡献：{c['lines']} / {c['background']} / {c['source']}。",
            f"完整代数分解误差：{effect['full_gap_decomposition_error']}。",
            '实测减111残余包含条件P/Q输入、电压与总表计量噪声以及数值拟合差异。因子贡献按六种替换次序平均分配，未假设因素独立。']
    lines += ['',f"实际去重评价{summary['unique_evaluation_calls']}次，拟合{summary['fit_seconds']:.3f}s；完整目标、拟合状态、源码/输入哈希和调用费用见同目录JSON。"]
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf8')


def run(root,output,predeclared_output,*,budget_seconds=600.,resume=False):
    began=perf_counter();output=Path(output)
    if not np.isfinite(budget_seconds) or budget_seconds<=0:raise ValueError('positive finite budget required')
    frozen,raw,source,prior=ready(root,predeclared_output)
    if output.exists() and not resume:raise FileExistsError('retain previous exploration; use --resume')
    output.mkdir(parents=True,exist_ok=True)
    config=dict(schema='mobile_v4_exploratory_development_gap_freeze_v1',source_sha256=frozen['source_sha256'],
        policy_id=frozen['policy_id'],raw_source=source,completed_predeclared_attribution=prior,methods=METHODS,index=INDEX,regime=REGIME,
        source_file_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        generic_attribution_sha256=sha256(Path(core.__file__).read_bytes()).hexdigest(),
        protocol_sha256=sha256(Path(__file__).with_name('ev_mobile_v4_gap_case_protocol.md').read_bytes()).hexdigest(),
        selected_after_observing_development_reversal=True,independent_confirmation=False,used_to_change_policy=False)
    freeze=output/'freeze.json'
    if freeze.exists():
        if core._read(freeze)!=core.io.serial(config):raise ValueError('exploratory source/config changed across resume')
    else:core._write(freeze,config)
    meta_path=output/'history_001'/'history.json.gz';status='complete'
    if meta_path.exists():meta=core._read(meta_path)
    else:
        start=perf_counter();counters={}
        try:
            with core.dynamic.count_physics_calls() as counters:meta=prepare(raw,frozen,source)
            meta['preparation_counters']=dict(counters);core._write(meta_path,meta)
        except Exception as exc:
            core._write(output/'preparation_failure.json',dict(failure=type(exc).__name__+': '+str(exc),
                traceback=traceback.format_exc(),seconds=perf_counter()-start,counters=dict(counters)))
            meta=None;status='preparation_failure'
    if meta is not None:
        context=core.restore(meta)
        for bits in core.STATES:
            for method in METHODS:
                if perf_counter()-began>=budget_seconds:status='checkpointed_budget';break
                row=core.evaluate_slot(output,meta,context,REGIME,bits,method)
                core._write(output/'status.json',dict(status='running',last_slot=row,seconds=perf_counter()-began))
                print(core.io.serial(dict(state=row['state'],method=method,status=row['status'],elapsed_seconds=perf_counter()-began)),flush=True)
            if status=='checkpointed_budget':break
    if core.study.source_hashes()!=frozen['source_sha256']:raise RuntimeError('frozen science changed')
    summary=summarize(output)
    if status=='complete' and not summary['event']['complete']:status='complete_with_failures'
    summary['invocation_seconds']=perf_counter()-began
    core._write(output/'summary.json',summary);render(summary,output/'summary.md',status)
    core._write(output/'status.json',dict(status=status,summary=summary))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    core._write(output/('invocation_'+stamp+'.json'),dict(status=status,seconds=perf_counter()-began,resumed=resume))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--predeclared-output',required=True)
    parser.add_argument('--budget-seconds',type=float,default=600.);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();result=run(args.root,args.output,args.predeclared_output,budget_seconds=args.budget_seconds,resume=args.resume)
    print(dict(completed_target_slots=result['completed_target_slots'],candidate_slots=result['candidate_slots'],
        event_complete=result['event']['complete']),flush=True)


if __name__=='__main__':main()
