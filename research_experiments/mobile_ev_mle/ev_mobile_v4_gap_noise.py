"""Frozen conditional future-noise diagnosis of development001/shift MC vs FC.

One historical data set and the two saved actions stay fixed.  Only five
future metering white-noise arrays vary across the declared 32 seeds.
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
from . import ev_mobile_v4_gap_case as gap


METHODS = gap.METHODS
SEEDS = tuple(range(2026102200, 2026102232))
CHANNELS = ('p', 'q', 'v', 'master_p', 'master_q')
TIE_TOL = 1.e-9
TARGET_SLOTS = len(SEEDS)*len(METHODS)
CANDIDATE_SLOTS = 4*TARGET_SLOTS


def white_noise(seed, shapes):
    """Five independent channels, common across actions at one replicate."""
    if tuple(shapes) != CHANNELS:
        raise ValueError('five channel shapes must follow the frozen order')
    streams = np.random.SeedSequence(int(seed)).spawn(len(CHANNELS))
    return {k:np.random.default_rng(s).standard_normal(tuple(shapes[k]))
            for k,s in zip(CHANNELS, streams)}


def add_white(clean, meter, white):
    """Add random error only; clean data already contain EV and known biases."""
    if set(white) != set(CHANNELS): raise ValueError('all five white-noise channels required')
    result = dict(clean)
    for key in CHANNELS:
        value, eps = np.asarray(clean[key],float), np.asarray(white[key],float)
        sigma = np.asarray(getattr(meter,key+'_sigma'),float)
        if eps.shape != value.shape: raise ValueError('white shape differs: '+key)
        if not np.isfinite(eps).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError('finite white and positive finite sigma required: '+key)
        result[key] = value+sigma*eps
    return core.io.observations_record(core._observation(result))


def readiness(root, gap_output):
    root, gap_output = Path(root), Path(gap_output)
    frozen = core._read(root/'freeze.json')
    if core.study.source_hashes() != frozen['source_sha256']:
        raise ValueError('128 scientific sources differ from their freeze')
    summary = core._read(gap_output/'summary.json')
    if summary.get('completed_target_slots') != 16 or not summary.get('event',{}).get('complete'):
        raise ValueError('complete the separately authorized development gap case first')
    meta_path = gap_output/'history_001'/'history.json.gz'; meta = core._read(meta_path)
    if meta['spec']['index'] != 1 or set(meta['powers']) != set(METHODS) or meta['spec']['regimes'] != ['shift']:
        raise ValueError('only saved development001 shift MC/FC actions authorized')
    if meta['observed_D_retrained'] or meta['planning_called']:
        raise ValueError('reuse saved historical references and actions')
    clean, sources = {}, {}
    for method in METHODS:
        slot_path = gap_output/'history_001'/'shift'/(method+'_111.json')
        slot = core._read(slot_path); source = slot['target_source']; path = Path(source['path'])
        if core._source(path) != source: raise ValueError('saved clean111 target changed')
        target = core._read(path)
        request = core.target_identity(meta,meta['true_by_regime']['shift'],meta['powers'][method])
        if not target.get('success') or target.get('target_request') != request:
            raise ValueError('saved target is not the exact fixed-action true111 target')
        clean[method] = target['observations']; sources[method] = source
    if any(clean[m][k] is not None for m in METHODS for k in ('source_p','source_q')):
        raise ValueError('this declared case has no independent source meter')
    shapes = {k:np.asarray(clean[METHODS[0]][k]).shape for k in CHANNELS}
    if any(np.asarray(clean[m][k]).shape != shapes[k] for m in METHODS for k in CHANNELS):
        raise ValueError('the paired actions require identical meter array shapes')
    if shapes['p'] != (12,3) or shapes['master_p'] != (12,):
        raise ValueError('retain declared12-by3 future')
    for m in METHODS:
        if not np.array_equal(clean[m]['root_v'],meta['true_by_regime']['shift']['root']):
            raise ValueError('fixed calibrated root changed')
    inputs = dict(meta=meta,clean=clean,shapes=shapes,
        history_source=core._source(meta_path),gap_summary_source=core._source(gap_output/'summary.json'),
        clean_sources=sources,noiseless111_delta=summary['event']['delta_111'],
        original_observed_delta=summary['event']['delta_actual_halfM'])
    return frozen, inputs


def evaluate_slot(output, meta, context, clean, seed, method, *, evaluate=None):
    """Atomic paired-action leaf; failed fits are cached and remain in counts."""
    output = Path(output); path = output/'replicates'/str(seed)/(method+'.json')
    if path.exists(): return core._read(path)
    noise_path = output/'white_noise'/(str(seed)+'.json.gz')
    white_record = core._read(noise_path)
    target_path = output/'targets'/str(seed)/(method+'.json.gz')
    if not target_path.exists():
        began = perf_counter()
        try:
            obs = add_white(clean,context['fm'],white_record['white'])
            target = dict(success=True,observations=obs,white_source=core._source(noise_path),
                clean_observation_sha256=core.fingerprint(clean),history_sha256=core.fingerprint(meta['observed']))
        except Exception as exc:
            target = dict(success=False,failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        target['seconds'] = perf_counter()-began; core._write(target_path,target)
    target = core._read(target_path)
    slot = dict(seed=seed,method=method,history_index=1,regime='shift',expected_candidate_slots=4,
        target_source=core._source(target_path),white_sha256=white_record['white_sha256'],cache_reused=False)
    if not target.get('success'):
        slot.update(status='target_failure',failure=target.get('failure'),
            candidate_slots=dict(expected=4,attempted=0,success=0,failed=0,not_run=4))
        core._write(path,slot); return slot
    identity = core.fit_identity(meta,target['observations']); fit_path = output/'fit_cache'/(identity+'.json.gz')
    if fit_path.exists(): fitted = core._read(fit_path); slot['cache_reused'] = True
    else:
        began = perf_counter()
        with core.dynamic.count_physics_calls() as counters:
            try:
                evaluator = core.study.evaluate_future if evaluate is None else evaluate
                result = evaluator(context['bank'],context['D'],context['dm'],core._observation(target['observations']),
                    context['fm'],context['fits'],meta['fit_options'],meta['fit_seed'])
                fitted = dict(outcome=result,fit_identity=identity,fit_seed=meta['fit_seed'],
                    true_name=meta['true_name'],candidate_names=meta['candidate_names'])
            except Exception as exc:
                fitted = dict(outcome=dict(complete=False,comparisons={},failures={}),
                    failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),fit_identity=identity,
                    true_name=meta['true_name'],candidate_names=meta['candidate_names'])
        fitted.update(seconds=perf_counter()-began,counters=dict(counters)); core._write(fit_path,fitted)
    values = core.outcome_values(fitted['outcome'],meta['true_name'],meta['candidate_names'])
    slot.update(status='complete' if values['complete'] else 'fit_failure',**values,
        fit_identity=identity,fit_source=core._source(fit_path),failure=fitted.get('failure'),
        fit_failures=fitted['outcome'].get('failures',{}))
    core._write(path,slot); return slot


def paired_statistics(slots, inputs):
    pairs = []
    for seed in SEEDS:
        methods = {m:slots.get((seed,m),{}) for m in METHODS}
        values = {m:r.get('halfM') if r.get('status')=='complete' else None for m,r in methods.items()}
        complete = all(v is not None and np.isfinite(v) for v in values.values())
        pairs.append(dict(seed=seed,complete=complete,halfM=values,
            delta_halfM=values[METHODS[1]]-values[METHODS[0]] if complete else None,
            status={m:r.get('status','not_run') for m,r in methods.items()}))
    values = np.asarray([r['delta_halfM'] for r in pairs if r['complete']],float)
    result = dict(history_denominator=1,conditional_repeat_denominator=len(SEEDS),complete_pairs=len(values),
        incomplete_pairs=len(SEEDS)-len(values),pairs=pairs,noiseless111_delta=inputs['noiseless111_delta'],
        original_observed_delta=inputs['original_observed_delta'],tie_tolerance=TIE_TOL)
    if len(values):
        mean = float(np.mean(values)); sd = float(np.std(values,ddof=1)) if len(values)>1 else None
        result.update(mean=mean,sample_sd=sd,Monte_Carlo_SE=None if sd is None else sd/np.sqrt(len(values)),
            quantiles={str(q):float(np.quantile(values,q)) for q in (.05,.25,.5,.75,.95)},
            minimum=float(np.min(values)),maximum=float(np.max(values)),
            win=int(np.count_nonzero(values>TIE_TOL)),tie=int(np.count_nonzero(np.abs(values)<=TIE_TOL)),
            loss=int(np.count_nonzero(values < -TIE_TOL)),
            mean_minus_noiseless111=mean-inputs['noiseless111_delta'],
            original_minus_conditional_mean=inputs['original_observed_delta']-mean,
            fraction_at_or_below_original=float(np.mean(values <= inputs['original_observed_delta'])),
            original_deviation_in_sample_sd=None if sd is None or sd==0 else (inputs['original_observed_delta']-mean)/sd)
    return result


def summarize(output):
    output = Path(output); input_path = output/'inputs.json.gz'; rows=[]; slots={}
    inputs = core._read(input_path) if input_path.exists() else dict(noiseless111_delta=None,original_observed_delta=None)
    for seed in SEEDS:
        for method in METHODS:
            path = output/'replicates'/str(seed)/(method+'.json')
            if path.exists(): row=core._read(path);rows.append(row);slots[seed,method]=row
    candidate = {k:sum(r.get('candidate_slots',{}).get(k,0) for r in rows)
                 for k in ('expected','attempted','success','failed','not_run')}
    missing=TARGET_SLOTS-len(rows); candidate['expected']+=4*missing; candidate['not_run']+=4*missing
    caches=[core._read(p) for p in sorted((output/'fit_cache').glob('*.json.gz'))]
    targets=[core._read(p) for p in sorted((output/'targets').glob('*/*.json.gz'))]
    count=Counter()
    for c in caches:count.update(c.get('counters',{}))
    unique=[core.outcome_values(c['outcome'],c['true_name'],c['candidate_names']) for c in caches]
    return dict(schema='mobile_v4_development_conditional_noise_summary_v1',history_denominator=1,
        independent_confirmation=False,selected_after_observing_development_reversal=True,used_to_change_policy=False,
        target_slot_denominator=TARGET_SLOTS,completed_target_slots=len(rows),candidate_slot_denominator=CANDIDATE_SLOTS,
        candidate_slots=candidate,slot_status=dict(Counter(r['status'] for r in rows)),
        unique_evaluation_calls=len(caches),cached_slot_reuses=sum(r.get('cache_reused',False) for r in rows),
        unique_attempted_joint_candidate_posts=sum(v['candidate_slots']['attempted'] for v in unique),
        unique_successful_joint_candidate_posts=sum(v['candidate_slots']['success'] for v in unique),
        fit_seconds=sum(c.get('seconds',0.) for c in caches),target_generation_seconds=sum(t['seconds'] for t in targets),
        counters=dict(count),statistics=paired_statistics(slots,inputs))


def render(summary,path,status):
    s=summary['statistics'];lines=['# 开发001：固定动作的条件计量噪声重复','',
        '仅一份开发历史。真线路、负荷、源与MC/FC原动作固定，只重抽未来五个计量通道。32个条件未来不计为32份独立历史，未用于调整策略。','',
        f"状态{status}；动作槽{summary['completed_target_slots']}/64，候选成功{summary['candidate_slots']['success']}/256，失败{summary['candidate_slots']['failed']}，未运行{summary['candidate_slots']['not_run']}。",
        f"完整配对{s['complete_pairs']}/32。111无噪声差{s['noiseless111_delta']}；原带噪差{s['original_observed_delta']}。",'']
    if 'mean' in s:
        lines += [f"- MC减FC的半M差：均值{s['mean']}，样本SD{s['sample_sd']}，Monte Carlo SE{s['Monte_Carlo_SE']}。",
            f"- 胜/平/负：{s['win']}/{s['tie']}/{s['loss']}；分位数{s['quantiles']}。",
            f"- 原差减条件均值：{s['original_minus_conditional_mean']}；原差在本条件样本中的下侧频率：{s['fraction_at_or_below_original']}。",
            '该频率描述固定历史与轨迹下的未来计量实现，不是拓扑正确概率。所有失败保留分母；完整拟合、白噪声及费用见JSON。']
    lines += ['',f"实际调用evaluate_future {summary['unique_evaluation_calls']}次；拟合{summary['fit_seconds']:.3f}s。"]
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf8')


def run(root,output,gap_output,*,budget_seconds=600.,resume=False):
    began=perf_counter();output=Path(output)
    if not np.isfinite(budget_seconds) or budget_seconds<=0:raise ValueError('positive finite budget required')
    frozen,inputs=readiness(root,gap_output)
    if output.exists() and not resume:raise FileExistsError('retain previous conditional diagnostic; use --resume')
    output.mkdir(parents=True,exist_ok=True)
    config=dict(schema='mobile_v4_development_conditional_noise_freeze_v1',policy_id=frozen['policy_id'],
        source_sha256=frozen['source_sha256'],seeds=SEEDS,methods=METHODS,channels=CHANNELS,tie_tolerance=TIE_TOL,
        history_index=1,regime='shift',history_denominator=1,target_slots=TARGET_SLOTS,candidate_slots=CANDIDATE_SLOTS,
        fit_seed=inputs['meta']['fit_seed'],inputs_sha256=core.fingerprint(inputs),
        source_file_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        generic_attribution_sha256=sha256(Path(core.__file__).read_bytes()).hexdigest(),
        prior_gap_module_sha256=sha256(Path(gap.__file__).read_bytes()).hexdigest(),
        protocol_sha256=sha256(Path(__file__).with_name('ev_mobile_v4_gap_noise_protocol.md').read_bytes()).hexdigest(),
        independent_confirmation=False,used_to_change_policy=False,selector_called=False,training_called=False,
        paired_white=True,noise_changes_only_future_meters=True)
    freeze=output/'freeze.json'
    if freeze.exists():
        if core._read(freeze)!=core.io.serial(config):raise ValueError('conditional diagnostic changed across resume')
    else:core._write(freeze,config)
    input_path=output/'inputs.json.gz'
    if input_path.exists():
        if core.fingerprint(core._read(input_path))!=config['inputs_sha256']:raise ValueError('saved fixed inputs changed')
    else:core._write(input_path,inputs)
    meta=inputs['meta'];context=core.restore(meta);D_identity=core.fingerprint(core.io.observations_record(context['D']))
    status='complete'
    for seed in SEEDS:
        noise_path=output/'white_noise'/(str(seed)+'.json.gz')
        white=white_noise(seed,inputs['shapes']);white_identity=core.fingerprint(white)
        if noise_path.exists():
            saved=core._read(noise_path)
            if saved['white_sha256']!=white_identity or core.fingerprint(saved['white'])!=white_identity:
                raise ValueError('paired frozen white noise changed')
        else:core._write(noise_path,dict(seed=seed,white=white,white_sha256=white_identity))
        for method in METHODS:
            if perf_counter()-began>=budget_seconds:status='checkpointed_budget';break
            row=evaluate_slot(output,meta,context,inputs['clean'][method],seed,method)
            if core.fingerprint(core.io.observations_record(context['D']))!=D_identity:
                raise RuntimeError('future diagnostic mutated historical observations')
            core._write(output/'status.json',dict(status='running',seed=seed,method=method,last_status=row['status'],seconds=perf_counter()-began))
            print(dict(seed=seed,method=method,status=row['status'],elapsed_seconds=perf_counter()-began),flush=True)
        if status=='checkpointed_budget':break
    if core.study.source_hashes()!=frozen['source_sha256']:raise RuntimeError('frozen128 scientific sources changed')
    summary=summarize(output)
    if status=='complete' and summary['statistics']['complete_pairs']!=len(SEEDS):status='complete_with_failures'
    summary['invocation_seconds']=perf_counter()-began
    core._write(output/'summary.json',summary);render(summary,output/'summary.md',status)
    core._write(output/'status.json',dict(status=status,summary=summary))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    core._write(output/('invocation_'+stamp+'.json'),dict(status=status,seconds=perf_counter()-began,resumed=resume))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--gap-output',required=True)
    parser.add_argument('--budget-seconds',type=float,default=600.);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();summary=run(args.root,args.output,args.gap_output,budget_seconds=args.budget_seconds,resume=args.resume)
    print(dict(completed_target_slots=summary['completed_target_slots'],candidate_slots=summary['candidate_slots'],
        complete_pairs=summary['statistics']['complete_pairs']),flush=True)


if __name__=='__main__':main()
