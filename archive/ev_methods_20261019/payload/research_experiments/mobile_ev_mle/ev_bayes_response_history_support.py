"""Posthoc historical support at fixed ML and bounded-posterior line centers.

Only observed T/V meters enter this computation.  P/Q inputs are their bias-
corrected readings, while a source at the same physical interface is fitted
per time.  This conditional-output local profile differs from the Bayesian
latent-P/Q integral.  No future, topology truth or action enters the fit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .ev_bayes_response_analysis import read, serial, coverage
from .ev_bayes_response_diagnostics import _candidate, _line_center, _finite, _require
from .likelihood import Observations, MeasurementModel


OUTPUTS = ('v', 'master_p', 'master_q')
OBS_FIELDS = ('p', 'q', 'v', 'root_v', 'master_p', 'master_q', 'source_p', 'source_q')
FIT_FIELDS = ('r_bounds', 'x_bounds', 'source_p_bounds', 'source_q_bounds', 'voltage_bounds')


def history_inputs(record):
    """Broadcast on each ORIGINAL block, then concatenate along time."""
    blocks = [record['observed_training'], record['observed_evidence']]
    meters = [record['train_meter'], record['evidence_meter']]
    H, V = (len(row['p']) for row in blocks); n = len(blocks[0]['p'][0])
    obs = {}
    for field in OBS_FIELDS:
        values = [row.get(field) for row in blocks]
        _require(all(v is None for v in values) or all(v is not None for v in values), 'both observed blocks must share channels')
        obs[field] = None if values[0] is None else np.concatenate([_finite(v) for v in values])
    combined = dict(fit_bias=False, calibration=None)
    for channel in ('p', 'q', 'v', 'master_p', 'master_q', 'source_p', 'source_q'):
        values = [meter[channel+'_sigma'] for meter in meters]
        _require(all(v is None for v in values) or all(v is not None for v in values), 'both meter blocks must share channels')
        if values[0] is None:
            combined[channel+'_sigma'] = None
        else:
            combined[channel+'_sigma'] = np.concatenate([np.broadcast_to(_finite(v), (T, n) if channel in ('p', 'q', 'v') else (T,)).copy()
                for v, T in zip(values, (H, V))])
    for channel in ('p', 'q', 'v'):
        combined['bias_'+channel] = np.concatenate([np.broadcast_to(_finite(meter['bias_'+channel]), (T, n)).copy()
            for meter, T in zip(meters, (H, V))])
    _require(all(not m['fit_bias'] and m['calibration'] is None for m in meters), 'known independent biases required')
    return Observations(**obs), MeasurementModel(**combined), H, V


def source_fit(candidate, observations, meter, r, x, *, options, multistart, max_nfev, seed):
    from .ev_fixed_train_fit import fit_fixed_inputs
    # Identical automatic master-minus-terminal start and paired restart draws
    # for both line centers; no future or Bayesian source center is supplied.
    return fit_fixed_inputs(candidate, observations, meter, fixed_r=r, fixed_x=x,
        **options, multistart=multistart, max_nfev=max_nfev, seed=seed, tol=1e-10)


def replay_profile(candidate, fit, observations, meter, H):
    """Independent branch-current AC/SSE replay of the saved source optimum."""
    from .ev_bayes_small_audit import _physics
    p = observations.p-np.broadcast_to(meter.bias_p, observations.p.shape)
    q = observations.q-np.broadcast_to(meter.bias_q, observations.q.shape)
    latent = np.c_[p, q, fit.source_p, fit.source_q]
    physical = _physics(candidate, fit.r, fit.x, latent, observations.root_v)
    blocks = {}; total = 0.
    for label, slc in (('training', slice(0, H)), ('evidence', slice(H, None)), ('all', slice(None))):
        channels = {}
        for channel in OUTPUTS:
            value = physical[channel]+(np.broadcast_to(meter.bias_v, observations.v.shape) if channel == 'v' else 0.)
            sigma = np.broadcast_to(getattr(meter, channel+'_sigma'), value.shape)
            residual = (getattr(observations, channel)-value)/sigma
            channels[channel] = float(np.sum(residual[slc]**2))
        blocks[label] = dict(channel_SSE=channels, SSE=sum(channels.values()), half_SSE=.5*sum(channels.values()))
    total = blocks['all']['SSE']
    return dict(blocks=blocks, SSE=total, half_SSE=.5*total, saved_loss_identity_error=abs(total-fit.loss),
        ac_equation_residual=float(physical['equation']), power_balance_residual=float(physical['balance']),
        min_voltage=float(np.min(physical['v'])), max_voltage=float(np.max(physical['v'])))


def _ratio_check(r, x, bounds):
    if bounds is None: return dict(checked=False)
    ratio = _finite(x)/_finite(r); low, high = bounds
    return dict(checked=True, feasible=bool(np.all(ratio >= low-1e-12) and np.all(ratio <= high+1e-12)),
        min_ratio=float(np.min(ratio)), max_ratio=float(np.max(ratio)), bounds=bounds)


def support_record(record, *, multistart=2, max_nfev=100, fitter=None):
    began = perf_counter(); fitter = source_fit if fitter is None else fitter
    result = dict(spec=record.get('spec'), status=record.get('status'), candidates=[], failures=[],
        observed_only=True, topology_truth_used=False, future_used=False, source_interface=1,
        line_centers_fixed=True, P_Q_inputs='bias-corrected T+V readings held fixed',
        objective='conditional v/master P/master Q Gaussian output half-SSE; no P/Q density, process or line prior',
        multistart=multistart, max_nfev=max_nfev, global_optimality_certified=False)
    names = record.get('candidate_names', [])
    result['denominators'] = dict(candidate_slots_expected=len(names), line_center_fit_slots_expected=2*len(names),
        fit_slots_attempted=0, fit_slots_successful=0, fit_slots_failed=0, source_solver_calls=0, ac_evaluations=0)
    try:
        observed, meter, H, V = history_inputs(record)
        joint = record['prediction']['joint']; options = {k: joint['options'][k] for k in FIT_FIELDS}
        fits = {f.get('candidate', f.get('name')): f for f in record['training_result']['fits']}
        result.update(training_count=H, evidence_count=V, residual_count=(H+V)*(observed.n+2),
            source_variables=2*(H+V), fit_options=options,
            restart_seed_rule='SeedSequence([history_seed, 7100, candidate_index]); same starts at both line centers')
        for index, name in enumerate(names):
            row = dict(candidate=name, centers={}, failures=[]); result['candidates'].append(row)
            try:
                trained = fits[name]; candidate = _candidate(trained, observed.n)
                updated = joint['candidate_updates'][name]
                centers = dict(train_ML=tuple(_finite(trained[k]) for k in ('r', 'x')),
                    bounded_posterior=_line_center(updated, candidate.m))
            except Exception as exc:
                failure = dict(candidate=name, stage='candidate_input', failure=type(exc).__name__+': '+str(exc))
                row['failures'].append(failure); result['failures'].append(failure); continue
            seed = int(np.random.SeedSequence([int(record['spec']['seed']), 7100, index]).generate_state(1)[0])
            for label, (r, x) in centers.items():
                result['denominators']['fit_slots_attempted'] += 1
                try:
                    ratio = _ratio_check(r, x, joint['options'].get('ratio_bounds'))
                    _require(not ratio['checked'] or ratio['feasible'], 'fixed center outside original cable ratio domain')
                    fit = fitter(candidate, observed, meter, r, x, options=options,
                        multistart=multistart, max_nfev=max_nfev, seed=seed)
                    diagnostics = fit.diagnostics
                    result['denominators']['source_solver_calls'] += diagnostics.get('solver_calls', 0)
                    result['denominators']['ac_evaluations'] += diagnostics.get('ac_evaluations', 0)
                    saved = dict(success=bool(fit.success), SSE=float(fit.loss) if np.isfinite(fit.loss) else None,
                        half_SSE=.5*float(fit.loss) if np.isfinite(fit.loss) else None,
                        r=r, x=x, ratio_domain=ratio, status=int(fit.status), message=str(fit.message),
                        source_state=None, diagnostics=diagnostics)
                    if np.isfinite(fit.source_p).all() and np.isfinite(fit.source_q).all():
                        saved['source_state'] = dict(source_p=fit.source_p, source_q=fit.source_q)
                        saved['independent_replay'] = replay_profile(candidate, fit, observed, meter, H)
                    row['centers'][label] = saved
                    if fit.success:
                        result['denominators']['fit_slots_successful'] += 1
                    else:
                        failure = dict(candidate=name, center=label, stage='local_source_fit', failure=str(fit.message))
                        row['failures'].append(failure); result['failures'].append(failure)
                except Exception as exc:
                    failure = dict(candidate=name, center=label, stage='fixed_center_source_profile', failure=type(exc).__name__+': '+str(exc))
                    row['failures'].append(failure); result['failures'].append(failure)
            if all(row['centers'].get(k, {}).get('success') for k in ('train_ML', 'bounded_posterior')):
                a, b = (row['centers'][k]['SSE'] for k in ('train_ML', 'bounded_posterior'))
                row['SSE_increase_bounded_minus_ML'] = b-a
                row['half_SSE_increase_bounded_minus_ML'] = .5*(b-a)
                row['SSE_ratio_bounded_over_ML'] = b/a if a > 0 else None
                row['interpretation'] = 'local conditional measured-PQ historical support comparison; no universal rejection threshold assigned'
    except Exception as exc:
        result['failures'].append(dict(stage='observed_inputs', failure=type(exc).__name__+': '+str(exc)))
    count = result['denominators']; count['fit_slots_failed'] = count['line_center_fit_slots_expected']-count['fit_slots_successful']
    result['success'] = not result['failures'] and count['fit_slots_successful'] == count['line_center_fit_slots_expected']
    result['seconds'] = perf_counter()-began
    return result


def markdown(result):
    lines = ['# 固定线路中心的历史AC支持检查', '',
        f"阶段{result['phase']}；记录{len(result['records'])}/{result['coverage']['expected_histories']}。", '',
        '训练MLE线路与有界TV后验线路中心分别固定，用相同历史T+V计量、相同独立校准偏差及相同源初值，重新拟合每时段同接口源。',
        '目标是实测P/Q给定时的电压及总表P/Q标准化平方误差。P/Q计量误差在此只通过输入产生影响；该目标不同于Bayes潜在P/Q积分证据。只比较两线路中心的条件历史支持，不据此套用未经校准的拒绝阈值。', '',
        '|历史|候选|训练线SSE|后验线SSE|SSE增加|', '|---:|---|---:|---:|---:|']
    for history in result['records']:
        for row in history['candidates']:
            centers = row['centers']; a = centers.get('train_ML', {}).get('SSE'); b = centers.get('bounded_posterior', {}).get('SSE')
            fmt = lambda v: '—' if v is None else f'{v:.5g}'
            lines.append('|'+str(history.get('spec', {}).get('index'))+'|'+row['candidate']+'|'+fmt(a)+'|'+fmt(b)+'|'+fmt(row.get('SSE_increase_bounded_minus_ML'))+'|')
    lines += ['', '所有候选、源拟合失败、局部求解状态和分母保留。每个保存源状态用独立branch-current AC公式复算。局部求解器成功对应可达到的profile值，全局最小值未认证。', '']
    return '\n'.join(lines)


def run(directory, phase, output, *, multistart=2, max_nfev=100):
    directory, output = Path(directory), Path(output)
    if (output/'history_support.json').exists(): raise FileExistsError(output/'history_support.json')
    freeze = read(directory/'freeze.json')
    expected = [s for s in freeze['specifications'][phase] if (s['index'] in (0, 1) if phase == 'development' else s.get('legacy_audit', False))]
    records, present, inputs = [], [], []
    for spec in expected:
        path = directory/phase/('history_'+str(spec['index']).zfill(3)+'.json.gz')
        if not path.exists(): continue
        raw = read(path); _require(raw.get('policy_id') == freeze['policy_id'], 'record/freeze policy identity')
        records.append(support_record(raw, multistart=multistart, max_nfev=max_nfev)); present.append(raw['spec'])
        inputs.append(dict(file=str(path.resolve()), sha256=sha256(path.read_bytes()).hexdigest()))
        del raw
    result = dict(schema='ev_bayes_response_fixed_center_history_support_v1', phase=phase,
        policy_id=freeze['policy_id'], coverage=coverage(expected, present), records=records,
        input_files=inputs, tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        source_freeze_sha256=sha256((directory/'freeze.json').read_bytes()).hexdigest(),
        source_fit_slots_expected=sum(r['denominators']['line_center_fit_slots_expected'] for r in records),
        future_or_truth_used=False)
    output.mkdir(parents=True, exist_ok=True)
    (output/'history_support.json').write_text(json.dumps(serial(result), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    (output/'history_support.md').write_text(markdown(result), encoding='utf-8')
    print('RESPONSE_HISTORY_SUPPORT_SAVED', phase, result['coverage'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=('development', 'evaluation'), default='development')
    parser.add_argument('--multistart', type=int, default=2); parser.add_argument('--max-nfev', type=int, default=100)
    args = parser.parse_args()
    if args.multistart < 1 or args.max_nfev < 1: parser.error('positive solver budgets required')
    run(args.input, args.phase, args.output, multistart=args.multistart, max_nfev=args.max_nfev)


if __name__ == '__main__': main()
