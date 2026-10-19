"""Additional relative/absolute-error gates; the primary alpha=.1 rule is intact.

For candidate k, R_k=S_k-min_j S_j and A_k=S_k/N, with N the COUNT of
conditional V/root-PQ output observations, not residual degrees of freedom.
Adding a common constant to all S leaves R invariant: [1,2] and [1001,1002]
both give [0,1].  A retains this shared deterioration.

Calibrate true-candidate R and A separately at alpha_R=alpha_A=.05.  Retain k
iff R_k<=tau_R and A_k<=tau_A.  An incomplete run keeps all and has BOTH
calibration scores zero.  For IID calibration/test histories and an entirely
frozen procedure with the true candidate present, the n+1 order-statistic
argument gives P(R_true>tau_R)<=alpha_R and P(A_true>tau_A)<=alpha_A.
Union bound gives marginal false rejection <=.1, without independence of R/A.
The result concerns the declared finite-mixture distribution, not each regime.

Within one run N is common to all candidates, so the set also equals
  {k:S_k<=min(S_min+tau_R,N*tau_A)}.
It is empty exactly when S_min/N>tau_A (tau_R>=0).  Empty means every candidate
failed the calibrated absolute-error gate; it does not diagnose the cause.
Pressure cases outside the calibration mixture receive empirical diagnostics.

This analysis imports only the standard-library-only analyzer, never AC/fits.
Its auxiliary thresholds are durably saved before opening evaluation records.
It never changes the original ten-policy primary thresholds or observations.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import math
from pathlib import Path

from . import ev_systematic_analyze as base


ALPHA_R=.05
ALPHA_A=.05


def output_count(record,method,inference):
    """Actual conditional output count: T*(n terminal V + root P + root Q)."""
    if inference not in base.INFERENCE:raise ValueError('unknown inference mode')
    config=record.get('config',{});power=method.get('power_kw')
    H,V,T=(config.get(k) for k in ('training_points','validation_points','horizon'))
    if any(not isinstance(v,int) or isinstance(v,bool) for v in (H,V,T)) or H<=0 or V<0 or T<=0:
        raise ValueError('positive training/horizon and nonnegative validation counts required')
    if not isinstance(power,list) or len(power)!=T or not power or not isinstance(power[0],list) or not power[0]:
        raise ValueError('power_kw must have the declared (horizon,terminal) shape')
    n=len(power[0])
    if any(not isinstance(row,list) or len(row)!=n or any(not base._finite(p) for p in row) for row in power):
        raise ValueError('finite rectangular power curve required')
    return ((V+T) if inference=='fixed_holdout' else (H+V+T))*(n+2)


def scores(record,name,inference):
    method=base._methods(record)[name];value=base._mode(record,name,inference)
    error=None;N=None
    try:N=output_count(record,method,inference)
    except ValueError as exc:error=str(exc)
    complete=bool(value['complete'] and N is not None)
    candidate_scores={}
    if complete:
        minimum=min(value['losses'].values())
        candidate_scores={k:dict(R=max(0.,S-minimum),A=S/N) for k,S in value['losses'].items()}
    true=record['true_name']
    return dict(complete=complete,candidate_miss=value['candidate_miss'],bank=value['bank'],
        Nobs=N,candidate_scores=candidate_scores,
        R_true=candidate_scores[true]['R'] if complete and not value['candidate_miss'] else 0.,
        A_true=candidate_scores[true]['A'] if complete and not value['candidate_miss'] else 0.,
        metadata_failure=error,source_complete=value['complete'])


def calibrate(phase,output,*,alpha_R=ALPHA_R,alpha_A=ALPHA_A,expected_count=None):
    rows=phase['records']
    if phase['phase']!='calibration':raise ValueError('calibration phase required')
    if expected_count is not None and len(rows)!=expected_count:raise ValueError('incomplete declared calibration batch')
    if not rows or len({r['seed'] for r in rows})!=len(rows):raise ValueError('independent unique calibration history seeds required')
    if not 0<alpha_R<1 or not 0<alpha_A<1 or alpha_R+alpha_A>1:
        raise ValueError('positive gate alphas with sum<=1 required')
    identity=phase['identity']
    if not any(k in identity for k in ('source_id','source_sha256_digest')) or not any(k in identity for k in ('config_id','policy_id','manifest_config_digest')):
        raise ValueError('declared frozen source and configuration identities required')
    methods={}
    for name in base.METHODS:
        methods[name]={}
        for inference in base.INFERENCE:
            values=[scores(r,name,inference) for r in rows]
            if any(v['candidate_miss'] for v in values):raise ValueError('formal calibration prohibits candidate miss')
            methods[name][inference]=dict(R=base.conformal_threshold([v['R_true'] for v in values],alpha_R),
                A=base.conformal_threshold([v['A_true'] for v in values],alpha_A),
                incomplete_calibration=sum(not v['complete'] for v in values),
                metadata_failures=sum(v['metadata_failure'] is not None for v in values),
                R_scores=[v['R_true'] for v in values],A_scores=[v['A_true'] for v in values],
                Nobs_counts=dict(Counter(str(v['Nobs']) for v in values)))
    declared=phase.get('manifest',{}).get('config',{}).get('formal_cases')
    result=dict(schema='ev_systematic_auxiliary_adequacy_v1',identity=identity,calibration_count=len(rows),
        calibration_files=phase['files'],methods=methods,alpha_R=alpha_R,alpha_A=alpha_A,
        marginal_false_rejection_bound=alpha_R+alpha_A,
        failure_rule='incomplete or illegal run keeps all; both true calibration scores zero',
        declared_calibration_cases=declared,
        calibration_case_scope='declared IID mixture' if declared is not None else 'support not supplied',
        primary_thresholds_modified=False,additional_diagnostic=True,
        thresholds_saved_before_evaluation_read=True,
        source_sha256=dict(adequacy=sha256(Path(__file__).read_bytes()).hexdigest(),
            analyzer=sha256(Path(base.__file__).read_bytes()).hexdigest()),
        guarantee_scope='marginal IID frozen full procedure; no per-regime or pressure-distribution guarantee')
    base._write(Path(output)/'adequacy_thresholds.json',result,immutable=True)
    return result


def _limit(record):
    return math.inf if record['infinite'] else record['threshold']


def decision(record,name,inference,threshold):
    value=scores(record,name,inference);true=record['true_name']
    if not value['complete']:
        return dict(**value,retained=value['bank'],category='candidate_miss' if value['candidate_miss'] else 'failed_keep_all',
            false_reject=None if value['candidate_miss'] else False,all_models_fail_adequacy=False)
    R,A=_limit(threshold['R']),_limit(threshold['A'])
    retained=sorted(k for k,v in value['candidate_scores'].items() if v['R']<=R and v['A']<=A)
    all_bad=all(v['A']>A for v in value['candidate_scores'].values())
    if bool(not retained)!=all_bad:raise AssertionError('common-N empty-set identity failed')
    category=('candidate_miss' if value['candidate_miss'] else 'empty_all_models_fail_adequacy' if not retained else
        'unique_correct' if retained==[true] else 'unique_wrong' if len(retained)==1 else 'unresolved')
    return dict(**value,retained=retained,category=category,
        false_reject=None if value['candidate_miss'] else true not in retained,all_models_fail_adequacy=all_bad,
        minimum_A=min(v['A'] for v in value['candidate_scores'].values()))


def _summary(rows,thresholds):
    result={}
    for name in base.METHODS:
        result[name]={}
        for inference in base.INFERENCE:
            values=[decision(r,name,inference,thresholds['methods'][name][inference]) for r in rows]
            counts=Counter(v['category'] for v in values)
            result[name][inference]=dict(denominator=len(rows),
                **{k:counts[k] for k in ('unique_correct','unique_wrong','unresolved',
                    'empty_all_models_fail_adequacy','failed_keep_all','candidate_miss')},
                false_reject=sum(v['false_reject'] is True for v in values),
                false_reject_rate=sum(v['false_reject'] is True for v in values)/len(rows),
                all_models_fail_adequacy=sum(v['all_models_fail_adequacy'] for v in values),
                complete=sum(v['complete'] for v in values),
                metadata_failures=sum(v['metadata_failure'] is not None for v in values),
                minimum_A=base._stats(v.get('minimum_A') for v in values),
                A_true=base._stats(v['A_true'] for v in values if v['complete'] and not v['candidate_miss']),
                retained_count=base._stats(len(v['retained']) for v in values))
    return result


def analyze(phase,thresholds,output,*,formal=False,expected_count=None,calibration_records=None):
    rows=phase['records'];path=Path(output)/'adequacy_thresholds.json'
    if not path.exists() or base._read(path)!=thresholds:
        raise ValueError('matching auxiliary thresholds must be durably saved first')
    base._identity_match(thresholds['identity'],phase['identity'])
    if formal:
        if phase['phase']!='evaluation':raise ValueError('formal evaluation phase required')
        if expected_count is not None and len(rows)!=expected_count:raise ValueError('incomplete declared evaluation batch')
        if len({r['seed'] for r in rows})!=len(rows):raise ValueError('independent unique evaluation history seeds required')
        if calibration_records is not None and {r['seed'] for r in rows}&{r['seed'] for r in calibration_records}:
            raise ValueError('calibration/evaluation histories overlap')
        if any(scores(r,n,m)['candidate_miss'] for r in rows for n in base.METHODS for m in base.INFERENCE):
            raise ValueError('formal evaluation prohibits candidate miss')
    groups=defaultdict(list)
    for row in rows:groups[row['case_id']].append(row)
    declared=thresholds.get('declared_calibration_cases')
    outside=sum(r['case_id'] not in declared for r in rows) if declared is not None else None
    result=dict(schema='ev_systematic_auxiliary_adequacy_results_v1',phase=phase['phase'],
        identity=phase['identity'],source_files=phase['files'],additional_diagnostic=True,primary_thresholds_modified=False,
        threshold_file_sha256=sha256(path.read_bytes()).hexdigest(),overall=_summary(rows,thresholds),
        by_case={k:_summary(v,thresholds) for k,v in sorted(groups.items())},
        outside_declared_calibration_case_support=outside,
        by_history=[dict(case_id=r['case_id'],seed=r['seed'],true_name=r['true_name'],
            outside_declared_calibration_case_support=(r['case_id'] not in declared) if declared is not None else None,
            methods={name:{mode:decision(r,name,mode,thresholds['methods'][name][mode]) for mode in base.INFERENCE}
                for name in base.METHODS}) for r in rows],
        scope='additional frozen-rule comparison; outside-mixture stress flags are empirical; empty sets count as false rejection when truth is in bank')
    base._write(Path(output)/('adequacy_evaluation.json' if formal else 'adequacy_development.json'),result)
    return result


def _markdown(result):
    lines=['# 附加绝对误差门诊断','',
        '原primary规则保持不变。附加规则分别以0.05校准相对SSE差和每观测SSE；空集表示全部候选未通过绝对误差门。',
        '空集保留在误拒分母；计算不完整保留全部。压力场景的触发率是经验结果。','',
        '|方法|推断|总数|误拒|唯一正确|唯一错误|未决|空集|失败保留全部|漏真|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in base.METHODS:
        for mode in base.INFERENCE:
            row=result['overall'][name][mode]
            lines.append('|'+ '|'.join((name,mode,*(str(row[k]) for k in ('denominator','false_reject',
                'unique_correct','unique_wrong','unresolved','empty_all_models_fail_adequacy','failed_keep_all','candidate_miss'))))+'|')
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration',type=Path);parser.add_argument('--evaluation',type=Path)
    parser.add_argument('--development',type=Path);parser.add_argument('--thresholds',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.development is not None:
        if args.thresholds is None or args.calibration is not None or args.evaluation is not None:
            parser.error('development needs existing --thresholds and no formal directories')
        thresholds=base._read(args.thresholds)
        base._write(args.output/'adequacy_thresholds.json',thresholds,immutable=True)
        phase=base.read_phase(args.development,expected_phase='development')
        result=analyze(phase,thresholds,args.output)
    else:
        if args.calibration is None or args.evaluation is None or args.thresholds is not None:
            parser.error('formal mode needs --calibration and --evaluation')
        calibration=base.read_phase(args.calibration,expected_phase='calibration')
        thresholds=calibrate(calibration,args.output,expected_count=108)
        # Evaluation is first opened after immutable auxiliary threshold save.
        phase=base.read_phase(args.evaluation,expected_phase='evaluation')
        result=analyze(phase,thresholds,args.output,formal=True,expected_count=72,
            calibration_records=calibration['records'])
    (args.output/'adequacy_summary.md').write_text(_markdown(result),encoding='utf-8')


if __name__=='__main__':main()
