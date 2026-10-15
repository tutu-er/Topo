"""Predeclared H=8/32/96 conditional calibration, separate from the primary rule.

H is the known training-point count, fixed before calibration/evaluation is
read.  For each policy/inference and H, calibrate R=S_true-min S at alpha=.1
using rank ceil((n_H+1)*.9).  Insufficient rank or an absent group gives infinity;
an absent/unregistered group keeps all.  Numerical/operation failures keep all
and contribute score zero, exactly as the original global rule.

Conditional on H and its calibration count n_H, IID calibration/test histories
within that group make the n_H+1 scores exchangeable.  Their order statistic
gives P(R_test>tau_H | H,n_H)<=.1; ties are conservative.  The group H=96 still
contains a mixture of regimes and truth structures.  No finer conditional
coverage is claimed, and no additional groups are selected from results.

The original global alpha=.1 and the separate adequacy dual gate are unchanged.
This module uses only standard-library analysis, with no new AC, fit or action.
Auxiliary thresholds and the same-cohort global reference are saved BEFORE the
evaluation directory is first opened.  The optional primary file is only read
to verify numerical agreement; it is never written.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from . import ev_systematic_analyze as base


GROUPS=('8','32','96')
ALPHA=.1


def training_group(record):
    value=record.get('config',{}).get('training_points')
    return str(value) if isinstance(value,int) and not isinstance(value,bool) and str(value) in GROUPS else None


def _cutoff(values):
    if values:return base.conformal_threshold(values,ALPHA)
    return dict(n=0,alpha=ALPHA,order_rank=1,threshold=None,infinite=True)


def _check_primary(primary,identity,files,reference):
    base._identity_match(identity,primary['identity'])
    if primary['alpha']!=ALPHA:raise ValueError('primary alpha differs from predeclared .1')
    if sorted(row['sha256'] for row in files)!=sorted(row['sha256'] for row in primary['calibration_files']):
        raise ValueError('primary calibration cohort differs')
    for name in base.METHODS:
        for mode in base.INFERENCE:
            for key in ('n','alpha','order_rank','threshold','infinite'):
                if primary['methods'][name][mode][key]!=reference[name][mode][key]:
                    raise ValueError('same-cohort global threshold differs from primary')


def calibrate(phase,output,*,expected_count=None,primary_thresholds=None):
    rows=phase['records'];identity=phase['identity']
    if phase['phase']!='calibration':raise ValueError('calibration phase required')
    if expected_count is not None and len(rows)!=expected_count:raise ValueError('incomplete declared calibration batch')
    if not rows or len({r['seed'] for r in rows})!=len(rows):raise ValueError('independent unique calibration history seeds required')
    if any(training_group(r) is None for r in rows):raise ValueError('calibration H outside predeclared 8/32/96 groups')
    if not any(k in identity for k in ('source_id','source_sha256_digest')) or not any(k in identity for k in ('config_id','policy_id','manifest_config_digest')):
        raise ValueError('frozen source/configuration identities required')
    groups={h:[r for r in rows if training_group(r)==h] for h in GROUPS}
    methods={};global_reference={}
    for name in base.METHODS:
        methods[name]={};global_reference[name]={}
        for mode in base.INFERENCE:
            all_values=[base._mode(r,name,mode) for r in rows]
            if any(v['candidate_miss'] for v in all_values):raise ValueError('formal calibration prohibits candidate miss')
            global_reference[name][mode]=dict(**_cutoff([v['nonconformity'] for v in all_values]),
                incomplete_calibration=sum(not v['complete'] for v in all_values))
            methods[name][mode]={}
            for h in GROUPS:
                values=[base._mode(r,name,mode) for r in groups[h]]
                scores=[v['nonconformity'] for v in values]
                methods[name][mode][h]=dict(**_cutoff(scores),scores=scores,
                    incomplete_calibration=sum(not v['complete'] for v in values),missing_group=not values)
    if primary_thresholds is not None:_check_primary(primary_thresholds,identity,phase['files'],global_reference)
    result=dict(schema='ev_systematic_predeclared_H_stratification_v1',alpha=ALPHA,groups=list(GROUPS),
        grouping_variable='config.training_points; known before action and outcome',identity=identity,
        calibration_count=len(rows),calibration_files=phase['files'],methods=methods,global_reference=global_reference,
        group_counts={h:len(groups[h]) for h in GROUPS},
        group_truth_counts={h:dict(Counter(str(r['true_name']) for r in groups[h])) for h in GROUPS},
        primary_thresholds_checked=primary_thresholds is not None,primary_thresholds_modified=False,
        adequacy_rule_combined=False,additional_comparison=True,thresholds_saved_before_evaluation_read=True,
        failure_rule='incomplete: keep all, calibration score zero; missing group: keep all',
        source_sha256=dict(stratified=sha256(Path(__file__).read_bytes()).hexdigest(),
            analyzer=sha256(Path(base.__file__).read_bytes()).hexdigest()),
        guarantee_scope='conditional within predeclared H under conditional exchangeability; H=96 remains a regime/truth mixture')
    base._write(Path(output)/'stratified_thresholds.json',result,immutable=True)
    return result


def decision(record,name,mode,thresholds,*,global_rule=False):
    h=training_group(record);missing=h is None
    if global_rule:
        cutoff=thresholds['global_reference'][name][mode]
    else:
        cutoff=thresholds['methods'][name][mode].get(h)
        missing=missing or cutoff is None or cutoff['n']==0
        if cutoff is None:cutoff=_cutoff([])
    result=base.decision(record,name,mode,cutoff)
    if not global_rule and missing and result['complete'] and not result['candidate_miss']:
        result.update(category='missing_group_keep_all',retained=result['bank'],false_reject=False)
    return dict(**result,H_group=h,missing_group=missing if not global_rule else False,
        calibration_group_n=cutoff['n'],infinite_threshold=cutoff['infinite'],rule='global' if global_rule else 'H_stratified')


def _summary(rows,thresholds,*,global_rule=False):
    result={}
    for name in base.METHODS:
        result[name]={}
        for mode in base.INFERENCE:
            values=[decision(r,name,mode,thresholds,global_rule=global_rule) for r in rows]
            counts=Counter(v['category'] for v in values)
            result[name][mode]=dict(denominator=len(rows),
                **{k:counts[k] for k in ('unique_correct','unique_wrong','unresolved','failed_keep_all',
                    'missing_group_keep_all','candidate_miss')},
                false_reject=sum(v['false_reject'] is True for v in values),
                false_reject_rate=sum(v['false_reject'] is True for v in values)/len(rows) if rows else None,
                missing_group=sum(v['missing_group'] for v in values),
                infinite_threshold=sum(v['infinite_threshold'] for v in values),
                retained_count=base._stats(len(v['retained']) for v in values),
                planning_seconds=base._stats(base._methods(r)[name].get('planning_seconds') for r in rows),
                inference_seconds=base._stats(v['seconds'] for v in values),
                total_seconds=base._stats(v['total_seconds'] for v in values))
    return result


def _paired_delta(stratified,global_rule):
    return {name:{mode:dict(denominator=stratified[name][mode]['denominator'],
        delta_unique_correct=stratified[name][mode]['unique_correct']-global_rule[name][mode]['unique_correct'],
        delta_false_reject=stratified[name][mode]['false_reject']-global_rule[name][mode]['false_reject'],
        inference_and_planning_cost_unchanged=True) for mode in base.INFERENCE} for name in base.METHODS}


def analyze(phase,thresholds,output,*,expected_count=None,calibration_records=None):
    began=perf_counter();rows=phase['records'];path=Path(output)/'stratified_thresholds.json'
    if phase['phase']!='evaluation':raise ValueError('evaluation phase required')
    if expected_count is not None and len(rows)!=expected_count:raise ValueError('incomplete declared evaluation batch')
    if not path.exists() or base._read(path)!=thresholds:raise ValueError('matching thresholds must be durably saved first')
    base._identity_match(thresholds['identity'],phase['identity'])
    if len({r['seed'] for r in rows})!=len(rows):raise ValueError('independent unique evaluation history seeds required')
    if calibration_records is not None and {r['seed'] for r in rows}&{r['seed'] for r in calibration_records}:
        raise ValueError('calibration/evaluation histories overlap')
    if any(base._mode(r,name,mode)['candidate_miss'] for r in rows for name in base.METHODS for mode in base.INFERENCE):
        raise ValueError('formal evaluation prohibits candidate miss')
    grouped={h:[r for r in rows if training_group(r)==h] for h in GROUPS}
    unsupported=[r for r in rows if training_group(r) is None]
    overall=_summary(rows,thresholds);global_rule=_summary(rows,thresholds,global_rule=True)
    by_group={}
    for h,group in grouped.items():
        local=_summary(group,thresholds);reference=_summary(group,thresholds,global_rule=True)
        by_group[h]=dict(histories=len(group),stratified=local,global_reference=reference,
            paired_count_delta=_paired_delta(local,reference))
    result=dict(schema='ev_systematic_H_stratified_results_v1',phase='evaluation',identity=phase['identity'],
        source_files=phase['files'],threshold_file_sha256=sha256(path.read_bytes()).hexdigest(),histories=len(rows),
        overall=overall,global_reference=global_rule,paired_count_delta=_paired_delta(overall,global_rule),by_group=by_group,
        unsupported_group_histories=len(unsupported),
        by_history=[dict(case_id=r['case_id'],seed=r['seed'],H_group=training_group(r),true_name=r['true_name'],
            methods={name:{mode:dict(stratified=decision(r,name,mode,thresholds),
                global_reference=decision(r,name,mode,thresholds,global_rule=True)) for mode in base.INFERENCE}
                for name in base.METHODS}) for r in rows],
        analysis_seconds=perf_counter()-began,primary_thresholds_modified=False,adequacy_rule_combined=False,
        scope='same evaluation histories and original denominators; groups fixed before calibration/evaluation; no new actions or fits')
    base._write(Path(output)/'stratified_evaluation.json',result)
    return result


def _markdown(result):
    lines=['# 按已知训练长度分层的附加校准对照','',
        '事前固定H=8/32/96；每组使用自身样本数的0.1阶统计阈值。缺组或样本不足取无穷阈值。',
        '原global primary及绝对误差双门保持独立；规划和拟合数据相同，计算成本不因这个分组重做。','',
        '|H|方法|推断|同组总数|global唯一正确|分层唯一正确|global误拒|分层误拒|分层失败保留全部|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    tables={'all':dict(histories=result['histories'],stratified=result['overall'],global_reference=result['global_reference']),**result['by_group']}
    for h,table in tables.items():
        for name in base.METHODS:
            for mode in base.INFERENCE:
                g=table['global_reference'][name][mode];s=table['stratified'][name][mode]
                lines.append('|'+ '|'.join((h,name,mode,str(s['denominator']),str(g['unique_correct']),str(s['unique_correct']),
                    str(g['false_reject']),str(s['false_reject']),str(s['failed_keep_all'])))+'|')
    lines.extend(['','条件覆盖依赖组内校准/评价可交换。H=96内仍混合多个场景；未声称每种场景或真结构单独覆盖。'])
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration',type=Path,required=True);parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--primary-thresholds',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();calibration=base.read_phase(args.calibration,expected_phase='calibration')
    primary=base._read(args.primary_thresholds) if args.primary_thresholds is not None else None
    thresholds=calibrate(calibration,args.output,expected_count=108,primary_thresholds=primary)
    # The first evaluation read follows the durable, immutable threshold write.
    evaluation=base.read_phase(args.evaluation,expected_phase='evaluation')
    result=analyze(evaluation,thresholds,args.output,expected_count=72,calibration_records=calibration['records'])
    (args.output/'stratified_summary.md').write_text(_markdown(result),encoding='utf-8')


if __name__=='__main__':main()
