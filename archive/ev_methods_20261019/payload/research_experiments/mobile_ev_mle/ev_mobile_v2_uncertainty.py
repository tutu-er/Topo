"""Wilson intervals and predeclared paired counts for saved EV decisions.

No fit, action, threshold or rule is changed. One physical bank is processed at
a time. Evaluation histories, including failed-keep-all cases, stay in N.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import math
from pathlib import Path

from . import ev_reliability_v2 as rel
from . import ev_systematic_analyze as io

Z95=1.959963984540054
RULE_REFERENCE='H_R'
RULE_CHALLENGER='label_AIC'
FAMILIES=('square','scalar','cap','jointgrad')
MODES=('fixed_total','fixed_holdout','joint_total')
EVENTS=('unique_correct','false_reject')


def wilson(count,n):
    if not isinstance(count,int) or isinstance(count,bool) or not isinstance(n,int) or isinstance(n,bool) or n<=0 or not 0<=count<=n:
        raise ValueError('integer 0<=count<=N with N>0 required')
    p=count/n;denominator=1+Z95**2/n
    center=(p+Z95**2/(2*n))/denominator
    half=Z95*math.sqrt(p*(1-p)/n+Z95**2/(4*n*n))/denominator
    return dict(count=count,N=n,estimate=p,lower95=0. if count==0 else max(0.,center-half),
                upper95=1. if count==n else min(1.,center+half),
                method='Wilson score; nominal 95 percent')


def paired(reference,challenger,*,event):
    if event not in EVENTS or len(reference)!=len(challenger) or not reference:
        raise ValueError('nonempty matched event arrays required')
    if any(not isinstance(v,bool) for v in [*reference,*challenger]):raise ValueError('Boolean events required')
    table=Counter((a,b) for a,b in zip(reference,challenger));higher_is_better=event=='unique_correct'
    improve=table[False,True] if higher_is_better else table[True,False]
    regress=table[True,False] if higher_is_better else table[False,True]
    return dict(N=len(reference),improvement=improve,regression=regress,same=len(reference)-improve-regress,
        both_event_false=table[False,False],reference_false_challenger_true=table[False,True],
        reference_true_challenger_false=table[True,False],both_event_true=table[True,True],
        net_improvement_percentage_points=100*(improve-regress)/len(reference),
        favorable_event='true' if higher_is_better else 'false')


def _event(decision,event):
    return decision['category']=='unique_correct' if event=='unique_correct' else decision['false_reject']


def _validated(report,phase,bank,expected_count):
    if bank not in ('n3','n4'):raise ValueError('bank must be n3 or n4')
    if report.get('schema')!='ev_reliability_v2_evaluation_v1' or report.get('phase')!='evaluation' or report.get('thresholds_development_only') is not False:
        raise ValueError('formal saved reliability evaluation required')
    if phase.get('phase')!='evaluation':raise ValueError('evaluation record projection required')
    io._identity_match(report['identity'],phase['identity']);methods=rel.declared_methods(phase)
    records=phase['records'];rows=report.get('by_history',[])
    if len(rows)!=expected_count or len(records)!=expected_count:raise ValueError('complete expected evaluation count required')
    if len({r['seed'] for r in rows})!=expected_count or len({r['seed'] for r in records})!=expected_count:
        raise ValueError('independent evaluation histories require unique seeds')
    lookup={(r['case_id'],r['seed']):r for r in records}
    if {(r['case_id'],r['seed']) for r in rows}!=set(lookup):raise ValueError('report and record history identities differ')
    for row in rows:
        record=lookup[row['case_id'],row['seed']]
        if record.get('candidate_miss') or record['true_name']!=row['true_name']:raise ValueError('truth label or candidate coverage differs')
        parameters,error=rel.candidate_parameters(record)
        if error or any(p['n']!=int(bank[1:]) for p in parameters.values()):raise ValueError('record physical bank differs')
        if set(row.get('rules',{}))!=set(rel.RULES):raise ValueError('all five declared rules required')
        candidate_bank=set(io._bank(record))
        for rule in rel.RULES:
            if set(row['rules'][rule])!=set(methods):raise ValueError('all declared methods required')
            for method in methods:
                if set(row['rules'][rule][method])!=set(MODES):raise ValueError('all three inference modes required')
                for decision in row['rules'][rule][method].values():
                    category=decision['category'];kept=decision['retained'];true=row['true_name']
                    if not isinstance(kept,list) or any(not isinstance(k,str) for k in kept) or len(set(kept))!=len(kept) or set(kept)-candidate_bank:
                        raise ValueError('retained candidates must be a unique in-bank list')
                    if not isinstance(decision.get('false_reject'),bool):
                        raise ValueError('explicit Boolean rejection required')
                    if decision['false_reject']!=(true not in kept):raise ValueError('rejection event disagrees with retained set')
                    if category=='failed_keep_all':
                        if set(kept)!=candidate_bank:raise ValueError('failure must retain every candidate')
                    elif category!=('unique_correct' if kept==[true] else 'unique_wrong' if len(kept)==1 else 'unresolved'):
                        raise ValueError('category disagrees with retained set')
    # Recount all supplied published cells, so a wrong denominator is visible.
    for rule in rel.RULES:
        for method in methods:
            for mode in MODES:
                decisions=[r['rules'][rule][method][mode] for r in rows]
                stored=report['overall'][rule][method][mode]
                if stored['denominator']!=expected_count or stored['unique_correct']!=sum(_event(d,'unique_correct') for d in decisions) or stored['false_reject']!=sum(_event(d,'false_reject') for d in decisions):
                    raise ValueError('saved overall count differs from original histories')
    return methods,rows


def summarize(report,phase,*,bank,expected_count=48):
    methods,rows=_validated(report,phase,bank,expected_count);n=len(rows)
    intervals={};main_pairs={};mobility={}
    def events(rule,method,mode,event):return [_event(r['rules'][rule][method][mode],event) for r in rows]
    for method in methods:
        intervals[method]={};main_pairs[method]={}
        for mode in MODES:
            intervals[method][mode]={}
            for rule in rel.RULES:
                decisions=[r['rules'][rule][method][mode] for r in rows];categories=Counter(d['category'] for d in decisions)
                intervals[method][mode][rule]=dict(
                    **{event:wilson(sum(_event(d,event) for d in decisions),n) for event in EVENTS},
                    unresolved=categories['unresolved'],failed_keep_all=categories['failed_keep_all'],denominator=n)
            main_pairs[method][mode]={event:paired(events(RULE_REFERENCE,method,mode,event),
                events(RULE_CHALLENGER,method,mode,event),event=event) for event in EVENTS}
    for family in FAMILIES:
        fixed,mobile=family+'_fixed',family+'_mobile'
        if fixed not in methods or mobile not in methods:continue
        mobility[family]={mode:{rule:{event:paired(events(rule,fixed,mode,event),events(rule,mobile,mode,event),event=event)
            for event in EVENTS} for rule in rel.RULES} for mode in MODES}
    examples=sorted({0,1,min(3,n),n//2,n-1,n})
    return dict(schema='ev_mobile_v2_uncertainty_v1',physical_bank=bank,N=n,methods=list(methods),modes=list(MODES),
        rules=list(rel.RULES),intervals=intervals,
        main_rule_comparison=dict(reference=RULE_REFERENCE,challenger=RULE_CHALLENGER,paired=main_pairs),
        mobility_comparison=dict(reference='fixed',challenger='mobile',paired=mobility),
        resolution=dict(one_history_percentage_points=100/n,examples=[wilson(x,n) for x in examples],
            statement='One changed history shifts a rate by 100/N percentage points; intervals, not just rounded counts, describe finite-sample resolution.'),
        seed_count=len({r['seed'] for r in rows}),source_identity=phase['identity'],source_files=phase['files'],
        saved_threshold_sha256=report.get('threshold_sha256'),
        interval_scope='Descriptive Wilson score intervals at nominal 95 percent, separately for each event rate of the frozen calibrated procedure under the declared IID evaluation-history distribution.',
        simultaneous_coverage_claimed=False,significance_tests_performed=False,best_rule_selected=False,
        upstream_dependency='Frozen index/seed schedule and source integrity are checked by merge/controller; this tool checks bank, unique identities and event counts.',
        failure_rule='Failed-keep-all histories remain in N; their unique-correct and rejection indicators are both false.',
        interpretation='Rates refer to repeated histories of the frozen procedure, not correctness probabilities for a particular topology.')


def markdown(value):
    def percent(x):return f'{100*x:.1f}%'
    lines=[f'# {value["physical_bank"]}：独立评价的不确定性','',
        f'分母{value["N"]}条独立历史。每改变1条，比例改变{value["resolution"]["one_history_percentage_points"]:.2f}个百分点。',
        'Wilson区间为逐项名义95%；描述已冻结阈值流程在声明模拟分布中的事件率。各比较分别报告，不提供同时覆盖或显著性结论。','',
        '|事件数|样本比例|95% Wilson区间|','|---:|---:|---:|']
    for row in value['resolution']['examples']:
        lines.append(f'|{row["count"]}/{row["N"]}|{percent(row["estimate"])}|{percent(row["lower95"])}–{percent(row["upper95"])}|')
    lines.extend(['','下面显示主比较的joint_total；JSON保留全部11方法、3口径、5规则和同规则移动对照。',
        '“改善/退步/同”按同一历史配对；误拒以减少为改善，唯一正确以增加为改善。未决与失败均保留在原分母。','',
        '|方法|label_AIC唯一正确及区间|label_AIC误拒及区间|相对H_R：正确改善/退步/同|相对H_R：误拒改善/退步/同|',
        '|---|---|---|---|---|'])
    for method in value['methods']:
        cell=value['intervals'][method]['joint_total']['label_AIC'];pair=value['main_rule_comparison']['paired'][method]['joint_total']
        def interval(event):
            row=cell[event]
            return f'{row["count"]}/{row["N"]} [{percent(row["lower95"])}, {percent(row["upper95"])}]'
        def counts(event):
            row=pair[event];return f'{row["improvement"]}/{row["regression"]}/{row["same"]}'
        lines.append('|'+ '|'.join((method,interval('unique_correct'),interval('false_reject'),counts('unique_correct'),counts('false_reject')))+'|')
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evaluation-report',type=Path,required=True)
    parser.add_argument('--records',type=Path,required=True);parser.add_argument('--bank',choices=('n3','n4'),required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=io._read(args.evaluation_report);phase=rel.read_phase(args.records,expected_phase='evaluation')
    value=summarize(report,phase,bank=args.bank)
    value['evaluation_report']=dict(path=str(args.evaluation_report.resolve()),sha256=sha256(args.evaluation_report.read_bytes()).hexdigest())
    io._write(args.output.with_suffix('.json'),value);args.output.with_suffix('.md').write_text(markdown(value),encoding='utf-8')


if __name__=='__main__':main()
