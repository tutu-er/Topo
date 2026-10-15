"""Standard-library-only analysis and separate calibration of ten EV policies.

The formal command saves immutable calibration thresholds BEFORE opening any
evaluation record.  Incomplete inference/illegal operation keeps every candidate
and has calibration nonconformity zero.  Decision counts retain all histories.
This module imports no numerical fitter, selector, simulator or AC implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from decimal import Decimal,ROUND_CEILING
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics


METHODS=('none','constant','highlow','square_fixed','square_mobile','nearest_fixed',
         'gradac_fixed','scalar_fixed','scalar_hist_fixed','scalar_mobile')
INFERENCE=('fixed_total','fixed_holdout','joint_total')
TIE_ABS=1e-8
TIE_REL=1e-10
PHASE_ALIASES={'dev':'development','cal':'calibration','eval':'evaluation'}


def _read(path):
    path=Path(path)
    with (gzip.open(path,'rt',encoding='utf-8') if path.suffix=='.gz' else path.open(encoding='utf-8')) as stream:
        return json.load(stream)


def _encoded(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'


def _write(path,value,*,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);text=_encoded(value)
    if immutable and path.exists():
        if _read(path)!=value:raise ValueError('existing thresholds differ; retain them and use a fresh output directory')
        return
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(text,encoding='utf-8');temporary.replace(path)


def _finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _phase(value):
    return PHASE_ALIASES.get(value,value)


def _stable_identity(manifest,records):
    """Compare declared global identities, never each case's different physics."""
    identity={}
    for key in ('source_id','config_id'):
        found=[]
        for row in (manifest,*records):
            for container in (row,row.get('checks',{})):
                value=container.get(key)
                if value is not None:found.append(str(value))
        if len(set(found))>1:raise ValueError('inconsistent '+key+' within phase')
        if found:identity[key]=found[0]
    if 'config_id' not in identity:
        found=[str(container['policy_id']) for row in (manifest,*records)
               for container in (row,row.get('checks',{})) if container.get('policy_id') is not None]
        if len(set(found))>1:raise ValueError('inconsistent policy_id within phase')
        if found:identity['policy_id']=found[0]
    hashes=[]
    for row in (manifest,*records):
        value=row.get('source_sha256')
        if value is not None:hashes.append(sha256(_encoded(value).encode()).hexdigest())
    if len(set(hashes))>1:raise ValueError('inconsistent source_sha256 within phase')
    if hashes:identity['source_sha256_digest']=hashes[0]
    # A manifest-level config is the whole experiment, not a case-specific model.
    if manifest.get('config') is not None and not any(k in identity for k in ('config_id','policy_id')):
        identity['manifest_config_digest']=sha256(_encoded(manifest['config']).encode()).hexdigest()
    return identity


def _validate_record(record,expected_phase):
    required=('case_id','seed','phase','true_name','candidate_miss','status','methods')
    if any(key not in record for key in required):raise ValueError('case record missing required schema fields')
    if expected_phase is not None and _phase(record['phase'])!=expected_phase:
        raise ValueError('record phase differs from requested phase')
    if record.get('source_unchanged') is False:raise ValueError('record reports source drift during execution')
    names=[row.get('name') for row in record['methods']]
    if len(names)!=len(METHODS) or set(names)!=set(METHODS):
        raise ValueError('every case must retain exactly the ten declared methods')
    for method in record['methods']:
        if set(method.get('inference',{}))!=set(INFERENCE):
            raise ValueError('every method must retain all three inference modes')
        for mode in method['inference'].values():
            if not isinstance(mode.get('losses'),dict) or not isinstance(mode.get('complete'),bool):
                raise ValueError('inference requires losses mapping and explicit complete bool')
            if mode.get('M') is not None and not _finite(mode['M']):raise ValueError('M must be finite or None')


def read_phase(directory,*,expected_phase=None):
    """Read explicit case/trial records; preserve failed cases and input hashes."""
    directory=Path(directory);expected_phase=_phase(expected_phase)
    metadata={}
    for name in ('freeze.json','manifest.json'):
        path=directory/name
        if path.exists():
            row=_read(path)
            for key,value in row.items():
                if key in metadata and metadata[key]!=value and key in ('source_id','config_id','policy_id','source_sha256'):
                    raise ValueError('freeze/manifest identity mismatch')
                metadata[key]=value
    paths=sorted({p for pattern in ('case_*.json','case_*.json.gz','trial_*.json','trial_*.json.gz')
                  for p in directory.glob(pattern)})
    records=[];files=[];seen=set()
    for path in paths:
        record=_read(path)
        if not isinstance(record,dict) or 'methods' not in record:continue
        _validate_record(record,expected_phase)
        key=(record['case_id'],record['seed'])
        if key in seen:raise ValueError('duplicate case/history record: '+str(key))
        seen.add(key);records.append(record)
        files.append(dict(path=str(path.resolve()),sha256=sha256(path.read_bytes()).hexdigest()))
    if not records:raise ValueError('no case/trial records found')
    phases={_phase(row['phase']) for row in records}
    if len(phases)!=1:raise ValueError('mixed phases in one directory')
    return dict(directory=str(directory.resolve()),phase=next(iter(phases)),manifest=metadata,
        identity=_stable_identity(metadata,records),records=records,files=files)


def _methods(record):
    return {method['name']:method for method in record['methods']}


def _bank(record):
    value=record.get('candidate_names',record.get('candidates',()))
    names=[row if isinstance(row,str) else row.get('name',row.get('candidate')) for row in value]
    return sorted(set(name for name in names if name is not None) or
                  set(name for method in record['methods'] for mode in method['inference'].values()
                      for name in mode['losses']))


def _mode(record,method_name,inference):
    method=_methods(record)[method_name];raw=method['inference'][inference]
    bank=_bank(record);losses=raw['losses'];true=record['true_name']
    operation_ok=method.get('operation',{}).get('success',True)
    complete=bool(raw['complete'] and operation_ok and bank and set(losses)==set(bank)
                  and all(_finite(v) and v>=0 for v in losses.values()))
    miss=bool(record['candidate_miss'] or true is None or (bank and true not in bank))
    margin=None;rank=None;rank_upper=None;minimum=[];nonconformity=0.
    if complete:
        best=min(losses.values());minimum=sorted(name for name,value in losses.items() if value==best)
        if not miss:
            nonconformity=max(0.,losses[true]-best)
            rank=1+sum(value<losses[true] for value in losses.values())
            rank_upper=sum(value<=losses[true] for value in losses.values())
            if len(bank)>1:margin=min(value for name,value in losses.items() if name!=true)-losses[true]
            if raw.get('M') is not None and margin is not None and abs(raw['M']-margin)>1e-7*(1+abs(margin)):
                raise ValueError('reported M differs from same-loss SSE margin')
    return dict(complete=complete,candidate_miss=miss,losses=losses,bank=bank,M=margin,
        nonconformity=nonconformity,true_rank=rank,true_rank_upper=rank_upper,
        minimum_candidates=minimum,seconds=raw.get('seconds'),total_seconds=raw.get('total_seconds'),raw_complete=raw['complete'],
        operation_success=operation_ok)


def _stats(values):
    values=[float(value) for value in values if _finite(value)]
    return dict(n=len(values),mean=statistics.fmean(values) if values else None,
        median=statistics.median(values) if values else None,min=min(values) if values else None,
        max=max(values) if values else None,sum=sum(values))


def _ranks(values):
    answer=[0.]*len(values);order=sorted(range(len(values)),key=lambda i:values[i]);start=0
    while start<len(order):
        end=start+1
        while end<len(order) and values[order[end]]==values[order[start]]:end+=1
        rank=(start+1+end)/2
        for index in order[start:end]:answer[index]=rank
        start=end
    return answer


def spearman(left,right):
    """Average ranks for ties; None if fewer than two or zero rank variance."""
    if len(left)!=len(right):raise ValueError('paired rank arrays required')
    if len(left)<2:return None
    x,y=_ranks(left),_ranks(right);mx,my=statistics.fmean(x),statistics.fmean(y)
    xx=sum((value-mx)**2 for value in x);yy=sum((value-my)**2 for value in y)
    return None if xx==0 or yy==0 else sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(xx*yy)


def _baseline(method):
    return 'square_mobile' if method.endswith('_mobile') else 'square_fixed'


def _wtl(records,method,inference):
    counts=dict(win=0,tie=0,loss=0,incomplete=0,denominator=len(records));deltas=[]
    for record in records:
        a=_mode(record,method,inference)['M'];b=_mode(record,_baseline(method),inference)['M']
        if a is None or b is None:counts['incomplete']+=1;continue
        delta=a-b;deltas.append(delta);tol=TIE_ABS+TIE_REL*max(abs(a),abs(b))
        counts['win' if delta>tol else 'loss' if delta< -tol else 'tie']+=1
    return dict(**counts,baseline=_baseline(method),delta_M=_stats(deltas),
        valid_pairs=len(deltas),tie_atol=TIE_ABS,tie_rtol=TIE_REL)


def _comparison(record):
    result={}
    for a,b in (('fixed_total','joint_total'),('fixed_holdout','joint_total')):
        rows=[];unique=[];seen=set();candidate_correlations=[]
        for method in record['methods']:
            left=_mode(record,method['name'],a);right=_mode(record,method['name'],b)
            if left['M'] is not None and right['M'] is not None:
                rows.append((left['M'],right['M']))
                key=method.get('action_key',method['name'])
                if not isinstance(key,(str,int,float,bool,type(None))):key=_encoded(key)
                if key not in seen:unique.append(rows[-1]);seen.add(key)
            if left['complete'] and right['complete']:
                names=sorted(set(left['losses'])&set(right['losses']))
                candidate_correlations.append(spearman([left['losses'][k] for k in names],[right['losses'][k] for k in names]))
        result[a+'_vs_'+b]=dict(method_M_spearman=spearman(*zip(*rows)) if rows else None,
            unique_action_M_spearman=spearman(*zip(*unique)) if unique else None,
            method_pairs=len(rows),unique_actions=len(unique),candidate_loss_rank_spearman=_stats(candidate_correlations))
    return result


def summarize(records):
    """Uncalibrated descriptive results, with every case in each denominator."""
    result=dict(histories=len(records),case_counts=dict(Counter(r['case_id'] for r in records)),
        truth_counts=dict(Counter(str(r['true_name']) for r in records)),
        status_counts=dict(Counter(r['status'] for r in records)),
        candidate_miss=sum(bool(r['candidate_miss']) for r in records),
        unique_seed_count=len({r['seed'] for r in records}),
        unique_seed_truth_count=len({(r['seed'],r['true_name']) for r in records}),methods={})
    for name in METHODS:
        methods=[_methods(r)[name] for r in records]
        row=dict(denominator=len(records),planning_failure=sum(bool(m.get('planning_failure')) for m in methods),
            operation_failure=sum(not m.get('operation',{}).get('success',True) for m in methods),
            planning_seconds=_stats(m.get('planning_seconds') for m in methods),
            total_seconds=_stats(m.get('total_seconds') for m in methods),
            parameter_error={key:_stats(m.get('parameter_error',{}).get(key) for m in methods)
                             for key in ('training_R','training_X','joint_R','joint_X')},inference={})
        for inference in INFERENCE:
            values=[_mode(r,name,inference) for r in records]
            row['inference'][inference]=dict(denominator=len(records),complete=sum(v['complete'] for v in values),
                incomplete=sum(not v['complete'] for v in values),candidate_miss=sum(v['candidate_miss'] for v in values),
                M=_stats(v['M'] for v in values),M_missing=sum(v['M'] is None for v in values),
                seconds=_stats(v['seconds'] for v in values),
                total_seconds=_stats(v['total_seconds'] for v in values),
                strict_unique_top_correct=sum(v['minimum_candidates']==[r['true_name']] for r,v in zip(records,values)),
                strict_unique_top_wrong=sum(v['complete'] and not v['candidate_miss'] and len(v['minimum_candidates'])==1
                    and v['minimum_candidates']!=[r['true_name']] for r,v in zip(records,values)),
                minimum_tie=sum(v['complete'] and len(v['minimum_candidates'])>1 for v in values),
                true_rank=_stats(v['true_rank'] for v in values),vs_same_domain_square=_wtl(records,name,inference))
        result['methods'][name]=row
    return result


def descriptive_analysis(phase):
    records=phase['records'];groups=defaultdict(list)
    for record in records:groups[record['case_id']].append(record)
    return dict(phase=phase['phase'],identity=phase['identity'],source_files=phase['files'],
        overall=summarize(records),by_case={key:summarize(value) for key,value in sorted(groups.items())},
        by_history=[dict(case_id=r['case_id'],seed=r['seed'],true_name=r['true_name'],
            candidate_miss=r['candidate_miss'],methods={name:{inference:_mode(r,name,inference)
            for inference in INFERENCE} for name in METHODS},comparison=_comparison(r)) for r in records],
        scope='Descriptive only; paired development regimes sharing a seed are not independent histories.')


def conformal_threshold(scores,alpha=.1):
    if not scores or not 0<float(alpha)<1:raise ValueError('nonempty scores and 0<alpha<1 required')
    if not all(_finite(value) and value>=0 for value in scores):raise ValueError('finite nonnegative scores required')
    rank=int((Decimal(len(scores)+1)*(Decimal(1)-Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING))
    infinite=rank>len(scores)
    return dict(n=len(scores),alpha=float(alpha),order_rank=rank,threshold=None if infinite else sorted(scores)[rank-1],infinite=infinite)


def calibrate(phase,output,*,alpha=.1,expected_count=None):
    records=phase['records']
    if phase['phase']!='calibration':raise ValueError('calibration phase required')
    if expected_count is not None and len(records)!=expected_count:raise ValueError('calibration count differs from declared complete batch')
    if len({r['seed'] for r in records})!=len(records):raise ValueError('formal calibration requires independent unique history seeds')
    if any(_mode(r,name,mode)['candidate_miss'] for r in records for name in METHODS for mode in INFERENCE):
        raise ValueError('formal calibration prohibits candidate miss')
    identity=phase['identity']
    if not any(k in identity for k in ('source_id','source_sha256_digest')) or not any(k in identity for k in ('config_id','policy_id','manifest_config_digest')):
        raise ValueError('formal phase requires declared source and configuration identities')
    values={}
    for name in METHODS:
        values[name]={}
        for inference in INFERENCE:
            rows=[_mode(r,name,inference) for r in records];scores=[row['nonconformity'] for row in rows]
            values[name][inference]=dict(**conformal_threshold(scores,alpha),
                incomplete_calibration=sum(not row['complete'] for row in rows),scores=scores)
    result=dict(schema='ev_systematic_separate_thresholds_v1',identity=identity,
        analyzer_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        calibration_files=phase['files'],calibration_count=len(records),alpha=float(alpha),methods=values,
        failure_rule='incomplete inference or illegal operation: keep all candidates, calibration score=0',
        nonconformity='max(0,SSE_true-min_candidate_SSE)',
        thresholds_saved_before_evaluation_read=True,
        simultaneous_coverage_claimed=False,
        coverage_scope='marginal over the declared IID regime/truth mixture and frozen full procedure')
    _write(Path(output)/'thresholds.json',result,immutable=True)
    return result


def _identity_match(calibration,evaluation):
    common=set(calibration)&set(evaluation)
    if not any(k in common for k in ('source_id','source_sha256_digest')) or not any(k in common for k in ('config_id','policy_id','manifest_config_digest')):
        raise ValueError('calibration/evaluation lack common source and configuration identities')
    for key in common:
        if calibration[key]!=evaluation[key]:raise ValueError('calibration/evaluation '+key+' differs')


def decision(record,name,inference,threshold):
    value=_mode(record,name,inference);true=record['true_name'];bank=value['bank']
    if value['candidate_miss']:
        return dict(category='candidate_miss',retained=bank,false_reject=None,**value)
    if not value['complete']:
        return dict(category='failed_keep_all',retained=bank,false_reject=False,**value)
    best=min(value['losses'].values());limit=math.inf if threshold['infinite'] else threshold['threshold']
    retained=sorted(k for k,v in value['losses'].items() if max(0.,v-best)<=limit)
    category=('unique_correct' if retained==[true] else 'unique_wrong' if len(retained)==1 else 'unresolved')
    return dict(category=category,retained=retained,false_reject=true not in retained,**value)


def _decision_summary(records,thresholds):
    result={}
    for name in METHODS:
        result[name]={}
        for inference in INFERENCE:
            rows=[decision(r,name,inference,thresholds['methods'][name][inference]) for r in records]
            counts=Counter(row['category'] for row in rows)
            result[name][inference]=dict(denominator=len(records),
                **{k:counts[k] for k in ('unique_correct','unique_wrong','unresolved','failed_keep_all','candidate_miss')},
                false_reject=sum(row['false_reject'] is True for row in rows),
                false_reject_rate=sum(row['false_reject'] is True for row in rows)/len(records),
                retained_count=_stats(len(row['retained']) for row in rows),
                unresolved_including_failed=counts['unresolved']+counts['failed_keep_all'])
    return result


def evaluate(phase,thresholds,output,*,expected_count=None,calibration_records=None):
    records=phase['records']
    if phase['phase']!='evaluation':raise ValueError('evaluation phase required')
    if expected_count is not None and len(records)!=expected_count:raise ValueError('evaluation count differs from declared complete batch')
    _identity_match(thresholds['identity'],phase['identity'])
    if len({r['seed'] for r in records})!=len(records):raise ValueError('formal evaluation requires independent unique history seeds')
    if calibration_records is not None and {r['seed'] for r in records}&{r['seed'] for r in calibration_records}:
        raise ValueError('calibration and evaluation histories overlap')
    if any(_mode(r,name,mode)['candidate_miss'] for r in records for name in METHODS for mode in INFERENCE):
        raise ValueError('formal evaluation prohibits candidate miss')
    threshold_path=Path(output)/'thresholds.json'
    if not threshold_path.exists() or _read(threshold_path)!=thresholds:
        raise ValueError('matching thresholds must be durably saved before evaluation')
    groups=defaultdict(list)
    for record in records:groups[record['case_id']].append(record)
    result=dict(phase='evaluation',identity=phase['identity'],source_files=phase['files'],
        threshold_file=str(threshold_path.resolve()),threshold_file_sha256=sha256(threshold_path.read_bytes()).hexdigest(),
        overall=_decision_summary(records,thresholds),
        by_case={key:_decision_summary(value,thresholds) for key,value in sorted(groups.items())},
        descriptive=descriptive_analysis(phase),
        by_history=[dict(case_id=r['case_id'],seed=r['seed'],true_name=r['true_name'],methods={name:
            {inference:decision(r,name,inference,thresholds['methods'][name][inference]) for inference in INFERENCE}
            for name in METHODS}) for r in records],
        counts_scope='Categories partition the original histories. False rejection overlaps decision categories; failures keep all.')
    _write(Path(output)/'evaluation_summary.json',result)
    return result


def _markdown(summary,decisions=None):
    overall=summary['overall'];lines=['# EV多场景统一对照','',f"历史槽数：{overall['histories']}；不同seed：{overall['unique_seed_count']}。",'',
        'M=min(竞争者SSE)−真候选SSE；较大表示该次观测的排错裕量较大。不同推断路线分别校准。','',
        '|方法|推断|完整/总数|平均M|中位M|对同域平方W/T/L/缺失|平均规划秒|平均总秒|',
        '|---|---|---:|---:|---:|---|---:|---:|']
    def fmt(value):return '—' if value is None else format(value,'.6g')
    for name in METHODS:
        row=overall['methods'][name]
        for inference in INFERENCE:
            v=row['inference'][inference];w=v['vs_same_domain_square']
            lines.append('|'+ '|'.join((name,inference,str(v['complete'])+'/'+str(v['denominator']),
                fmt(v['M']['mean']),fmt(v['M']['median']),'/'.join(str(w[k]) for k in ('win','tie','loss','incomplete')),
                fmt(row['planning_seconds']['mean']),fmt(v['total_seconds']['mean'] if v['total_seconds']['n']
                    else row['total_seconds']['mean'])))+'|')
    if decisions is not None:
        lines.extend(['','|方法|推断|误拒/总数|唯一正确|唯一错误|未决|失败保留全部|漏真|',
            '|---|---|---:|---:|---:|---:|---:|---:|'])
        for name in METHODS:
            for inference in INFERENCE:
                v=decisions['overall'][name][inference]
                lines.append('|'+ '|'.join((name,inference,str(v['false_reject'])+'/'+str(v['denominator']),
                    *(str(v[k]) for k in ('unique_correct','unique_wrong','unresolved','failed_keep_all','candidate_miss'))))+'|')
        lines.extend(['','未决列只含已完整计算、仍保留多个候选的历史；计算失败另列并保留全部候选。',
            '阈值针对声明的IID混合分布，分场景计数不承诺同样覆盖；分数和似然不表示拓扑正确概率。'])
    else:lines.extend(['','开发表包含配对及压力场景；不同场景共享seed的槽不当作独立历史。无正式校准结论。'])
    return '\n'.join(lines)+'\n'


def analyze_development(directory,output):
    phase=read_phase(directory,expected_phase='development');result=descriptive_analysis(phase)
    _write(Path(output)/'development_summary.json',result)
    path=Path(output)/'development_summary.md';path.write_text(_markdown(result),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development',type=Path);parser.add_argument('--calibration',type=Path)
    parser.add_argument('--evaluation',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--alpha',type=float,default=.1)
    args=parser.parse_args()
    if args.development is not None:
        if args.calibration is not None or args.evaluation is not None:parser.error('choose development or formal mode')
        analyze_development(args.development,args.output);return
    if args.calibration is None or args.evaluation is None:parser.error('formal mode needs calibration and evaluation')
    calibration=read_phase(args.calibration,expected_phase='calibration')
    thresholds=calibrate(calibration,args.output,alpha=args.alpha,expected_count=108)
    # The first evaluation-directory read is deliberately after immutable save.
    evaluation=read_phase(args.evaluation,expected_phase='evaluation')
    result=evaluate(evaluation,thresholds,args.output,expected_count=72,calibration_records=calibration['records'])
    _write(args.output/'calibration_summary.json',descriptive_analysis(calibration))
    (args.output/'summary.md').write_text(_markdown(result['descriptive'],result),encoding='utf-8')


if __name__=='__main__':main()
