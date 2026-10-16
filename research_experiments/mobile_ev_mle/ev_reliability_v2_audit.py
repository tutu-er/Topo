"""Read-only audit of tied aggregate results; no AC call or action selection."""
from __future__ import annotations
import argparse
from collections import Counter
from decimal import Decimal,ROUND_CEILING
from hashlib import sha256
import itertools
import json
from pathlib import Path

from . import ev_systematic_analyze as base
from . import ev_reliability_v2 as rel


TIED_METHODS=('square_fixed','square_mobile','nearest_fixed','scalar_fixed','scalar_hist_fixed')


def curve_difference(a,b):
    if not isinstance(a,list) or not isinstance(b,list) or len(a)!=len(b):
        return None
    if any(len(x)!=len(y) for x,y in zip(a,b)):
        return None
    return max((abs(float(x)-float(y)) for ar,br in zip(a,b) for x,y in zip(ar,br)),default=0.)


def audit(phase,thresholds,*,raw_directory=None):
    records=phase['records'];pairs={}
    for left,right in itertools.combinations(TIED_METHODS,2):
        results=[]
        for row in records:
            methods=base._methods(row)
            a=base.decision(row,left,'joint_total',thresholds['methods'][left]['joint_total'])
            b=base.decision(row,right,'joint_total',thresholds['methods'][right]['joint_total'])
            delta=curve_difference(methods[left].get('power_kw'),methods[right].get('power_kw'))
            order_a=sorted(a['losses'],key=lambda k:(a['losses'][k],k))
            order_b=sorted(b['losses'],key=lambda k:(b['losses'][k],k))
            results.append(dict(seed=row['seed'],case_id=row['case_id'],curve_max_difference_kw=delta,
                same_curve_1e_6_kw=delta is not None and delta<=1e-6,
                same_raw_argmin=a['minimum_candidates']==b['minimum_candidates'],
                same_full_loss_order=order_a==order_b,same_retained=a['retained']==b['retained'],
                same_decision_category=a['category']==b['category'],
                M_difference=None if a['M'] is None or b['M'] is None else b['M']-a['M']))
        pairs[left+'__'+right]=dict(denominator=len(records),
            **{key:sum(r[key] for r in results) for key in ('same_curve_1e_6_kw','same_raw_argmin',
                'same_full_loss_order','same_retained','same_decision_category')},histories=results)
    failed=[]
    for record in records:
        reference=base.decision(record,'square_fixed','joint_total',thresholds['methods']['square_fixed']['joint_total'])
        if not reference['false_reject']:
            continue
        parameters,error=rel.candidate_parameters(record)
        summary=dict(seed=record['seed'],case_id=record['case_id'],true_name=record['true_name'],
            candidate_parameters=parameters,structural_error=error,methods={})
        for method in base.METHODS:
            original=base.decision(record,method,'joint_total',thresholds['methods'][method]['joint_total'])
            penalized=rel.scored_mode(record,method,'joint_total','global_AIC')
            summary['methods'][method]=dict(losses=original['losses'],M=original['M'],
                threshold=thresholds['methods'][method]['joint_total'],retained=original['retained'],
                candidate_minus_true={k:s-original['losses'][record['true_name']] for k,s in original['losses'].items()},
                AIC_scores=penalized['scores'])
        if raw_directory is not None:
            files=list(Path(raw_directory).glob('*_'+str(record['seed'])+'.json.gz'))
            if len(files)!=1:raise ValueError('expected exactly one raw history file')
            path=files[0];raw=base._read(path)
            summary['raw_file']=str(path.resolve());summary['raw_sha256']=sha256(path.read_bytes()).hexdigest()
            summary['shared_fits']={}
            for method in ('square_fixed','scalar_fixed','scalar_mobile'):
                action=base._methods(raw)[method]['action_key'];comparisons=raw['future_records'][action]['comparisons']
                summary['shared_fits'][method]={}
                for name,comparison in comparisons.items():
                    post=comparison['joint']['post'];optimizer=post.get('optimizer') or {}
                    summary['shared_fits'][method][name]=dict(loss=post['loss'],state=post['state'],
                        blocks={k:dict(loss=v['loss'],channel_loss=v['channel_loss']) for k,v in post['blocks'].items()},
                        selected_origin=post.get('selected_origin'),success=post.get('success'),
                        optimization_failure=post.get('optimization_failure'),
                        diagnostics=optimizer.get('diagnostics',{}))
        failed.append(summary)
    return dict(schema='ev_reliability_v2_tie_audit_v1',history_count=len(records),
        source_files=phase['files'],truth_counts=dict(Counter(r['true_name'] for r in records)),
        tied_methods=list(TIED_METHODS),pairwise=pairs,square_false_rejections=failed,
        thresholds={m:thresholds['methods'][m]['joint_total'] for m in TIED_METHODS},
        statement='Old evaluation is now development evidence; no action or threshold is selected here.')


def independent_verify(calibration, evaluation, thresholds, report):
    """Recompute scalar scores, ranks and decisions without the v2 scorer.

    This quick check is for complete formal records with valid stored supports.
    It verifies every method/inference/rule, not only the published table rows.
    """
    errors=[];cal_checks=0;decision_checks=0
    def compute(record,method,mode,rule):
        bank=sorted(row['name'] for row in record['candidates'])
        dims={row['name']:2*len(row['supports']) for row in record['candidates']}
        method_row=next(row for row in record['methods'] if row['name']==method)
        inf=method_row['inference'][mode]
        complete=bool(inf['complete'] and method_row.get('operation',{}).get('success',True)
                      and sorted(inf['losses'])==bank)
        gamma=2 if rule in ('global_AIC','label_AIC') else 0
        scored={key:float(value)+gamma*dims[key] for key,value in inf['losses'].items()}
        best=min(scored.values()) if complete else 0
        relative={key:max(0.,value-best) for key,value in scored.items()} if complete else {}
        return bank,complete,relative
    def group(record,rule,candidate):
        if rule.startswith('label_'):
            row=next(row for row in record['candidates'] if row['name']==candidate)
            supports=sorted(tuple(sorted(s)) for s in row['supports'])
            n=max(i for support in supports for i in support)+1
            return json.dumps(dict(n=n,supports=supports),sort_keys=True,separators=(',',':'))
        return str(record['config']['training_points']) if rule=='H_R' else 'all'
    for rule in rel.RULES:
        for method in thresholds['methods']:
            for mode in base.INFERENCE:
                groups={}
                for row in calibration['records']:
                    _,complete,relative=compute(row,method,mode,rule)
                    key=group(row,rule,row['true_name'])
                    groups.setdefault(key,[]).append(relative[row['true_name']] if complete else 0.)
                stored=thresholds['rules'][rule][method][mode]
                for key,value in stored.items():
                    scores=groups.get(key,[]);n=len(scores)
                    rank=int((Decimal(n+1)*(Decimal(1)-Decimal(str(thresholds['alpha'])))).to_integral_value(rounding=ROUND_CEILING))
                    limit=None if rank>n else sorted(scores)[rank-1]
                    cal_checks+=1
                    if (value['n'],value['order_rank'],value['threshold'],value['scores'])!=(n,rank,limit,scores):
                        errors.append(dict(kind='threshold',rule=rule,method=method,mode=mode,group=key))
    saved={row['seed']:row for row in report['by_history']}
    for row in evaluation['records']:
        for rule in rel.RULES:
            for method in thresholds['methods']:
                for mode in base.INFERENCE:
                    bank,complete,relative=compute(row,method,mode,rule)
                    kept=[]
                    if not complete:kept=bank
                    else:
                        for key in bank:
                            threshold=thresholds['rules'][rule][method][mode].get(group(row,rule,key))
                            if threshold is None or threshold['infinite'] or relative[key]<=threshold['threshold']:
                                kept.append(key)
                    category=('failed_keep_all' if not complete else 'unique_correct' if kept==[row['true_name']]
                              else 'unique_wrong' if len(kept)==1 else 'unresolved')
                    actual=saved[row['seed']]['rules'][rule][method][mode]
                    decision_checks+=1
                    if actual['retained']!=kept or actual['category']!=category or actual['false_reject']!=(row['true_name'] not in kept):
                        errors.append(dict(kind='decision',seed=row['seed'],rule=rule,method=method,mode=mode))
    return dict(success=not errors,threshold_checks=cal_checks,decision_checks=decision_checks,errors=errors,
                coverage='independent scalar arithmetic on all complete, valid-support formal records')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--thresholds',type=Path,required=True)
    parser.add_argument('--raw-directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--verify-results',type=Path)
    args=parser.parse_args()
    if args.verify_results:
        if not args.calibration:parser.error('--verify-results requires --calibration')
        result=independent_verify(rel.read_phase(args.calibration),rel.read_phase(args.directory),
                                  base._read(args.thresholds),base._read(args.verify_results))
    else:
        result=audit(base.read_phase(args.directory),base._read(args.thresholds),raw_directory=args.raw_directory)
    base._write(args.output,result)


if __name__=='__main__':main()
