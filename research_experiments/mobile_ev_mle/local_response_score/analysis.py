"""Paired descriptive summaries; no fitting, model selection or threshold tuning."""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import numpy as np

from ..local_rx_identifiability.experiment import read, write
from ..local_rx_three_day.experiment import key
from .experiment import METHODS, sha

NAMES=('star','pair01','pair02','pair12')
TOL=1e-5


def stats(values):
    array=np.asarray([v for v in values if v is not None and np.isfinite(v)],float)
    return dict(count=len(array),mean=float(array.mean()) if len(array) else None,
                median=float(np.median(array)) if len(array) else None,
                minimum=float(array.min()) if len(array) else None,
                maximum=float(array.max()) if len(array) else None)


def rank(values,available=True):
    if not available or len(values)!=4 or not np.isfinite(list(values.values())).all():
        return dict(available=False)
    order=sorted(NAMES,key=lambda name:(values[name],name))
    first=[name for name in order if values[name]-values[order[0]] <= TOL]
    return dict(available=True,order=order,first=first,values=values,
                margin_true=min(values[name] for name in NAMES if name!='star')-values['star'])


def comparison(a,b):
    if not a.get('available') or not b.get('available'):
        return dict(available=False)
    agrees=ties=0
    for x,y in combinations(NAMES,2):
        da=a['values'][x]-a['values'][y]; db=b['values'][x]-b['values'][y]
        sa=0 if abs(da)<=TOL else np.sign(da)
        sb=0 if abs(db)<=TOL else np.sign(db)
        agrees+=sa==sb;ties+=sa==0 or sb==0
    return dict(available=True,top_same=set(a['first'])==set(b['first']),
                full_order_same=a['order']==b['order'],pairwise_agreement=int(agrees),
                pairwise_denominator=6,tied_pair_count=ties)


def summarize(directory):
    directory=Path(directory);config=read(directory/'configuration.json')
    rows=[];conditions=[]
    for spec in config['tasks']:
        name=key(spec);record=read(directory/'records'/f'{name}.json.gz')
        # Evaluation label is read only here, after every score is sealed.
        truth=read(Path(config['baseline'])/'truth'/f'{name}.json.gz')
        if truth['candidate_name']!='star':raise ValueError('unexpected evaluation label')
        item=dict(key=name,spec=spec,status=record['status'],ranks={},comparisons={})
        for method in METHODS:
            table=record['methods'].get(method,{})
            ranks={}
            for phase in ('train','holdout_own','holdout_AC','holdout_profile'):
                values={};qualified=True
                for candidate,entry in table.items():
                    fit=entry['training'];hold=entry['holdout']
                    target=(fit if phase=='train' else hold.get('own',{}) if phase=='holdout_own'
                            else hold.get('ac',{}) if phase=='holdout_AC' else hold.get('profile',{}))
                    if 'loss' not in target:qualified=False;continue
                    if not fit.get('qualified'):qualified=False
                    if phase=='holdout_AC' or (phase=='holdout_own' and method=='AC'):
                        qualified=qualified and bool(target.get('qualified'))
                    values[candidate]=float(target['loss'])
                ranks[phase]=rank(values,qualified)
            item['ranks'][method]=ranks
            for candidate,entry in table.items():
                fit=entry['training'];hold=entry['holdout'];ac=hold.get('ac',{})
                rows.append(dict(**spec,key=name,method=method,candidate=candidate,
                    train_qualified=fit.get('qualified',False),train_loss=fit.get('loss'),
                    train_seconds=fit.get('seconds'),train_ac_calls=fit.get('ac_calls',0),
                    train_response_calls=fit.get('response_calls',0),
                    train_gap=fit.get('absolute_gap'),train_pg=fit.get('projected_gradient'),
                    original_line_domain_feasible=hold.get('original_line_domain_feasible',False),
                    holdout_loss=hold.get('own',{}).get('loss'),
                    holdout_ac_loss=ac.get('loss'),holdout_ac_qualified=ac.get('qualified',False),
                    holdout_ac_status=ac.get('status'),holdout_ac_seconds=ac.get('seconds'),
                    holdout_ac_calls=ac.get('ac_calls',0),
                    holdout_profile_loss=hold.get('profile',{}).get('loss'),
                    evaluation_label_read_after_scores=True))
        for method in METHODS[1:]:
            item['comparisons'][method]={phase:comparison(item['ranks']['AC'][phase],item['ranks'][method][phase])
                for phase in ('train','holdout_own','holdout_AC','holdout_profile')}
            # Direct scalar-score ranking against full AC's noisy holdout ranking.
            item['comparisons'][method]['response_vs_full_AC_holdout']=comparison(
                item['ranks']['AC']['holdout_own'],item['ranks'][method]['holdout_own'])
        conditions.append(item)
    groups=[]
    for network in ('all','paper15','paper11'):
        for length in ('all',288,672):
            selected=[c for c in conditions if (network=='all' or c['spec']['network']==network)
                      and (length=='all' or c['spec']['N']==length)]
            selected_keys={c['key'] for c in selected}
            group=dict(network=network,N=length,conditions=len(selected),methods={})
            for method in METHODS:
                subset=[row for row in rows if row['key'] in selected_keys and row['method']==method]
                values=dict(training_qualified=sum(r['train_qualified'] for r in subset),candidate_denominator=len(subset),
                    original_line_domain_feasible=sum(r['original_line_domain_feasible'] for r in subset),
                    holdout_ac_qualified=sum(r['holdout_ac_qualified'] for r in subset),
                    train_seconds=stats([r['train_seconds'] for r in subset]),
                    train_seconds_total=sum(r['train_seconds'] or 0 for r in subset),
                    train_gap=stats([r['train_gap'] for r in subset]),
                    holdout_ac_seconds=stats([r['holdout_ac_seconds'] for r in subset]),
                    holdout_ac_calls_total=sum(r['holdout_ac_calls'] for r in subset),rankings={})
                for phase in ('train','holdout_own','holdout_AC','holdout_profile'):
                    rankings=[c['ranks'][method][phase] for c in selected if c['ranks'][method][phase]['available']]
                    values['rankings'][phase]=dict(available=len(rankings),condition_denominator=len(selected),
                        true_in_first=sum('star' in r['first'] for r in rankings),
                        true_unique_first=sum(r['first']==['star'] for r in rankings),
                        margin_true=stats([r['margin_true'] for r in rankings]))
                if method!='AC':
                    comp=[c['comparisons'][method]['response_vs_full_AC_holdout'] for c in selected
                          if c['comparisons'][method]['response_vs_full_AC_holdout']['available']]
                    training=[c['comparisons'][method]['train'] for c in selected if c['comparisons'][method]['train']['available']]
                    values['AC_comparison']=dict(paired=len(comp),condition_denominator=len(selected),
                        same_holdout_top=sum(c['top_same'] for c in comp),
                        same_holdout_full_order=sum(c['full_order_same'] for c in comp),
                        holdout_pairwise_agreement=sum(c['pairwise_agreement'] for c in comp),
                        holdout_pairwise_denominator=6*len(comp),
                        same_training_top=sum(c['top_same'] for c in training),
                        same_training_full_order=sum(c['full_order_same'] for c in training),
                        train_pairwise_agreement=sum(c['pairwise_agreement'] for c in training),
                        train_pairwise_denominator=6*len(training))
                group['methods'][method]=values
            groups.append(group)
    answer=dict(rows=rows,conditions=conditions,groups=groups,
                independent_network_seed_families=8,paired_training_lengths=True,
                field_data=False,true_topology_labels_used_for_scoring=False,
                truth_only_used_for_posthoc_descriptive_accuracy=True,
                cached_AC_cost='original training fit seconds from frozen batch, not rerun wall time',
                output_reproducibility='scores/data hashes in records; source manifest in configuration')
    same_parameters=[]; score_shift=[]; fitted_parameter_shift=[]
    for c in conditions:
        base=c['ranks']['AC']['holdout_AC']
        fixed=c['ranks']['AC']['holdout_profile']
        changed=c['ranks']['RESP']['holdout_profile']
        same_parameters.append(comparison(base,fixed))
        if base.get('available') and fixed.get('available') and changed.get('available'):
            score_shift.append(fixed['margin_true']-base['margin_true'])
            fitted_parameter_shift.append(changed['margin_true']-fixed['margin_true'])
    answer['same_AC_training_parameters']=dict(
        conditions=len(same_parameters),same_holdout_top=sum(c.get('top_same',False) for c in same_parameters),
        pairwise_agreement=sum(c.get('pairwise_agreement',0) for c in same_parameters),
        pairwise_denominator=6*sum(c.get('available',False) for c in same_parameters),
        margin_shift_due_to_scoring=stats(score_shift),
        margin_shift_due_to_refitting_response_parameters=stats(fitted_parameter_shift))
    if (directory/'benchmark.json').exists():
        timing=read(directory/'benchmark.json')
        answer['fresh_timing']=dict(wall_seconds=timing['seconds'],
            all_losses_reproduced=timing['all_losses_reproduced'],
            qualified=timing['qualified'],denominator=timing['denominator'],methods={})
        for method in METHODS:
            subset=[r for r in timing['rows'] if r['method']==method]
            answer['fresh_timing']['methods'][method]=dict(
                seconds=stats([r.get('seconds') for r in subset]),
                seconds_total=sum(r.get('seconds',0) for r in subset),
                ac_calls=sum(r.get('ac_calls',0) for r in subset),
                qualified=sum(r['qualified'] for r in subset),denominator=len(subset))
    if (directory/'physics.json').exists():
        physics=read(directory/'physics.json');pairs=[p for r in physics['records'] for p in r.get('pairs',[])]
        candidates=[(name,v) for r in physics['records'] for name,v in r.get('candidates',{}).items()]
        envelopes=[e for _,v in candidates for e in v['envelopes']]
        answer['physics']=dict(pair_denominator=len(pairs),
            uniform_radius_below_fitted_difference=sum(p['AC_radius_below_fitted_difference'] for p in pairs),
            uniform_radius_to_fitted_difference=stats([p['radius_to_difference'] for p in pairs]),
            centered_radius_below_centered_difference=sum(p['centered_radius_below_centered_difference'] for p in pairs),
            fitted_pair_response_RMS_sigma=stats([p['fitted_linear_response_distance']['per_coordinate_RMS'] for p in pairs]),
            public_to_observed_tightening=stats([v['public_to_meter_radius_ratio'] for _,v in candidates]),
            mean_feasible_remainder_RMS_sigma=stats([v['feasible_AC_point_remainder']['per_coordinate_RMS'] for _,v in candidates]),
            uniform_centered_radius_RMS_sigma=stats([v['uniform_radius']['per_coordinate_RMS'] for _,v in candidates]),
            uniform_absolute_radius_RMS_sigma=stats([v['uniform_absolute_radius']['per_coordinate_RMS'] for _,v in candidates]),
            envelope_qualified=sum(e['qp']['qualified'] for e in envelopes),envelope_denominator=len(envelopes),
            envelope_gap=stats([e['qp'].get('absolute_gap') for e in envelopes]),
            conditional_envelope_exceeded=sum(e['rule'].get('envelope_exceeded',False) for e in envelopes),
            active_rejections=sum(e['rule'].get('reject',False) for e in envelopes),
            strict_rejection_activated=False)
    write(directory/'summary.json',answer)
    group=groups[0]
    print(dict(conditions=group['conditions'],independent_families=8,
               methods={m:dict(train_true=v['rankings']['train']['true_unique_first'],
                              holdout_true=v['rankings']['holdout_own']['true_unique_first'],
                              train_seconds=v['train_seconds'],AC_comparison=v.get('AC_comparison'),
                              ac_qualified=v['holdout_ac_qualified']) for m,v in group['methods'].items()},
               physics=answer.get('physics')),flush=True)
    return answer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True)
    summarize(parser.parse_args().directory)


if __name__=='__main__':main()
