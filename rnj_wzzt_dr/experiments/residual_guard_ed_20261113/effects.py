"""Describe where the already completed DR evidence gains occur."""
from collections import defaultdict
import numpy as np
from ... import io
from ..linear_ac_population_20261113.analyze import load_phase
from ..unified_posterior_dr_20261112.history_scan import ROOT


def summarize(pairs,name,group):
    valid=[(b,o) for b,o in pairs if b['status']=='complete' and o['status']=='complete' and b['p_true'] is not None]
    return dict(group=name,stratum=group,inputs=len({b['id'] for b,o in pairs}),days=len(pairs),valid_truth_days=len(valid),
        baseline_mean=None if not valid else float(np.mean([b['p_true'] for b,o in valid])),
        optimized_mean=None if not valid else float(np.mean([o['p_true'] for b,o in valid])),
        mean_posterior_gain=None if not valid else float(np.mean([o['p_true']-b['p_true'] for b,o in valid])),
        gain_over_1pp=sum(o['p_true']-b['p_true']>.01 for b,o in valid),drop_over_1pp=sum(o['p_true']-b['p_true']<-.01 for b,o in valid),
        baseline_recovered=sum(b['map_correct'] for b,o in valid),optimized_recovered=sum(o['map_correct'] for b,o in valid),
        rescued_recovery=sum(not b['map_correct'] and o['map_correct'] for b,o in valid),
        harmed_recovery=sum(b['map_correct'] and not o['map_correct'] for b,o in valid),
        baseline_pass90=sum(b['true_pass90'] for b,o in valid),optimized_pass90=sum(o['true_pass90'] for b,o in valid),
        wrong_unique_before=sum(b['wrong_unique'] for b,o in pairs),wrong_unique_after=sum(o['wrong_unique'] for b,o in pairs),
        failed_quote=sum(o['planning_failed'] for b,o in pairs),mean_fee=float(np.mean([o['cost'] for b,o in pairs])))


def prior_effects(old):
    root=old/'cap30';rows=load_phase(root,'population');base={(r['id'],r['day']):r for r in rows if r['strategy']=='no_dr'}
    entries={e['request']['id']:e for e in io.read(root/'inputs.json')};history={}
    truth={v['id']:v for v in io.read(ROOT/'outputs/unified_posterior_dr_20261112_v1/evaluation.json')}
    for identity,e in entries.items():
        score=io.read(e['scores']['path']);r=base[identity,0]
        if not r['truth_covered']:history[identity]=None;continue
        # Existing complete evaluation identifies the true candidate only for
        # retrospective strata; it never enters a planning decision.
        index=next(i for i,m in enumerate(score['models']) if m['topology_id']==truth[identity]['truth_id'])
        history[identity]=score['models'][index]['probability']
    groups=defaultdict(list)
    def bucket(value):return '<50%' if value<.5 else '50%-90%' if value<.9 else '90%-99%' if value<.99 else '>=99%'
    for r in rows:
        if r['strategy']!='optimized_g2':continue
        b=base[r['id'],r['day']];pair=(b,r);p=history[r['id']]
        if p is None:groups['coverage','missing_truth'].append(pair);continue
        groups['coverage','truth_present'].append(pair)
        groups['historical_true_weight',bucket(p)].append(pair)
        groups['observed_no_action_true_weight',bucket(b['p_true'])].append(pair)
        groups['historical_MAP','correct' if b['historical_target_correct'] else 'wrong'].append(pair)
        groups['network',b['case']].append(pair)
    return [summarize(v,*k) for k,v in sorted(groups.items())]
