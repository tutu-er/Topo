"""A preregistered second site heuristic using historical blocks and zero leaves.

No actual-root path, new natural data or physical truth is an input. The rule
changes sites only; price/response quantities, budgets and replay are unchanged.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from . import study
from .physical_bridge import load_history
from .local_replay import plan_case,effective_family


def sites(old,indices,seed):
    points=[old['base_points'][i] for i in indices]
    fs=[{tuple(sorted(s)) for s in p['supports']} for p in points]
    crosses=[]
    for i,f in enumerate(fs):
        for j,g in enumerate(fs[i+1:],i+1):
            for c in sorted(f):
                for d in sorted(g):
                    a,b=set(c),set(d)
                    if a&b and not a<=b and not b<=a:crosses.append((i,j,c,d))
    if not crosses:raise ValueError('no historical crossing')
    i,j,c,d=sorted(crosses)[0];rng=np.random.default_rng(seed)
    intersection=int(rng.choice(sorted(set(c)&set(d))))
    # Raw support lists always retain leaves. Use effective positive weights,
    # rather than interpreting listed zero leaves as observations of a line.
    positive=[effective_family(p) for p in points]
    disagree=[k for k in range(len(old['labels'])) if 0<sum((k,) in f for f in positive)<len(points)]
    if not disagree:raise ValueError('no historical zero/nonzero leaf disagreement')
    strength={}
    for k in disagree:
        values=[]
        for p in points:
            v=0.
            for support,r,x in zip(p['supports'],p['r'],p['x']):
                if support==[k]:v+=r+.35*x
            values.append(v)
        strength[k]=max(values)-min(values)
    maximum=max(strength.values())
    tied=[k for k in disagree if strength[k]>=maximum-1e-12 and k!=intersection]
    leaf=int(rng.choice(sorted(tied)))
    selected=[old['labels'][intersection],old['labels'][leaf]]
    return selected,dict(rule='crossing intersection + largest raw leaf zero/positive response-weight disagreement; proposal only',
        seed=seed,crossing_indices=[list(c),list(d)],intersection_candidates=sorted(set(c)&set(d)),
        leaf_candidates=[old['labels'][k] for k in disagree],
        historical_leaf_spreads={str(old['labels'][k]):v for k,v in strength.items()},
        selected_labels=selected,true_topology_used=False,actual_future_or_scores_read=False)


def main():
    a=argparse.ArgumentParser();a.add_argument('--planning-root',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True);x=a.parse_args()
    cfg=study.verify(x.planning_root)
    if x.out.exists():raise RuntimeError('existing site-ablation results retained')
    x.out.mkdir()
    row=next(r for r in cfg['cases'] if r['job_id']==cfg['local_price_case'])
    old=study.read(row['existing_result']['path'])
    pair,receipt=sites(old,cfg['representative_indices'],cfg['seeds']['site'])
    protocol=dict(created_utc=study.timestamp(),status='frozen_before_new_site_LP',
        source_sha256={str(p):study.sha(p) for p in Path(__file__).parent.glob('*.py')},
        parent_protocol_sha256=study.sha(x.planning_root/'protocol.json'),
        old_result_sha256=row['existing_result']['catalog_sha256'],active_labels=pair,site_rule=receipt,
        representative_indices=cfg['representative_indices'],same_price_budget=1.6,
        same_reduction_kw=.2,same_reduction_duration_hours=1.,same_current_replay_domain=True,
        actual_evaluation_phase='already computing separate fixed controls; no actual values or structure scores are input/read for this ablation',
        interpretation='site proposal is a heuristic; final price objective still full WZZT output error, not leaf-spread score')
    study.write(x.out/'protocol.json',protocol)
    history=load_history(row)
    result=plan_case(history,old,cfg['representative_indices'],pair,x.out/'response_table',
        wall_seconds=600.,tie_seed=cfg['seeds']['tie'],path_seconds=15.,max_LP_calls=6000,
        proposal_scope='historical_representative_union')
    print(json.dumps({k:result.get(k) for k in ('status','LP_calls','training_pairs_completed','accepted_moves','accepted_crossing_moves')}),flush=True)


if __name__=='__main__':main()
