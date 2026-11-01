"""Post-freeze sensitivity to retaining complete historical model families.

No prices, physics, topology search or production defaults change. All new
coefficients are fitted on training only. This named selection-portfolio audit
is kept separate from the original complete RNJ+WZZT actual outcomes.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import pickle
import sys
import traceback

import numpy as np

from . import continuous_search as base
from .audit_existing import read,write,ref,bound
from .profile_margin import validation
from research_experiments.wzzt_dr_selection.dr_pool_v2.engine import PoolLP
from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize,arrays,matrices
from research_experiments.wzzt_dr_selection.dr_pool_v2.selection import select
from research_experiments.wzzt_dr_selection.dr_balanced_design.budget_reference_diagnostic import _family,_error


def run(source,actual_root,output):
    actual_root,output=Path(actual_root),Path(output);output.mkdir(parents=True,exist_ok=False)
    freeze=actual_root/'test/fits_frozen.json'
    if not freeze.exists():raise RuntimeError('actual fits must be frozen before this diagnostic')
    scored=read(actual_root/'test/scored.json');config=read(actual_root/'protocol.json')
    ctx=base.context(source);c=ctx['source'];labels=c['labels']
    with bound(c['history_packet']).open('rb') as stream:history=pickle.load(stream)['history']
    historical=read(bound(c['historical_baseline_result']))
    sources={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    write(output/'protocol.json',dict(status='frozen',created_utc=base.now(),id='post_actual_retained_history_portfolio_audit_v1',
        actual_fits_frozen=ref(freeze),actual_scored=ref(actual_root/'test/scored.json'),
        actual_price_freeze=ref(actual_root/'price_freeze.json'),history=c['history_packet'],
        historical_fit=c['historical_baseline_result'],source_sha256=sources,
        expected_records=8,distinct_historical_families=len(ctx['families']),maximum_new_training_LP=136,
        no_price_update=True,new_AC=0,new_RNJ_W_search=0,original_3pct=.03,original_contraction=.16,
        scope='named post-evaluation sensitivity, not the original pipeline or a new independent price test'))
    rows=[];calls=0
    for original in scored['rows']:
        row=dict(policy=original['policy'],path=original['path'],status='started')
        try:
            with bound(original['packet']).open('rb') as stream:packet=pickle.load(stream)
            train,_,_=normalize(history['training']+packet['training'],'train')
            valid,_,_=normalize(history['validation']+packet['validation'],'valid')
            tr,va=arrays(train),arrays(valid)
            assert tr[0].shape[0]==va[0].shape[0]==480
            actual=read(bound(original['result']))
            lp=PoolLP(train,ctx['pool'],lp_method=2)
            refitted=[];frozen=[];families=[]
            try:
                for f in ctx['families']:
                    p=lp.fit(f['requested']);p=dict(p,validation_mae=validation(p,va),
                        origin='retained_historical_joint_refit',historical_family_id=f['family_id'])
                    refitted.append(p);families.append(f['requested'])
                    old=historical['selection_candidates'][f['source_indices'][0]]
                    rr,xx=f['historical_r'],f['historical_x']
                    frozen.append(dict(old,origin='historical_frozen_predictor',
                        historical_family_id=f['family_id'],
                        objective=float(np.abs(tr[0]@rr.T+tr[1]@xx.T-tr[2]).mean()),
                        validation_mae=float(np.abs(va[0]@rr.T+va[1]@xx.T-va[2]).mean())))
                receipts=list(lp.solves);calls+=len(receipts)
            finally:lp.close()
            points=actual['selection_candidates']+refitted
            retained=select(points,actual['calibrations']['raw'])
            fixed=select(frozen,historical['calibrations']['raw'])
            coarse_index=families.index(ctx['coarse']);fine_index=families.index(ctx['fine'])
            v0=refitted[coarse_index]['validation_mae'];v1=refitted[fine_index]['validation_mae']
            h0=frozen[coarse_index]['validation_mae'];h1=frozen[fine_index]['validation_mae']
            margin=v0-1.03*min(p['validation_mae'] for p in points)-1e-12
            predicted=[]
            for index in [coarse_index,fine_index]:
                p=refitted[index];rr,xx=matrices(len(labels),p['supports'],p['r'],p['x'])
                f=ctx['families'][index]
                predicted.append(float(np.abs(va[0]@(rr-f['historical_r']).T+va[1]@(xx-f['historical_x']).T).mean()))
            pair_change=(v0-1.03*v1)-(h0-1.03*h1)
            triangle=predicted[0]+1.03*predicted[1]
            assert abs(pair_change)<=triangle+1e-12
            # Recompute the original selector order independently.
            best=min(p['validation_mae'] for p in points)
            eligible=[i for i,p in enumerate(points) if p['validation_mae']<=1.03*best+1e-12]
            selected=min(eligible,key=lambda i:(len(points[i]['supports']),points[i]['validation_mae'],i))
            assert selected==retained['selection_index']
            # This is the first truth read of this audit, after all new fits and
            # both selectors above are fixed; it does not feed any design.
            truth=read(Path(original['physical_output'])/'truth_evaluation.json')['true_supports']
            truth={tuple(sorted(s)) for s in truth}
            errors=_error(_family(retained,labels),truth)
            fixed_errors=_error(_family(fixed,labels),truth)
            private=dict(refitted=refitted,retained_selection=retained,frozen_selection=fixed,LP_receipts=receipts)
            with (output/f"private_{original['record_id']}.pkl").open('wb') as stream:pickle.dump(private,stream,protocol=4)
            row.update(status='complete',new_training_LP_calls=len(receipts),
                original_full_errors=original['models']['multistart_original_rule']['full_errors'],
                retained_history_full_errors=errors['full_errors'],retained_history_missing=errors['missing'],
                retained_history_extra=errors['extra'],retained_selected_origin=retained['selected_origin'] if 'selected_origin' in retained else retained['origin'],
                retained_raw_blocks=len(retained['selected_uncontracted']['supports']),
                retained_final_blocks=len(retained['supports']),current_coarse_exit_margin=margin,
                current_coarse_eligible=bool(v0<=1.03*best+1e-12),
                frozen_predictor_pool_full_errors=fixed_errors['full_errors'],
                frozen_predictor_coarse_margin=h0-1.03*min(p['validation_mae'] for p in frozen)-1e-12,
                pair_frozen_margin=h0-1.03*h1-1e-12,pair_joint_refit_margin=v0-1.03*v1-1e-12,
                pair_margin_change=pair_change,validation_prediction_update_bound=triangle,
                raw_target=ctx['target'] in {tuple(s) for s in retained['selected_uncontracted']['supports']},
                final_target=ctx['target'] in {tuple(s) for s in retained['supports']},actual_fee=original['actual_payment'])
        except Exception as error:
            row.update(status='failed',error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        rows.append(row);write(output/'progress.json',dict(status='running',rows=rows))
        print(__import__('json').dumps({k:row.get(k) for k in ['policy','path','status','original_full_errors',
            'retained_history_full_errors','current_coarse_exit_margin','frozen_predictor_pool_full_errors']}),flush=True)
    summary=dict(status='complete',rows=rows,denominator=len(scored['rows']),
        completed=sum(r['status']=='complete' for r in rows),new_training_LP_calls=calls,
        new_AC=0,new_RNJ_W_search=0,new_actual_observations=0,
        original_actual_results_unchanged=True,quotes_updated_after_actual=False,
        portfolio_scope='actual selection candidates plus ALL17 historical complete-family training refits',
        frozen_scope='historical17 predictors with historical calibration, only original validation selection; not RNJ+W')
    write(output/'summary.json',summary);write(output/'ended.json',dict(status='complete'))
    return summary


if __name__=='__main__':
    for directory in (base.REPO,base.REPO/'rnj_wzzt_core'):
        if str(directory) not in sys.path:sys.path.insert(0,str(directory))
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=base.DEFAULT_SOURCE)
    p.add_argument('--actual-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.actual_root,a.output)
