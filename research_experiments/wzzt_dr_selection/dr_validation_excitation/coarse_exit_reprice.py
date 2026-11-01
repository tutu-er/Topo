"""Exact current-coarse exit, reusing all completed full-family training fits.

This development correction is weaker than excluding EVERY no-C model. The
entire original family pool remains in the validation minimum. It changes no
3% rule, historical coefficient fit, physical domain or identifier.
"""
from __future__ import annotations
import argparse
from copy import copy
import hashlib
from pathlib import Path
import pickle
import sys

import numpy as np

from . import continuous_search as base
from .affine_solver import solve_affine
from .audit_continuous import audit_inner
from .audit_existing import read,write,ref,bound


def projection(tables, coarse_id):
    """A grouping adapter, NOT a change in actual block-C presence.

    Solver's legacy 'has_target' is the comparison-side flag here. Group0 is
    only F0. Group1 contains ALL families including a comparison alias of F0,
    so the expression equals v_F0-1.03*min_all(v)-eps even when negative.
    """
    result=[]
    for table in tables:
        families=[];coarse=None
        next_id=max(f['family_id'] for f in table['families'])+1
        for entry in table['families']:
            e=dict(entry, original_has_target=entry['has_target'],
                original_family_id=entry['family_id'],has_target=entry['family_id']!=coarse_id,
                comparison_alias=False)
            families.append(e)
            if entry['family_id']==coarse_id:coarse=e
        if coarse is None:raise ValueError('locked current coarse family missing')
        families.append(dict(coarse,family_id=next_id,has_target=True,comparison_alias=True))
        result.append(dict(table,families=families,comparison_available=True,
            comparison_kind='current_coarse_vs_best_over_all_families',
            actual_distinct_family_count=len(table['families'])))
    return result


def checked_solve(problem,tables,train,folder,seconds):
    folder.mkdir(parents=True,exist_ok=False)
    solved=solve_affine(problem,tables,train,seconds=seconds)
    solved['margin_formula']='current coarse validation-1.03*minimum over ALL historical families-1e-12'
    solved['group_flag_semantics']='comparison side; original block presence preserved separately'
    write(folder/'optimization.json',solved)
    if not solved.get('feasible_incumbent'):raise RuntimeError('no independently feasible price incumbent')
    audit=audit_inner(problem,tables,train,solved);write(folder/'independent_audit.json',audit)
    if not audit['audit_passed']:raise RuntimeError('exact inner arithmetic audit failed')
    check=base.evaluate_tables(problem,tables,solved['amplitudes'])
    check['criterion']='current coarse exits actual 3% band of all historical family predictions'
    for row in check['rows']:
        row['current_coarse_validation']=row.pop('no_target_validation')
        row['best_pool_validation']=row.pop('with_target_validation')
    write(folder/'criterion_check.json',check)
    return solved,check


def run(source,old_plan,output,seconds=30.):
    old_plan,output=Path(old_plan),Path(output)
    if not (old_plan/'ended.json').exists():raise RuntimeError('do not replay a still-running price batch')
    old=read(old_plan/'summary.json');old_protocol=read(old_plan/'protocol.json')
    ctx=base.context(source);problem=ctx['problem']
    coarse_id=next(f['family_id'] for f in ctx['families'] if f['requested']==ctx['coarse'])
    output.mkdir(parents=True,exist_ok=False)
    sources={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    protocol=dict(status='frozen',id='current_coarse_exit_continuous_reprice_v2',created_utc=base.now(),
        prior_protocol=ref(old_plan/'protocol.json'),prior_summary=ref(old_plan/'summary.json'),
        source_sha256=sources,history=ctx['source']['history_packet'],historical_fit=ctx['source']['historical_baseline_result'],
        problem=ctx['source']['problem'],budget=1.6,probability=.9,
        criterion='M0=v_current_coarse-1.03*min_ALL_historical_family_validation-1e-12',
        planning_samples=10,planning_seed=old_protocol['planning_seed'],model_holdout_samples=10,
        model_holdout_seed=old_protocol['model_holdout_seed'],coarse_family_id=coarse_id,
        inner_seconds=seconds,training_fit_reuse='all completed continuous training quotes from v1, no refit rerun',
        source_selection_before_actual=True,case_expansion=0,new_training_LP=0,
        alias_semantics='one expression alias of current coarse permits both comparison sides, not a new topology',
        actual_diagnostic_if_unmet='retain best phaseI quote; never call it90% feasible')
    write(output/'protocol.json',protocol);base._snapshot(output,sources)
    allzero=np.zeros(16)
    frozen_source=old_plan/'historical_frozen/private_tables.pkl'
    with frozen_source.open('rb') as stream:tables=pickle.load(stream)
    frozen,check=checked_solve(problem,projection(tables,coarse_id),allzero,output/'historical_frozen',seconds)
    write(output/'frozen_table_source.json',ref(frozen_source))
    rows=[]
    for record in old['evaluations']:
        row=dict(index=record['index'],status='not_available',training_prices=record['training_prices'])
        if record['status']=='complete':
            source_path=Path(record['folder'])/'private_tables.pkl'
            train=allzero.copy();train[[2,6]]=np.asarray(record['training_prices'])*problem.elasticities[[2,6]]
            try:
                with source_path.open('rb') as stream:tables=pickle.load(stream)
                solved,criterion=checked_solve(problem,projection(tables,coarse_id),train,
                    output/'outer'/f"eval_{record['index']:03d}",seconds)
                row.update(status='complete',quantile_margin=criterion['quantile_margin'],passes=criterion['passes'],
                    denominator=criterion['denominator'],amplitudes=solved['amplitudes'],prices=solved['prices'],
                    expected_cost=solved['original_evaluation']['expected_cost'],table_source=ref(source_path),
                    primary=solved['primary'],secondary=solved['secondary'],new_training_LP=0)
            except Exception as error:
                import traceback
                row.update(status='failed',error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        rows.append(row);write(output/'progress.json',dict(status='running',rows=rows))
        print(__import__('json').dumps(dict(stage='coarse_reprice',index=row['index'],status=row['status'],
            quantile_margin=row.get('quantile_margin'),passes=row.get('passes'))),flush=True)
    completed=[r for r in rows if r['status']=='complete']
    if not completed:raise RuntimeError('no coarse-exit price optimizer returned an audited incumbent')
    best=min(completed,key=lambda r:(-r['quantile_margin'],r['expected_cost'],r['index']))
    quotes=dict(zero=allzero.tolist(),rotation=read(bound(ctx['source']['plans']['rotation']))['global_expected_amplitudes'],
        continuous_validation=best['amplitudes'],historical_frozen=frozen['amplitudes'])
    selection=dict(status='frozen',created_utc=base.now(),quotes=quotes,selected_outer_index=best['index'],
        model_90_sample_feasible=best['passes']>=9,criterion=best,
        quote_status='candidate_needs_independent_validation' if best['passes']>=9 else 'diagnostic_phaseI_not90_feasible',
        planning_criterion=protocol['criterion'],whole_pool_retained=True,global_outer_optimality=False)
    write(output/'price_selection.json',selection)
    summary=dict(status='complete',evaluations=rows,completed=len(completed),denominator=len(rows),
        selected_refit_sample90_feasible=best['passes']>=9,selected_outer_index=best['index'],
        refit_arm_quantile=best['quantile_margin'],frozen_arm_quantile=check['quantile_margin'],
        price_selection=ref(output/'price_selection.json'),new_training_LP_calls=0,
        prior_training_LP_calls=old['total_training_LP_calls'],
        corrected_criterion=protocol['criterion'],case_expansion=0,new_actual_future=0)
    write(output/'summary.json',summary);write(output/'ended.json',dict(status='complete'))
    return summary


def holdout(source,planning_root,output):
    root,output=Path(planning_root),Path(output);output.mkdir(parents=True,exist_ok=False)
    decision=read(root/'price_selection.json');protocol=read(root/'protocol.json')
    ctx=base.context(source);scenarios=base.draw_scenarios(ctx,10,protocol['model_holdout_seed'])
    write(output/'protocol.json',dict(status='frozen',price_selection=ref(root/'price_selection.json'),
        samples=10,seed=protocol['model_holdout_seed'],criterion=protocol['criterion'],new_actual_future=0))
    cache={};rows=[];lp_calls=0
    for policy,amplitude in decision['quotes'].items():
        amp=np.asarray(amplitude,float);train=amp.copy();train[8:]=0.;key=tuple(train)
        if key not in cache:
            tables,called=base.affine_tables(ctx,scenarios,train);cache[key]=tables;lp_calls+=len(called)
        checked=base.evaluate_tables(ctx['problem'],projection(cache[key],protocol['coarse_family_id']),amp)
        checked['criterion']=protocol['criterion']
        for row in checked['rows']:
            row['current_coarse_validation']=row.pop('no_target_validation');row['best_pool_validation']=row.pop('with_target_validation')
        row=dict(policy=policy,**checked)
        if policy=='historical_frozen':
            own,_=base.affine_tables(ctx,scenarios,train,historical=True)
            row['own_frozen_predictor']=base.evaluate_tables(ctx['problem'],projection(own,protocol['coarse_family_id']),amp)
        rows.append(row);write(output/'progress.json',dict(status='running',rows=rows))
        print(__import__('json').dumps(dict(stage='model_holdout',policy=policy,passes=row['passes'],
            denominator=row['denominator'],quantile_margin=row['quantile_margin'])),flush=True)
    summary=dict(status='complete',rows=rows,training_LP_calls=lp_calls,distinct_training_quotes=len(cache),
        scenario_count=10,new_actual_future=0,new_AC=0,price_selection=ref(root/'price_selection.json'),
        selection_updated_after_holdout=False,scope='independent conditional model scenes, not actual AC or field probability')
    write(output/'summary.json',summary);write(output/'ended.json',dict(status='complete'))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['reprice','holdout'])
    p.add_argument('--source',type=Path,default=base.DEFAULT_SOURCE);p.add_argument('--planning-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--inner-seconds',type=float,default=30.)
    a=p.parse_args()
    if a.phase=='reprice':run(a.source,a.planning_root,a.output,a.inner_seconds)
    else:holdout(a.source,a.planning_root,a.output)


if __name__=='__main__':
    for path in (base.REPO,base.REPO/'rnj_wzzt_core'):
        if str(path) not in sys.path:sys.path.insert(0,str(path))
    main()
