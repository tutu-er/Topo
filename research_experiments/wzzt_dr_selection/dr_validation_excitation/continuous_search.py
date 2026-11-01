"""Continuous quotes for the original joint-fit/3%-validation selection rule.

Outer variables are two training-day prices (spread types). Each complete
historical family is refitted on all training rows. Conditional on those prices,
validation parameters are FROZEN and an exact finite-scenario MISOCP chooses
validation-day types and continuous prices. This is a bounded local outer
search, not a global certificate for all training types/prices or all trees.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import traceback
import zipfile

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
from scipy.optimize import minimize

from research_experiments.wzzt_dr_selection.dr_pool_v2.engine import PoolLP, canonical
from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize, arrays, matrices
from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
from research_experiments.wzzt_dr_selection.dr_conic_pricing.model import PriceProblem, evaluate_plan
from .audit_existing import read, ref, bound, write, family
from .affine_solver import solve_affine
from .audit_continuous import audit_inner

REPO = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = Path(r'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_conic_effects_v1/run_v1')


def now(): return datetime.now(timezone.utc).isoformat()


def quantile_margin(values, probability=.9):
    values = np.asarray(values, float)
    return float(np.sort(values)[len(values)-int(np.ceil(probability*len(values)))])


def context(source):
    """Historical data only. Never loads a saved test packet, graph or truth."""
    source = Path(source)
    c = read(source/'main/protocol.json')
    problem = PriceProblem.from_dict(read(bound(c['problem'])))
    zero = read(bound(c['plans']['zero']))
    with bound(c['history_packet']).open('rb') as stream:
        history = pickle.load(stream)['history']
    baseline = read(bound(c['historical_baseline_result']))
    a = baseline['selection_candidates'][c['competition']['selected_index']]
    b = baseline['selection_candidates'][c['competition']['alternative_index']]
    coarse, fine = canonical(family(a)), canonical(family(b))
    added = set(fine)-set(coarse)
    if len(added) != 1: raise ValueError('one explicit historical disputed addition required')
    target = next(iter(added))
    unique = {}
    for index, point in enumerate(baseline['selection_candidates']):
        if point.get('preprocess', 'raw') != 'raw':
            raise ValueError('do not mix raw and AC/demeaned validation scores')
        key = canonical(family(point))
        if key not in unique:
            rr, xx = matrices(len(c['labels']), point['supports'], point['r'], point['x'])
            unique[key] = dict(family_id=len(unique), requested=key,
                has_target=target in {tuple(s) for s in point['supports']},
                requested_has_target=target in key,
                source_indices=[], historical_r=rr, historical_x=xx)
        unique[key]['source_indices'].append(index)
    families = list(unique.values())
    if not any(f['has_target'] for f in families) or all(f['has_target'] for f in families):
        raise ValueError('both historical structural explanations are needed')
    r_h, x_h = matrices(len(c['labels']), b['supports'], b['r'], b['x'])
    days = []
    for day in history['training']+history['validation']:
        observed, _, _ = normalize([day], 'historical_day'); days.append(arrays(observed))
    errors = np.stack([v[2]-v[0]@r_h.T-v[1]@x_h.T for v in days])
    bias = errors.mean(axis=0)
    observed, _, _ = normalize(history['training'], 'past_train'); past_train = arrays(observed)
    observed, _, _ = normalize(history['validation'], 'past_valid'); past_valid = arrays(observed)
    n = len(c['labels'])
    # Original catalogue: H is P-then-Q, time-major inside each event block.
    h = np.asarray(problem.profiles)
    hp = np.stack([h[e*2*96*n:e*2*96*n+96*n].reshape(96,n,16) for e in range(4)])
    hq = np.stack([h[e*2*96*n+96*n:(e+1)*2*96*n].reshape(96,n,16) for e in range(4)])
    response = []
    for event in range(4):
        d = np.zeros((16,2))
        if event in (0,2):
            start = event//2*8
            d[start:start+8] = np.asarray(zero['response_factors_by_event'][event])
        response.append(d)
    return dict(source=c, problem=problem, families=families,
        pool=canonical([s for f in families for s in f['requested']]),
        coarse=coarse, fine=fine, target=target,
        r_h=r_h, x_h=x_h, days=days, bias=bias, centered=errors-bias,
        past_train=past_train, past_valid=past_valid, hp=hp, hq=hq,
        response=np.stack(response), original_selection_points=len(baseline['selection_candidates']),
        historical_acceptance=a['objective']-b['objective']-1e-10-.001*a['objective'])


def draw_scenarios(ctx, samples, seed):
    rng = np.random.default_rng(seed); result = []
    for scenario in range(samples):
        day_ids = rng.integers(0,len(ctx['days']),4)
        error_ids = rng.integers(0,len(ctx['days']),4)
        signs = rng.choice([-1.,1.],4)
        z = rng.uniform(-1,1,(4,2))
        days = []
        for event in range(4):
            p,q,_ = ctx['days'][day_ids[event]]
            error = ctx['bias']+signs[event]*ctx['centered'][error_ids[event]]
            y = p@ctx['r_h'].T+q@ctx['x_h'].T+error
            days.append((p,q,y))
        result.append(dict(scenario=scenario, day_ids=day_ids.tolist(), error_ids=error_ids.tolist(),
            signs=signs.tolist(), factors=z.tolist(), days=days,
            multiplier=1+np.einsum('ejk,ek->ej',ctx['response'],z)))
    return result


def event_data(ctx, scenario, event, amplitudes):
    actual = np.asarray(amplitudes)*scenario['multiplier'][event]
    dp = np.einsum('tnj,j->tn',ctx['hp'][event],actual)/ctx['source']['base_kw']
    dq = np.einsum('tnj,j->tn',ctx['hq'][event],actual)/ctx['source']['base_kw']
    p,q,y = scenario['days'][event]
    return p+dp,q+dq,y+dp@ctx['r_h'].T+dq@ctx['x_h'].T


def training_data(ctx, scenario, amplitudes):
    future = [event_data(ctx,scenario,e,amplitudes) for e in (0,1)]
    return tuple(np.vstack((ctx['past_train'][k],future[0][k],future[1][k])) for k in range(3))


def fit_families(ctx, scenario, train_amplitudes, *, historical=False):
    if historical:
        return [{**f, 'r_matrix':f['historical_r'], 'x_matrix':f['historical_x']}
                for f in ctx['families']], float(ctx['historical_acceptance']), []
    lp = PoolLP(None,ctx['pool'],data=training_data(ctx,scenario,train_amplitudes),lp_method=2)
    fits = []; objectives = {}
    try:
        for f in ctx['families']:
            point = lp.fit(f['requested'])
            rr,xx = matrices(len(ctx['source']['labels']),point['supports'],point['r'],point['x'])
            objectives[f['requested']] = point['objective']
            fits.append({**f,'has_target':ctx['target'] in {tuple(s) for s in point['supports']},
                'r_matrix':rr,'x_matrix':xx})
        j0,j1 = objectives[ctx['coarse']],objectives[ctx['fine']]
        return fits,float(j0-j1-1e-10-.001*j0),list(lp.solves)
    finally: lp.close()


def affine_tables(ctx, scenarios, train_amplitudes, *, historical=False, progress=None):
    """Weights may depend on training quotes; never on validation quotes/data."""
    tables = []; solves = []
    n = len(ctx['source']['labels']); size=(ctx['past_valid'][0].shape[0]+192)*n
    for scenario in scenarios:
        fits,gain,called = fit_families(ctx,scenario,train_amplitudes,historical=historical)
        solves.extend(called); entries=[]
        hp,hq = ctx['hp'][2][:,:,8:],ctx['hq'][2][:,:,8:]
        multiplier = scenario['multiplier'][2,8:]*ctx['problem'].elasticities[8:]/ctx['source']['base_kw']
        for f in fits:
            rr,xx=f['r_matrix'],f['x_matrix']
            p,q,y=ctx['past_valid']
            const=float(np.abs(p@rr.T+q@xx.T-y).sum())
            p,q,y=scenario['days'][3]
            const+=float(np.abs(p@rr.T+q@xx.T-y).sum())
            p,q,y=scenario['days'][2]
            base=(p@rr.T+q@xx.T-y).ravel()
            # Derivative of this frozen predictor MINUS reference observation.
            slopes=(np.einsum('tkj,ik->tij',hp,rr-ctx['r_h'])+
                 np.einsum('tkj,ik->tij',hq,xx-ctx['x_h']))*multiplier
            slopes=slopes.reshape(96*n,8)
            dynamic=np.any(slopes!=0,axis=1)
            const+=float(np.abs(base[~dynamic]).sum())
            entries.append(dict(family_id=f['family_id'],has_target=bool(f['has_target']),
                constant_loss=const/size,residual_base=base[dynamic],residual_price=slopes[dynamic],
                training_weights_scope='historical_frozen' if historical else 'training_refit_then_validation_frozen'))
        available=any(e['has_target'] for e in entries) and any(not e['has_target'] for e in entries)
        tables.append(dict(scenario=scenario['scenario'],families=entries,
            comparison_available=available,acceptance_margin=gain,validation_size=size))
        if progress: progress(dict(scenarios_completed=len(tables),scenarios_planned=len(scenarios),training_LP_calls=len(solves)))
    return tables,solves


def evaluate_tables(problem,tables,amplitudes):
    prices=np.asarray(amplitudes)/problem.elasticities; rows=[]
    for table in tables:
        losses={e['family_id']:float(e['constant_loss']+
            np.abs(e['residual_base']+e['residual_price']@prices[8:]).sum()/table['validation_size'])
            for e in table['families']}
        no_values=[losses[e['family_id']] for e in table['families'] if not e['has_target']]
        yes_values=[losses[e['family_id']] for e in table['families'] if e['has_target']]
        no=min(no_values) if no_values else None
        yes=min(yes_values) if yes_values else None
        margin=no-1.03*yes-1e-12 if no is not None and yes is not None else None
        rows.append(dict(scenario=table['scenario'],no_target_validation=no,with_target_validation=yes,
            margin=margin,acceptance_margin=table['acceptance_margin'],
            comparison_available=margin is not None,
            excitation_pass=bool(margin is not None and margin>1e-12 and table['acceptance_margin']>1e-12),family_losses=losses))
    required=int(np.ceil(.9*len(rows)))
    eligible=sorted(r['margin'] for r in rows if r['comparison_available'] and r['acceptance_margin']>1e-12)
    gamma=eligible[len(eligible)-required] if len(eligible)>=required else None
    return dict(rows=rows,passes=sum(r['excitation_pass'] for r in rows),denominator=len(rows),
        quantile_margin=gamma,
        criterion='all listed no-C and with-C families; original 3% plus original acceptance; no validation refit')


def _snapshot(output, sources):
    with zipfile.ZipFile(output/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for index,p in enumerate(sorted(sources)):
            if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sources[p]:
                raise RuntimeError('source drift before snapshot')
            archive.write(p,arcname=f'{index:03d}_'+Path(p).name)


def plan(source,output, *, max_outer_evaluations=14, inner_seconds=30.):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    sources=source_hashes()
    sources.update({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    c=read(Path(source)/'main/protocol.json')
    protocol=dict(status='frozen',id='conditional_validation_continuous_price_v1',created_utc=now(),
        source_root=str(Path(source).resolve()),history=c['history_packet'],historical_fit=c['historical_baseline_result'],
        problem=c['problem'],budget=1.6,planning_samples=10,planning_seed=2026110131,
        model_holdout_samples=10,model_holdout_seed=2026110132,probability=.9,
        search=dict(training_indices=[2,6],outer='Powell bounded continuous, exact inner finite-scenario MISOCP',
            max_outer_evaluations=max_outer_evaluations,inner_seconds=inner_seconds,
            training_type_scope='spread on each node, independent prices including zero',
            validation_type_scope='all four original types per node, optional zero',
            seeds_prices=[[0.,0.],[.8,.8],[1.6,0.],[0.,1.6]],
            gamma_units='micro pu^2',phaseI='maximize empirical90% validation-exclusion margin',
            phaseII='minimum original expected fee once positive or within primary optimizer tolerance'),
        event='original training acceptance and all historical no-C families outside original 3% band',
        parameter_rules=['historical frozen predictor diagnostic','complete-family training refit, then all validation weights frozen'],
        source_sha256=sources,no_test_future_or_truth=True,new_physical=0,
        actual_pilot_if_unmet='freeze best phaseI as explicitly infeasible diagnostic, plus baselines; no 90% guarantee',
        actual_paths=[1200,1201],actual_response_seed=2026110133,actual_replicate_base=470000)
    write(output/'protocol.json',protocol);_snapshot(output,sources)
    ctx=context(source);problem=ctx['problem'];scenarios=draw_scenarios(ctx,10,protocol['planning_seed'])
    write(output/'historical_family_manifest.json',dict(selection_points=ctx['original_selection_points'],
        distinct_complete_families=len(ctx['families']),target_labels=[c['labels'][i] for i in ctx['target']],
        families=[dict(family_id=f['family_id'],has_target=f['has_target'],requested=f['requested'],
            requested_has_target=f['requested_has_target'],
            source_indices=f['source_indices']) for f in ctx['families']]))
    write(output/'scenario_lock.json',dict(samples=[{k:v for k,v in s.items() if k not in ('days','multiplier')} for s in scenarios]))
    allzero=np.zeros(problem.n_contracts)
    # Genuine globally solved validation quote for the user's historical-freeze arm.
    fixed_dir=output/'historical_frozen';fixed_dir.mkdir()
    fixed_tables,lp_calls=affine_tables(ctx,scenarios,allzero,historical=True)
    assert not lp_calls
    with (fixed_dir/'private_tables.pkl').open('wb') as stream:pickle.dump(fixed_tables,stream,protocol=4)
    fixed=solve_affine(problem,fixed_tables,allzero,seconds=inner_seconds)
    write(fixed_dir/'optimization.json',fixed)
    if not fixed.get('amplitudes'):raise RuntimeError('historical-frozen optimization has no usable diagnostic incumbent')
    fixed_audit=audit_inner(problem,fixed_tables,allzero,fixed)
    write(fixed_dir/'independent_audit.json',fixed_audit)
    if not fixed_audit['audit_passed']:raise RuntimeError('historical frozen inner audit failed')
    fixed_check=evaluate_tables(problem,fixed_tables,fixed['amplitudes'])
    write(fixed_dir/'criterion_check.json',fixed_check)
    ledger=[];cache={}
    limit=int(max_outer_evaluations)
    def assess(prices):
        prices=np.asarray(prices,float)
        key=tuple(np.round(prices,10))
        if key in cache:return cache[key]
        if len(ledger)>=limit:raise RuntimeError('outer evaluation cap reached')
        index=len(ledger);folder=output/'outer'/f'eval_{index:03d}';folder.mkdir(parents=True)
        train=allzero.copy();train[[2,6]]=prices*problem.elasticities[[2,6]]
        row=dict(index=index,training_prices=prices.tolist(),status='started')
        ledger.append(row);write(output/'progress.json',dict(status='running',evaluations=ledger))
        try:
            tables,solves=affine_tables(ctx,scenarios,train,
                progress=lambda p:write(folder/'progress.json',p))
            with (folder/'private_tables.pkl').open('wb') as stream:pickle.dump(tables,stream,protocol=4)
            write(folder/'training_LP_receipts.json',dict(calls=len(solves),solves=solves))
            solved=solve_affine(problem,tables,train,seconds=inner_seconds)
            write(folder/'optimization.json',solved)
            if not solved.get('amplitudes'):raise RuntimeError('conditional inner has no feasible diagnostic incumbent')
            audit=audit_inner(problem,tables,train,solved)
            write(folder/'independent_audit.json',audit)
            if not audit['audit_passed']:raise RuntimeError('conditional inner exact replay audit failed')
            check=evaluate_tables(problem,tables,solved['amplitudes']);write(folder/'criterion_check.json',check)
            row.update(status='complete',training_LP_calls=len(solves),amplitudes=solved['amplitudes'],
                prices=solved['prices'],quantile_margin=check['quantile_margin'],passes=check['passes'],
                denominator=check['denominator'],primary_status=solved.get('primary',{}).get('status_code'),
                expected_cost=float(problem.expected_cost_coefficients@np.square(solved['amplitudes'])),
                folder=str(folder),optimization=ref(folder/'optimization.json'))
            value=-check['quantile_margin']*1e6
        except Exception as e:
            row.update(status='failed',error_type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
            value=1e6
        write(folder/'ended.json',row);write(output/'progress.json',dict(status='running',evaluations=ledger))
        print(json.dumps(dict(stage='outer',index=index,training_prices=prices.tolist(),status=row['status'],
            quantile_margin=row.get('quantile_margin'),passes=row.get('passes'))),flush=True)
        cache[key]=value
        return value
    for start in protocol['search']['seeds_prices']:assess(start)
    completed=[r for r in ledger if r['status']=='complete']
    if not completed:raise RuntimeError('all conditional continuous searches failed')
    start=min(completed,key=lambda r:(-r['quantile_margin'],r['expected_cost'],r['index']))['training_prices']
    remaining=limit-len(ledger)
    outer_result=minimize(assess,start,method='Powell',bounds=[(0.,5/3),(0.,5/3)],
        options=dict(maxfev=remaining,maxiter=2,xtol=1e-3,ftol=1e-7))
    completed=[r for r in ledger if r['status']=='complete']
    best=min(completed,key=lambda r:(-r['quantile_margin'],r['expected_cost'],r['index']))
    rotation=read(bound(c['plans']['rotation']))['global_expected_amplitudes']
    quotes=dict(zero=allzero.tolist(),rotation=rotation,
        continuous_validation=best['amplitudes'],historical_frozen=fixed['amplitudes'])
    decision=dict(status='frozen',created_utc=now(),quotes=quotes,
        selected_outer_index=best['index'],model_90_sample_feasible=best['passes']>=9,
        quote_status='candidate_needs_independent_validation' if best['passes']>=9 else 'diagnostic_phaseI_not90_feasible',
        criterion=best,original_3pct_unchanged=True,budget=1.6,
        selected_without_actual_future=True,global_outer_optimality=False,
        fixed_historical_scope='all historical parameters stay fixed; not original joint-training-update rule')
    write(output/'price_selection.json',decision)
    summary=dict(status='complete',evaluations=ledger,completed=len(completed),planned_max=limit,
        failed=sum(r['status']!='complete' for r in ledger),
        total_training_LP_calls=sum(r.get('training_LP_calls',0) for r in ledger),
        old_quotes_only=False,new_continuous_quotes=True,price_selection=ref(output/'price_selection.json'),
        frozen_arm_quantile=fixed_check['quantile_margin'],refit_arm_quantile=best['quantile_margin'],
        selected_refit_sample90_feasible=best['passes']>=9,
        probability_scope='finite conditional scenario excitation, not topology correctness or field calibration',
        outer_solver=dict(success=bool(outer_result.success),status=int(outer_result.status),
            message=str(outer_result.message),nfev=int(outer_result.nfev)))
    write(output/'summary.json',summary);write(output/'ended.json',dict(status='complete',summary=ref(output/'summary.json')))
    return summary


def check_model_holdout(source,planning_root,output):
    planning_root,output=Path(planning_root),Path(output);output.mkdir(parents=True,exist_ok=False)
    lock=ref(planning_root/'price_selection.json');decision=read(bound(lock))
    ctx=context(source);protocol=read(planning_root/'protocol.json')
    write(output/'protocol.json',dict(status='frozen',price_selection=lock,
        samples=protocol['model_holdout_samples'],seed=protocol['model_holdout_seed'],
        scope='independent conditional-model whole paths, not actual AC or field validation'))
    scenarios=draw_scenarios(ctx,protocol['model_holdout_samples'],protocol['model_holdout_seed'])
    rows=[];solves=[];cache={}
    for policy,amplitudes in decision['quotes'].items():
        amp=np.asarray(amplitudes);train=amp.copy();train[8:]=0.
        key=tuple(train)
        if key not in cache:
            tables,called=affine_tables(ctx,scenarios,train);cache[key]=tables;solves.extend(called)
        table=cache[key];checked=evaluate_tables(ctx['problem'],table,amp)
        raw=evaluate_plan(ctx['problem'],amp)
        # The historical-frozen quote also gets the ORIGINAL training update.
        rr=dict(policy=policy,**checked,expected_cost=raw['expected_cost'],hard_budget_bound=raw['hard_budget_bound'],
            original_feasible=raw['numerical_feasible'])
        if policy=='historical_frozen':
            frozen_tables,_=affine_tables(ctx,scenarios,train,historical=True)
            rr['own_frozen_predictor_check']=evaluate_tables(ctx['problem'],frozen_tables,amp)
        rows.append(rr);write(output/'progress.json',dict(status='running',rows=rows))
    summary=dict(status='complete',rows=rows,training_LP_calls=len(solves),
        distinct_training_quotes=len(cache),scenario_count=len(scenarios),new_actual_future=0,new_AC=0,
        price_selection=lock,selection_updated_after_holdout=False)
    write(output/'summary.json',summary);write(output/'ended.json',dict(status='complete'))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['plan','holdout'])
    p.add_argument('--source',type=Path,default=DEFAULT_SOURCE);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--planning-root',type=Path);p.add_argument('--max-outer-evaluations',type=int,default=14)
    p.add_argument('--inner-seconds',type=float,default=30.)
    a=p.parse_args()
    for directory in (REPO,REPO/'rnj_wzzt_core'):
        if str(directory) not in sys.path:sys.path.insert(0,str(directory))
    if a.phase=='plan':plan(a.source,a.output,max_outer_evaluations=a.max_outer_evaluations,inner_seconds=a.inner_seconds)
    else:check_model_holdout(a.source,a.planning_root,a.output)


if __name__=='__main__':main()
