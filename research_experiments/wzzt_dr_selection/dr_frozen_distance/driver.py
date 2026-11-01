"""One-day stochastic DR test with historical predictors frozen end to end."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import traceback
import zipfile

for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
import numpy as np

from .frozen_pool import freeze_pool,predict,score_frozen
from .design import build_design,draw_scenarios,plan_payload
from .distance_solver import solve_distance,evaluate_distance
from research_experiments.wzzt_dr_selection.dr_conic_pricing.effect_test.prepare import _sources
from research_experiments.wzzt_dr_selection.dr_conic_pricing.model import evaluate_plan

REPO=Path(__file__).resolve().parents[3]
DEFAULT_CASE=Path(r'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_conic_effects_v1/run_v1/main/protocol.json')


def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(r):
    p=Path(r['path'])
    if sha(p)!=r['sha256']:raise RuntimeError('frozen input/source drift: '+str(p))
    return p


def config(root):
    root=Path(root).resolve();c=read(root/'protocol.json')
    if c['status']!='frozen':raise RuntimeError('unfrozen protocol')
    for p,h in c['source_sha256'].items():
        if sha(p)!=h:raise RuntimeError('source drift: '+p)
    checked(c['frozen_pool']);checked(c['history_packet'])
    return root,c


def prepare(case_protocol,root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    old=read(case_protocol)
    pool=freeze_pool(case_protocol,root/'frozen_pool.json')
    with checked(pool['history_packet']).open('rb') as stream:history=pickle.load(stream)['history']
    design=build_design(pool,history,units=5);problem=design['problem']
    write(root/'problem.json',problem.to_dict());write(root/'resource_qualification.json',design['qualification'])
    write(root/'guided_shapes.json',design['guided_shape_audit'])
    extras=[str(p.resolve()) for p in Path(__file__).parent.glob('*.py')]
    extras += [str(REPO/'research_experiments/wzzt_dr_selection/dr_conic_pricing/resources.py'),
        str(REPO/'research_experiments/wzzt_dr_selection/dr_conic_pricing/model.py')]
    sources,algorithm=_sources(REPO,[dict(source_root=old['source_root'])],extras)
    c=dict(status='frozen',id='frozen_dispute_distance_one_day_v1',created_utc=now(),repository=str(REPO),
        case=old['case'],old_case_protocol=ref(case_protocol),frozen_pool=ref(root/'frozen_pool.json'),
        history_packet=pool['history_packet'],source_root=old['source_root'],source_metadata=old['source_metadata'],
        labels=pool['labels'],root_label=pool['root_label'],base_kw=pool['base_kw'],problem=ref(root/'problem.json'),
        source_sha256=sources,algorithm_source_sha256=algorithm,
        voltage_sigma=.0005,pq_sigma=.005,budget=1.6,aggregate_units=5,
        original_history_rows=[288,288],new_training_rows=0,new_validation_rows=96,cumulative_validation_rows=384,
        policies=['zero','rotation','optimized'],planning_samples=64,planning_seed=2026110151,
        model_holdout_samples=1000,model_holdout_seed=2026110152,
        actual_paths=[1400,1401,1402,1403],actual_profiles=[0,1,2,0],actual_response_seed=2026110153,
        actual_replicate_base=510000,solver_seconds=60.,
        event='90% same-background increase in exact frozen-predictor L1 distance',
        validation_rule='historical_validation plus one new day, original3%/support count/order; historical.16 output frozen',
        parameter_rule='historical pool/R/X/order/calibration frozen; no new fit, new search or new candidate',
        candidate_scope='historical selected point plus its next accepted structural competitor only',
        resource_scope='5x original synthetic0.2kW units per qualified node, embedded existing component; same fleet all policies',
        budget_intensity_note='same1.6 total cap, now concentrated onone day instead ofoldfour events; not algorithm-only old/new comparison',
        scenario_scope='uniform six past complete P/Q templates; shared two-factor continuous response; no empirical behavior calibration',
        truth_boundary='no true network or new physical future before price freeze; truth only after frozen scoring',
        expected_actual_records=12)
    write(root/'protocol.json',c)
    with zipfile.ZipFile(root/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for i,p in enumerate(sorted(sources)):z.write(p,arcname=f'{i:03d}_'+Path(p).name)
    scenarios,draws=draw_scenarios(design,c['planning_samples'],c['planning_seed'])
    write(root/'planning_draws.json',dict(status='frozen',draws=draws))
    with (root/'private_planning_scenarios.pkl').open('wb') as stream:pickle.dump(scenarios,stream,protocol=4)
    solved=solve_distance(problem,scenarios,seconds=c['solver_seconds'],probability=.9)
    write(root/'optimization.json',solved)
    if not solved.get('feasible_incumbent'):raise RuntimeError('no verified feasible distance quote; retain failed solve')
    rotation=np.zeros(problem.n_contracts);active=[0,6] # focus early, partner late
    caps=problem.amplitude_upper[active]/problem.elasticities[active]
    beta=problem.reduction_hours[active]*problem.elasticities[active]*(1+problem.q[active])
    price=min(float(caps.min()),float(np.sqrt(problem.budget/beta.sum())))
    rotation[active]=price*problem.elasticities[active]
    values=dict(zero=np.zeros(problem.n_contracts),rotation=rotation,optimized=np.asarray(solved['amplitudes']))
    plans={};summary=[]
    for policy,x in values.items():
        payload=plan_payload(design,x,policy);file=root/'plans'/(policy+'.json');write(file,payload);plans[policy]=ref(file)
        replay=evaluate_distance(scenarios,payload['prices'],.9)
        summary.append(dict(policy=policy,prices=payload['prices'],amplitudes=x.tolist(),
            expected_payment=payload['expected_payment'],hard_payment_bound=payload['hard_payment_bound'],
            quantile_micro=replay['quantile_micro'],positive_passes=replay['positive_pass_count'],
            denominator=c['planning_samples']))
    write(root/'planning_summary.json',dict(status='complete',table=summary,
        optimized_gamma_micro=solved['quantile_micro'],primary=solved['primary'],secondary=solved['secondary'],
        zero_new_parameter_fits=True,no_true_topology_used=True))
    freeze=dict(status='frozen',created_utc=now(),protocol=ref(root/'protocol.json'),frozen_pool=c['frozen_pool'],
        problem=c['problem'],plans=plans,source_snapshot=ref(root/'source_snapshot.zip'),
        optimization=ref(root/'optimization.json'),planning_summary=ref(root/'planning_summary.json'),
        gamma_native=solved['quantile_micro']/1e6,positive_gamma=solved['quantile_micro']>1e-6,
        models_and_prices_fixed_before_model_holdout=True,models_and_prices_fixed_before_actual=True,
        no_test_truth_or_future_read=True)
    write(root/'price_freeze.json',freeze)
    write(root/'preparation_ended.json',dict(status='complete',price_freeze=ref(root/'price_freeze.json'),
        source_snapshot=ref(root/'source_snapshot.zip'),new_parameter_fits=0,new_AC=0))
    print(json.dumps(dict(stage='prepared',pool_size=2,resources=design['qualification']['selected_labels'],
        gamma_micro=solved['quantile_micro'],prices=solved['prices'],primary_status=solved['primary']['status_code'])),flush=True)
    return freeze


def model_holdout(root):
    root,c=config(root);freeze=read(root/'price_freeze.json');checked(freeze['protocol'])
    output=root/'model_holdout';output.mkdir(exist_ok=False)
    pool=read(checked(c['frozen_pool']))
    with checked(c['history_packet']).open('rb') as stream:history=pickle.load(stream)['history']
    design=build_design(pool,history,units=c['aggregate_units'])
    rng=np.random.default_rng(c['model_holdout_seed']);price={p:np.asarray(read(checked(freeze['plans'][p]))['prices']) for p in c['policies']}
    values={p:[] for p in c['policies']};gamma=freeze['gamma_native'];N=96*len(c['labels'])
    for s in range(c['model_holdout_samples']):
        day=int(rng.integers(6));z=rng.uniform(-1,1,2)
        coeff=design['delta_columns']*(design['problem'].elasticities*(1+design['problem'].response_factors@z))
        base=design['base_delta'][day];d0=float(np.abs(base).sum()/N)
        for p,prices in price.items():values[p].append(float(np.abs(base+coeff@prices).sum()/N-d0))
    table=[]
    for p,gain in values.items():
        gain=np.asarray(gain)
        table.append(dict(policy=p,denominator=len(gain),positive_gain_count=int((gain>1e-12).sum()),
            target_gain_count=int((gain>=gamma-1e-12).sum()) if freeze['positive_gamma'] else None,
            target_gamma_native=gamma,empirical_target_rate=float((gain>=gamma-1e-12).mean()) if freeze['positive_gamma'] else None,
            gain_min=float(gain.min()),gain_mean=float(gain.mean()),gain_p10=float(np.quantile(gain,.1)),gain_max=float(gain.max())))
    write(output/'summary.json',dict(status='complete',seed=c['model_holdout_seed'],table=table,
        price_freeze=ref(root/'price_freeze.json'),scope='independent working-law scenarios, not physical recovery probability',new_fit_calls=0))
    print(json.dumps(dict(stage='model_holdout',table=table)),flush=True)


def observe(root):
    root,c=config(root);freeze=read(root/'price_freeze.json')
    checked(freeze['protocol']);checked(freeze['source_snapshot'])
    for item in freeze['plans'].values():checked(item)
    output=root/'actual';output.mkdir(exist_ok=False);rows=[]
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join((str(REPO),str(REPO/'rnj_wzzt_core')))
    for position,path in enumerate(c['actual_paths']):
        factors=np.random.default_rng(np.random.SeedSequence([c['actual_response_seed'],path])).uniform(-1,1,2).tolist()
        for policy in c['policies']:
            identity=f'{policy}__path{path}';record=output/'records'/identity;record.mkdir(parents=True)
            request=dict(status='frozen',phase='test',case=c['case'],record_id=identity,policy=policy,path=path,
                replicate=c['actual_replicate_base']+10*path,profile_index=c['actual_profiles'][position],day_offset=6,
                labels=c['labels'],root_label=c['root_label'],base_kw=c['base_kw'],factors=factors,
                voltage_sigma=c['voltage_sigma'],pq_sigma=c['pq_sigma'],budget=1.6,
                protocol=ref(root/'protocol.json'),plan=freeze['plans'][policy],frozen_pool=c['frozen_pool'],
                source_metadata=c['source_metadata'],repo_root=str(REPO),source_root=c['source_root'],source_sha256=c['source_sha256'])
            file=record/'request.json';write(file,request)
            row=dict(record_id=identity,policy=policy,path=path,status='started',request=ref(file),physical_output=str(record/'physical'))
            try:
                command=[sys.executable,'-u','-B','-m','research_experiments.wzzt_dr_selection.dr_frozen_distance.one_day_worker',
                    '--request',str(file),'--expected-sha',sha(file),'--output',str(record/'physical')]
                with (record/'physical.log').open('w',encoding='utf-8') as log:
                    process=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
                receipt=read(record/'physical/events_receipt.json')
                row.update(status='observed' if receipt['status']=='complete' else 'physical_failed',worker_status=receipt['status'],
                    worker_exit_code=process.returncode,receipt=ref(record/'physical/events_receipt.json'),
                    actual_payment=receipt.get('actual_payment'),worst_payment=receipt.get('worst_payment'),
                    packet=ref(record/'physical/packet.pkl') if (record/'physical/packet.pkl').exists() else None)
            except Exception as error:
                row.update(status='physical_failed',error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            rows.append(row);write(output/'progress.json',dict(status='running',rows=rows,planned=12))
            print(json.dumps(dict(stage='physical',done=len(rows),planned=12,policy=policy,path=path,status=row['status'])),flush=True)
    write(output/'observations_frozen.json',dict(status='frozen',rows=rows,planned=12,created_utc=now(),
        price_freeze=ref(root/'price_freeze.json'),new_fit_calls=0,controller_truth_read=False))


def score(root):
    root,c=config(root);output=root/'actual';ledger=read(output/'observations_frozen.json');pool=read(checked(c['frozen_pool']))
    if (output/'scored.json').exists():raise FileExistsError('retain original scored results')
    projections=[]
    for original in ledger['rows']:
        row=dict(original)
        if row['status']=='observed':
            with checked(row['packet']).open('rb') as stream:packet=pickle.load(stream)
            if len(packet['validation'])!=1 or len(packet['validation'][0]['P_terminal'])!=96:
                raise ValueError('exactly one new validation day required')
            result=score_frozen(pool,packet['validation'][0])
            file=output/'scores'/f"{row['record_id']}.json";write(file,result)
            row.update(status='predictions_frozen',scores=ref(file),new_distance=result['new_prediction_pair_distance_l1'])
        projections.append(row)
    write(output/'predictions_frozen.json',dict(status='frozen',created_utc=now(),rows=projections,
        frozen_pool=c['frozen_pool'],observations_frozen=ref(output/'observations_frozen.json'),
        no_true_structure_read_by_controller=True,new_fit_calls=0,new_candidate_generation_calls=0))
    # Only after all frozen prediction/selection files exist may truth enter.
    rows=[]
    for original in projections:
        row=dict(original)
        if row['status']=='predictions_frozen':
            result=read(checked(row['scores']));truth=read(Path(row['physical_output'])/'truth_evaluation.json')['true_supports']
            true={tuple(sorted(s)) for s in truth}
            def errors(blocks):
                blocks={tuple(sorted(s)) for s in blocks}
                return dict(full_errors=len(blocks^true),missing=[list(s) for s in sorted(true-blocks)],extra=[list(s) for s in sorted(blocks-true)])
            candidate_errors={m['candidate_id']:errors(m['frozen_output_support_labels']) for m in pool['models']}
            correct=[k for k,v in candidate_errors.items() if v['full_errors']==0]
            actual_losses={r['candidate_id']:r['cumulative_validation_mae'] for r in result['rows']}
            new_losses={r['candidate_id']:r['new_validation_mae'] for r in result['rows']}
            coverage=bool(correct);wrong=[k for k in candidate_errors if k not in correct]
            exclusion={k:actual_losses[k]-1.03*min(actual_losses.values())-1e-12 for k in wrong} if coverage else {}
            day_exclusion={k:new_losses[k]-1.03*min(new_losses.values())-1e-12 for k in wrong} if coverage else {}
            row.update(status='complete',candidate_structural_errors=candidate_errors,correct_candidate_ids=correct,
                true_structure_in_frozen_pool=coverage,**errors(result['selected_output_support_labels']),
                selected_candidate=result['selected_candidate_id'],
                day_only_errors=errors(result['day_only_selection']['selected_output_support_labels']),
                cumulative_wrong_exit_margins=exclusion,day_only_wrong_exit_margins=day_exclusion,
                wrong_model_excluded=bool(exclusion and all(x>0 for x in exclusion.values())),
                day_only_wrong_model_excluded=bool(day_exclusion and all(x>0 for x in day_exclusion.values())),
                cumulative_pair_margin=result['cumulative_pair_exclusion_margin'],
                day_only_pair_margin=result['day_only_pair_exclusion_margin'],parameters_changed=False,
                cumulative_validation_rows=result['cumulative_validation_rows'])
        rows.append(row)
    freeze=read(root/'price_freeze.json');gamma=freeze['gamma_native']
    for row in rows:
        baseline=next(r for r in rows if r['path']==row['path'] and r['policy']=='zero')
        if row['status']==baseline['status']=='complete':
            gain=row['new_distance']-baseline['new_distance']
            row.update(distance_gain=gain,distance_positive=gain>1e-12,
                distance_target_met=bool(gain>=gamma-1e-12) if freeze['positive_gamma'] else None,
                final_error_improvement=baseline['full_errors']-row['full_errors'])
    write(output/'scored.json',dict(status='complete',rows=rows,planned=12,
        predictions_frozen=ref(output/'predictions_frozen.json'),frozen_pool=c['frozen_pool'],
        new_parameter_fits=0,new_pool_regenerations=0))
    table=[]
    for policy in c['policies']:
        group=[r for r in rows if r['policy']==policy];ok=[r for r in group if r['status']=='complete']
        table.append(dict(policy=policy,denominator=4,complete=len(ok),unresolved=4-len(ok),
            full_errors=[r.get('full_errors') for r in group],recoveries=sum(r['full_errors']==0 for r in ok),
            wrong_excluded=sum(r['wrong_model_excluded'] for r in ok),day_only_wrong_excluded=sum(r['day_only_wrong_model_excluded'] for r in ok),
            mean_payment=sum(r['actual_payment'] for r in ok)/4 if len(ok)==4 else None,
            distance_gains=[r.get('distance_gain') for r in group],
            positive_distance_count=sum(r.get('distance_positive',False) for r in ok),
            target_distance_count=sum(r.get('distance_target_met') is True for r in ok)))
    summary=dict(status='complete',table=table,rows=rows,independent_actual_days=4,requested_records=12,
        completed=sum(r['status']=='complete' for r in rows),new_parameter_fits=0,new_search_calls=0,
        source_pool_size=2,history_training_rows=288,history_validation_rows=288,new_validation_rows=96,
        cumulative_validation_rows=384,actual_AC_time_points=sum(96 for r in rows if r['status']=='complete'),
        price_freeze=ref(root/'price_freeze.json'),gamma_native=gamma,
        probability_scope='90% prediction-distance-gain event only; four physical days do not certify population probability',
        no_true_topology_in_price_design=True)
    write(root/'summary.json',summary);write(root/'ended.json',dict(status='complete',summary=ref(root/'summary.json')))
    print(json.dumps(dict(stage='scored',table=table)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','model_holdout','observe','score'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--case-protocol',type=Path,default=DEFAULT_CASE)
    a=p.parse_args()
    for folder in (REPO,REPO/'rnj_wzzt_core'):
        if str(folder) not in sys.path:sys.path.insert(0,str(folder))
    if a.phase=='prepare':prepare(a.case_protocol,a.root)
    elif a.phase=='model_holdout':model_holdout(a.root)
    elif a.phase=='observe':observe(a.root)
    else:score(a.root)


if __name__=='__main__':main()
