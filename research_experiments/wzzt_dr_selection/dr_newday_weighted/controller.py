"""Five predeclared historical conditions; frozen prices and new-day rules."""
from __future__ import annotations
import argparse,os,sys,pickle,subprocess,traceback,zipfile,math
from pathlib import Path
from datetime import datetime,timezone
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from .case_inputs import build_cases
from .design_adapter import choose_design
from .weighted_selection import score as weighted_score
from .pool_competition import choose_final_competition
from research_experiments.wzzt_dr_selection.dr_frozen_distance.driver import read,write,ref,checked,sha
from research_experiments.wzzt_dr_selection.dr_frozen_distance.frozen_pool import freeze_pool
from research_experiments.wzzt_dr_selection.dr_frozen_distance.design import draw_scenarios,plan_payload,warm_quotes
from research_experiments.wzzt_dr_selection.dr_frozen_distance.distance_solver import solve_distance,evaluate_distance
from research_experiments.wzzt_dr_selection.dr_conic_pricing.effect_test.prepare import _sources

REPO=Path(__file__).resolve().parents[3]
POLICIES=['zero','rotation','optimized']
def now():return datetime.now(timezone.utc).isoformat()
def load_pickle(item):
    with checked(item).open('rb') as stream:return pickle.load(stream)
def save_pickle(path,obj):
    Path(path).write_bytes(pickle.dumps(obj,protocol=4))
def historical_eligible(pool):
    loss={m['candidate_id']:m['historical_validation_mae'] for m in pool['models']};best=min(loss.values())
    return [k for k in pool['selection_order'] if loss[k]<=1.03*best+1e-12]


def config(root):
    root=Path(root).resolve();p=read(root/'protocol.json')
    if p['status']!='frozen':raise ValueError('unfrozen protocol')
    for file,h in p['source_sha256'].items():
        if sha(file)!=h:raise RuntimeError('source drift: '+file)
    end=read(root/'preparation_ended.json');f=read(checked(end['price_freeze']))
    checked(f['protocol']);checked(f['source_snapshot'])
    for c in p['cases']:
        for k in ['case_protocol','pool','runtime_protocol']:
            checked(c[k])
        if c['status']=='qualified':
            checked(c['problem']);checked(c['design']);checked(c['qualification'])
            for item in f['cases'][c['name']]['plans'].values():checked(item)
    return root,p,f


def prepare(root,reuse_root=None):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    inputs=build_cases();cases=[]
    for i,c in enumerate(inputs):
        folder=root/'cases'/c['name'];folder.mkdir(parents=True)
        write(folder/'case_protocol.json',c)
        pool=freeze_pool(folder/'case_protocol.json',folder/'frozen_pool.json',competition_selector=choose_final_competition)
        history=load_pickle(c['history_packet'])['history']
        item=dict(name=c['name'],index=i,case=c['case'],status='qualified',
            case_protocol=ref(folder/'case_protocol.json'),pool=ref(folder/'frozen_pool.json'),
            history_eligible_ids=historical_eligible(pool),
            final_output_distinct={tuple(sorted(s)) for s in pool['models'][0]['frozen_output_supports']}!={tuple(sorted(s)) for s in pool['models'][1]['frozen_output_supports']},
            no_historical_final_dispute=pool['competition_rule'].get('no_final_dispute',False),
            planning_seed=2026110171+10*i,holdout_seed=2026110172+10*i,response_seed=2026110173+10*i,
            paths=[1500+10*i+j for j in range(4)],profiles=[0,1,2,0])
        try:
            design=choose_design(pool,history)
            write(folder/'problem.json',design['problem'].to_dict())
            write(folder/'qualification.json',design['qualification'])
            write(folder/'guided_shapes.json',design['guided_shape_audit'])
            save_pickle(folder/'design.pkl',design)
            item.update(problem=ref(folder/'problem.json'),qualification=ref(folder/'qualification.json'),design=ref(folder/'design.pkl'))
        except Exception as error:
            item.update(status='qualification_failed',error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        cases.append(item)
    extras=[str(p.resolve()) for p in Path(__file__).parent.glob('*.py')]
    extras += [str(p.resolve()) for p in (REPO/'research_experiments/wzzt_dr_selection/dr_frozen_distance').glob('*.py')]
    sources,algorithm=_sources(REPO,[dict(source_root=c['source_root']) for c in inputs],extras)
    for item,c in zip(cases,inputs):
        folder=root/'cases'/item['name']
        nodes=read(checked(item['qualification']))['selected_labels'] if item['status']=='qualified' else []
        runtime=dict(status='frozen',name=c['name'],case=c['case'],source_sha256=sources,
            voltage_sigma=c['voltage_sigma'],pq_sigma=c['pq_sigma'],allowed_voltage_sigma=[c['voltage_sigma']],
            expected_resource_nodes=nodes,frozen_pool=item['pool'],labels=c['labels'],root_label=c['root_label'],base_kw=c['base_kw'],
            response_seed=item['response_seed'],paths=item['paths'],replicate_base=610000)
        write(folder/'runtime_protocol.json',runtime);item['runtime_protocol']=ref(folder/'runtime_protocol.json')
    p=dict(status='frozen',id='five_case_newday_weighted_v2',created_utc=now(),repository=str(REPO),cases=cases,
        source_sha256=sources,algorithm_source_sha256=algorithm,budget=1.6,planning_samples=64,holdout_samples=1000,
        solver_seconds=60.,replicate_base=610000,policies=POLICIES,primary_rule='w2_tau2pct',
        new_day_weights=[1,2,4],new_day_tolerances=[.03,.02,.01],historical_tolerance=.03,history_gate=True,
        original_ungated_baseline=True,full_history_rows=[288,288],new_validation_rows=96,
        actual_validation_rows=384,primary_effective_weight_rows=480,expected_records=60,
        inventory_rule='largest historically feasible common integer1..5 at a local dispute pair; all policies same inventory',
        case_rule='all five existing compatible histories from three networks; no filtering on new test truth or outcomes',
        objective='same-background frozen-predictor distance-gain scenario90% quantile; unchanged from preceding test',
        predecessor_scope='v1 historical raw-pair quotes retained but undeployed; three pairs had identical final structures, repaired before any actual/holdout',
        reuse_root=str(Path(reuse_root).resolve()) if reuse_root else None,
        probability_scope='90% refers to prediction-distance gain only; finite-scenario approximation and separate working-law holdout',
        truth_boundary='all quotes frozen before all physical observations; all selections frozen before truth labels',
        weight_scope='new-day validation sample weights only; all R/X and historical row weights fixed',
        new_day_rule='scoreS=Nh*vh+w*Nd*vd,referencej=minS,band=.03Nh*vhj+tau*wNd*vdj; original raw-support/S/order selection',
        old_result_scope='previous four-day rule replay is exploratory, not a new independent rule test')
    write(root/'protocol.json',p)
    with zipfile.ZipFile(root/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for i,file in enumerate(sorted(sources)):archive.write(file,arcname=f'{i:03d}_'+Path(file).name)
    frozen={};planning=[]
    for item,c in zip(cases,inputs):
        if item['status']!='qualified':
            planning.append(dict(condition=item['name'],status=item['status']));continue
        folder=root/'cases'/item['name'];design=load_pickle(item['design']);problem=design['problem']
        scenarios,draws=draw_scenarios(design,p['planning_samples'],item['planning_seed'])
        write(folder/'planning_draws.json',dict(status='frozen',draws=draws));save_pickle(folder/'planning_scenarios.pkl',scenarios)
        reused=None
        if not item['final_output_distinct'] or item['no_historical_final_dispute']:
            optimized=np.zeros(problem.n_contracts);gamma=0.;planning_status='no_final_structure_dispute_no_purchase'
            solved=dict(status=planning_status,feasible_incumbent=True,solver_calls=0,primary=None,secondary=None,prices=optimized.tolist(),quantile_micro=0.)
        elif len(item['history_eligible_ids'])==2:
            old_folder=Path(reuse_root)/'cases'/item['name'] if reuse_root else None
            if old_folder and (old_folder/'optimization.json').exists():
                old_pool=read(old_folder/'frozen_pool.json');new_pool=read(checked(item['pool']))
                same=all(old_pool['models'][j]['point']==new_pool['models'][j]['point'] and old_pool['models'][j]['R']==new_pool['models'][j]['R'] and old_pool['models'][j]['X']==new_pool['models'][j]['X'] for j in (0,1))
                if same and old_pool['calibration']==new_pool['calibration'] and sha(old_folder/'problem.json')==sha(folder/'problem.json') and sha(old_folder/'planning_scenarios.pkl')==sha(folder/'planning_scenarios.pkl'):
                    solved=read(old_folder/'optimization.json');reused=ref(old_folder/'optimization.json')
                else:solved=None
            else:solved=None
            if solved is None:
                solved=solve_distance(problem,scenarios,seconds=p['solver_seconds'],probability=.9,warm_start_prices=warm_quotes(problem))
            if solved.get('feasible_incumbent'):
                optimized=np.asarray(solved['amplitudes']);gamma=solved['quantile_micro']/1e6;planning_status='planned'
            else:optimized=np.zeros(problem.n_contracts);gamma=0.;planning_status='optimizer_failed_zero_fallback'
        else:
            optimized=np.zeros(problem.n_contracts);gamma=0.;planning_status='one_historical_eligible_no_purchase'
            solved=dict(status=planning_status,feasible_incumbent=True,solver_calls=0,primary=None,secondary=None,prices=optimized.tolist(),quantile_micro=0.)
        write(folder/'optimization.json',solved)
        rotation=np.zeros(problem.n_contracts);active=[0,6]
        cap=(problem.amplitude_upper/problem.elasticities)[active]
        beta=(problem.hard_cost_coefficients*problem.elasticities**2)[active]
        price=min(float(cap.min()),float(np.sqrt(problem.budget/beta.sum())))
        rotation[active]=price*problem.elasticities[active]
        plans={};table=[]
        for policy,x in dict(zero=np.zeros(problem.n_contracts),rotation=rotation,optimized=optimized).items():
            plan=plan_payload(design,x,policy);file=folder/(policy+'_plan.json');write(file,plan);plans[policy]=ref(file)
            replay=evaluate_distance(scenarios,plan['prices'],.9)
            table.append(dict(policy=policy,prices=plan['prices'],amplitudes=plan['amplitudes'],
                expected_payment=plan['expected_payment'],hard_payment_bound=plan['hard_payment_bound'],
                quantile_micro=replay['quantile_micro'],positive_passes=replay['positive_pass_count'],denominator=64))
        freeze=dict(plans=plans,gamma_native=gamma,positive_gamma=gamma>1e-12,planning_status=planning_status,
            optimization=ref(folder/'optimization.json'))
        if reused:freeze['reused_quote_from']=reused
        frozen[item['name']]=freeze
        result=dict(condition=item['name'],status=planning_status,table=table,
            history_eligible_ids=item['history_eligible_ids'],gamma_native=gamma,primary=solved.get('primary'),secondary=solved.get('secondary'))
        planning.append(result);write(root/'planning_progress.json',dict(status='running',cases=planning,planned=5))
        print(dict(stage='planned',condition=item['name'],status=planning_status,gamma_micro=gamma*1e6),flush=True)
    write(root/'planning_summary.json',dict(status='complete',cases=planning))
    f=dict(status='frozen',created_utc=now(),protocol=ref(root/'protocol.json'),source_snapshot=ref(root/'source_snapshot.zip'),
        cases=frozen,planning_summary=ref(root/'planning_summary.json'),no_actual_or_truth_read=True)
    write(root/'price_freeze.json',f);write(root/'preparation_ended.json',dict(status='complete',price_freeze=ref(root/'price_freeze.json')))


def holdout(root):
    root,p,f=config(root);out=root/'model_holdout';out.mkdir(exist_ok=False);result=[]
    for c in p['cases']:
        if c['status']!='qualified':result.append(dict(condition=c['name'],status=c['status']));continue
        design=load_pickle(c['design']);problem=design['problem'];gamma=f['cases'][c['name']]['gamma_native']
        price={k:np.asarray(read(checked(v))['prices']) for k,v in f['cases'][c['name']]['plans'].items()}
        rng=np.random.default_rng(c['holdout_seed']);values={k:[] for k in POLICIES};N=96*len(problem.metadata['node_labels'])
        for i in range(p['holdout_samples']):
            day=int(rng.integers(6));z=rng.uniform(-1,1,2)
            coeff=design['delta_columns']*(problem.elasticities*(1+problem.response_factors@z));base=design['base_delta'][day]
            for policy,quote in price.items():values[policy].append(float((np.abs(base+coeff@quote)-np.abs(base)).sum()/N))
        table=[]
        for policy,gain in values.items():
            gain=np.asarray(gain);table.append(dict(policy=policy,denominator=1000,positive_count=int((gain>1e-12).sum()),
                target_count=int((gain>=gamma-1e-12).sum()) if gamma>1e-12 else None,
                gain_mean=float(gain.mean()),gain_p10=float(np.quantile(gain,.1)),gamma_native=gamma))
        result.append(dict(condition=c['name'],status='complete',seed=c['holdout_seed'],table=table))
    write(out/'summary.json',dict(status='complete',cases=result,price_freeze=ref(root/'price_freeze.json'),new_fit_calls=0,new_AC=0))
    print(dict(stage='holdout_complete',conditions=len(result)),flush=True)


def observe(root):
    root,p,f=config(root);output=root/'actual';output.mkdir(exist_ok=False);rows=[]
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join((str(REPO),str(REPO/'rnj_wzzt_core')))
    for c in p['cases']:
        declared=read(checked(c['case_protocol']))
        for pos,path in enumerate(c['paths']):
            factors=np.random.default_rng(np.random.SeedSequence([c['response_seed'],path])).uniform(-1,1,2).tolist()
            for policy in POLICIES:
                identity=f"{c['name']}__{policy}__path{path}";record=output/'records'/identity;record.mkdir(parents=True)
                row=dict(condition=c['name'],case=c['case'],policy=policy,path=path,record_id=identity,status='started',physical_output=str(record/'physical'))
                if c['status']!='qualified':
                    row.update(status='not_qualified',reason=c.get('message'));rows.append(row);continue
                request=dict(status='frozen',phase='test',case=c['case'],record_id=identity,policy=policy,path=path,
                    replicate=p['replicate_base']+10*path,profile_index=c['profiles'][pos],day_offset=6,
                    labels=declared['labels'],root_label=declared['root_label'],base_kw=declared['base_kw'],factors=factors,
                    voltage_sigma=declared['voltage_sigma'],pq_sigma=declared['pq_sigma'],budget=1.6,
                    protocol=c['runtime_protocol'],plan=f['cases'][c['name']]['plans'][policy],frozen_pool=c['pool'],
                    source_metadata=declared['source_metadata'],repo_root=str(REPO),source_root=declared['source_root'],source_sha256=p['source_sha256'])
                file=record/'request.json';write(file,request);row['request']=ref(file)
                try:
                    command=[sys.executable,'-u','-B','-m','research_experiments.wzzt_dr_selection.dr_frozen_distance.one_day_worker',
                        '--request',str(file),'--expected-sha',sha(file),'--output',str(record/'physical')]
                    with (record/'physical.log').open('w',encoding='utf-8') as log:
                        run=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
                    receipt=read(record/'physical/events_receipt.json')
                    row.update(status='observed' if receipt['status']=='complete' else 'physical_failed',worker_exit=run.returncode,
                        receipt=ref(record/'physical/events_receipt.json'),actual_payment=receipt.get('actual_payment'),
                        packet=ref(record/'physical/packet.pkl') if (record/'physical/packet.pkl').exists() else None)
                except Exception as error:
                    row.update(status='physical_failed',error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
                rows.append(row);write(output/'progress.json',dict(status='running',rows=rows,planned=60))
                print(dict(stage='observed',done=len(rows),planned=60,condition=c['name'],policy=policy,status=row['status']),flush=True)
    write(output/'observations_frozen.json',dict(status='frozen',created_utc=now(),rows=rows,planned=60,
        price_freeze=ref(root/'price_freeze.json'),new_fit_calls=0,controller_truth_read=False))


def score(root):
    root,p,f=config(root);out=root/'actual';ledger=read(out/'observations_frozen.json');checked(ledger['price_freeze'])
    if (out/'scored.json').exists():raise FileExistsError('retain completed scores')
    configs={c['name']:c for c in p['cases']};pools={c['name']:read(checked(c['pool'])) for c in p['cases']};rows=[]
    for original in ledger['rows']:
        row=dict(original)
        if row['status']=='observed':
            packet=load_pickle(row['packet'])
            if len(packet['validation'])!=1 or len(packet['validation'][0]['P_terminal'])!=96:raise ValueError('one complete96-row day')
            result=weighted_score(pools[row['condition']],packet['validation'][0],history_gate=True)
            file=out/'scores'/(row['record_id']+'.json');write(file,result)
            row.update(status='predictions_frozen',scores=ref(file),new_distance=result['new_prediction_pair_distance_l1'])
        rows.append(row)
    write(out/'predictions_frozen.json',dict(status='frozen',created_utc=now(),rows=rows,
        observations_frozen=ref(out/'observations_frozen.json'),new_fit_calls=0,no_true_structure_read=True))
    for row in rows:
        if row['status']!='predictions_frozen':continue
        receipt=read(checked(row['receipt']));truthfile=Path(row['physical_output'])/'truth_evaluation.json'
        checked(dict(path=str(truthfile),sha256=receipt['private_files']['truth_evaluation.json']['sha256']))
        truth={tuple(sorted(s)) for s in read(truthfile)['true_supports']};pool=pools[row['condition']]
        def errors(blocks):
            blocks={tuple(sorted(s)) for s in blocks}
            return dict(full_errors=len(blocks^truth),missing=[list(s) for s in sorted(truth-blocks)],extra=[list(s) for s in sorted(blocks-truth)])
        candidate_errors={m['candidate_id']:errors(m['frozen_output_support_labels']) for m in pool['models']}
        correct=[k for k,v in candidate_errors.items() if v['full_errors']==0]
        result=read(checked(row['scores']));evaluations=[];wrong=[k for k in candidate_errors if k not in correct]
        for rule in result['rules']:
            evaluations.append(dict(rule_id=rule['rule_id'],new_day_weight=rule['new_day_weight'],new_day_fraction=rule['new_day_fraction'],
                selected_candidate=rule['selected_candidate_id'],**errors(rule['selected_output_support_labels']),
                eligible_candidate_ids=rule['eligible_candidate_ids'],exit_margins=rule['exit_margins'],
                wrong_models_excluded=bool(correct and wrong and all(k not in rule['eligible_candidate_ids'] for k in wrong))))
        original=result['original_ungated_selection']
        evaluations.append(dict(rule_id='original_ungated_w1_tau3pct',selected_candidate=original['selected_candidate_id'],
            **errors(original['selected_output_support_labels']),eligible_candidate_ids=original['eligible_candidate_ids']))
        evaluations.append(dict(rule_id='day_only_3pct',selected_candidate=original['day_only_selection']['selected_candidate_id'],
            **errors(original['day_only_selection']['selected_output_support_labels'])))
        main=next(e for e in evaluations if e['rule_id']==p['primary_rule'])
        row.update(status='complete',candidate_errors=candidate_errors,correct_candidate_ids=correct,
            historical_eligible_candidate_ids=result['historical_eligible_candidate_ids'],evaluations=evaluations,
            primary_rule=p['primary_rule'],full_errors=main['full_errors'],missing=main['missing'],extra=main['extra'],
            selected_candidate=main['selected_candidate'],parameters_changed=False)
    for row in rows:
        zero=next(x for x in rows if x['condition']==row['condition'] and x['path']==row['path'] and x['policy']=='zero')
        if row['status']==zero['status']=='complete':
            gamma=f['cases'][row['condition']]['gamma_native'];gain=row['new_distance']-zero['new_distance']
            row.update(distance_gain=gain,distance_target_met=(gain>=gamma-1e-12) if gamma>1e-12 else None,
                paired_DR_error_reduction=zero['full_errors']-row['full_errors'])
            old=next(e for e in row['evaluations'] if e['rule_id']=='original_ungated_w1_tau3pct')
            row['rule_error_reduction']=old['full_errors']-row['full_errors']
    write(out/'scored.json',dict(status='complete',rows=rows,planned=60,predictions_frozen=ref(out/'predictions_frozen.json'),new_fit_calls=0))
    tables=[]
    for c in p['cases']:
        for policy in POLICIES:
            group=[r for r in rows if r['condition']==c['name'] and r['policy']==policy];ok=[r for r in group if r['status']=='complete']
            known=[r['actual_payment'] for r in group if r.get('actual_payment') is not None]
            tables.append(dict(condition=c['name'],case=c['case'],policy=policy,denominator=4,complete=len(ok),unresolved=4-len(ok),
                full_errors=[r.get('full_errors') for r in group],recoveries=sum(r['full_errors']==0 for r in ok),
                mean_payment=sum(known)/4 if len(known)==4 else None,known_payment_total=sum(known),unknown_payment_records=4-len(known),
                distance_gains=[r.get('distance_gain') for r in group],
                DR_improved=sum(r.get('paired_DR_error_reduction',0)>0 for r in ok),DR_worsened=sum(r.get('paired_DR_error_reduction',0)<0 for r in ok),
                rule_improved=sum(r.get('rule_error_reduction',0)>0 for r in ok),rule_worsened=sum(r.get('rule_error_reduction',0)<0 for r in ok)))
    summary=dict(status='complete',primary_rule=p['primary_rule'],table=tables,rows=rows,planned=60,
        complete=sum(r['status']=='complete' for r in rows),conditions=5,physical_networks=3,predeclared_case_days=20,
        actual_validation_rows=384,primary_effective_weight_rows=480,new_LP=0,new_fit_calls=0,new_search_calls=0,
        AC_attempted_points=sum(96 for r in rows if r.get('receipt') and read(checked(r['receipt'])).get('AC_attempted')),
        valid_AC_points=96*sum(r['status']=='complete' for r in rows),price_freeze=ref(root/'price_freeze.json'))
    write(root/'summary.json',summary);write(root/'ended.json',dict(status='complete',summary=ref(root/'summary.json')))
    print(dict(stage='scored',complete=summary['complete'],planned=60,table=tables),flush=True)


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('phase',choices=['prepare','holdout','observe','score']);a.add_argument('--root',type=Path,required=True)
    a.add_argument('--reuse-root',type=Path)
    x=a.parse_args()
    if x.phase=='prepare':prepare(x.root,x.reuse_root)
    else:globals()[x.phase](x.root)
if __name__=='__main__':main()
