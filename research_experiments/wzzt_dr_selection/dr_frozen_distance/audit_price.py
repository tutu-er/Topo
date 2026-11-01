"""Independent arithmetic replay: no solver, actual observations, or truth read."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import traceback

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np


def run(root):
    root=Path(root).resolve();output=root/'independent_price_audit.json'
    if output.exists():raise FileExistsError('retain existing price audit')
    checks=[];opened=[]
    def check(name,condition):
        checks.append(dict(name=name,passed=bool(condition)))
        if not condition:raise AssertionError(name)
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def read(path):
        path=Path(path);opened.append(str(path));return json.loads(path.read_text(encoding='utf-8'))
    def checked(item):
        p=Path(item['path']);check('hash_'+p.name,sha(p)==item['sha256']);return p
    def close(name,a,b,atol=1.e-12):
        check(name,bool(np.allclose(a,b,rtol=0,atol=atol)))
    audit=dict(status='started',created_utc=datetime.now(timezone.utc).isoformat(),
        new_solver_calls=0,new_parameter_fits=0,new_AC=0,
        read_scope='planning artifacts, historical measured P/Q and fixed fits only; no actual observations/truth read')
    try:
        protocol=read(root/'protocol.json');pool=read(checked(protocol['frozen_pool']))
        problem=read(checked(protocol['problem']));freeze=read(root/'price_freeze.json')
        ended=read(root/'preparation_ended.json');checked(ended['price_freeze'])
        checked(freeze['protocol']);checked(freeze['source_snapshot'])
        for p,wanted in protocol['source_sha256'].items():check('source_hash_'+Path(p).name,sha(p)==wanted)
        pool_hash=pool['pool_sha256'];payload={k:v for k,v in pool.items() if k!='pool_sha256'}
        check('pool_internal_fingerprint',hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==pool_hash)
        history_path=checked(protocol['history_packet']);opened.append(str(history_path))
        with history_path.open('rb') as stream:history=pickle.load(stream)['history']
        old=read(checked(pool['historical_fit']))
        labels=pool['labels'];n=len(labels);T=96;normalizer=T*n;base_kw=float(pool['base_kw'])
        check('full_historical_split',sum(len(d['P_terminal']) for d in history['training'])==288 and sum(len(d['P_terminal']) for d in history['validation'])==288)
        check('historical_terminal_order',list(history['labels'])==labels)
        matrices=[];parameter_replay=[]
        for model in pool['models']:
            point=old['selection_candidates'][model['historical_index']]
            check('historical_point_copy_'+model['candidate_id'],point==model['point'])
            current=[]
            for key in ('r','x'):
                saved=np.asarray(point[key],float)
                check('parameter_bytes_'+model['candidate_id']+'_'+key,saved.tobytes()==np.asarray(model['point'][key],float).tobytes())
                check('parameter_original_zero_weak_domain_'+model['candidate_id']+'_'+key,bool(np.all((saved>=0)&(saved<=2))))
                matrix=np.zeros((n,n))
                for support,value in zip(point['supports'],saved,strict=True):
                    indicator=np.zeros(n);indicator[support]=1.
                    matrix+=float(value)*np.outer(indicator,indicator)
                frozen=np.asarray(model[key.upper()],float)
                check('matrix_bytes_'+model['candidate_id']+'_'+key,matrix.tobytes()==frozen.tobytes())
                current.append(matrix)
            matrices.append(current)
            parameter_replay.append(dict(candidate_id=model['candidate_id'],historical_index=model['historical_index'],
                r_bytes_unchanged=True,x_bytes_unchanged=True,R_reconstructed_bytes_equal=True,X_reconstructed_bytes_equal=True,
                raw_support_count=len(point['supports']),historical_validation_mae=model['historical_validation_mae']))
        check('candidate_order',pool['selection_order']==[m['candidate_id'] for m in pool['models']])
        check('historical_calibration_unchanged',pool['calibration']==old['calibrations']['raw'])
        days=history['training']+history['validation']
        P=np.stack([d['P_terminal'].to_numpy(float) for d in days]);Q=np.stack([d['Q_terminal'].to_numpy(float) for d in days])
        check('six_complete_historical_PQ_days',P.shape==Q.shape==(6,96,n))
        lam=np.asarray(problem['elasticities'],float);J=len(lam)
        D=np.asarray(problem['response_factors'],float);q=np.abs(D).sum(axis=1)
        H=np.asarray(problem['profiles'],float);Hp=H[:normalizer].reshape(T,n,J);Hq=H[normalizer:].reshape(T,n,J)
        e=np.asarray(problem['reduction_hours'],float)
        caps=np.minimum(np.asarray(problem['price_upper']),np.asarray(problem['actual_amplitude_upper'])/((1+q)*lam))
        hard_beta=e*lam*(1+q);expected_beta=e*lam
        A=np.asarray(problem['device_matrix'],float);rhs=np.asarray(problem['device_rhs'],float)
        groups=problem['exclusive_groups'];budget=float(problem['budget'])
        check('original_budget_1_6',budget==protocol['budget']==1.6)
        check('N5_response_parameters',np.all(lam==.5) and protocol['aggregate_units']==5)
        plan={p:read(checked(freeze['plans'][p])) for p in protocol['policies']}
        model0,model1=matrices
        def base_prediction(day):return P[day]@model1[0].T+Q[day]@model1[1].T-(P[day]@model0[0].T+Q[day]@model0[1].T)
        base_difference=np.stack([base_prediction(day) for day in range(6)])
        delta_columns=[]
        for j in range(J):
            dp=Hp[:,:,j]/base_kw;dq=Hq[:,:,j]/base_kw
            delta_columns.append((dp@model1[0].T+dq@model1[1].T-(dp@model0[0].T+dq@model0[1].T)).ravel())
        delta_columns=np.column_stack(delta_columns)
        def domain(price):
            price=np.asarray(price,float);amplitude=lam*price
            device=A@amplitude+np.abs((A*amplitude)@D).sum(axis=1)
            violation=max(0.,float(np.max(-price,initial=0.)),float(np.max(price-caps,initial=0.)),
                float(np.max(device-rhs,initial=0.)),float(hard_beta@(price*price)-budget),
                max((np.count_nonzero(amplitude[g]>0)-1 for g in groups),default=0))
            return dict(maximum_violation=float(violation),expected_payment=float(expected_beta@(price*price)),
                hard_payment=float(hard_beta@(price*price)),amplitude=amplitude)
        def gain(day,z,price):
            actual=lam*np.asarray(price)*(1+D@z)
            pp=P[day]+(Hp@actual)/base_kw;qq=Q[day]+(Hq@actual)/base_kw
            first=pp@model0[0].T+qq@model0[1].T
            second=pp@model1[0].T+qq@model1[1].T
            return float(np.abs(second-first).mean()-np.abs(base_difference[day]).mean())
        for policy,data in plan.items():
            check('plan_price_elasticity_'+policy,np.array_equal(np.asarray(data['elasticities']),lam))
            close('plan_mean_price_'+policy,np.asarray(data['amplitudes']),lam*np.asarray(data['prices']))
            check('plan_original_domain_'+policy,domain(data['prices'])['maximum_violation']<=1.e-7)
            close('plan_expected_fee_'+policy,domain(data['prices'])['expected_payment'],data['expected_payment'])
            close('plan_hard_fee_'+policy,domain(data['prices'])['hard_payment'],data['hard_payment_bound'])
            reconstructed_p=np.zeros_like(Hp);reconstructed_q=np.zeros_like(Hq)
            resources={r['resource_id']:r for r in data['resources']}
            for j,module in enumerate(data['modules']):
                identity,shape=next(iter(module['profiles'].items()));shape=np.asarray(shape,float)
                idx=labels.index(resources[identity]['node_label'])
                reconstructed_p[:,idx,j]=shape
                reconstructed_q[:,idx,j]=float(resources[identity]['reactive_ratio'])*shape
                check('zero_energy_'+policy+'_'+str(j),abs(.25*shape.sum())<=1.e-12)
            check('physical_profile_bytes_P_'+policy,reconstructed_p.tobytes()==Hp.tobytes())
            check('physical_profile_bytes_Q_'+policy,reconstructed_q.tobytes()==Hq.tobytes())
        optimization=read(root/'optimization.json');planning=read(root/'planning_summary.json')
        draws=read(root/'planning_draws.json')['draws']
        scenario_path=root/'private_planning_scenarios.pkl';opened.append(str(scenario_path))
        with scenario_path.open('rb') as stream:stored_scenarios=pickle.load(stream)
        rng=np.random.default_rng(protocol['planning_seed']);generated=[];replay={p:[] for p in plan}
        for index,draw in enumerate(draws):
            day=int(rng.integers(6));z=rng.uniform(-1,1,2)
            check('planning_day_'+str(index),day==draw['historical_day'])
            check('planning_factors_'+str(index),np.array_equal(z,np.asarray(draw['factors'])))
            coefficient=delta_columns*(lam*(1+D@z))
            base=base_difference[day].ravel()
            close('scenario_base_'+str(index),base,stored_scenarios[index]['base'],1.e-15)
            close('scenario_price_columns_'+str(index),coefficient,stored_scenarios[index]['price_coeff'],1.e-15)
            check('scenario_normalizer_'+str(index),stored_scenarios[index]['normalizer']==normalizer)
            generated.append(dict(base=base,coefficient=coefficient))
            for policy,data in plan.items():replay[policy].append(gain(day,z,data['prices']))
        S=len(draws);required=int(np.ceil(.9*S));gamma=float(freeze['gamma_native'])
        check('64_scenarios_and58_required',S==64 and required==optimization['required_scenarios']==58)
        planning_table=[]
        for row in planning['table']:
            policy=row['policy'];values=np.asarray(replay[policy]);quantile=float(np.sort(values)[S-required]*1.e6)
            close('planning_quantile_'+policy,quantile,row['quantile_micro'],1.e-8)
            check('planning_positive_count_'+policy,int((values>1.e-12).sum())==row['positive_passes'])
            planning_table.append(dict(policy=policy,denominator=S,quantile_micro=quantile,
                positive_gain_count=int((values>1.e-12).sum()),target_gamma_count=int((values>=gamma-1.e-12).sum()),
                expected_payment=domain(plan[policy]['prices'])['expected_payment']))
        close('optimized_all64_saved_gains',replay['optimized'],optimization['scenario_gain_native'],1.e-14)
        close('freeze_gamma_replay',np.sort(replay['optimized'])[S-required],gamma,1.e-14)
        check('primary_timeout_preserved',optimization['primary']['status_code']==9 and not optimization['primary']['solver_reported_optimal'])
        close('primary_bound_gap',optimization['primary']['best_bound']-optimization['primary']['objective_value'],optimization['primary']['absolute_gap'],1.e-12)
        check('bound_not_claimed_optimal',optimization['primary']['best_bound']>=optimization['quantile_micro']-1.e-7 and not optimization['conditional_finite_scenario_optimality'])
        check('no_fee_phase_claim',optimization['secondary'] is None)
        # Independently regenerate exactly the predeclared451 physical-domain
        # starts. No solve and no candidate or actual observation is generated.
        warm=[np.zeros(J)]
        for first in groups[0]:
            for second in groups[1]:
                for angle in np.linspace(0,np.pi/2,9):
                    for radius in (.5,1.):
                        c=np.zeros(J)
                        c[first]=min(caps[first],radius*np.sqrt(budget/hard_beta[first])*np.cos(angle))
                        c[second]=min(caps[second],radius*np.sqrt(budget/hard_beta[second])*np.sin(angle))
                        c[np.abs(c)<1.e-12]=0.;warm.append(c)
        maximum_warm_violation=0.;best=0.;best_fee=0.;best_index=0
        B=np.stack([s['base'] for s in generated]);C=np.stack([s['coefficient'] for s in generated])
        for index,candidate in enumerate(warm):
            physical=domain(candidate);maximum_warm_violation=max(maximum_warm_violation,physical['maximum_violation'])
            check('warm_domain_'+str(index),physical['maximum_violation']<=1.e-7)
            shift=np.einsum('sij,j->si',C,candidate)
            gains=(np.abs(B+shift)-np.abs(B)).sum(axis=1)/normalizer
            quantile=float(np.sort(gains)[S-required]*1.e6)
            if quantile>best+1.e-8 or (abs(quantile-best)<=1.e-8 and physical['expected_payment']<best_fee):
                best=quantile;best_fee=physical['expected_payment'];best_index=index
        check('451_warm_starts_all_verified',len(warm)==optimization['provided_warm_start_count']==optimization['verified_warm_start_count']==451)
        close('warm_best_quantile',best,optimization['warm_start_best_quantile_micro'],1.e-8)
        check('warm_best_index',best_index==optimization['warm_start_best_index'])
        check('returned_quote_equals_warm_incumbent',np.array_equal(np.asarray(optimization['prices']),np.asarray(optimization['warm_start_best_prices'])))
        # Held-out random sequence is independent of planning and quotes. It
        # uses the declared working law, not any new physical observations.
        holdout=read(root/'model_holdout/summary.json');rng=np.random.default_rng(protocol['model_holdout_seed'])
        held={p:[] for p in plan}
        for index in range(protocol['model_holdout_samples']):
            day=int(rng.integers(6));z=rng.uniform(-1,1,2)
            for policy,data in plan.items():held[policy].append(gain(day,z,data['prices']))
        held_table=[]
        for row in holdout['table']:
            values=np.asarray(held[row['policy']]);positive=int((values>1.e-12).sum());target=int((values>=gamma-1.e-12).sum())
            check('holdout_denominator_'+row['policy'],len(values)==row['denominator']==1000)
            check('holdout_positive_'+row['policy'],positive==row['positive_gain_count'])
            check('holdout_target_'+row['policy'],target==row['target_gain_count'])
            for key,value in [('gain_min',values.min()),('gain_mean',values.mean()),('gain_p10',np.quantile(values,.1)),('gain_max',values.max())]:
                close('holdout_'+row['policy']+'_'+key,float(value),row[key],1.e-14)
            held_table.append(dict(policy=row['policy'],denominator=len(values),positive_gain_count=positive,
                target_gain_count=target,target_gamma_native=gamma,empirical_target_rate=target/len(values)))
        check('optimized_target90_not_validated',next(r for r in held_table if r['policy']=='optimized')['empirical_target_rate']<.9)
        check('no_actual_or_truth_artifacts_read',all('/actual/' not in p.replace('\\','/').lower() and 'truth_evaluation' not in p.lower() for p in opened))
        audit.update(status='complete',check_count=len(checks),passed=sum(c['passed'] for c in checks),checks=checks,
            parameter_replay=parameter_replay,planning_table=planning_table,held_out_table=held_table,
            warm_start_count=len(warm),warm_start_maximum_violation=maximum_warm_violation,
            warm_start_best_index=best_index,warm_start_best_quantile_micro=best,
            optimizer_added_continuous_improvement_over_seed=False,
            optimizer_primary_status=optimization['primary'],optimizer_secondary=optimization['secondary'],
            read_artifact_count=len(opened),actual_observations_read=0,actual_truth_read=0,
            conclusion='positive paired gains verified; frozen optimized target reached846/1000, so its90% magnitude promise is not validated; returned price is verified deterministic warm incumbent, not proven optimal',
            audit_source_sha256=sha(__file__))
    except Exception as error:
        audit.update(status='failed',check_count=len(checks),passed=sum(c['passed'] for c in checks),checks=checks,
            error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        output.write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        raise
    output.write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=audit['status'],checks=audit['check_count'],passed=audit['passed'],planning=planning_table,holdout=held_table),ensure_ascii=False))
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();run(args.root)
