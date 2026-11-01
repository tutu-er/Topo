"""Paired working-law objective curves, without new data, solves or AC.

Reuses the original1000working-law draws per case. The six duration and six
amplitude settings were fixed in the preceding dose protocol. Historical P/Q
only determines shapes/prices; no actual-day observations or true topology is
opened by this script. The lower10% quantile is the gain attainable in90% of
these same scenarios, not a new independent probability validation.
"""
from __future__ import annotations
import argparse,json,math,sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import dose_duration as dose
import numpy as np

DEFAULT_DOSE=Path(__file__).resolve().parent/'dose_run_v2'
def finite_stats(gain,gamma,probability=.9):
    gain=np.asarray(gain,float)
    if gain.ndim!=1 or not len(gain) or not np.isfinite(gain).all():raise ValueError('finite nonempty gain vector required')
    required=int(math.ceil(probability*len(gain)-1e-12))
    quantile=float(np.sort(gain)[::-1][required-1])
    return dict(denominator=len(gain),required_scenarios=required,sample_90pct_lower_gain=quantile,
        standard_p10=float(np.quantile(gain,.1)),gain_mean=float(gain.mean()),gain_minimum=float(gain.min()),
        gain_maximum=float(gain.max()),positive_count=int((gain>1e-12).sum()),
        target_count=int((gain>=gamma-1e-12).sum()),fraction_above_original_gamma=float((gain>=gamma-1e-12).mean()),
        original_gamma_native=float(gamma))


def replay_draws(seed,count=1000):
    rng=np.random.default_rng(seed);days=[];factors=[]
    for _ in range(count):
        days.append(int(rng.integers(6)));factors.append(rng.uniform(-1,1,2))
    return np.asarray(days,int),np.asarray(factors,float)


def gains_for_samples(base_by_day,delta_columns,amplitude,factor_loadings,day_ids,factors):
    """Exact paired average L1 gain on a batch of frozen complete-day draws."""
    base=np.asarray(base_by_day,float)[day_ids]
    realized=np.asarray(amplitude)*(1+np.asarray(factors)@np.asarray(factor_loadings).T)
    effect=realized@np.asarray(delta_columns,float).T
    return (np.abs(base+effect)-np.abs(base)).mean(axis=1)


def run(root,dose_root,output):
    root=Path(root).resolve();dose_root=Path(dose_root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    p=dose.read(root/'protocol.json');f=dose.read(root/'price_freeze.json');prior=dose.read(root/'model_holdout/summary.json')
    dp=dose.read(dose_root/'protocol.json')
    if dp['duration_slots']!=list(dose.SLOTS_GRID) or dp['amplitude_alpha']!=list(dose.ALPHAS):raise ValueError('preceding numeric grid changed')
    # No actual/ directory or dose summary with old actual-day gains is opened.
    sources=[Path(__file__).resolve(),Path(dose.__file__).resolve()]
    sources += [Path(s) for s in dp['source_sha256'] if Path(s).resolve()!=Path(dose.__file__).resolve()]
    sources=list(dict.fromkeys(sources))
    case_inputs=[]
    for c in p['cases']:
        cp=dose.read(dose.checked(c['case_protocol']))
        case_inputs.append(dict(condition=c['name'],pool=c['pool'],history=cp['history_packet'],problem=c['problem'],
            plan=f['cases'][c['name']]['plans']['optimized'],holdout_seed=c['holdout_seed']))
    protocol=dict(status='frozen',created_utc=dose.now(),id='paired_working_objective_v1',
        original_root=str(root),preceding_dose_protocol=dose.ref(dose_root/'protocol.json'),
        original_protocol=dose.ref(root/'protocol.json'),price_freeze=dose.ref(root/'price_freeze.json'),
        original_working_holdout=dose.ref(root/'model_holdout/summary.json'),cases=case_inputs,
        duration_slots=list(dose.SLOTS_GRID),amplitude_alpha=list(dose.ALPHAS),samples_per_case=1000,
        sample_rule='same original interleaved RNG: day=in­tegers(6),then uniform[-1,1]^2, repeated1000',
        objective_rule='descending sort index899:900of1000 gains are at least this value; standardp10 also reported',
        source_sha256={str(s):dose.sha(s) for s in sources},new_AC=0,new_fit_calls=0,new_solver_calls=0,new_observations=0,
        scope='paired reuse of previously seen1000working-law scenarios, not new independent validation or field probability',
        price_rule='original relative two-node quote direction, common maximal feasible scale; no reoptimization or actual-day tuning',
        amplitude_scope='alpha>1requires originalbudget relaxation; sqrt2 has3.2hard cap, not1.6')
    dose.write(output/'protocol.json',protocol);shutil.copyfile(__file__,output/'source_snapshot.py')
    checks=[];tables=[];private_draws={};private_gains={}
    def check(name,condition,**details):
        checks.append(dict(name=name,passed=bool(condition),**details))
        if not condition:raise AssertionError(name)
    prior_cases={x['condition']:x for x in prior['cases']}
    for c,inputs in zip(p['cases'],case_inputs):
        pool=dose.read(dose.checked(inputs['pool']));history=dose.unpickle(inputs['history'])['history'];plan=dose.read(dose.checked(inputs['plan']))
        problem_old=dose.PriceProblem.from_dict(dose.read(dose.checked(inputs['problem'])))
        labels=pool['labels'];n=len(labels);base_kw=pool['base_kw'];T=96
        dr=np.asarray(pool['models'][1]['R'])-np.asarray(pool['models'][0]['R']);dx=np.asarray(pool['models'][1]['X'])-np.asarray(pool['models'][0]['X'])
        days=history['training']+history['validation'];check(c['name']+'_history6days',len(days)==6 and all(len(x['P_terminal'])==96 for x in days))
        hp=np.stack([x['P_terminal'].to_numpy(float) for x in days]);hq=np.stack([x['Q_terminal'].to_numpy(float) for x in days])
        direction=np.sign(hp.mean(axis=0)@dr.T+hq.mean(axis=0)@dx.T)
        base_by_day=(hp@dr.T+hq@dx.T).reshape(6,T*n)
        resources=[dose.ResourceDay(**r) for r in plan['resources']]
        active=np.flatnonzero(np.asarray(plan['prices'])>1e-12).tolist()
        columns={next(iter(plan['modules'][j]['profiles'])):j for j in active};active=[columns[r.resource_id] for r in resources]
        check(c['name']+'_two_original_active_guides',len(active)==2 and all('history_contrast' in plan['modules'][i]['module_id'] for i in active))
        prices0=np.asarray(plan['prices'])[active];lam=np.asarray(plan['elasticities'])[active];D=np.asarray(plan['response_factors'])[active]
        caps=np.asarray(problem_old.price_upper)[active]
        mean=np.r_[hp.mean(axis=0).ravel(),hq.mean(axis=0).ravel()]*base_kw
        marginal=[];original_shapes=[]
        for r,j in zip(resources,active):
            i=labels.index(r.node_label);g=dose.historical_gradient(direction,dr[:,i],dx[:,i],r.reactive_ratio,base_kw)
            marginal.append(g);h=np.asarray(plan['modules'][j]['profiles'][r.resource_id]);original_shapes.append(h)
            check(c['name']+'_'+r.resource_id+'_history_shape',np.array_equal(dose.contrast_shape(g,8),h))
        variants=[]
        for slots in dose.SLOTS_GRID:
            shapes=[dose.contrast_shape(g,slots) for g in marginal]
            pr=dose.build_problem(resources,shapes,labels,lam,D,caps,mean)
            ray=dose.largest_common_scale(pr,prices0)
            if slots==8:check(c['name']+'_m8_same_prices',np.allclose(ray['prices'],prices0,rtol=0,atol=1e-10))
            variants.append(dict(kind='duration',value=slots/4,slots=slots,problem=pr,amplitude=ray['amplitude'],beta=ray['beta']))
        base_problem=dose.build_problem(resources,original_shapes,labels,lam,D,caps,mean)
        for alpha in dose.ALPHAS:
            variants.append(dict(kind='amplitude',value=alpha,slots=8,problem=base_problem,amplitude=alpha*lam*prices0,beta=alpha))
        day_ids,z=replay_draws(c['holdout_seed'])
        private_draws[c['name']]=dict(seed=c['holdout_seed'],day_ids=day_ids.tolist(),factors=z.tolist())
        gamma=f['cases'][c['name']]['gamma_native'];old_policy=next(x for x in prior_cases[c['name']]['table'] if x['policy']=='optimized')
        for v in variants:
            pr=v['problem'];a=v['amplitude'];H=pr.profiles
            hpcol=H[:T*n].reshape(T,n,-1);hqcol=H[T*n:].reshape(T,n,-1)
            columns=(np.einsum('tkj,ik->tij',hpcol,dr)+np.einsum('tkj,ik->tij',hqcol,dx)).reshape(T*n,-1)/base_kw
            gain=gains_for_samples(base_by_day,columns,a,pr.response_factors,day_ids,z)
            stats=finite_stats(gain,gamma)
            aud=dose.evaluate_plan(pr,a)
            row=dict(condition=c['name'],kind=v['kind'],value=v['value'],reduction_hours=v['slots']/4,total_active_hours=v['slots']/2,
                beta=v['beta'],prices=(a/pr.elasticities).tolist(),mean_amplitudes_kw=a.tolist(),expected_payment=aud['expected_cost'],
                hard_budget_bound=aud['hard_budget_bound'],within_original_domain=aud['numerical_feasible'],**stats)
            tables.append(row);private_gains[c['name']+'__'+v['kind']+'__'+str(v['value'])]=gain.tolist()
            if (v['kind']=='duration' and v['slots']==8) or (v['kind']=='amplitude' and v['value']==1):
                check(c['name']+'_'+v['kind']+'_target_count_replay',stats['target_count']==old_policy['target_count'])
                check(c['name']+'_'+v['kind']+'_positive_replay',stats['positive_count']==old_policy['positive_count'])
                check(c['name']+'_'+v['kind']+'_mean_replay',abs(stats['gain_mean']-old_policy['gain_mean'])<1e-12)
                check(c['name']+'_'+v['kind']+'_p10_replay',abs(stats['standard_p10']-old_policy['gain_p10'])<1e-12)
            check(c['name']+'_'+v['kind']+'_'+str(v['value'])+'_rank90',int((gain>=stats['sample_90pct_lower_gain']-1e-16).sum())>=900)
            check(c['name']+'_'+v['kind']+'_'+str(v['value'])+'_zero_coeff_ifzero',not(v['kind']=='amplitude' and v['value']==0) or np.all(gain==0))
    check('all5x12_variants',len(tables)==60)
    for file,h in protocol['source_sha256'].items():check('source_unchanged_'+Path(file).name,dose.sha(file)==h)
    dose.write(output/'private_draws.json',private_draws);dose.write(output/'private_gains.json',private_gains)
    result=dict(status='complete',created_utc=dose.now(),protocol=dose.ref(output/'protocol.json'),tables=tables,
        cases=5,paired_scenarios_per_case=1000,variants_per_case=12,new_AC=0,new_fit_calls=0,new_solver_calls=0,new_observations=0,
        source_scope=protocol['scope'],objective_scope='90% attained paired distance gain, not topology recovery or rejection confidence')
    dose.write(output/'summary.json',result);dose.write(output/'audit.json',dict(status='complete',check_count=len(checks),checks=checks,failures=[],
        new_AC=0,new_fit_calls=0,new_solver_calls=0,no_actual_day_or_static_topology_fields_opened=True))
    dose.write(output/'public_scalars.json',{k:v for k,v in result.items() if k not in ('protocol',)})
    dose.write(output/'ended.json',dict(status='complete',summary=dose.ref(output/'summary.json'),audit=dose.ref(output/'audit.json')))
    print(json.dumps(dict(status='complete',case_variants=60,working_scenarios=5000,new_AC=0,check_count=len(checks))))
    return result

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,default=dose.DEFAULT_ROOT)
    a.add_argument('--dose-root',type=Path,default=DEFAULT_DOSE);a.add_argument('--output',type=Path,required=True)
    x=a.parse_args();run(x.root,x.dose_root,x.output)
