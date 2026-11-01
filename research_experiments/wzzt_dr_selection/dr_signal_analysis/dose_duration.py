"""Frozen-model input sensitivity, without new AC, fitting or price search.

Schedules use historical P/Q only. Archived natural P/Q and response factors
are then used for retrospective counterfactual prediction-distance arithmetic.
No voltage, AC results or static true topology is inspected. These curves are
not new delivered/observed experiments and do not evaluate model rejection.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,pickle,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import replace
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='1'
REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
for _path in (REPO,REPO/'rnj_wzzt_core'):
    if str(_path) not in sys.path:sys.path.insert(0,str(_path))
import numpy as np
from research_experiments.wzzt_dr_selection.dr_day_modules.day_schedule import ResourceDay,DayModule
from research_experiments.wzzt_dr_selection.dr_conic_pricing.resources import resource_contract_data
from research_experiments.wzzt_dr_selection.dr_conic_pricing.model import PriceProblem,evaluate_plan
from research_experiments.wzzt_dr_selection.dr_frozen_distance.design import contrast_shape

DEFAULT_ROOT=Path(r'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_newday_weighted_v1/run_v2')
SLOTS_GRID=(4,8,16,24,32,48)
ALPHAS=(0.,.25,.5,.75,1.,math.sqrt(2.))

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(item):
    p=Path(item['path'])
    if sha(p)!=item['sha256']:raise ValueError('frozen input changed: '+str(p))
    return p
def unpickle(item):
    with checked(item).open('rb') as stream:return pickle.load(stream)


def summaries(values):
    a=np.asarray(values,float)
    return dict(mean=float(a.mean()),minimum=float(a.min()),maximum=float(a.max())) if len(a) else dict(mean=None,minimum=None,maximum=None)


def paired_metrics(base,effect):
    base=np.asarray(base,float);effect=np.asarray(effect,float)
    if base.shape!=effect.shape or not np.isfinite(base).all() or not np.isfinite(effect).all():raise ValueError('finite matching prediction arrays required')
    d0=float(np.abs(base).mean());d=float(np.abs(base+effect).mean())
    r0=float(np.sqrt(np.mean(base**2)));r=float(np.sqrt(np.mean((base+effect)**2)))
    return dict(D1_baseline=d0,D1=d,D1_gain=d-d0,relative_D1_gain_pct=100*(d-d0)/d0 if d0>0 else None,
        D2_baseline=r0,D2=r,D2_gain=r-r0,DR_prediction_rms=float(np.sqrt(np.mean(effect**2))))


def historical_gradient(direction,dr_column,dx_column,reactive_ratio,base_kw):
    direction=np.asarray(direction,float);ratio=np.broadcast_to(np.asarray(reactive_ratio,float),(direction.shape[0],))
    return (direction@np.asarray(dr_column)+ratio*(direction@np.asarray(dx_column)))/(direction.size*base_kw)


def largest_common_scale(problem,prices):
    """Exact ray intersection with the original box-robust convex domain."""
    prices=np.asarray(prices,float);a=problem.elasticities*prices
    if prices.shape!=(problem.n_contracts,) or np.any(prices<0) or not np.any(prices):raise ValueError('nonzero nonnegative complete price direction required')
    zero=evaluate_plan(problem,np.zeros_like(a))
    if not zero['numerical_feasible']:raise ValueError('zero resource domain infeasible')
    bounds=[]
    for j in np.flatnonzero(a>0):bounds.append((float(problem.amplitude_upper[j]/a[j]),f'amplitude_or_price:{j}'))
    lhs=problem.device_matrix@a+np.abs((problem.device_matrix*a)@problem.response_factors).sum(axis=1)
    for i,value in enumerate(lhs):
        if value>1e-15:bounds.append((float(problem.device_rhs[i]/value),f'device:{i}'))
    cost=float(problem.hard_cost_coefficients@(a*a))
    if cost>0:bounds.append((float(np.sqrt(problem.budget/cost)),'hard_budget'))
    beta=max(0.,min(v for v,_ in bounds))
    if abs(beta-1.)<1e-7:beta=1.
    audit=evaluate_plan(problem,beta*a)
    if not audit['numerical_feasible']:raise ValueError('ray endpoint fails original domain replay')
    return dict(beta=beta,binding=[name for value,name in bounds if abs(value-beta)<=1e-6],
        amplitude=beta*a,prices=beta*prices,audit=audit)


def build_problem(resources,shapes,labels,elasticities,factors,price_caps,mean,cap=1.,budget=1.6):
    modules=[DayModule(r.resource_id+'_duration',{r.resource_id:np.asarray(h,float)},cap,
        'historical-only contrast schedule; no new price optimization') for r,h in zip(resources,shapes)]
    data=resource_contract_data(resources,modules,node_labels=labels,elasticities=elasticities,
        price_upper=price_caps,response_factors=factors)
    metadata=data.pop('metadata');metadata.update(scope='retrospective input-only duration sensitivity',new_AC=0,new_fits=0)
    return PriceProblem(mu0=mean,background_factors=np.zeros((len(mean),len(factors[0]))),
        benefit_directions=np.zeros((1,len(mean))),benefit_offsets=[0.],risk_allocations=[.1],budget=budget,
        metadata=metadata,**data)


def check_four_corners(problem,amplitudes):
    """Directly check supported amplitudes and devices, independent of norms."""
    maximum=0.
    for z0 in (-1.,1.):
        for z1 in (-1.,1.):
            actual=np.asarray(amplitudes)*(1+problem.response_factors@np.array([z0,z1]))
            maximum=max(maximum,float(np.max(problem.device_matrix@actual-problem.device_rhs,initial=0.)),
                float(np.max(actual-problem.actual_amplitude_upper,initial=0.)),float(np.max(-actual,initial=0.)))
    return maximum


def run(root,output):
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    p=read(root/'protocol.json');f=read(root/'price_freeze.json');ledger=read(root/'actual/scored.json')
    if ledger['status']!='complete' or read(root/'ended.json')['status']!='complete':raise ValueError('original test must be finished')
    refs={k:ref(root/k) for k in ['protocol.json','price_freeze.json','actual/scored.json','independent_physical_audit_v2.json']}
    sources=[Path(__file__).resolve(),REPO/'research_experiments/wzzt_dr_selection/dr_conic_pricing/model.py',
        REPO/'research_experiments/wzzt_dr_selection/dr_conic_pricing/resources.py',
        REPO/'research_experiments/wzzt_dr_selection/dr_day_modules/day_schedule.py',
        REPO/'research_experiments/wzzt_dr_selection/dr_frozen_distance/design.py']
    case_refs=[];physical_refs=[]
    for c in p['cases']:
        inp=read(checked(c['case_protocol']))
        case_refs.append(dict(condition=c['name'],pool=c['pool'],history=inp['history_packet'],
            problem=c['problem'],optimized_plan=f['cases'][c['name']]['plans']['optimized']))
        for row in ledger['rows']:
            if row['condition']==c['name'] and row['policy']=='optimized' and row['status']=='complete':
                path=Path(row['physical_output'])/'physical_truth.pkl';receipt=read(checked(row['receipt']))
                physical_refs.append(dict(condition=c['name'],path=row['path'],file=dict(path=str(path),sha256=receipt['private_files']['physical_truth.pkl']['sha256'])))
    protocol=dict(status='frozen',created_utc=now(),id='dose_duration_input_counterfactual_v1',
        original_root=str(root),inputs=refs,cases=case_refs,physical_input_refs=physical_refs,
        source_sha256={str(s):sha(s) for s in sources},duration_slots=list(SLOTS_GRID),amplitude_alpha=list(ALPHAS),
        original_budget=1.6,new_solver_calls=0,new_AC=0,new_fits=0,new_observations=0,
        scope='retrospective frozen-predictor input curves, not new observed responses or rejection tests',
        design_boundary='duration shapes from six historical days; relative original prices fixed; no observed future used to select shapes/prices',
        measurement_scope='physical archived natural P/Q and same archived response factors; no metering redraw or V analysis',
        changed_budget_scope='alpha>1 is explicitly budget-relaxed diagnostic; alpha sqrt2 requires about3.2 instead of1.6',
        denominator=dict(predeclared_case_days=20,archived_valid_case_days=19,unresolved_case_days=1))
    write(output/'protocol.json',protocol);shutil.copyfile(__file__,output/'source_snapshot.py')
    audits=[];rows=[];definitions=[];old_audit=read(root/'independent_physical_audit_v2.json')
    old_by_id={r['record_id']:r for r in old_audit['rows']}
    physical_map={(x['condition'],x['path']):x['file'] for x in physical_refs}
    def check(name,condition,**detail):
        audits.append(dict(name=name,passed=bool(condition),**detail))
        if not condition:raise AssertionError(name)
    for c,cref in zip(p['cases'],case_refs):
        pool=read(checked(cref['pool']));history=unpickle(cref['history'])['history'];plan=read(checked(cref['optimized_plan']))
        saved_problem=PriceProblem.from_dict(read(checked(cref['problem'])))
        labels=pool['labels'];n=len(labels);base_kw=pool['base_kw']
        dr=np.asarray(pool['models'][1]['R'])-np.asarray(pool['models'][0]['R']);dx=np.asarray(pool['models'][1]['X'])-np.asarray(pool['models'][0]['X'])
        days=history['training']+history['validation'];check(c['name']+'_full6history',len(days)==6 and all(len(d['P_terminal'])==96 for d in days))
        hp=np.stack([d['P_terminal'].to_numpy(float) for d in days]);hq=np.stack([d['Q_terminal'].to_numpy(float) for d in days])
        delta=hp.mean(axis=0)@dr.T+hq.mean(axis=0)@dx.T;direction=np.sign(delta)
        active=np.flatnonzero(np.asarray(plan['prices'])>1e-12).tolist();check(c['name']+'_two_active_guides',len(active)==2 and all('history_contrast' in plan['modules'][i]['module_id'] for i in active))
        resources=[ResourceDay(**r) for r in plan['resources']];resource_ids=[r.resource_id for r in resources]
        cols={next(iter(plan['modules'][j]['profiles'])):j for j in active};check(c['name']+'_one_per_resource',set(cols)==set(resource_ids))
        active=[cols[r] for r in resource_ids]
        price0=np.array(plan['prices'])[active];lam=np.array(plan['elasticities'])[active];d=np.array(plan['response_factors'])[active]
        caps=np.asarray(saved_problem.price_upper)[active]
        mean=np.r_[hp.mean(axis=0).ravel(),hq.mean(axis=0).ravel()]*base_kw
        marginals=[];original_shapes=[]
        for resource,j in zip(resources,active):
            i=labels.index(resource.node_label);g=historical_gradient(direction,dr[:,i],dx[:,i],resource.reactive_ratio,base_kw)
            marginals.append(g);old_shape=np.asarray(plan['modules'][j]['profiles'][resource.resource_id],float);original_shapes.append(old_shape)
            check(c['name']+'_'+resource.resource_id+'_baseline_shape_match',np.array_equal(contrast_shape(g,8),old_shape))
        base_problem=build_problem(resources,original_shapes,labels,lam,d,caps,mean)
        cases_def=[]
        for slots in SLOTS_GRID:
            shapes=[contrast_shape(g,slots) for g in marginals]
            pr=build_problem(resources,shapes,labels,lam,d,caps,mean)
            ray=largest_common_scale(pr,price0)
            check(c['name']+f'_duration{slots}_cornerdomain',check_four_corners(pr,ray['amplitude'])<=1e-7)
            if slots==8:
                check(c['name']+'_baseline_scale_match',abs(ray['beta']-1)<=1e-8)
                check(c['name']+'_baseline_amplitudes_match',np.allclose(ray['amplitude'],np.asarray(plan['amplitudes'])[active],atol=1e-10,rtol=0))
            cases_def.append(dict(kind='duration',value=slots/4,slots=slots,beta=ray['beta'],problem=pr,amplitude=ray['amplitude'],audit=ray['audit'],binding=ray['binding']))
        for alpha in ALPHAS:
            a=alpha*lam*price0;au=evaluate_plan(base_problem,a)
            # Increasing only this diagnostic budget checks the remaining
            # device/price constraints; it is not a deployable original plan.
            relaxed=replace(base_problem,budget=max(1.6,float(au['hard_budget_bound'])+1e-6))
            resource_check=evaluate_plan(relaxed,a)
            check(c['name']+f'_alpha{alpha}_quadraticcost',abs(au['hard_budget_bound']-alpha**2*plan['hard_payment_bound'])<=1e-8)
            check(c['name']+f'_alpha{alpha}_resourceaudit',resource_check['numerical_feasible'])
            cases_def.append(dict(kind='amplitude',value=alpha,slots=8,beta=alpha,problem=base_problem,amplitude=a,audit=au,
                binding=[],resource_only_feasible=resource_check['numerical_feasible']))
        for definition in cases_def:
            pr=definition['problem'];a=definition['amplitude'];au=definition['audit'];slots=definition['slots']
            definitions.append(dict(condition=c['name'],kind=definition['kind'],value=definition['value'],
                reduction_hours=slots/4,total_active_hours=slots/2,beta=definition['beta'],
                prices=(a/pr.elasticities).tolist(),mean_amplitudes_kw=a.tolist(),maximum_supported_amplitudes_kw=((1+pr.q)*a).tolist(),
                expected_payment=au['expected_cost'],hard_payment_bound=au['hard_budget_bound'],budget_ratio=au['hard_budget_bound']/1.6,
                within_original_domain=au['numerical_feasible'],resource_only_feasible=definition.get('resource_only_feasible',True),
                binding=definition['binding'],no_new_price_optimization=True))
            for original in ledger['rows']:
                if original['condition']!=c['name'] or original['policy']!='optimized':continue
                row=dict(condition=c['name'],case=c['case'],path=original['path'],kind=definition['kind'],value=definition['value'],
                    status='unresolved_original_resource_failure',original_record=original['record_id'])
                if original['status']!='complete':rows.append(row);continue
                physical=unpickle(physical_map[(c['name'],original['path'])])
                p0=np.asarray(physical['natural_P_pu']);q0=np.asarray(physical['natural_Q_pu']);z=np.asarray(physical['factors'])
                actual=a*(1+pr.response_factors@z);increment=pr.profiles@actual
                dp=increment[:96*n].reshape(96,n);dq=increment[96*n:].reshape(96,n)
                base=p0@dr.T+q0@dx.T;effect=(dp@dr.T+dq@dx.T)/base_kw
                metrics=paired_metrics(base,effect)
                row.update(status='counterfactual_evaluated',**metrics,
                    actual_modeled_payment=float(np.sum(pr.reduction_hours*(a/pr.elasticities)*actual)),
                    actual_amplitudes_kw=actual.tolist(),total_shifted_kwh=float(pr.reduction_hours@actual))
                if (definition['kind']=='duration' and slots==8) or (definition['kind']=='amplitude' and definition['value']==1):
                    prior=old_by_id[original['record_id']]
                    check(c['name']+f"_{original['path']}_{definition['kind']}_D_replay",abs(metrics['D1']-prior['frozen_distance_physical'])<1e-12)
                    check(c['name']+f"_{original['path']}_{definition['kind']}_gain_replay",abs(metrics['D1_gain']-prior['paired_physical_distance_gain'])<1e-12)
                    check(c['name']+f"_{original['path']}_{definition['kind']}_fee_replay",abs(row['actual_modeled_payment']-prior['actual_payment'])<1e-9)
                rows.append(row)
    tables=[]
    for definition in definitions:
        group=[r for r in rows if r['condition']==definition['condition'] and r['kind']==definition['kind'] and r['value']==definition['value']]
        valid=[r for r in group if r['status']=='counterfactual_evaluated']
        tables.append(dict(**definition,denominator=4,complete=len(valid),unresolved=4-len(valid),
            D1_baseline=summaries([r['D1_baseline'] for r in valid]),D1=summaries([r['D1'] for r in valid]),
            D1_gain=summaries([r['D1_gain'] for r in valid]),D2_baseline=summaries([r['D2_baseline'] for r in valid]),
            D2=summaries([r['D2'] for r in valid]),D2_gain=summaries([r['D2_gain'] for r in valid]),
            DR_prediction_rms=summaries([r['DR_prediction_rms'] for r in valid]),
            modeled_payment=summaries([r['actual_modeled_payment'] for r in valid])))
    for file,h in protocol['source_sha256'].items():check('source_unchanged_'+Path(file).name,sha(file)==h)
    check('all_240_arithmetic_rows_retained',len(rows)==240)
    check('one_failed_case_day_per_variant',sum(r['status']!='counterfactual_evaluated' for r in rows)==12)
    result=dict(status='complete',created_utc=now(),protocol=ref(output/'protocol.json'),rows=rows,tables=tables,
        planned_case_days=20,valid_archived_case_days=19,unresolved_archived_case_days=1,
        arithmetic_variants=12,arithmetic_rows=240,new_AC=0,new_fit_calls=0,new_solver_calls=0,new_observations=0,
        scope='frozen-model retrospective physical-input curves, not actual new network operation or rejection tests')
    write(output/'summary.json',result);write(output/'audit.json',dict(status='complete',check_count=len(audits),checks=audits,failures=[],
        new_AC=0,new_fit_calls=0,new_solver_calls=0,raw_PQV_RX_exported=False))
    write(output/'ended.json',dict(status='complete',summary=ref(output/'summary.json'),audit=ref(output/'audit.json')))
    print(json.dumps(dict(status='complete',checks=len(audits),variants=12,arithmetic_rows=240,new_AC=0,
        cases=len(p['cases']),valid_archived_case_days=19,unresolved_archived_case_days=1),ensure_ascii=False))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=DEFAULT_ROOT);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.root,args.output)
