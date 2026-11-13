"""Read-only full-native-bank Gaussian certificate audit of sealed plans.

No optimization, physical simulation or future measurement arrays are read.
Offline qualification fields in the existing manifest are never used. The
selected representative must beat every different-topology native
parameter point; this is sufficient for native-family selection, not necessary.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil
import numpy as np
from scipy.stats import norm

from ... import io
from ...history import freeze_history
from ...observations import load
from ..customer_flexibility_20261113.model import CustomerCase
from ..posterior_mse_20261112.scoring import active_topology,topology_id
from ..unified_posterior_dr_20261112.history_scan import load_request
from .data import checked


# driver.quote dispatches both pq_trust modes to the same symmetric_guard
# evidence model. Only the admissible action domain differs, not mean/risk.
NOMINAL={'symmetric','symmetric_guard','symmetric_full','symmetric_full_guard','weighted_nominal',
    'pq_trust1','pq_trust2'}
CALIBRATED={'multiplicative_absolute','capped_absolute','weighted_capped'}
CAPPED={'capped_absolute','weighted_capped'}


def load_frozen_bank(entry):
    """Historical inputs only; deliberately avoid the offline truth loader."""
    score=deepcopy(io.read(checked(entry['score'])))
    request=entry['request']
    if entry['pool']=='large96':
        native,_,validation=load_request(request)
    elif entry['pool']=='customer':
        validation=load(checked(request['observed_refs']['validation']))
        native=io.read(Path(request['directory'])/'native_result.json')
    else:
        raise ValueError('Unsupported frozen source: '+entry['pool'])
    frozen=freeze_history(native,validation)
    bank=[]
    for k,row in enumerate(native['frozen_candidates']):
        active=active_topology(row['point'],validation.labels,native['config']['zero_tol'])
        bank.append(dict(index=k,H=np.asarray(frozen.H[k]),history_sse=float(frozen.sse[k]),
            topology_id=topology_id(active)))
    for model,param in zip(score['models'],score['parameters'],strict=True):
        k=model['index']
        if param['index']!=k or not np.allclose(param['H'],frozen.H[k],rtol=1e-12,atol=1e-12):
            raise ValueError('Representative H differs from native bank')
        if not np.isclose(param['history_sse'],frozen.sse[k],rtol=1e-11,atol=1e-13):
            raise ValueError('Representative historical score differs')
        if model['topology_id']!=bank[k]['topology_id']:
            raise ValueError('Representative topology differs from native bank')
    score.update(reference_mse=native['reference_mse'],validation_tie=native['config']['validation_tie'])
    return score,validation,bank


def fitted_means(score,validation):
    """Exact V2 multiplicative calibration, reconstructed from old validation."""
    H=np.asarray([p['H'] for p in score['parameters']],float)
    P=np.c_[validation.p,validation.q]
    prediction=np.einsum('tc,knc->ktn',P,H)
    residual=validation.y[None]-prediction
    factor=1+np.sum(prediction*residual,axis=1)/(np.sum(prediction**2,axis=1)+len(P)*score['sigma2'])
    K=factor[:,:,None]*H
    corrected=validation.y[None]-np.einsum('tc,knc->ktn',P,K)
    variance=np.maximum(score['sigma2'],np.sum(corrected**2,axis=(1,2))/max(1,(len(P)-1)*H.shape[1]))
    return K,variance


def failures(Ht,history_t,mean,variance,other_H,other_history,P,P0,sigma2,tie,capped=False):
    """Analytic marginal event probabilities, without assuming independence."""
    if len(other_H)==0:
        return dict(failure=np.empty(0),actual_failure=np.empty(0),mean=np.empty(0),variance=np.empty(0))
    D=Ht[None]-np.asarray(other_H)
    d=np.einsum('tc,knc->ktn',P,D)
    d0=np.einsum('tc,knc->ktn',P0,D)
    J=np.sum(d*d,axis=(1,2))/(2*sigma2)
    rho=np.einsum('tn,ktn->k',P@(mean-Ht).T,d)/sigma2
    rho0=np.einsum('tn,ktn->k',P0@(mean-Ht).T,d0)/sigma2
    h=(np.asarray(other_history)-history_t)/(2*sigma2)
    mu=h+J+rho
    used=np.minimum(mu,h+J+rho0) if capped else mu
    var=2*(variance/sigma2)*J
    sd=np.sqrt(var)
    def probability(m):
        ans=norm.cdf(np.divide(tie-m,sd,out=np.zeros_like(m),where=sd>0))
        ans[sd==0]=(m[sd==0]<=tie).astype(float)
        return ans
    return dict(failure=probability(used),actual_failure=probability(mu),mean=mu,variance=var)


def undominated_exact_H(bank,indices):
    """For identical H retain minimum historical loss (nested error events)."""
    groups={}
    for k in indices:
        H=np.ascontiguousarray(bank[k]['H'],dtype=np.float64)
        key=(H.shape,H.tobytes())
        if key not in groups or bank[k]['history_sse']<bank[groups[key]]['history_sse']:
            groups[key]=k
    return list(groups.values())


def audit_target(score,bank,forecast,power,mean,variance,target,tie,capped):
    Ht=np.asarray(score['parameters'][target]['H'],float)
    history_t=float(score['parameters'][target]['history_sse'])
    identity=score['models'][target]['topology_id']
    reps=[int(m['index']) for m in score['models'] if m['topology_id']!=identity]
    raw=[k for k,b in enumerate(bank) if b['topology_id']!=identity]
    unique=undominated_exact_H(bank,raw)
    omitted=[k for k in raw if k not in set(reps)]
    values=failures(Ht,history_t,mean,variance,[b['H'] for b in bank],
        [b['history_sse'] for b in bank],power,forecast,score['sigma2'],tie,capped)
    f=values['failure'];actual=values['actual_failure']
    total=lambda indices:float(f[indices].sum())
    maximum=lambda indices:float(f[indices].max()) if indices else 0.
    return dict(target=target,representative_competitors=len(reps),native_competitors=len(raw),
        native_distinct_H_competitors=len(unique),omitted_native_parameter_points=len(omitted),
        representative_failure_sum=total(reps),native_raw_failure_sum=total(raw),
        native_distinct_H_failure_sum=total(unique),native_actual_mean_failure_sum=float(actual[raw].sum()),
        native_worst_pair_failure=maximum(raw),omitted_worst_pair_failure=maximum(omitted),
        native_raw_recovery_lower=max(0.,1-total(raw)),
        native_distinct_H_recovery_lower=max(0.,1-total(unique)),
        representative_pass90=total(reps)<=.1+1e-8,
        native_raw_pass90=total(raw)<=.1+1e-8,native_distinct_H_pass90=total(unique)<=.1+1e-8,
        worst_omitted_index=None if not omitted else int(max(omitted,key=lambda k:f[k])))


def run(run_dir,out):
    if out.exists():
        raise ValueError('Preserve previous audit; use a new output directory')
    if not (run_dir/'complete.json').exists():
        raise ValueError('Audit only a completed frozen batch')
    protocol=io.read(run_dir/'protocol.json');entries=io.read(run_dir/'inputs.json')
    records=[];skipped=Counter();errors=[];quotes=[]
    for entry in entries:
        score,val,bank=load_frozen_bank(entry)
        case=CustomerCase(io.read(run_dir/'resource_configs'/(entry['case']+'.json')))
        means,variances=fitted_means(score,val)
        for day in range(protocol['days']):
            directory=run_dir/'days'/entry['id']/f'day{day:03d}'
            if not (directory/'seal.json').exists():
                skipped['unsealed_days']+=1
                continue
            seal=io.read(directory/'seal.json')
            for name,key in [('planner.npz','planner'),('plans.json','plans')]:
                if io.ref(directory/name)['sha256']!=seal[key]['sha256']:
                    raise ValueError('Sealed planning input drift: '+str(directory/name))
            plans=io.read(directory/'plans.json')
            # Read only the power-unit base, not clean/observed future P/Q/V.
            base=float(io.read(directory/'generation.json')['base_kw'])
            with np.load(directory/'planner.npz') as z:
                forecast=z['forecast'].copy()
                controls={mode:z[mode].copy() for mode in plans}
            cells=score['cells']+forecast.shape[0]*case.N
            tie=max(1e-7,cells*score['reference_mse']*score['validation_tie']/(2*score['sigma2']))
            for mode,plan in plans.items():
                if mode not in NOMINAL|CALIBRATED or 'targets' not in plan:
                    skipped[mode]+=1
                    continue
                power=forecast+case.shift_pq_pu(controls[mode],base)
                own=[]
                for target in plan['targets']:
                    nominal=mode in NOMINAL
                    mean=np.asarray(score['parameters'][target]['H']) if nominal else means[target]
                    variance=score['sigma2'] if nominal else float(variances[target])
                    row=audit_target(score,bank,forecast,power,mean,variance,target,tie,mode in CAPPED)
                    row.update(id=entry['id'],day=day,mode=mode,planned_feasible=bool(plan['feasible']),
                        plan_status=plan['status'],zero_action=bool(np.max(abs(controls[mode]),initial=0)<=1e-10))
                    if plan['feasible'] and not row['representative_pass90']:
                        errors.append(dict(id=entry['id'],day=day,mode=mode,target=target,
                            issue='Recorded feasible representative plan fails independent risk sum',sum=row['representative_failure_sum']))
                    records.append(row);own.append(row)
                quotes.append(dict(id=entry['id'],day=day,mode=mode,planned_feasible=bool(plan['feasible']),
                    plan_status=plan['status'],zero_action=bool(np.max(abs(controls[mode]),initial=0)<=1e-10),
                    targets=len(own),representative_pass90=all(r['representative_pass90'] for r in own),
                    native_raw_pass90=all(r['native_raw_pass90'] for r in own),
                    native_distinct_H_pass90=all(r['native_distinct_H_pass90'] for r in own),
                    native_worst_pair_over10=any(r['native_worst_pair_failure']>.1+1e-8 for r in own)))
    summary={}
    for mode in sorted(set(protocol['modes']) & (NOMINAL|CALIBRATED)):
        allq=[q for q in quotes if q['mode']==mode]
        feasible=[q for q in allq if q['planned_feasible']]
        r=[v for v in records if v['mode']==mode and v['planned_feasible']]
        summary[mode]=dict(planned_quotes=len(entries)*protocol['days'],recorded_quotes=len(allq),
            unaudited_quotes=len(entries)*protocol['days']-len(allq),
            feasible_quotes=len(feasible),feasible_target_checks=len(r),
            feasible_zero_quotes=sum(q['zero_action'] for q in feasible),
            feasible_nonzero_quotes=sum(not q['zero_action'] for q in feasible),
            feasible_zero_native_raw_pass=sum(q['zero_action'] and q['native_raw_pass90'] for q in feasible),
            feasible_nonzero_native_raw_pass=sum(not q['zero_action'] and q['native_raw_pass90'] for q in feasible),
            all_executed_zero_quotes=sum(q['zero_action'] for q in allq),
            all_executed_zero_native_raw_pass=sum(q['zero_action'] and q['native_raw_pass90'] for q in allq),
            fallback_zero_quotes=sum(not q['planned_feasible'] and q['zero_action'] for q in allq),
            fallback_zero_native_raw_pass=sum(not q['planned_feasible'] and q['zero_action'] and q['native_raw_pass90'] for q in allq),
            feasible_quotes_representative_pass=sum(q['representative_pass90'] for q in feasible),
            feasible_quotes_native_raw_pass=sum(q['native_raw_pass90'] for q in feasible),
            feasible_quotes_native_distinct_H_pass=sum(q['native_distinct_H_pass90'] for q in feasible),
            feasible_quotes_native_raw_over10=sum(not q['native_raw_pass90'] for q in feasible),
            feasible_quotes_native_distinct_H_over10=sum(not q['native_distinct_H_pass90'] for q in feasible),
            feasible_quotes_native_worst_pair_over10=sum(q['native_worst_pair_over10'] for q in feasible),
            maximum_feasible_native_failure_sum=max((v['native_raw_failure_sum'] for v in r),default=None),
            maximum_feasible_omitted_pair_failure=max((v['omitted_worst_pair_failure'] for v in r),default=None),
            omitted_parameter_checks=sum(v['omitted_native_parameter_points'] for v in r))
    result=dict(protocol=io.ref(run_dir/'protocol.json'),source=io.ref(__file__),summary=summary,
        skipped_modes_or_days=dict(skipped),errors=errors,passed=not errors,
        interpretation='Each chosen representative is treated as the true declared mean (or its fitted multiplicative mean). Sum of Gaussian marginal failure probabilities is a sufficient union bound over all different-topology native parameter points. Values above 0.1 fail this sufficient certificate; they do not prove actual native recovery below 90%, because events overlap and another same-topology parameter point may rescue selection. No rejection guarantee. Identical frozen-H competitors are additionally deduplicated by minimum historical SSE, preserving nested-event union coverage.',
        scope='Uses sealed forecast/control, old validation/native parameters, plan receipts and metadata base_kw only. The existing manifest contains offline qualification fields, which are unused; separate truth records and future observation/outcome files are not opened. No new planning or simulation.')
    out.mkdir(parents=True)
    shutil.copyfile(__file__,out/'source_snapshot.py')
    result['source_snapshot']=io.ref(out/'source_snapshot.py')
    io.write(out/'audit.json',result);io.write(out/'targets.json',records);io.write(out/'quotes.json',quotes)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=run(args.run.resolve(),args.out.resolve())
    import json
    print(json.dumps(dict(summary=result['summary'],errors=result['errors'],passed=result['passed']),indent=2))
