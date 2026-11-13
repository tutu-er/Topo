"""Read-only recomputation of sealed fresh-day results, fees and certificates."""
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.special import logsumexp

from ... import io
from ...observations import load
from ..customer_flexibility_20261113.model import CustomerCase
from .solver import Recovery


ROOT = Path(__file__).resolve().parents[3]


def verify(out, *, partial=False):
    protocol = io.read(out/'protocol.json')
    checks = dict(completed_days=0, failed_jobs=0, strategy_rows=0, failed_strategy_rows=0,
        successful_certificates=0, explicit_zero_fallbacks=0, requested_resource_checks=0,
        delivered_resource_checks=0, ac_score_recomputations=0)
    errors, manifest, certificates = [], [], []
    maximum = dict(probability=0., loss=0., fee=0., physical_power=0., measured_channel=0.)

    def require(condition, description):
        if not condition:
            raise AssertionError(description)

    def hashcheck(path, expected):
        actual = io.ref(path)
        require(actual['sha256'] == expected, 'hash mismatch: '+str(path))
        manifest.append(actual)

    hashcheck(out/'protocol.json', io.read(out/'lock.json')['protocol']['sha256'])
    for relative, sha in protocol['sources'].items():
        hashcheck(ROOT/relative, sha)
        hashcheck(out/'source_snapshot'/relative, sha)
    truth = io.read(out/'offline_truth.json')
    true_clades = set(map(tuple,truth['truth_clades']))
    for entry in protocol['entries']:
        hashcheck(entry['score']['path'],entry['score']['sha256'])
        for ref in entry['observed_refs'].values():
            hashcheck(ref['path'],ref['sha256'])
        score = io.read(entry['score']['path'])
        validation = load(entry['observed_refs']['validation']['path'])
        config = out/'resource_configs'/f"{entry['adoption']}.json"
        reference_config = ROOT/'outputs/customer_flexibility_sweep_20261113_v1/resource_configs'/config.name
        hashcheck(config,io.ref(reference_config)['sha256'])
        case = CustomerCase(io.read(config))
        H = np.asarray([p['H'] for p in score['parameters']])
        history = np.array([p['history_sse'] for p in score['parameters']])
        target = score['map_index']
        ids = [p['topology_id'] for p in score['models']]
        truth_index = next((i for i,x in enumerate(ids) if x == truth['truth_id']),None)
        for day in range(entry['new_days']):
            d = out/'confirmation'/entry['id']/f'day{day:03d}'
            if (d/'worker_failure.json').exists():
                checks['failed_jobs'] += 1
                manifest.append(io.ref(d/'worker_failure.json'))
                continue
            if not (d/'complete.json').exists():
                continue
            try:
                completion = io.read(d/'complete.json')
                for field in ('results','plans'):
                    hashcheck(d/(field+'.json'),completion[field]['sha256'])
                seal = io.read(d/'policy_seal.json')
                for field,name in [('planner','planner.npz'),('plans','plans.json'),('protocol','protocol.json')]:
                    hashcheck((out if field=='protocol' else d)/name,seal[field]['sha256'])
                require(seal['score']['sha256']==entry['score']['sha256'],'wrong frozen score in seal')
                require(seal['actual_response_or_execution_voltage_opened'] is False,'policy seal admits future voltage')
                generation = io.read(d/'generation.json')
                base = generation['base_kw']
                with np.load(d/'planner.npz') as z:
                    plans_np = {k:z[k].copy() for k in z.files}
                with np.load(d/'physical_background.npz') as z:
                    background = {k:z[k].copy() for k in z.files}
                eta = background['response_multiplier']
                require(np.all((eta>=.9)&(eta<=1.)),'resource response outside declared interval')
                plans = io.read(d/'plans.json')
                for mode,receipt in plans.items():
                    planned_fee = case.fee(plans_np[mode])
                    rotation = 'old_rotation' if mode=='old_gamma2' else mode+'_rotation'
                    rotation_fee = case.fee(plans_np[rotation])
                    require(abs(rotation_fee-receipt['rotation_planned_fee_yuan'])<1e-8,'rotation fee receipt mismatch')
                    require(receipt['rotation_same_planned_fee']==bool(abs(planned_fee-rotation_fee)<=1e-7),'rotation equal-fee flag mismatch')
                    if receipt['feasible']:
                        require(abs(planned_fee-receipt['cost_yuan'])<1e-8,'quoted fee differs from planned fee')
                        lower = receipt.get('lower_bound_yuan')
                        if lower is not None:
                            require(lower<=planned_fee+1e-7*max(1.,planned_fee),'fee lowerbound exceeds feasible upperbound')
                        if receipt.get('source') in ('verified_initial_retained','supporting_LP_initial_point_retained'):
                            require(abs(receipt['initial']['cost_yuan']-planned_fee)<1e-8,'retained initial cost mismatch')
                            require(abs(receipt['improvement_from_verified_initial_yuan'])<1e-8,'retained initial falsely reports improvement')
                    else:
                        require(np.all(plans_np[mode]==0),'unsuccessful quote executes an action')
                recovery = {mode:Recovery(case,score,validation,plans_np['forecast'],base,mode,entry['voltage_sigma'])
                            for mode in ('nominal','calibrated')}
                for mode in ('nominal','calibrated'):
                    receipt = plans[mode]
                    if receipt['feasible']:
                        actual = recovery[mode].verify_expansion(plans_np[mode])
                        require(actual['feasible'],'reported feasible action fails original Gaussian constraints')
                        require(actual['model_recovery_lower']>=.9-2e-8,'joint recovery certificate below90%')
                        saved = receipt.get('final',receipt['baseline'])
                        for field in ('model_recovery_lower','pair_failure_sum','reject_failure','bias_energy'):
                            require(np.isclose(actual[field],saved[field],atol=1e-10,rtol=1e-9),'stored certificate mismatch '+field)
                        checks['successful_certificates'] += 1
                        certificates.append(dict(id=entry['id'],day=day,mode=mode,
                            actual_lower=actual['model_recovery_lower'],source=receipt['source']))
                    else:
                        require(receipt['explicit_zero_fallback'],'failed quote has no declared fallback')
                        require(np.all(plans_np[mode]==0),'failed quote executes nonzero action')
                rows = io.read(d/'results.json')
                require(len(rows)==len(protocol['strategies']),'strategy denominator changed')
                require({r['strategy'] for r in rows}==set(protocol['strategies']),'strategy names changed')
                for row in rows:
                    strategy = row['strategy']; u = plans_np[strategy]; delivered = u*eta[None]
                    checks['strategy_rows'] += 1
                    require(case.audit(u)['passed'],'requested resource violation')
                    checks['requested_resource_checks'] += 1
                    require(case.audit(delivered)['passed'],'delivered resource violation')
                    checks['delivered_resource_checks'] += 1
                    fees = [float(case.dt*np.sum(case.fee_plus*np.maximum(x,0)+case.fee_minus*np.maximum(-x,0))) for x in (u,delivered)]
                    err = max(abs(fees[0]-row['planned_fee_yuan']),abs(fees[1]-row['settled_fee_yuan']))
                    maximum['fee']=max(maximum['fee'],err)
                    require(err<1e-8,'fee arithmetic mismatch')
                    require(abs(float(case.dt*np.abs(delivered).sum())-row['delivered_abs_kwh'])<1e-8,'delivered energy mismatch')
                    checks['explicit_zero_fallbacks'] += int(row['zero_fallback'])
                    if row['zero_fallback']:
                        require(np.all(u==0),'fallback row contains action')
                    if row['status'] != 'complete':
                        checks['failed_strategy_rows'] += 1
                        continue
                    path = d/(strategy+'.npz')
                    manifest.append(io.ref(path))
                    with np.load(path) as z:
                        values={k:z[k].copy() for k in z.files}
                    shift = np.zeros((96,2*case.N))
                    for h in range(case.H):
                        for r,node in enumerate(case.node):
                            shift[4*h:4*h+4,node] += delivered[h,r]/base
                            shift[4*h:4*h+4,case.N+node] += case.qratio[r]*delivered[h,r]/base
                    pe = background['P']+shift[:,:case.N]
                    qe = background['Q']+shift[:,case.N:]
                    err=max(float(np.max(abs(pe-values['true_P']))),float(np.max(abs(qe-values['true_Q']))))
                    maximum['physical_power']=max(maximum['physical_power'],err)
                    require(err<1e-12,'true physical P/Q does not match delivered action')
                    expected = dict(P=pe*(1+generation['pq_sigma']*background['meter_p']),
                        Q=qe*(1+generation['pq_sigma']*background['meter_q']),
                        V=values['true_V']*(1+generation['voltage_sigma']*background['meter_v']),
                        root=background['root']*(1+generation['voltage_sigma']*background['meter_root']))
                    err=max(float(np.max(abs(expected[k]-values[k]))) for k in expected)
                    maximum['measured_channel']=max(maximum['measured_channel'],err)
                    require(err<1e-12,'meter channel mismatch')
                    P=np.c_[values['P'],values['Q']]
                    Y=values['root'][:,None]**2-values['V']**2
                    loss=history+np.sum((Y[None]-np.einsum('tc,knc->ktn',P,H))**2,axis=(1,2))
                    err=float(np.max(abs(loss-values['cumulative_sse'])))
                    maximum['loss']=max(maximum['loss'],err)
                    require(err<1e-10,'cumulative representative loss mismatch')
                    ll=-(loss-loss.min())/(2*score['sigma2']); ll-=logsumexp(ll)
                    err=float(np.max(abs(np.exp(ll)-np.exp(values['log_probability']))))
                    maximum['probability']=max(maximum['probability'],err)
                    require(err<1e-9,'posterior weight mismatch')
                    best=int(np.argmax(ll))
                    require(best==row['selected_index'],'representative winner mismatch')
                    require((ids[best]==truth['truth_id'])==row['map_correct'],'representative recovery mismatch')
                    require(abs(np.exp(ll[target])-row['p_target'])<1e-9,'target probability mismatch')
                    require((truth_index is None and row['p_true'] is None) or
                        (truth_index is not None and abs(np.exp(ll[truth_index])-row['p_true'])<1e-9),'truth probability mismatch')
                    native_loss=np.array([p['history_sse']+np.sum((Y-P@np.asarray(p['H']).T)**2) for p in score['native_bank']])
                    scaled=native_loss/(score['cells']+Y.size)/score['reference_mse']
                    eligible=np.flatnonzero(scaled<=scaled.min()+score['validation_tie'])
                    winner=min(eligible,key=lambda i:(score['native_bank'][i]['internal'],scaled[i],i))
                    require(winner==row['native_selected_index'],'native tie/selection mismatch')
                    chosen=score['native_bank'][winner]
                    clades=set(map(tuple,chosen['active_clades']))
                    require((chosen['topology_id']==truth['truth_id'])==row['native_map_correct'],'native recovery mismatch')
                    require(len(true_clades-clades)==row['native_missing'] and len(clades-true_clades)==row['native_extra'],'native structure counts mismatch')
                    for factor,rule in row['rejection_rules'].items():
                        B=(score['cells']+Y.size)*float(factor)*8*entry['voltage_sigma']**2
                        lognull=-B/(2*score['sigma2']); raw=-loss/(2*score['sigma2'])
                        den=logsumexp(np.r_[lognull,raw]); reject=B<=loss[best]
                        require(reject==rule['rejected'],'rejection decision mismatch')
                        require(abs(np.exp(raw[target]-den)-rule['p_target'])<1e-9,'rejected target weight mismatch')
                        require(rule['correct']==bool(not reject and best==truth_index),'rejected recovery mismatch')
                    checks['ac_score_recomputations'] += 1
                require(completion['failed_rows']==sum(r['status']!='complete' for r in rows),'failure marker mismatch')
                checks['completed_days'] += 1
            except Exception as exc:
                errors.append(dict(id=entry['id'],day=day,error=repr(exc)))
    if not partial and checks['completed_days']+checks['failed_jobs'] != protocol['total_new_days']:
        errors.append(dict(error='Not all planned days have verified completion or preserved worker failure'))
    return dict(status='passed' if not errors else 'failed',scope='sealed-input and stored-result arithmetic; AC equations not re-solved',
        partial=partial,protocol=io.ref(out/'protocol.json'),checks=checks,maximum_errors=maximum,
        errors=errors,certificates=certificates,input_manifest=manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing verification receipt')
    result=verify(args.run.resolve(),partial=args.partial)
    io.write(args.output,result)
    print(json.dumps({k:result[k] for k in ('status','partial','checks','maximum_errors','errors')},indent=2))
    if result['errors']:
        raise SystemExit(1)
