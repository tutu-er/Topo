"""Prespecified post-evaluation electrical location ablation.

This tool is outside the frozen selector. It is run only after evaluation has
finished. It does not retrain, plan an EV action or apply a calibrated test to
the counterfactual. Tests inject a mock backend and perform no AC operations.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import numpy as np


INDICES=(0,12,24,36)
FAMILIES=('square','scalar','cap','jointgrad')
CHANNELS=('p','q','v','root_v','master_p','master_q','source_p','source_q')
SCHEMA='EV_posthoc_location_ablation_v1'


def _get(obj,key):return obj.get(key) if isinstance(obj,dict) else getattr(obj,key)


def verify_observations(saved,replayed,*,atol=1e-12,rtol=1e-12):
    channels={};success=True
    for name in CHANNELS:
        a,b=_get(saved,name),_get(replayed,name)
        if a is None or b is None:
            ok=a is None and b is None;channels[name]=dict(success=ok,present=False);success&=ok;continue
        a,b=np.asarray(a,float),np.asarray(b,float)
        ok=bool(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a,b,atol=atol,rtol=rtol))
        channels[name]=dict(success=ok,present=True,max_absolute_error=(float(np.max(abs(a-b)))
            if a.shape==b.shape else None));success&=ok
    return dict(success=bool(success),channels=channels,atol=atol,rtol=rtol,
        source_scope='Source meter arrays are checked if saved. Latent source is regenerated once and shared by both actions.')


def relocate_net_power(power):
    u=np.asarray(power,float)
    if u.ndim!=2 or not u.size or not np.isfinite(u).all():raise ValueError('finite time-by-site realized EV power required')
    result=np.zeros_like(u);result[:,0]=u.sum(axis=1);return result


def _key(power):return sha256(np.asarray(power,dtype='<f8').tobytes()).hexdigest()


def _operation_fields(method):
    op=method.get('operation',{}).get('audit',{});moves=op.get('moves')
    return dict(moved=None if moves is None else bool(moves),moves=moves,
        worker_minutes=op.get('driver_cost'),driving_energy_kwh=op.get('travel_energy_kwh'))


class ACBackend:
    """Existing generator plus restored history cache; no new training fit."""
    def __init__(self):
        from . import ev_mobile_v2_study as study
        self.study=study;self.current_sources=study.source_hashes();self.counters={}

    def regenerate(self,record):
        if record.get('source_sha256')!=self.current_sources:
            raise ValueError('current scientific sources differ from saved evaluation; use its frozen source snapshot')
        case=self.study.make_case(record['case_id'],record['seed'],record['family_index'])
        expected=[c['name'] for c in record['candidates']]
        if [c.name for c in case['bank']]!=expected:raise ValueError('regenerated candidate bank differs')
        return case

    def observe(self,case,power):
        return self.study.observe_future(case,power)

    def restore(self,record,case):
        from . import likelihood as lk
        from . import ev_systematic_refit as refit
        cfg=record['config'];options=refit._options({**cfg['training'],**cfg['source'],'ratio_bounds':cfg['ratio_bounds']})
        training=lk.Observations(**record['training']);validation=lk.Observations(**record['validation'])
        tm=lk.MeasurementModel(**record['training_meter']);vm=lk.MeasurementModel(**record['validation_meter'])
        fm=lk.MeasurementModel(**record['future_meter'])
        fits={row.get('candidate',row.get('name')):row for row in record['training_result']['fits']}
        histories={}
        for candidate in case['bank']:
            saved=record['history_prefits'][candidate.name];fixed=saved['fixed'];joint=saved['joint']
            val=fixed['validation_fit'];fit=fits[candidate.name]
            histories[candidate.name]=refit.RefitHistory(
                refit._fingerprint(candidate,(training,validation),(tm,vm),options),fit,val,fixed,
                refit._join_state(fit['r'],fit['x'],(fit,val)),joint,
                {key:np.asarray(joint['state'][key],float) for key in ('r','x','source_p','source_q')},saved['seconds'])
        return dict(bank=case['bank'],training=training,validation=validation,tm=tm,vm=vm,fm=fm,
            fits=fits,histories=histories,options=options,seed=record['seed'])

    def fit(self,context,future):
        from . import ev_systematic_refit as refit
        comparisons={};failures={};counter={};began=perf_counter()
        for j,candidate in enumerate(context['bank']):
            try:
                comparisons[candidate.name]=refit.compare_fixed_and_joint(candidate,context['training'],
                    context['validation'],future,context['tm'],context['vm'],context['fm'],
                    training_fit=context['fits'][candidate.name],history_cache=context['histories'][candidate.name],
                    fit_options=context['options'],seed=context['seed']+1000+j,call_counter=counter)
            except Exception as exc:failures[candidate.name]=type(exc).__name__+': '+str(exc)
        losses={name:row['joint']['joint_total'] for name,row in comparisons.items()}
        complete=bool(not failures and len(comparisons)==len(context['bank']) and all(
            row['joint']['post']['success'] for row in comparisons.values()) and
            all(np.isfinite(value) for value in losses.values()))
        return dict(complete=complete,losses=losses,
            comparisons=comparisons,failures=failures,counters=counter,seconds=perf_counter()-began,
            training_recomputed=False,EV_planning_called=False,history_restored=True)


def _margin(losses,truth):
    rivals=[v for name,v in losses.items() if name!=truth]
    return None if truth not in losses or not rivals else min(rivals)-losses[truth]


def audit_record(record,backend):
    """Audit one of the eight prespecified evaluation histories."""
    began=perf_counter()
    if record.get('phase')!='evaluation' or record.get('index') not in INDICES:
        raise ValueError('only the prespecified evaluation indices are eligible')
    methods={m['name']:m for m in record.get('methods',[])}
    result=dict(schema=SCHEMA,bank=record.get('physical_bank'),index=record['index'],seed=record['seed'],
        case_id=record['case_id'],true_name=record.get('true_name'),policy_id=record.get('policy_id'),
        source_sha256=record.get('source_sha256'),methods=[],counterfactuals={},
        tracking_error_model='No stochastic tracking error is simulated; supplied EV power is realized power.',
        command_derating='6/1.05 is a command limit, not a simulated tracking distribution.',
        same_actual_tracking_rule='Move the same realized per-EV net power; do not redraw node-specific actuation noise.',
        counterfactual_is_executable_policy=False,statistical_decision_applied=False,
        backend_call_counts=dict(case_reconstruction=0,original_future_replays=0,counterfactual_future_generations=0,
            history_cache_restorations=0,counterfactual_fit_batches=0),
        decomposition='M_mobile-M_fixed = (M_mobile-M_site0_same_net)+(M_site0_same_net-M_fixed)')
    calls=result['backend_call_counts']
    try:
        calls['case_reconstruction']+=1;case=backend.regenerate(record)
    except Exception as exc:
        result.update(failure=type(exc).__name__+': '+str(exc),success=False,seconds=perf_counter()-began)
        result['methods']=[dict(family=f,success=False,**_operation_fields(methods.get(f+'_mobile',{})),
            failure=result['failure']) for f in FAMILIES]
        return result
    replays={};replay_checks={};context=None
    # Replay all original observed curves first, before any new counterfactual fit.
    for family in FAMILIES:
        name=family+'_mobile';method=methods.get(name,{})
        try:
            u=np.asarray(method['power_kw'],float);action=method['action_key']
            original=record['future_records'][action]['observations']
            if _key(u) not in replays:
                calls['original_future_replays']+=1
                replays[_key(u)]=backend.observe(case,u)
            replay_checks[family]=verify_observations(original,replays[_key(u)])
            if not replay_checks[family]['success']:raise ValueError('regenerated original future meters differ')
        except Exception as exc:
            result['methods'].append(dict(family=family,success=False,**_operation_fields(method),
                failure=type(exc).__name__+': '+str(exc)))
    failed={m['family'] for m in result['methods']}
    for family in FAMILIES:
        if family in failed:continue
        method=methods[family+'_mobile'];fixed=methods.get(family+'_fixed',{});u=np.asarray(method['power_kw'],float)
        row=dict(family=family,success=False,original_replay=replay_checks[family],**_operation_fields(method))
        try:
            mobile_mode=method['inference']['joint_total'];fixed_mode=fixed['inference']['joint_total']
            if not mobile_mode['complete'] or not fixed_mode['complete']:
                raise ValueError('original fixed/mobile inference incomplete')
            if not method['operation'].get('success'):raise ValueError('original mobile operation failed')
            op=method['operation'].get('audit',{});moves=int(op['moves']);u0=relocate_net_power(u)
            equal=bool(np.allclose(u,u0,atol=1e-12,rtol=0.))
            if not moves and not equal:raise ValueError('unmoved EV has remote electrical power')
            Mm=float(mobile_mode['M']);Mf=float(fixed_mode['M']);key=_key(u0)
            if not np.isfinite([Mm,Mf]).all():raise ValueError('original margins must be finite')
            reused=False
            if equal:
                Mu=Mm;counter=None;space=0.
            else:
                if key not in result['counterfactuals']:
                    if context is None:
                        calls['history_cache_restorations']+=1;context=backend.restore(record,case)
                    calls['counterfactual_future_generations']+=1;future=backend.observe(case,u0)
                    calls['counterfactual_fit_batches']+=1;counter=backend.fit(context,future)
                    result['counterfactuals'][key]=dict(power_kw=u0,observations={name:_get(future,name) for name in CHANNELS},**counter)
                else:reused=True
                counter=result['counterfactuals'][key]
                if not counter['complete']:raise ValueError('counterfactual candidate fits incomplete')
                Mu=_margin(counter['losses'],record['true_name'])
                if Mu is None or not np.isfinite(Mu):raise ValueError('counterfactual margin undefined or nonfinite')
                space=Mm-Mu
            curve=Mu-Mf;total=Mm-Mf;error=abs(total-space-curve)
            if error>1e-10*(1+abs(total)+abs(space)+abs(curve)):raise ValueError('decomposition identity failed')
            drive=float(op.get('travel_energy_kwh',0.));worker=float(op.get('driver_cost',0.))
            row.update(success=True,moved=bool(moves),moves=moves,worker_minutes=worker,driving_energy_kwh=drive,
                counterfactual_power_kw=u0,counterfactual_key=key,counterfactual_fit_reused=reused,
                counterfactual_fit_skipped_same_electrical_power=equal,
                M_mobile=Mm,M_fixed=Mf,M_site0_same_net=Mu,M_space=space,M_curve_extra=curve,
                total_M_gain=total,identity_error=error,
                counterfactual_terminal_energy_increment_kwh=drive,
                counterfactual_scope='Same net curve at site0; absent driving increases terminal battery energy. This is electrical attribution only.')
        except Exception as exc:row['failure']=type(exc).__name__+': '+str(exc)
        result['methods'].append(row)
    result.update(success=bool(len(result['methods'])==len(FAMILIES) and all(m['success'] for m in result['methods'])),
        original_unique_curves_replayed=len(replays),counterfactual_unique_fit_batches=len(result['counterfactuals']),
        seconds=perf_counter()-began)
    return result


def run(n3_directories,n4_directories,output,*,backend=None):
    from . import gradient_price_study as io
    output=Path(output)
    if output.exists():raise ValueError('retain prior spatial audit; choose a new output directory')
    output.mkdir(parents=True)
    backend=ACBackend() if backend is None else backend;histories=[]
    for bank,directories in (('n3',n3_directories),('n4',n4_directories)):
        for index in INDICES:
            choices=[p for directory in map(Path,directories) for p in directory.glob(f'case_{index:03d}_*.json.gz')]
            try:
                if len(choices)!=1:raise ValueError('one saved evaluation record required')
                path=choices[0];record=io.read_json(path)
                if record.get('physical_bank')!=bank:raise ValueError('evaluation bank differs')
                row=audit_record(record,backend);row['raw_source']=dict(path=str(path),sha256=sha256(path.read_bytes()).hexdigest())
            except Exception as exc:
                row=dict(schema=SCHEMA,bank=bank,index=index,success=False,failure=type(exc).__name__+': '+str(exc),
                    methods=[dict(family=f,success=False,failure='history unavailable or invalid') for f in FAMILIES])
            io.write_json(output/f'{bank}_{index:03d}.json.gz',row,compressed=True);histories.append(row)
    slots=[method for h in histories for method in h['methods']]
    summary=dict(schema=SCHEMA,predeclared_indices=list(INDICES),history_denominator=8,method_slot_denominator=32,
        complete_histories=sum(h['success'] for h in histories),complete_slots=sum(m['success'] for m in slots),
        failed_slots=sum(not m['success'] for m in slots),actual_moved_slots=sum(bool(m.get('moved')) for m in slots),
        counterfactual_unique_fit_batches=sum(h.get('counterfactual_unique_fit_batches',0) for h in histories),
        counterfactual_fit_seconds=sum(c.get('seconds',0.) for h in histories for c in h.get('counterfactuals',{}).values()),
        no_new_training=True,no_new_EV_planning=True,no_calibrated_counterfactual_decision=True,
        tracking_error_simulated=False,source_file_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Selected eight histories retained; only moved electrical powers need refitting; nonmoving spatial terms equal zero.')
    summary['by_family']={}
    for family in FAMILIES:
        rows=[r for r in slots if r['family']==family];good=[r for r in rows if r['success']]
        moved=[r for r in good if r.get('moved')]
        summary['by_family'][family]=dict(denominator=8,complete=len(good),failed=len(rows)-len(good),
            actual_moved=sum(bool(r.get('moved')) for r in rows),complete_moved=len(moved),
            moved_space_positive=sum(r['M_space']>1e-7 for r in moved),
            moved_space_zero=sum(abs(r['M_space'])<=1e-7 for r in moved),
            moved_space_negative=sum(r['M_space']<-1e-7 for r in moved),
            mean_space_complete=(None if not good else float(np.mean([r['M_space'] for r in good]))),
            mean_curve_extra_complete=(None if not good else float(np.mean([r['M_curve_extra'] for r in good]))))
    io.write_json(output/'summary.json',summary);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--n3-directory',type=Path,action='append',required=True)
    parser.add_argument('--n4-directory',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(run(args.n3_directory,args.n4_directory,args.output),ensure_ascii=False))
