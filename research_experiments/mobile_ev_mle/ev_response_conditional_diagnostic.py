"""Development-only latent-state conditional future integration.

No policy is selected or changed. Exactly four existing development histories
are read; formal calibration/evaluation directories are never opened. The true
training-end latent DGP state is oracle conditioning information used only by
this independent mechanism diagnostic.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import fields
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import ev_dynamic_loads as loads
from . import ev_dynamic_study as dynamic
from . import ev_fixed_train_study as fixed_study
from . import ev_fixed_train_design as fixed
from . import gradient_price_study as io
from .information_context import meter_from_record
from .likelihood import FitResult,RadialCandidate
from . import likelihood as lk

HERE=Path(__file__).parent
CONFIG=dict(version='development_latent_conditioned_future_integral_v1',
    histories=[994101,994102,994103,994104],paths_per_history=32,
    history_root_seeds=[995901,995902,995903,995904],
    methods=['square','nearest','gradac','contrast_joint','contrast_box','contrast_load'],
    source_fit_restart_seed_rule='history root seed + 10000; fixed across paths and actions',
    future_measurement_noise=False,training_lines_fixed=True,validation_losses_fixed=True,
    conditioning='true training-end common/individual/PF/source latent states; no validation conditioning',
    statistic_unit='four independent histories; conditional paths are numerical integration only',
    selector_use_allowed=False,formal_calibration_use_allowed=False)


def _hash(value):
    return hashlib.sha256(json.dumps(io.serial(value),sort_keys=True,separators=(',',':'),
        allow_nan=False).encode()).hexdigest()


def stream_specs(config,n):
    return dict(common=(np.array([0.]),config['common_sd_kw'],config['common_phi']),
        individual=(np.zeros(n),config['individual_sd_kw'],config['individual_phi']),
        pf=(np.full(n,config['pf_mean']),config['pf_sd'],config['pf_phi']),
        source_p=(np.array([config['source_p_mean_kw']]),config['source_p_sd_kw'],config['source_p_phi']),
        source_q=(np.array([config['source_q_mean_kvar']]),config['source_q_sd_kvar'],config['source_q_phi']))


def physical_from_latent(latent,config):
    means=np.resize(np.asarray(config['mean_p_kw'],float),latent['individual'].shape[1])
    raw_p=means+latent['common']+latent['individual']
    p=np.maximum(raw_p,config['positive_floor_kw'])
    pf=np.clip(latent['pf'],*config['pf_bounds'])
    sp=np.maximum(latent['source_p'][:,0],0.)
    sq=np.maximum(latent['source_q'][:,0],0.)
    return dict(p_kw=p,q_kvar=p*np.tan(np.arccos(pf)),pf=pf,
        source_p_kw=sp,source_q_kvar=sq,root_v=np.full(len(p),config['root_v']),
        clips=dict(p=int(np.count_nonzero(p!=raw_p)),
            pf=int(np.count_nonzero(pf!=latent['pf'])),
            source_p=int(np.count_nonzero(sp!=latent['source_p'][:,0])),
            source_q=int(np.count_nonzero(sq!=latent['source_q'][:,0]))))


def reconstruct_prefix(record):
    """Exactly reproduce training physical prefix using saved child seeds."""
    background=record['background'];meta=background['metadata'];config=meta['config']
    H=len(record['training']['p']);n=len(record['training']['p'][0])
    latent={}
    for name,(mean,sd,phi) in stream_specs(config,n).items():
        latent[name]=loads._stationary_ar1(np.random.default_rng(meta['child_seeds'][name]),
            H,mean,sd,phi,len(mean))
    physical=physical_from_latent(latent,config);checks={}
    for name in ('p_kw','q_kvar','pf','source_p_kw','source_q_kvar','root_v'):
        saved=np.asarray(background[name],float)[:H]
        checks[name]=dict(exact=bool(np.array_equal(physical[name],saved)),
            max_abs=float(np.max(abs(physical[name]-saved),initial=0.)),
            shape=list(saved.shape))
    if not all(row['exact'] for row in checks.values()):
        raise ValueError('saved training DGP prefix failed exact reconstruction: '+str(checks))
    state={name:values[-1].copy() for name,values in latent.items()}
    return dict(config=config,H=H,n=n,state=state,state_sha256=_hash(state),
        checks=checks,physical_prefix_sha256=_hash({key:physical[key] for key in checks}),
        child_seeds=meta['child_seeds'])


def conditional_moments(state,mean,sd,phi,steps):
    h=np.asarray(steps,float)
    state=np.asarray(state,float);mean=np.asarray(mean,float)
    return mean+phi**h[:,None]*(state-mean),sd**2*(1-phi**(2*h))


def continue_latent(state,config,n,steps,stream_seeds):
    if int(steps)!=steps or steps<1:raise ValueError('positive integer steps required')
    result={}
    for name,(mean,sd,phi) in stream_specs(config,n).items():
        rng=np.random.default_rng(stream_seeds[name])
        innovations=rng.standard_normal((steps,len(mean)))*sd*np.sqrt(1-phi**2)
        values=np.empty_like(innovations);current=np.asarray(state[name],float).copy()
        for h in range(steps):
            current=mean+phi*(current-mean)+innovations[h];values[h]=current
        result[name]=values
    return result


def conditional_path(prefix,root_seed,path_index,paths,gap,horizon):
    root=np.random.SeedSequence(int(root_seed));child=root.spawn(int(paths))[int(path_index)]
    names=list(stream_specs(prefix['config'],prefix['n']))
    seeds={name:int(s.generate_state(1)[0]) for name,s in zip(names,child.spawn(len(names)))}
    latent=continue_latent(prefix['state'],prefix['config'],prefix['n'],gap+horizon,seeds)
    full=physical_from_latent(latent,prefix['config'])
    future={name:values[gap:].copy() for name,values in full.items() if name!='clips'}
    return future,dict(root_seed=int(root_seed),path_index=int(path_index),stream_seeds=seeds,
        first_future_step=gap+1,gap=gap,horizon=horizon,clips_all_generated_steps=full['clips'],
        physical_future_sha256=_hash(future))


def mc_stats(values,planned):
    finite=[float(x) for x in values if x is not None and np.isfinite(x)]
    n=len(finite);mean=float(np.mean(finite)) if finite else None
    sd=float(np.std(finite,ddof=1)) if n>1 else None
    return dict(planned=planned,complete=n,failed=planned-n,mean=mean,
        sample_sd=sd,MC_standard_error=None if sd is None else sd/np.sqrt(n),
        minimum=min(finite) if finite else None,maximum=max(finite) if finite else None,
        mean_scope='all fixed paths' if n==planned else 'successful paths only; failed paths retained',
        complete_path_integral_available=n==planned,independent_history_count=1)


def restore_models(record):
    bank=[]
    for row in record['candidates']:
        family=[tuple(s) for s in row['supports'] if 1<len(s)<3]
        candidate=RadialCandidate.from_supports(row['name'],3,family)
        if candidate.supports!=tuple(tuple(s) for s in row['supports']):raise ValueError('edge order changed')
        bank.append(candidate)
    if [c.name for c in bank]!=['star','clade_01']:raise ValueError('strict declared pair required')
    fitted={}
    for row in record['training_result']['fits']:
        data={field.name:row[field.name] for field in fields(FitResult)}
        for key in ('r','x','p','q','source_p','source_q','bias_p','bias_q','bias_v'):
            data[key]=np.asarray(data[key],float)
        data['predicted']={key:np.asarray(val) for key,val in data['predicted'].items()}
        fitted[row['candidate']]=FitResult(**data)
    val={row['candidate']:float(row['loss']) for row in record['validation_result']['fits']}
    if set(val)!=set(fitted) or not record['validation_result']['all_complete']:
        raise ValueError('complete original validation source fits required')
    return bank,fitted,val,meter_from_record(record['future_meter'])


def gather_actions(record):
    raw={row['name']:row for row in record['methods']};actions={};aliases={};original={}
    for name in CONFIG['methods']:
        row=raw[name];power=np.asarray(row['power_kw'],float)
        if power.shape!=(6,3) or not np.isfinite(power).all():raise ValueError('executed finite action required')
        key=hashlib.sha256(fixed._control_key(power)).hexdigest()
        actions.setdefault(key,power);aliases.setdefault(key,[]).append(name)
        diagnostic=record['planning_diagnostics']['methods'][name]
        original[name]=dict(curve_id=key,original_clean_M=row.get('clean_future_M'),
            original_noisy_M=row.get('M'),point_D=diagnostic.get('future_profile_distance'),
            joint=diagnostic.get('contrast_joint',{}).get('score'),
            box=diagnostic.get('contrast_box',{}).get('score'),
            load=diagnostic.get('contrast_load',{}).get('score'))
    return actions,aliases,original


def evaluate_action(future,power,record,bank,fitted,val,meter,*,seed):
    """Complete noiseless future AC response and original fixed-line source fits."""
    began=perf_counter();truth=record['truth'];true_name=record['true_name']
    candidate=next(c for c in bank if c.name==true_name)
    try:
        physical=lk.ac_predict(candidate,np.asarray(truth['r']),np.asarray(truth['x']),
            future['p_kw']/20.+power/20.,future['q_kvar']/20.,
            future['source_p_kw']/20.,future['source_q_kvar']/20.,future['root_v'])
        white={key:np.zeros_like(physical[key]) for key in dynamic.base.CHANNELS}
        observed=dynamic.base.observation(physical,meter,white)
        fits,block=fixed_study.fit_bank(bank,observed,meter,trained=fitted,
            options={**record['config']['profile'],**record['config']['source']},seed=seed)
        unchanged=all(np.array_equal(fits[name].r,fitted[name].r) and
            np.array_equal(fits[name].x,fitted[name].x) for name in fits)
        losses={name:val[name]+fit.loss for name,fit in fits.items()}
        complete=block['all_complete'] and len(fits)==2 and unchanged
        M=float(next(v for name,v in losses.items() if name!=true_name)-losses[true_name]) if complete else None
        return dict(success=complete,M=M,loss_by_candidate=losses,source_fits=block,
            line_parameters_unchanged=unchanged,seconds=perf_counter()-began,
            physical={key:physical[key] for key in ('p','q','v','root_v','master_p','master_q')},
            observations=io.observations_record(observed))
    except (ValueError,RuntimeError,AssertionError,FloatingPointError,np.linalg.LinAlgError) as exc:
        return dict(success=False,M=None,failure=type(exc).__name__+':'+str(exc),seconds=perf_counter()-began)


def _sign(value,tol=1e-6):
    return None if value is None else 1 if value>tol else -1 if value < -tol else 0


def run_history(record,*,root_seed):
    began=perf_counter();prefix=reconstruct_prefix(record);bank,fitted,val,meter=restore_models(record)
    actions,aliases,original=gather_actions(record);N=CONFIG['paths_per_history']
    line_ids={name:fixed._line_id(fit.r,fit.x) for name,fit in fitted.items()}
    paths=[];gap=record['config']['forecast_gap'];horizon=record['config']['horizon']
    with dynamic.count_physics_calls() as counter:
        for index in range(N):
            future,evidence=conditional_path(prefix,root_seed,index,N,gap,horizon)
            values={key:evaluate_action(future,u,record,bank,fitted,val,meter,
                seed=int(root_seed)+10000) for key,u in actions.items()}
            paths.append(dict(path_index=index,generation=evidence,future=future,actions=values))
    out=dict(seed=record['seed'],true_name=record['true_name'],prefix=prefix,
        root_seed=root_seed,paths=paths,aliases=aliases,original=original,methods={},paired=[],
        actual_solver_calls=counter,planned_paths=N,unique_actions=len(actions),
        validation_losses=val,scope=CONFIG['conditioning'],seconds=perf_counter()-began,
        lines_unchanged=all(line_ids[name]==fixed._line_id(fit.r,fit.x) for name,fit in fitted.items()))
    for name,row in original.items():
        key=row['curve_id'];values=[p['actions'][key]['M'] for p in paths]
        out['methods'][name]=dict(row,conditional_M=mc_stats(values,N))
    keys=list(actions)
    for i,left in enumerate(keys):
        for right in keys[:i]:
            left_name,right_name=aliases[left][0],aliases[right][0]
            values=[]
            for p in paths:
                a,b=p['actions'][left],p['actions'][right]
                values.append(a['M']-b['M'] if a['success'] and b['success'] else None)
            stats=mc_stats(values,N);a,b=original[left_name],original[right_name]
            single=None if a['original_clean_M'] is None or b['original_clean_M'] is None else a['original_clean_M']-b['original_clean_M']
            pair=dict(left=aliases[left],right=aliases[right],paired_differences=values,
                conditional_delta=stats,original_clean_delta=single,
                original_single_vs_conditional_mean_agree=_sign(single)==_sign(stats['mean']),
                predictive_comparisons={})
            for metric in ('point_D','joint','box','load'):
                delta=None if a[metric] is None or b[metric] is None else a[metric]-b[metric]
                pair['predictive_comparisons'][metric]=dict(delta=delta,
                    agrees_original_single=None if delta is None or single is None else _sign(delta)==_sign(single),
                    agrees_conditional_mean=None if delta is None or stats['mean'] is None else _sign(delta)==_sign(stats['mean']))
            out['paired'].append(pair)
    return out


def summarize(histories):
    rows=[]
    for history in histories:
        local=dict(seed=history['seed'],unique_actions=history['unique_actions'],planned_paths=history['planned_paths'],
            methods=history['methods'],pairs=history['paired'],seconds=history['seconds'],
            failed_action_path_slots=sum(not action['success'] for p in history['paths'] for action in p['actions'].values()),
            expected_action_path_slots=history['unique_actions']*history['planned_paths'])
        rows.append(local)
    return dict(config=CONFIG,independent_histories=len(histories),histories=rows,
        interpretation='MC errors concern conditional integration at a true latent state; no new independent histories',
        policy_changed=False,used_for_calibration=False,formal_records_read=False)


def render(summary):
    lines=['# 已固定动作的条件未来积分诊断','',
        '4份独立开发历史，每份32条条件未来；真实训练末底层状态仅用于离线机制分析。',
        '固定原验证损失和训练R/X，所有动作共享每条未来路径；未来计量噪声去除。','']
    fmt=lambda value:'—' if value is None else format(value,'.6g')
    for history in summary['histories']:
        lines.extend([f"## {history['seed']}",'',
            f"去重动作 {history['unique_actions']}，拟合失败 {history['failed_action_path_slots']}/{history['expected_action_path_slots']}。",'',
            '|策略|原单路径clean M|条件M均值|积分标准误|完整路径|','|---|---:|---:|---:|---:|'])
        for name,row in history['methods'].items():
            s=row['conditional_M'];lines.append(f"|{name}|{fmt(row['original_clean_M'])}|{fmt(s['mean'])}|{fmt(s['MC_standard_error'])}|{s['complete']}/{s['planned']}|")
        lines.extend(['','|动作对(左减右)|原clean差|条件差均值|积分标准误|D排序：单路径/均值|','|---|---:|---:|---:|---|'])
        for pair in history['pairs']:
            s=pair['conditional_delta'];d=pair['predictive_comparisons']['point_D']
            lines.append('|'+','.join(pair['left'])+' − '+','.join(pair['right'])+'|'+
                '|'.join([fmt(pair['original_clean_delta']),fmt(s['mean']),fmt(s['MC_standard_error']),
                    str(d['agrees_original_single'])+'/'+str(d['agrees_conditional_mean'])])+'|')
        lines.append('')
    lines.extend(['32路径是数值积分，不增加独立历史样本量。条件包含真实底层状态，',
        '与部署时仅按带误差计量进行预测不同。结果不进入当前策略、正式校准或独立评价。',''])
    return '\n'.join(lines)


def run(directory,output):
    directory,output=Path(directory),Path(output);output.mkdir(parents=True,exist_ok=True)
    frozen=io.read_json(directory/'freeze.json');sources=dict(frozen['source_sha256'])
    for name in ('ev_response_conditional_diagnostic.py','test_ev_response_conditional_diagnostic.py',
                 'ev_response_conditional_diagnostic.md'):
        sources[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    for name,expected in sources.items():
        path=(HERE.parents[1]/name) if name.startswith('rnj_wzzt_core/') else HERE/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('frozen source differs: '+name)
    inputs={}
    for seed in CONFIG['histories']:
        path=directory/f'trial_{seed}.json.gz'
        inputs[str(seed)]=dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest=dict(config=CONFIG,source_sha256=sources,input_files=inputs,
        original_policy_id=frozen['policy_id'],diagnostic_id=_hash(dict(config=CONFIG,sources=sources,inputs=inputs)))
    freeze_path=output/'freeze.json'
    if freeze_path.exists():raise ValueError('existing diagnostic directory retained; use another output')
    io.write_json(freeze_path,manifest)
    snapshot={name:base64.b64encode(((HERE.parents[1]/name) if name.startswith('rnj_wzzt_core/') else HERE/name).read_bytes()).decode()
        for name in sources}
    io.write_json(output/'source_snapshot.json.gz',dict(source_sha256=sources,source_base64=snapshot),compressed=True)
    histories=[]
    for seed,root in zip(CONFIG['histories'],CONFIG['history_root_seeds']):
        record=io.read_json(inputs[str(seed)]['path'])
        if record['seed']!=seed or record['policy_id']!=frozen['policy_id']:raise ValueError('unexpected development input')
        history=run_history(record,root_seed=root)
        io.write_json(output/f'conditional_{seed}.json.gz',history,compressed=True);histories.append(history)
        io.write_json(output/'checkpoint.json',dict(completed_histories=len(histories),planned_histories=4,last_seed=seed,
            diagnostic_id=manifest['diagnostic_id']))
        print(json.dumps(dict(seed=seed,seconds=history['seconds'],unique_actions=history['unique_actions'],
            means={name:row['conditional_M']['mean'] for name,row in history['methods'].items()})),flush=True)
    summary=summarize(histories);summary['manifest']=manifest
    io.write_json(output/'summary.json',summary);(output/'summary.md').write_text(render(summary),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.directory,args.output)


if __name__=='__main__':main()
