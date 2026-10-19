"""Three-event, 2^3 offline attribution of complete joint-fit score ordering.

Saved EV actions, H/V observations, candidate library and inference options
are immutable. True factors only generate noiseless future targets. No action
planning or new training is performed. Each candidate post is checkpointed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from itertools import permutations, product
import json
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np

from . import ev_dynamic_study as dynamic
from . import ev_mobile_v2_study as frozen
from . import ev_systematic_refit as refit
from . import gradient_price_study as io
from . import likelihood as lk
from .ev_mobile_v2_spatial_audit import ACBackend, verify_observations

EVENTS=(('n3',9),('n3',20),('n4',14))
METHODS=('scalar_fixed','scalar_mobile')
FACTORS=('lines','background','source')
STATES=tuple(product((0,1),repeat=3))


def fingerprint(value):
    return sha256(json.dumps(io.serial(value),sort_keys=True,separators=(',',':'),
                             allow_nan=False).encode()).hexdigest()


def shapley_change(values):
    """Mean marginal *increase*: sum equals value(111)-value(000)."""
    if set(values)!=set(STATES) or not np.isfinite(list(values.values())).all():
        raise ValueError('Eight complete finite factor states required')
    result=dict.fromkeys(FACTORS,0.)
    for order in permutations(range(3)):
        state=[0,0,0]
        for i in order:
            before=values[tuple(state)];state[i]=1
            result[FACTORS[i]]+=(values[tuple(state)]-before)/6.
    difference=float(values[(1,1,1)]-values[(0,0,0)])
    error=float(sum(result.values())-difference)
    if abs(error)>1e-9*(1+max(abs(v) for v in values.values())):
        raise ValueError('Shapley arithmetic identity failed')
    return dict(contributions=result,total_change=difference,identity_error=error)


def margin(losses,truth,expected):
    if set(losses)!=set(expected) or truth not in losses or not all(np.isfinite(v) for v in losses.values()):
        return None
    return float(min(v for k,v in losses.items() if k!=truth)-losses[truth])


def make_target(candidate,parameters,power,fm,*,base_kw,ac_predict=lk.ac_predict):
    """Clean future output, calibrated biases retained; EV enters P once."""
    power=np.asarray(power,float);p=np.asarray(parameters['p'])+power/base_kw
    q=np.asarray(parameters['q']);T,n=p.shape
    physical=ac_predict(candidate,parameters['r'],parameters['x'],p,q,
        parameters['sp'],parameters['sq'],parameters['root'])
    expand=lambda value:np.broadcast_to(np.asarray(value,float),(T,n))
    obs=lk.Observations(p=p+expand(fm.bias_p),q=q+expand(fm.bias_q),
        v=physical['v']+expand(fm.bias_v),root_v=parameters['root'],
        master_p=physical['master_p'],master_q=physical['master_q'],
        source_p=parameters['sp'] if fm.source_p_sigma is not None else None,
        source_q=parameters['sq'] if fm.source_q_sigma is not None else None)
    ranges={key:[float(np.min(value)),float(np.max(value))] for key,value in
        dict(p_total=p,q=q,voltage=physical['v'],source_p=parameters['sp'],
             source_q=parameters['sq'],r=parameters['r'],x=parameters['x']).items()}
    return obs,dict(ranges=ranges,ac_equation_residual=physical['ac_equation_residual'],
        power_balance_residual=physical['power_balance_residual'],iterations=physical['iterations'],
        future_white_noise=False,calibrated_bias_retained=True,EV_in_total_PQ_once=True)


def factor_parameters(predicted,true,bits):
    line=true if bits[0] else predicted;load=true if bits[1] else predicted;source=true if bits[2] else predicted
    return dict(r=line['r'],x=line['x'],p=load['p'],q=load['q'],sp=source['sp'],sq=source['sq'],root=predicted['root'])


def prepare(raw,backend,path):
    case=backend.regenerate(raw)
    checks=dict(training=verify_observations(raw['training'],case['train']),
                validation=verify_observations(raw['validation'],case['validation']))
    if not all(v['success'] for v in checks.values()):raise ValueError('Regenerated H/V differ from saved measurements')
    if any(not np.array_equal(raw['truth'][k],case['truth'][k]) for k in ('r','x')):
        raise ValueError('Regenerated true R/X differ')
    context=backend.restore(raw,case);truth=raw['true_name']
    if len(context['bank'])!=4:raise ValueError('Only the original four-candidate events are authorized')
    candidate=next(c for c in context['bank'] if c.name==truth)
    if candidate.supports!=case['truth']['candidate'].supports:raise ValueError('True reference supports differ')
    T=raw['config']['horizon'];sl=slice(case['future_start'],case['future_start']+T)
    fit=context['fits'][truth];forecast=raw['forecast'];source=forecast['source_by_candidate'][truth]
    predicted=dict(r=np.asarray(fit['r']),x=np.asarray(fit['x']),p=np.asarray(forecast['p']),q=np.asarray(forecast['q']),
        sp=np.asarray(source['source_p']),sq=np.asarray(source['source_q']),root=np.asarray(forecast['root']))
    true=dict(r=np.asarray(raw['truth']['r']),x=np.asarray(raw['truth']['x']),p=case['p'][sl],q=case['q'][sl],
              sp=case['source_p'][sl],sq=case['source_q'][sl],root=case['root_v'][sl])
    if not np.array_equal(predicted['root'],true['root']):raise ValueError('Root differs; three-factor attribution would omit a factor')
    methods={m['name']:m for m in raw['methods']}
    powers={name:np.asarray(methods[name]['power_kw'],float) for name in METHODS}
    noisy={name:methods[name]['inference']['joint_total']['M'] for name in METHODS}
    if not all(methods[name]['inference']['joint_total']['complete'] and noisy[name] is not None for name in METHODS):
        raise ValueError('Original noisy joint scores incomplete')
    histories={name:dict(fingerprint=h.fingerprint,training_fit=h.training_fit,
        fixed_validation_fit=h.fixed_validation_fit,fixed_pre=h.fixed_pre,fixed_state=h.fixed_state,
        joint_pre=h.joint_pre,joint_state=h.joint_state,seconds=h.seconds) for name,h in context['histories'].items()}
    primitive=dict(schema='restored_original_history_cache_v1',training=raw['training'],validation=raw['validation'],
        training_meter=raw['training_meter'],validation_meter=raw['validation_meter'],future_meter=raw['future_meter'],
        options=context['options'],histories=histories,seed=raw['seed'],candidate_names=[c.name for c in context['bank']],
        checks=checks,predicted=predicted,true=true,power_kw=powers,noisy_actual_M=noisy,
        source_interface=1,no_training_recomputed=True,raw_path=str(path.resolve()),
        raw_sha256=sha256(path.read_bytes()).hexdigest())
    return context,candidate,predicted,true,powers,primitive


def summarize(output):
    output=Path(output);events=[];attempts=complete=failed=0;fit_seconds=0.;counters={}
    for bank,index in EVENTS:
        directory=output/f'{bank}_{index:03d}';meta_path=directory/'history.json.gz'
        event=dict(bank=bank,index=index,success=False,states={},failures=[])
        if not meta_path.exists():
            event['failure']='history not prepared';events.append(event);continue
        meta=io.read_json(meta_path);names=meta['candidate_names'];truth=meta['true_name']
        values={method:{} for method in METHODS}
        for bits in STATES:
            key=''.join(map(str,bits));state={}
            for method in METHODS:
                path=directory/f'{method}_{key}.json.gz'
                if not path.exists():
                    state[method]=dict(complete=False,failure='not run');continue
                target=io.read_json(path);rows=target.get('candidates',{})
                attempts+=len(rows);complete+=sum(r.get('joint_success',False) for r in rows.values())
                failed+=sum(not r.get('joint_success',False) for r in rows.values())
                for r in rows.values():
                    fit_seconds+=r.get('seconds',0.)
                    for k,v in r.get('counters',{}).items():counters[k]=counters.get(k,0)+v
                done=bool(set(rows)==set(names) and all(r.get('joint_success') for r in rows.values()))
                losses={name:r['joint_loss'] for name,r in rows.items() if r.get('joint_loss') is not None}
                M=margin(losses,truth,names) if done else None
                fixed_done=bool(set(rows)==set(names) and all(r.get('fixed_success') for r in rows.values()))
                fixed_losses={name:r['fixed_loss'] for name,r in rows.items() if r.get('fixed_loss') is not None}
                fixed_M=margin(fixed_losses,truth,names) if fixed_done else None
                state[method]=dict(complete=done,M=M,losses=losses,fixed_M=fixed_M,
                    attained_M=margin(losses,truth,names),fixed_losses=fixed_losses,
                    candidate_denominator=len(names),attempted=len(rows))
                if M is not None:values[method][bits]=M
            if all(state[m].get('M') is not None for m in METHODS):
                state['delta_M']=state['scalar_mobile']['M']-state['scalar_fixed']['M']
            event['states'][key]=state
        if all(len(values[m])==8 for m in METHODS):
            delta={bits:values['scalar_mobile'][bits]-values['scalar_fixed'][bits] for bits in STATES}
            noisy=meta['noisy_actual_M'];noisy_delta=noisy['scalar_mobile']-noisy['scalar_fixed']
            event.update(success=True,Shapley_delta_M=shapley_change(delta),
                Shapley_M={m:shapley_change(v) for m,v in values.items()},
                original_noisy_delta_M=noisy_delta,noiseless_111_delta_M=delta[(1,1,1)],
                observed_minus_111_residual=noisy_delta-delta[(1,1,1)],
                residual_meaning='future metering noise plus changed conditional P/Q inputs; not a pure noise causal effect')
        event['history_source']=dict(path=str(meta_path),sha256=sha256(meta_path.read_bytes()).hexdigest())
        events.append(event)
    return dict(history_denominator=3,complete_histories=sum(r['success'] for r in events),
        target_denominator=48,candidate_post_denominator=192,attempted_candidate_posts=attempts,
        complete_joint_posts=complete,failed_attempted_joint_posts=failed,not_run_candidate_posts=192-attempts,
        candidate_fit_seconds=fit_seconds,counters=counters,events=events)


def render(summary,path,status):
    lines=['# 三条既有变化事件：完整AC判别分数的离线归因','',
        '仅使用N3 index9/20和N4 index14原scalar固定/移动动作。H/V计量、训练参数和推断域均保持原样。三个因素为真参考的线路、未来背景P/Q、参考源，各取训练预测或仿真真实值；未来计量无白噪声，保留已知偏置与原计量标准差。全部四候选重新拟合H+V+未来的共享线路及逐时源。','',
        r'$M(S,u)=\min_{b\ne a}L_b(H,V,E_S(u))-L_a(H,V,E_S(u))$，其中$a$只在离线定义真参考，$L$是局部取得的标准化总SSE。令$\Delta M(S)=M(S,u_M)-M(S,u_F)$。', '',
        '按六种因素纠正顺序平均分配ΔM(111)−ΔM(000)。负贡献表示改用真实因素后，移动相对固定的收益下降。原带噪实际ΔM减111无噪ΔM另列；该残余同时含未来计量噪声与条件P/Q输入改变。','',
        f"状态：{status}。完成{summary['complete_histories']}/3历史；尝试{summary['attempted_candidate_posts']}/192候选post，成功{summary['complete_joint_posts']}，尝试失败{summary['failed_attempted_joint_posts']}，未运行{summary['not_run_candidate_posts']}。",'',
        '|事件|ΔM000|ΔM111|线路贡献|背景贡献|源贡献|原带噪ΔM|残余|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for e in summary['events']:
        if not e['success']:
            lines.append(f"|{e['bank']}/{e['index']}|未完成|||||||" );continue
        c=e['Shapley_delta_M']['contributions']
        lines.append(f"|{e['bank']}/{e['index']}|{e['states']['000']['delta_M']:.6g}|{e['states']['111']['delta_M']:.6g}|{c['lines']:.6g}|{c['background']:.6g}|{c['source']:.6g}|{e['original_noisy_delta_M']:.6g}|{e['observed_minus_111_residual']:.6g}|")
    lines.extend(['','48个未来目标及各候选完整取得值、失败、状态、种子、原历史缓存和调用耗时分别保存在同目录。统计单位为三条已知事件；结果用于机制解释，不改变已冻结策略或形成独立效果评价。',''])
    Path(path).write_text('\n'.join(lines),encoding='utf-8')


def run(root,output,*,budget_seconds=480.,resume=False):
    began=perf_counter();root,output=Path(root),Path(output)
    if not np.isfinite(budget_seconds) or budget_seconds<=0:raise ValueError('Positive finite diagnostic budget required')
    if output.exists() and not resume:raise FileExistsError('Retain existing diagnostic directory; use --resume after review')
    output.mkdir(parents=True,exist_ok=True);backend=ACBackend()
    config=dict(schema='mobile_v3_complete_score_attribution_v1',events=EVENTS,methods=METHODS,factors=FACTORS,
        target_count=48,candidate_posts=192,source_sha256=backend.current_sources,
        tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),fit_seed='raw seed+1000+candidate_index, matching original driver',
        history_retrained=False,selector_called=False,calibration_changed=False)
    freeze=output/'freeze.json'
    if freeze.exists():
        if io.read_json(freeze)!=io.serial(config):raise ValueError('Diagnostic source/config changed across resume')
    else:io.write_json(freeze,config)
    status='running';stop=False;preparation_failures=[]
    with dynamic.count_physics_calls() as physics:
        for bank,index in EVENTS:
            if perf_counter()-began>=budget_seconds:status='checkpointed_budget';break
            directory=output/f'{bank}_{index:03d}';directory.mkdir(exist_ok=True)
            try:
                paths=list((root/('mobile_v2_formal_eval_'+bank)).glob(f'case_{index:03d}_*.json.gz'))
                if len(paths)!=1:raise ValueError('One original raw record required')
                path=paths[0];raw=io.read_json(path)
                context,candidate,pred,true,powers,meta=prepare(raw,backend,path)
                meta['true_name']=raw['true_name'];history_path=directory/'history.json.gz'
                if history_path.exists():
                    if fingerprint(io.read_json(history_path))!=fingerprint(meta):raise ValueError('Restored history differs on resume')
                else:io.write_json(history_path,meta,compressed=True)
                for bits in STATES:
                    state=''.join(map(str,bits));parameters=factor_parameters(pred,true,bits)
                    for method in METHODS:
                        if perf_counter()-began>=budget_seconds:status='checkpointed_budget';stop=True;break
                        file=directory/f'{method}_{state}.json.gz'
                        if file.exists():target=io.read_json(file)
                        else:
                            target=dict(method=method,state=state,factors=dict(zip(FACTORS,bits)),candidates={})
                        if set(target['candidates'])=={c.name for c in context['bank']}:continue
                        generation_began=perf_counter()
                        try:
                            future,diagnostics=make_target(candidate,parameters,powers[method],context['fm'],
                                base_kw=raw['config']['base_kw'],ac_predict=lk.ac_predict)
                            observation=io.observations_record(future);target_hash=fingerprint(observation)
                            if target.get('target_sha256') is not None and target['target_sha256']!=target_hash:
                                raise ValueError('Noiseless target changed on resume')
                            target.update(observations=observation,target_sha256=target_hash,AC=diagnostics,
                                parameters=parameters,power_kw=powers[method],target_seconds=perf_counter()-generation_began)
                            io.write_json(file,target,compressed=True)
                        except Exception as exc:
                            target.update(target_failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
                            io.write_json(file,target,compressed=True);continue
                        for j,competitor in enumerate(context['bank']):
                            if competitor.name in target['candidates']:continue
                            if perf_counter()-began>=budget_seconds:status='checkpointed_budget';stop=True;break
                            count={};started=perf_counter();seed=context['seed']+1000+j
                            result=dict(seed=seed,future_source_seed=seed+3,joint_fit_seed=seed+4,
                                joint_success=False,fixed_success=False,joint_loss=None,fixed_loss=None)
                            try:
                                comparison=refit.compare_fixed_and_joint(competitor,context['training'],context['validation'],future,
                                    context['tm'],context['vm'],context['fm'],training_fit=context['fits'][competitor.name],
                                    history_cache=context['histories'][competitor.name],fit_options=context['options'],
                                    seed=seed,call_counter=count)
                                result.update(comparison=comparison,joint_success=bool(comparison['joint']['post']['success']),
                                    fixed_success=bool(comparison['fixed']['post']['success']),
                                    joint_loss=comparison['joint']['joint_total'],fixed_loss=comparison['fixed']['fixed_total'])
                            except Exception as exc:result.update(failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
                            result.update(counters=count,seconds=perf_counter()-started)
                            target['candidates'][competitor.name]=result;io.write_json(file,target,compressed=True)
                            io.write_json(output/'status.json',dict(status='running',bank=bank,index=index,method=method,state=state,
                                last_candidate=competitor.name,elapsed_seconds=perf_counter()-began,physics=dict(physics)))
                        print(json.dumps(dict(bank=bank,index=index,method=method,state=state,
                            attempted=len(target['candidates']),seconds=perf_counter()-began),ensure_ascii=False),flush=True)
                        if stop:break
                    if stop:break
                if stop:break
            except Exception as exc:
                failure=dict(bank=bank,index=index,failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
                preparation_failures.append(failure);io.write_json(directory/'preparation_failure.json',failure)
        if status=='running':status='complete'
    summary=summarize(output)
    if status=='complete' and (summary['complete_joint_posts']!=192 or preparation_failures):
        status='finished_with_failures'
    record=dict(status=status,created_utc=datetime.now(timezone.utc).isoformat(),summary=summary,
        elapsed_seconds=perf_counter()-began,budget_seconds=budget_seconds,physics=physics,
        preparation_failures=preparation_failures,soft_budget_stop='checked between atomic candidate posts')
    io.write_json(output/'summary.json',record);io.write_json(output/'status.json',record)
    render(summary,output/'summary.md',status)
    print(json.dumps(dict(status=status,complete_histories=summary['complete_histories'],
        attempted_candidate_posts=summary['attempted_candidate_posts'],seconds=record['elapsed_seconds']),ensure_ascii=False),flush=True)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs'))
    p.add_argument('--output',type=Path,default=Path('outputs/mobile_v3_score_attribution_v1'))
    p.add_argument('--budget-seconds',type=float,default=480.);p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.root,a.output,budget_seconds=a.budget_seconds,resume=a.resume)


if __name__=='__main__':main()
