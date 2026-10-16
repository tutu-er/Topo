"""Offline AC response attribution for 8 prespecified + 3 changed v2 histories.

Only saved actions are replayed. No line/source fit, control optimization,
threshold change, action choice using truth, or new evaluation is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations, permutations, product
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import ev_dynamic_loads as dynamic
from . import ev_mobile_v2_study as study
from . import gradient_price_study as io
from . import likelihood as lk

FACTORS=('lines','background','source')
METHODS=('scalar_fixed','scalar_mobile','square_fixed','square_mobile')
INDICES=(0,12,24,36)
EXTRA=(('n3',9),('n3',20),('n4',14))
CHANNELS=('v','master_p','master_q')


def shapley_reduction(errors):
    """Allocate E(predicted)-E(true) as mean marginal correction over 6 orders."""
    if set(errors)!=set(product((0,1),repeat=3)):
        raise ValueError('all eight correction states required')
    result=dict.fromkeys(FACTORS,0.)
    for order in permutations(range(3)):
        state=[0,0,0]
        for factor in order:
            before=errors[tuple(state)];state[factor]=1
            result[FACTORS[factor]]+=(before-errors[tuple(state)])/6.
    residual=sum(result.values())-(errors[(0,0,0)]-errors[(1,1,1)])
    if abs(residual)>1e-10*(1+abs(errors[(0,0,0)])):
        raise ValueError('Shapley efficiency failed')
    return dict(contributions=result,efficiency_error=residual)


def pairwise_ranking(values,*,tol=1e-8):
    """Compare frozen predicted/true response strengths; ties remain explicit."""
    rows=[]
    for a,b in combinations(values,2):
        dp=values[a]['predicted']-values[b]['predicted']
        dt=values[a]['true']-values[b]['true']
        sign=lambda v:0 if abs(v)<=tol else 1 if v>0 else -1
        sp,st=sign(dp),sign(dt)
        category='both_tied' if sp==st==0 else 'predicted_tie' if not sp else 'true_tie' if not st else 'reversed' if sp!=st else 'same_order'
        rows.append(dict(a=a,b=b,predicted_difference=dp,true_difference=dt,category=category))
    return dict(tolerance=tol,pairs=rows,counts=dict(Counter(r['category'] for r in rows)))


def _moments(values):
    values=np.asarray(values,float)
    return dict(n=int(values.size),mean=float(np.mean(values)),median=float(np.median(values)),
                min=float(np.min(values)),max=float(np.max(values))) if values.size else dict(n=0)


def _same(saved,replayed):
    checks={}
    for name in ('p','q','v','root_v','master_p','master_q','source_p','source_q'):
        a,b=saved.get(name),getattr(replayed,name)
        if a is None or b is None:
            checks[name]=dict(success=a is None and b is None,present=False);continue
        a,b=np.asarray(a,float),np.asarray(b,float)
        checks[name]=dict(success=bool(a.shape==b.shape and np.allclose(a,b,atol=1e-12,rtol=1e-12)),
            max_absolute_error=float(np.max(abs(a-b))) if a.shape==b.shape else None,present=True)
    return dict(success=all(v['success'] for v in checks.values()),channels=checks)


def _sigma(meter,T,n):
    return dict(v=np.broadcast_to(np.asarray(meter['v_sigma'],float),(T,n)),
        **{k:np.broadcast_to(np.asarray(meter[k+'_sigma'],float),(T,)) for k in ('master_p','master_q')})


def _SSE(difference,sigma):
    result={k:float(np.sum((difference[k]/sigma[k])**2)) for k in CHANNELS}
    return dict(total=sum(result.values()),channels=result)


def _response(candidate,r,x,p,q,sp,sq,root,u,base_kw):
    physical=lk.ac_predict(candidate,r,x,p+u/base_kw,q,sp,sq,root)
    return {k:physical[k] for k in CHANNELS},dict(iterations=physical['iterations'],
        ac_equation_residual=physical['ac_equation_residual'],power_balance_residual=physical['power_balance_residual'],
        voltage_range=[float(physical['v'].min()),float(physical['v'].max())])


def audit_record(raw,*,expected_sources=None):
    began=perf_counter()
    row={k:raw[k] for k in ('physical_bank','index','seed','case_id','family_index','true_name','policy_id')}
    row.update(success=False,actions=[],failures=[],AC_response_calls=0)
    try:
        expected_sources=study.source_hashes() if expected_sources is None else expected_sources
        if raw['source_sha256']!=expected_sources:raise ValueError('frozen scientific source mismatch')
        spec=study.specification(raw['physical_bank'],'evaluation',raw['index'])
        if any(spec[k]!=raw[k] for k in spec):raise ValueError('saved history differs from frozen seed schedule')
        case=study.make_case(raw['case_id'],raw['seed'],raw['family_index'])
        candidate=next(k for k in case['bank'] if k.name==raw['true_name'])
        if candidate.supports!=case['truth']['candidate'].supports:raise ValueError('truth supports disagree')
        row['training_replay']=_same(raw['training'],case['train'])
        row['validation_replay']=_same(raw['validation'],case['validation'])
        if not row['training_replay']['success'] or not row['validation_replay']['success']:
            raise ValueError('regenerated training/validation differs')
        for k in ('r','x'):
            if not np.array_equal(raw['truth'][k],case['truth'][k]):raise ValueError('regenerated true line bytes differ')
        observed={k:getattr(case['train'],k) for k in ('p','q','v','root_v','master_p','master_q','source_p','source_q') if getattr(case['train'],k) is not None}
        shape=case['train'].p.shape
        bp=np.broadcast_to(np.asarray(case['tm'].bias_p,float),shape).copy()
        bq=np.broadcast_to(np.asarray(case['tm'].bias_q,float),shape).copy()
        replayhash=dynamic._hash(dict(observations=observed,bias_p=bp,bias_q=bq))
        if replayhash!=raw['forecast']['metadata']['training_sha256']:raise ValueError('forecast training provenance hash mismatch')
        row['training_sha256']=replayhash
        T,n=raw['config']['horizon'],case['train'].n
        sl=slice(case['future_start'],case['future_start']+T)
        true=dict(p=case['p'][sl],q=case['q'][sl],sp=case['source_p'][sl],sq=case['source_q'][sl],
                  root=case['root_v'][sl],r=case['truth']['r'],x=case['truth']['x'])
        trained=next(k for k in raw['training_result']['fits'] if k.get('candidate',k.get('name'))==raw['true_name'])
        forecast=raw['forecast'];src=forecast['source_by_candidate'][raw['true_name']]
        pred=dict(p=np.asarray(forecast['p']),q=np.asarray(forecast['q']),sp=np.asarray(src['source_p']),sq=np.asarray(src['source_q']),
                  root=np.asarray(forecast['root']),r=np.asarray(trained['r']),x=np.asarray(trained['x']))
        if not np.array_equal(pred['root'],true['root']):raise ValueError('root differs; three-factor decomposition would omit a factor')
        row['factor_inputs']=dict(predicted=pred,true=true,future_start=case['future_start'],horizon=T,
            base_kw=case['base_kw'],source_interface=1)
        sigma=_sigma(raw['future_meter'],T,n);row['sigma']=sigma
        methods={m['name']:m for m in raw['methods']};controls={name:np.asarray(methods[name]['power_kw'],float) for name in METHODS}
        baselines={};responses={};diagnostics={};cache={}
        # Generation above calls AC once for full background; the counter below
        # records the additional decomposition-response calls only.
        for bits in product((0,1),repeat=3):
            params=dict(r=(true if bits[0] else pred)['r'],x=(true if bits[0] else pred)['x'],
                p=(true if bits[1] else pred)['p'],q=(true if bits[1] else pred)['q'],
                sp=(true if bits[2] else pred)['sp'],sq=(true if bits[2] else pred)['sq'],root=true['root'])
            try:
                baselines[bits],bd=_response(candidate,**params,u=np.zeros((T,n)),base_kw=case['base_kw'])
                row['AC_response_calls']+=1
                for name,u in controls.items():
                    key=(bits,u.astype('<f8').tobytes())
                    if key not in cache:
                        cache[key]=_response(candidate,**params,u=u,base_kw=case['base_kw']);row['AC_response_calls']+=1
                    responses[bits,name],diagnostics[bits,name]=cache[key]
            except Exception as exc:
                row['failures'].append(dict(state=bits,failure=type(exc).__name__+': '+str(exc)))
        for name,u in controls.items():
            action=dict(name=name,success=False,power_kw=u,operation=methods[name].get('operation'),states={})
            try:
                if len(baselines)!=8 or any((bits,name) not in responses for bits in product((0,1),repeat=3)):
                    raise ValueError('one or more factor states failed AC')
                target=responses[(1,1,1),name];zero=baselines[(1,1,1)]
                delta_true={k:target[k]-zero[k] for k in CHANNELS}
                for bits in product((0,1),repeat=3):
                    values=responses[bits,name];delta={k:values[k]-baselines[bits][k] for k in CHANNELS}
                    difference={k:delta[k]-delta_true[k] for k in CHANNELS}
                    full={k:values[k]-target[k] for k in CHANNELS}
                    strength=_SSE(delta,sigma);key=''.join(map(str,bits))
                    action['states'][key]=dict(increment_error=_SSE(difference,sigma),full_response_error=_SSE(full,sigma),
                        increment_strength=strength,AC=diagnostics[bits,name])
                action['attribution']={}
                for metric in ('increment_error','full_response_error'):
                    component={}
                    for channel in ('total',)+CHANNELS:
                        errors={tuple(map(int,key)):val[metric]['total'] if channel=='total' else val[metric]['channels'][channel] for key,val in action['states'].items()}
                        component[channel]=shapley_reduction(errors)
                    action['attribution'][metric]=component
                action.update(success=True,true_increment=delta_true,
                    predicted_increment={k:responses[(0,0,0),name][k]-baselines[(0,0,0)][k] for k in CHANNELS},
                    relative_increment_error=(action['states']['000']['increment_error']['total']/action['states']['111']['increment_strength']['total'] if action['states']['111']['increment_strength']['total']>0 else None))
            except Exception as exc:action['failure']=type(exc).__name__+': '+str(exc)
            row['actions'].append(action)
        good=[a for a in row['actions'] if a['success']]
        row['ranking']={}
        if len(good)==len(METHODS):
            for channel in ('total',)+CHANNELS:
                values={a['name']:dict(predicted=(a['states']['000']['increment_strength']['total'] if channel=='total' else a['states']['000']['increment_strength']['channels'][channel]),
                    true=(a['states']['111']['increment_strength']['total'] if channel=='total' else a['states']['111']['increment_strength']['channels'][channel])) for a in good}
                row['ranking'][channel]=pairwise_ranking(values)
        row['success']=len(good)==len(METHODS) and not row['failures']
    except Exception as exc:row['failure']=type(exc).__name__+': '+str(exc)
    row['seconds']=perf_counter()-began
    return row


def run(root,output):
    root,output=Path(root),Path(output)
    if output.exists() or output.with_suffix('.md').exists():raise FileExistsError('preserve prior diagnostic outputs')
    slots=[(bank,index,'predeclared_representative') for bank in ('n3','n4') for index in INDICES]
    slots.extend((bank,index,'observed_decision_change_explanation') for bank,index in EXTRA)
    current=study.source_hashes();rows=[]
    for bank,index,role in slots:
        paths=list((root/('mobile_v2_formal_eval_'+bank)).glob(f'case_{index:03d}_*.json.gz'))
        try:
            if len(paths)!=1:raise ValueError('one saved raw record required')
            path=paths[0];content=path.read_bytes();raw=io.read_json(path)
            row=audit_record(raw,expected_sources=current)
            row.update(role=role,raw_source=dict(path=str(path.resolve()),bytes=len(content),sha256=sha256(content).hexdigest()))
        except Exception as exc:row=dict(physical_bank=bank,index=index,role=role,success=False,actions=[],failure=type(exc).__name__+': '+str(exc))
        rows.append(row)
        print(json.dumps(dict(bank=bank,index=index,success=row['success'],seconds=row.get('seconds')),ensure_ascii=False),flush=True)
    actions=[(r,a) for r in rows for a in r['actions']];complete=[(r,a) for r,a in actions if a['success']]
    summary=dict(history_denominator=11,complete_histories=sum(r['success'] for r in rows),
        action_denominator=44,complete_actions=len(complete),failed_actions=44-len(complete),
        AC_response_calls=sum(r.get('AC_response_calls',0) for r in rows),
        no_parameter_optimization=True,no_EV_planning=True,truth_used_only_for_offline_attribution=True)
    summary['by_method']={}
    for name in METHODS:
        selected=[a for r,a in complete if a['name']==name]
        summary['by_method'][name]=dict(denominator=11,complete=len(selected),
            increment_error=_moments([a['states']['000']['increment_error']['total'] for a in selected]),
            full_response_error=_moments([a['states']['000']['full_response_error']['total'] for a in selected]),
            Shapley={f:_moments([a['attribution']['increment_error']['total']['contributions'][f] for a in selected]) for f in FACTORS})
    payload=dict(schema='mobile_v3_offline_three_factor_response_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        scope='11 fixed diagnostic histories, four saved actions each; no policy performance or probability claim.',
        factor_order=list(FACTORS),state_bits='1 replaces this factor by simulator truth; 000 current forecast, 111 full truth.',
        equations=dict(increment='Delta_mu_S(u)=mu(theta_S,PQ_S+u,source_S)-mu(theta_S,PQ_S,source_S)',
            error='E(S,u)=||W[Delta_mu_S(u)-Delta_mu_111(u)]||^2',
            full='E_full(S,u)=||W[mu_S(u)-mu_111(u)]||^2',
            Shapley='phi_i=mean over 6 factor orders of E(before correcting i)-E(after correcting i); sum(phi)=E(000)-E(111).',
            strength='||W Delta_mu_S(u)||^2: AC response size only, not competing-topology separation.'),
        root_rule='Root identical at 1pu in all states; otherwise diagnostic fails rather than omits a factor.',
        whitening='Same saved future V/rootP/rootQ standard deviations; no new meter noise or bias.',
        exception_rule='Keep all 11 histories and 44 method slots; missing or failed states block only their attribution.',
        action_selection='All four actions copied from raw v2; no action is chosen with any true factor.',
        source_sha256=current,tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),summary=summary,histories=rows)
    io.write_json(output,payload)
    render(payload,output.with_suffix('.md'))
    return payload


def render(data,target):
    lines=['# 已保存 EV 动作的三因素 AC 响应分解','',
        '固定使用两银行 index 0/12/24/36 共八条预声明代表，加三条已观察到判别变化的解释例，共 11 条。每条仅重放 scalar/square 的固定与移动四条已保存动作。真实因素仅作离线归因；没有优化 EV、线路或源。三条解释例不能作为新方法效果样本。','',
        '令 $\mu$ 是真候选结构上的 AC 末端电压与根 P/Q；$u$ 是已保存净功率。定义增量 $\Delta\mu(u)=\mu(u)-\mu(0)$。在相同负荷和源下相减，固定计量偏置被消去。将线路、背景负荷、参考源各取训练预测值或仿真真实值，得到八种响应。','',
        '$$E(S,u)=\|W[\Delta\mu_S(u)-\Delta\mu_{111}(u)]\|^2.$$',
        '',
        '$S$ 的三位依次为线路、背景 P/Q、源 P/Q；1 表示用真实值纠正该因素，0 表示保留训练预测。$W$ 使用原未来计量标准差。根电压在所有情形相同。完整响应误差另存，主归因使用增量误差。',
        '',
        '每个因素的贡献取六种纠正顺序中误差减少量的平均值（Shapley 分摊）。三者之和严格等于原预测增量误差，因为全部纠正后误差为零；负贡献表示该因素单独纠正可能破坏原有误差抵消。该分摊解释当前模型误差，未作独立因果结论。','',
        f"完成 {data['summary']['complete_histories']}/11 条历史、{data['summary']['complete_actions']}/44 个动作；额外 AC 响应调用 {data['summary']['AC_response_calls']} 次（轨迹重建本身的 AC 生成另计）。",'',
        '|银行/index/场景|方法|原增量误差|线路贡献|负荷贡献|源贡献|误差/真实响应平方|','|---|---|---:|---:|---:|---:|---:|']
    for row in data['histories']:
        for a in row['actions']:
            if not a['success']:
                lines.append(f"|{row['physical_bank']}/{row['index']}/{row.get('case_id','unknown')}|{a['name']}|失败：{a.get('failure')}|||||");continue
            e=a['states']['000']['increment_error']['total'];v=a['attribution']['increment_error']['total']['contributions']
            ratio=a['relative_increment_error']
            lines.append(f"|{row['physical_bank']}/{row['index']}/{row['case_id']}|{a['name']}|{e:.6g}|{v['lines']:.6g}|{v['background']:.6g}|{v['source']:.6g}|{ratio:.6g}|")
    lines.extend(['','响应大小的动作排序、V/rootP/rootQ 分通道误差、全部八种替换值和数组、训练/验证逐通道复现、训练预测来源 SHA256 均保存在 JSON。响应大小表示电气变化幅度，拓扑间距离仍需同时比较竞争模型。下一步应依据这份诊断提出策略，再独立冻结与评价。',''])
    Path(target).write_text('\n'.join(lines),encoding='utf8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('outputs'))
    parser.add_argument('--output',type=Path,default=Path('outputs/mobile_v3_error_decomposition.json'))
    args=parser.parse_args();answer=run(args.root,args.output)
    print(json.dumps(answer['summary'],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
