"""Offline free-tangent versus saved AC-profile audit, without optimization.

Only development records and already AC-scored projection proposals are used.
The free LS solution uses the same column-normalized numerical span as the
projection proxy. It is one normalized-minimum-norm representative when the
nuisance matrix is rank deficient, not a bounded physical parameter estimate.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np

from . import ev_fixed_train_design as fixed
from . import ev_mobile_cases_v2 as cases
from . import ev_mobile_v3_ac as ac
from . import ev_mobile_v3_projection as proxy
from . import ev_original_review as independent
from . import gradient_price_study as io
from . import likelihood as lk

HERE=Path(__file__).resolve().parent
METHODS=('projection_fixed','projection_mobile')


class FreeLS:
    def __init__(self,J):
        self.J=np.asarray(J,float);self.scale=np.linalg.norm(self.J,axis=0)
        nz=self.scale>0;self.nonzero=nz
        normalized=self.J[:,nz]/self.scale[nz]
        self.U,self.s,self.Vh=np.linalg.svd(normalized,full_matrices=False)
        self.rank,self.threshold=proxy._rank(self.s,normalized.shape,None,0.)

    def solve(self,target):
        target=np.asarray(target,float)
        scaled=-(self.Vh[:self.rank].T/self.s[:self.rank])@(self.U[:,:self.rank].T@target) if self.rank else np.zeros(sum(self.nonzero))
        z=np.zeros(len(self.scale));z[self.nonzero]=scaled/self.scale[self.nonzero]
        residual=target+self.J@z
        return z,residual,float(.5*(residual@residual))


def channel_squares(vector,H,T,n):
    vector=np.asarray(vector,float);answer={};start=0
    for label,count in (('H',H),('E',T)):
        values={};offset=start
        for channel,width in (('v',n),('master_p',1),('master_q',1)):
            values[channel]=float(np.sum(vector[offset:offset+count*width]**2));offset+=count*width
        answer[label]=dict(total=sum(values.values()),channels=values);start=offset
    if start!=len(vector):raise ValueError('channel-major history/future residual size mismatch')
    return dict(total=float(vector@vector),blocks=answer)


def shifted_state(context,b,z):
    line=context['lines'][b.name];fit=context['train_fits'][b.name]
    H=len(fit.source_p);T=context['control_shape'][0];m=b.m
    if len(z)!=2*m+2*H+2*T:raise ValueError('nuisance coordinate order mismatch')
    h=z[2*m:2*m+2*H].reshape(H,2);e=z[2*m+2*H:].reshape(T,2)
    return dict(r=line['r']+z[:m],x=line['x']+z[m:2*m],
        source_H=np.column_stack((fit.source_p,fit.source_q))+h,
        source_E=np.column_stack((context['forecasts'][b.name]['source_p'],context['forecasts'][b.name]['source_q']))+e)


def domain_check(state,cfg):
    result={};maximum={};tol=1e-10
    def check(name,values,low,high):
        a=np.asarray(values,float);low=np.broadcast_to(low,a.shape);high=np.broadcast_to(high,a.shape)
        result[name]=int(np.sum((a<low-tol)|(a>high+tol)))
        maximum[name]=float(max(0.,np.max(low-a,initial=0.),np.max(a-high,initial=0.)))
    check('R_box',state['r'],*cfg['training']['r_bounds']);check('X_box',state['x'],*cfg['training']['x_bounds'])
    low,high=cfg['ratio_bounds'];check('X_R_cone',state['x'],low*state['r'],high*state['r'])
    for label in ('H','E'):
        check('source_'+label+'_P',state['source_'+label][:,0],*cfg['source']['source_p_bounds'])
        check('source_'+label+'_Q',state['source_'+label][:,1],*cfg['source']['source_q_bounds'])
    return dict(feasible=not any(result.values()),violations=result,max_absolute_violation=maximum,tolerance=tol)


def _white(physical,meter):
    return independent.whiten(physical,meter,('v','master_p','master_q'))


def nonlinear_residual(context,a,b,u,state):
    """Separate branch-current AC; no fitting and no latent future inputs."""
    args=context['_rebuild_inputs'];Hobs=args['training'];tm=args['train_meter'];fm=context['future_meter']
    pH=Hobs.p-fixed._broadcast(tm.bias_p,Hobs.p.shape,'P bias');qH=Hobs.q-fixed._broadcast(tm.bias_q,Hobs.q.shape,'Q bias')
    pE=context['background']['p']+u/context['base_kw'];qE=context['background']['q']
    h=independent.forward_physics(b,state['r'],state['x'],pH,qH,state['source_H'][:,0],state['source_H'][:,1],Hobs.root_v)
    e=independent.forward_physics(b,state['r'],state['x'],pE,qE,state['source_E'][:,0],state['source_E'][:,1],context['root_v'])
    line=context['lines'][a.name];s=context['forecasts'][a.name]
    ref=independent.forward_physics(a,line['r'],line['x'],pE,qE,s['source_p'],s['source_q'],context['root_v'])
    observed=SimpleNamespace(**{k:getattr(Hobs,k) for k in ('v','master_p','master_q')})
    observed_dict={k:getattr(observed,k) for k in ('v','master_p','master_q')}
    white_observed=independent.whiten(observed_dict,tm,('v','master_p','master_q'),observation=True)
    r=np.r_[_white(h,tm)-white_observed,_white(e,fm)-_white(ref,fm)]
    return r,dict(AC_equation=max(v['equation_residual'] for v in (h,e,ref)),balance=max(v['balance_residual'] for v in (h,e,ref)),
        voltage_min=min(v['voltage_min'] for v in (h,e,ref)),voltage_max=max(v['voltage_max'] for v in (h,e,ref)))


def build_context(record,parent):
    bank=cases.candidate_bank('remote8' if record['spec']['bank']=='n3' else 'twinblock12')
    fits={}
    for saved in record['training_result']['fits']:
        row=dict(saved)
        for key in ('r','x','source_p','source_q','p','q'):row[key]=np.asarray(row[key],float)
        fits[row['candidate']]=SimpleNamespace(**row)
    for b in bank:
        if tuple(map(tuple,fits[b.name].supports))!=b.supports:raise ValueError('training candidate support/edge order mismatch')
    tr=lk.Observations(**record['training']);tm=lk.MeasurementModel(**record['training_meter']);fm=lk.MeasurementModel(**record['future_meter'])
    ctx=fixed.build_fixed_contexts(tr,bank,fits,tm,fm,-fixed.CONFIG['command_power_kw'],fixed.CONFIG['command_power_kw'],parent,
        base_kw=record['config']['base_kw'],forecast=record['forecast'],**record['config']['source'])
    if not ctx['success']:raise ValueError(ctx['diagnostics']['failure'])
    return ctx


def _aggregate(values):
    grouped=defaultdict(list)
    for (a,b),v in values.items():grouped[a].append((b,v))
    nearest={a:dict(b=min(v,key=lambda p:p[1])[0],half_gap=min(v,key=lambda p:p[1])[1]) for a,v in grouped.items()}
    return dict(score=float(np.mean([v['half_gap'] for v in nearest.values()])),nearest=nearest)


def audit_proposal(context,record,name,step,before_u,before_profile):
    began=perf_counter();proposal=np.asarray(step['master']['power_kw'],float);H=context['_rebuild_inputs']['training'].t;T,n=proposal.shape
    data=ac.geometry_inputs(context,before_u);bank={b.name:b for b in context['bank']};solvers={}
    for row in data['own']:solvers[row['b']]=FreeLS(row['J'])
    own={row['a']:solvers[row['b']].solve(row['residual'])[2] for row in data['own']}
    ac_before={(p['a'],p['b']):p['half_gap'] for p in before_profile['pairs']}
    ac_after={(p['a'],p['b']):p['half_gap'] for p in step['AC_profile']['pairs']}
    expected=set((r['a'],r['b']) for r in data['pairs'])
    if set(ac_before)!=expected or set(ac_after)!=expected:raise ValueError('complete AC pair bank required')
    pairs=[];values_before={};values_after={};du=(proposal-before_u).ravel()
    for row in data['pairs']:
        a,b=row['a'],row['b'];solver=solvers[b];entry=dict(a=a,b=b,states={})
        for label,delta,power in (('parent',np.zeros_like(du),before_u),('proposal',du,proposal)):
            target=row['residual']+row['B']@delta;z,residual,q=solver.solve(target);signed=q-own[a]
            (values_before if label=='parent' else values_after)[a,b]=signed
            state=shifted_state(context,bank[b],z);domain=domain_check(state,record['config']);base=context['lines'][b]
            theta=np.r_[base['r'],base['x']];m=bank[b].m
            result=dict(q_half_SSE=q,own_baseline_half_SSE=own[a],signed_gap=signed,
                saved_AC_gap=(ac_before if label=='parent' else ac_after)[a,b],linear_residual=channel_squares(residual,H,T,n),
                nuisance_rank=solver.rank,nuisance_columns=solver.J.shape[1],rank_threshold=solver.threshold,
                delta_theta_relative_L2=float(np.linalg.norm(z[:2*m])/np.linalg.norm(theta)),
                delta_R_relative_L2=float(np.linalg.norm(z[:m])/np.linalg.norm(base['r'])),
                delta_X_relative_L2=float(np.linalg.norm(z[m:2*m])/np.linalg.norm(base['x'])),
                nuisance_step=z,domain=domain,independent_nonlinear_check=dict(attempted=False))
            if domain['feasible']:
                try:
                    nonlinear,physical=nonlinear_residual(context,bank[a],bank[b],power,state)
                    check=dict(attempted=True,success=True,physical=physical,residual=channel_squares(nonlinear,H,T,n),
                        linearization_error=channel_squares(nonlinear-residual,H,T,n))
                    if label=='proposal':
                        zero=shifted_state(context,bank[b],np.zeros_like(z))
                        r0u,_=nonlinear_residual(context,bank[a],bank[b],proposal,zero)
                        rzu0,_=nonlinear_residual(context,bank[a],bank[b],before_u,state)
                        pure_control=r0u-(row['residual']+row['B']@du)
                        nuisance_curve=rzu0-(row['residual']+row['J']@z)
                        cross=nonlinear-rzu0-r0u+row['residual']
                        error=nonlinear-residual
                        identity=error-pure_control-nuisance_curve-cross
                        check['error_components']=dict(pure_control=channel_squares(pure_control,H,T,n),
                            nuisance_nonlin=channel_squares(nuisance_curve,H,T,n),control_nuisance_cross=channel_squares(cross,H,T,n),
                            vector_identity_max_abs=float(np.max(abs(identity))))
                    result['independent_nonlinear_check']=check
                except Exception as exc:result['independent_nonlinear_check']=dict(attempted=True,success=False,failure=type(exc).__name__+': '+str(exc))
            entry['states'][label]=result
        entry['delta_proxy_gap']=entry['states']['proposal']['signed_gap']-entry['states']['parent']['signed_gap']
        entry['delta_AC_gap']=ac_after[a,b]-ac_before[a,b]
        pairs.append(entry)
    parent=_aggregate(values_before);after=_aggregate(values_after)
    if abs(after['score']-step['master']['exact_score'])>1e-6*(1+abs(after['score'])):raise ValueError('recomputed free proxy differs from saved master exact score')
    full_before=_aggregate(ac_before);full_after=_aggregate(ac_after)
    for recomputed,saved in ((full_before['score'],step['before_score']),(full_after['score'],step['AC_score'])):
        if abs(recomputed-saved)>1e-8*(1+abs(saved)):raise ValueError('saved AC aggregation mismatch')
    delta_proxy=after['score']-parent['score'];delta_AC=full_after['score']-full_before['score']
    return dict(success=True,name=name,step=step['step'],accepted=step.get('accepted'),
        parent_power_kw=before_u,proposal_power_kw=proposal,parent_proxy=parent,proposal_proxy=after,
        parent_AC=full_before,proposal_AC=full_after,delta_proxy=delta_proxy,delta_AC=delta_AC,
        proxy_increase_AC_not_increase=delta_proxy>1e-7 and delta_AC<=1e-7,
        master={k:step['master'].get(k) for k in ('exact_score','objective_lower','objective_upper','gap','completed','unfinished','selected_origin','scope')},
        pairs=pairs,seconds=perf_counter()-began)


def audit_record(record):
    answer=dict(spec=record['spec'],methods=[],success=True)
    for name in METHODS:
        raw=record.get('planning',{}).get(name,{}).get('raw') or {};origin=raw.get('parent_origin')
        parents=[i for i in raw.get('initials',[]) if i.get('origin')==origin and i.get('success') and i.get('profile',{}).get('success')]
        current=None if not parents else np.asarray(parents[0]['profile']['power_kw'],float)
        current_profile=None if not parents else parents[0]['profile']
        method=dict(name=name,parent_origin=origin,steps=[],recorded_steps=len(raw.get('steps',[])))
        for step in raw.get('steps',[]):
            profile=step.get('AC_profile') or {};master=step.get('master') or {}
            if not profile.get('success'):
                method['steps'].append(dict(step=step['step'],eligible=False,reason='no complete AC-scored new proposal',
                    failure=step.get('failure'),reason_saved=step.get('reason'),master_exact_score=master.get('exact_score')));continue
            try:
                if current is None:raise ValueError('complete saved parent profile unavailable')
                proposed=np.asarray(master['power_kw'],float)
                if np.allclose(proposed,current,atol=1e-8,rtol=0):raise ValueError('AC-scored proposal duplicates parent')
                context=build_context(record,current)
                row=audit_proposal(context,record,name,step,current,current_profile);row['eligible']=True
            except Exception as exc:row=dict(step=step['step'],eligible=True,success=False,failure=type(exc).__name__+': '+str(exc));answer['success']=False
            method['steps'].append(row)
            if step.get('accepted') and profile.get('success'):current=np.asarray(master['power_kw'],float);current_profile=profile
        answer['methods'].append(method)
    return answer


def run(root,output):
    root,output=Path(root),Path(output)
    if output.exists() or output.with_suffix('.md').exists():raise FileExistsError(output)
    freeze=io.read_json(root/'freeze.json');source_checks={}
    for name in ('ev_mobile_v3_ac.py','ev_mobile_v3_projection.py','ev_fixed_train_design.py','likelihood.py','symmetric_ac_local.py'):
        digest=sha256((HERE/name).read_bytes()).hexdigest();source_checks[name]=digest
        if digest!=freeze['source_sha256'][name]:raise ValueError('used scientific module differs from frozen pilot: '+name)
    rows=[]
    for spec in freeze['specifications']['development']:
        path=root/'development'/(str(spec['index']).zfill(3)+'.json.gz')
        try:
            data=path.read_bytes();record=json.loads(gzip.decompress(data))
            if record['spec']!=spec or record['policy_id']!=freeze['policy_id']:raise ValueError('frozen identity mismatch')
            row=audit_record(record);row['source']=dict(path=str(path.resolve()),bytes=len(data),sha256=sha256(data).hexdigest())
        except Exception as exc:row=dict(spec=spec,success=False,methods=[],failure=type(exc).__name__+': '+str(exc))
        rows.append(row);print(json.dumps(dict(index=spec['index'],success=row['success'])),flush=True)
    steps=[s for r in rows for m in r['methods'] for s in m['steps']];eligible=[s for s in steps if s['eligible']];good=[s for s in eligible if s.get('success')]
    pair_states=[state for s in good for p in s['pairs'] for state in p['states'].values()]
    summary=dict(history_denominator=len(rows),complete_history_audits=sum(r['success'] for r in rows),method_denominator=2*len(rows),
        recorded_steps=len(steps),complete_AC_scored_proposals=len(eligible),audited_proposals=len(good),
        proxy_increase_AC_not_increase=sum(s['proxy_increase_AC_not_increase'] for s in good),
        pair_state_denominator=len(pair_states),parameter_domain_feasible=sum(s['domain']['feasible'] for s in pair_states),
        independent_nonlinear_success=sum(s['independent_nonlinear_check'].get('success',False) for s in pair_states),
        domain_violations={k:sum(s['domain']['violations'][k] for s in pair_states) for k in (pair_states[0]['domain']['violations'] if pair_states else [])})
    result=dict(schema='mobile_v3_saved_proxy_vs_AC_audit_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Development only, complete saved AC-scored projection proposals; no source/line fits or action optimization.',
        policy_id=freeze['policy_id'],used_scientific_source_sha256=source_checks,
        independent_physics_source_sha256=sha256(Path(independent.__file__).read_bytes()).hexdigest(),
        numerical_LS='Same column normalization/default numerical rank as frozen projector; normalized minimum-norm z representative.',
        equations=dict(proxy='q_ab(u)=||r_ab+B_ab*(u-u0)+J_b*z*_ab(u)||^2/2-q_aa(u0)',
            nonlinear_error='r_NL(u,z)-[r0+B*du+J*z]=control_curve+nuisance_curve+cross',
            cross='r_NL(u,z)-r_NL(u0,z)-r_NL(u,0)+r0'),
        interpretation='Proxy global gap measures only frozen optimization. Reference error is not tested here; same-step nonlinear error diagnoses the free tangent, without refitting.',
        caution='Squared norms of three vector components need not add because of cross inner products; vector reconstruction is checked.',
        summary=summary,histories=rows,tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    io.write_json(output,result);render(result,output.with_suffix('.md'));return result


def render(data,target):
    lines=['# 自由切线代理与完整 AC 预测的离线审计','',
        '仅使用开发数据中已完成完整 AC 评分的新提案。重建同一父点的 r、B、J，按原数值秩求自由最小二乘。没有重新拟合线路/源或重新规划动作。未评分提案保留在步骤分母。','',
        f"历史 {data['summary']['complete_history_audits']}/{data['summary']['history_denominator']}；完整 AC 新提案 {data['summary']['complete_AC_scored_proposals']}，成功审计 {data['summary']['audited_proposals']}；代理升而 AC 不升 {data['summary']['proxy_increase_AC_not_increase']}。",'',
        '|历史/方法|代理父→提案|AC父→提案|Δ代理|ΔAC|代理全局间隙|接受|','|---|---:|---:|---:|---:|---:|---|']
    for r in data['histories']:
        for m in r['methods']:
            for s in m['steps']:
                if s.get('success'):
                    lines.append(f"|{r['spec']['index']}/{m['name']}|{s['parent_proxy']['score']:.6g}→{s['proposal_proxy']['score']:.6g}|{s['parent_AC']['score']:.6g}→{s['proposal_AC']['score']:.6g}|{s['delta_proxy']:.6g}|{s['delta_AC']:.6g}|{s['master']['gap']}|{s['accepted']}|")
                else:lines.append(f"|{r['spec']['index']}/{m['name']}|{s.get('reason',s.get('failure'))}||||||")
    lines.extend(['','参数步的盒/比例/源违约、相对线路变化、全有向对 q/AC 半差，以及可行参数步的独立 AC 余项按 H/E 和 V/rootP/rootQ 保存在 JSON。自由参数越界时保留失败状态并跳过物理回放，不将自由切线值当可执行 AC 解释。','',
        '纯控制曲率、参数非线性和控制×参数交叉项按向量恒等式重建；各自平方范数通常不能相加。该检查可以区分代理优化未收敛和局部切线偏差，真实参考模型误差仍由另一项离线响应归因检查。',''])
    Path(target).write_text('\n'.join(lines),encoding='utf8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/mobile_v3_pilot2'))
    p.add_argument('--output',type=Path,default=Path('outputs/mobile_v3_pilot2/proxy_AC_audit_v1.json'))
    a=p.parse_args();x=run(a.root,a.output);print(json.dumps(x['summary']),flush=True)


if __name__=='__main__':main()
