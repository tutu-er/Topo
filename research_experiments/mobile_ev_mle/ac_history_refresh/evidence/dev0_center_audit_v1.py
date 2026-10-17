"""Saved development-zero audit only: no fitting, planning, or new samples."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import numpy as np
from research_experiments.mobile_ev_mle.ac_history_refresh import io,optimizer,diagnostics
from research_experiments.mobile_ev_mle.ev_bayes_response_truncation import eta_to_rx
from research_experiments.mobile_ev_mle.symmetric_ac_local import mean_and_jacobian


def main():
    began=perf_counter()
    ap=Path('outputs/ev_ac_repair_development_v3/development/history_000.json.gz')
    bp=Path('outputs/ev_ac_history_refresh_development_v1/development/history_000.json.gz')
    a,b=io.load_record(ap),io.load_record(bp);inputs=io.prepare_history(a)
    options={k:inputs['options'][k] for k in ('source_p_bounds','source_q_bounds','voltage_bounds')}
    histories={};objective_calls=0
    for candidate in inputs['candidates']:
        row=b['history_refresh']['candidates'][candidate.name];fit=row['MAP'];post=row['posterior']
        problem=optimizer.ACHistoryProblem(candidate,inputs['history'],inputs['history_meter'],
            inputs['line_priors'][candidate.name],inputs['K_D'],options)
        closure=diagnostics.center_closure(problem,fit['map_w'],post['bounded']['mean'])
        maprx=eta_to_rx(np.asarray(fit['eta']));boundedrx=eta_to_rx(np.asarray(post['bounded']['mean'])[:candidate.m*2])
        histories[candidate.name]=dict(MAP_objective=fit['objective'],MAP_terms=fit['objective_terms'],
            MAP_to_bounded_center=closure,MAP_r=maprx[0],MAP_x=maprx[1],
            bounded_r=boundedrx[0],bounded_x=boundedrx[1],
            line_prior=post['line_prior'],
            evidence={k:post[k] for k in ('log_evidence_white','gaussian_log_density',
                'gaussian_logdet','gaussian_quadratic','logZprior','logZpost')},
            A_historical_weight=next(iter(a['actions'].values()))['propagation']['banks']['G00']['prior'][
                inputs['candidate_names'].index(candidate.name)],
            B_historical_weight=b['history_refresh']['context']['prior'][inputs['candidate_names'].index(candidate.name)])
        objective_calls+=problem.counters['evaluations']
    true_name=a['true_name'];true_index=a['true_index'];candidate=inputs['candidates'][true_index]
    tr,tx=map(np.asarray,(a['true_lines_evaluation_only']['r'],a['true_lines_evaluation_only']['x']))
    frow=b['history_refresh']['context']['candidate_updates'][true_name]
    fm=np.asarray(frow['TV_posterior_future']['mean']);d=2*candidate.m
    br,bx=eta_to_rx(fm[:d]);forecast=fm[d:].reshape(inputs['horizon'],2*candidate.n+2)
    ar=np.asarray(a['posterior_centers'][true_name]['r']);ax=np.asarray(a['posterior_centers'][true_name]['x'])
    mr,mx=eta_to_rx(np.asarray(b['history_refresh']['candidates'][true_name]['MAP']['eta']))
    def physical(r,x,state,power,root):
        state=np.array(state,copy=True);state[:,:candidate.n]+=power/inputs['base_kw']
        out=mean_and_jacobian(candidate,r,x,state[:,:candidate.n],state[:,candidate.n:2*candidate.n],
            state[:,-2],state[:,-1],root,inputs['future_meter'],with_jacobian=False)
        return out
    def response_error(prediction,truth):
        T,n=inputs['horizon'],candidate.n;start=0;out={}
        for name,size in dict(p=T*n,q=T*n,v=T*n,master_p=T,master_q=T).items():
            delta=np.asarray(prediction)[start:start+size]-np.asarray(truth)[start:start+size]
            out[name]=dict(RMS_white=float(np.sqrt(np.mean(delta**2))),signed_mean_white=float(delta.mean()))
            if name=='v':
                sigma=np.broadcast_to(np.asarray(inputs['future_meter'].v_sigma),(T,n)).ravel()
                out[name].update(RMS_volts=float(np.sqrt(np.mean((delta*sigma*230.)**2))),
                                 signed_mean_volts=float(np.mean(delta*sigma*230.)))
            start+=size
        return out
    future={};forward_calls=0
    for key,action in b['actions'].items():
        power=np.asarray(action['power']);future[key]=dict(methods=action['methods'],regimes={})
        for regime,values in action['regimes'].items():
            truth_inputs=a['actions'][key]['regimes'][regime]['truth_inputs_evaluation_only']
            z=np.c_[truth_inputs['p'],truth_inputs['q'],truth_inputs['source_p'],truth_inputs['source_q']]
            clean=np.asarray(values['clean']);checked=physical(tr,tx,z,power,truth_inputs['root_v'])
            if not np.allclose(checked['mean'],clean,atol=2e-10,rtol=0):raise ValueError('truth replay mismatch')
            rows={}
            for label,r,x,s in (('MAP_line_true_state',mr,mx,z),('B_bounded_line_true_state',br,bx,z),
                ('A_bounded_line_true_state',ar,ax,z),('true_line_B_forecast',tr,tx,forecast),
                ('B_bounded_line_B_forecast',br,bx,forecast)):
                out=physical(r,x,s,power,truth_inputs['root_v']);forward_calls+=1
                rows[label]=response_error(out['mean'],clean)
            for prefix,record in (('A',a),('B',b)):
                for model in ('G00','G11'):
                    bank=record['actions'][key]['propagation']['banks'][model]
                    mean=np.asarray(bank['means'])[0,true_index]
                    rows[prefix+'_'+model]=response_error(mean,clean)
            future[key]['regimes'][regime]=dict(errors=rows,
                B_G00_clean_score=values['scores']['G00']['clean'],
                B_G11_clean_score=values['scores']['G11']['clean'])
            forward_calls+=1
    result=dict(scope='development-zero offline localization only; truth appears in labeled forward-response audits',
        input_sha256={str(p):sha256(p.read_bytes()).hexdigest() for p in (ap,bp)},
        histories=histories,true_name=true_name,true_lines=dict(r=tr,x=tx),future=future,
        costs=dict(history_forward_Jacobian_or_AC_calls=objective_calls,future_AC_calls=forward_calls,
                   new_optimizations=0,new_cases=0,new_actions=0,new_noise=0),
        seconds=perf_counter()-began,changes_policy_or_posterior=False)
    io.save_record('research_experiments/mobile_ev_mle/ac_history_refresh/evidence/dev0_center_audit_v1.json',result)
    for name,row in histories.items():
        closure=row['MAP_to_bounded_center'];print(name,'weight',row['B_historical_weight'],
            'AC-affine halfSSE',closure['likelihood_difference']['AC_minus_affine_half_SSE'],
            'closure white',closure['channels']['v']['closure']['RMS'],flush=True)
    for action in future.values():
        for regime,row in action['regimes'].items():print(action['methods'],regime,
            {k:round(v['v']['RMS_volts'],7) for k,v in row['errors'].items()},flush=True)


if __name__=='__main__':main()
