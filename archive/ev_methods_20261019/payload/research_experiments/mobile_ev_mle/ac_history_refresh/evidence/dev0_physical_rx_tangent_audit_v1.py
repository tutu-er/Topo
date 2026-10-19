"""Oracle development diagnostic: retain exact R/X displacement, zero fitting."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import numpy as np
from research_experiments.mobile_ev_mle.ac_history_refresh import io,optimizer
from research_experiments.mobile_ev_mle.ev_bayes_response_truncation import eta_to_rx


def main():
    began=perf_counter()
    ap=Path('outputs/ev_ac_repair_development_v3/development/history_000.json.gz')
    bp=Path('outputs/ev_ac_history_refresh_development_v1/development/history_000.json.gz')
    a,b=io.load_record(ap),io.load_record(bp);inputs=io.prepare_history(a)
    name=a['true_name'];candidate=inputs['candidates'][a['true_index']];d=2*candidate.m
    fit=b['history_refresh']['candidates'][name]['MAP'];w0=np.asarray(fit['map_w'])
    options={key:inputs['options'][key] for key in ('source_p_bounds','source_q_bounds','voltage_bounds')}
    problem=optimizer.ACHistoryProblem(candidate,inputs['history'],inputs['history_meter'],
        inputs['line_priors'][name],inputs['K_D'],options)
    center=problem.evaluate(w0);r0,x0=eta_to_rx(w0[:d]);Jeta=center['J_eta']
    # eta=(a,b)=(log R, log(X/R)); these are chain-rule identities.
    Jr=(Jeta[:,0::2]-Jeta[:,1::2])/r0
    Jx=Jeta[:,1::2]/x0
    r=np.asarray(a['true_lines_evaluation_only']['r'])
    x=np.asarray(a['true_lines_evaluation_only']['x'])
    eta=np.c_[np.log(r),np.log(x/r)].ravel();target=np.r_[eta,w0[d:]]
    actual=problem.evaluate(target,with_jacobian=False)
    f0=np.asarray(center['mean']);ftrue=np.asarray(actual['mean']);y=problem.observed
    models=dict(MAP_AC=f0,true_RX_AC=ftrue,
        eta_tangent=f0+Jeta@(eta-w0[:d]),
        physical_RX_tangent=f0+Jr@(r-r0)+Jx@(x-x0))
    T,n=inputs['history'].t,candidate.n;rows={}
    for label,mean in models.items():
        residual=mean-y;closure=mean-ftrue;channels={};start=0
        for channel,width in dict(p=T*n,q=T*n,v=T*n,master_p=T,master_q=T).items():
            rblock=residual[start:start+width];cblock=closure[start:start+width]
            row=dict(SSE=float(rblock@rblock),RMS_white=float(np.sqrt(np.mean(rblock**2))),
                closure_RMS_white=float(np.sqrt(np.mean(cblock**2))),
                closure_maxabs_white=float(np.max(abs(cblock))))
            if channel=='v':
                sigma=np.broadcast_to(np.asarray(inputs['history_meter'].v_sigma),(T,n)).ravel()
                row['closure_RMS_volts_at_230V']=float(np.sqrt(np.mean((cblock*sigma*230.)**2)))
            channels[channel]=row;start+=width
        rows[label]=dict(measurement_half_SSE=.5*float(residual@residual),channels=channels,
            complete_mean_closure_RMS_white=float(np.sqrt(np.mean(closure**2))))
    chain=np.empty_like(Jeta)
    chain[:,0::2]=Jr*r0+Jx*x0;chain[:,1::2]=Jx*x0
    result=dict(scope='dev0 saved true-star oracle forward audit only; neither A nor B algorithm changed',
        candidate=name,input_sha256={str(path):sha256(path.read_bytes()).hexdigest() for path in (ap,bp)},
        diagnostic_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        chain_rule_reconstruction_maxabs=float(np.max(abs(chain-Jeta))),
        fixed_state='exact same saved historical MAP P/Q and source P/Q at all compared points',
        center_r=r0,center_x=x0,diagnostic_true_r=r,diagnostic_true_x=x,
        rows=rows,physical=dict(MAP_ok=center['physical_ok'],true_RX_point_ok=actual['physical_ok'],
            failures=actual['physical_failures'],eta_primal_violation=float(np.max(problem.A@eta-problem.h,initial=0))),
        costs=dict(AC_calls=problem.ac_calls,AC_states=problem.ac_states,new_optimizations=0,
            new_cases=0,new_actions=0,new_samples=0,new_noise=0),seconds=perf_counter()-began,
        changes_policy_or_posterior=False,truth_used_only_in_labeled_offline_audit=True)
    io.save_record('research_experiments/mobile_ev_mle/ac_history_refresh/evidence/dev0_physical_rx_tangent_audit_v1.json',result)
    for name,row in rows.items():print(name,'halfSSE',row['measurement_half_SSE'],
        'V closure white',row['channels']['v']['closure_RMS_white'],flush=True)


if __name__=='__main__':main()
