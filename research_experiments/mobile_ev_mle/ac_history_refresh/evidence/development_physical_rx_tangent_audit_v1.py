"""Eight saved development points; compare tangent coordinates without fitting."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import numpy as np
from research_experiments.mobile_ev_mle.ac_history_refresh import io,optimizer
from research_experiments.mobile_ev_mle.ev_bayes_response_truncation import eta_to_rx


def audit(index):
    began=perf_counter();a_path=Path(f'outputs/ev_ac_repair_development_v3/development/history_{index:03d}.json.gz')
    b_path=Path(f'outputs/ev_ac_history_refresh_development_v1/development/history_{index:03d}.json.gz')
    record=dict(development=index,status='input_failure',AC_calls=0,AC_states=0)
    if not a_path.exists() or not b_path.exists():
        record['failure']='saved development A/B input missing';return record
    record['input_sha256']={str(p):sha256(p.read_bytes()).hexdigest() for p in (a_path,b_path)}
    a,b=io.load_record(a_path),io.load_record(b_path);inputs=io.prepare_history(a)
    record['spec']=inputs['spec'];name=a['true_name'];candidate=inputs['candidates'][a['true_index']]
    record['candidate']=name;fit=b['history_refresh']['candidates'][name]['MAP']
    record['B_MAP_status']=fit['status'];record['B_MAP_success']=fit['success']
    w0=np.asarray(fit['map_w']);d=2*candidate.m
    options={key:inputs['options'][key] for key in ('source_p_bounds','source_q_bounds','voltage_bounds')}
    problem=optimizer.ACHistoryProblem(candidate,inputs['history'],inputs['history_meter'],
        inputs['line_priors'][name],inputs['K_D'],options)
    try:
        center=problem.evaluate(w0);r0,x0=eta_to_rx(w0[:d]);Jeta=center['J_eta']
        Jr=(Jeta[:,0::2]-Jeta[:,1::2])/r0;Jx=Jeta[:,1::2]/x0
        r=np.asarray(a['true_lines_evaluation_only']['r']);x=np.asarray(a['true_lines_evaluation_only']['x'])
        eta=np.c_[np.log(r),np.log(x/r)].ravel();target=np.r_[eta,w0[d:]]
        actual=problem.evaluate(target,with_jacobian=False)
        record['physical']=dict(MAP_ok=center['physical_ok'],MAP_failures=center['physical_failures'],
            true_RX_point_ok=actual['physical_ok'],true_RX_failures=actual['physical_failures'],
            eta_primal_violation=float(np.max(problem.A@eta-problem.h,initial=0)))
        if not center['physical_ok'] or not actual['physical_ok'] or record['physical']['eta_primal_violation']>1e-10:
            record.update(status='physical_or_original_domain_failure',failure='retain failed forward point');return record
        f0=np.asarray(center['mean']);ftrue=np.asarray(actual['mean']);y=problem.observed
        models=dict(MAP_AC=f0,true_RX_AC=ftrue,eta_tangent=f0+Jeta@(eta-w0[:d]),
                    physical_RX_tangent=f0+Jr@(r-r0)+Jx@(x-x0));T,n=inputs['history'].t,candidate.n;rows={}
        for label,mean in models.items():
            residual=mean-y;closure=mean-ftrue;channels={};start=0
            for channel,width in dict(p=T*n,q=T*n,v=T*n,master_p=T,master_q=T).items():
                rr=residual[start:start+width];ee=closure[start:start+width]
                channels[channel]=dict(SSE=float(rr@rr),RMS_white=float(np.sqrt(np.mean(rr**2))),
                    closure_RMS_white=float(np.sqrt(np.mean(ee**2))),closure_maxabs_white=float(np.max(abs(ee))))
                start+=width
            rows[label]=dict(measurement_half_SSE=.5*float(residual@residual),channels=channels)
        chain=np.empty_like(Jeta);chain[:,0::2]=Jr*r0+Jx*x0;chain[:,1::2]=Jx*x0
        record.update(status='ok',rows=rows,chain_rule_reconstruction_maxabs=float(np.max(abs(chain-Jeta))),
            center_r=r0,center_x=x0,diagnostic_true_r=r,diagnostic_true_x=x,
            fixed_state='same saved B MAP historical terminal P/Q and source P/Q')
    except Exception as exc:record.update(status='forward_or_Jacobian_failure',failure=type(exc).__name__+': '+str(exc))
    finally:record.update(AC_calls=problem.ac_calls,AC_states=problem.ac_states,seconds=perf_counter()-began)
    return record


def main():
    began=perf_counter();rows=[]
    for index in range(8):
        try:row=audit(index)
        except Exception as exc:row=dict(development=index,status='input_or_initialization_failure',failure=type(exc).__name__+': '+str(exc),AC_calls=0,AC_states=0)
        rows.append(row)
        if row['status']=='ok':
            print(index,row['candidate'],[(name,row['rows'][name]['measurement_half_SSE'],
                row['rows'][name]['channels']['v']['closure_RMS_white']) for name in
                ('true_RX_AC','eta_tangent','physical_RX_tangent')],flush=True)
        else:print(index,row['status'],row.get('failure'),flush=True)
    valid=[r for r in rows if r['status']=='ok']
    result=dict(scope='eight saved development true-line oracle forward diagnostics only; A/B inference unchanged',
        diagnostic_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),records=rows,
        denominator=8,valid_count=len(valid),failure_count=8-len(valid),
        physical_RX_lower_voltage_closure_count=sum(r['rows']['physical_RX_tangent']['channels']['v']['closure_RMS_white']<
            r['rows']['eta_tangent']['channels']['v']['closure_RMS_white'] for r in valid),
        costs=dict(AC_calls=sum(r['AC_calls'] for r in rows),AC_states=sum(r['AC_states'] for r in rows),
            new_optimizations=0,new_cases=0,new_actions=0,new_samples=0,new_noise=0),
        seconds=perf_counter()-began,truth_used_only_in_labeled_offline_audit=True,
        changes_A_or_B_policy_or_posterior=False,used_for_B_numerical_or_statistical_choice=False)
    io.save_record('research_experiments/mobile_ev_mle/ac_history_refresh/evidence/development_physical_rx_tangent_audit_v1.json',result)


if __name__=='__main__':main()
