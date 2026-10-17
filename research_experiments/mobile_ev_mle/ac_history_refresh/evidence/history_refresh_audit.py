"""Read-only B provenance and complete historical trajectory replay.

No fit, QP, NNLS, GHK, posterior update, action selection, or noise generation
is called. Saved active multipliers are independently substituted into KKT.
Forward replay has explicit floating-point matching tolerances; it does not
change the frozen scientific stopping thresholds or failure classifications.
"""
from __future__ import annotations
import argparse
import base64
import gzip
from hashlib import sha256,sha1
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.linalg import solve_triangular
from .. import io,optimizer
from ... import ev_bayes_response_ac as old
from ... import ev_systematic_refit as refit
from ...information_context import meter_from_record
from ...likelihood import concat_observations,RadialCandidate

MATCH=dict(phi_abs=2e-7,phi_rel=1e-10,array_abs=2e-10,array_rel=2e-10,
           slope_abs=2e-7,slope_rel=2e-9,kkt_abs=1e-8)


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class Checks:
    def __init__(self):self.count=0;self.failures=[];self.notes=[]
    def check(self,condition,name,**details):
        self.count+=1
        if not bool(condition):self.failures.append(dict(check=name,**details))
        return bool(condition)
    def close(self,actual,expected,name,*,absolute=None,relative=None):
        a,b=np.asarray(actual,float),np.asarray(expected,float)
        absolute=MATCH['array_abs'] if absolute is None else absolute
        relative=MATCH['array_rel'] if relative is None else relative
        ok=a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,atol=absolute,rtol=relative)
        error=float(np.max(abs(a-b),initial=0)) if a.shape==b.shape else None
        return self.check(ok,name,maximum_absolute_error=error,atol=absolute,rtol=relative)
    def result(self):return dict(checks=self.count,failures=self.failures,notes=self.notes,passed=not self.failures)


def _phi_tolerance(value):return MATCH['phi_abs']+MATCH['phi_rel']*abs(value)


def _forward(problem,eta,v,with_jacobian):
    w=np.r_[eta,problem.state_cholesky@v]
    out=problem.evaluate(w,with_jacobian=with_jacobian)
    residual=np.asarray(out['mean'])-problem.observed;delta=eta-problem.line_mean
    terms=dict(measurement=.5*float(residual@residual),
        line_prior=.5*float(delta@problem.line_precision@delta),process_prior=.5*float(v@v))
    return dict(w=w,response=out,residual=residual,terms=terms,phi=sum(terms.values()))


def _physical(check,problem,point,label):
    response=point['response'];check.check(np.isfinite(point['w']).all(),label+'/finite_state')
    violation=float(np.max(problem.A@point['w'][:problem.line_dimension]-problem.h,initial=0))
    # The original scientific line gate is checked; extra allowance is only rounding.
    tolerance=getattr(problem,'options',optimizer.DEFAULTS)['qp_feasibility_tol']
    check.check(violation<=tolerance+1e-10,label+'/original_eta_domain',violation=violation)
    if not hasattr(problem,'candidate'):return dict(physical_ok=bool(response.get('physical_ok',True)),
        physical_failures=response.get('physical_failures',[]),line_violation=violation)
    n=problem.candidate.n;z=point['w'][problem.line_dimension:].reshape(problem.latent_shape)
    phy=response['physical'];voltage=abs(np.asarray(phy['voltage']));fail=[]
    lo,hi=problem.options['voltage_bounds']
    if voltage.min()<lo or voltage.max()>hi:fail.append('voltage_domain')
    for j,name in ((-2,'source_p'),(-1,'source_q')):
        lo,hi=problem.options[name+'_bounds']
        if np.any(z[:,j]<lo) or np.any(z[:,j]>hi):fail.append(name+'_domain')
    check.check((not fail)==response['physical_ok'],label+'/independent_physical_flag')
    check.check(sorted(fail)==sorted(response.get('physical_failures',[])),label+'/independent_physical_reasons')
    loads=np.zeros_like(np.asarray(phy['loads']),dtype=complex)
    loads[:,0]=z[:,-2]+1j*z[:,-1]
    loads[:,list(problem.candidate.terminal_edges)]+=z[:,:n]+1j*z[:,n:2*n]
    check.close(np.asarray(phy['loads']).real,loads.real,label+'/single_source_P_interface')
    check.close(np.asarray(phy['loads']).imag,loads.imag,label+'/single_source_Q_interface')
    return dict(physical_ok=not fail,physical_failures=fail,line_violation=violation,
                source_interface=1,min_voltage=float(voltage.min()),max_voltage=float(voltage.max()))


def _kkt_with_saved_dual(problem,eta,v,point,saved,check,label):
    out=point['response'];Jeta=np.asarray(out['J_eta']);Jstate=np.asarray(out['J_state'])
    Jv=Jstate@problem.state_cholesky;r=point['residual']
    ge=Jeta.T@r+problem.line_precision@(eta-problem.line_mean);gv=Jv.T@r+v
    De=np.sqrt(np.diag(problem.line_precision)+np.sum(Jeta*Jeta,axis=0))
    Dv=np.sqrt(1+np.sum(Jv*Jv,axis=0));dual=np.asarray(saved['dual'],float)
    check.check(dual.shape==(len(problem.h),),label+'/dual_shape')
    if dual.shape!=(len(problem.h),):return None
    check.check(np.min(dual,initial=0)>=-1e-12,label+'/nonnegative_saved_multipliers')
    slack=problem.h-problem.A@eta
    line=float(np.max(abs(ge+problem.A.T@dual)/De,initial=0));state=float(np.max(abs(gv)/Dv,initial=0))
    complement=float(np.max(abs(dual*slack),initial=0)/(1+np.max(abs(dual),initial=0)*np.max(abs(problem.h),initial=0)))
    result=dict(scaled=max(line,state,complement),line_scaled=line,state_scaled=state,
                complementarity_scaled=complement,eta_primal_violation=float(np.max(-slack,initial=0)))
    for key,value in result.items():
        if key in saved:check.close(value,saved[key],label+'/'+key,absolute=MATCH['kkt_abs'],relative=1e-7)
    for key,value in dict(gradient_eta=ge,gradient_white_state=gv,column_scale_eta=De,column_scale_white_state=Dv).items():
        if key in saved:check.close(value,saved[key],label+'/'+key,absolute=2e-7,relative=2e-9)
    return result


def audit_candidate(problem,fit):
    """Replay every saved trial and accepted direction, using no optimization."""
    began=perf_counter();check=Checks();steps=[];ambiguities=[];options=fit.get('options',{})
    status=fit.get('status','missing');success=bool(fit.get('success',False));before=dict(problem.counters)
    ac_before=getattr(problem,'ac_calls',0);states_before=getattr(problem,'ac_states',0)
    required=('initial_eta','initial_latent','map_w','trace','options')
    missing=[k for k in required if k not in fit]
    if missing:
        check.check(not success,'success_requires_replay_fields',missing=missing)
        return dict(scientific_status=status,scientific_success=success,scope='declared failure has no trajectory',
            missing_fields=missing,**check.result(),costs=dict(AC_calls=0,AC_states=0,Jacobian_calls=0,new_optimizations=0))
    eta=np.asarray(fit['initial_eta'],float).copy()
    v=solve_triangular(problem.state_cholesky,np.asarray(fit['initial_latent'],float).reshape(-1),lower=True)
    accepted_count=0;last_relative=np.inf;current=None
    try:
        for index,record in enumerate(fit['trace']):
            try:current=_forward(problem,eta,v,True)
            except Exception as exc:
                check.check(not success and 'error' in record,'current_forward_failure_saved',iteration=index,error=str(exc));break
            phy=_physical(check,problem,current,'step%d/current'%index)
            if 'objective_before' in record:
                check.close(current['phi'],record['objective_before'],'step%d/original_objective'%index,
                    absolute=MATCH['phi_abs'],relative=MATCH['phi_rel'])
            if index==0 and fit.get('initial_objective') is not None:
                check.close(current['phi'],fit['initial_objective'],'initial/original_objective',
                    absolute=MATCH['phi_abs'],relative=MATCH['phi_rel'])
            if 'terms_before' in record:
                for key,value in current['terms'].items():check.close(value,record['terms_before'][key],
                    'step%d/term_%s'%(index,key),absolute=MATCH['phi_abs'],relative=MATCH['phi_rel'])
            if 'direction_eta' not in record or 'direction_white_state' not in record:
                check.check(not success,'missing_direction_is_declared_failure',iteration=index)
                steps.append(dict(iteration=index,scope='failure before direction',physical=phy));continue
            pe=np.asarray(record['direction_eta']);pv=np.asarray(record['direction_white_state'])
            derivative=current['response']['J_eta']@pe+current['response']['J_state']@(problem.state_cholesky@pv)
            slope=float(current['residual']@derivative+(eta-problem.line_mean)@problem.line_precision@pe+v@pv)
            check.close(slope,record['directional_derivative'],'step%d/full_directional_derivative'%index,
                        absolute=MATCH['slope_abs'],relative=MATCH['slope_rel'])
            trials=[];accepted=[]
            for j,trial in enumerate(record.get('trials',[])):
                alpha=float(trial['alpha']);label='step%d/trial%d'%(index,j);row=dict(alpha=alpha)
                check.close(alpha,options['beta']**j,label+'/backtracking_schedule')
                if 'backtrack' in trial:check.check(trial['backtrack']==j,label+'/backtrack_index')
                try:
                    value=_forward(problem,eta+alpha*pe,v+alpha*pv,False)
                    physical=_physical(check,problem,value,label)
                    check.check(bool(trial.get('physical_ok'))==physical['physical_ok'],label+'/physical_flag')
                    if 'physical_failures' in trial:check.check(sorted(trial['physical_failures'])==sorted(physical.get('physical_failures',[])),label+'/physical_reasons')
                    row.update(objective=value['phi'],physical=physical)
                    if 'objective' in trial:check.close(value['phi'],trial['objective'],label+'/objective',
                        absolute=MATCH['phi_abs'],relative=MATCH['phi_rel'])
                    margin=value['phi']-current['phi']-options['armijo']*alpha*slope
                    row['Armijo_margin']=margin;tau=_phi_tolerance(current['phi'])
                    if trial.get('accepted'):
                        check.check(physical['physical_ok'],label+'/accepted_physical')
                        check.check(margin<=tau,label+'/accepted_full_Armijo',margin=margin,replay_tolerance=tau)
                        accepted.append((alpha,value))
                    elif physical['physical_ok']:
                        check.check(margin>=-tau,label+'/rejected_not_clearly_Armijo',margin=margin,replay_tolerance=tau)
                    if abs(margin)<=tau:ambiguities.append(dict(iteration=index,trial=j,margin=margin))
                except Exception as exc:
                    check.check(not trial.get('accepted') and not trial.get('physical_ok') and bool(trial.get('error')),
                                label+'/forward_exception_saved',error=type(exc).__name__+': '+str(exc))
                    row.update(forward_failure=type(exc).__name__+': '+str(exc))
                trials.append(row)
            alpha=record.get('alpha')
            if alpha is not None and float(alpha)>0:
                check.check(len(accepted)==1,'step%d/single_accepted_trial'%index)
                if accepted:
                    actual_alpha,value=accepted[0];check.close(actual_alpha,alpha,'step%d/accepted_alpha'%index)
                    check.check(record['objective_after']<=record['objective_before'],'step%d/saved_original_objective_monotone'%index)
                    check.close(value['phi'],record['objective_after'],'step%d/objective_after'%index,
                        absolute=MATCH['phi_abs'],relative=MATCH['phi_rel'])
                    last_relative=abs(value['phi']-current['phi'])/(1+abs(current['phi']))
                eta=eta+float(alpha)*pe;v=v+float(alpha)*pv;accepted_count+=1
            elif alpha is not None and float(alpha)==0:
                check.check(len(accepted)==0,'step%d/zero_alpha_no_trial_acceptance'%index)
                check.check(max(np.max(abs(pe),initial=0),np.max(abs(pv),initial=0))<options['relative_obj_tol'],
                            'step%d/zero_alpha_tiny_direction'%index)
                check.check(record.get('relative_objective_change')==0,'step%d/zero_alpha_effective_relative_change'%index)
                last_relative=0.
            steps.append(dict(iteration=index,objective_before=current['phi'],directional_derivative=slope,
                current_state_sha256=sha256(np.asarray(current['w'],dtype='<f8').tobytes()).hexdigest(),
                retained_state_sha256=sha256(np.r_[eta,problem.state_cholesky@v].astype('<f8').tobytes()).hexdigest(),
                retained_eta=eta.copy(),physical=phy,trials=trials,accepted_alpha=alpha))
        try:final=_forward(problem,eta,v,True)
        except Exception as jac_error:
            # A saved numerical failure may be a Jacobian failure at an otherwise
            # valid retained point. Check the forward objective without that J.
            check.check(not success and status=='numerical_failure','final/Jacobian_failure_declared',
                        error=type(jac_error).__name__+': '+str(jac_error))
            try:final=_forward(problem,eta,v,False)
            except Exception as forward_error:
                check.check(not success and fit.get('objective') is None,'final/forward_failure_declared',
                            error=type(forward_error).__name__+': '+str(forward_error))
                final=None
        if final is None:
            result=dict(final_objective=None,final_KKT=None,
                final_replay_scope='saved numerical failure; complete forward response unavailable')
            costs=dict(AC_calls=getattr(problem,'ac_calls',0)-ac_before,
                AC_states=getattr(problem,'ac_states',0)-states_before,
                evaluations=problem.counters['evaluations']-before.get('evaluations',0),
                Jacobian_calls=problem.counters['Jacobian_calls']-before.get('Jacobian_calls',0),
                new_optimizations=0,new_GHK_calls=0)
            return dict(scientific_status=status,scientific_success=success,trajectory=steps,
                costs=costs,seconds=perf_counter()-began,**result,**check.result())
        final_physical=_physical(check,problem,final,'final')
        check.close(final['w'],fit['map_w'],'final/reconstructed_state')
        if fit.get('objective') is not None:check.close(final['phi'],fit['objective'],'final/original_objective',
            absolute=MATCH['phi_abs'],relative=MATCH['phi_rel'])
        kkt=None
        if isinstance(fit.get('kkt'),dict) and 'dual' in fit['kkt'] and 'J_eta' in final['response']:
            kkt=_kkt_with_saved_dual(problem,eta,v,final,fit['kkt'],check,'final/KKT')
        if success:
            check.check(final_physical['physical_ok'],'success/final_physical')
            check.check(kkt is not None,'success/KKT_saved')
            if kkt is not None:
                check.check(fit['kkt']['scaled']<=options['scaled_kkt_tol'],'success/original_KKT_gate')
                check.check(kkt['scaled']<=options['scaled_kkt_tol']+MATCH['kkt_abs'],'success/replayed_KKT_gate')
            # A zero direction has effective objective change zero even if the
            # optimizer's old last_relative scalar retains the previous step.
            effective=0. if not fit['trace'] else last_relative
            check.check(effective<=options['relative_obj_tol'],'success/effective_relative_objective_gate',effective=effective)
        declared_counters=fit.get('counters',{})
        if 'accepted_steps' in declared_counters:check.check(declared_counters['accepted_steps']==accepted_count,'saved/accepted_step_count')
        result=dict(final_objective=final['phi'],final_physical=final_physical,final_KKT=kkt,
            final_state_sha256=sha256(np.asarray(final['w'],dtype='<f8').tobytes()).hexdigest(),
            effective_relative_objective_change=0. if not fit['trace'] else last_relative,
            saved_relative_objective_change=fit.get('relative_objective_change'))
    except Exception as exc:
        check.check(False,'trajectory_auditor_exception',error=type(exc).__name__+': '+str(exc));result={}
    costs=dict(AC_calls=getattr(problem,'ac_calls',0)-ac_before,AC_states=getattr(problem,'ac_states',0)-states_before,
        evaluations=problem.counters['evaluations']-before.get('evaluations',0),
        Jacobian_calls=problem.counters['Jacobian_calls']-before.get('Jacobian_calls',0),new_optimizations=0,new_GHK_calls=0)
    return dict(scientific_status=status,scientific_success=success,
        scope='all saved current/trial/accepted points forward replay; final stationarity with saved active multipliers',
        trajectory=steps,Armijo_decisions_at_replay_roundoff=ambiguities,costs=costs,
        seconds=perf_counter()-began,**result,**check.result())


def audit_runtime_dependencies(root,version,*,execution_root=None,expected_count=7):
    """Optional post-hoc environment lock, explicitly outside the main freeze."""
    root=Path(root);meta=root/'runtime_dependency_git_audit.json';snapshot_path=root/'runtime_dependency_snapshot.json.gz'
    check=Checks()
    if not meta.exists() and not snapshot_path.exists():
        return dict(provided=False,scope='no supplementary runtime lock supplied',**check.result())
    check.check(meta.exists() and snapshot_path.exists(),'runtime/both_supplementary_files_present')
    if not meta.exists() or not snapshot_path.exists():return dict(provided=True,**check.result())
    details=io.load_record(meta);snapshot=io.load_record(snapshot_path);entries=details['rows']
    check.check(len(entries)==expected_count,'runtime/supplementary_count',actual=len(entries))
    check.check({r['path'] for r in entries}==set(snapshot),'runtime/snapshot_exact_manifest')
    work=Path(execution_root) if execution_root is not None else Path(version['source_paths']['B/optimizer.py']).parents[3]
    rows=[]
    for entry in entries:
        name=entry['path'];row=dict(path=name,expected=entry['sha256'])
        try:
            payload=base64.b64decode(snapshot[name],validate=True)
            row.update(restored_sha256=sha256(payload).hexdigest(),live_sha256=digest(work/name))
            check.check(row['restored_sha256']==entry['sha256'],'runtime/restored/'+name)
            check.check(row['live_sha256']==entry['sha256'],'runtime/live/'+name)
            tree=entry.get('git_tree_at_A_freeze','').split('\t',1)
            object_id=tree[0].split()[-1] if tree else ''
            actual_object=sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
            LF=payload.replace(b'\r\n',b'\n')
            normalized_object=sha1(b'blob '+str(len(LF)).encode()+b'\0'+LF).hexdigest()
            check.check(len(tree)==2 and tree[1]==name,'runtime/saved_git_tree_path/'+name)
            check.check(actual_object==object_id or normalized_object==object_id,'runtime/saved_git_blob_bytes_or_EOL_equivalence/'+name)
            row['restored_Git_blob_sha1']=actual_object
            row['CRLF_to_LF_Git_blob_sha1']=normalized_object
            row['Git_blob_comparison']='raw bytes' if actual_object==object_id else 'CRLF-to-LF text equivalence'
        except Exception as exc:check.check(False,'runtime/'+name,error=type(exc).__name__+': '+str(exc))
        rows.append(row)
    check.check(details.get('primary_source_manifests_changed') is False,'runtime/original_freeze_unmodified')
    return dict(provided=True,rows=rows,metadata_sha256=digest(meta),snapshot_sha256=digest(snapshot_path),
        scope='supplementary dependencies: snapshot/live raw-byte SHA256; Git blob raw or explicit CRLF-to-LF equivalence',
        outside_original_main_manifest=True,retroactive_freeze_claim=False,**check.result())


def audit_freeze(root,*,expected_sources=165):
    """Hash restored snapshot bytes in memory; never execute restored code."""
    root=Path(root);check=Checks();version=io.load_record(root/'freeze.json')
    check.check(canonical({k:v for k,v in version.items() if k!='policy_id'})==version['policy_id'],'B/freeze_identity')
    manifest=version['source_sha256'];check.check(len(manifest)==expected_sources,'B/frozen_source_count',actual=len(manifest))
    check.check(sum(k.startswith('A/') for k in manifest)==version['A_source_count'],'B/A_source_count')
    check.check(sum(k.startswith('B/') for k in manifest)==version['B_source_count'],'B/B_source_count')
    snapshot=io.load_record(root/'source_snapshot.json.gz');check.check(set(snapshot)==set(manifest),'B/snapshot_exact_manifest')
    rows={}
    for name,expected in manifest.items():
        row=dict(expected=expected)
        try:
            restored=base64.b64decode(snapshot[name],validate=True);row['restored_sha256']=sha256(restored).hexdigest()
            row['live_sha256']=digest(version['source_paths'][name])
            check.check(row['restored_sha256']==expected,'snapshot/'+name)
            check.check(row['live_sha256']==expected,'live/'+name)
        except Exception as exc:check.check(False,'source/'+name,error=type(exc).__name__+': '+str(exc))
        rows[name]=row
    Aroot=Path(version['input_A_root']);A=io.load_record(Aroot/'freeze.json')
    check.check(digest(Aroot/'freeze.json')==version['input_A_freeze_sha256'],'A/bound_freeze_hash')
    check.check(A['policy_id']==version['input_A_policy_id'],'A/bound_policy')
    Aidentity={k:v for k,v in A.items() if k!='policy_id'}
    Ahash=sha256(json.dumps(Aidentity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    check.check(Ahash==A['policy_id'],'A/freeze_identity_original_serialization')
    for name,value in A['source_sha256'].items():check.check(manifest.get('A/'+name)==value,'A/source_manifest/'+name)
    for phase in ('calibration','evaluation'):
        check.check(version['specifications'][phase]==A['specifications'][phase],'A/B_specifications/'+phase)
    runtime=audit_runtime_dependencies(root,version)
    check.check(runtime['passed'],'supplementary_runtime_dependencies')
    return version,dict(source_rows=rows,snapshot_sha256=digest(root/'source_snapshot.json.gz'),
        supplementary_runtime=runtime,
        source_scope='main manifest source files plus supplied supplementary dependencies; decoded bytes, no source execution',**check.result())


def audit_bindings(root,version):
    root=Path(root);Aroot=Path(version['input_A_root']);check=Checks()
    calibration=io.load_record(root/'calibration.json');binding=io.load_record(root/'evaluation_calibration_binding.json')
    check.check(calibration['policy_id']==version['policy_id'],'calibration/B_policy')
    check.check(calibration['input_A_policy_id']==version['input_A_policy_id'],'calibration/A_policy')
    check.check(calibration.get('old_A_calibration_used') is False,'calibration/old_A_not_used')
    check.check(calibration.get('fixed_before_evaluation') is True,'calibration/fixed_before_evaluation')
    expected_B={};expected_A={}
    for spec in version['specifications']['calibration']:
        name='history_%03d.json.gz'%spec['index']
        expected_B[name]=digest(root/'calibration'/name);expected_A[name]=digest(Aroot/'calibration'/name)
    check.check(calibration['input_B_sha256']==expected_B,'calibration/all_B_raw_bindings')
    check.check(calibration['input_A_sha256']==expected_A,'calibration/all_A_raw_bindings')
    check.check(calibration['histories']==len(expected_B),'calibration/full_failure_denominator')
    expected=dict(policy_id=version['policy_id'],calibration_sha256=digest(root/'calibration.json'),
        G11_multiplier=calibration['G11_multiplier'],input_A_policy_id=version['input_A_policy_id'])
    check.check(binding==expected,'evaluation/immutable_B_calibration_binding')
    Acal=io.load_record(Aroot/'calibration.json');Abind=io.load_record(Aroot/'evaluation_calibration_binding.json')
    check.check(Acal['policy_id']==version['input_A_policy_id'],'A/calibration_policy')
    Acal_inputs={str(Path('calibration')/k).replace('\\','/'):v for k,v in expected_A.items()}
    original_A_inputs={key.replace('\\','/'):value for key,value in Acal.get('input_sha256',{}).items()}
    check.check(original_A_inputs==Acal_inputs,'A/calibration_original_raw_bindings')
    check.check(Abind==dict(policy_id=version['input_A_policy_id'],calibration_sha256=digest(Aroot/'calibration.json')),
                'A/evaluation_original_calibration_binding')
    return binding,dict(B_calibration_sha256=digest(root/'calibration.json'),**check.result())


def _prior_without_integration(m,options):
    intervals={key:np.broadcast_to(np.asarray(options[key],float),(m,2)).copy()
               for key in ('r_bounds','x_bounds','ratio_bounds')}
    logs={key:np.log(intervals[key]) for key in intervals};mean=np.empty(2*m);sd=np.empty(2*m)
    mean[0::2]=logs['r_bounds'].mean(1);mean[1::2]=logs['ratio_bounds'].mean(1)
    sd[0::2]=np.diff(logs['r_bounds'],axis=1).ravel()/np.sqrt(12.)
    sd[1::2]=np.diff(logs['ratio_bounds'],axis=1).ravel()/np.sqrt(12.)
    bounds=dict(edges=m,**intervals,log_R=logs['r_bounds'],log_X=logs['x_bounds'],
        log_ratio=logs['ratio_bounds'],coordinate_order='interleaved_logR_logXoverR')
    bounds['a_outer']=np.c_[np.maximum(bounds['log_R'][:,0],bounds['log_X'][:,0]-bounds['log_ratio'][:,1]),
                            np.minimum(bounds['log_R'][:,1],bounds['log_X'][:,1]-bounds['log_ratio'][:,0])]
    return dict(mean=mean,cov=np.diag(sd*sd),bounds=bounds)


def prepare_without_integrals(A,B,settings,check):
    """Reconstruct H+V/kernel from meters; reuse saved prior normalization only."""
    training=io._observation(A['observed_training'],'training');evidence=io._observation(A['observed_evidence'],'evidence')
    tm,em=(meter_from_record(A[key]) for key in ('train_meter','evidence_meter'))
    history=concat_observations(training,evidence);meter=refit.concat_models((training,evidence),(tm,em))
    check.check(not meter.fit_bias and meter.calibration is None,'known_calibrated_meter_bias')
    names=A['candidate_names'];lookup={row['name']:row for row in A['candidate_structures']}
    candidates=[RadialCandidate.from_supports(name,history.n,lookup[name]['supports']) for name in names]
    options=io._public_options(A);hyper=old.estimate_process_hyperparameters(training,tm,start_time=0,options=options)
    horizon=int(A['config']['horizon']);mean,K=old.process_kernel(hyper,history.t+horizon,start_time=0)
    check.check(not np.any(mean),'original_zero_mean_process');split=history.t*(2*history.n+2);KD=K[:split,:split]
    check.check(canonical(io._serial(hyper))==canonical(B['process_hyperparameters']),'training_only_process_hyperparameters')
    priors={};initials={};saved_rows=B.get('history_refresh',{}).get('candidates',{})
    for candidate in candidates:
        row=saved_rows.get(candidate.name,{})
        saved=row.get('MAP',{}).get('prior_info',{}).get('line_prior',row.get('posterior',{}).get('line_prior'))
        public=_prior_without_integration(candidate.m,options)
        if saved is not None:
            for key in ('mean','cov'):check.close(saved[key],public[key],candidate.name+'/original_line_prior_'+key)
            for key,value in public['bounds'].items():
                if isinstance(value,str):check.check(saved['bounds'][key]==value,candidate.name+'/prior_'+key)
                else:check.close(saved['bounds'][key],value,candidate.name+'/prior_'+key)
            check.check(np.isfinite(saved.get('log_mass',np.nan)) and saved['log_mass']<=0,
                        candidate.name+'/saved_proper_prior_normalizer_finite')
            priors[candidate.name]=saved
        else:priors[candidate.name]=public
        initials[candidate.name]=io._initial(candidate,history,meter,training,tm,priors[candidate.name],A)
    plans={}
    for label in io.PLAN_LABELS:
        if label in A.get('plans',{}):
            row=A['plans'][label];power=np.asarray(row['power'],float)
            if power.shape==(horizon,history.n) and np.isfinite(power).all():
                plans[label]=dict(action_key=str(row['action_key']),domain=str(row['domain']),power=power)
    future_root=np.broadcast_to(np.asarray(settings['future_root_v'],float),(horizon,)).copy()
    identity=dict(spec=A.get('spec',{}),candidate_names=names,candidate_structures=A['candidate_structures'],
        initializations=initials,observed_training={key:getattr(training,key) for key in io.OBSERVATION_KEYS},
        observed_evidence={key:getattr(evidence,key) for key in io.OBSERVATION_KEYS},
        train_meter=A['train_meter'],evidence_meter=A['evidence_meter'],future_meter=A['future_meter'],
        options=options,base_kw=float(settings['base_kw']),future_root_v=future_root,horizon=horizon,plans=plans)
    check.check(canonical(io._serial(identity))==B['observed_input_sha256'],'observed_only_input_fingerprint')
    return dict(history=history,meter=meter,candidates=candidates,names=names,options=options,
                K_D=KD,line_priors=priors,initializations=initials,plans=plans)


def audit_record(A,B,settings,*,prepare=prepare_without_integrals,problem_factory=optimizer.ACHistoryProblem):
    began=perf_counter();check=Checks();rows={}
    prefit_status=B.get('status')
    refresh_exception=B.get('history_refresh',{}).get('status')=='history_refresh_exception'
    if prefit_status in ('process_failure','input_failure') or refresh_exception:
        check.check(bool(B.get('failures')),'process_failure_has_reason')
        if prefit_status=='process_failure':check.check(B.get('keep_all_on_failure') is True,'process_failure_retains_candidates')
        check.check(not any(row.get('MAP') for row in B.get('history_refresh',{}).get('candidates',{}).values()),
                    'declared_prefit_failure_has_no_hidden_MAP')
        return dict(scientific_status=prefit_status,scope='raw/meta binding only; no historical trajectory was produced',
            declared_failures=B.get('failures'),candidates={},costs=dict(AC_calls=0,AC_states=0,new_optimizations=0,new_GHK_calls=0),
            seconds=perf_counter()-began,**check.result())
    try:
        p=prepare(A,B,settings,check)
        check.check(B['candidate_names']==p['names'],'complete_candidate_bank')
        check.check(set(B.get('history_refresh',{}).get('candidates',{}))==set(p['names']),'no_candidate_dropped')
        check.check(set(B['plans'])==set(p['plans']),'all_original_available_plans')
        for label,plan in p['plans'].items():
            for key in ('action_key','domain'):check.check(B['plans'][label][key]==plan[key],'plan/'+label+'/'+key)
            check.close(B['plans'][label]['power'],plan['power'],'plan/'+label+'/power')
        for key,action in B.get('actions',{}).items():
            check.check(key in A.get('actions',{}),'action_from_A/'+key)
            if key not in A.get('actions',{}):continue
            check.close(action['power'],A['actions'][key]['power'],'action_power/'+key)
            for regime,row in action.get('regimes',{}).items():
                original=A['actions'][key].get('regimes',{}).get(regime)
                if row.get('status')=='measurement_missing':check.check(original is None,'original_measurement_missing/'+key+'/'+regime);continue
                if original is None:check.check(False,'unexpected_future_measurement/'+key+'/'+regime);continue
                for field in ('values','clean'):
                    if field in row:check.check(np.array_equal(np.asarray(row[field]),np.asarray(original[field])),
                                               'original_future_'+field+'/'+key+'/'+regime)
        context=B['history_refresh'].get('context')
        if context is not None:check.check(context.get('source_interface')==1,'forecast_same_source_interface')
        physical={key:p['options'][key] for key in ('source_p_bounds','source_q_bounds','voltage_bounds')}
        KDhash=sha256(np.asarray(p['K_D'],dtype='<f8').tobytes()).hexdigest()
        for candidate in p['candidates']:
            name=candidate.name;row=B['history_refresh']['candidates'].get(name,{})
            fit=row.get('MAP')
            if fit is None:
                check.check(row.get('status') in ('initialization_failure','candidate_exception'),'declared_missing_MAP/'+name)
                rows[name]=dict(scientific_status=row.get('status'),scope='declared failure before MAP',
                    declared_failure=row.get('failure',row.get('initialization')),passed=True,costs=dict(AC_calls=0,AC_states=0,new_optimizations=0))
                continue
            check.check(fit['prior_info']['process_covariance_sha256']==KDhash,name+'/original_history_kernel')
            check.check(fit['prior_info'].get('original_prior_reused_once') is True,name+'/original_prior_count')
            check.check(fit['prior_info'].get('history_likelihood_count')==1,name+'/history_likelihood_count')
            initial=p['initializations'][name]
            check.close(fit['initial_eta'],initial['eta'],name+'/observed_only_initial_eta')
            check.close(fit['initial_latent'],initial['latent'],name+'/observed_only_initial_state')
            problem=problem_factory(candidate,p['history'],p['meter'],p['line_priors'][name],p['K_D'],options=physical)
            rows[name]=audit_candidate(problem,fit)
            rows[name]['posterior_scientific_status']=row.get('posterior',{}).get('status')
            # GHK failure is retained as inference failure; it does not invalidate a valid MAP replay.
            check.check(rows[name]['passed'],name+'/trajectory_replay',details=rows[name]['failures'])
    except Exception as exc:check.check(False,'record_reconstruction',error=type(exc).__name__+': '+str(exc))
    costs={key:sum(row.get('costs',{}).get(key,0) for row in rows.values())
           for key in ('AC_calls','AC_states','Jacobian_calls','evaluations','new_optimizations','new_GHK_calls')}
    return dict(scientific_status=B.get('status'),declared_failures=B.get('failures'),candidates=rows,costs=costs,
        prior_normalization_scope='saved proper normalizer consistency only; no normalization integral repeated',
        source_scope='observed H+V and training-only kernel; future data used only for equality checks',
        seconds=perf_counter()-began,**check.result())


def _audit_bound_record(root,version,binding,phase,spec):
    name='history_%03d.json.gz'%spec['index'];path=root/phase/name;Apath=Path(version['input_A_root'])/phase/name
    meta=io.load_record(path.with_name(path.name.replace('.json.gz','.metadata.json')))
    A,B=io.load_record(Apath),io.load_record(path);check=Checks()
    check.check(meta['policy_id']==version['policy_id'],'metadata/B_policy')
    check.check(meta['input_A_policy_id']==version['input_A_policy_id'],'metadata/A_policy')
    check.check(meta['result_sha256']==digest(path),'metadata/B_raw_hash')
    check.check(meta['source_A_sha256']==digest(Apath),'metadata/A_raw_hash')
    check.check(meta.get('science_source_unchanged') is True,'metadata/science_unchanged')
    check.check(meta['spec']==spec and A['spec']==spec and B['spec']==spec,'metadata/A_B_spec')
    check.check(meta.get('calibration_binding')==(binding if phase=='evaluation' else None),'metadata/calibration_binding')
    if B.get('status')!='process_failure':
        check.check(B['source_A']['sha256']==digest(Apath),'raw/A_source_hash')
        check.check(Path(B['source_A']['path']).resolve()==Apath.resolve(),'raw/A_source_path')
    if 'code_initial_sha' in A:
        check.check(A.get('source_unchanged_during_run') is True,'A/source_unchanged_flag')
        check.check(A['code_initial_sha']==A.get('code_end_sha'),'A/worker_begin_end_source')
        for key,value in A['code_initial_sha'].items():
            check.check(version['source_sha256'].get('A/'+key)==value,'A/frozen_worker_source/'+key)
    else:check.check(A.get('status')=='process_failure','A/missing_worker_source_is_declared_failure')
    if check.result()['passed']:
        audit=audit_record(A,B,version['settings'])
        check.check(audit['passed'],'history_numerical_and_reuse_audit',details=audit['failures'])
    else:audit=dict(scientific_status=B.get('status'),scope='numeric replay withheld: raw/meta binding mismatch',
                    costs=dict(AC_calls=0,AC_states=0,new_optimizations=0,new_GHK_calls=0),passed=False)
    return dict(phase=phase,index=spec['index'],provenance=check.result(),audit=audit)


def audit_run(root,*,phases=('calibration','evaluation'),expected_sources=165):
    began=perf_counter();root=Path(root);controller=io.load_record(root/'controller.json')
    if len(set(phases))!=len(phases):raise ValueError('audit phases must be unique')
    if controller.get('status')!='statistical_complete':
        raise RuntimeError('postprocessing waits for B statistical_complete; no main raw was opened')
    audit_hash_start=digest(__file__)
    version,source=audit_freeze(root,expected_sources=expected_sources)
    if not source['passed']:
        return dict(schema='independent_B_history_replay_audit_v1',policy_id=version['policy_id'],passed=False,
            history_denominator=sum(len(version['specifications'][p]) for p in phases),
            source=source,records=[],unassessed_reason='frozen live/snapshot source mismatch; numeric replay withheld',
            costs=dict(AC_calls=0,AC_states=0,new_optimizations=0,new_GHK_calls=0,seconds=perf_counter()-began),
            audit_source_sha256=audit_hash_start)
    binding,bindings=audit_bindings(root,version)
    records=[];global_check=Checks()
    for phase in phases:
        if phase not in ('calibration','evaluation'):raise ValueError('calibration/evaluation only')
        for spec in version['specifications'][phase]:
            try:record=_audit_bound_record(root,version,binding,phase,spec)
            except Exception as exc:
                record=dict(phase=phase,index=spec['index'],provenance=dict(passed=False,
                    failures=[dict(check='raw_or_metadata_unreadable',error=type(exc).__name__+': '+str(exc))]),
                    audit=dict(scientific_status='unreadable_record',scope='retained denominator; no numeric replay',
                               costs=dict(AC_calls=0,AC_states=0,new_optimizations=0,new_GHK_calls=0)))
            records.append(record);global_check.check(record['provenance']['passed'],'history/%s/%d'%(phase,spec['index']))
            print(phase,spec['index'],'PASS' if record['provenance']['passed'] else 'REVIEW',
                'science',record['audit']['scientific_status'],'audit_AC',record['audit']['costs']['AC_calls'],flush=True)
    costs={key:sum(row['audit']['costs'].get(key,0) for row in records)
           for key in ('AC_calls','AC_states','Jacobian_calls','evaluations','new_optimizations','new_GHK_calls')}
    costs.update(seconds=perf_counter()-began,charged_to_scientific_fit=False)
    return dict(schema='independent_B_history_replay_audit_v1',policy_id=version['policy_id'],phases=list(phases),
        history_denominator=sum(len(version['specifications'][p]) for p in phases),records=records,
        source=source,bindings=bindings,history_checks=global_check.result(),
        passed=source['passed'] and bindings['passed'] and global_check.result()['passed'],costs=costs,
        scope='complete saved trajectory forward replay; final KKT with saved multipliers; no re-optimization or GHK',
        floating_replay_matching_tolerances=MATCH,scientific_failures_preserved=True,
        no_new_cases_actions_noise=True,source_files_changed=False,
        audit_source_sha256_start=audit_hash_start,audit_source_sha256_end=digest(__file__))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--phases',nargs='+',
        choices=('calibration','evaluation'),default=['calibration','evaluation'])
    args=parser.parse_args();value=audit_run(args.root,phases=args.phases);io.save_record(args.output,value)
    print(json.dumps(dict(passed=value['passed'],histories=value['history_denominator'],costs=value['costs'])))


if __name__=='__main__':main()
