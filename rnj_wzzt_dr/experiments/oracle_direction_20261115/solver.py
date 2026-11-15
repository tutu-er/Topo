"""Oracle diagnostic: maximize common distance/error improvement on the original domain.

Geometry is supplied by the caller. No model fitting, action compression,
support search or cost optimization is performed here.
"""
from __future__ import annotations

import hashlib
import json
from math import fsum
from pathlib import Path
from time import perf_counter

import numpy as np

from ..absolute_identification_20261114.solver import seed_items
from ..distribution_gate_20261114.solver import quadratic_box_upper
from ..flexibility_extended_solve_20261114.solver import _Recorder, _number, _reference

POSITIVE_THRESHOLD=1e-6


def array_hash(value):
    a=np.ascontiguousarray(value,dtype=np.float64)
    return hashlib.sha256(str(a.shape).encode('ascii')+a.tobytes()).hexdigest()


def polynomial_change(Q,a,u):
    """Independent scalar sum; a is the COMPLETE linear coefficient, not half."""
    quadratic=(u[:,:,None]*Q*u[:,None,:]).ravel()
    linear=(a*u).ravel()
    value=float(fsum(quadratic.tolist()+linear.tolist()))
    absolute=float(fsum(abs(quadratic).tolist()+abs(linear).tolist()))
    roundoff=64*np.finfo(float).eps*max(absolute,np.finfo(float).tiny)
    return value,roundoff


class Geometry:
    def __init__(self,case,geometry):
        self.shape=tuple(case.shape);self.H,self.R=self.shape
        self.mode=geometry['mode']
        if self.mode not in ('distance_only','protected'):raise ValueError('Unknown oracle mode')
        self.J_q=np.asarray(geometry['J_q'],float).copy();self.J_a=np.asarray(geometry['J_a'],float).copy()
        self.J_c=np.asarray(geometry['J_c'],float).copy();self.e_q=np.asarray(geometry['e_q'],float).copy()
        self.e_a=np.asarray(geometry['e_a'],float).copy();self.e_c=float(geometry['e_c'])
        self.scales=np.asarray(geometry['j_scales'],float).copy();self.count=len(self.J_c)
        expected={'J_q':(self.count,self.H,self.R,self.R),'J_a':(self.count,self.H,self.R),
            'J_c':(self.count,),'e_q':(self.H,self.R,self.R),'e_a':self.shape,'scales':(self.count,)}
        for name,shape in expected.items():
            value=getattr(self,name)
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Invalid '+name+' dimensions/values')
        if not self.count or not np.isfinite(self.e_c) or self.e_c<0 or np.any(self.J_c<0):raise ValueError('Nonnegative baseline sums and a nonempty competitor bank required')
        if not np.array_equal(self.scales,np.maximum(1.,self.J_c)):raise ValueError('j_scales must equal max(1,J0)')
        self.symmetry=[]
        for name in ('J_q','e_q'):
            Q=getattr(self,name);scale=float(np.max(abs(Q),initial=0.));error=float(np.max(abs(Q-np.swapaxes(Q,-1,-2)),initial=0.))
            if error>64*np.finfo(float).eps*max(scale,np.finfo(float).tiny):raise ValueError(name+' not symmetric within roundoff')
            Q=(Q+np.swapaxes(Q,-1,-2))/2
            eigenvalues=np.linalg.eigvalsh(Q);spectral=float(np.max(abs(eigenvalues),initial=0.))
            tolerance=128*np.finfo(float).eps*self.R*max(spectral,np.finfo(float).tiny)
            if float(np.min(eigenvalues)) < -tolerance:raise ValueError(name+' is not a squared-response PSD form')
            setattr(self,name,Q);self.symmetry.append(dict(name=name,maximum_asymmetry=error,minimum_eigenvalue=float(np.min(eigenvalues)),psd_roundoff_tolerance=tolerance))
        if not np.isfinite(case.lower).all() or not np.isfinite(case.upper).all():raise ValueError('Finite original resource amplitude bounds required')
        self.box=[]
        for j in range(self.count):
            # Existing helper uses2*a; this geometry supplies the full coefficient.
            bound=quadratic_box_upper(self.J_q[j],self.J_a[j]/2.,0.,case.lower,case.upper)
            self.box.append(dict(competitor=j,distance_increment_upper=bound['upper'],
                z_upper=bound['upper']/self.scales[j],roundoff_padding=bound['roundoff_padding']))
        self.box_upper=float(min(b['z_upper'] for b in self.box))
        if self.box_upper<0:raise ValueError('Box bound contradicts feasible zero action')
        self.upper=min(self.box_upper,1.) if self.mode=='protected' and self.e_c>0 else self.box_upper
        self.source_metadata=geometry.get('metadata',geometry.get('source_metadata'))
        for name in expected:getattr(self,name).setflags(write=False)

    def metadata(self):
        return dict(mode=self.mode,shape=list(self.shape),competitor_count=self.count,
            coefficients='sum_h u_h.T Q_h u_h + a_h.T u_h + c; a includes the full linear coefficient',
            baseline_J=self.J_c.tolist(),baseline_error=self.e_c,j_scales=self.scales.tolist(),
            array_hashes={k:array_hash(getattr(self,k)) for k in ('J_q','J_a','J_c','e_q','e_a','scales')},
            geometry_checks=self.symmetry,box_upper=self.box_upper,objective_upper=self.upper,resource_box_checks=self.box,
            error_relative_improvement_unavailable=self.e_c==0,caller_metadata=self.source_metadata)

    def values(self,u):
        values=[polynomial_change(self.J_q[j],self.J_a[j],u) for j in range(self.count)]
        gains=np.array([v[0] for v in values]);errors=np.array([v[1] for v in values])
        change,roundoff=polynomial_change(self.e_q,self.e_a,u)
        return gains,errors,change,roundoff

    def check(self,case,u,z):
        u=np.asarray(case.array(u),float);z=float(z)
        if not np.isfinite(z):raise ValueError('Nonfinite objective value')
        gains,error,change,eerror=self.values(u);margins=gains-z*self.scales
        tolerances=error+64*np.finfo(float).eps*abs(z*self.scales)
        error_margin=-change-z*self.e_c if self.mode=='protected' else None
        error_tolerance=eerror+64*np.finfo(float).eps*abs(z*self.e_c)
        physical=case.audit(u);distance_ok=bool(np.all(margins>=-tolerances))
        protected_ok=self.mode!='protected' or error_margin>=-error_tolerance
        return dict(passed=bool(physical['passed'] and z>=0 and z<=self.upper+64*np.finfo(float).eps*max(1.,self.upper) and distance_ok and protected_ok),
            physical=physical,z=z,J=(self.J_c+gains).tolist(),J0=self.J_c.tolist(),J_increment=gains.tolist(),
            distance_margins=margins.tolist(),distance_roundoff=tolerances.tolist(),distance_passed=distance_ok,
            error=self.e_c+change,error0=self.e_c,error_change=change,
            error_relative_improvement=None if self.e_c==0 else -change/self.e_c,
            protected_margin=error_margin,protected_roundoff=error_tolerance,protected_passed=bool(protected_ok),
            criterion='Original physical audit and scalar fsum expanded inequalities with arithmetic roundoff only; no action clipping',
            all_raw_quadratic_margins_nonnegative=bool(np.all(margins>=0) and (error_margin is None or error_margin>=0)))

    def best_value_at(self,case,u):
        u=np.asarray(case.array(u),float);gains,error,change,eerror=self.values(u)
        caps=list((gains-error)/self.scales)
        if self.mode=='protected' and self.e_c>0:caps.append((-change-eerror)/self.e_c)
        caps.append(self.upper);raw=float(min(caps))
        # Recompute the scalar z of the same action; never project/crop the action.
        z=max(0.,raw)
        audit=self.check(case,u,z)
        audit['direct_z_cap_after_roundoff']=raw
        return z,audit


class OracleRecorder(_Recorder):
    def __init__(self,directory,began):
        self.upper_history=[]
        super().__init__(directory,began)
    def event(self,row):
        if row.get('event')=='mip' and row.get('bound') is not None:
            self.upper_history.append(dict(upper_bound=row['bound'],native_runtime_seconds=row.get('native_runtime_seconds'),wall_seconds=row.get('wall_seconds')))
        super().event(row)


def certify_bounds(lower,analytic_upper,native_upper,history):
    tolerance=1e-7*max(1.,abs(lower))
    conflicts=[r for r in history if r['upper_bound'] < lower-tolerance]
    final_conflict=native_upper is not None and native_upper<lower-tolerance
    withdrawn=bool(conflicts or final_conflict)
    upper=analytic_upper if native_upper is None or withdrawn else min(analytic_upper,max(lower,native_upper))
    if analytic_upper<lower-tolerance:raise ValueError('Analytic upper bound conflicts with a verified action')
    upper=max(lower,upper)
    return dict(lower_bound=lower,upper_bound=upper,analytic_box_upper_bound=analytic_upper,
        raw_native_upper_bound=native_upper,native_upper_bound_withdrawn=withdrawn,
        terminal_bound_conflict=bool(final_conflict),historical_bound_conflicts=conflicts,
        absolute_gap=upper-lower,relative_gap=(upper-lower)/max(abs(lower),1e-10),
        bounds_closed_numerically=bool(upper-lower<=1e-7*max(1.,abs(lower),abs(upper))),
        bound_convention='MAXIMIZATION: verified action gives LB; native branch-and-bound and resource-box relaxation give UB. Cost is not this objective.',
        upper_bound_source='analytic_resource_box' if native_upper is None or withdrawn or analytic_upper<=native_upper else 'native_ObjBound',
        positive_existence_demonstrated=bool(lower>POSITIVE_THRESHOLD))


def finish(recorder,receipt):
    receipt.update(native_attributes=recorder.native_attributes,native_attribute_unavailable=recorder.unavailable_attributes,
        native_attribute_invalid=recorder.invalid_attributes,solver_source=_reference(__file__),
        recorder_source=_reference(Path(__file__).parents[1]/'flexibility_extended_solve_20261114/solver.py'),
        raw_bound_scope='Oracle geometry maximum common improvement on the unchanged physical resource domain. No topology recovery or non-oracle reliability guarantee.',
        global_optimality_verified=False)
    recorder.event(dict(event='finish',wall_seconds=perf_counter()-recorder.began,status=receipt['status'],
        source=receipt['source'],z=receipt['z'],lower_bound=receipt['bounds']['lower_bound'],upper_bound=receipt['bounds']['upper_bound']))
    if recorder.stream is not None:recorder.stream.close();recorder.stream=None
    files={}
    for name,path in (('progress',recorder.trace_path if recorder.trace_owned else None),('native_log',recorder.native_path if recorder.native_log_configured else None)):
        if path is not None and path.exists():files[name]=_reference(path)
    receipt.update(seconds=perf_counter()-recorder.began,diagnostic_files=files,record_directory=None if recorder.directory is None else str(recorder.directory),
        logging_errors=recorder.errors,trace_rows=recorder.trace_rows,native_log_configured=recorder.native_log_configured,
        trace_status='not_requested' if recorder.directory is None else 'recording_errors' if recorder.errors else 'complete')
    if recorder.directory is not None:
        try:
            path=recorder.directory/'diagnostics.json'
            with path.open('x',encoding='utf8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
            receipt['diagnostics_receipt']=_reference(path)
        except Exception as exc:
            recorder.error('write_oracle_diagnostics',exc);receipt.update(logging_errors=recorder.errors,trace_status='recording_errors')
    receipt['seconds']=perf_counter()-recorder.began
    return receipt


def plan(case,geometry,seeds,seconds,env,record_directory=None):
    if not np.isfinite(seconds) or seconds<=0:raise ValueError('Positive finite total quote seconds required')
    began=perf_counter();recorder=OracleRecorder(record_directory,began);g=Geometry(case,geometry)
    zero=np.zeros(case.shape);z,zero_audit=g.best_value_at(case,zero)
    if not zero_audit['passed']:raise ValueError('Zero action must be feasible in the original resource domain')
    point=zero.copy();bestz=0.;bestname='baseline_zero';records=[];best_audit=zero_audit
    for name,value in [('baseline_zero',zero),*[(str(n),v) for n,v in seed_items(seeds)]]:
        try:
            u=np.asarray(case.array(value),float);value_z,audit=g.best_value_at(case,u);fee=float(case.fee(u))
            if not np.isfinite(fee) or fee< -1e-9:raise ValueError('Invalid original fee')
            records.append(dict(name=name,feasible=audit['passed'],z=value_z,cost_yuan=fee,action_sha256=array_hash(u),direct_audit=audit))
            if audit['passed'] and value_z>bestz+1e-12:point,bestz,bestname,best_audit=u.copy(),value_z,name,audit
        except Exception as exc:records.append(dict(name=name,feasible=False,error=repr(exc)))
    initial=dict(name=bestname,z=bestz,cost_yuan=float(case.fee(point)),action_sha256=array_hash(point),
        audit=best_audit,candidates=records,preparation_seconds=perf_counter()-began)
    seedz=bestz;source='verified_seed_retained';native_upper=None;native_audit=None;native_status=None;invoked=False;exception=None
    remaining=seconds-(perf_counter()-began)
    if remaining>0:
        import gurobipy as gp
        try:
            with gp.Model('oracle_direction_'+g.mode,env=env) as model:
                for key,value in dict(OutputFlag=0,Threads=1,Seed=0,NonConvex=2,FeasibilityTol=1e-9,NumericFocus=1,MIPGap=1e-4).items():model.setParam(key,value)
                recorder.configure(model);u,_=case.attach(model);variable=model.addVar(lb=0.,ub=g.upper,name='common_improvement_z')
                def expr(Q,a):return sum(u[h]@Q[h]@u[h] for h in range(g.H))+(a*u).sum()
                for j in range(g.count):model.addConstr(expr(g.J_q[j]/g.scales[j],g.J_a[j]/g.scales[j])>=variable,name=f'all_native_distance_{j}')
                if g.mode=='protected':
                    if g.e_c>0:model.addConstr(expr(g.e_q/g.e_c,g.e_a/g.e_c)+variable<=0.,name='oracle_error_relative_improvement')
                    else:model.addConstr(expr(g.e_q,g.e_a)<=0.,name='oracle_zero_error_nonincrease')
                model.setObjective(variable,gp.GRB.MAXIMIZE);u.Start=point;variable.Start=bestz
                remaining=seconds-(perf_counter()-began)
                if remaining>0:
                    model.Params.TimeLimit=remaining;invoked=True
                    model.optimize(recorder.callback) if recorder.directory is not None else model.optimize()
                    recorder.capture(model);native_status=int(model.Status)
                    try:native_upper=_number(model.ObjBound,gp.GRB.INFINITY)
                    except (AttributeError,gp.GurobiError):pass
                    if model.SolCount:
                        candidate=np.asarray(u.X);native_z=float(variable.X);value_z,audit=g.best_value_at(case,candidate)
                        native_audit=dict(native_reported_z=native_z,recomputed_z=value_z,at_native_z=g.check(case,candidate,native_z),at_recomputed_z=audit,
                            objective_scalar_recomputed_from_same_action=True,action_was_changed=False)
                        if audit['passed'] and value_z>bestz+1e-12:
                            point,bestz,best_audit=candidate.copy(),value_z,audit;source='native_found_positive' if seedz<=POSITIVE_THRESHOLD else 'native_improved_seed'
        except Exception as exc:exception=repr(exc)
    bounds=certify_bounds(bestz,g.upper,native_upper,recorder.upper_history)
    receipt=dict(mode=g.mode,objective='maximize common relative improvement z',objective_sense='maximize',
        status='positive_witness_found' if bounds['positive_existence_demonstrated'] else 'no_positive_witness_found',
        feasible=True,z=bestz,source=source,initial=initial,final=best_audit,geometry=g.metadata(),bounds=bounds,
        positive_threshold=POSITIVE_THRESHOLD,positive_existence_demonstrated=bounds['positive_existence_demonstrated'],
        native_status=native_status,optimizer_invoked=invoked,solver_exception=exception,
        native_infeasibility_withdrawn=native_status==3,infeasibility_proved=False,
        native_returned_point_audit=native_audit,retained_verified_initial=source=='verified_seed_retained',
        improvement_from_verified_initial_z=bestz-seedz,cost_yuan=float(case.fee(point)),cost_is_objective=False,
        upper_bound_history=recorder.upper_history,action_sha256=array_hash(point),
        error_relative_improvement_unavailable=g.e_c==0,requested_seconds=float(seconds),
        time_scope='Geometry validation, box bound, original seed audits, model construction and native solve; final direct audit/logging wall time separately included in seconds',
        no_positive_scope='Zero is always feasible; failure to find z>1e-6 is not an infeasibility conclusion.',
        action_space='Full original H-by-R array, no resource compression, no changed fees, no auxiliary search or action clipping')
    return finish(recorder,receipt),point,bestz
