"""History-only diagnosis of retained common-gradient precision failures.

No nonlinear refits or EV actions are selected. The only proposed steps solve
frozen-Jacobian bounded GN models and are independently replayed through AC.
All tolerance hooks are restored within this process.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from decimal import Decimal, localcontext
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import time

import numpy as np

from .. import likelihood as lk
from ..ev_rx_fit import _RxInputProblem
from ..information_context import meter_from_record
from . import benchmark
from .blocks import column_norms, gradient, matvec, problem_blocks, projected_gradient
from .linear import solve_block_step


def stable_changes(residual, trial, jacobian_step=None):
    """Equivalent reductions, with long compensated summation as a check."""
    r, rn = np.asarray(residual, float).ravel(), np.asarray(trial, float).ravel()
    change = rn-r
    answer = dict(raw_actual=.5*float(r@r-rn@rn),
        stable_actual=.5*float((r-rn)@(r+rn)),
        compensated_actual=.5*math.fsum(float(a)*float(b) for a,b in zip(r-rn,r+rn)),
        absolute_residual_change=float(np.max(abs(change), initial=0.)))
    if jacobian_step is not None:
        jd = np.asarray(jacobian_step, float).ravel()
        answer.update(raw_predicted=.5*float(r@r-(r+jd)@(r+jd)),
            stable_predicted=-float(r@jd)-.5*float(jd@jd),
            compensated_predicted=-math.fsum(float(a)*float(b) for a,b in zip(r,jd))
                -.5*math.fsum(float(a)*float(a) for a in jd))
    return answer


@contextmanager
def _ac_precision(tol):
    original = lk.ac_predict
    def prediction(*args, **kwargs):
        kwargs['tol'] = tol
        return original(*args, **kwargs)
    lk.ac_predict = prediction
    try:
        yield
    finally:
        lk.ac_predict = original


def _build(whitelist, endpoint):
    package = benchmark.load_history_package(whitelist)
    record = benchmark._read(endpoint)
    task = record['task']
    item = package['items'][task['network']]
    declaration = next(row['candidate'] for row in item['selected_candidates']
        if row['candidate']['name'] == task['candidate_name'])
    candidate = lk.RadialCandidate.from_supports(declaration['name'], declaration['n'], declaration['supports'])
    observations = lk.Observations(**item['observed_history'])
    model = meter_from_record(item['history_meter'])
    options = item['fit_options']
    domain = {k:options[k] for k in ('r_bounds','x_bounds','ratio_bounds',
        'source_p_bounds','source_q_bounds','voltage_bounds')}
    problem = _RxInputProblem(candidate, observations, model, initial=record['state'], **domain)
    r,alpha = problem.mapping.inverse(record['state']['r'],record['state']['x'])
    full = np.r_[r,alpha,np.c_[record['state']['source_p'],record['state']['source_q']].ravel()]
    vector = full[problem.free]
    return problem,vector,record


def _label(problem,index):
    full = int(np.flatnonzero(problem.free)[index])
    if full < problem.m:
        return 'R_'+str(full)
    if full < problem.shared:
        return 'alpha_'+str(full-problem.m)
    local = full-problem.shared
    return ('source_P_' if local%2==0 else 'source_Q_')+str(local//2)


def _column(line,source,index):
    d = line.shape[-1]
    if index < d:
        return line[:,:,index].ravel()
    local = index-d
    answer = np.zeros(source.shape[:2])
    answer[local//2] = source[local//2,:,local%2]
    return answer.ravel()


def _evaluate(problem, vector, tol=2e-13):
    # A fresh cache is required when changing AC accuracy at identical x.
    problem.base.cache = {}
    with _ac_precision(tol):
        residual = problem.evaluate(vector).copy()
        line,source = problem_blocks(problem,vector)
    g = gradient(line,source,residual)
    norms = column_norms(line,source)
    jac_scale = np.divide(1.,norms,out=np.ones_like(norms),where=norms>0)
    physical = problem.base.cache['physical']
    summary = dict(loss=float(residual@residual),common_pg=projected_gradient(
        vector,g,problem.lower,problem.upper,problem.scale),instantaneous_jac_pg=projected_gradient(
        vector,g,problem.lower,problem.upper,jac_scale),gradient_infinity=float(np.max(abs(g))),
        ac_equation_residual=float(physical['ac_equation_residual']),
        power_balance_residual=float(physical['power_balance_residual']),
        ac_iterations=int(physical['iterations']))
    return residual,line,source,g,jac_scale,summary


def _residual_only(problem,vector,tol=1e-15):
    problem.base.cache = {}
    with _ac_precision(tol):
        return problem.evaluate(vector).copy()


def _grad_details(problem,vector,residual,line,source,g):
    pg = np.clip(-g*problem.scale,(problem.lower-vector)/problem.scale,
                 (problem.upper-vector)/problem.scale)
    top = np.argsort(-abs(pg))[:8]
    rows = []
    for index in top:
        derivative = _column(line,source,index)
        terms = (derivative*residual).reshape(problem.t,problem.n+2)
        channel = dict(v=math.fsum(float(z) for z in terms[:,:problem.n].ravel()),
            master_p=math.fsum(float(z) for z in terms[:,problem.n]),
            master_q=math.fsum(float(z) for z in terms[:,problem.n+1]))
        compensated = math.fsum(float(z) for z in terms.ravel())
        rows.append(dict(index=int(index),label=_label(problem,int(index)),
            pg=float(pg[index]),raw_gradient=float(g[index]),fixed_scale=float(problem.scale[index]),
            compensated_gradient=compensated,compensated_minus_plain=compensated-float(g[index]),
            channel_contributions=channel,
            sum_abs_terms=float(np.sum(abs(terms))),
            cancellation_ratio=float(np.sum(abs(terms)))/max(abs(compensated),1e-300),
            value=float(vector[index]),lower=float(problem.lower[index]),upper=float(problem.upper[index])))
    full = problem.expand(vector)
    _,_,width,_,_ = problem.mapping.section(full[:problem.m])
    alpha = full[problem.m:problem.shared]
    return rows,dict(collapsed_sections=int(np.count_nonzero(width==0)),
        minimum_section_width=float(np.min(width)),
        alpha_near_lower=int(np.count_nonzero(alpha<=1e-10)),
        alpha_near_upper=int(np.count_nonzero(alpha>=1.-1e-10)),
        top_coordinate_is_interior=bool(vector[top[0]]>problem.lower[top[0]]+1e-9
            and vector[top[0]]<problem.upper[top[0]]-1e-9))


def _fd(problem,vector,residual,line,source,g):
    d = line.shape[-1]
    pg = np.clip(-g*problem.scale,(problem.lower-vector)/problem.scale,
                 (problem.upper-vector)/problem.scale)
    selected = list(np.argsort(-abs(pg))[:4])
    for index in (problem.m,problem.m+5):
        if index < d and index not in selected:
            selected.append(index)
    rows = []
    center = _residual_only(problem,vector)
    for index in selected[:8]:
        exact = _column(line,source,int(index))
        checks = []
        for size in (1e-3,1e-4,1e-5,1e-6,1e-7):
            h = float(size*problem.scale[index])
            plus,minus = vector.copy(),vector.copy()
            plus[index] += h;minus[index] -= h
            if plus[index]<=problem.upper[index] and minus[index]>=problem.lower[index]:
                estimate = (_residual_only(problem,plus)-_residual_only(problem,minus))/(2*h)
                scheme = 'centered'
            elif plus[index]+h<=problem.upper[index]:
                double = vector.copy();double[index]+=2*h
                estimate = (-3*center+4*_residual_only(problem,plus)-_residual_only(problem,double))/(2*h)
                scheme = 'forward_3point'
            elif minus[index]-h>=problem.lower[index]:
                double = vector.copy();double[index]-=2*h
                estimate = (3*center-4*_residual_only(problem,minus)+_residual_only(problem,double))/(2*h)
                scheme = 'backward_3point'
            else:
                continue
            relative = float(np.linalg.norm(estimate-exact)/max(np.linalg.norm(exact),1e-300))
            checks.append(dict(scaled_h=size,h=h,scheme=scheme,relative_column_error=relative,
                fd_gradient=math.fsum(float(a)*float(b) for a,b in zip(estimate,center)),
                exact_gradient=float(g[index])))
        rows.append(dict(index=int(index),label=_label(problem,int(index)),checks=checks,
            best_relative_column_error=min((c['relative_column_error'] for c in checks),default=None)))
    return rows


def _profile_spectrum(line,source,scale):
    projected=[]
    for time in range(len(line)):
        u,s,_=np.linalg.svd(source[time],full_matrices=False)
        rank=int(np.count_nonzero(s>np.finfo(float).eps*max(source[time].shape)*s[0]))
        block=line[time]*scale[None,:]
        projected.append(block-u[:,:rank]@(u[:,:rank].T@block))
    matrix=np.vstack(projected)
    _,singular,right=np.linalg.svd(matrix,full_matrices=False)
    return dict(singular_values=singular.tolist(),condition=float(singular[0]/singular[-1])
        if singular[-1]>0 else None,rank_threshold=float(np.finfo(float).eps*max(matrix.shape)*singular[0]),
        smallest_direction=right[-1].tolist(),meaning='local source-profile derivative; not a confidence set')


def _linear_trials(problem,vector,residual,line,source,jac_scale,baseline):
    d=line.shape[-1]
    gl=line*jac_scale[:d][None,None,:]
    sl=source*jac_scale[d:].reshape(problem.t,1,2)
    records=[]
    for damping in (1e-12,1e-8,1e-4,.01,1.,10.):
        step=solve_block_step(residual.reshape(source.shape[:2]),gl,sl,
            (problem.lower-vector)/jac_scale,(problem.upper-vector)/jac_scale,
            damping=damping,kkt_tol=1e-10,max_iter=2000)
        eta=step['step'];delta=jac_scale*eta
        trial=np.maximum(problem.lower,np.minimum(vector+delta,problem.upper))
        actual_delta=trial-vector
        jd=matvec(line,source,actual_delta).ravel()
        rr,ll,ss,gg,aa,after=_evaluate(problem,trial)
        metrics=stable_changes(residual,rr,jd)
        records.append(dict(damping=damping,inner_success=step['success'],inner_kkt=step['kkt'],
            inner_error_bound=step['error_bound'],step_norm=float(np.linalg.norm(actual_delta)),
            largest_step=float(np.max(abs(actual_delta))),loss=after['loss'],common_pg=after['common_pg'],
            common_pg_ratio=after['common_pg']/baseline['common_pg'],
            reference_loss_difference=after['loss']-baseline['loss'],**metrics))
    return records


def decimal_evaluate(problem,vector,*,digits=50,timeout=40.):
    """Higher precision AC using only the standard-library Decimal module.

    Decimal.from_float preserves the exact binary input numbers. The complex
    arithmetic is written as real/imaginary pairs. It solves the same physical
    constant-power equations and does not evaluate or alter optimizer states.
    This is an arithmetic cross-check, not an interval certificate.
    """
    began=time.perf_counter()
    base=problem.base
    r,x,source=problem.unpack(vector)
    def dec(z):return Decimal.from_float(float(z))
    with localcontext() as context:
        context.prec=digits
        zero=Decimal(0);one=Decimal(1);half=Decimal('.5')
        m=problem.m
        zr,zx=[dec(v) for v in r],[dec(v) for v in x]
        b=base.candidate.B
        zbus=[]
        for i in range(m):
            row=[]
            for j in range(m):
                row.append((sum((dec(b[i,e]*b[j,e])*zr[e] for e in range(m)),zero),
                            sum((dec(b[i,e]*b[j,e])*zx[e] for e in range(m)),zero)))
            zbus.append(row)
        residual=[];voltage_rows=[];master_rows=[];max_iterations=0
        threshold=Decimal(10)**(-(digits-8))
        for time_index in range(problem.t):
            if time.perf_counter()-began>timeout:
                raise TimeoutError('declared Decimal arithmetic wall budget exceeded')
            root=dec(base.observations.root_v[time_index])
            lp=[zero for _ in range(m)];lq=[zero for _ in range(m)]
            lp[0],lq[0]=dec(source[time_index,0]),dec(source[time_index,1])
            for terminal,node in enumerate(base.candidate.terminal_edges):
                lp[node]+=dec(base.p[time_index,terminal]);lq[node]+=dec(base.q[time_index,terminal])
            nonzero=[j for j in range(m) if lp[j] or lq[j]]
            vr=[root for _ in range(m)];vi=[zero for _ in range(m)]
            def currents(real,imag):
                ir=[zero for _ in range(m)];ii=[zero for _ in range(m)]
                for j in nonzero:
                    denominator=real[j]*real[j]+imag[j]*imag[j]
                    ir[j]=(lp[j]*real[j]+lq[j]*imag[j])/denominator
                    ii[j]=(lp[j]*imag[j]-lq[j]*real[j])/denominator
                return ir,ii
            for iteration in range(1,401):
                ir,ii=currents(vr,vi)
                nr=[];ni=[]
                for i in range(m):
                    real=sum((zbus[i][j][0]*ir[j]-zbus[i][j][1]*ii[j] for j in nonzero),zero)
                    imag=sum((zbus[i][j][0]*ii[j]+zbus[i][j][1]*ir[j] for j in nonzero),zero)
                    nr.append(root-real);ni.append(-imag)
                delta=max(max(abs(a-b) for a,b in zip(nr,vr)),max(abs(a-b) for a,b in zip(ni,vi)))
                vr,vi=nr,ni
                if delta<=threshold:break
            else:
                raise RuntimeError('Decimal AC failed to converge')
            max_iterations=max(max_iterations,iteration)
            ir,ii=currents(vr,vi)
            mp=root*sum(ir,zero);mq=-root*sum(ii,zero)
            magnitudes=[]
            for terminal,node in enumerate(base.candidate.terminal_edges):
                magnitude=(vr[node]*vr[node]+vi[node]*vi[node]).sqrt()
                magnitudes.append(magnitude)
                residual.append((magnitude+dec(base.bias['v'][time_index,terminal])
                    -dec(base.observations.v[time_index,terminal]))/dec(base.sigmas['v'][time_index,terminal]))
            residual.append((mp-dec(base.observations.master_p[time_index]))/dec(base.sigmas['master_p'][time_index]))
            residual.append((mq-dec(base.observations.master_q[time_index]))/dec(base.sigmas['master_q'][time_index]))
            voltage_rows.append(magnitudes);master_rows.append((mp,mq))
        objective=half*sum((z*z for z in residual),zero)
        return dict(objective=objective,residual=residual,voltage=voltage_rows,master=master_rows,
                    digits=digits,iterations=max_iterations,seconds=time.perf_counter()-began)


def _polish(problem,vector,*,target=1e-5,max_steps=10):
    r,line,source,g,scale,before=_evaluate(problem,vector)
    fixed_scale=scale.copy();d=line.shape[-1];current=vector.copy()
    trace=[];entry_F=.5*before['loss']
    allowance=256*np.finfo(float).eps*max(1.,entry_F)
    for iteration in range(max_steps):
        if before['common_pg']<=target:break
        gl=line*fixed_scale[:d][None,None,:]
        sl=source*fixed_scale[d:].reshape(problem.t,1,2)
        step=solve_block_step(r.reshape(source.shape[:2]),gl,sl,
            (problem.lower-current)/fixed_scale,(problem.upper-current)/fixed_scale,
            damping=1e-4,kkt_tol=1e-10,max_iter=2000)
        trial=np.maximum(problem.lower,np.minimum(current+fixed_scale*step['step'],problem.upper))
        rr,ll,ss,gg,aa,after=_evaluate(problem,trial)
        metrics=stable_changes(r,rr,matvec(line,source,trial-current).ravel())
        voltage=abs(problem.base.cache['physical']['voltage'])
        rtrial,xtrial,_=problem.unpack(trial)
        physical=bool(np.min(voltage)>=problem.base.voltage_bounds[0] and np.max(voltage)<=problem.base.voltage_bounds[1]
            and problem.mapping.contains(rtrial,xtrial)
            and np.all(trial>=problem.lower) and np.all(trial<=problem.upper)
            and after['ac_equation_residual']<=2e-10 and after['power_balance_residual']<=2e-10)
        accepted=bool(metrics['compensated_predicted']>0 and physical
            and metrics['compensated_actual']>=-allowance
            and .5*after['loss']<=entry_F+allowance
            and after['common_pg']<before['common_pg'])
        trace.append(dict(iteration=iteration,accepted=accepted,physical_pass=physical,
            damping=1e-4,loss_before=before['loss'],loss_after=after['loss'],
            common_pg_before=before['common_pg'],common_pg_after=after['common_pg'],
            numerical_loss_allowance=allowance,inner_kkt=step['kkt'],**metrics))
        if not accepted:break
        current,r,line,source,before=trial,rr,ll,ss,after
    return current,dict(target=target,max_steps=max_steps,trace=trace,
        success=bool(before['common_pg']<=target),final=before,
        step_scale='instantaneous inverse Jacobian column norms at polish entry, held fixed',
        fixed_step_scale=fixed_scale.tolist(),lambda_definition='lambda/2 * ||physical_step / fixed_step_scale||^2, lambda=1e-4',
        acceptance_is_strict_objective_descent=False,allowance_is_certified_objective_bound=False)


def polish_diagnosis(whitelist,output):
    output=Path(output);plan=benchmark._read(output/'polish_plan.json')
    result=benchmark._read(output/'result_fast.json')
    rows=[]
    for file in ('reference_jac_precision_0.json','block_jac_precision_0.json'):
        path=benchmark.STAGE/'outputs/cpu_precision_v1'/file
        problem,vector,record=_build(whitelist,path)
        original,line,source,g,scale,before=_evaluate(problem,vector)
        polished,details=_polish(problem,vector,target=plan['common_pg_target'],max_steps=plan['max_steps'])
        numeric={}
        try:
            high_before=decimal_evaluate(problem,vector,digits=50,timeout=40.)
            high_after=decimal_evaluate(problem,polished,digits=50,timeout=40.)
            with localcontext() as context:
                context.prec=50
                actual=high_before['objective']-high_after['objective']
                selected=[]
                for index in np.argsort(-abs(g*problem.scale))[:4]:
                    column=_column(line,source,int(index))
                    high_gradient=sum((Decimal.from_float(float(a))*b for a,b in zip(column,high_before['residual'])),Decimal(0))
                    selected.append(dict(index=int(index),label=_label(problem,int(index)),
                        fp64_gradient=float(g[index]),decimal_residual_gradient=str(high_gradient),
                        fixed_scale_difference=float((high_gradient-Decimal.from_float(float(g[index])))*Decimal.from_float(float(problem.scale[index])))))
            numeric=dict(status='complete',digits=50,mpmath_installed=False,
                longdouble_same_precision_as_float64=bool(np.finfo(np.longdouble).eps==np.finfo(float).eps),
                F_before=str(high_before['objective']),F_after=str(high_after['objective']),
                higher_precision_actual_decrease=str(actual),seconds=high_before['seconds']+high_after['seconds'],
                selected_gradient_using_decimal_residual=selected,
                note='AC/residual/objective use Decimal50; these diagnostic gradient products keep the analytic FP64 Jacobian. No interval certification.')
        except (TimeoutError,RuntimeError) as error:
            numeric=dict(status='failed',error=repr(error))
        rows.append(dict(endpoint=file,before=before,polish=details,high_precision=numeric,
            retained_state=problem.state_record(polished)))
    payload=dict(status='diagnosis_and_polish_complete',created_utc=datetime.now(timezone.utc).isoformat(),
        original_frozen_batch_modified=False,nonlinear_solver_source_modified=False,EV_actions_selected=0,
        polish_plan=plan,endpoints=rows,fast_diagnosis_sha256=hashlib.sha256((output/'result_fast.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    benchmark._write(output/'polish_result.json',payload)
    return payload


def run(whitelist,output):
    output=Path(output)
    plan=benchmark._read(output/'plan.json')
    if plan['status']!='declared_before_diagnostic_computation':
        raise ValueError('a saved diagnostic plan is required')
    endpoints=[]
    for file in plan['inputs'][:2]:
        path=benchmark.STAGE/file
        problem,vector,record=_build(whitelist,path)
        residual,line,source,g,jac_scale,summary=_evaluate(problem,vector)
        top,boundary=_grad_details(problem,vector,residual,line,source,g)
        tighter=[]
        for tol in (2e-15,1e-15):
            rr,ll,ss,gg,aa,state=_evaluate(problem,vector,tol)
            tighter.append(dict(ac_tolerance=tol,**state,
                residual_max_change=float(np.max(abs(rr-residual))),
                gradient_max_change=float(np.max(abs(gg-g))),
                common_gradient_change_ratio=abs(state['common_pg']-summary['common_pg'])/summary['common_pg']))
        fd=_fd(problem,vector,residual,line,source,g)
        trials=_linear_trials(problem,vector,residual,line,source,jac_scale,summary)
        endpoints.append(dict(endpoint=path.name,endpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            baseline=summary,top_coordinates=top,boundary=boundary,tighter_AC=tighter,
            finite_differences=fd,
            profile_fixed_scale=_profile_spectrum(line,source,problem.scale[:line.shape[-1]]),
            profile_jac_scale=_profile_spectrum(line,source,jac_scale[:line.shape[-1]]),
            frozen_model_trials=trials,
            maximum_feasible_trial_gradient_improvement=max(summary['common_pg']/r['common_pg'] for r in trials),
            planned_thresholds=plan['tests']))
    result=dict(status='fast_diagnosis_complete',created_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=hashlib.sha256((output/'plan.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_whitelist_sha256=hashlib.sha256(Path(whitelist).read_bytes()).hexdigest(),
        nonlinear_fits_started=0,EV_actions_selected=0,original_frozen_batch_modified=False,
        high_precision_status='not yet attempted',endpoints=endpoints)
    benchmark._write(output/'result_fast.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--whitelist',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--polish',action='store_true')
    args=parser.parse_args()
    if args.polish:
        result=polish_diagnosis(args.whitelist,args.output)
        print(benchmark._serial(dict(status=result['status'],endpoints=[dict(endpoint=e['endpoint'],
            success=e['polish']['success'],final_pg=e['polish']['final']['common_pg'],high_precision=e['high_precision']) for e in result['endpoints']])))
        return
    result=run(args.whitelist,args.output)
    print(benchmark._serial(dict(status=result['status'],endpoints=[dict(endpoint=e['endpoint'],baseline=e['baseline'],
        best_pg=min(r['common_pg'] for r in e['frozen_model_trials'])) for e in result['endpoints']])))


if __name__=='__main__':
    main()
