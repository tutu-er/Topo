"""Hourly SDP lifting for initial points, followed by original-domain checks.

The optional CVXPY worker receives arrays only. It may run in an already
installed conda environment. No SDP objective is treated as a certified bound.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np

BUFFER = 1e-8
ROUNDING_DRAWS = 16


def plain(value):
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, np.generic): return plain(value.item())
    if isinstance(value, dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write(path, value):
    Path(path).write_text(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def reference(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=sha256(path.read_bytes()).hexdigest())


def runtime():
    """Use installed packages only; never install or inject foreign site-packages."""
    if importlib.util.find_spec('cvxpy') is not None and importlib.util.find_spec('scs') is not None:
        return Path(sys.executable)
    envs=Path(sys.executable).resolve().parent.parent
    for name in ('poweropt','tdm'):
        directory=envs/name
        if (directory/'python.exe').is_file() and (directory/'Lib/site-packages/cvxpy/__init__.py').is_file() and (directory/'Lib/site-packages/scs/__init__.py').is_file():
            return directory/'python.exe'
    return None


def problem_arrays(case,evidence):
    H,R=case.shape
    lower,upper=np.asarray(case.lower,float),np.asarray(case.upper,float)
    if lower.shape!=(H,R) or upper.shape!=(H,R) or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower>upper):
        raise ValueError('Finite original hourly action bounds required')
    q=np.asarray(evidence.gate_q,float)
    if q.shape!=(len(evidence.rows),H,R,R):raise ValueError('Hourly MGF matrices required')
    guards=list(evidence.guards)
    result=dict(lower=lower,upper=upper,
        A_eq=np.asarray(getattr(case,'encoded_A_eq',case.A_eq),float).reshape(-1,H*R),
        b_eq=np.asarray(getattr(case,'encoded_b_eq',case.b_eq),float),
        A_ub=np.asarray(getattr(case,'encoded_A_ub',case.A_ub),float).reshape(-1,H*R),
        b_ub=np.asarray(getattr(case,'encoded_b_ub',case.b_ub),float),
        original_A_eq=np.asarray(case.A_eq,float).reshape(-1,H*R),original_b_eq=np.asarray(case.b_eq,float),
        original_A_ub=np.asarray(case.A_ub,float).reshape(-1,H*R),original_b_ub=np.asarray(case.b_ub,float),
        fee_plus=np.asarray(case.fee_plus,float)*case.dt,
        fee_minus=np.asarray(case.fee_minus,float)*case.dt,
        gate_q=q,gate_a=np.asarray(evidence.gate_a,float),gate_c=np.asarray(evidence.gate_c,float),
        guard_q=np.asarray([np.broadcast_to(g[3],(H,R,R)) for g in guards],float).reshape(-1,H,R,R),
        guard_a=np.asarray([g[4] for g in guards],float).reshape(-1,H,R))
    if any(not np.isfinite(x).all() for x in result.values()):raise ValueError('Finite SDP coefficients required')
    return result


def lifted_values(Q,a,c,mean,moments):
    return np.einsum('khij,hij->k',Q,moments)+2*np.einsum('khi,hi->k',a,mean)+c


def linear_audit(data, action, tolerance=1e-7):
    x=np.asarray(action,float).ravel()
    if x.shape!=data['lower'].ravel().shape or not np.isfinite(x).all():
        return dict(passed=False,reason='nonfinite_or_wrong_shape')
    eq=float(np.max(abs(data['A_eq']@x-data['b_eq']),initial=0.))
    ub=max(0.,float(np.max(data['A_ub']@x-data['b_ub'],initial=0.)),
           float(np.max(data['lower'].ravel()-x)),float(np.max(x-data['upper'].ravel())))
    return dict(passed=bool(eq<=tolerance and ub<=tolerance),maximum_equality_error=eq,
                maximum_inequality_violation=ub,tolerance=tolerance,
                scope='Encoded merged physical and node-time band linear rows, including declared numerical buffer; original-domain audit is separate')


def moment_audit(data,mean,moments):
    gate=lifted_values(data['gate_q'],data['gate_a'],data['gate_c'],mean,moments)
    guards=lifted_values(data['guard_q'],data['guard_a'],np.zeros(len(data['guard_q'])),mean,moments)
    blocks=[];rlt=0.
    for h,(mu,W,lo,hi) in enumerate(zip(mean,moments,data['lower'],data['upper'])):
        active=np.flatnonzero((lo!=0.)|(hi!=0.));m=mu[active];w=W[np.ix_(active,active)]
        Z=np.block([[np.ones((1,1)),m[None]],[m[:,None],w]])
        eig=np.linalg.eigvalsh((Z+Z.T)/2);threshold=1e-6*max(1.,float(abs(eig).max()))
        blocks.append(dict(hour=h,active_dimension=len(active),minimum_eigenvalue=float(eig.min()),
            numerical_rank=int(np.sum(eig>threshold)),rank_threshold=threshold,
            leading_eigenvalue_fraction=float(eig[-1]/sum(np.maximum(eig,0))) if np.any(eig>0) else None))
        l,u=lo[active],hi[active]
        if len(active):
            lows=[l[:,None]*m[None]+m[:,None]*l[None]-l[:,None]*l[None],
                  u[:,None]*m[None]+m[:,None]*u[None]-u[:,None]*u[None]]
            highs=[u[:,None]*m[None]+m[:,None]*l[None]-u[:,None]*l[None],
                   l[:,None]*m[None]+m[:,None]*u[None]-l[:,None]*u[None]]
            rlt=max(rlt,*[float(np.max(v-w)) for v in lows],*[float(np.max(w-v)) for v in highs])
    return dict(linear=linear_audit(data,mean),blocks=blocks,
        maximum_McCormick_violation=max(0.,rlt),gate_margins=gate,
        minimum_lifted_gate_minus_buffer=float(np.min(gate-BUFFER)) if len(gate) else None,
        scalar_guard_gains=guards,minimum_scalar_guard=float(np.min(guards)) if len(guards) else None,
        interpretation='Numerical primal diagnostics only; no validated dual certificate or QCQP lower bound')


def rounded_actions(mean,moments,seed,draws=ROUNDING_DRAWS):
    rng=np.random.default_rng(int(seed));factors=[];diagnostics=[]
    for h,(mu,W) in enumerate(zip(mean,moments)):
        covariance=(W+W.T)/2-np.outer(mu,mu)
        eig,vectors=np.linalg.eigh(covariance)
        factors.append(vectors*np.sqrt(np.maximum(eig,0))[None,:])
        diagnostics.append(dict(hour=h,minimum_covariance_eigenvalue=float(eig.min()),
            negative_eigenvalue_mass_clipped_for_rounding=float(-np.minimum(eig,0).sum())))
    answer={'mean':np.asarray(mean,float).copy()}
    for k in range(draws):
        answer[f'gaussian_{k:02d}']=np.asarray([mu+factor@rng.standard_normal(len(mu)) for mu,factor in zip(mean,factors)])
    return answer,diagnostics


def worker(problem_path,options_path,directory):
    """Only serialized coefficients enter the SDP process; no project imports."""
    began=perf_counter();directory=Path(directory)
    with np.load(problem_path,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    options=json.loads(Path(options_path).read_text(encoding='utf8'))
    import cvxpy as cp
    import scs
    H,R=data['lower'].shape
    active=[np.flatnonzero((data['lower'][h]!=0.)|(data['upper'][h]!=0.)) for h in range(H)]
    flat=np.concatenate([h*R+a for h,a in enumerate(active)]).astype(int)
    if not len(flat):
        write(directory/'worker.json',dict(status='all_actions_fixed_zero',optimizer_invoked=False,
            raw_sdp_objective_diagnostic=0.,backend='SCS',cvxpy_version=cp.__version__,scs_version=scs.__version__))
        np.savez_compressed(directory/'moments.npz',mean=np.zeros((H,R)),moments=np.zeros((H,R,R)))
        return
    x=cp.Variable(len(flat));absolute=cp.Variable(len(flat),nonneg=True)
    lo=data['lower'].ravel()[flat];hi=data['upper'].ravel()[flat]
    constraints=[x>=lo,x<=hi,absolute>=x,absolute>=-x]
    if len(data['b_eq']):constraints.append(data['A_eq'][:,flat]@x==data['b_eq'])
    if len(data['b_ub']):constraints.append(data['A_ub'][:,flat]@x<=data['b_ub'])
    means=[];moments=[];offset=0
    for h,indices in enumerate(active):
        n=len(indices)
        if n==0:means.append(None);moments.append(None);continue
        mu=x[offset:offset+n];W=cp.Variable((n,n),symmetric=True);offset+=n
        col=cp.reshape(mu,(n,1),order='C');row=cp.reshape(mu,(1,n),order='C')
        constraints.append(cp.bmat([[np.ones((1,1)),row],[col,W]]) >> 0)
        l=data['lower'][h,indices];u=data['upper'][h,indices]
        constraints.extend([
            W>=cp.multiply(l[:,None],row)+cp.multiply(col,l[None])-np.outer(l,l),
            W>=cp.multiply(u[:,None],row)+cp.multiply(col,u[None])-np.outer(u,u),
            W<=cp.multiply(u[:,None],row)+cp.multiply(col,l[None])-np.outer(u,l),
            W<=cp.multiply(l[:,None],row)+cp.multiply(col,u[None])-np.outer(l,u)])
        means.append(mu);moments.append(W)
    def lifted(q,a,c):
        expression=float(c)
        for h,indices in enumerate(active):
            if len(indices):expression+=cp.sum(cp.multiply(q[h][np.ix_(indices,indices)],moments[h]))+2*a[h,indices]@means[h]
        return expression
    for q,a,c in zip(data['gate_q'],data['gate_a'],data['gate_c']):
        value=lifted(q,a,c)
        if np.isscalar(value):
            if value<BUFFER:raise ValueError('Constant buffered gate infeasible')
        else:constraints.append(value>=BUFFER)
    for q,a in zip(data['guard_q'],data['guard_a']):
        value=lifted(q,a,0.)
        if not np.isscalar(value):constraints.append(value>=0.)
    plus=data['fee_plus'].ravel()[flat];minus=data['fee_minus'].ravel()[flat]
    objective=((plus-minus)@x+(plus+minus)@absolute)/2
    problem=cp.Problem(cp.Minimize(objective),constraints)
    remaining=max(.01,float(options['worker_seconds'])-(perf_counter()-began))
    problem.solve(solver='SCS',eps_abs=1e-5,eps_rel=1e-5,max_iters=100000,
                  time_limit_secs=remaining,verbose=False)
    stats=problem.solver_stats;extra=stats.extra_stats or {};info=extra.get('info',{})
    receipt=dict(status=problem.status,optimizer_invoked=True,backend='SCS',cvxpy_version=cp.__version__,
        scs_version=scs.__version__,python=sys.executable,raw_sdp_objective_diagnostic=problem.value,
        solver_solve_seconds=stats.solve_time,solver_setup_seconds=stats.setup_time,
        iterations=stats.num_iters,raw_solver_info=info,
        raw_primal_residual=info.get('res_pri'),raw_dual_residual=info.get('res_dual'),
        raw_primal_dual_gap=info.get('gap'),canonicalization_and_solve_wall_seconds=perf_counter()-began,
        constraints_count=len(constraints),active_scalar_actions=len(flat),fixed_zero_actions=H*R-len(flat),
        covariance_blocks=H,McCormick_envelopes=True,bound_validated=False,effective_qcqp_lower_bound=None)
    if x.value is not None and all(W is None or W.value is not None for W in moments):
        mean=np.zeros((H,R));Wfull=np.zeros((H,R,R))
        for h,indices in enumerate(active):
            if len(indices):
                mean[h,indices]=means[h].value
                Wfull[h][np.ix_(indices,indices)]=moments[h].value
        if np.isfinite(mean).all() and np.isfinite(Wfull).all():
            np.savez_compressed(directory/'moments.npz',mean=mean,moments=Wfull)
            receipt['moment_audit']=moment_audit(data,mean,Wfull)
    write(directory/'worker.json',receipt)


def project(case, point, seconds, env):
    """A convex physical-domain projection; no probability constraint is repaired."""
    import gurobipy as gp
    began=perf_counter()
    with gp.Model('sdp_physical_projection',env=env) as model:
        model.Params.OutputFlag=0;model.Params.Threads=1;model.Params.FeasibilityTol=1e-9
        u,_=case.attach(model,name='sdp_projection')
        scale=np.maximum(1.,np.maximum(abs(case.lower),abs(case.upper)))
        difference=(u-point)/scale
        model.setObjective((difference*difference).sum())
        model.Params.TimeLimit=max(.001,seconds-(perf_counter()-began))
        model.optimize()
        result=None if not model.SolCount else np.asarray(u.X).copy()
        return result,dict(status=int(model.Status),sol_count=int(model.SolCount),
            seconds=perf_counter()-began,normalized_squared_distance=None if result is None else float(np.square((result-point)/scale).sum()),
            scope='Convex projection onto original physical contracts plus node-time bands only')


def solve(case,evidence,*,seconds=120.,seed=0,record_directory=None):
    began=perf_counter()
    if not np.isfinite(seconds) or seconds<=0:raise ValueError('Positive finite SDP total budget required')
    directory=Path(record_directory).resolve() if record_directory is not None else Path(tempfile.mkdtemp(prefix='topo_sdp_'))
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'receipt.json').exists() or (directory/'problem.npz').exists():raise FileExistsError('Preserve earlier SDP attempt')
    receipt=dict(mode=evidence.mode,seed=int(seed),requested_seconds=float(seconds),source=reference(__file__),
        role='Initial-point generator only; original QCQP subsequently classifies every resource-feasible warm point',
        linear_encoding='Use case.encoded_A_eq/b_eq and encoded_A_ub/b_ub when supplied, otherwise original rows; projection calls case.attach; final original acceptance uses case.audit',
        bound_validated=False,effective_qcqp_lower_bound=None,raw_sdp_objective_diagnostic=None,
        returned_probability_feasible_count=0,returned_resource_only_count=0,candidates=[],
        time_scope='Serialization, existing-runtime startup, SDP construction/solve, Gaussian rounding, physical projections and original direct audits included')
    accepted={};files={}
    def finish(status):
        receipt.update(status=status,seconds=perf_counter()-began,returned_candidates=list(accepted),
            budget_overrun_seconds=max(0.,perf_counter()-began-seconds))
        for name in ('problem.npz','options.json','worker.json','moments.npz','candidate_actions.npz','stdout.txt','stderr.txt'):
            path=directory/name
            if path.exists():files[name]=reference(path)
        receipt['files']=files
        write(directory/'receipt.json',receipt)
        return receipt,accepted
    if evidence.mode not in ('mgf_fixed','mgf_forecast'):
        receipt['reason']='Only original quadratic MGF margins are lifted; normal square-root law is not replaced'
        return finish('unsupported_mode')
    executable=runtime()
    if executable is None:return finish('sdp_backend_unavailable')
    data=problem_arrays(case,evidence)
    np.savez_compressed(directory/'problem.npz',**data)
    worker_allowance=max(.01,.7*seconds-(perf_counter()-began))
    write(directory/'options.json',dict(worker_seconds=worker_allowance,seed=int(seed),rounding_draws=ROUNDING_DRAWS,
        rows=evidence.rows,retained_guard_pairs=[[int(g[0]),int(g[1])] for g in evidence.guards],
        gate_buffer=BUFFER,physical_domain='Merged encoded state restoration and node-time P/Q bands; original linear rows retained separately for audit'))
    receipt['runtime_python']=str(executable);tick=perf_counter()
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[key]='1'
    try:
        completed=subprocess.run([str(executable),'-B',str(Path(__file__).resolve()),'--worker',
            str(directory/'problem.npz'),str(directory/'options.json'),str(directory)],
            capture_output=True,text=True,encoding='utf8',errors='replace',env=env,
            timeout=max(.01,.78*seconds-(perf_counter()-began)),
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        (directory/'stdout.txt').write_text(completed.stdout,encoding='utf8')
        (directory/'stderr.txt').write_text(completed.stderr,encoding='utf8')
        receipt.update(worker_returncode=completed.returncode,worker_wall_seconds=perf_counter()-tick)
    except subprocess.TimeoutExpired as exc:
        for name,value in (('stdout.txt',exc.stdout),('stderr.txt',exc.stderr)):
            if value is not None:
                (directory/name).write_text(value.decode('utf8','replace') if isinstance(value,bytes) else value,encoding='utf8')
        receipt.update(worker_wall_seconds=perf_counter()-tick,worker_timeout=True,
            error='SDP worker terminated at allocated total wall time; no new seed available')
        return finish('sdp_worker_timeout')
    if not (directory/'worker.json').exists():
        return finish('sdp_worker_failed')
    result=json.loads((directory/'worker.json').read_text(encoding='utf8'));receipt['relaxation']=result
    receipt['raw_sdp_objective_diagnostic']=result.get('raw_sdp_objective_diagnostic')
    if not (directory/'moments.npz').exists():return finish('sdp_no_moment_point')
    with np.load(directory/'moments.npz',allow_pickle=False) as archive:
        mean,moments=archive['mean'].copy(),archive['moments'].copy()
    receipt['independent_moment_audit']=moment_audit(data,mean,moments)
    candidates,rounding=rounded_actions(mean,moments,seed)
    receipt['rounding_covariance_diagnostics']=rounding
    saved={};seen=set();projection_env=None
    try:
        for name,point in candidates.items():
            if perf_counter()-began>=seconds:
                receipt['candidates'].append(dict(name=name,status='not_attempted_budget_exhausted'));continue
            tick=perf_counter();saved['raw_'+name]=point
            audit=linear_audit(data,point);record=dict(name=name,raw_linear_audit=audit,
                admitted_to_pool=False,probability_feasible=None,pool_name='sdp_'+name)
            if not audit['passed'] or not case.audit(point)['passed']:
                try:
                    if projection_env is None:
                        import gurobipy as gp
                        projection_env=gp.Env(empty=True);projection_env.setParam('OutputFlag',0);projection_env.start()
                    remaining=seconds-(perf_counter()-began)
                    if remaining<=.02:raise TimeoutError('No projection budget remains')
                    point,projection=project(case,point,min(2.,remaining),projection_env)
                    record['projection']=projection
                except Exception as exc:record.update(status='projection_failed',error=repr(exc));point=None
            if point is not None:
                saved['projected_'+name]=point
                physical=case.audit(point);linear=linear_audit(data,point)
                record.update(physical_audit=physical,independent_linear_audit=linear,
                    activation_cost_yuan=float(case.fee(point)))
                if physical['passed'] and linear['passed']:
                    try:
                        original=evidence.verify_expansion(point);record['original_evidence_audit']=original
                        valid=bool(original['feasible']);record['probability_feasible']=valid
                        digest=sha256(np.ascontiguousarray(point).tobytes()).hexdigest()
                        record['action_sha256']=digest
                        if digest in seen:record['status']='duplicate_action'
                        else:
                            seen.add(digest);accepted['sdp_'+name]=point.copy()
                            record['admitted_to_pool']=True
                            record['status']='original_probability_feasible_seed' if valid else 'resource_feasible_probability_infeasible_warm_point'
                            key='returned_probability_feasible_count' if valid else 'returned_resource_only_count'
                            receipt[key]+=1
                    except Exception as exc:record.update(status='original_audit_failed',error=repr(exc))
                else:record['status']='physical_or_band_rejected'
            record['seconds']=perf_counter()-tick;receipt['candidates'].append(record)
    finally:
        if projection_env is not None:projection_env.dispose()
    if saved:
        np.savez_compressed(directory/'candidate_actions.npz',**saved,**{'returned_'+k:v for k,v in accepted.items()})
        action_ref=reference(directory/'candidate_actions.npz')
        for record in receipt['candidates']:
            if record.get('admitted_to_pool'):
                record.update(action_ref=action_ref,array_key='returned_'+record['pool_name'])
    return finish('seeds_generated' if accepted else 'no_accepted_candidate')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',nargs=3,required=True)
    args=parser.parse_args();worker(*args.worker)
