"""Experiment-only bounded finite-pool LP enumeration and rooted metrics.

No production solver implementation or default is changed. The adapter is
installed only inside the worker process for the duration of one experiment.
"""
from __future__ import annotations
from dataclasses import replace
from time import perf_counter
import numpy as np
from rnj_wzzt.estimation import laminar_l1_milp as core

class ExperimentBudgetExpired(RuntimeError):pass

class BoundedPath:
    def __init__(self, *, total_seconds=12.0, solve_seconds=2.0, enumerate_pool=True):
        self.total_seconds=float(total_seconds);self.solve_seconds=float(solve_seconds)
        self.enumerate_pool=bool(enumerate_pool);self.solves=[];self.enumerations=[];self.path=[];self.attempts=[]
        self.r_bound=2.0;self.x_bound=2.0;self.initial_count=0;self.bound_restart_count=0
    def __enter__(self):
        self.started=perf_counter();self.deadline=self.started+self.total_seconds
        self.original={name:getattr(core,name) for name in ('_run_milp','_solve_extension_prepared','_make_path_point','_solve_fixed_prepared')}
        core._run_milp=self.run_solver;core._solve_extension_prepared=self.extension
        core._make_path_point=self.make_point;core._solve_fixed_prepared=self.fixed
        return self
    def __exit__(self,*args):
        for name,value in self.original.items():setattr(core,name,value)
    def remaining(self):return self.deadline-perf_counter()
    def run_solver(self,*args,**kw):
        if self.remaining()<=0:raise ExperimentBudgetExpired('experiment total model budget exhausted')
        kw['time_limit']=min(self.solve_seconds,self.remaining(),kw.get('time_limit') or self.solve_seconds)
        result,diag=self.original['_run_milp'](*args,**kw);self.solves.append(diag.to_dict())
        return result,diag
    def fixed(self,prepared,supports,**kw):
        self.r_bound=float(kw['r_upper_bound']);self.x_bound=float(kw['x_upper_bound'])
        if self.remaining()<=0:raise ExperimentBudgetExpired('experiment total model budget exhausted before LP')
        return self.original['_solve_fixed_prepared'](prepared,supports,**kw)
    def make_point(self,**kw):
        p=self.original['_make_path_point'](**kw)
        if p.iteration==self.initial_count:
            if self.path:self.bound_restart_count+=1
            self.path=[]
        self.path.append(p)
        return p
    def diagnostic(self,status,message,objective=None,elapsed=0.0):
        return core.SolverDiagnostics(status=status,success=status==0,message=message,objective=objective,
            dual_bound=objective if status==0 else None,mip_gap=0.0 if status==0 else None,
            node_count=0,runtime_seconds=elapsed,variable_count=0,binary_variable_count=0,constraint_count=0)
    def blank(self,prepared,diag):
        return core.ExtensionSolution(support=None,r_values=np.empty(0),x_values=np.empty(0),
            intercepts=np.empty((prepared.scenario_count,prepared.n)),objective=diag.objective,diagnostics=diag)
    def extension(self,prepared,supports,**kw):
        start=perf_counter();pool=kw.get('candidate_supports')
        if self.remaining()<=0:
            answer=self.blank(prepared,self.diagnostic(1,'experiment total budget exhausted before extension'))
        elif not self.enumerate_pool or pool is None:
            answer=self.original['_solve_extension_prepared'](prepared,supports,**kw)
        else:
            candidates=tuple(c for c in pool if core.is_admissible_extension(c,supports,prepared.n))
            record={'existing_supports':supports,'candidate_count':len(candidates),'candidate_results':[]}
            if not candidates:
                answer=self.blank(prepared,self.diagnostic(2,'finite candidate pool has no admissible extension'))
            else:
                best=None;best_support=None;all_optimal=True
                for support in candidates:
                    if self.remaining()<=0:
                        all_optimal=False;record['budget_exhausted']=True;break
                    st=perf_counter()
                    try:
                        solution=self.fixed(prepared,(*supports,support),r_upper_bound=kw['r_upper_bound'],
                            x_upper_bound=kw['x_upper_bound'],time_limit=self.solve_seconds,presolve=kw['presolve'],disp=kw['disp'])
                        certified=core.solver_diagnostics_prove_optimality(solution.diagnostics)
                        record['candidate_results'].append({'support':support,'objective':solution.objective,
                            'certified_optimal':certified,'diagnostics':solution.diagnostics.to_dict(),'wall_seconds':perf_counter()-st})
                        all_optimal &= certified
                        if certified and (best is None or solution.objective<best.objective):best=solution;best_support=support
                    except Exception as exc:
                        all_optimal=False;record['candidate_results'].append({'support':support,'error':repr(exc),'wall_seconds':perf_counter()-st})
                        if isinstance(exc,ExperimentBudgetExpired):break
                record['all_candidates_certified_optimal']=all_optimal and len(record['candidate_results'])==len(candidates)
                record['wall_seconds']=perf_counter()-start
                if record['all_candidates_certified_optimal'] and best is not None:
                    # The bound is the minimum of certified LP optima over the
                    # exhaustive supplied finite domain, not a HiGHS MIP bound.
                    diag=self.diagnostic(0,'EXPERIMENT: exact finite candidate enumeration; every fixed-support LP optimal',best.objective,record['wall_seconds'])
                    answer=core.ExtensionSolution(best_support,best.r_values,best.x_values,best.intercepts,best.objective,diag)
                else:
                    answer=self.blank(prepared,self.diagnostic(1,'EXPERIMENT: incomplete finite candidate enumeration',None if best is None else best.objective,record['wall_seconds']))
            self.enumerations.append(record)
        self.attempts.append(answer)
        return answer
    def fit(self,scenarios,*,validation_scenarios,initial_supports,candidate_supports,max_atoms=None):
        self.initial_count=len(initial_supports)
        try:
            result=core.fit_laminar_l1_sensitivity(scenarios,validation_scenarios=validation_scenarios,
                initial_supports=initial_supports,candidate_supports=candidate_supports,max_atoms=max_atoms,
                r_upper_bound=2.0,x_upper_bound=2.0,time_limit=self.solve_seconds,mip_rel_gap=0.0)
            return result
        except ExperimentBudgetExpired:
            if not self.path:raise
            selected_index=core._choose_path_point(self.path);s=self.path[selected_index]
            return core.LaminarL1Result(tuple(scenarios[0]['P_terminal'].columns),s.support_indices,s.support_labels,
                s.r_values,s.x_values,s.intercepts,s.r_matrix,s.x_matrix,s.train_mae,s.validation_mae,
                selected_index,tuple(self.path),tuple(self.attempts),'experiment_total_budget',self.r_bound,self.x_bound,
                self.bound_restart_count)
    def records(self):
        return {'adapter':'bounded_exact_lp_enumeration' if self.enumerate_pool else 'bounded_native_milp',
            'certificate_scope':('all certified fixed-support LP optima for one supplied finite-domain extension; forward path remains greedy'
                if self.enumerate_pool else 'native MILP optimality requires the strict per-solve status and gap certificate; domain is the supplied pool or all admissible subsets when no pool is supplied; forward path remains greedy'),
            'solver_calls':self.solves,'enumerations':self.enumerations,'model_wall_seconds':perf_counter()-self.started,
            'solver_seconds':sum(d['runtime_seconds'] for d in self.solves)}


def serialized(clades):return sorted([sorted(map(int,c)) for c in clades],key=lambda x:(len(x),x))

def family_scores(predicted,truth):
    tp=len(predicted&truth);p=tp/len(predicted) if predicted else float(not truth)
    r=tp/len(truth) if truth else 1.0
    return {'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.0,'exact':float(predicted==truth),'tp':tp,'fp':len(predicted-truth),'fn':len(truth-predicted)}

def rooted_scores(predicted,truth,terminals):
    universe=frozenset(terminals)
    def top(c):return {s for s in c if not any(s<t for t in c)}
    def bottom(c):return {s for s in c if not any(t<s for t in c)}
    def partition(c):
        maximal=top(c);covered=frozenset().union(*maximal) if maximal else frozenset()
        return maximal|{frozenset({i}) for i in universe-covered}
    def parent(c,t):return min((s for s in c if t in s),key=len,default=universe)
    output={}
    for prefix,p,t in [('clade',predicted,truth),('root_clade',top(predicted),top(truth)),('terminal_clade',bottom(predicted),bottom(truth)),('root_partition',partition(predicted),partition(truth))]:
        output.update({f'{prefix}_{k}':v for k,v in family_scores(p,t).items()})
    output['terminal_parent_exact']=sum(parent(predicted,t)==parent(truth,t) for t in terminals)/len(terminals)
    output['terminal_parent_count']=len(terminals)
    return output


def validation_selection(candidate, reference):
    """Select solely by validation MAE; exact ties favor the RNJ reference."""
    chosen=candidate if candidate.get('validation_mae',float('inf'))<reference['validation_mae'] else reference
    fields={'validation_selected_method':chosen['method']}
    for key,value in chosen.items():
        if not key.startswith('validation_selected_') and key.endswith(('_f1','_exact','_mae','_rmse','_relative_error')):
            fields[f'validation_selected_{key}']=value
    return fields
