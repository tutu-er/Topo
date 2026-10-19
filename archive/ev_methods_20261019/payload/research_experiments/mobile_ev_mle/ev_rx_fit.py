"""Local AC fits in the original R/X box intersected with a cable X/R cone.

The residuals, physical AC Jacobian, fixed input P/Q, and one physical source
interface are those of ev_fixed_train_fit.  Only the training line domain is
changed.  Parameterization derivatives are used by the optimizer; information
diagnostics always use the same physical R/X/source coordinates and original
box scales as the old fitter.  No global optimum or confidence set is claimed.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from . import likelihood as lk
from .ev_fixed_train_fit import (
    OUTPUT_CHANNELS, _FixedInputProblem, _compact_bias, fit_fixed_inputs,
)


DEFAULT_RATIO_BOUNDS = (.083/.642, .080/.208)


class _RxMap:
    """Exact closed-box/cone map, including its collapsed corner sections.

    At each max/min kink use the derivative from increasing R (right side).
    The inverse is for initialization: out-of-domain points are projected by
    clipping R and then X, not by solving a Euclidean projection problem.
    """
    def __init__(self, m, r_bounds, x_bounds, ratio_bounds=DEFAULT_RATIO_BOUNDS):
        self.m = int(m)
        self.rlo, self.rhi = lk._bounds(r_bounds, (m,), "r_bounds")
        self.xlo, self.xhi = lk._bounds(x_bounds, (m,), "x_bounds")
        try:
            low, high = map(float, ratio_bounds)
        except (TypeError, ValueError) as error:
            raise ValueError("ratio_bounds requires two finite positive values") from error
        if not np.isfinite([low, high]).all() or not 0 < low < high:
            raise ValueError("ratio_bounds requires 0 < lower < upper")
        if np.any(self.rlo < 0) or np.any(self.xlo < 0):
            raise ValueError("R/X bounds must be nonnegative")
        self.rho_low, self.rho_high = low, high
        self.lower = np.maximum(self.rlo, self.xlo/high)
        self.upper = np.minimum(self.rhi, self.xhi/low)
        if np.any(self.lower > self.upper):
            raise ValueError("original R/X box and ratio cone have empty intersection")
        self.low_switch = self.xlo/low
        self.high_switch = self.xhi/high

    def section(self, r):
        r = lk._array(r, (self.m,), "R")
        low = np.maximum(self.xlo, self.rho_low*r)
        high = np.minimum(self.xhi, self.rho_high*r)
        # At an exact common corner a roundoff difference may be one ulp.
        width = np.maximum(high-low, 0.)
        low_prime = np.where(r >= self.low_switch, self.rho_low, 0.)
        high_prime = np.where(r < self.high_switch, self.rho_high, 0.)
        return low, high, width, low_prime, high_prime

    def forward(self, r, alpha):
        r = lk._array(r, (self.m,), "R")
        alpha = lk._array(alpha, (self.m,), "alpha")
        low, _, width, low_prime, high_prime = self.section(r)
        x = low+alpha*width
        derivative_r = low_prime+alpha*(high_prime-low_prime)
        return r.copy(), x, derivative_r, width

    def inverse(self, r, x):
        r = np.clip(lk._array(r, (self.m,), "initial.R"), self.lower, self.upper)
        low, high, width, _, _ = self.section(r)
        x = np.clip(lk._array(x, (self.m,), "initial.X"), low, high)
        alpha = np.divide(x-low, width, out=np.full(self.m, .5), where=width > 0)
        return r, np.clip(alpha, 0., 1.)

    def contains(self, r, x, *, atol=1e-12):
        r, x = np.asarray(r), np.asarray(x)
        return bool(np.isfinite(r).all() and np.isfinite(x).all()
            and np.all(r >= self.rlo-atol) and np.all(r <= self.rhi+atol)
            and np.all(x >= self.xlo-atol) and np.all(x <= self.xhi+atol)
            and np.all(x >= self.rho_low*r-atol) and np.all(x <= self.rho_high*r+atol))

    def constraint_record(self, r, x, *, tolerance=1e-9):
        r, x = np.asarray(r), np.asarray(x)
        slacks = dict(r_lower=r-self.rlo, r_upper=self.rhi-r,
            x_lower=x-self.xlo, x_upper=self.xhi-x,
            ratio_lower=x-self.rho_low*r, ratio_upper=self.rho_high*r-x)
        return dict(feasible=self.contains(r, x), X_over_R=(x/r).tolist(),
            slacks_pu={k:v.tolist() for k,v in slacks.items()},
            active={k:(v <= tolerance).tolist() for k,v in slacks.items()},
            activity_tolerance_pu=tolerance,
            active_ratio_bound_count=int(sum(np.count_nonzero(slacks[k] <= tolerance)
                                            for k in ("ratio_lower", "ratio_upper"))))


class _RxInputProblem:
    """Optimizer-coordinate wrapper around the unchanged physical problem."""
    def __init__(self, candidate, observations, model, *, initial=None,
                 r_bounds=(1e-6,.1), x_bounds=(1e-6,.1),
                 ratio_bounds=DEFAULT_RATIO_BOUNDS, source_p_bounds=(-.5,.5),
                 source_q_bounds=(-.5,.5), voltage_bounds=(.85,1.15), call_counter=None):
        self.base = _FixedInputProblem(candidate, observations, model, initial=initial,
            r_bounds=r_bounds, x_bounds=x_bounds, source_p_bounds=source_p_bounds,
            source_q_bounds=source_q_bounds, voltage_bounds=voltage_bounds, call_counter=call_counter)
        self.m, self.t, self.n = self.base.m, self.base.t, self.base.n
        self.mapping = _RxMap(self.m, r_bounds, x_bounds, ratio_bounds)
        self.shared = 2*self.m
        self.full_lower = np.r_[self.mapping.lower, np.zeros(self.m), self.base.lower[self.shared:]]
        self.full_upper = np.r_[self.mapping.upper, np.ones(self.m), self.base.upper[self.shared:]]
        # A zero-width feasible R interval is an exactly fixed corner.  Remove
        # its redundant coordinates instead of giving least_squares equal bounds.
        r_free = self.mapping.upper > self.mapping.lower
        _,_,corner_width,_,_ = self.mapping.section(self.mapping.lower)
        alpha_free = r_free | (corner_width > 0)
        self.free = np.r_[r_free, alpha_free, np.ones(2*self.t,dtype=bool)]
        self.anchor = self.full_lower.copy()
        self.lower, self.upper = self.full_lower[self.free], self.full_upper[self.free]
        if initial is None:
            r = .5*(self.mapping.lower+self.mapping.upper)
            alpha = np.full(self.m,.5)
        else:
            r, alpha = self.mapping.inverse(self.base.start[:self.m], self.base.start[self.m:self.shared])
        full_start = np.r_[r, alpha, self.base.start[self.shared:]]
        self.margin = np.minimum(1e-10*np.maximum(self.upper-self.lower,1), .01*(self.upper-self.lower))
        self.start = np.clip(full_start[self.free], self.lower+self.margin, self.upper-self.margin)
        full_scale = np.r_[np.maximum(.1*(self.mapping.upper-self.mapping.lower),1e-4),
                           np.full(self.m,.1), np.full(2*self.t,.01)]
        self.scale = full_scale[self.free]
        self.width = len(self.start)

    def expand(self, vector):
        vector = np.asarray(vector,float)
        if vector.shape != (self.width,):
            raise ValueError("packed optimizer vector has wrong width")
        answer = self.anchor.copy(); answer[self.free] = vector
        return answer

    def physical_vector(self, vector):
        full = self.expand(vector)
        r, x, _, _ = self.mapping.forward(full[:self.m],full[self.m:self.shared])
        return np.r_[r,x,full[self.shared:]]

    def unpack(self, vector):
        return self.base.unpack(self.physical_vector(vector))

    def evaluate(self, vector):
        return self.base.evaluate(self.physical_vector(vector))

    def physical_jacobian(self, vector):
        return self.base.jacobian(self.physical_vector(vector))

    def jacobian(self, vector):
        full = self.expand(vector)
        _, _, derivative_r, derivative_alpha = self.mapping.forward(full[:self.m],full[self.m:self.shared])
        physical = self.physical_jacobian(vector)
        jr, jx = physical[:,:self.m], physical[:,self.m:self.shared]
        mapped = np.c_[jr+jx*derivative_r[None,:], jx*derivative_alpha[None,:], physical[:,self.shared:]]
        return mapped[:,self.free]

    def state_record(self, vector):
        r,x,s = self.unpack(vector); full=self.expand(vector)
        return dict(r=r.tolist(),x=x.tolist(),source_p=s[:,0].tolist(),source_q=s[:,1].tolist(),
                    optimizer_alpha=full[self.m:self.shared].tolist())


def _physical_information(problem, vector):
    """Physical-coordinate spectra with the old original-box scale, not alpha."""
    physical = problem.physical_jacobian(vector)
    scale = problem.base.scale
    singular = np.linalg.svd(physical*scale[None,:],compute_uv=False)
    rank_tol = 1e-7*singular[0] if len(singular) else 0.
    rank = int(np.count_nonzero(singular > rank_tol))
    nuisance = physical[:,problem.shared:]
    left, source_singular, _ = np.linalg.svd(nuisance,full_matrices=False)
    source_rank = int(np.count_nonzero(source_singular > (1e-10*source_singular[0] if len(source_singular) else 0)))
    shared = physical[:,:problem.shared]*scale[:problem.shared][None,:]
    projected = shared-left[:,:source_rank]@(left[:,:source_rank].T@shared)
    _, values, right = np.linalg.svd(projected,full_matrices=False)
    shared_rank = int(np.count_nonzero(values > (1e-7*values[0] if len(values) else 0)))
    source_by_time = np.stack([nuisance[t*(problem.n+2):(t+1)*(problem.n+2),2*t:2*t+2]
                               for t in range(problem.t)])
    return dict(information_coordinate_system="physical_R_then_X_then_per_time_source_PQ",
        information_scale_policy="unchanged_ev_fixed_train_fit_original_box_scale",
        common_physical_scale=scale.tolist(), physical_jacobian_column_count=physical.shape[1],
        local_jacobian_rank=rank,local_jacobian_rank_tolerance=float(rank_tol),
        local_scaled_singular_values=singular.tolist(),local_shared_rank=shared_rank,
        local_projected_shared_singular_values=values.tolist(),local_source_rank=source_rank,
        local_identifiability_full_column_rank=bool(rank==physical.shape[1]),
        rank_is_global_identifiability_certificate=False,
        physical_shared_whitened_jacobian=physical[:,:problem.shared].tolist(),
        source_whitened_jacobian_by_time=source_by_time.tolist(),
        profiled_scaled_physical_shared_jacobian=projected.tolist(),
        weakest_shared_scaled_direction=right[-1].tolist() if len(right) else [],
        weakest_shared_physical_direction=(scale[:problem.shared]*right[-1]).tolist() if len(right) else [],
        source_projection_scope="unconstrained_local_source_span; activity recorded separately; not a confidence set",
        ratio_constraint_does_not_add_observed_information=True)


def fit_rx_inputs(candidate, observations, model, *, initial=None, fixed_r=None, fixed_x=None,
                  r_bounds=(1e-6,.1), x_bounds=(1e-6,.1),
                  source_p_bounds=(-.5,.5), source_q_bounds=(-.5,.5),
                  voltage_bounds=(.85,1.15),multistart=2,max_nfev=250,tol=1e-10,
                  seed=0,call_counter=None,ratio_bounds=DEFAULT_RATIO_BOUNDS):
    """Return one retained local FitResult, with the original box AND X/R cone.

    Fixed-line calls validate the domain, then delegate unchanged source fitting
    to fit_fixed_inputs.  The training wrapper uses exact chain-rule derivatives
    and keeps failed local restarts in diagnostics.  Initial guesses can be a
    FitResult, attribute object, or mapping, as in the old interface.
    """
    began=perf_counter()
    if (fixed_r is None)!=(fixed_x is None):
        raise ValueError("fixed_r and fixed_x must be supplied together")
    mapping=_RxMap(candidate.m,r_bounds,x_bounds,ratio_bounds)
    if fixed_r is not None:
        r=lk._array(fixed_r,(candidate.m,),"fixed_r");x=lk._array(fixed_x,(candidate.m,),"fixed_x")
        if not mapping.contains(r,x): raise ValueError("fixed R/X must lie in original box and ratio cone")
        fit=fit_fixed_inputs(candidate,observations,model,initial=initial,fixed_r=r,fixed_x=x,
            r_bounds=r_bounds,x_bounds=x_bounds,source_p_bounds=source_p_bounds,source_q_bounds=source_q_bounds,
            voltage_bounds=voltage_bounds,multistart=multistart,max_nfev=max_nfev,tol=tol,seed=seed,call_counter=call_counter)
        fit.diagnostics.update(ratio_bounds=list(ratio_bounds),ratio_constraint_enforced=True,
            line_domain="original_box_intersect_X_over_R_cone",parameterization="fixed_physical_R_X",
            information_coordinate_system="physical_source_PQ_only; lines fixed",ratio_constraints=mapping.constraint_record(r,x))
        return fit
    for value,name in ((multistart,"multistart"),(max_nfev,"max_nfev")):
        if int(value)!=value or value<1: raise ValueError(name+" must be a positive integer")
    if not np.isfinite(tol) or tol<=np.finfo(float).eps:
        raise ValueError("tol must be finite and greater than machine epsilon")
    problem=_RxInputProblem(candidate,observations,model,initial=initial,
        r_bounds=r_bounds,x_bounds=x_bounds,ratio_bounds=ratio_bounds,
        source_p_bounds=source_p_bounds,source_q_bounds=source_q_bounds,
        voltage_bounds=voltage_bounds,call_counter=call_counter)
    if call_counter is not None: call_counter["fit_calls"]=call_counter.get("fit_calls",0)+1
    rng=np.random.default_rng(seed);starts=[problem.start]
    for _ in range(1,int(multistart)):
        full=problem.expand(problem.start)
        full[:problem.m]*=np.exp(rng.normal(0,.5,problem.m))
        full[problem.m:problem.shared]+=rng.normal(0,.25,problem.m)
        full[problem.shared:]+=rng.normal(0,.01,2*problem.t)
        starts.append(np.clip(full[problem.free],problem.lower+problem.margin,problem.upper-problem.margin))
    restarts=[];retained=[];initial_loss=None;solver_calls=0
    for index,start in enumerate(starts):
        record=dict(index=index,initial_state=problem.state_record(start),success=False)
        try:
            residual=problem.evaluate(start);record["initial_loss"]=float(residual@residual)
            if initial_loss is None: initial_loss=record["initial_loss"]
            if call_counter is not None: call_counter["solver_calls"]=call_counter.get("solver_calls",0)+1
            solver_calls+=1
            result=least_squares(problem.evaluate,start,jac=problem.jacobian,
                bounds=(problem.lower,problem.upper),x_scale=problem.scale,method="trf",tr_solver="exact",
                ftol=tol,xtol=tol,gtol=tol,max_nfev=int(max_nfev))
            residual=problem.evaluate(result.x);loss=float(residual@residual)
            voltage=abs(problem.base.cache["physical"]["voltage"])
            voltage_ok=bool(np.min(voltage)>=voltage_bounds[0] and np.max(voltage)<=voltage_bounds[1])
            r,x,_=problem.unpack(result.x);domain_ok=problem.mapping.contains(r,x)
            valid=bool(result.success and voltage_ok and domain_ok and np.isfinite(loss))
            record.update(loss=loss,success=valid,solver_success=bool(result.success),status=int(result.status),
                message=str(result.message),nfev=int(result.nfev),optimality=float(result.optimality),
                physical_voltage_feasible=voltage_ok,ratio_domain_feasible=domain_ok,
                active_bound_count=int(np.count_nonzero(result.active_mask)),attained_state=problem.state_record(result.x))
            retained.append((valid,loss,result,record))
        except (RuntimeError,FloatingPointError,np.linalg.LinAlgError,ValueError) as error:
            record.update(loss=None,solver_success=False,status=-99,message=str(error),nfev=None)
        restarts.append(record)
    base=problem.base
    diagnostics=dict(objective="conditional_known_diagonal_Gaussian_output_given_measured_PQ",
        measurement_channels=list(OUTPUT_CHANNELS),source_node=1,source_spatial_location_shared=True,
        source_joint_voltage_and_master_estimation=True,source_anchored=True,root_voltage_known=True,
        line_parameters_fixed=False,fixed_lines=False,P_Q_inputs_fixed=True,P_Q_measurement_error_propagated=False,
        P_Q_likelihood_included=False,input_EV_already_in_total_PQ=True,source_sign="positive_consumption",
        impedance_units="pu",known_x_over_r=False,bias_policy="known_calibrated",
        line_domain="original_box_intersect_X_over_R_cone",ratio_constraint_enforced=True,
        ratio_bounds=[mapping.rho_low,mapping.rho_high],r_bounds_original=[mapping.rlo.tolist(),mapping.rhi.tolist()],
        x_bounds_original=[mapping.xlo.tolist(),mapping.xhi.tolist()],effective_r_bounds=[mapping.lower.tolist(),mapping.upper.tolist()],
        parameterization="X=max(xL,rhoL*R)+alpha*(min(xU,rhoU*R)-max(xL,rhoL*R))",
        kink_derivative="right derivative in R at max/min switches",shared_parameter_count=problem.shared,
        source_parameter_count=2*problem.t,joint_parameter_count=base.width,optimizer_parameter_count=problem.width,
        optimizer_shared_parameter_count=int(np.count_nonzero(problem.free[:problem.shared])),
        residual_count=problem.t*(problem.n+2),output_normalizer=base.normalizer,initial_loss=initial_loss,
        restarts=restarts,solver="scipy.least_squares_trf_exact_rx_chainrule",solver_calls=solver_calls,
        total_nfev=sum(r.get("nfev") or 0 for r in restarts),ac_evaluations=base.ac_calls,
        restart_rule="R log-Normal(sd=.5), alpha additive Normal(sd=.25), source Normal(sd=.01pu); clipped",
        default_line_initialization="effective R interval midpoint and alpha=.5; explicit physical initials are clipped into the exact domain",
        global_optimality_certified=False,global_loss_lower_bound=None,loglikelihood_is_topology_probability=False,
        ac_solution_branch="root_initialized_fixed_point; other equilibria not enumerated")
    biases=[_compact_bias(base.bias[k]) for k in ("p","q","v")]
    if not retained:
        r,x,_=problem.unpack(problem.start)
        diagnostics.update(failure_kind="computation",seconds=perf_counter()-began)
        return lk.FitResult(candidate.name,r.copy(),x.copy(),base.p.copy(),base.q.copy(),
            np.full(problem.t,np.nan),np.full(problem.t,np.nan),*biases,np.inf,-np.inf,False,-99,
            "all numerical restarts failed",0,np.inf,{},diagnostics)
    valid,loss,result,record=min(retained,key=lambda item:(not item[0],item[1]))
    problem.evaluate(result.x);diagnostics.update(_physical_information(problem,result.x))
    phy=base.cache["physical"];r,x,source=problem.unpack(result.x)
    finite_losses=[row["loss"] for row in restarts if row.get("loss") is not None]
    diagnostics.update(ac_evaluations=base.ac_calls,physical_voltage_feasible=record["physical_voltage_feasible"],
        ratio_constraints=mapping.constraint_record(r,x),optimizer_alpha=problem.expand(result.x)[problem.m:problem.shared].tolist(),
        min_voltage=float(np.min(abs(phy["voltage"]))),max_voltage=float(np.max(abs(phy["voltage"]))),
        ac_equation_residual=float(phy["ac_equation_residual"]),power_balance_residual=float(phy["power_balance_residual"]),
        active_bound_count=record["active_bound_count"],restart_loss_spread=float(max(finite_losses)-min(finite_losses)),
        failure_kind=None if valid else ("physical_domain" if not record["physical_voltage_feasible"] or not record["ratio_domain_feasible"] else "computation"),
        seconds=perf_counter()-began)
    predicted=dict(phy,p=base.p+base.bias["p"],q=base.q+base.bias["q"],
                   v=phy["v"]+base.bias["v"],physical_v=phy["v"])
    return lk.FitResult(candidate.name,r.copy(),x.copy(),base.p.copy(),base.q.copy(),
        source[:,0].copy(),source[:,1].copy(),*biases,loss,-.5*(loss+base.normalizer),valid,
        int(result.status),str(result.message),int(result.nfev),float(result.optimality),predicted,diagnostics)


__all__=["fit_rx_inputs","DEFAULT_RATIO_BOUNDS"]
