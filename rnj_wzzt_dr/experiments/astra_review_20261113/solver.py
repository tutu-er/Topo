"""Gaussian native-bank recovery QCQP; scoring matrices remain frozen."""
from time import perf_counter
import numpy as np
from scipy.optimize import brentq
from scipy.stats import ncx2, norm

from ..customer_flexibility_20261113.placement import terms


def calibration(score, validation, mode):
    H = np.asarray(score['parameters'][score['map_index']]['H'])
    sigma2 = float(score['sigma2'])
    if mode == 'nominal':
        return H.copy(), np.zeros(H.shape[0]), sigma2, dict(method='frozen_target_mean', variance=sigma2)
    if mode != 'calibrated':
        raise ValueError(mode)
    P = np.c_[validation.p, validation.q]
    V = P@H.T
    R = validation.y-V
    vc = V-V.mean(0)
    slope = np.sum(vc*(R-R.mean(0)), axis=0)/(np.sum(vc**2, axis=0)+len(P)*sigma2)
    intercept = R.mean(0)-slope*V.mean(0)
    residual = R-intercept-slope*V
    variance = max(sigma2, float(np.sum(residual**2)/max(1, (len(P)-2)*H.shape[0])))
    return (1+slope[:, None])*H, intercept, variance, dict(
        method='validation_only_ridge_affine_residual', time_points=len(P),
        coefficient_count=2*H.shape[0], variance=variance,
        scale_range=[float((1+slope).min()), float((1+slope).max())],
        selection='Chosen using previously opened 32 AC days; independent new paths used for evaluation')


class Recovery:
    def __init__(self, case, score, validation, forecast, base_kw, mode, voltage_sigma):
        self.case, self.score, self.forecast, self.base_kw = case, score, forecast, base_kw
        self.sigma2 = float(score['sigma2'])
        t = score['map_index']
        self.H = np.asarray(score['parameters'][t]['H'])
        self.history = float(score['parameters'][t]['history_sse'])
        target_id = score['models'][t]['topology_id']
        self.competitors = [p for p in score['native_bank'] if p['topology_id'] != target_id]
        K, offset, variance, self.calibration = calibration(score, validation, mode)
        self.K, self.offset, self.variance = K, offset, variance
        self.ratio = variance/self.sigma2
        self.cells = score['cells']+forecast.shape[0]*self.H.shape[0]
        self.B = 2*8*voltage_sigma**2*self.cells
        self.alpha0 = .02
        self.tie = max(1e-8, self.cells*score['reference_mse']*score['validation_tie']/(2*self.sigma2))
        L = np.array([(p['history_sse']-self.history)/(2*self.sigma2) for p in self.competitors])
        weights = np.maximum(1e-6, np.exp(-.5*np.clip(L, -40, 80)))
        self.alpha = .08*weights/weights.sum() if len(weights) else weights
        self.z = norm.isf(self.alpha)
        self.muQ, self.mua, self.mu0, self.jQ, self.ja, self.j0, self.jupper = [], [], [], [], [], [], []
        for p, history in zip(self.competitors, L):
            other = np.asarray(p['H'])
            D = self.H-other
            d = forecast@D.T
            b = forecast@(K-self.H).T+offset
            Q, a = terms(case, self.H, other, K, forecast, self.sigma2, base_kw)
            effect = (D[:, case.node]+D[:, case.N+case.node]*case.qratio)/base_kw
            positive, negative = np.maximum(effect, 0), np.minimum(effect, 0)
            lo = d+np.repeat(case.lower,case.repeat,axis=0)@positive.T+np.repeat(case.upper,case.repeat,axis=0)@negative.T
            hi = d+np.repeat(case.upper,case.repeat,axis=0)@positive.T+np.repeat(case.lower,case.repeat,axis=0)@negative.T
            self.jupper.append(float(np.maximum(lo*lo,hi*hi).sum()/(2*self.sigma2)))
            a += case.repeat*(offset@effect)[None]/(2*self.sigma2)
            self.muQ.append(Q); self.mua.append(a)
            self.mu0.append(history+np.sum(d*d)/(2*self.sigma2)+np.sum(b*d)/self.sigma2)
            Qj, aj = terms(case, self.H, other, self.H, forecast, self.sigma2, base_kw)
            self.jQ.append(Qj); self.ja.append(aj); self.j0.append(np.sum(d*d)/(2*self.sigma2))
        for name in ('muQ', 'mua', 'mu0', 'jQ', 'ja', 'j0'):
            setattr(self, name, np.asarray(getattr(self, name)))
        # Exact noncentral chi-square rejection radius, calculated once.
        m = forecast.shape[0]*self.H.shape[0]
        threshold = (self.B-self.history)/variance
        self.reject_minimum = float(ncx2.sf(threshold, m, 0)) if threshold > 0 else 1.
        self.radius = None
        if self.reject_minimum <= self.alpha0:
            hi = 1.
            while ncx2.sf(threshold, m, hi) < self.alpha0:
                hi *= 2
            self.radius = float(brentq(lambda x:ncx2.sf(threshold, m, x)-self.alpha0, 0, hi)*variance)
        E = K-self.H
        self.bias0 = forecast@E.T+offset
        self.bias_effect = (E[:, case.node]+E[:, case.N+case.node]*case.qratio)/base_kw
        self.biasQ = case.repeat*self.bias_effect.T@self.bias_effect/self.sigma2
        self.biasa = self.bias0.reshape(case.H, case.repeat, case.N).sum(1)@self.bias_effect/self.sigma2
        self.biasc = float(np.sum(self.bias0**2)/self.sigma2)

    def direct(self, u):
        P = self.forecast+self.case.shift_pq_pu(u, self.base_kw)
        bias = P@(self.K-self.H).T+self.offset
        mu, J = [], []
        for p in self.competitors:
            d = P@(self.H-np.asarray(p['H'])).T
            J.append(np.sum(d*d)/(2*self.sigma2))
            mu.append((p['history_sse']-self.history)/(2*self.sigma2)+J[-1]+np.sum(bias*d)/self.sigma2)
        mu, J = np.array(mu), np.array(J)
        sd = np.sqrt(np.maximum(0, 2*self.ratio*J))
        failure = norm.cdf(np.divide(self.tie-mu, sd, out=np.full_like(mu, -np.inf), where=sd>0))
        failure[(sd == 0) & (mu <= self.tie)] = 1.
        m = P.shape[0]*self.H.shape[0]
        reject = float(ncx2.sf((self.B-self.history)/self.variance, m, np.sum(bias*bias)/self.variance))
        margins = mu-self.z*sd-self.tie
        feasible = (self.radius is not None and np.min(margins, initial=np.inf) >= -1e-7
                    and np.all(failure <= self.alpha+1e-9) and reject <= self.alpha0+1e-9
                    and np.sum(bias*bias) <= self.radius+1e-7*self.sigma2)
        return dict(feasible=bool(feasible), mu=mu, J=J, minimum_pair_margin=float(np.min(margins, initial=np.inf)) if len(margins) else None,
            pair_failure_sum=float(failure.sum()), pair_failure_max=float(failure.max(initial=0.)), reject_failure=reject,
            model_recovery_lower=max(0., 1-reject-float(failure.sum())),
            bias_energy=float(np.sum(bias*bias)))

    def verify_expansion(self, u):
        direct = self.direct(u)
        if len(self.competitors):
            for name, q, a, c in [('mu',self.muQ,self.mua,self.mu0),('J',self.jQ,self.ja,self.j0)]:
                v = np.einsum('hi,kij,hj->k',u,q,u)+2*np.einsum('khi,hi->k',a,u)+c
                if not np.allclose(v, direct[name], atol=2e-8, rtol=2e-8):
                    raise ArithmeticError('Direct and expanded '+name+' disagree')
        energy = (np.einsum('hi,ij,hj',u,self.biasQ,u)+2*np.sum(self.biasa*u)+self.biasc)*self.sigma2
        if not np.isclose(energy, direct['bias_energy'], atol=1e-11, rtol=1e-8):
            raise ArithmeticError('Bias quadratic mismatch')
        return direct


def plan(case, score, validation, forecast, base_kw, *, mode, seconds, env, training=None, voltage_sigma=.0005):
    import gurobipy as gp
    began = perf_counter()
    model = Recovery(case, score, validation, np.asarray(forecast), base_kw, mode, voltage_sigma)
    zero = np.zeros(case.shape)
    baseline = model.verify_expansion(zero)
    common = dict(mode=mode, calibration=model.calibration, rejection_B=model.B,
        rejection_risk=.02, competitor_risk=.08, competitors=len(model.competitors),
        constrained_bank='all frozen native points of different target topology',
        baseline=baseline, confidence_scope='fixed point forecast, nominal response and declared Gaussian residual law')
    if baseline['feasible']:
        return dict(common, feasible=True, status='zero_sufficient', source='zero', cost_yuan=0.,
            lower_bound_yuan=0., relative_gap=0., global_optimality_verified=True, seconds=perf_counter()-began), zero
    if model.radius is None:
        return dict(common, feasible=False, status='reject_reference_infeasible_in_model',
            infeasibility_proved=True, reject_minimum=model.reject_minimum,
            cost_yuan=None, lower_bound_yuan=None, seconds=perf_counter()-began), None
    seed, warm, initial = None, None, dict(source='resource_LP_direction_and_line_search', feasible=False, cost_yuan=None)
    tick = perf_counter()
    if len(model.competitors):
        margins = baseline['mu']-model.z*np.sqrt(2*model.ratio*baseline['J'])
        index = int(np.argmin(margins))
        J = max(model.j0[index], 1e-15)
        gradient = 2*model.mua[index]-model.z[index]*np.sqrt(2*model.ratio)*model.ja[index]/np.sqrt(J)
        try:
            direction, _ = case.linear_optimum(-gradient)
            candidates = []
            for fraction in np.linspace(0, 1, 41):
                point = fraction*direction
                info = model.verify_expansion(point)
                candidates.append((info['minimum_pair_margin'], point))
                if info['feasible'] and case.audit(point)['passed']:
                    seed = point.copy(); break
            warm = seed if seed is not None else max(candidates, key=lambda x:x[0])[1]
        except Exception as exc:
            initial['error'] = repr(exc)
    initial.update(seconds=perf_counter()-tick, feasible=seed is not None, cost_yuan=None if seed is None else case.fee(seed))
    with gp.Model('astra_direct_native_recovery', env=env) as m:
        for key,value in dict(OutputFlag=0,Threads=1,NonConvex=2,FeasibilityTol=1e-9,NumericFocus=1,MIPGap=1e-4).items():
            m.setParam(key,value)
        u, fee = case.attach(m)
        for k in range(len(model.competitors)):
            J = sum(u[h]@model.jQ[k]@u[h] for h in range(case.H))+2*(model.ja[k]*u).sum()+model.j0[k]
            mu = sum(u[h]@model.muQ[k]@u[h] for h in range(case.H))+2*(model.mua[k]*u).sum()+model.mu0[k]
            s = m.addVar(lb=0., ub=np.sqrt(model.jupper[k])*(1+1e-9), name='distance_'+str(k))
            m.addConstr(J <= s*s)
            m.addConstr(mu >= model.z[k]*np.sqrt(2*model.ratio)*s+model.tie+1e-8)
            if warm is not None:
                s.Start = float(np.sqrt(model.direct(warm)['J'][k]))
        bias = sum(u[h]@model.biasQ@u[h] for h in range(case.H))+2*(model.biasa*u).sum()+model.biasc
        m.addConstr(bias <= model.radius/model.sigma2)
        m.setObjective(fee)
        if warm is not None:
            u.Start = warm
        m.Params.TimeLimit = max(.001, seconds-(perf_counter()-began))
        m.optimize()
        try:
            bound = float(m.ObjBound)
            if not np.isfinite(bound): bound=None
        except (AttributeError,gp.GurobiError): bound=None
        point, source = seed, 'verified_initial_retained'
        if m.SolCount:
            candidate = np.asarray(u.X)
            info = model.verify_expansion(candidate)
            if info['feasible'] and case.audit(candidate)['passed'] and (point is None or case.fee(candidate)<case.fee(point)-1e-9):
                point, source = candidate, 'native_found' if seed is None else 'native_improved_initial'
        if point is None:
            return dict(common, feasible=False, status='native_infeasible' if m.Status==3 else 'unresolved',
                native_status=int(m.Status), initial=initial, lower_bound_yuan=bound, cost_yuan=None,
                infeasibility_proved=m.Status==3, seconds=perf_counter()-began), None
        cost = case.fee(point)
        conflict = bound is not None and bound>cost+1e-7*max(1.,cost)
        lower = 0. if bound is None or conflict else max(0.,min(bound,cost))
        gap = (cost-lower)/max(cost,1e-12)
        return dict(common, feasible=True, status='feasible', source=source, native_status=int(m.Status),
            initial=initial, cost_yuan=cost, lower_bound_yuan=lower, relative_gap=gap,
            lower_bound_withdrawn=conflict, global_optimality_verified=gap<=1e-4,
            improvement_from_verified_initial_yuan=None if seed is None else case.fee(seed)-cost,
            final=model.verify_expansion(point), resource_audit=case.audit(point), seconds=perf_counter()-began), point
