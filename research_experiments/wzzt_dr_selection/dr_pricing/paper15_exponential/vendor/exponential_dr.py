"""指数概率条件下的DR规划：与 relative_gate_paper/main.tex 对应的单文件实现。

一、输入和概率含义
------------------
模型0是当前模型，模型1是备选模型；H0、H1由历史拟合后固定。本文件不拟合
拓扑，也不重估线路参数。用列向量表示整个规划时段的数据：

    P = p_hat + B @ u + xi,   xi ~ N(0, SigmaP)
    V = H1 @ P + epsilon,    epsilon ~ N(0, SigmaV), epsilon 与 xi 独立。

u[n]为名义调节量；P[d]可包含所有节点、时段的P/Q；V[m]为电压响应。
H0、H1形状(m,d)，B形状(d,n)，SigmaP形状(d,d)，SigmaV形状(m,m)。
SigmaP允许奇异、包含跨节点/时段相关；SigmaV必须正定。两协方差固定，
不随u变化。价格无关的固定加性背景/交付误差可合并计入SigmaP；乘性响应、
饱和后的误差一般随计划变化，不直接适用这一固定协方差版本。尤其不能
把付费计划的响应方差当作无DR计划的响应方差。SigmaV仅计给定P后剩余
的电压误差，避免把同一功率预测误差算两次。所有量须使用一致单位。

平方损失是标准化后的总和，不是MAE、均方值或未经缩放的物理SSE：

    S_k = (V - Hk @ P).T @ solve(SigmaV, V - Hk @ P)
    L = S_0 - kappa * S_1,           kappa >= 1
    目标：Pr_1(L > delta) >= beta,    0 < beta < 1。

Pr_1是在模型1为电压参考、上述高斯分布成立时的执行概率，不是拓扑真实概率。
delta可直接给定；或仅给history=(S0h,S1h,delta_all)，程序计算
delta=delta_all-(S0h-kappa*S1h)，使历史只计一次。所有历史损失亦须为同一
标准化总和。平均损失应先乘观测数；采用J=S/2时须同步换算容差。

二、由LaTeX公式得到的可求解约束
--------------------------------
固定 lambda=1/(4*kappa)，对失败事件使用

    Pr_1(L <= delta) <= exp(lambda*delta) * E_1[exp(-lambda*L)]。

令 DeltaH=H1-H0，d(u)=DeltaH@(p_hat+B@u)，SigmaD=DeltaH@SigmaP@DeltaH.T。
先对epsilon、再对xi作高斯积分，得到

    G = SigmaV + SigmaD / (2*(kappa+1))
    W = solve(G, I) / (4*(kappa+1))
    h = delta/(4*kappa) - log(1-beta)
        - 0.5*(logdet(G)-logdet(SigmaV))
        - (m/2)*log((kappa+1)/(2*kappa))。

因此 d(u).T@W@d(u) >= h 是目标概率的充分条件。概率积分已解析完成，
优化器不调用抽样或数值积分。失败概率上界的对数为

    log_failure_bound = log(1-beta) + h - d(u).T@W@d(u)。

此上界有保守性，lambda也未声称最优。score随u是凸二次函数，但要求
score>=h的可行域通常非凸；二次费用+线性设备约束构成非凸QCQP。
build_gate内部展开Q/linear/constant，仅供现有二次求解接口使用。

三、调用方法
------------
基本计算仅需NumPy和SciPy；求解还需有许可的gurobipy。

    gate = build_gate(H0, H1, p_hat, B, SigmaP, SigmaV,
                      kappa=1.03, beta=0.90, delta=remaining_loss)
    result = solve_continuous(gate, cost, lower, upper,
                              A_eq=A_energy, b_eq=b_energy)
    if result['has_solution']:
        prediction = gate.probability_bound(result['u'])  # 规划充分界
    actual = gate.observed_loss(P_actual, V_actual)       # 实测损失，不重拟合

solve_continuous仅用于费用sum(cost*u**2)、线性可行域的连续调节模型；它不
替代已有小时合同、饱和、效率、SOC和截止时间图。已有设备模型直接复用：

    from research_experiments.wzzt_dr_selection.dr_pricing.bayes_qcqp.solver import solve
    result = solve(problem, gate, gate.h, private_output_dir, seconds=120.)

该接口中B的列必须与原device的交付u[R,T].ravel()一致，即资源优先、时段
次之；gate.value接受这一二维数组。原solver的日志字段evidence/threshold
在这里分别表示本文件的score/h；历史名称expected_log_evidence不改变计算。
必须保留本文件的输入和协方差，原solver单独保存的Q不足以恢复概率假设。
直接运行 python exponential_dr.py 即执行main示例；仅演示公式和全局界，
不是paper15效果测试。

四、评价边界
------------
observed_loss检查同一冻结模型的平方改善条件。原RNJ+WZZT的MAE、3%验证、
收缩及最终错漏须另走原接口评价；本文件不修改这些默认规则，不把概率
充分条件通过写成最终恢复。重新拟合后的模型也不再是本概率计算的固定对。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math

import numpy as np
from scipy.linalg import cho_solve, solve_triangular


def _array(value, name, ndim):
    a = np.array(value, dtype=float, copy=True)
    if a.ndim != ndim or not a.size or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected nonempty finite {ndim}-D array")
    return a


def _covariance(value, name, size, *, positive):
    a = _array(value, name, 2)
    if a.shape != (size, size):
        raise ValueError(f"{name}: expected shape {(size, size)}")
    scale = float(np.max(np.abs(a)))
    if np.max(np.abs(a-a.T)) > 1e-10 * scale:
        raise ValueError(f"{name}: covariance must be symmetric")
    a = (a+a.T)/2
    if positive:
        try:
            np.linalg.cholesky(a)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"{name}: covariance must be positive definite") from exc
    else:
        # No artificial variance/impedance floor. Reject any negative eigenvalue
        # beyond eigensolver roundoff, and retain the supplied symmetric matrix.
        if np.linalg.eigvalsh(a)[0] < -100*np.finfo(float).eps*size*scale:
            raise ValueError(f"{name}: covariance must be positive semidefinite")
    return a


@dataclass(frozen=True)
class DRGate:
    """Frozen inputs, W/h and solver coefficients; arrays are read-only copies."""
    H0: np.ndarray
    H1: np.ndarray
    p_hat: np.ndarray
    B: np.ndarray
    SigmaP: np.ndarray
    SigmaV: np.ndarray
    W: np.ndarray
    h: float
    kappa: float
    beta: float
    delta: float
    d0: np.ndarray
    D: np.ndarray
    Q: np.ndarray
    linear: np.ndarray
    constant: float
    voltage_cholesky: np.ndarray
    kind: str = "relative_squared_loss_exponential_conditional_model1"

    def _control(self, u):
        # resource-major flattening also matches the existing device solver.
        u = np.asarray(u, dtype=float).ravel()
        if u.shape != (self.B.shape[1],) or not np.isfinite(u).all():
            raise ValueError("u: wrong size or nonfinite delivery")
        return u

    def value(self, u):
        """Return score=d(u)'W d(u), compared with h in the solver."""
        d = self.d0 + self.D @ self._control(u)
        return float(d @ self.W @ d)

    def gradient(self, u):
        """Exact score gradient with respect to flattened nominal delivery."""
        d = self.d0 + self.D @ self._control(u)
        return 2*self.D.T @ (self.W @ d)

    def probability_bound(self, u):
        """Return a sufficient lower bound, not the exact success probability."""
        score = self.value(u)
        log_failure = math.log1p(-self.beta) + self.h - score
        failure = math.exp(min(0., log_failure))
        success = -math.expm1(min(0., log_failure))
        return dict(score=score, threshold=self.h, margin=score-self.h,
                    log_failure_upper=log_failure, failure_upper=failure,
                    success_lower=success, sufficient_condition_met=score >= self.h)

    def observed_loss(self, P, V):
        """Score one actual full-day vector using the frozen models and scale."""
        P, V = _array(P, "P", 1), _array(V, "V", 1)
        if P.shape != self.p_hat.shape or V.shape != self.d0.shape:
            raise ValueError("P/V: actual observation dimensions differ from model")
        r0 = solve_triangular(self.voltage_cholesky, V-self.H0@P, lower=True)
        r1 = solve_triangular(self.voltage_cholesky, V-self.H1@P, lower=True)
        S0, S1 = float(r0@r0), float(r1@r1)
        L = S0-self.kappa*S1
        return dict(S0=S0, S1=S1, L=L, delta=self.delta,
                    margin=L-self.delta, accepted=L > self.delta)


def build_gate(H0, H1, p_hat, B, SigmaP, SigmaV, *, kappa=1.03,
               beta=0.90, delta=None, history=None):
    """Precompute W/h. Exactly one of delta or history=(S0h,S1h,delta_all)."""
    H0, H1 = _array(H0, "H0", 2), _array(H1, "H1", 2)
    p_hat, B = _array(p_hat, "p_hat", 1), _array(B, "B", 2)
    m, d = H0.shape
    if H1.shape != (m, d) or p_hat.shape != (d,) or B.shape[0] != d:
        raise ValueError("Incompatible H0/H1/p_hat/B dimensions")
    if not np.isfinite([kappa, beta]).all() or kappa < 1 or not 0 < beta < 1:
        raise ValueError("Require finite kappa>=1 and 0<beta<1")
    if (delta is None) == (history is None):
        raise ValueError("Supply exactly one of delta or history")
    if history is not None:
        historical = _array(history, "history", 1)
        if historical.shape != (3,) or (historical[:2] < 0).any():
            raise ValueError("history must be (nonnegative S0h, S1h, delta_all)")
        S0h, S1h, delta_all = historical
        delta = delta_all-(S0h-kappa*S1h)
    if np.ndim(delta) != 0 or not np.isfinite(delta):
        raise ValueError("delta must be one finite standardized loss sum")
    SigmaP = _covariance(SigmaP, "SigmaP", d, positive=False)
    SigmaV = _covariance(SigmaV, "SigmaV", m, positive=True)
    DH = H1-H0
    SigmaD = DH @ SigmaP @ DH.T
    G = SigmaV + SigmaD/(2*(kappa+1))
    chol_G, chol_V = np.linalg.cholesky(G), np.linalg.cholesky(SigmaV)
    # Cholesky solves and log determinants avoid explicit matrix inversion/dets.
    W = cho_solve((chol_G, True), np.eye(m))/(4*(kappa+1))
    W = (W+W.T)/2
    h = (delta/(4*kappa) - math.log1p(-beta)
         - np.log(np.diag(chol_G)).sum() + np.log(np.diag(chol_V)).sum()
         - m/2*math.log((kappa+1)/(2*kappa)))
    d0, D = DH@p_hat, DH@B
    Q, linear, constant = D.T@W@D, 2*D.T@W@d0, float(d0@W@d0)
    Q = (Q+Q.T)/2
    arrays = (H0,H1,p_hat,B,SigmaP,SigmaV,W,d0,D,Q,linear,chol_V)
    if not all(np.isfinite(a).all() for a in arrays) or not np.isfinite([h,constant]).all():
        raise ValueError("Nonfinite derived coefficients; rescale input units")
    for a in arrays:
        a.setflags(write=False)
    return DRGate(H0,H1,p_hat,B,SigmaP,SigmaV,W,float(h),float(kappa),
                  float(beta),float(delta),d0,D,Q,linear,constant,chol_V)


def solve_continuous(gate, cost, lower, upper, *, A_ub=None, b_ub=None,
                     A_eq=None, b_eq=None, seconds=30., mip_gap=1e-3, seed=0):
    """min sum(cost*u^2), bounds/linear constraints, score>=h, via Gurobi.

    A_ub@u<=b_ub, A_eq@u=b_eq may describe energy and other linear limits.
    NonConvex=2 uses global spatial branching; a time limit remains unresolved
    unless a feasible incumbent and its reported bound meet the requested gap.
    qualified permits 1e-7 numerical feasibility tolerance; the separate
    audit.sufficient_condition_met reports score>=h without that tolerance.
    This simple continuous domain is separate from the original device graph.
    """
    n = gate.B.shape[1]
    cost, lower, upper = [_array(a, name, 1) for a,name in
                           ((cost,"cost"),(lower,"lower"),(upper,"upper"))]
    if any(a.shape != (n,) for a in (cost,lower,upper)):
        raise ValueError("cost/lower/upper must have one entry per control")
    if (cost < 0).any() or (lower > upper).any():
        raise ValueError("Require nonnegative cost and lower<=upper")
    if not np.isfinite([seconds,mip_gap]).all() or seconds<=0 or not 0<=mip_gap<1:
        raise ValueError("Invalid time limit or relative gap")
    linear_domain = []
    for A,b,equality in ((A_ub,b_ub,False),(A_eq,b_eq,True)):
        if A is None and b is None:
            continue
        if A is None or b is None:
            raise ValueError("Both A and b are required")
        A, b = _array(A,"A",2), _array(b,"b",1)
        if A.shape != (len(b),n):
            raise ValueError("Incompatible linear constraint dimensions")
        linear_domain.append((A,b,equality))

    def check(u):
        violation = max(0., float(np.max(lower-u)), float(np.max(u-upper)))
        for A,b,equality in linear_domain:
            residual = A@u-b
            violation = max(violation, float(np.max(np.abs(residual) if equality else residual)))
        return dict(domain_violation=violation, **gate.probability_bound(u))

    # With nonnegative cost a feasible zero-control plan is globally minimal.
    zero = np.zeros(n)
    audit = check(zero)
    if audit['domain_violation'] == 0 and audit['sufficient_condition_met']:
        return dict(status="CERTIFIED_COST_ZERO", has_solution=True,
                    u=zero, objective=0., lower_bound=0., verified_gap=0.,
                    qualified=True, global_gap_target_met=True, audit=audit)
    import gurobipy as gp
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model("exponential_dr", env=env) as model:
            for name,value in dict(NonConvex=2,TimeLimit=seconds,MIPGap=mip_gap,
                                   Threads=1,Seed=seed,DualReductions=0,
                                   FeasibilityTol=1e-8,OptimalityTol=1e-8).items():
                model.setParam(name,value)
            u = model.addMVar(n, lb=lower, ub=upper, name="u")
            for A,b,equality in linear_domain:
                model.addConstr(A@u == b if equality else A@u <= b)
            model.addConstr(u@gate.Q@u + gate.linear@u + gate.constant >= gate.h)
            model.setObjective(u@np.diag(cost)@u, gp.GRB.MINIMIZE)
            model.optimize()
            result = dict(status=int(model.Status),has_solution=bool(model.SolCount),
                          qualified=False,global_gap_target_met=False)
            try:
                bound = float(model.ObjBound)
                result['lower_bound'] = bound if abs(bound)<1e100 else None
            except (AttributeError,gp.GurobiError):
                result['lower_bound'] = None
            if not model.SolCount:
                return result
            x = np.asarray(u.X)
            audit = check(x)
            objective = float(np.dot(cost,x*x))
            bound = result['lower_bound']
            gap = None if bound is None else max(0.,objective-bound)/max(abs(objective),1e-12)
            # Record numerical feasibility separately from the exact inequality.
            qualified = audit['domain_violation'] <= 1e-7 and audit['margin'] >= -1e-7
            result.update(u=x,objective=objective,solver_objective=float(model.ObjVal),
                          reported_gap=float(model.MIPGap),verified_gap=gap,
                          qualified=qualified,audit=audit,
                          global_gap_target_met=bool(qualified and gap is not None and gap<=mip_gap))
            return result


if __name__ == "__main__":
    # Installation example only: one standardized voltage, one free adjustment.
    # No physical case, historical data, resource budget or selection rule is changed.
    gate = build_gate([[0.]],[[1.]],[0.],[[1.]],[[0.]],[[1.]],
                      kappa=1.03,beta=.90,delta=0.)
    result = solve_continuous(gate,[1.],[-5.],[5.],mip_gap=1e-5)
    print(json.dumps(result,default=lambda a:a.tolist(),indent=2,allow_nan=False))
