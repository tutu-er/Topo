"""P-only targeted dictionary planner with the reach test (PROTOCOL v3 section 5).

Probes are u = kappa * D_sigma * s (kW), constant over the four slots of an hour.
The topology condition is the conservative form (15a): q_j(U) >= eta_j with
eta_j = 8 gamma_j, gamma_j = omega (ln 19 - E_j) + ln(1/delta_j).
"""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from . import model, scores
from . import scenario as sc

SPH = 4


def hour_slots(h):
    return np.arange(SPH*h, SPH*h+SPH)


class Gate:
    """Everything the planner may use at one gate: fits, scores and forecasts."""

    def __init__(self, inp, settings, cands, thetas, covs, sigma2, E, leader, equivalent, weights,
                 forecast, soc, spent, block_hour):
        self.inp, self.s, self.cands = inp, settings, cands
        self.thetas, self.covs, self.sigma2 = thetas, covs, sigma2
        self.E, self.L, self.eq, self.w = np.asarray(E), leader, equivalent, weights
        self.Pf, self.Qf, self.V0f = forecast            # 96 x n pu, 96 x n pu, 96 (v0 squared)
        self.soc, self.spent, self.hb = soc, spent, block_hour
        self.res, self.base_kw = inp.res, inp.base_kw
        self.sig_kw = inp.sigma_kw
        self.n = inp.n
        self.smax = settings.transformer_kva[inp.case]/inp.base_kw
        self.tr = max(0, min(settings.horizon_hours, settings.hours-block_hour))
        self.plan_hours = list(range(block_hour, block_hour+self.tr))
        self.rest_hours = list(range(block_hour, settings.hours))
        self._cache = {}
        self.plausible = [k for k in np.argsort(-np.asarray(weights)) if weights[k] >= settings.plausible_weight][:8]
        if leader not in self.plausible:
            self.plausible = [leader]+self.plausible[:7]

    # ------------------------------------------------------------ candidate data
    def cdata(self, k):
        """Forecast predictions and operating-point sensitivity of candidate k, rest of day."""
        if k not in self._cache:
            sl = np.concatenate([hour_slots(h) for h in self.rest_hours])
            c, th = self.cands[k], self.thetas[k]
            P, Q, V0 = self.Pf[sl], self.Qf[sl], self.V0f[sl]
            loss = self.s.use_loss_term
            mu = c.predict(th, P, Q, V0, loss=loss)
            if getattr(self, "offsets", None) is not None:
                mu = mu+self.offsets[k]                    # meter offsets enter the measured y
            M = c.sensitivity(th, P, Q, V0, loss=loss)
            self._cache[k] = dict(slots=sl, mu=mu, M=M)
        return self._cache[k]

    def wh(self, X):
        """GLS-whiten a T x n array per time slot (shared root-meter error); identity if gamma = 0."""
        g = getattr(self, "root_gamma", 0.)
        return X-g*X.mean(axis=1, keepdims=True) if g else X

    def whp(self, Phi):
        return model.whiten_slots(Phi, self.n, getattr(self, "root_gamma", 0.))

    def rows(self, k, hours):
        d = self.cdata(k)
        idx = np.concatenate([np.arange(SPH*(h-self.hb), SPH*(h-self.hb)+SPH) for h in hours])
        return d["mu"][idx], d["M"][idx], d["slots"][idx]

    # --------------------------------------------------------------- evidence
    def threshold(self):
        """1/t with t = alpha/(1-alpha); halved threshold (x2) when the tail uses order 1."""
        return (1-self.s.alpha)/self.s.alpha*(2. if self.s.tail_order == 1. else 1.)

    def c_tail(self, j):
        if self.s.tail_order == 1.:
            return self.threshold()*math.exp(-self.E[j])
        return math.sqrt(self.threshold())*math.exp(-.5*self.E[j])

    def gamma(self, j, delta_j):
        return self.s.omega*(math.log(self.threshold())-self.E[j])+math.log(1./delta_j)

    def eta(self, j, delta_j):
        return max(2*self.gamma(j, delta_j)/(self.s.omega*(1-self.s.omega)), 0.)

    def delta_mu(self, j, hours, u_pu_hours):
        muL, ML, _ = self.rows(self.L, hours)
        muj, Mj, _ = self.rows(j, hours)
        U = np.repeat(np.asarray(u_pu_hours), SPH, axis=0)          # T x n
        return (muL-muj)+np.einsum("tim,tm->ti", ML-Mj, U)

    def q_full(self, j, hours, u_pu_hours):
        """q_j(U) = dmu' S_jw^-1 dmu with parameter uncertainty (15a)."""
        dmu = self.wh(self.delta_mu(j, hours, u_pu_hours)).ravel()
        U = np.repeat(np.asarray(u_pu_hours), SPH, axis=0)
        _, _, sl = self.rows(j, hours)
        P, Q = self.Pf[sl]+U, self.Qf[sl]
        S = self.sigma2*np.eye(dmu.size)
        for k, wgt in ((self.L, self.s.omega), (j, 1-self.s.omega)):
            Phi = self.whp(self.cands[k].design(P, Q))
            S += wgt*Phi@self.covs[k]@Phi.T
        cf = cho_factor(S)
        return float(dmu@cho_solve(cf, dmu))

    def q_frozen_hour(self, j, h, u_pu):
        dmu = self.wh(self.delta_mu(j, [h], [u_pu]))
        return float(np.sum(dmu**2)/self.sigma2)

    # ---------------------------------------------------------------- shapes
    def shapes(self, rivals):
        n, out = self.n, []
        for i in range(n):
            for sgn in (1., -1.):
                s = np.zeros(n); s[i] = sgn
                out.append(("node%d%+d" % (i, int(sgn)), s))
        for j in rivals:
            nodes = self.differing_nodes(j)
            if 1 < len(nodes) < n:
                for sgn in (1., -1.):
                    s = np.zeros(n); s[nodes] = sgn
                    out.append(("group%d%+d" % (j, int(sgn)), s))
        return out

    def differing_nodes(self, j):
        fl, fj = self.cands[self.L].family, self.cands[j].family
        nodes = []
        for i in range(self.n):
            cl = frozenset(C for C in fl if i in C and len(C) > 1)
            cj = frozenset(C for C in fj if i in C and len(C) > 1)
            if cl != cj:
                nodes.append(i)
        return nodes

    # ------------------------------------------------------------- windows
    def headroom_hour(self, h, hours_held=1, energy=True):
        res, s = self.res, self.s
        lo, hi = s.soc_bounds
        e = res["battery_kwh"]
        if energy:
            up_b = np.minimum(res["battery_kw"], np.maximum((hi-self.soc)*e/(s.battery_eta*hours_held), 0.))
            dn_b = np.minimum(res["battery_kw"], np.maximum((self.soc-lo)*e*s.battery_eta/hours_held, 0.))
        else:
            up_b = dn_b = res["battery_kw"]
        ev = res["ev_kw"] if sc.ev_available(s, h) else 0.*res["ev_kw"]
        return up_b+ev, dn_b+s.ev_down_fraction*ev

    @property
    def cexp(self):
        """Exponent a of the cost in the level: c(k s) = k^a c(s); 2 for contract (3), 1 for linear DR."""
        return 1 if self.s.cost_model == "linear" else 2

    def level_cost(self, k, unit):
        return float(unit*k**self.cexp)

    def level_for_cost(self, cost, unit):
        return (max(cost, 0.)/unit)**(1./self.cexp)

    def unit_cost(self, s_kw, h):
        """Hourly cost (CNY) of the block s_kw at unit level: contract (3) or linear DR settlement."""
        up, dn = self.headroom_hour(h, energy=False)
        cap = np.where(s_kw >= 0, up, dn)
        if np.any((cap <= 1e-9) & (np.abs(s_kw) > 0)):
            return np.inf
        if self.s.cost_model == "linear":
            return float(np.sum(np.where(s_kw >= 0, self.s.price_up_cny, self.s.price_down_cny)*np.abs(s_kw)))
        price = sc.node_price(self.s, self.res, h)
        return float(np.sum(price*s_kw**2/np.maximum(cap, 1e-9)))

    def ceilings_hour(self, s_kw, h, energy=True, hours_held=1):
        """kappa caps for one hour: headroom/energy, voltage, transformer, unbalance, linearisation."""
        s, n = self.s, self.n
        up, dn = self.headroom_hour(h, hours_held, energy)
        cap = np.where(s_kw >= 0, up, dn)
        nz = np.abs(s_kw) > 1e-12
        kA = float(np.min(cap[nz]/np.abs(s_kw[nz]))) if np.any(nz) else np.inf
        s_pu = s_kw/self.base_kw
        sl = hour_slots(h)
        V0 = self.V0f[sl]
        # robust voltage over plausible candidates
        kV = np.inf
        lo_band, hi_band = s.voltage_band
        for k in self.plausible:
            mu, M, _ = self.rows(k, [h])
            c, th = self.cands[k], self.thetas[k]
            vkey = ("volt", k, h)
            if vkey not in self._cache:
                loss = s.use_loss_term
                f_lo, f_hi = 1-2*s.forecast_sigma, 1+2*s.forecast_sigma
                self._cache[vkey] = (
                    np.sqrt(np.maximum(V0[:, None]-mu, .25)),
                    np.sqrt(np.maximum(V0[:, None]-c.predict(th, self.Pf[sl]*f_lo, self.Qf[sl]*f_lo, V0, loss), .25)),
                    np.sqrt(np.maximum(V0[:, None]-c.predict(th, self.Pf[sl]*f_hi, self.Qf[sl]*f_hi, V0, loss), .25)))
            v_hat, hv, lv = self._cache[vkey]
            g1 = -np.einsum("tim,m->ti", M, s_pu)/(2*v_hat)
            flow = c.Z@s_pu                                  # probe flow per support
            phi = c.Z.T*flow                                 # n x S (r columns)
            cov_r = self.covs[k][:c.S, :c.S]
            b1 = np.sqrt(np.maximum(np.einsum("js,st,jt->j", phi, cov_r, phi), 0.))/(2*v_hat)
            up_den = g1+s.z_voltage*b1
            dn_den = -g1+s.z_voltage*b1
            with np.errstate(divide="ignore", invalid="ignore"):
                a = np.where(up_den > 0, np.maximum(hi_band-hv, 0.)/up_den, np.inf)
                b = np.where(dn_den > 0, np.maximum(lv-lo_band, 0.)/dn_den, np.inf)
            kV = min(kV, float(np.min(a)), float(np.min(b)))
        # transformer
        P0 = self.Pf[sl].sum(axis=1); Q0 = self.Qf[sl].sum(axis=1)
        dP = float(s_pu.sum())
        room = np.sqrt(np.maximum(self.smax**2-Q0**2, 0.))
        if abs(dP) < 1e-15:
            kS = np.inf
        elif dP > 0:
            kS = float(np.min(np.maximum(room-P0, 0.)/dP))
        else:
            kS = float(np.min(np.maximum(room+P0, 0.)/(-dP)))
        # linearisation remainder (27), leader parameters
        c, th = self.cands[self.L], self.thetas[self.L]
        r, x = th[:c.S], th[c.S:]
        a_f, b_f = c.flows(self.Pf[sl], self.Qf[sl])
        flow_bg = np.hypot(a_f, b_f)                        # T x S
        xi = 2*s.forecast_sigma*flow_bg
        dS = np.abs(c.Z@s_pu)                               # S
        muL, _, _ = self.rows(self.L, [h])
        v2 = np.maximum(V0[:, None]-muL, .25)               # T x n
        w = .25*(r**2+x**2)                                 # S
        Aq = (w*dS**2)@c.Z                                  # n
        Bq = (2*w*dS*xi)@c.Z                                # T x n
        sig_e = math.sqrt(self.sigma2)
        Aq = Aq[None, :]/v2; Bq = Bq/v2
        with np.errstate(divide="ignore", invalid="ignore"):
            root = np.where(Aq > 0, (-Bq+np.sqrt(Bq**2+4*Aq*sig_e))/(2*Aq),
                            np.where(Bq > 0, sig_e/Bq, np.inf))
        kL = float(np.min(root))
        return dict(A=kA, V=kV, S=kS, U=kV/s.unbalance_factor, L=kL)

    def window(self, s_kw):
        parts = dict(A=np.inf, V=np.inf, S=np.inf, U=np.inf, L=np.inf)
        for h in self.plan_hours:
            c = self.ceilings_hour(s_kw, h, energy=True, hours_held=self.tr)
            for key in parts:
                parts[key] = min(parts[key], c[key])
        unit = max(self.unit_cost(s_kw, h) for h in self.plan_hours)
        if not np.isfinite(unit):
            parts["B"] = 0.
        elif unit <= 0:
            parts["B"] = np.inf
        else:
            left = max(self.s.daily_cap_cny-self.spent, 0.)
            parts["B"] = min(self.level_for_cost(self.s.hourly_cap_cny, unit),
                             self.level_for_cost(left/self.tr, unit))
        return min(parts.values()), parts, unit

    # ------------------------------------------------------------ reach test
    def reach(self, j):
        """Q_bar_j: frozen-form bound on q_j for any dictionary plan of the rest of the day."""
        total = 0.
        shapes = self.shapes([j])
        for h in self.rest_hours:
            best = self.q_frozen_hour(j, h, np.zeros(self.n))
            for _, s in shapes:
                s_kw = s*self.sig_kw
                key = ("ceil", h, s.tobytes())
                if key not in self._cache:
                    self._cache[key] = min(self.ceilings_hour(s_kw, h, energy=False).values())
                kbar = self._cache[key]
                if np.isfinite(kbar) and kbar > 0:
                    best = max(best, self.q_frozen_hour(j, h, kbar*s_kw/self.base_kw))
            total += best
        return total

    # ------------------------------------------------------------ active set
    def active_set(self, rivals_sorted):
        s = self.s
        c = {j: self.c_tail(j) for j in rivals_sorted}
        total = sum(c.values())
        active, beyond, reach_log = [], [], {}
        tail = total
        if not s.tail_guarantee:
            # best-effort variant (development only): the first max_active reachable
            # rivals by BIC gap, scanning at most 10; the union-bound tail is not required
            for j in rivals_sorted[:10]:
                if len(active) >= s.max_active:
                    break
                Q = self.reach(j)
                eta_lenient = self.eta(j, s.delta)
                reach_log[j] = (Q, eta_lenient)
                (beyond if Q < eta_lenient else active).append(j)
            if not active:
                return [], beyond, reach_log, tail, None
            return active, beyond, reach_log, tail-sum(c[j] for j in active), s.delta/len(active)
        for j in rivals_sorted:
            if tail <= s.delta/2 or len(active) >= s.max_active:
                break
            Q = self.reach(j)
            eta_lenient = self.eta(j, s.delta)
            reach_log[j] = (Q, eta_lenient)
            if Q < eta_lenient:
                beyond.append(j)
            else:
                active.append(j); tail -= c[j]
        beyond_mass = sum(c[j] for j in beyond)
        if beyond_mass > s.delta/2 or tail >= s.delta:
            return [], beyond, reach_log, tail, None
        if not active:
            return [], beyond, reach_log, tail, None
        delta_j = (s.delta-tail)/len(active)
        return active, beyond, reach_log, tail, delta_j

    # ------------------------------------------------- certified planner (revision)
    def cert_coef(self, j, t):
        """t^-1/2 exp(-(E_j + beta_j)/2): rival j's weight in the miss bound B (Theorem 1)."""
        N_now = getattr(self, "N_rows", None)
        beta = 0.
        if N_now:
            N_end = N_now+SPH*self.tr*self.n
            beta = .5*(self.cands[j].p-self.cands[self.L].p)*math.log(N_end/N_now)
        return math.exp(-.5*(self.E[j]+beta))/math.sqrt(t)

    def plan_certified(self, rivals_sorted):
        """min_u c_h(u) + V [B(u) - delta]_+ over dictionary shapes x levels (PLANNER_REVISION.md section 5).

        B(u) = t^-1/2 sum_j exp(-(E_j+beta_j)/2 - q_j(u)/8) over the cert_rivals nearest rivals,
        the others held at q_j = 0 (conservative). q_j is quadratic in the level of a shape, with
        S_j evaluated at the forecast operating point (one Cholesky factor per rival).
        """
        s = self.s
        out = dict(block_kw=np.zeros(self.n), reason="", active=[], beyond=[], reach={}, kappa=0., cost=0.,
                   window_empty=False, shape=None, q_planned={}, eta={}, parts=None, B0=None, B=None)
        if self.tr <= 0:
            out["reason"] = "no_hours_left"; return out
        t = s.alpha/(1-s.alpha)
        R = list(rivals_sorted[:s.cert_rivals])
        out["active"] = R[:s.max_active]                  # logged for progress and b/J diagnostics
        B_rest = sum(self.cert_coef(j, t) for j in rivals_sorted[s.cert_rivals:])
        data = []
        for j in R:
            muL, ML, sl = self.rows(self.L, self.plan_hours)
            muj, Mj, _ = self.rows(j, self.plan_hours)
            d = self.wh(muL-muj).ravel()
            S = self.sigma2*np.eye(d.size)
            for k in (self.L, j):
                Phi = self.whp(self.cands[k].design(self.Pf[sl], self.Qf[sl]))
                S += .5*Phi@self.covs[k]@Phi.T
            cf = cho_factor(S)
            Sd = cho_solve(cf, d)
            data.append((self.cert_coef(j, t), ML-Mj, cf, Sd, float(d@Sd)))

        def bound(qs):
            return B_rest+sum(c*math.exp(-q/8.) for (c, *_), q in zip(data, qs))
        B0 = bound([x[4] for x in data])
        V, delta = s.cert_value_cny, s.delta
        best = (V*max(B0-delta, 0.), None, 0., None, 0., B0)
        for name, shape in self.shapes(R[:s.max_active]):
            s_kw = shape*self.sig_kw
            kmax, parts, unit = self.window(s_kw)
            if not np.isfinite(kmax) or kmax <= 0 or not np.isfinite(unit):
                continue
            U = np.repeat((s_kw/self.base_kw)[None, :], SPH*self.tr, axis=0)
            quad = []
            for c, D, cf, Sd, a in data:
                g = self.wh(np.einsum("tim,tm->ti", D, U)).ravel()
                quad.append((a, float(g@Sd), float(g@cho_solve(cf, g))))
            for k in np.linspace(0., kmax, s.kappa_grid+1)[1:]:
                Bk = bound([a+2*b*k+cc*k*k for a, b, cc in quad])
                obj = self.level_cost(k, unit)+V*max(Bk-delta, 0.)
                if obj < best[0]-1e-12:
                    best = (obj, name, float(k), s_kw, self.level_cost(k, unit), Bk)
        _, name, k, s_kw, cost, Bk = best
        out.update(B0=float(B0), B=float(Bk))
        if name is None:
            out["reason"] = "certified_without_block" if B0 <= delta else "no_gain"
            return out
        out.update(reason="certified" if Bk <= delta else "best_effort", shape=name, kappa=k, cost=cost,
                   window_empty=bool(Bk > delta), block_kw=k*s_kw)
        return out

    # ------------------------------------------- Bayes-risk greedy (strong greedy)
    def plan_bayes_greedy(self, rivals_sorted):
        """One-step greedy on the Bhattacharyya bound of the Bayes misidentification risk.

        U(u) = sum over pairs (c, c') of operational classes of
               sqrt(W_c W_c') exp(-q_cc'(u)/8),   W_c = sum of the posterior weights in class c,
        q_cc' evaluated on the classes' representatives: the Bhattacharyya bound on the
        probability that the MAP class after the next hour is wrong, with equivalent
        candidates treated as one hypothesis. The block minimises c(u) + V U(u) over the dictionary.
        For a fixed menu of blocks with frozen sensitivities, U(0) - U(A) is monotone
        submodular in the set A of executed blocks, so greedy schedules are
        near-optimal (PLANNER_REVISION.md section 10).
        """
        s = self.s
        out = dict(block_kw=np.zeros(self.n), reason="", active=list(rivals_sorted[:s.max_active]), beyond=[],
                   reach={}, kappa=0., cost=0., window_empty=False, shape=None, q_planned={}, eta={}, parts=None,
                   B0=None, B=None)
        if self.tr <= 0:
            out["reason"] = "no_hours_left"; return out
        # Operational classes (manuscript (19)) carry the prior mass: equivalent candidates are one
        # hypothesis. The leader's class is the gate's equivalence set (the stopping rule's own);
        # the other candidates with E < 25 are grouped around representatives on the same full-day
        # predicted voltages, at most bayes_members classes (2026-11-22 fix: members chosen by
        # individual weight were all twins of the leader whenever it had many equivalents).
        pool = [int(k) for k in np.argsort(self.E) if self.E[k] < 25.]
        if self.L not in pool:
            pool = [self.L]+pool
        volt = {}

        def v_of(k):
            if k not in volt:
                volt[k] = scores.voltages(self.cands[k], self.thetas[k], self.Pf, self.Qf, self.V0f,
                                          s.use_loss_term)
            return volt[k]
        members, mass = [self.L], [0.]
        for k in pool:
            if k == self.L or self.eq[k]:
                mass[0] += float(self.w[k]); continue
            vk = v_of(k)
            hit = next((i for i in range(1, len(members))
                        if float(np.max(np.abs(vk-v_of(members[i])))) <= s.equivalence_pu), None)
            if hit is not None:
                mass[hit] += float(self.w[k])
            elif len(members) < s.bayes_members:
                members.append(k); mass.append(float(self.w[k]))
        w = np.asarray(mass); w = w/w.sum()
        cls = list(range(len(members)))
        hours = self.plan_hours[:1]
        data = []
        for a in range(len(members)):
            for b in range(a):
                ha, hb_ = members[a], members[b]
                mua, Ma, sl = self.rows(ha, hours)
                mub, Mb, _ = self.rows(hb_, hours)
                d = self.wh(mua-mub).ravel()
                S = self.sigma2*np.eye(d.size)
                for k in (ha, hb_):
                    Phi = self.whp(self.cands[k].design(self.Pf[sl], self.Qf[sl]))
                    S += .5*Phi@self.covs[k]@Phi.T
                cf = cho_factor(S)
                Sd = cho_solve(cf, d)
                data.append((math.sqrt(w[a]*w[b]), Ma-Mb, cf, Sd, float(d@Sd)))
        if not data:
            out["reason"] = "one_class"; return out

        def risk(qs):
            return sum(om*math.exp(-q/8.) for (om, *_), q in zip(data, qs))
        U0 = risk([x[4] for x in data])
        V = s.cert_value_cny
        best = (V*U0, None, 0., None, 0., U0)
        for name, shape in self.shapes(rivals_sorted[:s.max_active]):
            s_kw = shape*self.sig_kw
            kmax, parts, unit = self.window(s_kw)
            if not np.isfinite(kmax) or kmax <= 0 or not np.isfinite(unit):
                continue
            U = np.repeat((s_kw/self.base_kw)[None, :], SPH, axis=0)
            quad = []
            for om, D, cf, Sd, a in data:
                g = self.wh(np.einsum("tim,tm->ti", D, U)).ravel()
                quad.append((a, float(g@Sd), float(g@cho_solve(cf, g))))
            for k in np.linspace(0., kmax, s.kappa_grid+1)[1:]:
                Uk = risk([a+2*b*k+cc*k*k for a, b, cc in quad])
                obj = self.level_cost(k, unit)+V*Uk
                if obj < best[0]-1e-12:
                    best = (obj, name, float(k), s_kw, self.level_cost(k, unit), Uk)
        _, name, k, s_kw, cost, Uk = best
        out.update(B0=float(U0), B=float(Uk), classes=len(set(cls)), members=len(members))
        if name is None:
            out["reason"] = "no_gain"; return out
        out.update(reason="bayes_greedy", shape=name, kappa=k, cost=cost, block_kw=k*s_kw)
        return out

    # ------------------------------------------------------- heuristic baselines
    def plan_heuristic(self, rivals_sorted, kind, target_cost=None):
        """Simple rules for comparison: single-node +/- blocks scored on the next hour only.

        greedy: largest |mu_L - mu_j1|^2 against the nearest non-equivalent rival j1.
        qbc:    largest posterior-weighted disagreement sum_k w_k |mu_k - mu_bar|^2 over the
                plausible candidates (query by committee).
        The level is the hour's ceiling (window incl. caps), or the level whose cost equals
        target_cost (matched-cost variant).
        """
        out = dict(block_kw=np.zeros(self.n), reason=kind, active=list(rivals_sorted[:1]), beyond=[], reach={},
                   kappa=0., cost=0., window_empty=False, shape=None, q_planned={}, eta={}, parts=None)
        if self.tr <= 0 or (target_cost is not None and target_cost <= 0):
            return out
        h = self.plan_hours[0]
        if kind == "greedy":
            members, w = [self.L, rivals_sorted[0]], None
        else:
            members = list(self.plausible)
            w = np.array([self.w[k] for k in members]); w = w/w.sum()
        rows = {k: self.rows(k, [h]) for k in members}
        best = (-np.inf, None, None, 0.)
        for name, shape in self.shapes([]):                  # single-node shapes only
            s_kw = shape*self.sig_kw
            kmax, parts, unit = self.window(s_kw)
            if not np.isfinite(kmax) or kmax <= 0 or not np.isfinite(unit) or unit <= 0:
                continue
            k = kmax if target_cost is None else min(kmax, self.level_for_cost(target_cost, unit))
            u = np.repeat((k*s_kw/self.base_kw)[None, :], SPH, axis=0)
            pred = {m: rows[m][0]+np.einsum("tim,tm->ti", rows[m][1], u) for m in members}
            if kind == "greedy":
                score = float(np.sum((pred[members[0]]-pred[members[1]])**2))
            else:
                mbar = sum(wi*pred[m] for wi, m in zip(w, members))
                score = float(sum(wi*np.sum((pred[m]-mbar)**2) for wi, m in zip(w, members)))
            if score > best[0]:
                best = (score, name, k*s_kw, self.level_cost(k, unit))
        if best[1] is not None:
            out.update(shape=best[1], block_kw=best[2], cost=best[3], kappa=float(np.max(np.abs(best[2]))))
        return out

    # ------------------------------------------------------------ dictionary plan
    def plan(self, rivals_sorted):
        s = self.s
        out = dict(block_kw=np.zeros(self.n), reason="", active=[], beyond=[], reach={}, kappa=0.,
                   cost=0., window_empty=False, shape=None, q_planned={}, eta={}, parts=None)
        if self.tr <= 0:
            out["reason"] = "no_hours_left"; return out
        active, beyond, reach_log, tail, delta_j = self.active_set(rivals_sorted)
        out.update(active=active, beyond=beyond, reach={int(k): [float(a), float(b)] for k, (a, b) in reach_log.items()},
                   tail=float(tail))
        if not active:
            out["reason"] = "beyond_reach" if beyond else "no_active"; return out
        etas = {j: self.eta(j, delta_j) for j in active}
        out["eta"] = {int(j): float(v) for j, v in etas.items()}
        active = [j for j in active if etas[j] > 0]          # eta_j = 0: already met
        if not active:
            out["reason"] = "already_met"; return out
        zero = [np.zeros(self.n)]*self.tr
        if all(self.q_full(j, self.plan_hours, zero) >= etas[j] for j in active):
            out["reason"] = "background_suffices"; return out
        best, fallback = None, None
        for name, shape in self.shapes(active):
            s_kw = shape*self.sig_kw
            kmax, parts, unit = self.window(s_kw)
            if not np.isfinite(kmax) or kmax <= 0 or not np.isfinite(unit):
                continue

            def ratio(k):
                u = [k*s_kw/self.base_kw]*self.tr
                return min(self.q_full(j, self.plan_hours, u)/etas[j] for j in active)
            grid = np.linspace(0., kmax, s.kappa_grid+1)[1:]
            prev, hit = 0., None
            for k in grid:
                if ratio(k) >= 1.:
                    hit = k; break
                prev = k
            if hit is None:
                r_top = ratio(kmax)
                if fallback is None or r_top > fallback[0]:
                    fallback = (r_top, name, s_kw, kmax, unit, parts)
                continue
            lo, hi = prev, hit
            for _ in range(s.bisection_steps):
                mid = .5*(lo+hi)
                if ratio(mid) >= 1.:
                    hi = mid
                else:
                    lo = mid
                if hi-lo <= 1e-4*hi:
                    break
            cost = self.level_cost(hi, unit)
            if best is None or cost < best[0]:
                best = (cost, name, s_kw, hi, unit, parts, lo)
        if best is not None:
            cost, name, s_kw, k, unit, parts, below = best
            out.update(reason="admissible", shape=name, kappa=float(k), kappa_below=float(below))
        elif fallback is not None:
            _, name, s_kw, k, unit, parts = fallback
            out.update(reason="window_empty", shape=name, kappa=float(k), window_empty=True)
        else:
            out["reason"] = "no_feasible_shape"; return out
        out["block_kw"] = k*s_kw
        out["cost"] = self.level_cost(k, self.unit_cost(s_kw, self.hb))
        out["parts"] = {key: float(v) for key, v in parts.items()}
        u = [out["block_kw"]/self.base_kw]*self.tr
        out["q_planned"] = {int(j): float(self.q_full(j, self.plan_hours, u)) for j in active}
        return out


def block_cost(settings, res, hour, u_kw):
    """Cost of an executed block, one hour: contract (3) or linear DR settlement per kWh moved."""
    u_kw = np.asarray(u_kw, float)
    if settings.cost_model == "linear":
        return float(settings.price_up_cny*np.sum(np.maximum(u_kw, 0.))
                     + settings.price_down_cny*np.sum(np.maximum(-u_kw, 0.)))
    up = res["battery_kw"]+(res["ev_kw"] if sc.ev_available(settings, hour) else 0.)
    dn = res["battery_kw"]+settings.ev_down_fraction*(res["ev_kw"] if sc.ev_available(settings, hour) else 0.)
    cap = np.where(u_kw >= 0, up, dn)
    price = sc.node_price(settings, res, hour)
    return float(np.sum(price*u_kw**2/np.maximum(cap, 1e-9)))


def check_block(settings, res, hour, soc, rng):
    """Broad zero-sum check (24): random signs at rho x headroom, buys = sells."""
    up = res["battery_kw"]+(res["ev_kw"] if sc.ev_available(settings, hour) else 0.)
    dn = np.minimum(res["battery_kw"], np.maximum((soc-settings.soc_bounds[0])*res["battery_kwh"], 0.))
    signs = rng.choice([-1., 1.], size=len(up))
    if signs.min() == signs.max():
        signs[rng.integers(len(signs))] *= -1
    u = settings.check_rho*np.where(signs > 0, up, dn)*signs
    pos, neg = u[u > 0].sum(), -u[u < 0].sum()
    if pos <= 0 or neg <= 0:
        return np.zeros_like(u)
    if pos > neg:
        u[u > 0] *= neg/pos
    else:
        u[u < 0] *= pos/neg
    return u


def random_block(settings, res, hour, cost, rng, soc):
    """Random single-node block with the given hourly cost, within headroom."""
    n = len(res["battery_kw"])
    i = int(rng.integers(n)); sgn = rng.choice([-1., 1.])
    up = res["battery_kw"][i]+(res["ev_kw"][i] if sc.ev_available(settings, hour) else 0.)   # contract basis only
    # the cap is the state-of-charge-aware headroom every other arm uses (2026-11-23: the old cap let a full
    # battery charge at its rated power, which run.py's headroom_excess record caught in section 18)
    hu, hd = sc.headroom(settings, res, hour, soc)
    cap = hu[i] if sgn > 0 else hd[i]
    if settings.cost_model == "linear":
        unit = settings.price_up_cny if sgn > 0 else settings.price_down_cny
        u = np.zeros(n); u[i] = sgn*min(max(cost, 0.)/unit, cap)
        return u
    price = sc.node_price(settings, res, hour)[i]
    contract = up if sgn > 0 else res["battery_kw"][i]+settings.ev_down_fraction*(
        res["ev_kw"][i] if sc.ev_available(settings, hour) else 0.)
    level = math.sqrt(max(cost, 0.)*contract/price) if price > 0 else 0.
    u = np.zeros(n); u[i] = sgn*min(level, cap)
    return u
