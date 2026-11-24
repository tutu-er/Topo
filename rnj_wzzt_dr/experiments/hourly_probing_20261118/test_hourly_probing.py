"""Unit tests from PROTOCOL v3 section 9 (the subset the development code covers)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from rnj_wzzt_dr.identification.optimization import QPFitter
from rnj_wzzt_dr.physical import backend
from rnj_wzzt_dr.physical.backend import Observation

from . import bank, config, model, planner, run, scores, unbalanced
from . import scenario as sc

S = config.Settings()


def true_theta(network, labels, cand):
    order, parent, incoming = backend._radial_order(network)
    zb = network["base_kv"]**2/network["base_mva"]
    desc = {b: set() for b in order}
    for i, l in enumerate(labels):
        desc[l].add(i)
    for c in reversed(order[1:]):
        desc[parent[c]].update(desc[c])
    th = np.zeros(2*cand.S)
    for c in order[1:]:
        k = cand.family.index(tuple(sorted(desc[c])))
        th[k] += 2*incoming[c]["r_ohm"]/zb; th[cand.S+k] += 2*incoming[c]["x_ohm"]/zb
    return th


@pytest.fixture(scope="module")
def light_input():
    return bank.build("soumalas11", 5, S, history=(96, 48), use_identifier=False)


@pytest.mark.parametrize("case", ["soumalas11", "pengwah18"])
def test_loss_term_cuts_ac_gap_and_sensitivity_matches_ac(case):
    net, labels, _, _ = sc.network_info(case)
    P, Q, root = sc.generate_load(case, 2, 1)
    P, Q = 1.3*P, 1.3*Q
    v, _ = backend._ac_voltages(net, labels, P, Q, root)
    y = root[:, None]**2-v**2
    cand = model.Candidate(sc.truth_family(net, labels), len(labels))
    th = true_theta(net, labels, cand)
    gap_lin = np.abs(y-cand.linear(th, P, Q)).max()
    gap_loss = np.abs(y-cand.predict(th, P, Q, root**2)).max()
    assert gap_loss <= .1*gap_lin
    t, m, h = 40, 2, 1e-4
    Pp = P[t:t+1].copy(); Pp[0, m] += h
    vp, _ = backend._ac_voltages(net, labels, Pp, Q[t:t+1], root[t:t+1])
    dy_ac = ((root[t]**2-vp**2)-y[t:t+1])[0]/h
    M = cand.sensitivity(th, P[t:t+1], Q[t:t+1], root[t:t+1]**2)[0][:, m]
    assert np.max(np.abs(M-dy_ac)) <= .02*np.max(np.abs(dy_ac))


def test_three_phase_equal_split_matches_backend():
    net, labels, _, _ = sc.network_info("pengwah18")
    P, Q, root = sc.generate_load("pengwah18", 3, 1)
    v, _ = backend._ac_voltages(net, labels, P, Q, root)
    tp = unbalanced.ThreePhase(net, labels, equal=True)
    assert np.abs(tp.terminal_voltages(P, Q, root)-v).max() <= 1e-6


def test_refit_matches_qpfitter_without_floor_and_keeps_floor(light_input):
    inp = light_input
    obs = Observation(inp.hist["P"], inp.hist["Q"], inp.hist["Y"], inp.labels)
    fitter = QPFitter(obs, bound=2.)
    for fam in inp.families[:12]:
        c = model.Candidate(fam, inp.n)
        Phi = c.design(inp.hist["P"], inp.hist["Q"])
        f = model.Fit(c, np.zeros(2*c.S), 0., 2.).refit(Phi, inp.hist["Y"].ravel(), None, use_loss=False)
        ref = fitter.fit([list(C) for C in c.family])
        ours = f.sse/inp.hist["Y"].size
        assert abs(ours-ref["mse"]) <= 1e-9*ref["mse"]
        r_min = .0075
        g = model.Fit(c, np.zeros(2*c.S), r_min, 2.).refit(Phi, inp.hist["Y"].ravel(), None, use_loss=False)
        r = g.theta[:c.S]
        assert np.all(r[~c.is_single] >= r_min-1e-12)
        cov = g.covariance(1e-6)
        at_floor = np.flatnonzero((~c.is_single) & (np.abs(r-r_min) < 1e-9))
        for k in at_floor:
            assert np.all(cov[k] == 0) and c.S == len(c.family)   # stays in the candidate


def test_extra_line_rival_bic_tail_and_aligned_error():
    """Manuscript Proposition 6: P(E_W < ln19) <= 19/N under white noise; aligned error flips."""
    net, labels, _, _ = sc.network_info("soumalas11")
    n = len(labels)
    P, Q, _ = sc.generate_load("soumalas11", 4, 1)
    T = sc.truth_family(net, labels)
    cT = model.Candidate(T, n)
    W = list(T)+[(9, 10)]                                    # extra line under node 4
    cW = model.Candidate(W, n)
    th = true_theta(net, labels, cT)
    PhiT, PhiW = cT.design(P, Q), cW.design(P, Q)
    mu = PhiT@th
    N = mu.size
    sigma = 1.4e-3
    ln19 = math.log(19.)
    e_dir = PhiW[:, cW.family.index((9, 10))]
    proj = PhiT@np.linalg.lstsq(PhiT, e_dir, rcond=None)[0]
    e_dir = e_dir-proj; e_dir /= np.linalg.norm(e_dir)
    beta_star = math.sqrt(2*(math.log(N)+ln19)/N)
    rng = np.random.default_rng(0)
    below, lead = 0, 0
    for seed in range(1000):
        noise = sigma*rng.normal(size=N)
        for aligned, y in ((False, mu+noise), (True, mu+noise+2*beta_star*sigma*math.sqrt(N)*e_dir)):
            fT = model.Fit(cT, np.zeros(2*cT.S), 0., 2.).refit(PhiT, y, None, use_loss=False)
            fW = model.Fit(cW, np.zeros(2*cW.S), 0., 2.).refit(PhiW, y, None, use_loss=False)
            J, _ = scores.bic([fT.sse, fW.sse], [cT.p, cW.p], N)
            if aligned:
                lead += J[1] < J[0]-ln19
            else:
                below += J[1]-J[0] < ln19
    p = 19/N
    assert below/1000 <= p+3*math.sqrt(p*(1-p)/1000)
    assert lead/1000 >= .9


def _gate(inp, settings, rival_filter=None, zero_cov=False, block_hour=10):
    cands = [model.Candidate(f, inp.n) for f in inp.families]
    zbase = inp.network["base_kv"]**2/inp.network["base_mva"]
    fits, J, sigma2 = bank.passive_fit(inp.families, inp.n, inp.hist, 2*settings.r_min_ohm/zbase, 2.)
    L = int(np.argmin(J)); E = J-J[L]
    w = np.exp(-(J-J.min())); w /= w.sum()
    covs = [np.zeros((2*c.S, 2*c.S)) if zero_cov else f.covariance(sigma2) for c, f in zip(cands, fits)]
    eq = np.zeros(len(cands), bool); eq[L] = True
    fc = (inp.day["P"], inp.day["Q"], inp.root_profile**2)
    gt = planner.Gate(inp, settings, cands, [f.theta for f in fits], covs, sigma2, E, L, eq, w, fc,
                      np.full(inp.n, .5), 0., block_hour)
    rivals = sorted([k for k in range(len(cands)) if k != L and (rival_filter is None or rival_filter(cands[k], cands[L]))],
                    key=lambda k: E[k])
    return gt, rivals, cands, L, E


def test_frozen_q_matches_closed_form_and_plan_meets_condition(light_input):
    inp = light_input
    gt, rivals, cands, L, E = _gate(inp, S, zero_cov=True,
                                    rival_filter=lambda c, cl: c.key < cl.key)     # missing-line rivals
    j = rivals[0]
    u = np.zeros(inp.n); u[3] = 1.5/inp.base_kw
    hours = gt.plan_hours
    q = gt.q_full(j, hours, [u]*len(hours))
    closed = sum(gt.q_frozen_hour(j, h, u) for h in hours)
    assert abs(q-closed) <= 1e-8*max(closed, 1.)
    # give the rival a small deficit so that the planner must buy
    gt.E = gt.E.copy(); gt.E[j] = 0.
    out = gt.plan([j])
    if out["reason"] == "admissible":
        eta = out["eta"][j]
        k = out["kappa"]; s_kw = out["block_kw"]/k
        assert gt.q_full(j, hours, [k*s_kw/inp.base_kw]*len(hours)) >= eta*(1-1e-9)
        assert gt.q_full(j, hours, [out["kappa_below"]*s_kw/inp.base_kw]*len(hours)) < eta
    assert np.all(np.isfinite(out["block_kw"]))


def test_active_set_tail_and_reach_bound(light_input):
    inp = light_input
    gt, rivals, cands, L, E = _gate(inp, S)
    active, beyond, reach_log, tail, delta_j = gt.active_set(rivals)
    if active:
        assert tail <= S.delta/2+1e-12 or len(active) == S.max_active
    # reach bound >= frozen q of random dictionary plans for the rest of the day
    j = rivals[0]
    Qbar = gt.reach(j)
    rng = np.random.default_rng(1)
    shapes = gt.shapes([j])
    for _ in range(20):
        total = 0.
        for h in gt.rest_hours:
            _, s = shapes[rng.integers(len(shapes))]
            s_kw = s*gt.sig_kw
            kbar = min(gt.ceilings_hour(s_kw, h, energy=False).values())
            k = rng.uniform(0, kbar) if np.isfinite(kbar) else 0.
            total += gt.q_frozen_hour(j, h, k*s_kw/inp.base_kw)
        assert total <= Qbar*(1+1e-9)


def test_extra_line_rival_at_floor_is_beyond_reach(light_input):
    inp = light_input
    gt, rivals, cands, L, E = _gate(inp, S, rival_filter=lambda c, cl: c.key > cl.key, block_hour=12)
    assert cands[L].key == inp.truth_key
    floor = 2*S.r_min_ohm/.8
    at_floor = [j for j in rivals
                if any(abs(gt.thetas[j][k]-floor) < 1e-9 for k in range(cands[j].S) if not cands[j].is_single[k])]
    j = at_floor[0]
    gt.E = gt.E.copy(); gt.E[j] = 1.                       # closest rival, inside ln 19
    Q = gt.reach(j)
    assert Q < gt.eta(j, S.delta)
    out = gt.plan([j])
    assert out["reason"] == "beyond_reach" and not np.any(out["block_kw"])


def test_zero_sum_check_and_random_block():
    res = sc.resources(S, np.full(5, 4.), 200.)
    rng = np.random.default_rng(3)
    for h in (3, 12, 20):
        u = planner.check_block(S, res, h, np.full(5, .5), rng)
        assert abs(u.sum()) <= 1e-12
        up, dn = sc.headroom(S, res, h, np.full(5, .5))
        assert np.all(u <= up+1e-12) and np.all(-u <= dn+1e-12)
        r = planner.random_block(S, res, h, .2, rng, np.full(5, .5))
        assert np.count_nonzero(r) == 1
    # random blocks stay within the state-of-charge-aware headroom, also with full and empty batteries
    s = config.Settings(cost_model="linear", price_down_cny=4., price_up_cny=1.2)
    for soc0 in (S.soc_bounds[0], .5, S.soc_bounds[1]):
        soc = np.full(5, soc0)
        for h in (3, 12, 20):
            up, dn = sc.headroom(s, res, h, soc)
            for _ in range(20):
                r = planner.random_block(s, res, h, 50., rng, soc)
                assert np.all(r <= up+1e-12) and np.all(-r <= dn+1e-12)


def test_replay_arms_are_paired(light_input):
    inp = light_input
    s = S.with_(hours=8)
    base = run.run(inp, "t_opt_P", 0, s)
    fr = run.run(inp, "frozen_replay", 0, s, replay=base["schedule"])
    assert np.allclose(fr["schedule"]["blocks"], base["schedule"]["blocks"])
    rm = run.run(inp, "random_matched_P", 0, s, replay=base["schedule"])
    for h, cost in enumerate(base["schedule"]["costs"]):
        assert rm["schedule"]["costs"][h] <= cost+1e-9
        if cost == 0:
            assert rm["schedule"]["costs"][h] == 0
    for g in base["gates"]:
        assert g["g"] <= s.hours        # each plan sealed before its hour is revealed


def test_meter_offsets_match_free_intercepts_and_recover_bias():
    """Demeaned refit = bounded LS with a free constant per meter (FWL); offsets
    absorb a fixed voltage-meter bias, and the closure keeps the truth."""
    from scipy.optimize import lsq_linear
    net, labels, _, _ = sc.network_info("soumalas11")
    n = len(labels)
    P, Q, root = sc.generate_load("soumalas11", 3, 1)
    v, _ = backend._ac_voltages(net, labels, P, Q, root)
    bias = .0005*np.random.default_rng(1).normal(size=n)
    y = root[:, None]**2-(v*(1+bias))**2
    cand = model.Candidate(sc.truth_family(net, labels), n)
    Phi = cand.design(P, Q)
    zb = net["base_kv"]**2/net["base_mva"]
    f = model.Fit(cand, np.zeros(2*cand.S), 2*.003/zb, 2.).refit(Phi, y.ravel(), None, offsets=True)
    E = np.tile(np.eye(n), (P.shape[0], 1))
    lb = np.concatenate((f.lb, np.full(n, -np.inf))); ub = np.concatenate((f.ub, np.full(n, np.inf)))
    aug = lsq_linear(np.hstack((Phi, E)), y.ravel(), bounds=(lb, ub), method="trf", tol=1e-12, max_iter=20000)
    assert np.allclose(f.theta, aug.x[:2*cand.S], atol=1e-6)
    assert np.allclose(f.offset, aug.x[2*cand.S:], atol=1e-6)
    resid = y.ravel()-np.hstack((Phi, E))@aug.x
    assert abs(f.sse-resid@resid) <= 1e-6*max(resid@resid, 1e-12)
    # the offset is about -2 bias V^2 per meter (the linear model error is common to both fits)
    assert np.corrcoef(f.offset, -2*bias*(v**2).mean(0))[0, 1] > .95
    # closure from a one-edit neighbour finds the truth and ranks it first with offsets
    hist = dict(P=P, Q=Q, Y=y, V0sq=root**2)
    start = [model.canonical_family(bank.neighbours(cand.family, n)[0], n)]
    D, fits, J, _, rec = bank.closure(start, n, hist, 2*.003/zb, 2., offsets=True)
    keys = [model.nonsingleton(fm) for fm in D]
    assert cand.key in keys and keys[int(np.argmin(J))] == cand.key and rec["capped"] is None


@pytest.mark.parametrize("case", ["soumalas11", "pengwah18", "flynn16", "paper15"])
def test_truth_is_one_edit_from_every_merged_segment(case):
    """A short segment merged away (its support dropped) must be recoverable by the
    closure in one edit, on every internal segment of every case feeder."""
    net, labels, _, _ = sc.network_info(case)
    n = len(labels)
    truth = model.canonical_family(sc.truth_family(net, labels), n)
    key = model.nonsingleton(truth)
    for C in key:
        if len(C) == n:
            continue
        merged = model.canonical_family([S for S in truth if S != C], n)
        keys = {model.nonsingleton(model.canonical_family(f, n)) for f in bank.neighbours(merged, n)}
        assert key in keys, (case, C)


def test_certified_miss_bound_holds_and_matches_v3_allocation():
    """Theorem 1 by Monte Carlo (independent and fully correlated increments) and Corollary 1."""
    rng = np.random.default_rng(7)
    t, delta = S.alpha/(1-S.alpha), S.delta
    for trial in range(20):
        m = int(rng.integers(2, 12))
        E = rng.uniform(0., 6., m); q = rng.uniform(0., 60., m)
        B = float(np.sum(np.exp(-(E/2+q/8))))/math.sqrt(t)
        for corr in (False, True):
            z = rng.normal(size=(20000, 1 if corr else m))
            lam = q/2+np.sqrt(q)*z
            miss = np.mean(np.sum(np.exp(-(E+lam)), axis=1) > t)
            assert miss <= B+3*math.sqrt(max(B*(1-B), 1e-4)/20000)
    # Corollary 1: with delta_j = the rival's own term, q_j equals eta_j(delta_j) (omega = 1/2)
    E = np.array([.5, 1.5, 3.]); q = np.array([20., 15., 4.])
    dj = np.exp(-(E/2+q/8))/math.sqrt(t)
    eta = 8*(.5*(math.log(1/t)-E)+np.log(1/dj))
    assert np.allclose(q, eta)


def test_certified_planner_and_heuristics(light_input):
    inp = light_input
    gt, rivals, cands, L, E = _gate(inp, S)
    gt.E = gt.E.copy()
    for j in rivals[:5]:
        gt.E[j] = min(gt.E[j], 1.)                          # several near rivals: v3 would be idle
    out = gt.plan_certified(rivals)
    assert out["B"] <= out["B0"]+1e-12
    if np.any(out["block_kw"]):
        kmax, _, unit = gt.window(out["block_kw"]/out["kappa"])
        assert out["kappa"] <= kmax*(1+1e-9)
        assert out["cost"]+S.cert_value_cny*max(out["B"]-S.delta, 0.) \
            <= S.cert_value_cny*max(out["B0"]-S.delta, 0.)+1e-9
    # with no value on certainty nothing is bought
    gt0, rivals0, *_ = _gate(inp, S.with_(cert_value_cny=0.))
    assert not np.any(gt0.plan_certified(rivals0)["block_kw"])
    for kind in ("greedy", "qbc"):
        h = gt.plan_heuristic(rivals, kind)
        assert np.all(np.isfinite(h["block_kw"])) and np.count_nonzero(h["block_kw"]) <= 1
        if np.any(h["block_kw"]):
            m = gt.plan_heuristic(rivals, kind, target_cost=.5*h["cost"])
            assert m["cost"] <= .5*h["cost"]*(1+1e-9)


def test_root_whitening_is_gls_for_the_shared_root_error():
    """whiten_slots + refit = generalised least squares with per-slot covariance I + rho 11'."""
    from scipy.optimize import lsq_linear
    net, labels, _, _ = sc.network_info("soumalas11")
    n = len(labels)
    P, Q, root = sc.generate_load("soumalas11", 4, 1)
    v, _ = backend._ac_voltages(net, labels, P, Q, root)
    rng = np.random.default_rng(2)
    vm = v*(1+.0005*rng.normal(size=v.shape)); rm = root*(1+.0005*rng.normal(size=root.shape))
    y = rm[:, None]**2-vm**2
    gamma = model.root_gamma(rm**2, y)
    rho = (1/(1-gamma)**2-1)/n
    assert .9 < rho < 1.2                                   # root and terminal errors of equal size
    cand = model.Candidate(sc.truth_family(net, labels), n)
    Phi = cand.design(P, Q)
    zb = net["base_kv"]**2/net["base_mva"]
    f = model.Fit(cand, np.zeros(2*cand.S), 2*.003/zb, 2.).refit(Phi, y.ravel(), None, gamma=gamma)
    Lc = np.linalg.cholesky(np.eye(n)+rho*np.ones((n, n)))
    Wi = np.linalg.inv(Lc)
    T = P.shape[0]
    Aw = np.einsum("ij,tjk->tik", Wi, Phi.reshape(T, n, -1)).reshape(Phi.shape)
    bw = np.einsum("ij,tj->ti", Wi, y).ravel()
    ref = lsq_linear(Aw, bw, bounds=(f.lb, f.ub), method="bvls", tol=1e-12)
    assert np.allclose(f.theta, ref.x, atol=1e-7)
    r = bw-Aw@ref.x
    assert abs(f.sse-r@r) <= 1e-6*(r@r)


def test_bayes_risk_reduction_is_monotone_submodular_and_planner_runs(light_input):
    """F(A) = U(0) - U(A) with U(A) = sum_p w_p exp(-sum_{b in A} q_pb/8): monotone and submodular."""
    rng = np.random.default_rng(11)
    for _ in range(50):
        pairs, menu = int(rng.integers(2, 8)), 6
        w = rng.uniform(0, 1, pairs); q = rng.exponential(5., (pairs, menu))
        F = lambda A: float(np.sum(w*(1-np.exp(-q[:, sorted(A)].sum(axis=1)/8)))) if A else 0.
        A = set(rng.choice(menu, 2, replace=False).tolist()); B = A | {int(x) for x in rng.choice(menu, 2)}
        for b in range(menu):
            if b in B:
                continue
            assert F(B | {b}) >= F(B)-1e-12                     # monotone
            assert F(A | {b})-F(A) >= F(B | {b})-F(B)-1e-12     # diminishing returns
    inp = light_input
    gt, rivals, cands, L, E = _gate(inp, S)
    gt.E = gt.E.copy()
    for j in rivals[:5]:
        gt.E[j] = min(gt.E[j], 1.)
    gt.w = np.exp(-gt.E); gt.w = gt.w/gt.w.sum()
    out = gt.plan_bayes_greedy(rivals)
    if out["B0"] is not None:
        assert out["B"] <= out["B0"]+1e-12
        assert out["cost"]+S.cert_value_cny*out["B"] <= S.cert_value_cny*out["B0"]+1e-9


def test_linear_dr_cost_model():
    """Linear DR settlement: CNY per kWh moved, reductions and increases priced separately."""
    s = config.Settings(cost_model="linear", price_down_cny=4., price_up_cny=1.2)
    res = dict(battery_kw=np.full(3, 5.), ev_kw=np.zeros(3), battery_kwh=np.full(3, 13.5),
               households=np.ones(3), base_kw=1.)
    u = np.array([2., -1., 0.])
    assert planner.block_cost(s, res, 12, u) == pytest.approx(1.2*2+4.*1)
    rng = np.random.default_rng(0)
    for _ in range(10):
        v = planner.random_block(s, res, 12, 2.4, rng, np.full(3, .5))
        assert planner.block_cost(s, res, 12, v) == pytest.approx(2.4)
    # the contract model is unchanged by default
    assert planner.block_cost(S, res, 12, u) == pytest.approx(
        float(np.sum(sc.node_price(S, res, 12)*u**2/5.)))


def test_committed_block_counts_against_budget():
    """The block sealed for the current hour counts against the day's budget and battery energy."""
    s = config.Settings(cost_model="linear", price_down_cny=4., price_up_cny=1.2)
    res = dict(battery_kw=np.full(3, 5.), ev_kw=np.zeros(3), battery_kwh=np.full(3, 13.5),
               households=np.ones(3), base_kw=1.)
    blocks = np.zeros((s.hours, 3))
    blocks[5] = [-2., 0., 1.]
    soc = np.full(3, .5)
    spent, soc_c = run.committed(s, res, 5, 6, blocks, 10., soc)
    assert spent == pytest.approx(10.+4.*2+1.2*1)
    assert soc_c[0] < .5 and soc_c[1] == pytest.approx(.5) and soc_c[2] > .5
    assert np.all(soc == .5)                       # the caller's state is not changed
    # nothing sealed for the current hour: unchanged
    spent0, soc0 = run.committed(s, res, 4, 5, blocks, 10., soc)
    assert spent0 == 10. and np.allclose(soc0, soc)


def test_locked_settings_are_the_confirmed_ones():
    """The locked selector settings (PROTOCOL section 11, 2026-11-23) are exactly those confirmed in
    DEVELOPMENT_RUNS section 18, and the locked values cannot be overridden."""
    from .experiments import settings_for
    for cond, bias in (("lock_short01", .0005), ("lock_noisy", .001)):
        tested = settings_for(cond).with_(cert_value_cny=2000.)          # arm bgreedy_V2000_P
        noise = dict(meter_bias_sigma=bias, voltage_sigma=tested.voltage_sigma)
        assert config.locked_settings(**noise) == tested
    assert config.LOCKED_ARM == "bgreedy_P"
    with pytest.raises(ValueError):
        config.locked_settings(cert_value_cny=200.)
