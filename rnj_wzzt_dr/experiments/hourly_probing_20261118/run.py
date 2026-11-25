"""Seal-then-reveal hourly runner for one input, one path and one arm.

At gate g (start of hour g) the data run to the end of hour g-1. The runner
refits and scores every candidate, applies the feedback rules and seals the
block for hour g + lead. Only then is hour g simulated on the true network
and metered. Truth enters only the physics and the evaluation fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import numpy as np

from . import model, planner, scores
from . import scenario as sc



@dataclass
class Input:
    name: str
    case: str
    seed: int
    stratum: str
    network: dict
    labels: tuple
    n: int
    base_kw: float
    hist: dict                  # metered P, Q, Y, V0sq (T x n / T)
    day: dict                   # true background P, Q, root (slots of the day x n / slots)
    families: list
    theta0: list
    truth_key: frozenset
    truth_in_bank: bool
    res: dict
    sigma_kw: np.ndarray
    root_profile: np.ndarray    # history mean root voltage per slot of the day
    meta: dict = field(default_factory=dict)


def _forecast(inp, settings, rng):
    T, n = inp.day["P"].shape
    e = np.zeros((T, n))
    shock = rng.normal(0., 1., (T, n))
    e[0] = shock[0]
    for t in range(1, T):
        e[t] = .9*e[t-1]+math.sqrt(1-.81)*shock[t]
    factor = 1.+settings.forecast_sigma*e
    return inp.day["P"]*factor, inp.day["Q"]*factor, inp.root_profile**2


def committed(s, res, g, hb, blocks, spent, soc):
    """Spend and battery state after the blocks already sealed for hours g..hb-1 but not yet run.

    The block for hour hb is sealed at gate g while the block for hour g is still to run, so the
    daily budget and the battery energy must count it (fixed 2026-11-23, DEVELOPMENT_RUNS section 16).
    """
    soc = np.array(soc, float)
    for h in range(g, min(hb, s.hours)):
        if np.any(blocks[h] != 0):
            spent += planner.block_cost(s, res, h, blocks[h])
            soc = sc.update_soc(s, res, soc, sc.split_block(s, res, h, soc, blocks[h]))
    return float(spent), soc


def run(inp, arm, path, settings, replay=None):
    """Return the trajectory record of `arm` on `path` (paired across arms)."""
    started = time.perf_counter()
    n, s = inp.n, settings
    SPH = s.slots_per_hour                        # rows per hour (4: 15-minute data, 1: hourly)
    if arm == "bgreedy_lowV_P":                       # frugal strong greedy: price of certainty V/5
        s = s.with_(cert_value_cny=s.cert_value_cny/5.)
    elif arm.startswith("bgreedy_V") and arm.endswith("_P"):   # V sweep: bgreedy_V<CNY>_P
        s = s.with_(cert_value_cny=float(arm[len("bgreedy_V"):-2]))
    cands = [model.Candidate(f, n) for f in inp.families]
    if s.identifiable_params:                         # count identifiable parameters only (REAL_NETWORK_CASE.md)
        for c in cands:
            c.p = model.identifiable_p(c, inp.hist["P"], inp.hist["Q"], s.pq_sigma)
    K = len(cands)
    truth_idx = next((k for k, c in enumerate(cands) if c.key == inp.truth_key), None)
    streams = np.random.SeedSequence([2026111901, int(inp.seed), int(path)]).spawn(6)
    rng_fc, rng_gain, rng_meter, rng_arm, rng_chk, _ = [np.random.default_rng(x) for x in streams]
    Pf, Qf, V0f = _forecast(inp, s, rng_fc)
    gains = rng_gain.uniform(*s.delivery_gain, n)
    meters = sc.make_meters(s, n, inp.seed)
    physics = sc.Physics(inp.network, inp.labels, s, seed=inp.seed, households=inp.res["households"])
    meter_seeds = rng_meter.integers(0, 2**31, s.hours)
    v_bg = physics.voltages(inp.day["P"], inp.day["Q"], inp.day["root"])     # evaluation only

    H = inp.hist
    T_day = SPH*s.hours
    day_P = np.zeros((T_day, n)); day_Q = np.zeros((T_day, n)); day_Y = np.zeros((T_day, n)); day_V0 = np.zeros(T_day)
    blocks = np.zeros((s.hours, n))
    fits = [model.Fit(c, th, 2*s.r_min_ohm/(inp.network["base_kv"]**2/inp.network["base_mva"]), s.upper_bound)
            for c, th in zip(cands, inp.theta0)]
    theta_frozen = [f.theta.copy() for f in fits]
    phi_hist = [c.design(H["P"], H["Q"]) for c in cands]
    gamma = model.root_gamma(H["V0sq"], H["Y"]) if s.root_whitening else 0.
    soc = np.full(n, .5)
    spent, root_kwh = 0., 0.
    frozen = arm == "frozen_replay"
    probing = arm in ("t_opt_P", "check_then_t_opt", "cert_P", "qbc_P", "greedy_P") or arm.startswith("bgreedy")
    closure = inp.meta.get("closure") or {}
    closure_ok = closure.get("capped") is None and not closure.get("unexpanded_within_margin", 0)
    gate_log, block_log = [], []
    status = dict(leader=None, count=0, confirmed=False, stopped=False, check_done=0, check_until=-1)
    ever_wrong, first_correct, ever_wrong_types, wrong_leaders = False, None, set(), set()
    progress = []                                     # (gate, G) at probed gates
    last_beyond, last_reach = [], {}
    violations = dict(ac_band_hours=0, deadband_hours=0, max_v=0., min_v=9.)
    headroom_excess = dict(hours=0, kwh=0., max_kw=0.)
    pending_eval = {}
    # prequential check (evaluation only): per candidate, summed standardized one-hour-ahead prediction errors
    pre = dict(probe=np.zeros(K), probe_df=np.zeros(K), passive=np.zeros(K), passive_df=np.zeros(K))
    sigma2_prev = None

    def data_upto(g):
        m = SPH*g
        P = np.vstack((H["P"], day_P[:m])); Q = np.vstack((H["Q"], day_Q[:m]))
        Y = np.vstack((H["Y"], day_Y[:m])); V0 = np.concatenate((H["V0sq"], day_V0[:m]))
        return P, Q, Y, V0, m

    for g in range(s.hours+1):
        t0 = time.perf_counter()
        P, Q, Y, V0, m = data_upto(g)
        y = Y.ravel()
        if s.prequential_check and g >= 1 and not frozen and sigma2_prev is not None:
            # each candidate, as fitted before hour g-1 arrived, predicts hour g-1 (evaluation only)
            h = g-1; slh = slice(SPH*h, SPH*g)
            key = "probe" if np.any(blocks[h] != 0) else "passive"
            for k, (c, f) in enumerate(zip(cands, fits)):
                q, df = _prequential(c, f, phi_hist[k], day_P[:SPH*h], day_Q[:SPH*h], day_P[slh], day_Q[slh],
                                     day_Y[slh], day_V0[slh], n, gamma, s, sigma2_prev)
                pre[key][k] += q; pre[key+"_df"][k] += df
        for k, (c, f) in enumerate(zip(cands, fits)):
            Phi = phi_hist[k] if m == 0 else np.vstack((phi_hist[k], c.design(day_P[:m], day_Q[:m])))
            if frozen:
                lam = c.loss_term(theta_frozen[k], P, Q, V0).ravel() if s.use_loss_term else None
                f.frozen_sse(Phi, y, lam, offsets=s.meter_offsets, gamma=gamma)
                A = model.whiten_slots(Phi, n, gamma)
                A = model.demean_rows(A, n) if s.meter_offsets else A
                f.G = A.T@A
            else:
                lam = c.loss_term(f.theta, P, Q, V0).ravel() if s.use_loss_term else None
                f.refit(Phi, y, lam, steps=s.first_gate_fixed_point if g == 0 else 1,
                        P=P, Q=Q, v0sq=V0, use_loss=s.use_loss_term, offsets=s.meter_offsets, gamma=gamma)
        N = y.size
        J, sigma2 = scores.bic([f.sse for f in fits], [c.p+(n if s.meter_offsets else 0) for c in cands], N)
        L = int(np.argmin(J)); E = J-J[L]
        sigma2_prev = sigma2
        w = np.exp(-(J-J.min())); w /= w.sum()
        thetas = [f.theta for f in fits]
        near = [k for k in range(K) if k != L and E[k] < 25]
        eq = np.zeros(K, bool); eq[L] = True
        if near:
            vL = scores.voltages(cands[L], thetas[L], Pf, Qf, V0f, s.use_loss_term)
            for k in near:
                vk = scores.voltages(cands[k], thetas[k], Pf, Qf, V0f, s.use_loss_term)
                eq[k] = float(np.max(np.abs(vk-vL))) <= s.equivalence_pu
        ok, mass = scores.posterior_ok(E, eq, L, s.alpha)
        if arm == "check_then_t_opt" and g <= status["check_until"]:
            ok = False
        if ok and status["leader"] == L:
            status["count"] += 1
        elif ok:
            status["leader"], status["count"] = L, 1
        else:
            status["leader"], status["count"] = None, 0
        was = status["confirmed"]
        status["confirmed"] = status["count"] >= s.confirm_hours
        if status["confirmed"] and cands[L].key != inp.truth_key:
            ever_wrong = True; ever_wrong_types.add(scores.rival_type(cands[L].key, inp.truth_key))
            wrong_leaders.add(L)
        if status["confirmed"] and cands[L].key == inp.truth_key and first_correct is None:
            first_correct = g
        rivals = sorted([k for k in range(K) if k != L and not eq[k]], key=lambda k: E[k])
        Emin = float(E[rivals[0]]) if rivals else float("inf")
        # realised evidence for blocks whose data just arrived
        for entry in block_log:
            if entry["hour"] == g-1 and "E_after" not in entry:
                entry["E_after"] = {str(j): float(E[j]) for j in entry["active"]}
                entry["leader_after"] = L
        gate = dict(g=g, leader=L, leader_correct=cands[L].key == inp.truth_key, Emin=Emin, mass=float(mass),
                    ok=bool(ok), confirmed=bool(status["confirmed"]), sigma2=sigma2, n_rivals=len(rivals),
                    n_equivalent=int(eq.sum()-1),
                    adequacy=float(fits[L].sse/(N-cands[L].p-(n if s.meter_offsets else 0))
                                   /((4. if gamma else 8.)*s.voltage_sigma**2*1.02**4)))
        # evaluation only: where the truth stands, and how far the leader's voltages are from AC truth
        if truth_idx is not None:
            gate.update(truth_E=float(E[truth_idx]), truth_eq=bool(eq[truth_idx]))
        if s.prequential_check:
            ratio = lambda key: np.where(pre[key+"_df"] > 0, pre[key]/np.maximum(pre[key+"_df"], 1.), np.nan)
            rp, rq = ratio("probe"), ratio("passive")
            gate.update(pre_L_probe=float(pre["probe"][L]), pre_L_probe_df=float(pre["probe_df"][L]),
                        pre_L_pass=float(pre["passive"][L]), pre_L_pass_df=float(pre["passive_df"][L]),
                        pre_min_probe=float(np.nanmin(rp)) if np.any(np.isfinite(rp)) else None,
                        pre_min_pass=float(np.nanmin(rq)) if np.any(np.isfinite(rq)) else None,
                        pre_eq_min_probe=float(np.nanmin(np.where(eq, rp, np.nan))) if np.any(np.isfinite(rp[eq])) else None)
            if truth_idx is not None:          # evaluation only
                gate.update(pre_truth_probe=float(rp[truth_idx]) if np.isfinite(rp[truth_idx]) else None,
                            pre_truth_pass=float(rq[truth_idx]) if np.isfinite(rq[truth_idx]) else None)
        if status["confirmed"] or g == s.hours:
            vhat = scores.voltages(cands[L], thetas[L], inp.day["P"], inp.day["Q"], inp.day["root"]**2,
                                   s.use_loss_term)
            gate["vdev_true"] = float(np.max(np.abs(vhat-v_bg)))
        # --------------------------------------------------------- seal next block
        hb = g+s.gate_lead_hours
        spent_c, soc_c = committed(s, inp.res, g, hb, blocks, spent, soc)
        u = np.zeros(n); plan = None
        if hb < s.hours and g < s.hours:
            if arm == "frozen_replay" and replay is not None:
                u = np.asarray(replay["blocks"][hb], float)
            elif arm == "random_matched_P" and replay is not None:
                cost = replay["costs"][hb]
                if cost > 0:
                    u = planner.random_block(s, inp.res, hb, cost, rng_arm, soc_c)
            elif arm == "qbc_matched_P" and replay is not None:
                cost = replay["costs"][hb]
                if cost > 0 and rivals:
                    covs = [f.covariance(sigma2) for f in fits]
                    gt = planner.Gate(inp, s, cands, thetas, covs, sigma2, E, L, eq, w, (Pf, Qf, V0f),
                                      soc_c, spent_c, hb)
                    if s.meter_offsets:
                        gt.offsets = [f.offset for f in fits]
                    plan = gt.plan_heuristic(rivals, "qbc", target_cost=cost)
                    u = plan["block_kw"]
            elif probing and not ok and not status["confirmed"] and not status["stopped"] and rivals \
                    and (not s.trust_gate or arm in ("t_opt_P", "check_then_t_opt")
                         or (closure_ok and gate["adequacy"] <= s.adequacy_max)):
                if arm == "check_then_t_opt" and status["check_done"] < s.check_hours:
                    u = planner.check_block(s, inp.res, hb, soc_c, rng_chk)
                    status["check_done"] += 1; status["check_until"] = hb+1
                    plan = dict(reason="check", shape="zero_sum_check", active=[], beyond=[], reach={},
                                cost=planner.block_cost(s, inp.res, hb, u), window_empty=False)
                else:
                    covs = [f.covariance(sigma2) for f in fits]
                    gt = planner.Gate(inp, s, cands, thetas, covs, sigma2, E, L, eq, w, (Pf, Qf, V0f),
                                      soc_c, spent_c, hb)
                    if s.meter_offsets:
                        gt.offsets = [f.offset for f in fits]
                    gt.N_rows = N
                    gt.root_gamma = gamma
                    if arm == "cert_P":
                        plan = gt.plan_certified(rivals)
                    elif arm.startswith("bgreedy"):
                        plan = gt.plan_bayes_greedy(rivals)
                    elif arm in ("qbc_P", "greedy_P"):
                        plan = gt.plan_heuristic(rivals, arm[:-2])
                    else:
                        plan = gt.plan(rivals)
                    u = plan["block_kw"]
                    last_beyond = list(plan["beyond"]); last_reach = plan["reach"]
                    if plan["active"]:
                        G = min(E[j] for j in plan["active"])
                        if np.any(u != 0):
                            progress.append(G)
                            if len(progress) > s.no_progress_hours and \
                                    progress[-1]-progress[-1-s.no_progress_hours] < s.no_progress_gain:
                                status["stopped"] = True
            blocks[hb] = u
            if np.any(u != 0) or plan is not None:
                entry = dict(hour=hb, sealed_at=g, u_kw=u.tolist(), cost=planner.block_cost(s, inp.res, hb, u),
                             leader=L, E_seal={str(j): float(E[j]) for j in (plan or {}).get("active", [])},
                             active=list((plan or {}).get("active", [])), beyond=list((plan or {}).get("beyond", [])),
                             reach=(plan or {}).get("reach", {}), reason=(plan or {}).get("reason", "replay"),
                             shape=(plan or {}).get("shape"), window_empty=bool((plan or {}).get("window_empty", False)),
                             q_planned=(plan or {}).get("q_planned", {}), eta=(plan or {}).get("eta", {}),
                             parts=(plan or {}).get("parts"), kappa=(plan or {}).get("kappa", 0.),
                             cert_B0=(plan or {}).get("B0"), cert_B=(plan or {}).get("B"),
                             theta_L=thetas[L].copy(), rival_thetas={j: thetas[j].copy() for j in (plan or {}).get("active", [])})
                block_log.append(entry)
        gate["seconds"] = time.perf_counter()-t0
        gate_log.append(gate)
        if g >= s.hours:
            break
        # --------------------------------------------------------- reveal hour g
        sl = planner.hour_slots(g, SPH)
        u_kw = blocks[g]
        delivered = gains*u_kw
        probe = np.repeat((delivered/inp.base_kw)[None, :], SPH, axis=0)
        Pt, Qt, rt = inp.day["P"][sl], inp.day["Q"][sl], inp.day["root"][sl]
        v_true = physics.voltages(Pt, Qt, rt, probe)
        if physics.three_phase is not None:
            allv = physics.three_phase.all_phase_magnitudes(Pt, Qt, rt, probe)
        else:
            allv = v_true
        lo, hi = float(allv.min()), float(allv.max())
        violations["min_v"] = min(violations["min_v"], lo); violations["max_v"] = max(violations["max_v"], hi)
        violations["ac_band_hours"] += int(lo < .95 or hi > 1.05)
        violations["deadband_hours"] += int(lo < .953 or hi > 1.039)
        rng_h = np.random.default_rng(int(meter_seeds[g]))
        p_m, q_m, y_m, _, r_m = meters.read(rng_h, Pt+probe, Qt, v_true, rt)
        day_P[sl], day_Q[sl], day_Y[sl], day_V0[sl] = p_m, q_m, y_m, r_m**2
        # evaluation only: delivered power beyond what batteries and EVs can give this hour
        up, dn = sc.headroom(s, inp.res, g, soc)
        excess = np.maximum(delivered-up, 0.)+np.maximum(-delivered-dn, 0.)
        if excess.sum() > 1e-6:
            headroom_excess["hours"] += 1; headroom_excess["kwh"] += float(excess.sum())
            headroom_excess["max_kw"] = max(headroom_excess["max_kw"], float(excess.max()))
        battery = sc.split_block(s, inp.res, g, soc, delivered)
        soc = sc.update_soc(s, inp.res, soc, battery)
        if np.any(u_kw != 0):
            spent += planner.block_cost(s, inp.res, g, u_kw)
            root_kwh += float(delivered.sum())
            entry = next((e for e in block_log if e["hour"] == g), None)
            if entry is not None and entry["active"]:
                entry["bias"] = _bias_ratio(cands, entry, Pt, Qt, rt, probe, v_true, physics, inp, s)
    final = gate_log[-1]
    Lf = final["leader"]
    if status["confirmed"]:
        outcome = "correct" if cands[Lf].key == inp.truth_key else "wrong"
    elif status["stopped"]:
        outcome = "unresolved_no_progress"
    elif probing and any(E_k < s.ln19 for E_k in [E[j] for j in last_beyond]):
        outcome = "unresolved_beyond_reach"
    else:
        outcome = "undecided"
    rec = dict(input=inp.name, case=inp.case, stratum=inp.stratum, arm=arm, path=int(path), outcome=outcome,
               final_leader_correct=bool(final["leader_correct"]),
               final_type=scores.rival_type(cands[Lf].key, inp.truth_key),
               ever_confirmed_wrong=bool(ever_wrong), ever_wrong_types=sorted(ever_wrong_types),
               wrong_vdev=max((g_["vdev_true"] for g_ in gate_log if g_.get("confirmed") and not g_["leader_correct"]
                               and "vdev_true" in g_), default=None),
               final_vdev=final.get("vdev_true"), wrong_leaders=sorted(int(k) for k in wrong_leaders),
               first_correct_gate=first_correct, probed_hours=int(np.sum(np.any(blocks != 0, axis=1))),
               cost_cny=float(spent), root_kwh=float(root_kwh), violations=violations,
               headroom_excess=headroom_excess,
               beyond_at_last_gate=[dict(type=scores.rival_type(cands[j].key, inp.truth_key), E=float(E[j]),
                                         reach=last_reach.get(int(j))) for j in last_beyond],
               gates=gate_log, blocks=[_clean(e, cands, inp) for e in block_log],
               seconds=time.perf_counter()-started, K=K, truth_in_bank=truth_idx is not None,
               exact_truth=None if "exact_truth_key" not in inp.meta else dict(
                   in_bank=bool(inp.meta.get("exact_truth_in_bank")),
                   final_confirmed=bool(status["confirmed"] and cands[Lf].key == inp.meta["exact_truth_key"]),
                   ever_confirmed=any(g_["confirmed"] and cands[g_["leader"]].key == inp.meta["exact_truth_key"]
                                      for g_ in gate_log)),
               refit_failures=int(sum(getattr(f, "failures", 0) for f in fits)),
               schedule=dict(blocks=blocks.tolist(),
                             costs=[planner.block_cost(s, inp.res, h, blocks[h]) for h in range(s.hours)]))
    return rec


def _prequential(c, f, phi_hist_k, P_past, Q_past, Pn, Qn, Yn, V0n, n, gamma, s, sigma2):
    """Standardized one-hour-ahead prediction error of one candidate (evaluation only).

    The candidate as fitted on the data before the new hour predicts the new hour's rows. The whitened
    prediction error r has covariance sigma2 (I + A Sigma A') under the candidate, where A is the new design
    (demeaned against the past means when meters carry offsets) and Sigma = G^-1 on the free coordinates.
    Returns r' (I + A G^-1 A')^-1 r / sigma2 and its degrees of freedom (the number of rows).
    """
    Phi_n = c.design(Pn, Qn)
    lam = c.loss_term(f.theta, Pn, Qn, V0n).ravel() if s.use_loss_term else 0.
    r = Yn.ravel()-lam-Phi_n@f.theta-np.tile(f.offset, Pn.shape[0])
    if s.meter_offsets:
        Phi_p = phi_hist_k if P_past.shape[0] == 0 else np.vstack((phi_hist_k, c.design(P_past, Q_past)))
        p = Phi_n.shape[1]
        Phi_n = (Phi_n.reshape(-1, n, p)-Phi_p.reshape(-1, n, p).mean(axis=0, keepdims=True)).reshape(-1, p)
    A = model.whiten_slots(Phi_n, n, gamma)
    rw = model.whiten_slots(r[:, None], n, gamma).ravel()
    M = np.eye(len(rw))+A@f.covariance(1.)@A.T
    return float(rw@np.linalg.solve(M, rw))/sigma2, len(rw)


def _bias_ratio(cands, entry, Pt, Qt, rt, probe, v_true, physics, inp, s):
    """AC bias ratio b/J (PROTOCOL section 8), evaluation only."""
    y_with = rt[:, None]**2-v_true**2
    v_wo = physics.voltages(Pt, Qt, rt, None)
    y_wo = rt[:, None]**2-v_wo**2
    L = entry["leader"]; cL = cands[L]
    out = {}
    for loss in (True, False):
        yL_w = cL.predict(entry["theta_L"], Pt+probe, Qt, rt**2, loss)
        yL_o = cL.predict(entry["theta_L"], Pt, Qt, rt**2, loss)
        for j in entry["active"]:
            cj = cands[j]; th = entry["rival_thetas"][j]
            yj_w = cj.predict(th, Pt+probe, Qt, rt**2, loss)
            yj_o = cj.predict(th, Pt, Qt, rt**2, loss)
            e = (y_with-yL_w).ravel(); d = (yL_w-yj_w).ravel()
            ep = ((y_with-y_wo)-(yL_w-yL_o)).ravel(); dp = ((yL_w-yj_w)-(yL_o-yj_o)).ravel()
            whole = 2*float(e@d)/max(float(d@d), 1e-30)
            part = 2*float(ep@dp)/max(float(dp@dp), 1e-30)
            out["%d_%s" % (j, "loss" if loss else "linear")] = dict(
                whole=whole, probe=part, type=scores.rival_type(cj.key, inp.truth_key))
    return out


def _clean(entry, cands, inp):
    e = {k: v for k, v in entry.items() if k not in ("theta_L", "rival_thetas")}
    e["active_types"] = [scores.rival_type(cands[j].key, inp.truth_key) for j in entry["active"]]
    e["beyond_types"] = [scores.rival_type(cands[j].key, inp.truth_key) for j in entry["beyond"]]
    return e
