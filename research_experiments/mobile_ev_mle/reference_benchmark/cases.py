"""Reference-network simulation with observed-only planning inputs.

The generator owns future background, true lines and source realizations.  The
planner receives arrived P/Q/V and public meter/device quantities only.  Every
future EV action is evaluated with complete nonlinear AC and paired meters.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

from .. import likelihood as lk
from ..ev_original_study import observation
from .networks import build_network, _finalize, CIGRE_URL, CIGRE_SHA256
from .representation import encode_physical_tree, physical_tree_ac, public_fit_options


CHANNELS = ("p", "q", "v", "master_p", "master_q")


def _seed(seed, channel):
    return int(np.random.SeedSequence([int(seed), int(channel)]).generate_state(1)[0])


def _paper_transformer(network):
    """Explicit extension: original lines unchanged, declared physical root Z.

    The per-unit transformer R and |Z| follow the CIGRE reference, while its
    0.2 MVA rating is a declared scenario choice, not an original paper asset.
    """
    n = deepcopy(network)
    old_root = n["root_bus"]
    new_root = min(b["bus_id"] for b in n["buses"]) - 1
    tr = dict(sn_mva=n["base_mva"], vn_hv_kv=20., vn_lv_kv=n["base_kv"],
              vkr_percent=1., vk_percent=4.123106, pfe_kw=0.,
              i0_percent=0., shift_degree=30., tap_pos=0.)
    zbase = tr["vn_lv_kv"] ** 2 / tr["sn_mva"]
    r = .01 * zbase
    x = math.sqrt((tr["vk_percent"] / 100.) ** 2 - .01 ** 2) * zbase
    n["buses"].insert(0, dict(bus_id=new_root, pd_kw=0., qd_kvar=0.,
        nominal_kv=n["base_kv"], original_nominal_kv=20., source_label="declared_transformer_primary_referred_to_LV"))
    branch_id = max(b["branch_id"] for b in n["branches"]) + 1
    n["branches"].insert(0, dict(branch_id=branch_id, from_bus=new_root,
        to_bus=old_root, r_ohm=r, x_ohm=x, status=True,
        branch_type="declared_cigre_type_transformer_lv_equivalent", length_km=None,
        rating_a=tr["sn_mva"] * 1000. / (math.sqrt(3.) * tr["vn_lv_kv"]), transformer=tr))
    n["root_bus"] = new_root
    n["source_bus_suggestion"] = old_root
    n["provenance"]["modifications"] += [
        "add explicitly declared CIGRE-type transformer upstream of old root; retain old root as fixed source interface"]
    n["provenance"]["added_transformer"] = dict(**tr, source_url=CIGRE_URL,
        source_sha256=CIGRE_SHA256, rating_is_scenario_choice=True,
        original_paper_asset=False, r_ohm_lv=r, x_ohm_lv=x)
    n["metadata"]["upstream_common_positive_line_added"] = True
    n["metadata"]["added_transformer"] = deepcopy(tr)
    n["metadata"]["original_physical_line_impedances_unchanged"] = True
    return _finalize(n)


def native_network(name):
    n = build_network(name)
    return _paper_transformer(n) if n["name"] in ("paper11", "paper15") else n


def _window_arrays(window, prepared, name, observation_ids, h, t):
    stamps = list(window["timestamps"])
    p = np.asarray(window["p_relative"], float)
    q = np.asarray(window["q_relative"], float)
    pv = np.asarray(window["pv_relative"], float)
    profile_names = list(prepared["profile_names"])
    if len(profile_names) != len(set(profile_names)):
        raise ValueError("profile names must be unique")
    expected = (h + t, len(profile_names))
    if len(stamps) != h + t or p.shape != expected or q.shape != expected or pv.shape != (h + t,):
        raise ValueError("prepared window dimensions must match H+T and the frozen profile list")
    if not all(np.isfinite(a).all() for a in (p, q, pv)):
        raise ValueError("finite signed net P/Q and PV factors required")
    if np.any(pv < 0) or np.any(pv > 1 + 1e-12):
        raise ValueError("PV capacity factor must lie in [0,1]")
    if int(window.get("history_rows", h)) != h or int(window.get("horizon_rows", t)) != t:
        raise ValueError("window split differs from frozen H/T")
    assignments = prepared["assignments"][name]
    chosen = []
    for bus in observation_ids:
        profile = assignments.get(str(bus), assignments.get(bus))
        if profile not in profile_names:
            raise ValueError(f"no frozen profile assignment for {name} physical bus {bus}")
        chosen.append(profile_names.index(profile))
    return p[:, chosen].copy(), q[:, chosen].copy(), pv.copy(), stamps, chosen


def meter_for_nominal(p0_kw, q0_kvar, base_kw):
    p0, q0 = np.asarray(p0_kw, float), np.asarray(q0_kvar, float)
    return lk.MeasurementModel(
        p_sigma=.002 * np.maximum(p0, 1.) / base_kw,
        q_sigma=.002 * np.maximum.reduce((abs(q0), .2 * p0, np.ones_like(p0))) / base_kw,
        v_sigma=.0005,
        master_p_sigma=.001 * max(float(p0.sum()), 20.) / base_kw,
        master_q_sigma=.001 * max(float(abs(q0).sum()), .2 * float(p0.sum()), 20.) / base_kw,
        bias_p=0., bias_q=0., bias_v=0., fit_bias=False)


def _fit_domain(network, source_cap_kw, config):
    # Only the public asset multiset is used. No descendant/path maximum or
    # candidate-specific true line coordinate enters this domain.
    r_total = float(sum(b["r_ohm"] for b in network["branches"]))
    x_total = float(sum(b["x_ohm"] for b in network["branches"]))
    ratio = tuple(config.get("fit_ratio_bounds", (.05, 5.)))
    cfg = config.get("fit", {})
    options = public_fit_options(base_kv=network["base_kv"], base_kw=network["base_kw"],
        r_ohm_upper=r_total, x_ohm_upper=x_total, max_series_edges=1,
        ratio_bounds=ratio, source_p_kw_bound=1.1 * source_cap_kw,
        source_q_kvar_bound=.4 * source_cap_kw, voltage_bounds=(.8, 1.2),
        multistart=config.get("fit_multistart", cfg.get("multistart", 2)),
        max_nfev=config.get("fit_max_nfev", cfg.get("max_nfev", 250)),
        tol=config.get("fit_tol", cfg.get("tol", 1e-8)))
    # This is an audit only, not a tightening rule. Every positive source-asset
    # ratio and every exact virtual zero is inside the public common cone.
    for b in network["branches"]:
        r, x = b["r_ohm"], b["x_ohm"]
        if r and not ratio[0] * r <= x <= ratio[1] * r:
            raise ValueError("declared public ratio domain excludes a supplied physical asset")
    return options, dict(rule="sum of complete public asset inventory, not any true path",
        r_ohm_upper=r_total, x_ohm_upper=x_total, r_x_lower_ohm=0.,
        ratio_bounds=list(ratio), actual_virtual_edge_pattern_used=False,
        source_p_bounds_kw=[-1.1 * source_cap_kw, 1.1 * source_cap_kw],
        source_q_bounds_kvar=[-.4 * source_cap_kw, .4 * source_cap_kw],
        source_mode_used=False)


def _physical_audit(network, encoded, p_kw, q_kvar, sp_kw, sq_kvar, root, physical):
    oracle = physical_tree_ac(network["branches"], network["observation_bus_ids"],
        network["root_bus"], network["base_kv"], network["base_kw"], p_kw, q_kvar,
        encoded.source_bus, sp_kw, sq_kvar, root)
    errors = {"v": float(np.max(abs(oracle["v"] - physical["v"]))),
              "master_p": float(np.max(abs(oracle["master_p"] - physical["master_p"]))),
              "master_q": float(np.max(abs(oracle["master_q"] - physical["master_q"])))}
    if max(errors.values()) > 2e-10:
        raise RuntimeError(f"physical-tree versus encoded AC discrepancy: {errors}")
    return dict(independent_physical_tree_oracle=True, rows=len(p_kw),
        voltage_max_error_pu=errors["v"], master_p_max_error_pu=errors["master_p"],
        master_q_max_error_pu=errors["master_q"],
        oracle_ac_residual=oracle["ac_equation_residual"],
        oracle_power_balance_residual=oracle["power_balance_residual"],
        complete_original_injections_retained=True, source_bus=encoded.source_bus,
        passed=True, supplied_to_selector=False)


def make_case(spec, window, prepared, config):
    """Build one native reference-network realization; no outcome filtering."""
    h, t = int(config["H"]), int(config["T"])
    if h < 1 or t < 1:
        raise ValueError("positive H and T required")
    name = spec["network"]
    network = native_network(name)
    ids = network["observation_bus_ids"]
    by_bus = {b["bus_id"]: b for b in network["buses"]}
    p0 = np.array([by_bus[b]["pd_kw"] for b in ids], float)
    q0 = np.array([by_bus[b]["qd_kvar"] for b in ids], float)
    pf, qf, pv, stamps, chosen = _window_arrays(window, prepared, name, ids, h, t)
    p_kw, q_kvar = pf * p0, qf * q0
    cap = min(.3 * float(p0.sum()), 80.)
    if cap <= 0:
        raise ValueError("positive public reference load needed for source scale")
    mode = spec["source_mode"]
    if mode in ("PV", "DER"):
        sp_kw = -cap * pv
    elif mode == "none":
        sp_kw = np.zeros(h + t)
    else:
        raise ValueError("source_mode must be registered none or PV")
    sq_kvar = .2 * sp_kw
    base = network["base_kw"]
    p, q, sp, sq = p_kw / base, q_kvar / base, sp_kw / base, sq_kvar / base
    root = np.ones(h + t)
    encoded = encode_physical_tree(network["branches"], ids, network["root_bus"],
        network["base_kv"], base, network["source_bus"], name="simulator_truth")
    physical = lk.ac_predict(encoded.candidate, encoded.r, encoded.x, p, q, sp, sq, root)
    historical = {k: (v[:h] if isinstance(v, np.ndarray) and v.shape[0] == h + t else v)
                  for k, v in physical.items()}
    hm = meter_for_nominal(p0, q0, base)
    # Independent named streams preserve the same history noise across methods
    # and paired source conditions whenever physical_seed is shared by driver.
    seed = int(spec.get("physical_seed", spec["seed"]))
    white = {k: np.random.default_rng(_seed(seed, 100 + j)).standard_normal(
        (h, len(ids)) if k in ("p", "q", "v") else h) for j, k in enumerate(CHANNELS)}
    history = observation(historical, hm, white)
    options, domain = _fit_domain(network, cap, config)
    audit = _physical_audit(network, encoded, p_kw[:h], q_kvar[:h], sp_kw[:h],
                            sq_kvar[:h], root[:h], historical)
    # Future meters have exactly the same absolute standard deviations. No
    # repeating-count bonus, 30/T precision boost, or future-noise scaling.
    fm = deepcopy(hm)
    operating = (.9, 1.1)
    return dict(spec=deepcopy(spec), history=history, history_meter=hm, future_meter=fm,
        timestamps=stamps[:h], future_timestamps=stamps[h:],
        p=p, q=q, source_p=sp, source_q=sq, root_v=root,
        truth=dict(candidate=encoded.candidate, r=encoded.r, x=encoded.x),
        network=network, encoded_tree=encoded, base_kw=base, base_kv=network["base_kv"],
        H=h, T=t, dt_hours=float(config.get("dt_hours", prepared.get("config", {}).get("dt_hours", .25))),
        nominal_p_kw=p0, nominal_q_kvar=q0,
        observation_bus_ids=tuple(ids), fit_options=options,
        physics_audit=dict(history=audit),
        provenance=dict(network=deepcopy(network["provenance"]),
            physical_tree_sha256=network["physical_sha256"],
            physical_representation=encoded.record(),
            input_window=dict(index=window.get("index"), phase=window.get("phase"),
                forecast_cutoff_utc=window.get("forecast_cutoff_utc"),
                relative_profile_names=list(prepared["profile_names"]),
                assigned_profile_indices=chosen, transformation="original nominal P0/Q0 times frozen relative P/Q profile",
                profile_provenance=deepcopy(prepared.get("data_provenance", {}))),
            scaling=dict(history_normalization=False, future_normalization=False,
                         preserved_original_native_load_scale=True),
            source=dict(mode=mode, physical_bus=encoded.source_bus, P_sign="positive_consumption",
                active_capacity_kw=cap, actual_Q_to_P=.2,
                model_Q_to_P_fixed=False, both_modes_share_bounds=True,
                apparent_power_box_envelope_kva=float(np.hypot(1.1 * cap, .4 * cap)),
                added_unmetered_DER=True,
                load_meters_exclude_separate_interface_DER=True,
                direct_source_telemetry=False, PV_driver="prepared public PV capacity-factor profile"),
            meter=dict(voltage_sigma_pu=.0005, history_future_same_sigma=True,
                no_repeat_precision_bonus=True, fixed_calibrated_bias=0.,
                terminal_P_sigma_kw=(np.asarray(hm.p_sigma) * base).tolist(),
                terminal_Q_sigma_kvar=(np.asarray(hm.q_sigma) * base).tolist(),
                master_P_sigma_kw=float(hm.master_p_sigma * base),
                master_Q_sigma_kvar=float(hm.master_q_sigma * base),
                assumptions="independent additive Gaussian channel/time errors; nominal-scaled absolute sigma"),
            fit_domain=domain, calculation_voltage_bounds=[.8, 1.2],
            operational_voltage_bounds=list(operating),
            actual_initial_voltage_range=[float(physical["v"].min()), float(physical["v"].max())],
            actual_background_operational_voltage_ok=bool(
                np.min(physical["v"]) >= operating[0] and np.max(physical["v"]) <= operating[1]),
            source_Q_included=True, EV_Q=0., EV_added_once=True,
            known_root=True, known_calibrated_bias=True, EV_tracking_exact=True,
            future_available_to_selector=False))


def planning_inputs(case):
    """Strict boundary: arrived measurements and common public meter metadata."""
    return dict(history=deepcopy(case["history"]),
        history_meter=deepcopy(case["history_meter"]), future_meter=deepcopy(case["future_meter"]),
        timestamps=list(case["timestamps"]), horizon=case["T"],
        base_kw=case["base_kw"], public_parameters=dict(
            base_kv=case["base_kv"], base_kw=case["base_kw"],
            observation_bus_ids=list(case["observation_bus_ids"]),
            units="three-phase total kW/kvar; line-line kV; per-unit metering",
            site_interpretation="existing measured physical load bus; MV ports are LV stations referred through service transformer"))


def actual_observation(case, power_kw, white, *, clean=False):
    """Execute one plan against future truth, then apply supplied paired noise.

    Return (meter observations, full AC state, latent background without EV),
    matching full_pipeline.cases. The independent oracle is audit data only.
    """
    h, t = case["H"], case["T"]
    power = np.asarray(power_kw, float)
    n = len(case["observation_bus_ids"])
    if power.shape != (t, n) or not np.isfinite(power).all():
        raise ValueError("EV power must be a finite future-time by observed-bus kW matrix")
    if set(white) != set(CHANNELS):
        raise ValueError("paired white errors required for all five channels")
    for k in CHANNELS:
        shape = (t, n) if k in ("p", "q", "v") else (t,)
        if np.asarray(white[k]).shape != shape or not np.isfinite(white[k]).all():
            raise ValueError(f"invalid paired future noise shape for {k}")
    truth = case["truth"]
    p = case["p"][h:h+t] + power / case["base_kw"]
    q = case["q"][h:h+t]
    sp, sq, root = (case[k][h:h+t] for k in ("source_p", "source_q", "root_v"))
    physical = lk.ac_predict(truth["candidate"], truth["r"], truth["x"], p, q, sp, sq, root)
    audit = _physical_audit(case["network"], case["encoded_tree"],
        p * case["base_kw"], q * case["base_kw"], sp * case["base_kw"],
        sq * case["base_kw"], root, physical)
    audit.update(EV_energy_kwh_by_bus=(power.sum(axis=0) * case["dt_hours"]).tolist(), EV_Q_kvar=0.,
        operating_voltage_ok=bool(physical["v"].min() >= .9 and physical["v"].max() <= 1.1))
    physical["reference_audit"] = audit
    noise = {k: np.zeros_like(np.asarray(v)) for k, v in white.items()} if clean else white
    observed = observation(physical, case["future_meter"], noise)
    return observed, physical, dict(p=case["p"][h:h+t].copy(), q=case["q"][h:h+t].copy(),
        source_p=sp.copy(), source_q=sq.copy(), root_v=root.copy())
