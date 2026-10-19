"""Exact AC global ambiguity despite locally full-rank stem information.

The source remains at physical node 1. Terminal meters include EV once.
This is an identifiability witness, not an optimized/global-MLE claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


CHANNELS = ("p", "q", "v", "master_p", "master_q")


def reflect_stem(theta, kappa):
    theta = np.asarray(theta, float)
    if theta.shape != (2,) or not np.isfinite(theta).all() or not np.isfinite(kappa):
        raise ValueError("theta must be two finite stem coordinates, kappa finite")
    a = np.array([1., kappa])
    reflection = 2 * np.outer(a, a) / (a @ a) - np.eye(2)
    return reflection @ theta


def stem_voltage_squared(theta, master_p, master_q, root_v=1.):
    """Exact squared magnitude when root sending-end complex power is given."""
    r, x = np.asarray(theta, float)
    p, q, v0 = np.broadcast_arrays(np.asarray(master_p, float), np.asarray(master_q, float), np.asarray(root_v, float))
    if not np.isfinite(np.r_[p.ravel(), q.ravel(), v0.ravel()]).all() or np.any(v0 <= 0):
        raise ValueError("finite root powers and positive known root magnitude required")
    return v0**2 - 2 * (r*p + x*q) + (r*r + x*x) * (p*p + q*q) / v0**2


def stem_invariant_jacobian(theta, kappa):
    r, x = np.asarray(theta, float)
    return np.array([[1., kappa], [2*r, 2*x]])


def _apis(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    if not (root / "research_experiments/mobile_ev_mle/likelihood.py").is_file():
        raise ValueError("provide the Topo repository root for a staging execution")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from research_experiments.mobile_ev_mle.candidates import RadialCandidate
    from research_experiments.mobile_ev_mle.likelihood import ac_predict
    from research_experiments.mobile_ev_mle.simulation import model
    from research_experiments.mobile_ev_mle.symmetric_ac_local import mean_and_jacobian, historical_profile_geometry
    return root, RadialCandidate, ac_predict, model, mean_and_jacobian, historical_profile_geometry


def construct_example(topology="star", *, root=None, points=20, history_points=12,
                      kappa=1/3, with_ev=True, varying_pf=False):
    root, Candidate, ac_predict, model, mean_and_jacobian, profile = _apis(root)
    if topology not in ("star", "cherry_01"):
        raise ValueError("counterexample topology must be star or cherry_01")
    if not 2 <= history_points < points:
        raise ValueError("need at least two historical points and a future block")
    candidate = Candidate.from_supports(topology, 3, () if topology == "star" else ((0, 1),))
    time = np.arange(points, dtype=float)
    base_p = .09 + .01*np.cos(.8*time[:, None] + np.arange(3))
    base_q = .03 + .004*np.sin(.4*time[:, None] + np.arange(3))
    ev_kw = np.zeros_like(base_p)
    if with_ev:
        ev_kw[history_points:, 0] = 3.
    p, q = base_p + ev_kw/20., base_q.copy()
    master_p = .58 + .10*np.sin(.5*time)
    master_q = kappa*master_p + (.04*np.cos(.31*time) if varying_pf else 0.)
    master = master_p + 1j*master_q
    current_squared = abs(master)**2  # known root_v=1 in this witness
    theta = np.array([.005, .0035])
    reflected = reflect_stem(theta, kappa)
    r, x = np.full(candidate.m, .012), np.full(candidate.m, .006)
    r[0], x[0] = theta
    reflected_r, reflected_x = r.copy(), x.copy()
    reflected_r[0], reflected_x[0] = reflected
    z, zp = complex(*theta), complex(*reflected)
    v1_complex = 1. - z*np.conj(master)
    v1_magnitude = abs(v1_complex)
    auxiliary_r, auxiliary_x = r.copy(), x.copy()
    auxiliary_r[0] = auxiliary_x[0] = 0.  # computation of downstream demand only
    downstream = ac_predict(candidate, auxiliary_r, auxiliary_x, p, q, 0., 0., v1_magnitude)
    downstream_power = downstream["master_p"] + 1j*downstream["master_q"]
    source = master - z*current_squared - downstream_power
    reflected_source = source - (zp-z)*current_squared
    first = ac_predict(candidate, r, x, p, q, source.real, source.imag, 1.)
    second = ac_predict(candidate, reflected_r, reflected_x, p, q,
                        reflected_source.real, reflected_source.imag, 1.)
    meter = model(3)
    geometries, historical_geometries, means = [], [], []
    for rr, xx, ss in ((r, x, source), (reflected_r, reflected_x, reflected_source)):
        response = mean_and_jacobian(candidate, rr, xx, p, q, ss.real, ss.imag, 1., meter)
        stem_columns = response["line_jacobian"][:, [0, candidate.m]]
        geometry = profile(stem_columns, response["state_jacobian"], rcond=1e-12)
        singular = np.linalg.svd(geometry["factor"], compute_uv=False)
        geometries.append(dict(stem_column_indices=[0, candidate.m],
            observation_scope="joint history and future",
            profiled_shared_line_parameter_count=2,is_full_line_beta_profile=False,
            all_other_line_parameters_fixed=True, nuisance_columns=response["state_jacobian"].shape[1],
            nuisance_order="timewise terminal P/Q and node1 source P/Q",
            projected_stem_rank=geometry["projected_line_rank"], projected_stem_singular_values=singular,
            projected_precision=geometry["precision"], projection_rcond=1e-12,
            local_profile_rank_is_global_identifiability=False))
        h=history_points
        historical=mean_and_jacobian(candidate,rr,xx,p[:h],q[:h],ss.real[:h],ss.imag[:h],1.,meter)
        historical_geometry=profile(historical['line_jacobian'][:,[0,candidate.m]],
                                    historical['state_jacobian'],rcond=1e-12)
        historical_geometries.append(dict(observation_scope="history only",stem_column_indices=[0,candidate.m],
            profiled_shared_line_parameter_count=2,is_full_line_beta_profile=False,
            all_other_line_parameters_fixed=True,nuisance_columns=historical['state_jacobian'].shape[1],
            projected_stem_rank=historical_geometry['projected_line_rank'],
            projected_stem_singular_values=np.linalg.svd(historical_geometry['factor'],compute_uv=False),
            projected_precision=historical_geometry['precision'],projection_rcond=1e-12,
            local_profile_rank_is_global_identifiability=False))
        means.append(response["mean"])
    return dict(root=root, candidate=candidate, meter=meter, topology=topology,
        points=points, history_points=history_points, time=time, kappa=kappa,
        base_p=base_p, base_q=base_q, ev_kw=ev_kw, p_total=p, q_total=q,
        theta=theta, reflected_theta=reflected, r=r, x=x,
        reflected_r=reflected_r, reflected_x=reflected_x,
        target_master=master, source=source, reflected_source=reflected_source,
        source_location=1, first=first, second=second, downstream=downstream,
        target_v1_complex=v1_complex, whitened_means=means, geometries=geometries,
        historical_geometries=historical_geometries,
        with_ev=with_ev, varying_pf=varying_pf)


def gaussian_point_comparison(example, *, noise_seed=981001):
    """Nonzero noise, two feasible parameter-state point likelihoods only."""
    means = example["whitened_means"]
    noise = np.random.default_rng(noise_seed).normal(size=len(means[0]))
    observed_whitened = means[0] + noise
    losses = [float(np.sum((observed_whitened-mean)**2)) for mean in means]
    meter, first = example["meter"], example["first"]
    normalizer = 0.
    for channel in CHANNELS:
        value = np.asarray(first[channel])
        sigma = np.broadcast_to(np.asarray(getattr(meter, channel+"_sigma"), float), value.shape)
        normalizer += float(np.log(2*np.pi*sigma**2).sum())
    logliks = [-.5*(normalizer+loss) for loss in losses]
    return dict(noise_seed=noise_seed, nonzero_measurement_noise=True,
        whitened_noise_squared_norm=float(noise@noise), loss_at_feasible_points=losses,
        loglikelihood_at_feasible_points=logliks, loglik_absolute_difference=abs(logliks[0]-logliks[1]),
        noise_count=len(noise), independent_known_Gaussian_covariance=True,
        parameters_optimized=False, either_point_claimed_global_MLE=False)


def summarize_example(example):
    a, b = example["first"], example["second"]
    source, sourcep = example["source"], example["reflected_source"]
    channels = {channel: float(np.max(abs(np.asarray(a[channel])-np.asarray(b[channel])))) for channel in CHANNELS}
    line_bounds = bool(all(np.all((value >= 1e-6) & (value <= .1)) for value in (
        example["r"], example["x"], example["reflected_r"], example["reflected_x"])))
    source_bounds = bool(all(np.all((value >= -.5) & (value <= .5)) for value in (
        source.real, source.imag, sourcep.real, sourcep.imag)))
    master_actual = a["master_p"]+1j*a["master_q"]
    v1_squared = abs(example["target_v1_complex"])**2
    analytic = stem_voltage_squared(example["theta"], example["target_master"].real,
        example["target_master"].imag)
    local_lower=np.maximum(1e-6,.7*example['theta'])
    local_upper=np.minimum(.1,1.3*example['theta'])
    data = dict(topology=example["topology"], source_node=1, root_voltage=1.,
        history_points=example["history_points"], future_points=example["points"]-example["history_points"],
        line_parameters_shared_for_all_history_and_future=True,
        theta=example["theta"], reflected_theta=example["reflected_theta"],
        original_center_30pct_stem_trust_lower=local_lower,
        original_center_30pct_stem_trust_upper=local_upper,
        reflected_stem_inside_single_30pct_trust=bool(np.all((example['reflected_theta']>=local_lower)&
            (example['reflected_theta']<=local_upper))),
        root_X_relative_change=float(abs(example['reflected_theta'][1]-example['theta'][1])/example['theta'][1]),
        r=example["r"], x=example["x"], reflected_r=example["reflected_r"], reflected_x=example["reflected_x"],
        fixed_master_Q_over_P=example["kappa"] if not example["varying_pf"] else None,
        source_P_ranges=[[float(value.real.min()), float(value.real.max())] for value in (source, sourcep)],
        source_Q_ranges=[[float(value.imag.min()), float(value.imag.max())] for value in (source, sourcep)],
        source_pointwise_change_max=float(np.max(abs(sourcep-source))),
        source_P_history=source.real[:example["history_points"]], source_P_future=source.real[example["history_points"]:],
        source_Q_history=source.imag[:example["history_points"]], source_Q_future=source.imag[example["history_points"]:],
        reflected_source_P=sourcep.real, reflected_source_Q=sourcep.imag,
        final_parameters_within_declared_line_bounds=line_bounds,
        both_source_paths_within_declared_bounds=source_bounds,
        voltage_min=min(float(a["v"].min()), float(b["v"].min())),
        voltage_max=max(float(a["v"].max()), float(b["v"].max())),
        channel_mean_max_absolute_differences=channels,
        max_whitened_mean_difference=float(np.max(abs(example["whitened_means"][0]-example["whitened_means"][1]))),
        analytic_voltage_identity_error=float(np.max(abs(v1_squared-analytic))),
        original_target_master_error=float(np.max(abs(master_actual-example["target_master"]))),
        max_AC_equation_residual=max(a["ac_equation_residual"],b["ac_equation_residual"]),
        max_power_balance_residual=max(a["power_balance_residual"],b["power_balance_residual"]),
        all_node_complex_voltage_difference=float(np.max(abs(a["voltage"]-b["voltage"]))),
        profile_geometries=example["geometries"],
        history_only_profile_geometries=example["historical_geometries"],
        invariant_map_determinants=[float(np.linalg.det(stem_invariant_jacobian(theta,example["kappa"])))
            for theta in (example["theta"],example["reflected_theta"])],
        EV_added_exactly_once=bool(np.array_equal(example["p_total"],example["base_p"]+example["ev_kw"]/20.)),
        EV_future_grid_energy_kwh=float(example["ev_kw"].sum()*.25),
        gaussian_point_comparison=gaussian_point_comparison(example))
    return data


def _finite(value):
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, np.generic):
        return _finite(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, dict):
        return {key: _finite(item) for key,item in value.items()}
    return value


def run_counterexamples(out, *, root=None):
    root, *_ = _apis(root)
    names = ["research_experiments/mobile_ev_mle/likelihood.py",
             "research_experiments/mobile_ev_mle/candidates.py",
             "research_experiments/mobile_ev_mle/symmetric_ac_local.py",
             "research_experiments/mobile_ev_mle/simulation.py",
             "research_experiments/mobile_ev_probing/ac_general_v9.py"]
    hashes = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    fixed = [summarize_example(construct_example(topology,root=root)) for topology in ("star","cherry_01")]
    varying = summarize_example(construct_example("cherry_01",root=root,varying_pf=True))
    if hashes != {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}:
        raise ValueError("counterexample source changed during validation")
    result = dict(exact_AC_identifiability_counterexample=True,
        scope="two line/source explanations of one fixed topology under fixed sending-end power factor",
        topology_indistinguishability_proved=False, every_EV_action_ineffective_claim=False,
        either_point_global_MLE_claim=False, physical_source_interface_fixed_node1=True,
        proper_source_process_prior_not_included=True,
        source_sha256=hashes, audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixed_PF_examples=fixed, varying_PF_reflection_failure=varying)
    out = Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/"ac_reflection_counterexample.json").write_text(json.dumps(_finite(result),ensure_ascii=False,
        indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    result=run_counterexamples(args.out,root=args.root)
    print(json.dumps(dict(fixed_PF_examples=len(result['fixed_PF_examples']),
        profile_ranks=[x['profile_geometries'][0]['projected_stem_rank'] for x in result['fixed_PF_examples']],
        global_MLE_claim=False)))
