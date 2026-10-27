"""Generate physical days once, then freeze measurements and source closure."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import shutil
import sys

from .common import CASES, CONDITIONS, METHODS, SIZES, sha, write

ROOT = Path(__file__).resolve().parents[2]


def prepare(output):
    import numpy as np
    from research_experiments.rnj.prepare_source_ac288_runtime import verify_runtime
    runtime = ROOT/"outputs/source_ac288_runtime_20261018"
    verify_runtime(runtime)
    sys.path.insert(0, str(runtime))
    from rnj_wzzt.scenario import simulation as sim
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output/"source"
    source.mkdir()
    shutil.copytree(runtime/"rnj_wzzt", source/"rnj_wzzt", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (source/"research_experiments").mkdir()
    (source/"research_experiments/__init__.py").write_text("", encoding="utf-8")
    for name in ("wzzt_systematic", "wzzt_solver_tuning", "rnj"):
        shutil.copytree(ROOT/"research_experiments"/name, source/"research_experiments"/name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        cases=CASES, sample_sizes=SIZES, methods=METHODS, conditions=CONDITIONS,
        train_replicate=0, test_replicate=1, physical_days_per_split=3, physical_points_per_day=96,
        training_sampling="nested every 12/6/3/1 physical points on each day; total N=24/48/96/288",
        profiles=["default", "cloudy_variable_pv", "evening_peak_high_load"],
        train_profile_seeds=[42, 7, 21], test_seed_offset=100003,
        root_mean=1.02, root_sigma=.003, impedance_scale=1., source_present=False,
        low_xr="R fixed; X/R=.30 backbone and .06 service; controlled ablation",
        bootstrap=dict(replicates=100, confidence=.75, max_groups=2, min_group_size=2,
                       max_subclade_frequency=.15, fine_tolerance_ratio=.5,
                       block_length="max(1, round(points_per_day/24)); approximately one physical hour"),
        solve=dict(per_solve_seconds=60., search_wall_seconds=180., hard_worker_seconds=600.,
                   relative_gain=.001, absolute_gain=1e-10, uncertainty_fraction=.5, requested_gap=.0001,
                   initial_weight_bound=2., max_bound_expansions=4, bound_active_fraction=.995,
                   forward_step_limit="4*n", family_limit="2*n-1", zero_prune_tolerance=1e-9),
        ac_selection="minimum training voltage MSE plus one SE over 3 day blocks; fewest supports then earliest; no test access",
        ac_fit="shared impedances; measured PQ/root fixed; local least_squares max_nfev=12, impedance [0,1] pu",
        NJ="ordered R/X -> observed-root classical Q-criterion NJ, RX75 collapse .16; optional fixed-tree L1/AC prediction refit",
        WZZT="singleton init; compatible fully corrective one-atom forward search; train RX75 floor .16",
        RNJ_WZZT="singleton plus at most two stable terminal peer groups; atomic membership compression; empty groups retained",
        scope="first full no-Source round; one paired training draw and independent test per condition; not Monte Carlo robustness",
        workers=4, solver_threads=1,
        environment={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "gurobipy")},
        python=sys.version, jobs=[])
    for case in CASES:
        for condition, settings in CONDITIONS.items():
            directory = output/"data"/case/condition
            directory.mkdir(parents=True)
            context = sim._prepare_case_context(case, 96, scenario_suite="reference", root_observation="noisy",
                        root_meter_noise_rel=settings["v_noise"], root_sigma=.003, impedance_scale=1.)
            if settings["low_xr"]:
                service = context.net.branches["branch_type"].str.contains("service", case=False)
                context.net.branches.loc[:, "x_ohm"] = context.net.branches["r_ohm"]*np.where(service, .06, .30)
            network = directory/"truth_network"
            network.mkdir()
            context.net.buses.to_csv(network/"buses.csv", index=False)
            context.net.branches.to_csv(network/"branches.csv", index=False)
            write(network/"metadata.json", dict(root_bus=int(context.net.root_bus), base_kv=float(context.net.base_kv),
                                               base_mva=float(context.net.base_mva)))
            diagnostics=[]
            for split, replicate in (("train", 0), ("test", 1)):
                pool = {name: [] for name in ("P", "Q", "V", "root")}
                true_pool = {name: [] for name in ("P", "Q", "V", "root")}
                for day in range(3):
                    base = sim._build_base_trajectory(context, day, replicate)
                    base = replace(base, p_base=base.p_base*settings["load_factor"], q_base=base.q_base*settings["load_factor"])
                    ac, voltage = sim._solve_physical_ac(context, base.p_base, base.q_base, base.root_true)
                    ptrue, qtrue = base.p_base.loc[:, context.terminals], base.q_base.loc[:, context.terminals]
                    observed = sim._observe_terminal_channels(context, base, ptrue, qtrue, voltage,
                                pq_noise_rel=settings["pq_noise"], v_noise_rel=settings["v_noise"])
                    for key, name in zip(pool, ("p_measured", "q_measured", "voltage_measured", "root_observed"), strict=True):
                        pool[key].append(observed[name].to_numpy())
                    for key, values in zip(true_pool, (ptrue, qtrue, voltage, base.root_true), strict=True):
                        true_pool[key].append(values.to_numpy())
                    drop=base.root_true.to_numpy()[:, None]-voltage.to_numpy()
                    diagnostics.append(dict(split=split, day=day, profile=base.profile, seed=base.shifted_seed,
                        converged=int(ac["converged"].sum()), voltage_min=float(voltage.min().min()),
                        voltage_max=float(voltage.max().max()), mean_drop=float(drop.mean()), max_drop=float(drop.max()),
                        true_pq_condition=float(np.linalg.cond(np.c_[ptrue.to_numpy(), qtrue.to_numpy()])),
                        settings=observed["scenario_settings"]))
                np.savez_compressed(directory/f"{split}_observed.npz", labels=context.terminals, **{k: np.stack(v) for k,v in pool.items()})
                np.savez_compressed(directory/f"{split}_truth.npz", labels=context.terminals, **{k: np.stack(v) for k,v in true_pool.items()})
            write(directory/"diagnostics.json", diagnostics)
            for size in SIZES:
                for method in METHODS:
                    job_id=f"{case}__{condition}__N{size}__{method}"
                    job_dir=output/"jobs"/job_id
                    job_dir.mkdir(parents=True)
                    payload=dict(job_id=job_id, case=case, condition=condition, N=size, method=method,
                                 train_observed=str(directory/"train_observed.npz"), solve=protocol["solve"])
                    write(job_dir/"payload.json", payload)
                    protocol["jobs"].append(job_id)
            print("prepared", case, condition, flush=True)
    protocol["expected_jobs"]=len(protocol["jobs"])
    write(output/"protocol.json", protocol)
    files=[p for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += [p for p in (output/"data").rglob("*") if p.is_file()]
    files += list((output/"jobs").glob("*/payload.json")) + [output/"protocol.json"]
    write(output/"frozen_manifest.json", {str(p.relative_to(output)).replace("\\", "/"): sha(p) for p in sorted(files)})
    return output


if __name__ == "__main__":
    prepare(sys.argv[1])
