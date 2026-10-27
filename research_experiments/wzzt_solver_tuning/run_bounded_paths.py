"""Small no-Source pilot of approximate forward search plus AC path screening.

The old core and saved results are never modified. Fixed supports are solved by
the old strict refit; approximate extensions are decoded by this research driver.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from time import perf_counter

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd

from .profile_gaps import ROOT, RUN, RUNTIME, JOBS, read, write, sha
from .analyze_profiles import observations, candidate_point
from .bounded_solver import solve


def raw_model(model):
    return dict(objective_coefficients=np.asarray(model.variables.objective),
        lower_bounds=np.asarray(model.variables.lower), upper_bounds=np.asarray(model.variables.upper),
        integrality=np.asarray(model.variables.integrality),
        constraint_row_index=np.asarray(model.rows.row_index, dtype=int),
        constraint_col_index=np.asarray(model.rows.col_index, dtype=int),
        constraint_data=np.asarray(model.rows.data), constraint_lower=np.asarray(model.rows.lower),
        constraint_upper=np.asarray(model.rows.upper))


def run_job(job, output, protocol):
    from rnj_wzzt.estimation import laminar_l1_milp as core
    from research_experiments.rnj.rx75_edge_floor import enforce_rx75_floor
    from .ac_screen import select_ac_one_se
    directory = RUN/"jobs"/job
    initial = read(directory/"initialization.json")
    calibration = read(directory/"rx75_calibration.json")
    observed = observations(job)
    p, q, v, root = observed
    labels = list(p.columns)
    scenes = [{"P_terminal": p, "Q_terminal": q,
        "drop_target": pd.DataFrame(root.to_numpy()[:, None]**2-v.to_numpy()**2, index=p.index, columns=p.columns)}]
    prepared = core._prepare_scenarios(scenes)
    supports = tuple(tuple(s) for s in initial["initial_supports"])
    if np.any(np.load(directory/"model.npz")["P_source"] != 0):
        raise ValueError("pilot only supports no-Source inputs")
    output.mkdir()
    started = perf_counter()
    path, attempts = [], []
    reason = "iteration_limit"
    with enforce_rx75_floor(core, None, calibration):
        fixed = core.solve_fixed_support_l1(scenes, supports, r_upper_bound=2., x_upper_bound=2.,
            time_limit=30., solver="gurobi")
        current = fixed.objective
        point = candidate_point(labels, supports, fixed.r_values, fixed.x_values, observed, refit=True)
        point["search_objective"] = current
        path.append(point)
        for iteration in range(len(labels)-1):
            remaining = protocol["method_seconds"]-(perf_counter()-started)
            if remaining <= 1:
                reason = "method_wall_limit"
                break
            if len(supports) >= 2*len(labels)-1:
                reason = "family_size_limit"
                break
            model = core.build_extension_model(prepared, supports, r_upper_bound=2., x_upper_bound=2.)
            raw = raw_model(model)
            info = None
            for stage, gap in enumerate(protocol.get("gap_schedule", [protocol["requested_gap"]])):
                remaining = protocol["method_seconds"]-(perf_counter()-started)
                if remaining <= 1:
                    break
                target = output/f"attempt_{iteration:03d}_stage_{stage}"
                target.mkdir()
                np.savez_compressed(target/"model.npz", **raw)
                primal, info = solve(raw, current=current, seconds=min(protocol["per_solve_seconds"], remaining),
                    relative_gain=protocol["relative_gain"], uncertainty_fraction=protocol["uncertainty_fraction"],
                    requested_gap=gap)
                info.update(old_supports=[list(s) for s in supports],
                    variable_slices={k:[v.start,v.stop] for k,v in model.variables.slices.items()},
                    initial_upper_bound_touches=bool(max(path[0]["r"]+path[0]["x"]) >= 1.99),
                    iteration=iteration, refinement_stage=stage)
                if primal is not None:
                    np.savez_compressed(target/"primal.npz", primal=primal)
                attempts.append(info)
                write(target/"result.json", info)
                if info["decision"] != "refine_or_report_unresolved":
                    break
            if info is None:
                reason = "method_wall_limit"
                break
            if info["decision"] != "accept_bounded_improvement":
                reason = info["decision"]
                break
            new_support = core._round_and_validate_support(model.expand_z_values(primal[model.z]), supports, len(labels))
            extended = (*supports, new_support)
            r_values = core._bounded_weights(np.r_[primal[model.r],primal[model.new_r]],2.)
            x_values = core._bounded_weights(np.r_[primal[model.x],primal[model.new_x]],2.)
            point = candidate_point(labels, extended, r_values, x_values, observed, refit=True)
            if abs(point["physical_l1_objective"]-info["objective"]) > 1e-8:
                raise RuntimeError("incumbent physical L1 does not reproduce MILP objective")
            if max(np.r_[r_values,x_values]) >= 1.99:
                reason = "coefficient_bound_requires_expansion"
                break
            supports, current = extended, info["objective"]
            point["search_objective"] = current
            path.append(point)
            write(output/"checkpoint.json", {"path": path, "attempts": attempts, "stop_reason": "running"})
            print(job, "accepted",len(path)-1,"loss",current,"gap",info["relative_gap"], flush=True)
        selected = select_ac_one_se(path)
    result = {"job_id": job, "stop_reason": reason,
        "status": "complete" if reason == "stop_bound_proves_no_material_gain" else "incomplete",
        "runtime_seconds": perf_counter()-started,
        "path": path, "attempts": attempts, "last_index": len(path)-1,
        "ac_one_se_index": selected, "truth_used": False,
        "scope": "bounded no-Source pilot; original initial supports/floor; no swaps or zero-pruning; fixed box [0,2]; AC training screen"}
    write(output/"result.json", result)
    print(job,"STOP",reason,"path",len(path),"ac_index",selected,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/wzzt_solver_tuning_20261027/single_pass_paths")
    parser.add_argument("--refine", action="store_true", help="Refine unresolved 1%% solves to 0.1%% and 0.01%%")
    parser.add_argument("--single-pass", action="store_true", help="Default: 0.01%% final gap with gain callback, avoiding cold restarts")
    parser.add_argument("--coarse", action="store_true", help="Historical comparison: 1%% gap without refinement")
    parser.add_argument("--jobs", nargs="+", choices=JOBS, default=list(JOBS))
    args = parser.parse_args()
    if sum((args.refine,args.single_pass,args.coarse))>1:
        parser.error("choose one of --refine, --single-pass or --coarse")
    protocol = dict(jobs=args.jobs, requested_gap=.0001, uncertainty_fraction=.5, relative_gain=.001,
        absolute_gain=1e-10, per_solve_seconds=60., method_seconds=240.,
        ac_rule="four temporal blocks; simplest path point within one block SE of minimum AC voltage MSE",
        ac_refit="local shared r/x in [0,1] pu, 12 evaluations; all 96 rows; measured PQ/root fixed",
        scope="isolated pilot; gain uncertainty <= half realized gain (at least 2/3 of best possible one-step improvement)",
        production_defaults_changed=False, input_sha256={})
    if args.refine:
        protocol["requested_gap"] = .01
        protocol["gap_schedule"] = [.01,.001,.0001]
    if args.coarse:
        protocol["requested_gap"] = .01
    for path in Path(__file__).parent.glob("*.py"):
        protocol["input_sha256"][str(path.relative_to(ROOT))] = sha(path)
    for job in args.jobs:
        directory=RUN/"jobs"/job
        for name in ("initialization.json","rx75_calibration.json","payload.json","model.npz"):
            protocol["input_sha256"][str((directory/name).relative_to(ROOT))]=sha(directory/name)
        for path in (Path(read(directory/"payload.json")["input_dir"])/"observed").glob("*.csv"):
            protocol["input_sha256"][str(path.relative_to(ROOT))]=sha(path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write(args.output_dir/"protocol.json",protocol)
    import shutil
    snapshot=args.output_dir/"producer_source_snapshot"
    snapshot.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path,snapshot/path.name)
    sys.path.insert(0,str(RUNTIME))
    for job in args.jobs:
        run_job(job,args.output_dir/job,protocol)
    write(args.output_dir/"completion.json",dict(expected=len(args.jobs),returned=len(args.jobs)))


if __name__ == "__main__":
    main()
