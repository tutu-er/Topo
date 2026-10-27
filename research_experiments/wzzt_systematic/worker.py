"""Fit using the payload's training observations only; checkpoint every step."""
from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter
import traceback

from .common import (ac_diagnostic, linear_loss, matrices, observed, raw_model, read,
                     scenes_from_observed, write)


def fit(payload_path):
    import numpy as np
    from rnj_wzzt.estimation import laminar_l1_milp as core
    from research_experiments.rnj.source_ac96_trees import fit_tree
    from research_experiments.rnj.rx75_edge_floor import enforce_rx75_floor, calibrate_rx75_floor
    from research_experiments.wzzt_solver_tuning.bounded_solver import solve
    from research_experiments.wzzt_solver_tuning.ac_screen import select_ac_one_se
    payload = read(payload_path)
    output = Path(payload_path).parent
    settings = payload["solve"]
    started = perf_counter()
    obs = observed(payload["train_observed"], payload["N"])
    scenes = scenes_from_observed(obs)
    labels = list(obs[0].columns)
    n = len(labels)
    front = fit_tree(scenes, None, "NJ")
    calibration = calibrate_rx75_floor(front["R"], front["X"])
    result = dict(job_id=payload["job_id"], method=payload["method"], labels=labels,
        N=payload["N"], calibration=calibration, front_diagnostics=front["diagnostics"],
        front_seconds=perf_counter()-started, groups=[], bootstrap_seconds=0.,
        path=[], attempts=[], fixed_solves=[], bound_expansions=[], truth_used=False,
        data_access="payload contains training observations only; no test/truth paths")
    initial = tuple((i,) for i in range(n))
    groups = None
    if payload["method"] == "RNJ+WZZT":
        from rnj_wzzt import pipeline
        from research_experiments.rnj.select_rnj_blocks96 import select_rnj_blocks96
        then = perf_counter()
        diag = {}
        _, _, selected, _, _ = select_rnj_blocks96(scenes, labels, 1, runtime_pipeline=pipeline,
            sensitivity_fit=(front["R"], front["X"], front["projected_r2_score"], front["condition"]),
            sensitivity_diagnostics=front["diagnostics"], diagnostics=diag,
            block_length=max(1, round(payload["N"]/3/24)))
        position = {label: i for i,label in enumerate(labels)}
        groups = tuple(sorted((tuple(sorted(position[x] for x in clade)) for clade in selected), key=lambda s:(len(s),s)))
        initial = tuple(sorted(set(initial)|set(groups), key=lambda s:(len(s),s)))
        result.update(groups=[list(s) for s in groups], bootstrap_seconds=perf_counter()-then)
        write(output/"bootstrap.json", diag)
    bound = settings["initial_weight_bound"]
    reason = "not_started"

    def checkpoint():
        result.update(stop_reason=reason, search_seconds=perf_counter()-started)
        write(output/"checkpoint.json", result)

    def point(supports, r, x, objective):
        rr, xx = matrices(n, supports, r, x)
        metric = linear_loss(obs, rr, xx)
        if abs(metric["mae"]-objective) > 1e-8:
            raise RuntimeError("decoded physical objective disagrees with solver objective")
        return dict(supports=[list(s) for s in supports], r=np.asarray(r).tolist(), x=np.asarray(x).tolist(),
                    search_objective=float(objective), linear_train=metric)

    def fixed(supports, epoch, purpose):
        remaining = settings["search_wall_seconds"]-(perf_counter()-started)
        if remaining < 1:
            raise TimeoutError("method wall exhausted before fixed fit")
        fit = core.solve_fixed_support_l1(scenes, supports, r_upper_bound=bound, x_upper_bound=bound,
                                         time_limit=min(30., remaining), solver="gurobi")
        result["fixed_solves"].append(dict(epoch=epoch, purpose=purpose, supports=supports,
                                           diagnostics=fit.diagnostics.to_dict()))
        return fit

    with enforce_rx75_floor(core, None, calibration):
        if payload["method"] == "NJ":
            # Topology is untouched by this common impedance fit; dense matrices
            # remain the declared NJ R/X estimator.
            supports = tuple(sorted(set(initial)|{tuple(sorted(labels.index(t) for t in c)) for c in front["clades"]},
                                    key=lambda s:(len(s),s)))
            solution = fixed(supports, 0, "NJ_fixed_tree_prediction_refit")
            result["path"] = [point(supports, solution.r_values, solution.x_values, solution.objective)]
            result["NJ_clades"] = [sorted(c) for c in front["clades"]]
            result["NJ_R"] = front["R"].tolist()
            result["NJ_X"] = front["X"].tolist()
            result["NJ_linear_train"] = linear_loss(obs, front["R"], front["X"])
            reason = "NJ_finished"
        else:
            prepared = core._prepare_scenarios(scenes)
            finished = False
            for epoch in range(settings["max_bound_expansions"]+1):
                if finished:
                    break
                if perf_counter()-started >= settings["search_wall_seconds"]-1:
                    reason = "method_wall_limit"
                    break
                family = initial
                solution = fixed(family, epoch, "initial_or_expanded_restart")
                current = solution.objective
                path = [point(family, solution.r_values, solution.x_values, current)]
                result["path"] = path
                seen = {tuple(sorted(family))}
                restart = False
                for iteration in range(4*n):
                    remaining = settings["search_wall_seconds"]-(perf_counter()-started)
                    if remaining < 1:
                        reason = "method_wall_limit"
                        break
                    if max(path[-1]["r"]+path[-1]["x"]) >= settings["bound_active_fraction"]*bound:
                        restart, reason = True, "coefficient_bound_requires_expansion"
                        break
                    if len(family) >= 2*n-1:
                        reason = "family_size_limit"
                        break
                    model = core.build_extension_model(prepared, family, r_upper_bound=bound, x_upper_bound=bound,
                                                       terminal_groups=groups)
                    raw = raw_model(model)
                    target = output/f"epoch{epoch}_step{iteration}"
                    target.mkdir()
                    np.savez_compressed(target/"model.npz", **raw)
                    primal, info = solve(raw, current=current, seconds=min(settings["per_solve_seconds"],remaining),
                        relative_gain=settings["relative_gain"], uncertainty_fraction=settings["uncertainty_fraction"],
                        requested_gap=settings["requested_gap"])
                    info.update(epoch=epoch, iteration=iteration, old_supports=family,
                        variable_slices={k:[v.start,v.stop] for k,v in model.variables.slices.items()},
                        coefficient_bound=bound, current_objective=current, terminal_partition=model.partition.groups)
                    if primal is not None:
                        np.savez_compressed(target/"primal.npz", primal=primal)
                    write(target/"result.json", info)
                    result["attempts"].append(info)
                    if info["decision"] != "accept_bounded_improvement":
                        reason = info["decision"]
                        break
                    new = core._round_and_validate_support(model.expand_z_values(primal[model.z]), family, n)
                    extended = (*family, new)
                    r = core._bounded_weights(np.r_[primal[model.r], primal[model.new_r]], bound)
                    x = core._bounded_weights(np.r_[primal[model.x], primal[model.new_x]], bound)
                    if max(np.r_[r,x]) >= settings["bound_active_fraction"]*bound:
                        restart, reason = True, "coefficient_bound_requires_expansion"
                        break
                    candidate = point(extended, r, x, info["objective"])
                    # Remove only zero, non-initial atoms, then verify by a
                    # corrective fixed refit. Positive wrong atoms remain.
                    kept = tuple(s for s,rr,xx in zip(extended,r,x,strict=True)
                                 if s in initial or max(rr,xx)>settings["zero_prune_tolerance"])
                    if len(kept) < len(extended):
                        reduced = fixed(kept, epoch, "verify_zero_pruning")
                        if reduced.objective <= info["objective"]+1e-10:
                            extended = kept
                            candidate = point(kept, reduced.r_values, reduced.x_values, reduced.objective)
                    family = tuple(extended)
                    current = candidate["search_objective"]
                    path.append(candidate)
                    checkpoint()
                    print(payload["job_id"], "epoch",epoch,"step",iteration,"objective",current,flush=True)
                    canonical=tuple(sorted(family))
                    if canonical in seen:
                        reason = "family_cycle"
                        break
                    seen.add(canonical)
                else:
                    reason = "forward_step_limit"
                if restart and epoch < settings["max_bound_expansions"]:
                    result["bound_expansions"].append(dict(epoch=epoch, old_bound=bound, path=path))
                    bound *= 2
                    checkpoint()
                    continue
                finished = True
                checkpoint()
    result.update(stop_reason=reason, search_seconds=perf_counter()-started,
                  status="complete" if reason in {"NJ_finished", "stop_bound_proves_no_material_gain", "family_size_limit"} else "incomplete",
                  final_bound=bound)
    write(output/"checkpoint.json", result)
    for i, candidate in enumerate(result["path"]):
        candidate["ac"] = ac_diagnostic(labels, candidate["supports"], candidate["r"], candidate["x"], obs, refit=True)
        write(output/"checkpoint.json", result)
    result["selected_index"] = select_ac_one_se(result["path"])
    result["last_index"] = len(result["path"])-1
    result["elapsed_seconds"] = perf_counter()-started
    write(output/"result.json", result)
    print(payload["job_id"], "DONE",reason,"AC index",result["selected_index"],flush=True)


if __name__ == "__main__":
    try:
        fit(Path(sys.argv[1]))
    except Exception as exc:
        output=Path(sys.argv[1]).parent
        write(output/"error.json", dict(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()))
        raise
