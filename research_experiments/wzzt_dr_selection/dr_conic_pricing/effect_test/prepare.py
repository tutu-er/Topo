"""Freeze a 16-record real-identifier pilot for continuous conic DR prices.

Only existing historical observations and saved historical fit points are read
here. No future, graph, truth evaluation, AC, or new topology fit is loaded.
The linear benefit is fixed-reference signed voltage prediction separation;
its translation to final topology error is the experimental question.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import zipfile

SCRATCH = Path(r"C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a")
DEFAULT_REPO = Path(r"C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo")
MAIN = SCRATCH/"wzzt_targeted_pricing_v1/flynn_run_v1"
CONTROL = SCRATCH/"wzzt_budget_reference_diagnostic_v1/run_v1"
POLICIES = ("zero", "rotation", "socp", "misocp")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(ref):
    path = Path(ref["path"])
    if sha(path) != ref["sha256"]:
        raise RuntimeError("input hash drift: "+str(path))
    return path


def configure(repo_root):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    repo = Path(repo_root).resolve()
    for folder in (repo, repo/"rnj_wzzt_core"):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    return repo


def _source_cases():
    old_main = read(MAIN/"protocol.json")
    old_control = read(CONTROL/"protocol.json")
    control = old_control["cases"]["flynn16"]
    return [
        dict(key="main", case_index=0, case="flynn16", job_id=old_main["job_id"],
             voltage_sigma=.0005, history_packet=old_main["history_packet"],
             baseline_result=reference(MAIN/"baseline_fit/result.json"),
             original_protocol=reference(MAIN/"protocol.json"),
             source_root=old_main["source_root"], source_metadata=old_main["source_metadata"],
             root_label=old_main["root_label"], base_kw=old_main["base_kw"]),
        dict(key="zero_control", case_index=1, case="flynn16", job_id=control["job_id"],
             voltage_sigma=.0002, history_packet=control["history_packet"],
             baseline_result=control["baseline_frozen_result"],
             original_protocol=reference(CONTROL/"protocol.json"),
             source_root=old_control["physical_source_root"], source_metadata=control["source_metadata"],
             root_label=control["root_label"], base_kw=200.),
    ]


def _sources(repo, cases, extras):
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
    from research_experiments.wzzt_dr_selection.dr_day_modules.day_physical_worker import PRODUCER_MODULES, LOCAL_SOURCES
    sources = source_hashes()
    task = repo/"research_experiments/wzzt_dr_selection"
    local = list((task/"dr_conic_pricing").glob("*.py"))
    local += list((task/"dr_conic_pricing/effect_test").glob("*.py"))
    local += [task/"dr_day_modules/run_day_trial.py", task/"dr_pool_v2/matrix_runner.py",
              task/"dr_balanced_design/budget_reference_diagnostic.py", Path(__file__).resolve()]
    local += [repo/name for name in LOCAL_SOURCES]
    local += [Path(name).resolve() for name in extras]
    for case in cases:
        local += [Path(case["source_root"]).joinpath(*name.split(".")).with_suffix(".py")
                  for name in PRODUCER_MODULES]
    for path in local:
        sources[str(path.resolve())] = sha(path)
    return sources, source_hashes()


def _matrices(point, n):
    import numpy as np
    R, X = np.zeros((n, n)), np.zeros((n, n))
    if not len(point["supports"]) == len(point["r"]) == len(point["x"]):
        raise ValueError("saved historical fit dimensions differ")
    for C, r, x in zip(point["supports"], point["r"], point["x"]):
        z = np.zeros(n)
        z[C] = 1.
        R += float(r)*np.outer(z, z)
        X += float(x)*np.outer(z, z)
    return R, X


def _choose_competition(baseline):
    """Use actual selected point and a declared historical alternative, no truth."""
    points = baseline["selection_candidates"]
    selected = baseline["models"]["multistart_original_rule"]["selected_uncontracted"]
    index = selected.get("selection_index")
    if index is None or not 0 <= index < len(points):
        raise ValueError("saved selected raw index is unavailable")
    base = points[index]
    path_index = base.get("path_index")
    same_next = [(i, p) for i, p in enumerate(points)
                 if path_index is not None and p.get("origin") == base.get("origin")
                 and p.get("path_index") == path_index+1 and len(p["supports"]) > len(base["supports"])]
    if same_next:
        other_index, other = min(same_next, key=lambda v: v[0])
        rule = "next accepted point on the selected historical W path"
    else:
        cutoff = 1.03*min(p["validation_mae"] for p in points)+1e-12
        candidates = [(i, p) for i, p in enumerate(points)
                      if len(p["supports"]) > len(base["supports"]) and p["validation_mae"] <= cutoff]
        if not candidates:
            raise ValueError("no higher-support historical eligible competitor; no direction fabricated")
        other_index, other = min(candidates, key=lambda v: (len(v[1]["supports"]), v[1]["validation_mae"], v[0]))
        rule = "smallest higher-support historical eligible point; validation/order break ties"
    return base, other, dict(selected_index=index, alternative_index=other_index, rule=rule,
                            selected_origin=base.get("origin"), selected_path_index=path_index,
                            alternative_origin=other.get("origin"), alternative_path_index=other.get("path_index"),
                            selected_support_count=len(base["supports"]), alternative_support_count=len(other["supports"]),
                            planning_scope="fixed historical fitted references, not full parameter-face separation")


def _build_problem(history, baseline):
    import numpy as np
    from research_experiments.wzzt_dr_selection.dr_day_modules.day_schedule import ResourceDay, DayModule, shift_shape
    from research_experiments.wzzt_dr_selection.dr_conic_pricing.resources import resource_contract_data
    from research_experiments.wzzt_dr_selection.dr_conic_pricing.model import PriceProblem
    labels = list(history["labels"])
    n = len(labels)
    base_kw = float(history["base_kw"])
    days = history["training"]+history["validation"]
    if len(days) != 6 or any(len(s["P_terminal"]) != 96 for s in days):
        raise ValueError("six complete historical days required")
    observed = np.stack([np.r_[s["P_terminal"].to_numpy().ravel(), s["Q_terminal"].to_numpy().ravel()]
                         *base_kw for s in days])
    mu_day = observed.mean(axis=0)
    centered = observed-mu_day
    _, singular, vT = np.linalg.svd(centered, full_matrices=False)
    # Six samples contain at most five nonzero centered directions. Preserve
    # all five, not an outcome-selected low-rank truncation.
    rank = 5
    F_day = vT[:rank].T*(singular[:rank]*np.sqrt(3./5.))
    primary, alternative, competition = _choose_competition(baseline)
    r0, x0 = _matrices(primary, n)
    r1, x1 = _matrices(alternative, n)
    dr, dx = r1-r0, x1-x0
    p_forecast = mu_day[:96*n].reshape(96, n)/base_kw
    q_forecast = mu_day[96*n:].reshape(96, n)/base_kw
    delta_forecast = p_forecast@dr.T+q_forecast@dx.T
    signs = np.sign(delta_forecast)
    g_day = np.r_[(signs@dr).ravel(), (signs@dx).ravel()]*1e6/(96*n*base_kw)
    if not np.isfinite(g_day).all() or np.max(np.abs(g_day)) == 0:
        raise ValueError("historical references give no nonzero declared linear direction")
    resources_dto = [dict(resource_id=key, node_label=node, baseline_kw=.2, power_upper_kw=.4,
                          prefix_delta_lower_kwh=-.4, prefix_delta_upper_kwh=.4,
                          available=True, reactive_ratio=.35, incentive_per_kwh=0.,
                          electricity_price_per_kwh=0.,
                          declaration_source="explicit synthetic .2kW resource; not field behavior calibration")
                     for key, node in (("anchor", 201), ("partner", 204))]
    resources = [ResourceDay(**row) for row in resources_dto]
    early, late = shift_shape(range(24, 28), range(28, 32)), shift_shape(range(76, 80), range(80, 84))
    shapes = [("early", early), ("late", late), ("spread", early+late), ("opposedlate", -late)]
    local_modules = []
    local_d = []
    for resource, d in (("anchor", [.15, .05]), ("partner", [.15, -.05])):
        for kind, shape in shapes:
            local_modules.append(DayModule(resource+"_"+kind, {resource: shape}, .2,
                                           "predeclared whole-day contract; independent resource metering"))
            local_d.append(d)
    local = resource_contract_data(resources, local_modules, node_labels=labels,
                                   elasticities=[.1]*8, price_upper=[2.]*8,
                                   response_factors=local_d)
    day_m, j = 2*96*n, 16
    background_cols = 4*rank
    factors = background_cols+4
    H = np.zeros((4*day_m, j))
    F = np.zeros((4*day_m, factors))
    D = np.zeros((j, factors))
    A_local = np.asarray(local["device_matrix"])
    A = np.zeros((2*len(A_local), j))
    for event in range(4):
        F[event*day_m:(event+1)*day_m, event*rank:(event+1)*rank] = F_day
    groups, ids = [], []
    for paid_position, event in enumerate((0, 2)):
        start = paid_position*8
        H[event*day_m:(event+1)*day_m, start:start+8] = np.asarray(local["profiles"])
        D[start:start+8, background_cols+paid_position*2:background_cols+paid_position*2+2] = local_d
        A[paid_position*len(A_local):(paid_position+1)*len(A_local), start:start+8] = A_local
        groups.extend([[start+k for k in group] for group in local["exclusive_groups"]])
        ids.extend([f"e{event}_"+s for s in local["contract_ids"]])
    g = np.zeros((2, 4*day_m))
    for k, event in enumerate((0, 2)):
        g[k, event*day_m:(event+1)*day_m] = g_day
    problem = PriceProblem(
        mu0=np.tile(mu_day, 4), background_factors=F, profiles=H, response_factors=D,
        elasticities=[.1]*j, reduction_hours=local["reduction_hours"]*2,
        price_upper=[2.]*j, actual_amplitude_upper=[.2]*j,
        benefit_directions=g, benefit_offsets=[0., 0.], risk_allocations=[.05, .05],
        budget=1.6, device_matrix=A, device_rhs=local["device_rhs"]*2,
        contract_ids=ids, exclusive_groups=groups,
        metadata=dict(source="existing six-day historical observed P/Q and saved historical fits",
                      calibrated=False, node_labels=labels, slots=96, events=4,
                      stacking="event blocks; each P_then_Q_time_major", interval_hours=.25,
                      benefit_units="micro-pu2 signed fixed-reference voltage prediction separation",
                      background_covariance="six observed-day sample covariance; five factors per event; independent event blocks",
                      response_law="independent of background; two shared continuous U[-1,1] factors per paid day",
                      response_relative_rows={"anchor": [.15, .05], "partner": [.15, -.05]},
                      actual_background_law="unchanged original natural-load producer, may differ from planning factor approximation",
                      competition=competition, no_true_topology=True, no_actual_future=True,
                      renewable_scope="no PV added to this existing case; natural net-load uncertainty only"))
    module_dto = [dict(module_id=m.module_id, profiles={k: v.tolist() for k, v in m.profiles.items()},
                       amplitude_upper_kw=m.amplitude_upper_kw, description=m.description) for m in local_modules]
    return problem, resources_dto, module_dto, local_d, competition


def _plan(problem, resources, modules, local_d, amplitudes, policy, threshold):
    import numpy as np
    from research_experiments.wzzt_dr_selection.dr_conic_pricing.model import evaluate_plan
    x = np.asarray(amplitudes, float)
    check = evaluate_plan(problem, x)
    if not check["numerical_feasible"] or (x < 0).any():
        raise ValueError("planned amplitudes fail the original model/domain: "+policy)
    payload = dict(schema_version=1, status="frozen", policy=policy,
                   resources_by_event=[], modules_by_event=[], expected_amplitudes_by_event=[],
                   prices_by_event=[], response_factors_by_event=[], contract_ids_by_event=[],
                   reduction_hours_by_event=[], global_expected_amplitudes=x.tolist(),
                   hard_budget_bound=check["hard_budget_bound"], expected_cost=check["expected_cost"],
                   excitation_threshold=float(threshold), planning_benefit_lower_bound=check["reliable_benefit"],
                   response_scope="whole-resource/day continuous amplitude; no clipping or resampling",
                   budget=1.6)
    for event in range(4):
        payload["resources_by_event"].append(resources)
        if event in (0, 2):
            start = (event//2)*8
            payload["modules_by_event"].append(modules)
            payload["expected_amplitudes_by_event"].append(x[start:start+8].tolist())
            payload["prices_by_event"].append((x[start:start+8]/problem.elasticities[start:start+8]).tolist())
            payload["response_factors_by_event"].append(local_d)
            payload["contract_ids_by_event"].append(list(problem.contract_ids[start:start+8]))
            payload["reduction_hours_by_event"].append(problem.reduction_hours[start:start+8].tolist())
        else:
            for name in ("modules_by_event", "expected_amplitudes_by_event", "prices_by_event",
                         "response_factors_by_event", "contract_ids_by_event", "reduction_hours_by_event"):
                payload[name].append([])
    return payload, check


def prepare(master_root, *, repo_root=DEFAULT_REPO, extra_sources=(), moment_validation_samples=2000):
    repo = configure(repo_root)
    import numpy as np
    from research_experiments.wzzt_dr_selection.dr_conic_pricing.cli import environment
    from research_experiments.wzzt_dr_selection.dr_conic_pricing.solver import solve_price
    from research_experiments.wzzt_dr_selection.dr_conic_pricing.sampling import validate_frozen_plan
    if (isinstance(moment_validation_samples, bool) or not isinstance(moment_validation_samples, int)
            or moment_validation_samples < 0):
        raise ValueError("moment_validation_samples must be a nonnegative integer")
    root = Path(master_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cases = _source_cases()
    sources, algorithm = _sources(repo, cases, extra_sources)
    preparation = dict(status="frozen", id="continuous_conic_effect_pilot_v1", created_utc=now(),
                       repository=str(repo), cases=cases, policies=list(POLICIES), test_paths=[1100, 1101],
                       source_sha256=sources, algorithm_source_sha256=algorithm,
                       environment=environment(),
                       response_seed=2026110101, replicate_base=450000, event_profiles=[0, 2, 0, 2],
                       moment_validation_samples=int(moment_validation_samples), independent_requested_records=16,
                       read_boundary="only saved historical observations/results; no future or true topology parsed",
                       selection_scope="known failed cohort plus retained known-correct control; not population case selection")
    write(root/"preparation_protocol.json", preparation)
    case_roots = {}
    try:
        for spec in cases:
            history_path, baseline_path = checked(spec["history_packet"]), checked(spec["baseline_result"])
            checked(spec["original_protocol"])
            checked(spec["source_metadata"])
            with history_path.open("rb") as stream:
                history = pickle.load(stream)["history"]
            baseline = read(baseline_path)
            for split in ("training", "validation"):
                if sum(len(s["P_terminal"]) for s in history[split]) != 288:
                    raise ValueError("original history is not288/288")
            subroot = root/spec["key"]
            subroot.mkdir()
            problem, resource_dto, module_dto, local_d, competition = _build_problem(history, baseline)
            problem_path = subroot/"problem.json"
            write(problem_path, problem.to_dict())
            solutions = {}
            for mode in ("socp", "misocp"):
                solved = solve_price(problem, mode=mode, allowed_contracts=[2, 6, 10, 14] if mode == "socp" else None,
                                     time_limit=60., threads=1, seed=0)
                write(subroot/(mode+"_optimization.json"), solved)
                if not solved["feasible_incumbent"]:
                    raise RuntimeError(mode+" has no feasible price plan for "+spec["key"])
                solutions[mode] = solved
                if moment_validation_samples:
                    validated = validate_frozen_plan(problem, solved["amplitudes"],
                        samples=moment_validation_samples, seed=2026110102+spec["case_index"]*10+(mode == "misocp"),
                        numerical_tolerance=solved["numerical_check_tolerance"])
                    write(subroot/(mode+"_moment_validation.json"), validated)
            # The simple rotation does not use g: one spread contract on anchor
            # during train event0, one on partner in validation event2.
            rotation = np.zeros(problem.n_contracts)
            active = [2, 14]
            common_price_cap = min(problem.amplitude_upper[k]/problem.elasticities[k] for k in active)
            cost_denominator = sum(problem.reduction_hours[k]*problem.elasticities[k]*(1+problem.q[k]) for k in active)
            common_price = min(common_price_cap, float(np.sqrt(problem.budget/cost_denominator)))
            rotation[active] = common_price*problem.elasticities[active]
            amplitudes = dict(zero=np.zeros(problem.n_contracts), rotation=rotation,
                              socp=solutions["socp"]["amplitudes"], misocp=solutions["misocp"]["amplitudes"])
            # A single common evaluation threshold permits an honest comparison
            # of the four strategies' physical excitation event probabilities.
            threshold = min(solutions[m]["achieved_benefit_lower_bound"] for m in ("socp", "misocp"))
            plans, details = {}, {}
            for policy in POLICIES:
                payload, analytic = _plan(problem, resource_dto, module_dto, local_d,
                                          amplitudes[policy], policy, threshold)
                file = subroot/"plans"/(policy+".json")
                write(file, payload)
                plans[policy] = reference(file)
                details[policy] = dict(expected_payment=analytic["expected_cost"],
                                       hard_budget_bound=analytic["hard_budget_bound"],
                                       reliable_benefit=analytic["reliable_benefit"],
                                       positive_contract_ids=[problem.contract_ids[k] for k,a in enumerate(amplitudes[policy]) if a > 1e-6])
            protocol = dict(id="continuous_conic_effect_case_v1", status="frozen", created_utc=now(),
                repository=str(repo), case=spec["case"], case_key=spec["key"], case_index=spec["case_index"],
                job_id=spec["job_id"], source_root=spec["source_root"], source_metadata=spec["source_metadata"],
                history_packet=spec["history_packet"], historical_baseline_result=spec["baseline_result"],
                root_label=spec["root_label"], base_kw=spec["base_kw"], labels=list(history["labels"]),
                voltage_sigma=spec["voltage_sigma"], pq_sigma=.005,
                plans=plans, policies=list(POLICIES), policy_details=details, problem=reference(problem_path),
                source_sha256=sources, algorithm_source_sha256=algorithm, competition=competition,
                environment=environment(),
                test_paths=[1100, 1101], response_seed=2026110101,
                replicate_base=450000, event_profiles=[0, 2, 0, 2], response_law="continuous_shared_two_factor_per_event",
                original_history_rows=[288, 288], future_rows=[480, 480], hard_four_event_budget=1.6,
                incentive_budget=1.6, lp_method=2, search_seconds=25., workers=2,
                maximum_unique_fits=8, maximum_requested_records=8, fit_worker_wall_seconds=3600,
                excitation_threshold=threshold,
                probability_scope="90pct fixed linear excitation guarantee under planning moments, not topology recovery",
                actual_law_boundary="unchanged original natural-load producer; DR factors independent of it, correlated across resources",
                no_truth_or_actual_future_in_planning=True, all_available_history_used=True)
            write(subroot/"protocol.json", protocol)
            case_roots[spec["key"]] = str(subroot)
            print(json.dumps(dict(stage="prepared_case", case=spec["key"], competition=competition,
                                  policies=details), allow_nan=False), flush=True)
        master = dict(status="frozen", id="continuous_conic_effect_master_v1", created_utc=now(),
                      preparation_protocol=reference(root/"preparation_protocol.json"),
                      case_roots=case_roots, case_protocols={k: reference(Path(v)/"protocol.json") for k,v in case_roots.items()},
                      policies=list(POLICIES), test_paths=[1100, 1101], requested_records=16,
                      source_sha256=sources, algorithm_source_sha256=algorithm,
                      prices_frozen_before_any_future=True, source_scope="full historical observations; no newidentifier orAC called")
        write(root/"price_freeze.json", master)
        with zipfile.ZipFile(root/"source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for index, filename in enumerate(sorted(sources)):
                archive.write(filename, arcname=f"{index:03d}_"+Path(filename).name)
        write(root/"preparation_ended.json", dict(status="complete", completed_utc=now(),
              price_freeze=reference(root/"price_freeze.json"), source_snapshot=reference(root/"source_snapshot.zip"),
              new_identifier_calls=0, new_AC_calls=0, graph_or_future_parsed=False))
        return master
    except Exception as error:
        import traceback
        write(root/"preparation_ended.json", dict(status="failed", completed_utc=now(),
              error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
              new_identifier_calls=0, new_AC_calls=0, graph_or_future_parsed=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO))
    parser.add_argument("--extra-source", action="append", default=[])
    parser.add_argument("--moment-validation-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    prepare(args.root, repo_root=args.repo_root, extra_sources=args.extra_source,
            moment_validation_samples=args.moment_validation_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
