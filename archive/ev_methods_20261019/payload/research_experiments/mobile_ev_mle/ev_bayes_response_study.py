"""Frozen systematic EV design with shared, bounded line-response uncertainty."""
from __future__ import annotations

import argparse
import base64
import gzip
from dataclasses import asdict, is_dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter
import traceback

import numpy as np
import scipy

from . import ev_bayes_small_ac as fixed_ac
from . import ev_bayes_small_probability as probability
from . import ev_bayes_small_milp as old_milp
from . import ev_bayes_small_study as small
from . import ev_bayes_response_ac as joint_ac
from . import ev_bayes_response_cases as cases
from . import ev_bayes_response_milp as new_milp
from . import ev_mobile_v4_domain as nested
from . import ev_mobile_v3_cases as public
from . import ev_rx_study as rx
from . import gradient_price_study as io
from . import ev_original_study as generator
from . import likelihood as lk
from . import ev_systematic_refit as refit
from . import ev_mobile_v4_study as legacy
from .ev_rx_fit import fit_rx_inputs
from .ev_rx_precision import solve_squared_precision, precision_weights
from .information_context import meter_to_record
from .symmetric_ac_local import mean_and_jacobian, whiten_observations


HERE = Path(__file__).resolve().parent
CONFIG = dict(version="bayes_response_v1", horizon=8, evidence=20,
    planning_samples_per_candidate=256, independent_samples_per_candidate=1024,
    actual_noise_repeats=32, source_interface=1, solver_seconds=5.,
    max_proposals=1, segments=4, exp_segments=8, power_radius_kw=None,
    square_base_iterations=1, square_budget_iterations=2,
    abstention_cost=.1, credible_mass=.9, fit_max_nfev=300, fit_multistart=1,
    regimes=["regular", "shift"], prior_topology="uniform before training",
    posterior_models=["joint", "fixed", "no_cross", "line_known", "evidence_only"],
    primary_target="expected MAP classification error of the stated predictive bank",
    line_scope="one trained expansion; shared bounded posterior updated by observed evidence",
    probability_scope="empirical local-linear history evidence and moment Gaussian future",
    random_stream_rule="SeedSequence([history_seed, stream_tag, 20261017]); no additive offsets",
    posterior_options=dict(truncation=dict(min_ess_fraction=0.)),
    objective_acceptance="same planning CRN, full AC means and new propagated covariance",
    solver_start="identical complete parent-operation partial MIP start for old/new/fixed BC; other coordinates unspecified",
    sampling_total_precision_per_channel=30.,
    old_science_unchanged=True, evaluation_used_for_tuning=False)


def serial(value):
    if is_dataclass(value):
        return serial(asdict(value))
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return io.serial(value)


def save_record(path, value):
    """Lossless, atomic gzip JSON; avoid a second recursive serialization pass."""
    path = Path(path)
    data = (json.dumps(serial(value), ensure_ascii=False, sort_keys=True, allow_nan=False)+"\n").encode("utf-8")
    temporary = path.with_name(path.name+".tmp")
    temporary.write_bytes(gzip.compress(data, compresslevel=1, mtime=0))
    temporary.replace(path)


def source_hashes():
    result = small.hashes()
    for suffix in ("ac", "truncation", "cases", "milp", "study"):
        path = HERE / ("ev_bayes_response_" + suffix + ".py")
        result[path.name] = sha256(path.read_bytes()).hexdigest()
    for name in ("ev_bayes_response_protocol.md", "ev_bayes_response_math.md"):
        path = HERE / name
        if path.exists():
            result[name] = sha256(path.read_bytes()).hexdigest()
    return result


def freeze(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = dict(config=CONFIG, specifications={p: cases.specifications(p)
        for p in ("development", "calibration", "evaluation")},
        source_sha256=source_hashes(),
        environment=dict(python=platform.python_version(), numpy=np.__version__,
            scipy=scipy.__version__, platform=platform.platform()),
        base_commit="f172dd41", prior_stage_commit="a1b176da")
    result["policy_id"] = io.identity(result["config"], result["source_sha256"])
    path = directory / "freeze.json"
    if path.exists():
        if io.read_json(path) != serial(result):
            raise ValueError("preserve old records: config/source freeze changed")
        return io.read_json(path)
    io.write_json(path, result)
    snapshot = {name: base64.b64encode(((HERE.parents[1]/name)
        if name.startswith("rnj_wzzt_core/") else HERE/name).read_bytes()).decode()
        for name in result["source_sha256"]}
    io.write_json(directory/"source_snapshot.json.gz",
        dict(source_sha256=result["source_sha256"], source_base64=snapshot), compressed=True)
    return serial(result)


def score_summary(result):
    return {k: v for k, v in result.items()
        if k not in ("samples", "stratum_samples", "strata_samples")}


def stream_seed(seed, tag):
    return [int(seed), int(tag), 20261017]


def stream_rng(seed, tag):
    return np.random.default_rng(stream_seed(seed, tag))


class Objective:
    """Only historical context, public operations, and integration RNG enter."""
    def __init__(self, context, noise, model):
        self.context, self.noise, self.model = context, noise, model
        self.cache = {}
        self.calls = self.ac_states = 0
        self.seconds = 0.
        self.objective_id = io.identity(dict(model=model.__name__,
            log_prior=context["log_prior"], noise_sha256=sha256(noise.tobytes()).hexdigest()), {})

    def get(self, power):
        key = nested.action_key(power)
        if key not in self.cache:
            began = perf_counter()
            forecast = self.model.forecast_bank(self.context, power, with_jacobian=True)
            score = probability.evaluate_bank(forecast["bank"], self.noise,
                abstention_cost=CONFIG["abstention_cost"])
            self.cache[key] = dict(power_kw=np.asarray(power).copy(), forecast=forecast,
                score=score, score_value=-float(score["risk"]),
                objective_id=self.objective_id, seconds=perf_counter()-began)
            self.calls += 1
            self.ac_states += int(forecast.get("AC_states", len(forecast["physical"])*len(power)))
            self.seconds += perf_counter()-began
        return self.cache[key]


def square_family(domains, meter, constant, highlow, *, iterations):
    began = perf_counter()
    plans, logs = {}, {}
    for name, domain in domains.items():
        parents = [nested.lift_operation(domains[p], domain, plans[p])
            for p in nested.PARENTS[name]]
        initial = dict(constant=constant, highlow=highlow,
            **{"parent_"+str(i): domain.decode(x)["u_kw"] for i, x in enumerate(parents)})
        result = solve_squared_precision(domain, meter, initial,
            exchange_options=dict(total_seconds=iterations*(CONFIG["solver_seconds"]+1.),
                solver_seconds=CONFIG["solver_seconds"], max_iterations=iterations,
                control_segments=CONFIG["segments"], residual_segments=4, max_segments=8))
        candidates = list(parents)
        if result.get("success"):
            candidates.append(np.asarray(result["operation"]["x"]))
        if not candidates:
            # Constant is an exactly completed operation, independent of solver score.
            candidates.append(small.complete_curve(domain, constant))
        weights = precision_weights(meter, domain.indices.u.shape)
        Q = lambda x: float(np.sum(weights*domain.decode(x)["u_kw"]**2))
        winner = candidates[0]
        for x in candidates[1:]:
            if Q(x) > Q(winner)+1e-9:
                winner = x
        plans[name] = np.asarray(winner).copy()
        logs[name] = dict(result=result, Q=Q(winner),
            inherited_Q_not_decreased=all(Q(winner) >= Q(x)-1e-8 for x in parents))
    return plans, logs, perf_counter()-began


def risk_family(domains, objective, squares, baselines, *, optimizer=None, extra_plans=None):
    began = perf_counter()
    plans, logs = {}, {}
    for name, domain in domains.items():
        started = perf_counter()
        options = [("parent:"+p, nested.lift_operation(domains[p], domain, plans[p]))
            for p in nested.PARENTS[name]]
        for label, x in baselines.items():
            options.append((label, x if name == "fixed_charge"
                else nested.lift_operation(domains["fixed_charge"], domain, x)))
        options.append(("square:"+name, squares[name]))
        if extra_plans is not None:
            options.append(("risk_ranked:"+name, extra_plans[name]))
        ranked = [(label, x, objective.get(domain.decode(x)["u_kw"])) for label, x in options]
        scored_keys = {nested.action_key(domain.decode(x)["u_kw"]) for _, x in options}
        origin, winner, record = ranked[0]
        for label, x, value in ranked[1:]:
            if value["score_value"] > record["score_value"]+1e-12:
                origin, winner, record = label, x, value
        initial_risk = -record["score_value"]
        events = []
        for iteration in range(CONFIG["max_proposals"] if optimizer is not None else 0):
            anchor = domain.decode(winner)["u_kw"]
            forecast = record["forecast"]
            try:
                result = optimizer(domain, forecast["bank"],
                    forecast["jacobian"], anchor, segments=CONFIG["segments"],
                    exp_segments=CONFIG["exp_segments"], seconds=CONFIG["solver_seconds"],
                    power_radius=CONFIG["power_radius_kw"], operation_start=winner)
            except Exception as exc:
                result = dict(success=False, operation_x=None,
                    failure=type(exc).__name__+": "+str(exc))
            event = dict(iteration=iteration, milp=result, accepted=False,
                parent_risk=-record["score_value"], anchor_power_kw=anchor.copy())
            proposal = result.get("operation_x")
            if proposal is not None:
                try:
                    x = np.asarray(proposal)
                    if not domain.check(x)["feasible"]:
                        raise ValueError("MILP proposal violates original EV operation domain")
                    predicted = objective.get(domain.decode(x)["u_kw"])
                    scored_keys.add(nested.action_key(domain.decode(x)["u_kw"]))
                    event.update(proposed_risk=-predicted["score_value"],
                        risk_gain=predicted["score_value"]-record["score_value"])
                    if predicted["score_value"] > record["score_value"]+1e-12:
                        origin, winner, record = "BC_MILP", x, predicted
                        event["accepted"] = True
                except Exception as exc:
                    event["full_AC_failure"] = type(exc).__name__+": "+str(exc)
            events.append(event)
        plans[name] = np.asarray(winner).copy()
        parent_risks = {p: -objective.get(domains[p].decode(plans[p])["u_kw"])["score_value"]
            for p in nested.PARENTS[name]}
        logs[name] = dict(origin=origin, initial_risk=initial_risk,
            final_risk=-record["score_value"], parent_risks=parent_risks,
            inheritance_not_worse=all(-record["score_value"] <= r+1e-12
                for r in parent_risks.values()), events=events,
            accepted_proposals=sum(e["accepted"] for e in events),
            failed_proposals=sum(not bool(e["milp"].get("success")) for e in events),
            scored_action_keys=sorted(scored_keys),
            required_prediction_seconds=sum(objective.cache[key]["seconds"] for key in scored_keys),
            cache_scope="same-objective reusable actions; required time is observed action cost, not a cold rerun",
            seconds=perf_counter()-started)
    return plans, logs, perf_counter()-began


def credible_membership(posterior, mass):
    order = np.argsort(-posterior, axis=1, kind="stable")
    ordered = np.take_along_axis(posterior, order, axis=1)
    previous = np.cumsum(ordered, axis=1)-ordered
    selected = previous < mass
    membership = np.zeros_like(selected)
    np.put_along_axis(membership, order, selected, axis=1)
    return membership


def evaluate_actual(bank, clean, white, *, true_index):
    """One physical trajectory, R paired meter noises; no repeated-history claim."""
    values = np.asarray(clean)[None]+np.asarray(white)
    result = probability.gaussian_posterior(values, bank["means"][0],
        bank["covariances"][0], log_prior=bank["log_prior"])
    post = np.asarray(result["posterior"])
    chosen = np.argmax(post, axis=1)
    members = credible_membership(post, CONFIG["credible_mass"])
    count = members.sum(axis=1)
    truth = np.eye(post.shape[1])[true_index]
    rows = dict(correct=(chosen == true_index).astype(int),
        confidence=post.max(axis=1), true_probability=post[:, true_index],
        posterior=post, brier=np.sum((post-truth)**2, axis=1),
        negative_log_true_probability=-result["log_posterior"][:, true_index],
        unique_correct=((count == 1) & (chosen == true_index)).astype(int),
        wrong_unique=((count == 1) & (chosen != true_index)).astype(int),
        unresolved=(count > 1).astype(int),
        true_rejected=(~members[:, true_index]).astype(int), credible_size=count)
    mean = bank["means"][0, true_index]
    variance = np.diag(bank["covariances"][0, true_index])
    rows["meter_interval90_coverage"] = np.mean(abs(values-mean) <=
        1.6448536269514722*np.sqrt(variance), axis=1)
    n = (len(clean)//CONFIG["horizon"]-2)//3
    v_indices = np.arange(2*CONFIG["horizon"]*n, 3*CONFIG["horizon"]*n)
    rows["voltage_interval90_coverage"] = np.mean(abs(values[:, v_indices]-mean[v_indices]) <=
        1.6448536269514722*np.sqrt(variance[v_indices]), axis=1)
    summary = {k: float(np.mean(v)) for k, v in rows.items() if k != "posterior"}
    return dict(samples=rows, summary=summary, n_noise=len(values),
        n_independent_histories=1, true_index=int(true_index))


def _actual_physical(case, power, other):
    if not np.array_equal(case["train"].v, other["train"].v):
        raise AssertionError("future regime changed observed training")
    start, T = case["future_start"], CONFIG["horizon"]
    sl = slice(start, start+T)
    truth = other["truth"]
    response = mean_and_jacobian(truth["candidate"], truth["r"], truth["x"],
        other["p"][sl]+power/other["base_kw"], other["q"][sl],
        other["source_p"][sl], other["source_q"][sl], other["root_v"][sl],
        case["fm"], with_jacobian=False)
    return response, {name: np.asarray(other[name])[sl]
        for name in ("p", "q", "source_p", "source_q", "root_v")}


def _forecast(objectives, context, power, model):
    if model in objectives:
        return objectives[model].get(power)["forecast"]
    options = dict(covariance_mode=model) if model in ("no_cross", "line_known") else dict(
        weights_mode="evidence_only")
    full = objectives["joint"].get(power)["forecast"]
    return joint_ac.ablate_forecast(context, full, **options)


def first_observation(physical, meter, white):
    """Undo channel whitening only; this observed future is audit-only."""
    t, n = np.asarray(physical["p"]).shape
    blocks, start = {}, 0
    for channel in ("p", "q", "v", "master_p", "master_q"):
        shape = (t, n) if channel in ("p", "q", "v") else (t,)
        size = int(np.prod(shape))
        blocks[channel] = np.asarray(white[start:start+size]).reshape(shape)
        start += size
    if start != len(white):
        raise ValueError("future noise has extra/missing channels")
    return generator.observation(physical, meter, blocks)


def legacy_audit(case, fits, first, methods, seed):
    """Prespecified complete nonlinear joint-line audit, separate from Bayes."""
    D = lk.concat_observations(case["train"], case["evidence"])
    dm = refit.concat_models((case["train"], case["evidence"]), (case["tm"], case["em"]))
    options = {**case["config"]["training"], **case["config"]["source"],
        "ratio_bounds": case["config"]["ratio_bounds"]}
    fixed, failures, rows = {}, {}, {}
    for index, candidate in enumerate(case["bank"]):
        fit = fits[candidate.name]
        try:
            fixed[candidate.name] = fit_rx_inputs(candidate, D, dm, initial=fit,
                fixed_r=fit.r, fixed_x=fit.x, seed=seed+index, **options)
        except Exception as exc:
            failures[candidate.name] = type(exc).__name__+": "+str(exc)
    if failures:
        return dict(complete=False, prefit_failures=failures, results=rows)
    true_name = next(c.name for c in case["bank"] if c.supports == case["truth"]["candidate"].supports)
    for method in ("constant", "square__mobile_bidir", "joint_bc__mobile_bidir"):
        if method not in methods:
            failures[method] = "required plan unavailable"
            continue
        key = methods[method]["action_key"]
        for regime in CONFIG["regimes"]:
            ident = regime+"__"+key
            if ident in rows:
                continue
            try:
                result = legacy.evaluate_future(case["bank"], D, dm,
                    first[(regime,key)], case["fm"], fixed, options, seed+1000)
                if result["complete"]:
                    result["M"] = min(v for k,v in result["losses"].items() if k != true_name)-result["losses"][true_name]
                    result["selected"] = min(result["losses"], key=result["losses"].get)
                rows[ident] = result
            except Exception as exc:
                rows[ident] = dict(complete=False, failure=type(exc).__name__+": "+str(exc))
    return dict(complete=not failures and all(r["complete"] for r in rows.values()),
        failures=failures, results=rows, scope="one predeclared noise per physical regime; local joint R/X fits")


def run(spec):
    began = perf_counter()
    case = cases.make_case(spec)
    record = dict(spec=spec, config=CONFIG, provenance=case["provenance"],
        observed_training=io.observations_record(case["train"]),
        observed_evidence=io.observations_record(case["evidence"]),
        train_meter=meter_to_record(case["tm"]), evidence_meter=meter_to_record(case["em"]),
        future_meter=meter_to_record(case["fm"]), candidate_names=[c.name for c in case["bank"]],
        candidate_miss=False, methods={}, failures={})
    cfg = case["config"]
    cfg["training"]["max_nfev"] = CONFIG["fit_max_nfev"]
    cfg["training"]["multistart"] = CONFIG["fit_multistart"]
    fit_seed = int(np.random.SeedSequence(stream_seed(spec["seed"], 1000)).generate_state(1)[0])
    fits, training = rx.train_bank(case["bank"], case["train"], case["tm"], cfg, fit_seed)
    record["training_result"] = training
    if not training["all_complete"]:
        return dict(record, status="training_failure", seconds=perf_counter()-began)
    print("TRAINED", spec["phase"], spec["index"], spec["bank"], spec["H"],
        round(perf_counter()-began, 2), flush=True)
    contexts, context_records = {}, {}
    for label, module in (("fixed", fixed_ac), ("joint", joint_ac)):
        try:
            posterior_options = dict(CONFIG["posterior_options"],
                r_bounds=cfg["training"]["r_bounds"], x_bounds=cfg["training"]["x_bounds"],
                ratio_bounds=cfg["ratio_bounds"], **cfg["source"],
                seed=int(np.random.SeedSequence(stream_seed(spec["seed"], 5000)).generate_state(1)[0]))
            options = dict(options=posterior_options) if label == "joint" else {}
            contexts[label], context_records[label] = module.build_context(
                case["train"], case["evidence"], case["tm"], case["em"], case["bank"], fits,
                case["fm"], horizon=CONFIG["horizon"], base_kw=case["base_kw"], **options)
        except Exception as exc:
            record["failures"][label+"_inference"] = type(exc).__name__+": "+str(exc)
            if hasattr(exc, "evidence"):
                context_records[label+"_failure_evidence"] = exc.evidence
        print("CONTEXT", label, label in contexts, round(perf_counter()-began, 2), flush=True)
    record["prediction"] = context_records
    T, n, K = CONFIG["horizon"], case["train"].n, len(case["bank"])
    dimension = T*(3*n+2)
    noise = stream_rng(spec["seed"], 2000).standard_normal(
        (1, K, CONFIG["planning_samples_per_candidate"], dimension))
    independent = stream_rng(spec["seed"], 3000).standard_normal(
        (1, K, CONFIG["independent_samples_per_candidate"], dimension))
    modules = dict(fixed=fixed_ac, joint=joint_ac)
    objectives = {label: Objective(context, noise, modules[label])
        for label, context in contexts.items()}
    template = replace(public.service_domain(T, n, mobile=True).config, n_slots=T)
    domains = nested.build_nested_domains(template)
    record["operation_config"] = asdict(template)
    constant = np.zeros((T, n))
    constant[:, 0] = (template.terminal_energy_kwh-template.initial_soc*
        template.battery_capacity_kwh)/(template.eta_charge*T*template.dt_hours)
    highlow = np.zeros_like(constant)
    highlow[:T//2, 0] = constant[0, 0]*T/(T//2)
    baselines = {name: small.complete_curve(domains["fixed_charge"], power)
        for name, power in (("constant", constant), ("highlow", highlow))}
    families, logs, costs = {}, {}, {}
    families["square"], logs["square"], costs["square"] = square_family(
        domains, case["fm"], constant, highlow, iterations=CONFIG["square_base_iterations"])
    families["square_budget"], logs["square_budget"], costs["square_budget"] = square_family(
        domains, case["fm"], constant, highlow, iterations=CONFIG["square_budget_iterations"])
    if "joint" in objectives:
        families["ranked"], logs["ranked"], costs["ranked"] = risk_family(
            domains, objectives["joint"], families["square"], baselines)
        for label, optimizer in (("joint_oldbc", new_milp.solve_legacy_bc_milp),
                                 ("joint_bc", new_milp.solve_bc_milp)):
            families[label], logs[label], costs[label] = risk_family(domains,
                objectives["joint"], families["square"], baselines, optimizer=optimizer,
                extra_plans=families["ranked"])
    if "fixed" in objectives:
        families["fixed_bc"], logs["fixed_bc"], costs["fixed_bc"] = risk_family(
            domains, objectives["fixed"], families["square"], baselines,
            optimizer=new_milp.solve_legacy_bc_milp)
    record.update(planning_logs=logs, planning_seconds=costs)
    powers = {"none": np.zeros_like(constant)}
    for name, x in baselines.items():
        powers[name] = domains["fixed_charge"].decode(x)["u_kw"]
        record["methods"][name] = dict(domain="fixed_charge", operation_x=x,
            operation=domains["fixed_charge"].check(x), decoded=domains["fixed_charge"].decode(x))
    for family, plans in families.items():
        for domain_name, x in plans.items():
            method = family+"__"+domain_name
            powers[method] = domains[domain_name].decode(x)["u_kw"]
            record["methods"][method] = dict(domain=domain_name, operation_x=x,
                operation=domains[domain_name].check(x), decoded=domains[domain_name].decode(x))
    record["methods"]["none"] = dict(domain=None, operation_task_feasible=False)
    actions = {}
    integrals = {}
    for method, power in powers.items():
        key = nested.action_key(power)
        record["methods"][method].update(action_key=key, power_kw=power)
        if key in actions:
            continue
        row = dict(power_kw=power, predictions={}, actual={})
        for model in CONFIG["posterior_models"]:
            if model in ("fixed", "joint") and model not in objectives:
                continue
            if model not in ("fixed", "joint") and "joint" not in objectives:
                continue
            try:
                forecast = _forecast(objectives, contexts.get("joint"), power, model)
                checked = probability.evaluate_bank(forecast["bank"], independent,
                    abstention_cost=CONFIG["abstention_cost"])
                integrals[(key, model)] = checked
                # Full future matrices are compact relative to training state matrices.
                row["predictions"][model] = dict(forecast=forecast,
                    independent_score=score_summary(checked))
            except Exception as exc:
                row.setdefault("prediction_failures", {})[model] = type(exc).__name__+": "+str(exc)
        actions[key] = row
    paired = {}
    comparisons = [(f+"__"+d, "square__"+d)
        for f in ("ranked", "joint_oldbc", "joint_bc", "fixed_bc", "square_budget")
        if f in families for d in domains]
    comparisons += [("joint_bc__"+d, "ranked__"+d) for d in domains if "joint_bc" in families]
    comparisons += [("joint_bc__"+d, "joint_oldbc__"+d) for d in domains if "joint_bc" in families]
    for left, right in comparisons:
        lk_, rk_ = record["methods"][left]["action_key"], record["methods"][right]["action_key"]
        for model in objectives:
            if (lk_, model) in integrals and (rk_, model) in integrals:
                paired[left+"_minus_"+right+"__"+model] = probability.compare_evaluations(
                    integrals[(lk_, model)], integrals[(rk_, model)])
    record["independent_paired"] = paired
    print("PLANNED", spec["phase"], spec["index"], len(actions),
        round(perf_counter()-began, 2), flush=True)
    truth_index = next(i for i, c in enumerate(case["bank"])
        if c.supports == case["truth"]["candidate"].supports)
    white = stream_rng(spec["seed"], 4000).standard_normal(
        (CONFIG["actual_noise_repeats"], dimension))
    truths, first = {}, {}
    for regime in CONFIG["regimes"]:
        replay = cases.make_case(spec, regime=regime)
        for key, row in actions.items():
            physical, truth_inputs = _actual_physical(case, row["power_kw"], replay)
            truths[regime] = truth_inputs
            if spec["legacy_audit"]:
                first[(regime,key)] = first_observation(physical["physical"], case["fm"], white[0])
            evaluations = {}
            for model, predicted in row["predictions"].items():
                evaluations[model] = evaluate_actual(predicted["forecast"]["bank"],
                    physical["mean"], white, true_index=truth_index)
            row["actual"][regime] = dict(models=evaluations,
                clean_whitened=physical["mean"], noise_seed=stream_seed(spec["seed"], 4000),
                ac_equation_residual=float(physical["physical"]["ac_equation_residual"]),
                power_balance_residual=float(physical["physical"]["power_balance_residual"]),
                v_min=float(np.min(physical["physical"]["v"])),
                v_max=float(np.max(physical["physical"]["v"])))
    if spec["legacy_audit"]:
        record["legacy_joint_audit"] = legacy_audit(case, fits, first, record["methods"],
            int(np.random.SeedSequence(stream_seed(spec["seed"], 6000)).generate_state(1)[0]))
    zero_key = record["methods"]["none"]["action_key"]
    zero = actions[zero_key]
    if "joint" in contexts:
        for key, row in actions.items():
            if "joint" not in row["predictions"] or "joint" not in zero["predictions"]:
                continue
            a = row["predictions"]["joint"]["forecast"]
            b = zero["predictions"]["joint"]["forecast"]
            increments = []
            for index, (candidate, F, G) in enumerate(zip(contexts["joint"]["rows"],
                    a["joint_jacobian"], b["joint_jacobian"])):
                delta = np.asarray(F)-np.asarray(G)
                C, d = candidate["joint_covariance"], candidate["line_dimension"]
                covariance = delta @ C @ delta.T
                line = delta[:,:d] @ C[:d,:d] @ delta[:,:d].T
                state = delta[:,d:] @ C[d:,d:] @ delta[:,d:].T
                cross = covariance-line-state
                increments.append(dict(candidate=candidate["candidate"].name,
                    mean=a["bank"]["means"][0,index]-b["bank"]["means"][0,index],
                    variance=np.maximum(np.diag(covariance),0),
                    line_variance=np.diag(line),state_variance=np.diag(state),
                    cross_variance=np.diag(cross)))
            row["response_increment"] = dict(predictions=increments,
                meter_noise_included=False, shared_line_and_background=True,
                true_index=int(truth_index), actual={regime:
                    row["actual"][regime]["clean_whitened"]-zero["actual"][regime]["clean_whitened"]
                    for regime in CONFIG["regimes"]})
    record.update(actions=actions, true_name=case["bank"][truth_index].name,
        true_lines=dict(r=case["truth"]["r"], x=case["truth"]["x"]),
        true_future_inputs_evaluation_only=truths,
        objective_costs={name: dict(cache_calls=o.calls, ac_states=o.ac_states,
            seconds=o.seconds, included_in_family_seconds=True,
            actions={key:dict(seconds=value["seconds"],
                forecast_seconds=value["forecast"].get("seconds"),
                AC_states=value["forecast"].get("AC_states")) for key,value in o.cache.items()})
            for name, o in objectives.items()},
        status="complete" if not record["failures"] else "inference_failure",
        seconds=perf_counter()-began)
    return record


def main():
    import faulthandler
    faulthandler.dump_traceback_later(300, repeat=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("development", "calibration", "evaluation"),
        default="development")
    parser.add_argument("--indices", nargs="*", type=int)
    parser.add_argument("--worker-id", default="interactive")
    args = parser.parse_args()
    root = Path(args.output)
    frozen = freeze(root)
    for spec in cases.specifications(args.phase):
        if args.indices is not None and spec["index"] not in args.indices:
            continue
        path = root/args.phase/("history_%03d.json.gz" % spec["index"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            continue
        try:
            result = run(spec)
        except Exception as exc:
            result = dict(spec=spec, status="execution_failure",
                failure=type(exc).__name__+": "+str(exc), traceback=traceback.format_exc())
        result["policy_id"] = frozen["policy_id"]
        save_record(path, result)
        io.write_json(root/("checkpoint_"+args.phase+"_"+args.worker_id+".json"), dict(phase=args.phase,
            last_index=spec["index"], last_status=result["status"], policy_id=frozen["policy_id"],
            files=[p.name for p in sorted((root/args.phase).glob("history_*.json.gz"))]))
        print("SAVED", args.phase, spec["index"], result["status"], flush=True)


if __name__ == "__main__":
    main()
