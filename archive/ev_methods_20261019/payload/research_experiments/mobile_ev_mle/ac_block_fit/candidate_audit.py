"""History-only multi-candidate numerical and local EV-response audit.

This driver never executes an EV action or reads a future outcome.  Three
networks reuse observed historical raw fields; case33 can use a declared
controlled native-network arithmetic fixture while its scientific raw remains
unfinished.  Candidates are selected by declared bank order, including failures.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

from . import benchmark as b


RAW_FIELDS = {key: None for key in ("spec", "observed_history", "history_meter",
    "candidate_structures", "fit_options", "freeze_identity", "config")}
METHODS = ("ref_jac", "block_jac_gap")
THRESHOLDS = dict(common_projected_gradient=1e-5, loss_atol=1e-4, loss_rtol=1e-6,
                  rx_relative=1e-3, response_relative=1e-3, gradient_relative=1e-3,
                  relative_denominator_floor=1e-12)


def _text(path):
    path = Path(path); data = path.read_bytes()
    return (gzip.decompress(data) if path.suffix == ".gz" else data).decode("utf8")


def project_raw(path):
    text = _text(path)
    result, end = b._project_object(text, 0, RAW_FIELDS)
    if b._space(text, end) != len(text):
        raise ValueError("trailing data in historical raw")
    return result


def _field_position(text, position, name):
    """Locate a named object value without decoding values of other fields."""
    position = b._space(text, position)
    if text[position] != "{":
        raise ValueError("expected object while selecting historical prepared data")
    decoder = json.JSONDecoder(); position += 1
    while True:
        position = b._space(text, position)
        if text[position] == "}":
            raise KeyError(name)
        key, position = decoder.raw_decode(text, position)
        position = b._space(text, position)
        if text[position] != ":":
            raise ValueError("missing JSON colon")
        position = b._space(text, position+1)
        if key == name:
            return position
        position = b._space(text, b._skip_value(text, position))
        if text[position] == ",":
            position += 1
        elif text[position] != "}":
            raise ValueError("missing JSON object delimiter")


def _array_prefix(text, position, count):
    position = b._space(text, position)
    if text[position] != "[":
        raise ValueError("expected historical-prefix JSON array")
    decoder, rows = json.JSONDecoder(), []
    position += 1
    for index in range(count):
        position = b._space(text, position)
        if text[position] == "]":
            raise ValueError("prepared array shorter than requested historical prefix")
        row, position = decoder.raw_decode(text, position)
        rows.append(row)
        position = b._space(text, position)
        if index < count-1:
            if text[position] != ",":
                raise ValueError("missing historical-prefix array delimiter")
            position += 1
    # The remaining future entries are not decoded or inspected.
    return rows


def prepared_history(path, history=192):
    """Extract only development window 0's first H values and public mappings."""
    text, decoder = _text(path), json.JSONDecoder()
    profiles, _ = decoder.raw_decode(text, _field_position(text, 0, "profile_names"))
    assignments_at = _field_position(text, 0, "assignments")
    assignment, _ = decoder.raw_decode(text, _field_position(text, assignments_at, "case33"))
    windows_at = _field_position(text, 0, "windows")
    development_at = _field_position(text, windows_at, "development")
    if text[development_at] != "[":
        raise ValueError("development windows must be an array")
    first_window = b._space(text, development_at+1)
    h, _ = decoder.raw_decode(text, _field_position(text, first_window, "history_rows"))
    if h != history:
        raise ValueError("requested historical prefix differs from frozen window split")
    values = {key: _array_prefix(text, _field_position(text, first_window, key), history)
              for key in ("p_relative", "q_relative", "timestamps")}
    return dict(**values, profile_names=profiles, assignment=assignment,
                history_rows=history, development_window=0, future_values_decoded=False)


def controlled_case33(prepared_path, count=4):
    """Numerical fixture: public physical network and predeclared contractions."""
    import numpy as np
    from .. import likelihood as lk
    from ..reference_benchmark import cases
    from ..reference_benchmark.config import CONFIG
    from ..reference_benchmark.representation import encode_physical_tree
    from ..ev_original_study import observation
    from ..gradient_price_study import observations_record
    from ..information_context import meter_to_record
    if count < 1:
        raise ValueError("positive controlled candidate count required")
    public = cases.native_network("case33")
    observed = prepared_history(prepared_path, int(CONFIG["H"]))
    ids = public["observation_bus_ids"]; by_bus = {row["bus_id"]: row for row in public["buses"]}
    nominal_p = np.asarray([by_bus[bus]["pd_kw"] for bus in ids])
    nominal_q = np.asarray([by_bus[bus]["qd_kvar"] for bus in ids])
    columns = [observed["profile_names"].index(observed["assignment"][str(bus)]) for bus in ids]
    p = np.asarray(observed["p_relative"])[:, columns]*nominal_p/public["base_kw"]
    q = np.asarray(observed["q_relative"])[:, columns]*nominal_q/public["base_kw"]
    encoded = encode_physical_tree(public["branches"], ids, public["root_bus"],
        public["base_kv"], public["base_kw"], public["source_bus"], name="controlled_native_case33")
    physical = lk.ac_predict(encoded.candidate, encoded.r, encoded.x, p, q,
                             source_p=0., source_q=0., root_v=1.)
    meter = cases.meter_for_nominal(nominal_p, nominal_q, public["base_kw"])
    seed, h = int(CONFIG["seed_bases"]["development"]), len(p)
    white = {key: np.random.default_rng(cases._seed(seed, 100+j)).standard_normal(
        (h, len(ids)) if key in ("p", "q", "v") else h) for j, key in enumerate(cases.CHANNELS)}
    history = observation(physical, meter, white)
    public_options, _ = cases._fit_domain(public, min(.3*float(nominal_p.sum()), 80.), CONFIG)
    n = len(ids)
    internal = [support for support in encoded.candidate.supports if 1 < len(support) < n]
    # Order and contractions are declared from public network structure alone.
    families = [tuple(internal), tuple(internal[:-1]), tuple(internal[1:]), ()]
    candidates, seen = [], set()
    for family in families:
        candidate = lk.RadialCandidate.from_supports("controlled_case33_%02d" % len(candidates), n, family)
        if candidate.supports not in seen:
            seen.add(candidate.supports)
            candidates.append(dict(name=candidate.name, n=n, supports=candidate.supports))
    if count > len(candidates):
        raise ValueError("controlled fixture declares at most four distinct candidates")
    return dict(spec=dict(network="case33", phase="development", cluster_index=0, source_mode="none"),
        observed_history=observations_record(history), history_meter=meter_to_record(meter),
        candidate_structures=candidates[:count], fit_options=public_options,
        freeze_identity="controlled_native_case33_arithmetic_fixture",
        config=dict(base_kw=public["base_kw"]),
        source_kind="controlled_native_case33_arithmetic_fixture",
        source_raw_file=str(Path(prepared_path).resolve()), source_raw_sha256=b._digest(prepared_path),
        fixture_provenance=dict(known_native_tree_used_to_construct_candidate_library=True,
            scientific_topology_identification_efficacy=False, original_scientific_denominator=False,
            simulator_native_RX_not_in_worker_package=True,
            no_future_values_decoded=True, public_network_sha256=public["physical_sha256"],
            original_H192_noise_stream=True, physical_ac_equation_residual=physical["ac_equation_residual"],
            physics_source_P_Q_zero=True, fitting_source_P_Q_both_free=True))


def prepare_inputs(source_directory, output, *, networks=b.NETWORKS, count=4,
                   prepared_path=None):
    """Export first K candidate declarations, without inspecting fit outcomes."""
    source_directory, output = Path(source_directory), Path(output)
    if output.exists():
        raise FileExistsError("retain previous numerical inputs: "+str(output))
    if count < 1:
        raise ValueError("positive candidate count required")
    items, missing = {}, []
    for network in networks:
        path = source_directory/("history_000_"+network+"_none.json.gz")
        if path.exists():
            raw = project_raw(path)
            raw.update(source_kind="observed_history_from_saved_development_raw",
                       source_raw_file=str(path.resolve()), source_raw_sha256=b._digest(path))
        elif network == "case33" and prepared_path is not None:
            raw = controlled_case33(prepared_path, count)
        else:
            missing.append(dict(network=network, reason="historical raw unavailable")); continue
        if raw["spec"].get("phase") != "development" or raw["spec"].get("cluster_index") != 0 or \
                raw["spec"].get("source_mode") != "none":
            raise ValueError("numerical audit accepts only declared development condition 0/none")
        history = raw["observed_history"]
        selected = [dict(candidate=row, roles=["declared_candidate_order"],
                         historical_fit_success=False, historical_state=None)
                    for row in raw["candidate_structures"][:count]]
        content = dict(observed_history=history, history_meter=raw["history_meter"],
            candidate_structures=raw["candidate_structures"], fit_options=raw["fit_options"],
            selected_candidates=selected)
        items[network] = dict(**content, spec=raw["spec"], history_length=len(history["p"]),
            base_kw=float(raw["config"]["base_kw"]), source_kind=raw["source_kind"],
            source_raw_file=raw["source_raw_file"], source_raw_sha256=raw["source_raw_sha256"],
            frozen_source_identity=raw["freeze_identity"],
            history_content_sha256=sha256(b._bytes(content)).hexdigest(),
            selection_rule="first K public candidate declarations in saved bank order; no fit outcome used",
            historical_fit_summary_decoded=False,
            fixture_provenance=raw.get("fixture_provenance"), scientific_effectiveness_evaluated=False)
    payload = dict(schema=b.PACKAGE_SCHEMA, items=items, missing=missing, created_utc=b._utc(),
        fits_started=0, audit_kind="multi_candidate_numerical_consistency",
        data_boundary=dict(future_measurements_used=False, true_RX_passed_to_worker=False,
            historical_fit_summary_decoded=False, source_projection=RAW_FIELDS,
            case33_fixture_labeled_separately=True), thresholds=THRESHOLDS)
    payload["payload_sha256"] = sha256(b._bytes(payload)).hexdigest()
    b._write(output, payload)
    return payload


def local_response(candidate, data, meter, state, *, base_kw, last_periods=1):
    """AC outputs and d(outputs)/d(P,Q) at the last arrived working points."""
    import numpy as np
    from .. import likelihood as lk
    from ..ev_fixed_train_fit import subset_observations, subset_model
    if base_kw <= 0 or last_periods < 1:
        raise ValueError("positive physical power base and working-point count required")
    ids = np.arange(max(0, data.t-last_periods), data.t)
    local = subset_observations(data, ids); lm = subset_model(meter, data, ids)
    p = local.p-lk._array(lm.bias_p, local.p.shape, "bias_p")
    q = local.q-lk._array(lm.bias_q, local.q.shape, "bias_q")
    sp, sq = (np.asarray(state[key], float)[ids] for key in ("source_p", "source_q"))
    physical = lk.ac_predict(candidate, state["r"], state["x"], p, q, sp, sq, local.root_v)
    derivatives = lk._response_derivatives(candidate, physical, estimate_source=True)
    columns = slice(2*candidate.m, 2*candidate.m+2*data.n)
    sigma = np.column_stack((lk._array(lm.v_sigma, p.shape, "v_sigma"),
        lk._array(lm.master_p_sigma, (len(ids),), "master_p_sigma"),
        lk._array(lm.master_q_sigma, (len(ids),), "master_q_sigma")))
    mu = np.column_stack((physical["v"]+lk._array(lm.bias_v, p.shape, "bias_v"),
                          physical["master_p"], physical["master_q"]))
    jac = np.concatenate((derivatives["v"][:, :, columns],
        derivatives["master_p"][:, None, columns], derivatives["master_q"][:, None, columns]), axis=1)
    return dict(working_periods=ids.tolist(), voltage_PQ_derivative=jac[:, :data.n],
        response_mu=mu, response_PQ_derivative_per_pu=jac,
        whitened_mu=mu/sigma, whitened_PQ_derivative_per_kw=jac/sigma[:, :, None]/base_kw,
        sigma=sigma, base_kw=float(base_kw), source_held_at_its_same_period_fitted_value=True,
        lines_held_at_training_value=True, future_used=False,
        derivative_method="implicit exact AC with all terminal P/Q right sides",
        scope="last arrived history points; local response audit, not a future performance claim")


def worker(whitelist, task):
    result = b.run_worker(whitelist, task)
    result["candidate_audit_source_sha256"] = b._digest(__file__)
    if result["status"] != "returned":
        return result
    began = perf_counter()
    try:
        package = b.load_history_package(whitelist); item = package["items"][task["network"]]
        candidate, data, meter, _, _, _ = b._reconstruct(item, task)
        result["local_response"] = local_response(candidate, data, meter, result["attained_state"],
                                                   base_kw=item["base_kw"])
    except Exception as error:
        result["local_response"] = dict(available=False, error=type(error).__name__+": "+str(error))
    result["response_audit_seconds"] = perf_counter()-began
    return b._serial(result)


@contextmanager
def _candidate_worker_dispatch():
    """Process-local supervisor routing only; no scientific fitter is patched."""
    original = b._worker_command
    def command(whitelist, task_file, result_file):
        return [sys.executable, "-m", "research_experiments.mobile_ev_mle.ac_block_fit.candidate_audit",
                "worker", "--whitelist", str(whitelist), "--task", str(task_file), "--result", str(result_file)]
    b._worker_command = command
    try:
        yield
    finally:
        b._worker_command = original


def tasks(package, *, prefixes=(24, 48), networks=None, repeats=1, inner_tol=1e-10):
    return b.make_tasks(package, networks=networks, prefixes=prefixes, algorithms=METHODS,
        starts=("prefix_linear",), repeats=repeats, candidate_roles=("declared_candidate_order",),
        inner_tol=inner_tol)


def _relative(a, reference):
    import numpy as np
    a, reference = np.asarray(a, float), np.asarray(reference, float)
    absolute = float(np.linalg.norm(a-reference))
    denominator = float(np.linalg.norm(reference))
    return dict(absolute_l2=absolute, reference_l2=denominator,
                relative=absolute/max(denominator, THRESHOLDS["relative_denominator_floor"]))


def _numerical_usable(worker):
    audit = worker.get("audit") or {}
    return bool(worker.get("status") == "returned" and audit.get("available") and
        audit.get("physical_voltage_feasible") and audit.get("line_domain_feasible") and
        audit.get("source_and_optimizer_box_feasible") and
        audit.get("reference_scaled_projected_gradient", float("inf")) <= THRESHOLDS["common_projected_gradient"])


def compare_candidate(ref, block):
    import numpy as np
    result = dict(reference_numerically_usable=_numerical_usable(ref),
        block_numerically_usable=_numerical_usable(block),
        reference_solver_success=ref.get("solver_success"), block_solver_success=block.get("solver_success"))
    if not ref.get("audit", {}).get("available") or not block.get("audit", {}).get("available"):
        return dict(**result, paired_state_available=False)
    loss_ref, loss_new = ref["audit"]["loss"], block["audit"]["loss"]
    tolerance = THRESHOLDS["loss_atol"]+THRESHOLDS["loss_rtol"]*abs(loss_ref)
    ref_state, new_state = ref["attained_state"], block["attained_state"]
    rx = _relative(np.r_[new_state["r"], new_state["x"]], np.r_[ref_state["r"], ref_state["x"]])
    result.update(paired_state_available=True, reference_loss=loss_ref, block_loss=loss_new,
        loss_difference=loss_new-loss_ref, loglik_difference=-.5*(loss_new-loss_ref),
        loss_tolerance=tolerance, loss_agreement=abs(loss_new-loss_ref) <= tolerance,
        rx=rx, rx_agreement=rx["relative"] <= THRESHOLDS["rx_relative"],
        parameter_difference_is_true_RX_error=False)
    ra, rb = ref.get("local_response", {}), block.get("local_response", {})
    if "voltage_PQ_derivative" in ra and "voltage_PQ_derivative" in rb:
        sensitivity = _relative(rb["voltage_PQ_derivative"], ra["voltage_PQ_derivative"])
        prediction = _relative(rb["response_mu"], ra["response_mu"])
        result.update(response_derivative=sensitivity,
            response_agreement=sensitivity["relative"] <= THRESHOLDS["response_relative"],
            response_prediction=prediction)
    return result


def pair_metric(first, second):
    """Exact zero-injection derivative of a declared local AC model distance.

    D(u)=1/2 sum ||W(mu_a(u)-mu_b(u))||^2, fitted source/lines held fixed.
    Its derivative is d'*(J_a-J_b).  SQ coefficients retain the squared response
    separation per time/site.  These are local audit quantities; the production
    future-forecast G-SQ indicator must be checked separately by its own code.
    """
    import numpy as np
    from ..g_precision.coefficients import build_tasks_from_responses
    a, z = first["local_response"], second["local_response"]
    delta = np.asarray(a["whitened_mu"])-np.asarray(z["whitened_mu"])
    jac = np.asarray(a["whitened_PQ_derivative_per_kw"])-np.asarray(z["whitened_PQ_derivative_per_kw"])
    n = np.asarray(a["voltage_PQ_derivative"]).shape[1]
    distance_gradient = np.einsum("tk,tkj->tj", delta, jac)
    square = .5*np.sum(jac[:, :, :n]**2, axis=1)
    sigma = np.asarray(a["sigma"])
    response = {
        "a": dict(mean=np.asarray(a["response_mu"]),
            jacobian=np.asarray(a["response_PQ_derivative_per_pu"])[:, :, :n]/a["base_kw"]),
        "b": dict(mean=np.asarray(z["response_mu"]),
            jacobian=np.asarray(z["response_PQ_derivative_per_pu"])[:, :, :n]/z["base_kw"])}
    sq = build_tasks_from_responses(response, sigma, include_intercept=False)[0]
    aq = build_tasks_from_responses(response, sigma, include_intercept=True)[0]
    power = np.zeros((len(delta), n)); power[:, 0] = 1.
    return dict(distance=.5*float(np.sum(delta**2)),
        distance_gradient_PQ_per_kw=distance_gradient,
        distance_gradient_EV_P_per_kw=distance_gradient[:, :n],
        local_square_separation_coefficient=square,
        audit_power_kw=power, G_SQ_task=sq.record(), G_AQ_task=aq.record(),
        G_SQ_value=sq.value(power), G_AQ_value=aq.value(power),
        G_SQ_nonzero_power_gradient=2*sq.quadratic*power,
        G_AQ_nonzero_power_gradient=aq.linear+2*aq.quadratic*power,
        production_coefficients_function_used=True,
        scope="production G-SQ/G-AQ coefficient code at last arrived operating point, declared 1kW/site0 action; source/lines fixed; no future prediction")


def summarize(output, whitelist):
    from itertools import combinations
    package = b.load_history_package(whitelist)
    groups, outcomes = {}, []
    for path in sorted((Path(output)/"tasks").glob("*/result.json")):
        envelope = b._read(path); task = envelope["task"]
        key = (task["network"], task["prefix"], task["repeat"])
        groups.setdefault(key, {}).setdefault(task["algorithm"], {})[task["candidate_name"]] = envelope
        outcomes.append(dict(task=task, status=envelope["status"], worker_status=(envelope.get("worker") or {}).get("status")))
    comparisons, rankings, metrics = [], [], []
    for key, methods in sorted(groups.items()):
        network, prefix, repeat = key
        declared = [row["candidate"]["name"] for row in package["items"][network]["selected_candidates"]]
        for name in declared:
            reference = methods.get(METHODS[0], {}).get(name, {}).get("worker")
            block = methods.get(METHODS[1], {}).get(name, {}).get("worker")
            comparisons.append(dict(network=network, prefix=prefix, repeat=repeat, candidate=name,
                **(compare_candidate(reference, block) if reference and block else
                   dict(paired_state_available=False, reason="missing returned worker; timeout/failure retained"))))
        for first, second in combinations(declared, 2):
            pair = dict(network=network, prefix=prefix, repeat=repeat, candidates=[first, second])
            values = {}
            for method in METHODS:
                workers = [methods.get(method, {}).get(name, {}).get("worker") for name in (first, second)]
                if all(worker and worker.get("audit", {}).get("available") for worker in workers):
                    values[method] = [worker["audit"]["loss"] for worker in workers]
                if all(worker and "whitened_mu" in worker.get("local_response", {}) for worker in workers):
                    pair[method] = pair_metric(*workers)
            if set(values) == set(METHODS):
                lr, ln = values[METHODS[0]], values[METHODS[1]]
                tolerance = sum(THRESHOLDS["loss_atol"]+THRESHOLDS["loss_rtol"]*abs(v) for v in lr)
                dr, dn = lr[0]-lr[1], ln[0]-ln[1]
                tied = abs(dr) <= tolerance or abs(dn) <= tolerance
                rankings.append(dict(network=network, prefix=prefix, repeat=repeat, candidates=[first, second],
                    reference_loss_difference=dr, block_loss_difference=dn, pair_tolerance=tolerance,
                    numerical_tie=tied, strict_reversal=bool(not tied and dr*dn < 0)))
            else:
                rankings.append(dict(network=network, prefix=prefix, repeat=repeat, candidates=[first, second],
                    available=False, reason="candidate failure/timeout retained"))
            if all(method in pair for method in METHODS):
                for field in ("distance_gradient_EV_P_per_kw", "local_square_separation_coefficient",
                              "G_SQ_nonzero_power_gradient", "G_AQ_nonzero_power_gradient"):
                    difference = _relative(pair[METHODS[1]][field], pair[METHODS[0]][field])
                    pair[field+"_comparison"] = dict(**difference,
                        agreement=difference["relative"] <= THRESHOLDS["gradient_relative"])
            metrics.append(pair)
    result = dict(scope="multi-candidate numerical consistency; no scientific EV efficacy evaluation",
        thresholds=THRESHOLDS, outcomes=outcomes, candidate_comparisons=comparisons,
        pair_rankings=rankings, local_pair_metrics=metrics,
        failure_and_timeout_denominators_retained=True,
        production_G_SQ_coefficient_code_checked_at_nonzero_power=True,
        production_forecast_G_SQ_checked=False, numerical_local_extrema_retained=True)
    b._write(Path(output)/"candidate_summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source-directory", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--prepared", type=Path)
    prepare.add_argument("--networks", default=",".join(b.NETWORKS))
    prepare.add_argument("--candidates", type=int, default=4)
    run = sub.add_parser("run")
    run.add_argument("--whitelist", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--networks", default=None)
    run.add_argument("--prefixes", default="24,48")
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--timeout", type=float, default=60.)
    run.add_argument("--inner-tol", type=float, default=1e-10)
    single = sub.add_parser("worker")
    single.add_argument("--whitelist", required=True, type=Path)
    single.add_argument("--task", required=True, type=Path)
    single.add_argument("--result", required=True, type=Path)
    report = sub.add_parser("summarize")
    report.add_argument("--whitelist", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare_inputs(args.source_directory, args.output, networks=b._csv(args.networks),
            count=args.candidates, prepared_path=args.prepared)
        print(json.dumps(dict(networks=list(payload["items"]), missing=payload["missing"], fits_started=0)))
    elif args.command == "worker":
        b._write(args.result, worker(args.whitelist, b._read(args.task)))
    elif args.command == "summarize":
        result = summarize(args.output, args.whitelist)
        print(json.dumps(dict(outcomes=len(result["outcomes"]), comparisons=len(result["candidate_comparisons"]))))
    else:
        package = b.load_history_package(args.whitelist)
        declared = tasks(package, prefixes=b._csv(args.prefixes, int), repeats=args.repeats,
            networks=None if args.networks is None else b._csv(args.networks), inner_tol=args.inner_tol)
        b._write(args.output/"candidate_protocol.json", dict(thresholds=THRESHOLDS,
            input_sha256=b._digest(args.whitelist), methods=METHODS, tasks=declared,
            source_sha256=b._digest(__file__), planned_before_numerical_execution=True))
        with _candidate_worker_dispatch():
            b.run_matrix(args.whitelist, declared, args.output,
                         joint_timeout=args.timeout, fixed_timeout=args.timeout)
        summarize(args.output, args.whitelist)


if __name__ == "__main__":
    main()
