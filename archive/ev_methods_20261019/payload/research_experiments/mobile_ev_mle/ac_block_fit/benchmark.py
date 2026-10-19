"""History-only AC numerical benchmarks, with isolated workers and timeouts.

No fit runs on import, export, or plan.  A run requires the explicit ``run``
subcommand.  Source raw files are lexically projected onto an allowlist before
JSON values are decoded: evaluation, true lines, future metering and future
forecasts are not passed to numerical workers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep
import traceback


ALGORITHMS = ("ref_sparse", "ref_jac", "block_reference", "block_jac",
    "block_reference_gap", "block_jac_gap", "block_gpu", "fixed_ref", "fixed_2d", "fixed_batch")
DEFAULT_ALGORITHMS=("ref_sparse","block_reference","block_jac","fixed_ref","fixed_2d")
START_MODES = ("prefix_linear", "historical_warm")
NETWORKS = ("cigre_residential", "paper11", "paper15", "case33")
OBS_FIELDS = ("p", "q", "v", "root_v", "master_p", "master_q", "source_p", "source_q")
SOURCE_PROJECTION = {
    "spec": None, "observed_history": None, "history_meter": None,
    "candidate_structures": None, "fit_options": None, "freeze_identity": None,
    "planning": {"history": None},
}
PACKAGE_SCHEMA = "history_only_ac_numerics_v1"
STAGE = Path(__file__).resolve().parents[3]


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _serial(value):
    if isinstance(value, dict):
        return {str(k): _serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serial(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _serial(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _bytes(value):
    return (json.dumps(_serial(value), ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False)+"\n").encode()


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = _bytes(value)
    if path.suffix == ".gz":
        data = gzip.compress(data, compresslevel=3, mtime=0)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_bytes(data)
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            sleep(min(.05*2**attempt, .5))


def _read(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def _space(text, position):
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _skip_value(text, position):
    """Skip an unselected JSON value lexically, without decoding its numbers."""
    position = _space(text, position)
    if position >= len(text):
        raise ValueError("missing JSON value")
    first = text[position]
    if first in "[{":
        stack, string, escaped = [], False, False
        for index in range(position, len(text)):
            character = text[index]
            if string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    string = False
            elif character == '"':
                string = True
            elif character in "[{":
                stack.append(character)
            elif character in "]}":
                if not stack or (stack.pop(), character) not in (("[", "]"), ("{", "}")):
                    raise ValueError("unbalanced excluded JSON container")
                if not stack:
                    return index+1
        raise ValueError("unterminated excluded JSON container")
    if first == '"':
        escaped = False
        for index in range(position+1, len(text)):
            character = text[index]
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                return index+1
        raise ValueError("unterminated excluded JSON string")
    index = position
    while index < len(text) and text[index] not in ",}] \t\r\n":
        index += 1
    if index == position:
        raise ValueError("empty excluded JSON scalar")
    return index


def _project_object(text, position, fields):
    position = _space(text, position)
    if position >= len(text) or text[position] != "{":
        raise ValueError("expected a JSON object for history projection")
    decoder, result = json.JSONDecoder(), {}
    position += 1
    while True:
        position = _space(text, position)
        if position >= len(text):
            raise ValueError("unterminated projected JSON object")
        if text[position] == "}":
            position += 1
            break
        key, position = decoder.raw_decode(text, position)
        if not isinstance(key, str):
            raise ValueError("JSON object key must be a string")
        position = _space(text, position)
        if text[position] != ":":
            raise ValueError("missing projected JSON colon")
        position = _space(text, position+1)
        if key in fields:
            if key in result:
                raise ValueError("duplicate selected JSON field: "+key)
            if fields[key] is None:
                result[key], position = decoder.raw_decode(text, position)
            else:
                result[key], position = _project_object(text, position, fields[key])
        else:
            position = _skip_value(text, position)
        position = _space(text, position)
        if position >= len(text):
            raise ValueError("unterminated projected JSON object")
        if text[position] == ",":
            position += 1
        elif text[position] != "}":
            raise ValueError("invalid delimiter after projected JSON value")
    missing = set(fields)-set(result)
    if missing:
        raise ValueError("missing selected JSON fields: "+str(sorted(missing)))
    return result, position


def project_source(path):
    """Read only declared historical JSON values from a source raw container."""
    path = Path(path)
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    text = data.decode("utf8")
    result, end = _project_object(text, 0, SOURCE_PROJECTION)
    if _space(text, end) != len(text):
        raise ValueError("trailing data after source raw object")
    return result


def _finite_scalar(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _valid_state(state, n_edges, history_length):
    if not isinstance(state, dict):
        return False
    for key, length in (("r", n_edges), ("x", n_edges),
                        ("source_p", history_length), ("source_q", history_length)):
        values = state.get(key)
        if not isinstance(values, list) or len(values) != length or not all(_finite_scalar(v) for v in values):
            return False
    return True


def export_history(source_directory, output, *, networks=NETWORKS):
    """Export date-0, no-source historical data; never start numerical fitting."""
    source_directory, output = Path(source_directory), Path(output)
    items, missing = {}, []
    for network in networks:
        if network not in NETWORKS:
            raise ValueError("unregistered network: "+network)
        path = source_directory/("history_000_"+network+"_none.json.gz")
        if not path.exists():
            missing.append(dict(network=network, reason="source development raw not yet saved"))
            continue
        projected = project_source(path)
        spec = projected["spec"]
        if spec.get("network") != network or spec.get("phase") != "development" or \
                spec.get("cluster_index") != 0 or spec.get("source_mode") != "none":
            raise ValueError("source specification differs from declared numerical development condition")
        history = projected["observed_history"]
        if set(history) != set(OBS_FIELDS):
            raise ValueError("historical observation fields differ from the fixed schema")
        history_length = len(history["p"])
        if not history_length:
            raise ValueError("empty historical record")
        candidates = projected["candidate_structures"]
        fits = projected["planning"]["history"]["candidates"]
        eligible = []
        for candidate in candidates:
            name = candidate["name"]; fit = fits.get(name, {})
            if _finite_scalar(fit.get("loss")) and _valid_state(fit.get("state"),
                    len(candidate["supports"]), history_length):
                eligible.append((candidate, fit))
        if not eligible:
            missing.append(dict(network=network, reason="no finite arrived-history candidate/state"))
            continue
        best = min(eligible, key=lambda pair: (float(pair[1]["loss"]), pair[0]["name"]))
        slow = max(eligible, key=lambda pair: (float(pair[1].get("seconds", 0.)), pair[0]["name"]))
        chosen = {}
        for role, (candidate, fit) in (("lowest_history_loss", best), ("slow_history_fit", slow)):
            name = candidate["name"]
            if name in chosen:
                chosen[name]["roles"].append(role)
            else:
                chosen[name] = dict(candidate=candidate, roles=[role],
                    historical_state={k: fit["state"][k] for k in ("r", "x", "source_p", "source_q")},
                    historical_fit_success=bool(fit.get("success")),
                    historical_loss=float(fit["loss"]),
                    historical_seconds=float(fit.get("seconds", 0.)))
        history_content = dict(observed_history=history, history_meter=projected["history_meter"],
            candidate_structures=candidates, fit_options=projected["fit_options"],
            selected_candidates=list(chosen.values()))
        items[network] = dict(**history_content, spec=spec, history_length=history_length,
            source_raw_file=str(path.resolve()), source_raw_sha256=_digest(path),
            frozen_source_identity=projected["freeze_identity"],
            history_content_sha256=sha256(_bytes(history_content)).hexdigest(),
            selection_rule="lowest finite historical loss and most historical fit seconds; ties by name",
            known_native_topology_used=False, scientific_effectiveness_evaluated=False)
    payload = dict(schema=PACKAGE_SCHEMA, items=items, missing=missing,
        data_boundary=dict(selected_source_fields=SOURCE_PROJECTION,
            excluded_value_decoding=False, numerical_workers_receive_only_this_package=True,
            future_measurements_used=False, true_lines_used=False, future_predictions_used=False),
        created_utc=_utc(), fits_started=0)
    payload["payload_sha256"] = sha256(_bytes(payload)).hexdigest()
    if output.exists():
        raise FileExistsError("retain existing whitelist; choose a new output: "+str(output))
    _write(output, payload)
    return payload


def load_history_package(path):
    payload = _read(path)
    if payload.get("schema") != PACKAGE_SCHEMA:
        raise ValueError("not a history-only numerical package")
    expected = payload.get("payload_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if expected != sha256(_bytes(unsigned)).hexdigest():
        raise ValueError("history-only package content hash differs")
    return payload


def make_tasks(package, *, networks=None, prefixes=(12, 48, 192),
               algorithms=DEFAULT_ALGORITHMS, starts=("prefix_linear",), repeats=3,
               candidate_roles=("lowest_history_loss",), inner_tol=1e-10):
    """Declare an alternating-order matrix, without running any fit."""
    if repeats < 1 or int(repeats) != repeats:
        raise ValueError("positive integer repeats required")
    if set(algorithms)-set(ALGORITHMS) or set(starts)-set(START_MODES):
        raise ValueError("unknown algorithm or start mode")
    if not prefixes or any(int(p) != p or p < 1 for p in prefixes):
        raise ValueError("prefixes must be positive integers")
    if inner_tol <= 0 or not math.isfinite(inner_tol):
        raise ValueError("inner_tol must be positive and finite")
    networks = tuple(package["items"]) if networks is None else tuple(networks)
    unknown = set(networks)-set(package["items"])
    if unknown:
        raise ValueError("networks absent from history-only package: "+str(sorted(unknown)))
    tasks = []
    for network in networks:
        item = package["items"][network]
        for candidate in item["selected_candidates"]:
            if not set(candidate["roles"]) & set(candidate_roles):
                continue
            for prefix in prefixes:
                if prefix > item["history_length"]:
                    raise ValueError("prefix exceeds available historical length")
                for start in starts:
                    for repeat in range(repeats):
                        order = tuple(algorithms) if repeat % 2 == 0 else tuple(reversed(algorithms))
                        for algorithm in order:
                            task = dict(network=network, candidate_name=candidate["candidate"]["name"],
                                prefix=int(prefix), start=start, repeat=int(repeat), algorithm=algorithm,
                                inner_tol=float(inner_tol), seed=20421800,
                                package_sha256=package["payload_sha256"],
                                history_content_sha256=item["history_content_sha256"])
                            task["task_id"] = sha256(_bytes(task)).hexdigest()[:20]
                            tasks.append(task)
    return tasks


def _task_identity(task):
    return sha256(_bytes(task)).hexdigest()


def _reconstruct(item, task):
    import numpy as np
    from .. import likelihood as lk
    from ..information_context import meter_from_record
    from ..ev_fixed_train_fit import subset_observations, subset_model
    from ..reference_benchmark.fitting import history_linear_initial
    if task["history_content_sha256"] != item["history_content_sha256"]:
        raise ValueError("task history content differs")
    selected = next((row for row in item["selected_candidates"]
                     if row["candidate"]["name"] == task["candidate_name"]), None)
    if selected is None:
        raise ValueError("task candidate not in the historical selection")
    declaration = selected["candidate"]
    candidate = lk.RadialCandidate.from_supports(declaration["name"], declaration["n"], declaration["supports"])
    full = lk.Observations(**item["observed_history"])
    full_meter = meter_from_record(item["history_meter"])
    prefix = int(task["prefix"])
    if prefix < 1 or prefix > full.t:
        raise ValueError("invalid historical prefix")
    data = subset_observations(full, slice(0, prefix))
    meter = subset_model(full_meter, full, slice(0, prefix))
    options = dict(item["fit_options"])
    options["multistart"] = 1
    if task["start"] == "prefix_linear":
        initial, record = history_linear_initial(candidate, data, meter,
            **{key: options[key] for key in ("r_bounds", "x_bounds", "ratio_bounds",
                                            "source_p_bounds", "source_q_bounds")})
        record.update(uses_full_history_fitted_state=False, prefix_time_count=prefix,
                      temporal_interpretation="initialization from this prefix only")
    elif task["start"] == "historical_warm":
        state = selected["historical_state"]
        initial = dict(r=np.asarray(state["r"], float), x=np.asarray(state["x"], float),
            source_p=np.asarray(state["source_p"][:prefix], float),
            source_q=np.asarray(state["source_q"][:prefix], float))
        record = dict(method="saved_full_history_fit_reoptimization",
            uses_full_history_fitted_state=True, training_time_count=full.t, prefix_time_count=prefix,
            temporal_interpretation="numerical warm reoptimization; not a prefix-only online estimate",
            original_history_fit_success=selected["historical_fit_success"])
    else:
        raise ValueError("unknown initialization")
    return candidate, data, meter, options, initial, record


def audit_fit(candidate, data, meter, options, fitted, *, fixed_lines):
    """Independent full-AC residual and block-gradient replay at returned state."""
    import numpy as np
    from ..ev_rx_fit import _RxInputProblem
    from ..reference_benchmark.fitting import _SourceInputProblem
    from .blocks import problem_blocks, gradient, projected_gradient
    state = dict(r=fitted.r, x=fitted.x, source_p=fitted.source_p, source_q=fitted.source_q)
    if not all(np.isfinite(value).all() for value in state.values()):
        return dict(available=False, reason="nonfinite retained state")
    domain = {key: options[key] for key in ("r_bounds", "x_bounds", "ratio_bounds",
        "source_p_bounds", "source_q_bounds", "voltage_bounds")}
    if fixed_lines:
        problem = _SourceInputProblem(candidate, data, meter, initial=state,
            fixed_r=fitted.r, fixed_x=fitted.x, call_counter=None, **domain)
        vector = np.c_[fitted.source_p, fitted.source_q].ravel()
    else:
        problem = _RxInputProblem(candidate, data, meter, initial=state,
                                  call_counter=None, **domain)
        r, alpha = problem.mapping.inverse(fitted.r, fitted.x)
        full = np.r_[r, alpha, np.c_[fitted.source_p, fitted.source_q].ravel()]
        vector = full[problem.free]
    residual = problem.evaluate(vector)
    line, source = problem_blocks(problem, vector)
    grad = gradient(line, source, residual)
    rows = residual.reshape(data.t, data.n+2)
    phy = problem.base.cache["physical"]
    voltage = abs(phy["voltage"])
    rebuilt_r, rebuilt_x, _ = problem.unpack(vector)
    source_feasible = bool(np.all(vector >= problem.lower-1e-12) and
                           np.all(vector <= problem.upper+1e-12))
    return dict(available=True, loss=float(residual@residual),
        reported_loss_absolute_difference=float(abs(fitted.loss-residual@residual)),
        loglik=-.5*float(residual@residual+problem.base.normalizer),
        reported_loglik_absolute_difference=float(abs(fitted.loglik+.5*(residual@residual+problem.base.normalizer))),
        channel_sse=dict(v=float(np.sum(rows[:, :data.n]**2)),
                         master_p=float(np.sum(rows[:, data.n]**2)),
                         master_q=float(np.sum(rows[:, data.n+1]**2))),
        reference_scaled_projected_gradient=projected_gradient(vector, grad,
            problem.lower, problem.upper, problem.scale),
        gradient_inf=float(np.max(abs(grad), initial=0.)),
        source_and_optimizer_box_feasible=source_feasible,
        line_domain_feasible=bool(problem.mapping.contains(fitted.r, fitted.x)),
        reconstruction_rx_max_abs=float(max(np.max(abs(rebuilt_r-fitted.r), initial=0.),
                                           np.max(abs(rebuilt_x-fitted.x), initial=0.))),
        physical_voltage_feasible=bool(np.min(voltage) >= options["voltage_bounds"][0] and
                                      np.max(voltage) <= options["voltage_bounds"][1]),
        voltage_range=[float(np.min(voltage)), float(np.max(voltage))],
        ac_equation_residual=float(phy["ac_equation_residual"]),
        power_balance_residual=float(phy["power_balance_residual"]),
        residual_count=int(residual.size), shared_free_parameters=int(line.shape[-1]),
        source_parameters=2*data.t)


def _source_fingerprints():
    names = ("benchmark.py", "blocks.py", "fixed_source.py", "linear.py", "nonlinear.py",
             "reference_variants.py", "gpu.py", "batched_source.py")
    paths = [Path(__file__).with_name(name) for name in names]
    paths.extend(Path(__file__).parent.parent/relative for relative in
        ("likelihood.py", "ev_rx_fit.py", "ev_fixed_train_fit.py", "reference_benchmark/fitting.py",
         "reference_benchmark/derivatives.py"))
    return {str(path.relative_to(STAGE)): _digest(path) for path in paths if path.exists()}


def run_worker(whitelist, task):
    """Run exactly one declared historical task in the caller's isolated process."""
    started = perf_counter()
    record = dict(task=task, task_sha256=_task_identity(task), started_utc=_utc(),
                  pid=os.getpid(), status="started", future_used=False, truth_used=False)
    try:
        package = load_history_package(whitelist)
        if task["package_sha256"] != package["payload_sha256"]:
            raise ValueError("task package hash differs")
        if task["algorithm"] not in ALGORITHMS:
            raise ValueError("unknown worker algorithm")
        item = package["items"][task["network"]]
        setup = perf_counter()
        candidate, data, meter, options, initial, init_record = _reconstruct(item, task)
        setup_seconds = perf_counter()-setup
        record.update(initialization=init_record, fit_options=options,
            initial_state_sha256=sha256(_bytes(initial)).hexdigest(),
            input_whitelist_sha256=_digest(whitelist), source_hashes=_source_fingerprints(),
            fixed_lines=task["algorithm"].startswith("fixed_"),
            fixed_line_anchor=("prefix_linear_initial" if task["start"] == "prefix_linear"
                               else "full_history_trained_lines"))
        algorithm = task["algorithm"]
        if algorithm in ("ref_sparse", "fixed_ref"):
            from ..reference_benchmark.fitting import fit_rx_inputs as fit
            extra = {}
        elif algorithm == 'ref_jac':
            from .reference_variants import fit_reference_jac as fit
            extra = dict(inner_lsmr_tol=float(task['inner_tol']))
        elif algorithm.startswith('block_'):
            from .nonlinear import fit_rx_blocks as fit
            extra = dict(scale_mode="reference" if algorithm.startswith('block_reference') else "jac",
                         inner_tol=float(task["inner_tol"]),
                         inner_acceptance='certified_gap' if algorithm.endswith('_gap') else 'kkt')
            if algorithm=='block_gpu':
                extra.update(backend='gpu',gpu_dependency_path=os.environ.get('TOPO_CUPY_DEPENDENCY_PATH'))
        elif algorithm=='fixed_batch':
            from .batched_source import fit_batched_sources as fit
            extra={}
        else:
            from .fixed_source import fit_fixed_sources as fit
            extra = {}
        if algorithm.startswith("fixed_"):
            extra.update(fixed_r=initial["r"], fixed_x=initial["x"])
        counter = {}; began = perf_counter()
        fitted = fit(candidate, data, meter, initial=initial, seed=int(task["seed"]),
                     call_counter=counter, **options, **extra)
        fitting_seconds = perf_counter()-began
        audit_began = perf_counter()
        try:
            audit = audit_fit(candidate, data, meter, options, fitted,
                              fixed_lines=algorithm.startswith("fixed_"))
        except Exception as error:
            audit = dict(available=False, error=type(error).__name__+": "+str(error),
                         traceback=traceback.format_exc())
        import numpy as np
        import scipy
        state = dict(r=fitted.r, x=fitted.x, source_p=fitted.source_p, source_q=fitted.source_q)
        record.update(status="returned", solver_success=bool(fitted.success),
            fit=fitted.summary(), attained_state=state, counters=counter, audit=audit,
            timings=dict(initialization_seconds=setup_seconds, fit_seconds=fitting_seconds,
                         audit_seconds=perf_counter()-audit_began),
            environment=dict(python=sys.version, executable=sys.executable,
                numpy=np.__version__, scipy=scipy.__version__,
                threads={name: os.environ.get(name) for name in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}),
            budget_scope=("max_nfev per period per restart" if algorithm in ("fixed_2d","fixed_batch")
                          else "max_nfev for one full-history residual-vector optimization"),
            optimizer_success_is_global_certificate=False,
            fitted_history_prefix=data.t)
    except Exception as error:
        record.update(status="exception", exception=type(error).__name__+": "+str(error),
                      traceback=traceback.format_exc())
    record.update(finished_utc=_utc(), wall_seconds=perf_counter()-started)
    return _serial(record)


def _worker_command(whitelist, task_file, result_file):
    return [sys.executable, "-m", "research_experiments.mobile_ev_mle.ac_block_fit.benchmark",
            "worker", "--whitelist", str(whitelist), "--task", str(task_file),
            "--result", str(result_file)]


def _terminate_own_child(process):
    """Only terminate the exact Popen handle created for this numerical task."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3.)
    except subprocess.TimeoutExpired:
        process.kill(); process.wait(timeout=3.)


def run_task_subprocess(whitelist, task, output, *, timeout_seconds):
    """Bound one owned worker; retain timeouts rather than replace their scores."""
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise ValueError("positive finite subprocess timeout required")
    output = Path(output); job = output/task["task_id"]
    job.mkdir(parents=True, exist_ok=True)
    result_path = job/"result.json"
    if result_path.exists():
        saved = _read(result_path)
        if saved.get("task_sha256") != _task_identity(task):
            raise ValueError("existing task result has a different declaration")
        return saved
    task_file, worker_result = job/"task.json", job/"worker_result.json"
    _write(task_file, task)
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
        PYTHONPATH=str(STAGE)+os.pathsep+str(STAGE/"rnj_wzzt_core"))
    record = dict(task=task, task_sha256=_task_identity(task), timeout_seconds=float(timeout_seconds),
                  started_utc=_utc(), status="started", owned_child_only=True)
    began = perf_counter()
    with (job/"stdout.log").open("w", encoding="utf8") as stdout, \
            (job/"stderr.log").open("w", encoding="utf8") as stderr:
        process = subprocess.Popen(_worker_command(Path(whitelist).resolve(),
            task_file.resolve(), worker_result.resolve()), cwd=STAGE, env=environment,
            stdout=stdout, stderr=stderr)
        record["pid"] = process.pid
        _write(job/"started.json", record)
        try:
            exit_code = process.wait(timeout=timeout_seconds)
            record["exit_code"] = exit_code
            if exit_code == 0 and worker_result.exists():
                worker = _read(worker_result)
                if worker.get("task_sha256") != _task_identity(task):
                    raise ValueError("worker result declaration hash differs")
                record.update(status=worker["status"], worker=worker)
            else:
                record.update(status="worker_failed", result_file_present=worker_result.exists())
        except subprocess.TimeoutExpired:
            _terminate_own_child(process)
            record.update(status="timeout", exit_code=process.returncode,
                          retained_in_denominator=True, loss=None)
        finally:
            # Also release this exact child if result parsing itself raised.
            _terminate_own_child(process)
    record.update(finished_utc=_utc(), process_wall_seconds=perf_counter()-began)
    _write(result_path, record)
    return record


def run_matrix(whitelist, tasks, output, *, joint_timeout=120., fixed_timeout=30.):
    """Sequential outer tasks preserve CPU headroom for the frozen old batch."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    lock = output/"benchmark.lock"
    descriptor = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    records = []
    try:
        _write(output/"plan.json", dict(whitelist=str(Path(whitelist).resolve()), tasks=tasks,
            joint_timeout=joint_timeout, fixed_timeout=fixed_timeout,
            created_utc=_utc(), simultaneous_numerical_workers=1))
        for position, task in enumerate(tasks):
            timeout = fixed_timeout if task["algorithm"].startswith("fixed_") else joint_timeout
            result = run_task_subprocess(whitelist, task, output/"tasks", timeout_seconds=timeout)
            records.append(result)
            _write(output/"progress.json", dict(completed=len(records), total=len(tasks),
                status="running", updated_utc=_utc(), last_task=task,
                outcomes=[dict(task_id=r["task"]["task_id"], status=r["status"]) for r in records]))
            print("%d/%d %s %s %s H%d %s" % (position+1, len(tasks),
                task["network"], task["candidate_name"], task["algorithm"],
                task["prefix"], result["status"]), flush=True)
        _write(output/"completed.json", dict(status="numerical_matrix_complete", count=len(records),
            failed_and_timeout_retained=True, finished_utc=_utc(),
            outcomes=[dict(task_id=r["task"]["task_id"], status=r["status"]) for r in records]))
    finally:
        if lock.exists() and lock.read_text() == str(os.getpid()):
            lock.unlink()
    return records


def _csv(value, convert=str):
    return tuple(convert(part) for part in value.split(",") if part)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="export historical allowlist; no fits")
    export.add_argument("--source-directory", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--networks", default=",".join(NETWORKS))
    for name in ("plan", "run"):
        choice = sub.add_parser(name, help="declare tasks" if name == "plan" else "explicitly run declared tasks")
        choice.add_argument("--whitelist", required=True, type=Path)
        choice.add_argument("--output", required=True, type=Path)
        choice.add_argument("--networks", default=None)
        choice.add_argument("--prefixes", default="12,48,192")
        choice.add_argument("--algorithms", default=",".join(DEFAULT_ALGORITHMS))
        choice.add_argument("--starts", default="prefix_linear")
        choice.add_argument("--candidate-roles", default="lowest_history_loss")
        choice.add_argument("--repeats", type=int, default=3)
        choice.add_argument("--inner-tol", type=float, default=1e-10)
        choice.add_argument("--joint-timeout", type=float, default=120.)
        choice.add_argument("--fixed-timeout", type=float, default=30.)
    worker = sub.add_parser("worker", help="one isolated numerical task")
    worker.add_argument("--whitelist", required=True, type=Path)
    worker.add_argument("--task", required=True, type=Path)
    worker.add_argument("--result", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "export":
        payload = export_history(args.source_directory, args.output, networks=_csv(args.networks))
        print(json.dumps(dict(path=str(args.output), networks=list(payload["items"]),
            missing=payload["missing"], fits_started=0), ensure_ascii=False))
    elif args.command == "worker":
        _write(args.result, run_worker(args.whitelist, _read(args.task)))
    else:
        payload = load_history_package(args.whitelist)
        tasks = make_tasks(payload, networks=None if args.networks is None else _csv(args.networks),
            prefixes=_csv(args.prefixes, int), algorithms=_csv(args.algorithms),
            starts=_csv(args.starts), repeats=args.repeats,
            candidate_roles=_csv(args.candidate_roles), inner_tol=args.inner_tol)
        if args.command == "plan":
            _write(args.output, dict(tasks=tasks, count=len(tasks), fits_started=0,
                joint_timeout=args.joint_timeout, fixed_timeout=args.fixed_timeout))
            print(json.dumps(dict(path=str(args.output), task_count=len(tasks), fits_started=0)))
        else:
            run_matrix(args.whitelist, tasks, args.output,
                       joint_timeout=args.joint_timeout, fixed_timeout=args.fixed_timeout)


if __name__ == "__main__":
    main()
