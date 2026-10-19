"""Isolated common-gradient numerical audit; no EV efficacy experiment.

The original CIGRE historical whitelist, candidate, parameter domain and
prefix-only initializer are reused.  Each fit runs in an owned child process.
Only this audit's temporary hooks alter termination settings, and finally
restores them.  The currently running comparison matrices are not changed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from . import benchmark


TARGET = 1e-5
ALGORITHMS = ('reference_jac_precision', 'block_jac_precision')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def _common_gradient_stop(common_scale):
    from . import nonlinear
    original = nonlinear.projected_gradient

    def common(vector, grad, lower, upper, scale):
        return original(vector, grad, lower, upper, common_scale)

    nonlinear.projected_gradient = common
    try:
        yield
    finally:
        nonlinear.projected_gradient = original


@contextmanager
def _reference_precision(target, actual_calls):
    from ..reference_benchmark import fitting
    original = fitting.least_squares

    def precise(*args, **kwargs):
        kwargs.update(x_scale='jac', ftol=None, xtol=None, gtol=target)
        actual_calls.append(dict(x_scale=kwargs['x_scale'], ftol=kwargs['ftol'],
            xtol=kwargs['xtol'], gtol=kwargs['gtol'], max_nfev=kwargs['max_nfev'],
            tr_options=kwargs.get('tr_options')))
        return original(*args, **kwargs)

    fitting.least_squares = precise
    try:
        yield
    finally:
        fitting.least_squares = original


def worker(whitelist, task, output):
    import numpy as np
    from ..ev_rx_fit import _RxInputProblem
    from ..reference_benchmark import fitting
    from . import nonlinear

    output = Path(output)
    progress = output.with_suffix('.progress.json')
    record = dict(task=task, status='started', pid=os.getpid(), started_utc=_utc(),
                  target_common_projected_gradient=TARGET, truth_used=False, future_used=False)
    benchmark._write(progress, dict(phase='initialization', pid=os.getpid()))
    try:
        package = benchmark.load_history_package(whitelist)
        candidate, data, meter, options, initial, init_record = benchmark._reconstruct(
            package['items'][task['network']], task)
        domain = {k: options[k] for k in ('r_bounds', 'x_bounds', 'ratio_bounds',
            'source_p_bounds', 'source_q_bounds', 'voltage_bounds')}
        problem = _RxInputProblem(candidate, data, meter, initial=initial, **domain)
        common_scale = problem.scale.copy()
        record.update(input_sha256=_sha(whitelist),
            initial_state_sha256=hashlib.sha256(benchmark._bytes(initial)).hexdigest(),
            initialization=init_record, fit_options=options,
            common_scale=common_scale.tolist(),
            scale_definition='fixed declared R/alpha optimizer characteristic scale; same for both algorithms',
            source_hashes={**benchmark._source_fingerprints(),
                str(Path(__file__).relative_to(benchmark.STAGE)):_sha(__file__)},
            hooks='child-process only; finally restored; GN steps keep jac column scaling')
        actual_calls = []
        counter = {}
        began = time.perf_counter()
        benchmark._write(progress, dict(phase='fitting', pid=os.getpid(), fit_start_monotonic=began))
        if task['algorithm'] == 'reference_jac_precision':
            with _reference_precision(TARGET, actual_calls):
                fitted = fitting.fit_rx_inputs(candidate, data, meter, initial=initial,
                    seed=task['seed'], call_counter=counter, **options)
        else:
            with _common_gradient_stop(common_scale):
                fitted = nonlinear.fit_rx_blocks(candidate, data, meter, initial=initial,
                    seed=task['seed'], call_counter=counter, scale_mode='jac',
                    inner_acceptance='certified_gap', stationarity_tol=TARGET, **options)
        seconds = time.perf_counter()-began
        benchmark._write(progress, dict(phase='independent_audit', pid=os.getpid(), fit_seconds=seconds))
        audit = benchmark.audit_fit(candidate, data, meter, options, fitted, fixed_lines=False)
        physical = bool(audit.get('available') and audit.get('physical_voltage_feasible')
            and audit.get('line_domain_feasible') and audit.get('source_and_optimizer_box_feasible')
            and audit.get('ac_equation_residual', 1) < 2e-10
            and audit.get('power_balance_residual', 1) < 2e-10)
        common_pg = audit.get('reference_scaled_projected_gradient', np.inf)
        record.update(status='returned', fit_seconds=seconds, solver_success=bool(fitted.success),
            fit=fitted.summary(), audit=audit, counters=counter, actual_scipy_calls=actual_calls,
            common_gradient_pass=bool(common_pg <= TARGET), physical_pass=physical,
            precision_qualified=bool(fitted.success and common_pg <= TARGET and physical),
            state=dict(r=fitted.r, x=fitted.x, source_p=fitted.source_p, source_q=fitted.source_q))
    except Exception as error:
        record.update(status='exception', exception=repr(error), traceback=traceback.format_exc())
    record['finished_utc'] = _utc()
    benchmark._write(output, record)
    benchmark._write(progress, dict(phase='finished', pid=os.getpid(), status=record['status']))
    return record


def _run_child(whitelist, task, output, fit_timeout):
    path = Path(output)/(task['algorithm']+'_'+str(task['repeat'])+'.json')
    task_path = path.with_suffix('.task.json')
    benchmark._write(task_path, task)
    command = [sys.executable, '-m', 'research_experiments.mobile_ev_mle.ac_block_fit.precision_audit',
        'worker', '--whitelist', str(Path(whitelist).resolve()), '--task', str(task_path.resolve()),
        '--output', str(path.resolve())]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    env['PYTHONPATH'] = str(benchmark.STAGE)+os.pathsep+str(benchmark.STAGE/'rnj_wzzt_core')
    with path.with_suffix('.stdout.log').open('wb') as stdout, path.with_suffix('.stderr.log').open('wb') as stderr:
        process = subprocess.Popen(command, cwd=benchmark.STAGE, env=env,
            stdout=stdout, stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        began = time.perf_counter()
        timeout_reason = None
        while process.poll() is None:
            progress_path = path.with_suffix('.progress.json')
            try:
                progress = json.loads(progress_path.read_bytes()) if progress_path.exists() else {}
            except (ValueError, OSError):
                progress = {}
            if progress.get('phase') == 'fitting':
                if time.perf_counter()-progress['fit_start_monotonic'] > fit_timeout:
                    timeout_reason = 'fit wall-clock budget exceeded'
                    break
            elif time.perf_counter()-began > fit_timeout+30.:
                timeout_reason = 'setup/audit process safety budget exceeded'
                break
            time.sleep(.05)
        if timeout_reason is not None:
            benchmark._terminate_own_child(process)
            record = dict(status='timeout', task=task, pid=process.pid,
                timeout_seconds=fit_timeout, reason=timeout_reason,
                owned_child_only=True, precision_qualified=False,
                future_used=False, truth_used=False, finished_utc=_utc())
            benchmark._write(path, record)
            return record
        if path.exists():
            return json.loads(path.read_bytes())
        record = dict(status='exception', task=task, pid=process.pid,
            exit_code=process.returncode, exception='child exited without result',
            precision_qualified=False)
        benchmark._write(path, record)
        return record


def run(whitelist, output, *, fit_timeout=15.):
    output = Path(output)
    if output.exists():
        raise FileExistsError('retain precision audit; choose a new output directory')
    output.mkdir(parents=True)
    package = benchmark.load_history_package(whitelist)
    tasks = benchmark.make_tasks(package, networks=('cigre_residential',), prefixes=(192,),
        algorithms=('ref_jac',), starts=('prefix_linear',), repeats=1)
    if len(tasks) != 1:
        raise ValueError('expected one declared CIGRE historical candidate')
    base_task = tasks[0]
    plan = dict(status='declared_before_first_fit', created_utc=_utc(),
        purpose='same fixed optimizer-coordinate gradient numerical precision audit; no EV effect claims',
        target_common_projected_gradient=TARGET,
        scope='CIGRE residential date-0 development historical H192, no source; same whitelist/start/domain',
        common_scale='original problem.scale held fixed for independent audit and new-solver stop only',
        reference_settings=dict(x_scale='jac', ftol=None, xtol=None, gtol=TARGET,
                                LSMR='unchanged original 1e-10; maxiter unchanged'),
        block_settings=dict(scale_mode='jac', inner_acceptance='certified_gap', stationarity_tol=TARGET,
            stopping_hook='use fixed original problem.scale, preserving jac GN step scaling'),
        full_AC_acceptance_unchanged=True,
        fit_wall_timeout_seconds=fit_timeout,
        initial_pair=2, conditional_total=6,
        repeat_rule='Only if BOTH first fits are successful, physical, and common PG <= target, run two more alternating-order pairs.',
        first_order=list(ALGORITHMS), failed_and_unqualified_retained=True,
        whitelist=str(Path(whitelist).resolve()), whitelist_sha256=_sha(whitelist),
        template=base_task, source_sha256=_sha(__file__), scientific_conditions_created=0)
    benchmark._write(output/'plan.json', plan)
    records = []
    for repeat in range(3):
        order = ALGORITHMS if repeat % 2 == 0 else ALGORITHMS[::-1]
        for algorithm in order:
            task = dict(base_task, algorithm=algorithm, repeat=repeat)
            records.append(_run_child(whitelist, task, output, fit_timeout))
        if repeat == 0 and not all(row.get('precision_qualified') for row in records):
            break
    pairs = []
    for repeat in range((len(records)+1)//2):
        group = {r['task']['algorithm']:r for r in records if r['task']['repeat'] == repeat}
        old, new = group[ALGORITHMS[0]], group[ALGORITHMS[1]]
        same = bool(old.get('initial_state_sha256') and old.get('initial_state_sha256') == new.get('initial_state_sha256')
            and old.get('input_sha256') == new.get('input_sha256') and old.get('common_scale') == new.get('common_scale'))
        qualified = bool(same and old.get('precision_qualified') and new.get('precision_qualified'))
        pairs.append(dict(repeat=repeat, same_input_start_and_common_scale=same, qualified=qualified,
            old_status=old['status'], new_status=new['status'],
            old_common_pg=old.get('audit',{}).get('reference_scaled_projected_gradient'),
            new_common_pg=new.get('audit',{}).get('reference_scaled_projected_gradient'),
            old_loss=old.get('fit',{}).get('loss'), new_loss=new.get('fit',{}).get('loss'),
            old_seconds=old.get('fit_seconds'), new_seconds=new.get('fit_seconds'),
            qualified_speed_ratio=old['fit_seconds']/new['fit_seconds'] if qualified else None))
    ratios = [p['qualified_speed_ratio'] for p in pairs if p['qualified']]
    summary = dict(status='complete', finished_utc=_utc(), plan=plan,
        attempted_fits=len(records), failed_or_unqualified=sum(not r.get('precision_qualified') for r in records),
        pairs=pairs, qualified_pairs=len(ratios),
        median_qualified_speed_ratio=statistics.median(ratios) if ratios else None,
        sources_modified=False, main_matrices_modified=False,
        comparison='qualified precision pairs only; TRF gtol has its own internal scaling, common PG is checked independently')
    benchmark._write(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'worker'):
        item = sub.add_parser(name)
        item.add_argument('--whitelist', type=Path, required=True)
        item.add_argument('--output', type=Path, required=True)
        if name == 'worker':
            item.add_argument('--task', type=Path, required=True)
        else:
            item.add_argument('--fit-timeout', type=float, default=15.)
    args = parser.parse_args()
    if args.command == 'worker':
        worker(args.whitelist, json.loads(args.task.read_bytes()), args.output)
    else:
        result = run(args.whitelist, args.output, fit_timeout=args.fit_timeout)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
