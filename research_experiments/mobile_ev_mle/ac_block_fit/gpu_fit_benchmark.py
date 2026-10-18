"""Same-start CPU/GPU hybrid full-fit arithmetic experiment, at most 12 fits.

No topology selection, future observation, or original frozen result is used.
Each fit has its own child process and retained failure/timeout record. CPU/GPU
order alternates for three repeats. GPU imports/configuration, full fitting,
and independent CPU physical/gradient audit are timed separately.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from . import benchmark as bm


STAGE = Path(__file__).resolve().parents[3]


def reconstruct(fixture, whitelist):
    """Return one declared observed-only start, with no future/truth fields."""
    if fixture == 'cigre':
        package = bm.load_history_package(whitelist)
        tasks = bm.make_tasks(package, networks=('cigre_residential',), prefixes=(192,),
            algorithms=('block_jac',), starts=('prefix_linear',), repeats=1)
        if len(tasks) != 1:
            raise ValueError('expected exactly one declared CIGRE historical candidate')
        task = tasks[0]
        candidate, data, meter, options, initial, initialization = bm._reconstruct(
            package['items']['cigre_residential'], task)
        declaration = dict(scope='independent observed-history whitelist only',
            scientific_conditions_created=0, future_used=False, truth_used=False,
            task=task, whitelist_sha256=bm._digest(whitelist), initialization=initialization)
    elif fixture == 'controlled32':
        import numpy as np
        from .. import likelihood as lk
        from ..reference_benchmark.fitting import history_linear_initial
        from .gpu_benchmark import fixture as numerical_fixture
        candidate, generated = numerical_fixture(32, 192, zero_edges=False)
        physical = lk.ac_predict(candidate, **generated)
        rng = np.random.default_rng(20421890)
        meter = lk.MeasurementModel(p_sigma=1e-4, q_sigma=1e-4, v_sigma=1e-4,
            master_p_sigma=1e-4, master_q_sigma=1e-4)
        data = lk.Observations(p=generated['p']+rng.normal(0, 1e-4, generated['p'].shape),
            q=generated['q']+rng.normal(0, 1e-4, generated['q'].shape),
            v=physical['v']+rng.normal(0, 1e-4, physical['v'].shape),
            root_v=generated['root_v'], master_p=physical['master_p']+rng.normal(0, 1e-4, 192),
            master_q=physical['master_q']+rng.normal(0, 1e-4, 192))
        options = dict(r_bounds=(5e-5, .02), x_bounds=(5e-6, .01), ratio_bounds=(.1, .5),
            source_p_bounds=(-.2, .2), source_q_bounds=(-.2, .2), voltage_bounds=(.8, 1.2),
            multistart=1, max_nfev=60, tol=1e-8)
        initial, initialization = history_linear_initial(candidate, data, meter,
            **{key: options[key] for key in ('r_bounds', 'x_bounds', 'ratio_bounds',
                                            'source_p_bounds', 'source_q_bounds')})
        declaration = dict(scope='controlled arithmetic fixture, no scientific policy evaluation',
            scientific_conditions_created=0, future_used=False, truth_used_in_fitter=False,
            generated_parameters_used_only_to_create_observations=True,
            selected_initialization_uses_only_observed_PQV_and_master=True,
            exact_zero_edges=0, seed=20421890, initialization=initialization)
    else:
        raise ValueError('unknown fixture')
    options.update(max_nfev=60, tol=1e-8, multistart=1)
    return candidate, data, meter, options, initial, declaration


def worker(args):
    """One fit; stdout/stderr and result are retained by the parent."""
    started = time.perf_counter()
    record = dict(fixture=args.fixture, backend=args.backend, repeat=args.repeat,
        started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        full_fit_speedup_scope='GPU AC+implicit derivatives, CPU QR/BVLS, all per-fit transfers included')
    try:
        phase = time.perf_counter()
        candidate, data, meter, options, initial, declaration = reconstruct(args.fixture, args.whitelist)
        record.update(setup_seconds=time.perf_counter()-phase, declaration=declaration,
            n=candidate.n, m=candidate.m, t=data.t, fit_options=options,
            initial_state_sha256=sha256(bm._bytes(initial)).hexdigest(),
            input_content_sha256=sha256(bm._bytes(dict(p=data.p,q=data.q,v=data.v,root=data.root_v,
                master_p=data.master_p,master_q=data.master_q))).hexdigest())
        import numpy as np
        import scipy
        record['environment'] = dict(python=sys.version, numpy=np.__version__, numpy_path=np.__file__,
            scipy=scipy.__version__, original_workers_still_running=True,
            threads={key: os.environ.get(key) for key in
                     ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        if args.backend == 'gpu':
            from .gpu import configure_dependency_path
            phase = time.perf_counter(); configure_dependency_path(args.dependencies)
            record['gpu_dependency_configuration_seconds'] = time.perf_counter()-phase
            phase = time.perf_counter(); import cupy as cp
            record['gpu_import_seconds'] = time.perf_counter()-phase
            record['environment'].update(cupy=cp.__version__, gpu_driver=cp.cuda.runtime.driverGetVersion())
        from .nonlinear import fit_rx_blocks
        common = dict(inner_tol=1e-10, inner_maxiter=2000, scale_mode='jac',
                      inner_acceptance='certified_gap', inner_gap_rtol=1e-8)
        record['algorithm_settings'] = common
        counter = {}
        bm._write(args.progress, dict(phase='fitting', started_unix=time.time(),
            fixture=args.fixture, backend=args.backend, repeat=args.repeat))
        phase = time.perf_counter()
        fitted = fit_rx_blocks(candidate, data, meter, initial=initial, **options, **common,
            backend=args.backend, gpu_dependency_path=str(args.dependencies) if args.backend == 'gpu' else None,
            seed=20421800, call_counter=counter)
        record['fit_seconds'] = time.perf_counter()-phase
        bm._write(args.progress, dict(phase='independent_CPU_audit', started_unix=time.time(),
            fixture=args.fixture, backend=args.backend, repeat=args.repeat))
        phase = time.perf_counter()
        audit = bm.audit_fit(candidate, data, meter, options, fitted, fixed_lines=False)
        record['independent_CPU_audit_seconds'] = time.perf_counter()-phase
        record.update(status='returned', solver_success=bool(fitted.success), fit=fitted.summary(),
            state=dict(r=fitted.r, x=fitted.x, source_p=fitted.source_p, source_q=fitted.source_q),
            prediction={key: fitted.predicted[key] for key in ('physical_v','master_p','master_q')},
            audit=audit, counters=counter)
    except Exception as error:
        record.update(status='exception', error=type(error).__name__+': '+str(error),
                      traceback=traceback.format_exc())
    record.update(total_worker_seconds=time.perf_counter()-started,
        completed_utc=datetime.now(timezone.utc).isoformat(), source_hashes=bm._source_fingerprints())
    bm._write(args.output, record)
    bm._write(args.progress, dict(phase='finished', started_unix=time.time(), status=record['status']))
    return 0 if record['status'] == 'returned' else 1


def compare_pairs(records):
    import numpy as np
    pairs = []
    for fixture in ('cigre', 'controlled32'):
        for repeat in range(3):
            rows = {r['backend']:r for r in records if r['fixture']==fixture and r['repeat']==repeat}
            pair = dict(fixture=fixture, repeat=repeat, available=False)
            if set(rows)=={'cpu','gpu'} and all(r['status']=='returned' for r in rows.values()):
                cpu, gpu = rows['cpu'], rows['gpu']
                pair.update(available=True, same_initial=cpu['initial_state_sha256']==gpu['initial_state_sha256'],
                    same_inputs=cpu['input_content_sha256']==gpu['input_content_sha256'],
                    cpu_loss=cpu['fit']['loss'], gpu_loss=gpu['fit']['loss'],
                    loss_difference=float(gpu['fit']['loss']-cpu['fit']['loss']),
                    relative_loss_difference=float(abs(gpu['fit']['loss']-cpu['fit']['loss'])/max(1.,abs(cpu['fit']['loss']))),
                    cpu_fit_seconds=cpu['fit_seconds'],gpu_fit_seconds=gpu['fit_seconds'],
                    full_fit_speedup=cpu['fit_seconds']/gpu['fit_seconds'],
                    cpu_solver_success=cpu['solver_success'],gpu_solver_success=gpu['solver_success'])
                pair['state_max_abs_difference'] = {key:float(np.max(abs(np.asarray(cpu['state'][key])-
                    np.asarray(gpu['state'][key])))) for key in ('r','x','source_p','source_q')}
                pair['prediction_max_abs_difference'] = {key:float(np.max(abs(np.asarray(cpu['prediction'][key])-
                    np.asarray(gpu['prediction'][key])))) for key in ('physical_v','master_p','master_q')}
                pair['cpu_reference_pg'] = cpu['audit'].get('reference_scaled_projected_gradient')
                pair['gpu_reference_pg'] = gpu['audit'].get('reference_scaled_projected_gradient')
            pairs.append(pair)
    summary = {}
    for fixture in ('cigre','controlled32'):
        rows = [p for p in pairs if p['fixture']==fixture]
        available = [p for p in rows if p['available']]
        summary[fixture] = dict(pairs_total=3,pairs_returned=len(available),
            median_cpu_fit_seconds=statistics.median(p['cpu_fit_seconds'] for p in available) if available else None,
            median_gpu_fit_seconds=statistics.median(p['gpu_fit_seconds'] for p in available) if available else None,
            median_paired_full_fit_speedup=statistics.median(p['full_fit_speedup'] for p in available) if available else None,
            all_inputs_and_initials_equal=all(p['same_initial'] and p['same_inputs'] for p in available),
            cpu_success=sum(p['cpu_solver_success'] for p in available),
            gpu_success=sum(p['gpu_solver_success'] for p in available))
    return dict(pairs=pairs, summary=summary)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    source_hashes = bm._source_fingerprints()
    records = []; order = []
    environment = dict(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(STAGE)+';'+str(STAGE/'rnj_wzzt_core'))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        environment[key]='1'
    # CIGRE is completed first. No new fixture is launched if its interface fails.
    for fixture in ('cigre', 'controlled32'):
        fixture_failed = False
        for repeat in range(3):
            backends = ('cpu','gpu') if repeat%2==0 else ('gpu','cpu')
            for backend in backends:
                tag=f'{fixture}_{repeat}_{backend}'; order.append(tag)
                result_path=args.output/(tag+'.json'); progress=args.output/(tag+'.progress.json')
                if result_path.exists():
                    raise FileExistsError('refuse to overwrite completed fit: '+str(result_path))
                command=[sys.executable,'-m','research_experiments.mobile_ev_mle.ac_block_fit.gpu_fit_benchmark',
                    'worker','--fixture',fixture,'--backend',backend,'--repeat',str(repeat),
                    '--whitelist',str(args.whitelist),'--dependencies',str(args.dependencies),
                    '--output',str(result_path),'--progress',str(progress)]
                start=time.time(); timed_out=False
                with open(args.output/(tag+'.stdout.log'),'w',encoding='utf8') as out, \
                     open(args.output/(tag+'.stderr.log'),'w',encoding='utf8') as err:
                    child=subprocess.Popen(command,cwd=str(STAGE),env=environment,stdout=out,stderr=err)
                    while child.poll() is None:
                        status=bm._read(progress) if progress.exists() else {}
                        if status.get('phase')=='fitting' and time.time()-status['started_unix']>args.fit_timeout:
                            timed_out=True; child.kill(); child.wait(timeout=10); break
                        if time.time()-start>args.fit_timeout+180:
                            timed_out=True; child.kill(); child.wait(timeout=10); break
                        time.sleep(.2)
                if timed_out or not result_path.exists():
                    record=dict(fixture=fixture,backend=backend,repeat=repeat,
                        status='timeout' if timed_out else 'worker_missing_result',
                        fit_timeout_seconds=args.fit_timeout,total_worker_seconds=time.time()-start,
                        denominator_retained=True,child_pid=child.pid,exit_code=child.returncode)
                    bm._write(result_path,record)
                else:
                    record=bm._read(result_path)
                records.append(record)
                fixture_failed |= record['status']!='returned'
                summary=dict(status='running',records=records,order=order,source_hashes=source_hashes,
                    fit_budget=60,fit_timeout_seconds=args.fit_timeout,expected_fit_count=12,
                    failures_retained=True,scientific_conditions_created=0,**compare_pairs(records))
                bm._write(args.output/'summary.json',summary)
                print(tag,record['status'],record.get('fit_seconds'),flush=True)
        if fixture=='cigre' and fixture_failed:
            print('CIGRE interface/computation failure retained; controlled32 was not started',flush=True)
            break
    summary.update(status='complete',completed_fit_count=len(records),
                   scientific_policy_comparison_performed=False)
    bm._write(args.output/'summary.json',summary)
    return 0


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('run');p.add_argument('--whitelist',type=Path,default=STAGE/'outputs/history_only_v1.json.gz')
    p.add_argument('--dependencies',type=Path,default=STAGE/'outputs/gpu_deps')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--fit-timeout',type=float,default=120.)
    p=sub.add_parser('worker');p.add_argument('--whitelist',type=Path,required=True)
    p.add_argument('--dependencies',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--progress',type=Path,required=True);p.add_argument('--fixture',choices=('cigre','controlled32'),required=True)
    p.add_argument('--backend',choices=('cpu','gpu'),required=True);p.add_argument('--repeat',type=int,required=True)
    args=parser.parse_args()
    return run(args) if args.mode=='run' else worker(args)


if __name__=='__main__':
    raise SystemExit(main())
