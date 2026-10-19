"""Frozen observed-only train/holdout comparison on existing ordinary data."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path
from time import perf_counter
import traceback
import sys
import platform
import numpy as np
import scipy

from ..local_rx_identifiability.cases import make_case, candidate_bank
from ..local_rx_identifiability.experiment import read, write
from ..local_rx_three_day.experiment import key
from . import scoring as sc
from .ac_holdout import fit_fixed

ROOT = Path(__file__).resolve().parents[3]
METHODS = ('AC', 'RESP', 'QP_N', 'QP_U')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    previous = read(ROOT/'outputs/local_rx_three_day_main_v1/configuration.json')['source_manifest']
    names = set(previous)
    names.update('research_experiments/mobile_ev_mle/local_response_score/'+name
                 for name in ('__init__.py','experiment.py','ac_holdout.py',
                              'scoring.py','remainder.py','rejection.py'))
    for name, digest in previous.items():
        if sha(ROOT/name) != digest:
            raise ValueError('old frozen scientific source changed: '+name)
    return {name:sha(ROOT/name) for name in sorted(names)}


def specifications(pilot=False):
    if pilot:
        return [dict(network='paper15', seed=32991, N=288, boundary='U')]
    return [dict(network=net, seed=seed, N=n, boundary='U')
            for net in ('paper15','paper11') for seed in range(33001,33005)
            for n in (288,672)]


def observed_inputs(spec, baseline):
    old_path = Path(baseline)/'inference'/f'{key(spec)}.json.gz'
    old = read(old_path)
    generated = make_case(spec['network'], 'ordinary', spec['seed'], total=768)
    train = {k:np.asarray(old['observed'][k], float) for k in ('p','q','v')}
    hold = {k:np.asarray(generated['observed'][k], float)[672:768] for k in train}
    for channel, value in train.items():
        if not np.array_equal(value, np.asarray(generated['observed'][channel])[:spec['N']]):
            raise ValueError('existing training observation mismatch: '+channel)
    if old.get('root_known') is not None or old.get('root_meter') is not None:
        raise ValueError('only U terminal meter comparison is authorized here')
    payload = dict(spec=spec, train=train, holdout=hold,
                   sigma=old['sigma'], domain=old['domain'],
                   old_inference_sha256=sha(old_path),
                   holdout_indices=[672,768], train_indices=[0,spec['N']],
                   disjoint_training_holdout=True, truth_used=False,
                   prediction_uses_observed_holdout_PQ=True,
                   interpretation='fixed-training-model score of independent noisy holdout meters, not a forecast of their true values')
    return payload, old


def original_domain(r, x, domain):
    z, ratio = np.hypot(r,x), np.asarray(x)/r
    return bool(np.min(z)>=domain['z_min']-1e-10 and np.max(z)<=domain['z_max']+1e-10
                and np.min(ratio)>=domain['ratio_min']-1e-10
                and np.max(ratio)<=domain['ratio_max']+1e-10)


def _score_and_backsub(candidate, payload, fitted, method):
    obs, sigma, domain = payload['holdout'], payload['sigma'], payload['domain']
    if 'r' not in fitted:
        return dict(status='training_failed', qualified=False)
    r, x = np.asarray(fitted['r']), np.asarray(fitted['x'])
    answer = dict(profile=sc.score_fixed(candidate, obs, sigma, r, x),
                  original_line_domain_feasible=original_domain(r,x,domain))
    if method.startswith('QP'):
        answer['own'] = sc.score_fixed(candidate, obs, sigma, r, x,
                                      weight='nominal' if method=='QP_N' else 'upper', domain=domain)
    else:
        answer['own'] = answer['profile']
    if answer['original_line_domain_feasible']:
        try:
            answer['ac'] = fit_fixed(candidate, obs, sigma, domain, r, x)
        except Exception as error:
            answer['ac'] = dict(qualified=False,status='failed',error=repr(error),
                                traceback=traceback.format_exc())
    else:
        answer['ac'] = dict(qualified=False,status='outer_parameter_not_in_original_domain',
                            physical_backsub_skipped=True)
    if method=='AC':
        answer['own'] = answer['ac']
    return answer


def run_one(spec, directory, baseline):
    started = perf_counter(); directory = Path(directory); name=key(spec)
    output = directory/'records'/f'{name}.json.gz'
    if output.exists():
        raise FileExistsError('refusing to overwrite '+str(output))
    record = dict(spec=spec,status='failed',methods={}, failures=[])
    try:
        payload, old = observed_inputs(spec, baseline)
        write(directory/'inputs'/f'{name}.json.gz', payload)
        record['input_sha256'] = sha(directory/'inputs'/f'{name}.json.gz')
        record['methods'] = {m:{} for m in METHODS}
        for index, candidate in enumerate(candidate_bank()):
            # No simulator object or truth dictionary is supplied to any fit.
            training = dict(AC=old['fits'][candidate.name])
            tasks = [('RESP', lambda:sc.fit_profile(candidate,payload['train'],payload['sigma'],
                        payload['domain'],multistart=4,max_iter=500,seed=spec['seed']+index)),
                     ('QP_N',lambda:sc.fit_qp(candidate,payload['train'],payload['sigma'],payload['domain'],weight='nominal')),
                     ('QP_U',lambda:sc.fit_qp(candidate,payload['train'],payload['sigma'],payload['domain'],weight='upper'))]
            for method, operation in tasks:
                try:
                    training[method] = operation()
                except Exception as error:
                    training[method] = dict(qualified=False,status='failed',error=repr(error),
                                             traceback=traceback.format_exc())
                    record['failures'].append(dict(candidate=candidate.name,method=method,phase='training',error=repr(error)))
            for method, fitted in training.items():
                record['methods'][method][candidate.name] = dict(
                    training=fitted, holdout=_score_and_backsub(candidate,payload,fitted,method))
        record['status']='completed'
        # Seal every observed-only score before optional diagnostics can read truth.
        record['truth_used']=False
    except Exception as error:
        record['error']=repr(error); record['traceback']=traceback.format_exc()
    record['seconds']=perf_counter()-started
    write(output,record)
    return dict(spec=spec,status=record['status'],seconds=record['seconds'],
                failures=len(record['failures']),error=record.get('error'),
                qualified={method:sum(bool(v['training'].get('qualified')) for v in table.values())
                           for method,table in record['methods'].items()})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--baseline',default='outputs/local_rx_three_day_main_v1')
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();directory=Path(args.output)
    if directory.exists():raise FileExistsError('new output directory required')
    directory.mkdir(parents=True)
    config=dict(tasks=specifications(args.pilot), baseline=str(Path(args.baseline).resolve()),
                source_manifest=sources(), protocol_sha256=sha(Path(__file__).with_name('protocol.md')),
                workers=args.workers, pilot=args.pilot, methods=METHODS,
                v_ref=1., profile_starts=4, profile_max_iter=500,
                holdout_fixed_line_max_iter=100, independent_families=1 if args.pilot else 8,
                environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()))
    write(directory/'configuration.json',config)
    start=perf_counter();completed=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(run_one,s,str(directory),args.baseline) for s in config['tasks']]
        for job in as_completed(jobs):
            result=job.result();completed.append(result);print(result,flush=True)
            write(directory/'progress.json',dict(completed=completed,expected=len(jobs),seconds=perf_counter()-start))
    write(directory/'completed.json',dict(completed=completed,expected=len(jobs),seconds=perf_counter()-start))


if __name__=='__main__':main()
