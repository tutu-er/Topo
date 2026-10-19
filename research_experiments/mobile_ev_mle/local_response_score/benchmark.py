"""Fresh paired timing replay; never changes the frozen scientific results."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

from ..local_rx_identifiability.cases import candidate_bank
from ..local_rx_identifiability.experiment import read, write
from ..local_rx_three_day.experiment import key
from ..local_rx_three_day.solver import fit as ac_fit
from .experiment import sha
from .scoring import fit_profile, fit_qp


def replay(spec,directory):
    folder=Path(directory);name=key(spec)
    payload=read(folder/'inputs'/f'{name}.json.gz'); reference=read(folder/'records'/f'{name}.json.gz')
    rows=[]
    for index,c in enumerate(candidate_bank()):
        common=(c,payload['train'],payload['sigma'],payload['domain'])
        for method in ('AC','RESP','QP_N','QP_U'):
            try:
                if method=='AC':
                    fit=ac_fit(*common,multistart=4,max_iter=200,seed=spec['seed']+index)
                elif method=='RESP':
                    fit=fit_profile(*common,multistart=4,max_iter=500,seed=spec['seed']+index)
                else:
                    fit=fit_qp(*common,weight='nominal' if method=='QP_N' else 'upper')
                difference=fit['loss']-reference['methods'][method][c.name]['training']['loss']
                rows.append(dict(key=name,spec=spec,candidate=c.name,method=method,
                    seconds=fit['seconds'],loss=fit['loss'],qualified=fit['qualified'],
                    loss_difference_from_frozen=difference,
                    loss_reproduced=abs(difference)<=1e-6,
                    ac_calls=fit.get('ac_calls',0),response_calls=fit.get('response_calls',0)))
            except Exception as error:
                rows.append(dict(key=name,spec=spec,candidate=c.name,method=method,
                                 qualified=False,error=repr(error)))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True)
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    folder=Path(args.directory);target=folder/'benchmark.json'
    if target.exists():raise FileExistsError('benchmark replay cannot overwrite')
    config=read(folder/'configuration.json')
    manifest=dict(config['source_manifest']);manifest[str(Path(__file__).resolve())]=sha(__file__)
    for name,digest in manifest.items():
        path=Path(name) if Path(name).is_absolute() else Path(__file__).resolve().parents[3]/name
        if sha(path)!=digest:raise ValueError('frozen source mismatch: '+name)
    write(folder/'benchmark_configuration.json',dict(source_manifest=manifest,
        tasks=config['tasks'],workers=args.workers,
        scientific_policy='identical frozen settings; timing replay only, no tuning',
        AC=dict(starts=4,max_iter=200),RESP=dict(starts=4,max_iter=500)))
    start=perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(replay,s,str(folder)) for s in config['tasks']]
        for job in as_completed(jobs):
            part=job.result();rows.extend(part)
            print(dict(completed_conditions=len(rows)//16,expected=len(jobs),
                       qualified=sum(r['qualified'] for r in rows)),flush=True)
            write(folder/'benchmark_progress.json',dict(rows=rows,seconds=perf_counter()-start))
    result=dict(rows=rows,seconds=perf_counter()-start,workers=args.workers,
                independent_families=config['independent_families'],
                qualified=sum(r['qualified'] for r in rows),denominator=len(rows),
                all_losses_reproduced=all(r.get('loss_reproduced',False) for r in rows))
    write(target,result)
    print(dict(seconds=result['seconds'],qualified=result['qualified'],denominator=len(rows),
               all_losses_reproduced=result['all_losses_reproduced']),flush=True)


if __name__=='__main__':main()
