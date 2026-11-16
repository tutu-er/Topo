"""Paired two/eight-day identification; truth remains an offline diagnostic.

All old runs are immutable. No procurement, fitting-penalty change, or
true-support candidate injection is performed here.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback

import numpy as np

from ... import io
from ...history import fit_observed, freeze_history
from ...identification.optimization import QPFitter, errors, response_matrices
from ...observations import fingerprint
from ..distribution_panel_20261115.data import additional_search, append_candidate
from ..flexibility_sweep_20261113.fitting import native_json_record
from ..measured_pq_20261114 import validate
from . import data
from .metrics import metrics, fixed_support_diagnostics


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT/'outputs/history10_rx_20261116_v1'
OLD_INPUTS = ROOT/'outputs/rolling15_development_20261115_v1/inputs.json'
ARCHIVE = ROOT/'outputs/measured_pq_sources_20261114/paired_pq_15min.npz'


def utc():
    return datetime.now(timezone.utc).isoformat()


def checked(ref):
    if io.ref(ref['path'])['sha256'] != ref['sha256']:
        raise ValueError('Sealed file changed: '+ref['path'])
    return Path(ref['path'])


def arrays(ref):
    with np.load(checked(ref), allow_pickle=False) as z:
        return {key:z[key].copy() for key in z.files}


def source_paths():
    paths = {Path(__file__).with_name('PROTOCOL.md')}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.suffix == '.py' and p.is_relative_to(ROOT) and 'outputs' not in p.parts:
                if not p.name.startswith('test_'):
                    paths.add(p)
    # Independent post-processing has its own hash receipt and does not execute
    # in data generation or fitting; it may be completed while the batch runs.
    paths.update(p for p in Path(__file__).parent.glob('*.py')
                 if not p.name.startswith('test_') and p.name != 'audit_report.py')
    for case in ('soumalas11', 'pengwah18'):
        paths.update(ROOT/'topology_identification_standalone/cases'/case/name
                     for name in ('metadata.json','buses.csv','branches.csv'))
    return sorted(paths)


def prepare(out, *, workers=3):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Preserve existing preparation: '+str(out))
    if not 1 <= workers <= 3:
        raise ValueError('Use one to three independent single-thread workers')
    paths = source_paths()
    out.mkdir(parents=True)
    refs = []
    for p in paths:
        dest = out/'source_snapshot'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest)
        refs.append(io.ref(p))
    tasks = [dict(id=f'{case}__d{days:02d}__h{h:02d}', case=case,
                  train_days=days, history_repeat=h)
             for case in ('soumalas11','pengwah18') for days in (2,8) for h in range(4)]
    protocol = dict(created_utc=utc(), source_document=io.ref(Path(__file__).with_name('PROTOCOL.md')),
        sources=refs, old_inputs=io.ref(OLD_INPUTS), archive=io.ref(ARCHIVE),
        tasks=tasks, planned_fits=len(tasks), workers=workers, history_repeats=4,
        identifier_method='101', method_seconds=120., additional_E_seconds=60.,
        identifier_config=asdict(validate._config(120.)),
        start_date='2024-01-21', window_days=11, fit_days=[2,8],
        train_indices={'2':[576,768], '4':[384,768], '8':[0,768]},
        validation_indices=[768,960], new_day_indices=[960,1056],
        observed_error=dict(P=.005,Q=.005,V=.0005,root=.0005),
        noise_seed=2026111601, root_physical_variation=.003,
        scale_rule='Unchanged old January4 node scales; no long-history rescaling',
        regularizer='No coefficient ridge or L1; unchanged structural alpha=.003',
        truth_policy='Never passed to identifier or model selection; offline metrics only after candidate seal',
        diagnostic_training_days=[2,4,8], diagnostic_laws=['cleanlinear','cleanAC','Vonly','all'],
        diagnostic_reference_mse=1e-6,
        interpretation='Two known test networks, one common physical time window; repetitions are paired meter-noise draws',
        source_unchanged_at_prepare=True, environment=validate._environment(), **validate._git_receipt(paths))
    io.write(out/'protocol.json', protocol)
    worlds = data.prepare_worlds(out/'data', OLD_INPUTS, ARCHIVE, io.ref(out/'protocol.json'))
    if isinstance(worlds, dict):
        worlds = worlds['worlds']
    io.write(out/'worlds.json', worlds)
    if len(worlds) != 2 or {w['case'] for w in worlds} != {'soumalas11','pengwah18'}:
        raise RuntimeError('Preparation did not produce both locked physical worlds; failures retained')
    io.write(out/'prepared.json',dict(status='prepared',protocol=io.ref(out/'protocol.json'),
        worlds=io.ref(out/'worlds.json'),created_utc=utc()))
    return protocol


def verify_sources(protocol):
    for ref in [*protocol['sources'], protocol['old_inputs'], protocol['archive']]:
        checked(ref)


def resource_nodes(world):
    old = world['old_entry']
    devices = io.read(checked(old['equipment_spec']))['devices']
    return {'P': sorted({int(d['node_label']) for d in devices}),
            'Q': sorted({int(d['node_label']) for d in devices if d['kind'] in ('pv','battery')})}


def load_world(out, task):
    return next(w for w in io.read(Path(out)/'worlds.json') if w['case'] == task['case'])


def save_observation(path, obs):
    fields = dict(P=obs.p,Q=obs.q,Y=obs.y,labels=obs.labels)
    if obs.voltage is not None:
        fields.update(V=obs.voltage,root=obs.root)
    np.savez_compressed(path,**fields)
    return io.ref(path)


def training_bounds(world, days):
    ix = np.asarray(world['train_indices'][str(days)],int)
    if not len(ix) or not np.array_equal(ix,np.arange(ix[0],ix[-1]+1)):
        raise ValueError('Expected one chronological contiguous training slice')
    return int(ix[0]),int(ix[-1]+1)


def fit_task(out, task):
    out = Path(out)
    directory = out/'fits'/task['id']
    if (directory/'complete.json').exists():
        done = io.read(directory/'complete.json')
        if done['task'] != task:
            raise ValueError('Saved attempt has different locked coordinates')
        checked(done['result'])
        return done
    if directory.exists():
        raise RuntimeError('Partial fit preserved; no automatic budget reset: '+str(directory))
    directory.mkdir(parents=True)
    io.write(directory/'started.json',dict(task=task,pid=os.getpid(),created_utc=utc()))
    began = perf_counter(); stage = 'observations'
    try:
        protocol=io.read(out/'protocol.json');verify_sources(protocol)
        world=load_world(out,task)
        clean=arrays(world['files']['clean.npz'])
        noise=arrays(world['files'][f"noise_h{task['history_repeat']:02d}.npz"])
        observed=data.observations(clean,noise,law='all')
        splits=data.slice_observations(observed,world,task['train_days'])
        training,validation=splits['train'],splits['validation']
        train_ref=save_observation(directory/'train.npz',training)
        val_ref=save_observation(directory/'validation.npz',validation)
        io.write(directory/'observations_sealed.json',dict(task=task,train=train_ref,validation=val_ref,
            training_sha256=fingerprint(training),validation_sha256=fingerprint(validation),
            observed_error=protocol['observed_error'],new_day_not_given_to_identifier=True))
        stage='native101'
        original=fit_observed(training,validation,method='101',config=validate._config(protocol['method_seconds']))
        io.write(directory/'original_native.json',native_json_record(original))
        stage='additional_original_E'
        extra=additional_search(original,training,seconds=protocol['additional_E_seconds'])
        io.write(directory/'additional_search.json',extra)
        native=append_candidate(original,extra.get('closed_point'),training,validation)
        io.write(directory/'native.json',native_json_record(native))
        frozen=freeze_history(native,validation)
        seal=dict(task=task,created_utc=utc(),native=io.ref(directory/'native.json'),
            original=io.ref(directory/'original_native.json'),additional=io.ref(directory/'additional_search.json'),
            parameter_sha256=frozen.parameter_sha256,train=train_ref,validation=val_ref,
            truth_used=False,new_day_used=False)
        io.write(directory/'candidates_sealed.json',seal)
        # Physical truth and the evaluation scores are first consumed here.
        stage='offline_evaluation'
        physical=arrays(world['files']['Hstar.npz'])
        Hstar=physical['Hstar'];truth=json.loads(str(physical['support_json']))
        new_ref=save_observation(directory/'new_day.npz',splits['new_day'])
        fitted=dict(status='fitted',seconds=perf_counter()-began,result=native,frozen=frozen)
        evaluation=validate._evaluate_identifier(fitted,splits,truth)
        original_fitted=dict(status='fitted',seconds=float(original.get('total_seconds',0.)),
                             result=original,frozen=freeze_history(original,validation))
        original_evaluation=validate._evaluate_identifier(original_fitted,splits,truth)
        candidates=[]
        for i,row in enumerate(native['frozen_candidates']):
            candidates.append(dict(native_index=i,
                **validate._topology(row['point'],truth,training.n,native['config']['zero_tol']),
                validation_mse=errors(validation,row['point'])['mse']))
        good=[r for r in candidates if r['internal_exact']]
        roles=[('history_selected',evaluation['history_selected']),
               ('cumulative_selected',evaluation['cumulative_selected'])]
        if good:
            roles.append(('best_true_internal',min(good,key=lambda r:(r['validation_mse'],r['native_index']))['native_index']))
        matrix_rows=[]
        start,stop=training_bounds(world,task['train_days'])
        for role,index in roles:
            point=native['frozen_candidates'][index]['point']
            R,X=response_matrices(point,training.n);H=np.c_[R,X]
            matrix_rows.append(dict(role=role,native_index=index,
                metrics=metrics(H,Hstar,clean['P'][start:stop],clean['Q'][start:stop],
                                resource_nodes(world),labels=world['labels']),
                design=fixed_support_diagnostics(training.p,training.q,point['supports']),
                losses={name:errors(obs,point) for name,obs in splits.items()}))
        record=dict(status='complete',task=task,seconds=perf_counter()-began,
            train=train_ref,validation=val_ref,new_day=new_ref,candidates_seal=io.ref(directory/'candidates_sealed.json'),
            evaluation=evaluation,original_evaluation=original_evaluation,
            additional_search_status=extra['status'],matrix_rows=matrix_rows,candidates=candidates,
            source_unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in protocol['sources']))
    except Exception as exc:
        record=dict(status='failed',task=task,stage=stage,seconds=perf_counter()-began,
                    error=repr(exc),traceback=traceback.format_exc())
    io.write(directory/'result.json',record)
    done=dict(status=record['status'],task=task,result=io.ref(directory/'result.json'),created_utc=utc())
    io.write(directory/'complete.json',done)
    return done


def run(out):
    out=Path(out);protocol=io.read(out/'protocol.json');verify_sources(protocol)
    if not (out/'prepared.json').exists():
        raise RuntimeError('Complete source/AC preparation first')
    done=[];began=perf_counter()
    with ProcessPoolExecutor(max_workers=protocol['workers']) as pool:
        futures={pool.submit(fit_task,str(out),task):task for task in protocol['tasks']}
        for future in as_completed(futures):
            result=future.result();done.append(result)
            print(json.dumps(dict(id=result['task']['id'],status=result['status'],
                                  attempted=len(done),planned=len(protocol['tasks']))),flush=True)
    receipt=dict(status='complete' if all(r['status']=='complete' for r in done) else 'complete_with_failures',
        attempted=len(done),successful=sum(r['status']=='complete' for r in done),planned=len(protocol['tasks']),
        cells=done,seconds=perf_counter()-began,
        source_unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in protocol['sources']))
    io.write(out/'fits_complete.json',receipt)
    return receipt


def offline(out):
    out=Path(out);protocol=io.read(out/'protocol.json');verify_sources(protocol)
    target=out/'offline'
    if (target/'complete.json').exists():
        checked(io.read(target/'complete.json')['diagnostics']);return io.read(target/'complete.json')
    if target.exists():
        raise RuntimeError('Partial offline diagnostics preserved; inspect before restarting')
    target.mkdir();io.write(target/'started.json',dict(pid=os.getpid(),created_utc=utc()))
    rows=[];began=perf_counter()
    for world in io.read(out/'worlds.json'):
        clean=arrays(world['files']['clean.npz']);physical=arrays(world['files']['Hstar.npz'])
        truth=json.loads(str(physical['support_json']));Hstar=physical['Hstar'];n=len(world['labels'])
        old=io.read(checked(world['old_entry']['native_copy']))
        old_val=world['old_entry']['files']['validation.npz']
        from ...observations import load
        old_validation=load(checked(old_val))
        good=[i for i,r in enumerate(old['frozen_candidates'])
              if validate._topology(r['point'],truth,n,old['config']['zero_tol'])['internal_exact']]
        supports=[('oracle_physical_support',truth),
                  ('old_historical_winner',old['frozen_candidates'][old['selected']['index']]['point']['supports'])]
        if good:
            index=min(good,key=lambda i:(errors(old_validation,old['frozen_candidates'][i]['point'])['mse'],i))
            supports.append(('old_correct_internal',old['frozen_candidates'][index]['point']['supports']))
        aliases={}
        for role,family in supports:
            key=tuple(sorted(tuple(sorted(C)) for C in family))
            aliases.setdefault(key,[]).append(role)
        for h in range(4):
            noise=arrays(world['files'][f'noise_h{h:02d}.npz'])
            for law in protocol['diagnostic_laws']:
                observed=data.observations(clean,noise,law=law,Hstar=Hstar)
                for days in protocol['diagnostic_training_days']:
                    splits=data.slice_observations(observed,world,days);train=splits['train']
                    fitter=QPFitter(train,reference_mse=protocol['diagnostic_reference_mse'],bound=2.,qp_seconds=300.)
                    start,stop=training_bounds(world,days)
                    for family,roles in aliases.items():
                        item=dict(case=world['case'],history_repeat=h,train_days=days,law=law,
                            support_role=roles[0],support_aliases=roles,
                            truth_role='Offline parameter diagnostic; never inserted in native bank',
                            deterministic_across_repeats=law in ('cleanlinear','cleanAC'))
                        try:
                            point=fitter.fit(family);R,X=response_matrices(point,n);H=np.c_[R,X]
                            item.update(status='complete',point=point,qp_record=fitter.records[-1],
                                topology=validate._topology(point,truth,n,1e-9),
                                metrics=metrics(H,Hstar,clean['P'][start:stop],clean['Q'][start:stop],
                                                resource_nodes(world),labels=world['labels']),
                                design=fixed_support_diagnostics(train.p,train.q,family),
                                losses={name:errors(obs,point) for name,obs in splits.items()})
                        except Exception as exc:
                            item.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
                        rows.append(item)
        print(json.dumps(dict(offline_case=world['case'],rows=len(rows))),flush=True)
    io.write(target/'diagnostics.json',dict(rows=rows,protocol=io.ref(out/'protocol.json'),
        source_unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in protocol['sources'])))
    receipt=dict(status='complete' if all(r['status']=='complete' for r in rows) else 'complete_with_failures',
        rows=len(rows),successful=sum(r['status']=='complete' for r in rows),seconds=perf_counter()-began,
        diagnostics=io.ref(target/'diagnostics.json'))
    io.write(target/'complete.json',receipt);return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--offline-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.prepare_only:
        receipt=prepare(args.out,workers=args.workers)
        print(json.dumps(dict(status='prepared',planned_fits=receipt['planned_fits'])),flush=True)
    elif args.offline_only:
        print(json.dumps(offline(args.out)),flush=True)
    elif args.resume:
        print(json.dumps(run(args.out)),flush=True)
    else:
        parser.error('Choose --prepare-only, --resume, or --offline-only')


if __name__=='__main__':
    main()
