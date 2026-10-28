"""A fixed-data, fixed-bound four-extension initialization audit.

The old leaf-plus-RNJ prefix is reused. Only a two-RNJ-block initial family
is newly solved, with every original terminal column retained. This prefix
is intentionally incomplete as a topology recovery experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import platform
import sys
from time import perf_counter

import numpy as np
import scipy

from research_experiments.wzzt_dr_selection.end_to_end import wzzt_adapter as observed
from .audit_rnj_family import recovery

core = observed.core


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def observation_only(scenarios):
    return [{key:s[key] for key in ('name', 'P_terminal', 'Q_terminal', 'drop_target')}
            for s in scenarios]


def new_supports(points):
    previous = set()
    rows = []
    for index, point in enumerate(points):
        current = set(map(tuple, point['support_labels']))
        rows.append({'path_index':index, 'new_support_labels':[list(s) for s in sorted(current-previous)],
                     'removed_support_labels':[list(s) for s in sorted(previous-current)]})
        previous = current
    return rows


def describe(points, truth, terminals):
    result = []
    for point, changes in zip(points, new_supports(points), strict=True):
        max_r = max(point['r_values'], default=0.)
        max_x = max(point['x_values'], default=0.)
        result.append({**changes, 'stored_support_count':len(point['support_labels']),
            'train_mae':point['train_mae'], 'validation_mae':point['validation_mae'],
            'validation_se':point['validation_se'], 'max_r':max_r, 'max_x':max_x,
            'touches_bound_2':max(max_r,max_x)>=1.99,
            'recovery':recovery(point['support_labels'],point['r_values'],point['x_values'],truth,terminals)})
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    root=args.reference_run.resolve();out=args.output.resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    protocol=out.with_name(out.stem+'_protocol.json')
    if out.exists() or protocol.exists():
        raise FileExistsError('Existing audit; inspect its checkpoint, never duplicate a prefix')
    lock=out.with_suffix('.lock')
    with lock.open('x',encoding='utf-8') as stream:
        json.dump({'pid':os.getpid(),'created_utc':now()},stream)
    inputs=[root/'prepared_case.pkl',root/'selection.json',root/'fit.json',root/'completion.json']
    hashes={str(p):sha(p) for p in inputs}
    selection=json.loads(inputs[1].read_text(encoding='utf-8'))
    old=json.loads(inputs[2].read_text(encoding='utf-8'))
    with inputs[0].open('rb') as stream:
        case,_=pickle.load(stream)
    training=observation_only(case.training);validation=observation_only(case.validation)
    terminals=list(training[0]['P_terminal'].columns)
    positions={t:i for i,t in enumerate(terminals)}
    selected=[tuple(positions[t] for t in s) for s in selection['selected']]
    if len(selected)!=2 or len(terminals)!=15:
        raise ValueError('This locked prefix is only for N0 15-terminal, two-RNJ initialization')
    if old['r_upper_bound']!=2 or old['x_upper_bound']!=2 or old['bound_expansions']!=0:
        raise ValueError('Stored leaf-plus-RNJ reference has a different coefficient domain')
    old_prefix=old['path'][:5]
    if len(old_prefix)!=5:
        raise ValueError('Stored reference does not contain four accepted extensions')
    selected_set={tuple(sorted(s)) for s in selected}
    expected=selected_set|{(i,) for i in range(len(terminals))}
    if {tuple(s) for s in old_prefix[0]['support_indices']}!=expected:
        raise ValueError('Stored reference initializer differs from all leaves plus selected RNJ')
    options={'initial_supports':selected,'max_atoms':len(selected)+4,
        'r_upper_bound':2.,'x_upper_bound':2.,'max_bound_expansions':0,
        'time_limit':1800.,'solver':'highs'}
    effective={k:v.default for k,v in inspect.signature(core.fit_laminar_l1_sensitivity).parameters.items()
               if v.default is not inspect.Parameter.empty and k not in ('validation_scenarios','progress_callback')}
    effective.update(options)
    meta={'scope':__doc__,'created_utc':now(),'status':'planned',
        'condition_count':2,'new_search_conditions':1,'reused_search_conditions':1,
        'new_rnj_or_bootstrap_calls':0,'new_observations':0,'new_price_conditions':0,
        'new_maximum_accepted_extensions':4,
        'prefix_is_not_full_recovery_test':True,
        'bounds_policy':'fixed2 for both initializations; no expansion; boundary hit remains unresolved',
        'comparison_scope':'same bounded raw mean-L1 objective; no conclusion about unbounded candidate order',
        'truth_policy':'No truth in solver input or candidate selection; recovery counts only after returned prefix',
        'training_rows':sum(len(s['P_terminal']) for s in training),
        'validation_rows':sum(len(s['P_terminal']) for s in validation),
        'terminal_labels':terminals,'selected_rnj_support_labels':selection['selected'],
        'new_effective_options':effective,'reused_prefix_path_indices':list(range(5)),
        'reused_extensions':old['attempted_extensions'][:4],
        'source_sha256':{str(Path(p).resolve()):sha(p) for p in (__file__,core.__file__,observed.__file__)},
        'input_sha256':hashes,
        'environment':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'numpy':np.__version__,'scipy':scipy.__version__}}
    observed._write_json(protocol,meta)
    result={**meta,'protocol_sha256':sha(protocol),'status':'running','events':[],
        'reused_leaf_plus_rnj_path':old_prefix,'new_path':[],'attempted_extensions':[]}
    observed._write_json(out,result)
    start=perf_counter()
    def progress(state):
        result['events'].append({'event':state['event'],'seconds':perf_counter()-start,
            'path_length':len(state['path']),'attempt_count':len(state['attempted_extensions'])})
        result['new_path']=[observed._path(p,state['path'][i-1] if i else None)
                            for i,p in enumerate(state['path'])]
        result['attempted_extensions']=[{**observed._plain(a),'diagnostics':observed._diagnostic(a.diagnostics)}
                                        for a in state['attempted_extensions']]
        result['last_stop_reason']=state['stop_reason']
        observed._write_json(out,result)
        print(f"{state['event']} path={len(state['path'])} attempts={len(state['attempted_extensions'])}",flush=True)
    try:
        fitted=core.fit_laminar_l1_sensitivity(training,validation_scenarios=validation,
            progress_callback=progress,**options)
        for point in fitted.path:
            train_mae,_,_=core.evaluate_l1_matrices(training,point.r_matrix,point.x_matrix)
            val_mae,_,_=core.evaluate_l1_matrices(validation,point.r_matrix,point.x_matrix)
            if not np.isclose(train_mae,point.train_mae,atol=1e-10,rtol=1e-7):
                raise ValueError('New prefix training objective does not reproduce')
            if not np.isclose(val_mae,point.validation_mae,atol=1e-10,rtol=1e-7):
                raise ValueError('New prefix validation objective does not reproduce')
        for point in old_prefix:
            r,x=core.build_matrices_from_atoms(len(terminals),point['support_indices'],
                point['r_values'],point['x_values'])
            train_mae,_,_=core.evaluate_l1_matrices(training,r,x)
            if not np.isclose(train_mae,point['train_mae'],atol=1e-10,rtol=1e-7):
                raise ValueError('Reused prefix training objective does not reproduce')
        unresolved=fitted.stop_reason in ('extension_not_proven_optimal','extension_without_incumbent','bound_expansion_limit')
        # Only now use truth to explain the saved algorithmic choices.
        truth=set(case.truth_clades)
        result.update({'status':'unresolved' if unresolved else 'prefix_completed',
            'stop_reason':fitted.stop_reason,'accepted_new_extension_count':len(fitted.path)-1,
            'new_milp_calls':len(fitted.attempted_extensions),
            'new_certified_extension_count':sum(core.solver_diagnostics_prove_optimality(a.diagnostics) for a in fitted.attempted_extensions),
            'truncated_by_predeclared_four_extension_cap':fitted.stop_reason=='max_atoms',
            'prefix_validation_selected_index':fitted.selected_path_index,
            'recovery_evaluation_is_posthoc':True,
            'leaf_plus_rnj_prefix':describe(old_prefix,truth,terminals),
            'rnj_only_prefix':describe(result['new_path'],truth,terminals),
            'finished_utc':now(),'elapsed_seconds':perf_counter()-start,
            'inputs_unchanged':all(sha(p)==h for p,h in hashes.items())})
        observed._write_json(out,result)
        print(json.dumps({k:result[k] for k in ('status','stop_reason','accepted_new_extension_count',
            'new_milp_calls','new_certified_extension_count','truncated_by_predeclared_four_extension_cap')},indent=2))
    except Exception as exc:
        result.update({'status':'failed','exception_type':type(exc).__name__,'exception':str(exc),
            'finished_utc':now(),'elapsed_seconds':perf_counter()-start})
        observed._write_json(out,result)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
