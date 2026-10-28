"""Five fixed-family LPs for the saved RNJ-only first decision; no new MILP."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys
import numpy as np
import scipy


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def semantic_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def frozen_first(snapshot):
    fields=('iteration','support_labels','r_values','x_values','train_mae',
            'validation_mae','validation_se','accepted_gain','solver')
    return {key:snapshot['new_path'][1][key] for key in fields}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--n0-run', type=Path, required=True)
    args=parser.parse_args(argv)
    repo=args.repo.resolve()
    sys.path.insert(0, str(repo));sys.path.insert(0, str(repo/'rnj_wzzt_core'))
    from research_experiments.wzzt_dr_selection.baseline_alignment import bound_dependence as bd
    folder=repo/'research_experiments/wzzt_dr_selection/evidence/baseline_bounds_init_v3'
    protocol_path=folder/'rnj_only_bound_protocol.json'
    output=folder/'rnj_only_bound_results.json'
    lock=folder/'rnj_only_bound.lock'
    if protocol_path.exists() or output.exists():
        raise FileExistsError('Existing five-LP evidence; no duplicate solves')
    initial_path=folder/'initialization.json'
    snapshot_bytes=initial_path.read_bytes();snapshot=json.loads(snapshot_bytes)
    first=frozen_first(snapshot)
    base=snapshot['selected_rnj_support_labels']
    expected=[101,102,*range(106,116)]
    if len(base)!=2 or first['support_labels'] != [*base,expected]:
        raise ValueError('Unexpected RNJ-only predecessor or first winner')
    if not first['solver']['numerical_optimum_certified'] or first['accepted_gain']<=0:
        raise ValueError('First accepted point must be a numerically optimal winner')
    original_protocol=folder/'bounds_protocol.json'
    original_results=folder/'bounds_results.json'
    old=json.loads(original_protocol.read_text(encoding='utf-8'))
    files=[original_protocol,original_results,args.n0_run/'prepared_case.pkl']
    input_hashes={str(p.resolve()):bd.sha(p) for p in files}
    if input_hashes[str(files[-1].resolve())] != old['input_sha256'][str(files[-1].resolve())]:
        raise ValueError('Frozen observations changed since original seven LPs')
    if input_hashes[str(files[-1].resolve())] != snapshot['input_sha256'][str(files[-1].resolve())]:
        raise ValueError('RNJ-only run used different observations')
    tasks=[{'id':'rnj_only_actual_first_winner','supports':first['support_labels'],
            'saved_bounded2_first':first}]
    for previous in old['tasks']:
        if 'post_hoc_true_alternative' in previous:
            block=previous['post_hoc_true_alternative']
            tasks.append({'id':'rnj_only_'+previous['id'],'supports':[*base,block],
                'post_hoc_true_alternative':block,'saved_bounded2_first':None})
    if len(tasks)!=5 or len({tuple(map(tuple,t['supports'])) for t in tasks})!=5:
        raise ValueError('Exactly five distinct preregistered families required')
    with files[-1].open('rb') as stream:
        case,_=pickle.load(stream)
    training,validation=case.training,case.validation
    a,y=bd.design(training,first['support_labels'])
    av,yv=bd.design(validation,first['support_labels'])
    saved_weights=np.r_[first['r_values'],first['x_values']]
    reconstructed={'train_mae':float(np.mean(np.abs(y-a@saved_weights))),
                   'validation_mae':float(np.mean(np.abs(yv-av@saved_weights))),
                   'max_weight':float(saved_weights.max())}
    for key in ('train_mae','validation_mae'):
        if not np.isclose(reconstructed[key],first[key],atol=1e-10,rtol=1e-7):
            raise ValueError('Saved first winner does not match independent design')
    source_paths=[Path(__file__),Path(bd.__file__)]
    protocol={'created_utc':bd.now(),'question':'Under the two-RNJ predecessor, does removing cap2 allow any of the four previously named true additions to beat the actual first winner?',
        'scope':'fixed-support post hoc diagnosis; no global unbounded support search',
        'planned_new_lp_count':5,'previous_seven_lp_count':7,'combine_denominators':False,
        'new_milp_calls':0,'new_observations':0,'new_random_realizations':0,
        'truth_use':'Four previously listed true alternatives are diagnostic only, never estimator targets',
        'solver':'scipy.optimize.linprog/highs','objective':'raw mean absolute residual; all terminal rows; no intercept',
        'weight_bounds':[0,None],'per_lp_limit_seconds':1800.,
        'validation':'same frozen observations; evaluation only',
        'first_point_extraction_utc':bd.now(),
        'growing_initialization_file':str(initial_path.resolve()),
        'initialization_sha256_at_extraction':hashlib.sha256(snapshot_bytes).hexdigest(),
        'frozen_first_point':first,'frozen_first_point_sha256':semantic_sha(first),
        'growing_file_verification':'Check the extracted first-point content after solving, not whole-file immutability',
        'initial_supports':base,'tasks':tasks,'saved_first_independent_recomputation':reconstructed,
        'source_sha256':{str(p.resolve()):bd.sha(p) for p in source_paths},
        'input_sha256':input_hashes,
        'original_solver_tests':'Reuse existing test_wzzt_bound_dependence.py result: 3 passed; no new unit-test solves',
        'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()}}
    with lock.open('x',encoding='utf-8') as stream:
        stream.write(bd.now())
    write(protocol_path,protocol)
    result={'status':'running','protocol_sha256':bd.sha(protocol_path),'planned_new_lp_count':5,
            'previous_seven_lp_count':7,'combine_denominators':False,'attempted_lp_count':0,
            'completed_lp_count':0,'failed_or_unresolved_lp_count':0,'new_milp_calls':0,'records':[]}
    write(output,result)
    try:
        for task in tasks:
            a,y=bd.design(training,task['supports']);av,yv=bd.design(validation,task['supports'])
            rec={'id':task['id'],'supports':task['supports'],'observation_count':len(y)}
            result['attempted_lp_count']+=1
            write(output,result)
            fit=bd.solve_nonnegative_l1(a,y,time_limit=1800.,upper_bound=None)
            rec['unbounded_above']=fit
            if fit['weights'] is not None:
                weights=np.asarray(fit['weights']);k=len(task['supports'])
                rec.update({'r_values':weights[:k].tolist(),'x_values':weights[k:].tolist(),
                    'train_mae':float(np.mean(np.abs(y-a@weights))),
                    'validation_mae':float(np.mean(np.abs(yv-av@weights)))})
            result['records'].append(rec)
            result['completed_lp_count']+=fit['status']=='complete'
            result['failed_or_unresolved_lp_count']+=fit['status']!='complete'
            write(output,result)
        checks={'immutable_inputs_unchanged':all(bd.sha(p)==h for p,h in input_hashes.items()),
                'sources_unchanged':all(bd.sha(p)==h for p,h in protocol['source_sha256'].items()),
                'growing_first_point_unchanged':semantic_sha(frozen_first(json.loads(initial_path.read_bytes())))==protocol['frozen_first_point_sha256']}
        winner=result['records'][0]
        if result['completed_lp_count']==5:
            result['comparison']={'actual_first_cap2_train_mae':first['train_mae'],
                'actual_first_unbounded_train_mae':winner['train_mae'],
                'actual_first_train_gain_from_removing_cap':first['train_mae']-winner['train_mae'],
                'alternatives':[{'id':r['id'],'block':t['post_hoc_true_alternative'],
                    'unbounded_mae_minus_actual_winner':r['train_mae']-winner['train_mae'],
                    'dual_objective_minus_actual_winner_primal':r['unbounded_above']['dual_objective']-winner['train_mae']}
                    for r,t in zip(result['records'][1:],tasks[1:])],
                'scope':'Only these four diagnostic true additions versus this known feasible false winner, under the original two-RNJ predecessor; not the global unbounded winner'}
        result.update({'status':'complete' if result['completed_lp_count']==5 and all(checks.values()) else 'complete_with_unresolved',
                       'verification':checks,'finished_utc':bd.now()})
        write(output,result)
        print(json.dumps({'status':result['status'],'completed':result['completed_lp_count'],
            'verification':checks,'comparison':result.get('comparison'),
            'records':[{'id':r['id'],'train_mae':r.get('train_mae'),'max_weight':r['unbounded_above'].get('max_weight'),
                'checks':r['unbounded_above'].get('checks')} for r in result['records']]},indent=2))
    except Exception as exc:
        result.update({'status':'failed','error':repr(exc),'finished_utc':bd.now()})
        write(output,result)
        raise
    finally:
        lock.unlink()


if __name__=='__main__':
    main()
