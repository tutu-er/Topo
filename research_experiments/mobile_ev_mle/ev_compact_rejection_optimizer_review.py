"""Saved-state branch/SSE replay of the four-condition optimizer diagnostic.

This entry point does not call any fitter, generator or selector.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
from pathlib import Path
import numpy as np
from .information_fit_recovery import read_json,write_json,independent_replay,exact_candidate
from .information_replay import _obs,_meter
from .likelihood import concat_observations
from . import ev_original_review as core

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review_directory(directory):
    directory=Path(directory);protocol=read_json(directory/'protocol.json');summary=read_json(directory/'summary.json')
    checks=[];conditions=[];inputs={};errors=[]
    for path,expected in protocol['input_sha256'].items():
        core.require(sha(path)==expected,'immutable parent input changed: '+path);inputs[path]=expected
    for name,expected in protocol['auxiliary_source_sha256'].items():
        core.require(sha(HERE/name)==expected,'diagnostic auxiliary source changed: '+name)
    parent=next(Path(p).parent for p in inputs if p.endswith('freeze.json'))
    freeze=read_json(parent/'freeze.json');source=core.source_audit(parent,freeze)
    core.require(source['current_source_matches'] and source['source_count']==72,'source72 mismatch')
    core.require(protocol['source72_sha256']==freeze['source_sha256'],'source manifest differs')
    csv_rows={(int(r['seed']),r['method'],r['candidate']):r for r in csv.DictReader((directory/'candidate_comparison.csv').open(encoding='utf-8-sig'))}
    original_count=0;new_count=0;restart_count=0;ac_calls=0;maxeq=0.;maxbalance=0.;maxpredict=0.
    for target in protocol['targets']:
        seed,method=target['seed'],target['method'];rawpath=parent/f'trial_{seed}.json.gz';trial=read_json(rawpath)
        record=next(r for r in trial['rows'] if r['method']==method)
        path=directory/f'condition_{seed}_{method}.json.gz';saved=read_json(path)
        core.require(saved['input_raw_sha256']==sha(rawpath),'condition raw binding')
        core.require(saved['history_observations']==trial['history'] and saved['future_observations']==record['observations'],'same saved observation requirement')
        core.require(saved['joint_meter']==trial['measurement_models']['jm'],'same joint meter requirement')
        core.require(saved['source_interface']==trial['source_interface']==1,'source interface')
        options=saved['options'];box=trial['config']['physical']['asset_bounds']
        core.require(options['r_bounds']==box['r_bounds'] and options['x_bounds']==box['x_bounds'],'saved asset box')
        joint=concat_observations(_obs(trial['history']),_obs(record['observations']));meter=_meter(saved['joint_meter'])
        original_losses={};best_losses={};candidate_checks=[]
        core.require(len(saved['candidates'])==4 and {c['candidate'] for c in saved['candidates']}=={c['name'] for c in record['fits']},'all candidate coverage')
        bank=[r['name'] for r in trial['proposal']['bank']]
        for item in saved['candidates']:
            name=item['candidate'];original=item['original'];run=item['recovery'];candidate=exact_candidate(original,joint.n)
            old=next(r for r in record['fits'] if r['name']==name)
            core.require({k:v for k,v in original.items() if k!='independent_replay'}==old,'original retained state changed')
            core.require(run['seed']==trial['seeds']['policy']+802+bank.index(name),'restart seed rule')
            core.require(run['options']['multistart']==8 and run['options']['max_nfev']==500 and len(run['starts'])==8,'eight-start expanded budget')
            states=[('original',original)];original_count+=1
            for start in run['starts']:
                restart_count+=1
                for key in ('initial','returned','last_finite_state'):
                    if key in start:states.append((f"restart_{start['index']}_{key}",start[key]));new_count+=1
            if run.get('returned_best_fit') is not None:states.append(('returned_best_fit',run['returned_best_fit']));new_count+=1
            attained=[]
            for label,state in states:
                replay=independent_replay(candidate,state,joint,meter)
                for key in ('r','x'):
                    lo,hi=options[key+'_bounds'];value=np.asarray(state[key],float)
                    core.require(np.all(value>=np.asarray(lo)-1e-12) and np.all(value<=np.asarray(hi)+1e-12),'original box in replay')
                core.require(replay['success'],'independent saved-state replay')
                if 'independent_replay' in state:
                    core.close(replay['full_SSE'],state['independent_replay']['full_SSE'],'saved vs repeat SSE')
                    core.close(replay['loglik'],state['independent_replay']['loglik'],'saved vs repeat Gaussian loglik')
                maxeq=max(maxeq,replay['AC_equation_residual']);maxbalance=max(maxbalance,replay['AC_balance_residual'])
                maxpredict=max(maxpredict,max(replay['prediction_max_abs_errors'].values(),default=0.))
                attained.append((replay['full_SSE'],label,replay))
            minimum=min(x[0] for x in attained);best=run['best_feasible_attained']
            core.close(best['SSE'],minimum,'minimum feasible incumbent including original')
            core.close(best['loglik'],next(a[2]['loglik'] for a in attained if a[1]==best['label']),'best Gaussian loglik')
            core.require(best['SSE']<=original['loss']+1e-7,'retained original upper bound')
            original_losses[name]=original['loss'];best_losses[name]=best['SSE'];ac_calls+=run['attempted_AC_calls']['attempted']
            line=csv_rows[(seed,method,name)]
            for key,value in [('original_SSE',original['loss']),('best_attained_SSE',best['SSE']),('improvement',original['loss']-best['SSE'])]:
                core.close(float(line[key]),value,'CSV replay: '+key)
            candidate_checks.append(dict(candidate=name,states_replayed=len(states),original_SSE=original['loss'],best_attained_SSE=best['SSE'],improvement=original['loss']-best['SSE']))
        true=target['true_name'];old_gap=original_losses[true]-min(original_losses.values());new_gap=best_losses[true]-min(best_losses.values())
        core.close(saved['summary']['original_true_gap'],old_gap,'old true gap')
        core.close(saved['summary']['new_best_attained_true_gap'],new_gap,'new true gap')
        core.require(saved['summary']['new_calibrated_decision'] is None and saved['summary']['original_decision_retained_unchanged'],'posthoc no reclassification')
        conditions.append(dict(seed=seed,method=method,true_name=true,original_true_gap=old_gap,new_best_attained_true_gap=new_gap,gap_change=new_gap-old_gap,candidates=candidate_checks))
    core.require(len(csv_rows)==16 and len(conditions)==4,'four histories and sixteen candidates')
    core.require(restart_count==summary['restart_attempts'] and ac_calls==summary['AC_calls'],'call/attempt accounting')
    return dict(success=True,errors=errors,policy_id=protocol['policy_id'],seeds=[r['seed'] for r in protocol['targets']],
        independent_histories=4,method_conditions=4,candidate_fits=16,restart_attempts=restart_count,
        original_states_replayed=original_count,new_saved_states_replayed=new_count,total_states_replayed=original_count+new_count,
        AC_calls=ac_calls,max_AC_equation_residual=maxeq,max_AC_balance_residual=maxbalance,max_saved_prediction_error=maxpredict,
        source72_current_matches=True,parent_input_sha256=inputs,protocol_sha256=sha(directory/'protocol.json'),
        diagnostic_summary_sha256=sha(directory/'summary.json'),candidate_CSV_sha256=sha(directory/'candidate_comparison.csv'),
        diagnostic_condition_sha256={p.name:sha(p) for p in directory.glob('condition_*.json.gz')},
        reviewer_sha256=sha(Path(__file__)),conditions=conditions,new_fits=0,new_selector_runs=0,
        scope='separate branch/SSE/loglik arithmetic on saved finite states; no global lower bound or recalibrated decisions')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True,type=Path);args=parser.parse_args()
    result=review_directory(args.directory);write_json(args.directory/'review_full.json',result)
    print({k:result[k] for k in ('success','total_states_replayed','restart_attempts','AC_calls')})


if __name__=='__main__':main()
