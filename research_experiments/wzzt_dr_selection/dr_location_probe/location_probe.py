"""Bounded P15-location diagnosis. Prices and algorithm stay frozen.

Private trajectories and checkpoints are written only to the declared output.
Truth is used after fitting; best-location summaries are explicitly post hoc.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import platform
import sys
import traceback
import numpy as np
import pandas as pd
from research_experiments.wzzt_dr_selection.dr_pool_v2 import runner as r
from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
from research_experiments.wzzt_dr_selection.dr_pool_v2.matrix_runner import SharedFitter, observation_identity, BatchBoundary
from research_experiments.wzzt_dr_selection.end_to_end import physical_case as physical, price_response as prices
from research_experiments.wzzt_dr_selection.end_to_end.runner import _pickle, _physical_options


NOISES = {'reference': .0002, 'high_voltage_noise': .001}
COUNTS = (24, 96)


def make_protocol(source, old, output):
    plan = r.read(source/'plan.json')
    meta = r.read(source/'summary.json')
    cfg = r.read(source/'run_state.json')['provenance']['config']
    labels = meta['case']['terminals']
    candidates = {a['name']: a for a in plan['candidates']}
    actions = [{'node': None, 'name': 'no_reward', 'menus': ['no_reward']*len(labels),
                'planning_audit': candidates['no_reward']}]
    excluded = []
    for i, label in enumerate(labels):
        name = f'node_{i}_P_15'
        audit = candidates[name]
        if not audit['feasible']:
            excluded.append({'node': label, 'audit': audit})
            continue
        menus = ['no_reward']*len(labels); menus[i] = 'P_15'
        actions.append({'node': label, 'name': name, 'menus': menus, 'planning_audit': audit})
    inputs = [source/n for n in ('plan.json', 'summary.json', 'run_state.json', 'privatebundle.pkl')]
    inputs += [old/'physical_v1'/'manifest.json', old/'identification_v1'/'protocol.json']
    old_alg = r.read(inputs[-1])['algorithm_source_sha256']
    if old_alg != source_hashes():
        raise ValueError('Algorithm differs from frozen baseline')
    old_manifest = r.read(inputs[-2])
    for s in old_manifest['scenarios']:
        folder = Path(s['source_run'])
        inputs += [folder/'history'/(day+'.pkl') for day in meta['wzzt_training_dates']]
        for strategy in ('no_reward', 'planned_d_opt', 'fixed_same_menu_first_node'):
            inputs += [folder/'strategies'/strategy/(phase+'.pkl') for phase in ('action', 'validation')]
    contract = dict(version='p15_all_locations_v1', source=str(source), old_run=str(old), output=str(output),
        actions=actions, excluded_planning_actions=excluded, noise=NOISES, sample_sizes=COUNTS,
        response_modes=[{'index': 0, 'weight': .5}, {'index': 1, 'weight': .5}],
        planned_cells=len(actions)*8, expected_unique_fits=(len(actions)-1)*8+4,
        maximum_new_fits=(len(actions)-3)*8, search_seconds_per_start=25.,
        config=cfg, source_input_sha256={str(p):r.sha(p) for p in sorted(set(inputs))},
        algorithm_source_sha256=old_alg,
        controller_source_sha256={str(Path(__file__).resolve()):r.sha(__file__),
            str(Path(physical.__file__).resolve()):r.sha(physical.__file__),
            str(Path(prices.__file__).resolve()):r.sha(prices.__file__),
            str(Path(r.__file__).resolve()):r.sha(r.__file__)},
        models=list(r.MODEL_NAMES), truth_to_planner=False, truth_to_identifier=False,
        response_scope='two same-date group trajectories; empirical conditional modes, not population probability',
        reference_scope='all feasible SINGLE-node P15 assignments only; post-hoc best fixed node, not a deployable policy',
        stop_rule='no new dates, mappings, amplitude, price score or automatic extension; retain all failures',
        primary_metrics=['complete_effective_family', 'internal_f1', 'root_matched'],
        baseline='frozen D-opt P15 at115; no reward; uniform location over preflight feasible nodes',
        planning_uses_full24h=True, validation_role='model selection; not untouched predictive test',
        environment={'python':sys.version, 'executable':sys.executable, 'platform':platform.platform(),
                     'numpy':np.__version__, 'pandas':pd.__version__})
    contract['fingerprint'] = r.fingerprint(contract)
    return contract


def period(source, old, output, protocol, action, noise, mode, phase):
    node = action['node']
    strategy = {None:'no_reward', 101:'fixed_same_menu_first_node', 115:'planned_d_opt'}.get(node)
    scenario = old/'physical_v1'/'scenarios'/f'{noise}__response{mode}'
    if strategy is not None:
        path = scenario/'strategies'/strategy/(phase+'.pkl')
        return r.load_pickle(path), path, True
    target = output/'physical'/noise/action['name']/f'{phase}_{mode if phase=="action" else 0}.pkl'
    failure = target.with_suffix('.failure.json')
    if failure.exists():
        raise RuntimeError('Prior physical failure retained: '+str(failure))
    if target.exists():
        return r.load_pickle(target), target, True
    try:
        bundle = r.load_pickle(source/'privatebundle.pkl')
        cfg = protocol['config']; plan = r.read(source/'plan.json')
        hashes = bundle['data_manifest']['source_hashes']
        mean = next(Path(p) for p in hashes if p.endswith('control_training_menu_factors.csv'))
        signal = next(Path(p) for p in hashes if p.endswith('price_signals.csv'))
        for p in (mean, signal):
            if r.sha(p) != hashes[str(p)]: raise ValueError('Response source changed')
        model = prices.MenuModel.load(mean, signal)
        p = bundle['actual_days'][phase]
        q = physical.synthetic_q_from_reference_pf(p, case_key=cfg['case_key'])
        bank = np.asarray(bundle['response_banks'][phase]['P_15'])
        assert bank.shape == ((2 if phase=='action' else 1), 24, 1)
        factor = np.broadcast_to(bank[mode if phase=='action' else 0], p.shape).copy()
        response = prices.realize_response(p,q,action,model,{'P_15':factor},
            settlement_baseline_kw=np.asarray(plan['forecast_p_kw']),budget_nok=cfg['planning']['budget_nok'],
            power_limits_kw=cfg['planning']['power_limits_kw'],
            max_daily_energy_change_fraction=cfg['planning']['max_daily_energy_change_fraction'])
        pp = pd.DataFrame(response['p_kw'],index=p.index,columns=p.columns)
        qq = pd.DataFrame(response['q_kvar'],index=p.index,columns=p.columns)
        options = _physical_options(cfg)
        options.update(v_noise_rel=NOISES[noise],root_meter_noise_rel=NOISES[noise])
        simulation = physical.simulate_observations(physical.build_reference_case(cfg['case_key']),pp,qq,
            seed=cfg['seeds'][phase],name=action['name']+'_'+phase,**options)
        data = {'response':response,'simulation':simulation}
        _pickle(target,data)
        r.write(target.with_suffix('.receipt.json'),{'created_at':r.now(),'phase':phase,'node':node,
            'noise':noise,'mode':mode if phase=='action' else 0,'sha256':r.sha(target),
            'period':r._periods(data,data)['action']})
        return data,target,False
    except Exception as exc:
        r.write(failure,{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise


def summarize(protocol, output):
    rows = [r.read(p) for p in sorted((output/'records').glob('*.json'))]
    evaluated = [v for v in rows if v['status']=='evaluated']
    locations = [a['node'] for a in protocol['actions'] if a['node'] is not None]
    lookup = {(v['noise'],v['N'],v['node'],v['mode']):v for v in rows}
    comparisons = []
    for noise in NOISES:
        for n in COUNTS:
            for model in r.MODEL_NAMES:
                table = []
                for node in [None,*locations]:
                    pair = [lookup.get((noise,n,node,m)) for m in (0,1)]
                    complete = all(x and x['status']=='evaluated' and x['deployment_valid'] for x in pair)
                    item = {'node':node,'complete':complete}
                    for metric in protocol['primary_metrics']:
                        item[metric] = sum(.5*float(x['evaluation']['models'][model]['final'][metric]) for x in pair) if complete else None
                    if complete:
                        item['action_payment_nok'] = sum(.5*x['periods']['action']['payment_nok'] for x in pair)
                        item['validation_payment_nok'] = sum(.5*x['periods']['validation']['payment_nok'] for x in pair)
                    table.append(item)
                full = all(x['complete'] for x in table)
                comparison = {'noise':noise,'N':n,'model':model,'complete_location_set':full,'locations':table}
                if full:
                    base = table[0]; paid = table[1:]
                    comparison['uniform_location'] = {k:sum(x[k] for x in paid)/len(paid) for k in protocol['primary_metrics']}
                    comparison['posthoc_best_fixed_by_metric'] = {}
                    for k in protocol['primary_metrics']:
                        best = max(x[k] for x in paid)
                        comparison['posthoc_best_fixed_by_metric'][k] = {'value':best,'nodes':[x['node'] for x in paid if abs(x[k]-best)<1e-12]}
                    comparison['nodes_improving_any_metric_vs_no_reward'] = [x['node'] for x in paid if any(x[k]>base[k]+1e-12 for k in protocol['primary_metrics'])]
                    comparison['nodes_worsening_any_metric_vs_no_reward'] = [x['node'] for x in paid if any(x[k]<base[k]-1e-12 for k in protocol['primary_metrics'])]
                comparisons.append(comparison)
    caches = [r.read(p) for p in (output/'fit_cache').glob('*/cache_record.json')]
    return {'status':'complete' if len(rows)==protocol['planned_cells'] else 'partial',
        'planned_cells':protocol['planned_cells'],'recorded_cells':len(rows),'evaluated_cells':len(evaluated),
        'failed_cells':len(rows)-len(evaluated),'invalid_deployment_cells':sum(not v['deployment_valid'] for v in evaluated),
        'new_fit_count':len(caches),'new_complete_fits':sum(x['status']=='complete' for x in caches),
        'old_reused_cells':sum(v.get('old_cache_reused',False) for v in rows),
        'wall_limited_cells':sum(v.get('evaluation',{}).get('search_wall_limited',False) for v in rows),
        'records':rows,'comparisons':comparisons,'updated_at':r.now(),
        'source_inputs_unchanged':all(r.sha(p)==h for p,h in protocol['source_input_sha256'].items()),
        'scope':protocol['reference_scope'],'probability_scope':protocol['response_scope']}


def run(source, old, output, max_new_fits):
    source=Path(source).resolve();old=Path(old).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    contract=make_protocol(source,old,output)
    with r._exclusive_lock(output,contract['fingerprint']):
        lock=output/'protocol.json'
        if lock.exists():
            protocol=r.read(lock)
            if protocol['fingerprint']!=contract['fingerprint']:raise ValueError('Protocol changed')
        else:
            protocol=dict(contract,created_at=r.now());r.write(lock,protocol)
        shared=SharedFitter(output/'fit_cache',sources=protocol['algorithm_source_sha256'],max_new_fits=max_new_fits)
        source_meta=r.read(source/'summary.json')
        stopped=False
        for noise in NOISES:
            history_dir=old/'physical_v1'/'scenarios'/f'{noise}__response0'/'history'
            history=[r.observed_only(r.load_pickle(history_dir/(d+'.pkl'))) for d in source_meta['wzzt_training_dates']]
            for action in protocol['actions']:
                for mode in (0,1):
                    for n in COUNTS:
                        key=f'{noise}__{action["name"]}__mode{mode}__N{n}'
                        target=output/'records'/(key+'.json')
                        if target.exists():continue
                        row={'key':key,'noise':noise,'node':action['node'],'mode':mode,'N':n,'created_at':r.now(),'status':'running'}
                        try:
                            act,ap,_=period(source,old,output,protocol,action,noise,mode,'action')
                            val,vp,_=period(source,old,output,protocol,action,noise,mode,'validation')
                            train=r.subsample_training(history+[r.observed_only(act['simulation'])],n)
                            valid=[r.observed_only(val['simulation'])]
                            obskey,identity=observation_identity(train,valid,sources=protocol['algorithm_source_sha256'])
                            old_cache=old/'identification_v1'/'fit_cache'/obskey
                            if (old_cache/'cache_record.json').exists():
                                meta=r.read(old_cache/'cache_record.json')
                                if meta['identity']!=identity or meta['status']!='complete':raise ValueError('Old cache not matching/complete')
                                fitpath=old_cache/'result.json'
                                if r.sha(fitpath)!=meta['result_sha256']:raise ValueError('Old result changed')
                                fit=r.read(fitpath); row['old_cache_reused']=True
                            else:
                                fit=shared(train,valid,output/'requests'/key)
                                fitpath=Path(fit['matrix_cache']['cache_result']);row['old_cache_reused']=False
                            periods=r._periods(act,val)
                            deploy=all(x['response_audit']['valid'] and x['physical_checks']['usable_ac_solution'] and x['physical_checks']['voltage_limits_pass'] for x in periods.values())
                            truth=act['simulation']['truth']['truth_supports']
                            assert truth==val['simulation']['truth']['truth_supports']
                            row.update(status='evaluated',observation_key=obskey,fit_result=str(fitpath),
                                fit_sha256=r.sha(fitpath),physical_files={str(p):r.sha(p) for p in (ap,vp)},
                                periods=periods,deployment_valid=deploy,
                                evaluation=r.evaluate_fit(fit,truth,list(train[0]['P_terminal'].columns)),
                                identifier_seconds=fit.get('elapsed_seconds',fit.get('seconds')))
                        except BatchBoundary:
                            stopped=True;break
                        except Exception as exc:
                            row.update(status='failed',exception={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
                        row['finished_at']=r.now();r.write(target,row)
                        r.write(output/'summary.json',summarize(protocol,output))
                        print(json.dumps({'cell':key,'status':row['status'],'new_fits':shared.new_count}),flush=True)
                    if stopped:break
                if stopped:break
            if stopped:break
        result=summarize(protocol,output);result['batch_new_fits']=shared.new_count
        result['stopped_at_batch_limit']=stopped;r.write(output/'summary.json',result)
        print(json.dumps({k:result[k] for k in ('status','recorded_cells','planned_cells','failed_cells','new_fit_count','old_reused_cells')}),flush=True)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--old',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-new-fits',type=int,default=8)
    a=p.parse_args()
    if a.max_new_fits<1:raise ValueError('positive batch size required')
    result=run(a.source,a.old,a.output,a.max_new_fits)
    raise SystemExit(0 if result['failed_cells']==0 else 2)
