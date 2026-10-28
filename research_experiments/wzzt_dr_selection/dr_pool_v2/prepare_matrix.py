"""Frozen DR policies crossed with all used empirical response modes and V noise.

All raw or household-derived trajectories remain outside git. No identification
or price optimization is called here; planning reuses the historical mean plan.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from ..end_to_end import physical_case as physical, price_response as prices
from ..end_to_end.runner import _sha, _json, _pickle, _unpickle, _exclusive_lock, _now, _physical_options

STRATEGIES=('no_reward','fixed_P2_first_node','planned_d_opt','fixed_same_menu_first_node')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def prepare(source, output):
    source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    with _exclusive_lock(output,'dr_pool_factorial_v1'):
        if (output/'manifest.json').exists():
            previous=read(output/'manifest.json')
            if previous['status']=='complete':return previous
            raise RuntimeError('Incomplete physical preparation retained; inspect instead of rerunning')
        if (output/'protocol.json').exists():raise RuntimeError('Interrupted preparation: inspect saved states, do not silently repeat physical generation')
        state=read(source/'run_state.json');summary=read(source/'summary.json');plan=read(source/'plan.json')
        bundle=_unpickle(source/'privatebundle.pkl');cfg=state['provenance']['config'];labels=summary['case']['terminals']
        orig=read(source/'response_draws.json')
        assert summary['status']=='complete' and len(labels)==15
        mean=next(Path(p) for p in bundle['data_manifest']['source_hashes'] if p.endswith('control_training_menu_factors.csv'))
        signal=next(Path(p) for p in bundle['data_manifest']['source_hashes'] if p.endswith('price_signals.csv'))
        for p in (mean,signal):
            if _sha(p)!=bundle['data_manifest']['source_hashes'][str(p)]:raise ValueError('Response input hash changed')
        model=prices.MenuModel.load(mean,signal)
        selected=plan['selected_assignment'];active=[m for m in selected['menus'] if m!='no_reward']
        if active!=['P_15'] or selected['menus'][-1]!='P_15':raise ValueError('Locked design expected P15 at terminal115')
        assignments={s:read(source/'strategies'/s/'result.json')['assignment'] for s in STRATEGIES[:3]}
        assignments[STRATEGIES[3]]={'name':'node_0_P_15','menus':['P_15']+['no_reward']*(len(labels)-1)}
        candidate={a['name']:a for a in plan['candidates']}
        for a in assignments.values():
            if not candidate[a['name']]['feasible']:raise ValueError('Predeclared policy was infeasible at planning')
        for menu in ('P_2','P_15'):
            assert np.asarray(bundle['response_banks']['action'][menu]).shape==(2,24,1)
            assert np.asarray(bundle['response_banks']['validation'][menu]).shape==(1,24,1)
        inputs=[source/'run_state.json',source/'summary.json',source/'plan.json',source/'privatebundle.pkl',source/'response_draws.json',mean,signal]
        for day in summary['wzzt_training_dates']:inputs.append(source/'history'/(day+'.pkl'))
        for s in STRATEGIES[:3]:
            inputs.extend(source/'strategies'/s/f for f in ('result.json','action.pkl','validation.pkl'))
        hashes={str(p):_sha(p) for p in inputs}
        protocol={'created_at':_now(),'source':str(source),'input_sha256':hashes,
          'source_sha256':{str(Path(p).resolve()):_sha(p) for p in (__file__,physical.__file__,prices.__file__)},
          'policies':assignments,'noise':{'reference':.0002,'high_voltage_noise':.001},
          'response_modes':[{'index':0,'weight':.5},{'index':1,'weight':.5}],
          'response_coupling':'same U selects same P2/P15 index; singleton validation response; conditional empirical support only',
          'sample_sizes':[24,48,96],'planned_comparison_cells':48,'expected_unique_observation_tasks':42,
          'forecast':'same historical P/Q mean and original frozen full24h plan at every N/noise/mode',
          'truth_to_estimator':False,'no_reward_duplicates':'same observations across response modes, must reuse fits',
          'evaluation_scope':'one fixed household/date mapping, two empirical response modes, two meter-noise amplitudes; no independent population recovery probability',
          'test_day':'validation used in selection; no untouched predictive test day in source v1',
          'identification_scope':'finite-pool RNJ/WZZT; no free-support MILP'}
        _json(output/'protocol.json',protocol)
        net=physical.build_reference_case(cfg['case_key']);forecast=np.asarray(plan['forecast_p_kw'])
        limits={k:cfg['planning'][k] for k in ('power_limits_kw','max_daily_energy_change_fraction')}
        histories={};generation=[];manifest={'status':'running','scenarios':[],'protocol_sha256':_sha(output/'protocol.json')}
        for noise,level in protocol['noise'].items():
            physical_options=_physical_options(cfg);physical_options.update(v_noise_rel=level,root_meter_noise_rel=level)
            histdir=output/'physical_history'/noise;histdir.mkdir(parents=True,exist_ok=True)
            for day in summary['wzzt_training_dates']:
                original=_unpickle(source/'history'/(day+'.pkl'))
                if noise=='reference':sim=original
                else:
                    day_index=summary['history_dates'].index(day)
                    sim=physical.simulate_observations(net,original['truth']['P_kw'],original['truth']['Q_kvar'],
                       seed=cfg['seeds']['history']+day_index,name=original['observed']['name'],**physical_options)
                    for key in ('P_terminal','Q_terminal'):
                        if not np.array_equal(sim['observed'][key],original['observed'][key]):raise AssertionError('V-only noise condition changed planning P/Q')
                    generation.append({'kind':'history','noise':noise,'day':day,'checks':sim['checks']})
                _pickle(histdir/(day+'.pkl'),sim)
            for mode in (0,1):
                caseid=f'{noise}__response{mode}';caseout=output/'scenarios'/caseid;caseout.mkdir(parents=True,exist_ok=True)
                (caseout/'history').mkdir(exist_ok=True)
                for day in summary['wzzt_training_dates']:shutil.copyfile(histdir/(day+'.pkl'),caseout/'history'/(day+'.pkl'))
                newcfg=copy.deepcopy(cfg);newcfg['physical'].update(v_noise_rel=level,root_meter_noise_rel=level)
                _json(caseout/'run_state.json',{'status':'complete','provenance':{'config':newcfg},'source_v1':str(source)})
                _json(caseout/'plan.json',plan)
                draws={phase:{m:np.broadcast_to(np.asarray(bundle['response_banks'][phase][m])[mode if phase=='action' else 0],(24,len(labels))).copy() for m in ('P_2','P_15')} for phase in ('action','validation')}
                _json(caseout/'response_draws.json',{'used_indices':{'action':mode,'validation':0},'weight':.5,'coupling':'joint P2/P15 empirical mode','original_metadata':orig['bank_metadata']})
                for s,a in assignments.items():
                    dest=caseout/'strategies'/s;dest.mkdir(parents=True,exist_ok=True)
                    record={'strategy':s,'assignment':a,'planning_audit':candidate[a['name']],'periods':{},'stage_status':'physical_complete','core_call_count':0}
                    for phase in ('action','validation'):
                        existing=None
                        used_menus=[m for m in a['menus'] if m!='no_reward']
                        same_orig=all(mode==orig['draws']['action']['draws'][m]['index'] for m in used_menus)
                        if noise=='reference' and s in STRATEGIES[:3] and (s=='no_reward' or phase=='validation' or same_orig):
                            existing=source/'strategies'/s/(phase+'.pkl')
                        if mode==1 and s=='no_reward':existing=output/'scenarios'/f'{noise}__response0'/'strategies'/s/(phase+'.pkl')
                        if mode==1 and phase=='validation':existing=output/'scenarios'/f'{noise}__response0'/'strategies'/s/(phase+'.pkl')
                        if existing is not None:
                            data=_unpickle(existing);shutil.copyfile(existing,dest/(phase+'.pkl'))
                            response,sim=data['response'],data['simulation']
                        else:
                            p=bundle['actual_days'][phase];q=physical.synthetic_q_from_reference_pf(p,case_key=cfg['case_key'])
                            response=prices.realize_response(p,q,a,model,draws[phase],settlement_baseline_kw=forecast,budget_nok=cfg['planning']['budget_nok'],**limits)
                            pp=pd.DataFrame(response['p_kw'],index=p.index,columns=labels);qq=pd.DataFrame(response['q_kvar'],index=p.index,columns=labels)
                            sim=physical.simulate_observations(net,pp,qq,seed=cfg['seeds'][phase],name=s+'_'+phase,**physical_options)
                            _pickle(dest/(phase+'.pkl'),{'response':response,'simulation':sim})
                            generation.append({'kind':'period','noise':noise,'mode':mode,'strategy':s,'phase':phase,'checks':sim['checks']})
                        record['periods'][phase]={'total_payment_nok':response['total_payment_nok'],'payment_nok_per_node':response['payment_nok_per_node'],'credits_nok_per_node':response['credits_nok_per_node'],'response_audit':response['audit'],'physical_checks':sim['checks']}
                    _json(dest/'result.json',record)
                _json(caseout/'summary.json',dict(summary,status='complete',scope='DR v2 fixed-policy factorial observations; no identification yet'))
                manifest['scenarios'].append({'id':caseid,'source_run':str(caseout),'additional_strategy':str(caseout/'strategies'/STRATEGIES[3]),'noise':noise,'response_mode':mode,'response_weight':.5})
                _json(output/'manifest.json',manifest)
        for p,h in hashes.items():
            if _sha(p)!=h:raise AssertionError('Frozen input changed during physical generation')
        manifest.update(status='complete',physical_generation=generation,new_simulated_hours=24*len(generation),input_hashes_unchanged=True)
        _json(output/'manifest.json',manifest)
        return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=prepare(a.source,a.output);print(json.dumps({'status':r['status'],'scenarios':len(r['scenarios']),'new_simulated_hours':r.get('new_simulated_hours')}))
