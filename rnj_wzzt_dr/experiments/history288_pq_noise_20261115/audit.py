"""Independent arithmetic audit of history-density/PQ-error experiments.

Only sealed arrays are read. No identifier, optimizer, production scorer,
equipment simulator or AC solver is invoked. Truth is post-execution evaluation.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

from ..equipment_flex_20261115 import audit as equipment
from ..single_target_20261115 import audit as single

base,panel,prob,previous,ROOT=equipment.base,equipment.panel,equipment.prob,equipment.previous,equipment.ROOT


def standard_noise(seed,original_index,history_repeat,split,path,n):
    """Recreate the locked streams independently of the data generator."""
    size=288 if split==0 else 96
    return {key:np.random.default_rng(np.random.SeedSequence([
        int(seed),2718,int(original_index),int(history_repeat),int(split),int(path),component
    ])).normal(size=(size,) if key=='root' else (size,n))
        for key,component in (('P',2),('Q',3),('V',4),('root',5))}


def noisy_observation(clean,indices,noise,pq_sd,voltage_sd=.0005):
    """P/Q and squared voltage observations; no synthetic error is added twice."""
    if not np.isfinite([pq_sd,voltage_sd]).all() or min(pq_sd,voltage_sd)<0:
        raise ValueError('Nonnegative finite error scales required')
    out={}
    for key in ('P','Q','V','root'):
        original=np.asarray(clean[key])[indices]
        z=np.asarray(noise[key])
        if original.shape!=z.shape:raise ValueError('Noise/clean dimensions differ')
        out[key]=original*(1+(pq_sd if key in ('P','Q') else voltage_sd)*z)
    out['Y']=out['root'][:,None]**2-out['V']**2
    return out


def validate_timeline(timestamps):
    values=[datetime.fromisoformat(str(t)) for t in timestamps]
    if len(values)!=384 or any(t.tzinfo is None for t in values):
        raise ValueError('Expected 384 timezone-aware timestamps')
    if any((b-a).total_seconds()!=900 for a,b in zip(values,values[1:])):
        raise ValueError('Expected consecutive quarter-hour observations')
    return dict(train_indices=list(range(192)),validation_indices=list(range(192,288)),
        evaluation_indices=list(range(288,384)),history_rows=288,
        training_rows=192,validation_rows=96,evaluation_rows=96,
        first=str(timestamps[0]),last=str(timestamps[-1]),
        cumulative_scoring='96 held-out validation plus 96 evaluation points; 192 training points excluded')


def bank_values(native,labels,train,validation,truth):
    """Rebuild frozen parameters and compare both original/enriched banks."""
    n=len(labels);Hs=[];active=[];rows=[]
    proper=lambda groups:{g for g in groups if 1<len(g)<n}
    for index,candidate in enumerate(native['frozen_candidates']):
        point=candidate['point'];H=np.zeros((n,2*n));supports=set()
        for support,r,x in zip(point['supports'],point['r'],point['x'],strict=True):
            block=np.ix_(support,support);H[:,:n][block]+=r;H[:,n:][block]+=x
            if max(r,x)>native['config']['zero_tol']:supports.add(tuple(support))
        P=np.c_[train['P'],train['Q']];V=np.c_[validation['P'],validation['Q']]
        st=float(np.square(train['Y']-P@H.T).sum());sv=float(np.square(validation['Y']-V@H.T).sum())
        Hs.append(H);active.append(supports)
        rows.append(dict(native_index=index,training_sse=st,training_mse=st/train['Y'].size,
            validation_sse=sv,validation_mse=sv/validation['Y'].size,
            internal_count=sum(1<len(g)<n for g in point['supports']),
            internal_correct=proper(supports)==proper(truth),full_support_exact=supports==truth,
            internal_missing=len(proper(truth)-proper(supports)),internal_extra=len(proper(supports)-proper(truth)),
            full_support_missing=len(truth-supports),full_support_extra=len(supports-truth)))
    return np.asarray(Hs),active,rows


def strict_winner(sse,internal,cells,reference_mse,tie):
    scaled=np.asarray(sse)/cells/reference_mse
    eligible=np.flatnonzero(scaled<=scaled.min()+tie)
    return min(map(int,eligible),key=lambda j:(internal[j],scaled[j],j))


class Audit(equipment.Audit):
    def __init__(self):
        super().__init__();self.data_records=[];self.fit_records=[];self.cross_level={};self.fixed_world={};self.validated_data={}

    def data(self,entry,protocol):
        """Rebuild observations from sealed clean arrays and independent streams."""
        clean=base.native.arrays(self.checked(entry['physical_cache']['original']))
        timeline=validate_timeline(clean['timestamps']);n=len(clean['labels'])
        self.require(clean['P'].shape==clean['Q'].shape==clean['V'].shape==(384,n),'Clean cache shape differs')
        self.require(all(np.isfinite(clean[k]).all() for k in ('P','Q','V','root')),'Nonfinite physical cache')
        train=base.native.arrays(self.checked(entry['files']['train.npz']))
        validation=base.native.arrays(self.checked(entry['files']['validation.npz']))
        self.require(train['Y'].shape==(192,n) and validation['Y'].shape==(96,n),'192/96 history split differs')
        self.require(entry['meter']['additional_V_relative_sd']==.0005,'Voltage/root error changed with PQ level')
        self.require(entry.get('forecast_relative_sd',.05)==.05,'Forecast SD changed')
        history=base.native.arrays(self.checked(entry['history_noise']))
        evaluation=base.native.arrays(self.checked(entry['evaluation_noise']))
        standard=base.native.arrays(self.checked(entry['standard_noise']))
        expected=standard_noise(entry['noise_seed'],entry['original_input_index'],entry['history_repeat'],0,0,n)
        for key in expected:
            self.same('history_standard_noise_'+key,history[key],expected[key],atol=0,rtol=0)
            self.require(evaluation[key].shape==(16,96)+( (() if key=='root' else (n,)) ),'Evaluation noise dimensions differ')
            self.same('standard_noise_composition_'+key,standard[key],np.concatenate((history[key],evaluation[key][0])),atol=0,rtol=0)
        for path in range(16):
            expected=standard_noise(entry['noise_seed'],entry['original_input_index'],entry['history_repeat'],1,path,n)
            for key in expected:self.same('evaluation_standard_noise_'+key,evaluation[key][path],expected[key],atol=0,rtol=0)
        sampling=base.read(self.checked(entry['files']['observations_sealed.json']))
        self.require(sampling['new_day_used_for_fitting'] is False,'Future observations used for fitting')
        allobs=noisy_observation(clean,slice(None),standard,entry['pq_relative_sd'])
        splits={'train':(0,192,train),'validation':(192,288,validation),
                'new_day':(288,384,base.native.arrays(self.checked(entry['files']['new_day.npz'])))}
        for name,(a,b,observed) in splits.items():
            self.require(sampling['sampling'][name]['indices']==list(range(a,b)),'Sampling indices differ')
            self.require(sampling['sampling'][name]['timestamps']==clean['timestamps'][a:b].tolist(),'Sampling timestamps differ')
            self.same('labels_'+name,observed['labels'],clean['labels'],atol=0,rtol=0)
            for key in allobs:self.same('reconstructed_'+name+'_'+key,observed[key],allobs[key][a:b],atol=0,rtol=0)
        group=(entry['case'],entry['history_repeat'])
        refs={key:entry[key] for key in ('physical_cache','resource_context','equipment_spec','forecast_ref','history_noise','evaluation_noise')}
        if group in self.cross_level:self.require(refs==self.cross_level[group],'Resources/physical/noise sources changed across PQ levels')
        else:self.cross_level[group]=refs
        world=(entry['case'],entry['start_date']);fixed={k:refs[k] for k in ('physical_cache','resource_context','equipment_spec','forecast_ref')}
        if world in self.fixed_world:self.require(fixed==self.fixed_world[world],'Physical world/forecast/equipment changed across historical repeats')
        else:self.fixed_world[world]=fixed
        self.data_records.append(dict(**panel.identity(entry),history_repeat=entry['history_repeat'],pq_relative_sd=entry['pq_relative_sd'],
            timeline=timeline,source_clean=entry['physical_cache']['original'],additional_voltage_relative_sd=.0005,
            all_observations_reconstructed=True,noise_streams_reconstructed=17,source_noise_is_simulated_addition=True))
        return clean,train,validation,timeline

    def fits(self,entry,ctx,train,validation):
        sealed=base.read(self.checked(entry['files']['candidates_sealed.json']))
        observations=base.read(self.checked(entry['files']['observations_sealed.json']))
        result=base.read(self.checked(entry['files']['result.json']))
        self.require(sealed['native']==entry['native_copy'] and sealed['score']==entry['score'],'Candidate seal differs')
        self.require(sealed['truth_used'] is False and sealed['new_day_values_used'] is False,'Truth/future used for candidate fit')
        self.require(sealed['training']==observations['observations']['train'] and sealed['validation']==observations['observations']['validation'],'Fitted observations differ')
        original=base.read(self.checked(entry['files']['original_101_native.json']))
        enriched=ctx['native'];count=len(original['frozen_candidates'])
        self.require(original['frozen_candidates']==enriched['frozen_candidates'][:count],'Original fitted points changed during enrichment')
        search=base.read(self.checked(entry['files']['search_60s.json']))
        self.require(len(enriched['frozen_candidates'])==count+int(search.get('closed_point') is not None),'Unexpected enrichment count')
        if search.get('closed_point') is not None:
            self.require(enriched['frozen_candidates'][-1]['point']==search['closed_point'],'New point differs from sealed same-pool search')
            events=[r for r in original['events'] if r.get('stage')=='E' and all(k in r for k in ('pool','keep','point'))]
            self.require(bool(events) and search['original_E_pool']==events[-1]['pool'] and search['original_E_keep']==events[-1]['keep']
                and search['original_E_warm']==events[-1]['point'],'Additional search changed the original recorded pool/start')
            self.require(set(map(tuple,search['closed_point']['supports']))<=set(map(tuple,events[-1]['pool'])),'Added support outside original candidate pool')
        self.require(not search['truth_used'] and not search['validation_used'] and not search['new_day_used'],'Extra search accessed evaluation information')
        models={}
        for kind,native in (('original',original),('enriched',enriched)):
            H,active,rows=bank_values(native,ctx['labels'],train,validation,ctx['truth'])
            winner=strict_winner([r['validation_sse'] for r in rows],[r['internal_count'] for r in rows],validation['Y'].size,native['reference_mse'],native['config']['validation_tie'])
            self.require(winner==native['selected']['index']==result['methods'][kind]['history_selected'],'Historical selection differs')
            recorded=result['methods'][kind]
            self.require(recorded['bank_has_internal_truth']==any(r['internal_correct'] for r in rows),'Recorded internal coverage differs')
            self.require(recorded['bank_has_full_truth']==any(r['full_support_exact'] for r in rows),'Recorded full support coverage differs')
            self.require(recorded['history_topology']['internal_exact']==rows[winner]['internal_correct'],'Historical topology result differs')
            models[kind]=dict(candidates=rows,bank_size=len(rows),history_selected=winner,
                history_correct=rows[winner]['internal_correct'],history_full_support_exact=rows[winner]['full_support_exact'],
                truth_in_native_bank=any(r['internal_correct'] for r in rows),full_support_truth_in_native_bank=any(r['full_support_exact'] for r in rows),
                actual_fit_seconds=recorded['seconds'],stop_reason=recorded['stop_reason'],total_cap_exceeded=recorded['total_cap_exceeded'],
                QP_count=recorded['recorded_native_QP_count'],MIQP_count=recorded['MIQP_count'],native=entry['files']['original_101_native.json' if kind=='original' else '101_native.json'])
            if kind=='enriched':
                self.same('full_native_H',H,ctx['H'],atol=1e-14)
                self.same('full_native_validation_sse',[r['validation_sse'] for r in rows],ctx['history'],atol=1e-13)
                self.require(ctx['score']['cells']==96*ctx['n'],'Historical score counted training data')
                groups={}
                for i,a in enumerate(active):groups.setdefault(tuple(sorted(g for g in a if 1<len(g)<ctx['n'])),[]).append(i)
                reps=sorted(min(ids,key=lambda j:(rows[j]['training_mse'],j)) for ids in groups.values())
                self.require([r['index'] for r in ctx['score']['models']]==reps,'Representative selection differs from training-only rule')
                self.same('historical_sigma2',ctx['score']['sigma2'],min(rows[i]['validation_mse'] for i in reps),atol=1e-15)
        self.fit_records.append(dict(**panel.identity(entry),history_repeat=entry['history_repeat'],pq_relative_sd=entry['pq_relative_sd'],
            models=models,source_candidate_seal=entry['files']['candidates_sealed.json'],source_extra_search=entry['files']['search_60s.json'],
            scoring_rows='96 validation plus96 evaluation; training192 excluded',extra_search_status=search['status']))

    def coverage(self,entry,ctx,model):
        proper=lambda groups:{g for g in groups if 1<len(g)<ctx['n']}
        truth=[i for i,a in enumerate(ctx['active']) if proper(a)==proper(ctx['truth'])]
        targets=[model['representatives'][t] for t in model['targets']]
        return dict(**panel.identity(entry),history_repeat=entry['history_repeat'],pq_relative_sd=entry['pq_relative_sd'],
            native_points=len(ctx['H']),representative_families=len(model['representatives']),true_native_indices=truth,
            truth_in_native_bank=bool(truth),truth_in_target_set=bool(set(truth)&set(targets)),
            selected_target_native_indices=targets,history_native_selected=ctx['target'],history_correct=ctx['target'] in truth,
            full_support_truth_in_native_bank=any(a==ctx['truth'] for a in ctx['active']),measured_terminals=ctx['n'])

    def band_arrays(self,entry,seal):
        train=base.native.arrays(self.checked(seal['train']));sampling=base.read(self.checked(seal['sampling_source']))
        self.require(seal['sampling_source']==entry['files']['observations_sealed.json'],'Band training source changed')
        b=equipment.band.reconstruct_band(train['P'],train['Q'],sampling['sampling']['train']['timestamps'],sampling['sampling']['new_day']['timestamps'][::4])
        arrays=base.native.arrays(self.checked(seal['bandwidth_npz']))
        self.same('new_train_only_band_slope',arrays['kappa'],b['slope'],atol=1e-10)
        self.same('new_train_only_band_width',arrays['band_kvar'],b['width_pu']*entry['base_kw'],atol=1e-10)
        self.require(np.all(arrays['counts']==8),'Two15-minute training days require8points per clock hour')
        return arrays

    def randomness(self,folder,entry,protocol,config,repeat):
        random=base.native.arrays(self.file(folder/f'execution{repeat:02d}'/'paired_randomness.npz'))
        source=base.native.arrays(self.checked(entry['evaluation_noise']))
        for i,key in enumerate(('P','Q','V','root')):
            self.same('sealed_evaluation_noise_'+key,random['noise'+str(i)],source[key][repeat],atol=0,rtol=0)
        eta=np.empty(len(config['resources']));lineage=[]
        for device in config['devices']:
            key=['equipment_flex_whole_device_v1',int(protocol['seed']),int(entry['original_input_index']),
                int(entry['history_repeat']),int(repeat),device['id']]
            digest=hashlib.sha256(json.dumps(key,separators=(',',':')).encode()).digest()
            value=float(np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest,dtype='<u4').tolist())).uniform(.9,1.))
            cols=[j for j,r in enumerate(config['resources']) if r['physical_device_id']==device['id']]
            eta[cols]=value;lineage.append((device['id'],cols,value,key,digest.hex()))
        self.same('whole_device_response_stream',random['eta'],eta,atol=0,rtol=0)
        saved=base.read(self.file(folder/f'execution{repeat:02d}'/'randomness_lineage.json'))
        self.references(saved);self.require(saved['evaluation_noise']==entry['evaluation_noise'],'Evaluation noise source changed')
        self.require(len(saved['devices'])==len(lineage),'Response identity count differs')
        for row,(device,cols,value,key,digest) in zip(saved['devices'],lineage,strict=True):
            self.require(row['physical_device_id']==device and row['columns']==cols,'Device response identity differs')
            self.same('response_lineage_value',row['value'],value,atol=0,rtol=0)
            self.require(row['keyed_stream']['key']==key and row['keyed_stream']['sha256']==digest,'Response key differs')
        return random

    def case(self,folder,entry,p,metadata,branches,buses):
        self.where=str(folder);self.references(entry)
        seal=base.read(self.file(folder/'seal.json'));self.references(seal)
        for key,expected in [('score',entry['score']),('train',entry['files']['train.npz']),('validation',entry['files']['validation.npz'])]:
            self.require(seal[key]==expected,'Frozen historical input changed: '+key)
        if entry['id'] not in self.validated_data:
            clean,train,validation,timeline=self.data(entry,p)
            ctx=self.context(entry);self.fits(entry,ctx,train,validation)
            self.validated_data[entry['id']]=(clean,train,validation,timeline,ctx)
        clean,train,validation,timeline,ctx=self.validated_data[entry['id']]
        prior=folder
        planner=base.native.arrays(self.checked(seal['planner']));forecast=planner['forecast']
        locked_forecast=base.native.arrays(self.checked(entry['forecast_ref']))[entry.get('forecast_key','forecast00')]
        self.same('original_forecast',forecast,locked_forecast,atol=0,rtol=0)
        quotes=single.policy_names(p);policies=['no_dfs',*quotes]
        self.require(set(planner)==set(policies+['forecast']),'Action list differs')
        config=base.read(self.checked(seal['resource_case']));spec=base.read(self.checked(seal['equipment_spec']))
        self.require(spec==base.read(self.checked(entry['equipment_spec'])),'Original device specification changed')
        self.equipment(entry,config,spec,forecast);bands=self.band_arrays(entry,seal)
        noise=panel.noise_parameters(entry,p)
        model=single.historical_target(prob.build_model(ctx['score'],validation,ctx['native'],
            meter_relative_sd=noise['meter_relative_sd'],forecast_relative_sd=noise['forecast_relative_sd']))
        self.input_records.append(self.coverage(entry,ctx,model))
        self.input_records[-1].update(history_p_map=ctx['score']['p_map'],score_sigma2=ctx['sigma2'],
            training_band_kvar=dict(minimum=float(bands['band_kvar'].min()),median=float(np.median(bands['band_kvar'])),maximum=float(bands['band_kvar'].max())),
            training_band_points_per_hour=8)
        seeds=base.native.arrays(self.checked(seal['shared_initial_points']))
        self.require(set(seeds)=={'zero'},'Unexpected nonzero or previous optimized seed')
        self.same('only_zero_seed',seeds['zero'],np.zeros_like(planner['no_dfs']),atol=0,rtol=0)
        self.same('zero_policy',planner['no_dfs'],seeds['zero'],atol=0,rtol=0)
        plans=base.read(self.checked(seal['plans']));self.require(set(plans)==set(quotes),'Missing quotes')
        saved_audits=base.read(self.checked(seal['prediction_audits']));kp=p['reservation_prices']['kappaP'];kq=p['reservation_prices']['kappaQ']
        forecast_values={};action_values={};seed_cache={}
        def seed_check(name,mode):
            if (name,mode) not in seed_cache:
                dv,bd,shift=self.action(entry,config,bands,seeds[name]);gate=self.gates(model,forecast,forecast+shift,mode)
                seed_cache[name,mode]=(dict(gate,feasible=gate['feasible'] and dv['passed'] and bd['passed']),equipment.fees(config,seeds[name],seeds[name],kp,kq)['planned']['activation_yuan'])
            return seed_cache[name,mode]
        for name in quotes:
            u=planner[name];q=plans[name];mode=name.removeprefix('single_')
            self.same('frozen_mode_action',base.native.arrays(self.file(folder/(name+'_action.npz')))['action'],u,atol=0,rtol=0)
            self.require(base.read(self.file(folder/(name+'_plan.json')))==q,'Mode receipt changed after sealing')
            dv,bd,shift=self.action(entry,config,bands,u);self.require(dv['passed'] and bd['passed'],'Executed request violates original physical/band domain')
            bill=equipment.fees(config,u,u,kp,kq)['planned'];direct=self.gates(model,forecast,forecast+shift,mode,saved_audits[mode][name])
            self.require(q['initial_points']==seal['shared_initial_points'],'Quote used a different seed portfolio')
            super().single_quote(entry,name,q,u,direct,bill,seeds,seed_check)
            quote_row=self.quote_records[-1];certificate=quote_row['certificate']
            if q.get('resource_box_checks'):
                box=equipment.independent_box_checks(model,forecast,config,entry['base_kw'],mode)
                self.require(len(box)==len(q['resource_box_checks']),'Missing box certificate rows')
                for actual,saved in zip(box,q['resource_box_checks'],strict=True):
                    self.same('independent_probability_box_upper',saved['upper'],actual['upper'],atol=1e-7)
                    self.require(actual['impossible']==saved['impossible'],'Box infeasibility flag differs')
                if q['status']=='infeasible_by_probability_box_bound':self.require(any(r['impossible'] for r in box),'No independent negative box certificate')
                quote_row['independent_resource_box_checks']=box
            encoded=bd['encoded_passed'] and all(r['margin']>=1e-8-1e-9 for r in direct['rows']) and all(g['gain']>=-1e-9 for g in direct['guards'])
            certificate.update(encoded_domain_feasible=encoded,original_domain_feasible=dv['passed'] and bd['passed'] and direct['feasible'],
                bound_scope='New device domain plus encoded original node/hour band and single-target gates; activation fee only.',
                lower_bound_valid_for_old_free_Q_domain=False)
            if not encoded and certificate.get('lower_bound_yuan') not in (None,0):
                certificate.update(raw_encoded_lower_bound_yuan=certificate['lower_bound_yuan'],lower_bound_yuan=0.,
                    relative_gap=1. if bill['activation_yuan']>0 else 0.,lower_bound_withdrawn=True)
            quote_row.update(planning_band=bd,requested_devices=dv,shared_preparation_seconds=q.get('shared_preparation_seconds'),
                total_method_seconds_with_shared_preparation=q.get('total_method_seconds_with_shared_preparation'),
                zero_direct_gate=self.gates(model,forecast,forecast,mode),
                zero_normal_direct_gate=self.gates(model,forecast,forecast,'normal_fixed'),
                historical_target_internal_correct=self.input_records[-1]['history_correct'])
            forecast_values[name]=self.evaluate(ctx,forecast+shift);action_values[name]=(dv,bd,bill)
        zero_dv,zero_band,_=self.action(entry,config,bands,planner['no_dfs'])
        forecast_values['no_dfs']=self.evaluate(ctx,forecast)
        action_values['no_dfs']=(zero_dv,zero_band,equipment.fees(config,planner['no_dfs'],planner['no_dfs'],kp,kq)['planned'])
        done=base.read(self.file(folder/'complete.json'));rows=base.read(self.checked(done['results']))
        expected={(r,n) for r in range(p['repeats']) for n in policies}
        self.require(len(rows)==len(expected) and {(r['repeat'],r['strategy']) for r in rows}==expected,'Execution denominator differs')
        zero_forecast=self.evaluate(ctx,forecast);zeros={}
        for row in rows:
            repeat,name=row['repeat'],row['strategy'];path=folder/f'execution{repeat:02d}'/name
            record=dict(**panel.identity(entry),forecast=0,repeat=repeat,strategy=name,source=str(path),original_status=row['status'],history_repeat=entry['history_repeat'],pq_relative_sd=entry['pq_relative_sd'])
            try:
                self.require(base.read(self.file(path/'receipt.json'))==row,'Execution ledger differs')
                if row['status']!='complete':
                    record.update(audit_status='recorded_failure',error=row.get('error'),evaluation={'native_correct':False});self.records.append(record);continue
                self.references(row);q=plans.get(name,dict(zero_fallback=False,new_gate_feasible=False,mode='no_dfs'))
                if name!='no_dfs':self.require(row['quote_zero_fallback']==q['zero_fallback'] and row['quote_new_constraints_feasible']==q['new_gate_feasible'],'Executed quote flags differ')
                random=self.randomness(folder,entry,p,config,repeat)
                obs=base.native.arrays(self.file(path/'observations.npz'));u=planner[name];delivered=u*random['eta'][None]
                self.same('sealed_request',obs['requested_action'],u,atol=0,rtol=0)
                self.same('shared_device_delivery',obs['delivered_action'],delivered,atol=0,rtol=0)
                dv,bd,shift=self.action(entry,config,bands,delivered);self.require(dv['passed'],'Delivered device limits fail')
                requested=action_values[name][0]
                state_json=base.read(self.checked(row['device_states']['json']))
                self.tree(state_json['requested'],requested['states'],'requested_states')
                self.tree(state_json['delivered'],dv['states'],'delivered_states')
                state_arrays=base.native.arrays(self.checked(row['device_states']['arrays']))
                for key,value in state_arrays.items():
                    reference=state_json
                    for part in key.split('__'):reference=reference[part]
                    self.same('state_archive_'+key,value,reference,atol=0,rtol=0)
                self.require(row['delivery']['device_availability_clipping'] is False and row['delivery']['pv_clipped_kwh']==0,'Unannounced availability clipping')
                self.same('true_net_P',obs['true_P'],ctx['baseline_P']+shift[:,:ctx['n']],atol=1e-13)
                self.same('true_net_Q',obs['true_Q'],ctx['baseline_Q']+shift[:,ctx['n']:],atol=1e-13)
                for label,k,sd in [('P',0,noise['pq_relative_sd']),('Q',1,noise['pq_relative_sd']),('V',2,noise['meter_relative_sd']),('root',3,noise['meter_relative_sd'])]:
                    self.same('meter_'+label,obs[label],obs['true_'+label]*(1+sd*random['noise'+str(k)]),atol=1e-13)
                self.same('squared_voltage_Y',obs['Y'],obs['root'][:,None]**2-obs['V']**2,atol=1e-13)
                bill=equipment.fees(config,u,delivered,kp,kq)
                for when in ('planned','settled'):
                    self.costs(row[when+'_cost'],bill[when],when)
                    self.same(when+'_fee',row[when+'_fee_yuan'],bill[when]['total_yuan'],atol=1e-8)
                for field,value in [('requested_quantities',u),('delivered_quantities',delivered)]:
                    for key,expected_quantity in equipment.quantity(value,config).items():self.same(field+'_'+key,row[field][key],expected_quantity,atol=1e-9)
                evaluation,values=self.score(ctx,obs,row);evaluation['full_native_diagnostic']=previous.full_native_diagnostic(ctx,values)
                self.pair_file(self.file(path/'pair_evidence.npz'),ctx,values)
                if repeat not in zeros:
                    zpath=prior/f'execution{repeat:02d}'/'no_dfs';zo=base.native.arrays(self.file(zpath/'observations.npz'))
                    zr=base.read(self.file(zpath/'receipt.json'));ze,zv=self.score(ctx,zo,zr)
                    zeros[repeat]=(zo,ze,zv,zr)
                zero_obs,zero_evaluation,zero_values,zero_row=zeros[repeat]
                if not np.count_nonzero(delivered):
                    for key in ('P','Q','Y','V','root','true_P','true_Q','true_V','true_root'):
                        self.same('zero_preserves_original_'+key,obs[key],zero_obs[key],atol=0,rtol=0)
                self.reported_evidence(ctx,row['evidence'],values,zero_values,forecast_values[name],zero_forecast)
                changes=base.native.arrays(self.file(path/'pair_changes.npz'))
                for key in ('J','J_without_common','J_common','J_cross','bias','meter','observed_logodds','observed_logodds_without_common'):
                    expected_changes=[self.directed(ctx,values,i,j)[key]-self.directed(ctx,zero_values,i,j)[key] for i,j in combinations(range(len(ctx['H'])),2)]
                    self.same('pair_changes_'+key,changes[key],expected_changes,atol=1e-8)
                delta=np.c_[obs['true_P']-zero_obs['true_P'],obs['true_Q']-zero_obs['true_Q']]
                dy=obs['true_root'][:,None]**2-obs['true_V']**2-zero_obs['true_root'][:,None]**2+zero_obs['true_V']**2
                errorK=np.square(dy[None]-np.einsum('tc,knc->ktn',delta,model['means'])).mean((1,2))/model['sigma2']
                errorH=np.square(dy[None]-np.einsum('tc,knc->ktn',delta,model['H'][model['representatives']])).mean((1,2))/model['sigma2']
                for key,value in [('actual_increment_response_error_K_over_sigma2_by_representative',errorK),('actual_increment_response_error_H_over_sigma2_by_representative',errorH)]:self.same(key,row[key],value,atol=1e-9)
                ac=base.native.arrays(self.checked(row['physical_state']))
                if 'time_slice' in row['physical_state']:
                    start,end=row['physical_state']['time_slice'];length=len(ac['voltage_complex_pu'])
                    ac={k:v[start:end] if v.ndim and len(v)==length else v for k,v in ac.items()}
                electrical,branch_values,bus_values=base.operation.electrical(self.operation,ac,obs,metadata,branches,buses)
                cost_energy=equipment.energy_bill(ac['root_supply_kw']);self.same('root_energy_cost',row['root_energy_value_yuan'],cost_energy,atol=1e-8)
                zero_ac=base.native.arrays(self.checked(zero_row['physical_state']))
                if 'time_slice' in zero_row['physical_state']:
                    start,end=zero_row['physical_state']['time_slice'];length=len(zero_ac['voltage_complex_pu'])
                    zero_ac={k:v[start:end] if v.ndim and len(v)==length else v for k,v in zero_ac.items()}
                zero_energy=equipment.energy_bill(zero_ac['root_supply_kw'])
                self.same('baseline_root_energy_cost',row['baseline_root_energy_value_yuan'],zero_energy,atol=1e-8)
                self.same('root_energy_cost_change',row['root_energy_value_change_yuan'],cost_energy-zero_energy,atol=1e-8)
                voltage_ok=electrical['cells_below_095']+electrical['cells_above_105']==0
                devP=np.zeros_like(ctx['baseline_P']);devQ=np.zeros_like(ctx['baseline_Q'])
                for device in config['devices']:
                    node=ctx['labels'].index(device['node_label'])
                    devP[:,node]+=device['baseline_p_kw'];devQ[:,node]+=device['baseline_q_kvar']
                residualP=ctx['baseline_P']*entry['base_kw']-devP
                record.update(audit_status='passed',requested_devices=requested,delivered_devices=dv,
                    planning_band=action_values[name][1],delivered_band=bd,planned_cost=bill['planned'],settled_cost=bill['settled'],
                    requested_violations={},delivered_violations={},evaluation=evaluation,electrical=electrical,
                    branches=branch_values,buses=bus_values,voltage_screen_passed=voltage_ok,
                    baseline_root_energy_value_yuan=zero_energy,root_energy_value_yuan=cost_energy,root_energy_value_change_yuan=cost_energy-zero_energy,
                    service_plus_root_energy_change_yuan=bill['settled']['total_yuan']+cost_energy-zero_energy,
                    actual_net_baseline_decomposition=dict(negative_background_P_cells=int((residualP<0).sum()),
                        newly_negative_P_cells=int(((residualP<0)&(ctx['baseline_P']>=0)).sum()),
                        minimum_background_P_kw=float(residualP.min()),scope='Actual net profile minus declared device baseline, post-execution only; no baseline is added to the net.'),
                    recovered_and_voltage_screen_passed=bool(evaluation['native_correct'] and voltage_ok),
                    quote_new_gate_feasible=q['new_gate_feasible'],quote_zero_fallback=q['zero_fallback'],base_mode=q['mode'],is_rotation=False,
                    actual_H_increment_error_over_sigma2=errorH.tolist(),actual_K_increment_error_over_sigma2=errorK.tolist(),
                    thermal_rating_status='unavailable_not_verified',reused_action_source=row.get('reused_action_source'),baseline_source=row['baseline_source'])
            except Exception as exc:record.update(audit_status='failed',error=repr(exc),traceback=traceback.format_exc())
            self.records.append(record)


def snapshot_sources(audit,root,protocol):
    for item in protocol['sources']:
        relative=Path(item['path']).resolve().relative_to(ROOT);saved=Path(root)/'source_snapshot'/relative
        audit.file(saved,item['sha256']);audit.source_snapshots[(str(Path(item['path']).resolve()),item['sha256'])]=saved
        current=base.ref(item['path'])
        if current['sha256']!=item['sha256']:
            audit.post_execution_source_changes.append(dict(executed_source=item,snapshot=base.ref(saved),current_source=current))


def information_boundary(path,data_path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    calls=lambda f:[n for n in ast.walk(f) if isinstance(n,ast.Call)]
    work=calls(functions['run_case']);solve=[n for n in work if ast.unparse(n.func)=='engine.plan']
    seals=[n for n in work if ast.unparse(n.func)=='io.write' and n.args and ast.unparse(n.args[0])=='seal_path']
    executions=[n for n in work if ast.unparse(n.func)=='execute']
    ex=calls(functions['execute']);truth=[n for n in ex if ast.unparse(n.func)=='old.context']
    checks=[n for n in ex if ast.unparse(n.func)=='paired.check_refs']
    forbidden=[n for n in work if ast.unparse(n.func) in ('old.context','physical_response','physical.offline_truth')]
    plan_source=ast.get_source_segment(Path(path).read_text(encoding='utf-8-sig'),functions['context_planning'])
    expected=['case',"planning['score']","planning['val']",'forecast',"planning['base']"]
    planning=(len(solve)==len(seals)==len(executions)==1 and not forbidden and
        solve[0].lineno<seals[0].lineno<executions[0].lineno and [ast.unparse(x) for x in solve[0].args]==expected
        and len(truth)==1 and any(n.lineno<truth[0].lineno for n in checks)
        and all(key not in plan_source for key in ('physical_cache','new_day','evaluation_noise')))
    dt=ast.parse(Path(data_path).read_text(encoding='utf-8-sig'))
    fit=next(n for n in dt.body if isinstance(n,ast.FunctionDef) and n.name=='fit_cell');fc=calls(fit)
    ident=[n for n in fc if ast.unparse(n.func)=='fit_observed'];offline=[n for n in fc if ast.unparse(n.func)=='physical.offline_truth']
    candidate_seal=[n for n in fc if ast.unparse(n.func)=='io.write' and n.args and 'candidates_sealed.json' in ast.unparse(n.args[0])]
    fit_ok=(len(ident)==len(offline)==len(candidate_seal)==1 and
        [ast.unparse(a) for a in ident[0].args]==["splits['train']","splits['validation']"] and
        ident[0].lineno<candidate_seal[0].lineno<offline[0].lineno)
    return dict(passed=planning and fit_ok,planning=planning,fitting=fit_ok,
        source=base.ref(path),data_source=base.ref(data_path),scope='Historical-only fit and quote arguments; candidate/action seals precede offline truth and actual AC observations.')


def audit_data_manifest(audit,index):
    manifest=base.read(audit.file(index));protocol=base.read(audit.checked(manifest['protocol']))
    data_root=Path(manifest['protocol']['path']).parent;snapshot_sources(audit,data_root,protocol)
    audit.references(protocol.get('input_refs',[]));audit.references(protocol.get('previous_source_snapshots',[]))
    audit.require(manifest['source_unchanged'] and manifest['planned_inputs']==40,'Incomplete/changed data manifest')
    tasks={t['id']:t for t in protocol['tasks']};attempts=manifest['attempts'];entries=manifest['inputs']
    audit.require(len(tasks)==len(attempts)==40 and {r['task']['id'] for r in attempts}==set(tasks),'Missing fit attempts')
    grid={(t['case'],t['history_repeat'],t['pq_relative_sd']) for t in tasks.values()}
    audit.require(len(grid)==40 and {t[1] for t in grid}==set(range(4)) and {t[2] for t in grid}=={0.,.0025,.005,.01,.02},'Locked data grid differs')
    audit.require(protocol['method_budget_seconds']==120. and protocol['additional_E_pool_search_seconds']==60.,'Fit budgets changed')
    successful={r['task']['id'] for r in attempts if r['status']=='complete'}
    audit.require(successful=={e['id'] for e in entries},'Successful fit ledger differs')
    for entry in entries:
        audit.where='data/'+entry['id']
        try:
            audit.references(entry)
            clean,train,validation,timeline=audit.data(entry,protocol)
            ctx=audit.context(entry);audit.fits(entry,ctx,train,validation)
            audit.validated_data[entry['id']]=(clean,train,validation,timeline,ctx)
        except Exception as exc:audit.task_errors.append(dict(id=entry['id'],stage='data_audit',error=repr(exc),traceback=traceback.format_exc()))
    return manifest,protocol,tasks


def write_jsonlines(out,name,rows):
    with (out/(name+'.jsonl')).open('w',encoding='utf8') as stream:
        for row in rows:panel.json_line(stream,row)
    return base.ref(out/(name+'.jsonl'))


def finish_hashes(audit):
    for path,sha in audit.hashes.items():
        if base.ref(path)['sha256']!=sha:audit.task_errors.append(dict(path=path,error='Source changed during audit'))


def run_data(index,out):
    index,out=Path(index).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'audit.py');audit=Audit()
    manifest,protocol,tasks=audit_data_manifest(audit,index);finish_hashes(audit)
    result=dict(status='passed' if not audit.task_errors else 'failed',source=base.ref(__file__),source_index=base.ref(index),
        planned_fits=40,audited_fits=len(audit.fit_records),original_failed_fits=manifest['failures'],
        task_errors=audit.task_errors,observations=audit.data_records,fits=audit.fit_records,
        maximum_numeric_errors=dict(audit.maxima),post_execution_source_changes=audit.post_execution_source_changes,
        input_refs=[dict(path=k,sha256=v) for k,v in audit.hashes.items()])
    base.write(out/'report.json',result);base.write(out/'complete.json',dict(status=result['status'],report=base.ref(out/'report.json')))
    return result


def run(source,out):
    source,out=Path(source).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    if not (source/'complete.json').exists():raise ValueError('Completed run required')
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'audit.py');audit=Audit()
    p=base.read(audit.file(source/'protocol.json'));complete=base.read(audit.file(source/'complete.json'))
    snapshot_sources(audit,source,p)
    audit.require(complete['source_unchanged'] and p['oracle'] is False,'Changed runtime or unexpected oracle')
    manifest,dp,tasks=audit_data_manifest(audit,audit.checked(p['input_index']))
    data_path=Path(manifest['protocol']['path']).parent/'source_snapshot'/Path(__file__).with_name('data.py').relative_to(ROOT)
    boundary=information_boundary(source/'source_snapshot'/p['driver_relative_path'],data_path)
    audit.require(boundary['passed'],'Information boundary differs')
    entries=base.read(audit.checked(p['inputs']));available={e['id']:e for e in manifest['inputs']}
    for entry in entries:
        audit.require(entry['id'] in available,'Unplanned fitted input')
        for key,value in available[entry['id']].items():audit.require(entry.get(key)==value,'Fitted input changed: '+key)
    expected_tasks={k:t for k,t in tasks.items() if not p['smoke'] or (t['history_repeat']==0 and t['pq_relative_sd']==.005)}
    names=['no_dfs',*single.policy_names(p)]
    expected=len(expected_tasks)*len(names)*p['repeats']
    audit.require(expected==p['expected_rows'] and names==p['policies'],'Run denominator differs')
    preparation=base.read(audit.checked(p['preparation_failures']))
    failures={r['id']:r for r in complete.get('failed_tasks',[])}
    success={r['id'] for r in complete['receipts']}
    audit.require(not(success&set(failures)) and success|set(failures)=={e['id'] for e in entries},'Runtime task ledger differs')
    unavailable=set(expected_tasks)-{e['id'] for e in entries}
    data_failed={r['task']['id'] for r in manifest['failures']}
    prep_failed={r['id'] for r in preparation['planning_failures']}
    audit.require(unavailable==(data_failed|prep_failed)&set(expected_tasks),'Unavailable inputs lack an explicit failure receipt')
    audit.require(p['unavailable_policy_rows']==len(unavailable)*len(names)*p['repeats'],'Unavailable row denominator differs')
    own=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=base.ref(__file__),source_protocol=base.ref(source/'protocol.json'),
        expected_rows=expected,planning_boundary=boundary,environment=dict(python=sys.version,numpy=np.__version__),
        scope='Independent saved-array noise, fit/scoring, device, gate, cost and full AC checks; no new fit/optimization/AC',
        helpers=[base.ref(m.__file__) for m in (equipment,single,equipment.band,panel,previous,base,base.native,base.operation,prob)])
    base.write(out/'protocol.json',own)
    mapping={e['id']:e for e in entries}
    for tag,task in expected_tasks.items():
        if tag not in mapping or tag in failures:
            for r in range(p['repeats']):
                for name in names:audit.records.append(dict(**panel.identity(task),history_repeat=task['history_repeat'],pq_relative_sd=task['pq_relative_sd'],
                    forecast=0,repeat=r,strategy=name,audit_status='recorded_failure',original_status='unavailable_input' if tag not in mapping else 'task_failed',evaluation={'native_correct':False}))
            continue
        entry=mapping[tag]
        try:
            casepath=source/'source_snapshot/topology_identification_standalone/cases'/entry['case']
            metadata=base.read(audit.file(casepath/'metadata.json'))
            with audit.file(casepath/'branches.csv').open(encoding='utf-8-sig',newline='') as stream:branches=list(csv.DictReader(stream))
            with audit.file(casepath/'buses.csv').open(encoding='utf-8-sig',newline='') as stream:buses=list(csv.DictReader(stream))
            audit.case(source/'cases'/tag,entry,p,metadata,branches,buses)
        except Exception as exc:audit.task_errors.append(dict(id=tag,stage='runtime_audit',error=repr(exc),traceback=traceback.format_exc()))
        base.write(out/'progress.json',dict(rows=len(audit.records),expected=expected,task_errors=len(audit.task_errors),
            row_errors=sum(r['audit_status']=='failed' for r in audit.records)))
    finish_hashes(audit)
    passed=len(audit.records)==expected and not audit.task_errors and all(r['audit_status'] in ('passed','recorded_failure') for r in audit.records)
    collections={'records':audit.records,'quotes':audit.quote_records}
    refs={key:write_jsonlines(out,key,value) for key,value in collections.items()}
    result=dict(status='passed' if passed else 'failed',protocol=base.ref(out/'protocol.json'),source_protocol=base.ref(source/'protocol.json'),source_inputs=p['inputs'],
        source_data_index=p['input_index'],expected_rows=expected,audited_rows=len(audit.records),expected_quotes=p['expected_quotes'],audited_quotes=len(audit.quote_records),
        expected_DR_rows=p['expected_DR_rows'],expected_zero_rows=p['expected_zero_rows'],
        audit_status_counts=dict(Counter(r['audit_status'] for r in audit.records)),input_coverage=audit.input_records,data_records=audit.data_records,
        fit_records=audit.fit_records,device_checks=audit.device_checks,task_errors=audit.task_errors,original_failed_fits=manifest['failures'],
        original_preparation_failures=preparation,original_failed_tasks=complete.get('failed_tasks',[]),original_quote_exceptions=complete.get('quote_exceptions',0),
        original_execution_failed_rows=complete.get('failed_rows',0),post_execution_source_changes=audit.post_execution_source_changes,
        maximum_numeric_errors=dict(audit.maxima),maximum_electrical_errors=dict(audit.operation.maxima),counts=dict(audit.counts),
        input_refs=[dict(path=k,sha256=v) for k,v in audit.hashes.items()],**refs)
    base.write(out/'report.json',result);base.write(out/'complete.json',dict(status=result['status'],report=base.ref(out/'report.json')))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--run',type=Path);group.add_argument('--data-index',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();result=run(args.run,args.out) if args.run else run_data(args.data_index,args.out)
    print({k:result[k] for k in ('status','expected_rows','audited_rows','audited_fits','task_errors') if k in result})
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
