"""Independent saved-data replay of the fixed-waveform location ablation.

No action selection, new fit, or experiment generator is called. The separate
branch-current sweep and fit arithmetic live in the outside-freeze reviewer.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from .ev_original_review import (arr,close,require,source_audit,forward_physics,
    replay_fits,replay_operation_record,service_config,identity)
from .information_fit_recovery import read_json,write_json
from .information_replay import _obs,_meter
from .likelihood import RadialCandidate,concat_observations

HERE=Path(__file__).resolve().parent
METHODS=('charge_mobile','v2g_mobile')
KINDS=('noisy','noiseless')
CHANNELS=('p','q','v','master_p','master_q','source_p','source_q')


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curve_hash(power):
    a=np.asarray(power,dtype='<f8').copy();a[a==0]=0.
    return hashlib.sha256(a.tobytes()).hexdigest()


def signed_margin(costs,true):
    wrong=min((k for k in costs if k!=true),key=lambda k:(costs[k],k))
    return costs[wrong]-costs[true],wrong


def fit_channels(saved,obs,fm,H,hfits):
    computed=[]
    for fit in saved['fits']:
        channels={}
        for k in CHANNELS:
            observed=getattr(obs,k)
            if observed is None:continue
            prediction=fit.get('predicted',{}).get(k)
            if prediction is None and k in ('source_p','source_q'):prediction=fit.get(k)
            require(prediction is not None,'missing saved prediction:'+k)
            residual=(arr(prediction)[H:]-observed)/np.asarray(getattr(fm,k+'_sigma'))
            channels[k]=float(np.sum(residual**2))
        Hcost=float(fit['loss'])-sum(channels.values())
        computed.append(dict(candidate=fit['name'],joint_SSE=float(fit['loss']),historical_SSE=Hcost,
            historical_penalty=Hcost-float(hfits[fit['name']]['loss']),future_channel_SSE=channels))
    require({x['candidate'] for x in computed}=={x['candidate'] for x in saved['candidate_costs']},'channel candidate coverage')
    old={x['candidate']:x for x in saved['candidate_costs']}
    for row in computed:
        for k in ('joint_SSE','historical_SSE','historical_penalty'):close(old[row['candidate']][k],row[k],'saved candidate cost:'+k)
        require(set(old[row['candidate']]['future_channel_SSE'])==set(row['future_channel_SSE']),'saved channel coverage')
        for k,v in row['future_channel_SSE'].items():close(old[row['candidate']]['future_channel_SSE'][k],v,'saved future channel:'+k)
    return computed


def replay_history(saved,parent,expected_hash,*,methods=None,service_config_builder=None):
    """Replay saved states; optional adapters only change method/config labels.

    Defaults preserve the original stage-16 review. Compact callers retain
    their two method names and pass the corresponding signed mobile domain.
    """
    methods=METHODS if methods is None else tuple(methods)
    service_config_builder=service_config if service_config_builder is None else service_config_builder
    seed=parent['seed'];H=parent['config']['history'];cfg=parent['config'];truth=parent['truth']
    require(saved['seed']==seed and saved['parent_trial_sha256']==expected_hash,'parent identity')
    require(saved['new_selector_runs']==0,'unexpected selector call')
    candidates={r['name']:RadialCandidate.from_supports(r['name'],3,r['supports']) for r in parent['proposal']['bank']}
    true_names=[n for n,c in candidates.items() if tuple(s for s in c.supports if 1<len(s)<3)==tuple(map(tuple,truth['family']))]
    require(len(true_names)==1,'true candidate mapping');true=true_names[0]
    hm,fm,jm=(_meter(parent['measurement_models'][k]) for k in ('hm','fm','jm'))
    history=_obs(parent['history']);hfits={f['name']:f for f in parent['historical_fits']}
    hist=replay_fits(parent['historical_fits'],candidates,history,hm,parent['historical_failures'],complete=True,config=cfg,domain=parent['domain'])
    shared={k:arr(truth[k])[H:].tolist() for k in ('p','q','source_p','source_q','root_v')}
    shared.update(r=truth['r'],x=truth['x'],family=truth['family'],future_white={k:arr(v)[H:].tolist() for k,v in truth['white'].items()})
    require(saved['shared_future_inputs']==shared and identity(shared)==saved['shared_future_input_sha256'],'shared future input digest')
    require([r['method'] for r in saved['conditions']]==list(methods),'method denominator/order')
    parentrows={r['method']:r for r in parent['rows']}
    cache={curve_hash(parentrows[m]['power_kw']):parentrows[m] for m in methods}
    counts=Counter();conditions=[];candidate_rows=[];saved_fit_states=hist['replayed_states'];maxnoise=0.
    for condition in saved['conditions']:
        method=condition['method'];original=condition['original'];remap=condition['remap'];row=parentrows[method]
        require(condition['status']=='complete','preserved incomplete condition')
        u=arr(original['power_kw']);v=arr(remap['power_kw']);net=u.sum(axis=1);dt=cfg['dt_hours']
        close(u,row['power_kw'],'original frozen action',atol=0,rtol=0)
        close(v[:,0],net,'remap total waveform',atol=0,rtol=0);close(v[:,1:],np.zeros_like(v[:,1:]),'remap destination',atol=0,rtol=0)
        close(condition['waveform_max_abs_difference_kw'],np.max(abs(net-v.sum(axis=1))),'waveform diagnostic')
        close(condition['grid_energy_difference_kwh'],dt*(u.sum()-v.sum()),'grid energy diagnostic')
        close(condition['grid_throughput_difference_kwh'],dt*(abs(u).sum()-abs(v).sum()),'grid throughput diagnostic')
        close(dt*u.sum(),dt*v.sum(),'same net grid energy');close(dt*abs(u).sum(),dt*abs(v).sum(),'same grid throughput')
        resource=condition['resource_scope'];op=resource['original_executed_operation']
        op_audit=replay_operation_record(op,service_config_builder(cfg,method));close(op['u_kw'],u,'executed operation versus evaluation')
        terminal=cfg['device']['initial_energy_kwh']+dt*np.sum(cfg['device']['eta_charge']*np.maximum(net,0)-np.maximum(-net,0)/cfg['device']['eta_discharge'])
        close(resource['grid_net_kwh'],dt*u.sum(),'resource grid energy');close(resource['grid_throughput_kwh'],dt*abs(u).sum(),'resource throughput')
        close(resource['remap_hypothetical_terminal_without_driving_kwh'],terminal,'remap battery accounting')
        close(resource['remap_terminal_excess_over_original_target_kwh'],terminal-cfg['device']['terminal_energy_kwh'],'remap battery service difference')
        close(terminal-cfg['device']['terminal_energy_kwh'],op['travel_energy_kwh'],'driving energy retained in scope')
        require(resource['remap_has_no_operation_feasibility_or_equal_service_claim'],'service scope')
        variant_results={}
        for label,variant,power in (('original',original,u),('remap',remap,v)):
            key=curve_hash(power);require(variant['curve_sha256']==key,'curve byte identity')
            phy=forward_physics(RadialCandidate.from_supports('independent_truth',3,truth['family']),truth['r'],truth['x'],
                arr(truth['p'])[H:]+power/cfg['base_kw'],arr(truth['q'])[H:],arr(truth['source_p'])[H:],arr(truth['source_q'])[H:],arr(truth['root_v'])[H:])
            for k in ('p','q','v','master_p','master_q'):close(variant['physical'][k],phy[k],'true AC, EV added once:'+k,atol=3e-10,rtol=1e-10)
            require(phy['voltage_min']>=.9 and phy['voltage_max']<=1.1,'physical evaluation range')
            new=(label=='remap' and key not in cache)
            require(variant['new_forward_AC_calls']==int(new),'forward cache count')
            counts['forward_AC_calls']+=int(new)
            kind_results={}
            for kind in KINDS:
                detail=variant[kind];obs=_obs(detail['observations']);require(detail['success'] and not detail['failures'],'fit completeness')
                require(detail['cache_hit']==(not new),'cache flag')
                if not new:
                    expected=cache[key]
                    if 'noisy' in expected:old=expected[kind]
                    else:old=expected if kind=='noisy' else expected['noiseless_AC_refit']
                    require(detail['fits']==old['fits'],'copied fit state identity')
                    require(detail['new_fit_calls']==0 and detail['new_fit_AC_calls']==0,'cache has new fit calls')
                else:
                    require(detail['new_fit_calls']==4 and detail['new_fit_calls']==detail['cost']['fit_calls'],'new candidate fit count')
                    require(detail['new_fit_AC_calls']==detail['cost']['ac_calls'],'new fit AC count')
                counts['candidate_fit_calls']+=detail['new_fit_calls'];counts['fit_AC_calls']+=detail['new_fit_AC_calls']
                channel_white={}
                for k in CHANNELS:
                    observed=getattr(obs,k)
                    if observed is None:continue
                    bias=getattr(fm,'bias_'+k) if k in ('p','q','v') else 0.
                    w=(observed-phy[k]-bias)/np.asarray(getattr(fm,k+'_sigma'))
                    wanted=arr(truth['white'][k])[H:] if kind=='noisy' else np.zeros_like(w)
                    error=float(np.max(abs(w-wanted)));maxnoise=max(maxnoise,error);channel_white[k]=error
                    require(error<=1e-8,'paired whitened observation error:'+k)
                audit=replay_fits(detail['fits'],candidates,concat_observations(history,obs),jm,detail['failures'],complete=True,config=cfg,domain=parent['domain'])
                saved_fit_states+=audit['replayed_states'];costs=fit_channels(detail,obs,fm,H,hfits)
                losses={f['name']:float(f['loss']) for f in detail['fits']};margin,nearest=signed_margin(losses,true)
                close(detail['M'],margin,'signed fit margin');close(detail['half_M'],.5*margin,'half margin');require(detail['nearest_wrong']==nearest,'nearest wrong')
                require(detail['losses']==losses and not detail['candidate_miss'],'candidate loss coverage')
                for c in costs:candidate_rows.append(dict(seed=seed,family=truth['family'],method=method,variant=label,kind=kind,true_candidate=true,nearest_wrong=nearest,**c))
                kind_results[kind]=dict(M=margin,nearest_wrong=nearest,white_error=channel_white,fit_audit=audit,channel_costs=costs)
            variant_results[label]=kind_results
            if label=='remap':cache.setdefault(key,variant)
        for kind,field in (('noisy','delta_M'),('noiseless','delta_noiseless_M')):
            close(condition[field],variant_results['original'][kind]['M']-variant_results['remap'][kind]['M'],'paired margin delta')
        conditions.append(dict(method=method,status=condition['status'],new_remap=bool(remap['new_forward_AC_calls']),
            grid_energy_kwh=float(dt*u.sum()),grid_throughput_kwh=float(dt*abs(u).sum()),
            remap_terminal_extra_kwh=float(terminal-cfg['device']['terminal_energy_kwh']),original_operation=op_audit,
            delta_M=condition['delta_M'],delta_noiseless_M=condition['delta_noiseless_M'],variants=variant_results))
    require(dict(counts)==saved['counts'],'per history new calls')
    return dict(seed=seed,family=truth['family'],success=True,independent_histories=1,conditions=conditions,counts=dict(counts),
        saved_fit_states_replayed=saved_fit_states,physical_states_replayed=4,max_abs_white_error=maxnoise),candidate_rows


def channel_summary(rows):
    by={(r['seed'],r['method'],r['kind'],r['variant'],r['candidate']):r for r in rows};records=[]
    for seed,method,kind in sorted(set((r['seed'],r['method'],r['kind']) for r in rows)):
        entries=[r for r in rows if (r['seed'],r['method'],r['kind'])==(seed,method,kind)];true=entries[0]['true_candidate']
        wrong=sorted({r['candidate'] for r in entries if r['candidate']!=true})
        for candidate in wrong:
            a,b=(by[seed,method,kind,v,candidate] for v in ('original','remap'))
            ta,tb=(by[seed,method,kind,v,true] for v in ('original','remap'))
            d=dict(seed=seed,method=method,kind=kind,candidate=candidate,
                original_nearest=a['nearest_wrong'],remap_nearest=b['nearest_wrong'],
                fixed_wrong_margin_delta=(a['joint_SSE']-ta['joint_SSE'])-(b['joint_SSE']-tb['joint_SSE']),
                historical_margin_delta=(a['historical_SSE']-ta['historical_SSE'])-(b['historical_SSE']-tb['historical_SSE']))
            for k in a['future_channel_SSE']:
                d['future_'+k+'_margin_delta']=(a['future_channel_SSE'][k]-ta['future_channel_SSE'][k])-(b['future_channel_SSE'][k]-tb['future_channel_SSE'][k])
            close(d['fixed_wrong_margin_delta'],d['historical_margin_delta']+sum(v for k,v in d.items() if k.startswith('future_')),'fixed wrong channel identity')
            records.append(d)
    summary=[]
    for method in METHODS:
        for kind in KINDS:
            block=[r for r in records if r['method']==method and r['kind']==kind]
            switches={r['seed'] for r in block if r['original_nearest']!=r['remap_nearest']}
            summary.append(dict(method=method,kind=kind,histories=len({r['seed'] for r in block}),nearest_switch_histories=len(switches),
                fixed_wrong_pairs=len(block),mean_fixed_wrong_margin_delta=float(np.mean([r['fixed_wrong_margin_delta'] for r in block])),
                mean_fixed_wrong_future_V_margin_delta=float(np.mean([r['future_v_margin_delta'] for r in block])),
                by_wrong_candidate=[dict(candidate=c,n=len([r for r in block if r['candidate']==c]),
                    mean_future_V_margin_delta=float(np.mean([r['future_v_margin_delta'] for r in block if r['candidate']==c])),
                    mean_fixed_wrong_margin_delta=float(np.mean([r['fixed_wrong_margin_delta'] for r in block if r['candidate']==c])))
                    for c in sorted({r['candidate'] for r in block})]))
    return summary,records


def review_directory(path,output=None):
    directory=Path(path);freeze=read_json(directory/'freeze.json');parentdir=Path(freeze['parent_directory'])
    sources=source_audit(directory,freeze);require(sources['current_source_matches'],'current source changed since ablation')
    require(file_hash(parentdir/'freeze.json')==freeze['parent_freeze_sha256'],'parent freeze changed')
    hashes={r['name']:r['sha256'] for r in freeze['parent_trial_hashes']};histories=[];candidates=[];errors=[];seen=set()
    for file in sorted(directory.glob('history_*.json.gz')):
        saved=read_json(file);seed=saved['seed'];require(seed in freeze['seeds'] and seed not in seen,'independent seed identity');seen.add(seed)
        require(saved['source_sha256']==freeze['source_sha256'] and saved['parameter_sha256']==freeze['parameter_sha256'],'ablation freeze identity')
        parentpath=parentdir/('trial_'+str(seed)+'.json.gz');expected=hashes[parentpath.name];require(file_hash(parentpath)==expected,'frozen parent trial changed')
        try:
            audit,rows=replay_history(saved,read_json(parentpath),expected);histories.append(audit);candidates.extend(rows)
        except (ValueError,KeyError,RuntimeError,IndexError,TypeError) as e:errors.append(dict(seed=seed,error=type(e).__name__+':'+str(e)))
    channel_stats,channel_rows=channel_summary(candidates)
    counts={k:sum(t['counts'][k] for t in histories) for k in ('candidate_fit_calls','fit_AC_calls','forward_AC_calls')}
    require(read_json(directory/'summary.json')['counts']==counts,'summary call counts')
    csvrows=list(csv.DictReader((directory/'candidate_channel_costs.csv').open(encoding='utf-8-sig',newline='')))
    csvindex={(int(r['seed']),r['method'],r['variant'],r['kind'],r['candidate']):r for r in csvrows}
    require(len(csvindex)==len(csvrows)==len(candidates),'candidate CSV denominator')
    for r in candidates:
        old=csvindex[r['seed'],r['method'],r['variant'],r['kind'],r['candidate']]
        for k in ('joint_SSE','historical_SSE','historical_penalty'):close(float(old[k]),r[k],'CSV cost:'+k)
        for k,v in r['future_channel_SSE'].items():close(float(old['future_'+k+'_SSE']),v,'CSV future channel:'+k)
    out=dict(success=not errors and len(histories)==len(freeze['seeds']),planned_histories=len(freeze['seeds']),audited_histories=len(histories),
        planned_conditions=freeze['planned_conditions'],audited_conditions=sum(len(h['conditions']) for h in histories),review_errors=errors,
        source_audit=sources,counts=counts,total_new_counted_AC_calls=counts['fit_AC_calls']+counts['forward_AC_calls'],
        saved_fit_states_replayed=sum(h['saved_fit_states_replayed'] for h in histories),physical_states_replayed=sum(h['physical_states_replayed'] for h in histories),
        exact_remap_cache_conditions=sum(not c['new_remap'] for h in histories for c in h['conditions']),
        candidate_csv_rows_crosschecked=len(candidates),max_abs_white_error=max((h['max_abs_white_error'] for h in histories),default=None),
        formal_calibration_histories=0,formal_rejection_evaluation_histories=0,channel_summary=channel_stats,fixed_wrong_channel_records=channel_rows,histories=histories,
        reviewer=dict(file=Path(__file__).name,sha256=file_hash(__file__),science_source=False,no_new_fit=True,no_selector=True,no_experiment_generator=True),
        scope='12 independent parent histories; 24 paired electrical conditions; same grid curve, unequal battery service without original driving')
    if output is not None:write_json(output,out)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True,type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    r=review_directory(a.directory,a.output);print(json.dumps({k:r[k] for k in ('success','audited_histories','audited_conditions','review_errors','counts','saved_fit_states_replayed','physical_states_replayed','exact_remap_cache_conditions','max_abs_white_error')}))


if __name__=='__main__':main()
