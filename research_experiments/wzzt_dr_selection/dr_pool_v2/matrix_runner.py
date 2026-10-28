"""Checkpointed multi-source DR comparison with exact observation-fit reuse.

The entire manifest/sample grid is locked once. --max-new-fits limits only one
invocation; resume retains the same protocol and never reruns terminal failures.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import re
import traceback

from . import runner
from .adapter import source_hashes as algorithm_sources
from .observations import normalize, input_hash


class BatchBoundary(BaseException):
    """Intentional non-error boundary, bypassing per-task Exception handling."""


class CachedFitFailure(RuntimeError):
    pass


def observation_identity(training, validation, *, root_label=1, search_seconds=25., sources=None):
    train,days,_=normalize(training,'training')
    valid,valid_days,_=normalize(validation,'validation')
    identity={'training':input_hash(train,days),'validation':input_hash(valid,valid_days),
        'terminal_labels':list(train[0].columns),'root_label':int(root_label),
        'search_seconds':float(search_seconds),'algorithm_source_sha256':algorithm_sources() if sources is None else sources}
    return runner.fingerprint(identity),identity


class SharedFitter:
    def __init__(self, folder, *, sources, max_new_fits=None, fit_function=None):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.sources=sources;self.max_new_fits=max_new_fits;self.fit_function=fit_function
        self.new_count=0;self.reused_count=0

    def __call__(self, training, validation, output_dir, *, root_label=1, search_seconds=25.):
        key,identity=observation_identity(training,validation,root_label=root_label,
            search_seconds=search_seconds,sources=self.sources)
        cache=self.folder/key;meta_path=cache/'cache_record.json';result_path=cache/'result.json'
        request=Path(output_dir);request.mkdir(parents=True,exist_ok=True)
        reused=meta_path.exists()
        if reused:
            meta=runner.read(meta_path)
            if meta['identity']!=identity:raise RuntimeError('cache identity mismatch')
            if result_path.exists():
                result=runner.read(result_path)
                if meta.get('result_sha256') and runner.sha(result_path)!=meta['result_sha256']:
                    raise RuntimeError('cached result hash changed')
                if result.get('status')=='complete':
                    meta.update(status='complete',result_sha256=runner.sha(result_path))
                    runner.write(meta_path,meta)
                elif meta['status']=='running':
                    meta.update(status='failed',exception=result.get('exception',{'message':'identifier returned noncomplete result'}))
                    runner.write(meta_path,meta)
            elif meta['status']=='running':
                meta.update(status='interrupted',exception={'message':'incomplete cached attempt; no budget extension or restart'})
                runner.write(meta_path,meta)
            self.reused_count+=1
        else:
            if self.max_new_fits is not None and self.new_count>=self.max_new_fits:
                raise BatchBoundary()
            cache.mkdir(parents=True,exist_ok=True)
            meta={'key':key,'identity':identity,'status':'running','created_at':runner.now(),
                'first_request':str(request.resolve()),'result_path':str(result_path.resolve())}
            runner.write(meta_path,meta)
            self.new_count+=1
            try:
                if self.fit_function is None:
                    from .adapter import fit_observed
                    fn=fit_observed
                else:fn=self.fit_function
                result=fn(training,validation,cache,root_label=root_label,search_seconds=search_seconds)
                if not result_path.exists():runner.write(result_path,result)
                meta.update(status='complete' if result.get('status')=='complete' else 'failed',
                    result_sha256=runner.sha(result_path),finished_at=runner.now())
            except Exception as exc:
                meta.update(status='failed',finished_at=runner.now(),
                    exception={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
                if result_path.exists():meta['result_sha256']=runner.sha(result_path)
                runner.write(meta_path,meta)
                runner.write(request/'fit_request.json',{'cache_key':key,'status':'failed','reused':False,
                    'reused_from':None,'cache_result':str(result_path),'exception':meta['exception']})
                raise
            runner.write(meta_path,meta)
        reference={'cache_key':key,'status':meta['status'],'reused':reused,
            'reused_from':meta['first_request'] if reused else None,'cache_result':str(result_path.resolve()),
            'cache_record':str(meta_path.resolve())}
        runner.write(request/'fit_request.json',reference)
        if meta['status']!='complete':
            raise CachedFitFailure('cached '+meta['status']+' fit '+key+': '+str(meta.get('exception')))
        returned=copy.deepcopy(runner.read(result_path))
        returned['matrix_cache']=reference
        # Local materialization permits the ordinary runner to resume its task
        # without a second adapter call. Numerical output remains unchanged.
        runner.write(request/'result.json',returned)
        return returned


def build_inventory(manifest, sample_sizes, *, sources):
    inventory=[];hashes={};missing=[]
    for scenario in manifest['scenarios']:
        source=Path(scenario['source_run']).resolve();additional=Path(scenario['additional_strategy']).resolve()
        common={'scenario_id':scenario['id'],'noise':scenario['noise'],
            'response_mode':scenario['response_mode'],'response_weight':float(scenario['response_weight'])}
        strategies=[*runner.STRATEGIES,runner.EXTRA_STRATEGY]
        try:
            summary,paths=runner.source_files(source,additional)
            hashes.update({str(p):runner.sha(p) for p in paths if p.exists()})
            missing.extend(str(p) for p in paths if not p.exists())
            history=[runner.observed_only(runner.load_pickle(source/'history'/(d+'.pkl')))
                for d in summary['wzzt_training_dates']]
            for count in sample_sizes:
                for strategy in strategies:
                    row={**common,'strategy':strategy,'N':count}
                    try:
                        folder=additional if strategy==runner.EXTRA_STRATEGY else source/'strategies'/strategy
                        action=runner.load_pickle(folder/'action.pkl');validation=runner.load_pickle(folder/'validation.pkl')
                        train=runner.subsample_training(history+[runner.observed_only(action['simulation'])],count)
                        valid=[runner.observed_only(validation['simulation'])]
                        key,_=observation_identity(train,valid,sources=sources)
                        row.update(observation_key=key,preflight='ready')
                    except Exception as exc:
                        row.update(observation_key=None,preflight='source_unavailable',
                            exception={'type':type(exc).__name__,'message':str(exc)})
                    inventory.append(row)
        except Exception as exc:
            for count in sample_sizes:
                for strategy in strategies:
                    inventory.append({**common,'strategy':strategy,'N':count,'observation_key':None,
                        'preflight':'source_unavailable','exception':{'type':type(exc).__name__,'message':str(exc)}})
    return inventory,hashes,sorted(set(missing))


def weighted_results(rows, scenarios, sample_sizes):
    """Never renormalize away missing or failed empirical response modes."""
    by_noise=defaultdict(list)
    for scenario in scenarios:by_noise[scenario['noise']].append(scenario)
    lookup={(r['scenario_id'],r['strategy'],r['N']):r for r in rows}
    outputs=[]
    for noise,conditions in by_noise.items():
        total_weight=sum(float(s['response_weight']) for s in conditions)
        for count in sample_sizes:
            for strategy in (*runner.STRATEGIES,runner.EXTRA_STRATEGY):
                for model in runner.MODEL_NAMES:
                    contributing=[]
                    for scenario in conditions:
                        item=lookup.get((scenario['id'],strategy,count))
                        if item and item['status']=='evaluated':
                            metric=item['evaluation']['models'][model]['final']
                            contributing.append((float(scenario['response_weight']),metric))
                    used=sum(w for w,_ in contributing)
                    complete=len(contributing)==len(conditions) and abs(total_weight-1.)<1e-12
                    result={'noise':noise,'N':count,'strategy':strategy,'model':model,
                        'declared_response_modes':len(conditions),'evaluated_response_modes':len(contributing),
                        'declared_weight':total_weight,'evaluated_weight':used,'missing_weight':total_weight-used,
                        'complete_empirical_support':complete,'weight_interpretation':'declared empirical whole-day modes; not a population probability',
                        'weighted_internal_f1':None,'weighted_root_matched':None,
                        'weighted_internal_exact_fraction':None,'weighted_effective_family_fraction':None}
                    if complete:
                        result.update(weighted_internal_f1=sum(w*m['internal_f1'] for w,m in contributing),
                            weighted_root_matched=sum(w*m['root_matched'] for w,m in contributing),
                            weighted_internal_exact_fraction=sum(w*int(m['internal_exact']) for w,m in contributing),
                            weighted_effective_family_fraction=sum(w*int(m['complete_effective_family']) for w,m in contributing))
                    outputs.append(result)
    return outputs


def matrix_summary(protocol, output):
    rows=[];out=Path(output)
    for item in protocol['task_inventory']:
        folder=out/'scenarios'/item['scenario_id']/'tasks'/f"{item['strategy']}__N{item['N']}"
        record=folder/'task_record.json'
        if record.exists():
            row={**runner.read(record),**{k:item[k] for k in ('scenario_id','noise','response_mode','response_weight','observation_key')}}
            request=folder/'identifier'/'fit_request.json'
            if request.exists():row['fit_cache']=runner.read(request)
        else:
            failure=out/'scenarios'/item['scenario_id']/'scenario_failure.json'
            if failure.exists():row={**item,'status':'source_failed','exception':runner.read(failure)}
            elif item['preflight']!='ready':row={**item,'status':'source_unavailable'}
            else:row={**item,'status':'pending'}
        rows.append(row)
    cache=[]
    for path in (out/'fit_cache').glob('*/cache_record.json'):
        item=runner.read(path);cache.append({'key':item['key'],'status':item['status'],'first_request':item['first_request']})
    pending=sum(r['status']=='pending' for r in rows)
    return {'status':'complete' if pending==0 else 'partial','planned_cells':len(rows),
        'recorded_terminal_cells':len(rows)-pending,'pending_cells':pending,
        'evaluated_cells':sum(r['status']=='evaluated' for r in rows),
        'failed_or_unavailable_cells':sum(r['status'] not in ('pending','evaluated') for r in rows),
        'planned_unique_observation_fits':protocol['planned_unique_observation_fits'],
        'attempted_unique_fits':len(cache),'completed_unique_fits':sum(c['status']=='complete' for c in cache),
        'failed_unique_fits':sum(c['status']=='failed' for c in cache),
        'interrupted_unique_fits':sum(c['status'] in ('running','interrupted') for c in cache),
        'reused_evaluated_cells':sum(r.get('fit_cache',{}).get('reused',False) for r in rows if r['status']=='evaluated'),
        'search_wall_limited_cells':sum(r.get('evaluation',{}).get('search_wall_limited',False) for r in rows),
        'response_distribution_scope':'two weighted empirical whole-day response modes, fixed dates/households; not population recovery probability',
        'sample_sizes_share_physical_trajectories':True,'no_noise_condition_weighting':True,
        'untouched_predictive_test':False,'records':rows,'fit_cache_records':cache,
        'weighted_results':weighted_results(rows,protocol['scenarios'],protocol['sample_sizes'])}


def run_matrix(manifest_file, output_dir, *, sample_sizes=(24,48,96), resume=False, max_new_fits=None, fit_function=None):
    manifest_file=Path(manifest_file).resolve();out=Path(output_dir).resolve()
    manifest=runner.read(manifest_file)
    if manifest.get('status')!='complete':raise ValueError('physical manifest is not complete')
    scenarios=manifest['scenarios']
    if not scenarios or len({s['id'] for s in scenarios})!=len(scenarios):raise ValueError('scenario IDs must be unique')
    for scenario in scenarios:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',scenario['id']):raise ValueError('unsafe scenario ID')
        if not 0<float(scenario['response_weight'])<=1:raise ValueError('invalid empirical response weight')
    sample_sizes=tuple(sample_sizes)
    if not sample_sizes or len(set(sample_sizes))!=len(sample_sizes) or any(n not in runner.SAMPLE_COUNTS for n in sample_sizes):
        raise ValueError('sample sizes must be unique values from24,48,96')
    if max_new_fits is not None and (not isinstance(max_new_fits,int) or max_new_fits<1):raise ValueError('max-new-fits must be positive')
    alg=algorithm_sources()
    inventory,source_inputs,missing=build_inventory(manifest,sample_sizes,sources=alg)
    controller_files=sorted(Path(__file__).parent.glob('*.py'))
    controller_files += sorted(Path(__file__).parent.glob('provenance.json'))
    contract={'version':'dr_pool_matrix_v1','manifest':str(manifest_file),'manifest_sha256':runner.sha(manifest_file),
        'scenarios':scenarios,'sample_sizes':list(sample_sizes),'planned_cells':len(inventory),
        'planned_unique_observation_fits':len({r['observation_key'] for r in inventory if r['observation_key'] is not None}),
        'task_inventory':inventory,'source_input_sha256':source_inputs,'missing_source_files':missing,
        'algorithm_source_sha256':alg,'controller_source_sha256':{str(p.resolve()):runner.sha(p) for p in controller_files},
        'cache_key':'normalized training/validation P,Q,V,root + labels + physical dates + root/algorithm configuration/source hash',
        'batch_policy':'max-new-fits is an invocation limit only; same full protocol on resume; cached failures also terminal',
        'price_reoptimization':False,'sample_sizes_are_nested':True,'untouched_predictive_test':False,
        'weight_policy':'retain each declared response weight; never renormalize missing or failed cells; no overall noise mixture'}
    identity=runner.fingerprint(contract)
    out.mkdir(parents=True,exist_ok=True)
    with runner._exclusive_lock(out,identity):
        path=out/'protocol.json'
        if path.exists():
            protocol=runner.read(path)
            if not resume:raise FileExistsError('existing matrix run; use unchanged --resume')
            if protocol['fingerprint']!=identity:raise ValueError('matrix resume fingerprint changed')
        else:
            protocol={**contract,'fingerprint':identity,'created_at':runner.now()};runner.write(path,protocol)
        shared=SharedFitter(out/'fit_cache',sources=alg,max_new_fits=max_new_fits,fit_function=fit_function)
        stopped=False
        for scenario in scenarios:
            target=out/'scenarios'/scenario['id']
            if (target/'scenario_failure.json').exists():continue
            try:
                runner.run(scenario['source_run'],target,additional_strategy=scenario['additional_strategy'],
                    resume=(target/'protocol.json').exists(),sample_sizes=sample_sizes,fitter=shared)
            except BatchBoundary:
                stopped=True;break
            except Exception as exc:
                runner.write(target/'scenario_failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
            runner.write(out/'summary.json',matrix_summary(protocol,out))
        summary=matrix_summary(protocol,out)
        summary.update(batch_new_unique_fits=shared.new_count,batch_cache_hits=shared.reused_count,
            stopped_at_invocation_limit=stopped,updated_at=runner.now(),protocol_sha256=runner.sha(path),
            source_inputs_unchanged=all(Path(p).exists() and runner.sha(p)==h for p,h in source_inputs.items()),
            missing_sources_unchanged=all(not Path(p).exists() for p in missing))
        if not summary['source_inputs_unchanged'] or not summary['missing_sources_unchanged']:summary['status']='source_changed'
        runner.write(out/'summary.json',summary)
        return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--sample-sizes',type=int,nargs='+',default=[24,48,96]);p.add_argument('--resume',action='store_true')
    p.add_argument('--max-new-fits',type=int)
    args=p.parse_args(argv)
    result=run_matrix(args.manifest,args.output_dir,sample_sizes=args.sample_sizes,resume=args.resume,max_new_fits=args.max_new_fits)
    print(json.dumps({k:result[k] for k in ('status','planned_cells','evaluated_cells','pending_cells','failed_or_unavailable_cells','planned_unique_observation_fits','attempted_unique_fits','reused_evaluated_cells','stopped_at_invocation_limit')},indent=2))
    return 0 if result['status'] in ('complete','partial') and result['failed_or_unavailable_cells']==0 else 2


if __name__=='__main__':raise SystemExit(main())
