"""Prospective isolated pilot: observed-only planning, sealed-plan AC evaluation."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict, replace, is_dataclass
from hashlib import sha256
import gzip,json,os,sys,traceback,zipfile
from pathlib import Path
from time import perf_counter
import numpy as np
from .. import likelihood as lk
from .. import ev_ac_repair_refit as refit
from .. import ev_systematic_refit as legacy
from .. import ev_bayes_response_cases as cases
from .. import ev_bayes_small_study as operation
from .. import ev_mobile_v3_cases as public
from .. import ev_mobile_v4_domain as nested
from .. import gradient_price_study as io
from ..information_context import meter_to_record
from ..block_joint_margin import study as previous
from .config import CONFIG
from .coefficients import QuadraticTask,build_tasks_from_responses,score_tasks
from .context import FixedResponses
from .milp import solve_tasks

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT=ROOT/'outputs'/'g_precision_v1'


def serial(value):
    if is_dataclass(value):return serial(asdict(value))
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    if isinstance(value,np.ndarray):return serial(value.tolist())
    if isinstance(value,np.generic):return serial(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,Path):return str(value)
    return value


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(serial(value),ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
    if path.suffix=='.gz':data=gzip.compress(data,compresslevel=3,mtime=0)
    temporary=path.with_name(path.name+'.tmp');temporary.write_bytes(data);temporary.replace(path)


def read(path):
    p=Path(path);data=p.read_bytes()
    if p.suffix=='.gz':data=gzip.decompress(data)
    return json.loads(data)


def source_hashes():
    paths=[]
    for relative in ('research_experiments/mobile_ev_mle','research_experiments/mobile_ev_probing','rnj_wzzt_core'):
        paths.extend(p for p in (ROOT/relative).rglob('*.py')
                     if not any(x in p.parts for x in ('evidence','__pycache__'))
                     and not any(x.startswith('results') for x in p.parts)
                     and p != Path(__file__).with_name('analysis.py'))
    for relative in ('research_experiments/mobile_ev_mle/METHOD_NAMES.md',
                     'research_experiments/mobile_ev_mle/g_precision/math.md',
                     'research_experiments/mobile_ev_mle/g_precision/protocol.md'):
        paths.append(ROOT/relative)
    return {p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def freeze(output,config):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    data=dict(config=config,source_sha256=source_hashes(),base_commit='a72f083f',
              truth_in_planning=False,independent_unit='history; paired regimes are not independent histories')
    identity=sha256(json.dumps(serial(data),sort_keys=True).encode()).hexdigest()
    data['identity']=identity;path=output/'freeze.json'
    if path.exists():
        if read(path)!=serial(data):raise ValueError('frozen science changed: use a new output version')
    else:
        write(path,data)
        with zipfile.ZipFile(output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in data['source_sha256']:z.write(ROOT/name,name)
        write(output/'environment.json',dict(python=sys.version,numpy=np.__version__,
            executable=sys.executable,threads={k:os.environ.get(k) for k in
                ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},source_count=len(data['source_sha256'])))
    return data


def specifications(phase,config=CONFIG):
    seeds=config[phase+'_seeds'];result=[]
    for i,seed in enumerate(seeds):
        if phase=='development':
            bank,family,H,excitation=('n3',0,32,'ordinary') if i==0 else ('n4',1,96,'correlated')
        else:
            bank='n3' if i<4 else 'n4';family=i%4
            H=32 if family%2==0 else 96
            excitation='ordinary' if (family+(i//4))%2==0 else 'correlated'
        result.append(dict(phase=phase,index=i,seed=seed,bank=bank,family_index=family,
            H=H,excitation=excitation,innovation='nominal'))
    return result


def precision_task(meter,shape):
    from ..ev_rx_precision import precision_weights
    return QuadraticTask('public_precision',0.,np.zeros(shape),precision_weights(meter,shape))


def result_plan(name,domain,x):
    return previous._plan_record(name,domain,np.asarray(x))


def plan_family(domains,tasks,baselines,config,*,extra=None):
    plans={};logs={};began=perf_counter()
    for name,domain in domains.items():
        parents={key:nested.lift_operation(domains['fixed_charge'],domain,x)
                 for key,x in baselines.items()}
        if extra is not None:parents['same_domain_P-SQ']=extra[name]
        for parent in nested.PARENTS[name]:
            parents['parent_'+parent]=nested.lift_operation(domains[parent],domain,plans[parent])
        result=solve_tasks(domain,tasks,parents,iterations=config['iterations'],
            segments=config['segments'],max_segments=config['max_segments'],
            solver_seconds=config['solver_seconds'],total_seconds=config['total_seconds_per_domain'],
            backend=config['backend'],relative_gap=config['relative_gap'],absolute_gap=config['absolute_gap'])
        if result.get('operation_x') is None:
            raise RuntimeError('No feasible '+name+' plan: '+str(result.get('failures')))
        plans[name]=np.asarray(result['operation_x']).copy();logs[name]=result
    return plans,logs,perf_counter()-began


def select(observed,fit_options,config,seed,progress):
    """No case, truth, family index, future path, or future noise argument."""
    began=perf_counter()
    D=lk.concat_observations(observed['training'],observed['evidence'])
    dm=legacy.concat_models((observed['training'],observed['evidence']),
        (observed['train_meter'],observed['evidence_meter']))
    history=refit.fit_history(observed['candidates'],D,dm,fit_options,seed)
    progress('history_fitted',dict(complete=history.complete,failures=history.failures))
    template=replace(public.service_domain(observed['horizon'],D.n,mobile=True).config,
        n_slots=observed['horizon'])
    domains=nested.build_nested_domains(template);shape=domains['mobile_bidir'].indices.u.shape
    constant=np.zeros(shape)
    constant[:,template.initial_site]=(template.terminal_energy_kwh-
        template.initial_soc*template.battery_capacity_kwh)/(template.eta_charge*shape[0]*template.dt_hours)
    highlow=np.zeros(shape);highlow[:shape[0]//2,template.initial_site]=2*constant[0,template.initial_site]
    baseline_began=perf_counter()
    constant_x=operation.complete_curve(domains['fixed_charge'],constant)
    highlow_x=operation.complete_curve(domains['fixed_charge'],highlow)
    baseline_seconds=perf_counter()-baseline_began
    baselines=dict(constant=constant_x,highlow=highlow_x)
    psq,logs,seconds=plan_family(domains,[precision_task(observed['future_meter'],shape)],baselines,config)
    plans={'CC':result_plan('fixed_charge',domains['fixed_charge'],constant_x),
           'HL':result_plan('fixed_charge',domains['fixed_charge'],highlow_x),
           'P-SQ':result_plan('mobile_bidir',domains['mobile_bidir'],psq['mobile_bidir'])}
    zero=np.zeros(shape);plans['OBS']=dict(domain='passive_no_battery_task',power=zero,
        action_key=nested.action_key(zero),passive_control=True)
    result=dict(history=history.record(),plans=plans,domain_config=asdict(template),failures={},
        families={'P-SQ':logs},cost={'history_seconds':history.seconds,
            'baseline_completion_seconds':baseline_seconds,'P-SQ_seconds':seconds},
        declared_targets={'G-SQ':'minimum all-pair incremental affine response distance',
                          'G-AQ':'minimum all-pair total affine response distance'})
    progress('P-SQ_fixed',dict(seconds=seconds))
    context=None
    try:
        context=FixedResponses(observed['candidates'],D,dm,observed['future_meter'],history,
            horizon=observed['horizon'],base_kw=observed['base_kw'])
        result['cost']['response_setup_seconds']=context.seconds
        for method,include in [('G-SQ',False),('G-AQ',True)]:
            tasks=build_tasks_from_responses(context.zero,context.sigma,include_intercept=include)
            family,rows,spent=plan_family(domains,tasks,baselines,config,extra=psq)
            result['families'][method]=rows;result['cost'][method+'_seconds']=spent
            plan=result_plan('mobile_bidir',domains['mobile_bidir'],family['mobile_bidir'])
            try:
                context.scores(plan['power'])
            except Exception as exc:
                result['failures'][method+'_forecast_physical']=type(exc).__name__+': '+str(exc)
                plan=deepcopy(plans['P-SQ']);plan['fallback_reason']='forecast physical failure'
            plans[method]=plan
            result.setdefault('tasks',{})[method]=[t.record() for t in tasks]
            progress(method+'_fixed',dict(seconds=spent,score=score_tasks(tasks,plan['power'])))
        result['nominal_AC']={}
        for method,plan in plans.items():
            try:result['nominal_AC'][method]=context.scores(plan['power'])
            except Exception as exc:
                result['nominal_AC'][method]=dict(failure=type(exc).__name__+': '+str(exc))
        result['context']=context.record()
    except Exception as exc:
        result['failures']['response_or_planning']=type(exc).__name__+': '+str(exc)
    for method in ('G-SQ','G-AQ'):
        if method not in plans:
            plans[method]=deepcopy(plans['P-SQ']);plans[method]['fallback_reason']='new-method planning failure'
    result['cost']['selection_total_seconds']=perf_counter()-began
    result['selector_uses_truth']=False;result['selector_uses_future_noise']=False
    return result,D,dm,history


def run(spec,output,config=CONFIG):
    output=Path(output);frozen=freeze(output,config)
    path=output/spec['phase']/('history_%03d.json.gz'%spec['index'])
    if path.exists():raise FileExistsError(path)
    began=perf_counter()
    def progress(stage,details=None):
        item=dict(spec=spec,stage=stage,details=details,seconds=perf_counter()-began,
            freeze_identity=frozen['identity'])
        write(output/'checkpoint.json',item)
        with (output/'progress.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(serial(item),ensure_ascii=False,sort_keys=True)+'\n')
        print('%s %s %s %.1fs'%(spec['phase'],spec['index'],stage,item['seconds']),flush=True)
    record=dict(spec=spec,config=config,freeze_identity=frozen['identity'],failures={})
    try:
        progress('started')
        case=cases.make_case(spec);observed=cases.planning_inputs(case)
        fit_options=previous._fit_options(case['config'],config)
        record.update(observed_training=io.observations_record(observed['training']),
            observed_evidence=io.observations_record(observed['evidence']),
            train_meter=meter_to_record(observed['train_meter']),
            evidence_meter=meter_to_record(observed['evidence_meter']),
            future_meter=meter_to_record(observed['future_meter']),fit_options=fit_options,
            candidate_structures=[dict(name=c.name,n=c.n,supports=c.supports) for c in observed['candidates']],
            provenance=case['provenance'])
        planning,D,dm,history=select(observed,fit_options,config,previous._seed(spec['seed'],1000),progress)
        record['planning']=planning;record['failures'].update(planning['failures'])
        progress('selection_sealed',dict(action_keys={m:p['action_key'] for m,p in planning['plans'].items()},
            truth_seen_by_selector=False))
        record['evaluation']=previous._evaluate(case,spec,observed,planning['plans'],D,dm,history,
            fit_options,config,lambda stage,details=None:progress(stage,dict(
                failure=details.get('failure') if isinstance(details,dict) else None)))
        for key,action in record['evaluation']['actions'].items():
            if action.get('failures'):record['failures']['evaluation_'+key]=action['failures']
        record['status']='complete' if not record['failures'] else 'partial_failure'
    except Exception as exc:
        record['status']='failed';record['failures']['exception']=dict(message=type(exc).__name__+': '+str(exc),
            traceback=traceback.format_exc())
    record['seconds']=perf_counter()-began
    record['source_unchanged']=source_hashes()==frozen['source_sha256']
    write(path,record);progress('history_finished',dict(status=record['status'],path=str(path)))
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['development','evaluation'],required=True)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--index',type=int)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    frozen=freeze(args.output,CONFIG)
    if args.phase=='evaluation':
        required=[args.output/'development'/('history_%03d.json.gz'%s['index']) for s in specifications('development')]
        if not all(p.exists() for p in required):raise RuntimeError('development must finish before evaluation')
        if not all(read(p).get('status')=='complete' for p in required):
            raise RuntimeError('development contains failures; preserve evidence before evaluation')
    for spec in specifications(args.phase):
        if args.index is not None and spec['index']!=args.index:continue
        path=args.output/spec['phase']/('history_%03d.json.gz'%spec['index'])
        if path.exists() and args.resume:
            if read(path)['freeze_identity']!=frozen['identity']:raise ValueError('resume freeze mismatch')
            print('retained existing',path,flush=True);continue
        run(spec,args.output)


if __name__=='__main__':main()
