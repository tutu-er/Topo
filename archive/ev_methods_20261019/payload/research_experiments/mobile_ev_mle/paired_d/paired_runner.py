"""Fresh paired D runner; importing this file does not generate history.

Actual execution requires an explicitly frozen manifest.  The parent must
review this runner and authorize pilot before invoking --phase pilot.
All output paths are separate from original C/D evidence.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,json,math,platform,sys,os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import scipy
from scipy.stats import beta,binomtest
try:
    from . import paired_contracts as contracts
except ImportError:
    import paired_contracts as contracts

HERE=Path(__file__).resolve().parent
ROOT=Path(r'C:\Users\23761\.codex\worktrees\36a1\Topo')
FROZEN_D=ROOT/'research_experiments/mobile_ev_mle/outputs/adaptive_rolling/frozen_D_execution.json'
CHANNELS=('p','q','v','master_p','master_q','root_v')
METHODS=contracts.METHODS


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def finite(value):
    if isinstance(value,np.ndarray):return finite(value.tolist())
    if isinstance(value,np.generic):return finite(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {str(k):finite(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite(v) for v in value]
    return value
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(finite(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(value):return hashlib.sha256(json.dumps(finite(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def runtime():return dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
def runner_hashes():
    return {p.name:sha(p) for p in (HERE/'paired_runner.py',HERE/'paired_contracts.py',HERE/'paired_D_protocol.md')}


def ensure_staging(path):
    resolved=Path(path).resolve()
    if not resolved.is_relative_to(HERE.resolve()):
        raise ValueError('paired outputs must remain inside the approved D_paired_plan staging directory')
    return resolved


class ReplayMismatch(RuntimeError):pass


class RealProject:
    """All generator/truth/AC access is confined to this evaluator facade."""
    def __init__(self):
        sys.dont_write_bytecode=True
        sys.path.insert(0,str(ROOT))
        from research_experiments.mobile_ev_mle import adaptive_rolling_batch as batch
        self.batch=batch;self.rolling=batch.rolling
        self.original=read(FROZEN_D)
    def original_source_hashes(self):return self.batch.source_hashes()
    def source_corpus(self):
        # Explicit original dependencies plus every production rnj_wzzt .py
        # covers deferred imports without importing unused experiment modules.
        files=set();base=ROOT/'research_experiments/mobile_ev_mle'
        for key in self.original['calibration']['source_sha256']:
            name=key.split('::',1)[1] if key.startswith('loaded::') else key
            files.add((base/name).resolve())
        import rnj_wzzt
        for directory in rnj_wzzt.__path__:
            files.update(Path(directory).resolve().rglob('*.py'))
        for name,module in list(sys.modules.items()):
            if name.startswith(('research_experiments.','rnj_wzzt.')) or name=='ac_general_v9':
                filename=getattr(module,'__file__',None)
                if filename and filename.endswith('.py'):files.add(Path(filename).resolve())
        return {str(p):sha(p) for p in sorted(files)}
    def verify_loaded_corpus(self,corpus):
        for name,module in list(sys.modules.items()):
            if name.startswith(('research_experiments.','rnj_wzzt.')) or name=='ac_general_v9':
                filename=getattr(module,'__file__',None)
                if filename and filename.endswith('.py'):
                    path=str(Path(filename).resolve())
                    if path not in corpus or sha(path)!=corpus[path]:raise ValueError('unfrozen loaded project source '+name)
    def fresh_trace(self,seed,phase,policy,folder):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        cfg=self.batch.batch_config(phase=phase,count=1,start_seed=seed,policy=policy)
        try:
            raw,metrics=self.batch._counted_trial(seed,policy=policy,stress=None,output=folder)
            trace=self.batch.normalize_trial(raw,metrics,cfg)
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            trace=self.batch._failure_trial(seed,cfg,error)
        dump(folder/f'trace_{seed}.json',trace)
        archive=folder/f'rolling_{seed}_observations.npz'
        return trace,archive
    def path(self,seed,policy):return self.rolling._evaluation_path(seed,policy,None)
    def observe(self,path,rows,action=None):return self.rolling.observe(path,rows,action)
    def action(self,name,u,sites):
        return self.rolling.EVAction(name,np.asarray(u)/20.,np.zeros_like(u),np.asarray(sites),name)
    def concat(self,h,y):return self.rolling.concat_observations(h,y)
    def validate_operations(self,pair,config,trace):
        def decode(p):
            return self.rolling.EVPlan(tuple(self.rolling.Arc(**a) for a in p['arcs']),np.ones(len(p['arcs'])),
                np.asarray(p['p_plus_kw']),np.asarray(p['p_minus_kw']),np.asarray(p['u_kw']),np.asarray(p['energy_kwh']))
        cfg=self.rolling.EVConfig(**config);first=decode(pair['openloop']);actual=decode(pair['rolling'])
        for plan in [first,actual]:
            checked=self.rolling.check_plan(plan,cfg,tolerance=2e-6)
            if not checked['feasible']:raise ValueError('saved plan violates original operational checker: '+str(checked['issues']))
        prefix=self.rolling.execute_prefix(first,cfg,2);remaining=self.rolling.remaining_config(cfg,prefix)
        saved=trace['records'][0]['executed'];second=trace['selections'][1]['ev_config']
        if prefix.next_site is None or prefix.arrival_remaining_slots:raise ValueError('unfinished prefix transit')
        for key in ['grid_energy_kwh','throughput_kwh','driving_energy_kwh']:
            if abs(getattr(prefix,key)-saved[key])>2e-6:raise ValueError('prefix '+key+' mismatch')
        if not np.array_equal(prefix.u_kw,np.asarray(saved['u_kw'])):raise ValueError('actual prefix power mismatch')
        if abs(prefix.energy_kwh[-1]-saved['energy_kwh'][-1])>2e-6:raise ValueError('prefix SOC mismatch')
        if second['initial_site']!=remaining.initial_site or abs(second['initial_soc']-remaining.initial_soc)>2e-6:
            raise ValueError('second-round position/SOC reset')
        if abs(second['net_grid_energy_kwh']-remaining.net_grid_energy_kwh)>2e-6:raise ValueError('second grid task reset')
        if not np.array_equal(np.asarray(second['access']),remaining.arrays()[0]):raise ValueError('remaining access mismatch')
        for key in ['terminal_site','terminal_energy_kwh','max_throughput_kwh','require_terminal_connection']:
            if second.get(key)!=getattr(remaining,key):raise ValueError('remaining terminal/throughput contract mismatch')
        return dict(prefix=finite(asdict(prefix)),remaining=finite(asdict(remaining)),original_checker_verified=True)
    def fit_bank(self,H20,initial,seed):
        bank=tuple(self.rolling.RadialCandidate.from_supports(name,3,family) for name,family in zip(
            contracts.FAMILIES,self.rolling.FAMILIES))
        with self.batch._count_ac_calls() as calls:
            began=perf_counter()
            bank_failure=False
            try:fitted,failed=self.rolling.fit_bank(bank,H20,self.rolling.model(3),initial=initial,seed=seed,multistart=2,max_nfev=300)
            except (RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                fitted={};failed={'fit_bank_execution':str(error)};bank_failure=True
            seconds=perf_counter()-began
        losses=[float(fitted[c.name].loss) if c.name in fitted else math.inf for c in bank]
        success=[bool(c.name in fitted and fitted[c.name].success) for c in bank]
        rows=[self.rolling.fit_record(c,fitted[c.name]) for c in bank if c.name in fitted]
        latent={c.name:{key:np.asarray(getattr(fitted[c.name],key)) for key in ('p','q','source_p','source_q')}
                for c in bank if c.name in fitted}
        return dict(losses=losses,success=success,fits=rows,fit_failures=failed,latent=latent,
                    seconds=seconds,ac_calls=calls['count'],counter_includes_failed_entries=True,
                    bank_execution_failure=bank_failure,missing_objects_not_claim_every_optimizer_failed=bank_failure)


def freeze(path,project=None):
    path=ensure_staging(path)
    project=project or RealProject();original=project.original
    expected=original['calibration']['source_sha256']
    actual=project.original_source_hashes()
    if expected!=actual:raise ValueError('original D source contract changed; refuse freeze')
    if original['calibration']['runtime']!=runtime():raise ValueError('original numerical runtime differs')
    manifest=dict(version='fresh_paired_D_common_RX_v1',declared_at_utc=datetime.now(timezone.utc).isoformat(),
        original_frozen_D_sha256=sha(FROZEN_D),original_source_sha256=actual,
        original_policy=original['calibration']['policy'],original_initial_ev=original['calibration']['initial_ev_config'],
        population=original['calibration']['population'],schedule=contracts.SCHEDULE,methods=list(METHODS),
        source_sha256=runner_hashes(),runtime=runtime(),
        actual_loaded_and_deferred_core_source_sha256=project.source_corpus() if hasattr(project,'source_corpus') else {},
        execution_resources=dict(BLAS_OMP_MKL_threads=1,process_priority='below_normal'),
        score_axes=['shared_H14','rolling_common_initial_H20','openloop_common_initial_H20'],
        common_initial='R/X only from first-round H14 fit; meter-initialized other states',
        common_fit_seed='policy_seed+2200; same candidate index offsets',multistart=2,max_nfev=300,
        selected_actions_fixed_before_replay_and_refit=True,gates_influence_actions=False,
        actual_history_unit='fresh full exogenous path plus both fixed saved action evaluations',
        conditional_alpha_spending_claim=False,
        replay_tolerances=dict(exact=['p','q','root_v'],absolute_other=2e-10),
        new_formal_samples_generated_after_freeze=True,original_C_D_thresholds_reused=False)
    manifest['policy_id']=digest(manifest)
    if Path(path).exists():raise ValueError('manifest already exists; never overwrite an execution freeze')
    dump(path,manifest);return manifest


def verify_manifest(manifest,project):
    body={k:v for k,v in manifest.items() if k!='policy_id'}
    if digest(body)!=manifest['policy_id']:raise ValueError('paired manifest identity differs from its body')
    if manifest['source_sha256']!=runner_hashes():raise ValueError('paired source/protocol changed after freeze')
    if manifest['runtime']!=runtime():raise ValueError('paired runtime changed')
    if manifest['original_source_sha256']!=project.original_source_hashes():raise ValueError('original controller source changed')
    if manifest['original_frozen_D_sha256']!=sha(FROZEN_D):raise ValueError('original D manifest changed')
    if manifest['schedule']!=contracts.SCHEDULE:raise ValueError('paired schedule changed')
    corpus=manifest.get('actual_loaded_and_deferred_core_source_sha256',{})
    for filename,expected in corpus.items():
        if sha(filename)!=expected:raise ValueError('frozen core source changed '+filename)
    if hasattr(project,'verify_loaded_corpus'):project.verify_loaded_corpus(corpus)


def actual_worker_resources():
    for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
        if os.environ.get(name)!='1':raise ValueError('actual paired run requires '+name+'=1 before Python imports')
    if os.name=='nt':
        import ctypes
        kernel=ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
        kernel.GetPriorityClass.argtypes=[ctypes.c_void_p];kernel.GetPriorityClass.restype=ctypes.c_uint32
        handle=kernel.GetCurrentProcess()
        if not kernel.SetPriorityClass(handle,0x4000):raise ctypes.WinError()
        if kernel.GetPriorityClass(handle)!=0x4000:raise RuntimeError('below-normal priority was not applied')
    return dict(threads=1,process_priority='below_normal' if os.name=='nt' else 'not_windows')


def observation_arrays(obs):return {key:np.asarray(getattr(obs,key)) for key in CHANNELS}


def compare_replay(rebuilt,archive,tolerances):
    checks={}
    for name,a in rebuilt.items():
        if name not in archive:raise ReplayMismatch('missing archive channel '+name)
        b=np.asarray(archive[name])
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ReplayMismatch('replay shape or finite-value contract '+name)
        delta=np.abs(a-b);maximum=float(delta.max(initial=0))
        channel=next((c for c in sorted(CHANNELS,key=len,reverse=True) if name.endswith('_'+c)),None)
        if channel is None:raise ReplayMismatch('unknown replay channel '+name)
        exact=channel in tolerances['exact']
        limit=0. if exact else tolerances['absolute_other']
        checks[name]=dict(byte_equal=a.tobytes()==b.tobytes(),exact_values=np.array_equal(a,b),
                         maximum_absolute_difference=maximum,tolerance=limit,
                         unequal_value_count=int(np.count_nonzero(a!=b)))
        if maximum>limit:raise ReplayMismatch('replay exceeds tolerance: '+name+' '+str(checks[name]))
    return checks


def replay_pair(seed,trace,archive,manifest,project):
    pair=contracts.saved_action_pair(trace['selections'][0]['actual_plan'],trace['realized_plan'])
    operations={m:contracts.operation_audit(pair[k],manifest['original_initial_ev']) for m,k in zip(METHODS,['rolling','openloop'])}
    original_operations=project.validate_operations(pair,manifest['original_initial_ev'],trace)
    path,policy_seed,truth,case=project.path(seed,manifest['original_policy'])
    if truth!=trace['true_index'] or policy_seed!=trace['policy_seed'] or case!=trace['case']:
        raise ReplayMismatch('regenerated evaluator identity differs')
    H0,_=project.observe(path,slice(0,12))
    rolling=pair['rolling'];openloop=pair['openloop']
    a1=project.action('shared_prefix',np.asarray(rolling['u_kw'])[:2],operations[METHODS[0]]['sites'][:2])
    Y1,P1=project.observe(path,slice(12,14),a1);H14=project.concat(H0,Y1)
    aR=project.action(METHODS[0],np.asarray(rolling['u_kw'])[2:],operations[METHODS[0]]['sites'][2:])
    YR,PR=project.observe(path,slice(14,20),aR);HR=project.concat(H14,YR)
    rebuilt={**{'history_'+k:v for k,v in observation_arrays(H0).items()},
             **{'round_1_observed_'+k:v for k,v in observation_arrays(Y1).items()},
             **{'round_2_observed_'+k:v for k,v in observation_arrays(YR).items()},
             **{'round_1_joint_'+k:v for k,v in observation_arrays(H14).items()},
             **{'round_2_joint_'+k:v for k,v in observation_arrays(HR).items()}}
    if not archive.exists():raise ReplayMismatch('mandatory fresh observation archive absent')
    with np.load(archive,allow_pickle=False) as saved:
        check=compare_replay(rebuilt,saved,manifest['replay_tolerances'])
        if not np.array_equal(np.asarray(rolling['u_kw']),saved['realized_u_kw']):raise ReplayMismatch('realized action archive differs')
    aO=project.action(METHODS[1],np.asarray(openloop['u_kw'])[2:],operations[METHODS[1]]['sites'][2:])
    YO,PO=project.observe(path,slice(14,20),aO);HO=project.concat(H14,YO)
    voltage={METHODS[0]:np.r_[P1['v'].ravel(),PR['v'].ravel()],METHODS[1]:np.r_[P1['v'].ravel(),PO['v'].ravel()]}
    physical={m:dict(voltage_min=float(v.min()),voltage_max=float(v.max()),operational_voltage_ok=bool(v.min()>=.9 and v.max()<=1.1)) for m,v in voltage.items()}
    return dict(observations={METHODS[0]:HR,METHODS[1]:HO},shared_H14=H14,physical=physical,
                plans=pair,operations=operations,original_operations=original_operations,replay=check,
                truth=truth,truth_r=np.asarray(path.r),truth_x=np.asarray(path.x))


def bank_record(method,fit,physical,selection_failure,truth,truth_r,truth_x):
    losses=fit['losses'];valid=all(fit['success']) and all(v is not None and math.isfinite(v) and v>=0 for v in losses)
    margin=min(v for j,v in enumerate(losses) if j!=truth)-losses[truth] if valid else None
    true_fit=next((f for f in fit['fits'] if f['name']==contracts.FAMILIES[truth] and f['success']),None)
    impedance=None
    if true_fit is not None:
        impedance=float(np.sqrt(np.mean(np.r_[((np.asarray(true_fit['r_pu'])-truth_r)/truth_r)**2,
                                                ((np.asarray(true_fit['x_pu'])-truth_x)/truth_x)**2])))
    return dict(method=method,round=2,status='complete' if valid else 'joint_fit_failure',
        **{k:v for k,v in fit.items() if k!='latent'},**physical,selection_failure=selection_failure,
        wrong_margin=margin,impedance_relative_rmse=impedance,global_MLE_certified=False)


def failed_pair(seed,trace,error,manifest,seconds):
    first=trace.get('records',[dict(losses=[None]*4,success=[False]*4)])[0]
    valid_first=sum(bool(ok and v is not None and math.isfinite(v) and v>=0) for ok,v in zip(first['success'],first['losses']))
    return dict(seed=seed,history_id=str(seed),policy_id=manifest['policy_id'],case=trace.get('case'),
        true_index=trace.get('true_index'),truth_topology=trace.get('truth_topology'),
        candidate_names=list(contracts.FAMILIES),independent_history_unit=True,first_bank=first,
        records=[dict(method=m,round=2,status='paired_history_unavailable',losses=[None]*4,success=[False]*4,
                     fits=[],fit_failures={'paired_trace':str(error)},selection_failure=None,
                     operational_voltage_ok=None,full_evidence_unavailable=True) for m in METHODS],
        score_record=finite(asdict(contracts.PairScore(math.inf,math.inf,False,12,12-valid_first))),
        seconds=seconds,paired_trace_failure=True,error=str(error),
        source_sha256=manifest['source_sha256'],actual_actions_not_modified=True,
        failure_placeholder_is_not_claim_every_optimizer_failed=True)


def verify_finished(row,out,manifest,project):
    try:verify_manifest(manifest,project)
    except ValueError:
        dump(Path(out)/f"pending_source_drift_{row['seed']}.json",row)
        raise
    return row


def run_history(seed,phase,out,manifest,project=None):
    out=ensure_staging(out);project=project or RealProject();began=perf_counter()
    if isinstance(project,RealProject):actual_worker_resources()
    verify_manifest(manifest,project)
    folder=out/'fresh_traces'/str(seed)
    trace,archive=project.fresh_trace(seed,phase,manifest['original_policy'],folder)
    trace_json=folder/f'trace_{seed}.json'
    provenance=dict(trace_json=str(trace_json),trace_sha256=sha(trace_json),
                    archive=str(archive),archive_sha256=sha(archive) if archive.exists() else None)
    if trace.get('execution_failure'):
        row=failed_pair(seed,trace,'original fresh shadow trajectory incomplete',manifest,perf_counter()-began)
        row['fresh_trace']=provenance;return verify_finished(row,out,manifest,project)
    # Replay mismatches are contract violations, not low likelihood. They stop
    # the batch before scoring instead of being silently zero-weighted.
    try:replay=replay_pair(seed,trace,archive,manifest,project)
    except ReplayMismatch:raise
    except (RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
        row=failed_pair(seed,trace,error,manifest,perf_counter()-began)
        row.update(fresh_trace=provenance,numerical_replay_failure=True)
        return verify_finished(row,out,manifest,project)
    first=trace['records'][0]
    try:initial=contracts.common_initial(first['fits'])
    except ValueError as error:
        row=failed_pair(seed,trace,error,manifest,perf_counter()-began)
        row.update(fresh_trace=provenance,replay_checks=replay['replay'],initialization_failure=True)
        for record in row['records']:
            record.update(replay['physical'][record['method']])
            record['selection_failure']=(any(s.get('status')!='complete' for s in trace['selections'])
                if record['method']==METHODS[0] else trace['selections'][0].get('status')!='complete')
        return verify_finished(row,out,manifest,project)
    common_seed=trace['policy_seed']+2200
    order=list(METHODS) if seed%2 else list(reversed(METHODS))
    fits={m:project.fit_bank(replay['observations'][m],initial,common_seed) for m in order}
    failures={METHODS[0]:any(s.get('status')!='complete' for s in trace['selections']),
              METHODS[1]:trace['selections'][0].get('status')!='complete'}
    records=[bank_record(m,fits[m],replay['physical'][m],failures[m],replay['truth'],replay['truth_r'],replay['truth_x']) for m in METHODS]
    selector_calls=trace.get('ac_call_metrics',{}).get('selection_calls',[])
    for record in records:
        wanted=[1,2] if record['method']==METHODS[0] else [1]
        calls=[v for v in selector_calls if v['round'] in wanted]
        available=len(calls)==len(wanted)
        record['policy_selection_seconds']=sum(v['seconds'] for v in calls) if available else None
        record['policy_selection_AC_calls']=sum(v['ac_calls'] for v in calls) if available else None
    success=[first['success']]+[fits[m]['success'] for m in METHODS]
    score=contracts.pair_score(first['losses'],fits[METHODS[0]]['losses'],fits[METHODS[1]]['losses'],replay['truth'],success=success)
    arrays={}
    for m in METHODS:
        arrays.update({m+'_joint_'+k:v for k,v in observation_arrays(replay['observations'][m]).items()})
        arrays[m+'_actual_u_kw']=np.asarray(replay['plans']['rolling' if m==METHODS[0] else 'openloop']['u_kw'])
        for name,state in fits[m]['latent'].items():
            arrays.update({m+'__'+name+'__fit_'+k:v for k,v in state.items()})
    paired_archive=out/f'trial_{seed}_paired_observations.npz';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(paired_archive,**arrays)
    row=dict(seed=seed,history_id=str(seed),policy_id=manifest['policy_id'],population=manifest['population'],
        case=trace['case'],truth_topology=replay['truth'],true_index=replay['truth'],candidate_names=list(contracts.FAMILIES),
        independent_history_unit=True,first_bank=first,records=records,score_record=asdict(score),
        fresh_trace=provenance,paired_archive_sha256=sha(paired_archive),replay_checks=replay['replay'],
        operational_checks=replay['original_operations'],operation_costs=replay['operations'],
        common_fit_seed=common_seed,fit_order=order,common_initial_R_X=initial,
        fitted_latent_initial_rule='bias-corrected each-method total meters and root-master residual; no warm states',
        actual_actions_not_modified=True,new_selection_performed_after_fresh_trace=False,
        original_trace_selection_costs=trace['selection'],original_trace_seconds=trace['seconds'],
        original_trace_AC_metrics=trace['ac_call_metrics'],source_sha256=manifest['source_sha256'],
        original_shadow_final_fit_is_archive_overhead_not_paired_fit=True,
        seconds=perf_counter()-began,paired_trace_failure=False,global_MLE_certified=False)
    return verify_finished(row,out,manifest,project)


def run_batch(phase,out,manifest_file,project=None,now=None):
    out=ensure_staging(out);project=project or RealProject();manifest=read(manifest_file);verify_manifest(manifest,project)
    now=now or (lambda:datetime.now(timezone.utc));schedule=manifest['schedule']
    if phase=='pilot':seeds=schedule['pilot_seeds']
    elif phase in ['calibration','evaluation']:
        pilot=read(out.parent/'pilot'/'summary.json')
        if not pilot.get('formal_schedule_eligible'):raise ValueError('pilot did not qualify frozen formal schedule')
        if pilot['policy_id']!=manifest['policy_id']:raise ValueError('pilot belongs to another wrapper freeze')
        n=schedule[phase+'_count'];start=schedule[phase+'_start_seed'];seeds=list(range(start,start+n))
    else:raise ValueError('phase must be pilot, calibration or evaluation')
    if len(seeds)!=len(set(seeds)):raise ValueError('duplicate history seed')
    out.mkdir(parents=True,exist_ok=True)
    header=dict(phase=phase,seeds=seeds,manifest=manifest,manifest_sha256=sha(manifest_file))
    cfg=out/'config.json'
    if cfg.exists() and read(cfg)!=finite(header):raise ValueError('changed paired batch on resume')
    dump(cfg,header);rows=[];stop='count_complete'
    cutoff=datetime.fromisoformat(schedule['no_new_history_after_utc'])
    for seed in seeds:
        saved=out/f'trial_{seed}.json'
        if saved.exists():
            row=read(saved)
            if row['policy_id']!=manifest['policy_id']:raise ValueError('saved paired history differs from freeze')
            if row.get('source_sha256')!=manifest['source_sha256']:raise ValueError('saved paired source differs from freeze')
            provenance=row.get('fresh_trace',{})
            if provenance.get('trace_sha256') and sha(provenance['trace_json'])!=provenance['trace_sha256']:
                raise ValueError('fresh trace changed on resume')
            if provenance.get('archive_sha256') and sha(provenance['archive'])!=provenance['archive_sha256']:
                raise ValueError('fresh archive changed on resume')
            if row.get('paired_archive_sha256') and sha(out/f'trial_{seed}_paired_observations.npz')!=row['paired_archive_sha256']:
                raise ValueError('paired fit archive changed on resume')
        else:
            if now()>=cutoff:stop='wall_clock_budget_checkpoint';break
            verify_manifest(manifest,project)
            dump(out/'progress.json',dict(state='history_running',completed=len(rows),seed=seed,requested=len(seeds)))
            try:row=run_history(seed,phase,out,manifest,project)
            except ReplayMismatch as error:
                dump(out/f'replay_mismatch_{seed}.json',dict(seed=seed,error=str(error),state='contract_checkpoint'))
                stop='replay_contract_checkpoint';break
            except ValueError as error:
                dump(out/f'contract_failure_{seed}.json',dict(seed=seed,error=str(error),state='contract_checkpoint'))
                stop='source_or_contract_checkpoint';break
            dump(saved,row)
        rows.append(row)
        dump(out/'progress.json',dict(state='history_recorded',completed=len(rows),requested=len(seeds),last_seed=seed))
    seconds=[r.get('seconds',0.) for r in rows];mean=float(np.mean(seconds)) if seconds else None
    summary=dict(phase=phase,policy_id=manifest['policy_id'],history_count=len(rows),requested_history_count=len(seeds),
        all_requested_histories_attempted=len(rows)==len(seeds),stop_reason=stop,mean_seconds=mean,
        complete_history_count=sum(r['score_record']['complete'] for r in rows),
        paired_trace_failure_count=sum(r.get('paired_trace_failure',False) for r in rows),
        formal_schedule_eligible=phase=='pilot' and len(rows)==2 and mean<=45 and all(r['score_record']['complete']
            and all(z.get('operational_voltage_ok') is True for z in r['records']) for r in rows),
        pilot_is_not_formal_sample=True,expansion_justified=False)
    dump(out/'summary.json',summary)
    dump(out/'progress.json',dict(state='count_complete' if len(rows)==len(seeds) else 'checkpoint',**summary))
    return summary


def cp(k,n,confidence=.95):
    tail=(1-confidence)/2
    return [0. if k==0 else float(beta.ppf(tail,k,n-k+1)),1. if k==n else float(beta.ppf(1-tail,k+1,n-k))] if n else [0.,1.]
def event(values):
    n=len(values);k=sum(values)
    return dict(count=k,history_count=n,rate=k/n if n else None,CP95=cp(k,n),unit='independent_history')
def gate(bank,thresholds):
    loss=np.asarray([np.nan if v is None else v for v in bank['losses']],float)
    ok=bool(np.all(bank['success']) and np.isfinite(loss).all() and (loss>=0).all())
    rel=(loss-loss.min()>thresholds['relative']) if ok else np.zeros(4,bool)
    absolute=(loss>thresholds['absolute']) if ok else np.zeros(4,bool)
    return dict(complete=ok,relative=rel.tolist(),absolute=absolute.tolist(),combined=(rel|absolute).tolist())
def evaluation_events(row,thresholds):
    first=gate(row['first_bank'],thresholds);t=row['true_index'];out={}
    covered=isinstance(t,int) and not isinstance(t,bool) and 0<=t<4
    for record in row['records']:
        final=gate(record,thresholds);retained=~np.asarray(final['combined']);physical=record.get('operational_voltage_ok',False)
        unique=bool(covered and final['complete'] and first['complete'] and physical and retained.sum()==1 and retained[t])
        out[record['method']]=dict(first_gate=first,final_gate=final,
            false_rejection=bool(covered and (first['combined'][t] or final['combined'][t])),truth_omission=not covered,
            final_unique_correct=unique,algorithm_unique_correct=unique and not record.get('selection_failure',False),
            unresolved=bool(retained.sum()>1),empty=bool(retained.sum()==0),
            wrong_unique=bool(retained.sum()==1 and (not covered or not retained[t])),
            fit_failure=not(first['complete'] and final['complete']),physical_failure=physical is False,
            physical_evidence_unavailable=physical is None,
            selection_evidence_unavailable=record.get('selection_failure') is None,
            selection_failure=bool(record.get('selection_failure',False)))
    return out


def replay_saved_objective(folder,row,record,fit):
    """Saved-state Gaussian evaluation only; no optimizer or generator."""
    sys.path.insert(0,str(ROOT))
    from research_experiments.mobile_ev_mle.likelihood import ac_predict
    from research_experiments.mobile_ev_mle.candidates import RadialCandidate
    from research_experiments.mobile_ev_mle.simulation import model
    path=Path(folder)/f"trial_{row['seed']}_paired_observations.npz";m=record['method']
    with np.load(path,allow_pickle=False) as archive:
        obs={k:np.array(archive[m+'_joint_'+k]) for k in CHANNELS}
        states={k:np.array(archive[m+'__'+fit['name']+'__fit_'+k]) for k in ['p','q','source_p','source_q']}
    cand=RadialCandidate.from_supports(fit['name'],3,fit['supports']);meter=model(3)
    physical=ac_predict(cand,fit['r_pu'],fit['x_pu'],states['p'],states['q'],states['source_p'],states['source_q'],obs['root_v'])
    residuals=[];normalizer=0.
    for key in ['p','q','v','master_p','master_q']:
        bias=getattr(meter,'bias_'+key) if key in ['p','q','v'] else 0.
        sigma=np.broadcast_to(getattr(meter,key+'_sigma'),obs[key].shape)
        residuals.append(((physical[key]+bias-obs[key])/sigma).ravel())
        normalizer+=float(np.log(2*np.pi*sigma**2).sum())
    residual=np.concatenate(residuals);loss=float(residual@residual);loglik=-.5*(loss+normalizer)
    delta=loss-fit['loss'];ldelta=loglik-fit['loglik']
    valid=abs(delta)<=1e-6*max(1.,loss) and abs(ldelta)<=1e-6*max(1.,abs(loglik))
    return dict(history_id=row['history_id'],method=m,candidate=fit['name'],
                status='verified' if valid else 'mismatch',loss_difference=delta,loglik_difference=ldelta,
                global_optimum_certified=False)


def collect(calibration,evaluation,out,*,replay=True):
    out=ensure_staging(out)
    cal,ev=Path(calibration),Path(evaluation);a,b=read(cal/'config.json'),read(ev/'config.json')
    if a['manifest']!=b['manifest']:raise ValueError('paired calibration/evaluation freeze differs')
    if a['manifest']['source_sha256']!=runner_hashes():raise ValueError('paired collector source differs from frozen runner')
    if a['phase']!='calibration' or b['phase']!='evaluation':raise ValueError('incorrect paired phases')
    for folder,cfg in [(cal,a),(ev,b)]:
        if not read(folder/'summary.json')['all_requested_histories_attempted']:raise ValueError('incomplete predeclared paired batch')
        if {p.stem for p in folder.glob('trial_[0-9]*.json')}!={f'trial_{seed}' for seed in cfg['seeds']}:
            raise ValueError('paired batch seed/count contract differs')
    if set(a['seeds'])&set(b['seeds']):raise ValueError('paired calibration/evaluation overlaps')
    cr=[read(cal/f'trial_{s}.json') for s in a['seeds']];er=[read(ev/f'trial_{s}.json') for s in b['seeds']]
    recomputed=[]
    for r in cr+er:
        if r.get('policy_id')!=a['manifest']['policy_id'] or not r.get('independent_history_unit'):
            raise ValueError('paired history is not from this independent frozen policy')
        if r.get('candidate_names')!=list(contracts.FAMILIES) or [z['method'] for z in r['records']]!=list(METHODS):
            raise ValueError('paired candidate/method axis changed')
        banks=[r['first_bank']]+r['records']
        losses=[[np.inf if v is None else v for v in z['losses']] for z in banks]
        fresh=contracts.pair_score(*losses,r['true_index'],success=[z['success'] for z in banks])
        if fresh.complete!=r['score_record']['complete']:raise ValueError('stored paired score complete flag differs')
        if fresh.complete and any(abs(getattr(fresh,k)-r['score_record'][k])>1e-10 for k in ['relative','absolute']):
            raise ValueError('stored paired score differs from saved losses')
        recomputed.append(fresh)
    q={k:contracts.upper_quantile([getattr(r,k) for r in recomputed[:len(cr)]],.05) for k in ['relative','absolute']}
    thresholds={k:v['threshold'] for k,v in q.items()};evaluated=[evaluation_events(r,thresholds) for r in er]
    metrics={m:{key:event([z[m][key] for z in evaluated]) for key in ['false_rejection','algorithm_unique_correct','unresolved','empty','wrong_unique','fit_failure','physical_failure','physical_evidence_unavailable','selection_failure','selection_evidence_unavailable','truth_omission']} for m in METHODS}
    pairs={}
    for key in ['algorithm_unique_correct','false_rejection','unresolved']:
        plus=sum(z[METHODS[0]][key] and not z[METHODS[1]][key] for z in evaluated)
        minus=sum(z[METHODS[1]][key] and not z[METHODS[0]][key] for z in evaluated);n=len(er)
        hi=cp(plus,n,.975);lo=cp(minus,n,.975)
        pairs[key]=dict(effect=(plus-minus)/n,event_0_to_1=plus,event_1_to_0=minus,tied=n-plus-minus,
            conservative_exact95=[max(-1.,hi[0]-lo[1]),min(1.,hi[1]-lo[0])],
            exact_McNemar_pvalue=1. if plus+minus==0 else float(binomtest(plus,plus+minus,.5).pvalue),history_count=n,
            event_direction='positive change is worse' if key in ['false_rejection','unresolved'] else 'positive change is better')
        if key=='algorithm_unique_correct':pairs[key].update(gain_event_count=plus,loss_event_count=minus)
    continuous={}
    for key in ['wrong_margin','impedance_relative_rmse','seconds','ac_calls','policy_selection_seconds','policy_selection_AC_calls']:
        values=[]
        for r in er:
            by={z['method']:z for z in r['records']};x,y=by[METHODS[0]].get(key),by[METHODS[1]].get(key)
            if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):values.append(x-y)
        continuous[key]=dict(mean_paired_difference=float(np.mean(values)) if values else None,
            paired_standard_error=float(np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else None,
            available_pair_count=len(values),missing_pair_count=len(er)-len(values),full_history_count=len(er),
            descriptive_available_subset=True)
    replay_results=[]
    if replay:
        for row in er:
            for record in row['records']:
                for fit in record.get('fits',[]):
                    replay_results.append(replay_saved_objective(ev,row,record,fit))
    result=dict(manifest=a['manifest'],calibration=q,calibration_failure_count=sum(not r['score_record']['complete'] for r in cr),
        combined_rank_tail_bound=sum(v['rank_tail_bound'] for v in q.values()),methods=metrics,paired=pairs,
        continuous_paired=continuous,whole_pair_false_rejection=event([any(z[m]['false_rejection'] for m in METHODS) for z in evaluated]),
        scope='fresh frozen numerical-policy declared-population marginal rank gate',conditional_alpha_spending=False,
        global_MLE_claim=False,C_or_original_D_threshold_reused=False,pilot_in_denominators=False,
        primary_endpoint='algorithm_unique_correct',secondary_intervals_descriptive=True,
        saved_objective_replay=dict(requested=replay,available_fit_count=len(replay_results),
            verified_count=sum(z['status']=='verified' for z in replay_results),
            mismatch_count=sum(z['status']=='mismatch' for z in replay_results),not_global_MLE_certificate=True),
        calibration_input_hashes={p.name:sha(p) for p in sorted(cal.glob('trial_[0-9]*.json'))},
        evaluation_input_hashes={p.name:sha(p) for p in sorted(ev.glob('trial_[0-9]*.json'))})
    dump(Path(out)/'paired_summary.json',result);dump(Path(out)/'history_gates.json',evaluated)
    dump(Path(out)/'saved_objective_replay.json',replay_results)
    if any(z['status']=='mismatch' for z in replay_results):raise ValueError('paired saved objective replay mismatch')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path);p.add_argument('--manifest',type=Path)
    p.add_argument('--phase',choices=['pilot','calibration','evaluation']);p.add_argument('--output',type=Path)
    p.add_argument('--collect',action='store_true');p.add_argument('--calibration',type=Path);p.add_argument('--evaluation',type=Path)
    a=p.parse_args()
    if a.freeze:print(json.dumps(freeze(a.freeze)));return
    if a.collect:print(json.dumps(finite(collect(a.calibration,a.evaluation,a.output))));return
    if a.phase:
        if a.manifest is None or a.output is None:p.error('actual execution requires --manifest and --output')
        print(json.dumps(run_batch(a.phase,a.output,a.manifest)));return
    p.print_help()


if __name__=='__main__':main()
