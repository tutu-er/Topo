"""Two sanitized H-only pilots. Clone frozen search; only upper backend differs.

All outputs are new staging files. No original trial, truth, or future data read.
"""
from pathlib import Path
import copy,datetime,gzip,hashlib,inspect,json,marshal,platform,sys,time,traceback,types
import numpy as np
import scipy

sys.dont_write_bytecode=True
FROZEN=Path(r'C:\Users\23761\.codex\worktrees\36a1\Topo')
BASE=FROZEN/'research_experiments/mobile_ev_mle'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(FROZEN))
from research_experiments.mobile_ev_mle import ev_information_milp as old
from research_experiments.mobile_ev_mle import ev_operations_milp as operations
from research_experiments.mobile_ev_mle import shared_power_square_upper as shared
from research_experiments.mobile_ev_mle.adaptive_history import build_history_context,build_frozen_pairs,evaluate_frozen_pair_actual
from research_experiments.mobile_ev_mle.adaptive_milp_study import configurations
from research_experiments.mobile_ev_mle.likelihood import Observations
from research_experiments.mobile_ev_mle.simulation import model as measurement_model
from research_experiments.mobile_ev_mle.candidates import RadialCandidate

CASES=[('710009','continuous_eig_Honly',28101301),('720001','shared_upper_Honly',28101302)]
FIELDS=['p','q','v','master_p','master_q','root_v']
SUPPORTS={'clade_02':((0,2),),'clade_01':((0,1),),'star':(),'clade_12':((1,2),)}
SETTINGS={'max_iterations':3,'solver_seconds':3.,'total_seconds':25.,'segments':2,'max_segments':8,
          'relative_gap':.001,'absolute_gap':1e-4}
ORIGINAL_SEARCH=old.adaptive_search
ORIGINAL_UPPER=ORIGINAL_SEARCH.__globals__['upper_master']
ORIGINAL_SOLVE=operations.LinearMILP.solve

def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else '+inf' if x>0 else '-inf' if x<0 else 'nan'
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    return x
def encode(x):return json.dumps(clean(x),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def dump(path,x):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(encode(x)+b'\n')
def packed(path,x):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(gzip.compress(encode(x),mtime=0))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def code_sha(fn):return hashlib.sha256(marshal.dumps(fn.__code__)).hexdigest()
def progress(stage,**values):print(json.dumps({'stage':stage,**clean(values)},ensure_ascii=False),flush=True)
def function_id(fn):return {'module':fn.__module__,'qualname':fn.__qualname__,'code_sha256':code_sha(fn)}
def qpoint(pairs,u):
    rows=[]
    for p in pairs:
        r=old.solve_pair(p,u)
        rows.append({'pair':p.name,'u_kw':u,**{k:r[k] for k in ['success','status','message','z','q_lower','q_upper','numeric_gap','optimality','roundoff_guard','acceleration']}})
    valid=all(r['success'] for r in rows)
    return {'complete':valid,'q_lower':min(r['q_lower'] for r in rows) if valid else None,
            'q_upper':min(r['q_upper'] for r in rows) if valid else None,'records':rows}


def make_adapter(backend,qp_trace):
    g=ORIGINAL_SEARCH.__globals__.copy()
    def trace_qp(pair,u,*args,**kwargs):
        started=time.perf_counter()
        try:
            result=old.solve_pair(pair,u,*args,**kwargs)
            qp_trace.append({'pair':pair.name,'u_kw':np.asarray(u).copy(),'seconds':time.perf_counter()-started,
                             **{k:result[k] for k in ['success','status','message','z','q_lower','q_upper','numeric_gap','optimality','roundoff_guard','acceleration']}})
            return result
        except Exception as error:
            qp_trace.append({'pair':pair.name,'u_kw':np.asarray(u).copy(),'seconds':time.perf_counter()-started,
                             'exception':type(error).__name__,'message':str(error),'success':False})
            raise
    trace_qp.__wrapped__=old.solve_pair
    g['solve_pair']=trace_qp
    g['upper_master']=backend
    clone=types.FunctionType(ORIGINAL_SEARCH.__code__,g,ORIGINAL_SEARCH.__name__+'_upper_adapter',
                             ORIGINAL_SEARCH.__defaults__,ORIGINAL_SEARCH.__closure__)
    clone.__kwdefaults__=copy.deepcopy(ORIGINAL_SEARCH.__kwdefaults__)
    clone.__annotations__=ORIGINAL_SEARCH.__annotations__.copy()
    assert clone.__code__ is ORIGINAL_SEARCH.__code__ and clone.__kwdefaults__==ORIGINAL_SEARCH.__kwdefaults__
    assert ORIGINAL_SEARCH.__globals__['upper_master'] is ORIGINAL_UPPER
    diff=[k for k in ORIGINAL_SEARCH.__globals__ if g[k] is not ORIGINAL_SEARCH.__globals__[k]]
    assert set(diff)==({'solve_pair'} if backend is ORIGINAL_UPPER else {'solve_pair','upper_master'})
    return clone,{'same_code_object':True,'keyword_defaults_copied':True,'changed_globals':diff,
                  'common_pass_through_instrumentation':['solve_pair'],'unwrapped_qp_target':function_id(old.solve_pair),
                  'policy_backend_target':function_id(backend),'original_global_upper_unmodified':True}


def run_backend(case_dir,label,backend,config,pairs,initial_us,seed):
    target=case_dir/label;target.mkdir(parents=True,exist_ok=True)
    qp_trace=[];calls=[];clone,adapter_audit=make_adapter(backend,qp_trace)
    dump(target/'adapter_audit.json',adapter_audit)
    def tracked_solve(model,*args,**kwargs):
        objective=args[0] if args else kwargs.get('objective')
        objective={} if objective is None else dict(objective)
        assert kwargs.get('cmax',True)
        started=time.perf_counter()
        try:
            result=ORIGINAL_SOLVE(model,*args,**kwargs)
            calls.append({'model':model,'objective':objective,'solver_seconds':time.perf_counter()-started,
                          'configured_time_limit':kwargs.get('time_limit'),
                          'result':result,'class':('upper_shared' if 'shared_squared_score_scaled' in model.variables else
                              'upper_original' if 'squared_score_scaled' in model.variables else 'direction_lower' if 'distance_scaled' in model.variables else 'operations_seed')})
            return result
        except Exception as error:
            calls.append({'model':model,'objective':objective,'solver_seconds':time.perf_counter()-started,
                          'exception':type(error).__name__,'message':str(error),'class':'unknown_failed_model'})
            raise
    operations.LinearMILP.solve=tracked_solve
    started=time.perf_counter()
    try:
        result=clone(config,pairs,initial_us=initial_us,seed=seed,**SETTINGS)
    finally:
        operations.LinearMILP.solve=ORIGINAL_SOLVE
        assert ORIGINAL_SEARCH.__globals__['upper_master'] is ORIGINAL_UPPER
        assert old.upper_master is ORIGINAL_UPPER and operations.LinearMILP.solve is ORIGINAL_SOLVE
    external_seconds=time.perf_counter()-started
    # Serialize snapshots only after search; the common tracer did no file IO in its budget.
    solver_rows=[];incumbents={}
    for i,call in enumerate(calls):
        model=call['model'];r=call.get('result')
        info={'call':i,'class':call['class'],'solver_call_seconds':call['solver_seconds'],
              'configured_time_limit':call.get('configured_time_limit'),'n_variables':model.n_variables,'n_rows':len(model.rows)}
        if r is not None:
            info.update(status=int(r.status),success=bool(r.success),message=str(r.message),
                        objective_value=r.objective_value,objective_bound=r.objective_bound,
                        mip_gap=getattr(r,'mip_gap',None),raw_minimization_fun=getattr(r,'fun',None),
                        raw_minimization_mip_dual_bound=getattr(r,'mip_dual_bound',None))
            if r.x is not None:incumbents[f'model_{i}_x']=r.x.copy()
        else:info.update(exception=call['exception'],message=call['message'])
        snapshot,digest=shared.model_snapshot(model,call['objective'],{'backend':label,'call':i,'class':call['class'],
                                    'objective_direction':'maximize','objective_is_scaled':call['class']!='operations_seed'})
        info['snapshot_sha256']=digest;info['snapshot_file']=f'model_{i}_snapshot.json.gz'
        packed(target/info['snapshot_file'],snapshot);solver_rows.append(info)
    if incumbents:np.savez_compressed(target/'master_incumbents.npz',**incumbents)
    packed(target/'qp_trace.json.gz',qp_trace);dump(target/'solver_calls.json',solver_rows)
    plan=result['plan'];saved={k:v for k,v in result.items() if k!='plan'}
    saved['plan']=None if plan is None else plan.as_dict()
    saved['external_call_seconds']=external_seconds
    saved['raw_global_gap']=None if result.get('proxy_upper') is None else result['proxy_upper']-result['proxy_lower']
    saved['original_absolute_gap_is_clipped_by_frozen_code']=True
    saved['policy_source_unchanged']=True
    dump(target/'search_result.json',saved)
    return result,saved,qp_trace,solver_rows


def audit_selected(case_dir,label,result,context,raw,pairs,config,baselines):
    target=case_dir/label;plan=result['plan'];rows=[]
    if plan is None:
        summary={'available':False,'reason':'no operations-validated selected plan'};dump(target/'selected_audit.json',summary);return summary
    u=plan.u_kw;physical=operations.check_plan(plan,config,tolerance=2e-6)
    raw_controls={'u_kw':u,'minimum':float(u.min()),'maximum':float(u.max()),
                  'strict_negative_count':int(np.count_nonzero(u<0)),
                  'tiny_nonzero_abs_below_2e-6':int(np.count_nonzero((u!=0)&(abs(u)<2e-6))),
                  'strict_multi_site_slot_count':int(np.count_nonzero(np.count_nonzero(u!=0,axis=1)>1)),
                  'above_tol_multi_site_slot_count':int(np.count_nonzero(np.count_nonzero(abs(u)>2e-6,axis=1)>1)),
                  'clipping_or_rounding_performed':False}
    qp= qpoint(pairs,u)
    dump(target/'selected_point_qp.json',qp)
    for p,original,r in zip(pairs,raw,qp['records']):
        entry={'pair':p.name,'profile_q_lower':r['q_lower'],'profile_q_upper':r['q_upper'],
               'profile_numeric_gap':r['numeric_gap'],'qp_success':r['success']}
        try:
            actual=evaluate_frozen_pair_actual(context,original,u,np.asarray(r['z']))
            entry.update({k:v for k,v in actual.items() if k not in ['actual_residual','affine_residual']})
            cost=actual['actual_soft_cost']
            entry['history_share']=sum(actual['actual_history_costs'])/cost if cost>0 else None
            entry['future_share']=actual['actual_future_distance_squared']/cost if cost>0 else None
            entry['is_feasible_nonlinear_infimum_upper_witness']=bool(actual['physically_feasible'])
            entry['is_nonlinear_design_bound']=False
        except Exception as error:
            entry.update(rollback_exception=type(error).__name__,rollback_error=str(error),
                         is_feasible_nonlinear_infimum_upper_witness=False,is_nonlinear_design_bound=False)
        rows.append(entry)
    dump(target/'full_AC_rollback.json',rows)
    intervals={}
    if qp['complete']:
        for name,baseline in baselines.items():
            if baseline['complete']:intervals[name]=[qp['q_lower']-baseline['q_upper'],qp['q_upper']-baseline['q_lower']]
    active=min(rows,key=lambda r:r['profile_q_lower'])
    tol=1e-4+.001*max(1,abs(result['proxy_lower']))
    summary={'available':True,'physical_EV_plan':physical,'raw_controls':raw_controls,
             'selected_profile_complete':qp['complete'],'selected_q_lower':qp['q_lower'],'selected_q_upper':qp['q_upper'],
             'reported_global_LB_eligible_under_declared_operation_tolerance':bool(physical['feasible'] and result['success']),
             'strict_single_site_shared_decomposition_condition':raw_controls['strict_multi_site_slot_count']==0,
             'raw_negative_controls_can_make_AC_rollback_unavailable':raw_controls['strict_negative_count']>0,
             'gain_intervals_vs_initial_actions':intervals,'absolute_stop_tolerance':tol,
             'relative_stop_uses_history_including_total_proxy_LB':True,
             'active_pair_by_profile_lower':active,'all_rollback_pairs':len(rows),
             'physically_feasible_rollback_pairs':sum(r.get('physically_feasible',False) for r in rows),
             'rollback_unavailable_pairs':sum('rollback_exception' in r for r in rows),
             'nonlinear_infimum_computed':False,'actual_topology_elimination_evaluated':False}
    dump(target/'selected_audit.json',summary);return summary


def read_case(ident,folder):
    jp=BASE/'outputs'/folder/f'H_{ident}.json';npz=jp.with_suffix('.npz')
    m=json.loads(jp.read_text(encoding='utf-8-sig'))
    keys=['history_'+k for k in FIELDS] if ident=='710009' else FIELDS
    if ident=='710009':assert m['does_not_contain_truth_or_future'] and m['allowed_npz_keys']==keys
    else:assert not m['actual_future_or_truth_passed'] and m['allowed_array_keys']==keys and m['known_history_EV_kw']==0
    with np.load(npz,allow_pickle=False) as data:
        assert set(data.files)==set(keys)
        arrays={field:data[key].copy() for field,key in zip(FIELDS,keys)}
    return Observations(**arrays),{'json_path':str(jp),'npz_path':str(npz),'json_sha256':sha(jp),'npz_sha256':sha(npz),
                                 'array_keys_used':keys,'metadata_values_not_used':['old centers','old actions','original trial'],
                                 'actual_future_or_truth_used':False}


def main():
    frozen=json.loads((BASE/'outputs/adaptive_milp/frozen_C_execution.json').read_text(encoding='utf-8-sig'))
    initial_hashes={name:sha((BASE/name).resolve()) for name in frozen['source_sha256']}
    assert initial_hashes==frozen['source_sha256']
    _,config,_=configurations()
    inputs=[]
    for ident,folder,seed in CASES:
        _,audit=read_case(ident,folder);inputs.append({'history_id':ident,'context_seed':seed,**audit})
    protocol={'scope':'two independent sanitized H-only selector pilots; descriptive, no topology outcome evaluation',
              'cases':inputs,'context':{'max_centers':2,'max_attempts_per_candidate':3,'multistart':2,'max_nfev':300,
                 'line_trust_fraction':.30,'source_delta_bound':.05,'center_loss_slack':12.,'horizon':8,'forecast':'mean',
                 'centers_insufficient_policy':'retain actual finite pair count; never manufacture or delete centers'},
              'candidate_bank':list(SUPPORTS),'search_settings':SETTINGS,'operation_config':vars(config),
              'initial_actions':['constant_site0_3kW_8slots','none'],'original_action_library_passed':False,
              'adapter':'types.FunctionType same __code__/defaults/kwdefaults, copied globals, only upper backend policy differs',
              'common_instrumentation':'pass-through QP and MILP tracing; snapshots and all file IO occur after each search budget',
              'acceptance':['sanitized six-array contracts match','both variants share identical reconstructed pair fingerprints',
                  'actual B exact time-block support passes, or new backend incompatibility preserved',
                  'old function global upper/master solve restored after every run','all raw negative/simultaneous/tiny powers retained',
                  'infeasible selected action never reported as certified lower bound','all QP primal/tangent gaps and failures retained',
                  'all pair H-only AC rollback feasibility/error/history-future shares retained',
                  'no upper incumbent promoted to proxy benefit LB','core hash equality before/after','offline saved evidence replay'],
              'prohibited':['truth or actual future/noise reads','eleven-action library','C/D result overwrite','tiny coefficient clipping',
                            'nonlinear design bound claims','topology elimination or correctness probability claims','frozen source edits'],
              'runtime_scope':'same machine, fixed two histories, old then shared; not a performance generalization'}
    dump(OUT/'protocol.json',protocol)
    dump(OUT/'code_and_input_hashes_before.json',{'frozen_core':initial_hashes,'new_shared_module':sha(BASE/'shared_power_square_upper.py'),
         'pilot_driver':sha(Path(__file__)),'original_search_function':function_id(ORIGINAL_SEARCH),
         'original_search_source_text_sha256':hashlib.sha256(inspect.getsource(ORIGINAL_SEARCH).encode()).hexdigest(),
         'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},'inputs':inputs})
    progress('protocol_and_acceptance_saved')
    outcomes=[]
    for ident,folder,seed in CASES:
        case_dir=OUT/f'H_{ident}';case_dir.mkdir(parents=True,exist_ok=True)
        try:
            history,input_audit=read_case(ident,folder)
            candidates=[RadialCandidate.from_supports(name,3,support) for name,support in SUPPORTS.items()]
            start=time.perf_counter()
            context=build_history_context(history,candidates,measurement_model(3),horizon=8,max_centers=2,max_attempts_per_candidate=3,
                     multistart=2,max_nfev=300,line_trust_fraction=.30,source_delta_bound=.05,center_loss_slack=12.,seed=seed,
                     forecast_support=[dict(name='mean',baseline='mean',source_baseline='mean')])
            dump(case_dir/'context.json',{'seconds':time.perf_counter()-start,'summary':context['summary'],'attempts':context['attempts']})
            if context['status']!='ready':raise RuntimeError('history context not ready; failure preserved, no candidate removed')
            constant=np.zeros((8,3));constant[:,0]=3.;none=np.zeros_like(constant)
            raw=build_frozen_pairs(context,constant,source_mode='local_refit');pairs=[old.FrozenPair(**r) for r in raw]
            supports=[];compatible=True
            for p in pairs:
                try:a=shared.certify_time_block_support(p.B,8,3);supports.append({'pair':p.name,'passed':True,'zero_rows':a['zero_rows'],'row_times':a['row_times'],'d':a['d']})
                except ValueError as error:compatible=False;supports.append({'pair':p.name,'passed':False,'error':str(error)})
            dump(case_dir/'exact_support.json',{'all_pass':compatible,'pairs':supports,'no_near_zero_deleted':True})
            arrays={}
            for i,p in enumerate(pairs):
                for key in ['c','B','A','F','lower','upper','scale']:arrays[f'pair_{i}_{key}']=getattr(p,key)
            np.savez_compressed(case_dir/'frozen_pairs.npz',**arrays)
            fingerprints={p.name:p.fingerprint() for p in pairs}
            dump(case_dir/'pair_manifest.json',{'pair_count':len(pairs),'names':[p.name for p in pairs],'fingerprints':fingerprints,
                 'archive_sha256':sha(case_dir/'frozen_pairs.npz'),'source_input':input_audit,'raw_pair_metadata':[{k:v for k,v in r.items() if k=='metadata'} for r in raw]})
            baselines={'constant':qpoint(pairs,constant),'none':qpoint(pairs,none)}
            packed(case_dir/'initial_action_QP.json.gz',baselines)
            progress('case_ready',history=ident,pairs=len(pairs),exact_support=compatible)
            case_results={'history_id':ident,'pair_count':len(pairs),'source_input':input_audit,'exact_support_pass':compatible,'backends':{}}
            for label,backend in [('original',ORIGINAL_UPPER),('shared',shared.upper_master_shared)]:
                if label=='shared' and not compatible:
                    case_results['backends'][label]={'failure':'exact support incompatible; compression refused'};continue
                result,saved,qp_trace,solvers=run_backend(case_dir,label,backend,config,pairs,[constant,none],seed)
                assert {p.name:p.fingerprint() for p in pairs}==fingerprints
                audit=audit_selected(case_dir,label,result,context,raw,pairs,config,baselines)
                case_results['backends'][label]={'search':{k:v for k,v in saved.items() if k not in ['records','plan','solver_records']},
                    'saved_solver_records':saved.get('solver_records'),'milp_calls':solvers,'QP_call_count':len(qp_trace),'selected_audit':audit}
                progress('backend_complete',history=ident,backend=label,status=result['status'],LB=result.get('proxy_lower'),UB=result.get('proxy_upper'),
                         seconds=saved['external_call_seconds'],qp_calls=len(qp_trace),rollback_feasible=audit.get('physically_feasible_rollback_pairs'))
            dump(case_dir/'case_summary.json',case_results);outcomes.append(case_results)
        except Exception as error:
            failure={'history_id':ident,'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),
                     'preserved':True,'no_pairs_or_coefficients_silently_removed':True}
            dump(case_dir/'pilot_failure.json',failure);outcomes.append(failure);progress('case_failure',**failure)
    after={name:sha((BASE/name).resolve()) for name in initial_hashes}
    assert after==initial_hashes and ORIGINAL_SEARCH.__globals__['upper_master'] is ORIGINAL_UPPER
    assert operations.LinearMILP.solve is ORIGINAL_SOLVE
    dump(OUT/'code_hashes_after.json',{'frozen_core':after,'unchanged':True,'original_global_upper_restored':True,
        'original_MILP_solve_restored':True,'new_shared_module':sha(BASE/'shared_power_square_upper.py')})
    dump(OUT/'pilot_summary.json',{'cases':outcomes,'core_sources_unchanged':True,'actual_future_or_truth_used':False,
          'topology_outcome_evaluated':False,'all_failures_preserved':True,'done_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    progress('pilot_complete',cases=len(outcomes),output=str(OUT))

if __name__=='__main__':main()
