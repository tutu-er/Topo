"""Authorized H-only 710009 one-round old/new upper comparison. No future/truth reads."""
from pathlib import Path
import datetime,gzip,hashlib,json,platform,sys,time,traceback
import numpy as np
import scipy

sys.dont_write_bytecode=True
FROZEN=Path(r'C:\Users\23761\.codex\worktrees\36a1\Topo')
BASE=FROZEN/'research_experiments/mobile_ev_mle'
SANITIZED=BASE/'outputs/continuous_eig_Honly'
OUT=Path(__file__).resolve().parent
PROTO=OUT.parent
sys.path.insert(0,str(FROZEN));sys.path.insert(0,str(PROTO))
import shared_power_square_upper as shared
from research_experiments.mobile_ev_mle import ev_information_milp as old
from research_experiments.mobile_ev_mle import ev_operations_milp as operations
from research_experiments.mobile_ev_mle.adaptive_history import build_history_context,build_frozen_pairs
from research_experiments.mobile_ev_mle.likelihood import Observations
from research_experiments.mobile_ev_mle.simulation import model as measurement_model
from research_experiments.mobile_ev_mle.candidates import RadialCandidate

def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else '+inf' if x>0 else '-inf' if x<0 else 'nan'
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    return x
def encoded(x):return json.dumps(clean(x),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def dump(name,x):
    with (OUT/name).open('xb') as f:f.write(encoded(x)+b'\n')
def packed(name,x):
    with (OUT/name).open('xb') as f:f.write(gzip.compress(encoded(x),mtime=0))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def progress(stage,**items):print(json.dumps({'stage':stage,**items},ensure_ascii=False),flush=True)
def point(pairs,u,label):
    records=[];began=time.perf_counter()
    for p in pairs:
        began_one=time.perf_counter();r=old.solve_pair(p,u)
        records.append({'pair':p.name,'seconds':time.perf_counter()-began_one,
                        **{k:r[k] for k in ['success','status','message','z','q_lower','q_upper','numeric_gap','optimality','roundoff_guard','acceleration']}})
    complete=all(r['success'] for r in records)
    return {'label':label,'u_kw':u,'complete':complete,'seconds':time.perf_counter()-began,
            'q_lower':min(r['q_lower'] for r in records) if complete else None,
            'q_upper':min(r['q_upper'] for r in records) if complete else None,'pair_results':records}
def result_record(out,total,solve_entry,solve_seconds):
    r=out['result']
    return {'status':int(r.status),'success':bool(r.success),'message':str(r.message),
            'incumbent':out['q_incumbent'],'global_best_bound':out['q_proxy_upper'],
            'mip_gap':getattr(r,'mip_gap',None),'mip_node_count':getattr(r,'mip_node_count',None),
            'raw_minimization_fun':getattr(r,'fun',None),'raw_minimization_mip_dual_bound':getattr(r,'mip_dual_bound',None),
            'model_variables':out['n_variables'],'model_rows':out['n_rows'],'scale':out['scale'],
            'call_seconds':total,'build_before_solve_seconds':solve_entry,'solver_call_seconds':solve_seconds,
            'post_solve_seconds':total-solve_entry-solve_seconds,'solver_budget_seconds':3,
            'incumbent_is_proxy_benefit_lower_bound':False,'global_MLE_or_nonlinear_certificate':False}

phase='protocol'
try:
    metadata_path=SANITIZED/'H_710009.json';npz_path=SANITIZED/'H_710009.npz'
    allowed=['history_p','history_q','history_v','history_master_p','history_master_q','history_root_v']
    metadata=json.loads(metadata_path.read_text(encoding='utf-8-sig'))
    assert metadata['does_not_contain_truth_or_future'] and metadata['allowed_npz_keys']==allowed
    frozen=json.loads((BASE/'outputs/adaptive_milp/frozen_C_execution.json').read_text(encoding='utf-8-sig'))
    frozen_hashes={name:sha((BASE/name).resolve()) for name in frozen['source_sha256']}
    assert frozen_hashes==frozen['source_sha256']
    config={'history_id':'710009','fresh_reconstruction_seed':28101301,'horizon':8,'candidate_order':list(metadata['centres']),
            'max_centers':2,'max_attempts_per_candidate':3,'multistart':2,'max_nfev':300,
            'line_trust_fraction':.30,'source_delta_bound':.05,'center_loss_slack':12.,
            'forecast':'mean','constant_action_kw':3.,'constant_site':0,'segments':2,'solver_seconds_per_upper':3.,
            'witness_policy':'one convex-QP feasible z per pair at constant-site-0 action',
            'initial_diagnostic_actions':['constant_site_0_3kW','none'],'expected_pair_count':24,
            'old_upper_first_then_shared_upper':True,'whole_search':False,'actual_future':False,'action_library_read':False}
    protocol={'scope':'one authorized H-only single upper round; shared identical reconstructed pair/witness/config',
              'source_json_allowed_used_keys':['history_id','centres candidate names only','allowed_npz_keys','does_not_contain_truth_or_future'],
              'npz_allowed_used_arrays':allowed,'fresh_context_uses_existing_centres_as_initializers':False,
              'runtime_instrumentation':'temporarily wrap operation_model/LinearMILP.solve only to snapshot and time; restore in finally; source files unchanged',
              'bound_semantics':'maximization global best bound only is a full frozen-proxy upper; incumbent is not a benefit lower',
              'numerical_failures_preserved':True,'tiny_cross_time_coefficients_never_deleted':True,
              'comparison_is_descriptive_single_case_same_machine':True,
              'authorization_limit':'no new selector search, no second H, no actual future or original eleven-action library'}
    dump('protocol.json',protocol);dump('config.json',config)
    dump('source_hashes_before.json',{'sanitized_json_sha256':sha(metadata_path),'sanitized_npz_sha256':sha(npz_path),
                                    'frozen_core_sha256':frozen_hashes,
                                    'new_module_sha256':sha(PROTO/'shared_power_square_upper.py'),'driver_sha256':sha(Path(__file__)),
                                    'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__})
    progress('protocol_saved')
    phase='history_context'
    with np.load(npz_path,allow_pickle=False) as archive:
        assert archive.files==allowed
        arrays={name:archive[name].copy() for name in allowed}
    history=Observations(**{name.removeprefix('history_'):value for name,value in arrays.items()})
    n=history.n;assert n==3 and history.t==12
    supports={'star':(), 'clade_01':((0,1),),'clade_02':((0,2),),'clade_12':((1,2),)}
    assert set(config['candidate_order'])==set(supports)
    candidates=[RadialCandidate.from_supports(name,n,supports[name]) for name in config['candidate_order']]
    meter=measurement_model(3)
    began=time.perf_counter()
    context=build_history_context(history,candidates,meter,horizon=8,max_centers=2,max_attempts_per_candidate=3,
             multistart=2,max_nfev=300,line_trust_fraction=.30,source_delta_bound=.05,center_loss_slack=12.,
             seed=28101301,forecast_support=[dict(name='mean',baseline='mean',source_baseline='mean')])
    context_seconds=time.perf_counter()-began
    dump('context_summary.json',{'seconds':context_seconds,'summary':context['summary'],'attempts':context['attempts']})
    if context['status']!='ready':raise RuntimeError('fresh history context not ready; no failed candidate removed')
    constant=np.zeros((8,3));constant[:,0]=3.
    began=time.perf_counter();raw=build_frozen_pairs(context,constant,source_mode='local_refit')
    pairs=[old.FrozenPair(**row) for row in raw];freeze_seconds=time.perf_counter()-began
    if len(pairs)!=24:raise RuntimeError('fresh reconstruction produced '+str(len(pairs))+' pairs, expected 24; preserved')
    phase='exact_support'
    support=[]
    for p in pairs:
        a=shared.certify_time_block_support(p.B,8,3)
        support.append({'pair':p.name,'B_shape':p.B.shape,'zero_rows':a['zero_rows'],'row_times':a['row_times'],
                        'cross_time_exact_row_support_pass':True,'d':a['d']})
    dump('exact_support_audit.json',{'pair_count':24,'all_exact_support_pass':True,'near_zero_removed':False,'pairs':support})
    progress('context_and_support_ready',context_seconds=context_seconds,freeze_seconds=freeze_seconds,pairs=len(pairs))
    phase='constant_witness_qp'
    constant_point=point(pairs,constant,'constant_site_0_3kW');dump('constant_qp.json',constant_point)
    if not constant_point['complete']:raise RuntimeError('constant witness QP failed; all pair failures preserved')
    witnesses=[[np.asarray(r['z'],float)] for r in constant_point['pair_results']]
    none_point=point(pairs,np.zeros_like(constant),'none');dump('none_qp.json',none_point)
    # Persist the exact same reconstructed matrices and witness vectors used by both methods.
    npz_arrays={}
    for k,(p,group) in enumerate(zip(pairs,witnesses)):
        for key in ['c','B','A','F','lower','upper','scale']:npz_arrays[f'pair_{k}_{key}']=getattr(p,key)
        npz_arrays[f'pair_{k}_witness_0']=group[0]
    np.savez_compressed(OUT/'reconstructed_pairs_and_witnesses.npz',**npz_arrays)
    dump('pair_manifest.json',{'names':[p.name for p in pairs],'fingerprints':{p.name:p.fingerprint() for p in pairs},
                              'archive_sha256':sha(OUT/'reconstructed_pairs_and_witnesses.npz'),
                              'context_seconds':context_seconds,'freeze_seconds':freeze_seconds})
    phase='old_upper'
    captured=[];solver_timing={};upper_start=time.perf_counter()
    original_operation=old.operation_model;original_solve=operations.LinearMILP.solve
    def capture_operation(cfg):
        model,indices=original_operation(cfg);captured.append((model,indices));return model,indices
    def timed_solve(self,*args,**kwargs):
        solver_timing['entry_seconds']=time.perf_counter()-upper_start;start=time.perf_counter()
        try:return original_solve(self,*args,**kwargs)
        finally:solver_timing['seconds']=time.perf_counter()-start
    old.operation_model=capture_operation;operations.LinearMILP.solve=timed_solve
    try:old_out=old.upper_master(context_config:=__import__('research_experiments.mobile_ev_mle.adaptive_milp_study',fromlist=['configurations']).configurations()[1],pairs,witnesses,segments=2,time_limit=3)
    finally:old.operation_model=original_operation;operations.LinearMILP.solve=original_solve
    old_total=time.perf_counter()-upper_start;assert len(captured)==1
    old_model,old_indices=captured[0];old_objective=int(old_model.variables['squared_score_scaled'])
    old_record=result_record(old_out,old_total,solver_timing['entry_seconds'],solver_timing['seconds'])
    old_snapshot,old_digest=shared.model_snapshot(old_model,{old_objective:1.},{'method':'original upper_master','scale':old_out['scale'],'pair_fingerprints':{p.name:p.fingerprint() for p in pairs}})
    packed('old_model_snapshot.json.gz',old_snapshot);old_record['snapshot_sha256']=old_digest
    dump('old_upper_result.json',old_record);progress('old_upper_complete',**old_record)
    phase='shared_upper'
    solver_timing={};upper_start=time.perf_counter();operations.LinearMILP.solve=timed_solve
    try:new_out=shared.upper_master_shared(context_config,pairs,witnesses,segments=2,time_limit=3)
    finally:operations.LinearMILP.solve=original_solve
    new_total=time.perf_counter()-upper_start
    new_record=result_record(new_out,new_total,solver_timing['entry_seconds'],solver_timing['seconds'])
    packed('shared_model_snapshot.json.gz',new_out['snapshot']);new_record['snapshot_sha256']=new_out['snapshot_sha256']
    new_record['shared_graph_variables']=new_out['shared_graph_variables'];new_record['secant_errors']=new_out['secant_errors']
    dump('shared_upper_result.json',new_record);progress('shared_upper_complete',**{k:v for k,v in new_record.items() if k!='secant_errors'})
    phase='selected_point_checks'
    checks={};solver_arrays={}
    for label,out in [('old',old_out),('shared',new_out)]:
        if out['result'].x is None:checks[label]={'available':False,'reason':'master returned no incumbent'};continue
        solver_arrays[label+'_master_x']=out['result'].x
        qp=point(pairs,out['plan'].u_kw,label+'_selected');dump(label+'_selected_point_qp.json',qp)
        exact_single_site=all(np.count_nonzero(row!=0.0)<=1 for row in out['plan'].u_kw)
        identity_errors=[]
        for p,group in zip(pairs,witnesses):
            r=p.c+p.A@group[0];u=out['plan'].u_kw.ravel();constant_cost=float(r@r+np.sum((p.F@group[0])**2))
            linear=2*p.B.T@r;d=np.sum(p.B*p.B,axis=0)
            residual=r+p.B@u;direct=float(residual@residual+np.sum((p.F@group[0])**2))
            identity_errors.append(direct-float(constant_cost+linear@u+d@(u*u)))
        checks[label]={'available':True,'qp_complete':qp['complete'],'qp_q_lower':qp['q_lower'],'qp_q_upper':qp['q_upper'],
                       'point_qp_seconds':qp['seconds'],'physical_plan':operations.check_plan(out['plan'],context_config,tolerance=2e-6),
                       'selected_controls_exact_single_site':exact_single_site,'raw_decomposition_differences':identity_errors,
                       'max_absolute_decomposition_difference':max(abs(v) for v in identity_errors),'u_kw':out['plan'].u_kw}
    if solver_arrays:np.savez_compressed(OUT/'master_incumbents.npz',**solver_arrays)
    dump('selected_point_checks.json',checks)
    after={name:sha((BASE/name).resolve()) for name in frozen_hashes};assert after==frozen_hashes
    dump('source_hashes_after.json',{'frozen_core_sha256':after,'unchanged':True,'new_module_sha256':sha(PROTO/'shared_power_square_upper.py')})
    summary={'history_id':'710009','fresh_seed':28101301,'same_frozen_pairs_and_witnesses':True,'pair_count':24,
             'old':old_record,'shared':new_record,'selected_point_checks':checks,
             'original_11_action_library_read':False,'actual_future_or_truth_read':False,'whole_search_run':False,
             'core_sources_unchanged':True,'wallclock_scope':'single run, same machine, not a cross-hardware/runtime guarantee',
             'new_numerical_failure':not all(v.get('physical_plan',{}).get('feasible',True) and v.get('selected_controls_exact_single_site',True) for v in checks.values()),
             'upper_incumbent_is_not_proxy_gain_lower_bound':True,'done_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    dump('comparison_summary.json',summary);progress('comparison_complete',file=str(OUT/'comparison_summary.json'))
except Exception as error:
    failure={'phase':phase,'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),
             'preserved':True,'no_coefficients_clipped_or_pairs_removed':True}
    try:dump('comparison_failure.json',failure)
    except FileExistsError:pass
    progress('comparison_failure',**failure)
    raise
