"""Replay saved pilot evidence. No fit optimizer or selector is called.

QP/model checks use saved matrices. AC replay uses saved H-fitted state and frozen helpers.
"""
from pathlib import Path
from types import SimpleNamespace
import gzip,hashlib,json,math,os,sys
import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
FROZEN=Path(os.environ.get('FROZEN_TOPO_ROOT',r'C:\Users\23761\.codex\worktrees\36a1\Topo'))
sys.path.insert(0,str(FROZEN))
from research_experiments.mobile_ev_mle.adaptive_history import evaluate_frozen_pair_actual
from research_experiments.mobile_ev_mle.likelihood import Observations,MeasurementModel
from research_experiments.mobile_ev_mle.candidates import RadialCandidate
SUPPORTS={'clade_02':((0,2),),'clade_01':((0,1),),'star':(),'clade_12':((1,2),)}
def js(path):return json.loads(path.read_text(encoding='utf-8'))
def zipped(path):return json.loads(gzip.decompress(path.read_bytes()))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def restore(s):
    c={'history':Observations(**s['history']),'model':MeasurementModel(**s['measurement_model']),
       'config':s['config'],'channels':tuple(s['channels']),'scenarios':s['scenarios'],'centres':{}}
    for scenario in c['scenarios']:
        for key in ['p0','q0','root_v']:scenario[key]=np.asarray(scenario[key])
    for name,rows in s['centres'].items():
        cand=RadialCandidate.from_supports(name,3,SUPPORTS[name])
        c['centres'][name]=[{'candidate':cand,'beta':np.asarray(r['beta']),'history_state':np.asarray(r['history_state']),
                  'fit':SimpleNamespace(**{k:np.asarray(v) for k,v in r['fit'].items()})} for r in rows]
    return c

before=js(ROOT/'code_and_input_hashes_before.json');after=js(ROOT/'code_hashes_after.json')
assert before['frozen_core']==after['frozen_core'] and before['new_shared_module']==after['new_shared_module']
assert after['original_global_upper_restored'] and after['original_MILP_solve_restored']
for name,expected in before['frozen_core'].items():assert sha((FROZEN/'research_experiments/mobile_ev_mle'/name).resolve())==expected
assert sha(FROZEN/'research_experiments/mobile_ev_mle/shared_power_square_upper.py')==before['new_shared_module']
max_primal=0.;max_tangent=0.;max_model_violation=0.;max_AC_cost=0.;max_AC_norm=0.
qp_count=0;model_count=0;AC_count=0;failures=[];case_results=[]
for ident in ['710009','720001']:
    case=ROOT/f'H_{ident}';manifest=js(case/'pair_manifest.json');assert manifest['pair_count']==24
    assert sha(case/'frozen_pairs.npz')==manifest['archive_sha256']
    with np.load(case/'frozen_pairs.npz',allow_pickle=False) as data:
        pairs=[{k:data[f'pair_{i}_{k}'].copy() for k in ['c','B','A','F','lower','upper','scale']} for i in range(24)]
    for name,p in zip(manifest['names'],pairs):
        h=hashlib.sha256(name.encode())
        for key in ['c','B','A','F','lower','upper','scale']:
            a=p[key];h.update(str(a.shape).encode());h.update(np.ascontiguousarray(a).tobytes())
        assert h.hexdigest()==manifest['fingerprints'][name]
        for row in p['B']:assert len(np.unique(np.flatnonzero(row!=0.0)//3))<=1
    pair_map=dict(zip(manifest['names'],pairs))
    def replay_QP(records):
        global qp_count,max_primal,max_tangent
        for r in records:
            if 'exception' in r:failures.append({'history':ident,**r});continue
            p=pair_map[r['pair']];u=np.asarray(r['u_kw']).ravel();z=np.asarray(r['z'])
            assert np.all(z>=p['lower']) and np.all(z<=p['upper'])
            C=np.vstack([p['A'],p['F']]);d=np.r_[p['c']+p['B']@u,np.zeros(len(p['F']))]
            residual=d+C@z;primal=float(residual@residual);grad=2*C.T@residual
            correction=np.minimum(grad*(p['lower']-z),grad*(p['upper']-z))
            lower=max(0,primal+float(correction.sum())-r['roundoff_guard'])
            pd=abs(primal-r['q_upper']);ld=abs(lower-r['q_lower'])
            max_primal=max(max_primal,pd);max_tangent=max(max_tangent,ld)
            assert pd<1e-8*max(1,abs(primal)) and ld<1e-8*max(1,abs(lower))
            assert r['numeric_gap']==r['q_upper']-r['q_lower'];qp_count+=1
    baselines=zipped(case/'initial_action_QP.json.gz')
    for value in baselines.values():replay_QP(value['records'])
    replay_context=restore(zipped(case/'AC_replay_context.json.gz'))
    for label in ['original','shared']:
        target=case/label;adapter=js(target/'adapter_audit.json')
        assert adapter['same_code_object'] and adapter['keyword_defaults_copied'] and adapter['original_global_upper_unmodified']
        expected={'solve_pair'} if label=='original' else {'solve_pair','upper_master'}
        assert set(adapter['changed_globals'])==expected
        replay_QP(zipped(target/'qp_trace.json.gz'))
        point=js(target/'selected_point_qp.json');replay_QP(point['records'])
        result=js(target/'search_result.json');audit=js(target/'selected_audit.json')
        assert result['raw_global_gap']==result['proxy_upper']-result['proxy_lower']
        assert result['relative_gap']==result['absolute_gap']/max(1,abs(result['proxy_lower']))
        assert audit['absolute_stop_tolerance']==1e-4+.001*max(1,abs(result['proxy_lower']))
        assert result['status']=='proxy_gap_met' and result['absolute_gap']<=audit['absolute_stop_tolerance']
        assert audit['raw_controls']['clipping_or_rounding_performed'] is False
        for key,baseline in baselines.items():assert audit['gain_intervals_vs_initial_actions'][key]==[
            point['q_lower']-baseline['q_upper'],point['q_upper']-baseline['q_lower']]
        saved_AC=js(target/'full_AC_rollback.json');u=np.asarray(audit['raw_controls']['u_kw'])
        for p,md,q,previous in zip(pairs,manifest['raw_pair_metadata'],point['records'],saved_AC):
            raw={**p,**md}
            r=evaluate_frozen_pair_actual(replay_context,raw,u,np.asarray(q['z']))
            max_AC_cost=max(max_AC_cost,abs(r['actual_soft_cost']-previous['actual_soft_cost']),abs(r['affine_soft_cost']-previous['affine_soft_cost']))
            max_AC_norm=max(max_AC_norm,abs(r['joint_taylor_error_norm']-previous['joint_taylor_error_norm']))
            assert bool(r['physically_feasible'])==bool(previous['physically_feasible'])
            assert abs(r['actual_soft_cost']-(sum(r['actual_history_costs'])+r['actual_future_distance_squared']))<1e-8
            assert previous['nonlinear_infimum_computed'] is False and previous['is_nonlinear_design_bound'] is False
            AC_count+=1
        incumbent_arrays={}
        if (target/'master_incumbents.npz').exists():
            with np.load(target/'master_incumbents.npz',allow_pickle=False) as data:incumbent_arrays={k:data[k].copy() for k in data.files}
        for model in js(target/'solver_calls.json'):
            raw=gzip.decompress((target/model['snapshot_file']).read_bytes());assert hashlib.sha256(raw).hexdigest()==model['snapshot_sha256']
            snapshot=json.loads(raw);assert len(snapshot['lower'])==model['n_variables'] and len(snapshot['rows'])==model['n_rows']
            x=incumbent_arrays.get(f"model_{model['call']}_x")
            if x is not None:
                violations=[]
                for v,lo,hi in zip(x,snapshot['lower'],snapshot['upper']):violations.extend([float(lo)-v,v-float(hi)])
                for row,lo,hi in zip(snapshot['rows'],snapshot['row_lower'],snapshot['row_upper']):
                    value=math.fsum(float(c)*x[int(k)] for k,c in row.items());violations.extend([float(lo)-value,value-float(hi)])
                violation=max(0,max(violations));max_model_violation=max(max_model_violation,violation)
                assert violation<2e-6
                objective=math.fsum(float(c)*x[int(k)] for k,c in snapshot['objective'].items())
                assert abs(objective-model['objective_value'])<1e-8
            elif model.get('objective_value') is not None:raise AssertionError('incumbent objective without saved x')
            model_count+=1
        case_results.append({'history':ident,'backend':label,'pair_count':24,'QP_trace_rows':len(zipped(target/'qp_trace.json.gz')),
                             'profile_gain_vs_constant':audit['gain_intervals_vs_initial_actions']['constant'],
                             'AC_pairs_replayed':24,'raw_global_gap':result['raw_global_gap']})
assert qp_count==696 and model_count==16 and AC_count==96
assert max_AC_cost<1e-8 and max_AC_norm<1e-10
print(json.dumps({'status':'PASS','fit_optimizer_or_selector_called':False,'QP_records_replayed':qp_count,
        'MILP_snapshots_verified':model_count,'full_AC_saved_witness_records_replayed':AC_count,
        'max_primal_difference':max_primal,'max_tangent_difference':max_tangent,
        'max_model_row_violation':max_model_violation,'max_actual_or_affine_AC_cost_difference':max_AC_cost,
        'max_taylor_error_norm_difference':max_AC_norm,'source_hashes_unchanged':15,
        'QP_exceptions_preserved':len(failures),'cases':case_results},ensure_ascii=False,indent=2))
