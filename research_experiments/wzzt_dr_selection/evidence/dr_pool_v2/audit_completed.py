"""Independent arithmetic replay only: no fitting, QP, LP, MILP or simulation."""
import argparse,hashlib,json,pickle,sys,platform
import scipy
import gurobipy as gp
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
parser=argparse.ArgumentParser()
parser.add_argument('--run-root',type=Path,required=True)
parser.add_argument('--repo-root',type=Path,required=True)
parser.add_argument('--receipt',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.repo_root.resolve()))
from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import arrays
RUN=args.run_root.resolve()
from rnj_wzzt.estimation import multiscenario
core_files={str(p.resolve()) for p in Path(multiscenario.__file__).parents[1].rglob('*.py')}
runtime_environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,gurobi=list(gp.gurobi.version()))
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def obj(p):return pickle.loads(p.read_bytes())
def get_obs(scene,rows):
    o=scene['observed'];out=tuple(o[k].iloc[rows].copy() for k in ('P_terminal','Q_terminal','V_terminal','root_voltage'))
    for c in out:assert c.index.equals(out[0].index)
    for c in out[:3]:assert c.columns.equals(out[0].columns)
    y=out[3].to_numpy()[:,None]**2-out[2].to_numpy()**2
    assert np.allclose(y,o['drop_target'].iloc[rows].to_numpy(),rtol=0,atol=1e-14)
    return out

def join(items):return tuple(pd.concat([o[i] for o in items]) for i in range(4))
def input_hash(o,days):
    h=hashlib.sha256()
    for v in o:h.update(pd.util.hash_pandas_object(v,index=True).values.tobytes())
    h.update(json.dumps(list(o[0].columns),default=str).encode());h.update(json.dumps(days).encode());return h.hexdigest()
def build(point,n):
    z=np.zeros((len(point['supports']),n))
    for j,s in enumerate(point['supports']):z[j,s]=1.
    return z.T@(np.asarray(point['r'])[:,None]*z),z.T@(np.asarray(point['x'])[:,None]*z)
def loss(point,o):
    R,X=build(point,o[0].shape[1]);p,q,v,u=[x.to_numpy() for x in o]
    y=u[:,None]**2-v**2
    # Sum each block separately, rather than only relying on matrix reconstruction.
    block=np.zeros_like(y)
    for s,r,x in zip(point['supports'],point['r'],point['x']):
        block[:,s]+=(r*p[:,s].sum(1)+x*q[:,s].sum(1))[:,None]
    prediction=p@R.T+q@X.T
    assert np.allclose(block,prediction,rtol=0,atol=1e-13)
    return float(np.mean(abs(prediction-y)))
def same(a,b):return a['supports']==b['supports'] and np.array_equal(a['r'],b['r']) and np.array_equal(a['x'],b['x'])
protocol=read(RUN/'protocol.json');scenarios={s['id']:s for s in protocol['scenarios']}
receipt={'created_at':datetime.now(timezone.utc).isoformat(),'audit_source_sha256':sha(Path(__file__)),'mode':'arithmetic replay only; no solve or simulation','protocol_sha256':sha(RUN/'protocol.json'),'records':[]}
for result in sorted((RUN/'fit_cache').glob('*/result.json')):
    fit=read(result)
    if fit['status']!='complete':continue
    key=result.parent.name;row=next(r for r in protocol['task_inventory'] if r['observation_key']==key)
    source=Path(scenarios[row['scenario_id']]['source_run']);meta=read(source/'summary.json')
    paths=[source/'history'/(date+'.pkl') for date in meta['wzzt_training_dates']]
    folder=source/'strategies'/row['strategy'];paths.extend([folder/'action.pkl',folder/'validation.pkl'])
    step=96//row['N'];ix=np.arange(0,24,step)
    tr=join([get_obs(obj(p),ix) for p in paths[:3]]+[get_obs(obj(paths[3])['simulation'],ix)])
    va=get_obs(obj(paths[4])['simulation'],np.arange(24))
    out={'cache_key':key,'scenario_id':row['scenario_id'],'strategy':row['strategy'],'N':row['N'],'result_sha256':sha(result),'source_observation_sha256':{str(p):sha(p) for p in paths},'issues':[],'models':{}}
    def check(test,msg):
        if not bool(test):out['issues'].append(msg)
    days=[str(d) for d in tr[0].index.normalize()];vd=[str(d) for d in va[0].index.normalize()]
    counts={d:days.count(d) for d in dict.fromkeys(days)};vcounts={d:vd.count(d) for d in dict.fromkeys(vd)}
    check(counts==fit['training_day_counts'],'training date counts');check(vcounts==fit['validation_day_counts'],'validation date counts')
    check(len(tr[0])==fit['training_rows']==row['N'],'training rows');check(len(va[0])==fit['validation_rows']==24,'validation rows')
    check(len(tr[0].index.intersection(va[0].index))==0,'train validation overlap')
    check(tr[0].columns.tolist()==fit['terminal_labels'],'terminal labels')
    check(input_hash(tr,days)==fit['input_sha256']['training'],'training hash mismatch');check(input_hash(va,vd)==fit['input_sha256']['validation'],'validation hash mismatch')
    expected=[tr[0].to_numpy().copy(),tr[1].to_numpy().copy(),tr[3].to_numpy()[:,None]**2-tr[2].to_numpy()**2]
    for d in dict.fromkeys(days):
        ids=np.array(days)==d
        for value in expected:value[ids]-=np.mean(value[ids],axis=0)
    computed=arrays(tr,'demean',day_ids=days)
    out['demean_max_abs_difference']=max(float(np.max(abs(a-b))) for a,b in zip(expected,computed))
    check(out['demean_max_abs_difference']<1e-14,'daily demeaning mismatch');out['training_day_counts']=counts
    out['source_closure_count']=len(fit['source_sha256'])
    out['source_closure_matches_protocol']=fit['source_sha256']==protocol['algorithm_source_sha256']
    out['core_source_closure_complete']=core_files.issubset(set(fit['source_sha256']))
    out['current_source_hashes_match']=all(sha(Path(p))==h for p,h in fit['source_sha256'].items())
    out['environment_matches_runtime']=fit['environment']==runtime_environment
    check(out['source_closure_matches_protocol'],'protocol source closure mismatch')
    check(out['core_source_closure_complete'],'core source closure incomplete')
    check(out['current_source_hashes_match'],'source changed since solve')
    check(out['environment_matches_runtime'],'environment mismatch')
    sols=fit['LP_solves'];out['LP_count']=len(sols);out['LP_status_counts']={str(s):sum(x['status']==s for x in sols) for s in {x['status'] for x in sols}}
    out['max_LP_objective_discrepancy']=max(abs(s['objective']-s['physical']) for s in sols)
    out['max_LP_constraint_violation']=max(s['max_constraint_violation'] for s in sols);out['max_LP_bound_violation']=max(s['max_bound_violation'] for s in sols)
    check(all(s['status']==2 and s['accepted'] for s in sols),'LP not optimal accepted');check(out['max_LP_constraint_violation']<=1e-7 and out['max_LP_bound_violation']<=1e-7,'LP feasibility')
    errors=[]
    for point in fit['selection_candidates']:
        check(all(np.isfinite(w).all() and min(w)>=-1e-9 and max(w)<=2.+1e-9 for w in (point['r'],point['x'])),'candidate weight bound/finite')
        check(all(s in fit['pool_support_indices'] for s in point['supports']),'candidate outside pool')
        family=[set(s) for s in point['supports']]
        check(all(not (a&b) or a<=b or b<=a for a in family for b in family),'candidate nonlaminar')
        errors.extend([abs(loss(point,tr)-point['objective']),abs(loss(point,va)-point['validation_mae'])])
    out['selection_candidate_count']=len(fit['selection_candidates']);out['candidate_loss_max_abs_difference']=max(errors)
    check(max(errors)<1e-12,'stored candidate loss mismatch')
    for name,m in fit['models'].items():
        before=m['selected_uncontracted'];R,X=build(m,len(fit['terminal_labels']));rb,xb=build(before,len(fit['terminal_labels']))
        train=loss(m,tr);valid=loss(m,va);bt=loss(before,tr);bv=loss(before,va)
        matrixerror=max(float(np.max(abs(R-np.asarray(m[k])))) for k in ('R','r_matrix'))
        matrixerror=max(matrixerror,max(float(np.max(abs(X-np.asarray(m[k])))) for k in ('X','x_matrix')))
        errors=[abs(train-m['train_mae']),abs(train-m['objective']),abs(valid-m['validation_mae']),abs(bt-before['objective']),abs(bv-before['validation_mae']),abs(bv-m['selection_validation_mae'])]
        check(matrixerror<1e-13,name+' matrix discrepancy');check(max(errors)<1e-12,name+' loss discrepancy')
        check(m['support_labels']==[[fit['terminal_labels'][i] for i in s] for s in m['supports']],name+' support labels')
        check(all([i] in m['supports'] for i in range(len(R))),name+' missing terminal leaf')
        check(all(min(w)>=-1e-9 and max(w)<=2.+1e-9 for w in (m['r'],m['x'])),name+' weight bounds')
        candidates=fit['bank'] if name=='rnj_bank' else [p for p in fit['selection_candidates'] if name=='multistart_original_rule' or p['origin'] in ('tree_bank','tree_initialized_correction')]
        best=min(p['validation_mae'] for p in candidates);eligible=[i for i,p in enumerate(candidates) if p['validation_mae']<=1.03*best+1e-12]
        win=min(eligible,key=lambda i:(len(candidates[i]['supports']),candidates[i]['validation_mae'],i));check(same(before,candidates[win]),name+' original selection mismatch');check(m['selection_index']==win,name+' selection index')
        cal=fit['calibrations']['raw'];keep=[i for i,s in enumerate(before['supports']) if len(s)==1 or cal['a']*before['r'][i]+cal['b']*before['x'][i]+1e-10>=.16*cal['median_root_depth']]
        reduced={k:[before[k][i] for i in keep] for k in ('supports','r','x')};check(same(m,reduced),name+' contraction mismatch')
        out['models'][name]={'final_train_mae':train,'final_validation_mae':valid,'pre_contraction_train_mae':bt,'pre_contraction_validation_mae':bv,'final_support_count':len(m['supports']),'pre_contraction_support_count':len(before['supports']),'matrix_max_abs_difference':matrixerror,'loss_max_abs_difference':max(errors),'pre_contraction_R_sha256':hashlib.sha256(rb.tobytes()).hexdigest(),'pre_contraction_X_sha256':hashlib.sha256(xb.tobytes()).hexdigest(),'selection_index':win}
    out['search_stop_statuses']=fit['search_stop_statuses'];out['pass']=not out['issues'];receipt['records'].append(out)
receipt['completed_fits_audited']=len(receipt['records']);receipt['passed_fits']=sum(r['pass'] for r in receipt['records']);receipt['LPs_audited']=sum(r['LP_count'] for r in receipt['records']);receipt['all_passed']=bool(receipt['records']) and all(r['pass'] for r in receipt['records']);receipt['no_solves_or_code_changes']=True
receipt['technical_conclusion']='Observed data, date grouping, arithmetic losses, model selection, contraction, source hashes and fixed-family LP statuses agree; continuing the already declared bounded matrix is technically supported.' if receipt['all_passed'] else 'Stop and inspect audit issues before continuing.'
receipt['runtime_environment']=runtime_environment
receipt['recorded_algorithm_source_count']=len(protocol['algorithm_source_sha256'])
summary=read(RUN/'summary.json')
receipt['matrix_summary_counts']={k:summary[k] for k in ('status','planned_cells','evaluated_cells','failed_or_unavailable_cells','planned_unique_observation_fits','completed_unique_fits','reused_evaluated_cells')}
receipt['all_expected_unique_results_audited']=receipt['completed_fits_audited']==protocol['planned_unique_observation_fits']==42
if not receipt['all_expected_unique_results_audited']:receipt['all_passed']=False
receipt['technical_conclusion']='All 42 unique fits in the completed declared matrix are arithmetically consistent with source observations, frozen selection rules, LP status records, source closure and runtime environment. Efficacy interpretation is separate.' if receipt['all_passed'] else 'Inspect audit issues before interpreting results.'
path=args.receipt.resolve();path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(receipt,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in receipt.items() if k!='records'},indent=2));print('issues',[(r['strategy'],r['N'],r['issues']) for r in receipt['records'] if r['issues']])
