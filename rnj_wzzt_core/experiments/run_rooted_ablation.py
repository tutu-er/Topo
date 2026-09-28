"""Predeclared rooted RNJ/wzzT ablations on paired low-data AC measurements.

Only experiment helpers are used to adapt bounded finite-pool LP enumeration;
production algorithms and defaults remain unchanged. See protocol.json.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,pickle,sys,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
sys.dont_write_bytecode=True
CORE=Path(__file__).resolve().parents[1];ROOT=CORE.parent
sys.path.insert(0,str(CORE))
import numpy as np
import pandas as pd
from rnj_wzzt.scenario.simulation import _simulate_pool,_terminal_buses
from rnj_wzzt.estimation.multiscenario import fit_projected_sensitivity,preprocess_scenarios
from rnj_wzzt.estimation.preprocessing import RECIPE
from rnj_wzzt.estimation.laminar_l1_milp import solve_fixed_support_l1,build_matrices_from_atoms,solver_diagnostics_prove_optimality,is_admissible_extension
from rnj_wzzt.graph.sensitivity_geometry import sensitivity_geometry
from rnj_wzzt.graph.rooted_neighbor_joining import rooted_neighbor_joining
from rnj_wzzt.graph.rooted_hierarchy import PseudoCluster,aggregate_rooted_scenarios,rooted_clades,rooted_tree_from_clades
from rnj_wzzt.graph.bootstrap import _boundary_cherries,_moving_block_bootstrap_copy,_select_disjoint
from rnj_wzzt.pipeline import _expand_pseudo_result_clades
from rnj_wzzt.reporting import _truth_nontrivial_clades
from rnj_wzzt.models.lin_distflow import build_reduced_sensitivity_matrices
from rooted_ablation_support import (BoundedPath,rooted_scores,serialized,validation_selection,
    _rnj_reduced_candidate_pool,_map_clade_to_reduced_support)
from rooted_study_baselines import infer_classical_nj

CASES=('paper15','soumalas11','flynn16','pengwah18')
METHODS=('rnj_R','rnj_X','rnj_RX75','rnj_fixed_tree_lp','classical_nj_validation','wzzt_unrestricted','wzzt_shared_pool','rnj_wzzt_frozen','rnj_wzzt_contracted')

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(set,frozenset,tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,Path):return str(x)
    return x

def write_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(payload),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def fingerprint():
    paths=[*sorted((CORE/'rnj_wzzt').rglob('*.py')),Path(__file__),CORE/'experiments/rooted_ablation_support.py',CORE/'experiments/rooted_study_baselines.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def dataset_arrays(scenarios):
    return {'A':np.vstack([np.hstack([s['P_terminal'],s['Q_terminal']]) for s in scenarios]),
        'Y':np.vstack([s['drop_target'] for s in scenarios]),'scenario':np.repeat(np.arange(len(scenarios)),[len(s['P_terminal']) for s in scenarios]),
        'root_true':np.concatenate([s['root_voltage_true'] for s in scenarios]),'root_observed':np.concatenate([s['root_voltage'] for s in scenarios]),
        'P_true':np.vstack([s['P_true'] for s in scenarios]),'Q_true':np.vstack([s['Q_true'] for s in scenarios]),'V_true':np.vstack([s['V_terminal_true'] for s in scenarios]),'V_measured':np.vstack([s['V_terminal'] for s in scenarios])}

def subset(scenarios,n):
    result=[]
    for i,s in enumerate(scenarios):
        item={**s,'name':f'scenario_{i}'}
        for key,value in s.items():
            if isinstance(value,(pd.DataFrame,pd.Series)) and len(value)==96:item[key]=value.iloc[::96//n].copy()
        item['source_sample_indices']=list(range(0,96,96//n))
        item['scenario_settings']={**s['scenario_settings'],'observed_sample_count':n,'observation_interval_hours':24.0/n}
        item['diagnostics']={**s['diagnostics'],'observed_sample_count':n,'selected_sample_count':n,'sample_basis':'full_physical_grid; selected measurement count updated without recomputing full-grid diagnostics'}
        result.append(item)
    return result

def prediction_metrics(sets,r,x,truth):
    answer={}
    for split,scenarios in sets.items():
        residual=np.vstack([s['drop_target'].to_numpy()-s['P_terminal'].to_numpy()@r.T-s['Q_terminal'].to_numpy()@x.T for s in scenarios])
        answer[f'{split}_mae']=float(np.mean(abs(residual)));answer[f'{split}_rmse']=float(np.sqrt(np.mean(residual**2)))
    answer['r_relative_error']=float(np.linalg.norm(r-truth[0])/max(np.linalg.norm(truth[0]),1e-15))
    answer['x_relative_error']=float(np.linalg.norm(x-truth[1])/max(np.linalg.norm(truth[1]),1e-15))
    for label,a,t in [('r',r,truth[0]),('x',x,truth[1])]:
        da=np.diag(a)[:,None]+np.diag(a)[None,:]-2*a;dt=np.diag(t)[:,None]+np.diag(t)[None,:]-2*t
        answer[f'{label}_distance_relative_error']=float(np.linalg.norm(da-dt)/max(np.linalg.norm(dt),1e-15))
    return answer

def infer_rnj(r,x,terminals,root,mode='RX_75R_25X'):
    geom=sensitivity_geometry(r,x,mode);tau=.16*max(float(np.median(geom.root_depths)),1e-12)
    tree=rooted_neighbor_joining(geom.shared_paths,geom.root_depths,terminals,root,group_tolerance=tau)
    return tree,rooted_clades(tree.edges,root,terminals),tau

def fixed_fit(scenarios,clades,terminals,solve_seconds):
    pos={t:i for i,t in enumerate(terminals)}
    supports=tuple([(i,) for i in range(len(terminals))]+[tuple(sorted(pos[t] for t in c)) for c in sorted(clades,key=lambda z:(len(z),tuple(sorted(z))))])
    sol=solve_fixed_support_l1(scenarios,supports,r_upper_bound=2,x_upper_bound=2,time_limit=solve_seconds)
    r,x=build_matrices_from_atoms(len(terminals),sol.supports,sol.r_values,sol.x_values)
    active={frozenset(terminals[j] for j in s) for i,s in enumerate(sol.supports) if 1<len(s)<len(terminals) and (sol.r_values[i]>1e-9 or sol.x_values[i]>1e-9)}
    return sol,r,x,active

def run_job(job):
    started=time.perf_counter();out=Path(job['directory']);out.mkdir(parents=True,exist_ok=True)
    with Path(job['cache']).open('rb') as f:cache=pickle.load(f)
    net=cache['net'];terminals=_terminal_buses(net);root=int(net.root_bus);n=len(terminals)
    sets={split:subset(raw,job['samples'] if split!='test' else 96) for split,raw in cache['sets'].items()}
    truth=np.asarray(build_reduced_sensitivity_matrices(net,terminals,voltage_model='squared-voltage'))
    truth_clades=_truth_nontrivial_clades(net,terminals)
    packed={f'{split}_{key}':value for split,scenarios in sets.items() for key,value in dataset_arrays(scenarios).items()}
    packed.update(R_true=truth[0],X_true=truth[1],terminals=np.asarray(terminals))
    np.savez_compressed(out/'inputs.npz',**packed)
    metadata={'case':job['case'],'repeat':job['repeat'],'samples':job['samples'],'pool_mode':job['pool_mode'],'root':root,'terminals':terminals,
        'truth_clades':serialized(truth_clades),'truth_edges':net.branches.to_dict(orient='records'),'scenario_settings':[[s['scenario_settings'] for s in sets[k]] for k in sets],
        'scenario_diagnostics':[[s['diagnostics'] for s in sets[k]] for k in sets],
        'root_full_terminal_stem':'excluded from nontrivial clade metrics; this score cannot identify subdivision or degree-2 hidden nodes',
        'sample_budget':{'training_snapshots':3*job['samples'],'validation_snapshots':3*job['samples'],'test_snapshots':288,'training_plus_selection_snapshots':6*job['samples']}}
    write_json(out/'metadata.json',metadata)
    prep_start=time.perf_counter();training=preprocess_scenarios(sets['train'],RECIPE)
    qp_diag={};fit=fit_projected_sensitivity(training,constraint_mode='ordered',diagnostics=qp_diag);r,x=fit[:2]
    qp_seconds=time.perf_counter()-prep_start;rows=[];details={};arrays={}
    basefields={key:job[key] for key in ('case','repeat','samples','pool_mode')}
    basefields.update(scenario_suite='reference',root_observation='noisy',terminal_count=n,train_snapshots=3*job['samples'],validation_snapshots=3*job['samples'],test_snapshots=288,search_budget_seconds=job['search_budget'])
    def save_method(name,clades,rr,xx,*,native=None,extra=None,required_seconds=0.0,search_seconds=0.0,post_seconds=0.0,status='complete',stop=''):
        row=basefields|{'method':name,'status':status,'stop_reason':stop,'preprocessing_seconds':required_seconds,'search_seconds':search_seconds,'original_terminal_refit_seconds':post_seconds,'elapsed_seconds':required_seconds+search_seconds+post_seconds}
        row['topology_available']=clades is not None
        if clades is not None:row.update(rooted_scores(clades,truth_clades,terminals))
        if rr is not None:row.update(prediction_metrics(sets,rr,xx,truth))
        if extra:row.update({k:v for k,v in extra.items() if isinstance(v,(str,int,float,bool)) or v is None})
        tree=None if clades is None else rooted_tree_from_clades(clades,terminals,root)
        detail={'row':row,'clades':None if clades is None else serialized(clades),'tree_edges':None if tree is None else tree.edges,'tree_edge_lengths':'unit placeholders for structural tree; coefficient matrices are stored separately','native':native,'extra':extra}
        write_json(out/f'{name}.json',detail)
        if rr is not None:
            np.savez_compressed(out/f'{name}.npz',R=rr,X=xx,R_true=truth[0],X_true=truth[1])
            arrays[name]=(rr,xx)
        details[name]=detail;rows.append(row);return row
    channel_results={};rnj_time=0.0
    for label,mode in [('R','R'),('X','X'),('RX75','RX_75R_25X')]:
        st=time.perf_counter();tree,clades,tau=infer_rnj(r,x,terminals,root,mode);seconds=time.perf_counter()-st
        channel_results[label]=(tree,clades,tau)
        if label=='RX75':rnj_time=seconds
        save_method(f'rnj_{label}',clades,r,x,native={'tree_edges':tree.edges,'tau':tau,'regression_diagnostics':qp_diag},required_seconds=qp_seconds+seconds)
    rnj_clades=channel_results['RX75'][1]
    st=time.perf_counter();rnj_fixed=None
    try:
        sol,fr,fx,active=fixed_fit(sets['train'],rnj_clades,terminals,job['solve_seconds'])
        rnj_fixed=save_method('rnj_fixed_tree_lp',rnj_clades,fr,fx,native={'diagnostics':sol.diagnostics.to_dict(),'supports':sol.supports,'r_values':sol.r_values,'x_values':sol.x_values,'positive_weight_clades':serialized(active)},required_seconds=qp_seconds+rnj_time,post_seconds=time.perf_counter()-st)
    except Exception as exc:
        save_method('rnj_fixed_tree_lp',rnj_clades,None,None,native={'error':repr(exc)},required_seconds=qp_seconds+rnj_time,post_seconds=time.perf_counter()-st,status='refit_failed',stop='fixed_tree_lp_failed')
    # Classical NJ tuning is predeclared and entirely validation-based.
    st=time.perf_counter();nj_candidates=[];nj_models=[]
    for factor in (0.0,.04,.08,.16):
        try:
            nt=infer_classical_nj(r,x,terminals,root,collapse_factor=factor);nc=rooted_clades(nt.edges,root,terminals)
            ns,nr,nx,na=fixed_fit(sets['train'],nc,terminals,job['solve_seconds']);metrics=prediction_metrics(sets,nr,nx,truth)
            nj_candidates.append({'collapse_factor':factor,'validation_mae':metrics['validation_mae'],'clades':serialized(nc),'diagnostics':ns.diagnostics.to_dict()})
            nj_models.append((metrics['validation_mae'],factor,nc,nr,nx,nt.edges))
        except Exception as exc:nj_candidates.append({'collapse_factor':factor,'error':repr(exc)})
    if nj_models:
        best=min(nj_models,key=lambda z:(z[0],z[1]))
        save_method('classical_nj_validation',best[2],best[3],best[4],native={'selected_collapse_factor':best[1],'candidates':nj_candidates,'tree_edges':best[5]},required_seconds=qp_seconds,search_seconds=time.perf_counter()-st)
    else:save_method('classical_nj_validation',None,None,None,native={'candidates':nj_candidates},status='failed',stop='all_classical_nj_candidates_failed',required_seconds=qp_seconds,search_seconds=time.perf_counter()-st)
    # Truth is not used for resampling, block selection, contraction, or candidates.
    st=time.perf_counter();rng=np.random.default_rng(20_260_908+1009*job['repeat']+job['samples']);counts={};bootstrap_failures=[]
    block_length=max(1,round(job['samples']/24))
    for bidx in range(job['bootstrap']):
        try:
            sample=_moving_block_bootstrap_copy(training,rng,block_length);bf=fit_projected_sensitivity(sample,constraint_mode='ordered')
            _,bc,_=infer_rnj(bf[0],bf[1],terminals,root)
            for c in _boundary_cherries(bc):counts[c]=counts.get(c,0)+1
        except Exception as exc:bootstrap_failures.append({'replicate':bidx,'error':repr(exc)})
    cherries=_boundary_cherries(rnj_clades);confidence={c:counts.get(c,0)/job['bootstrap'] for c in cherries|set(counts)}
    selected=_select_disjoint({c for c in cherries if confidence[c]>=.75},confidence,2)
    bootstrap_seconds=time.perf_counter()-st
    selected_precision=len(set(selected)&_boundary_cherries(truth_clades))/len(selected) if selected else None
    identity={t:frozenset({t}) for t in terminals};pos={t:i for i,t in enumerate(terminals)}
    pools={mode:_rnj_reduced_candidate_pool(rnj_clades,terminals,identity,include_one_edit=mode=='rnj_one_edit') for mode in ('rnj','rnj_one_edit')}
    coverage={mode:{'candidate_count':len(pool),'truth_recall':len({frozenset(terminals[i] for i in c) for c in pool}&truth_clades)/len(truth_clades),'support_indices':pool} for mode,pool in pools.items()}
    write_json(out/'selection.json',{'bootstrap_replicates':job['bootstrap'],'block_length':block_length,'confidence_threshold':.75,'maximum_selected':2,'confidence':[{'clade':sorted(c),'frequency':confidence[c]} for c in sorted(confidence,key=lambda z:(len(z),tuple(sorted(z))))],'selected':serialized(selected),'selected_truth_precision':selected_precision,'bootstrap_failures':bootstrap_failures,'bootstrap_seconds':bootstrap_seconds,'pool_coverage':coverage,'truth_used_for_selection':False})
    candidates=pools[job['pool_mode']]
    variants=['wzzt_unrestricted','wzzt_shared_pool','rnj_wzzt_frozen','rnj_wzzt_contracted']
    for method in variants:
        native=None;error=None;search_seconds=0;post_seconds=0;contraction_seconds=0;phase='search';clades=None
        raw_train=sets['train'];raw_validation=sets['validation'];members=identity;initial=[(i,) for i in range(n)];pool=None if method=='wzzt_unrestricted' else candidates
        required=0.0 if method=='wzzt_unrestricted' else qp_seconds+rnj_time
        if method in ('rnj_wzzt_frozen','rnj_wzzt_contracted'):required+=bootstrap_seconds
        if method=='rnj_wzzt_frozen':initial += [tuple(sorted(pos[t] for t in c)) for c in selected]
        elif method=='rnj_wzzt_contracted':
            st=time.perf_counter();clusters=[PseudoCluster(900000+i,c,confidence[c],tuple(),tuple()) for i,c in enumerate(selected)]
            raw_train,members=aggregate_rooted_scenarios(sets['train'],terminals,r,x,clusters,voltage_mode='deembedded_vsq',deembedding_weight=.5)
            raw_validation,check=aggregate_rooted_scenarios(sets['validation'],terminals,r,x,clusters,voltage_mode='deembedded_vsq',deembedding_weight=.5)
            if members!=check:raise RuntimeError('train/validation contraction mappings differ')
            labels=list(raw_train[0]['P_terminal'].columns);initial=[(i,) for i in range(len(labels))]
            mapped=[]
            for original_support in candidates:
                original_clade=frozenset(terminals[i] for i in original_support)
                reduced=_map_clade_to_reduced_support(original_clade,labels,members)
                if reduced is not None and len(reduced)>1:mapped.append(reduced)
            pool=tuple(sorted(set(mapped),key=lambda z:(len(z),z)))
            contraction_seconds=time.perf_counter()-st;required+=contraction_seconds
        current_labels=list(raw_train[0]['P_terminal'].columns)
        admissible_pool=None if pool is None else tuple(support for support in pool if is_admissible_extension(support,initial,len(current_labels)))
        current_pool_clades=set() if admissible_pool is None else {frozenset().union(*(members[int(current_labels[i])] for i in support)) for support in admissible_pool}
        current_pool_clades={c for c in current_pool_clades if 1<len(c)<n}
        frozen_clades={frozenset().union(*(members[int(current_labels[i])] for i in support)) for support in initial}
        frozen_clades={c for c in frozen_clades if 1<len(c)<n}
        actual_recall=None if pool is None else len((current_pool_clades|frozen_clades)&truth_clades)/len(truth_clades)
        st=time.perf_counter()
        adapter=BoundedPath(total_seconds=job['search_budget'],solve_seconds=job['solve_seconds'],enumerate_pool=pool is not None)
        try:
            with adapter:
                result=adapter.fit(raw_train,validation_scenarios=raw_validation,initial_supports=initial,candidate_supports=pool)
            search_seconds=time.perf_counter()-st
            clades=_expand_pseudo_result_clades(result,members,n)
            native={'summary':result.summary(),'R':result.r_matrix,'X':result.x_matrix,'terminal_labels':result.terminal_labels,'pseudo_members':{str(k):sorted(v) for k,v in members.items()},'path':[{'iteration':p.iteration,'support_labels':p.support_labels,'validation_mae':p.validation_mae,'validation_se':p.validation_se,'train_mae':p.train_mae,'solver':p.solver.to_dict()} for p in result.path], 'attempts':[{'support':a.support,'diagnostics':a.diagnostics.to_dict()} for a in result.attempted_extensions], 'audit':adapter.records()}
            if method!='rnj_wzzt_contracted':native['original_space_metrics']=prediction_metrics(sets,result.r_matrix,result.x_matrix,truth)
            stop=result.stop_reason;status='partial' if any(s in stop for s in ('not_proven','budget','bound_expansion_limit','without_incumbent')) else 'complete'
            phase='refit';pst=time.perf_counter();sol,pr,px,active=fixed_fit(sets['train'],clades,terminals,job['solve_seconds']);post_seconds=time.perf_counter()-pst
            native['mapped_pool_clades']=serialized(current_pool_clades)
            native['frozen_clades']=serialized(frozen_clades)
            native['common_refit']={'diagnostics':sol.diagnostics.to_dict(),'r_values':sol.r_values,'x_values':sol.x_values,'support_indices':sol.supports,'positive_weight_clades':serialized(active)}
            row=save_method(method,clades,pr,px,native=native,required_seconds=required,search_seconds=search_seconds,post_seconds=post_seconds,status=status,stop=stop,extra={'candidate_count':None if pool is None else len(pool),'initial_admissible_candidate_count':None if admissible_pool is None else len(admissible_pool),'candidate_truth_recall':actual_recall,'effective_admissible_truth_recall':actual_recall,'candidate_pool_only_truth_recall':None if pool is None else len(current_pool_clades&truth_clades)/len(truth_clades),'original_pool_truth_recall':None if pool is None else coverage[job['pool_mode']]['truth_recall'],'selected_block_count':len(selected) if 'rnj_wzzt' in method else 0,'frozen_block_precision':selected_precision if 'rnj_wzzt' in method else None,'bound_expansions':result.bound_expansions,'selected_path_index':result.selected_path_index,'search_terminal_count':len(result.terminal_labels),'solver_seconds':sum(d['runtime_seconds'] for d in adapter.solves),'contraction_seconds':contraction_seconds})
        except Exception as exc:
            if phase=='refit':
                post_seconds=time.perf_counter()-pst
                native['common_refit']={'error':repr(exc),'traceback':traceback.format_exc()}
                row=save_method(method,clades,None,None,native=native,required_seconds=required,search_seconds=search_seconds,post_seconds=post_seconds,status='refit_failed',stop='common_original_terminal_refit_failed',extra={'candidate_count':None if pool is None else len(pool),'candidate_truth_recall':actual_recall,'selected_block_count':len(selected) if 'rnj_wzzt' in method else 0})
            else:
                search_seconds=time.perf_counter()-st
                row=save_method(method,None,None,None,native={'error':repr(exc),'traceback':traceback.format_exc(),'audit':adapter.records()},required_seconds=required,search_seconds=search_seconds,status='failed',stop='search_failed',extra={'candidate_count':None if pool is None else len(pool),'selected_block_count':len(selected) if 'rnj_wzzt' in method else 0})
        # Selection comparator is the same original-terminal fixed-tree LP.
        if rnj_fixed is not None:
            row.update(validation_selection(row,rnj_fixed))
            details[method]['row']=row;write_json(out/f'{method}.json',details[method])
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    write_json(out/'result.json',{'job':job,'rows':rows,'job_wall_seconds':time.perf_counter()-started,'common_preprocessing_seconds':qp_seconds,'bootstrap_seconds':bootstrap_seconds,'failures':[r['method'] for r in rows if r['status'] in ('failed','refit_failed')]})
    return {'directory':str(out),'rows':rows,'job_wall_seconds':time.perf_counter()-started}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/root_information_study_20260908/ablation')
    parser.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES));parser.add_argument('--samples',nargs='+',type=int,default=[8,16,32])
    parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--bootstrap',type=int,default=12);parser.add_argument('--search-budget',type=float,default=12.0);parser.add_argument('--solve-seconds',type=float,default=2.0)
    parser.add_argument('--include-one-edit',action='store_true');parser.add_argument('--only-one-edit',action='store_true');parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(n<4 or 96%n for n in args.samples):raise ValueError('samples must divide 96 and be >=4')
    source=fingerprint();manifest=out/'source_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=source:raise RuntimeError('source fingerprint changed; use a new output directory to preserve evidence')
    write_json(manifest,source)
    config=vars(args)|{'output':str(out),'fixed_replicates':{r:{'train':3*r,'validation':3*r+1,'test':3*r+2} for r in range(args.repeats)}}
    if (out/'config.json').exists():
        previous=json.loads((out/'config.json').read_text())
        protected=('cases','samples','repeats','bootstrap','search_budget','solve_seconds','include_one_edit','only_one_edit')
        if any(previous.get(k)!=config.get(k) for k in protected):raise RuntimeError('experiment protocol options changed; use a new output directory')
    write_json(out/'config.json',config)
    write_json(out/'protocol.json',{'model':'observed root; raw squared-voltage drop = PR + QX; no free terminal or scenario bias; old intercept results require artifacts/observed_root_model_20260928/before','scenario_suite':'reference','root_observation':'noisy','pq_noise_rel':.005,'terminal_voltage_noise_rel':.0002,'root_meter_noise_rel':.0002,'impedance_scale':1.0,'physical_samples_per_scenario':96,'train_and_validation_independent':True,'test_samples_per_scenario':96,'total_selection_snapshots':'6N','root_and_terminal_partitions':'root=maximal nontrivial clades; terminal=minimal nontrivial clades; each predicted tree classified independently; root partition includes uncovered singleton terminals','full_terminal_stem':'not scored as a nontrivial clade; hidden degree-2 subdivisions unidentifiable','candidate_pool_primary':'rnj','one_edit_supplement':'predeclared all four cases and all repeats at N16','same_candidate_domain':'wzzt_shared_pool and rnj_wzzt_frozen have same pool; contraction maps each supplied original candidate exactly into reduced space, excluding supports that cut frozen blocks, without generating new reduced-coordinate one-edit neighbors; unrestricted control has all subsets','budgets':'same structural search budget for every forward method, per-solve bound, original default max_atoms and bound expansion; preprocessing and common original-terminal refit additional and timed','finite_pool_certification':'complete enumeration of certified fixed-support LPs establishes only candidate-domain single-step optimum; incomplete enumeration is never accepted as exact','common_refit':'all structure-method clades are refitted on original terminal training data by same L1 LP, upper bounds2; structure retained even if atom coefficient zero, positive-weight alternative clades saved','validation_selection':'core one-standard-error path selection then same original-terminal LP comparison vs RNJ fixed-tree LP; tie favors RNJ; no test/truth selection','bootstrap':'12 predeclared replicates, .75 threshold, at most2 blocks, circular block length approximates1hour but at least1 observed sample; empirical stability only','classical_nj':'root included as observed distance node; collapse factors0,.04,.08,.16 selected by zero-bias validation MAE after fixed-tree LP'})
    caches={}
    for case in args.cases:
        for repeat in range(args.repeats):
            path=out/'cache'/case/f'repeat_{repeat:02d}.pickle';path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                sets={};net=None
                for split,rep in [('train',3*repeat),('validation',3*repeat+1),('test',3*repeat+2)]:
                    net,sets[split]=_simulate_pool(case,96,rep,3,.005,.0002,scenario_suite='reference',root_observation='noisy')
                with path.open('wb') as f:pickle.dump({'net':net,'sets':sets},f,protocol=pickle.HIGHEST_PROTOCOL)
            caches[case,repeat]=str(path)
    if args.prepare_only:return
    jobs=[]
    for case in args.cases:
        for repeat in range(args.repeats):
            settings=[] if args.only_one_edit else [('rnj',n) for n in args.samples]
            if args.include_one_edit or args.only_one_edit:settings.append(('rnj_one_edit',16))
            for pool_mode,n in settings:
                directory=out/('primary' if pool_mode=='rnj' else 'one_edit')/case/f'repeat_{repeat:02d}'/f'N_{n:02d}'
                jobs.append({'case':case,'repeat':repeat,'samples':n,'pool_mode':pool_mode,'cache':caches[case,repeat],'directory':str(directory),'bootstrap':args.bootstrap,'search_budget':args.search_budget,'solve_seconds':args.solve_seconds})
    write_json(out/'planned_jobs.json',jobs);completed=[];failures=[]
    def collect():
        all_rows=[r for item in completed for r in item['rows']]
        pd.DataFrame(all_rows).to_csv(out/'metrics.csv',index=False)
        write_json(out/'result_index.json',{'expected_jobs':len(jobs),'completed_jobs':sum(not x.get('group_failed',False) for x in completed),'recorded_jobs':len(completed),'failed_jobs':failures,'jobs':[{'directory':x['directory'],'job_wall_seconds':x['job_wall_seconds']} for x in completed],'rows':len(all_rows)})
    pending=[]
    for job in jobs:
        result_file=Path(job['directory'])/'result.json'
        if result_file.exists():
            result=json.loads(result_file.read_text());completed.append({'directory':job['directory'],'rows':result['rows'],'job_wall_seconds':result['job_wall_seconds']})
        else:pending.append(job)
    collect()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(run_job,job):job for job in pending}
        for future in as_completed(futures):
            job=futures[future]
            try:completed.append(future.result());print(json.dumps({'done':len(completed),'total':len(jobs),'case':job['case'],'repeat':job['repeat'],'N':job['samples'],'pool':job['pool_mode']}),flush=True)
            except Exception as exc:
                failure={'job':job,'error':repr(exc),'traceback':traceback.format_exc()};failures.append(failure);write_json(Path(job['directory'])/'group_failure.json',failure)
                failure_rows=[{key:job[key] for key in ('case','repeat','samples','pool_mode')}|{'method':method,'status':'group_failed','topology_available':False,'stop_reason':repr(exc),'scenario_suite':'reference','root_observation':'noisy'} for method in METHODS]
                pd.DataFrame(failure_rows).to_csv(Path(job['directory'])/'metrics.csv',index=False)
                completed.append({'directory':job['directory'],'rows':failure_rows,'job_wall_seconds':None,'group_failed':True})
                print(json.dumps({'failed':job,'error':repr(exc)}),flush=True)
            collect()
if __name__=='__main__':main()
