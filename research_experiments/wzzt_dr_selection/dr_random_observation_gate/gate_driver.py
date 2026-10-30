"""Small observed-only random-control candidate gate using the real LP/search.

This module creates no data and performs no optimization on import. `prepare`
freezes observed packet/discovery hashes; only an explicit `run` calls solvers.
The simulator, price design and truth evaluator stay outside this identifier.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import pickle,platform,sys
from time import perf_counter
import traceback

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'

REPO=Path(os.environ.get('WZZT_REPO_ROOT',r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo'))
sys.path[:0]=[str(REPO),str(REPO/'rnj_wzzt_core')]


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def plain(x):
    import numpy as np
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def canonical(family):return tuple(sorted(set(tuple(sorted(int(i) for i in s)) for s in family),key=lambda s:(len(s),s)))
def compatible(a,b):
    a,b=set(a),set(b)
    return not a&b or a<=b or b<=a
def validate_supports(supports,n,*,family=False):
    result=canonical(supports)
    if any(not s or len(s)!=len(set(s)) or min(s)<0 or max(s)>=n for s in result):
        raise ValueError('candidate support indices outside the observed terminal order')
    if family and any(not compatible(a,b) for j,a in enumerate(result) for b in result[:j]):
        raise ValueError('direct family is not laminar; no silent repair')
    return result
def internal_universe(n):
    from itertools import combinations
    if n not in (3,4):raise ValueError('only preregistered n3/n4 gates are supported')
    return tuple(s for k in range(2,n+1) for s in combinations(range(n),k))
def legal_families(n,pool=None):
    """All8 n3 or all52 n4 families; leaf coefficients always allow zero."""
    internal=internal_universe(n) if pool is None else tuple(s for s in canonical(pool) if len(s)>1)
    leaves=[(j,) for j in range(n)];result=[]
    for bits in range(1<<len(internal)):
        chosen=[s for j,s in enumerate(internal) if (bits>>j)&1]
        if all(compatible(a,b) for j,a in enumerate(chosen) for b in chosen[:j]):
            result.append(canonical(leaves+chosen))
    return tuple(sorted(set(result),key=lambda f:(len(f),f)))


class Budget:
    def __init__(self,seconds,max_calls):
        self.started=perf_counter();self.deadline=self.started+float(seconds)
        self.max_calls=int(max_calls);self.calls=0
    def check(self):
        if perf_counter()>=self.deadline:raise RuntimeError('preregistered gate wall limit reached')
        if self.calls>=self.max_calls:raise RuntimeError('preregistered real LP call limit reached')


class SearchView:
    """Only proposals change; fixed-support fitting uses the unmodified PoolLP."""
    def __init__(self,lp,pool,budget):self.lp,self.pool,self.n,self.budget=lp,canonical(pool),lp.n,budget
    def fit(self,family):
        family=canonical(family)
        if family not in self.lp.cache:
            self.budget.check()
            self.lp.model.Params.TimeLimit=min(15.,max(.001,self.budget.deadline-perf_counter()))
        before=len(self.lp.solves)
        try:return self.lp.fit(family)
        finally:self.budget.calls+=len(self.lp.solves)-before


def _load_packet(path,*,future=False):
    packet=pickle.loads(Path(path).read_bytes())
    if not isinstance(packet,dict) or set(packet)-{'training','validation','metadata'}:
        raise ValueError('identifier packet must contain observed splits/metadata only')
    if not all(isinstance(packet.get(role),list) for role in ('training','validation')):
        raise ValueError('observed training and validation scenario lists required')
    meta=packet.get('metadata',{})
    if not meta.get('observed_only',False):raise ValueError('observed_only declaration required')
    if set(meta)&{'truth','true_supports','physical_network','R_true','X_true','natural_P','natural_Q'}:
        raise ValueError('evaluator truth/latent natural baseline must not enter identifier metadata')
    if future and meta.get('contains_history') is not False:
        raise ValueError('future packet must explicitly exclude old history; old rows are prepended by driver')
    return packet


def _observations(training,validation):
    from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize,input_hash
    train,days,counts=normalize(training,'training');valid,vdays,vcounts=normalize(validation,'validation')
    if not train[0].columns.equals(valid[0].columns):raise ValueError('train/validation terminal order differs')
    if len(train[0].index.intersection(valid[0].index)):raise ValueError('train/validation times overlap')
    return train,valid,days,dict(training=input_hash(train,days),validation=input_hash(valid,vdays)),dict(training=counts,validation=vcounts)


def observed_training_for_discovery(history_packet,future_packet):
    """Convert only observed training splits to the random_code pure interface."""
    from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize,arrays
    result={};labels=None
    for name,packet in (('history_train',history_packet),('future_train',future_packet)):
        obs,_,_=normalize(packet['training'],name)
        current=list(obs[0].columns)
        if labels is not None and current!=labels:raise ValueError('discovery terminal order mismatch')
        labels=current;p,q,y=arrays(obs)
        result[name]=dict(P=p,Q=q,Y=y)
    return result


def _fixed_source_hashes():
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
    result=source_hashes();result[str(Path(__file__).resolve())]=sha(__file__)
    return result


def prepare(root,config_path):
    """Freeze paths and hashes only; no packet loading, calibration or solver."""
    root=Path(root).resolve();config_path=Path(config_path).resolve()
    if root.exists():raise FileExistsError('existing gate root retained')
    config=read(config_path)
    if config.get('status')!='root_frozen_before_solver':raise ValueError('root protocol must be frozen before execution')
    settings=config['settings']
    if settings.get('coefficient_bound')!=2.:raise ValueError('keep actual coefficient box[0,2]; zero is always allowed')
    if settings.get('calibration_snapshot')!='current_observed':
        raise ValueError('this gate applies one current-observed calibration to every method')
    if settings.get('max_n') not in (3,4):raise ValueError('max_n must match a preregistered small gate')
    if settings['max_n']==4 and not config.get('n3_gate_passed',False):raise ValueError('n4 requires root-declared n3 acceptance')
    if float(settings['search_seconds'])<=0 or float(settings['wall_seconds'])<=0 or int(settings['max_LP_calls'])<=0:
        raise ValueError('positive locked budgets required')
    histories={r['id']:r for r in config['histories']}
    if len(histories)!=len(config['histories']):raise ValueError('duplicate historical input ids')
    inputs={str(config_path):sha(config_path)}
    for row in config['histories']+config['snapshots']:
        p=Path(row['path']).resolve()
        if sha(p)!=row['sha256']:raise ValueError('observed packet hash mismatch')
        inputs[str(p)]=row['sha256']
    snapshot_ids=set()
    for row in config['snapshots']:
        if row['id'] in snapshot_ids or row['history_id'] not in histories:raise ValueError('snapshot identity mismatch')
        snapshot_ids.add(row['id'])
        discovery=row.get('discovery')
        if discovery:
            p=Path(discovery['path']).resolve()
            if sha(p)!=discovery['sha256']:raise ValueError('discovery file hash mismatch')
            inputs[str(p)]=discovery['sha256']
    methods=['historical_rnj_refit','current_rebuilt_rnj','old_pool_wzzt','current_pool_wzzt',
             'expanded_pool_wzzt','same_pool_tree_enumeration','all_legal_family_reference','direct_control_tree_refit']
    cfg=dict(config,prepared_utc=now(),config_path=str(config_path),input_sha256=inputs,
             source_sha256=_fixed_source_hashes(),
             label='observed-only decision-chain gate; synthetic data/price/truth handled outside this identifier',
             methods=methods,
             inherited_rules=dict(gain_absolute=1e-10,gain_relative=.001,validation_fraction=.03,
                                  internal_resolution_factor=.16,zero_leaf_weights_allowed=True,internal_numerical_prune=1e-9))
    root.mkdir(parents=True);write(root/'protocol.json',cfg)
    write(root/'protocol.sha256.json',dict(sha256=sha(root/'protocol.json')))
    return cfg


def verify(root):
    root=Path(root);cfg=read(root/'protocol.json')
    if sha(root/'protocol.json')!=read(root/'protocol.sha256.json')['sha256']:raise RuntimeError('frozen protocol drift')
    for group in ('source_sha256','input_sha256'):
        for path,h in cfg[group].items():
            if sha(path)!=h:raise RuntimeError(group+' drift: '+path)
    return cfg


def prepare_observed_state(packet,settings,budget,out,*,context):
    """Build one observed bank and starts; no manual/truth calibration is used."""
    from research_experiments.wzzt_dr_selection.dr_pool_v2.engine import build_bank,PoolLP
    from research_experiments.wzzt_dr_selection.dr_pool_v2.selection import measure,select,contract,package
    train,valid,days,hashes,counts=_observations(packet['training'],packet['validation'])
    n=len(train[0].columns)
    if n>settings['max_n']:raise ValueError('case exceeds declared gate size')
    budget.check();started=perf_counter()
    bank,pool,fronts,calibrations=build_bank(train,day_ids=days,root_label=int(settings['root_label']))
    calibration=calibrations['raw'];native=[b for b in bank if b['preprocess']=='raw' and b['method']=='RNJ']
    if len(native)!=15:raise RuntimeError('expected original15 native RNJ families')
    lp=None;result=dict(status='running',context=context,terminal_labels=list(train[0].columns),calibration=calibration,
        original_calibrations=calibrations,fronts=fronts,pool=pool,old_pool=pool,
        native_bank=native,input_sha256=hashes,day_counts=counts,
        applied_calibration='raw calibration from this observed training snapshot')
    try:
        lp=PoolLP(train,pool,'raw',bound=2.);view=SearchView(lp,pool,budget)
        fitted=[measure(dict(view.fit(b['family']),origin='tree_bank',config=b['config'],native_family=b['family']),train,valid) for b in native]
        primary=next(p for p in fitted if p['config']=='raw_RX75_0.16')
        starts=[dict(name='primary',family=primary['native_family'])]
        options=[]
        for i,point in enumerate(fitted):
            cut=contract(point,calibration)
            options.append(measure(dict(view.fit(cut['supports']),source_index=i),train,valid))
        seen={canonical(primary['native_family'])}
        for index,point in sorted(enumerate(options),key=lambda t:(t[1]['validation_mae'],len(t[1]['supports']),t[0])):
            family=canonical(point['supports'])
            if family in seen:continue
            seen.add(family);starts.append(dict(name=f'multistart_{len(starts)-1}',family=family,source_index=index))
            if len(starts)==3:break
        result.update(status='complete',fitted_bank=fitted,starts=starts,
            selected=package(select(fitted,calibration),result['terminal_labels'],train,valid))
    except Exception as error:
        result.update(status='unresolved',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        result['LP_solves']=[] if lp is None else lp.solves
        result['seconds']=perf_counter()-started
        if lp is not None:lp.close()
        write(out,result)
    return result


def prepare_history(packet,settings,budget,out):
    """Historical supports/starts are retained only for the old-pool ablation."""
    return prepare_observed_state(packet,settings,budget,out,context='historical_before_new_control')


def _run_method(name,history,train,valid,pool,settings,budget,*,families=None,direct_family=None,day_ids=None,current_state=None):
    from research_experiments.wzzt_dr_selection.dr_pool_v2.engine import PoolLP,search,build_bank
    from research_experiments.wzzt_dr_selection.dr_pool_v2.selection import measure,select,package
    lp=None;started=perf_counter();result=dict(status='running',method=name,proposal_pool=canonical(pool),paths={})
    try:
        budget.check();lp=PoolLP(train,pool,'raw',bound=2.);view=SearchView(lp,pool,budget)
        candidates=[]
        if name=='current_rebuilt_rnj':
            if current_state is None:raise ValueError('current bank must be shared, not rebuilt once per method')
            result.update(current_dense_fronts=current_state['fronts'],
                current_geometry_calibrations=current_state['original_calibrations'],
                output_calibration='one current-observed raw calibration shared by every method',
                bank_rebuilt_inside_method=False)
            for tree in current_state['native_bank']:
                candidates.append(measure(dict(view.fit(tree['family']),origin='current_native_RNJ',config=tree['config']),train,valid))
        if name in ('old_pool_wzzt','current_pool_wzzt','expanded_pool_wzzt','historical_rnj_refit'):
            for tree in history['native_bank']:
                candidates.append(measure(dict(view.fit(tree['family']),origin='historical_native_RNJ',config=tree['config']),train,valid))
        if name in ('old_pool_wzzt','current_pool_wzzt','expanded_pool_wzzt'):
            for start in history['starts']:
                path=search(view,start['family'],seconds=float(settings['search_seconds']),repair=True,max_steps=3*lp.n)
                result['paths'][start['name']]=path
                candidates.extend(measure(dict(p,origin=start['name'],path_index=i),train,valid) for i,p in enumerate(path['path']))
        if families is not None:
            for i,family in enumerate(families):
                candidates.append(measure(dict(view.fit(family),origin=name,family_index=i),train,valid))
        if direct_family is not None:
            candidates.append(measure(dict(view.fit(direct_family),origin='control_direct_tree'),train,valid))
        if not candidates:raise RuntimeError('method has no data-derived complete family')
        result.update(status='complete',points=candidates,
            selected=package(select(candidates,history['calibration']),history['terminal_labels'],train,valid),
            all_scans_complete=all(path['reason']=='pool_local_optimum' for path in result['paths'].values()))
    except Exception as error:
        result.update(status='unresolved',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        result['LP_solves']=[] if lp is None else lp.solves
        result['seconds']=perf_counter()-started
        if lp is not None:lp.close()
    return result


def identify_snapshot(history_packet,future_packet,history_state,discovery,settings,budget,out):
    """No true family/parameter input; all methods get the same observed data."""
    training=history_packet['training']+future_packet['training'];validation=history_packet['validation']+future_packet['validation']
    train,valid,days,hashes,counts=_observations(training,validation);n=len(train[0].columns)
    if list(train[0].columns)!=history_state['terminal_labels']:raise ValueError('history/current terminal order changed')
    if n>settings['max_n']:raise ValueError('snapshot exceeds registered gate size')
    proposals=validate_supports(discovery.get('candidate_supports',[]),n)
    old=validate_supports(history_state['old_pool'],n)
    current_file=Path(out).with_name(Path(out).stem+'_current_state.json')
    current=prepare_observed_state(dict(training=training,validation=validation),settings,budget,current_file,
                                   context='all_current_observed_training')
    if current['status']!='complete':
        result=dict(status='snapshot_state_unresolved',input_sha256=hashes,
            current_state_path=str(current_file),current_state_status=current['status'],
            methods={},method_statuses={'not_started_current_state_unresolved':8},
            truth_input_allowed=False)
        write(out,result);return result
    current_pool=validate_supports(current['pool'],n)
    expanded=canonical([*current_pool,*proposals]);universe=canonical([(j,) for j in range(n)]+list(internal_universe(n)))
    if set(expanded)-set(universe):raise ValueError('unexpected support outside complete small-system universe')
    result=dict(status='running',started_utc=now(),input_sha256=hashes,day_counts=counts,
        terminal_labels=list(train[0].columns),training_rows=len(train[0]),validation_rows=len(valid[0]),
        historical_calibration=history_state['calibration'],applied_calibration=current['calibration'],
        calibration_snapshot='current_observed',current_state_path=str(current_file),discovery=discovery,
        old_pool=old,current_pool=current_pool,expanded_pool=expanded,
        new_supports=canonical(set(expanded)-set(current_pool)),
        new_supports_vs_historical=canonical(set(expanded)-set(old)),
        full_legal_family_count=len(legal_families(n)),same_pool_family_count=len(legal_families(n,expanded)),methods={},
        truth_input_allowed=False)
    historical_with_current_cal=dict(history_state,calibration=current['calibration'])
    specifications=[('historical_rnj_refit',old,None,None),
        ('current_rebuilt_rnj',current_pool,None,None),
        ('old_pool_wzzt',old,None,None),('current_pool_wzzt',current_pool,None,None),
        ('expanded_pool_wzzt',expanded,None,None),
        ('same_pool_tree_enumeration',expanded,legal_families(n,expanded),None),
        ('all_legal_family_reference',universe,legal_families(n),None)]
    direct=discovery.get('direct_family')
    if direct is not None:
        try:
            # A balanced code may not observe the full-root component. Permit
            # it with its original zero lower bound instead of denying the
            # simple direct-tree baseline a nuisance block WZZT can fit.
            direct=validate_supports([*direct,*[(j,) for j in range(n)],tuple(range(n))],n,family=True)
            result['direct_tree_full_root_support_permitted']=True
            specifications.append(('direct_control_tree_refit',canonical([*universe,*direct]),None,direct))
        except ValueError as error:
            result['methods']['direct_control_tree_refit']=dict(status='invalid_discovered_family',error=str(error),no_silent_repair=True,LP_solves=[])
    else:result['methods']['direct_control_tree_refit']=dict(status='no_direct_family',LP_solves=[])
    for name,pool,families,direct in specifications:
        try:budget.check()
        except RuntimeError as error:
            result['methods'][name]=dict(status='not_started_budget',error=str(error),LP_solves=[]);continue
        method_state=historical_with_current_cal if name in ('historical_rnj_refit','old_pool_wzzt') else current
        result['methods'][name]=_run_method(name,method_state,train,valid,pool,settings,budget,
            families=families,direct_family=direct,day_ids=days,current_state=current)
        write(out,result)
    result.update(status='complete_record',ended_utc=now(),
        method_statuses=dict(Counter(m['status'] for m in result['methods'].values())),
        full_price_method_implemented=False,global_topology_probability_claim=False,
        separate_method_LP_contexts=True,shared_snapshot_preprocessing_seconds=current['seconds'],
        per_method_seconds_exclude_shared_preprocessing=True,shared_cross_method_LP_cache=False)
    write(out,result);return result


def run(root):
    root=Path(root);cfg=verify(root)
    if (root/'run.lock').exists():raise FileExistsError('gate already running or interrupted; inspect instead of retry')
    if (root/'summary.json').exists():raise FileExistsError('completed/partial gate retained; do not restart')
    write(root/'run.lock',dict(pid=os.getpid(),started_utc=now()))
    settings=cfg['settings'];budget=Budget(settings['wall_seconds'],settings['max_LP_calls'])
    histories={};records=[]
    try:
        for row in cfg['histories']:
            packet=None
            try:
                budget.check();packet=_load_packet(row['path'])
                state=prepare_history(packet,settings,budget,root/'history'/f"{row['id']}.json")
            except Exception as error:
                state=dict(status='unresolved',error_type=type(error).__name__,error=str(error),
                           traceback=traceback.format_exc(),LP_solves=[])
                write(root/'history'/f"{row['id']}.json",state)
            histories[row['id']]=(packet,state)
        for row in cfg['snapshots']:
            try:
                old,state=histories[row['history_id']]
                if state['status']!='complete':
                    record=dict(id=row['id'],status='history_unresolved',history_id=row['history_id'])
                else:
                    future=_load_packet(row['path'],future=True)
                    discovery=read(row['discovery']['path']) if row.get('discovery') else dict(status='no_control_discovery',candidate_supports=[],direct_family=None)
                    if discovery.get('true_structure_used',False):raise ValueError('truth-derived candidate input forbidden')
                    result=identify_snapshot(old,future,state,discovery,settings,budget,root/'records'/f"{row['id']}.json")
                    record=dict(id=row['id'],status=result['status'],history_id=row['history_id'],
                                result_path=str(root/'records'/f"{row['id']}.json"),method_statuses=result['method_statuses'])
            except Exception as error:
                record=dict(id=row['id'],status='snapshot_unresolved',history_id=row['history_id'],
                            error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
                write(root/'records'/f"{row['id']}.json",record)
            records.append(record);write(root/'records.json',records)
            print(json.dumps(dict(id=row['id'],status=record['status'],processed=len(records),real_LP_calls=budget.calls)),flush=True)
    finally:
        summary=dict(status='finished_bounded_run',planned_records=len(cfg['snapshots']),processed_records=len(records),
            record_statuses=dict(Counter(r['status'] for r in records)),real_LP_calls=budget.calls,
            seconds=perf_counter()-budget.started,truth_evaluation_not_performed=True,
            source_sha256=cfg['source_sha256'],environment=dict(python=sys.version,platform=platform.platform()))
        write(root/'summary.json',summary);(root/'run.lock').unlink(missing_ok=True)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':
        if a.config is None:p.error('--config required for prepare')
        result=prepare(a.root,a.config)
    else:result=run(a.root)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','planned_records','processed_records','real_LP_calls','seconds')}),flush=True)

if __name__=='__main__':main()
