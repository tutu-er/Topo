"""Independent confirmation audit: no confirm/selector/project imports.

All structure selectors, matrix sums, loss scores, contract accounting, and
random-stream replays are implemented directly below. No optimizer/AC/future
generation is executed. Targets enter only the final evaluation step.
"""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
from collections import Counter,defaultdict
from itertools import combinations,permutations
from random import Random
from copy import deepcopy
import json,pickle,sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'wzzt_b_confirm_v1'
METHODS=('current_rebuilt_rnj','current_pool_wzzt','current_root_augmented_rnj')
RULES=('original_3pct_contract','strict_raw_or_contract')
POLICIES=('zero','single2','rotation','pair1','pair_rotation')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'))
def hvalue(v):return sha256(canonical(v).encode()).hexdigest()
def matrix(p):
    R=np.zeros((3,3));X=np.zeros((3,3))
    for s,r,x in zip(p['supports'],p['r'],p['x'],strict=True):
        z=np.zeros(3);z[s]=1.;zz=np.outer(z,z);R+=r*zz;X+=x*zz
    return R,X
def effective(p):return {tuple(s) for s,r,x in zip(p['supports'],p['r'],p['x'],strict=True) if max(r,x)>1e-9}
def loss(p,data):
    P,Q,Y=data;R,X=matrix(p);return float(np.mean(np.abs(P@R.T+Q@X.T-Y)))
def contracted(p,cal):
    keep=[i for i,(s,r,x) in enumerate(zip(p['supports'],p['r'],p['x'],strict=True)) if len(s)==1 or cal['a']*r+cal['b']*x+1e-10>=.16*cal['median_root_depth']]
    out=deepcopy(p)
    for k in ('supports','r','x'):out[k]=[p[k][i] for i in keep]
    return out
def choose(scored,rule):
    if rule==RULES[0]:
        best=min(s['raw_val'] for s in scored);eligible=[i for i,s in enumerate(scored) if s['raw_val']<=1.03*best+1e-12]
        i=min(eligible,key=lambda j:(len(scored[j]['raw']['supports']),scored[j]['raw_val'],j))
        return dict(point=scored[i]['contract'],validation_mae=scored[i]['contract_val'],source_index=i,variant='contract')
    pool=[(i,v,s[v+'_val'],s[v]) for i,s in enumerate(scored) for v in ('raw','contract')]
    best=min(p[2] for p in pool);eligible=[p for p in pool if p[2]<=best+1e-12]
    i,v,score,p=min(eligible,key=lambda q:(len(q[3]['supports']),q[0],int(q[1]=='contract')))
    return dict(point=p,validation_mae=score,source_index=i,variant=v)
def data(scenes):
    return tuple(np.concatenate(v) for v in ([np.asarray(s['P_terminal'],float) for s in scenes],
        [np.asarray(s['Q_terminal'],float) for s in scenes],
        [np.asarray(s['V_root_observed'],float)[:,None]**2-np.asarray(s['V_terminal'],float)**2 for s in scenes]))
def measures(p,target):
    chosen=effective(p);target=set(map(tuple,target));missing=target-chosen;extra=chosen-target
    ans=dict(errors=len(missing)+len(extra),missing=len(missing),extra=len(extra),full=int(not missing and not extra),false_positive=int(bool(extra)),chosen=sorted(chosen))
    for label,k in (('leaf',1),('proper',2),('root',3)):
        ans[label+'_missing']=sum(len(s)==k for s in missing);ans[label+'_extra']=sum(len(s)==k for s in extra)
        ans[label+'_errors']=ans[label+'_missing']+ans[label+'_extra']
    return ans
METRICS=('errors','missing','extra','full','false_positive','leaf_missing','leaf_extra','leaf_errors','proper_missing','proper_extra','proper_errors','root_missing','root_extra','root_errors')

def catalog():
    choices={}
    for i in range(3):
        choices[('single2',f'single_{i}_p2')]=dict(id=f'single_{i}_p2',kind='single2',price_rows=[[2 if k==i else 0 for k in range(3)]]*2,pair_rows=[[i,(i+1)%3]]*2)
    for i,j in permutations(range(3),2):
        name=f'rotation_{i}_{j}_p2';choices[('rotation',name)]=dict(id=name,kind='rotation',price_rows=[[2 if k==v else 0 for k in range(3)] for v in (i,j)],pair_rows=[[v,(v+1)%3] for v in (i,j)])
    for i,j in combinations(range(3),2):
        name=f'pair_{i}_{j}_p1';choices[('pair1',name)]=dict(id=name,kind='pair1',price_rows=[[int(k in (i,j)) for k in range(3)]]*2,pair_rows=[[i,j]]*2)
    for p,q in permutations(tuple(combinations(range(3),2)),2):
        name=f'pair_rotation_{p[0]}_{p[1]}__{q[0]}_{q[1]}_p1';choices[('pair_rotation',name)]=dict(id=name,kind='pair_rotation',price_rows=[[int(k in pair) for k in range(3)] for pair in (p,q)],pair_rows=[list(p),list(q)])
    return choices

def run():
    if not (ROOT/'run_ended.json').exists():raise RuntimeError('wait for run_ended; audit does not resume experiment')
    if (HERE/'summary.json').exists() and not ((HERE/'format_failed_v1.zip').exists() and read(HERE/'summary.json')['status']=='failed'):
        raise FileExistsError('audit already completed; retain original')
    inputs=[ROOT/n for n in ('protocol.json','protocol.sha256.json','actions_frozen.json','run_started.json','run_ended.json','evaluations.json','summary.json')]
    write(HERE/'protocol.json',dict(status='frozen_before_independent_replay',created_utc=datetime.now(timezone.utc).isoformat(),
        input_sha256={str(p):sha(p) for p in inputs},audit_source_sha256=sha(__file__),
        scope='verify fixed70 confirmation, no new fitting/simulation or imports from confirm/selector',
        selectors='independent implementation of original3%+.16 and strict raw/contract 1e-12 tie rules',
        no_new_LP=0,no_AC=0,no_future=True,prior_test_count_reused=0))
    cfg=read(ROOT/'protocol.json');actions=read(ROOT/'actions_frozen.json');started=read(ROOT/'run_started.json');ended=read(ROOT/'run_ended.json');rows=read(ROOT/'evaluations.json');stored_summary=read(ROOT/'summary.json')
    checks=[]
    def check(label,ok,**extra):checks.append(dict(label=label,ok=bool(ok),**extra))
    def close(a,b,tol=1e-12):return np.allclose(a,b,rtol=0,atol=tol)
    check('protocol_hash',sha(ROOT/'protocol.json')==read(ROOT/'protocol.sha256.json')['sha256']==started['protocol_sha256'])
    check('actions_hash',sha(ROOT/'actions_frozen.json')==started['actions_sha256'])
    check('before_future_flags',actions['truth_read'] is False and actions['actual_future_generated'] is False)
    check('protocol_predates_run',datetime.fromisoformat(cfg['created_utc'])<=datetime.fromisoformat(started['created_utc']))
    check('all_source_hashes',all(sha(p)==h for p,h in cfg['source_sha256'].items()))
    check('source_snapshot',sha(ROOT/'source_snapshot.zip')==cfg['source_snapshot_sha256'])
    for k in ('history','history_state','rules_source','old_settings_protocol'):
        check('input_hash_'+k,sha(cfg[k]['path'])==cfg[k]['sha256'])
    check('old_natural_and_noise',cfg['seeds']['natural']==884001 and cfg['regime']['pq_relative']==.005 and cfg['regime']['voltage_relative']==.0002)
    check('two_new_streams',len(cfg['flows'])==2 and {f['behavior'] for f in cfg['flows']}=={934031,934032} and {f['measurement'] for f in cfg['flows']}=={934131,934132})
    check('same_historical_rows',cfg['historical_rows']==[288,288] and cfg['actual_rows']==[480,480])
    check('fixed_budget',cfg['budget']==1.6)
    check('action_protocol_binding',actions['actions']==cfg['actions'] and actions['ordered_jobs']==cfg['ordered_jobs'])
    menus=catalog();byflow={a['flow']['id']:a for a in actions['actions']}
    for a in actions['actions']:
        f=a['flow'];check(f['id']+'_zero_action',a['menus']['zero']['id']=='zero')
        for i,policy in enumerate(POLICIES[1:]):
            available=[m for (kind,_),m in menus.items() if kind==policy]
            prefix='wzzt-b-baseline' if policy=='pair_rotation' else 'wzzt-baseline'
            seed=f['baseline_seed']+i;named=int.from_bytes(sha256(f'{prefix}|{policy}|{seed}'.encode()).digest(),'big')
            expected=available[Random(named).randrange(len(available))];m=a['menus'][policy]
            check(f['id']+'_'+policy+'_pretruth_seed_choice',all(m[k]==expected[k] for k in ('id','kind','price_rows','pair_rows')))
    jobs=[dict(flow=a['flow']['id'],case=c['id'],policy=p) for a in cfg['actions'] for c in cfg['cases'] for p in POLICIES]
    Random(cfg['order_seed']).shuffle(jobs)
    check('job_order_predeclared',jobs==cfg['ordered_jobs'] and len(jobs)==70 and len({tuple(j.values()) for j in jobs})==70)
    check('all70_rows_recorded',len(rows)==70 and [tuple(r[k] for k in ('flow','case','policy')) for r in rows]==[tuple(j[k] for k in ('flow','case','policy')) for j in jobs])
    check('all70_complete',ended['status']=='all_complete' and all(r['execution_status']=='executed' and all(r['evaluated'][m][q]['status']=='complete' for m in METHODS for q in RULES) for r in rows))

    histories=pickle.loads(Path(cfg['history']['path']).read_bytes());fitrows={r['fit_id']:r for r in rows};fit_choices={};all_point_scores={};LP=0;LP_status=Counter();rawpoint_count=0;valdiff=0.;traindiff=0.;variantcounts=Counter();pathaccepted=0;pathrejected=0
    for fid,row in fitrows.items():
        recordpath=ROOT/'records'/(fid+'.json');rec=read(recordpath);receipt=read(row['execution_receipt']);h=receipt['packet_sha256'];packetpath=ROOT/'private_observations'/'packets'/(h+'.pkl')
        check(fid+'_fit_id_binding',fid=='s_'+h[:20] and sha(packetpath)==h)
        future=pickle.loads(packetpath.read_bytes());tr=data(histories['training']+future['training']);va=data(histories['validation']+future['validation'])
        check(fid+'_merged_shapes',tr[0].shape==(480,3) and va[0].shape==(480,3))
        cal=rec['applied_calibration'];fit_choices[fid]={};all_point_scores[fid]={}
        check(fid+'_all_methods_complete',rec['all_methods_complete'] and rec['all_W_scans_complete'])
        for r in rec.get('shared_preprocessing_LP_solves',[]):LP+=1;LP_status[r['status']]+=1
        for method in METHODS:
            m=rec['methods'][method];scored=[]
            for i,p in enumerate(m['points']):
                q=contracted(p,cal);rv=loss(p,va);cv=loss(q,va);rt=loss(p,tr);ct=loss(q,tr)
                valdiff=max(valdiff,abs(rv-p['validation_mae']));traindiff=max(traindiff,abs(rt-p['objective']));rawpoint_count+=1
                check(fid+'_'+method+f'_point{i}_bounds',all(0<=v<=2 for v in p['r']+p['x']))
                sets=list(map(set,p['supports']));check(fid+'_'+method+f'_point{i}_laminar',all(not(a&b) or a<=b or b<=a for a,b in combinations(sets,2)))
                check(fid+'_'+method+f'_point{i}_score_replay',close(rv,p['validation_mae']) and close(rt,p['objective']))
                scored.append(dict(raw=p,contract=q,raw_val=rv,contract_val=cv,raw_train=rt,contract_train=ct))
                stored=rec['selection_comparison'][method]['candidate_validation_losses'][i]
                check(fid+'_'+method+f'_point{i}_comparison_scores',close(stored['raw'],rv) and close(stored['contract'],cv))
            fit_choices[fid][method]={q:choose(scored,q) for q in RULES};all_point_scores[fid][method]=scored
            sel=m['selected'];actual=dict(supports=sel['support_indices'],r=sel['r_values'],x=sel['x_values']);aR,aX=matrix(actual)
            choice=fit_choices[fid][method][RULES[0]];bR,bX=matrix(choice['point'])
            check(fid+'_'+method+'_original_output_matrix_replay',np.array_equal(aR,bR) and np.array_equal(aX,bX))
            for rule in RULES:
                ch=fit_choices[fid][method][rule];claimed=rec['selection_comparison'][method]['choices'][rule];R,X=matrix(ch['point']);R0,X0=matrix(claimed['point'])
                check(fid+'_'+method+'_'+rule+'_independent_selector',ch['source_index']==claimed['source_index'] and ch['variant']==claimed['variant'] and np.array_equal(R,R0) and np.array_equal(X,X0) and close(ch['validation_mae'],claimed['validation_mae']))
                if rule==RULES[1]:variantcounts[ch['variant']]+=1
            solves=m.get('LP_solves',[]);LP+=len(solves)
            for j,r in enumerate(solves):
                LP_status[r['status']]+=1
                check(fid+'_'+method+f'_LP{j}_status',r['status']==2)
                if 'physical' in r:check(fid+'_'+method+f'_LP{j}_reported_objective',close(r['objective'],r['physical'],1e-8))
                if 'max_constraint_violation' in r:check(fid+'_'+method+f'_LP{j}_reported_violation',r['max_constraint_violation']<=1e-7 and r['max_bound_violation']<=1e-7)
            paths=m.get('paths',{});paths=list(paths.values()) if isinstance(paths,dict) else paths
            for j,path in enumerate(paths):
                if not isinstance(path,dict) or 'path' not in path:continue
                pp=path['path']
                for i,decision in enumerate(path['decisions']):
                    current=decision['current'];gain=decision['gain'];accept=gain>1e-10+.001*current
                    check(fid+'_'+method+f'_path{j}_decision{i}_current',close(current,pp[min(i,len(pp)-1)]['objective']))
                    if i<len(pp)-1:
                        pathaccepted+=1;check(fid+'_'+method+f'_path{j}_accepted{i}',accept and close(current-pp[i+1]['objective'],gain))
                    else:
                        pathrejected+=1;check(fid+'_'+method+f'_path{j}_stopped{i}',not accept or path['reason'] in ('pool_wall_limit','step_limit'))
                    for c in decision['candidates']:
                        removed=list(map(set,c['removed']));C=set(c['candidate'])
                        check(fid+'_'+method+f'_path{j}_decision{i}_crossing',all(s&C and not s<=C and not C<=s for s in removed))
    check('LP_total_matches',LP==ended['LP_calls']==stored_summary['LP_calls']==592)
    check('LP_budget_respected',LP<=cfg['computation']['max_LP_calls'])
    check('unique54_and_cache16',len(fitrows)==54==stored_summary['different_observations'] and sum(r.get('reused_observed_fit',False) for r in rows)==70-len(fitrows))

    probs=np.array([0.,.5,.75]);wave=np.r_[np.full(4,-.2),np.full(4,.2)];reuse=defaultdict(list);fee={};receiptmeta=[]
    evaluated=[];mechanisms=[]
    for row in rows:
        flow=byflow[row['flow']]['flow'];menu=byflow[row['flow']]['menus'][row['policy']];receipt=read(row['execution_receipt']);ctl=receipt['controls']
        P=np.array(ctl['prices']);Z=np.array(ctl['accepted']);S=np.array(ctl['orientation']);pairs=np.array(ctl['pairs']);dp=np.array(ctl['delta_p_kw']);dq=np.array(ctl['delta_q_kvar'])
        expectedP=np.array(menu['price_rows'])[np.arange(4)%2];expected_pairs=np.array(menu['pair_rows'])[np.arange(4)%2];expectedZ=[];expectedS=[]
        for e in range(4):
            expectedZ.append(np.random.default_rng(np.random.SeedSequence([flow['behavior'],76001,e])).random(3)<probs[expectedP[e]])
            expectedS.append(1 if np.random.default_rng(np.random.SeedSequence([flow['behavior'],76002,e])).integers(0,2) else -1)
        expected_dp=np.zeros((4,8,3))
        for e,(a,b) in enumerate(expected_pairs):expected_dp[e,:,a]=expectedS[e]*wave*expectedZ[e][a];expected_dp[e,:,b]=-expectedS[e]*wave*expectedZ[e][b]
        prefix=row['flow']+'_'+row['case']+'_'+row['policy'];pay=.2*np.sum(expectedP*np.array(expectedZ),axis=1)
        check(prefix+'_frozen_menu',receipt['menu']==menu and row['menu_id']==menu['id'])
        check(prefix+'_prices_pairs',np.array_equal(P,expectedP) and np.array_equal(pairs,expected_pairs))
        check(prefix+'_Z_and_direction',np.array_equal(Z,expectedZ) and np.array_equal(S,expectedS))
        check(prefix+'_waveform',close(dp,expected_dp) and close(dq,.35*expected_dp))
        power=.2+dp;energy=np.cumsum(power,axis=1)*.25
        check(prefix+'_power_energy',np.min(power)>=-1e-12 and np.max(power)<=.4+1e-12 and np.min(energy)>=-1e-12 and np.max(energy)<=.4+1e-12 and close(energy[:,-1],.4))
        check(prefix+'_net_task',close(dp.sum(axis=1)*.25,0))
        check(prefix+'_payment',close(pay,ctl['committed_payment']) and close(pay.sum(),receipt['actual_payment']) and close(pay.sum(),row['actual_payment']))
        check(prefix+'_worst_budget',close(.4*np.array(menu['price_rows']).sum(),receipt['worst_payment']) and pay.sum()<=receipt['worst_payment']+1e-12 and receipt['worst_payment']<=1.6)
        check(prefix+'_seed_time_binding',receipt['behavior_seed']==flow['behavior'] and receipt['measurement_seed']==flow['measurement'] and receipt['natural_seed']==884001 and datetime.fromisoformat(receipt['created_utc'])>=datetime.fromisoformat(started['created_utc']))
        check(prefix+'_action_binding',receipt['actions_sha256']==started['actions_sha256'])
        ident=dict(P=P.tolist(),Z=Z.astype(int).tolist(),S=S.tolist(),pairs=pairs.tolist(),dp=dp.tolist(),dq=dq.tolist(),pay=pay.tolist());g=(row['flow'],row['policy']);reuse[g].append((row['case'],hvalue(ident)));fee[g]=float(pay.sum())
        receiptmeta.append(dict(name=Path(row['execution_receipt']).name,sha256=sha(row['execution_receipt']),flow=row['flow'],case=row['case'],policy=row['policy'],packet_sha256=receipt['packet_sha256']))
        case=next(c for c in cfg['cases'] if c['id']==row['case']);target={(0,),(1,),(2,),(0,1,2)}
        if case['c']>0:target.add(tuple(case['pair']))
        for method in METHODS:
            choices=fit_choices[row['fit_id']][method];scores=all_point_scores[row['fit_id']][method]
            for rule in RULES:
                chosen=choices[rule];metric=measures(chosen['point'],target);claimed=row['evaluated'][method][rule]
                check(prefix+'_'+method+'_'+rule+'_target',set(map(tuple,claimed['target']))==target)
                check(prefix+'_'+method+'_'+rule+'_evaluation',metric['errors']==claimed['final_errors'] and set(map(tuple,metric['chosen']))==set(map(tuple,claimed['supports'])) and chosen['source_index']==claimed['source_index'] and close(chosen['validation_mae'],claimed['validation_mae']))
                evaluated.append(dict(flow=row['flow'],case=row['case'],group='zero' if case['c']==0 else 'weak' if case['c']==.02 else 'strong',policy=row['policy'],method=method,rule=rule,variant=chosen['variant'],source_index=chosen['source_index'],**metric))
            original_raw=scores[choices[RULES[0]]['source_index']]['raw'];original_final=choices[RULES[0]]['point'];strict=choices[RULES[1]]['point'];best=min(v['raw_val'] for v in scores)
            correct=[v for v in scores if measures(v['raw'],target)['errors']==0]
            mechanisms.append(dict(flow=row['flow'],case=row['case'],policy=row['policy'],method=method,
                correct_raw_saved=bool(correct),correct_raw_in_3pct=any(v['raw_val']<=1.03*best+1e-12 for v in correct),
                original_raw_correct=measures(original_raw,target)['errors']==0,original_final_correct=measures(original_final,target)['errors']==0,
                strict_correct=measures(strict,target)['errors']==0,original_raw_to_final_removed=sorted(effective(original_raw)-effective(original_final))))
    for (flow,policy),values in reuse.items():check(flow+'_'+policy+'_seven_counterfactual_controls',len(values)==7 and len({v[0] for v in values})==7 and len({v[1] for v in values})==1)
    idx={(r['flow'],r['case'],r['policy'],r['method'],r['rule']):r for r in evaluated}
    groups=[]
    for m in METHODS:
        for rule in RULES:
            for policy in ('all',*POLICIES):
                for group in ('all','zero','weak','strong'):
                    rr=[r for r in evaluated if r['method']==m and r['rule']==rule and (policy=='all' or r['policy']==policy) and (group=='all' or r['group']==group)]
                    groups.append(dict(method=m,rule=rule,policy=policy,group=group,n=len(rr),**{k:sum(r[k] for r in rr) for k in METRICS}))
    paired=[];flow_policies=[];cross_method=[]
    for method in METHODS:
        for rule in RULES:
            for a,b in [('single2','zero'),('rotation','zero'),('pair1','zero'),('pair_rotation','zero'),('pair_rotation','single2'),('pair_rotation','rotation'),('pair_rotation','pair1')]:
                cc=Counter();delta=0
                for flow in cfg['flows']:
                    for case in cfg['cases']:
                        x=idx[(flow['id'],case['id'],a,method,rule)];y=idx[(flow['id'],case['id'],b,method,rule)];d=x['errors']-y['errors'];delta+=d;cc['better' if d<0 else 'worse' if d>0 else 'same']+=1
                paired.append(dict(method=method,rule=rule,policy_a=a,policy_b=b,n=14,counts=dict(cc),error_delta=delta))
            for flow in cfg['flows']:
                for policy in POLICIES:
                    rr=[r for r in evaluated if r['flow']==flow['id'] and r['policy']==policy and r['method']==method and r['rule']==rule]
                    flow_policies.append(dict(flow=flow['id'],policy=policy,method=method,rule=rule,n=7,actual_payment=fee[(flow['id'],policy)],**{k:sum(r[k] for r in rr) for k in METRICS}))
    for rule in RULES:
        for paid_only in (False,True):
            cc=Counter();delta=0;identical=[]
            for row in rows:
                if paid_only and row['policy']=='zero':continue
                a=idx[(row['flow'],row['case'],row['policy'],'current_pool_wzzt',rule)];b=idx[(row['flow'],row['case'],row['policy'],'current_root_augmented_rnj',rule)]
                d=a['errors']-b['errors'];delta+=d;cc['W_better' if d<0 else 'W_worse' if d>0 else 'same']+=1
                identical.append(a['chosen']==b['chosen'])
            cross_method.append(dict(rule=rule,paid_only=paid_only,n=len(identical),identical_supports=sum(identical),counts=dict(cc),W_minus_rootaug_errors=delta))
    mechanism_groups=[]
    for method in METHODS:
        rr=[r for r in mechanisms if r['method']==method]
        mechanism_groups.append(dict(method=method,n=len(rr),correct_raw_saved=sum(r['correct_raw_saved'] for r in rr),correct_raw_in_3pct=sum(r['correct_raw_in_3pct'] for r in rr),
            original_raw_correct=sum(r['original_raw_correct'] for r in rr),original_final_correct=sum(r['original_final_correct'] for r in rr),strict_correct=sum(r['strict_correct'] for r in rr),
            raw_correct_lost_in_contract=sum(r['original_raw_correct'] and not r['original_final_correct'] for r in rr),
            original_raw_wrong_strict_correct=sum(not r['original_raw_correct'] and r['strict_correct'] for r in rr)))
    for s in stored_summary['results']:
        g=next(g for g in groups if g['method']==s['method'] and g['rule']==s['rule'] and g['policy']==s['policy'] and g['group']=='all')
        check('summary_'+s['method']+'_'+s['rule']+'_'+s['policy'],g['n']==s['complete'] and g['errors']==s['final_errors'] and g['full']==s['full'])
    for f in cfg['flows']:
        for policy in POLICIES:check(f['id']+'_'+policy+'_summary_payment',close(fee[(f['id'],policy)],stored_summary['payments_per_four_event_deployment'][f['id']+'|'+policy]))
    failures=[c for c in checks if not c['ok']]
    summary=dict(status='passed' if not failures else 'failed',created_utc=datetime.now(timezone.utc).isoformat(),audit_source_sha256=sha(__file__),
        protocol_sha256=sha(ROOT/'protocol.json'),actions_sha256=sha(ROOT/'actions_frozen.json'),audit_protocol_sha256=sha(HERE/'protocol.json'),
        checks=len(checks),failures=failures,planned_maps=70,complete_maps=len(rows),unique_fits=len(fitrows),cache_reused_rows=70-len(fitrows),
        LP_receipts=LP,LP_status_counts=dict(LP_status),raw_points_replayed=rawpoint_count,raw_and_contract_validation_replays=2*rawpoint_count,
        raw_validation_maxdiff=valdiff,raw_training_maxdiff=traindiff,strict_variant_counts=dict(variantcounts),
        accepted_path_steps=pathaccepted,stopped_path_steps=pathrejected,control_deployments=len(reuse),four_events_per_deployment=True,
        payments=[dict(flow=f,policy=p,payment=v) for (f,p),v in sorted(fee.items())],groups=groups,paired=paired,flow_policies=flow_policies,
        cross_method=cross_method,mechanisms=mechanism_groups,no_new_LP=0,no_AC=0,no_future_generation=True,
        no_confirm_or_selector_import=True,old_tests_not_repeated=True,
        prior_audit_format_failure=dict(archive_sha256=sha(HERE/'format_failed_v1.zip'),
            reason='420 tuple-vs-JSON-list comparisons; all numerical values and normalized supports already agreed; original source/receipts preserved; check count not doubled') if (HERE/'format_failed_v1.zip').exists() else None,
        scope='two new behavior/measurement streams with one old natural-load stream and seven counterfactual models; not70 independent samples; fixed exploration, no new price optimizer')
    write(HERE/'checks.json',checks);write(HERE/'rows.json',evaluated);write(HERE/'mechanism_rows.json',mechanisms);write(HERE/'receipt_hashes.json',receiptmeta)
    write(HERE/'summary.json',summary)
    aggregate={k:v for k,v in summary.items() if k not in ('failures',)};aggregate['failures_count']=len(failures);aggregate['detailed_summary_sha256']=sha(HERE/'summary.json');write(HERE/'aggregate_summary.json',aggregate)
    return summary

if __name__=='__main__':
    s=run();print(json.dumps({k:v for k,v in s.items() if k not in ('groups','paired','flow_policies','mechanisms','cross_method')},ensure_ascii=False))
    sys.exit(0 if s['status']=='passed' else 1)
