"""Three historical conditions, frozen prices, causal delivery, original AC.

No RNJ/WZZT reimplementation or parameter refit: reuse historical candidates
and the existing weighted selection. Whole random paths are paired by policy.
"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,json,pickle,hashlib,sys,time,traceback,zipfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
sys.path.insert(0,str(REPO))
from devices import Devices
from planner import Probability,solve
from research_experiments.wzzt_dr_selection.dr_heterogeneous_resources.build_allocations import profiles
from research_experiments.wzzt_dr_selection.dr_newday_weighted.case_inputs import build_cases
from research_experiments.wzzt_dr_selection.dr_newday_weighted.pool_competition import choose_final_competition
from research_experiments.wzzt_dr_selection.dr_frozen_distance.frozen_pool import freeze_pool,_power,_target
from research_experiments.wzzt_dr_selection.dr_newday_weighted.weighted_selection import score
from research_experiments.wzzt_dr_selection.dr_spatial_correlated.ac_bridge import open_runtime,build_background

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(r):
    p=Path(r['path'])
    if sha(p)!=r['sha256']:raise ValueError('input hash drift: '+str(p))
    return p
def dump(p,v):Path(p).write_bytes(pickle.dumps(v,protocol=4))
def load(p):return pickle.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()


def sources():
    task=REPO/'research_experiments/wzzt_dr_selection'
    source=Path(r'D:/0-github_workspace/Topo/outputs/wzzt_paper_20261029_full/source')
    files=list(Path(__file__).parent.glob('*.py'))+list(source.rglob('*.py'))
    for folder in ['dr_heterogeneous_resources','dr_conditional_gaussian','dr_newday_weighted','dr_frozen_distance','dr_pool_v2','dr_competition_bridge','dr_conic_pricing','end_to_end','dr_spatial_correlated']:
        files+=list((task/folder).rglob('*.py'))
    return sorted(set(p.resolve() for p in files))


def make_packet(case,pool,history,fit,devices):
    labels=pool['labels'];r0=np.asarray(fit['fronts']['raw']['R']);x0=np.asarray(fit['fronts']['raw']['X'])
    if fit['terminal_labels']!=labels:raise ValueError('historical labels mismatch')
    train=[]
    for s in history['training']:
        p,q=_power(s,labels);y=_target(s,p.shape)
        if p.shape!=(96,len(labels)):raise ValueError('complete history required')
        train.append(np.stack([p,q,y-p@r0.T-q@x0.T]))
    a=np.stack(train);out=dict(r0=r0,x0=x0,labels=labels,pool=pool)
    for identity,k in [('coarse','C'),('refined','F')]:
        mod=next(m for m in pool['models'] if m['candidate_id']==identity)
        r,x=np.asarray(mod['R']),np.asarray(mod['X'])
        rr=a[:,2]+np.einsum('btn,jn->btj',a[:,0],r0-r)+np.einsum('btn,jn->btj',a[:,1],x0-x)
        rr=rr.reshape(3,-1)*1e6;mu=rr.mean(0)
        out['mu_'+k]=mu;out['L_'+k]=(rr-mu).T/np.sqrt(2.)
        out['effect_'+k]=np.stack([(r0-r)[:,devices.node[z]]+devices.qratio[z]*(x0-x)[:,devices.node[z]] for z in range(devices.R)],axis=1)*1e6/pool['base_kw']
    h={m['candidate_id']:m['historical_validation_mae']*1e6 for m in pool['models']}
    out['threshold']=1.5*(1.03*h['refined']-h['coarse'])+2.5e-6
    out['eligible']=[k for k in h if h[k]<=1.03*min(h.values())+1e-6]
    out['history_losses_micro']=h
    out['has_dispute']=not pool['competition_rule'].get('no_final_dispute',False) and len(out['eligible'])==2
    return out


def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    lib=REPO/'research_experiments/wzzt_dr_selection/dr_heterogeneous_resources/library.json'
    alloc=lib.with_name('case_allocations.json')
    allcases=build_cases()
    names=['paper15_b2_v0005','flynn16_b4_v0005','pengwah18_b5_v0002']
    selected=[next(c for c in allcases if c['name']==name) for name in names]
    source_files=sources()
    protocol=dict(id='heterogeneous_resource_pilot_v1',status='frozen',created_utc=now(),cases=selected,
        library=ref(lib),allocations=ref(alloc),source_sha256={str(p):sha(p) for p in source_files},
        case_rule='two previously declared historical disagreements plus retained pengwah b5 zero-yield control; no new test outcome filtering',
        full_history_rows=[288,288],resource_domain='full_inventory',new_day_weight=2,new_day_fraction=.03,history_fraction=.03,
        control='all nodes and four resource classes, 24 hourly continuous requests expanded to96 slots; causal deadline/SOC projection',
        background_law='reuse three complete training-day joint Gaussian [P,Q,reference residual]; rank<=2 retained to isolate resource extension; working distribution not calibrated',
        response_law='new library node/time/type continuous correlated multipliers, causal state feasible delivery',
        planning_samples=32,planning_seed=2026110501,response_seed=2026110502,optimizer_seed=2026110503,
        qualification_samples=128,qualification_seed=2026110504,qualification_response_seed=2026110505,
        optimizer=dict(maxiter=18,seconds_per_start=120,starts=2,target=.95),
        primary_policies=['zero','rotation','cost_first','optimized'],
        baseline_rule='two predetermined feasible physical schedules each scaled to equal optimized prepaid fee if attainable; otherwise report saturation and unequal cost',
        no_money_budget_upper=True,matched_legacy_2kw_not_used=True,
        paths=[dict(day=j,profile_index=j%3,replicate=980000+137*j,response_seed=2026110600+j) for j in range(6)],
        laws=['main','stress'],independent_backgrounds_per_case=6,policies=4,expected_records=144,
        truth_gate='prices frozen before backgrounds/AC, selections frozen before true supports evaluated',
        evaluation='new independent synthetic AC backgrounds, original P/Q and V meter noise per historical case; no physics/noise/budget tuning',
        identifier='historical RNJ+WZZT frozen pair/frozen RX, original.16 output mapping and w2 common3% final selection; no new W search',
        interpretation='local finite-integration price design, not global optimum or 95 percent physical recovery guarantee')
    write(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for i,p in enumerate(source_files):z.write(p,f'{i:03d}_{p.name}')
    cfg=read(lib);allocs={c['case']:c for c in read(alloc)['cases']};rows=[]
    for c in selected:
        d=root/c['name'];d.mkdir();write(d/'case_protocol.json',c)
        pool=freeze_pool(d/'case_protocol.json',d/'pool.json',competition_selector=choose_final_competition)
        hist=load(checked(c['history_packet']));history=hist.get('history',hist)
        fit=read(checked(c['historical_baseline_result']))
        dev=Devices(allocs[c['case']],cfg,profiles)
        if dev.labels!=c['labels']:raise ValueError('resource label order mismatch')
        packet=make_packet(c,pool,history,fit,dev);dump(d/'problem.pkl',packet)
        rows.append(dict(condition=c['name'],case=c['case'],has_dispute=packet['has_dispute'],eligible=packet['eligible'],
            threshold_micro=packet['threshold'],history_losses_micro=packet['history_losses_micro'],
            final_dispute=pool['competition_rule']['final_symmetric_difference'],
            historical_selected_origin=pool['competition_rule']['selected_origin'],
            historical_competitor_origin=pool['competition_rule']['alternative_origin'],
            inventory_kw=float(dev.cap.sum()),decision_variables=dev.R*24,problem=ref(d/'problem.pkl'),pool=ref(d/'pool.json')))
    write(root/'screening.json',dict(rows=rows,history_only=True,new_fits=0,truth_read=False))
    write(root/'run_lock.json',dict(status='prepared',pid=None))
    print(json.dumps(rows,ensure_ascii=False),flush=True)


def config(root):
    root=Path(root);p=read(root/'protocol.json')
    for f,h in p['source_sha256'].items():
        if sha(f)!=h:raise ValueError('frozen source changed: '+f)
    cfg=read(checked(p['library']));alloc={c['case']:c for c in read(checked(p['allocations']))['cases']}
    return root,p,cfg,alloc


def simple_plan(dev,name):
    # Prespecified hourly exploration. Cost-first changes ordering only, not
    # a truth/reference residual or an observed future.
    order=np.arange(dev.R)
    if name=='cost_first':order=np.argsort(np.mean(dev.cost_full_activation,axis=1),kind='stable')
    phase=np.zeros(dev.R,int);phase[order]=np.arange(dev.R)%4
    x=np.zeros((dev.R,24))
    for r in range(dev.R):
        x[r]=np.where((np.arange(24)//2+phase[r])%2==0,1.,-1.)
        if name=='cost_first':x[r]*=1./(1.+np.flatnonzero(order==r)[0]/max(dev.R/4,1))
        if dev.kind[r]==1:x[r]=np.minimum(x[r],0.)
    return dev.nominal(x)


def optimize(root):
    root,p,cfg,alloc=config(root)
    write(root/'run_lock.json',dict(status='optimizing',pid=os.getpid()))
    allplans={};summary=[]
    for i,c in enumerate(p['cases']):
        folder=root/c['name'];receipt=folder/'price_receipt.json'
        if receipt.exists():
            saved=read(receipt);checked(saved['plans']);allplans[c['name']]=saved;summary.append(saved['summary']);continue
        packet=load(folder/'problem.pkl');dev=Devices(alloc[c['case']],cfg,profiles)
        if packet['has_dispute']:
            prob=Probability(packet,dev,p['planning_samples'],p['planning_seed']+i,p['response_seed']+i)
            sol=solve(prob,p['optimizer_seed']+i,maxiter=p['optimizer']['maxiter'],seconds=p['optimizer']['seconds_per_start'])
            dump(folder/'solver_private.pkl',sol)
            optimized=sol['best']['mean_u'];planning_probability=sol['best']['probability'];fee=sol['best']['cost']
            qualification=Probability(packet,dev,p['qualification_samples'],p['qualification_seed']+i,p['qualification_response_seed']+i)
            qualification_p=qualification.evaluate(sol['best']['x'])['probability']
            status=[t['status'] for t in sol['traces']]
            trace=[{k:v for k,v in t.items()} for t in sol['traces']]
        else:
            optimized=np.zeros((dev.R,96));planning_probability=None;qualification_p=None;fee=0.;status=['no_historical_dispute_no_purchase'];trace=[]
        plans={'zero':dict(mean_u=np.zeros_like(optimized),cost=0.),'optimized':dict(mean_u=optimized,cost=fee)}
        for name in ['rotation','cost_first']:
            full=simple_plan(dev,name);maximum=dev.fees(full)
            alpha=min(1.,np.sqrt(fee/maximum)) if maximum>0 else 0.
            value=alpha*full
            plans[name]=dict(mean_u=value,cost=dev.fees(value),requested_cost=fee,scale=alpha,full_cost=maximum,
                cost_matched=abs(dev.fees(value)-fee)<1e-7)
        for name,item in plans.items():
            audit=dev.audit(item['mean_u'])
            if not audit['passed']:raise AssertionError('nominal energy infeasible: '+name)
            item['nominal_audit']=audit
        dump(folder/'plans.pkl',plans)
        s=dict(condition=c['name'],planning_probability=planning_probability,independent_working_probability=qualification_p,
            optimizer_status=status,optimized_cost=fee,global_optimal=False,has_dispute=packet['has_dispute'],
            policy_costs={name:v['cost'] for name,v in plans.items()},policy_nominal_audits={name:v['nominal_audit'] for name,v in plans.items()})
        saved=dict(summary=s,plans=ref(folder/'plans.pkl'),solver_trace=trace)
        write(receipt,saved);allplans[c['name']]=saved;summary.append(s)
        print('planned',json.dumps(s,ensure_ascii=False),flush=True)
    write(root/'price_freeze.json',dict(status='frozen',protocol=ref(root/'protocol.json'),cases=allplans,truth_read=False))
    write(root/'planning_summary.json',dict(rows=summary))
    write(root/'run_lock.json',dict(status='prices_frozen',pid=None))


def execute(runtime,bg,dev,u,multiplier,record):
    delivered=dev.project(u*multiplier);audit=dev.audit(delivered)
    if not audit['passed']:raise ArithmeticError('causal delivery violates device domain')
    dp,dq=dev.aggregate(delivered);base=runtime['metadata']['base_kw']
    pf=bg.p_base+dp/base;qf=bg.q_base+dq/base
    sim=runtime['sim'];net=runtime['context'].net;labels=runtime['labels']
    ac=sim.solve_ac_power_flow_timeseries(net,pf,qf,v_root=bg.root_true,max_iter=100,tol=1e-10)
    voltage=ac['V_bus_mag'].loc[:,labels]
    check=runtime['checker'](net,pf,qf,ac,(.9,1.1),1e-8)
    converged=int(np.asarray(ac['converged'],bool).sum())
    summary=dict(status='complete' if converged==96 and check['electrical_equations_pass'] else 'physical_unresolved',
        converged=converged,voltage_min=float(voltage.to_numpy().min()),voltage_max=float(voltage.to_numpy().max()),
        voltage_violation_cells=int(check['voltage_violation_cells']),thermal_limits_checked=False,resource_audit=audit,
        planned_cost=dev.fees(u),recourse_abs_kwh=float(.25*np.abs(delivered-u*multiplier).sum()),
        actual_resource_wear_proxy_fee=dev.fees(delivered),wear_proxy_not_extra_payment=True,
        max_increment_error_kw=float(max(np.max(np.abs((pf-bg.p_base).to_numpy()*base-dp)),np.max(np.abs((qf-bg.q_base).to_numpy()*base-dq)))))
    scene=None
    if summary['status']=='complete':
        measured=sim._observe_terminal_channels(runtime['context'],bg,pf,qf,voltage,pq_noise_rel=runtime['pq_sigma'],v_noise_rel=runtime['voltage_sigma'])
        scene=runtime['legacy'].scene(measured['p_measured'].to_numpy(),measured['q_measured'].to_numpy(),measured['voltage_measured'].to_numpy(),measured['root_observed'].to_numpy(),labels,record,6)
    return dict(summary=summary,scene=scene,private=dict(natural_p=bg.p_base,natural_q=bg.q_base,physical_p=pf,physical_q=qf,
        delivered=delivered,multiplier=multiplier,nominal=u,AC=ac,checks=check,node_delta_p=dp,node_delta_q=dq))


def observe(root):
    root,p,cfg,alloc=config(root);f=read(root/'price_freeze.json');checked(f['protocol'])
    write(root/'run_lock.json',dict(status='observing',pid=os.getpid()));rows=[]
    for ci,c in enumerate(p['cases']):
        dev=Devices(alloc[c['case']],cfg,profiles);plans=load(checked(f['cases'][c['name']]['plans']))
        runtime=open_runtime(c['source_root'],c['source_metadata'],repo_root=REPO,protocol_ref=ref(root/'protocol.json'),labels=c['labels'],case=c['case'],v_sigma=c['voltage_sigma'],pq_sigma=c['pq_sigma'])
        for d in p['paths']:
            bg=build_background(runtime,d['profile_index'],d['replicate']+10000*ci)
            for law in p['laws']:
                multiplier=dev.draw(d['response_seed']+ci*100,1,stress=law=='stress')[0]
                for name in p['primary_policies']:
                    record=f'{c["name"]}_d{d["day"]}_{law}_{name}';folder=root/'actual'/record;folder.mkdir(parents=True,exist_ok=True)
                    receipt=folder/'receipt.json'
                    if receipt.exists():
                        old=read(receipt)
                        if 'packet' in old:checked(old['packet'])
                        rows.append(old);continue
                    row=dict(record_id=record,condition=c['name'],case=c['case'],law=law,policy=name,day=d['day'],fee=plans[name]['cost'])
                    try:
                        result=execute(runtime,bg,dev,plans[name]['mean_u'],multiplier,record)
                        dump(folder/'result.pkl',result)
                        row.update(status=result['summary']['status'],physical=result['summary'],packet=ref(folder/'result.pkl'))
                    except Exception as e:row.update(status='failed',error=repr(e),traceback=traceback.format_exc())
                    write(receipt,row);rows.append(row);write(root/'checkpoint.json',dict(completed=len(rows),planned=p['expected_records'],last=record))
            print('observed',c['name'],d['day'],len(rows),'/',p['expected_records'],flush=True)
    write(root/'observations_frozen.json',dict(rows=rows,price_freeze=ref(root/'price_freeze.json')))
    write(root/'run_lock.json',dict(status='observations_frozen',pid=None))


def evaluate(root):
    root,p,cfg,alloc=config(root);ledger=read(root/'observations_frozen.json');checked(ledger['price_freeze'])
    rows=[]
    for old in ledger['rows']:
        row={k:v for k,v in old.items() if k not in ['traceback','packet']}
        if old['status']=='complete':
            result=load(checked(old['packet']));pool=read(root/old['condition']/'pool.json')
            s=score(pool,result['scene'],history_gate=True)
            chosen=next(r for r in s['rules'] if r['new_day_weight']==2 and r['new_day_fraction']==.03)
            write(root/'scores'/(old['record_id']+'.json'),s)
            loss={r['candidate_id']:r['new_validation_mae'] for r in s['loss_rows']}
            row.update(status='predicted',selected=chosen['selected_candidate_id'],coarse_excluded='coarse' not in chosen['eligible_candidate_ids'],
                D_micro=1e6*(loss['coarse']-1.03*loss['refined']),margin_micro=1e6*chosen['exit_margins']['coarse'],
                coarse_loss_micro=1e6*loss['coarse'],refined_loss_micro=1e6*loss['refined'])
        rows.append(row)
    write(root/'predictions_frozen.json',dict(rows=rows,truth_read=False))
    candidate_errors={}
    for c in p['cases']:
        runtime=open_runtime(c['source_root'],c['source_metadata'],repo_root=REPO,protocol_ref=ref(root/'protocol.json'),labels=c['labels'],case=c['case'],v_sigma=c['voltage_sigma'],pq_sigma=c['pq_sigma'])
        truth=runtime['legacy'].evaluate_supports([],runtime['context'].net)
        write(root/c['name']/'truth_private.json',truth);truthset={tuple(sorted(b)) for b in truth['missed_blocks']}
        pool=read(root/c['name']/'pool.json');errs={}
        for mod in pool['models']:
            raw={tuple(sorted(c['labels'][i] for i in b)) for b in mod['point']['supports']}
            final={tuple(sorted(b)) for b in mod['frozen_output_support_labels']}
            errs[mod['candidate_id']]=dict(raw_errors=len(truthset^raw),errors=len(truthset^final),missing=len(truthset-final),extra=len(final-truthset))
        candidate_errors[c['name']]=errs
    for row in rows:
        if row['status']=='predicted':row.update(status='complete',**candidate_errors[row['condition']][row['selected']])
    tables=[];pairs=[]
    for c in p['cases']:
        for law in p['laws']:
            for name in p['primary_policies']:
                group=[r for r in rows if r['condition']==c['name'] and r['law']==law and r['policy']==name];ok=[r for r in group if r['status']=='complete']
                table=dict(condition=c['name'],law=law,policy=name,denominator=6,completed=len(ok),unresolved=6-len(ok),
                    final_errors=sum(r['errors'] for r in ok),full_recoveries=sum(r['errors']==0 for r in ok),
                    fee_per_day=group[0]['fee'],mean_D_micro=float(np.mean([r['D_micro'] for r in ok])) if ok else None,
                    mean_margin_micro=float(np.mean([r['margin_micro'] for r in ok])) if ok else None,
                    mean_actual_abs_DR_kwh=float(np.mean([r['physical']['resource_audit']['absolute_DR_kwh'] for r in ok])) if ok else None,
                    mean_recourse_abs_kwh=float(np.mean([r['physical']['recourse_abs_kwh'] for r in ok])) if ok else None,
                    coarse_exclusions=sum(r['coarse_excluded'] for r in ok),raw_final_difference_count=sum(r['raw_errors']!=r['errors'] for r in ok))
                tables.append(table)
            for other in ['zero','rotation','cost_first']:
                delta=[]
                for day in range(6):
                    r=[r for r in rows if r['condition']==c['name'] and r['law']==law and r['day']==day]
                    a=next(x for x in r if x['policy']=='optimized');b=next(x for x in r if x['policy']==other)
                    if a['status']==b['status']=='complete':delta.append(a['errors']-b['errors'])
                pairs.append(dict(condition=c['name'],law=law,reference=other,better=sum(x<0 for x in delta),same=sum(x==0 for x in delta),worse=sum(x>0 for x in delta),unresolved=6-len(delta)))
    summary=dict(status='complete',tables=tables,pairs=pairs,candidate_errors=candidate_errors,
        records=len(rows),complete=sum(r['status']=='complete' for r in rows),failed_unresolved=sum(r['status']!='complete' for r in rows),
        independent_backgrounds=18,paired_response_laws=2,resource_violation_records=sum(not r.get('physical',{}).get('resource_audit',{}).get('passed',True) for r in rows),
        voltage_violation_records=sum(r.get('physical',{}).get('voltage_violation_cells',0)>0 for r in rows),
        AC_converged_points=sum(r.get('physical',{}).get('converged',0) for r in rows),
        new_parameter_fits=0,new_W_search=0,scope='frozen historical RNJ+WZZT candidate final selection; effect at validation/output not new discovery')
    write(root/'scored_private.json',dict(rows=rows));write(root/'summary.json',summary)
    write(root/'ended.json',dict(status='complete',summary=ref(root/'summary.json')));write(root/'run_lock.json',dict(status='complete',pid=None))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['prepare','optimize','observe','evaluate']);a.add_argument('--output',required=True,type=Path)
    args=a.parse_args()
    try:globals()[args.phase](args.output)
    except Exception as e:
        if args.output.exists():write(args.output/f'failure_{args.phase}_{int(time.time())}.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
