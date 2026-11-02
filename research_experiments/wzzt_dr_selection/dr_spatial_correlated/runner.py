"""Bounded spatial/correlated DR planning and original AC evaluation.

Full historical models stay frozen. True network and new physical backgrounds
are accessed only after all prices, capacities, and response seeds are locked.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,pickle,hashlib,platform,sys,time,zipfile,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import scipy

REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
PRIVATE=Path(r'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
sys.path.insert(0,str(REPO))
from spatial_optimizer import SpatialProbability,cost,feasible,solve_local
from response_law import MAIN_LAW,MISSPEC_LAW,draw_base_normals,standardized_from_base
from research_experiments.wzzt_dr_selection.dr_newday_weighted.weighted_selection import score

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(r):
    p=Path(r['path'])
    if sha(p)!=r['sha256']:raise ValueError('frozen input drift: '+str(p))
    return p
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def loadp(p):
    with Path(p).open('rb') as f:return pickle.load(f)
def dumpp(p,v):
    with Path(p).open('wb') as f:pickle.dump(v,f,protocol=4)

def source_files(source_root):
    s=list(Path(source_root).rglob('*.py'))+list(Path(__file__).parent.glob('*.py'))
    task=REPO/'research_experiments/wzzt_dr_selection'
    for f in ['dr_conditional_gaussian/problem.py','dr_conditional_gaussian/conditional_integral.py',
              'dr_competition_bridge/physical_bridge.py','end_to_end/physical_case.py',
              'dr_frozen_distance/frozen_pool.py','dr_newday_weighted/weighted_selection.py',
              'dr_pool_v2/selection.py','dr_pool_v2/observations.py']:s.append(task/f)
    return sorted(set(x.resolve() for x in s))

def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    parent=PRIVATE/'wzzt_conditional_dr_stage/run_v1'
    pp=read(parent/'protocol.json');oldref=read(parent/'problem_frozen.json')['input']
    physical=read(PRIVATE/'wzzt_conditional_ac_stage/run_v1/protocol.json')
    sources=source_files(physical['source_root'])
    protocol=dict(id='spatial_correlated_DR_v1',status='frozen',created_utc=now(),case='paper15',
        condition='paper15_b2_v0005',inputs=pp['inputs'],parent_problem=oldref,
        legacy_price=ref(parent/'price_freeze.json'),source_root=physical['source_root'],
        source_metadata=physical['source_metadata'],source_sha256={str(f):sha(f) for f in sources},
        history_rows=[288,288],T=96,dt=.25,base_kw=200.,resource_total_kw=2.,beta_per_capacity=.5,
        capacity_rule='max(0,min(1,min historical P kW,min historical Q kvar/.35)), all six complete historical days',
        allocation='continuous, time-invariant daily enrolled capacity a_i; sum a<=2; abs u_it<=a_i; no positive lower a',
        price='abs u/(.5a), zero a requires zero u; same maximum price2; prepaid .25 sum u^2/(.5a)',
        energy_rule='no net-energy equality or ramp rule, unchanged from prior Gaussian model',
        response_main=MAIN_LAW.to_dict(),response_misspec=MISSPEC_LAW.to_dict(),
        correlation_calibration='prespecified synthetic dependence/stress assumptions; not empirically calibrated',
        rule=dict(history_gate=True,new_weight=2,new_fraction=.03,history_fraction=.03),
        optimizer=dict(target=.95,penalty=1000.,initial_radius=.3,max_radius=.8,min_radius=1e-5,maxiter=45,seconds=180.),
        planning_samples=128,base_seed=2026110301,response_seed=2026110302,
        diagnostic_samples=512,diagnostic_base_seed=2026110303,diagnostic_response_seed=2026110304,
        starts=['legacy_fixed_plan','zero_or_uniform_gradient'],selection='cheapest planning-feasible returned candidate; otherwise highest probability diagnostic, no actual observations',
        comparisons=['zero','legacy_iid_pair','fixed_correlated','open_correlated','rotation_uniform','rotation_selected_capacity'],
        main_baseline='rotation_uniform has full eligible node access, fixed equal allocation, prelocked 6h phases, cost matched to open strategy',
        phase_seed=2026110305,voltage_sigma=.0005,pq_sigma=.005,
        days=[dict(day=i,profile_index=i%3,replicate=950000+97*i,response_seed=2026110310+i) for i in range(12)],
        observed_laws=['main','misspec'],main_observation='both',
        actual_scope='12 shared new background replicates x two paired response laws; these are not24 independent background days',
        execution='original AC and meter functions; no clipping, redraw, or excursion-based omission',
        truth_gate='network only after price freeze, truth supports only after predictions frozen',
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()))
    write(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for i,f in enumerate(sources):z.write(f,f'{i:03d}_{f.name}')
    old=loadp(checked(oldref));pool=read(checked(pp['inputs']['pool']));labels=pool['labels']
    packet=loadp(checked(pp['inputs']['history']));history=packet['history'];del packet
    p=np.concatenate([s['P_terminal'].to_numpy() for s in history['training']+history['validation']])*200
    q=np.concatenate([s['Q_terminal'].to_numpy() for s in history['training']+history['validation']])*200
    if p.shape!=(576,15):raise ValueError('full original history required')
    caps=np.maximum(0,np.minimum.reduce([np.ones(15),p.min(0),q.min(0)/.35]));active=np.flatnonzero(caps>0)
    models={k['candidate_id']:k for k in pool['models']};new=dict(old)
    new.update(shape=(len(active),96),active_positions=active.tolist(),all_node_count=15,
        control_labels=[labels[i] for i in active],capacity_caps=caps[active],pool=pool,
        cv=np.full((len(active),96),np.sqrt(.025/3)),elasticities=np.full((len(active),96),.5),
        lower=-np.broadcast_to(caps[active,None],(len(active),96)).copy(),
        upper=np.broadcast_to(caps[active,None],(len(active),96)).copy(),response_factor=None)
    for key,k in [('C','coarse'),('F','refined')]:
        r=np.asarray(models[k]['R']);x=np.asarray(models[k]['X'])
        new['effect_'+key]=((old['r0']-r)+.35*(old['x0']-x))[:,active]*1e6/200
    legacy=read(checked(protocol['legacy_price']))['policies']['continuous']
    u=np.zeros(new['shape']);a=np.zeros(len(active))
    for j,resource in enumerate(old['resources']):
        i=new['control_labels'].index(resource['node_label']);u[i]=legacy['mean_u'][j];a[i]=1.
    if not feasible(u,a,new['capacity_caps'],tol=1e-10):raise ValueError('old policy not embedded in new domain')
    new.update(legacy_u=u,legacy_a=a)
    phases=np.random.default_rng(protocol['phase_seed']).permutation(len(active))%2
    new['rotation_sign']=np.where((np.arange(96)[None,:]//24+phases[:,None])%2==0,1.,-1.)
    dumpp(root/'problem.pkl',new)
    write(root/'problem_frozen.json',dict(input=ref(root/'problem.pkl'),controls=int(np.prod(new['shape'])),
        capacity_variables=len(active),labels=new['control_labels'],caps=new['capacity_caps'].tolist(),
        excluded_labels=[labels[i] for i in range(15) if caps[i]==0],legacy_embedded=True,
        full_historical_rows=576,threshold_micro=new['threshold'],new_parameter_fits=0,
        capacity_rows=[dict(node_label=labels[i],pmin_kw=float(p.min(0)[i]),qmin_kvar=float(q.min(0)[i]),cap_kw=float(caps[i])) for i in range(15)]))
    write(root/'run_lock.json',dict(status='prepared',pid=None))
    print('prepared',len(active),'nodes',np.prod(new['shape']),'controls',flush=True)

def config(root):
    root=Path(root);p=read(root/'protocol.json')
    for f,h in p['source_sha256'].items():
        if sha(f)!=h:raise ValueError('source drift '+f)
    for r in p['inputs'].values():checked(r)
    packet=loadp(checked(read(root/'problem_frozen.json')['input']))
    return root,p,packet

def gradient_audit(problem,packet):
    rng=np.random.default_rng(2026110306);a=np.full(packet['shape'][0],2/packet['shape'][0]);u=rng.uniform(-.5,.5,packet['shape'])*a[:,None]
    r=problem.evaluate(u);g=r['gradient']
    if g is None:return dict(passed=False,reason='gradient invalid')
    rows=[]
    for j in range(3):
        direction=rng.normal(size=u.shape);direction/=np.linalg.norm(direction);eps=2e-5
        fd=(problem.evaluate(u+eps*direction)['probability']-problem.evaluate(u-eps*direction)['probability'])/(2*eps)
        analytic=float(np.sum(g.reshape(u.shape)*direction))
        rows.append(dict(fd=float(fd),analytic=analytic,error=abs(fd-analytic)))
    return dict(passed=bool(max(r['error'] for r in rows)<1e-6),rows=rows)

def optimize(root):
    root,p,k=config(root)
    if (root/'price_freeze.json').exists():raise RuntimeError('already frozen; no duplicate solves')
    problem=SpatialProbability(k,p['planning_samples'],p['base_seed'],p['response_seed'])
    ga=gradient_audit(problem,k);write(root/'gradient_audit.json',ga)
    if not ga['passed']:raise RuntimeError('new correlation gradient failed')
    zero=np.zeros(k['shape']);a_uniform=np.full(k['shape'][0],2/k['shape'][0]);zero_r=problem.evaluate(zero)
    gradient_start=.25*a_uniform[:,None]*np.sign(zero_r['gradient'].reshape(k['shape']))
    records=[]
    write(root/'run_lock.json',dict(status='optimizing',pid=os.getpid()))
    for domain in ['fixed','open']:
        starts=[('legacy',k['legacy_u'],k['legacy_a'])]
        starts.append(('zero' if domain=='fixed' else 'uniform_gradient',zero if domain=='fixed' else gradient_start,
                       k['legacy_a'] if domain=='fixed' else a_uniform))
        caps=k['legacy_a'] if domain=='fixed' else k['capacity_caps']
        for name,u,a in starts:
            path=root/f'solve_{domain}_{name}.json'
            if path.exists():records.append(read(path));continue
            def checkpoint(trace,candidates):write(root/f'checkpoint_{domain}_{name}.json',dict(trace=trace,last=candidates[-1]))
            try:
                result=solve_local(problem,caps,u,a,p['optimizer'],fixed_capacity=k['legacy_a'] if domain=='fixed' else None,callback=checkpoint)
                result.update(domain=domain,start=name)
            except Exception as e:result=dict(domain=domain,start=name,status='failed',error=repr(e),traceback=traceback.format_exc(),best=None)
            write(path,result);records.append(result)
            print('solve',domain,name,result['status'],None if result['best'] is None else (result['best']['planning_probability'],result['best']['cost']),flush=True)
    def choose(domain):
        candidates=[r['best'] for r in records if r['best'] is not None and (r['domain']==domain or domain=='open')]
        # Fixed-domain candidates belong to the open domain, so retain them exactly.
        legacy_r=problem.evaluate(k['legacy_u']);candidates.append(dict(mean_u=k['legacy_u'].tolist(),capacity_kw=k['legacy_a'].tolist(),
            planning_probability=legacy_r['probability'],cost=cost(k['legacy_u'],k['legacy_a'])))
        good=[c for c in candidates if c['planning_probability']>=.95-1e-7]
        return min(good,key=lambda c:c['cost']) if good else max(candidates,key=lambda c:(c['planning_probability'],-c['cost']))
    selected={key:choose(key) for key in ['fixed','open']}
    pol={}
    def add(name,u,a,role):
        u,a=np.asarray(u),np.asarray(a)
        if not feasible(u,a,k['capacity_caps'],tol=1e-7):raise AssertionError('frozen policy resource violation')
        pol[name]=dict(mean_u=u.tolist(),capacity_kw=a.tolist(),cost=cost(u,a),role=role)
    add('zero',zero,np.zeros_like(k['legacy_a']),'no reward natural new day')
    add('legacy_iid_pair',k['legacy_u'],k['legacy_a'],'previous independent-error design, no reoptimization')
    for key in ['fixed','open']:
        s=selected[key];add(key+'_correlated',s['mean_u'],s['capacity_kw'],'original conditional probability reoptimized under correlated response; '+key+' domain')
    open_cost=pol['open_correlated']['cost']
    for name,a in [('rotation_uniform',a_uniform),('rotation_selected_capacity',np.asarray(selected['open']['capacity_kw']))]:
        scale=np.sqrt(open_cost/(48*a.sum())) if open_cost else 0.
        if scale>1+1e-8:raise ValueError('same-cost rotation infeasible; do not enlarge resources')
        add(name,scale*a[:,None]*k['rotation_sign'],a,
            'prelocked all-eligible-node equal capacity simple exploration' if name=='rotation_uniform' else 'optimized allocation reused, time/sign ablation only')
    qualification=SpatialProbability(k,p['diagnostic_samples'],p['diagnostic_base_seed'],p['diagnostic_response_seed'])
    for name,v in pol.items():
        u=np.asarray(v['mean_u']);pr=problem.evaluate(u);qr=qualification.evaluate(u)
        v.update(planning_probability=pr['probability'],diagnostic_probability=qr['probability'],diagnostic_se=qr['conditional_mc_se'],
            planning_target_met=bool(pr['probability']>=.95-1e-7))
        print('freeze',name,v['cost'],pr['probability'],qr['probability'],flush=True)
    write(root/'planning_summary.json',dict(results=[{key:v for key,v in r.items() if key not in ['trace','best']} for r in records],
        policies={name:{key:v for key,v in r.items() if key!='mean_u'} for name,r in pol.items()},global_optimality_proven=False))
    write(root/'price_freeze.json',dict(status='frozen',created_utc=now(),policies=pol,problem=ref(root/'problem.pkl'),protocol=ref(root/'protocol.json'),future_read=False))
    write(root/'run_lock.json',dict(status='prices_frozen',pid=None))

def observe(root):
    from ac_bridge import open_runtime,build_background,execute_policy
    root,p,k=config(root);f=read(root/'price_freeze.json');checked(f['protocol']);checked(f['problem'])
    if (root/'observations_frozen.json').exists():raise RuntimeError('observations already complete')
    rt=open_runtime(p['source_root'],p['source_metadata'],repo_root=REPO,protocol_ref=ref(root/'protocol.json'),
        labels=k['pool']['labels'],case='paper15',v_sigma=.0005,pq_sigma=.005)
    rows=[];write(root/'run_lock.json',dict(status='observing',pid=os.getpid()))
    for d in p['days']:
        bg=build_background(rt,d['profile_index'],d['replicate'])
        innovations=draw_base_normals(d['response_seed'],1,15,96)
        for lname,law in [('main',MAIN_LAW),('misspec',MISSPEC_LAW)]:
            z=standardized_from_base(innovations,15,96,law)[0,k['active_positions'],:]
            for name,v in f['policies'].items():
                identity=f"day{d['day']:02d}_{lname}_{name}";folder=root/'actual'/identity;folder.mkdir(parents=True,exist_ok=True)
                receipt=folder/'receipt.json';out=folder/'result.pkl'
                if receipt.exists():
                    r=read(receipt)
                    if 'packet' in r:checked(r['packet'])
                    rows.append(r);continue
                try:
                    result=execute_policy(rt,bg,np.asarray(v['mean_u']),z,resource_labels=k['control_labels'],
                        capacity_kw=np.asarray(v['capacity_kw']),cv=law.relative_sd,multiplier_mean=law.mean,
                        day_offset=6+d['day'],name=identity)
                    dumpp(out,result)
                    r=dict(record_id=identity,**d,law=lname,policy=name,status='observed',packet=ref(out),physical=result['summary'],fee=v['cost'])
                except Exception as e:r=dict(record_id=identity,**d,law=lname,policy=name,status='failed',fee=v['cost'],error=repr(e),traceback=traceback.format_exc())
                write(receipt,r);rows.append(r)
                write(root/'checkpoint_actual.json',dict(completed=len(rows),planned=144,last=identity))
        print('AC day',d['day'],len(rows),'/144',flush=True)
    write(root/'observations_frozen.json',dict(status='frozen',rows=rows,price_freeze=ref(root/'price_freeze.json')))
    write(root/'run_lock.json',dict(status='observations_frozen',pid=None))

def evaluate(root):
    from ac_bridge import open_runtime
    root,p,k=config(root);f=read(root/'price_freeze.json');ledger=read(root/'observations_frozen.json');checked(ledger['price_freeze'])
    rows=[]
    for r0 in ledger['rows']:
        row={key:v for key,v in r0.items() if key not in ['packet','traceback']}
        try:
            if r0['status']!='observed':raise ValueError('physical observation failed')
            result=loadp(checked(r0['packet']));scene=result['scenes']['both']
            if scene is None:raise ValueError('physical unresolved')
            s=score(k['pool'],scene,history_gate=True)
            q=next(q for q in s['rules'] if q['new_day_weight']==2 and q['new_day_fraction']==.03)
            losses={r['candidate_id']:r['new_validation_mae'] for r in s['loss_rows']}
            out=root/'scores'/(r0['record_id']+'.json');write(out,s)
            row.update(status='predicted',selected=q['selected_candidate_id'],coarse_excluded='coarse' not in q['eligible_candidate_ids'],
                refined_excluded='refined' not in q['eligible_candidate_ids'],D_micro=1e6*(losses['coarse']-1.03*losses['refined']),
                margin_micro=1e6*q['exit_margins']['coarse'],score=ref(out))
        except Exception as e:row.update(status='unresolved',error=repr(e))
        rows.append(row)
    write(root/'predictions_frozen.json',dict(status='frozen',rows=rows,truth_read=False))
    rt=open_runtime(p['source_root'],p['source_metadata'],repo_root=REPO,protocol_ref=ref(root/'protocol.json'),labels=k['pool']['labels'],case='paper15',v_sigma=.0005,pq_sigma=.005)
    truth=rt['legacy'].evaluate_supports([],rt['context'].net);write(root/'truth_private.json',truth)
    ts={tuple(x) for x in truth['missed_blocks']}
    def errors(s):
        ss={tuple(sorted(x)) for x in s};return dict(missing=len(ts-ss),extra=len(ss-ts),errors=len(ts^ss))
    models={}
    for m in k['pool']['models']:
        models[m['candidate_id']]=dict(raw=errors([[k['pool']['labels'][i] for i in b] for b in m['point']['supports']]),final=errors(m['frozen_output_support_labels']))
    for row in rows:
        if row['status']=='predicted':row.update(status='complete',**models[row['selected']]['final'],raw_errors=models[row['selected']]['raw']['errors'])
    tables=[];pairs=[]
    for law in ['main','misspec']:
        for name,v in f['policies'].items():
            group=[r for r in rows if r['law']==law and r['policy']==name];valid=[r for r in group if r['status']=='complete']
            comparisons=[]
            for r in valid:
                zero=next(z for z in rows if z['law']==law and z['policy']=='zero' and z['day']==r['day'])
                if zero['status']=='complete':comparisons.append(r['errors']-zero['errors'])
            tables.append(dict(law=law,policy=name,planned=12,complete=len(valid),unresolved=12-len(valid),recoveries=sum(r['errors']==0 for r in valid),
                total_errors=sum(r['errors'] for r in valid),raw_errors=sum(r['raw_errors'] for r in valid),
                mean_D_micro=float(np.mean([r['D_micro'] for r in valid])) if valid else None,
                coarse_excluded=sum(r['coarse_excluded'] for r in valid),refined_excluded=sum(r['refined_excluded'] for r in valid),
                fee_per_day=v['cost'],capacity_total=sum(v['capacity_kw']),better_vs_zero=sum(x<0 for x in comparisons),worse_vs_zero=sum(x>0 for x in comparisons)))
        for other in ['fixed_correlated','legacy_iid_pair','rotation_uniform','rotation_selected_capacity']:
            dd=[]
            for d in p['days']:
                a=next(r for r in rows if r['day']==d['day'] and r['law']==law and r['policy']=='open_correlated')
                b=next(r for r in rows if r['day']==d['day'] and r['law']==law and r['policy']==other)
                if a['status']==b['status']=='complete':dd.append(a['errors']-b['errors'])
            pairs.append(dict(law=law,comparison='open_vs_'+other,better=sum(x<0 for x in dd),worse=sum(x>0 for x in dd),same=sum(x==0 for x in dd),unresolved=12-len(dd)))
    write(root/'scored_private.json',dict(rows=rows))
    write(root/'summary.json',dict(status='complete',tables=tables,pairs=pairs,models=models,planned_AC_days=144,
        observations=sum(r['status']=='observed' for r in ledger['rows']),complete_selections=sum(r['status']=='complete' for r in rows),
        independent_backgrounds=12,paired_laws=2,profile_strata=3,new_fits=0,new_W_search=0,
        scope='same-case frozen historical pair; new correlated/stress response and original AC+meter evaluation'))
    write(root/'ended.json',dict(status='complete',summary=ref(root/'summary.json')))
    write(root/'run_lock.json',dict(status='complete',pid=None))
    print(json.dumps(read(root/'summary.json'),ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','optimize','observe','evaluate']);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();globals()[args.command](args.output)
