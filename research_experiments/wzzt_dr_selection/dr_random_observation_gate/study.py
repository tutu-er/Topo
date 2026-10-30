"""Matched-history n3 mechanism gate. Simulation and evaluation stay outside fit.

Synthetic linear voltage-drop model, not AC or a field-calibrated EV study.
All past rows are preserved. No price or proposal uses true blocks/parameters.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pickle,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

REPO=Path(os.environ.get('WZZT_REPO_ROOT',r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo'))
sys.path[:0]=[str(REPO),str(REPO/'rnj_wzzt_core'),str(Path(__file__).resolve().parent)]

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def encode(value):return (json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(encode(value))
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value

def prepare(root):
    root=Path(root).resolve()
    if root.exists():raise FileExistsError('existing generation root retained')
    source={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
    source.update(source_hashes())
    bridge=REPO/'research_experiments/wzzt_dr_selection/dr_competition_bridge/physical_bridge.py'
    source[str(bridge)]=sha(bridge)
    cfg=dict(status='frozen_before_data_or_solver',created_utc=now(),repo_root=str(REPO),
        model='linear y=Rp+Xq, converted to positive root/terminal voltages; not AC',n=3,labels=[101,102,103],
        base_kw=200.,kappa=.35,L=2.6,b=.4,c_levels=[0.,.02,.6],true_pairs=[[0,1],[0,2],[1,2]],
        truth_rule='K=diag(L-3b-2c*1_C)+bJ+c*z_C*z_C.T; R=X=K/(1+kappa); K1=L1 for every truth',
        parameter_domain='all fitted r/x independently in[0,2], no positive lower bound or fixed R:X',
        historical_days_per_split=3,samples_per_day=96,future_event_roles=['training','training','validation','validation'],
        all_available_history_used=True,noise_regimes=[dict(id='exact',pq_relative=0.,voltage_relative=0.),
            dict(id='current_noise',pq_relative=.005,voltage_relative=.001)],
        main_policies=['zero_sham','pair_random'],diagnostic_policies=['single_random','rotation_random','pair_all_on_conditional'],
        diagnostic_scope='exact strong c=.6 only, all3 true pairs; not independent behavior trials',
        additional_stress='b=0,c=.6,exact,all3 true pairs with zero/pair; preserves zero common block',
        acceptance='price0/1/2 ->0/.5/.75, independent per node/event; synthetic, not empirical',
        control='acceptance first; independent whole-event random +/- orientation; 4 events, not32 randomizations',
        price_scope='fixed predeclared equal-budget policies; no new optimized price rule',
        fixed_pair=[0,2],fixed_single=1,device=dict(baseline_kw=.2,baseline_kvar=.07,power_upper_kw=.4,
            energy_kWh=.4,window_hours=[19,21],efficiency=1.,non_V2G=True),
        budget=1.6,peak_delta_kw=.2,peak_delta_kvar=.07,shift_energy_kWh=.2,
        discovery=dict(exact_y_error_bound=1e-12,pq_noise_default='unresolved',allow_unclosed_heuristic=False),
        fit_settings=dict(coefficient_bound=2.,max_n=3,root_label=1,search_seconds=25.,
            wall_seconds=900.,max_LP_calls=3000,include_current_rebuilt_rnj=True,
            calibration_snapshot='current_observed',n4_gate_passed=False),
        seeds=dict(natural=884001,measurement=884002,acceptance_and_independent_code=884003),
        expected_case_policy_evaluations=51,maximum_requested_unique_snapshots=32,
        no_old_observation_test_or_household_files_read=True,source_sha256=source,
        randomization_scope='one shared natural/measurement/acceptance/code stream across counterfactual truths and policies; no general success rate')
    root.mkdir(parents=True);write(root/'generation_protocol.json',cfg)
    write(root/'generation_protocol.sha256.json',dict(sha256=sha(root/'generation_protocol.json')))
    return cfg

def verify(root):
    root=Path(root);p=root/'generation_protocol.json'
    if sha(p)!=read(root/'generation_protocol.sha256.json')['sha256']:raise RuntimeError('generation protocol drift')
    cfg=read(p)
    for path,h in cfg['source_sha256'].items():
        if sha(path)!=h:raise RuntimeError('frozen generation source drift '+path)
    return cfg

def control(policy,cfg):
    from random_code import sample_controls,WAVE_KW,DEVICE
    price=np.zeros((2,3),int);pair=np.array([[0,2],[0,2]])
    if policy in ('pair_random','pair_all_on_conditional'):price[:,[0,2]]=1
    elif policy=='single_random':price[:,1]=2;pair=np.array([[1,0],[1,0]])
    elif policy=='rotation_random':price[0,0]=2;price[1,2]=2
    elif policy!='zero_sham':raise ValueError('unknown fixed policy')
    seed=cfg['seeds']['acceptance_and_independent_code']
    if policy!='pair_all_on_conditional':return sample_controls(price,pair,seed=seed,dependence='local')
    # A preregistered conditional state, not a favorable draw or a retry.
    z=price[np.arange(4)%2]>0;orientation=np.empty(4,int);delta=np.zeros((4,8,3))
    for e in range(4):
        rng=np.random.default_rng(np.random.SeedSequence([seed,76002,e]));orientation[e]=1 if rng.integers(0,2) else -1
        a,b=pair[e%2];delta[e,:,a]=orientation[e]*WAVE_KW;delta[e,:,b]=-orientation[e]*WAVE_KW
    full=price[np.arange(4)%2];payments=.2*np.sum(full*z,axis=1)
    return dict(prices=full,accepted=z,orientation=orientation,pairs=pair[np.arange(4)%2],
        delta_p_kw=delta,delta_q_kvar=.35*delta,committed_payment=payments,actual_payment=None,
        worst_payment=1.6,base_kw=200.,dependence='conditional_all_on',seed=seed,
        acceptance_before_code=True,independent_named_random_streams=True,code_status='issued; execution must be verified',
        whole_event_randomization_units=4,device_declaration=DEVICE,no_forced_participation=False,
        conditional_state_not_randomized_trial=True,pattern_probability_under_local_pair_price1=1/256,
        partial_response_cost_retained=True)

def natural_day(day,cfg):
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seeds']['natural'],day]))
    t=np.arange(96)/96
    scalar_kw=1.+.15*np.sin(2*np.pi*(t-.12))+.03*rng.uniform(-1,1,96)
    p=np.repeat((scalar_kw/cfg['base_kw'])[:,None],3,axis=1)
    return p,cfg['kappa']*p,1.+.003*np.sin(2*np.pi*(t+.07*day))

def observed_scene(p,q,R,X,day,regime,cfg,ideal_drop=None):
    from research_experiments.wzzt_dr_selection.dr_competition_bridge.physical_bridge import scene
    _,_,root=natural_day(day,cfg);drop=p@R.T+q@X.T
    if ideal_drop is not None:
        if np.max(abs(drop-ideal_drop))>1e-14:raise AssertionError('canonical algebra and physical linear model disagree')
        drop=np.asarray(ideal_drop)
    if (root[:,None]**2-drop<=0).any():raise ValueError('nonpositive modeled voltage')
    v=np.sqrt(root[:,None]**2-drop)
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seeds']['measurement'],day]))
    ep,eq,ev,er=(rng.standard_normal(shape) for shape in (p.shape,q.shape,v.shape,root.shape))
    pm=p*(1+regime['pq_relative']*ep);qm=q*(1+regime['pq_relative']*eq)
    vm=v*(1+regime['voltage_relative']*ev);rm=root*(1+regime['voltage_relative']*er)
    return scene(pm,qm,vm,rm,cfg['labels'],'observed_day_'+str(day),day)

def truth(b,c,pair,cfg):
    n=3;z=np.zeros(n);z[pair]=1.
    d=cfg['L']-3*b-2*c*z
    if (d<0).any():raise ValueError('unphysical negative leaf weight')
    K=np.diag(d)+b*np.ones((n,n))+c*np.outer(z,z);R=X=K/(1+cfg['kappa'])
    coeff=list(d)+[b,c]
    if any(w/(1+cfg['kappa'])>2+1e-12 for w in coeff):raise ValueError('true coefficients exceed unchanged fit box')
    family=[tuple([i]) for i,v in enumerate(d) if v>0]
    if b>0:family.append((0,1,2))
    if c>0:family.append(tuple(pair))
    if not np.allclose(K@np.ones(n),cfg['L']*np.ones(n),atol=1e-14):raise AssertionError('matched-history identity failed')
    return R,X,dict(b=b,c=c,pair=pair,d=d.tolist(),R=R.tolist(),X=X.tolist(),K=K.tolist(),
                    effective_supports=[list(s) for s in sorted(family,key=lambda a:(len(a),a))])

def save_packet(root,packet):
    data=pickle.dumps(packet,protocol=4);h=hashlib.sha256(data).hexdigest();path=root/'packets'/(h+'.pkl')
    if not path.exists():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    return dict(path=str(path),sha256=h)

def execute_future(R,X,draw,regime,cfg):
    scenes=[];records=[]
    for e in range(4):
        p,q,_=natural_day(6+e,cfg);dp=np.zeros_like(p);dq=np.zeros_like(q)
        dp[76:84]=draw['delta_p_kw'][e]/cfg['base_kw'];dq[76:84]=draw['delta_q_kvar'][e]/cfg['base_kw']
        z=draw['accepted'][e];device=.2+draw['delta_p_kw'][e];energy=np.cumsum(device,axis=0)*.25
        checks=dict(task_power=bool(((device>=-1e-12)&(device<=.4+1e-12)).all()),
                    task_energy=bool(np.allclose(energy[-1],.4,atol=1e-12)),
                    nonnegative_residual=bool((p[76:84,z]*200>=.2).all() and(q[76:84,z]*200>=.07).all()),
                    net_energy=bool(np.max(abs(dp.sum(axis=0)*200*.25))<1e-12),
                    peak_increment=bool(np.max(abs(dp)*200)<=.2+1e-12))
        if not all(checks.values()):raise ValueError('declared task could not execute; no clipping or redraw')
        canonical=cfg['L']*p[:,[0]]*np.ones((1,3))+dp@R.T+dq@X.T
        scenes.append(observed_scene(p+dp,q+dq,R,X,6+e,regime,cfg,canonical))
        records.append(dict(event=e,checks=checks,payment=.2*float(draw['prices'][e]@z),accepted=z.astype(int).tolist(),
            orientation=int(draw['orientation'][e]),delivered_reduction_kWh=.2*int(z.sum())))
    packet=dict(training=scenes[:2],validation=scenes[2:],metadata=dict(observed_only=True,contains_history=False,model='linear_drop'))
    return packet,records

def generate(root):
    from gate_driver import observed_training_for_discovery
    from random_code import controls_as_samples,discover_candidates,observation_moments
    root=Path(root).resolve();cfg=verify(root)
    if (root/'config.json').exists() or (root/'generation_started.json').exists():raise FileExistsError('generated state retained; no implicit rerun')
    write(root/'generation_started.json',dict(started_utc=now(),protocol_sha256=sha(root/'generation_protocol.json')))
    histories=[];history_packets={};snapshots={};evaluations=[];case_index=[];control_receipts={}
    R0,X0,_=truth(cfg['b'],0.,[0,1],cfg)
    for regime in cfg['noise_regimes']:
        scenes=[]
        for day in range(6):
            pp,qq,_=natural_day(day,cfg)
            scenes.append(observed_scene(pp,qq,R0,X0,day,regime,cfg,cfg['L']*pp[:,[0]]*np.ones((1,3))))
        packet=dict(training=scenes[:3],validation=scenes[3:],metadata=dict(observed_only=True,contains_history=True,model='linear_drop'))
        ref=save_packet(root,packet);hid=regime['id'];histories.append(dict(id=hid,**ref));history_packets[hid]=packet
    units=[]
    for regime in cfg['noise_regimes']:
        for c in cfg['c_levels']:
            for pair in cfg['true_pairs']:units.append((regime,cfg['b'],c,pair,False))
    exact=cfg['noise_regimes'][0]
    for pair in cfg['true_pairs']:units.append((exact,0.,.6,pair,True))
    for index,(regime,b,c,pair,stress) in enumerate(units):
        R,X,target=truth(b,c,pair,cfg);case_id=f'case_{index:03d}';target['case_id']=case_id
        case_index.append(dict(id=case_id,noise_id=regime['id'],truth=target,zero_root_stress=stress))
        # Prove history equality for each actual truth; generator facts never
        # enter the observed metadata or candidate discovery function.
        same_history=[]
        for day in range(6):
            pp,qq,_=natural_day(day,cfg)
            same_history.append(observed_scene(pp,qq,R,X,day,regime,cfg,cfg['L']*pp[:,[0]]*np.ones((1,3))))
        candidate_h=dict(training=same_history[:3],validation=same_history[3:],metadata=dict(observed_only=True,contains_history=True,model='linear_drop'))
        if save_packet(root,candidate_h)['sha256']!=next(h['sha256'] for h in histories if h['id']==regime['id']):
            raise AssertionError('true model leaked into matched historical bytes')
        policies=['zero_sham','pair_random']
        if not stress and regime['id']=='exact' and c==.6:policies+=cfg['diagnostic_policies']
        for policy in policies:
            draw=control(policy,cfg);future,event_records=execute_future(R,X,draw,regime,cfg)
            history=history_packets[regime['id']]
            observed=observed_training_for_discovery(history,future)
            logged=controls_as_samples(draw,(0,1));logged['control_status']='verified_exact_in_declared_task_simulation'
            controls=dict(future_train=logged)
            settings=dict(pq_observation_mode='trusted_exact' if regime['pq_relative']==0 else 'unknown_measurement_error',
                executed_control_mode='verified_exact',power_unit='pu',base_kw=200.,
                y_error_bound=1e-12 if regime['voltage_relative']==0 else None,
                historical_y_error_bound=1e-12 if regime['voltage_relative']==0 else None,
                band_fixed_before_codes=True,allow_unclosed_heuristic=False,max_candidates=64)
            discovery=discover_candidates(observed,controls,settings)
            discovery['true_structure_used']=False
            moments=observation_moments(observed,controls)
            ref=save_packet(root,future)
            identity=hashlib.sha256(encode([regime['id'],ref['sha256'],discovery])).hexdigest()
            if identity not in snapshots:
                sid='s_'+identity[:20];discovery_path=root/'discovery'/(identity+'.json');write(discovery_path,plain(discovery))
                snapshots[identity]=dict(id=sid,history_id=regime['id'],**ref,
                    discovery=dict(path=str(discovery_path),sha256=sha(discovery_path)))
                write(root/'control_records'/(identity+'.json'),plain(dict(policy=policy,
                    control_draw=draw,execution=event_records,discovery_config=settings,observation_moments=moments,
                    actual_payment=sum(r['payment'] for r in event_records),worst_payment=draw['worst_payment'],
                    field_calibration=False,observed_counterfactual_baseline_used=False)))
            evaluations.append(dict(case_id=case_id,snapshot_id=snapshots[identity]['id'],policy=policy,
                conditional_state=policy=='pair_all_on_conditional',actual_payment=sum(r['payment'] for r in event_records),
                history_id=regime['id'],target_family=target['effective_supports'],
                truth_used_only_after_identifier=True))
    if len(evaluations)!=cfg['expected_case_policy_evaluations']:raise AssertionError('evaluation denominator mismatch')
    if len(snapshots)>cfg['maximum_requested_unique_snapshots']:raise AssertionError('snapshot cap exceeded')
    write(root/'evaluator_truth.json',dict(cases=case_index,evaluations=evaluations,identifier_input=False))
    config=dict(status='root_frozen_before_solver',settings=cfg['fit_settings'],histories=histories,
        snapshots=list(snapshots.values()),generation_protocol=dict(path=str(root/'generation_protocol.json'),sha256=sha(root/'generation_protocol.json')),
        truth_input_allowed=False,price_optimized=False)
    write(root/'config.json',config)
    summary=dict(status='generated',case_units=len(units),evaluations=len(evaluations),unique_history_packets=len(histories),
        unique_future_snapshots=len(snapshots),maximum_requested_unique_snapshots=32,
        matched_history_hashes={h['id']:h['sha256'] for h in histories},
        random_trials='one shared response/code stream; conditional all-on separate, no general recovery frequency',
        LP_calls=0,denseQP_calls=0,AC_calls=0,price_budget=1.6)
    write(root/'generation_summary.json',summary);return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','generate'));p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();result=prepare(a.root) if a.mode=='prepare' else generate(a.root)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','case_units','evaluations','unique_history_packets','unique_future_snapshots')}))

if __name__=='__main__':main()
