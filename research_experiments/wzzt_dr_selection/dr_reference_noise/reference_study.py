"""Frozen reference-noise ablation; legacy loads, controls and fitting stay fixed."""
from __future__ import annotations
import argparse,hashlib,json,os,pickle,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

REPO=Path(os.environ.get('WZZT_REPO_ROOT',r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo'))
LEGACY=REPO/'research_experiments/wzzt_dr_selection/dr_random_observation_gate'
sys.path[:0]=[str(LEGACY),str(REPO),str(REPO/'rnj_wzzt_core')]
import study as original
from gate_driver import observed_training_for_discovery
from random_code import controls_as_samples,discover_candidates,observation_moments

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def encode(v):return (json.dumps(original.plain(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(encode(v))

def configuration():
    return dict(status='frozen_before_generation_or_solver',created_utc=now(),repo_root=str(REPO),
        model='same linear y=Rp+Xq as random_observation_gate; no AC or field calibration',
        labels=[101,102,103],n=3,base_kw=200.,kappa=.35,L=2.6,b=.4,c_levels=[0.,.02,.6],
        true_pairs=[[0,1],[0,2],[1,2]],
        noise_regimes=[dict(id='voltage_only',pq_relative=0.,voltage_relative=.0002),
                       dict(id='power_only',pq_relative=.005,voltage_relative=0.),
                       dict(id='combined_reference',pq_relative=.005,voltage_relative=.0002)],
        noise_interpretation='relative Gaussian standard deviation per 15min-grid reading; root/terminal equal V rates; no instrument calibration',
        no_averaging_filter_or_fixed_gain_model=True,
        historical_days_per_split=3,samples_per_day=96,
        future_event_roles=['training','training','validation','validation'],all_available_history_used=True,
        main_policies=['zero_sham','pair_random'],
        stress='b=0,c=.6,all3 pairs in each of the three noise regimes; no removal of weak/zero weights',
        budget=1.6,peak_delta_kw=.2,peak_delta_kvar=.07,shift_energy_kWh=.2,
        acceptance='unchanged price0/1/2 -> probability0/.5/.75; acceptance before independent whole-event +/- code',
        prices_or_response_optimized=False,fixed_pair=[0,2],
        seeds=dict(natural=884001,measurement=884002,acceptance_and_independent_code=884003),
        randomization_scope='same streams as previous experiment across all noises/truths/policies; not independent trials',
        device=dict(baseline_kw=.2,baseline_kvar=.07,power_upper_kw=.4,energy_kWh=.4,
                    window_hours=[19,21],efficiency=1.,non_V2G=True),
        fit_settings=dict(coefficient_bound=2.,max_n=3,root_label=1,search_seconds=25.,wall_seconds=900.,
            max_LP_calls=3000,include_current_rebuilt_rnj=True,calibration_snapshot='current_observed',n4_gate_passed=False),
        new_baseline='current_root_augmented_rnj: each native current RNJ family plus optional fullroot, independent LP, original validation/contract',
        expected_case_units=36,expected_case_policy_evaluations=72,maximum_unique_snapshots=33,
        impedance_scale=1.,old_exact_high_results_not_rerun=True,
        simulation_truth_visible_only_to_generator_and_postselection_evaluator=True)

def prepare(root):
    root=Path(root).resolve()
    if root.exists():raise FileExistsError('existing generation root retained')
    cfg=configuration()
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
    sources=source_hashes()
    for folder in (LEGACY,Path(__file__).resolve().parent):
        for p in folder.glob('*.py'):sources[str(p)]=sha(p)
    bridge=REPO/'research_experiments/wzzt_dr_selection/dr_competition_bridge/physical_bridge.py'
    sources[str(bridge)]=sha(bridge)
    cfg['source_sha256']=sources;root.mkdir(parents=True)
    write(root/'generation_protocol.json',cfg);write(root/'generation_protocol.sha256.json',dict(sha256=sha(root/'generation_protocol.json')))
    return dict(status=cfg['status'],expected_case_units=36,expected_evaluations=72)

def verify(root):
    p=Path(root)/'generation_protocol.json';cfg=read(p)
    if sha(p)!=read(Path(root)/'generation_protocol.sha256.json')['sha256']:raise RuntimeError('protocol drift')
    for p,h in cfg['source_sha256'].items():
        if sha(p)!=h:raise RuntimeError('source drift '+p)
    return cfg

def history_packet(R,X,regime,cfg):
    scenes=[]
    for day in range(6):
        pp,qq,_=original.natural_day(day,cfg)
        scenes.append(original.observed_scene(pp,qq,R,X,day,regime,cfg,cfg['L']*pp[:,[0]]*np.ones((1,3))))
    return dict(training=scenes[:3],validation=scenes[3:],metadata=dict(observed_only=True,contains_history=True,model='linear_drop'))

def generate(root):
    root=Path(root).resolve();cfg=verify(root)
    if (root/'generation_started.json').exists():raise FileExistsError('prior generation retained, no rerun')
    write(root/'generation_started.json',dict(started_utc=now(),protocol_sha256=sha(root/'generation_protocol.json')))
    histories=[];history={};snaps={};cases=[];evaluations=[]
    R0,X0,_=original.truth(cfg['b'],0.,[0,1],cfg)
    for regime in cfg['noise_regimes']:
        packet=history_packet(R0,X0,regime,cfg);ref=original.save_packet(root,packet)
        history[regime['id']]=packet;histories.append(dict(id=regime['id'],**ref))
    units=[]
    for regime in cfg['noise_regimes']:
        for c in cfg['c_levels']:
            for pair in cfg['true_pairs']:units.append((regime,cfg['b'],c,pair,False))
        for pair in cfg['true_pairs']:units.append((regime,0.,.6,pair,True))
    for index,(regime,b,c,pair,stress) in enumerate(units):
        R,X,target=original.truth(b,c,pair,cfg);cid=f'case_{index:03d}';target['case_id']=cid
        cases.append(dict(id=cid,noise_id=regime['id'],truth=target,zero_root_stress=stress))
        regenerated=original.save_packet(root,history_packet(R,X,regime,cfg))
        if regenerated['sha256']!=next(h['sha256'] for h in histories if h['id']==regime['id']):
            raise AssertionError('matched-history bytes differ across true structures')
        for policy in cfg['main_policies']:
            draw=original.control(policy,cfg);future,receipts=original.execute_future(R,X,draw,regime,cfg)
            observed=observed_training_for_discovery(history[regime['id']],future)
            logged=controls_as_samples(draw,(0,1));logged['control_status']='verified_exact_in_declared_task_simulation'
            controls=dict(future_train=logged)
            settings=dict(pq_observation_mode='trusted_exact' if regime['pq_relative']==0 else 'unknown_measurement_error',
                executed_control_mode='verified_exact',power_unit='pu',base_kw=200.,
                y_error_bound=None,historical_y_error_bound=None,band_fixed_before_codes=True,
                allow_unclosed_heuristic=False,max_candidates=64)
            discovery=discover_candidates(observed,controls,settings);discovery['true_structure_used']=False
            moments=observation_moments(observed,controls);ref=original.save_packet(root,future)
            ident=hashlib.sha256(encode([regime['id'],ref['sha256'],discovery])).hexdigest()
            if ident not in snaps:
                sid='s_'+ident[:20];dp=root/'discovery'/(ident+'.json');write(dp,discovery)
                snaps[ident]=dict(id=sid,history_id=regime['id'],**ref,discovery=dict(path=str(dp),sha256=sha(dp)))
                write(root/'control_records'/(ident+'.json'),dict(policy=policy,control_draw=draw,
                    execution=receipts,discovery_config=settings,observation_moments=moments,
                    actual_payment=sum(x['payment'] for x in receipts),worst_payment=draw['worst_payment'],
                    field_calibration=False,observed_counterfactual_baseline_used=False))
            evaluations.append(dict(case_id=cid,snapshot_id=snaps[ident]['id'],policy=policy,conditional_state=False,
                actual_payment=sum(x['payment'] for x in receipts),history_id=regime['id'],
                target_family=target['effective_supports'],truth_used_only_after_identifier=True))
    if len(cases)!=36 or len(evaluations)!=72 or len(snaps)>33:raise AssertionError('locked denominator mismatch')
    write(root/'evaluator_truth.json',dict(cases=cases,evaluations=evaluations,identifier_input=False))
    write(root/'config.json',dict(status='root_frozen_before_solver',settings=cfg['fit_settings'],histories=histories,
        snapshots=list(snaps.values()),generation_protocol=dict(path=str(root/'generation_protocol.json'),sha256=sha(root/'generation_protocol.json')),
        truth_input_allowed=False,price_optimized=False))
    result=dict(status='generated',case_units=len(cases),evaluations=len(evaluations),unique_history_packets=len(histories),
        unique_future_snapshots=len(snaps),historical_rows_per_split=288,current_rows_per_split=480,
        all_legacy_load_control_random_streams_retained=True,impedance_scale=1.,
        matched_history_hashes={h['id']:h['sha256'] for h in histories},LP_calls=0,denseQP_calls=0,AC_calls=0)
    write(root/'generation_summary.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','generate']);p.add_argument('--root',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.root) if a.mode=='prepare' else generate(a.root)))
