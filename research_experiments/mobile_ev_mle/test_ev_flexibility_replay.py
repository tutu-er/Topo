"""Meaningful negative controls for independent flexibility evidence auditing."""
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
import numpy as np
import pytest
from . import ev_flexibility_replay as audit
from .ev_service_milp import EVServiceConstraints
from .likelihood import ac_predict, RadialCandidate
from .information_context import meter_to_record
from .simulation import model


def battery_action(u,site=0):
    c=EVServiceConstraints(terminal_energy=1.3425,p_discharge_max=.3)
    d=asdict(c);expense=0. if site==0 else .01
    p=np.zeros((4,3));p[:,site]=u;u=np.asarray(u)
    energy=np.r_[1.2-expense,1.2-expense+np.cumsum(.25*(.95*np.maximum(u,0)-np.maximum(-u,0)/.95))]
    ready=0 if site==0 else 1
    return dict(site=site,p=p.tolist(),q=np.zeros_like(p).tolist(),sites=np.where(np.arange(4)>=ready,site,-1).tolist(),soc=(energy/3).tolist(),solver={}),d


def test_fixed_terminal_service_and_v2g_losses_are_independently_reconstructed():
    constant,d=battery_action([.15]*4);v2g,_=battery_action([.3,.3,.3,-.27075])
    a,b=audit.operation_audit(constant,d),audit.operation_audit(v2g,d)
    assert a['terminal_energy_kwh']==pytest.approx(b['terminal_energy_kwh'])
    assert a['grid_charge_kwh']==pytest.approx(3.)
    assert b['grid_discharge_kwh']>0 and b['throughput_kwh']>a['throughput_kwh']
    assert b['grid_net_kwh']>a['grid_net_kwh']


@pytest.mark.parametrize('damage',['terminal','simultaneous','access','wrong_soc','Q'])
def test_invalid_saved_operations_do_not_pass(damage):
    p,d=battery_action([.15]*4)
    if damage=='terminal':d['terminal_energy']+=.01
    elif damage=='simultaneous':p['solver']={'split_charge_pu':[.16]*4,'split_discharge_pu':[.01]*4}
    elif damage=='access':d['availability']=[[False,True,True]]*4
    elif damage=='wrong_soc':p['soc'][2]+=.001
    else:p['q'][0][0]=.001
    with pytest.raises(ValueError):audit.operation_audit(p,d)


def test_driving_energy_is_restored_before_common_departure_target():
    u=[0.]+[.1525/(.95*.25*3)]*3;p,d=battery_action(u,site=1)
    r=audit.operation_audit(p,d)
    assert r['drive_kwh']==pytest.approx(.2)
    assert r['terminal_energy_kwh']==pytest.approx(26.85)
    assert r['grid_charge_kwh']==pytest.approx(3.+.2/.95)


def test_selection_failure_and_expanded_guard_are_not_AC_design_success():
    r={'method':'charge_mobile','selection':{'status':'selection_failure_fallback'}}
    assert audit.selection_kind(r,{'activated':True})=='fallback_after_design_failure'
    assert audit.selection_kind(r,{'activated':False})=='not_activated'
    r['selection']['status']='expanded_domain_incumbent_guard'
    assert audit.selection_kind(r,{'activated':True})=='expanded_domain_guard'


def paired_fixture():
    trials=[]
    for h,values in [('a',[1.,2.,3.]),('b',[10.,20.,30.])]:
        for quality,value in zip(audit.QUALITY,values):
            trials.append(dict(history_id=h,quality=quality,records=[dict(method='charge_fixed',status='complete',signed_margin_full_SSE=value),dict(method='constant_fixed',status='complete',signed_margin_full_SSE=0.)]))
    return trials


def test_three_quality_rows_form_two_independent_history_pairs():
    r=audit.grouped_pairs(paired_fixture(),'charge_fixed','constant_fixed')
    assert r['complete_H']==r['planned_H']==2
    assert r['H_mean_effects']=={'a':2.,'b':20.}
    assert r['mean_effect']==11. and r['between_H_descriptive_SE']==9.


def test_failed_condition_removes_no_planned_history_from_denominator():
    rows=paired_fixture();rows[-1]['records'][0]['status']='joint_fit_failure'
    r=audit.grouped_pairs(rows,'charge_fixed','constant_fixed')
    assert r['complete_H']==1 and r['planned_H']==2 and len(r['failed_conditions'])==1


def test_snapshot_bytes_and_source_cartesian_coverage_are_checked(tmp_path,monkeypatch):
    root=tmp_path/'research_experiments'/'checker';root.mkdir(parents=True);monkeypatch.setattr(audit,'HERE',root)
    hashes={};sources={}
    for i in range(21):
        name=f'source{i}.py';data=f'# source {i}\n'.encode();(root/name).write_bytes(data)
        hashes[name]=hashlib.sha256(data).hexdigest();sources[name]=gzip.compress(data,mtime=0).hex()
    directory=tmp_path/'records';directory.mkdir();snapshot={'encoding':'gzip_hex','hashes':hashes,'sources':sources}
    path=directory/'source_snapshot.json.gz';path.write_bytes(gzip.compress(json.dumps(snapshot).encode(),mtime=0))
    assert audit.source_audit(directory,{'source_sha256':hashes})['source_count']==21
    snapshot['sources'].pop('source0.py');path.write_bytes(gzip.compress(json.dumps(snapshot).encode(),mtime=0))
    with pytest.raises(ValueError,match='coverage'):audit.source_audit(directory,{'source_sha256':hashes})


def physics_fixture():
    candidate=RadialCandidate.from_supports('truth',3,())
    r=np.full(candidate.m,.01);x=np.full(candidate.m,.007)
    p=np.full((16,3),.12);q=np.full((16,3),.04);sp=np.full(16,.01);sq=np.full(16,.003)
    action,_=battery_action([.15]*4);total=p[12:]+np.asarray(action['p']);reactive=q[12:]
    physical=ac_predict(candidate,r,x,total,reactive,sp[12:],sq[12:],root_v=1.)
    meter=model(3);white={k:np.full((4,3) if k in ('p','q','v') else (4,),.17) for k in ('p','q','v','master_p','master_q')}
    observations=dict(root_v=[1.]*4,source_p=None,source_q=None)
    saved_physical={}
    for k in white:
        saved_physical[k]=np.asarray(physical[k]).tolist();mean=np.asarray(physical[k])
        if k in ('p','q','v'):mean=mean+getattr(meter,'bias_'+k)
        observations[k]=(mean+getattr(meter,k+'_sigma')*white[k]).tolist()
    trial=dict(truth=dict(p=p.tolist(),q=q.tolist(),source_p=sp.tolist(),source_q=sq.tolist(),supports=[],r_pu=r.tolist(),x_pu=x.tolist()),measurement_models={'future':meter_to_record(meter)},white_future_noise={k:v.tolist() for k,v in white.items()})
    row=dict(action=action,physical=saved_physical,observations=observations,voltage_min=float(abs(physical['voltage']).min()),voltage_max=float(abs(physical['voltage']).max()))
    return trial,row


def test_different_branch_sweep_checks_true_response_and_exactly_one_EV_injection():
    trial,row=physics_fixture();assert audit.future_measurement_audit(trial,row)['success']
    row['physical']['p'][0][0]+=.15
    with pytest.raises(ValueError,match='EV exactly once'):audit.future_measurement_audit(trial,row)


def test_future_bias_and_noise_are_checked_separately_from_physical_generation():
    trial,row=physics_fixture();row['observations']['v'][0][0]+=.001
    with pytest.raises(ValueError,match='future meter:v'):audit.future_measurement_audit(trial,row)


def coverage_fixture():
    obs={k:[[0.,0.,0.]] if k in ('p','q','v') else [0.] for k in ('p','q','v','master_p','master_q')}
    freeze=dict(seeds=[1,2],methods=list(audit.METHODS),quality=list(audit.QUALITY),config={},source_sha256={})
    rows=[]
    for h in ('1','2'):
        for q in audit.QUALITY:
            rows.append(dict(history_id=h,quality=q,run_id=h+'_'+q,config={},source_sha256={},old_threshold_applied=False,selector_uses_truth=False,selector_uses_future_noise=False,selection=dict(selector_uses_truth=False,selector_uses_future_noise=False),independent_history_unit=True,multiple_quality_rows_are_not_independent_histories=True,history_hash=audit.fingerprint(obs),history_observations=deepcopy(obs),historical_fits=[],candidates=[],truth={},white_future_noise={},records=[dict(method=m) for m in audit.METHODS]))
    return rows,freeze


@pytest.mark.parametrize('damage',['missing_quality','duplicate_method','source_change','sameH_fit_change'])
def test_incomplete_or_inconsistent_saved_coverage_is_rejected(damage):
    rows,freeze=coverage_fixture();assert audit.validate_coverage(rows,freeze,2)==['1','2']
    if damage=='missing_quality':rows.pop()
    elif damage=='duplicate_method':rows[0]['records'][0]['method']=rows[0]['records'][1]['method']
    elif damage=='source_change':rows[0]['source_sha256']={'wrong':'source'}
    else:rows[0]['candidates']=[{'name':'changed'}]
    with pytest.raises(ValueError):audit.validate_coverage(rows,freeze,2)


def test_compact_removes_only_named_selection_diagnostics_and_keeps_fit_states():
    trial=dict(selection=dict(context=dict(local_scaled_singular_values=[1.,2.],rank=2),bounds=[0.,1.],local_projected_shared_singular_values=[3.]),historical_fits=[dict(p=[[1.]],local_scaled_singular_values=[4.])],records=[dict(physical={'v':[.99]},fits=[dict(source_p=[.1],r=[.01])])])
    removed=[];result=audit.compact_trial(trial,removed=removed,selection_only=True)
    assert len(removed)==2 and result['selection']=={'context':{'rank':2},'bounds':[0.,1.]}
    assert result['historical_fits']==trial['historical_fits'] and result['records']==trial['records']
    assert 'local_scaled_singular_values' in trial['selection']['context']  # original untouched


def test_compact_scientific_comparison_rejects_changed_results():
    left={'methods':{'charge_fixed':{'mean':1.}},'seconds':1.,'input_sha256':{'a':'raw'},'compact_manifest':None}
    right={'methods':{'charge_fixed':{'mean':1.}},'seconds':2.,'input_sha256':{'a':'compact'},'compact_manifest':{'success':True}}
    assert audit.compare_scientific_summaries(left,right)['success']
    right['methods']['charge_fixed']['mean']=2.
    with pytest.raises(ValueError,match='scientific results differ'):audit.compare_scientific_summaries(left,right)


def test_global_compact_drops_exactly_two_diagnostics_without_state_loss():
    raw={'selection':{'local_scaled_singular_values':[1.]},'fits':[{'local_scaled_singular_values':[2.],'local_projected_shared_singular_values':[3.],'rank':1,'r':[.01],'p':[[.1]],'loss':2.}]}
    removed=[];small=audit.compact_trial(raw,removed=removed)
    assert len(removed)==3
    assert small['fits']==[{'rank':1,'r':[.01],'p':[[.1]],'loss':2.}]


def test_formal_single_quality_does_not_require_three_rows_per_history():
    rows=[dict(history_id=h,quality='uniform',records=[dict(method='charge_fixed',status='complete',signed_margin_full_SSE=v),dict(method='constant_fixed',status='complete',signed_margin_full_SSE=0.)]) for h,v in [('a',2.),('b',20.)]]
    value=audit.grouped_pairs(rows,'charge_fixed','constant_fixed',quality_count=1)
    assert value['complete_H']==2 and value['within_H_quality_count']==1 and value['mean_effect']==11.


def test_formal_envelope_validates_one_quality_and_policy_subset_without_using_outcomes():
    policy={'science.py':'x'};hashes={**policy,'ev_flexibility_formal.py':'y','ev_flexibility_formal_protocol.md':'z'}
    label={'seed':1,'case':'star','quality':'temporal','family':[]}
    obs={k:[[0.,0.,0.]] if k in ('p','q','v') else [0.] for k in ('p','q','v','master_p','master_q')}
    freeze={'schema':'ev_service_formal_v1','methods':list(audit.METHODS),'source_sha256':hashes,'policy_id':'fixed','schedule':{'evaluation':[label]},'policy_config':{}}
    record={'history_id':'1','run_id':'1_temporal','case':'star','quality':'temporal','formal':{'policy_id':'fixed','phase':'evaluation','labels':label,'runner_kind':'real','one_quality_per_H':True,'old_threshold_applied':False},'source_sha256':policy,'config':{},'history_observations':obs,'history_hash':audit.fingerprint(obs),'truth':{'supports':[]},'selector_uses_truth':False,'selector_uses_future_noise':False,'records':[{'method':m} for m in audit.METHODS]}
    assert audit.validate_formal_coverage([record],freeze,'evaluation')==['1']
    record['formal']['runner_kind']='test_double'
    with pytest.raises(ValueError):audit.validate_formal_coverage([record],freeze,'evaluation')


def test_runner_exception_keeps_unknown_cost_and_all_nine_failure_rows():
    record={'runner_exception':'original failure','records':[{'method':m,'status':'runner_exception'} for m in audit.METHODS],'candidates':[],'historical_fits':[],'total_ac_calls':None,'history_id':'1','run_id':'1_uniform','quality':'uniform','case':'weak'}
    replay=audit.audit_trial(record)
    assert replay['algorithm_ac_calls'] is None and len(replay['methods'])==9
    assert all(r['selection_kind']=='runner_exception' for r in replay['methods'])
    record['total_ac_calls']=0
    with pytest.raises(ValueError,match='unknown runner cost'):audit.audit_trial(record)
