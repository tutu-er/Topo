"""Independent saved-cache replay and retained-slot denominators."""
from copy import deepcopy
from hashlib import sha256
from itertools import product

import numpy as np
import pytest

from . import ev_mobile_v4_cached_audit as audit
from . import ev_mobile_v2_review as physical
from . import gradient_price_study as io
from .likelihood import RadialCandidate


def fixture(tmp_path,*,failed=False):
    root=tmp_path/'cached';(root/'history_000'/'regular').mkdir(parents=True)
    (root/'targets').mkdir();(root/'fit_cache').mkdir()
    names=('a','b','c','d');bank={n:RadialCandidate.from_supports(n,2,()) for n in names}
    meter=dict(p_sigma=.001,q_sigma=.001,v_sigma=.001,master_p_sigma=.001,master_q_sigma=.001,
               bias_p=0.,bias_q=0.,bias_v=0.)
    parameters=dict(r=[.01,.02,.015],x=[.003,.006,.0045],p=[[.1,.08]],q=[[.03,.02]],sp=[.05],sq=[.02],root=[1.])
    cfg=dict(training=dict(r_bounds=[.001,.1],x_bounds=[.0001,.1],voltage_bounds=[.85,1.15]),
        source=dict(source_p_bounds=[-.5,.5],source_q_bounds=[-.5,.5]),ratio_bounds=[.1,.4],source_interface=1)
    def observation(p):
        phy=physical.physical.forward_physics(bank['a'],parameters['r'],parameters['x'],[[p,.08]],parameters['q'],
            parameters['sp'],parameters['sq'],parameters['root'])
        return dict(p=[[p,.08]],q=parameters['q'],v=phy['v'],master_p=phy['master_p'],master_q=phy['master_q'],root_v=[1.])
    D,future=observation(.1),observation(.105)
    one_state=dict(r=parameters['r'],x=parameters['x'],source_p=[.05],source_q=[.02])
    trained={}
    for name,candidate in bank.items():
        score=physical.replay_state(candidate,one_state,(('observed',D,meter),),cfg)
        trained[name]=dict(**score,name=name,supports=candidate.supports,success=True,**deepcopy(one_state),
            p=D['p'],q=D['q'],predicted={k:D[k] for k in ('p','q','v','master_p','master_q')})
    options={**cfg['training'],**cfg['source'],'ratio_bounds':cfg['ratio_bounds']}
    methods=('m0','m1');power=[[.1,0.]]
    meta=dict(spec=dict(index=0),candidate_names=list(names),true_name='a',source_interface=1,
        candidate_bank=[dict(name=n,n=2,supports=bank[n].supports) for n in names],observed=D,
        observed_meter=meter,future_meter=meter,trained=trained,predicted=parameters,
        true_by_regime=dict(regular=deepcopy(parameters)),powers={m:power for m in methods},
        fit_options=options,fit_seed=7,base_kw=20.)
    target=dict(success=True,observations=future,parameters=parameters,power_kw=power,
        true_reference='a',physical=dict(future_white_noise=False))
    target['target_request']=audit._target_request(meta,target)
    target_path=root/'targets'/(target['target_request']+'.json.gz')
    state=dict(r=parameters['r'],x=parameters['x'],source_p=[.05,.05],source_q=[.02,.02])
    blocks=(('observed',D,meter),('future',future,meter));comparisons={}
    for name,candidate in bank.items():
        score=physical.replay_state(candidate,state,blocks,cfg)
        fixed=dict(**score,usable=True,state=deepcopy(state))
        joint=dict(**score,success=not(failed and name=='b'),state=deepcopy(state),feasible_witness=deepcopy(fixed))
        comparisons[name]=dict(fixed=fixed,joint=joint,fixed_source_success=True)
    identity=audit._fit_request(meta,target)
    cached=dict(fit_identity=identity,fit_seed=7,outcome=dict(complete=not failed,comparisons=comparisons,
        losses={name:r['joint']['loss'] for name,r in comparisons.items()},failures={}))
    fit_path=root/'fit_cache'/(identity+'.json.gz')
    physical.write(root/'freeze.json',dict(indices=[0],regimes=['regular'],methods=list(methods),source_sha256={},policy_id='p'))
    physical.write(root/'history_000'/'history.json.gz',io.serial(meta))
    physical.write(target_path,io.serial(target));physical.write(fit_path,io.serial(cached))
    for bits in product((0,1),repeat=3):
        state=''.join(map(str,bits))
        for method in methods:
            slot=dict(index=0,regime='regular',method=method,state=state,factors=dict(zip(audit.FACTORS,bits)),
                target_request=target['target_request'],target_source=dict(sha256=sha256(target_path.read_bytes()).hexdigest()),
                fit_identity=identity,fit_source=dict(sha256=sha256(fit_path.read_bytes()).hexdigest()),
                complete=not failed,status='fit_failure' if failed else 'complete',
                candidate_slots=dict(expected=4,attempted=4,success=3 if failed else 4,failed=1 if failed else 0,not_run=0),
                M=None if failed else 0.,halfM=None if failed else 0.)
            physical.write(root/'history_000'/'regular'/(method+'_'+state+'.json'),slot)
    return root


def test_all_slots_remain_when_sixteen_targets_reuse_one_four_candidate_fit(tmp_path):
    root=fixture(tmp_path)
    result=audit.audit_directory(root,tmp_path/'audit.json')
    assert result['success'],result['errors']
    assert result['expected_target_slots']==16 and result['expected_joint_candidate_slots']==64
    assert result['unique_fit_cache_files']==1 and result['unique_joint_candidate_denominator']==4
    assert result['metrics']['saved_states_replayed']==13  # 4 D + 1 target + 4 fixed + 4 joint.
    assert result['metrics']['AC_block_replays']==21
    assert result['metrics']['slot_margin_checks']==16
    assert result['optimizer_calls']==result['production_AC_calls']==result['new_simulation_calls']==0


def test_declared_optimizer_failure_retains_slot_and_unique_candidate_counts(tmp_path):
    root=fixture(tmp_path,failed=True)
    result=audit.audit_directory(root,tmp_path/'audit.json')
    assert result['success'],result['errors']
    assert result['expected_joint_candidate_slots']==64
    assert result['metrics']['recorded_failed_joint_candidate_slots']==16
    assert result['metrics']['unique_failed_joint_candidates']==1
    assert len(result['recorded_failures'])==16


def test_missing_slot_stays_in_original_denominator(tmp_path):
    root=fixture(tmp_path)
    (root/'history_000'/'regular'/'m0_000.json').unlink()
    result=audit.audit_directory(root,tmp_path/'audit.json')
    assert not result['success']
    assert result['completed_target_slots']==15 and result['expected_joint_candidate_slots']==64
    assert result['missing'] and result['unique_joint_candidate_denominator']==4


def test_joint_state_tamper_detected_even_with_updated_file_hash_links(tmp_path):
    root=fixture(tmp_path)
    fit_path=next((root/'fit_cache').glob('*.json.gz'))
    fitted=physical.read(fit_path)
    fitted['outcome']['comparisons']['a']['joint']['state']['r']=[.02,.02,.015]
    physical.write(fit_path,fitted)
    for path in (root/'history_000'/'regular').glob('*.json'):
        slot=physical.read(path);slot['fit_source']['sha256']=sha256(fit_path.read_bytes()).hexdigest()
        physical.write(path,slot)
    result=audit.audit_directory(root,tmp_path/'audit.json')
    assert not result['success']
    assert any('joint cached state' in row['check'] for row in result['errors'])


def noise_fixture(tmp_path):
    source=fixture(tmp_path)
    meta=physical.read(source/'history_000'/'history.json.gz')
    meta['observed_meter']['bias_v']=.0002;meta['future_meter']['bias_v']=.0002
    meta['observed']['v']=(np.asarray(meta['observed']['v'])+.0002).tolist()
    for row in meta['trained'].values():
        row['predicted']['v']=(np.asarray(row['predicted']['v'])+.0002).tolist()
    meta['powers']['m1']=[[.2,0.]]
    root=tmp_path/'noise';root.mkdir()
    parameters=meta['true_by_regime']['regular'];bank=audit._bank(meta);cfg=audit._cfg(meta)
    methods=('m0','m1');seeds=(17,18);clean={}
    for method in methods:
        p=np.asarray(parameters['p'])+np.asarray(meta['powers'][method])/meta['base_kw']
        phy=physical.physical.forward_physics(bank['a'],parameters['r'],parameters['x'],p,parameters['q'],
            parameters['sp'],parameters['sq'],parameters['root'])
        clean[method]=io.serial(dict(p=p,q=parameters['q'],v=phy['v']+.0002,
            master_p=phy['master_p'],master_q=phy['master_q'],root_v=parameters['root']))
    shapes={name:np.asarray(clean['m0'][name]).shape for name in audit.NOISE_CHANNELS}
    inputs=dict(meta=meta,clean=clean,shapes=shapes)
    physical.write(root/'inputs.json.gz',inputs)
    frozen=dict(schema='mobile_v4_development_conditional_noise_freeze_v1',seeds=list(seeds),methods=list(methods),
        history_index=0,regime='regular',target_slots=4,candidate_slots=16,
        inputs_sha256=audit.fingerprint(inputs),source_sha256={},policy_id='p')
    physical.write(root/'freeze.json',frozen)
    for i,seed in enumerate(seeds):
        white={name:np.full(shape,.1*(j+1)*(i+1)).tolist() for j,(name,shape) in enumerate(shapes.items())}
        white_path=root/'white_noise'/(str(seed)+'.json.gz')
        physical.write(white_path,dict(seed=seed,white=white,white_sha256=audit.fingerprint(white)))
        for method in methods:
            noisy=deepcopy(clean[method])
            for name in audit.NOISE_CHANNELS:
                noisy[name]=(np.asarray(noisy[name])+np.asarray(meta['future_meter'][name+'_sigma'])*white[name]).tolist()
            target=dict(success=True,observations=noisy,white_source=dict(sha256=sha256(white_path.read_bytes()).hexdigest()),
                clean_observation_sha256=audit.fingerprint(clean[method]),history_sha256=audit.fingerprint(meta['observed']))
            target_path=root/'targets'/str(seed)/(method+'.json.gz');physical.write(target_path,target)
            identity=audit._fit_request(meta,target);comparisons={}
            state=dict(r=parameters['r'],x=parameters['x'],source_p=[.05,.05],source_q=[.02,.02])
            blocks=(('observed',meta['observed'],meta['observed_meter']),('future',noisy,meta['future_meter']))
            for name,candidate in bank.items():
                score=physical.replay_state(candidate,state,blocks,cfg)
                fixed=dict(**score,usable=True,state=deepcopy(state))
                joint=dict(**score,success=True,state=deepcopy(state),feasible_witness=deepcopy(fixed))
                comparisons[name]=dict(fixed=fixed,joint=joint,fixed_source_success=True)
            fitted=dict(fit_identity=identity,fit_seed=meta['fit_seed'],outcome=dict(complete=True,comparisons=comparisons,
                losses={name:row['joint']['loss'] for name,row in comparisons.items()},failures={}))
            fit_path=root/'fit_cache'/(identity+'.json.gz');physical.write(fit_path,io.serial(fitted))
            slot=dict(seed=seed,method=method,history_index=0,regime='regular',white_sha256=audit.fingerprint(white),
                target_source=dict(sha256=sha256(target_path.read_bytes()).hexdigest()),fit_identity=identity,
                fit_source=dict(sha256=sha256(fit_path.read_bytes()).hexdigest()),complete=True,status='complete',
                candidate_slots=dict(expected=4,attempted=4,success=4,failed=0,not_run=0),M=0.,halfM=0.)
            physical.write(root/'replicates'/str(seed)/(method+'.json'),slot)
    return root


def test_noise_schema_replays_measured_inputs_and_same_seed_common_white(tmp_path):
    root=noise_fixture(tmp_path)
    result=audit.audit_noise_directory(root,tmp_path/'noise_audit.json')
    assert result['success'],result['errors']
    assert result['independent_history_denominator']==1 and result['conditional_noise_seed_denominator']==2
    assert result['expected_target_slots']==4 and result['expected_joint_candidate_slots']==16
    assert result['metrics']['saved_states_replayed']==38 and result['metrics']['AC_block_replays']==70
    assert result['metrics']['paired_white_seed_checks']==2
    assert result['metrics']['noise_channel_relation_checks']==20
    assert result['metrics']['max_noise_relation_error']==0.
    assert not result['noisy_target_zero_residual_required']
    assert any(row['replay_loss']>0. for row in result['states'] if row['check'].startswith('joint/'))
    assert all(row['measured_input_noise_RMS_pu']['p']>0. for row in result['slots'])
    assert result['optimizer_calls']==result['production_AC_calls']==result['new_simulation_calls']==0


def test_noise_relation_detects_double_bias_and_final_audit_waits_for_all_slots(tmp_path):
    root=noise_fixture(tmp_path);inputs=physical.read(root/'inputs.json.gz')
    target=physical.read(root/'targets'/'17'/'m0.json.gz')
    white=physical.read(root/'white_noise'/'17.json.gz')['white']
    noisy=deepcopy(target['observations']);noisy['v']=(np.asarray(noisy['v'])+.0002).tolist()
    with pytest.raises(ValueError,match='bias/EV once'):
        audit.check_noise_relation(inputs['clean']['m0'],noisy,inputs['meta']['future_meter'],white)
    (root/'replicates'/'17'/'m0.json').unlink()
    output=tmp_path/'too_early.json'
    with pytest.raises(ValueError,match='every declared noise slot'):
        audit.audit_noise_directory(root,output)
    assert not output.exists()
