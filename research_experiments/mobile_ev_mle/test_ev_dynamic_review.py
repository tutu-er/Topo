"""Synthetic finite-state evidence for the independent dynamic reviewer.

No fitter or policy is used to manufacture these fixtures. Saved source states
are arbitrary finite explanations whose independent AC costs are known.
"""
import base64
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json

import numpy as np
import pytest

from . import ev_dynamic_review as review
from . import ev_dynamic_loads as loads
from .likelihood import MeasurementModel, Observations, RadialCandidate


def fixture():
    H,V,T,n,bkw=8,4,6,3,20.
    cfg=dict(training_points=H,validation_points=V,horizon=T,forecast_gap=V,
        forecast='ar1',mean_policy_ablation=False,diagnostics=False,
        source_interface=1,PQ_inputs_fixed=True,
        source=dict(source_p_bounds=[-.5,.5],source_q_bounds=[-.5,.5]),
        select_modes=['none','constant'])
    device=dict(base_kw=bkw,dt_hours=.25,eta_charge=.95,eta_discharge=.95,
        initial_energy_kwh=24.,terminal_energy_kwh=26.85,capacity_kwh=60.,
        command_power_kw=6./1.05,throughput_kwh=7.5,drive_kwh=.2,max_moves=1,
        travel_slots=1,setup_slots=1,travel_worker_minutes=15.,setup_worker_minutes=15.,
        driver_budget_minutes=30.,min_arrival_throughput_kwh=.25)
    meter=MeasurementModel(.0006,.0004,.00012,.001,.001,
        bias_p=[-.00015,0.,.00015],bias_q=[.0001,0.,-.0001],bias_v=[-.00004,0.,.00004])
    fm=MeasurementModel(.0006/np.sqrt(5),.0004/np.sqrt(5),np.full((T,n),.00012/np.sqrt(5)),
        .001/np.sqrt(5),.001/np.sqrt(5),bias_p=meter.bias_p,bias_q=meter.bias_q,bias_v=meter.bias_v)
    bank=[RadialCandidate.from_supports('a',n,()),RadialCandidate.from_supports('b',n,((0,1),))]
    lines={c.name:dict(r=np.full(c.m,.004),x=np.full(c.m,.001)) for c in bank}
    background=loads.generate_background(319,H+V+T,n)
    p,q=np.asarray(background['p_kw'])/bkw,np.asarray(background['q_kvar'])/bkw
    sp,sq=np.asarray(background['source_p_kw'])/bkw,np.asarray(background['source_q_kvar'])/bkw
    root=background['root_v'];physical=review.forward_physics(bank[0],**lines['a'],p=p,q=q,sp=sp,sq=sq,root=root)
    rng=np.random.default_rng(419)
    white={k:rng.standard_normal((H+V+T,n) if k in ('p','q','v') else H+V+T) for k in review.CHANNELS}
    def observation(phy,met,noise):
        values={k:np.asarray(phy[k])+np.asarray(getattr(met,k+'_sigma'))*noise[k]
            +(np.asarray(getattr(met,'bias_'+k)) if k in ('p','q','v') else 0.) for k in review.CHANNELS}
        return Observations(**values,root_v=np.asarray(phy['root_v']))
    allobs=observation(physical,meter,white)
    def sliced_obs(obs,part):return Observations(**{k:np.asarray(getattr(obs,k))[part] for k in (*review.CHANNELS,'root_v')})
    tr,va=sliced_obs(allobs,slice(0,H)),sliced_obs(allobs,slice(H,H+V))
    def record_fit(candidate,obs,met,part):
        pp=obs.p-np.asarray(met.bias_p);qq=obs.q-np.asarray(met.bias_q)
        phy=review.forward_physics(candidate,**lines[candidate.name],p=pp,q=qq,sp=sp[part],sq=sq[part],root=obs.root_v)
        means={k:np.asarray(phy[k])+(np.asarray(getattr(met,'bias_'+k)) if k in ('p','q','v') else 0.) for k in review.CHANNELS}
        loss=0.;normalizer=0.
        for k in ('v','master_p','master_q'):
            y=np.asarray(getattr(obs,k));sigma=np.broadcast_to(getattr(met,k+'_sigma'),y.shape)
            loss+=float(np.sum(((means[k]-y)/sigma)**2));normalizer+=float(np.log(2*np.pi*sigma**2).sum())
        return dict(name=candidate.name,candidate=candidate.name,supports=candidate.supports,**lines[candidate.name],
            p=pp,q=qq,source_p=sp[part],source_q=sq[part],bias_p=met.bias_p,bias_q=met.bias_q,bias_v=met.bias_v,
            predicted=means,loss=loss,half_loss=loss/2,loglik=-.5*(loss+normalizer),success=True,
            global_MLE_certified=False,diagnostics=dict(source_interface=1,objective_channels=['v','master_p','master_q']))
    def fit_block(obs,met,part):return dict(fits=[record_fit(c,obs,met,part) for c in bank],failures={},all_complete=True)
    train_block=fit_block(tr,meter,slice(0,H));val_block=fit_block(va,meter,slice(H,H+V))
    train_fits={r['name']:r for r in train_block['fits']}
    fore=loads.fit_forecast(tr,meter,train_fits,horizon=T,gap=V,base_kw=bkw)
    rows=[]
    high=np.zeros((T,n));high[0,0]=high[2,0]=device['command_power_kw'];high[4,0]=12.-2*device['command_power_kw']
    constant=np.zeros((T,n));constant[:,0]=2.
    for name,power in (('none',np.zeros((T,n))),('constant',constant),('highlow',high)):
        phy=review.forward_physics(bank[0],**lines['a'],p=p[H+V:]+power/bkw,q=q[H+V:],sp=sp[H+V:],sq=sq[H+V:],root=root[H+V:])
        phy={k:phy[k] for k in (*review.CHANNELS,'root_v')}
        actual=observation(phy,fm,{k:v[H+V:] for k,v in white.items()})
        clean=review._observation(phy,fm)
        noisy,cleanfit=fit_block(actual,fm,slice(H+V,None)),fit_block(clean,fm,slice(H+V,None))
        valcost={r['name']:r['loss'] for r in val_block['fits']}
        combined={r['name']:r['loss']+valcost[r['name']] for r in noisy['fits']}
        cleancombined={r['name']:r['loss']+valcost[r['name']] for r in cleanfit['fits']}
        if name=='none':op=dict(success=True,battery_service=False)
        else:
            C=float(.25*power.sum());energy=np.r_[24.,24.+np.cumsum(.95*.25*power.sum(axis=1))]
            op=dict(success=True,charging_kwh=C,discharging_kwh=0.,grid_energy_kwh=C,throughput_kwh=C,
                driving_kwh=0.,moves=0,worker_minutes=0.,energy_kwh=energy,
                power_variance=float(np.var(power.sum(axis=1))),charging_efficiency=.95,discharging_efficiency=.95)
        rows.append(dict(name=name,status='complete',power_kw=power,operation=op,physical=phy,
            observations={k:getattr(actual,k) for k in review.OBS_CHANNELS},noisy=noisy,noiseless=cleanfit,
            M=review._margin(combined,'a'),clean_future_M=review._margin(cleancombined,'a'),loss_by_candidate=combined))
    payload=(review.HERE/'ev_dynamic_loads.py').read_bytes();hashes={'ev_dynamic_loads.py':hashlib.sha256(payload).hexdigest()}
    frozen=dict(config=cfg,source_sha256=hashes,policy_id=review._identity(dict(config=cfg,source_sha256=hashes)))
    snapshot=dict(source_sha256=hashes,source_base64={'ev_dynamic_loads.py':base64.b64encode(payload).decode()})
    record=dict(schema='ev_ordinary_dynamic_v1',seed=991111,status='complete',config=cfg,
        policy_id=frozen['policy_id'],source_sha256=hashes,source_unchanged=True,
        training={k:getattr(tr,k) for k in review.OBS_CHANNELS},training_meter=asdict(meter),
        validation={k:getattr(va,k) for k in review.OBS_CHANNELS},validation_meter=asdict(meter),future_meter=asdict(fm),
        training_result=train_block,validation_result=val_block,background=background,
        candidates=[dict(name=c.name,supports=c.supports) for c in bank],true_name='a',candidate_miss=False,
        truth=dict(supports=bank[0].supports,**lines['a']),seeds=dict(background=319,noise=419),forecast=fore,
        selection=dict(configuration=device,uses_validation=False,uses_truth=False,uses_future_noise=False,
            random_used_by_primary=False,line_parameters_unchanged=True,predicted_AC={},methods={},all_prediction_records=[]),
        methods=rows,contrasts={},training_parameter_error=dict(R=0.,X=0.),
        actual_solver_calls=dict(AC_batches=0,AC_snapshots=0,AC_failures=0,QP_calls=0,
            independent_replay_AC_batches=16,independent_replay_AC_snapshots=104,independent_replay_AC_failures=0),
        validation_used_by_selector=False,future_noise_used_by_selector=False,one_trained_parameter_vector=True)
    return review._serial(record),review._serial(frozen),snapshot


def _failed(result,part):
    return [c for c in result['checks'] if part in c['name'] and c['status']=='failed']


def price_fixture():
    trial,freeze,snapshot=fixture();fore=trial['forecast'];meter=review._meter(trial['future_meter'])
    trained={r['name']:r for r in trial['training_result']['fits']}
    bank={r['name']:review._candidate(r['name'],r['supports'],3) for r in trial['candidates']}
    u=np.zeros((6,3));u[:,0]=2.;entries=[]
    for a,b in (('a','b'),('b','a')):
        sa,sb=fore['source_by_candidate'][a],fore['source_by_candidate'][b]
        def score(power):
            p=np.asarray(fore['p'])+power/20.;q=np.asarray(fore['q'])
            reference=review.forward_physics(bank[a],trained[a]['r'],trained[a]['x'],p,q,sa['source_p'],sa['source_q'],fore['root'])
            competitor=review.forward_physics(bank[b],trained[b]['r'],trained[b]['x'],p,q,sb['source_p'],sb['source_q'],fore['root'])
            return .5*sum(float(np.sum(((competitor[k]-reference[k])/np.asarray(getattr(meter,k+'_sigma')))**2)) for k in ('v','master_p','master_q'))
        value=score(u);gradient=np.zeros_like(u);eps=2e-3
        for t,j in np.ndindex(u.shape):
            plus=u.copy();minus=u.copy();plus[t,j]+=eps;minus[t,j]-=eps
            gradient[t,j]=(score(plus)-score(minus))/(2*eps)
        normalizer=sum(float(np.log(2*np.pi*np.broadcast_to(getattr(meter,k+'_sigma'),(6,3) if k=='v' else (6,))**2).sum()) for k in ('v','master_p','master_q'))
        entries.append(dict(name=a+'__to__'+b,a=a,b=b,weight=.5,source_p=sb['source_p'],source_q=sb['source_q'],
            source_solver=dict(success=True,r=trained[b]['r'],x=trained[b]['x'],loss=2*value,loglik=-.5*(2*value+normalizer)),
            half_SSE=value,gradient_per_kw=gradient))
    mean=.5*(entries[0]['gradient_per_kw']+entries[1]['gradient_per_kw'])
    trial['price_diagnostics']={'constant':dict(complete_AC=dict(success=True,power_kw=u,pairs=entries,
        gradient_per_kw=mean,mean_half_SSE=.5*sum(r['half_SSE'] for r in entries),temporal_spread_by_site=np.ptp(mean,axis=0)))}
    return review._serial(trial),freeze,snapshot


def test_price_witness_residual_FD_matches_independent_scalar_cost_FD(monkeypatch):
    trial,freeze,snapshot=price_fixture()
    from . import likelihood
    def forbidden(*a,**k):raise AssertionError('production AC or source fit invoked')
    monkeypatch.setattr(likelihood,'ac_predict',forbidden)
    monkeypatch.setattr(likelihood,'fit_candidate',forbidden)
    result=review.review_trial(trial,freeze=freeze,snapshot=snapshot)
    assert result['success'],result['checks']
    assert result['counts']['price_pairs_verified']==2
    assert result['counts']['price_replay_AC_batches']==20
    price=next(c['result'] for c in result['checks'] if c['name']=='price/constant')
    assert price['coordinates']==[[0,0],[5,2]] and price['no_new_source_fit']


@pytest.mark.parametrize('damage',['pair_gradient','mean_gradient','source_witness'])
def test_price_witness_tampering_is_rejected(damage):
    trial,freeze,snapshot=price_fixture();price=trial['price_diagnostics']['constant']['complete_AC']
    if damage=='pair_gradient':price['pairs'][0]['gradient_per_kw'][0][0]+=.1
    elif damage=='mean_gradient':price['gradient_per_kw'][5][2]+=.1
    else:price['pairs'][0]['source_p'][0]=.6
    result=review.review_trial(trial,freeze=freeze,snapshot=snapshot)
    assert not result['success'] and _failed(result,'price/constant')


def test_legacy_price_without_witness_is_explicitly_unavailable():
    trial,freeze,snapshot=price_fixture();price=trial['price_diagnostics']['constant']['complete_AC']
    del price['power_kw']
    for row in price['pairs']:
        for key in ('a','b','weight','source_p','source_q','source_solver'):del row[key]
    result=review.review_trial(trial,freeze=freeze,snapshot=snapshot)
    assert not result['success'] and result['counts']['price_witnesses_unavailable']==2
    assert any(c['status']=='unavailable' and 'legacy_witness' in c['name'] for c in result['checks'])


def test_complete_evidence_replayed_without_production_AC_or_fitter(monkeypatch):
    trial,freeze,snapshot=fixture()
    from . import likelihood
    def forbidden(*args,**kwargs):raise AssertionError('production AC/fitter called')
    monkeypatch.setattr(likelihood,'ac_predict',forbidden)
    monkeypatch.setattr(likelihood,'fit_candidate',forbidden)
    result=review.review_trial(trial,freeze=freeze,snapshot=snapshot)
    assert result['success'],result['checks']
    assert result['counts']['fit_records_seen']==16
    assert result['counts']['fit_states_verified']==16
    assert result['no_new_fit'] and result['no_selector'] and result['no_production_AC_predictor']
    assert result['calibrated_rejection_statistics_computed'] is False


@pytest.mark.parametrize('damage',['EV_twice','theta','SSE','M','forecast','snapshot'])
def test_independent_tamper_checks(damage):
    trial,freeze,snapshot=fixture()
    if damage=='EV_twice':trial['methods'][1]['physical']['p'][0][0]+=trial['methods'][1]['power_kw'][0][0]/20.
    elif damage=='theta':trial['methods'][1]['noisy']['fits'][0]['r'][0]+=1e-15
    elif damage=='SSE':trial['methods'][1]['noisy']['fits'][0]['loss']+=1.
    elif damage=='M':trial['methods'][1]['M']+=1.
    elif damage=='forecast':trial['forecast']['p'][0][0]+=.01
    else:snapshot['source_base64']['ev_dynamic_loads.py']=base64.b64encode(b'tampered').decode()
    result=review.review_trial(trial,freeze=freeze,snapshot=snapshot)
    assert not result['success'] and result['check_counts'].get('failed',0)>0


def test_future_mutation_does_not_change_training_forecast_reconstruction():
    trial,freeze,snapshot=fixture()
    trial['background']['p_kw'][-1][0]+=4.
    result=review.review_trial(trial,freeze=freeze,snapshot=snapshot)
    assert _failed(result,'ordinary_process')
    forecast=next(c for c in result['checks'] if c['name']=='forecast/training_only_reconstruction')
    assert forecast['status']=='passed'


def test_missing_and_failed_records_are_retained_in_denominators():
    trial,freeze,snapshot=fixture()
    trial['methods'][1]['noisy']['fits'].pop()
    result=review.review_trial(trial,freeze=freeze,snapshot=snapshot)
    assert not result['success']
    assert result['counts']['missing_fit_records']==1
    assert result['check_counts']['unavailable']>=1
    assert _failed(result,'coverage')
    incomplete=deepcopy(trial);incomplete['methods'][2]=dict(name='highlow',status='physical_or_fit_failure',power_kw=trial['methods'][2]['power_kw'],failure='retained AC failure')
    result=review.review_trial(incomplete,freeze=freeze,snapshot=snapshot)
    assert any(f.get('method')=='highlow' for f in result['experiment_failures'])
    assert not result['success']


def test_energy_throughput_contrasts_and_no_EV_exception():
    trial,freeze,snapshot=fixture();a,b=trial['methods'][1:]
    trial['contrasts']['timing']=dict(left=a['name'],right=b['name'],delta_M=a['M']-b['M'],valid=True)
    result=review.review_trial(trial,freeze=freeze,snapshot=snapshot)
    assert result['success'],result['checks']
    contrast=next(c['result'] for c in result['checks'] if c['name']=='contrast/timing')
    assert contrast['matched_grid_energy_kwh'] and contrast['matched_throughput_kwh']
    trial['methods'][0]['operation']['battery_service']=True
    assert _failed(review.review_trial(trial,freeze=freeze,snapshot=snapshot),'operation/none')


def test_unassociated_planning_profiles_and_empty_directory_are_unavailable(tmp_path):
    trial,freeze,snapshot=fixture()
    trial['selection']['all_prediction_records']=[dict(pairs=[dict(name='a__to__b')])]
    result=review.review_trial(trial,freeze=freeze,snapshot=snapshot)
    assert result['counts']['planning_profiles_unavailable']==1 and not result['success']
    empty=review.review_directory(tmp_path)
    assert not empty['success'] and empty['trial_count']==0 and 'pending' in empty


def test_associated_planning_source_states_have_independent_full_AC_costs():
    trial,freeze,snapshot=fixture()
    trained={r['name']:r for r in trial['training_result']['fits']}
    meter=review._meter(trial['future_meter']);fore=trial['forecast']
    bank={r['name']:review._candidate(r['name'],r['supports'],3) for r in trial['candidates']}
    power=np.zeros((6,3));power[:,0]=2.;p=np.asarray(fore['p'])+power/20.;q=np.asarray(fore['q'])
    entries=[]
    for a,b in (('a','b'),('b','a')):
        sa,sb=fore['source_by_candidate'][a],fore['source_by_candidate'][b]
        ref=review.forward_physics(bank[a],trained[a]['r'],trained[a]['x'],p,q,sa['source_p'],sa['source_q'],fore['root'])
        obs=review._observation(ref,meter)
        comp=review.forward_physics(bank[b],trained[b]['r'],trained[b]['x'],obs.p-np.asarray(meter.bias_p),obs.q-np.asarray(meter.bias_q),sb['source_p'],sb['source_q'],fore['root'])
        cost=0.;normalizer=0.
        for k in ('v','master_p','master_q'):
            mean=np.asarray(comp[k])+(np.asarray(meter.bias_v) if k=='v' else 0.)
            y=np.asarray(getattr(obs,k));sigma=np.broadcast_to(getattr(meter,k+'_sigma'),y.shape)
            cost+=float(np.sum(((mean-y)/sigma)**2));normalizer+=float(np.log(2*np.pi*sigma**2).sum())
        entries.append(dict(name=a+'__to__'+b,a=a,b=b,success=True,source_p=sb['source_p'],source_q=sb['source_q'],
            solver=dict(r=trained[b]['r'],x=trained[b]['x'],loss=cost,loglik=-.5*(cost+normalizer)),
            joint=cost,added=cost/2,source_interface=1,line_parameters_fixed=True,EV_in_total_PQ_once=True))
    prediction=dict(pairs=entries)
    trial['selection']['predicted_AC']['constant']=prediction
    trial['selection']['all_prediction_records']=[prediction]
    result=review.review_trial(trial,freeze=freeze,snapshot=snapshot)
    assert result['success'],result['checks']
    assert result['counts']['planning_profiles_verified']==2
    assert result['counts']['planning_profiles_unavailable']==0


def test_directory_review_preserves_raw_bytes_and_records_input_hash(tmp_path):
    trial,freeze,snapshot=fixture()
    for name,value in (('freeze.json',freeze),('source_snapshot.json.gz',snapshot),('trial_991111.json.gz',trial)):
        data=json.dumps(value,allow_nan=False).encode()
        (tmp_path/name).write_bytes(gzip.compress(data,mtime=0) if name.endswith('.gz') else data)
    evidence={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    result=review.review_directory(tmp_path,tmp_path/'audit.json')
    assert result['success'] and result['verified_trial_count']==1
    assert result['trials'][0]['input_sha256']==evidence['trial_991111.json.gz']
    for name,digest in evidence.items():assert hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()==digest
    with pytest.raises(ValueError,match='overwrite'):
        review.review_directory(tmp_path,tmp_path/'freeze.json')
