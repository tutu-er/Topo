"""Tamper checks for the read-only reviewer, using tiny attainable AC states."""
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pytest

from . import ev_mobile_v2_review as review
from . import ev_systematic_refit as refit
from . import ev_fixed_train_design as fixed
from . import ev_mobile_cases_v2 as cases
from . import likelihood as lk
from .information_context import meter_to_record
from .simulation import model


def _profile_fixture():
    """Two models, four attained states; no parameter optimization is run."""
    H,T,n=3,2,3;bank={c.name:c for c in cases.candidate_bank('remote8')[:2]}
    names=list(bank);cfg=cases.case_config('remote8');cfg['horizon']=T
    tm=fm=model(n);p=np.array([[.12,.14,.11],[.13,.13,.12],[.11,.15,.13]])
    q=.3*p;sp=np.full(H,.08);sq=np.full(H,.025)
    trained={name:dict(r=np.full(c.m,.003),x=np.full(c.m,.0008),source_p=sp.copy(),source_q=sq.copy())
             for name,c in bank.items()}
    a=names[0];physical=lk.ac_predict(bank[a],trained[a]['r'],trained[a]['x'],p,q,sp,sq,np.ones(H))
    training=lk.Observations(p=p+tm.bias_p,q=q+tm.bias_q,
        v=physical['v']+tm.bias_v+np.linspace(-1e-5,1e-5,H*n).reshape(H,n),
        master_p=physical['master_p']+2e-5,master_q=physical['master_q']-1e-5,root_v=np.ones(H))
    forecast=dict(p=np.tile(p.mean(axis=0),(T,1)),q=np.tile(q.mean(axis=0),(T,1)),root=np.ones(T),
        source_by_candidate={name:dict(source_p=np.full(T,.08),source_q=np.full(T,.025)) for name in names})
    record=dict(config=cfg,training={key:np.asarray(getattr(training,key)).tolist()
        for key in ('p','q','v','master_p','master_q','root_v')},training_meter=meter_to_record(tm),
        future_meter=meter_to_record(fm),forecast=forecast)
    options={**cfg['training'],**cfg['source'],'ratio_bounds':cfg['ratio_bounds']}
    # Refit adapter has a deliberately narrower supported-options signature.
    options=refit._options(options)
    u=np.zeros((T,n));u[:,0]=[2.,-1.];references=[];prices={};losses={}
    total=dict(p=forecast['p']+u/20.,q=forecast['q']);columns=np.array([t*(2*n+2)+j for t in range(T) for j in range(n)])
    for a in names:
        source=forecast['source_by_candidate'][a]
        ga=fixed._geometry(bank[a],trained[a],total,forecast['root'],fm,
            fixed._source_vector(source['source_p'],source['source_q']))
        future=lk.Observations(p=total['p']+fm.bias_p,q=total['q']+fm.bias_q,
            v=ga['physical']['v']+fm.bias_v,master_p=ga['physical']['master_p'],
            master_q=ga['physical']['master_q'],root_v=np.ones(T))
        obs=lk.concat_observations(training,future);meter=refit.concat_models((training,future),(tm,fm));fits={}
        for b in names:
            state=dict(r=trained[b]['r'],x=trained[b]['x'],source_p=np.full(H+T,.08),source_q=np.full(H+T,.025))
            score=refit._score(bank[b],obs,meter,state,options,(H,T),('training','future'))
            gb=fixed._geometry(bank[b],state,total,forecast['root'],fm,
                fixed._source_vector(state['source_p'][-T:],state['source_q'][-T:]))
            residual=np.asarray(score['blocks']['future']['residual'])
            r=np.r_[residual[:,:n].ravel(),residual[:,n],residual[:,n+1]]
            price=(((gb['state_jacobian']-ga['state_jacobian'])[:,columns]/20.).T@r).reshape(T,n)
            fits[b]=dict(score,state=state,success=True,gradient_reliable=True,gradient_per_kw=price,
                future_reference_derivative_subtracted=True)
            prices[a,b]=price;losses[a,b]=score['loss']
        references.append(dict(a=a,fits=fits,success=True))
    pairs=[dict(a=a,b=b,success=True,half_gap=.5*(losses[a,b]-losses[a,a]),
        own_loss=losses[a,a],rival_loss=losses[a,b],gradient_per_kw=prices[a,b]-prices[a,a])
        for a in names for b in names if a!=b]
    groups=[[p for p in pairs if p['a']==a] for a in names]
    score=float(np.mean([group[0]['half_gap'] for group in groups]))
    aggregate=dict(success=True,score=score,active_groups=groups,
        gradient_per_kw=np.mean([group[0]['gradient_per_kw'] for group in groups],axis=0))
    profile=dict(power_kw=u,references=references,pairs=pairs,aggregation=aggregate,score=score,success=True,
        source_interface=1,uses_training_history=True,line_parameters_fixed_for_references=True,
        one_trained_parameter_vector=True,rival_line_parameters_shared_history_and_prediction=True,
        uses_truth=False,uses_validation=False,uses_future_noise=False)
    return record,profile,bank,trained


def test_tiny_joint_states_and_independent_physical_envelope_prices_replay():
    record,profile,bank,trained=_profile_fixture();metrics=defaultdict(float)
    review._joint_profile(record,profile,bank,trained,metrics)
    assert metrics['joint_profile_checks']==1 and metrics['joint_predictive_fit_checks']==4
    assert metrics['max_envelope_price_scaled_error']<2e-5


@pytest.mark.parametrize('tamper', ['separate_lines','future_twice','history_loss','price','score','truth_flag','active_price'])
def test_joint_review_rejects_changed_lines_repeated_data_scores_prices_and_hidden_inputs(tamper):
    record,profile,bank,trained=_profile_fixture();row=profile['references'][0]['fits']['clade_01']
    if tamper=='separate_lines':row['state']['r']=np.r_[row['state']['r'],row['state']['r']]
    elif tamper=='future_twice':row['state']['source_p']=np.r_[row['state']['source_p'],[.08,.08]]
    elif tamper=='history_loss':row['blocks']['training']['loss']+=1.
    elif tamper=='price':row['gradient_per_kw']=row['gradient_per_kw']+1.
    elif tamper=='score':profile['score']+=1.
    elif tamper=='active_price':
        profile['aggregation']=deepcopy(profile['aggregation'])
        profile['aggregation']['active_groups'][0][0]['gradient_per_kw']+=1.
    else:profile['uses_truth']=True
    with pytest.raises(ValueError):review._joint_profile(record,profile,bank,trained,defaultdict(float))


def _cache_fixture():
    profile=dict(power_kw=[[1.,0.,0.]],seconds=.12,joint_fit_calls=16,
        fixed_source_witness_fit_calls=16,success=True,deadline_reached=False)
    metadata={name:dict(raw={},planning_actual_seconds=0.) for name in review.METHODS}
    for name,hits,new in (('jointgrad_fixed',0,1),('jointgrad_mobile',1,0)):
        metadata[name]=dict(planning_actual_seconds=.15 if new else .01,raw=dict(
            initial_candidates=[dict(profile=deepcopy(profile))],starts=[],counters=dict(
                profile_evaluations=new,cache_hits=hits,actual_profile_seconds=.12*new,
                joint_fit_calls=16*new,fixed_source_witness_fit_calls=16*new,
                payment_log=[dict(seconds=.12,joint_fit_calls=16,success=True)] if new else [])))
    return metadata


def test_shared_joint_cache_pays_first_profile_once_and_only_new_mobile_cost():
    metrics=defaultdict(float);review._cache_costs(_cache_fixture(),metrics)
    assert metrics['incremental_cache_cost_checks']==2


@pytest.mark.parametrize('tamper', ['charge_cache_twice','omit_witness_calls','changed_cached_value'])
def test_shared_joint_cache_tampering_is_rejected(tamper):
    metadata=_cache_fixture();mobile=metadata['jointgrad_mobile']['raw'];fixed_raw=metadata['jointgrad_fixed']['raw']
    if tamper=='charge_cache_twice':mobile['counters']['actual_profile_seconds']=.12
    elif tamper=='omit_witness_calls':fixed_raw['counters']['fixed_source_witness_fit_calls']=0
    else:mobile['initial_candidates'][0]['profile']['seconds']=.13
    with pytest.raises(ValueError):review._cache_costs(metadata,defaultdict(float))


def _operation_fixture():
    domain=cases.service_domain('remote8',mobile=False);ix=domain.indices;T=8;p=1.5
    x=np.zeros(domain.model.n_variables);x[ix.u[:,0]]=p;x[ix.charge[:,0]]=p;x[ix.mode]=1
    x[ix.energy]=24.+np.arange(T+1)*.25*.95*p
    for t in range(T):
        j=next(j for j,a in enumerate(ix.arcs) if a.kind=='connected' and a.origin==0 and a.departure==t)
        x[ix.arc[j]]=1
    audit=domain.check(x);assert audit['feasible']
    return domain,deepcopy(audit)


def test_operation_route_replays_without_any_optimization():
    domain,audit=_operation_fixture();replayed=review._operation(domain,audit)
    assert replayed['feasible'] and replayed['driver_cost']==0.


@pytest.mark.parametrize('tamper', ['battery','extra_arc','fake_drive'])
def test_operation_energy_or_route_tampering_is_rejected(tamper):
    domain,audit=_operation_fixture()
    if tamper=='battery':audit['energy_kwh'][-1]+=.2
    elif tamper=='extra_arc':audit['route'].append(deepcopy(audit['route'][-1]))
    else:audit['route'][0]['driving_energy_kwh']=.2
    with pytest.raises(ValueError):review._operation(domain,audit)
