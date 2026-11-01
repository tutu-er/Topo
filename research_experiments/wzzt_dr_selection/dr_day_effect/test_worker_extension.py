"""Pure uncapped cost/source gates and mocked zero-action execution tests.

The complete-run test uses synthetic in-memory arrays and mocked AC/producer
functions; no real future, graph, AC solve or parameter fitting is performed.
"""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

default_worker=Path(__file__).resolve().parents[1]/'dr_frozen_distance/one_day_worker.py'
worker_file=Path(os.environ.get('WZZT_WORKER_UNDER_TEST',str(default_worker)))
spec=importlib.util.spec_from_file_location('day_effect_worker_under_test',worker_file)
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


def save(path,value):
    Path(path).write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf-8')


def ref(path):return dict(path=str(Path(path).resolve()),sha256=worker.sha(path))


@pytest.fixture
def contract(tmp_path):
    h=np.zeros(96);h[::2]=-1.;h[1::2]=1.
    resources=[dict(resource_id=k,node_label=n,baseline_kw=1.,power_upper_kw=2.,
        prefix_delta_lower_kwh=-2.,prefix_delta_upper_kwh=2.,available=True,reactive_ratio=.35)
        for k,n in [('first',201),('second',213)]]
    plan=dict(resources=resources,
        modules=[dict(module_id=k,profiles={k:h.tolist()},amplitude_upper_kw=1.) for k in ['first','second']],
        amplitudes=[.4,.4],prices=[.8,.8],elasticities=[.5,.5],response_factors=[[.15,.05],[.15,-.05]],
        budget=None,cap_enforced=False,expected_payment=7.68,hard_payment_bound=9.216)
    path=tmp_path/'plan.json';save(path,plan)
    request=dict(repo_root=str(Path.cwd()),plan_path=str(path),factors=[.2,-.3],labels=list(range(201,217)),
        expected_resource_nodes=[201,213],payment_mode='frozen_plan_bound',optimizer_payment_cap=None,
        payment_bound=9.216)
    return request,plan,path


def test_uncapped_plan_above_one_point_six_has_derived_execution_bound(contract):
    request,_,_=contract
    _,detail,fees=worker.audit_plan(request)
    assert fees['expected_payment']==pytest.approx(7.68)
    assert fees['worst_payment']==pytest.approx(9.216)
    assert fees['actual_payment']<=fees['payment_bound']
    assert fees['payment_bound_pass'] and fees['budget_pass'] is None
    assert fees['optimizer_payment_cap'] is None and fees['payment_cap_enforced'] is False
    assert len(detail['corners'])==4 and detail['audit']['resource_feasible']


def test_identical_plan_keeps_legacy_budget_rejection(contract):
    request,_,_=contract
    request.pop('payment_mode');request['incentive_budget']=1.6
    with pytest.raises(ValueError,match='exceeds1.6'):worker.audit_plan(request)


@pytest.mark.parametrize('bound',[9.215,9.217])
def test_requested_bound_must_match_independent_frozen_plan_replay(contract,bound):
    request,_,_=contract;request['payment_bound']=bound
    with pytest.raises(ValueError,match='payment_bound differs'):worker.audit_plan(request)


@pytest.mark.parametrize('field',['hard_payment_bound','expected_payment'])
def test_declared_cost_tamper_rejected_even_before_execution(contract,field):
    request,plan,path=contract;plan[field]+=.1;save(path,plan)
    with pytest.raises(ValueError,match=field+' differs'):worker.audit_plan(request)


@pytest.mark.parametrize('field,value',[('budget',1.6),('cap_enforced',True)])
def test_uncapped_dto_must_explicitly_remove_cap(contract,field,value):
    request,plan,path=contract;plan[field]=value;save(path,plan)
    with pytest.raises(ValueError,match='budget null'):worker.audit_plan(request)


@pytest.fixture
def gate(tmp_path):
    repo=tmp_path/'repo';source=tmp_path/'producer'
    sources={str(worker_file.resolve()):worker.sha(worker_file)}
    for relative in worker.HELPER_FILES:
        path=repo/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# never imported\n')
        sources[str(path.resolve())]=worker.sha(path)
    for name in worker.PRODUCER_MODULES:
        path=source.joinpath(*name.split('.')).with_suffix('.py');path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('# producer never imported\n');sources[str(path.resolve())]=worker.sha(path)
    plan=tmp_path/'gate_plan.json';plan.write_text('not JSON: source gate must not parse it')
    pool=tmp_path/'pool.json';pool.write_text('not JSON: worker only checks hash')
    metadata=tmp_path/'metadata.json';save(metadata,{})
    protocol=dict(status='frozen',payment_mode='frozen_plan_bound',optimizer_payment_cap=None,
        payment_bound=9.216,source_sha256=sources,plans={'candidate':ref(plan)},frozen_pool=ref(pool),
        voltage_sigma=.0005,pq_sigma=.005)
    proto=tmp_path/'protocol.json';save(proto,protocol)
    request=dict(status='frozen',phase='test',case='flynn16',record_id='pure_gate',policy='candidate',
        path=1,replicate=400000,day_offset=6,profile_index=0,labels=list(range(201,217)),root_label=1,
        base_kw=200.,factors=[.1,-.1],voltage_sigma=.0005,pq_sigma=.005,
        payment_mode='frozen_plan_bound',optimizer_payment_cap=None,payment_bound=9.216,
        protocol=ref(proto),plan=ref(plan),frozen_pool=ref(pool),repo_root=str(repo.resolve()),
        source_root=str(source.resolve()),source_sha256=sources,source_metadata=ref(metadata))
    path=tmp_path/'request.json';save(path,request)
    return path,request,proto,protocol,repo


def test_uncapped_hash_gate_keeps_plan_and_pool_unparsed(gate):
    path,_,_,_,_=gate
    request=worker.load_request(path,worker.sha(path))
    assert request['payment_bound']==9.216 and request['incentive_budget'] is None


@pytest.mark.parametrize('field,value',[
    ('payment_mode','legacy_1p6'),('optimizer_payment_cap',1.6),('payment_bound',9.217)])
def test_uncapped_request_and_protocol_contract_must_agree(gate,field,value):
    path,request,proto,protocol,_=gate
    protocol[field]=value;save(proto,protocol);request['protocol']=ref(proto);save(path,request)
    with pytest.raises(ValueError):worker.load_request(path,worker.sha(path))


def test_uncapped_source_tamper_is_rejected_before_plan_parse(gate):
    path,_,_,_,repo=gate
    (repo/worker.HELPER_FILES[0]).write_text('# changed after freeze\n')
    with pytest.raises(RuntimeError,match='source hash mismatch'):worker.load_request(path,worker.sha(path))


def test_zero_requires_exact_zero_for_every_resource():
    zero=np.zeros(96)
    assert worker.zero_resource_execution(dict(resources={'a':dict(executed_delta_kw=zero.tolist())}))
    weak=zero.copy();weak[1]=1e-15
    assert not worker.zero_resource_execution(dict(resources={'a':dict(executed_delta_kw=weak.tolist())}))
    assert not worker.zero_resource_execution(dict(resources={
        'a':dict(executed_delta_kw=weak.tolist()),'b':dict(executed_delta_kw=(-weak).tolist())}))


def _mock_runtime(monkeypatch,tmp_path,request):
    calls=dict(context=0,metadata=0,base=0,ac=0,checker=0,observe=0)
    labels=request['labels'];index=pd.date_range('2000-01-07',periods=96,freq='15min')
    pp=np.full((96,len(labels)),.01);qq=np.full_like(pp,.0035)
    # Natural export and negative residual composition at the flexible nodes.
    pp[:,[labels.index(201),labels.index(213)]]=-.001
    qq[:,[labels.index(201),labels.index(213)]]=-.001
    base=SimpleNamespace(p_base=pd.DataFrame(pp,index=index,columns=labels),
        q_base=pd.DataFrame(qq,index=index,columns=labels),root_true=pd.Series(np.ones(96),index=index),shifted_seed=123)
    context=SimpleNamespace(net=SimpleNamespace(check_terminal_load_only=lambda **_:None))
    def prepare(*args,**kwargs):calls['context']+=1;return context
    def build(*args,**kwargs):calls['base']+=1;return base
    def solve(context,p,q,root):
        calls['ac']+=1
        return dict(converged=pd.Series(np.ones(96,bool))),pd.DataFrame(np.ones((96,len(labels))),index=index,columns=labels)
    def observe(context,base,p,q,v,**kwargs):
        calls['observe']+=1
        return dict(p_measured=p,q_measured=q,voltage_measured=v,root_observed=base.root_true)
    def check(*args,**kwargs):
        calls['checker']+=1
        return dict(electrical_equations_pass=True,voltage_limits_pass=True,voltage_violation_cells=0,
            max_nodal_p_residual_pu=0.,max_nodal_q_residual_pu=0.,max_distflow_identity_error_pu2=0.)
    source=tmp_path/'physical_checker.py';source.write_text('# mocked physical checker\n')
    source_root=tmp_path/'source'
    runtime={name:'0'*64 for name in worker.PRODUCER_MODULES}
    request.update(source_root=str(source_root),root_label=1,base_kw=200.,case='flynn16',record_id='mocked',
        policy='candidate',path=1,profile_index=0,replicate=400000,day_offset=6,
        protocol={'sha256':'1'*64},voltage_sigma=.0005,pq_sigma=.005)
    request['checked_source_sha256']={str(source_root.joinpath(*name.split('.')).with_suffix('.py').resolve()):digest
        for name,digest in runtime.items()}
    request['checked_source_sha256'][str(source.resolve())]=worker.sha(source)
    pool=tmp_path/'frozen_pool.json';pool.write_text('pool not parsed')
    request['frozen_pool_path']=str(pool)
    def metadata(*args):
        calls['metadata']+=1;return dict(root_label=1,base_kw=200.,base_kv=.4,base_mva=.2)
    sim=SimpleNamespace(_prepare_case_context=prepare,_build_base_trajectory=build,
        _solve_physical_ac=solve,_observe_terminal_channels=observe)
    def scene(p,q,v,root,labels,name,offset):
        return dict(P_terminal=pd.DataFrame(p,index=index,columns=labels),
            Q_terminal=pd.DataFrame(q,index=index,columns=labels),V_terminal=pd.DataFrame(v,index=index,columns=labels))
    legacy=SimpleNamespace(frozen_source_metadata=metadata,_simulation_runtime=lambda _: (sim,runtime),
        _physical_check_function=lambda _: (check,source),validate_context_metadata=lambda *args:None,
        scene=scene,evaluate_supports=lambda *args:dict(missed_blocks=[],full_effective_truth_blocks=0))
    monkeypatch.setattr(worker,'_legacy',lambda _:legacy)
    return calls


def test_zero_skips_only_component_qualification_and_still_runs_full_ac_observation(contract,tmp_path,monkeypatch):
    request,plan,path=contract
    plan.update(amplitudes=[0.,0.],prices=[0.,0.],expected_payment=0.,hard_payment_bound=0.)
    save(path,plan);request['payment_bound']=0.
    calls=_mock_runtime(monkeypatch,tmp_path,request)
    def forbidden(*args):raise AssertionError('zero must not use flexible-component qualification')
    monkeypatch.setattr(worker,'contained_component_check',forbidden)
    output=tmp_path/'zero_output';output.mkdir()
    receipt=worker.run(request,output,dict(request_sha256_expected='2'*64,private_files={}))
    assert receipt['status']=='complete' and receipt['observed_validation_rows']==96
    assert receipt['component_check_status']=='not_required_zero_increment'
    assert receipt['whole_target_terminal_nonnegative'] is False
    assert receipt['actual_payment']==0 and receipt['payment_bound']==0
    assert calls==dict(context=1,metadata=1,base=1,ac=1,checker=1,observe=1)
    assert (output/'packet.pkl').exists()


def test_weak_nonzero_cannot_skip_component_qualification(contract,tmp_path,monkeypatch):
    request,plan,path=contract
    plan.update(amplitudes=[1e-15,0.],prices=[2e-15,0.],expected_payment=2.4e-29,hard_payment_bound=2.88e-29)
    save(path,plan);request['payment_bound']=2.88e-29
    calls=_mock_runtime(monkeypatch,tmp_path,request);checked=[]
    def component(*args):checked.append(True);raise ValueError('retained nonzero qualification failure')
    monkeypatch.setattr(worker,'contained_component_check',component)
    output=tmp_path/'weak_output';output.mkdir()
    receipt=worker.run(request,output,dict(request_sha256_expected='2'*64,private_files={}))
    assert checked==[True] and receipt['status']=='physical_failed'
    assert calls['ac']==calls['observe']==0
    assert receipt['actual_payment']>0 and receipt['failure_retained']
