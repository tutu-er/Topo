"""Pure contract/source-gate tests; no physical generator, AC or fit is run."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


worker_file=Path(__file__).with_name('one_day_worker.py')
spec=importlib.util.spec_from_file_location('frozen_one_day_worker_under_test',worker_file)
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf-8')


def ref(path):return dict(path=str(path),sha256=worker.sha(path))


@pytest.fixture
def contract(tmp_path):
    from research_experiments.wzzt_dr_selection.dr_day_modules.day_schedule import shift_shape
    resources=[dict(resource_id=key,node_label=node,baseline_kw=1.,power_upper_kw=2.,
        prefix_delta_lower_kwh=-2.,prefix_delta_upper_kwh=2.,available=True,reactive_ratio=.35)
        for key,node in [('anchor',201),('partner',213)]]
    early=shift_shape(range(24,28),range(28,32));late=shift_shape(range(76,80),range(80,84))
    contrast=shift_shape(range(32,40),range(40,48))
    shapes=[early,late,early+late,-late,contrast]
    modules=[dict(module_id=key+'_'+str(j),profiles={key:shape.tolist()},amplitude_upper_kw=1.)
             for key in ['anchor','partner'] for j,shape in enumerate(shapes)]
    prices=np.zeros(10);prices[[1,6]]=.6;elastic=np.full(10,.5)
    plan=dict(resources=resources,modules=modules,prices=prices.tolist(),elasticities=elastic.tolist(),
              amplitudes=(elastic*prices).tolist(),response_factors=[[.15,.05]]*5+[[.15,-.05]]*5)
    path=tmp_path/'plan.json';save(path,plan)
    request=dict(repo_root=str(Path.cwd()),plan_path=str(path),factors=[.2,-.3],labels=list(range(201,217)),incentive_budget=1.6)
    return request,plan,path


def test_continuous_generic_price_response_and_corner_budget(contract):
    request,plan,path=contract
    _,detail,pay=worker.audit_plan(request)
    assert np.allclose(detail['actual_amplitudes'],np.array(plan['amplitudes'])*(1+np.array(plan['response_factors'])@np.array(request['factors'])))
    assert detail['audit']['resource_feasible'] and len(detail['corners'])==4
    assert pay['expected_payment']==pytest.approx(.36)
    assert pay['worst_payment']==pytest.approx(.432)
    assert pay['actual_payment']==pytest.approx(.3708)
    assert pay['exact_box_worst_payment']<=pay['worst_payment']
    assert pay['budget_pass'] and pay['actual_payment']<=1.6


def test_configurable_elasticity_not_old_hardcoded_point_one(contract):
    request,plan,path=contract
    plan['elasticities']=[.4]*5+[.6]*5
    plan['amplitudes']=(np.array(plan['prices'])*plan['elasticities']).tolist();save(path,plan)
    _,detail,pay=worker.audit_plan(request)
    assert detail['expected_amplitudes'][1]==pytest.approx(.24)
    assert detail['expected_amplitudes'][6]==pytest.approx(.36)
    assert pay['actual_payment']==pytest.approx(.37188)


def test_inconsistent_mean_price_rejected(contract):
    request,plan,path=contract;plan['amplitudes'][1]+=.01;save(path,plan)
    with pytest.raises(ValueError,match='elasticity times price'):worker.audit_plan(request)


def test_two_positive_types_rejected_even_if_tiny(contract):
    request,plan,path=contract;plan['prices'][0]=1e-12;plan['amplitudes'][0]=.5e-12;save(path,plan)
    with pytest.raises(ValueError,match='exactly zero'):worker.audit_plan(request)


def test_support_corner_violation_is_not_clipped(contract):
    request,plan,path=contract
    request['factors']=[0.,0.]
    plan['prices']=[0.]*10;plan['prices'][1]=1.
    plan['amplitudes']=(np.array(plan['prices'])*.5).tolist()
    plan['modules'][1]['amplitude_upper_kw']=.5
    save(path,plan)
    with pytest.raises(ValueError,match='support corner'):worker.audit_plan(request)


def test_hard_budget_not_just_realized_payment(contract):
    request,plan,path=contract
    plan['prices'][1]=plan['prices'][6]=1.4
    plan['amplitudes']=(np.array(plan['prices'])*.5).tolist();save(path,plan)
    with pytest.raises(ValueError,match='hard payment'):worker.audit_plan(request)


def test_contained_component_checks_both_channels(contract):
    request,_,_=contract
    _,detail,_=worker.audit_plan(request)
    pp=np.full((96,16),2./200);qq=np.full((96,16),.7/200)
    r=worker.contained_component_check(pp,qq,detail['resources'],request['labels'],200.)
    assert r['minimum_nonflexible_p_kw']==pytest.approx(1.)
    assert r['minimum_nonflexible_q_kvar']==pytest.approx(.35)
    qq[20,request['labels'].index(213)]=.34/200
    with pytest.raises(ValueError,match='cannot contain') as exc:
        worker.contained_component_check(pp,qq,detail['resources'],request['labels'],200.)
    assert exc.value.component_report['minimum_nonflexible_q_kvar']==pytest.approx(-.01)


@pytest.fixture
def gate_request(tmp_path):
    repo=tmp_path/'repo';source=tmp_path/'source'
    sources={str(worker_file.resolve()):worker.sha(worker_file)}
    for relative in worker.HELPER_FILES:
        p=repo/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fake dependency\n',encoding='utf-8');sources[str(p.resolve())]=worker.sha(p)
    for name in worker.PRODUCER_MODULES:
        p=source.joinpath(*name.split('.')).with_suffix('.py');p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fake producer never imported\n',encoding='utf-8');sources[str(p.resolve())]=worker.sha(p)
    plan=tmp_path/'plan.json';plan.write_text('intentionally not parsed at source gate',encoding='utf-8')
    pool=tmp_path/'frozen_pool.json';pool.write_text('invalidJSON proves pool not parsed',encoding='utf-8')
    metadata=tmp_path/'metadata.json';save(metadata,{})
    protocol=tmp_path/'protocol.json';save(protocol,dict(status='frozen',source_sha256=sources,
         plans={'zero':ref(plan)},frozen_pool=ref(pool),voltage_sigma=.0005,pq_sigma=.005))
    request=dict(status='frozen',phase='test',case='flynn16',record_id='zero_path1401',policy='zero',
        path=1401,replicate=524010,profile_index=1,labels=list(range(201,217)),root_label=1,base_kw=200.,
        factors=[.1,.2],voltage_sigma=.0005,pq_sigma=.005,budget=1.6,
        protocol=ref(protocol),plan=ref(plan),frozen_pool=ref(pool),source_sha256=sources,
        source_metadata=ref(metadata),repo_root=str(repo),source_root=str(source))
    request_path=tmp_path/'request.json';save(request_path,request)
    return request_path,request,protocol,repo


def test_hash_gate_does_not_parse_frozen_pool_or_plan(gate_request):
    path,request,_,_=gate_request
    loaded=worker.load_request(path,worker.sha(path))
    assert loaded['profile_index']==1
    assert loaded['frozen_pool_path']==request['frozen_pool']['path']
    assert loaded['plan_path']==request['plan']['path']


def test_source_tamper_rejected_before_plan_parse(gate_request):
    path,_,_,repo=gate_request
    (repo/worker.HELPER_FILES[0]).write_text('# changed',encoding='utf-8')
    with pytest.raises(RuntimeError,match='source hash mismatch'):worker.load_request(path,worker.sha(path))


def test_missing_worker_dependency_rejected(gate_request):
    path,request,protocol,_=gate_request
    request['source_sha256'].pop(str(worker_file.resolve()))
    parent=json.loads(protocol.read_text(encoding='utf-8'));parent['source_sha256']=request['source_sha256'];save(protocol,parent)
    request['protocol']=ref(protocol);save(path,request)
    with pytest.raises(ValueError,match='omits worker'):worker.load_request(path,worker.sha(path))


def test_request_hash_must_be_prelocked(gate_request):
    path,_,_,_=gate_request
    with pytest.raises(RuntimeError,match='request differs'):worker.load_request(path,'0'*64)
