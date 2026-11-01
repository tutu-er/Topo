"""Pure request gates and zero-response DTO tests; no AC or new future."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

candidate=Path(__file__).with_name('one_day_worker.py')
if candidate.exists():
    spec=importlib.util.spec_from_file_location('_scoped_one_day_worker',candidate)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
else:
    from research_experiments.wzzt_dr_selection.dr_frozen_distance import one_day_worker as worker


def _request_fixture(tmp_path,*,case='paper15',noise=.0002,nodes=(103,101),explicit=True):
    repo=tmp_path/'repo';source=tmp_path/'producer';repo.mkdir();source.mkdir()
    sources={str(Path(worker.__file__).resolve()):worker.sha(worker.__file__)}
    for relative in worker.HELPER_FILES:
        p=repo/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# unchanged helper\n')
        sources[str(p.resolve())]=worker.sha(p)
    for name in worker.PRODUCER_MODULES:
        p=source.joinpath(*name.split('.')).with_suffix('.py');p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# producer not loaded\n')
        sources[str(p.resolve())]=worker.sha(p)
    def file(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value),encoding='utf-8')
        return dict(path=str(p.resolve()),sha256=worker.sha(p))
    plan=file('plan.json',{});pool=file('pool.json',{});metadata=file('metadata.json',{})
    labels=list(range(101,116)) if case=='paper15' else list(range(301,319)) if case=='pengwah18' else list(range(201,217))
    protocol=dict(status='frozen',voltage_sigma=noise,pq_sigma=.005,source_sha256=sources,
        frozen_pool=pool,plans=dict(zero=plan))
    if explicit:protocol.update(allowed_voltage_sigma=[noise],expected_resource_nodes=list(nodes))
    proto=file('protocol.json',protocol)
    request=dict(status='frozen',phase='test',case=case,record_id='gated',policy='zero',path=1,
        labels=labels,root_label=1,base_kw=200.,replicate=400000,day_offset=6,profile_index=0,
        factors=[.1,-.1],voltage_sigma=noise,pq_sigma=.005,budget=1.6,protocol=proto,plan=plan,
        frozen_pool=pool,repo_root=str(repo.resolve()),source_root=str(source.resolve()),
        source_sha256=sources,source_metadata=metadata)
    req=file('request.json',request)
    return Path(req['path']),req['sha256'],request,protocol


@pytest.mark.parametrize('case,noise,nodes',[
    ('paper15',.0005,(103,101)),('pengwah18',.0002,(307,312)),('flynn16',.0002,(202,213)),
])
def test_explicit_multicase_scope_is_hash_bound_without_loading_producer(tmp_path,case,noise,nodes):
    path,digest,_,_=_request_fixture(tmp_path,case=case,noise=noise,nodes=nodes)
    before=set(sys.modules)
    loaded=worker.load_request(path,digest)
    assert loaded['expected_resource_nodes']==list(nodes)
    assert loaded['voltage_sigma']==noise
    assert 'rnj_wzzt.scenario.simulation' not in set(sys.modules)-before


def test_legacy_default_remains_flynn_original_noise_nodes(tmp_path):
    path,digest,_,_=_request_fixture(tmp_path,case='flynn16',noise=.0005,nodes=(201,213),explicit=False)
    assert worker.load_request(path,digest)['expected_resource_nodes']==[201,213]


def test_undeclared_noise_change_is_rejected(tmp_path):
    path,digest,request,_=_request_fixture(tmp_path,case='flynn16',noise=.0005,nodes=(201,213),explicit=False)
    request['voltage_sigma']=.0002;path.write_text(json.dumps(request),encoding='utf-8')
    with pytest.raises(ValueError,match='declared original voltage'):worker.load_request(path,worker.sha(path))


def test_scope_drift_cannot_be_overridden_by_request(tmp_path):
    path,digest,request,_=_request_fixture(tmp_path)
    request['expected_resource_nodes']=[103,105];path.write_text(json.dumps(request),encoding='utf-8')
    with pytest.raises(ValueError,match='hash-bound protocol'):worker.load_request(path,worker.sha(path))


def test_changed_protocol_without_matching_request_ref_is_rejected(tmp_path):
    path,digest,request,protocol=_request_fixture(tmp_path)
    protocol['expected_resource_nodes']=[103,105]
    Path(request['protocol']['path']).write_text(json.dumps(protocol),encoding='utf-8')
    with pytest.raises(RuntimeError,match='protocol hash mismatch'):worker.load_request(path,digest)


@pytest.mark.parametrize('field,value',[
    ('allowed_voltage_sigma',[.001]),('allowed_voltage_sigma',[.0002,.0002]),
    ('expected_resource_nodes',[103,103]),('expected_resource_nodes',[999]),
])
def test_invalid_declared_scope_is_rejected(field,value):
    request=dict(labels=list(range(101,116)),voltage_sigma=.0002,pq_sigma=.005)
    protocol=dict(allowed_voltage_sigma=[.0002],expected_resource_nodes=[103,101]);protocol[field]=value
    with pytest.raises(ValueError):worker.contract_scope(request,protocol)


def test_zero_response_resource_audit_uses_declared_nodes_and_inventory(tmp_path):
    # One module per resource suffices for this generic gate. No producer read.
    shape=[0.]*96;shape[24]=-1.;shape[28]=1.
    resources=[dict(resource_id=identity,node_label=node,baseline_kw=.8,power_upper_kw=1.6,
        prefix_delta_lower_kwh=-1.6,prefix_delta_upper_kwh=1.6,available=True,
        reactive_ratio=.35,incentive_per_kwh=0.,electricity_price_per_kwh=0.)
        for identity,node in [('focus',307),('partner',312)]]
    modules=[dict(module_id=identity,profiles={identity:shape},amplitude_upper_kw=.8,description='zero')
        for identity in ('focus','partner')]
    plan=dict(resources=resources,modules=modules,amplitudes=[0.,0.],prices=[0.,0.],
        elasticities=[.4,.4],response_factors=[[.15,.05],[.15,-.05]])
    planfile=tmp_path/'plan.json';planfile.write_text(json.dumps(plan),encoding='utf-8')
    request=dict(repo_root=str(Path(__file__).resolve().parents[0]),plan_path=str(planfile),
        expected_resource_nodes=[307,312],labels=list(range(301,319)),factors=[.1,-.1],incentive_budget=1.6)
    # Existing resource module is imported from pytest's explicitly set repo path.
    _,detail,fees=worker.audit_plan(request)
    assert [r.node_label for r in detail['resources']]==[307,312]
    assert fees['actual_payment']==0
    request['expected_resource_nodes']=[307,313]
    with pytest.raises(ValueError,match='declared scope'):worker.audit_plan(request)
