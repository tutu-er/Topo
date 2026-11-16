"""Independent score/provenance regressions; no fit or optimizer calls."""
from copy import deepcopy
import json

import numpy as np
import pytest

from rnj_wzzt_dr.experiments.rnj_pool_only_20261116.compare import (
    Reader, audit_native, compare, loss, manifest, response, sha, strict_selection,
    topology, training_score)


def point(value):
    return dict(supports=[[0], [1], [2]], r=[value]*3, x=[0.]*3)


def fixture():
    obs=dict(P=np.ones((4,3)),Q=np.zeros((4,3)),Y=np.ones((4,3)),labels=[101,102,103])
    config=dict(alpha=.003,zero_tol=1e-9,gain_tol=1e-10,validation_tie=1e-12,internal_floor=.08)
    current,raw,closed=point(.3),point(.8),point(1.)
    for p in (current,raw,closed):p.update(objective=training_score(p,obs,1.,config['alpha']))
    candidates=[]
    for p in (current,closed):
        h=response(p,3);candidates.append(dict(point=p,labels=obs['labels'],R=h[:,:3].tolist(),X=h[:,3:].tolist(),validation_mse=loss(h,obs)['mse']))
    solver=dict(status=9,objective=raw['objective'],bound=0.,gap=1.,seconds=.1,solve_started=True,
                fixed_family_qp_objective=closed['objective'],scope='finite laminar support pool')
    event=dict(stage='E',accepted=True,before_score=current['objective'],after_score=closed['objective'],
        point=closed,initial_point=current,initial_score=current['objective'],initial_source='current',
        raw_point=raw,native_returned_score=raw['objective'],closed_point=closed,closed_score=closed['objective'],
        native_improved_initial=True,pretrial_QP_count=0,solver=solver)
    native=dict(method='101',config=config,reference_mse=1.,frozen_candidates=candidates,
        selected=dict(index=1),path=[current,closed],events=[event],MIQP=[solver],QP=[],total_seconds=.2)
    return native,obs


def test_outer_products_and_full_vs_internal_scope():
    p=point(.1);p['supports'].append([0,1,2]);p['r'].append(.2);p['x'].append(.3)
    h=response(p,3)
    np.testing.assert_allclose(h[:,:3],.2*np.ones((3,3))+.1*np.eye(3))
    np.testing.assert_allclose(h[:,3:],.3*np.ones((3,3)))
    t=topology(p,[[0],[1],[2]],3,1e-9)
    assert t['internal_exact'] and not t['full_exact'] and t['full_extra']==1


def test_strict_tie_counts_stored_supports_even_when_zero():
    a=point(1.);b=deepcopy(a);b['supports'].append([0,1]);b['r'].append(0.);b['x'].append(0.)
    _,obs=fixture()
    assert strict_selection([b,a],[obs],1.,1e-12)['selected']==1


def test_cumulative_uses_cells_not_equal_day_means():
    _,obs=fixture();val={**obs,'P':obs['P'][:1],'Q':obs['Q'][:1],'Y':obs['Y'][:1]}
    future={**obs,'Y':3*obs['Y']}
    selected=strict_selection([point(1.),point(3.)],[val,future],1.,1e-12)
    assert selected['selected']==1 and selected['cells']==15


def test_native_improvement_and_qp_closure_are_separate():
    native,obs=fixture();r=audit_native(native,obs,obs,obs,[[0],[1],[2]],response(point(1.),3))
    event=r['events'][0]
    assert event['initial_evidence']=='recomputed_point'
    assert event['native_gain']==pytest.approx(.45)
    assert event['closure_gain']==pytest.approx(.04)
    assert r['MIQP_timeouts']==1 and r['phases']['validation']['topology']['full_exact']
    assert r['phases']['validation']['matrix']['R_relative_error']==0.


def test_legacy_initial_unknown_is_not_inferred_from_final():
    native,obs=fixture()
    for k in ('initial_point','initial_score','initial_source','native_improved_initial'):
        native['events'][0].pop(k)
    r=audit_native(native,obs,obs,None)
    assert r['events'][0]['initial_evidence']=='unavailable'
    assert r['events'][0]['native_gain'] is None


def test_reference_initialization_gain_is_not_native_search_gain():
    native,obs=fixture();event=native['events'][0]
    reference=point(.7);initial=training_score(reference,obs,1.,.003)
    event.update(initial_point=reference,initial_score=initial,initial_source='single_RNJ_reference',
                 current_to_initial_gain=event['before_score']-initial)
    result=audit_native(native,obs,obs,None)['events'][0]
    assert result['current_score']==pytest.approx(.49)
    assert result['current_to_initial_gain']==pytest.approx(.40)
    assert result['native_gain']==pytest.approx(.05)
    assert result['closure_gain']==pytest.approx(.04)
    assert result['initial_source']=='single_RNJ_reference'
    event['current_to_initial_gain']=.45
    with pytest.raises(ValueError,match='initialization gain'):audit_native(native,obs,obs,None)


def test_false_native_improvement_and_wrong_selected_fail():
    native,obs=fixture();native['events'][0]['native_improved_initial']=False
    with pytest.raises(ValueError,match='improvement flag'):audit_native(native,obs,obs,None)
    native,obs=fixture();native['selected']['index']=0
    with pytest.raises(ValueError,match='winner'):audit_native(native,obs,obs,None)


def write_json(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')
    return dict(path=str(path.resolve()),sha256=sha(path))


def sealed_run(tmp_path,name,failed=False):
    native,obs=fixture();root=tmp_path/name;root.mkdir()
    files={}
    for role in ('train','validation','new_day'):
        p=root/(role+'.npz');np.savez(p,**obs);files[role+'.npz']=dict(path=str(p),sha256=sha(p))
    physical=root/'Hstar.npz';np.savez(physical,Hstar=response(point(1.),3),labels=obs['labels'],support_json=json.dumps([[0],[1],[2]]))
    native_ref=write_json(root/'native.json',native)
    folder=root/'cases'/'test';folder.mkdir(parents=True)
    write_json(folder/'result.json',dict(status='failed',error='Time-budget construction failed') if failed else dict(status='complete',native=native_ref,sources_unchanged=True))
    entry=dict(id='test',case='toy',files=files,Hstar_ref=dict(path=str(physical),sha256=sha(physical)))
    inputs=write_json(root/'inputs.json',[entry]);write_json(root/'protocol.json',dict(inputs=inputs))
    return root,entry


def test_paired_report_and_explicit_failure_denominator(tmp_path):
    old,_=sealed_run(tmp_path,'old');new,_=sealed_run(tmp_path,'new',failed=True)
    report=compare(old,new,tmp_path/'report')
    assert report['status']=='passed' and report['planned_pairs']==1 and report['completed_pairs']==0
    assert report['rows'][0]['new']['status']=='failed'
    assert (tmp_path/'report'/'comparison.csv').exists()
    with pytest.raises(ValueError,match='overwrite'):compare(old,new,tmp_path/'report')


def test_hash_mismatch_and_different_paired_data_fail_closed(tmp_path):
    p=tmp_path/'source.json';ref=write_json(p,{'a':1});p.write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):Reader().json(ref)
    old,_=sealed_run(tmp_path,'old');new,entry=sealed_run(tmp_path,'new')
    p=new/'train.npz';_,obs=fixture();np.savez(p,**{**obs,'Y':obs['Y']*2})
    entry['files']['train.npz']['sha256']=sha(p)
    inputs=write_json(new/'inputs.json',[entry]);write_json(new/'protocol.json',dict(inputs=inputs))
    result=compare(old,new,tmp_path/'report')
    assert result['status']=='failed' and result['completed_pairs']==0
    assert 'Paired observations differ' in result['errors'][0]['error']


def test_completed_pair_includes_future_and_native_gain(tmp_path):
    old,_=sealed_run(tmp_path,'old');new,_=sealed_run(tmp_path,'new')
    result=compare(old,new,tmp_path/'report')
    assert result['status']=='passed' and result['completed_pairs']==1
    assert result['rows'][0]['new']['phases']['cumulative']['cells']==24
    assert result['rows'][0]['new']['events'][0]['native_gain']==pytest.approx(.45)


def test_actual_inputs_ref_schema_verifies_private_snapshot(tmp_path):
    root,_=sealed_run(tmp_path,'run');snapshot=tmp_path/'old_source'/'rnj_wzzt_dr';snapshot.mkdir(parents=True)
    source=write_json(snapshot/'source.json',{'old_version':True})
    write_json(root/'protocol.json',dict(inputs=4,inputs_ref=dict(path=str(root/'inputs.json'),sha256=sha(root/'inputs.json')),
        source_root=str(snapshot),sources=[source]))
    reader=Reader();_,entries=manifest(reader,root)
    assert list(entries)==['test'] and str((snapshot/'source.json').resolve()) in reader.sources
    (snapshot/'source.json').write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):manifest(Reader(),root)
