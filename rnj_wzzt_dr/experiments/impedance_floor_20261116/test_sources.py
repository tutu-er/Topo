"""Input-boundary and sealed-file tests; no parameter fitting or AC."""
import json
from pathlib import Path

import numpy as np
import pytest

from . import sources


def write(path, value):
    Path(path).write_text(json.dumps(value), encoding='utf-8')
    return sources.ref(path)


def tiny_entry(tmp_path):
    labels = [101, 102]
    p = np.arange(12, dtype=float).reshape(6, 2)/100
    q = p/3
    v = np.full_like(p, 1.0); root = np.full(6, 1.02)
    path = tmp_path/'day.npz'; np.savez(path, P=p, Q=q, V=v, root=root, labels=labels)
    oref = sources.ref(path)
    native = dict(selection_metric='validation_mse', truth_used=False, test_used=False,
                  reference_mse=1e-5, config=dict(validation_tie=1e-12, zero_tol=1e-9, response_bound=2.),
                  frozen_candidates=[dict(index=7, point=dict(supports=[[0],[1]],r=[.1,.1],x=[.1,.1]))],
                  selected=dict(index=7))
    nref = write(tmp_path/'native.json', native)
    pref = write(tmp_path/'protocol.json', {})
    seal = write(tmp_path/'seal.json', dict(native=nref,train=oref,validation=oref,truth_used=False))
    return dict(availability='available', labels=labels, protocol_ref=pref,
                native_wrapper_key=None, native_ref=nref, candidate_seal_ref=seal,
                train_ref=oref, validation_ref=oref, train_rows=[0,4], validation_rows=[4,6],
                new_day_ref={'path':str(tmp_path/'MUST_NOT_OPEN.npz'), 'sha256':'x'},
                evaluation_case_refs={'MUST_NOT_OPEN':True}, zbase_ohm=.8)


def test_training_never_opens_heldout_or_physical_truth(tmp_path, monkeypatch):
    entry = tiny_entry(tmp_path)
    original = sources.checked; opened=[]
    def tracked(reference):
        opened.append(reference['path'])
        assert 'MUST_NOT_OPEN' not in reference['path']
        return original(reference)
    monkeypatch.setattr(sources, 'checked', tracked)
    result = sources.load_training(entry)
    assert result['train'].points == 4 and result['validation'].points == 2
    np.testing.assert_array_equal(result['validation'].p, np.arange(12).reshape(6,2)[4:]/100)
    np.testing.assert_allclose(result['train'].y, 1.02**2-1.)
    assert result['expected_selected_index'] == 7  # preserve raw ID, not ordinal 0
    assert not any('MUST_NOT_OPEN' in name for name in opened)


def test_observation_mutation_is_rejected(tmp_path):
    entry = tiny_entry(tmp_path)
    with Path(entry['train_ref']['path']).open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='Frozen source changed'):
        sources.load_training(entry)


def test_valid_file_from_wrong_candidate_seal_is_rejected(tmp_path):
    entry=tiny_entry(tmp_path)
    other=tmp_path/'other.npz'
    np.savez(other,P=np.ones((6,2)),Q=np.ones((6,2)),Y=np.ones((6,2)),labels=[101,102])
    entry['train_ref']=sources.ref(other)
    with pytest.raises(ValueError,match='Candidate seal/historical observation mismatch'):
        sources.load_training(entry)


def test_reversed_or_out_of_bounds_slice_rejected(tmp_path):
    entry = tiny_entry(tmp_path)
    for rows in ([5,4], [0,7], [-1,2]):
        with pytest.raises(ValueError, match='row range'):
            sources.slice_observation(entry['train_ref'], rows, entry['labels'])


def test_wrong_terminal_order_rejected(tmp_path):
    entry = tiny_entry(tmp_path)
    with pytest.raises(ValueError, match='terminal order'):
        sources.slice_observation(entry['train_ref'], [0,4], [102,101])


def test_zero_impedance_physical_branch_is_not_removed():
    branches = [dict(from_bus=0,to_bus=1,r_ohm=0,x_ohm=0,status=True),
                dict(from_bus=1,to_bus=101,r_ohm=0,x_ohm=0,status=True),
                dict(from_bus=1,to_bus=102,r_ohm=1,x_ohm=1,status=True)]
    assert sources.support_truth(branches, [101,102], 0) == [[0],[1],[0,1]]


def test_disconnected_or_non_tree_truth_rejected():
    with pytest.raises(ValueError, match='disconnected'):
        sources.support_truth([dict(from_bus=9,to_bus=101)], [101], 0)
    with pytest.raises(ValueError, match='not a rooted'):
        sources.support_truth([dict(from_bus=0,to_bus=101),dict(from_bus=9,to_bus=101)], [101], 0)


def test_pressure_truth_is_support_only(tmp_path):
    b=tmp_path/'branches.csv'
    b.write_text('from_bus,to_bus,r_ohm,x_ohm,status\n0,101,0,0,True\n0,102,1,1,True\n',encoding='utf-8')
    m=write(tmp_path/'metadata.json',dict(root_bus=0,base_kv=.4,base_mva=.2))
    t=write(tmp_path/'truth.json',dict(labels=[101,102],proper_clades=[]))
    e=dict(labels=[101,102],evaluation_case_refs={'branches.csv':sources.ref(b),'metadata.json':m},
           legacy_truth_ref=t,new_day_ref=None)
    result=sources.load_evaluation(e)
    assert result['truth_supports']==[[0],[1]] and result['new_day'] is None
    assert result['Hstar'] is None and not result['true_response_available']


def test_unavailable_entry_remains_explicit_failure():
    with pytest.raises(ValueError,match='Planned input unavailable'):
        sources.load_training(dict(availability='unavailable',load_errors=['missing original fit']))


def test_corrupt_group_keeps_all_planned_denominators(tmp_path, monkeypatch):
    def group(root, name):
        if name == 'history288':
            raise ValueError('changed original protocol')
        return [dict(id=name+'__'+str(i),source_group=name,availability='available')
                for i in range(sources.GROUPS[name][1])]
    monkeypatch.setattr(sources,'_inventory_group',group)
    rows=sources.inventory(tmp_path)
    assert len(rows)==456
    missing=[r for r in rows if r['availability']=='unavailable']
    assert len(missing)==40 and all(r['source_group']=='history288' for r in missing)
    assert all('changed original protocol' in r['load_errors'][0] for r in missing)
