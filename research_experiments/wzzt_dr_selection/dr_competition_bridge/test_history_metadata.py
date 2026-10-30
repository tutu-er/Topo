"""Source-binding boundaries, using generated observations only."""
import hashlib
import json
import numpy as np
import pytest
from research_experiments.wzzt_dr_selection.dr_competition_bridge.physical_bridge import (
    history_source_metadata, load_history,
)


def binding(tmp_path, case='flynn16'):
    path = tmp_path / 'networks' / case / 'reference' / 'metadata.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(root_bus=1, base_kv=.4, base_mva=.2,
        source=dict(case_name='flynn_balanced_lv_16', unused='not returned'))), encoding='utf-8')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                case_name='flynn_balanced_lv_16')


def row_with_observed(tmp_path, case, labels=(201, 202)):
    row = dict(job_id='generated-fixture', case=case, N=288, condition='reference')
    for role in ('train', 'validation'):
        path = tmp_path / (role + '_observed.npz')
        np.savez(path, labels=np.asarray(labels), P=np.full((3, 96, 2), .01),
            Q=np.full((3, 96, 2), .003), V=np.full((3, 96, 2), .98), root=np.ones((3, 96)))
        row[role] = dict(path=str(path), catalog_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return row


def test_nonlegacy_requires_explicit_metadata():
    with pytest.raises(ValueError, match='explicit source metadata'):
        history_source_metadata(dict(case='flynn16', condition='reference'), expected_case='flynn16')


def test_base_mismatch_rejected(tmp_path):
    with pytest.raises(ValueError, match='power base'):
        history_source_metadata(dict(case='flynn16', condition='reference'),
            expected_case='flynn16', base_kw=100., source_metadata=binding(tmp_path))


def test_hash_and_case_binding_rejected(tmp_path):
    bound = binding(tmp_path)
    bad = {**bound, 'sha256': '0' * 64}
    with pytest.raises(RuntimeError, match='hash mismatch'):
        history_source_metadata(dict(case='flynn16', condition='reference'),
            expected_case='flynn16', source_metadata=bad)
    with pytest.raises(ValueError, match='case_name mismatch'):
        history_source_metadata(dict(case='flynn16', condition='reference'),
            expected_case='flynn16', source_metadata={**bound, 'case_name': 'paper_style_terminal_lv_15'})
    with pytest.raises(ValueError, match='path does not match'):
        history_source_metadata(dict(case='paper15', condition='reference'),
            expected_case='paper15', source_metadata=bound)


def test_generic_history_preserves_all_rows_and_disjoint_synthetic_dates(tmp_path):
    history = load_history(row_with_observed(tmp_path, 'flynn16'),
        expected_case='flynn16', source_metadata=binding(tmp_path))
    assert history['labels'] == [201, 202] and history['root_label'] == 1
    assert history['base_mva'] == .2 and history['base_kw'] == 200.
    assert history['receipt']['rows'] == {'training': 288, 'validation': 288}
    assert len(history['training']) == len(history['validation']) == 3
    assert history['training'][-1]['P_terminal'].index[-1] < history['validation'][0]['P_terminal'].index[0]
    assert history['receipt']['NPZ_test_loaded'] is False and history['receipt']['truth_loaded'] is False
    assert 'unused' not in json.dumps(history['receipt']['source_metadata'])


def test_legacy_paper15_call_retained(tmp_path):
    history = load_history(row_with_observed(tmp_path, 'paper15'))
    assert history['base_kw'] == 200. and history['root_label'] == 1
    assert history['receipt']['source_metadata']['source_binding'].startswith('legacy paper15')


def test_root_terminal_overlap_rejected(tmp_path):
    with pytest.raises(ValueError, match='root cannot be a terminal'):
        load_history(row_with_observed(tmp_path, 'flynn16', labels=(1, 201)),
            expected_case='flynn16', source_metadata=binding(tmp_path))
