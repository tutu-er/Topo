"""Research-scope filtering tests; no fitting, optimization or AC."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rnj_wzzt_dr import evaluation_scope as scope


@pytest.fixture
def catalog(tmp_path):
    entries = [
        {'id': 'main', 'source_group': 'large96_main', 'status': 'failed',
         'availability': 'unavailable', 'load_errors': ['preserve failure'], 'nested': {'values': [1]}},
        {'id': 'current40', 'source_group': 'history288', 'status': 'unresolved'},
        {'id': 'current16', 'source_group': 'history10', 'status': 'complete'},
        {'id': 'pressure', 'source_group': 'large96_pressure', 'status': 'failed'}]
    path = tmp_path/'inputs.json'
    path.write_text(json.dumps(entries), encoding='utf-8')
    config = dict(schema='research_evaluation_scope_v1', scope_id='test',
        active_groups=dict(large96_main=1, history288=1, history10=1),
        retired_groups=dict(large96_pressure=1), active_count=3, historical_count=4,
        source_inventory=dict(path='inputs.json', sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    config_path = tmp_path/'scope.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    with patch.object(scope, 'ROOT', tmp_path), patch.object(scope, 'CONFIG_PATH', config_path):
        yield entries, path, config, config_path


def test_default_retains_failed_and_unresolved_active_entries(catalog):
    entries, _, _, _ = catalog
    result = scope.load_catalog()
    assert [r['id'] for r in result] == ['main', 'current40', 'current16']
    assert result[0]['status'] == 'failed'
    assert result[0]['load_errors'] == ['preserve failure']
    assert result[1]['status'] == 'unresolved'
    assert all(r['evaluation_scope']['status'] == 'active' for r in result)
    assert entries[0]['status'] == 'failed'


def test_historical_opt_in_labels_retired_without_overwriting_failure(catalog):
    result = scope.load_catalog(include_retired=True)
    assert len(result) == 4
    assert result[-1]['evaluation_scope']['status'] == 'retired'
    assert not result[-1]['evaluation_scope']['included_in_default']
    assert result[-1]['status'] == 'failed'


def test_selection_deep_copies_and_supports_subsets(catalog):
    entries, _, _, _ = catalog
    before = deepcopy(entries)
    result = scope.select_entries([entries[0], entries[-1]])
    result[0]['nested']['values'].append(2)
    assert entries == before
    assert 'evaluation_scope' not in entries[0]


def test_unknown_group_is_not_silently_filtered(catalog):
    entries, _, _, _ = catalog
    with pytest.raises(ValueError, match='Unknown source group'):
        scope.select_entries(entries + [{'id': 'unknown', 'source_group': 'new_unreviewed_group'}])


def test_duplicate_ids_are_rejected_even_if_retired(catalog):
    entries, _, _, _ = catalog
    with pytest.raises(ValueError, match='unique'):
        scope.select_entries(entries + [deepcopy(entries[-1])])


def test_explicit_boolean_required(catalog):
    with pytest.raises(TypeError, match='boolean'):
        scope.load_catalog(include_retired='false')


def test_changed_inventory_hash_rejected(catalog):
    _, path, _, _ = catalog
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='SHA-256'):
        scope.load_catalog()


def test_missing_entry_not_hidden_by_new_valid_hash(catalog):
    entries, path, config, config_path = catalog
    path.write_text(json.dumps(entries[:-1]), encoding='utf-8')
    config['source_inventory']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    config_path.write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(ValueError, match='counts changed'):
        scope.load_catalog()


def test_overlapping_groups_rejected(catalog):
    _, _, config, config_path = catalog
    config['retired_groups']['history10'] = 1
    config_path.write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(ValueError, match='both active and retired'):
        scope.load_catalog()


def test_scope_counts_and_source_root_are_checked(catalog):
    _, _, config, config_path = catalog
    config['active_count'] = 2
    config_path.write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(ValueError, match='totals'):
        scope.load_catalog()
    config['active_count'] = 3
    config['source_inventory']['path'] = '../outside.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(ValueError, match='inside the workspace'):
        scope.load_catalog()


def test_cli_preserves_existing_output(catalog, tmp_path):
    out = tmp_path/'export.json'
    assert scope.main(['--out', str(out)]) == 0
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        scope.main(['--include-retired', '--out', str(out)])
    assert out.read_bytes() == before
