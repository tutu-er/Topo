"""Fingerprint the complete scientific dependency tree even inside outputs/."""
from types import SimpleNamespace
import sys
import pytest
from . import io


def test_execution_inside_outputs_does_not_exclude_all_science():
    fingerprints=io.source_hashes()
    for name in (
        'research_experiments/mobile_ev_mle/full_pipeline/study.py',
        'research_experiments/mobile_ev_mle/full_pipeline/solver.py',
        'research_experiments/mobile_ev_mle/likelihood.py',
        'research_experiments/mobile_ev_mle/ev_rx_fit.py',
        'rnj_wzzt_core/rnj_wzzt/__init__.py',
        'terminal_case33/__init__.py',
    ):
        assert name in fingerprints, name
    assert len(fingerprints)>100
    assert all(len(value)==64 for value in fingerprints.values())
    assert not any('/evidence/' in name or '/outputs/' in name for name in fingerprints)


def test_freeze_rejects_repository_import_outside_snapshot(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'terminal_case33.external_test',
        SimpleNamespace(__file__=str(io.ROOT.parent/'outside_execution_snapshot.py')))
    prepared=tmp_path/'prepared.json';prepared.write_text('{}',encoding='utf8')
    with pytest.raises(RuntimeError,match='escaped execution snapshot'):
        io.freeze(tmp_path/'frozen',{},prepared)
