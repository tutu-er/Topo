"""Relocated sealed assets retain their hashes and readable references."""
import hashlib
import json
from unittest.mock import patch

from rnj_wzzt_dr import io, paths


def test_sealed_metadata_is_rebound_without_rewriting_bytes(tmp_path):
    root = tmp_path/'checkout'
    folder = root/'data'/'history10_rx_20261116_v1'
    folder.mkdir(parents=True)
    payload = folder/'payload.bin'
    payload.write_bytes(b'unchanged measurement payload')
    old = root/'outputs'/'history10_rx_20261116_v1'
    reference = dict(path=str(old/'payload.bin'), sha256=hashlib.sha256(payload.read_bytes()).hexdigest())
    metadata = folder/'sealed.json'
    metadata.write_text(json.dumps({'input': reference}), encoding='utf8')
    before = metadata.read_bytes()
    with patch.object(paths, 'ROOT', root), patch.object(paths, 'LEGACY_ROOT', root):
        loaded = io.read(old/'sealed.json')
        assert loaded['input'] == io.ref(payload)
        assert paths.resolve_path(old/'payload.bin') == payload
    assert metadata.read_bytes() == before


def test_missing_replacement_and_unrelated_outputs_are_not_redirected(tmp_path):
    with patch.object(paths, 'ROOT', tmp_path), patch.object(paths, 'LEGACY_ROOT', tmp_path):
        missing = tmp_path/'outputs'/'history10_rx_20261116_v1'/'missing.npz'
        assert paths.resolve_path(missing) == missing
        unrelated = tmp_path/'outputs'/'new_experiment'/'result.json'
        assert paths.resolve_path(unrelated) == unrelated
