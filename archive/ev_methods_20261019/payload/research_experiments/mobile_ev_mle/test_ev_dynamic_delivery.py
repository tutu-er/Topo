"""Tiny archive fixtures only; no research batches or solver imports."""
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from . import ev_dynamic_delivery as delivery


def _phases(root, *, bad_mtime=False):
    phases = {}
    # More than two stream chunks; the same decoded snapshot is shared by phases.
    snapshot = (b'{"opaque_source_snapshot":"not executed"}\n'*60000)+b"end"
    for index,label in enumerate(("dev","calibration","evaluation")):
        path = root/label
        path.mkdir(parents=True)
        (path/"freeze.json").write_bytes(b'{"policy_id":"fixture"}\n')
        (path/"source_snapshot.json.gz").write_bytes(gzip.compress(snapshot,mtime=0))
        trial = json.dumps(dict(seed=index,opaque="saved record")).encode()
        (path/("trial_"+str(index)+".json.gz")).write_bytes(
            gzip.compress(trial,mtime=7 if bad_mtime and index == 0 else 0))
        (path/"checkpoint.json").write_bytes(json.dumps(dict(completed=index)).encode())
        (path/"ignored.txt").write_bytes(b"outside whitelist")
        phases[label] = path
    return phases


def _tar_payloads(path):
    with tarfile.open(path,"r:xz") as archive:
        return [(member,archive.extractfile(member).read()) for member in archive]


def _rewrite(path, output, *, unsafe_path=None, extra=False, symlink=False):
    payloads = _tar_payloads(path)
    manifest = json.loads(payloads[0][1])
    if unsafe_path is not None:
        manifest["entries"][0]["logical_path"] = unsafe_path
        manifest["manifest_id"] = delivery._hash(
            {k:v for k,v in manifest.items() if k != "manifest_id"})
    data = delivery._json_bytes(manifest)
    with tarfile.open(output,"w:xz") as archive:
        archive.addfile(delivery._tar_info("manifest.json",len(data)),io.BytesIO(data))
        for index,(member,data) in enumerate(payloads[1:]):
            if symlink and index == 0:
                link = delivery._tar_info(member.name,0)
                link.type = tarfile.SYMTYPE
                link.linkname = "../../outside"
                archive.addfile(link)
            else:
                archive.addfile(delivery._tar_info(member.name,len(data)),io.BytesIO(data))
        if extra:
            archive.addfile(delivery._tar_info("unlisted.txt",1),io.BytesIO(b"x"))


def test_pack_restore_all_original_sha_equal_dedup_and_reproducible(tmp_path):
    phases = _phases(tmp_path/"inputs")
    packed = delivery.pack_archive(phases,tmp_path/"evidence.tar.xz")
    manifest = packed["manifest"]
    assert manifest["logical_files"] == 12
    assert manifest["unique_objects"] < manifest["logical_files"]
    assert manifest["unique_decoded_bytes"] < manifest["logical_decoded_bytes"]
    assert {e["logical_path"].split("/")[0] for e in manifest["entries"]} == set(phases)
    assert "ignored.txt" not in json.dumps(manifest)
    sidecar = json.loads(Path(packed["sidecar"]).read_bytes())
    assert sidecar["archive_sha256"] == hashlib.sha256(Path(packed["archive"]).read_bytes()).hexdigest()
    restored = delivery.restore_archive(packed["archive"],tmp_path/"restored")
    assert restored["exact_byte_restore"] and len(restored["restored_files"]) == 12
    for entry in manifest["entries"]:
        phase,name = entry["logical_path"].split("/")
        original = phases[phase]/name
        recovered = tmp_path/"restored"/phase/name
        assert original.read_bytes() == recovered.read_bytes()
        assert delivery._sha_file(recovered) == entry["original_sha256"]
    actual_files = {str(p.relative_to(tmp_path/"restored")).replace("\\","/")
                    for p in (tmp_path/"restored").rglob("*") if p.is_file()}
    assert actual_files == {e["logical_path"] for e in manifest["entries"]}
    repeated = delivery.pack_archive(dict(reversed(list(phases.items()))),tmp_path/"again.tar.xz")
    assert repeated["archive_sha256"] == packed["archive_sha256"]
    assert not list(tmp_path.glob(".ev-delivery-*"))


@pytest.mark.parametrize("unsafe",["../outside/freeze.json","/dev/freeze.json",
                                  r"dev\freeze.json","dev/freeze.json:stream"])
def test_illegal_paths_rejected_before_output_writes(tmp_path,unsafe):
    phases = _phases(tmp_path/"inputs")
    packed = delivery.pack_archive(phases,tmp_path/"good.tar.xz")
    bad = tmp_path/"bad.tar.xz"
    _rewrite(packed["archive"],bad,unsafe_path=unsafe)
    with pytest.raises(ValueError,match="unsafe|whitelist"):
        delivery.restore_archive(bad,tmp_path/"out")
    assert not (tmp_path/"out").exists()
    assert not (tmp_path/"outside").exists()


def test_nonempty_output_and_existing_archive_refused_without_changes(tmp_path):
    phases = _phases(tmp_path/"inputs")
    packed = delivery.pack_archive(phases,tmp_path/"good.tar.xz")
    output = tmp_path/"notempty"
    output.mkdir()
    marker = output/"keep.txt"
    marker.write_bytes(b"keep unchanged")
    with pytest.raises(ValueError,match="empty"):
        delivery.restore_archive(packed["archive"],output)
    assert list(output.iterdir()) == [marker] and marker.read_bytes() == b"keep unchanged"
    original_sha = delivery._sha_file(packed["archive"])
    with pytest.raises(FileExistsError):
        delivery.pack_archive(phases,packed["archive"])
    assert delivery._sha_file(packed["archive"]) == original_sha


def test_gzip_reconstruction_failure_saves_verified_raw_and_reports(tmp_path):
    phases = _phases(tmp_path/"inputs",bad_mtime=True)
    packed = delivery.pack_archive(phases,tmp_path/"good.tar.xz")
    with pytest.raises(delivery.ByteRestoreError,match="raw saved") as caught:
        delivery.restore_archive(packed["archive"],tmp_path/"restored")
    report = caught.value.report
    assert not report["exact_byte_restore"] and len(report["failures"]) == 1
    failed = report["failures"][0]
    assert failed["logical_path"] == "dev/trial_0.json.gz"
    assert not (tmp_path/"restored"/failed["logical_path"]).exists()
    raw = tmp_path/"restored"/failed["raw_fallback_path"]
    assert raw.read_bytes() == gzip.decompress((phases["dev"]/"trial_0.json.gz").read_bytes())
    entry = next(e for e in packed["manifest"]["entries"] if e["logical_path"] == failed["logical_path"])
    assert delivery._sha_file(raw) == entry["decoded_sha256"]
    assert failed["raw_fallback_path"] == entry["raw_fallback_path"]
    assert report["pack_environment"]["python"]
    assert report["restore_environment"]["zlib_runtime"]
    assert not list(tmp_path.glob(".ev-delivery-*"))


@pytest.mark.parametrize("kind",["extra","symlink"])
def test_unlisted_or_nonregular_members_rejected(tmp_path,kind):
    phases = _phases(tmp_path/"inputs")
    packed = delivery.pack_archive(phases,tmp_path/"good.tar.xz")
    bad = tmp_path/"bad.tar.xz"
    _rewrite(packed["archive"],bad,extra=kind=="extra",symlink=kind=="symlink")
    with pytest.raises(ValueError,match="unlisted|nonregular"):
        delivery.restore_archive(bad,tmp_path/"out")
    assert not (tmp_path/"out").exists()
