"""Archive completed synthetic follow-up and independent interface evidence."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def export(root,repo,build,out):
    root,repo,build,out=map(Path,(root,repo,build,out));out.mkdir(parents=True,exist_ok=True)
    # Receipts are byte-hashed. Preserve them across Git newline conversions.
    (out/'.gitattributes').write_bytes(b'* -text\n')
    p=read(root/'protocol.json');audit=read(root/'audit_final.json')
    assert audit['all_passed'] and audit['recorded_cells']==210 and audit['unique_fits']==104
    receipts=['protocol.json','observed_history.json','observed_training_history.json','all_past_witness.json',
              'history_audit.json','planning.json','policies.json','test_draws.json','summary.json','analysis.json',
              'audit_first8.json','audit_final.json']
    for name in receipts:shutil.copyfile(root/name,out/name)
    for name in ['ac_interface_tests.txt','ac_interface_test_receipt.json']:shutil.copyfile(build/name,out/name)
    entries={}
    for category,pattern in [('record','records/*.json'),('input','inputs/*.json'),
        ('fit_result','fits/*/result.json'),('fit_checkpoint','fits/*/checkpoint.json'),
        ('history_actual_fit','history_fit/*.json')]:
        for f in sorted(root.glob(pattern)):entries['run/'+f.relative_to(root).as_posix()]=(f,category)
    entries['run/history_audit_checkpoint.json']=(root/'history_audit_checkpoint.json','history_audit_checkpoint')
    for name,expected in p['source_sha256'].items():
        path=Path(name);assert sha(path.read_bytes())==expected,str(path)
        entries['frozen_source/'+path.relative_to(repo).as_posix()]=(path,'frozen_source')
    for name in ['analyze.py','audit_followup.py','export_evidence.py']:
        entries['postprocess_source/'+name]=(build/name,'postprocess_source')
    ac=read(build/'ac_interface_test_receipt.json')
    # Original test receipt hashes captured LF text; Windows saved CRLF bytes.
    # Preserve both originals and declare the distinction in the manifest.
    ac_log=build/'ac_interface_tests.txt'
    assert ac['exit_code']==0 and sha(ac_log.read_text(encoding='utf-8').encode())==ac['output_sha256']
    for name,expected in ac['source_sha256'].items():
        path=repo/name;assert sha(path.read_bytes())==expected,name
        entries['AC_interface_source/'+name]=(path,'AC_interface_source')
    members=[];archive=out/'synthetic_run.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,(f,category) in sorted(entries.items()):
            data=f.read_bytes();info=zipfile.ZipInfo(name,date_time=(2026,10,29,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
            members.append(dict(path=name,category=category,bytes=len(data),sha256=sha(data)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in members:assert sha(z.read(row['path']))==row['sha256']
    counts=dict(Counter(row['category'] for row in members))
    assert [counts[k] for k in ['record','input','fit_result','fit_checkpoint']]==[210,104,104,104]
    manifest=dict(implementation_commit='98ad71d7',export_source_sha256=sha(Path(__file__).read_bytes()),
        AC_test_log_hash=dict(original_receipt_semantics='UTF-8 captured text normalized to LF',
            receipt_sha256=ac['output_sha256'],saved_file_bytes_sha256=sha(ac_log.read_bytes())),
        archive_counts=counts,members=members,readback_all_passed=True,
        files={f.name:dict(bytes=f.stat().st_size,sha256=sha(f.read_bytes())) for f in sorted(out.iterdir()) if f.is_file() and f.name!='manifest.json'},
        scope='Complete synthetic history/control stage; no raw household load, no AC enhancement applied to DR, no new fitting in export.')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(counts=counts,archive_bytes=archive.stat().st_size,readback_all_passed=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for flag in ['root','repo','build','out']:p.add_argument('--'+flag,type=Path,required=True)
    a=p.parse_args();export(a.root,a.repo,a.build,a.out)
