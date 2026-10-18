"""Package only this numerical follow-up and replay its CPU tests in isolation.

No scientific worker is started. Original failure records and exact old source
snapshots are retained. No CUDA dependency or old evidence ZIP is repackaged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from importlib.metadata import version as distribution_version
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import xml.etree.ElementTree as ET
import zipfile


STAGE=Path(__file__).resolve().parents[3]
MODULE=Path(__file__).resolve().parent
MODULE_REL=Path('research_experiments/mobile_ev_mle/ac_block_fit')
ORIGINAL=Path('D:/0-github_workspace/Topo/outputs/ev_reference_stage_20261018')
EVIDENCE_ROOTS=(
    'common_gradient_diagnosis_v1','candidate_h24_v1','candidate_h24_polish_v1',
    'polished_wrapper_v1','polished_wrapper_v2','polished_wrapper_independent_review_v1',
    'case33_crosswarm_v1','finite_ev_ac_audit_v1','case33_geometry_v1')


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')


def assert_original_unchanged(original=ORIGINAL):
    freeze_path=Path(original)/'outputs/reference_v1/freeze.json'
    freeze=_read(freeze_path)
    with zipfile.ZipFile(MODULE/'evidence/numerical_evidence_v1.zip') as archive:
        archived_freeze=json.loads(archive.read('dependencies/original_freeze.json'))
    if freeze['source_sha256']!=archived_freeze['source_sha256'] or freeze['identity']!=archived_freeze['identity']:
        raise RuntimeError('original seal differs from the already archived v1 seal')
    if len(freeze['source_sha256'])!=893:
        raise RuntimeError('expected the original 893-file scientific seal')
    changed=[name for name,expected in freeze['source_sha256'].items()
        if not (Path(original)/name).is_file() or _digest(Path(original)/name)!=expected]
    if changed:
        raise RuntimeError('original frozen scientific source changed/missing: '+repr(changed))
    return freeze_path,freeze


def _junit(path):
    root=ET.parse(path).getroot()
    suites=[root] if root.tag=='testsuite' else list(root.iter('testsuite'))
    return {key:sum(int(s.attrib.get(key,'0')) for s in suites)
        for key in ('tests','failures','errors','skipped')}


def collect_worker(args):
    """Run all CPU tests, then report actual imported project files."""
    began=perf_counter();root=Path(args.root).resolve();output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    # Ancestor pyproject.toml can set pytest's pythonpath to the primary
    # checkout. Use an empty local config and explicit paths for this replay.
    configuration=output/'pytest.ini'
    configuration.write_text('[pytest]\n',encoding='utf-8')
    sys.path[:0]=[str(root/'rnj_wzzt_core'),str(root)]
    import pytest
    status=pytest.main([str(root/MODULE_REL),'-q','-p','no:cacheprovider',
        '-c',str(configuration),'--rootdir='+str(root),
        '--ignore='+str(root/MODULE_REL/'test_gpu.py'),
        '--basetemp='+str(output/'pytest_temp'),
        '--junitxml='+str(output/'tests.xml')])
    local,foreign=[],[]
    project_roots=(Path('D:/0-github_workspace/Topo').resolve(),
        Path('C:/Users/23761/.codex/worktrees/36a1/Topo').resolve())
    for name,module in sorted(tuple(sys.modules.items())):
        file=getattr(module,'__file__',None)
        if not file:continue
        path=Path(file).resolve()
        if path.suffix=='.pyc' and path.with_suffix('.py').is_file():path=path.with_suffix('.py')
        if path.is_relative_to(root):
            if path.is_file():local.append(dict(module=name,path=path.relative_to(root).as_posix(),sha256=_digest(path)))
        elif any(path.is_relative_to(project) for project in project_roots):
            foreign.append(dict(module=name,file=str(path)))
    record=dict(status='passed' if status==0 and not foreign else 'failed',pytest_exit_code=int(status),
        test_counts=_junit(output/'tests.xml'),seconds=perf_counter()-began,
        root=str(root),cwd=os.getcwd(),python=sys.version,python_executable=sys.executable,
        libraries={name:distribution_version(name) for name in ('numpy','scipy','pytest')},
        high_precision_backend='Python standard-library Decimal; no mpmath dependency',
        pytest_configuration=str(configuration),pytest_configuration_sha256=_digest(configuration),
        local_imports=local,foreign_project_imports=foreign,
        every_project_import_from_declared_root=not foreign,
        environment={key:os.environ.get(key) for key in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE',
            'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    _write(output/'test_receipt.json',record)
    return 0 if record['status']=='passed' else 1


def run_tests(root,output,python):
    root=Path(root).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('use a new CPU test output directory: '+str(output))
    output.mkdir(parents=True)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root)+os.pathsep+str(root/'rnj_wzzt_core'))
    env.update({key:'1' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    command=[str(python),str(root/MODULE_REL/'followup_package.py'),'_collect_worker',
             '--root',str(root),'--output',str(output)]
    with (output/'stdout.log').open('w',encoding='utf-8') as stdout,(output/'stderr.log').open('w',encoding='utf-8') as stderr:
        result=subprocess.run(command,cwd=root,env=env,stdout=stdout,stderr=stderr,timeout=600)
    receipt=_read(output/'test_receipt.json')
    _write(output/'invocation.json',dict(command=command,returncode=result.returncode,
        original_scientific_worker_started=False,scientific_conditions_created=0))
    if result.returncode or receipt['status']!='passed':
        raise RuntimeError('CPU test/import-origin verification failed; retained evidence: '+str(output))
    return receipt


def dependency_closure(receipt,freeze):
    """Observed test imports plus the previously verified 55-file CPU subset."""
    names=set()
    previous=MODULE/'evidence/reference_source_subset.zip'
    with zipfile.ZipFile(previous) as archive:
        for name in archive.namelist():
            if name.endswith('/'):continue
            if name not in freeze['source_sha256']:
                raise RuntimeError('unsealed file in previous dependency subset: '+name)
            if sha256(archive.read(name)).hexdigest()!=freeze['source_sha256'][name]:
                raise RuntimeError('previous subset differs from original seal: '+name)
            names.add(name)
    previous_count=len(names)
    for row in receipt['local_imports']:
        name=row['path']
        if name.startswith(MODULE_REL.as_posix()+'/'):continue
        if name not in freeze['source_sha256']:
            raise RuntimeError('test imported unsealed non-current source: '+name)
        names.add(name)
    for name in names:
        path=STAGE/name
        if not path.is_file() or _digest(path)!=freeze['source_sha256'][name]:
            raise RuntimeError('development dependency differs from original frozen byte version: '+name)
    return sorted(names),dict(previous_subset_files=previous_count,dependency_files=len(names),
        observed_import_dependency_files=len({row['path'] for row in receipt['local_imports']
             if not row['path'].startswith(MODULE_REL.as_posix()+'/')}),
        all_dependency_bytes_match_original_freeze=True)


def prepare(output,python):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    freeze_path,freeze=assert_original_unchanged()
    tests=run_tests(STAGE,output/'stage_cpu_tests',python)
    dependencies,closure=dependency_closure(tests,freeze)
    record=dict(status='prepared_not_packaged',dependencies=dependencies,closure=closure,
        original_frozen_files_verified=len(freeze['source_sha256']),original_freeze_sha256=_digest(freeze_path),
        test_receipt='stage_cpu_tests/test_receipt.json',scientific_conditions_created=0)
    _write(output/'prepared.json',record)
    return record


def _completion_gates():
    results={}
    for name,count in (('candidate_h24_v1',32),('candidate_h24_polish_v1',32),('case33_crosswarm_v1',8)):
        path=STAGE/'outputs'/name/'completed.json';record=_read(path)
        if record.get('status')!='numerical_matrix_complete' or record.get('count')!=count:
            raise RuntimeError('follow-up matrix not complete: '+name)
        if len(record.get('outcomes',[]))!=count:
            raise RuntimeError('incomplete failure/success denominator: '+name)
        results[name]=dict(status=record['status'],count=count,completion_sha256=_digest(path))
    path=STAGE/'outputs/finite_ev_ac_audit_v1/completed.json';record=_read(path)
    if record.get('status')!='complete':raise RuntimeError('finite EV AC audit not complete')
    results['finite_ev_ac_audit_v1']=dict(status=record['status'],completion_sha256=_digest(path))
    path=STAGE/'outputs/case33_geometry_v1/completed.json';record=_read(path)
    if record.get('status')!='complete':raise RuntimeError('case33 coordinate geometry audit not complete')
    results['case33_geometry_v1']=dict(status=record['status'],completion_sha256=_digest(path))
    return results


def _member_map(dependencies,prepared_output,freeze_path):
    members={}
    def add(path,name):
        path=Path(path)
        if not path.is_file():raise FileNotFoundError(path)
        if name in members:raise ValueError('duplicate archive name: '+name)
        members[name]=path
    for path in sorted(MODULE.iterdir()):
        if path.is_file() and (path.suffix in ('.py','.md','.json') or path.name=='.gitattributes'):
            add(path,'source/'+MODULE_REL.as_posix()+'/'+path.name)
    for name in dependencies:add(STAGE/name,'source/'+name)
    add(freeze_path,'dependencies/original_freeze.json')
    for name in ('history_only_v1.json.gz','candidate_history_v1.json.gz','candidate_h24_polish_plan_v1.json'):
        add(STAGE/'outputs'/name,'data/'+name)
    # Two small saved inputs are required by the 50/60-digit diagnostic. They
    # are not the old broad numerical evidence or its GPU artifacts.
    for name in ('reference_jac_precision_0.json','block_jac_precision_0.json'):
        add(STAGE/'outputs/cpu_precision_v1'/name,'replay_inputs/cpu_precision_v1/'+name)
    for name in EVIDENCE_ROOTS:
        folder=STAGE/'outputs'/name
        for path in sorted(folder.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo'):
                add(path,'evidence/'+name+'/'+path.relative_to(folder).as_posix())
    # Retain the first packaging attempt that caught ancestor pytest config
    # selecting imports from the primary checkout; it is a setup failure,
    # separately named and never counted as a scientific result.
    previous_failure=STAGE/'outputs/followup_prepare_v1'
    if previous_failure.is_dir():
        for path in sorted(previous_failure.rglob('*')):
            if path.is_file() and 'pytest_temp' not in path.parts:
                add(path,'evidence/packaging_import_origin_failure/'+path.relative_to(previous_failure).as_posix())
    for path in sorted(prepared_output.rglob('*')):
        if path.is_file() and 'pytest_temp' not in path.parts:
            add(path,'verification/preparation/'+path.relative_to(prepared_output).as_posix())
    return members


def _zip(target,members):
    manifest={name:dict(sha256=_digest(path),bytes=path.stat().st_size) for name,path in sorted(members.items())}
    temporary=target.with_name(target.name+'.writing')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,path in sorted(members.items()):archive.write(path,name)
        archive.writestr('manifest.json',json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    temporary.replace(target)
    verify_archive(target,manifest)
    return manifest


def verify_archive(path,expected=None):
    with zipfile.ZipFile(path) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        if expected is not None and manifest!=expected:raise RuntimeError('manifest content changed')
        if set(archive.namelist())!=set(manifest)|{'manifest.json'}:
            raise RuntimeError('archive has unmanifested, missing or duplicate members')
        if len(archive.namelist())!=len(manifest)+1:raise RuntimeError('duplicate ZIP members')
        for name,entry in manifest.items():
            content=archive.read(name)
            if len(content)!=entry['bytes'] or sha256(content).hexdigest()!=entry['sha256']:
                raise RuntimeError('archive member hash/size mismatch: '+name)
    return manifest


def package(output,python,finalize=False):
    if not finalize:raise ValueError('final package requires explicit --finalize after report/checkpoint are finished')
    began=perf_counter();output=Path(output).resolve()
    old_evidence=MODULE/'evidence/numerical_evidence_v1.zip'
    old_evidence_digest=_digest(old_evidence)
    if output.exists() and any(output.iterdir()):raise FileExistsError('use an empty output directory: '+str(output))
    output.mkdir(parents=True,exist_ok=True)
    gates=_completion_gates()
    # Keep execution/test paths short on Windows; the authoritative archive
    # and receipts may live inside the module's evidence directory.
    work=STAGE/'outputs/followup_pack_work_v1'
    if work.exists():raise FileExistsError('do not overwrite a previous packaging work directory: '+str(work))
    work.mkdir(parents=True)
    preparation=prepare(work/'preparation',python)
    freeze_path,freeze=assert_original_unchanged()
    members=_member_map(preparation['dependencies'],work/'preparation',freeze_path)
    target=output/'numerical_followup_evidence_v1.zip'
    before=_zip(target,members)
    restore=work/'independent_restore'
    if restore.exists():raise FileExistsError(restore)
    restore.mkdir()
    with zipfile.ZipFile(target) as archive:
        for name in archive.namelist():
            destination=(restore/name).resolve()
            if not destination.is_relative_to(restore.resolve()):raise RuntimeError('unsafe ZIP member: '+name)
        archive.extractall(restore)
    for name,entry in before.items():
        if _digest(restore/name)!=entry['sha256']:raise RuntimeError('restored member hash mismatch: '+name)
    tests=run_tests(restore/'source',work/'restored_cpu_tests',python)
    # The replay imports must come from the extracted tree. Their byte hashes
    # are also checked against the source members that will enter the final ZIP.
    for row in tests['local_imports']:
        member='source/'+row['path']
        if member not in before or before[member]['sha256']!=row['sha256']:
            raise RuntimeError('restored CPU import differs from packaged source: '+member)
    for path in sorted((work/'restored_cpu_tests').iterdir()):
        if path.is_file():members['verification/restored_cpu_tests/'+path.name]=path
    source_before={name:entry for name,entry in before.items() if name.startswith('source/')}
    final=_zip(target,members)
    source_after={name:entry for name,entry in final.items() if name.startswith('source/')}
    if source_before!=source_after:raise RuntimeError('source changed after restored CPU tests')
    assert_original_unchanged()
    if _digest(old_evidence)!=old_evidence_digest:
        raise RuntimeError('old numerical_evidence_v1.zip changed during packaging')
    receipt=dict(status='verified_followup_package_complete',created_utc=datetime.now(timezone.utc).isoformat(),
        archive=target.name,sha256=_digest(target),bytes=target.stat().st_size,members=len(final),
        all_members_reopened_and_hash_verified=True,independent_restore_root=str(restore/'source'),
        all_restored_members_hash_verified=True,restored_imports_match_packaged_sources=True,
        restored_source_equals_final_archive=True,restored_CPU_test_counts=tests['test_counts'],
        restored_CPU_project_imports=len(tests['local_imports']),
        every_restored_project_import_from_restore_root=tests['every_project_import_from_declared_root'],
        original_frozen_files_unchanged=len(freeze['source_sha256']),original_policy=freeze['identity'],
        dependency_closure=preparation['closure'],completion_gates=gates,
        old_numerical_evidence_v1_modified=False,old_evidence_sha256=old_evidence_digest,GPU_binaries_included=False,
        scientific_conditions_created=0,seconds=perf_counter()-began)
    _write(output/'package_receipt.json',receipt)
    _write(output/'replay_receipt.json',dict(status='independent_CPU_restore_verified',
        archive=target.name,archive_sha256=receipt['sha256'],
        restored_root=str(restore/'source'),test_counts=tests['test_counts'],
        project_imports_verified=len(tests['local_imports']),
        every_project_import_from_restore_root=tests['every_project_import_from_declared_root'],
        all_members_hash_verified=True,restored_source_identical_to_final_archive=True,
        original_frozen_scientific_sources_unchanged=receipt['original_frozen_files_unchanged'],
        test_receipt_archive_member='verification/restored_cpu_tests/test_receipt.json',
        scientific_samples_created=0))
    print(json.dumps(receipt,ensure_ascii=False))
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    collect=sub.add_parser('_collect_worker');collect.add_argument('--root',type=Path,required=True);collect.add_argument('--output',type=Path,required=True)
    for name in ('prepare','package'):
        item=sub.add_parser(name);item.add_argument('--output',type=Path,required=True)
        item.add_argument('--python',type=Path,default=Path(sys.executable))
        if name=='package':item.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    if args.command=='_collect_worker':return collect_worker(args)
    if args.command=='prepare':print(json.dumps(prepare(args.output,args.python)));return 0
    package(args.output,args.python,args.finalize);return 0


if __name__=='__main__':raise SystemExit(main())
