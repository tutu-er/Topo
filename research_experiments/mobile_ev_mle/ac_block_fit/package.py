"""Package this numerical study, without GPU binaries or scientific reruns."""
from pathlib import Path
from hashlib import sha256
import argparse,json,zipfile,importlib,sys


STAGE=Path(__file__).resolve().parents[3]
MODULE=Path(__file__).parent


def package(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    members={}
    def add(path,name):
        path=Path(path)
        if not path.is_file():raise FileNotFoundError(path)
        if name in members:raise ValueError('duplicate archive member: '+name)
        members[name]=path
    for phase in ('cpu_screen_v1','cpu_compare_v2','cpu_tolerance_1e10','cpu_tolerance_1e8'):
        root=STAGE/'outputs'/phase
        complete=json.loads((root/'completed.json').read_bytes())
        if complete.get('status')!='numerical_matrix_complete':raise RuntimeError('matrix unfinished: '+phase)
        for name in ('plan.json','completed.json','summary.json'):
            add(root/name,phase+'/'+name)
        for path in root.glob('source_*'):
            if path.is_file():add(path,phase+'/'+path.name)
        for path in root.glob('tasks/*/*'):
            if path.name in ('task.json','result.json','started.json','stdout.log','stderr.log'):
                add(path,phase+'/'+path.relative_to(root).as_posix())
    gpu=STAGE/'outputs/gpu_fit_v1'
    final=json.loads((gpu/'summary.json').read_bytes())
    if final.get('status')!='complete':raise RuntimeError('GPU fit comparison unfinished')
    for path in gpu.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.log'):
            add(path,'gpu_fit_v1/'+path.relative_to(gpu).as_posix())
    for root_name in ('cpu_precision_v1','gpu_arithmetic_source'):
        root=STAGE/'outputs'/root_name
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.log','.py'):
                add(path,root_name+'/'+path.relative_to(root).as_posix())
    for pattern in ('gpu_benchmark*.json','gpu_test_receipt.json','batched_source_h192_v*.json'):
        for path in (STAGE/'outputs').glob(pattern):add(path,'development_history/'+path.name)
    for name in ('history_only_v1.json.gz','response_endpoint_audit.json','numerical_final_audit.json',
                 'gpu_requirements.txt'):
        add(STAGE/'outputs'/name,name)
    source=Path(r'D:/0-github_workspace/Topo/outputs/ev_reference_stage_20261018')
    frozen=json.loads((source/'outputs/reference_v1/freeze.json').read_bytes())
    changed=[name for name,h in frozen['source_sha256'].items()
             if sha256((source/name).read_bytes()).hexdigest()!=h]
    if changed:raise RuntimeError('original frozen scientific sources changed: '+str(changed))
    add(source/'outputs/reference_v1/freeze.json','dependencies/original_freeze.json')
    # Package only the imported dependency closure relevant to these solvers
    # and their tests, rather than unrelated old experiments in the full seal.
    for path in MODULE.glob('*.py'):
        if path.name!='__init__.py':
            importlib.import_module('research_experiments.mobile_ev_mle.ac_block_fit.'+path.stem)
    dependencies={}
    for loaded in tuple(sys.modules.values()):
        file=getattr(loaded,'__file__',None)
        if not file:continue
        path=Path(file).resolve()
        if not path.is_relative_to(STAGE) or path.is_relative_to(MODULE):continue
        name=path.relative_to(STAGE).as_posix()
        if name not in frozen['source_sha256']:continue
        if sha256(path.read_bytes()).hexdigest()!=frozen['source_sha256'][name]:
            raise RuntimeError('development dependency differs from sealed version: '+name)
        dependencies[name]=path
    subset=output/'reference_source_subset.zip'
    with zipfile.ZipFile(subset,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,path in sorted(dependencies.items()):archive.write(path,name)
    add(subset,'dependencies/reference_source_subset.zip')
    for path in MODULE.glob('*.py'):add(path,'current_source/'+path.name)
    for name in ('README.md','math.md','protocol.md','REPORT.md','REPRODUCE.md'):add(MODULE/name,'current_source/'+name)
    manifest={name:dict(sha256=sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
              for name,path in sorted(members.items())}
    target=output/'numerical_evidence_v1.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,path in sorted(members.items()):archive.write(path,name)
        archive.writestr('manifest.json',json.dumps(manifest,sort_keys=True,indent=2))
    with zipfile.ZipFile(target) as archive:
        for name,record in manifest.items():
            if sha256(archive.read(name)).hexdigest()!=record['sha256']:
                raise RuntimeError('archive replay hash mismatch: '+name)
    receipt=dict(archive=target.name,sha256=sha256(target.read_bytes()).hexdigest(),
        bytes=target.stat().st_size,members=len(manifest),all_members_reopened_and_hash_verified=True,
        original_frozen_scientific_files_unchanged=len(frozen['source_sha256']),
        reference_source_subset_files=len(dependencies),
        original_policy=frozen['identity'],GPU_binaries_included=False,
        scientific_samples_or_calibration_created=0)
    (output/'package_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt))
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=MODULE/'evidence')
    package(parser.parse_args().output)


if __name__=='__main__':main()
