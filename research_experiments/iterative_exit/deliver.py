"""Archive verified implementation evidence without rerunning optimization."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib
from .run import ROOT,CHANGED,read,write,sha


def deliver(output):
    output=Path(output);delivery=ROOT/'docs'/output.name;p=read(output/'protocol.json');audit=read(delivery/'smoke_verification.json')
    assert audit['passed'] and audit['returned']==4 and read(delivery/'tests.json')['passed']
    assert sha(Path(__file__).with_name('run.py'))==audit['analyzer_sha256']
    manifest=p['parent_manifest'];assert sha(manifest['path'])==manifest['sha256']
    for name,record in manifest['files'].items():assert sha(Path(manifest['path']).parent/name)==record['sha256']
    archive=delivery/'records.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for folder in ('source','parents','jobs','analysis'):
            for f in (output/folder).rglob('*'):
                if f.is_file() and f.suffix in ('.py','.json','.csv','.log'):z.write(f,f.relative_to(output))
        for name in ('protocol.json','catalog.json','completion.json'):z.write(output/name,name)
        current=[ROOT/name for name in CHANGED]+list(Path(__file__).parent.glob('*.py'))
        current.extend(ROOT/'research_experiments/tests'/name for name in ('test_iterative_exit.py','test_joint_five.py'))
        for f in current:z.write(f,'current_source/'+f.relative_to(ROOT).as_posix())
        for name in ('DESIGN.md','README.md','tests.json','smoke_verification.json','smoke_per_model.csv'):
            z.write(delivery/name,'reports/'+name)
    with ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in p['source_hashes'].items():assert hashlib.sha256(z.read('source/'+name)).hexdigest()==digest
    write(delivery/'delivery_audit.json',dict(passed=True,archive_sha256=sha(archive),returned=4,
        tests=97,independent_LPs=audit['independent_LPs'],historical_files_preserved=len(manifest['files']),
        scope='implementation/invariant verification; not a 120-input recovery benchmark'))
    write(delivery/'manifest.json',dict(files={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(delivery.iterdir()) if f.is_file() and f.name!='manifest.json'}))
    write(output/'state.json',dict(stage='complete',returned=4,verified=True))
    print('Integrated iteration policy implementation archived and verified')


if __name__=='__main__':
    import sys
    deliver(Path(sys.argv[1]).resolve())
