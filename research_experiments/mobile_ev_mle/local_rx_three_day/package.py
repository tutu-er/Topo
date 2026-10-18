"""Compact evidence with an actual two-condition archived-source restoration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def key(s):return f"{s['network']}_{s['seed']}_N{s['N']}_{s['boundary']}"


def package(directory,destination,restore):
    directory,destination,restore=(Path(p).resolve() for p in (directory,destination,restore))
    if destination.exists() and any(destination.iterdir()):raise FileExistsError('new evidence directory required')
    if restore.exists():raise FileExistsError('new restoration directory required')
    if not read(directory/'audit.json')['audit_pass']:raise ValueError('audit not passed')
    source=Path(__file__).resolve().parents[3];module=Path(__file__).resolve().parent
    config=read(directory/'configuration.json')
    for name,digest in config['source_manifest'].items():
        if sha(source/name)!=digest:raise ValueError('frozen source changed: '+name)
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('configuration.json','completed.json','summary.json','candidate_details.csv',
                 'results_generated.md','audit.json','unit_tests.xml','new_unit_tests.xml','test_summary.json',
                 'local_gradient_scale.json'):
        shutil.copyfile(directory/name,destination/name)
    source_names=set(config['source_manifest'])
    # Independent replay depends on the prior saved-state auditor, which is
    # not imported by the new scientific generator/solver.
    source_names.add('research_experiments/mobile_ev_mle/local_rx_identifiability/audit.py')
    source_names.update(str(p.relative_to(source)).replace('\\','/')
                        for pattern in ('*.py','*.md') for p in module.glob(pattern))
    with zipfile.ZipFile(destination/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(source_names):z.write(source/name,name)
    selected=[s for s in config['tasks'] if s['seed']==33001]
    with zipfile.ZipFile(destination/'representative_records.zip','w',zipfile.ZIP_DEFLATED) as z:
        for spec in selected:
            for folder in ('inference','results','truth'):
                p=directory/folder/(key(spec)+'.json.gz')
                z.write(p,str(p.relative_to(directory)).replace('\\','/'))
    manifest={str(p.relative_to(directory)).replace('\\','/'):dict(sha256=sha(p),bytes=p.stat().st_size)
              for p in directory.rglob('*') if p.is_file()}
    write(destination/'full_local_manifest.json',dict(directory=str(directory),files=manifest,
         original_records_retained_locally=True,representative_conditions=selected))
    restore.mkdir(parents=True)
    replay_specs=config['tasks'][:2];replay_names={key(s)+'.json.gz' for s in replay_specs}
    for archive,sub in (('source_snapshot.zip','source'),('representative_records.zip','records')):
        target=restore/sub;target.mkdir()
        with zipfile.ZipFile(destination/archive) as z:
            for name in z.namelist():
                if not (target/name).resolve().is_relative_to(target):raise ValueError('unsafe archive entry')
                if sub=='records' and Path(name).name not in replay_names:continue
                z.extract(name,target)
    restored_config=dict(config,tasks=replay_specs,representative_replay=True,independent_random_families=1)
    write(restore/'records/configuration.json',restored_config)
    for spec in replay_specs:
        for folder in ('inference','results','truth'):
            name=f'{folder}/{key(spec)}.json.gz'
            if sha(restore/'records'/name)!=manifest[name]['sha256']:raise AssertionError('restored record hash')
    command=[sys.executable,'-m','research_experiments.mobile_ev_mle.local_rx_three_day.audit',
             '--directory',str(restore/'records')]
    result=subprocess.run(command,cwd=restore/'source',capture_output=True,text=True)
    (destination/'restored_stdout.log').write_text(result.stdout,encoding='utf-8')
    (destination/'restored_stderr.log').write_text(result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('restored audit failed; logs retained')
    restored_audit=read(restore/'records/audit.json')
    if not restored_audit['audit_pass']:raise AssertionError('restored audit not passed')
    write(destination/'restore_receipt.json',dict(passed=True,restored_conditions=len(replay_specs),
         representative_archive_conditions=len(selected),scientific_sources=len(config['source_manifest']),
         total_snapshot_sources=len(source_names),independent_audit=restored_audit,
         command=command,scope='two U/R states replayed, not 48-condition refitting',
         data='synthetic benchmark observations and simulation truth; no field data'))
    write(destination/'manifest.json',{p.name:dict(sha256=sha(p),bytes=p.stat().st_size)
          for p in destination.iterdir() if p.is_file()})
    return dict(passed=True,restored_conditions=2,archived_conditions=len(selected),
                total_bytes=sum(p.stat().st_size for p in destination.iterdir() if p.is_file()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True)
    parser.add_argument('--destination',required=True);parser.add_argument('--restore',required=True)
    a=parser.parse_args();print(json.dumps(package(a.directory,a.destination,a.restore)))


if __name__=='__main__':main()
