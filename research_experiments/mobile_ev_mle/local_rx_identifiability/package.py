"""Package compact evidence and actually restore/replay the first paired slots."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from .experiment import read, write, task_key


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package(directory, destination, restore):
    directory, destination, restore = (Path(p).resolve() for p in (directory, destination, restore))
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('evidence destination must be empty; never replace old evidence')
    if restore.exists():
        raise FileExistsError('restore directory must be new')
    audit = read(directory/'audit.json')
    if not audit['audit_pass']:
        raise ValueError('independent main audit has not passed')
    relative=read(directory/'relative_bootstrap_summary.json')
    if relative['completed_slots'] != 16:
        raise ValueError('relative bootstrap is incomplete')
    if not read(directory/'relative_audit.json')['audit_pass']:
        raise ValueError('relative bootstrap independent audit has not passed')
    source=Path(__file__).resolve().parents[3]
    frozen=read(directory/'source_manifest.json')
    for name,digest in frozen.items():
        if sha(source/name)!=digest:
            raise ValueError('frozen scientific source changed: '+name)
    destination.mkdir(parents=True,exist_ok=True)
    copies=('configuration.json','completed.json','source_manifest.json','summary.json',
            'candidate_details.csv','audit.json','bootstrap_summary.json',
            'relative_bootstrap_summary.json','relative_audit.json','results_generated.md',
            'unit_tests.xml')
    for name in copies: shutil.copyfile(directory/name,destination/name)
    module=Path(__file__).resolve().parent
    sources=set(frozen)
    sources.update(str(p.relative_to(source)).replace('\\','/') for p in module.glob('*.py'))
    sources.update(str(p.relative_to(source)).replace('\\','/') for p in module.glob('*.md'))
    with zipfile.ZipFile(destination/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(sources): z.write(source/name,name)
    config=read(directory/'configuration.json')
    selected=[s for s in config['tasks'] if s['seed']==31001 and
              (s['N']==96 or (s['network']=='paper15' and s['regime']=='correlated'))]
    with zipfile.ZipFile(destination/'representative_records.zip','w',zipfile.ZIP_DEFLATED) as z:
        for spec in selected:
            for folder in ('inference','results','truth'):
                p=directory/folder/(task_key(spec)+'.json.gz')
                z.write(p,str(p.relative_to(directory)).replace('\\','/'))
        for folder in ('bootstrap','relative_bootstrap'):
            for p in sorted((directory/folder).glob('*.json')):
                z.write(p,str(p.relative_to(directory)).replace('\\','/'))
    files={str(p.relative_to(directory)).replace('\\','/'):dict(sha256=sha(p),bytes=p.stat().st_size)
           for p in directory.rglob('*') if p.is_file()}
    write(destination/'full_local_manifest.json',dict(directory=str(directory),files=files,
          raw_records_are_retained_locally=True,representative_specifications=selected))
    # No deletion or overwrite: restore into a brand-new directory.
    restore.mkdir(parents=True)
    restored_config=dict(config,tasks=config['tasks'][:2])
    replay_keys={task_key(spec)+'.json.gz' for spec in restored_config['tasks']}
    for archive,sub in (('source_snapshot.zip','source'),('representative_records.zip','records')):
        target=restore/sub;target.mkdir()
        with zipfile.ZipFile(destination/archive) as z:
            for name in z.namelist():
                if not (target/name).resolve().is_relative_to(target): raise ValueError('unsafe archive path')
            # The archive retains ten full conditions. This replay selects the
            # first paired K/U conditions; do not feed eight unconfigured slots
            # to a strict two-slot audit.
            for name in z.namelist():
                parts=Path(name).parts
                if archive=='representative_records.zip' and parts[0] in ('inference','results','truth'):
                    if parts[-1] not in replay_keys: continue
                z.extract(name,target)
    write(restore/'records/configuration.json',restored_config)
    # The original outputs for these two slots have exactly the same bytes.
    for spec in restored_config['tasks']:
        for folder in ('inference','results','truth'):
            name=f'{folder}/{task_key(spec)}.json.gz'
            if sha(restore/'records'/name)!=files[name]['sha256']: raise AssertionError('restore hash mismatch')
    command=[sys.executable,'-m','research_experiments.mobile_ev_mle.local_rx_identifiability.audit',
             '--directory',str(restore/'records')]
    result=subprocess.run(command,cwd=restore/'source',capture_output=True,text=True)
    (destination/'restored_stdout.log').write_text(result.stdout,encoding='utf-8')
    (destination/'restored_stderr.log').write_text(result.stderr,encoding='utf-8')
    if result.returncode: raise RuntimeError('restored independent audit failed; retained logs')
    restored_audit=read(restore/'records/audit.json')
    if not restored_audit['audit_pass']: raise AssertionError('restored evidence audit did not pass')
    write(destination/'restore_receipt.json',dict(passed=True,
        restored_slots=restored_audit['configured_slots'],
        restored_base_condition_trajectories=restored_audit['base_condition_trajectories'],
        restored_independent_random_families=restored_audit['independent_random_families'],
        restored_independent_audit=restored_audit,
        snapshot_sources=len(sources),frozen_scientific_sources=len(frozen),
        full_local_files=len(files),selected_slots=len(selected),
        command=command,exit_code=result.returncode,
        archived_data_scope='synthetic observations and simulation truth; no field/private dataset'))
    manifest={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in destination.iterdir() if p.is_file()}
    write(destination/'manifest.json',manifest)
    return dict(passed=True,selected_slots=len(selected),frozen_sources=len(frozen),
                total_bytes=sum(p.stat().st_size for p in destination.iterdir() if p.is_file()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True)
    parser.add_argument('--destination',required=True);parser.add_argument('--restore',required=True)
    args=parser.parse_args();print(json.dumps(package(args.directory,args.destination,args.restore)))


if __name__=='__main__':main()
