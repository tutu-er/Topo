"""Compact reproducible evidence and actual restored-source replay."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from ..local_rx_identifiability.experiment import read,write
from ..local_rx_three_day.experiment import key
from .experiment import sha


def build(directory,destination,restore):
    directory,destination,restore=(Path(p).resolve() for p in (directory,destination,restore))
    if destination.exists() and any(destination.iterdir()):raise FileExistsError('new evidence destination required')
    if restore.exists():raise FileExistsError('new restoration destination required')
    root=Path(__file__).resolve().parents[3]; module=Path(__file__).parent
    config=read(directory/'configuration.json')
    if not read(directory/'audit.json')['audit_pass']:raise ValueError('input/physics audit not passed')
    for name,digest in config['source_manifest'].items():
        if sha(root/name)!=digest:raise ValueError('frozen science changed: '+name)
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('configuration.json','completed.json','summary.json','audit.json',
                 'benchmark.json','benchmark_configuration.json','unit_tests.xml'):
        shutil.copyfile(directory/name,destination/name)
    if (directory/'physics_independent_audit.json').exists():
        shutil.copyfile(directory/'physics_independent_audit.json',destination/'physics_independent_audit.json')
    selected=[s for s in config['tasks'] if s['seed']==33001]
    names=set(config['source_manifest'])
    names.update(str(p.relative_to(root)).replace('\\','/') for p in module.rglob('*')
                 if p.is_file() and p.suffix in ('.py','.md') and 'evidence' not in p.parts)
    # The independent auditor imports the original case generator and AC
    # likelihood dependencies already frozen in source_manifest.
    with zipfile.ZipFile(destination/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):archive.write(root/name,name)
    baseline=Path(config['baseline'])
    with zipfile.ZipFile(destination/'representative_records.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for spec in selected:
            name=key(spec)+'.json.gz'
            for folder in ('inputs','records'):
                archive.write(directory/folder/name,f'{folder}/{name}')
            for folder in ('inference','truth'):
                archive.write(baseline/folder/name,f'baseline/{folder}/{name}')
    physics=read(directory/'physics.json')
    compact=dict(source_manifest=physics['source_manifest'],seconds=physics['seconds'],
                 rejection_stage=physics['rejection_stage'],records=[])
    for record in physics['records']:
        item={k:v for k,v in record.items() if k!='candidates'}
        item['candidates']={}
        for name,c in record.get('candidates',{}).items():
            row={k:v for k,v in c.items() if k not in ('public','meter','fixed_lines_only','envelopes')}
            row['envelopes']=[]
            for envelope in c['envelopes']:
                qp=envelope['qp'];rule=envelope['rule']
                row['envelopes'].append(dict(alpha=envelope['alpha'],
                    qp={k:v for k,v in qp.items() if k not in ('delta','delta_lower','delta_upper','covariance')},
                    rule={k:v for k,v in rule.items() if k!='square_noise'},
                    square_noise={k:v for k,v in rule.get('square_noise',{}).items() if k!='per_time_squared_bound'}))
            item['candidates'][name]=row
        compact['records'].append(item)
    write(destination/'physics_summary.json',compact)
    manifest={str(p.relative_to(directory)).replace('\\','/'):dict(sha256=sha(p),bytes=p.stat().st_size)
              for p in directory.rglob('*') if p.is_file()}
    write(destination/'full_local_manifest.json',dict(directory=str(directory),files=manifest,
        representative_conditions=selected,original_files_retained=True,
        initial_physics_center_alignment_error_retained='physics_center_alignment_v1.json' in manifest))
    restore.mkdir(parents=True)
    for filename,target_name in (('source_snapshot.zip','source'),('representative_records.zip','records')):
        target=restore/target_name;target.mkdir()
        with zipfile.ZipFile(destination/filename) as archive:
            for name in archive.namelist():
                if not (target/name).resolve().is_relative_to(target):raise ValueError('unsafe archive path')
                archive.extract(name,target)
    # Replay two predeclared paper15 3/7-day states, keep the other archived
    # network as additional evidence, not as an unperformed replay.
    replay_specs=[s for s in selected if s['network']=='paper15']
    replay_config=dict(config,tasks=replay_specs,baseline=str(restore/'records/baseline'))
    write(restore/'records/configuration.json',replay_config)
    for spec in selected:
        for folder in ('inputs','records'):
            filename=folder+'/'+key(spec)+'.json.gz'
            if sha(restore/'records'/filename)!=manifest[filename]['sha256']:
                raise AssertionError('restored byte hash mismatch')
    command=[sys.executable,'-m','research_experiments.mobile_ev_mle.local_response_score.audit',
             '--directory',str(restore/'records')]
    result=subprocess.run(command,cwd=restore/'source',capture_output=True,text=True)
    (destination/'restored_stdout.log').write_text(result.stdout,encoding='utf-8')
    (destination/'restored_stderr.log').write_text(result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('archived-source replay failed, logs retained')
    audit=read(restore/'records/audit.json')
    write(destination/'restore_receipt.json',dict(passed=audit['audit_pass'],
        restored_conditions=len(replay_specs),archived_conditions=len(selected),
        snapshot_sources=len(names),frozen_scientific_sources=len(config['source_manifest']),
        independent_audit=audit,command=command,
        scope='saved-state arithmetic/physics audit from restored sources, not a 16-condition refit'))
    write(destination/'manifest.json',{p.name:dict(sha256=sha(p),bytes=p.stat().st_size)
                                     for p in destination.iterdir() if p.is_file()})
    return dict(passed=audit['audit_pass'],archived_conditions=len(selected),
                restored_conditions=len(replay_specs),bytes=sum(p.stat().st_size for p in destination.iterdir()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True)
    parser.add_argument('--destination',required=True);parser.add_argument('--restore',required=True)
    args=parser.parse_args();print(build(args.directory,args.destination,args.restore),flush=True)


if __name__=='__main__':main()
