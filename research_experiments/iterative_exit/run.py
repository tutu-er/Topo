"""Frozen four-case entrypoint check for the integrated iteration policy."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter,sleep
import traceback
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from research_experiments.rebuild_benchmark.run import ROOT,read,write,sha

PARENT=ROOT/'outputs/rnj_four_methods_20261031'
SNAPSHOT=ROOT/'outputs/flexible_exit_20261101/source'
DELIVERY=ROOT/'docs/iterative_zero_exit_20261101'
CHANGED=('research_experiments/joint_five/search.py','research_experiments/rnj_count/engine.py',
    'research_experiments/rnj_small_seed/engine.py','research_experiments/flexible_exit/model.py')
LIMITS=dict(seconds=14400.,prefix_seconds=7200.,per_mip=3600.,lp_seconds=300.,post_reserve=600.)


def freeze(output):
    if output.exists():raise FileExistsError('New output directory required.')
    parent=read(PARENT/'protocol.json');catalog=read(PARENT/'catalog.json')
    jobs=[j for j in parent['jobs'] if catalog[j]['panel']=='main' and catalog[j]['bundle']==2 and catalog[j]['N']==48 and catalog[j]['voltage_sigma']==.0002]
    assert len(jobs)==4
    output.mkdir(parents=True);shutil.copytree(SNAPSHOT,output/'source')
    files=[ROOT/name for name in CHANGED]+list(Path(__file__).parent.glob('*.py'))
    for file in files:
        dest=output/'source'/file.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
    refs={}
    for job in jobs:
        directory=output/'parents'/job;directory.mkdir(parents=True)
        common=PARENT/'common'/job/'result.json';old=PARENT/'evaluation'/job/'small3_original_exit/result.json'
        for file,name in ((common,'common.json'),(old,'historical.json')):shutil.copy2(file,directory/name)
        refs[job]={str(common):sha(common),str(old):sha(old)}
    write(output/'catalog.json',{j:catalog[j] for j in jobs})
    manifest=ROOT/'docs/rnj_four_methods_20261031/manifest.json';m=read(manifest)
    for name,record in m['files'].items():assert sha(manifest.parent/name)==record['sha256']
    write(output/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,limits=LIMITS,
        scope='entrypoint/invariant smoke check: four cases, bundle2, N48, voltage sigma .0002; not the 120-input performance benchmark',
        policy='immediate provisional-zero release; one training-ranked deletion only on stalled prefix; unmasked/masked continuations; validation after full training',
        hard_priors=[],zero_tolerance=1e-9,zero_loss_tolerance=1e-10,coefficient=.003,resolution=.24,
        input_hashes={j:parent['input_hashes'][j] for j in jobs},references=refs,
        parent_manifest=dict(path=str(manifest),sha256=sha(manifest),files=m['files']),
        source_hashes={f.relative_to(output/'source').as_posix():sha(f) for f in (output/'source').rglob('*.py')},
        truth_used_in_training=False,test_opened_by_worker=False,retuning=False))
    write(output/'state.json',dict(stage='frozen',planned=4))
    delivery=ROOT/'docs'/output.name;delivery.mkdir(parents=True,exist_ok=True)
    if (delivery/'protocol.json').exists():raise FileExistsError('Do not overwrite an existing delivered protocol.')
    shutil.copy2(output/'protocol.json',delivery/'protocol.json')


def worker(payload):
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.rebuild_benchmark.engine import prepare
    from .engine import fit_mainline
    p=read(payload);then=perf_counter();target=Path(p['result']);common=read(p['common']);old=read(p['historical'])
    for key in ('common','historical'):assert sha(p[key])==p[key+'_sha256']
    assert {s:sha(p[s]) for s in ('train','validation')}==p['input_hashes']
    data=prepare(observed(p['train'],p['N']));validation=prepare(observed(p['validation']))
    limits=dict(p['limits']);limits['seconds']-=common['seconds']+perf_counter()-then
    result=fit_mainline(data,validation,common,reference=old['trajectory']['reference_MAE'],**limits,
        checkpoint=lambda row:write(target.with_name('checkpoint.json'),row),
        heartbeat=lambda row:write(target.with_name('progress.json'),row))
    result.update(job=p['job'],protocol_sha256=p['protocol_sha256'],input_hashes=p['input_hashes'],
        fresh_seconds=perf_counter()-then,common_recorded_seconds=common['seconds'])
    result['cumulative_seconds']=result['fresh_seconds']+common['seconds']
    assert result['cumulative_seconds']<=p['limits']['seconds']+1.
    write(target,result)


def run(output):
    protocol=read(output/'protocol.json');catalog=read(output/'catalog.json');active={}
    for name,digest in protocol['source_hashes'].items():assert sha(output/'source'/name)==digest
    env=os.environ.copy();env['PYTHONPATH']=str(output/'source')+os.pathsep+str(output/'source/rnj_wzzt_core')
    for job in protocol['jobs']:
        target=output/'jobs'/job/'result.json'
        if target.exists():continue
        meta=catalog[job];directory=output/'parents'/job
        p=dict(job=job,train=meta['train'],validation=meta['validation'],N=meta['N'],
            common=str(directory/'common.json'),historical=str(directory/'historical.json'),
            common_sha256=sha(directory/'common.json'),historical_sha256=sha(directory/'historical.json'),
            input_hashes={s:protocol['input_hashes'][job][s] for s in ('train','validation')},
            result=str(target),limits=protocol['limits'],protocol_sha256=sha(output/'protocol.json'))
        payload=target.with_name('payload.json');write(payload,p);log=target.with_name('worker.log').open('w',encoding='utf-8')
        proc=subprocess.Popen([sys.executable,'-B','-m','research_experiments.iterative_exit.run','worker',str(payload)],
            cwd=output/'source',env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        active[job]=proc,log,target
    while active:
        for job,(proc,log,target) in list(active.items()):
            if proc.poll() is not None:
                log.close();del active[job]
                if not target.exists() and not target.with_name('error.json').exists():write(target.with_name('error.json'),dict(exit_code=proc.returncode))
                print('finished',job,target.exists(),flush=True)
        if active:sleep(.5)
    failures=[j for j in protocol['jobs'] if not (output/'jobs'/j/'result.json').exists()]
    write(output/'completion.json',dict(planned=4,returned=4-len(failures),failures=failures))


def analyze(output):
    import pandas as pd
    from research_experiments.local_rebuild.model import canonical
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.rebuild_benchmark.engine import prepare
    from research_experiments.rebuild_benchmark.analyze import score
    from research_experiments.joint_exit.analyze import independent_profile
    from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
    p=read(output/'protocol.json');catalog=read(output/'catalog.json');rows=[];checks=0;profiles=[]
    delivery=ROOT/'docs'/output.name;delivery.mkdir(parents=True,exist_ok=True)
    complete=read(output/'completion.json');assert complete['returned']==4 and not complete['failures']
    for name,digest in p['source_hashes'].items():assert sha(output/'source'/name)==digest
    m=p['parent_manifest'];assert sha(m['path'])==m['sha256']
    for name,record in m['files'].items():assert sha(Path(m['path']).parent/name)==record['sha256']
    for job in p['jobs']:
        meta=catalog[job];r=read(output/'jobs'/job/'result.json');common=read(output/'parents'/job/'common.json')
        old=read(output/'parents'/job/'historical.json');labels=common['labels']
        assert r['protocol_sha256']==sha(output/'protocol.json') and r['recovery_attempts']<=1 and not r['LP_failures']
        assert not r['truth_used'] and not r['test_opened'] and not r['total_cap_exceeded']
        assert {s:sha(meta[s]) for s in ('train','validation','test')}==p['input_hashes'][job]
        data={s:prepare(observed(meta[s],meta['N'] if s=='train' else None)) for s in ('train','validation','test')}
        truth=physical_truth(Path(meta['truth_network']),labels);cache={}
        def verify(point):
            family=canonical(point['supports'])
            if family not in cache:
                cache[family]=independent_profile(data['train'],family);profiles.append(dict(job=job,family=family,**cache[family]))
            assert abs(cache[family]['objective']-point['objective'])<2e-7
        for name,branch in r['branches'].items():
            for point in branch['trajectory']['path']:
                verify(point)
                if name=='masked':assert tuple(r['selected_block']) not in canonical(point['supports'])
            for zero in branch['trajectory'].get('zero_events',[]):
                if zero['accepted']:assert zero['after']<=zero['before']+zero['loss_tolerance']
        selected=dict(historical=old['choices']['strict_0.24']['bank'],
            zero_only_own=r['choices']['baseline'],zero_only_with_rnj=r['baseline_with_rnj'],
            new_branch=r['new_branch'],own_output=r['own_output'],with_rnj_fallback=r['with_rnj_fallback'])
        for name,choice in selected.items():
            if choice is None:
                rows.append(dict(job=job,case=meta['case'],view=name,available=False,reason='not_triggered'));continue
            verify(choice['point']);metrics=score(choice['point'],data,truth,labels,canonical(common['pool']))
            assert abs(metrics['validation_mae']-choice['validation'])<1e-10
            rows.append(dict(job=job,case=meta['case'],view=name,available=True,selected_source=choice.get('source',choice.get('origin')),
                recovery_attempts=r['recovery_attempts'],mask_adopted=r['mask_adopted'],**metrics))
        checks+=8
    dest=output/'analysis';dest.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/'per_model.csv',index=False);shutil.copy2(dest/'per_model.csv',delivery/'smoke_per_model.csv')
    write(dest/'independent_profiles.json',profiles)
    write(delivery/'smoke_verification.json',dict(passed=True,returned=4,checks=checks,independent_LPs=len(profiles),
        no_120_input_rebenchmark=True,optimizer_source_frozen=True,analyzer_sha256=sha(Path(__file__))))
    shutil.copy2(output/'completion.json',delivery/'completion.json')
    print(pd.DataFrame(rows)[['case','view','available','clade_f1','exact','selected_source']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','run','worker','analyze'));parser.add_argument('path')
    args=parser.parse_args();path=Path(args.path).resolve()
    try:{'freeze':freeze,'run':run,'worker':worker,'analyze':analyze}[args.command](path)
    except Exception:
        if args.command=='worker':write(Path(read(path)['result']).with_name('error.json'),dict(traceback=traceback.format_exc()))
        raise
