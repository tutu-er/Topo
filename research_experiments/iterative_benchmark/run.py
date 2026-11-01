"""Immutable matched-budget replay; truth/test are available only to analysis."""
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
NEW_PARENT=ROOT/'outputs/flexible_exit_20261101/source'
METHODS=('before','after')
LIMITS=dict(total=14400.,prefix=7200.,per_mip=3600.,LP=300.,post_reserve=600.,mip_gap=.0001,threads=1)
OVERLAYS=('joint_five','rnj_count','rnj_small_seed','flexible_exit','rnj_union_long','iterative_exit','iterative_benchmark')


def freeze(output):
    import numpy as np,scipy,gurobipy as gp
    if output.exists():raise FileExistsError('New immutable output directory required.')
    parent=read(PARENT/'protocol.json');assert len(parent['jobs'])==120
    assert read(PARENT/'analysis/verification.json')['passed']
    for name,digest in parent['source_hashes'].items():assert sha(PARENT/'source'/name)==digest
    output.mkdir(parents=True);catalog=read(PARENT/'catalog.json');references={};adapters={}
    for method in METHODS:
        source=output/'sources'/method
        shutil.copytree(PARENT/'source' if method=='before' else NEW_PARENT,source)
        directories=OVERLAYS if method=='after' else ('iterative_benchmark',)
        for directory in directories:
            for file in (ROOT/'research_experiments'/directory).glob('*.py'):
                dest=source/'research_experiments'/directory/file.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
        # Only numerical stopping accuracy is harmonized in the copied sources.
        # Algorithm changes remain isolated to their own before/after snapshots.
        adapters[method]=[]
        for relative in ('research_experiments/rebuild_benchmark/engine.py','research_experiments/joint_exit/model.py'):
            file=source/relative;original=file.read_text(encoding='utf-8')
            updated=original.replace('MIPGap=.001','MIPGap=.0001').replace('requested_relative_gap=.001','requested_relative_gap=.0001')
            assert original!=updated
            before=sha(file);file.write_text(updated,encoding='utf-8',newline='\n')
            adapters[method].append(dict(file=relative,before_sha256=before,after_sha256=sha(file),change='common MIPGap 0.0001'))
    for job in parent['jobs']:
        common=PARENT/'common'/job/'result.json';historical=PARENT/'evaluation'/job/'small3_original_exit/result.json'
        dest=output/'common'/job;dest.mkdir(parents=True)
        shutil.copy2(common,dest/'common.json');shutil.copy2(historical,dest/'historical.json')
        references[job]=dict(common_sha256=sha(common),historical_sha256=sha(historical))
        assert {s:sha(catalog[job][s]) for s in ('train','validation','test')}==parent['input_hashes'][job]
    manifest=ROOT/'docs/rnj_four_methods_20261031/manifest.json';m=read(manifest)
    for name,record in m['files'].items():assert sha(manifest.parent/name)==record['sha256']
    write(output/'catalog.json',{j:catalog[j] for j in parent['jobs']})
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=parent['jobs'],methods=METHODS,planned=240,workers=8,
        limits=LIMITS,retuning=False,coefficient=.003,resolution=.24,hard_priors=[],positive_impedance_floor=None,
        before='original frozen small3_original_exit algorithm; fresh solves with the enlarged common budget',
        after='immediate zero release; one stalled-prefix deletion and unmasked/masked resume; final exit',
        selection='same historical strict .24 library/refit for each path plus singleton; same RNJ pool; source explicitly retained; native after selection also saved',
        timing='fresh worker wall time plus identical recorded common RNJ frontend; validation included; branches share total, not separate 4h allowances',
        total_max_method_hours=960.,total_max_new_worker_hours=960.-sum(read(output/'common'/j/'common.json')['seconds'] for j in parent['jobs'])*2/3600.,
        source_hashes={method:{f.relative_to(output/'sources'/method).as_posix():sha(f) for f in (output/'sources'/method).rglob('*.py')} for method in METHODS},
        transport_adapters=adapters,input_hashes=parent['input_hashes'],references=references,
        parent_manifest=dict(path=str(manifest),sha256=sha(manifest),files=m['files']),
        environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,gurobi=gp.gurobi.version(),cpus=os.cpu_count()),
        truth_used_in_training=False,test_opened_by_worker=False)
    write(output/'protocol.json',protocol);write(output/'state.json',dict(stage='frozen',planned=240,returned=0))
    delivery=ROOT/'docs'/output.name;delivery.mkdir(parents=True,exist_ok=False);shutil.copy2(output/'protocol.json',delivery/'protocol.json')
    print('FROZEN 120 matched inputs x2 fresh methods; maximum total method budget 960 hours',flush=True)


def common_select(bank,common,paths,validation):
    from research_experiments.joint_exit.selection import library,choose
    star=bank.fit([(i,) for i in range(bank.data.n)])
    rnj=[dict(row,source='complete_rnj') for row in library(bank,common['bank'],validation,resolution=.24)]
    choices={}
    for name,(active,points) in paths.items():
        rows=library(active,[*points,star],validation,resolution=.24)
        for row in rows:row['source']=name if row['index']<len(points) else 'singleton'
        choices[name]=choose(rows,'strict')
    def select(names,with_rnj=True):
        rows=[*rnj] if with_rnj else []
        rows.extend(choices[name] for name in names if name in choices)
        return choose([dict(row,index=i) for i,row in enumerate(rows)],'strict')
    return choices,dict(final=select(paths),own=select(paths,False),zero_only=select(['baseline']),
        zero_only_own=select(['baseline'],False),without_mask=select(['baseline','release']))


def worker(payload):
    import numpy as np
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.rebuild_benchmark.engine import prepare
    from research_experiments.lp_rebuild.engine import FloorBank
    from research_experiments.rnj_small_seed.engine import prefix,exit_repair
    from research_experiments.local_rebuild.model import canonical
    p=read(payload);target=Path(p['result']);then=perf_counter();common=read(p['common']);historical=read(p['historical'])
    assert sha(p['common'])==p['common_sha256'] and sha(p['historical'])==p['historical_sha256']
    assert {s:sha(p[s]) for s in ('train','validation')}==p['input_hashes']
    data=prepare(observed(p['train'],p['N']));validation=prepare(observed(p['validation']))
    limits=p['limits'];deadline=then+limits['total']-common['seconds'];structural=deadline-limits['post_reserve'];banks=[]
    class TimedBank(FloorBank):
        def fit(self,family,*,preserve=False):
            left=deadline-perf_counter()
            if left<=.05:raise TimeoutError('Matched total method deadline before LP closure')
            self.seconds=min(limits['LP'],left);self.lp.model.Params.TimeLimit=self.seconds
            return super().fit(family,preserve=preserve)
    try:
        if p['method']=='before':
            bank=TimedBank(data,common['pool'],common['calibration'],seconds=limits['LP']);banks.append(bank)
            bank.joint_exit_seed_pool=canonical(common['pool']);bank.seed_start_pool=bank.joint_exit_seed_pool
            t=prefix(bank,common['initial'],mode='small3',reference=p['reference'],strategy='original',
                seconds=min(limits['prefix'],max(.1,(structural-perf_counter())/2)),per_mip=limits['per_mip'])
            write(target.with_name('checkpoint.json'),dict(stage='prefix_complete',trajectory=t))
            q,event=exit_repair(bank,common['initial'],t['point'],np.asarray(common['kernel']),reference=p['reference'],
                prefix_seconds=0.,warm_candidates=common['bank'],total_seconds=max(.1,structural-perf_counter()))
            path=[*t['path'],q['point']] if q['accepted'] else t['path']
            result=dict(prefix=t,branches=dict(baseline=dict(trajectory={**t,'path':path,'point':q['point']},exit=q,exit_event=event)),
                recovery_attempts=0,selected_block=None,stalled=False,ranking={})
            paths={'baseline':(bank,path)}
        else:
            from research_experiments.iterative_exit.engine import fit_mainline
            from research_experiments.flexible_exit.model import MaskBank
            result=fit_mainline(data,validation,common,reference=p['reference'],seconds=deadline-perf_counter(),
                prefix_seconds=limits['prefix'],per_mip=limits['per_mip'],lp_seconds=limits['LP'],post_reserve=limits['post_reserve'],
                checkpoint=lambda r:write(target.with_name('checkpoint.json'),r),heartbeat=lambda r:write(target.with_name('progress.json'),r))
            result['native_selection']=result['with_rnj_fallback'];paths={}
            for name,branch in result['branches'].items():
                forbidden=branch.get('forbidden',())
                active=MaskBank(data,common['pool'],common['calibration'],forbidden=forbidden,deadline=deadline,seconds=limits['LP'])
                banks.append(active);paths[name]=(active,branch['trajectory']['path'])
            bank=paths['baseline'][0]
        choices,views=common_select(bank,common,paths,validation)
        fresh=perf_counter()-then
        result.update(job=p['job'],method=p['method'],choices_matched=choices,views=views,
            input_hashes=p['input_hashes'],protocol_sha256=p['protocol_sha256'],common_sha256=p['common_sha256'],
            limits=limits,common_seconds=common['seconds'],fresh_seconds=fresh,total_seconds=fresh+common['seconds'],
            total_cap_exceeded=fresh+common['seconds']>limits['total']+1.,truth_used=False,test_opened=False,
            post_LP_failures=[row for bank in banks for row in bank.failures])
        write(target,result)
    finally:
        for bank in banks:bank.close()


def completion(output):
    p=read(output/'protocol.json');rows=[]
    for method in METHODS:
        returned=[j for j in p['jobs'] if (output/'jobs'/j/method/'result.json').exists()]
        failures=[dict(job=j,error=read(output/'jobs'/j/method/'error.json')) for j in p['jobs']
            if j not in returned and (output/'jobs'/j/method/'error.json').exists()]
        rows.append(dict(method=method,planned=120,returned=len(returned),failures=failures))
    return dict(planned=240,returned=sum(r['returned'] for r in rows),methods=rows)


def run(output):
    p=read(output/'protocol.json');catalog=read(output/'catalog.json');queue=[];active={};started=perf_counter()
    for method in METHODS:
        for name,digest in p['source_hashes'][method].items():assert sha(output/'sources'/method/name)==digest
    # Round-robin cases and balanced pair order reduce queue/order timing bias.
    cases=sorted({m['case'] for m in catalog.values()});jobs=[j for k in range(30) for c in cases for j in [x for x in p['jobs'] if catalog[x]['case']==c][k:k+1]]
    assert set(jobs)==set(p['jobs'])
    for index,job in enumerate(jobs):
        meta=catalog[job];common=output/'common'/job/'common.json';old=output/'common'/job/'historical.json'
        for method in METHODS[::1 if index%2==0 else -1]:
            target=output/'jobs'/job/method/'result.json'
            if target.exists() or target.with_name('error.json').exists():continue
            queue.append(dict(job=job,method=method,N=meta['N'],train=meta['train'],validation=meta['validation'],
                common=str(common),historical=str(old),**p['references'][job],reference=read(old)['trajectory']['reference_MAE'],
                input_hashes={s:p['input_hashes'][job][s] for s in ('train','validation')},result=str(target),limits=p['limits'],protocol_sha256=sha(output/'protocol.json')))
    while queue or active:
        while queue and len(active)<p['workers']:
            payload=queue.pop(0);target=Path(payload['result']);file=target.with_name('payload.json');write(file,payload)
            source=output/'sources'/payload['method'];env=os.environ.copy();env['PYTHONPATH']=str(source)+os.pathsep+str(source/'rnj_wzzt_core')
            log=target.with_name('worker.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-B','-m','research_experiments.iterative_benchmark.run','worker',str(file)],cwd=source,env=env,
                stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active[(payload['job'],payload['method'])]=(proc,log,target,perf_counter())
        for key,(proc,log,target,start) in list(active.items()):
            if proc.poll() is None:continue
            log.close();del active[key]
            if not target.exists() and not target.with_name('error.json').exists():write(target.with_name('error.json'),dict(exit_code=proc.returncode))
            print('finished',*key,'OK' if target.exists() else 'ERROR',round(perf_counter()-start,2),flush=True)
        state=completion(output);state.update(stage='running' if active or queue else 'finished',active=[dict(job=k[0],method=k[1],pid=v[0].pid,seconds=perf_counter()-v[3]) for k,v in active.items()],queued=len(queue),batch_wall_seconds=perf_counter()-started)
        write(output/'state.json',state)
        if active:sleep(2.)
    write(output/'completion.json',completion(output));print('COMPLETE',completion(output)['returned'],'/240',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','run','worker','status'));parser.add_argument('path')
    a=parser.parse_args();path=Path(a.path).resolve()
    try:
        if a.command=='status':print(completion(path))
        else:globals()[a.command](path)
    except Exception:
        if a.command=='worker':write(Path(read(path)['result']).with_name('error.json'),dict(traceback=traceback.format_exc()))
        raise
