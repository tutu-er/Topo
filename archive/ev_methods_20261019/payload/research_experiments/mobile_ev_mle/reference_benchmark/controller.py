"""Resumable frozen reference batch; calibration is sealed before evaluation."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
import argparse,os,time,traceback
from pathlib import Path
from . import study,analysis
from .config import CONFIG
from .io import ROOT,read,write,freeze,digest,serial


def now():return datetime.now(timezone.utc).isoformat()


def seal_calibration(output,prepared,identity):
    """No evaluation record is opened when producing the source-blind rule."""
    output=Path(output)
    cal=analysis.records(output,'calibration')
    expected=len(study.specifications(prepared,'calibration'))
    if len(cal)!=expected:raise ValueError('all calibration results required before threshold freeze')
    expected_specs={(s['cluster_index'],s['network'],s['source_mode'])
                    for s in study.specifications(prepared,'calibration')}
    actual_specs={(r['spec']['cluster_index'],r['spec']['network'],r['spec']['source_mode']) for r in cal}
    if actual_specs!=expected_specs:raise ValueError('calibration condition identities differ')
    if any(r.get('freeze_identity')!=identity for r in cal):raise ValueError('calibration policy mismatch')
    # Use precisely the same durable threshold schema as final analysis,
    # including calibration-file fingerprints. A second bare threshold file
    # would conflict with the later evidence replay.
    thresholds=serial(analysis.freeze_thresholds(output,methods=CONFIG['methods'],
        alpha=CONFIG['alpha'],expected_conditions=expected))
    tp=output/'thresholds.json'
    receipt=dict(recorded_utc=now(),threshold_sha256=digest(tp),
        evaluation_existing=len(list((output/'evaluation').glob('*.json.gz'))),
        evaluation_outcomes_read=0,condition_threshold_uses_source_label=False,
        policy_identity=identity,calibration_records=len(cal))
    write(output/'calibration_freeze_receipt.json',receipt)
    return receipt


def saved_or_pending(output,specs,identity):
    """A saved failed result is complete computational work, not a resampling request."""
    pending=[];saved=[]
    for s in specs:
        path=study.path_for(output,s)
        if path.exists():
            r=read(path)
            if r.get('freeze_identity')!=identity:raise ValueError('saved policy mismatch '+str(path))
            if r.get('spec')!=s:raise ValueError('saved condition mismatch '+str(path))
            saved.append(dict(spec=s,path=str(path),status=r['status'],resumed=True))
        else:pending.append(s)
    return saved,pending


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/reference_v1')
    p.add_argument('--prepared',type=Path,default=ROOT/'outputs/inputs_v2/prepared.json.gz')
    p.add_argument('--workers',type=int,default=CONFIG['max_workers'])
    p.add_argument('--phases',nargs='+',choices=['development','calibration','evaluation'],
        default=['development','calibration','evaluation'])
    a=p.parse_args()
    if a.workers<1 or a.workers>CONFIG['max_workers']:raise ValueError('workers outside frozen budget')
    if a.phases!=sorted(a.phases,key=['development','calibration','evaluation'].index):raise ValueError('phase order required')
    a.output.mkdir(parents=True,exist_ok=True);lock=a.output/'controller.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    state=dict(status='starting',pid=os.getpid(),started_utc=now(),workers=a.workers,
        output=str(a.output),prepared=str(a.prepared),active_phase=None,live_worker_pids=[],
        scientific_success_assumed=False,phases={},exceptions=[],completed=[],per_worker_threads=1)
    try:
        prepared=read(a.prepared)
        for phase in a.phases:
            if len(prepared['windows'][phase])!=CONFIG['phase_counts'][phase]:
                raise ValueError('prepared window count differs from frozen scientific config')
        sealed=freeze(a.output,CONFIG,a.prepared);state['freeze_identity']=sealed['identity']
        write(a.output/'controller.json',state)
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for phase in a.phases:
                if phase=='evaluation':
                    seal_calibration(a.output,prepared,sealed['identity'])
                specs=study.specifications(prepared,phase)
                saved,pending=saved_or_pending(a.output,specs,sealed['identity'])
                state.update(status='running',active_phase=phase)
                state['phases'][phase]=dict(total=len(specs),computed=len(saved),remaining=len(pending))
                state['completed'].extend(saved);write(a.output/'controller.json',state)
                tasks={pool.submit(study.run,s,a.output,a.prepared):s for s in pending}
                while tasks:
                    done,_=wait(tasks,timeout=5,return_when=FIRST_COMPLETED)
                    for future in done:
                        spec=tasks.pop(future)
                        try:row=future.result()
                        except Exception as exc:
                            row=dict(spec=spec,status='worker_exception',seconds=None,exception=repr(exc))
                            # Retain the missing outcome in its original denominator.
                            path=study.path_for(a.output,spec)
                            write(path,dict(spec=spec,config=CONFIG,status='failed',
                                freeze_identity=sealed['identity'],source_unchanged=True,
                                failures=dict(worker_exception=repr(exc)),traceback=traceback.format_exc()))
                        state['completed'].append(row)
                        state['phases'][phase]['computed']+=1
                        state['phases'][phase]['remaining']=len(tasks)
                        print('computed',phase,spec['network'],spec['cluster_index'],spec['source_mode'],row['status'],flush=True)
                    state['live_worker_pids']=sorted(pool._processes)
                    state['updated_utc']=now();write(a.output/'controller.json',state)
        state.update(status='statistical_complete',active_phase=None,live_worker_pids=[],
            completed_utc=now(),computation_complete=True)
        write(a.output/'controller.json',state)
    except Exception as exc:
        state.update(status='blocked',error=repr(exc),traceback=traceback.format_exc(),updated_utc=now())
        write(a.output/'controller.json',state);raise
    finally:
        # Only this process's own lock is removed.
        if lock.exists() and lock.read_text(encoding='utf-8')==str(os.getpid()):lock.unlink()


if __name__=='__main__':main()
