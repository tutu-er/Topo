"""Resume the already frozen EV calibration/evaluation workflow.

No experiment objective or sample count is changed. External calibration jobs
finish first; both immutable threshold files gate evaluation. No physical AC
audit is launched. Worker restart runs only missing contiguous index ranges.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from hashlib import sha256

from . import ev_reliability_v2 as rel
from . import ev_systematic_analyze as io
from .ev_mobile_v2_analysis_audit import EXPECTED_METHODS

POLICY='35e7a11ba65a5366930888272f516c3165355f5dbe20de69919734dfd45e23e3'
BANKS=('n3','n4')
SEEDS={('n3','calibration'):101210001,('n3','evaluation'):101220001,
       ('n4','calibration'):101310001,('n4','evaluation'):101320001}
MODULE='research_experiments.mobile_ev_mle.ev_mobile_v2_orchestrate'
STUDY='research_experiments.mobile_ev_mle.ev_mobile_v2_study'


def utc():return datetime.now(timezone.utc).isoformat()


CHECKPOINT_RETRY_DELAYS=(.05,.1,.2,.4,.8,1.)


def atomic_checkpoint_write(path,value,*,delays=CHECKPOINT_RETRY_DELAYS):
    """Atomically replace operational metadata, retrying transient Windows locks.

    A unique sibling temp prevents competing writes from sharing a temp name.
    A persistent error is raised with the completed temp path retained; the
    original checkpoint is never truncated. Scientific case files are not used.
    """
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    temporary=Path(name)
    try:
        with os.fdopen(descriptor,'wb') as stream:
            stream.write(io._encoded(value).encode('utf-8'));stream.flush();os.fsync(stream.fileno())
        for attempt in range(len(delays)+1):
            try:
                os.replace(temporary,path)
                return
            except OSError as exc:
                transient=isinstance(exc,PermissionError) or getattr(exc,'winerror',None) in (5,32,33)
                if not transient or attempt==len(delays):raise
                time.sleep(delays[attempt])
    except BaseException as exc:
        exc.add_note('Operational checkpoint original retained after failed atomic write; temporary payload: '+str(temporary))
        raise


def progress_json(path):
    # The frozen study writes its small checkpoint directly; a poll can catch
    # the short truncate/write interval. Retry progress only, never case data.
    try:return io._read(path),None
    except FileNotFoundError:return {},None
    except json.JSONDecodeError as exc:return {},type(exc).__name__


def part_directory(root,bank,phase,part):
    return Path(root)/'outputs'/f'mobile_v2_formal_{"cal" if phase=="calibration" else "eval"}_{bank}_part{part}'


def part_progress(directory,bank,phase,part):
    directory=Path(directory);width=48 if phase=='calibration' else 24
    start=part*width;stop=start+width;first=SEEDS[bank,phase]
    expected={f'case_{i:03d}_{first+i}.json.gz' for i in range(start,stop)}
    actual={p.name for p in directory.glob('case_*.json.gz')}
    if actual-expected:raise ValueError('extra/wrong-seed record in '+str(directory))
    if actual and not (directory/'freeze.json').exists():raise ValueError('record directory lacks frozen policy: '+str(directory))
    checkpoint,checkpoint_error=progress_json(directory/'checkpoint.json')
    freeze,freeze_error=progress_json(directory/'freeze.json')
    if freeze and freeze.get('policy_id')!=POLICY:
        raise ValueError('unexpected frozen policy in '+str(directory))
    if checkpoint and (checkpoint.get('phase')!=phase or checkpoint.get('bank')!=bank):
        raise ValueError('checkpoint bank/phase differs in '+str(directory))
    marker=io._read(directory/'orchestration_complete.json') if (directory/'orchestration_complete.json').exists() else {}
    recovered=(marker.get('policy_id'),marker.get('phase'),marker.get('bank'),marker.get('start'),marker.get('stop'),marker.get('completed'))==(POLICY,phase,bank,start,stop,width)
    finished=(actual==expected and (checkpoint.get('last_index')==stop-1 or (phase=='evaluation' and recovered)))
    if phase=='calibration':finished=finished and checkpoint.get('completed')==48
    return dict(directory=str(directory),count=len(actual),expected=width,complete=bool(finished and not checkpoint_error and not freeze_error),
        last_index=checkpoint.get('last_index'),read_error=checkpoint_error or freeze_error)


def missing_ranges(directory,bank,part):
    """Skip existing evaluation cases on restart; never overwrite a case."""
    progress=part_progress(directory,bank,'evaluation',part)
    directory=Path(directory);start=24*part;stop=start+24;first=SEEDS[bank,'evaluation'];ranges=[];begin=None
    for i in range(start,stop):
        missing=not (directory/f'case_{i:03d}_{first+i}.json.gz').exists()
        if missing and begin is None:begin=i
        if not missing and begin is not None:ranges.append((begin,i));begin=None
    if begin is not None:ranges.append((begin,stop))
    return ranges


def pid_alive(pid):
    if not isinstance(pid,int) or pid<=0:return False
    if os.name=='nt':
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=(ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong);kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong));kernel.GetExitCodeProcess.restype=ctypes.c_int
        kernel.CloseHandle.argtypes=(ctypes.c_void_p,);kernel.CloseHandle.restype=ctypes.c_int
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:return False
        try:
            code=ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:kernel.CloseHandle(handle)
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
    except PermissionError:return True


def single_thread_env():
    env=os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
    return env


def require_disjoint(calibration,evaluation):
    if {r['seed'] for r in calibration['records']}&{r['seed'] for r in evaluation['records']}:
        raise ValueError('calibration/evaluation seed overlap')


def require_thresholds(root):
    paths={bank:Path(root)/'outputs'/f'mobile_v2_formal_{bank}_analysis'/'reliability_v2_thresholds.json' for bank in BANKS}
    for bank,path in paths.items():
        if not path.exists():raise ValueError('both immutable threshold files must precede evaluation: '+bank)
        value=io._read(path)
        if (value.get('calibration_count')!=96 or value.get('identity',{}).get('policy_id')!=POLICY
            or set(value.get('rules',{}))!=set(rel.RULES) or value.get('alpha')!=.1
            or value.get('gamma_AIC')!=2. or tuple(value.get('methods',()))!=EXPECTED_METHODS
            or value.get('tool_sha256')!=sha256(Path(rel.__file__).read_bytes()).hexdigest()):
            raise ValueError('threshold identity/count/rules differ: '+bank)
    return paths


def check_current_sources(freeze):
    here=Path(__file__).resolve().parent
    for name,digest in freeze['source_sha256'].items():
        path=(here.parents[1]/name) if name.startswith('rnj_wzzt_core/') else here/name
        if sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('frozen scientific source changed: '+name)


def evaluation_worker(root,part,python):
    root=Path(root).resolve();status=root/'outputs'/f'mobile_v2_evaluation_worker{part}.json'
    require_thresholds(root)
    state=dict(worker=part,pid=os.getpid(),status='running',started_utc=utc(),policy_id=POLICY)
    atomic_checkpoint_write(status,state)
    try:
        for bank in BANKS:
            directory=part_directory(root,bank,'evaluation',part)
            for start,stop in missing_ranges(directory,bank,part):
                state.update(bank=bank,start=start,stop=stop);atomic_checkpoint_write(status,state)
                args=[str(python),'-m',STUDY,'--bank',bank,'--phase','evaluation','--start',str(start),'--stop',str(stop),'--output',str(directory)]
                value=subprocess.run(args,cwd=root,env=single_thread_env(),stdin=subprocess.DEVNULL,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if value.returncode:raise RuntimeError('evaluation child exited '+str(value.returncode))
            if part_progress(directory,bank,'evaluation',part)['count']!=24:
                raise RuntimeError('evaluation child returned without all declared records')
            atomic_checkpoint_write(directory/'orchestration_complete.json',dict(policy_id=POLICY,phase='evaluation',bank=bank,
                start=24*part,stop=24*(part+1),completed=24,worker=part,completed_utc=utc()))
        state.update(status='complete',finished_utc=utc());atomic_checkpoint_write(status,state)
    except BaseException as exc:
        state.update(status='failed',failure=type(exc).__name__+': '+str(exc),finished_utc=utc());atomic_checkpoint_write(status,state)
        raise


class Controller:
    def __init__(self,root,*,python=sys.executable,poll_seconds=30.,with_selection_metadata=False):
        self.root=Path(root).resolve();self.python=str(python);self.poll_seconds=float(poll_seconds)
        if not 1<=self.poll_seconds<=60:raise ValueError('poll interval must be within 1..60 seconds')
        self.with_selection_metadata=with_selection_metadata;self.processes={};self.progress_read_errors={};self.last_progress=None
        self.path=self.root/'outputs'/'mobile_v2_orchestration_checkpoint.json'
        self.state=io._read(self.path) if self.path.exists() else dict(schema='ev_mobile_v2_orchestration_v1',policy_id=POLICY,steps={},workers={},events=[])
        if self.state.get('policy_id')!=POLICY:raise ValueError('controller checkpoint policy differs')
        owner=self.state.get('controller_pid')
        if owner!=os.getpid() and pid_alive(owner):raise RuntimeError('another controller process is active')
        self.state.update(controller_pid=None if self.state.get('status')=='statistical_complete' else os.getpid(),
            status='statistical_complete' if self.state.get('status')=='statistical_complete' else 'running',physical_audit_started=False)
        self.save()

    def save(self):
        self.state['updated_utc']=utc();atomic_checkpoint_write(self.path,self.state)

    def done(self,name,**details):
        self.state['steps'][name]=dict(completed_utc=utc(),**details);self.save()
        print(json.dumps(dict(utc=utc(),completed_step=name)),flush=True)

    def progress(self,phase):
        values={f'{bank}_{part}':part_progress(part_directory(self.root,bank,phase,part),bank,phase,part) for bank in BANKS for part in (0,1)}
        for key,value in values.items():
            self.progress_read_errors[key]=self.progress_read_errors.get(key,0)+1 if value.get('read_error') else 0
            if self.progress_read_errors[key]>=3:raise RuntimeError('repeated unreadable progress metadata: '+value['directory'])
        brief=dict(stage=phase,counts={key:value['count'] for key,value in values.items()})
        self.state['progress']=brief;self.save()
        if brief!=self.last_progress:
            print(json.dumps(dict(utc=utc(),**brief)),flush=True);self.last_progress=brief
        return all(value['complete'] for value in values.values())

    def merge_phase(self,bank,phase):
        from . import ev_mobile_v2_merge as merger
        short='cal' if phase=='calibration' else 'eval';count=96 if phase=='calibration' else 48
        output=self.root/'outputs'/f'mobile_v2_formal_{short}_{bank}'
        if output.exists():
            if (output/'merge_incomplete.json').exists():raise RuntimeError('incomplete merge preserved: '+str(output))
            verified=merger.verify(output,bank=bank,phase=phase,expected_count=count,expected_policy_id=POLICY)
            manifest=io._read(output/'merge_manifest.json')
            if verified.get('success') is False:raise RuntimeError('existing merge verification failed')
            if (manifest.get('physical_bank'),manifest.get('phase'),manifest.get('case_count'),manifest.get('policy_id'))!=(bank,phase,count,POLICY):
                raise ValueError('existing merged directory differs from frozen task')
        else:
            merger.merge([part_directory(self.root,bank,phase,p) for p in (0,1)],output,
                bank=bank,phase=phase,expected_count=count,expected_policy_id=POLICY)
        self.done('merge_'+short+'_'+bank,directory=str(output),count=count)
        return output

    def prepare_thresholds(self):
        for bank in BANKS:
            merged=self.merge_phase(bank,'calibration')
            cal=rel.read_phase(merged,expected_phase='calibration')
            out=self.root/'outputs'/f'mobile_v2_formal_{bank}_analysis'
            rel.calibrate(cal,out,alpha=.1,expected_count=96)
            self.done('thresholds_'+bank,path=str(out/'reliability_v2_thresholds.json'))
        require_thresholds(self.root);self.done('both_thresholds_saved')

    def launch_evaluation(self):
        require_thresholds(self.root)
        freeze=io._read(part_directory(self.root,'n3','calibration',0)/'freeze.json')
        check_current_sources(freeze)
        for part in (0,1):
            old=self.state['workers'].get(str(part),{})
            if pid_alive(old.get('pid')):continue
            if all(part_progress(part_directory(self.root,b,'evaluation',part),b,'evaluation',part)['complete'] for b in BANKS):continue
            log=self.root/'outputs'/f'mobile_v2_formal_eval_worker{part}.log'
            args=[self.python,'-m',MODULE,'--root',str(self.root),'--python',self.python,'--worker',str(part)]
            with log.open('ab') as stream:
                process=subprocess.Popen(args,cwd=self.root,env=single_thread_env(),stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.processes[part]=process
            self.state['workers'][str(part)]=dict(pid=process.pid,started_utc=utc(),log=str(log));self.save()
        self.done('evaluation_launched')

    def workers_finished(self):
        for part in (0,1):
            process=self.processes.get(part)
            returncode=None if process is None else process.poll()
            worker=self.state['workers'].get(str(part),{})
            path=self.root/'outputs'/f'mobile_v2_evaluation_worker{part}.json'
            status=io._read(path) if path.exists() else {}
            complete=all(part_progress(part_directory(self.root,b,'evaluation',part),b,'evaluation',part)['complete'] for b in BANKS)
            if complete:
                if pid_alive(worker.get('pid')) and status.get('status')!='complete':return False
                if (returncode is not None and returncode!=0) or status.get('status')=='failed':
                    event=dict(worker=part,records_complete=True,failure='worker tail failure; full immutable records will be verified before merge')
                    if event not in self.state['events']:self.state['events'].append(event);self.save()
                continue
            if returncode is not None and returncode!=0:
                raise RuntimeError(f'evaluation worker {part} failed; existing cases retained; see its log')
            if status.get('status')=='failed' and not pid_alive(worker.get('pid')):
                raise RuntimeError(f'evaluation worker {part} failed; restart resumes only missing records')
            if pid_alive(worker.get('pid')) and status.get('status')!='complete':return False
        return True

    def analyze(self):
        from . import ev_reliability_v2_audit as statistic
        from . import ev_mobile_v2_analysis_audit as accounting
        from . import ev_mobile_v2_results as results
        for bank in BANKS:
            merged=self.merge_phase(bank,'evaluation')
            cal=rel.read_phase(self.root/'outputs'/f'mobile_v2_formal_cal_{bank}',expected_phase='calibration')
            ev=rel.read_phase(merged,expected_phase='evaluation');require_disjoint(cal,ev)
            out=self.root/'outputs'/f'mobile_v2_formal_{bank}_analysis';thresholds=io._read(out/'reliability_v2_thresholds.json')
            decisions=rel.evaluate(ev,thresholds,out,expected_count=48)
            independent=statistic.independent_verify(cal,ev,thresholds,decisions)
            io._write(out/'independent_statistics.json',independent)
            if not independent['success']:raise RuntimeError(bank+' statistical arithmetic audit failed')
            costs=accounting.audit_pair(cal,ev,calibration_count=96,evaluation_count=48)
            io._write(out/'independent_accounting.json',costs)
            if not costs['success']:raise RuntimeError(bank+' cost/loader audit failed')
            metadata=None
            if self.with_selection_metadata:
                metadata=results.extract_selection_metadata(ev,merged);io._write(out/'selection_metadata.json',metadata)
                io._write(out/'prediction_rank_diagnostic.json',results.prediction_rank_diagnostic(ev,metadata))
            io._write(out/'results.json',results.analyze(ev,thresholds=thresholds,selection_metadata=metadata))
            self.done('statistics_'+bank,independent_statistics=True,accounting=True,selection_metadata=metadata is not None)

    def advance(self):
        if self.state.get('status')=='statistical_complete':return True
        if 'both_thresholds_saved' not in self.state['steps']:
            if not self.progress('calibration'):return False
            self.prepare_thresholds()
        require_thresholds(self.root)
        if 'evaluation_launched' not in self.state['steps']:self.launch_evaluation()
        else:
            # Recover dead workers only when this is a fresh controller start.
            if not self.processes:self.launch_evaluation()
        ready=self.progress('evaluation')
        if not self.workers_finished() or not ready:return False
        self.analyze();self.state.update(status='statistical_complete',controller_pid=None,
            selection_metadata_pending=not self.with_selection_metadata,physical_audit_pending=True);self.save()
        print(json.dumps(dict(utc=utc(),status=self.state['status'],physical_audit_started=False)),flush=True)
        return True

    def run(self):
        if self.state.get('status')=='statistical_complete':return
        stop=threading.Event()
        def heartbeat():
            while not stop.wait(self.poll_seconds):
                print(json.dumps(dict(utc=utc(),status=self.state['status'],
                    progress=self.state.get('progress'),completed_steps=list(self.state['steps']))),flush=True)
        monitor=threading.Thread(target=heartbeat,daemon=True);monitor.start()
        try:
            while not self.advance():time.sleep(self.poll_seconds)
        except BaseException as exc:
            self.state['events'].append(dict(utc=utc(),failure=type(exc).__name__+': '+str(exc)))
            self.state.update(status='failed',controller_pid=None,failure=type(exc).__name__+': '+str(exc));self.save();raise
        finally:stop.set();monitor.join(timeout=1.)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--python',default=sys.executable);parser.add_argument('--poll-seconds',type=float,default=30.)
    parser.add_argument('--worker',type=int,choices=(0,1));parser.add_argument('--with-selection-metadata',action='store_true')
    args=parser.parse_args()
    if args.worker is not None:evaluation_worker(args.root,args.worker,args.python)
    else:Controller(args.root,python=args.python,poll_seconds=args.poll_seconds,with_selection_metadata=args.with_selection_metadata).run()


if __name__=='__main__':main()
