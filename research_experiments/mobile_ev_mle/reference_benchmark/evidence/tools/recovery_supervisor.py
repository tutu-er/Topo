"""Wait for the existing run, resume its saved conditions once, then finish evidence."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from runtime_resume import ROOT,io,read_shared_json,process_is_alive


def now():return datetime.now(timezone.utc).isoformat()


def inventory(output):
    return {phase:len(list((output/phase).glob('*.json.gz')))
            for phase in ('development','calibration','evaluation')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args();output=args.output.resolve()
    for path in (output,args.prepared.resolve(),args.destination.resolve()):
        if not path.is_relative_to(ROOT/'outputs'):raise ValueError('task paths must remain inside execution outputs')
    lock=output/'recovery_supervisor.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write(str(os.getpid()))
    record=dict(pid=os.getpid(),started_utc=now(),status='waiting_existing_processes',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        creates_new_scientific_batch=False,terminates_existing_processes=False,
        changes_scientific_source=False,resume_attempted=False,steps=[])
    def save(**fields):
        record.update(fields,updated_utc=now(),actual_saved_conditions=inventory(output))
        io.write(output/'recovery.json',record)
    def step(name,arguments):
        with (output/(name+'.stdout.log')).open('w',encoding='utf-8') as stdout, \
             (output/(name+'.stderr.log')).open('w',encoding='utf-8') as stderr:
            child=subprocess.Popen([sys.executable,*map(str,arguments)],cwd=ROOT,
                stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            save(status='running_'+name,active_child_pid=child.pid)
            while child.poll() is None:
                save();time.sleep(30)
        record['steps'].append(dict(step=name,pid=child.pid,exit_code=child.returncode,ended_utc=now()))
        save(active_child_pid=None)
        if child.returncode:raise RuntimeError(name+' stopped; see its saved stderr; no automatic second resume')
    try:
        save()
        states=[read_shared_json(path) for path in (output/'controller.json',output/'controller.json.tmp') if path.exists()]
        pids=set()
        for state in states:
            if state.get('pid'):pids.add(int(state['pid']))
            pids.update(map(int,state.get('live_worker_pids',[])))
        save(original_recorded_pids=sorted(pids))
        while True:
            active=sorted(pid for pid in pids if process_is_alive(pid))
            save(active_existing_pids=active,controller_status=read_shared_json(output/'controller.json').get('status'))
            if not active:break
            time.sleep(30)
        state=read_shared_json(output/'controller.json')
        if state.get('status')!='statistical_complete':
            save(resume_attempted=True)
            step('runtime_resume',[ROOT/'outputs/reference_tools/runtime_resume.py','--resume',
                '--output',output,'--prepared',args.prepared,'--workers',args.workers])
        state=read_shared_json(output/'controller.json')
        if state.get('status')!='statistical_complete':
            raise RuntimeError('controller has not reached statistical_complete after saved-run recovery')
        step('recovered_postprocessing',[ROOT/'outputs/reference_tools/finish_watch.py',
            '--output',output,'--destination',args.destination])
        post=read_shared_json(output/'postprocessor.json')
        if post.get('status')!='ready_for_result_review':raise RuntimeError('postprocessing did not pass evidence gates')
        save(status='ready_for_result_review',completed_utc=now(),
             next_action='review report and restored evidence; commit/push verified deliverables')
    except Exception as error:
        save(status='needs_review',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        if lock.exists() and lock.read_text(encoding='utf-8')==str(os.getpid()):lock.unlink()


if __name__=='__main__':main()
