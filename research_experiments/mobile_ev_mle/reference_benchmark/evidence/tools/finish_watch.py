"""Complete the frozen batch's prespecified analysis and evidence, without new experiments."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,subprocess,sys,time,traceback


def now():return datetime.now(timezone.utc).isoformat()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True)+'\n',encoding='utf-8')
    temporary.replace(path)


def read(path):return json.loads(Path(path).read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve();root=Path(__file__).resolve().parents[2]
    state_path=output/'postprocessor.json';lock=output/'postprocessor.lock'
    with lock.open('x',encoding='utf-8') as handle:handle.write(str(os.getpid()))
    state=dict(pid=os.getpid(),started_utc=now(),status='waiting_for_frozen_computation',
               creates_new_scientific_batches=False,selector_or_threshold_tuning=False,
               output=str(output),destination=str(args.destination),steps=[])
    def save(**kwargs):state.update(kwargs,updated_utc=now());write(state_path,state)
    def run_step(label,arguments):
        save(status='running_postprocessing',stage=label)
        started=time.monotonic()
        with (output/(label+'.stdout.log')).open('w',encoding='utf-8') as stdout, \
             (output/(label+'.stderr.log')).open('w',encoding='utf-8') as stderr:
            result=subprocess.run([sys.executable,*map(str,arguments)],cwd=root,
                stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        state['steps'].append(dict(stage=label,exit_code=result.returncode,seconds=time.monotonic()-started))
        save()
        if result.returncode:raise RuntimeError(label+' failed; see saved stderr')
    try:
        save()
        while True:
            path=output/'controller.json'
            if path.exists():
                controller=read(path);status=controller.get('status')
                save(controller_status=status,active_phase=controller.get('active_phase'),
                     phases=controller.get('phases',{}),controller_updated_utc=controller.get('updated_utc'))
                if status=='statistical_complete':break
                if status=='blocked':raise RuntimeError('controller blocked; no replacement batch started')
                clock=controller.get('updated_utc',controller.get('started_utc'))
                if clock and (datetime.now(timezone.utc)-datetime.fromisoformat(clock)).total_seconds()>300:
                    raise RuntimeError('controller heartbeat stale for >300 seconds; inspect PID, do not restart blindly')
            time.sleep(30)
        run_step('analysis',['-m','research_experiments.mobile_ev_mle.reference_benchmark.analysis','--output',output])
        run_step('diagnostics',[root/'outputs/reference_tools/diagnostics.py','--output',output])
        audit=read(output/'audit.json')
        if not audit.get('passed'):raise RuntimeError('physical or evidence audit failed; retain diagnostics for review')
        run_step('package_evidence',[root/'outputs/reference_tools/package_evidence.py',
            '--output',output,'--destination',args.destination])
        delivery=read(args.destination/'delivery_manifest.json')
        if not delivery.get('ready_for_review'):
            raise RuntimeError('evidence package is not ready for review; inspect delivery_manifest.json')
        save(status='ready_for_result_review',stage=None,completed_utc=now(),
             results_complete=True,git_published=False,
             next_action='review results and evidence; commit/push only verified task deliverables')
    except Exception as error:
        save(status='needs_review',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        if lock.exists() and lock.read_text(encoding='utf-8')==str(os.getpid()):lock.unlink()


if __name__=='__main__':main()
