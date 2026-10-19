"""Operational handoff only: finish the one frozen B evaluation after calibration."""
from pathlib import Path
import sys,os,time,subprocess
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[variable]='1'
WORK=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(WORK))
from research_experiments.mobile_ev_mle.ac_history_refresh import control,io


def decision(state,lock_exists,pid_alive):
    if state.get('status')=='statistical_complete':return 'done'
    if state.get('phase')=='evaluation':return 'evaluation_already_started'
    if state.get('status')=='calibration_complete':return 'wait' if lock_exists else 'launch'
    return 'wait' if pid_alive else 'calibration_controller_stopped'


def run(root):
    root=Path(root).resolve();version=control.frozen(root)
    receipt=root/'automatic_evaluation_handoff.json'
    if receipt.exists():raise FileExistsError('preserve previous handoff receipt')
    control.atomic_json(receipt,dict(status='waiting_for_calibration',pid=os.getpid(),policy_id=version['policy_id']))
    while True:
        control.frozen(root);state=io.load_record(root/'controller.json')
        action=decision(state,(root/'controller_lock.json').exists(),control._pid_alive(state['pid']))
        if action=='wait':time.sleep(20);continue
        if action!='launch':
            control.atomic_json(receipt,dict(status=action,pid=os.getpid(),policy_id=version['policy_id'],controller=state))
            if action=='calibration_controller_stopped':raise RuntimeError(action)
            return
        # All 32 calibration records and the frozen independent choice are
        # verified here, before any evaluation raw is opened by the worker.
        binding=control.bind_evaluation(root,version)
        command=[sys.executable,'-u','-m','research_experiments.mobile_ev_mle.ac_history_refresh.control',
                 'run','--root',str(root),'--phase','evaluation']
        with (root/'controller.eval.stdout.log').open('xb',buffering=0) as out, (root/'controller.eval.stderr.log').open('xb',buffering=0) as err:
            child=subprocess.Popen(command,cwd=WORK,stdout=out,stderr=err,env=dict(os.environ),
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            control.atomic_json(receipt,dict(status='evaluation_started',pid=os.getpid(),controller_pid=child.pid,
                command=command,policy_id=version['policy_id'],calibration_binding=binding,
                wrapper_sha256=control.file_hash(__file__),no_scientific_source_or_inputs_changed=True))
            code=child.wait()
        control.atomic_json(root/'automatic_evaluation_exit.json',dict(returncode=code,wrapper_pid=os.getpid(),
            controller_pid=child.pid,controller=io.load_record(root/'controller.json'),policy_id=version['policy_id']))
        if code:raise RuntimeError('frozen evaluation controller exit '+str(code))
        return


if __name__=='__main__':run(sys.argv[1])
