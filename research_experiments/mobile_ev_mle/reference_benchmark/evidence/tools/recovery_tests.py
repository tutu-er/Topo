"""Fault-injection checks for waiting/resuming only; no scientific work is run."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys,uuid
from unittest.mock import patch
import recovery_supervisor as s
import finish_watch as w
from runtime_resume import ROOT,io


def main():
    root=(ROOT/'outputs/reference_tools/tests'/('recovery_'+uuid.uuid4().hex[:12])).resolve()
    assert root.is_relative_to(ROOT/'outputs/reference_tools/tests')
    root.mkdir(parents=True);checks=[]
    def fixture(name):
        out=root/name;out.mkdir()
        io.write(out/'controller.json',dict(status='running',pid=123,live_worker_pids=[456],
            updated_utc=(datetime.now(timezone.utc)-timedelta(hours=1)).isoformat()))
        return out
    def argv(out):
        return ['recovery_supervisor.py','--output',str(out),'--prepared',str(root/'prepared.json.gz'),
                '--destination',str(root/'delivery')]
    active=fixture('active')
    with patch.object(sys,'argv',argv(active)),patch.object(s,'process_is_alive',return_value=True), \
         patch.object(s.time,'sleep',side_effect=RuntimeError('end waiting fixture')), \
         patch.object(s.subprocess,'Popen') as launch:
        try:s.main()
        except RuntimeError as error:assert str(error)=='end waiting fixture'
        else:raise AssertionError('waiting fixture unexpectedly continued')
        assert not launch.called
    checks.append('active_original_processes_never_restarted_or_terminated')
    idle=fixture('idle');calls=[]
    class Completed:
        pid=987;returncode=0
        def __init__(self,args,**kwargs):
            calls.append(args)
            if '--resume' in args:io.write(idle/'controller.json',dict(status='statistical_complete',pid=123,live_worker_pids=[]))
            else:io.write(idle/'postprocessor.json',dict(status='ready_for_result_review'))
        def poll(self):return 0
    with patch.object(sys,'argv',argv(idle)),patch.object(s,'process_is_alive',return_value=False), \
         patch.object(s.subprocess,'Popen',Completed):s.main()
    assert len(calls)==2 and '--resume' in calls[0] and str(calls[1][1]).endswith('finish_watch.py')
    assert io.read(idle/'recovery.json')['status']=='ready_for_result_review'
    checks.append('exited_run_resumes_once_before_postprocessing')
    failed=fixture('failed');fail_calls=[]
    class Failed:
        pid=888;returncode=7
        def __init__(self,args,**kwargs):fail_calls.append(args)
        def poll(self):return 7
    with patch.object(sys,'argv',argv(failed)),patch.object(s,'process_is_alive',return_value=False), \
         patch.object(s.subprocess,'Popen',Failed):
        try:s.main()
        except RuntimeError as error:assert 'no automatic second resume' in str(error)
        else:raise AssertionError('failed resume not reported')
    assert len(fail_calls)==1 and io.read(failed/'recovery.json')['status']=='needs_review'
    checks.append('failed_resume_retained_without_retrying_scientific_conditions')
    stale=fixture('stale_watcher')
    with patch.object(sys,'argv',['finish_watch.py','--output',str(stale),'--destination',str(root/'delivery')]), \
         patch.object(w,'process_is_alive',return_value=True), \
         patch.object(w.time,'sleep',side_effect=RuntimeError('end stale fixture')),patch.object(w.subprocess,'run') as work:
        try:w.main()
        except RuntimeError as error:assert str(error)=='end stale fixture'
        else:raise AssertionError('stale fixture unexpectedly completed')
        assert not work.called
    assert io.read(stale/'postprocessor.json')['health']=='heartbeat_stale_but_processes_active'
    checks.append('stale_heartbeat_with_live_processes_is_not_declared_dead')
    frozen=io.read(ROOT/'outputs/reference_v1/freeze.json')
    assert io.source_hashes()==frozen['source_sha256']
    result=dict(passed=len(checks),checks=checks,science_fit_calls=0,science_plan_calls=0,
                scientific_sources_unchanged=len(frozen['source_sha256']),fixture_root=str(root))
    io.write(ROOT/'outputs/reference_tools/recovery_selftest_receipt.json',result)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
