"""Orchestration tests use fake processes; no scientific history is executed."""
from hashlib import sha256
import gzip
import json
from pathlib import Path
import sys
import time

import pytest

from . import ev_bayes_response_recovery_controller as c


def frozen():
    return dict(policy_id="fixed-policy", source_sha256={}, specifications={
        "calibration": [dict(index=i, phase="calibration") for i in range(4)],
        "evaluation": [dict(index=i, phase="evaluation") for i in range(2)]})


def save(root, phase, index, status="complete", policy="fixed-policy"):
    path = c.record_path(root, phase, index)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(dict(spec=dict(index=index, phase=phase),
        policy_id=policy, status=status, seconds=2.)).encode()))
    return path


def test_native_codes():
    assert c.native_exit(3221225477)
    assert c.native_exit(-1073741819)
    assert not c.native_exit(0)
    assert not c.native_exit(1)
    assert not c.native_exit(None)


def test_final_scientific_failure_is_preserved(tmp_path):
    save(tmp_path, "calibration", 1, "inference_failure")
    r = c.FinalRecords(tmp_path, frozen())
    assert r.get("calibration", 1)["status"] == "inference_failure"
    assert c.eligible_indices(frozen(), "calibration", r, set(), c.EventLog(tmp_path/"events")) == [0, 2, 3]


def test_final_wrong_policy_blocks(tmp_path):
    save(tmp_path, "calibration", 0, policy="wrong")
    with pytest.raises(RuntimeError, match="policy mismatch"):
        c.FinalRecords(tmp_path, frozen()).get("calibration", 0)


def test_final_wrong_identity_blocks(tmp_path):
    path=save(tmp_path, "calibration", 0)
    path.write_bytes(gzip.compress(json.dumps(dict(policy_id="fixed-policy", spec=dict(index=2, phase="calibration"),status="complete")).encode()))
    with pytest.raises(RuntimeError, match="identity mismatch"):
        c.FinalRecords(tmp_path, frozen()).get("calibration", 0)


def test_unreadable_final_never_overwritten(tmp_path):
    path=c.record_path(tmp_path,"calibration",0);path.parent.mkdir();path.write_bytes(b"partial")
    with pytest.raises(gzip.BadGzipFile):
        c.FinalRecords(tmp_path, frozen()).summary("calibration")
    assert path.read_bytes()==b"partial"


def test_unexpected_final_blocks(tmp_path):
    save(tmp_path,"calibration",9)
    with pytest.raises(RuntimeError,match="unexpected"):
        c.FinalRecords(tmp_path,frozen()).summary("calibration")


def test_reserved_adopted_indices_not_scheduled(tmp_path):
    r=c.FinalRecords(tmp_path,frozen());e=c.EventLog(tmp_path/"events")
    assert c.eligible_indices(frozen(),"calibration",r,{1,3},e)==[0,2]


def test_first_unsaved_legacy_native_failure_counted_once(tmp_path):
    save(tmp_path,"calibration",0)
    previous=dict(phase="calibration",controller_pid=10,workers=[dict(pid=11,exit_code=3221225477,indices=[0,2])])
    records=c.FinalRecords(tmp_path,frozen());events=c.EventLog(tmp_path/"events")
    c.import_original_failure(previous,records,events)
    c.import_original_failure(previous,records,events)
    assert len(events.attempts("calibration",2))==1
    assert len(events.attempts("calibration",0))==0
    assert c.eligible_indices(frozen(),"calibration",records,set(),events)==[1,2,3]


def test_second_native_failure_stops(tmp_path):
    records=c.FinalRecords(tmp_path,frozen());events=c.EventLog(tmp_path/"events")
    events.append(event="original_native_failure",phase="calibration",index=0,exit_code=3221225477)
    events.append(event="attempt_started",phase="calibration",index=0)
    events.append(event="attempt_finished",phase="calibration",index=0,exit_code=3221225477)
    with pytest.raises(RuntimeError,match="retry limit"):
        c.eligible_indices(frozen(),"calibration",records,set(),events)


def test_non_native_failure_never_retried(tmp_path):
    records=c.FinalRecords(tmp_path,frozen());events=c.EventLog(tmp_path/"events")
    events.append(event="attempt_started",phase="calibration",index=0)
    events.append(event="attempt_finished",phase="calibration",index=0,exit_code=1)
    with pytest.raises(RuntimeError,match="non-native"):
        c.eligible_indices(frozen(),"calibration",records,set(),events)


def test_unaccounted_attempt_blocks(tmp_path):
    events=c.EventLog(tmp_path/"events")
    events.append(event="attempt_started",phase="calibration",index=0)
    with pytest.raises(RuntimeError,match="unaccounted"):
        c.eligible_indices(frozen(),"calibration",c.FinalRecords(tmp_path,frozen()),set(),events)


def test_event_log_survives_resume(tmp_path):
    path=tmp_path/"events";first=c.EventLog(path)
    first.append(event="original_native_failure",phase="calibration",index=2,exit_code=3221225477)
    second=c.EventLog(path)
    assert len(second.attempts("calibration",2))==1


def test_frozen_hashes_checked_without_import(tmp_path):
    here=tmp_path/"research_experiments"/"mobile_ev_mle";here.mkdir(parents=True)
    science=here/"science.py";science.write_text("raise RuntimeError('do not execute')",encoding="utf-8")
    spec=dict(policy_id="P",source_sha256={"science.py":sha256(science.read_bytes()).hexdigest()})
    assert c.verify_sources(spec,here)["files"]==1
    science.write_text("changed",encoding="utf-8")
    with pytest.raises(RuntimeError,match="source mismatch"):
        c.verify_sources(spec,here)


def test_frozen_core_and_relative_path_mapping(tmp_path):
    here=tmp_path/"research_experiments"/"mobile_ev_mle";here.mkdir(parents=True)
    target=tmp_path/"rnj_wzzt_core"/"x.py";target.parent.mkdir();target.write_bytes(b"science")
    relative=here.parent/"legacy.py";relative.write_bytes(b"legacy")
    hashes={"rnj_wzzt_core/x.py":sha256(target.read_bytes()).hexdigest(),"../legacy.py":sha256(relative.read_bytes()).hexdigest()}
    assert c.verify_sources(dict(policy_id="P",source_sha256=hashes),here)["files"]==2


def test_previous_live_controller_rejected(tmp_path):
    with pytest.raises(RuntimeError,match="previous controller"):
        c.acquire_lock(tmp_path/"lock",dict(controller_pid=123),alive=lambda pid:pid==123)
    assert not (tmp_path/"lock").exists()


def test_both_locks_owned_and_old_lock_archived(tmp_path):
    (tmp_path/"controller.lock").write_text(json.dumps(dict(pid=123)))
    locks=c.acquire_runtime_locks(tmp_path,dict(controller_pid=123),alive=lambda pid:False)
    assert len(locks)==2
    assert {c.read(p)["pid"] for p in locks}=={c.os.getpid()}
    assert len(list(tmp_path.glob("stale_controller_*")))==1


def test_other_live_lock_rejected_without_mutation(tmp_path):
    path=tmp_path/"controller.lock";path.write_text(json.dumps(dict(pid=123)))
    with pytest.raises(RuntimeError,match="lock still active"):
        c.acquire_runtime_locks(tmp_path,dict(controller_pid=123),alive=lambda pid:True)
    assert c.read(path)["pid"]==123
    assert not (tmp_path/"recovery_controller.lock").exists()


def test_partial_preserved_losslessly(tmp_path):
    partial=c.record_path(tmp_path,"calibration",2);partial.parent.mkdir();partial=partial.with_name(partial.name+".tmp")
    partial.write_bytes(b"partial original")
    runtime=tmp_path/"runtime";runtime.mkdir()
    row=c.preserve_partial(tmp_path,"calibration",2,runtime,"attempt")
    assert partial.read_bytes()==b"partial original"
    assert Path(row["preserved"]).read_bytes()==partial.read_bytes()
    assert row["sha256"]==sha256(partial.read_bytes()).hexdigest()


def test_single_history_command_and_thread_environment(tmp_path):
    command=c.worker_command(tmp_path,"evaluation",17,"worker")
    assert command[-2:]==["--indices","17"]
    assert command[0]==sys.executable
    assert "ev_bayes_response_runtime_entry" in command[2]
    assert all(c.worker_environment()[name]=="1" for name in c.NUMERIC_THREADS)


class FakeProcess:
    pid=25
    image=sys.executable
    created_filetime=100
    def __init__(self,pid=25,code=None):self.pid=pid;self.code=code;self.closed=False
    def poll(self):return self.code
    def close(self):self.closed=True


def test_adoption_keeps_entire_legacy_partition_reserved(tmp_path):
    events=c.EventLog(tmp_path/"events")
    active=c.adopt_workers(dict(workers=[dict(pid=25,exit_code=None,indices=[1,3])]),"calibration",events,process_factory=FakeProcess)
    assert active[0]["indices"]==[1,3]
    assert not active[0]["process"].closed
    assert c.worker_rows(active)[0]["created_filetime"]==100


def test_adoption_rejects_pid_reuse(tmp_path):
    with pytest.raises(RuntimeError,match="creation time changed"):
        c.adopt_workers(dict(workers=[dict(pid=25,exit_code=None,indices=[1],created_filetime=99)]),"calibration",c.EventLog(tmp_path/"events"),process_factory=FakeProcess)


def test_finished_scientific_failure_does_not_retry(tmp_path):
    save(tmp_path,"calibration",0,"execution_failure")
    process=FakeProcess(code=0);events=c.EventLog(tmp_path/"events")
    events.append(event="attempt_started",phase="calibration",index=0,attempt_id="A")
    worker=dict(process=process,phase="calibration",indices=[0],adopted=False,started_monotonic=time.monotonic(),log=None,attempt_id="A")
    c.finish_worker(worker,c.FinalRecords(tmp_path,frozen()),events)
    assert len(events.finished_attempts("calibration",0))==1
    assert c.eligible_indices(frozen(),"calibration",c.FinalRecords(tmp_path,frozen()),set(),events)==[1,2,3]


def test_adopted_legacy_native_failure_counts_only_first_missing(tmp_path):
    save(tmp_path,"calibration",1)
    process=FakeProcess(code=3221225477);events=c.EventLog(tmp_path/"events")
    worker=dict(process=process,phase="calibration",indices=[1,3],adopted=True,started_monotonic=time.monotonic(),log=None,attempt_id=None)
    c.finish_worker(worker,c.FinalRecords(tmp_path,frozen()),events)
    assert len(events.attempts("calibration",3))==1
    assert len(events.attempts("calibration",1))==0


def test_adopted_recovery_attempt_has_one_completion(tmp_path):
    events=c.EventLog(tmp_path/"events")
    events.append(event="attempt_started",phase="calibration",index=0,attempt_id="A")
    process=FakeProcess(code=3221225477)
    worker=dict(process=process,phase="calibration",indices=[0],adopted=True,started_monotonic=time.monotonic(),log=None,attempt_id="A")
    c.finish_worker(worker,c.FinalRecords(tmp_path,frozen()),events)
    assert len(events.attempts("calibration",0))==1
    assert len(events.finished_attempts("calibration",0))==1


def test_recovered_dead_tracked_attempt_does_not_double_count(tmp_path):
    events=c.EventLog(tmp_path/"events")
    events.append(event="attempt_started",phase="calibration",index=0,attempt_id="A")
    previous=dict(phase="calibration",workers=[dict(pid=25,exit_code=3221225477,indices=[0],attempt_id="A")])
    c.import_original_failure(previous,c.FinalRecords(tmp_path,frozen()),events)
    assert len(events.attempts("calibration",0))==1
    assert len(events.finished_attempts("calibration",0))==1


def test_missing_record_non_native_exit_preserves_query_handle(tmp_path):
    process=FakeProcess(code=0)
    worker=dict(process=process,phase="calibration",indices=[0],adopted=False,started_monotonic=time.monotonic(),log=None,attempt_id="A")
    with pytest.raises(RuntimeError,match="without native"):
        c.finish_worker(worker,c.FinalRecords(tmp_path,frozen()),c.EventLog(tmp_path/"events"))
    assert not process.closed


def test_development_gate_requires_original_success_and_exact_files(tmp_path):
    f=frozen();f["specifications"]["development"]=[dict(index=0,phase="development")]
    save(tmp_path,"development",0)
    audit=dict(success=True,policy_id="fixed-policy",coverage=dict(development=dict(exact_coverage=True)))
    c.atomic_json(tmp_path/"audit_development.json",audit)
    assert c.development_gate(tmp_path,f)["histories"]==1
    audit["success"]=False;c.atomic_json(tmp_path/"audit_development.json",audit)
    with pytest.raises(RuntimeError,match="development audit"):
        c.development_gate(tmp_path,f)


def test_development_gate_rejects_missing_record(tmp_path):
    f=frozen();f["specifications"]["development"]=[dict(index=0,phase="development")]
    c.atomic_json(tmp_path/"audit_development.json",dict(success=True,policy_id="fixed-policy",coverage=dict(development=dict(exact_coverage=True))))
    with pytest.raises(RuntimeError,match="development audit"):
        c.development_gate(tmp_path,f)
