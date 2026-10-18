"""Declared eight-fit scope and immutable source/saved-start checks."""
from copy import deepcopy
from pathlib import Path
import pytest

from . import benchmark as b
from . import crosswarm_audit as cw


def test_crosswarm_declares_exact_eight_fits_and_source_snapshot(tmp_path):
    source = tmp_path/"pilot"; tasks = []
    for name in cw.NAMES:
        for method in cw.METHODS:
            task = dict(network="case33", candidate_name=name, prefix=24, repeat=0,
                algorithm=method, seed=1, start="prefix_linear", inner_tol=1e-10,
                package_sha256="fixture", history_content_sha256="fixture")
            task["task_id"] = b._task_identity(task)[:20]; tasks.append(task)
            b._write(source/"tasks"/task["task_id"]/"worker_result.json",
                dict(status="returned", attained_state=dict(r=[1.], x=[.5], source_p=[0.], source_q=[0.])))
    whitelist = tmp_path/"whitelist.json"
    payload = dict(schema=b.PACKAGE_SCHEMA, items={}, fits_started=0)
    payload["payload_sha256"] = cw.sha256(b._bytes(payload)).hexdigest(); b._write(whitelist, payload)
    b._write(source/"completed.json", dict(count=4)); b._write(source/"plan.json", dict(tasks=tasks))
    plan = cw.prepare(source, whitelist, tmp_path/"cross")
    assert len(plan["tasks"]) == 8
    assert len({task["task_id"] for task in plan["tasks"]}) == 8
    assert plan["reference_settings"]["ftol"] is None
    assert plan["reference_settings"]["xtol"] is None
    assert plan["block_settings"]["stationarity_tol"] == 1e-5
    for name in cw.NAMES:
        for warm in cw.METHODS:
            pair = [task for task in plan["tasks"] if task["candidate_name"] == name and task["warm_method"] == warm]
            assert pair[0]["endpoint_sha256"] == pair[1]["endpoint_sha256"]
    assert (tmp_path/"cross"/"source_snapshot"/"manifest.json").exists()


def test_changed_kernel_is_retained_before_running_optimizer(tmp_path):
    task = dict(kernel_sha256={"research_experiments/mobile_ev_mle/ac_block_fit/crosswarm_audit.py": "wrong"})
    result = cw.worker(tmp_path/"not_read", task)
    assert result["status"] == "exception"
    assert "kernel changed" in result["exception"]
