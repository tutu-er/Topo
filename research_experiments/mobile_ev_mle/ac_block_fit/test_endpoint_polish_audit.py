"""Remaining-budget accounting and immutable failed-endpoint denominators."""
from pathlib import Path
import numpy as np

from . import benchmark as b
from . import candidate_audit as ca
from . import endpoint_polish_audit as ep
from .test_benchmark import source_fixture


def tiny_source(tmp_path, *, original_success=True, original_nfev=12, original_budget=50):
    import gzip
    source, raw = source_fixture(tmp_path)
    raw["config"] = dict(base_kw=100.)
    raw["fit_options"]["max_nfev"] = original_budget
    path = source/"history_000_paper11_none.json.gz"; path.write_bytes(gzip.compress(b._bytes(raw)))
    whitelist = tmp_path/"candidates.json.gz"
    package = ca.prepare_inputs(source, whitelist, networks=("paper11",), count=1)
    task = ca.tasks(package, prefixes=(2,))[0]
    item = package["items"]["paper11"]
    _, data, _, _, _, _ = b._reconstruct(item, task)
    state = dict(raw["planning"]["history"]["candidates"][task["candidate_name"]]["state"])
    state["source_p"] = np.asarray(state["source_p"])[:data.t]
    state["source_q"] = np.asarray(state["source_q"])[:data.t]
    pilot = tmp_path/"pilot"
    worker_path = pilot/"tasks"/task["task_id"]/"worker_result.json"
    worker = dict(status="returned", task=task, task_sha256=b._task_identity(task),
        attained_state=state, solver_success=original_success,
        fit=dict(nfev=original_nfev, status=2 if original_success else 0, loss=0., success=original_success),
        fit_options=item["fit_options"], audit=dict(reference_scaled_projected_gradient=0.))
    b._write(worker_path, worker)
    b._write(worker_path.with_name("result.json"), dict(task=task, worker=worker, status="returned"))
    b._write(pilot/"plan.json", dict(tasks=[task]))
    b._write(pilot/"completed.json", dict(count=1))
    return pilot, whitelist, worker_path


def test_plan_preserves_hash_and_subtracts_consumed_budget(tmp_path):
    pilot, whitelist, old_path = tiny_source(tmp_path, original_nfev=12, original_budget=50)
    old_hash = b._digest(old_path)
    plan = ep.prepare_plan(pilot, whitelist, tmp_path/"endpoint_plan.json")
    task = plan["tasks"][0]
    assert task["remaining_nfev"] == 38
    assert plan["original_task_denominator"] == 1
    assert task["endpoint_sha256"] == old_hash
    assert b._digest(old_path) == old_hash
    assert not plan["original_failure_can_be_promoted"]


def test_original_failure_cannot_be_promoted_even_if_stationary(tmp_path):
    pilot, whitelist, _ = tiny_source(tmp_path, original_success=False, original_nfev=50, original_budget=50)
    plan = ep.prepare_plan(pilot, whitelist, tmp_path/"endpoint_plan.json")
    seen = {}
    def no_trials(problem, vector, **kwargs):
        seen.update(kwargs)
        residual = problem.evaluate(vector)
        return dict(x=vector, fun=residual, success=True, final_common_pg=0.,
                    trial_nfev=0, trace=[], initial_F=.5*float(residual@residual),
                    final_F=.5*float(residual@residual))
    result = ep.worker(whitelist, plan["tasks"][0], polisher=no_trials)
    assert result["status"] == "returned", result
    assert seen["remaining_nfev"] == 0
    assert not result["solver_success"]
    assert result["original_failure_retained"]
    assert result["total_optimization_nfev"] == 50


def test_original_endpoint_hash_change_is_retained_exception(tmp_path):
    pilot, whitelist, old_path = tiny_source(tmp_path)
    plan = ep.prepare_plan(pilot, whitelist, tmp_path/"endpoint_plan.json")
    old_path.write_bytes(old_path.read_bytes()+b" ")
    result = ep.worker(whitelist, plan["tasks"][0], polisher=lambda *a, **k: None)
    assert result["status"] == "exception"
    assert "hash changed" in result["exception"]
