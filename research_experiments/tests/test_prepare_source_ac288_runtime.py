"""Frozen-runtime isolation and real small grouped/Source compatibility checks."""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import warnings
import zipfile

import pytest

from research_experiments.rnj import prepare_source_ac288_runtime as prepare


def source_package(tmp_path):
    """An explicit directory override, independent of the original checkout."""
    path = tmp_path / "source" / "rnj_wzzt"
    files, _ = prepare.archived_source_inventory()
    for name, data in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return path


def test_observer_strip_preserves_ast_and_detects_a_math_change():
    source = prepare.archived_source_inventory()[0][prepare.CORE_RELATIVE].decode("utf-8")
    observed = prepare.add_passive_observer(source)
    assert "progress_callback: Callable[[dict], None] | None = None" in observed
    assert prepare.verify_observer_only(source, observed)
    assert '"terminal_groups": partition.groups' in observed
    changed = observed.replace("improvement_abs_tol: float = 1e-10", "improvement_abs_tol: float = 2e-10")
    assert changed != observed
    with pytest.raises(ValueError, match="algorithm AST"):
        prepare.verify_observer_only(source, changed)


def test_dependency_closure_rejects_missing_package_module():
    files = {"__init__.py": b"from rnj_wzzt import present\n",
             "present.py": b"from rnj_wzzt.missing import thing\n"}
    with pytest.raises(ValueError, match="closure is incomplete"):
        prepare.dependency_closure(files)


def test_complete_working_tree_snapshot_is_immutable_and_never_edits_live_source(tmp_path):
    source = source_package(tmp_path)
    before = prepare._read_inventory(source)
    runtime = tmp_path / "runtime"
    manifest = prepare.ensure_runtime(runtime, source)
    assert manifest["observer_changed_files"] == [prepare.CORE_RELATIVE]
    assert "solver_budgets.py" in manifest["source_sha256"]
    assert "observed_input.py" in manifest["source_sha256"]
    assert not manifest["dependency_closure"]["missing_local_imports"]
    assert prepare.verify_runtime(runtime) == manifest
    assert prepare.ensure_runtime(runtime, source) == manifest
    assert prepare._read_inventory(source) == before
    changed = runtime / "rnj_wzzt" / "solver_budgets.py"
    changed.write_bytes(changed.read_bytes() + b"\n# test tampering\n")
    tampered = changed.read_bytes()
    with pytest.raises(ValueError, match="frozen runtime changed"):
        prepare.ensure_runtime(runtime, source)
    assert changed.read_bytes() == tampered
    assert prepare._read_inventory(source) == before


def test_archive_inventory_matches_original_frozen_bytes_and_default_is_portable(tmp_path, monkeypatch):
    files, evidence = prepare.archived_source_inventory()
    assert len(files) == evidence["file_count"] == 34
    assert all(name.endswith(".py") for name in files)
    assert {name: prepare.sha256_bytes(data) for name, data in files.items()} == evidence["source_sha256"]
    original = prepare.DEFAULT_RUNTIME / "source_snapshot" / "rnj_wzzt"
    if original.exists():
        assert files == prepare._read_inventory(original)
    # Default preparation must not depend on the original machine's worktree.
    monkeypatch.setattr(prepare, "DEFAULT_SOURCE_PACKAGE", tmp_path / "absent_worktree")
    frozen = tmp_path / "from_archive"
    manifest = prepare.ensure_runtime(frozen)
    assert manifest["source_sha256"] == evidence["source_sha256"]
    assert manifest["runtime_sha256"] == evidence["frozen_runtime_evidence"]["runtime_sha256"]
    assert manifest["source_archive"]["sha256"] == evidence["archive_sha256"]
    before = (frozen / "runtime_manifest.json").read_bytes()
    assert prepare.ensure_runtime(frozen) == manifest
    assert (frozen / "runtime_manifest.json").read_bytes() == before
    override = prepare.ensure_runtime(tmp_path / "from_directory", source_package(tmp_path))
    assert override["source_sha256"] == manifest["source_sha256"]
    assert override["runtime_sha256"] == manifest["runtime_sha256"]


@pytest.mark.parametrize("member", ["../escape.py", "/rnj_wzzt/escape.py",
    "rnj_wzzt/../escape.py", "rnj_wzzt/./escape.py", "rnj_wzzt//escape.py",
    "rnj_wzzt\\escape.py", "C:/rnj_wzzt/escape.py", "rnj_wzzt/cache.pyc",
    "rnj_wzzt/__pycache__/cache.py", "rnj_wzzt/link.py"])
def test_archive_rejects_unsafe_members_before_reading_or_extracting(tmp_path, member):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        info = zipfile.ZipInfo(member)
        info.filename = member  # Preserve the hostile wire name on Windows.
        if member == "rnj_wzzt/link.py":
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
        output.writestr(info, b"../outside")
    with pytest.raises(ValueError, match="unsafe runtime archive member"):
        prepare.read_archive_inventory(archive)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["unsafe.zip"]


def test_archive_rejects_duplicate_members_and_hash_or_inventory_tampering(tmp_path):
    archive = tmp_path / "duplicate.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("rnj_wzzt/__init__.py", b"")
            output.writestr("rnj_wzzt/__init__.py", b"changed")
    with pytest.raises(ValueError, match="duplicate runtime archive member"):
        prepare.read_archive_inventory(archive)
    with pytest.raises(ValueError, match="archive SHA256"):
        prepare.read_archive_inventory(prepare.DEFAULT_ARCHIVE, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="archive inventory"):
        prepare.read_archive_inventory(prepare.DEFAULT_ARCHIVE, expected_inventory={})


SMALL_SOLVE = r'''
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from research_experiments.rnj.prepare_source_ac288_runtime import activate_runtime,verify_loaded_modules
manifest=activate_runtime(sys.argv[1])
from rnj_wzzt.estimation import laminar_l1_milp as core
from rnj_wzzt.estimation.unmetered_inputs import SourceInputs
rng=np.random.default_rng(20261018)
n,T=3,16
p=rng.uniform(.02,.35,(T,n));q=rng.uniform(-.05,.13,(T,n))
rp=np.array([.2,.3,.4]);xp=np.array([.08,.1,.12])
R=np.diag(rp)+.25*np.ones((n,n));X=np.diag(xp)+.15*np.ones((n,n))
index=pd.Index(range(T));labels=[301,302,303]
calls=[]; original=core._run_milp
def unchanged_backend(*args,**kwargs):
    result,diag=original(*args,**kwargs)
    calls.append({'status':diag.status,'solver':diag.solver,'limit':kwargs['time_limit']})
    return result,diag
core._run_milp=unchanged_backend
answers=[]
for enabled in (False,True):
    uP=rng.uniform(.02,.18,T) if enabled else np.zeros(T)
    uQ=rng.uniform(-.06,.09,T) if enabled else np.zeros(T)
    target=p@R.T+q@X.T+uP[:,None]*.25+uQ[:,None]*.15
    scenarios=[]; sources=[]
    for begin,end in ((0,6),(6,11),(11,16)):
        block=slice(begin,end)
        scenarios.append({'P_terminal':pd.DataFrame(p[block],index=index[block],columns=labels),
                          'Q_terminal':pd.DataFrame(q[block],index=index[block],columns=labels),
                          'drop_target':pd.DataFrame(target[block],index=index[block],columns=labels)})
        sources.append(SourceInputs(pd.Series(uP[block],index=index[block]),
                                   pd.Series(uQ[block],index=index[block]),'independent_linear'))
    events=[]
    def callback(state):
        events.append({'event':state['event'],'source_enabled':state['source_enabled'],
                       'groups':state['terminal_groups'],'selected_index':state['selected_path_index']})
        # Mutating the observer's copy must never affect the active search.
        state['path'][0].r_values[:]=999.
        if state['path'][0].source_h is not None: state['path'][0].source_h[:]=9.
    options={'progress_callback':callback} if manifest['observer_enabled'] else {}
    fitted=core.fit_laminar_l1_sensitivity(scenarios,source_inputs=sources,
        initial_supports=[(0,),(1,),(2,)],terminal_groups=[(0,1,2)],
        r_upper_bound=2.,x_upper_bound=2.,time_limit=10.,solver='gurobi',**options)
    assert np.max(abs(fitted.r_matrix-R))<1e-7
    assert np.max(abs(fitted.x_matrix-X))<1e-7
    assert fitted.validation_mae is None
    assert fitted.selected_path_index==len(fitted.path)-1
    assert fitted.terminal_groups==((0,1,2),)
    if enabled:
        assert len(fitted.source_provenance)==3
        assert fitted.source_h.tolist()==[0.,0.,0.,1.]
        assert np.max(abs(fitted.source_response_r-.25))<1e-7
        assert np.max(abs(fitted.source_response_x-.15))<1e-7
    else: assert fitted.source_h is None
    answers.append({'source_enabled':enabled,'supports':fitted.support_indices,
        'R':fitted.r_matrix.tolist(),'X':fitted.x_matrix.tolist(),
        'h':None if fitted.source_h is None else fitted.source_h.tolist(),
        'train_mae':fitted.train_mae,'selected_index':fitted.selected_path_index,
        'stop':fitted.stop_reason,'path_length':len(fitted.path),'groups':fitted.terminal_groups,
        'events':events})
origins=verify_loaded_modules(sys.argv[1])
assert origins and all(str(Path(sys.argv[1]).resolve()) in value for value in origins.values())
print(json.dumps({'answers':answers,'calls':calls,'origins':origins}))
'''


def test_real_gurobi_grouped_source_and_no_source_match_pristine_runtime(tmp_path):
    pytest.importorskip("gurobipy")
    original, observed = tmp_path / "pristine", tmp_path / "observed"
    prepare.ensure_runtime(original, observer=False)
    prepare.ensure_runtime(observed, observer=True)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    outputs = []
    for runtime in (original, observed):
        result = subprocess.run([sys.executable, "-B", "-c", SMALL_SOLVE, str(runtime)],
                                cwd=prepare.ROOT, env=env, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append(json.loads(result.stdout.strip().splitlines()[-1]))
    before, after = outputs
    for a, b in zip(before["answers"], after["answers"]):
        assert a["events"] == []
        assert b["events"][0]["event"] == "initialized"
        assert b["events"][-1]["event"] == "stopped"
        assert all(e["source_enabled"] == b["source_enabled"] for e in b["events"])
        assert all(e["groups"] == [[0, 1, 2]] for e in b["events"])
        for key in ("source_enabled", "supports", "h", "selected_index", "stop", "path_length", "groups"):
            assert a[key] == b[key]
        assert a["train_mae"] == pytest.approx(b["train_mae"], abs=1e-10)
        import numpy as np
        np.testing.assert_allclose(a["R"], b["R"], atol=1e-9)
        np.testing.assert_allclose(a["X"], b["X"], atol=1e-9)
    assert before["calls"] == after["calls"]
    assert all(c["solver"] == "gurobi" and c["limit"] == 10. for c in after["calls"])
