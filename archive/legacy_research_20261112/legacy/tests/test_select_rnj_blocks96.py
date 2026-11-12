from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research_experiments.rnj.select_rnj_blocks96 import select_rnj_blocks96


@dataclass(frozen=True)
class Prepared:
    p: tuple
    q: tuple
    time_indices: tuple
    provenance: tuple


def scenario():
    t = np.arange(12)
    p = pd.DataFrame({11: t, 12: t*t+1, 13: t*3+2, 14: t*5+3}, index=pd.Index(t, name="t"))
    return {"name": "full", "P_terminal": p, "Q_terminal": p*.2, "drop_target": p*.1}


def controlled_runtime(*, pair=True, internal_count=0, full_fine=False,
                       full_identifiable=True, invalid_bootstraps=()):
    """Hand prescribed RNJ trees distinguish coarse peers from finer layers."""
    a = frozenset([11, 12] if pair else [11, 12, 13])
    outer = frozenset([11, 12, 13]) if pair else None
    child = frozenset([11, 12])
    fit_calls, sampled_indices = [], []
    def align(scenarios):
        return [{"name": s["name"], **{k: s[k].loc[s["P_terminal"].index].copy()
                for k in ("P_terminal", "Q_terminal", "drop_target")}} for s in scenarios]
    def fit(scenarios, *, diagnostics, source_inputs=None):
        name = scenarios[0]["name"]
        fit_calls.append(name)
        if source_inputs is not None:
            p = scenarios[0]["P_terminal"][11].to_numpy()
            np.testing.assert_array_equal(source_inputs.p[0], p+100)
            np.testing.assert_array_equal(source_inputs.q[0], p*2-30)
            assert source_inputs.time_indices[0].equals(scenarios[0]["P_terminal"].index)
        index = int(name[6:]) if name.startswith("sample") else None
        diagnostics["source_adjusted_identifiable"] = full_identifiable if index is None else index not in invalid_bootstraps
        return np.eye(4), np.eye(4), 1., 1.
    def tree(scenarios, terminals, root, *, tolerance_factor, sensitivity_fit, tree_details=None):
        name = scenarios[0]["name"]
        fine = tolerance_factor < .1
        clades = {a} | ({outer} if outer is not None else set())
        if not pair and fine:
            index = int(name[6:]) if name.startswith("sample") else None
            if index is None and full_fine or index is not None and index < internal_count:
                clades.add(child)
        if tree_details is not None: tree_details.update(peers={a})
        return clades, 1., 1.
    def sample(scenarios, rng, block_length, *, return_indices=False):
        index = rng.integers(0, len(scenarios[0]["P_terminal"]), size=len(scenarios[0]["P_terminal"]))
        name = "sample"+str(len(sampled_indices))
        sampled_indices.append(index.tolist())
        result = [{"name": name, **{k: s[k].iloc[index].reset_index(drop=True)
                   for k in ("P_terminal", "Q_terminal", "drop_target")}} for s in scenarios]
        return (result, tuple(index.copy() for _ in scenarios)) if return_indices else result
    def choose(candidates, confidence, maximum_count):
        # Independent simple ordering for this fixture's at-most-one eligible
        # candidate. Selection internals remain the frozen runtime's job.
        return sorted(candidates, key=lambda c: tuple(sorted(c)))[:maximum_count]
    def copy_amplitudes(source, *, expected_lengths, time_indices):
        ps, qs = [], []
        for p, q, labels, reference in zip(source.p, source.q, source.time_indices, time_indices):
            positions = labels.get_indexer(reference)
            if (positions < 0).any(): raise ValueError("misaligned Source labels")
            ps.append(np.asarray(p)[positions].copy()); qs.append(np.asarray(q)[positions].copy())
        if all(not np.any(p) and not np.any(q) for p, q in zip(ps, qs)): return None
        return tuple(ps), tuple(qs)
    pipeline = SimpleNamespace(align_scenarios=align, fit_projected_sensitivity=fit, _rx75_rnj_clades=tree,
        _moving_block_bootstrap_copy=sample, _select_disjoint=choose, PreparedSourceInputs=Prepared,
        prepare_source_inputs=lambda scenarios, sources: sources)
    return pipeline, SimpleNamespace(_copy_source_amplitudes=copy_amplitudes), fit_calls, sampled_indices, a


def call(runtime, source=None, **kwargs):
    pipeline, source_module, _, _, _ = runtime
    diagnostics = {}
    result = select_rnj_blocks96([scenario()], [11, 12, 13, 14], 1,
        runtime_pipeline=pipeline, runtime_source=source_module, source_inputs=source,
        diagnostics=diagnostics, **kwargs)
    return result, diagnostics


def test_two_terminal_peer_eligible_ordinary_outer_clade_is_diagnostic_only():
    runtime = controlled_runtime(pair=True)
    (_, peer_confidence, selected, _, _), diagnostics = call(runtime, bootstrap_replicates=8)
    assert selected == [frozenset([11, 12])] and peer_confidence[frozenset([11, 12])] == 1
    outer = next(r for r in diagnostics["candidate_records"] if r["clade"] == [11, 12, 13])
    assert outer["ordinary_clade_frequency"] == 1 and outer["atomic_peer_frequency"] == 0
    assert not outer["eligible"] and "not_complete_direct_terminal_peer" in outer["rejection_reasons"]
    assert len(runtime[2]) == 9  # one full fit plus eight bootstrap fits, not sixteen
    assert diagnostics["parameters"]["minimum_block_size"] == 2
    assert diagnostics["parameters"]["maximum_subclade_frequency"] == .15
    assert len(diagnostics["bootstrap_records"]) == 8
    assert all(row["peers"] == [[11, 12]] and row["coarse_clades"] == [[11, 12], [11, 12, 13]]
               and row["identifiable"] and row["valid_for_frequency_counts"] for row in diagnostics["bootstrap_records"])


@pytest.mark.parametrize("count,accepted", [(15, True), (16, False)])
def test_internal_frequency_point15_is_inclusive_and_point16_rejected(count, accepted):
    runtime = controlled_runtime(pair=False, internal_count=count)
    (_, _, selected, _, _), diagnostics = call(runtime)
    record = next(r for r in diagnostics["candidate_records"] if r["is_full_peer"])
    assert record["subclade_frequency"] == count/100
    assert bool(selected) is accepted
    assert ("internal_subclade_frequency_above_threshold" in record["rejection_reasons"]) is (not accepted)


def test_full_fine_veto_still_applies_even_when_other_thresholds_pass():
    runtime = controlled_runtime(pair=False, full_fine=True)
    (_, _, selected, _, _), diagnostics = call(runtime, bootstrap_replicates=5)
    assert selected == [] and diagnostics["eligible"] == set()
    record = next(r for r in diagnostics["candidate_records"] if r["is_full_peer"])
    assert record["subclade_frequency"] == 0 and record["atomic_peer_frequency"] == 1
    assert record["rejection_reasons"] == ["full_data_fine_subclade_veto"]


def test_source_labels_and_both_amplitudes_resample_with_observations_without_mutation():
    s = scenario(); index = s["P_terminal"].index[::-1]
    p, q = index.to_numpy()+100., index.to_numpy()*2.-30
    source = Prepared((p.copy(),), (q.copy(),), (index.copy(),), ("externally_known",))
    first = controlled_runtime()
    result1, diagnostics1 = call(first, source, bootstrap_replicates=7, seed=41)
    second = controlled_runtime()
    result2, diagnostics2 = call(second, source, bootstrap_replicates=7, seed=41)
    assert result1 == result2 and diagnostics1 == diagnostics2
    assert first[3] == second[3] and len(first[2]) == 8
    np.testing.assert_array_equal(source.p[0], p)
    np.testing.assert_array_equal(source.q[0], q)
    assert source.time_indices[0].equals(index)


def test_zero_source_has_same_screen_and_empty_selection_is_retained():
    s = scenario(); zeros = np.zeros(12)
    zero_source = Prepared((zeros,), (zeros,), (s["P_terminal"].index,), ("known_zero",))
    result0, diagnostic0 = call(controlled_runtime(), bootstrap_replicates=5, minimum_block_size=3)
    result1, diagnostic1 = call(controlled_runtime(), zero_source, bootstrap_replicates=5, minimum_block_size=3)
    assert result0 == result1 and diagnostic0 == diagnostic1
    assert result1[2] == [] and diagnostic1["empty_selection_retained"]
    assert diagnostic1["eligible"] == set()


@pytest.mark.parametrize("invalid", ["full", "all_bootstraps"])
def test_rank_gates_prevent_freezing_without_masking_invalid_replicates(invalid):
    runtime = controlled_runtime(full_identifiable=invalid != "full", invalid_bootstraps=range(5) if invalid == "all_bootstraps" else ())
    (_, confidence, selected, _, _), diagnostics = call(runtime, bootstrap_replicates=5)
    assert not selected
    assert diagnostics["valid_bootstrap_replicates"] == (0 if invalid == "all_bootstraps" else 5)
    reasons = diagnostics["candidate_records"][0]["rejection_reasons"]
    assert ("source_adjusted_design_unidentifiable" if invalid == "full" else "no_identifiable_bootstrap_replicates") in reasons
    if invalid == "all_bootstraps": assert confidence[frozenset([11, 12])] == 0
    if invalid == "all_bootstraps":
        assert all(not r["valid_for_frequency_counts"] and r["peers"] is None and r["coarse_clades"] is None
                   for r in diagnostics["bootstrap_records"])


def test_reused_full_fit_skips_one_fit_but_preserves_bootstrap_work():
    runtime = controlled_runtime()
    (_, _, selected, _, _), diagnostics = call(runtime, bootstrap_replicates=5,
        sensitivity_fit=(np.eye(4), np.eye(4), 1., 1.))
    assert selected and len(runtime[2]) == 5 and "full" not in runtime[2]
    assert diagnostics["supplied_sensitivity_fit_reused"]


def test_reused_source_fit_cannot_bypass_adjusted_design_rank_gate():
    runtime = controlled_runtime()
    s = scenario(); t = s["P_terminal"].index.to_numpy()
    # The hand-written P/Q columns are polynomial/collinear, so supplying a
    # plausible R/X pair must not permit freezing under the Source rank gate.
    source = Prepared((t+100.,), (t*2.-30,), (s["P_terminal"].index,), ("known",))
    (_, _, selected, _, _), diagnostics = call(runtime, source, bootstrap_replicates=3,
        sensitivity_fit=(np.eye(4), np.eye(4), 1., 1.))
    assert selected == [] and not diagnostics["source_adjusted_identifiable"]
    assert diagnostics["source_fit"]["projected_design_rank"] < 8
    assert len(runtime[2]) == 3 and "full" not in runtime[2]


@pytest.mark.parametrize("kwargs", [{"minimum_block_size": 1}, {"maximum_subclade_frequency": 1.1},
                                    {"fine_tolerance_ratio": 0}, {"bootstrap_replicates": True}, {"confidence_threshold": "bad"}])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError): call(controlled_runtime(), **kwargs)


def test_small_real_frozen_runtime_bootstrap_source_and_shared_pair():
    root = Path(__file__).resolve().parents[1]
    runtime = root/"outputs/source_ac288_runtime_20261018"
    if not (runtime/"rnj_wzzt/pipeline.py").exists(): pytest.skip("frozen runtime not prepared")
    # A separate process prevents runtime imports from contaminating other tests.
    code = r'''
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from research_experiments.rnj.select_rnj_blocks96 import select_rnj_blocks96
assert not any(n=='rnj_wzzt' or n.startswith('rnj_wzzt.') for n in sys.modules)
sys.path.insert(0,sys.argv[1])
from rnj_wzzt import pipeline
from rnj_wzzt.estimation import laminar_l1_source as source_module
rng=np.random.default_rng(27); labels=[11,12,13,14]; index=pd.Index(range(96),name='t')
p=rng.uniform(.001,.04,(96,4));q=rng.uniform(-.01,.03,(96,4))
r=np.eye(4)*.03;x=np.eye(4)*.02;r[:2,:2]+=.4;x[:2,:2]+=.2
up=.03+.01*np.sin(np.arange(96)/7);uq=.02*np.cos(np.arange(96)/9)
gR=np.array([.4,.4,0,0]);gX=np.array([.2,.2,0,0])
y=p@r+q@x+up[:,None]*gR+uq[:,None]*gX
s={'name':'tiny96','P_terminal':pd.DataFrame(p,index=index,columns=labels),
   'Q_terminal':pd.DataFrame(q,index=index,columns=labels),'drop_target':pd.DataFrame(y,index=index,columns=labels)}
u=pipeline.PreparedSourceInputs((up,),(uq,),(index,),('synthetic_linear',))
d={};result=select_rnj_blocks96([s],labels,1,runtime_pipeline=pipeline,runtime_source=source_module,
 source_inputs=u,sensitivity_fit=(r,x,1.,1.),diagnostics=d,bootstrap_replicates=8)
assert result[2]==[frozenset([11,12])],result[2]
assert d['valid_bootstrap_replicates']==8 and d['source_fit']['projected_design_rank']==8
assert d['supplied_sensitivity_fit_reused'] and d['candidate_records'][0]['size']==2
assert d['full_tree']['edges'] and d['full_fine_tree']['edges']
assert len(d['bootstrap_records'])==8 and all(len(b['resample_indices'][0])==96 for b in d['bootstrap_records'])
print(json.dumps({'selected':[sorted(c) for c in result[2]],'valid':d['valid_bootstrap_replicates']}))
'''
    process = subprocess.run([sys.executable, "-c", code, str(runtime)], cwd=root, text=True,
                             capture_output=True, timeout=90)
    assert process.returncode == 0, process.stdout+process.stderr
    assert json.loads(process.stdout.strip()) == {"selected": [[11, 12]], "valid": 8}
