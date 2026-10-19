import importlib.util
from pathlib import Path
import numpy as np
import json
import zipfile

spec=importlib.util.spec_from_file_location("ranking_experiment",Path(__file__).with_name("experiment.py"))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_static_candidate_tree_equipment():
    candidates=module.catalog(); assert len(candidates)==10
    original={e["branch_id"]:e for e in candidates[0]["branches"]}
    assert [c["edit"].get("leaf") for c in candidates[6:]]==[18,22,25,33]
    signatures=[]
    for c in candidates:
        module.rooted(c["branches"])
        signatures.append(sorted(tuple(sorted((e["from_bus"],e["to_bus"]))) for e in c["branches"]))
        for e in c["branches"]:
            if e["branch_id"] in original:
                old=original[e["branch_id"]]
                assert (e["r_ohm"],e["x_ohm"])==(old["r_ohm"],old["x_ohm"])
    assert len({str(s) for s in signatures})==10


def test_rank_inversion_and_common_bias():
    a=[1.,3.,4.]; names=["true","b","c"]
    r=module.ranking(a,np.array(a)+100.,names)
    assert r["top1_set_equal"] and r["inversions"]==0
    assert all(abs(p["gap_error"])<1e-12 for p in r["pairs"])
    r=module.ranking(a,[5.,2.,4.],names)
    assert r["inversions"]==2 and r["true_pair_inversions"]==2
    assert r["AC_winner_regret"]==2.


def test_ties_and_unavailable_retained():
    r=module.ranking([1.,1.,2.],[1.,1.+1e-9,3.],["true","b","c"])
    assert r["AC_top"]==r["proposed_top"]==["true","b"]
    assert r["pairs"][0]["AC_tie"] and not r["pairs"][0]["inversion"]
    assert not module.ranking([1.,float("nan")],[1.,2.],["true","b"])["available"]


def test_context_preserves_physical_observations():
    for candidate in module.catalog():
        ctx=module.make_context(candidate)
        assert ctx["ids"]==list(range(2,34))
        assert len(ctx["network"]["branches"])==32
        z=np.r_[ctx["p0"],ctx["q0"]]/ctx["network"]["base_kw"]*.4
        y,_=module.base.response(ctx,"ac",z)
        independent,_=module.base.independent_output(ctx,z)
        np.testing.assert_allclose(y,independent,atol=1e-10,rtol=0)


def test_fitter_input_boundary(tmp_path):
    archive=tmp_path/"input.zip"
    raw=dict(state="evening",power_sigma=.002,seed=2026101801,clean=False,
        zobs=[1.],yobs=[2.],sigma_z=[.1],sigma_y=[.2],z_true=[99.],y_true=[100.],
        methods={"oracle":"do not read"})
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("sample.json",json.dumps(raw))
    record=module.observations(archive)[0]
    assert "z_true" not in record and "y_true" not in record and "methods" not in record
    assert record["zobs"]==[1.] and record["yobs"]==[2.]
