"""Pure fixtures; no AC, fitter, optimizer, strategy, or physical simulation."""
from copy import deepcopy
import gzip
import hashlib
import json
import math

import pytest

from . import ev_dynamic_collect as collector
from .ev_rx_analyze import METHODS, DOMAINS, analyze, markdown, stats


FAMILIES=("star","clade_01","clade_02","clade_12")
SUPPORTS=[[0,1,2],[0],[1],[2]]


def _write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,sort_keys=True).encode()
    path.write_bytes(gzip.compress(data,mtime=0) if path.suffix==".gz" else data)


def _freeze(directory,domain,seeds):
    config=dict(select_modes=list(METHODS[:-1]),training_domain=domain,development_seeds=seeds,
                mean_policy_ablation=False,diagnostics=False)
    sources={"fixture_only.py":"a"*64}
    freeze=dict(config=config,source_sha256=sources,
        policy_id=collector.object_hash(dict(config=config,source_sha256=sources)))
    _write(directory/"freeze.json",freeze)
    return freeze


def _fit(name,r,x,loss,*,fixed=False):
    return dict(name=name,candidate=name,supports=SUPPORTS,r=r,x=x,loss=loss,success=True,
                diagnostics=dict(line_parameters_fixed=fixed,local_projected_shared_singular_values=[1.,.01]))


def _trial(freeze,seed,*,cone=False):
    r=[.0105 if cone else .011]*4;x=[.0022 if cone else .0024]*4
    tr=[_fit(name,r,x,1.) for name in FAMILIES]
    val=[_fit(name,r,x,1.,fixed=True) for name in FAMILIES]
    pairs=[dict(name=a+"->"+b,a=a,b=b,success=True,added=0. if a=="star" and b=="clade_01" else float(i+j+1))
           for i,a in enumerate(FAMILIES) for j,b in enumerate(FAMILIES) if a!=b]
    pred=dict(success=True,pairs=pairs)
    raw=dict(completed=True,gap=.1,target_gap=.2,objective_lower=1.,objective_upper=1.1,
        required_pairs=[p["name"] for p in pairs],status="gap_reached",inner_qp_failures=0,qp_calls=2)
    methods=[];selection={};predictions={}
    for name in METHODS:
        extra=(2. if cone else 0.)+(3. if name.startswith("nearest_") else 0.)
        future=[_fit(f,r,x,3. if f=="star" else 9.+extra,fixed=True) for f in FAMILIES]
        losses={f["candidate"]:f["loss"]+1 for f in future}
        methods.append(dict(name=name,power_kw=[[2.,0.,0.],[2.,0.,0.]],status="complete",
            physical=dict(p=[[.1,.1,.1],[.1,.1,.1]],v=[[1.,1.,1.],[1.,1.,1.]]),
            operation=dict(success=True,grid_energy_kwh=3.,throughput_kwh=7.5,driving_kwh=0.,worker_minutes=0.,moves=0),
            loss_by_candidate=losses,M=min(v for k,v in losses.items() if k!="star")-losses["star"],
            clean_future_M=5.,clean_future_complete=True,clean_future_scope="same noisy validation, future noise removed",
            noisy=dict(all_complete=True,failures={},fits=future),
            noiseless=dict(all_complete=True,failures={},fits=future)))
        predictions[name]=deepcopy(pred)
        selection[name]=dict(selected_origin="MILP",selected_is_new_MILP_proposal=name.startswith("nearest_"),
            selected_score=2.,initial_best_score=1.,raw=deepcopy(raw),raw_prediction=deepcopy(pred),failures=[])
        if name.startswith("square_"):
            selection[name]=dict(selected_origin="exchange_evaluation:1",Q=2.,best_initial_Q=1.,
                exchange=deepcopy(raw),objective_lower=1.,objective_upper=1.1,gap=.1,completed=True,
                exchange_configuration=dict(absolute_gap=.2,relative_gap=0.),failures=[])
    error=.001 if cone else .002
    response={name:dict(success=True,channels={c:dict(RMSE=error,relative_L2=error/.01,
        true_increment=[.01,.01],fitted_increment=[.01+error,.01+error]) for c in ("v","master_p","master_q")})
        for name in ("constant","highlow")}
    return dict(freeze,seed=seed,family_index=0,true_name="star",source_unchanged=True,status="complete",
        candidates=[dict(name=n,supports=SUPPORTS) for n in FAMILIES],candidate_miss=False,
        training=dict(p=[[.1,.12,.08]],q=[[.03,.04,.02]]),validation=dict(p=[[.1,.12,.08]]),
        training_meter=dict(v_sigma=.001),validation_meter=dict(v_sigma=.001),future_meter=dict(v_sigma=.001),
        background=dict(p_kw=[[2.,2.4,1.6]]),truth=dict(r=[.01]*4,x=[.002]*4,supports=SUPPORTS),
        training_result=dict(all_complete=True,failures={},fits=tr),
        validation_result=dict(all_complete=True,failures={},fits=val),
        training_parameter_error=dict(R=.05 if cone else .1,X=.1 if cone else .2),
        line_response_diagnostics=response,methods=methods,
        selection=dict(methods=selection,predictions=predictions,model_task_count=12,
            uses_truth=False,uses_validation=False,uses_future_noise=False),
        seconds=10.,actual_solver_calls=dict(AC_batches=20,AC_snapshots=100,QP_calls=12),
        validation_used_by_selector=False,future_noise_used_by_selector=False)


def _directories(tmp_path,seeds=(1,2)):
    box,cone=tmp_path/"box",tmp_path/"cone"
    return box,cone,_freeze(box,"box",list(seeds)),_freeze(cone,"cone",list(seeds))


def test_empty_running_directories_keep_planned_19_method_slots(tmp_path):
    box,cone,_,_=_directories(tmp_path)
    result=analyze(box,cone)
    assert result["status"]=="partial_or_failed"
    assert result["paired"]["paired_histories"]==0
    for group in result["groups"].values():
        assert group["planned_method_slots"]==38
        assert group["pending_method_slots"]==38 and group["failed_method_slots"]==0
        assert len(group["methods"])==19
        assert group["methods"]["nearest_v2g_mobile"]["M"]["total"]==2
    assert "pending" in markdown(result)


def test_partial_pair_parameter_response_and_nearest_contrasts(tmp_path):
    box,cone,fb,fc=_directories(tmp_path)
    _write(box/"trial_1.json.gz",_trial(fb,1));_write(cone/"trial_1.json.gz",_trial(fc,1,cone=True))
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for d in (box,cone) for p in d.iterdir()}
    result=analyze(box,cone)
    assert result["status"]=="partial_or_failed"
    pair=result["paired"]
    assert pair["paired_histories"]==1 and pair["pending_histories"]==1
    assert pair["R_error_delta"]["mean"]==pytest.approx(-.05)
    assert pair["X_error_delta"]["mean"]==pytest.approx(-.1)
    assert pair["common_response"]["constant"]["v"]["mean"]==pytest.approx(-.001)
    assert pair["methods"]["nearest_v2g_fixed"]["delta_M"]["mean"]==pytest.approx(2.)
    for group in result["groups"].values():
        assert group["complete_method_slots"]==19 and group["pending_method_slots"]==19
        for domain in DOMAINS:
            s=group["nearest_contrasts"][domain+"_vs_square"]["delta_M"]
            assert s["mean"]==3 and s["valid"]==1 and s["total"]==2
        s=group["methods"]["nearest_v2g_mobile"]
        assert s["new_MILP"]==s["substantive_new_MILP"]==1
        assert s["proxy_completed"]==s["selected_AC_all_pairs"]==s["proxy_all_pairs_required"]==1
        assert s["near_zero_pairs"]["mean"]==1
    assert result["analyzer"]["new_fit_calls"]==result["analyzer"]["new_AC_calls"]==0
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before}


def test_missing_method_failure_and_missing_pair_are_not_dropped(tmp_path):
    box,cone,fb,fc=_directories(tmp_path,seeds=(1,))
    a,b=_trial(fb,1),_trial(fc,1,cone=True)
    b["methods"]=[m for m in b["methods"] if m["name"]!="square_v2g_mobile"]
    b["status"]="partial_failure"
    b["selection"]["predictions"]["nearest_charge_fixed"]["pairs"].pop()
    _write(box/"trial_1.json.gz",a);_write(cone/"trial_1.json.gz",b)
    result=analyze(box,cone);g=result["groups"]["cone"]
    assert g["observed_method_slots"]==19 and g["complete_method_slots"]==18 and g["failed_method_slots"]==1
    assert g["methods"]["square_v2g_mobile"]["failed"]==1
    assert g["nearest_contrasts"]["v2g_mobile_vs_square"]["delta_M"]["valid"]==0
    assert g["methods"]["nearest_charge_fixed"]["selected_AC_all_pairs"]==0
    assert result["status"]=="partial_or_failed"


def test_pairing_mismatch_and_corrupted_observation_are_preserved(tmp_path):
    box,cone,fb,fc=_directories(tmp_path,seeds=(1,))
    a,b=_trial(fb,1),_trial(fc,1,cone=True)
    b["training"]["p"][0][0]+=.01
    _write(box/"trial_1.json.gz",a);_write(cone/"trial_1.json.gz",b)
    result=analyze(box,cone)
    assert result["paired"]["pairing_failures"]==1
    assert result["paired"]["methods"]["nearest_v2g_mobile"]["delta_M"]["valid"]==0
    # A present but corrupt record is a failed input, never a not-yet-arrived one.
    (cone/"trial_1.json.gz").write_bytes(b"partial gzip")
    result=analyze(box,cone)
    assert result["groups"]["cone"]["input_errors"]
    assert result["groups"]["cone"]["pending_method_slots"]==0
    assert result["groups"]["cone"]["failed_method_slots"]==19
    assert result["groups"]["cone"]["input_failed_seeds"]==["1"]


def test_common_response_needs_same_curve_and_verified_increment_arithmetic(tmp_path):
    box,cone,fb,fc=_directories(tmp_path,seeds=(1,))
    a,b=_trial(fb,1),_trial(fc,1,cone=True)
    next(m for m in b["methods"] if m["name"]=="constant")["power_kw"][0][0]+=1
    b["line_response_diagnostics"]["highlow"]["channels"]["v"]["RMSE"]+=1
    _write(box/"trial_1.json.gz",a);_write(cone/"trial_1.json.gz",b)
    result=analyze(box,cone)
    assert result["paired"]["common_response"]["constant"]["v"]["valid"]==0
    assert result["paired"]["common_response"]["highlow"]["v"]["valid"]==0
    assert not result["groups"]["cone"]["records"]["1"]["response"]["highlow"]["success"]


def test_complete_batch_and_tolerance_counts(tmp_path):
    box,cone,fb,fc=_directories(tmp_path,seeds=(1,))
    _write(box/"trial_1.json.gz",_trial(fb,1));_write(cone/"trial_1.json.gz",_trial(fc,1,cone=True))
    result=analyze(box,cone)
    assert result["status"]=="complete"
    assert stats([0.,1e-6,-2e-4,None],4,tolerance=1e-4)==dict(total=4,valid=3,unavailable=1,
        mean=pytest.approx(-.000199/3),median=0.,minimum=-2e-4,maximum=1e-6,sum=pytest.approx(-.000199),
        positive=0,negative=1,ties=2,sign_tolerance=1e-4)
    with pytest.raises(ValueError,match="finite"):
        analyze(box,cone,zero_tolerance=math.nan)


def test_square_reads_exchange_and_does_not_invent_new_proposal_flags(tmp_path):
    box,cone,fb,fc=_directories(tmp_path,seeds=(1,))
    _write(box/"trial_1.json.gz",_trial(fb,1));_write(cone/"trial_1.json.gz",_trial(fc,1,cone=True))
    result=analyze(box,cone);group=result["groups"]["cone"]
    square=group["methods"]["square_v2g_fixed"]
    assert square["proxy_available"]==square["proxy_completed"]==1
    assert square["proxy_gap"]["mean"]==.1
    assert square["new_MILP"] is None and square["new_MILP_flags_available"]==0
    design=group["records"]["1"]["methods"]["square_v2g_fixed"]["design"]
    assert design["proxy"]["record_field"]=="exchange"
    assert design["selected_score"]==2. and design["score_units"]=="weighted kW squared"
    assert design["proxy"]["all_expected_pairs_required"] is None
    assert design["selected_prediction"]["complete"]
