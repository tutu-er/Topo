"""Small ownership/tamper fixtures; no grid model, planner, or AC is imported."""
from copy import deepcopy
import gzip
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys
import random

import pytest

from . import ev_mobile_v2_cache_owner_audit as audit


def profile(power,seconds,*,deadline=False,success=True):
    return dict(power_kw=[[power,0.]],method='ac',seconds=seconds,paid_pair_fits=2,
        pairs=[dict(success=success),dict(success=success)],success=success,deadline_reached=deadline)


def event(p,hit):
    result=dict(method='ac',cache_hit=hit,actual_seconds=0. if hit else p['seconds'],
        paid_pair_fits=0 if hit else p['paid_pair_fits'],source_profile_count=len(p['pairs']))
    if not hit:result['deadline_reached']=p['deadline_reached']
    return result


def set_calls(record,name,calls,*,initial=None):
    entries=[dict(profile=deepcopy(p)) for p,_ in calls] if initial is None else initial
    events=[event(p,hit) for p,hit in calls]
    raw=dict(initial_candidates=entries,grids=[],counters=dict(cache_payment_log=events,
        profile_evaluations=sum(not hit for _,hit in calls),profile_cache_hits=sum(hit for _,hit in calls),
        pair_fit_calls=sum(0 if hit else p['paid_pair_fits'] for p,hit in calls),
        actual_profile_seconds=sum(0. if hit else p['seconds'] for p,hit in calls)))
    record['selection']['methods'][name]['raw']=raw


def fixture():
    deps=dict(square_fixed=[],square_mobile=['square_fixed'],scalar_fixed=['square_fixed'],
        scalar_mobile=['square_fixed','square_mobile','scalar_fixed'],cap_fixed=['square_fixed','scalar_fixed'],
        cap_mobile=['square_fixed','square_mobile','scalar_mobile','cap_fixed'])
    r=dict(schema='mobile_reliability_EV_v2',physical_bank='n3',phase='development',index=0,seed=101110001,
        status='complete',selection=dict(methods={name:dict(dependencies=parents,success=True,
            planning_actual_seconds=10.,planning_seconds=30.,raw={}) for name,parents in deps.items()}))
    for name in audit.DECLARED_ORDER:set_calls(r,name,[])
    a,b=profile(1.,.1),profile(2.,.2)
    set_calls(r,'scalar_fixed',[(a,False)])
    set_calls(r,'scalar_mobile',[(a,True),(b,False)])
    set_calls(r,'cap_fixed',[(a,True),(b,True),(b,True)])
    set_calls(r,'cap_mobile',[(b,True)])
    r['selection']['methods']['cap_fixed']['raw']['selected_profile']=deepcopy(b)
    return r,a,b


def test_owner_leak_count_unique_fee_and_ancestors():
    r,_,_=fixture();out=audit.audit_record(r,path='declared/case_000.json.gz')
    assert out['success'] and out['complete']
    cap=out['methods']['cap_fixed']
    assert cap['unowned_hit_queries']==2 and cap['unowned_unique_keys']==1
    assert cap['unowned_original_seconds_per_query']==pytest.approx(.4)
    assert cap['unowned_unique_original_seconds']==pytest.approx(.2)
    assert cap['unowned_unique_original_pair_fits']==2
    assert cap['saved_path_planning_seconds_plus_known_missing_profiles']==pytest.approx(30.2)
    assert cap['unowned_queries'][0]['owner_method']=='scalar_mobile'
    assert 'initial_candidates[1]' in cap['unowned_queries'][0]['profile_path']
    assert 'scalar_mobile.raw.counters' in cap['unowned_queries'][0]['owner_path']
    for name in ('scalar_fixed','scalar_mobile','cap_mobile'):
        assert out['methods'][name]['unowned_hit_queries']==0
    assert len(out['events'])==7 and len(out['owners'])==2


@pytest.mark.parametrize('tamper', ['hit_time','hit_pair_fits','returned_original_time','counter','deadline_missing','wrong_curve','dependency_cycle'])
def test_counter_and_provenance_tampering_is_invalid(tamper):
    r,_,_=fixture();m=r['selection']['methods'];raw=m['cap_fixed']['raw']
    e=raw['counters']['cache_payment_log'][1]
    if tamper=='hit_time':e['actual_seconds']=.2
    elif tamper=='hit_pair_fits':e['paid_pair_fits']=2
    elif tamper=='returned_original_time':raw['initial_candidates'][1]['profile']['seconds']=.3
    elif tamper=='counter':raw['counters']['profile_cache_hits']=99
    elif tamper=='deadline_missing':del m['scalar_mobile']['raw']['counters']['cache_payment_log'][1]['deadline_reached']
    elif tamper=='wrong_curve':raw['initial_candidates'][1]['profile']['power_kw'][0][0]=999.
    else:m['scalar_fixed']['dependencies']=['cap_fixed']
    out=audit.audit_record(r)
    assert not out['success'] and not out['complete'] and out['errors']
    assert any(v['ownership_accounting']=='invalid' for v in out['methods'].values())


def test_exact_float_bytes_do_not_round_nearby_curves_or_negative_zero():
    assert audit._key(profile(1.,.1))!=audit._key(profile(1.+1e-12,.1))
    assert audit._key(profile(0.,.1))!=audit._key(profile(-0.,.1))


def test_non_deadline_failure_is_cached_and_charged_but_deadline_is_not_cached():
    r,_,_=fixture();failed=profile(3.,.4,success=False);stopped=profile(4.,.3,deadline=True,success=False)
    set_calls(r,'scalar_fixed',[(failed,False),(stopped,False)])
    set_calls(r,'scalar_mobile',[(failed,True),(stopped,False)])
    set_calls(r,'cap_fixed',[]);set_calls(r,'cap_mobile',[])
    out=audit.audit_record(r)
    assert out['success'] and out['complete']
    assert [v['cached'] for v in out['owners']]==[True,False,False]
    assert out['methods']['scalar_fixed']['logged_attempted_pair_task_calls']==4


def test_discarded_failed_cap_profile_does_not_shift_later_call_alignment():
    r,a,b=fixture();failed=profile(5.,.37,success=False)
    set_calls(r,'cap_fixed',[(a,True),(failed,False),(b,True)],
        initial=[dict(profile=deepcopy(a)),dict(success=False),dict(profile=deepcopy(b))])
    out=audit.audit_record(r);row=out['methods']['cap_fixed']
    assert out['success'] and not out['complete']
    assert row['ownership_accounting']=='lower_bound'
    assert row['logged_actual_profile_seconds']==pytest.approx(.37)
    assert row['event_calls']==3 and row['aligned_event_calls']==2
    assert row['unmapped_event_calls']==1
    assert row['unowned_hit_queries']==1 and row['unowned_unique_original_seconds']==pytest.approx(.2)
    assert out['events'][4]['key'] is None


def test_missing_owner_log_yields_unknown_in_cap_fixed_but_preserves_every_denominator():
    r,_,_=fixture();r['selection']['methods']['scalar_mobile']['raw']={'failure':'outer failure lost counters'}
    out=audit.audit_record(r)
    assert out['success'] and not out['complete']
    assert out['methods']['cap_fixed']['unresolved_owner_queries']==2
    assert out['methods']['cap_fixed']['unowned_hit_queries']==0
    assert out['methods']['cap_fixed']['ownership_accounting']=='lower_bound'
    assert out['methods']['cap_mobile']['ownership_provable_paid_without_identified_owner']==1
    assert out['methods']['scalar_mobile']['logged_actual_profile_seconds'] is None
    summary=audit.summarize([out])
    assert all(v['history_denominator']==1 for v in summary['methods'].values())


def test_ambiguous_event_alignment_is_not_forced_and_selected_is_not_a_call():
    r,a,_=fixture()
    set_calls(r,'cap_fixed',[(a,True),(a,True)],initial=[dict(profile=deepcopy(a))])
    out=audit.audit_record(r);row=out['methods']['cap_fixed']
    assert out['success'] and not out['complete']
    assert row['aligned_event_calls']==0 and row['unmapped_event_calls']==2
    assert row['unresolved_owner_queries']==2


def test_partial_alignment_matches_independent_tiny_exhaustive_audit(monkeypatch):
    rng=random.Random(718)
    for E in range(1,7):
        for P in range(1,E+1):
            possible=[[rng.random()<.6 for _ in range(P)] for _ in range(E)]
            monkeypatch.setattr(audit,'_matches',lambda e,p,m=possible:m[e][p])
            calls=[('call'+str(j),j) for j in range(P)]
            valid=[order for order in combinations(range(E),P) if all(possible[i][j] for j,i in enumerate(order))]
            expected={}
            if valid:
                for j in range(P):
                    choices={order[j] for order in valid}
                    if len(choices)==1:expected[choices.pop()]=calls[j]
            found,complete=audit._alignment(list(range(E)),calls)
            assert found==expected
            assert complete==bool(valid and len(expected)==E==P)


def test_compact_or_pipeline_missing_evidence_is_not_zero_cost_certificate():
    r,_,_=fixture();r['delivery_provenance']={'schema':'systematic_compact_delivery_v1'}
    out=audit.audit_record(r)
    assert out['success'] and not out['complete'] and not out['full_raw_evidence_available']
    assert all(v['ownership_accounting']=='lower_bound' for v in out['methods'].values())
    r.pop('delivery_provenance');r['status']='pipeline_failure';r.pop('selection')
    out=audit.audit_record(r)
    assert out['success'] and not out['complete']
    assert all(not v['metadata_available'] for v in out['methods'].values())


def test_files_cli_hashes_denominators_no_overwrite_and_no_scientific_imports(tmp_path):
    r,_,_=fixture();path=tmp_path/'case_000.json.gz';path.write_bytes(gzip.compress(json.dumps(r).encode()))
    out=audit.audit_files([path,path],expected_count=1)
    assert out['summary']['history_denominator']==1 and out['records'][0]['input_bytes']==path.stat().st_size
    with pytest.raises(ValueError):audit.audit_files([path],expected_count=2)
    target=tmp_path/'owner.json.gz'
    args=[sys.executable,'-m',audit.__name__,'--file',str(path),'--expected-count','1','--output',str(target)]
    done=subprocess.run(args,text=True,capture_output=True,check=False)
    assert done.returncode==0,done.stderr
    saved=json.loads(gzip.decompress(target.read_bytes()))
    assert saved['summary']['methods']['cap_fixed']['histories_with_known_unowned_hit']==1
    again=subprocess.run(args,text=True,capture_output=True,check=False)
    assert again.returncode!=0 and 'already exists' in again.stderr
    code='import sys; import '+audit.__name__+'; print(any("likelihood" in n or "ev_mobile_v2_design" in n for n in sys.modules))'
    check=subprocess.run([sys.executable,'-c',code],text=True,capture_output=True,check=True)
    assert check.stdout.strip()=='False'
