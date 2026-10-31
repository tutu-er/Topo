"""Freeze historical-only selection, qualify devices, lock a four-menu library.

This is the preparation for real full-identifier simulations, not a replacement
for that experiment. No original test, network edges, or topology truth is read.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import pickle
import random
import sys
import traceback
import numpy as np

REPO=Path(r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo')
TASK=REPO/'research_experiments/wzzt_dr_selection'
SCRATCH=Path(r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
sys.path.insert(0,str(REPO))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def prepare(root):
    from research_experiments.wzzt_dr_selection.dr_competition_bridge import physical_bridge as old
    from research_experiments.wzzt_dr_selection.dr_coded_selection import coded_physical as coded
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'preparation_protocol.json').exists():raise FileExistsError('locked preparation already exists')
    inventory_path=TASK/'evidence/scenario_review_v1/source_inventory.json'
    audit_path=SCRATCH/'wzzt_dr_scenario_review_v1/candidate_audit.json'
    diagnostic_path=SCRATCH/'wzzt_b_joint_mechanism_v1/output_support_diagnostic.json'
    inventory=read(inventory_path);audit=read(audit_path);diag=read(diagnostic_path)
    ids=['main__flynn16__b5__reference__v0002__N288',
         'main__pengwah18__b2__reference__v0002__N288',
         'main__pengwah18__b4__reference__v0002__N288',
         'main__pengwah18__b5__reference__v0002__N288']
    eligible=[r for r in inventory['rows'] if r['job_id'] in ids]
    assert len(eligible)==4 and all(r['N']==288 and r['voltage_sigma']==.0002 and r['pq_sigma']==.005 for r in eligible)
    rng=random.Random(9583107)
    selected=[rng.choice(sorted([r for r in eligible if r['case']==case],key=lambda r:r['job_id']))
              for case in sorted({r['case'] for r in eligible})]
    bindings={str(p):sha(p) for p in (Path(__file__),inventory_path,audit_path,diagnostic_path,
        Path(old.__file__),Path(coded.__file__))}
    source_rows=[]
    for row in selected:
        metadata=Path(r'D:\0-github_workspace\Topo\outputs\wzzt_paper_20261029_full\networks')/row['case']/'reference/metadata.json'
        meta=read(metadata)
        source_rows.append(dict(index_row={k:v for k,v in row.items() if k not in ('test','existing_result')},
             source_metadata=dict(path=str(metadata),sha256=sha(metadata),case_name=meta['source']['case_name'])))
    p=dict(id='budget_reference_diagnostic_preparation_v1',created_utc=datetime.now(timezone.utc).isoformat(),
           source_sha256=bindings,eligible_history_only_ids=ids,selected=source_rows,
           selection_seed=9583107,partner_seed=9583108,
           selection_rule='one random historical-output-disagreement source per network; closed flynn b7 excluded; no truth-error screening',
           retain_unquotable_sources_no_replacement=True,expected_full_history_rows=[288,288],
           allowed_inputs=['source metadata','old historical candidate summaries','train/validation observed NPZ'],
           forbidden_inputs=['old test NPZ','old true_network','old diagnostics','actual future'],
           device=dict(coded.DEVICE),future_runs_in_preparation=0)
    write(root/'preparation_protocol.json',p)
    rows=[]
    for item in source_rows:
        source=item['index_row'];case=source['case'];out=root/case;out.mkdir()
        record=dict(case=case,job_id=source['job_id'],status='started')
        rows.append(record)
        try:
            history=old.load_history(source,expected_case=case,base_kw=200.,source_metadata=item['source_metadata'])
            forecast=old.forecast_mean(history);labels=history['labels'];n=len(labels)
            previous=next(r for r in audit['records'] if r['job_id']==source['job_id'])
            dispute=next(r for r in diag['rows'] if r['job_id']==source['job_id'])['proper_disputed_supports']
            disputed=[set(s) for s in dispute]
            union=set.union(*disputed);anchors=sorted(union-set.intersection(*disputed))
            anchor=random.Random(f"9583108:anchor:{source['job_id']}").choice(anchors)
            families=[{frozenset(s) for s in f} for f in previous['contracted_families']]
            common=set.intersection(*families)
            regions=[set(s) for s in common if union<=set(s) and len(s)>len(union)]
            region=min(regions,key=lambda s:(len(s),tuple(sorted(s)))) if regions else set(range(n))
            qualification=[]
            scenes=history['training']+history['validation']+[None]
            available=np.ones(n,bool)
            for index,scene in enumerate(scenes):
                power=forecast['P_pu'] if scene is None else scene['P_terminal'].to_numpy()
                reactive=forecast['Q_pu'] if scene is None else scene['Q_terminal'].to_numpy()
                for sign in (1,-1):
                    delta=coded.waveforms('same',np.full(n,sign),[])
                    q=coded.device_precheck(power,reactive,np.ones(n,bool),delta,base_kw=200.)
                    good=np.array([s=='available' for s in q['node_status']]);available&=good
                    qualification.append(dict(day=index,sign=sign,available=good.astype(int).tolist()))
            candidates=sorted(i for i in region-union if available[i])
            record.update(labels=labels,anchor=labels[anchor],anchor_qualified=bool(available[anchor]),
                          common_region=[labels[i] for i in sorted(region)],
                          partner_candidates=[labels[i] for i in candidates],
                          qualified_nodes=[labels[i] for i in range(n) if available[i]],
                          qualification=qualification,source_metadata=item['source_metadata'],
                          history_receipt=history['receipt'],no_true_topology_or_future=True)
            if not available[anchor] or not candidates:
                record['status']='unquotable_history_device_contract';continue
            partner=random.Random(f"9583108:partner:{source['job_id']}").choice(candidates)
            pair=[labels[anchor],labels[partner]]
            plans={};directions={};templates={};pairs={}
            for policy in ('zero','single','rotation','opposed'):
                prices=np.zeros((2,n),int);direction=np.zeros((2,n),int)
                if policy=='single':prices[:,anchor]=2;direction[:,anchor]=1
                if policy=='rotation':prices[0,anchor]=2;prices[1,partner]=2;direction[0,anchor]=direction[1,partner]=1
                if policy=='opposed':prices[:,[anchor,partner]]=1;direction[:,anchor]=1;direction[:,partner]=-1
                old.validate_plan(prices,n)
                plans[policy]=prices.tolist();directions[policy]=direction.tolist()
                templates[policy]='opposed' if policy=='opposed' else 'same';pairs[policy]=[pair,pair]
            lock=dict(status='frozen',case=case,base_kw=200.,root_label=history['root_label'],
                source_metadata=item['source_metadata'],labels=labels,plans=plans,directions=directions,
                templates=templates,active_labels_by_row=pairs,orientations_by_event=[1,1,1,1],
                device_contract=dict(baseline_kw=.2,baseline_kvar=.07,upper_kw=.4,start_hour=19,end_hour=21,
                    initial_energy_kWh=0.,final_energy_kWh=.4,efficiency=1.,non_V2G=True),
                design_scope='history-only positions, fixed budget library for model-known offline diagnostic; not deployable optimized DR')
            write(out/'plans.json',lock)
            for policy in plans:coded._lock(out/'plans.json',sha(out/'plans.json'),policy,old)
            with (out/'history.pkl').open('xb') as stream:pickle.dump(dict(history=history,forecast=forecast),stream,protocol=4)
            record.update(status='qualified_and_actions_frozen',partner=labels[partner],
                          plan_lock=dict(path=str(out/'plans.json'),sha256=sha(out/'plans.json')),
                          history_packet=dict(path=str(out/'history.pkl'),sha256=sha(out/'history.pkl')),
                          root_label=history['root_label'],training_rows=288,validation_rows=288)
        except Exception as exc:
            record.update(status='preparation_failed',exception_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        finally:write(out/'preparation_result.json',record)
    write(root/'preparation_ended.json',dict(status='complete',rows=rows,expected_sources=2,
        qualified=sum(r['status']=='qualified_and_actions_frozen' for r in rows),
        real_identifier_calls=0,LP_calls=0,AC_calls=0,actual_future=False))
    print(json.dumps(dict(status='prepared',sources=[{k:r.get(k) for k in ('case','job_id','status','anchor','partner')} for r in rows]),ensure_ascii=True),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    prepare(parser.parse_args().root)
