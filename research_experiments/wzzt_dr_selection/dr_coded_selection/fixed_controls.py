"""Freeze seven predeclared controls if the full price gate is unresolved.

This never substitutes zero for an unfinished WZZT price method. That method's
four policy/law/checkpoint entries remain separately price_unresolved.
Only historical observations and ordinary input design are used here.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from coded_planner import read,write,sha,timestamp,ordinary_tables,_choice
from coded_model import TEMPLATES,directions,expand_prices,contract_spec

POLICIES=('zero','single','rotation','same','opposed','alternating','input_best')

def prepare(planning_root,root,repo):
    planning_root,root,repo=map(lambda p:Path(p).resolve(),(planning_root,root,repo))
    if root.exists():raise FileExistsError('existing controls root retained')
    cfg=read(planning_root/'protocol.json');gate=read(planning_root/'case_plans.json')
    if gate['status']!='planning_unresolved':raise ValueError('controls-only mode requires an unresolved full price gate')
    if gate['protocol_sha256']!=sha(planning_root/'protocol.json'):raise RuntimeError('original price protocol drift')
    frozen=read(planning_root/'source_snapshot_before_controller_fix.json')
    if frozen['sha256']!=sha(planning_root/'source_snapshot_before_controller_fix.zip'):
        raise RuntimeError('frozen original implementation snapshot drift')
    if frozen['protocol_sha256']!=gate['protocol_sha256']:raise RuntimeError('snapshot/protocol mismatch')
    # Only the separately documented controller interface fix may differ after
    # price computation; no planner/identifier source is changed to rescue it.
    allow={'controller.py','test_controller.py'}
    source_differences=[]
    for p,h in cfg['source_sha256'].items():
        if sha(p)!=h:
            if Path(p).name not in allow:raise RuntimeError('non-controller price source drift: '+p)
            source_differences.append(dict(path=p,old_sha256=h,new_sha256=sha(p)))
    from research_experiments.wzzt_dr_selection.dr_competition_bridge.physical_bridge import load_history
    history=load_history(cfg['pilot_row']);positions=[history['labels'].index(v) for v in cfg['active_labels']]
    root.mkdir(parents=True)
    write(root/'history_receipt.json',history['receipt'])
    unresolved=dict(protocol=dict(path=str(planning_root/'protocol.json'),sha256=sha(planning_root/'protocol.json')),
                    result=dict(path=str(planning_root/'case_plans.json'),sha256=sha(planning_root/'case_plans.json')))
    case=dict(job_id=cfg['pilot_job'],bundle=8,voltage_sigma=.001,history_row=cfg['pilot_row'],
              active_labels=cfg['active_labels'],locks={},site_receipt=cfg['site_receipt'])
    for law in ('local','common'):
        rows=ordinary_tables(history,cfg['active_labels'],law)
        write(root/f'ordinary_{law}.json',rows)
        one=_choice([r for r in rows if np.count_nonzero(np.any(np.asarray(r['small_menu'])>0,axis=0))<=1],530015,True)
        rotation=_choice([r for r in rows if r['small_menu']==[[2,0],[0,2]]],530016,True)
        best=_choice(rows,530014,True)
        choices=dict(zero=dict(template='same',small_menu=[[0,0],[0,0]],value=None),single=one,
            rotation=rotation,same=dict(template='same',small_menu=[[1,1],[1,1]],value=None),
            opposed=dict(template='opposed',small_menu=[[1,1],[1,1]],value=None),
            alternating=dict(template='alternating_opposed',small_menu=[[1,1],[1,1]],value=None),input_best=best)
        lock=root/'locks'/f"{cfg['pilot_job']}_{law}.json"
        write(lock,dict(status='frozen',created_utc=timestamp(),job_id=cfg['pilot_job'],dependence=law,
            labels=history['labels'],active_labels=cfg['active_labels'],
            plans={k:expand_prices(v['small_menu'],positions,15).tolist() for k,v in choices.items()},
            directions={k:directions(v['template'],positions,15).tolist() for k,v in choices.items()},
            templates={k:v['template'] for k,v in choices.items()},selection_receipts=choices,
            device_contract=cfg['device_contract'],source_qualification=cfg['source_qualification'],
            contract=cfg['contract'],template_specs={k:contract_spec(k) for k in TEMPLATES},
            controls_only=True,full_price_method='price_unresolved; no assigned policy or zero fallback',
            planning_unresolved=unresolved,future_read_before_freeze=False,
            early_price_optimization_claim=False,
            scope='same predeclared two sites and three templates; input selection only; fixed controls are not a solved WZZT price policy'))
        case['locks'][law]=dict(path=str(lock),sha256=sha(lock))
    result=dict(status='complete',controls_only=True,created_utc=timestamp(),cases=[case],policies=list(POLICIES),
        planning_unresolved=unresolved,full_wzzt_price_status='price_unresolved',
        missing_price_method_records=[dict(policy='wzzt_best',dependence=law,checkpoint=point,status='price_unresolved')
                                      for law in ('local','common') for point in ('early','final')],
        actual_control_records_planned=28,original_total_method_records_planned=32,
        source_changes_after_price_gate=source_differences,
        original_frozen_source_archive=dict(path=str(planning_root/'source_snapshot_before_controller_fix.zip'),sha256=frozen['sha256']),
        actual_future_read=False,LP_calls=0,AC_calls=0,
        locked_panel=gate['locked_panel'],
        scope='complete fixed-control plans only; full WZZT price method remains unresolved and unexecuted')
    write(root/'case_plans.json',result)
    print(json.dumps({k:result[k] for k in ('status','controls_only','actual_control_records_planned','full_wzzt_price_status')}))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--planning-root',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--repo-root',type=Path,required=True)
    a=p.parse_args();sys.path[:0]=[str(a.repo_root),str(a.repo_root/'rnj_wzzt_core')]
    prepare(a.planning_root,a.root,a.repo_root)

if __name__=='__main__':main()
