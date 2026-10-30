"""Independent pure-array plan/physical binding check; no profiles or solvers."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np

REPO=Path(r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo')
ROOT=Path(r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    sys.path[:0]=[str(REPO),str(REPO/'rnj_wzzt_core'),str(ROOT/'dr_coded_plan_stage')]
    from coded_model import apply_response,directions,expand_prices
    from coded_planner import select_historical_pair
    physical=module('_independent_coded_physical',ROOT/'dr_coded_physical_stage/coded_physical.py')
    p=np.full((96,15),.02);q=np.full((96,15),.008);pair=[3,8]
    menu=expand_prices([[1,1],[1,1]],pair,15)
    checked=0;maxdiff=0.
    for kind in ('same','opposed','alternating_opposed'):
        for key in range(4):
            z=np.zeros(15,dtype=bool);z[pair]=[(key>>0)&1,(key>>1)&1]
            a,b,ca=apply_response(p,q,menu[0],z,base_kw=200.,template_kind=kind,active_positions=pair)
            x,y,cb=physical.apply_coded_response(p,q,menu[0],z,directions(kind,pair,15)[0],kind,pair)
            assert cb['status']=='complete'
            expected=np.zeros_like(p)
            w=np.array([-.2,.2]*4) if kind=='alternating_opposed' else np.r_[np.full(4,-.2),np.full(4,.2)]
            for k,pos in enumerate(pair):
                expected[76:84,pos]=w*z[pos]*(-1 if k==1 and kind!='same' else 1)
            for value,reference in ((a,x),(b,y),(a-p,expected/200),(b-q,.35*expected/200)):
                diff=float(np.max(abs(value-reference)));maxdiff=max(maxdiff,diff);assert diff<1e-14
            assert np.max(abs(expected.sum(axis=0)*.25))<1e-12
            checked+=1
    inventory=json.loads((REPO/'research_experiments/wzzt_dr_selection/evidence/scenario_review_v1/source_inventory.json').read_text(encoding='utf-8'))
    row=next(r for r in inventory['rows'] if r['case']=='paper15' and r['bundle']==8 and r['voltage_sigma']==.001)
    old=json.loads(Path(row['existing_result']['path']).read_text(encoding='utf-8'))
    indices,labels,site=select_historical_pair(old,530013)
    qualification=json.loads((ROOT/'wzzt_dr_coded_qualification_v1/qualification.json').read_text(encoding='utf-8'))
    qcase=next(r for r in qualification['records'] if r['job_id']==row['job_id'])
    conflicts=[label for label in labels if next(n for n in qcase['nodes'] if n['label']==label)['observed_negative_remainder_cells']]
    result=dict(status='passed',full_wave_mask_checks=checked,max_array_difference=maxdiff,
        pilot=row['job_id'],representative_indices=indices,active_labels=labels,site_receipt=site,
        historical_device_conflicts=conflicts,LP_calls=0,AC_calls=0,new_profiles=0,truth_read=False)
    (ROOT/'dr_coded_root_stage/integration_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
