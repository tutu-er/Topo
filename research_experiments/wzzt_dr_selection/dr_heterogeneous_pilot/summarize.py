"""Post-freeze aggregate diagnostics; never writes or changes a price."""
from pathlib import Path
import argparse,json,pickle,hashlib
import numpy as np


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def load(p):return pickle.loads(Path(p).read_bytes())


def summarize(root):
    root=Path(root);protocol=read(root/'protocol.json');prices=read(root/'price_freeze.json')
    data=read(root/'scored_private.json')['rows'];observed={r['record_id']:r for r in read(root/'observations_frozen.json')['rows']}
    allocation={c['case']:c for c in read(protocol['allocations']['path'])['cases']}
    events=[];unique=set();total_volume={};plan_details=[]
    for c in protocol['cases']:
        packet=load(root/c['name']/'problem.pkl');plans=load(prices['cases'][c['name']]['plans']['path'])
        nodes=allocation[c['case']]['nodes'];labels=c['labels'];cap=sum(n['total_kw'] for n in nodes) if 'total_kw' in nodes[0] else None
        pool=read(root/c['name']/'pool.json');truth=read(root/c['name']/'truth_private.json');ts={tuple(sorted(b)) for b in truth['missed_blocks']}
        differences={}
        for m in pool['models']:
            blocks={tuple(sorted(b)) for b in m['frozen_output_support_labels']}
            differences[m['candidate_id']]=dict(missing=[list(b) for b in sorted(ts-blocks)],extra=[list(b) for b in sorted(blocks-ts)],
                historical_origin=pool['competition_rule']['selected_origin' if m['candidate_id']=='coarse' else 'alternative_origin'])
        opt=plans['optimized']['mean_u'];node_abs=.25*np.abs(opt).reshape(len(nodes),4,96).sum((1,2))
        top=sorted(zip(labels,node_abs),key=lambda x:x[1],reverse=True)
        plan_details.append(dict(condition=c['name'],threshold_micro=packet['threshold'],candidate_posthoc_errors=differences,
            nominal_absolute_kwh=float(.25*np.abs(opt).sum()),nominal_net_kwh=float(.25*opt.sum()),
            nominal_type_absolute_kwh={k:float(.25*np.abs(opt[i::4]).sum()) for i,k in enumerate(['shift','curtail','ev','battery'])},
            top_nodes_absolute_kwh=[dict(label=int(k),kwh=float(v)) for k,v in top[:5]],
            active_nodes_over_01_kwh=int((node_abs>.01).sum()),cost_first_cost_matched=plans['cost_first']['cost_matched'],
            rotation_cost_matched=plans['rotation']['cost_matched']))
        for law in protocol['laws']:
            for policy in protocol['primary_policies']:
                group=[r for r in data if r['condition']==c['name'] and r['law']==law and r['policy']==policy and r['status']=='complete']
                statistics=[]
                for row in group:
                    obj=load(observed[row['record_id']]['packet']['path']);v=obj['private'];u=v['delivered']
                    p=v['natural_p'].to_numpy()*c['base_kw'];scene=obj['scene']
                    h=hashlib.sha256()
                    for key in ['P_terminal','Q_terminal','V_terminal','V_root_observed']:
                        h.update(np.asarray(scene[key],dtype='<f8').tobytes())
                    unique.add(h.hexdigest())
                    positive_kwh=float(.25*np.maximum(p,0).sum());volume=float(.25*np.abs(u).sum())
                    statistics.append(dict(positive_kwh=positive_kwh,volume=volume,net=float(.25*u.sum()),ratio=volume/positive_kwh,
                        type_volume=[float(.25*np.abs(u[i::4]).sum()) for i in range(4)],
                        minV=row['physical']['voltage_min'],maxV=row['physical']['voltage_max']))
                events.append(dict(condition=c['name'],law=law,policy=policy,n=len(group),
                    mean_background_positive_kwh=float(np.mean([s['positive_kwh'] for s in statistics])),
                    mean_absolute_dr_share=float(np.mean([s['ratio'] for s in statistics])),
                    mean_net_dr_kwh=float(np.mean([s['net'] for s in statistics])),
                    mean_type_absolute_kwh=dict(zip(['shift','curtail','ev','battery'],np.mean([s['type_volume'] for s in statistics],axis=0).tolist())),
                    minV=float(min(s['minV'] for s in statistics)),maxV=float(max(s['maxV'] for s in statistics)),
                    D_range_micro=[float(min(s['D_micro'] for s in group)),float(max(s['D_micro'] for s in group))],
                    paired_remaining_gap_micro=float(packet['threshold']-np.mean([s['D_micro'] for s in group]))))
    out=dict(plans=plan_details,actual=events,distinct_observations=len(unique),total_records=len(data),
        independence='18 background paths; main/stress and policies paired; zero schedules yield duplicates',
        true_supports_used_only_after_selection=True,new_planning_calls=0,new_AC_calls=0)
    (root/'aggregate_diagnostics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);summarize(p.parse_args().output)
