"""Trace already fitted candidates, never refit or call a project selector."""
from pathlib import Path
from collections import Counter
import json
from audit import read,canonical,eff,chosen,sha

ROOT=Path(r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\wzzt_b_design_v1')
OUT=Path(__file__).resolve().parent/'run_v1'
rows=read(ROOT/'evaluations.json');crosses=[];better=[];seen=set()

def basic(p):
    return dict(supports=p['supports'],r=p['r'],x=p['x'],objective=p.get('objective'),validation_mae=p.get('validation_mae'),
                origin=p.get('origin'),path_index=p.get('path_index'))

for row in rows:
    ev=row.get('evaluated',{});we=ev.get('current_pool_wzzt',{});ae=ev.get('current_root_augmented_rnj',{})
    if we.get('status')!='complete':continue
    d=read(ROOT/'records'/(row['fit_id']+'.json'));w=d['methods']['current_pool_wzzt'];a=d['methods']['current_root_augmented_rnj']
    target=set(map(tuple,we['target']));cal=d['applied_calibration'];ws=w['selected'];ars=a['selected'];raw=ws['selected_uncontracted'];arraw=ars['selected_uncontracted']
    wf=dict(supports=ws['support_indices'],r=ws['r_values'],x=ws['x_values']);af=dict(supports=ars['support_indices'],r=ars['r_values'],x=ars['x_values'])
    entry=dict(flow=row['flow'],case=row['case'],menu_id=row['menu_id'],fit_id=row['fit_id'],W_final_errors=we['final_errors'],AR_final_errors=ae.get('final_errors'),
               W_raw_errors=we['raw_errors'],AR_raw_errors=ae.get('raw_errors'),W_selected=basic(raw),AR_selected=basic(arraw),target=sorted(target),
               W_min_validation=min(p['validation_mae'] for p in w['points']),AR_min_validation=min(p['validation_mae'] for p in a['points']))
    if ae.get('status')=='complete' and we['final_errors']<ae['final_errors']:
        contrib={}
        for name,pred in [('W',eff(wf)),('AR',eff(af))]:
            err=pred^target;contrib[name]={typ:sum((len(s)==1 if typ=='leaf' else len(s)==2 if typ=='proper' else len(s)==3) for s in err) for typ in ('leaf','proper','root')}
        entry['error_type_contributions']=contrib
        entry['selection_origin_type']=('native' if raw.get('origin')=='historical_native_RNJ' else 'path_start' if raw.get('path_index')==0 else 'accepted_path')
        origin=raw.get('origin');pi=raw.get('path_index')
        if origin in w['paths'] and pi and pi>0:
            pp=w['paths'][origin]['path'];entry['selected_before']=basic(pp[pi-1]);entry['selected_after']=basic(pp[pi])
            entry['selected_support_added']=sorted(eff(pp[pi])-eff(pp[pi-1]));entry['selected_support_removed']=sorted(eff(pp[pi-1])-eff(pp[pi]))
        better.append(entry)
    if row['fit_id'] in seen:continue
    seen.add(row['fit_id']);byfirst={};local=[]
    for name,path in w['paths'].items():
        j=0
        for dec in path['decisions']:
            best=None
            for c in dec['candidates']:
                if best is None or c['objective']<best['objective']-1e-12:best=c
            take=best is not None and dec['current']-best['objective']>1e-10+.001*dec['current']
            if take:
                before=path['path'][j];after=path['path'][j+1];j+=1
                if best['removed']:
                    byfirst.setdefault(name,j)
                    measured=next(p for p in w['points'] if p.get('origin')==name and p.get('path_index')==j)
                    local.append(dict(**entry,path=name,accepted_step=j,candidate=best['candidate'],removed=best['removed'],before=basic(before),after=basic(measured),
                        before_errors=len(eff(before)^target),after_errors=len(eff(after)^target),gain=dec['gain'],threshold=1e-10+.001*dec['current'],
                        final_from_this_lineage=(raw.get('origin')==name and raw.get('path_index',-1)>=j),
                        final_retains_candidate=tuple(best['candidate']) in eff(wf)))
    if local:
        remaining=[p for p in w['points'] if not(p.get('origin') in byfirst and p.get('path_index',-1)>=byfirst[p['origin']])]
        idx,rr,ff=chosen(remaining,cal);without=len(eff(ff)^target)
        for x in local:
            x['saved_point_deletion_counterfactual']=dict(no_refit=True,removed_cross_lineages=byfirst,
                remaining_points=len(remaining),original_final_errors=we['final_errors'],without_cross_lineage_final_errors=without,
                without_selected=basic(rr),without_final=sorted(eff(ff)),
                interpretation='selection ablation of already fitted path points, not a rerun of W without crossing search')
        crosses.extend(local)

result=dict(status='complete_read_only_trace',W_better_maps=len(better),W_better_unique_fits=len({x['fit_id'] for x in better}),
    better_origins=dict(Counter(x['selection_origin_type'] for x in better)),cross_accepts=len(crosses),cross_unique_fits=len({x['fit_id'] for x in crosses}),
    final_direct_cross_lineages=sum(x['final_from_this_lineage'] for x in crosses),
    cross_saved_deletion_effect=dict(Counter('helps' if x['W_final_errors']<x['saved_point_deletion_counterfactual']['without_cross_lineage_final_errors'] else 'harms' if x['W_final_errors']>x['saved_point_deletion_counterfactual']['without_cross_lineage_final_errors'] else 'same' for x in crosses)),
    better=better,crosses=crosses,source_sha256=sha(__file__),new_LP_calls=0,new_AC_calls=0)
(OUT/'structural_trace.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('better','crosses')},indent=2))
for x in crosses:
    cf=x['saved_point_deletion_counterfactual']
    print(x['fit_id'],x['flow'],x['case'],x['menu_id'],'cross',x['before_errors'],'->',x['after_errors'],'final',x['W_final_errors'],'without',cf['without_cross_lineage_final_errors'],'origin',x['W_selected']['origin'])
