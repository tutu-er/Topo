"""Independent saved-array replay, without AC, optimization or fitter calls."""
import argparse,hashlib,importlib.util,json,pickle,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
HELPER=REPO/'research_experiments/wzzt_dr_selection/dr_spatial_correlated/audit/audit_results.py'
spec=importlib.util.spec_from_file_location('independent_old_audit',HELPER)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checkref(r):
    assert sha(r['path'])==r['sha256'],r['path']
    return Path(r['path'])
def load(r):
    with checkref(r).open('rb') as f:return pickle.load(f)

def audit(root):
    assert (root/'ended.json').is_file()
    cfg=read(root/'protocol.json');freeze=read(root/'price_freeze.json');ledger=read(root/'observations_frozen.json')
    rows=read(root/'scored.json')['rows'];summary=read(root/'summary.json');p=load(read(root/'problem_frozen.json')['input']);pool=p['pool']
    counts=Counter();maxerr=defaultdict(float);fail=[]
    def ck(name,condition,detail=None):
        counts[name]+=1
        if not bool(condition):fail.append(dict(check=name,detail=detail))
    def close(name,a,b,tol=1e-12,detail=None):
        a,b=np.asarray(a,float),np.asarray(b,float)
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():ck(name,False,detail);return
        e=float(np.max(np.abs(a-b),initial=0));maxerr[name]=max(maxerr[name],e);ck(name,e<=tol,dict(record=detail,error=e))
    ck('exact_frozen_price_file',sha(root/'price_freeze.json')==cfg['parent_prices']['sha256'])
    for path,digest in cfg['source_sha256'].items():ck('source_hash',sha(path)==digest,path)
    with zipfile.ZipFile(root/'source_snapshot.zip') as z:
        saved=Counter(hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist())
        ck('source_snapshot_closure',saved==Counter(cfg['source_sha256'].values()))
    history=load(cfg['inputs']['history'])['history'];models={m['candidate_id']:m for m in pool['models']}
    h={name:sum(float(np.abs(np.asarray(s['drop_target'])-np.asarray(s['P_terminal'])@np.asarray(m['R']).T-np.asarray(s['Q_terminal'])@np.asarray(m['X']).T).sum()) for s in history['validation']) for name,m in models.items()}
    ck('full_history',sum(len(s['P_terminal']) for s in history['training'])==288 and sum(len(s['P_terminal']) for s in history['validation'])==288)
    truth={tuple(sorted(x)) for x in read(checkref(cfg['parent_truth']))['missed_blocks']}
    model_errors={name:helper.errors(m['frozen_output_support_labels'],truth) for name,m in models.items()}
    for name,m in models.items():
        cal=pool['calibration'];point=m['point']
        kept=[s for s,r,x in zip(point['supports'],point['r'],point['x']) if len(s)==1 or cal['a']*r+cal['b']*x+1e-10>=.16*cal['median_root_depth']]
        ck('frozen_output_rule',kept==m['frozen_output_supports'])
        close('historical_loss',h[name],m['historical_validation_abs_sum'],1e-10,name)
    oldledger=read(checkref(cfg['parent_observations']));oldmap={r['record_id']:r for r in oldledger['rows']};scored={r['record_id']:r for r in rows}
    ck('complete_144_records',len(ledger['rows'])==len(oldmap)==len(scored)==144 and len(set(scored))==144)
    physical=[]
    for item in ledger['rows']:
        key=item['record_id'];row=scored[key];oldrec=oldmap[key]
        if item['status']!='observed':ck('failure_retained',row['status']=='unresolved',key);continue
        new=load(item['packet']);old=load(oldrec['packet']);v=new['private'];o=old['private'];s=new['summary']
        a=np.asarray(freeze['policies'][item['policy']]['capacity_kw']);u=np.asarray(freeze['policies'][item['policy']]['mean_u'])
        expected=np.sign(o['delivered_kw'])*np.minimum(np.abs(o['delivered_kw']),a[:,None])
        for name in ['natural_P_pu','natural_Q_pu','root_true','Z','mean_u','capacity_kw','prices','cv','multiplier_mean']:
            close('same_'+name,v[name],o[name],0.,key)
        close('raw_delivery',v['proposed_delivered_kw'],o['delivered_kw'],0.,key)
        close('saturation_projection',v['delivered_kw'],expected,0.,key)
        close('component_bound',np.maximum(np.abs(v['delivered_kw'])-a[:,None],0.),np.zeros_like(u),0.,key)
        ck('total_resource_bound',np.max(np.sum(np.abs(v['delivered_kw']),axis=0))<=2+1e-12,key)
        ck('no_redraw',s['no_redraw'] and s['delivery_rule']=='device_clip' and not s['no_clipping'],key)
        quote=np.divide(2*np.abs(u),a[:,None],out=np.zeros_like(u),where=np.broadcast_to(a[:,None]>0,u.shape))
        close('frozen_prepaid_fee',.25*np.sum(quote*np.abs(u)),item['fee'],1e-9,key)
        close('fee_vs_uncapped',s['planned_cost'],old['summary']['planned_cost'],0.,key)
        dp=np.zeros((96,15))
        for j,col in enumerate(p['active_positions']):dp[:,col]=expected[j]
        close('physical_P',v['physical_P_pu'],np.asarray(v['natural_P_pu'])+dp/200,1e-14,key)
        close('physical_Q',v['physical_Q_pu'],np.asarray(v['natural_Q_pu'])+.35*dp/200,1e-14,key)
        nclose=bool(np.array_equal(expected,o['delivered_kw']))
        if nclose:
            close('no_clip_same_AC',v['voltage_true'],o['voltage_true'],0.,key)
            for field in ['P_terminal','Q_terminal','drop_target']:close('no_clip_same_observation',new['scenes']['both'][field],old['scenes']['both'][field],0.,key)
        rng=np.random.default_rng(s['meter_seed']);meter=v['both_measured_channels']
        for field,raw,sd in [('p_measured',v['physical_P_pu'],.005),('q_measured',v['physical_Q_pu'],.005),('voltage_measured',v['voltage_true'],.0005)]:
            raw=np.asarray(raw);pred=raw+rng.normal(0.,sd*np.maximum(np.abs(raw),1e-12),size=raw.shape)
            close('same_meter_innovations_'+field,meter[field],pred,0.,key)
        close('same_root_meter',meter['root_observed'],o['both_measured_channels']['root_observed'],0.,key)
        y=np.asarray(meter['root_observed'])[:,None]**2-np.asarray(meter['voltage_measured'])**2
        close('squared_voltage_target',new['scenes']['both']['drop_target'],y,0.,key)
        for kind,e in helper.electrical_replay(v).items():close('AC_balance_'+kind,e,0.,1e-8,key)
        rr=helper.selection_replay(pool,h,new['scenes']['both'])
        ck('original_final_selection',rr['selected']==row['selected'],key)
        close('original_D',rr['D_micro'],row['D_micro'],1e-7,key)
        close('original_margin',rr['margin_micro'],row['margin_micro'],1e-7,key)
        ck('truth_error_replay',model_errors[row['selected']]['errors']==row['errors'],key)
        close('actual_kwh',.25*np.abs(expected).sum(),row['abs_kwh'],1e-12,key)
        ck('energy_not_increased',row['abs_kwh']<=row['raw_abs_kwh']+1e-12,key)
        ck('fee_unchanged_per_record',row['fee']==row['old_fee'],key)
        physical.append(dict(record_id=key,converged=s['ac_converged_samples'],voltage_min=s['voltage_min_pu'],voltage_max=s['voltage_max_pu'],voltage_bad=s['voltage_violation_cells'],resource_bad=s['resource_violation_cells'],no_clip=nclose))
    for table in summary['tables']:
        rr=[r for r in rows if r['law']==table['law'] and r['policy']==table['policy']];vv=[r for r in rr if r['status']=='complete']
        ck('summary_denominator',table['planned']==len(rr)==12 and table['complete']==len(vv))
        ck('summary_error_total',table['errors']==sum(r['errors'] for r in vv))
        ck('summary_recoveries',table['recoveries']==sum(r['errors']==0 for r in vv))
        close('summary_abs_energy',table['abs_kwh_per_day'],np.mean([r['abs_kwh'] for r in vv]),1e-12)
    result=dict(status='passed' if not fail else 'failed',counts=counts,checks=sum(counts.values()),failures=fail,max_errors=maxerr,
        independent_helpers=dict(path=str(HELPER),sha256=sha(HELPER)),source_sha256=sha(__file__),protocol_sha256=sha(root/'protocol.json'),
        physical=dict(records=len(physical),converged_samples=sum(r['converged'] for r in physical),resource_violation_cells=sum(r['resource_bad'] for r in physical),voltage_violation_cells=sum(r['voltage_bad'] for r in physical),
            voltage_min=min(r['voltage_min'] for r in physical),voltage_max=max(r['voltage_max'] for r in physical),unchanged_delivery_records=sum(r['no_clip'] for r in physical)),
        new_AC_calls=0,new_optimizations=0)
    (root/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    if fail:raise AssertionError('independent checks failed')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);audit(ap.parse_args().run)
