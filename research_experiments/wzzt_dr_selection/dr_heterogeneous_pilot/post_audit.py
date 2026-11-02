"""Read-only independent device, payment, observation and selection audit."""
from pathlib import Path
import argparse,json,pickle,hashlib,sys
import numpy as np
REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
sys.path.insert(0,str(REPO))
from research_experiments.wzzt_dr_selection.dr_heterogeneous_resources.build_allocations import profiles

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def load(p):return pickle.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(root):
    root=Path(root);prot=read(root/'protocol.json');freeze=read(root/'price_freeze.json')
    if not (root/'predictions_frozen.json').exists():raise RuntimeError('selection must freeze before truth audit')
    cfg=read(prot['library']['path']);alloc={c['case']:c for c in read(prot['allocations']['path'])['cases']}
    ledger=read(root/'observations_frozen.json')['rows'];scored={r['record_id']:r for r in read(root/'scored_private.json')['rows']}
    failures=[];count=0
    def ck(name,value):
        nonlocal count
        count+=1
        if not value:failures.append(name)
    def close(a,b,tol=1e-7):return bool(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0.)<=tol)
    ck('complete_record_denominator',len(ledger)==prot['expected_records'])
    ck('protocol_frozen',sha(root/'protocol.json')==freeze['protocol']['sha256'])
    for file,h in prot['source_sha256'].items():ck('source:'+file,sha(file)==h)
    for name in ['library','allocations']:ck(name,sha(prot[name]['path'])==prot[name]['sha256'])
    shared={};observations={};model_cache={};case_by={c['name']:c for c in prot['cases']}
    successful=0;price_plans={}
    for name,rec in freeze['cases'].items():
        ck('plans:'+name,sha(rec['plans']['path'])==rec['plans']['sha256']);price_plans[name]=load(rec['plans']['path'])
    for row in ledger:
        rid=row['record_id']
        if row['status']!='complete':continue
        successful+=1;ck(rid+':packet_hash',sha(row['packet']['path'])==row['packet']['sha256'])
        obj=load(row['packet']['path']);v=obj['private'];scene=obj['scene'];case=case_by[row['condition']]
        nodes=alloc[row['case']]['nodes'];R=4*len(nodes);u=np.asarray(v['delivered']);nom=np.asarray(v['nominal'])
        ck(rid+':finite_delivered',np.isfinite(u).all() and u.shape==(R,96))
        group=(row['condition'],row['day'])
        if group not in shared:shared[group]=(v['natural_p'].copy(),v['natural_q'].copy())
        ck(rid+':paired_background',np.array_equal(v['natural_p'],shared[group][0]) and np.array_equal(v['natural_q'],shared[group][1]))
        pp=np.zeros((96,len(nodes)));qq=pp.copy();fee=0.;state_ok=True
        for i,node in enumerate(nodes):
            d=profiles(node,cfg);block=u[4*i:4*i+4];nblock=nom[4*i:4*i+4]
            ck(rid+f':power_{i}',bool((block>=d['delta_lower_kw']-1e-8).all() and (block<=d['delta_upper_kw']+1e-8).all()))
            e=.25*np.cumsum(block[0]);cap=node['shiftable_kw']
            ck(rid+f':shift_{i}',abs(e[-1])<1e-7 and np.max(np.abs(e),initial=0.)<=2*cap+1e-7)
            ck(rid+f':curtail_{i}',float(.25*block[1].sum())>=-2*node['curtailable_kw']-1e-7 and (block[1]<=1e-8).all())
            ck(rid+f':EV_windows_{i}',close(block[2,32:72],0) and abs(block[2,:32].sum())<1e-7 and abs(block[2,72:].sum())<1e-7)
            p_ev=d['baseline_kw'][2]+block[2];E=node['ev_battery_kwh'];eta_ev=.9
            morning_initial=.4*E+eta_ev*node['ev_grid_task_kwh']*6/14
            for start,end,initial in [(0,32,morning_initial),(72,96,.4*E)]:
                states=initial+.25*eta_ev*np.cumsum(p_ev[start:end])
                ck(rid+f':EV_SOC_{i}_{start}',bool(np.min(states,initial=initial)>=.2*E-1e-7 and np.max(states,initial=initial)<=.9*E+1e-7))
            E=node['battery_energy_kwh'];eta=np.sqrt(.85)
            states=.5*E+.25*np.cumsum(np.where(block[3]>=0,eta*block[3],block[3]/eta))
            ck(rid+f':battery_{i}',bool(np.min(states,initial=.5*E)>=.2*E-1e-7 and np.max(states,initial=.5*E)<=.8*E+1e-7 and abs(states[-1]-.5*E)<1e-7))
            pp[:,i]=block.sum(0);qq[:,i]=block.T@d['q_over_p']
            A=np.where(nblock>=0,d['delta_upper_kw'],-d['delta_lower_kw'])
            fee+=.25*np.sum(np.divide(d['cost_full_activation']*nblock*nblock,A,out=np.zeros_like(A),where=A>0))
        ck(rid+':prepaid_fee',abs(fee-row['fee'])<1e-7)
        ck(rid+':frozen_nominal',np.array_equal(nom,price_plans[row['condition']][row['policy']]['mean_u']))
        ck(rid+':physical_P',close((v['physical_p']-v['natural_p']).to_numpy()*case['base_kw'],pp))
        ck(rid+':physical_Q',close((v['physical_q']-v['natural_q']).to_numpy()*case['base_kw'],qq))
        ck(rid+':AC_converged',bool(np.asarray(v['AC']['converged']).all()))
        ck(rid+':electrical_equations',v['checks']['electrical_equations_pass'])
        p=np.asarray(scene['P_terminal']);q=np.asarray(scene['Q_terminal']);voltage=np.asarray(scene['V_terminal']);rootv=np.asarray(scene['V_root_observed']).reshape(-1)
        y=rootv[:,None]**2-voltage**2;pool=read(root/row['condition']/'pool.json');losses={};mixed={};truth=read(root/row['condition']/'truth_private.json')
        ts={tuple(sorted(b)) for b in truth['missed_blocks']}
        for m in pool['models']:
            losses[m['candidate_id']]=float(np.abs(y-p@np.asarray(m['R']).T-q@np.asarray(m['X']).T).mean())
            mixed[m['candidate_id']]=.6*m['historical_validation_mae']+.4*losses[m['candidate_id']]
        h={m['candidate_id']:m['historical_validation_mae'] for m in pool['models']}
        historical=[k for k in pool['selection_order'] if h[k]<=1.03*min(h.values())+1e-12]
        ref=min(historical,key=lambda k:(mixed[k],pool['selection_order'].index(k)))
        eligible=[k for k in historical if mixed[k]<=1.03*mixed[ref]+1e-12]
        models={m['candidate_id']:m for m in pool['models']}
        chosen=min(eligible,key=lambda k:(len(models[k]['point']['supports']),mixed[k],pool['selection_order'].index(k)))
        ck(rid+':selection',chosen==scored[rid]['selected'])
        predicted={tuple(sorted(b)) for b in models[chosen]['frozen_output_support_labels']}
        ck(rid+':full_error',len(ts^predicted)==scored[rid]['errors'])
        ck(rid+':D',abs(1e6*(losses['coarse']-1.03*losses['refined'])-scored[rid]['D_micro'])<1e-7)
    for name,plans in price_plans.items():
        for simple in ['rotation','cost_first']:
            if plans[simple]['cost_matched']:ck(name+':'+simple+':same_fee',abs(plans[simple]['cost']-plans['optimized']['cost'])<1e-7)
    out=dict(status='passed' if not failures else 'failed',checks=count,failures=failures,complete_records_audited=successful,expected_records=144,
        source_sha256=sha(__file__),new_optimizations=0,new_AC_calls=0,new_identifier_calls=0,
        scope='independent energy/payment/PQ and fixed selector arithmetic; no new simulation or empirical calibration')
    (root/'independent_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print(json.dumps(out));return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);audit(p.parse_args().output)
