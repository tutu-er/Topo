"""Explicit public aggregate allowlist; no household/observed/model trajectories."""
from pathlib import Path
import argparse,json,hashlib,sys,platform,xml.etree.ElementTree as ET
import numpy as np
import scipy


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def export(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    protocol=read(root/'protocol.json');summary=read(root/'summary.json');diagnostics=read(root/'aggregate_diagnostics.json')
    write(output/'summary.json',summary);write(output/'aggregate_diagnostics.json',diagnostics)
    write(output/'audit_summary.json',read(root/'independent_audit.json'))
    planning=[];traces=[];streams=[]
    for i,c in enumerate(protocol['cases']):
        r=read(root/c['name']/'price_receipt.json');s=r['summary']
        planning.append({k:v for k,v in s.items() if k!='policy_nominal_audits'})
        traces.append(dict(condition=c['name'],stages=r['solver_trace']))
        streams.append(dict(condition=c['name'],base_integration_seed=protocol['planning_seed']+i,
            response_integration_seed=protocol['response_seed']+i,initial_optimizer_seed=protocol['optimizer_seed']+i,
            repair_initialization_seed=protocol['optimizer_seed']+i+30,
            independent_base_seed=protocol['qualification_seed']+i,independent_response_seed=protocol['qualification_response_seed']+i,
            paths=[dict(day=d['day'],profile=d['profile_index'],replicate=d['replicate']+10000*i,response_seed=d['response_seed']+100*i) for d in protocol['paths']]))
    write(output/'planning_summary.json',dict(rows=planning,all_target_failures_retained=True))
    write(output/'solver_status.json',dict(stages=traces,initial_attempt=read(root.parent/'run_v1'/'planning_summary.json')))
    keep=['id','created_utc','case_rule','full_history_rows','resource_domain','new_day_weight','new_day_fraction','history_fraction','control',
          'background_law','response_law','planning_samples','qualification_samples','optimizer','planning_repair','primary_policies','baseline_rule',
          'no_money_budget_upper','matched_legacy_2kw_not_used','laws','independent_backgrounds_per_case','expected_records','truth_gate','evaluation','identifier','interpretation']
    public={k:protocol[k] for k in keep};public['cases']=[{k:c[k] for k in ['name','case','labels','base_kw','voltage_sigma','pq_sigma']} for c in protocol['cases']]
    public['streams']=streams;public['protocol_sha256']=sha(root/'protocol.json');public['price_freeze_sha256']=sha(root/'price_freeze.json')
    public['source_count']=len(protocol['source_sha256']);public['source_snapshot_sha256']=sha(root/'source_snapshot.zip')
    public['resource_config_sha256']={k:protocol[k]['sha256'] for k in ['library','allocations']}
    write(output/'protocol_public.json',public)
    write(output/'source_hashes.json',dict(physical_and_planning=protocol['source_sha256'],
        post_processing={str(p):sha(p) for p in Path(__file__).parent.glob('*.py') if p.name in ['post_audit.py','summarize.py','export_results.py']}))
    tests=[]
    for filename in ['tests.xml','repair_tests.xml']:
        tree=ET.parse(root.parent/filename)
        for t in tree.iter('testcase'):tests.append(dict(name=t.attrib['name'],seconds=float(t.attrib['time']),passed=not any(t.find(x) is not None for x in ['failure','error','skipped'])))
    write(output/'tests.json',dict(passed=sum(t['passed'] for t in tests),total=len(tests),tests=tests))
    environment=dict(captured='post_run_record',python=sys.version,executable=sys.executable,platform=platform.platform(),
        numpy=np.__version__,scipy=scipy.__version__,source_parent_commit='4c1ae64e',threads=dict(OPENBLAS=1,OMP=1,MKL=1))
    write(output/'environment.json',environment);write(root/'environment.json',environment)
    # Plot paired paths, not independent error bars for correlated policies.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=read(root/'scored_private.json')['rows'];fig,axs=plt.subplots(1,2,figsize=(11,4.7),sharex=True)
    policy_names=['zero','rotation','optimized'];colors=['#9aa3af','#e89b40','#2773b8']
    for ax,case in zip(axs,protocol['cases'][:2]):
        threshold=next(d['threshold_micro'] for d in diagnostics['plans'] if d['condition']==case['name'])
        for day in range(6):
            values=[next(r['D_micro'] for r in rows if r['condition']==case['name'] and r['law']=='main' and r['policy']==p and r['day']==day) for p in policy_names]
            ax.plot(range(3),values,color='#b8c0cc',lw=.8,alpha=.65,zorder=1)
        for x,p in enumerate(policy_names):
            main=[r['D_micro'] for r in rows if r['condition']==case['name'] and r['law']=='main' and r['policy']==p]
            stress=[r['D_micro'] for r in rows if r['condition']==case['name'] and r['law']=='stress' and r['policy']==p]
            ax.scatter(np.full(6,x-.04),main,s=30,color=colors[x],label='Main response' if x==2 else None,zorder=3)
            ax.scatter(np.full(6,x+.04),stress,s=28,facecolors='none',edgecolors=colors[x],label='Response mismatch' if x==2 else None,zorder=3)
        ax.axhline(threshold,color='#b43d43',ls='--',label=f'Original exclusion threshold {threshold:.2f}')
        ax.set_title(case['name'].replace('_v0005',''));ax.set_xticks(range(3),['No reward','Same-cost rotation','Probability-guided DR'])
        ax.set_ylabel(r'$L_C-1.03L_F$ ($10^{-6}$ pu$^2$)');ax.grid(axis='y',alpha=.15)
        ax.legend(fontsize=8,loc='upper left');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Actual noisy AC: final-selection margin on six paired new days',fontsize=12)
    fig.text(.5,.015,'Frozen historical candidates and R/X; full new resource inventory; 95% planning feasibility not reached.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(output/'effect_summary.png',dpi=170);fig.savefig(output/'effect_summary.pdf');plt.close(fig)
    manifest={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}
    write(output/'manifest.json',dict(files=manifest,allowlist_only=True,excluded=['P/Q/V/AC trajectories','full R/X','full u/prices','random multiplier arrays']))
    print(json.dumps(dict(output=str(output),files=len(manifest),tests=len(tests),records=summary['records'],distinct_observations=diagnostics['distinct_observations'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();export(a.run,a.output)
