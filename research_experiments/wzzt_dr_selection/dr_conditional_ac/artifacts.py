"""Export aggregate-only evidence and a figure from completed AC observations."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STAGE=Path(r'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_conditional_ac_stage')
root=STAGE/'run_v1';out=STAGE/'public';out.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(name,x):(out/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
summary=read(root/'summary.json');save('summary.json',summary)
p=read(root/'protocol.json')
configuration={k:v for k,v in p.items() if k not in ['inputs','high_prices','low_prices','source_metadata','source_sha256']}
configuration['protocol_sha256']=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()
configuration['input_sha256']={k:r['sha256'] for k,r in p['inputs'].items()}
configuration['price_sha256']={k:p[k]['sha256'] for k in ['high_prices','low_prices']}
configuration['source_metadata_sha256']=p['source_metadata']['sha256']
configuration['executed_source_sha256']=p['source_sha256']
save('configuration.json',configuration)
for src,name in [(root/'history_load_summary.json','history_load_summary.json'),(root/'ac_post_analysis.json','ac_post_analysis.json'),
    (STAGE.parent/'wzzt_conditional_dr_stage/history_load_audit.json','historical_DR_load_audit.json')]:
    data=read(src)
    for key in ['input_hashes','input_refs','source','inputs']:data.pop(key,None)
    save(name,data)
ledger=read(root/'observations_frozen.json')['rows']
physical=dict(AC_days=len(ledger),AC_converged_timepoints=sum(r['physical']['ac_converged_samples'] for r in ledger),
    all_equations_pass=all(r['physical']['electrical_equations_pass'] for r in ledger),
    voltage_min=min(r['physical']['voltage_min_pu'] for r in ledger),
    voltage_max=max(r['physical']['voltage_max_pu'] for r in ledger),
    voltage_violation_cells=sum(r['physical']['voltage_violation_cells'] for r in ledger),
    resource_violation_days={name:sum(r['physical']['resource_violation'] for r in ledger if r['policy']==name) for name in ['zero','continuous_high','rotation_high','continuous_low','rotation_low']},
    max_distflow_identity_error=max(r['physical']['max_distflow_identity_error_pu2'] for r in ledger),
    clipping=False,redraw=False,failed_days=sum(r['physical']['status']!='complete' for r in ledger),thermal_limits_checked=False)
save('physical_summary.json',physical)
rows=read(root/'scored_private.json')['rows'];names=['zero','rotation_low','continuous_low','rotation_high','continuous_high']
labels=['No DR','Rotation\n47.20','Optimized\n47.20','Rotation\n81.02','Optimized\n81.02']
colors=['#64748b','#93c5fd','#2563eb','#86efac','#15803d']
fig,ax=plt.subplots(1,2,figsize=(11.2,4.4),layout='constrained')
for i,name in enumerate(names):
    values=[r['D_micro'] for r in rows if r['noise']=='both' and r['policy']==name]
    offsets=np.linspace(-.13,.13,len(values))
    ax[0].scatter(i+offsets,values,color=colors[i],s=25,alpha=.85)
    ax[0].plot([i-.2,i+.2],[np.mean(values)]*2,color='black',lw=2)
ax[0].axhline(9.1791911884,color='#dc2626',ls='--',label='3% rejection threshold = 9.18')
ax[0].set_xticks(range(5),labels);ax[0].set_ylabel(r'New-day $L_C-1.03 L_F$ ($10^{-6}$ pu$^2$)')
ax[0].set_title('Actual AC + voltage and P/Q noise');ax[0].legend(fontsize=8,loc='upper left');ax[0].grid(axis='y',alpha=.2)
variants=['none','pq','v','both'];vlabels=['No new noise','P/Q only','Voltage only','Both']
x=np.arange(4);width=.15
for i,name in enumerate(names):
    vals=[next(r['recoveries'] for r in summary['tables'] if r['noise']==v and r['policy']==name) for v in variants]
    ax[1].bar(x+(i-2)*width,vals,width,color=colors[i],label=labels[i].replace('\n',' '))
ax[1].set_xticks(x,vlabels);ax[1].set_ylim(0,13);ax[1].set_yticks(range(0,13,2));ax[1].set_ylabel('Complete recoveries / 12 new days')
ax[1].set_title('Same AC; four paired measurement variants');ax[1].grid(axis='y',alpha=.2);ax[1].set_axisbelow(True)
ax[1].legend(fontsize=7,loc='upper left',bbox_to_anchor=(0,-.16),ncol=3)
for ext in ['png','pdf']:fig.savefig(out/('ac_effect.'+ext),dpi=180)
plt.close(fig)
print('public aggregate evidence and figure written',out)
