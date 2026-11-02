"""Plot aggregate scalar validation results, never export P/Q/V or prices."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path(__file__).resolve().parent
out=base/'public';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(2,2,figsize=(12.3,8.4),constrained_layout=True)
colors={'zero':'#83929e','continuous':'#2265ac','rotation':'#df8e32','single':'#5d9983'}
names={'zero':'No DR','continuous':'Continuous design','rotation':'Equal-cost rotation','single':'Equal-cost single'}
stats=[]
for col,(folder,title) in enumerate([('run_v1','Original mean-bound design'),('delivery_v1','Delivery-derated diagnostic')]):
    summary=json.loads((base/folder/'summary.json').read_text(encoding='utf-8'))
    scalar=np.load(base/folder/'evaluation_scalars.npz')
    rows=summary['rows'];x=np.arange(len(rows));prob=np.array([r['event_rate'] for r in rows])
    joint=np.array([r['event_and_resource_valid']/r['paths'] for r in rows])
    ax=axes[0,col];ax.bar(x-.17,prob,width=.34,color=[colors[r['policy']] for r in rows],label='Final recovery / exclusion')
    ax.bar(x+.17,joint,width=.34,color=[colors[r['policy']] for r in rows],alpha=.35,hatch='//',label='Recovery AND device bounds')
    ax.axhline(.95,ls='--',lw=1.2,color='#ac3030',label='95% excitation requirement')
    for i,p in enumerate(prob):ax.text(i-.17,p+.018,f'{100*p:.2f}%',ha='center',fontsize=9)
    ax.set_xticks(x,[names[r['policy']].replace('Equal-cost ','Equal-cost\n').replace('Continuous ','Continuous\n') for r in rows])
    ax.set_ylim(0,1.10);ax.set_ylabel('Fraction of 2,000 independent whole-day paths')
    ax.set_title(title+f"\nPaid plans: {rows[1]['planned_cost']:.2f} contract cost units",fontsize=11)
    if col==0:ax.legend(loc='upper left',fontsize=8)
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    ax=axes[1,col]
    for i,r in enumerate(rows):
        d=scalar[r['policy']+'__D'];q=np.quantile(d,[.05,.5,.95]);mean=float(np.mean(d))
        ax.errorbar(i,q[1],yerr=np.array([[q[1]-q[0]],[q[2]-q[1]]]),fmt='o',capsize=5,color=colors[r['policy']],lw=2)
        ax.plot(i,mean,'x',color='black',ms=6)
        stats.append(dict(batch=folder,policy=r['policy'],D_q05_median_q95=q.tolist(),D_mean=mean,
            required_h=r['threshold'],paths=r['paths'],cost=r['planned_cost']))
    ax.axhline(rows[0]['threshold'],ls='--',color='#ac3030',label=f"Required D > {rows[0]['threshold']:.3f}")
    ax.set_xticks(x,[names[r['policy']].replace('Equal-cost ','Equal-cost\n').replace('Continuous ','Continuous\n') for r in rows])
    ax.set_ylabel(r'$D=\ell_C-1.03\ell_F$ (micro p.u.$^2$)')
    ax.set_title('Median and 5–95% range; black x = mean',fontsize=10)
    ax.grid(axis='y',alpha=.18);ax.legend(fontsize=8,loc='upper left')
fig.suptitle('Frozen RNJ + WZZT candidates, complete history, original 3% validation\nGaussian working-model replay; no new AC or field validation',fontsize=13)
fig.savefig(out/'conditional_dr_effects.png',dpi=180)
fig.savefig(out/'conditional_dr_effects.pdf')
(out/'figure_statistics.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf-8')
print(str(out/'conditional_dr_effects.png'))
