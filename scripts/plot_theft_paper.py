import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'theft_wzzt/outputs/theft_paper_v2'
s=json.loads((OUT/'summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors=['#7d8fa3','#e29b42','#206b85']
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
cases=['paper15','soumalas11','flynn16'];x=np.arange(3)
for j,method in enumerate(['stable_positive_lp','fixed_stable','shrinkage']):
 vals=[];low=[];high=[]
 for case in cases:
  if method=='stable_positive_lp':r=next(v for v in s['paired_point_ablation'] if v['case']==case)['unrestricted']
  else:r=next(v for v in s['localization'] if v['case']==case and v['method']==method)['point']
  vals.append(100*r['rate']);low.append(100*(r['rate']-r['ci95'][0]));high.append(100*(r['ci95'][1]-r['rate']))
 axes[0,0].bar(x+(j-1)*.24,vals,.23,color=colors[j],label=['Free edge refit','Fixed R/X','Shrinkage'][j],yerr=[low,high],capsize=2)
axes[0,0].set(xticks=x,xticklabels=cases,ylim=(0,105),ylabel='Point accuracy (%)',title='A  Paired fresh tests (100 events per network)')
axes[0,0].legend(loc='lower left',fontsize=8)
rows=[next(r for r in s['localization'] if r['case']==c and r['method']=='shrinkage') for c in cases]
axes[0,1].bar(x,[100*r['region_coverage']['rate'] for r in rows],.55,color=colors[2])
axes[0,1].axhline(95,color='#9c3d42',ls='--',lw=1,label='95% target')
axes[0,1].set(xticks=x,xticklabels=cases,ylim=(0,105),ylabel='Candidate-set coverage (%)',title='B  Region coverage and set size')
for i,r in enumerate(rows):axes[0,1].text(i,10,f"mean {r['region_size_mean']:.2f} / {r['candidate_count']}\nregions",ha='center',color='white')
axes[0,1].legend(loc='upper center',fontsize=8)
conds=['nominal','drift10','hetero10','loss_under25'];xx=np.arange(4)
for j,(key,label) in enumerate([('transferred_nominal_fpr','Nominal calibration'),('null_fpr','Known-condition calibration'),('composite_fpr','Composite null bank')]):
 vals=[100*next(r for r in s['groups'] if r['case']=='paper15' and r['condition']==c and r['method']=='shrinkage')[key]['rate'] for c in conds]
 axes[1,0].bar(xx+(j-1)*.24,vals,.23,color=colors[j],label=label)
axes[1,0].axhline(5,color='#9c3d42',ls='--',lw=1)
axes[1,0].set(xticks=xx,xticklabels=['Nominal','R/X +10%','Edge +/-10%','Loss -25%'],ylim=(0,105),ylabel='False alarms (%)',title='C  Calibration transfer (100 nulls per condition)')
axes[1,0].legend(loc='upper left',fontsize=8)
for j,(key,label) in enumerate([('composite_detection','Detection'),('amplitude_window_coverage','Whole-window amplitude coverage')]):
 vals=[100*r[key]['rate'] for r in rows]
 axes[1,1].bar(x+(j-.5)*.3,vals,.29,color=[colors[2],colors[1]][j],label=label)
axes[1,1].axhline(95,color='#9c3d42',ls='--',lw=1)
axes[1,1].set(xticks=x,xticklabels=cases,ylim=(0,105),ylabel='Rate (%)',title='D  Composite alarm and amplitude intervals')
axes[1,1].legend(loc='lower left',fontsize=8)
fig.suptitle('Frozen V2 study: regional inference, calibration and limitations',fontsize=15)
fig.savefig(OUT/'paper_results.png',dpi=180);fig.savefig(OUT/'paper_results.svg')
print(OUT/'paper_results.png')
