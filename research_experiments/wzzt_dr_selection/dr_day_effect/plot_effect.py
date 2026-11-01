"""Static actual-effect chart from public scalar evidence only."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(source,output):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    gaps=json.loads((source/'gaps.json').read_text(encoding='utf-8'))['rows'];summary=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),gridspec_kw={'width_ratios':[1.25,1]})
    fig.suptitle('Prespecified main rule: historical 3%, new day 2%, new-day weight 2',fontsize=12,y=1.045,fontweight='bold')
    x=np.arange(len(gaps));width=.24
    axes[0].bar(x-width,[r['needed_increment_micro'] for r in gaps],width,color='#a1a9b2',label='Needed: 120% of baseline shortfall')
    axes[0].bar(x,[r['rotation_low']['provided_increment_micro'] for r in gaps],width,color='#dd8b29',label='Matched-cost rotation')
    axes[0].bar(x+width,[r['optimized_low']['provided_increment_micro'] for r in gaps],width,color='#2166ac',label='Optimized low-cost curve')
    axes[0].set_xticks(x,[f"D{r['day']}" for r in gaps]);axes[0].set_ylabel(r'Validation-margin increase ($10^{-6}$ p.u.$^2$)')
    axes[0].set_title('A  Six days that were wrong without DR',loc='left',fontsize=11,fontweight='bold')
    axes[0].legend(frameon=False,fontsize=8.5,loc='upper right');axes[0].axhline(0,color='#555555',lw=.8)
    a=summary['tables'];y=np.array([t['attained']/12*100 for t in a]);lo=np.array([t['attainment_ci95'][0]*100 for t in a]);hi=np.array([t['attainment_ci95'][1]*100 for t in a])
    colors=['#7f8c8d','#dd8b29','#2166ac','#e7b36c','#72a6d3'];xx=np.arange(5)
    axes[1].bar(xx,y,color=colors,width=.62)
    axes[1].errorbar(xx,y,yerr=np.stack([y-lo,hi-y]),fmt='none',color='#333333',capsize=4,lw=1.1)
    for i,t in enumerate(a):axes[1].text(i,hi[i]+2,f"{t['attained']}/12",ha='center',fontsize=9)
    axes[1].axhline(90,color='#82589f',ls='--',lw=1.1);axes[1].text(4.3,91.5,'90% target',ha='right',fontsize=9,color='#82589f')
    axes[1].set_xticks(xx,['No DR','Rotate\nlow','Optimize\nlow','Rotate\nhigh','Optimize\nhigh'])
    axes[1].set_ylim(0,104);axes[1].set_ylabel('Observed excitation attainment (%)')
    axes[1].set_title('B  Twelve prelocked independent simulated days',loc='left',fontsize=10.5,fontweight='bold')
    for ax in axes:ax.set_axisbelow(True);ax.grid(axis='y',color='#dddddd',lw=.6,alpha=.7)
    fig.text(.07,.045,'Error bars: per-policy exact 95% binomial intervals. Optimized successes were the same six days already correct without DR.',fontsize=8.5,color='#555555')
    fig.text(.07,.005,'60/60 AC strategy-days completed; one planning-qualified case out of five. Fixed historical candidates/RX, no refit or new W search.',fontsize=8.5,color='#555555')
    fig.subplots_adjust(left=.07,right=.985,top=.90,bottom=.19,wspace=.28)
    for suffix in ('png','svg','pdf'):
        p=output/('actual_effect.'+suffix);fig.savefig(p,dpi=220,bbox_inches='tight')
        if suffix=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(a.source,a.output)
