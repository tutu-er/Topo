"""Static scientific figures from scalar-only frozen-data extraction."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS={'zero':'#7f8c8d','rotation':'#dd8b29','optimized':'#2166ac'}
NAMES={'zero':'No DR','rotation':'Rotation','optimized':'Designed DR'}
LABELS=['paper15 / b2','flynn16 / b4','flynn16 / b5','pengwah18 / b5','flynn16 / b7']

def plot(source,out):
    source=Path(source);out=Path(out);out.mkdir(exist_ok=False,parents=True)
    data=json.loads(source.read_text());records=[r for r in data['records'] if r['status']=='complete']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.titleweight':'bold'})
    fig,axes=plt.subplots(2,1,figsize=(10.5,7.4),sharex=True,gridspec_kw={'height_ratios':[1,1]})
    x=np.arange(5);width=.22
    for pol,offset in [('zero',-width),('rotation',0),('optimized',width)]:
        vals=[next(a for a in c['policies'] if a['policy']==pol) for c in data['cases']]
        axes[0].bar(x+offset,[v['distance_controlled_micro'] for v in vals],width,
            color=COLORS[pol],alpha=.85,label=NAMES[pol])
        if pol!='zero':
            gain_offset=width*(-.55 if pol=='rotation' else .55)
            axes[1].bar(x+gain_offset,[v['distance_gain_micro'] for v in vals],width*.88,
                color=COLORS[pol],alpha=.85,label=NAMES[pol])
            for i,c in enumerate(data['cases']):
                samples=[r['distance_gain_micro'] for r in records if r['condition']==c['condition'] and r['policy']==pol]
                jitter=np.linspace(-width*.22,width*.22,len(samples))
                axes[1].scatter(i+gain_offset+jitter,samples,s=19,color=COLORS[pol],edgecolor='white',linewidth=.45,zorder=5)
    for i,c in enumerate(data['cases']):
        gamma=c['policies'][0]['frozen_gamma_micro']
        axes[1].hlines(gamma,i-.22,i+.30,color='#82589f',linestyle='--',lw=1.5,zorder=4)
        axes[0].text(i,165,f"{c['valid_days']}/{c['planned_days']} days",ha='center',fontsize=9,color='#555555')
    axes[0].set_ylim(0,178)
    axes[0].set_ylabel(r'Prediction distance $D$ ($10^{-6}$ p.u.$^2$)')
    axes[0].set_title('A  Baseline and post-DR fitted-candidate prediction distance',loc='left')
    axes[0].legend(frameon=False,ncol=3,loc='upper right',bbox_to_anchor=(1,.80))
    axes[1].axhline(0,color='#5b5b5b',lw=.85)
    axes[1].set_ylabel(r'Paired distance increase $\Delta D$ ($10^{-6}$ p.u.$^2$)')
    axes[1].set_title('B  Same-day increase; points are the individual predeclared days',loc='left')
    axes[1].set_xticks(x,LABELS)
    axes[1].legend(handles=[Line2D([0],[0],color='#82589f',ls='--',lw=1.5,
        label=r'Frozen increment target $\gamma$ (not a rejection threshold)')],frameon=False,loc='upper left')
    axes[1].set_ylim(-.6,13.8)
    for a in axes:a.set_axisbelow(True);a.grid(axis='y',color='#dddddd',lw=.6,alpha=.7)
    fig.text(.09,.012,'Distances use observed P/Q from the completed simulated runs.\n19 valid paired days / 20 planned; one flynn16/b7 day failed resource qualification and remains unresolved.',fontsize=8.7,color='#555555',va='bottom')
    fig.subplots_adjust(left=.09,right=.985,bottom=.09,top=.94,hspace=.20)
    save(fig,out,'distance_baseline_increment')

    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4))
    for ax,(condition,path,rid,title) in zip(axes,[
        ('paper15_b2_v0005',1501,'w1_tau2pct','paper15 / b2 / day 1501; w = 1, new-day band = 2%'),
        ('flynn16_b4_v0005',1512,'w2_tau1pct','flynn16 / b4 / day 1512; w = 2, new-day band = 1%')]):
        group=[next(r for r in data['rule_records'] if r['condition']==condition and r['path']==path and r['rule_id']==rid and r['policy']==pol)
            for pol in ('zero','rotation','optimized')]
        values=[r['exit_margins_micro']['coarse'] for r in group]
        ax.bar(np.arange(3),values,width=.60,color=[COLORS[p] for p in ('zero','rotation','optimized')],alpha=.88)
        ax.axhline(0,color='#228344',lw=1.3,ls='--')
        for i,r in enumerate(group):
            v=values[i];ax.text(i,v+(.025 if v>=0 else -.025),f'{v:+.3f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=10)
        ax.set_xticks(np.arange(3),[NAMES[p] for p in ('zero','rotation','optimized')])
        ax.set_title(title,fontsize=10.5,pad=14)
        ax.set_ylabel(r'Coarse-candidate exit margin ($10^{-6}$ p.u.$^2$)')
        ax.grid(axis='y',color='#dddddd',lw=.6,alpha=.7);ax.set_axisbelow(True)
        lo=min(values);hi=max(values);span=hi-lo
        ax.set_ylim(lo-.22*span,hi+.25*span)
    fig.text(.09,.035,'Observed P/Q/V; zero is the criterion boundary: positive excludes the coarse candidate.\nPrespecified sensitivity rules; the primary rule is reported separately.',fontsize=8.7,color='#555555')
    fig.subplots_adjust(left=.09,right=.985,bottom=.16,top=.84,wspace=.32)
    save(fig,out,'actual_exit_margin_examples')
    print(json.dumps(dict(status='complete',files=sorted(p.name for p in out.iterdir()),new_experiments=0)))

def save(fig,out,name):
    for suffix in ('png','svg','pdf'):fig.savefig(out/(name+'.'+suffix),dpi=220)
    plt.close(fig)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    x=a.parse_args();plot(x.source,x.output)
