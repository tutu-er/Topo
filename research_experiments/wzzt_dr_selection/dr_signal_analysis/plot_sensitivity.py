"""Model-only strength/duration curves from scalar counterfactual results."""
from pathlib import Path
import argparse,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS=['#2166ac','#dd8b29','#27875c','#82589f','#b74141']
LABELS=['paper15 / b2','flynn16 / b4','flynn16 / b5','pengwah18 / b5','flynn16 / b7']
MARKERS=['o','s','^','D','v']

def save(fig,out,stem):
    for extension in ('png','svg','pdf'):
        fig.savefig(out/(stem+'.'+extension),dpi=220,bbox_inches='tight')
    plt.close(fig)


def plot(source,out):
    a=json.loads(Path(source).read_text(encoding='utf-8'));out=Path(out);out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.titleweight':'bold'})
    cases=list(dict.fromkeys(t['condition'] for t in a['tables']))
    duration=[x for x in a['aggregate'] if x['kind']=='duration']
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.4),sharex=True)
    for i,name in enumerate(cases):
        t=sorted([t for t in a['tables'] if t['condition']==name and t['kind']=='duration'],key=lambda t:t['value'])
        x=[t['value'] for t in t];style=dict(color=COLORS[i],marker=MARKERS[i],ms=4,lw=1.2,alpha=.88)
        axes[0,0].plot(x,[t['D1_gain']['mean']*1e6 for t in t],**style)
        axes[0,1].plot(x,[t['D2_gain']['mean']*1e6 for t in t],**style)
        axes[1,0].plot(x,[t['beta'] for t in t],**style)
        axes[1,1].plot(x,[t['hard_payment_bound'] for t in t],**style)
    x=[d['value'] for d in duration]
    axes[0,0].plot(x,[d['D1_gain_mean']*1e6 for d in duration],color='black',lw=2.2,marker='o',ms=4)
    axes[0,1].plot(x,[d['D2_gain_mean']*1e6 for d in duration],color='black',lw=2.2,marker='o',ms=4)
    axes[1,0].plot(x,np.sqrt(2/np.asarray(x)),color='#333333',ls=':',lw=1.6)
    axes[1,1].axhline(1.6,color='#555555',ls='--',lw=1)
    titles=['A  L1 prediction-distance increase','B  RMS prediction-distance increase',
        'C  Amplitude relative to the original 2 h action','D  Required hard-payment bound']
    ylabels=[r'$\Delta D_1$ ($10^{-6}$ p.u.$^2$)',r'$\Delta D_2$ ($10^{-6}$ p.u.$^2$)',
        r'Amplitude multiplier $\beta$',r'Payment bound (budget $\leq 1.6$)']
    for ax,title,ylabel in zip(axes.flat,titles,ylabels):
        ax.set_title(title,loc='left',fontsize=10);ax.set_ylabel(ylabel)
        ax.axvline(2,color='#777777',ls='--',lw=.9,alpha=.7);ax.set_xticks(x)
        ax.set_axisbelow(True);ax.grid(color='#dddddd',lw=.6,alpha=.7)
    for ax in axes[1]:ax.set_xlabel('Reduction hours per day (equal-duration rebound)')
    axes[1,0].set_ylim(0,1.55);axes[1,1].set_ylim(.75,1.68)
    handles=[Line2D([0],[0],color=c,marker=m,lw=1.4,label=l) for c,m,l in zip(COLORS,MARKERS,LABELS)]
    handles += [Line2D([0],[0],color='black',lw=2.2,label='Mean across 19 valid days')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.965),ncol=3,frameon=False,fontsize=9)
    fig.suptitle('Same-budget duration sensitivity: frozen models, sites and relative prices',y=1.005,fontsize=13,fontweight='bold')
    fig.text(.07,.022,'Model-only counterfactual on archived physical P/Q: 19 valid days / 20 planned. No new AC, observations or selection.',fontsize=9,color='#555555')
    fig.text(.07,.002,'Dashed vertical: original 2 h reduction + 2 h rebound. Dotted curve in C: budget-only amplitude scaling.',fontsize=9,color='#555555')
    fig.subplots_adjust(left=.07,right=.985,top=.86,bottom=.12,hspace=.30,wspace=.23)
    save(fig,out,'duration_same_budget')

    amplitude=[x for x in a['aggregate'] if x['kind']=='amplitude']
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5))
    for i,name in enumerate(cases):
        t=sorted([t for t in a['tables'] if t['condition']==name and t['kind']=='amplitude'],key=lambda t:t['value'])
        axes[0].plot([t['value'] for t in t],[t['D1_gain']['mean']*1e6 for t in t],
            color=COLORS[i],marker=MARKERS[i],lw=1.2,ms=4,alpha=.85)
    x=np.asarray([d['value'] for d in amplitude]);y=[d['D1_gain_mean']*1e6 for d in amplitude]
    axes[0].plot(x,y,color='black',lw=2.2,marker='o',ms=5)
    axes[1].plot(x,1.6*x*x,color='#333333',lw=2,marker='o',ms=5)
    axes[1].axhline(1.6,color='#555555',ls='--',lw=1)
    for ax in axes:
        ax.axvspan(1,math.sqrt(2)+.02,color='#f1d7b8',alpha=.42)
        ax.axvline(1,color='#666666',ls='--',lw=1)
        ax.set_xlabel(r'Amplitude multiplier $\alpha$ (original shape fixed)')
        ax.set_xticks(x,['0','.25','.5','.75','1',r'$\sqrt{2}$'])
        ax.set_axisbelow(True);ax.grid(color='#dddddd',lw=.6,alpha=.7)
    axes[0].set_title('A  Predicted distance gain',loc='left',fontsize=11)
    axes[0].set_ylabel(r'$\Delta D_1$ ($10^{-6}$ p.u.$^2$)')
    axes[1].set_title(r'B  Payment grows as $\alpha^2$',loc='left',fontsize=11)
    axes[1].set_ylabel('Required hard-payment bound')
    axes[1].annotate('Original budget 1.6',xy=(1,1.6),xytext=(.12,2.2),arrowprops=dict(arrowstyle='->',color='#555555'),fontsize=9)
    axes[1].annotate('2x budget: 3.2',xy=(math.sqrt(2),3.2),xytext=(.65,3.6),arrowprops=dict(arrowstyle='->',color='#555555'),fontsize=9)
    axes[1].set_ylim(-.05,4.)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.985),ncol=3,frameon=False,fontsize=9)
    fig.text(.07,.035,'Archived physical P/Q counterfactuals. Shading exceeds the original budget; resource checks only, no new delivered experiments.',fontsize=9,color='#555555')
    fig.subplots_adjust(left=.07,right=.985,top=.79,bottom=.17,wspace=.25)
    save(fig,out,'strength_and_budget')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(a.source,a.output)
