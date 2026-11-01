"""Direct evaluation of the same frozen prediction-distance planning goal."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
COLORS=['#2166ac','#dd8b29','#27875c','#82589f','#b74141']
LABELS=['paper15 / b2','flynn16 / b4','flynn16 / b5','pengwah18 / b5','flynn16 / b7']
MARKERS=['o','s','^','D','v']

def plot(source,out):
    a=json.loads(Path(source).read_text(encoding='utf-8'));out=Path(out);out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.titleweight':'bold'})
    cases=list(dict.fromkeys(t['condition'] for t in a['tables']))
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.8))
    for i,name in enumerate(cases):
        rows=sorted([r for r in a['tables'] if r['condition']==name and r['kind']=='duration'],key=lambda r:r['value'])
        x=[r['value'] for r in rows]
        style=dict(color=COLORS[i],marker=MARKERS[i],ms=4,lw=1.5,alpha=.9)
        axes[0].plot(x,[r['sample_90pct_lower_gain']/r['original_gamma_native'] for r in rows],**style)
        axes[1].plot(x,[100*r['fraction_above_original_gamma'] for r in rows],**style)
    axes[0].axhline(1,color='#555555',ls='--',lw=1.1)
    axes[1].axhline(90,color='#27875c',ls='--',lw=1.1)
    axes[0].set_title('A  90%-attainable gain / original target',loc='left',fontsize=11)
    axes[1].set_title('B  Scenarios reaching the original target',loc='left',fontsize=11)
    axes[0].set_ylabel(r'Empirical $\hat{\gamma}_{90}$ / frozen $\gamma$')
    axes[1].set_ylabel('Fraction of working scenarios (%)')
    axes[0].set_ylim(.4,1.9);axes[1].set_ylim(0,104)
    for ax in axes:
        ax.axvline(2,color='#777777',ls=':',lw=1)
        ax.set_xticks([1,2,4,6,8,12]);ax.set_xlabel('Reduction hours per day (equal-duration rebound)')
        ax.set_axisbelow(True);ax.grid(color='#dddddd',lw=.6,alpha=.7)
    handles=[Line2D([0],[0],color=c,marker=m,lw=1.5,label=l) for c,m,l in zip(COLORS,MARKERS,LABELS)]
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.52,1.005),frameon=False,fontsize=9)
    fig.text(.075,.035,'Same original 1000 working-law draws per case; original targets unchanged. Budget cap 1.6, no new observations or AC.',fontsize=9,color='#555555')
    fig.text(.075,.005,'Dashed levels: target gain and 90% requirement. Dotted vertical: original 2 h reduction + 2 h rebound.',fontsize=9,color='#555555')
    fig.subplots_adjust(left=.075,right=.99,top=.80,bottom=.20,wspace=.25)
    for extension in ('png','svg','pdf'):fig.savefig(out/('planning_objective_duration.'+extension),dpi=220,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    x=p.parse_args();plot(x.source,x.output)
