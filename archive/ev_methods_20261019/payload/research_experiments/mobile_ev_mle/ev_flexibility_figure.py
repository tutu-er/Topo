"""Render one declared development example, showing both gain and failure."""
from pathlib import Path
import gzip,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent

def main():
    raw=HERE/'evidence/ev_flexibility/v1_development/compact_full_diagnostics/trial_981004_spatial.json.gz'
    trial=json.loads(gzip.decompress(raw.read_bytes()));by={r['method']:r for r in trial['records']}
    methods=('charge_fixed','charge_mobile','v2g_fixed')
    labels=('Fixed charging','Mobile charging','Fixed V2G / ESS')
    colours=('#2369A0','#24846C','#BD6140')
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.6),layout='constrained')
    for method,label,colour,style,order in zip(methods,labels,colours,('--','-','-.'),(3,2,1)):
        row=by[method];site=row['action']['site'];power=np.array(row['action']['p'])[:,site]*20
        axes[0].stairs(power,np.arange(5)*.25,label=label,color=colour,linewidth=2.2,linestyle=style,zorder=order)
    axes[0].axhline(0,color='#777777',linewidth=.8);axes[0].set(xlabel='Time (h)',ylabel='Net grid power (kW)',title='Same departure battery energy')
    axes[0].legend(fontsize=8,loc='lower left');axes[0].set_ylim(-6.3,7)
    for ax,key,title in ((axes[1],'predicted_score','Local information score'),(axes[2],'signed_margin_full_SSE','Complete AC discrimination margin')):
        values=[by[m]['selection'][key] if key=='predicted_score' else by[m][key] for m in methods]
        ax.bar(np.arange(3),values,color=colours,width=.58);ax.axhline(0,color='#777777',linewidth=.8)
        ax.set_xticks(np.arange(3),('Fixed\ncharge','Mobile\ncharge','Fixed\nV2G'),fontsize=9);ax.set_title(title,fontsize=11)
        for x,y in enumerate(values):ax.annotate(f'{y:.2f}',(x,y),xytext=(0,5 if y>=0 else -13),textcoords='offset points',ha='center',fontsize=9)
        ax.margins(y=.20)
    axes[1].set_ylabel('Planning score (not a probability)');axes[2].set_ylabel('Wrong-model SSE minus true-model SSE')
    fig.suptitle('Development H981004: same departure energy, different locations and charging curves',fontsize=10)
    target=HERE/'evidence/ev_flexibility/curve_example.png';fig.savefig(target,dpi=180);plt.close(fig)
    print(target)

if __name__=='__main__':main()
