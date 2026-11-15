"""Compact scientific comparison from a completed independent report."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from ... import io


def plot(report_path, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    report_path=Path(report_path).resolve();source=io.ref(report_path);data=io.read(report_path)
    if data.get('status')!='complete' or not data.get('sources_unchanged'):
        raise ValueError('Completed independently audited report required')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    cases=data['cases'];modes=data['modes'];x=np.arange(len(modes))
    families=[('unbanded','Single target'),('band','Node/hour band')]
    if data['variant']=='band_sdp':families.append(('band_sdp','Band + SDP seeds'))
    colors=['#586F9B','#D3813C','#33958A'];width=.75/len(families)
    mode_labels={'normal_fixed':'Normal','mgf_fixed':'MGF fixed','mgf_forecast':'MGF forecast'}
    with plt.rc_context({'font.size':10,'axes.titlesize':12,'axes.labelsize':11,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(3,len(cases),figsize=(4.2*len(cases),9.0),squeeze=False)
        for col,case in enumerate(cases):
            short=case['dimensions']['case'].replace('soumalas11','Soumalas').replace('pengwah18','Pengwah')
            title=f"{short}, noise x{case['dimensions'].get('meter_multiplier','?')}"
            zero=case['strategies']['no_dfs'];n=zero['n']
            for family_index,(prefix,label) in enumerate(families):
                values=[case['strategies'].get(prefix+'__'+mode) for mode in modes]
                recovery=[v['recovered'] if v else np.nan for v in values]
                fees=[v['settled_total_yuan']['mean'] if v else np.nan for v in values]
                evidence=[v['historical_fixed_pair']['increments']['clean_logodds']['mean'] if v else np.nan for v in values]
                pos=x+(family_index-(len(families)-1)/2)*width
                for row,y in enumerate((recovery,fees,evidence)):
                    bars=axes[row,col].bar(pos,y,width,label=label,color=colors[family_index],alpha=.92)
                    for bar,value in zip(bars,values):
                        if value and value['zero_fallback_rows']:
                            bar.set_hatch('//');bar.set_edgecolor('#333333')
                for posi,y,value in zip(pos,recovery,values):
                    if np.isfinite(y):
                        suffix=f" F{value['zero_fallback_rows']}" if value['zero_fallback_rows'] else ''
                        axes[0,col].text(posi,y+max(.1,n*.025),str(int(y))+suffix,ha='center',va='bottom',fontsize=8)
            axes[0,col].set_title(title)
            axes[0,col].axhline(zero['recovered'],ls=':',lw=1.4,color='#333333',label='No procurement')
            axes[0,col].set_ylim(0,n+2);axes[0,col].set_yticks(sorted(set((0,n//2,n))))
            for row in range(3):
                axes[row,col].set_xticks(x,[mode_labels[m] for m in modes])
                axes[row,col].grid(axis='y',alpha=.2);axes[row,col].set_axisbelow(True)
            axes[2,col].axhline(0,color='#444444',lw=.8)
        axes[0,0].set_ylabel('Internal recovery / paired paths')
        axes[1,0].set_ylabel('Mean total settlement (CNY)')
        axes[2,0].set_ylabel('Clean AC log-score gain')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=len(labels),bbox_to_anchor=(.5,1.0),frameon=False)
        prefix='SMOKE: ' if data.get('smoke') else ''
        fig.text(.5,.015,prefix+'Known days and paired paths. F = zero fallback (hatched), not a feasible quote. SDP seed time is additional.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.045,1,.95))
        fig.savefig(out/'comparison.png',dpi=200);fig.savefig(out/'comparison.pdf');plt.close(fig)
    if io.ref(report_path)!=source:raise ValueError('Report changed while plotting')
    receipt=dict(status='complete',source_report=source,source=io.ref(__file__),
        figures=[io.ref(out/name) for name in ('comparison.png','comparison.pdf')],
        scope='Aggregate audited internal recovery, total settlement and fixed historical pair clean-AC evidence; no additional experiments')
    io.write(out/'receipt.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();print(plot(args.report,args.out)['status'])
