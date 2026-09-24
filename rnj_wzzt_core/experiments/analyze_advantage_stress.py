"""Regenerate figures and comparison tables from a completed stress run."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stress_support import paired_summary, write_json


def analyze(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.read_csv(directory/'results.csv')
    output = directory/'analysis'
    output.mkdir(exist_ok=True)
    scores = frame.groupby(['tier','stress','method']).agg(
        attempts=('status','size'), available=('clade_f1','count'),
        mean_f1=('clade_f1','mean'), exact=('clade_exact','mean'),
        mean_seconds=('elapsed_seconds','mean')).reset_index()
    scores.to_csv(output/'condition_scores.csv', index=False)
    # Paired comparisons always retain missing observations and job identity.
    for ref in ('rnj_RX75','classical_nj_validation','wzzt_rnj_pool','wzzt_rnj_pool_native'):
        rows = paired_summary(frame.to_dict(orient='records'), ref)
        pd.DataFrame(rows).to_csv(output/f'paired_vs_{ref}.csv', index=False)
    for tier in ('geometry','synthetic','ac'):
        current = frame[frame.tier==tier]
        if current.empty:
            continue
        matrix = current.pivot_table(index='method',columns='stress',values='clade_f1',aggfunc='mean')
        fig, ax = plt.subplots(figsize=(max(8,len(matrix.columns)*1.3), max(4,len(matrix)*.43+1.8)))
        plotted = ax.imshow(matrix.to_numpy(),vmin=0,vmax=1,cmap='viridis',aspect='auto')
        ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=25,ha='right')
        ax.set_yticks(range(len(matrix)),matrix.index)
        for i in range(len(matrix)):
            for j in range(len(matrix.columns)):
                value=matrix.iloc[i,j]
                ax.text(j,i,'NA' if np.isnan(value) else f'{value:.3f}',ha='center',va='center',
                        color='black' if value>.65 else 'white',fontsize=8)
        ax.set_title(f'{tier}: mean rooted-clade F1 (fixed benchmark bank)')
        fig.colorbar(plotted,ax=ax,label='F1')
        fig.tight_layout()
        fig.savefig(output/f'{tier}_f1.png',dpi=170)
        plt.close(fig)
        current.groupby(['shape','n','stress','method']).clade_f1.mean().reset_index().to_csv(output/f'{tier}_stratified.csv',index=False)
    structural=frame[frame.method.str.startswith('wzzt_')].copy()
    if 'candidate_truth_recall' in structural and structural.candidate_truth_recall.notna().any():
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for tier,color in [('synthetic','#377eb8'),('ac','#e41a1c')]:
            part=structural[structural.tier==tier]
            axes[0].scatter(part.candidate_truth_recall,part.clade_recall,alpha=.4,s=18,label=tier,c=color)
        axes[0].plot([0,1],[0,1],'k--',lw=1)
        axes[0].set(xlabel='Raw candidate truth recall (diagnostic only)',ylabel='Recovered clade recall',title='Candidate coverage limits recovery')
        axes[0].legend()
        structural['incomplete_or_failed'] = structural.search_incomplete.fillna(False).astype(bool) | structural.status.ne('complete')
        counts=structural.groupby('method').incomplete_or_failed.mean()
        axes[1].barh(counts.index,counts.values,color='#7b6ba8')
        axes[1].set(xlabel='Fraction of incomplete or failed searches',title='All attempted structured models')
        fig.tight_layout()
        fig.savefig(output/'candidate_and_budget.png',dpi=170)
        plt.close(fig)
    failure=frame[frame.status!='complete']
    failure.to_csv(output/'all_noncomplete_rows.csv',index=False)
    write_json(output/'overview.json',{
        'method_rows':len(frame),'job_count':frame.job_id.nunique(),
        'status_counts':frame.status.value_counts().to_dict(),
        'tier_jobs':frame.groupby('tier').job_id.nunique().to_dict(),
        'source_protocol':str(directory/'protocol.json'),
        'interval_interpretation':'Descriptive cluster-bootstrap intervals on this fixed benchmark bank; no multiplicity-adjusted superiority claim.'})
    print(scores.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    analyze(parser.parse_args().directory)