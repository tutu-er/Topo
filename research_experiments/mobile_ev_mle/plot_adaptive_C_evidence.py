"""Read-only plots of a completed frozen evaluation; never tune a policy."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
LABELS={'none':'No EV','constant_fixed':'Constant','fixed_pulse':'Fixed pulse','random_action':'Random','legacy_library_proxy':'Library (11)','fixed_mle_milp':'Fixed MLE\nMILP','refit_milp':'Refit\nMILP'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=HERE/'outputs/adaptive_milp');p.add_argument('--output',type=Path,default=HERE/'evidence');args=p.parse_args()
 audit=args.base/'evaluation_v3_audit/audit_summary.json';metrics=args.base/'evaluation_v3_empirical/history_method_metrics.csv'
 summary=json.loads(audit.read_text(encoding='utf-8'))
 assert summary['evaluation_history_count']==256 and not summary['out_of_population_stress']
 methods=list(LABELS);out=args.output;out.mkdir(exist_ok=True,parents=True)
 strong=[sum(v['methods'][m]['events']['algorithm_unique_correct_retention']['count'] for k,v in summary['descriptive_strata'].items() if k.endswith('_strong')) for m in methods]
 nstrong=sum(v['methods']['none']['events']['unresolved']['history_count'] for k,v in summary['descriptive_strata'].items() if k.endswith('_strong'))
 nweak=sum(v['methods']['none']['events']['unresolved']['history_count'] for k,v in summary['descriptive_strata'].items() if k.endswith('_weak'))
 nstar=summary['descriptive_strata']['star']['methods']['none']['events']['unresolved']['history_count']
 weak=[sum(v['methods'][m]['events']['algorithm_unique_correct_retention']['count'] for k,v in summary['descriptive_strata'].items() if k.endswith('_weak')) for m in methods]
 stars=[summary['descriptive_strata']['star']['methods'][m]['events']['algorithm_unique_correct_retention']['count'] for m in methods]
 assert weak==[0,0,0,1,0,0,0] and stars==[0]*7
 assert (nstar,nweak,nstrong)==(65,102,89)
 fig,ax=plt.subplots(figsize=(9,4.3));positions=np.arange(len(methods));heights=np.array(strong)/nstrong
 ax.bar(positions,heights,color=['#9ba6af','#79aeca','#79aeca','#79aeca','#79aeca','#bea783','#376784'],width=.68)
 for i,k in enumerate(strong):ax.text(i,heights[i]+.014,f'{k}/{nstrong}',ha='center',fontsize=10)
 ax.set_xticks(positions,[LABELS[m] for m in methods],fontsize=10);ax.set_ylim(0,1.03);ax.set_ylabel('Unique correct retention fraction');ax.set_title('Frozen evaluation: strong cherry histories')
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 fig.text(.5,.02,f'Weak cherry: random 1/{nweak}, others 0/{nweak}.  Star: all 0/{nstar}.  Failures retained.',ha='center',fontsize=10);fig.subplots_adjust(bottom=.19,left=.09,right=.99,top=.9)
 fig.savefig(out/'adaptive_C_retention.png',dpi=160);fig.savefig(out/'adaptive_C_retention.pdf');plt.close(fig)
 with metrics.open(encoding='utf-8-sig',newline='') as f:rows=[row for row in csv.DictReader(f) if row['method']=='refit_milp']
 assert len(rows)==256
 fig,ax=plt.subplots(figsize=(7.4,4.8));counts={}
 for group,color,marker in [('star','#777','o'),('weak_cherry','#78a8c4','o'),('strong_cherry','#ba652e','^')]:
  points=[]
  for row in rows:
   if row['stratum']!=group or not row['wrong_margin']:continue
   interval=json.loads(row['refit_proxy_gain_vs_constant_fixed']) if row['refit_proxy_gain_vs_constant_fixed'] else None
   if interval and interval.get('lower') is not None and interval.get('upper') is not None:
    points.append(((interval['lower']+interval['upper'])/2,float(row['margin_gain_vs_constant_fixed'])))
  counts[group]=len(points)
  if points:
   a=np.array(points);ax.scatter(a[:,0],a[:,1],s=23,alpha=.65,c=color,marker=marker,label=f'{group.replace("_"," ")} (n={len(points)})')
 ax.axhline(0,color='black',lw=.7);ax.axvline(0,color='black',lw=.7);ax.set_xscale('symlog',linthresh=.05);ax.set_yscale('symlog',linthresh=5)
 ax.set_xticks([-100,-1,0,1,100,1000],['-100','-1','0','1','100','1000']);ax.set_yticks([-100,-10,0,10,100,1000],['-100','-10','0','10','100','1000'])
 ax.set_xlabel('Selected refit proxy gain versus constant (interval midpoint)');ax.set_ylabel('Actual AC signed loss-margin gain versus constant');ax.set_title('Model-distance gain and actual discrimination gain');ax.legend(frameon=False);ax.grid(alpha=.2);fig.tight_layout()
 fig.savefig(out/'adaptive_C_proxy_actual.png',dpi=160);fig.savefig(out/'adaptive_C_proxy_actual.pdf');plt.close(fig)
 outputs=['adaptive_C_retention.png','adaptive_C_retention.pdf','adaptive_C_proxy_actual.png','adaptive_C_proxy_actual.pdf']
 manifest=dict(evaluation_histories=256,strata_counts=dict(star=nstar,weak_cherry=nweak,strong_cherry=nstrong),strong_unique_counts=dict(zip(methods,strong)),weak_unique_counts=dict(zip(methods,weak)),star_unique_counts=dict(zip(methods,stars)),scatter_available_count=counts,truth_used_for_post_evaluation_strata_only=True,policy_or_gate_tuning_performed=False,proxy_interval_midpoint_not_AC_bound=True,axes_symlog_thresholds=dict(proxy=.05,actual=5),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [audit,metrics]},output_sha256={f:hashlib.sha256((out/f).read_bytes()).hexdigest() for f in outputs})
 (out/'adaptive_C_plot_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(strong_counts=strong,scatter_counts=counts)))
if __name__=='__main__':main()
