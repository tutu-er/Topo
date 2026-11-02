"""Publish scalar paired results and standard scientific plots, no trajectories."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args();root,out=args.run,args.output;out.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=read(root/'summary.json');audit=read(root/'audit.json')
assert audit['status']=='passed' and (root/'ended.json').exists()
for name in ['protocol.json','summary.json','scored.json','audit.json','problem_frozen.json']:
    shutil.copyfile(root/name,out/name)
order=['zero','legacy_iid_pair','fixed_correlated','open_correlated','rotation_uniform','rotation_selected_capacity']
labels=['No DR','Legacy pair','Fixed pair','Open nodes','Uniform\nrotation','Selected-cap.\nrotation']
lookup={(r['law'],r['policy']):r for r in s['tables']};x=np.arange(6);width=.35
plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(13.2,8),layout='constrained')
for col,law in enumerate(['main','misspec']):
    rows=[lookup[law,k] for k in order]
    for ax,key1,key2,title,unit in [(axes[0,col],'old_errors','errors','Final missing + extra blocks','Total over 12 paths'),(axes[1,col],'raw_abs_kwh_per_day','abs_kwh_per_day','Absolute delivered DR','kWh / day')]:
        before=[r[key1] for r in rows];after=[r[key2] for r in rows]
        b1=ax.bar(x-width/2,before,width,color='#B9C2CA',label='Unbounded delivery')
        b2=ax.bar(x+width/2,after,width,color='#2878A0',label='Device-saturated delivery')
        for bars,values in [(b1,before),(b2,after)]:
            ax.bar_label(bars,labels=[f'{v:.0f}' if key1=='old_errors' else f'{v:.1f}' for v in values],fontsize=8,padding=2)
        ax.set_title(('Main response law' if law=='main' else 'Response-law misspecification')+' — '+title,loc='left',fontsize=11)
        ax.set_ylabel(unit);ax.set_xticks(x,labels,rotation=25,ha='right');ax.set_ylim(0,max(before+after)*1.25+1)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.get_xticklabels()[3].set_color('#136181');ax.get_xticklabels()[3].set_fontweight('bold')
axes[0,0].legend(fontsize=8)
fig.suptitle('Same prices and random paths: effect of device saturation',fontsize=15,fontweight='bold')
fees=[lookup['main',k]['fee_per_day'] for k in order]
fig.supxlabel('12 shared background paths, two paired response laws; frozen historical R/X and original selection rule.\n'
    +'Daily prepaid fees unchanged: '+', '.join(f'{v:.2f}' for v in fees)+'.  After saturation: zero device/voltage violations; 13,824 AC points converged.',fontsize=9)
for ext in ['png','pdf','svg']:
    p=out/('saturation_comparison.'+ext);fig.savefig(p,dpi=170,bbox_inches='tight')
    if ext=='svg':p.write_text('\n'.join(l.rstrip() for l in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
plt.close(fig)
manifest=dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'},
    source=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__))),
    private_refs={name:dict(path=str(root/name),sha256=sha(root/name)) for name in ['price_freeze.json','observations_frozen.json','predictions_frozen.json','ended.json']},
    privacy='Scalar results, configuration and outcome figures only; no household/P/Q/V/RX/u/price/random trajectories.')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(files=len(manifest['files']),output=str(out))))
