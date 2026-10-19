"""Read saved evaluation only; figures do not feed back to the controller."""
from __future__ import annotations
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':13,'axes.titlesize':13,'axes.labelsize':13,'xtick.labelsize':11.5,'ytick.labelsize':11.5})

LABELS={'constant':'Constant','charge_fixed':'Fixed charge','v2g_fixed':'Fixed V2G',
        'charge_mobile':'Mobile charge','v2g_mobile':'Mobile V2G'}

def create_figures(trial_file,output):
    trial_file,output=Path(trial_file),Path(output);output.mkdir(parents=True,exist_ok=True)
    raw=trial_file.read_bytes();r=json.loads(gzip.decompress(raw));rows={x['method']:x for x in r['rows']}
    methods=list(LABELS);time=np.arange(r['config']['horizon']+1)*r['config']['dt_hours']*60
    cmap=['#2166ac','#d6604d','#439660']
    fig,axes=plt.subplots(2,2,figsize=(9.2,6.),sharex=True,sharey=True,layout='constrained')
    for ax,method in zip(axes.ravel(),methods[1:]):
        row=rows[method];power=np.asarray(row['power_kw']);sel=r['selection']['methods'][method]
        op=(sel.get('operation') if (sel.get('guard') or {}).get('accepted') else r['selection'].get('constant_operation')) or {}
        for j,color in enumerate(cmap):
            ax.step(time,np.r_[power[:,j],power[-1,j]],where='post',color=color,label=f'Site {j}',lw=1.8)
        ax.axhline(2.,color='#555555',ls='--',lw=1.,label='Constant at site 0')
        for arc in op.get('route',[]):
            if arc['kind'] in ('travel','setup'):
                ax.axvspan(time[arc['departure']],time[arc['arrival']],color='#777777',alpha=.15)
        ax.set_title(LABELS[method]);ax.set_ylim(-6.7,6.7);ax.set_yticks([-6,-3,0,3,6]);ax.grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('Elapsed time (min)')
    for ax in axes[:,0]:ax.set_ylabel('Grid active power (kW)')
    axes[0,0].legend(loc='lower left',fontsize=11,ncol=2)
    fig.suptitle(f'Frozen policy actions, history {r["seed"]}; same terminal battery service',fontsize=11)
    fig.savefig(output/f'ev_controls_{r["seed"]}.pdf');fig.savefig(output/f'ev_controls_{r["seed"]}.png',dpi=160);plt.close(fig)
    true=rows['constant'].get('true_name');wrong=[x['name'] for x in r['historical_fits'] if x['name']!=true]
    voltage=[];joint=[];details=[]
    for method in methods:
        row=rows[method];audit=row['noiseless_AC_refit'];by={x['candidate']:x for x in audit['candidate_distances']}
        voltage.append([np.sqrt(by[n]['future_channel_SSE']['v']/(r['config']['horizon']*3)) for n in wrong])
        losses={x['name']:x['loss'] for x in audit['fits']}
        joint.append(min(losses[n] for n in wrong)-losses[true])
        details.append(dict(method=method,wrong_candidate_order=wrong,future_voltage_whitened_RMS=voltage[-1],
            noiseless_joint_signed_margin=joint[-1],grid_net_kwh=row['grid_net_kwh'],grid_throughput_kwh=row['grid_throughput_kwh']))
    fig,axes=plt.subplots(1,2,figsize=(9.,4.5),layout='constrained')
    x=np.arange(len(methods));width=.24
    for j,n in enumerate(wrong):axes[0].bar(x+(j-1)*width,np.asarray(voltage)[:,j],width,label=n)
    axes[0].set_ylabel('Future voltage error (whitened RMS)');axes[0].legend(fontsize=11)
    axes[1].bar(x,joint,color=['#777777','#2166ac','#67a9cf','#b2182b','#ef8a62'])
    axes[1].set_ylabel('Noiseless joint SSE margin');axes[1].axhline(0,color='black',lw=.8)
    for ax in axes:
        ax.set_xticks(x,[LABELS[m].replace(' ','\n') for m in methods],fontsize=11);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Complete AC refits to H + true noiseless future; all wrong candidates retained',fontsize=12)
    fig.savefig(output/f'ev_ac_separation_{r["seed"]}.pdf');fig.savefig(output/f'ev_ac_separation_{r["seed"]}.png',dpi=160);plt.close(fig)
    record=dict(trial=str(trial_file),trial_sha256=hashlib.sha256(raw).hexdigest(),seed=r['seed'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy_version=np.__version__,matplotlib_version=matplotlib.__version__,
        example_rule='first predeclared evaluation cherry12 history 988404; no case selection by outcome',
        truth_used_for_offline_figures_only=True,rows=details)
    (output/'figure_values.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return record


def create_location_figures(ablation_directory,output):
    """All 12 histories, including same-curve caches and adverse locations."""
    directory,output=Path(ablation_directory),Path(output);output.mkdir(parents=True,exist_ok=True)
    summary_path=directory/'summary.json';raw=summary_path.read_bytes();summary=json.loads(raw)
    freeze_path=directory/'freeze.json';freeze=json.loads(freeze_path.read_bytes());seeds=freeze['seeds']
    if len(seeds)!=12 or len(set(seeds))!=12:raise ValueError('all 12 predeclared histories required')
    methods=('charge_mobile','v2g_mobile');rows=summary['rows'];data=[]
    if len(rows)!=24:raise ValueError('all 24 method-history conditions required')
    for method in methods:
        by={r['seed']:r for r in rows if r['method']==method}
        if len(by)!=12 or set(by)!=set(seeds):raise ValueError('location figure lost a history')
        data.append([by[s] for s in seeds])
    fig,axes=plt.subplots(2,1,figsize=(10.,6.5),sharex=True,sharey=True,layout='constrained')
    x=np.arange(len(seeds));blue='#2166ac';orange='#d95f02'
    values=[r[key] for group in data for r in group for key in ('delta_M','delta_noiseless_M') if r[key] is not None]
    low=min(values+[0.]);high=max(values+[0.]);pad=.10*max(1.,high-low)
    for ax,method,group in zip(axes,methods,data):
        actual=np.asarray([np.nan if r['delta_M'] is None else r['delta_M'] for r in group])
        mean=np.asarray([np.nan if r['delta_noiseless_M'] is None else r['delta_noiseless_M'] for r in group])
        cached=np.asarray([r.get('remap_noisy_cache_hit') is True for r in group])
        ax.axhline(0,color='#444444',lw=1.)
        ax.vlines(x-.10,0,actual,color=blue,alpha=.28,lw=1.2)
        ax.vlines(x+.10,0,mean,color=orange,alpha=.28,lw=1.2)
        ax.scatter(x-.10,actual,s=40,color=blue,marker='o',label='Paired observed future',zorder=3)
        ax.scatter(x+.10,mean,s=44,color=orange,marker='^',label='True noiseless future',zorder=3)
        if np.any(cached):ax.scatter(x[cached],np.zeros(cached.sum()),s=78,facecolors='none',edgecolors='black',
            marker='D',linewidths=1.1,label='Identical curve, cached',zorder=4)
        failed=~np.isfinite(actual)|~np.isfinite(mean)
        if np.any(failed):ax.scatter(x[failed],np.full(failed.sum(),low-pad/2),marker='x',color='black',s=42,label='Unavailable condition')
        ax.set_title(LABELS[method],loc='left',fontsize=11)
        ax.set_ylim(low-pad,high+pad);ax.set_ylabel(r'Joint SSE margin gain $\Delta M$')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='lower right',ncol=3,fontsize=11)
    family={'star':'star','cherry_01':'01','cherry_02':'02','cherry_12':'12'}
    labels=[str(r['seed'])+'\n'+family[r['family']] for r in data[0]]
    axes[-1].set_xticks(x,labels,fontsize=11.5)
    axes[-1].set_xlabel('All predeclared histories / true family shown only for offline reading')
    fig.suptitle('Electrical location ablation: original placement minus site-0 remap\n'
        'Same EV-terminal P(t), energy and noise; battery service differs by removed driving loss',fontsize=11)
    files=[]
    for suffix in ('pdf','png'):
        path=output/f'ev_location_ablation_delta_M.{suffix}'
        fig.savefig(path,**({'dpi':180} if suffix=='png' else {}));files.append(path)
    plt.close(fig)
    histories=[directory/f'history_{seed}.json.gz' for seed in seeds]
    record=dict(summary=str(summary_path.resolve()),summary_sha256=hashlib.sha256(raw).hexdigest(),
        freeze_sha256=hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
        parameter_sha256=freeze['parameter_sha256'],frozen_scientific_source_sha256=freeze['source_sha256'],
        plotting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy_version=np.__version__,matplotlib_version=matplotlib.__version__,
        all_histories_included=seeds,planned_conditions=24,displayed_conditions=24,
        statistical_unit='one history; four fixed family strata, three histories each',
        selection_rule='all predeclared histories, both methods; zero caches and adverse conditions retained',
        rows=[dict(seed=r['seed'],family=r['family'],method=r['method'],delta_M=r['delta_M'],
            delta_noiseless_M=r['delta_noiseless_M'],cache_hit=r.get('remap_noisy_cache_hit'),status=r['status'])
            for group in data for r in group],
        history_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in histories},
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (output/'location_figure_values.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--trial',type=Path);group.add_argument('--location-ablation',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=create_figures(a.trial,a.output) if a.trial else create_location_figures(a.location_ablation,a.output)
    print(json.dumps(dict(output=str(a.output),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        displayed_conditions=result.get('displayed_conditions'),seed=result.get('seed')),ensure_ascii=False))
