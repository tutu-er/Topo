"""Standalone scientific figures from the audited grid summary only."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from ... import io


def run(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.font_manager import FontProperties
    source=io.read(report)
    if (source.get('smoke') or not source.get('inputs_unchanged') or len(source['cells'])!=36
            or source.get('forecasts')!=1 or source.get('repeats')!=16):
        raise ValueError('A complete, audited 36-cell report is required')
    out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
    else:
        plt.rcParams['font.family']='DejaVu Sans'
    plt.rcParams['axes.unicode_minus']=False
    modes=('normal_fixed','mgf_fixed','mgf_forecast')
    labels=('正态约束','固定功率指数约束','含预测误差指数约束')
    strides=(1,2,4);multipliers=(.5,1.,2.)
    by={(r['stride'],r['meter_multiplier']):r for r in source['by_stride_meter']}
    def contrast(s,m,mode,kind):
        cell=by[s,m]
        if kind==0:return 100*cell['paired_vs_no_dfs'][mode]['pooled_path_difference']
        a,b=cell['groups'][mode],cell['groups']['no_dfs']
        return 100*(a['native_correct_and_true_high']-b['native_correct_and_true_high'])/a['n']
    panels=[[np.array([[contrast(s,m,mode,kind) for m in multipliers] for s in strides])
             for mode in modes] for kind in (0,1)]
    fig,axes=plt.subplots(2,3,figsize=(13.2,8.4),layout='constrained')
    for kind in (0,1):
        limit=max(1.,max(float(np.max(np.abs(x))) for x in panels[kind]))
        for ax,mode,title,values in zip(axes[kind],modes,labels,panels[kind]):
            im=ax.imshow(values,cmap='RdBu',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
            ax.set_title(title,fontsize=12)
            ax.set_xticks(range(3),['0.25% / 0.025%','0.5% / 0.05%','1% / 0.1%'],fontsize=8)
            ax.set_yticks(range(3),['192 / 96','96 / 48','48 / 24'])
            ax.set_xlabel('P/Q / 电压量测标准差',fontsize=10)
            ax.set_ylabel('训练点 / 验证点',fontsize=10)
            for i,s in enumerate(strides):
                for j,m in enumerate(multipliers):
                    cell=by[s,m];pair=cell['paired_vs_no_dfs'][mode]
                    if kind==0:
                        bill=cell['groups'][mode]['settled_cost_mean_completed']['total_yuan']
                        price='费用缺失' if bill is None else f'{bill:.2f}元'
                        text=f"+{pair['gained']} / −{pair['lost']}\n{price}"
                    else:
                        a,b=cell['groups'][mode],cell['groups']['no_dfs']
                        text=f"{b['native_correct_and_true_high']} → {a['native_correct_and_true_high']}\n共{a['n']}条"
                    ax.text(j,i,text,ha='center',va='center',fontsize=10,
                            color='white' if abs(values[i,j])>.65*limit else 'black')
        colorbar=fig.colorbar(im,ax=axes[kind],fraction=.025,pad=.025)
        colorbar.set_label(('内部恢复率变化（百分点）' if kind==0 else '正确且权重≥0.9的比例变化（百分点）'),fontsize=9)
    fig.suptitle('采购效果：每格4个固定网络日 × 16条配对路径',fontsize=13)
    fig.text(.5,.005,'上：新增 / 丢失恢复、平均结算费。下：正确且权重≥0.9的路径数。预测误差固定5%。',ha='center',fontsize=9)
    fig.savefig(out/'grid_effects.png',dpi=180,bbox_inches='tight')
    fig.savefig(out/'grid_effects.pdf',bbox_inches='tight')
    plt.close(fig)
    io.write(out/'receipt.json',dict(report=io.ref(report),source=io.ref(__file__),
        png=io.ref(out/'grid_effects.png'),pdf=io.ref(out/'grid_effects.pdf'),
        color_units='Top: recovery percentage-point change. Bottom: change in correct AND weight>=0.9 frequency. All four fixed worlds and failures retained.',
        counts='Paired gained/lost recoveries;16 paths per world per condition,64 per displayed cell',
        costs='Mean capacity plus delivered activation bill; not optimization lower bounds',
        scope='Descriptive sensitivity on4 known worlds; not independent date confirmation',
        environment=dict(matplotlib=matplotlib.__version__,numpy=np.__version__)))
    return out/'grid_effects.png'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(run(args.report,args.out))
