"""Two compact scientific figures from completed summary statistics only."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ... import io


def render(summary, out):
    s = io.read(summary); out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ev = s['overall']['0.10/old_gamma2']['nearest_evidence']
    mean = lambda name: ev[name]['mean']
    pieces = [mean('forecast_J_gain'),
        mean('exact_physical_J_gain')-mean('forecast_J_gain'),
        mean('actual_physical_J_gain')-mean('exact_physical_J_gain'),
        mean('actual_parameter_bias_change'), mean('actual_AC_nonlinearity_change'),
        mean('actual_meter_change')]
    actual = mean('actual_logodds_change')
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6), gridspec_kw={'width_ratios':[1.1, 1.25]})
    ax = axes[0]; level = 0.
    names = ['Forecast', 'Forecast\nerror', 'Delivery', 'Parameter\nbias', 'AC\ncorrection', 'Meters', 'Measured']
    for i, value in enumerate(pieces):
        bottom = 0 if i == 0 else level
        ax.bar(i, value, bottom=bottom, color='#315e86' if i == 0 else ('#52a38c' if value>=0 else '#cf7954'), width=.67)
        level += value
        ax.text(i, (level if value>=0 else bottom)+.04, f'{value:+.3f}' if i else f'{value:.3f}',
                ha='center', fontsize=9)
    ax.bar(6, actual, color='#244f6a', width=.67)
    ax.text(6, actual+.04, f'{actual:.3f}', ha='center', fontsize=10)
    ax.set_xticks(range(7), names, fontsize=9)
    ax.set_ylim(0, 2.45); ax.set_ylabel('Mean change in target-versus-runner log odds')
    ax.set_title('A. Forecast-to-execution decomposition', loc='left', fontsize=12)
    ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    ratios = []
    season = {'2024-01-21':'winter','2024-05-09':'spring','2024-07-15':'summer','2024-10-14':'autumn'}
    for key, row in s['per_history'].items():
        if not key.endswith('/old_gamma2'):
            continue
        identity = key.split('/')[1]; case, date = identity.split('__')
        ratios.append((case+' / '+season[date], row['realization']['actual_logodds_change']['ratio'],
                       case=='pengwah18' and date=='2024-10-14'))
    ratios.sort(key=lambda x:x[1]); ax = axes[1]
    y=np.arange(len(ratios))
    ax.barh(y, [r[1] for r in ratios], color=['#cf7954' if r[2] else '#315e86' for r in ratios])
    ax.set_yticks(y,[r[0]+(' *' if r[2] else '') for r in ratios],fontsize=9)
    ax.axvline(1., color='black', ls='--', lw=1)
    ax.set_xlim(0,1.52); ax.set_xlabel('Sum measured gain / sum forecast gain')
    ax.set_title('B. Each fixed background input (32 paths)',loc='left',fontsize=12)
    for i, (_,value,_) in enumerate(ratios):
        ax.text(value+.018,i,f'{value:.1%}',va='center',fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', pad=.4))
    ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Measured-P/Q inputs: original gain-2 policy, fixed historical target and runner',fontsize=13,y=1.01)
    fig.text(.02,-.01,'16 fixed physical days; 4 forecasts x 8 delivery/meter paths each.  * Wrong target; true topology absent from the original pool.',fontsize=9)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out/'evidence_delivery.png',dpi=170,bbox_inches='tight')
    fig.savefig(out/'evidence_delivery.svg',bbox_inches='tight')
    svg = out/'evidence_delivery.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    plt.close(fig)
    io.write(out/'FIGURE_RECEIPT.json',dict(source=io.ref(__file__),summary=io.ref(summary),
        figure=io.ref(out/'evidence_delivery.png'),interpretation='Descriptive conditional-path means; not independent field-day confidence intervals'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();render(a.summary,a.out)
