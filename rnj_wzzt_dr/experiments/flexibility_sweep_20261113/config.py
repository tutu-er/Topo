"""All experimental choices, fixed before any new fitting or AC evaluation."""
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np

from ... import io


ROOT = Path(__file__).resolve().parents[3]
STUDY_SEED = 2026111329
WORKERS = 4
GAMMAS = [float(np.log(1.5)), float(np.log(1.1))]
FIT_BUDGET = dict(total_seconds=90., per_mip_seconds=30., qp_seconds=10., output_reserve_seconds=10.)


def resource_configs():
    current = io.read(ROOT/'rnj_wzzt_dr/experiments/customer_flexibility_20261113/case_augmented.json')
    higher = deepcopy(current)
    higher['id'] = 'paper15_higher_PV_EV_v1'
    higher['description'] = 'Higher pre-existing customer DER adoption: add PV104 and EV107 to the current ten-resource catalog.'
    higher['placement_rule'] = 'Fixed before this sweep from the preceding development audit; no new-grid truth, fitted parameters or new-day outcomes used.'
    pv = deepcopy(current['resources'][0])
    pv.update(id='pv104', node_label=104, fee_positive_yuan_per_kwh=.30)
    ev = deepcopy(next(r for r in current['resources'] if r['id'] == 'ev104'))
    ev.update(id='ev107', node_label=107)
    higher['resources'].extend([pv, ev])
    return dict(current=current, higher=higher)


def grid(seed_index):
    rows = []
    for adoption, n, sigma, profile in product(['higher', 'current'], [48, 96, 192],
                                              [.0005, .001, .002], ['baseline', 'synchronous', 'evening_peak']):
        if adoption == 'current' and n != 96:
            continue
        rows.append(dict(adoption=adoption, history_points=n, voltage_sigma=sigma,
                         profile=profile, history_seed=seed_index))
    pilot = [('higher', 96, .0005, 'baseline'), ('higher', 48, .001, 'synchronous'),
             ('higher', 192, .002, 'evening_peak'), ('current', 96, .001, 'baseline')]
    rows.sort(key=lambda x: (pilot.index((x['adoption'], x['history_points'], x['voltage_sigma'], x['profile']))
                            if (x['adoption'], x['history_points'], x['voltage_sigma'], x['profile']) in pilot else 100,
                            x['adoption'], x['history_points'], x['voltage_sigma'], x['profile']))
    for i, row in enumerate(rows):
        row.update(id=f's{seed_index}_c{i:03d}', index=36*seed_index+i)
    return rows
