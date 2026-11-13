"""Fixed customer-resource catalog and proportional aggregate-group scaling."""
from copy import deepcopy
from pathlib import Path

import numpy as np

from ... import io
from ..customer_flexibility_20261113.model import CustomerCase


ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'outputs/customer_flexibility_sweep_20261113_v1/resource_configs/higher.json'


def resources(entry, scale=1.):
    """Same12 ordinal resource locations; scale2 means two identical groups.

    Only entry.case and entry.labels are read. No topology truth, fitted model,
    forecast or future observation enters placement or capacity selection.

    For a common installed scale2 baseline, either contract scale1 (the other
    group stays at baseline) or contract both groups at scale2. The evaluator
    must add the same installed baseline for both contract comparisons.
    """
    scale=float(scale)
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('A positive finite catalog capacity multiple is required')
    labels=list(entry['labels'])
    if not labels or len(labels)!=len(set(labels)):
        raise ValueError('Unique nonempty observed-terminal labels required')
    n=len(labels)
    c=deepcopy(io.read(SOURCE))
    c.update(id=entry['case']+f'_fixed12_DER_groups{scale:g}',network_case=entry['case'],
        expected_terminal_labels=labels,aggregate_capacity_multiple=scale,
        catalog_source=io.ref(SOURCE),
        description='Fixed12 resource groups at ordinal-mapped terminal locations; proportional identical-group capacity and energy scaling.',
        placement_rule='Map original15-node ordinal locations linearly to N before new outcomes; only case label and terminal labels are used.',
        scaling_rule='Scale PV DC/AC ratings, EV charging power and requested battery energy, thermal power/tank/baseline heat demand/electric buffer together. Prices, efficiency, clocks, temperatures and retention unchanged.')
    for r in c['resources']:
        r['node_label']=labels[round((r['node_label']-101)*(n-1)/14)]
        if r['kind']=='pv':
            for name in ('dc_peak_kwp','inverter_kva'):r[name]*=scale
        elif r['kind']=='ev':
            for name in ('power_kw','battery_energy_required_kwh'):r[name]*=scale
        elif r['kind']=='thermal':
            r['power_kw']*=scale
        else:
            raise ValueError('Unsupported source resource kind: '+r['kind'])
    thermal=c['thermal']
    thermal['water_litres']*=scale
    thermal['baseline_power_kw']={key:value*scale for key,value in thermal['baseline_power_kw'].items()}
    if 'electric_buffer_kwh' in thermal:thermal['electric_buffer_kwh']*=scale
    return CustomerCase(c,labels)
