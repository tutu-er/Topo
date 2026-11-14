"""Independent replay with device-wise Q pairing across unequal resource sets.

All numerical, resource, score and AC-observation checks are inherited from
the sealed reactive verifier. Only cross-placement Q multiplier matching is
changed: match the same (resource ID, measured terminal), not array length.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np

from ... import io
from ..reactive_flex_20261114 import verify as base


class Verify(base.Verify):
    def __init__(self):
        super().__init__()
        self.current_Q_identities=None

    def set_Q_identities(self, resources):
        identities=[(str(r['id']),int(r['node_label'])) for r in resources]
        self.require(len(identities)==len(set(identities)), 'Repeated Q device identity')
        self.current_Q_identities=identities

    def config_check(self, entry, config, labels):
        P_R=super().config_check(entry,config,labels)
        self.set_Q_identities(config['resources'][P_R:])
        return P_R

    def paired(self, key, name, value):
        if name != 'eta_Q':
            return super().paired(key,name,value)
        self.require(self.current_Q_identities is not None, 'Q catalog must be checked before multiplier pairing')
        value=np.asarray(value,float)
        self.require(value.shape==(len(self.current_Q_identities),), 'Q multiplier/catalog length differs')
        self.require(np.isfinite(value).all(), 'Nonfinite Q multiplier')
        for identity,multiplier in zip(self.current_Q_identities,value,strict=True):
            item=(key,name,identity)
            if item in self.shared:
                self.same('cross_placement_eta_Q_same_device',multiplier,self.shared[item],atol=0,rtol=0)
                self.counts['cross_placement_Q_device_multipliers_verified'] += 1
            else:
                self.shared[item]=float(multiplier)
                self.counts['distinct_Q_device_path_multipliers'] += 1
        self.counts['Q_multiplier_arrays_matched_by_identity'] += 1


def run(run_path,destination):
    # base.run creates its verifier through the module's Verify name. Replace
    # that factory only within this call and restore it even if replay fails.
    old_factory=base.Verify
    base.Verify=Verify
    try:
        report=base.run(run_path,destination)
    finally:
        base.Verify=old_factory
    destination=Path(destination).resolve()
    own_source=io.ref(__file__)
    report['sources'].append(own_source)
    report['Q_cross_placement_pairing']=dict(
        identity=['resource_id','node_label'],
        matching='Only shared physical devices are compared; extra devices and unrelated identities are not falsely paired',
        exact_multiplier_equality=True,
        interpretation='Same-device delivery multiplier is shared; delivered kvar need not match because scheduled actions differ',
        original_checks='All original stream regeneration, device execution, fees, native scores and directed evidence checks retained')
    shutil.copyfile(__file__,destination/'wide_verifier.py')
    io.write(destination/'report.json',report)
    io.write(destination/'complete.json',dict(report=io.ref(destination/'report.json'),
        added_verifier_source=own_source))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    report=run(args.run,args.out)
    print(dict(status=report['status'],counts=report['counts'],errors=len(report['errors'])))
    if report['errors']:
        raise SystemExit(1)
