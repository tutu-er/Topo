"""Thin observed-only extension: RNJ families with an optional common-root block.

Import and prepare do not call calibration, fitting or optimization. The original
driver is read without editing it. Only explicit run invokes its real PoolLP.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path

REPO=Path(os.environ.get('WZZT_REPO_ROOT',r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo'))
BASE_DRIVER_PATH=REPO/'research_experiments'/'wzzt_dr_selection'/'dr_random_observation_gate'/'gate_driver.py'
BASE_DRIVER_SHA256='19194d5342a0b0c90ad2ab1faff000ca14e3459317f44316b29b4b095c6ea7b1'
if hashlib.sha256(BASE_DRIVER_PATH.read_bytes()).hexdigest()!=BASE_DRIVER_SHA256:
    raise RuntimeError('original frozen driver source changed')
_spec=importlib.util.spec_from_file_location('_wzzt_reference_noise_original_driver',BASE_DRIVER_PATH)
base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(base)
_BASE_IDENTIFY=base.identify_snapshot
_BASE_METHOD=base._run_method
METHOD='current_root_augmented_rnj'


def validate_settings(settings):
    if settings.get('max_n')!=3:raise ValueError('reference-noise gate is n3 only')
    if settings.get('coefficient_bound')!=2.:raise ValueError('original nonnegative coefficient domain [0,2] required')
    if settings.get('calibration_snapshot')!='current_observed':raise ValueError('one current-observed calibration required')


def augmented_native_families(current):
    """Preserve native15 order; root is an allowed LP column, never a positive constraint."""
    n=len(current['terminal_labels'])
    if n!=3:raise ValueError('n3 only')
    native=current['native_bank']
    if len(native)!=15:raise ValueError('expected same current native15 RNJ bank')
    root=tuple(range(n));families=[]
    for tree in native:
        family=base.validate_supports([*tree['family'],root],n,family=True)
        families.append(family)
    return tuple(families)


def run_augmented(current,train,valid,settings,budget,*,day_ids=None):
    """Independent real PoolLP through the untouched fixed-family method runner."""
    validate_settings(settings)
    families=augmented_native_families(current)
    pool=base.canonical([*current['pool'],(0,1,2)])
    result=_BASE_METHOD(METHOD,current,train,valid,pool,settings,budget,
        families=families,day_ids=day_ids,current_state=current)
    result.update(native_family_count=len(families),
        unique_augmented_family_count=len(set(families)),
        augmented_native_families=families,native_configs=[tree['config'] for tree in current['native_bank']],
        current_geometry_calibrations=current['original_calibrations'],
        output_calibration='one current-observed raw calibration shared by every method',
        bank_rebuilt_inside_method=False,full_root_support_permitted=True,
        full_root_positive_weight_required=False,coefficient_lower_bound=0.,coefficient_upper_bound=2.,
        zero_leaf_weights_allowed=True,shared_cross_method_LP_cache=False,
        family_source='same current raw native15 RNJ families, each with optional universal support')
    return result


def identify_snapshot(history_packet,future_packet,history_state,discovery,settings,budget,out):
    validate_settings(settings)
    result=_BASE_IDENTIFY(history_packet,future_packet,history_state,discovery,settings,budget,out)
    if result['status']!='complete_record':
        result.setdefault('methods',{})[METHOD]=dict(status='not_started_current_state_unresolved',LP_solves=[])
        result['method_statuses']=dict(Counter(m['status'] for m in result['methods'].values()))
        base.write(out,result);return result
    current=base.read(result['current_state_path'])
    if current['calibration']!=result['applied_calibration']:
        raise RuntimeError('new method must use exactly the saved shared current calibration')
    train,valid,days,_,_=base._observations(history_packet['training']+future_packet['training'],
        history_packet['validation']+future_packet['validation'])
    try:
        budget.check()
        result['methods'][METHOD]=run_augmented(current,train,valid,settings,budget,day_ids=days)
    except Exception as error:
        result['methods'][METHOD]=dict(status='not_started_budget' if isinstance(error,RuntimeError) and
            ('limit' in str(error) or 'budget' in str(error)) else 'unresolved',
            error_type=type(error).__name__,error=str(error),LP_solves=[])
    result.update(method_statuses=dict(Counter(m['status'] for m in result['methods'].values())),
        root_augmented_rnj_method=METHOD,root_augmented_rnj_uses_same_current_bank=True,
        wrapper_source_sha256=base.sha(__file__))
    base.write(out,result);return result


def prepare(root,config_path):
    config=base.read(config_path);validate_settings(config['settings'])
    cfg=base.prepare(root,config_path)
    cfg['methods'].append(METHOD)
    cfg['source_sha256'][str(Path(__file__).resolve())]=base.sha(__file__)
    cfg.update(label='reference observation-noise gate with optional-root RNJ control',
        root_augmented_rnj=dict(method=METHOD,baseline='same current native15 RNJ bank',
            independent_PoolLP=True,root_lower_bound=0.,root_upper_bound=2.,
            validation_fraction=.03,internal_resolution_factor=.16,
            calibration_snapshot='current_observed',max_n=3),
        original_driver_sha256=BASE_DRIVER_SHA256)
    base.write(Path(root)/'protocol.json',cfg)
    base.write(Path(root)/'protocol.sha256.json',dict(sha256=base.sha(Path(root)/'protocol.json')))
    return cfg


def verify(root):
    cfg=base.verify(root);validate_settings(cfg['settings'])
    if cfg['methods'].count(METHOD)!=1:raise RuntimeError('new baseline missing or repeated')
    if cfg['source_sha256'].get(str(Path(__file__).resolve()))!=base.sha(__file__):
        raise RuntimeError('wrapper source not frozen')
    return cfg


def run(root):
    verify(root)
    original=base.identify_snapshot
    try:
        base.identify_snapshot=identify_snapshot
        return base.run(root)
    finally:base.identify_snapshot=original


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--config',type=Path)
    args=parser.parse_args()
    if args.mode=='prepare':
        if args.config is None:parser.error('--config required for prepare')
        result=prepare(args.root,args.config)
    else:result=run(args.root)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','planned_records','processed_records','real_LP_calls','seconds')}),flush=True)


if __name__=='__main__':main()
