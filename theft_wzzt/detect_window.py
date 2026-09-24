"""Measured-only stationary-window detection with finite LP enumeration.
Use supplied nominal protocol calibration only; source/profile calibration is
simulation-conditional and does not certify physical-node uniqueness.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from theft_wzzt.theft.identified_tree import load_identified
from theft_wzzt.theft.credibility import calibrated_rank
from paper_study.lp_profile import lp_profile
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('measured_entry',ROOT/'detect_simple.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


def detect(measurements,identified,*,calibration=None,alpha=.05):
    if not 0<alpha<1:raise ValueError('Require 0 < alpha < 1')
    tree,data,bounds,_,signature=entry.prepare(measurements,identified,(32,64))
    if calibration is not None:
        if calibration['protocol_signature']!=signature:raise ValueError('Calibration measurement/tree protocol mismatch')
        if Path(__file__).name not in calibration['inference_source_sha256']:raise ValueError('Calibration entry/rule mismatch')
        for name,expected in calibration['inference_source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('Inference code changed; recalibration required')
        if calibration['condition']!='nominal':raise ValueError('This entry supports nominal calibration only')
        null=np.asarray(calibration['null_gains'],float)
        if null.ndim!=1 or not len(null) or not np.isfinite(null).all():raise ValueError('Invalid null calibration')
    fit=lp_profile(tree,data,bounds)
    verdict=dict(alarm=None,status='uncalibrated',rank=None)
    region_set=None;interval=None
    if calibration is not None:
        rank=calibrated_rank(fit['gain'],null)
        enough=1/(len(null)+1)<=alpha
        verdict=dict(alarm=bool(rank<=alpha) if enough else None,status='calibrated' if enough else 'insufficient_calibration',rank=rank,null_n=len(null),alpha=alpha)
        if alpha!=calibration['set_alpha']:raise ValueError('Location and amplitude cutoffs require the calibration alpha')
        minimum=min(r['loss'] for r in fit['profile'])
        region_set=[r['location'] for r in fit['profile'] if r['loss']-minimum<=calibration['location_cutoff']+1e-7]
        radius=calibration['amplitude_radius_pu']
        interval=dict(lower=np.maximum(data.amplitude-radius,0).tolist(),upper=(data.amplitude+radius).tolist())
    return dict(protocol_signature=signature,window=[32,64],decision=verdict,fit=fit,
        calibrated_candidate_regions=region_set,
        reported_regions=region_set if verdict['alarm'] else None,
        amplitude_point_pu=data.amplitude.tolist(),amplitude_interval_pu=interval,
        interpretation='Region and whole-window amplitude coverage are separately marginal under the labelled simulation calibration law, not guarantees for field data or conditional on an alarm. Fixed-window, single stationary source only.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('measurements');parser.add_argument('--tree',default=str(ROOT/'outputs/theft/identified_tree.json'));parser.add_argument('--calibration');parser.add_argument('--output',required=True)
    args=parser.parse_args();target=Path(args.output)
    if target.exists():raise FileExistsError('Refusing to overwrite output')
    calibration=json.loads(Path(args.calibration).read_text()) if args.calibration else None
    result=detect(entry.load_measurements(args.measurements),load_identified(args.tree),calibration=calibration)
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
