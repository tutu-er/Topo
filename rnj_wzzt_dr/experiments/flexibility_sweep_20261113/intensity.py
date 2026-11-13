"""Pre-confirmation procurement-intensity supplement, reusing frozen fits.

Development paths were already opened by the parent study. The supplement
uses the SAME confirmation paths as the parent: these are paired intensity
comparisons, never an additional independent-day sample.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from ... import io
from .config import ROOT
from .simulation import runtime
from .study import verify_sources, utc, require


def prepare(base, out):
    require(not out.exists(), 'Use a fresh supplement directory')
    parent = verify_sources(base)
    require(not (base/'confirmation').exists(), 'Intensity choices must precede confirmation observations')
    selection = io.read(base/'development_selection.json')
    selection = [e for e in selection if e['qualification'] in ('weak', 'secondary')]
    require(selection, 'No qualified weak historical candidate exists')
    out.mkdir(parents=True)
    for name in ('requests.json', 'offline_truth.json'):
        shutil.copyfile(base/name, out/name)
    for p in base.glob('history_jobs_s*.json'):
        shutil.copyfile(p, out/p.name)
    for name in ('current', 'higher'):
        target = out/'resource_configs'/f'{name}.json'
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(base/'resource_configs'/f'{name}.json', target)
    protocol = dict(parent)
    protocol.update(created_utc=utc(), kind='preconfirmation_intensity_supplement',
        parent_protocol=io.ref(base/'protocol.json'), source_study=str(base), gamma=[2., 4.],
        resource_config_refs={name: io.ref(out/'resource_configs'/f'{name}.json') for name in ('current', 'higher')},
        development_rule='Reuse the already-opened two development days for each qualified weak parent history; not independent new data. No extra fitting or history selection.',
        confirmation_rule='Use exactly the parent sealed confirmation IDs and 32 paired paths; no supplement-dependent ID or threshold tuning after confirmation.',
        rationale='The first weak development case had nonzero gain at log(1.5), with remaining bounded resource headroom; compare larger prespecified evidence purchases2 and4.',
        new_identifier_fits=0, new_parameter_fits=0,
        inference='Paired intensity supplement on the same parent confirmation days. Do not add its day counts to the parent study.')
    rel = Path(__file__).relative_to(ROOT).as_posix()
    protocol['sources'] = dict(parent['sources'], **{rel: io.ref(__file__)['sha256']})
    io.write(out/'protocol.json', protocol)
    io.write(out/'lock.json', dict(created_utc=utc(), protocol=io.ref(out/'protocol.json')))
    io.write(out/'development_selection.json', selection)
    runtime(out)  # Copy and verify the physical runtime before parallel workers.
    return protocol


def seal(base, out):
    verify_sources(out)
    require(not (base/'confirmation').exists() and not (out/'confirmation').exists(), 'Confirmation already opened')
    target = out/'confirmation_selection.json'
    require(not target.exists(), 'Supplement confirmation already sealed')
    original = io.read(base/'confirmation_selection.json')
    original.update(source_protocol=io.ref(out/'protocol.json'),
        parent_confirmation_selection=io.ref(base/'confirmation_selection.json'),
        same_paired_paths_as_parent=True, created_utc=utc())
    io.write(target, original)
    return original


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'seal'])
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = (prepare if args.action == 'prepare' else seal)(args.base.resolve(), args.out.resolve())
    print('done', args.action)
