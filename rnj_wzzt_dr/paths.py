"""Resolve retained assets after cleanup without rewriting sealed JSON bytes."""
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = Path('D:/0-github_workspace/Topo')
DATASETS = frozenset({
    'astra_inventory_20261114', 'distribution_panel_confirmation_data_20261115_v1',
    'equipment_flex_execution_20261115_v1', 'flexibility_grid_execution_20261115_v1',
    'history10_rx_20261116_v1', 'history288_pq_noise_data_20261115_v1',
    'history288_pq_noise_execution_20261115_v1', 'impedance_floor_20261116_v1',
    'measured_pq_sources_20261114', 'wzzt_long_grid_data_20261117_v1',
})


def resolve_path(path):
    """Use a retained replacement only when it exists inside this checkout."""
    original = Path(path).expanduser().resolve()
    for base in (ROOT, LEGACY_ROOT):
        try:
            relative = original.relative_to(base)
        except ValueError:
            continue
        parts = relative.parts
        target = None
        if len(parts) >= 2 and parts[0] == 'outputs' and parts[1] in DATASETS:
            target = ROOT/'data'/Path(*parts[1:])
        elif len(parts) >= 2 and parts[0] == 'topology_identification_standalone':
            if parts[1] == 'cases':
                target = ROOT/'rnj_wzzt_dr'/'physical'/Path(*parts[1:])
            elif parts[1] == 'outputs':
                target = ROOT/'data'/'benchmarks'/Path(*parts[2:])
            else:
                target = ROOT/'data'/'legacy_sources'/relative
        if target is not None and target.exists():
            return target.resolve()
    return original


def relocate_references(value):
    """Rebind paths in the loaded view; stored bytes and SHA-256 stay intact."""
    if isinstance(value, dict):
        return {key: relocate_references(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate_references(item) for item in value]
    if isinstance(value, str) and re.match(
            r'^(?:[A-Za-z]:[\\/]|(?:outputs|topology_identification_standalone)[\\/])', value):
        resolved = resolve_path(value)
        if resolved != Path(value).resolve():
            return str(resolved)
    return value
