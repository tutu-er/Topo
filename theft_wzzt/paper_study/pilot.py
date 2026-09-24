"""Development-only pilot. Checkpoints retain successful and failed variants."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from theft_wzzt.theft.theft_simulation import TheftSpec, simulate_theft_scenarios
from theft_wzzt.theft.identified_tree import load_identified, to_theft_tree, region_label
from paper_study.profile import fixed_profile, stable_profile
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/theft_paper'
spec = importlib.util.spec_from_file_location('simple_entry', ROOT/'detect_simple.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)

def main():
    identified = load_identified(ROOT/'outputs/theft/identified_tree.json')
    weights = to_theft_tree(identified)[1]
    for name, bus in [('null', None), ('internal', 2), ('terminal', 108)]:
        amplitude = np.zeros(96); amplitude[45:51] = 8.
        thefts = None if bus is None else [TheftSpec(bus, tuple(amplitude))]
        net, scenario = simulate_theft_scenarios('paper15', thefts, replicate=200)
        measurements = {k: scenario[k] for k in entry.MATRIX_CHANNELS+entry.SERIES_CHANNELS+('scenario_settings',)}
        truth = None if bus is None else region_label(bus, net, identified)
        for window in [(44,52), (32,64)]:
            tree, data, bounds, _, _ = entry.prepare(measurements, identified, window)
            for method in ['fixed_free', 'fixed_stable', 'stable_positive', 'stable_optional']:
                path = OUT/f'pilot_{name}_{window[1]-window[0]}_{method}.json'
                if path.exists():
                    continue
                print(f'START {name} {window} {method}', flush=True)
                try:
                    if method.startswith('fixed'):
                        result = fixed_profile(tree, data, weights, stable=method=='fixed_stable')
                    else:
                        result = stable_profile(tree, data, bounds, optional_activity=method=='stable_optional')
                    result.update(experiment=name, replicate=200, window=window, truth_region=truth,
                                  exact_region_match=None if truth is None else result['best_location']==truth)
                    result['truth_profile_gap'] = None if truth is None else next(row['loss'] for row in result['profile'] if row['location']==truth)-result['h1_loss']
                    print(json.dumps({k: result[k] for k in ['method','gain','best_location','truth_region','truth_profile_gap','seconds']}), flush=True)
                except Exception as error:
                    result = dict(experiment=name, window=window, method=method, error=repr(error))
                    print('FAILED '+repr(error), flush=True)
                path.write_text(json.dumps(result, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
