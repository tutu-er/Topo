import importlib.util
from pathlib import Path
import numpy as np
from theft_wzzt.theft.identified_tree import load_identified
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('window_entry',ROOT/'detect_window.py')
window=importlib.util.module_from_spec(spec);spec.loader.exec_module(window)

def test_measured_only_entry_is_invariant_to_poisoned_truth():
    measured=window.entry.load_measurements(ROOT/'outputs/theft_simplification/measurements_exp2.json')
    tree=load_identified(ROOT/'outputs/theft/identified_tree.json')
    first=window.detect(measured,tree)
    measured.update(loss_p_true=object(),loss_p_notheft=object(),loss_q_true=object(),theft_truth=object())
    second=window.detect(measured,tree)
    assert first['fit']['gain']==second['fit']['gain']
    assert first['decision']['alarm'] is None
    assert first['reported_regions'] is None
    assert first['amplitude_interval_pu'] is None

def test_invalid_calibration_rejected_before_inference():
    import pytest
    measured=window.entry.load_measurements(ROOT/'outputs/theft_simplification/measurements_exp2.json')
    tree=load_identified(ROOT/'outputs/theft/identified_tree.json')
    with pytest.raises(ValueError,match='protocol mismatch'):
        window.detect(measured,tree,calibration={'protocol_signature':'wrong'})
