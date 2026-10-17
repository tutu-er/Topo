import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('calibration_then_evaluation',Path(__file__).with_name('calibration_then_evaluation.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_launch_requires_complete_calibration_and_released_lock():
    state=dict(status='calibration_complete',phase='calibration')
    assert module.decision(state,True,True)=='wait'
    assert module.decision(state,False,True)=='launch'


def test_stopped_incomplete_controller_is_not_restarted():
    assert module.decision(dict(status='running',phase='calibration'),False,False)=='calibration_controller_stopped'


def test_existing_evaluation_is_not_duplicated():
    assert module.decision(dict(status='running',phase='evaluation'),True,True)=='evaluation_already_started'
    assert module.decision(dict(status='statistical_complete',phase='evaluation'),False,False)=='done'
