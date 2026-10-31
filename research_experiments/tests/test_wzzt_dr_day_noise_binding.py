"""Fairness regression: future voltage and root noise inherit the source level."""
import ast
from pathlib import Path
import pytest
from research_experiments.wzzt_dr_selection.dr_day_modules.day_physical_worker import voltage_noise_binding


def test_existing_request_retains_original_point_zero_two_percent_default():
    assert voltage_noise_binding({}, {}) == .0002


def test_source_point_zero_five_percent_is_preserved_explicitly():
    assert voltage_noise_binding({'voltage_sigma':.0005}, {'voltage_sigma':.0005}) == .0005


def test_silent_future_noise_reduction_is_rejected():
    with pytest.raises(ValueError, match='differs'):
        voltage_noise_binding({}, {'voltage_sigma':.0005})


def test_outside_existing_grid_is_rejected():
    with pytest.raises(ValueError):
        voltage_noise_binding({'voltage_sigma':.0003}, {})


def test_boolean_is_not_a_noise_level():
    with pytest.raises(ValueError):
        voltage_noise_binding({'voltage_sigma':True}, {})


def test_both_voltage_and_root_observer_use_the_bound_value():
    source=Path(__file__).parents[1]/'wzzt_dr_selection/dr_day_modules/day_physical_worker.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    calls={}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ('_prepare_case_context','_observe_terminal_channels'):
                calls[node.func.attr]=node
    for name,key in (('_prepare_case_context','root_meter_noise_rel'),('_observe_terminal_channels','v_noise_rel')):
        expression=next(k.value for k in calls[name].keywords if k.arg==key)
        assert isinstance(expression,ast.Subscript)
        assert isinstance(expression.value,ast.Name) and expression.value.id=='request'
        assert isinstance(expression.slice,ast.Constant) and expression.slice.value=='voltage_sigma'
