"""Reject incomplete resource declarations instead of inferring them from bus IDs."""

import numpy as np
import pandas as pd
import pytest

from rnj_wzzt.data.paper_style_terminal_lv import (
    build_paper_style_resource_assignment,
    build_paper_style_terminal_lv_case,
)
from rnj_wzzt.scenario.profiles import _build_terminal_profiles


@pytest.fixture
def explicit_resources():
    net = build_paper_style_terminal_lv_case()
    return net, build_paper_style_resource_assignment(net)


@pytest.mark.parametrize("field", [
    "customer_type", "pv_capacity_kw", "wind_capacity_kw", "small_generator_capacity_kw",
])
def test_missing_resource_field_is_an_error(explicit_resources, field):
    net, assignment = explicit_resources
    with pytest.raises(ValueError, match=f"missing required columns: {field}"):
        _build_terminal_profiles(net, assignment.drop(columns=field), 8, 42)


@pytest.mark.parametrize("invalid", [None, np.nan, pd.NA, "unknown", ""])
def test_invalid_customer_type_is_an_error(explicit_resources, invalid):
    net, assignment = explicit_resources
    assignment.loc[0, "customer_type"] = invalid
    with pytest.raises(ValueError, match="bus 101: customer_type must be"):
        _build_terminal_profiles(net, assignment, 8, 42)


@pytest.mark.parametrize("field", [
    "pv_capacity_kw", "wind_capacity_kw", "small_generator_capacity_kw",
])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, -0.01, None, "invalid"])
def test_invalid_capacity_is_an_error(explicit_resources, field, invalid):
    net, assignment = explicit_resources
    assignment[field] = assignment[field].astype(object)
    assignment.loc[0, field] = invalid
    with pytest.raises(ValueError, match=f"bus 101: {field} must be finite and nonnegative"):
        _build_terminal_profiles(net, assignment, 8, 42)
