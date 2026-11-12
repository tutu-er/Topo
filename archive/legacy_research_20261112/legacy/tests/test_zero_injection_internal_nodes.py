from terminal_case33.data.case33bw_raw import load_raw_case33bw
from terminal_case33.scenario.terminalize import terminalize_case33


def test_hidden_internal_zero_injection_and_unobserved():
    net = terminalize_case33(load_raw_case33bw(), seed=2)
    hidden = net.buses[net.buses["bus_type"].eq("hidden_internal")]
    assert not hidden.empty
    assert (hidden["pd_kw"] == 0.0).all()
    assert (hidden["qd_kvar"] == 0.0).all()
    assert (~hidden["is_observed"]).all()

