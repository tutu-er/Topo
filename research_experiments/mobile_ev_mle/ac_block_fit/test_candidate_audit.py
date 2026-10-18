"""Input-boundary, candidate-denominator and local derivative audit tests."""
from copy import deepcopy
import gzip
import json
import numpy as np

from . import candidate_audit as audit
from . import benchmark as b
from .test_benchmark import source_fixture
from .test_fixed_source import fixture


def test_raw_projection_does_not_read_fit_summary_or_future(tmp_path, monkeypatch):
    source, raw = source_fixture(tmp_path)
    raw["config"] = dict(base_kw=100.)
    path = source/"history_000_paper11_none.json.gz"
    path.write_bytes(gzip.compress(b._bytes(raw)))
    original = json.JSONDecoder.raw_decode; decoded = []
    def spy(self, text, idx=0):
        decoded.append(text[idx:idx+100])
        return original(self, text, idx)
    monkeypatch.setattr(json.JSONDecoder, "raw_decode", spy)
    projected = audit.project_raw(path)
    assert "planning" not in projected and "evaluation" not in projected
    assert not any(value.startswith('{"history":') for value in decoded)
    assert not any(value.startswith("1234567") for value in decoded)


def test_first_candidate_order_includes_failed_history_without_reading_it(tmp_path):
    source, raw = source_fixture(tmp_path)
    raw["config"] = dict(base_kw=100.)
    first = raw["candidate_structures"][0]
    second = dict(first, name="second_declared")
    raw["candidate_structures"] = [first, second]
    raw["planning"]["history"]["candidates"][first["name"]]["success"] = False
    raw["planning"]["history"]["candidates"][first["name"]]["loss"] = None
    path = source/"history_000_paper11_none.json.gz"
    path.write_bytes(gzip.compress(b._bytes(raw)))
    output = tmp_path/"whitelist.json.gz"
    package = audit.prepare_inputs(source, output, networks=("paper11",), count=2)
    assert [row["candidate"]["name"] for row in package["items"]["paper11"]["selected_candidates"]] == \
           [first["name"], second["name"]]
    assert not package["items"]["paper11"]["historical_fit_summary_decoded"]
    assert len(audit.tasks(package, prefixes=(2,))) == 4
    assert b.load_history_package(output)["payload_sha256"] == package["payload_sha256"]


def test_prepared_prefix_excludes_future_and_other_phases(tmp_path):
    prepared = dict(profile_names=["p0"], assignments=dict(case33={"2": "p0"}),
        windows=dict(development=[dict(history_rows=2, p_relative=[[1.], [2.], [1234567.]],
            q_relative=[[.2], [.4], [987654.]], timestamps=["h0", "h1", "future"])],
            evaluation=[dict(secret=[999999.])]))
    path = tmp_path/"prepared.json.gz"; path.write_bytes(gzip.compress(b._bytes(prepared)))
    selected = audit.prepared_history(path, history=2)
    assert selected["p_relative"] == [[1.], [2.]]
    assert selected["timestamps"] == ["h0", "h1"]
    assert not selected["future_values_decoded"]


def test_response_derivative_and_exact_local_distance_gradient():
    from .. import likelihood as lk
    c, data, meter, _, state = fixture(periods=5)
    first = dict(local_response=audit.local_response(c, data, meter, state, base_kw=100.))
    perturbed = dict(state, r=state["r"]*1.1, x=state["x"]*.9)
    second = dict(local_response=audit.local_response(c, data, meter, perturbed, base_kw=100.))
    metric = audit.pair_metric(first, second)
    baseline = metric["distance"]
    assert baseline > 0
    ids = first["local_response"]["working_periods"]
    site, time = 1, 0
    h = 1e-3
    def distance(power):
        inputs = data.p[ids].copy(); inputs[time, site] += power/100.
        means = []
        for parameter in (state, perturbed):
            phy = lk.ac_predict(c, parameter["r"], parameter["x"], inputs, data.q[ids],
                parameter["source_p"][ids], parameter["source_q"][ids], data.root_v[ids])
            means.append(np.column_stack((phy["v"], phy["master_p"], phy["master_q"])))
        delta = (means[0]-means[1])/np.asarray(first["local_response"]["sigma"])
        return .5*np.sum(delta**2)
    numeric = (distance(h)-distance(-h))/(2*h)
    expected = metric["distance_gradient_EV_P_per_kw"][time, site]
    assert abs(numeric-expected) < 1e-5*max(1., abs(expected))
    assert np.asarray(first["local_response"]["voltage_PQ_derivative"]).shape == (1, 3, 6)
    assert metric["production_coefficients_function_used"]
    assert metric["G_SQ_value"] > 0
    power = np.asarray(metric["audit_power_kw"])
    sq = metric["G_SQ_task"]
    expected = 2*np.asarray(sq["quadratic"])*power
    assert np.array_equal(metric["G_SQ_nonzero_power_gradient"], expected)
    assert np.max(abs(metric["G_SQ_nonzero_power_gradient"])) > 0


def test_candidate_compare_retains_RX_disagreement_and_solver_failure():
    common = dict(status="returned", solver_success=False,
        audit=dict(available=True, loss=10., physical_voltage_feasible=True, line_domain_feasible=True,
            source_and_optimizer_box_feasible=True, reference_scaled_projected_gradient=1e-7),
        attained_state=dict(r=[.01], x=[.005], source_p=[0.], source_q=[0.]))
    changed = deepcopy(common); changed["attained_state"]["r"] = [.02]
    result = audit.compare_candidate(common, changed)
    assert result["loss_agreement"]
    assert not result["rx_agreement"]
    assert not result["reference_solver_success"]
    assert result["reference_numerically_usable"]
    assert not result["parameter_difference_is_true_RX_error"]
