"""Small independent algebra checks; not a grid recovery or AC experiment.

Run with the Topo Python environment. Only prints JSON; does not import or
modify production modules or historical outputs.
"""
import json

import numpy as np


def common_path_response(paths, resistances, observed, candidates):
    return np.array([
        [sum(resistances[e] for e in set(paths[o]) & set(paths[h]))
         for h in candidates]
        for o in observed
    ], dtype=float)


def main():
    # Root 0 -> junction 1 -> metered leaves 2,3; unmetered spur 1 -> 4.
    paths = {1: (0,), 2: (0, 1), 3: (0, 2), 4: (0, 3)}
    resistance = (1.0, 1.0, 2.0, 1.0)
    observed = (2, 3)
    candidates = (1, 4)
    response = common_path_response(paths, resistance, observed, candidates)
    known = common_path_response(paths, resistance, observed, observed)
    augmented = common_path_response(paths, resistance, (2, 3, 4), candidates)

    # Constant hidden load contributes a location-independent difference.
    base_power = np.array([0.4, 0.7])
    probe = np.array([0.15, -0.05])
    amplitude = 0.25
    before = known @ base_power[:, None] + amplitude * response
    after = known @ (base_power + probe)[:, None] + amplitude * response
    delta = after - before
    assert np.allclose(delta[:, 0], delta[:, 1], atol=1e-14)
    assert np.allclose(response[:, 0], response[:, 1], atol=1e-14)
    assert not np.allclose(augmented[:, 0], augmented[:, 1])

    # A common reference bias absorbs the trunk voltage response. Adding a
    # signed total-meter channel separates existence from location.
    voltage_nuisance = np.ones((2, 1))
    voltage_projector = np.eye(2) - voltage_nuisance @ np.linalg.pinv(voltage_nuisance)
    projected_voltage_norm = np.linalg.norm(voltage_projector @ response[:, 0])
    balance_nuisance = np.array([[1.0], [1.0], [0.0]])
    balance_projector = np.eye(3) - balance_nuisance @ np.linalg.pinv(balance_nuisance)
    voltage_and_balance_signal = np.r_[response[:, 0], 1.0]
    projected_joint_norm = np.linalg.norm(balance_projector @ voltage_and_balance_signal)
    assert projected_voltage_norm < 1e-14
    assert abs(projected_joint_norm - 1.0) < 1e-14

    # Parameter flexibility can erase a pairwise separation even if there is
    # no common voltage bias: choose one permissible nuisance direction.
    h1 = np.array([1.0, 2.0, 1.0])
    h2 = np.array([1.0, 1.0, 1.0])
    parameter_nuisance = np.array([[0.0], [1.0], [0.0]])
    parameter_projector = np.eye(3) - parameter_nuisance @ np.linalg.pinv(parameter_nuisance)
    projected_pair = np.linalg.norm(parameter_projector @ (h1 - h2))
    assert np.linalg.norm(h1 - h2) == 1.0
    assert projected_pair < 1e-14

    result = {
        "scope": "independent deterministic linear algebra; no AC, noise calibration, or production recovery validation",
        "tree_edges": [[0, 1], [1, 2], [1, 3], [1, 4]],
        "observed_nodes": list(observed),
        "candidate_nodes": list(candidates),
        "terminal_response": response.tolist(),
        "terminal_location_gap": float(np.linalg.norm(response[:, 0] - response[:, 1])),
        "added_internal_voltage_location_gap": float(np.linalg.norm(augmented[:, 0] - augmented[:, 1])),
        "probe_difference_gap": float(np.linalg.norm(delta[:, 0] - delta[:, 1])),
        "voltage_signal_after_reference_bias_projection": float(projected_voltage_norm),
        "joint_signal_after_reference_bias_projection": float(projected_joint_norm),
        "raw_pair_gap_with_parameter_nuisance": float(np.linalg.norm(h1 - h2)),
        "projected_pair_gap_with_parameter_nuisance": float(projected_pair),
        "assertions_passed": 7,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
