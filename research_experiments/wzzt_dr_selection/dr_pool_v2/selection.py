"""Original 3-percent selection followed by fixed RX75 output contraction."""
from __future__ import annotations
from copy import deepcopy
from .observations import matrices, loss


def contract(point, calibration):
    threshold = .16 * calibration["median_root_depth"]
    keep = [i for i, s in enumerate(point["supports"]) if len(s) == 1 or
            calibration["a"]*point["r"][i] + calibration["b"]*point["x"][i] + 1e-10 >= threshold]
    return dict(point, supports=[point["supports"][i] for i in keep],
                r=[point["r"][i] for i in keep], x=[point["x"][i] for i in keep])


def measure(point, training, validation):
    r, x = matrices(training[0].shape[1], point["supports"], point["r"], point["x"])
    return dict(point, objective=loss(training, r, x)["mae"], validation_mae=loss(validation, r, x)["mae"])


def select(points, calibration):
    if not points:
        raise ValueError("empty final model list")
    best = min(p["validation_mae"] for p in points)
    eligible = [i for i, p in enumerate(points) if p["validation_mae"] <= 1.03*best+1e-12]
    index = min(eligible, key=lambda i:(len(points[i]["supports"]), points[i]["validation_mae"], i))
    chosen = dict(points[index], selection_index=index)
    result = contract(chosen, calibration)
    result["selection_validation_mae"] = chosen["validation_mae"]
    result["selected_uncontracted"] = deepcopy(chosen)
    result["pre_contraction_support_indices"] = deepcopy(chosen["supports"])
    result["pre_contraction_r_values"] = list(chosen["r"])
    result["pre_contraction_x_values"] = list(chosen["x"])
    return result


def package(point, labels, training, validation):
    point = measure(point, training, validation)
    r, x = matrices(len(labels), point["supports"], point["r"], point["x"])
    return dict(point, support_indices=point["supports"],
                support_labels=[[labels[i] for i in s] for s in point["supports"]],
                r_values=point["r"], x_values=point["x"],
                r_matrix=r.tolist(), x_matrix=x.tolist(), R=r.tolist(), X=x.tolist(),
                train_mae=point["objective"], validation_mae=point["validation_mae"],
                selected_origin=point.get("origin"))
