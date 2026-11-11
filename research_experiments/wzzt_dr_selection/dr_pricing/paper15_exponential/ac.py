"""Portable, freeze-gated adapter to the original paper15 AC execution.

Importing this module reads no network, natural load, price or topology truth.
The vendored equations, meter observer and resource rules are exact source
copies. This file only loads relative inputs and converts their containers.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor_ac"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _checked(reference, relative_to):
    if not isinstance(reference, dict) or not {"path", "sha256"} <= reference.keys():
        raise ValueError("An explicit path and SHA-256 reference is required")
    path = Path(reference["path"])
    if not path.is_absolute():
        path = Path(relative_to) / path
    path = path.resolve()
    if _sha(path) != reference["sha256"]:
        raise ValueError("Changed frozen file: " + str(path))
    return path


def _freeze(path):
    path = Path(path).resolve()
    record = _read(path)
    if record.get("status") != "frozen":
        raise ValueError("Physical execution requires an already frozen quote")
    reference = record.get("quote", record.get("contracts"))
    quote = _checked(reference, path.parent)
    return dict(path=path, sha256=_sha(path), quote=quote,
                quote_sha256=_sha(quote), record=record)


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Unavailable vendored module: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _vendor():
    manifest = _read(VENDOR / "vendor_manifest.json")
    for filename, row in manifest["files"].items():
        path = (VENDOR / filename).resolve()
        if VENDOR.resolve() not in path.parents or _sha(path) != row["vendored_sha256"]:
            raise ValueError("Changed original AC/device source: " + filename)
    sys.path.insert(0, str(VENDOR))
    sim = importlib.import_module("rnj_wzzt.scenario.simulation")
    # A loaded copy is accepted only if its full source is byte-identical.
    for filename in manifest["files"]:
        if not filename.startswith("rnj_wzzt/") or filename.endswith("/__init__.py"):
            continue
        module_name = filename[:-3].replace("/", ".")
        module = importlib.import_module(module_name)
        if _sha(module.__file__) != manifest["files"][filename]["vendored_sha256"]:
            raise RuntimeError("Use a clean worker: an incompatible rnj_wzzt is already loaded")
    helpers = _load_file("_paper15_original_execution", VENDOR / "execution.py")
    devices = _load_file("_paper15_original_devices", VENDOR / "devices.py")
    profiles = _load_file("_paper15_original_resource_profiles", VENDOR / "resource_profiles.py")
    return sim, helpers, devices, profiles, manifest


def _array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float64).tobytes()).hexdigest()


def _background_hashes(background):
    return {key: _array_hash(getattr(background, key))
            for key in ("p_base", "q_base", "root_true")}


def _physical_index(arrays, metadata):
    if "time_index_ns" in arrays:
        return pd.DatetimeIndex(pd.to_datetime(arrays["time_index_ns"]))
    if metadata.get("index_kind") == "range":
        start, step = int(metadata.get("index_start", 0)), int(metadata.get("index_step", 1))
        return pd.RangeIndex(start, start + 96 * step, step)
    return pd.date_range(metadata.get("index_start", "2000-01-01"), periods=96, freq="15min")


def _audit_pricing_resources(inputs_dir, original_path, device):
    """Compare the declared planning domain with the original physical devices."""
    inputs_dir, original_path = Path(inputs_dir), Path(original_path)
    candidates = []
    if original_path.is_file():
        candidates.append(original_path)
    candidates.extend((inputs_dir / "pricing.json", inputs_dir / "case.json",
                       inputs_dir.parent / "case.json"))
    declarations, fingerprints = {}, {}
    for candidate in candidates:
        candidate = candidate.resolve()
        if not candidate.exists() or str(candidate) in declarations:
            continue
        value = _read(candidate)
        if "pricing_case" in value:
            pricing_path = (candidate.parent / value["pricing_case"]).resolve()
            fingerprints[str(candidate)] = _sha(candidate)
            declarations[str(candidate)] = value
            candidates.append(pricing_path)
        elif {"resource_ids", "node_labels", "arrays_file"} <= value.keys():
            declarations[str(candidate)] = value
    audits = []
    attributes = {"resource_lower_kw":"delta_lower_kw", "resource_upper_kw":"delta_upper_kw",
                  "resource_cost":"cost_full_activation", "resource_efficiency":"eta",
                  "resource_energy_lower_kwh":"energy_low", "resource_energy_upper_kwh":"energy_high",
                  "resource_reset":"reset"}
    for filename, declaration in declarations.items():
        if "resource_ids" not in declaration:
            continue
        if declaration["resource_ids"] != device.ids or declaration["node_labels"] != device.labels:
            raise ValueError("Planning and physical resource/terminal identities differ")
        array_path = (Path(filename).parent / declaration["arrays_file"]).resolve()
        with np.load(array_path, allow_pickle=False) as arrays:
            for name, attribute in attributes.items():
                if name not in arrays or not np.array_equal(arrays[name], getattr(device, attribute)):
                    raise ValueError("Planning differs from original device property: " + name)
        fingerprints[filename] = _sha(filename)
        fingerprints[str(array_path)] = _sha(array_path)
        audits.append(dict(path=filename, arrays=str(array_path), properties=list(attributes),
                           resource_count=device.R, identity_and_values_pass=True))
    if not audits:
        raise ValueError("An explicit planning resource declaration is required for the AC audit")
    return dict(passed=True, declarations=audits, fingerprints=fingerprints)


def open_runtime(case_dir, freeze_path):
    """Verify the price freeze, then load physical data and original producer.

    case_dir may be the inputs directory or its parent package directory.
    Required inputs are physical.json, physical.npz and devices.json. Planning
    case.json/arrays.npz, when present, are checked against the original device
    capabilities without changing them.
    """
    frozen = _freeze(freeze_path)
    original_path = Path(case_dir).resolve()
    case_dir = original_path.parent if original_path.is_file() else original_path
    if not (case_dir / "physical.json").exists():
        case_dir = case_dir / "inputs"
    metadata_path = case_dir / "physical.json"
    metadata = _read(metadata_path)
    if metadata.get("case_name") != "paper15":
        raise ValueError("This physical adapter supports the fixed paper15 case")
    if metadata.get("power_unit") != "pu" or metadata.get("voltage_unit") != "pu":
        raise ValueError("Explicit p.u. natural-power and root-voltage units are required")
    labels = list(metadata["labels"])
    if labels != list(range(101, 116)):
        raise ValueError("Original paper15 terminal ordering is required")
    sim, helpers, device_module, profile_module, manifest = _vendor()
    context = sim._prepare_case_context("paper15", 96,
        scenario_suite=metadata.get("scenario_suite", "reference"),
        root_observation="noisy", root_meter_noise_rel=float(metadata["v_sigma"]),
        root_sigma=float(metadata.get("root_sigma", .003)),
        impedance_scale=float(metadata["impedance_scale"]))
    locked = dict(root_label=int(metadata["root_label"]), base_kv=float(metadata["base_kv"]),
                  base_mva=float(metadata["base_mva"]), base_kw=float(metadata["base_kw"]))
    helpers.validate_context_metadata(context, locked, labels)
    if not np.isclose(locked["base_kw"], locked["base_mva"]*1000, atol=1e-12, rtol=0):
        raise ValueError("Power base differs from the original network")
    context.net.check_terminal_load_only(strict=True)

    physical_path = case_dir / metadata.get("arrays_file", "physical.npz")
    with np.load(physical_path, allow_pickle=False) as arrays:
        p = np.asarray(arrays["natural_p"], float).copy()
        q = np.asarray(arrays["natural_q"], float).copy()
        root = np.asarray(arrays["root_true"], float).copy()
        index = _physical_index(arrays, metadata)
    if (p.shape != (96, 15) or q.shape != p.shape or root.shape != (96,)
            or not all(np.isfinite(value).all() for value in (p, q, root))
            or not (root > 0).all() or not index.is_unique or not index.is_monotonic_increasing):
        raise ValueError("The sealed physical day must contain all aligned finite 96-slot channels")
    background_metadata = metadata.get("background", metadata)
    background = sim._BaseTrajectory(
        p_base=pd.DataFrame(p, index=index, columns=labels),
        q_base=pd.DataFrame(q, index=index, columns=labels),
        root_true=pd.Series(root, index=index, name="V_root_true"),
        **{key: background_metadata[key] for key in
           ("name", "profile", "shifted_seed", "scenario_index", "replicate")})

    resource_path = case_dir / metadata.get("devices_file", "devices.json")
    resources = _read(resource_path)
    device = device_module.Devices(resources["allocation"], resources["config"], profile_module.profiles)
    if device.R != 60 or device.T != 96 or device.labels != labels:
        raise ValueError("The original 60-resource, 96-slot paper15 domain is required")
    resource_audit = _audit_pricing_resources(case_dir, original_path, device)
    return dict(sim=sim, context=context, checker=helpers._physical_checks,
                legacy=SimpleNamespace(scene=helpers.scene), execution=helpers.execute,
                truth_function=helpers.effective_truth_supports,
                metadata=locked, labels=labels, background=background, device=device,
                pq_sigma=float(metadata["pq_sigma"]), voltage_sigma=float(metadata["v_sigma"]),
                freeze=frozen, physical_fingerprints=_background_hashes(background),
                physical_reference=dict(path=str(physical_path),sha256=_sha(physical_path)),
                input_fingerprints={p.name:_sha(p) for p in (metadata_path,physical_path,resource_path)},
                physical_metadata_sha256=_sha(metadata_path),
                physical_metadata_hash=_sha(metadata_path),
                resource_identity_audit=resource_audit,
                vendor_manifest_sha256=_sha(VENDOR / "vendor_manifest.json"))


def evaluate(runtime, u, multiplier, meter_seed, record):
    """Run original resource projection, AC and noisy observation once.

    meter_seed controls observation noise only. True natural P/Q and root
    voltage are never regenerated or adjusted for a policy or random path.
    Returned power_channels use p.u.; drop_micro uses 1e-6 p.u.^2.
    """
    frozen = runtime["freeze"]
    if _sha(frozen["path"]) != frozen["sha256"] or _sha(frozen["quote"]) != frozen["quote_sha256"]:
        raise ValueError("Price freeze changed before physical execution")
    device = runtime["device"]
    u, multiplier = np.asarray(u, float), np.asarray(multiplier, float)
    if (u.shape != (60, 96) or multiplier.shape != u.shape
            or not np.isfinite(u).all() or not np.isfinite(multiplier).all()
            or not device.audit(u)["passed"]):
        raise ValueError("Finite, feasible original-domain nominal power and multiplier are required")
    if isinstance(meter_seed, (bool, np.bool_)) or not isinstance(meter_seed, (int, np.integer)) or meter_seed < 0:
        raise ValueError("A nonnegative integer meter stream seed is required")
    rng = np.random.default_rng(int(meter_seed))
    shifted = int(rng.integers(1, 2**62))
    replicate = int(rng.integers(1, 2**31))
    background = replace(runtime["background"], shifted_seed=shifted, replicate=replicate)
    if _background_hashes(background) != runtime["physical_fingerprints"]:
        raise ArithmeticError("Meter-only override changed physical natural data")
    result = runtime["execution"](runtime, background, device, u, multiplier, str(record))
    scene = result["scene"]
    power = drop = None
    if scene is not None:
        power = np.column_stack((np.asarray(scene["P_terminal"]), np.asarray(scene["Q_terminal"])))
        voltage = np.asarray(scene["V_terminal"])
        observed_root = np.asarray(scene["V_root_observed"]).reshape(-1, 1)
        drop = 1e6 * (observed_root**2 - voltage**2)
    if _background_hashes(runtime["background"]) != runtime["physical_fingerprints"]:
        raise ArithmeticError("Original natural data mutated during execution")
    result.update(power=power, power_channels=power, drop_micro=drop,
                  delivered=result["private"]["delivered"],
                  meter_record=dict(stream_seed=int(meter_seed),terminal_seed=shifted+2000,
                    root_seed=70000019+100003*replicate+int(background.scenario_index),
                    identifiers_overridden_for_observation_only=True,
                    physical_reference=runtime["physical_reference"]),
                  physical_fingerprints=runtime["physical_fingerprints"])
    return result


def finalize(runtime, predictions_freeze_path, selected_supports=None):
    """Read true supports only after the caller has frozen all selections.

    predictions_freeze_path is JSON with status='frozen' and a path/SHA-256
    reference named predictions. The caller must keep the returned truth in
    private evaluation output; it is not a planning input.
    """
    freeze_path = Path(predictions_freeze_path).resolve()
    freeze = _read(freeze_path)
    if freeze.get("status") != "frozen":
        raise ValueError("All predictions must be frozen before truth evaluation")
    prediction_path = _checked(freeze["predictions"], freeze_path.parent)
    truth = {tuple(sorted(block)) for block in runtime["truth_function"](runtime["context"].net)}
    supports = [list(x) for x in sorted(truth,key=lambda x:(len(x),x))]
    output = dict(supports=supports, truth_supports=supports,
                  predictions_sha256=_sha(prediction_path),scope="this frozen physical case")
    if selected_supports is not None:
        selected = {tuple(sorted(block)) for block in selected_supports}
        output.update(missing=len(truth-selected),extra=len(selected-truth),
                      wrong_or_missing=len(truth ^ selected),complete=truth==selected)
    return output
