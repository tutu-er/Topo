"""Freeze the complete archived 72af runtime without editing a live checkout.

The snapshot includes tracked and untracked package source files. The optional
observer is installed only in the frozen copy; stripping this observer must
reproduce the original whole-module AST exactly. No estimator runs here.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_PACKAGE = Path("C:/Users/23761/.codex/worktrees/72af/Topo/rnj_wzzt_core/rnj_wzzt")
DEFAULT_ARCHIVE = ROOT / "research_experiments/runtime_snapshots/source_ac288_72af_20261018.zip"
DEFAULT_ARCHIVE_MANIFEST = DEFAULT_ARCHIVE.with_suffix(".manifest.json")
DEFAULT_RUNTIME = ROOT / "outputs/source_ac288_runtime_20261018"
CORE_RELATIVE = "estimation/laminar_l1_milp.py"
FIT_NAME = "fit_laminar_l1_sensitivity"


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _read_inventory(package):
    files = {}
    for path in sorted(Path(package).rglob("*")):
        relative = path.relative_to(package)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ValueError(f"package snapshot rejects symlinks: {relative}")
        if path.is_file():
            files[relative.as_posix()] = path.read_bytes()
    if "__init__.py" not in files or CORE_RELATIVE not in files:
        raise ValueError("source_package must be a complete rnj_wzzt package")
    return files


def read_archive_inventory(archive_path, *, expected_sha256=None, expected_inventory=None):
    """Read validated package members in memory; never extract archive paths."""
    payload = Path(archive_path).read_bytes()
    if expected_sha256 is not None and sha256_bytes(payload) != expected_sha256:
        raise ValueError("runtime source archive SHA256 differs from its manifest")
    files, seen, total = {}, set(), 0
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        for member in members:
            # ZipInfo normalizes backslashes on Windows; validate the raw name.
            name = member.orig_filename
            parts = name.split("/")
            mode = member.external_attr >> 16
            if (name != member.filename or "\\" in name or ":" in name or name.startswith("/")
                    or any(part in {"", ".", ".."} for part in parts)
                    or len(parts) < 2 or parts[0] != "rnj_wzzt"
                    or member.is_dir() or stat.S_ISLNK(mode)
                    or stat.S_IFMT(mode) not in {0, stat.S_IFREG}
                    or member.flag_bits & 1 or "__pycache__" in parts
                    or not name.endswith(".py")):
                raise ValueError(f"unsafe runtime archive member: {name!r}")
            if name in seen:
                raise ValueError(f"duplicate runtime archive member: {name!r}")
            seen.add(name)
            total += member.file_size
            if member.file_size > 8 * 1024 * 1024 or total > 64 * 1024 * 1024:
                raise ValueError("runtime source archive exceeds the source-size limit")
        for member in members:
            files[member.filename.removeprefix("rnj_wzzt/")] = archive.read(member)
    if "__init__.py" not in files or CORE_RELATIVE not in files:
        raise ValueError("archive must contain a complete rnj_wzzt package")
    actual = {name: sha256_bytes(value) for name, value in files.items()}
    if expected_inventory is not None and actual != expected_inventory:
        raise ValueError("runtime source archive inventory differs from its manifest")
    return files


def archived_source_inventory():
    evidence = json.loads(DEFAULT_ARCHIVE_MANIFEST.read_text(encoding="utf-8"))
    if evidence["archive_filename"] != DEFAULT_ARCHIVE.name:
        raise ValueError("runtime archive manifest names a different archive")
    files = read_archive_inventory(DEFAULT_ARCHIVE, expected_sha256=evidence["archive_sha256"],
                                   expected_inventory=evidence["source_sha256"])
    return files, evidence


def dependency_closure(files):
    """Validate local imports statically, including lazily imported Source code."""
    modules = {}
    for name in files:
        if not name.endswith(".py"):
            continue
        pieces = list(Path(name).with_suffix("").parts)
        if pieces[-1] == "__init__":
            pieces.pop()
        modules[".".join(["rnj_wzzt", *pieces])] = name
    graph, external, missing = {}, set(), []
    for module, name in modules.items():
        tree = ast.parse(files[name].decode("utf-8"), filename=name)
        local = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = module if name.endswith("__init__.py") else module.rpartition(".")[0]
                    parts = package.split(".")
                    base = parts[:len(parts) - node.level + 1]
                    target = ".".join([*base, *([node.module] if node.module else [])])
                else:
                    target = node.module or ""
                targets = [target]
                targets.extend(f"{target}.{alias.name}" for alias in node.names
                               if f"{target}.{alias.name}" in modules)
            else:
                continue
            for target in targets:
                if target == "rnj_wzzt" or target.startswith("rnj_wzzt."):
                    if target not in modules:
                        missing.append({"importer": name, "missing_module": target})
                    else:
                        local.add(target)
                elif target:
                    external.add(target.split(".")[0])
        graph[module] = sorted(local)
    if missing:
        raise ValueError(f"package local-import closure is incomplete: {missing}")
    return {"local_module_count": len(modules), "local_import_graph": graph,
            "external_top_level_imports": sorted(external), "missing_local_imports": []}


class _RemoveObserver(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name == "observe":
            return None
        if node.name == FIT_NAME:
            pairs = [(arg, default) for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                     if arg.arg != "progress_callback"]
            node.args.kwonlyargs = [arg for arg, default in pairs]
            node.args.kw_defaults = [default for arg, default in pairs]
        return self.generic_visit(node)

    def visit_If(self, node):
        if any(isinstance(value, ast.Name) and value.id == "progress_callback"
               for value in ast.walk(node.test)):
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "observe":
            return None
        return self.generic_visit(node)


def verify_observer_only(original, patched):
    original_ast = ast.parse(original)
    observed_ast = ast.parse(patched)
    restored = _RemoveObserver().visit(deepcopy(observed_ast))
    if ast.dump(original_ast, include_attributes=False) != ast.dump(restored, include_attributes=False):
        raise ValueError("observer stripping changed the original algorithm AST")
    return True


def add_passive_observer(source):
    """Add the same passive events as the D-drive observer, in one function."""
    newline = "\r\n" if "\r\n" in source else "\n"
    normalized = source.replace("\r\n", "\n")
    module = ast.parse(normalized)
    fit = next((node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == FIT_NAME), None)
    if fit is None:
        raise ValueError("full core fit entrypoint is missing")
    if any(isinstance(node, ast.Name) and node.id in {"observe", "progress_callback"} for node in ast.walk(fit)):
        raise ValueError("source fit already has an observer; review its semantics before freezing")
    lines = normalized.splitlines(keepends=True)
    block = "".join(lines[fit.lineno - 1:fit.end_lineno])
    def replace_once(before, after):
        nonlocal block
        if block.count(before) != 1:
            raise ValueError(f"observer anchor no longer unique: {before!r}")
        block = block.replace(before, after, 1)
    replace_once('    solver: str = "highs",\n',
                 '    solver: str = "highs",\n    progress_callback: Callable[[dict], None] | None = None,\n')
    replace_once('    _validate_solver(solver)\n',
                 '    _validate_solver(solver)\n    if progress_callback is not None and not callable(progress_callback):\n        raise ValueError("progress_callback must be callable or None")\n')
    anchor = '    seen_families: set[frozenset[IndexSupport]] = {frozenset(initial_family)}\n'
    observer = '''
    def observe(event):
        if progress_callback is not None:
            from copy import deepcopy
            progress_callback(deepcopy({
                "event": event,
                "terminal_labels": prepared.labels,
                "initial_supports": initial_family,
                "terminal_groups": partition.groups,
                "source_enabled": training_source is not None,
                "source_provenance": () if training_source is None else training_source.provenance,
                "path": tuple(path),
                "attempted_extensions": tuple(attempts),
                "selected_path_index": _choose_path_point(path),
                "r_upper_bound": r_bound,
                "x_upper_bound": x_bound,
                "bound_expansions": bound_expansions,
                "stop_reason": stop_reason if event == "stopped" else "in_progress",
            }))

    observe("initialized")
'''
    replace_once(anchor, anchor + observer)
    replace_once('        attempts.append(extension)\n', '        attempts.append(extension)\n        observe("extension_attempted")\n')
    replace_once('                stop_reason = "bound_expansion_limit"\n                break\n            if r_active:',
                 '                stop_reason = "bound_expansion_limit"\n                break\n            observe("bounds_expansion_requested")\n            if r_active:')
    replace_once('            iteration = frozen_support_count + 1\n            continue',
                 '            iteration = frozen_support_count + 1\n            observe("bounds_restarted")\n            continue')
    replace_once('            path_point(current, iteration, gain=float(gain), solver=extension.diagnostics)\n        )\n',
                 '            path_point(current, iteration, gain=float(gain), solver=extension.diagnostics)\n        )\n        observe("path_accepted")\n')
    replace_once('    selected_index = _choose_path_point(path)\n',
                 '    observe("stopped")\n    selected_index = _choose_path_point(path)\n')
    patched = "".join(lines[:fit.lineno - 1]) + block + "".join(lines[fit.end_lineno:])
    patched = patched.replace("\n", newline)
    verify_observer_only(source, patched)
    return patched


def verify_runtime(runtime_dir):
    root = Path(runtime_dir).resolve()
    manifest = json.loads((root / "runtime_manifest.json").read_text(encoding="utf-8"))
    actual = _read_inventory(root / "rnj_wzzt")
    original = _read_inventory(root / "source_snapshot" / "rnj_wzzt")
    if set(actual) != set(manifest["runtime_sha256"]) or set(original) != set(manifest["source_sha256"]):
        raise ValueError("runtime or original snapshot file inventory changed")
    for relative, expected in manifest["runtime_sha256"].items():
        if sha256_bytes(actual[relative]) != expected:
            raise ValueError(f"frozen runtime changed: {relative}")
    for relative, expected in manifest["source_sha256"].items():
        if sha256_bytes(original[relative]) != expected:
            raise ValueError(f"original frozen snapshot changed: {relative}")
    changed = sorted(name for name in actual if actual[name] != original[name])
    allowed = [CORE_RELATIVE] if manifest["observer_enabled"] else []
    if changed != allowed or changed != manifest["observer_changed_files"]:
        raise ValueError("frozen package has changes beyond the declared observer")
    if sha256_bytes((root / manifest["observer_diff"]).read_bytes()) != manifest["observer_diff_sha256"]:
        raise ValueError("observer diff evidence changed")
    if manifest["observer_enabled"]:
        verify_observer_only(original[CORE_RELATIVE].decode("utf-8"), actual[CORE_RELATIVE].decode("utf-8"))
    elif actual != original:
        raise ValueError("unobserved runtime differs from original package")
    dependency_closure(actual)
    return manifest


def ensure_runtime(runtime_dir=DEFAULT_RUNTIME, source_package=None, *, observer=True):
    """Freeze bytes/hash/closure once; existing snapshots are verified, never replaced."""
    root = Path(runtime_dir).resolve()
    if source_package is None:
        files, archive_evidence = archived_source_inventory()
        source_reference = str(DEFAULT_ARCHIVE) + "!/rnj_wzzt"
        source_policy = "repository archive of the complete approved 72af working-tree package"
    else:
        source_package = Path(source_package).resolve()
        if root == source_package or source_package in root.parents or root in source_package.parents:
            raise ValueError("runtime must be separate from the live source package")
        files, archive_evidence = _read_inventory(source_package), None
        source_reference = str(source_package)
        source_policy = "complete working-tree package including untracked source; no live checkout edits"
    closure = dependency_closure(files)
    runtime = dict(files)
    if observer:
        runtime[CORE_RELATIVE] = add_passive_observer(files[CORE_RELATIVE].decode("utf-8")).encode("utf-8")
    expected_source = {name: sha256_bytes(value) for name, value in files.items()}
    expected_runtime = {name: sha256_bytes(value) for name, value in runtime.items()}
    if (root / "runtime_manifest.json").exists():
        manifest = verify_runtime(root)
        if (manifest["source_sha256"] != expected_source or manifest["runtime_sha256"] != expected_runtime
                or manifest["observer_enabled"] != bool(observer)):
            raise ValueError("requested source or observer differs from frozen runtime; choose a new runtime directory")
        return manifest
    if root.exists() and any(root.iterdir()):
        raise ValueError("nonempty destination has no valid runtime manifest; preserve it and choose a new directory")
    root.mkdir(parents=True, exist_ok=True)
    for folder, inventory in ((root / "source_snapshot" / "rnj_wzzt", files), (root / "rnj_wzzt", runtime)):
        for name, value in inventory.items():
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
    changes = [name for name in files if files[name] != runtime[name]]
    diff = "".join(difflib.unified_diff(files[CORE_RELATIVE].decode("utf-8").splitlines(keepends=True),
                    runtime[CORE_RELATIVE].decode("utf-8").splitlines(keepends=True),
                    fromfile="source_snapshot/rnj_wzzt/" + CORE_RELATIVE,
                    tofile="rnj_wzzt/" + CORE_RELATIVE)) if observer else ""
    (root / "observer_only.diff").write_text(diff, encoding="utf-8", newline="")
    after = archived_source_inventory()[0] if source_package is None else _read_inventory(source_package)
    if after != files:
        raise ValueError("source package changed during copying; preserve snapshot and retry in a new directory")
    versions = {}
    for package in ("numpy", "pandas", "scipy", "networkx", "gurobipy", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_package": source_reference, "runtime_directory": str(root),
        "import_root": str(root), "package_directory": str(root / "rnj_wzzt"),
        "source_policy": source_policy,
        "source_sha256": expected_source, "runtime_sha256": expected_runtime,
        "observer_enabled": bool(observer), "observer_changed_files": changes,
        "observer_stripped_ast_equal": True, "observer_diff": "observer_only.diff",
        "observer_diff_sha256": sha256_bytes((root / "observer_only.diff").read_bytes()),
        "observer_events": ["initialized", "extension_attempted", "bounds_expansion_requested",
                            "bounds_restarted", "path_accepted", "stopped"] if observer else [],
        "observer_state": ["event", "terminal_labels", "initial_supports", "terminal_groups", "source_enabled",
                           "source_provenance", "path", "attempted_extensions", "selected_path_index",
                           "r_upper_bound", "x_upper_bound", "bound_expansions", "stop_reason"] if observer else [],
        "source_path_fields": ["source_h", "source_response_r", "source_response_x", "source_provenance"],
        "dependency_closure": closure, "environment_package_versions": versions,
        "preparer_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "estimator_or_solver_invoked": False,
    }
    if archive_evidence is not None:
        manifest["source_archive"] = {"path": str(DEFAULT_ARCHIVE),
                                      "sha256": archive_evidence["archive_sha256"],
                                      "manifest_path": str(DEFAULT_ARCHIVE_MANIFEST)}
    (root / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False,
                                                   allow_nan=False) + "\n", encoding="utf-8")
    verify_runtime(root)
    return manifest


def activate_runtime(runtime_dir):
    """Pin a verified runtime before any rnj_wzzt import in a fresh worker."""
    root = Path(runtime_dir).resolve()
    manifest = verify_runtime(root)
    if any(name == "rnj_wzzt" or name.startswith("rnj_wzzt.") for name in sys.modules):
        raise RuntimeError("activate frozen runtime before importing any rnj_wzzt module")
    sys.path.insert(0, str(root))
    return manifest


def verify_loaded_modules(runtime_dir):
    package = (Path(runtime_dir).resolve() / "rnj_wzzt").resolve()
    origins = {}
    for name, module in sorted(sys.modules.items()):
        if name != "rnj_wzzt" and not name.startswith("rnj_wzzt."):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(package):
            raise RuntimeError(f"rnj_wzzt module escaped frozen runtime: {name}: {origin}")
        origins[name] = str(Path(origin).resolve())
    return origins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, default=None,
                        help="explicit working-tree package override; default is the repository archive")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--no-observer", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = (verify_runtime(args.output_dir) if args.verify_only else
                ensure_runtime(args.output_dir, args.source_package, observer=not args.no_observer))
    print(json.dumps({key: manifest[key] for key in
                      ("runtime_directory", "observer_enabled", "observer_changed_files", "observer_stripped_ast_equal")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
