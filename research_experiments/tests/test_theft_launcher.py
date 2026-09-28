"""Check process isolation and CLI compatibility without running experiments."""

from pathlib import Path
import os
import subprocess
import sys

import pytest

from research_experiments.theft import run


@pytest.mark.parametrize("exit_code", [0, 7])
def test_launcher_forwards_arguments_workspace_and_exit_code(monkeypatch, exit_code):
    captured = {}

    def fake_run(command, *, cwd, check):
        captured.update(command=command, cwd=cwd, check=check)
        return subprocess.CompletedProcess(command, exit_code)

    monkeypatch.setattr(run.subprocess, "run", fake_run)
    monkeypatch.setattr(run.sys, "executable", "path with spaces/python.exe")
    arguments = ["null", "--replicates", "3", "--tree", "true"]

    assert run.main(arguments) == exit_code
    assert captured == {
        "command": [
            "path with spaces/python.exe",
            str(run.THEFT_WORKSPACE / "experiments" / "run_theft.py"),
            *arguments,
        ],
        "cwd": run.THEFT_WORKSPACE,
        "check": False,
    }
    assert arguments == ["null", "--replicates", "3", "--tree", "true"]


def test_launcher_reads_cli_arguments(monkeypatch):
    monkeypatch.setattr(run.sys, "argv", ["run.py", "--help"])

    def fake_run(command, **kwargs):
        assert command[-1:] == ["--help"]
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run.subprocess, "run", fake_run)
    assert run.main() == 0


@pytest.mark.parametrize("entrypoint", ["file", "module"])
def test_original_help_through_both_entrypoints(entrypoint, tmp_path):
    repository = Path(run.__file__).resolve().parents[2]
    if entrypoint == "file":
        command = [sys.executable, str(Path(run.__file__).resolve()), "--help"]
        working_directory = tmp_path
    else:
        command = [sys.executable, "-m", "research_experiments.theft.run", "--help"]
        working_directory = repository
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        command,
        cwd=working_directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--replicates" in result.stdout
    assert "m7_report" in result.stdout
    assert "--tree" in result.stdout
