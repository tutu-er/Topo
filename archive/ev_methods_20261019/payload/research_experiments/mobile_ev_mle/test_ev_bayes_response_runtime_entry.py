"""Diagnostic-only wrapper contract; no scientific experiment is launched."""
import sys

import pytest

from . import ev_bayes_response_runtime_entry as entry


def test_only_periodic_timer_is_disabled_and_restored(monkeypatch):
    calls = []
    original = lambda *a, **k: calls.append((a, k))
    monkeypatch.setattr(entry.faulthandler, "dump_traceback_later", original)
    cancelled = []
    monkeypatch.setattr(entry.faulthandler, "cancel_dump_traceback_later", lambda: cancelled.append(1))
    enable = entry.faulthandler.enable
    def frozen():
        entry.faulthandler.dump_traceback_later(300, repeat=True)
        assert entry.faulthandler.enable is enable
        return 17
    assert entry.run_frozen_main(frozen) == 17
    assert not calls and len(cancelled) == 2
    assert entry.faulthandler.dump_traceback_later is original


@pytest.mark.parametrize("error", [ValueError("science failed"), SystemExit(2), KeyboardInterrupt()])
def test_exceptions_are_preserved_and_timer_function_restored(monkeypatch, error):
    original = lambda *a, **k: None
    monkeypatch.setattr(entry.faulthandler, "dump_traceback_later", original)
    monkeypatch.setattr(entry.faulthandler, "cancel_dump_traceback_later", lambda: None)
    def frozen(): raise error
    with pytest.raises(type(error)) as caught:
        entry.run_frozen_main(frozen)
    assert caught.value is error
    assert entry.faulthandler.dump_traceback_later is original


def test_exact_cli_arguments_are_preserved(monkeypatch):
    argv = ["entry", "--output", "frozen", "--phase", "calibration", "--indices", "6"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(entry.faulthandler, "cancel_dump_traceback_later", lambda: None)
    seen = []
    entry.run_frozen_main(lambda: seen.append(sys.argv.copy()))
    assert seen == [argv] and sys.argv is argv
