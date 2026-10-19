"""Run the frozen scientific CLI without Python 3.11 periodic stack dumps.

The native dump_frame crash is archived separately.  No scientific option,
random stream, solver budget or saved-result rule changes here.  Fatal exception
reporting remains available; only dump_traceback_later's diagnostic timer is
suppressed in this fresh worker process.
"""
from __future__ import annotations

import faulthandler


def run_frozen_main(main=None):
    if main is None:
        from .ev_bayes_response_study import main
    original = faulthandler.dump_traceback_later
    faulthandler.cancel_dump_traceback_later()

    def no_periodic_dump(*args, **kwargs):
        return None

    faulthandler.dump_traceback_later = no_periodic_dump
    try:
        return main()
    finally:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later = original


if __name__ == "__main__":
    run_frozen_main()
