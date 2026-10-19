"""Keep retired EV experiments out of ordinary test discovery."""

import os


# Stored results may themselves contain test snapshots and temporary modules.
# Exclude them from recursive discovery, including legacy review. Explicit
# command-line file paths can still bypass pytest's directory ignore rules.
collect_ignore_glob = ["results*", "archive*"]

if os.environ.get("TOPO_RUN_LEGACY_EV_TESTS") != "1":
    collect_ignore_glob.append("test_*.py")
