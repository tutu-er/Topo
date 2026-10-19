"""Keep frozen evidence snapshots and generated outputs out of active tests.

Archives preserve old test source bytes for audit, not test discovery. This
also avoids machine-specific imports in historical snapshots on fresh clones.
"""

collect_ignore = ["evidence", "outputs"]
