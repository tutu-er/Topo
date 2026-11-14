"""Choose complete shared calendar blocks using source coverage alone.

This never reads voltage fits, candidate quality, truth, or recovery outcomes.
The initial fixed-date failures remain in their original validation output.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from ... import io
from .profiles import START_DATES


def qualify(archive, output, *, targets=START_DATES, max_forward_days=31):
    archive, output = Path(archive).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve the existing qualification protocol")
    if max_forward_days < 0:
        raise ValueError("Nonnegative calendar search length required")
    receipt = io.ref(archive)
    with np.load(archive, allow_pickle=False) as data:
        times = pd.DatetimeIndex(pd.to_datetime(data["timestamps_utc"], utc=True))
        if len(data["feeder_ids"]) < 18 or len(set(map(int, data["feeder_ids"]))) != len(data["feeder_ids"]):
            raise ValueError("18 distinct source feeders required")
        counts, p, q = data["valid_minutes"][:, :18], data["p_kw"][:, :18], data["q_kvar"][:, :18]
    rows = []
    for target in targets:
        candidates = []
        selected = None
        for offset in range(max_forward_days+1):
            date = (pd.Timestamp(target)+pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            local = pd.Timestamp(date+"T08:00:00", tz="Europe/Berlin")
            dst_ok = (local+pd.DateOffset(days=4)).tz_convert("UTC")-local.tz_convert("UTC") == pd.Timedelta(days=4)
            wanted = pd.date_range(local.tz_convert("UTC"), periods=384, freq="15min")
            ix = times.get_indexer(wanted)
            missing = int(np.sum(ix < 0))*18
            found = ix[ix >= 0]
            missing += int(np.sum((counts[found] < 12) | ~np.isfinite(p[found]) | ~np.isfinite(q[found])))
            candidates.append(dict(date=date, incomplete_node_time_bins=missing, dst_ok=bool(dst_ok)))
            if missing == 0 and dst_ok:
                selected = date
                break
        rows.append(dict(target=target, selected=selected, examined=candidates))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), archive=receipt,
                  selection="First common complete 4-day block, all first18 feeders, forward calendar search only",
                  max_forward_days=max_forward_days, inputs_read="timestamps, P/Q finite flags, valid-minute counts; no model results",
                  initial_fixed_date_failures_preserved=True, qualification_source=io.ref(__file__),
                  rows=rows, all_qualified=all(row["selected"] is not None for row in rows))
    io.write(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.archive, args.out)
    print({row["target"]: row["selected"] for row in result["rows"]})
