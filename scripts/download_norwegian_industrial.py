"""Download the CC BY 4.0 Norwegian industrial-grid dataset from Zenodo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen


RECORD_ID = "10361330"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"


def download(output: str | Path) -> None:
    """Download all files in Zenodo record 10361330 with progress output."""

    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    with urlopen(API_URL) as response:
        record = json.load(response)
    files = record["files"]
    for index, item in enumerate(files, start=1):
        destination = target / item["key"]
        if destination.exists() and destination.stat().st_size == int(item["size"]):
            status = "cached"
        else:
            with urlopen(item["links"]["self"]) as response:
                destination.write_bytes(response.read())
            status = "downloaded"
        print(f"[{index:02d}/{len(files):02d}] {status:10s} {item['key']}", flush=True)
    (target / "SOURCE.txt").write_text(
        "Dataset for a Norwegian medium and low voltage power distribution system with industrial loads\n"
        "Version 2.3, Zenodo record 10361330\n"
        "DOI: 10.5281/zenodo.10361330\n"
        "License: CC BY 4.0\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data_external/norwegian_industrial_zenodo_7123537",
    )
    args = parser.parse_args()
    download(args.output)


if __name__ == "__main__":
    main()
