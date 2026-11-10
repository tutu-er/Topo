"""Audit the 90% target conditional on a specified delivery path."""
import argparse
import hashlib
import json
from pathlib import Path

import model_test as experiment


def main(archive, output):
    # Run the same two axes in a separate process with a conditional target.
    experiment.TARGET = 0.90
    experiment.main(archive, output)
    record = experiment.read(output)
    record["hashes"][Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record["interpretation"] = "p(background passes | specified delivery), numerical first crossing on each scanned axis"
    record["endpoint_grids"] = [{"angles": 4096, "offset": 0.5}, {"angles": 8192, "offset": 0.25}]
    fraction = record["shortfall"]["roots"][-1]["crossing"]
    energy = record["nominal"]["absolute_DR_kwh"]
    peak = record["nominal"]["max_sum_abs_kw"]
    record["units_at_conditional_crossing"] = {
        "shortfall_fraction": fraction,
        "shortfall_average_gross_kw": fraction * energy / 24,
        "shortfall_gross_kwh": fraction * energy,
        "shortfall_at_peak_gross_kw": fraction * peak,
        "delay_minutes": record["delay"]["roots"][-1]["crossing"],
    }
    for axis in ("shortfall", "delay"):
        assert all(a >= record["target"] > b for a, b in record[axis]["probabilities_at_interval"])
        assert record[axis]["at_crossing"]["passed"]
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.archive, args.output)
