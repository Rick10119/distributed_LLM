#!/usr/bin/env python3
"""Extract model-ready base-load profiles from the historical peak-screen table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


KEEP = [
    "industry_code",
    "industry_name_cn",
    "temporal_scenario",
    "hour",
    "baseline_load_mw",
    "base_normalized_load",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lineage-output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    missing = set(KEEP) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing base-load fields: {sorted(missing)}")
    output = frame[KEEP].drop_duplicates().sort_values(
        ["industry_code", "temporal_scenario", "hour"]
    )
    expected = 31 * output["temporal_scenario"].nunique() * 24
    if len(output) != expected or output["industry_code"].nunique() != 31:
        raise ValueError("Historical table does not provide complete 31-industry profiles")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.lineage_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.lineage_output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": str(args.input),
                "source_sha256": digest,
                "operation": "select base-load identity, normalized shape, temporal scenario and hour; discard all historical equal-electricity AI fields",
                "output": str(args.output),
                "rows": len(output),
                "industries": 31,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
