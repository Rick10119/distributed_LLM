#!/usr/bin/env python3
"""Combine scenario-level summaries for one industry without changing values."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = [pd.read_csv(path, encoding="utf-8-sig") for path in args.inputs]
    combined = pd.concat(frames, ignore_index=True)
    if combined[["industry_code", "scenario"]].duplicated().any():
        raise ValueError("Duplicate industry-scenario rows in combined summary")
    if combined["industry_code"].nunique() > 1:
        if "production_load_mode" not in combined.columns:
            raise ValueError("National aggregation requires recorded production-load modes")
        modes = set(combined["production_load_mode"].astype(str))
        valid_modes = {"calibrated_registry", "legacy_industry_electricity_share"}
        if not modes.issubset(valid_modes):
            raise ValueError(f"Unsupported production-load modes: {sorted(modes)}")
        c36_modes = set(
            combined.loc[combined["industry_code"].eq("C36"), "production_load_mode"].astype(str)
        )
        if c36_modes != {"calibrated_registry"}:
            raise ValueError("National aggregation requires the approved C36 calibrated boundary")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
