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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
