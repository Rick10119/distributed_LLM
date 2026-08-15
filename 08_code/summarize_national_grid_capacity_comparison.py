#!/usr/bin/env python3
"""Compare distributed architectures with one all-industry cloud center."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ARCHITECTURE_LABELS = {
    "IF": "factory_distributed",
    "IG": "group_deployment",
    "IG_1host": "group_single_host",
    "IG_multisite": "group_multisite",
    "II_1host": "industry_deployment",
}
GROUP_ARCHITECTURES = {"IF", "IG_1host", "IG_multisite"}
LEGACY_ARCHITECTURES = {"IF", "IG", "II_1host"}


def load_national(path: Path) -> pd.DataFrame:
    national = pd.read_csv(path, encoding="utf-8-sig")
    if {"architecture", "base_load_case", "industry"}.issubset(national.columns):
        national = national.loc[national["base_load_case"].eq("actual_load")].copy()
        national["scenario"] = national["architecture"].astype(str)
        national["industry_code"] = national["industry"].astype(str)
        national["industry_equivalent_incremental_grid_expansion_mw"] = national[
            "industry_equivalent_sum_incremental_grid_peak_mw"
        ]
    return national


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--national-summary", type=Path, required=True)
    parser.add_argument("--cloud-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--done-output", type=Path, required=True)
    args = parser.parse_args()

    national = load_national(args.national_summary)
    cloud = pd.read_csv(args.cloud_summary, encoding="utf-8-sig")
    present = set(national["scenario"])
    allowed = (GROUP_ARCHITECTURES, LEGACY_ARCHITECTURES, {"IG_1host"})
    if present not in allowed:
        raise AssertionError(
            "national summary must contain IG_1host, the full IF / IG_1host / IG_multisite set, "
            "or the legacy IF, IG and II_1host set"
        )
    if national["industry_code"].nunique() != 31:
        raise AssertionError("national summary must contain all 31 industries")
    if len(cloud) != 1 or int(cloud.iloc[0]["industry_count"]) != 31:
        raise AssertionError("cloud reference must be one center serving all 31 industries")

    cloud_mw = float(cloud.iloc[0]["incremental_grid_expansion_mw"])
    rows = []
    for scenario, group in national.groupby("scenario"):
        architecture_mw = float(group["industry_equivalent_incremental_grid_expansion_mw"].sum())
        avoided_mw = cloud_mw - architecture_mw
        rows.append(
            {
                "comparison_basis": "one_greenfield_cloud_center_serving_all_31_industries",
                "scenario": scenario,
                "architecture": ARCHITECTURE_LABELS[scenario],
                "architecture_national_incremental_grid_expansion_mw": architecture_mw,
                "all_industry_cloud_incremental_grid_expansion_mw": cloud_mw,
                "avoided_grid_connection_capacity_vs_cloud_mw": avoided_mw,
                "avoided_grid_connection_capacity_vs_cloud_fraction": avoided_mw / cloud_mw,
                "c2_direction_holds": bool(avoided_mw > 0),
            }
        )

    result = pd.DataFrame(rows).sort_values("scenario")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    payload = {
        "status": "validated",
        "comparison_basis": "one greenfield cloud center serving all 31 industries",
        "architectures": sorted(ARCHITECTURE_LABELS),
        "all_architectures_reduce_capacity_vs_cloud": bool(result["c2_direction_holds"].all()),
    }
    args.done_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
