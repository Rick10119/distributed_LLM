#!/usr/bin/env python3
"""Validate and combine the 31-industry group-architecture core package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_PAIRS = {
    ("IF", "actual_load"),
    ("IG_1host", "actual_load"),
    ("IG_1host", "zero_load"),
    ("IG_multisite", "actual_load"),
}
SCALABLE_COLUMNS = [
    "installed_gpu_server_groups",
    "installed_cpu_server_groups",
    "annual_server_cost_rmb",
    "annual_ai_energy_cost_rmb",
    "annual_incremental_maximum_demand_cost_rmb",
    "annual_incremental_total_cost_rmb",
    "annual_ai_facility_energy_twh",
    "sum_incremental_grid_peak_mw",
    "weekly_service_units",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--industries", nargs="+", required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--alignment-output", type=Path, required=True)
    parser.add_argument("--lineage-output", type=Path, required=True)
    parser.add_argument("--done-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    alignments = []
    lineages = []
    factory_counts = {}
    for industry in args.industries:
        folder = args.root / industry
        summary = pd.read_csv(folder / "summary.csv", encoding="utf-8-sig")
        pairs = set(zip(summary["architecture"], summary["base_load_case"]))
        if len(summary) != 4 or pairs != EXPECTED_PAIRS:
            raise ValueError(f"{industry}: expected four registered architecture/load pairs, found {pairs}")
        if set(summary["industry"].astype(str)) != {industry}:
            raise ValueError(f"{industry}: summary industry label is inconsistent")
        actual = summary[summary["base_load_case"].eq("actual_load")]
        if set(actual["architecture"]) != {"IF", "IG_1host", "IG_multisite"} or len(actual) != 3:
            raise ValueError(f"{industry}: actual-load core comparison is incomplete")
        if not bool(summary.loc[summary["architecture"].eq("IF"), "installed_server_groups_integer"].all()):
            raise ValueError(f"{industry}: IF installed capacity must be integer")
        if bool(summary.loc[summary["architecture"].ne("IF"), "installed_server_groups_integer"].any()):
            raise ValueError(f"{industry}: group architectures must use continuous installed capacity")
        reference = float(summary["weekly_service_units"].iloc[0])
        error = (summary["weekly_service_units"] - reference).abs().max() / max(abs(reference), 1e-12)
        if error > 1e-7:
            raise ValueError(f"{industry}: service conservation error {error:.3g}")
        summaries.append(actual)

        alignment = pd.read_csv(folder / "load_alignment_value.csv", encoding="utf-8-sig")
        if len(alignment) != 1 or alignment["architecture"].iloc[0] != "IG_1host":
            raise ValueError(f"{industry}: expected one IG_1host load-alignment row")
        alignments.append(alignment)

        lineage = pd.read_csv(folder / "curve_lineage.csv", encoding="utf-8-sig")
        lineage.insert(0, "industry", industry)
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        expected_factories = int(metadata["synthetic_factory_count"])
        if len(lineage) != expected_factories:
            raise ValueError(f"{industry}: curve lineage does not match its representative factory count")
        factory_counts[industry] = expected_factories
        group_share = float(metadata["group_share"])
        if not 0 < group_share <= 1:
            raise ValueError(f"{industry}: invalid representative group share {group_share}")
        multiplier = 1.0 / group_share
        actual = actual.copy()
        actual["representative_group_share"] = group_share
        actual["industry_equivalent_multiplier"] = multiplier
        for column in SCALABLE_COLUMNS:
            actual[f"industry_equivalent_{column}"] = actual[column].astype(float) * multiplier
        summaries[-1] = actual

        alignment = alignment.copy()
        alignment["representative_group_share"] = group_share
        alignment["industry_equivalent_multiplier"] = multiplier
        for column in [name for name in alignment.columns if name.startswith("load_alignment_value_") or name == "avoided_incremental_grid_peak_mw"]:
            alignment[f"industry_equivalent_{column}"] = alignment[column].astype(float) * multiplier
        alignments[-1] = alignment
        lineages.append(lineage)

    if len(args.industries) != 31 or len(set(args.industries)) != 31:
        raise ValueError("The national core package requires exactly 31 unique industries")
    core = pd.concat(summaries, ignore_index=True)
    alignment = pd.concat(alignments, ignore_index=True)
    lineage = pd.concat(lineages, ignore_index=True)
    if len(core) != 93 or len(alignment) != 31:
        raise AssertionError("Unexpected national core output dimensions")
    for path in (args.summary_output, args.alignment_output, args.lineage_output, args.done_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    core.to_csv(args.summary_output, index=False, encoding="utf-8-sig")
    alignment.to_csv(args.alignment_output, index=False, encoding="utf-8-sig")
    lineage.to_csv(args.lineage_output, index=False, encoding="utf-8-sig")
    payload = {
        "status": "validated",
        "industry_count": 31,
        "actual_load_architectures": ["IF", "IG_1host", "IG_multisite"],
        "actual_load_rows": 93,
        "ig_1host_zero_load_pair_rows": 31,
        "II_1host_in_core": False,
        "factory_counts": factory_counts,
        "evidence_status": "outputs_validated_not_interpreted",
    }
    args.done_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
