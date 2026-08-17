#!/usr/bin/env python3
"""Validate and combine the 31-industry group-architecture core package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_ARCHITECTURES = ["IF", "IG_1host", "IG_multisite"]
PAIRS_BY_ARCHITECTURE = {
    "IF": {("IF", "actual_load")},
    "IG_1host": {("IG_1host", "actual_load"), ("IG_1host", "zero_load")},
    "IG_multisite": {("IG_multisite", "actual_load")},
}
SCALABLE_COLUMNS = [
    "installed_gpu_server_groups",
    "installed_cpu_server_groups",
    "average_required_gpu_server_groups",
    "average_required_cpu_server_groups",
    "average_required_total_server_groups",
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
    parser.add_argument("--architectures", nargs="+", default=DEFAULT_ARCHITECTURES)
    parser.add_argument(
        "--enforce-core-capacity-boundary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the core 15% average-demand headroom and zero N+1 spare. Disable for national OAT capacity screens.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    architectures = list(args.architectures)
    expected_pairs = set()
    for architecture in architectures:
        if architecture not in PAIRS_BY_ARCHITECTURE:
            raise ValueError(f"Unsupported architecture: {architecture}")
        expected_pairs |= PAIRS_BY_ARCHITECTURE[architecture]
    summaries = []
    alignments = []
    lineages = []
    factory_counts = {}
    ai_deployment_node_counts = {}
    production_load_modes = {}
    for industry in args.industries:
        folder = args.root / industry
        summary = pd.read_csv(folder / "summary.csv", encoding="utf-8-sig")
        pairs = set(zip(summary["architecture"], summary["base_load_case"]))
        if not expected_pairs.issubset(pairs):
            raise ValueError(f"{industry}: expected {sorted(expected_pairs)}, found {pairs}")
        summary = summary.loc[
            [pair in expected_pairs for pair in zip(summary["architecture"], summary["base_load_case"])]
        ].copy()
        if set(summary["industry"].astype(str)) != {industry}:
            raise ValueError(f"{industry}: summary industry label is inconsistent")
        actual = summary[summary["base_load_case"].eq("actual_load")]
        if set(actual["architecture"]) != set(architectures) or len(actual) != len(architectures):
            raise ValueError(f"{industry}: actual-load comparison is incomplete")
        if "IF" in architectures and not bool(summary.loc[summary["architecture"].eq("IF"), "installed_server_groups_integer"].all()):
            raise ValueError(f"{industry}: IF installed capacity must be integer")
        if bool(summary.loc[summary["architecture"].ne("IF"), "installed_server_groups_integer"].any()):
            raise ValueError(f"{industry}: group architectures must use continuous installed capacity")
        if bool(summary["online_server_groups_integer"].any()):
            raise ValueError(f"{industry}: hourly online capacity must remain continuous")
        reserve_values = summary["planning_reserve_fraction"].astype(float)
        spare_values = summary["n_plus_spare_server_groups"].astype(float)
        if reserve_values.nunique(dropna=False) != 1:
            raise ValueError(f"{industry}: planning reserve is inconsistent across architectures")
        if spare_values.nunique(dropna=False) != 1:
            raise ValueError(f"{industry}: N+k spare count is inconsistent across architectures")
        if args.enforce_core_capacity_boundary:
            if not ((reserve_values - 0.15).abs() <= 1e-12).all():
                raise ValueError(f"{industry}: core planning headroom must be 15%")
            if not (spare_values == 0.0).all():
                raise ValueError(f"{industry}: N+1 belongs to sensitivity analysis, not the core")
        reference = float(summary["weekly_service_units"].iloc[0])
        error = (summary["weekly_service_units"] - reference).abs().max() / max(abs(reference), 1e-12)
        if error > 1e-7:
            raise ValueError(f"{industry}: service conservation error {error:.3g}")
        summaries.append(actual)

        if "IG_1host" in architectures:
            alignment = pd.read_csv(folder / "load_alignment_value.csv", encoding="utf-8-sig")
            if len(alignment) != 1 or alignment["architecture"].iloc[0] != "IG_1host":
                raise ValueError(f"{industry}: expected one IG_1host load-alignment row")
            alignments.append(alignment)

        lineage = pd.read_csv(folder / "curve_lineage.csv", encoding="utf-8-sig")
        lineage.insert(0, "industry", industry)
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        load_calibration = metadata.get("load_calibration", {})
        production_load_modes[industry] = str(load_calibration.get("mode", "missing"))
        if "planning_reserve_fraction" in metadata:
            if abs(float(metadata["planning_reserve_fraction"]) - float(reserve_values.iloc[0])) > 1e-12:
                raise ValueError(f"{industry}: metadata planning reserve does not match summary")
        if "n_plus_spare_server_groups" in metadata:
            if abs(float(metadata["n_plus_spare_server_groups"]) - float(spare_values.iloc[0])) > 1e-12:
                raise ValueError(f"{industry}: metadata N+k spare does not match summary")
        if args.enforce_core_capacity_boundary:
            if abs(float(metadata.get("planning_reserve_fraction", reserve_values.iloc[0])) - 0.15) > 1e-12:
                raise ValueError(f"{industry}: metadata does not record the 15% core headroom")
            if abs(float(metadata.get("n_plus_spare_server_groups", spare_values.iloc[0]))) > 1e-12:
                raise ValueError(f"{industry}: metadata must record zero N+1 for the core")
        expected_factories = int(metadata["production_load_site_count"])
        expected_nodes = int(metadata["modeled_routing_node_count"])
        deployment_counts = {
            key: int(value)
            for key, value in metadata["ai_deployment_node_count_by_architecture"].items()
        }
        if deployment_counts != {
            "IF": expected_factories,
            "IG_1host": 1,
            "IG_multisite": expected_nodes,
        }:
            raise ValueError(
                f"{industry}: inconsistent architecture-specific AI deployment counts"
            )
        if metadata.get("multisite_AI_deployment_points_equal_routing_nodes") is not True:
            raise ValueError(f"{industry}: multisite AI deployment points must equal routing nodes")
        if metadata.get("electrical_load_aggregation_at_AI_nodes") is not False:
            raise ValueError(f"{industry}: production loads must remain on independent electrical boundaries")
        if len(lineage) != expected_nodes:
            raise ValueError(f"{industry}: curve lineage does not match its modeled routing-node count")
        if int(lineage["represented_production_load_site_count"].sum()) != expected_factories:
            raise ValueError(f"{industry}: representative-node weights do not reconstruct production-load sites")
        factory_counts[industry] = expected_factories
        ai_deployment_node_counts[industry] = deployment_counts
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

        if alignments:
            alignment = alignments[-1].copy()
            alignment["representative_group_share"] = group_share
            alignment["industry_equivalent_multiplier"] = multiplier
            alignment["production_load_mode"] = str(load_calibration["mode"])
            alignment["production_load_boundary_id"] = str(load_calibration["boundary_id"])
            for column in [name for name in alignment.columns if name.startswith("load_alignment_value_") or name == "avoided_incremental_grid_peak_mw"]:
                alignment[f"industry_equivalent_{column}"] = alignment[column].astype(float) * multiplier
            alignments[-1] = alignment
        lineages.append(lineage)

    if len(args.industries) != 31 or len(set(args.industries)) != 31:
        raise ValueError("The national core package requires exactly 31 unique industries")
    valid_load_modes = {"calibrated_registry", "legacy_industry_electricity_share"}
    observed_load_modes = set(production_load_modes.values())
    if not observed_load_modes or not observed_load_modes.issubset(valid_load_modes):
        raise ValueError(f"Unsupported or missing production-load modes: {observed_load_modes}")
    if production_load_modes.get("C36") != "calibrated_registry":
        raise ValueError("C36 must use the approved calibrated production-load registry boundary")
    load_mode_counts = pd.Series(production_load_modes).value_counts().to_dict()
    core = pd.concat(summaries, ignore_index=True)
    alignment = pd.concat(alignments, ignore_index=True) if alignments else pd.DataFrame()
    lineage = pd.concat(lineages, ignore_index=True)
    expected_actual_rows = 31 * len(architectures)
    expected_alignment_rows = 31 if "IG_1host" in architectures else 0
    if len(core) != expected_actual_rows or len(alignment) != expected_alignment_rows:
        raise AssertionError("Unexpected national output dimensions")
    for path in (args.summary_output, args.alignment_output, args.lineage_output, args.done_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    core.to_csv(args.summary_output, index=False, encoding="utf-8-sig")
    alignment.to_csv(args.alignment_output, index=False, encoding="utf-8-sig")
    lineage.to_csv(args.lineage_output, index=False, encoding="utf-8-sig")
    payload = {
        "status": "validated",
        "industry_count": 31,
        "actual_load_architectures": architectures,
        "actual_load_rows": expected_actual_rows,
        "ig_1host_zero_load_pair_rows": expected_alignment_rows,
        "II_1host_in_core": False,
        "factory_counts": factory_counts,
        "production_load_site_counts": factory_counts,
        "ai_deployment_node_counts": ai_deployment_node_counts,
        "production_load_mode_by_industry": production_load_modes,
        "production_load_mode_counts": load_mode_counts,
        "production_load_boundary_status": "mixed_explicit_C36_calibrated_other_industries_legacy_compatibility",
        "evidence_status": "outputs_validated_not_interpreted",
    }
    args.done_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
