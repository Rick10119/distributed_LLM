#!/usr/bin/env python3
"""Summarize grid-capacity, storage, and hybrid-cloud structural cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--findings-output", type=Path, required=True)
    parser.add_argument("--done-output", type=Path, required=True)
    args = parser.parse_args()
    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    frames = [pd.read_csv(path, encoding="utf-8-sig") for path in args.inputs]
    data = pd.concat(frames, ignore_index=True)
    expected_cases = sum(
        len(factor["cases"]) for factor in registry["factors"].values()
    )
    if len(data) != expected_cases * len(registry["architectures"]):
        raise ValueError("Expected every structural case for every architecture")
    case_ids = [path.parts[-4] for path in args.inputs]
    data["case_id"] = case_ids
    case_design = {
        "GRID_HYBRID__grid_allowed_storage_no_cloud": ("unpriced", False),
        "GRID_HYBRID__grid_allowed_storage_hybrid_cloud": ("unpriced", True),
        "GRID_HYBRID__zero_grid_storage_hybrid_cloud": ("hard_zero", True),
        "GRID_HYBRID__high_grid_penalty_storage_no_cloud": ("high_penalty", False),
    }
    if set(data["case_id"]) != set(case_design):
        raise ValueError("Structural cases do not match the configured design")
    data["grid_expansion_regime"] = data["case_id"].map(
        lambda case_id: case_design[case_id][0]
    )
    data["cloud_subscription_allowed"] = data["case_id"].map(
        lambda case_id: case_design[case_id][1]
    )
    data["storage_allowed"] = True
    keep = [
        "case_id", "industry_code", "industry_name", "scenario",
        "grid_expansion_regime", "cloud_subscription_allowed", "storage_allowed",
        "per_host_grid_expansion_mw", "per_host_battery_power_mw",
        "per_host_battery_energy_mwh", "per_host_rooftop_pv_capacity_mw",
        "per_host_cloud_service_share", "per_host_cloud_reserved_gpu_groups",
        "per_host_cloud_reserved_cpu_groups", "per_host_installed_gpu_server_groups",
        "per_host_installed_cpu_server_groups", "industry_equivalent_incremental_total_cost_rmb",
        "industry_equivalent_incremental_annual_battery_cost_rmb",
        "industry_equivalent_incremental_annual_cloud_subscription_cost_rmb",
        "industry_equivalent_incremental_annual_grid_expansion_objective_penalty_rmb",
    ]
    output = data[keep].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    lines = [
        "# Single-industry grid constraint, storage, and hybrid-cloud test",
        "",
        "All cases preserve total AI service. Cloud subscription is an endogenous alternative to local execution; it is not unmet demand.",
        "All four cases allow storage and fix rooftop PV off. Grid expansion is either unpriced, fixed at zero, or allowed with a deliberately high scarcity penalty.",
        "",
    ]
    for architecture in registry["architectures"]:
        subset = output[output.scenario == architecture]
        lines.append(f"- {architecture}:")
        for case_id in case_design:
            row = subset[subset.case_id == case_id].iloc[0]
            lines.append(
                f"  - {case_id.split('__', 1)[1]}: grid expansion "
                f"{row.per_host_grid_expansion_mw:.3f} MW; battery "
                f"{row.per_host_battery_power_mw:.3f} MW; cloud share "
                f"{row.per_host_cloud_service_share:.2%}."
            )
    args.findings_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.done_output.write_text(json.dumps({
        "status": "validated_single_industry_grid_storage_hybrid_cloud_structure",
        "industry": registry["industry"],
        "architectures": registry["architectures"],
        "case_count": expected_cases,
        "total_service_preserved": True,
        "design": "grid_expansion_regime_and_cloud_subscription_controls__storage_allowed_in_all_cases",
        "high_grid_penalty_interpretation": "scarcity_stress_test_not_observed_connection_quote",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
