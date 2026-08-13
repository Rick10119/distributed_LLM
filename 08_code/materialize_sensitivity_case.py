#!/usr/bin/env python3
"""Materialize one bounded OAT case as a normal core-model run config."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ALLOWED_OVERRIDE_PATHS = {
    "paths.hourly_industry_profiles",
    "demand.flexibility_scenario",
    "demand.effective_service.parameter_case",
    "demand.effective_service.compute_efficiency_case",
    "server.installed_reserve_fraction",
    "server.n_plus_spare_server_groups",
    "server.maximum_wall_power_kw",
    "server.online_idle_wall_power_kw",
    "server.marginal_facility_multiplier",
    "compute_hardware.cpu_server_overrides.marginal_facility_multiplier",
    "compute_hardware.cpu_server_overrides.installed_reserve_fraction",
    "compute_hardware.routing_case",
    "compute_hardware.cpu_time_multiplier_factor",
    "model.server_groups_integer",
    "energy.maximum_demand_rmb_per_kw_month",
    "energy.battery_investment_enabled",
    "energy.pv_capacity_mode",
    "energy.grid_expansion_limit_mw",
    "energy.grid_expansion_objective_penalty_rmb_per_mw_year",
    "hybrid_cloud.enabled",
    "hybrid_cloud.maximum_cloud_service_share",
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def leaf_paths(mapping: dict[str, Any], prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            paths |= leaf_paths(value, path)
        else:
            paths.add(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    if args.case_id == registry["reference_case"]["case_id"]:
        factor_id, case_name = "REF", "baseline"
        factor = {
            "label_cn": "同版本基准",
            "baseline_value": registry["reference_case"]["display_value"],
            "primary_claims": ["C2"],
        }
        case = registry["reference_case"]
    else:
        factor_id, case_name = args.case_id.split("__", 1)
        factor = registry["factors"][factor_id]
        case = factor["cases"][case_name]
    overrides = case.get("overrides", {})
    unknown = leaf_paths(overrides) - ALLOWED_OVERRIDE_PATHS
    if unknown:
        raise ValueError(f"Unsupported smoke-test overrides: {sorted(unknown)}")

    payload: dict[str, Any] = {
        "run_config_path": args.output.as_posix(),
        "model_version": registry["model_version"],
        "selected_industries": registry.get("selected_industries", [registry["industry"]]),
        "selected_scenarios": registry["architectures"],
        "industry_parameter_case": "base",
        "sensitivity_metadata": {
            "sensitivity_version": registry["sensitivity_version"],
            "factor_id": factor_id,
            "factor_label_cn": factor["label_cn"],
            "case_name": case_name,
            "display_value": case["display_value"],
            "baseline_value": factor["baseline_value"],
            "primary_claims": factor["primary_claims"],
            "baseline_invariant": bool(registry.get("baseline_invariant", False)),
        },
    }
    common_overrides = registry.get("common_overrides", {})
    unknown_common = leaf_paths(common_overrides) - ALLOWED_OVERRIDE_PATHS
    if unknown_common:
        raise ValueError(f"Unsupported common smoke-test overrides: {sorted(unknown_common)}")
    payload = deep_merge(payload, common_overrides)
    payload = deep_merge(payload, overrides)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
