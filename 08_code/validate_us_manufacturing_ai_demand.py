#!/usr/bin/env python3
"""Validate U.S. manufacturing demand and cost outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


TASKS = {"office", "agent", "vision", "maintenance", "scheduling", "simulation"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", type=Path, required=True)
    parser.add_argument("--task-summary", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--local-cost", type=Path, required=True)
    parser.add_argument("--cloud-cost", type=Path, required=True)
    parser.add_argument("--done-output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    service = pd.read_csv(args.service, encoding="utf-8-sig")
    task = pd.read_csv(args.task_summary, encoding="utf-8-sig")
    validation = pd.read_csv(args.validation, encoding="utf-8-sig")
    local = pd.read_csv(args.local_cost, encoding="utf-8-sig")
    cloud = pd.read_csv(args.cloud_cost, encoding="utf-8-sig")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if len(service) != 21 * 6 * 3 or service["naics3"].nunique() != 21:
        raise ValueError("Expected 21 NAICS3 x six tasks x three cases")
    if set(service["task_id"]) != TASKS or set(service["parameter_case"]) != {"low", "base", "high"}:
        raise ValueError("Task/case coverage incomplete")
    if not service["adoption_fraction"].between(0, 1).all() or not service["coverage_fraction"].between(0, 1).all():
        raise ValueError("Adoption/coverage must remain in [0,1]")
    if (service["effective_service_units_day"] <= 0).any():
        raise ValueError("Service demand must be positive")
    detail_total = service.groupby(["parameter_case", "task_id"])["effective_service_units_day"].sum().sort_index()
    task_total = task.set_index(["parameter_case", "task_id"])["effective_service_units_day"].sort_index()
    if (detail_total - task_total).abs().max() > 1e-6:
        raise ValueError("Task and industry service totals do not reconcile")
    non_token = task[~task["task_id"].isin(["office", "agent"])]
    if (non_token[["annual_input_tokens", "annual_output_tokens"]].abs().max().max() != 0):
        raise ValueError("Non-token tasks generated tokens")
    if set(validation["installed_reserve_fraction"]) != {0.10}:
        raise ValueError("Installed reserve must be exactly one 10% parameter")
    external = config["external_scale_validation"]
    for row in validation.itertuples(index=False):
        ratio = float(row.annual_ai_facility_energy_twh) / float(external["china_energy_twh"][row.parameter_case])
        if not float(external["expected_us_to_china_ratio_low"]) <= ratio <= float(external["expected_us_to_china_ratio_high"]):
            raise ValueError(f"{row.parameter_case} bottom-up U.S. energy is outside the external scale check")
    if "macro_calibration_multiplier" in service.columns:
        raise ValueError("Top-down macro calibration multiplier must not remain in service output")
    expected_local = local["annual_server_capital_facility_maintenance_cost_usd"] + local["annual_electricity_cost_usd"]
    if (expected_local - local["annual_local_cost_usd"]).abs().max() > 1e-5:
        raise ValueError("Local cost components do not reconcile")
    expected_cloud = cloud["annual_api_token_cost_usd"] + cloud["annual_residual_reserved_gpu_payment_usd"] + cloud["annual_cloud_storage_cost_usd_proxy"]
    if (expected_cloud - cloud["annual_full_cloud_cost_usd"]).abs().max() > 1e-5:
        raise ValueError("Cloud components do not reconcile")
    if cloud["ondemand_displayed"].astype(bool).any():
        raise ValueError("On-demand GPU must not be displayed")
    base_cloud_ratio = float(cloud.loc[cloud["parameter_case"] == "base", "ratio_to_local_same_case"].min())
    premium = config["core_cloud_premium_target"]
    if abs(base_cloud_ratio - float(premium["target_ratio"])) > float(premium["tolerance"]):
        raise ValueError("Least-cost base cloud provider is outside the configured core premium target")
    payload = {
        "status": "validated",
        "rows": len(service),
        "naics3": 21,
        "tasks": sorted(TASKS),
        "parameter_cases": ["low", "base", "high"],
        "checks": [
            "activity/task/case coverage", "adoption and coverage bounds", "industry/task reconciliation",
            "tokens restricted to office and agent", "single 10% installed reserve", "local cost reconciliation",
            "bottom-up formula and external scale check", "cloud cost reconciliation", "on-demand GPU excluded",
            "least-cost base cloud premium target",
        ],
    }
    args.done_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
