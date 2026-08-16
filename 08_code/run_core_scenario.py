#!/usr/bin/env python3
"""Run one industry and one equal-service deployment scenario."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_code"))

from core.capacity import (
    arrival_time_required_server_groups,
    average_required_server_groups,
    local_installed_capacity_floor,
)
from core.config import deep_merge, load_config, write_resolved_config
from core.data import load_industry_inputs, read_core_grid_energy_prices, scale_task_workload, scale_workload
from core.io import read_json, write_csv
from core.model import optimize_host
from core.representative_group import read_representative_groups, scenario_scale


COST_COMPONENTS = (
    "annual_server_cost_rmb",
    "annual_pv_cost_rmb",
    "annual_battery_cost_rmb",
    "annual_flat_energy_cost_rmb",
    "annual_maximum_demand_cost_rmb",
    "annual_model_initialization_cost_rmb",
    "annual_model_storage_cost_rmb",
    "annual_model_operations_cost_rmb",
    "annual_grid_expansion_objective_penalty_rmb",
    "annual_cloud_subscription_cost_rmb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--hourly-output", type=Path, required=True)
    parser.add_argument("--resolved-config-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(ROOT, args.defaults, args.config)
    group = read_representative_groups(
        ROOT / config["paths"]["representative_group_report"]
    )[args.industry]
    host_count = int(config.get("industry_host_counts", {}).get(args.industry, 1))
    scale = scenario_scale(
        group,
        config["industry_parameter_case"],
        args.scenario,
        industry_host_count=host_count,
    )
    inputs = load_industry_inputs(config, args.industry)
    hardware_mode = str(config["compute_hardware"]["mode"])
    rigid_by_task = None
    heterogeneous_hardware = None
    minimum_installed_hardware_groups = None
    if hardware_mode == "heterogeneous_cpu_gpu":
        routing = yaml.safe_load(
            (ROOT / config["compute_hardware"]["routing_config"]).read_text(encoding="utf-8")
        )
        routing_case = str(
            config["compute_hardware"].get(
                "routing_case", routing["active_core_routing_case"]
            )
        )
        if routing_case not in routing["routing_cases"]:
            raise ValueError(f"Unsupported CPU/GPU routing case: {routing_case}")
        cpu_fractions = {
            task: float(value)
            for task, value in routing["routing_cases"][routing_case].items()
        }
        cpu_multipliers = {
            task: float(value)
            for task, value in routing["core_cpu_server_hour_per_reference_l20_accelerator_hour"].items()
            if task != "rationale"
        }
        cpu_time_factor = float(
            config["compute_hardware"].get("cpu_time_multiplier_factor", 1.0)
        )
        cpu_multipliers = {
            task: value * cpu_time_factor for task, value in cpu_multipliers.items()
        }
        cpu_server = deep_merge(
            dict(routing["cpu_server"]),
            dict(config["compute_hardware"].get("cpu_server_overrides", {})),
        )
        cpu_price_case = str(config["compute_hardware"].get("local_cpu_price_case", "base"))
        cpu_price_cases = cpu_server.get("purchase_cost_cases_rmb", {})
        if cpu_price_case not in cpu_price_cases:
            raise ValueError(f"Unsupported local CPU purchase-price case: {cpu_price_case}")
        cpu_server["purchase_price_case"] = cpu_price_case
        cpu_server["purchase_cost_rmb"] = float(cpu_price_cases[cpu_price_case])
        cpu_server.setdefault("normal_dispatch_reserve_fraction", 0.0)
        rigid_by_task, jobs = scale_task_workload(inputs, scale.ai_service_scale_per_host)
        rigid = sum(rigid_by_task.values(), np.zeros_like(inputs.rigid_service_units))
        arrival_by_task = {task: values.copy() for task, values in rigid_by_task.items()}
        for job in jobs:
            arrival_by_task[job.task_id][job.release_hour] += job.amount_service_units
        gpu_compute = np.zeros_like(rigid)
        cpu_compute = np.zeros_like(rigid)
        for task, arrivals in arrival_by_task.items():
            cpu_fraction = float(cpu_fractions.get(task, 0.0))
            gpu_compute += arrivals * inputs.accelerator_h_per_service_unit * (1.0 - cpu_fraction)
            cpu_compute += arrivals * inputs.accelerator_h_per_service_unit * cpu_fraction * float(cpu_multipliers.get(task, 1.0))
        arrival_peak_gpu_groups = float(np.max(gpu_compute)) / float(config["server"]["accelerators_per_server"])
        arrival_peak_cpu_groups = float(np.max(cpu_compute)) / float(cpu_server["service_capacity_cpu_server_h_per_h"])
        average_required_gpu_groups = average_required_server_groups(
            float(gpu_compute.sum()),
            len(gpu_compute),
            float(config["server"]["accelerators_per_server"]),
        )
        average_required_cpu_groups = average_required_server_groups(
            float(cpu_compute.sum()),
            len(cpu_compute),
            float(cpu_server["service_capacity_cpu_server_h_per_h"]),
        )
        minimum_installed_hardware_groups = {
            "gpu": local_installed_capacity_floor(average_required_gpu_groups, float(config["server"]["installed_reserve_fraction"]), int(config["server"]["n_plus_spare_server_groups"])),
            "cpu": local_installed_capacity_floor(
                average_required_cpu_groups,
                float(cpu_server["installed_reserve_fraction"]),
                int(config["server"]["n_plus_spare_server_groups"]),
            ) if average_required_cpu_groups > 0 else 0.0,
        }
        heterogeneous_hardware = {
            "routing_case": routing_case,
            "cpu_fraction_by_task": cpu_fractions,
            "cpu_compute_multiplier_by_task": cpu_multipliers,
            "cpu_server": cpu_server,
        }
    else:
        rigid, jobs = scale_workload(inputs, scale.ai_service_scale_per_host)
    grid_prices = read_core_grid_energy_prices(config)
    baseline = read_json(args.baseline_summary)
    baseline_model = baseline["model"]
    baseline_net_peak_mw = float(baseline_model["grid_import_peak_mw"])
    # Capacity planning uses mean compute demand across the modeled horizon.
    # Hourly throughput and job deadlines remain responsible for any capacity
    # needed above the mean, so temporal shifting retains its planning value.
    if hardware_mode != "heterogeneous_cpu_gpu":
        arrival_peak_gpu_groups = arrival_time_required_server_groups(
            rigid,
            ((job.release_hour, job.amount_service_units) for job in jobs),
            inputs.accelerator_h_per_service_unit,
            float(config["server"]["accelerators_per_server"]),
        )
        average_required_gpu_groups = average_required_server_groups(
            (float(rigid.sum()) + sum(job.amount_service_units for job in jobs))
            * inputs.accelerator_h_per_service_unit,
            len(rigid),
            float(config["server"]["accelerators_per_server"]),
        )
        arrival_peak_cpu_groups = 0.0
        average_required_cpu_groups = 0.0
        minimum_installed_server_groups = local_installed_capacity_floor(
            average_required_gpu_groups,
            float(config["server"]["installed_reserve_fraction"]),
            int(config["server"]["n_plus_spare_server_groups"]),
        )
    else:
        minimum_installed_server_groups = None
    if bool(config.get("hybrid_cloud", {}).get("enabled", False)):
        minimum_installed_server_groups = None
        minimum_installed_hardware_groups = {}
    result = optimize_host(
        config,
        base_load_mw=inputs.base_load_mw * scale.base_load_scale_per_host,
        pv_capacity_factor=inputs.pv_capacity_factor,
        roof_area_m2=inputs.roof_area_proxy_m2,
        rigid_service_units=rigid,
        flexible_jobs=jobs,
        grid_energy_price_rmb_per_mwh=grid_prices,
        existing_grid_capacity_mw=baseline_net_peak_mw,
        minimum_installed_server_groups=minimum_installed_server_groups,
        rigid_service_units_by_task=rigid_by_task,
        heterogeneous_hardware=heterogeneous_hardware,
        minimum_installed_hardware_groups=minimum_installed_hardware_groups,
    )
    represented_days = float(result.summary["represented_days"])
    row: dict[str, object] = {
        "model_version": config["model_version"],
        "industry_code": args.industry,
        "industry_name": inputs.industry_name,
        "scenario": args.scenario,
        "parameter_case": config["industry_parameter_case"],
        "compute_efficiency_case": config["demand"]["effective_service"]["compute_efficiency_case"],
        "compute_efficiency_evidence_status": config["demand"]["effective_service"]["compute_efficiency_evidence_status"],
        "server_maximum_wall_power_kw": config["server"]["maximum_wall_power_kw"],
        "server_marginal_facility_multiplier": config["server"]["marginal_facility_multiplier"],
        "server_groups_integer": result.summary["server_groups_integer"],
        "compute_hardware_mode": hardware_mode,
        "active_hardware_routing_case": heterogeneous_hardware["routing_case"] if heterogeneous_hardware else "gpu_only",
        "local_cpu_purchase_price_case": cpu_server["purchase_price_case"] if heterogeneous_hardware else "not_applicable",
        "local_cpu_purchase_price_rmb": cpu_server["purchase_cost_rmb"] if heterogeneous_hardware else 0.0,
        "grid_energy_price_mode": result.summary["grid_energy_price_mode"],
        "pv_capacity_mode": result.summary["pv_capacity_mode"],
        "grid_capacity_upgrade_boundary": config["model"]["grid_capacity_upgrade_boundary"],
        "hybrid_cloud_enabled": bool(config.get("hybrid_cloud", {}).get("enabled", False)),
        "maximum_cloud_service_share": float(config.get("hybrid_cloud", {}).get("maximum_cloud_service_share", 0.0)),
        "per_host_cloud_service_share": result.summary["cloud_service_share"],
        "per_host_cloud_reserved_gpu_groups": result.summary["cloud_reserved_gpu_groups"],
        "per_host_cloud_reserved_cpu_groups": result.summary["cloud_reserved_cpu_groups"],
        "industry_equivalent_annual_cloud_service_units": result.summary["annual_cloud_service_units"] * scale.equivalent_host_multiplier,
        "no_ai_optimized_net_grid_peak_mw": baseline_net_peak_mw,
        "per_host_existing_grid_capacity_mw": result.summary["existing_grid_capacity_mw"],
        "roof_area_proxy_m2": inputs.roof_area_proxy_m2,
        "roof_area_case": inputs.roof_area_case,
        "roof_source_naics": inputs.roof_source_naics,
        "roof_mapping_type": inputs.roof_mapping_type,
        "roof_evidence_grade": inputs.roof_evidence_grade,
        "per_host_rooftop_pv_limit_mw": result.summary["rooftop_pv_limit_mw"],
        "mean_grid_energy_price_rmb_per_mwh": result.summary["mean_grid_energy_price_rmb_per_mwh"],
        "minimum_grid_energy_price_rmb_per_mwh": result.summary["minimum_grid_energy_price_rmb_per_mwh"],
        "maximum_grid_energy_price_rmb_per_mwh": result.summary["maximum_grid_energy_price_rmb_per_mwh"],
        "battery_energy_capex_rmb_per_kwh": result.summary["battery_energy_capex_rmb_per_kwh"],
        "battery_power_capex_rmb_per_kw": result.summary["battery_power_capex_rmb_per_kw"],
        "battery_annualized_cost_rmb_per_mw_year": result.summary["battery_annualized_cost_rmb_per_mw_year"],
        "group_share": scale.group_share,
        "group_factory_count": scale.group_factory_count,
        "factory_activity_share": scale.base_load_scale_per_host,
        "ai_service_scale_per_host": scale.ai_service_scale_per_host,
        "equivalent_host_multiplier": scale.equivalent_host_multiplier,
        "physical_host_count": scale.physical_host_count,
        "horizon_hours": result.summary["horizon_hours"],
        "represented_days": represented_days,
        "storage_cycle_horizon_hours": result.summary["storage_cycle_horizon_hours"],
        "maximum_flexible_deadline_h": result.summary["maximum_flexible_deadline_h"],
        "industry_daily_effective_service_units": inputs.daily_effective_service_units,
        "per_host_daily_effective_service_units": float((result.hourly["ai_executed_service_units"] + result.hourly["ai_cloud_service_units"]).sum()) / represented_days,
        "per_host_daily_local_service_units": float(result.hourly["ai_executed_service_units"].sum()) / represented_days,
        "per_host_daily_cloud_service_units": float(result.hourly["ai_cloud_service_units"].sum()) / represented_days,
        "reconstructed_industry_daily_effective_service_units": float((result.hourly["ai_executed_service_units"] + result.hourly["ai_cloud_service_units"]).sum()) / represented_days * scale.equivalent_host_multiplier,
        "accelerator_h_per_service_unit": inputs.accelerator_h_per_service_unit,
        "industry_reference_daily_accelerator_h": inputs.reference_daily_accelerator_h,
        "external_energy_low_twh": inputs.external_energy_low_twh,
        "external_energy_central_twh": inputs.external_energy_central_twh,
        "external_energy_high_twh": inputs.external_energy_high_twh,
        "derived_reference_energy_twh": inputs.derived_reference_energy_twh,
        "external_energy_alignment_ratio": inputs.external_energy_alignment_ratio,
        "reference_energy_server_groups": inputs.reference_energy_server_groups,
        "capacity_reference_server_groups_unshifted_arrival_peak": arrival_peak_gpu_groups,
        "capacity_reference_server_groups_horizon_average": average_required_gpu_groups,
        "minimum_installed_server_groups_from_capacity_rule": minimum_installed_server_groups,
        "capacity_reference_gpu_server_groups_unshifted_arrival_peak": arrival_peak_gpu_groups,
        "capacity_reference_cpu_server_groups_unshifted_arrival_peak": arrival_peak_cpu_groups,
        "capacity_reference_gpu_server_groups_horizon_average": average_required_gpu_groups,
        "capacity_reference_cpu_server_groups_horizon_average": average_required_cpu_groups,
        "minimum_installed_gpu_server_groups_from_capacity_rule": minimum_installed_hardware_groups["gpu"] if minimum_installed_hardware_groups else minimum_installed_server_groups,
        "minimum_installed_cpu_server_groups_from_capacity_rule": minimum_installed_hardware_groups["cpu"] if minimum_installed_hardware_groups else 0.0,
        "installed_capacity_rule": config["server"]["installed_capacity_rule"],
        "n_plus_spare_server_groups": config["server"]["n_plus_spare_server_groups"],
        "per_host_installed_server_groups": result.summary["installed_server_groups"],
        "industry_equivalent_installed_server_groups": result.summary["installed_server_groups"] * scale.equivalent_host_multiplier,
        "per_host_installed_gpu_server_groups": result.summary["installed_gpu_server_groups"],
        "per_host_installed_cpu_server_groups": result.summary["installed_cpu_server_groups"],
        "industry_equivalent_installed_gpu_server_groups": result.summary["installed_gpu_server_groups"] * scale.equivalent_host_multiplier,
        "industry_equivalent_installed_cpu_server_groups": result.summary["installed_cpu_server_groups"] * scale.equivalent_host_multiplier,
        "per_host_ai_facility_peak_mw": result.summary["ai_facility_peak_mw"],
        "per_host_annual_ai_facility_energy_twh": result.summary["annual_ai_facility_energy_twh"],
        "industry_equivalent_annual_ai_facility_energy_twh": result.summary["annual_ai_facility_energy_twh"] * scale.equivalent_host_multiplier,
        "industry_equivalent_annual_gpu_facility_energy_twh": result.summary["annual_gpu_facility_energy_twh"] * scale.equivalent_host_multiplier,
        "industry_equivalent_annual_cpu_facility_energy_twh": result.summary["annual_cpu_facility_energy_twh"] * scale.equivalent_host_multiplier,
        "per_host_annual_model_initialization_energy_twh": result.summary["annual_model_initialization_energy_twh"],
        "industry_equivalent_annual_model_initialization_energy_twh": result.summary["annual_model_initialization_energy_twh"] * scale.equivalent_host_multiplier,
        "industry_equivalent_annual_ai_facility_energy_including_initialization_twh": result.summary["annual_ai_facility_energy_including_initialization_twh"] * scale.equivalent_host_multiplier,
        "required_model_replicas": result.summary["required_model_replicas"],
        "minimum_server_groups_for_model_state": result.summary["minimum_server_groups_for_model_state"],
        "model_vram_gb_per_replica": result.summary["model_vram_gb_per_replica"],
        "per_host_model_storage_required_gb": result.summary["model_storage_required_gb"],
        "per_host_model_storage_available_gb": result.summary["model_storage_available_gb"],
    }
    realized_average_groups: dict[str, float] = {}
    realized_peak_groups: dict[str, float] = {}
    installed_groups: dict[str, float] = {}
    for hardware, capacity in {
        "gpu": float(config["server"]["accelerators_per_server"]),
        "cpu": float(cpu_server["service_capacity_cpu_server_h_per_h"]) if heterogeneous_hardware else 1.0,
    }.items():
        compute = result.hourly[f"{hardware}_compute_h"].to_numpy(dtype=float)
        average_groups = float(np.mean(compute)) / capacity
        peak_groups = float(np.max(compute, initial=0.0)) / capacity
        installed_value = float(result.summary[f"installed_{hardware}_server_groups"])
        realized_average_groups[hardware] = average_groups
        realized_peak_groups[hardware] = peak_groups
        installed_groups[hardware] = installed_value
        ratio = installed_value / average_groups if average_groups > 1e-12 else np.nan
        row[f"per_host_realized_average_required_{hardware}_server_groups"] = average_groups
        row[f"per_host_realized_peak_required_{hardware}_server_groups"] = peak_groups
        row[f"installed_{hardware}_capacity_to_average_local_demand_ratio"] = ratio
        row[f"realized_{hardware}_installed_capacity_utilization_fraction"] = 1.0 / ratio if np.isfinite(ratio) and ratio > 0 else np.nan
        row[f"realized_{hardware}_capacity_redundancy_fraction_above_average"] = ratio - 1.0 if np.isfinite(ratio) else np.nan
    average_total_groups = sum(realized_average_groups.values())
    installed_total_groups = sum(installed_groups.values())
    total_ratio = installed_total_groups / average_total_groups if average_total_groups > 1e-12 else np.nan
    row["per_host_realized_average_required_total_server_groups"] = average_total_groups
    row["installed_total_server_capacity_to_average_local_demand_ratio"] = total_ratio
    row["realized_total_installed_capacity_utilization_fraction"] = 1.0 / total_ratio if np.isfinite(total_ratio) and total_ratio > 0 else np.nan
    row["realized_total_capacity_redundancy_fraction_above_average"] = total_ratio - 1.0 if np.isfinite(total_ratio) else np.nan
    for key in (
        "rooftop_pv_capacity_mw",
        "battery_power_mw",
        "battery_energy_mwh",
        "grid_expansion_mw",
        "grid_import_peak_mw",
        "annual_grid_energy_twh",
        "annual_objective_rmb",
    ):
        row[f"per_host_{key}"] = result.summary[key]
        delta = float(result.summary[key]) - float(baseline_model[key])
        row[f"per_host_incremental_{key}"] = delta
        row[f"industry_equivalent_incremental_{key}"] = delta * scale.equivalent_host_multiplier
    aggregate_cost = 0.0
    for key in COST_COMPONENTS:
        unit_delta = float(result.summary[key]) - float(baseline_model[key])
        aggregate_delta = unit_delta * scale.equivalent_host_multiplier
        row[f"per_host_incremental_{key}"] = unit_delta
        row[f"industry_equivalent_incremental_{key}"] = aggregate_delta
        aggregate_cost += aggregate_delta
    row["industry_equivalent_incremental_total_cost_rmb"] = aggregate_cost
    row["per_host_incremental_annual_gpu_server_cost_rmb"] = float(result.summary["annual_gpu_server_cost_rmb"])
    row["per_host_incremental_annual_cpu_server_cost_rmb"] = float(result.summary["annual_cpu_server_cost_rmb"])
    row["industry_equivalent_incremental_annual_gpu_server_cost_rmb"] = float(result.summary["annual_gpu_server_cost_rmb"]) * scale.equivalent_host_multiplier
    row["industry_equivalent_incremental_annual_cpu_server_cost_rmb"] = float(result.summary["annual_cpu_server_cost_rmb"]) * scale.equivalent_host_multiplier
    row["industry_equivalent_incremental_annual_gpu_electricity_cost_rmb"] = float(result.summary["annual_gpu_electricity_cost_rmb"]) * scale.equivalent_host_multiplier
    row["industry_equivalent_incremental_annual_cpu_electricity_cost_rmb"] = float(result.summary["annual_cpu_electricity_cost_rmb"]) * scale.equivalent_host_multiplier
    server_split = row["industry_equivalent_incremental_annual_gpu_server_cost_rmb"] + row["industry_equivalent_incremental_annual_cpu_server_cost_rmb"]
    if abs(server_split - row["industry_equivalent_incremental_annual_server_cost_rmb"]) > max(1e-2, abs(server_split) * 1e-9):
        raise ValueError("Joint physical server-cost split does not reconcile")

    hourly = result.hourly.copy()
    hourly.insert(0, "scenario", args.scenario)
    hourly.insert(0, "industry_code", args.industry)
    hourly["industry_equivalent_ai_executed_service_units"] = (
        hourly["ai_executed_service_units"] * scale.equivalent_host_multiplier
    )
    hourly["industry_equivalent_ai_compute_accelerator_h"] = (
        hourly["ai_compute_accelerator_h"] * scale.equivalent_host_multiplier
    )
    hourly["industry_equivalent_ai_facility_power_mw"] = (
        hourly["ai_facility_power_mw"] * scale.equivalent_host_multiplier
    )
    write_csv(pd.DataFrame([row]), args.summary_output)
    write_csv(hourly, args.hourly_output)
    write_resolved_config(config, args.resolved_config_output)


if __name__ == "__main__":
    main()
