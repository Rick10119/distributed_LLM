#!/usr/bin/env python3
"""Validate one completed core scenario against physical and accounting invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_code"))

from core.capacity import local_installed_capacity_floor
from core.config import load_config


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
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--hourly", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def close(left: float, right: float, relative: float = 1e-7, absolute: float = 1e-6) -> bool:
    return abs(left - right) <= max(absolute, relative * max(abs(left), abs(right)))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    args = parse_args()
    config = load_config(ROOT, args.defaults, args.config)
    summary_frame = pd.read_csv(args.summary, encoding="utf-8-sig")
    require(len(summary_frame) == 1, "summary must contain one row")
    row = summary_frame.iloc[0]
    baseline = json.loads(args.baseline_summary.read_text(encoding="utf-8"))["model"]
    hourly = pd.read_csv(args.hourly, encoding="utf-8-sig").sort_values("hour")
    horizon = int(config["model"]["horizon_hours"])
    represented_days = horizon / 24.0
    require(len(hourly) == horizon and list(hourly["hour"]) == list(range(horizon)), "hourly output does not cover the configured continuous horizon")
    require(int(row["horizon_hours"]) == horizon, "scenario horizon mismatch")
    require(int(row["storage_cycle_horizon_hours"]) == horizon, "storage must cycle only over the full horizon")
    flexibility = pd.read_csv(
        ROOT / config["paths"]["flexibility_mapping"], encoding="utf-8-sig"
    )
    selected_flexibility = flexibility[
        flexibility["scenario"] == config["demand"]["flexibility_scenario"]
    ]
    require(not selected_flexibility.empty, "configured flexibility scenario is absent")
    expected_maximum_deadline = float(
        selected_flexibility[["intraday_deadline_h", "batch_deadline_h"]].max().max()
    )
    require(
        close(float(row["maximum_flexible_deadline_h"]), expected_maximum_deadline),
        "maximum flexible deadline does not match the selected flexibility scenario",
    )
    require(row["industry_code"] == args.industry, "industry code mismatch")
    require(row["scenario"] == args.scenario, "scenario mismatch")
    require(row["model_version"] == config["model_version"], "model version mismatch")
    require(
        str(row["grid_energy_price_mode"]) == config["energy"]["grid_energy_price_mode"],
        "grid-energy price mode mismatch",
    )
    require(
        str(row["pv_capacity_mode"]) == config["energy"]["pv_capacity_mode"],
        "PV capacity mode mismatch",
    )
    baseline_net_peak = float(baseline["grid_import_peak_mw"])
    require(
        str(row["grid_capacity_upgrade_boundary"])
        == "no_ai_optimized_net_peak_zero_headroom_credit",
        "grid-capacity upgrade boundary mismatch",
    )
    require(
        close(float(row["no_ai_optimized_net_grid_peak_mw"]), baseline_net_peak),
        "reported no-AI optimized net peak does not match the baseline solve",
    )
    require(
        close(float(row["per_host_existing_grid_capacity_mw"]), baseline_net_peak),
        "AI scenario existing capacity must equal the optimized no-AI net grid peak",
    )
    expected_expansion = max(
        0.0,
        float(row["per_host_grid_import_peak_mw"]) - baseline_net_peak,
    )
    require(
        close(float(row["per_host_grid_expansion_mw"]), expected_expansion),
        "grid expansion must equal the positive AI net-peak increment over the no-AI optimized net peak",
    )
    grid_limit = config["energy"].get("grid_expansion_limit_mw")
    if grid_limit is not None:
        require(
            float(row["per_host_grid_expansion_mw"]) <= float(grid_limit) + 1e-6,
            "grid expansion exceeds the configured structural limit",
        )
    require(
        row["compute_efficiency_case"] == config["demand"]["effective_service"]["compute_efficiency_case"],
        "compute-efficiency case mismatch",
    )

    require(
        close(
            float(row["reconstructed_industry_daily_effective_service_units"]),
            float(row["industry_daily_effective_service_units"]),
        ),
        "equal-service reconstruction failed",
    )
    require(
        close(
            float(
                (
                    hourly["ai_executed_service_units"]
                    + hourly["ai_cloud_service_units"]
                ).sum()
            )
            / represented_days,
            float(row["per_host_daily_effective_service_units"]),
        ),
        "hourly service does not match scenario summary",
    )

    installed = hourly["installed_server_groups"].to_numpy(dtype=float)
    online = hourly["online_server_groups"].to_numpy(dtype=float)
    service = hourly["ai_executed_service_units"].to_numpy(dtype=float)
    cloud_service = hourly["ai_cloud_service_units"].to_numpy(dtype=float)
    require(np.all(cloud_service >= -1e-7), "cloud service is negative")
    require(
        float(row["per_host_cloud_service_share"])
        <= float(row["maximum_cloud_service_share"]) + 1e-7,
        "cloud service exceeds the configured maximum share",
    )
    execution = hourly["ai_compute_accelerator_h"].to_numpy(dtype=float)
    conversion = float(row["accelerator_h_per_service_unit"])
    require(np.allclose(execution, service * conversion), "service-to-compute conversion failed")
    require(np.all(installed >= -1e-7), "installed server count is negative")
    require(np.all(online >= -1e-7), "online server count is negative")
    require(np.all(online <= installed + 1e-6), "online servers exceed installed servers")
    if config["model"]["server_groups_integer"]:
        require(np.max(np.abs(installed - np.rint(installed))) < 1e-5, "installed servers are non-integer")
        require(np.max(np.abs(online - np.rint(online))) < 1e-5, "online servers are non-integer")
    accelerators = float(config["server"]["accelerators_per_server"])
    reserve = float(config["server"]["installed_reserve_fraction"])
    heterogeneous = str(row.get("compute_hardware_mode", "gpu_only")) == "heterogeneous_cpu_gpu"
    cloud_enabled = bool(row.get("hybrid_cloud_enabled", False))
    if not heterogeneous:
        require(np.all(execution <= accelerators * online + 1e-5), "online throughput constraint violated")
        average_groups = float(row["per_host_realized_average_required_gpu_server_groups"])
        installed_groups = float(row["per_host_installed_gpu_server_groups"])
        require(installed_groups + 1e-5 >= average_groups * (1.0 + reserve), "installed capacity is below average-demand planning headroom")
        if not cloud_enabled:
            capacity_floor = local_installed_capacity_floor(average_groups, reserve, int(config["server"]["n_plus_spare_server_groups"]))
            require(installed_groups + 1e-5 >= capacity_floor, "installed capacity does not satisfy average-demand headroom and N+spare")
        elif average_groups > 1e-9:
            require(installed_groups + 1e-5 >= average_groups + int(config["server"]["n_plus_spare_server_groups"]), "hybrid local capacity does not satisfy conditional N+spare")
    else:
        require(np.allclose(hourly["ai_facility_power_mw"], hourly["gpu_facility_power_mw"] + hourly["cpu_facility_power_mw"]), "heterogeneous hardware power components do not sum")
        for hardware in ("gpu", "cpu"):
            average_groups = float(row[f"per_host_realized_average_required_{hardware}_server_groups"])
            installed_groups = float(row[f"per_host_installed_{hardware}_server_groups"])
            hardware_reserve = reserve if hardware == "gpu" else float(config["compute_hardware"].get("cpu_server_overrides", {}).get("installed_reserve_fraction", reserve))
            require(installed_groups + 1e-5 >= average_groups * (1.0 + hardware_reserve), f"{hardware.upper()} installed capacity is below average-demand planning headroom")
            if cloud_enabled and average_groups > 1e-9:
                require(installed_groups + 1e-5 >= average_groups + int(config["server"]["n_plus_spare_server_groups"]), f"Hybrid local {hardware.upper()} capacity does not satisfy conditional N+spare")
        if not cloud_enabled:
            require(float(row["per_host_installed_gpu_server_groups"]) + 1e-5 >= float(row["minimum_installed_gpu_server_groups_from_capacity_rule"]), "GPU installed capacity is below its planning floor")
            require(float(row["per_host_installed_cpu_server_groups"]) + 1e-5 >= float(row["minimum_installed_cpu_server_groups_from_capacity_rule"]), "CPU installed capacity is below its planning floor")
    require(
        float(row["per_host_installed_server_groups"]) + 1e-7
        >= float(row["minimum_server_groups_for_model_state"]),
        "minimum model replica/VRAM constraint violated",
    )
    require(
        float(row["per_host_model_storage_available_gb"]) + 1e-7
        >= float(row["per_host_model_storage_required_gb"]),
        "model storage capacity constraint violated",
    )
    alignment = config["demand"]["external_energy_alignment"]
    ratio = float(row["external_energy_alignment_ratio"])
    require(np.isfinite(ratio) and ratio > 0.0, "external-alignment ratio is invalid")
    inside_industry_band = (
        float(alignment["warning_ratio_low"])
        <= ratio
        <= float(alignment["warning_ratio_high"])
    )
    if alignment.get("per_industry_check", "hard") == "hard":
        require(
            inside_industry_band,
            "derived reference electricity is outside the external-alignment band",
        )

    server = config["server"]
    pue = float(server["marginal_facility_multiplier"])
    maximum = float(server["maximum_wall_power_kw"])
    idle = float(server["online_idle_wall_power_kw"])
    standby = float(server["cold_spare_standby_power_kw"])
    require(
        close(float(row["server_maximum_wall_power_kw"]), maximum),
        "server maximum wall power does not match the active configuration",
    )
    if not heterogeneous:
        reconstructed_power = pue * (standby * installed + (idle - standby) * online + (maximum - idle) / accelerators * execution) / 1000.0
        require(np.max(np.abs(reconstructed_power - hourly["ai_facility_power_mw"].to_numpy(dtype=float))) < 1e-5, "AI facility power equation failed")
    else:
        require(close(float(row["industry_equivalent_incremental_annual_server_cost_rmb"]), float(row["industry_equivalent_incremental_annual_gpu_server_cost_rmb"]) + float(row["industry_equivalent_incremental_annual_cpu_server_cost_rmb"]), relative=1e-8), "GPU and CPU server costs do not sum to the physical-model server cost")
    roof_limit_mw = (
        float(row["roof_area_proxy_m2"])
        * float(config["factory"]["roof_usable_fraction"])
        * float(config["factory"]["pv_module_efficiency"])
        * float(config["factory"]["pv_realization_fraction"])
        / 1000.0
    )
    require(
        close(float(row["per_host_rooftop_pv_limit_mw"]), roof_limit_mw),
        "reported rooftop limit does not match the industry roof proxy",
    )
    require(float(row["per_host_rooftop_pv_capacity_mw"]) <= roof_limit_mw + 1e-7, "rooftop PV exceeds roof limit")
    if config["energy"]["pv_capacity_mode"] == "existing_rooftop_at_model_limit":
        require(
            close(
                float(row["per_host_rooftop_pv_capacity_mw"]),
                float(row["roof_area_proxy_m2"])
                * float(config["factory"]["roof_usable_fraction"])
                * float(config["factory"]["pv_module_efficiency"])
                * float(config["factory"]["pv_realization_fraction"])
                / 1000.0,
            ),
            "existing rooftop PV must equal the configured roof-limit boundary",
        )
    elif config["energy"]["pv_capacity_mode"] == "none":
        require(
            close(float(row["per_host_rooftop_pv_capacity_mw"]), 0.0),
            "no-DER core baseline must have zero rooftop PV capacity",
        )
    if not config["energy"]["battery_investment_enabled"] and config["energy"].get("battery_fixed_power_mw") is None:
        require(
            close(float(row["per_host_battery_power_mw"]), 0.0),
            "no-DER core baseline must have zero battery power",
        )
    prices = hourly["grid_energy_price_rmb_per_mwh"].to_numpy(dtype=float)
    require(np.isfinite(prices).all(), "grid-energy prices contain invalid values")
    if config["energy"]["grid_energy_price_mode"] == "guangdong_spot_retail_representative_week":
        require(float(np.ptp(prices)) > 0.0, "active spot-retail price vector is unexpectedly flat")

    multiplier = float(row["equivalent_host_multiplier"])
    for key in COST_COMPONENTS:
        unit = float(row[f"per_host_incremental_{key}"])
        aggregate = float(row[f"industry_equivalent_incremental_{key}"])
        require(close(aggregate, unit * multiplier, relative=1e-8), f"aggregation failed for {key}")
    component_sum = sum(float(row[f"industry_equivalent_incremental_{key}"]) for key in COST_COMPONENTS)
    require(close(component_sum, float(row["industry_equivalent_incremental_total_cost_rmb"])), "cost components do not sum to total")
    objective_delta = float(row["industry_equivalent_incremental_annual_objective_rmb"])
    require(close(component_sum, objective_delta, relative=1e-7, absolute=10.0), "incremental objective does not reconcile with components")

    payload = {
        "status": "validated",
        "model_version": config["model_version"],
        "industry_code": args.industry,
        "scenario": args.scenario,
        "external_alignment": {
            "mode": alignment.get("per_industry_check", "hard"),
            "ratio_to_industry_central_allocation": ratio,
            "inside_warning_band": inside_industry_band,
        },
        "checks": [
            "168-hour continuous representative week",
            "full-horizon rather than daily storage cycle",
            "configured flexible-deadline scenario",
            "equal-service reconstruction",
            "service-to-compute conversion and external-energy diagnostic",
            "configured continuous-or-integer server-group capacity",
            "minimum model replica/VRAM and bundled-storage capacity",
            "initialization, storage, and operations lifecycle costs",
            "facility-power identity",
            "joint CPU/GPU capacity, power, and server-cost identity",
            "configured existing-PV roof boundary",
            "configured hourly grid-energy price boundary",
            "optimized no-AI net-peak zero-headroom grid-capacity boundary",
            "host-to-industry aggregation",
            "incremental-cost reconciliation",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
