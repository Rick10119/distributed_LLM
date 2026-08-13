"""Minimum 24-hour China prototype for local, cloud, and hybrid AI.

The existing AI kW curves are interpreted as full-load-equivalent IT task
demand.  This is a synthetic scaling convention, not a measured L20 workload.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_PROFILE = ROOT / "04_cases" / "two_user_typical_day.csv"
PARAMETERS = ROOT / "02_data" / "china_enterprise_ai_cost_parameters.csv"
LOAD_OUTPUT = ROOT / "05_results" / "china_prototype_load_profiles.csv"
SUMMARY_OUTPUT = ROOT / "05_results" / "china_prototype_screen_summary.csv"

CASES = {
    "steel": {
        "base_col": "steel_base_kw",
        "inference_col": "steel_inference_kw",
        "shiftable_col": "steel_shifted_finetune_kw",
        "tight_capacity_kw": 630.0,
        "spare_capacity_kw": 650.0,
    },
    "office": {
        "base_col": "office_base_kw",
        "inference_col": "office_inference_kw",
        "shiftable_col": "office_shifted_finetune_kw",
        "tight_capacity_kw": 1000.0,
        "spare_capacity_kw": 1250.0,
    },
}

MODES = {
    "local": 1.0,
    "cloud": 0.0,
    "hybrid_50": 0.5,
}


def read_parameters() -> dict[str, dict[str, str]]:
    with PARAMETERS.open(encoding="utf-8-sig", newline="") as handle:
        return {row["parameter_id"]: row for row in csv.DictReader(handle)}


def number(params: dict[str, dict[str, str]], key: str) -> float:
    return float(params[key]["base_value"])


def read_profiles() -> list[dict[str, str]]:
    with INPUT_PROFILE.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def capital_recovery_factor(rate: float, life: float) -> float:
    return rate * (1.0 + rate) ** life / ((1.0 + rate) ** life - 1.0)


def tou_price(hour: int, valley: float, flat: float, peak: float) -> float:
    if 0 <= hour < 8:
        return valley
    if 10 <= hour < 12 or 14 <= hour < 19:
        return peak
    return flat


def local_server_count(
    peak_task_kw: float,
    full_power_kw: float,
    target_utilization: float,
    reserve_fraction: float,
) -> int:
    if peak_task_kw <= 0:
        return 0
    return math.ceil(
        peak_task_kw
        * (1.0 + reserve_fraction)
        / (full_power_kw * target_utilization)
    )


def cloud_equivalent_servers(
    peak_task_kw: float, full_power_kw: float, target_utilization: float
) -> float:
    if peak_task_kw <= 0:
        return 0.0
    return peak_task_kw / (full_power_kw * target_utilization)


def facility_load(
    task_kw: float,
    equivalent_servers: float,
    idle_power_kw: float,
    full_power_kw: float,
    pue: float,
) -> float:
    if equivalent_servers <= 0:
        return 0.0
    dynamic_fraction = (full_power_kw - idle_power_kw) / full_power_kw
    it_power_kw = equivalent_servers * idle_power_kw + task_kw * dynamic_fraction
    return it_power_kw * pue


def main() -> None:
    params = read_parameters()
    source = read_profiles()

    full_power = number(params, "L09")
    idle_power = number(params, "L10")
    local_utilization = number(params, "L12")
    local_pue = number(params, "L17")
    reserve = number(params, "L16")
    cloud_pue = number(params, "U02")
    cloud_utilization = number(params, "U03")
    days = number(params, "U04")
    discount_rate = number(params, "U01")
    life = number(params, "L13")
    server_capex = number(params, "L02")
    maintenance_fraction = number(params, "L14")
    facility_capex_fraction = number(params, "L15")
    demand_charge = number(params, "E05")
    valley_price = number(params, "E02")
    flat_price = number(params, "E01")
    peak_price = number(params, "E03")
    grid_capex_rmb_per_kw = number(params, "X01") * 10_000.0 / 1_000.0
    cloud_monthly_price = number(params, "C17")
    cloud_hourly_price = number(params, "C18")

    crf = capital_recovery_factor(discount_rate, life)
    load_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for case, case_info in CASES.items():
        base = [float(row[case_info["base_col"]]) for row in source]
        inference = [float(row[case_info["inference_col"]]) for row in source]
        shiftable = [float(row[case_info["shiftable_col"]]) for row in source]
        task = [a + b for a, b in zip(inference, shiftable)]
        base_peak = max(base)

        for mode, local_share in MODES.items():
            local_task = [value * local_share for value in task]
            cloud_task = [value * (1.0 - local_share) for value in task]
            local_peak_task = max(local_task)
            cloud_peak_task = max(cloud_task)

            local_servers = local_server_count(
                local_peak_task, full_power, local_utilization, reserve
            )
            cloud_physical_servers = cloud_equivalent_servers(
                cloud_peak_task, full_power, cloud_utilization
            )
            cloud_contract_instances = (
                math.ceil(cloud_physical_servers) if cloud_physical_servers > 0 else 0
            )

            local_facility = [
                facility_load(
                    value,
                    local_servers,
                    idle_power,
                    full_power,
                    local_pue,
                )
                for value in local_task
            ]
            cloud_facility = [
                facility_load(
                    value,
                    cloud_physical_servers,
                    idle_power,
                    full_power,
                    cloud_pue,
                )
                for value in cloud_task
            ]
            enterprise_total = [a + b for a, b in zip(base, local_facility)]

            for index, row in enumerate(source):
                load_rows.append(
                    {
                        "case": case,
                        "mode": mode,
                        "hour": int(row["hour"]),
                        "base_load_kw": round(base[index], 6),
                        "ai_task_fle_it_kw": round(task[index], 6),
                        "local_ai_facility_kw": round(local_facility[index], 6),
                        "cloud_ai_facility_kw": round(cloud_facility[index], 6),
                        "enterprise_total_kw": round(enterprise_total[index], 6),
                    }
                )

            local_energy_day = sum(local_facility)
            cloud_energy_day = sum(cloud_facility)
            enterprise_peak = max(enterprise_total)
            incremental_demand = max(0.0, enterprise_peak - base_peak)
            energy_cost_day = sum(
                value
                * tou_price(hour, valley_price, flat_price, peak_price)
                for hour, value in enumerate(local_facility)
            )
            annual_local_energy_cost = energy_cost_day * days
            annual_demand_cost = incremental_demand * demand_charge * 12.0
            annualized_local_capex = (
                local_servers
                * server_capex
                * (1.0 + facility_capex_fraction)
                * crf
            )
            annual_local_maintenance = (
                local_servers * server_capex * maintenance_fraction
            )
            annual_cloud_monthly = cloud_contract_instances * cloud_monthly_price * 12.0

            hourly_instances = [
                math.ceil(value / (full_power * cloud_utilization)) if value > 0 else 0
                for value in cloud_task
            ]
            annual_cloud_ondemand = sum(hourly_instances) * cloud_hourly_price * days
            annual_direct_monthly = (
                annualized_local_capex
                + annual_local_maintenance
                + annual_local_energy_cost
                + annual_demand_cost
                + annual_cloud_monthly
            )
            annual_direct_ondemand = (
                annualized_local_capex
                + annual_local_maintenance
                + annual_local_energy_cost
                + annual_demand_cost
                + annual_cloud_ondemand
            )

            tight_expansion = max(
                0.0, enterprise_peak - float(case_info["tight_capacity_kw"])
            )
            spare_expansion = max(
                0.0, enterprise_peak - float(case_info["spare_capacity_kw"])
            )
            cloud_expansion = max(cloud_facility)
            total_tight_expansion = tight_expansion + cloud_expansion
            total_spare_expansion = spare_expansion + cloud_expansion

            summary_rows.append(
                {
                    "case": case,
                    "mode": mode,
                    "local_share": local_share,
                    "local_servers": local_servers,
                    "cloud_physical_equivalent_servers": round(cloud_physical_servers, 6),
                    "cloud_contract_instances": cloud_contract_instances,
                    "local_ai_energy_kwh_day": round(local_energy_day, 6),
                    "cloud_ai_energy_kwh_day": round(cloud_energy_day, 6),
                    "enterprise_peak_kw": round(enterprise_peak, 6),
                    "incremental_max_demand_kw": round(incremental_demand, 6),
                    "cloud_dc_incremental_peak_kw": round(max(cloud_facility), 6),
                    "enterprise_tight_expansion_kw": round(tight_expansion, 6),
                    "enterprise_spare_expansion_kw": round(spare_expansion, 6),
                    "cloud_dc_expansion_kw": round(cloud_expansion, 6),
                    "total_grid_expansion_tight_kw": round(
                        total_tight_expansion, 6
                    ),
                    "total_grid_expansion_spare_kw": round(
                        total_spare_expansion, 6
                    ),
                    "enterprise_tight_grid_capex_rmb": round(
                        tight_expansion * grid_capex_rmb_per_kw, 2
                    ),
                    "enterprise_spare_grid_capex_rmb": round(
                        spare_expansion * grid_capex_rmb_per_kw, 2
                    ),
                    "cloud_dc_grid_capex_rmb": round(
                        cloud_expansion * grid_capex_rmb_per_kw, 2
                    ),
                    "total_grid_capex_tight_rmb": round(
                        total_tight_expansion * grid_capex_rmb_per_kw, 2
                    ),
                    "total_grid_capex_spare_rmb": round(
                        total_spare_expansion * grid_capex_rmb_per_kw, 2
                    ),
                    "annualized_local_capex_rmb": round(annualized_local_capex, 2),
                    "annual_local_maintenance_rmb": round(
                        annual_local_maintenance, 2
                    ),
                    "annual_local_energy_rmb": round(annual_local_energy_cost, 2),
                    "annual_incremental_demand_rmb": round(annual_demand_cost, 2),
                    "annual_cloud_monthly_rmb": round(annual_cloud_monthly, 2),
                    "annual_cloud_ondemand_rmb": round(annual_cloud_ondemand, 2),
                    "annual_enterprise_direct_monthly_rmb": round(
                        annual_direct_monthly, 2
                    ),
                    "annual_enterprise_direct_ondemand_rmb": round(
                        annual_direct_ondemand, 2
                    ),
                }
            )

    with LOAD_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=load_rows[0].keys())
        writer.writeheader()
        writer.writerows(load_rows)

    with SUMMARY_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
