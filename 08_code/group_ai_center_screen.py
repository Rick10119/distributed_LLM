"""Compare single-site indivisibility and a 20-factory group AI centre.

The calculation uses the extended manufacturing workload.  Ordinary edge CV
is excluded because it remains at each production line under every architecture.
All central workloads are assumed synchronous in this first conservative screen.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import china_minimum_prototype as core


ROOT = Path(__file__).resolve().parents[1]
EXTENDED = ROOT / "05_results" / "manufacturing_ai_extended_load_screen.csv"
BASE_PROFILE = ROOT / "05_results" / "typical_manufacturing_hourly_profiles.csv"
OUTPUT = ROOT / "05_results" / "group_ai_center_screen.csv"

GROUP_FACTORIES = 20
GPUS_PER_SERVER = 2
TARGET_BUSY_FRACTION = 0.65
FULL_POWER_KW = 1.30
IDLE_POWER_KW = 0.42
LOCAL_PUE = 1.60
CENTRAL_PUE = 1.60  # Hold PUE fixed so the result isolates pooling and scale.


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def facility_power(gpu_load: float, servers: int, pue: float) -> float:
    dynamic_per_gpu = (FULL_POWER_KW - IDLE_POWER_KW) / GPUS_PER_SERVER
    return (servers * IDLE_POWER_KW + gpu_load * dynamic_per_gpu) * pue


def main() -> None:
    params = core.read_parameters()
    workload = rows(EXTENDED)
    gpu_one = [float(row["extended_central_gpu_h"]) for row in workload]
    profile = [row for row in rows(BASE_PROFILE) if row["mode"] == "local"]
    profile.sort(key=lambda row: int(row["hour"]))
    baseline_grid = [float(row["baseline_grid_existing_der_kw"]) for row in profile]

    server_capex = core.number(params, "L02")
    facility_capex_fraction = core.number(params, "L15")
    maintenance = core.number(params, "L14")
    discount = core.number(params, "U01")
    life = core.number(params, "L13")
    days = core.number(params, "U04")
    demand_charge = core.number(params, "E05")
    cloud_monthly = core.number(params, "C17")
    cloud_hourly = core.number(params, "C18")
    crf = core.capital_recovery_factor(discount, life)
    annual_resource_per_server = (
        server_capex * (1.0 + facility_capex_fraction) * crf
        + server_capex * maintenance
    )
    upfront_per_server = server_capex * (1.0 + facility_capex_fraction)
    prices = [
        core.tou_price(
            hour,
            core.number(params, "E02"),
            core.number(params, "E01"),
            core.number(params, "E03"),
        )
        for hour in range(24)
    ]

    peak_one = max(gpu_one)
    minimum_servers = math.ceil(peak_one / GPUS_PER_SERVER)
    headroom_servers = math.ceil(
        peak_one / (GPUS_PER_SERVER * TARGET_BUSY_FRACTION)
    )
    production_servers = headroom_servers + 1

    scenarios: list[dict[str, object]] = []

    def add_local(
        name: str,
        factories_served: int,
        sites: int,
        servers_per_site: int,
        total_servers: int,
        gpu_profile: list[float],
        pue: float,
    ) -> None:
        if sites == 1:
            power = [facility_power(g, total_servers, pue) for g in gpu_profile]
            host_grid_peak = max(
                baseline_grid[h] + power[h] for h in range(24)
            )
            demand_increment = max(
                0.0,
                host_grid_peak - max(baseline_grid),
            )
        else:
            one_power = [facility_power(g, servers_per_site, pue) for g in gpu_one]
            power = [value * sites for value in one_power]
            host_grid_peak = max(
                baseline_grid[h] + one_power[h] for h in range(24)
            )
            one_increment = max(
                0.0,
                max(baseline_grid[h] + one_power[h] for h in range(24))
                - max(baseline_grid),
            )
            demand_increment = one_increment * sites
        annual_energy = sum(power[h] * prices[h] for h in range(24)) * days
        annual_demand = demand_increment * demand_charge * 12.0
        scenarios.append(
            {
                "scenario": name,
                "factories_served": factories_served,
                "deployment_sites": sites,
                "servers_per_site_2xl20": servers_per_site,
                "total_servers_2xl20": total_servers,
                "upfront_local_investment_rmb": total_servers * upfront_per_server,
                "annualized_compute_resource_rmb": total_servers * annual_resource_per_server,
                "annual_local_energy_rmb": annual_energy,
                "annual_incremental_demand_charge_rmb": annual_demand,
                "annual_cloud_bill_rmb": 0.0,
                "annual_enterprise_direct_rmb": total_servers * annual_resource_per_server
                + annual_energy
                + annual_demand,
                "daily_central_facility_kwh": sum(power),
                "peak_central_facility_kw": max(power),
                "peak_host_site_ai_kw": max(power) if sites == 1 else max(power) / sites,
                "peak_host_site_grid_kw": host_grid_peak,
                "peak_gpu_load": max(gpu_profile),
                "installed_gpu_average_utilization": sum(gpu_profile)
                / (total_servers * GPUS_PER_SERVER * 24.0),
                "sizing_rule": "integer minimum"
                if name == "single_minimum_capacity"
                else (
                    "65% peak busy ceiling"
                    if name == "single_operating_headroom"
                    else "65% peak busy ceiling plus one spare per site"
                ),
            }
        )

    add_local(
        "single_minimum_capacity",
        1,
        1,
        minimum_servers,
        minimum_servers,
        gpu_one,
        LOCAL_PUE,
    )
    add_local(
        "single_operating_headroom",
        1,
        1,
        headroom_servers,
        headroom_servers,
        gpu_one,
        LOCAL_PUE,
    )
    add_local(
        "single_production_nplus1",
        1,
        1,
        production_servers,
        production_servers,
        gpu_one,
        LOCAL_PUE,
    )

    single_cloud_instances_hourly = [
        math.ceil(value / (GPUS_PER_SERVER * TARGET_BUSY_FRACTION))
        for value in gpu_one
    ]
    single_cloud_contract = headroom_servers
    single_cloud_monthly = single_cloud_contract * cloud_monthly * 12.0
    single_cloud_ondemand = (
        sum(single_cloud_instances_hourly) * cloud_hourly * days
    )
    single_cloud_bill = min(single_cloud_monthly, single_cloud_ondemand)
    scenarios.append(
        {
            "scenario": "single_public_cloud",
            "factories_served": 1,
            "deployment_sites": 0,
            "servers_per_site_2xl20": 0,
            "total_servers_2xl20": 0,
            "upfront_local_investment_rmb": 0.0,
            "annualized_compute_resource_rmb": 0.0,
            "annual_local_energy_rmb": 0.0,
            "annual_incremental_demand_charge_rmb": 0.0,
            "annual_cloud_bill_rmb": single_cloud_bill,
            "annual_enterprise_direct_rmb": single_cloud_bill,
            "daily_central_facility_kwh": 0.0,
            "peak_central_facility_kw": 0.0,
            "peak_host_site_ai_kw": 0.0,
            "peak_host_site_grid_kw": max(baseline_grid),
            "peak_gpu_load": peak_one,
            "installed_gpu_average_utilization": "",
            "sizing_rule": f"best of {single_cloud_contract} monthly instances and hourly autoscaling",
        }
    )

    gpu_group = [value * GROUP_FACTORIES for value in gpu_one]
    group_headroom_servers = math.ceil(
        max(gpu_group) / (GPUS_PER_SERVER * TARGET_BUSY_FRACTION)
    )
    group_production_servers = group_headroom_servers + 1
    add_local(
        "group_distributed_nplus1",
        GROUP_FACTORIES,
        GROUP_FACTORIES,
        production_servers,
        GROUP_FACTORIES * production_servers,
        gpu_group,
        LOCAL_PUE,
    )
    add_local(
        "group_center_at_host_factory_nplus1",
        GROUP_FACTORIES,
        1,
        group_production_servers,
        group_production_servers,
        gpu_group,
        CENTRAL_PUE,
    )

    cloud_instances_hourly = [
        math.ceil(value / (GPUS_PER_SERVER * TARGET_BUSY_FRACTION))
        for value in gpu_group
    ]
    cloud_contract = group_headroom_servers
    cloud_bill_monthly = cloud_contract * cloud_monthly * 12.0
    cloud_bill_ondemand = sum(cloud_instances_hourly) * cloud_hourly * days
    cloud_bill = min(cloud_bill_monthly, cloud_bill_ondemand)
    scenarios.append(
        {
            "scenario": "group_public_cloud",
            "factories_served": GROUP_FACTORIES,
            "deployment_sites": 0,
            "servers_per_site_2xl20": 0,
            "total_servers_2xl20": 0,
            "upfront_local_investment_rmb": 0.0,
            "annualized_compute_resource_rmb": 0.0,
            "annual_local_energy_rmb": 0.0,
            "annual_incremental_demand_charge_rmb": 0.0,
            "annual_cloud_bill_rmb": cloud_bill,
            "annual_enterprise_direct_rmb": cloud_bill,
            "daily_central_facility_kwh": 0.0,
            "peak_central_facility_kw": 0.0,
            "peak_host_site_ai_kw": 0.0,
            "peak_host_site_grid_kw": max(baseline_grid),
            "peak_gpu_load": max(gpu_group),
            "installed_gpu_average_utilization": "",
            "sizing_rule": f"best of {cloud_contract} monthly instances and hourly autoscaling",
        }
    )

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scenarios[0].keys())
        writer.writeheader()
        writer.writerows(scenarios)

    for scenario in scenarios:
        print(
            scenario["scenario"],
            scenario["total_servers_2xl20"],
            round(float(scenario["annual_enterprise_direct_rmb"]), 2),
            round(float(scenario["peak_host_site_ai_kw"]), 2),
        )


if __name__ == "__main__":
    main()
