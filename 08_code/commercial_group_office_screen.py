"""Screen a commercial office group that can pool AI at one office building."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import china_minimum_prototype as core
from run_typical_manufacturing_base import baseline_peak_shaving_dispatch


ROOT = Path(__file__).resolve().parents[1]
PV_INPUT = ROOT / "04_cases" / "two_user_pv_battery_typical_day.csv"
CASE_OUTPUT = ROOT / "04_cases" / "commercial_office_typical_day.csv"
RESULT_OUTPUT = ROOT / "05_results" / "commercial_group_office_screen.csv"

EMPLOYEES = 800
ACTIVE_AI_USERS = 640
FLOOR_AREA_M2 = 16_000.0
ANNUAL_EUI_KWH_M2 = 88.5
WORKDAYS = 250
NONWORKDAYS = 115
NONWORKDAY_ENERGY_RATIO = 0.35
GROUP_BUILDINGS = 20

ROOF_AREA_M2 = 1_600.0
ROOF_USABLE_FRACTION = 0.65
MODULE_EFFICIENCY = 0.22
PV_REALIZATION_RATIO = 0.80
BATTERY_POWER_KW = 100.0
BATTERY_ENERGY_KWH = 200.0
BATTERY_ROUNDTRIP_EFFICIENCY = 0.90
CONNECTION_CAPACITY_KW = 630.0

GPUS_PER_SERVER = 2
TARGET_BUSY_FRACTION = 0.65
FULL_POWER_KW = 1.30
IDLE_POWER_KW = 0.42
PUE = 1.60

# High-integration knowledge-work scenario, not a forecast.
HUMAN_TASKS_DAY = ACTIVE_AI_USERS * 20
HUMAN_CALLS_PER_TASK = 2
WORKFLOW_TASKS_DAY = 2_500 * 0.40
WORKFLOW_CALLS_PER_TASK = 10
BATCH_TASKS_DAY = 200
BATCH_CALLS_PER_TASK = 15
INPUT_ALPHA = 0.15
L20_EQUIV_TOKENS_S = 80.0

HUMAN_HOURS = list(range(7, 21))
HUMAN_WEIGHTS = [0.02, 0.06, 0.10, 0.12, 0.10, 0.06, 0.09, 0.12, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02]
WORKFLOW_HOURS = list(range(8, 20))
WORKFLOW_WEIGHTS = [0.04, 0.08, 0.12, 0.12, 0.08, 0.10, 0.12, 0.12, 0.10, 0.07, 0.04, 0.01]
BATCH_HOURS = list(range(0, 6))
OFFICE_LOAD_SHAPE = [
    0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.30, 0.50,
    0.80, 1.00, 1.00, 1.00, 0.90, 1.00, 1.00, 1.00,
    1.00, 0.90, 0.70, 0.50, 0.40, 0.35, 0.30, 0.25,
]


def distribute(total: float, hours: list[int], weights: list[float]) -> list[float]:
    values = [0.0] * 24
    for hour, weight in zip(hours, weights):
        values[hour] = total * weight
    return values


def facility_power(gpu_load: float, servers: int) -> float:
    dynamic_per_gpu = (FULL_POWER_KW - IDLE_POWER_KW) / GPUS_PER_SERVER
    return (servers * IDLE_POWER_KW + gpu_load * dynamic_per_gpu) * PUE


def build_case() -> list[dict[str, float | int]]:
    weekday_kwh = FLOOR_AREA_M2 * ANNUAL_EUI_KWH_M2 / (
        WORKDAYS + NONWORKDAYS * NONWORKDAY_ENERGY_RATIO
    )
    load_scale = weekday_kwh / sum(OFFICE_LOAD_SHAPE)
    human = distribute(HUMAN_TASKS_DAY, HUMAN_HOURS, HUMAN_WEIGHTS)
    workflow = distribute(WORKFLOW_TASKS_DAY, WORKFLOW_HOURS, WORKFLOW_WEIGHTS)
    batch = distribute(
        BATCH_TASKS_DAY,
        BATCH_HOURS,
        [1.0 / len(BATCH_HOURS)] * len(BATCH_HOURS),
    )
    result: list[dict[str, float | int]] = []
    for hour in range(24):
        human_calls = human[hour] * HUMAN_CALLS_PER_TASK
        workflow_calls = workflow[hour] * WORKFLOW_CALLS_PER_TASK
        batch_calls = batch[hour] * BATCH_CALLS_PER_TASK
        calls = human_calls + workflow_calls + batch_calls
        input_tokens = (
            human_calls * 2_000
            + workflow_calls * 5_000
            + batch_calls * 8_000
        )
        output_tokens = (
            human_calls * 300
            + workflow_calls * 300
            + batch_calls * 500
        )
        gpu_h = (
            output_tokens + INPUT_ALPHA * input_tokens
        ) / (L20_EQUIV_TOKENS_S * 3_600.0)
        result.append(
            {
                "hour": hour,
                "office_base_load_kw": OFFICE_LOAD_SHAPE[hour] * load_scale,
                "human_tasks": human[hour],
                "workflow_tasks": workflow[hour],
                "batch_tasks": batch[hour],
                "llm_calls": calls,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "l20_gpu_compute_h": gpu_h,
            }
        )
    with CASE_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result[0].keys())
        writer.writeheader()
        writer.writerows(result)
    return result


def main() -> None:
    params = core.read_parameters()
    case = build_case()
    base = [float(row["office_base_load_kw"]) for row in case]
    gpu_one = [float(row["l20_gpu_compute_h"]) for row in case]

    pv_cap_kwp = (
        ROOF_AREA_M2
        * ROOF_USABLE_FRACTION
        * MODULE_EFFICIENCY
        * PV_REALIZATION_RATIO
    )
    with PV_INPUT.open(encoding="utf-8-sig", newline="") as handle:
        pv_source = [row for row in csv.DictReader(handle) if row["case"] == "office"]
    pv_source.sort(key=lambda row: int(row["hour"]))
    pv = [float(row["pv_kw"]) * pv_cap_kwp / 600.0 for row in pv_source]
    battery, _ = baseline_peak_shaving_dispatch(
        base,
        pv,
        BATTERY_POWER_KW,
        BATTERY_ENERGY_KWH,
        BATTERY_ROUNDTRIP_EFFICIENCY,
    )
    baseline_grid = [max(0.0, base[h] - pv[h] - battery[h]) for h in range(24)]

    server_capex = core.number(params, "L02")
    facility_capex_fraction = core.number(params, "L15")
    maintenance = core.number(params, "L14")
    crf = core.capital_recovery_factor(
        core.number(params, "U01"), core.number(params, "L13")
    )
    upfront_per_server = server_capex * (1.0 + facility_capex_fraction)
    annual_resource_per_server = upfront_per_server * crf + server_capex * maintenance
    demand_charge = core.number(params, "E05")
    cloud_monthly = core.number(params, "C17")
    cloud_hourly = core.number(params, "C18")
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
    headroom_servers = math.ceil(peak_one / (GPUS_PER_SERVER * TARGET_BUSY_FRACTION))
    production_servers = headroom_servers + 1
    results: list[dict[str, object]] = []

    def add_local(
        scenario: str,
        buildings: int,
        sites: int,
        servers_per_site: int,
        total_servers: int,
        gpu_profile: list[float],
    ) -> None:
        if sites == 1:
            power = [facility_power(value, total_servers) for value in gpu_profile]
            host_power = power
        else:
            host_power = [facility_power(value, servers_per_site) for value in gpu_one]
            power = [value * sites for value in host_power]
        host_grid_peak = max(baseline_grid[h] + host_power[h] for h in range(24))
        demand_increment = max(0.0, host_grid_peak - max(baseline_grid)) * sites
        idle_kw_total = total_servers * IDLE_POWER_KW * PUE
        dynamic_power = [power[h] - idle_kw_total for h in range(24)]
        annual_energy = (
            sum(idle_kw_total * prices[h] for h in range(24)) * 365.0
            + sum(dynamic_power[h] * prices[h] for h in range(24)) * WORKDAYS
        )
        annual_demand = demand_increment * demand_charge * 12.0
        annualized_capex = total_servers * upfront_per_server * crf
        annual_maintenance = total_servers * server_capex * maintenance
        direct = annualized_capex + annual_maintenance + annual_energy + annual_demand
        results.append(
            {
                "scenario": scenario,
                "buildings_served": buildings,
                "deployment_sites": sites,
                "servers_per_site_2xl20": servers_per_site,
                "total_servers_2xl20": total_servers,
                "upfront_local_investment_rmb": total_servers * upfront_per_server,
                "annualized_local_capex_rmb": annualized_capex,
                "annual_local_maintenance_rmb": annual_maintenance,
                "annual_local_energy_rmb": annual_energy,
                "annual_local_max_demand_rmb": annual_demand,
                "annual_enterprise_direct_rmb": direct,
                "annual_cloud_bill_rmb": 0.0,
                "typical_workday_ai_kwh": sum(power),
                "peak_host_ai_kw": max(host_power),
                "peak_host_grid_kw_existing_der": host_grid_peak,
                "host_connection_capacity_kw": CONNECTION_CAPACITY_KW,
                "host_expansion_screen_kw": max(0.0, host_grid_peak - CONNECTION_CAPACITY_KW),
                "peak_gpu_load": max(gpu_profile),
                "workday_installed_gpu_utilization": sum(gpu_profile)
                / (total_servers * GPUS_PER_SERVER * 24.0),
            }
        )

    add_local("single_minimum_capacity", 1, 1, minimum_servers, minimum_servers, gpu_one)
    add_local("single_operating_headroom", 1, 1, headroom_servers, headroom_servers, gpu_one)
    add_local("single_production_nplus1", 1, 1, production_servers, production_servers, gpu_one)

    def add_cloud(scenario: str, buildings: int, gpu_profile: list[float]) -> None:
        contract = math.ceil(max(gpu_profile) / (GPUS_PER_SERVER * TARGET_BUSY_FRACTION))
        hourly = [
            math.ceil(value / (GPUS_PER_SERVER * TARGET_BUSY_FRACTION))
            for value in gpu_profile
        ]
        monthly_bill = contract * cloud_monthly * 12.0
        ondemand_bill = sum(hourly) * cloud_hourly * WORKDAYS
        bill = min(monthly_bill, ondemand_bill)
        results.append(
            {
                "scenario": scenario,
                "buildings_served": buildings,
                "deployment_sites": 0,
                "servers_per_site_2xl20": 0,
                "total_servers_2xl20": 0,
                "upfront_local_investment_rmb": 0.0,
                "annualized_local_capex_rmb": 0.0,
                "annual_local_maintenance_rmb": 0.0,
                "annual_local_energy_rmb": 0.0,
                "annual_local_max_demand_rmb": 0.0,
                "annual_enterprise_direct_rmb": bill,
                "annual_cloud_bill_rmb": bill,
                "typical_workday_ai_kwh": 0.0,
                "peak_host_ai_kw": 0.0,
                "peak_host_grid_kw_existing_der": max(baseline_grid),
                "host_connection_capacity_kw": CONNECTION_CAPACITY_KW,
                "host_expansion_screen_kw": 0.0,
                "peak_gpu_load": max(gpu_profile),
                "workday_installed_gpu_utilization": "",
            }
        )

    add_cloud("single_public_cloud", 1, gpu_one)
    gpu_group = [value * GROUP_BUILDINGS for value in gpu_one]
    group_headroom = math.ceil(max(gpu_group) / (GPUS_PER_SERVER * TARGET_BUSY_FRACTION))
    group_production = group_headroom + 1
    add_local(
        "group_distributed_nplus1",
        GROUP_BUILDINGS,
        GROUP_BUILDINGS,
        production_servers,
        GROUP_BUILDINGS * production_servers,
        gpu_group,
    )
    add_local(
        "group_center_at_office_nplus1",
        GROUP_BUILDINGS,
        1,
        group_production,
        group_production,
        gpu_group,
    )
    add_cloud("group_public_cloud", GROUP_BUILDINGS, gpu_group)

    with RESULT_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"office_workday_kwh={sum(base):.2f}")
    print(f"office_peak_kw={max(base):.2f}")
    print(f"calls_day={sum(float(row['llm_calls']) for row in case):.0f}")
    print(f"gpu_h_day={sum(gpu_one):.2f}")
    print(f"peak_gpu={peak_one:.2f}")
    for result in results:
        print(
            result["scenario"],
            result["total_servers_2xl20"],
            round(float(result["annual_enterprise_direct_rmb"]), 2),
            round(float(result["peak_host_ai_kw"]), 2),
            round(float(result["host_expansion_screen_kw"]), 2),
        )


if __name__ == "__main__":
    main()
