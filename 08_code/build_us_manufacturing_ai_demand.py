#!/usr/bin/env python3
"""Build bottom-up U.S. manufacturing six-task AI demand and costs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("office", "agent", "vision", "maintenance", "scheduling", "simulation")
TASK_NAMES = {
    "office": "Office knowledge/RAG/Copilot",
    "agent": "Business-process agent",
    "vision": "Vision anomaly review",
    "maintenance": "Predictive maintenance",
    "scheduling": "Production scheduling",
    "simulation": "R&D simulation/digital twin",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_value(frame: pd.DataFrame, parameter_id: str, case: str) -> float:
    rows = frame[frame["parameter_id"] == parameter_id]
    if len(rows) != 1:
        raise ValueError(f"Expected one parameter row for {parameter_id}")
    value = float(rows.iloc[0][f"{case}_value"])
    if value <= 0:
        raise ValueError(f"Non-positive parameter {parameter_id}/{case}")
    return value


def capital_recovery_factor(rate: float, years: float) -> float:
    factor = (1 + rate) ** years
    return rate * factor / (factor - 1)


def storage_cost(storage_gb: float, settings: dict) -> float:
    first_tier = float(settings["storage_first_tier_gb"])
    first = min(storage_gb, first_tier)
    second = max(storage_gb - first_tier, 0.0)
    return 12 * (
        first * float(settings["storage_first_tier_usd_per_gb_month"])
        + second * float(settings["storage_second_tier_usd_per_gb_month"])
    )


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    maximum = float(values.max())
    if values.shape != (24,) or maximum <= 0:
        raise ValueError("Task shape must contain 24 values with positive maximum")
    return values / maximum


def task_shapes(path: Path, settings: dict) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame[
        (frame["record_type"] == settings["profile_record_type"])
        & (frame["scenario"] == settings["profile_scenario"])
    ].sort_values("hour")
    if len(frame) != 24:
        raise ValueError("Transferred workload profile must have 24 hourly rows")

    def extract(name: str) -> np.ndarray:
        pattern = re.compile(rf"(?:^|; )?{re.escape(name)}=([0-9.]+)")
        result = []
        for text in frame["derivation_or_definition"].astype(str):
            match = pattern.search(text)
            result.append(float(match.group(1)) if match else 0.0)
        return np.asarray(result)

    office = normalized(extract("human_tasks") + 0.05)
    agent = normalized(extract("transaction_tasks") + extract("batch_tasks") + 0.10)
    industrial = normalized(extract("machine_ai_tasks") + extract("vlm_escalations") + 0.05)
    scheduling = np.full(24, 0.20)
    for hour, value in {5: 1.2, 6: 1.6, 7: 1.0, 13: 1.0, 14: 1.5, 15: 1.0, 21: 1.0, 22: 1.5, 23: 1.2}.items():
        scheduling[hour] = value
    simulation = np.full(24, 0.10)
    simulation[[0, 1, 2, 3, 4, 5, 22, 23]] = 1.0
    return {
        "office": office,
        "agent": agent,
        "vision": industrial,
        "maintenance": normalized(0.75 * industrial + 0.25),
        "scheduling": normalized(scheduling),
        "simulation": normalized(simulation),
    }


def allocate_daily_to_shape(daily: float, shape: np.ndarray) -> np.ndarray:
    if daily < 0:
        raise ValueError("Daily demand cannot be negative")
    return daily * shape / shape.sum()


def physical_load(
    task_summary: pd.DataFrame,
    shapes: dict[str, np.ndarray],
    accelerator_h_per_service: float,
    compute: dict,
    gpu_tasks: list[str],
) -> dict[str, object]:
    """Convert a national daily task mix to pooled capacity and electricity."""
    hourly_service = np.zeros(24)
    hourly_accel = np.zeros(24)
    hourly_gpu_accel = np.zeros(24)
    for row in task_summary.itertuples(index=False):
        profile = allocate_daily_to_shape(float(row.effective_service_units_day), shapes[row.task_id])
        hourly_service += profile
        hourly_accel += profile * accelerator_h_per_service
        if row.task_id in gpu_tasks:
            hourly_gpu_accel += profile * accelerator_h_per_service
    utilization = float(compute["installed_utilization"])
    reserve = float(compute["installed_reserve_fraction"])
    accelerators_per_server = float(compute["accelerators_per_server"])
    idle_kw = float(compute["server_idle_power_kw"])
    max_kw = float(compute["server_maximum_power_kw"])
    pue = float(compute["marginal_facility_multiplier"])
    dynamic_kw_per_accelerator = (max_kw - idle_kw) / accelerators_per_server
    required_without_reserve = float(hourly_accel.max()) / (accelerators_per_server * utilization)
    installed_servers = required_without_reserve * (1 + reserve)
    required_online_servers = np.maximum(
        hourly_accel / (accelerators_per_server * utilization), 1.0
    )
    active_servers = np.minimum(required_online_servers, required_without_reserve)
    reserve_servers = installed_servers - required_without_reserve
    facility_kw = pue * (
        active_servers * idle_kw
        + hourly_accel * dynamic_kw_per_accelerator
        + reserve_servers * float(compute.get("cold_spare_standby_power_kw", 0.02))
    )
    return {
        "hourly_service": hourly_service,
        "hourly_accel": hourly_accel,
        "hourly_gpu_accel": hourly_gpu_accel,
        "required_without_reserve": required_without_reserve,
        "installed_servers": installed_servers,
        "annual_energy_twh": facility_kw.sum() / 24 * 8760 / 1e9,
    }


def build(config: dict, output_dir: Path) -> dict[str, object]:
    activity_path = ROOT / "02_data/raw/curated/us_manufacturing_activity_naics3_2022.csv"
    mecs_path = ROOT / "02_data/raw/curated/us_manufacturing_mecs_2022.csv"
    berd_path = ROOT / "02_data/raw/curated/us_manufacturing_berd_2023.csv"
    task_parameter_path = ROOT / "02_data/processed/us_demand/us_task_driver_parameters_v0.1.csv"
    efficiency_path = ROOT / "02_data/processed/compute_efficiency/accelerator_h_per_service_unit.csv"
    activity = pd.read_csv(activity_path, encoding="utf-8-sig")
    activity = activity[activity["size_class_official"].str.startswith("EMPSZFE 001:")].copy()
    if len(activity) != 21 or activity["naics3"].nunique() != 21:
        raise ValueError("Activity input must have 21 NAICS3 total rows")
    mecs = pd.read_csv(mecs_path, encoding="utf-8-sig")
    berd = pd.read_csv(berd_path, encoding="utf-8-sig")
    task_parameters = pd.read_csv(task_parameter_path, encoding="utf-8-sig").set_index("task")
    if set(task_parameters.index) != set(TASKS):
        raise ValueError("Task parameter input must contain six tasks")
    efficiency = pd.read_csv(efficiency_path, encoding="utf-8-sig")
    selected_efficiency = efficiency[efficiency["efficiency_case"] == config["compute"]["efficiency_case"]]
    if len(selected_efficiency) != 1:
        raise ValueError("Compute efficiency case missing or duplicated")
    accelerator_h_per_service = float(selected_efficiency.iloc[0]["accelerator_h_per_service_unit"])

    mecs_valid = mecs.dropna(subset=["establishments", "purchased_electricity_mwh"]).copy()
    mecs_average = mecs_valid["purchased_electricity_mwh"].sum() / mecs_valid["establishments"].sum()
    mecs_valid["equipment_index"] = mecs_valid["electricity_mwh_per_establishment"] / mecs_average
    equipment = {
        str(int(row.naics_or_group)): float(row.equipment_index)
        for row in mecs_valid.itertuples(index=False)
    }
    # NAICS 337 purchased electricity is quality-suppressed in MECS. The
    # transparent neutral fallback preserves the nationwide weighted-average
    # intensity instead of treating the suppressed value as zero.
    equipment_proxy_status = {code: "observed_MECS" for code in equipment}

    berd_total = berd[berd["naics_or_group"] == "31-33"]
    if len(berd_total) != 1:
        raise ValueError("BERD manufacturing total missing")
    berd_industry = berd[berd["naics_or_group"] != "31-33"].copy()
    berd_emp = {
        str(row.naics_or_group): float(row.domestic_rd_employment)
        for row in berd_industry.itertuples(index=False)
    }
    combined_313_316 = float(berd_emp["313-16"])
    combined_codes = [313, 314, 315, 316]
    combined_activity = activity[activity["naics3"].isin(combined_codes)][["naics3", "employment"]].copy()
    combined_activity["share"] = combined_activity["employment"] / combined_activity["employment"].sum()
    for row in combined_activity.itertuples(index=False):
        berd_emp[str(row.naics3)] = combined_313_316 * float(row.share)

    records = []
    annual_days = float(config["annualization_days"])
    for case in config["parameter_cases"]:
        for row in activity.sort_values("naics3").itertuples(index=False):
            naics = str(row.naics3)
            for task in TASKS:
                adoption = float(task_parameters.loc[task, f"adoption_2030_{case}"])
                unit_service = float(config["effective_service_units_per_workload_unit"][task])
                workload_units = float(config["workload_units_per_active_driver_day"][case][task])
                if task in {"office", "agent"}:
                    driver = float(row.employment)
                    driver_unit = "employees"
                    applicability = 1.0
                elif task == "simulation":
                    driver = float(berd_emp[naics])
                    driver_unit = "domestic_R&D_employees"
                    applicability = 1.0
                else:
                    driver = float(row.establishments)
                    driver_unit = "establishments"
                    applicability = float(equipment.get(naics, 1.0)) if task == "maintenance" else 1.0
                service = driver * adoption * workload_units * unit_service * applicability
                records.append({
                    "demand_case_version": config["demand_case_version"],
                    "model_version": config["model_version"],
                    "country": config["country"],
                    "year": config["reference_year"],
                    "parameter_case": case,
                    "naics3": naics,
                    "industry_name": row.industry_name,
                    "task_id": task,
                    "task_name": TASK_NAMES[task],
                    "driver": config["drivers"][task],
                    "driver_value": driver,
                    "driver_unit": driver_unit,
                    "adoption_fraction": adoption,
                    "coverage_fraction": 1.0,
                    "industry_applicability": applicability,
                    "workload_unit_definition": config["workload_unit_definition"][task],
                    "workload_units_per_active_driver_day": workload_units,
                    "effective_service_units_per_workload_unit": unit_service,
                    "effective_service_units_day": service,
                    "annual_effective_service_units": service * annual_days,
                    "service_unit": "reference_L20_equivalent_accelerator_hour",
                    "adoption_evidence_grade": task_parameters.loc[task, "evidence_grade"],
                    "intensity_evidence_status": "bottom_up_operational_scenario_not_US_observation",
                    "industry_applicability_status": (
                        equipment_proxy_status.get(naics, "neutral_1_for_suppressed_MECS")
                        if task == "maintenance" else "not_applicable_or_equal_weight"
                    ),
                })
    detail = pd.DataFrame(records)
    audit_rows = []
    for (case, task), group in detail.groupby(["parameter_case", "task_id"], sort=True):
        active_driver_equivalents = float(
            (group["driver_value"] * group["adoption_fraction"] * group["industry_applicability"]).sum()
        )
        audit_rows.append({
            "parameter_case": case,
            "task_id": task,
            "driver": config["drivers"][task],
            "active_driver_equivalents": active_driver_equivalents,
            "workload_unit_definition": config["workload_unit_definition"][task],
            "workload_units_per_active_driver_day": float(
                config["workload_units_per_active_driver_day"][case][task]
            ),
            "effective_service_units_per_workload_unit": float(
                config["effective_service_units_per_workload_unit"][task]
            ),
            "effective_service_units_day": float(group["effective_service_units_day"].sum()),
            "formula": "active_driver_equivalents * workload_units_per_active_driver_day * effective_service_units_per_workload_unit",
            "parameter_evidence_status": "operational_scenario_requires_manufacturer_logs",
        })
    parameter_audit = pd.DataFrame(audit_rows)
    parameter_audit["share_of_case_effective_service"] = parameter_audit["effective_service_units_day"] / parameter_audit.groupby(
        "parameter_case"
    )["effective_service_units_day"].transform("sum")

    shapes = task_shapes(ROOT / config["task_shapes"]["profile_file"], config["task_shapes"])

    task_summary = detail.groupby(["parameter_case", "task_id", "task_name"], as_index=False).agg(
        effective_service_units_day=("effective_service_units_day", "sum"),
        annual_effective_service_units=("annual_effective_service_units", "sum"),
    )
    token_settings = config["token_parameters"]
    task_summary["annual_input_tokens"] = 0.0
    task_summary["annual_output_tokens"] = 0.0
    for task in ("office", "agent"):
        selected = task_summary["task_id"] == task
        for case in config["parameter_cases"]:
            row_mask = selected & (task_summary["parameter_case"] == case)
            detail_mask = (detail["parameter_case"] == case) & (detail["task_id"] == task)
            active_employee_days = float(
                (detail.loc[detail_mask, "driver_value"] * detail.loc[detail_mask, "adoption_fraction"]
                 * detail.loc[detail_mask, "coverage_fraction"]).sum()
            ) * annual_days
            business_tasks = active_employee_days * float(
                token_settings[task]["business_tasks_per_active_employee_day"]
            ) * float(token_settings["task_intensity_multiplier_by_case"][case])
            calls = business_tasks * float(token_settings[task]["calls_per_business_task"])
            task_summary.loc[row_mask, "annual_input_tokens"] = calls * float(
                token_settings[task]["input_tokens_per_call"]
            )
            task_summary.loc[row_mask, "annual_output_tokens"] = calls * float(
                token_settings[task]["output_tokens_per_call"]
            )
    task_summary["token_boundary"] = np.where(
        task_summary["task_id"].isin(["office", "agent"]),
        "API_tokens",
        "reserved_GPU_no_token_conversion",
    )

    compute_rows = []
    local_rows = []
    cloud_rows = []
    cost_parameters = pd.read_csv(ROOT / config["local_cost"]["parameter_file"], encoding="utf-8-sig")
    electricity = parameter_value(
        cost_parameters,
        config["local_cost"]["electricity_parameter_id"],
        config["local_cost"]["electricity_price_case"],
    )
    cloud_gpu_price = parameter_value(
        cost_parameters,
        config["cloud_cost"]["gpu_parameter_id"],
        config["cloud_cost"]["gpu_price_case"],
    )
    api_prices = pd.read_csv(ROOT / config["cloud_cost"]["api_price_file"], encoding="utf-8-sig")
    api_prices = api_prices[api_prices["provider"].isin(config["cloud_cost"]["formal_providers"])].copy()
    if set(api_prices["provider"]) != set(config["cloud_cost"]["formal_providers"]):
        raise ValueError("Formal U.S. provider panel incomplete")
    compute = config["compute"]
    local = config["local_cost"]
    reserve = float(compute["installed_reserve_fraction"])

    for case in config["parameter_cases"]:
        summary_case = task_summary[task_summary["parameter_case"] == case]
        load = physical_load(
            summary_case, shapes, accelerator_h_per_service, compute, config["cloud_cost"]["gpu_tasks"]
        )
        hourly_accel = load["hourly_accel"]
        hourly_gpu_accel = load["hourly_gpu_accel"]
        required_without_reserve = float(load["required_without_reserve"])
        installed_servers = float(load["installed_servers"])
        annual_energy_twh = float(load["annual_energy_twh"])
        gpu_contracted = math.ceil(
            float(hourly_gpu_accel.max())
            / (float(compute["accelerators_per_server"]) * float(compute["installed_utilization"]))
        )
        total_service = float(summary_case["effective_service_units_day"].sum())
        storage_gb = (
            float(config["cloud_cost"]["storage_reference_gb"])
            * total_service
            / float(config["cloud_cost"]["storage_reference_service_units_day"])
        )
        storage_payment = storage_cost(storage_gb, config["cloud_cost"])
        compute_rows.append({
            "demand_case_version": config["demand_case_version"], "model_version": config["model_version"],
            "parameter_case": case, "daily_effective_service_units": total_service,
            "accelerator_h_per_service_unit": accelerator_h_per_service,
            "daily_accelerator_h": total_service * accelerator_h_per_service,
            "peak_accelerator_h_per_hour": float(hourly_accel.max()),
            "installed_servers_before_reserve": required_without_reserve,
            "installed_reserve_fraction": reserve,
            "installed_dual_l20_servers": installed_servers,
            "annual_ai_facility_energy_twh": annual_energy_twh,
            "residual_gpu_peak_accelerator_h_per_hour": float(hourly_gpu_accel.max()),
            "contracted_dual_gpu_equivalent_instances": gpu_contracted,
            "storage_gb_proxy": storage_gb,
        })
        for server_case in local["server_price_cases"]:
            purchase = parameter_value(cost_parameters, local["server_parameter_id"], server_case)
            crf = capital_recovery_factor(float(local["discount_rate"]), float(local["economic_life_years"]))
            coefficient = (1 + float(local["facility_capex_fraction"])) * crf + float(local["annual_maintenance_fraction"])
            server_cost = installed_servers * purchase * coefficient
            energy_cost = annual_energy_twh * 1e9 * electricity
            local_rows.append({
                "demand_case_version": config["demand_case_version"], "model_version": config["model_version"],
                "country": "US", "parameter_case": case, "server_price_case": server_case,
                "installed_dual_l20_servers": installed_servers, "installed_reserve_fraction": reserve,
                "annual_ai_facility_energy_twh": annual_energy_twh, "server_purchase_price_usd": purchase,
                "electricity_price_usd_per_kwh": electricity,
                "annual_server_capital_facility_maintenance_cost_usd": server_cost,
                "annual_electricity_cost_usd": energy_cost,
                "annual_local_cost_usd": server_cost + energy_cost,
                "annual_local_cost_billion_usd": (server_cost + energy_cost) / 1e9,
                "cost_boundary": "national_pooled_owned_capacity_single_10pct_reserve",
            })
        annual_input = float(summary_case["annual_input_tokens"].sum())
        annual_output = float(summary_case["annual_output_tokens"].sum())
        gpu_payment = gpu_contracted * cloud_gpu_price
        for price in api_prices.itertuples(index=False):
            api_cost = (
                annual_input / 1e6 * float(price.input_usd_per_mtoken)
                + annual_output / 1e6 * float(price.output_usd_per_mtoken)
            )
            total = api_cost + gpu_payment + storage_payment
            cloud_rows.append({
                "demand_case_version": config["demand_case_version"], "model_version": config["model_version"],
                "country": "US", "parameter_case": case, "provider": price.provider,
                "model_id": price.model_id, "annual_input_tokens": annual_input,
                "annual_output_tokens": annual_output, "input_usd_per_mtoken": price.input_usd_per_mtoken,
                "output_usd_per_mtoken": price.output_usd_per_mtoken,
                "annual_api_token_cost_usd": api_cost,
                "contracted_dual_gpu_equivalent_instances": gpu_contracted,
                "reserved_gpu_unit_price_usd_per_year": cloud_gpu_price,
                "annual_residual_reserved_gpu_payment_usd": gpu_payment,
                "cloud_storage_gb_proxy": storage_gb,
                "annual_cloud_storage_cost_usd_proxy": storage_payment,
                "annual_full_cloud_cost_usd": total,
                "annual_full_cloud_cost_billion_usd": total / 1e9,
                "gpu_billing_mode": "one_year_standard_reserved_all_upfront",
                "ondemand_displayed": False,
                "cost_boundary": "office_agent_API_plus_four_task_reserved_GPU_plus_S3",
            })

    compute_frame = pd.DataFrame(compute_rows)
    local_frame = pd.DataFrame(local_rows)
    cloud_frame = pd.DataFrame(cloud_rows)
    base_local = local_frame[local_frame["server_price_case"] == "base"].set_index("parameter_case")
    cloud_frame["ratio_to_local_same_case"] = cloud_frame.apply(
        lambda row: row.annual_full_cloud_cost_usd / base_local.loc[row.parameter_case, "annual_local_cost_usd"], axis=1
    )
    cloud_frame["difference_to_local_same_case_usd"] = cloud_frame.apply(
        lambda row: row.annual_full_cloud_cost_usd - base_local.loc[row.parameter_case, "annual_local_cost_usd"], axis=1
    )
    comparison = pd.concat([
        base_local.reset_index().assign(
            option=lambda x: "US local / " + x["parameter_case"],
            option_group="us_local_owned",
            provider="local",
            model_id="dual_L20",
            annual_cost_usd=lambda x: x["annual_local_cost_usd"],
            annual_cost_billion_usd=lambda x: x["annual_local_cost_billion_usd"],
            ratio_to_local_same_case=1.0,
        )[["parameter_case", "option", "option_group", "provider", "model_id", "annual_cost_usd", "annual_cost_billion_usd", "ratio_to_local_same_case"]],
        cloud_frame.assign(
            option=lambda x: "US full cloud " + x["provider"] + " / " + x["model_id"],
            option_group="us_full_cloud_hybrid",
            annual_cost_usd=lambda x: x["annual_full_cloud_cost_usd"],
            annual_cost_billion_usd=lambda x: x["annual_full_cloud_cost_billion_usd"],
        )[["parameter_case", "option", "option_group", "provider", "model_id", "annual_cost_usd", "annual_cost_billion_usd", "ratio_to_local_same_case"]],
    ], ignore_index=True)

    # One-at-a-time cost-ratio sensitivity around the base demand case. Demand
    # is held fixed here so the table isolates economic-boundary leverage.
    base_local_row = local_frame.query("parameter_case == 'base' and server_price_case == 'base'").iloc[0]
    base_cloud_row = cloud_frame.query("parameter_case == 'base'").sort_values("annual_full_cloud_cost_usd").iloc[0]
    base_server_price = float(base_local_row.server_purchase_price_usd)
    base_server_count = float(base_local_row.installed_dual_l20_servers)
    base_energy_cost = float(base_local_row.annual_electricity_cost_usd)
    base_gpu_payment = float(base_cloud_row.annual_residual_reserved_gpu_payment_usd)
    base_token_cost = float(base_cloud_row.annual_api_token_cost_usd)
    base_storage_cost = float(base_cloud_row.annual_cloud_storage_cost_usd_proxy)

    def local_cost_at(life: float, server_price_multiplier: float = 1.0, energy_multiplier: float = 1.0) -> float:
        crf = capital_recovery_factor(float(local["discount_rate"]), life)
        coefficient = (1 + float(local["facility_capex_fraction"])) * crf + float(local["annual_maintenance_fraction"])
        return base_server_count * base_server_price * server_price_multiplier * coefficient + base_energy_cost * energy_multiplier

    def cloud_cost_at(gpu_multiplier: float = 1.0, token_multiplier: float = 1.0) -> float:
        return base_gpu_payment * gpu_multiplier + base_token_cost * token_multiplier + base_storage_cost

    sensitivity_rows = []
    core_life = float(local["economic_life_years"])
    for life in local["economic_life_sensitivity_years"]:
        local_cost_value = local_cost_at(float(life))
        sensitivity_rows.append({
            "parameter": "owned_server_economic_life_years", "case": str(life),
            "parameter_multiplier": float(life) / core_life,
            "annual_local_cost_usd": local_cost_value,
            "annual_cloud_cost_usd": cloud_cost_at(),
            "least_cost_cloud_to_local_ratio": cloud_cost_at() / local_cost_value,
            "interpretation": "asset-life boundary; demand fixed",
        })
    for parameter, multiplier in (("owned_server_price", 0.9), ("owned_server_price", 1.1),
                                  ("owned_electricity_price", 0.9), ("owned_electricity_price", 1.1),
                                  ("cloud_reserved_gpu_price", 0.9), ("cloud_reserved_gpu_price", 1.1),
                                  ("cloud_api_token_price", 0.9), ("cloud_api_token_price", 1.1)):
        local_value = local_cost_at(core_life,
                                    server_price_multiplier=multiplier if parameter == "owned_server_price" else 1.0,
                                    energy_multiplier=multiplier if parameter == "owned_electricity_price" else 1.0)
        cloud_value = cloud_cost_at(
            gpu_multiplier=multiplier if parameter == "cloud_reserved_gpu_price" else 1.0,
            token_multiplier=multiplier if parameter == "cloud_api_token_price" else 1.0,
        )
        sensitivity_rows.append({
            "parameter": parameter, "case": f"{multiplier:.1f}x", "parameter_multiplier": multiplier,
            "annual_local_cost_usd": local_value, "annual_cloud_cost_usd": cloud_value,
            "least_cost_cloud_to_local_ratio": cloud_value / local_value,
            "interpretation": "one-at-a-time +/-10%; base demand fixed",
        })
    cost_sensitivity = pd.DataFrame(sensitivity_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(ROOT / "02_data/processed/us_demand/us_manufacturing_ai_effective_service_2030.csv", index=False, encoding="utf-8-sig")
    task_summary.to_csv(output_dir / "us_national_task_summary.csv", index=False, encoding="utf-8-sig")
    detail.groupby(["parameter_case", "naics3", "industry_name"], as_index=False).agg(
        effective_service_units_day=("effective_service_units_day", "sum"),
        annual_effective_service_units=("annual_effective_service_units", "sum"),
    ).to_csv(output_dir / "us_naics3_task_summary.csv", index=False, encoding="utf-8-sig")
    compute_frame.to_csv(output_dir / "us_demand_validation.csv", index=False, encoding="utf-8-sig")
    scale_validation = compute_frame[["parameter_case", "annual_ai_facility_energy_twh"]].copy()
    external = config["external_scale_validation"]
    scale_validation["china_same_case_energy_twh"] = scale_validation["parameter_case"].map(external["china_energy_twh"])
    scale_validation["us_to_china_energy_ratio"] = (
        scale_validation["annual_ai_facility_energy_twh"] / scale_validation["china_same_case_energy_twh"]
    )
    scale_validation["validation_band_low"] = float(external["expected_us_to_china_ratio_low"])
    scale_validation["validation_band_high"] = float(external["expected_us_to_china_ratio_high"])
    scale_validation["inside_external_validation_band"] = scale_validation["us_to_china_energy_ratio"].between(
        float(external["expected_us_to_china_ratio_low"]), float(external["expected_us_to_china_ratio_high"])
    )
    scale_validation["validation_role"] = "external_check_not_input_to_demand_formula"
    scale_validation.to_csv(output_dir / "us_macro_alignment.csv", index=False, encoding="utf-8-sig")
    parameter_audit.to_csv(output_dir / "us_bottom_up_parameter_audit.csv", index=False, encoding="utf-8-sig")
    local_frame.to_csv(output_dir / "us_local_cost.csv", index=False, encoding="utf-8-sig")
    cloud_frame.to_csv(output_dir / "us_full_cloud_cost.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    cost_sensitivity.to_csv(output_dir / "us_cost_ratio_sensitivity.csv", index=False, encoding="utf-8-sig")

    lineage = {
        "schema_version": 1,
        "demand_case_version": config["demand_case_version"],
        "formula": "service = US driver * adoption * workload_units_per_active_driver_day * effective_service_per_workload_unit * industry_applicability",
        "inputs": {str(path.relative_to(ROOT)): {"sha256": sha256(path)} for path in (
            activity_path, mecs_path, berd_path, task_parameter_path, efficiency_path
        )},
        "task_shapes": config["task_shapes"],
        "external_scale_validation": external,
        "installed_reserve_fraction_applied_once": reserve,
        "token_tasks": config["cloud_cost"]["token_tasks"],
        "gpu_tasks": config["cloud_cost"]["gpu_tasks"],
        "notes": [
            "U.S. official activity and adoption determine active drivers and NAICS3/task allocation.",
            "Task-specific operational units per adopted driver create the national total without a top-down multiplier.",
            "The same-case China energy ratio is only an external validation check.",
            "Office/agent only generate API tokens; four physical tasks remain reserved GPU.",
            "No on-demand GPU option is produced.",
        ],
    }
    (ROOT / "02_data/processed/us_demand/us_manufacturing_ai_demand_lineage.json").write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "detail": detail, "task_summary": task_summary, "parameter_audit": parameter_audit, "compute": compute_frame,
        "local": local_frame, "cloud": cloud_frame, "comparison": comparison,
        "cost_sensitivity": cost_sensitivity,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = build(config, args.output_dir)
    base_local = result["local"].query("parameter_case == 'base' and server_price_case == 'base'").iloc[0]
    base_cloud = result["cloud"].query("parameter_case == 'base'").sort_values("annual_full_cloud_cost_usd")
    table = "\n".join(
        f"| {row.provider} | {row.annual_api_token_cost_usd/1e9:.3f} | {row.annual_residual_reserved_gpu_payment_usd/1e9:.3f} | {row.annual_cloud_storage_cost_usd_proxy/1e6:.3f} | {row.annual_full_cloud_cost_billion_usd:.3f} | {row.ratio_to_local_same_case:.2f} |"
        for row in base_cloud.itertuples(index=False)
    )
    sensitivity = result["cost_sensitivity"]
    sensitivity_table = "\n".join(
        f"| {row.parameter} | {row.case} | {row.least_cost_cloud_to_local_ratio:.3f} |"
        for row in sensitivity.itertuples(index=False)
    )
    findings = f"""# 美国制造业自身需求下的本地与完整云化成本

版本：`{config['demand_case_version']}`；年份：2030 base。

## 中心结果

美国官方活动量、采用率、每个采用者的任务单元数和单任务计算强度共同产生 **{result['compute'].query("parameter_case == 'base'").iloc[0].daily_effective_service_units:,.0f} 有效服务单位/日**。本地全国池化需要 **{base_local.installed_dual_l20_servers:,.0f} 台双 L20 服务器**，15% 装机裕量只计算一次；设施年用电 **{base_local.annual_ai_facility_energy_twh:.3f} TWh**，年核心成本 **{base_local.annual_local_cost_billion_usd:.3f} 十亿美元**。

| 美国完整云化厂商 | Token API | 四任务预留 GPU | 存储（百万美元） | 总成本（十亿美元） | 相对本地 |
|---|---:|---:|---:|---:|---:|
{table}

## 云/本地比值的关键参数

以下为基准需求不变的一次一参数敏感性；云端取最低价正式厂商。

| 参数 | 参数情景 | 云/本地 |
|---|---:|---:|
{sensitivity_table}

服务器经济寿命是最强杠杆；上表同时给出3/4/5/5.5/6/7年，避免把会计折旧寿命、实际服役寿命和AI加速器经济寿命混为一谈。预留GPU单价±10%、本地服务器价格±10%仍是次一级关键杠杆；Token价格±10%影响很小，因为当前最低价云账单主要由四项物理任务的预留GPU构成。

## 解释边界

- 总量完全由六任务自下而上相加；不存在按中国用电反推的全国乘数。
- 中美同档用电比例只作为制造业经济规模外部校验，不进入需求公式。
- office/agent 使用美国区域 Token API；vision、maintenance、scheduling、simulation 使用预留 GPU；未展示按量 GPU。
- 本地按全国充分池化容量筛查，不把 establishment 当 firm/group，也不加入每厂一台服务器的离散下限。
- 单位服务强度和24小时任务形状仍从现有模型迁移，属于情景假设；美国官方数据负责活动量、采用率和行业结构。
- 云账单不叠加云商内部服务器、电力、折旧；存储不含请求、检索和出口流量。
- 核心本地服务器经济寿命为5年；5.5年和6年为有公司披露支持的替代情景，7年为条件性延寿情景。最低价正式云厂商相对本地的核心目标仍为约1.20倍。
- 公开证据并非单向：Meta披露多数服务器和网络资产5.5年，Amazon为5—6年、Alphabet通常为6年；但Amazon也因AI/机器学习技术迭代而把部分资产由6年缩回5年。微软研究的AI机群TCO模拟则显示，多数B100之前GPU代际延长到6年以上仍可能经济。故5年是保守基准，6年可辩护，7年以上不能作为无条件行业平均值。
"""
    (args.output_dir / "findings.md").write_text(findings, encoding="utf-8")
    checks = {
        "status": "validated_us_bottom_up_demand_cost_screen",
        "demand_case_version": config["demand_case_version"],
        "detail_rows": len(result["detail"]),
        "tasks": sorted(result["detail"]["task_id"].unique()),
        "naics3_count": int(result["detail"]["naics3"].nunique()),
        "parameter_cases": sorted(result["detail"]["parameter_case"].unique()),
        "installed_reserve_fraction_applied_once": config["compute"]["installed_reserve_fraction"],
        "token_tasks": config["cloud_cost"]["token_tasks"],
        "gpu_tasks": config["cloud_cost"]["gpu_tasks"],
        "ondemand_displayed": False,
        "minimum_base_cloud_to_local_ratio": float(base_cloud["ratio_to_local_same_case"].min()),
    }
    (args.output_dir / "validated.done.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
