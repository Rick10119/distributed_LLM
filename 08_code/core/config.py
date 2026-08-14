"""Configuration loading and validation for core scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import yaml


VALID_SCENARIOS = {"IF", "IG", "IG_1host", "IG_multisite", "II_1host", "II_multihost"}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return payload


def load_config(root: Path, defaults_path: Path, run_path: Path) -> dict[str, Any]:
    defaults = read_yaml(root / defaults_path)
    run = read_yaml(root / run_path)
    config = deep_merge(defaults, run)
    config["_root"] = str(root.resolve())
    config["_defaults_path"] = str(defaults_path)
    config["_run_path"] = str(run_path)
    resolve_scenario_selections(root, config)
    resolve_compute_efficiency(config)
    resolve_model_state(config)
    validate_config(config)
    return config


def resolve_scenario_selections(root: Path, config: dict[str, Any]) -> None:
    """Apply mainline named-case selections without copying values between files."""
    registry_path = Path(config.get("scenario_registry_path", "config/scenarios/mainline.yaml"))
    registry = read_yaml(root / registry_path)
    hardware = registry.get("compute_hardware", {})
    local_cpu_case = str(hardware.get("local_cpu_price_case", "base"))
    cloud_capacity_case = str(hardware.get("china_cloud_capacity_price_case", "base"))
    if local_cpu_case not in {"low", "base", "high"}:
        raise ValueError("Scenario registry local_cpu_price_case must be low, base, or high")
    if cloud_capacity_case not in {"low", "base", "high"}:
        raise ValueError("Scenario registry china_cloud_capacity_price_case must be low, base, or high")
    config.setdefault("compute_hardware", {})["local_cpu_price_case"] = local_cpu_case
    config.setdefault("full_cloud_cost", {})["main_gpu_price_case"] = cloud_capacity_case
    config["full_cloud_cost"]["main_cpu_price_case"] = cloud_capacity_case


def resolve_compute_efficiency(config: dict[str, Any]) -> None:
    service = config["demand"]["effective_service"]
    selected_case = str(service.get("compute_efficiency_case", ""))
    path = rooted(config, config["paths"]["model_ready_compute_efficiency"])
    frame = __import__("pandas").read_csv(path, encoding="utf-8-sig")
    selected = frame[frame["efficiency_case"] == selected_case]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one compute-efficiency row for {selected_case}")
    service["accelerator_h_per_service_unit"] = float(
        selected["accelerator_h_per_service_unit"].iloc[0]
    )
    service["compute_efficiency_evidence_status"] = str(
        selected["evidence_status"].iloc[0]
    )


def resolve_model_state(config: dict[str, Any]) -> None:
    selection = config.get("model_state", {})
    path = rooted(config, config["paths"]["model_ready_model_lifecycle"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("parameter_case") != payload.get("parameter_case"):
        raise ValueError("Selected model-state case does not match prepared lifecycle parameters")
    enabled = bool(selection.get("enabled", False))
    config["model_state"] = {**payload, "enabled": enabled}


def validate_config(config: dict[str, Any]) -> None:
    model_version = str(config.get("model_version", ""))
    if not re.fullmatch(r"v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)", model_version):
        raise ValueError("model_version must use semantic version form vMAJOR.MINOR.PATCH")
    industries = config.get("selected_industries")
    scenarios = config.get("selected_scenarios")
    if not isinstance(industries, list) or not industries:
        raise ValueError("selected_industries must be a non-empty list")
    if not all(isinstance(code, str) and code.startswith("C") for code in industries):
        raise ValueError("selected_industries contains an invalid industry code")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("selected_scenarios must be a non-empty list")
    unknown = set(scenarios) - VALID_SCENARIOS
    if unknown:
        raise ValueError(f"Unknown core scenarios: {sorted(unknown)}")
    case = config.get("industry_parameter_case")
    if case not in {"low", "base", "high"}:
        raise ValueError("industry_parameter_case must be low, base, or high")
    model = config.get("model", {})
    horizon = int(model.get("horizon_hours", 0))
    if horizon < 24 or horizon > 8784 or horizon % 24 != 0:
        raise ValueError(
            "The model requires 24 to 8784 hours in whole days"
        )
    if model.get("load_profile_mode") not in {"measured_continuous", "typical_day"}:
        raise ValueError(
            "load_profile_mode must be measured_continuous or typical_day"
        )
    if model.get("load_profile_mode") == "typical_day" and horizon != 24:
        raise ValueError("typical_day load_profile_mode requires model.horizon_hours = 24")
    if model.get("grid_capacity_upgrade_boundary") != "no_ai_optimized_net_peak_zero_headroom_credit":
        raise ValueError(
            "The active core model must use the optimized no-AI net grid peak with zero existing-headroom credit"
        )
    if "existing_headroom_fraction" in model:
        raise ValueError(
            "existing_headroom_fraction is retired from the active core model; "
            "all positive AI-driven net-peak increments require capacity upgrades"
        )
    if not isinstance(model.get("server_groups_integer"), bool):
        raise ValueError("server_groups_integer must be boolean")
    for field in ("installed_server_groups_integer", "online_server_groups_integer"):
        if field in model and not isinstance(model[field], bool):
            raise ValueError(f"{field} must be boolean")
    solver = model.get("solver", {})
    if solver.get("name") not in {"gurobi", "highs"}:
        raise ValueError("The core solver must be gurobi or highs")
    if float(solver.get("mip_gap", -1.0)) < 0.0:
        raise ValueError("Solver mip_gap must be non-negative")
    if int(solver.get("threads", 0)) < 1:
        raise ValueError("Solver threads must be at least one")
    energy = config.get("energy", {})
    if energy.get("grid_energy_price_mode") not in {
        "flat_tariff",
        "guangdong_spot_retail_representative_week",
    }:
        raise ValueError("Unsupported core grid-energy price mode")
    if energy.get("pv_capacity_mode") not in {
        "none",
        "existing_rooftop_at_model_limit",
        "optimize_new_rooftop_pv",
    }:
        raise ValueError("Unsupported core PV capacity mode")
    battery_cost = energy.get("battery_cost", {})
    if int(battery_cost.get("source_year", 0)) != 2025:
        raise ValueError("The active battery-cost boundary must use the 2025 source table")
    if battery_cost.get("energy_technology") != "battery storage":
        raise ValueError("Battery energy cost must use the battery storage technology row")
    if battery_cost.get("power_technology") != "battery inverter":
        raise ValueError("Battery power cost must use the battery inverter technology row")
    if float(battery_cost.get("eur_to_rmb", 0.0)) <= 0.0:
        raise ValueError("Battery-cost EUR-to-RMB conversion must be positive")
    if not isinstance(energy.get("battery_investment_enabled"), bool):
        raise ValueError("battery_investment_enabled must be boolean")
    grid_limit = energy.get("grid_expansion_limit_mw")
    if grid_limit is not None and float(grid_limit) < 0.0:
        raise ValueError("grid_expansion_limit_mw must be non-negative or null")
    grid_penalty = float(
        energy.get("grid_expansion_objective_penalty_rmb_per_mw_year", 0.0)
    )
    if not np.isfinite(grid_penalty) or grid_penalty < 0.0:
        raise ValueError(
            "grid_expansion_objective_penalty_rmb_per_mw_year must be finite and non-negative"
        )
    fixed_battery = energy.get("battery_fixed_power_mw")
    if fixed_battery is not None and float(fixed_battery) < 0.0:
        raise ValueError("battery_fixed_power_mw must be non-negative when provided")
    hybrid = config.get("hybrid_cloud", {})
    if not isinstance(hybrid.get("enabled", False), bool):
        raise ValueError("hybrid_cloud.enabled must be boolean")
    for field in (
        "gpu_annual_subscription_rmb_per_group",
        "gpu_capacity_accelerator_h_per_h",
        "cpu_annual_subscription_rmb_per_group",
        "cpu_capacity_local_server_h_per_h",
    ):
        if float(hybrid.get(field, 0.0)) <= 0.0:
            raise ValueError(f"hybrid_cloud.{field} must be positive")
    cloud_share = float(hybrid.get("maximum_cloud_service_share", -1.0))
    if not 0.0 <= cloud_share <= 1.0:
        raise ValueError("hybrid_cloud.maximum_cloud_service_share must be in [0,1]")
    factory = config.get("factory", {})
    if factory.get("roof_area_mode") != "industry_us_mecs_2022_reference":
        raise ValueError("The active rooftop boundary must use industry US MECS 2022 references")
    if "roof_area_m2" in factory:
        raise ValueError("Global factory roof_area_m2 is retired; use the industry rooftop table")
    for field in ("roof_usable_fraction", "pv_module_efficiency", "pv_realization_fraction"):
        value = float(factory.get(field, 0.0))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"factory {field} must be in (0,1]")
    demand = config.get("demand", {})
    hardware = config.get("compute_hardware", {})
    if hardware.get("mode") not in {"gpu_only", "heterogeneous_cpu_gpu"}:
        raise ValueError("compute_hardware.mode must be gpu_only or heterogeneous_cpu_gpu")
    if hardware.get("mode") == "heterogeneous_cpu_gpu" and not hardware.get("routing_config"):
        raise ValueError("heterogeneous CPU/GPU mode requires a routing_config")
    if hardware.get("local_cpu_price_case", "base") not in {"low", "base", "high"}:
        raise ValueError("compute_hardware.local_cpu_price_case must be low, base, or high")
    service = demand.get("effective_service", {})
    if service.get("parameter_case") not in {"low", "base", "high"}:
        raise ValueError("effective-service parameter_case must be low, base, or high")
    if float(service.get("accelerator_h_per_service_unit", 0)) <= 0:
        raise ValueError("accelerator_h_per_service_unit must be positive")
    if service.get("compute_efficiency_case") not in {"efficient", "base", "conservative"}:
        raise ValueError("compute_efficiency_case must be efficient, base, or conservative")
    alignment = demand.get("external_energy_alignment", {})
    if not (
        0 < float(alignment.get("warning_ratio_low", 0))
        <= 1.0
        <= float(alignment.get("warning_ratio_high", 0))
    ):
        raise ValueError("external-energy alignment ratios must bracket 1.0")
    if alignment.get("per_industry_check", "hard") not in {"hard", "diagnostic"}:
        raise ValueError("per_industry_check must be hard or diagnostic")
    server = config.get("server", {})
    if server.get("installed_capacity_rule") != "max_10pct_planning_headroom_and_n_plus_1":
        raise ValueError(
            "server installed_capacity_rule must be max_10pct_planning_headroom_and_n_plus_1"
        )
    for field in ("installed_reserve_fraction", "normal_dispatch_reserve_fraction"):
        value = float(server.get(field, -1.0))
        if not 0.0 <= value < 1.0:
            raise ValueError(f"server {field} must be in [0,1)")
    if int(server.get("n_plus_spare_server_groups", -1)) not in {0, 1}:
        raise ValueError("The server-capacity screen permits zero or one N+ spare group")
    if float(server.get("maximum_wall_power_kw", 0)) <= float(
        server.get("online_idle_wall_power_kw", 0)
    ):
        raise ValueError("maximum server power must exceed online-idle power")
    if float(server.get("online_idle_wall_power_kw", 0)) < float(
        server.get("cold_spare_standby_power_kw", 0)
    ):
        raise ValueError("online-idle power cannot be below standby power")
    model_state = config.get("model_state", {})
    if bool(model_state.get("enabled")):
        if int(model_state.get("minimum_server_groups_for_model_state", 0)) < 1:
            raise ValueError("Enabled model state requires at least one server group")
        if float(model_state.get("model_storage_required_gb_per_deployment", 0)) <= 0:
            raise ValueError("Enabled model state requires positive model storage")
    payment = config.get("enterprise_payment", {})
    if int(payment.get("accelerators_per_cloud_instance", 0)) < 1:
        raise ValueError("Cloud billing proxy requires at least one accelerator per instance")
    if payment.get("price_cases") != ["low", "base", "high"]:
        raise ValueError("enterprise_payment price_cases must be low, base, high")
    api_token = config.get("api_token_cost", {})
    if api_token.get("cost_case_version") != "api_token_cost_v1.1.0":
        raise ValueError("Unexpected API Token cost case version")
    if api_token.get("included_core_tasks") != ["office", "agent"]:
        raise ValueError("First API Token cost case must include only office and agent")
    for field in ("cache_hit_rate", "batch_share", "retry_rate"):
        value = float(api_token.get(field, -1))
        if not 0 <= value <= 1:
            raise ValueError(f"api_token_cost {field} must be in [0,1]")
    workload_weight = float(api_token.get("large_workload_weight_when_unobserved", -1))
    if not 0 <= workload_weight <= 1:
        raise ValueError("api_token_cost large_workload_weight_when_unobserved must be in [0,1]")
    full_cloud = config.get("full_cloud_cost", {})
    if full_cloud.get("scenario_version") != "full_cloud_cost_v1.0.0":
        raise ValueError("Unexpected full-cloud cost scenario version")
    if full_cloud.get("token_tasks") != ["office", "agent"]:
        raise ValueError("Full-cloud Token tasks must be office and agent")
    if full_cloud.get("gpu_tasks") != ["vision", "maintenance", "scheduling", "simulation"]:
        raise ValueError("Full-cloud GPU tasks must cover the four non-Token tasks")
    if full_cloud.get("gpu_billing_mode") != "cloud_reserved_capacity":
        raise ValueError("Formal full-cloud scenario must use reserved GPU capacity")
    if full_cloud.get("main_gpu_price_case") not in {"low", "base", "high"}:
        raise ValueError("Full-cloud main GPU price case must be low, base, or high")
    if full_cloud.get("main_cpu_price_case", "base") not in {"low", "base", "high"}:
        raise ValueError("Full-cloud main CPU price case must be low, base, or high")
    if full_cloud.get("cloud_storage_price_case") not in {"low", "base", "high"}:
        raise ValueError("Full-cloud storage price case must be low, base, or high")
    if full_cloud.get("storage_reference_architecture") not in {"IF", "IG", "II_1host"}:
        raise ValueError("Full-cloud storage reference architecture is invalid")
    if bool(full_cloud.get("display_ondemand", True)):
        raise ValueError("Formal full-cloud reporting must not display on-demand GPU billing")
    if full_cloud.get("formal_country") != "China":
        raise ValueError("Core full-cloud formal country must be China")
    if full_cloud.get("formal_allowed_providers") != ["Alibaba Cloud", "DeepSeek"]:
        raise ValueError("China formal comparison must contain only Alibaba Cloud and DeepSeek")


def rooted(config: dict[str, Any], configured_path: str) -> Path:
    return Path(config["_root"]) / configured_path


def write_resolved_config(config: dict[str, Any], path: Path) -> None:
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
