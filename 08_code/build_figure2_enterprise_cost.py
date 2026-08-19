#!/usr/bin/env python3
"""Build the four-panel China–U.S. enterprise-cost Figure 2 draft."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "distributed_llm_matplotlib"))

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy as np
    import pandas as pd
    import yaml
except ModuleNotFoundError:
    analysis_python = Path("/opt/anaconda3/bin/python")
    if analysis_python.exists() and os.environ.get("FIGURE2_DRAFT_PYTHON_REEXEC") != "1":
        os.environ["FIGURE2_DRAFT_PYTHON_REEXEC"] = "1"
        os.execv(str(analysis_python), [str(analysis_python), *sys.argv])
    raise


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS_PATH = ROOT / "config/defaults.yaml"
DEFAULT_CONFIG = yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))
DEFAULT_MODEL_VERSION = str(DEFAULT_CONFIG["model_version"])
DEFAULT_RESULTS_ROOT = ROOT / str(DEFAULT_CONFIG["paths"]["results_root"])
VERSION_RESULT_ROOT = DEFAULT_RESULTS_ROOT / DEFAULT_MODEL_VERSION / "result"
SENSITIVITY_VERSION_ROOT = DEFAULT_RESULTS_ROOT / "sensitivity" / DEFAULT_MODEL_VERSION
RESULT_ROOT = VERSION_RESULT_ROOT / "manuscript_figures"
CHINA_PROVIDERS = ["DeepSeek", "Alibaba Cloud"]
DEMAND_CASES = ["low", "base", "high"]
CHINA_COST_COLUMNS = [
    "industry_equivalent_installed_gpu_server_groups",
    "industry_equivalent_installed_cpu_server_groups",
    "industry_equivalent_annual_ai_energy_cost_rmb",
    "industry_equivalent_annual_incremental_maximum_demand_cost_rmb",
    "industry_equivalent_annual_incremental_total_cost_rmb",
    "industry_equivalent_annual_ai_facility_energy_twh",
]
US_PROVIDERS = ["OpenAI", "Anthropic", "Google"]
US_COST_COLUMNS = [
    "local_total_annual_cost_usd",
    "cloud_total_annual_cost_usd",
    "local_gpu_servers",
    "local_cpu_servers",
    "annual_facility_energy_twh",
    "cloud_gpu_reserved_instances",
    "cloud_cpu_reserved_instances",
    "cloud_gpu_reserved_cost_usd",
    "cloud_cpu_reserved_cost_usd",
    "cloud_token_api_cost_usd",
]
COLORS = {
    "gpu": "#496F95", "cpu": "#86A8C2", "energy": "#D6A457",
    "demand": "#9B735E", "token": "#8A6DAA",
    "low": "#A8B3BA", "base": "#385E7A", "high": "#A45C40",
}


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _crf(rate: float, years: float) -> float:
    return rate * (1.0 + rate) ** years / ((1.0 + rate) ** years - 1.0)


def _annual_server_cost(server: dict, discount_rate: float) -> float:
    capex = float(server["purchase_cost_rmb"])
    return capex * (
        (1.0 + float(server["facility_capex_fraction"]))
        * _crf(discount_rate, float(server["economic_life_years"]))
        + float(server["annual_maintenance_fraction"])
    )


def _industry_token_demand(
    workload: pd.DataFrame,
    industry_code: str,
    employment: float,
    large_share: float,
    demand_case: str,
) -> tuple[float, float]:
    annual_input = 0.0
    annual_output = 0.0
    for raw_task in ("office_rag_copilot", "business_agent"):
        selected = workload[
            workload["industry_code"].eq(industry_code)
            & workload["task_type"].eq(raw_task)
            & workload["scenario_year"].astype(str).eq("2030")
            & workload["scenario_level"].eq(demand_case)
        ]
        by_size = {row.enterprise_size_class: row for row in selected.itertuples(index=False)}
        if set(by_size) != {"large", "sme"}:
            raise ValueError(f"Missing {demand_case} workload rows for {industry_code}/{raw_task}")
        for size, weight in (("large", large_share), ("sme", 1.0 - large_share)):
            row = by_size[size]
            tasks = (
                employment * weight * float(row.ai_adoption_rate)
                * float(row.active_user_or_equipment_share)
                * float(row.service_intensity_per_driver_day)
            )
            calls = tasks * float(row.calls_per_task)
            annual_input += calls * float(row.input_tokens_per_call) * 365.0
            annual_output += calls * float(row.output_tokens_per_call) * 365.0
    return annual_input, annual_output


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _national_demand_ratios(service: pd.DataFrame) -> dict[str, float]:
    totals = service.groupby("parameter_case")["effective_service_units_day"].sum()
    base = float(totals.get("base", 0.0))
    if base <= 0:
        raise ValueError("Base effective-service demand must be positive")
    return {case: float(totals.get(case, base)) / base for case in DEMAND_CASES}


def _national_demand_totals(service: pd.DataFrame) -> dict[str, float]:
    totals = service.groupby("parameter_case")["effective_service_units_day"].sum()
    missing = [case for case in DEMAND_CASES if float(totals.get(case, 0.0)) <= 0.0]
    if missing:
        raise ValueError(f"Positive national effective-service demand is required for {missing}")
    return {case: float(totals[case]) for case in DEMAND_CASES}


def _us_results_ready(national_path: Path, validation_path: Path) -> bool:
    """Return true only for a complete U.S. run using the China-aligned sizing rule."""
    if (
        not national_path.exists()
        or national_path.stat().st_size == 0
        or not validation_path.exists()
        or validation_path.stat().st_size == 0
    ):
        return False
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(national_path, encoding="utf-8-sig")
    except (json.JSONDecodeError, OSError, pd.errors.ParserError):
        return False
    if validation.get("aggregation_boundary") != "industry_sum_china_aligned_continuous_capacity":
        return False
    if validation.get("installed_capacity_rule") != "daily_average_plus_planning_headroom_and_n_plus_spare":
        return False
    required = {"parameter_case", "provider", *US_COST_COLUMNS}
    if not required.issubset(frame.columns):
        return False
    selected = frame[
        frame.get("cpu_server_price_case", pd.Series("base", index=frame.index)).eq("base")
    ]
    expected = {(case, provider) for case in DEMAND_CASES for provider in US_PROVIDERS}
    observed = set(zip(selected["parameter_case"], selected["provider"]))
    return expected.issubset(observed) and not selected[list(US_COST_COLUMNS)].isna().any().any()


def _industry_demand_ratios(service: pd.DataFrame, industries: list[str]) -> pd.DataFrame:
    grouped = service.groupby(["industry_code", "parameter_case"])["effective_service_units_day"].sum()
    pivot = grouped.unstack("parameter_case")
    pivot.index = pivot.index.astype(str)
    global_ratios = _national_demand_ratios(service)
    ratios = pd.DataFrame(index=industries, columns=DEMAND_CASES, dtype=float)
    for case in DEMAND_CASES:
        if case in pivot and "base" in pivot:
            case_ratio = pivot[case] / pivot["base"].replace(0.0, np.nan)
            ratios[case] = case_ratio.reindex(industries).fillna(global_ratios[case])
        else:
            ratios[case] = global_ratios[case]
    ratios["base"] = 1.0
    return ratios


def _extract_china_case(frame: pd.DataFrame, industries: list[str]) -> pd.DataFrame:
    if {"architecture", "industry"}.issubset(frame.columns):
        selected = frame[frame["architecture"].eq("IG_1host")].copy()
        selected["industry"] = selected["industry"].astype(str)
        selected = selected.drop_duplicates("industry", keep="last").set_index("industry")
    else:
        selected = pd.DataFrame(index=pd.Index([], name="industry"))
    selected = selected.reindex(industries)
    for column in CHINA_COST_COLUMNS:
        if column not in selected:
            selected[column] = np.nan
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected["architecture"] = "IG_1host"
    return selected


def _prepare_china_cases(
    frames: dict[str, pd.DataFrame], industries: list[str], demand_ratios: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], set[str]]:
    extracted = {case: _extract_china_case(frame, industries) for case, frame in frames.items()}
    base = extracted["base"].copy()
    base[CHINA_COST_COLUMNS] = base[CHINA_COST_COLUMNS].fillna(0.0)
    prepared: dict[str, pd.DataFrame] = {}
    imputed_cases: set[str] = set()
    for case in DEMAND_CASES:
        selected = extracted[case].copy()
        missing = selected[CHINA_COST_COLUMNS].isna()
        if bool(missing.to_numpy().any()):
            imputed_cases.add(case)
        for column in CHINA_COST_COLUMNS:
            scaled_base = base[column] * demand_ratios.loc[industries, case]
            selected[column] = selected[column].fillna(scaled_base).fillna(0.0)
        selected["demand_case"] = case
        prepared[case] = selected.reset_index()
    return prepared, imputed_cases


def _prepare_us_cases(
    frame: pd.DataFrame, demand_ratios: dict[str, float],
) -> tuple[pd.DataFrame, set[str]]:
    if "cpu_server_price_case" in frame:
        frame = frame[frame["cpu_server_price_case"].eq("base")].copy()
    required_keys = {"parameter_case", "provider"}
    if required_keys.issubset(frame.columns):
        frame = frame.drop_duplicates(["parameter_case", "provider"], keep="last")
        frame = frame.set_index(["parameter_case", "provider"])
    else:
        frame = pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=["parameter_case", "provider"]))
    expected = pd.MultiIndex.from_product(
        [DEMAND_CASES, US_PROVIDERS], names=["parameter_case", "provider"]
    )
    frame = frame.reindex(expected)
    for column in US_COST_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    base = frame.xs("base", level="parameter_case")[US_COST_COLUMNS].fillna(0.0)
    imputed_cases: set[str] = set()
    for case in DEMAND_CASES:
        missing = frame.loc[pd.IndexSlice[case, :], US_COST_COLUMNS].isna()
        if bool(missing.to_numpy().any()):
            imputed_cases.add(case)
        for provider in US_PROVIDERS:
            for column in US_COST_COLUMNS:
                key = (case, provider)
                if pd.isna(frame.loc[key, column]):
                    frame.loc[key, column] = float(base.loc[provider, column]) * demand_ratios[case]
    frame[list(US_COST_COLUMNS)] = frame[list(US_COST_COLUMNS)].astype(float)
    frame["cpu_server_price_case"] = "base"
    return frame.reset_index(), imputed_cases


def prepare(
    core_path: Path,
    china_low_path: Path,
    china_high_path: Path,
    defaults_path: Path,
    routing_path: Path,
    service_path: Path,
    workload_path: Path,
    baseline_path: Path,
    api_price_path: Path,
    us_national_path: Path,
    us_validation_path: Path,
    us_demand_service_path: Path,
    us_cost_config_path: Path,
    us_parameters_path: Path,
    model_version: str,
) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_path, encoding="utf-8-sig")
    industries = baseline["industry_code"].astype(str).tolist()
    service = pd.read_csv(service_path, encoding="utf-8-sig")
    china_national_demand_ratios = _national_demand_ratios(service)
    china_demand_ratios = _industry_demand_ratios(service, industries)
    china_frames = {
        "low": _read_csv_or_empty(china_low_path),
        "base": _read_csv_or_empty(core_path),
        "high": _read_csv_or_empty(china_high_path),
    }
    china_ig, china_imputed_cases = _prepare_china_cases(
        china_frames, industries, china_demand_ratios
    )

    defaults = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
    routing_doc = yaml.safe_load(routing_path.read_text(encoding="utf-8"))
    active_route = routing_doc["routing_cases"][routing_doc["active_core_routing_case"]]
    discount = float(defaults["model"]["discount_rate"])
    gpu_annual = _annual_server_cost(defaults["server"], discount)
    cpu_annual = _annual_server_cost(routing_doc["cpu_server"], discount)
    workload = pd.read_csv(workload_path, encoding="utf-8-sig")
    prices = pd.read_csv(api_price_path, encoding="utf-8-sig")
    prices = prices[
        prices["provider"].isin(CHINA_PROVIDERS)
        & _as_bool(prices["active_for_baseline"])
        & _as_bool(prices["mainstream_representative"])
    ]
    if set(prices["provider"]) != set(CHINA_PROVIDERS) or prices["provider"].duplicated().any():
        raise ValueError("Expected one active price for each China provider")
    usd_fx = pd.to_numeric(
        prices.loc[prices["currency"].eq("USD"), "fx_to_cny"], errors="coerce"
    ).dropna().unique()
    if len(usd_fx) != 1 or float(usd_fx[0]) <= 0:
        raise ValueError("Expected one positive CNY-per-USD exchange rate in active price inputs")
    cny_per_usd = float(usd_fx[0])

    cloud_cfg = defaults["hybrid_cloud"]
    gpu_subscription = float(cloud_cfg["gpu_annual_subscription_rmb_per_group"])
    cpu_subscription = float(cloud_cfg["cpu_annual_subscription_rmb_per_group"])
    cpu_relative_capacity = float(cloud_cfg["cpu_capacity_local_server_h_per_h"])
    rows: list[dict[str, object]] = []
    china_cloud_components: dict[str, dict[str, dict[str, float]]] = {}

    for demand_case in DEMAND_CASES:
        ig = china_ig[demand_case].set_index("industry")
        rows.append({
            "panel": "a", "country": "China", "demand_case": demand_case,
            "option": "cn_local", "label": "本地自建", "component": "总成本",
            "value": float(ig["industry_equivalent_annual_incremental_total_cost_rmb"].sum()),
            "unit": "RMB/year",
            "evidence_status": "baseline_scaled_by_demand_ratio" if demand_case in china_imputed_cases else "available_IG_1host",
        })
        provider_values = {
            provider: {"GPU容量付款": 0.0, "CPU容量付款": 0.0, "Token API": 0.0}
            for provider in CHINA_PROVIDERS
        }
        case_service = service[service["parameter_case"].eq(demand_case)]
        for industry in baseline.itertuples(index=False):
            code = str(industry.industry_code)
            industry_service = case_service[case_service["industry_code"].eq(code)]
            if set(industry_service["task_id"]) != {
                "office", "agent", "vision", "maintenance", "scheduling", "simulation"
            }:
                raise ValueError(f"Service input incomplete for {code}/{demand_case}")
            gpu_service = {
                item.task_id: float(item.effective_service_units_day)
                * (1.0 - float(active_route.get(item.task_id, 0.0)))
                for item in industry_service.itertuples(index=False)
            }
            residual_gpu_share = sum(
                value for task, value in gpu_service.items() if task not in {"office", "agent"}
            ) / sum(gpu_service.values())
            gpu_payment = (
                float(ig.loc[code, "industry_equivalent_installed_gpu_server_groups"])
                * residual_gpu_share * gpu_subscription
            )
            cpu_payment = (
                float(ig.loc[code, "industry_equivalent_installed_cpu_server_groups"])
                / cpu_relative_capacity * cpu_subscription
            )
            large_share = (
                float(industry.large_enterprise_share)
                if pd.notna(industry.large_enterprise_share)
                else float(defaults["api_token_cost"]["large_workload_weight_when_unobserved"])
            )
            annual_input, annual_output = _industry_token_demand(
                workload, code, float(industry.employment_2023_10k_person) * 10_000.0,
                large_share, demand_case,
            )
            for price in prices.itertuples(index=False):
                token_payment = (
                    annual_input / 1_000_000.0 * float(price.input_per_mtoken)
                    + annual_output / 1_000_000.0 * float(price.output_per_mtoken)
                ) * float(price.fx_to_cny)
                provider_values[price.provider]["GPU容量付款"] += gpu_payment
                provider_values[price.provider]["CPU容量付款"] += cpu_payment
                provider_values[price.provider]["Token API"] += token_payment
        china_cloud_components[demand_case] = provider_values
        for provider in CHINA_PROVIDERS:
            option = "cn_cloud_deepseek" if provider == "DeepSeek" else "cn_cloud_alibaba"
            rows.append({
                "panel": "a", "country": "China", "demand_case": demand_case,
                "option": option, "label": "DeepSeek*" if provider == "DeepSeek" else "阿里云*",
                "component": "总成本", "value": sum(provider_values[provider].values()),
                "unit": "RMB/year",
                "evidence_status": "partly_baseline_scaled_by_demand_ratio" if demand_case in china_imputed_cases else "provisional_china_cloud_price_proxy",
            })

    base_ig = china_ig["base"]
    china_local_components = {
        "GPU服务器及设施": float(base_ig["industry_equivalent_installed_gpu_server_groups"].sum()) * gpu_annual,
        "CPU服务器及设施": float(base_ig["industry_equivalent_installed_cpu_server_groups"].sum()) * cpu_annual,
        "电量": float(base_ig["industry_equivalent_annual_ai_energy_cost_rmb"].sum()),
        "最大需量": float(base_ig["industry_equivalent_annual_incremental_maximum_demand_cost_rmb"].sum()),
    }
    for component, value in china_local_components.items():
        rows.append({
            "panel": "c", "country": "China", "demand_case": "base",
            "option": "cn_local", "label": "本地自建", "component": component,
            "value": value, "unit": "RMB/year",
            "evidence_status": "baseline_scaled_by_demand_ratio" if "base" in china_imputed_cases else "available_IG_1host",
        })
    for provider in CHINA_PROVIDERS:
        option = "cn_cloud_deepseek" if provider == "DeepSeek" else "cn_cloud_alibaba"
        for component, value in china_cloud_components["base"][provider].items():
            rows.append({
                "panel": "c", "country": "China", "demand_case": "base",
                "option": option, "label": "DeepSeek*" if provider == "DeepSeek" else "阿里云*",
                "component": component, "value": value, "unit": "RMB/year",
                "evidence_status": "provisional_china_cloud_price_proxy",
            })

    us_demand_service = pd.read_csv(us_demand_service_path, encoding="utf-8-sig")
    us_demand_ratios = _national_demand_ratios(us_demand_service)
    china_demand_totals = _national_demand_totals(service)
    us_demand_totals = _national_demand_totals(us_demand_service)
    us_results_ready = _us_results_ready(us_national_path, us_validation_path)
    us_national, us_imputed_cases = _prepare_us_cases(
        _read_csv_or_empty(us_national_path), us_demand_ratios
    )

    us_config = yaml.safe_load(us_cost_config_path.read_text(encoding="utf-8"))["us_cost_environment"]
    us_parameters = pd.read_csv(us_parameters_path, encoding="utf-8-sig").set_index("parameter_id")

    def us_parameter(parameter_id: str) -> float:
        return float(us_parameters.loc[parameter_id, "base_value"])

    us_coeff = (
        (1.0 + float(us_config["shared_facility_capex_fraction"]))
        * _crf(float(us_config["shared_discount_rate"]), float(us_config["shared_economic_life_years"]))
        + float(us_config["shared_annual_maintenance_fraction"])
    )
    inferred_us_local_components: dict[str, dict[str, float]] = {}
    inferred_us_cloud_components: dict[str, dict[str, dict[str, float]]] = {}
    if not us_results_ready:
        us_gpu_annual = us_parameter(str(us_config["local_gpu_server_parameter_id"])) * us_coeff
        us_cpu_annual = us_parameter(str(us_config["local_cpu_server_parameter_id"])) * us_coeff
        us_electricity = us_parameter(str(us_config["industrial_electricity_parameter_id"]))
        us_cloud_gpu_annual = us_parameter(str(us_config["cloud_gpu_reserved_parameter_id"]))
        us_cloud_cpu_annual = us_parameter(str(us_config["cloud_cpu_reserved_parameter_id"]))
        for demand_case in DEMAND_CASES:
            demand_scale = us_demand_totals[demand_case] / china_demand_totals[demand_case]
            ig = china_ig[demand_case]
            components = {
                "GPU服务器及设施": float(ig["industry_equivalent_installed_gpu_server_groups"].sum())
                * demand_scale * us_gpu_annual,
                "CPU服务器及设施": float(ig["industry_equivalent_installed_cpu_server_groups"].sum())
                * demand_scale * us_cpu_annual,
                "电量": float(ig["industry_equivalent_annual_ai_facility_energy_twh"].sum())
                * demand_scale * 1e9 * us_electricity,
            }
            inferred_us_local_components[demand_case] = components
            mask = us_national["parameter_case"].eq(demand_case)
            us_national.loc[mask, "local_total_annual_cost_usd"] = sum(components.values())
            china_cloud_reference = china_cloud_components[demand_case][CHINA_PROVIDERS[0]]
            gpu_reserved = (
                float(china_cloud_reference["GPU容量付款"])
                / gpu_subscription * demand_scale
            )
            cpu_reserved = (
                float(china_cloud_reference["CPU容量付款"])
                / cpu_subscription * demand_scale
            )
            inferred_us_cloud_components[demand_case] = {}
            for provider in US_PROVIDERS:
                provider_mask = mask & us_national["provider"].eq(provider)
                token_cost = float(
                    us_national.loc[provider_mask, "cloud_token_api_cost_usd"].iloc[0]
                )
                if not np.isfinite(token_cost) or token_cost < 0.0:
                    raise ValueError(
                        f"A valid existing U.S. Token API cost is required for {provider}/{demand_case}"
                    )
                cloud_components = {
                    "GPU容量付款": gpu_reserved * us_cloud_gpu_annual,
                    "CPU容量付款": cpu_reserved * us_cloud_cpu_annual,
                    "Token API": token_cost,
                }
                inferred_us_cloud_components[demand_case][provider] = cloud_components
                us_national.loc[provider_mask, "cloud_gpu_reserved_instances"] = gpu_reserved
                us_national.loc[provider_mask, "cloud_cpu_reserved_instances"] = cpu_reserved
                us_national.loc[provider_mask, "cloud_gpu_reserved_cost_usd"] = cloud_components["GPU容量付款"]
                us_national.loc[provider_mask, "cloud_cpu_reserved_cost_usd"] = cloud_components["CPU容量付款"]
                us_national.loc[provider_mask, "cloud_total_annual_cost_usd"] = sum(cloud_components.values())

    us_local_status = (
        "available_us_native_demand_heterogeneous_hardware"
        if us_results_ready
        else "china_capacity_scaled_by_us_to_china_effective_service_ratio_and_repriced_with_us_local_inputs"
    )
    us_cloud_status = (
        "available_us_native_demand_cloud_payment_proxy"
        if us_results_ready
        else "china_cloud_capacity_scaled_by_us_to_china_effective_service_ratio_and_repriced_with_us_cloud_inputs"
    )
    for demand_case in DEMAND_CASES:
        selected = us_national[us_national["parameter_case"].eq(demand_case)]
        if selected["local_total_annual_cost_usd"].round(2).nunique() != 1:
            raise ValueError(f"U.S. local cost differs by provider for {demand_case}")
        rows.append({
            "panel": "b", "country": "US", "demand_case": demand_case,
            "option": "us_local", "label": "本地自建", "component": "总成本",
            "value": float(selected.iloc[0]["local_total_annual_cost_usd"]),
            "unit": "USD/year",
            "evidence_status": us_local_status,
        })
        for item in selected.itertuples(index=False):
            rows.append({
                "panel": "b", "country": "US", "demand_case": demand_case,
                "option": f"us_cloud_{str(item.provider).lower()}", "label": item.provider,
                "component": "总成本", "value": float(item.cloud_total_annual_cost_usd),
                "unit": "USD/year",
                "evidence_status": us_cloud_status,
            })

    us_base = us_national[us_national["parameter_case"].eq("base")]
    us_reference = us_base.iloc[0]
    us_local_components = (
        inferred_us_local_components["base"]
        if not us_results_ready
        else {
            "GPU服务器及设施": float(us_reference["local_gpu_servers"])
            * us_parameter(str(us_config["local_gpu_server_parameter_id"])) * us_coeff,
            "CPU服务器及设施": float(us_reference["local_cpu_servers"])
            * us_parameter(str(us_config["local_cpu_server_parameter_id"])) * us_coeff,
            "电量": float(us_reference["annual_facility_energy_twh"]) * 1e9
            * us_parameter(str(us_config["industrial_electricity_parameter_id"])),
        }
    )
    if abs(sum(us_local_components.values()) - float(us_reference["local_total_annual_cost_usd"])) > 1.0:
        raise ValueError("U.S. local components do not reconcile")
    for component, value in us_local_components.items():
        rows.append({
            "panel": "d", "country": "US", "demand_case": "base",
            "option": "us_local", "label": "本地自建", "component": component,
            "value": value, "unit": "USD/year",
            "evidence_status": us_local_status,
        })
    for item in us_base.itertuples(index=False):
        for component, value in (
            ("GPU容量付款", float(item.cloud_gpu_reserved_cost_usd)),
            ("CPU容量付款", float(item.cloud_cpu_reserved_cost_usd)),
            ("Token API", float(item.cloud_token_api_cost_usd)),
        ):
            rows.append({
                "panel": "d", "country": "US", "demand_case": "base",
                "option": f"us_cloud_{str(item.provider).lower()}", "label": item.provider,
                "component": component, "value": value, "unit": "USD/year",
                "evidence_status": us_cloud_status,
            })

    output = pd.DataFrame(rows)
    output.insert(0, "model_version", model_version)
    output["value_local_currency"] = output["value"]
    output["local_currency_unit"] = output["unit"]
    output["local_currency_per_usd"] = np.where(output["country"].eq("China"), cny_per_usd, 1.0)
    output["value_usd"] = output["value_local_currency"] / output["local_currency_per_usd"]
    output["national_demand_ratio_to_base"] = output.apply(
        lambda row: china_national_demand_ratios[row["demand_case"]]
        if row["country"] == "China"
        else us_demand_ratios[row["demand_case"]],
        axis=1,
    )
    local_fallback = output["evidence_status"].eq(
        "china_capacity_scaled_by_us_to_china_effective_service_ratio_and_repriced_with_us_local_inputs"
    )
    cloud_fallback = output["evidence_status"].eq(
        "china_cloud_capacity_scaled_by_us_to_china_effective_service_ratio_and_repriced_with_us_cloud_inputs"
    )
    output["source_note"] = ""
    output.loc[local_fallback, "source_note"] = (
        "U.S. local fallback: China same-demand installed capacity and energy, scaled by the "
        "U.S./China effective-service demand ratio and repriced with U.S. GPU, CPU, and electricity inputs"
    )
    output.loc[cloud_fallback, "source_note"] = (
        "U.S. cloud fallback: China same-demand GPU/CPU cloud capacity, scaled by the U.S./China "
        "effective-service demand ratio and repriced with U.S. cloud inputs; Token API retains the "
        "available U.S. provider estimate"
    )
    return output


def _style(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(colors="#333333")
    ax.grid(axis=grid_axis, color="#D7D7D7", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)


def _plot_cost_by_demand(
    ax: plt.Axes, data: pd.DataFrame, local_options: list[str], cloud_options: list[str],
    divisor: float, ylabel: str, title: str,
) -> None:
    pivot = data.pivot(index="option", columns="demand_case", values="value") / divisor
    x = np.arange(len(DEMAND_CASES), dtype=float)
    local = pivot.loc[local_options, DEMAND_CASES].to_numpy(dtype=float)
    cloud = pivot.loc[cloud_options, DEMAND_CASES].to_numpy(dtype=float)
    local_min = local.min(axis=0)
    local_max = local.max(axis=0)
    cloud_min = cloud.min(axis=0)
    cloud_max = cloud.max(axis=0)
    local_x = x - 0.035
    cloud_x = x + 0.035

    ax.fill_between(local_x, local_min, local_max, color=COLORS["base"], alpha=0.18, zorder=1)
    ax.plot(local_x, local_min, color=COLORS["base"], linewidth=2.0, marker="o", markersize=5, zorder=3)
    ax.plot(local_x, local_max, color=COLORS["base"], linewidth=2.0, marker="o", markersize=5, zorder=3)
    ax.vlines(local_x, local_min, local_max, color=COLORS["base"], linewidth=4.0, alpha=0.8, zorder=2)
    ax.hlines(local_min, local_x - 0.055, local_x + 0.055, color=COLORS["base"], linewidth=1.2, zorder=3)
    ax.hlines(local_max, local_x - 0.055, local_x + 0.055, color=COLORS["base"], linewidth=1.2, zorder=3)

    ax.fill_between(cloud_x, cloud_min, cloud_max, color=COLORS["high"], alpha=0.18, zorder=1)
    ax.plot(cloud_x, cloud_min, color=COLORS["high"], linewidth=2.0, marker="o", markersize=5, zorder=3)
    ax.plot(cloud_x, cloud_max, color=COLORS["high"], linewidth=2.0, marker="o", markersize=5, zorder=3)
    ax.vlines(cloud_x, cloud_min, cloud_max, color=COLORS["high"], linewidth=4.0, alpha=0.8, zorder=2)
    ax.hlines(cloud_min, cloud_x - 0.055, cloud_x + 0.055, color=COLORS["high"], linewidth=1.2, zorder=3)
    ax.hlines(cloud_max, cloud_x - 0.055, cloud_x + 0.055, color=COLORS["high"], linewidth=1.2, zorder=3)
    for provider_index, provider_values in enumerate(cloud):
        offset = (provider_index - (len(cloud_options) - 1) / 2.0) * 0.035
        ax.scatter(
            cloud_x + offset, provider_values, s=26, color=COLORS["high"],
            edgecolor="white", linewidth=0.45, zorder=4,
        )
    label_offset = max(float(local_max.max()), float(cloud_max.max())) * 0.025
    for xx, low, high in zip(local_x, local_min, local_max):
        ax.text(xx, high + label_offset, f"{low:.1f}–{high:.1f}", ha="right", va="bottom", fontsize=10)
    for xx, low, high in zip(cloud_x, cloud_min, cloud_max):
        ax.text(xx, high + label_offset, f"{low:.1f}–{high:.1f}", ha="left", va="bottom", fontsize=10)
    ax.set_xticks(x, ["低需求", "中需求", "高需求"])
    ax.set_xlabel("AI服务需求水平")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=13)
    ax.set_ylim(0, max(float(local_max.max()), float(cloud_max.max())) * 1.18)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="-", linewidth=2.0,
                   color=COLORS["base"], label="本地自建范围", markersize=5),
            Line2D([0], [0], marker="o", linestyle="-", linewidth=2.0,
                   color=COLORS["high"], label="云端采购范围", markersize=5),
        ], frameon=False, ncol=2, fontsize=10, loc="upper left",
    )
    _style(ax, "y")


def _plot_composition(
    ax: plt.Axes, data: pd.DataFrame, options: list[str], labels: list[str],
    divisor: float, ylabel: str, title: str, local_count: int,
) -> None:
    components = [
        ("GPU服务器及设施", COLORS["gpu"]), ("CPU服务器及设施", COLORS["cpu"]),
        ("电量", COLORS["energy"]), ("最大需量", COLORS["demand"]),
        ("GPU容量付款", COLORS["gpu"]), ("CPU容量付款", COLORS["cpu"]),
        ("Token API", COLORS["token"]),
    ]
    x = np.arange(len(options), dtype=float)
    bottom = np.zeros(len(options))
    for component, color in components:
        values = np.array([
            data[data["option"].eq(option) & data["component"].eq(component)]["value"].sum() / divisor
            for option in options
        ])
        bars = ax.bar(x, values, bottom=bottom, width=0.68, color=color)
        if component in {"GPU容量付款", "CPU容量付款", "Token API"}:
            for bar in bars[local_count:]:
                bar.set_hatch("///")
                bar.set_edgecolor("white")
                bar.set_linewidth(0.5)
        bottom += values
    for xx, total in zip(x, bottom):
        ax.text(xx, total + max(bottom) * 0.025, f"{total:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=13)
    ax.set_ylim(0, max(bottom) * 1.19)
    _style(ax, "y")


def plot(data: pd.DataFrame, outputs: list[Path]) -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False, "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 12,
    })
    plot_data = data.copy()
    plot_data["value"] = plot_data["value_usd"]
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.2))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.13, wspace=0.25, hspace=0.38)
    cn_options = ["cn_local", "cn_cloud_deepseek", "cn_cloud_alibaba"]
    cn_labels = ["本地自建\n集团单节点", "DeepSeek*", "阿里云*"]
    us_options = ["us_local", "us_cloud_google", "us_cloud_openai", "us_cloud_anthropic"]
    us_labels = ["本地自建\n行业分别定容", "Google", "OpenAI", "Anthropic"]

    _plot_cost_by_demand(
        axes[0, 0], plot_data[plot_data["panel"].eq("a")], cn_options[:1], cn_options[1:], 1e9,
        "企业年成本/付款（十亿美元/年）", "a  中国：不同需求水平下的本地与云端成本",
    )
    _plot_cost_by_demand(
        axes[0, 1], plot_data[plot_data["panel"].eq("b")], us_options[:1], us_options[1:], 1e9,
        "企业年成本/付款（十亿美元/年）", "b  美国：不同需求水平下的本地与云端成本",
    )
    top_max = plot_data[plot_data["panel"].isin(["a", "b"])]["value"].max() / 1e9
    for ax in axes[0]:
        ax.set_ylim(0, top_max * 1.18)
    _plot_composition(
        axes[1, 0], plot_data[plot_data["panel"].eq("c")], cn_options, cn_labels, 1e9,
        "企业年成本/付款（十亿美元/年）", "c  中国：基准需求下的成本构成", 1,
    )
    _plot_composition(
        axes[1, 1], plot_data[plot_data["panel"].eq("d")], us_options, us_labels, 1e9,
        "企业年成本/付款（十亿美元/年）", "d  美国：基准需求下的成本构成", 1,
    )
    bottom_max = (
        plot_data[plot_data["panel"].isin(["c", "d"])]
        .groupby(["panel", "option"])["value"]
        .sum()
        .max()
        / 1e9
    )
    for ax in axes[1]:
        ax.set_ylim(0, bottom_max * 1.19)
    fig.legend(
        handles=[
            Patch(facecolor=COLORS["gpu"], label="GPU服务器/容量"),
            Patch(facecolor=COLORS["cpu"], label="CPU服务器/容量"),
            Patch(facecolor=COLORS["energy"], label="电量"),
            Patch(facecolor=COLORS["demand"], label="最大需量"),
            Patch(facecolor=COLORS["token"], hatch="///", edgecolor="white", label="Token API"),
        ], frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.057), fontsize=10,
    )
    if data["evidence_status"].str.contains("china_.*capacity_scaled_by_us_to_china", regex=True).any():
        fig.text(
            0.5, 0.018,
            "注：美国本地及云容量成本为后备推算值——按中美有效服务需求比缩放中国同需求情景的装机、能耗和云容量，并采用美国价格重新计价；Token API沿用美国厂商估计，正式结果完成后自动替换。",
            ha="center", va="bottom", fontsize=8.8, color="#555555",
        )
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, default=VERSION_RESULT_ROOT / "group_architecture_core/national/core_scenarios.csv")
    parser.add_argument("--china-low", type=Path, default=SENSITIVITY_VERSION_ROOT / "national_oat_group_architecture/result/PHY01__low/core_scenarios.csv")
    parser.add_argument("--china-high", type=Path, default=SENSITIVITY_VERSION_ROOT / "national_oat_group_architecture/result/PHY01__high/core_scenarios.csv")
    parser.add_argument("--defaults", type=Path, default=DEFAULTS_PATH)
    parser.add_argument("--routing", type=Path, default=ROOT / "config/compute_hardware/cpu_gpu_routing_v1.yaml")
    parser.add_argument("--service", type=Path, default=ROOT / "02_data/processed/effective_service/manufacturing_ai_effective_service_2030.csv")
    parser.add_argument("--workload", type=Path, default=ROOT / "02_data/raw/curated/china_manufacturing_ai_workload_parameters.csv")
    parser.add_argument("--baseline", type=Path, default=ROOT / "02_data/china_manufacturing_sector_baseline.csv")
    parser.add_argument("--api-prices", type=Path, default=ROOT / "02_data/processed/api_token_cost/api_token_prices_v1_1.csv")
    parser.add_argument("--us-national", type=Path, default=VERSION_RESULT_ROOT / "cost_benchmark/heterogeneous_hardware_v1/us_naics3/us_national_comparison.csv")
    parser.add_argument("--us-validation", type=Path, default=VERSION_RESULT_ROOT / "cost_benchmark/heterogeneous_hardware_v1/us_naics3/validated.done.json")
    parser.add_argument("--us-demand-service", type=Path, default=ROOT / "02_data/processed/us_demand/us_manufacturing_ai_effective_service_2030.csv")
    parser.add_argument("--us-cost-config", type=Path, default=ROOT / "config/cost_cases/us_heterogeneous_cpu_v1.yaml")
    parser.add_argument("--us-parameters", type=Path, default=ROOT / "02_data/processed/cost_benchmark/us_core_cost_parameters_v1.csv")
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--data-output", type=Path, default=RESULT_ROOT / "figure2_enterprise_cost_data.csv")
    parser.add_argument("--png-output", type=Path, default=RESULT_ROOT / "figure2_enterprise_cost.png")
    parser.add_argument("--pdf-output", type=Path, default=RESULT_ROOT / "figure2_enterprise_cost.pdf")
    parser.add_argument("--svg-output", type=Path, default=RESULT_ROOT / "figure2_enterprise_cost.svg")
    parser.add_argument("--validation-output", type=Path, default=RESULT_ROOT / "figure2_enterprise_cost.validated.done.json")
    args = parser.parse_args()
    data = prepare(
        args.core, args.china_low, args.china_high, args.defaults, args.routing,
        args.service, args.workload, args.baseline, args.api_prices,
        args.us_national, args.us_validation, args.us_demand_service, args.us_cost_config,
        args.us_parameters, args.model_version,
    )
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.data_output, index=False, encoding="utf-8-sig")
    plot(data, [args.png_output, args.pdf_output, args.svg_output])
    metadata = {
        "status": "iterative_draft_not_release_ready",
        "model_version": args.model_version,
        "panels": ["china_demand_range", "us_demand_range", "china_base_cost_composition", "us_base_cost_composition"],
        "china_local_evidence": "available_31_industry_IG_1host_with_baseline_demand_scaling_fallback",
        "china_cloud_evidence": "provisional_current_price_proxy",
        "us_evidence": (
            "available_21_naics_native_demand_results"
            if not data["evidence_status"].str.contains("china_capacity_scaled", na=False).any()
            else "us_local_and_cloud_capacity_fallback_from_china_scaled_by_demand_and_repriced_with_us_inputs"
        ),
        "baseline_scaled_output_rows": int(data["evidence_status"].str.contains("baseline_scaled", na=False).sum()),
        "missing_value_fallback": "base_case_value_times_same_country_effective_service_ratio;_unfinished_us_local_and_cloud_capacity_use_china_capacity_times_cross_country_demand_ratio_and_us_prices;_us_token_api_retains_available_provider_estimate",
        "china_scaling_level": "industry_specific_effective_service_ratio",
        "us_scaling_level": "national_effective_service_ratio",
        "china_national_demand_ratios": {
            case: float(data.loc[(data["country"].eq("China")) & (data["demand_case"].eq(case)), "national_demand_ratio_to_base"].iloc[0])
            for case in DEMAND_CASES
        },
        "us_national_demand_ratios": {
            case: float(data.loc[(data["country"].eq("US")) & (data["demand_case"].eq(case)), "national_demand_ratio_to_base"].iloc[0])
            for case in DEMAND_CASES
        },
        "currency": "USD/year",
        "china_cny_per_usd": float(data.loc[data["country"].eq("China"), "local_currency_per_usd"].iloc[0]),
        "exchange_rate_source": "active USD-denominated row in api_token_prices input",
        "excluded": ["zero_production_load_counterfactual", "break_even_thresholds", "mechanism_decomposition"],
        "manuscript_numbers_updated": False,
    }
    args.validation_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Figure 2 written to {args.png_output}")


if __name__ == "__main__":
    main()
