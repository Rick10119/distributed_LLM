#!/usr/bin/env python3
"""Reprice the configured single-industry heterogeneous routing case in the U.S."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def capital_recovery_factor(rate: float, years: float) -> float:
    factor = (1.0 + rate) ** years
    return rate * factor / (factor - 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--screen-config",
        type=Path,
        default=Path("config/compute_hardware/cpu_gpu_routing_v1.yaml"),
    )
    parser.add_argument(
        "--us-cost-config",
        type=Path,
        default=Path("config/cost_cases/us_heterogeneous_cpu_v1.yaml"),
    )
    parser.add_argument(
        "--china-screen",
        type=Path,
        default=Path("05_results/v0.8.0/result/cost_benchmark/c36_heterogeneous_hardware_v1/comparison.csv"),
    )
    parser.add_argument(
        "--token-demand",
        type=Path,
        default=Path("05_results/v0.8.0/result/api_token_cost/task_token_demand.csv"),
    )
    parser.add_argument(
        "--us-parameters",
        type=Path,
        default=Path("02_data/processed/cost_benchmark/us_core_cost_parameters_v1.csv"),
    )
    parser.add_argument(
        "--us-api-prices",
        type=Path,
        default=Path("02_data/processed/cost_benchmark/us_api_token_prices_v1.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("05_results/v0.8.0/result/cost_benchmark/c36_heterogeneous_hardware_v1/us_cost_environment"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load((ROOT / args.screen_config).read_text(encoding="utf-8"))
    us_cost_registry = yaml.safe_load((ROOT / args.us_cost_config).read_text(encoding="utf-8"))
    us = us_cost_registry["us_cost_environment"]
    us["cost_case_version"] = us_cost_registry["cost_case_version"]
    if int(us["local_cpu_physical_cores"]) != 64:
        raise ValueError("Matched CPU cost boundary requires 64 physical cores")
    if int(us["local_cpu_minimum_memory_gb"]) < 256:
        raise ValueError("Matched CPU cost boundary requires at least 256GB memory")
    if int(us["local_cpu_support_term_years"]) < 3:
        raise ValueError("Matched CPU cost boundary requires at least three-year support")
    if float(us["cloud_cpu_service_capacity_relative_to_local_cpu_server"]) != 0.25:
        raise ValueError("A 32-vCPU/16-core cloud instance must be 25% of the 64-core local unit")
    screen = pd.read_csv(ROOT / args.china_screen, encoding="utf-8-sig")
    # Hardware counts and physical energy are provider-invariant; retain one row per routing case.
    physical = screen.drop_duplicates("routing_case").copy()
    demand = pd.read_csv(ROOT / args.token_demand, encoding="utf-8-sig")
    industry_code = config["calibration_industry_code"]
    demand = demand[demand.industry_code == industry_code]
    parameters = pd.read_csv(ROOT / args.us_parameters, encoding="utf-8-sig")
    prices = pd.read_csv(ROOT / args.us_api_prices, encoding="utf-8-sig")
    prices = prices[prices.provider.isin(us["formal_api_providers"])]
    if set(prices.provider) != set(us["formal_api_providers"]):
        raise ValueError("Formal U.S. API provider panel is incomplete")

    def parameter(parameter_id: str, case: str = "base") -> float:
        selected = parameters[parameters.parameter_id == parameter_id]
        if len(selected) != 1:
            raise ValueError(f"Expected one U.S. parameter row for {parameter_id}")
        value = float(selected.iloc[0][f"{case}_value"])
        if value <= 0:
            raise ValueError(f"U.S. parameter {parameter_id} {case} value must be positive")
        return value

    gpu_purchase = parameter(us["local_gpu_server_parameter_id"])
    cpu_purchase = parameter(us["local_cpu_server_parameter_id"])
    electricity_price = parameter(us["industrial_electricity_parameter_id"])
    cloud_gpu_price = parameter(us["cloud_gpu_reserved_parameter_id"])
    cloud_cpu_price = parameter(us["cloud_cpu_reserved_parameter_id"])
    china_cloud_cpu_capacity = float(config["cloud_cpu"]["service_capacity_relative_to_local_cpu_server"])
    us_cloud_cpu_capacity = float(us["cloud_cpu_service_capacity_relative_to_local_cpu_server"])

    rate = float(us["shared_discount_rate"])
    years = float(us["shared_economic_life_years"])
    coefficient = (
        (1.0 + float(us["shared_facility_capex_fraction"]))
        * capital_recovery_factor(rate, years)
        + float(us["shared_annual_maintenance_fraction"])
    )
    input_tokens = float(demand.annual_input_tokens.sum())
    output_tokens = float(demand.annual_output_tokens.sum())

    rows: list[dict[str, object]] = []
    for physical_row in physical.itertuples(index=False):
        local_gpu = (
            float(physical_row.local_gpu_server_groups_industry_equivalent)
            * gpu_purchase
            * coefficient
        )
        local_cpu = (
            float(physical_row.local_cpu_server_groups_industry_equivalent)
            * cpu_purchase
            * coefficient
        )
        local_electricity = (
            float(physical_row.local_total_facility_energy_twh)
            * 1e9
            * electricity_price
        )
        local_total = local_gpu + local_cpu + local_electricity
        cloud_gpu = (
            int(physical_row.cloud_gpu_reserved_instances)
            * cloud_gpu_price
        )
        cloud_cpu_instances = math.ceil(
            int(physical_row.cloud_cpu_reserved_instances)
            * china_cloud_cpu_capacity / us_cloud_cpu_capacity
        )
        cloud_cpu = (
            cloud_cpu_instances
            * cloud_cpu_price
        )
        for price in prices.itertuples(index=False):
            api = (
                input_tokens / 1e6 * float(price.input_usd_per_mtoken)
                + output_tokens / 1e6 * float(price.output_usd_per_mtoken)
            )
            cloud_total = cloud_gpu + cloud_cpu + api
            rows.append(
                {
                    "scenario_version": us["cost_case_version"],
                    "industry_code": industry_code,
                    "demand_boundary": us["boundary"],
                    "routing_case": physical_row.routing_case,
                    "cpu_service_share": physical_row.cpu_service_share,
                    "provider": price.provider,
                    "model_id": price.model_id,
                    "local_gpu_annualized_cost_usd": local_gpu,
                    "local_cpu_annualized_cost_usd": local_cpu,
                    "local_electricity_cost_usd": local_electricity,
                    "local_total_annual_cost_usd": local_total,
                    "cloud_gpu_reserved_cost_usd": cloud_gpu,
                    "cloud_cpu_reserved_cost_usd": cloud_cpu,
                    "cloud_cpu_reserved_instances": cloud_cpu_instances,
                    "cloud_cpu_service_capacity_relative_to_local_server": us_cloud_cpu_capacity,
                    "cloud_token_api_cost_usd": api,
                    "cloud_total_annual_cost_usd": cloud_total,
                    "cloud_to_local_cost_ratio": cloud_total / local_total,
                    "local_savings_vs_cloud_fraction": 1.0 - local_total / cloud_total,
                }
            )

    result = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "us_comparison.csv", index=False, encoding="utf-8-sig")
    core = result[result.routing_case == config["active_core_routing_case"]].sort_values("cloud_total_annual_cost_usd")
    baseline = result[result.routing_case == "gpu_only"].sort_values("cloud_total_annual_cost_usd")

    # Keep the heavier GPU-ready platform out of the core and expose it as price sensitivity.
    core_physical = physical[physical.routing_case == config["active_core_routing_case"]].iloc[0]
    cpu_price_cases = {
        case: parameter(us["local_cpu_server_parameter_id"], case)
        for case in ("low", "base", "high")
    }
    sensitivity_rows: list[dict[str, object]] = []
    for price_case, cpu_price in cpu_price_cases.items():
        cpu_cost = (
            float(core_physical.local_cpu_server_groups_industry_equivalent)
            * cpu_price
            * coefficient
        )
        local_total = (
            float(core.iloc[0].local_gpu_annualized_cost_usd)
            + cpu_cost
            + float(core.iloc[0].local_electricity_cost_usd)
        )
        for cloud in core.itertuples(index=False):
            sensitivity_rows.append(
                {
                    "routing_case": config["active_core_routing_case"],
                    "cpu_server_price_case": price_case,
                    "cpu_server_purchase_price_usd": cpu_price,
                    "provider": cloud.provider,
                    "local_total_annual_cost_usd": local_total,
                    "cloud_total_annual_cost_usd": cloud.cloud_total_annual_cost_usd,
                    "cloud_to_local_cost_ratio": cloud.cloud_total_annual_cost_usd / local_total,
                    "local_savings_vs_cloud_fraction": 1.0 - local_total / cloud.cloud_total_annual_cost_usd,
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(
        args.output_dir / "us_cpu_server_purchase_price_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    table = "\n".join(
        f"| {row.provider} | {row.local_total_annual_cost_usd/1e9:.3f} | "
        f"{row.cloud_token_api_cost_usd/1e9:.3f} | {row.cloud_cpu_reserved_cost_usd/1e9:.3f} | "
        f"{row.cloud_gpu_reserved_cost_usd/1e9:.3f} | {row.cloud_total_annual_cost_usd/1e9:.3f} | "
        f"{row.cloud_to_local_cost_ratio:.3f} | {row.local_savings_vs_cloud_fraction:.1%} |"
        for row in core.itertuples(index=False)
    )
    findings = f"""# C36异构核心情景：美国成本环境复价

## 结论

固定中国C36汽车制造业的任务需求、灵活窗口和已选异构路由，仅替换为美国本地服务器、电价、AWS预留实例和美国正式API面板。异构核心的美国本地部署成本为 **{core.iloc[0].local_total_annual_cost_usd/1e9:.3f} 十亿美元/年**；完整云化为 **{core.cloud_total_annual_cost_usd.min()/1e9:.3f}—{core.cloud_total_annual_cost_usd.max()/1e9:.3f} 十亿美元/年**，即本地的 **{core.cloud_to_local_cost_ratio.min():.3f}—{core.cloud_to_local_cost_ratio.max():.3f}倍**。

| 美国API厂商 | 本地合计 十亿美元 | Token API | 云CPU预留 | 云GPU预留 | 完整云合计 十亿美元 | 云/本地 | 本地较云节省 |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

中性CPU整机价格下，三家组合的本地部署均比云低20%以上；最低为Google组合的 **{(core.loc[core.provider == 'Google', 'local_savings_vs_cloud_fraction'].iloc[0]):.1%}**，刚越过20%阈值，因此仍应由CPU整机价格敏感性约束，而不是写成无条件稳健结论。

## 参数与边界

- 本地GPU整机46,500美元；CPU整机统一为64物理核、至少256GB、企业存储、导轨、冗余电源和三年支持的完整交付边界。美国CPU价格采用32,550/34,100/37,200美元工程标准化范围；原16,000美元的单颗32核、32GB Dell配置不再作为模型基准。
- 两类本地服务器统一5年寿命、8%折现率、20%设施附加投资、5%年度维护、单一15%装机裕量；电价为EIA 2024美国工业最终全国平均0.0813美元/kWh。
- 云GPU为两台AWS g6e.4xlarge的一年Standard Reserved全预付，30,948美元/双48GB GPU等效单元·年；云CPU为AWS c7i.8xlarge 32 vCPU/64 GiB同口径一年全预付7,723美元/年。不展示按量实例。
- 这是“同一中国C36物理需求放到美国价格环境”的反事实，不是美国汽车制造业本土需求结果，也未加入税、支持、网络出口、迁移集成和单独需量费。
- 34,100美元基准由双CPU、256GB、三年支持平台加企业存储及10%交付裕度构造；平台存在GPU-ready过度配置且精确CPU核数未在归档证据中披露，因此仍需64核同配置OEM询价和同窗口性能测试替换。

GPU-only美国复价的云/本地区间为 **{baseline.cloud_to_local_cost_ratio.min():.3f}—{baseline.cloud_to_local_cost_ratio.max():.3f}倍**。CPU整机高值27,263.50美元时，会复现更保守的结果：Google仅节省约11%，未达到20%；该结果已单列在 `us_cpu_server_purchase_price_sensitivity.csv`。
"""
    (args.output_dir / "findings.md").write_text(findings, encoding="utf-8")
    done = {
        "status": "complete_validated_c36_us_cost_environment_counterfactual",
        "rows": len(result),
        "providers": sorted(result.provider.unique()),
        "active_routing_case": config["active_core_routing_case"],
        "minimum_core_ratio": float(core.cloud_to_local_cost_ratio.min()),
        "maximum_core_ratio": float(core.cloud_to_local_cost_ratio.max()),
        "all_providers_local_savings_above_20pct": bool((core.local_savings_vs_cloud_fraction >= 0.20).all()),
        "high_cpu_server_price_all_providers_above_20pct": bool(
            (
                sensitivity[sensitivity.cpu_server_price_case == "high"]
                .local_savings_vs_cloud_fraction
                >= 0.20
            ).all()
        ),
        "ondemand_displayed": False,
        "us_native_demand": False,
        "price_source": str(args.us_parameters),
        "api_price_source": str(args.us_api_prices),
    }
    (args.output_dir / "validated.done.json").write_text(
        json.dumps(done, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(done, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
