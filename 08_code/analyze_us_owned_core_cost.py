#!/usr/bin/env python3
"""Calculate a fixed-physics U.S. local-server core-cost counterfactual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


SCENARIOS = ("IF", "IG", "II_1host")


def capital_recovery_factor(rate: float, years: float) -> float:
    if rate <= 0 or years <= 0:
        raise ValueError("Discount rate and asset life must be positive")
    factor = (1.0 + rate) ** years
    return rate * factor / (factor - 1.0)


def parameter_row(frame: pd.DataFrame, parameter_id: str) -> pd.Series:
    selected = frame[frame["parameter_id"] == parameter_id]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one parameter row for {parameter_id}")
    return selected.iloc[0]


def adjust_installed_reserve(
    installed_servers: float,
    facility_energy_twh: float,
    original_reserve: float,
    target_reserve: float,
    standby_power_kw: float,
    facility_multiplier: float,
) -> tuple[float, float]:
    """National-scale sensitivity: resize capacity and remove cold-spare energy."""
    if original_reserve < 0 or target_reserve < 0 or target_reserve > original_reserve:
        raise ValueError("Reserve fractions must be non-negative and target cannot exceed original")
    adjusted_servers = installed_servers * (1.0 + target_reserve) / (1.0 + original_reserve)
    removed_servers = installed_servers - adjusted_servers
    removed_energy_twh = removed_servers * standby_power_kw * facility_multiplier * 8760.0 / 1e9
    adjusted_energy = facility_energy_twh - removed_energy_twh
    if adjusted_energy <= 0:
        raise ValueError("Reserve adjustment produced non-positive facility energy")
    return adjusted_servers, adjusted_energy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--national-summary", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--cost-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--findings-output", type=Path, required=True)
    parser.add_argument("--done-output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.cost_config.read_text(encoding="utf-8"))
    national = pd.read_csv(args.national_summary, encoding="utf-8-sig")
    parameters = pd.read_csv(args.parameters, encoding="utf-8-sig")

    model_version = str(config["model_version"])
    cost_case_version = str(config["cost_case_version"])
    if set(national["model_version"]) != {model_version} or len(national) != 93:
        raise ValueError("U.S. cost case requires the active 31-industry, three-architecture summary")
    if set(national["scenario"]) != set(SCENARIOS):
        raise ValueError("National summary must contain IF, IG, and II_1host")

    server_parameter = parameter_row(parameters, str(config["server_purchase_parameter_id"]))
    electricity_parameter = parameter_row(parameters, str(config["electricity_parameter_id"]))
    if server_parameter["parameter_status"] != "engineering_BOM_proxy_not_observed_complete_quote":
        raise ValueError("U.S. server price must retain its engineering BOM evidence label")
    electricity_price = float(electricity_parameter[str(config["electricity_price_case"]) + "_value"])
    if electricity_price <= 0:
        raise ValueError("U.S. electricity price must be positive")

    rate = float(config["discount_rate"])
    life = float(config["economic_life_years"])
    facility_fraction = float(config["facility_capex_fraction"])
    maintenance_fraction = float(config["annual_maintenance_fraction"])
    crf = capital_recovery_factor(rate, life)
    annualized_purchase_coefficient = (1.0 + facility_fraction) * crf + maintenance_fraction
    energy_column = str(config["physical_energy_column"])
    original_reserve = float(config["original_installed_reserve_fraction"])
    target_reserve = float(config["us_installed_reserve_fraction"])
    standby_power_kw = float(config["cold_spare_standby_power_kw"])
    facility_multiplier = float(config["marginal_facility_multiplier"])

    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        selected = national[national["scenario"] == scenario]
        baseline_installed_servers = float(
            selected["industry_equivalent_installed_server_groups"].sum()
        )
        baseline_facility_energy_twh = float(selected[energy_column].sum())
        installed_servers, facility_energy_twh = adjust_installed_reserve(
            baseline_installed_servers,
            baseline_facility_energy_twh,
            original_reserve,
            target_reserve,
            standby_power_kw,
            facility_multiplier,
        )
        for price_case in config["server_price_cases"]:
            purchase_price = float(server_parameter[f"{price_case}_value"])
            annual_server_cost = installed_servers * purchase_price * annualized_purchase_coefficient
            annual_electricity_cost = facility_energy_twh * 1e9 * electricity_price
            total = annual_server_cost + annual_electricity_cost
            rows.append(
                {
                    "model_version": model_version,
                    "cost_case_version": cost_case_version,
                    "country": config["country"],
                    "scenario": scenario,
                    "server_price_case": price_case,
                    "model_installed_dual_l20_servers_at_single_reserve_parameter": baseline_installed_servers,
                    "adjusted_installed_dual_l20_servers": installed_servers,
                    "original_installed_reserve_fraction": original_reserve,
                    "us_installed_reserve_fraction": target_reserve,
                    "baseline_annual_ai_facility_energy_twh": baseline_facility_energy_twh,
                    "adjusted_annual_ai_facility_energy_twh": facility_energy_twh,
                    "server_purchase_price_usd": purchase_price,
                    "electricity_price_usd_per_kwh": electricity_price,
                    "capital_recovery_factor": crf,
                    "annualized_purchase_and_maintenance_coefficient": annualized_purchase_coefficient,
                    "annual_server_capital_facility_maintenance_cost_usd": annual_server_cost,
                    "annual_electricity_cost_usd": annual_electricity_cost,
                    "annual_us_owned_core_cost_usd": total,
                    "annual_us_owned_core_cost_billion_usd": total / 1e9,
                    "server_price_evidence_level": server_parameter["evidence_level"],
                    "server_price_parameter_status": server_parameter["parameter_status"],
                    "result_status": config["result_status"],
                }
            )

    result = pd.DataFrame(rows)
    base = result[result["server_price_case"] == "base"].copy()
    ig = base[base["scenario"] == "IG"].iloc[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")

    table = "\n".join(
        f"| {row.scenario} | {row.model_installed_dual_l20_servers_at_single_reserve_parameter:,.0f} | "
        f"{row.adjusted_installed_dual_l20_servers:,.0f} | "
        f"{row.adjusted_annual_ai_facility_energy_twh:.3f} | "
        f"{row.annual_server_capital_facility_maintenance_cost_usd/1e9:.3f} | "
        f"{row.annual_electricity_cost_usd/1e9:.3f} | {row.annual_us_owned_core_cost_billion_usd:.3f} |"
        for row in base.itertuples(index=False)
    )
    findings = f"""# 美国本地双 L20 服务器核心成本反事实

模型版本：`{model_version}`；成本案例：`{cost_case_version}`。

## 中心结果

中国与美国本地部署统一使用模型中唯一的装机峰值裕量参数 **{target_reserve:.0%}**。在保持中国制造业 `{model_version}` 任务需求和运行负荷不变、采用美国服务器工程 BOM 基准价 **46,500 美元/台**及 EIA 2024 全国工业全包电价 **{electricity_price:.4f} 美元/kWh**时，IG 架构的美国本地部署核心年成本为 **{ig['annual_us_owned_core_cost_billion_usd']:.3f} 十亿美元/年**。

| 架构 | 模型服务器数（单一15%裕量） | 成本计算服务器数 | 年设施用电（TWh） | 年化服务器及设施维护（十亿美元） | 电费（十亿美元） | 核心总成本（十亿美元/年） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

服务器采购价低/基准/高值为44,400/46,500/50,800美元/台，详细敏感性见 `us_owned_core_cost.csv`。服务器经济寿命采用 **{float(config['economic_life_years']):g}年**；电价在本轮固定为全国最终年度中心值，避免把Texas、Virginia和California地理差异误写成统计置信区间。

## 解释边界

这是使用中国任务负荷的美国价格反事实，不是美国制造业需求预测。15%装机裕量已经在核心物理模型中逐节点重新求解，不再做第二次后处理；参考能耗校验也读取同一个参数，但只作诊断，不会再次增加服务器。成本只包含服务器采购及20%设施附加投资的年化、5%年度维护费和全包工业电费；不含销售税、单独需量费、电网扩容、土地、水、光伏、电池和模型运维人员。5年是AI服务器保守经济寿命，不是物理报废年限；公司披露和研究支持5.5—6年替代情景，7年以上须满足工作负载稳定、维护和兼容性条件。服务器价是C级工程BOM代理，不是已观测完整整机报价，正式RFQ取得后应替换。
"""
    args.findings_output.write_text(findings, encoding="utf-8")

    component_error = (
        result["annual_us_owned_core_cost_usd"]
        - result["annual_server_capital_facility_maintenance_cost_usd"]
        - result["annual_electricity_cost_usd"]
    ).abs().max()
    payload = {
        "status": "complete_validated_us_owned_core_cost_proxy",
        "model_version": model_version,
        "cost_case_version": cost_case_version,
        "server_price_evidence_level": str(server_parameter["evidence_level"]),
        "server_price_parameter_status": str(server_parameter["parameter_status"]),
        "electricity_price_final_year": True,
        "separate_demand_charge_added": False,
        "physical_load_reoptimized": False,
        "original_installed_reserve_fraction": original_reserve,
        "us_installed_reserve_fraction": target_reserve,
        "reserve_adjustment_method": str(config["reserve_adjustment_method"]),
        "rows": len(result),
        "component_reconciliation_max_abs_usd": float(component_error),
        "formal_rfQ_still_required": True,
    }
    args.done_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
