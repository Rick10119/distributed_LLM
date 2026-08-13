#!/usr/bin/env python3
"""Compare owned deployment costs with same-service GPU-IaaS payment proxies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_code"))

from core.config import load_config, rooted  # noqa: E402


PRICE_COLUMNS = {"low": "low_value", "base": "base_value", "high": "high_value"}
SCENARIOS = ["IF", "IG", "II_1host"]


def numeric_parameter(frame: pd.DataFrame, parameter_id: str, price_case: str) -> float:
    row = frame[frame["parameter_id"] == parameter_id]
    if len(row) != 1:
        raise ValueError(f"Expected one price parameter row for {parameter_id}")
    try:
        value = float(row[PRICE_COLUMNS[price_case]].iloc[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric {price_case} value for {parameter_id}") from exc
    if value <= 0:
        raise ValueError(f"Price parameter {parameter_id} must be positive")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--national-summary", type=Path, required=True)
    parser.add_argument("--hourly-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--break-even-output", type=Path, required=True)
    parser.add_argument("--findings-output", type=Path, required=True)
    parser.add_argument("--done-output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(ROOT, args.defaults, args.config)
    model_version = str(config["model_version"])
    payment = config["enterprise_payment"]
    accelerators_per_instance = int(payment["accelerators_per_cloud_instance"])
    price_cases = list(payment["price_cases"])

    national = pd.read_csv(args.national_summary, encoding="utf-8-sig")
    if len(national) != 93 or set(national["scenario"]) != set(SCENARIOS):
        raise ValueError("Subscription analysis requires 31 industries and three architectures")
    if set(national["model_version"]) != {model_version}:
        raise ValueError("National summary version does not match active config")

    hourly = pd.concat(
        [pd.read_csv(path, encoding="utf-8-sig") for path in args.hourly_inputs],
        ignore_index=True,
    )
    if set(hourly["scenario"]) != {"II_1host"} or hourly["industry_code"].nunique() != 31:
        raise ValueError("Cloud payment proxy requires all 31 II_1host hourly profiles")
    hours_per_industry = hourly.groupby("industry_code")["hour"].nunique()
    if set(hours_per_industry.index) != set(national["industry_code"].unique()) or not (
        hours_per_industry == int(config["model"]["horizon_hours"])
    ).all():
        raise ValueError("Cloud payment hourly coverage is incomplete")

    compute_profile = hourly.groupby("hour", as_index=False).agg(
        national_accelerator_h=("industry_equivalent_ai_compute_accelerator_h", "sum")
    )
    horizon_hours = len(compute_profile)
    represented_days = horizon_hours / 24.0
    peak_accelerator_h_per_hour = float(compute_profile["national_accelerator_h"].max())
    annual_accelerator_h = (
        float(compute_profile["national_accelerator_h"].sum())
        / represented_days
        * float(config["model"]["annualization_days"])
    )
    contracted_instances = math.ceil(peak_accelerator_h_per_hour / accelerators_per_instance)
    billed_instance_h = annual_accelerator_h / accelerators_per_instance
    annual_service_units = (
        float(national.groupby("scenario")["industry_daily_effective_service_units"].sum().iloc[0])
        * float(config["model"]["annualization_days"])
    )

    prices = pd.read_csv(
        rooted(config, config["paths"]["enterprise_ai_cost_parameters"]),
        encoding="utf-8-sig",
    )
    reserved_id = str(payment["cloud_reserved_annual_parameter_id"])
    ondemand_id = str(payment["cloud_ondemand_hourly_parameter_id"])

    owned_costs = {
        scenario: float(
            national.loc[
                national["scenario"] == scenario,
                "industry_equivalent_incremental_total_cost_rmb",
            ].sum()
        )
        for scenario in SCENARIOS
    }
    underlying_resource_proxy = owned_costs["II_1host"]
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        cost = owned_costs[scenario]
        rows.append(
            {
                "model_version": model_version,
                "mode": f"owned_{scenario}",
                "architecture_or_billing": scenario,
                "price_case": "model_result",
                "payment_boundary": "owned_physical_capacity_and_incremental_operating_cost",
                "annual_enterprise_payment_rmb": cost,
                "rmb_per_effective_service_unit": cost / annual_service_units,
                "rmb_per_accelerator_h": cost / annual_accelerator_h,
                "ratio_to_owned_IF": cost / owned_costs["IF"],
                "ratio_to_owned_IG": cost / owned_costs["IG"],
                "ratio_to_owned_II_1host": cost / owned_costs["II_1host"],
                "contracted_dual_l20_instances": 0,
                "billed_dual_l20_instance_h_year": 0.0,
                "unit_price_rmb": 0.0,
                "unit_price_basis": "not_applicable",
                "subscription_to_underlying_resource_proxy_ratio": 0.0,
            }
        )

    for price_case in price_cases:
        annual_price = numeric_parameter(prices, reserved_id, price_case)
        hourly_price = numeric_parameter(prices, ondemand_id, price_case)
        reserved_cost = contracted_instances * annual_price
        ondemand_cost = billed_instance_h * hourly_price
        for mode, cost, unit_price, basis, contracted, billed in (
            (
                "cloud_reserved_capacity",
                reserved_cost,
                annual_price,
                "RMB/dual-L20-instance-year",
                contracted_instances,
                0.0,
            ),
            (
                "cloud_ondemand",
                ondemand_cost,
                hourly_price,
                "RMB/dual-L20-instance-hour",
                0,
                billed_instance_h,
            ),
        ):
            rows.append(
                {
                    "model_version": model_version,
                    "mode": mode,
                    "architecture_or_billing": mode,
                    "price_case": price_case,
                    "payment_boundary": str(payment["cloud_product_boundary"]),
                    "annual_enterprise_payment_rmb": cost,
                    "rmb_per_effective_service_unit": cost / annual_service_units,
                    "rmb_per_accelerator_h": cost / annual_accelerator_h,
                    "ratio_to_owned_IF": cost / owned_costs["IF"],
                    "ratio_to_owned_IG": cost / owned_costs["IG"],
                    "ratio_to_owned_II_1host": cost / owned_costs["II_1host"],
                    "contracted_dual_l20_instances": contracted,
                    "billed_dual_l20_instance_h_year": billed,
                    "unit_price_rmb": unit_price,
                    "unit_price_basis": basis,
                    "subscription_to_underlying_resource_proxy_ratio": cost
                    / underlying_resource_proxy,
                }
            )

    comparison = pd.DataFrame(rows)
    break_even = pd.DataFrame(
        [
            {
                "model_version": model_version,
                "owned_architecture": scenario,
                "owned_annual_cost_rmb": cost,
                "break_even_reserved_rmb_per_dual_l20_instance_year": cost
                / contracted_instances,
                "break_even_reserved_rmb_per_dual_l20_instance_month": cost
                / contracted_instances
                / 12.0,
                "break_even_ondemand_rmb_per_dual_l20_instance_hour": cost
                / billed_instance_h,
            }
            for scenario, cost in owned_costs.items()
        ]
    )

    base_reserved = comparison[
        (comparison["mode"] == "cloud_reserved_capacity")
        & (comparison["price_case"] == "base")
    ].iloc[0]
    base_ondemand = comparison[
        (comparison["mode"] == "cloud_ondemand")
        & (comparison["price_case"] == "base")
    ].iloc[0]
    reserved_range = comparison[comparison["mode"] == "cloud_reserved_capacity"][
        "annual_enterprise_payment_rmb"
    ]
    ondemand_range = comparison[comparison["mode"] == "cloud_ondemand"][
        "annual_enterprise_payment_rmb"
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False, encoding="utf-8-sig")
    break_even.to_csv(args.break_even_output, index=False, encoding="utf-8-sig")

    findings = f"""# 全国同等服务量下的自建与GPU-IaaS付款比较

## 口径

- 使用`{model_version}`全国{annual_service_units / float(config['model']['annualization_days']):,.3f}有效服务单位/日和行业池逐时算力曲线，云端承担与三种自建架构相同的有效服务。
- 包年/包月容量按全国逐时加速器需求峰值除以每个双L20实例2张卡并向上取整，共{contracted_instances:,}个实例；按量付费按全年{billed_instance_h:,.0f}个双L20实例小时计费。
- 云价使用本地参数表C19和C18的低/基准/高价格情景。[阿里云ECS实例族文档](https://help.aliyun.com/en/ecs/user-guide/gpu-accelerated-compute-optimized-and-vgpu-accelerated-instance-families-1)确认gn8is为L20实例族，[阿里云PAI官方案例](https://help.aliyun.com/en/pai/use-cases/pai-eas-spot-best-practices)给出单L20按量目录价14.4元/小时；双卡价格仍是项目中的线性代理，不是实时成交价。
- 云订阅费属于企业付款边界，不与云提供商底层服务器、电力和接入成本叠加。它不是模型API、SaaS或社会资源成本；网络、云盘、流量、迁移集成、税费、合同折扣和软件服务均未计入。

## 结果

| 模式 | 年成本/付款（十亿元） | 相对IF自建 | 相对IG自建 |
| --- | ---: | ---: | ---: |
| IF自建 | {owned_costs['IF']/1e9:.3f} | 1.00 | {owned_costs['IF']/owned_costs['IG']:.2f} |
| IG自建 | {owned_costs['IG']/1e9:.3f} | {owned_costs['IG']/owned_costs['IF']:.2f} | 1.00 |
| II单节点物理成本代理 | {owned_costs['II_1host']/1e9:.3f} | {owned_costs['II_1host']/owned_costs['IF']:.2f} | {owned_costs['II_1host']/owned_costs['IG']:.2f} |
| GPU-IaaS容量订阅，基准价 | {float(base_reserved['annual_enterprise_payment_rmb'])/1e9:.3f} | {float(base_reserved['ratio_to_owned_IF']):.2f} | {float(base_reserved['ratio_to_owned_IG']):.2f} |
| GPU-IaaS按量，基准价 | {float(base_ondemand['annual_enterprise_payment_rmb'])/1e9:.3f} | {float(base_ondemand['ratio_to_owned_IF']):.2f} | {float(base_ondemand['ratio_to_owned_IG']):.2f} |

容量订阅在价格范围内为{reserved_range.min()/1e9:.3f}—{reserved_range.max()/1e9:.3f}十亿元/年，按量付费为{ondemand_range.min()/1e9:.3f}—{ondemand_range.max()/1e9:.3f}十亿元/年。当前基准公开价格代理下，容量订阅约为IF自建的{float(base_reserved['ratio_to_owned_IF']):.2f}倍，按量约为{float(base_ondemand['ratio_to_owned_IF']):.2f}倍。

与IF自建打平的双L20实例价格约为{float(break_even.loc[break_even.owned_architecture=='IF','break_even_reserved_rmb_per_dual_l20_instance_month'].iloc[0]):,.0f}元/月，或{float(break_even.loc[break_even.owned_architecture=='IF','break_even_ondemand_rmb_per_dual_l20_instance_hour'].iloc[0]):.2f}元/小时。与IG自建打平则约为{float(break_even.loc[break_even.owned_architecture=='IG','break_even_reserved_rmb_per_dual_l20_instance_month'].iloc[0]):,.0f}元/月，或{float(break_even.loc[break_even.owned_architecture=='IG','break_even_ondemand_rmb_per_dual_l20_instance_hour'].iloc[0]):.2f}元/小时。

## 解释边界

在当前超大、持续且接近平坦的全国工作负荷下，公开零售GPU-IaaS价格代理高于拥有设备的年化成本；这不能外推到低利用率单厂、短期项目或能使用大客户折扣的企业。订阅和自建还承担不同的技术淘汰、弹性、可用性、网络、集成和运维风险，因此这里只是企业付款筛查，不是完整采购建议。
"""
    args.findings_output.write_text(findings, encoding="utf-8")

    payload = {
        "status": "validated",
        "model_version": model_version,
        "industries": 31,
        "same_service_boundary": True,
        "cloud_product_boundary": payment["cloud_product_boundary"],
        "contracted_dual_l20_instances": contracted_instances,
        "annual_billed_dual_l20_instance_h": billed_instance_h,
        "checks": [
            "31-industry II_1host hourly coverage",
            "same national effective service",
            "positive low/base/high price cases",
            "reserved capacity covers national hourly compute peak",
            "cloud payment not added to underlying resource cost",
        ],
    }
    args.done_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
