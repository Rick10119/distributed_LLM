"""Calculate single-industry enterprise-paid AI costs from a joint PyPSA result.

The comparison separates the enterprise payment boundary from reconstructed
cloud resource cost.  It compares annualized local ownership with observed-
price proxy GPU-IaaS monthly reservation and derived on-demand billing.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "05_results"
sys.path.insert(0, str(ROOT / "08_code"))

import china_minimum_prototype as parameter_core  # noqa: E402


SUMMARY_INPUT = RESULTS / "c36_pypsa_joint_summary.csv"
HOURLY_INPUT = RESULTS / "c36_pypsa_joint_hourly.csv"
OUTPUT = RESULTS / "c36_enterprise_direct_cost_summary.csv"
FINDINGS = RESULTS / "c36_enterprise_direct_cost_findings.md"


def main() -> None:
    summary = pd.read_csv(SUMMARY_INPUT, encoding="utf-8-sig").set_index("scenario")
    hourly = pd.read_csv(HOURLY_INPUT, encoding="utf-8-sig")
    params = parameter_core.read_parameters()

    baseline = summary.loc["baseline_no_ai"]
    local = summary.loc["local_ai"]
    cloud = summary.loc["cloud_ai"]
    daily_service = float(cloud["daily_ai_service_accelerator_h"])
    annual_accelerator_h = daily_service * 365.0
    cloud_peak_accelerator_h = float(
        hourly.query("scenario == 'cloud_ai'")["ai_executed_accelerator_h"].max()
    )
    contracted_dual_l20_instances = math.ceil(cloud_peak_accelerator_h / 2.0)
    billed_dual_l20_instance_h = annual_accelerator_h / 2.0

    cloud_monthly_price = parameter_core.number(params, "C17")
    cloud_hourly_price = parameter_core.number(params, "C18")
    local_server_price = parameter_core.number(params, "L02")
    local_facility_fraction = parameter_core.number(params, "L15")
    local_server_groups = float(local["server_groups_2xl20_continuous"])
    local_initial_capex = (
        local_server_groups * local_server_price * (1.0 + local_facility_fraction)
    )

    increment_fields = {
        "annualized_local_compute_rmb": "annual_server_cost_rmb",
        "incremental_flat_energy_rmb": "annual_flat_energy_cost_rmb",
        "incremental_maximum_demand_rmb": "annual_maximum_demand_cost_rmb",
        "incremental_grid_capacity_rmb": "annual_grid_capacity_cost_rmb",
        "incremental_pv_rmb": "annual_pv_cost_rmb",
        "incremental_battery_rmb": "annual_battery_cost_rmb",
    }
    local_components = {
        output_name: float(local[source_name] - baseline[source_name])
        for output_name, source_name in increment_fields.items()
    }
    local_total = sum(local_components.values())
    if abs(local_total - float(local["incremental_vs_baseline_rmb"])) > 10.0:
        raise ValueError("local enterprise cost components do not reconcile")

    reserved_subscription = (
        contracted_dual_l20_instances * cloud_monthly_price * 12.0
    )
    ondemand_subscription = billed_dual_l20_instance_h * cloud_hourly_price
    cloud_underlying_resource_proxy = float(cloud["incremental_vs_baseline_rmb"])

    common = {
        "industry_code": "C36",
        "industry_name_cn": "汽车制造业",
        "daily_ai_service_accelerator_h": daily_service,
        "annual_ai_service_accelerator_h": annual_accelerator_h,
        "cloud_provider_physical_groups_including_reserve": float(
            cloud["server_groups_2xl20_continuous"]
        ),
        "cloud_underlying_resource_cost_proxy_rmb": cloud_underlying_resource_proxy,
        "network_and_integration_cost_status": "not_included",
        "result_scope": "industry_bucket_enterprise_payment_screen",
    }
    rows = [
        {
            **common,
            "mode": "local_purchase_annualized",
            "deployment": "local",
            "billing_basis": "annualized_owned_2xl20_servers",
            "owned_or_contracted_2xl20_groups": local_server_groups,
            "billed_2xl20_instance_h_year": 0.0,
            "initial_local_capex_rmb": local_initial_capex,
            **local_components,
            "annual_cloud_subscription_rmb": 0.0,
            "annual_enterprise_direct_cost_rmb": local_total,
        },
        {
            **common,
            "mode": "cloud_reserved_monthly",
            "deployment": "cloud",
            "billing_basis": "peak_service_capacity_x_12_months",
            "owned_or_contracted_2xl20_groups": contracted_dual_l20_instances,
            "billed_2xl20_instance_h_year": 0.0,
            "initial_local_capex_rmb": 0.0,
            **{key: 0.0 for key in local_components},
            "annual_cloud_subscription_rmb": reserved_subscription,
            "annual_enterprise_direct_cost_rmb": reserved_subscription,
        },
        {
            **common,
            "mode": "cloud_ondemand",
            "deployment": "cloud",
            "billing_basis": "executed_2xl20_instance_hours",
            "owned_or_contracted_2xl20_groups": 0.0,
            "billed_2xl20_instance_h_year": billed_dual_l20_instance_h,
            "initial_local_capex_rmb": 0.0,
            **{key: 0.0 for key in local_components},
            "annual_cloud_subscription_rmb": ondemand_subscription,
            "annual_enterprise_direct_cost_rmb": ondemand_subscription,
        },
    ]
    for row in rows:
        row["direct_cost_rmb_per_accelerator_h"] = (
            float(row["annual_enterprise_direct_cost_rmb"]) / annual_accelerator_h
        )
        row["direct_cost_ratio_to_local"] = (
            float(row["annual_enterprise_direct_cost_rmb"]) / local_total
        )
        if row["deployment"] == "cloud":
            row["subscription_to_cloud_resource_proxy_ratio"] = (
                float(row["annual_cloud_subscription_rmb"])
                / cloud_underlying_resource_proxy
            )
        else:
            row["subscription_to_cloud_resource_proxy_ratio"] = 0.0

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    reserved_break_even_monthly = local_total / contracted_dual_l20_instances / 12.0
    ondemand_break_even_hourly = local_total / billed_dual_l20_instance_h
    findings = f"""# C36企业直接成本初步比较

## 边界

- 使用同一C36 AI服务量和当前PyPSA最优物理容量，不重跑其他行业。
- 本地模式计入服务器及20%附属设施投资的年化成本、维护费，以及相对于无AI基线新增的平段电量费、最大需量费、接入、光伏和储能成本。
- 云端模式采用阿里云双L20 GPU IaaS公开价格代理：包月 {cloud_monthly_price:,.2f} 元/实例·月，按量 {cloud_hourly_price:.2f} 元/实例·小时。云订阅费已代表企业付款，不再叠加云端底层服务器、电力和接入成本。
- 包月合同数按优化后AI服务峰值除以每实例2张L20并向上取整，为 {contracted_dual_l20_instances:,} 个；云提供商内部10%物理备用不要求企业另买实例。
- 网络流量、迁移集成、云盘、软件、税费、合同折扣和SaaS/API收费均未计入。

## 结果

| 模式 | 企业年直接成本 | 单位服务成本 | 相对本地 | 初始本地资本支出 |
| --- | ---: | ---: | ---: | ---: |
| 本地购买并年化 | {local_total/1e8:.2f}亿元 | {local_total/annual_accelerator_h:.2f}元/加速器小时 | 1.00 | {local_initial_capex/1e8:.2f}亿元 |
| 云端包月容量 | {reserved_subscription/1e8:.2f}亿元 | {reserved_subscription/annual_accelerator_h:.2f}元/加速器小时 | {reserved_subscription/local_total:.2f} | 0 |
| 云端按量 | {ondemand_subscription/1e8:.2f}亿元 | {ondemand_subscription/annual_accelerator_h:.2f}元/加速器小时 | {ondemand_subscription/local_total:.2f} | 0 |

本地年直接成本分解为：服务器及附属设施年化和维护 {local_components['annualized_local_compute_rmb']/1e8:.2f} 亿元，新增电量费 {local_components['incremental_flat_energy_rmb']/1e8:.2f} 亿元，新增最大需量费 {local_components['incremental_maximum_demand_rmb']/1e8:.2f} 亿元；当前最优反事实下没有AI新增光储或接入投资。

若其他参数不变，云包月价格降至约 {reserved_break_even_monthly:,.0f} 元/双L20实例·月时与本地年化成本持平；云按量价格降至约 {ondemand_break_even_hourly:.2f} 元/双L20实例·小时才持平。当前公开价格代理分别为 {cloud_monthly_price:,.2f} 元/月和 {cloud_hourly_price:.2f} 元/小时。

## 解释限制

该结果说明在当前超大行业桶、较高持续利用率和公开零售云价下，本地拥有设备的企业支付成本低于GPU IaaS。它不等于普通单厂结论：行业桶消除了服务器离散投资和跨企业不能共享容量的问题，也没有计入企业自建运维组织、模型软件、网络安全和技术淘汰风险。云价格是公开参考价，不是大客户长期合同成交价；SaaS和模型API也不能用GPU实例小时直接替代。
"""
    FINDINGS.write_text(findings, encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(OUTPUT),
                "findings": str(FINDINGS),
                "rows": len(frame),
                "reserved_break_even_monthly_rmb": reserved_break_even_monthly,
                "ondemand_break_even_hourly_rmb": ondemand_break_even_hourly,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
