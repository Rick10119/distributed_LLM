#!/usr/bin/env python3
"""Add a source-audited full-cloud enterprise-payment scenario to the core model.

The module deliberately separates three evidence states:
1. official API list prices (model-ready price inputs),
2. existing project workload assumptions for office/RAG and agents,
3. reserved GPU-IaaS for the four non-Token tasks, allocated by effective service,
4. object storage for the existing IG model-storage footprint.

The formal comparison intentionally excludes on-demand GPU billing. The resulting
totals are payment scenarios, not quality-equivalent procurement recommendations
or replacements for a residual-workload peak-capacity re-optimization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_code"))

from core.config import load_config  # noqa: E402


TOKEN_TASK_MAP = {
    "office": "office_rag_copilot",
    "agent": "business_agent",
}
REQUIRED_PRICE_COLUMNS = {
    "cost_case_version",
    "provider",
    "model_id",
    "currency",
    "input_per_mtoken",
    "output_per_mtoken",
    "active_for_baseline",
    "benchmark_role",
    "mainstream_representative",
    "evidence_level",
    "model_status",
    "source_url",
    "fx_to_cny",
}


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def weighted_task_usage(
    workload_rows: pd.DataFrame,
    *,
    industry_code: str,
    task_type: str,
    employment_persons: float,
    large_enterprise_share: float,
) -> dict[str, float]:
    selected = workload_rows[
        (workload_rows["industry_code"] == industry_code)
        & (workload_rows["task_type"] == task_type)
        & (workload_rows["scenario_year"].astype(str) == "2030")
        & (workload_rows["scenario_level"] == "base")
    ]
    by_size = {str(row.enterprise_size_class): row for row in selected.itertuples()}
    if set(by_size) != {"large", "sme"}:
        raise ValueError(f"Expected large and sme workload rows for {industry_code}/{task_type}")
    if not 0 <= large_enterprise_share <= 1:
        raise ValueError("large_enterprise_share must be in [0,1]")

    totals = {"business_tasks_day": 0.0, "api_calls_day": 0.0, "input_tokens_day": 0.0, "output_tokens_day": 0.0}
    for size, weight in (("large", large_enterprise_share), ("sme", 1.0 - large_enterprise_share)):
        row = by_size[size]
        adoption = float(row.ai_adoption_rate)
        active_share = float(row.active_user_or_equipment_share)
        intensity = float(row.service_intensity_per_driver_day)
        calls_per_task = float(row.calls_per_task)
        input_tokens = float(row.input_tokens_per_call)
        output_tokens = float(row.output_tokens_per_call)
        tasks = employment_persons * weight * adoption * active_share * intensity
        calls = tasks * calls_per_task
        totals["business_tasks_day"] += tasks
        totals["api_calls_day"] += calls
        totals["input_tokens_day"] += calls * input_tokens
        totals["output_tokens_day"] += calls * output_tokens
    return totals


def api_bill_cny(input_tokens: float, output_tokens: float, price: pd.Series) -> float:
    fx = float(price["fx_to_cny"])
    if fx <= 0:
        raise ValueError("fx_to_cny must be positive")
    return (
        input_tokens / 1_000_000.0 * float(price["input_per_mtoken"])
        + output_tokens / 1_000_000.0 * float(price["output_per_mtoken"])
    ) * fx


def lifecycle_price(frame: pd.DataFrame, parameter_id: str, price_case: str) -> float:
    """Return one positive numeric lifecycle parameter for a selected price case."""
    selected = frame[frame["parameter_id"] == parameter_id]
    if len(selected) != 1:
        raise ValueError(f"Expected one lifecycle parameter row for {parameter_id}")
    if price_case not in {"low", "base", "high"}:
        raise ValueError(f"Unsupported lifecycle price case: {price_case}")
    value = float(selected.iloc[0][price_case])
    if value <= 0:
        raise ValueError(f"Lifecycle parameter {parameter_id}/{price_case} must be positive")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--national-summary", type=Path, required=True)
    parser.add_argument("--service-input", type=Path, required=True)
    parser.add_argument("--workload-input", type=Path, required=True)
    parser.add_argument("--industry-baseline", type=Path, required=True)
    parser.add_argument("--api-prices", type=Path, required=True)
    parser.add_argument("--cloud-comparison", type=Path, required=True)
    parser.add_argument("--lifecycle-parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mainstream-output", type=Path, required=True)
    parser.add_argument("--task-output", type=Path, required=True)
    parser.add_argument("--findings-output", type=Path, required=True)
    parser.add_argument("--done-output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(ROOT, args.defaults, args.config)
    model_version = str(config["model_version"])
    annual_days = float(config["model"]["annualization_days"])
    settings = config["api_token_cost"]
    full_cloud = config["full_cloud_cost"]

    national = pd.read_csv(args.national_summary, encoding="utf-8-sig")
    if set(national["model_version"]) != {model_version} or len(national) != 93:
        raise ValueError("API Token module requires the active 31-industry, three-architecture summary")
    owned_costs = {
        scenario: float(
            national.loc[national["scenario"] == scenario, "industry_equivalent_incremental_total_cost_rmb"].sum()
        )
        for scenario in ("IF", "IG", "II_1host")
    }

    service = pd.read_csv(args.service_input, encoding="utf-8-sig")
    service = service[
        (service["year"] == int(config["demand"]["year"]))
        & (service["parameter_case"] == config["demand"]["effective_service"]["parameter_case"])
    ]
    if service["industry_code"].nunique() != 31 or set(service["task_id"].unique()) != {
        "office", "agent", "vision", "maintenance", "scheduling", "simulation"
    }:
        raise ValueError("Effective-service input does not cover 31 industries and six tasks")
    service_column = str(config["demand"]["effective_service"]["source_column"])
    token_ready_service = float(service[service["task_id"].isin(TOKEN_TASK_MAP)][service_column].sum())
    total_service = float(service[service_column].sum())
    token_ready_service_share = token_ready_service / total_service

    workload = pd.read_csv(args.workload_input, encoding="utf-8-sig")
    baseline = pd.read_csv(args.industry_baseline, encoding="utf-8-sig")
    if baseline["industry_code"].nunique() != 31:
        raise ValueError("Industry baseline must cover 31 manufacturing industries")

    task_rows: list[dict[str, object]] = []
    for industry in baseline.itertuples():
        employment = float(industry.employment_2023_10k_person) * 10_000.0
        observed_large_share = float(industry.large_enterprise_share)
        large_share_observed = pd.notna(observed_large_share)
        large_share = (
            observed_large_share
            if large_share_observed
            else float(settings["large_workload_weight_when_unobserved"])
        )
        for core_task, raw_task in TOKEN_TASK_MAP.items():
            usage = weighted_task_usage(
                workload,
                industry_code=str(industry.industry_code),
                task_type=raw_task,
                employment_persons=employment,
                large_enterprise_share=large_share,
            )
            task_rows.append(
                {
                    "model_version": model_version,
                    "cost_case_version": settings["cost_case_version"],
                    "industry_code": industry.industry_code,
                    "industry_name_cn": industry.industry_name_cn,
                    "core_task_id": core_task,
                    "raw_task_type": raw_task,
                    "driver_proxy": "industry_employment_2023",
                    "large_workload_weight": large_share,
                    "large_workload_weight_observed": large_share_observed,
                    "business_tasks_day": usage["business_tasks_day"],
                    "api_calls_day": usage["api_calls_day"],
                    "input_tokens_day": usage["input_tokens_day"],
                    "output_tokens_day": usage["output_tokens_day"],
                    "annual_input_tokens": usage["input_tokens_day"] * annual_days,
                    "annual_output_tokens": usage["output_tokens_day"] * annual_days,
                    "cache_hit_rate": 0.0,
                    "batch_share": 0.0,
                    "retry_rate": 0.0,
                    "reasoning_tokens": 0.0,
                    "evidence_status": "sensitivity_only_existing_project_D_assumptions",
                    "limitation": "Employment is used as the eligible-user driver; the published industry baseline lacks large_enterprise_share, so the configured D-level large/SME weight is used. No enterprise trace validation.",
                }
            )
    task_detail = pd.DataFrame(task_rows)
    annual_input_tokens = float(task_detail["annual_input_tokens"].sum())
    annual_output_tokens = float(task_detail["annual_output_tokens"].sum())

    prices = pd.read_csv(args.api_prices, encoding="utf-8-sig")
    missing = REQUIRED_PRICE_COLUMNS - set(prices.columns)
    if missing:
        raise ValueError(f"API price table missing columns: {sorted(missing)}")
    active = prices[
        prices["active_for_baseline"].map(bool_value)
        & (prices["evidence_level"] == "A")
        & (prices["model_status"] == "model_ready_price_only")
    ].copy()
    if active.empty or active[["provider", "model_id", "service_tier"]].duplicated().any():
        raise ValueError("Active API price rows must be non-empty and unique")
    for column in ("input_per_mtoken", "output_per_mtoken", "fx_to_cny"):
        active[column] = pd.to_numeric(active[column], errors="raise")
        if (active[column] <= 0).any():
            raise ValueError(f"Active price column {column} must be positive")
    mainstream_prices = active[active["mainstream_representative"].map(bool_value)].copy()
    expected_mainstream_providers = {"Alibaba Cloud", "Anthropic", "DeepSeek", "Google", "OpenAI"}
    if (
        set(mainstream_prices["provider"]) != expected_mainstream_providers
        or mainstream_prices["provider"].duplicated().any()
    ):
        raise ValueError("Mainstream API comparison requires exactly one representative model per provider")

    residual_service_share = 1.0 - token_ready_service_share

    cloud = pd.read_csv(args.cloud_comparison, encoding="utf-8-sig")
    reserved_cloud = cloud[cloud["mode"] == full_cloud["gpu_billing_mode"]].copy()
    if set(reserved_cloud["price_case"]) != {"low", "base", "high"}:
        raise ValueError("Cloud comparison must contain low, base, and high reserved-capacity rows")
    if reserved_cloud["price_case"].duplicated().any():
        raise ValueError("Reserved-capacity cloud comparison must have one row per price case")

    lifecycle = pd.read_csv(args.lifecycle_parameters, encoding="utf-8-sig")
    storage_price = lifecycle_price(
        lifecycle,
        str(full_cloud["cloud_storage_parameter_id"]),
        str(full_cloud["cloud_storage_price_case"]),
    )
    storage_reference = str(full_cloud["storage_reference_architecture"])
    storage_rows = national[national["scenario"] == storage_reference]
    cloud_storage_gb = float(
        (
            storage_rows["per_host_model_storage_required_gb"]
            * storage_rows["equivalent_host_multiplier"]
        ).sum()
    )
    annual_cloud_storage_cost = cloud_storage_gb * storage_price * 12.0

    comparison_rows: list[dict[str, object]] = []
    for cloud_row in reserved_cloud.itertuples(index=False):
        full_gpu_payment = float(cloud_row.annual_enterprise_payment_rmb)
        residual_gpu_payment = full_gpu_payment * residual_service_share
        for price in active.itertuples(index=False):
            series = pd.Series(price._asdict())
            api_cost = api_bill_cny(annual_input_tokens, annual_output_tokens, series)
            total = residual_gpu_payment + api_cost + annual_cloud_storage_cost
            comparison_rows.append(
                {
                    "model_version": model_version,
                    "cost_case_version": settings["cost_case_version"],
                    "full_cloud_scenario_version": full_cloud["scenario_version"],
                    "provider": price.provider,
                    "platform": price.platform,
                    "model_id": price.model_id,
                    "model_version_or_price_period": price.model_version,
                    "benchmark_role": price.benchmark_role,
                    "mainstream_representative": bool_value(price.mainstream_representative),
                    "region": price.region,
                    "native_currency": price.currency,
                    "fx_to_cny": price.fx_to_cny,
                    "annual_input_tokens": annual_input_tokens,
                    "annual_output_tokens": annual_output_tokens,
                    "standard_input_price_native_per_mtoken": price.input_per_mtoken,
                    "standard_output_price_native_per_mtoken": price.output_per_mtoken,
                    "annual_api_token_cost_rmb": api_cost,
                    "token_ready_service_share_proxy": token_ready_service_share,
                    "residual_non_token_service_share_proxy": residual_service_share,
                    "cloud_reserved_price_case": cloud_row.price_case,
                    "full_workload_reserved_gpu_payment_rmb": full_gpu_payment,
                    "residual_reserved_gpu_payment_rmb_proxy": residual_gpu_payment,
                    "cloud_storage_reference_architecture": storage_reference,
                    "cloud_storage_gb_proxy": cloud_storage_gb,
                    "cloud_storage_price_rmb_per_gb_month": storage_price,
                    "annual_cloud_storage_cost_rmb_proxy": annual_cloud_storage_cost,
                    "annual_full_cloud_cost_rmb_proxy": total,
                    "ratio_to_owned_IF": total / owned_costs["IF"],
                    "ratio_to_owned_IG": total / owned_costs["IG"],
                    "ratio_to_owned_II_1host": total / owned_costs["II_1host"],
                    "difference_to_owned_IF_rmb": total - owned_costs["IF"],
                    "difference_to_owned_IG_rmb": total - owned_costs["IG"],
                    "difference_to_owned_II_1host_rmb": total - owned_costs["II_1host"],
                    "quality_equivalence_verified": False,
                    "workload_trace_verified": False,
                    "residual_compute_reoptimized": False,
                    "gpu_billing_mode": full_cloud["gpu_billing_mode"],
                    "result_status": full_cloud["result_status"],
                    "price_source_url": price.source_url,
                }
            )
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["cloud_reserved_price_case", "annual_full_cloud_cost_rmb_proxy"]
    )

    total_rows: list[dict[str, object]] = []
    for scenario in ("IF", "IG", "II_1host"):
        total = owned_costs[scenario]
        total_rows.append(
            {
                "option_name": f"自建 {scenario}",
                "annual_total_cost_billion_rmb": total / 1e9,
                "option_group": "owned_compute",
                "provider": "enterprise_owned",
                "model_id": scenario,
                "difference_to_IF_billion_rmb": (total - owned_costs["IF"]) / 1e9,
                "percent_difference_to_IF": total / owned_costs["IF"] - 1.0,
                "difference_to_IG_billion_rmb": (total - owned_costs["IG"]) / 1e9,
                "percent_difference_to_IG": total / owned_costs["IG"] - 1.0,
                "difference_to_II_billion_rmb": (total - owned_costs["II_1host"]) / 1e9,
                "percent_difference_to_II": total / owned_costs["II_1host"] - 1.0,
                "cost_boundary": "自建物理容量与增量运维成本",
                "comparison_status": "existing_core_result",
                "source_or_role": f"{model_version} national core scenario",
            }
        )

    mainstream_comparison = comparison[
        comparison["mainstream_representative"]
        & (comparison["cloud_reserved_price_case"] == full_cloud["main_gpu_price_case"])
    ].copy()
    formal_allowed_providers = set(full_cloud["formal_allowed_providers"])
    mainstream_comparison = mainstream_comparison[
        mainstream_comparison["provider"].isin(formal_allowed_providers)
    ].copy()
    if set(mainstream_comparison["provider"]) != formal_allowed_providers:
        raise ValueError("Formal China provider-origin screen is incomplete")
    for row in mainstream_comparison.itertuples(index=False):
        total = float(row.annual_full_cloud_cost_rmb_proxy)
        total_rows.append(
            {
                "option_name": f"完整云化 {row.provider} / {row.model_id}",
                "annual_total_cost_billion_rmb": total / 1e9,
                "option_group": "full_cloud_hybrid",
                "provider": row.provider,
                "model_id": row.model_id,
                "difference_to_IF_billion_rmb": (total - owned_costs["IF"]) / 1e9,
                "percent_difference_to_IF": total / owned_costs["IF"] - 1.0,
                "difference_to_IG_billion_rmb": (total - owned_costs["IG"]) / 1e9,
                "percent_difference_to_IG": total / owned_costs["IG"] - 1.0,
                "difference_to_II_billion_rmb": (total - owned_costs["II_1host"]) / 1e9,
                "percent_difference_to_II": total / owned_costs["II_1host"] - 1.0,
                "cost_boundary": "office+agent Token API + 四类剩余任务预留GPU-IaaS + 对象存储",
                "comparison_status": full_cloud["result_status"],
                "source_or_role": "mainstream_representative",
            }
        )

    main_reserved = reserved_cloud[
        reserved_cloud["price_case"] == full_cloud["main_gpu_price_case"]
    ].iloc[0]
    total = float(main_reserved["annual_enterprise_payment_rmb"])
    total_rows.append(
        {
            "option_name": "GPU-IaaS 容量订阅（预留基准价）",
            "annual_total_cost_billion_rmb": total / 1e9,
            "option_group": "gpu_iaas_reserved_reference",
            "provider": "cloud_price_proxy",
            "model_id": full_cloud["gpu_billing_mode"],
            "difference_to_IF_billion_rmb": (total - owned_costs["IF"]) / 1e9,
            "percent_difference_to_IF": total / owned_costs["IF"] - 1.0,
            "difference_to_IG_billion_rmb": (total - owned_costs["IG"]) / 1e9,
            "percent_difference_to_IG": total / owned_costs["IG"] - 1.0,
            "difference_to_II_billion_rmb": (total - owned_costs["II_1host"]) / 1e9,
            "percent_difference_to_II": total / owned_costs["II_1host"] - 1.0,
            "cost_boundary": "全工作负荷GPU-IaaS企业付款代理",
            "comparison_status": "existing_cloud_price_sensitivity",
            "source_or_role": f"{model_version} cloud subscription base case",
        }
    )
    total_comparison = pd.DataFrame(total_rows).sort_values("annual_total_cost_billion_rmb")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False, encoding="utf-8-sig")
    total_comparison.to_csv(args.mainstream_output, index=False, encoding="utf-8-sig")
    task_detail.to_csv(args.task_output, index=False, encoding="utf-8-sig")

    mainstream_rows = mainstream_comparison.sort_values("annual_full_cloud_cost_rmb_proxy")
    table_rows = []
    for row in mainstream_rows.itertuples(index=False):
        table_rows.append(
            f"| {row.provider} | `{row.model_id}` | {row.annual_api_token_cost_rmb/1e9:.3f} | "
            f"{row.residual_reserved_gpu_payment_rmb_proxy/1e9:.3f} | "
            f"{row.annual_cloud_storage_cost_rmb_proxy/1e9:.6f} | "
            f"{row.annual_full_cloud_cost_rmb_proxy/1e9:.3f} | {row.difference_to_owned_IG_rmb/1e9:+.3f} | "
            f"{row.ratio_to_owned_IG:.2f}× |"
        )
    representative_table = "\n".join(table_rows)
    low_rows = comparison[
        comparison["mainstream_representative"]
        & (comparison["cloud_reserved_price_case"] == "low")
    ].sort_values("annual_full_cloud_cost_rmb_proxy")
    conservative_low = low_rows.iloc[0]
    findings = f"""# 完整云化成本：Token API + 预留 GPU-IaaS + 对象存储

## 结论状态

本模块已经作为正式企业付款场景接入活动核心模型 `{model_version}`，场景版本为 `{full_cloud['scenario_version']}`，API 价格子版本为 `{settings['cost_case_version']}`。正式中国横向表只展示 **预留 GPU-IaaS**，不展示按量 GPU 订阅，并按厂商来源规则只保留Alibaba Cloud与DeepSeek；其他厂商和轻量型号仅保留在详细审计表中。

## 本轮实际计算

- Token-ready 子集只包括 `office` 和 `agent`；连续视觉、传感器推理、求解器和仿真没有被强制 Token 化。
- 现有项目 2030 基准参数用于 calls/task 和 input/output tokens；行业就业人数作为 eligible-user driver，大型/中小企业参数按 `large_enterprise_share` 混合。
- 当前31行业基线未观测到 `large_enterprise_share`，因此使用配置中的大型企业工作负荷权重 `{float(settings['large_workload_weight_when_unobserved']):.0%}`；这是 D 级敏感性假设。
- cache、batch、retry、reasoning 和邻接工具费均设为 0，代表不假定折扣、重试或隐藏推理量的透明起点。
- 原核心模型中 office+agent 的有效服务份额为 `{token_ready_service_share:.2%}`；其余四类任务占 `{residual_service_share:.2%}`，按这一份额分配全工作负荷预留 GPU 云付款。尚未按剩余任务峰值重新优化云实例数。
- 对象存储按 `{storage_reference}` 架构现有模型权重存储足迹 `{cloud_storage_gb:,.0f}` GB 和生命周期参数 `{full_cloud['cloud_storage_parameter_id']}` 的 `{storage_price:.3f}` 元/GB·月计费；公网流量、请求费和新增数据湖未纳入。
- 年输入 Token 代理为 `{annual_input_tokens/1e12:,.3f}` 万亿，年输出 Token 代理为 `{annual_output_tokens/1e12:,.3f}` 万亿。

## 与本地部署的总成本差异（预留 GPU 基准价）

既有方案年成本为：IF `{owned_costs['IF']/1e9:.3f}` 十亿元、IG `{owned_costs['IG']/1e9:.3f}` 十亿元、II_1host `{owned_costs['II_1host']/1e9:.3f}` 十亿元。主流厂商代表组如下：

| 厂商 | 代表模型 | Token API（十亿元） | 剩余任务预留GPU（十亿元） | 对象存储（十亿元） | 完整云化总成本（十亿元） | 相对IG差额（十亿元） | IG倍数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{representative_table}

即使采用预留 GPU 低价档与主流代表模型中最低的 `{conservative_low['provider']} / {conservative_low['model_id']}` 组合，完整云化年付款仍为 `{float(conservative_low['annual_full_cloud_cost_rmb_proxy'])/1e9:.3f}` 十亿元，是本地 IG 的 `{float(conservative_low['ratio_to_owned_IG']):.2f}` 倍。完整横向比较见 `mainstream_total_cost_comparison.csv`，低/基准/高预留价格敏感性见 `comparison.csv`。

本页不展示按量 GPU 订阅，因为它不符合持续制造业负荷的正式长期采购口径，且会机械性放大云端成本。底层云模块仍保留该原始结果用于内部校验，但不会进入本场景的正式比较表。

这些差额是付款/成本代理的筛查结果。模型之间尚未通过共同制造业质量、时延、可靠性和隐私门槛，不能把较低总成本直接解释为更优采购选择。

## 解释边界

1. 用企业 API usage logs 校准六任务的 `token_ready_share`、调用量、缓存、重试、reasoning 与工具调用；
2. 对相同制造业任务执行模型质量与 SLA benchmark；
3. 从核心优化中真正移除 API 承担的任务，按剩余四类任务的峰值重新优化预留 GPU 实例数，而不是按有效服务份额分摊。
"""
    args.findings_output.write_text(findings, encoding="utf-8")

    payload = {
        "status": "validated_formal_full_cloud_payment_scenario",
        "model_version": model_version,
        "cost_case_version": settings["cost_case_version"],
        "full_cloud_scenario_version": full_cloud["scenario_version"],
        "active_api_price_rows": len(active),
        "mainstream_representative_rows": len(mainstream_prices),
        "task_detail_rows": len(task_detail),
        "price_inputs_official_primary_sources": True,
        "workload_trace_verified": False,
        "quality_equivalence_verified": False,
        "residual_compute_reoptimized": False,
        "formal_payment_scenario_ready": True,
        "scientific_main_result_ready": False,
        "ondemand_displayed": False,
        "reserved_gpu_price_cases": ["low", "base", "high"],
        "cloud_storage_parameter_id": full_cloud["cloud_storage_parameter_id"],
        "checks": [
            "31-industry and six-task effective-service coverage",
            "office and agent only in Token workload",
            "positive active official API prices",
            "unique provider/model/tier price rows",
            "one mainstream representative model per provider",
            "IF, IG, II and base reserved GPU-IaaS rows included in the total-cost comparison",
            "on-demand GPU billing excluded from formal comparison outputs",
            "Token API, reserved GPU-IaaS, and object storage components reconcile to total",
            "native-currency prices converted by explicit FX input",
            "non-Token workload retained",
            "formal enterprise-payment scenario status propagated to outputs",
        ],
    }
    args.done_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
