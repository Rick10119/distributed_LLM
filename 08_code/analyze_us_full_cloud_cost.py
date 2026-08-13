#!/usr/bin/env python3
"""Calculate a version-aligned US-region full-cloud payment counterfactual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def parameter_value(parameters: pd.DataFrame, parameter_id: str, price_case: str) -> float:
    rows = parameters[parameters["parameter_id"] == parameter_id]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {parameter_id}, found {len(rows)}")
    column = f"{price_case}_value"
    value = rows.iloc[0][column]
    if pd.isna(value):
        raise ValueError(f"Missing {column} for {parameter_id}")
    return float(value)


def tiered_storage_cost_usd(
    storage_gb: float,
    first_tier_gb: float,
    first_rate: float,
    second_rate: float,
) -> float:
    first = min(storage_gb, first_tier_gb)
    second = max(storage_gb - first_tier_gb, 0.0)
    return 12.0 * (first * first_rate + second * second_rate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud-comparison", required=True)
    parser.add_argument("--token-demand", required=True)
    parser.add_argument("--china-full-cloud", required=True)
    parser.add_argument("--us-owned-cost", required=True)
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--api-prices", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    model_version = str(settings["physical_model_version"])

    cloud = pd.read_csv(args.cloud_comparison, encoding="utf-8-sig")
    tokens = pd.read_csv(args.token_demand, encoding="utf-8-sig")
    china_full_cloud = pd.read_csv(args.china_full_cloud, encoding="utf-8-sig")
    owned = pd.read_csv(args.us_owned_cost, encoding="utf-8-sig")
    parameters = pd.read_csv(args.parameters, encoding="utf-8-sig")
    api_prices = pd.read_csv(args.api_prices, encoding="utf-8-sig")

    if set(cloud["model_version"]) != {model_version}:
        raise ValueError("Cloud physical result version mismatch")
    if set(tokens["model_version"]) != {model_version}:
        raise ValueError("Token demand version mismatch")
    if len(api_prices) != 5 or api_prices["provider"].nunique() != 5:
        raise ValueError("US price panel must contain exactly five providers")
    if bool(settings["display_ondemand"]):
        raise ValueError("Formal US comparison must exclude on-demand GPU")

    reserved = cloud[
        (cloud["mode"] == "cloud_reserved_capacity")
        & (cloud["price_case"] == settings["gpu_price_case"])
    ]
    if len(reserved) != 1:
        raise ValueError("Expected one base reserved-capacity physical row")
    contracted_instances = float(reserved.iloc[0]["contracted_dual_l20_instances"])

    proxy_columns = ["token_ready_service_share_proxy", "cloud_storage_gb_proxy"]
    proxy = china_full_cloud[proxy_columns].drop_duplicates()
    if len(proxy) != 1:
        raise ValueError("Existing full-cloud model must provide one service/storage proxy")
    token_share = float(proxy.iloc[0]["token_ready_service_share_proxy"])
    residual_share = 1.0 - token_share
    storage_gb = float(proxy.iloc[0]["cloud_storage_gb_proxy"])

    annual_input_tokens = float(tokens["annual_input_tokens"].sum())
    annual_output_tokens = float(tokens["annual_output_tokens"].sum())
    reserved_unit_price = parameter_value(
        parameters, settings["gpu_parameter_id"], settings["gpu_price_case"]
    )
    full_gpu_payment = contracted_instances * reserved_unit_price
    residual_gpu_payment = full_gpu_payment * residual_share
    storage_payment = tiered_storage_cost_usd(
        storage_gb,
        float(settings["storage_first_tier_gb"]),
        float(settings["storage_first_tier_usd_per_gb_month"]),
        float(settings["storage_second_tier_usd_per_gb_month"]),
    )

    owned_base = owned[owned["server_price_case"] == "base"].copy()
    if set(owned_base["scenario"]) != {"IF", "IG", "II_1host"}:
        raise ValueError("US owned-cost result is incomplete")
    owned_costs = owned_base.set_index("scenario")["annual_us_owned_core_cost_usd"].to_dict()

    records: list[dict[str, object]] = []
    for row in api_prices.itertuples(index=False):
        api_cost = (
            annual_input_tokens / 1e6 * float(row.input_usd_per_mtoken)
            + annual_output_tokens / 1e6 * float(row.output_usd_per_mtoken)
        )
        total = residual_gpu_payment + api_cost + storage_payment
        record: dict[str, object] = {
            "model_version": model_version,
            "cost_case_version": settings["cost_case_version"],
            "country": "US",
            "region": row.region,
            "provider": row.provider,
            "platform": row.platform,
            "model_id": row.model_id,
            "annual_input_tokens": annual_input_tokens,
            "annual_output_tokens": annual_output_tokens,
            "input_usd_per_mtoken": row.input_usd_per_mtoken,
            "output_usd_per_mtoken": row.output_usd_per_mtoken,
            "annual_api_token_cost_usd": api_cost,
            "token_ready_service_share_proxy": token_share,
            "residual_non_token_service_share_proxy": residual_share,
            "contracted_dual_gpu_equivalent_instances": contracted_instances,
            "reserved_gpu_unit_price_usd_per_year": reserved_unit_price,
            "full_workload_reserved_gpu_payment_usd": full_gpu_payment,
            "residual_reserved_gpu_payment_usd_proxy": residual_gpu_payment,
            "cloud_storage_gb_proxy": storage_gb,
            "annual_cloud_storage_cost_usd_proxy": storage_payment,
            "annual_us_full_cloud_cost_usd": total,
            "annual_us_full_cloud_cost_billion_usd": total / 1e9,
            "ratio_to_us_owned_IF": total / owned_costs["IF"],
            "ratio_to_us_owned_IG": total / owned_costs["IG"],
            "ratio_to_us_owned_II_1host": total / owned_costs["II_1host"],
            "difference_to_us_owned_IG_usd": total - owned_costs["IG"],
            "gpu_hardware_basis": "2x AWS g6e.4xlarge = 2x L40S 48GB capacity proxy",
            "gpu_billing_mode": "one_year_standard_reserved_all_upfront",
            "ondemand_displayed": False,
            "quality_equivalence_verified": False,
            "residual_compute_reoptimized": False,
            "result_status": settings["result_status"],
            "price_source_url": row.source_url,
        }
        records.append(record)

    all_provider_audit = pd.DataFrame(records).sort_values("annual_us_full_cloud_cost_usd")
    all_provider_audit.to_csv(output_dir / "us_full_cloud_all_provider_audit.csv", index=False)
    us_allowed = set(settings["formal_us_allowed_providers"])
    detail = all_provider_audit[all_provider_audit["provider"].isin(us_allowed)].copy()
    if set(detail["provider"]) != us_allowed:
        raise ValueError("Formal US provider-origin screen did not resolve all configured providers")
    detail.to_csv(output_dir / "us_full_cloud_comparison.csv", index=False)

    total_rows: list[dict[str, object]] = []
    for row in owned_base.sort_values("scenario").itertuples(index=False):
        total_rows.append(
            {
                "option": f"US local {row.scenario}",
                "option_group": "us_local_owned",
                "provider": "local",
                "model_id": row.scenario,
                "annual_cost_billion_local_currency": row.annual_us_owned_core_cost_billion_usd,
                "local_currency": "USD",
                "ratio_to_us_owned_IG": row.annual_us_owned_core_cost_usd / owned_costs["IG"],
                "payment_boundary": "owned physical capacity and incremental operating cost",
            }
        )
    for row in detail.itertuples(index=False):
        total_rows.append(
            {
                "option": f"US full cloud {row.provider} / {row.model_id}",
                "option_group": "us_full_cloud_hybrid",
                "provider": row.provider,
                "model_id": row.model_id,
                "annual_cost_billion_local_currency": row.annual_us_full_cloud_cost_billion_usd,
                "local_currency": "USD",
                "ratio_to_us_owned_IG": row.ratio_to_us_owned_IG,
                "payment_boundary": "office+agent API + four residual tasks reserved GPU + S3",
            }
        )
    pd.DataFrame(total_rows).to_csv(output_dir / "us_local_cloud_total_comparison.csv", index=False)

    china_mainstream_audit = china_full_cloud[
        (china_full_cloud["mainstream_representative"].astype(str).str.lower() == "true")
        & (china_full_cloud["cloud_reserved_price_case"] == "base")
    ]
    if len(china_mainstream_audit) != 5:
        raise ValueError("Expected five base-price models in the existing China-facing panel")
    china_allowed = set(settings["formal_china_allowed_providers"])
    china_mainstream = china_mainstream_audit[
        china_mainstream_audit["provider"].isin(china_allowed)
    ].copy()
    if set(china_mainstream["provider"]) != china_allowed:
        raise ValueError("Formal China provider-origin screen did not resolve all configured providers")
    china_owned_ig = float(
        (china_mainstream["annual_full_cloud_cost_rmb_proxy"] / china_mainstream["ratio_to_owned_IG"]).median()
    )
    country_summary = pd.DataFrame(
        [
            {
                "price_environment": "China formal provider-origin panel",
                "local_IG_cost_billion_local_currency": china_owned_ig / 1e9,
                "local_currency": "CNY",
                "full_cloud_min_billion_local_currency": china_mainstream[
                    "annual_full_cloud_cost_rmb_proxy"
                ].min() / 1e9,
                "full_cloud_max_billion_local_currency": china_mainstream[
                    "annual_full_cloud_cost_rmb_proxy"
                ].max() / 1e9,
                "full_cloud_min_ratio_to_local_IG": china_mainstream["ratio_to_owned_IG"].min(),
                "full_cloud_max_ratio_to_local_IG": china_mainstream["ratio_to_owned_IG"].max(),
                "scope_note": "Formal provider-origin screen: Alibaba Cloud and DeepSeek only; China GPU/storage prices",
            },
            {
                "price_environment": "US regional panel",
                "local_IG_cost_billion_local_currency": owned_costs["IG"] / 1e9,
                "local_currency": "USD",
                "full_cloud_min_billion_local_currency": detail[
                    "annual_us_full_cloud_cost_billion_usd"
                ].min(),
                "full_cloud_max_billion_local_currency": detail[
                    "annual_us_full_cloud_cost_billion_usd"
                ].max(),
                "full_cloud_min_ratio_to_local_IG": detail["ratio_to_us_owned_IG"].min(),
                "full_cloud_max_ratio_to_local_IG": detail["ratio_to_us_owned_IG"].max(),
                "scope_note": "Formal provider-origin screen: OpenAI, Anthropic and Google only; US regional prices",
            },
        ]
    )
    country_summary.to_csv(output_dir / "country_price_environment_summary.csv", index=False)

    table_lines = []
    for row in detail.itertuples(index=False):
        table_lines.append(
            f"| {row.provider} / {row.model_id} | {row.annual_api_token_cost_usd/1e9:.3f} | "
            f"{row.residual_reserved_gpu_payment_usd_proxy/1e9:.3f} | "
            f"{row.annual_cloud_storage_cost_usd_proxy/1e6:.3f} | "
            f"{row.annual_us_full_cloud_cost_billion_usd:.3f} | {row.ratio_to_us_owned_IG:.3f} |"
        )
    cheapest = detail.iloc[0]
    findings = f"""# 美国本地与完整云化成本比较

模型版本：`{model_version}`；成本案例：`{settings['cost_case_version']}`。

## 结果

美国本地 IG 基准为 **{owned_costs['IG']/1e9:.3f} 十亿美元/年**，与中国本地部署共用唯一的10%装机峰值裕量。美国完整云化场景保留一年期预留 GPU，不展示按量 GPU；office/agent 走美国区 Token API，其余四类任务走预留 GPU，另加 S3 Standard。正式美国面板按厂商来源规则只保留OpenAI、Anthropic和Google；Qwen与DeepSeek只留在审计表，不进入美国正式结论。

| 美国区 API 代表模型 | API（十亿美元） | 剩余预留 GPU（十亿美元） | 存储（百万美元） | 合计（十亿美元） | 相对美国本地 IG |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table_lines)}

正式美国面板最低结果是 **{cheapest['provider']} / {cheapest['model_id']}：{cheapest['annual_us_full_cloud_cost_billion_usd']:.3f} 十亿美元/年，为本地 IG 的 {cheapest['ratio_to_us_owned_IG']:.3f} 倍**；三种合规组合范围为本地IG的 **{detail['ratio_to_us_owned_IG'].min():.3f}—{detail['ratio_to_us_owned_IG'].max():.3f}倍**。

中国正式面板按同一厂商来源规则只保留Alibaba Cloud和DeepSeek，排除OpenAI、Anthropic和Google。中国本地IG为 **{china_owned_ig/1e9:.3f}十亿元/年**，两个中国厂商完整云化组合为 **{china_mainstream['annual_full_cloud_cost_rmb_proxy'].min()/1e9:.3f}—{china_mainstream['annual_full_cloud_cost_rmb_proxy'].max()/1e9:.3f}十亿元/年**，即本地IG的 **{china_mainstream['ratio_to_owned_IG'].min():.3f}—{china_mainstream['ratio_to_owned_IG'].max():.3f}倍**。其中DeepSeek沿用现有全球API美元价折算人民币，仍不是严格中国大陆区域价格；取得中国大陆官方价后应替换。

## 口径与限制

- 预留 GPU 单价为 **{reserved_unit_price:,.0f} 美元/双卡等效单元·年**，对应两台 AWS `g6e.4xlarge` 一年标准全预付。两台合计 2 张 L40S 48GB、32 vCPU、256 GiB，与原双 L20 容量边界一致，但 L40S 性能更高，因此是偏保守的付款代理，不是原生 L20 报价。
- Token 可替代服务份额沿用现有模型的 {token_share:.3%}，剩余 GPU 份额为 {residual_share:.3%}；没有按任务峰值重新优化，因此仍是容量代理。
- S3 采用前 50TB 0.023 美元/GB·月、其后 0.022 美元/GB·月；请求、外网流量、支持、税和迁移集成未计入。
- 中美比较必须分别在本币价格环境内计算比值，不能把中国云价与美国本地成本交叉比较。
- 厂商来源筛选是本情景的政策假设，不代表技术上或法律上对所有企业均绝对不可用；未入选厂商保留在 `us_full_cloud_all_provider_audit.csv`。
"""
    (output_dir / "findings.md").write_text(findings, encoding="utf-8")

    done = {
        "status": "complete_validated_us_full_cloud_payment_counterfactual",
        "model_version": model_version,
        "cost_case_version": settings["cost_case_version"],
        "providers": int(len(detail)),
        "ondemand_displayed": False,
        "minimum_ratio_to_us_owned_IG": float(detail["ratio_to_us_owned_IG"].min()),
        "maximum_ratio_to_us_owned_IG": float(detail["ratio_to_us_owned_IG"].max()),
    }
    (output_dir / "validated.done.json").write_text(
        json.dumps(done, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
