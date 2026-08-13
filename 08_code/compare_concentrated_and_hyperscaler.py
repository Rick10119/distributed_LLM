"""Compare concentrated ownership with hyperscaler procurement at two boundaries.

The enterprise boundary captures payments and impacts at manufacturing premises.
The system boundary retains provider-side physical impacts. Provider water is an
equal-service proxy using II_1host IT electricity, not measured marginal cloud use.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "02_data" / "processed" / "resource_footprint"
RESULT_ROOT = ROOT / "05_results" / "v0.6.0" / "result"
RESOURCE_SUMMARY = DATA_ROOT / "resource_footprint_architecture_summary_mvp.csv"
PROVIDER_WATER = DATA_ROOT / "small_china_us_water_baseline_parameters.csv"
CLOUD_PAYMENT = RESULT_ROOT / "cloud_subscription" / "comparison.csv"
OUTPUT = DATA_ROOT / "concentrated_vs_hyperscaler_comparison.csv"
BASELINE_OUTPUT = DATA_ROOT / "small_china_us_water_baseline_comparison.csv"
LINEAGE_OUTPUT = DATA_ROOT / "concentrated_vs_hyperscaler_comparison.lineage.json"
FINDINGS_OUTPUT = RESULT_ROOT / "resource_footprint_mvp" / "concentrated_vs_hyperscaler_findings.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resource = pd.read_csv(RESOURCE_SUMMARY, encoding="utf-8-sig")
    provider = pd.read_csv(PROVIDER_WATER, encoding="utf-8-sig")
    payment = pd.read_csv(CLOUD_PAYMENT, encoding="utf-8-sig")
    ii = resource[
        (resource["architecture"] == "II_1host")
        & (resource["reuse_case"] == "conditional_reuse")
    ]
    if set(ii["water_case"]) != {"low_water", "central", "high_water"} or len(ii) != 3:
        raise ValueError("Expected three II_1host water cases under conditional reuse")
    expected_modes = {
        "small_factory_site",
        "china_hyperscaler_harmonized_baseline",
        "us_hyperscaler_harmonized_baseline",
    }
    if len(provider) != 3 or set(provider["comparison_mode"]) != expected_modes:
        raise ValueError("Expected one baseline parameter for small, China-cloud, and US-cloud modes")
    if (provider["site_water_use_l_per_kwh_it"] <= 0).any():
        raise ValueError("Baseline site-water parameters must be positive")
    base_cloud = payment[
        payment["mode"].isin(["cloud_reserved_capacity", "cloud_ondemand"])
        & payment["price_case"].eq("base")
    ]
    owned = payment[payment["mode"].eq("owned_II_1host")]
    if len(base_cloud) != 2 or len(owned) != 1:
        raise ValueError("Cloud payment comparison is missing base cloud or owned II rows")
    return ii, provider, pd.concat([owned, base_cloud], ignore_index=True)


def build_comparison() -> pd.DataFrame:
    ii, provider, payment = load_inputs()
    owned_payment = payment[payment["mode"].eq("owned_II_1host")].iloc[0]
    rows: list[dict[str, object]] = []

    central = ii[ii["water_case"].eq("central")].iloc[0]
    rows.append(
        {
            "deployment_mode": "concentrated_industry_owned_II_1host",
            "billing_mode": "owned_capacity",
            "physical_water_case": "baseline",
            "water_anchor": "large_facility_central_scenario",
            "site_water_use_l_per_kwh_it": central["national_site_water_use_m3"]
            / (central["national_annual_it_energy_twh"] * 1_000_000.0),
            "same_service_it_energy_twh_proxy": central["national_annual_it_energy_twh"],
            "annual_enterprise_payment_rmb": owned_payment["annual_enterprise_payment_rmb"],
            "payment_ratio_to_owned_II_1host": 1.0,
            "manufacturing_enterprise_site_water_m3": central["national_site_water_use_m3"],
            "system_allocated_site_water_m3": central["national_site_water_use_m3"],
            "manufacturing_site_new_land_status": "required_under_conditional_reuse_screen",
            "system_new_land_status": "required_under_conditional_reuse_screen",
            "manufacturing_site_continuous_cooling_noise_status": "present_screen_only",
            "system_continuous_cooling_noise_status": "present_screen_only",
            "provider_location_status": "31_industry_pool_hosts_not_real_cloud_locations",
            "physical_resource_status": "modeled_owned_capacity",
            "interpretation": "Enterprise and system boundaries coincide for owned concentrated deployment",
        }
    )
    it_energy = float(central["national_annual_it_energy_twh"])
    china_water = provider[
        provider["comparison_mode"].eq("china_hyperscaler_harmonized_baseline")
    ].iloc[0]
    for _, cloud in payment[payment["mode"].str.startswith("cloud_")].iterrows():
        allocated_water = (
            it_energy * float(china_water["site_water_use_l_per_kwh_it"]) * 1_000_000.0
        )
        rows.append(
            {
                "deployment_mode": "hyperscaler_GPU_IaaS_purchase",
                "billing_mode": cloud["mode"],
                "physical_water_case": "baseline",
                "water_anchor": "harmonized large-cloud cooling and humidification water baseline",
                "site_water_use_l_per_kwh_it": china_water["site_water_use_l_per_kwh_it"],
                "same_service_it_energy_twh_proxy": it_energy,
                "annual_enterprise_payment_rmb": cloud["annual_enterprise_payment_rmb"],
                "payment_ratio_to_owned_II_1host": cloud["ratio_to_owned_II_1host"],
                "manufacturing_enterprise_site_water_m3": 0.0,
                "system_allocated_site_water_m3": allocated_water,
                "manufacturing_site_new_land_status": "none_for_incremental_AI_compute_facility",
                "system_new_land_status": "unknown_depends_on_provider_marginal_capacity_realization",
                "manufacturing_site_continuous_cooling_noise_status": "none_for_incremental_AI_compute_facility",
                "system_continuous_cooling_noise_status": "present_at_provider_but_unquantified",
                "provider_location_status": "not_allocated_to_provider_facilities",
                "physical_resource_status": "same_service_II_IT_energy_times_harmonized_large_cloud_water_baseline",
                "interpretation": "Zero at the manufacturing-site boundary is a transfer to provider facilities, not zero system impact",
            }
        )
    comparison = pd.DataFrame(rows)
    if len(comparison) != 3:
        raise ValueError(f"Expected 3 baseline comparison rows, found {len(comparison)}")
    if not comparison.loc[
        comparison["deployment_mode"].eq("hyperscaler_GPU_IaaS_purchase"),
        "manufacturing_enterprise_site_water_m3",
    ].eq(0).all():
        raise ValueError("Cloud procurement must have zero manufacturing-site compute water")
    if (comparison["system_allocated_site_water_m3"] <= 0).any():
        raise ValueError("All system-boundary water proxies must be positive")
    return comparison


def build_small_china_us_baseline() -> pd.DataFrame:
    ii, parameters, _ = load_inputs()
    central = ii[ii["water_case"].eq("central")].iloc[0]
    common_it_energy = float(central["national_annual_it_energy_twh"])
    result = parameters.copy()
    result["same_service_it_energy_twh_proxy"] = common_it_energy
    result["annual_site_water_m3_proxy"] = (
        result["site_water_use_l_per_kwh_it"] * common_it_energy * 1_000_000.0
    )
    result["comparison_boundary"] = "same_IT_energy_and_onsite_cooling_humidification_water_boundary_not_split_withdrawal_or_consumption"
    return result.sort_values("site_water_use_l_per_kwh_it").reset_index(drop=True)


def write_findings(comparison: pd.DataFrame) -> None:
    owned = comparison[
        comparison["deployment_mode"].eq("concentrated_industry_owned_II_1host")
    ].iloc[0]
    reserved = comparison[
        comparison["billing_mode"].eq("cloud_reserved_capacity")
    ].iloc[0]
    ondemand = comparison[
        comparison["billing_mode"].eq("cloud_ondemand")
    ].iloc[0]
    baseline = build_small_china_us_baseline()
    baseline_rows = "\n".join(
        f"| {row.display_name} | {row.site_water_use_l_per_kwh_it:.3f} | "
        f"{row.annual_site_water_m3_proxy / 1e6:.3f} |"
        for row in baseline.itertuples()
    )
    findings = f"""# 行业集中自建与大型云厂商采购对照

更新日期：{date.today().isoformat()}  
功能单位：与 `v0.6.0` 全国制造业情景相同的有效 AI 服务  
结果性质：企业付款与现场资源的双边界筛查

## 核心结果

行业集中自建 `II_1host` 的年化企业成本为 {owned.annual_enterprise_payment_rmb / 1e9:.3f} 十亿元。公开价格代理下，大型云厂商 GPU-IaaS 容量订阅为 {reserved.annual_enterprise_payment_rmb / 1e9:.3f} 十亿元/年，是集中自建的 {reserved.payment_ratio_to_owned_II_1host:.2f} 倍；按量采购为 {ondemand.annual_enterprise_payment_rmb / 1e9:.3f} 十亿元/年，是 {ondemand.payment_ratio_to_owned_II_1host:.2f} 倍。

在制造企业现场边界，向大型云厂商采购后，增量 AI 计算设施的冷却用水、新增土地、机房和连续冷却噪声记为零；这些影响不是消失，而是转移到供应商数据中心。行业集中自建的中央水情景为 {owned.system_allocated_site_water_m3 / 1e6:.3f} 百万 m³/年，企业边界与系统边界相同。

在系统边界，使用 `II_1host` 的 {owned.same_service_it_energy_twh_proxy:.3f} TWh/年 IT 电量作为统一代理，并按用户要求统一水量口径。中国和美国大型云不再直接采用不可比的国家披露组合，而共同使用0.335 L/kWh-IT：该值由阿里主要风冷数据中心组合0.329与Microsoft FY25 Americas加湿/冷却WUE 0.340的等权均值0.3345四舍五入得到。小型工厂服务器使用不超过50 kW容量档的中央情景0.050：

| 基准模式 | WUE/现场水强度（L/kWh IT） | 同等IT电量下现场水代理（百万 m³/年） |
|---|---:|---:|
{baseline_rows}

统一大型云基准对应 {reserved.system_allocated_site_water_m3 / 1e6:.3f} 百万 m³/年，约为集中自建中央情景的 {reserved.system_allocated_site_water_m3 / owned.system_allocated_site_water_m3:.2f} 倍。三行均统一为“冷却和加湿现场水/IT电量”的筛查口径，但尚未进一步拆成取水、耗水和淡水份额。

## 尚不能比较的项目

- 云商是否用既有容量还是为新增需求扩建，因而系统边界的新增建筑和土地仍为未知；
- 云商负荷落在哪些数据中心，因而最大单点用水、水稀缺度和厂界噪声不能计算；
- 云商实际服务本研究工作负荷的服务器效率、利用率和 PUE，因而当前不能声称云采购比集中自建更省水；
- 中美大型云使用相同技术口径的研究构造基准，不代表任一国家的观测平均；集中自建指标也尚未校准为同口径取水或耗水；
- 两种模式的淡水份额仍未知。

因此，这一版可以回答“企业现场影响是否转移”和“公开采购付款大致多高”，但不能把组合 WUE 代理写成云采购的边际社会水足迹。
"""
    FINDINGS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_OUTPUT.write_text(findings, encoding="utf-8")


def main() -> None:
    comparison = build_comparison()
    baseline = build_small_china_us_baseline()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    baseline.to_csv(BASELINE_OUTPUT, index=False, encoding="utf-8-sig")
    write_findings(comparison)
    lineage = {
        "generated_on": date.today().isoformat(),
        "model_version": "v0.6.0",
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "inputs": {
            str(RESOURCE_SUMMARY.relative_to(ROOT)): sha256(RESOURCE_SUMMARY),
            str(PROVIDER_WATER.relative_to(ROOT)): sha256(PROVIDER_WATER),
            str(CLOUD_PAYMENT.relative_to(ROOT)): sha256(CLOUD_PAYMENT),
        },
        "outputs": [
            str(OUTPUT.relative_to(ROOT)),
            str(BASELINE_OUTPUT.relative_to(ROOT)),
            str(FINDINGS_OUTPUT.relative_to(ROOT)),
        ],
        "row_count": len(comparison),
        "boundary_rule": "enterprise manufacturing-site impacts are distinct from provider-side system impacts",
        "physical_proxy": "baseline-only comparison using small-facility central scenario and a shared harmonized large-cloud water anchor derived from Alibaba air-cooled and Microsoft Americas references",
        "excluded": [
            "electricity-generation water",
            "provider-specific workload energy and facility assignment",
            "marginal provider land and building construction",
            "provider boundary-noise propagation",
        ],
    }
    LINEAGE_OUTPUT.write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(lineage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
