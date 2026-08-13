"""HISTORICAL national equal-electricity local-versus-cloud screen.

This script only reproduces archived pre-v0.2.0 artifacts. Active national
comparisons use the versioned Snakemake equal-service workflow.

This deliberately avoids provincial and physical grid nodes.  Each of the 31
manufacturing divisions is treated as one local capacity bucket, while cloud
deployment is treated as one pooled capacity bucket.  Equal annual AI facility
electricity is imposed in both architectures to isolate timing, concentration,
and existing-headroom effects.  Results are screening bounds, not a network
planning estimate.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "05_results"
LEGACY_RESULTS = RESULTS / "archive" / "equal_electricity_national"
INPUT = LEGACY_RESULTS / "manufacturing_31sector_peak_screen.csv"
OUTPUT = LEGACY_RESULTS / "manufacturing_simple_local_cloud_grid_screen.csv"
FINDINGS = LEGACY_RESULTS / "manufacturing_simple_local_cloud_grid_findings.md"

LOCAL_HEADROOM_SHARES = [0.0, 0.001, 0.0025, 0.005, 0.01]
CLOUD_CASES = {
    "greenfield_no_headroom": 0.0,
    "reuse_half_ai_peak": 0.5,
}
GRID_CAPEX_RMB_PER_MW = 200.0 * 10_000.0
DISCOUNT_RATE = 0.08
GRID_LIFE_YEARS = 30


def crf(rate: float, years: int) -> float:
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    detail = pd.read_csv(INPUT, encoding="utf-8-sig")
    local = detail[detail["temporal_scenario"] == "task_timed"].copy()
    annualization = crf(DISCOUNT_RATE, GRID_LIFE_YEARS)
    rows: list[dict] = []

    for energy_scenario, frame in local.groupby("energy_scenario"):
        annual_ai_twh = float(frame["annual_ai_twh"].sum())
        ai_average_mw = annual_ai_twh * 1e6 / 8760
        local_ai_peak_sum_mw = float(frame["ai_peak_mw"].sum())
        local_increment_no_headroom_mw = float(frame["incremental_peak_load_mw"].sum())

        # A fully pooled cloud workload is represented by a flat profile.  With
        # equal annual facility electricity, its peak equals its average load.
        cloud_peak_mw = ai_average_mw
        for local_headroom_share in LOCAL_HEADROOM_SHARES:
            available_local_headroom = frame["baseline_peak_load_mw"] * local_headroom_share
            local_expansion_by_sector = (
                frame["incremental_peak_load_mw"] - available_local_headroom
            ).clip(lower=0)
            local_expansion_mw = float(local_expansion_by_sector.sum())
            local_max_single_mw = float(local_expansion_by_sector.max())
            affected_industries = int((local_expansion_by_sector > 1e-9).sum())

            for cloud_case, cloud_headroom_fraction in CLOUD_CASES.items():
                cloud_headroom_mw = cloud_peak_mw * cloud_headroom_fraction
                cloud_expansion_mw = max(0.0, cloud_peak_mw - cloud_headroom_mw)
                local_capex = local_expansion_mw * GRID_CAPEX_RMB_PER_MW
                cloud_capex = cloud_expansion_mw * GRID_CAPEX_RMB_PER_MW
                rows.append(
                    {
                        "energy_scenario": energy_scenario,
                        "annual_ai_twh_equal_both_architectures": annual_ai_twh,
                        "local_temporal_profile": "task_timed_by_industry",
                        "cloud_temporal_profile": "fully_flat_pooled",
                        "local_headroom_share_of_each_industry_baseline_peak": local_headroom_share,
                        "cloud_case": cloud_case,
                        "cloud_headroom_fraction_of_ai_peak": cloud_headroom_fraction,
                        "local_ai_peak_sum_mw": local_ai_peak_sum_mw,
                        "local_incremental_peak_before_headroom_mw": local_increment_no_headroom_mw,
                        "cloud_ai_peak_mw": cloud_peak_mw,
                        "local_grid_expansion_proxy_mw": local_expansion_mw,
                        "cloud_grid_expansion_proxy_mw": cloud_expansion_mw,
                        "local_minus_cloud_expansion_mw": local_expansion_mw - cloud_expansion_mw,
                        "local_max_single_bucket_expansion_mw": local_max_single_mw,
                        "cloud_max_single_bucket_expansion_mw": cloud_expansion_mw,
                        "local_industries_triggering_expansion": affected_industries,
                        "local_grid_capex_proxy_billion_rmb": local_capex / 1e9,
                        "cloud_grid_capex_proxy_billion_rmb": cloud_capex / 1e9,
                        "local_annualized_grid_cost_proxy_million_rmb_per_year": local_capex * annualization / 1e6,
                        "cloud_annualized_grid_cost_proxy_million_rmb_per_year": cloud_capex * annualization / 1e6,
                    }
                )

    write_csv(OUTPUT, rows)
    result = pd.DataFrame(rows)
    assert len(result) == 3 * len(LOCAL_HEADROOM_SHARES) * len(CLOUD_CASES)
    assert (result["local_grid_expansion_proxy_mw"] >= 0).all()
    assert (result["cloud_grid_expansion_proxy_mw"] >= 0).all()
    for (_, cloud_case), group in result.groupby(["energy_scenario", "cloud_case"]):
        ordered = group.sort_values("local_headroom_share_of_each_industry_baseline_peak")
        assert ordered["local_grid_expansion_proxy_mw"].is_monotonic_decreasing
    central = result[
        (result["energy_scenario"] == "central_14twh")
        & (result["cloud_case"] == "greenfield_no_headroom")
    ].sort_values("local_headroom_share_of_each_industry_baseline_peak")

    central_detail = local[local["energy_scenario"] == "central_14twh"]

    def local_expansion_at(headroom_share: float) -> float:
        return float(
            (
                central_detail["incremental_peak_load_mw"]
                - central_detail["baseline_peak_load_mw"] * headroom_share
            ).clip(lower=0).sum()
        )

    def crossing_share(target_mw: float) -> float:
        low, high = 0.0, 0.02
        for _ in range(80):
            midpoint = (low + high) / 2
            if local_expansion_at(midpoint) > target_mw:
                low = midpoint
            else:
                high = midpoint
        return (low + high) / 2

    greenfield_crossing = crossing_share(float(central.iloc[0]["cloud_grid_expansion_proxy_mw"]))
    half_cloud_crossing = crossing_share(float(central.iloc[0]["cloud_grid_expansion_proxy_mw"]) * 0.5)
    lines = [
        "# 全国制造业本地—云端简化电网筛查",
        "",
        "## 比较口径",
        "",
        "- 31个制造业大类分别视为31个本地容量桶；不区分省份、企业和真实电网节点。",
        "- 本地采用行业任务时序曲线；云端采用完全池化的平坦负荷，这一设定有利于云端。",
        "- 两种架构使用相同AI年电量，只识别负荷时序、空间集中和既有余量，不识别PUE、服务器利用率和硬件数量差异。",
        "- 本地余量按各行业原峰值的比例设置；云端主情景为绿地建设、无既有余量。",
        "- 扩容投资使用200万元/MW连续代理，并按8%折现率、30年寿命年化。",
        "",
        "## 14 TWh中心情景",
        "",
        "| 各行业既有余量/原峰值 | 本地扩容代理（MW） | 云端扩容代理（MW） | 本地－云端（MW） | 本地最大单桶（MW） | 触发扩容行业数 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in central.to_dict("records"):
        lines.append(
            f'| {row["local_headroom_share_of_each_industry_baseline_peak"] * 100:.2f}% '
            f'| {row["local_grid_expansion_proxy_mw"]:.1f} '
            f'| {row["cloud_grid_expansion_proxy_mw"]:.1f} '
            f'| {row["local_minus_cloud_expansion_mw"]:+.1f} '
            f'| {row["local_max_single_bucket_expansion_mw"]:.1f} '
            f'| {int(row["local_industries_triggering_expansion"])} |'
        )
    base = central[central["local_headroom_share_of_each_industry_baseline_peak"] == 0.0025].iloc[0]
    lines.extend(
        [
            "",
            "## 初步解释",
            "",
            f'中心情景下，若本地没有任何既有余量，本地扩容代理为 {central.iloc[0]["local_grid_expansion_proxy_mw"]:.1f} MW，高于平坦云端的 {central.iloc[0]["cloud_grid_expansion_proxy_mw"]:.1f} MW；本地并不天然节省扩容。',
            "",
            f'若各行业平均拥有相当于原峰值0.25%的可用余量，本地扩容代理降至 {base["local_grid_expansion_proxy_mw"]:.1f} MW，比绿地云端少 {abs(base["local_minus_cloud_expansion_mw"]):.1f} MW。对应连续投资代理分别约为 {base["local_grid_capex_proxy_billion_rmb"]:.2f} 和 {base["cloud_grid_capex_proxy_billion_rmb"]:.2f} 十亿元。',
            "",
            f'按这一简化模型，只要本地可用余量达到各行业原峰值的约 {greenfield_crossing * 100:.3f}%，本地总扩容代理就低于无余量绿地云端；若云端已有相当于AI峰值50%的可用余量，本地临界余量上升到约 {half_cloud_crossing * 100:.3f}%。',
            "",
            f'即使零余量，本地最大单一行业容量桶扩容也只有 {central.iloc[0]["local_max_single_bucket_expansion_mw"]:.1f} MW，而单一云端容量桶为 {central.iloc[0]["cloud_max_single_bucket_expansion_mw"]:.1f} MW。分布式部署最稳健的优势首先是降低最大单点规模，而总扩容是否更低取决于工厂既有余量。',
            "",
            "## 不能从本筛查推出的结论",
            "",
            "- 不能把31个行业桶当作真实电网节点，因此不能声称已经避免了具体输变电工程。",
            "- 未计入本地PUE较高、服务器空闲功率和最小整机投资，也未计入云端真实物理资源成本。",
            "- 未加入光伏储能；本地余量在行业内部是否与AI负荷同址仍未知。",
            "- 云端若利用已有数据中心余量，其扩容会低于绿地主情景，结果表保留了50%云端余量边界。",
        ]
    )
    FINDINGS.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "rows": len(rows),
                "central_greenfield": central[
                    [
                        "local_headroom_share_of_each_industry_baseline_peak",
                        "local_grid_expansion_proxy_mw",
                        "cloud_grid_expansion_proxy_mw",
                        "local_max_single_bucket_expansion_mw",
                    ]
                ].to_dict("records"),
                "output": str(OUTPUT),
                "findings": str(FINDINGS),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
