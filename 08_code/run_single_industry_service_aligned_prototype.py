"""Run one equal-service local-versus-cloud prototype for C36 automobiles.

This is deliberately a single-industry mechanism test.  It does not loop over
all 31 manufacturing divisions and does not overwrite historical national
screening outputs.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "02_data"
RESULTS = ROOT / "05_results"
sys.path.insert(0, str(ROOT / "08_code"))

import china_minimum_prototype as parameter_core  # noqa: E402
import run_manufacturing_31sector_peak_screen as profile_core  # noqa: E402
from service_aligned_flexibility_core import (  # noqa: E402
    Architecture,
    FlexibleArrival,
    ServerTechnology,
    calibrate_service_scale_to_reference_energy,
    facility_power_mw,
    optimize_peak_with_highs,
    provision_server_groups,
)


INDUSTRY_CODE = "C36"
INDUSTRY_NAME = "汽车制造业"
ENERGY_SCENARIO = "central_14twh"
FLEXIBILITY_SCENARIO = "central"
LOCAL_HEADROOM_SHARE_OF_BASE_PEAK = 0.0025
ANNUAL_DAYS = 365.0

SUMMARY_OUTPUT = RESULTS / "c36_service_aligned_prototype_summary.csv"
HOURLY_OUTPUT = RESULTS / "c36_service_aligned_prototype_hourly.csv"
FINDINGS_OUTPUT = RESULTS / "c36_service_aligned_prototype_findings.md"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_inputs() -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, dict[str, float]],
    float,
    float,
]:
    hourly = pd.read_csv(
        RESULTS / "manufacturing_31sector_hourly_peak_profiles.csv",
        encoding="utf-8-sig",
    )
    c36 = hourly[
        (hourly["industry_code"] == INDUSTRY_CODE)
        & (hourly["temporal_scenario"] == "task_timed")
    ].sort_values("hour")
    if len(c36) != 24:
        raise ValueError("C36 hourly baseline must contain 24 rows")
    base_load_mw = c36["baseline_load_mw"].to_numpy(dtype=float)
    base_normalized = c36["base_normalized_load"].to_numpy(dtype=float)

    office_shape, agent_shape = profile_core.parse_hourly_workload_shapes()
    task_shapes = profile_core.fixed_task_shapes(
        base_normalized, office_shape, agent_shape
    )

    task_results = pd.read_csv(
        RESULTS / "manufacturing_ai_task_hardware_demo_task_results.csv",
        encoding="utf-8-sig",
    )
    task_rows = task_results[
        (task_results["industry_code"] == INDUSTRY_CODE)
        & (task_results["year"] == 2030)
    ]
    if set(task_rows["task_id"]) != set(task_shapes):
        raise ValueError("C36 task set does not match the six workload shapes")
    unscaled_daily_task_service = {
        row.task_id: float(row.future_accelerator_gpu_h_day)
        for row in task_rows.itertuples()
    }

    mapping = pd.read_csv(
        DATA / "manufacturing_ai_task_flexibility_mapping.csv",
        encoding="utf-8-sig",
    )
    mapping = mapping[mapping["scenario"] == FLEXIBILITY_SCENARIO]
    flexibility = {
        row.task_id: {
            "R": float(row.rho_rigid),
            "F_day": float(row.rho_intraday),
            "F_batch": float(row.rho_batch),
            "F_day_deadline": int(row.intraday_deadline_h),
            "F_batch_deadline": int(row.batch_deadline_h),
        }
        for row in mapping.itertuples()
    }
    if set(flexibility) != set(task_shapes):
        raise ValueError("Flexibility matrix does not cover the six C36 tasks")

    allocation = pd.read_csv(
        RESULTS / "manufacturing_topdown_allocation_31sectors.csv",
        encoding="utf-8-sig",
    )
    allocation_row = allocation[allocation["industry_code"] == INDUSTRY_CODE].iloc[0]
    reference_energy_twh = float(allocation_row["central_14twh_allocation_twh"])
    baseline_energy_twh = float(
        pd.read_csv(
            RESULTS / "manufacturing_31sector_peak_screen.csv",
            encoding="utf-8-sig",
        ).query(
            "industry_code == @INDUSTRY_CODE and energy_scenario == @ENERGY_SCENARIO and temporal_scenario == 'task_timed'"
        )["baseline_annual_electricity_twh"].iloc[0]
    )
    return (
        base_load_mw,
        task_shapes,
        unscaled_daily_task_service,
        flexibility,
        reference_energy_twh,
        baseline_energy_twh,
    )


def main() -> None:
    (
        base_load_mw,
        task_shapes,
        unscaled_daily_task_service,
        flexibility,
        reference_energy_twh,
        baseline_energy_twh,
    ) = load_inputs()
    params = parameter_core.read_parameters()
    technology = ServerTechnology(
        accelerators_per_group=2.0,
        idle_power_kw=parameter_core.number(params, "L10"),
        maximum_power_kw=parameter_core.number(params, "L09"),
    )
    local_architecture = Architecture(
        name="IF_local_industry_bucket",
        pue=parameter_core.number(params, "L17"),
        target_installed_utilization=0.50,
        reserve_fraction=0.10,
    )
    cloud_architecture = Architecture(
        name="IC_cloud_pool",
        pue=parameter_core.number(params, "U02"),
        target_installed_utilization=parameter_core.number(params, "U03"),
        reserve_fraction=0.10,
    )

    unscaled_by_task: dict[str, np.ndarray] = {}
    for task_id, daily_service in unscaled_daily_task_service.items():
        shape = np.asarray(task_shapes[task_id], dtype=float)
        unscaled_by_task[task_id] = daily_service * shape / float(np.sum(shape))
    unscaled_total = sum(unscaled_by_task.values(), np.zeros(24))
    target_daily_energy_mwh = reference_energy_twh * 1e6 / ANNUAL_DAYS
    kappa, cloud_reference_groups, calibrated_daily_energy_mwh = (
        calibrate_service_scale_to_reference_energy(
            unscaled_total,
            target_daily_energy_mwh,
            technology,
            cloud_architecture,
        )
    )

    scaled_by_task = {
        task_id: values * kappa for task_id, values in unscaled_by_task.items()
    }
    total_arrival = sum(scaled_by_task.values(), np.zeros(24))
    rigid_arrival = np.zeros(24)
    intraday_arrival = np.zeros(24)
    batch_arrival = np.zeros(24)
    flexible_jobs: list[FlexibleArrival] = []
    for task_id, values in scaled_by_task.items():
        setting = flexibility[task_id]
        rigid_arrival += values * setting["R"]
        intraday_values = values * setting["F_day"]
        batch_values = values * setting["F_batch"]
        intraday_arrival += intraday_values
        batch_arrival += batch_values
        for hour in range(24):
            if intraday_values[hour] > 0:
                flexible_jobs.append(
                    FlexibleArrival(
                        release_hour=hour,
                        deadline_hours=int(setting["F_day_deadline"]),
                        amount_accelerator_h=float(intraday_values[hour]),
                        task_id=task_id,
                        flexibility_class="F_day",
                    )
                )
            if batch_values[hour] > 0:
                flexible_jobs.append(
                    FlexibleArrival(
                        release_hour=hour,
                        deadline_hours=int(setting["F_batch_deadline"]),
                        amount_accelerator_h=float(batch_values[hour]),
                        task_id=task_id,
                        flexibility_class="F_batch",
                    )
                )

    base_peak_mw = float(np.max(base_load_mw))
    local_capacity_mw = (
        1.0 + LOCAL_HEADROOM_SHARE_OF_BASE_PEAK
    ) * base_peak_mw
    architectures = (
        (local_architecture, base_load_mw),
        (cloud_architecture, np.zeros(24)),
    )
    summaries: list[dict[str, object]] = []
    hourly_rows: list[dict[str, object]] = []

    for architecture, optimization_base in architectures:
        groups = (
            cloud_reference_groups
            if architecture.name == cloud_architecture.name
            else provision_server_groups(
                float(np.max(total_arrival)), technology, architecture
            )
        )
        optimized = optimize_peak_with_highs(
            rigid_arrival,
            flexible_jobs,
            optimization_base,
            groups,
            technology,
            architecture,
        )
        profiles = {
            "unshifted": total_arrival,
            "optimized": optimized.daily_execution_accelerator_h,
        }
        for dispatch_case, execution in profiles.items():
            facility_power = facility_power_mw(
                execution, groups, technology, architecture.pue
            )
            combined_local = (
                base_load_mw + facility_power
                if architecture.name == local_architecture.name
                else facility_power
            )
            annual_energy_twh = float(np.sum(facility_power)) * ANNUAL_DAYS / 1e6
            facility_peak_mw = float(np.max(facility_power))
            combined_peak_mw = float(np.max(combined_local))
            if architecture.name == local_architecture.name:
                grid_expansion_mw = max(0.0, combined_peak_mw - local_capacity_mw)
            else:
                grid_expansion_mw = facility_peak_mw
            summaries.append(
                {
                    "industry_code": INDUSTRY_CODE,
                    "industry_name_cn": INDUSTRY_NAME,
                    "architecture": architecture.name,
                    "dispatch_case": dispatch_case,
                    "flexibility_scenario": FLEXIBILITY_SCENARIO,
                    "reference_cloud_energy_twh": reference_energy_twh,
                    "service_scale_kappa": kappa,
                    "daily_service_accelerator_h": float(np.sum(total_arrival)),
                    "rigid_service_share": float(np.sum(rigid_arrival) / np.sum(total_arrival)),
                    "intraday_service_share": float(np.sum(intraday_arrival) / np.sum(total_arrival)),
                    "batch_service_share": float(np.sum(batch_arrival) / np.sum(total_arrival)),
                    "installed_server_groups_2xl20": groups,
                    "installed_accelerators": groups * technology.accelerators_per_group,
                    "installed_utilization": float(
                        np.sum(execution)
                        / (groups * technology.accelerators_per_group * 24.0)
                    ),
                    "pue": architecture.pue,
                    "annual_facility_energy_twh": annual_energy_twh,
                    "facility_peak_mw": facility_peak_mw,
                    "baseline_peak_mw": base_peak_mw if architecture.name == local_architecture.name else 0.0,
                    "combined_peak_mw": combined_peak_mw,
                    "existing_capacity_with_headroom_mw": local_capacity_mw if architecture.name == local_architecture.name else 0.0,
                    "grid_expansion_proxy_mw": grid_expansion_mw,
                    "baseline_annual_electricity_twh": baseline_energy_twh,
                    "result_scope": "single_industry_bucket_mechanism_test",
                }
            )
            for hour in range(24):
                hourly_rows.append(
                    {
                        "industry_code": INDUSTRY_CODE,
                        "architecture": architecture.name,
                        "dispatch_case": dispatch_case,
                        "hour": hour,
                        "base_load_mw": base_load_mw[hour] if architecture.name == local_architecture.name else 0.0,
                        "rigid_arrival_accelerator_h": rigid_arrival[hour],
                        "intraday_arrival_accelerator_h": intraday_arrival[hour],
                        "batch_arrival_accelerator_h": batch_arrival[hour],
                        "executed_accelerator_h": execution[hour],
                        "facility_power_mw": facility_power[hour],
                        "combined_load_mw": combined_local[hour],
                    }
                )

    write_csv(SUMMARY_OUTPUT, summaries)
    write_csv(HOURLY_OUTPUT, hourly_rows)
    result = pd.DataFrame(summaries)
    local = result[result["architecture"] == local_architecture.name].set_index("dispatch_case")
    cloud = result[result["architecture"] == cloud_architecture.name].set_index("dispatch_case")
    lines = [
        "# C36汽车制造业服务对齐单行业原型结果",
        "",
        "## 场景",
        "",
        f"- 仅运行C36汽车制造业，不循环其余30个行业。",
        f"- IC参考设施年电量为 {reference_energy_twh:.3f} TWh，只用于反解共同服务规模。",
        "- 本地和云端使用同一中心灵活性矩阵与相同截止期；共同边缘闭环视觉不进入可搬移服务。",
        "- 调度先最小化叠加峰值，再在不恶化该目标的前提下最小化AI设施峰值，避免将可移动任务任意堆到单个低谷小时。",
        f"- 本地PUE={local_architecture.pue:.2f}、目标装机利用率50%；云端PUE={cloud_architecture.pue:.2f}、目标装机利用率65%；两者均含10%容量备用。",
        f"- 本地可靠余量取行业原峰值的 {LOCAL_HEADROOM_SHARE_OF_BASE_PEAK*100:.2f}%；云端按绿地无余量。",
        "- 所有安装服务器组在本轮均视为在线，因此灵活性改变峰值而不改变同一架构的日能耗。",
        "",
        "## 结果",
        "",
        "| 架构 | 调度 | 服务器组 | 年设施电量(TWh) | AI设施峰值(MW) | 叠加峰值(MW) | 扩容代理(MW) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['architecture']} | {row['dispatch_case']} | {int(row['installed_server_groups_2xl20'])} "
            f"| {row['annual_facility_energy_twh']:.3f} | {row['facility_peak_mw']:.1f} "
            f"| {row['combined_peak_mw']:.1f} | {row['grid_expansion_proxy_mw']:.1f} |"
        )
    local_peak_reduction = float(local.loc["unshifted", "combined_peak_mw"] - local.loc["optimized", "combined_peak_mw"])
    cloud_peak_reduction = float(cloud.loc["unshifted", "facility_peak_mw"] - cloud.loc["optimized", "facility_peak_mw"])
    lines.extend(
        [
            "",
            "## 初步解释",
            "",
            f"同一服务需求下，本地需要 {int(local.loc['optimized','installed_server_groups_2xl20'])} 组双L20服务器，云端需要 {int(cloud.loc['optimized','installed_server_groups_2xl20'])} 组。由于本地PUE更高且装机利用率目标更低，本地设施年电量为 {local.loc['optimized','annual_facility_energy_twh']:.3f} TWh，高于云端的 {cloud.loc['optimized','annual_facility_energy_twh']:.3f} TWh。",
            "",
            f"使用完全相同的任务柔性比例和截止期后，主动调度使本地叠加峰值降低 {local_peak_reduction:.1f} MW，使云端AI设施峰值降低 {cloud_peak_reduction:.1f} MW。两边均不是直接假设完全平坦。",
            "",
            f"在本地可靠余量等于行业原峰值0.25%的条件下，本地优化后扩容代理为 {local.loc['optimized','grid_expansion_proxy_mw']:.1f} MW；绿地云端为 {cloud.loc['optimized','grid_expansion_proxy_mw']:.1f} MW。该比较只是行业负荷桶机制测试，不代表汽车工厂实际同址余量或具体电网工程。",
            "",
            "## 限制",
            "",
            "- C36被视为一个可共享余量的行业负荷桶，不代表18899家企业或真实工厂节点。",
            "- 服务器空闲/满载功率、PUE、目标利用率和备用比例仍是筛查参数。",
            "- 代表性工作日按365天年化，尚未加入周末、季节和跨日随机性。",
            "- 本轮只验证服务对齐与灵活性机理，不计算企业账单、社会总成本或监管资本偏好。",
        ]
    )
    FINDINGS_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "industry": INDUSTRY_CODE,
                "summary": str(SUMMARY_OUTPUT),
                "hourly": str(HOURLY_OUTPUT),
                "findings": str(FINDINGS_OUTPUT),
                "rows": len(summaries),
                "kappa": kappa,
                "calibrated_reference_daily_mwh": calibrated_daily_energy_mwh,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
