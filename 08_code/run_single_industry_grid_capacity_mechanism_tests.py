#!/usr/bin/env python3
"""Run a bounded single-industry IG grid-capacity mechanism experiment.

This script is intentionally independent of the 31-industry workflow.  It
implements the Phase 0/1 storage controls, direct reserve-semantics tests, and
the limited integer-variable check specified in
the registered test plan.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_code"))

from core.config import load_config, validate_config
from core.data import FlexibleJob, load_industry_inputs, read_core_grid_energy_prices, scale_workload
from core.io import write_csv
from core.model import HostResult, optimize_host
from core.representative_group import read_representative_groups, scenario_scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults", type=Path, default=Path("config/defaults.yaml"))
    parser.add_argument("--config", type=Path, default=Path("config/runs/single_industry_core.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("05_results/v0.7.0/result/mechanism_tests/C36_IG_external_reserve"),
    )
    parser.add_argument(
        "--include-integer",
        action="store_true",
        help="Also run the limited installed-capacity integer check.",
    )
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="Run the 2-hour storage scan and bounded deadline diagnostics.",
    )
    return parser.parse_args()


def configured(
    base: dict,
    *,
    battery: bool,
    planning_reserve: float = 0.10,
    dispatch_reserve: float = 0.10,
    installed_integer: bool = False,
    online_integer: bool = False,
) -> dict:
    config = deepcopy(base)
    config["energy"]["battery_investment_enabled"] = battery
    config["energy"].pop("battery_fixed_power_mw", None)
    config["server"]["installed_reserve_fraction"] = planning_reserve
    config["server"]["normal_dispatch_reserve_fraction"] = dispatch_reserve
    config["model"]["server_groups_integer"] = installed_integer and online_integer
    config["model"]["installed_server_groups_integer"] = installed_integer
    config["model"]["online_server_groups_integer"] = online_integer
    validate_config(config)
    return config


def solve_no_ai(config: dict, inputs, prices: np.ndarray) -> HostResult:
    return optimize_host(
        config,
        base_load_mw=inputs.base_load_mw,
        pv_capacity_factor=inputs.pv_capacity_factor,
        roof_area_m2=inputs.roof_area_proxy_m2,
        grid_energy_price_rmb_per_mwh=prices,
    )


def solve_ai(
    config: dict,
    inputs,
    prices: np.ndarray,
    rigid: np.ndarray,
    jobs,
    baseline: HostResult,
    *,
    minimum_installed: float | None = None,
) -> HostResult:
    return optimize_host(
        config,
        base_load_mw=inputs.base_load_mw,
        pv_capacity_factor=inputs.pv_capacity_factor,
        roof_area_m2=inputs.roof_area_proxy_m2,
        rigid_service_units=rigid,
        flexible_jobs=jobs,
        grid_energy_price_rmb_per_mwh=prices,
        existing_grid_capacity_mw=float(baseline.summary["grid_import_peak_mw"]),
        minimum_installed_server_groups=minimum_installed,
    )


def coefficient_of_variation(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    mean = float(np.mean(array))
    return 0.0 if mean == 0.0 else float(np.std(array) / mean)


def summarize(
    case_id: str,
    role: str,
    result: HostResult,
    baseline: HostResult,
    *,
    planning_reserve: float,
    dispatch_reserve: float,
    installed_integer: bool,
    online_integer: bool,
    two_stage: bool = False,
) -> dict[str, object]:
    summary = result.summary
    hourly = result.hourly
    baseline_summary = baseline.summary
    grid_increment = max(
        0.0,
        float(summary["grid_import_peak_mw"])
        - float(baseline_summary["grid_import_peak_mw"]),
    )
    ai_peak = float(summary["ai_facility_peak_mw"])
    row = {
        "case_id": case_id,
        "role": role,
        "planning_reserve_fraction": planning_reserve,
        "normal_dispatch_reserve_fraction": dispatch_reserve,
        "installed_server_groups_integer": installed_integer,
        "online_server_groups_integer": online_integer,
        "external_planning_capacity": two_stage,
        "battery_investment_enabled": bool(summary["battery_investment_enabled"]),
        "no_ai_net_grid_peak_mw": float(baseline_summary["grid_import_peak_mw"]),
        "grid_import_peak_mw": float(summary["grid_import_peak_mw"]),
        "incremental_grid_capacity_mw": grid_increment,
        "ai_facility_peak_mw": ai_peak,
        "capacity_avoidance_fraction": np.nan if ai_peak == 0.0 else 1.0 - grid_increment / ai_peak,
        "battery_power_mw": float(summary["battery_power_mw"]),
        "battery_energy_mwh": float(summary["battery_energy_mwh"]),
        "incremental_battery_power_mw": float(summary["battery_power_mw"])
        - float(baseline_summary["battery_power_mw"]),
        "installed_server_groups": float(summary["installed_server_groups"]),
        "online_server_groups_min": float(hourly["online_server_groups"].min()),
        "online_server_groups_max": float(hourly["online_server_groups"].max()),
        "online_server_groups_cv": coefficient_of_variation(hourly["online_server_groups"]),
        "ai_compute_cv": coefficient_of_variation(hourly["ai_compute_accelerator_h"]),
        "ai_facility_power_cv": coefficient_of_variation(hourly["ai_facility_power_mw"]),
        "annual_objective_rmb": float(summary["annual_objective_rmb"]),
        "incremental_annual_objective_rmb": float(summary["annual_objective_rmb"])
        - float(baseline_summary["annual_objective_rmb"]),
        "annual_server_cost_rmb": float(summary["annual_server_cost_rmb"]),
        "annual_battery_cost_rmb": float(summary["annual_battery_cost_rmb"]),
        "annual_maximum_demand_cost_rmb": float(summary["annual_maximum_demand_cost_rmb"]),
        "annual_flat_energy_cost_rmb": float(summary["annual_flat_energy_cost_rmb"]),
    }
    for component in (
        "annual_server_cost_rmb",
        "annual_battery_cost_rmb",
        "annual_maximum_demand_cost_rmb",
        "annual_flat_energy_cost_rmb",
    ):
        row[f"incremental_{component}"] = float(summary[component]) - float(
            baseline_summary[component]
        )
    return row


def main() -> None:
    args = parse_args()
    base = load_config(ROOT, args.defaults, args.config)
    inputs_full = load_industry_inputs(base, "C36")
    group = read_representative_groups(
        ROOT / base["paths"]["representative_group_report"]
    )["C36"]
    scale = scenario_scale(group, base["industry_parameter_case"], "IG")
    rigid, jobs = scale_workload(inputs_full, scale.ai_service_scale_per_host)
    inputs = replace(
        inputs_full,
        base_load_mw=inputs_full.base_load_mw * scale.base_load_scale_per_host,
    )
    prices = read_core_grid_energy_prices(base)

    no_storage_config = configured(base, battery=False)
    storage_config = configured(base, battery=True)
    baseline_no_storage = solve_no_ai(no_storage_config, inputs, prices)
    baseline_storage = solve_no_ai(storage_config, inputs, prices)

    cases: list[tuple[str, str, HostResult, HostResult, dict[str, object]]] = []
    cases.append((
        "A_no_AI_no_storage", "storage_2x2", baseline_no_storage, baseline_no_storage,
        dict(planning_reserve=0.10, dispatch_reserve=0.10, installed_integer=False, online_integer=False),
    ))
    cases.append((
        "B_no_AI_storage", "storage_2x2", baseline_storage, baseline_storage,
        dict(planning_reserve=0.10, dispatch_reserve=0.10, installed_integer=False, online_integer=False),
    ))
    ai_no_storage = solve_ai(
        no_storage_config, inputs, prices, rigid, jobs, baseline_no_storage
    )
    cases.append((
        "C_AI_no_storage_R10_10", "storage_2x2", ai_no_storage, baseline_no_storage,
        dict(planning_reserve=0.10, dispatch_reserve=0.10, installed_integer=False, online_integer=False),
    ))
    ai_storage = solve_ai(storage_config, inputs, prices, rigid, jobs, baseline_storage)
    cases.append((
        "D_AI_storage_R10_10", "storage_2x2", ai_storage, baseline_storage,
        dict(planning_reserve=0.10, dispatch_reserve=0.10, installed_integer=False, online_integer=False),
    ))

    # Stage 1 establishes the external no-reserve capacity requirement under
    # the same tasks, prices, PV and endogenous-storage rules.
    no_reserve_config = configured(
        base, battery=True, planning_reserve=0.00, dispatch_reserve=0.00
    )
    no_reserve = solve_ai(
        no_reserve_config, inputs, prices, rigid, jobs, baseline_storage
    )
    required_groups = float(no_reserve.summary["installed_server_groups"])
    cases.append((
        "R00_capacity_reference", "external_capacity_reference", no_reserve, baseline_storage,
        dict(planning_reserve=0.00, dispatch_reserve=0.00, installed_integer=False, online_integer=False),
    ))

    no_reserve_no_storage_config = configured(
        base, battery=False, planning_reserve=0.00, dispatch_reserve=0.00
    )
    no_reserve_no_storage = solve_ai(
        no_reserve_no_storage_config,
        inputs,
        prices,
        rigid,
        jobs,
        baseline_no_storage,
    )
    required_groups_no_storage = float(
        no_reserve_no_storage.summary["installed_server_groups"]
    )
    p10_no_storage = solve_ai(
        no_reserve_no_storage_config,
        inputs,
        prices,
        rigid,
        jobs,
        baseline_no_storage,
        minimum_installed=required_groups_no_storage * 1.10,
    )
    cases.append((
        "P10_U100_no_storage", "external_planning_capacity_no_storage", p10_no_storage, baseline_no_storage,
        dict(planning_reserve=0.10, dispatch_reserve=0.00, installed_integer=False, online_integer=False, two_stage=True),
    ))

    # Stage 2 embeds planning reserve in an exogenous minimum installed stock.
    # Hourly operation may use every installed group; investment may exceed the
    # minimum if that is economically useful.
    external_cases: dict[str, HostResult] = {}
    for case_id, plan_reserve in (("P10_U100", 0.10), ("P20_U100", 0.20)):
        config = configured(
            base, battery=True, planning_reserve=0.00, dispatch_reserve=0.00
        )
        result = solve_ai(
            config,
            inputs,
            prices,
            rigid,
            jobs,
            baseline_storage,
            minimum_installed=required_groups * (1.0 + plan_reserve),
        )
        external_cases[case_id] = result
        cases.append((
            case_id, "external_planning_capacity", result, baseline_storage,
            dict(planning_reserve=plan_reserve, dispatch_reserve=0.00, installed_integer=False, online_integer=False, two_stage=True),
        ))

    # Keep the old one-stage 10% formulation only as a labelled comparator.
    cases.append((
        "LEGACY_R10_10", "legacy_hourly_reserve_comparator", ai_storage, baseline_storage,
        dict(planning_reserve=0.10, dispatch_reserve=0.10, installed_integer=False, online_integer=False),
    ))

    if args.include_integer:
        for case_id, installed_integer, online_integer in (
            ("P10_U100_integer_installed", True, False),
        ):
            config = configured(
                base,
                battery=True,
                planning_reserve=0.00,
                dispatch_reserve=0.00,
                installed_integer=installed_integer,
                online_integer=online_integer,
            )
            result = solve_ai(
                config,
                inputs,
                prices,
                rigid,
                jobs,
                baseline_storage,
                minimum_installed=required_groups * 1.10,
            )
            cases.append((
                case_id, "integer_check", result, baseline_storage,
                dict(planning_reserve=0.10, dispatch_reserve=0.00, installed_integer=installed_integer, online_integer=online_integer, two_stage=True),
            ))

    storage_scan_rows: list[dict[str, object]] = []
    flexibility_rows: list[dict[str, object]] = []
    if args.include_diagnostics:
        baseline_storage_power = float(baseline_storage.summary["battery_power_mw"])
        for added_power in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0):
            config = configured(
                base, battery=False, planning_reserve=0.00, dispatch_reserve=0.00
            )
            config["energy"]["battery_fixed_power_mw"] = baseline_storage_power + added_power
            validate_config(config)
            result = solve_ai(
                config,
                inputs,
                prices,
                rigid,
                jobs,
                baseline_storage,
                minimum_installed=required_groups * 1.10,
            )
            storage_scan_rows.append(
                {
                    "added_battery_power_mw": added_power,
                    "total_battery_power_mw": float(result.summary["battery_power_mw"]),
                    "battery_duration_h": float(config["energy"]["battery_duration_h"]),
                    "incremental_grid_capacity_mw": max(
                        0.0,
                        float(result.summary["grid_import_peak_mw"])
                        - float(baseline_storage.summary["grid_import_peak_mw"]),
                    ),
                    "incremental_annual_objective_rmb": float(result.summary["annual_objective_rmb"])
                    - float(baseline_storage.summary["annual_objective_rmb"]),
                    "ai_facility_power_cv": coefficient_of_variation(
                        result.hourly["ai_facility_power_mw"]
                    ),
                }
            )

        deadline_jobs = tuple(
            FlexibleJob(
                release_hour=job.release_hour,
                deadline_hours=min(job.deadline_hours, 24),
                amount_service_units=job.amount_service_units,
                task_id=job.task_id,
                flexibility_class=job.flexibility_class,
            )
            for job in jobs
        )
        rigid_at_arrival = rigid.copy()
        for job in jobs:
            rigid_at_arrival[job.release_hour] += job.amount_service_units
        for case_id, rigid_case, jobs_case in (
            ("deadline_max_24h", rigid, deadline_jobs),
            ("arrival_execution", rigid_at_arrival, ()),
        ):
            config = configured(
                base, battery=True, planning_reserve=0.00, dispatch_reserve=0.00
            )
            result = solve_ai(
                config,
                inputs,
                prices,
                rigid_case,
                jobs_case,
                baseline_storage,
                minimum_installed=required_groups * 1.10,
            )
            flexibility_rows.append(
                {
                    "case_id": case_id,
                    "maximum_deadline_h": max(
                        (job.deadline_hours for job in jobs_case), default=0
                    ),
                    "incremental_grid_capacity_mw": max(
                        0.0,
                        float(result.summary["grid_import_peak_mw"])
                        - float(baseline_storage.summary["grid_import_peak_mw"]),
                    ),
                    "battery_power_mw": float(result.summary["battery_power_mw"]),
                    "installed_server_groups": float(result.summary["installed_server_groups"]),
                    "ai_compute_cv": coefficient_of_variation(
                        result.hourly["ai_compute_accelerator_h"]
                    ),
                    "ai_facility_power_cv": coefficient_of_variation(
                        result.hourly["ai_facility_power_mw"]
                    ),
                    "incremental_annual_objective_rmb": float(result.summary["annual_objective_rmb"])
                    - float(baseline_storage.summary["annual_objective_rmb"]),
                }
            )

    rows: list[dict[str, object]] = []
    hourly_frames: list[pd.DataFrame] = []
    for case_id, role, result, baseline, settings in cases:
        rows.append(summarize(case_id, role, result, baseline, **settings))
        frame = result.hourly.copy()
        frame.insert(0, "case_id", case_id)
        hourly_frames.append(frame)
    summary = pd.DataFrame(rows)
    output_dir = ROOT / args.output_dir
    write_csv(summary, output_dir / "cases.csv")
    write_csv(pd.concat(hourly_frames, ignore_index=True), output_dir / "hourly.csv")
    if storage_scan_rows:
        write_csv(pd.DataFrame(storage_scan_rows), output_dir / "storage_scan.csv")
    if flexibility_rows:
        write_csv(pd.DataFrame(flexibility_rows), output_dir / "flexibility_diagnostics.csv")

    no_storage_increment = float(
        summary.loc[summary.case_id == "C_AI_no_storage_R10_10", "incremental_grid_capacity_mw"].iloc[0]
    )
    storage_increment = float(
        summary.loc[summary.case_id == "D_AI_storage_R10_10", "incremental_grid_capacity_mw"].iloc[0]
    )
    storage_avoidance = no_storage_increment - storage_increment
    baseline_storage_power = float(baseline_storage.summary["battery_power_mw"])
    ai_storage_power = float(ai_storage.summary["battery_power_mw"])
    incremental_storage_power = ai_storage_power - baseline_storage_power
    storage_cost_saving = float(ai_no_storage.summary["annual_objective_rmb"] - baseline_no_storage.summary["annual_objective_rmb"]) - float(ai_storage.summary["annual_objective_rmb"] - baseline_storage.summary["annual_objective_rmb"])
    integer_rows = summary[summary["role"] == "integer_check"]
    integer_note = "未运行。"
    if not integer_rows.empty:
        integer_row = integer_rows.iloc[0]
        continuous_row = summary[summary["case_id"] == "P10_U100"].iloc[0]
        integer_note = (
            f"装机从 {continuous_row.installed_server_groups:.6f} 组变为 "
            f"{integer_row.installed_server_groups:.0f} 组，接入容量仍为 "
            f"{integer_row.incremental_grid_capacity_mw:.6f} MW，总年成本变化 "
            f"{integer_row.annual_objective_rmb - continuous_row.annual_objective_rmb:,.0f} 元。"
        )
    diagnostic_note = "未运行。"
    if storage_scan_rows and flexibility_rows:
        storage_scan = pd.DataFrame(storage_scan_rows)
        first_zero = storage_scan[
            storage_scan["incremental_grid_capacity_mw"] <= 1e-6
        ]
        zero_note = "扫描范围内未归零"
        if not first_zero.empty:
            first_zero_index = int(first_zero.index[0])
            lower = (
                0.0
                if first_zero_index == 0
                else float(storage_scan.iloc[first_zero_index - 1].added_battery_power_mw)
            )
            upper = float(first_zero.iloc[0].added_battery_power_mw)
            zero_note = f"在新增 {lower:.2f}–{upper:.2f} MW 之间归零"
        flex = pd.DataFrame(flexibility_rows).set_index("case_id")
        diagnostic_note = (
            f"两小时储能离散扫描显示接入扩容量{zero_note}。"
            f"截止期缩至24小时的 AI 功率变异系数为 "
            f"{flex.loc['deadline_max_24h', 'ai_facility_power_cv']:.4f}；"
            f"到达即执行时为 {flex.loc['arrival_execution', 'ai_facility_power_cv']:.4f}。"
        )
    p10 = summary[summary["case_id"] == "P10_U100"].iloc[0]
    legacy = summary[summary["case_id"] == "LEGACY_R10_10"].iloc[0]
    p10_no_storage_row = summary[
        summary["case_id"] == "P10_U100_no_storage"
    ].iloc[0]
    external_note = (
        f"正确约束下，110%最低装机为 {required_groups * 1.10:.6f} 组，"
        f"实际装机 {p10.installed_server_groups:.6f} 组；在线服务器从 "
        f"{p10.online_server_groups_min:.3f} 到 {p10.online_server_groups_max:.3f} 组，"
        f"AI设施功率变异系数为 {p10.ai_facility_power_cv:.4f}。"
        f"相对旧约束，年成本降低 "
        f"{legacy.annual_objective_rmb - p10.annual_objective_rmb:,.0f} 元。"
        f"在双方都禁止储能的控制下，旧约束需要 "
        f"{no_storage_increment:.3f} MW 扩容，正确约束需要 "
        f"{p10_no_storage_row.incremental_grid_capacity_mw:.3f} MW。"
    )
    findings = f"""# C36 集团部署电网容量机制测试：第一轮

本轮仅运行 C36 汽车制造业集团共享部署，不更新31行业主结果。

## 储能控制实验

- 禁止储能时的 AI 接入容量增量：{no_storage_increment:.3f} MW；
- 无 AI 与有 AI 均允许储能时的容量增量：{storage_increment:.3f} MW；
- 在匹配基准下，储能缓解的 AI 接入容量：{storage_avoidance:.3f} MW。
- 无 AI 基准储能为 {baseline_storage_power:.3f} MW，有 AI 时为 {ai_storage_power:.3f} MW，AI 诱发新增储能为 {incremental_storage_power:.3f} MW；
- 允许储能相对于匹配的禁止储能对照，使 AI 增量年成本降低 {storage_cost_saving:,.0f} 元。

因此，当前联合优化并非“集团部署不增加储能”：它增加约 {incremental_storage_power:.2f} MW 的两小时储能，并把约 {no_storage_increment:.2f} MW 的接入扩容量降至零。此前图中的正扩容量是尚未按新边界重跑的过渡性固定调度估计。

## 外生规划容量与实际运行

`R00_capacity_reference` 先在无备用条件下确定所需服务器容量。`P10_U100` 再把最低装机设为该容量的110%，但逐时计算只受在线量不超过总装机约束，可以使用全部服务器；投资变量仍可选择超过最低值。`LEGACY_R10_10` 保留旧的逐时90.91%使用上限，仅作为比较，不再解释为独立规划裕量。

{external_note}

## 整数敏感性

{integer_note}

## 储能边际与任务柔性诊断

{diagnostic_note}

## 解释边界

本轮采用外生容量基准的两阶段求解，但第二阶段不固定装机：只施加最低装机下界，仍允许追加投资。上一轮名为 `R10_0` 的测试错误地保留了逐小时规划约束，已废止，不能用于判断备用容量价值。整数测试仅在命令行显式启用时运行。

这些结果是单行业机制识别，不代表全国效应。完整数值和逐时曲线见 `cases.csv` 与 `hourly.csv`。
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "findings.md").write_text(findings, encoding="utf-8")


if __name__ == "__main__":
    main()
