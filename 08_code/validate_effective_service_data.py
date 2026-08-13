#!/usr/bin/env python3
"""Validate processed service with the same hourly logic used by the core model."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd

from core.config import load_config
from core.data import load_industry_inputs


CASES = ("low", "base", "high")
TARGET_ATTRIBUTE = {
    "low": "external_energy_low_twh",
    "base": "external_energy_central_twh",
    "high": "external_energy_high_twh",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-input", type=Path, required=True)
    parser.add_argument("--defaults", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--findings-output", type=Path, required=True)
    parser.add_argument("--done-output", type=Path, required=True)
    args = parser.parse_args()

    root = Path.cwd()
    config = load_config(root, args.defaults, args.run_config)
    configured_service = root / config["paths"]["model_ready_task_service"]
    if configured_service.resolve() != args.service_input.resolve():
        raise ValueError("validation input must be the configured model-ready service table")
    frame = pd.read_csv(args.service_input, encoding="utf-8-sig")
    if len(frame) != 31 * 6 * 3:
        raise ValueError("model-ready service must contain 31 industries x 6 tasks x 3 cases")

    rows: list[dict[str, object]] = []
    for case in CASES:
        case_config = deepcopy(config)
        case_config["demand"]["effective_service"]["parameter_case"] = case
        for industry_code in config["selected_industries"]:
            inputs = load_industry_inputs(
                case_config,
                industry_code,
                enforce_external_alignment=False,
            )
            alignment = case_config["demand"]["external_energy_alignment"]
            target_energy = float(getattr(inputs, TARGET_ATTRIBUTE[case]))
            target_ratio = inputs.derived_reference_energy_twh / target_energy
            inside = (
                float(alignment["warning_ratio_low"])
                <= target_ratio
                <= float(alignment["warning_ratio_high"])
            )
            rows.append(
                {
                    "parameter_case": case,
                    "industry_code": industry_code,
                    "industry_name_cn": inputs.industry_name,
                    "effective_service_units_day": inputs.daily_effective_service_units,
                    "derived_reference_energy_twh": inputs.derived_reference_energy_twh,
                    "external_energy_low_twh": inputs.external_energy_low_twh,
                    "external_energy_central_twh": inputs.external_energy_central_twh,
                    "external_energy_high_twh": inputs.external_energy_high_twh,
                    "scenario_target_energy_twh": target_energy,
                    "scenario_target_alignment_ratio": target_ratio,
                    "central_alignment_ratio": inputs.external_energy_alignment_ratio,
                    "inside_industry_warning_band": inside,
                    "reference_energy_server_groups": inputs.reference_energy_server_groups,
                }
            )

    validation = pd.DataFrame.from_records(rows).sort_values(
        ["parameter_case", "industry_code"]
    )
    national = (
        validation.groupby("parameter_case", as_index=False)
        .agg(
            effective_service_units_day=("effective_service_units_day", "sum"),
            derived_reference_energy_twh=("derived_reference_energy_twh", "sum"),
            external_energy_low_twh=("external_energy_low_twh", "sum"),
            external_energy_central_twh=("external_energy_central_twh", "sum"),
            external_energy_high_twh=("external_energy_high_twh", "sum"),
            industries_inside_warning_band=("inside_industry_warning_band", "sum"),
        )
        .sort_values("parameter_case")
    )
    selected_case = config["demand"]["effective_service"]["parameter_case"]
    selected = national[national["parameter_case"] == selected_case]
    if len(selected) != 1:
        raise ValueError(f"missing selected parameter case: {selected_case}")
    selected_row = selected.iloc[0]
    energy = float(selected_row["derived_reference_energy_twh"])
    lower = float(selected_row["external_energy_low_twh"])
    central = float(selected_row["external_energy_central_twh"])
    upper = float(selected_row["external_energy_high_twh"])
    industries_valid = int(selected_row["industries_inside_warning_band"])
    industry_check_mode = config["demand"]["external_energy_alignment"].get(
        "per_industry_check", "hard"
    )
    if not lower <= energy <= upper:
        raise ValueError(f"selected service case implies {energy:.3f} TWh outside {lower}-{upper}")
    if industry_check_mode == "hard" and industries_valid != 31:
        invalid = validation[
            (validation["parameter_case"] == selected_case)
            & (~validation["inside_industry_warning_band"])
        ]["industry_code"].tolist()
        raise ValueError(f"industry alignment check failed for: {invalid}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.findings_output.parent.mkdir(parents=True, exist_ok=True)
    args.done_output.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(args.output, index=False, encoding="utf-8-sig")
    case_lines = [
        f"- {row.parameter_case}: {row.derived_reference_energy_twh:.3f} TWh/年；"
        f"行业校验带内 {int(row.industries_inside_warning_band)}/31。"
        for row in national.itertuples(index=False)
    ]
    args.findings_output.write_text(
        "\n".join(
            [
                "# 分任务有效服务量重算与外部一致性检查",
                "",
                "本检查直接调用核心模型的任务小时形状、服务器峰值容量和空闲功率逻辑。",
                "服务量由2023任务基线和分任务联合增长情景生成；外部用电区间仅用于软校验。",
                "模型不再使用统一服务强度乘数。",
                "",
                *case_lines,
                "",
                f"选择情景：{selected_case}。参考用电 {energy:.3f} TWh/年，",
                f"相对{central:.0f} TWh中心值偏差 {(energy / central - 1.0):+.2%}，",
                f"位于{lower:.0f}-{upper:.0f} TWh外部区间内；逐行业±25%检查为{industry_check_mode}模式，带内{industries_valid}/31。",
                "",
                "解释边界：Base是对参数相关性作收缩后的联合情景，不是文献点预测；",
                "任务模板回退、事件频率和单事件计算量仍需企业日志替换。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args.done_output.write_text(
        json.dumps(
            {
                "status": "passed",
                "validation_method": "core_hourly_shapes_and_server_capacity_logic",
                "selected_parameter_case": selected_case,
                "derived_reference_energy_twh": energy,
                "external_range_twh": [lower, central, upper],
                "industries_inside_warning_band": industries_valid,
                "per_industry_check": industry_check_mode,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
