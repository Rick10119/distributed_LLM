#!/usr/bin/env python3
"""Validate coarse compute-efficiency cases against the national electricity envelope."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_code"))

from core.config import load_config
from core.data import load_industry_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--findings-output", type=Path, required=True)
    parser.add_argument("--done-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(ROOT, args.defaults, args.config)
    cases = pd.read_csv(args.table, encoding="utf-8-sig")
    rows: list[dict[str, object]] = []
    for case in cases.itertuples():
        scenario_config = deepcopy(config)
        service = scenario_config["demand"]["effective_service"]
        service["compute_efficiency_case"] = case.efficiency_case
        service["accelerator_h_per_service_unit"] = float(case.accelerator_h_per_service_unit)
        industries = [
            load_industry_inputs(scenario_config, code, enforce_external_alignment=False)
            for code in scenario_config["selected_industries"]
        ]
        derived = sum(item.derived_reference_energy_twh for item in industries)
        low = sum(item.external_energy_low_twh for item in industries)
        central = sum(item.external_energy_central_twh for item in industries)
        high = sum(item.external_energy_high_twh for item in industries)
        rows.append(
            {
                "model_version": config["model_version"],
                "efficiency_case": case.efficiency_case,
                "accelerator_h_per_service_unit": case.accelerator_h_per_service_unit,
                "relative_compute_to_base": case.relative_compute_to_base,
                "derived_reference_energy_twh": derived,
                "external_low_twh": low,
                "external_central_twh": central,
                "external_high_twh": high,
                "inside_external_envelope": low <= derived <= high,
                "evidence_status": case.evidence_status,
            }
        )
    output = pd.DataFrame(rows)
    if not output["inside_external_envelope"].all():
        raise ValueError("At least one compute-efficiency case is outside the external energy envelope")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    rendered = output.set_index("efficiency_case")
    findings = f"""# 算力效率档位验证（{config['model_version']}）

主模型使用六类服务器侧因素的加权几何聚合，只保留高效、基准和保守三档。基准档继续使用 `1/3 accelerator-hour/service-unit` 作为归一化参照；活动运行选择 `{config['demand']['effective_service']['compute_efficiency_case']}` 档，不反向改写有效服务量。

| 档位 | accelerator-hour/service-unit | 相对基准计算量 | 全国参考电量（TWh/年） | 外部8—28 TWh包络 |
|---|---:|---:|---:|---|
| 高效 | {rendered.loc['efficient', 'accelerator_h_per_service_unit']:.3f} | {rendered.loc['efficient', 'relative_compute_to_base']:.3f} | {rendered.loc['efficient', 'derived_reference_energy_twh']:.3f} | 通过 |
| 基准 | {rendered.loc['base', 'accelerator_h_per_service_unit']:.3f} | 1.000 | {rendered.loc['base', 'derived_reference_energy_twh']:.3f} | 通过 |
| 保守 | {rendered.loc['conservative', 'accelerator_h_per_service_unit']:.3f} | {rendered.loc['conservative', 'relative_compute_to_base']:.3f} | {rendered.loc['conservative', 'derived_reference_energy_twh']:.3f} | 通过 |

这些档位是证据约束的敏感性包络，不是L20目标模型实测值，也不是2030硬件效率预测。TokenPowerBench的百分比只用于限定服务器侧方向和范围；ServeGen用于约束请求结构；NLR和Delavande等用于约束利用率、到达整形和低批量影响。六个因素存在相关性，因此用加权几何平均形成保守聚合，避免把各论文极端改善幅度简单连乘。活动运行另采用1.50 kW双L20整机满载功率高值；算力需求和整机功率是两个不同维度，但该联合参数组仍应解释为偏高用电边界，而非中心点实测。
"""
    args.findings_output.parent.mkdir(parents=True, exist_ok=True)
    args.findings_output.write_text(findings, encoding="utf-8")
    payload = {
        "status": "validated",
        "model_version": config["model_version"],
        "cases": len(output),
        "all_cases_inside_external_envelope": True,
        "base_preserves_calibrated_conversion": True,
        "active_compute_efficiency_case": config["demand"]["effective_service"]["compute_efficiency_case"],
        "target_hardware_benchmark_still_required": True,
    }
    args.done_output.parent.mkdir(parents=True, exist_ok=True)
    args.done_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
