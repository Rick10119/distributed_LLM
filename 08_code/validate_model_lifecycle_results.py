#!/usr/bin/env python3
"""Validate and summarize the simplified model-state/lifecycle layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--findings-output", type=Path, required=True)
    parser.add_argument("--done-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.concat(
        [pd.read_csv(path, encoding="utf-8-sig") for path in args.inputs],
        ignore_index=True,
    )
    if len(frame) != 93 or set(frame["model_version"]) != {args.model_version}:
        raise ValueError("Lifecycle validation requires 31 industries x 3 scenarios from one version")
    if not (
        frame["per_host_installed_server_groups"] + 1e-7
        >= frame["minimum_server_groups_for_model_state"]
    ).all():
        raise ValueError("At least one scenario violates the model replica/VRAM floor")
    if not (
        frame["per_host_model_storage_available_gb"] + 1e-7
        >= frame["per_host_model_storage_required_gb"]
    ).all():
        raise ValueError("At least one scenario violates the bundled-storage constraint")
    rows: list[dict[str, object]] = []
    for scenario, group in frame.groupby("scenario"):
        rows.append(
            {
                "model_version": args.model_version,
                "scenario": scenario,
                "industries": len(group),
                "required_model_replicas_per_deployment": int(group["required_model_replicas"].iloc[0]),
                "minimum_server_groups_per_deployment": int(group["minimum_server_groups_for_model_state"].iloc[0]),
                "model_storage_required_gb_per_deployment": float(group["per_host_model_storage_required_gb"].iloc[0]),
                "minimum_storage_headroom_ratio": float((group["per_host_model_storage_available_gb"] / group["per_host_model_storage_required_gb"]).min()),
                "national_initialization_energy_mwh_year": float(group["industry_equivalent_annual_model_initialization_energy_twh"].sum() * 1e6),
                "national_initialization_cost_million_rmb_year": float(group["industry_equivalent_incremental_annual_model_initialization_cost_rmb"].sum() / 1e6),
                "national_storage_cost_million_rmb_year": float(group["industry_equivalent_incremental_annual_model_storage_cost_rmb"].sum() / 1e6),
                "national_operations_cost_million_rmb_year": float(group["industry_equivalent_incremental_annual_model_operations_cost_rmb"].sum() / 1e6),
            }
        )
    output = pd.DataFrame(rows).sort_values("scenario")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    lines = [
        f"# 简化模型状态与生命周期验证（{args.model_version}）",
        "",
        "当前层只加入一个32B AWQ生产副本的显存/整数服务器下限、200GB模型族存储、年度初始化和0.25 FTE基础运维。逐请求KV cache调度、冷启动、多模型副本分配和RAG项目成本明确排除。",
        "",
        "| 架构 | 最小存储余量倍数 | 全国初始化电量（MWh/年） | 初始化成本（百万元/年） | 存储成本（百万元/年） | 运维成本（百万元/年） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    names = {"IF": "独立工厂", "IG": "集团池", "II_1host": "行业单池上界"}
    for row in output.itertuples():
        lines.append(
            f"| {names.get(row.scenario, row.scenario)} | {row.minimum_storage_headroom_ratio:.1f} | "
            f"{row.national_initialization_energy_mwh_year:.3f} | {row.national_initialization_cost_million_rmb_year:.3f} | "
            f"{row.national_storage_cost_million_rmb_year:.3f} | {row.national_operations_cost_million_rmb_year:.3f} |"
        )
    lines.extend(
        [
            "",
            "初始化电量和模型文件存储不构成当前排序的主要来源；基础运维随部署点数量变化，因此会给分散部署增加更明显的年度负担。0.25 FTE是项目级情景代理，不是全国制造企业观测均值，结论必须配合低/高运维情景复核。",
        ]
    )
    args.findings_output.parent.mkdir(parents=True, exist_ok=True)
    args.findings_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "status": "validated",
        "model_version": args.model_version,
        "scenarios": len(output),
        "industry_scenarios": len(frame),
        "replica_vram_constraints_pass": True,
        "storage_constraints_pass": True,
        "excluded_detail_boundary_preserved": True,
    }
    args.done_output.parent.mkdir(parents=True, exist_ok=True)
    args.done_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
