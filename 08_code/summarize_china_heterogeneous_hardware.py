#!/usr/bin/env python3
"""Aggregate 31 separately-sized China manufacturing heterogeneous cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--routing-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--findings-output", type=Path, required=True)
    parser.add_argument("--done-output", type=Path, required=True)
    args = parser.parse_args()
    detail = pd.concat([pd.read_csv(path, encoding="utf-8-sig") for path in args.inputs], ignore_index=True)
    if detail.industry_code.nunique() != 31:
        raise ValueError("China heterogeneous summary requires 31 industries")
    routing_config = yaml.safe_load(args.routing_config.read_text(encoding="utf-8"))
    active_routing_case = routing_config["active_core_routing_case"]
    core = detail[detail.routing_case == active_routing_case].copy()
    expected_architectures = {"IF", "IG", "II_1host"}
    if set(core["owned_architecture"]) != expected_architectures:
        raise ValueError("China heterogeneous comparison must contain IF, IG and II_1host")
    numeric = [
        "local_gpu_server_groups_industry_equivalent", "local_cpu_server_groups_industry_equivalent",
        "local_gpu_facility_energy_twh", "local_cpu_facility_energy_twh", "local_total_facility_energy_twh",
        "local_gpu_annualized_hardware_cost_rmb", "local_cpu_annualized_hardware_cost_rmb",
        "local_gpu_electricity_cost_rmb", "local_cpu_electricity_cost_rmb",
        "local_maximum_demand_cost_rmb",
        "local_battery_cost_rmb", "local_model_operations_cost_rmb",
        "local_other_modeled_cost_rmb",
        "local_joint_physical_annual_cost_rmb", "cloud_gpu_reserved_instances", "cloud_cpu_reserved_instances",
        "cloud_token_api_cost_rmb", "cloud_gpu_reserved_cost_rmb", "cloud_cpu_reserved_cost_rmb",
        "cloud_total_annual_cost_rmb",
    ]
    summary = core.groupby(["owned_architecture", "provider"], as_index=False)[numeric].sum()
    labels = {"IF": "工厂侧分布式、集团专网协同", "IG": "集团集中算力池", "II_1host": "大型集中节点"}
    summary.insert(1, "owned_architecture_label", summary.owned_architecture.map(labels))
    summary["incremental_private_network_cost_included"] = False
    summary["incremental_data_governance_cost_included"] = False
    summary["aggregation_boundary"] = "sum_of_31_industries_separately_sized"
    summary["cloud_to_local_cost_ratio"] = summary.cloud_total_annual_cost_rmb / summary.local_joint_physical_annual_cost_rmb
    summary["local_savings_vs_cloud_fraction"] = 1 - summary.local_joint_physical_annual_cost_rmb / summary.cloud_total_annual_cost_rmb
    savings_counts = (
        core.assign(ok=1-core.local_joint_physical_annual_cost_rmb/core.cloud_total_annual_cost_rmb >= .2)
        .groupby(["owned_architecture", "provider"]).ok.sum()
    )
    summary["industries_local_savings_above_20pct"] = [
        int(savings_counts.loc[(architecture, provider)])
        for architecture, provider in zip(summary.owned_architecture, summary.provider)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, encoding="utf-8-sig")
    lines = "\n".join(
        f"| {r.owned_architecture} | {r.provider} | {r.local_joint_physical_annual_cost_rmb/1e9:.3f} | {r.cloud_total_annual_cost_rmb/1e9:.3f} | {r.cloud_to_local_cost_ratio:.3f} | {r.local_savings_vs_cloud_fraction:.1%} | {int(r.industries_local_savings_above_20pct)}/31 |"
        for r in summary.itertuples(index=False)
    )
    args.findings_output.write_text(
        "# 中国31行业异构硬件成本汇总\n\nIF、IG和II_1host均在168小时物理优化中联合求解CPU/GPU装机、在线状态、任务调度、设施功率、电费、最大需量和接入容量；31个行业分别按各架构尺度定容后加总，不再使用口径校准差额。IF为核心，IG和II_1host为同口径对照。\n\n"
        "| 架构 | 云厂商 | 本地 十亿元/年 | 完整云 十亿元/年 | 云/本地 | 本地节省 | 达到20%的行业 |\n|---|---|---:|---:|---:|---:|---:|\n" + lines + "\n",
        encoding="utf-8",
    )
    payload = {"status":"complete_validated_china_31_industry_heterogeneous_cost", "industries":31, "providers":sorted(summary.provider.unique().tolist()), "owned_architectures":["IF","IG","II_1host"], "core_architecture":"IF", "incremental_private_network_cost_included":False, "incremental_data_governance_cost_included":False, "aggregation_boundary":"industry_architecture_sum"}
    args.done_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
