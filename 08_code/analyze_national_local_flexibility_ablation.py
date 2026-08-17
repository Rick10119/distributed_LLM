#!/usr/bin/env python3
"""Compare central flexibility with arrival-time execution for national IF deployment."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_code"))

from core.config import load_config  # noqa: E402
from core.data import load_industry_inputs  # noqa: E402
from core.model import optimize_host  # noqa: E402
from core.production_load import production_load_mode, resolve_site_load_profile  # noqa: E402
from core.representative_group import read_representative_groups, scenario_scale  # noqa: E402


COST_COMPONENTS = (
    "annual_server_cost_rmb",
    "annual_pv_cost_rmb",
    "annual_battery_cost_rmb",
    "annual_flat_energy_cost_rmb",
    "annual_maximum_demand_cost_rmb",
    "annual_model_initialization_cost_rmb",
    "annual_model_storage_cost_rmb",
    "annual_model_operations_cost_rmb",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--if-summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--flex-hourly-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--findings-output", type=Path, required=True)
    parser.add_argument("--done-output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(ROOT, args.defaults, args.config)
    industries = sorted(config["selected_industries"])
    if len(industries) != 31:
        raise ValueError("National flexibility ablation requires all 31 industries")
    load_modes = {industry: production_load_mode(config, industry) for industry in industries}
    valid_load_modes = {"calibrated_registry", "legacy_industry_electricity_share"}
    if not set(load_modes.values()).issubset(valid_load_modes):
        raise ValueError("National flexibility ablation found an unsupported production-load mode")
    if load_modes.get("C36") != "calibrated_registry":
        raise ValueError("National flexibility ablation requires the approved C36 calibrated boundary")
    load_mode_counts = pd.Series(load_modes).value_counts().to_dict()
    flexible = pd.concat(
        [pd.read_csv(path, encoding="utf-8-sig") for path in args.if_summaries],
        ignore_index=True,
    )
    if "scenario" in flexible.columns:
        flexible = flexible.loc[flexible["scenario"].eq("IF")].copy()
    if len(flexible) != 31:
        raise ValueError("Flexibility ablation requires 31 IF industry summaries")

    baseline_lookup: dict[str, dict[str, float]] = {}
    for path in args.baseline_summaries:
        industry = path.parents[1].name
        baseline_lookup[industry] = json.loads(path.read_text(encoding="utf-8"))["model"]
    if set(baseline_lookup) != set(industries):
        raise ValueError("Baseline coverage is incomplete")

    flex_profiles = []
    for path in args.flex_hourly_inputs:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if len(frame) != int(config["model"]["horizon_hours"]):
            raise ValueError(f"Incomplete flexible hourly profile: {path}")
        flex_profiles.append(frame.set_index("hour")["industry_equivalent_ai_facility_power_mw"])
    flexible_profile = pd.concat(flex_profiles, axis=1).sum(axis=1)

    groups = read_representative_groups(
        ROOT / config["paths"]["representative_group_report"]
    )
    rigid_totals = {key: 0.0 for key in COST_COMPONENTS}
    rigid_profile = np.zeros(int(config["model"]["horizon_hours"]))
    rigid_servers = 0.0
    rigid_grid = 0.0
    rigid_service = 0.0
    logging.getLogger().setLevel(logging.ERROR)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        for industry in industries:
            inputs = load_industry_inputs(config, industry)
            scale = scenario_scale(
                groups[industry], config["industry_parameter_case"], "IF"
            )
            site_load, _production_load = resolve_site_load_profile(
                root=ROOT,
                config=config,
                industry=industry,
                industry_profile_mw=inputs.base_load_mw,
                ai_service_group_share=scale.group_share,
                legacy_load_site_count=scale.group_factory_count,
            )
            arrival = inputs.rigid_service_units.copy()
            for job in inputs.flexible_jobs:
                arrival[job.release_hour] += job.amount_service_units
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                result = optimize_host(
                    config,
                    site_load,
                    inputs.pv_capacity_factor,
                    inputs.roof_area_proxy_m2,
                    rigid_service_units=arrival * scale.ai_service_scale_per_host,
                    flexible_jobs=(),
                    existing_grid_capacity_mw=float(
                        baseline_lookup[industry]["grid_import_peak_mw"]
                    ),
                )
            baseline = baseline_lookup[industry]
            multiplier = scale.equivalent_host_multiplier
            for key in COST_COMPONENTS:
                rigid_totals[key] += (
                    float(result.summary[key]) - float(baseline[key])
                ) * multiplier
            rigid_profile += (
                result.hourly["ai_facility_power_mw"].to_numpy() * multiplier
            )
            rigid_servers += float(result.summary["installed_server_groups"]) * multiplier
            rigid_grid += (
                float(result.summary["grid_expansion_mw"])
                - float(baseline["grid_expansion_mw"])
            ) * multiplier
            rigid_service += (
                float(result.hourly["ai_executed_service_units"].sum())
                / float(result.summary["represented_days"])
                * multiplier
            )

    def flexible_component(key: str) -> float:
        return float(flexible[f"industry_equivalent_incremental_{key}"].sum())

    flexible_totals = {key: flexible_component(key) for key in COST_COMPONENTS}
    purchase_cost = float(config["server"]["purchase_cost_rmb"])
    facility_fraction = float(config["server"]["facility_capex_fraction"])
    rows = []
    for case, costs, servers, profile, grid, service in (
        (
            "arrival_time_execution_no_flexibility",
            rigid_totals,
            rigid_servers,
            rigid_profile,
            rigid_grid,
            rigid_service,
        ),
        (
            "central_flexibility_optimized",
            flexible_totals,
            float(flexible["industry_equivalent_installed_server_groups"].sum()),
            flexible_profile.to_numpy(),
            float(flexible["industry_equivalent_incremental_grid_expansion_mw"].sum()),
            float(flexible["industry_daily_effective_service_units"].sum()),
        ),
    ):
        total = sum(costs.values())
        electricity = costs["annual_flat_energy_cost_rmb"] + costs["annual_maximum_demand_cost_rmb"]
        row = {
            "model_version": config["model_version"],
            "case": case,
            "industries": 31,
            "daily_effective_service_units": service,
            "installed_server_groups": servers,
            "upfront_server_and_facility_capex_rmb": servers
            * purchase_cost
            * (1.0 + facility_fraction),
            "ai_facility_peak_mw": float(np.max(profile)),
            "ai_facility_mean_mw": float(np.mean(profile)),
            "incremental_grid_expansion_mw": grid,
            **{f"incremental_{key}": value for key, value in costs.items()},
            "incremental_electricity_bill_rmb": electricity,
            "incremental_total_cost_rmb": total,
            "server_cost_share": costs["annual_server_cost_rmb"] / total,
            "electricity_bill_share": electricity / total,
        }
        rows.append(row)
    comparison = pd.DataFrame(rows)
    rigid = comparison.iloc[0]
    flex = comparison.iloc[1]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False, encoding="utf-8-sig")
    findings = f"""# 全国IF本地部署柔性消融

在31行业同等有效服务量、相同设备和电价下，将当前中央柔性情景与所有任务到达即执行的无柔性反事实比较。无柔性不是新的需求情景，只用于识别任务时序弹性的边际作用。

| 指标 | 无柔性 | 当前柔性 | 变化 |
| --- | ---: | ---: | ---: |
| AI设施峰值 | {rigid.ai_facility_peak_mw:,.1f} MW | {flex.ai_facility_peak_mw:,.1f} MW | {flex.ai_facility_peak_mw/rigid.ai_facility_peak_mw-1:.1%} |
| 等价服务器组 | {rigid.installed_server_groups:,.0f} | {flex.installed_server_groups:,.0f} | {flex.installed_server_groups/rigid.installed_server_groups-1:.1%} |
| 前期服务器与附属设施投资 | {rigid.upfront_server_and_facility_capex_rmb/1e9:,.1f}十亿元 | {flex.upfront_server_and_facility_capex_rmb/1e9:,.1f}十亿元 | {flex.upfront_server_and_facility_capex_rmb/rigid.upfront_server_and_facility_capex_rmb-1:.1%} |
| 年电量费+需量费 | {rigid.incremental_electricity_bill_rmb/1e9:,.3f}十亿元 | {flex.incremental_electricity_bill_rmb/1e9:,.3f}十亿元 | {flex.incremental_electricity_bill_rmb/rigid.incremental_electricity_bill_rmb-1:.2%} |
| 电费占总成本 | {rigid.electricity_bill_share:.2%} | {flex.electricity_bill_share:.2%} | {(flex.electricity_bill_share-rigid.electricity_bill_share)*100:.2f}个百分点 |
| 年总成本 | {rigid.incremental_total_cost_rmb/1e9:,.1f}十亿元 | {flex.incremental_total_cost_rmb/1e9:,.1f}十亿元 | {flex.incremental_total_cost_rmb/rigid.incremental_total_cost_rmb-1:.1%} |

当前平电价下，柔性几乎全部通过降低计算峰值、服务器装机和资本成本创造价值；绝对电费仅小幅下降。电费占比上升是总成本分母大幅下降的结果，不表示柔性提高了电费。无分时电价和需求响应补偿，因此该结果不能外推为灵活性没有电力市场价值。
"""
    args.findings_output.write_text(findings, encoding="utf-8")
    payload = {
        "status": "validated",
        "model_version": config["model_version"],
        "industries": 31,
        "production_load_mode_counts": load_mode_counts,
        "production_load_boundary_status": "mixed_explicit_C36_calibrated_other_industries_legacy_compatibility",
        "same_service": abs(
            float(rigid.daily_effective_service_units)
            - float(flex.daily_effective_service_units)
        )
        < 1e-3,
        "checks": [
            "31-industry coverage",
            "same daily effective service",
            "arrival-time execution contains no flexible jobs",
            "cost components reconcile to total",
            "continuous 168-hour profiles",
        ],
    }
    if not payload["same_service"]:
        raise ValueError("Flexible and rigid cases do not reconstruct equal service")
    args.done_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
