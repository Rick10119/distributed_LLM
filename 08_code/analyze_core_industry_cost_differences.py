#!/usr/bin/env python3
"""Explain cross-industry cost differences using core scenarios only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ARCHITECTURES = ["IF", "IG", "II_1host"]
COMPONENTS = {
    "gpu_server": "industry_equivalent_incremental_annual_gpu_server_cost_rmb",
    "cpu_server": "industry_equivalent_incremental_annual_cpu_server_cost_rmb",
    "pv": "industry_equivalent_incremental_annual_pv_cost_rmb",
    "battery": "industry_equivalent_incremental_annual_battery_cost_rmb",
    "grid_energy": "industry_equivalent_incremental_annual_flat_energy_cost_rmb",
    "maximum_demand": "industry_equivalent_incremental_annual_maximum_demand_cost_rmb",
    "model_initialization": "industry_equivalent_incremental_annual_model_initialization_cost_rmb",
    "model_storage": "industry_equivalent_incremental_annual_model_storage_cost_rmb",
    "model_operations": "industry_equivalent_incremental_annual_model_operations_cost_rmb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--national-input", type=Path, required=True)
    parser.add_argument("--service-input", type=Path, required=True)
    parser.add_argument("--routing-config", type=Path, required=True)
    parser.add_argument("--detail-output", type=Path, required=True)
    parser.add_argument("--association-output", type=Path, required=True)
    parser.add_argument("--decomposition-output", type=Path, required=True)
    parser.add_argument("--findings-output", type=Path, required=True)
    parser.add_argument("--done-output", type=Path, required=True)
    return parser.parse_args()


def workload_drivers(service_path: Path, routing_path: Path) -> pd.DataFrame:
    service = pd.read_csv(service_path, encoding="utf-8-sig")
    service = service.loc[service["parameter_case"].eq("base")].copy()
    routing = yaml.safe_load(routing_path.read_text(encoding="utf-8"))
    route_case = str(routing["active_core_routing_case"])
    cpu_fraction = routing["routing_cases"][route_case]
    cpu_multiplier = {
        key: value
        for key, value in routing["core_cpu_server_hour_per_reference_l20_accelerator_hour"].items()
        if key != "rationale"
    }
    service["cpu_fraction"] = service["task_id"].map(cpu_fraction).fillna(0.0).astype(float)
    service["cpu_multiplier"] = service["task_id"].map(cpu_multiplier).fillna(1.0).astype(float)
    service["cpu_service"] = service["effective_service_units_day"] * service["cpu_fraction"]
    service["cpu_compute"] = service["cpu_service"] * service["cpu_multiplier"]
    service["gpu_compute"] = service["effective_service_units_day"] * (1.0 - service["cpu_fraction"])
    totals = service.groupby(["industry_code", "industry_name_cn"], as_index=False).agg(
        task_service_units_day=("effective_service_units_day", "sum"),
        cpu_service_units_day=("cpu_service", "sum"),
        cpu_compute_units_day=("cpu_compute", "sum"),
        gpu_compute_units_day=("gpu_compute", "sum"),
    )
    shares = service.merge(
        totals[["industry_code", "task_service_units_day"]], on="industry_code", how="left"
    )
    shares["task_share_sq"] = (
        shares["effective_service_units_day"] / shares["task_service_units_day"]
    ) ** 2
    hhi = shares.groupby("industry_code", as_index=False)["task_share_sq"].sum().rename(
        columns={"task_share_sq": "task_mix_hhi"}
    )
    totals["cpu_routed_service_share"] = (
        totals["cpu_service_units_day"] / totals["task_service_units_day"]
    )
    totals["cpu_compute_share"] = totals["cpu_compute_units_day"] / (
        totals["cpu_compute_units_day"] + totals["gpu_compute_units_day"]
    )
    totals["active_routing_case"] = route_case
    return totals.merge(hhi, on="industry_code", how="left")


def finite_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float).div(denominator.astype(float).replace(0.0, np.nan))


def build_detail(core: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    required = {
        "industry_code", "industry_name", "scenario", "industry_daily_effective_service_units",
        "industry_equivalent_incremental_total_cost_rmb", "equivalent_host_multiplier",
        "industry_equivalent_annual_ai_facility_energy_twh",
        "industry_equivalent_incremental_grid_expansion_mw",
        "industry_equivalent_installed_gpu_server_groups",
        "industry_equivalent_installed_cpu_server_groups",
        "capacity_reference_gpu_server_groups_unshifted_arrival_peak",
        "capacity_reference_cpu_server_groups_unshifted_arrival_peak",
        *COMPONENTS.values(),
    }
    missing = sorted(required - set(core.columns))
    if missing:
        raise ValueError(f"National core summary misses columns: {missing}")
    core = core.loc[core["scenario"].isin(ARCHITECTURES)].copy()
    if len(core) != 31 * len(ARCHITECTURES):
        raise ValueError(f"Expected 93 core rows (31 industries x 3 architectures), got {len(core)}")
    if core.groupby("industry_code")["scenario"].nunique().ne(3).any():
        raise ValueError("Each industry must contain IF, IG and II_1host")

    detail = core.merge(drivers, on="industry_code", how="left", validate="many_to_one")
    service_relative_error = (
        detail["task_service_units_day"] - detail["industry_daily_effective_service_units"]
    ).abs() / detail["industry_daily_effective_service_units"].replace(0.0, np.nan)
    if service_relative_error.max() > 1e-6:
        raise ValueError("Task-level service totals do not match the core-scenario demand totals")
    component_reconstruction = detail[list(COMPONENTS.values())].sum(axis=1)
    component_error = (
        component_reconstruction - detail["industry_equivalent_incremental_total_cost_rmb"]
    ).abs()
    component_tolerance = np.maximum(
        1e-3, detail["industry_equivalent_incremental_total_cost_rmb"].abs() * 1e-9
    )
    if (component_error > component_tolerance).any():
        raise ValueError("Cost components do not reconstruct total annual cost")
    detail["annual_service_units"] = detail["industry_daily_effective_service_units"] * 365.0
    detail["annual_cost_rmb_per_service_unit"] = finite_ratio(
        detail["industry_equivalent_incremental_total_cost_rmb"], detail["annual_service_units"]
    )
    for label, column in COMPONENTS.items():
        detail[f"{label}_cost_rmb_per_service_unit"] = finite_ratio(
            detail[column], detail["annual_service_units"]
        )
        detail[f"{label}_cost_share"] = finite_ratio(
            detail[column], detail["industry_equivalent_incremental_total_cost_rmb"]
        )
    detail["energy_kwh_per_service_unit"] = finite_ratio(
        detail["industry_equivalent_annual_ai_facility_energy_twh"] * 1e9,
        detail["annual_service_units"],
    )
    detail["grid_expansion_mw_per_million_daily_service_units"] = finite_ratio(
        detail["industry_equivalent_incremental_grid_expansion_mw"],
        detail["industry_daily_effective_service_units"] / 1e6,
    )
    detail["equivalent_hosts_per_million_daily_service_units"] = finite_ratio(
        detail["equivalent_host_multiplier"],
        detail["industry_daily_effective_service_units"] / 1e6,
    )
    gpu_reference = (
        detail["capacity_reference_gpu_server_groups_unshifted_arrival_peak"]
        * detail["equivalent_host_multiplier"]
    )
    cpu_reference = (
        detail["capacity_reference_cpu_server_groups_unshifted_arrival_peak"]
        * detail["equivalent_host_multiplier"]
    )
    detail["gpu_installed_to_reference_ratio"] = finite_ratio(
        detail["industry_equivalent_installed_gpu_server_groups"], gpu_reference
    )
    detail["cpu_installed_to_reference_ratio"] = finite_ratio(
        detail["industry_equivalent_installed_cpu_server_groups"], cpu_reference
    )
    detail["absolute_cost_rank_within_architecture"] = detail.groupby("scenario")[
        "industry_equivalent_incremental_total_cost_rmb"
    ].rank(method="min", ascending=False)
    detail["unit_cost_rank_within_architecture"] = detail.groupby("scenario")[
        "annual_cost_rmb_per_service_unit"
    ].rank(method="min", ascending=False)
    return detail


def build_associations(detail: pd.DataFrame) -> pd.DataFrame:
    drivers = {
        "daily_service_scale": ("industry_daily_effective_service_units", "structural_input"),
        "cpu_routed_service_share": ("cpu_routed_service_share", "structural_input"),
        "cpu_compute_share": ("cpu_compute_share", "structural_input"),
        "task_mix_hhi": ("task_mix_hhi", "structural_input"),
        "deployment_fragmentation_intensity": (
            "equivalent_hosts_per_million_daily_service_units", "architecture_mechanism"
        ),
        "gpu_installed_capacity_overhead": (
            "gpu_installed_to_reference_ratio", "architecture_mechanism"
        ),
        "cpu_installed_capacity_overhead": (
            "cpu_installed_to_reference_ratio", "architecture_mechanism"
        ),
        "energy_intensity": ("energy_kwh_per_service_unit", "accounting_channel"),
        "grid_capacity_intensity": (
            "grid_expansion_mw_per_million_daily_service_units", "accounting_channel"
        ),
    }
    rows: list[dict[str, object]] = []
    for architecture in ARCHITECTURES:
        subset = detail.loc[detail["scenario"].eq(architecture)]
        for label, (column, role) in drivers.items():
            pair = subset[[column, "annual_cost_rmb_per_service_unit"]].dropna()
            rho = pair.corr(method="spearman").iloc[0, 1] if len(pair) >= 3 else np.nan
            rows.append({
                "scenario": architecture,
                "driver": label,
                "driver_role": role,
                "spearman_rho_with_unit_cost": rho,
                "industry_count": len(pair),
                "interpretation": "descriptive_association_not_causal_attribution",
            })
    return pd.DataFrame(rows)


def build_extreme_decomposition(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for architecture in ARCHITECTURES:
        subset = detail.loc[detail["scenario"].eq(architecture)].copy()
        low = subset.loc[subset["annual_cost_rmb_per_service_unit"].idxmin()]
        high = subset.loc[subset["annual_cost_rmb_per_service_unit"].idxmax()]
        total_gap = high["annual_cost_rmb_per_service_unit"] - low["annual_cost_rmb_per_service_unit"]
        for label in COMPONENTS:
            column = f"{label}_cost_rmb_per_service_unit"
            contribution = high[column] - low[column]
            rows.append({
                "scenario": architecture,
                "high_unit_cost_industry": high["industry_code"],
                "high_unit_cost_industry_name": high["industry_name"],
                "low_unit_cost_industry": low["industry_code"],
                "low_unit_cost_industry_name": low["industry_name"],
                "total_unit_cost_gap_rmb_per_service_unit": total_gap,
                "cost_component": label,
                "component_gap_rmb_per_service_unit": contribution,
                "component_share_of_total_gap": contribution / total_gap if total_gap else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    core = pd.read_csv(args.national_input, encoding="utf-8-sig")
    detail = build_detail(core, workload_drivers(args.service_input, args.routing_config))
    associations = build_associations(detail)
    decomposition = build_extreme_decomposition(detail)

    for output in (
        args.detail_output, args.association_output, args.decomposition_output,
        args.findings_output, args.done_output,
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.detail_output, index=False, encoding="utf-8-sig")
    associations.to_csv(args.association_output, index=False, encoding="utf-8-sig")
    decomposition.to_csv(args.decomposition_output, index=False, encoding="utf-8-sig")

    lines = [
        "# Core-scenario cross-industry cost differences",
        "",
        "This analysis changes no sensitivity parameter. It compares the 31 industries only under IF, IG and II_1host core scenarios.",
        "Absolute annual cost and cost per annual service unit are reported separately. Component differences are accounting identities; Spearman coefficients are descriptive associations and are not causal attribution.",
        "",
    ]
    for architecture in ARCHITECTURES:
        subset = detail.loc[detail["scenario"].eq(architecture)]
        high = subset.loc[subset["annual_cost_rmb_per_service_unit"].idxmax()]
        low = subset.loc[subset["annual_cost_rmb_per_service_unit"].idxmin()]
        lines.append(
            f"- {architecture}: unit cost ranges from {low['annual_cost_rmb_per_service_unit']:.4g} "
            f"RMB/service unit ({low['industry_code']}) to {high['annual_cost_rmb_per_service_unit']:.4g} "
            f"RMB/service unit ({high['industry_code']})."
        )
    args.findings_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "status": "validated_core_only_cross_industry_cost_analysis",
        "industry_count": int(detail["industry_code"].nunique()),
        "architectures": ARCHITECTURES,
        "row_count": len(detail),
        "sensitivity_parameters_changed": False,
        "outputs": {
            "detail": str(args.detail_output),
            "associations": str(args.association_output),
            "extreme_gap_decomposition": str(args.decomposition_output),
        },
    }
    args.done_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
