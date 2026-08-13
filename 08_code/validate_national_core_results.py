#!/usr/bin/env python3
"""Validate 31-industry equal-service reconstruction and national aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS = {"IF", "IG", "II_1host"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    if len(frame) != 31 * len(SCENARIOS):
        raise AssertionError("national summary must contain 31 industries x 3 scenarios")
    if frame["industry_code"].nunique() != 31 or set(frame["scenario"]) != SCENARIOS:
        raise AssertionError("national industry or scenario coverage is incomplete")
    if set(frame["model_version"]) != {args.model_version}:
        raise AssertionError("national summary mixes model versions")
    if frame[["industry_code", "scenario"]].duplicated().any():
        raise AssertionError("duplicate industry-scenario rows")

    for _, group in frame.groupby("industry_code"):
        demanded = group["industry_daily_effective_service_units"].to_numpy(float)
        rebuilt = group["reconstructed_industry_daily_effective_service_units"].to_numpy(float)
        if not np.allclose(demanded, demanded[0], rtol=1e-10, atol=1e-6):
            raise AssertionError("industry service demand differs across architectures")
        if not np.allclose(rebuilt, demanded, rtol=1e-9, atol=1e-5):
            raise AssertionError("industry service reconstruction failed")

    totals = (
        frame.groupby("scenario", as_index=False)
        .agg(
            national_daily_effective_service_units=("reconstructed_industry_daily_effective_service_units", "sum"),
            national_derived_reference_energy_twh=("derived_reference_energy_twh", "sum"),
            national_external_low_energy_twh=("external_energy_low_twh", "sum"),
            national_external_central_energy_twh=("external_energy_central_twh", "sum"),
            national_external_high_energy_twh=("external_energy_high_twh", "sum"),
            national_annual_ai_facility_energy_twh=("industry_equivalent_annual_ai_facility_energy_twh", "sum"),
            national_annual_model_initialization_energy_twh=("industry_equivalent_annual_model_initialization_energy_twh", "sum"),
            national_installed_server_groups=("industry_equivalent_installed_server_groups", "sum"),
            national_incremental_grid_expansion_mw=("industry_equivalent_incremental_grid_expansion_mw", "sum"),
            national_incremental_total_cost_rmb=("industry_equivalent_incremental_total_cost_rmb", "sum"),
            national_model_initialization_cost_rmb=("industry_equivalent_incremental_annual_model_initialization_cost_rmb", "sum"),
            national_model_storage_cost_rmb=("industry_equivalent_incremental_annual_model_storage_cost_rmb", "sum"),
            national_model_operations_cost_rmb=("industry_equivalent_incremental_annual_model_operations_cost_rmb", "sum"),
        )
        .sort_values("scenario")
    )
    service = totals["national_daily_effective_service_units"].to_numpy(float)
    if not np.allclose(service, service[0], rtol=1e-10, atol=1e-4):
        raise AssertionError("national service differs across architectures")
    if not (
        totals["national_annual_ai_facility_energy_twh"]
        .between(
            totals["national_external_low_energy_twh"],
            totals["national_external_high_energy_twh"],
        )
        .all()
    ):
        raise AssertionError("optimized national AI electricity is outside the external envelope")
    totals["optimized_to_external_central_ratio"] = (
        totals["national_annual_ai_facility_energy_twh"]
        / totals["national_external_central_energy_twh"]
    )

    conversion_values = frame["accelerator_h_per_service_unit"].unique()
    server_power_values = frame["server_maximum_wall_power_kw"].unique()
    efficiency_cases = frame["compute_efficiency_case"].unique()
    if len(conversion_values) != 1 or len(server_power_values) != 1 or len(efficiency_cases) != 1:
        raise AssertionError("national summary mixes active compute or server-power parameters")

    payload = {
        "status": "validated",
        "model_version": args.model_version,
        "industries": 31,
        "scenarios": sorted(SCENARIOS),
        "active_parameters": {
            "compute_efficiency_case": str(efficiency_cases[0]),
            "accelerator_h_per_service_unit": float(conversion_values[0]),
            "server_maximum_wall_power_kw": float(server_power_values[0]),
        },
        "checks": [
            "31-industry and three-scenario coverage",
            "one row per industry-scenario",
            "same effective-service demand across architectures within every industry",
            "scenario reconstruction equals industry demand",
            "same aggregate national effective service across architectures",
            "optimized national electricity inside the external 8-28 TWh envelope",
            "single model version",
            "lifecycle energy and cost components included in national aggregation",
        ],
        "national_totals": totals.to_dict("records"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
