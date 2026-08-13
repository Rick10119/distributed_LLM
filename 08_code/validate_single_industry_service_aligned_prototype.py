"""Validation checks for the C36 equal-service prototype."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "05_results"


def main() -> None:
    summary = pd.read_csv(
        RESULTS / "c36_service_aligned_prototype_summary.csv",
        encoding="utf-8-sig",
    )
    hourly = pd.read_csv(
        RESULTS / "c36_service_aligned_prototype_hourly.csv",
        encoding="utf-8-sig",
    )
    assert len(summary) == 4
    assert set(summary["architecture"]) == {
        "IF_local_industry_bucket",
        "IC_cloud_pool",
    }
    assert set(summary["dispatch_case"]) == {"unshifted", "optimized"}
    assert len(hourly) == 96
    assert (summary["daily_service_accelerator_h"].max() - summary["daily_service_accelerator_h"].min()) < 1e-6
    shares = summary.iloc[0][
        ["rigid_service_share", "intraday_service_share", "batch_service_share"]
    ].astype(float)
    assert abs(float(shares.sum()) - 1.0) < 1e-9
    cloud = summary[summary["architecture"] == "IC_cloud_pool"].set_index("dispatch_case")
    local = summary[summary["architecture"] == "IF_local_industry_bucket"].set_index("dispatch_case")
    assert abs(float(cloud.loc["unshifted", "annual_facility_energy_twh"]) - float(cloud.loc["unshifted", "reference_cloud_energy_twh"])) < 2e-6
    assert abs(float(cloud.loc["optimized", "annual_facility_energy_twh"]) - float(cloud.loc["unshifted", "annual_facility_energy_twh"])) < 1e-9
    assert abs(float(local.loc["optimized", "annual_facility_energy_twh"]) - float(local.loc["unshifted", "annual_facility_energy_twh"])) < 1e-9
    assert float(local.loc["optimized", "combined_peak_mw"]) <= float(local.loc["unshifted", "combined_peak_mw"]) + 1e-6
    assert float(cloud.loc["optimized", "facility_peak_mw"]) <= float(cloud.loc["unshifted", "facility_peak_mw"]) + 1e-6
    for keys, group in hourly.groupby(["architecture", "dispatch_case"]):
        assert list(group.sort_values("hour")["hour"]) == list(range(24))
        architecture, dispatch_case = keys
        expected_service = float(
            summary.query(
                "architecture == @architecture and dispatch_case == @dispatch_case"
            )["daily_service_accelerator_h"].iloc[0]
        )
        assert abs(float(group["executed_accelerator_h"].sum()) - expected_service) < 1e-5
        installed_accelerators = float(
            summary.query(
                "architecture == @architecture and dispatch_case == @dispatch_case"
            )["installed_accelerators"].iloc[0]
        )
        assert float(group["executed_accelerator_h"].max()) <= installed_accelerators + 1e-5
    print("C36 service-aligned prototype validation passed")


if __name__ == "__main__":
    main()
