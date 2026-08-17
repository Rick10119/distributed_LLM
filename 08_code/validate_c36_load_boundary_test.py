#!/usr/bin/env python3
"""Validate the formal C36 calibrated production-load boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--figure-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    new_meta = json.loads((args.new_root / "metadata.json").read_text(encoding="utf-8"))
    new_summary = pd.read_csv(args.new_root / "summary.csv", encoding="utf-8-sig")
    new_hourly = pd.read_csv(args.new_root / "hourly.csv", encoding="utf-8-sig")
    lineage = pd.read_csv(args.new_root / "curve_lineage.csv", encoding="utf-8-sig")
    figure = pd.read_csv(args.figure_data, encoding="utf-8-sig")

    calibration = new_meta["load_calibration"]
    expected_group_twh = (
        calibration["industry_activity_units_per_year"]
        * calibration["production_activity_share"]
        * calibration["integrated_production_electricity_intensity_mwh_per_unit"]
        / 1e6
    )
    assert np.isclose(
        expected_group_twh,
        calibration["representative_group_annual_electricity_twh"],
        rtol=0,
        atol=1e-10,
    )
    expected_factory_mean = expected_group_twh * 1e6 / 8760 / calibration["load_site_count"]
    assert np.isclose(
        expected_factory_mean,
        calibration["representative_site_mean_load_mw"],
        rtol=0,
        atol=1e-10,
    )
    assert calibration["boundary_id"] == "c36_integrated_oem_electricity_v1"
    assert calibration["mode"] == "calibrated_registry"
    assert calibration["formal_national_pool_eligible"] is True
    assert int(new_meta["production_load_site_count"]) == 9
    assert int(new_meta["modeled_routing_node_count"]) == 5
    assert new_meta["ai_deployment_node_count_by_architecture"] == {
        "IF": 9,
        "IG_1host": 1,
        "IG_multisite": 5,
    }
    assert new_meta["multisite_AI_deployment_points_equal_routing_nodes"] is True
    assert new_meta["electrical_load_aggregation_at_AI_nodes"] is False
    assert float(new_meta["group_share"]) == 0.085
    assert float(calibration["production_activity_share"]) == 0.085
    assert int(calibration["load_site_count"]) == 9
    assert "ai_factory_count" not in calibration
    assert lineage["represented_production_load_site_count"].tolist() == [2, 2, 2, 2, 1]
    assert not lineage["electrical_load_aggregation_at_AI_node"].any()
    assert (lineage["monday_to_tuesday_friday_mean_ratio"] >= 0.75).all()
    assert (lineage["weekday_daily_mean_cv"] <= 0.35).all()
    assert (lineage["weekly_peak_to_mean_ratio"] <= 3.0).all()
    assert "2018-06-18" not in set(lineage["week_start"])
    assert set(lineage["week_start"]) == {
        "2021-08-02",
        "2019-07-22",
        "2018-08-13",
        "2018-02-26",
        "2020-04-20",
    }

    expected_pairs = {
        ("IF", "actual_load"),
        ("IG_1host", "actual_load"),
        ("IG_1host", "zero_load"),
        ("IG_multisite", "actual_load"),
    }
    assert set(zip(new_summary.architecture, new_summary.base_load_case)) == expected_pairs
    service = new_summary["weekly_service_units"].to_numpy(float)
    assert np.max(np.abs(service - service[0])) / service[0] <= 1e-8
    assert max(new_meta["service_conservation_relative_error"].values()) <= 1e-8
    expected_hourly_rows = (5 + 1 + 1 + 5) * 168
    assert len(new_hourly) == expected_hourly_rows
    assert new_meta["IG_1host_factory_id"] == "R4"

    relative_metrics = figure[
        figure.metric.isin(
            [
                "base_load_fraction_of_node_weekly_peak",
                "ai_facility_power_fraction_of_node_weekly_peak",
            ]
        )
    ]
    assert relative_metrics.value.between(0, 1 + 1e-12).all()
    base_relative = relative_metrics[
        relative_metrics.metric.eq("base_load_fraction_of_node_weekly_peak")
    ]
    assert np.allclose(base_relative.groupby("factory_id").value.max(), 1.0)
    ai_relative = relative_metrics[
        relative_metrics.metric.eq("ai_facility_power_fraction_of_node_weekly_peak")
    ]
    ai_max = ai_relative.groupby("factory_id").value.max()
    assert np.allclose(ai_max[ai_max > 0], 1.0)

    multisite_hourly = new_hourly[new_hourly.architecture.eq("IG_multisite")]
    multisite_site_means = multisite_hourly.groupby("factory_id")["base_load_mw"].mean()
    assert len(multisite_site_means) == 5
    assert np.allclose(multisite_site_means, expected_factory_mean)
    one_host_hourly = new_hourly[
        new_hourly.architecture.eq("IG_1host")
        & new_hourly.base_load_case.eq("actual_load")
    ]
    host_node = new_meta["IG_1host_factory_id"]
    host_multisite = multisite_hourly[multisite_hourly.factory_id.eq(host_node)]["base_load_mw"].to_numpy()
    assert np.allclose(one_host_hourly["base_load_mw"].to_numpy(), host_multisite)

    peaks = new_summary.set_index(["architecture", "base_load_case"])["sum_incremental_grid_peak_mw"]
    assert np.isfinite(peaks.to_numpy()).all()
    assert (peaks >= 0).all()
    assert peaks.loc[("IG_multisite", "actual_load")] <= peaks.loc[("IG_1host", "actual_load")]
    payload = {
        "status": "validated_C36_formal_boundary",
        "unchanged_boundaries": {
            "group_share": 0.085,
            "physical_factory_count": 9,
            "modeled_routing_node_count": 5,
            "AI_service_conserved": True,
        },
        "representative_site_mean_load_mw": expected_factory_mean,
        "host_node": host_node,
        "IG_1host_incremental_grid_peak_mw": float(peaks.loc[("IG_1host", "actual_load")]),
        "IG_multisite_incremental_grid_peak_mw": float(peaks.loc[("IG_multisite", "actual_load")]),
        "selected_week_starts": lineage.set_index("modeled_routing_node_id")["week_start"].to_dict(),
        "zero_AI_nodes_in_multisite": ai_max[ai_max == 0].index.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("C36 load-boundary test validation passed")


if __name__ == "__main__":
    main()
