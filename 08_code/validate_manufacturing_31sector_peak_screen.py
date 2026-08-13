"""Integrity checks for archived equal-electricity 31-sector outputs."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "02_data"
RESULTS = ROOT / "05_results"
LEGACY_RESULTS = RESULTS / "archive" / "equal_electricity_national"


def main() -> None:
    selection = pd.read_csv(DATA / "manufacturing_31sector_curve_selection.csv", encoding="utf-8-sig")
    detail = pd.read_csv(LEGACY_RESULTS / "manufacturing_31sector_peak_screen.csv", encoding="utf-8-sig")
    summary = pd.read_csv(LEGACY_RESULTS / "manufacturing_31sector_peak_summary.csv", encoding="utf-8-sig")
    hourly = pd.read_csv(LEGACY_RESULTS / "manufacturing_31sector_hourly_peak_profiles.csv", encoding="utf-8-sig")
    aggregate = pd.read_csv(LEGACY_RESULTS / "manufacturing_31sector_aggregate_ai_profiles.csv", encoding="utf-8-sig")
    baseline = pd.read_csv(DATA / "manufacturing_31sector_electricity_baseline.csv", encoding="utf-8-sig")

    assert len(selection) == 31 and selection["industry_code"].nunique() == 31
    assert len(baseline) == 31 and baseline["industry_code"].nunique() == 31
    assert np.isclose(baseline["electricity_2030_twh"].sum(), 4799.3953722330625)
    assert selection["selected_curve_source_type"].value_counts().to_dict() == {
        "china_eweld_isic_specific": 26,
        "six_archetype_fallback": 3,
        "dedicated_external_facility": 1,
        "dedicated_external_sector_model": 1,
    }
    assert len(detail) == 31 * 3 * 3
    assert detail.groupby(["industry_code", "energy_scenario", "temporal_scenario"]).size().eq(1).all()
    assert len(summary) == 31 and summary["industry_code"].nunique() == 31
    assert len(hourly) == 31 * 3 * 24
    assert hourly.groupby(["industry_code", "temporal_scenario"])["hour"].nunique().eq(24).all()
    assert len(aggregate) == 3 * 3 * 24

    for (_, _), group in hourly.groupby(["industry_code", "temporal_scenario"]):
        assert np.isclose(group["base_normalized_load"].mean(), 1.0, atol=1e-6)
        assert np.isclose(group["ai_normalized_load"].mean(), 1.0, atol=1e-6)
        assert (group[["base_normalized_load", "ai_normalized_load"]].to_numpy() >= 0).all()

    flat = hourly[hourly["temporal_scenario"] == "flat"]
    assert np.allclose(flat["ai_normalized_load"], 1.0)
    assert (summary["central_ai_peak_mw"] >= summary["central_ai_average_mw"]).all()
    for column in [c for c in summary.columns if c.startswith("incremental_peak_per_ai_average_")]:
        assert (summary[column] >= -1e-8).all()
        assert (summary[column] <= summary["ai_peak_factor"] + 1e-8).all()

    expected_twh = {"lower_8twh": 8.0, "central_14twh": 14.0, "upper_28twh": 28.0}
    for scenario, total_twh in expected_twh.items():
        annual = detail[(detail["energy_scenario"] == scenario) & (detail["temporal_scenario"] == "task_timed")]["annual_ai_twh"].sum()
        assert np.isclose(annual, total_twh, atol=1e-6)
        for temporal in detail["temporal_scenario"].unique():
            profile = aggregate[(aggregate["energy_scenario"] == scenario) & (aggregate["temporal_scenario"] == temporal)]
            assert np.isclose(profile["aggregate_ai_load_mw"].mean(), total_twh * 1e6 / 8760, atol=1e-6)
            assert np.isclose(profile["aggregate_base_load_mw"].mean(), baseline["electricity_2030_twh"].sum() * 1e6 / 8760, atol=1e-6)
            assert np.allclose(
                profile["aggregate_combined_load_mw"],
                profile["aggregate_base_load_mw"] + profile["aggregate_ai_load_mw"],
            )

    figure_names = [
        "manufacturing_31sector_peak_example_profiles",
        "manufacturing_31sector_ai_average_peak",
        "manufacturing_31sector_aggregate_ai_profiles",
    ]
    for stem in figure_names:
        for extension in ("svg", "png"):
            path = LEGACY_RESULTS / "figures" / f"{stem}.{extension}"
            assert path.is_file() and path.stat().st_size > 1000

    central = aggregate[(aggregate["energy_scenario"] == "central_14twh") & (aggregate["temporal_scenario"] == "task_timed")]
    output = "\n".join(
        [
            "Manufacturing 31-sector peak screen validation: PASS",
            "Industries: 31",
            "Energy scenarios: 3",
            "Temporal scenarios: 3",
            f"Central average MW: {central['aggregate_ai_load_mw'].mean():.6f}",
            f"Central task-timed peak MW: {central['aggregate_ai_load_mw'].max():.6f}",
            f"Curve selection: {selection['selected_curve_source_type'].value_counts().to_dict()}",
        ]
    ) + "\n"
    (LEGACY_RESULTS / "manufacturing_31sector_peak_validation.txt").write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
