"""Validate single-industry PyPSA joint capacity-dispatch prototype outputs."""

from pathlib import Path

import pandas as pd

import china_minimum_prototype as parameter_core


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "05_results"


def main() -> None:
    summary = pd.read_csv(
        RESULTS / "c36_pypsa_joint_summary.csv", encoding="utf-8-sig"
    )
    hourly = pd.read_csv(
        RESULTS / "c36_pypsa_joint_hourly.csv", encoding="utf-8-sig"
    )
    assert set(summary["scenario"]) == {"baseline_no_ai", "local_ai", "cloud_ai"}
    assert len(summary) == 3
    assert len(hourly) == 72
    assert summary["solver"].str.contains("HiGHS").all()
    baseline = summary.query("scenario == 'baseline_no_ai'").iloc[0]
    assert abs(float(baseline["daily_ai_service_accelerator_h"])) < 1e-8
    assert abs(float(baseline["server_groups_2xl20_continuous"])) < 1e-8
    for scenario in ("local_ai", "cloud_ai"):
        row = summary.query("scenario == @scenario").iloc[0]
        profile = hourly.query("scenario == @scenario")
        assert abs(
            float(profile["ai_executed_accelerator_h"].sum())
            - float(row["daily_ai_service_accelerator_h"])
        ) < 1e-4
        assert float(row["server_groups_2xl20_continuous"]) > 0
        assert float(row["annual_ai_facility_energy_twh"]) > 0
        assert float(row["incremental_vs_baseline_rmb"]) > 0
        reserve = float(row["server_reserve_fraction"])
        assert (
            float(profile["ai_executed_accelerator_h"].max()) * (1.0 + reserve)
            <= float(row["server_accelerators"]) + 1e-4
        )
        params = parameter_core.read_parameters()
        idle = parameter_core.number(params, "L10")
        maximum = parameter_core.number(params, "L09")
        pue = float(row["ai_site_pue"])
        reconstructed = (
            pue
            * (
                float(row["server_groups_2xl20_continuous"]) * idle
                + profile["ai_executed_accelerator_h"]
                / 2.0
                * (maximum - idle)
            )
            / 1000.0
        )
        assert (
            reconstructed.reset_index(drop=True)
            - profile["ai_facility_power_mw"].reset_index(drop=True)
        ).abs().max() < 1e-5
    assert (
        summary["factory_rooftop_pv_mw"]
        <= summary["factory_rooftop_pv_limit_mw"] + 1e-6
    ).all()
    assert (summary["total_grid_expansion_mw"] >= -1e-7).all()
    assert (
        summary["incremental_factory_rooftop_pv_mw"].abs() < 1e-6
    ).all()
    assert (
        summary["incremental_factory_battery_power_mw"].abs() < 1e-6
    ).all()
    for _, group in hourly.groupby("scenario"):
        assert list(group.sort_values("hour")["hour"]) == list(range(24))
    print("C36 PyPSA joint prototype validation passed")


if __name__ == "__main__":
    main()
