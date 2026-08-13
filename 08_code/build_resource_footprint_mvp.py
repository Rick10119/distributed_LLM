"""Build an evidence-bounded on-site water and facility-reuse screening MVP.

The MVP deliberately reports site water use rather than calibrated withdrawal or
consumption. It also reports reuse constraint flags rather than unsupported area
intensities. Electricity-generation water is outside the boundary.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "05_results" / "v0.6.0" / "model"
DATA_ROOT = ROOT / "02_data" / "processed" / "resource_footprint"
WATER_PARAMETERS = DATA_ROOT / "resource_footprint_water_scenarios.csv"
REUSE_PARAMETERS = DATA_ROOT / "resource_footprint_reuse_scenarios.csv"
PROVINCE_ALLOCATION = DATA_ROOT / "province_industry_ai_allocation.csv"
ASSIGNMENTS_OUTPUT = DATA_ROOT / "resource_footprint_facility_assignments_mvp.csv"
PROVINCE_WATER_OUTPUT = DATA_ROOT / "province_site_water_mvp.csv"
SUMMARY_OUTPUT = DATA_ROOT / "resource_footprint_architecture_summary_mvp.csv"
LINEAGE_OUTPUT = DATA_ROOT / "resource_footprint_mvp.lineage.json"

ARCHITECTURES = ("IF", "IG", "II_1host")
WATER_CASES = ("low_water", "central", "high_water")
REUSE_CASES = ("high_reuse", "conditional_reuse", "low_reuse")
CURRENT_FACILITY_MULTIPLIER = 1.30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_facility(installed_it_capacity_kw: float) -> str:
    if installed_it_capacity_kw <= 50.0:
        return "small"
    if installed_it_capacity_kw <= 500.0:
        return "medium"
    return "large"


def load_model_summaries(model_root: Path = MODEL_ROOT) -> pd.DataFrame:
    paths = sorted(model_root.glob("C*/**/summary.csv"))
    frames = [pd.read_csv(path, encoding="utf-8-sig") for path in paths]
    if not frames:
        raise ValueError(f"No model summaries found under {model_root}")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["scenario"].isin(ARCHITECTURES)].copy()
    if len(frame) != 31 * len(ARCHITECTURES):
        raise ValueError(f"Expected 93 industry-architecture summaries, found {len(frame)}")
    if frame.duplicated(["industry_code", "scenario"]).any():
        raise ValueError("Duplicate industry-architecture model summaries found")
    if set(frame["scenario"]) != set(ARCHITECTURES):
        raise ValueError("The model summaries do not cover all required architectures")
    if not (frame["server_marginal_facility_multiplier"] == CURRENT_FACILITY_MULTIPLIER).all():
        raise ValueError("Unexpected current facility multiplier; IT-energy recovery would be invalid")
    return frame


def load_parameters() -> tuple[pd.DataFrame, pd.DataFrame]:
    water = pd.read_csv(WATER_PARAMETERS, encoding="utf-8-sig")
    reuse = pd.read_csv(REUSE_PARAMETERS, encoding="utf-8-sig")
    if len(water) != 9 or len(reuse) != 9:
        raise ValueError("Expected three facility scales by three cases in each parameter table")
    if water.duplicated(["water_case", "facility_scale"]).any():
        raise ValueError("Duplicate water parameter rows")
    if reuse.duplicated(["reuse_case", "facility_scale"]).any():
        raise ValueError("Duplicate reuse parameter rows")
    if set(water["water_case"]) != set(WATER_CASES):
        raise ValueError("Unexpected water cases")
    if set(reuse["reuse_case"]) != set(REUSE_CASES):
        raise ValueError("Unexpected reuse cases")
    for _, group in water.groupby("facility_scale"):
        ordered = group.set_index("water_case").loc[list(WATER_CASES), "site_water_use_l_per_kwh_it"]
        if not ordered.is_monotonic_increasing:
            raise ValueError("Water cases must be monotonic within each facility scale")
    reuse_columns = [
        "room_reuse",
        "electrical_reuse",
        "cooling_reuse",
        "fire_safety_reuse",
        "outdoor_space_reuse",
        "existing_site_land_available",
        "zero_new_land_conversion",
        "zero_new_building_area",
        "zero_new_indoor_outdoor_equipment_space",
    ]
    for column in reuse_columns:
        if not reuse[column].isin([0, 1]).all():
            raise ValueError(f"Reuse flag {column} must be binary")
    all_five = reuse[
        ["room_reuse", "electrical_reuse", "cooling_reuse", "fire_safety_reuse", "outdoor_space_reuse"]
    ].all(axis=1)
    if not (reuse["zero_new_indoor_outdoor_equipment_space"].astype(bool) == all_five).all():
        raise ValueError("Zero new equipment space requires all five reuse constraints")
    return water, reuse


def build_assignments() -> pd.DataFrame:
    model = load_model_summaries()
    water, reuse = load_parameters()

    selected = model[
        [
            "model_version",
            "industry_code",
            "industry_name",
            "scenario",
            "server_maximum_wall_power_kw",
            "server_marginal_facility_multiplier",
            "physical_host_count",
            "equivalent_host_multiplier",
            "per_host_installed_server_groups",
            "industry_equivalent_installed_server_groups",
            "per_host_ai_facility_peak_mw",
            "per_host_annual_ai_facility_energy_twh",
            "per_host_annual_model_initialization_energy_twh",
            "industry_equivalent_annual_ai_facility_energy_including_initialization_twh",
        ]
    ].copy()
    selected = selected.rename(columns={"scenario": "architecture"})
    selected["per_host_installed_it_capacity_kw"] = (
        selected["per_host_installed_server_groups"] * selected["server_maximum_wall_power_kw"]
    )
    selected["industry_equivalent_installed_it_capacity_mw"] = (
        selected["industry_equivalent_installed_server_groups"]
        * selected["server_maximum_wall_power_kw"]
        / 1000.0
    )
    selected["facility_scale"] = selected["per_host_installed_it_capacity_kw"].map(classify_facility)
    selected["per_host_annual_it_energy_twh"] = (
        selected["per_host_annual_ai_facility_energy_twh"]
        + selected["per_host_annual_model_initialization_energy_twh"]
    ) / selected["server_marginal_facility_multiplier"]
    selected["industry_equivalent_annual_it_energy_twh"] = (
        selected["industry_equivalent_annual_ai_facility_energy_including_initialization_twh"]
        / selected["server_marginal_facility_multiplier"]
    )

    assignments = selected.merge(water, on="facility_scale", how="left", validate="many_to_many")
    assignments = assignments.merge(reuse, on="facility_scale", how="left", validate="many_to_many")
    expected_rows = len(selected) * len(WATER_CASES) * len(REUSE_CASES)
    if len(assignments) != expected_rows:
        raise ValueError(f"Expected {expected_rows} assignment rows, found {len(assignments)}")

    assignments["per_host_site_water_use_m3"] = (
        assignments["per_host_annual_it_energy_twh"]
        * assignments["site_water_use_l_per_kwh_it"]
        * 1_000_000.0
    )
    assignments["industry_equivalent_site_water_use_m3"] = (
        assignments["industry_equivalent_annual_it_energy_twh"]
        * assignments["site_water_use_l_per_kwh_it"]
        * 1_000_000.0
    )
    assignments["unmet_reuse_constraint_count"] = 5 - assignments[
        ["room_reuse", "electrical_reuse", "cooling_reuse", "fire_safety_reuse", "outdoor_space_reuse"]
    ].sum(axis=1)
    assignments["water_result_boundary"] = "site_water_use_not_calibrated_withdrawal_or_consumption"
    assignments["freshwater_share"] = pd.NA
    assignments["scarcity_weighted_water"] = pd.NA
    assignments["new_floor_area_m2"] = pd.NA
    assignments["new_site_area_m2"] = pd.NA
    assignments["noise_result_status"] = "equipment_and_operating_condition_screen_only"
    assignments = assignments.sort_values(
        ["industry_code", "architecture", "water_case", "reuse_case"]
    ).reset_index(drop=True)
    return assignments


def build_province_water(assignments: pd.DataFrame) -> pd.DataFrame:
    allocation = pd.read_csv(PROVINCE_ALLOCATION, encoding="utf-8-sig")
    water_only = assignments[
        assignments["architecture"].isin(["IF", "IG"])
    ][
        [
            "industry_code",
            "industry_name",
            "architecture",
            "water_case",
            "facility_scale",
            "scenario_id",
            "site_water_use_l_per_kwh_it",
            "industry_equivalent_annual_it_energy_twh",
            "industry_equivalent_site_water_use_m3",
            "water_result_boundary",
        ]
    ].drop_duplicates()
    expected = 31 * 2 * len(WATER_CASES)
    if len(water_only) != expected:
        raise ValueError(f"Expected {expected} unique IF/IG water rows, found {len(water_only)}")
    province = water_only.merge(
        allocation[
            [
                "scenario_year",
                "industry_code",
                "province",
                "electricity_share_within_industry",
                "allocation_method",
                "source_id",
            ]
        ],
        on="industry_code",
        how="left",
        validate="many_to_many",
    )
    if len(province) != len(water_only) * 31:
        raise ValueError("Province merge did not produce exactly 31 rows per industry-architecture-water case")
    province["province_site_water_use_m3"] = (
        province["industry_equivalent_site_water_use_m3"]
        * province["electricity_share_within_industry"]
    )
    province["spatial_interpretation"] = "frozen_REFERENCE_electricity_share_scenario_not_observed_AI_location"
    province = province.sort_values(
        ["architecture", "water_case", "industry_code", "province"]
    ).reset_index(drop=True)
    return province


def build_architecture_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    grouped = assignments.groupby(["architecture", "water_case", "reuse_case"], sort=True)
    summary = grouped.agg(
        national_annual_it_energy_twh=("industry_equivalent_annual_it_energy_twh", "sum"),
        national_site_water_use_m3=("industry_equivalent_site_water_use_m3", "sum"),
        maximum_single_equivalent_site_water_use_m3=("per_host_site_water_use_m3", "max"),
        national_equivalent_installed_it_capacity_mw=("industry_equivalent_installed_it_capacity_mw", "sum"),
        equivalent_site_count=("equivalent_host_multiplier", "sum"),
    ).reset_index()

    additions = []
    for keys, group in grouped:
        architecture, water_case, reuse_case = keys
        weights = group["equivalent_host_multiplier"]
        additions.append(
            {
                "architecture": architecture,
                "water_case": water_case,
                "reuse_case": reuse_case,
                "equivalent_sites_requiring_new_land": float(
                    weights[group["zero_new_land_conversion"].eq(0)].sum()
                ),
                "equivalent_sites_requiring_new_building_area": float(
                    weights[group["zero_new_building_area"].eq(0)].sum()
                ),
                "equivalent_sites_requiring_new_equipment_space": float(
                    weights[group["zero_new_indoor_outdoor_equipment_space"].eq(0)].sum()
                ),
                "small_industry_architecture_rows": int(group["facility_scale"].eq("small").sum()),
                "medium_industry_architecture_rows": int(group["facility_scale"].eq("medium").sum()),
                "large_industry_architecture_rows": int(group["facility_scale"].eq("large").sum()),
            }
        )
    summary = summary.merge(
        pd.DataFrame(additions),
        on=["architecture", "water_case", "reuse_case"],
        validate="one_to_one",
    )
    summary["water_result_boundary"] = "site_water_use_not_calibrated_withdrawal_or_consumption"
    summary["area_result_boundary"] = "constraint_flags_only_no_square_metre_total"
    summary["noise_result_boundary"] = "equipment_and_operating_condition_screen_only"
    return summary.sort_values(["architecture", "water_case", "reuse_case"]).reset_index(drop=True)


def validate_outputs(
    assignments: pd.DataFrame, province: pd.DataFrame, summary: pd.DataFrame
) -> None:
    if len(assignments) != 31 * len(ARCHITECTURES) * len(WATER_CASES) * len(REUSE_CASES):
        raise ValueError("Assignment coverage failed")
    if len(summary) != len(ARCHITECTURES) * len(WATER_CASES) * len(REUSE_CASES):
        raise ValueError("Architecture summary coverage failed")
    if (assignments["industry_equivalent_site_water_use_m3"] < 0).any():
        raise ValueError("Negative site water use found")
    for (architecture, water_case), group in province.groupby(["architecture", "water_case"]):
        province_total = group["province_site_water_use_m3"].sum()
        assignment_rows = assignments[
            (assignments["architecture"] == architecture)
            & (assignments["water_case"] == water_case)
        ].drop_duplicates(["industry_code", "architecture", "water_case"])
        assignment_total = assignment_rows["industry_equivalent_site_water_use_m3"].sum()
        if abs(province_total - assignment_total) > max(1e-6, abs(assignment_total) * 1e-12):
            raise ValueError(f"Province water allocation failed for {architecture} {water_case}")
    for architecture, group in summary.groupby("architecture"):
        central = group[group["reuse_case"] == "conditional_reuse"].set_index("water_case")
        ordered = central.loc[list(WATER_CASES), "national_site_water_use_m3"]
        if not ordered.is_monotonic_increasing:
            raise ValueError(f"National water cases are not monotonic for {architecture}")


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    assignments = build_assignments()
    province = build_province_water(assignments)
    summary = build_architecture_summary(assignments)
    validate_outputs(assignments, province, summary)

    assignments.to_csv(ASSIGNMENTS_OUTPUT, index=False, encoding="utf-8-sig")
    province.to_csv(PROVINCE_WATER_OUTPUT, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")

    lineage = {
        "generated_on": date.today().isoformat(),
        "model_version": "v0.6.0",
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "inputs": {
            "model_summary_root": str(MODEL_ROOT.relative_to(ROOT)),
            "water_parameters": str(WATER_PARAMETERS.relative_to(ROOT)),
            "water_parameters_sha256": sha256(WATER_PARAMETERS),
            "reuse_parameters": str(REUSE_PARAMETERS.relative_to(ROOT)),
            "reuse_parameters_sha256": sha256(REUSE_PARAMETERS),
            "province_allocation": str(PROVINCE_ALLOCATION.relative_to(ROOT)),
            "province_allocation_sha256": sha256(PROVINCE_ALLOCATION),
        },
        "outputs": {
            "facility_assignments": str(ASSIGNMENTS_OUTPUT.relative_to(ROOT)),
            "province_site_water": str(PROVINCE_WATER_OUTPUT.relative_to(ROOT)),
            "architecture_summary": str(SUMMARY_OUTPUT.relative_to(ROOT)),
        },
        "row_counts": {
            "facility_assignments": len(assignments),
            "province_site_water": len(province),
            "architecture_summary": len(summary),
        },
        "validated": [
            "93 industry-architecture model summaries",
            "three water cases and three reuse cases for every facility scale",
            "water-case monotonicity within each scale and architecture",
            "province allocation conservation for IF and IG",
            "zero new equipment space only when all five reuse constraints pass",
        ],
        "excluded_from_mvp": [
            "electricity-generation water",
            "calibrated withdrawal-versus-consumption split",
            "freshwater source share",
            "scarcity-weighted water",
            "square-metre building and site totals",
            "boundary-noise propagation or exposed population",
            "II_1host province allocation before a candidate data-center location is defined",
        ],
        "interpretation": "evidence-bounded screening scenarios, not observed national facility statistics",
    }
    LINEAGE_OUTPUT.write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(lineage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
