"""Build the mainline land, building, and construction-material screening results.

The analysis consumes the versioned national core-scenario summary and therefore
updates automatically when installed server capacity changes. GFA-based building
shell estimates and top-down hyperscale material checks remain separate methods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import yaml


REQUIRED_SCENARIOS = {"IF", "IG", "II_1host"}
REQUIRED_SPACE_CASES = {
    "SMALL_FACTORY_REUSE",
    "CN_LARGE_EXISTING",
    "CN_LARGE_CAMPUS",
    "CN_LARGE_GREENFIELD",
    "US_LARGE_EXISTING",
    "US_LARGE_CAMPUS",
    "US_LARGE_GREENFIELD",
}
REQUIRED_ARCHETYPES = {"REUSE", "STEEL_FRAME", "RC_FRAME"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(
    national_summary_path: Path,
    space_parameters_path: Path,
    material_parameters_path: Path,
    crosscheck_parameters_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    national = pd.read_csv(national_summary_path, encoding="utf-8-sig")
    space = pd.read_csv(space_parameters_path, encoding="utf-8-sig")
    materials = pd.read_csv(material_parameters_path, encoding="utf-8-sig")
    crosscheck = pd.read_csv(crosscheck_parameters_path, encoding="utf-8-sig")
    return national, space, materials, crosscheck


def derive_common_installed_it_capacity_mw(national: pd.DataFrame) -> float:
    required = {
        "model_version",
        "industry_code",
        "scenario",
        "industry_equivalent_installed_server_groups",
        "server_maximum_wall_power_kw",
    }
    missing = required - set(national.columns)
    if missing:
        raise ValueError(f"National core summary is missing columns: {sorted(missing)}")
    if set(national["scenario"]) != REQUIRED_SCENARIOS:
        raise ValueError("National core summary must contain IF, IG, and II_1host")
    if national.duplicated(["industry_code", "scenario"]).any():
        raise ValueError("National core summary contains duplicate industry-scenario rows")
    if national["industry_code"].nunique() != 31:
        raise ValueError("Land-material mainline analysis requires the 31-industry national run")
    if national["model_version"].nunique() != 1:
        raise ValueError("National core summary mixes model versions")

    concentrated = national[national["scenario"] == "II_1host"].copy()
    capacity = (
        concentrated["industry_equivalent_installed_server_groups"]
        * concentrated["server_maximum_wall_power_kw"]
        / 1000.0
    ).sum()
    if capacity <= 0:
        raise ValueError("Derived common installed IT capacity must be positive")
    return float(capacity)


def validate_parameters(
    space: pd.DataFrame, materials: pd.DataFrame, crosscheck: pd.DataFrame
) -> None:
    if set(space["space_case_id"]) != REQUIRED_SPACE_CASES:
        raise ValueError("Unexpected land-space scenario coverage")
    if space.duplicated("space_case_id").any():
        raise ValueError("Duplicate land-space scenarios")
    if set(materials["archetype_id"]) != REQUIRED_ARCHETYPES:
        raise ValueError("Unexpected construction-material archetype coverage")
    if materials.duplicated("archetype_id").any():
        raise ValueError("Duplicate construction-material archetypes")
    if set(crosscheck["metric_id"]) != {
        "HYPERSCALE_CONCRETE",
        "HYPERSCALE_CONSTRUCTION_STEEL",
    }:
        raise ValueError("Unexpected hyperscale material cross-check coverage")

    for column in ["new_build_fraction", "new_land_conversion_fraction"]:
        if not space[column].between(0, 1).all():
            raise ValueError(f"{column} must lie between zero and one")
    material_columns = [
        "concrete_m3_per_m2_new_gfa",
        "rebar_kg_per_m2_new_gfa",
        "structural_steel_kg_per_m2_new_gfa",
    ]
    if (materials[material_columns] < 0).any().any():
        raise ValueError("Construction-material intensities cannot be negative")
    if not (
        materials.set_index("archetype_id").loc["REUSE", material_columns] == 0
    ).all():
        raise ValueError("The reuse archetype must have zero new structural-shell materials")


def build_space_results(space: pd.DataFrame, capacity_mw: float, model_version: str) -> pd.DataFrame:
    results = space.copy()
    results.insert(1, "model_version", model_version)
    results.insert(5, "common_installed_it_capacity_mw", capacity_mw)

    gfa = pd.to_numeric(results["gfa_intensity_m2_per_mw_it"], errors="coerce")
    site = pd.to_numeric(results["site_intensity_m2_per_mw_it"], errors="coerce")
    results["required_gross_floor_area_m2"] = capacity_mw * gfa
    results["total_site_area_m2"] = capacity_mw * site

    missing_new_gfa = results["new_build_fraction"].gt(0) & gfa.isna()
    missing_new_land = results["new_land_conversion_fraction"].gt(0) & site.isna()
    if missing_new_gfa.any() or missing_new_land.any():
        raise ValueError("Positive new-build or new-land fractions require matching area intensity")

    results["new_gross_floor_area_m2"] = (
        results["required_gross_floor_area_m2"] * results["new_build_fraction"]
    )
    results.loc[results["new_build_fraction"].eq(0), "new_gross_floor_area_m2"] = 0.0
    results["new_land_conversion_m2"] = (
        results["total_site_area_m2"] * results["new_land_conversion_fraction"]
    )
    results.loc[
        results["new_land_conversion_fraction"].eq(0), "new_land_conversion_m2"
    ] = 0.0
    results["building_footprint_m2"] = pd.NA
    results["area_boundary"] = (
        "gross_floor_site_and_new_land_reported_separately_building_footprint_NR"
    )
    return results.sort_values("space_case_id").reset_index(drop=True)


def build_material_results(
    space_results: pd.DataFrame, materials: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    archetypes = materials.set_index("archetype_id")
    for _, case in space_results.iterrows():
        applicable = ["REUSE"] if case["space_case_id"] == "SMALL_FACTORY_REUSE" else [
            "STEEL_FRAME",
            "RC_FRAME",
        ]
        for archetype_id in applicable:
            archetype = archetypes.loc[archetype_id]
            new_gfa = float(case["new_gross_floor_area_m2"])
            concrete = new_gfa * float(archetype["concrete_m3_per_m2_new_gfa"])
            rebar = new_gfa * float(archetype["rebar_kg_per_m2_new_gfa"]) / 1000.0
            structural = (
                new_gfa * float(archetype["structural_steel_kg_per_m2_new_gfa"]) / 1000.0
            )
            rows.append(
                {
                    "model_version": case["model_version"],
                    "space_case_id": case["space_case_id"],
                    "deployment_case": case["deployment_case"],
                    "country": case["country"],
                    "capacity_realization": case["capacity_realization"],
                    "common_installed_it_capacity_mw": case[
                        "common_installed_it_capacity_mw"
                    ],
                    "new_gross_floor_area_m2": new_gfa,
                    "archetype_id": archetype_id,
                    "structural_archetype": archetype["structural_archetype"],
                    "concrete_m3": concrete,
                    "rebar_t": rebar,
                    "structural_steel_t": structural,
                    "total_construction_steel_t": rebar + structural,
                    "cement_t": pd.NA,
                    "cement_status": "NR_requires_project_concrete_mix_design",
                    "method": "new_GFA_times_structural_archetype_intensity",
                    "parameter_status": archetype["parameter_status"],
                    "evidence_ids": archetype["evidence_ids"],
                    "interpretation": archetype["interpretation"],
                }
            )
    return pd.DataFrame(rows).sort_values(["space_case_id", "archetype_id"]).reset_index(
        drop=True
    )


def build_crosscheck_results(
    crosscheck: pd.DataFrame, capacity_mw: float, model_version: str
) -> pd.DataFrame:
    result = crosscheck.copy()
    result.insert(0, "model_version", model_version)
    result.insert(3, "common_installed_it_capacity_mw", capacity_mw)
    for case in ["low", "base", "high"]:
        parameter = pd.to_numeric(result[f"{case}_per_mw_it"], errors="coerce")
        result[f"{case}_total"] = parameter * capacity_mw
    result["method_boundary"] = (
        "facility_or_campus_scale_crosscheck_not_additive_with_GFA_based_materials"
    )
    return result.sort_values("metric_id").reset_index(drop=True)


def validate_results(
    space_results: pd.DataFrame,
    material_results: pd.DataFrame,
    crosscheck_results: pd.DataFrame,
) -> list[str]:
    checks: list[str] = []
    if len(space_results) != 7:
        raise ValueError("Expected seven land-space scenarios")
    if len(material_results) != 13:
        raise ValueError("Expected thirteen compatible space-material scenario rows")
    if len(crosscheck_results) != 2:
        raise ValueError("Expected two top-down material cross-check rows")
    checks.append("seven land-space scenarios and thirteen compatible material scenarios")

    small = space_results.set_index("space_case_id").loc["SMALL_FACTORY_REUSE"]
    if small["new_gross_floor_area_m2"] != 0 or small["new_land_conversion_m2"] != 0:
        raise ValueError("Small-factory reuse must have zero new building and land")
    if pd.notna(small["required_gross_floor_area_m2"]) or pd.notna(small["total_site_area_m2"]):
        raise ValueError("Small-factory total occupied area must remain NR")
    checks.append("small-factory reuse has zero new shell/land and NR total occupied area")

    indexed = space_results.set_index("space_case_id")
    for country in ["CN", "US"]:
        existing = indexed.loc[f"{country}_LARGE_EXISTING"]
        campus = indexed.loc[f"{country}_LARGE_CAMPUS"]
        greenfield = indexed.loc[f"{country}_LARGE_GREENFIELD"]
        if existing["new_gross_floor_area_m2"] != 0 or existing["new_land_conversion_m2"] != 0:
            raise ValueError(f"{country} existing-capacity boundary failed")
        if campus["new_gross_floor_area_m2"] <= 0 or campus["new_land_conversion_m2"] != 0:
            raise ValueError(f"{country} campus-expansion boundary failed")
        if greenfield["new_gross_floor_area_m2"] <= 0:
            raise ValueError(f"{country} greenfield new-building boundary failed")
        if abs(greenfield["new_land_conversion_m2"] - greenfield["total_site_area_m2"]) > 1e-6:
            raise ValueError(f"{country} greenfield land-conversion identity failed")
    checks.append("existing-capacity, campus-expansion, and greenfield boundaries are distinct")

    numeric_materials = material_results[
        ["concrete_m3", "rebar_t", "structural_steel_t", "total_construction_steel_t"]
    ]
    if (numeric_materials < 0).any().any():
        raise ValueError("Negative material quantity found")
    if not (
        material_results["total_construction_steel_t"]
        - material_results["rebar_t"]
        - material_results["structural_steel_t"]
    ).abs().lt(1e-8).all():
        raise ValueError("Construction-steel identity failed")
    checks.append("material quantities are nonnegative and steel components reconcile")

    if not bool(crosscheck_results["method_boundary"].str.contains("not_additive").all()):
        raise ValueError("Top-down cross-check boundary is missing")
    checks.append("top-down hyperscale check is explicitly non-additive with GFA method")
    return checks


def write_findings(
    path: Path,
    space_results: pd.DataFrame,
    material_results: pd.DataFrame,
    crosscheck_results: pd.DataFrame,
) -> None:
    space = space_results.set_index("space_case_id")
    materials = material_results.set_index(["space_case_id", "archetype_id"])
    concrete_cross = crosscheck_results.set_index("metric_id").loc["HYPERSCALE_CONCRETE"]
    steel_cross = crosscheck_results.set_index("metric_id").loc[
        "HYPERSCALE_CONSTRUCTION_STEEL"
    ]
    capacity = float(space_results["common_installed_it_capacity_mw"].iloc[0])

    lines = [
        "# Mainline land, building, and construction-material screening",
        "",
        f"Generated on {date.today().isoformat()} from the national core-scenario outputs.",
        "",
        f"The common comparison capacity is {capacity:,.4f} MW-IT, reconstructed from the II_1host installed server groups rather than hard-coded.",
        "",
        "## Space results",
        "",
        "| Case | Required GFA (m²) | Total site (m²) | New GFA (m²) | New land conversion (m²) |",
        "|---|---:|---:|---:|---:|",
    ]
    for case_id in [
        "SMALL_FACTORY_REUSE",
        "CN_LARGE_EXISTING",
        "CN_LARGE_CAMPUS",
        "CN_LARGE_GREENFIELD",
        "US_LARGE_EXISTING",
        "US_LARGE_CAMPUS",
        "US_LARGE_GREENFIELD",
    ]:
        row = space.loc[case_id]
        fmt = lambda value: "NR" if pd.isna(value) else f"{float(value):,.1f}"
        lines.append(
            f"| {case_id} | {fmt(row['required_gross_floor_area_m2'])} | "
            f"{fmt(row['total_site_area_m2'])} | {fmt(row['new_gross_floor_area_m2'])} | "
            f"{fmt(row['new_land_conversion_m2'])} |"
        )

    lines.extend(
        [
            "",
            "The small-factory reuse baseline has zero new shell and land but retains total occupied indoor area and retrofit materials as NR. Existing capacity, campus expansion, and greenfield are separate marginal-capacity realizations rather than assumed provider portfolio shares.",
            "",
            "## Greenfield building-shell material sensitivities",
            "",
            "| Space proxy | Structure | Concrete (m³) | Rebar (t) | Structural steel (t) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for case_id in ["CN_LARGE_GREENFIELD", "US_LARGE_GREENFIELD"]:
        for archetype_id in ["STEEL_FRAME", "RC_FRAME"]:
            row = materials.loc[(case_id, archetype_id)]
            lines.append(
                f"| {case_id} | {archetype_id} | {row['concrete_m3']:,.1f} | "
                f"{row['rebar_t']:,.1f} | {row['structural_steel_t']:,.1f} |"
            )

    lines.extend(
        [
            "",
            "These are common-structure sensitivities, not observed country averages. Cement remains NR because concrete mix designs and supplementary-cementitious-material shares are unavailable.",
            "",
            "## Separate top-down plausibility check",
            "",
            f"The facility/campus-scale concrete range is {concrete_cross['low_total']:,.1f}–{concrete_cross['high_total']:,.1f} m³, with a base of {concrete_cross['base_total']:,.1f} m³. The construction-steel point check is {steel_cross['base_total']:,.1f} t.",
            "",
            "The top-down and GFA-based methods have different boundaries. They must not be added, averaged, or interpreted as alternative estimates of the same quantity.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> None:
    input_paths = {
        "national_summary": Path(args.national_summary),
        "space_parameters": Path(args.space_parameters),
        "material_parameters": Path(args.material_parameters),
        "crosscheck_parameters": Path(args.crosscheck_parameters),
    }
    national, space, materials, crosscheck = load_inputs(*input_paths.values())
    registry_path = Path(args.scenario_registry)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    footprint = registry["resource_footprint"]
    selected_space = footprint["space"]["selected_case_ids"]
    selected_materials = footprint["materials"]["selected_archetype_ids"]
    space = space[space["space_case_id"].isin(selected_space)].copy()
    materials = materials[materials["archetype_id"].isin(selected_materials)].copy()
    validate_parameters(space, materials, crosscheck)
    capacity = derive_common_installed_it_capacity_mw(national)
    model_version = str(national["model_version"].iloc[0])
    space_results = build_space_results(space, capacity, model_version)
    material_results = build_material_results(space_results, materials)
    crosscheck_results = build_crosscheck_results(crosscheck, capacity, model_version)
    checks = validate_results(space_results, material_results, crosscheck_results)

    outputs = {
        "space": Path(args.space_output),
        "materials": Path(args.material_output),
        "crosscheck": Path(args.crosscheck_output),
        "lineage": Path(args.lineage_output),
        "findings": Path(args.findings_output),
        "done": Path(args.done_output),
    }
    for output in outputs.values():
        output.parent.mkdir(parents=True, exist_ok=True)
    space_results.to_csv(outputs["space"], index=False, encoding="utf-8-sig")
    material_results.to_csv(outputs["materials"], index=False, encoding="utf-8-sig")
    crosscheck_results.to_csv(outputs["crosscheck"], index=False, encoding="utf-8-sig")
    write_findings(outputs["findings"], space_results, material_results, crosscheck_results)

    lineage = {
        "generated_on": date.today().isoformat(),
        "model_version": model_version,
        "generator": "08_code/analyze_land_material_footprint.py",
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in input_paths.items()
        },
        "scenario_registry": {
            "path": str(registry_path),
            "sha256": sha256(registry_path),
            "space_case": footprint["space"]["case"],
            "material_case": footprint["materials"]["case"],
        },
        "outputs": {name: str(path) for name, path in outputs.items() if name != "done"},
        "row_counts": {
            "space_scenarios": len(space_results),
            "material_scenarios": len(material_results),
            "crosscheck_metrics": len(crosscheck_results),
        },
        "common_installed_it_capacity_mw": capacity,
        "validated": checks,
        "boundary": {
            "electricity_generation_land_and_materials": "excluded",
            "building_footprint": "NR",
            "cement": "NR_requires_project_mix_design",
            "retrofit_materials": "NR",
            "topdown_crosscheck": "separate_and_non_additive",
        },
    }
    outputs["lineage"].write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    done = {
        "status": "validated",
        "model_version": model_version,
        "generated_on": date.today().isoformat(),
        "common_installed_it_capacity_mw": capacity,
        "checks": checks,
        "lineage": str(outputs["lineage"]),
    }
    outputs["done"].write_text(
        json.dumps(done, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--national-summary", required=True)
    parser.add_argument("--scenario-registry", required=True)
    parser.add_argument("--space-parameters", required=True)
    parser.add_argument("--material-parameters", required=True)
    parser.add_argument("--crosscheck-parameters", required=True)
    parser.add_argument("--space-output", required=True)
    parser.add_argument("--material-output", required=True)
    parser.add_argument("--crosscheck-output", required=True)
    parser.add_argument("--lineage-output", required=True)
    parser.add_argument("--findings-output", required=True)
    parser.add_argument("--done-output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
