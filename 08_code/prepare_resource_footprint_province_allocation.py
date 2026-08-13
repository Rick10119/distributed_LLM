"""Build province-level manufacturing AI allocation weights for resource-footprint research."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

from extract_d_energy_manufacturing_baseline import INDUSTRY_MAP


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "02_data" / "external" / "d_energy.csv"
OUTPUT_DIR = ROOT / "02_data" / "processed" / "resource_footprint"
OUTPUT = OUTPUT_DIR / "province_industry_ai_allocation.csv"
DIAGNOSTICS = OUTPUT_DIR / "province_industry_ai_allocation_diagnostics.csv"
LINEAGE = OUTPUT_DIR / "province_industry_ai_allocation.lineage.json"
SCENARIO_YEAR = 2030
SOURCE_ID = "d_energy_REFERENCE_2030"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_allocation(source: Path = SOURCE, scenario_year: int = SCENARIO_YEAR) -> pd.DataFrame:
    frame = pd.read_csv(source)
    required = {
        "FINAL_ENERGY",
        "PROVINCE",
        "SCENARIO",
        "SECTOR",
        "SUBSECTOR",
        "UNIT",
        "YEAR",
        "VALUE",
    }
    if set(frame.columns) != required:
        raise ValueError(f"Unexpected source columns: {list(frame.columns)}")

    crosswalk = {
        name_en: {"industry_code": code, "industry_name_cn": name_cn}
        for code, (name_cn, name_en) in INDUSTRY_MAP.items()
    }
    selected = frame[
        (frame["FINAL_ENERGY"] == "electricity")
        & (frame["SCENARIO"] == "REFERENCE")
        & (frame["PROVINCE"] != "export")
        & (frame["YEAR"] == scenario_year)
        & (frame["SUBSECTOR"].isin(crosswalk))
    ].copy()

    if set(selected["UNIT"]) != {"gigajoule"}:
        raise ValueError("Expected selected electricity values to use gigajoule")
    if selected["PROVINCE"].nunique() != 31:
        raise ValueError("Expected 31 domestic provincial-level regions")
    if selected["SUBSECTOR"].nunique() != 31:
        raise ValueError("Expected 31 manufacturing divisions")
    if len(selected) != 31 * 31:
        raise ValueError(f"Expected 961 industry-province rows, found {len(selected)}")
    if selected.duplicated(["SUBSECTOR", "PROVINCE"]).any():
        raise ValueError("Duplicate industry-province rows found")
    if (selected["VALUE"] < 0).any():
        raise ValueError("Negative industry-province electricity values found")

    selected["industry_code"] = selected["SUBSECTOR"].map(
        lambda value: crosswalk[value]["industry_code"]
    )
    selected["industry_name_cn"] = selected["SUBSECTOR"].map(
        lambda value: crosswalk[value]["industry_name_cn"]
    )
    selected["industry_electricity_mwh"] = selected["VALUE"] / 3.6
    totals = selected.groupby("industry_code")["industry_electricity_mwh"].transform("sum")
    if (totals <= 0).any():
        raise ValueError("Every manufacturing division must have positive national electricity")
    selected["electricity_share_within_industry"] = selected["industry_electricity_mwh"] / totals
    selected["allocation_method"] = "within_industry_province_electricity_share"
    selected["structure_correction"] = "none"
    selected["zero_value_flag"] = selected["VALUE"].eq(0)
    selected["evidence_level"] = "scenario_model_anchor_not_enterprise_observation"
    selected["source_id"] = SOURCE_ID

    output = selected[
        [
            "industry_code",
            "industry_name_cn",
            "PROVINCE",
            "SUBSECTOR",
            "industry_electricity_mwh",
            "electricity_share_within_industry",
            "allocation_method",
            "structure_correction",
            "zero_value_flag",
            "evidence_level",
            "source_id",
        ]
    ].rename(
        columns={
            "PROVINCE": "province",
            "SUBSECTOR": "source_subsector_en",
        }
    )
    output.insert(0, "scenario_year", scenario_year)
    output = output.sort_values(["industry_code", "province"]).reset_index(drop=True)

    share_sums = output.groupby("industry_code")["electricity_share_within_industry"].sum()
    if not ((share_sums - 1.0).abs() < 1e-12).all():
        raise ValueError("Province allocation shares do not sum to one within every industry")
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    allocation = build_allocation()
    allocation.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    profile_groups: dict[tuple[float, ...], list[str]] = {}
    for industry_code, group in allocation.sort_values("province").groupby("industry_code"):
        profile = tuple(group["electricity_share_within_industry"].round(14))
        profile_groups.setdefault(profile, []).append(industry_code)
    group_lookup = {
        industry_code: ";".join(sorted(industry_codes))
        for industry_codes in profile_groups.values()
        for industry_code in industry_codes
    }

    diagnostics_rows = []
    for (industry_code, industry_name), group in allocation.groupby(
        ["industry_code", "industry_name_cn"]
    ):
        ranked = group.sort_values("electricity_share_within_industry", ascending=False)
        diagnostics_rows.append(
            {
                "scenario_year": SCENARIO_YEAR,
                "industry_code": industry_code,
                "industry_name_cn": industry_name,
                "top_province": ranked.iloc[0]["province"],
                "top_province_share": ranked.iloc[0]["electricity_share_within_industry"],
                "top3_province_share": ranked.head(3)["electricity_share_within_industry"].sum(),
                "province_share_hhi": (group["electricity_share_within_industry"] ** 2).sum(),
                "zero_value_province_count": int(group["zero_value_flag"].sum()),
                "identical_profile_group": group_lookup[industry_code],
                "identical_profile_group_size": len(group_lookup[industry_code].split(";")),
                "diagnostic_status": "recorded_not_current_calibration_task",
            }
        )
    diagnostics = pd.DataFrame(diagnostics_rows).sort_values("industry_code")
    diagnostics.to_csv(DIAGNOSTICS, index=False, encoding="utf-8-sig")

    zero_rows = allocation[allocation["zero_value_flag"]]
    lineage = {
        "generated_on": date.today().isoformat(),
        "scenario_year": SCENARIO_YEAR,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "output": str(OUTPUT.relative_to(ROOT)),
        "diagnostics": str(DIAGNOSTICS.relative_to(ROOT)),
        "rows": len(allocation),
        "manufacturing_industries": int(allocation["industry_code"].nunique()),
        "provinces": int(allocation["province"].nunique()),
        "zero_value_rows": len(zero_rows),
        "identical_profile_groups": [
            industry_codes
            for industry_codes in profile_groups.values()
            if len(industry_codes) > 1
        ],
        "share_validation": "each industry sums to 1 within absolute tolerance 1e-12",
        "interpretation": (
            "REFERENCE-scenario province allocation anchor, not observed enterprise AI adoption "
            "or an empirical factory-location sample"
        ),
        "allocation_decision": (
            "Frozen by user decision as the accepted REFERENCE spatial scenario. Record structural "
            "features for interpretation but do not recalibrate, replace, or add weight sensitivity."
        ),
    }
    LINEAGE.write_text(json.dumps(lineage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lineage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
