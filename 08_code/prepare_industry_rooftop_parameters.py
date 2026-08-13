#!/usr/bin/env python3
"""Prepare China-industry rooftop proxies from the U.S. MECS method boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SQFT_TO_M2 = 0.09290304


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--mecs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lineage-output", type=Path, required=True)
    args = parser.parse_args()

    crosswalk = pd.read_csv(args.crosswalk, encoding="utf-8-sig")
    mecs = pd.read_csv(args.mecs, encoding="utf-8-sig")
    if len(crosswalk) != 31 or crosswalk["industry_code"].nunique() != 31:
        raise ValueError("Rooftop crosswalk must contain exactly 31 unique industries")
    source = mecs[
        [
            "reference_year",
            "naics_or_group",
            "industry_name",
            "establishments",
            "enclosed_floorspace_million_sqft",
            "floorspace_sqft_per_establishment",
            "quality_flag",
            "source_table",
            "source_url",
        ]
    ].copy()
    source["naics_or_group"] = source["naics_or_group"].astype(int)
    merged = crosswalk.merge(
        source,
        left_on="us_naics_3",
        right_on="naics_or_group",
        how="left",
        validate="many_to_one",
        suffixes=("", "_mecs"),
    )
    if merged["floorspace_sqft_per_establishment"].isna().any():
        missing = merged.loc[
            merged["floorspace_sqft_per_establishment"].isna(), "industry_code"
        ].tolist()
        raise ValueError(f"Missing MECS rooftop proxy for {missing}")
    merged["us_mecs_enclosed_floorspace_m2_per_establishment"] = (
        merged["floorspace_sqft_per_establishment"] * SQFT_TO_M2
    )
    merged["roof_area_proxy_m2"] = merged[
        "us_mecs_enclosed_floorspace_m2_per_establishment"
    ]
    merged["roof_area_case"] = "us_mecs_2022_reference"
    merged["transfer_factor"] = 1.0
    merged["source_method"] = (
        "Namin_et_al_2023_enclosed_floorspace_as_rooftop_proxy_using_US_EIA_MECS"
    )
    merged["country_transfer_status"] = "US_reference_for_China_sensitivity_not_measured_China_roof"
    output_columns = [
        "industry_code",
        "industry_name_cn",
        "roof_area_case",
        "roof_area_proxy_m2",
        "transfer_factor",
        "us_naics_3",
        "us_industry_name",
        "mapping_type",
        "evidence_grade",
        "reference_year",
        "establishments",
        "enclosed_floorspace_million_sqft",
        "floorspace_sqft_per_establishment",
        "us_mecs_enclosed_floorspace_m2_per_establishment",
        "quality_flag",
        "source_table",
        "source_url",
        "source_method",
        "country_transfer_status",
        "mapping_note",
    ]
    output = merged[output_columns].sort_values("industry_code")
    values = output["roof_area_proxy_m2"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prepared roof areas must be finite and positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    lineage = {
        "status": "prepared",
        "parameter_boundary": "US_reference_proxy_for_China_manufacturing",
        "source_paper": "Namin, Eckelman and Isaacs (2023), doi:10.1088/2634-4505/acb5bf",
        "source_data": str(args.mecs),
        "crosswalk": str(args.crosswalk),
        "conversion_sqft_to_m2": SQFT_TO_M2,
        "industries": int(len(output)),
        "minimum_roof_area_m2": float(values.min()),
        "maximum_roof_area_m2": float(values.max()),
        "limitations": [
            "US average enclosed floorspace is not measured Chinese rooftop area",
            "enclosed floorspace may overstate roof correspondence for multistorey buildings",
            "enclosed floorspace omits outdoor production areas in process industries",
            "C42 and C43 use nearest-sector proxies",
        ],
    }
    args.lineage_output.parent.mkdir(parents=True, exist_ok=True)
    args.lineage_output.write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
