"""Integrity checks for the manufacturing load-archetype prototype."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = pd.read_csv(ROOT / "02_data" / "manufacturing_load_dataset_manifest.csv", encoding="utf-8-sig")
    crosswalk = pd.read_csv(ROOT / "02_data" / "manufacturing_load_curve_source_crosswalk.csv", encoding="utf-8-sig")
    profiles = pd.read_csv(ROOT / "05_results" / "manufacturing_load_archetype_hourly_profiles.csv", encoding="utf-8-sig")
    quality = pd.read_csv(ROOT / "05_results" / "manufacturing_load_archetype_quality.csv", encoding="utf-8-sig")

    assert len(manifest) == 10
    assert "mismatch" not in set(manifest["checksum_status"])
    for relative in manifest["relative_path"]:
        assert (ROOT / "02_data" / "raw_load_profiles" / relative).is_file()

    assert len(crosswalk) == 31
    assert crosswalk["china_code"].nunique() == 31
    assert crosswalk["primary_archetype"].nunique() == 6
    assert set(crosswalk["curve_evidence"]) == {"direct_EWELD", "partial_EWELD", "archetype_proxy"}

    main = profiles[profiles["source_role"] == "EWELD_main"]
    assert main["archetype"].nunique() == 6
    assert set(main["day_type"]) == {"weekday", "weekend"}
    counts = main.groupby(["archetype", "day_type"])["hour"].nunique()
    assert (counts == 24).all()
    assert profiles["normalized_load"].notna().all()
    assert (profiles["normalized_load"] >= 0).all()

    assert len(quality) == 6
    assert (quality["usable_eweld_users"] > 0).all()
    assert ((quality["weekday_load_factor"] > 0) & (quality["weekday_load_factor"] <= 1)).all()

    figure_names = [
        "manufacturing_load_archetypes_weekday.svg",
        "manufacturing_load_archetypes_weekday.png",
        "manufacturing_31sector_curve_coverage.svg",
        "manufacturing_31sector_curve_coverage.png",
    ]
    for name in figure_names:
        path = ROOT / "05_results" / "figures" / name
        assert path.is_file() and path.stat().st_size > 1000

    lines = [
        "Manufacturing load archetype validation: PASS",
        f"Downloaded files checked: {len(manifest)}",
        f"China manufacturing sectors mapped: {len(crosswalk)}",
        f"Main archetypes: {main['archetype'].nunique()}",
        f"EWELD main profile rows: {len(main)}",
        f"Evidence coverage: {crosswalk['curve_evidence'].value_counts().to_dict()}",
        f"EWELD usable users by archetype: {dict(zip(quality['archetype'], quality['usable_eweld_users']))}",
    ]
    output = "\n".join(lines) + "\n"
    (ROOT / "05_results" / "manufacturing_load_archetype_validation.txt").write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
