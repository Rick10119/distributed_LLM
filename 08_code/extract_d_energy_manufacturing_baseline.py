"""Extract the C13--C43 electricity baseline from external d_energy.csv.

The source is a provincial reference-scenario energy dataset in gigajoules.
This script keeps the source unchanged, excludes the synthetic ``export``
region, aggregates 31 mainland provincial-level regions, and converts GJ to
TWh.  The explicit English-to-GB/T manufacturing crosswalk prevents fuzzy
industry matching from silently changing the sample.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "02_data" / "external" / "d_energy.csv"
OUTPUT = ROOT / "02_data" / "manufacturing_31sector_electricity_baseline.csv"


INDUSTRY_MAP = {
    "C13": ("农副食品加工业", "processing of food from agricultural products"),
    "C14": ("食品制造业", "manufacture of foods"),
    "C15": ("酒、饮料和精制茶制造业", "manufacture of liquor beverages and refined tea"),
    "C16": ("烟草制品业", "manufacture of tobacco"),
    "C17": ("纺织业", "manufacture of textile"),
    "C18": ("纺织服装、服饰业", "manufacture of textile wearing apparel and accessories"),
    "C19": ("皮革、毛皮、羽毛及其制品和制鞋业", "manufacture of leather fur feather and related products and footwear"),
    "C20": ("木材加工和木、竹、藤、棕、草制品业", "processing of timber manufacture of wood bamboo rattan palm and straw products"),
    "C21": ("家具制造业", "manufacture of furniture"),
    "C22": ("造纸和纸制品业", "manufacture of paper and paper products"),
    "C23": ("印刷和记录媒介复制业", "printing and reproduction of recording media"),
    "C24": ("文教、工美、体育和娱乐用品制造业", "manufacture of articles for culture education arts and crafts sport and entertainment activities"),
    "C25": ("石油、煤炭及其他燃料加工业", "processing of petroleum coal and other fuels"),
    "C26": ("化学原料和化学制品制造业", "manufacture of raw chemical materials and chemical products"),
    "C27": ("医药制造业", "manufacture of medicines"),
    "C28": ("化学纤维制造业", "manufacture of chemical fibers"),
    "C29": ("橡胶和塑料制品业", "manufacture of rubber and plastics products"),
    "C30": ("非金属矿物制品业", "manufacture of non-metallic mineral products"),
    "C31": ("黑色金属冶炼和压延加工业", "smelting and pressing of ferrous metals"),
    "C32": ("有色金属冶炼和压延加工业", "smelting and pressing of non-ferrous metals"),
    "C33": ("金属制品业", "manufacture of metal products"),
    "C34": ("通用设备制造业", "manufacture of general purpose machinery"),
    "C35": ("专用设备制造业", "manufacture of special purpose machinery"),
    "C36": ("汽车制造业", "manufacture of automobiles"),
    "C37": ("铁路、船舶、航空航天和其他运输设备制造业", "manufacture of railway ship aerospace and other transport equipments"),
    "C38": ("电气机械和器材制造业", "manufacture of electrical machinery and apparatus"),
    "C39": ("计算机、通信和其他电子设备制造业", "manufacture of computers communication and other electronic equipment"),
    "C40": ("仪器仪表制造业", "manufacture of measuring instruments and machinery"),
    "C41": ("其他制造业", "other manufacture"),
    "C42": ("废弃资源综合利用业", "utilization of waste resources"),
    "C43": ("金属制品、机械和设备修理业", "repair service of metal products machinery and equipment"),
}


def main() -> None:
    frame = pd.read_csv(SOURCE)
    required = {"FINAL_ENERGY", "PROVINCE", "SCENARIO", "SECTOR", "SUBSECTOR", "UNIT", "YEAR", "VALUE"}
    if set(frame.columns) != required:
        raise ValueError(f"Unexpected columns: {list(frame.columns)}")
    if set(frame["UNIT"]) != {"gigajoule"}:
        raise ValueError("Expected all source values to use gigajoule")
    if set(frame["SCENARIO"]) != {"REFERENCE"}:
        raise ValueError("Expected a single REFERENCE scenario")

    electricity = frame[
        (frame["FINAL_ENERGY"] == "electricity")
        & (frame["PROVINCE"] != "export")
        & (frame["SUBSECTOR"].isin([value[1] for value in INDUSTRY_MAP.values()]))
    ].copy()
    if electricity["PROVINCE"].nunique() != 31:
        raise ValueError("Expected 31 domestic provincial-level regions")
    expected_rows = 31 * 31 * electricity["YEAR"].nunique()
    if len(electricity) != expected_rows:
        raise ValueError(f"Expected {expected_rows} complete rows, found {len(electricity)}")

    grouped = electricity.groupby(["SUBSECTOR", "YEAR"], as_index=False)["VALUE"].sum()
    pivot = grouped.pivot(index="SUBSECTOR", columns="YEAR", values="VALUE")
    rows = []
    for code, (name_cn, name_en) in INDUSTRY_MAP.items():
        values = pivot.loc[name_en]
        row = {
            "industry_code": code,
            "industry_name_cn": name_cn,
            "source_subsector_en": name_en,
            "source_scenario": "REFERENCE",
            "source_unit": "gigajoule",
            "domestic_province_count": 31,
            "electricity_2020_twh": float(values.loc[2020]) / 3.6e6,
            "electricity_2030_twh": float(values.loc[2030]) / 3.6e6,
        }
        row["electricity_2020_2030_growth_pct"] = (
            row["electricity_2030_twh"] / row["electricity_2020_twh"] - 1
        ) * 100
        rows.append(row)

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "source_rows": len(frame),
                "manufacturing_industries": len(rows),
                "domestic_provinces": electricity["PROVINCE"].nunique(),
                "year_range": [int(electricity["YEAR"].min()), int(electricity["YEAR"].max())],
                "manufacturing_electricity_2020_twh": sum(row["electricity_2020_twh"] for row in rows),
                "manufacturing_electricity_2030_twh": sum(row["electricity_2030_twh"] for row in rows),
                "output": str(OUTPUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
