#!/usr/bin/env python3
"""Build model-ready task service from documented source-side scenarios.

The legacy 2030 task table already embeds enterprise adoption and a simple
per-adopter growth path. This transform first reconstructs the corresponding
2023 task baseline, then applies the task-specific total-service growth factors
from the bottom-up research report. A separate global intensity multiplier is
therefore neither needed nor permitted downstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


TASKS = {"office", "agent", "vision", "maintenance", "scheduling", "simulation"}
CASES = ("low", "base", "high")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--industry-baseline", type=Path, required=True)
    parser.add_argument("--task-templates", type=Path, required=True)
    parser.add_argument("--growth-scenarios", type=Path, required=True)
    parser.add_argument("--template-fallbacks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--lineage-output", type=Path, required=True)
    args = parser.parse_args()

    industry = pd.read_csv(args.industry_baseline, encoding="utf-8-sig")
    templates = pd.read_csv(args.task_templates, encoding="utf-8-sig")
    growth = pd.read_csv(args.growth_scenarios, encoding="utf-8-sig")
    fallbacks = pd.read_csv(args.template_fallbacks, encoding="utf-8-sig")

    require_columns(
        industry,
        {
            "industry_code",
            "industry_name_cn",
            "ai_using_firms_2023",
            "adopting_firms_2030",
            "task_template",
            "sector_l20eq_gpu_h_day_2030",
        },
        "industry baseline",
    )
    require_columns(
        templates,
        {"industry_code", "year", "task_id", "sector_l20eq_gpu_h_day"},
        "task templates",
    )
    require_columns(
        growth,
        {
            "task_id",
            "task_name_cn",
            "legacy_per_adopter_growth_2023_2030",
            "growth_low",
            "growth_base",
            "growth_high",
            "evidence_status",
            "source",
        },
        "growth scenarios",
    )
    require_columns(
        fallbacks,
        {"requested_template", "resolved_template", "evidence_status"},
        "template fallbacks",
    )
    if len(industry) != 31 or industry["industry_code"].nunique() != 31:
        raise ValueError("industry baseline must contain exactly 31 industries")
    if set(growth["task_id"]) != TASKS or len(growth) != len(TASKS):
        raise ValueError("growth scenarios must contain exactly the six core tasks")

    template_rows = templates[templates["year"] == 2030].copy()
    fallback_map = dict(
        zip(fallbacks["requested_template"], fallbacks["resolved_template"])
    )
    growth_by_task = growth.set_index("task_id")
    records: list[dict[str, object]] = []

    for item in industry.itertuples(index=False):
        requested_template = str(item.task_template)
        resolved_template = fallback_map.get(requested_template, requested_template)
        selected = template_rows[template_rows["industry_code"] == resolved_template]
        if set(selected["task_id"]) != TASKS:
            raise ValueError(
                f"template {requested_template}->{resolved_template} does not cover six tasks"
            )
        total_template_service = float(selected["sector_l20eq_gpu_h_day"].sum())
        if total_template_service <= 0:
            raise ValueError(f"template {resolved_template} has non-positive service")
        adoption_growth = float(item.adopting_firms_2030) / float(
            item.ai_using_firms_2023
        )
        if adoption_growth <= 0:
            raise ValueError(f"{item.industry_code} has invalid adoption growth")

        for task in selected.itertuples(index=False):
            task_id = str(task.task_id)
            settings = growth_by_task.loc[task_id]
            task_share = float(task.sector_l20eq_gpu_h_day) / total_template_service
            legacy_task_2030 = (
                float(item.sector_l20eq_gpu_h_day_2030) * task_share
            )
            legacy_total_growth = adoption_growth * float(
                settings.legacy_per_adopter_growth_2023_2030
            )
            reconstructed_2023 = legacy_task_2030 / legacy_total_growth
            for case in CASES:
                task_growth = float(settings[f"growth_{case}"])
                records.append(
                    {
                        "industry_code": item.industry_code,
                        "industry_name_cn": item.industry_name_cn,
                        "year": 2030,
                        "parameter_case": case,
                        "task_id": task_id,
                        "task_name_cn": settings.task_name_cn,
                        "effective_service_units_day": reconstructed_2023
                        * task_growth,
                        "reconstructed_2023_service_units_day": reconstructed_2023,
                        "total_service_growth_2023_2030": task_growth,
                        "legacy_2030_task_service_units_day": legacy_task_2030,
                        "legacy_total_growth_2023_2030": legacy_total_growth,
                        "template_task_share": task_share,
                        "requested_task_template": requested_template,
                        "resolved_task_template": resolved_template,
                        "service_unit": "reference_L20_equivalent_accelerator_hour",
                        "evidence_status": settings.evidence_status,
                        "source": settings.source,
                    }
                )

    output = pd.DataFrame.from_records(records).sort_values(
        ["parameter_case", "industry_code", "task_id"]
    )
    expected_rows = 31 * len(TASKS) * len(CASES)
    if len(output) != expected_rows:
        raise ValueError(f"expected {expected_rows} output rows, found {len(output)}")

    summary = (
        output.groupby(
            ["parameter_case", "industry_code", "industry_name_cn"], as_index=False
        )
        .agg(
            effective_service_units_day=("effective_service_units_day", "sum"),
            reconstructed_2023_service_units_day=(
                "reconstructed_2023_service_units_day",
                "sum",
            ),
        )
        .sort_values(["parameter_case", "industry_code"])
    )
    summary["aggregate_growth_2023_2030"] = (
        summary["effective_service_units_day"]
        / summary["reconstructed_2023_service_units_day"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.lineage_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    summary.to_csv(args.summary_output, index=False, encoding="utf-8-sig")
    lineage = {
        "schema_version": 1,
        "output": str(args.output),
        "formula": (
            "service_2030 = legacy_task_service_2030 / "
            "[(adopting_firms_2030 / ai_using_firms_2023) * "
            "legacy_per_adopter_growth] * task_total_service_growth"
        ),
        "inputs": {
            str(path): {"sha256": sha256(path)}
            for path in (
                args.industry_baseline,
                args.task_templates,
                args.growth_scenarios,
                args.template_fallbacks,
            )
        },
        "rows": len(output),
        "parameter_cases": list(CASES),
        "tasks": sorted(TASKS),
        "notes": [
            "Growth factors are joint research scenarios, not observed forecasts.",
            "Four legacy auto-calibrated templates use documented structural fallbacks.",
            "Hardware efficiency and PUE are excluded from effective service.",
        ],
    }
    args.lineage_output.write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
