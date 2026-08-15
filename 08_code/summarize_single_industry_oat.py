#!/usr/bin/env python3
"""Summarize one configured industry's OAT cases against a same-code reference run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def read_one(path: Path) -> pd.Series:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if len(frame) != 1:
        raise ValueError(f"Expected one summary row: {path}")
    return frame.iloc[0]


def metrics(rows: dict[str, pd.Series]) -> dict[str, float]:
    grid = {
        key: float(row["industry_equivalent_incremental_grid_expansion_mw"])
        for key, row in rows.items()
    }
    costs = {
        key: float(row["industry_equivalent_incremental_total_cost_rmb"])
        for key, row in rows.items()
    }
    energy = {
        key: float(row["industry_equivalent_annual_ai_facility_energy_twh"])
        for key, row in rows.items()
    }
    result: dict[str, float] = {}
    for architecture in rows:
        prefix = architecture.lower()
        result[f"{prefix}_grid_expansion_mw"] = grid[architecture]
        result[f"{prefix}_owned_cost_rmb"] = costs[architecture]
        result[f"{prefix}_energy_twh"] = energy[architecture]
    if "IF" in grid and "II_1host" in grid and grid["II_1host"] > 0:
        result["screening_margin_if_vs_industry_node"] = (grid["II_1host"] - grid["IF"]) / grid["II_1host"]
    else:
        result["screening_margin_if_vs_industry_node"] = float("nan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--reference-summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--case-summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--case-configs", type=Path, nargs="+", required=True)
    parser.add_argument("--case-output", type=Path, required=True)
    parser.add_argument("--factor-output", type=Path, required=True)
    parser.add_argument("--done-output", type=Path, required=True)
    args = parser.parse_args()

    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    architectures = list(registry["architectures"])
    case_ids = [
        f"{factor_id}__{case_name}"
        for factor_id, factor in registry["factors"].items()
        for case_name in factor["cases"]
    ]
    expected = len(case_ids) * len(architectures)
    if len(args.case_summaries) != expected or len(args.case_configs) != len(case_ids):
        raise ValueError("Sensitivity inputs do not match the registry case count")
    if len(args.reference_summaries) != len(architectures):
        raise ValueError("Reference summaries do not match the registry architectures")

    reference_rows = {
        architecture: read_one(path)
        for architecture, path in zip(architectures, args.reference_summaries)
    }
    base = metrics(reference_rows)
    configs = {
        yaml.safe_load(path.read_text(encoding="utf-8"))["sensitivity_metadata"]["factor_id"]
        + "__"
        + yaml.safe_load(path.read_text(encoding="utf-8"))["sensitivity_metadata"]["case_name"]:
        yaml.safe_load(path.read_text(encoding="utf-8"))["sensitivity_metadata"]
        for path in args.case_configs
    }
    summary_map: dict[tuple[str, str], pd.Series] = {}
    for path in args.case_summaries:
        parts = path.parts
        case_id = next(value for value in case_ids if value in parts)
        row = read_one(path)
        summary_map[(case_id, str(row["scenario"]))] = row

    case_rows: list[dict[str, object]] = []
    for case_id in case_ids:
        rows = {architecture: summary_map[(case_id, architecture)] for architecture in architectures}
        value = metrics(rows)
        metadata = configs[case_id]
        swings = []
        for field in value:
            if field == "screening_margin_if_vs_industry_node":
                continue
            if value[field] > 0 and base[field] > 0:
                swings.append(abs(np.log(value[field] / base[field])))
        case_rows.append({
            "sensitivity_version": registry["sensitivity_version"],
            "industry_code": registry["industry"],
            "case_id": case_id,
            "factor_id": metadata["factor_id"],
            "factor_label_cn": metadata["factor_label_cn"],
            "case_name": metadata["case_name"],
            "tested_value": metadata["display_value"],
            "baseline_value": metadata["baseline_value"],
            "primary_claims": ";".join(metadata["primary_claims"]),
            **value,
            "max_abs_log_output_swing": max(swings, default=0.0),
            "industry_screen_direction_pass": (
                bool(value["screening_margin_if_vs_industry_node"] > 0)
                if np.isfinite(value["screening_margin_if_vs_industry_node"])
                else True
            ),
            "formal_c2_evaluated": False,
            "cost_claim_evaluated": False,
            "water_claim_evaluated": False,
            "land_claim_evaluated": False,
        })
    cases = pd.DataFrame(case_rows)

    factor_rows: list[dict[str, object]] = []
    base_margin = float(base["screening_margin_if_vs_industry_node"])
    for factor_id, group in cases.groupby("factor_id", sort=False):
        min_margin = float(group["screening_margin_if_vs_industry_node"].min())
        shrink = max(0.0, (base_margin - min_margin) / base_margin) if base_margin > 0 else np.nan
        flip = bool((~group["industry_screen_direction_pass"]).any())
        output_swing = float(group["max_abs_log_output_swing"].max())
        # C38 is a factor-screening industry only.  Its IF-versus-industry-node
        # direction is descriptive and must not determine the formal C2 rating.
        # Rank factors by their maximum physical-output ratio instead.
        if output_swing >= np.log(1.50):
            impact = "high"
        elif output_swing >= np.log(1.20):
            impact = "medium"
        else:
            impact = "low"
        factor_rows.append({
            "factor_id": factor_id,
            "factor_label_cn": group["factor_label_cn"].iloc[0],
            "test_industry": registry["industry"],
            "tested_range": " | ".join(group["tested_value"].astype(str)),
            "baseline_screening_margin_if_vs_industry_node": base_margin,
            "minimum_screening_margin_if_vs_industry_node": min_margin,
            "screening_margin_shrink_fraction": shrink,
            "max_abs_log_output_swing": output_swing,
            "industry_screen_direction_flip": flip,
            "formal_c2_status": "not_evaluated_until_national_cloud_comparison",
            "impact_class": impact,
            "cost_claim_status": "not_evaluated_in_physical_smoke",
            "water_claim_status": "not_applicable_to_smoke_factors",
            "land_claim_status": "not_applicable_to_smoke_factors",
            "single_industry_followup": "joint_corner_or_threshold" if impact == "high" else "retain_oat_result",
            "cross_industry_sensitivity_expansion": "no_by_design",
        })
    factors = pd.DataFrame(factor_rows)

    args.case_output.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(args.case_output, index=False, encoding="utf-8-sig")
    factors.to_csv(args.factor_output, index=False, encoding="utf-8-sig")
    payload = {
        "status": "validated_single_industry_oat",
        "sensitivity_version": registry["sensitivity_version"],
        "case_count": len(cases),
        "factor_count": len(factors),
        "reference_screening_margin_if_vs_industry_node": base_margin,
        "reference_case_id": registry["reference_case"]["case_id"],
        "factor_impact_classes": dict(zip(factors.factor_id, factors.impact_class)),
        "scope": "single-industry physical screen only; no factor is expanded across industries by default",
        "cross_industry_followup": "core scenarios only; explain cost differences without changing sensitivity parameters",
    }
    args.done_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
