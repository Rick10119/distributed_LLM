#!/usr/bin/env python3
"""Aggregate fine-grained inference evidence into auditable coarse model cases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


EXPECTED_CASES = {"efficient", "base", "conservative"}
EXPECTED_FACTORS = {
    "prefill_decode_mix",
    "context_length",
    "batch_size",
    "quantization",
    "serving_engine",
    "hardware_utilization",
}
REFERENCE_ACCELERATOR_H_PER_SERVICE_UNIT = 1.0 / 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lineage-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    if set(frame["efficiency_case"]) != EXPECTED_CASES:
        raise ValueError("Compute-efficiency cases must be efficient, base, and conservative")
    rows: list[dict[str, object]] = []
    for case, group in frame.groupby("efficiency_case", sort=False):
        if set(group["factor"]) != EXPECTED_FACTORS or len(group) != len(EXPECTED_FACTORS):
            raise ValueError(f"{case} must contain each of the six fine-grained factors once")
        weights = group["aggregation_weight"].astype(float)
        multipliers = group["unit_compute_multiplier"].astype(float)
        if abs(float(weights.sum()) - 1.0) > 1e-9 or (multipliers <= 0).any():
            raise ValueError(f"Invalid weights or multipliers for {case}")
        aggregate = math.exp(float((weights * multipliers.map(math.log)).sum()))
        sources = sorted(
            {
                source.strip()
                for value in group["evidence_source"].astype(str)
                for source in value.split(";")
            }
        )
        rows.append(
            {
                "efficiency_case": case,
                "accelerator_h_per_service_unit": REFERENCE_ACCELERATOR_H_PER_SERVICE_UNIT * aggregate,
                "relative_compute_to_base": aggregate,
                "aggregation_method": "weighted_geometric_mean_of_six_correlated_factors",
                "factor_count": len(group),
                "source_summary": "; ".join(sources),
                "evidence_status": (
                    "normalized_reference" if case == "base" else "evidence_bounded_sensitivity_not_target_hardware_benchmark"
                ),
            }
        )
    output = pd.DataFrame(rows)
    order = {"efficient": 0, "base": 1, "conservative": 2}
    output = output.sort_values("efficiency_case", key=lambda values: values.map(order))
    base_value = float(output.loc[output["efficiency_case"] == "base", "accelerator_h_per_service_unit"].iloc[0])
    if abs(base_value - REFERENCE_ACCELERATOR_H_PER_SERVICE_UNIT) > 1e-12:
        raise ValueError("Base compute-efficiency case must preserve the calibrated reference")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    lineage = {
        "status": "validated",
        "source": str(args.input),
        "output": str(args.output),
        "cases": output["efficiency_case"].tolist(),
        "minimum_accelerator_h_per_service_unit": float(output["accelerator_h_per_service_unit"].min()),
        "maximum_accelerator_h_per_service_unit": float(output["accelerator_h_per_service_unit"].max()),
        "base_preserves_external_calibration": True,
        "interpretation": "Sensitivity envelope aggregated from server-side evidence; not a direct L20 benchmark or a forecast of 2030 hardware.",
    }
    args.lineage_output.parent.mkdir(parents=True, exist_ok=True)
    args.lineage_output.write_text(json.dumps(lineage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
