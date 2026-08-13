#!/usr/bin/env python3
"""Prepare a deliberately simple, auditable model-state parameter bundle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--hardware-matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parameters = pd.read_csv(args.parameters, encoding="utf-8-sig").set_index("parameter_id")
    hardware = pd.read_csv(args.hardware_matrix, encoding="utf-8-sig")
    selected = hardware[
        (hardware["model_name"] == "Qwen2.5-32B-Instruct")
        & (hardware["precision"] == "INT4/AWQ")
        & (hardware["context_length"] == 8192)
        & (hardware["concurrent_requests"] == 4)
    ]
    if len(selected) != 1:
        raise ValueError("Expected one Qwen2.5-32B INT4/AWQ 8k x 4 hardware row")
    row = selected.iloc[0]

    def base(parameter_id: str) -> float:
        return float(parameters.loc[parameter_id, "base"])

    accelerators_per_server = 2
    replicas = 1
    recommended_gpus_per_replica = int(row["recommended_l20_gpu_count"])
    minimum_groups = math.ceil(replicas * recommended_gpus_per_replica / accelerators_per_server)
    payload = {
        "status": "validated_for_simple_scenario_use",
        "parameter_case": "base",
        "model_name": str(row["model_name"]),
        "model_version": str(row["model_version"]),
        "precision": str(row["precision"]),
        "context_length": int(row["context_length"]),
        "concurrent_requests": int(row["concurrent_requests"]),
        "required_model_replicas": replicas,
        "vram_gb_per_replica": float(row["total_vram_gb"]),
        "recommended_l20_gpus_per_replica": recommended_gpus_per_replica,
        "minimum_server_groups_for_model_state": minimum_groups,
        "model_storage_required_gb_per_deployment": base("ST001"),
        "model_storage_copy_equivalents": base("ST002"),
        "bundled_storage_gb_per_server_group": base("ST006") * 1000.0,
        "incremental_storage_capex_rmb_per_server_group": base("ST007"),
        "storage_lifetime_years": base("ST005"),
        "initialization_accelerator_h_per_version": base("LC020"),
        "major_versions_per_year": base("LC004"),
        "operations_fte_per_deployment": base("LC013"),
        "loaded_cost_rmb_per_fte_year": base("LC012"),
        "source_parameter_ids": ["LC004", "LC012", "LC013", "LC020", "ST001", "ST002", "ST005", "ST006", "ST007"],
        "evidence_boundary": "Official weights and bundled storage support capacity; initialization and operations are scenario proxies, not observed manufacturing averages.",
        "excluded_details": ["per_request_kv_cache_scheduling", "cold_start_dispatch", "multi_model_replica_assignment", "rag_project_cost"],
    }
    if payload["model_storage_required_gb_per_deployment"] > payload["bundled_storage_gb_per_server_group"] * minimum_groups:
        raise ValueError("Selected model state does not fit bundled server storage")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
