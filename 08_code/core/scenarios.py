"""Read and validate the mainline scenario-selection registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def read_scenario_registry(path: Path) -> dict[str, Any]:
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("Scenario registry must be a YAML mapping")
    countries = registry.get("countries", {})
    enabled = countries.get("enabled")
    if not isinstance(enabled, list) or not enabled or not set(enabled) <= {"china", "us"}:
        raise ValueError("countries.enabled must be a non-empty subset of [china, us]")
    hardware = registry.get("compute_hardware", {})
    if hardware.get("active_routing_case") not in {
        "gpu_only", "practice_routed", "cpu_heavy", "evidence_core_candidate"
    }:
        raise ValueError("Unsupported compute-hardware routing case")
    for field in ("local_cpu_price_case", "china_cloud_capacity_price_case"):
        if hardware.get(field, "base") not in {"low", "base", "high"}:
            raise ValueError(f"compute_hardware.{field} must be low, base, or high")
    footprint = registry.get("resource_footprint", {})
    for section in ("water", "space", "materials", "spatial_water"):
        if not isinstance(footprint.get(section), dict):
            raise ValueError(f"Missing resource_footprint.{section} scenario section")
    return registry


def require_matching_routing_case(registry: dict[str, Any], routing_config: dict[str, Any]) -> str:
    selected = str(registry["compute_hardware"]["active_routing_case"])
    if selected not in routing_config.get("routing_cases", {}):
        raise ValueError(f"Routing case {selected} is absent from the routing parameter file")
    configured = str(routing_config.get("active_core_routing_case"))
    if configured != selected:
        raise ValueError(
            "Scenario registry and routing parameter file disagree: "
            f"{selected} != {configured}"
        )
    return selected
