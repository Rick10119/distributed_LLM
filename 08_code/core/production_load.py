"""Resolve production-load profiles independently from AI-service scaling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CALIBRATED_MODE = "calibrated_registry"
LEGACY_MODE = "legacy_industry_electricity_share"
VALID_MODES = {CALIBRATED_MODE, LEGACY_MODE}


def _power_weights(raw: object, count: int, industry: str) -> tuple[list[float], str]:
    if count < 1:
        raise ValueError(f"{industry}: production-load site count must be positive")
    if raw is None or pd.isna(raw) or str(raw).strip().lower() in {"", "equal"}:
        return [1.0 / count] * count, "equal_site_power_proxy"
    values = np.asarray([float(value) for value in str(raw).split(";")], dtype=float)
    if len(values) != count or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(
            f"{industry}: load_site_power_weights must contain {count} positive semicolon-separated values"
        )
    values = values / values.sum()
    return values.tolist(), "calibrated_site_power_weights"


def _rooted(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def production_load_mode(config: dict[str, Any], industry: str) -> str:
    selection = config.get("production_load", {})
    overrides = selection.get("mode_by_industry", {})
    mode = str(overrides.get(industry, selection.get("default_mode", "")))
    if mode not in VALID_MODES:
        raise ValueError(
            f"{industry}: production-load mode must be explicitly one of {sorted(VALID_MODES)}"
        )
    return mode


def _registry_row(root: Path, config: dict[str, Any], industry: str) -> dict[str, Any]:
    registry_path = config.get("production_load", {}).get("registry_path")
    if not registry_path:
        raise ValueError("production_load.registry_path is required for calibrated loads")
    frame = pd.read_csv(_rooted(root, registry_path), encoding="utf-8-sig")
    selected = frame.loc[frame["industry_code"].astype(str).eq(industry)]
    if len(selected) != 1:
        raise ValueError(f"{industry}: calibrated mode requires exactly one registry row")
    row = selected.iloc[0].to_dict()
    if str(row["status"]) != "approved":
        raise ValueError(f"{industry}: production-load calibration is not approved")
    return row


def resolve_site_mean_load_mw(
    *,
    root: Path,
    config: dict[str, Any],
    industry: str,
    industry_mean_load_mw: float,
    ai_service_group_share: float,
    legacy_load_site_count: int,
) -> tuple[float, dict[str, Any]]:
    """Return one production site's mean load and complete boundary metadata."""
    mode = production_load_mode(config, industry)
    if mode == LEGACY_MODE:
        load_site_count = int(legacy_load_site_count)
        if load_site_count < 1:
            raise ValueError(f"{industry}: legacy production-load site count must be positive")
        group_mean_load_mw = float(industry_mean_load_mw) * float(ai_service_group_share)
        value = group_mean_load_mw / load_site_count
        site_power_weights, weight_source = _power_weights(None, load_site_count, industry)
        return value, {
            "mode": mode,
            "boundary_id": "legacy_industry_electricity_times_AI_share_divided_by_production_load_site_count",
            "method": "historical_compatibility_only",
            "industry_mean_load_mw": float(industry_mean_load_mw),
            "ai_service_group_share": float(ai_service_group_share),
            "production_activity_share": float(ai_service_group_share),
            "production_activity_share_source": "AI_service_group_share_compatibility_proxy",
            "load_site_count": load_site_count,
            "load_site_power_weights": site_power_weights,
            "load_site_power_weight_source": weight_source,
            "representative_group_mean_load_mw": group_mean_load_mw,
            "representative_site_mean_load_mw": value,
            "formal_national_pool_eligible": False,
        }

    row = _registry_row(root, config, industry)
    if str(row["method"]) != "activity_times_integrated_production_electricity_intensity":
        raise ValueError(f"{industry}: unsupported calibrated production-load method")
    positive = [
        "top_groups_activity_units_per_year", "top_group_count",
        "top_groups_combined_activity_share", "production_activity_share",
        "integrated_production_energy_intensity_mwh_per_unit",
        "benchmark_electricity_gwh", "benchmark_total_production_energy_gwh",
        "load_site_count",
    ]
    if any(float(row[field]) <= 0 for field in positive):
        raise ValueError(f"{industry}: calibrated production-load inputs must be positive")
    if float(row["top_groups_combined_activity_share"]) > 1:
        raise ValueError(f"{industry}: top-groups activity share exceeds one")
    if float(row["benchmark_electricity_gwh"]) > float(row["benchmark_total_production_energy_gwh"]):
        raise ValueError(f"{industry}: electricity exceeds total production energy")

    industry_activity = (
        float(row["top_groups_activity_units_per_year"])
        / float(row["top_groups_combined_activity_share"])
    )
    production_share = float(row["production_activity_share"])
    group_activity = industry_activity * production_share
    electricity_share = (
        float(row["benchmark_electricity_gwh"])
        / float(row["benchmark_total_production_energy_gwh"])
    )
    electricity_intensity = (
        float(row["integrated_production_energy_intensity_mwh_per_unit"])
        * electricity_share
    )
    group_electricity_twh = group_activity * electricity_intensity / 1e6
    load_site_count = int(row["load_site_count"])
    site_power_weights, weight_source = _power_weights(
        row.get("load_site_power_weights"), load_site_count, industry
    )
    value = group_electricity_twh * 1e6 / 8760.0 / load_site_count
    return value, {
        "mode": mode,
        "boundary_id": str(row["boundary_id"]),
        "method": str(row["method"]),
        "status": str(row["status"]),
        "activity_boundary": str(row["activity_boundary"]),
        "excluded_activity": str(row["excluded_activity"]),
        "activity_reference_year": int(row["activity_reference_year"]),
        "energy_benchmark_year": int(row["energy_benchmark_year"]),
        "top_groups_activity_units_per_year": float(row["top_groups_activity_units_per_year"]),
        "top_group_count": int(row["top_group_count"]),
        "top_groups_combined_activity_share": float(row["top_groups_combined_activity_share"]),
        "industry_activity_units_per_year": industry_activity,
        "production_activity_share": production_share,
        "representative_group_activity_units_per_year": group_activity,
        "load_site_count": load_site_count,
        "load_site_power_weights": site_power_weights,
        "load_site_power_weight_source": weight_source,
        "ai_service_group_share": float(ai_service_group_share),
        "integrated_production_energy_intensity_mwh_per_unit": float(row["integrated_production_energy_intensity_mwh_per_unit"]),
        "benchmark_electricity_share": electricity_share,
        "integrated_production_electricity_intensity_mwh_per_unit": electricity_intensity,
        "representative_group_annual_electricity_twh": group_electricity_twh,
        "representative_group_mean_load_mw": group_electricity_twh * 1e6 / 8760.0,
        "representative_site_annual_electricity_twh": group_electricity_twh / load_site_count,
        "representative_site_mean_load_mw": value,
        "scenario_interpretation": str(row["scenario_interpretation"]),
        "activity_source_url": str(row["activity_source_url"]),
        "energy_source_url": str(row["energy_source_url"]),
        "formal_national_pool_eligible": True,
    }


def scale_profile_to_site_mean(
    profile_mw: np.ndarray,
    site_mean_load_mw: float,
) -> np.ndarray:
    values = np.asarray(profile_mw, dtype=float)
    mean = float(values.mean())
    if not np.isfinite(values).all() or mean <= 0 or site_mean_load_mw <= 0:
        raise ValueError("Production-load profile and target mean must be finite and positive")
    return values / mean * float(site_mean_load_mw)


def resolve_site_load_profile(
    *,
    root: Path,
    config: dict[str, Any],
    industry: str,
    industry_profile_mw: np.ndarray,
    ai_service_group_share: float,
    legacy_load_site_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    mean, metadata = resolve_site_mean_load_mw(
        root=root,
        config=config,
        industry=industry,
        industry_mean_load_mw=float(np.asarray(industry_profile_mw).mean()),
        ai_service_group_share=ai_service_group_share,
        legacy_load_site_count=legacy_load_site_count,
    )
    return scale_profile_to_site_mean(industry_profile_mw, mean), metadata
