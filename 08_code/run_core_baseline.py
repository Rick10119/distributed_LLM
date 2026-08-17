#!/usr/bin/env python3
"""Optimize the no-AI representative-factory counterfactual."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_code"))

from core.config import load_config, write_resolved_config
from core.data import load_industry_inputs, read_core_grid_energy_prices
from core.io import write_csv, write_json
from core.model import optimize_host
from core.production_load import resolve_site_load_profile
from core.representative_group import read_representative_groups, scenario_scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--hourly-output", type=Path, required=True)
    parser.add_argument("--resolved-config-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(ROOT, args.defaults, args.config)
    groups = read_representative_groups(
        ROOT / config["paths"]["representative_group_report"]
    )
    group = groups[args.industry]
    scale = scenario_scale(group, config["industry_parameter_case"], "IF")
    inputs = load_industry_inputs(config, args.industry)
    site_load, production_load = resolve_site_load_profile(
        root=ROOT,
        config=config,
        industry=args.industry,
        industry_profile_mw=inputs.base_load_mw,
        ai_service_group_share=scale.group_share,
        legacy_load_site_count=scale.group_factory_count,
    )
    grid_prices = read_core_grid_energy_prices(config)
    result = optimize_host(
        config,
        base_load_mw=site_load,
        pv_capacity_factor=inputs.pv_capacity_factor,
        roof_area_m2=inputs.roof_area_proxy_m2,
        grid_energy_price_rmb_per_mwh=grid_prices,
    )
    payload = {
        "model_version": config["model_version"],
        "industry_code": args.industry,
        "industry_name": inputs.industry_name,
        "parameter_case": config["industry_parameter_case"],
        "group_share": scale.group_share,
        "group_factory_count": scale.group_factory_count,
        "ai_service_group_share": scale.group_share,
        "ai_factory_count": scale.group_factory_count,
        "production_load": production_load,
        "roof_area_proxy_m2": inputs.roof_area_proxy_m2,
        "roof_area_case": inputs.roof_area_case,
        "roof_source_naics": inputs.roof_source_naics,
        "roof_mapping_type": inputs.roof_mapping_type,
        "roof_evidence_grade": inputs.roof_evidence_grade,
        "model": result.summary,
    }
    write_json(payload, args.summary_output)
    write_csv(result.hourly, args.hourly_output)
    write_resolved_config(config, args.resolved_config_output)


if __name__ == "__main__":
    main()
