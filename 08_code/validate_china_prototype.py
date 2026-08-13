"""Deterministic validation checks for the China minimum prototype."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    full = rows(ROOT / "05_results" / "china_prototype_24_scenarios.csv")
    sensitivity = rows(ROOT / "05_results" / "china_prototype_sensitivity.csv")
    thresholds = rows(ROOT / "05_results" / "china_prototype_thresholds.csv")
    two_way = rows(ROOT / "05_results" / "china_prototype_two_way_sensitivity.csv")
    assert len(full) == 24
    assert len({row["scenario_id"] for row in full}) == 24
    assert len(sensitivity) == 100
    assert len(thresholds) == 6
    assert len(two_way) == 330

    for row in full:
        assert float(row["total_grid_expansion_kw"]) >= 0
        assert float(row["annual_enterprise_direct_best_rmb"]) >= 0
        assert float(row["annual_social_cost_existing_der_rmb"]) >= 0
        assert float(row["annual_social_cost_if_new_der_rmb"]) >= float(
            row["annual_social_cost_existing_der_rmb"]
        )
        if row["mode"] == "local":
            assert float(row["cloud_physical_equivalent_servers"]) == 0
        if row["mode"] == "cloud":
            assert int(row["local_servers"]) == 0

    index = {
        (r["case"], r["mode"], r["access_state"], r["der_state"]): r
        for r in full
    }
    for case in ("steel", "office"):
        for mode in ("local", "cloud", "hybrid_50"):
            for access in ("tight", "spare"):
                no_der = index[(case, mode, access, "no_der")]
                with_der = index[(case, mode, access, "existing_der")]
                assert float(with_der["total_grid_expansion_kw"]) <= float(
                    no_der["total_grid_expansion_kw"]
                ) + 1e-6

    for case in ("steel", "office"):
        price_rows = [
            r
            for r in sensitivity
            if r["case"] == case
            and r["dimension"] == "cloud_price_multiplier"
            and r["mode"] == "cloud"
        ]
        price_rows.sort(key=lambda r: float(r["parameter_value"]))
        social = [float(r["annual_social_cost_existing_der_rmb"]) for r in price_rows]
        direct = [float(r["annual_enterprise_direct_best_rmb"]) for r in price_rows]
        assert max(social) - min(social) < 1e-5
        assert all(a < b for a, b in zip(direct, direct[1:]))

        headroom_rows = [
            r
            for r in sensitivity
            if r["case"] == case
            and r["dimension"] == "connection_headroom_fraction"
            and r["mode"] == "local"
        ]
        headroom_rows.sort(key=lambda r: float(r["parameter_value"]))
        expansion = [float(r["total_grid_expansion_kw"]) for r in headroom_rows]
        assert all(a + 1e-6 >= b for a, b in zip(expansion, expansion[1:]))

        threshold_index = {
            row["threshold"]: float(row["value"])
            for row in thresholds
            if row["case"] == case
        }
        assert 0.15 < threshold_index["local_utilization_enterprise_switch"] < 0.95
        assert 0.15 < threshold_index["local_utilization_social_switch"] < 0.95
        assert threshold_index["local_utilization_enterprise_switch"] < threshold_index["local_utilization_social_switch"]
        assert 0.25 < threshold_index["cloud_price_multiplier_enterprise_switch"] < 1.5

        hybrid_rows = sorted(
            [r for r in sensitivity if r["case"] == case and r["dimension"] == "hybrid_local_share"],
            key=lambda r: float(r["parameter_value"]),
        )
        direct = [float(r["annual_enterprise_direct_best_rmb"]) for r in hybrid_rows]
        social = [float(r["annual_social_cost_existing_der_rmb"]) for r in hybrid_rows]
        grid = [float(r["total_grid_expansion_kw"]) for r in hybrid_rows]
        assert all(a + 1e-6 >= b for a, b in zip(direct, direct[1:]))
        assert all(a <= b + 1e-6 for a, b in zip(social, social[1:]))
        assert all(a + 1e-6 >= b for a, b in zip(grid, grid[1:]))

    for name in (
        "direct-cost-breakdown.svg",
        "social-cost-breakdown.svg",
        "der-capacity-effect.svg",
        "load-matching-mechanism.svg",
        "utilization-switch.svg",
        "hybrid-tradeoff.svg",
        "two-way-choice-map.svg",
    ):
        assert (ROOT / "05_results" / "figures" / name).exists()

    print("china_prototype_validation_passed")


if __name__ == "__main__":
    main()
