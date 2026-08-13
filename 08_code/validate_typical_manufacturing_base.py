"""Deterministic checks for the typical manufacturing base case."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    case_input = rows(ROOT / "04_cases" / "typical_machinery_manufacturing_case.csv")
    results = rows(ROOT / "05_results" / "typical_manufacturing_base_results.csv")
    profiles = rows(ROOT / "05_results" / "typical_manufacturing_hourly_profiles.csv")
    assert len(results) == 3
    assert len(profiles) == 72
    assert len(case_input) == 24
    assert abs(sum(float(row["llm_calls"]) for row in case_input) - 11_360.0) < 1e-6
    assert abs(sum(float(row["input_tokens"]) for row in case_input) - 43_040_000.0) < 1e-6
    assert abs(sum(float(row["output_tokens"]) for row in case_input) - 3_116_000.0) < 1e-6
    assert abs(sum(float(row["l20_gpu_compute_h"]) for row in case_input) - 33.2363) < 1e-6
    index = {row["mode"]: row for row in results}
    assert set(index) == {"local", "cloud", "hybrid_50"}

    for row in results:
        assert abs(float(row["daily_l20_gpu_compute_h"]) - 33.2363) < 1e-6
        assert abs(float(row["peak_l20_gpu_compute_h_per_hour"]) - 2.2424) < 1e-6
        assert float(row["enterprise_grid_expansion_kw"]) == 0.0
        assert float(row["annual_enterprise_direct_best_rmb"]) >= 0.0
        assert float(row["annual_social_cost_rmb"]) >= 0.0
        assert float(row["baseline_peak_no_der_kw"]) == 1000.0
        assert float(row["connection_capacity_kw"]) == 1250.0

    assert int(index["local"]["local_servers_2xl20"]) == 2
    assert int(index["cloud"]["local_servers_2xl20"]) == 0
    assert int(index["hybrid_50"]["local_servers_2xl20"]) == 1
    assert float(index["local"]["cloud_grid_expansion_kw"]) == 0.0
    assert float(index["cloud"]["cloud_grid_expansion_kw"]) > float(
        index["hybrid_50"]["cloud_grid_expansion_kw"]
    ) > 0.0

    direct = [
        float(index[mode]["annual_enterprise_direct_best_rmb"])
        for mode in ("local", "hybrid_50", "cloud")
    ]
    social = [
        float(index[mode]["annual_social_cost_rmb"])
        for mode in ("cloud", "hybrid_50", "local")
    ]
    assert direct[0] < direct[1] < direct[2]
    assert social[0] < social[1] < social[2]

    local_profiles = [row for row in profiles if row["mode"] == "local"]
    assert len(local_profiles) == 24
    battery_power = [float(row["battery_kw_positive_discharge"]) for row in local_profiles]
    battery_soc = [float(row["battery_soc_kwh"]) for row in local_profiles]
    assert max(abs(value) for value in battery_power) <= 250.0 + 1e-6
    assert min(battery_soc) >= -1e-6
    assert max(battery_soc) <= 500.0 + 1e-6
    assert abs(battery_soc[-1] - 250.0) < 1e-4

    for name in (
        "typical-manufacturing-mechanism.svg",
        "typical-manufacturing-costs.svg",
    ):
        assert (ROOT / "05_results" / "figures" / name).exists()

    print("typical_manufacturing_base_validation_passed")


if __name__ == "__main__":
    main()
