"""Validate a single-industry enterprise direct-cost comparison."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "05_results"


def main() -> None:
    result = pd.read_csv(
        RESULTS / "c36_enterprise_direct_cost_summary.csv", encoding="utf-8-sig"
    ).set_index("mode")
    assert set(result.index) == {
        "local_purchase_annualized",
        "cloud_reserved_monthly",
        "cloud_ondemand",
    }
    local = result.loc["local_purchase_annualized"]
    reserved = result.loc["cloud_reserved_monthly"]
    ondemand = result.loc["cloud_ondemand"]
    components = [
        "annualized_local_compute_rmb",
        "incremental_flat_energy_rmb",
        "incremental_maximum_demand_rmb",
        "incremental_grid_capacity_rmb",
        "incremental_pv_rmb",
        "incremental_battery_rmb",
    ]
    assert abs(float(local[components].sum()) - float(local["annual_enterprise_direct_cost_rmb"])) < 10.0
    assert int(reserved["owned_or_contracted_2xl20_groups"]) == 43337
    assert float(ondemand["billed_2xl20_instance_h_year"]) > 0
    assert float(reserved["annual_enterprise_direct_cost_rmb"]) > float(local["annual_enterprise_direct_cost_rmb"])
    assert float(ondemand["annual_enterprise_direct_cost_rmb"]) > float(reserved["annual_enterprise_direct_cost_rmb"])
    assert (result["direct_cost_rmb_per_accelerator_h"] > 0).all()
    assert result["network_and_integration_cost_status"].eq("not_included").all()
    print("C36 enterprise direct-cost validation passed")


if __name__ == "__main__":
    main()
