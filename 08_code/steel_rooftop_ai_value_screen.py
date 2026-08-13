"""Screen steel-factory rooftop PV support and avoided grid capacity investment.

All monetary values are undiscounted capacity-investment sensitivities at
alpha=1, not confirmed project cancellations. Rooftop PV nameplate is converted
to AC before it is compared with facility demand.
"""

from __future__ import annotations

import argparse


FACTORIES = 75
ROOF_PROXY_M2 = 73_257.5
DC_AC_RATIO = 1.20
LOCAL_PUE = 1.35
INDUSTRY_IT_PEAK_MW = 50.0
LOCAL_PROVISIONING_FACTOR = 1.45
NO_PV_GRID_UPGRADE_MW = 68.5125

ROOF_CASES = [
    ("C20", 0.128, 0.20),
    ("B20", 0.198, 0.20),
    ("B40", 0.198, 0.40),
    ("B100", 0.198, 1.00),
]


def hosting_rows() -> list[tuple[object, ...]]:
    required_it_per_factory = (
        INDUSTRY_IT_PEAK_MW * LOCAL_PROVISIONING_FACTOR / FACTORIES
    )
    rows = []
    for case_id, density, realization in ROOF_CASES:
        pv_dc_factory_mwp = ROOF_PROXY_M2 * density * realization / 1000.0
        pv_dc_total_mwp = pv_dc_factory_mwp * FACTORIES
        pv_ac_total_mw = pv_dc_total_mwp / DC_AC_RATIO
        solar_aligned_it_total_mw = pv_ac_total_mw / LOCAL_PUE

        annual_it_supported = {}
        for yield_kwh_kwp in (1000, 1300, 1600):
            annual_it_supported[yield_kwh_kwp] = (
                pv_dc_total_mwp
                * yield_kwh_kwp
                / (8760.0 * LOCAL_PUE)
            )

        rows.append(
            (
                case_id,
                density,
                realization,
                pv_dc_factory_mwp,
                pv_dc_total_mwp,
                solar_aligned_it_total_mw / FACTORIES,
                solar_aligned_it_total_mw,
                required_it_per_factory,
                solar_aligned_it_total_mw / INDUSTRY_IT_PEAK_MW,
                annual_it_supported[1000],
                annual_it_supported[1300],
                annual_it_supported[1600],
            )
        )
    return rows


def grid_rows() -> list[tuple[object, ...]]:
    base20_pv_dc_total_mwp = (
        ROOF_PROXY_M2 * 0.198 * 0.20 * FACTORIES / 1000.0
    )
    pv_ac_total_mw = base20_pv_dc_total_mwp / DC_AC_RATIO
    rows = []
    for critical_hour_fraction in (0.0, 0.05, 0.10, 0.20, 1.0):
        effective_pv_mw = pv_ac_total_mw * critical_hour_fraction
        avoided_mw = min(NO_PV_GRID_UPGRADE_MW, effective_pv_mw)
        remaining_mw = NO_PV_GRID_UPGRADE_MW - avoided_mw
        rows.append(
            (
                critical_hour_fraction,
                effective_pv_mw,
                avoided_mw,
                remaining_mw,
                avoided_mw * 2.0,
                avoided_mw * 5.0,
                avoided_mw * 10.0,
            )
        )
    return rows


def print_hosting() -> None:
    print(
        "roof_case_id,technical_density_kWp_m2,pv_realization_ratio,"
        "pv_limit_per_factory_MWp,pv_limit_75_factories_MWp,"
        "solar_aligned_it_per_factory_MW,solar_aligned_it_total_MW,"
        "required_distributed_it_per_factory_MW,"
        "solar_aligned_it_to_industry_peak_ratio,"
        "annual_average_it_supported_yield1000_MW,"
        "annual_average_it_supported_yield1300_MW,"
        "annual_average_it_supported_yield1600_MW"
    )
    for row in hosting_rows():
        print(
            f"{row[0]},{row[1]:.3f},{row[2]:.2f},"
            + ",".join(f"{value:.4f}" for value in row[3:])
        )


def print_grid() -> None:
    print(
        "critical_hour_pv_ac_fraction,effective_pv_at_critical_hour_MW,"
        "avoided_grid_upgrade_MW,remaining_grid_upgrade_MW,"
        "estimated_avoided_capacity_investment_alpha1_at_2mRMB_per_MW_mRMB,"
        "estimated_avoided_capacity_investment_alpha1_at_5mRMB_per_MW_mRMB,"
        "estimated_avoided_capacity_investment_alpha1_at_10mRMB_per_MW_mRMB"
    )
    for row in grid_rows():
        print(
            f"{row[0]:.2f}," + ",".join(f"{value:.4f}" for value in row[1:])
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", choices=("hosting", "grid"))
    args = parser.parse_args()
    if args.table == "hosting":
        print_hosting()
    else:
        print_grid()


if __name__ == "__main__":
    main()
