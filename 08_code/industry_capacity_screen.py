"""Transparent capacity-only screen for the industry compute-pool case.

This is not a power-flow model. It prints a CSV table so every preliminary
number in the research memo can be reproduced and assumptions can be changed.
"""

from __future__ import annotations


INDUSTRY_IT_PEAK_MW = 50.0
LOCAL_LOCKED_SHARE = 0.30
FACTORIES = 75

LOCAL = {
    "provisioning_factor": 1.40,
    "pue": 1.35,
    "headroom_share": 0.30,
}

SCENARIOS = [
    ("IF", 0, 1.45, 1.35, 0.30),
    ("IG", 5, 1.25, 1.25, 0.15),
    ("IR2", 2, 1.15, 1.22, 0.10),
    ("IR5", 5, 1.18, 1.24, 0.20),
    ("IC", 1, 1.12, 1.20, 0.05),
    ("IH", 3, 1.17, 1.23, 0.25),
]


def required_upgrade(
    it_peak_mw: float,
    provisioning_factor: float,
    pue: float,
    headroom_share: float,
) -> float:
    return it_peak_mw * provisioning_factor * pue * (1.0 - headroom_share)


def main() -> None:
    local_peak = INDUSTRY_IT_PEAK_MW * LOCAL_LOCKED_SHARE
    shared_peak = INDUSTRY_IT_PEAK_MW - local_peak
    common_local_upgrade = required_upgrade(local_peak, **LOCAL)

    print(
        "case_id,total_upgrade_MW,max_single_upgrade_MW,"
        "provisioning_factor,pue,grid_headroom_share"
    )
    for case_id, shared_nodes, factor, pue, headroom in SCENARIOS:
        if case_id == "IF":
            total = required_upgrade(
                INDUSTRY_IT_PEAK_MW, factor, pue, headroom
            )
            max_single = total / FACTORIES
        else:
            shared_upgrade = required_upgrade(
                shared_peak, factor, pue, headroom
            )
            total = common_local_upgrade + shared_upgrade
            max_single = max(
                common_local_upgrade / FACTORIES,
                shared_upgrade / shared_nodes,
            )
        print(
            f"{case_id},{total:.4f},{max_single:.4f},"
            f"{factor:.2f},{pue:.2f},{headroom:.2f}"
        )


if __name__ == "__main__":
    main()
