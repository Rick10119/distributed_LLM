"""Local server installed-capacity planning rules."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def arrival_time_required_server_groups(
    rigid_service_units: np.ndarray,
    flexible_releases: Iterable[tuple[int, float]],
    accelerator_h_per_service_unit: float,
    accelerators_per_server_group: float,
) -> float:
    """Compute the unshifted hourly peak capacity directly from task arrivals."""
    service = np.asarray(rigid_service_units, dtype=float).copy()
    for release_hour, amount_service_units in flexible_releases:
        service[int(release_hour) % len(service)] += float(amount_service_units)
    return (
        float(np.max(service, initial=0.0))
        * float(accelerator_h_per_service_unit)
        / float(accelerators_per_server_group)
    )


def average_required_server_groups(
    total_compute_h: float,
    horizon_hours: int,
    service_capacity_per_server_group_h_per_h: float,
) -> float:
    """Return server groups needed to serve the horizon workload at its mean rate."""
    compute = float(total_compute_h)
    hours = int(horizon_hours)
    capacity = float(service_capacity_per_server_group_h_per_h)
    if not math.isfinite(compute) or compute < 0.0:
        raise ValueError("Total compute must be finite and non-negative")
    if hours <= 0 or not math.isfinite(capacity) or capacity <= 0.0:
        raise ValueError("Horizon and server-group capacity must be positive")
    return compute / (hours * capacity)


def local_installed_capacity_floor(
    required_server_groups: float,
    planning_headroom_fraction: float,
    n_plus_spare_server_groups: int,
) -> float:
    """Return max(average-demand headroom, continuous N+k capacity)."""
    required = float(required_server_groups)
    fraction = float(planning_headroom_fraction)
    spare = int(n_plus_spare_server_groups)
    if not math.isfinite(required) or required < 0.0:
        raise ValueError("Required server groups must be finite and non-negative")
    if not 0.0 <= fraction < 1.0 or spare < 0:
        raise ValueError("Invalid local installed-capacity reserve parameters")
    # Integer deployment cases round this lower bound through the installed
    # server decision itself. Continuous group cases must not inherit an
    # artificial whole-server rounding step.
    return max(required * (1.0 + fraction), required + spare)
