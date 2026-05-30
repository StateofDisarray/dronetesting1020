"""Speed-profile-driven knot scheduling for sector splines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class SectorSpeedProfile:
    start: float
    mid: float
    end: float


_MIN_MULT = 1e-3


def schedule_knots(
    t_start: float, t_end: float, waypoints: NDArray[np.floating], profile: SectorSpeedProfile
) -> NDArray[np.floating]:
    """Allocate sector duration nonuniformly across waypoint intervals."""
    wps = np.asarray(waypoints, dtype=np.float64)
    if wps.ndim != 2:
        raise ValueError(f"waypoints must be 2-D, got ndim={wps.ndim}")
    n = wps.shape[0]
    if n == 0:
        raise ValueError("waypoints must contain at least one point")
    if n == 1:
        return np.array([float(t_start)], dtype=np.float64)
    duration = float(t_end) - float(t_start)
    if duration <= 0.0:
        raise ValueError(f"duration must be positive, got {duration}")
    n_seg = n - 1
    centers = (np.arange(n_seg) + 0.5) / n_seg
    progress_pts = np.array([0.0, 0.5, 1.0])
    mults = np.array([profile.start, profile.mid, profile.end], dtype=np.float64)
    seg_mult = np.interp(centers, progress_pts, mults)
    seg_mult = np.clip(seg_mult, _MIN_MULT, None)
    raw_durations = 1.0 / seg_mult
    raw_durations *= duration / raw_durations.sum()
    knots = np.empty(n, dtype=np.float64)
    knots[0] = float(t_start)
    knots[1:] = float(t_start) + np.cumsum(raw_durations)
    return knots
