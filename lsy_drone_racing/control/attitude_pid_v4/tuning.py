"""Central tuning module.

Defines leg times, speed profiles, per-section PID gains, route geometry
tuning, and active obstacle-clearance parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cascade_pid import PositionPidGains
from .speed_profile import SectorSpeedProfile


@dataclass(frozen=True)
class RouteTuning:
    """Per-route geometry + clearance tuning."""

    # Gate-0 biased crossing center
    gate0_lateral_offset_away_from_start: float = 0.035
    gate0_vertical_offset: float = 0.02

    # Gate-1 shifted entry/exit
    gate1_exit_offset: float = 0.35
    gate1_left_offset_toward_gate0: float = 0.02
    gate1_offset_active: bool = True

    # Gate-2 entry/exit distances
    gate2_entry_offset: float = 0.30
    gate2_minimal_exit_offset: float = 0.08

    # Gate-3 axis distances
    gate3_r_in: float = 0.2
    gate3_r_out: float = 0.4

    # Default gate-axis r_in/r_out
    default_r_in: float = 0.25
    default_r_out: float = 0.30

    # Fixed pole-arc points used by route 1
    gate1_pole_arc_points: tuple[tuple[float, float, float], ...] = (
        (1.08, -0.16, 0.86),
        (1.38, 0.12, 1.08),
    )

    # Fixed shaping points. Sector-3 mid waypoint moved north (y -0.42 ->
    # -0.30) so the curve between mid and the (-0.2, -0.75) gate-3 entry
    # passes ~0.31 m from obstacle 3 (was ~0.21 m); leaves headroom for the
    # ±0.15 m obstacle position randomization.
    start_waypoint: tuple[float, float, float] = (-1.5, 0.75, 0.05)
    route2_mid_waypoint: tuple[float, float, float] = (0.0, 0.25, 1.0)
    route3_mid_waypoint: tuple[float, float, float] = (-0.45, -0.30, 0.95)

    # Obstacle clearance — per sector (4 entries)
    clearance_triggers: tuple[float, float, float, float] = (0.15, 0.15, 0.15, 0.10)
    clearance_margins: tuple[float, float, float, float] = (0.03, 0.03, 0.03, 0.02)
    clearance_push_max: tuple[float, float, float, float] = (0.12, 0.12, 0.12, 0.06)


@dataclass(frozen=True)
class QualificationTuning:
    """Top-level tuning bundle consumed by QualificationController."""

    leg_times: tuple[float, float, float, float]
    speed_profiles: tuple[SectorSpeedProfile, ...]
    section_gains: tuple[PositionPidGains, ...]
    route: RouteTuning = field(default_factory=RouteTuning)
    time_limit: float = 25.0
    feedforward_scale: float = 0.75
    lateral_acc_limit: float = 16.0
    replan_gate_delta: float = 0.005
    replan_horizontal_distance: float = 0.7
    replan_obstacle_delta: float = 0.01
    replan_obstacle_distance: float = 1.5

    @property
    def leg_start_times(self) -> tuple[float, ...]:
        """Return the cumulative start time of each leg derived from the leg durations."""
        starts = [0.0]
        for t in self.leg_times[:-1]:
            starts.append(starts[-1] + float(t))
        return tuple(starts)


# Base per-section PID gains (legacy kp/ki/kd/i_clamp form).
_SECTION_BASE_GAINS = (
    # section 0
    {
        "kp": (0.60, 0.60, 1.65),
        "ki": (0.05, 0.05, 0.05),
        "kd": (0.35, 0.35, 0.50),
        "i_clamp": (1.5, 1.5, 0.4),
    },
    # section 1
    {
        "kp": (0.65, 0.65, 1.65),
        "ki": (0.045, 0.045, 0.05),
        "kd": (0.55, 0.55, 0.50),
        "i_clamp": (1.5, 1.5, 0.4),
    },
    # section 2
    {
        "kp": (0.65, 0.65, 1.55),
        "ki": (0.045, 0.045, 0.05),
        "kd": (0.45, 0.45, 0.50),
        "i_clamp": (1.5, 1.5, 0.4),
    },
    # section 3
    {
        "kp": (0.65, 0.65, 1.65),
        "ki": (0.045, 0.045, 0.05),
        "kd": (0.30, 0.30, 0.50),
        "i_clamp": (1.5, 1.5, 0.4),
    },
)


_NOMINAL_LEG_TIMES = (3.85, 2.5, 3.5, 2.25)
_GLOBAL_TIME_SCALE = 0.84
# Sector 3 was the second-largest failure source (180-deg turn through tight
# r_in=0.2). Stretching its time scale from 0.65 -> 0.82 gives the turn more
# headroom; nominal sector-3 time goes 1.23 s -> 1.55 s.
_LEG_TIME_SCALES = (0.52, 0.68, 0.65, 0.82)

_SPEED_PROFILES = (
    SectorSpeedProfile(1.0, 1.0, 1.4),
    SectorSpeedProfile(1.0, 1.8, 1.4),
    SectorSpeedProfile(1.0, 1.0, 1.0),
    SectorSpeedProfile(1.0, 1.0, 1.0),
)


def _build_section_gains() -> tuple[PositionPidGains, ...]:
    return tuple(
        PositionPidGains.from_xyz(
            np.array(spec["kp"]),
            np.array(spec["ki"]),
            np.array(spec["kd"]),
            np.array(spec["i_clamp"]),
        )
        for spec in _SECTION_BASE_GAINS
    )


def gate1_offset_tuning() -> QualificationTuning:
    """Build the qualification tuning preset with a gate-1 offset adjustment."""
    leg_times = tuple(
        _NOMINAL_LEG_TIMES[i] * _GLOBAL_TIME_SCALE * _LEG_TIME_SCALES[i] for i in range(4)
    )
    return QualificationTuning(
        leg_times=leg_times,
        speed_profiles=_SPEED_PROFILES,
        section_gains=_build_section_gains(),
        route=RouteTuning(gate1_offset_active=True),
    )
