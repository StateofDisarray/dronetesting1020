"""Sector route + reference construction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from .geometry import DEFAULT_OBSTACLES, gate_axis_points, gate_x_axis, gate_y_axis
from .speed_profile import SectorSpeedProfile, schedule_knots
from .tuning import RouteTuning

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

ROUTE_OVERRIDE_FILE = Path("logs/qualification_route_waypoints.json")
UNCHANGED_OVERRIDE_TOL = 1e-8
MAX_AVOID_DEPTH = 3
_CLEARANCE_SAMPLES = 40
SECTOR_OBSTACLE_INDEX = (0, 0, 1, 3)


def _signed_y_axis_toward(
    gate_pos: "NDArray[np.floating]",
    gate_rpy: "NDArray[np.floating]",
    target_pos: "NDArray[np.floating]",
) -> "NDArray[np.floating]":
    """Return gate-y axis (horizontal) signed so it points toward ``target_pos``."""
    y = gate_y_axis(gate_rpy)
    delta = (np.asarray(target_pos, dtype=np.float64) - np.asarray(gate_pos, dtype=np.float64))[:2]
    sign = 1.0 if float(np.dot(delta, y[:2])) >= 0.0 else -1.0
    return sign * y


def _signed_y_axis_away(
    gate_pos: "NDArray[np.floating]",
    gate_rpy: "NDArray[np.floating]",
    source_pos: "NDArray[np.floating]",
) -> "NDArray[np.floating]":
    """Return gate-y axis (horizontal) signed so it points away from ``source_pos``."""
    return _signed_y_axis_toward(
        gate_pos, gate_rpy, 2.0 * np.asarray(gate_pos) - np.asarray(source_pos)
    )


def build_route_waypoints(
    route_idx: int,
    gate_pos: "NDArray[np.floating]",
    gate_rpy: "NDArray[np.floating]",
    tuning: RouteTuning,
    extra: "NDArray[np.floating] | None" = None,
) -> "NDArray[np.floating]":
    """Construct the 3-D waypoints for one sector."""
    gp = np.asarray(gate_pos, dtype=np.float64)
    gr = np.asarray(gate_rpy, dtype=np.float64)
    if route_idx < 0 or route_idx > 3:
        raise ValueError(f"route_idx must be in [0, 3], got {route_idx}")

    if route_idx == 0:
        start = np.array(tuning.start_waypoint, dtype=np.float64)
        y_away = _signed_y_axis_away(gp[0], gr[0], start)
        gate0_biased = gp[0] + tuning.gate0_lateral_offset_away_from_start * y_away
        gate0_biased[2] += tuning.gate0_vertical_offset
        _, exit_pt = gate_axis_points(
            gp[0], gr[0], r_in=tuning.default_r_in, r_out=tuning.default_r_out
        )
        wps = [start, gate0_biased, exit_pt]
    elif route_idx == 1:
        _, gate0_exit = gate_axis_points(
            gp[0], gr[0], r_in=tuning.default_r_in, r_out=tuning.default_r_out
        )
        gate1_x = gate_x_axis(gr[1])
        y_toward_g0 = _signed_y_axis_toward(gp[1], gr[1], gp[0])
        entry_base, _ = gate_axis_points(
            gp[1], gr[1], r_in=tuning.default_r_in, r_out=tuning.default_r_out
        )
        if tuning.gate1_offset_active:
            shift = tuning.gate1_left_offset_toward_gate0 * y_toward_g0
            shifted_entry = entry_base + shift
            shifted_exit = gp[1] + tuning.gate1_exit_offset * gate1_x + shift
        else:
            shifted_entry = entry_base
            shifted_exit = gp[1] + tuning.gate1_exit_offset * gate1_x
        pole_arc = [np.array(p, dtype=np.float64) for p in tuning.gate1_pole_arc_points]
        wps = [gate0_exit, *pole_arc, shifted_entry, shifted_exit]
    elif route_idx == 2:
        gate1_x = gate_x_axis(gr[1])
        if tuning.gate1_offset_active:
            shift = tuning.gate1_left_offset_toward_gate0 * _signed_y_axis_toward(
                gp[1], gr[1], gp[0]
            )
            sector_start = gp[1] + tuning.gate1_exit_offset * gate1_x + shift
        else:
            sector_start = gp[1] + tuning.gate1_exit_offset * gate1_x
        mid = np.array(tuning.route2_mid_waypoint, dtype=np.float64)
        gate2_x = gate_x_axis(gr[2])
        entry = gp[2] - tuning.gate2_entry_offset * gate2_x
        center = gp[2].copy()
        exit_pt = gp[2] + tuning.gate2_minimal_exit_offset * gate2_x
        wps = [sector_start, mid, entry, center, exit_pt]
    elif route_idx == 3:
        gate2_x = gate_x_axis(gr[2])
        sector_start = gp[2] + tuning.gate2_minimal_exit_offset * gate2_x
        mid = np.array(tuning.route3_mid_waypoint, dtype=np.float64)
        gate3_x = gate_x_axis(gr[3])
        entry = gp[3] - tuning.gate3_r_in * gate3_x
        center = gp[3].copy()
        exit_pt = gp[3] + tuning.gate3_r_out * gate3_x
        wps = [sector_start, mid, entry, center, exit_pt]

    arr = np.stack(wps, axis=0)
    if extra is not None:
        # Insert avoidance waypoint at the location of the closest interior segment midpoint.
        ex = np.asarray(extra, dtype=np.float64).reshape(3)
        # Place it before the segment whose midpoint is closest in XY.
        seg_mids = 0.5 * (arr[:-1, :2] + arr[1:, :2])
        d2 = np.sum((seg_mids - ex[:2]) ** 2, axis=1)
        insert_at = int(np.argmin(d2)) + 1
        arr = np.insert(arr, insert_at, ex, axis=0)
    return arr


def _load_override_table() -> "list[np.ndarray] | None":
    if not ROUTE_OVERRIDE_FILE.exists():
        return None
    try:
        with open(ROUTE_OVERRIDE_FILE, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("route override unreadable: %s", exc)
        return None
    raw = (
        data.get("qualification_route_points_xy")
        or data.get("qualification_route_lengths")
        or data.get("waypoints_xy")
    )
    if not raw:
        return None
    try:
        return [np.asarray(r, dtype=np.float64) for r in raw]
    except (TypeError, ValueError) as exc:
        logger.warning("route override malformed: %s", exc)
        return None


def _apply_override(
    waypoints: "NDArray[np.floating]", override_xy: "NDArray[np.floating]"
) -> "NDArray[np.floating]":
    """Apply an XY override array to a waypoint array."""
    override_xy = np.asarray(override_xy, dtype=np.float64)
    if override_xy.ndim != 2 or override_xy.shape[1] != 2:
        return waypoints
    n_def = waypoints.shape[0]
    n_ov = override_xy.shape[0]
    out = waypoints.copy()
    if n_ov == n_def:
        # Mask unchanged points (within tolerance of default XY) so defaults stay active there.
        diff = np.linalg.norm(override_xy - waypoints[:, :2], axis=1)
        for i in range(n_def):
            ov = override_xy[i]
            if np.all(np.isfinite(ov)) and diff[i] > UNCHANGED_OVERRIDE_TOL:
                out[i, :2] = ov
        return out
    # Length mismatch: linearly interpolate Z over XY arc length.
    arc = np.zeros(n_ov)
    arc[1:] = np.cumsum(np.linalg.norm(np.diff(override_xy, axis=0), axis=1))
    total = arc[-1] if arc[-1] > 1e-9 else 1.0
    z_start, z_end = float(waypoints[0, 2]), float(waypoints[-1, 2])
    z_interp = z_start + (z_end - z_start) * (arc / total)
    return np.column_stack([override_xy, z_interp])


def _build_interpolator(
    route_idx: int, knots: "NDArray[np.floating]", waypoints: "NDArray[np.floating]"
):
    if route_idx <= 1:
        return CubicSpline(knots, waypoints, bc_type="natural")
    return PchipInterpolator(knots, waypoints)


def _check_clearance(
    reference,
    route_idx: int,
    t_start: float,
    t_end: float,
    tuning: RouteTuning,
    obstacles: "NDArray[np.floating]",
) -> "NDArray[np.floating] | None":
    """If the reference clips the sector's obstacle, return one push waypoint, else None."""
    obs_idx = SECTOR_OBSTACLE_INDEX[route_idx]
    if obs_idx >= obstacles.shape[0]:
        return None
    obs_xy = obstacles[obs_idx, :2]
    trigger = tuning.clearance_triggers[route_idx]
    margin = tuning.clearance_margins[route_idx]
    push_max = tuning.clearance_push_max[route_idx]
    ts = np.linspace(t_start, t_end, _CLEARANCE_SAMPLES)
    samples = reference(ts)
    deltas = samples[:, :2] - obs_xy[None, :]
    dists = np.linalg.norm(deltas, axis=1)
    j = int(np.argmin(dists))
    d_min = float(dists[j])
    if d_min >= trigger:
        return None
    push = min(max(trigger + margin - d_min, 0.0), push_max)
    direction = deltas[j]
    n = float(np.linalg.norm(direction))
    if n < 1e-6:
        direction = np.array([1.0, 0.0])
    else:
        direction = direction / n
    pushed_xy = samples[j, :2] + push * direction
    return np.array([pushed_xy[0], pushed_xy[1], samples[j, 2]], dtype=np.float64)


def build_reference_curve(
    route_idx: int,
    gate_pos: "NDArray[np.floating]",
    gate_rpy: "NDArray[np.floating]",
    t_start: float,
    duration: float,
    tuning: RouteTuning,
    profile: SectorSpeedProfile,
    obstacles: "NDArray[np.floating] | None" = None,
    extra: "NDArray[np.floating] | None" = None,
    overrides: "Sequence[np.ndarray] | None" = None,
    depth: int = 0,
):
    """Build the (interpolator, t_end) pair for one sector reference."""
    if obstacles is None:
        obstacles = DEFAULT_OBSTACLES
    t_end = float(t_start) + float(duration)
    waypoints = build_route_waypoints(route_idx, gate_pos, gate_rpy, tuning, extra=extra)
    if extra is None and overrides is not None and 0 <= route_idx < len(overrides):
        ov = overrides[route_idx]
        if ov is not None and ov.size:
            waypoints = _apply_override(waypoints, ov)
    knots = schedule_knots(t_start, t_end, waypoints, profile)
    reference = _build_interpolator(route_idx, knots, waypoints)
    if depth >= MAX_AVOID_DEPTH:
        return reference, t_end
    push = _check_clearance(reference, route_idx, t_start, t_end, tuning, obstacles)
    if push is None:
        return reference, t_end
    return build_reference_curve(
        route_idx,
        gate_pos,
        gate_rpy,
        t_start,
        duration,
        tuning,
        profile,
        obstacles=obstacles,
        extra=push,
        overrides=overrides,
        depth=depth + 1,
    )


def load_route_overrides() -> "list[np.ndarray] | None":
    return _load_override_table()
