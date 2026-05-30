"""Reference-tracking to attitude command conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .geometry import body_z_from_quat, euler_from_matrix

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from .cascade_pid import PositionPid

_LATERAL_ACC_LIMIT = 16.0
_FF_SCALE = 0.75
_EPS = 1e-6
# Maximum tilt of the desired thrust axis away from vertical (rad). Generous, so
# the clamp only catches pathological commands while preserving climb authority.
_MAX_TILT = 1.1
# Reference speed (m/s) below which the flight-direction heading is ill-defined
# and we fall back to the previously commanded yaw.
_YAW_SPEED_EPS = 1e-2

# Previously commanded yaw, used as the fallback heading when the horizontal
# reference speed is near zero (mirrors the single-controller usage pattern).
_prev_yaw = 0.0


def tracking_command(
    reference: "Callable[..., NDArray[np.floating]]",
    pos: "NDArray[np.floating]",
    vel: "NDArray[np.floating]",
    quat: "NDArray[np.floating]",
    t_eval: float,
    pid: PositionPid,
    dt: float,
    mass: float,
    gravity: float,
    lateral_acc_limit: float = _LATERAL_ACC_LIMIT,
    feedforward_scale: float = _FF_SCALE,
    max_tilt: float = _MAX_TILT,
) -> "NDArray[np.floating]":
    """Compute the attitude action [roll, pitch, yaw, collective_thrust]."""
    global _prev_yaw
    ref_pos = np.asarray(reference(t_eval), dtype=np.float64)
    ref_vel = np.asarray(reference.derivative(1)(t_eval), dtype=np.float64)
    ref_acc = np.asarray(reference.derivative(2)(t_eval), dtype=np.float64)

    lateral = ref_acc[:2]
    n = float(np.linalg.norm(lateral))
    if n > lateral_acc_limit:
        lateral = lateral * (lateral_acc_limit / n)
        ref_acc = np.array([lateral[0], lateral[1], ref_acc[2]], dtype=np.float64)

    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    pos_error = ref_pos - pos
    vel_error = ref_vel - vel
    pid_force = pid.update(pos_error, vel_error, dt)

    thrust_vec = pid_force + feedforward_scale * mass * ref_acc
    thrust_vec = thrust_vec.astype(np.float64)
    thrust_vec[2] += mass * gravity

    z_body = body_z_from_quat(quat)
    collective_thrust = float(np.dot(thrust_vec, z_body))

    # --- Tilt clamp (safety) --------------------------------------------------
    # Bound the tilt of the desired thrust axis away from vertical to `max_tilt`
    # by limiting the horizontal component of the thrust vector relative to its
    # vertical component, while keeping the vertical component >= mass*gravity so
    # the drone never loses climb authority.
    vert = float(thrust_vec[2])
    vert = max(vert, mass * gravity)
    thrust_vec[2] = vert
    horiz = thrust_vec[:2]
    horiz_norm = float(np.linalg.norm(horiz))
    max_horiz = vert * float(np.tan(max_tilt))
    if horiz_norm > max_horiz:
        horiz = horiz * (max_horiz / (horiz_norm + _EPS))
        thrust_vec[0] = float(horiz[0])
        thrust_vec[1] = float(horiz[1])

    # --- Desired thrust axis --------------------------------------------------
    thrust_norm = float(np.linalg.norm(thrust_vec))
    z_des = thrust_vec / (thrust_norm + _EPS)

    # --- Commanded yaw along the horizontal flight direction ------------------
    # Differential-flatness attitude: point the body x-axis along the reference
    # velocity heading instead of the fixed world-x reference. This keeps yaw
    # tracking travel and yields small, smooth roll/pitch.
    horiz_speed = float(np.hypot(ref_vel[0], ref_vel[1]))
    if horiz_speed > _YAW_SPEED_EPS:
        psi = float(np.arctan2(ref_vel[1], ref_vel[0]))
        _prev_yaw = psi
    else:
        psi = _prev_yaw
    x_c = np.array([np.cos(psi), np.sin(psi), 0.0], dtype=np.float64)

    y_des = np.cross(z_des, x_c)
    y_des = y_des / (np.linalg.norm(y_des) + _EPS)
    x_des = np.cross(y_des, z_des)
    r_des = np.column_stack([x_des, y_des, z_des])
    euler = euler_from_matrix(r_des)
    return np.array(
        [float(euler[0]), float(euler[1]), float(euler[2]), collective_thrust], dtype=np.float32
    )
