"""Reference-tracking to attitude command conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .cascade_pid import PositionPid
from .geometry import body_z_from_quat, euler_from_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray

_LATERAL_ACC_LIMIT = 16.0
_FF_SCALE = 0.75
_EPS = 1e-6


def tracking_command(
    reference,
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
) -> "NDArray[np.floating]":
    """Compute the attitude action [roll, pitch, yaw, collective_thrust]."""
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

    thrust_norm = float(np.linalg.norm(thrust_vec))
    z_des = thrust_vec / (thrust_norm + _EPS)
    y_des = np.cross(z_des, np.array([1.0, 0.0, 0.0]))
    y_des = y_des / (np.linalg.norm(y_des) + _EPS)
    x_des = np.cross(y_des, z_des)
    r_des = np.column_stack([x_des, y_des, z_des])
    euler = euler_from_matrix(r_des)
    return np.array(
        [float(euler[0]), float(euler[1]), float(euler[2]), collective_thrust], dtype=np.float32
    )
