"""Nominal track geometry and math helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray

DEFAULT_GATE_POS = np.array(
    [[0.5, 0.25, 0.7], [1.05, 0.75, 1.2], [-1.0, -0.25, 0.7], [0.0, -0.75, 1.2]], dtype=np.float64
)
DEFAULT_GATE_RPY = np.array(
    [[0.0, 0.0, -0.78], [0.0, 0.0, 2.35], [0.0, 0.0, 3.14], [0.0, 0.0, 0.0]], dtype=np.float64
)
DEFAULT_OBSTACLES = np.array(
    [[0.0, 0.75, 1.55], [1.0, 0.25, 1.55], [-1.5, -0.25, 1.55], [-0.5, -0.75, 1.55]],
    dtype=np.float64,
)


def normalize_gate_index(value: NDArray[np.floating] | int) -> int:
    """Normalize a target gate index into a Python int scalar."""
    arr = np.asarray(value)
    if arr.ndim > 1 or arr.size != 1:
        raise ValueError(f"target_gate must be scalar, got shape {arr.shape}")
    return int(arr.reshape(()).item())


def _horizontal_axis(rpy: NDArray[np.floating], axis_idx: int) -> NDArray[np.floating]:
    rot = R.from_euler("xyz", rpy).as_matrix()
    v = rot[:, axis_idx].copy()
    v[2] = 0.0
    n = float(np.linalg.norm(v))
    return v / (n + 1e-9)


def gate_x_axis(rpy: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return the horizontal gate x-axis unit vector for the given orientation."""
    return _horizontal_axis(rpy, 0)


def gate_y_axis(rpy: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return the horizontal gate y-axis unit vector for the given orientation."""
    return _horizontal_axis(rpy, 1)


def gate_axis_points(
    gate_pos: NDArray[np.floating],
    gate_rpy: NDArray[np.floating],
    r_in: float = 0.25,
    r_out: float = 0.3,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return entry and exit waypoints offset along the gate normal axis."""
    x = gate_x_axis(gate_rpy)
    entry = np.asarray(gate_pos, dtype=np.float64) - r_in * x
    exit_pt = np.asarray(gate_pos, dtype=np.float64) + r_out * x
    return entry, exit_pt


def body_z_from_quat(quat: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return the body z-axis direction in world frame from a quaternion."""
    qx, qy, qz, qw = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    return np.array(
        [2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        dtype=np.float64,
    )


def euler_from_matrix(mat: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert a rotation matrix into xyz Euler angles in radians."""
    return R.from_matrix(np.asarray(mat, dtype=np.float64)).as_euler("xyz", degrees=False)
